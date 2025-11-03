import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, Optional, Dict
import math
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import logging
from datetime import datetime
import wandb
from torch.cuda.amp import autocast, GradScaler

from Minkowski_backbone import (
    get_args_parser,
    voxel_dim,
    points_per_curve,
    points_per_patch_dim,
    num_of_gpus,
    MLP,
    MLP_hn,
)
from data_loader_abc import *

class Config:
    # Model Architecture
    D_MODEL = 384 
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.05

    # Latent Space
    CURVE_LATENT_DIM = 256
    PATCH_LATENT_DIM = 256

    # Data
    PATCH_NUM_POINTS = 400
    POINTS_PER_PATCH_DIM = 20
    MAX_CURVES = 100
    MAX_PATCHES = 50 
    
    # HyperNetwork
    HN_PE_DIM = 64
    HN_MLP_DIM = 128

    # Position Encoding
    PE_TEMPERATURE = 10000
    PE_SCALE = 2 * math.pi

    # Labels
    CURVE_NUM_CLASSES = 4
    PATCH_NUM_CLASSES = 6

    # VAE - 关键修改
    KL_WEIGHT_START = 0.0
    KL_WEIGHT_END = 0.0001  # 从0.1改为0.0001
    KL_WARMUP_EPOCHS = 100  # 从20改为100
    KL_FREE_BITS = 0.5  # Free bits per dimension

    # Training Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    EVAL_INTERVAL = 10  
    SAVE_INTERVAL = 50
    
    # Performance Optimization
    USE_AMP = False
    GRADIENT_ACCUMULATION_STEPS = 2
    LOG_INTERVAL = 200

    # Loss weights
    RECON_WEIGHT = 1.0
    ENDPOINT_WEIGHT = 1.0
    TOPOLOGY_WEIGHT = 0.5
    LABEL_WEIGHT = 0.5
    VALIDITY_WEIGHT = 0.1

    # Experimental options
    USE_TRANSFORMER = False  
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

def setup_logger(log_dir, rank=0):
    """Setup file logger"""
    if rank != 0:
        return None
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger(__name__)


class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, xyz_coords, voxel_dim=64):
        B, N, _ = xyz_coords.shape
        device = xyz_coords.device

        if self.normalize:
            coords = self.scale * xyz_coords / (voxel_dim - 1)
        else:
            coords = xyz_coords

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # coords: [B, N, 3], dim_t: [num_pos_feats]
        # pos: [B, N, 3, num_pos_feats]
        pos = coords[:, :, :, None] / dim_t

        pos_x = pos[:, :, 0, :]  # [B, N, num_pos_feats]
        pos_y = pos[:, :, 1, :]  # [B, N, num_pos_feats]
        pos_z = pos[:, :, 2, :]  # [B, N, num_pos_feats]

        pos_x_sin = pos_x[:, :, 0::2].sin()  # [B, N, num_pos_feats//2]
        pos_x_cos = pos_x[:, :, 1::2].cos()  # [B, N, num_pos_feats//2]
        
        pos_y_sin = pos_y[:, :, 0::2].sin()
        pos_y_cos = pos_y[:, :, 1::2].cos()
        
        pos_z_sin = pos_z[:, :, 0::2].sin()
        pos_z_cos = pos_z[:, :, 1::2].cos()

        pos_x_encoded = torch.stack([pos_x_sin, pos_x_cos], dim=3).flatten(2)  # [B, N, num_pos_feats]
        pos_y_encoded = torch.stack([pos_y_sin, pos_y_cos], dim=3).flatten(2)  # [B, N, num_pos_feats]
        pos_z_encoded = torch.stack([pos_z_sin, pos_z_cos], dim=3).flatten(2)  # [B, N, num_pos_feats]

        return torch.cat([pos_x_encoded, pos_y_encoded, pos_z_encoded], dim=2)  # [B, N, num_pos_feats*3]

class PointNetEncoder(nn.Module):
    """PointNet-based feature extractor"""
    def __init__(self, input_dim=3, output_dim=128, dropout=0.1):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, points):
        B, N, P, C = points.shape
        x = points.view(B * N, P, C)

        point_feat = self.point_mlp(x)
        global_feat, _ = torch.max(point_feat, dim=1)
        features = self.global_mlp(global_feat).view(B, N, -1)

        return features

class CurveEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        self.geom_encoder = PointNetEncoder(
            input_dim=3, output_dim=config.D_MODEL // 2, dropout=config.DROPOUT
        )

        self.endpoint_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, config.D_MODEL // 4),
        )

        self.closed_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, config.D_MODEL // 8),
        )

        self.label_embedding = nn.Embedding(
            config.CURVE_NUM_CLASSES, config.D_MODEL // 8
        )

        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)

        if config.USE_TRANSFORMER:
            self.pos_encoder = PositionEmbeddingSine3D(
                num_pos_feats=config.D_MODEL // 3,
                temperature=config.PE_TEMPERATURE,
                normalize=True,
                scale=config.PE_SCALE,
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                dim_feedforward=config.DIM_FEEDFORWARD,
                dropout=config.DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=config.NUM_LAYERS
            )

        self.to_mean = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)
        self.to_logvar = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)

    def forward(self, curve_points, endpoints, is_closed, labels, mask):
        B, N = curve_points.shape[:2]
        device = curve_points.device

        endpoints_flat = endpoints.reshape(B, N, 6)

        geom_feat = self.geom_encoder(curve_points)
        endpoint_feat = self.endpoint_encoder(endpoints_flat)
        closed_feat = self.closed_encoder(is_closed.unsqueeze(-1).float())
        label_feat = self.label_embedding(labels)

        all_feat = torch.cat([geom_feat, endpoint_feat, closed_feat, label_feat], dim=-1)
        tokens = self.fusion(all_feat)

        if self.config.USE_TRANSFORMER:
            centroids = curve_points.mean(dim=2)
            pos_enc = self.pos_encoder(centroids, voxel_dim=voxel_dim)
            tokens = tokens + pos_enc

            attn_mask = ~mask if mask is not None else None
            contextualized = self.transformer(tokens, src_key_padding_mask=attn_mask)
            contextualized = contextualized * mask.unsqueeze(-1).float()
        else:
            contextualized = tokens
            contextualized = contextualized * mask.unsqueeze(-1).float()

        mean = self.to_mean(contextualized)
        logvar = self.to_logvar(contextualized)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mean, logvar

class CurveDecoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        self.start_point_embed = MLP(
            config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 3, 3
        )

        self.curve_pe = self._init_curve_pe(points_per_curve, config.HN_PE_DIM)

        self.curve_shape_embed = MLP_hn(
            input_dim=config.HN_PE_DIM,
            hidden_dim=config.HN_MLP_DIM,
            output_dim=3,
            num_layers=3,
            input_dim_fea=config.CURVE_LATENT_DIM,
        )

        self.endpoints_head = MLP(
            config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 6, 3
        )
        self.closed_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 1, 3)
        self.label_head = MLP(
            config.CURVE_LATENT_DIM,
            config.CURVE_LATENT_DIM,
            config.CURVE_NUM_CLASSES,
            3,
        )
        self.validity_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 1, 3)

    def _init_curve_pe(self, num_points, pe_dim):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        coord = torch.arange(num_points, dtype=torch.float32, device=device) / (
            num_points - 1
        )
        coord = coord.view(-1, 1)

        exp = torch.arange(pe_dim // 2, dtype=torch.float32, device=device)
        base = 2.0 * torch.ones([pe_dim // 2], dtype=torch.float32, device=device)
        coeff = 2 * math.pi * torch.pow(base, exp).view(1, -1)
        mat = torch.mm(coord, coeff)

        sin_mat = torch.sin(mat)
        cos_mat = torch.cos(mat)
        pe = torch.cat([sin_mat, cos_mat], dim=1)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, z):
        B, N = z.shape[:2]

        start_point = self.start_point_embed(z).tanh() * 0.5
        curve_pe = self.curve_pe.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        shape_offset = self.curve_shape_embed(curve_pe.unsqueeze(0), z.unsqueeze(0))
        shape_offset = shape_offset.squeeze(0)
        points = start_point.unsqueeze(2) + shape_offset

        endpoints = self.endpoints_head(z).view(B, N, 2, 3)
        closed_logits = self.closed_head(z).squeeze(-1)
        label_logits = self.label_head(z)
        validity_logits = self.validity_head(z).squeeze(-1)

        return {
            "points": points, #[B N 34 3]
            "endpoints": endpoints, #[B N 2 3]
            "closed_logits": closed_logits, #[B N]
            "label_logits": label_logits,  #[B N]
            "validity_logits": validity_logits, #[B N]
        }

class PatchEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        self.geom_encoder = PointNetEncoder(
            input_dim=6, output_dim=config.D_MODEL // 3 * 2, dropout=config.DROPOUT
        )

        self.topo_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, config.D_MODEL // 6),
        )

        self.label_embedding = nn.Embedding(
            config.PATCH_NUM_CLASSES, config.D_MODEL // 6
        )

        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)

        if config.USE_TRANSFORMER:
            self.pos_encoder = PositionEmbeddingSine3D(
                num_pos_feats=config.D_MODEL // 3,
                temperature=config.PE_TEMPERATURE,
                normalize=True,
                scale=config.PE_SCALE,
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                dim_feedforward=config.DIM_FEEDFORWARD,
                dropout=config.DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=config.NUM_LAYERS
            )

        self.to_mean = nn.Linear(config.D_MODEL, config.PATCH_LATENT_DIM)
        self.to_logvar = nn.Linear(config.D_MODEL, config.PATCH_LATENT_DIM)

    def forward(self, patch_points, patch_normals, u_closed, v_closed, labels, mask):
        B, N = patch_points.shape[:2]

        geom_input = torch.cat([patch_points, patch_normals], dim=-1)
        geom_feat = self.geom_encoder(geom_input)

        topo_input = torch.stack([u_closed, v_closed], dim=-1).float()
        topo_feat = self.topo_encoder(topo_input)

        label_feat = self.label_embedding(labels)

        all_feat = torch.cat([geom_feat, topo_feat, label_feat], dim=-1)
        tokens = self.fusion(all_feat)

        if self.config.USE_TRANSFORMER:
            centroids = patch_points.mean(dim=2)
            pos_enc = self.pos_encoder(centroids, voxel_dim=voxel_dim)
            tokens = tokens + pos_enc

            attn_mask = ~mask if mask is not None else None
            contextualized = self.transformer(tokens, src_key_padding_mask=attn_mask)
            contextualized = contextualized * mask.unsqueeze(-1).float()
        else:
            contextualized = tokens
            contextualized = contextualized * mask.unsqueeze(-1).float()

        mean = self.to_mean(contextualized)
        logvar = self.to_logvar(contextualized)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mean, logvar

class PatchDecoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.patch_dim = int(math.sqrt(config.PATCH_NUM_POINTS))

        self.startpoint_embed = MLP(
            config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 3, 3
        )

        self.patch_pe_u, self.patch_pe_v = self._init_patch_pe(
            self.patch_dim, config.HN_PE_DIM
        )

        self.patch_shape_embed = MLP_hn(
            input_dim=config.HN_PE_DIM * 2,
            hidden_dim=config.HN_MLP_DIM,
            output_dim=6,
            num_layers=3,
            input_dim_fea=config.PATCH_LATENT_DIM,
        )

        self.u_closed_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 1, 3)
        self.v_closed_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 1, 3)
        self.label_head = MLP(
            config.PATCH_LATENT_DIM,
            config.PATCH_LATENT_DIM,
            config.PATCH_NUM_CLASSES,
            3,
        )
        self.validity_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 1, 3)

    def _init_patch_pe(self, dim, pe_dim):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        coord = torch.arange(dim, dtype=torch.float32, device=device) / (dim - 1)
        coord = coord.view(-1, 1)

        exp = torch.arange(pe_dim // 2, dtype=torch.float32, device=device)
        base = 2.0 * torch.ones([pe_dim // 2], dtype=torch.float32, device=device)
        coeff = 2 * math.pi * torch.pow(base, exp).view(1, -1)
        mat = torch.mm(coord, coeff)

        sin_mat = torch.sin(mat)
        cos_mat = torch.cos(mat)
        pe = torch.cat([sin_mat, cos_mat], dim=1)

        pe_u = nn.Parameter(pe, requires_grad=False)
        pe_v = nn.Parameter(pe, requires_grad=False)

        return pe_u, pe_v

    def forward(self, z):
        B, N = z.shape[:2]

        startpoint = self.startpoint_embed(z).tanh() * 0.5

        patch_pe = []
        for i in range(self.patch_dim):
            for j in range(self.patch_dim):
                pe_ij = torch.cat([self.patch_pe_u[i], self.patch_pe_v[j]], dim=0)
                patch_pe.append(pe_ij)
        patch_pe = torch.stack(patch_pe, dim=0)
        patch_pe = patch_pe.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

        output = self.patch_shape_embed(patch_pe.unsqueeze(0), z.unsqueeze(0))
        output = output.squeeze(0)

        shape_offset = output[..., :3]
        normals = output[..., 3:]

        points = startpoint.unsqueeze(2) + shape_offset
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        u_closed_logits = self.u_closed_head(z).squeeze(-1)
        v_closed_logits = self.v_closed_head(z).squeeze(-1)
        label_logits = self.label_head(z)
        validity_logits = self.validity_head(z).squeeze(-1)

        return {
            "points": points,
            "normals": normals,
            "u_closed_logits": u_closed_logits,
            "v_closed_logits": v_closed_logits,
            "label_logits": label_logits,
            "validity_logits": validity_logits,
        }

class KLScheduler:
    def __init__(self, start_weight=0.0, end_weight=0.0001, warmup_epochs=100):
        self.start = start_weight
        self.end = end_weight
        self.warmup = warmup_epochs

    def get_weight(self, epoch):
        if epoch >= self.warmup:
            return self.end
        
        progress = math.sqrt(epoch / self.warmup)
        return self.start + (self.end - self.start) * progress

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """Focal Loss for class imbalance"""
    if logits.dim() == 2:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    else:
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        focal_loss = focal_loss.view(targets.shape)

    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    return focal_loss

def compute_patch_vae_loss(
    pred, target, mean, logvar, mask, kl_weight, config, weighting=None
):
    """Compute patch VAE losses"""
    losses = {}

    kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    
    free_bits = torch.tensor(config.KL_FREE_BITS, device=kl_per_dim.device)
    kl_per_dim = torch.max(kl_per_dim, free_bits)
    
    kl = kl_per_dim.sum(dim=-1)  # [B, N]
    kl = (kl * mask.float()).sum() / mask.sum().clamp(min=1)
    
    kl = kl / config.PATCH_LATENT_DIM
    losses["kl"] = kl

    if weighting is not None:
        valid_weighting = weighting * mask.float()
        valid_weighting = (
            valid_weighting / (valid_weighting.sum() + 1e-8) * mask.sum().float()
        )
        weight_mask = valid_weighting.unsqueeze(-1).unsqueeze(-1)
    else:
        weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()

    recon_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points

    recon_normals = F.mse_loss(pred["normals"], target["normals"], reduction="none")
    recon_normals = (recon_normals * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_normals"] = recon_normals

    topo_weight = (
        weighting if weighting is not None else torch.ones_like(mask, dtype=torch.float)
    )
    topo_weight = topo_weight * mask.float()

    u_closed_loss = F.binary_cross_entropy_with_logits(
        pred["u_closed_logits"], target["u_closed"].float(), reduction="none"
    )
    u_closed_loss = (u_closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)

    v_closed_loss = F.binary_cross_entropy_with_logits(
        pred["v_closed_logits"], target["v_closed"].float(), reduction="none"
    )
    v_closed_loss = (v_closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)

    losses["u_closed"] = u_closed_loss
    losses["v_closed"] = v_closed_loss

    label_loss = F.cross_entropy(
        pred["label_logits"].view(-1, pred["label_logits"].size(-1)),
        target["labels"].view(-1),
        reduction="none",
    )
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss

    validity_loss = F.binary_cross_entropy_with_logits(
        pred["validity_logits"], mask.float(), reduction="mean"
    )
    losses["validity"] = validity_loss

    total_loss = (
        kl_weight * kl
        + config.RECON_WEIGHT * (recon_points + recon_normals)
        + config.TOPOLOGY_WEIGHT * (u_closed_loss + v_closed_loss)
        + config.LABEL_WEIGHT * label_loss
        + config.VALIDITY_WEIGHT * validity_loss
    )
    losses["total"] = total_loss

    return losses

def compute_curve_vae_loss(
    pred, target, mean, logvar, mask, kl_weight, config, weighting=None
):
    """Compute curve VAE losses"""
    losses = {}

    kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    
    free_bits = torch.tensor(config.KL_FREE_BITS, device=kl_per_dim.device)
    kl_per_dim = torch.max(kl_per_dim, free_bits)
    
    kl = kl_per_dim.sum(dim=-1)  # [B, N]
    kl = (kl * mask.float()).sum() / mask.sum().clamp(min=1)
    
    kl = kl / config.CURVE_LATENT_DIM
    losses["kl"] = kl

    if weighting is not None:
        valid_weighting = weighting * mask.float()
        valid_weighting = (
            valid_weighting / (valid_weighting.sum() + 1e-8) * mask.sum().float()
        )
        weight_mask = valid_weighting.unsqueeze(-1).unsqueeze(-1)
    else:
        weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()

    recon_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points

    open_mask = ~target["is_closed"]
    endpoint_mask = (mask & open_mask).unsqueeze(-1).unsqueeze(-1).float()
    
    recon_endpoints = F.mse_loss(
        pred["endpoints"], target["endpoints"], reduction="none"
    )
    
    if weighting is not None:
        endpoint_weighting = (weighting * mask.float() * open_mask.float()).unsqueeze(-1).unsqueeze(-1)
        endpoint_weighting = endpoint_weighting / (endpoint_weighting.sum() + 1e-8) * open_mask.sum().float().clamp(min=1)
        recon_endpoints = (recon_endpoints * endpoint_weighting).sum() / open_mask.sum().clamp(min=1)
    else:
        recon_endpoints = (recon_endpoints * endpoint_mask).sum() / open_mask.sum().clamp(min=1)
    
    losses["recon_endpoints"] = recon_endpoints

    topo_weight = (
        weighting if weighting is not None else torch.ones_like(mask, dtype=torch.float)
    )
    topo_weight = topo_weight * mask.float()

    closed_loss = F.binary_cross_entropy_with_logits(
        pred["closed_logits"], target["is_closed"].float(), reduction="none"
    )
    closed_loss = (closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)

    losses["closed"] = closed_loss

    label_loss = F.cross_entropy(
        pred["label_logits"].view(-1, pred["label_logits"].size(-1)),
        target["labels"].view(-1),
        reduction="none",
    )
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss

    validity_loss = F.binary_cross_entropy_with_logits(
        pred["validity_logits"], mask.float(), reduction="mean"
    )
    losses["validity"] = validity_loss

    total_loss = (
        kl_weight * kl
        + config.RECON_WEIGHT * (recon_points + recon_endpoints)
        + config.TOPOLOGY_WEIGHT * closed_loss
        + config.LABEL_WEIGHT * label_loss
        + config.VALIDITY_WEIGHT * validity_loss
    )
    losses["total"] = total_loss

    return losses

def process_batch_data(data_item, config, device):
    """Process batch data from dataloader - 优化版本"""
    corner_points = data_item[0]
    corner_batch_idx = data_item[1]
    batch_sample_id = data_item[2]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]

    processed_curves = None
    processed_patches = None

    # Process Curves
    if len(target_curves_list) > 0:
        batch_size = len(target_curves_list)
        max_n_curves = max([c["curve_points"].shape[0] for c in target_curves_list])
        max_n_curves = min(max_n_curves, config.MAX_CURVES)

        # 直接在GPU上创建张量
        curve_points_batch = torch.zeros(batch_size, max_n_curves, 34, 3, device=device, dtype=torch.float32)
        endpoints_batch = torch.zeros(batch_size, max_n_curves, 2, 3, device=device, dtype=torch.float32)
        is_closed_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool, device=device)
        labels_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.long, device=device)
        mask_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool, device=device)
        weighting_batch = torch.ones(batch_size, max_n_curves, dtype=torch.float32, device=device)

        for i, curve_dict in enumerate(target_curves_list):
            n_curves = min(curve_dict["curve_points"].shape[0], max_n_curves)
            
            if n_curves > 0:
                curve_points_batch[i, :n_curves].copy_(curve_dict["curve_points"][:n_curves], non_blocking=True)
                is_closed_batch[i, :n_curves].copy_(curve_dict["is_closed"][:n_curves], non_blocking=True)
                labels_batch[i, :n_curves].copy_(curve_dict["labels"][:n_curves], non_blocking=True)
                mask_batch[i, :n_curves] = True

                endpoint_indices = curve_dict["endpoints"][:n_curves].long()
                curve_points = curve_dict["curve_points"][:n_curves]
                is_closed = curve_dict["is_closed"][:n_curves]
                
                open_mask = ~is_closed
                if open_mask.any():
                    open_indices = torch.where(open_mask)[0]
                    for j in open_indices:
                        j_val = j.item()
                        idx0, idx1 = endpoint_indices[j_val]
                        idx0 = max(0, min(33, idx0.item()))
                        idx1 = max(0, min(33, idx1.item()))
                        endpoints_batch[i, j_val, 0].copy_(curve_points[j_val, idx0], non_blocking=True)
                        endpoints_batch[i, j_val, 1].copy_(curve_points[j_val, idx1], non_blocking=True)

                if "curve_length_weighting" in curve_dict:
                    weighting_batch[i, :n_curves].copy_(curve_dict["curve_length_weighting"][:n_curves], non_blocking=True)

        processed_curves = {
            "curve_points": curve_points_batch,
            "endpoints": endpoints_batch,
            "is_closed": is_closed_batch,
            "labels": labels_batch,
            "mask": mask_batch,
            "weighting": weighting_batch,
        }

    # Process Patches
    if len(target_patches_list) > 0:
        batch_size = len(target_patches_list)
        max_n_patches = max([len(p["patch_points"]) for p in target_patches_list])
        max_n_patches = min(max_n_patches, config.MAX_PATCHES)

        patch_points_batch = torch.zeros(batch_size, max_n_patches, 400, 3, device=device, dtype=torch.float32)
        patch_normals_batch = torch.zeros(batch_size, max_n_patches, 400, 3, device=device, dtype=torch.float32)
        u_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool, device=device)
        v_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool, device=device)
        labels_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.long, device=device)
        mask_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool, device=device)
        weighting_batch = torch.ones(batch_size, max_n_patches, dtype=torch.float32, device=device)

        for i, patch_dict in enumerate(target_patches_list):
            n_patches = min(len(patch_dict["patch_points"]), max_n_patches)

            if n_patches > 0:
                for j in range(n_patches):
                    patch_points_batch[i, j].copy_(patch_dict["patch_points"][j], non_blocking=True)
                    patch_normals_batch[i, j].copy_(patch_dict["patch_normals"][j], non_blocking=True)

                u_closed_batch[i, :n_patches].copy_(patch_dict["u_closed"][:n_patches], non_blocking=True)
                v_closed_batch[i, :n_patches].copy_(patch_dict["v_closed"][:n_patches], non_blocking=True)
                labels_batch[i, :n_patches].copy_(patch_dict["labels"][:n_patches], non_blocking=True)
                mask_batch[i, :n_patches] = True

                if "patch_area_weighting" in patch_dict:
                    weighting_batch[i, :n_patches].copy_(patch_dict["patch_area_weighting"][:n_patches], non_blocking=True)

        processed_patches = {
            "patch_points": patch_points_batch,
            "patch_normals": patch_normals_batch,
            "u_closed": u_closed_batch,
            "v_closed": v_closed_batch,
            "labels": labels_batch,
            "mask": mask_batch,
            "weighting": weighting_batch,
        }

    return processed_curves, processed_patches

class LossAccumulator:
    """优化版本 - 减少CPU-GPU同步"""
    def __init__(self):
        self.losses = {}
        self.count = 0
        
    def add(self, loss_dict):
        for key, val in loss_dict.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(val.detach())
        self.count += 1
    
    def get_averages(self):
        if self.count == 0:
            return {}
        averages = {}
        for key, vals in self.losses.items():
            if vals:
                stacked = torch.stack(vals)
                averages[key] = stacked.mean().item()
        return averages
    
    def reset(self):
        self.losses = {}
        self.count = 0

def compute_patch_metrics(pred, target, mask):
    """Compute patch evaluation metrics"""
    metrics = {}

    recon_error_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_error_points = (
        recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()

    recon_error_normals = F.mse_loss(
        pred["normals"], target["normals"], reduction="none"
    )
    recon_error_normals = (
        recon_error_normals * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error_normals"] = recon_error_normals.item()

    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    label_accuracy = correct_labels.sum().float() / mask.sum().float()
    metrics["label_accuracy"] = label_accuracy.item()

    pred_u_closed = torch.sigmoid(pred["u_closed_logits"]) > 0.5
    correct_u = (pred_u_closed == target["u_closed"]) & mask
    u_closed_accuracy = correct_u.sum().float() / mask.sum().float()
    metrics["u_closed_accuracy"] = u_closed_accuracy.item()

    pred_v_closed = torch.sigmoid(pred["v_closed_logits"]) > 0.5
    correct_v = (pred_v_closed == target["v_closed"]) & mask
    v_closed_accuracy = correct_v.sum().float() / mask.sum().float()
    metrics["v_closed_accuracy"] = v_closed_accuracy.item()

    pred_validity = torch.sigmoid(pred["validity_logits"]) > 0.5
    validity_correct = (pred_validity == mask).sum().float() / pred_validity.numel()
    metrics["validity_accuracy"] = validity_correct.item()

    topology_correct = correct_u & correct_v
    topology_accuracy = topology_correct.sum().float() / mask.sum().float()
    metrics["topology_accuracy"] = topology_accuracy.item()

    return metrics

def compute_curve_metrics(pred, target, mask):
    """Compute curve evaluation metrics"""
    metrics = {}

    recon_error_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_error_points = (
        recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()

    recon_error_endpoints = F.mse_loss(
        pred["endpoints"], target["endpoints"], reduction="none"
    )
    recon_error_endpoints = (
        recon_error_endpoints * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error_endpoints"] = recon_error_endpoints.item()

    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    label_accuracy = correct_labels.sum().float() / mask.sum().float()
    metrics["label_accuracy"] = label_accuracy.item()

    pred_closed = torch.sigmoid(pred["closed_logits"]) > 0.5
    correct_closed = (pred_closed == target["is_closed"]) & mask
    closed_accuracy = correct_closed.sum().float() / mask.sum().float()
    metrics["closed_accuracy"] = closed_accuracy.item()

    pred_validity = torch.sigmoid(pred["validity_logits"]) > 0.5
    validity_correct = (pred_validity == mask).sum().float() / pred_validity.numel()
    metrics["validity_accuracy"] = validity_correct.item()

    start_point_error = F.mse_loss(
        pred["endpoints"][:, :, 0], target["endpoints"][:, :, 0], reduction="none"
    )
    start_point_error = (
        start_point_error * mask.unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["start_point_error"] = start_point_error.item()

    end_point_error = F.mse_loss(
        pred["endpoints"][:, :, 1], target["endpoints"][:, :, 1], reduction="none"
    )
    end_point_error = (end_point_error * mask.unsqueeze(-1)).sum() / mask.sum().clamp(
        min=1
    )
    metrics["end_point_error"] = end_point_error.item()

    return metrics

def train_pipeline(rank, num_gpus, args, config):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23257",
        world_size=num_gpus,
        rank=rank,
    )
    
    if num_gpus > 1:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup experiment directory
    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "ckpt")

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        logger = setup_logger(experiment_dir)
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Training on {num_gpus} GPU(s)")
        logger.info(f"Mixed Precision: {config.USE_AMP}")
        logger.info(f"Gradient Accumulation Steps: {config.GRADIENT_ACCUMULATION_STEPS}")
        
        wandb.init(
            project="vae-cad-reconstruction",
            name=args.experiment_name,
            config={
                "d_model": config.D_MODEL,
                "curve_latent_dim": config.CURVE_LATENT_DIM,
                "patch_latent_dim": config.PATCH_LATENT_DIM,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": args.batch_size,
                "use_amp": config.USE_AMP,
            }
        )

    # Load data with optimization
    if args.quicktest:
        train_data, distribute_sampler = train_data_loader(
            args.batch_size,
            voxel_dim=voxel_dim,
            data_folder="data/train_small",
            feature_type=args.input_feature_type,
            pad1s=not args.backbone_feature_encode,
            rotation_augmentation=args.rotation_augment,
            random_angle=args.random_angle,
            with_normal=args.input_normal_signals,
            flag_quick_test=args.quicktest,
            flag_noise=args.noise,
            flag_grid=args.patch_grid,
            num_angle=args.num_angles,
            flag_patch_uv=args.patch_uv,
            dim_grid=points_per_patch_dim,
            eval_res_cov=args.extra_single_chamfer,
        )
        val_data, val_data_sampler = train_data_loader(
            args.batch_size,
            voxel_dim=voxel_dim,
            data_folder="data/train_small",
            feature_type=args.input_feature_type,
            pad1s=not args.backbone_feature_encode,
            rotation_augmentation=args.rotation_augment,
            with_normal=args.input_normal_signals,
            flag_quick_test=False,
            flag_noise=args.noise,
            flag_grid=args.patch_grid,
            num_angle=args.num_angles,
            flag_patch_uv=args.patch_uv,
            dim_grid=points_per_patch_dim,
            eval_res_cov=args.extra_single_chamfer,
        )
    else:
        if args.parsenet:
            train_folder = (
                "data/partial/train" if args.partial else "data/default/train"
            )
            train_data, distribute_sampler = train_data_loader(
                args.batch_size,
                voxel_dim=voxel_dim,
                data_folder=train_folder,
                feature_type=args.input_feature_type,
                pad1s=not args.backbone_feature_encode,
                rotation_augmentation=args.rotation_augment,
                random_angle=args.random_angle,
                with_normal=args.input_normal_signals,
                flag_quick_test=args.quicktest,
                flag_noise=args.noise,
                flag_grid=args.patch_grid,
                num_angle=args.num_angles,
                flag_patch_uv=args.patch_uv,
                dim_grid=points_per_patch_dim,
                eval_res_cov=args.extra_single_chamfer,
            )

        if not args.patch_grid:
            val_folder = "val_new_64"
        else:
            val_folder = "data/partial/val" if args.partial else "data/default/val"

        val_data, val_data_sampler = train_data_loader(
            args.batch_size,
            voxel_dim=voxel_dim,
            data_folder=val_folder,
            feature_type=args.input_feature_type,
            pad1s=not args.backbone_feature_encode,
            rotation_augmentation=args.rotation_augment,
            with_normal=args.input_normal_signals,
            flag_quick_test=False,
            flag_noise=args.noise,
            flag_grid=args.patch_grid,
            num_angle=args.num_angles,
            flag_patch_uv=args.patch_uv,
            dim_grid=points_per_patch_dim,
            eval_res_cov=args.extra_single_chamfer,
        )

    # Initialize models
    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)

    if num_gpus > 1:
        patch_encoder = nn.parallel.DistributedDataParallel(
            patch_encoder, device_ids=[rank]
        )
        patch_decoder = nn.parallel.DistributedDataParallel(
            patch_decoder, device_ids=[rank]
        )
        curve_encoder = nn.parallel.DistributedDataParallel(
            curve_encoder, device_ids=[rank]
        )
        curve_decoder = nn.parallel.DistributedDataParallel(
            curve_decoder, device_ids=[rank]
        )

    # Initialize optimizers
    patch_params = list(patch_encoder.parameters()) + list(patch_decoder.parameters())
    curve_params = list(curve_encoder.parameters()) + list(curve_decoder.parameters())

    patch_optimizer = torch.optim.AdamW(
        patch_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    curve_optimizer = torch.optim.AdamW(
        curve_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Initialize mixed precision scalers
    patch_scaler = GradScaler() if config.USE_AMP else None
    curve_scaler = GradScaler() if config.USE_AMP else None

    kl_scheduler = KLScheduler(
        config.KL_WEIGHT_START, config.KL_WEIGHT_END, config.KL_WARMUP_EPOCHS
    )

    # Load checkpoint if specified
    start_epoch = 0
    if (
        hasattr(args, "checkpoint_path")
        and args.checkpoint_path
        and os.path.exists(args.checkpoint_path)
    ):
        if rank == 0:
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        patch_encoder.load_state_dict(checkpoint["patch_encoder"])
        patch_decoder.load_state_dict(checkpoint["patch_decoder"])
        curve_encoder.load_state_dict(checkpoint["curve_encoder"])
        curve_decoder.load_state_dict(checkpoint["curve_decoder"])
        patch_optimizer.load_state_dict(checkpoint["patch_optimizer"])
        curve_optimizer.load_state_dict(checkpoint["curve_optimizer"])
        if config.USE_AMP and "patch_scaler" in checkpoint:
            patch_scaler.load_state_dict(checkpoint["patch_scaler"])
            curve_scaler.load_state_dict(checkpoint["curve_scaler"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    best_val_loss = float("inf")
    cur_epochs = start_epoch
    global_step = start_epoch * len(train_data)

    for epoch in range(start_epoch, args.max_training_iterations):
        # 延迟KL策略：前10000步不用KL
        if global_step < 10000:
            current_kl_weight = 0.0
        else:
            epoch_for_scheduler = (global_step - 10000) // len(train_data)
            current_kl_weight = kl_scheduler.get_weight(epoch_for_scheduler)

        if rank == 0 and epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.max_training_iterations} - KL Weight: {current_kl_weight:.6f}")

        if distribute_sampler is not None:
            distribute_sampler.set_epoch(cur_epochs)

        patch_encoder.train()
        patch_decoder.train()
        curve_encoder.train()
        curve_decoder.train()

        patch_loss_acc = LossAccumulator()
        curve_loss_acc = LossAccumulator()

        data_loader_iterator = iter(train_data)

        if rank == 0:
            pbar = tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}", leave=True, dynamic_ncols=True)
        else:
            pbar = range(len(train_data))

        for batch_idx in pbar:
            try:
                data_item = next(data_loader_iterator)
            except StopIteration:
                data_loader_iterator = iter(train_data)
                data_item = next(data_loader_iterator)
                cur_epochs += 1
                if distribute_sampler is not None:
                    distribute_sampler.set_epoch(cur_epochs)

            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            # Patch VAE training with mixed precision
            if processed_patches is not None:
                with autocast() if config.USE_AMP else torch.cuda.amp.autocast(enabled=False):
                    mean_p, logvar_p = patch_encoder(
                        processed_patches["patch_points"],
                        processed_patches["patch_normals"],
                        processed_patches["u_closed"],
                        processed_patches["v_closed"],
                        processed_patches["labels"],
                        processed_patches["mask"],
                    )

                    std_p = torch.exp(0.5 * logvar_p)
                    eps_p = torch.randn_like(std_p)
                    z_p = mean_p + eps_p * std_p

                    pred_patch = patch_decoder(z_p)

                    target_patch = {
                        "points": processed_patches["patch_points"],
                        "normals": processed_patches["patch_normals"],
                        "u_closed": processed_patches["u_closed"],
                        "v_closed": processed_patches["v_closed"],
                        "labels": processed_patches["labels"],
                    }

                    patch_losses = compute_patch_vae_loss(
                        pred_patch,
                        target_patch,
                        mean_p,
                        logvar_p,
                        processed_patches["mask"],
                        current_kl_weight,
                        config,
                        weighting=processed_patches["weighting"],
                    )

                loss = patch_losses["total"] / config.GRADIENT_ACCUMULATION_STEPS

                if config.USE_AMP:
                    patch_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    if config.USE_AMP:
                        patch_scaler.unscale_(patch_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            patch_params, max_norm=config.GRAD_CLIP_NORM
                        )
                        patch_scaler.step(patch_optimizer)
                        patch_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            patch_params, max_norm=config.GRAD_CLIP_NORM
                        )
                        patch_optimizer.step()
                    
                    patch_optimizer.zero_grad()

                patch_loss_acc.add(patch_losses)

            # Curve VAE training with mixed precision
            if processed_curves is not None:
                with autocast() if config.USE_AMP else torch.cuda.amp.autocast(enabled=False):
                    mean_c, logvar_c = curve_encoder(
                        processed_curves["curve_points"],
                        processed_curves["endpoints"],
                        processed_curves["is_closed"],
                        processed_curves["labels"],
                        processed_curves["mask"],
                    )

                    std_c = torch.exp(0.5 * logvar_c)
                    eps_c = torch.randn_like(std_c)
                    z_c = mean_c + eps_c * std_c

                    pred_curve = curve_decoder(z_c)

                    target_curve = {
                        "points": processed_curves["curve_points"],
                        "endpoints": processed_curves["endpoints"],
                        "is_closed": processed_curves["is_closed"],
                        "labels": processed_curves["labels"],
                    }

                    curve_losses = compute_curve_vae_loss(
                        pred_curve,
                        target_curve,
                        mean_c,
                        logvar_c,
                        processed_curves["mask"],
                        current_kl_weight,
                        config,
                        weighting=processed_curves["weighting"],
                    )

                loss = curve_losses["total"] / config.GRADIENT_ACCUMULATION_STEPS

                if config.USE_AMP:
                    curve_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    if config.USE_AMP:
                        curve_scaler.unscale_(curve_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            curve_params, max_norm=config.GRAD_CLIP_NORM
                        )
                        curve_scaler.step(curve_optimizer)
                        curve_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            curve_params, max_norm=config.GRAD_CLIP_NORM
                        )
                        curve_optimizer.step()
                    
                    curve_optimizer.zero_grad()

                curve_loss_acc.add(curve_losses)

            global_step += 1

            if rank == 0 and (batch_idx + 1) % config.LOG_INTERVAL == 0:
                patch_avgs = patch_loss_acc.get_averages()
                curve_avgs = curve_loss_acc.get_averages()
                postfix = {}
                if "total" in patch_avgs:
                    postfix["P_tot"] = f"{patch_avgs['total']:.3f}"
                    postfix["P_rec"] = f"{patch_avgs.get('recon_points', 0):.3f}"
                if "total" in curve_avgs:
                    postfix["C_tot"] = f"{curve_avgs['total']:.3f}"
                    postfix["C_rec"] = f"{curve_avgs.get('recon_points', 0):.3f}"
                if postfix and isinstance(pbar, tqdm):
                    pbar.set_postfix(postfix)

        cur_epochs += 1

        # epoch级别的wandb logging（参考第四份代码的8个图）
        avg_patch_losses = patch_loss_acc.get_averages()
        avg_curve_losses = curve_loss_acc.get_averages()

        if rank == 0:
            log_dict = {
                "epoch": epoch + 1,
                "kl_weight": current_kl_weight,
            }
            
            if avg_patch_losses:
                log_dict["train/patch_total"] = avg_patch_losses.get('total', 0)
                log_dict["train/patch_kl"] = avg_patch_losses.get('kl', 0)
                log_dict["train/patch_recon_points"] = avg_patch_losses.get('recon_points', 0)
                log_dict["train/patch_topology"] = (avg_patch_losses.get('u_closed', 0) + avg_patch_losses.get('v_closed', 0)) / 2
            
            if avg_curve_losses:
                log_dict["train/curve_total"] = avg_curve_losses.get('total', 0)
                log_dict["train/curve_kl"] = avg_curve_losses.get('kl', 0)
                log_dict["train/curve_recon_points"] = avg_curve_losses.get('recon_points', 0)
                log_dict["train/curve_topology"] = avg_curve_losses.get('closed', 0)
            
            wandb.log(log_dict)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1} - Patch: {avg_patch_losses.get('total', 0):.4f}, Curve: {avg_curve_losses.get('total', 0):.4f}")

        # Validation
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            val_metrics = val_pipeline(
                patch_encoder,
                patch_decoder,
                curve_encoder,
                curve_decoder,
                val_data,
                val_data_sampler, 
                epoch,  
                device,
                config,
                rank,
            )

            if rank == 0:
                val_log_dict = {"epoch": epoch + 1}
                
                if "patch" in val_metrics and val_metrics["patch"]:
                    val_log_dict["val/patch_recon_error"] = val_metrics['patch']['recon_error']
                    logger.info(f"Val Patch Recon Error: {val_metrics['patch']['recon_error']:.6f}")
                
                if "curve" in val_metrics and val_metrics["curve"]:
                    val_log_dict["val/curve_recon_error"] = val_metrics['curve']['recon_error']
                    logger.info(f"Val Curve Recon Error: {val_metrics['curve']['recon_error']:.6f}")
                
                wandb.log(val_log_dict)

                patch_loss = val_metrics.get("patch", {}).get("recon_error", 0)
                curve_loss = val_metrics.get("curve", {}).get("recon_error", 0)
                val_loss = patch_loss + curve_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_dict = {
                        "epoch": epoch,
                        "patch_encoder": patch_encoder.state_dict(),
                        "patch_decoder": patch_decoder.state_dict(),
                        "curve_encoder": curve_encoder.state_dict(),
                        "curve_decoder": curve_decoder.state_dict(),
                        "patch_optimizer": patch_optimizer.state_dict(),
                        "curve_optimizer": curve_optimizer.state_dict(),
                        "val_loss": val_loss,
                    }
                    if config.USE_AMP:
                        checkpoint_dict["patch_scaler"] = patch_scaler.state_dict()
                        checkpoint_dict["curve_scaler"] = curve_scaler.state_dict()
                    
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_dir, "best_model.pth"),
                    )
                    logger.info(f"Saved best model with val_loss: {val_loss:.6f}")

        # Save checkpoint
        if rank == 0 and (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "patch_encoder": patch_encoder.state_dict(),
                "patch_decoder": patch_decoder.state_dict(),
                "curve_encoder": curve_encoder.state_dict(),
                "curve_decoder": curve_decoder.state_dict(),
                "patch_optimizer": patch_optimizer.state_dict(),
                "curve_optimizer": curve_optimizer.state_dict(),
            }
            if config.USE_AMP:
                checkpoint_dict["patch_scaler"] = patch_scaler.state_dict()
                checkpoint_dict["curve_scaler"] = curve_scaler.state_dict()
            
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )
            logger.info(f"Saved checkpoint for epoch {epoch+1}")

    if rank == 0:
        print("\n" + "="*80)
        print("训练完成！各项损失含义解释：")
        print("="*80)
        print("【Patch 面片损失】:")
        print("  • total: 总损失 - 所有损失项的加权和，用于反向传播优化")
        print("  • kl: KL散度损失 - 确保编码后的潜在向量符合标准正态分布，实现正则化")
        print("  • recon_points: 点重建损失 - 衡量重建的3D点坐标与真实坐标的差异")
        print("  • topology: 拓扑损失 - 预测面片在U/V方向是否闭合的分类准确性")
        print()
        print("【Curve 曲线损失】:")
        print("  • total: 总损失 - 所有损失项的加权和，用于反向传播优化")
        print("  • kl: KL散度损失 - 确保编码后的潜在向量符合标准正态分布，实现正则化")
        print("  • recon_points: 点重建损失 - 衡量重建的3D曲线点坐标与真实坐标的差异")
        print("  • topology: 拓扑损失 - 预测曲线是否为闭合曲线的分类准确性")
        print()
        print("【验证损失】:")
        print("  • recon_error: 重建误差 - 在验证集上测量的几何重建精度，越小越好")
        print("="*80)
        
        wandb.finish()

def val_pipeline(
    patch_encoder,
    patch_decoder,
    curve_encoder,
    curve_decoder,
    val_dataloader,
    val_sampler,  
    epoch,  
    device,
    config,
    rank=0,
):
    """Validation pipeline with sampler support"""
    if val_sampler is not None:
        val_sampler.set_epoch(epoch)
    
    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()

    patch_recon_errors = []
    curve_recon_errors = []

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation", leave=False, dynamic_ncols=True) if rank == 0 else val_dataloader
        for batch_idx, data_item in enumerate(pbar):
            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            if processed_patches is not None:
                mean_p, _ = patch_encoder(
                    processed_patches["patch_points"],
                    processed_patches["patch_normals"],
                    processed_patches["u_closed"],
                    processed_patches["v_closed"],
                    processed_patches["labels"],
                    processed_patches["mask"],
                )

                pred_patch = patch_decoder(mean_p)

                recon_error = F.mse_loss(
                    pred_patch["points"], 
                    processed_patches["patch_points"], 
                    reduction="none"
                )
                recon_error = (recon_error * processed_patches["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_patches["mask"].sum().clamp(min=1)
                patch_recon_errors.append(recon_error)

            if processed_curves is not None:
                mean_c, _ = curve_encoder(
                    processed_curves["curve_points"],
                    processed_curves["endpoints"],
                    processed_curves["is_closed"],
                    processed_curves["labels"],
                    processed_curves["mask"],
                )

                pred_curve = curve_decoder(mean_c)

                recon_error = F.mse_loss(
                    pred_curve["points"], 
                    processed_curves["curve_points"], 
                    reduction="none"
                )
                recon_error = (recon_error * processed_curves["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_curves["mask"].sum().clamp(min=1)
                curve_recon_errors.append(recon_error)

    val_metrics = {}

    if patch_recon_errors:
        patch_errors_tensor = torch.stack(patch_recon_errors)
        val_metrics["patch"] = {
            "recon_error": patch_errors_tensor.mean().item()
        }

    if curve_recon_errors:
        curve_errors_tensor = torch.stack(curve_recon_errors)
        val_metrics["curve"] = {
            "recon_error": curve_errors_tensor.mean().item()
        }

    patch_encoder.train()
    patch_decoder.train()
    curve_encoder.train()
    curve_decoder.train()

    return val_metrics

def eval_pipeline(args, config):
    """Complete evaluation pipeline"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    points_per_patch_dim = (
        config.POINTS_PER_PATCH_DIM if hasattr(config, "POINTS_PER_PATCH_DIM") else 32
    )

    if args.quicktest:
        test_data, _ = train_data_loader(
            args.batch_size,
            voxel_dim=voxel_dim,
            data_folder="data/train_small",
            feature_type=args.input_feature_type,
            pad1s=not args.backbone_feature_encode,
            rotation_augmentation=args.rotation_augment,
            with_normal=args.input_normal_signals,
            flag_quick_test=False,
            flag_noise=args.noise,
            flag_grid=args.patch_grid,
            num_angle=args.num_angles,
            flag_patch_uv=args.patch_uv,
            dim_grid=points_per_patch_dim,
            eval_res_cov=args.extra_single_chamfer,
        )
    else:
        if not args.patch_grid:
            test_folder = "val_new_64"
        else:
            test_folder = "data/partial/val" if args.partial else "data/default/val"

        test_data, _ = train_data_loader(
            args.batch_size,
            voxel_dim=voxel_dim,
            data_folder=test_folder,
            feature_type=args.input_feature_type,
            pad1s=not args.backbone_feature_encode,
            rotation_augmentation=args.rotation_augment,
            with_normal=args.input_normal_signals,
            flag_quick_test=False,
            flag_noise=args.noise,
            flag_grid=args.patch_grid,
            num_angle=args.num_angles,
            flag_patch_uv=args.patch_uv,
            dim_grid=points_per_patch_dim,
            eval_res_cov=args.extra_single_chamfer,
        )

    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "ckpt")

    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)

    if not hasattr(args, "checkpoint_path") or not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified for evaluation")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    patch_encoder.load_state_dict(checkpoint["patch_encoder"])
    patch_decoder.load_state_dict(checkpoint["patch_decoder"])
    curve_encoder.load_state_dict(checkpoint["curve_encoder"])
    curve_decoder.load_state_dict(checkpoint["curve_decoder"])

    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()

    patch_metrics_all = []
    curve_metrics_all = []

    with torch.no_grad():
        for batch_idx, data_item in enumerate(tqdm(test_data, desc="Evaluating", leave=True, dynamic_ncols=True)):
            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            if processed_patches is not None:
                mean_p, _ = patch_encoder(
                    processed_patches["patch_points"],
                    processed_patches["patch_normals"],
                    processed_patches["u_closed"],
                    processed_patches["v_closed"],
                    processed_patches["labels"],
                    processed_patches["mask"],
                )

                pred_patch = patch_decoder(mean_p)

                target_patch = {
                    "points": processed_patches["patch_points"],
                    "normals": processed_patches["patch_normals"],
                    "u_closed": processed_patches["u_closed"],
                    "v_closed": processed_patches["v_closed"],
                    "labels": processed_patches["labels"],
                }

                patch_metrics = compute_patch_metrics(
                    pred_patch, target_patch, processed_patches["mask"]
                )
                patch_metrics_all.append(patch_metrics)

            if processed_curves is not None:
                mean_c, _ = curve_encoder(
                    processed_curves["curve_points"],
                    processed_curves["endpoints"],
                    processed_curves["is_closed"],
                    processed_curves["labels"],
                    processed_curves["mask"],
                )

                pred_curve = curve_decoder(mean_c)

                target_curve = {
                    "points": processed_curves["curve_points"],
                    "endpoints": processed_curves["endpoints"],
                    "is_closed": processed_curves["is_closed"],
                    "labels": processed_curves["labels"],
                }

                curve_metrics = compute_curve_metrics(
                    pred_curve, target_curve, processed_curves["mask"]
                )
                curve_metrics_all.append(curve_metrics)

    avg_patch_metrics = {}
    avg_curve_metrics = {}

    if patch_metrics_all:
        for key in patch_metrics_all[0].keys():
            avg_patch_metrics[key] = sum(m[key] for m in patch_metrics_all) / len(
                patch_metrics_all
            )

    if curve_metrics_all:
        for key in curve_metrics_all[0].keys():
            avg_curve_metrics[key] = sum(m[key] for m in curve_metrics_all) / len(
                curve_metrics_all
            )

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    if avg_patch_metrics:
        print("\n面片指标:")
        for key, val in avg_patch_metrics.items():
            print(f"  {key}: {val:.6f}")

    if avg_curve_metrics:
        print("\n曲线指标:")
        for key, val in avg_curve_metrics.items():
            print(f"  {key}: {val:.6f}")

    print("=" * 60)

    test_metrics = {"patch": avg_patch_metrics, "curve": avg_curve_metrics}

    if hasattr(args, "experiment_name") and args.experiment_name:
        results_path = os.path.join(args.checkpoint_dir, "test_results.txt")
        with open(results_path, "w", encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("测试结果\n")
            f.write("=" * 60 + "\n\n")

            if avg_patch_metrics:
                f.write("面片指标:\n")
                for key, value in avg_patch_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")

            if avg_curve_metrics:
                f.write("曲线指标:\n")
                for key, value in avg_curve_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")

        print(f"\n结果已保存到: {results_path}")

    return test_metrics
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(
        "Training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    num_of_gpus = torch.cuda.device_count()

    config = Config()

    if args.eval:
        eval_pipeline(args, config)
    else:
        if num_of_gpus > 1:
            mp.spawn(
                train_pipeline,
                args=(num_of_gpus, args, config),
                nprocs=num_of_gpus,
                join=True,
            )
        else:
            train_pipeline(0, 1, args, config)