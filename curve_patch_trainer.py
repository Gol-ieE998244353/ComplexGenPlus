import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, Optional, Dict
from Minkowski_backbone import (
    get_args_parser,
    voxel_dim,
    points_per_patch_dim,
    num_of_gpus,
    prepare_experiment_folders,
    MLP,
    MLP_hn,
)
from data_loader_abc import *
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import torch.distributed as dist


class Config:
    # Model Architecture
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Latent Space
    CURVE_LATENT_DIM = 128
    PATCH_LATENT_DIM = 128

    # Data
    CURVE_NUM_POINTS = 34
    PATCH_NUM_POINTS = 400  # 20x20
    POINTS_PER_PATCH_DIM = 20
    MAX_CURVES = 100
    MAX_PATCHES = 50

    # HyperNetwork (for decoder)
    HN_PE_DIM = 32
    HN_MLP_DIM = 128

    # Position Encoding (只在使用transformer时需要)
    VOXEL_DIM = 128
    PE_TEMPERATURE = 10000
    PE_SCALE = 2 * math.pi

    # Labels
    CURVE_NUM_CLASSES = 4
    PATCH_NUM_CLASSES = 6

    # VAE
    KL_WEIGHT_START = 0.0
    KL_WEIGHT_END = 0.1
    KL_WARMUP_EPOCHS = 10

    # Training
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    EVAL_INTERVAL = 5
    SAVE_INTERVAL = 10

    # Loss weights
    RECON_WEIGHT = 1.0
    ENDPOINT_WEIGHT = 1.0
    TOPOLOGY_WEIGHT = 0.5
    LABEL_WEIGHT = 0.5
    VALIDITY_WEIGHT = 0.1

    # === 实验性选项 ===
    USE_TRANSFORMER = False  # 是否使用transformer让curves/patches互相attention
    # 如果为False，每个curve/patch独立编码（推荐）


class PositionEmbeddingSine3D(nn.Module):
    """3D sinusoidal positional encoding (仅在USE_TRANSFORMER=True时使用)"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, xyz_coords, voxel_dim=64):
        B, N, _ = xyz_coords.shape

        if self.normalize:
            coords = self.scale * xyz_coords / (voxel_dim - 1)
        else:
            coords = xyz_coords

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=xyz_coords.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = coords[:, :, :, None] / dim_t
        pos_x, pos_y, pos_z = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]

        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_z = torch.stack(
            (pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        return torch.cat((pos_x, pos_y, pos_z), dim=2)


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
        """
        Args:
            points: [B, N, P, C]
        Returns:
            features: [B, N, output_dim]
        """
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

        # Geometry feature extraction (PointNet)
        self.geom_encoder = PointNetEncoder(
            input_dim=3, output_dim=config.D_MODEL // 4, dropout=config.DROPOUT
        )

        # Endpoint encoding
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, config.D_MODEL // 4),
        )

        # Closed flag encoding
        self.closed_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, config.D_MODEL // 4),
        )

        # Type embedding
        self.label_embedding = nn.Embedding(
            config.CURVE_NUM_CLASSES, config.D_MODEL // 4
        )

        # Feature fusion
        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)

        # Optional Transformer
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

        # Project to latent space
        self.to_mean = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)
        self.to_logvar = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)

    def forward(self, curve_points, endpoints, is_closed, labels, mask):
        """
        Args:
            curve_points: [B, N, 34, 3]
            endpoints: [B, N, 2, 3] - endpoint coordinates (zeros for closed curves)
            is_closed: [B, N] - bool (True=closed curve)
            labels: [B, N]
            mask: [B, N] - bool (True=valid)

        Returns:
            mean, logvar: [B, N, latent_dim]
        """
        B, N = curve_points.shape[:2]
        device = curve_points.device

        # Flatten endpoints for encoder: [B, N, 6]
        endpoints_flat = endpoints.reshape(B, N, 6)

        # Encode features
        geom_feat = self.geom_encoder(curve_points)  # [B, N, D/4]
        endpoint_feat = self.endpoint_encoder(endpoints_flat)  # [B, N, D/4]
        closed_feat = self.closed_encoder(
            is_closed.unsqueeze(-1).float()
        )  # [B, N, D/4]
        label_feat = self.label_embedding(labels)  # [B, N, D/4]

        # Fuse features
        all_feat = torch.cat(
            [geom_feat, endpoint_feat, closed_feat, label_feat], dim=-1
        )
        tokens = self.fusion(all_feat)  # [B, N, D_MODEL]

        # Optional transformer
        if self.config.USE_TRANSFORMER:
            centroids = curve_points.mean(dim=2)  # [B, N, 3]
            pos_enc = self.pos_encoder(centroids, voxel_dim=self.config.VOXEL_DIM)
            tokens = tokens + pos_enc

            attn_mask = ~mask if mask is not None else None
            contextualized = self.transformer(tokens, src_key_padding_mask=attn_mask)
            contextualized = contextualized * mask.unsqueeze(-1).float()
        else:
            contextualized = tokens
            contextualized = contextualized * mask.unsqueeze(-1).float()

        # Project to latent space
        mean = self.to_mean(contextualized)
        logvar = self.to_logvar(contextualized)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mean, logvar

class CurveDecoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # === 1. Start point predictor ===
        self.start_point_embed = MLP(
            config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 3, 3
        )

        # === 2. Parametric encoding ===
        self.curve_pe = self._init_curve_pe(config.CURVE_NUM_POINTS, config.HN_PE_DIM)

        # === 3. HyperNetwork ===
        self.curve_shape_embed = MLP_hn(
            input_dim=config.HN_PE_DIM,
            hidden_dim=config.HN_MLP_DIM,
            output_dim=3,
            num_layers=3,
            input_dim_fea=config.CURVE_LATENT_DIM,
        )

        # === 4. Attribute predictors ===
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
        """初始化参数化位置编码 t ∈ [0,1]"""
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
        """
        Args:
            z: [B, N, latent_dim]
        Returns:
            dict with reconstructed attributes
        """
        B, N = z.shape[:2]

        # 1. Predict start point
        start_point = self.start_point_embed(z).tanh() * 0.5

        # 2. Get parametric encoding
        curve_pe = self.curve_pe.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

        # 3. HyperNetwork生成curve shape
        shape_offset = self.curve_shape_embed(curve_pe.unsqueeze(0), z.unsqueeze(0))

        # 4. Final curve points
        points = start_point.unsqueeze(2) + shape_offset

        # 5. Predict attributes
        endpoints = self.endpoints_head(z).view(B, N, 2, 3)
        closed_logits = self.closed_head(z).squeeze(-1)
        label_logits = self.label_head(z)
        validity_logits = self.validity_head(z).squeeze(-1)

        return {
            "points": points,
            "endpoints": endpoints,
            "closed_logits": closed_logits,
            "label_logits": label_logits,
            "validity_logits": validity_logits,
        }

class PatchEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 几何编码（points + normals）
        self.geom_encoder = PointNetEncoder(
            input_dim=6, output_dim=config.D_MODEL // 2, dropout=config.DROPOUT
        )

        # 拓扑编码
        self.topo_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, config.D_MODEL // 4),
        )

        # 类型embedding
        self.label_embedding = nn.Embedding(
            config.PATCH_NUM_CLASSES, config.D_MODEL // 4
        )

        # 特征融合
        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)

        # [可选] Transformer
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

        # 投影到latent
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
            pos_enc = self.pos_encoder(centroids, voxel_dim=self.config.VOXEL_DIM)
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
    """Patch Decoder - 使用HyperNetwork"""

    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.patch_dim = int(math.sqrt(config.PATCH_NUM_POINTS))

        # Center point predictor
        self.startpoint_embed = MLP(
            config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 3, 3
        )

        # Parametric encoding
        self.patch_pe_u, self.patch_pe_v = self._init_patch_pe(
            self.patch_dim, config.HN_PE_DIM
        )

        # HyperNetwork
        self.patch_shape_embed = MLP_hn(
            input_dim=config.HN_PE_DIM * 2,
            hidden_dim=config.HN_MLP_DIM,
            output_dim=6,  # 3 for points + 3 for normals
            num_layers=3,
            input_dim_fea=config.PATCH_LATENT_DIM,
        )

        # Attribute predictors
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

        # 3. HyperNetwork生成shape + normals
        output = self.patch_shape_embed(patch_pe.unsqueeze(0), z.unsqueeze(0))
        shape_offset = output[..., :3]
        normals = output[..., 3:]

        # 4. Final points and normals
        points = startpoint.unsqueeze(2) + shape_offset
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        # 5. Predict attributes
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
    """简单的KL warmup调度器"""

    def __init__(self, start_weight=0.0, end_weight=0.1, warmup_epochs=10):
        self.start = start_weight
        self.end = end_weight
        self.warmup = warmup_epochs

    def get_weight(self, epoch):
        """线性warmup"""
        if epoch >= self.warmup:
            return self.end
        return self.start + (self.end - self.start) * (epoch / self.warmup)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """Focal Loss - 解决P0问题：类别不平衡

    FL(p_t) = -α(1-p_t)^γ log(p_t)

    Args:
        logits: 预测logits [B, N, C] or [B, N]
        targets: 目标标签 [B, N]
        alpha: 平衡因子
        gamma: 聚焦参数
    """
    if logits.dim() == 2:  # Binary classification
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    else:  # Multi-class
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
    """
    Compute patch VAE losses with improved loss functions and weighting support

    Args:
        pred: Predictions from decoder
        target: Target data
        mean, logvar: VAE latent parameters
        mask: Valid patch mask [B, N]
        kl_weight: Current KL annealing weight
        config: Configuration object
        weighting: Optional patch area weighting [B, N]
    """
    losses = {}

    # KL divergence loss
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    kl = (kl * mask.unsqueeze(-1).float()).sum() / mask.sum().clamp(min=1)
    losses["kl"] = kl

    # Apply weighting if provided
    if weighting is not None:
        # Normalize weighting to sum to number of valid patches
        valid_weighting = weighting * mask.float()
        valid_weighting = (
            valid_weighting / (valid_weighting.sum() + 1e-8) * mask.sum().float()
        )
        weight_mask = valid_weighting.unsqueeze(-1).unsqueeze(-1)
    else:
        weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()

    # Reconstruction losses with weighting
    recon_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points

    recon_normals = F.mse_loss(pred["normals"], target["normals"], reduction="none")
    recon_normals = (recon_normals * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_normals"] = recon_normals

    # Topology losses with Focal Loss if enabled
    topo_weight = (
        weighting if weighting is not None else torch.ones_like(mask, dtype=torch.float)
    )
    topo_weight = topo_weight * mask.float()

    if hasattr(config, "USE_FOCAL_LOSS") and config.USE_FOCAL_LOSS:
        u_closed_loss = focal_loss(
            pred["u_closed_logits"],
            target["u_closed"],
            getattr(config, "FOCAL_ALPHA", 0.25),
            getattr(config, "FOCAL_GAMMA", 2.0),
        )
        v_closed_loss = focal_loss(
            pred["v_closed_logits"],
            target["v_closed"],
            getattr(config, "FOCAL_ALPHA", 0.25),
            getattr(config, "FOCAL_GAMMA", 2.0),
        )
    else:
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

    # Label loss with weighting
    label_loss = F.cross_entropy(
        pred["label_logits"].view(-1, pred["label_logits"].size(-1)),
        target["labels"].view(-1),
        reduction="none",
    )
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss

    # Validity loss
    validity_loss = F.binary_cross_entropy_with_logits(
        pred["validity_logits"], mask.float(), reduction="mean"
    )
    losses["validity"] = validity_loss

    # Total loss with configurable weights
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
    """
    Compute curve VAE losses with improved loss functions and weighting support

    Args:
        pred: Predictions from decoder
        target: Target data
        mean, logvar: VAE latent parameters
        mask: Valid curve mask [B, N]
        kl_weight: Current KL annealing weight
        config: Configuration object
        weighting: Optional curve length weighting [B, N]
    """
    losses = {}

    # KL divergence loss
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    kl = (kl * mask.unsqueeze(-1).float()).sum() / mask.sum().clamp(min=1)
    losses["kl"] = kl

    # Apply weighting if provided
    if weighting is not None:
        # Normalize weighting to sum to number of valid curves
        valid_weighting = weighting * mask.float()
        valid_weighting = (
            valid_weighting / (valid_weighting.sum() + 1e-8) * mask.sum().float()
        )
        weight_mask = valid_weighting.unsqueeze(-1).unsqueeze(-1)
    else:
        weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()

    # Reconstruction losses with weighting
    recon_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points

    # Endpoint loss: only for open curves
    open_mask = ~target["is_closed"]  # [B, N]
    endpoint_mask = (mask & open_mask).unsqueeze(-1).unsqueeze(-1).float()  # [B, N, 1, 1]
    
    recon_endpoints = F.mse_loss(
        pred["endpoints"], target["endpoints"], reduction="none"
    )  # [B, N, 2, 3]
    
    if weighting is not None:
        endpoint_weighting = (weighting * mask.float() * open_mask.float()).unsqueeze(-1).unsqueeze(-1)
        endpoint_weighting = endpoint_weighting / (endpoint_weighting.sum() + 1e-8) * open_mask.sum().float().clamp(min=1)
        recon_endpoints = (recon_endpoints * endpoint_weighting).sum() / open_mask.sum().clamp(min=1)
    else:
        recon_endpoints = (recon_endpoints * endpoint_mask).sum() / open_mask.sum().clamp(min=1)
    
    losses["recon_endpoints"] = recon_endpoints

    # Closed flag loss with weighting
    topo_weight = (
        weighting if weighting is not None else torch.ones_like(mask, dtype=torch.float)
    )
    topo_weight = topo_weight * mask.float()

    if hasattr(config, "USE_FOCAL_LOSS") and config.USE_FOCAL_LOSS:
        closed_loss = focal_loss(
            pred["closed_logits"],
            target["is_closed"],
            getattr(config, "FOCAL_ALPHA", 0.25),
            getattr(config, "FOCAL_GAMMA", 2.0),
        )
    else:
        closed_loss = F.binary_cross_entropy_with_logits(
            pred["closed_logits"], target["is_closed"].float(), reduction="none"
        )
        closed_loss = (closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)

    losses["closed"] = closed_loss

    # Label loss with weighting
    label_loss = F.cross_entropy(
        pred["label_logits"].view(-1, pred["label_logits"].size(-1)),
        target["labels"].view(-1),
        reduction="none",
    )
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss

    # Validity loss
    validity_loss = F.binary_cross_entropy_with_logits(
        pred["validity_logits"], mask.float(), reduction="mean"
    )
    losses["validity"] = validity_loss

    # Total loss
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
    """
    Process batch data from dataloader into proper format for VAE training

    Args:
        data_item: Raw batch from dataloader
        config: Configuration object
        device: Target device

    Returns:
        processed_curves: Dict with curve data
        processed_patches: Dict with patch data
    """
    # Extract data components
    corner_points = data_item[0]  # [20, 3]
    corner_batch_idx = data_item[1]  # [20]
    batch_sample_id = data_item[2]  # list
    target_curves_list = data_item[6]  # list of curve dicts
    target_patches_list = data_item[7]  # list of patch dicts

    processed_curves = None
    processed_patches = None

    # ===== Process Curves =====
    if len(target_curves_list) > 0:
        batch_size = len(target_curves_list)

        # Find max number of curves in this batch
        max_n_curves = max([c["curve_points"].shape[0] for c in target_curves_list])
        max_n_curves = min(max_n_curves, config.MAX_CURVES)  # Cap at config limit

        # Initialize batch tensors
        curve_points_batch = torch.zeros(batch_size, max_n_curves, 34, 3, device=device)
        endpoints_batch = torch.zeros(batch_size, max_n_curves, 2, 3, device=device)
        is_closed_batch = torch.zeros(
            batch_size, max_n_curves, dtype=torch.bool, device=device
        )
        labels_batch = torch.zeros(
            batch_size, max_n_curves, dtype=torch.long, device=device
        )
        mask_batch = torch.zeros(
            batch_size, max_n_curves, dtype=torch.bool, device=device
        )
        weighting_batch = torch.ones(
            batch_size, max_n_curves, dtype=torch.float, device=device
        )

        # Fill batch tensors
        for i, curve_dict in enumerate(target_curves_list):
            n_curves = min(curve_dict["curve_points"].shape[0], max_n_curves)

            curve_points_batch[i, :n_curves] = curve_dict["curve_points"][:n_curves].to(
                device
            )
            is_closed_batch[i, :n_curves] = curve_dict["is_closed"][:n_curves].to(
                device
            )
            labels_batch[i, :n_curves] = curve_dict["labels"][:n_curves].to(device)
            mask_batch[i, :n_curves] = True

            # Extract endpoint coordinates for open curves
            endpoint_indices = curve_dict["endpoints"][:n_curves].long()  # [N, 2]
            curve_points = curve_dict["curve_points"][:n_curves]  # [N, 34, 3]
            is_closed = curve_dict["is_closed"][:n_curves]  # [N]
            
            for j in range(n_curves):
                if not is_closed[j]:  # Only for open curves
                    idx0, idx1 = endpoint_indices[j]
                    idx0 = idx0.clamp(min=0, max=33)
                    idx1 = idx1.clamp(min=0, max=33)
                    endpoints_batch[i, j, 0] = curve_points[j, idx0]
                    endpoints_batch[i, j, 1] = curve_points[j, idx1]
                # For closed curves, endpoints remain zeros

            # Handle weighting
            if "curve_length_weighting" in curve_dict:
                weighting_batch[i, :n_curves] = curve_dict["curve_length_weighting"][
                    :n_curves
                ].to(device)

        processed_curves = {
            "curve_points": curve_points_batch,
            "endpoints": endpoints_batch,
            "is_closed": is_closed_batch,
            "labels": labels_batch,
            "mask": mask_batch,
            "weighting": weighting_batch,
        }

    # ===== Process Patches =====
    if len(target_patches_list) > 0:
        batch_size = len(target_patches_list)

        # Find max number of patches in this batch
        max_n_patches = max([len(p["patch_points"]) for p in target_patches_list])
        max_n_patches = min(max_n_patches, config.MAX_PATCHES)  # Cap at config limit

        # Initialize batch tensors
        patch_points_batch = torch.zeros(
            batch_size, max_n_patches, 400, 3, device=device
        )
        patch_normals_batch = torch.zeros(
            batch_size, max_n_patches, 400, 3, device=device
        )
        u_closed_batch = torch.zeros(
            batch_size, max_n_patches, dtype=torch.bool, device=device
        )
        v_closed_batch = torch.zeros(
            batch_size, max_n_patches, dtype=torch.bool, device=device
        )
        labels_batch = torch.zeros(
            batch_size, max_n_patches, dtype=torch.long, device=device
        )
        mask_batch = torch.zeros(
            batch_size, max_n_patches, dtype=torch.bool, device=device
        )
        weighting_batch = torch.ones(
            batch_size, max_n_patches, dtype=torch.float, device=device
        )

        # Fill batch tensors
        for i, patch_dict in enumerate(target_patches_list):
            n_patches = min(len(patch_dict["patch_points"]), max_n_patches)

            # Handle list structure for patch points and normals
            for j in range(n_patches):
                patch_points_batch[i, j] = patch_dict["patch_points"][j].to(device)
                patch_normals_batch[i, j] = patch_dict["patch_normals"][j].to(device)

            u_closed_batch[i, :n_patches] = patch_dict["u_closed"][:n_patches].to(
                device
            )
            v_closed_batch[i, :n_patches] = patch_dict["v_closed"][:n_patches].to(
                device
            )
            labels_batch[i, :n_patches] = patch_dict["labels"][:n_patches].to(device)
            mask_batch[i, :n_patches] = True

            # Handle weighting
            if "patch_area_weighting" in patch_dict:
                weighting_batch[i, :n_patches] = patch_dict["patch_area_weighting"][
                    :n_patches
                ].to(device)

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

def compute_patch_metrics(pred, target, mask):
    """
    计算Patch的评估指标（不用于训练，只用于评估）

    Args:
        pred: 预测结果字典，包含 points, normals, u_closed_logits, v_closed_logits,
              label_logits, validity_logits
        target: 目标数据字典，包含 points, normals, u_closed, v_closed, labels
        mask: 有效patch的mask [B, N]

    Returns:
        metrics: 包含各种评估指标的字典
    """
    metrics = {}

    # 1. 重建误差 - Points (越小越好)
    recon_error_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_error_points = (
        recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()

    # 2. 重建误差 - Normals (越小越好)
    recon_error_normals = F.mse_loss(
        pred["normals"], target["normals"], reduction="none"
    )
    recon_error_normals = (
        recon_error_normals * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error_normals"] = recon_error_normals.item()

    # 3. 标签分类准确率 (越高越好)
    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    label_accuracy = correct_labels.sum().float() / mask.sum().float()
    metrics["label_accuracy"] = label_accuracy.item()

    # 4. U方向闭合预测准确率 (越高越好)
    pred_u_closed = torch.sigmoid(pred["u_closed_logits"]) > 0.5
    correct_u = (pred_u_closed == target["u_closed"]) & mask
    u_closed_accuracy = correct_u.sum().float() / mask.sum().float()
    metrics["u_closed_accuracy"] = u_closed_accuracy.item()

    # 5. V方向闭合预测准确率 (越高越好)
    pred_v_closed = torch.sigmoid(pred["v_closed_logits"]) > 0.5
    correct_v = (pred_v_closed == target["v_closed"]) & mask
    v_closed_accuracy = correct_v.sum().float() / mask.sum().float()
    metrics["v_closed_accuracy"] = v_closed_accuracy.item()

    # 6. 有效性预测准确率 (预测哪些patch是有效的)
    pred_validity = torch.sigmoid(pred["validity_logits"]) > 0.5
    validity_correct = (pred_validity == mask).sum().float() / pred_validity.numel()
    metrics["validity_accuracy"] = validity_correct.item()

    # 7. 总体拓扑准确率（综合u_closed和v_closed）
    topology_correct = correct_u & correct_v
    topology_accuracy = topology_correct.sum().float() / mask.sum().float()
    metrics["topology_accuracy"] = topology_accuracy.item()

    return metrics

def compute_curve_metrics(pred, target, mask):
    """
    计算Curve的评估指标（不用于训练，只用于评估）

    Args:
        pred: 预测结果字典，包含 points, endpoints, closed_logits, label_logits, validity_logits
        target: 目标数据字典，包含 points, endpoints, is_closed, labels
        mask: 有效curve的mask [B, N]

    Returns:
        metrics: 包含各种评估指标的字典
    """
    metrics = {}

    # 1. 重建误差 - Curve Points (越小越好)
    recon_error_points = F.mse_loss(pred["points"], target["points"], reduction="none")
    recon_error_points = (
        recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()

    # 2. 重建误差 - Endpoints (越小越好)
    recon_error_endpoints = F.mse_loss(
        pred["endpoints"], target["endpoints"], reduction="none"
    )
    recon_error_endpoints = (
        recon_error_endpoints * mask.unsqueeze(-1).unsqueeze(-1)
    ).sum() / mask.sum().clamp(min=1)
    metrics["recon_error_endpoints"] = recon_error_endpoints.item()

    # 3. 标签分类准确率 (越高越好)
    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    label_accuracy = correct_labels.sum().float() / mask.sum().float()
    metrics["label_accuracy"] = label_accuracy.item()

    # 4. 闭合标志预测准确率 (越高越好)
    pred_closed = torch.sigmoid(pred["closed_logits"]) > 0.5
    correct_closed = (pred_closed == target["is_closed"]) & mask
    closed_accuracy = correct_closed.sum().float() / mask.sum().float()
    metrics["closed_accuracy"] = closed_accuracy.item()

    # 5. 有效性预测准确率 (预测哪些curve是有效的)
    pred_validity = torch.sigmoid(pred["validity_logits"]) > 0.5
    validity_correct = (pred_validity == mask).sum().float() / pred_validity.numel()
    metrics["validity_accuracy"] = validity_correct.item()

    # 6. Endpoint预测误差（单独统计起点和终点）
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


# ============================================================================
# Updated Training Pipeline
# ============================================================================


def train_pipeline(rank, num_gpus, args, config):
    """
    Complete training pipeline with data loading and distributed setup
    """
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23257",
        world_size=num_gpus,
        rank=rank,
    )
    # ===== Setup distributed training =====
    if num_gpus > 1:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Training on {num_gpus} GPU(s)")

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
        # Training data
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

        # Validation data
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

    # ===== Setup experiment directories =====
    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "ckpt")

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Experiment directory: {experiment_dir}")
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    # ===== Initialize models =====
    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)

    # Wrap with DDP if multi-GPU
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

    # ===== Initialize optimizers =====
    patch_params = list(patch_encoder.parameters()) + list(patch_decoder.parameters())
    curve_params = list(curve_encoder.parameters()) + list(curve_decoder.parameters())

    patch_optimizer = torch.optim.AdamW(
        patch_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    curve_optimizer = torch.optim.AdamW(
        curve_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # ===== Initialize schedulers =====
    kl_scheduler = KLScheduler(
        config.KL_WEIGHT_START, config.KL_WEIGHT_END, config.KL_WARMUP_EPOCHS
    )

    # ===== Load checkpoint if specified =====
    start_epoch = 0
    if (
        hasattr(args, "checkpoint_path")
        and args.checkpoint_path
        and os.path.exists(args.checkpoint_path)
    ):
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        patch_encoder.load_state_dict(checkpoint["patch_encoder"])
        patch_decoder.load_state_dict(checkpoint["patch_decoder"])
        curve_encoder.load_state_dict(checkpoint["curve_encoder"])
        curve_decoder.load_state_dict(checkpoint["curve_decoder"])
        patch_optimizer.load_state_dict(checkpoint["patch_optimizer"])
        curve_optimizer.load_state_dict(checkpoint["curve_optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    # ===== Training loop =====
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Get current KL weight
        current_kl_weight = kl_scheduler.get_weight(epoch)

        if rank == 0:
            print(
                f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} - KL Weight: {current_kl_weight:.4f}"
            )

        # Set models to training mode
        patch_encoder.train()
        patch_decoder.train()
        curve_encoder.train()
        curve_decoder.train()

        # Training epoch
        patch_losses_all = []
        curve_losses_all = []

        if rank == 0:
            pbar = tqdm(train_data, desc=f"Epoch {epoch+1} [Train]")
        else:
            pbar = train_data

        for batch_idx, data_item in enumerate(pbar):
            # Process batch data
            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            # ===== Patch VAE Training =====
            if processed_patches is not None:
                # Forward pass
                mean_p, logvar_p = patch_encoder(
                    processed_patches["patch_points"],
                    processed_patches["patch_normals"],
                    processed_patches["u_closed"],
                    processed_patches["v_closed"],
                    processed_patches["labels"],
                    processed_patches["mask"],
                )

                # Reparameterization
                std_p = torch.exp(0.5 * logvar_p)
                eps_p = torch.randn_like(std_p)
                z_p = mean_p + eps_p * std_p

                # Decode
                pred_patch = patch_decoder(z_p)

                # Compute losses with weighting
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

                # Backward pass
                patch_optimizer.zero_grad()
                patch_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    patch_params, max_norm=config.GRAD_CLIP_NORM
                )
                patch_optimizer.step()

                patch_losses_all.append(patch_losses["total"].item())

            # ===== Curve VAE Training =====
            if processed_curves is not None:
                # Forward pass
                mean_c, logvar_c = curve_encoder(
                    processed_curves["curve_points"],
                    processed_curves["endpoints"],
                    processed_curves["is_closed"],
                    processed_curves["labels"],
                    processed_curves["mask"],
                )

                # Reparameterization
                std_c = torch.exp(0.5 * logvar_c)
                eps_c = torch.randn_like(std_c)
                z_c = mean_c + eps_c * std_c

                # Decode
                pred_curve = curve_decoder(z_c)

                # Compute losses with weighting
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

                # Backward pass
                curve_optimizer.zero_grad()
                curve_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    curve_params, max_norm=config.GRAD_CLIP_NORM
                )
                curve_optimizer.step()

                curve_losses_all.append(curve_losses["total"].item())

            # Update progress bar
            if rank == 0:
                postfix = {}
                if patch_losses_all:
                    postfix["patch_loss"] = f"{patch_losses_all[-1]:.4f}"
                if curve_losses_all:
                    postfix["curve_loss"] = f"{curve_losses_all[-1]:.4f}"
                if postfix:
                    pbar.set_postfix(postfix)

        # Compute average losses
        avg_patch_loss = (
            sum(patch_losses_all) / len(patch_losses_all) if patch_losses_all else 0
        )
        avg_curve_loss = (
            sum(curve_losses_all) / len(curve_losses_all) if curve_losses_all else 0
        )

        if rank == 0:
            print(
                f"Epoch {epoch+1} - Avg Patch Loss: {avg_patch_loss:.4f}, Avg Curve Loss: {avg_curve_loss:.4f}"
            )

        # Validation
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            if rank == 0:
                print("Running validation...")

            val_metrics = val_pipeline(
                patch_encoder,
                patch_decoder,
                curve_encoder,
                curve_decoder,
                val_data,
                device,
                config,
            )

            if rank == 0:
                if "patch" in val_metrics and val_metrics["patch"]:
                    print(
                        f"Validation - Patch Recon Error: {val_metrics['patch']['recon_error']:.6f}"
                    )
                    print(
                        f"Validation - Patch Label Acc: {val_metrics['patch']['label_accuracy']:.4f}"
                    )
                if "curve" in val_metrics and val_metrics["curve"]:
                    print(
                        f"Validation - Curve Recon Error: {val_metrics['curve']['recon_error']:.6f}"
                    )
                    print(
                        f"Validation - Curve Label Acc: {val_metrics['curve']['label_accuracy']:.4f}"
                    )

            # Save best model
            patch_loss = val_metrics.get("patch", {}).get("recon_error", 0)
            curve_loss = val_metrics.get("curve", {}).get("recon_error", 0)
            val_loss = patch_loss + curve_loss

            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "patch_encoder": patch_encoder.state_dict(),
                        "patch_decoder": patch_decoder.state_dict(),
                        "curve_encoder": curve_encoder.state_dict(),
                        "curve_decoder": curve_decoder.state_dict(),
                        "patch_optimizer": patch_optimizer.state_dict(),
                        "curve_optimizer": curve_optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    os.path.join(args.checkpoint_dir, "best_model.pth"),
                )
                print(f"Saved best model with val_loss: {val_loss:.6f}")

        # Save checkpoint
        if rank == 0 and (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "patch_encoder": patch_encoder.state_dict(),
                    "patch_decoder": patch_decoder.state_dict(),
                    "curve_encoder": curve_encoder.state_dict(),
                    "curve_decoder": curve_decoder.state_dict(),
                    "patch_optimizer": patch_optimizer.state_dict(),
                    "curve_optimizer": curve_optimizer.state_dict(),
                },
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )
            print(f"Saved checkpoint for epoch {epoch+1}")


def eval_pipeline(args, config):
    """
    Complete evaluation pipeline with data loading
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== Load test data =====
    voxel_dim = config.VOXEL_DIM if hasattr(config, "VOXEL_DIM") else 64
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

    # ===== Setup experiment directories =====
    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "ckpt")

    # ===== Initialize models =====
    print("Initializing models...")
    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)

    # ===== Load checkpoint =====
    if not hasattr(args, "checkpoint_path") or not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified in args for evaluation")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    patch_encoder.load_state_dict(checkpoint["patch_encoder"])
    patch_decoder.load_state_dict(checkpoint["patch_decoder"])
    curve_encoder.load_state_dict(checkpoint["curve_encoder"])
    curve_decoder.load_state_dict(checkpoint["curve_decoder"])

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if "val_loss" in checkpoint:
        print(f"  Validation loss at checkpoint: {checkpoint['val_loss']:.6f}")

    # ===== Set to evaluation mode =====
    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()

    # ===== Run evaluation on test set =====
    print("\nRunning evaluation on test set...")
    print("=" * 60)

    patch_metrics_all = []
    curve_metrics_all = []

    with torch.no_grad():
        for batch_idx, data_item in enumerate(tqdm(test_data, desc="Evaluating")):
            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            # Patch evaluation
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

            # Curve evaluation
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

    # ===== Compute average metrics =====
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

    # ===== Print results =====
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    if avg_patch_metrics:
        print("\n📊 PATCH METRICS:")
        print(
            f"  • Reconstruction Error (Points):  {avg_patch_metrics['recon_error']:.6f}"
        )
        print(
            f"  • Reconstruction Error (Normals): {avg_patch_metrics['recon_error_normals']:.6f}"
        )
        print(
            f"  • Label Accuracy:                 {avg_patch_metrics['label_accuracy']:.4f} ({avg_patch_metrics['label_accuracy']*100:.2f}%)"
        )
        print(
            f"  • U-Closed Accuracy:              {avg_patch_metrics['u_closed_accuracy']:.4f} ({avg_patch_metrics['u_closed_accuracy']*100:.2f}%)"
        )
        print(
            f"  • V-Closed Accuracy:              {avg_patch_metrics['v_closed_accuracy']:.4f} ({avg_patch_metrics['v_closed_accuracy']*100:.2f}%)"
        )
        print(
            f"  • Topology Accuracy (Overall):    {avg_patch_metrics['topology_accuracy']:.4f} ({avg_patch_metrics['topology_accuracy']*100:.2f}%)"
        )
        print(
            f"  • Validity Accuracy:              {avg_patch_metrics['validity_accuracy']:.4f} ({avg_patch_metrics['validity_accuracy']*100:.2f}%)"
        )

    if avg_curve_metrics:
        print("\n📈 CURVE METRICS:")
        print(
            f"  • Reconstruction Error (Points):  {avg_curve_metrics['recon_error']:.6f}"
        )
        print(
            f"  • Reconstruction Error (Endpoints):{avg_curve_metrics['recon_error_endpoints']:.6f}"
        )
        print(
            f"    - Start Point Error:            {avg_curve_metrics['start_point_error']:.6f}"
        )
        print(
            f"    - End Point Error:              {avg_curve_metrics['end_point_error']:.6f}"
        )
        print(
            f"  • Label Accuracy:                 {avg_curve_metrics['label_accuracy']:.4f} ({avg_curve_metrics['label_accuracy']*100:.2f}%)"
        )
        print(
            f"  • Closed Accuracy:                {avg_curve_metrics['closed_accuracy']:.4f} ({avg_curve_metrics['closed_accuracy']*100:.2f}%)"
        )
        print(
            f"  • Validity Accuracy:              {avg_curve_metrics['validity_accuracy']:.4f} ({avg_curve_metrics['validity_accuracy']*100:.2f}%)"
        )

    print("=" * 60)

    # ===== Save results =====
    test_metrics = {"patch": avg_patch_metrics, "curve": avg_curve_metrics}

    if hasattr(args, "experiment_name") and args.experiment_name:
        results_path = os.path.join(args.checkpoint_dir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")

            if avg_patch_metrics:
                f.write("PATCH METRICS:\n")
                for key, value in avg_patch_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")

            if avg_curve_metrics:
                f.write("CURVE METRICS:\n")
                for key, value in avg_curve_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")

        print(f"\n✓ Results saved to: {results_path}")

    return test_metrics


def val_pipeline(
    patch_encoder,
    patch_decoder,
    curve_encoder,
    curve_decoder,
    val_dataloader,
    device,
    config,
):
    """
    Validation pipeline during training
    """
    # Set to evaluation mode
    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()

    patch_metrics_all = []
    curve_metrics_all = []

    with torch.no_grad():
        for batch_idx, data_item in enumerate(val_dataloader):
            processed_curves, processed_patches = process_batch_data(
                data_item, config, device
            )

            # Patch validation
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

            # Curve validation
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

    # Compute average metrics
    val_metrics = {}

    if patch_metrics_all:
        avg_patch_metrics = {}
        for key in patch_metrics_all[0].keys():
            avg_patch_metrics[key] = sum(m[key] for m in patch_metrics_all) / len(
                patch_metrics_all
            )
        val_metrics["patch"] = avg_patch_metrics

    if curve_metrics_all:
        avg_curve_metrics = {}
        for key in curve_metrics_all[0].keys():
            avg_curve_metrics[key] = sum(m[key] for m in curve_metrics_all) / len(
                curve_metrics_all
            )
        val_metrics["curve"] = avg_curve_metrics

    # Restore training mode
    patch_encoder.train()
    patch_decoder.train()
    curve_encoder.train()
    curve_decoder.train()

    return val_metrics


if __name__ == "__main__":
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Parse arguments
    parser = argparse.ArgumentParser(
        "Training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # Get number of GPUs
    num_of_gpus = torch.cuda.device_count()
    print(f"Utilize {num_of_gpus} GPU(s)")

    # Get config
    config = Config()

    # Simple if-else: eval or train
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
