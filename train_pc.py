import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import logging
from datetime import datetime
import wandb
from torch.cuda.amp import autocast, GradScaler

from data_loader_optimized import *

class Config:
    D_MODEL = 512
    NHEAD = 8
    DROPOUT = 0.05
    CURVE_LATENT_DIM = 256
    PATCH_LATENT_DIM = 384
    PATCH_NUM_POINTS = 400
    MAX_CURVES = 100
    MAX_PATCHES = 50 
    HN_PE_DIM = 64
    HN_MLP_DIM = 128
    CURVE_NUM_CLASSES = 4
    PATCH_NUM_CLASSES = 6
    KL_WEIGHT_START = 0.0
    KL_WEIGHT_END = 0.0001
    KL_WARMUP_EPOCHS = 100  
    KL_FREE_BITS = 0.5  
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0 
    EVAL_INTERVAL = 100  
    SAVE_INTERVAL = 100
    USE_AMP = False
    GRADIENT_ACCUMULATION_STEPS = 1
    LOG_INTERVAL = 50000
    RECON_WEIGHT = 1.0
    ENDPOINT_WEIGHT = 0.5
    TOPOLOGY_WEIGHT = 0.5
    LABEL_WEIGHT = 0.5
    VALIDITY_WEIGHT = 0.5
    SCALE_FEATURE_DIM = 16
    CENTER_FEATURE_DIM = 16
    ENDPOINT_FEATURE_DIM = 48
    CLOSED_FEATURE_DIM = 12
    LABEL_FEATURE_DIM = 16
    TOPO_FEATURE_DIM = 20
    USE_SMOOTHNESS_LOSS = False       
    CURVE_SMOOTHNESS_TAU = 0.01
    PATCH_SMOOTHNESS_TAU = 0.02
    PATCH_NORMAL_TAU = 0.1
    SMOOTHNESS_WEIGHT = 0.1
    LOGVAR_MIN = -10.0
    LOGVAR_MAX = 10.0
    SCALE_MIN = 1e-4
    SCALE_MAX = 1e2
    LOG_SCALE_MIN = -10.0
    LOG_SCALE_MAX = 10.0

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_tanh_output=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_tanh_output = use_tanh_output
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norms = nn.ModuleList(nn.LayerNorm(k) for k in h)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = F.gelu(x)
            elif self.use_tanh_output:
                x = torch.tanh(x)
        return x

flag_hidden_layer = True
hn_hidden_dim = 128
hn_pe_dim = 64

class MLP_hn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_dim_fea):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        h_plus = [hidden_dim + 1] * (num_layers - 1)
        self.layers_dims = list(zip([input_dim + 1] + h_plus, h + [output_dim]))
        self.layers_size = [a * b for a, b in self.layers_dims]
        if not flag_hidden_layer:
            self.layer = nn.Linear(input_dim_fea, sum(self.layers_size))
        else:
            self.layer1 = nn.Linear(input_dim_fea, hn_hidden_dim)
            self.norm1 = nn.LayerNorm(hn_hidden_dim)
            self.layer2 = nn.Linear(hn_hidden_dim, sum(self.layers_size))
        
    def forward(self, x, feature):
        if not flag_hidden_layer:
            net_par = self.layer(feature)
        else:
            net_par = self.layer1(feature)
            net_par = self.norm1(net_par)
            net_par = F.gelu(net_par)
            net_par = self.layer2(net_par)
        net_par = torch.clamp(net_par, -5, 5)
        net_par = net_par / math.sqrt(hn_pe_dim)
        net_par_layers = torch.split(net_par, self.layers_size, dim=-1)
        for i in range(len(self.layers_size)):
            layer_par = net_par_layers[i].view(net_par.shape[0], net_par.shape[1], net_par.shape[2], self.layers_dims[i][0], self.layers_dims[i][1])
            x = torch.einsum('...ij,...jk->...ik', x, layer_par[..., :-1, :]) + layer_par[..., -1:, :]
            if i < self.num_layers - 1:
                x = F.gelu(x)
        return x

def get_args_parser():
    parser = argparse.ArgumentParser('VAE CAD Reconstruction', add_help=False)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--parsenet', action='store_true')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--quicktest', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_voxel_dim', default=128, type=int)
    parser.add_argument('--points_per_patch_dim', default=20, type=int)
    parser.add_argument('--input_feature_type', default='global', type=str)
    parser.add_argument('--input_normal_signals', action='store_true')
    parser.add_argument('--rotation_augment', action='store_true')
    parser.add_argument('--random_angle', action='store_true')
    parser.add_argument('--num_angles', type=int)
    parser.add_argument('--noise', default=0, type=int)
    parser.add_argument('--patch_grid', action='store_true')
    parser.add_argument('--patch_uv', action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--backbone_feature_encode', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--max_training_iterations', default=250001, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--gpu', default="0,1,2", type=str)
    parser.add_argument('--use_smoothness_loss', action='store_true')
    parser.add_argument('--curve_smoothness_tau', default=0.01, type=float)
    parser.add_argument('--patch_smoothness_tau', default=0.02, type=float)
    parser.add_argument('--smoothness_weight', default=0.1, type=float)
    parser.add_argument('--extra_single_chamfer', action='store_true')
    return parser

points_per_curve = 34

class Curve1DEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, output_dim)
    
    def forward(self, points):
        B, N, P, C = points.shape
        # Reshape to (B*N, C, P) for Conv1d
        x = points.view(B*N, P, C).transpose(1, 2)  # (B*N, C, P)
        x = self.encoder(x)  # (B*N, 256, P')
        x = self.pool(x).squeeze(-1)  # (B*N, 256)
        return self.proj(x).view(B, N, -1)  # (B, N, output_dim)

class Patch2DEncoder(nn.Module):
    def __init__(self, input_channels=6, output_dim=384, dropout=0.1, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 20->10
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 10->5
            nn.GroupNorm(32, 256),
            nn.GELU(),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 5->5
            nn.GroupNorm(32, 384),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, output_dim)
    
    def forward(self, patch_data):
        B, N, HW, C = patch_data.shape
        H = W = self.grid_size
        
        x = patch_data.view(B*N, H, W, C).permute(0, 3, 1, 2)  # (B*N, 6, 20, 20)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.proj(x).view(B, N, -1)

class CurveEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.geom_encoder = Curve1DEncoder(input_channels=3, output_dim=256, dropout=config.DROPOUT)
        self.scale_encoder = nn.Sequential(nn.Linear(1, config.SCALE_FEATURE_DIM), nn.LayerNorm(config.SCALE_FEATURE_DIM), nn.GELU())
        self.center_encoder = nn.Sequential(nn.Linear(3, config.CENTER_FEATURE_DIM), nn.LayerNorm(config.CENTER_FEATURE_DIM), nn.GELU())
        self.endpoint_encoder = nn.Sequential(nn.Linear(6, config.ENDPOINT_FEATURE_DIM), nn.LayerNorm(config.ENDPOINT_FEATURE_DIM), nn.GELU())
        self.closed_encoder = nn.Sequential(nn.Linear(1, config.CLOSED_FEATURE_DIM), nn.GELU())
        self.label_embedding = nn.Embedding(config.CURVE_NUM_CLASSES, config.LABEL_FEATURE_DIM)
        total_feat_dim = 256 + config.SCALE_FEATURE_DIM + config.CENTER_FEATURE_DIM + config.ENDPOINT_FEATURE_DIM + config.CLOSED_FEATURE_DIM + config.LABEL_FEATURE_DIM
        self.fusion = nn.Sequential(nn.Linear(total_feat_dim, config.D_MODEL), nn.LayerNorm(config.D_MODEL), nn.GELU())
        self.to_mean = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)
        self.to_logvar = nn.Linear(config.D_MODEL, config.CURVE_LATENT_DIM)

    def forward(self, curve_points, endpoints, is_closed, labels, scale, center, mask):
        B, N = curve_points.shape[:2]
        endpoints_flat = endpoints.reshape(B, N, 6)
        log_scale = torch.log(torch.clamp(scale, min=self.config.SCALE_MIN, max=self.config.SCALE_MAX))
        log_scale = torch.clamp(log_scale, self.config.LOG_SCALE_MIN, self.config.LOG_SCALE_MAX)
        scale_feat = self.scale_encoder(log_scale.unsqueeze(-1))
        center_clamped = torch.clamp(center, -100, 100)
        center_feat = self.center_encoder(center_clamped)
        geom_feat = self.geom_encoder(curve_points)
        endpoint_feat = self.endpoint_encoder(endpoints_flat)
        closed_feat = self.closed_encoder(is_closed.unsqueeze(-1).float())
        label_feat = self.label_embedding(labels)
        all_feat = torch.cat([geom_feat, scale_feat, center_feat, endpoint_feat, closed_feat, label_feat], dim=-1)
        tokens = self.fusion(all_feat)
        tokens = tokens * mask.unsqueeze(-1).float()
        mean = self.to_mean(tokens)
        logvar = self.to_logvar(tokens)
        logvar = torch.clamp(logvar, min=self.config.LOGVAR_MIN, max=self.config.LOGVAR_MAX)
        return mean, logvar

class CurveDecoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.start_point_embed = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 3, 3, use_tanh_output=True)
        self.curve_pe = self._init_curve_pe(points_per_curve, config.HN_PE_DIM)
        self.curve_shape_embed = MLP_hn(input_dim=config.HN_PE_DIM, hidden_dim=config.HN_MLP_DIM, output_dim=3, num_layers=3, input_dim_fea=config.CURVE_LATENT_DIM)
        self.endpoints_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 6, 3, use_tanh_output=True)
        self.closed_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 1, 3)
        self.label_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, config.CURVE_NUM_CLASSES, 3)
        self.validity_head = MLP(config.CURVE_LATENT_DIM, config.CURVE_LATENT_DIM, 1, 3)

    def _init_curve_pe(self, num_points, pe_dim):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        coord = torch.arange(num_points, dtype=torch.float32, device=device) / (num_points - 1)
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
        start_point = self.start_point_embed(z) * 0.5
        curve_pe = self.curve_pe.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        shape_offset = self.curve_shape_embed(curve_pe.unsqueeze(0), z.unsqueeze(0))
        shape_offset = shape_offset.squeeze(0)
        shape_offset = torch.clamp(shape_offset, -1, 1)
        points = start_point.unsqueeze(2) + shape_offset
        points = torch.clamp(points, -0.6, 0.6)
        endpoints = self.endpoints_head(z).view(B, N, 2, 3) * 0.5
        closed_logits = torch.clamp(self.closed_head(z).squeeze(-1), -10, 10)
        label_logits = torch.clamp(self.label_head(z), -10, 10)
        validity_logits = torch.clamp(self.validity_head(z).squeeze(-1), -10, 10)
        return {"points": points, "endpoints": endpoints, "closed_logits": closed_logits, "label_logits": label_logits, "validity_logits": validity_logits}

class PatchEncoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.geom_encoder = Patch2DEncoder(input_channels=6, output_dim=384, dropout=config.DROPOUT)
        self.scale_encoder = nn.Sequential(nn.Linear(1, config.SCALE_FEATURE_DIM), nn.LayerNorm(config.SCALE_FEATURE_DIM), nn.GELU())
        self.center_encoder = nn.Sequential(nn.Linear(3, config.CENTER_FEATURE_DIM), nn.LayerNorm(config.CENTER_FEATURE_DIM), nn.GELU())
        self.topo_encoder = nn.Sequential(nn.Linear(2, config.TOPO_FEATURE_DIM), nn.GELU())
        self.label_embedding = nn.Embedding(config.PATCH_NUM_CLASSES, config.LABEL_FEATURE_DIM)
        total_feat_dim = 384 + config.SCALE_FEATURE_DIM + config.CENTER_FEATURE_DIM + config.TOPO_FEATURE_DIM + config.LABEL_FEATURE_DIM
        self.fusion = nn.Sequential(nn.Linear(total_feat_dim, config.D_MODEL), nn.LayerNorm(config.D_MODEL), nn.GELU())
        self.to_mean = nn.Linear(config.D_MODEL, config.PATCH_LATENT_DIM)
        self.to_logvar = nn.Linear(config.D_MODEL, config.PATCH_LATENT_DIM)

    def forward(self, patch_points, patch_normals, u_closed, v_closed, labels, scale, center, mask):
        B, N = patch_points.shape[:2]
        log_scale = torch.log(torch.clamp(scale, min=self.config.SCALE_MIN, max=self.config.SCALE_MAX))
        log_scale = torch.clamp(log_scale, self.config.LOG_SCALE_MIN, self.config.LOG_SCALE_MAX)
        scale_feat = self.scale_encoder(log_scale.unsqueeze(-1))
        center_clamped = torch.clamp(center, -100, 100)
        center_feat = self.center_encoder(center_clamped)
        geom_input = torch.cat([patch_points, patch_normals], dim=-1)
        geom_feat = self.geom_encoder(geom_input)
        topo_input = torch.stack([u_closed, v_closed], dim=-1).float()
        topo_feat = self.topo_encoder(topo_input)
        label_feat = self.label_embedding(labels)
        all_feat = torch.cat([geom_feat, scale_feat, center_feat, topo_feat, label_feat], dim=-1)
        tokens = self.fusion(all_feat)
        tokens = tokens * mask.unsqueeze(-1).float()
        mean = self.to_mean(tokens)
        logvar = self.to_logvar(tokens)
        logvar = torch.clamp(logvar, min=self.config.LOGVAR_MIN, max=self.config.LOGVAR_MAX)
        return mean, logvar

class PatchDecoder(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        self.patch_dim = int(math.sqrt(config.PATCH_NUM_POINTS))
        self.startpoint_embed = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 3, 3, use_tanh_output=True)
        self.patch_pe_u, self.patch_pe_v = self._init_patch_pe(self.patch_dim, config.HN_PE_DIM)
        self.patch_shape_embed = MLP_hn(input_dim=config.HN_PE_DIM * 2, hidden_dim=config.HN_MLP_DIM, output_dim=6, num_layers=3, input_dim_fea=config.PATCH_LATENT_DIM)
        self.u_closed_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 1, 3)
        self.v_closed_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, 1, 3)
        self.label_head = MLP(config.PATCH_LATENT_DIM, config.PATCH_LATENT_DIM, config.PATCH_NUM_CLASSES, 3)
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
        return nn.Parameter(pe, requires_grad=False), nn.Parameter(pe, requires_grad=False)

    def forward(self, z):
        B, N = z.shape[:2]
        startpoint = self.startpoint_embed(z) * 0.5
        patch_pe = []
        for i in range(self.patch_dim):
            for j in range(self.patch_dim):
                pe_ij = torch.cat([self.patch_pe_u[i], self.patch_pe_v[j]], dim=0)
                patch_pe.append(pe_ij)
        patch_pe = torch.stack(patch_pe, dim=0).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        output = self.patch_shape_embed(patch_pe.unsqueeze(0), z.unsqueeze(0))
        output = output.squeeze(0)
        output = torch.clamp(output, -2, 2)
        shape_offset = output[..., :3]
        normals_raw = output[..., 3:]
        points = startpoint.unsqueeze(2) + shape_offset
        points = torch.clamp(points, -0.6, 0.6)
        normals = F.normalize(normals_raw, dim=-1, eps=1e-6)
        u_closed_logits = torch.clamp(self.u_closed_head(z).squeeze(-1), -10, 10)
        v_closed_logits = torch.clamp(self.v_closed_head(z).squeeze(-1), -10, 10)
        label_logits = torch.clamp(self.label_head(z), -10, 10)
        validity_logits = torch.clamp(self.validity_head(z).squeeze(-1), -10, 10)
        return {"points": points, "normals": normals, "u_closed_logits": u_closed_logits, "v_closed_logits": v_closed_logits, "label_logits": label_logits, "validity_logits": validity_logits}

def setup_logger(log_dir, rank=0):
    if rank != 0:
        return None
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file)])
    return logging.getLogger(__name__)

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

def compute_curve_smoothness_loss(points, mask, tau=0.01):
    B, N, P, _ = points.shape
    first_deriv = points[:, :, 1:, :] - points[:, :, :-1, :]
    second_deriv = first_deriv[:, :, 1:, :] - first_deriv[:, :, :-1, :]
    second_deriv = torch.clamp(second_deriv, -5, 5)
    curvature = torch.norm(second_deriv, dim=-1)
    penalty = torch.where(curvature < tau, 0.5 * curvature ** 2 / tau, curvature - 0.5 * tau)
    penalty = torch.clamp(penalty, max=5.0)
    mask_expanded = mask.unsqueeze(-1).float()
    loss = (penalty * mask_expanded[:, :, :penalty.shape[2]]).sum() / mask.sum().clamp(min=1)
    return loss

def compute_patch_smoothness_loss(points, normals, mask, position_tau=0.02, normal_tau=0.1):
    B, N, num_points, _ = points.shape
    grid_size = int(math.sqrt(num_points))
    grid_points = points.view(B, N, grid_size, grid_size, 3)
    grid_normals = normals.view(B, N, grid_size, grid_size, 3)
    laplacian = (grid_points[:, :, 2:, 1:-1, :] + grid_points[:, :, :-2, 1:-1, :] + grid_points[:, :, 1:-1, 2:, :] + grid_points[:, :, 1:-1, :-2, :] - 4 * grid_points[:, :, 1:-1, 1:-1, :])
    laplacian = torch.clamp(laplacian, -5, 5)
    laplacian_norm = torch.norm(laplacian, dim=-1)
    position_penalty = torch.where(laplacian_norm < position_tau, 0.5 * laplacian_norm ** 2 / position_tau, laplacian_norm - 0.5 * position_tau)
    position_penalty = torch.clamp(position_penalty, max=5.0)
    mask_expanded = mask.view(B, N, 1, 1).float()
    position_loss = (position_penalty * mask_expanded).sum() / mask.sum().clamp(min=1)
    normal_diff_u = grid_normals[:, :, 1:, :, :] - grid_normals[:, :, :-1, :, :]
    normal_diff_v = grid_normals[:, :, :, 1:, :] - grid_normals[:, :, :, :-1, :]
    normal_change_u = torch.norm(normal_diff_u, dim=-1).clamp(max=2.0)
    normal_change_v = torch.norm(normal_diff_v, dim=-1).clamp(max=2.0)
    penalty_u = torch.where(normal_change_u < normal_tau, 0.5 * normal_change_u ** 2 / normal_tau, normal_change_u - 0.5 * normal_tau)
    penalty_v = torch.where(normal_change_v < normal_tau, 0.5 * normal_change_v ** 2 / normal_tau, normal_change_v - 0.5 * normal_tau)
    penalty_u = torch.clamp(penalty_u, max=5.0)
    penalty_v = torch.clamp(penalty_v, max=5.0)
    normal_loss_u = (penalty_u * mask_expanded[:, :, :penalty_u.shape[2], :]).sum()
    normal_loss_v = (penalty_v * mask_expanded[:, :, :, :penalty_v.shape[3]]).sum()
    normal_loss = (normal_loss_u + normal_loss_v) / mask.sum().clamp(min=1)
    return position_loss + normal_loss

def compute_stable_kl_loss(mean, logvar, mask, config):
    logvar = torch.clamp(logvar, config.LOGVAR_MIN, config.LOGVAR_MAX)
    var = torch.exp(logvar)
    var = torch.clamp(var, min=1e-8, max=1e8)
    kl_per_dim = -0.5 * (1 + torch.log(var + 1e-8) - mean.pow(2) - var)
    free_bits = torch.tensor(config.KL_FREE_BITS, device=kl_per_dim.device)
    kl_per_dim = torch.clamp(kl_per_dim, min=0)
    kl_per_dim = torch.max(kl_per_dim, free_bits)
    kl = kl_per_dim.sum(dim=-1)
    kl = (kl * mask.float()).sum() / mask.sum().clamp(min=1)
    kl = kl / kl_per_dim.shape[-1]
    return kl

def compute_patch_vae_loss(pred, target, mean, logvar, mask, kl_weight, config):
    losses = {}
    kl = compute_stable_kl_loss(mean, logvar, mask, config)
    losses["kl"] = kl
    weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)
    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points
    recon_normals = F.mse_loss(pred["normals"], target["normals"], reduction="none")
    recon_normals = (recon_normals * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_normals"] = recon_normals
    topo_weight = mask.float()
    u_closed_loss = F.binary_cross_entropy_with_logits(pred["u_closed_logits"], target["u_closed"].float(), reduction="none")
    u_closed_loss = (u_closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    v_closed_loss = F.binary_cross_entropy_with_logits(pred["v_closed_logits"], target["v_closed"].float(), reduction="none")
    v_closed_loss = (v_closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["u_closed"] = u_closed_loss
    losses["v_closed"] = v_closed_loss
    label_loss = F.cross_entropy(pred["label_logits"].view(-1, pred["label_logits"].size(-1)), target["labels"].view(-1), reduction="none")
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss
    validity_loss = F.binary_cross_entropy_with_logits(pred["validity_logits"], mask.float(), reduction="mean")
    losses["validity"] = validity_loss
    if config.USE_SMOOTHNESS_LOSS:
        smoothness = compute_patch_smoothness_loss(pred["points"], pred["normals"], mask, position_tau=config.PATCH_SMOOTHNESS_TAU, normal_tau=config.PATCH_NORMAL_TAU)
        losses["smoothness"] = smoothness
    else:
        losses["smoothness"] = torch.tensor(0.0, device=pred["points"].device)
    total_loss = (kl_weight * kl + config.RECON_WEIGHT * (recon_points + recon_normals) + config.TOPOLOGY_WEIGHT * (u_closed_loss + v_closed_loss) + config.LABEL_WEIGHT * label_loss + config.VALIDITY_WEIGHT * validity_loss + config.SMOOTHNESS_WEIGHT * losses["smoothness"])
    total_loss = torch.clamp(total_loss, max=1e6)
    losses["total"] = total_loss
    return losses

def compute_curve_vae_loss(pred, target, mean, logvar, mask, kl_weight, config):
    losses = {}
    kl = compute_stable_kl_loss(mean, logvar, mask, config)
    losses["kl"] = kl
    weight_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)
    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
    recon_points = (recon_points * weight_mask).sum() / mask.sum().clamp(min=1)
    losses["recon_points"] = recon_points
    open_mask = ~target["is_closed"]
    endpoint_mask = (mask & open_mask).unsqueeze(-1).unsqueeze(-1).float()
    pred_endpoints_denorm = pred["endpoints"] * pred_scale + pred_center
    target_endpoints_denorm = target["endpoints"] * pred_scale + pred_center
    recon_endpoints = F.mse_loss(pred_endpoints_denorm, target_endpoints_denorm, reduction="none")
    recon_endpoints = (recon_endpoints * endpoint_mask).sum() / open_mask.sum().clamp(min=1)
    losses["recon_endpoints"] = recon_endpoints
    topo_weight = mask.float()
    closed_loss = F.binary_cross_entropy_with_logits(pred["closed_logits"], target["is_closed"].float(), reduction="none")
    closed_loss = (closed_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["closed"] = closed_loss
    label_loss = F.cross_entropy(pred["label_logits"].view(-1, pred["label_logits"].size(-1)), target["labels"].view(-1), reduction="none")
    label_loss = label_loss.view(pred["label_logits"].shape[:-1])
    label_loss = (label_loss * topo_weight).sum() / mask.sum().clamp(min=1)
    losses["label"] = label_loss
    validity_loss = F.binary_cross_entropy_with_logits(pred["validity_logits"], mask.float(), reduction="mean")
    losses["validity"] = validity_loss
    if config.USE_SMOOTHNESS_LOSS:
        smoothness = compute_curve_smoothness_loss(pred["points"], mask, tau=config.CURVE_SMOOTHNESS_TAU)
        losses["smoothness"] = smoothness
    else:
        losses["smoothness"] = torch.tensor(0.0, device=pred["points"].device)
    total_loss = (kl_weight * kl + config.RECON_WEIGHT * (recon_points + recon_endpoints) + config.TOPOLOGY_WEIGHT * closed_loss + config.LABEL_WEIGHT * label_loss + config.VALIDITY_WEIGHT * validity_loss + config.SMOOTHNESS_WEIGHT * losses["smoothness"])
    total_loss = torch.clamp(total_loss, max=1e6)
    losses["total"] = total_loss
    return losses

def move_to_device(data_dict, device):
    if data_dict is None:
        return None
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in data_dict.items()}

def compute_patch_metrics(pred, target, mask):
    metrics = {}
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)
    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_error_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
    recon_error_points = (recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()
    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    metrics["label_accuracy"] = (correct_labels.sum().float() / mask.sum().float()).item()
    return metrics

def compute_curve_metrics(pred, target, mask):
    metrics = {}
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)
    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_error_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
    recon_error_points = (recon_error_points * mask.unsqueeze(-1).unsqueeze(-1)).sum() / mask.sum().clamp(min=1)
    metrics["recon_error"] = recon_error_points.item()
    pred_labels = torch.argmax(pred["label_logits"], dim=-1)
    correct_labels = (pred_labels == target["labels"]) & mask
    metrics["label_accuracy"] = (correct_labels.sum().float() / mask.sum().float()).item()
    return metrics

class LossAccumulator:
    def __init__(self):
        self.losses = {}
        self.count = 0
    def add(self, loss_dict):
        for key, val in loss_dict.items():
            if torch.isnan(val) or torch.isinf(val):
                continue
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

def val_pipeline(patch_encoder, patch_decoder, curve_encoder, curve_decoder, 
                val_dataloader, val_sampler, epoch, device, config, rank=0):
    """
    """
    if val_sampler is not None:
        val_sampler.set_epoch(epoch)
    
    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()
    
    patch_recon_errors = []
    curve_recon_errors = []
    
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation", leave=False) if rank == 0 else val_dataloader
        for data_item in pbar:
            processed_curves, processed_patches = data_item
            processed_curves = move_to_device(processed_curves, device)
            processed_patches = move_to_device(processed_patches, device)
            
            if processed_patches is not None:
                try:
                    mean_p, _ = patch_encoder(
                        processed_patches["patch_points"], 
                        processed_patches["patch_normals"], 
                        processed_patches["u_closed"], 
                        processed_patches["v_closed"], 
                        processed_patches["labels"], 
                        processed_patches["scale"], 
                        processed_patches["center"], 
                        processed_patches["mask"]
                    )
                    pred_patch = patch_decoder(mean_p)
                    
                    pred_scale = processed_patches["scale"].unsqueeze(-1).unsqueeze(-1)
                    pred_center = processed_patches["center"].unsqueeze(-2)
                    pred_points_denorm = pred_patch["points"] * pred_scale + pred_center
                    target_points_denorm = processed_patches["patch_points"] * pred_scale + pred_center
                    
                    recon = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
                    recon = (recon * processed_patches["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_patches["mask"].sum().clamp(min=1)
                    
                    if not torch.isnan(recon):
                        patch_recon_errors.append(recon)
                except:
                    continue
            
            if processed_curves is not None:
                try:
                    mean_c, _ = curve_encoder(
                        processed_curves["curve_points"], 
                        processed_curves["endpoints"], 
                        processed_curves["is_closed"], 
                        processed_curves["labels"], 
                        processed_curves["scale"], 
                        processed_curves["center"], 
                        processed_curves["mask"]
                    )
                    pred_curve = curve_decoder(mean_c)
                    
                    pred_scale = processed_curves["scale"].unsqueeze(-1).unsqueeze(-1)
                    pred_center = processed_curves["center"].unsqueeze(-2)
                    pred_points_denorm = pred_curve["points"] * pred_scale + pred_center
                    target_points_denorm = processed_curves["curve_points"] * pred_scale + pred_center
                    
                    recon = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
                    recon = (recon * processed_curves["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_curves["mask"].sum().clamp(min=1)
                    
                    if not torch.isnan(recon):
                        curve_recon_errors.append(recon)
                except:
                    continue
    
    val_metrics = {}
    if patch_recon_errors:
        val_metrics["patch"] = {"recon_error": torch.stack(patch_recon_errors).mean().item()}
    if curve_recon_errors:
        val_metrics["curve"] = {"recon_error": torch.stack(curve_recon_errors).mean().item()}
    
    patch_encoder.train()
    patch_decoder.train()
    curve_encoder.train()
    curve_decoder.train()
    
    return val_metrics

def train_pipeline(rank, num_gpus, args, config):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23257", world_size=num_gpus, rank=rank)
    if num_gpus > 1:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    experiment_dir = os.path.join("experiments", args.experiment_name)
    args.checkpoint_dir = os.path.join(experiment_dir, "ckpt")
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        logger = setup_logger(experiment_dir)
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Training on {num_gpus} GPU(s)")
        logger.info(f"优化策略: 数据预处理前置到Dataset，利用DataLoader多进程并行")
        
        wandb.init(
            project="vae-cad-reconstruction",
            name=args.experiment_name,
            config={
                "d_model": config.D_MODEL,
                "curve_latent_dim": config.CURVE_LATENT_DIM,
                "patch_latent_dim": config.PATCH_LATENT_DIM,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": args.batch_size,
            }
        )
    
    if args.quicktest:
        train_data, distribute_sampler = train_data_loader_clean(
            args.batch_size, 
            data_folder="data/train_small", 
            rotation_augmentation=args.rotation_augment, 
            random_angle=args.random_angle, 
            flag_noise=args.noise, 
            flag_grid=args.patch_grid, 
            num_angle=args.num_angles, 
            dim_grid=args.points_per_patch_dim, 
            num_workers=4,
            rank=rank, 
            world_size=num_gpus
        )
        val_data, val_sampler = train_data_loader_clean(
            args.batch_size,
            data_folder="data/train_small",
            rotation_augmentation=False,
            random_angle=False,
            flag_noise=0,
            flag_grid=args.patch_grid,
            num_angle=4,
            dim_grid=args.points_per_patch_dim,
            num_workers=4,
            rank=rank,
            world_size=num_gpus
        )
    else:
        train_folder = "data/partial/train" if args.partial else "data/default/train"
        train_data, distribute_sampler = train_data_loader_clean(
            args.batch_size, 
            data_folder=train_folder, 
            rotation_augmentation=args.rotation_augment, 
            random_angle=args.random_angle, 
            flag_noise=args.noise, 
            flag_grid=args.patch_grid, 
            num_angle=args.num_angles, 
            dim_grid=args.points_per_patch_dim, 
            num_workers=4,
            rank=rank, 
            world_size=num_gpus
        )
        val_folder = "data/partial/val" if args.partial else "data/default/val"
        val_data, val_sampler = train_data_loader_clean(
            args.batch_size,
            data_folder=val_folder,
            rotation_augmentation=False,
            random_angle=False,
            flag_noise=0,
            flag_grid=args.patch_grid,
            num_angle=4,
            dim_grid=args.points_per_patch_dim,
            num_workers=4,
            rank=rank,
            world_size=num_gpus
        )
    
    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)
    
    if num_gpus > 1:
        patch_encoder = nn.parallel.DistributedDataParallel(patch_encoder, device_ids=[rank])
        patch_decoder = nn.parallel.DistributedDataParallel(patch_decoder, device_ids=[rank])
        curve_encoder = nn.parallel.DistributedDataParallel(curve_encoder, device_ids=[rank])
        curve_decoder = nn.parallel.DistributedDataParallel(curve_decoder, device_ids=[rank])
    
    patch_params = list(patch_encoder.parameters()) + list(patch_decoder.parameters())
    curve_params = list(curve_encoder.parameters()) + list(curve_decoder.parameters())
    patch_optimizer = torch.optim.AdamW(patch_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    curve_optimizer = torch.optim.AdamW(curve_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    patch_scaler = GradScaler() if config.USE_AMP else None
    curve_scaler = GradScaler() if config.USE_AMP else None
    kl_scheduler = KLScheduler(config.KL_WEIGHT_START, config.KL_WEIGHT_END, config.KL_WARMUP_EPOCHS)
    
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
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
    
    best_val_loss = float("inf")
    cur_epochs = start_epoch
    
    for epoch in range(start_epoch, args.max_training_iterations):
        current_kl_weight = kl_scheduler.get_weight(epoch)
        if rank == 0:
            logger.info(f"Epoch {epoch+1} - KL Weight: {current_kl_weight:.6f}")
        
        if distribute_sampler is not None:
            distribute_sampler.set_epoch(cur_epochs)
        
        patch_encoder.train()
        patch_decoder.train()
        curve_encoder.train()
        curve_decoder.train()
        
        patch_loss_acc = LossAccumulator()
        curve_loss_acc = LossAccumulator()
        
        data_loader_iterator = iter(train_data)
        pbar = tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}", leave=True) if rank == 0 else range(len(train_data))
        
        for batch_idx in pbar:
            try:
                data_item = next(data_loader_iterator)
            except StopIteration:
                data_loader_iterator = iter(train_data)
                data_item = next(data_loader_iterator)
                cur_epochs += 1
                if distribute_sampler is not None:
                    distribute_sampler.set_epoch(cur_epochs)
            
            processed_curves, processed_patches = data_item
            processed_curves = move_to_device(processed_curves, device)
            processed_patches = move_to_device(processed_patches, device)
            
            if processed_patches is not None:
                try:
                    with autocast() if config.USE_AMP else torch.cuda.amp.autocast(enabled=False):
                        mean_p, logvar_p = patch_encoder(
                            processed_patches["patch_points"], 
                            processed_patches["patch_normals"], 
                            processed_patches["u_closed"], 
                            processed_patches["v_closed"], 
                            processed_patches["labels"], 
                            processed_patches["scale"], 
                            processed_patches["center"], 
                            processed_patches["mask"]
                        )
                        std_p = torch.exp(0.5 * logvar_p).clamp(min=1e-8, max=10)
                        eps_p = torch.randn_like(std_p)
                        z_p = mean_p + eps_p * std_p
                        pred_patch = patch_decoder(z_p)
                        target_patch = {
                            "points": processed_patches["patch_points"], 
                            "normals": processed_patches["patch_normals"], 
                            "u_closed": processed_patches["u_closed"], 
                            "v_closed": processed_patches["v_closed"], 
                            "labels": processed_patches["labels"], 
                            "scale": processed_patches["scale"], 
                            "center": processed_patches["center"]
                        }
                        patch_losses = compute_patch_vae_loss(pred_patch, target_patch, mean_p, logvar_p, processed_patches["mask"], current_kl_weight, config)
                    
                    loss = patch_losses["total"] / config.GRADIENT_ACCUMULATION_STEPS
                    if torch.isnan(loss) or torch.isinf(loss):
                        if rank == 0:
                            logger.warning("NaN/Inf in patch loss! Skipping.")
                    else:
                        if config.USE_AMP:
                            patch_scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        patch_loss_acc.add(patch_losses)
                except RuntimeError as e:
                    if rank == 0:
                        logger.error(f"Runtime error in patch: {e}")
            
            if processed_curves is not None:
                try:
                    with autocast() if config.USE_AMP else torch.cuda.amp.autocast(enabled=False):
                        mean_c, logvar_c = curve_encoder(
                            processed_curves["curve_points"], 
                            processed_curves["endpoints"], 
                            processed_curves["is_closed"], 
                            processed_curves["labels"], 
                            processed_curves["scale"], 
                            processed_curves["center"], 
                            processed_curves["mask"]
                        )
                        std_c = torch.exp(0.5 * logvar_c).clamp(min=1e-8, max=10)
                        eps_c = torch.randn_like(std_c)
                        z_c = mean_c + eps_c * std_c
                        pred_curve = curve_decoder(z_c)
                        target_curve = {
                            "points": processed_curves["curve_points"], 
                            "endpoints": processed_curves["endpoints"], 
                            "is_closed": processed_curves["is_closed"], 
                            "labels": processed_curves["labels"], 
                            "scale": processed_curves["scale"], 
                            "center": processed_curves["center"]
                        }
                        curve_losses = compute_curve_vae_loss(pred_curve, target_curve, mean_c, logvar_c, processed_curves["mask"], current_kl_weight, config)
                    
                    loss = curve_losses["total"] / config.GRADIENT_ACCUMULATION_STEPS
                    if torch.isnan(loss) or torch.isinf(loss):
                        if rank == 0:
                            logger.warning("NaN/Inf in curve loss! Skipping.")
                    else:
                        if config.USE_AMP:
                            curve_scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        curve_loss_acc.add(curve_losses)
                except RuntimeError as e:
                    if rank == 0:
                        logger.error(f"Runtime error in curve: {e}")
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                if config.USE_AMP:
                    patch_scaler.unscale_(patch_optimizer)
                    torch.nn.utils.clip_grad_norm_(patch_params, max_norm=config.GRAD_CLIP_NORM)
                    patch_scaler.step(patch_optimizer)
                    patch_scaler.update()
                    
                    curve_scaler.unscale_(curve_optimizer)
                    torch.nn.utils.clip_grad_norm_(curve_params, max_norm=config.GRAD_CLIP_NORM)
                    curve_scaler.step(curve_optimizer)
                    curve_scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(patch_params, max_norm=config.GRAD_CLIP_NORM)
                    patch_optimizer.step()
                    torch.nn.utils.clip_grad_norm_(curve_params, max_norm=config.GRAD_CLIP_NORM)
                    curve_optimizer.step()
                
                patch_optimizer.zero_grad()
                curve_optimizer.zero_grad()
            
            if rank == 0 and (batch_idx + 1) % 100 == 0:
                patch_avgs = patch_loss_acc.get_averages()
                curve_avgs = curve_loss_acc.get_averages()
                postfix = {}
                if "total" in patch_avgs:
                    postfix["P"] = f"{patch_avgs['total']:.4f}"
                if "total" in curve_avgs:
                    postfix["C"] = f"{curve_avgs['total']:.4f}"
                if postfix and isinstance(pbar, tqdm):
                    pbar.set_postfix(postfix)
        
        cur_epochs += 1
        avg_patch_losses = patch_loss_acc.get_averages()
        avg_curve_losses = curve_loss_acc.get_averages()
        
        if rank == 0:
            log_dict = {"epoch": epoch + 1, "kl_weight": current_kl_weight}
            if avg_patch_losses:
                log_dict.update({f"train/patch_{k}": v for k, v in avg_patch_losses.items()})
            if avg_curve_losses:
                log_dict.update({f"train/curve_{k}": v for k, v in avg_curve_losses.items()})
            wandb.log(log_dict)
            logger.info(f"Epoch {epoch+1} - P:{avg_patch_losses.get('total', 0):.4f} C:{avg_curve_losses.get('total', 0):.4f}")
        
        # Validation
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            val_metrics = val_pipeline(
                patch_encoder, patch_decoder, 
                curve_encoder, curve_decoder, 
                val_data, val_sampler, 
                epoch, device, config, rank
            )
            
            if rank == 0 and val_metrics:
                val_log_dict = {"epoch": epoch + 1}
                if "patch" in val_metrics:
                    val_log_dict["val/patch_recon"] = val_metrics['patch']['recon_error']
                if "curve" in val_metrics:
                    val_log_dict["val/curve_recon"] = val_metrics['curve']['recon_error']
                wandb.log(val_log_dict)
                
                val_loss = val_metrics.get("patch", {}).get("recon_error", 0) + val_metrics.get("curve", {}).get("recon_error", 0)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "epoch": epoch, 
                        "patch_encoder": patch_encoder.state_dict(), 
                        "patch_decoder": patch_decoder.state_dict(), 
                        "curve_encoder": curve_encoder.state_dict(), 
                        "curve_decoder": curve_decoder.state_dict(), 
                        "patch_optimizer": patch_optimizer.state_dict(), 
                        "curve_optimizer": curve_optimizer.state_dict(),
                        "val_loss": val_loss
                    }, os.path.join(args.checkpoint_dir, "best_model.pth"))
                    logger.info(f"Best model saved with val_loss: {val_loss:.6f}")
        
        if rank == 0 and (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save({
                "epoch": epoch, 
                "patch_encoder": patch_encoder.state_dict(), 
                "patch_decoder": patch_decoder.state_dict(), 
                "curve_encoder": curve_encoder.state_dict(), 
                "curve_decoder": curve_decoder.state_dict(), 
                "patch_optimizer": patch_optimizer.state_dict(), 
                "curve_optimizer": curve_optimizer.state_dict()
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    if rank == 0:
        wandb.finish()

def parseargs():
    parser = argparse.ArgumentParser("VAE CAD Reconstruction", parents=[get_args_parser()])
    args = parser.parse_args()
    config = Config()
    if args.use_smoothness_loss:
        config.USE_SMOOTHNESS_LOSS = True
        config.CURVE_SMOOTHNESS_TAU = args.curve_smoothness_tau
        config.PATCH_SMOOTHNESS_TAU = args.patch_smoothness_tau
        config.SMOOTHNESS_WEIGHT = args.smoothness_weight
    return args, config

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    args, config = parseargs()
    if args.eval:
        print("Eval mode not implemented in optimized version")
    else:
        if num_gpus > 1:
            mp.spawn(train_pipeline, args=(num_gpus, args, config), nprocs=num_gpus, join=True)
        else:
            train_pipeline(0, 1, args, config)