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
    DROPOUT = 0.15
    CURVE_LATENT_DIM = 16
    PATCH_LATENT_DIM = 32
    PATCH_NUM_POINTS = 400
    MAX_CURVES = 100
    MAX_PATCHES = 50 
    MAX_TOTAL_ITEMS = 200
    BUCKET_BOUNDARIES = [10, 30, 60, 100, 200, 400, 800, 1600]
    HN_PE_DIM = 64
    HN_MLP_DIM = 128
    CURVE_NUM_CLASSES = 4
    PATCH_NUM_CLASSES = 6
    KL_WEIGHT_START = 0.0
    KL_WEIGHT_END = 0.0001
    KL_WARMUP_EPOCHS = 100  
    KL_FREE_BITS = 0.5  
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-6
    GRAD_CLIP_NORM = 1.0 
    EVAL_INTERVAL = 5
    SAVE_INTERVAL = 20
    USE_AMP = False
    GRADIENT_ACCUMULATION_STEPS = 1
    LOG_INTERVAL = 50000
    RECON_WEIGHT = 1.0
    ENDPOINT_WEIGHT = 0.5
    TOPOLOGY_WEIGHT = 0.5
    LABEL_WEIGHT = 0.5
    VALIDITY_WEIGHT = 0.5
    CURVATURE_WEIGHT = 0.5
    SCALE_FEATURE_DIM = 16
    CENTER_FEATURE_DIM = 16
    ENDPOINT_FEATURE_DIM = 48
    CLOSED_FEATURE_DIM = 12
    LABEL_FEATURE_DIM = 16
    TOPO_FEATURE_DIM = 20
    PATCH_NORMAL_TAU = 0.1
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
    parser.add_argument('--add_data_mode', action='store_true', help='Reset optimizer for training with new data')
    parser.add_argument('--dynamic_batch', action='store_true', help='Enable dynamic bucketed batch sizes')
    parser.add_argument('--bucketed_batch', action='store_true', help='Enable bucketed fixed batch sizes')
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
            nn.GELU()
        )
        
        # 2. 注意力模块 
        self.attention_net = nn.Sequential(
            nn.Conv1d(256, 64, 1), # 降维
            nn.Tanh(),             # 激活
            nn.Conv1d(64, 1, 1)    # 输出分数
        )
        
        # 3. 最大池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(512, output_dim) # 最终压缩回 256
        )
    
    def forward(self, points):
        B, N, P, C = points.shape
        # 变换维度: (B*N, C, P)
        x = points.view(B*N, P, C).transpose(1, 2)
        
        # 1. 提取逐点特征 (Local Features)
        feat = self.encoder(x)  # Shape: (B*N, 256, P)
        
        # 2. 分支一：Attention Pooling
        # 计算权重: (B*N, 1, P)
        scores = self.attention_net(feat)
        weights = F.softmax(scores, dim=2) 
        # 加权求和: sum( (B*N, 256, P) * (B*N, 1, P) ) -> (B*N, 256)
        global_feat_attn = torch.sum(feat * weights, dim=2)
        
        # 3. 分支二：Max Pooling
        # 取最大值: (B*N, 256)
        global_feat_max = self.max_pool(feat).squeeze(-1)
        
        # 4. Concatenation
        combined = torch.cat([global_feat_attn, global_feat_max], dim=1) # (B*N, 512)
        
        # 5. 最终投影
        out = self.proj(combined) # (B*N, 256)
        
        return out.view(B, N, -1)

class Patch2DEncoder(nn.Module):
    def __init__(self, input_channels=6, output_dim=384, dropout=0.1, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU()
        )
        
        self.normal_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU()
        )
        
        self.fusion_encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 384),
            nn.GELU(),
        )
        
        self.attn_net = nn.Sequential(
            nn.Conv2d(384, 64, 1),
            nn.Tanh(),
            nn.Conv2d(64, 1, 1)
        )
        
        # 2. 定义 2D MaxPool
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 3. 修改投影层输入维度 (384 * 2)
        self.proj = nn.Linear(384 * 2, output_dim)
    
    def forward(self, patch_data):
        B, N, HW, C = patch_data.shape
        H = W = self.grid_size
        
        positions = patch_data[..., :3].view(B*N, H, W, 3).permute(0, 3, 1, 2)
        normals = patch_data[..., 3:].view(B*N, H, W, 3).permute(0, 3, 1, 2)
        
        normals = F.normalize(normals, dim=1, eps=1e-8)
        
        pos_feat = self.position_encoder(positions)
        norm_feat = self.normal_encoder(normals)
        combined = torch.cat([pos_feat, norm_feat], dim=1)
        
        x = self.fusion_encoder(combined)
        
        feat_max = self.max_pool(x).flatten(1) # (B*N, 384)
        
        # Attention 分支
        scores = self.attn_net(x) # (B*N, 1, H, W)
        # 将 H,W 展平做 Softmax
        B_N, _, H, W = scores.shape
        scores = scores.view(B_N, 1, -1) # (B*N, 1, H*W)
        weights = F.softmax(scores, dim=-1)
        
        # x_flat = x.view(B_N, 384, -1) # (B*N, 384, H*W)
        x_flat = x.view(B_N, 384, -1) # (B*N, 384, H*W)
        feat_attn = torch.sum(x_flat * weights, dim=-1) # (B*N, 384)
            
        combined = torch.cat([feat_attn, feat_max], dim=1)
        return self.proj(combined).view(B, N, -1)
    

    
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
    def __init__(self, config):
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
        self.patch_position_embed = MLP_hn(input_dim=config.HN_PE_DIM * 2, hidden_dim=config.HN_MLP_DIM, output_dim=3, num_layers=3, input_dim_fea=config.PATCH_LATENT_DIM)
        # 直接输出3D向量而非球坐标
        self.patch_normal_embed = MLP_hn(input_dim=config.HN_PE_DIM * 2, hidden_dim=config.HN_MLP_DIM, output_dim=3, num_layers=3, input_dim_fea=config.PATCH_LATENT_DIM)
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
        position_offset = self.patch_position_embed(patch_pe.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
        position_offset = torch.clamp(position_offset, -2, 2)
        points = startpoint.unsqueeze(2) + position_offset
        points = torch.clamp(points, -0.6, 0.6)
        # 直接预测3D法向量并归一化
        normals_raw = self.patch_normal_embed(patch_pe.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
        normals = F.normalize(normals_raw, dim=-1, eps=1e-8)
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
    
    pred_n = F.normalize(pred["normals"], dim=-1, eps=1e-8)
    target_n = F.normalize(target["normals"], dim=-1, eps=1e-8)
    cosine_sim = (pred_n * target_n).sum(dim=-1, keepdim=True)
    cosine_loss = 1.0 - cosine_sim
    recon_normals = 0.5 * cosine_loss
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
    total_loss = (kl_weight * kl + config.RECON_WEIGHT * (recon_points + recon_normals * 0.3) + config.TOPOLOGY_WEIGHT * (u_closed_loss + v_closed_loss) + config.LABEL_WEIGHT * label_loss + config.VALIDITY_WEIGHT * validity_loss) #+ config.CURVATURE_WEIGHT * curvature_loss
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
    total_loss = (kl_weight * kl + config.RECON_WEIGHT * (recon_points + recon_endpoints) + config.TOPOLOGY_WEIGHT * closed_loss + config.LABEL_WEIGHT * label_loss + config.VALIDITY_WEIGHT * validity_loss) #+ config.CURVATURE_WEIGHT * curvature_loss
    total_loss = torch.clamp(total_loss, max=1e6)
    losses["total"] = total_loss
    return losses

def move_to_device(data_dict, device):
    """简化的数据移动函数 - 直接移动已处理好的数据到GPU"""
    if data_dict is None:
        return None
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in data_dict.items()}

def compute_patch_metrics(pred, target, mask):
    """计算patch的评估指标"""
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
    """计算curve的评估指标"""
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
            
            # 1. 运行 Encoder
            mean_p, z_p = None, None
            if processed_patches is not None:
                mean_p, _ = patch_encoder(
                    processed_patches["patch_points"], processed_patches["patch_normals"], 
                    processed_patches["u_closed"], processed_patches["v_closed"], 
                    processed_patches["labels"], processed_patches["scale"], 
                    processed_patches["center"], processed_patches["mask"]
                )
                z_p = mean_p # Validation 时通常直接用 mean
            
            mean_c, z_c = None, None
            if processed_curves is not None:
                mean_c, _ = curve_encoder(
                    processed_curves["curve_points"], processed_curves["endpoints"], 
                    processed_curves["is_closed"], processed_curves["labels"], 
                    processed_curves["scale"], processed_curves["center"], 
                    processed_curves["mask"]
                )
                z_c = mean_c

            # 3. 运行 Decoder
            # 处理 Patches
            if z_p is not None and processed_patches is not None:
                try:
                    pred_patch = patch_decoder(z_p)
                    pred_scale = processed_patches["scale"].unsqueeze(-1).unsqueeze(-1)
                    pred_center = processed_patches["center"].unsqueeze(-2)
                    pred_points_denorm = pred_patch["points"] * pred_scale + pred_center
                    target_points_denorm = processed_patches["patch_points"] * pred_scale + pred_center
                    
                    recon = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
                    recon = (recon * processed_patches["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_patches["mask"].sum().clamp(min=1)
                    if not torch.isnan(recon): patch_recon_errors.append(recon)
                except: pass

            # 处理 Curves
            if z_c is not None and processed_curves is not None:
                try:
                    pred_curve = curve_decoder(z_c)
                    pred_scale = processed_curves["scale"].unsqueeze(-1).unsqueeze(-1)
                    pred_center = processed_curves["center"].unsqueeze(-2)
                    pred_points_denorm = pred_curve["points"] * pred_scale + pred_center
                    target_points_denorm = processed_curves["curve_points"] * pred_scale + pred_center
                    
                    recon = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none")
                    recon = (recon * processed_curves["mask"].unsqueeze(-1).unsqueeze(-1)).sum() / processed_curves["mask"].sum().clamp(min=1)
                    if not torch.isnan(recon): curve_recon_errors.append(recon)
                except: pass

    val_metrics = {}
    if patch_recon_errors: val_metrics["patch"] = {"recon_error": torch.stack(patch_recon_errors).mean().item()}
    if curve_recon_errors: val_metrics["curve"] = {"recon_error": torch.stack(curve_recon_errors).mean().item()}
    
    patch_encoder.train(); patch_decoder.train()
    curve_encoder.train(); curve_decoder.train()
    
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
    
    # 使用优化后的dataloader - 数据已在Dataset中预处理
    use_bucketed = args.dynamic_batch or args.bucketed_batch
    max_total_items = config.MAX_TOTAL_ITEMS if args.dynamic_batch else None

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
            num_workers=4,  # 使用多进程加速数据预处理
            rank=rank, 
            world_size=num_gpus,
            use_bucketed_batch=use_bucketed,
            max_total_items=max_total_items,
            bucket_boundaries=config.BUCKET_BOUNDARIES,
            shuffle=True
        )
        # Validation data（不使用augmentation）
        val_data, val_sampler = train_data_loader_clean(
            args.batch_size,
            data_folder="data/train_small",
            rotation_augmentation=False,  # val不用augmentation
            random_angle=False,
            flag_noise=0,
            flag_grid=args.patch_grid,
            num_angle=4,
            dim_grid=args.points_per_patch_dim,
            num_workers=4,  # val用较少的worker
            rank=rank,
            world_size=num_gpus,
            use_bucketed_batch=use_bucketed,
            max_total_items=max_total_items,
            bucket_boundaries=config.BUCKET_BOUNDARIES,
            shuffle=False
        )
    else:
        train_folder = "data/partial/train" if args.partial else "data/incr"
        train_data, distribute_sampler = train_data_loader_clean(
            args.batch_size, 
            data_folder=train_folder, 
            rotation_augmentation=args.rotation_augment, 
            random_angle=args.random_angle, 
            flag_noise=args.noise, 
            flag_grid=args.patch_grid, 
            num_angle=args.num_angles, 
            dim_grid=args.points_per_patch_dim, 
            num_workers=2,  # 使用多进程加速数据预处理
            rank=rank, 
            world_size=num_gpus,
            use_bucketed_batch=use_bucketed,
            max_total_items=max_total_items,
            bucket_boundaries=config.BUCKET_BOUNDARIES,
            shuffle=True
        )
        # Validation data（不使用augmentation）
        val_folder = "data/partial/val" if args.partial else "data/large/val"
        val_data, val_sampler = train_data_loader_clean(
            args.batch_size,
            data_folder=val_folder,
            rotation_augmentation=False,  # val不用augmentation
            random_angle=False,
            flag_noise=0,
            flag_grid=args.patch_grid,
            num_angle=4,
            dim_grid=args.points_per_patch_dim,
            num_workers=2,  # val用较少的worker
            rank=rank,
            world_size=num_gpus,
            use_bucketed_batch=use_bucketed,
            max_total_items=max_total_items,
            bucket_boundaries=config.BUCKET_BOUNDARIES,
            shuffle=False
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
    scaler = GradScaler(enabled=config.USE_AMP)
    all_params = patch_params + curve_params
    kl_scheduler = KLScheduler(config.KL_WEIGHT_START, config.KL_WEIGHT_END, config.KL_WARMUP_EPOCHS)
    
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        if rank == 0:
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        # 定义一个简单的修复函数
        def fix_state_dict(state_dict, model_obj):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            # 检查当前模型是否使用了 DDP (即是否含有 module. 前缀)
            model_has_module = any(k.startswith('module.') for k in model_obj.state_dict().keys())
            
            for k, v in state_dict.items():
                if model_has_module and not k.startswith('module.'):
                    name = 'module.' + k  # 给权重加上前缀以匹配 DDP 模型
                elif not model_has_module and k.startswith('module.'):
                    name = k[7:]  # 去掉权重的前缀以匹配普通模型
                else:
                    name = k
                new_state_dict[name] = v
            return new_state_dict

        # 依次加载各个模块
        patch_encoder.load_state_dict(fix_state_dict(checkpoint["patch_encoder"], patch_encoder))
        patch_decoder.load_state_dict(fix_state_dict(checkpoint["patch_decoder"], patch_decoder))
        curve_encoder.load_state_dict(fix_state_dict(checkpoint["curve_encoder"], curve_encoder))
        curve_decoder.load_state_dict(fix_state_dict(checkpoint["curve_decoder"], curve_decoder))
        
            
        if args.add_data_mode: 
            if rank == 0:
                logger.info("Add-Data Mode: Resetting optimizer and epoch count.")
        else:
            patch_optimizer.load_state_dict(checkpoint["patch_optimizer"])
            curve_optimizer.load_state_dict(checkpoint["curve_optimizer"])
            if "scaler" in checkpoint and config.USE_AMP:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
    
    patch_optimizer.zero_grad(set_to_none=True)
    curve_optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")
    cur_epochs = start_epoch
    
    for epoch in range(start_epoch, args.max_training_iterations):
        current_kl_weight = kl_scheduler.get_weight(epoch)
        if rank == 0:
            logger.info(f"Epoch {epoch+1} - KL Weight: {current_kl_weight:.6f}")
        
        if distribute_sampler is not None and hasattr(distribute_sampler, "set_epoch"):
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
            
            # 1. 运行 Encoder
            # ================= 1. Forward Encoders =================
            z_p, mean_p, logvar_p = None, None, None
            if processed_patches is not None:
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                    mean_p, logvar_p = patch_encoder(
                        processed_patches["patch_points"], processed_patches["patch_normals"], 
                        processed_patches["u_closed"], processed_patches["v_closed"], 
                        processed_patches["labels"], processed_patches["scale"], 
                        processed_patches["center"], processed_patches["mask"]
                    )
                    std_p = torch.exp(0.5 * logvar_p).clamp(min=1e-8, max=10)
                    eps_p = torch.randn_like(std_p)
                    z_p = mean_p + eps_p * std_p

            z_c, mean_c, logvar_c = None, None, None
            if processed_curves is not None:
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                    mean_c, logvar_c = curve_encoder(
                        processed_curves["curve_points"], processed_curves["endpoints"], 
                        processed_curves["is_closed"], processed_curves["labels"], 
                        processed_curves["scale"], processed_curves["center"], 
                        processed_curves["mask"]
                    )
                    std_c = torch.exp(0.5 * logvar_c).clamp(min=1e-8, max=10)
                    eps_c = torch.randn_like(std_c)
                    z_c = mean_c + eps_c * std_c
                    
            # ================= 3. Forward Decoders & Loss =================
            total_loss = torch.tensor(0.0, device=device)
            
            # Patch Loss
            if z_p is not None and processed_patches is not None:
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
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
                    total_loss += patch_losses["total"]
                    patch_loss_acc.add(patch_losses)

            # Curve Loss
            if z_c is not None and processed_curves is not None:
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
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
                    total_loss += curve_losses["total"]
                    curve_loss_acc.add(curve_losses)
            # ================= 4. Backward & Step =================
            total_loss = total_loss / config.GRADIENT_ACCUMULATION_STEPS
            
            if not total_loss.requires_grad:
                continue

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if rank == 0:
                    logger.warning("NaN/Inf in total loss!")
                patch_optimizer.zero_grad(set_to_none=True)
                curve_optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(patch_optimizer)
                scaler.unscale_(curve_optimizer)
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=config.GRAD_CLIP_NORM)
                
                scaler.step(patch_optimizer)
                scaler.step(curve_optimizer)
                # scaler.step(optimizer)
                scaler.update()
                
                patch_optimizer.zero_grad(set_to_none=True)
                curve_optimizer.zero_grad(set_to_none=True)
                # optimizer.zero_grad(set_to_none=True)
            
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
                # 记录validation指标
                val_log_dict = {"epoch": epoch + 1}
                if "patch" in val_metrics:
                    val_log_dict["val/patch_recon"] = val_metrics['patch']['recon_error']
                if "curve" in val_metrics:
                    val_log_dict["val/curve_recon"] = val_metrics['curve']['recon_error']
                wandb.log(val_log_dict)
                
                # 保存best model
                val_loss = val_metrics.get("patch", {}).get("recon_error", 0) + val_metrics.get("curve", {}).get("recon_error", 0)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "epoch": epoch, 
                        "patch_encoder": patch_encoder.state_dict(), 
                        "patch_decoder": patch_decoder.state_dict(), 
                        "curve_encoder": curve_encoder.state_dict(), 
                        "curve_decoder": curve_decoder.state_dict(), 
                        "patch_optimizer": patch_optimizer.state_dict(), 
                        "curve_optimizer": curve_optimizer.state_dict(),
                        "val_loss": val_loss,
                        "scaler": scaler.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best_model.pth"))
                    logger.info(f"Best model saved with val_loss: {val_loss:.6f}")
        
        # 定期保存checkpoint
        if rank == 0 and (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint = {
                "epoch": epoch, 
                "patch_encoder": patch_encoder.state_dict(), 
                "patch_decoder": patch_decoder.state_dict(), 
                "curve_encoder": curve_encoder.state_dict(), 
                "curve_decoder": curve_decoder.state_dict(), 
                "patch_optimizer": patch_optimizer.state_dict(), 
                "curve_optimizer": curve_optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    if rank == 0:
        wandb.finish()

def parseargs():
    parser = argparse.ArgumentParser("VAE CAD Reconstruction", parents=[get_args_parser()])
    args = parser.parse_args()
    config = Config()
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