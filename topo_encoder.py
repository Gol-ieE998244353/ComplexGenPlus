import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class TopoConfig:
    CURVE_LATENT_DIM: int = 256
    PATCH_LATENT_DIM: int = 384
    HIDDEN_DIM: int = 512
    LATENT_DIM: int = 256
    NUM_ENC_LAYERS: int = 6
    DROPOUT: float = 0.1
    BOUNDARY_WEIGHT: float = 1.0
    DIRECTION_WEIGHT: float = 1.0
    CURVE_WEIGHT: float = 1.0
    PATCH_WEIGHT: float = 1.0
    NEIGHBOR_WEIGHT: float = 1.0
    KL_WEIGHT_START: float = 0.0
    KL_WEIGHT_END: float = 0.001
    KL_WARMUP_EPOCHS: int = 50
    CONSISTENCY_THRESHOLD: float = 0.1
    ASSIGNMENT_MODE: str = 'direct'

RELATIONS = ['next', 'prev', 'mate', 'mate_next', 'mate_prev']


class MessagePassingLayer(nn.Module):
    """消息传递层 - 输入:x[B,N,D], idx[B,N,5], valid[B,N,5] 输出:[B,N,D]"""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.rel_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(5)])
        self.self_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        neighbors = x.gather(1, idx.view(B, -1, 1).expand(-1, -1, D)).view(B, N, 5, D)
        agg = sum(proj(neighbors[:, :, i]) * valid[:, :, i:i+1].float() 
                  for i, proj in enumerate(self.rel_projs))
        agg = agg / valid.float().sum(-1, keepdim=True).clamp(min=1)
        out = self.out_proj(torch.cat([self.self_proj(x), agg], -1))
        x = self.norm(x + out)
        return self.norm2(x + self.ffn(x))


class TopoEncoder(nn.Module):
    """TopoEncoder - 需要邻居GT的消息传递编码器
    输入: curve_lat[B,Nc,Dc], patch_lat[B,Np,Dp], he_data Dict
    输出: {he_repr:[B,Nhe,D], mean:[B,Dlat], logvar:[B,Dlat], z:[B,Dlat]}
    """
    def __init__(self, config: TopoConfig):
        super().__init__()
        self.config = config
        D = config.HIDDEN_DIM
        in_dim = config.CURVE_LATENT_DIM + config.PATCH_LATENT_DIM + 2
        self.input_proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D), nn.GELU())
        self.dir_embed = nn.Embedding(2, D)
        self.enc_layers = nn.ModuleList([
            MessagePassingLayer(D, config.DROPOUT) for _ in range(config.NUM_ENC_LAYERS)
        ])
        self.to_mean = nn.Linear(D, config.LATENT_DIM)
        self.to_logvar = nn.Linear(D, config.LATENT_DIM)
        self.z_proj = nn.Linear(config.LATENT_DIM, D)
    
    def _prepare(self, he_data: Dict, device):
        idx = torch.stack([he_data[r].clamp(min=0).to(device) for r in RELATIONS], dim=2)
        valid = torch.stack([(he_data[r] >= 0).to(device) for r in RELATIONS], dim=2)
        return idx, valid, he_data['mask'].to(device)
    
    def _encode_input(self, curve_lat: torch.Tensor, patch_lat: torch.Tensor, 
                      he_data: Dict, device) -> torch.Tensor:
        c_idx = he_data['curve_idx'].clamp(min=0).to(device)
        p_idx = he_data['patch_idx'].clamp(min=0).to(device)
        he_c = curve_lat.gather(1, c_idx.unsqueeze(-1).expand(-1, -1, curve_lat.shape[-1]))
        he_p = patch_lat.gather(1, p_idx.unsqueeze(-1).expand(-1, -1, patch_lat.shape[-1]))
        he_p = he_p.masked_fill((he_data['patch_idx'] < 0).to(device).unsqueeze(-1), 0)
        bnd = he_data['is_boundary'].float().to(device).unsqueeze(-1)
        mate = (he_data['mate'] >= 0).float().to(device).unsqueeze(-1)
        x = self.input_proj(torch.cat([he_c, he_p, bnd, mate], -1))
        return x + self.dir_embed(he_data['direction'].long().to(device))
    
    def forward(self, curve_lat: torch.Tensor, patch_lat: torch.Tensor, 
                he_data: Dict) -> Dict:
        device = curve_lat.device
        idx, valid, mask = self._prepare(he_data, device)
        x = self._encode_input(curve_lat, patch_lat, he_data, device)
        for layer in self.enc_layers:
            x = layer(x, idx, valid)
        x_masked = x * mask.unsqueeze(-1).float()
        global_feat = x_masked.sum(1) / mask.sum(1, keepdim=True).float().clamp(min=1)
        mean = self.to_mean(global_feat)
        logvar = self.to_logvar(global_feat).clamp(-10, 10)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp().clamp(1e-6, 10)
        he_repr = x + self.z_proj(z).unsqueeze(1)
        return {'he_repr': he_repr, 'mean': mean, 'logvar': logvar, 'z': z}


class TopoDecoder(nn.Module):
    """TopoDecoder - 不需要邻居GT的MLP解码器
    输入: he_repr[B,Nhe,D]
    输出: {boundary_logits:[B,Nhe], direction_logits:[B,Nhe], 
           curve_latent:[B,Nhe,Dc], patch_latent:[B,Nhe,Dp],
           neighbor_curve: 5×[B,Nhe,Dc], neighbor_patch: 5×[B,Nhe,Dp]}
    """
    def __init__(self, config: TopoConfig):
        super().__init__()
        self.config = config
        D = config.HIDDEN_DIM
        self.boundary_head = nn.Linear(D, 1)
        self.direction_head = nn.Linear(D, 1)
        self.curve_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.CURVE_LATENT_DIM)
        )
        self.patch_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.PATCH_LATENT_DIM)
        )
        self.neighbor_curve_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.CURVE_LATENT_DIM))
            for _ in range(5)
        ])
        self.neighbor_patch_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.PATCH_LATENT_DIM))
            for _ in range(5)
        ])
    
    def forward(self, he_repr: torch.Tensor) -> Dict:
        return {
            'boundary_logits': self.boundary_head(he_repr).squeeze(-1),
            'direction_logits': self.direction_head(he_repr).squeeze(-1),
            'curve_latent': self.curve_head(he_repr),
            'patch_latent': self.patch_head(he_repr),
            'neighbor_curve': [head(he_repr) for head in self.neighbor_curve_heads],
            'neighbor_patch': [head(he_repr) for head in self.neighbor_patch_heads],
        }


class TopoVAE(nn.Module):
    """完整TopoVAE
    forward: curve_lat, patch_lat, he_data -> {enc输出 + dec输出}
    encode: curve_lat, patch_lat, he_data -> enc输出
    decode: he_repr -> dec输出
    """
    def __init__(self, config: TopoConfig):
        super().__init__()
        self.config = config
        self.encoder = TopoEncoder(config)
        self.decoder = TopoDecoder(config)
    
    def forward(self, curve_lat: torch.Tensor, patch_lat: torch.Tensor, 
                he_data: Dict) -> Dict:
        enc_out = self.encoder(curve_lat, patch_lat, he_data)
        dec_out = self.decoder(enc_out['he_repr'])
        return {**enc_out, **dec_out}
    
    def encode(self, curve_lat: torch.Tensor, patch_lat: torch.Tensor, 
               he_data: Dict) -> Dict:
        return self.encoder(curve_lat, patch_lat, he_data)
    
    def decode(self, he_repr: torch.Tensor) -> Dict:
        return self.decoder(he_repr)

def compute_topo_loss(pred: Dict, target: Dict, he_data: Dict, 
                      kl_w: float, cfg: TopoConfig) -> Dict:
    """计算完整topo损失"""
    device = pred['mean'].device if 'mean' in pred else pred['curve_latent'].device
    mask = he_data['mask'].to(device).float()
    n = mask.sum().clamp(min=1)
    losses = {}
    if 'mean' in pred and 'logvar' in pred:
        kl = -0.5 * (1 + pred['logvar'] - pred['mean'].pow(2) - pred['logvar'].exp()).sum(-1).mean()
        losses['kl'] = kl
    else:
        kl = torch.tensor(0.0, device=device)
        losses['kl'] = kl
    bnd = (F.binary_cross_entropy_with_logits(
        pred['boundary_logits'], he_data['is_boundary'].float().to(device), reduction='none'
    ) * mask).sum() / n
    dir_loss = (F.binary_cross_entropy_with_logits(
        pred['direction_logits'], he_data['direction'].float().to(device), reduction='none'
    ) * mask).sum() / n
    losses['boundary'] = bnd
    losses['direction'] = dir_loss
    curve = (F.mse_loss(pred['curve_latent'], target['curve_latent'].to(device), reduction='none').mean(-1) * mask).sum() / n
    losses['curve'] = curve
    p_mask = mask * (~he_data['is_boundary'].to(device)).float()
    patch = (F.mse_loss(pred['patch_latent'], target['patch_latent'].to(device), reduction='none').mean(-1) * p_mask).sum() / p_mask.sum().clamp(min=1)
    losses['patch'] = patch
    target_curve = target['curve_latent'].to(device)
    target_patch = target['patch_latent'].to(device)
    neighbor_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    for i, rel in enumerate(RELATIONS):
        rel_idx = he_data[rel].to(device)
        valid = (rel_idx >= 0) & mask.bool()
        if not valid.any():
            continue
        nb_curve_target = target_curve.gather(1, rel_idx.clamp(min=0).unsqueeze(-1).expand_as(target_curve))
        curve_loss = (F.mse_loss(pred['neighbor_curve'][i], nb_curve_target, reduction='none').mean(-1) * valid.float()).sum() / valid.float().sum()
        nb_patch_target = target_patch.gather(1, rel_idx.clamp(min=0).unsqueeze(-1).expand_as(target_patch))
        nb_is_boundary = he_data['is_boundary'].to(device).gather(1, rel_idx.clamp(min=0))
        patch_valid = valid & (~nb_is_boundary)
        if patch_valid.any():
            patch_loss = (F.mse_loss(pred['neighbor_patch'][i], nb_patch_target, reduction='none').mean(-1) * patch_valid.float()).sum() / patch_valid.float().sum()
        else:
            patch_loss = torch.tensor(0.0, device=device)
        neighbor_loss = neighbor_loss + curve_loss + patch_loss
        valid_count += 1
    neighbor_loss = neighbor_loss / max(valid_count, 1)
    losses['neighbor'] = neighbor_loss
    total = (kl_w * kl + cfg.BOUNDARY_WEIGHT * bnd + cfg.DIRECTION_WEIGHT * dir_loss + 
             cfg.CURVE_WEIGHT * curve + cfg.PATCH_WEIGHT * patch + 
             cfg.NEIGHBOR_WEIGHT * neighbor_loss)
    losses['total'] = total
    return losses


class KLScheduler:
    def __init__(self, start: float, end: float, warmup: int):
        self.start, self.end, self.warmup = start, end, warmup
    
    def get_weight(self, epoch: int) -> float:
        if epoch >= self.warmup:
            return self.end
        return self.start + (self.end - self.start) * epoch / self.warmup