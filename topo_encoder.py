"""
topo_encoder.py - 极简拓扑VAE

核心：消息传递 + VAE，无attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

@dataclass
class TopoConfig:
    CURVE_LATENT_DIM: int = 256
    PATCH_LATENT_DIM: int = 384
    HIDDEN_DIM: int = 512
    LATENT_DIM: int = 256
    NUM_LAYERS: int = 6
    DROPOUT: float = 0.1
    USE_CHECKPOINT: bool = False
    
    BOUNDARY_WEIGHT: float = 1.0
    DIRECTION_WEIGHT: float = 1.0
    CURVE_WEIGHT: float = 1.0
    PATCH_WEIGHT: float = 1.0
    KL_WEIGHT_START: float = 0.0
    KL_WEIGHT_END: float = 0.001
    KL_WARMUP_EPOCHS: int = 50

RELATIONS = ['next', 'prev', 'mate', 'mate_next', 'mate_prev']


class MessagePassingLayer(nn.Module):
    """消息传递：聚合5邻居，每种关系独立变换"""
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


class TopoVAE(nn.Module):
    def __init__(self, config: TopoConfig):
        super().__init__()
        self.config = config
        D = config.HIDDEN_DIM
        
        in_dim = config.CURVE_LATENT_DIM + config.PATCH_LATENT_DIM + 2
        self.input_proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D), nn.GELU())
        self.dir_embed = nn.Embedding(2, D)
        
        self.enc_layers = nn.ModuleList([MessagePassingLayer(D, config.DROPOUT) for _ in range(config.NUM_LAYERS)])
        self.to_mean = nn.Linear(D, config.LATENT_DIM)
        self.to_logvar = nn.Linear(D, config.LATENT_DIM)
        
        self.z_proj = nn.Linear(config.LATENT_DIM, D)
        self.dec_layers = nn.ModuleList([MessagePassingLayer(D, config.DROPOUT) for _ in range(max(1, config.NUM_LAYERS // 2))])
        
        self.boundary_head = nn.Linear(D, 1)
        self.direction_head = nn.Linear(D, 1)
        self.curve_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.CURVE_LATENT_DIM))
        self.patch_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, config.PATCH_LATENT_DIM))
    
    def _prepare(self, he_data: Dict, device):
        idx = torch.stack([he_data[r].clamp(min=0).to(device) for r in RELATIONS], dim=2)
        valid = torch.stack([(he_data[r] >= 0).to(device) for r in RELATIONS], dim=2)
        return idx, valid, he_data['mask'].to(device)
    
    def _encode_input(self, curve_lat, patch_lat, he_data, device):
        c_idx = he_data['curve_idx'].clamp(min=0).to(device)
        p_idx = he_data['patch_idx'].clamp(min=0).to(device)
        he_c = curve_lat.gather(1, c_idx.unsqueeze(-1).expand(-1, -1, curve_lat.shape[-1]))
        he_p = patch_lat.gather(1, p_idx.unsqueeze(-1).expand(-1, -1, patch_lat.shape[-1]))
        he_p = he_p.masked_fill((he_data['patch_idx'] < 0).to(device).unsqueeze(-1), 0)
        bnd = he_data['is_boundary'].float().to(device).unsqueeze(-1)
        mate = (he_data['mate'] >= 0).float().to(device).unsqueeze(-1)
        x = self.input_proj(torch.cat([he_c, he_p, bnd, mate], -1))
        return x + self.dir_embed(he_data['direction'].long().to(device))
    
    def forward(self, curve_lat: torch.Tensor, patch_lat: torch.Tensor, he_data: Dict) -> Dict:
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
        
        x = x + self.z_proj(z).unsqueeze(1)
        for layer in self.dec_layers:
            x = layer(x, idx, valid)
        
        return {
            'mean': mean, 'logvar': logvar, 'z': z,
            'boundary_logits': self.boundary_head(x).squeeze(-1),
            'direction_logits': self.direction_head(x).squeeze(-1),
            'curve_latent': self.curve_head(x),
            'patch_latent': self.patch_head(x),
        }

def compute_topo_loss(pred: Dict, target: Dict, he_data: Dict, kl_w: float, cfg: TopoConfig) -> Dict:
    device = pred['mean'].device
    mask = he_data['mask'].to(device).float()
    n = mask.sum().clamp(min=1)
    
    kl = -0.5 * (1 + pred['logvar'] - pred['mean'].pow(2) - pred['logvar'].exp()).sum(-1).mean()
    bnd = (F.binary_cross_entropy_with_logits(pred['boundary_logits'], he_data['is_boundary'].float().to(device), reduction='none') * mask).sum() / n
    dir_loss = (F.binary_cross_entropy_with_logits(pred['direction_logits'], he_data['direction'].float().to(device), reduction='none') * mask).sum() / n
    curve = (F.mse_loss(pred['curve_latent'], target['curve_latent'].to(device), reduction='none').mean(-1) * mask).sum() / n
    
    p_mask = mask * (~he_data['is_boundary'].to(device)).float()
    patch = (F.mse_loss(pred['patch_latent'], target['patch_latent'].to(device), reduction='none').mean(-1) * p_mask).sum() / p_mask.sum().clamp(min=1)
    
    total = kl_w * kl + cfg.BOUNDARY_WEIGHT * bnd + cfg.DIRECTION_WEIGHT * dir_loss + cfg.CURVE_WEIGHT * curve + cfg.PATCH_WEIGHT * patch
    return {'total': total, 'kl': kl, 'boundary': bnd, 'direction': dir_loss, 'curve': curve, 'patch': patch}


class KLScheduler:
    def __init__(self, start, end, warmup):
        self.start, self.end, self.warmup = start, end, warmup
    def get_weight(self, epoch):
        return self.end if epoch >= self.warmup else self.start + (self.end - self.start) * epoch / self.warmup


if __name__ == '__main__':
    cfg = TopoConfig()
    model = TopoVAE(cfg)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    B, Nc, Np, N = 2, 10, 8, 20
    he_data = {'mask': torch.ones(B, N, dtype=torch.bool), 'curve_idx': torch.randint(0, Nc, (B, N)),
               'patch_idx': torch.randint(-1, Np, (B, N)), 'direction': torch.randint(0, 2, (B, N), dtype=torch.bool),
               'is_boundary': torch.randint(0, 2, (B, N), dtype=torch.bool), **{r: torch.randint(-1, N, (B, N)) for r in RELATIONS}}
    
    pred = model(torch.randn(B, Nc, cfg.CURVE_LATENT_DIM), torch.randn(B, Np, cfg.PATCH_LATENT_DIM), he_data)
    target = {'curve_latent': torch.randn(B, N, cfg.CURVE_LATENT_DIM), 'patch_latent': torch.randn(B, N, cfg.PATCH_LATENT_DIM)}
    losses = compute_topo_loss(pred, target, he_data, 0.001, cfg)
    print("Losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})
    losses['total'].backward()
    print("✓ OK")