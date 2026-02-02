"""
flow_model.py - BRep Flow Model v13
[修改] topo latent完整参与flow，并通过topo_ae重建拓扑关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class FlowConfig:
    CURVE_LATENT_DIM: int = 16
    PATCH_LATENT_DIM: int = 32
    TOPO_LATENT_DIM: int = 128
    PC_FEATURE_DIM: int = 7
    MAX_CURVES: int = 200
    MAX_PATCHES: int = 100
    MAX_TOPO: int = 500
    NUM_PC_TOKENS: int = 64
    D_MODEL: int = 256
    NHEAD: int = 8
    NUM_LAYERS: int = 8
    DROPOUT: float = 0.0
    NUM_INFERENCE_STEPS: int = 50
    CFG_SCALE: float = 2.0
    P_UNCOND: float = 0.1
    CFG_START_T: float = 0.3
    LOGIT_MEAN: float = 0.0
    LOGIT_STD: float = 1.0
    TIME_SAMPLING_WARMUP_EPOCHS: int = 10
    RECON_LOSS_WEIGHT: float = 0.0
    PADDING_REG_WEIGHT: float = 0.1
    BBOX_LOSS_WEIGHT: float = 0.1
    TOPO_LOSS_WEIGHT: float = 1.0  # [修改] 新增

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class PointNetEncoder(nn.Module):
    """基于PCN思想的PointNet编码器"""
    def __init__(self, in_channels: int = 7, d_model: int = 256, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, d_model),
        )
        self.query = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, pc_features: torch.Tensor, pc_batch_idx: torch.Tensor,
                batch_size: int) -> Dict[str, torch.Tensor]:
        device = pc_features.device
        results = []
        for b in range(batch_size):
            mask = pc_batch_idx == b
            feat_b = pc_features[mask]
            if feat_b.shape[0] == 0:
                results.append(torch.zeros(self.num_tokens, self.d_model, device=device))
                continue
            f1 = self.mlp1(feat_b)
            g = f1.max(dim=0, keepdim=True)[0]
            g_expand = g.expand(f1.shape[0], -1)
            f2 = self.mlp2(torch.cat([f1, g_expand], dim=-1))
            f2 = f2.unsqueeze(0)
            q = self.query.unsqueeze(0)
            out = self.norm(q + self.cross_attn(q, f2, f2)[0])
            results.append(out.squeeze(0))
        tokens = torch.stack(results, dim=0)
        return {'tokens': tokens}

class AdaLNZero(nn.Module):
    """DiT风格的AdaLN-Zero"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)
    
    def forward(self, modulation: torch.Tensor):
        return self.mlp(modulation).chunk(6, dim=-1)

class ImprovedMMDiTBlock(nn.Module):
    """MMDiT Block with QK-Norm"""
    def __init__(self, d_model: int, nhead: int, n_channels: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_channels = n_channels
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.adaln = nn.ModuleList([AdaLNZero(d_model) for _ in range(n_channels)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model, elementwise_affine=False) for _ in range(n_channels)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model, elementwise_affine=False) for _ in range(n_channels)])
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
            for _ in range(n_channels)
        ])
    
    def forward(self, channels: List[torch.Tensor], time_emb: torch.Tensor,
                cond_tokens: torch.Tensor = None) -> List[torch.Tensor]:
        B, device = channels[0].shape[0], channels[0].device
        N_list = [c.shape[1] for c in channels]
        modulations = [self.adaln[i](time_emb) for i in range(self.n_channels)]
        channels_mod = []
        for i, c in enumerate(channels):
            sc1, sh1, g1, sc2, sh2, g2 = modulations[i]
            c_mod = self.norm1[i](c) * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
            channels_mod.append((c_mod, (sc2, sh2, g1, g2)))
        fused = torch.cat([cm[0] for cm in channels_mod], dim=1)
        if cond_tokens is not None:
            fused = torch.cat([fused, cond_tokens], dim=1)
        q = self.q_proj(fused).view(B, -1, self.nhead, self.head_dim)
        k = self.k_proj(fused).view(B, -1, self.nhead, self.head_dim)
        v = self.v_proj(fused).view(B, -1, self.nhead, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, channels[0].shape[-1])
        out = self.out_proj(out)
        outputs = []
        offset = 0
        for i, (c, (sc2, sh2, g1, g2)) in enumerate(zip(channels, [cm[1] for cm in channels_mod])):
            out_i = out[:, offset:offset+N_list[i]] * g1.unsqueeze(1)
            c = c + out_i
            c_mod2 = self.norm2[i](c) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
            c = c + self.ffn[i](c_mod2) * g2.unsqueeze(1)
            outputs.append(c)
            offset += N_list[i]
        return outputs

class FlowModel(nn.Module):
    """Flow Model - 预测velocity v = x_1 - x_0"""
    def __init__(self, config: FlowConfig, use_topo: bool = False):
        super().__init__()
        self.config = config
        self.use_topo = use_topo
        self.n_channels = 3 if use_topo else 2
        D = config.D_MODEL
        self.curve_proj = nn.Linear(config.CURVE_LATENT_DIM, D)
        self.patch_proj = nn.Linear(config.PATCH_LATENT_DIM, D)
        self.curve_unproj = nn.Linear(D, config.CURVE_LATENT_DIM)
        self.patch_unproj = nn.Linear(D, config.PATCH_LATENT_DIM)
        if use_topo:
            self.topo_proj = nn.Linear(config.TOPO_LATENT_DIM, D)
            self.topo_unproj = nn.Linear(D, config.TOPO_LATENT_DIM)
        self.type_embed = nn.Parameter(torch.randn(self.n_channels, D) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(D),
            nn.Linear(D, D * 2), nn.SiLU(), nn.Linear(D * 2, D),
        )
        self.blocks = nn.ModuleList([
            ImprovedMMDiTBlock(D, config.NHEAD, self.n_channels, config.DROPOUT)
            for _ in range(config.NUM_LAYERS)
        ])
        self.final_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(self.n_channels)])
    
    def forward(self, x_t: Dict[str, torch.Tensor], t: torch.Tensor,
                cond_tokens: torch.Tensor = None) -> Dict:
        if t.dim() == 2:
            t = t.squeeze(-1)
        time_emb = self.time_embed(t)
        c = self.curve_proj(x_t['curve']) + self.type_embed[0]
        p = self.patch_proj(x_t['patch']) + self.type_embed[1]
        channels = [c, p]
        if self.use_topo and 'topo' in x_t:
            channels.append(self.topo_proj(x_t['topo']) + self.type_embed[2])
        for block in self.blocks:
            channels = block(channels, time_emb, cond_tokens)
        c = self.final_norm[0](channels[0])
        p = self.final_norm[1](channels[1])
        result = {'curve': self.curve_unproj(c), 'patch': self.patch_unproj(p)}
        if self.use_topo and len(channels) > 2:
            result['topo'] = self.topo_unproj(self.final_norm[2](channels[2]))
        return {'v_pred': result}

class BrepFlow(nn.Module):
    """BRep Flow v13
    [修改] topo latent完整参与flow，并通过topo_ae重建拓扑关系
    """
    def __init__(self, config: FlowConfig, use_topo: bool = False,
                 curve_decoder=None, patch_decoder=None, topo_ae=None):  # [修改] 新增topo_ae
        super().__init__()
        self.config = config
        self.use_topo = use_topo
        self.pc_encoder = PointNetEncoder(
            in_channels=config.PC_FEATURE_DIM,
            d_model=config.D_MODEL,
            num_tokens=config.NUM_PC_TOKENS
        )
        self.flow_model = FlowModel(config, use_topo)
        self.curve_decoder = curve_decoder
        self.patch_decoder = patch_decoder
        if curve_decoder is not None:
            for p in curve_decoder.parameters():
                p.requires_grad = False
        if patch_decoder is not None:
            for p in patch_decoder.parameters():
                p.requires_grad = False
        # [修改] topo_ae用于重建拓扑
        self.topo_ae = topo_ae
        if topo_ae is not None:
            for p in topo_ae.parameters():
                p.requires_grad = False
    
    def _sample_time(self, batch_size: int, device: torch.device,
                     strategy: str = 'logit_normal', current_epoch: int = 0) -> torch.Tensor:
        if current_epoch < self.config.TIME_SAMPLING_WARMUP_EPOCHS:
            t = torch.rand(batch_size, device=device)
        elif strategy == 'logit_normal':
            u = torch.randn(batch_size, device=device) * self.config.LOGIT_STD + self.config.LOGIT_MEAN
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        return t.clamp(0.00001, 0.99999)
    
    def _get_n_valid(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从batch中获取n_valid"""
        B = batch['curve_latent'].shape[0]
        device = batch['curve_latent'].device
        n_valid_curves = batch.get('n_valid_curves')
        n_valid_patches = batch.get('n_valid_patches')
        if n_valid_curves is None:
            n_valid_curves = torch.full((B,), self.config.MAX_CURVES, device=device)
        if n_valid_patches is None:
            n_valid_patches = torch.full((B,), self.config.MAX_PATCHES, device=device)
        if not isinstance(n_valid_curves, torch.Tensor):
            n_valid_curves = torch.tensor(n_valid_curves, device=device)
        if not isinstance(n_valid_patches, torch.Tensor):
            n_valid_patches = torch.tensor(n_valid_patches, device=device)
        return {'n_valid_curves': n_valid_curves, 'n_valid_patches': n_valid_patches}
    
    def _padding_reg_loss(self, x_1_pred: Dict[str, torch.Tensor], 
                          n_valid: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算padding位置的regularization loss"""
        B = x_1_pred['curve'].shape[0]
        device = x_1_pred['curve'].device
        loss = torch.tensor(0.0, device=device)
        count = 0
        for b in range(B):
            nv = int(n_valid['n_valid_curves'][b].item())
            if nv < self.config.MAX_CURVES:
                padding = x_1_pred['curve'][b, nv:]
                loss = loss + (padding ** 2).mean()
                count += 1
        for b in range(B):
            nv = int(n_valid['n_valid_patches'][b].item())
            if nv < self.config.MAX_PATCHES:
                padding = x_1_pred['patch'][b, nv:]
                loss = loss + (padding ** 2).mean()
                count += 1
        return loss / max(count, 1)
    
    def training_step(self, batch: Dict[str, torch.Tensor],
                      time_sampling: str = 'logit_normal',
                      current_epoch: int = 0) -> Dict:
        """训练步骤 - topo完整参与flow并重建"""
        device = batch['curve_latent'].device
        B = batch['curve_latent'].shape[0]
        losses = {}
        n_valid = self._get_n_valid(batch)
        # Step 1: 编码点云条件
        pc_features = batch.get('pc_features')
        pc_batch_idx = batch.get('pc_batch_idx')
        cond_tokens = None
        if pc_features is not None and pc_batch_idx is not None:
            pc_out = self.pc_encoder(pc_features, pc_batch_idx, B)
            cond_tokens = pc_out['tokens']
        # Step 2: 准备flow数据 - curve, patch, topo都参与
        x_1 = {'curve': batch['curve_latent'], 'patch': batch['patch_latent']}
        if self.use_topo and 'topo_latent' in batch:
            x_1['topo'] = batch['topo_latent']
        x_0 = {k: torch.randn_like(v) for k, v in x_1.items()}
        v_target = {k: x_1[k] - x_0[k] for k in x_1}
        # Step 3: 采样时间并插值
        t = self._sample_time(B, device, time_sampling, current_epoch)
        t_expand = t.view(-1, 1, 1)
        x_t = {k: (1 - t_expand) * x_0[k] + t_expand * x_1[k] for k in x_1}
        # Step 4: Flow model forward
        out = self.flow_model(x_t, t, cond_tokens)
        v_pred = out['v_pred']
        # Step 5: Velocity Loss - curve, patch, topo
        curve_vel_loss = F.mse_loss(v_pred['curve'], v_target['curve'])
        patch_vel_loss = F.mse_loss(v_pred['patch'], v_target['patch'])
        losses['curve_vel_loss'] = curve_vel_loss
        losses['patch_vel_loss'] = patch_vel_loss
        total_loss = curve_vel_loss + patch_vel_loss
        if self.use_topo and 'topo' in v_pred:
            topo_vel_loss = F.mse_loss(v_pred['topo'], v_target['topo'])
            losses['topo_vel_loss'] = topo_vel_loss
            total_loss = total_loss + topo_vel_loss
        else:
            losses['topo_vel_loss'] = torch.tensor(0.0, device=device)
        # Step 6: Padding Regularization Loss
        x_1_pred = {k: x_t[k] + (1.0 - t_expand) * v_pred[k] for k in v_pred}
        padding_loss = self._padding_reg_loss(x_1_pred, n_valid)
        losses['padding_loss'] = padding_loss
        total_loss = total_loss + self.config.PADDING_REG_WEIGHT * padding_loss
        if self.use_topo and self.topo_ae is not None and 'topo_data' in batch:
            topo_pred = x_1_pred['topo']
            curve_gt = batch['curve_latent']
            patch_gt = batch['patch_latent']
            topo_data = batch['topo_data']
            pred_out = self.topo_ae.decoder(topo_pred)
            from topo_edge import build_topo_targets
            target = build_topo_targets(topo_data, curve_gt, patch_gt, device)
            from topo_encoder import compute_topo_loss, TopoConfig
            topo_cfg = TopoConfig()
            topo_losses = compute_topo_loss(pred_out, target, topo_data, topo_cfg)
            topo_recon_loss = (topo_losses.get('boundary', 0) + 
                              topo_losses.get('patch0', 0) + 
                              topo_losses.get('patch1', 0) +
                              topo_losses.get('adj_edge', 0) + 
                              topo_losses.get('adj_valid', 0))
            losses['topo_recon_loss'] = topo_recon_loss
            total_loss = total_loss + self.config.TOPO_LOSS_WEIGHT * topo_recon_loss
        else:
            losses['topo_recon_loss'] = torch.tensor(0.0, device=device)
        losses['total'] = total_loss
        losses['t_mean'] = t.mean()
        return losses
    
    @torch.no_grad()
    def generate(self, pc_features: torch.Tensor, pc_batch_idx: torch.Tensor,
                 batch_size: int, num_steps: int = None) -> Dict:
        """生成BRep latent - curve, patch, topo都生成"""
        num_steps = num_steps or self.config.NUM_INFERENCE_STEPS
        device = pc_features.device
        B = batch_size
        cfg_scale = self.config.CFG_SCALE
        pc_out = self.pc_encoder(pc_features, pc_batch_idx, B)
        cond_tokens = pc_out['tokens']
        x_t = {
            'curve': torch.randn(B, self.config.MAX_CURVES, self.config.CURVE_LATENT_DIM, device=device),
            'patch': torch.randn(B, self.config.MAX_PATCHES, self.config.PATCH_LATENT_DIM, device=device),
        }
        if self.use_topo:
            x_t['topo'] = torch.randn(B, self.config.MAX_TOPO, self.config.TOPO_LATENT_DIM, device=device)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            use_cfg = (cfg_scale > 1.0) and (t_val >= self.config.CFG_START_T)
            out_cond = self.flow_model(x_t, t, cond_tokens)
            v_pred = out_cond['v_pred']
            if use_cfg:
                out_uncond = self.flow_model(x_t, t, None)
                v_pred = {k: out_uncond['v_pred'][k] + cfg_scale * (v_pred[k] - out_uncond['v_pred'][k])
                          for k in v_pred}
            for k in x_t:
                x_t[k] = x_t[k] + dt * v_pred[k]
        return {
            'curve_latent': x_t['curve'],
            'patch_latent': x_t['patch'],
            'topo_latent': x_t.get('topo'),
        }
    
    @torch.no_grad()
    def denoise_from_t(self, x_t: Dict[str, torch.Tensor], t_start: float,
                       cond_tokens: torch.Tensor, num_steps: int = None) -> Dict:
        """从t_start积分到t=1"""
        num_steps = num_steps or self.config.NUM_INFERENCE_STEPS
        device = x_t['curve'].device
        B = x_t['curve'].shape[0]
        cfg_scale = self.config.CFG_SCALE
        steps_remaining = max(1, int((1.0 - t_start) * num_steps))
        dt = (1.0 - t_start) / steps_remaining
        x_t = {k: v.clone() for k, v in x_t.items()}
        t_val = t_start
        for _ in range(steps_remaining):
            t = torch.full((B,), t_val, device=device)
            use_cfg = (cfg_scale > 1.0) and (t_val >= self.config.CFG_START_T)
            out = self.flow_model(x_t, t, cond_tokens)
            v_pred = out['v_pred']
            if use_cfg:
                out_uncond = self.flow_model(x_t, t, None)
                v_pred = {k: out_uncond['v_pred'][k] + cfg_scale * (v_pred[k] - out_uncond['v_pred'][k])
                          for k in v_pred}
            for k in x_t:
                x_t[k] = x_t[k] + dt * v_pred[k]
            t_val += dt
        return {'curve': x_t['curve'], 'patch': x_t['patch'], 'topo': x_t.get('topo')}

def build_flow_system(config: FlowConfig, vae_ckpt: str = None, 
                      use_topo: bool = False, topo_ae_ckpt: str = None) -> BrepFlow:  # [修改] 新增topo_ae_ckpt
    """构建Flow系统"""
    curve_dec, patch_dec = None, None
    if vae_ckpt is not None:
        from train_pc_nm import CurveDecoder, PatchDecoder, Config
        cfg = Config()
        curve_dec = CurveDecoder(cfg)
        patch_dec = PatchDecoder(cfg)
        ckpt = torch.load(vae_ckpt, map_location='cpu')
        clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
        curve_dec.load_state_dict(clean(ckpt.get('curve_decoder', {})))
        patch_dec.load_state_dict(clean(ckpt.get('patch_decoder', {})))
        print(f"[INFO] Loaded VAE decoders from {vae_ckpt}")
    # [修改] 加载topo_ae
    topo_ae = None
    if use_topo and topo_ae_ckpt is not None:
        from topo_encoder import TopoVAE, TopoConfig
        topo_cfg = TopoConfig()
        topo_ae = TopoVAE(topo_cfg)
        ckpt = torch.load(topo_ae_ckpt, map_location='cpu')
        clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
        topo_ae.load_state_dict(clean(ckpt.get('model', ckpt)))
        print(f"[INFO] Loaded TopoVAE from {topo_ae_ckpt}")
    return BrepFlow(config, use_topo, curve_dec, patch_dec, topo_ae)