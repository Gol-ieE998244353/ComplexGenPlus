"""
train_flow.py - BRep Flow v12 训练脚本
[修改]
1. 从meta.pt加载归一化因子并传递给BrepFlow
2. [修改] 适配SimplePointEncoder (无PTv3依赖)
3. 可视化时使用count预测生成mask
"""
import argparse
import os
import gc
import logging
import math
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from flow_model import FlowConfig, BrepFlow, build_flow_system, load_topo_config_from_ckpt

def get_args():
    p = argparse.ArgumentParser("BRep Flow v13 Training")
    p.add_argument('--experiment_name', type=str, required=True)
    p.add_argument('--latent_folder', type=str, required=True)
    p.add_argument('--val_latent_folder', type=str, default=None)
    p.add_argument('--train_split', type=float, default=0.99)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--max_curves', type=int, default=200)
    p.add_argument('--max_patches', type=int, default=100)
    p.add_argument('--num_pc_tokens', type=int, default=64)
    p.add_argument('--max_topo', type=int, default=500)
    p.add_argument('--use_topo', action='store_true', default=False)
    p.add_argument('--vae_checkpoint', type=str, required=True)
    p.add_argument('--topo_ae_checkpoint', type=str, default=None,
                   help='TopoVAE checkpoint for topo loss computation')  # [修改] 新增
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--max_epochs', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_steps', type=int, default=10)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=0.5)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--time_sampling', type=str, default='none',
                   choices=['none', 'uniform', 'logit_normal', 'low_t_focus'])
    p.add_argument('--logit_mean', type=float, default=0.0)
    p.add_argument('--logit_std', type=float, default=1.0)
    p.add_argument('--time_warmup_epochs', type=int, default=10)
    p.add_argument('--low_t_dense_upper', type=float, default=0.3,
                   help='Upper bound for dense low-t sampling region')
    p.add_argument('--low_t_dense_ratio', type=float, default=0.8,
                   help='Sampling ratio allocated to low-t dense region')
    p.add_argument('--low_t_loss_boost', type=float, default=2.0,
                   help='Max multiplicative loss boost near t=0 when using low_t_focus')
    p.add_argument('--low_t_loss_gamma', type=float, default=1.0,
                   help='Decay power of low-t loss boost with increasing t')
    p.add_argument('--cfg_scale', type=float, default=2.0)
    p.add_argument('--p_uncond', type=float, default=0.1)
    p.add_argument('--num_inference_steps', type=int, default=50)
    p.add_argument('--simulator', type=str, default='euler',
                   choices=['euler', 'midpoint', 'rk4'],
                   help='Inference solver for generation and denoising')
    p.add_argument('--recon_loss_weight', type=float, default=0.0)
    p.add_argument('--padding_reg_weight', type=float, default=0.1)
    p.add_argument('--topo_loss_weight', type=float, default=1.0,
                   help='Weight for topology reconstruction loss')  # [修改] 新增
    p.add_argument('--bbox_loss_weight', type=float, default=0.0)
    p.add_argument('--valid-only', action='store_true',
                   help='Only apply valid-mask when computing velocity losses')
    p.add_argument('--use_reflow', action='store_true',
                   help='Enable reflow training to straighten early trajectories')
    p.add_argument('--reflow_steps', type=int, default=10,
                   help='Steps for reflow path integration during training')
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--vis_interval', type=int, default=5)
    p.add_argument('--vis_intermediate_ts', type=str, default='0.95,0.9,0.7,0.5,0.3,0.1')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--num_vis_samples', type=int, default=4)
    p.add_argument('--reset_scheduler', action='store_true', 
                   help='Reset scheduler when loading checkpoint')
    return p.parse_args()

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int):
    """
    1. 前 num_warmup_steps 步：线性增加到 base_lr
    2. 之后：一直保持 base_lr 不变
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class PrecomputedLatentDataset(Dataset):
    """预计算latent数据集 v13 - 无mask"""
    def __init__(self, latent_folder: str, max_curves: int, max_patches: int, use_topo: bool = False):
        self.max_curves = max_curves
        self.max_patches = max_patches
        self.use_topo = use_topo
        meta_path = os.path.join(latent_folder, 'meta.pt')
        if os.path.exists(meta_path):
            self.meta = torch.load(meta_path)
            print(f"[INFO] Meta: {self.meta.get('total_samples', 'N/A')} samples")
        else:
            self.meta = {}
        latent_files = sorted([
            os.path.join(latent_folder, f) for f in os.listdir(latent_folder)
            if f.startswith('latents_') and f.endswith('.pt')
        ])
        print(f"[INFO] Loading {len(latent_files)} files...")
        self.samples = []
        for fpath in tqdm(latent_files, desc="Loading"):
            self.samples.extend(torch.load(fpath))
        print(f"[INFO] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _pad_tensor(self, tensor: torch.Tensor, max_len: int) -> torch.Tensor:
        N = tensor.shape[0]
        if N >= max_len:
            return tensor[:max_len]
        padded = torch.zeros(max_len, *tensor.shape[1:], dtype=tensor.dtype)
        padded[:N] = tensor
        return padded
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {
            'curve_latent': self._pad_tensor(sample['curve_latent'], self.max_curves),
            'patch_latent': self._pad_tensor(sample['patch_latent'], self.max_patches),
            'curve_x0': self._pad_tensor(sample.get('curve_x0', sample['curve_latent']), self.max_curves),
            'patch_x0': self._pad_tensor(sample.get('patch_x0', sample['patch_latent']), self.max_patches),
            # [修改] 保留n_valid用于padding loss
            'n_valid_curves': sample.get('n_valid_curves', self.max_curves),
            'n_valid_patches': sample.get('n_valid_patches', self.max_patches),
        }
        if 'pc_features' in sample:
            result['pc_features'] = sample['pc_features']
        if self.use_topo and 'topo_latent' in sample:
            result['topo_latent'] = sample['topo_latent']
        if self.use_topo and 'topo_x0' in sample:
            result['topo_x0'] = sample['topo_x0']
        return result



def split_dataset(dataset: Dataset, train_ratio: float, seed: int = 42):
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_subset = Subset(dataset, indices[:n_train])
    val_subset = Subset(dataset, indices[n_train:])
    print(f"[INFO] Split: {n_train} train, {n_total - n_train} val")
    return train_subset, val_subset

def collate_fn_varlen_pc(batch):
    """拼接变长点云，添加batch_idx"""
    result = {}
    for key in ['curve_latent', 'patch_latent', 'topo_latent', 'curve_x0', 'patch_x0', 'topo_x0']:
        if key in batch[0]:
            result[key] = torch.stack([b[key] for b in batch])
    result['n_valid_curves'] = torch.tensor([b['n_valid_curves'] for b in batch])
    result['n_valid_patches'] = torch.tensor([b['n_valid_patches'] for b in batch])
    # [修改] 处理topo相关数据
    if 'topo_data' in batch[0]:
        result['n_valid_topo'] = torch.tensor([b['n_valid_topo'] for b in batch])
        topo_keys = ['mask', 'curve_idx', 'is_closed', 'is_boundary', 'patch0', 'patch1',
                     'adj_p0_e0', 'adj_p0_e1', 'adj_p1_e0', 'adj_p1_e1',
                     'adj_p0_e0_valid', 'adj_p0_e1_valid', 'adj_p1_e0_valid', 'adj_p1_e1_valid']
        result['topo_data'] = {k: torch.stack([b['topo_data'][k] for b in batch]) for k in topo_keys}
    if 'pc_features' in batch[0]:
        pc_list = []
        batch_idx_list = []
        for i, b in enumerate(batch):
            pc = b['pc_features']
            pc_list.append(pc)
            batch_idx_list.append(torch.full((pc.shape[0],), i, dtype=torch.long))
        result['pc_features'] = torch.cat(pc_list, dim=0)
        result['pc_batch_idx'] = torch.cat(batch_idx_list, dim=0)
    return result

class FlowTrainer:
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if world_size > 1 else ('cuda' if torch.cuda.is_available() else 'cpu')
        if world_size > 1:
            torch.cuda.set_device(rank)
        self.exp_dir = os.path.join('experiments', f'{args.experiment_name}_flow_v13')
        self.ckpt_dir = os.path.join(self.exp_dir, 'ckpt')
        self.vis_dir = os.path.join(self.exp_dir, 'vis')
        if rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
        self.logger = self._setup_logger() if rank == 0 else None
        meta_path = os.path.join(args.latent_folder, 'meta.pt')
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            curve_dim = meta.get('curve_latent_dim', 16)
            patch_dim = meta.get('patch_latent_dim', 32)
            pc_feat_dim = meta.get('pc_feature_dim', 7)
            topo_dim = meta.get('topo_latent_dim', 128)  # [修改] 读取topo_latent_dim
        else:
            curve_dim, patch_dim, pc_feat_dim, topo_dim = 16, 32, 7, 128
        if args.use_topo and args.topo_ae_checkpoint:
            topo_cfg = load_topo_config_from_ckpt(args.topo_ae_checkpoint)
            topo_dim = topo_cfg.HIDDEN_DIM
        # [修改] 构建FlowConfig
        self.config = FlowConfig(
            CURVE_LATENT_DIM=curve_dim,
            PATCH_LATENT_DIM=patch_dim,
            TOPO_LATENT_DIM=topo_dim,  # [修改]
            PC_FEATURE_DIM=pc_feat_dim,
            MAX_CURVES=args.max_curves,
            MAX_PATCHES=args.max_patches,
            MAX_TOPO=args.max_topo,  # [修改]
            NUM_PC_TOKENS=args.num_pc_tokens,
            D_MODEL=args.d_model,
            NHEAD=args.nhead,
            NUM_LAYERS=args.num_layers,
            NUM_INFERENCE_STEPS=args.num_inference_steps,
            INFERENCE_SOLVER=args.simulator,
            CFG_SCALE=args.cfg_scale,
            P_UNCOND=args.p_uncond,
            LOGIT_MEAN=args.logit_mean,
            LOGIT_STD=args.logit_std,
            TIME_SAMPLING_WARMUP_EPOCHS=args.time_warmup_epochs,
            LOW_T_DENSE_UPPER=args.low_t_dense_upper,
            LOW_T_DENSE_RATIO=args.low_t_dense_ratio,
            LOW_T_LOSS_BOOST=args.low_t_loss_boost,
            LOW_T_LOSS_GAMMA=args.low_t_loss_gamma,
            RECON_LOSS_WEIGHT=args.recon_loss_weight,
            PADDING_REG_WEIGHT=args.padding_reg_weight,
            BBOX_LOSS_WEIGHT=args.bbox_loss_weight,
            TOPO_LOSS_WEIGHT=args.topo_loss_weight,  # [修改] 新增
            VALID_ONLY=args.valid_only,
        )
        if self.logger:
            self.logger.info(f"Config: curves={args.max_curves}, patches={args.max_patches}, topo={args.max_topo}")
            self.logger.info(f"Loss weights: padding_reg={args.padding_reg_weight}, topo={args.topo_loss_weight}")
            self.logger.info(f"Valid-only loss mask: {args.valid_only}")
        # [修改] 传递topo_ae_checkpoint
        self.model = build_flow_system(self.config, args.vae_checkpoint, args.use_topo, 
                                       args.topo_ae_checkpoint).to(self.device)
        if args.compile and hasattr(torch, 'compile'):
            self.model.flow_model = torch.compile(self.model.flow_model, mode='reduce-overhead')
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        if self.logger:
            total = sum(p.numel() for p in self.model.parameters())
            train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Params: {total:,} total, {train:,} trainable")
        params = self.model.module.parameters() if world_size > 1 else self.model.parameters()
        self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        self.scaler = GradScaler(enabled=args.use_amp)
        self.train_loader, self.val_loader, self.train_sampler, _ = self._build_dataloaders()
        total_steps = args.max_epochs * len(self.train_loader) // args.grad_accum
        self.scheduler = get_constant_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.warmup_steps,
        )
        self.start_epoch, self.best_loss, self.global_step = 0, float('inf'), 0
        if args.checkpoint and os.path.exists(args.checkpoint):
            self._load_checkpoint(args.checkpoint)
        self.vis_intermediate_ts = [float(x) for x in args.vis_intermediate_ts.split(',')]
        self.visualizer = None
        if rank == 0:
            try:
                from vis import LatentVisualizer
                self.visualizer = LatentVisualizer(args.vae_checkpoint, self.device)
                if self.logger:
                    self.logger.info("LatentVisualizer initialized")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to init visualizer: {e}")
        if rank == 0 and HAS_WANDB:
            wandb.init(project='brep-flow-v13', name=args.experiment_name, config=vars(args))
    
    def _setup_logger(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.exp_dir, f'train_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _build_dataloaders(self):
        args = self.args
        full_dataset = PrecomputedLatentDataset(
            args.latent_folder, args.max_curves, args.max_patches, args.use_topo
        )
        if args.val_latent_folder:
            train_dataset = full_dataset
            val_dataset = PrecomputedLatentDataset(
                args.val_latent_folder, args.max_curves, args.max_patches, args.use_topo
            )
        else:
            train_dataset, val_dataset = split_dataset(full_dataset, args.train_split, args.split_seed)
        train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank, shuffle=True) if self.world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, self.world_size, self.rank, shuffle=False) if self.world_size > 1 else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  num_workers=0, pin_memory=True, drop_last=True,
                                  collate_fn=collate_fn_varlen_pc)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, sampler=val_sampler,
                                num_workers=0, pin_memory=True, drop_last=False,
                                collate_fn=collate_fn_varlen_pc)
        return train_loader, val_loader, train_sampler, val_sampler
    
    def _load_checkpoint(self, path):
        if self.logger:
            self.logger.info(f"Loading: {path}")
        ckpt = torch.load(path, map_location=self.device)
        m = self.model.module if self.world_size > 1 else self.model
        m.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except:
                pass
        if 'scheduler' in ckpt and not self.args.reset_scheduler:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except:
                pass
        elif self.args.reset_scheduler and self.logger:
            self.logger.info(f"Scheduler reset with new warmup={self.args.warmup_steps}")
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        del ckpt
        clear_memory()
    
    def _save_checkpoint(self, epoch, is_best=False):
        if self.rank != 0:
            return
        m = self.model.module if self.world_size > 1 else self.model
        state = {
            'epoch': epoch, 'global_step': self.global_step,
            'model': m.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        torch.save(state, os.path.join(self.ckpt_dir, f'epoch_{epoch+1}.pth'))
        if is_best:
            torch.save(state, os.path.join(self.ckpt_dir, 'best.pth'))
    
    def train_epoch(self, epoch):
        self.model.train()
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        # [修改] 更新loss字典
        losses_sum = {'total': 0.0, 'curve_vel_loss': 0.0, 'patch_vel_loss': 0.0, 
                      'padding_loss': 0.0, 'topo_vel_loss': 0.0, 'topo_recon_loss': 0.0,
                      'small_t_weight_mean': 0.0}
        n_batch = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=self.rank != 0)
        for batch_idx, batch in enumerate(pbar):
            try:
                batch_gpu = {k: v.to(self.device, non_blocking=True) 
                            for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # [修改] 处理嵌套的topo_data
                if 'topo_data' in batch:
                    batch_gpu['topo_data'] = {k: v.to(self.device, non_blocking=True) 
                                             for k, v in batch['topo_data'].items()}
                with autocast(enabled=self.args.use_amp):
                    m = self.model.module if self.world_size > 1 else self.model
                    losses = m.training_step(
                        batch_gpu,
                        self.args.time_sampling,
                        epoch,
                        use_reflow=self.args.use_reflow,
                        reflow_steps=self.args.reflow_steps,
                    )
                    loss = losses['total'] / self.args.grad_accum
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
                    if self.rank == 0 and self.global_step % 100 == 0 and self.logger:
                        loss_str = ", ".join([f"{k}={v.item():.4f}" if torch.is_tensor(v) else f"{k}={v:.4f}" 
                                             for k, v in losses.items() if k not in ['total', 't_mean']])
                        self.logger.info(f"Step {self.global_step}: grad={grad_norm:.4f}, {loss_str}")
                for k in losses_sum:
                    if k in losses:
                        v = losses[k]
                        losses_sum[k] += v.item() if torch.is_tensor(v) else v
                n_batch += 1
                if self.rank == 0:
                    cur_lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{losses['total'].item():.4f}", lr=f"{cur_lr:.2e}")
                if batch_idx % 50 == 0:
                    clear_memory()
            except RuntimeError as e:
                raise RuntimeError(f"Error at batch {batch_idx}: {e}")
        return {k: v / max(n_batch, 1) for k, v in losses_sum.items()}
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        losses_sum = {'total': 0.0, 'curve_vel_loss': 0.0, 'patch_vel_loss': 0.0, 
                      'padding_loss': 0.0, 'topo_vel_loss': 0.0, 'topo_recon_loss': 0.0,
                      'small_t_weight_mean': 0.0}  # [修改]
        n_batch = 0
        for batch in tqdm(self.val_loader, desc="Val", disable=self.rank != 0):
            try:
                batch_gpu = {k: v.to(self.device, non_blocking=True) 
                            for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # [修改] 处理嵌套的topo_data
                if 'topo_data' in batch:
                    batch_gpu['topo_data'] = {k: v.to(self.device, non_blocking=True) 
                                             for k, v in batch['topo_data'].items()}
                with autocast(enabled=self.args.use_amp):
                    m = self.model.module if self.world_size > 1 else self.model
                    losses = m.training_step(
                        batch_gpu,
                        'uniform',
                        epoch,
                        use_reflow=self.args.use_reflow,
                        reflow_steps=self.args.reflow_steps,
                    )
                if not torch.isnan(losses['total']):
                    for k in losses_sum:
                        if k in losses:
                            v = losses[k]
                            losses_sum[k] += v.item() if torch.is_tensor(v) else v
                    n_batch += 1
            except:
                continue
            finally:
                clear_memory()
        val_metrics = {k: v / max(n_batch, 1) for k, v in losses_sum.items()}

        probe_ts = [0.1, 0.3, 0.7, 0.9]
        for t_probe in probe_ts:
            probe_sum = 0.0
            probe_count = 0
            for batch in self.val_loader:
                try:
                    batch_gpu = {k: v.to(self.device, non_blocking=True)
                                for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    if 'topo_data' in batch:
                        batch_gpu['topo_data'] = {k: v.to(self.device, non_blocking=True)
                                                 for k, v in batch['topo_data'].items()}
                    with autocast(enabled=self.args.use_amp):
                        m = self.model.module if self.world_size > 1 else self.model
                        losses = m.training_step(
                            batch_gpu,
                            'uniform',
                            epoch,
                            use_reflow=self.args.use_reflow,
                            reflow_steps=self.args.reflow_steps,
                            forced_t=t_probe,
                        )
                    if not torch.isnan(losses['total']):
                        probe_sum += losses['total'].item() if torch.is_tensor(losses['total']) else float(losses['total'])
                        probe_count += 1
                except:
                    continue
                finally:
                    clear_memory()
            key = f'total_t{str(t_probe).replace(".", "p")}'
            val_metrics[key] = probe_sum / max(probe_count, 1)

        return val_metrics
    def _collect_vis_samples(self, num_samples: int) -> List[Dict]:
        """收集样本用于可视化"""
        collected = []
        for batch in self.val_loader:
            batch_gpu = {k: v.to(self.device, non_blocking=True) 
                        for k, v in batch.items() if isinstance(v, torch.Tensor)}
            B = batch_gpu['curve_latent'].shape[0]
            pc_batch_idx = batch_gpu.get('pc_batch_idx')
            pc_features = batch_gpu.get('pc_features')
            for i in range(B):
                sample = {
                    'curve_latent': batch_gpu['curve_latent'][i].cpu(),
                    'patch_latent': batch_gpu['patch_latent'][i].cpu(),
                    'n_valid_curves': batch_gpu['n_valid_curves'][i].item(),
                    'n_valid_patches': batch_gpu['n_valid_patches'][i].item(),
                }
                if pc_features is not None and pc_batch_idx is not None:
                    mask = pc_batch_idx == i
                    sample['pc_features'] = pc_features[mask].cpu()
                collected.append(sample)
                if len(collected) >= num_samples:
                    return collected
        return collected
    
    @torch.no_grad()
    def visualize(self, epoch):
        """可视化多个样本"""
        if self.rank != 0 or self.visualizer is None:
            return
        self.model.eval()
        m = self.model.module if self.world_size > 1 else self.model
        vis_epoch_dir = Path(self.vis_dir) / f'epoch_{epoch+1}'
        vis_epoch_dir.mkdir(parents=True, exist_ok=True)
        samples = self._collect_vis_samples(self.args.num_vis_samples)
        if len(samples) == 0:
            return
        actual_num = min(len(samples), self.args.num_vis_samples)
        if self.logger:
            self.logger.info(f"Visualizing {actual_num} samples")
        for sample_idx, sample in enumerate(samples[:actual_num]):
            try:
                sample_dir = vis_epoch_dir / f'sample_{sample_idx:02d}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                curve_lat = sample['curve_latent'].unsqueeze(0).to(self.device)
                patch_lat = sample['patch_latent'].unsqueeze(0).to(self.device)
                x_1 = {'curve': curve_lat, 'patch': patch_lat}
                # 点云条件
                cond_tokens = None
                pc_feat = sample.get('pc_features')
                if pc_feat is not None:
                    pc_feat = pc_feat.to(self.device)
                    pc_batch_idx = torch.zeros(pc_feat.shape[0], dtype=torch.long, device=self.device)
                    pc_out = m.pc_encoder(pc_feat, pc_batch_idx, 1)
                    cond_tokens = pc_out['tokens']
                # 固定噪声
                torch.manual_seed(42 + sample_idx)
                x_0 = {k: torch.randn_like(v) for k, v in x_1.items()}
                # 从不同t_start去噪
                for t_start in self.vis_intermediate_ts:
                    x_t = {k: (1 - t_start) * x_0[k] + t_start * x_1[k] for k in x_1}
                    result = m.denoise_from_t(
                        x_t,
                        t_start,
                        cond_tokens,
                        self.args.num_inference_steps,
                        simulator=self.args.simulator,
                    )
                    sample_out = {
                        'curve_latent': result['curve'][0].cpu(), 
                        'patch_latent': result['patch'][0].cpu(),
                    }
                    name = f'from_t_{t_start:.2f}'.replace('.', 'p')
                    self.visualizer.decode_and_visualize(sample_out, sample_dir / f'{name}.obj')
                # 纯生成
                if pc_feat is not None:
                    gen_out = m.generate(
                        pc_feat,
                        pc_batch_idx,
                        batch_size=1,
                        num_steps=self.args.num_inference_steps,
                        simulator=self.args.simulator,
                    )
                    sample_gen = {
                        'curve_latent': gen_out['curve_latent'][0].cpu(), 
                        'patch_latent': gen_out['patch_latent'][0].cpu(),
                    }
                    self.visualizer.decode_and_visualize(sample_gen, sample_dir / 'generated.obj')
                # GT
                sample_gt = {
                    'curve_latent': curve_lat[0].cpu(), 
                    'patch_latent': patch_lat[0].cpu(),
                }
                self.visualizer.decode_and_visualize(sample_gt, sample_dir / 'gt.obj')
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Vis failed for sample {sample_idx}: {e}")
        if self.logger:
            self.logger.info(f"Saved {actual_num} samples to {vis_epoch_dir}")
    
    def train(self):
        if self.logger:
            self.logger.info(f"Start epoch {self.start_epoch}, GPUs: {self.world_size}")
            self.logger.info(f"Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")
        for epoch in range(self.start_epoch, self.args.max_epochs):
            train_losses = self.train_epoch(epoch)
            if self.rank == 0:
                loss_str = " | ".join([f"{k}={v:.4f}" for k, v in train_losses.items()])
                self.logger.info(f"Epoch {epoch+1} Train: {loss_str}")
                if HAS_WANDB:
                    wandb.log({'epoch': epoch+1, **{f'train/{k}': v for k, v in train_losses.items()}})
            if (epoch + 1) % self.args.eval_interval == 0:
                val_losses = self.validate(epoch)
                if self.rank == 0:
                    loss_str = " | ".join([f"{k}={v:.4f}" for k, v in val_losses.items()])
                    self.logger.info(f"Epoch {epoch+1} Val: {loss_str}")
                    if HAS_WANDB:
                        wandb.log({'epoch': epoch+1, **{f'val/{k}': v for k, v in val_losses.items()}})
                    if val_losses['total'] < self.best_loss:
                        self.best_loss = val_losses['total']
                        self._save_checkpoint(epoch, is_best=True)
            if self.rank == 0 and (epoch + 1) % self.args.vis_interval == 0:
                self.visualize(epoch)
            if self.rank == 0 and (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch)
            clear_memory()
        if self.rank == 0 and HAS_WANDB:
            wandb.finish()


def train_worker(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group('nccl', init_method='tcp://127.0.0.1:23459', world_size=world_size, rank=rank)
    FlowTrainer(args, rank, world_size).train()
    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = get_args()
    world_size = torch.cuda.device_count()
    print(f"BRep Flow v13 | PointNet Encoder | GPUs: {world_size}")
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)
    else:
        train_worker(0, 1, args)

if __name__ == '__main__':
    main()