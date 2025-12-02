"""
train_topo.py - 拓扑VAE训练
"""

import argparse, os, logging, gc
from datetime import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import wandb

from halfedge import HalfEdgeBuilder
from topo_encoder import TopoVAE, TopoConfig, compute_topo_loss, KLScheduler


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    
    p.add_argument('--data_folder', type=str, default='data/default/train')
    p.add_argument('--val_folder', type=str, default='data/train_small')
    p.add_argument('--quicktest', action='store_true')
    p.add_argument('--patch_grid', action='store_true', default=True)
    p.add_argument('--points_per_patch_dim', type=int, default=20)
    p.add_argument('--rotation_augment', action='store_true')
    p.add_argument('--noise', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=2)
    
    p.add_argument('--curve_patch_checkpoint', type=str, required=True)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_halfedges', type=int, default=50000)
    
    p.add_argument('--kl_warmup', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=50)
    p.add_argument('--eval_interval', type=int, default=10)
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--use_amp', action='store_true', default=True)
    return p.parse_args()


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_frozen_encoders(ckpt_path: str, device: str):
    from train_pc_nm import CurveEncoder, PatchEncoder, Config
    cfg = Config()
    curve_enc, patch_enc = CurveEncoder(cfg), PatchEncoder(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
    curve_enc.load_state_dict(clean(ckpt['curve_encoder']))
    patch_enc.load_state_dict(clean(ckpt['patch_encoder']))
    del ckpt; clear_mem()
    for m in [curve_enc, patch_enc]:
        m.to(device).eval()
        for p in m.parameters(): p.requires_grad = False
    return curve_enc, patch_enc, cfg


@torch.no_grad()
def encode_cp(curves, patches, curve_enc, patch_enc, device, use_amp):
    def to_dev(d): 
        return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    c, p = to_dev(curves), to_dev(patches)
    with autocast(enabled=use_amp):
        cm, _ = curve_enc(
            c['curve_points'], c['endpoints'], c['is_closed'],
            c['labels'], c['scale'], c['center'], c['mask']
        )
        pm, _ = patch_enc(
            p['patch_points'], p['patch_normals'], p['u_closed'],
            p['v_closed'], p['labels'], p['scale'], p['center'], p['mask']
        )
    return cm.float(), pm.float()


def prepare_target(he_data, curve_lat, patch_lat, device):
    c_idx = he_data['curve_idx'].clamp(min=0).to(device)
    p_idx = he_data['patch_idx'].clamp(min=0).to(device)
    return {
        'curve_latent': curve_lat.gather(1, c_idx.unsqueeze(-1).expand(-1, -1, curve_lat.shape[-1])),
        'patch_latent': patch_lat.gather(1, p_idx.unsqueeze(-1).expand(-1, -1, patch_lat.shape[-1]))
    }


def train_epoch(model, loader, opt, scaler, curve_enc, patch_enc, he_builder, kl_w, cfg, device, rank, args):
    model.train()
    losses_sum, n_batch = {}, 0
    opt.zero_grad(set_to_none=True)
    
    for i, (curves, patches) in enumerate(tqdm(loader, disable=rank != 0)):
        if curves is None: continue
        try:
            he = he_builder.build_batch(curves, patches)
            if he['mask'].shape[1] > args.max_halfedges:
                he = {k: v[:, :args.max_halfedges] if isinstance(v, torch.Tensor) and v.dim() > 1 else v for k, v in he.items()}
            he = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in he.items()}
            if he['mask'].sum() == 0: continue
            
            c_lat, p_lat = encode_cp(curves, patches, curve_enc, patch_enc, device, args.use_amp)
            target = prepare_target(he, c_lat, p_lat, device)
            
            with autocast(enabled=args.use_amp):
                pred = model(c_lat, p_lat, he)
                losses = compute_topo_loss(pred, target, he, kl_w, cfg)
                loss = losses['total'] / args.grad_accum
            
            if torch.isnan(loss): opt.zero_grad(set_to_none=True); continue
            
            scaler.scale(loss).backward() if args.use_amp else loss.backward()
            
            if (i + 1) % args.grad_accum == 0:
                if args.use_amp: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt) if args.use_amp else opt.step()
                if args.use_amp: scaler.update()
                opt.zero_grad(set_to_none=True)
            
            for k, v in losses.items(): losses_sum[k] = losses_sum.get(k, 0) + v.item()
            n_batch += 1
            if i % 50 == 0: clear_mem()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower(): clear_mem(); opt.zero_grad(set_to_none=True); continue
            raise
    return {k: v / max(n_batch, 1) for k, v in losses_sum.items()}


@torch.no_grad()
def validate(model, loader, curve_enc, patch_enc, he_builder, kl_w, cfg, device, args):
    model.eval()
    losses_sum, n = {}, 0
    for curves, patches in loader:
        if curves is None: continue
        try:
            he = he_builder.build_batch(curves, patches)
            if he['mask'].shape[1] > args.max_halfedges:
                he = {k: v[:, :args.max_halfedges] if isinstance(v, torch.Tensor) and v.dim() > 1 else v for k, v in he.items()}
            he = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in he.items()}
            if he['mask'].sum() == 0: continue
            c_lat, p_lat = encode_cp(curves, patches, curve_enc, patch_enc, device, args.use_amp)
            target = prepare_target(he, c_lat, p_lat, device)
            with autocast(enabled=args.use_amp):
                pred = model(c_lat, p_lat, he)
                losses = compute_topo_loss(pred, target, he, kl_w, cfg)
            for k, v in losses.items(): losses_sum[k] = losses_sum.get(k, 0) + v.item()
            n += 1
        except: continue
        finally: clear_mem()
    return {k: v / max(n, 1) for k, v in losses_sum.items()}


def train(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group('nccl', init_method='tcp://127.0.0.1:23456', world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    exp_dir = os.path.join('experiments', args.experiment_name + '_topo')
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(f'{exp_dir}/train.log'), logging.StreamHandler()])
        logger = logging.getLogger()
        wandb.init(project='topo-vae', name=args.experiment_name, config=vars(args))
    
    from data_loader_optimized import train_data_loader_clean
    data_dir = 'data/train_small' if args.quicktest else args.data_folder
    val_dir = 'data/train_small' if args.quicktest else args.val_folder
    train_loader, train_sampler = train_data_loader_clean(
        args.batch_size, data_dir,
        rotation_augmentation=args.rotation_augment,
        flag_noise=args.noise,
        flag_grid=args.patch_grid,
        dim_grid=args.points_per_patch_dim,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size
    )
    val_loader, _ = train_data_loader_clean(
        args.batch_size, val_dir,
        rotation_augmentation=False,
        flag_noise=0,
        flag_grid=args.patch_grid,
        dim_grid=args.points_per_patch_dim,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size
    )
    
    curve_enc, patch_enc, base_cfg = load_frozen_encoders(args.curve_patch_checkpoint, device)
    
    topo_cfg = TopoConfig(CURVE_LATENT_DIM=base_cfg.CURVE_LATENT_DIM, PATCH_LATENT_DIM=base_cfg.PATCH_LATENT_DIM,
                          HIDDEN_DIM=args.hidden_dim, LATENT_DIM=args.latent_dim, NUM_LAYERS=args.num_layers,
                          DROPOUT=args.dropout, KL_WARMUP_EPOCHS=args.kl_warmup)
    model = TopoVAE(topo_cfg).to(device)
    if world_size > 1: model = DDP(model, device_ids=[rank])
    
    if rank == 0: logger.info(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    kl_sched = KLScheduler(topo_cfg.KL_WEIGHT_START, topo_cfg.KL_WEIGHT_END, topo_cfg.KL_WARMUP_EPOCHS)
    he_builder = HalfEdgeBuilder(endpoint_tol=1e-4)
    
    start_epoch, best_loss = 0, float('inf')
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        (model.module if world_size > 1 else model).load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()})
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        del ckpt; clear_mem()
    
    for epoch in range(start_epoch, args.max_epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        kl_w = kl_sched.get_weight(epoch)
        
        train_losses = train_epoch(model, train_loader, opt, scaler, curve_enc, patch_enc, he_builder, kl_w, topo_cfg, device, rank, args)
        
        if rank == 0:
            logger.info(f'Epoch {epoch+1} train: ' + ' '.join(f'{k}={v:.4f}' for k, v in train_losses.items()))
            wandb.log({f'train/{k}': v for k, v in train_losses.items()}, step=epoch)
        
        if (epoch + 1) % args.eval_interval == 0:
            val_losses = validate(model, val_loader, curve_enc, patch_enc, he_builder, kl_w, topo_cfg, device, args)
            if rank == 0:
                logger.info(f'Epoch {epoch+1} val: ' + ' '.join(f'{k}={v:.4f}' for k, v in val_losses.items()))
                wandb.log({f'val/{k}': v for k, v in val_losses.items()}, step=epoch)
                if val_losses.get('total', float('inf')) < best_loss:
                    best_loss = val_losses['total']
                    torch.save({'epoch': epoch, 'model': (model.module if world_size > 1 else model).state_dict(),
                                'optimizer': opt.state_dict(), 'best_loss': best_loss, 'config': topo_cfg},
                               os.path.join(ckpt_dir, 'best.pth'))
        
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            torch.save({'epoch': epoch, 'model': (model.module if world_size > 1 else model).state_dict(),
                        'optimizer': opt.state_dict(), 'config': topo_cfg}, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth'))
        clear_mem()
    
    if rank == 0: wandb.finish()
    if world_size > 1: dist.destroy_process_group()


def main():
    args = get_args()
    world_size = torch.cuda.device_count()
    print(f'TopoVAE Training | GPUs: {world_size} | {args.experiment_name}')
    if world_size > 1:
        mp.spawn(train, args=(world_size, args), nprocs=world_size)
    else:
        train(0, 1, args)


if __name__ == '__main__':
    main()