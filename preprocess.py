"""
preprocess.py - BRep数据预处理
python preprocess.py --data_folder data/default/train --output_folder latents/ \
       --vae_checkpoint experiments/aa/ckpt/checkpoint_epoch_1700.pth \
       --topo_ae_checkpoint experiments/aa/ckpt/checkpoint_epoch_1700.pth  \
       --visualize --num_vis_samples 10
"""
import argparse
import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

MAX_CURVES = 200
MAX_PATCHES = 100
MAX_TOPO = 1000  # [修改] 新增

def get_args():
    p = argparse.ArgumentParser("Preprocess BRep data")
    p.add_argument('--data_folder', type=str, required=True)
    p.add_argument('--output_folder', type=str, required=True)
    p.add_argument('--vae_checkpoint', type=str, required=True)
    p.add_argument('--topo_ae_checkpoint', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--voxel_dim', type=int, default=128)
    p.add_argument('--points_per_patch_dim', type=int, default=20)
    p.add_argument('--topo_hidden_dim', type=int, default=128)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--samples_per_file', type=int, default=2000)
    p.add_argument('--visualize', action='store_true', default=False)
    p.add_argument('--num_vis_samples', type=int, default=10)
    p.add_argument('--vis_output_dir', type=str, default=None)
    p.add_argument('--deterministic', action='store_true', default=False,
                   help='Use mean instead of reparameterized sampling for latent')
    p.add_argument('--padding_zero_tol', type=float, default=1e-6,
                   help='Tolerance for padding zero check')
    p.add_argument('--max_topo', type=int, default=1000,
                   help='Maximum number of topo edges to pad')  # [修改] 新增
    return p.parse_args()

def move_to_device(data, device):
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(x, device) for x in data)
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data

def pad_to_length(latent, target_length):
    """Pad latent到固定长度，padding部分用0填充"""
    current_length = latent.shape[0]
    if current_length >= target_length:
        return latent[:target_length]
    pad_shape = (target_length - current_length, latent.shape[1])
    padding = torch.zeros(pad_shape, dtype=latent.dtype, device=latent.device)
    return torch.cat([latent, padding], dim=0)

class BRepPreprocessor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.skip_topo = (args.topo_ae_checkpoint is None)  # [修改]
        self.deterministic = getattr(args, 'deterministic', False)
        self.padding_zero_tol = getattr(args, 'padding_zero_tol', 1e-6)
        self.max_topo = getattr(args, 'max_topo', 1000)  # [修改] 新增
        print(f"[INFO] Device: {self.device}, skip_topo: {self.skip_topo}, deterministic: {self.deterministic}")
        print(f"[INFO] MAX_TOPO: {self.max_topo}")  # [修改] 新增
        self._load_vae(args.vae_checkpoint)
        if not self.skip_topo:
            self._load_topo_ae(args.topo_ae_checkpoint)
        else:
            self.topo_ae = None
        self._init_topo_builder()
        self.global_max_curves = 0
        self.global_max_patches = 0
        self.global_max_topo = 0
        self.global_max_pc_points = 0
        self.padding_violation_count = 0

    def _load_vae(self, ckpt_path):
        from train_pc_nm import CurveEncoder, CurveDecoder, PatchEncoder, PatchDecoder, Config
        cfg = Config()
        self.curve_encoder = CurveEncoder(cfg).to(self.device).eval()
        self.patch_encoder = PatchEncoder(cfg).to(self.device).eval()
        self.curve_decoder = CurveDecoder(cfg).to(self.device).eval()
        self.patch_decoder = PatchDecoder(cfg).to(self.device).eval()
        ckpt = torch.load(ckpt_path, map_location=self.device)
        clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
        self.curve_encoder.load_state_dict(clean(ckpt.get('curve_encoder', {})))
        self.patch_encoder.load_state_dict(clean(ckpt.get('patch_encoder', {})))
        self.curve_decoder.load_state_dict(clean(ckpt.get('curve_decoder', {})))
        self.patch_decoder.load_state_dict(clean(ckpt.get('patch_decoder', {})))
        for m in [self.curve_encoder, self.patch_encoder, self.curve_decoder, self.patch_decoder]:
            for p in m.parameters():
                p.requires_grad = False
        print(f"[INFO] Loaded VAE from {ckpt_path}")

    def _load_topo_ae(self, ckpt_path):
        from topo_encoder import TopoVAE, TopoConfig  # [修改] TopoVAE
        topo_cfg = TopoConfig(HIDDEN_DIM=self.args.topo_hidden_dim)
        self.topo_ae = TopoVAE(topo_cfg).to(self.device).eval()  # [修改]
        ckpt = torch.load(ckpt_path, map_location=self.device)
        clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
        self.topo_ae.load_state_dict(clean(ckpt.get('model', ckpt)))
        for p in self.topo_ae.parameters():
            p.requires_grad = False
        self.topo_latent_dim = topo_cfg.HIDDEN_DIM  # [修改] 保存
        print(f"[INFO] Loaded TopoVAE from {ckpt_path}, latent_dim={self.topo_latent_dim}")  # [修改]

    def _init_topo_builder(self):
        if self.skip_topo:
            self.topo_builder = None
            return
        try:
            from topo_edge import TopoEdgeBuilder
            self.topo_builder = TopoEdgeBuilder()
        except ImportError:
            self.topo_builder = None

    def _reparameterize(self, mean, logvar):
        if self.deterministic:
            return mean
        std = torch.exp(0.5 * logvar).clamp(min=1e-8, max=10)
        eps = torch.randn_like(std)
        z = mean + eps * std * 0
        return z

    @torch.no_grad()
    def encode_curves(self, curves):
        curve_mean, curve_logvar = self.curve_encoder(
            curves['curve_points'], curves['endpoints'], curves['is_closed'],
            curves['labels'], curves['mask']
        )
        return self._reparameterize(curve_mean, curve_logvar)

    @torch.no_grad()
    def encode_patches(self, patches):
        patch_mean, patch_logvar = self.patch_encoder(
            patches['patch_points'], patches['patch_normals'], patches['u_closed'],
            patches['v_closed'], patches['labels'], patches['mask']
        )
        return patch_mean

    @torch.no_grad()
    def decode_curves(self, curve_latent):
        return self.curve_decoder(curve_latent)

    @torch.no_grad()
    def decode_patches(self, patch_latent):
        return self.patch_decoder(patch_latent)

    @torch.no_grad()
    def encode_topo(self, curve_latent, patch_latent, topo_data):
        if topo_data is None or self.topo_ae is None:
            return None, None
        mu, logvar = self.topo_ae.encoder(curve_latent, patch_latent, topo_data)  # [修改] 使用encoder
        if self.deterministic:
            topo_repr = mu
        else:
            topo_repr = self.topo_ae.reparameterize(mu, logvar)  # [修改]
        return topo_repr, topo_data['mask']

    def build_topo_data(self, curves, patches):
        if self.topo_builder is None:
            return None
        try:
            return self.topo_builder.build_batch(curves, patches)
        except Exception as e:
            print(f"[ERROR] TopoEdge build failed: {e}")
            return None

    def process_pointcloud(self, pc_data, batch_size):
        if pc_data is None:
            return None
        locations, features = pc_data
        batch_idx = locations[:, -1].long()
        results = []
        for b in range(batch_size):
            mask = batch_idx == b
            feat_b = features[mask].cpu()
            self.global_max_pc_points = max(self.global_max_pc_points, feat_b.shape[0])
            results.append(feat_b)
        return results

    def _verify_padding_zero(self, latent, n_valid, name, sample_idx):
        """验证padding位置是否为0，不为0则报错"""
        if n_valid >= latent.shape[0]:
            return True
        padding_part = latent[n_valid:]
        max_val = padding_part.abs().max().item()
        if max_val > self.padding_zero_tol:
            self.padding_violation_count += 1
            raise ValueError(
                f"[ERROR] Sample {sample_idx}: {name} padding not zero! "
                f"n_valid={n_valid}, padding max_abs={max_val:.6f} > tol={self.padding_zero_tol}"
            )
        return True

    @torch.no_grad()
    def process_batch(self, batch_idx, pc_data, curves, patches, save_gt_for_vis=False):
        B = curves['mask'].shape[0]
        curves = move_to_device(curves, self.device)
        patches = move_to_device(patches, self.device)
        curve_latent = self.encode_curves(curves)
        patch_latent = self.encode_patches(patches)
        topo_repr, topo_mask, topo_data_batch = None, None, None  # [修改] 新增topo_data_batch
        if not self.skip_topo:
            topo_data = self.build_topo_data(curves, patches)
            if topo_data is not None:
                topo_data = move_to_device(topo_data, self.device)
                topo_repr, topo_mask = self.encode_topo(curve_latent, patch_latent, topo_data)
                topo_data_batch = topo_data  # [修改] 保存
        pc_features_list = None
        if pc_data is not None:
            loc = pc_data[0].to(self.device)
            feat = pc_data[1].to(self.device)
            pc_features_list = self.process_pointcloud((loc, feat), B)
        n_curves = curves['mask'].sum(dim=1).max().item()
        n_patches = patches['mask'].sum(dim=1).max().item()
        self.global_max_curves = max(self.global_max_curves, int(n_curves))
        self.global_max_patches = max(self.global_max_patches, int(n_patches))
        if topo_mask is not None:
            self.global_max_topo = max(self.global_max_topo, int(topo_mask.sum(dim=1).max().item()))
        results = []
        for i in range(B):
            n_valid_curves = int(curves['mask'][i].sum().item())
            n_valid_patches = int(patches['mask'][i].sum().item())
            sample_idx = batch_idx * B + i
            self._verify_padding_zero(curve_latent[i], n_valid_curves, 'curve_latent', sample_idx)
            self._verify_padding_zero(patch_latent[i], n_valid_patches, 'patch_latent', sample_idx)
            curve_latent_padded = pad_to_length(curve_latent[i], MAX_CURVES)
            patch_latent_padded = pad_to_length(patch_latent[i], MAX_PATCHES)
            result = {
                'curve_latent': curve_latent_padded.cpu(),
                'patch_latent': patch_latent_padded.cpu(),
                'n_valid_curves': n_valid_curves,
                'n_valid_patches': n_valid_patches,
            }
            # [修改] 处理topo_latent和topo_data
            if topo_repr is not None:
                n_valid_topo = int(topo_mask[i].sum().item())
                self._verify_padding_zero(topo_repr[i], n_valid_topo, 'topo_latent', sample_idx)
                topo_latent_padded = pad_to_length(topo_repr[i], self.max_topo)
                result['topo_latent'] = topo_latent_padded.cpu()
                result['n_valid_topo'] = n_valid_topo
                # [修改] 保存topo_data用于训练时计算loss
                result['topo_data'] = {
                    'mask': topo_data_batch['mask'][i].cpu(),
                    'curve_idx': topo_data_batch['curve_idx'][i].cpu(),
                    'is_closed': topo_data_batch['is_closed'][i].cpu(),
                    'is_boundary': topo_data_batch['is_boundary'][i].cpu(),
                    'patch0': topo_data_batch['patch0'][i].cpu(),
                    'patch1': topo_data_batch['patch1'][i].cpu(),
                    'adj_p0_e0': topo_data_batch['adj_p0_e0'][i].cpu(),
                    'adj_p0_e1': topo_data_batch['adj_p0_e1'][i].cpu(),
                    'adj_p1_e0': topo_data_batch['adj_p1_e0'][i].cpu(),
                    'adj_p1_e1': topo_data_batch['adj_p1_e1'][i].cpu(),
                    'adj_p0_e0_valid': topo_data_batch['adj_p0_e0_valid'][i].cpu(),
                    'adj_p0_e1_valid': topo_data_batch['adj_p0_e1_valid'][i].cpu(),
                    'adj_p1_e0_valid': topo_data_batch['adj_p1_e0_valid'][i].cpu(),
                    'adj_p1_e1_valid': topo_data_batch['adj_p1_e1_valid'][i].cpu(),
                }
            if pc_features_list is not None:
                result['pc_features'] = pc_features_list[i]
            if save_gt_for_vis:
                result['gt_curves'] = {k: v[i].cpu() if isinstance(v, torch.Tensor) else v
                                       for k, v in curves.items()}
                result['gt_patches'] = {k: v[i].cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in patches.items()}
            results.append(result)
        return results

def main():
    args = get_args()
    global MAX_TOPO
    MAX_TOPO = args.max_topo  # [修改]
    os.makedirs(args.output_folder, exist_ok=True)
    preprocessor = BRepPreprocessor(args)
    from data_loader_optimized import train_data_loader_clean
    loader, _ = train_data_loader_clean(
        batch_size=args.batch_size, data_folder=args.data_folder,
        rotation_augmentation=False, random_angle=False, flag_noise=0,
        flag_grid=True, num_angle=4, dim_grid=args.points_per_patch_dim,
        num_workers=args.num_workers, with_pointcloud=True,
        voxel_dim=args.voxel_dim, feature_type='global', with_normal=True, pad1s=True
    )
    print(f"\n[INFO] Data: {args.data_folder}, Align: {MAX_CURVES}/{MAX_PATCHES}/{MAX_TOPO}")  # [修改]
    all_samples, file_idx, total_samples, vis_samples = [], 0, 0, []
    for batch_idx, data in enumerate(tqdm(loader, desc="Preprocessing", ncols=100)):
        try:
            if len(data) == 3:
                pc_data, curves, patches = data
            elif len(data) == 2:
                curves, patches = data
                pc_data = None
            else:
                continue
            if curves is None or patches is None:
                continue
            save_gt = args.visualize and len(vis_samples) < args.num_vis_samples
            results = preprocessor.process_batch(batch_idx, pc_data, curves, patches, save_gt)
            all_samples.extend(results)
            total_samples += len(results)
            if save_gt:
                for r in results:
                    if 'gt_curves' in r and len(vis_samples) < args.num_vis_samples:
                        vis_samples.append(r)
            if len(all_samples) >= args.samples_per_file:
                torch.save(all_samples, os.path.join(args.output_folder, f'latents_{file_idx:04d}.pt'))
                all_samples, file_idx = [], file_idx + 1
            if batch_idx % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()
        except ValueError as e:
            print(f"\n{e}")
            raise
        except Exception as e:
            print(f"\n[ERROR] Batch {batch_idx}: {e}")
            continue
    if all_samples:
        torch.save(all_samples, os.path.join(args.output_folder, f'latents_{file_idx:04d}.pt'))
    print(f"\n[INFO] Total: {total_samples} samples, {file_idx+1} files")
    # [修改] 更新meta
    meta = {
        'total_samples': total_samples, 'num_files': file_idx + 1,
        'max_curves': MAX_CURVES, 'max_patches': MAX_PATCHES,
        'max_topo': MAX_TOPO,  # [修改]
        'curve_latent_dim': 16, 'patch_latent_dim': 32, 'pc_feature_dim': 7,
        'topo_latent_dim': getattr(preprocessor, 'topo_latent_dim', 128),  # [修改]
        'global_max_curves': preprocessor.global_max_curves,
        'global_max_patches': preprocessor.global_max_patches,
        'global_max_topo': preprocessor.global_max_topo,  # [修改]
        'global_max_pc_points': preprocessor.global_max_pc_points,
        'skip_topo': preprocessor.skip_topo, 'deterministic': args.deterministic,
    }
    torch.save(meta, os.path.join(args.output_folder, 'meta.pt'))
    print(f"[INFO] Meta saved. max_curves={preprocessor.global_max_curves}, "
          f"max_patches={preprocessor.global_max_patches}, max_topo={preprocessor.global_max_topo}")  # [修改]
    if args.visualize and vis_samples:
        from vis import visualize_samples_with_gt_and_pc
        vis_dir = args.vis_output_dir or os.path.join(args.output_folder, 'visualization')
        try:
            visualize_samples_with_gt_and_pc(vis_samples, preprocessor, vis_dir, args.num_vis_samples, args.device)
        except Exception as e:
            print(f"[ERROR] Vis failed: {e}")

if __name__ == '__main__':
    main()