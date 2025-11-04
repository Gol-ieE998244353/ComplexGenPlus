# visualize_worst.py (完整文件 - 支持单/多GPU推理)
import argparse
import torch
import torch.distributed as dist
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ---- 你的模型/工具导入 ----
from curve_patch_trainer import (
    Config,
    PatchEncoder, PatchDecoder,
    CurveEncoder, CurveDecoder,
    process_batch_data,
)
from data_loader_abc import train_data_loader
import plywrite

# （顶部不再依赖全局 voxel_dim）
curve_type_list = np.array(['Circle', 'BSpline', 'Line', 'Ellipse'])
patch_type_list = np.array(['Cylinder', 'Torus', 'BSpline', 'Plane', 'Cone', 'Sphere'])
curve_colormap = {'Circle': np.array([255,0,0]), 'BSpline': np.array([255,255,0]), 'Line': np.array([0,255,0]), 'Ellipse': np.array([0,0,255])}
patch_colormap = {'Plane': np.array([0,255,0]), 'Cylinder': np.array([255,0,0]),  'Torus': np.array([255,128,0]), 'BSpline': np.array([255,255,0]), 'Cone': np.array([255,102,255]), 'Sphere': np.array([0,0,255])}


class Visualizer:
    def __init__(self, checkpoint_path, config, device='cuda'):
        self.config = config
        self.device = device
        self.results = []

        print(f"Loading checkpoint: {checkpoint_path}")
        self.load_models(checkpoint_path)

    def load_models(self, checkpoint_path):
        self.patch_encoder = PatchEncoder(self.config).to(self.device)
        self.patch_decoder = PatchDecoder(self.config).to(self.device)
        self.curve_encoder = CurveEncoder(self.config).to(self.device)
        self.curve_decoder = CurveDecoder(self.config).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理多GPU训练保存的checkpoint（去掉'module.'前缀）
        def remove_module_prefix(state_dict):
            """去掉DDP/DataParallel训练时添加的'module.'前缀"""
            new_state_dict = {}
            for k, v in state_dict.items():
                # 如果key以'module.'开头，去掉前缀
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # 7 = len('module.')
                else:
                    new_state_dict[k] = v
            return new_state_dict
        
        # 加载模型权重（自动处理单/多GPU checkpoint）
        self.patch_encoder.load_state_dict(remove_module_prefix(checkpoint['patch_encoder']))
        self.patch_decoder.load_state_dict(remove_module_prefix(checkpoint['patch_decoder']))
        self.curve_encoder.load_state_dict(remove_module_prefix(checkpoint['curve_encoder']))
        self.curve_decoder.load_state_dict(remove_module_prefix(checkpoint['curve_decoder']))

        self.patch_encoder.eval()
        self.patch_decoder.eval()
        self.curve_encoder.eval()
        self.curve_decoder.eval()
        
        print("✓ Models loaded successfully (auto-handled single/multi-GPU checkpoint)")

    @torch.no_grad()
    def evaluate(self, dataloader, max_samples=None):
        print("Evaluating dataset...")
        for batch_idx, data_item in enumerate(tqdm(dataloader)):
            if max_samples and batch_idx >= max_samples:
                break

            sample_id = data_item[5][0].replace("_fix.pkl", "")
            processed_curves, processed_patches = process_batch_data(
                data_item, self.config, self.device
            )

            result = {'sample_id': sample_id, 'curve_loss': 0.0, 'patch_loss': 0.0}

            # Evaluate patches
            if processed_patches is not None:
                mean_p, _ = self.patch_encoder(
                    processed_patches["patch_points"],
                    processed_patches["patch_normals"],
                    processed_patches["u_closed"],
                    processed_patches["v_closed"],
                    processed_patches["labels"],
                    processed_patches["mask"],
                )
                pred_p = self.patch_decoder(mean_p)

                mask = processed_patches["mask"].unsqueeze(-1).unsqueeze(-1)
                patch_error = ((pred_p["points"] - processed_patches["patch_points"]) ** 2 * mask).sum()
                patch_error = patch_error / processed_patches["mask"].sum().clamp(min=1)
                result['patch_loss'] = patch_error.item()

                result['patch_pred'] = pred_p
                result['patch_gt'] = processed_patches

            # Evaluate curves
            if processed_curves is not None:
                mean_c, _ = self.curve_encoder(
                    processed_curves["curve_points"],
                    processed_curves["endpoints"],
                    processed_curves["is_closed"],
                    processed_curves["labels"],
                    processed_curves["mask"],
                )
                pred_c = self.curve_decoder(mean_c)

                mask = processed_curves["mask"].unsqueeze(-1).unsqueeze(-1)
                curve_error = ((pred_c["points"] - processed_curves["curve_points"]) ** 2 * mask).sum()
                curve_error = curve_error / processed_curves["mask"].sum().clamp(min=1)
                result['curve_loss'] = curve_error.item()

                result['curve_pred'] = pred_c
                result['curve_gt'] = processed_curves

            self.results.append(result)

        print(f"Evaluated {len(self.results)} samples")

    def get_worst_samples(self, n=10, sort_by='total'):
        if sort_by == 'total':
            key = lambda x: x['curve_loss'] + x['patch_loss']
        elif sort_by == 'curve':
            key = lambda x: x['curve_loss']
        elif sort_by == 'patch':
            key = lambda x: x['patch_loss']

        sorted_results = sorted(self.results, key=key, reverse=True)
        return sorted_results[:n]

    def export_sample(self, result, output_dir):
        sample_id = result['sample_id']
        out_dir = Path(output_dir) / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export curves
        if 'curve_pred' in result:
            self._export_curves(result, out_dir)

        # Export patches
        if 'patch_pred' in result:
            self._export_patches(result, out_dir)

        # Export statistics
        stats_path = out_dir / "stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Sample: {sample_id}\n")
            f.write(f"Curve Loss: {result['curve_loss']:.6f}\n")
            f.write(f"Patch Loss: {result['patch_loss']:.6f}\n")
            f.write(f"Total Loss: {result['curve_loss'] + result['patch_loss']:.6f}\n")

    def _export_curves(self, result, out_dir):
        pred = result['curve_pred']
        gt_data = result['curve_gt']

        mask = gt_data['mask'][0].cpu().numpy()
        valid_idx = np.where(mask)[0]

        if len(valid_idx) == 0:
            return

        sample_id = result['sample_id']

        pred_pts = pred['points'][0][valid_idx].cpu().numpy().reshape(-1, 3)
        pred_labels = torch.argmax(pred['label_logits'][0], dim=-1)[valid_idx].cpu().numpy()

        pred_colors = []
        for label in pred_labels:
            color = curve_colormap[curve_type_list[label]]
            pred_colors.extend([color] * 34)
        pred_colors = np.array(pred_colors)

        plywrite.save_vert_color_ply(
            pred_pts, pred_colors,
            str(out_dir / f"{sample_id}_curve_pred.ply")
        )

        gt_pts = gt_data['curve_points'][0][valid_idx].cpu().numpy().reshape(-1, 3)
        gt_labels = gt_data['labels'][0][valid_idx].cpu().numpy()

        gt_colors = []
        for label in gt_labels:
            color = curve_colormap[curve_type_list[label]]
            gt_colors.extend([color] * 34)
        gt_colors = np.array(gt_colors)

        plywrite.save_vert_color_ply(
            gt_pts, gt_colors,
            str(out_dir / f"{sample_id}_curve_gt.ply")
        )

    def _export_patches(self, result, out_dir):
        pred = result['patch_pred']
        gt_data = result['patch_gt']

        mask = gt_data['mask'][0].cpu().numpy()
        valid_idx = np.where(mask)[0]

        if len(valid_idx) == 0:
            return

        sample_id = result['sample_id']

        pred_pts = pred['points'][0][valid_idx].cpu().numpy().reshape(-1, 3)
        pred_labels = torch.argmax(pred['label_logits'][0], dim=-1)[valid_idx].cpu().numpy()

        pred_colors = []
        for label in pred_labels:
            color = patch_colormap[patch_type_list[label]]
            pred_colors.extend([color] * 400)
        pred_colors = np.array(pred_colors)

        plywrite.save_vert_color_ply(
            pred_pts, pred_colors,
            str(out_dir / f"{sample_id}_patch_pred.ply")
        )

        gt_pts = gt_data['patch_points'][0][valid_idx].cpu().numpy().reshape(-1, 3)
        gt_labels = gt_data['labels'][0][valid_idx].cpu().numpy()

        gt_colors = []
        for label in gt_labels:
            color = patch_colormap[patch_type_list[label]]
            gt_colors.extend([color] * 400)
        gt_colors = np.array(gt_colors)

        plywrite.save_vert_color_ply(
            gt_pts, gt_colors,
            str(out_dir / f"{sample_id}_patch_gt.ply")
        )


def main():
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--data', default='data/default/', help='Test data folder')
    parser.add_argument('--output', default=None, help='Output directory (default: checkpoint parent dir / vis_output)')
    parser.add_argument('--n_worst', type=int, default=10, help='Number of worst samples')
    parser.add_argument('--sort_by', default='total', choices=['total', 'curve', 'patch'])
    parser.add_argument('--max_eval', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--input_voxel_dim', type=int, default=128, help='voxel dimension for dataloader')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_parent = checkpoint_path.parent.parent

    if args.output is None:
        args.output = str(checkpoint_parent / 'vis_output')
        print(f"Output directory automatically set to: {args.output}")

    config = Config()

    # Decide whether to initialize distributed group.
    # Only initialize if the environment indicates a multi-process launch (WORLD_SIZE>1).
    use_distributed = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size and world_size > 1:
        # We assume the user launched with torchrun or mp.spawn and env vars are set.
        use_distributed = True

    if use_distributed:
        # initialize distributed (only when torchrun/mp.spawn provided WORLD_SIZE/RANK)
        print(f"[Distributed] rank={rank}, world_size={world_size}, initializing process group...")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",  # use env to be compatible with torchrun
            world_size=world_size,
            rank=rank,
        )
        # set device based on rank
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = f"cuda:{rank % torch.cuda.device_count()}"
        else:
            device = "cpu"
    else:
        # Single-process fallback (safe mode) — won't init process group
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"[Single-process mode] Using device: {device}")

    # Load data — pass voxel dim from args (no global dependency)
    print(f"Loading data from: {args.data}")
    test_data, _ = train_data_loader(
        batch_size=1,
        voxel_dim=args.input_voxel_dim,
        data_folder=args.data,
        feature_type='global',
        pad1s=True,
        rotation_augmentation=True,
        with_normal=True,
        flag_quick_test=False,
        flag_noise=False,
        flag_grid=True,
        flag_patch_uv=True,
        dim_grid=20,
        eval_res_cov=False,
    )

    # Create visualizer
    vis = Visualizer(args.checkpoint, config, device)

    # Evaluate
    vis.evaluate(test_data, max_samples=args.max_eval)

    # Get worst samples
    worst = vis.get_worst_samples(n=args.n_worst, sort_by=args.sort_by)

    print(f"\nTop {args.n_worst} worst samples:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Sample ID':<30} {'Curve':<12} {'Patch':<12} {'Total':<12}")
    print("-" * 70)
    for i, r in enumerate(worst, 1):
        total = r['curve_loss'] + r['patch_loss']
        print(f"{i:<6} {r['sample_id']:<30} {r['curve_loss']:<12.6f} {r['patch_loss']:<12.6f} {total:<12.6f}")

    # Export
    print(f"\nExporting to: {args.output}")
    for r in tqdm(worst, desc="Exporting"):
        vis.export_sample(r, args.output)

    # Summary
    summary_path = Path(args.output) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Worst {args.n_worst} samples (sorted by {args.sort_by})\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(worst, 1):
            f.write(f"Rank {i}: {r['sample_id']}\n")
            f.write(f"  Curve Loss: {r['curve_loss']:.6f}\n")
            f.write(f"  Patch Loss: {r['patch_loss']:.6f}\n")
            f.write(f"  Total Loss: {r['curve_loss'] + r['patch_loss']:.6f}\n\n")

    print(f"\n✓ Done! Results saved to: {args.output}")
    print(f"✓ Summary: {summary_path}")
    print("\nVisualization:")
    print("  1. Open MeshLab")
    print("  2. Load *_pred.ply and *_gt.ply files")
    print("  3. Compare prediction with ground truth")

    # If we initialized distributed group, clean up (optional)
    if use_distributed and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    main()