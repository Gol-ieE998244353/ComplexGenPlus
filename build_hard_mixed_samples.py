import argparse
import heapq
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader_optimized import ABCDatasetOptimized, collate_function_global_topology
from train_pc_nm import Config, CurveDecoder, CurveEncoder, PatchDecoder, PatchEncoder


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"hard_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("hard_mix")


def remove_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def move_to_device(data_dict: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if data_dict is None:
        return None
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}


@dataclass
class HardPart:
    score: float
    source_index: int
    source_filename: str
    payload: Dict


class IndexedDataset(Dataset):
    """Wrap ABCDatasetOptimized and keep original sample index for traceability."""

    def __init__(self, base_dataset: ABCDatasetOptimized, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        locations, features, curves, patches = self.base_dataset[real_idx]
        return locations, features, curves, patches, real_idx


def collate_with_indices(batch):
    batch_no_index = [item[:4] for item in batch]
    merged = collate_function_global_topology(batch_no_index)
    if len(merged) == 3:
        _, curve_batch, patch_batch = merged
    else:
        curve_batch, patch_batch = merged
    indices = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return curve_batch, patch_batch, indices


def compute_curve_item_scores(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)

    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none").mean(dim=(-1, -2))

    pred_endpoints_denorm = pred["endpoints"] * pred_scale + pred_center
    target_endpoints_denorm = target["endpoints"] * pred_scale + pred_center
    endpoint_error = F.mse_loss(pred_endpoints_denorm, target_endpoints_denorm, reduction="none").mean(dim=(-1, -2))
    endpoint_error = endpoint_error * (~target["is_closed"]).float()

    closed_loss = F.binary_cross_entropy_with_logits(
        pred["closed_logits"], target["is_closed"].float(), reduction="none"
    )

    label_loss = F.cross_entropy(
        pred["label_logits"].reshape(-1, pred["label_logits"].shape[-1]),
        target["labels"].reshape(-1),
        reduction="none",
    ).view_as(mask)

    # Match the training objective style but keep it item-wise.
    score = recon_points + 0.5 * endpoint_error + 0.2 * closed_loss + 0.2 * label_loss
    return score * mask.float()


def compute_patch_item_scores(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_scale = target["scale"].unsqueeze(-1).unsqueeze(-1)
    pred_center = target["center"].unsqueeze(-2)

    pred_points_denorm = pred["points"] * pred_scale + pred_center
    target_points_denorm = target["points"] * pred_scale + pred_center
    recon_points = F.mse_loss(pred_points_denorm, target_points_denorm, reduction="none").mean(dim=(-1, -2))

    pred_n = F.normalize(pred["normals"], dim=-1, eps=1e-8)
    target_n = F.normalize(target["normals"], dim=-1, eps=1e-8)
    cosine_loss = (1.0 - (pred_n * target_n).sum(dim=-1)).mean(dim=-1)

    u_loss = F.binary_cross_entropy_with_logits(pred["u_closed_logits"], target["u_closed"].float(), reduction="none")
    v_loss = F.binary_cross_entropy_with_logits(pred["v_closed_logits"], target["v_closed"].float(), reduction="none")

    label_loss = F.cross_entropy(
        pred["label_logits"].reshape(-1, pred["label_logits"].shape[-1]),
        target["labels"].reshape(-1),
        reduction="none",
    ).view_as(mask)

    score = recon_points + 0.3 * cosine_loss + 0.2 * (u_loss + v_loss) + 0.2 * label_loss
    return score * mask.float()


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_raw_curve_from_batch(target_curve: Dict[str, torch.Tensor], b: int, j: int) -> Dict:
    scale = target_curve["scale"][b, j].item()
    center = tensor_to_np(target_curve["center"][b, j])
    norm_points = tensor_to_np(target_curve["points"][b, j])
    points = norm_points * scale + center

    diffs = points[1:] - points[:-1]
    curve_len = float(np.linalg.norm(diffs, axis=1).sum()) if len(points) > 1 else 0.0

    return {
        "points": points.astype(np.float32),
        "is_closed": bool(target_curve["is_closed"][b, j].item()),
        "endpoints": [0, int(max(len(points) - 1, 0))],
        "type": int(target_curve["labels"][b, j].item()),
        "curve_length": curve_len,
    }


def build_raw_patch_from_batch(target_patch: Dict[str, torch.Tensor], b: int, j: int) -> Dict:
    scale = target_patch["scale"][b, j].item()
    center = tensor_to_np(target_patch["center"][b, j])

    norm_points = tensor_to_np(target_patch["points"][b, j])
    normals = tensor_to_np(target_patch["normals"][b, j])
    points = norm_points * scale + center

    grid_normal = np.concatenate([points, normals], axis=1).astype(np.float32)

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    bbox_diag = max_vals - min_vals
    approx_area = float(max(bbox_diag[0] * bbox_diag[1], 1e-8))

    return {
        "type": int(target_patch["labels"][b, j].item()),
        "patch_points": grid_normal,
        "grid_normal": grid_normal,
        "curves": [],
        "u_closed": bool(target_patch["u_closed"][b, j].item()),
        "v_closed": bool(target_patch["v_closed"][b, j].item()),
        "patch_area": approx_area,
    }


def bounded_push(heap: List[Tuple[float, int, HardPart]], item: HardPart, max_size: int, counter: int) -> int:
    entry = (item.score, counter, item)
    if len(heap) < max_size:
        heapq.heappush(heap, entry)
    elif item.score > heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def pick_with_diversity(pool: List[HardPart], count: int, rng: random.Random) -> List[HardPart]:
    if count <= 0 or not pool:
        return []

    indices = list(range(len(pool)))
    rng.shuffle(indices)

    picked = []
    used_sources = set()

    for idx in indices:
        part = pool[idx]
        if part.source_index in used_sources:
            continue
        picked.append(part)
        used_sources.add(part.source_index)
        if len(picked) >= count:
            return picked

    while len(picked) < count:
        picked.append(pool[rng.randrange(len(pool))])
    return picked


def build_surface_points(curves: List[Dict], patches: List[Dict], max_points: int, rng: random.Random) -> np.ndarray:
    pcs = []

    for patch in patches:
        grid = np.asarray(patch["grid_normal"], dtype=np.float32)
        if grid.ndim == 2 and grid.shape[1] >= 6:
            pcs.append(grid[:, :6])

    for curve in curves:
        pts = np.asarray(curve["points"], dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 3:
            zeros_normal = np.zeros_like(pts, dtype=np.float32)
            pcs.append(np.concatenate([pts, zeros_normal], axis=1))

    if not pcs:
        # Keep sample valid in extreme edge cases.
        return np.zeros((1, 6), dtype=np.float32)

    merged = np.concatenate(pcs, axis=0).astype(np.float32)
    if merged.shape[0] > max_points:
        idx = np.arange(merged.shape[0])
        rng.shuffle(idx)
        merged = merged[idx[:max_points]]

    return merged


def filename_for_index(index_meta, fallback_idx: int) -> str:
    name = getattr(index_meta, "filename", None)
    if name:
        return str(name)
    packed_file = getattr(index_meta, "packed_file", "")
    if packed_file:
        return os.path.basename(str(packed_file))
    return f"sample_{fallback_idx:06d}.pkl"


def load_models(checkpoint_path: str, device: torch.device, logger: logging.Logger):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", Config())

    patch_encoder = PatchEncoder(config).to(device)
    patch_decoder = PatchDecoder(config).to(device)
    curve_encoder = CurveEncoder(config).to(device)
    curve_decoder = CurveDecoder(config).to(device)

    patch_encoder.load_state_dict(remove_ddp_prefix(checkpoint["patch_encoder"]))
    patch_decoder.load_state_dict(remove_ddp_prefix(checkpoint["patch_decoder"]))
    curve_encoder.load_state_dict(remove_ddp_prefix(checkpoint["curve_encoder"]))
    curve_decoder.load_state_dict(remove_ddp_prefix(checkpoint["curve_decoder"]))

    patch_encoder.eval()
    patch_decoder.eval()
    curve_encoder.eval()
    curve_decoder.eval()

    logger.info("Loaded checkpoint and model weights successfully")

    return config, patch_encoder, patch_decoder, curve_encoder, curve_decoder


def mine_hard_parts(args, logger: logging.Logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, patch_encoder, patch_decoder, curve_encoder, curve_decoder = load_models(args.checkpoint, device, logger)

    base_dataset = ABCDatasetOptimized(
        args.data_folder,
        random_rotation=False,
        random_angle=False,
        flag_noise=0,
        flag_curve_noise=0,
        flag_patch_noise=0,
        flag_grid=True,
        num_angles=4,
        dim_grid=args.grid_dim,
        with_pointcloud=False,
    )

    total_size = len(base_dataset)
    if total_size == 0:
        raise RuntimeError("No valid samples found in data folder")

    eval_size = min(args.max_eval, total_size) if args.max_eval and args.max_eval > 0 else total_size
    eval_indices = list(range(eval_size))

    dataset = IndexedDataset(base_dataset, eval_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_indices,
        pin_memory=torch.cuda.is_available(),
    )

    curve_all: List[HardPart] = []
    patch_all: List[HardPart] = []

    logger.info("Start mining hard parts from %d samples", eval_size)

    with torch.no_grad():
        for curve_batch, patch_batch, indices in tqdm(dataloader, desc="Mining"):
            curve_batch = move_to_device(curve_batch, device)
            patch_batch = move_to_device(patch_batch, device)

            curve_target = None
            curve_scores = None
            if curve_batch is not None:
                mean_c, _ = curve_encoder(
                    curve_batch["curve_points"],
                    curve_batch["endpoints"],
                    curve_batch["is_closed"],
                    curve_batch["labels"],
                    curve_batch["scale"],
                    curve_batch["center"],
                    curve_batch["mask"],
                )
                pred_curve = curve_decoder(mean_c)
                curve_target = {
                    "points": curve_batch["curve_points"],
                    "endpoints": curve_batch["endpoints"],
                    "is_closed": curve_batch["is_closed"],
                    "labels": curve_batch["labels"],
                    "scale": curve_batch["scale"],
                    "center": curve_batch["center"],
                }
                curve_scores = compute_curve_item_scores(pred_curve, curve_target, curve_batch["mask"])

            patch_target = None
            patch_scores = None
            if patch_batch is not None:
                mean_p, _ = patch_encoder(
                    patch_batch["patch_points"],
                    patch_batch["patch_normals"],
                    patch_batch["u_closed"],
                    patch_batch["v_closed"],
                    patch_batch["labels"],
                    patch_batch["scale"],
                    patch_batch["center"],
                    patch_batch["mask"],
                )
                pred_patch = patch_decoder(mean_p)
                patch_target = {
                    "points": patch_batch["patch_points"],
                    "normals": patch_batch["patch_normals"],
                    "u_closed": patch_batch["u_closed"],
                    "v_closed": patch_batch["v_closed"],
                    "labels": patch_batch["labels"],
                    "scale": patch_batch["scale"],
                    "center": patch_batch["center"],
                }
                patch_scores = compute_patch_item_scores(pred_patch, patch_target, patch_batch["mask"])

            batch_indices = indices.tolist()

            if curve_scores is not None:
                curve_scores_cpu = tensor_to_np(curve_scores)
                curve_mask_cpu = tensor_to_np(curve_batch["mask"]).astype(bool)
                for b, src_idx in enumerate(batch_indices):
                    src_meta = base_dataset.index_list[src_idx]
                    src_name = filename_for_index(src_meta, src_idx)
                    valid_js = np.where(curve_mask_cpu[b])[0]
                    for j in valid_js:
                        score = float(curve_scores_cpu[b, j])
                        if not np.isfinite(score):
                            continue
                        payload = build_raw_curve_from_batch(curve_target, b, int(j))
                        part = HardPart(score=score, source_index=int(src_idx), source_filename=src_name, payload=payload)
                        curve_all.append(part)

            if patch_scores is not None:
                patch_scores_cpu = tensor_to_np(patch_scores)
                patch_mask_cpu = tensor_to_np(patch_batch["mask"]).astype(bool)
                for b, src_idx in enumerate(batch_indices):
                    src_meta = base_dataset.index_list[src_idx]
                    src_name = filename_for_index(src_meta, src_idx)
                    valid_js = np.where(patch_mask_cpu[b])[0]
                    for j in valid_js:
                        score = float(patch_scores_cpu[b, j])
                        if not np.isfinite(score):
                            continue
                        payload = build_raw_patch_from_batch(patch_target, b, int(j))
                        part = HardPart(score=score, source_index=int(src_idx), source_filename=src_name, payload=payload)
                        patch_all.append(part)

    # Resolve desired pool sizes: allow args.hard_* to be integer (>1) or fraction (0< <=1)
    total_curves = len(curve_all)
    total_patches = len(patch_all)

    def resolve_size(arg_val, total_available, name):
        if arg_val is None:
            return total_available
        try:
            v = float(arg_val)
        except Exception:
            raise ValueError(f"Invalid {name} value: {arg_val}")
        if v <= 0:
            raise ValueError(f"{name} must be > 0")
        if v < 1.0:
            return max(1, int(total_available * v))
        return int(v)

    desired_curve_size = resolve_size(args.hard_curve_pool_size, total_curves, "hard_curve_pool_size")
    desired_patch_size = resolve_size(args.hard_patch_pool_size, total_patches, "hard_patch_pool_size")

    curve_pool = sorted(curve_all, key=lambda p: p.score, reverse=True)[:desired_curve_size]
    patch_pool = sorted(patch_all, key=lambda p: p.score, reverse=True)[:desired_patch_size]

    logger.info("Curve hard pool size requested: %s -> actual: %d", str(args.hard_curve_pool_size), len(curve_pool))
    logger.info("Patch hard pool size requested: %s -> actual: %d", str(args.hard_patch_pool_size), len(patch_pool))

    if len(curve_pool) == 0:
        raise RuntimeError("No hard curves mined. Cannot build valid synthetic samples.")

    return curve_pool, patch_pool


def synthesize_dataset(args, curve_pool: List[HardPart], patch_pool: List[HardPart], logger: logging.Logger):
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    records = []
    files_summary = []

    logger.info("Start synthesizing %d hard-mixed samples", args.num_synthetic_samples)

    samples_per_file = int(getattr(args, "samples_per_file", 1000))
    if samples_per_file <= 0:
        raise ValueError("samples_per_file must be > 0")
    # Prepare allocation of parts to samples without replacement
    total_curves = len(curve_pool)
    total_patches = len(patch_pool)
    num_samples = int(args.num_synthetic_samples)

    if num_samples <= 0:
        raise ValueError("num_synthetic_samples must be > 0")

    # Shuffle pools deterministically
    rng.shuffle(curve_pool)
    rng.shuffle(patch_pool)

    base_curves = total_curves // num_samples
    rem_curves = total_curves % num_samples
    base_patches = total_patches // num_samples
    rem_patches = total_patches % num_samples

    logger.info("Allocating curves: total=%d -> per-sample base=%d remainder=%d", total_curves, base_curves, rem_curves)
    logger.info("Allocating patches: total=%d -> per-sample base=%d remainder=%d", total_patches, base_patches, rem_patches)

    if base_curves == 0 and total_curves > 0:
        logger.warning("Some samples will receive 0 curves because pool smaller than num_synthetic_samples")
    if base_patches == 0 and total_patches > 0:
        logger.warning("Some samples will receive 0 patches because pool smaller than num_synthetic_samples")

    # Build assignments
    curve_assignments: List[List[HardPart]] = [[] for _ in range(num_samples)]
    patch_assignments: List[List[HardPart]] = [[] for _ in range(num_samples)]

    ci = 0
    for i in range(num_samples):
        take = base_curves + (1 if i < rem_curves else 0)
        if take > 0:
            curve_assignments[i] = curve_pool[ci:ci+take]
            ci += take

    pi = 0
    for i in range(num_samples):
        take = base_patches + (1 if i < rem_patches else 0)
        if take > 0:
            patch_assignments[i] = patch_pool[pi:pi+take]
            pi += take

    # Build samples and write grouped files
    current_samples = []
    current_records = []
    group_idx = 0

    for sid in range(num_samples):
        assigned_curves = curve_assignments[sid]
        assigned_patches = patch_assignments[sid]

        raw_curves = [c.payload for c in assigned_curves]
        raw_patches = [p.payload for p in assigned_patches]
        surface_points = build_surface_points(raw_curves, raw_patches, args.max_surface_points, rng)

        sample_meta_name = f"hardmix_{sid:06d}.pkl"

        sample = {
            "surface_points": surface_points.astype(np.float32),
            "curves": raw_curves,
            "patches": raw_patches,
            "filename": sample_meta_name,
        }

        record = {
            "file": None,
            "sample_name": sample_meta_name,
            "num_curves": len(raw_curves),
            "num_patches": len(raw_patches),
            "curve_sources": [c.source_filename for c in assigned_curves],
            "patch_sources": [p.source_filename for p in assigned_patches],
            "curve_scores": [float(c.score) for c in assigned_curves],
            "patch_scores": [float(p.score) for p in assigned_patches],
        }

        current_samples.append(sample)
        current_records.append(record)

        if len(current_samples) >= samples_per_file or sid == num_samples - 1:
            group_name = f"hardmix_group_{group_idx:06d}.pkl"
            group_path = out_dir / group_name

            with open(group_path, "wb") as f:
                pickle.dump({"samples": current_samples}, f)

            for rec in current_records:
                rec["file"] = group_name
                records.append(rec)

            files_summary.append({"file": group_name, "num_samples": len(current_samples)})

            logger.info("Wrote group %s with %d samples", group_name, len(current_samples))

            group_idx += 1
            current_samples = []
            current_records = []

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": args.num_synthetic_samples,
        "samples_per_file": samples_per_file,
        "num_files": len(files_summary),
        "curves_per_sample_base": base_curves,
        "curves_per_sample_remainder": rem_curves,
        "patches_per_sample_base": base_patches,
        "patches_per_sample_remainder": rem_patches,
        "hard_curve_pool_size": args.hard_curve_pool_size,
        "hard_patch_pool_size": args.hard_patch_pool_size,
        "files": files_summary,
        "records": records,
    }

    manifest_path = out_dir / "hardmix_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    logger.info("Synthetic dataset saved to: %s", out_dir)
    logger.info("Manifest saved to: %s", manifest_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard curve/patch parts with a checkpoint and repack them into synthetic training samples"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_folder", required=True, help="Input dataset folder (raw pkl folder)")
    parser.add_argument("--output_folder", required=True, help="Output folder for synthetic pkl files")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grid_dim", type=int, default=20, help="Patch grid size used in loader")
    parser.add_argument("--max_eval", type=int, default=0, help="Max number of samples to evaluate, 0 means all")

    parser.add_argument("--hard_curve_pool_size", type=float, default=8000.0,
                        help="If >=1: number of curves to keep; if <1: fraction of total mined curves to keep")
    parser.add_argument("--hard_patch_pool_size", type=float, default=8000.0,
                        help="If >=1: number of patches to keep; if <1: fraction of total mined patches to keep")

    parser.add_argument("--num_synthetic_samples", type=int, default=2000)
    parser.add_argument("--curves_per_sample", type=int, default=8)
    parser.add_argument("--patches_per_sample", type=int, default=6)
    parser.add_argument("--max_surface_points", type=int, default=6000)

    parser.add_argument("--samples_per_file", type=int, default=1000,
                        help="Number of synthetic samples to pack into a single pkl file (default 1000)")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_folder)
    logger = setup_logger(output_dir)
    if float(args.hard_curve_pool_size) <= 0:
        raise ValueError("hard_curve_pool_size must be > 0")
    if float(args.hard_patch_pool_size) <= 0:
        raise ValueError("hard_patch_pool_size must be > 0")

    curve_pool, patch_pool = mine_hard_parts(args, logger)
    # If user passed explicit curves_per_sample/patches_per_sample, log that they're ignored
    if hasattr(args, "curves_per_sample"):
        logger.info("Argument --curves_per_sample is ignored; allocation will be derived from pool sizes and num_synthetic_samples")
    if hasattr(args, "patches_per_sample"):
        logger.info("Argument --patches_per_sample is ignored; allocation will be derived from pool sizes and num_synthetic_samples")

    synthesize_dataset(args, curve_pool, patch_pool, logger)


if __name__ == "__main__":
    main()
