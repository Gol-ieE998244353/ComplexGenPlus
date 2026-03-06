import argparse
import csv
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from train_pc_nm import (
	Config,
	PatchEncoder,
	PatchDecoder,
	CurveEncoder,
	CurveDecoder,
	compute_patch_metrics,
	compute_curve_metrics,
)
from data_loader_optimized import (
	pack_pickle_files,
	data_loader_ABC,
	ABCDatasetOptimized,
	load_raw_sample,
)

def build_curve_patch_correspondence(patches, n_curves):
    """
    从patches的curves信息构建curve_patch_correspondence
    
    Args:
        patches: list of patch dicts, each containing 'curves' field
        n_curves: total number of curves
    
    Returns:
        curve_patch_corr: [n_curves, 2] 每条curve连接的两个patch索引，-1表示边界
    """
    curve_patch_corr = np.full((n_curves, 2), -1, dtype=np.int64)
    for patch_idx, patch in enumerate(patches):
        if 'curves' not in patch:
            continue
        
        for curve_idx in patch['curves']:
            if curve_idx >= n_curves:
                continue
            
            if curve_patch_corr[curve_idx, 0] == -1:
                curve_patch_corr[curve_idx, 0] = patch_idx
            elif curve_patch_corr[curve_idx, 1] == -1:
                curve_patch_corr[curve_idx, 1] = patch_idx
            else:
                raise Exception("Wrong!")
    
    return curve_patch_corr

def collate_function_with_topology(tensorlist):
    """
    包含拓扑信息的collate函数
    """
    batch_size = len(tensorlist)
    
    all_curves = [item[0] for item in tensorlist]
    all_patches = [item[1] for item in tensorlist]
    
    # === 处理Curves ===
    processed_curves = None
    if any(len(curves) > 0 for curves in all_curves):
        max_n_curves = max(len(curves) for curves in all_curves)
        
        curve_points_batch = torch.zeros(batch_size, max_n_curves, 34, 3, dtype=torch.float32)
        endpoints_batch = torch.zeros(batch_size, max_n_curves, 2, 3, dtype=torch.float32)
        is_closed_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool)
        labels_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.long)
        mask_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool)
        scale_batch = torch.ones(batch_size, max_n_curves, dtype=torch.float32)
        center_batch = torch.zeros(batch_size, max_n_curves, 3, dtype=torch.float32)
        
        for i, curves in enumerate(all_curves):
            n_curves = len(curves)
            if n_curves > 0:
                for j, curve in enumerate(curves):
                    if j >= max_n_curves:
                        break
                    curve_points_batch[i, j] = torch.from_numpy(curve['curve_points'])
                    endpoints_batch[i, j] = torch.from_numpy(curve['endpoints'])
                    is_closed_batch[i, j] = curve['is_closed']
                    labels_batch[i, j] = curve['label']
                    scale_batch[i, j] = curve['scale']
                    center_batch[i, j] = torch.from_numpy(curve['center'])
                    mask_batch[i, j] = True
        
        processed_curves = {
            "curve_points": curve_points_batch,
            "endpoints": endpoints_batch,
            "is_closed": is_closed_batch,
            "labels": labels_batch,
            "mask": mask_batch,
            "scale": scale_batch,
            "center": center_batch
        }
    
    # === 处理Patches ===
    processed_patches = None
    curve_patch_corr_batch = None
    
    if any(len(patches) > 0 for patches in all_patches):
        max_n_patches = max(len(patches) for patches in all_patches)
        max_n_curves = max(len(curves) for curves in all_curves) if processed_curves else 0
        
        patch_points_batch = torch.zeros(batch_size, max_n_patches, 400, 3, dtype=torch.float32)
        patch_normals_batch = torch.zeros(batch_size, max_n_patches, 400, 3, dtype=torch.float32)
        u_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        v_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        labels_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.long)
        mask_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        scale_batch = torch.ones(batch_size, max_n_patches, dtype=torch.float32)
        center_batch = torch.zeros(batch_size, max_n_patches, 3, dtype=torch.float32)
        
        # 构建curve_patch_correspondence
        curve_patch_corr_batch = torch.full((batch_size, max_n_curves, 2), -1, dtype=torch.long)
        
        for i, patches in enumerate(all_patches):
            n_patches = len(patches)
            n_curves = len(all_curves[i])
            
            if n_patches > 0:
                # 构建该样本的curve_patch_corr
                if n_curves > 0:
                    corr = build_curve_patch_correspondence(patches, n_curves)
                    curve_patch_corr_batch[i, :n_curves] = torch.from_numpy(corr)
                
                for j, patch in enumerate(patches):
                    if j >= max_n_patches:
                        break
                    
                    patch_pts = patch['patch_points']
                    patch_norms = patch['patch_normals']
                    
                    if len(patch_pts) == 400:
                        patch_points_batch[i, j] = torch.from_numpy(patch_pts)
                        patch_normals_batch[i, j] = torch.from_numpy(patch_norms)
                    elif len(patch_pts) == 100:
                        patch_points_batch[i, j, :100] = torch.from_numpy(patch_pts)
                        patch_normals_batch[i, j, :100] = torch.from_numpy(patch_norms)
                    
                    u_closed_batch[i, j] = patch['u_closed']
                    v_closed_batch[i, j] = patch['v_closed']
                    labels_batch[i, j] = patch['label']
                    scale_batch[i, j] = patch['scale']
                    center_batch[i, j] = torch.from_numpy(patch['center'])
                    mask_batch[i, j] = True
        
        processed_patches = {
            "patch_points": patch_points_batch,
            "patch_normals": patch_normals_batch,
            "u_closed": u_closed_batch,
            "v_closed": v_closed_batch,
            "labels": labels_batch,
            "mask": mask_batch,
            "scale": scale_batch,
            "center": center_batch
        }
    
    return (processed_curves, processed_patches, curve_patch_corr_batch)

def setup_logger(output_dir, rank=0):
	"""Create a simple console+file logger."""
	if rank != 0:
		return None

	os.makedirs(output_dir, exist_ok=True)
	log_file = Path(output_dir) / f"select_worst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s [%(levelname)s] %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	return logging.getLogger(__name__)


def setup_distributed():
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ['RANK'])
		world_size = int(os.environ['WORLD_SIZE'])
		local_rank = int(os.environ.get('LOCAL_RANK', 0))
		if not dist.is_initialized():
			dist.init_process_group(
				backend='nccl' if torch.cuda.is_available() else 'gloo',
				init_method='env://'
			)
			if torch.cuda.is_available():
				torch.cuda.set_device(local_rank)
	else:
		rank, world_size, local_rank = 0, 1, 0
		if not dist.is_initialized():
			dist.init_process_group(
				backend='nccl' if torch.cuda.is_available() else 'gloo',
				init_method='tcp://127.0.0.1:23456',
				world_size=1,
				rank=0
			)
			if torch.cuda.is_available():
				torch.cuda.set_device(0)
	return rank, world_size, local_rank


def ensure_packed_folder(data_folder, rank, world_size, logger=None):
	packed_dir = os.path.join(data_folder, "packed")
	if os.path.exists(packed_dir):
		return

	if rank == 0:
		os.makedirs(packed_dir, exist_ok=True)
		if logger:
			logger.info(f"Packing raw pickle files under {data_folder}")
		pack_pickle_files(data_folder, packed_dir)
	if world_size > 1:
		dist.barrier()


def load_checkpoint_robust(checkpoint_path, device, logger=None):
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	if logger:
		logger.info(f"Loading checkpoint from: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location=device)
	config = checkpoint.get('config', None)
	if config is None:
		if logger:
			logger.warning("Config not found in checkpoint, using default Config()")
		config = Config()
	else:
		if logger:
			logger.info("✓ Loaded config from checkpoint")

	if logger and 'epoch' in checkpoint:
		msg = f"Checkpoint epoch: {checkpoint['epoch']}"
		if 'val_loss' in checkpoint:
			msg += f", val_loss: {checkpoint['val_loss']:.6f}"
		logger.info(msg)

	return checkpoint, config


def remove_ddp_prefix(state_dict):
	return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def move_to_device(data_dict, device):
	if data_dict is None:
		return None
	return {
		k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
		for k, v in data_dict.items()
	}


def sanitize_loss(value):
	if value is None:
		return 0.0
	if isinstance(value, torch.Tensor):
		value = value.item()
	return value if math.isfinite(float(value)) else 0.0


def slice_tensor_dict(tensor_dict, index):
	return {k: v[index:index + 1].contiguous() for k, v in tensor_dict.items()}


class IndexedDataset(Dataset):
	def __init__(self, base_dataset, indices=None):
		self.base_dataset = base_dataset
		if indices is None:
			self.indices = list(range(len(base_dataset)))
		else:
			self.indices = list(indices)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		real_idx = self.indices[idx]
		sample = self.base_dataset[real_idx]
		curves, patches = self._split_curves_and_patches(sample)
		return curves, patches, real_idx

	@staticmethod
	def _split_curves_and_patches(sample):
		if isinstance(sample, (list, tuple)):
			if len(sample) >= 4:
				return sample[-2], sample[-1]
			if len(sample) >= 2:
				return sample[0], sample[1]
		if isinstance(sample, dict):
			return sample.get('curves'), sample.get('patches')
		raise ValueError("Unsupported sample format from base dataset")


def collate_with_indices(batch):
	pairs = [(curves, patches) for curves, patches, _ in batch]
	indices = torch.tensor([idx for _, _, idx in batch], dtype=torch.long)
	curve_batch, patch_batch, corr_batch = collate_function_with_topology(pairs)
	return curve_batch, patch_batch, corr_batch, indices


class HardSampleMiner:
	def __init__(self, checkpoint_data, config, device, logger=None, rank=0):
		self.config = config
		self.device = device
		self.logger = logger
		self.rank = rank
		self.results = []

		if self.logger and self.rank == 0:
			self.logger.info("Initializing HardSampleMiner")
			self.logger.info(f"Curve latent dim: {config.CURVE_LATENT_DIM}")
			self.logger.info(f"Patch latent dim: {config.PATCH_LATENT_DIM}")

		self._load_models(checkpoint_data)

	def _load_models(self, checkpoint):
		self.patch_encoder = PatchEncoder(self.config).to(self.device)
		self.patch_decoder = PatchDecoder(self.config).to(self.device)
		self.curve_encoder = CurveEncoder(self.config).to(self.device)
		self.curve_decoder = CurveDecoder(self.config).to(self.device)

		self.patch_encoder.load_state_dict(remove_ddp_prefix(checkpoint['patch_encoder']))
		self.patch_decoder.load_state_dict(remove_ddp_prefix(checkpoint['patch_decoder']))
		self.curve_encoder.load_state_dict(remove_ddp_prefix(checkpoint['curve_encoder']))
		self.curve_decoder.load_state_dict(remove_ddp_prefix(checkpoint['curve_decoder']))

		self.patch_encoder.eval()
		self.patch_decoder.eval()
		self.curve_encoder.eval()
		self.curve_decoder.eval()

		if self.logger and self.rank == 0:
			self.logger.info("✓ Loaded autoencoder weights")

	def evaluate(self, dataloader):
		self.results = []
		curve_losses, patch_losses = [], []

		if self.logger and self.rank == 0:
			self.logger.info("Starting evaluation across dataset")

		iterator = tqdm(dataloader, desc="Evaluating", disable=self.rank != 0)
		with torch.no_grad():
			for curve_batch, patch_batch, _, indices in iterator:
				curve_batch = move_to_device(curve_batch, self.device)
				patch_batch = move_to_device(patch_batch, self.device)

				patch_pred = None
				if patch_batch is not None:
					latent_p = self.patch_encoder(
						patch_batch["patch_points"],
						patch_batch["patch_normals"],
						patch_batch["u_closed"],
						patch_batch["v_closed"],
						patch_batch["labels"],
						patch_batch["scale"],
						patch_batch["center"],
						patch_batch["mask"],
					)
					if isinstance(latent_p, tuple):
						latent_p = latent_p[0]
					patch_pred = self.patch_decoder(latent_p)

				curve_pred = None
				if curve_batch is not None:
					latent_c = self.curve_encoder(
						curve_batch["curve_points"],
						curve_batch["endpoints"],
						curve_batch["is_closed"],
						curve_batch["labels"],
						curve_batch["scale"],
						curve_batch["center"],
						curve_batch["mask"],
					)
					if isinstance(latent_c, tuple):
						latent_c = latent_c[0]
					curve_pred = self.curve_decoder(latent_c)

				indices_list = indices.tolist()
				batch_size = len(indices_list)

				for local_idx in range(batch_size):
					sample_index = int(indices_list[local_idx])
					entry = {
						"index": sample_index,
						"curve_loss": 0.0,
						"patch_loss": 0.0,
						"total_loss": 0.0,
					}

					if patch_batch is not None:
						mask_slice = patch_batch["mask"][local_idx:local_idx + 1]
						if mask_slice.any():
							target_patch = {
								"points": patch_batch["patch_points"][local_idx:local_idx + 1],
								"normals": patch_batch["patch_normals"][local_idx:local_idx + 1],
								"u_closed": patch_batch["u_closed"][local_idx:local_idx + 1],
								"v_closed": patch_batch["v_closed"][local_idx:local_idx + 1],
								"labels": patch_batch["labels"][local_idx:local_idx + 1],
								"scale": patch_batch["scale"][local_idx:local_idx + 1],
								"center": patch_batch["center"][local_idx:local_idx + 1],
							}
							pred_patch = slice_tensor_dict(patch_pred, local_idx)
							metrics = compute_patch_metrics(pred_patch, target_patch, mask_slice)
							entry["patch_loss"] = sanitize_loss(metrics.get("recon_error"))
							patch_losses.append(entry["patch_loss"])

					if curve_batch is not None:
						mask_slice = curve_batch["mask"][local_idx:local_idx + 1]
						if mask_slice.any():
							target_curve = {
								"points": curve_batch["curve_points"][local_idx:local_idx + 1],
								"endpoints": curve_batch["endpoints"][local_idx:local_idx + 1],
								"is_closed": curve_batch["is_closed"][local_idx:local_idx + 1],
								"labels": curve_batch["labels"][local_idx:local_idx + 1],
								"scale": curve_batch["scale"][local_idx:local_idx + 1],
								"center": curve_batch["center"][local_idx:local_idx + 1],
							}
							pred_curve = slice_tensor_dict(curve_pred, local_idx)
							metrics = compute_curve_metrics(pred_curve, target_curve, mask_slice)
							entry["curve_loss"] = sanitize_loss(metrics.get("recon_error"))
							curve_losses.append(entry["curve_loss"])

					entry["total_loss"] = entry["curve_loss"] + entry["patch_loss"]
					self.results.append(entry)

		if self.logger and self.rank == 0:
			if curve_losses:
				self.logger.info(
					f"Curve recon loss - mean: {sum(curve_losses)/len(curve_losses):.6f}, std: {torch.tensor(curve_losses).std(unbiased=False).item():.6f}"
				)
			if patch_losses:
				self.logger.info(
					f"Patch recon loss - mean: {sum(patch_losses)/len(patch_losses):.6f}, std: {torch.tensor(patch_losses).std(unbiased=False).item():.6f}"
				)

	def gather_results(self):
		if not dist.is_initialized():
			return self.results

		world_size = dist.get_world_size()
		gathered = [None for _ in range(world_size)]
		dist.all_gather_object(gathered, self.results)

		merged = {}
		for part in gathered:
			if not part:
				continue
			for record in part:
				idx = record["index"]
				if idx not in merged or record["total_loss"] > merged[idx]["total_loss"]:
					merged[idx] = record

		merged_list = list(merged.values())
		merged_list.sort(key=lambda x: x["index"])
		self.results = merged_list
		return merged_list

	def export_prediction(self, dataloader, output_file):
		"""Run inference on a single sample and export gt/pred."""
		iterator = iter(dataloader)
		try:
			curve_batch, patch_batch, _, indices = next(iterator)
		except StopIteration:
			if self.logger: self.logger.error("Dataloader empty during export_prediction")
			return

		with torch.no_grad():
			curve_batch = move_to_device(curve_batch, self.device)
			patch_batch = move_to_device(patch_batch, self.device)

			patch_pred = None
			if patch_batch is not None:
				latent_p = self.patch_encoder(
					patch_batch["patch_points"],
					patch_batch["patch_normals"],
					patch_batch["u_closed"],
					patch_batch["v_closed"],
					patch_batch["labels"],
					patch_batch["scale"],
					patch_batch["center"],
					patch_batch["mask"],
				)
				if isinstance(latent_p, tuple):
					latent_p = latent_p[0]
				patch_pred = self.patch_decoder(latent_p)

			curve_pred = None
			if curve_batch is not None:
				latent_c = self.curve_encoder(
					curve_batch["curve_points"],
					curve_batch["endpoints"],
					curve_batch["is_closed"],
					curve_batch["labels"],
					curve_batch["scale"],
					curve_batch["center"],
					curve_batch["mask"],
				)
				if isinstance(latent_c, tuple):
					latent_c = latent_c[0]
				curve_pred = self.curve_decoder(latent_c)
			
			# Move to CPU for saving
			# Helper to detach and convert dictionary of tensors
			def to_numpy_dict(d):
				if d is None: return None
				return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

			export_data = {
				"gt_curves": to_numpy_dict(curve_batch),
				"gt_patches": to_numpy_dict(patch_batch),
				"pred_curves": to_numpy_dict(curve_pred),
				"pred_patches": to_numpy_dict(patch_pred),
			}
			
			output_path = Path(output_file)
			output_path.parent.mkdir(parents=True, exist_ok=True)
			with open(output_path, 'wb') as f:
				pickle.dump(export_data, f)
				
			if self.logger:
				self.logger.info(f"Exported prediction to {output_path}")


def select_worst_samples(results, n_worst, sort_by):
	if not results:
		return []

	if sort_by == 'curve':
		key_fn = lambda r: r['curve_loss']
	elif sort_by == 'patch':
		key_fn = lambda r: r['patch_loss']
	else:
		key_fn = lambda r: r['total_loss']

	sorted_results = sorted(results, key=key_fn, reverse=True)
	return sorted_results[:min(n_worst, len(sorted_results))]


def export_loss_to_csv(all_results, samples, output_file, logger=None):
	"""Export all sample losses and filenames to a CSV file."""
	if logger:
		logger.info(f"Exporting losses to {output_file}")
	
	with open(output_file, 'w', newline='') as csvfile:
		fieldnames = ['filename', 'curve_loss', 'patch_loss', 'total_loss']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		
		for record in all_results:
			index = record['index']
			filename = samples[index].filename if hasattr(samples[index], 'filename') else f"sample_{index}"
			writer.writerow({
				'filename': filename,
				'curve_loss': record['curve_loss'],
				'patch_loss': record['patch_loss'],
				'total_loss': record['total_loss']
			})
	
	if logger:
		logger.info(f"Exported {len(all_results)} loss records to {output_file}")


def export_dataset(selected_indices, samples, output_path, overwrite=False, logger=None):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_path.exists() and not overwrite:
		raise FileExistsError(f"Output file already exists: {output_path}")

	written = 0
	with open(output_path, 'wb') as wf:
		for idx in selected_indices:
			try:
				sample_data = load_raw_sample(samples[idx])
			except Exception as exc:
				if logger:
					logger.error(f"Failed to load sample at index {idx}: {exc}")
				continue
			pickle.dump(sample_data, wf)
			written += 1

	if logger:
		logger.info(f"✓ Wrote {written} samples to {output_path}")


def export_filenames(selected_indices, samples, output_path, overwrite=False, logger=None):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_path.exists() and not overwrite:
		raise FileExistsError(f"Output file already exists: {output_path}")

	written = 0
	with open(output_path, 'w', encoding='utf-8') as wf:
		for idx in selected_indices:
			sample_index = samples[idx]
			filename = getattr(sample_index, 'filename', None)
			if not filename:
				try:
					sample_data = load_raw_sample(sample_index)
					filename = sample_data.get('filename') if isinstance(sample_data, dict) else None
				except Exception:
					filename = None
			if not filename:
				filename = os.path.basename(sample_index.packed_file)
			if not filename:
				filename = f"sample_{idx}"
			wf.write(f"{filename}\n")
			written += 1

	if logger:
		logger.info(f"✓ Wrote {written} filenames to {output_path}")


def parse_arguments():
	parser = argparse.ArgumentParser(description='Select worst reconstruction samples and save to new dataset')
	parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
	parser.add_argument('--data_folder', required=True, help='Input folder containing raw/packed pkl files')
	
	# Action arguments (Mutually exclusive? Or mixable but with clear precedence)
	parser.add_argument('--export_sample', type=str, help='Export specific sample gt/pred by filename (skips full eval)')
	parser.add_argument('--export_loss', action='store_true', help='Export all sample losses to CSV')
	
	# Output configuration
	parser.add_argument('--output_file', help='Output pkl/txt file for worst samples or exported sample')
	parser.add_argument('--export_loss_file', type=str, default='losses.csv', help='Output CSV file for losses (used with --export_loss)')
	
	# Worst selection options (Only used if NOT exporting single sample)
	parser.add_argument('--n_worst', type=int, default=100, help='Number of worst samples to export')
	parser.add_argument('--sort_by', choices=['total', 'curve', 'patch'], default='total', help='Sorting metric')
	parser.add_argument('--filename_only', action='store_true', help='Only export filenames of selected samples to a txt file')

	# Evaluation configuration
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--grid_dim', type=int, default=20, help='Patch grid dimension')
	parser.add_argument('--rotation_augment', action='store_true', help='Apply rotation augmentation during preprocessing')
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--max_eval', type=int, default=None, help='Limit number of samples evaluated')
	parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if it exists')
	
	return parser.parse_args()


def export_single_sample(miner, dataset, samples, target_filename, output_file, device, logger):
	"""Find and export a single specific sample."""
	target_idx = -1
	
	# 1. Find the sample index
	if logger: logger.info(f"Searching for sample with filename: {target_filename}...")
	
	for i, s in enumerate(samples):
		# Create a candidate list of names to match against
		candidates = []
		
		# If SampleIndex has filename
		if hasattr(s, 'filename') and s.filename:
			candidates.append(s.filename)
			
		# Also check against basename of packed file if no explicit filename
		# or just allow checking by index if user passes integer string? No, explicit filename.
		
		# Strict match or substring? Let's do exact match on filename component.
		base_name = os.path.basename(candidates[0]) if candidates else ""
		
		if target_filename == base_name or target_filename in candidates:
			target_idx = i
			break
			
	if target_idx == -1:
		if logger: logger.error(f"Sample '{target_filename}' not found in dataset.")
		return

	if logger: logger.info(f"Found sample at index {target_idx}. Running inference...")

	# 2. Get data and run inference
	# We need to use the dataloader or dataset directly.
	# Create a mini dataset/loader for this one sample to reuse collate_fn
	indexed_dataset = IndexedDataset(dataset, [target_idx])
	dataloader = DataLoader(
		indexed_dataset,
		batch_size=1,
		collate_fn=collate_with_indices,
		shuffle=False,
		num_workers=0
	)
	
	miner.export_prediction(dataloader, output_file)


def main():
	args = parse_arguments()
	rank, world_size, local_rank = setup_distributed()

	# If output_file is not provided but might be needed
	if not args.output_file and not args.export_loss and not args.export_sample:
		# User didn't specify anything to do essentially? 
		# Or maybe they just wanted to run eval?
		# Original code required output_file.
		# If we are in "Select Worst" mode (default), we definitely need output_file.
		print("Error: --output_file is required for worst sample selection.")
		return

	# Setup Logging
	log_dir = Path(args.output_file).parent if args.output_file else Path(".")
	logger = setup_logger(log_dir, rank=rank)

	# ... (Load Checkpoint, Miner, Dataset) ...
	device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
	checkpoint, config = load_checkpoint_robust(args.checkpoint, device, logger)
	miner = HardSampleMiner(checkpoint, config, device, logger=logger, rank=rank)

	ensure_packed_folder(args.data_folder, rank, world_size, logger)
	samples = data_loader_ABC(args.data_folder)

	if not samples:
		if logger and rank == 0:
			logger.error("No samples found in the specified data folder")
		return

	base_dataset = ABCDatasetOptimized(
		args.data_folder,
		random_rotation=args.rotation_augment,
		random_angle=False,
		flag_noise=0,
		flag_grid=True,
		num_angles=4,
		dim_grid=args.grid_dim,
	)

	# --- Mode 1: Export Specific Sample ---
	if args.export_sample:
		if not args.output_file:
			if logger: logger.error("Please provide --output_file to save the exported sample data.")
			return
		export_single_sample(miner, base_dataset, samples, args.export_sample, args.output_file, device, logger)
		return

	# --- Mode 2/3: Full Evaluation (Loss Export / Worst Selection) ---
	total_samples = len(samples)
	eval_count = min(args.max_eval, total_samples) if args.max_eval else total_samples
	indices = list(range(eval_count))

	if logger and rank == 0:
		logger.info(f"Loaded {total_samples} samples, evaluating {eval_count}")

	indexed_dataset = IndexedDataset(base_dataset, indices)


	sampler = None
	if world_size > 1:
		sampler = DistributedSampler(
			indexed_dataset,
			num_replicas=world_size,
			rank=rank,
			shuffle=False,
			drop_last=False,
		)

	dataloader = DataLoader(
		indexed_dataset,
		batch_size=args.batch_size,
		collate_fn=collate_with_indices,
		shuffle=False,
		sampler=sampler,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
		multiprocessing_context='fork' if args.num_workers > 0 else None,
	)

	miner.evaluate(dataloader)
	all_results = miner.gather_results()

	if rank != 0:
		if dist.is_initialized():
			dist.barrier()
		return

	if not all_results:
		logger.error("No evaluation results were collected")
		return

	if args.export_loss:
		export_loss_to_csv(all_results, samples, args.export_loss_file, logger)

	if args.output_file:
		selected = select_worst_samples(all_results, args.n_worst, args.sort_by)
		if not selected:
			if logger: logger.error("No samples qualified for export")
			return

		if logger:
			logger.info(f"Top {len(selected)} worst samples (sorted by {args.sort_by}):")
			logger.info(f"{'Rank':<6} {'Idx':<10} {'Curve':<15} {'Patch':<15} {'Total':<15}")
			logger.info('-' * 70)
			for rank_id, record in enumerate(selected, 1):
				logger.info(
					f"{rank_id:<6} {record['index']:<10} {record['curve_loss']:<15.6f} {record['patch_loss']:<15.6f} {record['total_loss']:<15.6f}"
				)

		selected_indices = [r['index'] for r in selected]
		if args.filename_only:
			export_filenames(selected_indices, samples, args.output_file, args.overwrite, logger)
		else:
			export_dataset(selected_indices, samples, args.output_file, args.overwrite, logger)

	if dist.is_initialized():
		dist.barrier()
		dist.destroy_process_group()


if __name__ == '__main__':
	main()
