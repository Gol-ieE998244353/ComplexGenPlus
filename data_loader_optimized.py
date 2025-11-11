import numpy as np
import os
import torch
import math
from scipy.spatial.transform import Rotation as R
import pickle
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

average_patch_area = 0
average_squared_curve_length = 0
pack_size = 10000
th_norm = 1e-6
points_per_curve_dim = 34

SCALE_MIN = 1e-4
SCALE_MAX = 1e2
LOG_SCALE_MIN = -10.0
LOG_SCALE_MAX = 10.0

def pack_pickle_files(data_folder, packed_data_folder):
    """Pack pickle files for better I/O performance"""
    print(f"Packing data from {data_folder} to {packed_data_folder}")
    files = os.listdir(data_folder)
    random.shuffle(files)
    file_count = 0
    for file in files:
        if file.endswith(".pkl"):
            file_count += 1
            if file_count % pack_size == 1:
                packed_file = open(os.path.join(packed_data_folder, f"packed_{file_count-1:06d}.pkl"), "wb")
            with open(os.path.join(data_folder, file), "rb") as rf:
                sample = pickle.load(rf)
                sample['filename'] = file
                pickle.dump(sample, packed_file)
            if file_count % pack_size == 0:
                packed_file.close()
    if not packed_file.closed:
        packed_file.close()

def data_loader_ABC(data_folder):
    """Load ABC dataset"""
    def curve_type_to_id(str):
        if str == 'Circle': return 0
        if str == 'BSpline': return 1
        if str == 'Line': return 2
        return 3  # Ellipse
    
    def patch_type_to_id(str):
        if str == 'Cylinder': return 0
        if str == 'Torus': return 1
        if str in ['BSpline', 'Extrusion', 'Revolution']: return 2
        if str == 'Plane': return 3
        if str == 'Cone': return 4
        return 5  # Sphere
    
    sample_list = []
    print(f"Loading data from {data_folder}")
    
    if os.path.exists(os.path.join(data_folder, "packed")):
        print("Using packed pkl files")
        read_from_packed_pkl = True
        data_folder = os.path.join(data_folder, "packed")
    else:
        read_from_packed_pkl = False
    
    curve_length_stat = []
    patch_area_stat = []
    
    files = os.listdir(data_folder)
    file_count = 0
    
    for file in files:
        if not file.endswith(".pkl"):
            continue
        
        with open(os.path.join(data_folder, file), "rb") as rf:
            while True:
                try:
                    sample = pickle.load(rf)
                except EOFError:
                    break
                
                if read_from_packed_pkl:
                    file = sample['filename']
                
                file_count += 1
                processed_sample = {}
                processed_sample['surface_points'] = sample['surface_points']
                scale = 1.0
                translation = np.zeros(3)
                
                if processed_sample['surface_points'][:,:3].min() < -0.55 or processed_sample['surface_points'][:,:3].max() > 0.55:
                    continue
                
                processed_sample['curves'] = []
                
                if len(sample['curves']) == 0:
                    continue
                
                short_curves = False
                for curve_idx, curve in enumerate(sample['curves']):
                    processed_curve = {}
                    processed_curve['points'] = scale * (curve['points'] + translation)
                    
                    if processed_curve['points'].min() < -0.55 or processed_curve['points'].max() > 0.55:
                        continue
                    
                    processed_curve['is_closed'] = curve['is_closed']
                    
                    if not curve['is_closed']:
                        processed_curve['endpoints'] = [curve['start_vert_idx'], curve['end_vert_idx']]
                    else:
                        processed_curve['endpoints'] = [-1, -1]
                    
                    processed_curve['type'] = curve_type_to_id(curve['type'])
                    processed_curve['curve_length'] = curve['curve_length'] * scale
                    
                    if processed_curve['curve_length'] < 1e-3:
                        short_curves = True
                    else:
                        curve_length_stat.append(processed_curve['curve_length'])
                    
                    processed_sample['curves'].append(processed_curve)
                
                if short_curves:
                    continue
                
                processed_sample['patches'] = []
                for patch in sample['patches']:
                    processed_patch = {}
                    processed_patch['type'] = patch_type_to_id(patch['type'])
                    processed_patch['patch_points'] = patch['patch_points']
                    
                    if 'grid_normal' in patch:
                        processed_patch['grid_normal'] = patch['grid_normal']
                    if 'u_closed' in patch:
                        processed_patch['u_closed'] = patch['u_closed']
                    if 'v_closed' in patch:
                        processed_patch['v_closed'] = patch['v_closed']
                    
                    processed_patch['patch_area'] = patch['patch_area'] * scale * scale
                    patch_area_stat.append(processed_patch['patch_area'])
                    processed_sample['patches'].append(processed_patch)
                
                sample_list.append(processed_sample)
    
    print(f"Loaded {len(sample_list)} samples from {file_count} files")
    
    curve_length_stat = np.square(np.array(curve_length_stat))
    patch_area_stat = np.array(patch_area_stat)
    
    global average_patch_area, average_squared_curve_length
    average_patch_area = patch_area_stat.mean()
    average_squared_curve_length = curve_length_stat.mean()
    
    return sample_list

flag_normal_noise = True
r_normal_noise = 0.2

class ABCDatasetOptimized(Dataset):
    def __init__(self, data, random_rotation=False, random_angle=False, flag_noise=0, 
                 flag_grid=False, num_angles=4, dim_grid=10):
        self.data = data
        self.random_rotation_augmentation = random_rotation
        self.random_angle = random_angle
        self.flag_noise = flag_noise
        self.flag_grid = flag_grid
        self.num_angles = num_angles
        self.dim_grid = dim_grid
        
        # Pre-compute rotation matrices
        self.fourteen_mat = []
        for i in range(4):
            self.fourteen_mat.append(R.from_rotvec(np.pi/2 * i * np.array([0,1,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 1 * np.array([1,0,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 3 * np.array([1,0,0])).as_matrix())
        
        c = np.sqrt(3)/3
        s = -np.sqrt(6)/3
        cornerrot1 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
            self.fourteen_mat.append(np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot1).transpose())
        
        c = -np.sqrt(3)/3
        cornerrot2 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
            self.fourteen_mat.append(np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot2).transpose())
    
    def __len__(self):
        return len(self.data)
    
    def _process_curve(self, curve, rot=None):
        points = curve['points'].astype(np.float32).copy()
        
        if rot is not None:
            points = np.matmul(points, rot).astype(np.float32)
        
        points = np.clip(points, -1000, 1000)
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        extent = max_vals - min_vals
        scale = float(extent.max())
        scale = np.clip(scale, SCALE_MIN, SCALE_MAX)
        center = (min_vals + max_vals) / 2.0
        
        normalized_points = (points - center) / scale
        normalized_points = np.clip(normalized_points, -0.6, 0.6).astype(np.float32)
        
        endpoints = np.zeros((2, 3), dtype=np.float32)
        if not curve['is_closed']:
            endpoint_indices = curve['endpoints']
            if isinstance(endpoint_indices, (list, tuple, np.ndarray)):
                idx0, idx1 = int(endpoint_indices[0]), int(endpoint_indices[1])
            else:
                idx0, idx1 = int(endpoint_indices), int(endpoint_indices)
            idx0 = max(0, min(33, idx0))
            idx1 = max(0, min(33, idx1))
            endpoints[0] = normalized_points[idx0]
            endpoints[1] = normalized_points[idx1]
        
        return {
            'curve_points': normalized_points,
            'endpoints': endpoints,
            'is_closed': bool(curve['is_closed']),
            'label': int(curve['type']),
            'scale': scale,
            'center': center
        }
    
    def _process_patch(self, patch, item_points, rot=None):
        if not self.flag_grid:
            patch_data = item_points[patch['patch_points']]
            patch_points = patch_data[:, :3].astype(np.float32)
            patch_normals = patch_data[:, 3:].astype(np.float32)
        else:
            if 'grid_normal' in patch:
                grid_data = patch['grid_normal']
                if len(grid_data) == self.dim_grid * self.dim_grid:
                    patch_points = grid_data[:, :3].astype(np.float32)
                    patch_normals = grid_data[:, 3:].astype(np.float32)
                else:
                    tmp = grid_data.reshape(20, 20, -1)
                    tmp = tmp[::2, ::2].reshape(-1, 6)
                    patch_points = tmp[:, :3].astype(np.float32)
                    patch_normals = tmp[:, 3:].astype(np.float32)
            else:
                return None
        
        if rot is not None:
            combined = np.concatenate([patch_points, patch_normals], axis=-1).reshape(-1, 3)
            combined = np.dot(combined, rot).astype(np.float32).reshape(-1, 6)
            patch_points = combined[:, :3]
            patch_normals = combined[:, 3:]
        
        # Normalize patch normals
        patch_normal_norm = np.linalg.norm(patch_normals, axis=-1, keepdims=True)
        patch_normal_norm[patch_normal_norm < th_norm] = th_norm
        patch_normals = (patch_normals / patch_normal_norm).astype(np.float32)
        
        patch_points = np.clip(patch_points, -1000, 1000)
        min_vals = patch_points.min(axis=0)
        max_vals = patch_points.max(axis=0)
        extent = max_vals - min_vals
        scale = float(extent.max())
        scale = np.clip(scale, SCALE_MIN, SCALE_MAX)
        center = (min_vals + max_vals) / 2.0
        
        normalized_points = (patch_points - center) / scale
        normalized_points = np.clip(normalized_points, -0.6, 0.6).astype(np.float32)
        
        return {
            'patch_points': normalized_points,
            'patch_normals': patch_normals,
            'u_closed': bool(patch.get('u_closed', False)),
            'v_closed': bool(patch.get('v_closed', False)),
            'label': int(patch['type']),
            'scale': scale,
            'center': center
        }
    
    def __getitem__(self, idx):
        sample_data = self.data[idx % len(self.data)]
        item_points = sample_data['surface_points'].astype(np.float32).copy()
        
        curves = [dict(c) for c in sample_data['curves']]
        patches = [dict(p) for p in sample_data['patches']]

        # Add noise if requested
        if self.flag_noise > 0:
            sigma = {1: 0.01, 2: 0.02, 3: 0.05}[self.flag_noise]
            clip = 5.0 * sigma
            jittered_data_pts = np.clip(sigma * np.random.randn(item_points.shape[0], 3), -clip, clip)
            item_points[:,:3] += jittered_data_pts
            
            if flag_normal_noise:
                normal_noise = np.random.random_sample((item_points.shape[0], 3)) * 2 - 1
                normal_noise_norm = np.linalg.norm(normal_noise, axis=-1).reshape(-1, 1)
                normal_noise_norm[normal_noise_norm < th_norm] = th_norm
                normal_noise = normal_noise / normal_noise_norm
                new_normal = item_points[:,3:] + normal_noise * r_normal_noise
                new_normal_norm = np.linalg.norm(new_normal, axis=-1).reshape(-1, 1)
                new_normal_norm[new_normal_norm < th_norm] = th_norm
                item_points[:, 3:] = new_normal / new_normal_norm

        # Rotation augmentation
        rot = None
        if self.random_rotation_augmentation:
            if not self.random_angle:
                if self.num_angles == 4:
                    rot = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
                elif self.num_angles == 56:
                    rot = self.fourteen_mat[random.randint(0,13)]
                    rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
                    rot = np.matmul(rot_z, rot)
                elif self.num_angles == 14:
                    rot = self.fourteen_mat[random.randint(0,13)]
                else:  # -1
                    rotation_angle = np.random.uniform() * 2 * np.pi
                    cosval, sinval = np.cos(rotation_angle), np.sin(rotation_angle)
                    rot = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
            else:
                rot = R.random().as_matrix()
            
            item_points = np.dot(item_points.reshape(-1, 3), rot).reshape(-1, 6).astype(np.float32)

        processed_curves = []
        for curve in curves:
            processed_curve = self._process_curve(curve, rot)
            if processed_curve is not None:
                processed_curves.append(processed_curve)
        
        processed_patches = []
        for patch in patches:
            processed_patch = self._process_patch(patch, item_points, None)
            if processed_patch is not None:
                processed_patches.append(processed_patch)
        
        return (processed_curves, processed_patches)


def collate_function_optimized(tensorlist):
    """
    """
    batch_size = len(tensorlist)
    
    all_curves = [item[0] for item in tensorlist]
    all_patches = [item[1] for item in tensorlist]
    
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
    
    processed_patches = None
    if any(len(patches) > 0 for patches in all_patches):
        max_n_patches = max(len(patches) for patches in all_patches)
        
        patch_points_batch = torch.zeros(batch_size, max_n_patches, 400, 3, dtype=torch.float32)
        patch_normals_batch = torch.zeros(batch_size, max_n_patches, 400, 3, dtype=torch.float32)
        u_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        v_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        labels_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.long)
        mask_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
        scale_batch = torch.ones(batch_size, max_n_patches, dtype=torch.float32)
        center_batch = torch.zeros(batch_size, max_n_patches, 3, dtype=torch.float32)
        
        for i, patches in enumerate(all_patches):
            n_patches = len(patches)
            if n_patches > 0:
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
    
    return (processed_curves, processed_patches)


def train_data_loader_clean(batch_size=32, data_folder="data/default/train", 
                            rotation_augmentation=False, random_angle=False, 
                            flag_noise=0, flag_grid=False, num_angle=4, dim_grid=10,
                            num_workers=8, rank=0, world_size=1):
    # Pack pickle files if needed
    if not os.path.exists(os.path.join(data_folder, "packed")):
        if rank == 0:
            os.makedirs(os.path.join(data_folder, "packed"), exist_ok=True)
            pack_pickle_files(data_folder, os.path.join(data_folder, "packed"))
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
    
    train_dataset = ABCDatasetOptimized(
        data_loader_ABC(data_folder),
        random_rotation=rotation_augmentation,
        random_angle=random_angle,
        flag_noise=flag_noise,
        flag_grid=flag_grid,
        num_angles=num_angle,
        dim_grid=dim_grid
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        train_data = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_function_optimized,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            # persistent_workers=(num_workers > 0),
            # prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            multiprocessing_context='fork' if num_workers > 0 else None
        )
        return train_data, train_sampler
    else:
        train_data = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_function_optimized,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            # prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            multiprocessing_context='fork' if num_workers > 0 else None
        )
        return train_data, None