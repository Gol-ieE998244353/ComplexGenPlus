import numpy as np
import sys
# try:
#     from numpy import core
#     if 'numpy._core' not in sys.modules:
#         sys.modules['numpy._core'] = core
#     if 'numpy._core.multiarray' not in sys.modules:
#         sys.modules['numpy._core.multiarray'] = core.multiarray
# except ImportError:
#     pass
from numpy import core
import os
import torch
import math
from scipy.spatial.transform import Rotation as R
import pickle
import random
import gc
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from numpy import linalg

average_patch_area = 0
average_squared_curve_length = 0
pack_size = 10000
th_norm = 1e-6
points_per_curve_dim = 34

# 常量
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
    packed_file = None
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
                packed_file = None
    if packed_file is not None and not packed_file.closed:
        packed_file.close()


def _curve_type_to_id(s):
    """曲线类型转ID"""
    if isinstance(s, (int, np.integer)):
        return int(s)
    return {'Circle': 0, 'BSpline': 1, 'Line': 2}.get(s, 3)

def _patch_type_to_id(s):
    """面片类型转ID"""
    if isinstance(s, (int, np.integer)):
        return int(s)
    mapping = {'Cylinder': 0, 'Torus': 1, 'BSpline': 2, 'Extrusion': 2, 
               'Revolution': 2, 'Plane': 3, 'Cone': 4, 'Sphere': 5}
    return mapping.get(s, 5)


class LazyDataIndex:
    """惰性数据索引 - 只存储文件位置信息，不加载实际数据"""
    __slots__ = ['packed_file', 'offset', 'length', 'filename', 'curve_count', 'patch_count']
    
    def __init__(self, packed_file, offset, length, filename=None, curve_count=0, patch_count=0):
        self.packed_file = packed_file
        self.offset = offset
        self.length = length
        self.filename = filename
        self.curve_count = int(curve_count)
        self.patch_count = int(patch_count)


def build_lazy_index(data_folder):
    """构建惰性索引，只扫描文件位置"""
    print(f"Building lazy index from {data_folder}")
    
    if os.path.exists(os.path.join(data_folder, "packed")):
        data_folder = os.path.join(data_folder, "packed")
        read_from_packed = True
    else:
        read_from_packed = False
    
    index_list = []
    curve_length_stat = []
    patch_area_stat = []
    files = sorted(os.listdir(data_folder))
    
    for file in files:
        if not file.endswith(".pkl"):
            continue
        
        filepath = os.path.join(data_folder, file)
        
        if read_from_packed:
            with open(filepath, "rb") as rf:
                while True:
                    offset = rf.tell()
                    try:
                        sample = pickle.load(rf)
                    except EOFError:
                        break
                    
                    if not _validate_sample_quick(sample):
                        continue
                    
                    length = rf.tell() - offset
                    filename = sample.get('filename') if isinstance(sample, dict) else None
                    curve_count = len(sample.get('curves', []))
                    patch_count = len(sample.get('patches', []))
                    index_list.append(LazyDataIndex(filepath, offset, length, filename, curve_count, patch_count))
                    
                    for curve in sample.get('curves', []):
                        if curve.get('curve_length', 0) >= 1e-3:
                            curve_length_stat.append(curve['curve_length'])
                    for patch in sample.get('patches', []):
                        patch_area_stat.append(patch.get('patch_area', 0))
        else:
            with open(filepath, "rb") as rf:
                try:
                    sample = pickle.load(rf)
                except Exception:
                    continue
            if not _validate_sample_quick(sample):
                continue
            curve_count = len(sample.get('curves', []))
            patch_count = len(sample.get('patches', []))
            index_list.append(LazyDataIndex(filepath, 0, -1, os.path.basename(filepath), curve_count, patch_count))
    
    global average_patch_area, average_squared_curve_length
    if curve_length_stat:
        average_squared_curve_length = np.mean(np.square(curve_length_stat))
    if patch_area_stat:
        average_patch_area = np.mean(patch_area_stat)
    
    print(f"Built lazy index with {len(index_list)} samples")
    gc.collect()
    
    return index_list, data_folder


def _validate_sample_quick(sample):
    """快速验证样本有效性"""
    if not isinstance(sample, dict) or 'surface_points' not in sample:
        return False
    pts = np.asarray(sample['surface_points'])
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return False
    if not np.isfinite(pts[:, :3]).all():
        return False
    curves = sample.get('curves', [])
    if not curves:
        return False
    for curve in curves:
        curve_len = curve.get('curve_length')
        if curve_len is None:
            points = curve.get('points')
            if isinstance(points, (list, np.ndarray)) and len(points) >= 2:
                return True
        elif curve_len >= 1e-6:
            return True
    return False


def load_raw_sample(index_item):
    """根据惰性索引读取原始样本"""
    with open(index_item.packed_file, "rb") as rf:
        if index_item.offset > 0:
            rf.seek(index_item.offset)
        sample = pickle.load(rf)
    return sample


def load_and_process_sample(index_item):
    """按需加载并处理单个样本"""
    sample = load_raw_sample(index_item)

    surface_points = np.asarray(sample['surface_points'], dtype=np.float32)
    need_global_norm = False
    if surface_points.size and surface_points.shape[1] >= 3:
        min_val = surface_points[:, :3].min()
        max_val = surface_points[:, :3].max()
        need_global_norm = (min_val < -0.55) or (max_val > 0.55)

    global_center = None
    global_scale = 1.0
    if need_global_norm and surface_points.size:
        mins = surface_points[:, :3].min(axis=0)
        maxs = surface_points[:, :3].max(axis=0)
        global_center = (mins + maxs) * 0.5
        global_scale = float((maxs - mins).max())
        if global_scale < 1e-8:
            global_scale = 1.0
        surface_points[:, :3] = (surface_points[:, :3] - global_center) / global_scale
        np.clip(surface_points[:, :3], -0.6, 0.6, out=surface_points[:, :3])

    length_scale = 1.0 / global_scale if need_global_norm else 1.0
    area_scale = length_scale * length_scale

    processed = {
        'surface_points': surface_points,
        'curves': [],
        'patches': []
    }

    for curve in sample.get('curves', []):
        pts = np.asarray(curve.get('points', []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
            continue
        if need_global_norm and global_center is not None:
            pts = (pts - global_center) / global_scale
        endpoints = curve.get('endpoints')
        if endpoints is None:
            endpoints = [-1, -1] if curve.get('is_closed', False) else [0, pts.shape[0] - 1]
        processed['curves'].append({
            'points': pts.astype(np.float32),
            'is_closed': bool(curve.get('is_closed', False)),
            'endpoints': endpoints,
            'type': _curve_type_to_id(curve.get('type')),
            'curve_length': float(curve.get('curve_length', 0.0)) * length_scale
        })

    for patch in sample.get('patches', []):
        grid_normal = patch.get('grid_normal')
        if grid_normal is not None:
            grid_normal = np.asarray(grid_normal, dtype=np.float32)
            if grid_normal.ndim == 3:
                grid_normal = grid_normal.reshape(-1, grid_normal.shape[-1])
            if need_global_norm and global_center is not None and grid_normal.size:
                grid_normal[:, :3] = (grid_normal[:, :3] - global_center) / global_scale

        patch_points = patch.get('patch_points')
        if patch_points is not None and not isinstance(patch_points, list):
            patch_points = np.asarray(patch_points, dtype=np.float32)
            if patch_points.ndim == 2 and patch_points.shape[1] >= 3:
                if need_global_norm and global_center is not None:
                    patch_points[:, :3] = (patch_points[:, :3] - global_center) / global_scale

        processed['patches'].append({
            'type': _patch_type_to_id(patch.get('type')),
            'patch_points': patch_points,
            'curves': patch.get('curves', []),
            'grid_normal': grid_normal,
            'u_closed': patch.get('u_closed', False),
            'v_closed': patch.get('v_closed', False),
            'patch_area': float(patch.get('patch_area', 0.0)) * area_scale
        })

    return processed


# ============ 新增：点云体素化函数（从文档二移植） ============
def points2sparse_voxel(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    """
    向量化优化的体素化函数 - 速度提升 10x-50x
    """
    # 1. 坐标转换
    points = points_with_normal[:, :3] + 0.5  # to [0, 1]
    voxel_length = 1.0 / voxel_dim
    # 离散化坐标
    voxel_coord = np.clip(np.floor(points / voxel_length).astype(np.int32), 0, voxel_dim - 1)
    
    # 2. 计算特征
    points_normal_norm = linalg.norm(points_with_normal[:, 3:], axis=1, keepdims=True)
    points_normal_norm[points_normal_norm < th_norm] = th_norm
    
    if feature_type == 'local':
        local_pos = (points - voxel_coord.astype(np.float32) * voxel_length) * voxel_dim - 0.5
    elif feature_type == 'global':
        local_pos = points - 0.5
        
    # 构造原始特征: [Location, Normal, Count=1]
    feats = np.concatenate([
        local_pos, 
        points_with_normal[:, 3:] / points_normal_norm, 
        np.ones([local_pos.shape[0], 1], dtype=np.float32)
    ], axis=-1)

    # 3. 优化：使用 Numpy 向量化替代字典循环
    # 将 3D 坐标哈希为 1D 索引以便排序/去重
    keys = voxel_coord[:, 0] * (voxel_dim ** 2) + voxel_coord[:, 1] * voxel_dim + voxel_coord[:, 2]
    
    # 获取唯一体素的索引
    _, unique_indices, inverse_indices = np.unique(keys, return_index=True, return_inverse=True)
    
    # 聚合特征 (Mean Pooling)
    # 使用 np.add.at 进行聚合 (比循环快得多)
    num_unique = len(unique_indices)
    aggregated_features = np.zeros((num_unique, feats.shape[1]), dtype=np.float32)
    np.add.at(aggregated_features, inverse_indices, feats)
    
    # 恢复对应的 Voxel 坐标
    unique_voxel_coords = voxel_coord[unique_indices]
    
    # 4. 后处理 (归一化)
    points_in_voxel = aggregated_features[:, 6:]
    features = aggregated_features / points_in_voxel  # 平均化
    
    position = features[:, :3]
    normals = features[:, 3:6]
    pad_ones = features[:, 6:]
    
    # Normalize normals again
    normals /= (linalg.norm(normals, axis=-1, keepdims=True) + 1e-10)
    
    if with_normal and pad1s:
        final_features = np.concatenate([position, normals, pad_ones], axis=1)
    elif pad1s:
        final_features = np.concatenate([position, pad_ones], axis=1)
    elif with_normal:
        final_features = np.concatenate([position, normals], axis=1)
    else:
        final_features = position
        
    return unique_voxel_coords.astype(np.int32), final_features.astype(np.float32)


_ROTATION_MATRICES = None

def _get_rotation_matrices():
    """获取预计算的旋转矩阵"""
    global _ROTATION_MATRICES
    if _ROTATION_MATRICES is None:
        mats = []
        for i in range(4):
            mats.append(R.from_rotvec(np.pi/2 * i * np.array([0,1,0])).as_matrix().astype(np.float32))
        mats.append(R.from_rotvec(np.pi/2 * 1 * np.array([1,0,0])).as_matrix().astype(np.float32))
        mats.append(R.from_rotvec(np.pi/2 * 3 * np.array([1,0,0])).as_matrix().astype(np.float32))
        
        c, s = np.sqrt(3)/3, -np.sqrt(6)/3
        cornerrot1 = np.array([[c,0,-s],[0,1,0],[s,0,c]], dtype=np.float32)
        for i in range(4):
            rot = np.matmul(R.from_rotvec((np.pi/2 * i + np.pi/4) * np.array([0,0,1])).as_matrix(), cornerrot1)
            mats.append(rot.T.astype(np.float32))
        
        c = -np.sqrt(3)/3
        cornerrot2 = np.array([[c,0,-s],[0,1,0],[s,0,c]], dtype=np.float32)
        for i in range(4):
            rot = np.matmul(R.from_rotvec((np.pi/2 * i + np.pi/4) * np.array([0,0,1])).as_matrix(), cornerrot2)
            mats.append(rot.T.astype(np.float32))
        
        _ROTATION_MATRICES = mats
    return _ROTATION_MATRICES


flag_normal_noise = True
r_normal_noise = 0.2


class ABCDatasetOptimized(Dataset):
    """内存优化的数据集 - 惰性加载"""
    
    def __init__(self, data_folder, random_rotation=False, random_angle=False, 
                 flag_noise=0, flag_curve_noise=0, flag_patch_noise=0,
                 flag_grid=False, num_angles=4, dim_grid=10,
                 with_pointcloud=False, voxel_dim=32, feature_type='global', 
                 with_normal=True, pad1s=True):  # 新增：点云相关参数
        self.index_list, self.data_folder = build_lazy_index(data_folder)
        self.random_rotation_augmentation = random_rotation
        self.random_angle = random_angle
        self.flag_noise = flag_noise
        self.flag_curve_noise = flag_curve_noise
        self.flag_patch_noise = flag_patch_noise
        self.flag_grid = flag_grid
        self.num_angles = num_angles
        self.dim_grid = dim_grid
        
        # 新增：点云处理参数
        self.with_pointcloud = with_pointcloud
        self.voxel_dim = voxel_dim
        self.feature_type = feature_type
        self.with_normal = with_normal
        self.pad1s = pad1s
        
        self.rotation_matrices = _get_rotation_matrices()

    @staticmethod
    def _add_gaussian_noise(points, noise_level):
        if noise_level <= 0:
            return points
        sigma = {1: 0.01, 2: 0.02, 3: 0.05, 4: 0.5}[noise_level]
        clip = 5.0 * sigma
        noise = np.clip(
            sigma * np.random.randn(points.shape[0], points.shape[1]),
            -clip,
            clip
        ).astype(np.float32)
        return points + noise
    
    def __len__(self):
        return len(self.index_list)
    
    def _get_rotation(self):
        """获取随机旋转矩阵"""
        if not self.random_rotation_augmentation:
            return None
        
        if not self.random_angle:
            if self.num_angles == 4:
                return R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix().astype(np.float32)
            elif self.num_angles == 56:
                rot = self.rotation_matrices[random.randint(0, 13)]
                rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
                return np.matmul(rot_z, rot).astype(np.float32)
            elif self.num_angles == 14:
                return self.rotation_matrices[random.randint(0, 13)]
            else:
                angle = random.random() * 2 * np.pi
                c, s = np.cos(angle), np.sin(angle)
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:
            return R.random().as_matrix().astype(np.float32)
    
    def _process_curve(self, curve, rot=None):
        """处理单个curve"""
        points = np.asarray(curve.get('points', []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            return None
        
        if rot is not None:
            points = np.dot(points, rot)

        if self.flag_curve_noise > 0:
            points = self._add_gaussian_noise(points, self.flag_curve_noise)
        
        np.clip(points, -1000, 1000, out=points)
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        scale = float((max_vals - min_vals).max())
        scale = max(SCALE_MIN, min(SCALE_MAX, scale))
        center = (min_vals + max_vals) * 0.5
        
        normalized = (points - center) / scale
        np.clip(normalized, -0.6, 0.6, out=normalized)
        
        endpoints = np.zeros((2, 3), dtype=np.float32)
        if not curve.get('is_closed', False):
            ep = curve.get('endpoints')
            last_idx = normalized.shape[0] - 1
            if isinstance(ep, (list, tuple, np.ndarray)) and len(ep) == 2 and np.issubdtype(np.asarray(ep).dtype, np.number):
                idx0 = max(0, min(last_idx, int(ep[0])))
                idx1 = max(0, min(last_idx, int(ep[1])))
            else:
                idx0, idx1 = 0, last_idx
            endpoints[0] = normalized[idx0]
            endpoints[1] = normalized[idx1]
        
        return {
            'curve_points': normalized.astype(np.float32),
            'endpoints': endpoints,
            'is_closed': bool(curve['is_closed']),
            'label': int(curve['type']),
            'scale': float(scale),
            'center': center.astype(np.float32)
        }
    
    def _process_patch(self, patch, item_points, rot=None):
        """处理单个patch"""
        use_indices = False
        if not self.flag_grid:
            patch_points_data = patch.get('patch_points')
            if isinstance(patch_points_data, np.ndarray):
                patch_data = patch_points_data.astype(np.float32)
            elif isinstance(patch_points_data, (list, tuple)) and patch_points_data:
                if isinstance(patch_points_data[0], (list, tuple, np.ndarray)):
                    patch_data = np.asarray(patch_points_data, dtype=np.float32)
                else:
                    patch_data = None
            else:
                patch_data = np.asarray(patch_points_data, dtype=np.float32)
            
            if patch_data is None or patch_data.ndim != 2:
                patch_indices = patch.get('patch_points', [])
                patch_data = item_points[patch_indices]
                use_indices = True
            if patch_data.shape[1] >= 6:
                patch_points = patch_data[:, :3]
                patch_normals = patch_data[:, 3:6]
            else:
                patch_points = patch_data[:, :3]
                patch_normals = np.zeros_like(patch_points)
        else:
            grid_data = patch.get('grid_normal')
            if grid_data is None:
                return None

            grid_data = np.asarray(grid_data, dtype=np.float32)
            if grid_data.ndim == 3:
                grid_data = grid_data.reshape(-1, grid_data.shape[-1])
            grid_len = grid_data.shape[0]
            if grid_len == self.dim_grid * self.dim_grid:
                patch_points = grid_data[:, :3]
                patch_normals = grid_data[:, 3:6]
            else:
                grid_side = int(math.sqrt(grid_len))
                if grid_side * grid_side == grid_len and grid_side >= self.dim_grid:
                    tmp = grid_data.reshape(grid_side, grid_side, -1)
                    if grid_side % self.dim_grid == 0:
                        step = grid_side // self.dim_grid
                        tmp = tmp[::step, ::step]
                    else:
                        idx = np.round(np.linspace(0, grid_side - 1, self.dim_grid)).astype(int)
                        tmp = tmp[idx][:, idx]
                    tmp = tmp.reshape(-1, 6)
                    patch_points = tmp[:, :3]
                    patch_normals = tmp[:, 3:6]
                else:
                    return None

        if rot is not None and not use_indices:
            patch_points = np.dot(patch_points, rot)
            patch_normals = np.dot(patch_normals, rot)

        if self.flag_patch_noise > 0:
            patch_points = self._add_gaussian_noise(patch_points, self.flag_patch_noise)
        
        norm = np.linalg.norm(patch_normals, axis=-1, keepdims=True)
        norm = np.maximum(norm, th_norm)
        patch_normals = patch_normals / norm
        
        np.clip(patch_points, -1000, 1000, out=patch_points)
        min_vals = patch_points.min(axis=0)
        max_vals = patch_points.max(axis=0)
        scale = float((max_vals - min_vals).max())
        scale = max(SCALE_MIN, min(SCALE_MAX, scale))
        center = (min_vals + max_vals) * 0.5
        
        normalized = (patch_points - center) / scale
        np.clip(normalized, -0.6, 0.6, out=normalized)
        
        return {
            'patch_points': normalized.astype(np.float32),
            'patch_normals': patch_normals.astype(np.float32),
            'u_closed': bool(patch.get('u_closed', False)),
            'v_closed': bool(patch.get('v_closed', False)),
            'label': int(patch['type']),
            'curves': patch.get('curves', []),
            'scale': float(scale),
            'center': center.astype(np.float32)
        }
    
    def __getitem__(self, idx):
        sample_data = load_and_process_sample(self.index_list[idx % len(self.index_list)])
        item_points = sample_data['surface_points']
        
        # 添加噪声
        if self.flag_noise > 0:
            item_points[:, :3] = self._add_gaussian_noise(item_points[:, :3], self.flag_noise)
            
            if flag_normal_noise:
                normal_noise = (np.random.random((item_points.shape[0], 3)) * 2 - 1).astype(np.float32)
                norm = np.linalg.norm(normal_noise, axis=-1, keepdims=True)
                norm = np.maximum(norm, th_norm)
                normal_noise /= norm
                new_normal = item_points[:, 3:] + normal_noise * r_normal_noise
                norm = np.linalg.norm(new_normal, axis=-1, keepdims=True)
                norm = np.maximum(norm, th_norm)
                item_points[:, 3:] = new_normal / norm
        
        # 旋转增强
        rot = self._get_rotation()
        if rot is not None:
            item_points = np.dot(item_points.reshape(-1, 3), rot).reshape(-1, 6)
        
        # 处理curves和patches
        processed_curves = []
        for curve in sample_data['curves']:
            pc = self._process_curve(curve, rot)
            if pc is not None:
                processed_curves.append(pc)
        
        processed_patches = []
        for patch in sample_data['patches']:
            pp = self._process_patch(patch, item_points, rot)
            if pp is not None:
                processed_patches.append(pp)
        
        # 新增：条件性处理点云
        locations, features = None, None
        if self.with_pointcloud:
            locations, features = points2sparse_voxel(
                item_points, self.voxel_dim, self.feature_type, 
                self.with_normal, self.pad1s
            )
        
        return (locations, features, processed_curves, processed_patches)


class MixedABCDataset(Dataset):
    """Concatenate multiple ABC datasets while keeping source-level metadata for ratio sampling."""

    def __init__(self, data_folders, sampling_weights=None, **dataset_kwargs):
        if not data_folders:
            raise ValueError("data_folders must contain at least one dataset folder")

        self.datasets = [ABCDatasetOptimized(folder, **dataset_kwargs) for folder in data_folders]
        self.data_folders = list(data_folders)
        self.grouped_indices = []
        self.index_map = []
        self.index_list = []
        self.sample_counts = []

        offset = 0
        for dataset_id, dataset in enumerate(self.datasets):
            dataset_indices = list(range(offset, offset + len(dataset)))
            self.grouped_indices.append(dataset_indices)
            for local_idx, index_item in enumerate(dataset.index_list):
                self.index_map.append((dataset_id, local_idx))
                self.index_list.append(index_item)
                self.sample_counts.append(index_item.patch_count + index_item.curve_count)
            offset += len(dataset)

        self.sampling_weights = _normalize_sampling_weights(sampling_weights, len(self.datasets))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_id, local_idx = self.index_map[idx]
        return self.datasets[dataset_id][local_idx]


def _normalize_sampling_weights(weights, num_groups):
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if weights is None:
        return [1.0 / num_groups] * num_groups
    if len(weights) != num_groups:
        raise ValueError("sampling_weights length must match number of datasets")

    normalized = [float(w) for w in weights]
    if any(w < 0 for w in normalized):
        raise ValueError("sampling_weights must be non-negative")

    total = sum(normalized)
    if total <= 0:
        raise ValueError("sampling_weights must sum to a positive value")
    return [w / total for w in normalized]


def _compute_group_sample_counts(total_size, sampling_weights):
    raw_counts = [total_size * weight for weight in sampling_weights]
    counts = [int(math.floor(count)) for count in raw_counts]
    remainder = total_size - sum(counts)
    if remainder > 0:
        order = sorted(
            range(len(raw_counts)),
            key=lambda idx: raw_counts[idx] - counts[idx],
            reverse=True,
        )
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def _sample_grouped_indices(grouped_indices, sampling_weights, total_size, rng, shuffle=True):
    if total_size <= 0:
        return []

    sampled_indices = []
    counts = _compute_group_sample_counts(total_size, sampling_weights)
    for group_idx, sample_count in enumerate(counts):
        if sample_count <= 0:
            continue
        candidates = grouped_indices[group_idx]
        if not candidates:
            raise ValueError(f"Dataset group {group_idx} is empty and cannot be sampled")
        sampled_indices.extend(rng.choices(candidates, k=sample_count))

    if shuffle:
        rng.shuffle(sampled_indices)
    return sampled_indices


def collate_function_global_topology(batch_list):
    """
    全局拓扑处理的collate函数 - 支持点云
    """
    batch_size = len(batch_list)
    
    # 新增：提取点云数据
    all_locations = [item[0] for item in batch_list]
    all_features = [item[1] for item in batch_list]
    all_curves = [item[2] for item in batch_list]
    all_patches = [item[3] for item in batch_list]
    
    # 新增：拼接点云数据（如果存在）
    pointcloud_data = None
    if all_locations[0] is not None:  # 检查是否有点云数据
        locations_with_batch = [
            np.concatenate([all_locations[i], np.ones([all_locations[i].shape[0], 1], dtype=np.int32)*i], axis=-1) 
            for i in range(batch_size)
        ]
        pointcloud_data = (
            torch.from_numpy(np.concatenate(locations_with_batch, axis=0)),
            torch.from_numpy(np.concatenate(all_features, axis=0))
        )
    
    # === Step 1: 收集curves数据 ===
    processed_curves = None
    valid_curves = [c for c in all_curves if len(c) > 0]
    
    if valid_curves:
        max_n_curves = max(len(c) for c in all_curves)
        if max_n_curves > 0:
            curve_points_batch = torch.zeros(batch_size, max_n_curves, 34, 3, dtype=torch.float32)
            endpoints_batch = torch.zeros(batch_size, max_n_curves, 2, 3, dtype=torch.float32)
            is_closed_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool)
            labels_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.long)
            mask_batch = torch.zeros(batch_size, max_n_curves, dtype=torch.bool)
            scale_batch = torch.ones(batch_size, max_n_curves, dtype=torch.float32)
            center_batch = torch.zeros(batch_size, max_n_curves, 3, dtype=torch.float32)
            
            for i, curves in enumerate(all_curves):
                for j, curve in enumerate(curves):
                    if j >= max_n_curves:
                        break
                    curve_points_batch[i, j] = torch.from_numpy(curve['curve_points'])
                    endpoints_batch[i, j] = torch.from_numpy(curve['endpoints'])
                    is_closed_batch[i, j] = curve['is_closed']
                    labels_batch[i, j] = curve['label']
                    scale_batch[i, j] = float(curve['scale'])
                    center_batch[i, j] = torch.from_numpy(np.asarray(curve['center'], dtype=np.float32))
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
    
    # === Step 2: 收集patches数据 ===
    processed_patches = None
    valid_patches = [p for p in all_patches if len(p) > 0]
    
    if valid_patches:
        max_n_patches = max(len(p) for p in all_patches)
        if max_n_patches > 0:
            first_patch = valid_patches[0][0] if valid_patches[0] else None
            n_patch_pts = 100 if first_patch and len(first_patch['patch_points']) <= 100 else 400
            
            patch_points_batch = torch.zeros(batch_size, max_n_patches, n_patch_pts, 3, dtype=torch.float32)
            patch_normals_batch = torch.zeros(batch_size, max_n_patches, n_patch_pts, 3, dtype=torch.float32)
            u_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
            v_closed_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
            labels_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.long)
            mask_batch = torch.zeros(batch_size, max_n_patches, dtype=torch.bool)
            scale_batch = torch.ones(batch_size, max_n_patches, dtype=torch.float32)
            center_batch = torch.zeros(batch_size, max_n_patches, 3, dtype=torch.float32)
            patches_curves_info = []
            
            for i, patches in enumerate(all_patches):
                batch_patches_curves = []
                for j, patch in enumerate(patches):
                    if j >= max_n_patches:
                        break
                    
                    pts = patch['patch_points']
                    norms = patch['patch_normals']
                    n_pts = min(len(pts), n_patch_pts)
                    
                    patch_points_batch[i, j, :n_pts] = torch.from_numpy(pts[:n_pts])
                    patch_normals_batch[i, j, :n_pts] = torch.from_numpy(norms[:n_pts])
                    u_closed_batch[i, j] = patch['u_closed']
                    v_closed_batch[i, j] = patch['v_closed']
                    labels_batch[i, j] = patch['label']
                    scale_batch[i, j] = float(patch['scale'])
                    center_batch[i, j] = torch.from_numpy(np.asarray(patch['center'], dtype=np.float32))
                    mask_batch[i, j] = True
                    batch_patches_curves.append(patch.get('curves', []))
                
                patches_curves_info.append(batch_patches_curves)
            
            processed_patches = {
                "patch_points": patch_points_batch,
                "patch_normals": patch_normals_batch,
                "u_closed": u_closed_batch,
                "v_closed": v_closed_batch,
                "labels": labels_batch,
                "mask": mask_batch,
                "scale": scale_batch,
                "center": center_batch,
                "curves_info": patches_curves_info
            }
    
    # 新增：返回值包含点云数据（如果有）
    if pointcloud_data is not None:
        return (pointcloud_data, processed_curves, processed_patches)
    else:
        return (processed_curves, processed_patches)


class BucketedDynamicBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups by size buckets and caps total items per batch.
    不把相近复杂度样本分桶训练的话会爆显存，尤其是当batch_size较大时。
    这个采样器会根据样本的复杂度（如曲线数量、patch数量等）将样本分桶，并在每个batch中控制总的复杂度不超过max_total_items，
    从而更稳定地训练。
    """
    def __init__(
        self,
        counts,
        batch_size,
        max_total_items=None,
        bucket_boundaries=None,
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.counts = list(counts)
        self.batch_size = batch_size
        self.max_total_items = max_total_items
        self.boundaries = bucket_boundaries or [10, 30, 60, 100, 200, 400]
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _bucket_id(self, count):
        for i, bound in enumerate(self.boundaries):
            if count <= bound:
                return i
        return len(self.boundaries)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.counts)))
        if self.shuffle:
            rng.shuffle(indices)
        if self.world_size > 1:
            indices = indices[self.rank::self.world_size]

        buckets = {i: [] for i in range(len(self.boundaries) + 1)}
        for idx in indices:
            buckets[self._bucket_id(self.counts[idx])].append(idx)
        for key in buckets:
            if self.shuffle:
                rng.shuffle(buckets[key])

        for key in buckets:
            bucket = buckets[key]
            batch = []
            total = 0
            while bucket:
                idx = bucket.pop()
                cost = max(1, int(self.counts[idx]))
                if self.max_total_items is not None and batch and total + cost > self.max_total_items:
                    if not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                        yield batch
                    batch = []
                    total = 0
                batch.append(idx)
                total += cost
                if self.batch_size is not None and len(batch) >= self.batch_size:
                    if not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                        yield batch
                    batch = []
                    total = 0
            if batch and not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                yield batch

    def __len__(self):
        indices = list(range(len(self.counts)))
        if self.world_size > 1:
            indices = indices[self.rank::self.world_size]
        if not indices:
            return 0
        if self.max_total_items is None:
            if self.batch_size is None:
                return len(indices)
            return int(math.ceil(len(indices) / float(self.batch_size)))
        batch_count = 0
        total = 0
        batch_len = 0
        for idx in indices:
            cost = max(1, int(self.counts[idx]))
            if self.batch_size is not None and batch_len >= self.batch_size:
                batch_count += 1
                total = 0
                batch_len = 0
            if total + cost > self.max_total_items and batch_len > 0:
                batch_count += 1
                total = 0
                batch_len = 0
            total += cost
            batch_len += 1
        if batch_len > 0 and not (self.drop_last and self.batch_size and batch_len < self.batch_size):
            batch_count += 1
        return batch_count


class RatioDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that draws samples from multiple datasets using fixed source ratios."""

    def __init__(
        self,
        grouped_indices,
        sampling_weights,
        num_samples=None,
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.grouped_indices = [list(group) for group in grouped_indices]
        self.sampling_weights = _normalize_sampling_weights(sampling_weights, len(self.grouped_indices))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        base_num_samples = num_samples
        if base_num_samples is None:
            base_num_samples = sum(len(group) for group in self.grouped_indices)

        if drop_last:
            self.total_size = (base_num_samples // self.world_size) * self.world_size
        else:
            self.total_size = int(math.ceil(base_num_samples / float(self.world_size))) * self.world_size

        if self.total_size == 0 and base_num_samples > 0:
            self.total_size = self.world_size

        self.num_samples = self.total_size // self.world_size if self.world_size > 0 else 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        sampled_indices = _sample_grouped_indices(
            self.grouped_indices,
            self.sampling_weights,
            self.total_size,
            rng,
            shuffle=self.shuffle,
        )
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(sampled_indices[start:end])

    def __len__(self):
        return self.num_samples


class RatioBucketedDynamicBatchSampler(torch.utils.data.Sampler):
    """Bucketed sampler with per-dataset ratio sampling and DDP-safe epoch sizes."""

    def __init__(
        self,
        counts,
        grouped_indices,
        sampling_weights,
        batch_size,
        max_total_items=None,
        bucket_boundaries=None,
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
        num_samples=None,
    ):
        self.counts = list(counts)
        self.grouped_indices = [list(group) for group in grouped_indices]
        self.sampling_weights = _normalize_sampling_weights(sampling_weights, len(self.grouped_indices))
        self.batch_size = batch_size
        self.max_total_items = max_total_items
        self.boundaries = bucket_boundaries or [10, 30, 60, 100, 200, 400]
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        base_num_samples = num_samples
        if base_num_samples is None:
            base_num_samples = sum(len(group) for group in self.grouped_indices)

        if drop_last:
            self.total_size = (base_num_samples // self.world_size) * self.world_size
        else:
            self.total_size = int(math.ceil(base_num_samples / float(self.world_size))) * self.world_size

        if self.total_size == 0 and base_num_samples > 0:
            self.total_size = self.world_size

        self.num_samples = self.total_size // self.world_size if self.world_size > 0 else 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _bucket_id(self, count):
        for i, bound in enumerate(self.boundaries):
            if count <= bound:
                return i
        return len(self.boundaries)

    def _build_rank_indices(self):
        rng = random.Random(self.seed + self.epoch)
        sampled_indices = _sample_grouped_indices(
            self.grouped_indices,
            self.sampling_weights,
            self.total_size,
            rng,
            shuffle=self.shuffle,
        )
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return sampled_indices[start:end]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = self._build_rank_indices()

        buckets = {i: [] for i in range(len(self.boundaries) + 1)}
        for idx in indices:
            buckets[self._bucket_id(self.counts[idx])].append(idx)
        for key in buckets:
            if self.shuffle:
                rng.shuffle(buckets[key])

        for key in buckets:
            bucket = buckets[key]
            batch = []
            total = 0
            while bucket:
                idx = bucket.pop()
                cost = max(1, int(self.counts[idx]))
                if self.max_total_items is not None and batch and total + cost > self.max_total_items:
                    if not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                        yield batch
                    batch = []
                    total = 0
                batch.append(idx)
                total += cost
                if self.batch_size is not None and len(batch) >= self.batch_size:
                    if not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                        yield batch
                    batch = []
                    total = 0
            if batch and not (self.drop_last and self.batch_size and len(batch) < self.batch_size):
                yield batch

    def __len__(self):
        indices = self._build_rank_indices()
        if not indices:
            return 0
        if self.max_total_items is None:
            if self.batch_size is None:
                return len(indices)
            return int(math.ceil(len(indices) / float(self.batch_size)))

        batch_count = 0
        total = 0
        batch_len = 0
        for idx in indices:
            cost = max(1, int(self.counts[idx]))
            if self.batch_size is not None and batch_len >= self.batch_size:
                batch_count += 1
                total = 0
                batch_len = 0
            if total + cost > self.max_total_items and batch_len > 0:
                batch_count += 1
                total = 0
                batch_len = 0
            total += cost
            batch_len += 1
        if batch_len > 0 and not (self.drop_last and self.batch_size and batch_len < self.batch_size):
            batch_count += 1
        return batch_count


def data_loader_ABC(data_folder):
    """兼容接口 - 返回惰性索引列表"""
    index_list, _ = build_lazy_index(data_folder)
    return index_list


def train_data_loader_clean(batch_size=32, data_folder="data/default/train", 
                            rotation_augmentation=False, random_angle=False, 
                            flag_noise=0, flag_curve_noise=0, flag_patch_noise=0,
                            flag_grid=False, num_angle=4, dim_grid=10,
                            num_workers=8, rank=0, world_size=1,
                            with_pointcloud=False, voxel_dim=128, feature_type='global',
                            with_normal=True, pad1s=True,
                            use_bucketed_batch=False, max_total_items=None,
                            bucket_boundaries=None, shuffle=True,
                            data_folders=None, dataset_sampling_weights=None):  # 新增：点云相关参数
    """
    带全局拓扑处理的数据加载器 - 内存优化版 + 点云支持
    """
    dataset_folders = data_folders or [data_folder]
    for folder in dataset_folders:
        if not os.path.exists(os.path.join(folder, "packed")):
            if rank == 0:
                os.makedirs(os.path.join(folder, "packed"), exist_ok=True)
                pack_pickle_files(folder, os.path.join(folder, "packed"))
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()

    dataset_kwargs = dict(
        random_rotation=rotation_augmentation,
        random_angle=random_angle,
        flag_noise=flag_noise,
        flag_curve_noise=flag_curve_noise,
        flag_patch_noise=flag_patch_noise,
        flag_grid=flag_grid,
        num_angles=num_angle,
        dim_grid=dim_grid,
        with_pointcloud=with_pointcloud,
        voxel_dim=voxel_dim,
        feature_type=feature_type,
        with_normal=with_normal,
        pad1s=pad1s,
    )

    is_mixed_dataset = len(dataset_folders) > 1
    if is_mixed_dataset:
        train_dataset = MixedABCDataset(
            dataset_folders,
            sampling_weights=dataset_sampling_weights,
            **dataset_kwargs,
        )
    else:
        train_dataset = ABCDatasetOptimized(dataset_folders[0], **dataset_kwargs)
    
    effective_workers = min(num_workers, 2)
    
    if use_bucketed_batch or max_total_items is not None:
        counts = train_dataset.sample_counts if is_mixed_dataset else [idx.patch_count + idx.curve_count for idx in train_dataset.index_list]
        if is_mixed_dataset:
            batch_sampler = RatioBucketedDynamicBatchSampler(
                counts,
                train_dataset.grouped_indices,
                train_dataset.sampling_weights,
                batch_size=batch_size,
                max_total_items=max_total_items,
                bucket_boundaries=bucket_boundaries,
                shuffle=shuffle,
                drop_last=True,
                rank=rank,
                world_size=world_size,
            )
        else:
            batch_sampler = BucketedDynamicBatchSampler(
                counts,
                batch_size=batch_size,
                max_total_items=max_total_items,
                bucket_boundaries=bucket_boundaries,
                shuffle=shuffle,
                drop_last=True,
                rank=rank,
                world_size=world_size,
            )
        train_data = DataLoader(
            train_dataset, batch_sampler=batch_sampler,
            collate_fn=collate_function_global_topology,
            num_workers=effective_workers, pin_memory=False,
            persistent_workers=False
        )
        return train_data, batch_sampler

    if is_mixed_dataset or world_size > 1:
        if is_mixed_dataset:
            train_sampler = RatioDistributedSampler(
                train_dataset.grouped_indices,
                train_dataset.sampling_weights,
                shuffle=shuffle,
                drop_last=True,
                rank=rank,
                world_size=world_size,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True
            )
        train_data = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_function_global_topology,
            sampler=train_sampler, num_workers=effective_workers, pin_memory=False,
            drop_last=True, persistent_workers=True
        )
        return train_data, train_sampler

    train_data = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_function_global_topology,
        shuffle=shuffle, num_workers=effective_workers, pin_memory=False,
        drop_last=True, persistent_workers=True
    )
    return train_data, None