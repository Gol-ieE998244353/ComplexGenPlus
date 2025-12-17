import numpy as np
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
    return {'Circle': 0, 'BSpline': 1, 'Line': 2}.get(s, 3)

def _patch_type_to_id(s):
    """面片类型转ID"""
    mapping = {'Cylinder': 0, 'Torus': 1, 'BSpline': 2, 'Extrusion': 2, 
               'Revolution': 2, 'Plane': 3, 'Cone': 4, 'Sphere': 5}
    return mapping.get(s, 5)


class LazyDataIndex:
    """惰性数据索引 - 只存储文件位置信息，不加载实际数据"""
    __slots__ = ['packed_file', 'offset', 'length']
    
    def __init__(self, packed_file, offset, length):
        self.packed_file = packed_file
        self.offset = offset
        self.length = length


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
                    index_list.append(LazyDataIndex(filepath, offset, length))
                    
                    for curve in sample.get('curves', []):
                        if curve.get('curve_length', 0) >= 1e-3:
                            curve_length_stat.append(curve['curve_length'])
                    for patch in sample.get('patches', []):
                        patch_area_stat.append(patch.get('patch_area', 0))
        else:
            index_list.append(LazyDataIndex(filepath, 0, -1))
    
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
    if 'surface_points' not in sample:
        return False
    pts = sample['surface_points']
    if pts[:, :3].min() < -0.55 or pts[:, :3].max() > 0.55:
        return False
    if len(sample.get('curves', [])) == 0:
        return False
    for curve in sample.get('curves', []):
        if curve.get('curve_length', 0) < 1e-3:
            return False
    return True


def load_and_process_sample(index_item):
    """按需加载并处理单个样本"""
    with open(index_item.packed_file, "rb") as rf:
        if index_item.offset > 0:
            rf.seek(index_item.offset)
        sample = pickle.load(rf)
    
    processed = {
        'surface_points': sample['surface_points'].astype(np.float32),
        'curves': [],
        'patches': []
    }
    
    scale = 1.0
    translation = np.zeros(3, dtype=np.float32)
    
    for curve in sample.get('curves', []):
        pts = curve['points']
        if pts.min() < -0.55 or pts.max() > 0.55:
            continue
        
        processed['curves'].append({
            'points': (scale * (pts + translation)).astype(np.float32),
            'is_closed': curve['is_closed'],
            'endpoints': [-1, -1] if curve['is_closed'] else [curve['start_vert_idx'], curve['end_vert_idx']],
            'type': _curve_type_to_id(curve['type']),
            'curve_length': curve['curve_length'] * scale
        })
    
    for patch in sample.get('patches', []):
        processed['patches'].append({
            'type': _patch_type_to_id(patch['type']),
            'patch_points': patch['patch_points'],
            'curves': patch['curves'],
            'grid_normal': patch.get('grid_normal'),
            'u_closed': patch.get('u_closed', False),
            'v_closed': patch.get('v_closed', False),
            'patch_area': patch.get('patch_area', 0) * scale * scale
        })
    
    return processed


# ============ 新增：点云体素化函数（从文档二移植） ============
def points2sparse_voxel(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    """将点云转换为稀疏体素表示"""
    points = points_with_normal[:,:3] + 0.5  # to [0, 1]
    voxel_dict = {}
    voxel_length = 1.0 / voxel_dim
    voxel_coord = np.clip(np.floor(points / voxel_length).astype(np.int32), 0, voxel_dim-1)
    points_normal_norm = linalg.norm(points_with_normal[:,3:], axis=1, keepdims=True)
    points_normal_norm[points_normal_norm < th_norm] = th_norm
    
    if feature_type == 'local':
        local_coord = (points - voxel_coord.astype(np.float32)*voxel_length)*voxel_dim - 0.5
        local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, 
                                     np.ones([local_coord.shape[0], 1])], axis=-1)
    elif feature_type == 'global':
        local_coord = points - 0.5
        local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, 
                                     np.ones([local_coord.shape[0], 1])], axis=-1)
    
    for i in range(voxel_coord.shape[0]):
        coord_tuple = (voxel_coord[i,0], voxel_coord[i,1], voxel_coord[i,2])
        if coord_tuple not in voxel_dict:
            voxel_dict[coord_tuple] = local_coord[i]
        else:
            voxel_dict[coord_tuple] += local_coord[i]
    
    locations = np.array(list(voxel_dict.keys()))
    features = np.array(list(voxel_dict.values()))
    points_in_voxel = features[:,6:]
    features = features / points_in_voxel
    position = features[:,:3]
    normals = features[:,3:6]
    pad_ones = features[:,6:]
    normals /= linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    
    if with_normal and pad1s:
        features = np.concatenate([position, normals, pad_ones], axis=1)
    elif pad1s:
        features = np.concatenate([position, pad_ones], axis=1)
    elif with_normal:
        features = np.concatenate([position, normals], axis=1)
    else:
        features = position
    return locations.astype(np.int32), features.astype(np.float32)


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
                 flag_noise=0, flag_grid=False, num_angles=4, dim_grid=10,
                 with_pointcloud=False, voxel_dim=32, feature_type='global', 
                 with_normal=True, pad1s=True):  # 新增：点云相关参数
        self.index_list, self.data_folder = build_lazy_index(data_folder)
        self.random_rotation_augmentation = random_rotation
        self.random_angle = random_angle
        self.flag_noise = flag_noise
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
        points = curve['points']
        
        if rot is not None:
            points = np.dot(points, rot)
        
        np.clip(points, -1000, 1000, out=points)
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        scale = float((max_vals - min_vals).max())
        scale = max(SCALE_MIN, min(SCALE_MAX, scale))
        center = (min_vals + max_vals) * 0.5
        
        normalized = (points - center) / scale
        np.clip(normalized, -0.6, 0.6, out=normalized)
        
        endpoints = np.zeros((2, 3), dtype=np.float32)
        if not curve['is_closed']:
            ep = curve['endpoints']
            idx0 = max(0, min(33, int(ep[0]) if isinstance(ep, (list, tuple, np.ndarray)) else int(ep)))
            idx1 = max(0, min(33, int(ep[1]) if isinstance(ep, (list, tuple, np.ndarray)) else int(ep)))
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
        if not self.flag_grid:
            patch_data = item_points[patch['patch_points']]
            patch_points = patch_data[:, :3]
            patch_normals = patch_data[:, 3:]
        else:
            grid_data = patch.get('grid_normal')
            if grid_data is None:
                return None
            
            if len(grid_data) == self.dim_grid * self.dim_grid:
                patch_points = grid_data[:, :3]
                patch_normals = grid_data[:, 3:]
            else:
                tmp = grid_data.reshape(20, 20, -1)[::2, ::2].reshape(-1, 6)
                patch_points = tmp[:, :3]
                patch_normals = tmp[:, 3:]
        
        if rot is not None:
            patch_points = np.dot(patch_points, rot)
            patch_normals = np.dot(patch_normals, rot)
        
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
            sigma = {1: 0.01, 2: 0.02, 3: 0.05}[self.flag_noise]
            clip = 5.0 * sigma
            noise = np.clip(sigma * np.random.randn(item_points.shape[0], 3), -clip, clip).astype(np.float32)
            item_points[:, :3] += noise
            
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
            pp = self._process_patch(patch, item_points, None)
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


def data_loader_ABC(data_folder):
    """兼容接口 - 返回惰性索引列表"""
    index_list, _ = build_lazy_index(data_folder)
    return index_list


def train_data_loader_clean(batch_size=32, data_folder="data/default/train", 
                            rotation_augmentation=False, random_angle=False, 
                            flag_noise=0, flag_grid=False, num_angle=4, dim_grid=10,
                            num_workers=8, rank=0, world_size=1,
                            with_pointcloud=False, voxel_dim=128, feature_type='global',
                            with_normal=True, pad1s=True):  # 新增：点云相关参数
    """
    带全局拓扑处理的数据加载器 - 内存优化版 + 点云支持
    """
    if not os.path.exists(os.path.join(data_folder, "packed")):
        if rank == 0:
            os.makedirs(os.path.join(data_folder, "packed"), exist_ok=True)
            pack_pickle_files(data_folder, os.path.join(data_folder, "packed"))
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
    
    train_dataset = ABCDatasetOptimized(
        data_folder,
        random_rotation=rotation_augmentation,
        random_angle=random_angle,
        flag_noise=flag_noise,
        flag_grid=flag_grid,
        num_angles=num_angle,
        dim_grid=dim_grid,
        with_pointcloud=with_pointcloud,  # 新增
        voxel_dim=voxel_dim,  # 新增
        feature_type=feature_type,  # 新增
        with_normal=with_normal,  # 新增
        pad1s=pad1s  # 新增
    )
    
    effective_workers = min(num_workers, 2)
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        train_data = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_function_global_topology,
            sampler=train_sampler, num_workers=effective_workers, pin_memory=False,
            drop_last=True, persistent_workers=False
        )
        return train_data, train_sampler
    else:
        train_data = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_function_global_topology,
            shuffle=True, num_workers=effective_workers, pin_memory=False,
            drop_last=True, persistent_workers=False
        )
        return train_data, None