import numpy as np
import torch
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class HalfEdge:
    """半边数据 - 仅用于中间处理"""
    id: int
    curve_idx: int
    direction: bool
    patch_idx: int
    mate: int = -1
    next: int = -1
    prev: int = -1
    
    @property
    def is_boundary(self) -> bool:
        return self.patch_idx < 0


@dataclass
class Component:
    """一个loop或link作为基本单元"""
    __slots__ = ['id', 'patch_idx', 'curves', 'proposed_dirs', 'is_cycle']
    id: int
    patch_idx: int
    curves: List[int]
    proposed_dirs: List[bool]
    is_cycle: bool


# 预分配的numpy dtype用于批处理
HE_DTYPE = np.dtype([
    ('curve_idx', np.int32),
    ('direction', np.bool_),
    ('patch_idx', np.int32),
    ('is_boundary', np.bool_),
    ('mate', np.int32),
    ('next', np.int32),
    ('prev', np.int32),
])


class HalfEdgeBuilder:
    """半边结构构建器 - 内存优化版"""
    
    def __init__(self, endpoint_tol: float = 1e-4):
        self.tol = endpoint_tol
        self._precision = 6
    
    def build_batch(self, curves_batch: Dict, patches_batch: Dict) -> Dict:
        """构建整个batch的半边结构"""
        B = curves_batch['mask'].shape[0]
        
        # 预分配结果列表
        results = []
        for b in range(B):
            try:
                he_array = self._build_single_optimized(
                    curves_batch['endpoints'][b].cpu().numpy(),
                    curves_batch['is_closed'][b].cpu().numpy(),
                    curves_batch['mask'][b].cpu().numpy(),
                    patches_batch['mask'][b].cpu().numpy(),
                    patches_batch['curves_info'][b]
                )
                results.append(he_array)
            except Exception:
                # 错误时返回空数组
                results.append(np.zeros(0, dtype=HE_DTYPE))
        
        return self._collate_optimized(results)
    
    def _build_single_optimized(self, endpoints: np.ndarray, is_closed: np.ndarray,
                                 curves_mask: np.ndarray, patches_mask: np.ndarray,
                                 curves_info: List) -> np.ndarray:
        """构建单个样本 - 返回numpy结构化数组"""
        # Step 1-2: 建立映射
        c2p, p2c = self._build_mappings(curves_mask, patches_mask, curves_info)
        
        if not c2p:
            return np.zeros(0, dtype=HE_DTYPE)
        
        # 计算halfedge数量并预分配
        n_he = sum(2 for c in c2p if c2p[c])
        if n_he == 0:
            return np.zeros(0, dtype=HE_DTYPE)
        
        he_array = np.zeros(n_he, dtype=HE_DTYPE)
        he_array['mate'] = -1
        he_array['next'] = -1
        he_array['prev'] = -1
        
        # 创建halfedges
        c2he = {}
        idx = 0
        for c in sorted(c2p.keys()):
            if not curves_mask[c] or not c2p[c]:
                continue
            patches = c2p[c]
            if len(patches) > 2:
                raise ValueError(f"Curve {c} shared by {len(patches)} patches")
            
            he0_id, he1_id = idx, idx + 1
            he_array[he0_id] = (c, True, patches[0], patches[0] < 0, he1_id, -1, -1)
            he_array[he1_id] = (c, False, patches[1] if len(patches) > 1 else -1, 
                               len(patches) <= 1, he0_id, -1, -1)
            c2he[c] = [he0_id, he1_id]
            idx += 2
        
        if idx == 0:
            return np.zeros(0, dtype=HE_DTYPE)
        
        # 实际使用的数组
        he_array = he_array[:idx]
        
        # Step 3: 端点聚类
        ep2v = self._cluster_endpoints(endpoints, curves_mask, is_closed)
        
        # Step 4-6: 收集components
        components = self._collect_all_components_fast(p2c, is_closed, ep2v, he_array, c2he)
        
        # 确定方向
        self._determine_directions_fast(components, c2he, he_array)
        
        # Step 7: 应用方向和链接
        self._apply_directions_fast(components, he_array, c2he)
        
        # Step 8: 边界halfedge方向
        for i in range(len(he_array)):
            if he_array[i]['is_boundary'] and he_array[i]['mate'] >= 0:
                mate_idx = he_array[i]['mate']
                he_array[i]['direction'] = not he_array[mate_idx]['direction']
        
        return he_array
    
    def _build_single(self, endpoints: np.ndarray, is_closed: np.ndarray,
                      curves_mask: np.ndarray, patches_mask: np.ndarray,
                      curves_info: List) -> List[HalfEdge]:
        """兼容接口"""
        he_array = self._build_single_optimized(endpoints, is_closed, curves_mask, 
                                                 patches_mask, curves_info)
        return [HalfEdge(
            id=i,
            curve_idx=int(he_array[i]['curve_idx']),
            direction=bool(he_array[i]['direction']),
            patch_idx=int(he_array[i]['patch_idx']),
            mate=int(he_array[i]['mate']),
            next=int(he_array[i]['next']),
            prev=int(he_array[i]['prev'])
        ) for i in range(len(he_array))]
    
    def _build_mappings(self, curves_mask: np.ndarray, patches_mask: np.ndarray,
                        curves_info: List) -> Tuple[Dict, Dict]:
        """建立curve-patch双向映射"""
        c2p, p2c = {}, {}
        n_curves = len(curves_mask)
        
        for p in range(len(patches_mask)):
            if not patches_mask[p]:
                continue
            p2c[p] = []
            for c in self._flatten_curves_info(curves_info, p):
                if 0 <= c < n_curves and curves_mask[c]:
                    if c not in c2p:
                        c2p[c] = []
                    if p not in c2p[c]:
                        c2p[c].append(p)
                    if c not in p2c[p]:
                        p2c[p].append(c)
        return c2p, p2c
    
    def _flatten_curves_info(self, curves_info: List, patch_idx: int) -> List[int]:
        """展平curves_info"""
        if patch_idx >= len(curves_info) or not curves_info[patch_idx]:
            return []
        info = curves_info[patch_idx]
        if isinstance(info, (list, tuple)):
            if info and isinstance(info[0], (list, tuple)):
                return [c for loop in info for c in loop if isinstance(c, (int, np.integer))]
            return [c for c in info if isinstance(c, (int, np.integer))]
        return []
    
    def _cluster_endpoints(self, endpoints: np.ndarray, curves_mask: np.ndarray,
                           is_closed: np.ndarray) -> Dict[Tuple[int, int], int]:
        """精确匹配端点"""
        ep2v, point_to_vid = {}, {}
        current_v = 0
        
        for c in range(len(curves_mask)):
            if curves_mask[c] and not is_closed[c]:
                for ep_idx in [0, 1]:
                    pt = tuple(np.round(endpoints[c, ep_idx], self._precision))
                    if pt not in point_to_vid:
                        point_to_vid[pt] = current_v
                        current_v += 1
                    ep2v[(c, ep_idx)] = point_to_vid[pt]
        return ep2v
    
    def _collect_all_components_fast(self, p2c: Dict, is_closed: np.ndarray,
                                      ep2v: Dict, he_array: np.ndarray,
                                      c2he: Dict) -> List[Component]:
        """快速收集components"""
        components = []
        comp_id = 0
        
        for p in sorted(p2c.keys()):
            curves = p2c[p]
            closed_curves = [c for c in curves if is_closed[c]]
            open_curves = [c for c in curves if not is_closed[c]]
            
            # 闭合曲线
            for c in closed_curves:
                he_idx = self._get_he_idx_for_patch(c, p, he_array, c2he)
                if he_idx is not None:
                    components.append(Component(comp_id, p, [c], [True], True))
                    comp_id += 1
            
            # 开放曲线
            if open_curves:
                for comp in self._collect_open_components_fast(p, open_curves, ep2v):
                    comp.id = comp_id
                    comp_id += 1
                    components.append(comp)
        
        return components
    
    def _collect_open_components_fast(self, patch_idx: int, curves: List[int],
                                       ep2v: Dict) -> List[Component]:
        """快速收集开放曲线components"""
        adj = {}
        for c in curves:
            if (c, 0) not in ep2v or (c, 1) not in ep2v:
                continue
            v0, v1 = ep2v[(c, 0)], ep2v[(c, 1)]
            if v0 not in adj:
                adj[v0] = []
            if v1 not in adj:
                adj[v1] = []
            adj[v0].append((v1, c, True))
            adj[v1].append((v0, c, False))
        
        # 流形约束检查
        for v, neighbors in adj.items():
            if len(neighbors) > 2:
                raise ValueError(f"Vertex {v} degree {len(neighbors)} > 2")
        
        components = []
        visited = set()
        
        # 链（从度1顶点开始）
        for v in adj:
            if len(adj[v]) == 1:
                unvisited = [n for n in adj[v] if n[1] not in visited]
                if unvisited:
                    comp = self._traverse_fast(v, adj, visited, False, patch_idx)
                    if comp:
                        components.append(comp)
        
        # 环
        for v in adj:
            unvisited = [n for n in adj[v] if n[1] not in visited]
            if unvisited:
                comp = self._traverse_fast(v, adj, visited, True, patch_idx)
                if comp:
                    components.append(comp)
        
        return components
    
    def _traverse_fast(self, start_v: int, adj: Dict, visited: Set[int],
                       is_cycle: bool, patch_idx: int) -> Optional[Component]:
        """快速遍历"""
        edges = []
        current_v, prev_v = start_v, None
        
        while True:
            next_edge = None
            for nbr, c, d in adj.get(current_v, []):
                if c not in visited and (nbr != prev_v or (is_cycle and not edges)):
                    next_edge = (nbr, c, d)
                    break
            
            if not next_edge:
                break
            
            nbr_v, c, d = next_edge
            visited.add(c)
            edges.append((c, d))
            prev_v, current_v = current_v, nbr_v
            
            if is_cycle and current_v == start_v:
                break
        
        if not edges:
            return None
        return Component(-1, patch_idx, [e[0] for e in edges], [e[1] for e in edges], is_cycle)
    
    def _determine_directions_fast(self, components: List[Component],
                                    c2he: Dict, he_array: np.ndarray):
        """BFS确定方向"""
        if not components:
            return
        
        n = len(components)
        curve_to_comps = {}
        for i, comp in enumerate(components):
            for c in comp.curves:
                if c not in curve_to_comps:
                    curve_to_comps[c] = []
                curve_to_comps[c].append(i)
        
        # 邻接关系
        adj = {}
        for c, comps in curve_to_comps.items():
            if len(comps) == 2:
                if comps[0] not in adj:
                    adj[comps[0]] = []
                if comps[1] not in adj:
                    adj[comps[1]] = []
                adj[comps[0]].append((comps[1], c))
                adj[comps[1]].append((comps[0], c))
        
        # BFS
        flip = [None] * n
        for start in range(n):
            if flip[start] is not None:
                continue
            
            flip[start] = False
            queue = deque([start])
            
            while queue:
                curr = queue.popleft()
                curr_comp = components[curr]
                
                for neighbor, shared_curve in adj.get(curr, []):
                    curr_idx = curr_comp.curves.index(shared_curve)
                    curr_final = curr_comp.proposed_dirs[curr_idx] != flip[curr]
                    
                    neighbor_comp = components[neighbor]
                    neighbor_idx = neighbor_comp.curves.index(shared_curve)
                    required_flip = neighbor_comp.proposed_dirs[neighbor_idx] == curr_final
                    
                    if flip[neighbor] is None:
                        flip[neighbor] = required_flip
                        queue.append(neighbor)
                    elif flip[neighbor] != required_flip:
                        raise ValueError(f"Direction conflict: component {neighbor}")
        
        # 应用flip
        for i, comp in enumerate(components):
            if flip[i]:
                comp.proposed_dirs = [not d for d in comp.proposed_dirs]
    
    def _apply_directions_fast(self, components: List[Component],
                                he_array: np.ndarray, c2he: Dict):
        """应用方向和链接"""
        for comp in components:
            he_ids = []
            for curve, direction in zip(comp.curves, comp.proposed_dirs):
                he_idx = self._get_he_idx_for_patch(curve, comp.patch_idx, he_array, c2he)
                if he_idx is not None:
                    he_array[he_idx]['direction'] = direction
                    he_ids.append(he_idx)
            
            # 设置prev/next
            for i in range(1, len(he_ids)):
                he_array[he_ids[i-1]]['next'] = he_ids[i]
                he_array[he_ids[i]]['prev'] = he_ids[i-1]
            
            if comp.is_cycle and he_ids:
                if len(he_ids) == 1:
                    he_array[he_ids[0]]['next'] = he_ids[0]
                    he_array[he_ids[0]]['prev'] = he_ids[0]
                else:
                    he_array[he_ids[-1]]['next'] = he_ids[0]
                    he_array[he_ids[0]]['prev'] = he_ids[-1]
    
    def _get_he_idx_for_patch(self, curve: int, patch_idx: int,
                               he_array: np.ndarray, c2he: Dict) -> Optional[int]:
        """获取halfedge索引"""
        for he_id in c2he.get(curve, []):
            if he_id < len(he_array) and he_array[he_id]['patch_idx'] == patch_idx:
                return he_id
        return None
    
    def _get_he_for_patch(self, curve: int, patch_idx: int,
                          halfedges: List[HalfEdge], c2he: Dict) -> Optional[HalfEdge]:
        """兼容接口"""
        for he_id in c2he.get(curve, []):
            if halfedges[he_id].patch_idx == patch_idx:
                return halfedges[he_id]
        return None
    
    def _collate_optimized(self, results: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """高效整理batch结果"""
        B = len(results)
        max_he = max((len(r) for r in results), default=1)
        max_he = max(max_he, 1)
        
        # 预分配所有tensor
        tensors = {
            'curve_idx': torch.full((B, max_he), -1, dtype=torch.long),
            'direction': torch.zeros((B, max_he), dtype=torch.bool),
            'patch_idx': torch.full((B, max_he), -1, dtype=torch.long),
            'is_boundary': torch.zeros((B, max_he), dtype=torch.bool),
            'mate': torch.full((B, max_he), -1, dtype=torch.long),
            'next': torch.full((B, max_he), -1, dtype=torch.long),
            'prev': torch.full((B, max_he), -1, dtype=torch.long),
            'mask': torch.zeros((B, max_he), dtype=torch.bool),
        }
        
        # 批量填充
        for b, he_arr in enumerate(results):
            n = len(he_arr)
            if n > 0:
                tensors['curve_idx'][b, :n] = torch.from_numpy(he_arr['curve_idx'].astype(np.int64))
                tensors['direction'][b, :n] = torch.from_numpy(he_arr['direction'])
                tensors['patch_idx'][b, :n] = torch.from_numpy(he_arr['patch_idx'].astype(np.int64))
                tensors['is_boundary'][b, :n] = torch.from_numpy(he_arr['is_boundary'])
                tensors['mate'][b, :n] = torch.from_numpy(he_arr['mate'].astype(np.int64))
                tensors['next'][b, :n] = torch.from_numpy(he_arr['next'].astype(np.int64))
                tensors['prev'][b, :n] = torch.from_numpy(he_arr['prev'].astype(np.int64))
                tensors['mask'][b, :n] = True
        
        # 计算mate_prev和mate_next
        tensors['mate_prev'] = torch.full((B, max_he), -1, dtype=torch.long)
        tensors['mate_next'] = torch.full((B, max_he), -1, dtype=torch.long)
        
        # 向量化计算
        for b in range(B):
            mask = tensors['mask'][b]
            mate = tensors['mate'][b]
            valid_mate = (mate >= 0) & (mate < max_he) & mask
            
            if valid_mate.any():
                valid_indices = valid_mate.nonzero(as_tuple=True)[0]
                mate_indices = mate[valid_indices]
                tensors['mate_prev'][b, valid_indices] = tensors['prev'][b, mate_indices]
                tensors['mate_next'][b, valid_indices] = tensors['next'][b, mate_indices]
        
        return tensors
    
    def _collate(self, results: List[List[HalfEdge]]) -> Dict[str, torch.Tensor]:
        """兼容接口"""
        # 转换为numpy数组格式
        np_results = []
        for hes in results:
            if not hes:
                np_results.append(np.zeros(0, dtype=HE_DTYPE))
            else:
                arr = np.zeros(len(hes), dtype=HE_DTYPE)
                for i, he in enumerate(hes):
                    arr[i] = (he.curve_idx, he.direction, he.patch_idx, 
                              he.is_boundary, he.mate, he.next, he.prev)
                np_results.append(arr)
        
        return self._collate_optimized(np_results)