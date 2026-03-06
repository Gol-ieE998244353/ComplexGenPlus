import torch
import os
from pathlib import Path
import numpy as np

from train_pc_nm import CurveDecoder, PatchDecoder, Config
from vis.vis_util import gen_cylinder_quads, gen_cylinder_from_two_points
from vis.write_obj import write_obj_grouped

class LatentVisualizer:
    def __init__(self, vae_checkpoint, device):
        self.device = device
        self.config = Config()
        
        self.curve_decoder = CurveDecoder(self.config).to(self.device).eval()
        self.patch_decoder = PatchDecoder(self.config).to(self.device).eval()
        
        self._load_vae(vae_checkpoint)
        
    def _load_vae(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        clean = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
        
        if 'curve_decoder' in ckpt:
            self.curve_decoder.load_state_dict(clean(ckpt['curve_decoder']))
        if 'patch_decoder' in ckpt:
            self.patch_decoder.load_state_dict(clean(ckpt['patch_decoder']))
            
    @torch.no_grad()
    def decode_and_visualize(self, sample_out, save_path):
        """
        sample_out: dict with 'curve_latent' and 'patch_latent'
        save_path: Path or str
        """
        curve_latent = sample_out['curve_latent'].to(self.device)
        patch_latent = sample_out['patch_latent'].to(self.device)
        
        # Add batch dimension if needed
        if curve_latent.dim() == 2:
            curve_latent = curve_latent.unsqueeze(0)
        if patch_latent.dim() == 2:
            patch_latent = patch_latent.unsqueeze(0)
            
        curve_out = self.curve_decoder(curve_latent)
        patch_out = self.patch_decoder(patch_latent)
        
        # Get the first item in batch
        curve_points = curve_out['points'][0].cpu().numpy()
        curve_closed_logits = curve_out['closed_logits'][0].cpu().numpy()
        curve_validity = curve_out['validity_logits'][0].cpu().numpy()
        
        patch_points = patch_out['points'][0].cpu().numpy()
        patch_u_closed_logits = patch_out['u_closed_logits'][0].cpu().numpy()
        patch_validity = patch_out['validity_logits'][0].cpu().numpy()
        
        # Filter valid curves and patches (validity > 0)
        valid_curves = curve_validity > 0
        valid_patches = patch_validity > 0
        
        curve_points = curve_points[valid_curves]
        curve_closed_logits = curve_closed_logits[valid_curves]
        
        patch_points = patch_points[valid_patches]
        patch_u_closed_logits = patch_u_closed_logits[valid_patches]
        
        allverts_group = []
        allfaces_group = []
        allmtl_group = []
        all_group_name = []
        counter = 0
        
        # Process patches
        for gid in range(len(patch_points)):
            all_group_name.append(f'patch{gid}')
            allmtl_group.append(f'm{gid}')
            
            # patch_points[gid] is (400, 3), reshape to (20, 20, 3)
            pts = patch_points[gid].reshape(20, 20, 3)
            # Transpose to match gen_vis_result.py logic
            pts = np.transpose(pts, axes=(1, 0, 2)).reshape(-1, 3)
            allverts_group.append(pts)
            
            # u_closed_logits > 0 means closed
            is_closed = patch_u_closed_logits[gid] > 0
            faces = gen_cylinder_quads(20, 20, counter, flag_xclose=is_closed)
            allfaces_group.append(faces)
            counter += allverts_group[-1].shape[0]
            
        # Process curves
        for cid in range(len(curve_points)):
            c_verts = []
            c_faces = []
            pts = curve_points[cid] # (34, 3)
            
            for i in range(len(pts) - 1):
                cur_verts, cur_faces = gen_cylinder_from_two_points(pts[i], pts[i + 1], counter)
                if len(cur_faces) > 0:
                    c_verts.append(cur_verts)
                    c_faces += cur_faces
                    counter += cur_verts.shape[0]
                    
            # closed_logits > 0 means closed
            if curve_closed_logits[cid] > 0:
                cur_verts, cur_faces = gen_cylinder_from_two_points(pts[-1], pts[0], counter)
                if len(cur_faces) > 0:
                    c_verts.append(cur_verts)
                    c_faces += cur_faces
                    counter += cur_verts.shape[0]
                    
            if len(c_verts) > 0:
                all_group_name.append(f'curve{cid}')
                allmtl_group.append('cylinder')
                c_verts = np.concatenate(c_verts)
                allverts_group.append(c_verts)
                allfaces_group.append(c_faces)
                
        # Write to obj
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Copy mtl file if not exists
        mtl_src = os.path.join(os.path.dirname(__file__), 'complexgen.mtl')
        mtl_dst = os.path.join(save_dir, 'complexgen.mtl')
        if os.path.exists(mtl_src) and not os.path.exists(mtl_dst):
            import shutil
            shutil.copy(mtl_src, mtl_dst)
            
        write_obj_grouped(str(save_path), allverts_group, allfaces_group, allmtl_group, all_group_name)
