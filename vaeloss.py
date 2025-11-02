from Minkowski_backbone import get_args_parser

import argparse
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # backward-compatible signature: accuracy(output, target, topk=(1,))
    # extended signature supported: accuracy(output, target, topk=(1,), dim=1, ignore_index=None, average=True)
    # output: logits/probabilities tensor; target: labels (scalar per sample) or one-hot
    def _inner(output, target, topk=(1,), dim=1, ignore_index=None, average=True):
        if target.numel() == 0:
            if average:
                return [100.0 * torch.ones([], device=output.device) for _ in topk]
            else:
                return [torch.zeros([], device=output.device, dtype=torch.long) for _ in topk]

        maxk = max(topk)

        # move target to output device
        target = target.to(output.device)

        # if target is one-hot (NxC), convert to class indices
        if target.ndim > 1 and target.size(-1) > 1:
            try:
                target = target.argmax(dim=-1)
            except Exception:
                target = target.view(-1)

        # apply ignore mask if requested
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            if valid_mask.numel() == 0 or valid_mask.sum() == 0:
                # no valid elements
                if average:
                    return [100.0 * torch.ones([], device=output.device) * 0.0 for _ in topk]
                else:
                    return [torch.zeros([], device=output.device, dtype=torch.long) for _ in topk]
            # filter out ignored samples for counting; we'll still compute preds for all but mask later
        else:
            valid_mask = None

        # get top-k predictions along class dim
        _, pred = output.topk(maxk, dim, True, True)

        # normalize pred to shape (maxk, N) where N is number of samples considered
        if pred.ndim == 2:
            # common case: (N, maxk) -> (maxk, N)
            pred = pred.t()
            tgt = target.view(1, -1)
            if valid_mask is not None:
                tgt = tgt[:, valid_mask]
                pred = pred[:, valid_mask]
            correct = pred.eq(tgt.expand_as(pred))
        else:
            # general case: move class dim to last if needed, then flatten leading dims
            if dim != (pred.ndim - 1):
                pred = pred.transpose(dim, -1)
            leading = pred.shape[:-1]
            N = 1
            for s in leading:
                N *= s
            pred_flat = pred.reshape(N, -1).transpose(0, 1)  # (maxk, N)
            tgt = target.view(-1)
            if valid_mask is not None:
                tgt = tgt[valid_mask]
                pred_flat = pred_flat[:, valid_mask]
            correct = pred_flat.eq(tgt.view(1, -1).expand_as(pred_flat))

        res = []
        batch_size = correct.shape[1]
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            if average:
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(correct_k)
        return res

    # if caller passed only three args (output, target, topk) keep compatibility
    try:
        # inspect whether caller used additional kwargs by checking function signature of caller
        return _inner(output, target, topk)
    except TypeError:
        # fallback to safe call
        return _inner(output, target, topk)

@torch.no_grad()
def cyclic_curve_points(closed_single_curve_points):
    new_curve_points = closed_single_curve_points[:,:]
    possible_curves = [new_curve_points.roll(shifts=i, dims=1) for i in range(new_curve_points.shape[1])]
    reverse_src_points = torch.flip(new_curve_points, dims=(1,))
    possible_curves += [reverse_src_points.roll(shifts=i, dims=1) for i in range(reverse_src_points.shape[1])]
    possible_curves = torch.cat(possible_curves, dim=0)
    return possible_curves
    
@torch.jit.script
def emd_by_id(gt: Tensor, pred: Tensor, gtid: Tensor, points_per_patch_dim: int):
  #gt shape: N/1, 400, 3
  #pred shape: N, 400, 3
  gt_batch = gt[:, gtid, :].view(len(gt), -1, points_per_patch_dim * points_per_patch_dim, 3)
  pred_batch = pred.view(len(pred), 1, points_per_patch_dim * points_per_patch_dim, 3)
  dist = (gt_batch - pred_batch).square().sum(-1).mean(-1).min(-1).values
  return dist

parser = argparse.ArgumentParser('', parents=[get_args_parser()])
args = parser.parse_args()

curve_eos_coef_cal = args.curve_avg_count / (args.num_curve_queries - args.curve_avg_count) * args.global_invalid_weight
patch_eos_coef_cal = args.patch_avg_count / (args.num_patch_queries - args.patch_avg_count) * args.global_invalid_weight

points_per_patch_dim = args.points_per_patch_dim
points_per_curve = 34

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class VAELoss_Curve_Criterion(nn.Module):
    def __init__(self, weight_dict, eos_coef, losses): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 4 #'Circle' 'BSpline' 'Line' 'Ellipse'
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_closed_curve(self, outputs, targets, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "is_closed" containing a tensor of dim [nb_target_curves]
        prediction: Closed_logits
        """
        assert 'Closed_logits' in outputs
        ## outputs和targets数组的传入参数格式需要确认
        src_logits = outputs['Closed_logits'] # [batch_size, num_queries]
        target_classes = targets['is_closed'] 
        loss_curve_closed = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        ##
        losses = {'loss_curve_closed': loss_curve_closed}
        if log:
            #all elements, only for current version
            losses['closed_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        
        return losses
    
    def loss_valid_labels(self, outputs, targets, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'Validity_logits' in outputs
        ##
        src_logits = outputs['Validity_logits']
        target_classes = targets['is_valid']
        ##
        
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses = {'loss_valid_ce': loss_ce}

        if log:
            losses['valid_class_accuracy'] = accuracy(src_logits, target_classes)[0]

        return losses
    
    def loss_type_labels(self, outputs, targets, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_curves]
        """
        ##
        assert 'Label_logits' in outputs
        src_logits = outputs['Label_logits']
        target_classes = targets['labels']
        target_idx = target_classes.argmax(dim=-1)

        logits_flat = src_logits.view(-1, 4)       # shape: [batch_size*num_curves, 4]
        target_flat = target_idx.view(-1)      # shape: [batch_size*num_curves]
        # idx = self._get_src_permutation_idx(indices)
        # target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # assert(len(src_logits[idx].shape) == 2 and src_logits[idx].shape[1] == 4)
        loss_ce = F.cross_entropy(logits_flat, target_flat)
        losses = {'loss_curve_type_ce': loss_ce}

        if log:
            losses['type_class_accuracy'] = accuracy(logits_flat, target_flat)[0]
        return losses

    def loss_geometry(self, outputs, targets, num_curves, cycleid = None):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'points' in outputs
        src_curve_points = outputs['points'].view(-1, points_per_curve, 3) # [batch_size * num_queries, 34, 3]
        target_curve_points = targets['curve_points'].view(-1, points_per_curve, 3) # [batch_size * num_queries, 34, 3]
        target_curve_length_weight = targets['curve_length_weighting'].view(-1) # [batch_size * num_queries]
        assert(target_curve_length_weight.shape[0] == target_curve_points.shape[0])
        is_target_curve_closed = targets['is_closed'].view(-1) # [batch_size * num_queries]
        assert(src_curve_points.shape == target_curve_points.shape)
        
        if(False):
          #compute chamfer distance
          pairwise_distance = torch.cdist(src_curve_points, target_curve_points, p=2.0) #in shape [batch_size, src_curve_number, tgt_curve_number)
          #print("pairwise_distance shape=", pairwise_distance.shape)
          s2t = pairwise_distance.min(-1).values.mean(-1)
          t2s = pairwise_distance.min(-2).values.mean(-1)        
          loss_geometry = (s2t + t2s) / 2.0
        else:
            if not args.geom_l2:
                distance_forward = (src_curve_points - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
                distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
                loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
                if not args.curve_open_loss:
                    for i in range(is_target_curve_closed.shape[0]):
                        if(is_target_curve_closed[i]):
                            tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                            loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).square().sum(-1).mean(-1).min()
            else:
                distance_forward = (src_curve_points - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
                distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
                loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
                for i in range(is_target_curve_closed.shape[0]):
                    if(is_target_curve_closed[i]):
                        tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                        loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).norm(dim = -1).mean(-1).min()

        assert(loss_geometry.shape == target_curve_length_weight.shape)
        loss_geometry *= target_curve_length_weight
        losses = {}
        losses['loss_geometry'] = loss_geometry.sum() / num_curves        
        return losses
    
    def get_loss(self, loss, outputs, targets, num_corners, **kwargs):
        loss_map = {
            'labels': self.loss_valid_labels,
            'curve_type': self.loss_type_labels,
            'geometry': self.loss_geometry,
            'closed_curve': self.loss_closed_curve,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'geometry':
            return loss_map[loss](outputs, targets, num_corners, **kwargs)
        else:
            return loss_map[loss](outputs, targets, num_corners)
        
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_targets = sum(len(t["labels"]) for t in targets)
        num_targets = torch.as_tensor([num_targets], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_targets)
        num_targets = torch.clamp(num_targets / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_targets))

        return losses

class VAELoss_Curve(nn.Module):
    """Combined VAE loss"""
    def __init__(self, 
                kl_weight=0.001):
        super().__init__()
        self.kl_weight = kl_weight
        self.device = device
        
        ## 将原代码build_unified_model_tripath中build criterion部分独立出来

        ## 设置weight dict和loss list
        curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}
        curve_losses = ['labels', 'geometry', 'closed_curve', 'curve_type']

        self.curve_loss_criterion = VAELoss_Curve_Criterion(curve_weight_dict, curve_eos_coef_cal, curve_losses).to(device)

    def kl_divergence(self, mu, logvar):
        """KL divergence loss"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def forward(self, outputs, targets, mu, logvar):
        """
        Args:
            outputs: dict from decoder
            targets: dict of ground truth
            mu, logvar: latent parameters
        """
        losses = {}
        
        # KL divergence
        losses['kl'] = self.kl_divergence(mu, logvar)

        # Curve losses
        curve_loss_dict = self.curve_loss_criterion(outputs['curves'], targets['curves'])
        curve_weight_dict = self.curve_loss_criterion.weight_dict

        # Total loss
        total_loss = (
            self.kl_weight * losses['kl'] +
            sum(curve_loss_dict[k] * curve_weight_dict[k] for k in curve_loss_dict.keys() if k in curve_weight_dict)
        )
        
        losses['total'] = total_loss
        return losses

class VAELoss_Patch_Criterion(nn.Module):
    def __init__(self, weight_dict, eos_coef, losses): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 6 # Cylinder, Torus, BSpline, Plane, Cone, Sphere
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if args.patch_emd:
          self.emd_idlist = []
          base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base, i, [0,1]).flatten())
          
          base_t = base.transpose(0,1)
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base_t, i, [0,1]).flatten())

          self.register_buffer('emd_idlist', torch.cat(self.emd_idlist))
        if args.patch_uv:
          self.emd_idlist_u = []
          self.emd_idlist_v = []
          base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
          #set idlist u
          for i in range(points_per_patch_dim):
            cur_base = base.roll(shifts=i, dims = 0)
            for i in range(0,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
            cur_base = cur_base.transpose(0,1)
            for i in range(1,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
          self.emd_idlist_u = torch.cat(self.emd_idlist_u)

    def loss_closed_patch(self, outputs, targets, num_patches,log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "is_closed" containing a tensor of dim [nb_target_curves]
        prediction: closed_curve_logits
        """
        assert 'u_closed' in outputs or 'v_closed' in outputs

        src_logits = outputs['u_closed'] if 'u_closed' in outputs else outputs['v_closed']  # [batch_size, num_queries]
        target_classes = targets['u_closed'] if 'u_closed' in outputs else targets['v_closed']
        
        loss_curve_closed = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses = {'loss_patch_closed': loss_curve_closed}

        if log:
            losses['closed_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        
        return losses
    
    def loss_valid_labels(self, outputs, targets, num_patches, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'Validity_logits' in outputs
        src_logits = outputs['Validity_logits']
        target_classes = targets['is_valid']
        
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses = {'loss_valid_ce': loss_ce}

        if log:
            losses['valid_class_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        return losses
    
    def loss_type_labels(self, outputs, targets, num_patches, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_curves]
        """
        assert 'Label_logits' in outputs
        src_logits = outputs['Label_logits']
        target_classes = targets['labels']
        target_idx = target_classes.argmax(dim=-1)

        logits_flat = src_logits.view(-1, 6)       # shape: [batch_size*num_curves, 6]
        target_flat = target_idx.view(-1)      # shape: [batch_size*num_curves]
        # idx = self._get_src_permutation_idx(indices)
        # target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # assert(len(src_logits[idx].shape) == 2 and src_logits[idx].shape[1] == 4)
        loss_ce = F.cross_entropy(logits_flat, target_flat)
        losses = {'loss_curve_type_ce': loss_ce}

        if log:
            losses['type_class_accuracy'] = accuracy(logits_flat, target_flat)[0]
        return losses
    
    def loss_geometry(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'points' in outputs
        src_patch_points = outputs['points'].view(-1, points_per_patch_dim * points_per_patch_dim, 3)  # [batch_size * num_queries, 100, 3]
        if args.output_normal:
          src_patch_normals = outputs['pred_patch_normals'].view(-1, points_per_patch_dim * points_per_patch_dim, 3)
        target_patch_points_list = targets['patch_points'].view(-1, points_per_patch_dim * points_per_patch_dim, 3)
        target_patch_area_weighting = targets['patch_area_weighting'].view(-1)  # [batch_size * num_queries]
        assert(target_patch_area_weighting.shape[0] == target_patch_points_list.shape[0])

        if args.patch_uv:
          target_patch_uclosed = targets['u_closed'].view(-1)
          target_patch_vclosed = targets['v_closed'].view(-1)

        assert(target_patch_area_weighting.shape[0] == src_patch_points.shape[0])
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]

        if args.patch_normal:
          target_patch_normals = targets['patch_normals'].view(-1, points_per_patch_dim * points_per_patch_dim, 3)
        
        if args.patch_emd:
          target_patch_points_batch = torch.cat(target_patch_points).view(len(target_patch_points), -1, 3)  
          loss_geom = emd_by_id(target_patch_points_batch, src_patch_points, self.emd_idlist, points_per_patch_dim)
          
          if args.patch_uv:
            uclose_id = torch.where(target_patch_uclosed == 1)[0]
            if len(uclose_id) > 0:
              loss_geom[uclose_id] = emd_by_id(target_patch_points_batch[uclose_id], src_patch_points[uclose_id], self.emd_idlist_u, points_per_patch_dim)
              

          losses = {}
          losses['loss_geometry'] = loss_geom.mean()
          return losses

        ## 不会用到
        assert(len(src_patch_points) == len(target_patch_points))
        assert(target_patch_area_weighting.shape[0] == len(target_patch_points))
        #compute chamfer distance
        loss_geometry = []
        loss_patch_normal = []
        loss_patch_lap = []
        loss_output_normal_diff = []
        loss_output_normal_tangent = []
        for patch_idx in range(len(target_patch_points)):
          if not args.geom_l2:
            patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx], p=2.0).square() #in shape [src_patch_points, tgt_patch_points]
          else:
            patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx], p=2.0) #in shape [src_patch_points, tgt_patch_points]
          assert(len(patch_distance.shape) == 2)
          if(args.single_dir_patch_chamfer):
            loss_geometry.append(target_patch_area_weighting[patch_idx]*patch_distance.min(0).values.mean())
          else:
            loss_geometry.append(target_patch_area_weighting[patch_idx]*(patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
          if args.patch_normal:
            if args.single_dir_patch_chamfer:
              closest_id = torch.argmin(patch_distance, dim = 0)
              tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
              tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
              tangent_x = F.normalize(tangent_x, dim = -1)
              tangent_y = F.normalize(tangent_y, dim = -1)
              loss_patch_normal.append((tangent_x[closest_id] * target_patch_normals[patch_idx]).sum(-1).abs().mean())
              loss_patch_normal.append((tangent_y[closest_id] * target_patch_normals[patch_idx]).sum(-1).abs().mean())
            else:
              closest_id = torch.argmin(patch_distance, dim = 1)
              tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
              tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
              tangent_x = F.normalize(tangent_x, dim = -1)
              tangent_y = F.normalize(tangent_y, dim = -1)
              loss_patch_normal.append((tangent_x * target_patch_normals[patch_idx][closest_id]).sum(-1).abs().mean())
              loss_patch_normal.append((tangent_y * target_patch_normals[patch_idx][closest_id]).sum(-1).abs().mean())
          if args.output_normal:
            closest_id = torch.argmin(patch_distance, dim = 1)
            tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
            tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
            #normalize
            tangent_x = F.normalize(tangent_x, dim = -1)
            tangent_y = F.normalize(tangent_y, dim = -1)
            loss_output_normal_diff.append(torch.norm(target_patch_normals[patch_idx][closest_id] - src_patch_normals[patch_idx], dim = -1).mean())
            loss_output_normal_tangent.append((tangent_x * src_patch_normals[patch_idx]).sum(-1).abs().mean())
            loss_output_normal_tangent.append((tangent_y * src_patch_normals[patch_idx]).sum(-1).abs().mean())
          
          if args.patch_lap:
            x_minus = src_patch_points[patch_idx][outputs['mask_x_minus']]
            x_plus = src_patch_points[patch_idx][outputs['mask_x_plus']]
            y_minus = src_patch_points[patch_idx][outputs['mask_y_minus']]
            y_plus = src_patch_points[patch_idx][outputs['mask_y_plus']]
            loss_patch_lap.append((src_patch_points[patch_idx] - (x_minus + x_plus + y_minus + y_plus) / 4.0).norm(dim = -1).mean())
          if args.patch_lapboundary:
            loss_patch_lap.append(torch.mm(outputs['mat_lapboundary'], src_patch_points[patch_idx]).norm(dim = -1).mean())
            
        losses = {}
        losses['loss_geometry'] = sum(loss_geometry) / num_patches
        if args.patch_normal:
          losses['loss_patch_normal'] = sum(loss_patch_normal) / num_patches   
        if args.patch_lap or args.patch_lapboundary:
          losses['loss_patch_lap'] = sum(loss_patch_lap) / num_patches
        if args.output_normal:
          losses['output_normal_diff'] = sum(loss_output_normal_diff) / num_patches
          losses['output_normal_tangent'] = sum(loss_output_normal_tangent) / num_patches   
        return losses
    
    def get_loss(self, loss, outputs, targets, num_patches, **kwargs):
        loss_map = {
            'labels': self.loss_valid_labels,
            'patch_type': self.loss_type_labels,
            'geometry': self.loss_geometry,
            'closed_patch': self.loss_closed_patch
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_patches, **kwargs)
    
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        #t0 = time.time()
        if len(outputs) == 0:
          return {}, []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_targets = sum(len(t["labels"]) for t in targets)
        num_targets = torch.as_tensor([num_targets], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_targets)
        num_targets = torch.clamp(num_targets / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_targets))

        return losses

class VAELoss_Patch(nn.Module):
    """Combined VAE loss"""
    def __init__(self, 
                 kl_weight=0.001):
        super().__init__()
        self.kl_weight = kl_weight
        self.device = device
        
        ## 将原代码build_unified_model_tripath中build criterion部分独立出来
        ## 设置weight dict和loss list
        patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}
        ## if args.patch_close: 默认开启
        patch_weight_dict['loss_patch_closed'] = 1 
        if args.patch_normal:
            patch_weight_dict['loss_patch_normal'] = args.patch_normal_loss_coef
        ## if args.output_normal: 默认开启
        patch_weight_dict['output_normal_diff'] = args.output_normal_diff_coef
        patch_weight_dict['output_normal_tangent'] = args.output_normal_tangent_coef
        if args.extra_single_chamfer:
            patch_weight_dict['loss_single_cd'] = args.extra_single_chamfer_weight
        if args.patch_lap or args.patch_lapboundary:
            patch_weight_dict['loss_patch_lap'] = args.patch_lap_loss_coef

        patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
        if args.patch_close:
            patch_losses.append('closed_patch')

        self.patch_loss_criterion = VAELoss_Patch_Criterion(patch_weight_dict, patch_eos_coef_cal, patch_losses).to(device)

    def kl_divergence(self, mu, logvar):
        """KL divergence loss"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def forward(self, outputs, targets, mu, logvar):
        """
        Args:
            outputs: dict from decoder
            targets: dict of ground truth
            mu, logvar: latent parameters
        """
        losses = {}
        
        # KL divergence
        losses['kl'] = self.kl_divergence(mu, logvar)

        # Curve and Patch losses
        ## 不太确定这里传入的参数是否正确。原代码中传入的是每个检测模块对应的outputs和targets
        ## 所以说理论上应该传入curve的dict和patch的dict，但我不太确定这里是否传对了
        patch_loss_dict, patch_matching_indices = self.patch_loss_criterion(outputs, targets)
        patch_weight_dict = self.patch_loss_criterion.weight_dict

        # Total loss
        total_loss = (
            self.kl_weight * losses['kl'] +
            sum(patch_loss_dict[k] * patch_weight_dict[k] for k in patch_loss_dict.keys() if k in patch_weight_dict)
        )
        
        losses['total'] = total_loss
        return losses