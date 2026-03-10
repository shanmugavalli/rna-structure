"""
Loss functions for RNA 3D structure prediction
Includes FAPE, geometric constraints, and coordinate losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fape_loss(pred_coords, true_coords, clamp_distance=10.0, eps=1e-8):
    """
    Frame Aligned Point Error (FAPE)
    Rotation-invariant loss that measures local structure accuracy
    
    Args:
        pred_coords: (batch, seq_len, 3) - predicted coordinates
        true_coords: (batch, seq_len, 3) - true coordinates
        clamp_distance: Maximum distance to prevent outlier gradients
    Returns:
        loss: scalar
    """
    # Validate inputs for NaN/Inf
    if torch.isnan(pred_coords).any() or torch.isinf(pred_coords).any():
        print("[WARN] FAPE: pred_coords contains NaN/Inf")
        return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)
    if torch.isnan(true_coords).any() or torch.isinf(true_coords).any():
        print("[WARN] FAPE: true_coords contains NaN/Inf")
        return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)
    
    batch, seq_len, _ = pred_coords.shape
    
    # For each residue as anchor, compute local frame errors
    # This is a simplified version - full FAPE uses rigid transformations
    
    # Compute all pairwise distances in predictions and targets
    pred_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # (batch, L, L, 3)
    true_diff = true_coords.unsqueeze(2) - true_coords.unsqueeze(1)
    
    # Distance differences (with numerical stability)
    pred_dist = torch.norm(pred_diff, dim=-1, p=2)  # (batch, L, L)
    true_dist = torch.norm(true_diff, dim=-1, p=2)
    
    # Clamp distances to prevent numerical issues
    pred_dist = torch.clamp(pred_dist, min=0.0, max=1000.0)
    true_dist = torch.clamp(true_dist, min=0.0, max=1000.0)
    
    # Clamped L1 distance
    dist_error = torch.abs(pred_dist - true_dist)
    dist_error = torch.clamp(dist_error, max=clamp_distance)
    
    # Average over all pairs
    loss = dist_error.mean()
    
    # Ensure loss is finite
    if not torch.isfinite(loss):
        print("[WARN] FAPE loss is non-finite, returning 0")
        return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)
    
    return loss


def coordinate_loss(pred_coords, true_coords):
    """
    Simple coordinate MSE/L1 loss
    
    Args:
        pred_coords: (batch, seq_len, 3)
        true_coords: (batch, seq_len, 3)
    Returns:
        loss: scalar
    """
    # Smooth L1 loss (Huber loss)
    loss = F.smooth_l1_loss(pred_coords, true_coords)
    return loss


def rmsd_loss(pred_coords, true_coords, eps=1e-8):
    """
    Root Mean Square Deviation loss
    
    Args:
        pred_coords: (batch, seq_len, 3)
        true_coords: (batch, seq_len, 3)
    Returns:
        loss: scalar
    """
    # Center coordinates
    pred_center = pred_coords.mean(dim=1, keepdim=True)
    true_center = true_coords.mean(dim=1, keepdim=True)
    
    pred_centered = pred_coords - pred_center
    true_centered = true_coords - true_center
    
    # Optimal rotation (Kabsch algorithm - simplified)
    # For training, we just use MSE on centered coordinates
    msd = ((pred_centered - true_centered) ** 2).sum(dim=-1).mean()
    rmsd = torch.sqrt(msd + eps)
    
    return rmsd


def bond_distance_loss(coords, target_dist=6.0, tolerance=1.0):
    """
    Regularize consecutive residue distances (C1'-C1' backbone)
    Typical RNA C1'-C1' distance is ~5-7 Angstroms
    
    Args:
        coords: (batch, seq_len, 3)
        target_dist: Target distance between consecutive residues
        tolerance: Acceptable deviation
    Returns:
        loss: scalar
    """
    if torch.isnan(coords).any() or torch.isinf(coords).any():
        print("[WARN] Bond loss: coords contains NaN/Inf")
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    
    # Consecutive residue distances
    diff = coords[:, 1:, :] - coords[:, :-1, :]  # (batch, seq_len-1, 3)
    dist = torch.norm(diff, dim=-1, p=2)  # (batch, seq_len-1)
    
    # Clamp to prevent numerical issues
    dist = torch.clamp(dist, min=0.0, max=1000.0)
    
    # Penalize deviations outside tolerance
    error = F.relu(torch.abs(dist - target_dist) - tolerance)
    loss = error.mean()
    
    if not torch.isfinite(loss):
        print("[WARN] Bond loss is non-finite, returning 0")
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    
    return loss


def clash_penalty(coords, min_dist=3.0):
    """
    Penalize atoms that are too close (clash)
    
    Args:
        coords: (batch, seq_len, 3)
        min_dist: Minimum allowed distance between non-consecutive residues
    Returns:
        loss: scalar
    """
    if torch.isnan(coords).any() or torch.isinf(coords).any():
        print("[WARN] Clash penalty: coords contains NaN/Inf")
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    
    batch, seq_len, _ = coords.shape
    
    # Pairwise distances (with numerical stability)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, L, L, 3)
    dist = torch.norm(diff, dim=-1, p=2)  # (batch, L, L)
    
    # Clamp to prevent numerical issues
    dist = torch.clamp(dist, min=0.0, max=1000.0)
    
    # Create mask for non-consecutive residues (ignore i, i+1, i-1)
    mask = torch.ones_like(dist, dtype=torch.bool)
    for i in range(seq_len):
        mask[:, i, max(0, i-1):min(seq_len, i+2)] = False
    
    # Penalize distances below threshold
    clash = F.relu(min_dist - dist)
    clash = clash * mask.float()
    
    mask_count = mask.sum().float()
    if mask_count < 1.0:
        print("[WARN] Clash: no valid position pairs, returning 0")
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    
    loss = clash.sum() / mask_count
    
    if not torch.isfinite(loss):
        print("[WARN] Clash loss is non-finite, returning 0")
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    
    return loss


class StructureLoss(nn.Module):
    """Combined loss for structure prediction"""
    
    def __init__(self, weights=None):
        super().__init__()
        
        if weights is None:
            weights = {
                'fape': 1.0,
                'coord': 0.3,
                'bond': 0.5,
                'clash': 0.3,
            }
        self.weights = weights
    
    def forward(self, pred_coords, true_coords, all_coords=None):
        """
        Args:
            pred_coords: (batch, seq_len, 3) - final prediction
            true_coords: (batch, seq_len, 3) - ground truth
            all_coords: List of intermediate predictions (for auxiliary losses)
        Returns:
            total_loss: scalar
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        
        # Main losses on final prediction
        losses['fape'] = fape_loss(pred_coords, true_coords)
        losses['coord'] = coordinate_loss(pred_coords, true_coords)
        losses['bond'] = bond_distance_loss(pred_coords)
        losses['clash'] = clash_penalty(pred_coords)
        
        # Optional: Auxiliary loss on intermediate predictions
        if all_coords is not None and len(all_coords) > 1:
            aux_loss = 0.0
            for coords in all_coords[:-1]:  # Exclude final (already counted)
                aux_loss += fape_loss(coords, true_coords) * 0.2
            losses['auxiliary'] = aux_loss / (len(all_coords) - 1)
            self.weights['auxiliary'] = 0.1
        
        # Compute weighted total
        total = sum(self.weights.get(k, 0.0) * v for k, v in losses.items())
        
        return total, losses
    
    def update_weights(self, new_weights):
        """Update loss weights (for curriculum learning)"""
        self.weights.update(new_weights)
