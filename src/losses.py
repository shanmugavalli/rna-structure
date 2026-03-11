"""
Loss functions for RNA 3D structure prediction
Includes FAPE, geometric constraints, and coordinate losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_coord_mask(true_coords, coord_mask=None, max_coord_abs=2000.0, eps=1e-8):
    """Build a robust residue-validity mask from provided mask + coordinate sanity checks."""
    finite_mask = torch.isfinite(true_coords).all(dim=-1).float()
    bounded_mask = (true_coords.abs().amax(dim=-1) <= float(max_coord_abs)).float()
    safe_mask = finite_mask * bounded_mask
    if coord_mask is not None:
        safe_mask = safe_mask * coord_mask.float()
    # Keep a minimum epsilon to avoid divide-by-zero in downstream reductions.
    return torch.clamp(safe_mask, min=0.0, max=1.0)


def _masked_center_and_scale(coords, coord_mask, eps=1e-8):
    """Center and scale coordinates using only valid residues for stable loss magnitudes."""
    mask3 = coord_mask.unsqueeze(-1)
    valid_count = coord_mask.sum(dim=1, keepdim=True).clamp_min(eps)
    center = (coords * mask3).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)
    centered = (coords - center) * mask3
    # RMS radius as per-sample scale, clamped to avoid tiny/huge normalization factors.
    sq = (centered ** 2).sum(dim=-1)
    rms = torch.sqrt((sq * coord_mask).sum(dim=1, keepdim=True) / valid_count + eps)
    scale = rms.clamp(min=1.0, max=100.0).unsqueeze(-1)
    normalized = centered / scale
    return normalized


def fape_loss(pred_coords, true_coords, coord_mask=None, clamp_distance=10.0, eps=1e-8):
    """
    Frame Aligned Point Error (FAPE)
    Rotation-invariant loss that measures local structure accuracy
    
    Args:
        pred_coords: (batch, seq_len, 3) - predicted coordinates
        true_coords: (batch, seq_len, 3) - true coordinates
        coord_mask: (batch, seq_len) - 1 for valid positions, 0 for invalid (optional)
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
    
    # Apply mask if provided (only compute loss on valid positions)
    if coord_mask is not None:
        # Create pairwise mask: both i and j must be valid
        pairwise_mask = coord_mask.unsqueeze(2) * coord_mask.unsqueeze(1)  # (batch, L, L)
        dist_error = dist_error * pairwise_mask
        # Average only over valid pairs
        valid_count = pairwise_mask.sum() + eps
        loss = dist_error.sum() / valid_count
    else:
        # Average over all pairs
        loss = dist_error.mean()
    
    # Ensure loss is finite
    if not torch.isfinite(loss):
        print("[WARN] FAPE loss is non-finite, returning 0")
        return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)
    
    return loss


def coordinate_loss(pred_coords, true_coords, coord_mask=None, max_coord_abs=2000.0):
    """
    Simple coordinate MSE/L1 loss
    
    Args:
        pred_coords: (batch, seq_len, 3)
        true_coords: (batch, seq_len, 3)
        coord_mask: (batch, seq_len) - 1 for valid positions, 0 for invalid (optional)
    Returns:
        loss: scalar
    """
    safe_mask = _safe_coord_mask(true_coords, coord_mask=coord_mask, max_coord_abs=max_coord_abs)

    if safe_mask.sum() < 1:
        return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)

    pred_norm = _masked_center_and_scale(pred_coords, safe_mask)
    true_norm = _masked_center_and_scale(true_coords, safe_mask)

    mask_expanded = safe_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    loss_per_coord = F.smooth_l1_loss(pred_norm, true_norm, reduction='none')
    # Denominator counts scalar coordinates (x,y,z), not only residues.
    denom = (mask_expanded.sum() * pred_coords.shape[-1]).clamp_min(1e-8)
    loss = (loss_per_coord * mask_expanded).sum() / denom
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
    
    def __init__(self, weights=None, max_coord_abs=2000.0):
        super().__init__()
        
        if weights is None:
            weights = {
                'fape': 1.0,
                'coord': 0.3,
                'bond': 0.5,
                'clash': 0.3,
            }
        self.weights = weights
        self.max_coord_abs = float(max_coord_abs)
    
    def forward(self, pred_coords, true_coords, all_coords=None, coord_mask=None):
        """
        Args:
            pred_coords: (batch, seq_len, 3) - final prediction
            true_coords: (batch, seq_len, 3) - ground truth
            all_coords: List of intermediate predictions (for auxiliary losses)
            coord_mask: (batch, seq_len) - 1 for valid positions, 0 for invalid (optional)
        Returns:
            total_loss: scalar
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        safe_mask = _safe_coord_mask(true_coords, coord_mask=coord_mask, max_coord_abs=self.max_coord_abs)
        
        # Main losses on final prediction
        losses['fape'] = fape_loss(pred_coords, true_coords, coord_mask=safe_mask)
        losses['coord'] = coordinate_loss(
            pred_coords,
            true_coords,
            coord_mask=safe_mask,
            max_coord_abs=self.max_coord_abs,
        )
        losses['bond'] = bond_distance_loss(pred_coords)
        losses['clash'] = clash_penalty(pred_coords)
        
        # Optional: Auxiliary loss on intermediate predictions
        if all_coords is not None and len(all_coords) > 1:
            aux_loss = 0.0
            for coords in all_coords[:-1]:  # Exclude final (already counted)
                aux_loss += fape_loss(coords, true_coords, coord_mask=safe_mask) * 0.2
            losses['auxiliary'] = aux_loss / (len(all_coords) - 1)
            self.weights['auxiliary'] = 0.1
        
        # Compute weighted total
        total = sum(self.weights.get(k, 0.0) * v for k, v in losses.items())
        
        return total, losses
    
    def update_weights(self, new_weights):
        """Update loss weights (for curriculum learning)"""
        self.weights.update(new_weights)
