"""Compute TM-scores on validation set."""
import os, sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from config import cfg
from dataset import create_dataloaders
from model import build_model


def kabsch_rmsd(P, Q):
    """Optimal rotation + RMSD."""
    n = P.shape[0]
    if n < 3:
        return float(np.linalg.norm(P - Q) / np.sqrt(n + 1e-8))
    
    P_c = P - P.mean(axis=0, keepdims=True)
    Q_c = Q - Q.mean(axis=0, keepdims=True)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    P_rot = P_c @ R.T
    return float(np.sqrt(np.mean((P_rot - Q_c) ** 2)))


def tm_score(pred, true, d_0=None):
    """TM-score calculation."""
    n = len(pred)
    if d_0 is None:
        d_0 = 1.24 * (n - 15) ** 0.33
        d_0 = max(0.5, d_0)
    
    rmsd = kabsch_rmsd(pred, true)
    return float(np.mean(1.0 / (1.0 + (rmsd / d_0) ** 2)))


# Load model
_, val_loader = create_dataloaders(cfg)
model = build_model(cfg).to(cfg.device)
ckpt = torch.load('outputs/checkpoints/best_model.pt', map_location=cfg.device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

tm_scores = []
with torch.no_grad():
    for batch in val_loader:
        seq_tokens = batch['seq_tokens'].to(cfg.device)
        msa_tokens = batch['msa_tokens'].to(cfg.device)
        true_coords = batch['coords'].cpu().numpy()
        mask = batch.get('coord_mask', None)
        if mask is not None:
            mask = mask.cpu().numpy()
        
        if getattr(cfg, 'model_variant', 'full') == 'cpu_lite':
            residue_features = batch.get('residue_features', None)
            if residue_features is not None:
                residue_features = residue_features.to(cfg.device)
            pred_coords, _ = model(seq_tokens, msa_tokens, residue_features=residue_features)
        else:
            pred_coords, _ = model(seq_tokens, msa_tokens)
        
        pred_coords = pred_coords.cpu().numpy()
        
        for si in range(len(pred_coords)):
            pred = pred_coords[si]
            true = true_coords[si]
            m = mask[si] if mask is not None else np.ones(len(pred))
            valid = m > 0.5
            if valid.sum() >= 3:
                pred_v = pred[valid]
                true_v = true[valid]
                score = tm_score(pred_v, true_v)
                tm_scores.append(score)

print(f"Validation Set TM-Score:")
print(f"  Samples: {len(tm_scores)}")
print(f"  Mean: {np.mean(tm_scores):.4f}")
print(f"  Min: {np.min(tm_scores):.4f}")
print(f"  Max: {np.max(tm_scores):.4f}")
print(f"  Std: {np.std(tm_scores):.4f}")
