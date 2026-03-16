"""Evaluate validation TM-score for a trained checkpoint."""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "src")

from config import cfg
from dataset import create_dataloaders
from model import build_model


def kabsch_align(pred, true):
    """Return Kabsch-aligned coordinates (pred_aligned, true_centered)."""
    n = pred.shape[0]
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    true_c = true - true.mean(axis=0, keepdims=True)

    if n < 3:
        return pred_c, true_c

    h = pred_c.T @ true_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    pred_rot = pred_c @ r.T
    return pred_rot, true_c


def tm_score(pred, true):
    """TM-score from aligned coordinates."""
    n = len(pred)
    if n == 0:
        return 0.0
    n_eff = max(16, int(n))
    d0 = max(0.5, 1.24 * (n_eff - 15) ** 0.33)
    pred_aligned, true_centered = kabsch_align(pred, true)
    distances = np.linalg.norm(pred_aligned - true_centered, axis=1)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


@torch.no_grad()
def evaluate(checkpoint_path, max_samples=0):
    _, val_loader = create_dataloaders(cfg)

    model = build_model(cfg).to(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    per_target = []
    processed = 0

    for batch in tqdm(val_loader, desc="TM eval"):
        target_ids = batch["target_ids"]
        seq_tokens = batch["seq_tokens"].to(cfg.device)
        msa_tokens = batch["msa_tokens"].to(cfg.device)
        true_coords = batch["coords"].cpu().numpy()

        coord_mask = batch.get("coord_mask", None)
        if coord_mask is not None:
            coord_mask = coord_mask.cpu().numpy() > 0.5
        else:
            coord_mask = np.ones(true_coords.shape[:2], dtype=bool)

        residue_features = batch.get("residue_features", None)
        if residue_features is not None:
            residue_features = residue_features.to(cfg.device)

        if getattr(cfg, "model_variant", "full") == "cpu_lite":
            pred_coords, _ = model(seq_tokens, msa_tokens, residue_features=residue_features)
        else:
            pred_coords, _ = model(seq_tokens, msa_tokens)

        pred_coords = pred_coords.cpu().numpy()

        for i in range(pred_coords.shape[0]):
            valid = coord_mask[i]
            if valid.sum() < 3:
                continue

            score = tm_score(pred_coords[i][valid], true_coords[i][valid])
            per_target.append((target_ids[i], score, int(valid.sum())))
            processed += 1

            if max_samples > 0 and processed >= max_samples:
                break

        if max_samples > 0 and processed >= max_samples:
            break

    if not per_target:
        print("No valid validation samples found.")
        return

    scores = np.array([x[1] for x in per_target], dtype=float)

    print("Validation TM-score")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples: {len(scores)}")
    print(f"  Mean: {scores.mean():.6f}")
    print(f"  Median: {np.median(scores):.6f}")
    print(f"  Min: {scores.min():.6f}")
    print(f"  Max: {scores.max():.6f}")
    print(f"  Std: {scores.std():.6f}")

    top_k = sorted(per_target, key=lambda x: x[1], reverse=True)[:5]
    bot_k = sorted(per_target, key=lambda x: x[1])[:5]

    print("\nTop 5 targets")
    for tid, score, n_valid in top_k:
        print(f"  {tid}: tm={score:.6f} valid_res={n_valid}")

    print("\nBottom 5 targets")
    for tid, score, n_valid in bot_k:
        print(f"  {tid}: tm={score:.6f} valid_res={n_valid}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TM-score for a validation checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("outputs", "checkpoints", "best_model_tm.pt"),
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for quick debugging (0 = all validation samples)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    evaluate(args.checkpoint, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
