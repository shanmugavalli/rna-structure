"""
Inference script for test set predictions
Generates 5 diverse predictions per sequence
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from config import cfg
from model import build_model
from dataset import RNAStructureDataset, collate_fn
from torch.utils.data import DataLoader


def load_best_checkpoints(checkpoint_dir, top_k=3):
    """Load top-k checkpoints, preferring validation TM-score when available."""
    # Find all checkpoints
    checkpoint_files = glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    tm_ranked = []
    loss_ranked = []
    for path in checkpoint_files:
        # Prefer TM metadata stored inside checkpoint
        try:
            ckpt = torch.load(path, map_location='cpu')
            if isinstance(ckpt, dict) and ('val_tm' in ckpt):
                tm_ranked.append((float(ckpt['val_tm']), path))
                continue
        except Exception:
            pass

        # Fallback: parse validation loss from filename
        try:
            loss_str = path.split('_loss_')[-1].replace('.pt', '')
            loss = float(loss_str)
            loss_ranked.append((loss, path))
        except Exception:
            continue
    
    if tm_ranked:
        tm_ranked.sort(key=lambda x: x[0], reverse=True)
        top_checkpoints = [path for _, path in tm_ranked[:top_k]]
        print(f"Found {len(tm_ranked)} TM-scored checkpoints")
    else:
        loss_ranked.sort(key=lambda x: x[0])
        top_checkpoints = [path for _, path in loss_ranked[:top_k]]
        print(f"Found {len(loss_ranked)} loss-scored checkpoints")
    
    print(f"Using top {len(top_checkpoints)} checkpoints:")
    for i, path in enumerate(top_checkpoints, 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    return top_checkpoints


def load_models(checkpoint_paths, config):
    """Load multiple models from checkpoints"""
    models = []
    
    for path in checkpoint_paths:
        model = build_model(config).to(config.device)
        checkpoint = torch.load(path, map_location=config.device)
        
        # Load model state
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        models.append(model)
    
    return models


@torch.no_grad()
def predict_5_structures(models, seq_tokens, msa_tokens, config):
    """
    Generate 5 diverse predictions using ensemble strategy
    
    Strategy:
    - 3 predictions from different checkpoint models
    - 2 predictions from stochastic passes (dropout enabled)
    """
    predictions = []
    
    # Predictions from different models (deterministic)
    for model in models[:3]:
        model.eval()
        coords, _ = model(seq_tokens, msa_tokens)
        predictions.append(coords.cpu())
    
    # Stochastic predictions with dropout (if we have fewer than 3 models)
    if len(models) < 3:
        n_stochastic = 5 - len(predictions)
        model = models[0]
        model.train()  # Enable dropout
        
        for _ in range(n_stochastic):
            coords, _ = model(seq_tokens, msa_tokens)
            predictions.append(coords.cpu())
    else:
        # Use best model with dropout for remaining 2 predictions
        model = models[0]
        model.train()
        
        for _ in range(2):
            coords, _ = model(seq_tokens, msa_tokens)
            predictions.append(coords.cpu())
    
    # Ensure we have exactly 5 predictions
    while len(predictions) < 5:
        predictions.append(predictions[0])  # Duplicate if needed
    
    return predictions[:5]


def run_inference(config, test_csv_path, output_path='outputs/predictions/predictions.pt'):
    """Run inference on test set"""
    print("=" * 50)
    print("RNA Structure Prediction - Inference")
    print("=" * 50)
    
    # Load checkpoints
    checkpoint_paths = load_best_checkpoints(config.checkpoint_dir, top_k=3)
    models = load_models(checkpoint_paths, config)
    
    # Create test dataset
    test_dataset = RNAStructureDataset(
        seq_csv_path=test_csv_path,
        label_csv_path=None,  # No labels for test
        msa_dir=config.msa_dir,
        max_msa_seqs=config.max_msa_seqs,
        max_seq_len=config.max_seq_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for variable lengths
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print("Running predictions...")
    
    # Store all predictions
    all_predictions = {}
    
    for batch in tqdm(test_loader, desc="Inference"):
        target_id = batch['target_ids'][0]
        seq_tokens = batch['seq_tokens'].to(config.device)
        msa_tokens = batch['msa_tokens'].to(config.device)
        residue_features = batch.get('residue_features', None)
        if residue_features is not None:
            residue_features = residue_features.to(config.device)
        
        # Generate 5 predictions
        if getattr(config, 'model_variant', 'full') == 'cpu_lite':
            lite_predictions = []
            for model in models[:3]:
                model.eval()
                coords, _ = model(seq_tokens, msa_tokens, residue_features=residue_features)
                lite_predictions.append(coords.cpu())
            while len(lite_predictions) < 5:
                lite_predictions.append(lite_predictions[0])
            predictions = lite_predictions[:5]
        else:
            predictions = predict_5_structures(models, seq_tokens, msa_tokens, config)
        
        # Store predictions as (5, seq_len, 3) for submission formatter.
        pred_tensor = torch.stack(predictions).detach().cpu()
        if pred_tensor.ndim == 4 and pred_tensor.shape[1] == 1:
            pred_tensor = pred_tensor.squeeze(1)
        all_predictions[target_id] = pred_tensor.numpy()
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(all_predictions, output_path)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    return all_predictions


if __name__ == "__main__":
    # Update config for test data
    cfg.test_seq_path = 'data/raw/test_sequences.csv'
    
    predictions = run_inference(
        cfg,
        test_csv_path=cfg.test_seq_path,
        output_path='outputs/predictions/test_predictions.pt'
    )
    
    print(f"\nPredicted structures for {len(predictions)} targets")
    print("Ready for submission generation!")
