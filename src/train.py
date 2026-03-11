"""
Training script for RNA structure prediction
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
import torch
import torch.nn as nn
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from config import cfg
from model import build_model, EMAModel
from dataset import create_dataloaders
from losses import StructureLoss
from utils import save_checkpoint, load_checkpoint, AverageMeter


# Augmentation is now in dataset.py to be applied before batch collation


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config, ema=None):
    """Train for one epoch"""
    model.train()
    
    # Update loss weights based on curriculum
    loss_weights = cfg.get_loss_weights(epoch)
    criterion.update_weights(loss_weights)
    
    losses = AverageMeter()
    fape_losses = AverageMeter()
    coord_losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    # Note: GradScaler removed - AMP disabled in config
    accum_steps = max(1, config.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(pbar):
        # Skip if batch is None (all samples were corrupted)
        if batch is None:
            print(f"[WARN] Batch {step}: all samples corrupted, skipping")
            continue
        
        # Move to device
        seq_tokens = batch['seq_tokens'].to(config.device)
        msa_tokens = batch['msa_tokens'].to(config.device)
        true_coords = batch['coords'].to(config.device)
        residue_features = batch.get('residue_features', None)
        if residue_features is not None:
            residue_features = residue_features.to(config.device)
        coord_mask = batch.get('coord_mask', None)
        if coord_mask is not None:
            coord_mask = coord_mask.to(config.device)
        
        # Validate inputs for NaN/Inf
        if torch.isnan(seq_tokens).any() or torch.isnan(msa_tokens).any() or torch.isnan(true_coords).any():
            print(f"[WARN] Batch {step}: input contains NaN, skipping")
            continue
        if torch.isinf(seq_tokens).any() or torch.isinf(msa_tokens).any() or torch.isinf(true_coords).any():
            print(f"[WARN] Batch {step}: input contains Inf, skipping")
            continue
        
        # Augmentation already applied in dataset pipeline
        # Additional safeguard: verify augmented coords are still valid
        if torch.isnan(true_coords).any() or torch.isinf(true_coords).any():
            print(f"[WARN] Batch {step}: augmented coords contain NaN/Inf, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Forward pass (AMP disabled - using full float32)
        if getattr(config, 'model_variant', 'full') == 'cpu_lite':
            pred_coords, all_coords = model(seq_tokens, msa_tokens, residue_features=residue_features)
        else:
            pred_coords, all_coords = model(seq_tokens, msa_tokens)
        
        # Validate model outputs
        if torch.isnan(pred_coords).any() or torch.isinf(pred_coords).any():
            print(f"[WARN] Batch {step}: model output contains NaN/Inf, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        loss, loss_dict = criterion(pred_coords, true_coords, all_coords, coord_mask=coord_mask)
        
        # Check loss is finite before backward
        if not torch.isfinite(loss):
            print(f"[WARN] Batch {step}: loss is non-finite ({loss.item()}), skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient accumulation (float32, no scaler)
        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            # Clip gradients before optimizer step
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"[WARN] NaN/Inf gradients detected, skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
            else:
                # Optimizer step (no scaler.step needed - standard PyTorch)
                optimizer.step()
                
                # Scheduler step (AFTER optimizer step - correct order)
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update meters
        losses.update(loss.item())
        fape_losses.update(loss_dict['fape'].item())
        coord_losses.update(loss_dict['coord'].item())
        
        # Free memory explicitly
        del loss, loss_dict, pred_coords, all_coords
        if (step + 1) % accum_steps == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        if step % config.log_every_n_steps == 0:
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'fape': f'{fape_losses.avg:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    return {
        'train_loss': losses.avg,
        'train_fape': fape_losses.avg,
        'train_coord': coord_losses.avg
    }


@torch.no_grad()
def validate(model, val_loader, criterion, config):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    fape_losses = AverageMeter()
    coord_losses = AverageMeter()
    
    for batch in tqdm(val_loader, desc="Validation"):
        seq_tokens = batch['seq_tokens'].to(config.device)
        msa_tokens = batch['msa_tokens'].to(config.device)
        true_coords = batch['coords'].to(config.device)
        residue_features = batch.get('residue_features', None)
        if residue_features is not None:
            residue_features = residue_features.to(config.device)
        coord_mask = batch.get('coord_mask', None)
        if coord_mask is not None:
            coord_mask = coord_mask.to(config.device)
        
        # Forward pass
        if getattr(config, 'model_variant', 'full') == 'cpu_lite':
            pred_coords, all_coords = model(seq_tokens, msa_tokens, residue_features=residue_features)
        else:
            pred_coords, all_coords = model(seq_tokens, msa_tokens)
        
        # Compute loss
        loss, loss_dict = criterion(pred_coords, true_coords, all_coords, coord_mask=coord_mask)
        
        # Update meters
        losses.update(loss.item())
        fape_losses.update(loss_dict['fape'].item())
        coord_losses.update(loss_dict['coord'].item())
        
        # Free memory
        del loss, loss_dict, pred_coords, all_coords, seq_tokens, msa_tokens, true_coords
    
    # Clear cache after validation
    torch.cuda.empty_cache()
    
    return {
        'val_loss': losses.avg,
        'val_fape': fape_losses.avg,
        'val_coord': coord_losses.avg
    }


def main():
    """Main training loop"""
    # Mitigate fragmentation-related CUDA OOM - use newer PYTORCH_ALLOC_CONF
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    print("=" * 50)
    print("RNA 3D Structure Prediction - Option B Training")
    print("=" * 50)
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Epochs: {cfg.epochs}")
    print("=" * 50)
    max_train_minutes = getattr(cfg, 'max_train_minutes', 0)
    max_train_seconds = max(0, int(max_train_minutes * 60))
    safety_buffer_seconds = 20 * 60  # Leave buffer before hard notebook timeout
    train_start_time = time.time()
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = build_model(cfg).to(cfg.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create EMA model
    ema = EMAModel(model, decay=cfg.ema_decay)
    
    # Loss function
    criterion = StructureLoss(
        weights=cfg.loss_weights,
        max_coord_abs=getattr(cfg, 'coord_abs_threshold', 2000.0),
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = max(1, (len(train_loader) // max(1, cfg.grad_accum_steps))) * cfg.epochs
    pct_start = min(0.3, max(0.01, cfg.warmup_steps / total_steps))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_checkpoints = []  # Track top-k checkpoints
    
    history = {'train_loss': [], 'val_loss': [], 'val_fape': []}
    
    for epoch in range(1, cfg.epochs + 1):
        if max_train_seconds > 0 and epoch > 1:
            elapsed = time.time() - train_start_time
            avg_epoch_seconds = elapsed / (epoch - 1)
            projected_next_epoch_end = elapsed + avg_epoch_seconds
            budget_limit = max_train_seconds - safety_buffer_seconds
            if projected_next_epoch_end > budget_limit:
                print("\n[WARN] Stopping early to stay within Kaggle runtime limit.")
                print(f"[WARN] Elapsed: {elapsed/60:.1f} min | Budget: {max_train_seconds/60:.1f} min")
                time_budget_ckpt = os.path.join(cfg.checkpoint_dir, "time_budget_stop.pt")
                save_checkpoint(
                    time_budget_ckpt,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch - 1,
                    val_loss=best_val_loss,
                    ema_state=ema.shadow
                )
                break

        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, cfg, ema
        )
        
        # Validate
        if epoch % cfg.val_frequency == 0:
            val_metrics = validate(model, val_loader, criterion, cfg)
            
            # Log metrics
            for k, v in train_metrics.items():
                history.setdefault(k, []).append(v)
            for k, v in val_metrics.items():
                history.setdefault(k, []).append(v)
            
            print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val FAPE: {val_metrics['val_fape']:.4f}")
            
            # Save checkpoint
            val_loss = val_metrics['val_loss']
            if not np.isfinite(val_loss):
                print("[WARN] Validation loss is non-finite; skipping checkpoint save for this epoch")
            else:
                checkpoint_path = os.path.join(
                    cfg.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
                )

                save_checkpoint(
                    checkpoint_path,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    val_loss=val_loss,
                    ema_state=ema.shadow
                )
            
            # Track top-k checkpoints
            if np.isfinite(val_loss):
                best_checkpoints.append((val_loss, checkpoint_path))
                best_checkpoints.sort(key=lambda x: x[0])
            
            # Keep only top-k
            if len(best_checkpoints) > cfg.save_top_k:
                # Remove worst checkpoint
                _, remove_path = best_checkpoints.pop()
                if os.path.exists(remove_path):
                    os.remove(remove_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                save_checkpoint(
                    best_path,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    val_loss=val_loss,
                    ema_state=ema.shadow
                )
                print(f"[OK] Saved best model (val_loss: {val_loss:.4f})")
    
    # Save training history
    history_path = os.path.join(cfg.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Top checkpoints saved in: {cfg.checkpoint_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
