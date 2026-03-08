"""
Training script for RNA structure prediction
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from config import cfg
from model import RNAStructurePredictor, EMAModel
from dataset import create_dataloaders
from losses import StructureLoss
from utils import save_checkpoint, load_checkpoint, AverageMeter


def apply_augmentation(coords, config):
    """Apply data augmentation to coordinates"""
    batch_size, seq_len, _ = coords.shape
    
    # Random rotation
    if torch.rand(1).item() < config.augment_rotation:
        # Random rotation matrix for each sample in batch
        for i in range(batch_size):
            angle = torch.rand(3) * 2 * np.pi
            Rx = torch.tensor([[1, 0, 0],
                              [0, torch.cos(angle[0]), -torch.sin(angle[0])],
                              [0, torch.sin(angle[0]), torch.cos(angle[0])]])
            Ry = torch.tensor([[torch.cos(angle[1]), 0, torch.sin(angle[1])],
                              [0, 1, 0],
                              [-torch.sin(angle[1]), 0, torch.cos(angle[1])]])
            Rz = torch.tensor([[torch.cos(angle[2]), -torch.sin(angle[2]), 0],
                              [torch.sin(angle[2]), torch.cos(angle[2]), 0],
                              [0, 0, 1]])
            R = Rz @ Ry @ Rx
            coords[i] = coords[i] @ R.T.to(coords.device)
    
    # Gaussian noise
    if torch.rand(1).item() < config.augment_noise:
        noise = torch.randn_like(coords) * config.noise_std
        coords = coords + noise
    
    return coords


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
    scaler = torch.amp.GradScaler('cuda', enabled=(config.use_amp and config.device == 'cuda'))
    accum_steps = max(1, config.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(pbar):
        # Move to device
        seq_tokens = batch['seq_tokens'].to(config.device)
        msa_tokens = batch['msa_tokens'].to(config.device)
        true_coords = batch['coords'].to(config.device)
        
        # Apply augmentation to ground truth
        true_coords = apply_augmentation(true_coords, config)
        
        # Forward pass (mixed precision on CUDA)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(config.use_amp and config.device == 'cuda')):
            pred_coords, all_coords = model(seq_tokens, msa_tokens)
            loss, loss_dict = criterion(pred_coords, true_coords, all_coords)

        # Gradient accumulation
        scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update meters
        losses.update(loss.item())
        fape_losses.update(loss_dict['fape'].item())
        coord_losses.update(loss_dict['coord'].item())
        
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
        
        # Forward pass
        pred_coords, all_coords = model(seq_tokens, msa_tokens)
        
        # Compute loss
        loss, loss_dict = criterion(pred_coords, true_coords, all_coords)
        
        # Update meters
        losses.update(loss.item())
        fape_losses.update(loss_dict['fape'].item())
        coord_losses.update(loss_dict['coord'].item())
    
    return {
        'val_loss': losses.avg,
        'val_fape': fape_losses.avg,
        'val_coord': coord_losses.avg
    }


def main():
    """Main training loop"""
    # Mitigate fragmentation-related CUDA OOM on Kaggle.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    print("=" * 50)
    print("RNA 3D Structure Prediction - Option B Training")
    print("=" * 50)
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Epochs: {cfg.epochs}")
    print("=" * 50)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = RNAStructurePredictor(cfg).to(cfg.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create EMA model
    ema = EMAModel(model, decay=cfg.ema_decay)
    
    # Loss function
    criterion = StructureLoss(weights=cfg.loss_weights)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = max(1, (len(train_loader) // max(1, cfg.grad_accum_steps))) * cfg.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.warmup_steps / total_steps,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_checkpoints = []  # Track top-k checkpoints
    
    history = {'train_loss': [], 'val_loss': [], 'val_fape': []}
    
    for epoch in range(1, cfg.epochs + 1):
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
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir,
                f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
            )
            
            save_checkpoint(
                checkpoint_path,
                model=model.state_dict(),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                ema_state=ema.shadow
            )
            
            # Track top-k checkpoints
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
                    optimizer=optimizer,
                    scheduler=scheduler,
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
