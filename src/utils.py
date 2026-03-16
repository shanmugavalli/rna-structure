"""
Utility functions for training and evaluation
"""
import os
import torch


def _to_cpu_serializable(obj):
    """Recursively move tensor containers to CPU for portable checkpoints."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_serializable(v) for v in obj)
    return obj


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(path, **kwargs):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = _to_cpu_serializable(kwargs)
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model=None, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    if model is not None and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"Checkpoint loaded: {path}")
    return checkpoint


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
