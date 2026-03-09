"""
Configuration for RNA 3D Structure Prediction - Option B (Pragmatic Hybrid)

⚡ 8-HOUR TRAINING CONFIGURATION ⚡
This configuration is optimized to run safely on Kaggle P100 (16GB) with reduced OOM risk.

Key settings:
- 40 epochs (with gradient accumulation)
- 4 MSA transformer blocks
- 2 structure iterations
- 16 MSA sequences (reduced for memory safety)
- Batch size 1 with accumulation (effective batch larger)

Expected performance: lower peak memory usage and stable training on Kaggle GPU.

For full 33-hour training (epochs=80, msa_depth=6, max_msa_seqs=128, structure_iterations=3):
Expected performance: TM-score 0.60-0.65 (+0.20-0.25 vs baseline)
"""
import torch

class Config:
    """Model and training configuration"""
    
    # ============ Model Architecture ============
    # Embeddings
    vocab_size = 5  # A, C, G, U, + padding token
    embed_dim = 256
    max_seq_length = 384
    max_msa_seqs = 8  # Further reduced to avoid OOM (was 16)
    
    # MSA Transformer
    msa_depth = 2  # Reduced to 2 to save memory (was 3)
    n_heads = 8
    d_single = 256  # Single representation dimension
    d_pair = 128    # Pair representation dimension
    
    # Structure Module (Simplified - no full IPA)
    structure_hidden = 384
    structure_layers = 3
    structure_iterations = 2  # Reduced from 3 for faster training
    
    # ============ Training ============
    batch_size = 1  # Reduced to 1 to avoid OOM
    learning_rate = 1e-4
    weight_decay = 0.01
    epochs = 12  # Reduced from 40 for ~6-8 hour training (50min × 12 = 10 hours)
    warmup_steps = 200  # Reduced proportionally with epochs (was 500)
    grad_clip = 1.0
    grad_accum_steps = 8  # Increased to maintain effective batch size
    use_amp = True
    use_gradient_checkpointing = True  # Trade compute for memory
    
    # Loss weights (will be adjusted by curriculum)
    loss_weights = {
        'fape': 1.0,
        'coord': 0.3,
        'bond': 0.5,
        'clash': 0.3,
    }
    
    # ============ Regularization ============
    dropout = 0.1
    ema_decay = 0.999  # Exponential moving average
    
    # ============ Data ============
    train_seq_path = 'data/raw/train_sequences.csv'
    train_label_path = 'data/raw/train_labels.csv'
    val_seq_path = 'data/raw/validation_sequences.csv'
    val_label_path = 'data/raw/validation_labels.csv'
    test_seq_path = 'data/raw/test_sequences.csv'
    msa_dir = 'data/raw/MSA'
    
    # ============ Training Strategy ============
    # Data augmentation probabilities
    augment_rotation = 0.8
    augment_noise = 0.3
    noise_std = 0.1
    
    # Validation
    val_frequency = 1  # Validate every N epochs
    save_top_k = 5  # Save top K checkpoints
    
    # ============ Ensemble ============
    n_predictions = 5
    ensemble_method = 'checkpoint_diversity'  # or 'stochastic'
    
    # ============ Device ============
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2
    pin_memory = True
    
    # ============ Logging ============
    log_dir = 'outputs/logs'
    checkpoint_dir = 'outputs/checkpoints'
    prediction_dir = 'outputs/predictions'
    log_every_n_steps = 50
    
    # ============ Loss Curriculum ============
    @staticmethod
    def get_loss_weights(epoch):
        """Progressive loss weight adjustment (adjusted for 40-epoch training)"""
        if epoch < 5:
            # Early: focus on coordinate accuracy
            return {
                'fape': 1.0,
                'coord': 1.0,
                'bond': 0.0,
                'clash': 0.0,
            }
        elif epoch < 15:
            # Mid: add bond constraints
            return {
                'fape': 1.0,
                'coord': 0.5,
                'bond': 0.5,
                'clash': 0.0,
            }
        else:
            # Late: full geometric constraints
            return {
                'fape': 1.0,
                'coord': 0.3,
                'bond': 0.5,
                'clash': 0.3,
            }
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__class__.__dict__.items() 
                 if not k.startswith('_') and not callable(v)]
        return f"Config({', '.join(attrs)})"


# Global config instance
cfg = Config()
