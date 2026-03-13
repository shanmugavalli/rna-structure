"""
Configuration for RNA 3D Structure Prediction - Option B (Pragmatic Hybrid)

⚡ KAGGLE TIME-LIMIT SAFE CONFIGURATION ⚡
This configuration is optimized to finish reliably within a Kaggle GPU session.

Key settings:
- 8 epochs (with gradient accumulation)
- 2 MSA transformer blocks
- 2 structure iterations
- 6 MSA sequences (reduced for speed/memory)
- Batch size 1 with accumulation (effective batch larger)

Expected performance: lower peak memory usage and stable training on Kaggle GPU.

For full multi-day training (large GPUs only), increase depth/msa/length gradually.
"""
import torch

try:
    import torch_xla.core.xla_model as xm
except Exception:
    xm = None

class Config:
    """Model and training configuration"""
    
    # ============ Runtime Mode ============
    if xm is not None:
        try:
            _xla_device = xm.xla_device()
            runtime_mode = 'tpu'
        except Exception:
            _xla_device = None
            runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        _xla_device = None
        runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    model_variant = 'full' if runtime_mode == 'gpu' else 'cpu_lite'
    if runtime_mode == 'tpu':
        model_variant = 'full'

    # ============ Model Architecture ============
    # Embeddings
    vocab_size = 5  # A, C, G, U, + padding token
    embed_dim = 256
    max_seq_length = 384 if runtime_mode == 'tpu' else (512 if runtime_mode == 'gpu' else 192)
    max_msa_seqs = 12 if runtime_mode == 'tpu' else (6 if runtime_mode == 'gpu' else 1)
    
    # MSA Transformer
    msa_depth = 4 if runtime_mode == 'tpu' else (2 if runtime_mode == 'gpu' else 1)  # TPU can support deeper stacks
    n_heads = 8
    d_single = 256  # Single representation dimension
    d_pair = 128    # Pair representation dimension
    
    # Structure Module (Simplified - no full IPA)
    structure_hidden = 256 if runtime_mode in ('gpu', 'tpu') else 192
    structure_layers = 2 if runtime_mode in ('gpu', 'tpu') else 2
    structure_iterations = 3 if runtime_mode == 'tpu' else (2 if runtime_mode == 'gpu' else 1)

    # Lite model settings (used when model_variant == 'cpu_lite')
    feature_dim = 9
    lite_embed_dim = 64
    lite_hidden_dim = 128
    
    # ============ Training ============
    batch_size = 4 if runtime_mode == 'tpu' else (1 if runtime_mode == 'gpu' else 8)
    learning_rate = 1.5e-4 if runtime_mode == 'tpu' else (1e-4 if runtime_mode == 'gpu' else 2e-4)
    weight_decay = 0.01
    epochs = 40 if runtime_mode == 'tpu' else (20 if runtime_mode == 'gpu' else 20)
    warmup_steps = 200 if runtime_mode == 'tpu' else (120 if runtime_mode == 'gpu' else 60)
    grad_clip = 0.5  # Reduced from 1.0 for better gradient stability
    grad_accum_steps = 1 if runtime_mode == 'tpu' else (8 if runtime_mode == 'gpu' else 1)
    use_amp = False  # DISABLED: float16 AMP causes NaN accumulation with attention ops
    use_gradient_checkpointing = True if runtime_mode in ('gpu', 'tpu') else False
    
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

    # ============ Data Pipeline ============
    use_cached_dataset = True
    cache_dir = 'data/processed'
    use_msa = False if runtime_mode == 'cpu' else True
    cpu_train_max_samples = 2000 if runtime_mode == 'cpu' else 0
    cpu_val_max_samples = 600 if runtime_mode == 'cpu' else 0
    max_corruption_rate = 0.35
    coord_abs_threshold = 2000.0
    min_valid_ratio = 0.70
    max_outlier_rate = 0.25
    use_length_stratified_sampling = True
    length_bucket_boundaries = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512] if runtime_mode in ('gpu', 'tpu') else [64, 96, 128, 160, 192]
    length_sampling_power = 1.0
    length_bucket_strategy = 'quantile' if runtime_mode == 'gpu' else 'quantile'
    length_num_buckets = 4
    length_bucket_source = 'raw_seq_len' if runtime_mode == 'gpu' else 'raw_seq_len'
    generate_split_analysis = True
    analysis_dir = 'outputs/analysis'
    
    # ============ Training Strategy ============
    # Data augmentation probabilities
    augment_rotation = 0.8
    augment_noise = 0.3
    noise_std = 0.1
    
    # Validation
    val_frequency = 1 if runtime_mode == 'gpu' else 1
    save_top_k = 5  # Save top K checkpoints
    max_train_minutes = 0  # No time limit (set externally if needed)
    
    # ============ Ensemble ============
    n_predictions = 5
    ensemble_method = 'checkpoint_diversity'  # or 'stochastic'
    
    # ============ Device ============
    device = _xla_device if runtime_mode == 'tpu' else ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if runtime_mode == 'tpu' else (2 if runtime_mode == 'gpu' else 0)
    pin_memory = False if runtime_mode == 'tpu' else (True if runtime_mode == 'gpu' else False)
    
    # ============ Logging ============
    log_dir = 'outputs/logs'
    checkpoint_dir = 'outputs/checkpoints'
    prediction_dir = 'outputs/predictions'
    log_every_n_steps = 50
    
    # ============ Loss Curriculum ============
    @staticmethod
    def get_loss_weights(epoch):
        """Progressive loss weight adjustment (adjusted for 40-epoch training)"""
        if Config.model_variant == 'cpu_lite':
            if epoch < 8:
                return {
                    'fape': 1.0,
                    'coord': 1.2,
                    'bond': 0.1,
                    'clash': 0.0,
                }
            return {
                'fape': 1.0,
                'coord': 0.8,
                'bond': 0.2,
                'clash': 0.05,
            }

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
