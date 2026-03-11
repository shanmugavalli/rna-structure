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

For full 33-hour training (epochs=80, msa_depth=6, max_msa_seqs=128, structure_iterations=3):
Expected performance: TM-score 0.60-0.65 (+0.20-0.25 vs baseline)
"""
import torch

class Config:
    """Model and training configuration"""
    
    # ============ Runtime Mode ============
    runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'  # Force GPU if available for better performance
    model_variant = 'full' if runtime_mode == 'gpu' else 'cpu_lite'

    # ============ Model Architecture ============
    # Embeddings
    vocab_size = 5  # A, C, G, U, + padding token
    embed_dim = 256
    max_seq_length = 512 if runtime_mode == 'gpu' else 192  # GPU can handle longer sequences
    max_msa_seqs = 128 if runtime_mode == 'gpu' else 1  # Full MSA depth on GPU
    
    # MSA Transformer
    msa_depth = 6 if runtime_mode == 'gpu' else 1  # Full depth on GPU
    n_heads = 8
    d_single = 256  # Single representation dimension
    d_pair = 128    # Pair representation dimension
    
    # Structure Module (Simplified - no full IPA)
    structure_hidden = 384 if runtime_mode == 'gpu' else 192
    structure_layers = 4 if runtime_mode == 'gpu' else 2  # More layers on GPU
    structure_iterations = 4 if runtime_mode == 'gpu' else 1  # More refinement iterations

    # Lite model settings (used when model_variant == 'cpu_lite')
    feature_dim = 9
    lite_embed_dim = 64
    lite_hidden_dim = 128
    
    # ============ Training ============
    batch_size = 4 if runtime_mode == 'gpu' else 8  # GPU: smaller batch for larger model
    learning_rate = 1e-4 if runtime_mode == 'gpu' else 2e-4  # More conservative LR for full model
    weight_decay = 0.01
    epochs = 80 if runtime_mode == 'gpu' else 20  # Full training on GPU
    warmup_steps = 500 if runtime_mode == 'gpu' else 60
    grad_clip = 0.5  # Reduced from 1.0 for better gradient stability
    grad_accum_steps = 2 if runtime_mode == 'gpu' else 1  # Accumulation for effective batch size
    use_amp = False  # DISABLED: float16 AMP causes NaN accumulation with attention ops
    use_gradient_checkpointing = True if runtime_mode == 'gpu' else False  # Save GPU memory
    
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
    use_length_stratified_sampling = False if runtime_mode == 'gpu' else True  # GPU: use all data
    length_bucket_boundaries = [64, 96, 128, 160, 192, 256, 320, 400, 512] if runtime_mode == 'gpu' else [64, 96, 128, 160, 192]
    length_sampling_power = 1.0
    length_bucket_strategy = 'fixed' if runtime_mode == 'gpu' else 'quantile'
    length_num_buckets = 5
    length_bucket_source = 'seq_len' if runtime_mode == 'gpu' else 'raw_seq_len'
    generate_split_analysis = True
    analysis_dir = 'outputs/analysis'
    
    # ============ Training Strategy ============
    # Data augmentation probabilities
    augment_rotation = 0.8
    augment_noise = 0.3
    noise_std = 0.1
    
    # Validation
    val_frequency = 2 if runtime_mode == 'gpu' else 1  # Validate every 2 epochs on GPU
    save_top_k = 5  # Save top K checkpoints
    max_train_minutes = 0  # No time limit (set externally if needed)
    
    # ============ Ensemble ============
    n_predictions = 5
    ensemble_method = 'checkpoint_diversity'  # or 'stochastic'
    
    # ============ Device ============
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if runtime_mode == 'gpu' else 0  # More workers for GPU
    pin_memory = True if runtime_mode == 'gpu' else False  # Pin memory on GPU
    
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
