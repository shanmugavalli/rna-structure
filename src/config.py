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
import os

import torch

try:
    import torch_xla.core.xla_model as xm
except Exception:
    xm = None


def _env_str(name, default=""):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip()


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _detect_tpu_core_count():
    """Best-effort TPU core detection across torch-xla versions."""
    if xm is None:
        return 0

    # Newer torch-xla runtime API
    try:
        import torch_xla.runtime as xr  # type: ignore

        n = int(xr.global_runtime_device_count())
        if n > 0:
            return n
    except Exception:
        pass

    # Legacy API commonly available in Kaggle TPU runtimes
    try:
        n = int(xm.xrt_world_size())
        if n > 0:
            return n
    except Exception:
        pass

    # Fallback: count reported XLA devices
    try:
        devices = xm.get_xla_supported_devices(devkind="TPU")
        if devices:
            return len(devices)
    except Exception:
        pass

    try:
        devices = xm.get_xla_supported_devices()
        if devices:
            return len(devices)
    except Exception:
        pass

    # If TPU runtime is present but not queryable, assume single visible core.
    return 1

class Config:
    """Model and training configuration"""

    # ============ Environment Overrides ============
    force_runtime = _env_str('RNA_RUNTIME', '').lower()
    force_model_variant = _env_str('RNA_MODEL_VARIANT', '').lower()
    small_debug = _env_bool('RNA_SMALL_DEBUG', False)
    debug_use_train_as_val = _env_bool('RNA_DEBUG_USE_TRAIN_AS_VAL', False)
    
    # ============ Runtime Mode ============
    if force_runtime == 'cpu':
        _xla_device = None
        runtime_mode = 'cpu'
    elif force_runtime == 'gpu':
        _xla_device = None
        runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    elif force_runtime == 'tpu':
        if xm is not None:
            try:
                _xla_device = xm.xla_device()
                runtime_mode = 'tpu'
            except Exception:
                _xla_device = None
                runtime_mode = 'cpu'
        else:
            _xla_device = None
            runtime_mode = 'cpu'
    elif xm is not None:
        try:
            _xla_device = xm.xla_device()
            runtime_mode = 'tpu'
        except Exception:
            _xla_device = None
            runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        _xla_device = None
        runtime_mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    model_variant = 'full' if runtime_mode in ('gpu', 'tpu') else 'cpu_lite'
    if force_model_variant in {'full', 'cpu_lite'}:
        model_variant = force_model_variant
    tpu_core_count = _detect_tpu_core_count() if runtime_mode == 'tpu' else 0

    # ============ Model Architecture ============
    # TPU-safe profile: keep settings conservative to avoid XLA HBM compile OOM
    # on Kaggle v3 TPUs.
    # Embeddings
    vocab_size = 5  # A, C, G, U, + padding token
    embed_dim = _env_int('RNA_EMBED_DIM', 192 if runtime_mode == 'tpu' else 256)
    max_seq_length = _env_int('RNA_MAX_SEQ_LENGTH', 256 if runtime_mode == 'tpu' else (512 if runtime_mode == 'gpu' else 192))
    max_msa_seqs = _env_int('RNA_MAX_MSA_SEQS', 6 if runtime_mode in ('gpu', 'tpu') else 1)
    
    # MSA Transformer
    msa_depth = _env_int('RNA_MSA_DEPTH', 2 if runtime_mode in ('gpu', 'tpu') else 1)
    n_heads = _env_int('RNA_N_HEADS', 4 if runtime_mode == 'tpu' else 8)
    d_single = _env_int('RNA_D_SINGLE', 192 if runtime_mode == 'tpu' else 256)  # Single representation dimension
    d_pair = _env_int('RNA_D_PAIR', 96 if runtime_mode == 'tpu' else 128)    # Pair representation dimension
    
    # Structure Module (Simplified - no full IPA)
    structure_hidden = _env_int('RNA_STRUCTURE_HIDDEN', 192 if runtime_mode == 'tpu' else (256 if runtime_mode == 'gpu' else 192))
    structure_layers = _env_int('RNA_STRUCTURE_LAYERS', 2)
    structure_iterations = _env_int('RNA_STRUCTURE_ITERATIONS', 2 if runtime_mode in ('gpu', 'tpu') else 1)

    # Lite model settings (used when model_variant == 'cpu_lite')
    feature_dim = 9
    lite_embed_dim = 64
    lite_hidden_dim = 128
    
    # ============ Training ============
    batch_size = _env_int('RNA_BATCH_SIZE', 1 if runtime_mode == 'tpu' else (1 if runtime_mode == 'gpu' else 8))
    learning_rate = _env_float('RNA_LR', 1.5e-4 if runtime_mode == 'tpu' else (1e-4 if runtime_mode == 'gpu' else 2e-4))
    weight_decay = 0.01
    epochs = _env_int('RNA_EPOCHS', 60 if runtime_mode == 'tpu' else (20 if runtime_mode == 'gpu' else 20))
    warmup_steps = 200 if runtime_mode == 'tpu' else (120 if runtime_mode == 'gpu' else 60)
    grad_clip = 0.5  # Reduced from 1.0 for better gradient stability
    if runtime_mode == 'tpu':
        # Keep global effective batch size in a safe range while adapting to core count.
        if tpu_core_count <= 1:
            grad_accum_steps = 8
        elif tpu_core_count <= 4:
            grad_accum_steps = 4
        else:
            grad_accum_steps = 2
    else:
        grad_accum_steps = 8 if runtime_mode == 'gpu' else 1
    enable_runtime_tensor_checks = False if runtime_mode == 'tpu' else True
    if runtime_mode == 'tpu':
        # Prefer more epochs with fewer steps each for TPU runtime stability.
        if tpu_core_count <= 1:
            train_num_samples_per_epoch = 256
        elif tpu_core_count <= 4:
            train_num_samples_per_epoch = 512
        else:
            train_num_samples_per_epoch = 1024
    else:
        train_num_samples_per_epoch = 0
    max_steps_per_epoch = int(train_num_samples_per_epoch) if runtime_mode == 'tpu' else 0
    use_amp = False  # DISABLED: float16 AMP causes NaN accumulation with attention ops
    # torch.utils.checkpoint currently breaks on Kaggle TPU/XLA in this setup
    # (AttributeError: module 'torch' has no attribute 'xla').
    use_gradient_checkpointing = True if runtime_mode == 'gpu' else False
    
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
    use_msa = _env_bool('RNA_USE_MSA', False if runtime_mode == 'cpu' else True)
    cpu_train_max_samples = _env_int('RNA_CPU_TRAIN_MAX_SAMPLES', 2000 if runtime_mode == 'cpu' else 0)
    cpu_val_max_samples = _env_int('RNA_CPU_VAL_MAX_SAMPLES', 600 if runtime_mode == 'cpu' else 0)
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
    augment_rotation = _env_float('RNA_AUG_ROT', 0.8)
    augment_noise = _env_float('RNA_AUG_NOISE', 0.3)
    noise_std = _env_float('RNA_AUG_NOISE_STD', 0.1)
    
    # Validation
    val_frequency = _env_int('RNA_VAL_FREQUENCY', 3 if runtime_mode == 'tpu' else 1)
    save_top_k = 5  # Save top K checkpoints
    max_train_minutes = _env_float('RNA_MAX_TRAIN_MINUTES', 0)  # No time limit (set externally if needed)
    val_tm_on_tpu = _env_bool('RNA_VAL_TM_ON_TPU', True)
    val_tm_max_samples = _env_int('RNA_VAL_TM_MAX_SAMPLES', 96 if runtime_mode == 'tpu' else 0)
    
    # ============ Ensemble ============
    n_predictions = 5
    ensemble_method = 'checkpoint_diversity'  # or 'stochastic'
    
    # ============ Device ============
    device = _xla_device if runtime_mode == 'tpu' else ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = max(2, min(8, int(tpu_core_count))) if runtime_mode == 'tpu' else (2 if runtime_mode == 'gpu' else 0)
    pin_memory = False if runtime_mode == 'tpu' else (True if runtime_mode == 'gpu' else False)
    effective_batch_size = batch_size * max(1, int(tpu_core_count)) * grad_accum_steps

    if small_debug:
        # Fast iterative mode for local CPU debugging and metric sanity checks.
        max_seq_length = min(max_seq_length, _env_int('RNA_DEBUG_MAX_SEQ_LENGTH', 128))
        if runtime_mode == 'cpu':
            use_msa = _env_bool('RNA_DEBUG_USE_MSA', use_msa)
            max_msa_seqs = _env_int('RNA_DEBUG_MAX_MSA', max_msa_seqs)
            cpu_train_max_samples = _env_int('RNA_CPU_TRAIN_MAX_SAMPLES', max(64, min(cpu_train_max_samples or 256, 256)))
            cpu_val_max_samples = _env_int('RNA_CPU_VAL_MAX_SAMPLES', max(24, min(cpu_val_max_samples or 96, 96)))
        epochs = _env_int('RNA_EPOCHS', min(epochs, 8))
        val_frequency = 1

    if debug_use_train_as_val:
        val_seq_path = train_seq_path
        val_label_path = train_label_path
    
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
