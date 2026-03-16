# RNA 3D Structure Prediction - Improved Architecture

## Quick Start: Pragmatic Hybrid Approach (Recommended)

This document provides a concrete implementation guide for upgrading from the KISS baseline to a high-performance architecture.

---

## Architecture Overview

### Pipeline Flow

```
Input Data Sources
├── Sequence (ACGU string)
├── MSA (Multiple Sequence Alignment)
└── (Optional) Template structures from PDB

↓

Feature Extraction
├── Sequence Embedding (learned)
├── MSA Processing → Pair + Single representations
└── Positional Encoding

↓

Core Structure Prediction
├── MSA Transformer Blocks (4-8 layers)
│   ├── Row Attention (within sequences)
│   ├── Column Attention (across positions)
│   └── Triangle Attention (pair updates)
│
└── Structure Module (3-4 iterations)
    ├── Invariant Point Attention (IPA)
    ├── Backbone Frame Update
    └── Coordinate Prediction

↓

Output
└── Per-residue 3D coordinates (x, y, z) for C1' atoms
```

---

## Module Specifications

### Module 1: Input Processing

**File:** `src/modules/embeddings.py`

**Responsibilities:**
- Tokenize RNA sequence (A=0, C=1, G=2, U=3)
- Create learned embeddings (dim=256)
- Add positional encodings (sinusoidal or learned)
- Process MSA into tensor format

**Input:**
- Sequence: `(batch, N_residues)` integers
- MSA: `(batch, M_seqs, N_residues)` integers

**Output:**
- Sequence embeddings: `(batch, N_residues, 256)`
- MSA embeddings: `(batch, M_seqs, N_residues, 256)`

**Key Parameters:**
```python
vocab_size = 4  # A, C, G, U
embed_dim = 256
max_seq_length = 512
max_msa_seqs = 128
```

---

### Module 2: MSA Transformer

**File:** `src/modules/msa_module.py`

**Responsibilities:**
- Extract evolutionary patterns from MSA
- Generate pair representation (N×N) capturing residue-residue relationships
- Generate single representation (N) for each residue

**Architecture:**

```python
class MSATransformerBlock:
    def __init__(self, d_single=256, d_pair=128, n_heads=8):
        # Row attention: attend within each sequence
        self.row_attn = MultiHeadAttention(d_single, n_heads)
        
        # Column attention: attend across homologs
        self.col_attn = MultiHeadAttention(d_single, n_heads)
        
        # Update pair representation
        self.pair_update = TriangleAttention(d_pair)
        
        # Outer product mean: MSA → Pair
        self.outer_product = OuterProductMean(d_single, d_pair)
    
    def forward(self, msa, pair):
        # msa: (batch, M_seqs, N, d_single)
        # pair: (batch, N, N, d_pair)
        
        # Row attention
        msa = self.row_attn(msa)  # within sequence
        
        # Column attention
        msa = self.col_attn(msa)  # across sequences
        
        # Update pair from MSA
        pair = pair + self.outer_product(msa)
        
        # Triangle updates (geometric consistency)
        pair = self.pair_update(pair)
        
        return msa, pair
```

**Hyperparameters:**
- Depth: 4-8 blocks
- d_single: 256
- d_pair: 128
- Attention heads: 8
- Dropout: 0.1

**Output:**
- Single representation: `(batch, N, 256)` - per-residue features
- Pair representation: `(batch, N, N, 128)` - residue-residue features

---

### Module 3: Structure Module (IPA-based)

**File:** `src/modules/structure_module.py`

**Responsibilities:**
- Iteratively refine 3D structure predictions
- Use Invariant Point Attention (geometry-aware)
- Predict rotation + translation updates for each residue frame
- Generate final C1' coordinates

**Architecture:**

```python
class StructureModule:
    def __init__(self, d_single=384, n_heads=12, n_iterations=3):
        self.iterations = n_iterations
        
        # Invariant Point Attention
        self.ipa = InvariantPointAttention(
            d_single=d_single,
            n_heads=n_heads,
            n_query_points=4,
            n_point_values=8
        )
        
        # Backbone update network
        self.backbone_update = nn.Linear(d_single, 6)  # 3 rotation + 3 translation
        
        # Coordinate head
        self.coord_head = nn.Linear(d_single, 3)  # x, y, z
    
    def forward(self, single_repr, pair_repr):
        # Initialize frames (identity rotation, zero translation)
        frames = init_frames(batch_size, N)
        
        for i in range(self.iterations):
            # IPA: attention using both features and 3D geometry
            single_repr = self.ipa(single_repr, pair_repr, frames)
            
            # Predict frame update
            delta_frame = self.backbone_update(single_repr)
            frames = update_frames(frames, delta_frame)
            
            # Generate coordinates from frames
            coords = self.coord_head(single_repr)
            coords = apply_frame_to_coords(frames, coords)
        
        return coords  # (batch, N, 3)
```

**Key Concepts:**

**Invariant Point Attention (IPA):**
- Attends to both feature space AND 3D point clouds
- Ensures SE(3) equivariance (predictions are rotation/translation invariant)
- Core innovation from AlphaFold2

**Frame Representation:**
- Each residue has a local frame: rotation (3×3 matrix) + translation (3D vector)
- Frames are updated iteratively to refine structure

**Hyperparameters:**
- Iterations: 3-4 (with recycling)
- d_single: 384
- IPA heads: 12
- Query points per head: 4
- Dropout: 0.1

---

### Module 4: Loss Functions

**File:** `src/losses.py`

**Responsibilities:**
- FAPE (Frame Aligned Point Error) - primary loss
- Geometric constraint losses
- Auxiliary coordinate losses

**Loss Breakdown:**

```python
def compute_total_loss(pred_coords, true_coords, pred_frames):
    losses = {}
    
    # 1. FAPE Loss (primary)
    losses['fape'] = fape_loss(pred_coords, true_coords, pred_frames)
    # Weight: 1.0
    
    # 2. Coordinate RMSD (auxiliary)
    losses['rmsd'] = rmsd_loss(pred_coords, true_coords)
    # Weight: 0.2
    
    # 3. Bond distance regularization
    losses['bond'] = bond_distance_loss(pred_coords, target_dist=6.0)
    # Weight: 0.5
    
    # 4. Clash penalty
    losses['clash'] = clash_penalty(pred_coords, min_dist=3.0)
    # Weight: 0.3
    
    # 5. Smooth L1 (Huber)
    losses['smooth_l1'] = F.smooth_l1_loss(pred_coords, true_coords)
    # Weight: 0.2
    
    # Total weighted loss
    total = (1.0 * losses['fape'] + 
             0.2 * losses['rmsd'] +
             0.5 * losses['bond'] +
             0.3 * losses['clash'] +
             0.2 * losses['smooth_l1'])
    
    return total, losses
```

**FAPE Loss Details:**

```python
def fape_loss(pred_coords, true_coords, frames, clamp_distance=10.0):
    """
    Frame Aligned Point Error
    - Aligns predictions to true structure using predicted frames
    - Rotation-invariant loss
    - Clamps distances to avoid outlier gradients
    """
    # Transform coords to local frame
    pred_local = apply_inverse_frame(pred_coords, frames)
    true_local = apply_inverse_frame(true_coords, frames)
    
    # Compute clamped L1 distance
    dist = torch.norm(pred_local - true_local, dim=-1)
    dist = torch.clamp(dist, max=clamp_distance)
    
    return dist.mean()
```

---

## Training Configuration

### Hyperparameters

```python
# Model
config = {
    # Architecture
    'embed_dim': 256,
    'msa_depth': 6,
    'structure_iterations': 3,
    'ipa_heads': 12,
    
    # Data
    'max_seq_len': 512,
    'max_msa_seqs': 128,
    'batch_size': 8,
    
    # Training
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'epochs': 80,
    'warmup_steps': 1000,
    'grad_clip': 1.0,
    
    # Loss weights
    'loss_weights': {
        'fape': 1.0,
        'rmsd': 0.2,
        'bond': 0.5,
        'clash': 0.3,
        'smooth_l1': 0.2
    },
    
    # Regularization
    'dropout': 0.1,
    'ema_decay': 0.999,
    
    # Ensemble
    'n_predictions': 5,
    'ensemble_method': 'checkpoint_diversity'
}
```

### Loss Curriculum (Progressive Training)

```python
def get_loss_weights(epoch):
    """Gradually increase loss complexity"""
    if epoch < 10:
        # Early: focus on coordinate accuracy
        return {'fape': 1.0, 'rmsd': 1.0, 'bond': 0.0, 'clash': 0.0, 'smooth_l1': 0.5}
    elif epoch < 30:
        # Mid: add bond constraints
        return {'fape': 1.0, 'rmsd': 0.5, 'bond': 0.5, 'clash': 0.0, 'smooth_l1': 0.2}
    else:
        # Late: full geometric constraints
        return {'fape': 1.0, 'rmsd': 0.2, 'bond': 0.5, 'clash': 0.3, 'smooth_l1': 0.2}
```

### Data Augmentation

```python
def augment_structure(coords, sequence):
    """Apply random augmentations during training"""
    
    # 1. Random rotation (3D)
    if random.random() < 0.8:
        rotation = random_rotation_matrix()
        coords = coords @ rotation.T
    
    # 2. Random translation
    if random.random() < 0.5:
        translation = torch.randn(3) * 5.0
        coords = coords + translation
    
    # 3. Gaussian noise injection
    if random.random() < 0.3:
        noise = torch.randn_like(coords) * 0.1
        coords = coords + noise
    
    return coords, sequence
```

---

## Ensemble Strategy for 5 Predictions

### Strategy: Diverse Checkpoint Ensemble

```python
def generate_5_predictions(model_checkpoints, test_sequence, test_msa):
    """
    Generate 5 diverse predictions per sequence
    
    Method:
    - 3 predictions from different training checkpoints
    - 2 predictions from stochastic passes (dropout on)
    """
    predictions = []
    
    # Load 3 best checkpoints (by validation TM-score)
    ckpt_1 = load_checkpoint('outputs/checkpoints/best_epoch_45.pt')
    ckpt_2 = load_checkpoint('outputs/checkpoints/best_epoch_52.pt')
    ckpt_3 = load_checkpoint('outputs/checkpoints/best_epoch_58.pt')
    
    for ckpt in [ckpt_1, ckpt_2, ckpt_3]:
        model.load_state_dict(ckpt)
        model.eval()
        with torch.no_grad():
            pred = model(test_sequence, test_msa)
        predictions.append(pred)
    
    # 2 stochastic passes with dropout enabled
    model.load_state_dict(ckpt_1)  # Use best checkpoint
    model.train()  # Keep dropout active
    for _ in range(2):
        with torch.no_grad():
            pred = model(test_sequence, test_msa)
        predictions.append(pred)
    
    return predictions  # List of 5 predictions
```

### Alternative: Multi-Seed Ensemble

Train 5 separate models with different random seeds and combine:

```python
# Train 5 models
for seed in [42, 123, 256, 512, 999]:
    set_seed(seed)
    model = RNAStructureModel(config)
    train(model, train_loader, val_loader)
    save_checkpoint(f'model_seed_{seed}.pt')

# Inference: each model gives 1 prediction
predictions = [model_i(test_data) for model_i in models]
```

---

## Implementation Checklist

### Phase 1: Core Modules (Week 1)

- [ ] Implement `embeddings.py` (sequence + MSA tokenization)
- [ ] Implement MSA Transformer blocks
- [ ] Implement basic structure module (without IPA first)
- [ ] Implement FAPE loss + geometric losses
- [ ] Test each module independently

### Phase 2: Integration (Week 2)

- [ ] Integrate all modules into main model
- [ ] Set up training loop with loss curriculum
- [ ] Implement data augmentation
- [ ] Add logging and checkpointing
- [ ] Run sanity check on small dataset

### Phase 3: Full Training (Week 3)

- [ ] Train on full dataset
- [ ] Monitor validation TM-score
- [ ] Save multiple checkpoints
- [ ] Implement EMA for stable predictions

### Phase 4: Inference & Ensemble (Week 4)

- [ ] Implement 5-prediction ensemble strategy
- [ ] Generate submission file
- [ ] Validate submission format
- [ ] Analyze prediction diversity

---

## Expected Performance Improvements

| Metric | KISS Baseline | Improved Architecture | Delta |
|--------|---------------|----------------------|-------|
| TM-score (validation) | 0.30-0.40 | 0.55-0.70 | +0.25 |
| RMSD (Å) | 15-25 | 6-12 | -10 |
| Training time/epoch | 5 min | 25 min | +20 min |
| GPU memory | 4 GB | 16 GB | +12 GB |
| Implementation time | 2 days | 2-3 weeks | +2.5 weeks |

---

## Debugging Tips

### Common Issues

**1. NaN losses early in training**
- Solution: Reduce learning rate to 1e-5, add gradient clipping
- Check for coordinate explosions in structure module

**2. Model not improving after baseline**
- Solution: Verify MSA data is loaded correctly
- Check loss weights (FAPE might dominate)

**3. Predictions collapse to mean structure**
- Solution: Increase dropout, add more augmentation
- Reduce model capacity if overfitting

**4. Out of memory errors**
- Solution: Reduce batch size, MSA depth, or max_msa_seqs
- Use gradient checkpointing

### Validation Checks

```python
# Check MSA module output shapes
msa_emb, pair_rep = msa_module(msa_input)
assert msa_emb.shape == (batch, N, 256)
assert pair_rep.shape == (batch, N, N, 128)

# Check structure module outputs valid coordinates
coords = structure_module(single, pair)
assert coords.shape == (batch, N, 3)
assert not torch.isnan(coords).any()
assert coords.abs().max() < 1000  # Reasonable coordinate range

# Check loss values
loss, loss_dict = compute_loss(pred, true)
print(loss_dict)  # All components should be non-negative and reasonable
```

---

## Next Steps

1. **Review PRD.md** for full architectural context
2. **Set up environment** with required dependencies
3. **Start with Phase 1** implementation
4. **Iterate and improve** based on validation metrics

For questions, refer to:
- **AlphaFold2 paper**: Jumper et al., Nature 2021
- **RoseTTAFold paper**: Baek et al., Science 2021
- **SE(3)-Transformers**: https://github.com/FabianFuchsML/se3-transformer-public
