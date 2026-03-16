# Product Requirements Document (PRD)

## Overview

If you were handed only a melody and asked to compose a full symphony, you would rely on patterns, structure, and musical intuition to build the rest. In the same way, predicting Ribonucleic acid (RNA) 3D structure means using its sequence to determine how it folds into the shapes that define its function.

In Part 2 of the original Stanford RNA 3D Folding competition, participants help uncover how RNA molecules fold and function at the molecular level by developing machine learning models that predict RNA 3D structure from sequence alone.

## Description

RNA is essential to life’s core functions, but predicting its 3D structure remains difficult. Unlike proteins, where models like AlphaFold have made major progress, RNA modeling is still held back by limited data and the complexity of RNA folding.

This is the second Stanford RNA 3D Folding Challenge. The first marked a major milestone, where fully automated models matched human experts for the first time. Now, participants face even more complex targets, including ones with no structural templates, and a new evaluation metric designed to reward greater accuracy.

This work could solve a key challenge in molecular biology. Better RNA structure prediction could unlock new treatments, accelerate research, and deepen our understanding of how life works. The competition runs on a roughly two-month timeline, aiming to surface new breakthroughs ahead of the 17th Critical Assessment of Structure Prediction (CASP17) in April 2026.

This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, NVIDIA Healthcare, the AI@HHMI initiative of the Howard Hughes Medical Institute, and Stanford University School of Medicine.

## Evaluation

Submissions are scored using TM-score (template modeling score), which ranges from `0.0` to `1.0` (higher is better).

## Submission File

For each sequence in the test set, you can predict five structures. Your notebook should read `test_sequences.csv` and output `submission.csv`.

The submission file should contain `x`, `y`, `z` coordinates of the `C1'` atom in each residue across predicted structures 1 through 5.

Example format:

```csv
ID,resname,resid,x_1,y_1,z_1,...,x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,...,-7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,...,-7.953,10.02,12.127
```

## KISS Architecture (Train + Test)

Goal: keep the system simple enough to build fast, debug easily, and improve iteratively.

### 1) Minimal System Design

Use a single-model baseline with one clear pipeline:

1. Load and clean data.
2. Convert sequence into numeric tokens.
3. Predict per-residue 3D coordinates (`x, y, z`) for `C1'` atom.
4. Train with coordinate loss + geometric regularization.
5. Validate each epoch.
6. Run inference on test set.
7. Generate 5 predictions per sequence using test-time ensembling.

### 2) Folder Structure

```text
RNA-STRUCTURE/
	PRD.md
	data/
		raw/
			train_sequences.csv
			train_labels.csv
			test_sequences.csv
		processed/
	src/
		config.py
		dataset.py
		model.py
		losses.py
		train.py
		evaluate.py
		infer.py
		submit.py
		utils.py
	outputs/
		checkpoints/
		logs/
		predictions/
	requirements.txt
```

### 3) Model (KISS Baseline)

Start with a lightweight sequence model:

- Embedding layer for RNA tokens (`A, C, G, U`, plus optional special tokens).
- 2 to 4 BiLSTM layers (or a small Transformer encoder if preferred).
- MLP head that outputs `3` values per residue (`x, y, z`).

Why this baseline:

- Fast to train.
- Easy to debug.
- Strong enough to establish a useful score baseline.

### 4) Training Setup

Input:

- Sequence string per RNA molecule.

Target:

- Per-residue `C1'` coordinates.

Loss:

- Main: Smooth L1 (Huber) loss on coordinates.
- Optional regularizer: distance consistency between adjacent residues.

Validation:

- Primary metric: TM-score (if validation structure and toolchain available).
- Fallback metric during training: RMSD / MAE on coordinates.

Recommended defaults:

- Optimizer: `AdamW`
- LR: `1e-3`
- Batch size: `8` to `32` (based on GPU memory)
- Epochs: `30` to `80`
- Early stopping patience: `8`

### 5) Test-Time Prediction (5 Structures)

Generate 5 outputs per sequence using one of these simple methods:

1. Best 5 checkpoints by validation score.
2. Same checkpoint with 5 stochastic passes (dropout on).
3. Small ensemble of 3 checkpoints + 2 stochastic variants.

Keep method `1` for first version because it is deterministic and easy to reproduce.

### 6) End-to-End Script Responsibilities

- `train.py`: train model, save checkpoints, and log metrics.
- `evaluate.py`: compute validation metrics and rank checkpoints.
- `infer.py`: load top checkpoints and predict test coordinates.
- `submit.py`: format predictions into `submission.csv` with required columns.

### 7) Data + Model Flow

```text
train_sequences.csv + train_labels.csv
	-> dataset.py
	-> model.py
	-> train.py (fit + checkpoint)
	-> evaluate.py (score checkpoints)

test_sequences.csv
	-> infer.py (top-5 predictions)
	-> submit.py
	-> submission.csv
```

### 8) Practical Milestones

1. Build data loader and verify shapes.
2. Train 1 epoch on a tiny subset (sanity check).
3. Train full baseline and save checkpoints.
4. Produce valid `submission.csv`.
5. Improve only one thing at a time (model depth, augmentations, losses, or ensemble strategy).

### 9) Non-Goals for V1

Avoid these in first implementation to stay KISS:

- Complex multi-task objectives.
- Large geometric graph models.
- Heavy feature engineering pipelines.
- Multi-stage training.

Ship a clean baseline first, then iterate.

---

## Improved Architecture (Post-Baseline)

Once you have a working baseline with low scores, upgrade to a state-of-the-art architecture designed specifically for 3D structure prediction.

### Baseline Limitations

The KISS baseline (BiLSTM/Transformer + MLP) has fundamental weaknesses:

1. **Ignores MSA data** - Evolutionary information from multiple sequence alignments is unused.
2. **No geometric awareness** - Treats 3D coordinates as independent regression targets without structural constraints.
3. **No iterative refinement** - Single-pass prediction cannot self-correct.
4. **Limited long-range interactions** - BiLSTM struggles with RNA tertiary structure dependencies.
5. **No diversity in ensemble** - Using top-5 checkpoints gives similar predictions.

### Recommended Architecture: MSA-Enhanced Geometric Transformer

**Type:** Single sophisticated model (not ensemble of simple models)

**Why:** Modern structure prediction (AlphaFold2, RoseTTAFold, etc.) shows that one well-designed model outperforms ensembles of weak models.

#### Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                        │
│  - Target Sequence (N residues)                         │
│  - MSA (M sequences × N residues)                       │
│  - (Optional) Template structures from PDB_RNA          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              MSA FEATURE EXTRACTION                      │
│  - MSA Transformer (row/column attention)               │
│  - Outputs: Pair representation (N×N) + Single (N)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           STRUCTURE MODULE (Iterative)                   │
│  ┌─────────────────────────────────────────┐           │
│  │  Iteration 1, 2, 3 (recycle)            │           │
│  │  - Invariant Point Attention (IPA)      │           │
│  │  - SE(3)-Equivariant Layers             │           │
│  │  - Structure update + rotation/trans    │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              COORDINATE PREDICTION                       │
│  - Per-residue x, y, z for C1' atoms                    │
│  - Optional: Predict full atom coordinates              │
└─────────────────────────────────────────────────────────┘
```

### Specific Model Architecture

#### 1. MSA Module (Evoformer-style)

**Purpose:** Extract evolutionary and co-evolution patterns.

**Layers:**
- Input: MSA of shape `(M_sequences, N_residues, C_channels)`
- MSA row-wise attention: attention within each sequence
- MSA column-wise attention: attention across homologs for each position
- Outer product mean → pair representation `(N, N, C_pair)`
- Triangle attention: update pair features with geometric consistency
- Output: Single representation `(N, C_single)` + Pair `(N, N, C_pair)`

**Params:**
- MSA depth: 4-8 blocks
- Channels: 256 (single), 128 (pair)
- Attention heads: 8

#### 2. Structure Module (IPA-based)

**Purpose:** Predict 3D structure with geometric equivariance.

**Key innovation:** Invariant Point Attention (from AlphaFold2)
- Operates on both features AND 3D frames (rotation + translation)
- Ensures SE(3) equivariance (rotation/translation invariant predictions)

**Layers per iteration:**
- IPA layer (attends to both features and 3D points)
- Feed-forward network
- Backbone update (predict rotation + translation update)
- Generate C1' coordinates from current frame

**Params:**
- Iterations: 3-4 cycles with recycling
- IPA heads: 12
- Point pairs per head: 4
- Channels: 384

#### 3. Loss Functions (Multi-Objective)

**Coordinate losses:**
- FAPE (Frame Aligned Point Error) - primary loss, rotation-invariant
- Smooth L1 on coordinates (auxiliary)
- RMSD loss (auxiliary)

**Geometry regularizers:**
- Bond distance violations (consecutive C1' atoms should be ~5-7 Å)
- Clash penalty (no two atoms closer than 3 Å)
- Chirality constraints
- TM-score differentiable approximation (if possible)

**Weights:**
- FAPE: 1.0
- Bond distance: 0.5
- Clash: 0.3
- RMSD: 0.2

### Alternative: Hybrid Multi-Model Ensemble

If compute is limited or MSA processing is too slow, use a **two-tier ensemble**:

#### Tier 1: Diverse Base Models (3 models)

1. **Model A:** MSA Transformer (focus on evolutionary patterns)
2. **Model B:** Geometric GNN (focus on local geometry, graph of residue interactions)
3. **Model C:** Dilated Transformer (focus on long-range dependencies)

Train each independently with different:
- Random seeds
- Augmentations (rotation, noise injection)
- Loss weight combinations

#### Tier 2: Meta-Model (1 model)

- Input: Concatenate predictions from all 3 base models + confidence scores
- Architecture: Small MLP or ensemble averaging with learned weights
- Output: Final refined structure

**Ensemble Diversity for 5 Predictions:**
- Top 3 base model predictions
- Meta-model prediction
- Stochastic pass with dropout on best model

### Training Strategy Improvements

#### Data Augmentation

1. **Rotation augmentation:** Randomly rotate entire structures
2. **Noise injection:** Add Gaussian noise to coordinates during training
3. **MSA subsampling:** Randomly sample subset of MSA rows
4. **Coordinate dropout:** Randomly mask some residue coordinates

#### Advanced Training Techniques

1. **Exponential Moving Average (EMA):** Track EMA of model weights for stable inference
2. **Gradient clipping:** Clip to norm 1.0 to prevent instability
3. **Warm-up + cosine decay:** LR schedule with 1000-step warmup
4. **Mixed precision (FP16):** Faster training, lower memory
5. **Gradient accumulation:** Effective batch size of 64 with accumulation

#### Loss Curriculum

Start with simpler losses, gradually add complexity:

- Epochs 1-10: Coordinate loss only
- Epochs 11-30: Add bond distance regularizer
- Epochs 31+: Add all geometric constraints

### Data Utilization Improvements

#### 1. Use MSA Files

**Critical improvement:** The baseline ignores MSA data entirely.

- Parse `MSA/{target_id}.MSA.fasta` files
- Process into (M_sequences, N_residues) tensor
- Limit to top 128-256 sequences by E-value or identity
- Use in MSA Transformer module

#### 2. Template Structures

**Optional but powerful:** Use similar structures from `PDB_RNA/` as templates:

- For each target, search `pdb_seqres_NA.fasta` for homologs
- Load corresponding `.cif` file
- Extract C1' coordinates as template
- Align template to target sequence
- Feed as additional input channel

### Recommended Implementation Path

**Option A: Full Modern Architecture (Best Performance)**

Implement MSA-Enhanced Geometric Transformer with IPA layers.

**Pros:**
- State-of-the-art performance potential
- Uses all available data (MSA, sequences)
- Geometric awareness

**Cons:**
- Complex implementation (~2-3 weeks)
- High computational cost
- Requires understanding of equivariant networks

**Recommended if:** You have GPU resources and time to implement carefully.

---

**Option B: Pragmatic Hybrid (Good Performance, Faster)**

Skip full IPA implementation, use simpler improvements:

1. Add MSA Transformer module (use existing library like `fair-esm` or implement row/col attention)
2. Replace MLP head with SE(3)-Transformer final layers (use `e3nn` library)
3. Add FAPE loss
4. Train 3 models with different architectures (Transformer, GNN, CNN)
5. Ensemble predictions

**Pros:**
- Moderate complexity (~1 week)
- Significant improvement over baseline
- Can leverage existing libraries

**Cons:**
- Not as good as Option A
- Still requires understanding geometric losses

**Recommended if:** You want substantial improvement without full reimplementation.

---

**Option C: Incremental Upgrades (Moderate Performance, Fastest)**

Keep baseline architecture, upgrade training only:

1. Add MSA as additional input (concatenate MSA embeddings to sequence embeddings)
2. Upgrade to full Transformer (not BiLSTM)
3. Add geometric losses (bond distance, clash penalty)
4. Implement better ensembling (train 5 models with different seeds)
5. Add data augmentation (rotation, noise)
6. Use EMA for stable predictions

**Pros:**
- Minimal code changes (~2-3 days)
- Guaranteed improvement over baseline
- Low risk

**Cons:**
- Limited ceiling on performance
- Doesn't use geometric deep learning

**Recommended if:** You need quick wins and iterative improvement.

### Recommended Choice

**Start with Option B (Pragmatic Hybrid)**, then optionally upgrade to Option A if time permits.

### Updated Folder Structure

```text
RNA-STRUCTURE/
  PRD.md
  data/
    raw/
    processed/
      msa_processed/  # Preprocessed MSA tensors
      templates/      # Template structure features
  src/
    config.py
    modules/
      msa_module.py        # MSA Transformer
      structure_module.py  # IPA or SE(3) layers
      embeddings.py
    dataset.py
    model.py               # Main model integrating all modules
    losses.py              # FAPE, geometric losses
    train.py
    evaluate.py
    infer.py
    submit.py
    utils.py
  outputs/
    checkpoints/
    logs/
    predictions/
  requirements.txt
  README.md
```

### Key Dependencies

```text
torch>=2.0.0
numpy
pandas
biopython          # Parse FASTA, CIF files
einops             # Tensor operations
e3nn               # SE(3)-equivariant networks
fair-esm           # Pretrained MSA transformer (optional)
pytorch-lightning  # Training framework (optional but recommended)
wandb              # Experiment tracking
```

### Success Metrics

Track these to validate architectural improvements:

- **TM-score** (primary): Should improve from baseline by 0.1-0.2+
- **RMSD**: Should decrease
- **Diversity among 5 predictions**: Measure pairwise TM-score (want variety)
- **Training stability**: Loss should decrease smoothly
- **Validation gap**: Avoid overfitting

### Final Recommendation Summary

**Recommended Architecture:** MSA Transformer + Geometric Structure Module (IPA-based)  
**Recommended Path:** Option B (Pragmatic Hybrid) → Option A (Full Modern)  
**Priority Improvements:**
1. Use MSA data (biggest impact)
2. Geometric losses (FAPE, bond distances)
3. Iterative refinement (3-4 cycles)
4. Better ensemble diversity
5. Data augmentation


