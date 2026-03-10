"""
Dataset and DataLoader for RNA structure prediction
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from modules.embeddings import tokenize_sequence, tokenize_msa


def apply_augmentation(coords, config):
    """Apply data augmentation to coordinates with safety checks"""
    batch_size, seq_len, _ = coords.shape
    
    # Validate input
    if torch.isnan(coords).any() or torch.isinf(coords).any():
        print("[WARN] Augmentation: input coords contain NaN/Inf, returning unmodified")
        return coords
    
    # Random rotation
    if torch.rand(1).item() < config.augment_rotation:
        try:
            for i in range(batch_size):
                angle = torch.rand(3) * 2 * np.pi
                angle = angle.float()
                Rx = torch.tensor([[1, 0, 0],
                                  [0, torch.cos(angle[0]), -torch.sin(angle[0])],
                                  [0, torch.sin(angle[0]), torch.cos(angle[0])]], dtype=torch.float32)
                Ry = torch.tensor([[torch.cos(angle[1]), 0, torch.sin(angle[1])],
                                  [0, 1, 0],
                                  [-torch.sin(angle[1]), 0, torch.cos(angle[1])]], dtype=torch.float32)
                Rz = torch.tensor([[torch.cos(angle[2]), -torch.sin(angle[2]), 0],
                                  [torch.sin(angle[2]), torch.cos(angle[2]), 0],
                                  [0, 0, 1]], dtype=torch.float32)
                R = Rz @ Ry @ Rx
                result = coords[i] @ R.T.to(coords.device)
                
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"[WARN] Augmentation: rotation produced NaN/Inf, skipping for batch {i}")
                else:
                    coords[i] = result
        except Exception as e:
            print(f"[WARN] Augmentation rotation error: {e}")
    
    # Gaussian noise
    if torch.rand(1).item() < config.augment_noise:
        try:
            noise = torch.randn_like(coords) * config.noise_std
            coords_with_noise = coords + noise
            
            if torch.isnan(coords_with_noise).any() or torch.isinf(coords_with_noise).any():
                print("[WARN] Augmentation: noise produced NaN/Inf, returning coords without noise")
            else:
                coords = coords_with_noise
        except Exception as e:
            print(f"[WARN] Augmentation noise error: {e}")
    
    return coords


def parse_fasta(fasta_str):
    """Parse FASTA format string into list of sequences"""
    sequences = []
    current_seq = []
    
    for line in fasta_str.strip().split('\n'):
        if line.startswith('>'):
            if current_seq:
                sequences.append(''.join(current_seq))
                current_seq = []
        else:
            current_seq.append(line.strip())
    
    if current_seq:
        sequences.append(''.join(current_seq))
    
    return sequences


class RNAStructureDataset(Dataset):
    """Dataset for RNA structure prediction"""
    
    def __init__(self, seq_csv_path, label_csv_path=None, msa_dir=None, 
                 max_msa_seqs=128, max_seq_len=512):
        """
        Args:
            seq_csv_path: Path to sequences CSV
            label_csv_path: Path to labels CSV (None for test set)
            msa_dir: Directory containing MSA files
            max_msa_seqs: Maximum number of MSA sequences to use
            max_seq_len: Maximum sequence length
        """
        self.seq_df = pd.read_csv(seq_csv_path)
        self.label_df = pd.read_csv(label_csv_path) if label_csv_path else None
        self.msa_dir = msa_dir
        self.max_msa_seqs = max_msa_seqs
        self.max_seq_len = max_seq_len
        self.is_test = (label_csv_path is None)
        
    def __len__(self):
        return len(self.seq_df)
    
    def __getitem__(self, idx):
        """Get single RNA target"""
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        sequence = row['sequence']
        
        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        
        # Tokenize sequence
        seq_tokens = tokenize_sequence(sequence)
        
        # Validate tokens
        if torch.isnan(seq_tokens.float()).any():
            print(f"[ERROR] Dataset idx {idx} ({target_id}): seq_tokens contains NaN")
            raise ValueError(f"Corrupted sequence tokens for {target_id}")
        
        # Load MSA if available
        msa_tokens = self._load_msa(target_id, len(sequence))
        
        # Validate MSA
        if torch.isnan(msa_tokens.float()).any():
            print(f"[ERROR] Dataset idx {idx} ({target_id}): msa_tokens contains NaN")
            raise ValueError(f"Corrupted MSA tokens for {target_id}")
        
        item = {
            'target_id': target_id,
            'seq_tokens': seq_tokens,
            'msa_tokens': msa_tokens,
            'seq_len': len(sequence)
        }
        
        # Load labels if available (training/validation)
        if not self.is_test:
            coords = self._load_coordinates(target_id, len(sequence))
            
            # Validate coordinates
            if torch.isnan(coords).any():
                print(f"[ERROR] Dataset idx {idx} ({target_id}): coords contains NaN")
                print(f"[DEBUG] Coords shape: {coords.shape}, non-finite count: {(~torch.isfinite(coords)).sum().item()}")
                raise ValueError(f"Corrupted coordinates for {target_id}")
            
            if torch.isinf(coords).any():
                print(f"[ERROR] Dataset idx {idx} ({target_id}): coords contains Inf")
                raise ValueError(f"Coordinates contain Inf for {target_id}")
            
            # Apply augmentation (add batch dimension for augmentation function)
            coords_batch = coords.unsqueeze(0)  # (1, seq_len, 3)
            coords_batch = apply_augmentation(coords_batch, self._get_config())
            coords = coords_batch.squeeze(0)
            
            item['coords'] = coords
        
        return item
    
    def _get_config(self):
        """Return a minimal config object for augmentation"""
        from config import cfg
        return cfg
    
    def _load_msa(self, target_id, seq_len):
        """Load and process MSA for target"""
        if self.msa_dir is None:
            # No MSA available - use only target sequence
            msa_seqs = [self.seq_df[self.seq_df['target_id'] == target_id]['sequence'].iloc[0]]
        else:
            msa_path = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")
            
            if os.path.exists(msa_path):
                with open(msa_path, 'r') as f:
                    fasta_str = f.read()
                msa_seqs = parse_fasta(fasta_str)
                
                # Limit number of sequences
                if len(msa_seqs) > self.max_msa_seqs:
                    msa_seqs = msa_seqs[:self.max_msa_seqs]
            else:
                # Fallback: use only target sequence
                msa_seqs = [self.seq_df[self.seq_df['target_id'] == target_id]['sequence'].iloc[0]]
        
        # Tokenize MSA
        msa_tokens = tokenize_msa(msa_seqs)
        
        # Pad MSA depth if needed
        if msa_tokens.shape[0] < self.max_msa_seqs:
            padding = torch.full((self.max_msa_seqs - msa_tokens.shape[0], msa_tokens.shape[1]), 
                                4, dtype=torch.long)  # Pad token = 4
            msa_tokens = torch.cat([msa_tokens, padding], dim=0)
        
        # Truncate MSA length to match sequence
        msa_tokens = msa_tokens[:, :seq_len]
        
        return msa_tokens
    
    def _load_coordinates(self, target_id, seq_len):
        """Load ground truth C1' coordinates"""
        # Filter labels for this target
        target_labels = self.label_df[self.label_df['ID'].str.startswith(target_id + '_')]
        
        if len(target_labels) == 0:
            # No labels found - return zeros (shouldn't happen in training)
            return torch.zeros(seq_len, 3)
        
        # Extract first structure (x_1, y_1, z_1)
        coords = []
        nan_count = 0
        for row_idx, (_, row) in enumerate(target_labels.iterrows()):
            try:
                x = float(row['x_1'])
                y = float(row['y_1'])
                z = float(row['z_1'])
                
                # Check for NaN/Inf in individual values
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    nan_count += 1
                    # Replace with zero for corrupted positions
                    x = 0.0 if not np.isfinite(x) else x
                    y = 0.0 if not np.isfinite(y) else y
                    z = 0.0 if not np.isfinite(z) else z
                
                coords.append([x, y, z])
            except (ValueError, TypeError) as e:
                print(f"[WARN] {target_id} row {row_idx}: error parsing coords - {e}")
                coords.append([0.0, 0.0, 0.0])
                nan_count += 1
        
        if nan_count > 0:
            print(f"[WARN] {target_id}: {nan_count}/{len(target_labels)} positions had NaN/Inf, replaced with 0")
        
        coords = torch.tensor(coords, dtype=torch.float32)
        
        # Pad or truncate to seq_len
        if len(coords) < seq_len:
            padding = torch.zeros(seq_len - len(coords), 3)
            coords = torch.cat([coords, padding], dim=0)
        elif len(coords) > seq_len:
            coords = coords[:seq_len]
        
        return coords


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Find max sequence length in batch
    max_len = max(item['seq_len'] for item in batch)
    
    # Pad all sequences to max_len
    seq_tokens = []
    msa_tokens = []
    coords = []
    target_ids = []
    
    for item in batch:
        seq_len = item['seq_len']
        
        # Pad sequence tokens
        seq = item['seq_tokens']
        if len(seq) < max_len:
            seq = torch.cat([seq, torch.full((max_len - len(seq),), 4, dtype=torch.long)])
        seq_tokens.append(seq)
        
        # Pad MSA tokens
        msa = item['msa_tokens']
        if msa.shape[1] < max_len:
            padding = torch.full((msa.shape[0], max_len - msa.shape[1]), 4, dtype=torch.long)
            msa = torch.cat([msa, padding], dim=1)
        msa_tokens.append(msa)
        
        # Pad coordinates if available
        if 'coords' in item:
            coord = item['coords']
            if len(coord) < max_len:
                coord = torch.cat([coord, torch.zeros(max_len - len(coord), 3)])
            coords.append(coord)
        
        target_ids.append(item['target_id'])
    
    batch_dict = {
        'target_ids': target_ids,
        'seq_tokens': torch.stack(seq_tokens),
        'msa_tokens': torch.stack(msa_tokens),
    }
    
    if coords:
        batch_dict['coords'] = torch.stack(coords)
    
    return batch_dict


def create_dataloaders(config):
    """Create train and validation dataloaders"""
    train_dataset = RNAStructureDataset(
        seq_csv_path=config.train_seq_path,
        label_csv_path=config.train_label_path,
        msa_dir=config.msa_dir,
        max_msa_seqs=config.max_msa_seqs,
        max_seq_len=config.max_seq_length
    )
    
    val_dataset = RNAStructureDataset(
        seq_csv_path=config.val_seq_path,
        label_csv_path=config.val_label_path,
        msa_dir=config.msa_dir,
        max_msa_seqs=config.max_msa_seqs,
        max_seq_len=config.max_seq_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
