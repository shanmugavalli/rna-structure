"""Data preprocessing utilities for RNA structure training."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from modules.embeddings import tokenize_msa, tokenize_sequence


PAD_TOKEN = 4


def clean_sequence(sequence: str, max_seq_len: int) -> str:
    """Normalize sequence to RNA alphabet and trim to max length."""
    seq = str(sequence).upper().replace("T", "U")
    cleaned = []
    for ch in seq:
        if ch in {"A", "C", "G", "U", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("N")
    return "".join(cleaned)[:max_seq_len]


def parse_fasta(fasta_str: str) -> List[str]:
    """Parse FASTA content into sequence strings."""
    sequences = []
    current = []
    for line in fasta_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def load_msa_sequences(
    target_id: str,
    fallback_sequence: str,
    msa_dir: Optional[str],
    max_msa_seqs: int,
) -> List[str]:
    """Load MSA rows for a target, with safe fallback to single sequence."""
    if msa_dir is None:
        return [fallback_sequence]

    msa_path = os.path.join(msa_dir, f"{target_id}.MSA.fasta")
    if not os.path.exists(msa_path):
        return [fallback_sequence]

    with open(msa_path, "r", encoding="utf-8") as f:
        msa = parse_fasta(f.read())

    if not msa:
        return [fallback_sequence]

    return msa[:max_msa_seqs]


def load_coordinates(
    label_df: Optional[pd.DataFrame],
    target_id: str,
    seq_len: int,
    max_corruption_rate: float = 0.35,
    coord_abs_threshold: float = 2000.0,
    min_valid_ratio: float = 0.70,
    max_outlier_rate: float = 0.25,
    label_groups: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Load x_1/y_1/z_1 coordinates and validity mask for a target."""
    if label_groups is not None:
        target_labels = label_groups.get(target_id, pd.DataFrame())
    elif label_df is not None:
        target_labels = label_df[label_df["ID"].str.startswith(target_id + "_")]
    else:
        target_labels = pd.DataFrame()
    if len(target_labels) == 0:
        coords = torch.zeros(seq_len, 3, dtype=torch.float32)
        mask = torch.zeros(seq_len, dtype=torch.float32)
        return coords, mask, False

    coords = []
    mask = []
    invalid = 0
    outliers = 0

    for _, row in target_labels.iterrows():
        try:
            x = float(row["x_1"])
            y = float(row["y_1"])
            z = float(row["z_1"])
            finite = np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
            in_bounds = finite and (max(abs(x), abs(y), abs(z)) <= coord_abs_threshold)
            if in_bounds:
                coords.append([x, y, z])
                mask.append(1.0)
            else:
                coords.append([
                    0.0 if not np.isfinite(x) else x,
                    0.0 if not np.isfinite(y) else y,
                    0.0 if not np.isfinite(z) else z,
                ])
                mask.append(0.0)
                invalid += 1
                if finite and not in_bounds:
                    outliers += 1
        except (TypeError, ValueError):
            coords.append([0.0, 0.0, 0.0])
            mask.append(0.0)
            invalid += 1

    total = max(1, len(coords))
    invalid_rate = invalid / total
    outlier_rate = outliers / total
    valid_ratio = 1.0 - invalid_rate
    if (
        (invalid_rate > max_corruption_rate)
        or (valid_ratio < min_valid_ratio)
        or (outlier_rate > max_outlier_rate)
    ):
        return torch.zeros(seq_len, 3, dtype=torch.float32), torch.zeros(seq_len, dtype=torch.float32), False

    coord_tensor = torch.tensor(coords, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    if len(coord_tensor) < seq_len:
        pad_n = seq_len - len(coord_tensor)
        coord_tensor = torch.cat([coord_tensor, torch.zeros(pad_n, 3, dtype=torch.float32)], dim=0)
        mask_tensor = torch.cat([mask_tensor, torch.zeros(pad_n, dtype=torch.float32)], dim=0)
    elif len(coord_tensor) > seq_len:
        coord_tensor = coord_tensor[:seq_len]
        mask_tensor = mask_tensor[:seq_len]

    return coord_tensor, mask_tensor, True


def preprocess_to_cache(
    seq_csv_path: str,
    cache_path: str,
    label_csv_path: Optional[str] = None,
    msa_dir: Optional[str] = None,
    max_msa_seqs: int = 8,
    max_seq_len: int = 256,
    max_samples: int = 0,
    use_msa: bool = True,
    max_corruption_rate: float = 0.35,
    coord_abs_threshold: float = 2000.0,
    min_valid_ratio: float = 0.70,
    max_outlier_rate: float = 0.25,
) -> int:
    """Preprocess raw CSVs into a serialized tensor cache for faster CPU training."""
    seq_df = pd.read_csv(seq_csv_path, low_memory=False)
    label_df = (
        pd.read_csv(
            label_csv_path,
            usecols=["ID", "x_1", "y_1", "z_1"],
            dtype={"ID": "string"},
            low_memory=False,
        )
        if label_csv_path
        else None
    )
    label_groups = None
    if label_df is not None:
        label_df = label_df.copy()
        label_df["target_id"] = label_df["ID"].astype(str).str.split("_", n=1).str[0]
        label_groups = {target: group for target, group in label_df.groupby("target_id", sort=False)}

    records: List[Dict[str, torch.Tensor]] = []

    iterator = seq_df.itertuples(index=False)
    for idx, row in enumerate(tqdm(iterator, total=len(seq_df), desc="Preprocessing")):
        if max_samples > 0 and idx >= max_samples:
            break

        target_id = getattr(row, "target_id")
        raw_sequence = str(getattr(row, "sequence"))
        sequence = clean_sequence(raw_sequence, max_seq_len=max_seq_len)
        seq_tokens = tokenize_sequence(sequence)

        msa_sequences = [sequence]
        if use_msa:
            msa_sequences = load_msa_sequences(target_id, sequence, msa_dir, max_msa_seqs)
        msa_tokens = tokenize_msa(msa_sequences)

        if msa_tokens.shape[0] < max_msa_seqs:
            pad_rows = max_msa_seqs - msa_tokens.shape[0]
            pad = torch.full((pad_rows, msa_tokens.shape[1]), PAD_TOKEN, dtype=torch.long)
            msa_tokens = torch.cat([msa_tokens, pad], dim=0)

        msa_tokens = msa_tokens[:, : len(sequence)]

        sample: Dict[str, torch.Tensor] = {
            "target_id": target_id,
            "seq_tokens": seq_tokens,
            "msa_tokens": msa_tokens,
            "seq_len": torch.tensor(len(sequence), dtype=torch.long),
            "raw_seq_len": torch.tensor(len(raw_sequence), dtype=torch.long),
        }

        if label_df is not None:
            coords, coord_mask, ok = load_coordinates(
                label_df,
                target_id,
                len(sequence),
                max_corruption_rate=max_corruption_rate,
                coord_abs_threshold=coord_abs_threshold,
                min_valid_ratio=min_valid_ratio,
                max_outlier_rate=max_outlier_rate,
                label_groups=label_groups,
            )
            if not ok:
                continue
            sample["coords"] = coords
            sample["coord_mask"] = coord_mask

        records.append(sample)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(records, cache_path)
    return len(records)
