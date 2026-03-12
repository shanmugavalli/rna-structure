"""Dataset and DataLoader for RNA structure prediction."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_processing import (
    clean_sequence,
    load_coordinates,
    load_msa_sequences,
    preprocess_to_cache,
)
from feature_engineering import build_residue_features
from split_analysis import build_split_analysis
from modules.embeddings import tokenize_msa, tokenize_sequence


PAD_TOKEN = 4


def apply_augmentation(coords, config):
    """Apply coordinate augmentation with safety checks."""
    batch_size, _, _ = coords.shape

    if torch.isnan(coords).any() or torch.isinf(coords).any():
        print("[WARN] Augmentation: input coords contain NaN/Inf, returning unmodified")
        return coords

    if torch.rand(1).item() < config.augment_rotation:
        try:
            for i in range(batch_size):
                angle = torch.rand(3) * 2 * np.pi
                angle = angle.float()
                rx = torch.tensor(
                    [[1, 0, 0], [0, torch.cos(angle[0]), -torch.sin(angle[0])], [0, torch.sin(angle[0]), torch.cos(angle[0])]],
                    dtype=torch.float32,
                )
                ry = torch.tensor(
                    [[torch.cos(angle[1]), 0, torch.sin(angle[1])], [0, 1, 0], [-torch.sin(angle[1]), 0, torch.cos(angle[1])]],
                    dtype=torch.float32,
                )
                rz = torch.tensor(
                    [[torch.cos(angle[2]), -torch.sin(angle[2]), 0], [torch.sin(angle[2]), torch.cos(angle[2]), 0], [0, 0, 1]],
                    dtype=torch.float32,
                )
                rotation = rz @ ry @ rx
                result = coords[i] @ rotation.T.to(coords.device)
                if not (torch.isnan(result).any() or torch.isinf(result).any()):
                    coords[i] = result
        except Exception as e:
            print(f"[WARN] Augmentation rotation error: {e}")

    if torch.rand(1).item() < config.augment_noise:
        try:
            noise = torch.randn_like(coords) * config.noise_std
            coords_with_noise = coords + noise
            if not (torch.isnan(coords_with_noise).any() or torch.isinf(coords_with_noise).any()):
                coords = coords_with_noise
        except Exception as e:
            print(f"[WARN] Augmentation noise error: {e}")

    return coords


class RNAStructureDataset(Dataset):
    """On-the-fly dataset from raw CSV files."""

    def __init__(
        self,
        seq_csv_path,
        label_csv_path=None,
        msa_dir=None,
        max_msa_seqs=128,
        max_seq_len=512,
        use_msa=True,
        max_corruption_rate=0.35,
        coord_abs_threshold=2000.0,
        min_valid_ratio=0.70,
        max_outlier_rate=0.25,
    ):
        self.seq_df = pd.read_csv(seq_csv_path, low_memory=False)
        self.label_df = (
            pd.read_csv(
                label_csv_path,
                usecols=["ID", "x_1", "y_1", "z_1"],
                dtype={"ID": "string"},
                low_memory=False,
            )
            if label_csv_path
            else None
        )
        self.msa_dir = msa_dir
        self.max_msa_seqs = max_msa_seqs
        self.max_seq_len = max_seq_len
        self.is_test = label_csv_path is None
        self.use_msa = use_msa
        self.max_corruption_rate = max_corruption_rate
        self.coord_abs_threshold = coord_abs_threshold
        self.min_valid_ratio = min_valid_ratio
        self.max_outlier_rate = max_outlier_rate
        raw_sequences = self.seq_df["sequence"].astype(str).tolist()
        self.raw_seq_lens = [len(seq) for seq in raw_sequences]
        self.seq_lens = [len(clean_sequence(seq, max_seq_len=self.max_seq_len)) for seq in raw_sequences]

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        target_id = row["target_id"]
        sequence = clean_sequence(row["sequence"], max_seq_len=self.max_seq_len)

        seq_tokens = tokenize_sequence(sequence)
        if torch.isnan(seq_tokens.float()).any():
            return None

        if self.use_msa:
            msa_seqs = load_msa_sequences(target_id, sequence, self.msa_dir, self.max_msa_seqs)
        else:
            msa_seqs = [sequence]

        msa_tokens = tokenize_msa(msa_seqs)
        if msa_tokens.shape[0] < self.max_msa_seqs:
            pad_rows = self.max_msa_seqs - msa_tokens.shape[0]
            padding = torch.full((pad_rows, msa_tokens.shape[1]), PAD_TOKEN, dtype=torch.long)
            msa_tokens = torch.cat([msa_tokens, padding], dim=0)
        msa_tokens = msa_tokens[:, : len(sequence)]

        item = {
            "target_id": target_id,
            "seq_tokens": seq_tokens,
            "msa_tokens": msa_tokens,
            "residue_features": build_residue_features(seq_tokens),
            "seq_len": len(sequence),
            "raw_seq_len": len(str(row["sequence"])),
        }

        if not self.is_test:
            coords, coord_mask, ok = load_coordinates(
                self.label_df,
                target_id,
                len(sequence),
                max_corruption_rate=self.max_corruption_rate,
                coord_abs_threshold=self.coord_abs_threshold,
                min_valid_ratio=self.min_valid_ratio,
                max_outlier_rate=self.max_outlier_rate,
            )
            if not ok:
                return None

            coords_batch = apply_augmentation(coords.unsqueeze(0), self._get_config())
            item["coords"] = coords_batch.squeeze(0)
            item["coord_mask"] = coord_mask

        return item

    def _get_config(self):
        from config import cfg

        return cfg

    def get_sequence_lengths(self, source="seq_len"):
        if source == "raw_seq_len":
            return self.raw_seq_lens
        return self.seq_lens


class RNACachedDataset(Dataset):
    """Dataset backed by serialized tensor cache."""

    def __init__(self, cache_path, augment=False):
        self.records = torch.load(cache_path, map_location="cpu")
        self.augment = augment
        self.seq_lens = [int(record["seq_len"].item()) if isinstance(record["seq_len"], torch.Tensor) else int(record["seq_len"]) for record in self.records]
        self.raw_seq_lens = [
            int(record["raw_seq_len"].item()) if isinstance(record.get("raw_seq_len", record["seq_len"]), torch.Tensor)
            else int(record.get("raw_seq_len", record["seq_len"]))
            for record in self.records
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = {
            "target_id": record["target_id"],
            "seq_tokens": record["seq_tokens"],
            "msa_tokens": record["msa_tokens"],
            "residue_features": build_residue_features(record["seq_tokens"]),
            "seq_len": int(record["seq_len"].item()) if isinstance(record["seq_len"], torch.Tensor) else int(record["seq_len"]),
            "raw_seq_len": int(record["raw_seq_len"].item()) if isinstance(record.get("raw_seq_len", record["seq_len"]), torch.Tensor) else int(record.get("raw_seq_len", record["seq_len"])),
        }

        if "coords" in record:
            coords = record["coords"].clone()
            if self.augment:
                coords = apply_augmentation(coords.unsqueeze(0), self._get_config()).squeeze(0)
            item["coords"] = coords
            item["coord_mask"] = record["coord_mask"].clone()

        return item

    def _get_config(self):
        from config import cfg

        return cfg

    def get_sequence_lengths(self, source="seq_len"):
        if source == "raw_seq_len":
            return self.raw_seq_lens
        return self.seq_lens


def _length_bucket(length, boundaries):
    for idx, boundary in enumerate(boundaries):
        if length <= boundary:
            return idx
    return len(boundaries)


def _quantile_boundaries(lengths, num_buckets):
    """Build adaptive boundaries so buckets have similar sample counts."""
    if num_buckets < 2:
        return []
    quantiles = np.linspace(0.0, 1.0, int(num_buckets) + 1)[1:-1]
    bounds = sorted({int(np.quantile(lengths, q)) for q in quantiles})
    max_len = int(max(lengths))
    return [b for b in bounds if b < max_len]


def _build_length_stratified_sampler(dataset, boundaries, sampling_power=1.0, strategy="fixed", num_buckets=5, length_source="seq_len"):
    """Create weighted sampler to upsample underrepresented length buckets."""
    lengths = dataset.get_sequence_lengths(source=length_source) if hasattr(dataset, "get_sequence_lengths") else []
    if not lengths:
        return None, {}

    if strategy == "quantile":
        boundaries = _quantile_boundaries(lengths, num_buckets=num_buckets)
    else:
        max_len = int(max(lengths))
        boundaries = [int(b) for b in boundaries if int(b) < max_len]

    bucket_ids = [_length_bucket(length, boundaries) for length in lengths]
    n_buckets = len(boundaries) + 1
    counts = np.bincount(bucket_ids, minlength=n_buckets)

    weights = []
    for bucket_id in bucket_ids:
        bucket_count = max(1, counts[bucket_id])
        weights.append((1.0 / float(bucket_count)) ** float(sampling_power))

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    stats = {
        "length_source": length_source,
        "strategy": strategy,
        "boundaries": boundaries,
        "counts": counts.tolist(),
        "total": len(lengths),
    }
    return sampler, stats


def collate_fn(batch):
    """Collate variable-length samples and skip invalid rows."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    max_len = max(item["seq_len"] for item in batch)

    seq_tokens = []
    msa_tokens = []
    residue_features = []
    coords = []
    coord_masks = []
    target_ids = []

    for item in batch:
        seq = item["seq_tokens"]
        msa = item["msa_tokens"]
        feats = item["residue_features"]

        if len(seq) < max_len:
            seq = torch.cat([seq, torch.full((max_len - len(seq),), PAD_TOKEN, dtype=torch.long)])

        if msa.shape[1] < max_len:
            msa_pad = torch.full((msa.shape[0], max_len - msa.shape[1]), PAD_TOKEN, dtype=torch.long)
            msa = torch.cat([msa, msa_pad], dim=1)

        if feats.shape[0] < max_len:
            feat_pad = torch.zeros(max_len - feats.shape[0], feats.shape[1], dtype=torch.float32)
            feats = torch.cat([feats, feat_pad], dim=0)

        seq_tokens.append(seq)
        msa_tokens.append(msa)
        residue_features.append(feats)

        if "coords" in item:
            coord = item["coords"]
            coord_mask = item["coord_mask"]
            if len(coord) < max_len:
                coord = torch.cat([coord, torch.zeros(max_len - len(coord), 3)], dim=0)
                coord_mask = torch.cat([coord_mask, torch.zeros(max_len - len(coord_mask))], dim=0)
            coords.append(coord)
            coord_masks.append(coord_mask)

        target_ids.append(item["target_id"])

    batch_dict = {
        "target_ids": target_ids,
        "seq_tokens": torch.stack(seq_tokens),
        "msa_tokens": torch.stack(msa_tokens),
        "residue_features": torch.stack(residue_features),
    }

    if coords:
        batch_dict["coords"] = torch.stack(coords)
        batch_dict["coord_mask"] = torch.stack(coord_masks)

    return batch_dict


def _maybe_build_cache(config, split="train"):
    """Build serialized cache lazily to speed up repeated CPU runs."""
    cache_dir = getattr(config, "cache_dir", "data/processed")
    os.makedirs(cache_dir, exist_ok=True)

    if split == "train":
        seq_path = config.train_seq_path
        label_path = config.train_label_path
        max_samples = getattr(config, "cpu_train_max_samples", 0)
    else:
        seq_path = config.val_seq_path
        label_path = config.val_label_path
        max_samples = getattr(config, "cpu_val_max_samples", 0)

    cache_path = os.path.join(
        cache_dir,
        f"{split}_cache_len{int(config.max_seq_length)}_msa{int(config.max_msa_seqs)}_"
        f"corr{int(getattr(config, 'max_corruption_rate', 0.35) * 100)}.pt",
    )
    if os.path.exists(cache_path):
        return cache_path

    use_msa = getattr(config, "use_msa", True)
    n = preprocess_to_cache(
        seq_csv_path=seq_path,
        label_csv_path=label_path,
        cache_path=cache_path,
        msa_dir=config.msa_dir,
        max_msa_seqs=config.max_msa_seqs,
        max_seq_len=config.max_seq_length,
        max_samples=max_samples,
        use_msa=use_msa,
        max_corruption_rate=getattr(config, "max_corruption_rate", 0.35),
        coord_abs_threshold=getattr(config, "coord_abs_threshold", 2000.0),
        min_valid_ratio=getattr(config, "min_valid_ratio", 0.70),
        max_outlier_rate=getattr(config, "max_outlier_rate", 0.25),
    )
    print(f"[INFO] Built {split} cache with {n} samples at {cache_path}")
    return cache_path


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    use_cached = getattr(config, "use_cached_dataset", False)

    if use_cached:
        train_cache = _maybe_build_cache(config, split="train")
        val_cache = _maybe_build_cache(config, split="val")
        train_dataset = RNACachedDataset(train_cache, augment=True)
        val_dataset = RNACachedDataset(val_cache, augment=False)
    else:
        train_dataset = RNAStructureDataset(
            seq_csv_path=config.train_seq_path,
            label_csv_path=config.train_label_path,
            msa_dir=config.msa_dir,
            max_msa_seqs=config.max_msa_seqs,
            max_seq_len=config.max_seq_length,
            use_msa=getattr(config, "use_msa", True),
            max_corruption_rate=getattr(config, "max_corruption_rate", 0.35),
            coord_abs_threshold=getattr(config, "coord_abs_threshold", 2000.0),
            min_valid_ratio=getattr(config, "min_valid_ratio", 0.70),
            max_outlier_rate=getattr(config, "max_outlier_rate", 0.25),
        )
        val_dataset = RNAStructureDataset(
            seq_csv_path=config.val_seq_path,
            label_csv_path=config.val_label_path,
            msa_dir=config.msa_dir,
            max_msa_seqs=config.max_msa_seqs,
            max_seq_len=config.max_seq_length,
            use_msa=getattr(config, "use_msa", True),
            max_corruption_rate=getattr(config, "max_corruption_rate", 0.35),
            coord_abs_threshold=getattr(config, "coord_abs_threshold", 2000.0),
            min_valid_ratio=getattr(config, "min_valid_ratio", 0.70),
            max_outlier_rate=getattr(config, "max_outlier_rate", 0.25),
        )

    train_sampler = None
    if getattr(config, "use_length_stratified_sampling", False):
        boundaries = getattr(config, "length_bucket_boundaries", [64, 96, 128, 160, 192, 256, 320])
        sampling_power = getattr(config, "length_sampling_power", 1.0)
        strategy = getattr(config, "length_bucket_strategy", "fixed")
        num_buckets = getattr(config, "length_num_buckets", 5)
        length_source = getattr(config, "length_bucket_source", "seq_len")
        train_sampler, stats = _build_length_stratified_sampler(
            train_dataset,
            boundaries,
            sampling_power=sampling_power,
            strategy=strategy,
            num_buckets=num_buckets,
            length_source=length_source,
        )
        if stats:
            print(
                f"[INFO] Length stratified sampling enabled. "
                f"source={stats['length_source']} strategy={stats['strategy']} "
                f"boundaries={stats['boundaries']} counts={stats['counts']} total={stats['total']}"
            )
            if getattr(config, "generate_split_analysis", False):
                try:
                    out_dir = getattr(config, "analysis_dir", "outputs/analysis")
                    train_summary = build_split_analysis(
                        dataset=train_dataset,
                        boundaries=stats["boundaries"],
                        output_dir=out_dir,
                        split_name="train",
                        strategy=stats["strategy"],
                        length_source=stats["length_source"],
                    )
                    val_summary = build_split_analysis(
                        dataset=val_dataset,
                        boundaries=stats["boundaries"],
                        output_dir=out_dir,
                        split_name="val",
                        strategy=stats["strategy"],
                        length_source=stats["length_source"],
                    )
                    print(f"[INFO] Split analysis written: train={train_summary} val={val_summary}")
                except Exception as exc:
                    print(f"[WARN] Could not generate split analysis plots: {exc}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
