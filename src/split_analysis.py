"""Visualization and summary utilities for length-based split analysis."""
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_processing import clean_sequence, load_coordinates


def _length_bucket(length: int, boundaries: List[int]) -> int:
    for idx, boundary in enumerate(boundaries):
        if int(length) <= int(boundary):
            return idx
    return len(boundaries)


def _bucket_labels(boundaries: List[int]) -> List[str]:
    labels = []
    lower = 1
    for boundary in boundaries:
        labels.append(f"{lower}-{int(boundary)}")
        lower = int(boundary) + 1
    labels.append(f">{boundaries[-1]}" if boundaries else "all")
    return labels


def _extract_records_from_cached(dataset, length_source: str):
    records = []
    for record in dataset.records:
        seq_len = int(record["seq_len"].item()) if isinstance(record["seq_len"], torch.Tensor) else int(record["seq_len"])
        raw_len_val = record.get("raw_seq_len", record["seq_len"])
        raw_seq_len = int(raw_len_val.item()) if isinstance(raw_len_val, torch.Tensor) else int(raw_len_val)

        valid_ratio = np.nan
        coord_norm_mean = np.nan
        if "coord_mask" in record and "coords" in record:
            mask = record["coord_mask"].float().cpu().numpy()
            coords = record["coords"].float().cpu().numpy()
            valid = mask > 0.5
            valid_ratio = float(mask.mean()) if mask.size > 0 else np.nan
            if valid.any():
                coord_norm_mean = float(np.linalg.norm(coords[valid], axis=1).mean())

        length_value = raw_seq_len if length_source == "raw_seq_len" else seq_len
        records.append(
            {
                "seq_len": seq_len,
                "raw_seq_len": raw_seq_len,
                "length_value": int(length_value),
                "valid_ratio": valid_ratio,
                "coord_norm_mean": coord_norm_mean,
            }
        )
    return records


def _extract_records_from_raw(dataset, length_source: str):
    records = []
    label_df = getattr(dataset, "label_df", None)
    for _, row in dataset.seq_df.iterrows():
        raw_seq = str(row["sequence"])
        seq = clean_sequence(raw_seq, dataset.max_seq_len)
        seq_len = len(seq)
        raw_len = len(raw_seq)

        valid_ratio = np.nan
        coord_norm_mean = np.nan
        if label_df is not None:
            coords, mask, ok = load_coordinates(label_df, row["target_id"], seq_len)
            if ok:
                mask_np = mask.numpy()
                coords_np = coords.numpy()
                valid = mask_np > 0.5
                valid_ratio = float(mask_np.mean()) if mask_np.size > 0 else np.nan
                if valid.any():
                    coord_norm_mean = float(np.linalg.norm(coords_np[valid], axis=1).mean())

        length_value = raw_len if length_source == "raw_seq_len" else seq_len
        records.append(
            {
                "seq_len": seq_len,
                "raw_seq_len": raw_len,
                "length_value": int(length_value),
                "valid_ratio": valid_ratio,
                "coord_norm_mean": coord_norm_mean,
            }
        )
    return records


def _collect_records(dataset, length_source: str):
    if hasattr(dataset, "records"):
        return _extract_records_from_cached(dataset, length_source)
    return _extract_records_from_raw(dataset, length_source)


def build_split_analysis(dataset, boundaries: List[int], output_dir: str, split_name: str, strategy: str, length_source: str):
    """Create bucket-level count and label-distribution plots plus CSV summary."""
    os.makedirs(output_dir, exist_ok=True)
    rows = _collect_records(dataset, length_source=length_source)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["bucket_id"] = df["length_value"].apply(lambda x: _length_bucket(int(x), boundaries))

    labels = _bucket_labels(boundaries)
    n_buckets = len(labels)

    summary_rows = []
    for bucket_id in range(n_buckets):
        sub = df[df["bucket_id"] == bucket_id]
        summary_rows.append(
            {
                "bucket_id": bucket_id,
                "bucket_label": labels[bucket_id],
                "count": int(len(sub)),
                "mean_seq_len": float(sub["seq_len"].mean()) if len(sub) else np.nan,
                "mean_raw_seq_len": float(sub["raw_seq_len"].mean()) if len(sub) else np.nan,
                "mean_valid_ratio": float(sub["valid_ratio"].mean()) if len(sub) else np.nan,
                "mean_coord_norm": float(sub["coord_norm_mean"].mean()) if len(sub) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"{split_name}_bucket_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # 1) Sample count per bucket
    plt.figure(figsize=(10, 4))
    plt.bar(summary_df["bucket_label"], summary_df["count"], color="#2c7fb8")
    plt.title(f"{split_name} | Samples per bucket ({strategy}, {length_source})")
    plt.xlabel("Length bucket")
    plt.ylabel("Sample count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    count_plot = os.path.join(output_dir, f"{split_name}_bucket_counts.png")
    plt.savefig(count_plot, dpi=150)
    plt.close()

    # 2) Label validity ratio distribution per bucket
    validity_data = [df[df["bucket_id"] == i]["valid_ratio"].dropna().values for i in range(n_buckets)]
    if any(len(v) > 0 for v in validity_data):
        plt.figure(figsize=(10, 4))
        plt.boxplot(validity_data, labels=labels, showfliers=False)
        plt.title(f"{split_name} | Label validity ratio by bucket")
        plt.xlabel("Length bucket")
        plt.ylabel("Valid label ratio")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        validity_plot = os.path.join(output_dir, f"{split_name}_label_validity_boxplot.png")
        plt.savefig(validity_plot, dpi=150)
        plt.close()

    # 3) Coordinate norm distribution per bucket
    coord_data = [df[df["bucket_id"] == i]["coord_norm_mean"].dropna().values for i in range(n_buckets)]
    if any(len(v) > 0 for v in coord_data):
        plt.figure(figsize=(10, 4))
        plt.boxplot(coord_data, labels=labels, showfliers=False)
        plt.title(f"{split_name} | Coordinate norm by bucket")
        plt.xlabel("Length bucket")
        plt.ylabel("Mean ||coord||")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        coord_plot = os.path.join(output_dir, f"{split_name}_coord_norm_boxplot.png")
        plt.savefig(coord_plot, dpi=150)
        plt.close()

    return summary_path
