"""RNA domain-focused dataset analytics and reporting.

Generates summary CSVs and plots for:
- sequence-label consistency
- MSA quality
- geometry sanity
- train/validation drift
- noisy-target ranking
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class AnalyticsConfig:
    train_seq_path: str
    val_seq_path: str
    train_label_path: str
    val_label_path: str
    msa_dir: str
    output_dir: str
    max_targets: int = 0
    coord_abs_threshold: float = 2_000.0
    bond_min: float = 4.0
    bond_max: float = 8.0
    clash_distance: float = 3.0
    clash_subsample: int = 200


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_sequences(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["target_id", "sequence"], low_memory=False)
    df["sequence"] = df["sequence"].astype(str)
    df["seq_len"] = df["sequence"].str.len()
    gc = df["sequence"].str.count("G") + df["sequence"].str.count("C")
    df["gc_frac"] = gc / df["seq_len"].clip(lower=1)
    return df


def _load_labels(path: str, target_filter: Iterable[str] | None = None) -> pd.DataFrame:
    usecols = ["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"]
    if target_filter is None:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["target_id"] = df["ID"].astype(str).str.split("_", n=1).str[0]
    else:
        target_set = set(target_filter)
        chunks = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
            tgt = chunk["ID"].astype(str).str.split("_", n=1).str[0]
            mask = tgt.isin(target_set)
            if mask.any():
                sub = chunk.loc[mask].copy()
                sub["target_id"] = tgt.loc[mask].values
                chunks.append(sub)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=usecols + ["target_id"])

    df["resid"] = pd.to_numeric(df["resid"], errors="coerce")
    for c in ["x_1", "y_1", "z_1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pick_targets(train_seq: pd.DataFrame, val_seq: pd.DataFrame, max_targets: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if max_targets <= 0:
        return train_seq, val_seq
    t = train_seq.head(max_targets).copy()
    v = val_seq.head(max(1, max_targets // 5)).copy()
    return t, v


def _sequence_label_consistency(seq_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    seq_map = dict(zip(seq_df["target_id"], seq_df["sequence"]))
    records = []
    for target_id, grp in labels.groupby("target_id", sort=False):
        seq = seq_map.get(target_id, "")
        if not seq:
            continue

        resid = grp["resid"].to_numpy()
        valid_resid_mask = np.isfinite(resid) & (resid >= 1) & (resid <= len(seq))
        valid_resid = resid[valid_resid_mask].astype(np.int64)

        expected = np.array([seq[i - 1] for i in valid_resid], dtype=object)
        observed = grp.loc[valid_resid_mask, "resname"].astype(str).str.upper().to_numpy()
        match = expected == observed

        total = len(grp)
        records.append(
            {
                "target_id": target_id,
                "rows": total,
                "valid_resid_rate": float(valid_resid_mask.mean()) if total else np.nan,
                "resname_match_rate": float(match.mean()) if len(match) else np.nan,
                "resname_mismatch_rate": float(1.0 - match.mean()) if len(match) else np.nan,
            }
        )

    return pd.DataFrame(records)


def _pairwise_clash_rate(coords: np.ndarray, thresh: float, subsample: int) -> float:
    n = coords.shape[0]
    if n < 4:
        return np.nan
    if n > subsample:
        idx = np.linspace(0, n - 1, num=subsample, dtype=int)
        coords = coords[idx]
        n = coords.shape[0]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=2)
    d = dist[iu]
    if d.size == 0:
        return np.nan
    return float((d < thresh).mean())


def _geometry_sanity(labels: pd.DataFrame, cfg: AnalyticsConfig) -> pd.DataFrame:
    rows = []
    for target_id, grp in labels.groupby("target_id", sort=False):
        xyz = grp[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(xyz).all(axis=1)
        outlier_mask = np.abs(xyz).max(axis=1) > cfg.coord_abs_threshold
        invalid_mask = (~finite_mask) | outlier_mask

        bond_viol = []
        for _, cgrp in grp.groupby(["chain", "copy"], sort=False):
            cgrp = cgrp.sort_values("resid")
            cxyz = cgrp[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float64)
            cres = cgrp["resid"].to_numpy(dtype=np.float64)
            valid = np.isfinite(cxyz).all(axis=1)
            for i in range(len(cgrp) - 1):
                if not (valid[i] and valid[i + 1]):
                    continue
                if not (cres[i + 1] - cres[i] == 1):
                    continue
                d = float(np.linalg.norm(cxyz[i + 1] - cxyz[i]))
                bond_viol.append((d < cfg.bond_min) or (d > cfg.bond_max))

        valid_xyz = xyz[~invalid_mask]
        clash_rate = _pairwise_clash_rate(valid_xyz, cfg.clash_distance, cfg.clash_subsample)

        rows.append(
            {
                "target_id": target_id,
                "rows": int(len(grp)),
                "invalid_coord_rate": float(invalid_mask.mean()) if len(grp) else np.nan,
                "outlier_coord_rate": float(outlier_mask.mean()) if len(grp) else np.nan,
                "bond_violation_rate": float(np.mean(bond_viol)) if len(bond_viol) else np.nan,
                "clash_rate": clash_rate,
            }
        )

    return pd.DataFrame(rows)


def _msa_stats(target_ids: Iterable[str], msa_dir: str) -> pd.DataFrame:
    out = []
    for target_id in target_ids:
        path = os.path.join(msa_dir, f"{target_id}.MSA.fasta")
        if not os.path.exists(path):
            out.append({"target_id": target_id, "msa_depth": 0, "msa_gap_frac": np.nan, "msa_query_coverage": np.nan})
            continue

        seqs = []
        cur = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line)
            if cur:
                seqs.append("".join(cur))

        if not seqs:
            out.append({"target_id": target_id, "msa_depth": 0, "msa_gap_frac": np.nan, "msa_query_coverage": np.nan})
            continue

        lens = [len(s) for s in seqs]
        L = max(1, max(lens))
        padded = [s + ("-" * (L - len(s))) for s in seqs]
        arr = np.array([list(s) for s in padded], dtype="U1")

        gap_frac = float((arr == "-").mean())
        query = arr[0]
        coverage = float((query != "-").mean())

        out.append(
            {
                "target_id": target_id,
                "msa_depth": int(len(seqs)),
                "msa_gap_frac": gap_frac,
                "msa_query_coverage": coverage,
            }
        )

    return pd.DataFrame(out)


def _summarize_split(name: str, df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "seq_len",
        "gc_frac",
        "resname_mismatch_rate",
        "invalid_coord_rate",
        "outlier_coord_rate",
        "bond_violation_rate",
        "clash_rate",
        "msa_depth",
        "msa_gap_frac",
    ]
    rows = []
    for m in metrics:
        x = pd.to_numeric(df[m], errors="coerce").dropna()
        rows.append(
            {
                "split": name,
                "metric": m,
                "count": int(x.shape[0]),
                "mean": float(x.mean()) if len(x) else np.nan,
                "std": float(x.std()) if len(x) else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "p90": float(x.quantile(0.9)) if len(x) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _drift_report(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "seq_len",
        "gc_frac",
        "resname_mismatch_rate",
        "invalid_coord_rate",
        "outlier_coord_rate",
        "bond_violation_rate",
        "clash_rate",
        "msa_depth",
        "msa_gap_frac",
    ]
    rows = []
    for m in metrics:
        a = pd.to_numeric(train[m], errors="coerce").dropna()
        b = pd.to_numeric(val[m], errors="coerce").dropna()
        if len(a) == 0 or len(b) == 0:
            rows.append({"metric": m, "train_mean": np.nan, "val_mean": np.nan, "smd": np.nan})
            continue
        denom = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        smd = float((a.mean() - b.mean()) / denom) if denom > 1e-12 else 0.0
        rows.append({"metric": m, "train_mean": float(a.mean()), "val_mean": float(b.mean()), "smd": smd})
    return pd.DataFrame(rows).sort_values("smd", key=lambda s: np.abs(s), ascending=False)


def _noisy_rank(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for c in ["resname_mismatch_rate", "invalid_coord_rate", "outlier_coord_rate", "bond_violation_rate", "clash_rate"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    work["noise_score"] = (
        0.20 * work["resname_mismatch_rate"]
        + 0.35 * work["invalid_coord_rate"]
        + 0.25 * work["outlier_coord_rate"]
        + 0.15 * work["bond_violation_rate"]
        + 0.05 * work["clash_rate"]
    )
    return work.sort_values("noise_score", ascending=False)


def _plot_basic(train: pd.DataFrame, val: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)

    plt.figure(figsize=(8, 4))
    plt.hist(train["seq_len"].dropna(), bins=40, alpha=0.6, label="train")
    plt.hist(val["seq_len"].dropna(), bins=40, alpha=0.6, label="val")
    plt.title("Sequence Length Distribution")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "split_length_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(train["gc_frac"].dropna(), bins=30, alpha=0.6, label="train")
    plt.hist(val["gc_frac"].dropna(), bins=30, alpha=0.6, label="val")
    plt.title("GC Fraction Distribution")
    plt.xlabel("GC fraction")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "split_gc_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = pd.to_numeric(train["msa_depth"], errors="coerce")
    y = pd.to_numeric(train["invalid_coord_rate"], errors="coerce")
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xscale("symlog")
    plt.title("Train: MSA depth vs invalid label rate")
    plt.xlabel("MSA depth")
    plt.ylabel("invalid_coord_rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_msa_depth_vs_invalid_rate.png"), dpi=150)
    plt.close()


def run_analytics(cfg: AnalyticsConfig) -> None:
    _ensure_dir(cfg.output_dir)

    train_seq = _load_sequences(cfg.train_seq_path)
    val_seq = _load_sequences(cfg.val_seq_path)
    train_seq, val_seq = _pick_targets(train_seq, val_seq, cfg.max_targets)

    train_labels = _load_labels(cfg.train_label_path, target_filter=train_seq["target_id"].tolist())
    val_labels = _load_labels(cfg.val_label_path, target_filter=val_seq["target_id"].tolist())

    train_cons = _sequence_label_consistency(train_seq, train_labels)
    val_cons = _sequence_label_consistency(val_seq, val_labels)

    train_geo = _geometry_sanity(train_labels, cfg)
    val_geo = _geometry_sanity(val_labels, cfg)

    train_msa = _msa_stats(train_seq["target_id"].tolist(), cfg.msa_dir)
    val_msa = _msa_stats(val_seq["target_id"].tolist(), cfg.msa_dir)

    train_target = train_seq[["target_id", "seq_len", "gc_frac"]].merge(train_cons, on="target_id", how="left")
    train_target = train_target.merge(train_geo, on="target_id", how="left")
    train_target = train_target.merge(train_msa, on="target_id", how="left")

    val_target = val_seq[["target_id", "seq_len", "gc_frac"]].merge(val_cons, on="target_id", how="left")
    val_target = val_target.merge(val_geo, on="target_id", how="left")
    val_target = val_target.merge(val_msa, on="target_id", how="left")

    train_target.to_csv(os.path.join(cfg.output_dir, "train_target_metrics.csv"), index=False)
    val_target.to_csv(os.path.join(cfg.output_dir, "val_target_metrics.csv"), index=False)

    train_summary = _summarize_split("train", train_target)
    val_summary = _summarize_split("val", val_target)
    split_summary = pd.concat([train_summary, val_summary], ignore_index=True)
    split_summary.to_csv(os.path.join(cfg.output_dir, "split_metric_summary.csv"), index=False)

    drift = _drift_report(train_target, val_target)
    drift.to_csv(os.path.join(cfg.output_dir, "split_drift_report.csv"), index=False)

    noisy = _noisy_rank(train_target)
    noisy.to_csv(os.path.join(cfg.output_dir, "train_noisy_targets_ranked.csv"), index=False)

    _plot_basic(train_target, val_target, cfg.output_dir)

    topn = noisy.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(topn["target_id"].astype(str), topn["noise_score"])
    plt.gca().invert_yaxis()
    plt.title("Top noisy training targets")
    plt.xlabel("noise score")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "top_noisy_targets.png"), dpi=150)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RNA dataset analytics")
    p.add_argument("--train-seq", default="data/raw/train_sequences.csv")
    p.add_argument("--val-seq", default="data/raw/validation_sequences.csv")
    p.add_argument("--train-label", default="data/raw/train_labels.csv")
    p.add_argument("--val-label", default="data/raw/validation_labels.csv")
    p.add_argument("--msa-dir", default="data/raw/MSA")
    p.add_argument("--out", default="outputs/analysis")
    p.add_argument("--max-targets", type=int, default=0, help="0 means all targets")
    p.add_argument("--coord-abs-threshold", type=float, default=2000.0)
    p.add_argument("--bond-min", type=float, default=4.0)
    p.add_argument("--bond-max", type=float, default=8.0)
    p.add_argument("--clash-distance", type=float, default=3.0)
    p.add_argument("--clash-subsample", type=int, default=200)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = AnalyticsConfig(
        train_seq_path=args.train_seq,
        val_seq_path=args.val_seq,
        train_label_path=args.train_label,
        val_label_path=args.val_label,
        msa_dir=args.msa_dir,
        output_dir=args.out,
        max_targets=args.max_targets,
        coord_abs_threshold=args.coord_abs_threshold,
        bond_min=args.bond_min,
        bond_max=args.bond_max,
        clash_distance=args.clash_distance,
        clash_subsample=args.clash_subsample,
    )
    run_analytics(cfg)
    print(f"[OK] Analytics reports written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
