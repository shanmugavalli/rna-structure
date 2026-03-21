"""TPU-optimized data-parallel training for Kaggle TPU v5e-8 (8 chips).

Key optimizations over train.py:
  - torch_xla XMP (xla_multiprocessing): runs one process per chip → 8× data parallelism
  - bfloat16 forward pass + backward (native TPU precision, ~2× throughput vs float32)
  - fixed tensor shapes via collate_fn(fixed_len=max_seq_length) → zero XLA recompilations
  - no Python-level NaN / Inf checks inside the hot loop (graph-break / sync killer)
  - .item() calls deferred until after xm.mark_step() → no premature host sync
  - DistributedSampler + drop_last=True → constant step count per epoch
  - validation & checkpointing only on rank-0 (master) process

Usage (invoked from kaggle_train.ipynb cell 7):
    RNA_RUNTIME=tpu python src/train_tpu.py
"""
import os
import sys

# ── Prevent double XLA-client initialization ──────────────────────────────────
# config.py calls xm.xla_device() at class-definition time when RNA_RUNTIME=tpu.
# That initialises the XLA computation client in THIS process.  When xmp.spawn
# later queries available chips it calls InitializeComputationClient() again
# → SIGABRT ("can only be called once").  Setting RNA_NO_XLA_INIT=1 before
# importing config tells config.py to skip those calls; each spawned worker
# (fresh process) calls xm.xla_device() itself inside _train_fn.
os.environ['RNA_NO_XLA_INIT'] = '1'

import math
import time
import json
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import cfg
from dataset import (
    RNACachedDataset,
    RNAStructureDataset,
    collate_fn,
    _maybe_build_cache,
)
from model import build_model
from losses import StructureLoss
from utils import save_checkpoint


# ── TM-score helpers (CPU, master-only) ──────────────────────────────────────

def _kabsch_align(pred: np.ndarray, true: np.ndarray):
    n = pred.shape[0]
    pc = pred - pred.mean(0, keepdims=True)
    tc = true - true.mean(0, keepdims=True)
    if n < 3:
        return pc, tc
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return pc @ r.T, tc


def _tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    n = pred.shape[0]
    if n == 0:
        return 0.0
    d0 = max(0.5, 1.24 * (max(16, n) - 15) ** 0.33)
    p, t = _kabsch_align(pred, true)
    d = np.linalg.norm(p - t, axis=1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


# ── XLA sync helper (supports both torch_xla 2.x and legacy 1.x) ─────────────

def _sync():
    """Flush pending XLA computations to hardware."""
    if hasattr(torch_xla, 'sync'):
        torch_xla.sync()
    else:
        xm.mark_step()


# ── Per-chip worker ───────────────────────────────────────────────────────────

def _train_fn(rank: int, flags: dict):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    is_master = xm.is_master_ordinal()

    torch.manual_seed(42 + rank)

    # ── Datasets (worker init first, then coordinated cache build) ────────────
    # Rank-0 builds caches after TPU worker initialization; other workers wait
    # at rendezvous and then load the same cache files.
    train_cache = flags.get('train_cache')
    val_cache = flags.get('val_cache')

    if not train_cache or not val_cache:
        if is_master:
            print("[TPU] Building / verifying dataset caches (master only) ...")
            train_cache = _maybe_build_cache(cfg, split='train')
            val_cache = _maybe_build_cache(cfg, split='val')
            print(f"[TPU] train cache: {train_cache}")
            print(f"[TPU] val   cache: {val_cache}")

        # Wait until rank-0 finishes cache generation.
        xm.rendezvous('cache_ready')

        # Non-master workers resolve cache paths after barrier. If cache files
        # already exist this returns immediately without preprocessing.
        if not is_master:
            train_cache = _maybe_build_cache(cfg, split='train')
            val_cache = _maybe_build_cache(cfg, split='val')

    train_dataset = RNACachedDataset(train_cache, augment=True)
    val_dataset   = RNACachedDataset(val_cache,   augment=False)

    # Fixed-length collate avoids variable-shape graph recompilation on XLA.
    fixed_collate = partial(collate_fn, fixed_len=int(cfg.max_seq_length))

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,     # constant step count → zero recompilations
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, cfg.batch_size),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        collate_fn=fixed_collate,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if is_master:
        val_loader_raw = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            drop_last=False,
            collate_fn=fixed_collate,
        )

    # ── Model (bfloat16 on TPU) ───────────────────────────────────────────────
    model = build_model(cfg).to(torch.bfloat16).to(device)

    # ── Loss (float32 for numerical stability) ────────────────────────────────
    criterion = StructureLoss(
        weights=cfg.loss_weights,
        max_coord_abs=getattr(cfg, 'coord_abs_threshold', 2000.0),
    )

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    accum_steps      = max(1, cfg.grad_accum_steps)
    steps_per_epoch  = max(1, len(train_loader) // accum_steps)
    total_steps      = steps_per_epoch * cfg.epochs

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine decay: smoother than OneCycleLR on TPU (avoids peak-LR recompile)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.learning_rate * 0.05,
    )

    # ── Output dirs (master only) ─────────────────────────────────────────────
    if is_master:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        print(f"[TPU] {world_size} chips, batch/chip={cfg.batch_size}, "
              f"accum={accum_steps}, eff_batch={cfg.batch_size * world_size * accum_steps}")
        print(f"[TPU] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        print(f"[TPU] Steps/epoch: {steps_per_epoch} × {cfg.epochs} = {total_steps} total")
        print(f"[TPU] bfloat16: ON | max_seq_length: {cfg.max_seq_length}")

    best_val_tm  = 0.0
    history: dict = {}
    train_start  = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):

        # Time-budget guard (leave 15 min for val + checkpoint)
        max_minutes = getattr(cfg, 'max_train_minutes', 0)
        if max_minutes > 0 and epoch > 1:
            elapsed   = (time.time() - train_start) / 60
            avg_ep    = elapsed / (epoch - 1)
            if elapsed + avg_ep > max_minutes - 15:
                if is_master:
                    print(f"\n[TPU] Time budget reached at epoch {epoch}. Stopping.")
                break

        train_sampler.set_epoch(epoch)
        model.train()

        # Update loss curriculum weights each epoch
        epoch_weights = cfg.get_loss_weights(epoch)
        criterion.update_weights(epoch_weights)

        # MpDeviceLoader: background thread moves CPU batches → XLA device
        para_loader = pl.MpDeviceLoader(train_loader, device)

        # Accumulate loss on-device (avoids item() inside hot loop)
        epoch_loss_accum = torch.zeros(1, dtype=torch.float32, device=device)
        optimizer_steps  = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(para_loader):
            if batch is None:
                continue

            # Fetch tensors from device (MpDeviceLoader already placed them)
            seq_tokens  = batch['seq_tokens']               # (B, L) int64
            msa_tokens  = batch['msa_tokens']               # (B, S, L) int64
            true_coords = batch['coords'].to(torch.bfloat16)
            coord_mask  = batch.get('coord_mask', None)
            if coord_mask is not None:
                coord_mask = coord_mask.to(torch.bfloat16)

            # bfloat16 forward
            pred_coords, all_coords = model(seq_tokens, msa_tokens)

            # upcast to float32 for loss (prevents bf16 under/overflow in FAPE)
            pred_f32   = pred_coords.float()
            true_f32   = true_coords.float()
            all_f32    = [c.float() for c in all_coords]
            mask_f32   = coord_mask.float() if coord_mask is not None else None

            loss, _ = criterion(pred_f32, true_f32, all_f32, coord_mask=mask_f32)
            (loss / accum_steps).backward()

            # Accumulate loss scalar on device (no host sync)
            epoch_loss_accum = epoch_loss_accum + loss.detach()

            if (step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                xm.optimizer_step(optimizer, barrier=False)
                _sync()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        # Handle final partial accumulation bucket
        remainder = len(train_loader) % accum_steps
        if remainder != 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            xm.optimizer_step(optimizer, barrier=False)
            _sync()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        # Cross-chip loss reduction (single collective, then one .item() call)
        total_loss_tensor = xm.all_reduce(xm.REDUCE_SUM, epoch_loss_accum.clone())
        _sync()
        train_loss = (total_loss_tensor / max(1, optimizer_steps * world_size)).item()

        # ── Validation (master only; every val_frequency epochs) ─────────────
        val_tm   = 0.0
        val_loss = 0.0

        if is_master and epoch % cfg.val_frequency == 0:
            model.eval()
            tm_scores  = []
            val_losses = []

            val_para = pl.MpDeviceLoader(val_loader_raw, device)
            with torch.no_grad():
                for vbatch in val_para:
                    if vbatch is None:
                        continue
                    vs    = vbatch['seq_tokens']
                    vm    = vbatch['msa_tokens']
                    vc    = vbatch['coords'].to(torch.bfloat16)
                    vmask = vbatch.get('coord_mask', None)
                    if vmask is not None:
                        vmask = vmask.to(torch.bfloat16)

                    vp, va = model(vs, vm)

                    vp_f32   = vp.float()
                    vc_f32   = vc.float()
                    va_f32   = [c.float() for c in va]
                    vmask_f32= vmask.float() if vmask is not None else None

                    vloss, _ = criterion(vp_f32, vc_f32, va_f32, coord_mask=vmask_f32)
                    # Sync before .item() to avoid deferred computation
                    _sync()
                    val_losses.append(vloss.item())

                    # TM-score on CPU
                    pred_np = vp_f32.detach().cpu().numpy()
                    true_np = vc_f32.detach().cpu().numpy()
                    if vmask is not None:
                        mask_np = (vmask.detach().cpu().numpy() > 0.5)
                    else:
                        mask_np = np.ones(pred_np.shape[:2], dtype=bool)

                    for si in range(pred_np.shape[0]):
                        valid = mask_np[si]
                        if valid.sum() >= 3:
                            tm_scores.append(_tm_score(pred_np[si][valid], true_np[si][valid]))

            val_tm   = float(np.mean(tm_scores))   if tm_scores   else 0.0
            val_loss = float(np.mean(val_losses))  if val_losses  else 0.0

        # ── Wait for all chips before next epoch ──────────────────────────────
        xm.rendezvous('end_epoch')

        # ── Logging & checkpointing (master) ──────────────────────────────────
        if is_master:
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else cfg.learning_rate
            elapsed_min = (time.time() - train_start) / 60
            print(f"[Epoch {epoch:3d}/{cfg.epochs}] "
                  f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  Val TM: {val_tm:.4f}  "
                  f"LR: {lr_now:.2e}  Elapsed: {elapsed_min:.1f}m")

            history.setdefault('train_loss', []).append(train_loss)
            history.setdefault('val_loss',   []).append(val_loss)
            history.setdefault('val_tm',     []).append(val_tm)

            # Save best TM checkpoint (convert bf16 → fp32 for portability)
            if epoch % cfg.val_frequency == 0 and val_tm > best_val_tm:
                best_val_tm = val_tm
                _save_fp32_ckpt(model, cfg.checkpoint_dir, 'best_model.pt', epoch, val_loss, val_tm)
                print(f"  -> best_model.pt  (val_tm: {val_tm:.4f})")

            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                fname = f'checkpoint_epoch_{epoch}_tm_{val_tm:.4f}.pt'
                _save_fp32_ckpt(model, cfg.checkpoint_dir, fname, epoch, val_loss, val_tm)

            # Flush training history
            with open(os.path.join(cfg.log_dir, 'kaggle_gpu_train.log'), 'a') as f:
                f.write(f"[Epoch {epoch:3d}/{cfg.epochs}] "
                        f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  Val TM: {val_tm:.4f}  "
                        f"LR: {lr_now:.2e}\n")
            with open(os.path.join(cfg.log_dir, 'tpu_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

    if is_master:
        print(f"\n[TPU] Training done. Best Val TM: {best_val_tm:.4f}")


def _save_fp32_ckpt(model, ckpt_dir, fname, epoch, val_loss, val_tm):
    """Convert bf16 state dict to fp32 and save."""
    state = {
        k: v.float() if v.dtype == torch.bfloat16 else v
        for k, v in model.state_dict().items()
    }
    save_checkpoint(
        os.path.join(ckpt_dir, fname),
        model=state,
        epoch=epoch,
        val_loss=val_loss,
        val_tm=val_tm,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    flags = {'train_cache': None, 'val_cache': None}

    # Use torch_xla.launch (introduced in torch_xla 2.4, correct for PJRT/TPU v5e).
    # Unlike xmp.spawn, torch_xla.launch does NOT call GetComputationClient() in
    # the parent process, so there is no double-init SIGABRT even though
    # 'import torch_xla' already initialised the computation client above.
    # Each spawned worker starts fresh, imports torch_xla, and gets its own chip.
    print("[TPU] Launching workers with torch_xla.launch ...")
    try:
        torch_xla.launch(_train_fn, args=(flags,))
    except RuntimeError as exc:
        msg = str(exc)
        # Enforce multiprocess-only TPU execution.
        address_mismatch = (
            'slice_builder_worker_addresses' in msg
            and 'Expected 8 worker addresses, got 1' in msg
        )
        vfio_busy = ('/dev/vfio/' in msg and 'Device or resource busy' in msg)
        xla_reinit = ('InitializeComputationClient() can only be called once' in msg)

        if address_mismatch or vfio_busy or xla_reinit:
            raise RuntimeError(
                "TPU multiprocess initialization failed and single-process fallback is disabled. "
                "Restart the Kaggle runtime and re-run TPU launch. "
                f"Original error: {msg}"
            ) from exc
        else:
            raise


if __name__ == '__main__':
    main()
