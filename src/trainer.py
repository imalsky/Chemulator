#!/usr/bin/env python3
"""
trainer.py — Lightning trainer with adaptive stiff loss (no MPS)

Fast path:
- CSV-only logging; checkpoints: best.ckpt / last.ckpt in work_dir
- AMP precision from cfg (bf16/fp16/fp32); TF32/cudnn handled in hardware.optimize_hardware
- Optional torch.compile (Inductor) with sane defaults
- Fused AdamW when available; zero_grad(set_to_none=True)
- Cosine schedule with warmup
- Vectorized K handling; optional K-mask reduction
- Epoch callback to drive dataset.set_epoch(...) (anchors/offsets are dataset-controlled)

Dataset contract: (y_i, dt_norm[B,K,1], y_j[B,K,S], g[B,G], aux{...}[, k_mask[B,K]])

Requested changes applied previously:
- Removed time-edge weighting (time_edge_gain) completely.
- Removed fractional physical loss (use_fractional); physical loss is MAE in log10-physical space.

This revision:
- Print epoch summary at on_fit_epoch_end to ensure val metrics exist on multi-GPU.
"""

from __future__ import annotations

# ============================== GLOBAL CONSTANTS ==============================

# Checkpointing / logging
CKPT_FILENAME_BEST: str = "best"
CKPT_MONITOR: str = "val_loss"
CKPT_MODE: str = "min"
LOG_EVERY_N_STEPS: int = 200
ENABLE_CHECKPOINTING: bool = True
ENABLE_PROGRESS_BAR: bool = False
ENABLE_MODEL_SUMMARY: bool = False
DETECT_ANOMALY: bool = False
INFERENCE_MODE: bool = True
LR_LOGGING_INTERVAL: str = "epoch"   # for LearningRateMonitor

# Training defaults (used when cfg keys absent)
DEF_EPOCHS: int = 100
DEF_GRAD_CLIP: float = 0.0
DEF_ACCUMULATE: int = 1
DEF_MAX_TRAIN_STEPS_PER_EPOCH: int = 0
DEF_MAX_VAL_BATCHES: int = 0
DEF_LR: float = 1e-3
DEF_WEIGHT_DECAY: float = 1e-4
DEF_WARMUP_EPOCHS: int = 0
DEF_MIN_LR: float = 1e-6
DEF_BETA_KL: float = 0.0

# torch.compile defaults
COMPILE_ENABLE_DEFAULT: bool = False
COMPILE_BACKEND: str = "inductor"
COMPILE_MODE: str = "default"
COMPILE_DYNAMIC: bool = True
COMPILE_FULLGRAPH: bool = False

# AdaptiveStiffLoss defaults / numerics
LOSS_LAMBDA_PHYS: float = 1.0
LOSS_LAMBDA_Z: float = 0.1
LOSS_EPS_PHYS: float = 1e-20  # retained for compatibility; not used now that fractional loss is removed
LOG_STD_CLAMP_MIN: float = 1e-10
W_SPECIES_CLAMP_MIN: float = 0.5
W_SPECIES_CLAMP_MAX: float = 2.0
LOG10_TO_LIN_MIN: float = -45.0

# Resume behavior
ENV_RESUME_VAR: str = "RESUME"
RESUME_AUTO_SENTINEL: str = "auto"
LAST_CKPT_NAME: str = "last.ckpt"

SPECIES_RANGE_EPS: float = 1e-6
W_SPECIES_MEAN_EPS: float = 1e-12
WARMUP_START_FACTOR: float = 0.1

# ============================================================================

import os
PL_DEVICES: int = int(os.getenv("PL_DEVICES", "1"))

import json
import time
import math
import logging
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MeanMetric


# ------------------------------ Adaptive Loss --------------------------------

class AdaptiveStiffLoss(nn.Module):
    """
    Adaptive loss for stiff systems in log-normalized space (no atomic penalty).
    - Physical component: MAE in log10-physical space (ALWAYS; fractional disabled).
    - Stabilizer component: MSE in z-space.
    - No time-edge weighting (removed).
    """
    def __init__(
        self,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        species_log_min: torch.Tensor,
        species_log_max: torch.Tensor,
        *,
        lambda_phys: float = LOSS_LAMBDA_PHYS,
        lambda_z: float = LOSS_LAMBDA_Z,
        epsilon_phys: float = LOSS_EPS_PHYS,  # kept for signature compatibility; unused
    ) -> None:
        super().__init__()
        # Buffers for Lightning device moves
        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=LOG_STD_CLAMP_MIN))
        self.register_buffer("log_min", species_log_min.detach().clone())
        self.register_buffer("log_max", species_log_max.detach().clone())

        # Species weights ∝ sqrt(dynamic range), clipped
        rng = torch.clamp(self.log_max - self.log_min, min=SPECIES_RANGE_EPS)
        w = torch.sqrt(rng)
        w = w / (w.mean() + W_SPECIES_MEAN_EPS)
        self.register_buffer("w_species", torch.clamp(w, W_SPECIES_CLAMP_MIN, W_SPECIES_CLAMP_MAX))

        # Scalars
        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.eps_phys = float(epsilon_phys)  # not used with MAE log-phys (retained for compat)

    @torch.no_grad()
    def _z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.log_stds + self.log_means

    def forward(
            self,
            pred_z: torch.Tensor,  # [B,K,S]
            true_z: torch.Tensor,  # [B,K,S]
            t_norm: torch.Tensor,  # [B,K] or [B,K,1] (unused; kept for signature compatibility)
            mask: Optional[torch.Tensor] = None,  # [B,K] (optional)
            return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Stabilizer in z-space
        loss_z = (pred_z - true_z) ** 2  # [B,K,S]

        # Log-physical path (ALWAYS MAE in log10-physical)
        pred_log = self._z_to_log10(pred_z)
        true_log = self._z_to_log10(true_z)
        loss_phys = (pred_log - true_log).abs()  # [B,K,S]

        # Species weighting
        loss_phys = loss_phys * self.w_species  # [B,K,S]

        # Masked mean over valid positions (if provided)
        if mask is not None:
            m = mask.unsqueeze(-1).to(loss_phys.dtype)  # [B,K,1]
            loss_phys = loss_phys * m
            loss_z = loss_z * m

            if mask.dtype == torch.bool:
                valid_pairs = mask.sum()
            else:
                valid_pairs = (mask > 0).sum()
            if int(valid_pairs.item()) == 0:
                raise ValueError("AdaptiveStiffLoss: mask has zero valid [B,K] positions; cannot compute loss.")
            denom = valid_pairs.to(loss_phys.dtype) * loss_phys.shape[-1]
        else:
            denom = torch.tensor(loss_phys.numel(), dtype=loss_phys.dtype, device=loss_phys.device)

        phys = self.lambda_phys * loss_phys.sum() / denom
        zstb = self.lambda_z * loss_z.sum() / denom
        total = phys + zstb
        if return_components:
            return {"total": total, "phys": phys, "z": zstb}
        return total


# --------------------------- LightningModule wrapper --------------------------

class ModelTask(pl.LightningModule):
    """Generic LightningModule wrapping the model and adaptive stiff loss."""
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], work_dir: Path) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)

        tcfg = cfg.get("training", {})
        self.lr = float(tcfg.get("lr", DEF_LR))
        self.weight_decay = float(tcfg.get("weight_decay", DEF_WEIGHT_DECAY))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", DEF_WARMUP_EPOCHS))
        self.min_lr = float(tcfg.get("min_lr", DEF_MIN_LR))
        self.beta_kl = float(tcfg.get("beta_kl", DEF_BETA_KL))

        # Optional GT slicing when target subset used
        self._target_idx = self._resolve_target_indices()

        # Build loss from normalization manifest (no atomic term)
        self.criterion = self._build_loss()

        # Weighted, distributed-safe epoch metrics for validation
        self.val_loss_epoch = MeanMetric(sync_on_compute=True)
        self.val_phys_epoch = MeanMetric(sync_on_compute=True)
        self.val_z_epoch = MeanMetric(sync_on_compute=True)

        # Save compact hparams for repro
        self.save_hyperparameters({
            "cfg_min": {
                "training": {
                    "lr": self.lr, "weight_decay": self.weight_decay,
                    "warmup_epochs": self.warmup_epochs, "min_lr": self.min_lr,
                    "beta_kl": self.beta_kl,
                },
                "paths": {"processed_data_dir": cfg.get("paths", {}).get("processed_data_dir", "")},
            },
            "work_dir": str(work_dir),
        })

    # -------------------------------- helpers ---------------------------------

    def _resolve_target_indices(self) -> Optional[torch.Tensor]:
        data_cfg = self.cfg.get("data", {})
        species = list(data_cfg.get("species_variables") or [])
        targets = list(data_cfg.get("target_species") or species)
        if targets != species:
            name_to_idx = {n: i for i, n in enumerate(species)}
            idx = [name_to_idx[n] for n in targets]
            return torch.tensor(idx, dtype=torch.long)
        return None

    def _build_loss(self) -> nn.Module:
        manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        stats = manifest["per_key_stats"]
        meta = manifest.get("meta", {})

        data_cfg = self.cfg.get("data", {})
        targets = data_cfg.get("target_species")
        if targets:
            species_names = list(targets)
        else:
            species_names = list(data_cfg.get("species_variables") or meta.get("species_variables", []))
            if not species_names:
                raise RuntimeError("No species list available in cfg or manifest.meta.species_variables")

        log_means, log_stds, log_mins, log_maxs = [], [], [], []
        for n in species_names:
            s = stats[n]
            log_means.append(float(s.get("log_mean", 0.0)))
            log_stds.append(float(s.get("log_std", 1.0)))
            log_mins.append(float(s.get("log_min", -30.0)))
            log_maxs.append(float(s.get("log_max", 0.0)))

        loss_cfg = self.cfg.get("training", {}).get("adaptive_stiff_loss", {})
        # Note: use_fractional and time_edge_gain intentionally ignored/removed.
        return AdaptiveStiffLoss(
            log_means=torch.tensor(log_means, dtype=torch.float32),
            log_stds=torch.tensor(log_stds, dtype=torch.float32),
            species_log_min=torch.tensor(log_mins, dtype=torch.float32),
            species_log_max=torch.tensor(log_maxs, dtype=torch.float32),
            lambda_phys=float(loss_cfg.get("lambda_phys", LOSS_LAMBDA_PHYS)),
            lambda_z=float(loss_cfg.get("lambda_z", LOSS_LAMBDA_Z)),
            epsilon_phys=float(loss_cfg.get("epsilon_phys", LOSS_EPS_PHYS)),  # retained, but unused
        )

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.model(y_i, dt_norm, g)

    # --------------------------------- steps ---------------------------------

    def _unpack(self, batch):
        # (y_i, dt_norm, y_j, g, aux[, k_mask])
        if len(batch) == 6:
            y_i, dt_norm, y_j, g, _aux, k_mask = batch
        else:
            y_i, dt_norm, y_j, g, _aux = batch
            k_mask = None
        if self._target_idx is not None:
            idx = self._target_idx.to(y_j.device)
            y_j = y_j.index_select(dim=-1, index=idx)
        return y_i, dt_norm, y_j, g, k_mask


    def configure_optimizers(self):
        """
        AdamW (+ fused when safe) with Linear warmup -> Cosine.
        Returns ([optimizer], [scheduler_dict]) for broad PL compatibility.
        """
        # Detect fused AdamW support
        try:
            has_fused_flag = ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        except Exception:
            has_fused_flag = False

        # Effective gradient-clip from Trainer (if any)
        tr = getattr(self, "trainer", None)
        try:
            clip_val = float(getattr(tr, "gradient_clip_val", 0.0) or 0.0) if tr is not None else \
                       float(self.cfg.get("training", {}).get("gradient_clip", 0.0) or 0.0)
        except Exception:
            clip_val = 0.0

        # Some builds disable fused under FP16 + grad clipping
        is_fp16_amp = False
        try:
            from pytorch_lightning.plugins.precision.amp import AMPPrecisionPlugin
            is_fp16_amp = isinstance(getattr(self.trainer, "precision_plugin", None), AMPPrecisionPlugin) and \
                          getattr(self.trainer.precision_plugin, "precision", None) in ("16-mixed", "16-true")
        except Exception:
            pass

        use_fused = torch.cuda.is_available() and has_fused_flag and (clip_val == 0.0 or not is_fp16_amp)

        # Optimizer
        opt_kwargs = dict(lr=self.lr, weight_decay=self.weight_decay)
        if use_fused:
            opt_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(self.parameters(), **opt_kwargs)

        # Scheduler: linear warmup -> cosine
        scheds = []
        if self.warmup_epochs > 0:
            scheds.append(LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=max(1, self.warmup_epochs)))

        max_epochs = max(1, (getattr(self.trainer, "max_epochs", 1) or 1))
        t_max = max(1, max_epochs - self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.min_lr)

        scheduler = cosine if not scheds else SequentialLR(
            optimizer, schedulers=[scheds[0], cosine], milestones=[self.warmup_epochs]
        )

        # Return tuple/list form (avoids edge-case mis-detection on some PL builds)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": CKPT_MONITOR}]

    def training_step(self, batch, batch_idx: int):
        y_i, dt_in, y_j, g, k_mask = self._unpack(batch)
        pred = self(y_i, dt_in, g)
        if self._target_idx is not None:
            pred = pred.index_select(dim=-1, index=self._target_idx.to(pred.device))
        if pred.shape != y_j.shape:
            raise RuntimeError(f"Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")

        comps = self.criterion(pred, y_j, dt_in, k_mask, return_components=True)
        loss = comps["total"]

        # KL (VAE)
        if self.beta_kl > 0.0 and getattr(self.model, "vae_mode", False):
            kl = getattr(self.model, "kl_loss", None)
            if kl is not None:
                loss = loss + self.beta_kl * kl
                self.log("train_kl", self.beta_kl * kl, on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_phys", comps["phys"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_z", comps["z"], on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation with explicit sample-weighted accumulation.

        Weight = S * (#valid [B,K]) using k_mask if provided; otherwise B*K*S.
        Raises on non-finite components or zero valid pairs in the batch.
        Accumulates locally; global reduction occurs at epoch end.
        """
        # Unpack and forward
        y_i, dt_in, y_j, g, k_mask = self._unpack(batch)
        pred = self(y_i, dt_in, g)
        if self._target_idx is not None:
            pred = pred.index_select(dim=-1, index=self._target_idx.to(pred.device))

        # Shape sanity
        if pred.shape != y_j.shape:
            raise RuntimeError(f"[val] Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")

        # Per-batch loss components
        comps = self.criterion(pred, y_j, dt_in, k_mask, return_components=True)
        total = comps["total"].detach()
        phys = comps["phys"].detach()
        zstb = comps["z"].detach()

        # Guard against non-finite values
        if not torch.isfinite(total):
            raise ValueError(f"[val] Non-finite total loss at batch {batch_idx}")
        if not torch.isfinite(phys):
            raise ValueError(f"[val] Non-finite phys loss at batch {batch_idx}")
        if not torch.isfinite(zstb):
            raise ValueError(f"[val] Non-finite z loss at batch {batch_idx}")

        # Weight = S * (#valid [B,K])  (no time-edge weights)
        if k_mask is not None:
            valid_pairs = torch.count_nonzero(k_mask if k_mask.dtype == torch.bool else (k_mask > 0))
        else:
            B, K, _S = y_j.shape
            valid_pairs = torch.tensor(B * K, device=y_j.device, dtype=torch.long)

        if int(valid_pairs.item()) == 0:
            B, K, S = y_j.shape
            raise ValueError(
                f"[val] Zero valid [B,K] pairs in batch {batch_idx} (B={B}, K={K}, S={S}). "
                f"Check k_mask construction or dataset filtering."
            )

        S = torch.tensor(y_j.shape[-1], device=y_j.device, dtype=torch.float32)
        weight = (valid_pairs.to(torch.float32) * S).to(torch.float64)  # scalar

        # Local accumulators
        self._v_sum += (total.to(torch.float64) * weight)
        self._v_phys_sum += (phys.to(torch.float64) * weight)
        self._v_z_sum += (zstb.to(torch.float64) * weight)
        self._v_wsum += weight

        return total  # scalar for Lightning bookkeeping

    def on_validation_epoch_start(self) -> None:
        """
        Initialize DDP-safe weighted accumulators. Use float64 for stability.
        Must live on the LightningModule (not a Callback) so it's called
        before any validation_step on every rank.
        """
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device
        self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)       # sum(weight * total)
        self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)      # sum(weight)
        self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)  # sum(weight * phys)
        self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)     # sum(weight * z)

    # In class ModelTask(pl.LightningModule)
    def on_validation_epoch_end(self) -> None:
        """
        Finalize and log sample-weighted validation metrics once per epoch.

        - Skips sanity-check pass.
        - All-reduces accumulators across ranks if DDP is initialized.
        - If global weight is zero (no val batches anywhere), skip logging.
        - Logs plain scalars (already reduced); avoids TorchMetrics compute() on empty state.
        """
        tr = getattr(self, "trainer", None)
        if tr is not None and getattr(tr, "sanity_checking", False):
            # Reset and bail during Lightning’s sanity pass
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
                if hasattr(self, name):
                    setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        import torch.distributed as dist

        # Ensure accumulators exist even if validation never started
        if not hasattr(self, "_v_wsum"):
            dev = self.device if hasattr(self, "device") else next(self.parameters()).device
            self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)

        # Global reduction (SUM) if DDP is active
        if dist.is_available() and dist.is_initialized():
            for t in (self._v_sum, self._v_phys_sum, self._v_z_sum, self._v_wsum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        # If no validation batches globally, skip logging
        if float(self._v_wsum.item()) <= 0.0:
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
                setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        # Compute weighted means
        w = self._v_wsum
        val_loss = (self._v_sum / w).to(torch.float32)
        val_phys = (self._v_phys_sum / w).to(torch.float32)
        val_z = (self._v_z_sum / w).to(torch.float32)

        # Log as scalars; no extra sync (already all-reduced)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_phys", val_phys, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_z", val_z, prog_bar=False, on_epoch=True, sync_dist=False)

        # Reset for next epoch
        for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
            setattr(self, name, torch.zeros_like(getattr(self, name)))


# ------------------------------ Epoch drive callback --------------------------

class SetEpochCallback(Callback):
    """Ensure dataset-driven sampling is advanced per epoch."""
    def __init__(self, datasets: list) -> None:
        super().__init__()
        self.datasets = [d for d in datasets if d is not None]

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for ds in self.datasets:
            try:
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(trainer.current_epoch)
            except Exception:
                pass

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for ds in self.datasets:
            try:
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(trainer.current_epoch)
            except Exception:
                pass


# ------------------------------ Epoch summary callback ------------------------

class EpochSummaryCallback(Callback):
    """
    Short, nicely-justified epoch summary to stdout.
    Columns: train_loss | val_loss | lr | time

    Prints at on_fit_epoch_end so that val metrics are guaranteed to be present.
    """
    def __init__(self) -> None:
        super().__init__()
        self._printed_header = False
        self._t0: Optional[float] = None
        self._printed_epoch = -1

    def _fmt_secs(self, seconds: float) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _header(self) -> str:
        return (
            "EPOCH | "
            f"{'train':>11} | "
            f"{'val':>11} | "
            f"{'lr':>10} | "
            f"{'time':>8}"
        )

    def _sep(self) -> str:
        return (
            "----- | "
            + " | ".join([
                "-" * 11,
                "-" * 11,
                "-" * 10,
                "-" * 8,
            ])
        )

    def _get_metric(self, trainer: pl.Trainer, key_candidates: list[str]) -> float:
        # More robust on multi-GPU: consult both callback_metrics and logged_metrics.
        merged = {}
        try:
            merged.update(trainer.callback_metrics or {})
        except Exception:
            pass
        try:
            merged.update(trainer.logged_metrics or {})
        except Exception:
            pass

        for k in key_candidates:
            if k in merged:
                v = merged[k]
                try:
                    v = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                    return v
                except Exception:
                    continue
        return float("nan")

    def _emit(self, trainer: pl.Trainer) -> None:
        # Skip sanity check pass and silence non-zero ranks
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "is_global_zero", False):
            return

        if not self._printed_header:
            print(self._header(), flush=True)
            print(self._sep(), flush=True)
            self._printed_header = True

        epoch = trainer.current_epoch
        train_v = self._get_metric(trainer, ["train_loss", "train_loss_epoch", "train"])
        val_v = self._get_metric(trainer, ["val_loss", "val"])

        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")

        elapsed = (time.perf_counter() - self._t0) if self._t0 is not None else float("nan")
        line = (
            f"E{epoch:04d} | "
            f"{train_v:>11.3e} | "
            f"{val_v:>11.3e} | "
            f"{lr:>10.3e} | "
            f"{self._fmt_secs(elapsed):>8}"
        )
        print(line, flush=True)
        self._printed_epoch = epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # If no validation is configured at all, emit after training epoch.
        try:
            nval = trainer.num_val_batches
            if isinstance(nval, (list, tuple)):
                nval = sum(int(x) for x in nval)
            nval = int(nval)
        except Exception:
            nval = 0
        if nval == 0:
            self._emit(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Runs after validation metrics are committed
        self._emit(trainer)


# ------------------------------ External wrapper -----------------------------

class Trainer:
    """Compatibility wrapper: `from trainer import Trainer` -> .train() returns best val."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        work_dir: Path,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device

        self.logger = logger or logging.getLogger("trainer")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # Silence non-zero ranks to avoid duplicate INFO lines
        if int(os.getenv("RANK", "0")) != 0:
            self.logger.propagate = False
            self.logger.setLevel(logging.WARNING)  # or logging.ERROR to be stricter

        self.work_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.get("training", {})
        self.epochs = int(tcfg.get("epochs", DEF_EPOCHS))
        self.grad_clip = float(tcfg.get("gradient_clip", DEF_GRAD_CLIP))
        self.accumulate = int(tcfg.get("accumulate_grad_batches", DEF_ACCUMULATE))
        self.torch_compile = bool(tcfg.get("torch_compile", COMPILE_ENABLE_DEFAULT))

        # Optional limits (batches), distinct from dataset step bounds
        self.limit_train_batches = int(tcfg.get("max_train_steps_per_epoch", DEF_MAX_TRAIN_STEPS_PER_EPOCH) or 0)
        self.limit_val_batches = int(tcfg.get("max_val_batches", DEF_MAX_VAL_BATCHES) or 0)

    def _precision_from_cfg(self) -> str:
        # Accept both aliases and explicit Lightning strings
        mp = str(self.cfg.get("mixed_precision", {}).get("mode", "32-true")).lower().strip()
        aliases = {
            # BF16
            "bf16": "bf16-mixed",
            "bfloat16": "bf16-mixed",
            "bf16-mixed": "bf16-mixed",
            "bfloat16-mixed": "bf16-mixed",
            # FP16
            "fp16": "16-mixed",
            "float16": "16-mixed",
            "16": "16-mixed",
            "16-mixed": "16-mixed",
            # FP32 / none
            "none": "32-true",
            "fp32": "32-true",
            "32": "32-true",
            "32-true": "32-true",
        }
        return aliases.get(mp, "32-true")

    def _accelerator_from_device(self) -> str:
        if self.device.type == "cuda":
            return "gpu"
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def _resolve_resume_ckpt(self) -> Optional[str]:
        tcfg = self.cfg.get("training", {})
        p_cfg = tcfg.get("resume_path") or tcfg.get("resume")
        if isinstance(p_cfg, str) and p_cfg.strip():
            p = Path(p_cfg)
            return str(p) if p.is_file() else None
        env_resume = os.environ.get(ENV_RESUME_VAR, "").strip()
        if env_resume and env_resume.lower() != RESUME_AUTO_SENTINEL:
            p = Path(env_resume)
            return str(p) if p.is_file() else None
        if env_resume.lower() == RESUME_AUTO_SENTINEL or bool(tcfg.get("auto_resume", True)):
            p = self.work_dir / LAST_CKPT_NAME
            if p.is_file():
                return str(p)
        return None

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        if not self.torch_compile:
            return model
        try:
            compiled = torch.compile(
                model,
                backend=COMPILE_BACKEND,
                mode=COMPILE_MODE,
                dynamic=COMPILE_DYNAMIC,
                fullgraph=COMPILE_FULLGRAPH,
            )
            self.logger.info(f"torch.compile enabled ({COMPILE_BACKEND}, {COMPILE_MODE}, dynamic={COMPILE_DYNAMIC})")
            return compiled
        except Exception as e:
            self.logger.warning(f"torch.compile disabled ({e})")
            return model

    def train(self) -> float:
        # Optional model compile (Lightning handles AMP/precision)
        self.model = self._maybe_compile(self.model)
        task = ModelTask(self.model, self.cfg, self.work_dir)

        csv_logger = CSVLogger(save_dir=str(self.work_dir), name="csv")

        ckpt_cb = ModelCheckpoint(
            dirpath=str(self.work_dir),
            filename=CKPT_FILENAME_BEST,
            monitor=CKPT_MONITOR,
            mode=CKPT_MODE,
            save_top_k=1,
            save_last=True,
            verbose=False,
        )
        lr_cb = LearningRateMonitor(logging_interval=LR_LOGGING_INTERVAL)
        epoch_cb = SetEpochCallback([
            getattr(self.train_loader, "dataset", None),
            getattr(self.val_loader, "dataset", None) if self.val_loader is not None else None,
        ])
        summary_cb = EpochSummaryCallback()

        accelerator = self._accelerator_from_device()
        precision = self._precision_from_cfg()

        # Keep Lightning's cudnn.benchmark aligned with hardware.optimize_hardware
        det = bool(self.cfg.get("system", {}).get("deterministic", False))
        bench_cfg = bool(self.cfg.get("system", {}).get("cudnn_benchmark", True)) and not det

        trainer_kwargs = {}
        if self.limit_train_batches > 0:
            trainer_kwargs["limit_train_batches"] = self.limit_train_batches
        if self.limit_val_batches > 0:
            trainer_kwargs["limit_val_batches"] = self.limit_val_batches

        # External launch (mpiexec/torchrun) exports WORLD_SIZE/RANK.
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        externally_launched = world_size > 1

        # One process per GPU when launched externally; otherwise use configured count.
        devices = 1 if externally_launched else PL_DEVICES

        # Make PL's world-size match the external launcher for sampler correctness.
        # With devices=1 per rank, setting num_nodes=world_size yields world_size = num_nodes * devices.
        num_nodes = world_size if externally_launched else 1

        # Robust DDP strategy selection for older/newer Lightning:
        #  - Prefer DDPStrategy with NCCL if available.
        #  - Fallback to the string "ddp" to avoid passing unknown kwargs into torch.nn.parallel.DistributedDataParallel.
        strategy = "auto"
        if externally_launched or devices > 1:
            try:
                from pytorch_lightning.strategies import DDPStrategy
                try:
                    # Newer PL: accepts process_group_backend for init_process_group
                    strategy = DDPStrategy(process_group_backend="nccl")
                except TypeError:
                    # Older PL: no such kwarg; use defaults (env:// is inferred from env vars)
                    strategy = DDPStrategy()
            except Exception:
                # Very old PL or import issue: plain "ddp" string works broadly
                strategy = "ddp"

        pl_trainer = pl.Trainer(
            default_root_dir=str(self.work_dir),
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,                  # ensures PL's world_size matches the external world_size
            strategy=strategy,
            precision=precision,                  # e.g., "bf16-mixed"
            gradient_clip_val=self.grad_clip if self.grad_clip > 0 else 0.0,
            accumulate_grad_batches=max(1, self.accumulate),
            logger=csv_logger,
            callbacks=[ckpt_cb, lr_cb, epoch_cb, summary_cb],
            enable_checkpointing=ENABLE_CHECKPOINTING,
            enable_progress_bar=ENABLE_PROGRESS_BAR,
            benchmark=bench_cfg,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            enable_model_summary=ENABLE_MODEL_SUMMARY,
            detect_anomaly=DETECT_ANOMALY,
            inference_mode=INFERENCE_MODE,
            **trainer_kwargs,
        )

        ckpt_path = self._resolve_resume_ckpt()
        self.logger.info(f"resume: {ckpt_path if ckpt_path else 'fresh'}")

        pl_trainer.fit(
            task,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=ckpt_path,
        )

        best = ckpt_cb.best_model_score
        return float(best.item()) if best is not None else float("nan")
