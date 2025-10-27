#!/usr/bin/env python3
"""
Trainer for Stable LPV Koopman Autoencoder
==========================================

- Forward loss trains on all Δt offsets each step (covers full Δt support).
- Unified rollout loss: vectorized when TF=0; cached loop when TF>0.
- Teacher-forcing schedule (none | linear).
- Rollout horizon warmup.
- Cosine LR with linear warmup; AdamW; manual gradient clipping (fused-optimizer safe).
- Lightning checkpointing + best model export.
- Adaptive stiff loss (fractional physical + MSE normalized), device-safe.
- End-of-epoch one-line summary: epoch, train/val, ||g||, ||p||, lr, TF, H.

Config toggles:
  training.ema         → {"enable": bool, "decay": float}
  training.swa         → {"enable": bool, "epoch_start": float, "swa_lrs": float|null}
  training.model_soup  → {"enable": bool, "top_k": int}

Batch format: (y_i, dt_norm_offsets, y_j, g, [aux], [mask])
    y_i:    [B, S]
    dt:     [B, K]  (OFFSETS from anchor, normalized)
    y_j:    [B, K, S]
    g:      [B, G]
    mask:   [B, K] or None
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence

import json
import logging
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import torch
import optuna

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer as LightningTrainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import warnings

# Suppress dataloader worker warnings
warnings.filterwarnings( "ignore",message=".*does not have many workers.*")

# Suppress logging interval warning
warnings.filterwarnings( "ignore", message=".*is smaller than the logging interval.*")

# Try both modern and legacy Lightning callback import paths for EMA/SWA
try:
    from lightning.pytorch.callbacks import EMA as _EMA_CB, StochasticWeightAveraging as _SWA_CB  # type: ignore
except Exception:
    try:
        from pytorch_lightning.callbacks import EMA as _EMA_CB, StochasticWeightAveraging as _SWA_CB  # type: ignore
    except Exception:
        _EMA_CB = None
        _SWA_CB = None

# Cap residuals to prevent exploding gradients on rare bad batches
MAX_RESIDUAL = 1e4


# ------------------------------ Adaptive Stiff Loss -------------------------------

class AdaptiveStiffLoss(nn.Module):
    """
    Adaptive loss for stiff chemical systems.

    total = λ_mse * MSE(z-space) + λ_phys * mean_s( |pred_phys - true_phys| / (|true_phys| + eps_phys) )

    Features restored from the old trainer:
      • Species weights based on dynamic log-range (stabilizes influence across species)
      • Time-dependent weights that emphasize trajectory edges (t≈0 and t≈1)

    Fast path (no full denorm) supports per-species log normalizations:
      - "log-standard":   z = (log10(x) - log_mean) / log_std
      - "log-min-max":    z = (log10(x) - log_min) / (log_max - log_min)

    Args:
      manifest: normalization manifest with keys:
          - "per_key_stats": {key: {...}}
          - "normalization_methods": {key: "log-standard"|"log-min-max"|...}
      species_keys: list of species names (order must match model outputs)
      lambda_phys: weight for fractional physical error term
      lambda_mse:  weight for normalized-space MSE stabilizer
      epsilon_phys: small constant in denominator of fractional error
      rel_cap: optional clamp for fractional error (cap extreme outliers)
      time_edge_gain: gain ≥ 1.0; w(t)=1+(g-1)*(1-4t(1-t)) → g at t=0,1 and 1 at t=0.5
      device: torch device for internal buffers
    """

    def __init__(
        self,
        manifest: dict,
        species_keys: Sequence[str],
        lambda_phys: float = 1.0,
        lambda_mse: float = 0.5,
        epsilon_phys: float = 1e-25,
        rel_cap: Optional[float] = None,
        time_edge_gain: float = 1.0,
        device=None,
    ):
        super().__init__()

        if not isinstance(manifest, dict):
            raise TypeError("[AdaptiveStiffLoss] manifest must be a dict")
        if "per_key_stats" not in manifest or "normalization_methods" not in manifest:
            raise KeyError("[AdaptiveStiffLoss] manifest must contain 'per_key_stats' and 'normalization_methods'")

        if not species_keys:
            raise RuntimeError("[AdaptiveStiffLoss] species_keys is empty")

        self.manifest = manifest
        self.species_keys = list(species_keys)
        self.lambda_phys = float(lambda_phys)
        self.lambda_mse  = float(lambda_mse)
        self.eps_phys    = float(epsilon_phys)
        self.rel_cap     = float(rel_cap) if rel_cap is not None else None
        self.time_edge_gain = float(time_edge_gain)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        per_key_stats = manifest["per_key_stats"]
        norm_methods  = manifest["normalization_methods"]
        if not isinstance(per_key_stats, dict) or not isinstance(norm_methods, dict):
            raise TypeError("[AdaptiveStiffLoss] 'per_key_stats' and 'normalization_methods' must both be dicts.")

        # Build fast-path parameters and species weights
        log_scales: list[float] = []
        log_biases: list[float] = []
        log_mins: list[float] = []
        log_maxs: list[float] = []

        for key in self.species_keys:
            if key not in per_key_stats:
                raise KeyError(f"[AdaptiveStiffLoss] Stats for '{key}' not found in manifest['per_key_stats']")
            if key not in norm_methods:
                raise KeyError(f"[AdaptiveStiffLoss] Method for '{key}' not found in manifest['normalization_methods']")

            stats  = per_key_stats[key]
            method = str(norm_methods[key]).strip().lower()

            if method == "log-standard":
                if "log_mean" not in stats or "log_std" not in stats:
                    raise KeyError(f"[AdaptiveStiffLoss] '{key}' missing log_mean/log_std for log-standard")
                log_mean = float(stats["log_mean"])
                log_std  = float(stats["log_std"])
                if not (math.isfinite(log_mean) and math.isfinite(log_std) and log_std > 0):
                    raise ValueError(f"[AdaptiveStiffLoss] invalid log_mean/log_std for '{key}'")

                # For fast path:
                #   log10(x) = log_mean + log_std * z
                log_bias  = log_mean
                log_scale = log_std
                # For species weighting (log-range), prefer provided min/max if present:
                lo = float(stats.get("log_min", log_mean - 3.0 * log_std))
                hi = float(stats.get("log_max", log_mean + 3.0 * log_std))

            elif method == "log-min-max":
                if "log_min" not in stats or "log_max" not in stats:
                    raise KeyError(f"[AdaptiveStiffLoss] '{key}' missing log_min/log_max for log-min-max")
                lo = float(stats["log_min"])
                hi = float(stats["log_max"])
                if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                    raise ValueError(f"[AdaptiveStiffLoss] invalid log_min/log_max for '{key}'")

                # For fast path:
                #   log10(x) = log_min + (log_max - log_min) * z
                log_bias  = lo
                log_scale = hi - lo

            else:
                raise RuntimeError(
                    f"[AdaptiveStiffLoss] Species '{key}' uses unsupported method '{method}'. "
                    f"Only 'log-standard' and 'log-min-max' are supported by the fast path."
                )

            if not (math.isfinite(log_bias) and math.isfinite(log_scale) and log_scale > 0.0):
                raise ValueError(f"[AdaptiveStiffLoss] non-finite/invalid bias/scale for '{key}': "
                                 f"bias={log_bias}, scale={log_scale}")

            log_scales.append(log_scale)
            log_biases.append(log_bias)
            log_mins.append(lo)
            log_maxs.append(hi)

        # Register constant buffers for vectorized fast path
        self.register_buffer("_ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        self.register_buffer("_log_scale_vec", torch.tensor(log_scales, dtype=torch.float32))
        self.register_buffer("_log_bias_vec",  torch.tensor(log_biases, dtype=torch.float32))

        # Species weights based on dynamic range (in log-space)
        log_min_t = torch.tensor(log_mins, dtype=torch.float32)
        log_max_t = torch.tensor(log_maxs, dtype=torch.float32)
        log_range = torch.clamp(log_max_t - log_min_t, min=1e-6)
        w_species = torch.sqrt(log_range)
        w_species = w_species / (w_species.mean() + 1e-12)
        w_species = torch.clamp(w_species, 0.5, 2.0)
        self.register_buffer("w_species", w_species)

        # Fallback helper for full denormalization (only used if shapes mismatch)
        from normalizer import NormalizationHelper
        self._norm_helper = NormalizationHelper(self.manifest, device=self.device)

        # Flag to allow fast path when last-dim matches #species
        self._fast_ok = True

        # Final validation
        for name, buf in [("_log_scale_vec", self._log_scale_vec),
                          ("_log_bias_vec",  self._log_bias_vec),
                          ("w_species",       self.w_species)]:
            if not torch.isfinite(buf).all():
                raise RuntimeError(f"[AdaptiveStiffLoss] Non-finite values in buffer '{name}'")

    # --------------------------- internals ---------------------------

    def _time_weights(self, t01: torch.Tensor) -> torch.Tensor:
        """U-shaped weights: returns `gain` at 0 and 1, and 1 at 0.5."""
        if self.time_edge_gain <= 1.0:
            return torch.ones_like(t01)
        return 1.0 + (self.time_edge_gain - 1.0) * (1.0 - 4.0 * t01 * (1.0 - t01))

    # --------------------------- forward ---------------------------

    def forward(
        self,
        pred_norm: torch.Tensor,
        true_norm: torch.Tensor,
        t_norm: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        # Promote to float32 for stable reductions
        pz = pred_norm.float()
        tz = true_norm.float()
        if pz.shape[-1] != tz.shape[-1]:
            raise ValueError(f"[AdaptiveStiffLoss] pred/true last dim mismatch: {pz.shape[-1]} vs {tz.shape[-1]}")

        # 1) MSE in normalized space (per-sample: average over species)
        mse_per_sample = (pz - tz).square().mean(dim=-1)  # shape = pred_norm.shape[:-1]

        # 2) Fractional |Δ| / (|true| + eps) in physical space
        if self._fast_ok and self._log_scale_vec.numel() == pz.shape[-1]:
            # Compute log10(true) and Δlog10 = (pz - tz) * scale
            lnt = tz * self._log_scale_vec + self._log_bias_vec         # [..., S]
            dlt = (pz - tz) * self._log_scale_vec                        # [..., S]

            # true_phys = 10^lnt ; pred_phys - true_phys = true_phys * (10^dlt - 1)
            true_phys = torch.exp(lnt * self._ln10)
            delta_factor = torch.exp(dlt * self._ln10) - 1.0

            numer = (true_phys * delta_factor.abs())                     # [..., S]
            denom = true_phys.abs() + self.eps_phys                      # [..., S]
            frac  = numer / denom
        else:
            # Fallback: compute in physical units via helper
            # Ensure helper is initialized on correct device for any internal tensors
            from normalizer import NormalizationHelper
            self._norm_helper = NormalizationHelper(self.manifest, device=pz.device)

            pred_phys = self._norm_helper.denormalize(pz, self.species_keys)
            true_phys = self._norm_helper.denormalize(tz, self.species_keys)
            frac = (pred_phys - true_phys).abs() / (true_phys.abs() + self.eps_phys)

        # Numerical safety + optional cap
        frac = torch.nan_to_num(frac, nan=0.0, posinf=1e6, neginf=0.0)
        if self.rel_cap is not None:
            frac = frac.clamp_max(self.rel_cap)

        # Apply species weights before reducing across species
        if self.w_species.numel() == frac.shape[-1]:
            frac = frac * self.w_species

        frac_per_sample = frac.mean(dim=-1)  # shape = pred_norm.shape[:-1]

        # 3) Optional time weights (expect shape broadcastable to per-sample losses)
        if t_norm is not None:
            t = t_norm
            if t.ndim == frac_per_sample.ndim + 1 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            # Broadcast to per-sample shape if needed
            while t.ndim < frac_per_sample.ndim:
                t = t.unsqueeze(-1)
            t01 = torch.clamp(t, 0.0, 1.0).to(dtype=frac_per_sample.dtype, device=frac_per_sample.device)
            wt = self._time_weights(t01)
            mse_per_sample  = mse_per_sample * wt
            frac_per_sample = frac_per_sample * wt

        # 4) Optional mask over per-sample elements (e.g., [B,K])
        if mask is not None:
            m = mask
            if m.ndim == frac_per_sample.ndim + 1 and m.shape[-1] == 1:
                m = m.squeeze(-1)
            while m.ndim < frac_per_sample.ndim:
                m = m.unsqueeze(-1)
            m = m.to(dtype=frac_per_sample.dtype, device=frac_per_sample.device)
            denom = m.sum().clamp_min(1.0)
            mse_scalar  = (mse_per_sample  * m).sum() / denom
            frac_scalar = (frac_per_sample * m).sum() / denom
        else:
            mse_scalar  = mse_per_sample.mean()
            frac_scalar = frac_per_sample.mean()

        total = self.lambda_mse * mse_scalar + self.lambda_phys * frac_scalar

        if return_components:
            return {
                "total": total,
                "mse": mse_scalar.detach(),
                "frac": frac_scalar.detach(),
            }
        return total

# --------------------------- Utilities ---------------------------

def _validate_batch(batch, batch_idx: int = 0) -> Tuple[int, int, int]:
    if len(batch) < 4:
        raise RuntimeError(f"Batch has {len(batch)} items, expected at least 4 (y_i, dt, y_j, g, [aux], [mask])")
    y_i, dt, y_j, g = batch[:4]
    if not (isinstance(y_i, torch.Tensor) and isinstance(dt, torch.Tensor)
            and isinstance(y_j, torch.Tensor) and isinstance(g, torch.Tensor)):
        raise RuntimeError("Batch must contain tensors for y_i, dt, y_j, g")

    if y_i.ndim != 2:
        raise RuntimeError(f"y_i must be [B,S], got {tuple(y_i.shape)}")
    if dt.ndim != 2:
        raise RuntimeError(f"dt_norm must be [B,K], got {tuple(dt.shape)}")
    if y_j.ndim != 3:
        raise RuntimeError(f"y_j must be [B,K,S], got {tuple(y_j.shape)}")
    if g.ndim != 2:
        raise RuntimeError(f"g must be [B,G], got {tuple(g.shape)}")

    B, S = y_i.shape
    Bdt, K = dt.shape
    By, Ky, Sy = y_j.shape
    Bg, _ = g.shape

    if not (B == Bdt == By == Bg):
        raise RuntimeError(f"Batch size mismatch in batch {batch_idx}: y_i={B}, dt={Bdt}, y_j={By}, g={Bg}")
    if not (K == Ky):
        raise RuntimeError(f"K mismatch in batch {batch_idx}: dt={K}, y_j={Ky}")
    if not (S == Sy):
        raise RuntimeError(f"S mismatch in batch {batch_idx}: y_i={S}, y_j={Sy}")

    if len(batch) >= 6:
        mask = batch[5]
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.shape != (B, K):
                raise RuntimeError(f"mask must be [B,K]; got {None if mask is None else tuple(mask.shape)}")

    return B, K, S


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    clip: float = MAX_RESIDUAL,
    warn_threshold: float = 1e6,
) -> torch.Tensor:
    """Robust MSE in normalized space; float32 reductions; supports [B,K] mask."""
    pred32 = pred.float()
    target32 = target.float()

    resid = pred32 - target32
    if not torch.isfinite(resid).all():
        print("[WARN] _masked_mse: non-finite residuals detected; replacing with zeros.")
        resid = torch.nan_to_num(resid, nan=0.0, posinf=clip, neginf=-clip)

    if clip is not None and clip > 0:
        resid = resid.clamp(-clip, clip)

    per_bk = resid.square().mean(dim=-1)

    if mask is not None:
        m = mask.to(dtype=per_bk.dtype, device=per_bk.device)
        denom = m.sum().clamp_min(1.0)
        loss = (per_bk * m).sum() / denom
    else:
        loss = per_bk.mean()

    if torch.isfinite(loss) and float(loss) > warn_threshold:
        print(f"[WARN] _masked_mse: unusually large MSE={float(loss):.3e}.")
    return loss


# --------------------------- Callbacks ---------------------------

class EpochSetterCallback(Callback):
    """Log TF prob and rollout horizon at start of each epoch."""
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: "AutoEncoderModule") -> None:
        epoch = trainer.current_epoch
        tf_p = pl_module._get_teacher_forcing_prob(epoch)
        H = pl_module._get_rollout_horizon(epoch)
        pl_module.log("tf_prob", float(tf_p), on_epoch=True, prog_bar=False)
        pl_module.log("rollout_horizon", float(H), on_epoch=True, prog_bar=False)


class EpochSummaryLineCallback(Callback):
    """
    One-line, aligned epoch summary with timer.

    Columns:
      EPOCH | train | val | ||g|| | ||p|| | lr | tf | H | time
    - Prints once per epoch (after val if present, else after train).
    - Uses scientific notation and fixed widths for stable alignment.
    - Shows elapsed epoch time as mm:ss or h:mm:ss.
    """
    def __init__(self):
        super().__init__()
        self._header_printed = False
        self._printed_epoch = -1
        self._t0 = None  # epoch start time

        # Fixed column widths for aligned output
        self._w_num  = 10  # train/val/||g||/||p||/lr widths
        self._w_tf   = 5
        self._w_H    = 3
        self._w_time = 8   # supports h:mm:ss

    # ---------------- internal helpers ----------------

    def _fmt_secs(self, seconds: float) -> str:
        if seconds is None or seconds < 0:
            return "--:--"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def _maybe_print_header(self):
        if self._header_printed:
            return
        # Header line (short but informative)
        # train/val are total losses; ||g||, ||p|| are L2 norms; lr is group 0 LR
        hdr = (
            "EPOCH | "
            f"{'train':>{self._w_num}} | "
            f"{'val':>{self._w_num}} | "
            f"{'||g||':>{self._w_num}} | "
            f"{'||p||':>{self._w_num}} | "
            f"{'lr':>{self._w_num}} | "
            f"{'tf':>{self._w_tf}} | "
            f"{'H':>{self._w_H}} | "
            f"{'time':>{self._w_time}}"
        )
        sep = (
            "----- | "
            + " | ".join([
                "-" * self._w_num,  # train
                "-" * self._w_num,  # val
                "-" * self._w_num,  # ||g||
                "-" * self._w_num,  # ||p||
                "-" * self._w_num,  # lr
                "-" * self._w_tf,   # tf
                "-" * self._w_H,    # H
                "-" * self._w_time  # time
            ])
        )
        print(hdr, flush=True)
        print(sep, flush=True)
        self._header_printed = True

    def _emit(self, trainer: pl.Trainer, pl_module: "AutoEncoderModule", elapsed_s: float) -> None:
        if trainer.sanity_checking:
            return

        self._maybe_print_header()

        epoch = trainer.current_epoch
        m = trainer.callback_metrics

        def _getf(key: str) -> float:
            if key not in m:
                return float("nan")
            v = m[key]
            if torch.is_tensor(v):
                v = v.detach().float().cpu().item()
            return float(v)

        # Prefer train_total; fall back to train_mse; then module cache
        train_v = _getf("train_total")
        if math.isnan(train_v):
            train_v = _getf("train_mse")
        if math.isnan(train_v):
            train_v = _getf("train")
        if math.isnan(train_v):
            train_v = float(getattr(pl_module, "_last_train_metric", float("nan")))

        val_v = _getf("val")  # total validation loss you log

        tf_prob = pl_module._get_teacher_forcing_prob(epoch)
        H       = pl_module._get_rollout_horizon(epoch)

        # LR (group 0)
        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")

        # Norms
        with torch.no_grad():
            p2 = 0.0
            for p in pl_module.model.parameters():
                if p is None:
                    continue
                ps = p.detach().float().pow(2).sum()
                if torch.isfinite(ps):
                    p2 += float(ps.item())
            param_norm = math.sqrt(p2)

        grad_norm = float(getattr(pl_module, "_last_grad_norm", float("nan")))
        tstr = self._fmt_secs(elapsed_s)

        line = (
            f"E{epoch:04d} | "
            f"{train_v:>{self._w_num}.3e} | "
            f"{val_v:>{self._w_num}.3e} | "
            f"{grad_norm:>{self._w_num}.3e} | "
            f"{param_norm:>{self._w_num}.3e} | "
            f"{lr:>{self._w_num}.3e} | "
            f"{tf_prob:>{self._w_tf}.3f} | "
            f"{H:>{self._w_H}d} | "
            f"{tstr:>{self._w_time}}"
        )
        print(line, flush=True)

    # ---------------- Lightning hooks ----------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: "AutoEncoderModule") -> None:
        # Start timer for this epoch
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "AutoEncoderModule") -> None:
        # If there's NO validation, emit here; else let validation hook print
        nval = 0
        try:
            nval = int(trainer.num_val_batches)
        except Exception:
            try:
                nval = sum(int(x) for x in trainer.num_val_batches)
            except Exception:
                nval = 0
        if nval == 0:
            elapsed = (time.perf_counter() - self._t0) if self._t0 is not None else float("nan")
            self._emit(trainer, pl_module, elapsed)
            self._printed_epoch = trainer.current_epoch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: "AutoEncoderModule") -> None:
        # Print after validation (train metrics exist by now). Avoid double print.
        if self._printed_epoch == trainer.current_epoch:
            return
        elapsed = (time.perf_counter() - self._t0) if self._t0 is not None else float("nan")
        self._emit(trainer, pl_module, elapsed)
        self._printed_epoch = trainer.current_epoch


# --------------------------- Lightning Module ---------------------------

class AutoEncoderModule(LightningModule):
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], work_dir: Union[str, Path]):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)

        # Training config
        tcfg = cfg.get("training", {})
        aux = tcfg.get("auxiliary_losses", {})

        self.learning_rate = float(tcfg.get("lr", 1e-3))
        self.min_lr = float(tcfg.get("min_lr", 1e-6))
        self.weight_decay = float(tcfg.get("weight_decay", 0.0))
        self.grad_clip = float(tcfg.get("gradient_clip", 1.0))
        self.epochs = int(tcfg.get("epochs", 100))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", 10))

        # Rollout controls
        self.rollout_enabled = bool(aux.get("rollout_enabled", False))
        # NOTE: rollout_weight acts as a multiplier applied progressively with warmup
        self.rollout_weight = float(aux.get("rollout_weight", 0.0))
        self.max_rollout_horizon = int(aux.get("rollout_horizon", 4))
        self.rollout_warmup_epochs = int(aux.get("rollout_warmup_epochs", 30))

        # Teacher forcing schedule
        tf_cfg = aux.get("rollout_teacher_forcing", {})
        self.tf_mode = str(tf_cfg.get("mode", "none")).lower()  # "none" | "linear"
        self.tf_start_p = float(tf_cfg.get("start_p", 0.0))
        self.tf_end_p = float(tf_cfg.get("end_p", 0.0))
        self.tf_end_epoch = int(tf_cfg.get("end_epoch", 0))

        # Optional semigroup (off by default)
        sg = aux.get("semigroup", {})
        self.semigroup_enabled = bool(sg.get("enabled", False))
        self.semigroup_weight = float(sg.get("weight", 0.0))
        self.semigroup_warmup_epochs = int(sg.get("warmup_epochs", 0))

        # Loss (reverted to file-based manifest)
        self._setup_adaptive_stiff_loss()

        # Grad-norm cache for summary
        self._last_grad_norm: float = float("nan")
        self._last_train_metric: float = float("nan")

    # ---------- manifest + loss setup (file-based) ----------

    def _setup_adaptive_stiff_loss(self) -> None:
        """Load manifest from processed folder; build AdaptiveStiffLoss."""
        tcfg = self.cfg.get("training", {})
        loss_mode = tcfg.get("loss_mode", "adaptive_stiff")
        if loss_mode != "adaptive_stiff":
            self.criterion = None
            self.use_adaptive_stiff = False
            return

        paths = self.cfg.get("paths", {})
        try:
            processed_dir = Path(paths["processed_data_dir"]).expanduser().resolve()
        except Exception as e:
            raise RuntimeError("cfg['paths']['processed_data_dir'] is missing or invalid") from e

        norm_path = processed_dir / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization.json not found at: {norm_path}. "
                "Run preprocessing or point cfg.paths.processed_data_dir at a valid folder."
            )
        with open(norm_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        need_dt = bool(self.cfg.get("dataset", {}).get("require_dt_stats", False))
        dt = manifest.get("dt", None)
        has_dt = (isinstance(dt, dict) and ("log_min" in dt) and ("log_max" in dt))
        if not has_dt:
            msg = ("normalization.json is missing a valid 'dt' block "
                   "(expects keys: log_min, log_max).")
            if need_dt:
                raise ValueError(msg)
            else:
                print(f"[WARN] {msg} Proceeding without dt because require_dt_stats=False.")

        data_cfg = self.cfg.get("data", {})
        species_keys = (data_cfg.get("target_species")
                        or manifest.get("meta", {}).get("target_species")
                        or manifest.get("meta", {}).get("species_variables"))
        if not species_keys:
            raise ValueError(
                "Could not determine species_keys. "
                "Set cfg.data.target_species or ensure manifest.meta.species_variables exists."
            )
        species_keys = list(species_keys)

        pks = set((manifest.get("per_key_stats") or {}).keys())
        missing = [k for k in species_keys if k not in pks]
        if missing:
            raise RuntimeError(
                "Some species in target list are missing from manifest.per_key_stats: "
                + ", ".join(missing)
            )

        loss_cfg = tcfg.get("adaptive_stiff_loss", {})
        rel_cap = loss_cfg.get("rel_cap", tcfg.get("loss_rel_cap", None))
        time_edge_gain = float(loss_cfg.get("time_edge_gain", 2.0))

        self.criterion = AdaptiveStiffLoss(
            manifest=manifest,
            species_keys=species_keys,
            lambda_phys=float(loss_cfg.get("lambda_phys", 1.0)),
            lambda_mse=float(loss_cfg.get("lambda_mse", 0.1)),
            epsilon_phys=float(loss_cfg.get("epsilon_phys", 1e-20)),
            rel_cap=(float(rel_cap) if rel_cap is not None else None),
            time_edge_gain=time_edge_gain,
            device=self.device,
        )
        self.use_adaptive_stiff = True

        print(
            "Using AdaptiveStiffLoss (from file): "
            f"lambda_phys={loss_cfg.get('lambda_phys', 1.0)}, "
            f"lambda_mse={loss_cfg.get('lambda_mse', 0.1)}, "
            f"epsilon_phys={loss_cfg.get('epsilon_phys', 1e-20)}, "
            f"rel_cap={rel_cap}, "
            f"time_edge_gain={time_edge_gain}, "
            f"dt_present={has_dt}"
        )

    def configure_optimizers(self):
        # Try CUDA fused AdamW (Hopper: typically +5–15% step-time). Fallback to standard AdamW if unsupported.
        try:
            opt = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                fused=True,  # will raise TypeError on older/cu-less builds
            )
        except TypeError:
            opt = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        warm = LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, self.warmup_epochs),
        )
        cos = CosineAnnealingLR(
            opt,
            T_max=max(1, self.epochs - self.warmup_epochs),
            eta_min=self.min_lr,
        )
        sch = SequentialLR(opt, schedulers=[warm, cos], milestones=[self.warmup_epochs])

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }

    # ------------------ Schedules ------------------

    def _get_teacher_forcing_prob(self, epoch: int) -> float:
        if self.tf_mode == "none":
            return 0.0
        if self.tf_mode == "linear":
            if epoch >= self.tf_end_epoch:
                return self.tf_end_p
            t = max(0.0, min(1.0, epoch / max(1, self.tf_end_epoch)))
            return float((1 - t) * self.tf_start_p + t * self.tf_end_p)
        return 0.0

    def _get_rollout_horizon(self, epoch: int) -> int:
        if self.max_rollout_horizon <= 1:
            return 1
        ramp = max(1, self.rollout_warmup_epochs)
        frac = max(0.0, min(1.0, epoch / ramp))
        return int(max(1, round(frac * self.max_rollout_horizon)))

    # ---------- OFFSETS->STEP SIZES (normalized, via physical space) ----------

    def _offsets_to_steps_norm(self, dt_norm_offsets: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized offsets (from anchor) → normalized per-step increments.
        Converts in physical space; then re-normalizes increments.
        """
        d = getattr(self.model, "dynamics", None)
        if d is None or not hasattr(d, "dt_log_min") or not hasattr(d, "dt_log_max"):
            x = dt_norm_offsets
            return torch.diff(torch.cat([torch.zeros_like(x[:, :1]), x], dim=1), dim=1).clamp(0.0, 1.0)

        log_min = float(getattr(d, "dt_log_min").item() if torch.is_tensor(d.dt_log_min) else d.dt_log_min)
        log_max = float(getattr(d, "dt_log_max").item() if torch.is_tensor(d.dt_log_max) else d.dt_log_max)
        rng = max(1e-12, (log_max - log_min))
        ln10 = math.log(10.0)

        dt_offs_phys = torch.exp((log_min + dt_norm_offsets.clamp(0.0, 1.0) * rng) * ln10)  # [B,K]
        steps_phys = torch.cat([dt_offs_phys[:, :1], dt_offs_phys[:, 1:] - dt_offs_phys[:, :-1]], dim=1)
        tiny = torch.finfo(steps_phys.dtype).tiny
        steps_norm = (torch.log10(steps_phys.clamp_min(tiny)) - log_min) / rng
        return steps_norm.clamp(0.0, 1.0)

    def _compute_rollout_loss_unified(
            self,
            y_i: torch.Tensor,
            g: torch.Tensor,
            y_j: torch.Tensor,
            dt_offsets: torch.Tensor,
            mask: Optional[torch.Tensor],
            H: int,
            tf_prob: float,
    ) -> Optional[torch.Tensor]:
        """
        Unified rollout loss:
          - tf_prob == 0: no teacher forcing. We do a latent rollout using
            model.rollout_vectorized(...) and we REUSE the cached latent state
            and FiLM affine from model.forward() in this same batch, so we do NOT
            re-encode y_i.
          - tf_prob  > 0: teacher forcing. We run a per-step loop that can
            mix ground truth vs model predictions. FiLM params are computed once.

        Returns:
            scalar loss (Tensor) or None (if horizon < 2).
        """
        B, K_avail, _ = y_j.shape
        H = min(H, K_avail)
        if H < 2:
            return None

        # Convert absolute offsets [B,K_abs] -> incremental step sizes [B,H]
        dt_steps_norm = self._offsets_to_steps_norm(dt_offsets[:, :H])  # [B,H]

        # -------- branch: no teacher forcing --------
        if tf_prob < 1e-8:
            cached_z0 = getattr(self.model, "_cached_rollout_z0", None)
            cached_film = getattr(self.model, "_cached_rollout_film", None)

            # rollout_vectorized now takes caches to avoid double encoding
            y_pred = self.model.rollout_vectorized(
                y0=y_i,
                g=g,
                dt_steps_norm=dt_steps_norm,
                z0=cached_z0,
                film_cache=cached_film,
            )  # [B, H, S]

            y_tgt = y_j[:, :H, :]  # [B, H, S]
            rmask = mask[:, :H] if mask is not None else None  # [B, H]

            if self.use_adaptive_stiff and self.criterion is not None:
                return self.criterion(
                    pred_norm=y_pred,
                    true_norm=y_tgt,
                    t_norm=dt_offsets[:, :H],  # absolute normalized offsets
                    mask=rmask,
                    return_components=False,
                )

            return _masked_mse(y_pred, y_tgt, rmask)

        # -------- branch: teacher forcing (>0) --------
        use_film = (
                bool(getattr(self.model, "decoder_condition_on_g", False))
                and hasattr(self.model, "film")
                and (self.model.film is not None)
        )

        if use_film:
            gb = self.model.film.proj(g)  # [B, 2Z]
            gamma, beta = gb.chunk(2, dim=-1)  # [B,Z], [B,Z]
            alpha = float(self.model.film.alpha)

            def apply_film(z_lat: torch.Tensor) -> torch.Tensor:
                return (1.0 + alpha * torch.tanh(gamma)) * z_lat + beta
        else:
            def apply_film(z_lat: torch.Tensor) -> torch.Tensor:
                return z_lat

        preds = []
        y_curr = y_i

        use_delta = bool(getattr(self.model, "predict_logit_delta", False))
        if use_delta:
            base = y_curr if (self.model.S_out == self.model.S_in) else y_curr.index_select(1, self.model.target_idx)
            cur_logp = self.model._denorm_to_logp(base)  # [B,S_out]
        else:
            cur_logp = None

        for k in range(H):
            # Encode current state, step latent one dt, decode logits
            if hasattr(self.model, "encode") and hasattr(self.model, "decode"):
                z = self.model.encode(y_curr, g)  # [B,Z]
                z = self.model.dynamics.step(z, dt_steps_norm[:, k], g)  # [B,Z]
                if use_film:
                    z = apply_film(z)
                logits = self.model.decoder(z)  # [B,S]
            else:
                enc_out = self.model.encoder(y_curr, g)
                z = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
                z = self.model.dynamics.step(z, dt_steps_norm[:, k], g)
                if use_film:
                    z = apply_film(z)
                logits = self.model.decoder(z)  # [B,S]

            if use_delta:
                log_q = F.log_softmax(logits, dim=-1).float()
                combined = cur_logp + log_q
                log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
                y_next = self.model._head_from_logprobs(log_p)  # normalized
            else:
                y_next = self.model._softmax_head_from_logits(logits)

            preds.append(y_next)

            # Teacher forcing mix for next step
            if tf_prob >= 1.0:
                y_curr = y_j[:, k, :]
            else:
                if self.training:
                    coin = (torch.rand(B, device=y_i.device) < tf_prob).unsqueeze(-1)  # [B,1]
                    y_curr = torch.where(coin, y_j[:, k, :], y_next.detach())
                else:
                    y_curr = y_next

            if use_delta:
                base = y_curr if (self.model.S_out == self.model.S_in) else y_curr.index_select(1,
                                                                                                self.model.target_idx)
                cur_logp = self.model._denorm_to_logp(base)

        y_pred = torch.stack(preds, dim=1)  # [B, H, S]
        y_tgt = y_j[:, :H, :]
        rmask = mask[:, :H] if mask is not None else None

        if self.use_adaptive_stiff and self.criterion is not None:
            return self.criterion(
                pred_norm=y_pred,
                true_norm=y_tgt,
                t_norm=dt_offsets[:, :H],
                mask=rmask,
                return_components=False,
            )
        return _masked_mse(y_pred, y_tgt, rmask)

    def _compute_semigroup_loss(self, y_i: torch.Tensor, g: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            d = self.model.dynamics
            log_min = float(d.dt_log_min.item())
            log_max = float(d.dt_log_max.item())
            rng = max(1e-9, (log_max - log_min))
        except Exception:
            return None

        B = y_i.shape[0]
        ln10 = math.log(10.0)

        # sample total dt in normalized space
        dt_tot_norm = torch.rand(B, device=y_i.device, dtype=torch.float32)  # [B] in [0,1)
        dt_tot_phys = torch.exp((log_min + dt_tot_norm * rng) * ln10)  # [B] physical

        # random split
        u = torch.rand(B, device=y_i.device, dtype=torch.float32)  # [B] in [0,1)
        dt_a_phys = u * dt_tot_phys  # [B]
        dt_b_phys = dt_tot_phys - dt_a_phys  # [B]

        # avoid log(0), renormalize each chunk, clamp to [0,1]
        tiny = torch.finfo(dt_tot_phys.dtype).tiny
        dt_a_norm = (torch.log(dt_a_phys.clamp_min(tiny)) / ln10 - log_min) / rng
        dt_b_norm = (torch.log(dt_b_phys.clamp_min(tiny)) / ln10 - log_min) / rng

        dt_a_norm = dt_a_norm.clamp_(0.0, 1.0)
        dt_b_norm = dt_b_norm.clamp_(0.0, 1.0)
        dt_tot_norm = dt_tot_norm.clamp_(0.0, 1.0)

        # match model precision so torch.cat(...) inside step() doesn't dtype-mismatch
        dt_a_norm = dt_a_norm.to(y_i.dtype)
        dt_b_norm = dt_b_norm.to(y_i.dtype)
        dt_tot_norm = dt_tot_norm.to(y_i.dtype)

        # semigroup consistency
        y_a = self.model.step(y_i, dt_a_norm, g)
        y_ab = self.model.step(y_a, dt_b_norm, g)
        y_total = self.model.step(y_i, dt_tot_norm, g)

        res = (y_ab - y_total).clamp(-MAX_RESIDUAL, MAX_RESIDUAL)
        return res.pow(2).mean()

    # ------------------ Hooks for gradient diagnostics ------------------

    def on_before_optimizer_step(self, optimizer) -> None:
        """
        Compute & cache L2 grad norm; optionally clip grads manually.

        Note: Manual clipping is used instead of Lightning's gradient_clip_val because:
          - Works reliably with fused optimizers (fused=True in AdamW)
          - Compatible with torch.compile on the model
          - Avoids hook interception issues in mixed precision training
        Lightning's gradient_clip_val is set for monitoring/logging only.
        """
        # ---- manual clipping (safe with fused and non-fused) ----
        if self.grad_clip is not None and float(self.grad_clip) > 0.0:
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
            except Exception as e:
                print(f"[WARN] manual grad clip failed: {e}")

        # ---- grad-norm diagnostics ----
        g2 = 0.0
        for p in self.model.parameters():
            if p is not None and p.grad is not None:
                ps = p.grad.detach().float().pow(2).sum()
                if torch.isfinite(ps):
                    g2 += float(ps.item())
        self._last_grad_norm = math.sqrt(g2) if g2 > 0.0 else 0.0

    def on_train_epoch_end(self):
        """Cache a stable training metric so epoch summary has a value even on epoch 0."""
        try:
            m = self.trainer.callback_metrics if hasattr(self, "trainer") and self.trainer is not None else {}
            v = None
            for k in ("train_total", "train_mse"):
                if k in m:
                    v = m[k]
                    break
            if v is not None:
                if torch.is_tensor(v):
                    v = v.detach().float().cpu().item()
                self._last_train_metric = float(v)
        except Exception:
            pass

    # ------------------ Training / Validation ------------------

    def training_step(self, batch, batch_idx: int):
        _validate_batch(batch, batch_idx)
        y_i, dt_offsets, y_j, g = batch[:4]  # [B,S], [B,K], [B,K,S], [B,G]
        mask = batch[5] if len(batch) >= 6 else None

        # Forward loss on all K offsets
        pred = self.model(y_i, dt_offsets, g)  # [B,K,S] normalized
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite predictions in training batch {batch_idx}")

        if self.use_adaptive_stiff and self.criterion is not None:
            comp = self.criterion(
                pred_norm=pred,
                true_norm=y_j,
                t_norm=dt_offsets,  # [B,K] absolute normalized offsets
                mask=mask,  # [B,K] validity mask
                return_components=True,
            )
            forward_loss = comp["total"].float()
            self.log("train_mse_norm", comp["mse"].detach(), on_step=False, on_epoch=True, logger=True)
            self.log("train_frac_phys", comp["frac"].detach(), on_step=False, on_epoch=True, logger=True)
            self.log("train", forward_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            forward_loss = _masked_mse(pred, y_j, mask).float()
            self.log("train", forward_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # KL term (only meaningful in VAE mode)
        beta_kl = float(self.cfg.get("training", {}).get("beta_kl", 0.0))
        kl = getattr(self.model, "kl_loss", None)
        if beta_kl != 0.0 and kl is not None:
            if not torch.is_tensor(kl):
                kl = torch.as_tensor(kl, device=pred.device, dtype=pred.dtype)
            if not torch.isfinite(kl):
                kl = torch.zeros((), device=pred.device, dtype=pred.dtype)
            self.log("train_kl", kl.detach(), on_step=False, on_epoch=True, logger=True)
            kl_term = beta_kl * kl
        else:
            kl_term = pred.new_zeros(())

        total = forward_loss + kl_term

        # Optional rollout loss (multi-step consistency)
        if self.rollout_enabled and self.rollout_weight > 0:
            H = self._get_rollout_horizon(self.current_epoch)
            tfp = self._get_teacher_forcing_prob(self.current_epoch)
            warm = min(1.0, self.current_epoch / max(1, self.rollout_warmup_epochs))
            eff_w = self.rollout_weight * warm
            if eff_w > 1e-8 and H > 0:
                rloss = self._compute_rollout_loss_unified(y_i, g, y_j, dt_offsets, mask, H, tfp)
                if rloss is not None and torch.isfinite(rloss):
                    total = total + eff_w * rloss
                    self.log("rollout_loss", rloss.detach(), on_step=False, on_epoch=True, logger=True)
                    self.log("rollout_warm_weight", eff_w, on_step=False, on_epoch=True, logger=True)

        # Optional semigroup loss
        if self.semigroup_enabled and self.semigroup_weight > 0:
            sg_warm = min(1.0, self.current_epoch / max(1, self.semigroup_warmup_epochs))
            eff_sg = self.semigroup_weight * sg_warm
            if eff_sg > 1e-8:
                sgl = self._compute_semigroup_loss(y_i, g)
                if sgl is not None and torch.isfinite(sgl):
                    total = total + eff_sg * sgl
                    self.log("semigroup_loss", sgl.detach(), on_step=False, on_epoch=True, logger=True)
                    self.log("semigroup_warm_weight", eff_sg, on_step=False, on_epoch=True, logger=True)

        if not torch.isfinite(total):
            print(f"[WARN] training_step: non-finite total loss at batch {batch_idx}.")
        elif float(total) > 1e6:
            print(f"[WARN] training_step: very large TOTAL={float(total):.3e} at batch {batch_idx}.")

        self.log("train_total", total.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ---- critical: drop caches so previous graphs can be freed ----
        if hasattr(self.model, "_cached_rollout_z0"):
            self.model._cached_rollout_z0 = None
        if hasattr(self.model, "_cached_rollout_film"):
            self.model._cached_rollout_film = None

        return total

    def validation_step(self, batch, batch_idx: int):
        _validate_batch(batch, batch_idx)
        y_i, dt_offsets, y_j, g = batch[:4]
        mask = batch[5] if len(batch) >= 6 else None

        pred = self.model(y_i, dt_offsets, g)
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite predictions in validation batch {batch_idx}")

        if self.use_adaptive_stiff and self.criterion is not None:
            comp = self.criterion(
                pred_norm=pred,
                true_norm=y_j,
                t_norm=dt_offsets,  # [B,K] absolute normalized offsets
                mask=mask,
                return_components=True,
            )
            loss = comp["total"].float()
            self.log("val_mse_norm", comp["mse"].detach(), on_step=False, on_epoch=True, logger=True)
            self.log("val_frac_phys", comp["frac"].detach(), on_step=False, on_epoch=True, logger=True)
        else:
            loss = _masked_mse(pred, y_j, mask).float()

        if not torch.isfinite(loss):
            print(f"[WARN] validation_step: non-finite loss at batch {batch_idx}.")
        elif float(loss) > 1e6:
            print(f"[WARN] validation_step: very large MSE={float(loss):.3e} at batch {batch_idx}.")

        self.log("val", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # --------------------------- External Trainer Wrapper ---------------------------

class Trainer:
    """Thin wrapper to construct Lightning objects and run training."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        work_dir: Union[str, Path],
        device: torch.device,
        logger: logging.Logger,
        optuna_trial: Optional[Any] = None,
        optuna_monitor: str = "val",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.logger = logger

        # Optuna pruning context
        self.optuna_trial = optuna_trial
        self.optuna_monitor = optuna_monitor

        # Ensure output directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> float:
        compile_cfg = (self.cfg.get("torch_compile") or {})
        if bool(compile_cfg.get("enable", False)):
            backend   = compile_cfg.get("backend", "inductor")
            mode      = compile_cfg.get("mode", "reduce-overhead")
            dynamic   = bool(compile_cfg.get("dynamic", False))
            fullgraph = bool(compile_cfg.get("fullgraph", False))
            try:
                self.model = torch.compile(
                    self.model,
                    backend=backend,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=fullgraph,
                )
                try:
                    self.logger.info(
                        f"torch.compile enabled: backend={backend}, mode={mode}, "
                        f"dynamic={dynamic}, fullgraph={fullgraph}"
                    )
                except Exception:
                    print(
                        f"[INFO] torch.compile enabled: backend={backend}, mode={mode}, "
                        f"dynamic={dynamic}, fullgraph={fullgraph}"
                    )
            except Exception as e:
                try:
                    self.logger.warning(f"torch.compile failed ({e}); continuing uncompiled.")
                except Exception:
                    print(f"[WARN] torch.compile failed ({e}); continuing uncompiled.")

        module = AutoEncoderModule(self.model, self.cfg, self.work_dir)

        lcfg = self.cfg.get("lightning", {})
        tcfg = self.cfg.get("training", {})
        benchmark = bool(lcfg.get("benchmark", True))

        precision = str(lcfg.get("precision", "bf16-mixed"))
        devices = lcfg.get("devices", 1)
        accum = int(lcfg.get("accumulate_grad_batches", 1))
        strategy = lcfg.get("strategy", "auto")
        num_sanity = int(lcfg.get("num_sanity_val_steps", 0))
        max_epochs = int(tcfg.get("epochs", 100))
        resume_ckpt = lcfg.get("resume_from", None)

        # Metric to optimize / checkpoint / early-stop on.
        monitor_metric = self.optuna_monitor  # e.g. "val"

        # Loggers
        csv_logger = CSVLogger(save_dir=str(self.work_dir), name="logs", version="csv")
        tb_logger = TensorBoardLogger(save_dir=str(self.work_dir), name="tb")

        # Checkpointing (supports model soup top-k)
        ms_cfg = tcfg.get("model_soup", {})
        save_top_k = int(ms_cfg.get("top_k", 1)) if bool(ms_cfg.get("enable", False)) else 1

        ckpt_cb = ModelCheckpoint(
            dirpath=str(self.work_dir / "checkpoints"),
            filename=f"epoch{{epoch:04d}}-{monitor_metric}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        )

        lr_cb = LearningRateMonitor(logging_interval="epoch")

        es_patience = int(lcfg.get("early_stop_patience", 0))
        es_cb = (
            EarlyStopping(monitor=monitor_metric, mode="min", patience=es_patience)
            if es_patience > 0
            else None
        )

        epoch_cb = EpochSetterCallback()
        summary_cb = EpochSummaryLineCallback()

        callbacks = [ckpt_cb, lr_cb, epoch_cb, summary_cb]

        # ---------- EMA (guarded) ----------
        ema_cfg = tcfg.get("ema", {})
        if bool(ema_cfg.get("enable", False)):
            if _EMA_CB is not None and isinstance(_EMA_CB, type) and issubclass(_EMA_CB, Callback):
                decay = float(ema_cfg.get("decay", 0.999))
                try:
                    callbacks.append(_EMA_CB(decay=decay))
                except TypeError:
                    try:
                        callbacks.append(_EMA_CB(decay=decay, every_n_steps=1))
                    except Exception:
                        print("[WARN] EMA: could not construct callback; skipping.")
            else:
                print("[WARN] EMA callback not compatible with this Lightning Trainer; skipping.")

        # ---------- SWA (guarded) ----------
        swa_cfg = tcfg.get("swa", {})
        if bool(swa_cfg.get("enable", False)):
            if _SWA_CB is not None and isinstance(_SWA_CB, type) and issubclass(_SWA_CB, Callback):
                swa_lrs = swa_cfg.get("swa_lrs", None)
                epoch_start = float(swa_cfg.get("epoch_start", 0.8))
                callbacks.append(_SWA_CB(swa_lrs=swa_lrs, swa_epoch_start=epoch_start))
            else:
                print("[WARN] SWA callback not compatible with this Lightning Trainer; skipping.")

        # ---------- Optuna pruning (guarded exactly the same way) ----------
        if self.optuna_trial is not None:
            try:
                from optuna.integration import PyTorchLightningPruningCallback as _PruneCB  # type: ignore
                if isinstance(_PruneCB, type) and issubclass(_PruneCB, Callback):
                    try:
                        callbacks.append(
                            _PruneCB(
                                self.optuna_trial,
                                monitor=monitor_metric,
                            )
                        )
                    except Exception as e:
                        try:
                            self.logger.warning(f"Optuna pruning callback could not be constructed: {e}")
                        except Exception:
                            print(f"[WARN] Optuna pruning callback could not be constructed: {e}")
                else:
                    print("[WARN] Optuna pruning callback incompatible with this Lightning Trainer; skipping.")
            except Exception as e:
                try:
                    self.logger.warning(f"Optuna pruning callback not attached: {e}")
                except Exception:
                    print(f"[WARN] Optuna pruning callback not attached: {e}")

        # Early stopping goes last
        if es_cb is not None:
            callbacks.append(es_cb)

        trainer = LightningTrainer(
            max_epochs=max_epochs,
            devices=devices,
            precision=precision,
            strategy=strategy,
            accumulate_grad_batches=accum,
            num_sanity_val_steps=num_sanity,
            logger=[csv_logger, tb_logger],
            callbacks=callbacks,
            gradient_clip_val=0.0,
            gradient_clip_algorithm="norm",
            benchmark=benchmark,
            enable_progress_bar=False,
            log_every_n_steps=999999999,
        )

        trainer.fit(
            module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=resume_ckpt,
        )

        # Export best model checkpoint to a clean state dict
        best_val = None
        if ckpt_cb.best_model_score is not None:
            best_val = float(ckpt_cb.best_model_score.detach().cpu().item())

        if ckpt_cb.best_model_path:
            state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", {})
            model_state = {
                k.replace("model.", "", 1): v
                for k, v in sd.items()
                if k.startswith("model.")
            }
            torch.save(
                {"model": model_state, "config": self.cfg},
                self.work_dir / "best_model.pt",
            )
            msg = (
                f"Exported best_model.pt (metric {monitor_metric}={best_val:.4e})"
                if best_val is not None
                else "Saved best_model.pt (no best_val)"
            )
            try:
                self.logger.info(msg)
            except Exception:
                print(f"[INFO] {msg}")

        # Optional model soup across top-k checkpoints
        if bool(ms_cfg.get("enable", False)) and save_top_k > 1:
            try:
                best_k = ckpt_cb.best_k_models  # dict: path -> score
                ckpt_paths = list(best_k.keys())
                if len(ckpt_paths) >= 2:
                    avg_state: Dict[str, torch.Tensor] = {}
                    n = 0
                    for pth in ckpt_paths:
                        st = torch.load(pth, map_location="cpu", weights_only=False)
                        sd = st.get("state_dict", {})
                        model_sd = {
                            k.replace("model.", "", 1): v
                            for k, v in sd.items()
                            if k.startswith("model.")
                        }
                        if not avg_state:
                            for k, v in model_sd.items():
                                if torch.is_tensor(v):
                                    avg_state[k] = v.detach().clone().float()
                            n = 1
                        else:
                            for k, v in model_sd.items():
                                if torch.is_tensor(v) and k in avg_state:
                                    avg_state[k] += v.detach().clone().float()
                            n += 1
                    if n > 0:
                        for k in list(avg_state.keys()):
                            avg_state[k] /= float(n)
                        torch.save(
                            {"model": avg_state, "soup_members": ckpt_paths, "config": self.cfg},
                            self.work_dir / "soup_model.pt",
                        )
                        print(f"[ModelSoup] Saved soup_model.pt averaged from {n} checkpoints.")
            except Exception as e:
                print(f"[ModelSoup] Skipped due to error: {e}")

        return best_val if best_val is not None else float("inf")
