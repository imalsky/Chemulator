#!/usr/bin/env python3
"""
Lightning Trainer for CVL + Time-Warp Autoencoder
==================================================
(docstring unchanged - keeping your existing one)
"""
from __future__ import annotations

import json
import logging
import time
import math
import contextlib
import copy, gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer as LightningTrainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.tuner.tuning import Tuner as _PLTuner
    _TUNER_AVAILABLE = True
except Exception:
    try:
        from lightning.pytorch.tuner.tuning import Tuner as _PLTuner
        _TUNER_AVAILABLE = True
    except Exception:
        _PLTuner = None
        _TUNER_AVAILABLE = False

# Back-compat alias so older call-sites using `_PLTUNER` won't crash
_PLTUNER = _PLTuner

# --------------------------------------------------------------------------------------
# Loss: Combined (MSE in normalized space) + optional Tail-Huber
# --------------------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Primary stabilizer: MSE(pred_norm, target_norm).
    Optional Tail-Huber on standardized log10 space for trace species.
    """

    def __init__(
            self,
            *,
            tail_delta: float = 0.02,
            tail_weight: float = 0.0,
            tail_z_threshold: float = -1.0,
            time_weight_mode: str = "none",
    ):
        super().__init__()
        self.tail_delta = float(tail_delta)
        self.tail_weight = float(tail_weight)
        self.tail_z_threshold = float(tail_z_threshold)
        self.time_weight_mode = str(time_weight_mode)

    def _compute_time_weights(self, dt_norm: torch.Tensor) -> torch.Tensor:
        if self.time_weight_mode == "linear":
            return dt_norm.clamp(0, 1)
        elif self.time_weight_mode == "sqrt":
            return torch.sqrt(dt_norm.clamp(0, 1))
        elif self.time_weight_mode == "square":
            x = dt_norm.clamp(0, 1)
            return x * x
        else:
            return torch.ones_like(dt_norm)

    def _apply_mask_and_reduce(self, loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            m = mask
            while m.ndim < loss.ndim:
                m = m.unsqueeze(-1)
            m = m.to(loss.dtype)
            denom = m.sum()
            if loss.ndim >= 3 and (m.shape[-1] == 1):
                denom = (denom * loss.shape[-1]).clamp_min(1)
            else:
                denom = denom.clamp_min(1)
            return (loss * m).sum() / denom
        return loss.mean()

    def forward(
            self,
            pred_norm: torch.Tensor,
            target_norm: torch.Tensor,
            dt_norm: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes:
          - MSE in normalized space (always)
          - Optional Tail-Huber on low-abundance tail (when tail_weight > 0)
        Accepts mask with shapes [B], [B,K], [B,K,1], or [B,K,S]; broadcasts safely.
        """
        # ---- sanitize and build finite mask ----
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
        target_norm = torch.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # expected shapes: [B,K,S] (K may be 1)
        if pred_norm.ndim != 3 or target_norm.ndim != 3:
            raise RuntimeError(f"Expected pred/target to be 3D [B,K,S]; "
                               f"got pred={list(pred_norm.shape)}, target={list(target_norm.shape)}")

        B, K, S = pred_norm.shape
        finite = torch.isfinite(pred_norm) & torch.isfinite(target_norm)

        # ---- coerce user/loader mask to be broadcastable to [B,K,S] ----
        if mask is not None:
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=pred_norm.device)

            m = m.to(torch.bool)

            # Allow [B], [B,K], [B,K,1], or [B,K,S]
            if m.ndim == 1 and m.shape[0] == B:
                # [B] -> [B,1,1]
                m = m.view(B, 1, 1)
            elif m.ndim == 2 and m.shape[0] == B:
                # [B,K] -> [B,K,1]
                if m.shape[1] != K:
                    raise RuntimeError(f"2D mask has shape {list(m.shape)} but K={K}")
                m = m.unsqueeze(-1)
            elif m.ndim == 3:
                # [B,*,*] cases
                if list(m.shape) == [B, K, S]:
                    pass  # already perfect
                elif m.shape[0] == B and m.shape[1] == K and m.shape[2] == 1:
                    pass  # [B,K,1] ok
                elif m.shape[0] == B and m.shape[1] == 1 and m.shape[2] == 1:
                    # [B,1,1] ok
                    pass
                elif m.shape[0] == B and m.shape[1] == K and m.shape[2] == K:
                    # Rare bug: time mask repeated along last dim -> collapse to [B,K,1]
                    m = m[..., :1]
                else:
                    raise RuntimeError(f"3D mask {list(m.shape)} not compatible with [B,K,S]=[{B},{K},{S}]")
            else:
                raise RuntimeError(f"Mask ndim={m.ndim} not supported; provide [B], [B,K], [B,K,1], or [B,K,S]")

            # Now broadcast to data shape and AND with finite
            m = m.expand(B, K, 1) if m.shape[-1] == 1 else m
            mask = m & finite
        else:
            mask = finite

        # ---- core MSE per element ----
        r = pred_norm - target_norm
        mse = r.pow(2)

        # ---- optional Tail-Huber on low-abundance tail (target already in log-standard) ----
        tail_huber = None
        if self.tail_weight > 0.0:
            # target_norm is standardized log10 if you used "log-standard" normalization.
            z = target_norm  # standardized log10; lower = rarer species/trace tail
            tail_mask = (z <= self.tail_z_threshold)

            abs_r = r.abs()
            delta = self.tail_delta

            quad = 0.5 * (abs_r.clamp(max=delta) ** 2)
            lin = delta * (abs_r - 0.5 * delta).clamp_min(0.0)
            huber_full = torch.where(abs_r <= delta, quad, lin)

            # Apply tail mask and reduce using the same masking rule
            huber_tail = huber_full.masked_fill(~tail_mask, 0.0)
            tail_huber = self._apply_mask_and_reduce(huber_tail, mask)

        # ---- optional time weighting w(dt_norm) before reduction ----
        if (dt_norm is not None) and (self.time_weight_mode != "none"):
            # Accept [B,K] or [B,K,1]
            if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                dtw = dt_norm.squeeze(-1)  # [B,K]
            elif dt_norm.ndim == 2:
                dtw = dt_norm
            else:
                raise RuntimeError(f"dt_norm must be [B,K] or [B,K,1]; got {list(dt_norm.shape)}")

            dtw = torch.nan_to_num(dtw, nan=0.0, posinf=0.0, neginf=0.0)
            w = self._compute_time_weights(dtw)  # [B,K]
            w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
            w = w.unsqueeze(-1)  # [B,K,1]
            mse = mse * w  # broadcast over species

        # ---- masked reduction ----
        mse_reduced = self._apply_mask_and_reduce(mse, mask)

        total = mse_reduced
        if tail_huber is not None:
            total = total + self.tail_weight * tail_huber

        out = {"total": total, "mse": mse_reduced}
        if tail_huber is not None:
            out["tail_huber"] = tail_huber
        return out


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------

class EpochSetterCallback(Callback):
    """Log TF prob and rollout horizon each epoch (log-only; no annealing here)."""

    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        try:
            epoch = int(trainer.current_epoch)
            tf_p = float(pl_module._get_teacher_forcing_prob(epoch))
            H = int(pl_module._get_rollout_horizon(epoch))
            pl_module.log("tf_prob", tf_p, on_epoch=True, prog_bar=False)
            pl_module.log("rollout_horizon", H, on_epoch=True, prog_bar=False)
        except Exception:
            pass


class TimeWarpAnnealCallback(Callback):
    """Linearly anneal model.dynamics.timewarp.smax from start -> end by end_epoch."""

    def __init__(self, start: float, end: float, end_epoch: int):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.end_epoch = int(end_epoch)

    def on_train_epoch_start(self, trainer, pl_module):
        try:
            epoch = trainer.current_epoch
            if self.end_epoch <= 0:
                smax = self.end
            else:
                t = min(max(epoch, 0), self.end_epoch) / max(1, self.end_epoch)
                smax = self.start + (self.end - self.start) * t
            tw = getattr(pl_module.model, "dynamics", None)
            if tw is not None:
                tw = getattr(tw, "timewarp", None)
                if tw is not None and hasattr(tw, "smax"):
                    val = float(smax)
                    if isinstance(tw.smax, torch.Tensor):
                        tw.smax.data.fill_(val)
                    else:
                        setattr(tw, "smax", torch.tensor(val, device=pl_module.device))
                    pl_module.log("timewarp_smax", float(val), on_epoch=True, prog_bar=False)
        except Exception:
            pass


class GradNormLogger(Callback):
    """Accurate grad-norm logging after backward (does not clip)."""

    def on_after_backward(self, trainer, pl_module):
        try:
            g2 = 0.0
            for p in pl_module.model.parameters():
                if p.grad is not None:
                    v = p.grad.detach().float()
                    v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                    g2 += float((v * v).sum().cpu())
            pl_module.log("grad_norm", g2 ** 0.5, on_step=False, on_epoch=True, prog_bar=False)
        except Exception:
            pass


class NonFiniteParamGuard(Callback):
    """Detect and loudly log non-finite parameters AND gradients."""

    def __init__(self, py_logger: Optional[logging.Logger] = None):
        super().__init__()
        self._param_tripped = False
        self._grad_tripped = False  # ← NEW
        self.py_logger = py_logger or logging.getLogger("trainer")

    def on_after_backward(self, trainer, pl_module):  # ← NEW METHOD
        """Check gradients immediately after backward pass."""
        if self._grad_tripped:
            return

        try:
            with torch.no_grad():
                for name, p in pl_module.model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            self._grad_tripped = True
                            self.py_logger.error(
                                f"[GRAD GUARD] Epoch {trainer.current_epoch}, "
                                f"Batch {trainer.global_step}: "
                                f"First non-finite GRADIENT: {name}"
                            )
                            pl_module.log("grad_nonfinite_detected", 1.0,
                                          on_step=True, on_epoch=True, prog_bar=True)

                            # Log gradient statistics for debugging
                            try:
                                g = p.grad.detach().float()
                                n_nan = torch.isnan(g).sum().item()
                                n_inf = torch.isinf(g).sum().item()
                                self.py_logger.error(
                                    f"  → Gradient stats: {n_nan} NaNs, {n_inf} Infs"
                                )
                            except Exception:
                                pass
                            break
        except Exception as e:
            self.py_logger.warning(f"[GUARD] Gradient check failed: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check parameters after optimizer step."""
        if self._param_tripped:
            return

        try:
            with torch.no_grad():
                for name, p in pl_module.model.named_parameters():
                    if p is not None:
                        if not torch.isfinite(p).all():
                            self._param_tripped = True
                            self.py_logger.error(
                                f"[PARAM GUARD] Epoch {trainer.current_epoch}, "
                                f"Batch {batch_idx}: "
                                f"First non-finite PARAMETER: {name}"
                            )
                            pl_module.log("param_nonfinite", 1.0,
                                          on_step=True, on_epoch=True, prog_bar=True)

                            # Log parameter statistics
                            try:
                                n_nan = torch.isnan(p).sum().item()
                                n_inf = torch.isinf(p).sum().item()
                                self.py_logger.error(
                                    f"  → Param stats: {n_nan} NaNs, {n_inf} Infs"
                                )
                            except Exception:
                                pass
                            break
        except Exception as e:
            self.py_logger.warning(f"[GUARD] Parameter check failed: {e}")


class SimpleStatsCallback(Callback):
    """Log basic epoch stats: throughput, memory, norms."""

    def __init__(self, py_logger: Optional[logging.Logger] = None):
        super().__init__()
        self.py_logger = py_logger or logging.getLogger("trainer")
        self._epoch_start_time = None
        self._n_samples = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()
        self._n_samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                dt = batch[1]
                if isinstance(dt, torch.Tensor):
                    self._n_samples += int(dt.shape[0])
        except Exception:
            pass

    def on_train_epoch_end(self, trainer, pl_module):
        # ---- Throughput ----
        now = time.time()
        start = self._epoch_start_time or now
        elapsed = max(now - start, 1e-12)
        throughput = float(self._n_samples) / elapsed

        # ---- Parameter norm (safe, finite-masked) ----
        with torch.no_grad():
            device = pl_module.device
            p_sq = torch.tensor(0.0, device=device)
            for p in pl_module.model.parameters():
                if p is None:
                    continue
                p32 = p.detach().float()
                p32 = torch.where(torch.isfinite(p32), p32, torch.zeros_like(p32))
                p_sq += (p32 * p32).sum()
            p_norm = float(torch.sqrt(p_sq).detach().cpu())

        # ---- Gradient norm: use aggregated metric from GradNormLogger ----
        cm = trainer.callback_metrics
        g_norm_val = cm.get("grad_norm", None)
        if isinstance(g_norm_val, torch.Tensor):
            try:
                g_norm = float(g_norm_val.detach().cpu())
            except Exception:
                g_norm = float("nan")
        elif g_norm_val is None:
            # No epoch aggregation available (e.g., no backward ran); avoid misleading zeros.
            g_norm = float("nan")
        else:
            g_norm = float(g_norm_val)

        # ---- LR & memory ----
        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")

        max_mem = 0.0
        if torch.cuda.is_available():
            try:
                max_mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
            except Exception:
                max_mem = 0.0

        # ---- Losses from callback metrics (already reduced by Lightning) ----
        train_loss = float(cm.get("train_loss", float("nan")))
        val_loss = float(cm.get("val_loss", float("nan")))

        # ---- Log line ----
        self.py_logger.info(
            f"Epoch {trainer.current_epoch:3d} | "
            f"train={train_loss:.3e} val={val_loss:.3e} | "
            f"lr={lr:.2e} | "
            f"‖p‖={p_norm:.2e} ‖∇‖={g_norm:.2e} | "
            f"{throughput:.0f} samples/s | "
            f"GPU {max_mem:.1f}GB"
        )

        # ---- Metrics to Logger ----
        pl_module.log("throughput", throughput, on_epoch=True, prog_bar=False)
        pl_module.log("param_norm", p_norm, on_epoch=True, prog_bar=False)
        pl_module.log("grad_norm_epoch_end", g_norm, on_epoch=True, prog_bar=False)
        pl_module.log("gpu_mem_gb", max_mem, on_epoch=True, prog_bar=False)


# --------------------------------------------------------------------------------------
# Lightning Module
# --------------------------------------------------------------------------------------

class AutoEncoderModule(LightningModule):
    """Lightning wrapper around CVL + TimeWarp model."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], work_dir: Union[str, Path]):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.py_logger = logging.getLogger(__name__)

        tcfg = cfg.get("training", {})
        self.learning_rate = float(tcfg.get("lr", 1e-4))
        self.min_lr = float(tcfg.get("min_lr", 1e-6))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))
        self.grad_clip = float(tcfg.get("gradient_clip", 1.0))
        self.epochs = int(tcfg.get("epochs", 100))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", 10))

        lcfg = tcfg.get("loss", {}) or {}
        th = lcfg.get("tail_huber", {}) or {}
        self.criterion = CombinedLoss(
            tail_delta=float(th.get("delta", 0.02)),
            tail_weight=float(th.get("weight", 0.0)),
            tail_z_threshold=float(th.get("z_threshold", -1.0)),
            time_weight_mode=str(lcfg.get("time_weight_mode", "none")),
        )

        aux = tcfg.get("auxiliary_losses", {}) or {}
        self.rollout_enabled = bool(aux.get("rollout_enabled", True))
        self.rollout_weight = float(aux.get("rollout_weight", 0.5))
        self.max_rollout_horizon = int(aux.get("rollout_horizon", 8))
        self.semigroup_enabled = bool(aux.get("semigroup_enabled", True))
        self.semigroup_weight = float(aux.get("semigroup_weight", 0.1))

        # Stability knobs for early epochs
        self.rollout_warmup_epochs = int(aux.get("rollout_warmup_epochs", 10))
        self.rollout_fp32_epochs   = int(aux.get("rollout_fp32_epochs", 2))
        self.semigroup_warmup_epochs = int(aux.get("semigroup_warmup_epochs", 10))

        tf_cfg = aux.get("rollout_teacher_forcing", {}) or {}
        self.tf_mode = str(tf_cfg.get("mode", "linear"))
        self.tf_start = float(tf_cfg.get("start_p", 0.8))
        self.tf_end = float(tf_cfg.get("end_p", 0.0))
        self.tf_end_epoch = int(tf_cfg.get("end_epoch", 60))

        tw_cfg = tcfg.get("timewarp", {}) or {}
        self.timewarp_l1_weight = float(tw_cfg.get("l1_weight", 0.0))

        self._norm_manifest_path = None
        paths = cfg.get("paths", {}) or {}
        proc = paths.get("processed_data_dir", None)
        if proc is not None:
            for candidate in ["normalization.json", "manifest.json", "normalization/manifest.json"]:
                p = Path(proc) / candidate
                if p.exists():
                    self._norm_manifest_path = p
                    break

    def configure_optimizers(self):
        # Decide fused based on gradient clipping
        fused_ok = (float(getattr(self, "grad_clip", 0.0)) <= 0.0)
        try:
            if fused_ok:
                self.py_logger.info(
                    "Using fused AdamW (grad_clip=%.3f → fused=True).",
                    float(self.grad_clip)
                )
            else:
                self.py_logger.info(
                    "Gradient clipping enabled (grad_clip=%.3f) → forcing AdamW(fused=False).",
                    float(self.grad_clip)
                )
            opt = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                fused=fused_ok,  # <-- key change
            )
        except TypeError:
            # Older PyTorch without 'fused' kw; non-fused is used implicitly
            self.py_logger.info("AdamW(fused=...) not supported; using non-fused AdamW.")
            opt = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                foreach=False,
            )

        # --- keep your existing scheduler logic unchanged ---
        if self.warmup_epochs > 0:
            warm = LinearLR(opt, start_factor=0.01, total_iters=self.warmup_epochs)
            cos = CosineAnnealingLR(opt, T_max=max(self.epochs - self.warmup_epochs, 1), eta_min=self.min_lr)
            sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[self.warmup_epochs])
        else:
            sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.min_lr)

        scheduler_config = {
            "scheduler": sched,
            "interval": "epoch",
            "frequency": 1,
        }

        scfg = self.cfg.get("scheduler", {}) or {}
        use_plateau = bool(scfg.get("use_plateau_fallback", False))
        if use_plateau:
            plateau_pat = int(scfg.get("plateau_patience", 10))
            plateau_fac = float(scfg.get("plateau_factor", 0.5))
            plateau = ReduceLROnPlateau(opt, mode="min", patience=plateau_pat, factor=plateau_fac, min_lr=self.min_lr)
            scheduler_config = {
                "scheduler": plateau,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler_config,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """
        Run Lightning's closure to compute loss+backward, then guard and step.
        If any gradient is non-finite, skip the optimizer step.
        """
        # 1) ALWAYS execute the closure Lightning gives us (does backward())
        loss = None
        if optimizer_closure is not None:
            loss = optimizer_closure()

        # 2) Inspect gradients AFTER backward
        bad_name = None
        max_grad = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_name = name
                break
            try:
                max_grad = max(max_grad, float(p.grad.abs().max().item()))
            except Exception:
                pass

        # 3) If bad grads, skip the step
        if bad_name is not None:
            self.py_logger.error(
                f"[GRAD GUARD] Epoch {epoch}, Batch {batch_idx}: "
                f"Non-finite gradient in '{bad_name}'. Skipping optimizer step."
            )
            self.log("grad_nonfinite", 1.0, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            optimizer.zero_grad(set_to_none=True)
            return

        # (Optional) lightweight telemetry
        if batch_idx % 100 == 0:
            self.log("max_grad_magnitude", max_grad, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # 4) Now do the step (closure already executed above)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _get_teacher_forcing_prob(self, epoch: int) -> float:
        if self.tf_mode == "none":
            return 0.0
        if self.tf_mode == "constant":
            return float(self.tf_start)
        if self.tf_mode in ("linear", "cosine_ramp"):
            if epoch >= self.tf_end_epoch:
                return float(self.tf_end)
            t = epoch / max(1, self.tf_end_epoch)
            if self.tf_mode == "linear":
                return float(self.tf_start + (self.tf_end - self.tf_start) * t)
            else:
                return float(self.tf_end + 0.5 * (self.tf_start - self.tf_end) * (1 + math.cos(math.pi * t)))
        return 0.0

    def _get_rollout_horizon(self, epoch: int) -> int:
        if self.max_rollout_horizon <= 1:
            return 1
        t = min(max(epoch, 0), self.tf_end_epoch) / max(1, self.tf_end_epoch)
        H = 1 + int(round((self.max_rollout_horizon - 1) * t))
        return max(1, min(self.max_rollout_horizon, H))

    def _process_batch(self, batch):
        n = len(batch)
        if n == 4:
            y_i, dt, y_j, g = batch
            mask = None
        elif n == 5:
            y_i, dt, y_j, g, fifth = batch
            mask = None if isinstance(fifth, dict) else fifth
        elif n == 6:
            y_i, dt, y_j, g, aux, mask = batch
        else:
            raise RuntimeError(f"Unexpected batch format: {n} items")
        return y_i, dt, y_j, g, mask

    # ----------------------------
    # Numerically safe auxiliary losses
    # ----------------------------
    def _safe_mse_with_mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_resid: float = 1e4,
    ) -> torch.Tensor:
        # replace NaN/Inf and cap residuals to keep gradients finite
        r = torch.nan_to_num(pred - target, nan=0.0, posinf=0.0, neginf=0.0)
        r = r.clamp(-max_resid, max_resid)
        l = r * r  # elementwise squared error

        if mask is None:
            return l.mean()

        m = mask
        while m.ndim < l.ndim:
            m = m.unsqueeze(-1)
        m = m.to(l.dtype)
        denom = m.sum()
        if l.ndim >= 3 and (m.shape[-1] == 1):
            denom = (denom * l.shape[-1]).clamp_min(1)
        else:
            denom = denom.clamp_min(1)
        return (l * m).sum() / denom

    def training_step(self, batch, batch_idx):
        """Enhanced training step with detailed component-level logging."""
        opt = self.trainer.optimizers[0]
        current_lr = opt.param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        y_i, dt, y_j, g, mask = self._process_batch(batch)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        # Forward pass
        pred = self.model(y_i, dt, g)
        loss_dict = self.criterion(pred, y_j, dt, mask)
        total = loss_dict["total"]
        mse = loss_dict["mse"]

        # Batch-level logging (first 100 batches, then every 50)
        should_log_batch = batch_idx < 100 or batch_idx % 50 == 0
        if should_log_batch:
            self.log("train_mse_batch", mse, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            if "tail_huber" in loss_dict:
                self.log("train_tail_huber_batch", loss_dict["tail_huber"],
                         on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # Epoch-level aggregates
        self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if "tail_huber" in loss_dict:
            self.log("train_tail_huber", loss_dict["tail_huber"],
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Rollout loss (gradients enabled) with warmup + fp32 early
        rollout_loss = None
        if self.rollout_enabled and self.rollout_weight > 0:
            H = self._get_rollout_horizon(self.current_epoch)
            tf_p = self._get_teacher_forcing_prob(self.current_epoch)

            # warmup (ramps from 0 → 1 over rollout_warmup_epochs)
            rollout_warm = min(1.0, max(0.0, float(self.current_epoch) / max(1, self.rollout_warmup_epochs)))
            effective_rollout_w = self.rollout_weight * rollout_warm

            if effective_rollout_w > 1e-8:
                try:
                    rollout_loss = self._compute_rollout_loss(y_i, g, y_j, dt, mask, H, tf_p)
                    if rollout_loss is not None and torch.isfinite(rollout_loss):
                        total = total + effective_rollout_w * rollout_loss
                        self.log("rollout_loss", rollout_loss, on_step=False, on_epoch=True,
                                 prog_bar=False, logger=True)
                        if should_log_batch:
                            self.log("rollout_loss_batch", rollout_loss, on_step=True, on_epoch=False,
                                     prog_bar=False, logger=True)

                except Exception as e:
                    if batch_idx == 0:  # Only warn once per epoch
                        self.py_logger.warning(f"Rollout loss computation failed: {e}")

        # Semigroup loss with gentle warmup
        if self.semigroup_enabled and self.semigroup_weight > 0:
            try:
                sg_warm = min(1.0, max(0.0, float(self.current_epoch) / max(1, self.semigroup_warmup_epochs)))
                effective_sg_w = self.semigroup_weight * sg_warm
                if effective_sg_w > 1e-8:
                    sgl = self._compute_semigroup_loss(y_i, g)
                    if sgl is not None and torch.isfinite(sgl):
                        total = total + effective_sg_w * sgl
                        self.log("semigroup_loss", sgl, on_step=False, on_epoch=True,
                                 prog_bar=False, logger=True)
                        if should_log_batch:
                            self.log("semigroup_loss_batch", sgl, on_step=True, on_epoch=False,
                                     prog_bar=False, logger=True)
            except Exception as e:
                if batch_idx == 0:
                    self.py_logger.warning(f"Semigroup loss computation failed: {e}")

        # Time-warp L1
        if self.timewarp_l1_weight > 0:
            try:
                tw = getattr(getattr(self.model, "dynamics", None), "timewarp", None)
                s_mean = getattr(tw, "last_s_mean", None)
                if s_mean is not None and torch.isfinite(s_mean):
                    total = total + self.timewarp_l1_weight * s_mean
                    self.log("timewarp_l1", s_mean, on_step=False, on_epoch=True,
                             prog_bar=False, logger=True)
            except Exception:
                pass

        # Spot-check grad norm during the epoch (pre-backward grads may be missing; this is just a light probe)
        if batch_idx % 100 == 0:
            try:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().float().norm(2)
                        total_norm += float(param_norm.item()) ** 2
                total_norm = total_norm ** 0.5
                self.log("grad_norm_probe", total_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            except Exception:
                pass

        # NaN check with detailed diagnostics
        if not torch.isfinite(total):
            self.py_logger.error(
                f"[Batch {batch_idx}] Total loss is NaN/Inf! "
                f"MSE={mse:.3e}, rollout={rollout_loss}, "
                f"pred_range=[{pred.min():.3e}, {pred.max():.3e}]"
            )
            self.log("train_loss_nan", 1.0, on_step=True, on_epoch=False, prog_bar=True, logger=True)

            # Emergency diagnostics
            self.py_logger.error(f"  Input range: y_i=[{y_i.min():.3e}, {y_i.max():.3e}]")
            self.py_logger.error(f"  Target range: y_j=[{y_j.min():.3e}, {y_j.max():.3e}]")
            self.py_logger.error(f"  dt_norm range: [{dt.min():.3e}, {dt.max():.3e}]")

            return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

        if total.requires_grad and torch.isfinite(total):
            loss_value = float(total.item())
            if loss_value > 1e6:
                self.py_logger.warning(
                    f"[LOSS GUARD] Batch {batch_idx}: Loss extremely large ({loss_value:.3e}). "
                    f"May cause gradient overflow in backward pass."
                )
                self.log("loss_overflow_risk", 1.0, on_step=True, on_epoch=False, prog_bar=True)

            # Optional: Cap loss to prevent gradient explosion
            if loss_value > 1e8:
                self.py_logger.error(
                    f"[LOSS GUARD] Batch {batch_idx}: Loss={loss_value:.3e} exceeds safe threshold. "
                    f"Capping to 1e8."
                )
                total = total.clamp(max=1e8)

        # Normal logging
        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if should_log_batch:
            self.log("train_loss_batch", total, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return {"loss": total}

    def _compute_rollout_loss(
            self,
            y_i: torch.Tensor, g: torch.Tensor,
            y_j: torch.Tensor, dt_norm: torch.Tensor, mask: Optional[torch.Tensor],
            horizon: int, tf_prob: float
    ) -> Optional[torch.Tensor]:
        """
        Rollout loss computed in latent space:
          - Encode once (z_curr = enc(y_i, g))
          - Propagate one step in latent (dynamics.step on z)
          - Decode once per step for loss (needed to compare to y_true)
          - Teacher forcing (if used) only re-encodes ground truth for those samples.
        """
        if horizon <= 0:
            return None

        if dt_norm.ndim == 1:
            dt_norm = dt_norm.unsqueeze(1)  # [B] -> [B,1]
        elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)  # [B,K,1] -> [B,K]
        if dt_norm.ndim != 2:
            raise ValueError(f"dt_norm must be [B,K]; got {tuple(dt_norm.shape)}")
        B, K = dt_norm.shape
        H = min(K, max(1, horizon))

        # Early-epoch stability: force fp32 rollout if configured
        force_fp32 = (self.current_epoch < int(self.rollout_fp32_epochs))
        amp_ctx = (
            torch.autocast(device_type="cuda", enabled=not force_fp32)
            if torch.cuda.is_available() else contextlib.nullcontext()
        )
        g_cast = g.float() if force_fp32 else g
        y_anchor = y_i.float() if force_fp32 else y_i  # used if predict_delta=True

        losses: list[torch.Tensor] = []

        with amp_ctx:
            # Encode once at start
            z_curr = self.model.encoder(y_anchor, g_cast)

            for t in range(H):
                dt_step = dt_norm[:, t]
                dt_step = dt_step.float() if force_fp32 else dt_step

                # Latent step (no re-encode)
                z_next = self.model.dynamics.step(z_curr, dt_step, g_cast)

                # Decode once for loss at this step
                y_pred = self.model.decoder(z_next)
                if getattr(self.model, "predict_delta", False):
                    # Anchor-relative Δy semantics for rollout as well
                    y_pred = y_anchor + y_pred

                # Supervision target
                y_true = y_j[:, t] if (t < y_j.shape[1]) else y_j[:, -1]
                y_true = y_true.float() if force_fp32 else y_true

                # Optional step mask
                step_mask = None
                if mask is not None:
                    step_mask = (mask[:, t] if mask.ndim == 2 else mask[:, t, 0])

                # Numerically safe MSE
                l = self._safe_mse_with_mask(y_pred, y_true, step_mask)
                losses.append(l)

                # Teacher forcing: only re-encode ground truth for selected samples
                if t < H - 1:
                    use_tf = (torch.rand(B, device=self.device) < tf_prob).float().unsqueeze(-1)
                    if step_mask is not None:
                        use_tf = use_tf * step_mask.unsqueeze(-1).float()

                    z_tf = self.model.encoder(y_true.detach(), g_cast)  # encode GT only when used
                    z_curr = use_tf * z_tf.detach() + (1.0 - use_tf) * z_next.detach()

        return torch.stack(losses).mean() if losses else None

    def _compute_semigroup_loss(self, y_i: torch.Tensor, g: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            B = y_i.shape[0]
            device = y_i.device

            dt_a = torch.rand(B, device=device)
            dt_b = torch.rand(B, device=device)

            dyn = getattr(self.model, "dynamics", None)
            if dyn is not None and hasattr(dyn, "dt_log_min") and hasattr(dyn, "dt_log_max"):
                log_min = float(dyn.dt_log_min)
                log_max = float(dyn.dt_log_max)
                rng = max(log_max - log_min, 1e-12)

                dt_phys_a = (10.0 ** (log_min + dt_a * rng)).clamp_min(1e-30)
                dt_phys_b = (10.0 ** (log_min + dt_b * rng)).clamp_min(1e-30)
                dt_phys_ab = dt_phys_a + dt_phys_b

                dt_ab = (torch.log10(dt_phys_ab) - log_min) / rng
                dt_ab = dt_ab.clamp(0.0, 1.0)
            else:
                dt_ab = (dt_a + dt_b).clamp(0.0, 1.0)

            # Use public forward(): expects [B] or [B,1], returns [B,1,S]
            y_a = self.model(y_i, dt_a.unsqueeze(1), g)[:, 0]  # y(t+dt_a)
            y_ab = self.model(y_a, dt_b.unsqueeze(1), g)[:, 0]  # y(t+dt_a+dt_b) via two steps
            y_combined = self.model(y_i, dt_ab.unsqueeze(1), g)[:, 0]  # y(t+dt_a+dt_b) via combined dt

            return F.mse_loss(y_ab, y_combined)
        except Exception:
            return None

    def validation_step(self, batch, batch_idx):
        """Validation with robust failure handling and explicit val_loss logging."""
        y_i, dt, y_j, g, mask = self._process_batch(batch)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        # Forward pass with failure handling
        try:
            pred = self.model(y_i, dt, g)

            if not torch.isfinite(pred).all():
                try:
                    dtn = dt.squeeze(-1) if (dt.ndim == 3 and dt.shape[-1] == 1) else dt
                    dtp = self.model.dynamics._denorm_dt(dtn)
                    self.py_logger.error(
                        f"[Val Batch {batch_idx}] non-finite pred | "
                        f"dt_norm=[{float(dtn.min()):.3e},{float(dtn.max()):.3e}] "
                        f"dt_phys=[{float(dtp.min()):.3e},{float(dtp.max()):.3e}]"
                    )
                    if not getattr(self.model, "use_S", False):
                        A, _ = self.model.dynamics._build_A(g)
                        ev = torch.linalg.eigvalsh(A.float())
                        self.py_logger.error(
                            f"[Val Batch {batch_idx}] eig(A)=[{float(ev.min()):.3e},{float(ev.max()):.3e}]"
                        )
                except Exception:
                    pass

                fallback = torch.tensor(1e6, device=self.device)
                self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_loss_nan", 1.0, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                return {"val_loss": fallback}

        except Exception as e:
            self.py_logger.error(f"[Val Batch {batch_idx}] Forward pass failed: {e}")
            try:
                dtn = dt.squeeze(-1) if (dt.ndim == 3 and dt.shape[-1] == 1) else dt
                dtp = self.model.dynamics._denorm_dt(dtn)
                self.py_logger.error(
                    f"[Val Batch {batch_idx}] on-exception "
                    f"dt_norm=[{float(dtn.min()):.3e},{float(dtn.max()):.3e}] "
                    f"dt_phys=[{float(dtp.min()):.3e},{float(dtp.max()):.3e}]"
                )
            except Exception:
                pass
            fallback = torch.tensor(1e6, device=self.device)
            self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_loss_nan", 1.0, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            return {"val_loss": fallback}

        # Normal loss computation
        loss_dict = self.criterion(pred, y_j, dt, mask)
        total = loss_dict["total"]

        if not torch.isfinite(total):
            self.py_logger.error(f"[Val Batch {batch_idx}] NaN/Inf val loss")
            fallback = torch.tensor(1e6, device=self.device)
            self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_loss_nan", 1.0, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            return {"val_loss": fallback}

        self.log("val_loss", total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, logger=True)
        self.log("val_mse", loss_dict["mse"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, logger=True)
        if "tail_huber" in loss_dict:
            self.log("val_tail_huber", loss_dict["tail_huber"], on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False, logger=True)

        if batch_idx % 10 == 0:
            abs_error = (pred - y_j).abs()
            rel_error = abs_error / (y_j.abs() + 1e-8)
            self.log("val_abs_error_mean", abs_error.mean(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False, logger=True)
            self.log("val_abs_error_max", abs_error.max(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False, logger=True)
            self.log("val_rel_error_mean", rel_error.mean(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False, logger=True)

        return {"val_loss": total}

    def on_validation_epoch_end(self):
        # Ensure 'val_loss' exists for ModelCheckpoint even if every batch failed early
        if "val_loss" not in self.trainer.callback_metrics:
            self.log(
                "val_loss",
                torch.tensor(1e6, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    @torch.no_grad()
    def evaluate_rollout(self, batch, steps: int = 16) -> Dict[str, float]:
        y_i, dt, y_j, g, mask = self._process_batch(batch)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        if dt.ndim == 1:
            dt0 = dt
            steps = min(steps, 1)
        elif dt.ndim == 2:
            dt0 = dt[:, 0]
            steps = min(steps, dt.shape[1])
        else:
            raise ValueError(f"dt must be [B] or [B,K]; got {tuple(dt.shape)}")
        out = self.model.rollout(y_i, g, dt0, steps=steps)
        drift = (out[:, 1:] - out[:, :-1]).abs().mean().item()
        var = out.var(dim=1).mean().item()
        return {"rollout_drift": float(drift), "rollout_var": float(var)}

# --------------------------------------------------------------------------------------
# DataModule wrapper
# --------------------------------------------------------------------------------------

class FlexibleDataModule(pl.LightningDataModule):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader

        # Private storages with properties below
        self._batch_size = getattr(train_loader, "batch_size", None) or 32
        self._val_batch_size = getattr(val_loader, "batch_size", None) or self._batch_size

        # Cap enforcement (used during tuning)
        self._enforce_cap = False
        self._hard_cap = None

        self._train_kwargs = dict(
            num_workers=train_loader.num_workers,
            pin_memory=getattr(train_loader, "pin_memory", False),
            persistent_workers=getattr(train_loader, "persistent_workers", False),
            prefetch_factor=getattr(train_loader, "prefetch_factor", None),
            collate_fn=getattr(train_loader, "collate_fn", None),
            drop_last=getattr(train_loader, "drop_last", False),
        )
        self._val_kwargs = dict(
            num_workers=val_loader.num_workers,
            pin_memory=getattr(val_loader, "pin_memory", False),
            persistent_workers=getattr(val_loader, "persistent_workers", False),
            prefetch_factor=getattr(val_loader, "prefetch_factor", None),
            collate_fn=getattr(val_loader, "collate_fn", None),
            drop_last=getattr(val_loader, "drop_last", False),
        )

    # ----------------
    # Public controls
    # ----------------
    def enable_cap_enforcement(self, hard_cap: int):
        """Enable a hard upper bound for batch_size (used during tuner)."""
        self._hard_cap = int(hard_cap)
        self._enforce_cap = True

    def disable_cap_enforcement(self):
        self._enforce_cap = False
        self._hard_cap = None

    # ----------------
    # Properties
    # ----------------
    @property
    def batch_size(self) -> int:
        return int(self._batch_size)

    @batch_size.setter
    def batch_size(self, v: int):
        v = int(v)
        if self._enforce_cap and self._hard_cap is not None and v > self._hard_cap:
            # Make the tuner bail out immediately instead of “trying 16384”
            raise RuntimeError(f"Requested train batch_size {v} exceeds hard cap {self._hard_cap}.")
        self._batch_size = max(1, v)

    @property
    def val_batch_size(self) -> int:
        return int(self._val_batch_size)

    @val_batch_size.setter
    def val_batch_size(self, v: int):
        v = int(v)
        if self._enforce_cap and self._hard_cap is not None and v > self._hard_cap:
            # Validation should also respect cap if tuner messes with it
            raise RuntimeError(f"Requested val batch_size {v} exceeds hard cap {self._hard_cap}.")
        self._val_batch_size = max(1, v)

    # ----------------
    # Dataloaders
    # ----------------
    def train_dataloader(self):
        ds = self._train_loader.dataset
        return DataLoader(
            ds,
            batch_size=int(self.batch_size),
            shuffle=True,  # TRAIN MUST SHUFFLE
            **{k: v for k, v in self._train_kwargs.items() if v is not None}
        )

    def val_dataloader(self):
        ds = self._val_loader.dataset
        return DataLoader(
            ds,
            batch_size=int(self.val_batch_size),
            shuffle=False,
            **{k: v for k, v in self._val_kwargs.items() if v is not None}
        )


# --------------------------------------------------------------------------------------
# Public wrapper
# --------------------------------------------------------------------------------------

class Trainer:
    """Bridge that keeps main.py unchanged while using Lightning under the hood."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            cfg: Dict[str, Any],
            work_dir: Union[str, Path],
            device: torch.device,
            logger: logging.Logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.logger = logger

        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _save_json(self, path: Path, obj: Dict[str, Any]):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write {path.name}: {e}")

    def _maybe_tune_batch_size(
            self,
            module: AutoEncoderModule,  # ignored; we don't mutate it
            datamodule: FlexibleDataModule,
    ) -> Tuple[int, int]:
        """
        Decide/auto-tune batch sizes without touching CUDA.
        Returns: (final_train_bs, final_val_bs)
        """
        lcfg = self.cfg.get("lightning", {}) or {}
        tcfg = self.cfg.get("training", {}) or {}

        bs = tcfg.get("batch_size", getattr(self.train_loader, "batch_size", None))
        vbs = tcfg.get("val_batch_size", getattr(self.val_loader, "batch_size", None))
        if isinstance(bs, str):  bs = bs.lower()
        if isinstance(vbs, str): vbs = vbs.lower()

        # Hard cap from config limits
        user_cap = int(lcfg.get("auto_bs_user_cap", 32768))
        exp_bs_cap = int(lcfg.get("exp_batch_cap_for_matrix_exp", 8192))
        hard_cap = max(1, min(user_cap, exp_bs_cap))

        val_multiplier = float(lcfg.get("val_batch_size_multiplier", 2.0))
        initial_guess = int(lcfg.get("initial_batch_size_guess", datamodule.batch_size or 512))
        initial_guess = max(1, min(initial_guess, hard_cap))

        # If both fixed, just set and return
        if bs != "auto" and vbs != "auto":
            datamodule.batch_size = int(bs or datamodule.batch_size or 512)
            datamodule.val_batch_size = int(vbs or datamodule.val_batch_size or datamodule.batch_size * 2)
            return int(datamodule.batch_size), int(datamodule.val_batch_size)

        mode = str(lcfg.get("auto_scale_batch_size", "binsearch")).lower()
        if mode != "binsearch":
            if self.logger:
                self.logger.info("Overriding auto_scale_batch_size=%s → binsearch (safety).", mode)
            mode = "binsearch"

        ratio = max(1.0, float(hard_cap) / float(initial_guess))
        max_trials_user = int(lcfg.get("auto_bs_max_trials", 6))
        max_trials_cap = 1 + int(math.floor(math.log2(ratio)))  # includes the doubling steps
        max_trials = max(1, min(max_trials_user, max_trials_cap))

        # Seed datamodule batch size
        if isinstance(bs, int) and bs > 0:
            datamodule.batch_size = min(int(bs), hard_cap)
        elif bs == "auto":
            datamodule.batch_size = initial_guess

        tuned_train_bs = int(datamodule.batch_size)

        # Enforce cap during tuner: any attempt > hard_cap raises and stops the doubling
        datamodule.enable_cap_enforcement(hard_cap)

        if _TUNER_AVAILABLE and _PLTUNER is not None and bs == "auto":
            try:
                tmp_module = AutoEncoderModule(copy.deepcopy(self.model), self.cfg, self.work_dir).cpu()
                tmp_trainer = LightningTrainer(
                    accelerator="cpu",
                    devices=1,
                    precision="32",
                    max_epochs=1,
                    limit_train_batches=2,
                    limit_val_batches=1,
                    enable_checkpointing=False,
                    logger=CSVLogger(save_dir=str(self.work_dir), name=".bs_tune"),
                    enable_model_summary=False,
                    detect_anomaly=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                )
                tuner = _PLTUNER(tmp_trainer)
                tuned = tuner.scale_batch_size(
                    tmp_module,
                    datamodule=datamodule,
                    mode=mode,
                    steps_per_trial=1,
                    init_val=datamodule.batch_size,
                    max_trials=max_trials,
                )
                if isinstance(tuned, int) and tuned > 0:
                    tuned_train_bs = max(1, min(int(tuned), hard_cap))
                    if self.logger:
                        self.logger.info("Auto BS (CPU tuner): init=%d → tuned=%d (cap=%d, trials=%d)",
                                         initial_guess, tuned_train_bs, hard_cap, max_trials)
            except Exception as e:
                # Expect a RuntimeError once tuner tries > hard_cap; that is by design to stop it
                if self.logger:
                    self.logger.info("Batch-size tuner stopped at cap %d (reason: %s).", hard_cap, e)
            finally:
                datamodule.disable_cap_enforcement()
                try:
                    del tmp_trainer, tmp_module
                except Exception:
                    pass
                gc.collect()
        else:
            datamodule.disable_cap_enforcement()
            if self.logger:
                self.logger.info("Lightning Tuner unavailable or BS not 'auto'; using %d.", tuned_train_bs)

        final_train_bs = max(1, min(int(tuned_train_bs), hard_cap))
        datamodule.batch_size = final_train_bs

        if vbs == "auto":
            final_val_bs = int(final_train_bs * val_multiplier)
        else:
            final_val_bs = int(vbs or datamodule.val_batch_size or final_train_bs * val_multiplier)
        final_val_bs = max(1, min(int(final_val_bs), hard_cap))
        datamodule.val_batch_size = final_val_bs

        if self.logger:
            self.logger.info("Batch sizes: train=%d, val=%d (cap=%d, init=%d)",
                             final_train_bs, final_val_bs, hard_cap, initial_guess)
        return final_train_bs, final_val_bs

    def _maybe_tune_lr(
            self,
            module: AutoEncoderModule,  # ignored; we return a float
            datamodule: FlexibleDataModule,
            train_bs: int
    ) -> float:
        lcfg = self.cfg.get("lightning", {}) or {}
        tcfg = self.cfg.get("training", {}) or {}
        auto_lr = bool(lcfg.get("auto_lr_find", False))

        default_lr = float(tcfg.get("lr", 1e-4))
        min_lr_cfg = float(tcfg.get("min_lr", 1e-6))

        if not auto_lr:
            base_bs = int(lcfg.get("lr_base_batch_size", 8192))
            base_lr = default_lr
            scaled = base_lr * max(1, int(train_bs)) / max(1, base_bs)
            lr = max(min_lr_cfg, float(scaled))
            self._save_json(self.work_dir / "lr_find.json", {
                "mode": "linear_scale",
                "base_batch_size": base_bs,
                "train_batch_size": train_bs,
                "base_lr": base_lr,
                "chosen_lr": lr
            })
            if self.logger:
                self.logger.info("LR (linear scale): base %.2e @%d → %.2e @%d", base_lr, base_bs, lr, train_bs)
            return lr

        # Auto LR finder — CPU-only
        min_lr = float(lcfg.get("lr_find_min", 3e-6))
        max_lr = float(lcfg.get("lr_find_max", 2e-4))
        steps = int(lcfg.get("lr_find_steps", 50))
        train_limit = max(steps, 20)

        if not (_TUNER_AVAILABLE and _PLTUNER is not None):
            if self.logger:
                self.logger.info("PL Tuner not available; falling back to default LR %.2e.", default_lr)
            return default_lr

        tmp_module = AutoEncoderModule(copy.deepcopy(self.model), self.cfg, self.work_dir).cpu()
        tmp_trainer = LightningTrainer(
            accelerator="cpu",
            devices=1,
            precision="32",
            max_epochs=1,
            limit_train_batches=train_limit,
            limit_val_batches=1,
            enable_checkpointing=False,
            logger=CSVLogger(save_dir=str(self.work_dir), name=".lr_tune"),
            enable_model_summary=False,
            detect_anomaly=False,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
        )

        tuner = _PLTUNER(tmp_trainer)
        suggestion = None
        try:
            lr_finder = tuner.lr_find(
                tmp_module,
                datamodule=datamodule,
                min_lr=min_lr,
                max_lr=max_lr,
                num_training=steps,
                mode="exponential",
            )
            suggestion = lr_finder.suggestion()
            lr = float(suggestion) if suggestion is not None else default_lr
            lr = max(min_lr_cfg, lr)
            self._save_json(self.work_dir / "lr_find.json", {
                "mode": "lr_finder",
                "min_lr": min_lr, "max_lr": max_lr, "steps": steps,
                "suggestion": (None if suggestion is None else float(suggestion)),
                "chosen_lr": lr,
            })
            if self.logger:
                self.logger.info("LR finder (CPU): suggestion=%s → using %.2e",
                                 (None if suggestion is None else f"{float(suggestion):.2e}"), lr)
            return lr
        except Exception as e:
            if self.logger:
                self.logger.warning("LR finder failed on CPU; falling back to default. Reason: %s", e)
            return default_lr
        finally:
            try:
                del tmp_trainer, tmp_module
            except Exception:
                pass
            gc.collect()

    def train(self) -> float:
        """Execute full training loop with optional batch-size and LR tuning."""
        dm = FlexibleDataModule(self.train_loader, self.val_loader)

        # Tuning (CPU-only clones)
        train_bs, val_bs = self._maybe_tune_batch_size(module=None, datamodule=dm)
        lr = self._maybe_tune_lr(module=None, datamodule=dm, train_bs=train_bs)

        # Real module (fresh) — now we can go CUDA
        module = AutoEncoderModule(self.model, self.cfg, self.work_dir)
        module.learning_rate = float(lr)

        # ---- compile ONLY AFTER tuning (so tuners never see a compiled/bf16 graph) ----
        tc = self.cfg.get("lightning", {}).get("torch_compile", False)
        if tc:
            try:
                mode = tc if isinstance(tc, str) else "default"
                module.model = torch.compile(module.model, mode=mode, fullgraph=False)
                if self.logger:
                    self.logger.info(f"torch.compile enabled (mode={mode}, fullgraph=False)")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"torch.compile failed; continuing uncompiled. Reason: {e}")

        # ---- Main Trainer config ----
        lcfg = self.cfg.get("lightning", {}) or {}
        precision = lcfg.get("precision", "bf16-mixed")
        devices = lcfg.get("devices", 1)
        accelerator = lcfg.get("accelerator", "auto")
        accumulate = int(lcfg.get("accumulate_grad_batches", 1))
        deterministic = bool(lcfg.get("deterministic", False))
        benchmark = bool(lcfg.get("benchmark", True))
        max_epochs = int(lcfg.get("max_epochs", self.cfg.get("training", {}).get("epochs", 100)))
        strategy = lcfg.get("strategy", None) or "auto"
        detect_anomaly = bool(lcfg.get("detect_anomaly", False))
        num_sanity = int(lcfg.get("num_sanity_val_steps", 0))
        profiler = "simple" if bool(lcfg.get("profile", False)) else None
        enable_model_summary = bool(lcfg.get("enable_model_summary", False))
        val_limit_fraction = float(lcfg.get("val_limit_fraction", 0.1))

        fast_dev = bool(self.cfg.get("training", {}).get("fast_dev_run", False))
        if fast_dev:
            max_epochs = 1
            limit_train = 5
            limit_val = 2
            if self.logger:
                self.logger.info("Fast dev run: 1 epoch, 5 train batches, 2 val batches")
        else:
            limit_train = 1.0
            limit_val = val_limit_fraction
            if self.logger:
                self.logger.info(f"Validation will use {limit_val * 100:.1f}% of data")

        # ---- Callbacks ----
        ckpt_cb = ModelCheckpoint(
            dirpath=str(self.work_dir),
            filename="model-{epoch:03d}-{val_loss:.6f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        lrmon = LearningRateMonitor(logging_interval="epoch")
        csvlog = CSVLogger(save_dir=str(self.work_dir), name="logs")

        tw_cfg = self.cfg.get("training", {}).get("timewarp", {}) or {}
        smax_cfg = tw_cfg.get("smax_anneal", {}) or {}
        tw_cb = TimeWarpAnnealCallback(
            start=float(smax_cfg.get("start", 1e-2)),
            end=float(smax_cfg.get("end", 1.0)),
            end_epoch=int(smax_cfg.get("end_epoch", 60)),
        )

        epoch_setter = EpochSetterCallback()
        stats_cb = SimpleStatsCallback(py_logger=self.logger)
        grad_cb = GradNormLogger()
        finite_guard = NonFiniteParamGuard(py_logger=self.logger)

        trainer = LightningTrainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            max_epochs=max_epochs,
            limit_train_batches=limit_train,
            limit_val_batches=limit_val,
            accumulate_grad_batches=accumulate,
            deterministic=deterministic,
            benchmark=benchmark,
            strategy=strategy,
            logger=csvlog,
            callbacks=[ckpt_cb, lrmon, epoch_setter, tw_cb, stats_cb, grad_cb, finite_guard],
            log_every_n_steps=int(self.cfg.get("logging", {}).get("log_every_n_batches", 100)),
            enable_progress_bar=True,
            gradient_clip_val=float(self.cfg.get("training", {}).get("gradient_clip", 0.0)),
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=num_sanity,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            enable_model_summary=enable_model_summary,
        )

        dm.batch_size = train_bs
        dm.val_batch_size = val_bs

        trainer.fit(module, datamodule=dm)

        # ---- Extract best validation loss ----
        best_val = None
        if ckpt_cb.best_model_score is not None:
            best_val = float(ckpt_cb.best_model_score.cpu().item())
        else:
            try:
                best_val = float(trainer.callback_metrics["val_loss"].cpu().item())
            except Exception:
                best_val = None

        # ---- Legacy .pt export ----
        try:
            if ckpt_cb.best_model_path:
                from pathlib import PosixPath
                try:
                    import torch.serialization as ts
                    with ts.safe_globals([PosixPath, Path]):
                        state = torch.load(ckpt_cb.best_model_path, map_location="cpu")
                except Exception:
                    state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)

                sd = state.get("state_dict", {})
                model_state = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
                torch.save(
                    {"model": model_state, "best_val_loss": best_val, "config": self.cfg},
                    self.work_dir / "best_model.pt"
                )
                if self.logger:
                    self.logger.info(f"Saved legacy best_model.pt (val_loss={best_val:.6e})")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Legacy .pt export failed (non-critical): {e}")

        return best_val if best_val is not None else float("inf")