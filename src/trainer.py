#!/usr/bin/env python3
"""
Lightning Trainer for CVL Autoencoder (Simplified & Fixed)
===========================================================
Streamlined training pipeline with corrected autoregressive rollout:
- Combined MSE + optional Tail-Huber + optional fractional (denorm) loss
- Fixed rollout loss (proper predict_delta handling)
- Fixed semigroup loss (proper predict_delta handling)
- Teacher forcing curriculum
- Essential callbacks and monitoring
"""
from __future__ import annotations

import json
import logging
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer as LightningTrainer
from pytorch_lightning.callbacks import (
    Callback, ModelCheckpoint, LearningRateMonitor,
    EarlyStopping, StochasticWeightAveraging, RichProgressBar)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from normalizer import NormalizationHelper  # for fractional loss denorm


# --------------------------------------------------------------------------------------
# Loss: MSE (normalized) + optional Tail-Huber + optional fractional (denorm)
# --------------------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Primary loss: MSE(pred_norm, target_norm).
    Optional Tail-Huber on the normalized space to stabilize trace species.
    Optional fractional loss computed in **denormalized** (physical) space:
        frac = mean( ((y_phys_pred - y_phys_true) / (|y_phys_true| + eps))^2 )

    Pass `denorm_fn` if fractional_weight > 0 (maps [B,K,S]_norm -> [B,K,S]_phys).
    """

    def __init__(
            self,
            *,
            tail_delta: float = 0.02,
            tail_weight: float = 0.0,
            tail_z_threshold: float = -1.0,
            fractional_weight: float = 0.0,
            fractional_eps: float = 1e-8,
            denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.tail_delta = float(tail_delta)
        self.tail_weight = float(tail_weight)
        self.tail_z_threshold = float(tail_z_threshold)

        self.frac_weight = float(fractional_weight)
        self.frac_eps = float(fractional_eps)
        self.denorm_fn = denorm_fn if (fractional_weight > 0.0) else None

    def _apply_mask_and_reduce(self, loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            m = mask
            while m.ndim < loss.ndim:
                m = m.unsqueeze(-1)
            m = m.to(loss.dtype)
            denom = m.sum()
            # If mask is [B,K,1] and loss is [B,K,S], scale denominator by S
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
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes:
          total = MSE_norm
                  + tail_weight * TailHuber_norm[ z <= tail_z_threshold ]  (optional)
                  + frac_weight * FractionalLoss_phys                      (optional)
        Accepts mask with shapes [B], [B,K], [B,K,1], or [B,K,S].
        """
        # --- Sanitize inputs ---
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
        target_norm = torch.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

        if pred_norm.ndim != 3 or target_norm.ndim != 3:
            raise RuntimeError(
                f"Expected pred/target to be 3D [B,K,S]; got pred={list(pred_norm.shape)}, "
                f"target={list(target_norm.shape)}"
            )

        B, K, S = pred_norm.shape
        finite = torch.isfinite(pred_norm) & torch.isfinite(target_norm)

        # --- Coerce mask to be broadcastable to [B,K,S] ---
        if mask is not None:
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=pred_norm.device)
            m = m.to(torch.bool)

            if m.ndim == 1 and m.shape[0] == B:
                m = m.view(B, 1, 1)
            elif m.ndim == 2 and m.shape[0] == B:
                if m.shape[1] != K:
                    raise RuntimeError(f"2D mask has shape {list(m.shape)} but K={K}")
                m = m.unsqueeze(-1)
            elif m.ndim == 3:
                if list(m.shape) == [B, K, S]:
                    pass
                elif m.shape[0] == B and m.shape[1] == K and m.shape[2] == 1:
                    pass
                elif m.shape[0] == B and m.shape[1] == 1 and m.shape[2] == 1:
                    pass
                else:
                    raise RuntimeError(f"3D mask {list(m.shape)} not compatible with [B,K,S]=[{B},{K},{S}]")
            else:
                raise RuntimeError(f"Mask ndim={m.ndim} not supported")

            m = m.expand(B, K, 1) if m.shape[-1] == 1 else m
            mask = m & finite
        else:
            mask = finite

        out: Dict[str, torch.Tensor] = {}

        # --- Core MSE (normalized) ---
        r = pred_norm - target_norm
        mse = r.pow(2)
        mse_reduced = self._apply_mask_and_reduce(mse, mask)
        total = mse_reduced
        out["mse"] = mse_reduced

        # --- Optional Tail-Huber (normalized space, focused on tail region) ---
        if self.tail_weight > 0.0:
            z = target_norm
            tail_mask = (z <= self.tail_z_threshold)
            abs_r = r.abs()
            delta = self.tail_delta
            quad = 0.5 * (abs_r.clamp(max=delta) ** 2)
            lin = delta * (abs_r - 0.5 * delta).clamp_min(0.0)
            huber_full = torch.where(abs_r <= delta, quad, lin)
            huber_tail = huber_full.masked_fill(~tail_mask, 0.0)
            tail_huber = self._apply_mask_and_reduce(huber_tail, mask)
            out["tail_huber"] = tail_huber
            total = total + self.tail_weight * tail_huber

        # --- Optional fractional loss (denormalized physical space) — numerically safe ---
        if (self.frac_weight > 0.0) and (self.denorm_fn is not None):
            try:
                # Always compute denorm + fractional in full precision to avoid bf16 under/overflow
                if pred_norm.is_cuda:
                    ctx = torch.amp.autocast(device_type="cuda", enabled=False)
                else:
                    # No-op context on CPU
                    class _NoAutocast:
                        def __enter__(self): return None

                        def __exit__(self, exc_type, exc, tb): return False

                    ctx = _NoAutocast()

                with ctx:
                    pred_phys = self.denorm_fn(pred_norm.float())
                    targ_phys = self.denorm_fn(target_norm.float())

                    pred_phys = torch.nan_to_num(pred_phys, nan=0.0, posinf=0.0, neginf=0.0)
                    targ_phys = torch.nan_to_num(targ_phys, nan=0.0, posinf=0.0, neginf=0.0)

                    # Ignore tiny targets to avoid division by ~0 (trace species)
                    # tune these two thresholds as needed:
                    frac_ignore_below = 1e-12  # in physical units
                    frac_rel_cap = 1e3  # cap relative error magnitude

                    denom = targ_phys.abs()
                    small = denom < frac_ignore_below
                    safe_denom = torch.where(small, torch.full_like(denom, frac_ignore_below), denom)

                    rel = (pred_phys - targ_phys) / (safe_denom + self.frac_eps)
                    # cap extreme relative errors
                    rel = rel.clamp(min=-frac_rel_cap, max=frac_rel_cap)
                    # zero-out contribution where target is extremely small
                    rel = torch.where(small, torch.zeros_like(rel), rel)

                    frac = rel.pow(2)

                frac_reduced = self._apply_mask_and_reduce(frac, mask)
                out["fractional"] = frac_reduced
                total = total + self.frac_weight * frac_reduced
            except Exception:
                # If denorm or fractional fails for any reason, don't kill training; just skip this term.
                pass

        out["total"] = total
        return out


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------

class EpochSetterCallback(Callback):
    """Log teacher forcing probability and rollout horizon each epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        try:
            epoch = int(trainer.current_epoch)
            tf_p = float(pl_module._get_teacher_forcing_prob(epoch))
            H = int(pl_module._get_rollout_horizon(epoch))
            pl_module.log("tf_prob", tf_p, on_epoch=True, prog_bar=False)
            pl_module.log("rollout_horizon", H, on_epoch=True, prog_bar=False)
        except Exception:
            pass


class GradNormLogger(Callback):
    """Log gradient norm after backward pass."""

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


class NonFiniteGuard(Callback):
    """Detect and log non-finite parameters and gradients."""

    def __init__(self, py_logger: Optional[logging.Logger] = None):
        super().__init__()
        self._grad_tripped = False
        self._param_tripped = False
        self.py_logger = py_logger or logging.getLogger("trainer")

    def on_after_backward(self, trainer, pl_module):
        if self._grad_tripped:
            return
        try:
            with torch.no_grad():
                for name, p in pl_module.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        self._grad_tripped = True
                        self.py_logger.error(
                            f"[GUARD] Epoch {trainer.current_epoch}, Step {trainer.global_step}: "
                            f"Non-finite gradient in '{name}'"
                        )
                        pl_module.log("grad_nonfinite", 1.0, on_step=True, on_epoch=True, prog_bar=True)
                        break
        except Exception:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            # batch = (y_i, dt, y_j, g, [aux], [mask])
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                dt = batch[1]
                if isinstance(dt, torch.Tensor):
                    b = int(dt.shape[0])
                    k = int(dt.shape[1]) if dt.ndim >= 2 else 1
                    self._n_samples += b * k  # count anchors * times
        except Exception:
            pass


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
        now = time.time()
        start = self._epoch_start_time or now
        elapsed = max(now - start, 1e-12)
        throughput = float(self._n_samples) / elapsed

        # Parameter norm
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

        # Gradient norm from callback metrics
        cm = trainer.callback_metrics
        g_norm_val = cm.get("grad_norm", None)
        if isinstance(g_norm_val, torch.Tensor):
            g_norm = float(g_norm_val.detach().cpu())
        else:
            g_norm = float("nan")

        # Learning rate
        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")

        # GPU memory
        max_mem = 0.0
        if torch.cuda.is_available():
            try:
                max_mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
            except Exception:
                pass

        train_loss = float(cm.get("train_loss", float("nan")))
        val_loss = float(cm.get("val_loss", float("nan")))

        self.py_logger.info(
            f"Epoch {trainer.current_epoch:3d} | "
            f"train={train_loss:.3e} val={val_loss:.3e} | "
            f"lr={lr:.2e} | "
            f"||p||={p_norm:.2e} ||grad||={g_norm:.2e} | "
            f"{throughput:.0f} samples/s | "
            f"GPU {max_mem:.1f}GB"
        )

        pl_module.log("throughput", throughput, on_epoch=True, prog_bar=False)
        pl_module.log("param_norm", p_norm, on_epoch=True, prog_bar=False)
        pl_module.log("gpu_mem_gb", max_mem, on_epoch=True, prog_bar=False)


# --------------------------------------------------------------------------------------
# Lightning Module
# --------------------------------------------------------------------------------------

class AutoEncoderModule(LightningModule):
    """Lightning wrapper around the autoencoder model."""

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

        # ---- Build denorm helper for optional fractional loss ----
        frac_cfg = (tcfg.get("loss", {}) or {}).get("fractional", {}) or {}
        frac_w = float(frac_cfg.get("weight", 0.0))
        frac_eps = float(frac_cfg.get("eps", 1e-8))
        denorm_fn = None
        if frac_w > 0.0:
            # Load normalization manifest (same paths logic used elsewhere)
            cand = []
            if "paths" in cfg and "processed_data_dir" in cfg["paths"]:
                cand.append(Path(cfg["paths"]["processed_data_dir"]) / "normalization.json")
            cand.append(Path("data/processed/normalization.json"))
            manifest = None
            for p in cand:
                if p.exists():
                    with open(p, "r") as f:
                        manifest = json.load(f)
                    break
            if manifest is None:
                self.py_logger.warning(
                    "Fractional loss requested but normalization.json not found; disabling fractional term.")
                frac_w = 0.0
            else:
                # Species keys come from cfg.data.species_variables (same assumption as dataset/model)
                species_keys = list(cfg.get("data", {}).get("species_variables", []))
                if not species_keys:
                    self.py_logger.warning(
                        "Fractional loss requested but data.species_variables missing; disabling fractional term.")
                    frac_w = 0.0
                else:
                    norm_helper = NormalizationHelper(manifest)

                    def _denorm_fn(x_norm: torch.Tensor) -> torch.Tensor:
                        # x_norm: [B,K,S] -> denorm per species on last dim
                        B, K, S = x_norm.shape
                        x_flat = x_norm.reshape(-1, S)
                        x_phys_flat = norm_helper.denormalize(x_flat, species_keys)
                        return x_phys_flat.reshape(B, K, S)

                    denorm_fn = _denorm_fn

        # ---- Loss config (Tail-Huber + optional fractional) ----
        lcfg = tcfg.get("loss", {}) or {}
        th = lcfg.get("tail_huber", {}) or {}
        self.criterion = CombinedLoss(
            tail_delta=float(th.get("delta", 0.02)),
            tail_weight=float(th.get("weight", 0.0)),
            tail_z_threshold=float(th.get("z_threshold", -1.0)),
            fractional_weight=frac_w,
            fractional_eps=frac_eps,
            denorm_fn=denorm_fn,
        )

        # ---- Aux losses ----
        aux = tcfg.get("auxiliary_losses", {}) or {}
        self.rollout_enabled = bool(aux.get("rollout_enabled", True))
        self.rollout_weight = float(aux.get("rollout_weight", 0.5))
        self.max_rollout_horizon = int(aux.get("rollout_horizon", 8))
        self.rollout_use_cached = bool(aux.get("rollout_use_cached_encoding", False))

        self.semigroup_enabled = bool(aux.get("semigroup_enabled", True))
        self.semigroup_weight = float(aux.get("semigroup_weight", 0.1))

        self.rollout_warmup_epochs = int(aux.get("rollout_warmup_epochs", 10))
        self.semigroup_warmup_epochs = int(aux.get("semigroup_warmup_epochs", 10))

        tf_cfg = aux.get("rollout_teacher_forcing", {}) or {}
        self.tf_mode = str(tf_cfg.get("mode", "linear"))
        self.tf_start = float(tf_cfg.get("start_p", 0.8))
        self.tf_end = float(tf_cfg.get("end_p", 0.0))
        self.tf_end_epoch = int(tf_cfg.get("end_epoch", 60))

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.warmup_epochs > 0:
            warm = LinearLR(opt, start_factor=0.01, total_iters=self.warmup_epochs)
            cos = CosineAnnealingLR(opt, T_max=max(self.epochs - self.warmup_epochs, 1), eta_min=self.min_lr)
            sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[self.warmup_epochs])
        else:
            sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.min_lr)

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}
        }

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

    def _safe_mse_with_mask(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            max_resid: float = 1e4,
    ) -> torch.Tensor:
        r = torch.nan_to_num(pred - target, nan=0.0, posinf=0.0, neginf=0.0)
        r = r.clamp(-max_resid, max_resid)
        l = r * r
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
        opt = self.trainer.optimizers[0]
        current_lr = opt.param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        y_i, dt, y_j, g, mask = self._process_batch(batch)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        # Forward
        pred = self.model(y_i, dt, g)

        # Primary loss(es)
        loss_dict = self.criterion(pred, y_j, mask)
        total = loss_dict["total"]
        self.log("train_mse", loss_dict["mse"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if "tail_huber" in loss_dict:
            self.log("train_tail_huber", loss_dict["tail_huber"], on_step=False, on_epoch=True, logger=True)
        if "fractional" in loss_dict:
            self.log("train_fractional", loss_dict["fractional"], on_step=False, on_epoch=True, logger=True)

        # Rollout loss with warmup
        if self.rollout_enabled and self.rollout_weight > 0:
            H = self._get_rollout_horizon(self.current_epoch)
            tf_p = self._get_teacher_forcing_prob(self.current_epoch)
            rollout_warm = min(1.0, max(0.0, float(self.current_epoch) / max(1, self.rollout_warmup_epochs)))
            effective_rollout_w = self.rollout_weight * rollout_warm

            if effective_rollout_w > 1e-8:
                try:
                    if self.rollout_use_cached:
                        rollout_loss = self._compute_rollout_loss_cached(y_i, g, y_j, dt, mask, H, tf_p)
                    else:
                        rollout_loss = self._compute_rollout_loss(y_i, g, y_j, dt, mask, H, tf_p)

                    if rollout_loss is not None and torch.isfinite(rollout_loss):
                        total = total + effective_rollout_w * rollout_loss
                        self.log("rollout_loss", rollout_loss, on_step=False, on_epoch=True, logger=True)
                except Exception as e:
                    if batch_idx == 0:
                        self.py_logger.warning(f"Rollout loss computation failed: {e}")

        # Semigroup loss with warmup
        if self.semigroup_enabled and self.semigroup_weight > 0:
            try:
                sg_warm = min(1.0, max(0.0, float(self.current_epoch) / max(1, self.semigroup_warmup_epochs)))
                effective_sg_w = self.semigroup_weight * sg_warm
                if effective_sg_w > 1e-8:
                    sgl = self._compute_semigroup_loss(y_i, g)
                    if sgl is not None and torch.isfinite(sgl):
                        total = total + effective_sg_w * sgl
                        self.log("semigroup_loss", sgl, on_step=False, on_epoch=True, logger=True)
            except Exception as e:
                if batch_idx == 0:
                    self.py_logger.warning(f"Semigroup loss computation failed: {e}")

        if not torch.isfinite(total):
            self.py_logger.error(f"[Batch {batch_idx}] Total loss is NaN/Inf!")
            self.log("train_loss_nan", 1.0, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": total}

    def _compute_rollout_loss(
            self,
            y_i: torch.Tensor,
            g: torch.Tensor,
            y_j: torch.Tensor,
            dt_norm: torch.Tensor,
            mask: Optional[torch.Tensor],
            horizon: int,
            tf_prob: float
    ) -> Optional[torch.Tensor]:
        """
        Rollout loss computed autoregressively with teacher forcing.
        Uses model.step() which handles encoding, dynamics, g-conditioned decoding, and predict_delta.
        Correctly handles masks with shapes: [B], [B,K], [B,K,1], or [B,K,S].
        """
        if horizon <= 0:
            return None

        # Normalize dt_norm to [B,K]
        if dt_norm.ndim == 1:
            dt_norm = dt_norm.unsqueeze(1)
        elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)
        if dt_norm.ndim != 2:
            raise ValueError(f"dt_norm must be [B,K]; got {tuple(dt_norm.shape)}")

        B, K = dt_norm.shape
        S = y_i.shape[-1]
        H = min(K, max(1, horizon))

        y_curr = y_i
        losses = []

        for t in range(H):
            # One-step prediction through the model (includes predict_delta & g-conditioning)
            y_pred = self.model.step(y_curr, dt_norm[:, t], g)

            # Ground truth at this offset (fallback to last if K mismatch)
            y_true = y_j[:, t] if (t < y_j.shape[1]) else y_j[:, -1]

            # ----- Build per-step masks -----
            step_mask: Optional[torch.Tensor] = None  # [B,S] or [B] accepted by _safe_mse_with_mask
            tf_mask: Optional[torch.Tensor] = None  # [B] for teacher-forcing gate

            if mask is not None:
                if mask.ndim == 1 and mask.shape[0] == B:
                    # [B] => loss mask broadcast to [B,S]; TF gate is same [B]
                    tf_mask = mask
                    step_mask = mask.view(B, 1).expand(B, S)
                elif mask.ndim == 2 and mask.shape == (B, K):
                    # [B,K] => pick time slice -> [B]; expand for loss
                    tf_mask = mask[:, t]
                    step_mask = tf_mask.view(B, 1).expand(B, S)
                elif mask.ndim == 3:
                    if mask.shape[0] != B:
                        raise ValueError(f"Mask batch mismatch: {tuple(mask.shape)} vs B={B}")
                    if mask.shape[1] not in (1, K):
                        raise ValueError(f"3D mask second dim must be 1 or K; got {tuple(mask.shape)} with K={K}")
                    # Select time slice (use t or 0 if singleton)
                    time_idx = t if mask.shape[1] == K else 0
                    ms = mask[:, time_idx, ...]  # [B,S] or [B,1]
                    if ms.ndim == 2 and ms.shape[1] == S:
                        step_mask = ms
                        tf_mask = ms.all(dim=-1)  # [B]
                    elif ms.ndim == 2 and ms.shape[1] == 1:
                        # [B,1] -> broadcast for loss; TF gate is [B]
                        step_mask = ms.expand(B, S)
                        tf_mask = ms.squeeze(-1)
                    else:
                        raise ValueError(f"Unsupported 3D mask slice shape: {tuple(ms.shape)}; expected [B,S] or [B,1]")

            # Loss for this step
            l = self._safe_mse_with_mask(y_pred, y_true, step_mask)
            losses.append(l)

            # Teacher forcing: mix next y_curr
            if t < H - 1:
                use_tf = (torch.rand(B, device=self.device) < tf_prob).float()  # [B]
                if tf_mask is not None:
                    use_tf = use_tf * tf_mask.float()
                use_tf = use_tf.unsqueeze(-1)  # [B,1]
                y_curr = use_tf * y_true.detach() + (1.0 - use_tf) * y_pred.detach()

        return torch.stack(losses).mean() if losses else None

    def _compute_rollout_loss_cached(
            self,
            y_i: torch.Tensor,
            g: torch.Tensor,
            y_j: torch.Tensor,
            dt_norm: torch.Tensor,
            mask: Optional[torch.Tensor],
            horizon: int,
            tf_prob: float
    ) -> Optional[torch.Tensor]:
        """
        Rollout loss with cached encoding and A(g). Operates in latent space, then decodes.
        Correctly handles g-conditioned decoder and predict_delta.
        Correctly handles masks with shapes: [B], [B,K], [B,K,1], or [B,K,S].
        """
        if horizon <= 0:
            return None

        # Normalize dt_norm to [B,K]
        if dt_norm.ndim == 1:
            dt_norm = dt_norm.unsqueeze(1)
        elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)
        if dt_norm.ndim != 2:
            raise ValueError(f"dt_norm must be [B,K]; got {tuple(dt_norm.shape)}")

        B, K = dt_norm.shape
        S = y_i.shape[-1]
        H = min(K, max(1, horizon))
        predict_delta = getattr(self.model, "predict_delta", False)
        decoder_condition_on_g = getattr(self.model, "decoder_condition_on_g", False)

        # Cache encode and A(g)
        z_curr = self.model.encoder(y_i, g)  # [B,Z]
        A, _ = self.model.dynamics._build_A(g)

        y_curr = y_i
        losses = []

        for t in range(H):
            # Propagate latent with cached A
            dt_step = dt_norm[:, t]
            dt_phys = self.model.dynamics._denorm_dt(dt_step)

            with torch.amp.autocast("cuda", enabled=False):
                A_float = A.float()  # [B,Z,Z]
                dt_float = dt_phys.float().view(B, 1, 1)  # [B,1,1]
                Phi = torch.matrix_exp(A_float * dt_float)  # [B,Z,Z]
                z_next = torch.bmm(Phi, z_curr.float().unsqueeze(-1)).squeeze(-1)
                z_next = z_next.to(z_curr.dtype)  # [B,Z]

            # Decode (respect g-conditioning)
            if decoder_condition_on_g:
                dec_in = torch.cat([z_next, g], dim=-1)  # [B,Z+G]
            else:
                dec_in = z_next  # [B,Z]
            y_decoded = self.model.decoder(dec_in)  # [B,S]

            # Apply predict-delta if enabled
            y_pred = y_curr + y_decoded if predict_delta else y_decoded

            # Ground truth at this offset
            y_true = y_j[:, t] if (t < y_j.shape[1]) else y_j[:, -1]

            # ----- Build per-step masks -----
            step_mask: Optional[torch.Tensor] = None  # [B,S] or [B]
            tf_mask: Optional[torch.Tensor] = None  # [B]

            if mask is not None:
                if mask.ndim == 1 and mask.shape[0] == B:
                    tf_mask = mask
                    step_mask = mask.view(B, 1).expand(B, S)
                elif mask.ndim == 2 and mask.shape == (B, K):
                    tf_mask = mask[:, t]
                    step_mask = tf_mask.view(B, 1).expand(B, S)
                elif mask.ndim == 3:
                    if mask.shape[0] != B:
                        raise ValueError(f"Mask batch mismatch: {tuple(mask.shape)} vs B={B}")
                    if mask.shape[1] not in (1, K):
                        raise ValueError(f"3D mask second dim must be 1 or K; got {tuple(mask.shape)} with K={K}")
                    time_idx = t if mask.shape[1] == K else 0
                    ms = mask[:, time_idx, ...]  # [B,S] or [B,1]
                    if ms.ndim == 2 and ms.shape[1] == S:
                        step_mask = ms
                        tf_mask = ms.all(dim=-1)  # [B]
                    elif ms.ndim == 2 and ms.shape[1] == 1:
                        step_mask = ms.expand(B, S)
                        tf_mask = ms.squeeze(-1)
                    else:
                        raise ValueError(f"Unsupported 3D mask slice shape: {tuple(ms.shape)}; expected [B,S] or [B,1]")

            # Loss for this step
            l = self._safe_mse_with_mask(y_pred, y_true, step_mask)
            losses.append(l)

            # Teacher forcing in latent (and y_curr if predict_delta)
            if t < H - 1:
                use_tf = (torch.rand(B, device=self.device) < tf_prob).float()  # [B]
                if tf_mask is not None:
                    use_tf = use_tf * tf_mask.float()
                use_tf = use_tf.unsqueeze(-1)  # [B,1]

                # Next latent: either encode ground truth or keep z_next
                z_gt = self.model.encoder(y_true.detach(), g)
                z_curr = use_tf * z_gt.detach() + (1.0 - use_tf) * z_next.detach()

                # If predict_delta=True, y_curr participates in forming y_pred; keep it consistent
                if predict_delta:
                    y_curr = use_tf * y_true.detach() + (1.0 - use_tf) * y_pred.detach()

        return torch.stack(losses).mean() if losses else None

    def _compute_semigroup_loss(self, y_i: torch.Tensor, g: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Semigroup consistency: f(dt_a + dt_b, y) ≈ f(dt_b, f(dt_a, y))
        Model.forward() handles predict_delta internally, so just use outputs directly.
        """
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

            # Path 1: Two sequential steps
            # model.forward() handles predict_delta internally, returns absolute predictions
            y_a = self.model(y_i, dt_a.unsqueeze(1), g)[:, 0]
            y_ab_seq = self.model(y_a, dt_b.unsqueeze(1), g)[:, 0]

            # Path 2: Single combined step
            y_ab_combined = self.model(y_i, dt_ab.unsqueeze(1), g)[:, 0]

            return F.mse_loss(y_ab_seq, y_ab_combined)

        except Exception:
            return None

    def validation_step(self, batch, batch_idx):
        y_i, dt, y_j, g, mask = self._process_batch(batch)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        try:
            pred = self.model(y_i, dt, g)
            if not torch.isfinite(pred).all():
                self.py_logger.error(f"[Val Batch {batch_idx}] Non-finite predictions")
                fallback = torch.tensor(1e6, device=self.device)
                self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_loss_nan", 1.0, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                return {"val_loss": fallback}
        except Exception as e:
            self.py_logger.error(f"[Val Batch {batch_idx}] Forward pass failed: {e}")
            fallback = torch.tensor(1e6, device=self.device)
            self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return {"val_loss": fallback}

        loss_dict = self.criterion(pred, y_j, mask)
        total = loss_dict["total"]
        if not torch.isfinite(total):
            self.py_logger.error(f"[Val Batch {batch_idx}] NaN/Inf val loss")
            fallback = torch.tensor(1e6, device=self.device)
            self.log("val_loss", fallback, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return {"val_loss": fallback}

        self.log("val_loss", total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, logger=True)
        self.log("val_mse", loss_dict.get("mse", total), on_step=False, on_epoch=True, logger=True)
        if "tail_huber" in loss_dict:
            self.log("val_tail_huber", loss_dict["tail_huber"], on_step=False, on_epoch=True, logger=True)
        if "fractional" in loss_dict:
            self.log("val_fractional", loss_dict["fractional"], on_step=False, on_epoch=True, logger=True)

        if batch_idx % 10 == 0:
            abs_err = (pred - y_j).abs()
            rel_err = abs_err / (y_j.abs() + 1e-8)
            self.log("val_abs_err_mean", abs_err.mean(), on_step=False, on_epoch=True, logger=True)
            self.log("val_rel_err_mean", rel_err.mean(), on_step=False, on_epoch=True, logger=True)

        return {"val_loss": total}

    def on_validation_epoch_end(self):
        if "val_loss" not in self.trainer.callback_metrics:
            self.log("val_loss", torch.tensor(1e6, device=self.device), on_step=False, on_epoch=True, prog_bar=True)


# --------------------------------------------------------------------------------------
# Public wrapper
# --------------------------------------------------------------------------------------

class Trainer:
    """Simple wrapper around Lightning trainer."""

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

    def train(self) -> float:
        """Execute training loop."""
        module = AutoEncoderModule(self.model, self.cfg, self.work_dir)

        lcfg = self.cfg.get("lightning", {}) or {}
        tcfg = self.cfg.get("training", {}) or {}

        # ----------------------------
        # Runtime / hyperparameters
        # ----------------------------
        batch_size = int(tcfg.get("batch_size", self.train_loader.batch_size or 512))
        val_batch_size = int(tcfg.get("val_batch_size", batch_size * 2))

        precision = str(lcfg.get("precision", "bf16-mixed"))
        devices = lcfg.get("devices", 1)
        accelerator = lcfg.get("accelerator", "auto")
        accumulate = int(lcfg.get("accumulate_grad_batches", 1))
        strategy = lcfg.get("strategy", "auto")
        num_sanity_val_steps = int(lcfg.get("num_sanity_val_steps", 0))

        max_epochs = int(tcfg.get("epochs", 100))
        fast_dev = bool(tcfg.get("fast_dev_run", False))
        if fast_dev:
            max_epochs = 1
            limit_train = 5
            limit_val = 2
            self.logger.info("Fast dev run: 1 epoch, 5 train batches, 2 val batches")
        else:
            limit_train = 1.0
            limit_val = 1.0

        self.logger.info(f"[batch] accumulate_grad_batches = {accumulate}")

        # ----------------------------
        # Fresh DataLoaders (reuse dataset & settings from originals)
        # ----------------------------
        from torch.utils.data import DataLoader as TorchDataLoader
        train_loader = TorchDataLoader(
            self.train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=getattr(self.train_loader, "num_workers", 0),
            pin_memory=getattr(self.train_loader, "pin_memory", False),
            collate_fn=getattr(self.train_loader, "collate_fn", None),
            persistent_workers=getattr(self.train_loader, "persistent_workers", False),
            prefetch_factor=getattr(self.train_loader, "prefetch_factor", None),
        )
        val_loader = TorchDataLoader(
            self.val_loader.dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=getattr(self.val_loader, "num_workers", 0),
            pin_memory=getattr(self.val_loader, "pin_memory", False),
            collate_fn=getattr(self.val_loader, "collate_fn", None),
            persistent_workers=getattr(self.val_loader, "persistent_workers", False),
            prefetch_factor=getattr(self.val_loader, "prefetch_factor", None),
        )

        # ----------------------------
        # Checkpointing / logging
        # ----------------------------
        ckpt_dir = self.work_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
        )

        # Early stopping (sensible defaults; disable by setting patience<=0 in cfg if desired)
        es_patience = int(tcfg.get("early_stop_patience", 15))
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=max(0, es_patience),
            mode="min",
            min_delta=float(tcfg.get("early_stop_min_delta", 1e-5)),
            verbose=True,
        ) if es_patience > 0 else None

        # SWA (start late in training to stabilize)
        swa = StochasticWeightAveraging(
            swa_lrs=float(tcfg.get("swa_lrs", 1e-5)),
            swa_epoch_start=max(1, int(0.75 * max_epochs)),
        )

        lrmon = LearningRateMonitor(logging_interval="epoch")
        csvlog = CSVLogger(save_dir=str(self.work_dir), name="csv_logs")
        tblog = TensorBoardLogger(save_dir=str(self.work_dir), name="tb_logs")

        # Our custom callbacks (epoch stats go to your python logger/file)
        epoch_setter = EpochSetterCallback()
        stats_cb = SimpleStatsCallback(py_logger=self.logger)
        grad_cb = GradNormLogger()
        finite_guard = NonFiniteGuard(py_logger=self.logger)

        callbacks = [ckpt_cb, swa, lrmon, epoch_setter, stats_cb, grad_cb, finite_guard]
        if early_stop is not None:
            callbacks.insert(1, early_stop)

        # ----------------------------
        # Resume behavior (NO auto-resume unless asked)
        #   - If training.resume is a non-empty path and exists -> resume from it
        #   - elif training.auto_resume == True and last.ckpt exists -> resume
        #   - else -> start fresh
        # ----------------------------
        resume_ckpt: Optional[str] = None
        resume_cfg = tcfg.get("resume", None)
        auto_resume = bool(tcfg.get("auto_resume", False))

        if isinstance(resume_cfg, str) and len(resume_cfg.strip()) > 0:
            candidate = Path(resume_cfg.strip())
            if candidate.exists():
                resume_ckpt = str(candidate)
                self.logger.info(f"Resuming from checkpoint (training.resume): {resume_ckpt}")
            else:
                self.logger.warning(f"Requested resume path not found: {candidate}")
        elif auto_resume:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                resume_ckpt = str(last_ckpt)
                self.logger.info(f"Auto-resuming from checkpoint: {resume_ckpt}")

        # ----------------------------
        # Lightning Trainer
        # ----------------------------
        trainer = LightningTrainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            max_epochs=max_epochs,
            min_epochs=int(tcfg.get("min_epochs", 1)),
            limit_train_batches=limit_train,
            limit_val_batches=limit_val,
            accumulate_grad_batches=accumulate,
            num_sanity_val_steps=num_sanity_val_steps,

            logger=[csvlog, tblog],
            callbacks=callbacks,

            # Keep CLI quiet; our SimpleStatsCallback writes per-epoch to python logging
            log_every_n_steps=int(tcfg.get("log_every_n_steps", 50)),
            enable_progress_bar=bool(tcfg.get("enable_progress_bar", False)),
            enable_model_summary=True,

            gradient_clip_val=float(tcfg.get("gradient_clip", 0.0)),
            gradient_clip_algorithm="norm",
            benchmark=bool(self.cfg.get("system", {}).get("cudnn_benchmark", True)),
        )

        # ----------------------------
        # Fit
        # ----------------------------
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt,  # None => start fresh
        )

        # ----------------------------
        # Best val & export a slim .pt
        # ----------------------------
        best_val: Optional[float] = None
        if ckpt_cb.best_model_score is not None:
            best_val = float(ckpt_cb.best_model_score.cpu().item())
        else:
            try:
                best_val = float(trainer.callback_metrics["val_loss"].cpu().item())
            except Exception:
                pass

        try:
            if ckpt_cb.best_model_path:
                state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
                sd = state.get("state_dict", {})
                model_state = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
                torch.save(
                    {"model": model_state, "best_val_loss": best_val, "config": self.cfg},
                    self.work_dir / "best_model.pt"
                )
                self.logger.info(f"Saved best_model.pt (val_loss={best_val:.6e})")
        except Exception as e:
            self.logger.warning(f"Legacy .pt export failed: {e}")

        return best_val if best_val is not None else float("inf")
