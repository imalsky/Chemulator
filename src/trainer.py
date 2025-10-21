#!/usr/bin/env python3
"""
PyTorch Lightning Trainer for Koopman Autoencoder
==================================================
Simplified training loop with MSE loss, gradient clipping, and learning rate scheduling.
Designed for flow-map prediction with optional auxiliary losses.

Key Features:
- MSE loss with optional masking for invalid time steps
- Teacher forcing with configurable decay schedule
- Rollout loss for multi-step prediction stability
- Cosine annealing LR with linear warmup
- Gradient clipping for training stability
- PyTorch Lightning integration (checkpointing, logging, early stopping)
- Batch shape validation with clear error messages

Loss Components:
1. One-step prediction: MSE between predicted and target states
2. Rollout (optional): Multi-step forward prediction with error accumulation
3. Teacher forcing: Gradual transition from ground truth to autoregressive

Training Schedule:
- Warmup: Linear LR increase over first N epochs
- Main: Cosine annealing to min_lr
- Optional: Early stopping on validation loss

Note: Assumes K > 1 (multiple time predictions per batch).
"""
from __future__ import annotations

import logging
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer as LightningTrainer
from pytorch_lightning.callbacks import (
    Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


# Maximum residual magnitude before clipping (prevents gradient explosion)
MAX_RESIDUAL = 1e4

# --------------------------------------------------------------------------------------
# Batch Shape Validator
# --------------------------------------------------------------------------------------

def validate_batch_shape(batch, expected_K: int, expected_S: int, batch_idx: int = 0):
    """
    Validate batch has the expected shape. Fail fast with clear error.

    Expected batch: (y_i, dt_norm, y_j, g, aux, mask)
    - y_i: [B, S]
    - dt_norm: [B, K]
    - y_j: [B, K, S]
    - g: [B, G]
    - mask: [B, K] (optional)
    """
    if len(batch) < 4:
        raise RuntimeError(f"Batch has {len(batch)} items, expected at least 4")

    y_i, dt_norm, y_j, g = batch[:4]

    # Check y_i shape [B, S]
    if y_i.ndim != 2:
        raise RuntimeError(f"y_i must be [B, S], got shape {tuple(y_i.shape)}")
    B, S = y_i.shape
    if S != expected_S:
        raise RuntimeError(f"y_i has S={S}, expected S={expected_S}")

    # Check dt_norm shape [B, K]
    if dt_norm.ndim != 2:
        raise RuntimeError(f"dt_norm must be [B, K], got shape {tuple(dt_norm.shape)}")
    if dt_norm.shape[0] != B:
        raise RuntimeError(f"dt_norm batch mismatch: {dt_norm.shape[0]} vs {B}")
    K = dt_norm.shape[1]
    if K != expected_K:
        raise RuntimeError(f"dt_norm has K={K}, expected K={expected_K}")

    # Check y_j shape [B, K, S]
    if y_j.ndim != 3:
        raise RuntimeError(f"y_j must be [B, K, S], got shape {tuple(y_j.shape)}")
    if y_j.shape != (B, K, S):
        raise RuntimeError(f"y_j shape {tuple(y_j.shape)} doesn't match [B={B}, K={K}, S={S}]")

    # Check g shape [B, G]
    if g.ndim != 2:
        raise RuntimeError(f"g must be [B, G], got shape {tuple(g.shape)}")
    if g.shape[0] != B:
        raise RuntimeError(f"g batch mismatch: {g.shape[0]} vs {B}")

    # Check mask if present (should be [B, K])
    if len(batch) >= 6:
        mask = batch[5]
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise RuntimeError(f"mask must be a tensor, got {type(mask)}")
            if mask.shape != (B, K):
                raise RuntimeError(f"mask shape {tuple(mask.shape)} doesn't match [B={B}, K={K}]")


# --------------------------------------------------------------------------------------
# Simple MSE Loss with Mask
# --------------------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Robust L2 via Huber (SmoothL1) with optional [B,K] mask.
    pred, target: [B, K, S]
    mask: [B, K] or None
    """
    # Per-(B,K) mean over species
    per_bk = F.smooth_l1_loss(pred, target, beta=1.0, reduction="none").mean(dim=-1)  # [B, K]
    if mask is not None:
        m = mask.float()
        return (per_bk * m).sum() / m.sum().clamp_min(1.0)
    return per_bk.mean()


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------

class EpochSetterCallback(Callback):
    """Log TF/horizon and reseed datasets each epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        # Existing logging
        tf_p = pl_module._get_teacher_forcing_prob(epoch)
        H = pl_module._get_rollout_horizon(epoch)
        pl_module.log("tf_prob", tf_p, on_epoch=True, prog_bar=False)
        pl_module.log("rollout_horizon", float(H), on_epoch=True, prog_bar=False)

        maybe_loaders = []
        if getattr(trainer, "train_dataloader", None) is not None:
            maybe_loaders.append(trainer.train_dataloader)
        if getattr(trainer, "val_dataloaders", None) is not None:
            maybe_loaders.append(trainer.val_dataloaders)

        for dl in maybe_loaders:
            if dl is None:
                continue
            dls = dl if isinstance(dl, (list, tuple)) else [dl]
            for _dl in dls:
                ds = getattr(_dl, "dataset", None)
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(epoch)

class SimpleStatsCallback(Callback):
    """Log basic epoch stats."""

    def __init__(self, py_logger: Optional[logging.Logger] = None):
        super().__init__()
        self.py_logger = py_logger or logging.getLogger("trainer")
        self._epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - (self._epoch_start_time or time.time())

        cm = trainer.callback_metrics
        train_loss = float(cm.get("train_loss", float("nan")))
        val_loss = float(cm.get("val_loss", float("nan")))

        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except:
            lr = float("nan")

        self.py_logger.info(
            f"Epoch {trainer.current_epoch:3d} | "
            f"train={train_loss:.3e} val={val_loss:.3e} | "
            f"lr={lr:.2e} | "
            f"time={elapsed:.1f}s"
        )


# --------------------------------------------------------------------------------------
# Lightning Module
# --------------------------------------------------------------------------------------

class AutoEncoderModule(LightningModule):
    """Simplified Lightning wrapper - MSE only, simple shape validation."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], work_dir: Union[str, Path]):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.py_logger = logging.getLogger(__name__)

        # Get values directly from config
        tcfg = cfg["training"]
        self.learning_rate = float(tcfg["lr"])
        self.min_lr = float(tcfg["min_lr"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.grad_clip = float(tcfg["gradient_clip"])
        self.epochs = int(tcfg["epochs"])
        self.warmup_epochs = int(tcfg["warmup_epochs"])

        # Expected shapes from config
        dcfg = cfg["dataset"]
        self.expected_K = int(dcfg["times_per_anchor"])
        self.expected_S = len(cfg["data"]["species_variables"])

        if self.expected_K <= 1:
            raise ValueError("This trainer assumes K > 1 (times_per_anchor > 1)")
        if self.expected_S <= 1:
            raise ValueError("This trainer assumes S > 1 (multiple species)")

        # Auxiliary losses
        aux = tcfg["auxiliary_losses"]
        self.rollout_enabled = bool(aux["rollout_enabled"])
        self.rollout_weight = float(aux["rollout_weight"])
        self.max_rollout_horizon = int(aux["rollout_horizon"])
        self.rollout_use_cached = bool(aux["rollout_use_cached_encoding"])
        self.rollout_warmup_epochs = int(aux["rollout_warmup_epochs"])

        self.semigroup_enabled = bool(aux["semigroup_enabled"])
        self.semigroup_weight = float(aux["semigroup_weight"])
        self.semigroup_warmup_epochs = int(aux["semigroup_warmup_epochs"])

        # Teacher forcing
        tf_cfg = aux["rollout_teacher_forcing"]
        self.tf_mode = str(tf_cfg["mode"])
        self.tf_start = float(tf_cfg["start_p"])
        self.tf_end = float(tf_cfg["end_p"])
        self.tf_end_epoch = int(tf_cfg["end_epoch"])

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
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate"
            }
        }

    def _get_teacher_forcing_prob(self, epoch: int) -> float:
        if self.tf_mode == "none":
            return 0.0
        if self.tf_mode == "constant":
            return self.tf_start
        if epoch >= self.tf_end_epoch:
            return self.tf_end
        t = epoch / max(1, self.tf_end_epoch)
        if self.tf_mode == "linear":
            return self.tf_start + (self.tf_end - self.tf_start) * t
        else:  # cosine_ramp
            return self.tf_end + 0.5 * (self.tf_start - self.tf_end) * (1 + math.cos(math.pi * t))

    def _get_rollout_horizon(self, epoch: int) -> int:
        if self.max_rollout_horizon <= 1:
            return 1
        t = min(max(epoch, 0), self.tf_end_epoch) / max(1, self.tf_end_epoch)
        H = 1 + int(round((self.max_rollout_horizon - 1) * t))
        return max(1, min(self.max_rollout_horizon, H))

    def training_step(self, batch, batch_idx):
        # Validate batch shape on first batch of each epoch
        if batch_idx == 0:
            validate_batch_shape(batch, self.expected_K, self.expected_S, batch_idx)

        # Unpack batch (validated shape)
        y_i, dt, y_j, g = batch[:4]
        mask = batch[5] if len(batch) >= 6 else None

        # Forward
        pred = self.model(y_i, dt, g)

        # MSE loss
        mse = masked_mse_loss(pred, y_j, mask)
        total = mse
        self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Rollout loss
        if self.rollout_enabled and self.rollout_weight > 0:
            H = self._get_rollout_horizon(self.current_epoch)
            tf_p = self._get_teacher_forcing_prob(self.current_epoch)
            rollout_warm = min(1.0, self.current_epoch / max(1, self.rollout_warmup_epochs))
            effective_rollout_w = self.rollout_weight * rollout_warm

            if effective_rollout_w > 1e-8:
                if self.rollout_use_cached:
                    rollout_loss = self._compute_rollout_loss_cached(y_i, g, y_j, dt, mask, H, tf_p)
                else:
                    rollout_loss = self._compute_rollout_loss(y_i, g, y_j, dt, mask, H, tf_p)

                if rollout_loss is not None and torch.isfinite(rollout_loss):
                    total = total + effective_rollout_w * rollout_loss
                    self.log("rollout_loss", rollout_loss, on_step=False, on_epoch=True, logger=True)

        # Semigroup loss
        if self.semigroup_enabled and self.semigroup_weight > 0:
            sg_warm = min(1.0, self.current_epoch / max(1, self.semigroup_warmup_epochs))
            effective_sg_w = self.semigroup_weight * sg_warm
            if effective_sg_w > 1e-8:
                sgl = self._compute_semigroup_loss(y_i, g)
                if sgl is not None and torch.isfinite(sgl):
                    total = total + effective_sg_w * sgl
                    self.log("semigroup_loss", sgl, on_step=False, on_epoch=True, logger=True)

        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at epoch {self.current_epoch}, batch {batch_idx}")

        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total

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
        """Autoregressive rollout loss with teacher forcing."""
        if horizon <= 0:
            return None

        B, K = dt_norm.shape
        S = y_i.shape[-1]
        H = min(K, max(1, horizon))

        y_curr = y_i
        losses = []

        for t in range(H):
            # One-step prediction
            y_pred = self.model.step(y_curr, dt_norm[:, t], g)
            y_true = y_j[:, t]

            # Mask for this timestep
            step_mask = mask[:, t] if mask is not None else None

            # Loss
            residual = y_pred - y_true
            residual = residual.clamp(-MAX_RESIDUAL, MAX_RESIDUAL)
            loss = residual.pow(2).mean(dim=-1)  # [B]

            if step_mask is not None:
                loss = (loss * step_mask.float()).sum() / step_mask.float().sum().clamp_min(1.0)
            else:
                loss = loss.mean()

            losses.append(loss)

            # Teacher forcing
            if t < H - 1:
                use_tf = (torch.rand(B, device=self.device) < tf_prob).float().unsqueeze(-1)
                if step_mask is not None:
                    use_tf = use_tf * step_mask.float().unsqueeze(-1)
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
        """Cached encoding rollout loss."""
        if horizon <= 0:
            return None

        B, K = dt_norm.shape
        S = y_i.shape[-1]
        H = min(K, max(1, horizon))
        predict_delta = getattr(self.model, "predict_delta", False)
        decoder_condition_on_g = getattr(self.model, "decoder_condition_on_g", False)

        z_curr = self.model.encoder(y_i, g)
        use_lowrank = not self.model.dynamics.use_S

        y_curr = y_i
        losses = []

        for t in range(H):
            dtk_norm = dt_norm[:, t]

            # Latent step
            if use_lowrank:
                z_next = self.model.dynamics.propagate_K_lowrank(z_curr, g, dtk_norm.view(-1, 1))[:, 0]
            else:
                z_next = self.model.dynamics.step(z_curr, dtk_norm, g)

            # Decode
            dec_in = torch.cat([z_next, g], dim=-1) if decoder_condition_on_g else z_next
            y_decoded = self.model.decoder(dec_in)
            y_pred = y_curr + y_decoded if predict_delta else y_decoded

            y_true = y_j[:, t]

            # Mask for this timestep
            step_mask = mask[:, t] if mask is not None else None

            # Loss
            residual = y_pred - y_true
            residual = residual.clamp(-MAX_RESIDUAL, MAX_RESIDUAL)
            loss = residual.pow(2).mean(dim=-1)  # [B]

            if step_mask is not None:
                loss = (loss * step_mask.float()).sum() / step_mask.float().sum().clamp_min(1.0)
            else:
                loss = loss.mean()

            losses.append(loss)

            # Teacher forcing
            if t < H - 1:
                use_tf = (torch.rand(B, device=y_i.device) < tf_prob).float().unsqueeze(-1)
                if step_mask is not None:
                    use_tf = use_tf * step_mask.float().unsqueeze(-1)

                with torch.no_grad():
                    z_gt = self.model.encoder(y_true, g)
                z_curr = use_tf * z_gt + (1.0 - use_tf) * z_next.detach()

                if predict_delta:
                    y_curr = use_tf * y_true + (1.0 - use_tf) * y_pred.detach()

        return torch.stack(losses).mean() if losses else None

    def _compute_semigroup_loss(self, y_i: torch.Tensor, g: torch.Tensor) -> Optional[torch.Tensor]:
        """Semigroup consistency: f(dt_a + dt_b, y) ≈ f(dt_b, f(dt_a, y))"""
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

        # Two sequential steps
        y_a = self.model(y_i, dt_a.unsqueeze(1), g)[:, 0]
        y_ab_seq = self.model(y_a, dt_b.unsqueeze(1), g)[:, 0]

        # Single combined step
        y_ab_combined = self.model(y_i, dt_ab.unsqueeze(1), g)[:, 0]

        return F.mse_loss(y_ab_seq, y_ab_combined)

    def validation_step(self, batch, batch_idx):
        # Validate shape on first val batch
        if batch_idx == 0:
            validate_batch_shape(batch, self.expected_K, self.expected_S, batch_idx)

        y_i, dt, y_j, g = batch[:4]
        mask = batch[5] if len(batch) >= 6 else None

        pred = self.model(y_i, dt, g)

        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite predictions in validation batch {batch_idx}")

        loss = masked_mse_loss(pred, y_j, mask)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, logger=True)
        return loss


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

        lcfg = self.cfg["lightning"]
        tcfg = self.cfg["training"]

        # Use config values directly
        precision = str(lcfg["precision"])
        devices = lcfg["devices"]
        accelerator = lcfg["accelerator"]
        accumulate = int(lcfg["accumulate_grad_batches"])
        strategy = lcfg["strategy"]
        num_sanity_val_steps = int(lcfg["num_sanity_val_steps"])
        max_epochs = int(tcfg["epochs"])

        # --- SMALL OVERRIDE: force CPU if MPS is available and force_cpu_on_mps is True ---
        import os
        force_cpu_on_mps = bool(lcfg.get("force_cpu_on_mps", True))
        try:
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            has_mps = False
        if force_cpu_on_mps and has_mps:
            self.logger.warning("MPS detected, forcing CPU to avoid missing ops (e.g., torch.linalg.qr).")
            accelerator = "cpu"
            devices = 1
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # -------------------------------------------------------------------------------

        # Checkpointing
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

        # Early stopping
        es_patience = int(tcfg.get("early_stop_patience", 30))
        early_stop = None
        if es_patience > 0:
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=es_patience,
                mode="min",
                min_delta=1e-6,
                verbose=False,
            )

        # Callbacks
        lrmon = LearningRateMonitor(logging_interval="epoch")
        csvlog = CSVLogger(save_dir=str(self.work_dir), name="csv_logs")
        tblog = TensorBoardLogger(save_dir=str(self.work_dir), name="tb_logs")

        epoch_setter = EpochSetterCallback()
        stats_cb = SimpleStatsCallback(py_logger=self.logger)

        callbacks = [ckpt_cb, lrmon, epoch_setter, stats_cb]
        if early_stop is not None:
            callbacks.append(early_stop)

        # Simple resume logic: if training.resume is a valid path, resume from it
        resume_ckpt = None
        resume_cfg = tcfg.get("resume")
        if resume_cfg and isinstance(resume_cfg, (str, Path)):
            candidate = Path(resume_cfg)
            if candidate.exists():
                resume_ckpt = str(candidate)
                self.logger.info(f"Resuming from: {resume_ckpt}")
            else:
                self.logger.warning(f"Resume path not found: {candidate}")

        # Lightning Trainer
        trainer = LightningTrainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate,
            num_sanity_val_steps=num_sanity_val_steps,
            logger=[csvlog, tblog],
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=float(tcfg["gradient_clip"]),
            gradient_clip_algorithm="norm",
            benchmark=True,
        )

        # Fit
        trainer.fit(
            module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=resume_ckpt,
        )

        # Best val loss
        best_val = None
        if ckpt_cb.best_model_score is not None:
            best_val = float(ckpt_cb.best_model_score.cpu().item())

        # Save best model
        if ckpt_cb.best_model_path:
            state = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", {})
            model_state = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
            torch.save(
                {"model": model_state, "best_val_loss": best_val, "config": self.cfg},
                self.work_dir / "best_model.pt"
            )
            self.logger.info(f"Saved best_model.pt (val_loss={best_val:.6e})")

        return best_val if best_val is not None else float("inf")