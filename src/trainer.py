#!/usr/bin/env python3
"""
trainer.py

Autoregressive training with:
  - Teacher forcing (scheduled sampling)
  - Optional free-run burn-in (drift off-manifold before computing loss)
  - Optional rollout curriculum (start short, ramp long)
  - Configurable validation burn-in to match training regime

Keeps:
  - Same model architecture (model.py unchanged)
  - Same loss structure (log-space + normalized-space terms)

This file uses Lightning if available. It supports both:
  - lightning.pytorch (Lightning 2.x)
  - pytorch_lightning (older)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning import compatibility
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger


# ---------------------------- Loss ----------------------------


def _safe_clamp_std(std: torch.Tensor, min_std: float = 1e-10) -> torch.Tensor:
    """Clamp standard deviation to avoid division by zero."""
    return torch.clamp(std, min=min_std)


def _stack_log_stats(
    manifest: Dict, species_variables: List[str], device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract log-space statistics from manifest for all species.
    
    Returns:
        (log_means, log_stds, log_mins, log_maxs) tensors of shape [S]
    """
    per = manifest.get("per_key_stats") or manifest.get("stats") or {}
    log_means = []
    log_stds = []
    log_mins = []
    log_maxs = []
    
    for s in species_variables:
        st = per.get(s, {})
        log_means.append(float(st.get("log_mean", 0.0)))
        log_stds.append(float(st.get("log_std", 1.0)))
        log_mins.append(float(st.get("log_min", -30.0)))
        log_maxs.append(float(st.get("log_max", 10.0)))
    
    log_means_t = torch.tensor(log_means, device=device, dtype=dtype)
    log_stds_t = _safe_clamp_std(torch.tensor(log_stds, device=device, dtype=dtype))
    log_mins_t = torch.tensor(log_mins, device=device, dtype=dtype)
    log_maxs_t = torch.tensor(log_maxs, device=device, dtype=dtype)
    
    return log_means_t, log_stds_t, log_mins_t, log_maxs_t


class AdaptiveStiffLoss(nn.Module):
    """
    Combined loss for stiff ODE systems:
      - L1 in log10-physical space (derived from z using log_mean/log_std)
      - MSE in z-space
      - Per-species weighting based on log-range to bias stiff species
    """

    def __init__(
        self,
        *,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        log_mins: torch.Tensor,
        log_maxs: torch.Tensor,
        lambda_phys: float = 1.0,
        lambda_z: float = 0.1,
        eps_phys: float = 1e-20,
        w_species_clamp: Tuple[float, float] = (0.5, 2.0),
        range_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        # Register statistics as buffers (will be moved to correct device)
        self.register_buffer("log_means", log_means)
        self.register_buffer("log_stds", log_stds)
        self.register_buffer("log_mins", log_mins)
        self.register_buffer("log_maxs", log_maxs)

        # Loss weights
        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.eps_phys = float(eps_phys)
        self.wmin = float(w_species_clamp[0])
        self.wmax = float(w_species_clamp[1])
        self.range_eps = float(range_eps)

        # Compute per-species weights: wider log-range => heavier weight
        log_range = torch.clamp(self.log_maxs - self.log_mins, min=self.range_eps)
        w = log_range / torch.mean(log_range)
        self.register_buffer("w_species", torch.clamp(w, self.wmin, self.wmax))

    def z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        """Convert z-space to log10-physical space."""
        return z * self.log_stds + self.log_means

    def forward(self, pred_z: torch.Tensor, true_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_z: [B, K, S] predictions in z-space
            true_z: [B, K, S] targets in z-space
            
        Returns:
            Dictionary with loss components
        """
        # Z-space MSE loss
        z_mse = F.mse_loss(pred_z, true_z, reduction="none")  # [B, K, S]
        z_mse = (z_mse * self.w_species).mean()

        # Physical log-space L1 loss
        pred_log = self.z_to_log10(pred_z)
        true_log = self.z_to_log10(true_z)
        phys_l1 = torch.abs(pred_log - true_log)  # [B, K, S]
        phys_l1 = (phys_l1 * self.w_species).mean()

        # Combined loss
        total = self.lambda_phys * phys_l1 + self.lambda_z * z_mse

        # Diagnostic metric (computed without gradient)
        with torch.no_grad():
            mean_abs_log10 = torch.abs(pred_log - true_log).mean()

        return {
            "loss_total": total,
            "loss_phys_l1": phys_l1,
            "loss_z_mse": z_mse,
            "mean_abs_log10": mean_abs_log10,
        }


# ---------------------------- Schedules ----------------------------


@dataclass(frozen=True)
class TeacherForcingSchedule:
    """Schedule for teacher forcing probability decay."""
    start: float = 1.0
    end: float = 0.0
    decay_epochs: int = 0
    mode: str = "linear"  # linear | cosine

    def value(self, epoch: int) -> float:
        """Get teacher forcing probability for given epoch."""
        if self.decay_epochs <= 0:
            return float(self.start)
        t = min(max(epoch, 0), self.decay_epochs)
        u = t / float(self.decay_epochs)
        if self.mode == "cosine":
            u = 0.5 * (1.0 - math.cos(math.pi * u))
        return float(self.start + (self.end - self.start) * u)


@dataclass(frozen=True)
class RolloutCurriculum:
    """Curriculum for gradually increasing rollout length."""
    enabled: bool = False
    start_steps: int = 1
    ramp_epochs: int = 0  # ramp from start_steps to max over this many epochs

    def steps(self, epoch: int, max_steps: int) -> int:
        """Get number of rollout steps for given epoch."""
        if not self.enabled:
            return max_steps
        if self.ramp_epochs <= 0:
            return max_steps
        t = min(max(epoch, 0), self.ramp_epochs)
        u = t / float(self.ramp_epochs)
        k = int(round(self.start_steps + (max_steps - self.start_steps) * u))
        return max(1, min(max_steps, k))


# ---------------------------- Lightning Module ----------------------------


class FlowMapRolloutModule(pl.LightningModule):
    """
    Lightning module for autoregressive flow-map training.
    
    Features:
      - Configurable burn-in for both training and validation
      - Teacher forcing with scheduled decay
      - Rollout curriculum (optional)
    """

    def __init__(
        self,
        *,
        cfg: Dict,
        model: nn.Module,
        normalization_manifest: Dict,
        species_variables: List[str],
        work_dir: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.work_dir = Path(work_dir)

        # Training configuration
        tcfg = cfg.get("training", {})
        self.rollout_steps = int(tcfg.get("rollout_steps", 1))
        self.burn_in_steps = int(tcfg.get("burn_in_steps", 0))
        self.total_steps = self.rollout_steps + self.burn_in_steps

        # Burn-in configuration
        self.burn_in_add_noise_std = float(tcfg.get("burn_in_additive_noise_std", 0.0))
        self.burn_in_loss_weight = float(tcfg.get("burn_in_loss_weight", 0.0))
        
        # Validation burn-in: defaults to matching training burn-in
        # Set to 0 for "cold start" validation if desired
        self.val_burn_in_steps = int(tcfg.get("val_burn_in_steps", self.burn_in_steps))

        # Teacher forcing schedule
        tf_cfg = tcfg.get("teacher_forcing", {})
        self.tf_sched = TeacherForcingSchedule(
            start=float(tf_cfg.get("start", 1.0)),
            end=float(tf_cfg.get("end", 0.0)),
            decay_epochs=int(tf_cfg.get("decay_epochs", 0)),
            mode=str(tf_cfg.get("mode", "linear")).lower(),
        )

        # Rollout curriculum
        cur_cfg = tcfg.get("curriculum", {})
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            start_steps=int(cur_cfg.get("start_steps", 1)),
            ramp_epochs=int(cur_cfg.get("ramp_epochs", 0)),
        )

        # Build loss function (buffers created on CPU, moved in on_fit_start)
        log_means, log_stds, log_mins, log_maxs = _stack_log_stats(
            normalization_manifest,
            species_variables,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        lcfg = tcfg.get("loss", {})
        self.criterion = AdaptiveStiffLoss(
            log_means=log_means,
            log_stds=log_stds,
            log_mins=log_mins,
            log_maxs=log_maxs,
            lambda_phys=float(lcfg.get("lambda_phys", 1.0)),
            lambda_z=float(lcfg.get("lambda_z", 0.1)),
            eps_phys=float(lcfg.get("eps_phys", 1e-20)),
            w_species_clamp=tuple(lcfg.get("w_species_clamp", [0.5, 2.0])),
            range_eps=float(lcfg.get("range_eps", 1e-6)),
        )

        # Optimizer parameters
        self.lr = float(tcfg.get("lr", 1e-3))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))

    def on_fit_start(self) -> None:
        """Move criterion to correct device at start of training."""
        # Simply move criterion to device; buffers are already float32
        self.criterion = self.criterion.to(self.device)

    def configure_optimizers(self):
        """Configure optimizer."""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return opt

    def _autoregressive_unroll(
        self,
        y0: torch.Tensor,
        dt_norm: torch.Tensor,
        y_true: torch.Tensor,
        g: torch.Tensor,
        *,
        teacher_forcing_p: float,
        burn_in_steps: int,
        add_noise_std: float,
        max_rollout_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive unrolling with optional burn-in and teacher forcing.
        
        Args:
            y0: [B, S] initial state in z-space
            dt_norm: [B] or scalar normalized time step
            y_true: [B, K, S] ground truth for next K steps
            g: [B, G] global parameters
            teacher_forcing_p: probability of using ground truth as next input
            burn_in_steps: number of free-run steps before applying teacher forcing
            add_noise_std: noise to add during burn-in (helps with off-manifold drift)
            max_rollout_steps: maximum rollout steps (curriculum-controlled)
            
        Returns:
            (pred_all, true_all): predictions and targets of shape [B, K_eff, S]
        """
        B, K_total, S = y_true.shape
        K_eff = min(K_total, burn_in_steps + max_rollout_steps)
        burn = min(burn_in_steps, K_eff)

        state = y0
        preds: List[torch.Tensor] = []

        # Ensure dt shape is compatible with model: [B, 1, 1]
        if dt_norm.ndim == 0:
            dt_step = dt_norm.view(1).expand(B)
        else:
            dt_step = dt_norm
        dt_step = dt_step.to(state.device)

        for t in range(K_eff):
            # Model expects dt_norm [B, 1, 1] to produce [B, 1, S]
            out = self.model(state, dt_step.view(B, 1, 1), g)
            pred = out[:, 0, :]  # [B, S]
            preds.append(pred)

            # Stop after last prediction (no need to update state)
            if t == K_eff - 1:
                break

            # Determine next input state
            if t < burn:
                # Free-run burn-in: use prediction with optional noise
                state = pred
                if add_noise_std > 0.0:
                    state = state + add_noise_std * torch.randn_like(state)
            else:
                # Scheduled sampling: mix ground truth and predictions
                if teacher_forcing_p >= 1.0:
                    state = y_true[:, t, :]
                elif teacher_forcing_p <= 0.0:
                    state = pred
                else:
                    mask = (torch.rand((B,), device=state.device) < teacher_forcing_p).view(B, 1)
                    state = torch.where(mask, y_true[:, t, :], pred)

        pred_all = torch.stack(preds, dim=1)  # [B, K_eff, S]
        true_all = y_true[:, :K_eff, :]       # [B, K_eff, S]
        return pred_all, true_all

    def training_step(self, batch, batch_idx):
        """Training step with burn-in and teacher forcing."""
        y0, dt_norm, y_seq, g = batch

        # Get curriculum-adjusted rollout steps and teacher forcing probability
        k_roll = self.curriculum.steps(int(self.current_epoch), self.rollout_steps)
        tf_p = self.tf_sched.value(int(self.current_epoch))

        # Autoregressive unroll
        pred_all, true_all = self._autoregressive_unroll(
            y0=y0,
            dt_norm=dt_norm,
            y_true=y_seq,
            g=g,
            teacher_forcing_p=tf_p,
            burn_in_steps=self.burn_in_steps,
            add_noise_std=self.burn_in_add_noise_std,
            max_rollout_steps=k_roll,
        )

        # Compute loss on post-burn-in predictions
        burn = min(self.burn_in_steps, pred_all.shape[1])
        main_pred = pred_all[:, burn:, :]
        main_true = true_all[:, burn:, :]
        comps_main = self.criterion(main_pred, main_true)

        loss = comps_main["loss_total"]

        # Optionally add burn-in loss with reduced weight
        if burn > 0 and self.burn_in_loss_weight > 0.0:
            comps_burn = self.criterion(pred_all[:, :burn, :], true_all[:, :burn, :])
            loss = loss + self.burn_in_loss_weight * comps_burn["loss_total"]

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mean_abs_log10", comps_main["mean_abs_log10"], on_step=False, on_epoch=True)
        self.log("tf_prob", torch.tensor(tf_p, device=self.device), on_step=False, on_epoch=True)
        self.log("k_roll", torch.tensor(k_roll, device=self.device), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with configurable burn-in.
        
        By default, validation uses the same burn-in as training to ensure metrics
        are comparable. Set training.val_burn_in_steps=0 in config for cold-start
        validation (evaluating model's ability to predict from arbitrary initial states).
        """
        y0, dt_norm, y_seq, g = batch

        # Pure autoregressive (no teacher forcing, no noise)
        # Use full rollout steps (no curriculum in validation)
        pred_all, true_all = self._autoregressive_unroll(
            y0=y0,
            dt_norm=dt_norm,
            y_true=y_seq,
            g=g,
            teacher_forcing_p=0.0,
            burn_in_steps=self.val_burn_in_steps,
            add_noise_std=0.0,
            max_rollout_steps=self.rollout_steps,
        )

        # Compute loss on post-burn-in predictions (matching training regime)
        burn = min(self.val_burn_in_steps, pred_all.shape[1])
        main_pred = pred_all[:, burn:, :]
        main_true = true_all[:, burn:, :]
        comps = self.criterion(main_pred, main_true)

        # Log metrics
        self.log("val_loss", comps["loss_total"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mean_abs_log10", comps["mean_abs_log10"], on_step=False, on_epoch=True)

        # Also log cold-start metrics if using burn-in
        if self.val_burn_in_steps > 0:
            comps_cold = self.criterion(pred_all, true_all)
            self.log("val_loss_cold_start", comps_cold["loss_total"], on_step=False, on_epoch=True)
            self.log("val_mean_abs_log10_cold_start", comps_cold["mean_abs_log10"], on_step=False, on_epoch=True)

        return comps["loss_total"]


def _get_precision_for_accelerator(cfg_precision: str, accelerator: str) -> str:
    """
    Get appropriate precision string for the given accelerator.
    
    bf16-mixed requires GPU support. On CPU, fall back to 32-bit.
    
    Args:
        cfg_precision: Requested precision from config
        accelerator: "gpu" or "cpu"
        
    Returns:
        Valid precision string for the accelerator
    """
    # Normalize precision string
    precision = str(cfg_precision).lower().strip()
    
    # CPU doesn't support bf16-mixed or 16-mixed well
    if accelerator == "cpu":
        if precision in ("bf16-mixed", "16-mixed", "bf16", "16"):
            import warnings
            warnings.warn(
                f"Precision '{cfg_precision}' is not well-supported on CPU. "
                f"Falling back to '32-true'. Set training.precision='32-true' to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            return "32-true"
    
    return precision


def build_lightning_trainer(cfg: Dict, *, work_dir: Path) -> pl.Trainer:
    """
    Build PyTorch Lightning Trainer from configuration.
    
    Args:
        cfg: Configuration dictionary
        work_dir: Directory for checkpoints and logs
        
    Returns:
        Configured Trainer instance
    """
    tcfg = cfg.get("training", {})
    max_epochs = int(tcfg.get("max_epochs", 100))

    # Checkpointing configuration
    cpcfg = cfg.get("checkpoint", {})
    ckpt_every = int(cpcfg.get("save_every_n_epochs", 1))

    callbacks = [
        ModelCheckpoint(
            dirpath=str(work_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            every_n_epochs=ckpt_every,
        )
    ]

    # Logger
    logger = CSVLogger(save_dir=str(work_dir), name="csv", flush_logs_every_n_steps=100)

    # Hardware configuration
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = int(tcfg.get("devices", 1)) if accelerator == "gpu" else 1
    
    # Get precision with CPU fallback
    cfg_precision = str(tcfg.get("precision", "bf16-mixed"))
    precision = _get_precision_for_accelerator(cfg_precision, accelerator)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 200)),
        enable_checkpointing=True,
        enable_progress_bar=bool(tcfg.get("enable_progress_bar", False)),
        gradient_clip_val=float(tcfg.get("grad_clip", 0.0)),
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
    )

    return trainer
