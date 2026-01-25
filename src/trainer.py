#!/usr/bin/env python3
"""
trainer.py - PyTorch Lightning training module and trainer factory.

This module implements the training loop for flow-map models, including:
    - Autoregressive rollout with teacher forcing
    - Burn-in period for trajectory stabilization
    - Rollout length curriculum (optional)
    - Physics-aware adaptive loss
    - Learning rate scheduling with warmup

The core training paradigm is autoregressive: at each step, the model
predicts y_{t+1} from y_t (or ground truth if teacher forcing), and
the loss is computed over the entire predicted sequence.

Key Features:
    - TeacherForcingSchedule: Decays teacher forcing probability over epochs
    - RolloutCurriculum: Optionally ramps up rollout length during training
    - LateStageRolloutOverride: Use longer rollouts in final epochs
    - AdaptiveLoss: Combines MSE with physics-informed penalties
    - Burn-in: Optional warm-up period before loss computation

Usage:
    module = FlowMapRolloutModule(cfg, model, manifest, species_vars, work_dir)
    module.set_dataloaders(train_dl, val_dl, test_dl)
    trainer = build_lightning_trainer(cfg, work_dir=work_dir)
    trainer.fit(module)
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import normalize_dt_shape

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl  # type: ignore


# ==============================================================================
# Logging Callbacks
# ==============================================================================


class CSVLoggerCallback(pl.Callback):
    """
    Minimal CSV logger that writes metrics to work_dir/metrics.csv.

    This callback captures all logged metrics at epoch boundaries and
    writes them to a CSV file for easy analysis and plotting. Only logs
    at epoch end (not per-step) to avoid excessive I/O.

    Args:
        work_dir: Directory to write metrics.csv
        filename: Name of the CSV file (default: "metrics.csv")

    Output format:
        epoch,train_loss,val_loss,train_z_mae,val_z_mae,...
        0,0.123,0.145,0.089,0.092,...
        1,0.098,0.112,0.067,0.071,...
    """

    def __init__(self, work_dir: Path, filename: str = "metrics.csv") -> None:
        super().__init__()
        self.path = Path(work_dir) / filename
        self._fieldnames: Optional[List[str]] = None
        self._initialized = False

    def _init_writer(self, keys: List[str]) -> None:
        """Initialize CSV file with header row."""
        self._fieldnames = ["epoch"] + keys
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._initialized = True

    def _append_row(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics to the CSV file."""
        if self._fieldnames is None:
            return

        row = {"epoch": epoch}
        for k in self._fieldnames:
            if k == "epoch":
                continue
            v = metrics.get(k, None)
            if v is None:
                continue
            try:
                if hasattr(v, "detach"):
                    v = v.detach().cpu().item()
                row[k] = float(v)
            except Exception:
                pass

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def _log_epoch(self, trainer: pl.Trainer) -> None:
        """Write one epoch row to CSV (rank 0 only; skips sanity checking)."""
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        metrics = dict(trainer.callback_metrics or {})
        keys = sorted([k for k in metrics.keys() if isinstance(k, str)])

        if not self._initialized:
            self._init_writer(keys)

        self._append_row(trainer.current_epoch, metrics)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metrics at the end of each validation epoch."""
        self._log_epoch(trainer)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log metrics at the end of training epoch (fallback if no validation)."""
        num_val = getattr(trainer, "num_val_batches", 0)
        try:
            has_val = (
                any(int(x) > 0 for x in num_val)
                if isinstance(num_val, (list, tuple))
                else int(num_val) > 0
            )
        except Exception:
            has_val = True

        if not has_val:
            self._log_epoch(trainer)


# ==============================================================================
# Loss Function
# ==============================================================================


class AdaptiveLoss(nn.Module):
    """
    Physics-informed loss for chemical kinetics rollouts.

    The model predicts z-space states, where each species concentration is
    represented in a normalized log10 space:
        z = (log10(y) - mean_log10) / std_log10

    This loss combines three components:

    1. Data fidelity in z-space:
       - mse_z: mean squared error of (pred_z - true_z)
       - z_mae: mean absolute error of (pred_z - true_z)

    2. Physics plausibility penalty in log10 space:
       Penalizes predicted log10 values outside observed bounds.

    3. Logging-only metrics in log10 space:
       For interpretability, reports errors after converting back to log10 space.

    Total loss: loss_total = mse_z + lambda_z * z_mae + lambda_phys * phys_penalty

    Args:
        log_means: Per-species log10 means [1, 1, S]
        log_stds: Per-species log10 standard deviations [1, 1, S]
        log_mins: Per-species log10 minimums [1, 1, S]
        log_maxs: Per-species log10 maximums [1, 1, S]
        lambda_phys: Weight for physics penalty term
        lambda_z: Weight for z-space MAE term
    """

    def __init__(
        self,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        log_mins: torch.Tensor,
        log_maxs: torch.Tensor,
        *,
        lambda_phys: float = 1.0,
        lambda_z: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("log_means", log_means)
        self.register_buffer("log_stds", log_stds)
        self.register_buffer("log_mins", log_mins)
        self.register_buffer("log_maxs", log_maxs)

        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)

    def forward(
        self,
        pred_z: torch.Tensor,
        true_z: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss and metrics.

        Args:
            pred_z: Predicted states in z-space [B, K, S]
            true_z: Ground truth states in z-space [B, K, S]
            weights: Optional per-step weights [K] for burn-in weighting.
                     If provided, losses are normalized by the sum of active weights
                     (rather than the total number of elements), preventing gradient
                     dilution when some steps are downweighted or masked.

        Returns:
            Dictionary containing loss_total and component metrics.
        """
        if pred_z.shape != true_z.shape:
            raise ValueError(
                f"Shape mismatch: pred_z={tuple(pred_z.shape)} true_z={tuple(true_z.shape)}"
            )

        B, K, S = pred_z.shape

        if weights is not None:
            if weights.ndim != 1 or int(weights.shape[0]) != int(K):
                raise ValueError(
                    f"weights must have shape [K] with K={K}; got {tuple(weights.shape)}"
                )
            weights = weights.to(device=pred_z.device, dtype=pred_z.dtype)

        delta_z = pred_z - true_z

        # Weighted z-space errors (properly normalized by sum of active weights)
        if weights is not None:
            w = weights.view(1, K, 1)  # broadcast over batch/species
            w_sum = weights.sum(dtype=torch.float32)
            denom = (w_sum * float(B * S)).clamp_min(1e-12)

            mse_z = (delta_z.pow(2).mul(w).sum(dtype=torch.float32)) / denom
            z_mae = (delta_z.abs().mul(w).sum(dtype=torch.float32)) / denom
        else:
            mse_z = delta_z.pow(2).mean()
            z_mae = delta_z.abs().mean()

        pred_log = pred_z * self.log_stds + self.log_means
        true_log = true_z * self.log_stds + self.log_means
        delta_log = pred_log - true_log
        mse_log = delta_log.pow(2).mean()
        log_mae = delta_log.abs().mean()

        below = (self.log_mins - pred_log).clamp(min=0.0)
        above = (pred_log - self.log_maxs).clamp(min=0.0)

        # Weighted physics penalty (same normalization as z-space terms)
        if weights is not None:
            w = weights.view(1, K, 1)
            w_sum = weights.sum(dtype=torch.float32)
            denom = (w_sum * float(B * S)).clamp_min(1e-12)

            phys_penalty = ((below + above).pow(2).mul(w).sum(dtype=torch.float32)) / denom
        else:
            phys_penalty = (below + above).pow(2).mean()

        loss_total = mse_z + (self.lambda_z * z_mae) + (self.lambda_phys * phys_penalty)

        return {
            "loss_total": loss_total,
            "mse_z": mse_z,
            "mse": mse_z,
            "z_mae": z_mae,
            "mse_log": mse_log,
            "log_mae": log_mae,
            "phys_penalty": phys_penalty,
        }



def build_loss_buffers(
    manifest: Dict[str, Any],
    species: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract species normalization statistics from manifest into device tensors.

    Args:
        manifest: Normalization manifest dictionary containing per_key_stats
        species: Ordered list of species variable names
        device: Target device for tensors

    Returns:
        Tuple of (log_means, log_stds, log_mins, log_maxs), each [1, 1, S]

    Raises:
        KeyError: If manifest is missing required statistics
    """
    stats = manifest.get("per_key_stats")
    if stats is None:
        stats = manifest.get("species_stats") or manifest.get("stats")
    if stats is None:
        raise KeyError(
            "Normalization manifest missing 'per_key_stats' "
            "(or legacy 'species_stats'/'stats')"
        )

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for s in species:
        if s not in stats:
            raise KeyError(f"Normalization stats missing for species key: {s}")

        entry = stats[s]
        mean_val = entry.get("log10_mean", entry.get("log_mean"))
        std_val = entry.get("log10_std", entry.get("log_std"))
        min_val = entry.get("log10_min", entry.get("log_min"))
        max_val = entry.get("log10_max", entry.get("log_max"))

        if mean_val is None or std_val is None or min_val is None or max_val is None:
            raise KeyError(
                f"Missing log statistics for species '{s}'. "
                "Expected log10_* or log_* entries (mean, std, min, max)."
            )

        means.append(float(mean_val))
        stds.append(float(std_val))
        mins.append(float(min_val))
        maxs.append(float(max_val))

    log_means = torch.tensor(means, device=device).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device).view(1, 1, -1)
    log_mins = torch.tensor(mins, device=device).view(1, 1, -1)
    log_maxs = torch.tensor(maxs, device=device).view(1, 1, -1)
    return log_means, log_stds, log_mins, log_maxs


# ==============================================================================
# Schedules / Curricula
# ==============================================================================


@dataclass
class TeacherForcingSchedule:
    """
    Teacher forcing schedule with support for exponential, linear, and cosine decay.

    Modes:
        - exponential: prob(epoch) = max(end, start * decay^epoch)
        - linear: prob(epoch) = max(end, start - (start - end) * epoch / decay_epochs)
        - cosine: prob(epoch) = end + (start - end) * 0.5 * (1 + cos(pi * epoch / decay_epochs))

    Args:
        start: Initial teacher forcing probability (default: 1.0)
        end: Final/minimum teacher forcing probability (default: 0.0)
        decay_epochs: Number of epochs over which to decay (for linear/cosine)
        decay: Exponential decay factor per epoch (for exponential mode)
        mode: Decay mode - "exponential", "linear", or "cosine"
    """

    start: float = 1.0
    end: float = 0.0
    decay_epochs: int = 50
    decay: float = 0.98
    mode: str = "exponential"

    def prob(self, epoch: int) -> float:
        """Compute teacher forcing probability for the given epoch."""
        mode = self.mode.lower().strip()

        if mode == "linear":
            if self.decay_epochs <= 0:
                return float(self.end)
            frac = min(1.0, float(epoch) / float(self.decay_epochs))
            p = self.start - (self.start - self.end) * frac
            return max(self.end, float(p))

        elif mode == "cosine":
            if self.decay_epochs <= 0:
                return float(self.end)
            frac = min(1.0, float(epoch) / float(self.decay_epochs))
            p = self.end + (self.start - self.end) * 0.5 * (1.0 + math.cos(math.pi * frac))
            return max(self.end, float(p))

        else:
            p = self.start * (self.decay ** epoch)
            return max(self.end, float(p))


@dataclass
class RolloutCurriculum:
    """
    Rollout length curriculum that ramps K_roll over epochs.

    If enabled and ramp_epochs > 0, linearly interpolates from start_k to max_k
    over the specified number of epochs.

    Args:
        enabled: Whether curriculum is enabled
        start_k: Initial rollout length
        max_k: Maximum rollout length
        ramp_epochs: Number of epochs to ramp from start_k to max_k
    """

    enabled: bool
    start_k: int
    max_k: int
    ramp_epochs: int = 0

    def k_roll(self, epoch: int, max_epochs: int) -> int:
        """
        Compute rollout steps for the given epoch.

        Args:
            epoch: Current epoch
            max_epochs: Total training epochs (for safety bounds)

        Returns:
            Rollout length bounded to [start_k, max_k]
        """
        if not self.enabled:
            return self.max_k

        if self.ramp_epochs <= 0:
            return int(self.max_k)

        e = max(0, min(epoch, max_epochs))
        frac = min(1.0, float(e) / float(max(1, self.ramp_epochs)))
        k = int(round(self.start_k + frac * (self.max_k - self.start_k)))
        return int(max(self.start_k, min(self.max_k, k)))


@dataclass
class LateStageRolloutOverride:
    """
    Override rollout length in late training stages.

    When enabled and within the final epochs, overrides the rollout length
    with a longer value for fine-tuning on extended sequences.

    Args:
        enabled: Whether override is enabled
        long_rollout_steps: Override rollout steps
        final_epochs: Number of final epochs for which override applies
        apply_to_validation: Whether to apply override to validation
        apply_to_test: Whether to apply override to test
    """

    enabled: bool
    long_rollout_steps: int
    final_epochs: int
    apply_to_validation: bool = True
    apply_to_test: bool = True

    def applies(self, epoch: int, max_epochs: int, stage: str) -> bool:
        """
        Determine whether late-stage override applies.

        Args:
            epoch: Current epoch
            max_epochs: Total training epochs
            stage: One of "train", "val", "test"

        Returns:
            True if override should be used
        """
        if not self.enabled or self.final_epochs <= 0:
            return False

        if stage == "val" and not self.apply_to_validation:
            return False
        if stage == "test" and not self.apply_to_test:
            return False

        return epoch >= max(0, max_epochs - self.final_epochs)


# ==============================================================================
# Lightning Module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """
    PyTorch Lightning module for training flow-map rollout models.

    Implements autoregressive rollout with teacher forcing and optional burn-in.
    The model operates on z-space (normalized) states.

    Args:
        cfg: Configuration dictionary containing training parameters
        model: Neural network model with forward_step method
        normalization_manifest: Manifest with normalization statistics
        species_variables: List of species variable names
        work_dir: Working directory for outputs

    Configuration Keys (in cfg["training"]):
        - max_epochs: Total training epochs
        - lr: Learning rate
        - weight_decay: AdamW weight decay
        - rollout_steps: Base rollout length
        - burn_in_steps: Steps to run open-loop before teacher forcing
        - val_burn_in_steps: Steps to run open-loop for validation
        - burn_in_noise_std: Noise to add during burn-in
        - burn_in_loss_weight: Weight for burn-in steps in loss (0-1)
        - teacher_forcing: Schedule configuration dict
        - curriculum: Rollout curriculum configuration dict
        - long_rollout: Late-stage override configuration dict
        - scheduler: LR scheduler configuration dict
        - loss: Loss function weights dict
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        normalization_manifest: Dict[str, Any],
        species_variables: List[str],
        work_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.normalization_manifest = normalization_manifest
        self.species_variables = list(species_variables)
        self.work_dir = Path(work_dir) if work_dir is not None else Path(".")

        tcfg = dict(cfg.get("training", {}))
        self.save_hyperparameters({"training": tcfg})

        tf_cfg = dict(tcfg.get("teacher_forcing", {}) or {})
        self.tf_schedule = TeacherForcingSchedule(
            start=float(tf_cfg.get("start", tf_cfg.get("start_p", 1.0))),
            end=float(tf_cfg.get("end", tf_cfg.get("min_p", 0.0))),
            decay_epochs=int(tf_cfg.get("decay_epochs", 50)),
            decay=float(tf_cfg.get("decay", 0.98)),
            mode=str(tf_cfg.get("mode", "exponential")),
        )

        cur_cfg = dict(tcfg.get("curriculum", tcfg.get("rollout_curriculum", {})) or {})
        rollout_steps = int(tcfg.get("rollout_steps", 100))
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            start_k=int(cur_cfg.get("start_steps", cur_cfg.get("start_k", 1))),
            max_k=int(cur_cfg.get("max_k", cur_cfg.get("max", rollout_steps))),
            ramp_epochs=int(cur_cfg.get("ramp_epochs", 0)),
        )

        long_cfg = dict(tcfg.get("long_rollout", {}) or {})
        self.max_epochs = int(tcfg.get("max_epochs", 100))
        self.long_override = LateStageRolloutOverride(
            enabled=bool(long_cfg.get("enabled", False)),
            long_rollout_steps=int(long_cfg.get("long_rollout_steps", 0) or 0),
            final_epochs=int(
                long_cfg.get("long_ft_epochs", long_cfg.get("final_epochs", 0)) or 0
            ),
            apply_to_validation=bool(long_cfg.get("apply_to_validation", True)),
            apply_to_test=bool(long_cfg.get("apply_to_test", True)),
        )

        self.burn_in_steps = int(tcfg.get("burn_in_steps", 0))
        self.val_burn_in_steps = int(tcfg.get("val_burn_in_steps", self.burn_in_steps))
        self.burn_in_noise_std = float(tcfg.get("burn_in_noise_std", 0.0))
        self.burn_in_loss_weight = float(tcfg.get("burn_in_loss_weight", 0.0))

        lcfg = tcfg.get("loss", {})
        log_means, log_stds, log_mins, log_maxs = build_loss_buffers(
            normalization_manifest, species_variables, torch.device("cpu")
        )

        self.criterion = AdaptiveLoss(
            log_means,
            log_stds,
            log_mins,
            log_maxs,
            lambda_phys=float(lcfg.get("lambda_phys", 1.0)),
            lambda_z=float(lcfg.get("lambda_z", 0.1)),
        )

        self.lr = float(tcfg.get("lr", 1e-3))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))
        self.sched_cfg = dict(tcfg.get("scheduler", {}) or {})

        self._train_dl = self._val_dl = self._test_dl = None

    def set_dataloaders(self, train_dl, val_dl=None, test_dl=None) -> None:
        """
        Set dataloaders for training, validation, and testing.

        Args:
            train_dl: Training DataLoader
            val_dl: Validation DataLoader (optional)
            test_dl: Test DataLoader (optional)
        """
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._test_dl = test_dl

    def train_dataloader(self):
        """Return the training dataloader."""
        return self._train_dl

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self._val_dl

    def test_dataloader(self):
        """Return the test dataloader."""
        return self._test_dl

    def on_fit_start(self) -> None:
        """Move criterion buffers to the correct device after Lightning setup."""
        self.criterion.to(self.device)

    def _normalize_dt_for_unroll(self, dt: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """
        Normalize dt tensor to [B, K] for autoregressive unroll.

        Args:
            dt: Input dt tensor (various shapes supported)
            B: Batch size
            K: Sequence length

        Returns:
            Tensor of shape [B, K]
        """
        return normalize_dt_shape(dt, batch_size=B, seq_len=K, context="unroll")

    def _effective_k_roll(self, epoch: int, *, stage: str) -> int:
        """
        Compute effective rollout steps for the given stage and epoch.

        Args:
            epoch: Current training epoch
            stage: One of "train", "val", "test"

        Returns:
            Rollout length for this stage/epoch
        """
        k = self.curriculum.k_roll(epoch, self.max_epochs)

        if self.long_override.applies(epoch, self.max_epochs, stage):
            if self.long_override.long_rollout_steps > 0:
                k = int(self.long_override.long_rollout_steps)

        return int(k)

    def _teacher_forcing_prob(self, epoch: int) -> float:
        """Compute teacher forcing probability for the given epoch."""
        return float(self.tf_schedule.prob(epoch))

    def _get_burn_in_steps(self, stage: str) -> int:
        """Get burn-in steps for the given stage."""
        if stage == "train":
            return self.burn_in_steps
        return self.val_burn_in_steps

    def _autoregressive_unroll(
        self,
        y0: torch.Tensor,
        y_true: torch.Tensor,
        dt: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        tf_prob: float,
        k_roll: int,
        burn_in: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout with optional teacher forcing and burn-in.

        During burn-in steps, the model runs in open-loop mode (no teacher forcing)
        and optional noise is added for robustness. Loss weighting can be reduced
        for burn-in steps.

        Args:
            y0: Initial state in z-space [B, S]
            y_true: Ground truth trajectory in z-space [B, K, S]
            dt: Timestep tensor (various shapes supported)
            g: Global conditioning parameters [B, G] or None
            tf_prob: Teacher forcing probability (0-1)
            k_roll: Number of rollout steps
            burn_in: Number of burn-in steps (open-loop, reduced loss weight)

        Returns:
            Tuple of:
                - Predicted trajectory in z-space [B, k_roll, S]
                - Burn-in weight mask [k_roll] for loss weighting
        """
        B, S = y0.shape
        K = int(k_roll)

        if dt.ndim == 2 and dt.shape[1] >= K:
            dt_sliced = dt[:, :K]
        else:
            dt_sliced = dt

        dt_bk = self._normalize_dt_for_unroll(dt_sliced, B=B, K=K)

        if g is None:
            g = torch.zeros(B, 0, device=y0.device, dtype=y0.dtype)

        if burn_in >= K:
            warnings.warn(
                f"burn_in_steps ({burn_in}) >= rollout length ({K}). "
                f"Clamping burn_in to {K - 1} to leave at least one step for loss.",
                RuntimeWarning,
                stacklevel=3,
            )
            burn_in = K - 1

        preds: List[torch.Tensor] = []
        y_prev = y0

        burn_in_weights = torch.ones(K, device=y0.device, dtype=y0.dtype)
        if burn_in > 0:
            burn_in_weights[:burn_in] = self.burn_in_loss_weight

        for t in range(K):
            if t < burn_in and self.burn_in_noise_std > 0.0:
                noise = torch.randn_like(y_prev) * self.burn_in_noise_std
                y_input = y_prev + noise
            else:
                y_input = y_prev

            dt_t = dt_bk[:, t]
            y_next = self.model.forward_step(y_input, dt_t, g)

            preds.append(y_next.unsqueeze(1))

            if t >= burn_in and tf_prob > 0.0:
                use_truth = (torch.rand(B, device=y0.device) < tf_prob).view(B, 1)
                y_prev = torch.where(use_truth, y_true[:, t, :], y_next)
            else:
                y_prev = y_next

        return torch.cat(preds, dim=1), burn_in_weights

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Shared logic for train/val/test steps.

        Args:
            batch: Dictionary with keys 'y', 'dt', and optionally 'g'
            stage: One of "train", "val", "test"

        Returns:
            Total loss tensor for backpropagation
        """
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)

        B, K_full, S = y.shape
        epoch = int(self.current_epoch)

        K = int(max(1, K_full - 1))

        k_roll = int(min(self._effective_k_roll(epoch, stage=stage), K))
        k_roll = int(max(1, k_roll))

        burn_in = self._get_burn_in_steps(stage)

        tf_prob = float(self._teacher_forcing_prob(epoch) if stage == "train" else 0.0)

        y0 = y[:, 0, :]
        y_true = y[:, 1 : 1 + k_roll, :]

        y_pred, burn_in_weights = self._autoregressive_unroll(
            y0=y0,
            y_true=y_true,
            dt=dt,
            g=g,
            tf_prob=tf_prob,
            k_roll=k_roll,
            burn_in=burn_in, 
        )

        # Pass weights directly to criterion for consistent loss computation
        # This ensures that both the loss used for backpropagation AND the logged metrics
        # are computed with the same weighting scheme
        use_weights = burn_in > 0 and self.burn_in_loss_weight < 1.0
        weights = burn_in_weights if use_weights else None
        loss_dict = self.criterion(y_pred, y_true, weights=weights)

        loss_total = loss_dict["loss_total"]

        for k, v in loss_dict.items():
            if k == "loss_total":
                self.log(f"{stage}_loss", v, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"{stage}_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        if stage == "train":
            self.log(
                "train_tf_prob",
                torch.tensor(tf_prob, device=y.device),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train_k_roll",
                torch.tensor(k_roll, device=y.device),
                on_step=False,
                on_epoch=True,
            )
            if burn_in > 0:
                self.log(
                    "train_burn_in",
                    torch.tensor(burn_in, device=y.device),
                    on_step=False,
                    on_epoch=True,
                )

        return loss_total

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single training step."""
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single validation step."""
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single test step."""
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Uses AdamW optimizer with optional warmup and cosine decay schedule.

        Returns:
            Optimizer or dict with optimizer and lr_scheduler configuration
        """
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        sched_type = str(self.sched_cfg.get("name", self.sched_cfg.get("type", "none"))).lower()
        if sched_type in ("none", "off", "false", ""):
            return opt

        warmup_epochs = int(self.sched_cfg.get("warmup_epochs", 0))
        warmup_steps = int(self.sched_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(self.sched_cfg.get("min_lr_ratio", 0.0))

        max_steps = int(self.sched_cfg.get("max_steps", 0))
        if max_steps <= 0:
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except (AttributeError, RuntimeError):
                max_steps = 0

        if warmup_steps == 0 and warmup_epochs > 0 and max_steps > 0:
            steps_per_epoch = max_steps // max(1, self.max_epochs)
            warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if max_steps <= 0:
                return 1.0

            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ==============================================================================
# Trainer Factory
# ==============================================================================


def build_lightning_trainer(cfg: Dict[str, Any], work_dir: Path) -> pl.Trainer:
    """
    Build PyTorch Lightning Trainer from configuration.

    Creates a trainer with appropriate callbacks for checkpointing, early stopping,
    and CSV logging based on the configuration.

    Args:
        cfg: Configuration dictionary with 'training' section
        work_dir: Working directory for outputs and checkpoints

    Returns:
        Configured PyTorch Lightning Trainer

    Configuration Keys (in cfg["training"]):
        - max_epochs: Maximum training epochs
        - accelerator: Device accelerator ("auto", "gpu", "cpu")
        - devices: Device specification
        - precision: Training precision ("32-true", "16-mixed", "bf16-mixed")
        - grad_clip: Gradient clipping value (0 to disable)
        - checkpointing: Checkpoint configuration dict
        - early_stopping: Early stopping configuration dict
    """
    tcfg = dict(cfg.get("training", {}))
    max_epochs = int(tcfg.get("max_epochs", 1))
    accelerator = str(tcfg.get("accelerator", "auto"))
    devices = tcfg.get("devices", "auto")
    precision = tcfg.get("precision", "32-true")
    deterministic = bool(
        tcfg.get("deterministic", cfg.get("system", {}).get("deterministic", False))
    )

    callbacks: List[pl.Callback] = []

    callbacks.append(
        CSVLoggerCallback(
            work_dir=work_dir,
            filename=str(tcfg.get("metrics_csv", "metrics.csv"))
        )
    )

    ckpt_cfg = dict(tcfg.get("checkpointing", {}))
    ckpt_enabled = bool(ckpt_cfg.get("enabled", True))

    if ckpt_enabled:
        monitor = str(ckpt_cfg.get("monitor", "val_loss"))
        mode = str(ckpt_cfg.get("mode", "min"))
        save_top_k = int(ckpt_cfg.get("save_top_k", 1))
        save_last = bool(ckpt_cfg.get("save_last", True))
        every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1))

        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(work_dir),
                filename="best",
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
                save_last=save_last,
                auto_insert_metric_name=False,
                every_n_epochs=every_n_epochs,
            )
        )

    es_cfg = dict(tcfg.get("early_stopping", {}))
    es_enabled = bool(es_cfg.get("enabled", False))

    if es_enabled:
        es_monitor = str(es_cfg.get("monitor", "val_loss"))
        es_patience = int(es_cfg.get("patience", 20))
        es_mode = str(es_cfg.get("mode", "min"))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        es_verbose = bool(es_cfg.get("verbose", True))

        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=es_monitor,
                patience=es_patience,
                mode=es_mode,
                min_delta=es_min_delta,
                verbose=es_verbose,
            )
        )

    return pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        callbacks=callbacks,
        enable_checkpointing=ckpt_enabled,
        enable_progress_bar=bool(tcfg.get("enable_progress_bar", True)),
        enable_model_summary=bool(tcfg.get("enable_model_summary", True)),
        gradient_clip_val=float(tcfg.get("grad_clip", 0.0)) or None,
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        log_every_n_steps=1_000_000,
        logger=False,
        detect_anomaly=False,
        deterministic=deterministic,
    )
