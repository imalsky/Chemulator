# trainer.py
#!/usr/bin/env python3
"""
trainer.py - Lightning training module and trainer factory.

Implements an autoregressive rollout training loop with:
- teacher forcing schedule
- rollout length curriculum
- adaptive physics-aware loss
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import lightning.pytorch as pl
except ImportError:  # fallback for older installs
    import pytorch_lightning as pl  # type: ignore


# ==============================================================================
# Logging Callbacks
# ==============================================================================


class CSVLoggerCallback(pl.Callback):
    """
    Minimal CSV logger writing to work_dir/metrics.csv.

    Only logs at epoch boundaries (train/val/test) via callback_metrics.
    """

    def __init__(self, work_dir: Path, filename: str = "metrics.csv") -> None:
        super().__init__()
        self.path = Path(work_dir) / filename
        self._fieldnames: Optional[List[str]] = None
        self._initialized = False

    def _init_writer(self, keys: List[str]) -> None:
        self._fieldnames = ["epoch"] + keys
        # Ensure parent dir exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._initialized = True

    def _append_row(self, epoch: int, metrics: Dict[str, Any]) -> None:
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
                # Convert tensors
                if hasattr(v, "detach"):
                    v = v.detach().cpu().item()
                row[k] = float(v)
            except Exception:
                # Skip non-numeric
                pass
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        metrics = dict(trainer.callback_metrics or {})
        # Filter keys we care about
        keys = sorted([k for k in metrics.keys() if isinstance(k, str)])
        if not self._initialized:
            self._init_writer(keys)
        self._append_row(trainer.current_epoch, metrics)


# ==============================================================================
# Loss
# ==============================================================================


class AdaptiveLoss(nn.Module):
    """
    Physics-informed loss combining:
    - per-species normalized MSE
    - global log10 error term
    - optional physics penalty on z-space bounds
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
        w_species_clamp: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.register_buffer("log_means", log_means)
        self.register_buffer("log_stds", log_stds)
        self.register_buffer("log_mins", log_mins)
        self.register_buffer("log_maxs", log_maxs)
        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.w_min, self.w_max = float(w_species_clamp[0]), float(w_species_clamp[1])

    def forward(self, pred_z: torch.Tensor, true_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_z: [B, K, S]
            true_z: [B, K, S]
        Returns:
            dict with loss_total, mse, phys_penalty, log10_err
        """
        # Basic MSE in z-space
        mse = (pred_z - true_z).pow(2).mean()

        # log10 error proxy (absolute z error)
        log10_err = (pred_z - true_z).abs().mean()

        # Physics penalty: keep within observed normalized bounds
        below = (self.log_mins - pred_z).clamp(min=0.0)
        above = (pred_z - self.log_maxs).clamp(min=0.0)
        phys_penalty = (below + above).pow(2).mean()

        loss_total = mse + self.lambda_z * log10_err + self.lambda_phys * phys_penalty
        return {
            "loss_total": loss_total,
            "mse": mse,
            "phys_penalty": phys_penalty,
            "log10_err": log10_err,
        }


def build_loss_buffers(
    manifest: Dict[str, Any],
    species: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract per-species normalization stats into device tensors.

    Supports manifests that store stats under:
      - manifest["per_key_stats"]   (current)
      - manifest["stats"]           (legacy)

    Each species key must have:
      - log_mean, log_std, log_min, log_max
    """
    stats_root = manifest.get("per_key_stats") or manifest.get("stats")
    if not isinstance(stats_root, dict):
        raise KeyError(
            "Normalization manifest missing 'per_key_stats' (or legacy 'stats'). "
            f"Top-level keys: {list(manifest.keys())}"
        )

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for sp in species:
        entry = stats_root.get(sp)
        if entry is None:
            # Helpful message: show a few available keys
            avail = list(stats_root.keys())
            sample = avail[:20]
            raise KeyError(
                f"Missing stats for species '{sp}' in normalization manifest. "
                f"Available keys (first {len(sample)}): {sample}"
            )

        try:
            means.append(float(entry["log_mean"]))
            stds.append(float(entry["log_std"]))
            mins.append(float(entry["log_min"]))
            maxs.append(float(entry["log_max"]))
        except KeyError as e:
            raise KeyError(
                f"Stats entry for '{sp}' missing required key {e}. "
                f"Entry keys: {list(entry.keys())}"
            ) from e

    log_means = torch.tensor(means, device=device, dtype=torch.float32).view(1, 1, -1)
    log_stds  = torch.tensor(stds,  device=device, dtype=torch.float32).view(1, 1, -1)
    log_mins  = torch.tensor(mins,  device=device, dtype=torch.float32).view(1, 1, -1)
    log_maxs  = torch.tensor(maxs,  device=device, dtype=torch.float32).view(1, 1, -1)
    return log_means, log_stds, log_mins, log_maxs



# ==============================================================================
# Schedules
# ==============================================================================


@dataclass(frozen=True)
class TeacherForcingSchedule:
    """Linear or cosine decay of teacher forcing probability."""

    start: float = 1.0
    end: float = 0.0
    decay_epochs: int = 0
    mode: str = "linear"

    def __call__(self, epoch: int) -> float:
        if self.decay_epochs <= 0:
            return self.start
        t = min(max(epoch, 0), self.decay_epochs) / self.decay_epochs
        if self.mode == "cosine":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.start + (self.end - self.start) * t


@dataclass(frozen=True)
class RolloutCurriculum:
    """Epoch-based rollout length curriculum."""

    enabled: bool = False
    start_steps: int = 1
    ramp_epochs: int = 0

    def __call__(self, epoch: int, max_steps: int) -> int:
        if not self.enabled or self.ramp_epochs <= 0:
            return max_steps
        t = min(max(epoch, 0), self.ramp_epochs) / self.ramp_epochs
        k = int(round(self.start_steps + (max_steps - self.start_steps) * t))
        return max(1, min(k, max_steps))


@dataclass(frozen=True)
class LateStageRolloutOverride:
    """Override rollout length for the final epochs of training."""

    enabled: bool = False
    long_rollout_steps: int = 0
    final_epochs: int = 0
    apply_to_validation: bool = True
    apply_to_test: bool = True

    def active(self, epoch: int, max_epochs: int) -> bool:
        if not self.enabled or self.final_epochs <= 0:
            return False
        start = max(0, max_epochs - self.final_epochs)
        return epoch >= start


# ==============================================================================
# Lightning Module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """Lightning module for flow-map rollout prediction."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        normalization_manifest: Dict[str, Any],
        species_variables: List[str],
        work_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.species = species_variables
        self.work_dir = Path(work_dir) if work_dir else None

        tcfg = cfg.get("training", {})
        self.rollout_steps = int(tcfg.get("rollout_steps", 96))
        self.burn_in_steps = int(tcfg.get("burn_in_steps", 0))
        self.val_burn_in_steps = int(tcfg.get("val_burn_in_steps", self.burn_in_steps))
        self.burn_in_noise_std = float(tcfg.get("burn_in_noise_std", 0.0))
        self.burn_in_loss_weight = float(tcfg.get("burn_in_loss_weight", 0.0))

        tf_cfg = tcfg.get("teacher_forcing", {})
        self.tf_schedule = TeacherForcingSchedule(
            start=float(tf_cfg.get("start", 1.0)),
            end=float(tf_cfg.get("end", 0.0)),
            decay_epochs=int(tf_cfg.get("decay_epochs", 0)),
            mode=str(tf_cfg.get("mode", "linear")).lower(),
        )

        cur_cfg = tcfg.get("curriculum", {})
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            start_steps=int(cur_cfg.get("start_steps", 1)),
            ramp_epochs=int(cur_cfg.get("ramp_epochs", 0)),
        )

        # Late-stage rollout override (optional)
        long_cfg = dict(tcfg.get("long_rollout", {}) or {})
        self.max_epochs = int(tcfg.get("max_epochs", 100))
        self.long_override = LateStageRolloutOverride(
            enabled=bool(long_cfg.get("enabled", False)),
            long_rollout_steps=int(long_cfg.get("long_rollout_steps", 0) or 0),
            final_epochs=int(long_cfg.get("long_ft_epochs", long_cfg.get("final_epochs", 0)) or 0),
            apply_to_validation=bool(long_cfg.get("apply_to_validation", True)),
            apply_to_test=bool(long_cfg.get("apply_to_test", True)),
        )

        # Loss (on CPU initially, moved to device in on_fit_start)
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
            w_species_clamp=tuple(lcfg.get("w_species_clamp", [0.5, 2.0])),
        )

        # Optimizer config
        self.lr = float(tcfg.get("lr", 1e-3))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))
        self.sched_cfg = dict(tcfg.get("scheduler", {}) or {})

        # Dataloaders (set externally)
        self._train_dl = self._val_dl = self._test_dl = None

    def set_dataloaders(self, train_dl, val_dl=None, test_dl=None) -> None:
        self._train_dl, self._val_dl, self._test_dl = train_dl, val_dl, test_dl

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def test_dataloader(self):
        return self._test_dl

    def on_fit_start(self) -> None:
        self.criterion = self.criterion.to(self.device)

    def _model_step(
        self, state: torch.Tensor, dt_step: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Single model step using efficient forward_step method."""
        return self.model.forward_step(state, dt_step, g)

    def _autoregressive_unroll(
        self,
        y0: torch.Tensor,
        dt_norm: torch.Tensor,
        y_true: torch.Tensor,
        g: torch.Tensor,
        *,
        tf_prob: float,
        burn_in: int,
        noise_std: float,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out autoregressively over a sequence.

        Args:
            y0: [B, S] initial state.
            dt_norm: [B] or scalar normalized dt.
            y_true: [B, K, S] ground-truth sequence.
            g: [B, G] globals.
            tf_prob: teacher forcing probability after burn-in.
            burn_in: number of initial steps to run open-loop (optionally noisy).
            noise_std: Gaussian noise std added during burn-in.
            max_steps: optional cap on number of steps after burn-in (for curriculum).

        Returns:
            pred_seq: [B, K_eff, S] predictions
            true_seq: [B, K_eff, S] aligned ground truth
        """
        if y0.ndim != 2 or y_true.ndim != 3:
            raise ValueError("Expected y0[B,S], y_true[B,K,S]")

        B = y0.shape[0]
        K_total = y_true.shape[1]
        if K_total < 1:
            raise ValueError("y_true must have at least 1 step")

        # Determine effective unroll length
        if max_steps is None:
            K_eff = K_total
        else:
            K_eff = min(K_total, burn_in + max_steps)

        # Constant dt per batch
        dt_step = dt_norm if dt_norm.ndim <= 1 else dt_norm[:, 0]

        state = y0
        preds: List[torch.Tensor] = []

        for t in range(K_eff):
            pred = self._model_step(state, dt_step, g)  # [B, S]
            preds.append(pred)

            if t == K_eff - 1:
                break

            # Burn-in: always use prediction (plus optional noise)
            if t < burn_in:
                if noise_std > 0:
                    pred = pred + noise_std * torch.randn_like(pred)
                state = pred
                continue

            # After burn-in: teacher forcing with probability tf_prob
            if tf_prob <= 0.0:
                state = pred
            elif tf_prob >= 1.0:
                state = y_true[:, t, :]
            else:
                use_tf = (torch.rand(B, device=state.device) < tf_prob).view(B, 1)
                state = torch.where(use_tf, y_true[:, t, :], pred)

        pred_seq = torch.stack(preds, dim=1)
        true_seq = y_true[:, :K_eff, :]
        return pred_seq, true_seq

    def _effective_k_roll(self, epoch: int, *, stage: str) -> int:
        """
        Compute effective rollout steps for the given stage.

        Stage behavior:
          - train: use curriculum, then (optionally) override for the final epochs
          - val:   same as train, gated by apply_to_validation
          - test:  if apply_to_test, prefer long_rollout_steps whenever configured
        """
        k = int(self.curriculum(epoch, self.rollout_steps))

        if stage not in ("train", "val", "test"):
            raise ValueError(f"Unknown stage: {stage}")

        if stage == "val":
            apply = bool(self.long_override.apply_to_validation)
        elif stage == "test":
            apply = bool(self.long_override.apply_to_test)
        else:
            apply = True

        long_k = int(self.long_override.long_rollout_steps)

        # Test is typically run after training; if configured, evaluate at long horizon.
        if stage == "test":
            if self.long_override.enabled and apply and long_k > 0:
                return max(1, long_k)
            return max(1, k)

        if (
            self.long_override.enabled
            and apply
            and long_k > 0
            and self.long_override.active(epoch, self.max_epochs)
        ):
            k = long_k

        return max(1, int(k))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        y0, dt_norm, y_seq, g = batch
        epoch = self.current_epoch

        k_roll = self._effective_k_roll(epoch, stage="train")
        tf_prob = self.tf_schedule(epoch)

        # Single forward pass with teacher forcing
        pred_train, true_train = self._autoregressive_unroll(
            y0,
            dt_norm,
            y_seq,
            g,
            tf_prob=tf_prob,
            burn_in=self.burn_in_steps,
            noise_std=self.burn_in_noise_std,
            max_steps=k_roll,
        )

        burn = min(self.burn_in_steps, pred_train.shape[1])

        if pred_train.shape[1] <= burn:
            raise ValueError(
                f"No samples after burn-in: shape[1]={pred_train.shape[1]}, burn={burn}"
            )

        metrics = self.criterion(pred_train[:, burn:, :], true_train[:, burn:, :])
        loss_main = metrics["loss_total"]
        loss_total = loss_main

        if burn > 0 and self.burn_in_loss_weight > 0:
            burn_metrics = self.criterion(pred_train[:, :burn, :], true_train[:, :burn, :])
            loss_total = loss_total + self.burn_in_loss_weight * burn_metrics["loss_total"]
            self.log("train_burn_loss", burn_metrics["loss_total"], on_epoch=True, sync_dist=True)

        # Log actual optimization loss and core metrics
        self.log("train_loss", loss_total, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_main", loss_main, on_epoch=True, sync_dist=True)
        self.log("train_log10_err", metrics["log10_err"], on_epoch=True, sync_dist=True)
        self.log("tf_prob", float(tf_prob), on_epoch=True)
        self.log("k_roll", float(k_roll), on_epoch=True)

        return loss_total

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        y0, dt_norm, y_seq, g = batch
        k_roll = self._effective_k_roll(self.current_epoch, stage="val")

        pred_all, true_all = self._autoregressive_unroll(
            y0,
            dt_norm,
            y_seq,
            g,
            tf_prob=0.0,
            burn_in=self.val_burn_in_steps,
            noise_std=0.0,
            max_steps=k_roll,
        )

        burn = min(self.val_burn_in_steps, pred_all.shape[1])
        if pred_all.shape[1] <= burn:
            raise ValueError(
                f"No samples after validation burn-in: shape[1]={pred_all.shape[1]}, burn={burn}"
            )

        metrics = self.criterion(pred_all[:, burn:, :], true_all[:, burn:, :])

        self.log(
            "val_loss", metrics["loss_total"], prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log("val_log10_err", metrics["log10_err"], on_epoch=True, sync_dist=True)

        return metrics["loss_total"]

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        y0, dt_norm, y_seq, g = batch

        pred_all, true_all = self._autoregressive_unroll(
            y0,
            dt_norm,
            y_seq,
            g,
            tf_prob=0.0,
            burn_in=self.val_burn_in_steps,
            noise_std=0.0,
            max_steps=self._effective_k_roll(self.current_epoch, stage="test"),
        )

        burn = min(self.val_burn_in_steps, pred_all.shape[1])
        if pred_all.shape[1] <= burn:
            raise ValueError(
                f"No samples after test burn-in: shape[1]={pred_all.shape[1]}, burn={burn}"
            )

        metrics = self.criterion(pred_all[:, burn:, :], true_all[:, burn:, :])

        self.log(
            "test_loss", metrics["loss_total"], prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log("test_log10_err", metrics["log10_err"], on_epoch=True, sync_dist=True)

        return metrics["loss_total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        sched_name = str(self.sched_cfg.get("name", "none")).lower().strip()
        if sched_name in ("none", "", "null"):
            return optimizer

        if sched_name not in ("cosine_warmup", "cosine"):
            raise ValueError(f"Unknown scheduler: {sched_name}")

        warmup_epochs = int(self.sched_cfg.get("warmup_epochs", 0))
        min_lr_ratio = float(self.sched_cfg.get("min_lr_ratio", 0.01))
        max_epochs = int(self.cfg.get("training", {}).get("max_epochs", 100))

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            if max_epochs <= warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


# ==============================================================================
# Trainer Factory
# ==============================================================================


def _get_precision(cfg_precision: str, accelerator: str) -> str:
    """Determine precision setting based on hardware capabilities."""
    precision = str(cfg_precision).lower().strip()

    if accelerator == "cpu" and precision in ("bf16-mixed", "16-mixed", "bf16", "16"):
        import warnings

        warnings.warn(
            f"Precision '{cfg_precision}' not supported on CPU, using '32-true'",
            stacklevel=3,
        )
        return "32-true"

    if accelerator == "mps" and precision in ("bf16-mixed", "bf16"):
        import warnings

        warnings.warn(
            f"Precision '{cfg_precision}' not supported on MPS, using '32-true'",
            stacklevel=3,
        )
        return "32-true"

    # Lightning uses '32-true', '16-mixed', 'bf16-mixed'
    if precision in ("32", "32-true", "fp32"):
        return "32-true"
    if precision in ("16", "16-mixed", "fp16"):
        return "16-mixed"
    if precision in ("bf16", "bf16-mixed"):
        return "bf16-mixed"

    return str(cfg_precision)


class PrintSummaryCallback(pl.Callback):
    """Print a concise summary at the end of fit."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        metrics = dict(trainer.callback_metrics or {})
        keys = ["train_loss", "val_loss", "train_log10_err", "val_log10_err"]
        msg = []
        for k in keys:
            if k in metrics:
                v = metrics[k]
                try:
                    v = float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
                    msg.append(f"{k}={v:.6g}")
                except Exception:
                    pass
        if msg:
            print("[summary] " + " | ".join(msg))


def build_lightning_trainer(cfg: Dict[str, Any], *, work_dir: Path) -> pl.Trainer:
    """Build a PyTorch Lightning trainer from configuration."""
    tcfg = cfg.get("training", {})

    max_epochs = int(tcfg.get("max_epochs", 100))
    devices = tcfg.get("devices", "auto")
    accelerator = str(tcfg.get("accelerator", "auto")).lower().strip()

    # Resolve accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    precision = _get_precision(str(tcfg.get("precision", "32-true")), accelerator)

    # Callbacks
    callbacks: List[pl.Callback] = []
    callbacks.append(CSVLoggerCallback(work_dir))
    callbacks.append(PrintSummaryCallback())

    # Checkpointing
    ckpt_cfg = tcfg.get("checkpointing", {})
    save_top_k = int(ckpt_cfg.get("save_top_k", 1))
    monitor = str(ckpt_cfg.get("monitor", "val_loss"))
    mode = str(ckpt_cfg.get("mode", "min"))
    every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1))

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=str(work_dir),
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            every_n_epochs=every_n_epochs,
            save_last=False,
            enable_version_counter=False,
        )
    )

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=str(work_dir),
            filename="last",
            save_last=True,
            save_top_k=0,
            enable_version_counter=False,
        )
    )

    # Early stopping
    es_cfg = tcfg.get("early_stopping", {})
    if bool(es_cfg.get("enabled", False)):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=str(es_cfg.get("monitor", monitor)),
                patience=int(es_cfg.get("patience", 10)),
                mode=str(es_cfg.get("mode", mode)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                verbose=bool(es_cfg.get("verbose", True)),
            )
        )

    # Determinism
    deterministic = bool(cfg.get("system", {}).get("deterministic", False))

    # Workaround for macOS/MPI shutdown oddities: reduce threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    return pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=bool(tcfg.get("enable_progress_bar", True)),
        enable_model_summary=bool(tcfg.get("enable_model_summary", True)),
        gradient_clip_val=float(tcfg.get("grad_clip", 0.0)) or None,
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        log_every_n_steps=max_epochs + 1,  # Disable step-level logging
        logger=False,
        detect_anomaly=False,
        deterministic=deterministic,
    )
