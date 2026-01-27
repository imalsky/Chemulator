#!/usr/bin/env python3
"""
trainer.py - PyTorch Lightning training module and trainer factory.

Core behavior:

- Training loss (train_loss) is computed under the *training* rollout procedure:
    * autoregressive unroll with stochastic teacher forcing (per epoch schedule)
    * training burn-in settings
    * optional vectorized fast-path when teacher forcing probability is 1.0 and burn-in is 0
  This is the optimization objective and is the metric you'll usually see decrease fastest.

- Validation/test losses (val_loss/test_loss) are computed under *evaluation* rollouts:
    * open-loop autoregressive unroll (no teacher forcing)
    * validation burn-in settings

Because train_loss and val_loss are defined under different rollout procedures, they are
not directly comparable. A rank-zero warning is emitted at fit start.

Loss:
- Two-term loss: (lambda_log10_mae * mean(|Δlog10(y)|)) + (lambda_z_mse * mean((Δz)^2))
- Optional burn-in downweighting via per-step weights.

Logging (epoch-level only):
- train_loss (teacher-forced / training procedure; not directly comparable to val_loss)
- val_loss
- val_loss_log10_mae (component)
- val_loss_z_mse (component)
- lr
- epoch_time_sec
- train_tf_prob
- train_rollout_steps

This file is intentionally strict about configuration (not backward compatible with older
keys like train_mode/eval_mode/one_jump_k_roll).
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model import normalize_dt_shape

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl  # type: ignore

# Rank-zero warning helper (Lightning has moved this around across versions).
try:
    from lightning_utilities.core.rank_zero import rank_zero_warn  # type: ignore
except Exception:  # pragma: no cover
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_warn  # type: ignore
    except Exception:  # pragma: no cover
        def rank_zero_warn(msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore
            logging.getLogger(__name__).warning(msg)


# ==============================================================================
# Small utilities
# ==============================================================================


def _require_mapping(cfg: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    if key not in cfg:
        raise KeyError(f"Missing required config section: {context}.{key}")
    val = cfg[key]
    if not isinstance(val, Mapping):
        raise TypeError(f"Config section {context}.{key} must be a mapping/dict, got {type(val).__name__}")
    return val


def _require_key(cfg: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {context}.{key}")
    return cfg[key]


def _as_float(x: Any, *, context: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{context} must be a number, got {type(x).__name__}") from e


def _as_int(x: Any, *, context: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"{context} must be an int, got {type(x).__name__}") from e


# ==============================================================================
# CSV Logging
# ==============================================================================


_DEFAULT_CSV_FIELDS: Tuple[str, ...] = (
    "epoch_time_sec",
    "lr",
    "train_loss",
    "val_loss",
    "val_loss_log10_mae",
    "val_loss_z_mse",
    "train_tf_prob",
    "train_rollout_steps",
)


class CSVLoggerCallback(pl.Callback):
    """
    Minimal CSV logger that writes a fixed set of epoch-level metrics.

    During `fit`, the CSV row is written at the end of the training epoch (after validation
    in the standard fit loop) to avoid a lifecycle race where training-epoch metrics are
    finalized after validation. During `validate`/`test` runs, the row is written at the
    end of validation.

    Columns:
        epoch,<fields...>
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        filename: str = "metrics.csv",
        fields: Sequence[str] = _DEFAULT_CSV_FIELDS,
    ) -> None:
        super().__init__()
        self.path = Path(work_dir) / filename
        self.fieldnames: List[str] = ["epoch", *list(fields)]
        self._epoch_start_time: Optional[float] = None
        self._initialized = False
        self._in_fit = False  # True only during trainer.fit(...)

    def _init_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        self._initialized = True

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            if hasattr(v, "detach"):
                v = v.detach().cpu().item()
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _current_lr(trainer: pl.Trainer) -> Optional[float]:
        try:
            opt = trainer.optimizers[0]
        except Exception:
            return None
        try:
            return float(opt.param_groups[0]["lr"])
        except Exception:
            return None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self._epoch_start_time = time.perf_counter()

    def _write_epoch_row(self, trainer: pl.Trainer) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        if not self._initialized:
            self._init_file()

        metrics = dict(trainer.callback_metrics or {})
        row: Dict[str, Any] = {"epoch": int(trainer.current_epoch)}

        # Derived fields
        if self._epoch_start_time is not None:
            row["epoch_time_sec"] = float(time.perf_counter() - self._epoch_start_time)
        row["lr"] = self._current_lr(trainer)

        # Logged fields
        for k in self.fieldnames:
            if k == "epoch":
                continue
            if k in ("epoch_time_sec", "lr"):
                continue
            row[k] = self._to_float(metrics.get(k))

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Used to disambiguate `fit` vs `validate`/`test` when deciding when to write the CSV row.
        self._in_fit = True

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._in_fit = False

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # During `fit`, validation runs before training-epoch metrics are finalized into
        # `trainer.callback_metrics`. Defer writing until `on_train_epoch_end` to avoid
        # missing train_* columns (race with Lightning's metric finalization).
        if self._in_fit:
            return
        self._write_epoch_row(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # In `fit`, by the time this hook fires both training-epoch metrics (train_*) and
        # the most recent validation-epoch metrics (val_*) are available.
        self._write_epoch_row(trainer)


# ==============================================================================
# Loss
# ==============================================================================


_LOG_STD_CLAMP_MIN = 1e-10
_LOSS_DENOM_EPS = 1e-12


class AdaptiveLoss(nn.Module):
    """
    Two-term loss used for rollout training.

    Inputs/outputs are in z-space. Convert to log10 space via:
        log10(y) = z * log_std + log_mean
    """

    def __init__(
        self,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        *,
        lambda_log10_mae: float,
        lambda_z_mse: float,
    ) -> None:
        super().__init__()
        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=_LOG_STD_CLAMP_MIN))
        self.lambda_log10_mae = float(lambda_log10_mae)
        self.lambda_z_mse = float(lambda_z_mse)

    def z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.log_stds + self.log_means

    def forward(
        self,
        pred_z: torch.Tensor,
        true_z: torch.Tensor,
        *,
        step_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_z.shape != true_z.shape:
            raise ValueError(f"Shape mismatch: pred_z={tuple(pred_z.shape)} true_z={tuple(true_z.shape)}")
        if pred_z.ndim != 3:
            raise ValueError(f"Expected pred_z/true_z shape [B, K, S], got {tuple(pred_z.shape)}")

        B, K, S = pred_z.shape

        w_f32: Optional[torch.Tensor] = None
        denom: Optional[torch.Tensor] = None
        if step_weights is not None:
            if step_weights.ndim != 1 or int(step_weights.shape[0]) != int(K):
                raise ValueError(f"step_weights must have shape [K={K}], got {tuple(step_weights.shape)}")
            w_f32 = step_weights.to(device=pred_z.device, dtype=torch.float32).view(1, K, 1)
            denom = (w_f32.sum() * float(B * S)).clamp_min(_LOSS_DENOM_EPS)

        # z-space MSE
        diff_z = (pred_z - true_z).to(torch.float32)
        if w_f32 is None:
            z_mse = diff_z.square().mean()
        else:
            z_mse = diff_z.square().mul(w_f32).sum(dtype=torch.float32) / denom  # type: ignore[arg-type]

        # log10-space MAE
        pred_log10 = self.z_to_log10(pred_z)
        true_log10 = self.z_to_log10(true_z)
        abs_log10 = (pred_log10 - true_log10).abs().to(torch.float32)
        if w_f32 is None:
            log10_mae = abs_log10.mean()
        else:
            log10_mae = abs_log10.mul(w_f32).sum(dtype=torch.float32) / denom  # type: ignore[arg-type]

        loss_log10_mae = log10_mae * self.lambda_log10_mae
        loss_z_mse = z_mse * self.lambda_z_mse
        loss_total = loss_log10_mae + loss_z_mse

        return {
            "loss_total": loss_total,
            "loss_log10_mae": loss_log10_mae,
            "loss_z_mse": loss_z_mse,
        }


def build_loss_buffers(
    manifest: Mapping[str, Any],
    species: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (log_mean, log_std) buffers from the normalization manifest.

    Accepts either:
        per_key_stats[species]["log_mean"/"log_std"]
    or (legacy):
        per_key_stats[species]["log10_mean"/"log10_std"]

    Raises KeyError if required fields are missing.
    """
    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise KeyError("Normalization manifest missing required mapping: per_key_stats")

    means: List[float] = []
    stds: List[float] = []
    for s in species:
        if s not in stats:
            raise KeyError(f"Normalization stats missing for species key: {s}")
        entry = stats[s]
        if not isinstance(entry, Mapping):
            raise TypeError(f"per_key_stats[{s}] must be a mapping/dict, got {type(entry).__name__}")

        mu = entry.get("log_mean", entry.get("log10_mean"))
        sd = entry.get("log_std", entry.get("log10_std"))
        if mu is None or sd is None:
            raise KeyError(
                f"per_key_stats[{s}] must contain 'log_mean'/'log_std' (or legacy 'log10_mean'/'log10_std')."
            )

        means.append(float(mu))
        stds.append(float(sd))

    log_means = torch.tensor(means, device=device, dtype=torch.float32).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device, dtype=torch.float32).view(1, 1, -1)
    return log_means, log_stds


# ==============================================================================
# Schedules / curricula
# ==============================================================================


@dataclass(frozen=True)
class TeacherForcingSchedule:
    """
    Teacher forcing schedule.

    Supported modes:
      - exponential: p(epoch) = max(end, start * decay**epoch)
      - linear:      p(epoch) = max(end, start - (start-end) * epoch / decay_epochs)
      - cosine:      p(epoch) = end + (start-end) * 0.5 * (1 + cos(pi * epoch / decay_epochs))
    """

    start: float
    end: float
    mode: str = "exponential"
    decay_epochs: int = 50
    decay: float = 0.98

    def prob(self, epoch: int) -> float:
        e = max(0, int(epoch))
        mode = self.mode.lower().strip()

        if mode == "linear":
            if self.decay_epochs <= 0:
                p = self.end
            else:
                frac = min(1.0, float(e) / float(self.decay_epochs))
                p = self.start - (self.start - self.end) * frac
            return float(max(self.end, min(self.start, p)))

        if mode == "cosine":
            if self.decay_epochs <= 0:
                p = self.end
            else:
                frac = min(1.0, float(e) / float(self.decay_epochs))
                p = self.end + (self.start - self.end) * 0.5 * (1.0 + math.cos(math.pi * frac))
            return float(max(self.end, min(self.start, p)))

        # exponential (default)
        p = self.start * (self.decay**e)
        return float(max(self.end, min(self.start, p)))


@dataclass(frozen=True)
class RolloutCurriculum:
    """Optional rollout-length curriculum (linear ramp)."""

    enabled: bool
    start_k: int
    max_k: int
    ramp_epochs: int = 0

    def k_roll(self, epoch: int, max_epochs: int) -> int:
        if not self.enabled:
            return int(self.max_k)
        if self.ramp_epochs <= 0:
            return int(self.max_k)

        e = max(0, min(int(epoch), int(max_epochs)))
        frac = min(1.0, float(e) / float(max(1, self.ramp_epochs)))
        k = int(round(self.start_k + frac * (self.max_k - self.start_k)))
        return int(max(self.start_k, min(self.max_k, k)))


@dataclass(frozen=True)
class LateStageRolloutOverride:
    """Optional late-stage longer rollout override."""

    enabled: bool
    long_rollout_steps: int
    final_epochs: int
    apply_to_validation: bool = True
    apply_to_test: bool = True

    def applies(self, epoch: int, max_epochs: int, stage: str) -> bool:
        if not self.enabled or self.final_epochs <= 0:
            return False
        if stage == "val" and not self.apply_to_validation:
            return False
        if stage == "test" and not self.apply_to_test:
            return False
        return int(epoch) >= max(0, int(max_epochs) - int(self.final_epochs))


# ==============================================================================
# Lightning module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """
    Lightning module for flow-map rollout training.

    Batch format:
      batch["y"]:  [B, K_full, S] z-space states, where y[:, 0] is initial state.
      batch["dt"]: various shapes, normalized internally to [B, K] for K transitions.
      batch["g"]:  optional [B, G] conditioning vector.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        model: nn.Module,
        normalization_manifest: Mapping[str, Any],
        species_variables: Sequence[str],
        work_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.model = model
        self.normalization_manifest = dict(normalization_manifest)
        self.species_variables = list(species_variables)
        self.work_dir = Path(work_dir) if work_dir is not None else Path(".")

        tcfg = _require_mapping(self.cfg, "training", context="cfg")

        self.max_epochs = _as_int(_require_key(tcfg, "max_epochs", context="cfg.training"), context="cfg.training.max_epochs")
        self.rollout_steps = _as_int(_require_key(tcfg, "rollout_steps", context="cfg.training"), context="cfg.training.rollout_steps")
        self.lr = _as_float(_require_key(tcfg, "lr", context="cfg.training"), context="cfg.training.lr")
        self.weight_decay = _as_float(_require_key(tcfg, "weight_decay", context="cfg.training"), context="cfg.training.weight_decay")

        # Burn-in settings
        self.burn_in_steps = _as_int(tcfg.get("burn_in_steps", 0), context="cfg.training.burn_in_steps")
        self.val_burn_in_steps = _as_int(tcfg.get("val_burn_in_steps", self.burn_in_steps), context="cfg.training.val_burn_in_steps")
        self.burn_in_noise_std = _as_float(tcfg.get("burn_in_noise_std", 0.0), context="cfg.training.burn_in_noise_std")
        self.burn_in_loss_weight = _as_float(tcfg.get("burn_in_loss_weight", 0.0), context="cfg.training.burn_in_loss_weight")

        if self.burn_in_steps < 0 or self.val_burn_in_steps < 0:
            raise ValueError("burn_in_steps/val_burn_in_steps must be >= 0")
        if self.burn_in_noise_std < 0.0:
            raise ValueError("burn_in_noise_std must be >= 0")
        if not (0.0 <= self.burn_in_loss_weight <= 1.0):
            raise ValueError("burn_in_loss_weight must be in [0, 1]")

        # Teacher forcing schedule (required)
        tf_cfg = _require_mapping(tcfg, "teacher_forcing", context="cfg.training")
        tf_start = _as_float(_require_key(tf_cfg, "start", context="cfg.training.teacher_forcing"), context="cfg.training.teacher_forcing.start")
        tf_end = _as_float(_require_key(tf_cfg, "end", context="cfg.training.teacher_forcing"), context="cfg.training.teacher_forcing.end")
        tf_mode = str(tf_cfg.get("mode", "exponential"))
        self.tf_schedule = TeacherForcingSchedule(
            start=tf_start,
            end=tf_end,
            mode=tf_mode,
            decay_epochs=_as_int(tf_cfg.get("decay_epochs", 50), context="cfg.training.teacher_forcing.decay_epochs"),
            decay=_as_float(tf_cfg.get("decay", 0.98), context="cfg.training.teacher_forcing.decay"),
        )
        if not (0.0 <= self.tf_schedule.start <= 1.0 and 0.0 <= self.tf_schedule.end <= 1.0):
            raise ValueError("teacher_forcing.start/end must be within [0, 1]")
        if self.tf_schedule.end > self.tf_schedule.start:
            raise ValueError("teacher_forcing.end must be <= teacher_forcing.start")

        # Rollout curriculum (optional)
        cur_cfg = dict(tcfg.get("curriculum", {}) or {})
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            start_k=_as_int(cur_cfg.get("start_k", 1), context="cfg.training.curriculum.start_k"),
            max_k=_as_int(cur_cfg.get("max_k", self.rollout_steps), context="cfg.training.curriculum.max_k"),
            ramp_epochs=_as_int(cur_cfg.get("ramp_epochs", 0), context="cfg.training.curriculum.ramp_epochs"),
        )

        # Late-stage override (optional)
        long_cfg = dict(tcfg.get("long_rollout", {}) or {})
        self.long_override = LateStageRolloutOverride(
            enabled=bool(long_cfg.get("enabled", False)),
            long_rollout_steps=_as_int(long_cfg.get("long_rollout_steps", 0), context="cfg.training.long_rollout.long_rollout_steps"),
            final_epochs=_as_int(long_cfg.get("final_epochs", 0), context="cfg.training.long_rollout.final_epochs"),
            apply_to_validation=bool(long_cfg.get("apply_to_validation", True)),
            apply_to_test=bool(long_cfg.get("apply_to_test", True)),
        )

        # Loss (required explicit keys; no aliases/defaults)
        loss_cfg = _require_mapping(tcfg, "loss", context="cfg.training")
        lambda_log10_mae = _as_float(_require_key(loss_cfg, "lambda_log10_mae", context="cfg.training.loss"), context="cfg.training.loss.lambda_log10_mae")
        lambda_z_mse = _as_float(_require_key(loss_cfg, "lambda_z_mse", context="cfg.training.loss"), context="cfg.training.loss.lambda_z_mse")

        log_means, log_stds = build_loss_buffers(self.normalization_manifest, self.species_variables, torch.device("cpu"))
        self.criterion = AdaptiveLoss(
            log_means,
            log_stds,
            lambda_log10_mae=lambda_log10_mae,
            lambda_z_mse=lambda_z_mse,
        )

        # Scheduler (optional)
        self.sched_cfg = dict(tcfg.get("scheduler", {}) or {})

        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

        self._warned_metric_mismatch = False

        # Only persist the training section. This is intentionally strict/non-backward compatible.
        self.save_hyperparameters({"training": dict(tcfg)})

    def set_dataloaders(self, train_dl, val_dl=None, test_dl=None) -> None:
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._test_dl = test_dl

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def test_dataloader(self):
        return self._test_dl

    def on_fit_start(self) -> None:
        self.criterion.to(self.device)

        # Make the definitional mismatch explicit. This is deliberate (older strategy),
        # but it can look like a "bug" if you expect train/val to be directly comparable.
        if not self._warned_metric_mismatch:
            rank_zero_warn(
                "train_loss is computed under training rollouts (may include teacher forcing and training burn-in), "
                "while val_loss is computed open-loop (no teacher forcing) with validation burn-in. "
                "These metrics are not directly comparable."
            )
            self._warned_metric_mismatch = True

    def on_train_epoch_start(self) -> None:
        # Epoch-level training stats that are stable across batches.
        epoch = int(self.current_epoch)
        tf_prob = float(self.tf_schedule.prob(epoch))
        k_roll = int(self._effective_k_roll(epoch, stage="train"))

        self.log("train_tf_prob", torch.tensor(tf_prob, device=self.device), on_step=False, on_epoch=True)
        self.log("train_rollout_steps", torch.tensor(k_roll, device=self.device), on_step=False, on_epoch=True)

    def _effective_k_roll(self, epoch: int, *, stage: str) -> int:
        k = self.curriculum.k_roll(epoch, self.max_epochs)
        if self.long_override.applies(epoch, self.max_epochs, stage) and self.long_override.long_rollout_steps > 0:
            k = int(self.long_override.long_rollout_steps)
        return int(k)

    def _get_burn_in_steps(self, stage: str) -> int:
        return self.burn_in_steps if stage == "train" else self.val_burn_in_steps

    def _autoregressive_unroll(
        self,
        *,
        y0: torch.Tensor,          # [B, S]
        y_true: torch.Tensor,      # [B, K, S]
        dt_bk: torch.Tensor,
        g: Optional[torch.Tensor],
        tf_prob: float,
        burn_in: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressive rollout with stochastic teacher forcing.

        Returns:
          y_pred: [B, K, S]
          step_weights: Optional [K] (burn-in downweighting), or None
        """
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape [B, S], got {tuple(y0.shape)}")
        if y_true.ndim != 3:
            raise ValueError(f"y_true must have shape [B, K, S], got {tuple(y_true.shape)}")

        B, K, S = y_true.shape
        if int(y0.shape[0]) != int(B) or int(y0.shape[1]) != int(S):
            raise ValueError(f"y0 shape {tuple(y0.shape)} incompatible with y_true {tuple(y_true.shape)}")

        if K < 1:
            raise ValueError("Rollout length K must be >= 1")

        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if burn_in >= K:
            raise ValueError(f"burn_in ({burn_in}) must be < rollout length K ({K})")

        if dt_bk.shape != (B, K):
            raise ValueError(f"dt_bk must have shape [B={B}, K={K}], got {tuple(dt_bk.shape)}")

        if g is None:
            g = torch.zeros(B, 0, device=y0.device, dtype=y0.dtype)

        step_weights: Optional[torch.Tensor] = None
        if burn_in > 0 and self.burn_in_loss_weight < 1.0:
            step_weights = torch.ones(K, device=y0.device, dtype=y0.dtype)
            step_weights[:burn_in] = self.burn_in_loss_weight

        burn_noise: Optional[torch.Tensor] = None
        if burn_in > 0 and self.burn_in_noise_std > 0.0:
            burn_noise = torch.randn((B, burn_in, S), device=y0.device, dtype=y0.dtype) * self.burn_in_noise_std

        tf_mask: Optional[torch.Tensor] = None
        if tf_prob > 0.0:
            tf_mask = torch.rand((B, K), device=y0.device) < float(tf_prob)

        y_pred = y0.new_empty((B, K, S))
        y_prev = y0

        for t in range(K):
            y_input = y_prev
            if burn_noise is not None and t < burn_in:
                y_input = y_input + burn_noise[:, t, :]

            dt_t = dt_bk[:, t]
            y_next = self.model.forward_step(y_input, dt_t, g)
            y_pred[:, t, :] = y_next

            if tf_mask is not None and t >= burn_in:
                use_truth = tf_mask[:, t].view(B, 1)
                y_prev = torch.where(use_truth, y_true[:, t, :], y_next)
            else:
                y_prev = y_next

        return y_pred, step_weights

    def _vectorized_teacher_forced_rollout(
        self,
        *,
        y_in: torch.Tensor,       # [B, K, S] (ground-truth inputs y_t)
        dt_bk: torch.Tensor,
        g: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Vectorized teacher-forced rollout (bundled jumps).

        Equivalent to autoregressive rollout with teacher forcing probability 1.0
        and burn_in == 0, but computed in a single forward_step call by flattening
        (B, K) into the batch dimension.
        """
        if y_in.ndim != 3:
            raise ValueError(f"y_in must have shape [B, K, S], got {tuple(y_in.shape)}")

        B, K, S = y_in.shape
        if K < 1:
            raise ValueError("K must be >= 1")

        if dt_bk.shape != (B, K):
            raise ValueError(f"dt_bk must have shape [B={B}, K={K}], got {tuple(dt_bk.shape)}")

        if g is None:
            g = torch.zeros(B, 0, device=y_in.device, dtype=y_in.dtype)

        y_flat = y_in.reshape(B * K, S)
        dt_flat = dt_bk.reshape(B * K)

        G = int(g.shape[1]) if g.ndim == 2 else 0
        if G > 0:
            g_flat = g.unsqueeze(1).expand(B, K, G).reshape(B * K, G)
        else:
            g_flat = y_in.new_zeros((B * K, 0))

        y_next_flat = self.model.forward_step(y_flat, dt_flat, g_flat)
        if y_next_flat.shape != (B * K, S):
            raise RuntimeError(f"model.forward_step returned {tuple(y_next_flat.shape)}, expected {(B * K, S)}")

        return y_next_flat.view(B, K, S)

    def _shared_step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        if "y" not in batch or "dt" not in batch:
            raise KeyError("Batch must contain 'y' and 'dt' tensors.")

        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)

        if y.ndim != 3:
            raise ValueError(f"batch['y'] must have shape [B, K_full, S], got {tuple(y.shape)}")

        B, K_full, S = y.shape
        transitions = max(1, int(K_full) - 1)
        epoch = int(self.current_epoch)

        # Normalize dt to [B, transitions] and slice to the desired rollout length(s).
        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context=f"{stage}_unroll_full")

        if stage == "train":
            k_train = int(max(1, min(self._effective_k_roll(epoch, stage="train"), transitions)))
            burn_in_train = int(self._get_burn_in_steps("train"))
            if burn_in_train >= k_train:
                raise ValueError(f"train: burn_in ({burn_in_train}) must be < rollout steps ({k_train}).")

            dt_train = dt_full[:, :k_train]
            y_in_train = y[:, :k_train, :]           # y_t inputs (y0..y_{K-1})
            y_true_train = y[:, 1 : 1 + k_train, :]  # y_{t+1} targets

            tf_prob = float(self.tf_schedule.prob(epoch))

            # Fast path: exact teacher forcing, no burn-in.
            if tf_prob == 1.0 and burn_in_train == 0:
                y_pred_train = self._vectorized_teacher_forced_rollout(y_in=y_in_train, dt_bk=dt_train, g=g)
                step_weights_train = None
            else:
                y0 = y[:, 0, :]
                y_pred_train, step_weights_train = self._autoregressive_unroll(
                    y0=y0,
                    y_true=y_true_train,
                    dt_bk=dt_train,
                    g=g,
                    tf_prob=tf_prob,
                    burn_in=burn_in_train,
                )

            losses_train = self.criterion(y_pred_train, y_true_train, step_weights=step_weights_train)
            loss_opt = losses_train["loss_total"]

            # Log the optimization loss as train_loss (older strategy).
            self.log("train_loss", loss_opt.detach(), on_step=False, on_epoch=True, prog_bar=True)
            return loss_opt

        # Validation/test: open-loop autoregressive rollout (no teacher forcing).
        k_roll_sched = self._effective_k_roll(epoch, stage=stage)
        k_roll = int(max(1, min(int(k_roll_sched), transitions)))

        burn_in = int(self._get_burn_in_steps(stage))
        if burn_in >= k_roll:
            raise ValueError(f"{stage}: burn_in ({burn_in}) must be < rollout steps ({k_roll}).")

        dt_roll = dt_full[:, :k_roll]
        y_true = y[:, 1 : 1 + k_roll, :]
        y0 = y[:, 0, :]

        y_pred, step_weights = self._autoregressive_unroll(
            y0=y0,
            y_true=y_true,
            dt_bk=dt_roll,
            g=g,
            tf_prob=0.0,
            burn_in=burn_in,
        )

        losses = self.criterion(y_pred, y_true, step_weights=step_weights)
        loss_total = losses["loss_total"]

        if stage == "val":
            self.log("val_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("val_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        elif stage == "test":
            self.log("test_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)

        return loss_total

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sched_type = str(self.sched_cfg.get("type", "none")).lower().strip()
        if sched_type in ("none", "off", "false", ""):
            return opt

        if sched_type not in ("cosine_with_warmup", "warmup_cosine", "cosine"):
            raise ValueError(
                "scheduler.type must be one of: none, cosine_with_warmup (alias warmup_cosine), cosine"
            )

        warmup_epochs = _as_int(self.sched_cfg.get("warmup_epochs", 0), context="cfg.training.scheduler.warmup_epochs")
        warmup_steps = _as_int(self.sched_cfg.get("warmup_steps", 0), context="cfg.training.scheduler.warmup_steps")
        min_lr_ratio = _as_float(self.sched_cfg.get("min_lr_ratio", 0.0), context="cfg.training.scheduler.min_lr_ratio")

        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError("scheduler.min_lr_ratio must be in [0, 1]")

        max_steps = _as_int(self.sched_cfg.get("max_steps", 0), context="cfg.training.scheduler.max_steps")
        if max_steps <= 0:
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except Exception as e:
                raise ValueError(
                    "scheduler.max_steps must be set when trainer.estimated_stepping_batches is unavailable"
                ) from e

        if warmup_steps == 0 and warmup_epochs > 0:
            steps_per_epoch = max_steps // max(1, self.max_epochs)
            warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            step_i = int(step)
            if warmup_steps > 0 and step_i < warmup_steps:
                return float(step_i) / float(max(1, warmup_steps))

            progress = float(step_i - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# ==============================================================================
# Trainer factory
# ==============================================================================


def build_lightning_trainer(cfg: Mapping[str, Any], work_dir: Path) -> pl.Trainer:
    tcfg = _require_mapping(cfg, "training", context="cfg")

    max_epochs = _as_int(_require_key(tcfg, "max_epochs", context="cfg.training"), context="cfg.training.max_epochs")
    accelerator = str(tcfg.get("accelerator", "auto"))
    devices = tcfg.get("devices", "auto")
    precision = tcfg.get("precision", "32-true")

    deterministic = bool(tcfg.get("deterministic", cfg.get("system", {}).get("deterministic", False)))

    callbacks: List[pl.Callback] = [
        CSVLoggerCallback(
            work_dir=work_dir,
            filename=str(tcfg.get("metrics_csv", "metrics.csv")),
        )
    ]

    ckpt_cfg = dict(tcfg.get("checkpointing", {}) or {})
    ckpt_enabled = bool(ckpt_cfg.get("enabled", True))
    if ckpt_enabled:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(work_dir),
                filename="best",
                monitor=str(ckpt_cfg.get("monitor", "val_loss")),
                mode=str(ckpt_cfg.get("mode", "min")),
                save_top_k=_as_int(ckpt_cfg.get("save_top_k", 1), context="cfg.training.checkpointing.save_top_k"),
                save_last=bool(ckpt_cfg.get("save_last", True)),
                auto_insert_metric_name=False,
                every_n_epochs=_as_int(ckpt_cfg.get("every_n_epochs", 1), context="cfg.training.checkpointing.every_n_epochs"),
            )
        )

    es_cfg = dict(tcfg.get("early_stopping", {}) or {})
    if bool(es_cfg.get("enabled", False)):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=str(es_cfg.get("monitor", "val_loss")),
                patience=_as_int(es_cfg.get("patience", 20), context="cfg.training.early_stopping.patience"),
                mode=str(es_cfg.get("mode", "min")),
                min_delta=_as_float(es_cfg.get("min_delta", 0.0), context="cfg.training.early_stopping.min_delta"),
                verbose=bool(es_cfg.get("verbose", True)),
            )
        )

    grad_clip = tcfg.get("grad_clip", 0.0)
    grad_clip_val = None if float(grad_clip) == 0.0 else float(grad_clip)

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
        gradient_clip_val=grad_clip_val,
        accumulate_grad_batches=_as_int(tcfg.get("accumulate_grad_batches", 1), context="cfg.training.accumulate_grad_batches"),
        num_sanity_val_steps=_as_int(tcfg.get("num_sanity_val_steps", 0), context="cfg.training.num_sanity_val_steps"),
        log_every_n_steps=1_000_000,
        logger=False,
        detect_anomaly=False,
        deterministic=deterministic,
    )
