#!/usr/bin/env python3
"""
trainer.py - PyTorch Lightning training module and trainer factory.

Core behavior:

- Training loss (train_loss) is computed under the configured training rollout procedure:
    * If cfg.training.autoregressive_training.enabled is false (default):
        - autoregressive unroll with stochastic teacher forcing (per-epoch schedule)
        - training burn-in settings
        - optional vectorized fast-path when teacher forcing probability is 1.0
    * If cfg.training.autoregressive_training.enabled is true:
        - detached autoregressive rollout (pushforward + stop-grad between steps)
        - first `skip_steps` steps run under torch.no_grad() and are excluded from training
        - optional vectorized fast-path when teacher forcing probability is exactly 1.0 (loss masked for the excluded prefix)

- Validation/test losses (val_loss/test_loss) are computed under *evaluation* rollouts:
    * open-loop autoregressive unroll (no teacher forcing)
    * validation burn-in settings

Because train_loss and val_loss are defined under different rollout procedures, they are
not directly comparable. A rank-zero warning is emitted at fit start.

Loss:
- Two-term loss: (lambda_log10_mae * mean(|Δlog10(y)|)) + (lambda_z_mse * mean((Δz)^2))
- Optional burn-in downweighting via per-step weights.

Logging (epoch-level only):
- train_loss
- val_loss
- val_loss_log10_mae (component)
- val_loss_z_mse (component)
- epoch_time_sec

This file is intentionally strict about configuration (not backward compatible with older
keys like train_mode/eval_mode/one_jump_k_roll).

Normalization manifest schema (used for loss buffers):
- species_variables: list[str] of variable names
- stats mapping: one of per_key_stats / species_stats / stats; keyed by species name, each entry has log_mean/log_std
"""


from __future__ import annotations

import csv
import inspect
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


_LOSS_DENOM_EPS = 1e-3

# ==============================================================================
# Small utilities
# ==============================================================================


def _require_mapping(cfg: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    if key not in cfg:
        raise KeyError(f"Missing required mapping key '{key}' in {context}.")
    v = cfg[key]
    if not isinstance(v, Mapping):
        raise TypeError(f"Expected '{key}' in {context} to be a mapping/dict, got {type(v).__name__}.")
    return v


def _require_key(cfg: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required key '{key}' in {context}.")
    return cfg[key]


def _as_float(x: Any, *, context: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Expected float-like value for {context}, got {type(x).__name__}: {x}") from e


def _as_int(x: Any, *, context: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Expected int-like value for {context}, got {type(x).__name__}: {x}") from e


# ==============================================================================
# CSV epoch logger callback
# ==============================================================================


class CSVLoggerCallback(pl.Callback):
    """Write epoch-level metrics to work_dir/metrics.csv (rank-zero only).

    Strict schema:
      - The first epoch defines the CSV header.
      - If later epochs produce different metric keys, raise an error.
    """

    def __init__(self, *, work_dir: Path, filename: str = "metrics.csv") -> None:
        super().__init__()
        self.work_dir = Path(work_dir)
        self.filename = str(filename)
        self._header: Optional[List[str]] = None
        self._epoch_start_time: Optional[float] = None

    def _csv_path(self) -> Path:
        return self.work_dir / self.filename

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        metrics = dict(trainer.callback_metrics or {})
        row: Dict[str, Any] = {}

        row["epoch"] = int(getattr(trainer, "current_epoch", 0))
        if self._epoch_start_time is not None:
            row["epoch_time_sec"] = float(time.time() - self._epoch_start_time)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    continue
                row[k] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                row[k] = float(v)

        csv_path = self._csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Establish header on first write (sorted for determinism).
        if self._header is None:
            self._header = sorted(row.keys())
        else:
            if sorted(row.keys()) != self._header:
                raise RuntimeError(
                    f"CSVLoggerCallback schema mismatch. Expected header={self._header}, got keys={sorted(row.keys())}"
                )

        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._header})


# ==============================================================================
# Loss
# ==============================================================================


class AdaptiveLoss(nn.Module):
    """
    Two-term loss in latent z-space + log10-space. Uses per-species log10 normalization buffers.

    Data contract (repo-wide):
      - z := (log10(y_phys) - log10_mean) / log10_std
      - Therefore: log10(y_phys) = z * log10_std + log10_mean

    Components:
      - z_mse: mean((pred_z - true_z)^2)
      - log10_mae: mean(|log10(pred_y) - log10(true_y)|) computed via the denormalized log10 values

    Weighted total:
        loss_total = lambda_log10_mae * log10_mae + lambda_z_mse * z_mse
    """

    def __init__(
        self,
        log_means: torch.Tensor,  # [1,1,S]  (log10_mean)
        log_stds: torch.Tensor,   # [1,1,S]  (log10_std)
        *,
        lambda_log10_mae: float,
        lambda_z_mse: float,
    ) -> None:
        super().__init__()
        self.register_buffer("log_means", log_means.clone().detach())
        self.register_buffer("log_stds", log_stds.clone().detach())
        self.lambda_log10_mae = float(lambda_log10_mae)
        self.lambda_z_mse = float(lambda_z_mse)

    def _denormalize_log10(self, z: torch.Tensor) -> torch.Tensor:
        # z is normalized log10-space; output is log10(y_phys)
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

        # log10-space MAE (denormalize to log10(y_phys) and compare)
        pred_log10 = self._denormalize_log10(pred_z.to(torch.float32))
        true_log10 = self._denormalize_log10(true_z.to(torch.float32))
        diff_log10 = (pred_log10 - true_log10).abs()

        if w_f32 is None:
            log10_mae = diff_log10.mean()
        else:
            log10_mae = diff_log10.mul(w_f32).sum(dtype=torch.float32) / denom  # type: ignore[arg-type]

        loss_total = self.lambda_log10_mae * log10_mae + self.lambda_z_mse * z_mse
        return {"loss_total": loss_total, "loss_log10_mae": log10_mae, "loss_z_mse": z_mse}


# ==============================================================================
# Normalization buffers
# ==============================================================================


def build_loss_buffers(
    normalization_manifest: Mapping[str, Any],
    species_variables: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create [1,1,S] log10_mean and log10_std tensors in the order of `species_variables`.

    Accepted normalization manifest schema (as produced by preprocessing.py and validated in main.py):

        {
          "species_variables": ["sp1", "sp2", ...],
          "per_key_stats": {
            "sp1": {"log_mean": <float>, "log_std": <float>},
            "sp2": {"log_mean": <float>, "log_std": <float>}
          }
        }

    The stats mapping may alternatively appear under "species_stats" or "stats" (first one found is used).
    """
    if "species_variables" not in normalization_manifest:
        raise KeyError("Normalization manifest missing required key 'species_variables'.")
    man_species = normalization_manifest["species_variables"]
    if not isinstance(man_species, (list, tuple)):
        raise TypeError(
            f"Expected normalization_manifest['species_variables'] to be a list of names, got {type(man_species).__name__}."
        )
    for i, s in enumerate(man_species):
        if not isinstance(s, str):
            raise TypeError(
                f"Expected normalization_manifest['species_variables'][{i}] to be str, got {type(s).__name__}."
            )

    stats: Optional[Mapping[str, Any]] = None
    for key in ("per_key_stats", "species_stats", "stats"):
        v = normalization_manifest.get(key)
        if isinstance(v, Mapping):
            stats = v
            break
    if stats is None:
        raise KeyError(
            "Normalization manifest missing stats mapping under one of: per_key_stats, species_stats, stats."
        )

    means: List[float] = []
    stds: List[float] = []
    man_set = set(man_species)
    for name in species_variables:
        if name not in man_set:
            raise KeyError(
                f"Requested species variable '{name}' not listed in normalization_manifest['species_variables']."
            )
        entry = stats.get(name)
        if not isinstance(entry, Mapping):
            raise KeyError(f"Normalization stats missing entry for species variable: {name}")
        if "log_mean" not in entry or "log_std" not in entry:
            raise KeyError(f"Normalization stats entry for {name} missing log_mean/log_std keys.")
        means.append(_as_float(entry["log_mean"], context=f"normalization_manifest.stats[{name}].log_mean"))
        stds.append(_as_float(entry["log_std"], context=f"normalization_manifest.stats[{name}].log_std"))

    log_means = torch.tensor(means, device=device, dtype=torch.float32).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device, dtype=torch.float32).view(1, 1, -1)
    return log_means, log_stds


# ==============================================================================
# torch.compile helper
# ==============================================================================


def _try_torch_compile(fn: Any, *, cfg: Mapping[str, Any], context: str) -> Any:
    """Optionally wrap a callable with torch.compile (strict).

    If cfg.runtime.torch_compile.enabled is False, returns fn unchanged.
    If enabled is True, compilation must succeed (otherwise raises RuntimeError).
    """
    runtime = dict(cfg.get("runtime", {}) or {})
    tc = dict(runtime.get("torch_compile", {}) or {})

    enabled = tc.get("enabled", None)
    if enabled is None:
        raise KeyError("cfg.runtime.torch_compile.enabled is required (bool).")

    if not bool(enabled):
        return fn

    if not hasattr(torch, "compile"):
        raise RuntimeError(f"runtime.torch_compile.enabled=True but torch.compile is unavailable ({context}).")

    backend = str(tc.get("backend", "inductor"))
    mode = str(tc.get("mode", "reduce-overhead"))
    dynamic = bool(tc.get("dynamic", False))
    fullgraph = bool(tc.get("fullgraph", False))

    sig = inspect.signature(torch.compile)
    kwargs: Dict[str, Any] = {}
    if "backend" in sig.parameters:
        kwargs["backend"] = backend
    if "mode" in sig.parameters:
        kwargs["mode"] = mode
    if "dynamic" in sig.parameters:
        kwargs["dynamic"] = dynamic
    if "fullgraph" in sig.parameters:
        kwargs["fullgraph"] = fullgraph

    try:
        return torch.compile(fn, **kwargs)  # type: ignore[misc]
    except Exception as e:
        raise RuntimeError(f"torch.compile failed for {context}: {e}") from e


# ==============================================================================
# Rollout schedules
# ==============================================================================


@dataclass
class TeacherForcingSchedule:
    """
    Per-epoch schedule for teacher forcing probability.

    Supported modes:
      - constant: p(epoch) = p0
      - linear:   linearly interpolate from p0 to p1 over `ramp_epochs`
      - cosine:   cosine ramp from p0 to p1 over `ramp_epochs`
    """

    mode: str
    p0: float
    p1: float
    ramp_epochs: int

    def prob(self, epoch: int) -> float:
        e = int(max(0, epoch))
        mode = str(self.mode).lower()
        if mode == "constant" or self.ramp_epochs <= 0:
            return float(self.p0)

        t = min(1.0, float(e) / float(max(1, int(self.ramp_epochs))))
        if mode == "linear":
            return float(self.p0 + (self.p1 - self.p0) * t)
        if mode == "cosine":
            # t in [0,1]
            c = 0.5 * (1.0 - math.cos(math.pi * t))
            return float(self.p0 + (self.p1 - self.p0) * c)

        raise ValueError(f"Unknown teacher forcing schedule mode: {self.mode}")


@dataclass
class RolloutCurriculum:
    """
    Optional curriculum that ramps rollout steps over epochs.

    steps(epoch) = min(max_k, base_k + floor((max_k - base_k) * f(epoch)))

    Where f(epoch) is either linear or cosine in [0,1] over `ramp_epochs`.
    """

    enabled: bool
    mode: str
    base_k: int
    max_k: int
    ramp_epochs: int

    def steps(self, epoch: int) -> int:
        if not self.enabled:
            return int(self.max_k)

        base = int(self.base_k)
        max_k = int(self.max_k)
        if self.ramp_epochs <= 0:
            return max(1, min(max_k, base))

        e = int(max(0, epoch))
        t = min(1.0, float(e) / float(max(1, int(self.ramp_epochs))))

        mode = str(self.mode).lower()
        if mode == "linear":
            f = t
        elif mode == "cosine":
            f = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown rollout curriculum mode: {self.mode}")

        steps = base + int((max_k - base) * f)
        return max(1, min(max_k, steps))


@dataclass
class LateStageRolloutOverride:
    """
    Optional override to use longer rollouts in the last `final_epochs` of training.
    """

    enabled: bool
    long_rollout_steps: int
    final_epochs: int
    apply_to_validation: bool
    apply_to_test: bool

    def maybe_override(self, epoch: int, *, max_epochs: int, stage: str, current_k: int) -> int:
        if not self.enabled:
            return int(current_k)
        if self.final_epochs <= 0:
            return int(current_k)

        # last N epochs are [max_epochs-final_epochs, max_epochs-1]
        start = int(max_epochs) - int(self.final_epochs)
        if epoch >= start:
            if stage == "train":
                return int(max(current_k, self.long_rollout_steps))
            if stage == "val" and self.apply_to_validation:
                return int(max(current_k, self.long_rollout_steps))
            if stage == "test" and self.apply_to_test:
                return int(max(current_k, self.long_rollout_steps))

        return int(current_k)


# ==============================================================================
# Lightning module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """
    LightningModule wrapper around the underlying model (must implement forward_step).

    Batch schema expected:
      - y:  [B, K_full, S]  normalized log10-space state (z)
      - dt: [B, K_full-1] or [K_full-1] or scalar  (will be normalized to [B, transitions])
      - g:  optional globals [B, G] or [G] or empty (coerced/broadcast)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg: Mapping[str, Any],
        normalization_manifest: Mapping[str, Any],
        species_variables: Sequence[str],
    ) -> None:
        super().__init__()
        self.model = model

        # Optional torch.compile (strict). If enabled, keys must be explicit and compilation must succeed.
        runtime = dict(cfg.get("runtime", {}) or {})
        tc = dict(runtime.get("torch_compile", {}) or {})
        enabled = tc.get("enabled", None)
        if enabled is None:
            raise KeyError("cfg.runtime.torch_compile.enabled is required (bool).")

        self._compiled_forward_step: Optional[Any] = None
        self._compiled_open_loop_unroll: Optional[Any] = None

        if bool(enabled):
            compile_forward_step = bool(_require_key(tc, "compile_forward_step", context="runtime.torch_compile"))
            compile_open_loop = bool(_require_key(tc, "compile_open_loop_unroll", context="runtime.torch_compile"))

            if compile_forward_step:
                if not hasattr(self.model, "forward_step"):
                    raise RuntimeError("compile_forward_step=True but model has no forward_step method.")
                self._compiled_forward_step = _try_torch_compile(
                    self.model.forward_step,  # type: ignore[attr-defined]
                    cfg=cfg,
                    context="FlowMapRolloutModule.model.forward_step",
                )

            if compile_open_loop:
                self._compiled_open_loop_unroll = _try_torch_compile(
                    self._open_loop_unroll,
                    cfg=cfg,
                    context="FlowMapRolloutModule._open_loop_unroll",
                )

        self.normalization_manifest = dict(normalization_manifest)
        self.species_variables = list(species_variables)

        tcfg = _require_mapping(cfg, "training", context="cfg")
        mcfg = _require_mapping(cfg, "model", context="cfg")

        self.max_epochs = _as_int(_require_key(tcfg, "max_epochs", context="cfg.training"), context="cfg.training.max_epochs")
        # Rollout steps (single horizon for train/val/test; inference is autoregressive by feeding predictions back in).
        self.rollout_steps = _as_int(_require_key(tcfg, "rollout_steps", context="cfg.training"), context="cfg.training.rollout_steps")

        # Burn-in (required explicit keys)
        burn_cfg = _require_mapping(tcfg, "burn_in", context="cfg.training")
        self.burn_in_train = _as_int(_require_key(burn_cfg, "train", context="cfg.training.burn_in"), context="cfg.training.burn_in.train")
        self.burn_in_val = _as_int(_require_key(burn_cfg, "val", context="cfg.training.burn_in"), context="cfg.training.burn_in.val")
        self.burn_in_test = _as_int(_require_key(burn_cfg, "test", context="cfg.training.burn_in"), context="cfg.training.burn_in.test")

        # Teacher forcing schedule (required explicit keys)
        tf_cfg = _require_mapping(tcfg, "teacher_forcing", context="cfg.training")
        self.tf_schedule = TeacherForcingSchedule(
            mode=str(_require_key(tf_cfg, "mode", context="cfg.training.teacher_forcing")),
            p0=_as_float(_require_key(tf_cfg, "p0", context="cfg.training.teacher_forcing"), context="cfg.training.teacher_forcing.p0"),
            p1=_as_float(_require_key(tf_cfg, "p1", context="cfg.training.teacher_forcing"), context="cfg.training.teacher_forcing.p1"),
            ramp_epochs=_as_int(_require_key(tf_cfg, "ramp_epochs", context="cfg.training.teacher_forcing"), context="cfg.training.teacher_forcing.ramp_epochs"),
        )

        # Autoregressive training (pushforward + stop-grad) - optional.
        # When enabled, training uses a detached autoregressive rollout:
        #   - first `skip_steps` are rolled out under torch.no_grad() (no training)
        #   - remaining steps train per-step, detaching the state after every step
        ar_cfg = dict(tcfg.get("autoregressive_training", {}) or {})
        self.autoregressive_training = bool(ar_cfg.get("enabled", False))
        self.ar_no_grad_steps = _as_int(ar_cfg.get("no_grad_steps", 0), context="cfg.training.autoregressive_training.no_grad_steps")
        self.ar_backward_per_step = bool(ar_cfg.get("backward_per_step", True))
        self.ar_teacher_forcing_in_trained_steps = bool(ar_cfg.get("teacher_forcing_in_trained_steps", True))

        if self.autoregressive_training:
            # Required for per-step backward/step control.
            self.automatic_optimization = False

        # Optional curriculum
        cur_cfg = dict(tcfg.get("curriculum", {}) or {})
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            mode=str(cur_cfg.get("mode", "linear")),
            base_k=_as_int(cur_cfg.get("base_k", self.rollout_steps), context="cfg.training.curriculum.base_k"),
            max_k=_as_int(cur_cfg.get("max_k", self.rollout_steps), context="cfg.training.curriculum.max_k"),
            ramp_epochs=_as_int(cur_cfg.get("ramp_epochs", 0), context="cfg.training.curriculum.ramp_epochs"),
        )

        # Late-stage override (optional)
        long_cfg = dict(tcfg.get("long_rollout", {}) or {})
        long_enabled = bool(long_cfg.get("enabled", False))

        if long_enabled:
            apply_val = bool(_require_key(long_cfg, "apply_to_validation", context="cfg.training.long_rollout"))
            apply_test = bool(_require_key(long_cfg, "apply_to_test", context="cfg.training.long_rollout"))
        else:
            apply_val = False
            apply_test = False

        self.long_override = LateStageRolloutOverride(
            enabled=long_enabled,
            long_rollout_steps=_as_int(long_cfg.get("long_rollout_steps", 0), context="cfg.training.long_rollout.long_rollout_steps"),
            final_epochs=_as_int(long_cfg.get("final_epochs", 0), context="cfg.training.long_rollout.final_epochs"),
            apply_to_validation=apply_val,
            apply_to_test=apply_test,
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

        # dataloaders (wired from main)
        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

        self._warned_metric_mismatch = False

        # Only persist the training section. This is intentionally strict/non-backward compatible.
        self.save_hyperparameters({"training": dict(tcfg), "model": dict(mcfg)})

    # ------------------------
    # Dataloader wiring (optional)
    # ------------------------

    def set_dataloaders(self, train_dl: Any, val_dl: Any = None, test_dl: Any = None) -> None:
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._test_dl = test_dl

    def train_dataloader(self) -> Any:
        if self._train_dl is None:
            raise RuntimeError("train_dataloader requested but not set. Provide dataloaders via main or set_dataloaders().")
        return self._train_dl

    def val_dataloader(self) -> Any:
        return self._val_dl

    def test_dataloader(self) -> Any:
        return self._test_dl

    # ------------------------
    # Rollout utilities
    # ------------------------

    def _forward_step(self, y_t: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # y_t: [B, S], dt: [B], g: [B, G]
        if not hasattr(self.model, "forward_step"):
            raise AttributeError("Model must implement forward_step(y_t, dt, g)->y_{t+1}")

        fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        y_next = fn(y_t, dt, g)  # type: ignore[misc]
        return y_next

    def _coerce_g(self, g: Optional[torch.Tensor], batch_size: int, like: torch.Tensor, *, context: str) -> torch.Tensor:
        # Coerce globals to [B, G] (or [B,0] if absent)
        if g is None:
            return like.new_zeros((batch_size, 0))

        if not isinstance(g, torch.Tensor):
            raise TypeError(f"{context}: g must be a torch.Tensor or None, got {type(g).__name__}")

        if g.numel() == 0:
            return like.new_zeros((batch_size, 0))

        if g.ndim == 1:
            g = g.view(1, -1).expand(batch_size, -1)
        elif g.ndim == 2:
            if int(g.shape[0]) == 1 and batch_size > 1:
                g = g.expand(batch_size, -1)
            elif int(g.shape[0]) != batch_size:
                raise ValueError(f"{context}: g has batch {int(g.shape[0])}, expected {batch_size}")
        else:
            raise ValueError(f"{context}: g must have shape [G] or [B,G], got {tuple(g.shape)}")

        return g.to(device=like.device, dtype=like.dtype)

    def _effective_k_roll(self, epoch: int, *, stage: str) -> int:
        # Determine rollout steps under curriculum and long-rollout override.
        # Curriculum is a *training* schedule; evaluation uses the configured base horizon (+ optional long-rollout override).
        if stage == "train" and self.curriculum.enabled:
            base = int(self.curriculum.steps(epoch))
        else:
            base = int(self.rollout_steps)

        base = int(max(1, base))
        base = int(self.long_override.maybe_override(epoch, max_epochs=self.max_epochs, stage=stage, current_k=base))
        return int(max(1, base))

    def _get_burn_in_steps(self, stage: str) -> int:
        if stage == "train":
            return int(self.burn_in_train)
        if stage == "val":
            return int(self.burn_in_val)
        if stage == "test":
            return int(self.burn_in_test)
        raise ValueError(f"Unknown stage for burn-in: {stage}")

    def _autoregressive_unroll(
        self,
        *,
        y0: torch.Tensor,        # [B, S]
        y_true: torch.Tensor,    # [B, K, S] targets for steps 1..K
        dt_bk: torch.Tensor,     # [B, K]
        g: Optional[torch.Tensor],
        tf_prob: float,
        burn_in: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Autoregressive unroll for K steps (optionally stochastic teacher forcing)."""
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape [B,S], got {tuple(y0.shape)}")
        if y_true.ndim != 3:
            raise ValueError(f"y_true must have shape [B,K,S], got {tuple(y_true.shape)}")
        if dt_bk.ndim != 2:
            raise ValueError(f"dt_bk must have shape [B,K], got {tuple(dt_bk.shape)}")

        B, K, S = y_true.shape
        if y0.shape != (B, S):
            raise ValueError(f"y0 shape mismatch: got {tuple(y0.shape)}, expected {(B, S)}")
        if dt_bk.shape != (B, K):
            raise ValueError(f"dt_bk shape mismatch: got {tuple(dt_bk.shape)}, expected {(B, K)}")

        g_t = self._coerce_g(g, B, y_true, context="_autoregressive_unroll")

        y_pred = torch.empty((B, K, S), device=y_true.device, dtype=y_true.dtype)
        y_prev = y0

        burn_in = int(max(0, min(int(burn_in), K)))
        tf = float(tf_prob)

        mask_tf: Optional[torch.Tensor] = None
        if 0.0 < tf < 1.0 and burn_in < K:
            mask_tf = (torch.rand((B, K - burn_in), device=y_true.device) < tf)

        for k in range(K):
            if k < burn_in:
                y_pred[:, k, :] = y_true[:, k, :]
                y_prev = y_true[:, k, :]
                continue

            dt_k = dt_bk[:, k]
            y_next = self._forward_step(y_prev, dt_k, g_t)
            y_pred[:, k, :] = y_next

            if tf <= 0.0:
                y_prev = y_next
            elif tf >= 1.0:
                y_prev = y_true[:, k, :]
            else:
                assert mask_tf is not None
                use_tf = mask_tf[:, k - burn_in].view(B, 1)
                y_prev = torch.where(use_tf, y_true[:, k, :], y_next)

        if burn_in > 0:
            w = torch.ones((K,), device=y_true.device, dtype=torch.float32)
            w[:burn_in] = 0.0
            return y_pred, w

        return y_pred, None

    def _open_loop_unroll(
        self,
        *,
        y0: torch.Tensor,        # [B, S]
        y_true: torch.Tensor,    # [B, K, S] targets for steps 1..K (used for burn-in inputs)
        dt_bk: torch.Tensor,     # [B, K]
        g: Optional[torch.Tensor],
        burn_in: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Deterministic rollout for eval (tf_prob=0), with optional burn-in."""
        if y0.ndim != 2:
            raise ValueError(f"y0 must have shape [B,S], got {tuple(y0.shape)}")
        if y_true.ndim != 3:
            raise ValueError(f"y_true must have shape [B,K,S], got {tuple(y_true.shape)}")
        if dt_bk.ndim != 2:
            raise ValueError(f"dt_bk must have shape [B,K], got {tuple(dt_bk.shape)}")

        B, K, S = y_true.shape
        if y0.shape != (B, S):
            raise ValueError(f"y0 shape mismatch: got {tuple(y0.shape)}, expected {(B, S)}")
        if dt_bk.shape != (B, K):
            raise ValueError(f"dt_bk shape mismatch: got {tuple(dt_bk.shape)}, expected {(B, K)}")

        g_t = self._coerce_g(g, B, y_true, context="_open_loop_unroll")

        y_pred = torch.empty((B, K, S), device=y_true.device, dtype=y_true.dtype)
        y_prev = y0

        burn_in = int(max(0, min(int(burn_in), K)))

        for k in range(K):
            if k < burn_in:
                y_pred[:, k, :] = y_true[:, k, :]
                y_prev = y_true[:, k, :]
                continue

            dt_k = dt_bk[:, k]
            y_next = self._forward_step(y_prev, dt_k, g_t)
            y_pred[:, k, :] = y_next
            y_prev = y_next

        if burn_in > 0:
            w = torch.ones((K,), device=y_true.device, dtype=torch.float32)
            w[:burn_in] = 0.0
            return y_pred, w

        return y_pred, None

    def _vectorized_teacher_forced_rollout(
        self,
        *,
        y_in: torch.Tensor,      # [B, K, S] inputs y_t
        dt_bk: torch.Tensor,     # [B, K]
        g: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fast path for pure teacher-forcing:
        y_pred[t+1] = model.forward_step(y_t, dt_t, g)
        computed in a single batched call by flattening B*K.
        """
        if y_in.ndim != 3:
            raise ValueError(f"y_in must have shape [B, K, S], got {tuple(y_in.shape)}")

        B, K, S = y_in.shape
        if K < 1:
            raise ValueError("K must be >= 1")

        if dt_bk.shape != (B, K):
            raise ValueError(f"dt_bk must have shape [B={B}, K={K}], got {tuple(dt_bk.shape)}")

        g_t = self._coerce_g(g, B, y_in, context="_teacher_forced_vectorized")

        y_flat = y_in.reshape(B * K, S)
        dt_flat = dt_bk.reshape(B * K)

        G = int(g_t.shape[1]) if g_t.ndim == 2 else 0
        if G > 0:
            g_flat = g_t.unsqueeze(1).expand(B, K, G).reshape(B * K, G)
        else:
            g_flat = y_in.new_zeros((B * K, 0))

        y_next_flat = self._forward_step(y_flat, dt_flat, g_flat)
        if y_next_flat.shape != (B * K, S):
            raise RuntimeError(f"model.forward_step returned {tuple(y_next_flat.shape)}, expected {(B * K, S)}")

        return y_next_flat.view(B, K, S)

    # ------------------------
    # Train/val/test steps
    # ------------------------

    def _training_step_autoregressive_detached(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Detached autoregressive training regime (pushforward + stop-grad).

        Contract:
          - Roll out for N steps (N = effective train rollout horizon for this epoch).
          - The first `skip_steps` steps are rolled out under torch.no_grad() and do not contribute gradients.
            skip_steps := max(cfg.training.burn_in.train, cfg.training.autoregressive_training.no_grad_steps)
          - For the remaining steps, train on each 1-step prediction, but detach the state after every step so
            gradients never backpropagate through time.

        Special-case fast path:
          - If teacher forcing probability is exactly 1.0, use the vectorized teacher-forced rollout and
            simply zero-weight the first `skip_steps` steps.
        """
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

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/_training_step_autoregressive_detached")

        k_train = int(max(1, min(self._effective_k_roll(epoch, stage="train"), transitions)))
        dt_train = dt_full[:, :k_train]                 # [B, k_train]
        y_true = y[:, 1: 1 + k_train, :]                # [B, k_train, S]
        y0 = y[:, 0, :]                                 # [B, S]

        tf_prob = float(self.tf_schedule.prob(epoch))

        # Steps to exclude from gradient updates.
        skip_steps = int(max(0, max(int(self._get_burn_in_steps("train")), int(getattr(self, "ar_no_grad_steps", 0)))))
        if skip_steps >= k_train:
            raise ValueError(f"train: skip_steps ({skip_steps}) must be < rollout steps ({k_train}).")

        g_t = self._coerce_g(g, B, y0, context="_training_step_autoregressive_detached")

        # Manual optimization (required when autoregressive_training=True).
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            if len(opt) != 1:
                raise RuntimeError("Expected a single optimizer for manual optimization.")
            opt = opt[0]

        acc = getattr(self.trainer, "accumulate_grad_batches", 1)
        if isinstance(acc, Mapping):
            # Rare Lightning case: per-epoch dict. Best-effort: pick epoch 0 / first value.
            acc = int(acc.get(0, list(acc.values())[0]))  # type: ignore[index]
        acc = int(max(1, int(acc)))

        if (batch_idx % acc) == 0:
            opt.zero_grad(set_to_none=True)

        # Fast path: pure teacher forcing (vectorized) + masked loss for the excluded prefix.
        if tf_prob == 1.0:
            y_in = y[:, :k_train, :]  # y_t inputs
            y_pred = self._vectorized_teacher_forced_rollout(y_in=y_in, dt_bk=dt_train, g=g)

            step_weights = None
            if skip_steps > 0:
                w = torch.ones((k_train,), device=y_true.device, dtype=torch.float32)
                w[:skip_steps] = 0.0
                step_weights = w

            losses = self.criterion(y_pred, y_true, step_weights=step_weights)
            loss_total = losses["loss_total"]

            self.manual_backward(loss_total)

            # Optimizer step respecting gradient accumulation.
            is_last = (batch_idx + 1) >= int(getattr(self.trainer, "num_training_batches", batch_idx + 1))
            should_step = (((batch_idx + 1) % acc) == 0) or is_last
            if should_step:
                clip_val = getattr(self.trainer, "gradient_clip_val", None)
                if clip_val is not None and float(clip_val) > 0:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=float(clip_val),
                        gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
                    )
                opt.step()
                opt.zero_grad(set_to_none=True)

                # Step schedulers (this codebase uses interval="step").
                sch = self.lr_schedulers()
                if sch is not None:
                    if isinstance(sch, (list, tuple)):
                        for s in sch:
                            s.step()
                    else:
                        sch.step()

            self.log("train_loss", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
            return loss_total

        # Pushforward / warmup: roll out without gradients to move the state distribution.
        y_prev = y0
        if skip_steps > 0:
            with torch.no_grad():
                for k in range(skip_steps):
                    y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
        y_prev = y_prev.detach()

        num_train_steps = int(k_train - skip_steps)
        log10_mae_sum = y_prev.new_zeros(())
        z_mse_sum = y_prev.new_zeros(())
        total_sum = y_prev.new_zeros(())

        # Train on the remaining steps, detaching after every step.
        for k in range(skip_steps, k_train):
            y_prev = y_prev.detach()

            y_next = self._forward_step(y_prev, dt_train[:, k], g_t)
            y_tgt = y_true[:, k, :]

            diff_z = (y_next - y_tgt).to(torch.float32)
            z_mse = diff_z.square().mean()

            pred_log10 = y_next.to(torch.float32).unsqueeze(1) * self.criterion.log_stds + self.criterion.log_means
            true_log10 = y_tgt.to(torch.float32).unsqueeze(1) * self.criterion.log_stds + self.criterion.log_means
            log10_mae = (pred_log10 - true_log10).abs().mean()

            loss_step = self.criterion.lambda_log10_mae * log10_mae + self.criterion.lambda_z_mse * z_mse

            if bool(getattr(self, "ar_backward_per_step", True)):
                # Scale so the accumulated gradient matches the mean loss over trained steps.
                self.manual_backward(loss_step / float(num_train_steps))
            else:
                total_sum = total_sum + loss_step

            log10_mae_sum = log10_mae_sum + log10_mae.detach()
            z_mse_sum = z_mse_sum + z_mse.detach()

            # Optionally apply teacher forcing for the next state's input, but always detach across steps.
            if not bool(getattr(self, "ar_teacher_forcing_in_trained_steps", True)) or tf_prob <= 0.0:
                y_prev = y_next.detach()
            elif tf_prob >= 1.0:
                y_prev = y_tgt.detach()
            else:
                use_tf = (torch.rand((B,), device=y_next.device) < tf_prob).view(B, 1)
                y_prev = torch.where(use_tf, y_tgt, y_next).detach()

        if not bool(getattr(self, "ar_backward_per_step", True)):
            self.manual_backward(total_sum / float(num_train_steps))

        # Optimizer step respecting gradient accumulation.
        is_last = (batch_idx + 1) >= int(getattr(self.trainer, "num_training_batches", batch_idx + 1))
        should_step = (((batch_idx + 1) % acc) == 0) or is_last
        if should_step:
            clip_val = getattr(self.trainer, "gradient_clip_val", None)
            if clip_val is not None and float(clip_val) > 0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=float(clip_val),
                    gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
                )
            opt.step()
            opt.zero_grad(set_to_none=True)

            sch = self.lr_schedulers()
            if sch is not None:
                if isinstance(sch, (list, tuple)):
                    for s in sch:
                        s.step()
                else:
                    sch.step()

        # Log an epoch-averaged loss consistent with the default criterion weighting.
        loss_log10_mae = log10_mae_sum / float(num_train_steps)
        loss_z_mse = z_mse_sum / float(num_train_steps)
        loss_total = self.criterion.lambda_log10_mae * loss_log10_mae + self.criterion.lambda_z_mse * loss_z_mse

        self.log("train_loss", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", loss_log10_mae, on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", loss_z_mse, on_step=False, on_epoch=True)

        return loss_total

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

        # Canonicalize dt to [B, transitions] (assumed already normalized by preprocessing).
        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context=f"{stage}/_shared_step")

        if stage == "train":
            k_train = int(max(1, min(self._effective_k_roll(epoch, stage="train"), transitions)))
            burn_in_train = int(self._get_burn_in_steps("train"))
            if burn_in_train >= k_train:
                raise ValueError(f"train: burn_in ({burn_in_train}) must be < rollout steps ({k_train}).")

            dt_train = dt_full[:, :k_train]
            y_in_train = y[:, :k_train, :]           # y_t inputs (y0..y_{K-1})
            y_true_train = y[:, 1: 1 + k_train, :]   # y_{t+1} targets

            tf_prob = float(self.tf_schedule.prob(epoch))

            # Fast path: exact teacher forcing (burn_in only affects weighting).
            if tf_prob == 1.0:
                y_pred_train = self._vectorized_teacher_forced_rollout(y_in=y_in_train, dt_bk=dt_train, g=g)
                step_weights_train = None
                if burn_in_train > 0:
                    w = torch.ones((k_train,), device=y_true_train.device, dtype=torch.float32)
                    w[: int(min(burn_in_train, k_train))] = 0.0
                    step_weights_train = w
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

            # Log the optimization loss as train_loss.
            self.log("train_loss", loss_opt.detach(), on_step=False, on_epoch=True, prog_bar=True)
            return loss_opt

        # Validation/test: open-loop autoregressive rollout (no teacher forcing).
        k_roll_sched = self._effective_k_roll(epoch, stage=stage)
        k_roll = int(max(1, min(int(k_roll_sched), transitions)))

        burn_in = int(self._get_burn_in_steps(stage))
        if burn_in >= k_roll:
            raise ValueError(f"{stage}: burn_in ({burn_in}) must be < rollout steps ({k_roll}).")

        dt_roll = dt_full[:, :k_roll]
        y_true = y[:, 1: 1 + k_roll, :]
        y0 = y[:, 0, :]

        fn_unroll = self._compiled_open_loop_unroll if self._compiled_open_loop_unroll is not None else self._open_loop_unroll
        y_pred, step_weights = fn_unroll(
            y0=y0,
            y_true=y_true,
            dt_bk=dt_roll,
            g=g,
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
        if getattr(self, "autoregressive_training", False):
            return self._training_step_autoregressive_detached(batch, batch_idx)
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    # ------------------------
    # Fit hooks
    # ------------------------

    def on_fit_start(self) -> None:
        if not self._warned_metric_mismatch:
            self._warned_metric_mismatch = True
            if getattr(self, "autoregressive_training", False):
                rank_zero_warn(
                    "train_loss (detached autoregressive rollout; stop-grad between steps) may not be directly comparable "
                    "to val_loss (open-loop rollout). Compare within the same rollout procedure."
                )
            else:
                rank_zero_warn(
                    "train_loss (training rollout with teacher forcing) is not directly comparable to val_loss "
                    "(open-loop rollout). Compare within the same rollout procedure."
                )

    # ------------------------
    # Optimizer + scheduler
    # ------------------------

    def configure_optimizers(self) -> Any:
        tcfg = self.hparams["training"]
        opt_cfg = _require_mapping(tcfg, "optimizer", context="cfg.training")

        name = str(_require_key(opt_cfg, "name", context="cfg.training.optimizer")).lower()
        lr = _as_float(_require_key(opt_cfg, "lr", context="cfg.training.optimizer"), context="cfg.training.optimizer.lr")
        weight_decay = _as_float(opt_cfg.get("weight_decay", 0.0), context="cfg.training.optimizer.weight_decay")

        params = [p for p in self.parameters() if p.requires_grad]

        if name in ("adam", "adamw"):
            betas = opt_cfg.get("betas", (0.9, 0.999))
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise TypeError("optimizer.betas must be length-2 list/tuple.")
            b0 = _as_float(betas[0], context="cfg.training.optimizer.betas[0]")
            b1 = _as_float(betas[1], context="cfg.training.optimizer.betas[1]")

            eps = _as_float(opt_cfg.get("eps", 1e-8), context="cfg.training.optimizer.eps")
            if name == "adam":
                opt = torch.optim.Adam(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay)
            else:
                opt = torch.optim.AdamW(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer name: {name} (supported: adam, adamw)")

        if not self.sched_cfg or not bool(self.sched_cfg.get("enabled", False)):
            return opt

        sched_type = str(self.sched_cfg.get("type", "cosine_with_warmup")).lower().strip()
        if sched_type not in ("cosine_with_warmup", "cosine-warmup", "cosine"):
            raise ValueError(f"Unsupported scheduler type: {sched_type} (supported: cosine_with_warmup)")

        # Scheduler: cosine with warmup (steps)
        max_steps = int(self.trainer.estimated_stepping_batches)
        warmup_epochs = _as_int(self.sched_cfg.get("warmup_epochs", 0), context="cfg.training.scheduler.warmup_epochs")
        min_lr_ratio = _as_float(self.sched_cfg.get("min_lr_ratio", 0.0), context="cfg.training.scheduler.min_lr_ratio")

        # Convert warmup_epochs to warmup steps; best-effort.
        warmup_steps = 0
        if warmup_epochs > 0:
            try:
                steps_per_epoch = int(self.trainer.estimated_stepping_batches) // max(1, int(self.max_epochs))
                warmup_steps = int(warmup_epochs) * max(1, steps_per_epoch)
            except Exception:
                warmup_steps = 0

        warmup_steps = int(max(0, min(warmup_steps, max_steps)))

        def lr_lambda(step: int) -> float:
            s = int(step)
            if warmup_steps > 0 and s < warmup_steps:
                return float(s) / float(max(1, warmup_steps))

            # Cosine decay
            if max_steps <= warmup_steps:
                return float(min_lr_ratio)

            progress = float(s - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }


# ==============================================================================
# Lightning trainer factory
# ==============================================================================


def build_lightning_trainer(cfg: Mapping[str, Any], *, work_dir: Path) -> pl.Trainer:
    tcfg = _require_mapping(cfg, "training", context="cfg")
    runtime = dict(cfg.get("runtime", {}) or {})

    max_epochs = _as_int(_require_key(tcfg, "max_epochs", context="cfg.training"), context="cfg.training.max_epochs")
    devices = runtime.get("devices", "auto")
    accelerator = runtime.get("accelerator", "auto")
    precision = runtime.get("precision", "16-mixed")
    log_every_n_steps = _as_int(runtime.get("log_every_n_steps", 50), context="cfg.runtime.log_every_n_steps")

    # Optional checkpointing
    ckpt_cfg = dict(runtime.get("checkpointing", {}) or {})
    enable_ckpt = bool(ckpt_cfg.get("enabled", True))
    ckpt_every_n_epochs = _as_int(ckpt_cfg.get("every_n_epochs", 1), context="cfg.runtime.checkpointing.every_n_epochs")
    save_top_k = _as_int(ckpt_cfg.get("save_top_k", 1), context="cfg.runtime.checkpointing.save_top_k")
    monitor = str(ckpt_cfg.get("monitor", "val_loss"))

    callbacks: List[pl.Callback] = [CSVLoggerCallback(work_dir=Path(work_dir))]

    if enable_ckpt:
        ckpt_dir = Path(work_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="epoch{epoch:03d}-val{val_loss:.6f}",
                monitor=monitor,
                save_top_k=save_top_k,
                mode="min",
                every_n_epochs=ckpt_every_n_epochs,
                save_last=True,
            )
        )

    # Optional gradient clipping
    grad_clip = runtime.get("gradient_clip_val", None)
    gradient_clip_val = float(grad_clip) if grad_clip is not None else None

    # Strategy
    strategy = runtime.get("strategy", "auto")
    accumulate_grad_batches = _as_int(runtime.get("accumulate_grad_batches", 1), context="cfg.runtime.accumulate_grad_batches")

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        enable_progress_bar=bool(runtime.get("enable_progress_bar", True)),
        enable_checkpointing=bool(enable_ckpt),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=bool(runtime.get("deterministic", False)),
    )
    return trainer
