#!/usr/bin/env python3
"""
trainer.py - PyTorch Lightning training module and trainer factory.

Core behavior:

- Training loss (train_loss) is computed under one of two training procedures:
    * If cfg.training.autoregressive_training.enabled is false (default):
        - **one-jump training** (single-step supervised update): predict y_{t+1} from y_t using only the first transition in the batch
        - burn-in for training must be 0 (burn-in is only used for evaluation or the detached rollout regime)
    * If cfg.training.autoregressive_training.enabled is true:
        - detached autoregressive rollout (pushforward + stop-grad between steps)
        - first `skip_steps` steps run under torch.no_grad() and are excluded from training
        - remaining steps train per-step, detaching state between steps (no BPTT)

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
- train_loss_log10_mae
- train_loss_z_mse
- val_loss
- val_loss_log10_mae
- val_loss_z_mse
- test_loss (test phase only)
- epoch_time_sec
- lr
- train_tf_prob (always 0.0; teacher forcing removed)
- train_rollout_steps
- train_burn_in
- train_skip_steps

This file assumes configuration has been validated upstream (e.g., in main.py).

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
# CSV epoch logger callback
# ==============================================================================


class CSVLoggerCallback(pl.Callback):
    """Write epoch-level metrics to work_dir/metrics.csv (rank-zero only).

    This callback writes a fixed, stable schema to avoid schema drift across runs/configs.
    Missing metrics are written as empty cells. Unexpected metrics are ignored.

    Default columns:
      - epoch
      - train_loss, train_loss_log10_mae, train_loss_z_mse
      - val_loss, val_loss_log10_mae, val_loss_z_mse
      - lr
      - epoch_time_sec
      - train_tf_prob, train_rollout_steps, train_burn_in, train_skip_steps
    """

    DEFAULT_FIELDS: List[str] = [
        "epoch",
        "train_loss",
        "train_loss_log10_mae",
        "train_loss_z_mse",
        "val_loss",
        "val_loss_log10_mae",
        "val_loss_z_mse",
        "lr",
        "epoch_time_sec",
        "train_tf_prob",
        "train_rollout_steps",
        "train_burn_in",
        "train_skip_steps",
    ]

    def __init__(self, *, work_dir: Path, filename: str = "metrics.csv", fieldnames: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.work_dir = Path(work_dir)
        self.filename = str(filename)
        self._epoch_start_time: Optional[float] = None

        self._header: List[str] = list(fieldnames) if fieldnames is not None else list(self.DEFAULT_FIELDS)

    def _csv_path(self) -> Path:
        return self.work_dir / self.filename

    @staticmethod
    def _scalar_to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                return None
            return float(v.detach().cpu().item())
        if isinstance(v, (int, float)):
            return float(v)
        return None

    @staticmethod
    def _extract_lr(trainer: pl.Trainer, pl_module: pl.LightningModule) -> Optional[float]:
        # Best-effort extraction of current LR from the first optimizer param group.
        opt_obj: Any = None
        # Try trainer.optimizers (property or list)
        try:
            opt_obj = getattr(trainer, "optimizers", None)
            if callable(opt_obj):
                opt_obj = opt_obj()
        except Exception:
            opt_obj = None

        # Try module optimizers() accessor if needed
        if opt_obj is None:
            try:
                opt_obj = pl_module.optimizers()
            except Exception:
                opt_obj = None

        if isinstance(opt_obj, (list, tuple)):
            if len(opt_obj) == 0:
                return None
            opt = opt_obj[0]
        else:
            opt = opt_obj

        try:
            if opt is None or not hasattr(opt, "param_groups") or not opt.param_groups:
                return None
            lr = opt.param_groups[0].get("lr", None)
            return float(lr) if lr is not None else None
        except Exception:
            return None

    def _ensure_header_matches_file(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.reader(f)
                existing = next(reader, None)
            if existing is None:
                return
            if list(existing) != list(self._header):
                raise RuntimeError(
                    f"CSVLoggerCallback schema mismatch. Expected header={self._header}, got existing header={existing}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to validate existing CSV header at {csv_path}") from e

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        row: Dict[str, Any] = {k: "" for k in self._header}

        row["epoch"] = int(getattr(trainer, "current_epoch", 0))
        if self._epoch_start_time is not None:
            row["epoch_time_sec"] = float(time.time() - self._epoch_start_time)

        lr = self._extract_lr(trainer, pl_module)
        if lr is not None:
            row["lr"] = float(lr)

        for k in self._header:
            if k in ("epoch", "epoch_time_sec", "lr"):
                continue
            if k in metrics:
                v = self._scalar_to_float(metrics.get(k))
                if v is not None:
                    row[k] = v

        csv_path = self._csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_header_matches_file(csv_path)

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
    """Create [1,1,S] log10_mean/log10_std tensors in the order of `species_variables`."""

    stats = (
        normalization_manifest.get("per_key_stats")
        or normalization_manifest.get("species_stats")
        or normalization_manifest.get("stats")
    )

    means = [float(stats[name]["log_mean"]) for name in species_variables]
    stds = [float(stats[name]["log_std"]) for name in species_variables]

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
            compile_forward_step = bool(tc.get("compile_forward_step", False))
            compile_open_loop = bool(tc.get("compile_open_loop_unroll", False))

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

        tcfg = cfg["training"]
        mcfg = cfg["model"]

        self.max_epochs = int(tcfg["max_epochs"])
        # Rollout steps (single horizon for train/val/test; inference is autoregressive by feeding predictions back in).
        self.rollout_steps = int(tcfg["rollout_steps"])

        # Burn-in (required explicit keys)
        burn_cfg = tcfg["burn_in"]
        self.burn_in_train = int(burn_cfg["train"])
        self.burn_in_val = int(burn_cfg["val"])
        self.burn_in_test = int(burn_cfg["test"])

        # Autoregressive training (pushforward + stop-grad) - optional.
        # When enabled, training uses a detached autoregressive rollout:
        #   - first `skip_steps` are rolled out under torch.no_grad() (no training)
        #   - remaining steps train per-step, detaching the state after every step
        ar_cfg = dict(tcfg.get("autoregressive_training", {}) or {})
        self.autoregressive_training = bool(ar_cfg.get("enabled", False))
        self.ar_no_grad_steps = int(ar_cfg.get("no_grad_steps", 0))
        self.ar_backward_per_step = bool(ar_cfg.get("backward_per_step", True))

        if self.autoregressive_training:
            # Required for per-step backward/step control.
            self.automatic_optimization = False

        # Optional curriculum
        cur_cfg = dict(tcfg.get("curriculum", {}) or {})
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg.get("enabled", False)),
            mode=str(cur_cfg.get("mode", "linear")),
            base_k=int(cur_cfg.get("base_k", self.rollout_steps)),
            max_k=int(cur_cfg.get("max_k", self.rollout_steps)),
            ramp_epochs=int(cur_cfg.get("ramp_epochs", 0)),
        )

        # Loss (required explicit keys; no aliases/defaults)
        loss_cfg = tcfg["loss"]
        lambda_log10_mae = float(loss_cfg["lambda_log10_mae"])
        lambda_z_mse = float(loss_cfg["lambda_z_mse"])

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
        # Determine rollout steps under the optional curriculum.
        # Curriculum is a training schedule; evaluation uses the configured base horizon.
        if stage == "train" and self.curriculum.enabled:
            base = int(self.curriculum.steps(epoch))
        else:
            base = int(self.rollout_steps)

        base = int(max(1, base))
        return int(max(1, base))

    def _get_burn_in_steps(self, stage: str) -> int:
        if stage == "train":
            return int(self.burn_in_train)
        if stage == "val":
            return int(self.burn_in_val)
        if stage == "test":
            return int(self.burn_in_test)
        raise ValueError(f"Unknown stage for burn-in: {stage}")

    def _open_loop_unroll(
        self,
        *,
        y0: torch.Tensor,        # [B, S]
        y_true: torch.Tensor,    # [B, K, S] targets for steps 1..K (used for burn-in inputs)
        dt_bk: torch.Tensor,     # [B, K]
        g: Optional[torch.Tensor],
        burn_in: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Deterministic rollout for eval (open-loop), with optional burn-in."""

        B, K, S = y_true.shape
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

        """
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)

        B, K_full, S = y.shape
        transitions = max(1, int(K_full) - 1)
        epoch = int(self.current_epoch)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/_training_step_autoregressive_detached")

        k_train = int(max(1, min(self._effective_k_roll(epoch, stage="train"), transitions)))
        dt_train = dt_full[:, :k_train]                 # [B, k_train]
        y_true = y[:, 1: 1 + k_train, :]                # [B, k_train, S]
        y0 = y[:, 0, :]                                 # [B, S]

        # Steps to exclude from gradient updates.
        burn_in_train = int(self._get_burn_in_steps("train"))
        skip_steps = int(max(0, max(burn_in_train, int(getattr(self, "ar_no_grad_steps", 0)))))
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

            losses = self.criterion(y_next.unsqueeze(1), y_tgt.unsqueeze(1), step_weights=None)
            loss_step = losses["loss_total"]

            if bool(getattr(self, "ar_backward_per_step", True)):
                # Scale so the accumulated gradient matches the mean loss over trained steps.
                self.manual_backward(loss_step / float(num_train_steps))
            else:
                total_sum = total_sum + loss_step

            log10_mae_sum = log10_mae_sum + losses["loss_log10_mae"].detach()
            z_mse_sum = z_mse_sum + losses["loss_z_mse"].detach()
            y_prev = y_next.detach()

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

        self.log("train_tf_prob", float(0.0), on_step=False, on_epoch=True)
        self.log("train_rollout_steps", float(k_train), on_step=False, on_epoch=True)
        self.log("train_burn_in", float(burn_in_train), on_step=False, on_epoch=True)
        self.log("train_skip_steps", float(skip_steps), on_step=False, on_epoch=True)

        return loss_total

    def _training_step_one_jump(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single-step supervised training ("one-jump").

        Uses only the first transition in the sequence:
          y0 := y[:, 0, :]
          y1 := y[:, 1, :]
          dt0 := dt_full[:, 0]

        Rationale: this is the clean 1-step training regime used as a standalone stage (e.g. before
        switching to detached autoregressive rollout training). Multi-step teacher-forced rollout
        training is intentionally not supported in this refactored trainer.
        """
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)

        B, K_full, S = y.shape
        transitions = max(1, int(K_full) - 1)

        # Canonicalize dt to [B, transitions].
        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/_training_step_one_jump")

        # In one-jump mode, burn-in is nonsensical (K=1). Require train burn-in to be 0.
        burn_in_train = int(self._get_burn_in_steps("train"))
        if burn_in_train != 0:
            raise ValueError(
                f"one-jump training requires cfg.training.burn_in.train == 0, got {burn_in_train}."
            )

        if int(K_full) < 2:
            raise ValueError("one-jump training requires at least 2 states in the batch sequence (K_full >= 2).")

        y0 = y[:, 0, :]          # [B, S]
        y1 = y[:, 1, :]          # [B, S]
        dt0 = dt_full[:, 0]      # [B]

        g_t = self._coerce_g(g, B, y0, context="_training_step_one_jump")
        y_pred1 = self._forward_step(y0, dt0, g_t)  # [B, S]

        y_pred = y_pred1.unsqueeze(1)   # [B, 1, S]
        y_true = y1.unsqueeze(1)        # [B, 1, S]

        losses = self.criterion(y_pred, y_true, step_weights=None)
        loss_total = losses["loss_total"]

        # Epoch-level logs (stable schema).
        self.log("train_loss", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)

        # For schema stability, keep these fields even though they are not meaningful in one-jump mode.
        self.log("train_tf_prob", float(0.0), on_step=False, on_epoch=True)
        self.log("train_rollout_steps", float(1.0), on_step=False, on_epoch=True)
        self.log("train_burn_in", float(0.0), on_step=False, on_epoch=True)
        self.log("train_skip_steps", float(0.0), on_step=False, on_epoch=True)

        return loss_total

    def _eval_step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared validation/test step: open-loop autoregressive rollout (no teacher forcing)."""
        if stage not in ("val", "test"):
            raise ValueError(f"_eval_step stage must be 'val' or 'test', got {stage}")
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)

        B, K_full, S = y.shape
        transitions = max(1, int(K_full) - 1)
        epoch = int(self.current_epoch)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context=f"{stage}/_eval_step")

        k_roll_sched = self._effective_k_roll(epoch, stage=stage)
        k_roll = int(max(1, min(int(k_roll_sched), transitions)))

        burn_in = int(self._get_burn_in_steps(stage))
        if burn_in >= k_roll:
            raise ValueError(f"{stage}: burn_in ({burn_in}) must be < rollout steps ({k_roll}).")

        dt_roll = dt_full[:, :k_roll]
        y_true = y[:, 1 : 1 + k_roll, :]
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
        else:  # test
            self.log("test_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("test_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)

        return loss_total

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if getattr(self, "autoregressive_training", False):
            return self._training_step_autoregressive_detached(batch, batch_idx)
        return self._training_step_one_jump(batch, batch_idx)

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, stage="val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, stage="test")

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
                    "train_loss (one-jump training) is not directly comparable to val_loss "
                    "(open-loop rollout). Compare within the same rollout procedure."
                )

    # ------------------------
    # Optimizer + scheduler
    # ------------------------

    def configure_optimizers(self) -> Any:
        tcfg = self.hparams["training"]
        opt_cfg = tcfg["optimizer"]

        name = str(opt_cfg["name"]).lower()
        lr = float(opt_cfg["lr"])
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        params = [p for p in self.parameters() if p.requires_grad]

        if name in ("adam", "adamw"):
            betas = opt_cfg.get("betas", (0.9, 0.999))
            b0, b1 = betas
            eps = float(opt_cfg.get("eps", 1e-8))

            # These are optional config knobs (defaults are chosen for throughput on CUDA).
            use_fused = bool(opt_cfg.get("fused", True))
            use_foreach = bool(opt_cfg.get("foreach", True))

            if name == "adam":
                if use_fused:
                    try:
                        opt = torch.optim.Adam(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, fused=True)
                    except TypeError:
                        opt = torch.optim.Adam(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, foreach=use_foreach)
                else:
                    opt = torch.optim.Adam(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, foreach=use_foreach)
            else:  # adamw
                if use_fused:
                    try:
                        opt = torch.optim.AdamW(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, fused=True)
                    except TypeError:
                        opt = torch.optim.AdamW(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, foreach=use_foreach)
                else:
                    opt = torch.optim.AdamW(params, lr=lr, betas=(b0, b1), eps=eps, weight_decay=weight_decay, foreach=use_foreach)
        else:
            raise ValueError(f"Unsupported optimizer name: {name} (supported: adam, adamw)")



        if not self.sched_cfg or not bool(self.sched_cfg.get("enabled", False)):
            return opt

        sched_type = str(self.sched_cfg.get("type", "cosine_with_warmup")).lower().strip()

        # -------------------------
        # Plateau scheduler
        # -------------------------
        if sched_type in ("plateau", "reduce_on_plateau"):
            factor = float(self.sched_cfg.get("factor", 0.5))
            patience = int(self.sched_cfg.get("patience", 10))
            threshold = float(self.sched_cfg.get("threshold", 1e-4))
            min_lr = float(self.sched_cfg.get("min_lr", 1e-7))
            mode = str(self.sched_cfg.get("mode", "min"))
            monitor = str(self.sched_cfg.get("monitor", "val_loss"))

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
            )

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": monitor,
                    "name": "lr",
                },
            }

        # -------------------------
        # Cosine with warmup scheduler
        # -------------------------
        if sched_type not in ("cosine_with_warmup", "cosine-warmup", "cosine"):
            raise ValueError(f"Unsupported scheduler type: {sched_type} (supported: cosine_with_warmup, plateau)")

        max_steps = int(self.trainer.estimated_stepping_batches)
        warmup_epochs = int(self.sched_cfg.get("warmup_epochs", 0))
        min_lr_ratio = float(self.sched_cfg.get("min_lr_ratio", 0.0))

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
    tcfg = cfg["training"]
    runtime = dict(cfg.get("runtime", {}) or {})

    max_epochs = int(tcfg["max_epochs"])
    devices = runtime.get("devices", "auto")
    accelerator = runtime.get("accelerator", "auto")
    precision = runtime.get("precision", "16-mixed")
    log_every_n_steps = int(runtime.get("log_every_n_steps", 50))

    # Optional checkpointing
    ckpt_cfg = dict(runtime.get("checkpointing", {}) or {})
    enable_ckpt = bool(ckpt_cfg.get("enabled", True))
    ckpt_every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1))
    save_top_k = int(ckpt_cfg.get("save_top_k", 1))
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
    accumulate_grad_batches = int(runtime.get("accumulate_grad_batches", 1))

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,

        # Throughput: remove per-step logger/progress overhead (CSVLoggerCallback remains).
        logger=False,
        enable_progress_bar=bool(runtime.get("enable_progress_bar", False)),
        enable_model_summary=False,
        num_sanity_val_steps=0,

        enable_checkpointing=bool(enable_ckpt),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=bool(runtime.get("deterministic", False)),
    )

    return trainer
