#!/usr/bin/env python3
"""
trainer.py - LightningModule + Trainer factory.

Training styles (burn-in removed):

1) One-jump
   - training.rollout_steps must be 1
   - loss on the first transition only

2) Autoregressive rollout
   - Total rollout steps K := training.rollout_steps
   - Skip/warmup steps M := training.autoregressive_training.skip_steps
       * warmup is run under torch.no_grad(), then detached (pushforward trick input generation)
       * warmup steps are excluded from the loss via masking (and/or by construction in detached-per-step mode)
   - Gradients:
       * detach_between_steps=True  : stop-grad between steps (no BPTT across time)
       * detach_between_steps=False : BPTT across steps [M..K-1] (but never through warmup)

Optional curriculum (training only):
- training.curriculum ramps training rollout steps from start_steps -> rollout_steps.
- Validation/test always use rollout_steps (fixed horizon for comparability).

Logging:
- Uses Lightning CSVLogger. All self.log(...) metrics are written to metrics.csv.
- LearningRateMonitor logs LR automatically (naming depends on Lightning version).
- epoch_time_sec is logged manually (so we do not need Timer).

Batch contract (from dataset.py):
- y:  [B, K_full, S]   (states in z-space)
- dt: [B, K_full-1]    (per-transition normalized dt)
- g:  [B, G]           (G may be 0)

Note on "pushforward trick":
- skip_steps warmup runs A(u) forward without grad, and training loss is computed on later steps.
"""

from __future__ import annotations

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

from utils import PrecisionConfig, parse_precision_config
# Lightning imports with compatibility (lightning.pytorch vs pytorch_lightning)
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
    from pytorch_lightning.loggers import CSVLogger  # type: ignore

# Rank-zero warn helper (varies across versions)
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
# Optimizer parameter groups
# ==============================================================================

EXCLUDE_NORM_AND_BIAS_FROM_WEIGHT_DECAY_BY_DEFAULT = True

_NORM_MODULE_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
if hasattr(nn, "SyncBatchNorm"):
    _NORM_MODULE_TYPES = _NORM_MODULE_TYPES + (nn.SyncBatchNorm,)


def _build_optimizer_param_groups(
    module: nn.Module,
    *,
    weight_decay: float,
    exclude_norm_and_bias: bool,
) -> List[Dict[str, Any]]:
    """Create optimizer param groups, optionally excluding norm/bias from weight decay."""
    wd = float(weight_decay)
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]

    if wd == 0.0 or not exclude_norm_and_bias:
        return [{"params": [p for _, p in named], "weight_decay": wd}]

    norm_param_names: set[str] = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, _NORM_MODULE_TYPES):
            for pn, _ in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{pn}" if mod_name else pn
                norm_param_names.add(full)

    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, p in named:
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "bias" or name in norm_param_names:
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


# ==============================================================================
# Loss
# ==============================================================================


class HybridLoss(nn.Module):
    """Two-term loss: log10-space MAE and z-space MSE.

    z is normalized log10-space:
        z = (log10(y_phys) - log_mean) / log_std
    """

    def __init__(
        self,
        log_means: torch.Tensor,  # [1,1,S]
        log_stds: torch.Tensor,   # [1,1,S]
        *,
        lambda_log10_mae: float,
        lambda_z_mse: float,
        loss_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.loss_dtype = loss_dtype
        self.register_buffer("log_means", log_means.clone().detach().to(dtype=loss_dtype))
        self.register_buffer("log_stds", log_stds.clone().detach().to(dtype=loss_dtype))
        self.lambda_log10_mae = float(lambda_log10_mae)
        self.lambda_z_mse = float(lambda_z_mse)

    def _denormalize_log10(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.log_stds + self.log_means

    def forward(
        self,
        pred_z: torch.Tensor,  # [B,K,S]
        true_z: torch.Tensor,  # [B,K,S]
        *,
        step_weights: Optional[torch.Tensor] = None,  # [K]
    ) -> Dict[str, torch.Tensor]:
        if pred_z.shape != true_z.shape:
            raise ValueError(f"Shape mismatch: pred_z={tuple(pred_z.shape)} true_z={tuple(true_z.shape)}")
        if pred_z.ndim != 3:
            raise ValueError(f"Expected [B,K,S], got {tuple(pred_z.shape)}")

        B, K, S = pred_z.shape

        w: Optional[torch.Tensor] = None
        denom: Optional[torch.Tensor] = None
        if step_weights is not None:
            if step_weights.ndim != 1 or int(step_weights.shape[0]) != int(K):
                raise ValueError(f"step_weights must have shape [K={K}], got {tuple(step_weights.shape)}")
            w = step_weights.to(device=pred_z.device, dtype=self.loss_dtype).view(1, K, 1)
            denom = (w.sum() * float(B * S)).clamp_min(_LOSS_DENOM_EPS)

        diff_z = (pred_z - true_z).to(self.loss_dtype)
        if w is None:
            z_mse = diff_z.square().mean()
        else:
            z_mse = diff_z.square().mul(w).sum(dtype=self.loss_dtype) / denom  # type: ignore[arg-type]

        # log10(y_phys) = z * log_std + log_mean, so the log-mean cancels in differences:
        #   |pred_log10 - true_log10| = |(pred_z - true_z) * log_std|
        diff_log10 = diff_z.abs().mul(self.log_stds)

        if w is None:
            log10_mae = diff_log10.mean()
        else:
            log10_mae = diff_log10.mul(w).sum(dtype=self.loss_dtype) / denom  # type: ignore[arg-type]

        loss_total = self.lambda_log10_mae * log10_mae + self.lambda_z_mse * z_mse
        return {"loss_total": loss_total, "loss_log10_mae": log10_mae, "loss_z_mse": z_mse}


def build_loss_buffers(
    normalization_manifest: Mapping[str, Any],
    species_variables: Sequence[str],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create [1,1,S] log10_mean/log10_std tensors matching species_variables order."""
    if "per_key_stats" not in normalization_manifest:
        raise KeyError("Normalization manifest missing 'per_key_stats'.")
    stats = normalization_manifest["per_key_stats"]
    if not isinstance(stats, Mapping):
        raise TypeError("normalization manifest 'per_key_stats' must be a mapping.")

    means = [float(stats[name]["log_mean"]) for name in species_variables]
    stds = [float(stats[name]["log_std"]) for name in species_variables]
    log_means = torch.tensor(means, device=device, dtype=dtype).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device, dtype=dtype).view(1, 1, -1)
    return log_means, log_stds


# ==============================================================================
# torch.compile helper
# ==============================================================================


def _try_torch_compile(fn: Any, *, cfg: Mapping[str, Any], context: str) -> Any:
    """Optionally wrap a callable with torch.compile (strict)."""
    runtime = dict(cfg.get("runtime", {}) or {})
    tc = dict(runtime.get("torch_compile", {}) or {})
    if not bool(tc.get("enabled", False)):
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
# Curriculum
# ==============================================================================


@dataclass(frozen=True)
class RolloutCurriculum:
    """Ramps training rollout steps from start_steps -> end_steps over ramp_epochs.

    """

    enabled: bool
    mode: str
    start_steps: int
    end_steps: int
    ramp_epochs: int

    def steps(self, epoch: int) -> int:
        end_k = int(max(1, self.end_steps))
        if not self.enabled:
            return end_k

        start_k = int(max(1, self.start_steps))
        if self.ramp_epochs <= 0 or start_k >= end_k:
            return start_k

        e = int(max(0, epoch))
        t = min(1.0, float(e) / float(max(1, int(self.ramp_epochs))))
        mode = str(self.mode).lower()

        if mode == "linear":
            f = t
        elif mode == "cosine":
            f = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown curriculum.mode '{self.mode}' (expected: linear|cosine)")

        k = start_k + int((end_k - start_k) * f)
        return int(max(1, min(end_k, k)))


# ==============================================================================
# Lightning module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """LightningModule wrapper around the underlying model (must implement forward_step)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg: Mapping[str, Any],
        normalization_manifest: Mapping[str, Any],
        species_variables: Sequence[str],
        precision: Optional[PrecisionConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalization_manifest = dict(normalization_manifest)
        self.species_variables = list(species_variables)

        # Precision (centralized): controls dataset/model/loss dtypes and Lightning AMP settings.
        self.precision_config = precision if precision is not None else parse_precision_config(cfg)
        self.model_dtype = self.precision_config.model_dtype
        self.input_dtype = self.precision_config.input_dtype
        self.loss_dtype = self.precision_config.loss_dtype

        tcfg = dict(cfg["training"])
        mcfg = dict(cfg.get("model", {}) or {})

        self.max_epochs = int(tcfg["max_epochs"])
        self.rollout_steps = int(tcfg["rollout_steps"])
        if self.rollout_steps < 1:
            raise ValueError(f"training.rollout_steps must be >= 1, got {self.rollout_steps}")

        ar_cfg = dict(tcfg.get("autoregressive_training", {}) or {})
        self.autoregressive_training = bool(ar_cfg.get("enabled", False))
        if self.autoregressive_training:
            if "skip_steps" not in ar_cfg:
                raise KeyError("missing: training.autoregressive_training.skip_steps")
            if "detach_between_steps" not in ar_cfg:
                raise KeyError("missing: training.autoregressive_training.detach_between_steps")
            self.ar_skip_steps = int(ar_cfg["skip_steps"])
            self.ar_detach_between_steps = bool(ar_cfg["detach_between_steps"])
        else:
            self.ar_skip_steps = 0
            self.ar_detach_between_steps = True
        self.ar_backward_per_step = bool(ar_cfg.get("backward_per_step", True))

        if not self.autoregressive_training and self.rollout_steps != 1:
            raise ValueError(
                "One-jump training requires training.rollout_steps == 1. "
                f"Got rollout_steps={self.rollout_steps} with autoregressive_training.enabled=False."
            )

        if self.autoregressive_training:
            self.automatic_optimization = False

        # Curriculum (training only).
        cur_cfg = dict(tcfg.get("curriculum", {}) or {})
        cur_enabled = bool(cur_cfg.get("enabled", False))
        if cur_enabled:
            if "start_steps" not in cur_cfg:
                raise KeyError("missing: training.curriculum.start_steps")
            if "end_steps" not in cur_cfg:
                raise KeyError("missing: training.curriculum.end_steps")
            start_steps = int(cur_cfg["start_steps"])
            end_steps = int(cur_cfg["end_steps"])
        else:
            start_steps = int(cur_cfg.get("start_steps", 1))
            end_steps = int(cur_cfg.get("end_steps", self.rollout_steps))
        self.curriculum = RolloutCurriculum(
            enabled=cur_enabled,
            mode=str(cur_cfg.get("mode", "linear")),
            start_steps=start_steps,
            end_steps=end_steps,
            ramp_epochs=int(cur_cfg.get("ramp_epochs", 0)),
        )

        # Loss
        loss_cfg = dict(tcfg["loss"])
        log_means, log_stds = build_loss_buffers(
            self.normalization_manifest,
            self.species_variables,
            device=torch.device("cpu"),
            dtype=self.loss_dtype,
        )
        self.criterion = HybridLoss(
            log_means,
            log_stds,
            lambda_log10_mae=float(loss_cfg["lambda_log10_mae"]),
            lambda_z_mse=float(loss_cfg["lambda_z_mse"]),
            loss_dtype=self.loss_dtype,
        )

        self.sched_cfg = dict(tcfg.get("scheduler", {}) or {})

        # Optional compilation
        runtime = dict(cfg.get("runtime", {}) or {})
        tc = dict(runtime.get("torch_compile", {}) or {})
        self._compiled_forward_step: Optional[Any] = None
        self._compiled_open_loop_unroll: Optional[Any] = None
        if bool(tc.get("enabled", False)):
            if bool(tc.get("compile_forward_step", False)):
                if not hasattr(self.model, "forward_step"):
                    raise RuntimeError("compile_forward_step=True but model has no forward_step method.")
                self._compiled_forward_step = _try_torch_compile(
                    self.model.forward_step, cfg=cfg, context="FlowMapRolloutModule.model.forward_step"  # type: ignore[attr-defined]
                )
            if bool(tc.get("compile_open_loop_unroll", False)):
                self._compiled_open_loop_unroll = _try_torch_compile(
                    self._open_loop_unroll_raw_step, cfg=cfg, context="FlowMapRolloutModule._open_loop_unroll_raw_step"
                )

        # Dataloader wiring (kept for compatibility; main.py can also pass dataloaders directly)
        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

        # Epoch timing (logged; CSVLogger writes it)
        self._epoch_t0: Optional[float] = None

        # UX warnings
        self._warned_metric_mismatch = False

        # Preserve for checkpoint reproducibility
        self.save_hyperparameters({"training": dict(tcfg), "model": dict(mcfg)})

    # ------------------------
    # Dataloader wiring (optional)
    # ------------------------

    def set_dataloaders(self, train_dl: Any, val_dl: Any = None, test_dl: Any = None) -> None:
        self._train_dl, self._val_dl, self._test_dl = train_dl, val_dl, test_dl

    def train_dataloader(self) -> Any:
        if self._train_dl is None:
            raise RuntimeError("train_dataloader not set. Pass dataloaders to Trainer.fit(...) or call set_dataloaders().")
        return self._train_dl

    def val_dataloader(self) -> Any:
        return self._val_dl

    def test_dataloader(self) -> Any:
        return self._test_dl

    # ------------------------
    # Fit-time warning about metric comparability
    # ------------------------

    def on_fit_start(self) -> None:
        if self._warned_metric_mismatch:
            return
        self._warned_metric_mismatch = True

        if self.autoregressive_training:
            msg = (
                "train_loss (autoregressive training) may not be directly comparable to val_loss/test_loss "
                "(open-loop rollout evaluation), especially when using detach_between_steps=True and/or a rollout curriculum. "
                "Use the same split/stage for apples-to-apples comparisons, and log train_rollout_steps to track curriculum."
            )
        else:
            msg = (
                "train_loss (one-jump training) is not directly comparable to val_loss/test_loss (open-loop rollout evaluation). "
                "Compare within the same rollout procedure."
            )
        rank_zero_warn(msg)

    # ------------------------
    # Epoch timing
    # ------------------------

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.time()

    def on_train_epoch_end(self) -> None:
        if self._epoch_t0 is not None:
            self.log("epoch_time_sec", float(time.time() - self._epoch_t0), on_step=False, on_epoch=True)

    # ------------------------
    # Forward utilities
    # ------------------------

    def _forward_step(self, y_t: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "forward_step"):
            raise AttributeError("Model must implement forward_step(y_t, dt, g) -> y_{t+1}.")
        fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        return fn(y_t, dt, g)  # type: ignore[misc]

    def _coerce_g(self, g: Optional[torch.Tensor], batch_size: int, like: torch.Tensor) -> torch.Tensor:
        if g is None or g.numel() == 0:
            return like.new_zeros((batch_size, 0))

        if g.ndim == 1:
            g = g.view(1, -1).expand(batch_size, -1)
        elif g.ndim == 2:
            if int(g.shape[0]) == 1 and batch_size > 1:
                g = g.expand(batch_size, -1)
            elif int(g.shape[0]) != batch_size:
                raise ValueError(f"g batch dim mismatch: got {int(g.shape[0])}, expected {batch_size}")
        else:
            raise ValueError(f"g must have shape [G] or [B,G], got {tuple(g.shape)}")

        return g.to(device=like.device, dtype=like.dtype)

    def _train_rollout_steps(self, epoch: int, *, transitions: int) -> int:
        if not self.curriculum.enabled:
            return int(max(1, min(self.rollout_steps, transitions)))
        k = int(self.curriculum.steps(epoch))
        return int(max(1, min(self.rollout_steps, transitions, k)))

    def _eval_rollout_steps(self, *, transitions: int) -> int:
        return int(max(1, min(self.rollout_steps, transitions)))

    def _cast_batch_tensors(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        g: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Cast batch tensors to the configured input dtype.

        Lightning moves tensors to the target device; this function only adjusts dtype.
        """
        target = self.input_dtype
        if y.dtype != target:
            y = y.to(dtype=target)
        if dt.dtype != target:
            dt = dt.to(dtype=target)
        if g is not None and g.numel() > 0 and g.dtype != target:
            g = g.to(dtype=target)
        return y, dt, g

    @staticmethod
    def _step_weights(k: int, skip_steps: int, *, device: torch.device) -> Optional[torch.Tensor]:
        skip = int(max(0, skip_steps))
        if skip <= 0:
            return None
        w = torch.ones((int(k),), device=device, dtype=torch.float32)
        w[: min(skip, int(k))] = 0.0
        return w

    def _open_loop_unroll(
        self,
        *,
        y0: torch.Tensor,     # [B,S]
        dt_bk: torch.Tensor,  # [B,K]
        g: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Open-loop rollout: feed predictions back in (no teacher forcing)."""
        B, K = int(dt_bk.shape[0]), int(dt_bk.shape[1])
        g_t = self._coerce_g(g, B, y0)

        if not hasattr(self.model, "forward_step"):
            raise AttributeError("Model must implement forward_step(y_t, dt, g) -> y_{t+1}.")
        step_fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]

        y_pred = torch.empty((B, K, y0.shape[-1]), device=y0.device, dtype=y0.dtype)
        y_prev = y0
        for k in range(K):
            y_prev = step_fn(y_prev, dt_bk[:, k], g_t)  # type: ignore[misc]
            y_pred[:, k, :] = y_prev
        return y_pred

    def _open_loop_unroll_raw_step(
        self,
        *,
        y0: torch.Tensor,     # [B,S]
        dt_bk: torch.Tensor,  # [B,K]
        g: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Open-loop rollout using the raw (uncompiled) model step.

        This is used when compiling the unroll itself (compile_open_loop_unroll=True) so the
        compiler can see the full step-to-step dependency chain and fuse across steps.
        """
        B, K = int(dt_bk.shape[0]), int(dt_bk.shape[1])
        g_t = self._coerce_g(g, B, y0)

        if not hasattr(self.model, "forward_step"):
            raise AttributeError("Model must implement forward_step(y_t, dt, g) -> y_{t+1}.")
        step_fn = self.model.forward_step  # type: ignore[attr-defined]

        y_pred = torch.empty((B, K, y0.shape[-1]), device=y0.device, dtype=y0.dtype)
        y_prev = y0
        for k in range(K):
            y_prev = step_fn(y_prev, dt_bk[:, k], g_t)  # type: ignore[misc]
            y_pred[:, k, :] = y_prev
        return y_pred

    # ------------------------
    # Manual optimization helpers (autoregressive mode)
    # ------------------------

    def _get_single_optimizer(self) -> torch.optim.Optimizer:
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            if len(opt) != 1:
                raise RuntimeError("Expected exactly one optimizer.")
            return opt[0]
        return opt

    def _accumulate_grad_batches(self) -> int:
        acc = getattr(self.trainer, "accumulate_grad_batches", 1)
        if isinstance(acc, Mapping):
            acc = int(acc.get(0, list(acc.values())[0]))
        return int(max(1, int(acc)))

    def _maybe_step_optimizer(self, opt: torch.optim.Optimizer, batch_idx: int) -> None:
        acc = self._accumulate_grad_batches()
        is_last = (batch_idx + 1) >= int(getattr(self.trainer, "num_training_batches", batch_idx + 1))
        should_step = (((batch_idx + 1) % acc) == 0) or is_last
        if not should_step:
            return

        # Use the underlying torch optimizer for inspection/unscale when available.
        torch_opt = opt.optimizer if hasattr(opt, "optimizer") else opt

        self.log("lr", float(torch_opt.param_groups[0]["lr"]), on_step=False, on_epoch=True)

        # Unscale gradients so logged grad_norm is in true scale.
        # Only needed when a GradScaler is active (fp16 AMP); bf16/fp32 do not use one.
        _plugin = getattr(self.trainer.strategy, "precision_plugin", None) or getattr(self.trainer, "precision_plugin", None)
        if _plugin is not None and hasattr(_plugin, "unscale_gradients"):
            _plugin.unscale_gradients(torch_opt)

        gn2 = torch.zeros((), device=self.device, dtype=torch.float32)
        for pg in torch_opt.param_groups:
            for p in pg["params"]:
                if p.grad is None:
                    continue
                # Cast only the scalar norm to fp32 to avoid allocating a full fp32 gradient copy.
                n = p.grad.detach().norm(2).float()
                gn2 = gn2 + n * n
        self.log("grad_norm", torch.sqrt(gn2), on_step=False, on_epoch=True)

        clip_val = getattr(self.trainer, "gradient_clip_val", None)


        if clip_val is not None and float(clip_val) > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=float(clip_val),
                gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        # Step non-plateau schedulers here; plateau is stepped on validation end.
        sched = self.lr_schedulers()
        if sched is None:
            return
        sched_list = list(sched) if isinstance(sched, (list, tuple)) else [sched]
        for s in sched_list:
            sch = s.scheduler if hasattr(s, "scheduler") else s
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                continue
            sch.step()


    # ------------------------
    # Training / eval
    # ------------------------

    def _training_step_one_jump(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, _ = y.shape
        transitions = max(1, int(K_full) - 1)
        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/one_jump")

        if K_full < 2:
            raise ValueError("one-jump training requires K_full>=2 states per window.")

        y0 = y[:, 0, :]
        y1 = y[:, 1, :]
        dt0 = dt_full[:, 0]
        g_t = self._coerce_g(g, B, y0)

        y_pred1 = self._forward_step(y0, dt0, g_t)
        losses = self.criterion(y_pred1.unsqueeze(1), y1.unsqueeze(1), step_weights=None)

        self.log("train_loss", losses["loss_total"].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)

        opt0 = self.trainer.optimizers[0]
        torch_opt0 = opt0.optimizer if hasattr(opt0, "optimizer") else opt0
        self.log("lr", float(torch_opt0.param_groups[0]["lr"]), on_step=False, on_epoch=True)

        self.log("train_rollout_steps", float(1.0), on_step=False, on_epoch=True)

        self.log("train_skip_steps", float(0.0), on_step=False, on_epoch=True)
        self.log("train_detach_between_steps", float(1.0), on_step=False, on_epoch=True)

        return losses["loss_total"]


    def _training_step_autoregressive(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, S = y.shape
        transitions = max(1, int(K_full) - 1)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/autoregressive")
        k_train = self._train_rollout_steps(int(self.current_epoch), transitions=transitions)

        dt_train = dt_full[:, :k_train]
        y_true = y[:, 1 : 1 + k_train, :]
        y0 = y[:, 0, :]
        g_t = self._coerce_g(g, B, y0)

        skip = int(max(0, self.ar_skip_steps))
        if skip >= k_train:
            raise ValueError(f"skip_steps ({skip}) must be < rollout_steps ({k_train}).")

        opt = self._get_single_optimizer()
        acc = self._accumulate_grad_batches()
        if (batch_idx % acc) == 0:
            opt.zero_grad(set_to_none=True)

        # Warmup once: pushforward M steps without grad, then detach.
        y_prev = y0
        if skip > 0:
            with torch.no_grad():
                for k in range(skip):
                    y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
            y_prev = y_prev.detach()

        if self.ar_detach_between_steps:
            # Stop-grad between steps: no BPTT across time (but still trained on pushforward states).
            num_steps = int(k_train - skip)

            loss_total_sum = y_prev.new_zeros(())
            log10_mae_sum = y_prev.new_zeros(())
            z_mse_sum = y_prev.new_zeros(())

            for k in range(skip, k_train):
                y_next = self._forward_step(y_prev.detach(), dt_train[:, k], g_t)
                y_tgt = y_true[:, k, :]

                step_losses = self.criterion(y_next.unsqueeze(1), y_tgt.unsqueeze(1), step_weights=None)

                if self.ar_backward_per_step:
                    self.manual_backward(step_losses["loss_total"] / float(num_steps * acc))
                else:
                    loss_total_sum = loss_total_sum + step_losses["loss_total"]

                log10_mae_sum = log10_mae_sum + step_losses["loss_log10_mae"].detach()
                z_mse_sum = z_mse_sum + step_losses["loss_z_mse"].detach()

                y_prev = y_next.detach()

            if not self.ar_backward_per_step:
                self.manual_backward(loss_total_sum / float(num_steps * acc))

            self._maybe_step_optimizer(opt, batch_idx)

            loss_log10_mae = log10_mae_sum / float(num_steps)
            loss_z_mse = z_mse_sum / float(num_steps)
            loss_total = self.criterion.lambda_log10_mae * loss_log10_mae + self.criterion.lambda_z_mse * loss_z_mse

        else:
            # BPTT across the trained portion (after warmup). No gradients through warmup.
            y_pred = y0.new_zeros((B, k_train, S))

            for k in range(skip, k_train):
                y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
                y_pred[:, k, :] = y_prev

            step_weights = self._step_weights(k_train, skip, device=y_pred.device)
            losses = self.criterion(y_pred, y_true, step_weights=step_weights)

            loss_total = losses["loss_total"]
            loss_log10_mae = losses["loss_log10_mae"].detach()
            loss_z_mse = losses["loss_z_mse"].detach()

            self.manual_backward(loss_total / float(acc))
            self._maybe_step_optimizer(opt, batch_idx)

        self.log("train_loss", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", loss_log10_mae, on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", loss_z_mse, on_step=False, on_epoch=True)
        self.log("train_rollout_steps", float(k_train), on_step=False, on_epoch=True)
        self.log("train_skip_steps", float(skip), on_step=False, on_epoch=True)
        self.log("train_detach_between_steps", float(self.ar_detach_between_steps), on_step=False, on_epoch=True)

        return loss_total

    def _eval_step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        y = batch["y"]
        dt = batch["dt"]
        g = batch.get("g", None)
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, _ = y.shape
        transitions = max(1, int(K_full) - 1)
        k_eval = self._eval_rollout_steps(transitions=transitions)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context=f"{stage}/eval")
        dt_eval = dt_full[:, :k_eval]

        y0 = y[:, 0, :]
        y_true = y[:, 1 : 1 + k_eval, :]

        fn = self._compiled_open_loop_unroll if self._compiled_open_loop_unroll is not None else self._open_loop_unroll
        y_pred = fn(y0=y0, dt_bk=dt_eval, g=g)

        # Keep loss definition consistent by masking skip_steps the same way in val/test.
        step_weights = None
        if self.autoregressive_training:
            step_weights = self._step_weights(k_eval, self.ar_skip_steps, device=y_pred.device)

        losses = self.criterion(y_pred, y_true, step_weights=step_weights)
        loss_total = losses["loss_total"]

        if stage == "val":
            self.log("val_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("val_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        elif stage == "test":
            self.log("test_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("test_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        else:
            raise ValueError(f"Unknown eval stage: {stage}")

        return loss_total

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.autoregressive_training:
            return self._training_step_autoregressive(batch, batch_idx)
        return self._training_step_one_jump(batch)

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, stage="val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        """Step ReduceLROnPlateau schedulers when using manual optimization."""
        if not self.autoregressive_training:
            return
        if not self.sched_cfg or not bool(self.sched_cfg.get("enabled", False)):
            return

        metrics = getattr(self.trainer, "callback_metrics", {}) or {}
        monitor = str(self.sched_cfg.get("monitor", "val_loss"))
        metric = metrics.get(monitor, None)
        if metric is None:
            return

        metric_val = float(metric.detach().cpu().item()) if isinstance(metric, torch.Tensor) else float(metric)

        sched = self.lr_schedulers()
        if sched is None:
            return

        sched_list = list(sched) if isinstance(sched, (list, tuple)) else [sched]
        for s in sched_list:
            sch = s.scheduler if hasattr(s, "scheduler") else s
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(metric_val)


    # ------------------------
    # Optimizer / scheduler
    # ------------------------

    def configure_optimizers(self) -> Any:
        tcfg = dict(self.hparams["training"])
        opt_cfg = dict(tcfg["optimizer"])

        name = str(opt_cfg["name"]).lower()
        lr = float(opt_cfg["lr"])
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        exclude_nd = bool(
            opt_cfg.get(
                "exclude_norm_and_bias_from_weight_decay",
                EXCLUDE_NORM_AND_BIAS_FROM_WEIGHT_DECAY_BY_DEFAULT,
            )
        )
        param_groups = _build_optimizer_param_groups(self, weight_decay=weight_decay, exclude_norm_and_bias=exclude_nd)

        if name not in ("adam", "adamw"):
            raise ValueError(f"Unsupported optimizer '{name}' (supported: adam, adamw)")

        betas = opt_cfg.get("betas", (0.9, 0.999))
        b0, b1 = float(betas[0]), float(betas[1])
        eps = float(opt_cfg.get("eps", 1e-8))

        use_fused = bool(opt_cfg.get("fused", True))
        use_foreach = bool(opt_cfg.get("foreach", True))

        # Weight decay is handled by param groups, so pass 0 here.
        opt_wd_default = 0.0

        if name == "adam":
            if use_fused:
                try:
                    opt = torch.optim.Adam(
                        param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, fused=True
                    )
                except TypeError:
                    opt = torch.optim.Adam(
                        param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, foreach=use_foreach
                    )
            else:
                opt = torch.optim.Adam(
                    param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, foreach=use_foreach
                )
        else:  # adamw
            if use_fused:
                try:
                    opt = torch.optim.AdamW(
                        param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, fused=True
                    )
                except TypeError:
                    opt = torch.optim.AdamW(
                        param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, foreach=use_foreach
                    )
            else:
                opt = torch.optim.AdamW(
                    param_groups, lr=lr, betas=(b0, b1), eps=eps, weight_decay=opt_wd_default, foreach=use_foreach
                )

        sched_cfg = dict(tcfg.get("scheduler", {}) or {})
        if not bool(sched_cfg.get("enabled", False)):
            return opt

        sched_type = str(sched_cfg.get("type", "plateau")).lower().strip()

        if sched_type in ("plateau", "reduce_on_plateau", "reduceonplateau"):
            factor = float(sched_cfg.get("factor", 0.5))
            patience = int(sched_cfg.get("patience", 10))
            threshold = float(sched_cfg.get("threshold", 1e-4))
            min_lr = float(sched_cfg.get("min_lr", 1e-7))
            mode = str(sched_cfg.get("mode", "min"))
            monitor = str(sched_cfg.get("monitor", "val_loss"))

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr
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

        if sched_type not in ("cosine_with_warmup", "cosine-warmup", "cosine"):
            raise ValueError(f"Unsupported scheduler type '{sched_type}' (supported: plateau, cosine_with_warmup)")

        # Cosine schedule with optional warmup (step-based).
        if self.trainer is None:
            raise RuntimeError("Trainer must be attached before configure_optimizers for cosine scheduler.")
        max_steps = int(self.trainer.estimated_stepping_batches)
        if max_steps <= 0:
            raise RuntimeError(f"estimated_stepping_batches={max_steps}; cannot build cosine schedule.")

        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))

        warmup_steps = 0
        if warmup_epochs > 0:
            if self.trainer is None:
                raise RuntimeError("Trainer must be attached before configure_optimizers when warmup_epochs > 0.")
            steps_per_epoch = int(self.trainer.estimated_stepping_batches) // max(1, int(self.max_epochs))
            warmup_steps = int(warmup_epochs) * max(1, steps_per_epoch)


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
# Trainer factory
# ==============================================================================


def build_lightning_trainer(
    cfg: Mapping[str, Any],
    *,
    work_dir: Path,
    precision_config: Optional[PrecisionConfig] = None,
) -> pl.Trainer:
    tcfg = cfg["training"]
    runtime = dict(cfg.get("runtime", {}) or {})

    max_epochs = int(tcfg["max_epochs"])
    devices = runtime.get("devices", "auto")
    accelerator = runtime.get("accelerator", "auto")
    prec = precision_config if precision_config is not None else parse_precision_config(cfg)
    lightning_precision = prec.lightning_precision
    log_every_n_steps = int(runtime.get("log_every_n_steps", 50))

    ckpt_cfg = dict(runtime.get("checkpointing", {}) or {})
    enable_ckpt = bool(ckpt_cfg.get("enabled", True))
    ckpt_every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1))
    save_top_k = int(ckpt_cfg.get("save_top_k", 1))
    save_last = bool(ckpt_cfg.get("save_last", True))
    monitor = str(ckpt_cfg.get("monitor", "val_loss"))

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # CSVLogger: ensure log_dir resolves to work_dir (avoid versioned subdirs).
    # Using name="." is a robust way to keep outputs at save_dir across Lightning variants.
    csv_logger = CSVLogger(save_dir=str(work_dir), name=".", version="")

    # Warn if Lightning still chooses a subdirectory (best-effort; does not crash training).
    try:
        log_dir = Path(csv_logger.log_dir).resolve()
        if log_dir != work_dir.resolve():
            rank_zero_warn(f"CSVLogger log_dir is {log_dir}, not {work_dir}. metrics.csv will be written under log_dir.")
    except Exception:
        pass

    callbacks: List[pl.Callback] = [LearningRateMonitor(logging_interval="epoch")]

    if enable_ckpt:
        ckpt_dir = work_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="epoch{epoch:03d}-val{val_loss:.6f}",
                monitor=monitor,
                save_top_k=save_top_k,
                mode="min",
                every_n_epochs=ckpt_every_n_epochs,
                save_last=save_last,
            )
        )

    grad_clip = runtime.get("gradient_clip_val", None)
    gradient_clip_val = float(grad_clip) if grad_clip is not None else None

    strategy = runtime.get("strategy", "auto")
    accumulate_grad_batches = int(runtime.get("accumulate_grad_batches", 1))

    trainer_kwargs = dict(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=lightning_precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=bool(runtime.get("enable_progress_bar", False)),
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_checkpointing=bool(enable_ckpt),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=bool(runtime.get("deterministic", False)),
    )

    sig = inspect.signature(pl.Trainer)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sig.parameters}
    return pl.Trainer(**trainer_kwargs)
