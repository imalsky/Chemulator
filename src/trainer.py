#!/usr/bin/env python3
"""trainer.py

Pure-PyTorch training loop.

The original project used PyTorch Lightning. This codebase now avoids that
external dependency and implements a small, explicit training loop.

Design constraints requested for this codebase:
  - Fail fast with simple error messages.
  - Precision policy is centralized in cfg["precision"].
  - No silent fallbacks.
"""

from __future__ import annotations

import json
import math
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from utils import PrecisionPolicy
from normalizer import NormalizationHelper
from model import _z_to_log10_phys as _shared_z_to_log10_phys

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve config paths relative to repository root when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

_NORM_CODE_STANDARD = 0
_NORM_CODE_MIN_MAX = 1
_NORM_CODE_LOG_STANDARD = 2
_NORM_CODE_LOG_MIN_MAX = 3

_NORM_METHOD_TO_CODE = {
    "standard": _NORM_CODE_STANDARD,
    "min-max": _NORM_CODE_MIN_MAX,
    "log-standard": _NORM_CODE_LOG_STANDARD,
    "log-min-max": _NORM_CODE_LOG_MIN_MAX,
}


class AdaptiveStiffLoss(nn.Module):
    """Loss = lambda_phys * weighted_MAE(log10) + lambda_z * MSE(z).

    Supports arbitrary per-species normalization methods.
    Uses a direct z -> log10(physical) path to avoid extra denorm/log work.
    Hard-errors on non-positive physical values (spec requirement).
    """

    def __init__(
        self,
        *,
        norm_helper: NormalizationHelper,
        species_keys: list[str],
        species_log_min: Optional[torch.Tensor],
        species_log_max: Optional[torch.Tensor],
        lambda_phys: float,
        lambda_z: float,
        use_weighting: bool,
        weight_power: float,
        w_min: float,
        w_max: float,
    ) -> None:
        super().__init__()

        self.norm_helper = norm_helper
        self.species_keys = list(species_keys)

        if use_weighting:
            w_min_f = float(w_min)
            w_max_f = float(w_max)
            if w_min_f <= 0.0 or w_max_f <= 0.0 or w_min_f > w_max_f:
                raise ValueError("Invalid weighting bounds")

            if species_log_min is None or species_log_max is None:
                raise ValueError("species_log_min/species_log_max are required when use_weighting=true")

            weight_log_min = species_log_min.detach().clone()
            weight_log_max = species_log_max.detach().clone()
            self.register_buffer("weight_log_min", weight_log_min, persistent=False)
            self.register_buffer("weight_log_max", weight_log_max, persistent=False)

            rng = self.weight_log_max - self.weight_log_min
            if torch.any(rng <= 0):
                raise ValueError("Invalid species log range")

            w = torch.pow(rng, float(weight_power))
            mean_w = w.mean()
            if not torch.isfinite(mean_w) or float(mean_w.item()) <= 0.0:
                raise ValueError("Invalid weighting")

            w = w / mean_w
            if torch.any(w < w_min_f) or torch.any(w > w_max_f):
                lo = float(w.min())
                hi = float(w.max())
                raise ValueError(
                    f"Computed species weights out of bounds [{w_min_f:.6g}, {w_max_f:.6g}]: "
                    f"min={lo:.6g}, max={hi:.6g}"
                )
        else:
            # Keep attribute names stable even when weighting is off.
            self.register_buffer("weight_log_min", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("weight_log_max", torch.empty(0, dtype=torch.float32), persistent=False)
            w = torch.ones(len(species_keys), dtype=torch.float32)

        self.register_buffer("w_species", w)
        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.min_std = float(self.norm_helper.min_std)

        # Pre-resolve per-species conversion metadata used by z->log10(phys).
        method_code: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        mins: list[float] = []
        maxs: list[float] = []
        log_means: list[float] = []
        log_stds: list[float] = []
        log_mins: list[float] = []
        log_maxs: list[float] = []

        for key in self.species_keys:
            method_name = str(self.norm_helper.methods.get(key, ""))
            if method_name not in _NORM_METHOD_TO_CODE:
                raise ValueError(f"Unsupported normalization method for species '{key}': {method_name!r}")
            stats = self.norm_helper.per_key_stats.get(key)
            if stats is None:
                raise KeyError(f"Missing normalization stats for key: {key}")

            code = _NORM_METHOD_TO_CODE[method_name]
            method_code.append(code)

            if code == _NORM_CODE_STANDARD:
                for req_key in ("mean", "std"):
                    if req_key not in stats:
                        raise KeyError(f"Missing normalization stat '{req_key}' for key: {key}")
            elif code == _NORM_CODE_MIN_MAX:
                for req_key in ("min", "max"):
                    if req_key not in stats:
                        raise KeyError(f"Missing normalization stat '{req_key}' for key: {key}")
            elif code == _NORM_CODE_LOG_STANDARD:
                for req_key in ("log_mean", "log_std"):
                    if req_key not in stats:
                        raise KeyError(f"Missing normalization stat '{req_key}' for key: {key}")
            elif code == _NORM_CODE_LOG_MIN_MAX:
                for req_key in ("log_min", "log_max"):
                    if req_key not in stats:
                        raise KeyError(f"Missing normalization stat '{req_key}' for key: {key}")
            else:
                raise ValueError("Unexpected normalization method code")

            means.append(float(stats.get("mean", 0.0)))
            stds.append(float(stats.get("std", 1.0)))
            mins.append(float(stats.get("min", 0.0)))
            maxs.append(float(stats.get("max", 1.0)))
            log_means.append(float(stats.get("log_mean", 0.0)))
            log_stds.append(float(stats.get("log_std", 1.0)))
            log_mins.append(float(stats.get("log_min", 0.0)))
            log_maxs.append(float(stats.get("log_max", 1.0)))

        self.register_buffer("method_code", torch.tensor(method_code, dtype=torch.long), persistent=False)
        self.register_buffer("mean", torch.tensor(means, dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.tensor(stds, dtype=torch.float32), persistent=False)
        self.register_buffer("vmin", torch.tensor(mins, dtype=torch.float32), persistent=False)
        self.register_buffer("vmax", torch.tensor(maxs, dtype=torch.float32), persistent=False)
        self.register_buffer("log_mean", torch.tensor(log_means, dtype=torch.float32), persistent=False)
        self.register_buffer("log_std", torch.tensor(log_stds, dtype=torch.float32), persistent=False)
        self.register_buffer("stat_log_min", torch.tensor(log_mins, dtype=torch.float32), persistent=False)
        self.register_buffer("stat_log_max", torch.tensor(log_maxs, dtype=torch.float32), persistent=False)
        self.register_buffer("m_standard", self.method_code == _NORM_CODE_STANDARD, persistent=False)
        self.register_buffer("m_minmax", self.method_code == _NORM_CODE_MIN_MAX, persistent=False)
        self.register_buffer("m_log_standard", self.method_code == _NORM_CODE_LOG_STANDARD, persistent=False)
        self.register_buffer("m_log_minmax", self.method_code == _NORM_CODE_LOG_MIN_MAX, persistent=False)

        # Runtime normalization safety checks (spec requirement).
        if bool(torch.any(self.m_standard & (self.std <= 0.0))):
            raise ValueError("Invalid standard normalization stats: std must be > 0")
        if bool(torch.any(self.m_standard & (self.std < self.min_std))):
            raise ValueError("std below min_std in loss normalization path")
        if bool(torch.any(self.m_minmax & ((self.vmax - self.vmin) <= 0.0))):
            raise ValueError("Invalid min-max normalization stats: max must be > min")
        if bool(torch.any(self.m_log_standard & (self.log_std <= 0.0))):
            raise ValueError("Invalid log-standard normalization stats: log_std must be > 0")
        if bool(torch.any(self.m_log_standard & (self.log_std < self.min_std))):
            raise ValueError("log_std below min_std in loss normalization path")
        if bool(torch.any(self.m_log_minmax & ((self.stat_log_max - self.stat_log_min) <= 0.0))):
            raise ValueError("Invalid log-min-max normalization stats: log_max must be > log_min")

        # Cache the reciprocal of numel per forward to avoid per-batch tensor creation.
        self._cached_numel: int = 0
        self._cached_inv_denom: float = 0.0

    def _z_to_log10_phys(self, z: torch.Tensor) -> torch.Tensor:
        """Convert z-space species [..., S] directly to log10(physical) [..., S]."""
        z32 = z.to(torch.float32)
        return _shared_z_to_log10_phys(
            z32,
            method_code=self.method_code,
            mean=self.mean,
            std=self.std,
            vmin=self.vmin,
            vmax=self.vmax,
            log_mean=self.log_mean,
            log_std=self.log_std,
            log_min=self.stat_log_min,
            log_max=self.stat_log_max,
        )

    def forward(
        self,
        pred_z: torch.Tensor,  # [B,K,S]
        true_z: torch.Tensor,  # [B,K,S]
        *,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if pred_z.shape != true_z.shape:
            raise ValueError("Pred/GT shape mismatch")

        # Promote reductions to FP32 for stability under AMP.
        diff_z = (pred_z - true_z).to(torch.float32)
        mse_z = diff_z * diff_z

        pred_log10 = self._z_to_log10_phys(pred_z)
        true_log10 = self._z_to_log10_phys(true_z)
        abs_log = (pred_log10 - true_log10).abs()

        w = self.w_species.to(dtype=abs_log.dtype)
        weighted_abs_log = abs_log * w

        numel = abs_log.numel()
        if numel != self._cached_numel:
            self._cached_numel = numel
            self._cached_inv_denom = 1.0 / float(numel)
        inv_denom = self._cached_inv_denom

        mean_abs_log10 = abs_log.sum() * inv_denom
        weighted_mean_abs_log10 = weighted_abs_log.sum() * inv_denom
        mean_mse_z = mse_z.sum() * inv_denom

        phys = self.lambda_phys * weighted_mean_abs_log10
        zterm = self.lambda_z * mean_mse_z
        total = phys + zterm

        if return_components:
            return {
                "total": total,
                "phys": phys,
                "z": zterm,
                "mean_abs_log10": mean_abs_log10,
                "weighted_mean_abs_log10": weighted_mean_abs_log10,
            }
        return total


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


@dataclass
class _EpochMeters:
    loss_sum: float = 0.0
    phys_sum: float = 0.0
    z_sum: float = 0.0
    abslog_sum: float = 0.0
    weight_sum: float = 0.0

    def update(self, comps: Dict[str, torch.Tensor], weight: int) -> None:
        w = float(weight)
        self.loss_sum += float(comps["total"].item()) * w
        self.phys_sum += float(comps["phys"].item()) * w
        self.z_sum += float(comps["z"].item()) * w
        self.abslog_sum += float(comps["mean_abs_log10"].item()) * w
        self.weight_sum += w

    def means(self) -> Dict[str, float]:
        if self.weight_sum <= 0.0:
            raise ValueError("Empty epoch")
        return {
            "loss": self.loss_sum / self.weight_sum,
            "phys": self.phys_sum / self.weight_sum,
            "z": self.z_sum / self.weight_sum,
            "mean_abs_log10": self.abslog_sum / self.weight_sum,
        }


def _mult_err_proxy(mean_abs_log10: float) -> float:
    # 10**x - 1, computed stably as expm1(ln(10) * x)
    return float(math.expm1(math.log(10.0) * float(mean_abs_log10)))


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    map_location: torch.device,
) -> Dict[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(path, **load_kwargs)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore[arg-type]
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])  # type: ignore[arg-type]
    return ckpt


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class Trainer:
    """A small, explicit trainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        work_dir: Path,
        device: torch.device,
        logger,
        precision_policy: PrecisionPolicy,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.log = logger
        self.policy = precision_policy

        tcfg = cfg["training"]

        self.epochs = int(tcfg["epochs"])
        self.lr = float(tcfg["lr"])
        self.min_lr = float(tcfg["min_lr"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.warmup_epochs = int(tcfg["warmup_epochs"])
        self.warmup_start_factor = float(tcfg["warmup_start_factor"])

        self.gradient_clip = float(tcfg["gradient_clip"])

        if "max_train_steps_per_epoch" not in tcfg:
            raise KeyError("Missing config: training.max_train_steps_per_epoch")
        if "max_val_batches" not in tcfg:
            raise KeyError("Missing config: training.max_val_batches")
        self.max_train_steps_per_epoch = int(tcfg["max_train_steps_per_epoch"])
        self.max_val_batches = int(tcfg["max_val_batches"])
        if self.max_train_steps_per_epoch < 0:
            raise ValueError("training.max_train_steps_per_epoch must be >= 0")
        if self.max_val_batches < 0:
            raise ValueError("training.max_val_batches must be >= 0")
        self.save_every_n_epochs = int(tcfg.get("save_every_n_epochs", 1))
        if self.save_every_n_epochs < 1:
            self.save_every_n_epochs = 1

        opt_name = str(tcfg["optimizer"]).lower().strip()
        if opt_name != "adamw":
            raise ValueError("Unsupported optimizer")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = self._build_scheduler()

        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.policy.use_amp and self.device.type == "cuda" and self.policy.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()

        self.criterion = self._build_loss().to(self.device)

        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0

    # ------------------------------ loss ---------------------------------

    def _build_loss(self) -> nn.Module:
        processed_dir = _resolve_repo_path(self.cfg["paths"]["processed_data_dir"])  # validated in main
        manifest_path = processed_dir / "normalization.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        species = list(self.cfg.get("data", {}).get("species_variables") or [])
        if not species:
            raise ValueError("species_variables is empty")

        norm_helper = NormalizationHelper(manifest)
        loss_cfg = self.cfg["training"]["adaptive_stiff_loss"]

        use_weighting = bool(loss_cfg.get("use_weighting", False))
        species_log_min_tensor: Optional[torch.Tensor] = None
        species_log_max_tensor: Optional[torch.Tensor] = None
        if use_weighting:
            stats = manifest["per_key_stats"]
            log_mins: list[float] = []
            log_maxs: list[float] = []
            for s in species:
                row = stats[s]
                for k in ("log_min", "log_max"):
                    if k not in row:
                        raise KeyError(
                            f"Species '{s}' stats missing {k} (needed for weighting). "
                            "Regenerate preprocessing with training.adaptive_stiff_loss.use_weighting=true "
                            "or disable weighting."
                        )
                log_mins.append(float(row["log_min"]))
                log_maxs.append(float(row["log_max"]))

            species_log_min_tensor = torch.tensor(log_mins, dtype=torch.float32, device=self.device)
            species_log_max_tensor = torch.tensor(log_maxs, dtype=torch.float32, device=self.device)

        return AdaptiveStiffLoss(
            norm_helper=norm_helper,
            species_keys=species,
            species_log_min=species_log_min_tensor,
            species_log_max=species_log_max_tensor,
            lambda_phys=float(loss_cfg["lambda_phys"]),
            lambda_z=float(loss_cfg["lambda_z"]),
            use_weighting=use_weighting,
            weight_power=float(loss_cfg.get("weight_power", 0.5)),
            w_min=float(loss_cfg.get("w_min", 0.5)),
            w_max=float(loss_cfg.get("w_max", 2.0)),
        )

    # ------------------------------ scheduler ----------------------------

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")

        if self.warmup_epochs == 0:
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)

        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be < epochs")

        warm = LinearLR(
            self.optimizer,
            start_factor=self.warmup_start_factor,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.min_lr,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warm, cosine],
            milestones=[self.warmup_epochs],
        )

    # ------------------------------ resume -------------------------------

    def _resume_path(self) -> Optional[Path]:
        tcfg = self.cfg["training"]
        if "resume" not in tcfg:
            raise KeyError("Missing config: training.resume")
        resume = tcfg["resume"]
        if isinstance(resume, str) and resume.strip():
            p = Path(resume)
            if not p.is_file():
                raise FileNotFoundError("Resume checkpoint not found")
            return p

        return None

    def _maybe_resume(self) -> None:
        p = self._resume_path()
        if p is None:
            return

        ckpt = _load_checkpoint(
            p,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.log.info("Resumed from %s (epoch=%d)", str(p), self.start_epoch)

    # ------------------------------ train loop ---------------------------

    def _autocast_ctx(self):
        if not self.policy.use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)

        # CUDA autocast supports fp16 and bf16.
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.policy.amp_dtype)

        # CPU autocast only supports bf16 reliably.
        if self.device.type == "cpu" and self.policy.amp_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

        return torch.autocast(device_type=self.device.type, enabled=False)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=(self.device.type == "cuda"))

    def train(self) -> float:
        self.model.to(self.device)
        self.model.train()

        self._maybe_resume()

        metrics_path = self.work_dir / "metrics.jsonl"

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            train_m = _EpochMeters()

            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(self.train_loader):
                if self.max_train_steps_per_epoch and step >= self.max_train_steps_per_epoch:
                    break

                y_i, dt_norm, y_j, g = batch
                y_i = self._to_device(y_i)
                dt_norm = self._to_device(dt_norm)
                y_j = self._to_device(y_j)
                g = self._to_device(g)

                with self._autocast_ctx():
                    pred = self.model(y_i, dt_norm, g)
                    comps = self.criterion(pred, y_j, return_components=True)  # type: ignore[arg-type]
                    loss = comps["total"]

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite train loss")

                # Metrics weight = number of scalar elements
                B, K, S = y_j.shape
                train_m.update(comps, weight=B * K * S)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                if self.gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            train_means = train_m.means()
            train_mult = _mult_err_proxy(train_means["mean_abs_log10"])

            # Validation
            val_means: Optional[Dict[str, float]] = None
            val_mult: Optional[float] = None
            val_loss = float("nan")

            if self.val_loader is not None:
                self.model.eval()
                val_m = _EpochMeters()
                with torch.no_grad():
                    for bidx, batch in enumerate(self.val_loader):
                        if self.max_val_batches and bidx >= self.max_val_batches:
                            break

                        y_i, dt_norm, y_j, g = batch
                        y_i = self._to_device(y_i)
                        dt_norm = self._to_device(dt_norm)
                        y_j = self._to_device(y_j)
                        g = self._to_device(g)

                        with self._autocast_ctx():
                            pred = self.model(y_i, dt_norm, g)
                            comps = self.criterion(pred, y_j, return_components=True)  # type: ignore[arg-type]
                            loss = comps["total"]

                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite val loss")

                        B, K, S = y_j.shape
                        val_m.update(comps, weight=B * K * S)

                val_means = val_m.means()
                val_mult = _mult_err_proxy(val_means["mean_abs_log10"])
                val_loss = float(val_means["loss"])

                # Track best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    _save_checkpoint(
                        self.work_dir / "best.ckpt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        best_val_loss=self.best_val_loss,
                        cfg=self.cfg,
                    )

            # Always persist last.ckpt every epoch so explicit resume never depends on cadence.
            _save_checkpoint(
                self.work_dir / "last.ckpt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_val_loss=self.best_val_loss,
                cfg=self.cfg,
            )
            if self.save_every_n_epochs > 1 and (epoch + 1) % self.save_every_n_epochs == 0:
                _save_checkpoint(
                    self.work_dir / f"epoch_{epoch + 1:04d}.ckpt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    cfg=self.cfg,
                )

            # Log to console
            lr_now = float(self.optimizer.param_groups[0]["lr"])
            if val_means is None:
                self.log.info(
                    "Epoch %d/%d lr=%.3e train_loss=%.6g train_phys=%.6g train_z=%.6g train_mult=%.6g",
                    epoch + 1,
                    self.epochs,
                    lr_now,
                    train_means["loss"],
                    train_means["phys"],
                    train_means["z"],
                    train_mult,
                )
            else:
                self.log.info(
                    "Epoch %d/%d lr=%.3e train_loss=%.6g val_loss=%.6g train_mult=%.6g val_mult=%.6g",
                    epoch + 1,
                    self.epochs,
                    lr_now,
                    train_means["loss"],
                    val_loss,
                    train_mult,
                    float(val_mult),
                )

            # Append metrics JSONL
            record = {
                "epoch": int(epoch),
                "lr": lr_now,
                "train": {
                    "loss": train_means["loss"],
                    "phys": train_means["phys"],
                    "z": train_means["z"],
                    "mult_err_proxy": train_mult,
                },
                "val": None if val_means is None else {
                    "loss": val_means["loss"],
                    "phys": val_means["phys"],
                    "z": val_means["z"],
                    "mult_err_proxy": float(val_mult),
                },
                "best_val_loss": float(self.best_val_loss),
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if self.val_loader is None:
            return float("nan")
        return float(self.best_val_loss)
