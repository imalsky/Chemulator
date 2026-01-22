#!/usr/bin/env python3
"""trainer.py

Autoregressive training with:
  - Teacher forcing schedule (can increase or decrease)
  - Optional burn-in (free-run) with optional additive noise
  - Optional rollout curriculum
  - Optional cosine warmup LR scheduler

This implementation is intentionally small and robust across Lightning variants.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer pytorch_lightning (common in user envs). Fallback to lightning.pytorch if needed.
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
except ModuleNotFoundError:  # pragma: no cover
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar


# ---------------------------- Terminal logging ----------------------------

class TerminalMetricsCallback(pl.Callback):
    def __init__(self, *, every_n_steps: int = 50, log_on_step: bool = True, log_on_epoch: bool = True) -> None:
        super().__init__()
        self.every_n_steps = max(int(every_n_steps), 1)
        self.log_on_step = bool(log_on_step)
        self.log_on_epoch = bool(log_on_epoch)

    @staticmethod
    def _to_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if hasattr(x, "detach"):
                return float(x.detach().cpu().item())
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _get_lr(trainer: "pl.Trainer") -> Optional[float]:
        try:
            if not trainer.optimizers:
                return None
            opt = trainer.optimizers[0]
            if not opt.param_groups:
                return None
            return float(opt.param_groups[0].get("lr", None))
        except Exception:
            return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.log_on_step or not getattr(trainer, "is_global_zero", True):
            return
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or (step % self.every_n_steps) != 0:
            return

        loss_val = self._to_float(outputs)
        lr_val = self._get_lr(trainer)

        pbm = getattr(trainer, "progress_bar_metrics", {}) or {}
        extras = []
        for k in ("train_loss", "val_loss", "tf_prob", "k_roll"):
            if k in pbm:
                v = self._to_float(pbm.get(k))
                if v is not None:
                    extras.append(f"{k}={v:.3e}")

        msg = f"[step {step}]"
        if loss_val is not None:
            msg += f" loss={loss_val:.3e}"
        if lr_val is not None:
            msg += f" lr={lr_val:.3e}"
        if extras:
            msg += " | " + " ".join(extras)
        print(msg, flush=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self.log_on_epoch or not getattr(trainer, "is_global_zero", True):
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        pbm = getattr(trainer, "progress_bar_metrics", {}) or {}

        def get(name: str) -> Optional[float]:
            return self._to_float(pbm.get(name, None))

        parts = [f"[epoch {epoch}]"]
        for k in ("train_loss", "val_loss"):
            v = get(k)
            if v is not None:
                parts.append(f"{k}={v:.3e}")
        lr_val = self._get_lr(trainer)
        if lr_val is not None:
            parts.append(f"lr={lr_val:.3e}")
        tf_prob = get("tf_prob")
        if tf_prob is not None:
            parts.append(f"tf={tf_prob:.3f}")
        k_roll = get("k_roll")
        if k_roll is not None:
            parts.append(f"k={int(round(k_roll))}")
        print(" ".join(parts), flush=True)


class CSVTrainingLoggerCallback(pl.Callback):
    """Write a single CSV training log (training.csv) into work_dir.

    Writes one row per epoch (after validation) with whatever is available in trainer.callback_metrics.
    """

    def __init__(self, *, work_dir: Path) -> None:
        super().__init__()
        self.work_dir = Path(work_dir)
        self.path = self.work_dir / "training.csv"
        self._header_written = False
        self._fieldnames: List[str] = []

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if hasattr(v, "detach"):
                return float(v.detach().cpu().item())
            return float(v)
        except Exception:
            return None

    def on_fit_start(self, trainer, pl_module) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
        row: Dict[str, Any] = {}
        for k, v in metrics.items():
            row[str(k)] = self._to_float(v)
        row["epoch"] = int(getattr(trainer, "current_epoch", 0))
        row["global_step"] = int(getattr(trainer, "global_step", 0))

        if not self._header_written:
            keys = ["epoch", "global_step"] + sorted([k for k in row.keys() if k not in ("epoch", "global_step")])
            self._fieldnames = keys
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                w.writeheader()
                w.writerow({k: row.get(k, None) for k in self._fieldnames})
            self._header_written = True
        else:
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                w.writerow({k: row.get(k, None) for k in self._fieldnames})


# ---------------------------- Loss ----------------------------

def _stack_log_stats(
    normalization_manifest: Dict[str, Any],
    species_variables: List[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = normalization_manifest.get("per_key_stats", normalization_manifest.get("stats", {}))
    if not isinstance(stats, dict):
        raise ValueError("normalization.json missing per_key_stats/stats")

    means, stds, mins, maxs = [], [], [], []
    for name in species_variables:
        if name not in stats:
            raise KeyError(f"Missing stats for species '{name}' in normalization.json")
        s = stats[name]
        means.append(float(s.get("log_mean", 0.0)))
        stds.append(float(s.get("log_std", 1.0)))
        mins.append(float(s.get("log_min", -30.0)))
        maxs.append(float(s.get("log_max", 30.0)))

    log_means = torch.tensor(means, device=device, dtype=dtype).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device, dtype=dtype).view(1, 1, -1)
    log_mins = torch.tensor(mins, device=device, dtype=dtype).view(1, 1, -1)
    log_maxs = torch.tensor(maxs, device=device, dtype=dtype).view(1, 1, -1)
    return log_means, log_stds, log_mins, log_maxs


class AdaptiveStiffLoss(nn.Module):
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
        self.register_buffer("log_means", log_means)
        self.register_buffer("log_stds", log_stds)
        self.register_buffer("log_mins", log_mins)
        self.register_buffer("log_maxs", log_maxs)

        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.eps_phys = float(eps_phys)
        self.w_species_clamp = (float(w_species_clamp[0]), float(w_species_clamp[1]))
        self.range_eps = float(range_eps)

    def forward(self, pred_z: torch.Tensor, true_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mse = F.mse_loss(pred_z, true_z)

        pred_log = pred_z * self.log_stds + self.log_means
        true_log = true_z * self.log_stds + self.log_means
        pred_log = torch.clamp(pred_log, self.log_mins, self.log_maxs)
        true_log = torch.clamp(true_log, self.log_mins, self.log_maxs)

        ranges = torch.clamp(self.log_maxs - self.log_mins, min=self.range_eps)
        w = 1.0 / ranges
        w = torch.clamp(w, self.w_species_clamp[0], self.w_species_clamp[1])
        phys_mse = torch.mean(w * (pred_log - true_log) ** 2)

        loss_total = self.lambda_phys * phys_mse + self.lambda_z * z_mse
        mean_abs_log10 = torch.mean(torch.abs(pred_log - true_log))

        return {
            "loss_total": loss_total,
            "loss_phys": phys_mse,
            "loss_z": z_mse,
            "mean_abs_log10": mean_abs_log10,
        }


# ---------------------------- Schedules ----------------------------

@dataclass(frozen=True)
class TeacherForcingSchedule:
    start: float = 1.0
    end: float = 0.0
    decay_epochs: int = 0
    mode: str = "linear"  # linear | cosine

    def value(self, epoch: int) -> float:
        if self.decay_epochs <= 0:
            return float(self.start)
        t = min(max(epoch, 0), self.decay_epochs)
        u = t / float(self.decay_epochs)
        if self.mode == "cosine":
            u = 0.5 * (1.0 - math.cos(math.pi * u))
        return float(self.start + (self.end - self.start) * u)


@dataclass(frozen=True)
class RolloutCurriculum:
    enabled: bool = False
    start_steps: int = 1
    ramp_epochs: int = 0

    def steps(self, epoch: int, max_steps: int) -> int:
        if not self.enabled:
            return max_steps
        if self.ramp_epochs <= 0:
            return max_steps
        t = min(max(epoch, 0), self.ramp_epochs)
        u = t / float(self.ramp_epochs)
        k = int(round(self.start_steps + (max_steps - self.start_steps) * u))
        return max(1, min(max_steps, k))


# ---------------------------- Lightning module ----------------------------

class FlowMapRolloutModule(pl.LightningModule):
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        model: nn.Module,
        normalization_manifest: Dict[str, Any],
        species_variables: List[str],
        work_dir: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.work_dir = Path(work_dir)

        tcfg = cfg.get("training", {})

        self.rollout_steps = int(tcfg.get("rollout_steps", 1))
        self.burn_in_steps = int(tcfg.get("burn_in_steps", 0))
        self.total_steps = self.rollout_steps + self.burn_in_steps

        self.burn_in_add_noise_std = float(tcfg.get("burn_in_additive_noise_std", 0.0))
        self.burn_in_loss_weight = float(tcfg.get("burn_in_loss_weight", 0.0))

        # Validation burn-in (default: match training)
        self.val_burn_in_steps = int(tcfg.get("val_burn_in_steps", self.burn_in_steps))

        tf_cfg = tcfg.get("teacher_forcing", {})
        self.tf_sched = TeacherForcingSchedule(
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

        lcfg = tcfg.get("loss", {})
        log_means, log_stds, log_mins, log_maxs = _stack_log_stats(
            normalization_manifest,
            species_variables,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
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

        self.lr = float(tcfg.get("lr", 1e-3))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))

        # Scheduler config
        self.sched_cfg = dict(tcfg.get("scheduler", {}) or {})

        # Dataloaders (for PL versions/environments that require module methods)
        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

    def set_dataloaders(self, train_dl, val_dl=None, test_dl=None) -> None:
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._test_dl = test_dl

    def train_dataloader(self):
        if self._train_dl is None:
            raise RuntimeError("train_dataloader requested but not set; call lit_module.set_dataloaders(...)")
        return self._train_dl

    def val_dataloader(self):
        if self._val_dl is None:
            raise RuntimeError("val_dataloader requested but not set; call lit_module.set_dataloaders(...)")
        return self._val_dl

    def test_dataloader(self):
        if self._test_dl is None:
            raise RuntimeError("test_dataloader requested but not set; call lit_module.set_dataloaders(...)")
        return self._test_dl

    def on_fit_start(self) -> None:
        self.criterion = self.criterion.to(self.device)

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
        B, K_total, S = y_true.shape
        K_eff = min(K_total, burn_in_steps + max_rollout_steps)
        burn = min(burn_in_steps, K_eff)

        state = y0
        preds: List[torch.Tensor] = []

        if dt_norm.ndim == 0:
            dt_step = dt_norm.view(1).expand(B)
        else:
            dt_step = dt_norm
        dt_step = dt_step.to(state.device)

        for t in range(K_eff):
            out = self.model(state, dt_step.view(B, 1, 1), g)
            pred = out[:, 0, :]
            preds.append(pred)

            if t == K_eff - 1:
                break

            if t < burn:
                state = pred
                if add_noise_std > 0.0:
                    state = state + add_noise_std * torch.randn_like(state)
            else:
                if teacher_forcing_p >= 1.0:
                    state = y_true[:, t, :]
                elif teacher_forcing_p <= 0.0:
                    state = pred
                else:
                    mask = (torch.rand((B,), device=state.device) < teacher_forcing_p).view(B, 1)
                    state = torch.where(mask, y_true[:, t, :], pred)

        pred_all = torch.stack(preds, dim=1)
        true_all = y_true[:, :K_eff, :]
        return pred_all, true_all

    def training_step(self, batch, batch_idx):
        y0, dt_norm, y_seq, g = batch
        epoch = int(self.current_epoch)

        k_roll = self.curriculum.steps(epoch, self.rollout_steps)
        tf_p = self.tf_sched.value(epoch)

        # Optimization loss (uses configured tf/noise)
        pred_all_opt, true_all_opt = self._autoregressive_unroll(
            y0=y0,
            dt_norm=dt_norm,
            y_true=y_seq,
            g=g,
            teacher_forcing_p=tf_p,
            burn_in_steps=self.burn_in_steps,
            add_noise_std=self.burn_in_add_noise_std,
            max_rollout_steps=k_roll,
        )
        burn_opt = min(self.burn_in_steps, pred_all_opt.shape[1])
        main_pred_opt = pred_all_opt[:, burn_opt:, :]
        main_true_opt = true_all_opt[:, burn_opt:, :]
        comps_opt = self.criterion(main_pred_opt, main_true_opt)
        loss = comps_opt["loss_total"]
        if burn_opt > 0 and self.burn_in_loss_weight > 0.0:
            comps_burn = self.criterion(pred_all_opt[:, :burn_opt, :], true_all_opt[:, :burn_opt, :])
            loss = loss + self.burn_in_loss_weight * comps_burn["loss_total"]

        # Logged metric (match validation): pure autoregressive, no noise
        pred_all, true_all = self._autoregressive_unroll(
            y0=y0,
            dt_norm=dt_norm,
            y_true=y_seq,
            g=g,
            teacher_forcing_p=0.0,
            burn_in_steps=self.val_burn_in_steps,
            add_noise_std=0.0,
            max_rollout_steps=k_roll,
        )
        burn = min(self.val_burn_in_steps, pred_all.shape[1])
        comps = self.criterion(pred_all[:, burn:, :], true_all[:, burn:, :])

        self.log("train_loss", comps["loss_total"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mean_abs_log10", comps["mean_abs_log10"], prog_bar=True, on_step=False, on_epoch=True)

        self.log("tf_prob", torch.tensor(tf_p, device=self.device), prog_bar=True, on_step=False, on_epoch=True)
        self.log("k_roll", torch.tensor(k_roll, device=self.device), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y0, dt_norm, y_seq, g = batch
        epoch = int(self.current_epoch)
        k_roll = self.curriculum.steps(epoch, self.rollout_steps)

        pred_all, true_all = self._autoregressive_unroll(
            y0=y0,
            dt_norm=dt_norm,
            y_true=y_seq,
            g=g,
            teacher_forcing_p=0.0,
            burn_in_steps=self.val_burn_in_steps,
            add_noise_std=0.0,
            max_rollout_steps=k_roll,
        )
        burn = min(self.val_burn_in_steps, pred_all.shape[1])
        comps = self.criterion(pred_all[:, burn:, :], true_all[:, burn:, :])

        self.log("val_loss", comps["loss_total"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mean_abs_log10", comps["mean_abs_log10"], prog_bar=True, on_step=False, on_epoch=True)
        return comps["loss_total"]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        name = str(self.sched_cfg.get("name", "none")).lower().strip()
        if name in ("none", "", "null"):
            return opt

        if name not in ("cosine_warmup", "cosine"):
            raise ValueError(f"Unknown training.scheduler.name: {name!r}. Supported: 'none', 'cosine_warmup'.")

        warmup_steps = int(self.sched_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(self.sched_cfg.get("min_lr_ratio", 0.0))
        total_steps = int(self.sched_cfg.get("total_steps", 0))

        # Best-effort auto total_steps if not provided.
        if total_steps <= 0:
            try:
                if self.trainer is not None and getattr(self.trainer, "estimated_stepping_batches", None):
                    total_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                total_steps = 0

        if total_steps <= 0:
            raise ValueError(
                "training.scheduler.total_steps must be set (or be inferable). "
                "Set it explicitly in config for deterministic behavior."
            )

        total_steps = max(1, total_steps)
        warmup_steps = max(0, min(warmup_steps, total_steps))

        def lr_lambda(step: int) -> float:
            s = min(max(int(step), 0), total_steps)
            if warmup_steps > 0 and s < warmup_steps:
                return float(s) / float(max(1, warmup_steps))
            if total_steps <= warmup_steps:
                return 1.0
            progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


# ---------------------------- Trainer factory ----------------------------

def _get_precision_for_accelerator(cfg_precision: str, accelerator: str) -> str:
    precision = str(cfg_precision).lower().strip()

    if accelerator == "cpu":
        if precision in ("bf16-mixed", "16-mixed", "bf16", "16"):
            import warnings

            warnings.warn(
                f"Precision '{cfg_precision}' is not well-supported on CPU. Falling back to '32-true'.",
                UserWarning,
                stacklevel=3,
            )
            return "32-true"

    if accelerator == "mps":
        if precision in ("bf16-mixed", "bf16"):
            import warnings

            warnings.warn(
                f"Precision '{cfg_precision}' is not supported on MPS. Falling back to '32-true'. "
                "For mixed precision on MPS, consider training.precision='16-mixed'.",
                UserWarning,
                stacklevel=3,
            )
            return "32-true"

    return precision


def build_lightning_trainer(cfg: Dict[str, Any], *, work_dir: Path) -> pl.Trainer:
    tcfg = cfg.get("training", {})
    max_epochs = int(tcfg.get("max_epochs", 100))

    # Hardware
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = int(tcfg.get("devices", 1))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    precision = _get_precision_for_accelerator(str(tcfg.get("precision", "bf16-mixed")), accelerator)

    enable_progress_bar = bool(tcfg.get("enable_progress_bar", True))
    log_every_n_steps = int(tcfg.get("log_every_n_steps", 50))

    terminal_every = int(tcfg.get("terminal_log_every_n_steps", max(1, log_every_n_steps)))
    terminal_on_step = bool(tcfg.get("terminal_log_on_step", True))
    terminal_on_epoch = bool(tcfg.get("terminal_log_on_epoch", True))

    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_every = int(ckpt_cfg.get("save_every_n_epochs", 1))

    callbacks: List[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(work_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            every_n_epochs=ckpt_every,
        ),
        TerminalMetricsCallback(
            every_n_steps=terminal_every,
            log_on_step=terminal_on_step,
            log_on_epoch=terminal_on_epoch,
        ),
        CSVTrainingLoggerCallback(work_dir=work_dir),
    ]

    if enable_progress_bar:
        refresh_rate = int(tcfg.get("progress_bar_refresh_rate", 1))
        callbacks.append(TQDMProgressBar(refresh_rate=refresh_rate))

    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=bool(tcfg.get("enable_model_summary", True)),
        gradient_clip_val=float(tcfg.get("grad_clip", 0.0)),
        accumulate_grad_batches=int(tcfg.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
    )
    return trainer
