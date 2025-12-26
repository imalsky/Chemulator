#!/usr/bin/env python3
"""
trainer.py

- CSV-only logging; checkpoints: best.ckpt / last.ckpt in work_dir
- AMP precision from cfg (bf16/fp16/fp32); TF32/cudnn handled in main/hardware setup
- Optional torch.compile (Inductor)
- Cosine schedule with warmup
- Vectorized K handling; optional K-mask reduction
- Sample-weighted epoch reduction for train/val (matches exactly)

Dataset structure: (y_i, dt_norm[B,K,1], y_j[B,K,S], g[B,G], aux{...}[, k_mask[B,K]])

NOTE on "fractional error":
- This file logs/prints an *approximate* fractional error derived from the epoch-mean |Δlog10| metric:
    frac_err_proxy = 10**(mean_abs_log10_error) - 1
- It does NOT subsample 100k examples; it uses the already-aggregated epoch metric (val_phys/train_phys).
"""

from __future__ import annotations

# ============================== CONSTANTS =====================================

# Checkpointing / logging
CKPT_FILENAME_BEST: str = "best"
CKPT_MONITOR: str = "val_loss"
CKPT_MODE: str = "min"
LOG_EVERY_N_STEPS: int = 200
ENABLE_CHECKPOINTING: bool = True
ENABLE_PROGRESS_BAR: bool = False
ENABLE_MODEL_SUMMARY: bool = False
DETECT_ANOMALY: bool = False
INFERENCE_MODE: bool = True
LR_LOGGING_INTERVAL: str = "epoch"

# Training defaults (used when cfg keys absent)
DEF_EPOCHS: int = 100
DEF_GRAD_CLIP: float = 0.0
DEF_ACCUMULATE: int = 1
DEF_MAX_TRAIN_BATCHES: int = 0
DEF_MAX_VAL_BATCHES: int = 0
DEF_LR: float = 1e-3
DEF_WEIGHT_DECAY: float = 1e-4
DEF_WARMUP_EPOCHS: int = 0
DEF_MIN_LR: float = 1e-6

# torch.compile defaults
COMPILE_ENABLE_DEFAULT: bool = True
COMPILE_BACKEND: str | None = "inductor"
COMPILE_MODE: str = "default"
COMPILE_DYNAMIC: bool = False
COMPILE_FULLGRAPH: bool = False

# AdaptiveStiffLoss defaults / numerics
LOSS_LAMBDA_PHYS: float = 1.0
LOSS_LAMBDA_Z: float = 0.1
LOSS_EPS_PHYS: float = 1e-20
LOG_STD_CLAMP_MIN: float = 1e-10
W_SPECIES_CLAMP_MIN: float = 0.5
W_SPECIES_CLAMP_MAX: float = 2.0
SPECIES_RANGE_EPS: float = 1e-6
W_SPECIES_MEAN_EPS: float = 1e-12
WARMUP_START_FACTOR: float = 0.1

# Resume behavior
ENV_RESUME_VAR: str = "RESUME"
RESUME_AUTO_SENTINEL: str = "auto"
LAST_CKPT_NAME: str = "last.ckpt"

# ============================== IMPORTS =======================================

import os
import json
import time
import math
import logging
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    Callback,
    StochasticWeightAveraging,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ============================== LOSS ==========================================

class AdaptiveStiffLoss(nn.Module):
    """
    Loss function: MSE in z plus MAE in log10-physical space with species weighting.

    IMPORTANT: _z_to_log10 MUST remain differentiable. Do not decorate it with no_grad.
    """

    def __init__(
        self,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        species_log_min: torch.Tensor,
        species_log_max: torch.Tensor,
        *,
        lambda_phys: float = LOSS_LAMBDA_PHYS,
        lambda_z: float = LOSS_LAMBDA_Z,
        use_weighting: bool = True,
        weight_power: float = 0.5,
        w_min: float = W_SPECIES_CLAMP_MIN,
        w_max: float = W_SPECIES_CLAMP_MAX,
        epsilon_phys: float = LOSS_EPS_PHYS,  # retained for signature compatibility (unused in loss)
    ) -> None:
        super().__init__()

        # Buffers for Lightning device moves
        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=LOG_STD_CLAMP_MIN))
        self.register_buffer("log_min", species_log_min.detach().clone())
        self.register_buffer("log_max", species_log_max.detach().clone())

        if use_weighting:
            rng = torch.clamp(self.log_max - self.log_min, min=SPECIES_RANGE_EPS)
            w = torch.pow(rng, float(weight_power))
            w = w / (w.mean() + W_SPECIES_MEAN_EPS)
            w_final = torch.clamp(w, float(w_min), float(w_max))
        else:
            w_final = torch.ones_like(self.log_means)

        self.register_buffer("w_species", w_final)

        self.lambda_phys = float(lambda_phys)
        self.lambda_z = float(lambda_z)
        self.eps_phys = float(epsilon_phys)

    def _z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        """Convert normalized z back to log10 space (DIFFERENTIABLE)."""
        return z * self.log_stds + self.log_means

    @torch.no_grad()
    def z_to_log10_nograd(self, z: torch.Tensor) -> torch.Tensor:
        """Metrics-only helper (explicitly no-grad)."""
        return self._z_to_log10(z)

    def forward(
        self,
        pred_z: torch.Tensor,                 # [B,K,S]
        true_z: torch.Tensor,                 # [B,K,S]
        t_norm: torch.Tensor,                 # [B,K,1] (unused)
        mask: Optional[torch.Tensor] = None,  # [B,K]
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # MSE in normalized Z-space
        loss_z = (pred_z - true_z) ** 2

        # MAE in log10-physical space (MUST backprop to pred_z)
        pred_log = self._z_to_log10(pred_z)
        true_log = self._z_to_log10(true_z)
        loss_phys = (pred_log - true_log).abs() * self.w_species

        if mask is not None:
            m = mask.unsqueeze(-1).to(loss_phys.dtype)
            loss_phys = loss_phys * m
            loss_z = loss_z * m

            valid_pairs = torch.count_nonzero(mask if mask.dtype == torch.bool else (mask > 0))
            if int(valid_pairs.item()) == 0:
                raise ValueError("AdaptiveStiffLoss: mask has zero valid [B,K] positions; cannot compute loss.")
            denom = valid_pairs.to(loss_phys.dtype) * loss_phys.shape[-1]
        else:
            denom = torch.tensor(loss_phys.numel(), dtype=loss_phys.dtype, device=loss_phys.device)

        phys = self.lambda_phys * loss_phys.sum() / denom
        zstb = self.lambda_z * loss_z.sum() / denom
        total = phys + zstb

        if return_components:
            return {"total": total, "phys": phys, "z": zstb}
        return total


# ============================== LIGHTNING TASK ================================

class ModelTask(pl.LightningModule):
    """LightningModule wrapping the model + sample-weighted epoch metrics."""
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], work_dir: Path) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)

        tcfg = cfg.get("training", {})
        self.lr = float(tcfg.get("lr", DEF_LR))
        self.weight_decay = float(tcfg.get("weight_decay", DEF_WEIGHT_DECAY))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", DEF_WARMUP_EPOCHS))
        self.min_lr = float(tcfg.get("min_lr", DEF_MIN_LR))

        # Optional GT slicing when target subset used
        self._target_idx = self._resolve_target_indices()

        # Build loss from normalization manifest
        self.criterion = self._build_loss()

        self.save_hyperparameters({
            "cfg_min": {
                "training": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "warmup_epochs": self.warmup_epochs,
                    "min_lr": self.min_lr,
                },
                "paths": {"processed_data_dir": cfg.get("paths", {}).get("processed_data_dir", "")},
            },
            "work_dir": str(work_dir),
        })

    def _resolve_target_indices(self) -> Optional[torch.Tensor]:
        data_cfg = self.cfg.get("data", {})
        species = list(data_cfg.get("species_variables") or [])
        targets = list(data_cfg.get("target_species") or species)

        if not species or targets == species:
            return None

        name_to_idx = {n: i for i, n in enumerate(species)}
        missing = [n for n in targets if n not in name_to_idx]
        if missing:
            raise ValueError(f"target_species contains unknown names: {missing}")

        idx = [name_to_idx[n] for n in targets]
        return torch.tensor(idx, dtype=torch.long)

    def _build_loss(self) -> nn.Module:
        manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        stats = manifest["per_key_stats"]
        meta = manifest.get("meta", {})

        data_cfg = self.cfg.get("data", {})
        targets = data_cfg.get("target_species")
        if targets:
            species_names = list(targets)
        else:
            species_names = list(data_cfg.get("species_variables") or meta.get("species_variables", []))
            if not species_names:
                raise RuntimeError("No species list available in config or manifest.")

        log_means, log_stds, log_mins, log_maxs = [], [], [], []
        for n in species_names:
            s = stats[n]
            log_means.append(float(s.get("log_mean", 0.0)))
            log_stds.append(float(s.get("log_std", 1.0)))
            log_mins.append(float(s.get("log_min", -30.0)))
            log_maxs.append(float(s.get("log_max", 0.0)))

        loss_cfg = self.cfg.get("training", {}).get("adaptive_stiff_loss", {})

        return AdaptiveStiffLoss(
            log_means=torch.tensor(log_means, dtype=torch.float32),
            log_stds=torch.tensor(log_stds, dtype=torch.float32),
            species_log_min=torch.tensor(log_mins, dtype=torch.float32),
            species_log_max=torch.tensor(log_maxs, dtype=torch.float32),
            lambda_phys=float(loss_cfg.get("lambda_phys", LOSS_LAMBDA_PHYS)),
            lambda_z=float(loss_cfg.get("lambda_z", LOSS_LAMBDA_Z)),
            use_weighting=bool(loss_cfg.get("use_weighting", True)),
            weight_power=float(loss_cfg.get("weight_power", 0.5)),
            w_min=float(loss_cfg.get("w_min", W_SPECIES_CLAMP_MIN)),
            w_max=float(loss_cfg.get("w_max", W_SPECIES_CLAMP_MAX)),
            epsilon_phys=float(loss_cfg.get("epsilon_phys", LOSS_EPS_PHYS)),
        )

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.model(y_i, dt_norm, g)

    def _unpack(self, batch):
        # (y_i, dt_norm, y_j, g, aux[, k_mask])
        if len(batch) == 6:
            y_i, dt_norm, y_j, g, _aux, k_mask = batch
        else:
            y_i, dt_norm, y_j, g, _aux = batch
            k_mask = None

        if self._target_idx is not None:
            idx = self._target_idx.to(y_j.device)
            y_j = y_j.index_select(dim=-1, index=idx)
        return y_i, dt_norm, y_j, g, k_mask

    # --------------------------- optim / sched --------------------------------

    def configure_optimizers(self):
        tcfg = self.cfg.get("training", {})
        opt_name = str(tcfg.get("optimizer", "adamw")).lower().strip()

        if opt_name == "adamw":
            try:
                has_fused_flag = ("fused" in inspect.signature(torch.optim.AdamW).parameters)
            except Exception:
                has_fused_flag = False

            try:
                clip_val = float(getattr(self.trainer, "gradient_clip_val", 0.0) or 0.0)
            except Exception:
                clip_val = 0.0

            use_fused = bool(torch.cuda.is_available() and has_fused_flag and clip_val == 0.0)

            opt_kwargs = dict(lr=self.lr, weight_decay=self.weight_decay)
            if use_fused:
                opt_kwargs["fused"] = True
            optimizer = torch.optim.AdamW(self.parameters(), **opt_kwargs)

        elif opt_name == "lamb":
            import torch_optimizer as torchopt
            optimizer = torchopt.Lamb(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        else:
            raise ValueError(f"Unknown optimizer '{opt_name}'. Expected 'adamw' or 'lamb'.")

        # warmup -> cosine
        scheds = []
        if self.warmup_epochs > 0:
            scheds.append(
                LinearLR(
                    optimizer,
                    start_factor=WARMUP_START_FACTOR,
                    total_iters=max(1, self.warmup_epochs),
                )
            )

        max_epochs = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        t_max = max(1, max_epochs - self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.min_lr)

        scheduler = cosine if not scheds else SequentialLR(
            optimizer,
            schedulers=[scheds[0], cosine],
            milestones=[self.warmup_epochs],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": CKPT_MONITOR}]

    # --------------------------- steps / epoch metrics ------------------------

    def on_train_epoch_start(self) -> None:
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device
        self._t_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_wsum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_z_sum = torch.zeros((), dtype=torch.float64, device=dev)

    def training_step(self, batch, batch_idx: int):
        y_i, dt_in, y_j, g, k_mask = self._unpack(batch)
        pred = self(y_i, dt_in, g)

        if pred.shape != y_j.shape:
            raise RuntimeError(f"Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")

        comps = self.criterion(pred, y_j, dt_in, k_mask, return_components=True)
        loss = comps["total"]

        # sample-weighted accumulation (matches validation exactly)
        with torch.no_grad():
            if k_mask is not None:
                valid_pairs = torch.count_nonzero(k_mask if k_mask.dtype == torch.bool else (k_mask > 0))
            else:
                B, K, _S = y_j.shape
                valid_pairs = torch.tensor(B * K, device=y_j.device, dtype=torch.long)

            if int(valid_pairs.item()) > 0:
                S = torch.tensor(y_j.shape[-1], device=y_j.device, dtype=torch.float64)
                weight = valid_pairs.to(torch.float64) * S

                self._t_sum += (comps["total"].detach().to(torch.float64) * weight)
                self._t_phys_sum += (comps["phys"].detach().to(torch.float64) * weight)
                self._t_z_sum += (comps["z"].detach().to(torch.float64) * weight)
                self._t_wsum += weight

        return loss

    def on_train_epoch_end(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            for t in (self._t_sum, self._t_phys_sum, self._t_z_sum, self._t_wsum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if float(self._t_wsum.item()) <= 0.0:
            for name in ("_t_sum", "_t_phys_sum", "_t_z_sum", "_t_wsum"):
                setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        w = self._t_wsum
        train_loss = (self._t_sum / w).to(torch.float32)
        train_phys = (self._t_phys_sum / w).to(torch.float32)
        train_z = (self._t_z_sum / w).to(torch.float32)

        # Approx fractional error proxy from mean |Δlog10|:
        #   frac ≈ 10**(train_phys) - 1
        eps = float(getattr(self.criterion, "eps_phys", LOSS_EPS_PHYS))
        base10 = torch.tensor(10.0, device=train_phys.device, dtype=train_phys.dtype)
        train_frac = torch.pow(base10, train_phys) - 1.0
        if eps > 0.0:
            train_frac = torch.clamp(train_frac, min=eps)

        # already all-reduced (no sync_dist)
        self.log("train_loss", train_loss, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_phys", train_phys, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_z", train_z, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_frac", train_frac, prog_bar=False, on_epoch=True, sync_dist=False)

        for name in ("_t_sum", "_t_phys_sum", "_t_z_sum", "_t_wsum"):
            setattr(self, name, torch.zeros_like(getattr(self, name)))

    def on_validation_epoch_start(self) -> None:
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device
        self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)

    def validation_step(self, batch, batch_idx: int):
        y_i, dt_in, y_j, g, k_mask = self._unpack(batch)
        pred = self(y_i, dt_in, g)

        if pred.shape != y_j.shape:
            raise RuntimeError(f"[val] Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")

        comps = self.criterion(pred, y_j, dt_in, k_mask, return_components=True)
        total = comps["total"].detach()
        phys = comps["phys"].detach()
        zstb = comps["z"].detach()

        if not torch.isfinite(total):
            raise ValueError(f"[val] Non-finite total loss at batch {batch_idx}")
        if not torch.isfinite(phys):
            raise ValueError(f"[val] Non-finite phys loss at batch {batch_idx}")
        if not torch.isfinite(zstb):
            raise ValueError(f"[val] Non-finite z loss at batch {batch_idx}")

        if k_mask is not None:
            valid_pairs = torch.count_nonzero(k_mask if k_mask.dtype == torch.bool else (k_mask > 0))
        else:
            B, K, _S = y_j.shape
            valid_pairs = torch.tensor(B * K, device=y_j.device, dtype=torch.long)

        if int(valid_pairs.item()) == 0:
            B, K, S = y_j.shape
            raise ValueError(f"[val] Zero valid [B,K] pairs in batch {batch_idx} (B={B}, K={K}, S={S}).")

        S = torch.tensor(y_j.shape[-1], device=y_j.device, dtype=torch.float64)
        weight = valid_pairs.to(torch.float64) * S

        self._v_sum += (total.to(torch.float64) * weight)
        self._v_phys_sum += (phys.to(torch.float64) * weight)
        self._v_z_sum += (zstb.to(torch.float64) * weight)
        self._v_wsum += weight

        return total

    def on_validation_epoch_end(self) -> None:
        tr = getattr(self, "trainer", None)
        if tr is not None and getattr(tr, "sanity_checking", False):
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
                if hasattr(self, name):
                    setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        import torch.distributed as dist

        if not hasattr(self, "_v_wsum"):
            dev = self.device if hasattr(self, "device") else next(self.parameters()).device
            self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)

        if dist.is_available() and dist.is_initialized():
            for t in (self._v_sum, self._v_phys_sum, self._v_z_sum, self._v_wsum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if float(self._v_wsum.item()) <= 0.0:
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
                setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        w = self._v_wsum
        val_loss = (self._v_sum / w).to(torch.float32)
        val_phys = (self._v_phys_sum / w).to(torch.float32)
        val_z = (self._v_z_sum / w).to(torch.float32)

        # Approx fractional error proxy from mean |Δlog10|:
        #   frac ≈ 10**(val_phys) - 1
        eps = float(getattr(self.criterion, "eps_phys", LOSS_EPS_PHYS))
        base10 = torch.tensor(10.0, device=val_phys.device, dtype=val_phys.dtype)
        val_frac = torch.pow(base10, val_phys) - 1.0
        if eps > 0.0:
            val_frac = torch.clamp(val_frac, min=eps)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_phys", val_phys, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_z", val_z, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_frac", val_frac, prog_bar=False, on_epoch=True, sync_dist=False)

        for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_wsum"):
            setattr(self, name, torch.zeros_like(getattr(self, name)))


# ============================== CALLBACKS =====================================

class SetEpochCallback(Callback):
    """
    Minimal and safe:
      - Only touches the TRAIN dataset
      - Only once per epoch (train_epoch_start)
    """
    def __init__(self, train_dataset: Any) -> None:
        super().__init__()
        self.train_dataset = train_dataset

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ds = self.train_dataset
        if ds is None:
            return
        try:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(trainer.current_epoch)
        except Exception:
            pass


class EpochTableCallback(Callback):
    """
    Rank-0 epoch summary print.

    EPOCH | train       | val         | val_frac    | lr         | time
    ----- | ----------- | ----------- | ----------- | ---------- | --------
    E0001 | 2.345e-01   | 1.234e-01   | 5.678e-02   | 1.000e-03  | 12:34
    """
    def __init__(self) -> None:
        super().__init__()
        self._t0: float | None = None
        self._printed_header: bool = False
        self._last_printed_epoch: int = -1

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Start timing at the beginning of the training epoch.
        if trainer.is_global_zero and not trainer.sanity_checking:
            self._t0 = time.time()

    @staticmethod
    def _get_metric(metrics: dict, key: str) -> Optional[float]:
        for k in (key, f"{key}_epoch"):
            v = metrics.get(k)
            if v is None:
                continue
            try:
                return float(v.detach().cpu().item() if hasattr(v, "detach") else float(v))
            except Exception:
                continue
        return None

    @staticmethod
    def _get_lr(trainer: pl.Trainer) -> Optional[float]:
        try:
            if trainer.optimizers and trainer.optimizers[0].param_groups:
                return float(trainer.optimizers[0].param_groups[0].get("lr", None))
        except Exception:
            pass
        return None

    @staticmethod
    def _fmt_sci(x: Optional[float], width: int = 11) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return f"{'—':>{width}}"
        return f"{x:>{width}.3e}"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _maybe_print(self, trainer: pl.Trainer) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        # Guard against double prints if multiple hooks call into this.
        if trainer.current_epoch == self._last_printed_epoch:
            return

        elapsed = (time.time() - self._t0) if self._t0 is not None else 0.0

        metrics = dict(trainer.callback_metrics)
        tr = self._get_metric(metrics, "train_loss")
        va = self._get_metric(metrics, "val_loss")
        vf = self._get_metric(metrics, "val_frac")
        lr = self._get_lr(trainer)

        if not self._printed_header:
            print("EPOCH | train       | val         | val_frac    | lr         | time", flush=True)
            print("----- | ----------- | ----------- | ----------- | ---------- | --------", flush=True)
            self._printed_header = True

        epoch_str = f"E{trainer.current_epoch + 1:04d}"
        line = (
            f"{epoch_str} | "
            f"{self._fmt_sci(tr)} | "
            f"{self._fmt_sci(va)} | "
            f"{self._fmt_sci(vf)} | "
            f"{self._fmt_sci(lr)} | "
            f"{self._fmt_time(elapsed)}"
        )
        print(line, flush=True)

        self._last_printed_epoch = trainer.current_epoch

    def on_fit_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # This runs after Lightning has finalized epoch metrics.
        self._maybe_print(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Fallback for older Lightning versions / edge cases where on_fit_epoch_end
        # isn’t triggered as expected. Guard prevents double printing.
        self._maybe_print(trainer)



# ============================== WRAPPER =======================================

class Trainer:
    """Compatibility wrapper: `from trainer import Trainer` -> .train() returns best val."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        work_dir: Path,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        *,
        pl_precision_override: Any = None,  # optional override from main
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.pl_precision_override = pl_precision_override

        self.logger = logger or logging.getLogger("trainer")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # Silence non-zero ranks to avoid duplicate INFO lines
        if int(os.getenv("RANK", "0") or "0") != 0:
            self.logger.propagate = False
            self.logger.setLevel(logging.WARNING)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.get("training", {})
        syscfg = cfg.get("system", {})

        self.epochs = int(tcfg.get("epochs", DEF_EPOCHS))
        self.grad_clip = float(tcfg.get("gradient_clip", DEF_GRAD_CLIP))
        self.accumulate = int(tcfg.get("accumulate_grad_batches", DEF_ACCUMULATE))
        self.torch_compile = bool(tcfg.get("torch_compile", COMPILE_ENABLE_DEFAULT))

        # Determinism / cudnn benchmark (keep aligned with main)
        self.deterministic = bool(syscfg.get("deterministic", False))
        self.cudnn_benchmark = bool(syscfg.get("cudnn_benchmark", True)) and not self.deterministic

        # SWA config
        self.use_swa = bool(tcfg.get("use_swa", False))
        self.swa_lrs = tcfg.get("swa_lrs", None)
        self.swa_epoch_start = tcfg.get("swa_epoch_start", 0.8)
        self.swa_annealing_epochs = int(tcfg.get("swa_annealing_epochs", 10))
        self.swa_annealing_strategy = str(tcfg.get("swa_annealing_strategy", "cos"))

        # Compile knobs
        self.compile_backend = tcfg.get("torch_compile_backend", COMPILE_BACKEND)
        self.compile_mode = tcfg.get("torch_compile_mode", COMPILE_MODE)
        self.compile_dynamic = bool(tcfg.get("compile_dynamic", COMPILE_DYNAMIC))
        self.compile_fullgraph = bool(tcfg.get("compile_fullgraph", COMPILE_FULLGRAPH))

        # Optional limits (PL "limit_*_batches")
        self.limit_train_batches = int(tcfg.get("max_train_batches", DEF_MAX_TRAIN_BATCHES) or 0)
        self.limit_val_batches = int(tcfg.get("max_val_batches", DEF_MAX_VAL_BATCHES) or 0)

    def _precision_from_cfg(self) -> Any:
        if self.pl_precision_override is not None:
            return self.pl_precision_override

        mp = str(self.cfg.get("mixed_precision", {}).get("mode", "32-true")).lower().strip()
        aliases = {
            "bf16": "bf16-mixed",
            "bfloat16": "bf16-mixed",
            "bf16-mixed": "bf16-mixed",
            "bfloat16-mixed": "bf16-mixed",
            "fp16": "16-mixed",
            "float16": "16-mixed",
            "16": "16-mixed",
            "16-mixed": "16-mixed",
            "none": "32-true",
            "fp32": "32-true",
            "32": "32-true",
            "32-true": "32-true",
        }
        return aliases.get(mp, "32-true")

    def _accelerator_from_device(self) -> str:
        if self.device.type == "cuda":
            return "gpu"
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def _resolve_resume_ckpt(self) -> Optional[str]:
        tcfg = self.cfg.get("training", {})
        p_cfg = tcfg.get("resume_path") or tcfg.get("resume")
        if isinstance(p_cfg, str) and p_cfg.strip():
            p = Path(p_cfg)
            return str(p) if p.is_file() else None

        env_resume = os.environ.get(ENV_RESUME_VAR, "").strip()
        if env_resume and env_resume.lower() != RESUME_AUTO_SENTINEL:
            p = Path(env_resume)
            return str(p) if p.is_file() else None

        if env_resume.lower() == RESUME_AUTO_SENTINEL or bool(tcfg.get("auto_resume", True)):
            p = self.work_dir / LAST_CKPT_NAME
            if p.is_file():
                return str(p)

        return None

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        if not self.torch_compile:
            return model
        try:
            compiled = torch.compile(
                model,
                backend=self.compile_backend,
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
                fullgraph=self.compile_fullgraph,
            )
            self.logger.info(
                f"torch.compile enabled (backend={self.compile_backend}, mode={self.compile_mode}, "
                f"dynamic={self.compile_dynamic}, fullgraph={self.compile_fullgraph})"
            )
            return compiled
        except Exception as e:
            self.logger.warning(f"torch.compile disabled ({e})")
            return model

    def train(self) -> float:
        # Optional model compile (Lightning handles precision)
        self.model = self._maybe_compile(self.model)
        task = ModelTask(self.model, self.cfg, self.work_dir)

        csv_logger = CSVLogger(save_dir=str(self.work_dir), name="csv")

        ckpt_cb = ModelCheckpoint(
            dirpath=str(self.work_dir),
            filename=CKPT_FILENAME_BEST,
            monitor=CKPT_MONITOR,
            mode=CKPT_MODE,
            save_top_k=1,
            save_last=True,
            verbose=False,
        )
        lr_cb = LearningRateMonitor(logging_interval=LR_LOGGING_INTERVAL)

        # Only set epoch on TRAIN dataset (once per epoch)
        train_ds = getattr(self.train_loader, "dataset", None)
        epoch_set_cb = SetEpochCallback(train_ds)

        epoch_print_cb = EpochTableCallback()

        callbacks = [ckpt_cb, lr_cb, epoch_set_cb, epoch_print_cb]

        if self.use_swa:
            swa_kwargs = dict(
                swa_epoch_start=self.swa_epoch_start,
                annealing_epochs=self.swa_annealing_epochs,
                annealing_strategy=self.swa_annealing_strategy,
            )
            if self.swa_lrs is not None:
                swa_kwargs["swa_lrs"] = self.swa_lrs
            callbacks.append(StochasticWeightAveraging(**swa_kwargs))

        accelerator = self._accelerator_from_device()
        precision = self._precision_from_cfg()

        trainer_kwargs: dict[str, Any] = {}
        if self.limit_train_batches > 0:
            trainer_kwargs["limit_train_batches"] = self.limit_train_batches
        if self.limit_val_batches > 0:
            trainer_kwargs["limit_val_batches"] = self.limit_val_batches

        # DDP: only support externally-launched (torchrun/mpirun/srun).
        world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
        externally_launched = world_size > 1

        # Avoid Lightning spawn; this codebase builds datasets in main (often GPU-preloaded).
        if not externally_launched:
            requested = int(os.getenv("PL_DEVICES", "1") or "1")
            if requested != 1:
                self.logger.warning("PL_DEVICES>1 requested but spawn is unsafe here; forcing devices=1. Use torchrun/mpirun/srun.")
            devices = 1
        else:
            devices = 1  # per-rank process
        strategy: Any = "ddp" if externally_launched else "auto"

        base_kwargs = dict(
            default_root_dir=str(self.work_dir),
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            gradient_clip_val=self.grad_clip if self.grad_clip > 0 else 0.0,
            accumulate_grad_batches=max(1, self.accumulate),
            logger=csv_logger,
            callbacks=callbacks,
            enable_checkpointing=ENABLE_CHECKPOINTING,
            enable_progress_bar=ENABLE_PROGRESS_BAR,
            enable_model_summary=ENABLE_MODEL_SUMMARY,
            detect_anomaly=DETECT_ANOMALY,
            inference_mode=INFERENCE_MODE,
            benchmark=self.cudnn_benchmark,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            **trainer_kwargs,
        )

        # keep deterministic flag if supported by installed PL
        try:
            pl_trainer = pl.Trainer(deterministic=self.deterministic, **base_kwargs)
        except TypeError:
            pl_trainer = pl.Trainer(**base_kwargs)

        ckpt_path = self._resolve_resume_ckpt()
        self.logger.info(f"resume: {ckpt_path if ckpt_path else 'fresh'}")

        pl_trainer.fit(
            task,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=ckpt_path,
        )

        best = ckpt_cb.best_model_score
        return float(best.item()) if best is not None else float("nan")
