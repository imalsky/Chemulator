#!/usr/bin/env python3
"""
trainer.py

- CSV-only logging; checkpoints: best.ckpt / last.ckpt in work_dir
- AMP precision from cfg (bf16/fp16/fp32)
- Optional torch.compile (Inductor)
- Cosine schedule with warmup
- Vectorized K handling (fixed K per batch)
- Sample-weighted epoch reduction for train/val
- Rollout training for stable long-horizon autoregressive prediction

Dataset structure: (y_i, dt_norm[B,K,1], y_j[B,K,S], g[B,G])

NOTE on "multiplicative error proxy":
  mult_err_proxy = expm1(ln(10) * mean_abs_log10_error)
  This maps log10-MAE to a typical multiplicative factor minus 1.
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from utils import compute_rollout_loss_weights, get_curriculum_steps

# =============================================================================
# Constants
# =============================================================================

# Checkpoint settings
CKPT_FILENAME_BEST = "best"
CKPT_MONITOR = "val_loss"
CKPT_MODE = "min"

# Lightning settings
LOG_EVERY_N_STEPS = 200
ENABLE_CHECKPOINTING = True
ENABLE_PROGRESS_BAR = False
ENABLE_MODEL_SUMMARY = False
DETECT_ANOMALY = False
INFERENCE_MODE = True

# Training defaults
DEF_EPOCHS = 100
DEF_GRAD_CLIP = 0.0
DEF_ACCUMULATE = 1
DEF_MAX_TRAIN_BATCHES = 0
DEF_MAX_VAL_BATCHES = 0
DEF_LR = 1e-3
DEF_WEIGHT_DECAY = 1e-4
DEF_WARMUP_EPOCHS = 0
DEF_MIN_LR = 1e-6

# torch.compile defaults
COMPILE_ENABLE_DEFAULT = True
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "default"
COMPILE_DYNAMIC = False
COMPILE_FULLGRAPH = False

# Loss constants
LOSS_LAMBDA_PHYS = 1.0
LOSS_LAMBDA_Z = 0.1
LOSS_EPS_PHYS = 1e-20
LOG_STD_CLAMP_MIN = 1e-10
W_SPECIES_CLAMP_MIN = 0.5
W_SPECIES_CLAMP_MAX = 2.0
SPECIES_RANGE_EPS = 1e-6
W_SPECIES_MEAN_EPS = 1e-12
WARMUP_START_FACTOR = 0.1

# Math constants
LN10 = math.log(10.0)


# =============================================================================
# Loss Function
# =============================================================================


class AdaptiveStiffLoss(nn.Module):
    """
    Loss function: MSE in z plus MAE in log10-physical space with species weighting.

    Includes helpers for log10-space operations needed for rollout training.
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
        epsilon_phys: float = LOSS_EPS_PHYS,
    ) -> None:
        super().__init__()

        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=LOG_STD_CLAMP_MIN))
        self.register_buffer("ln10", torch.tensor(LN10, dtype=torch.float32))
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

    def z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        """Convert normalized z back to log10 space (DIFFERENTIABLE)."""
        return z * self.log_stds + self.log_means

    def log10_to_z(self, log10_vals: torch.Tensor) -> torch.Tensor:
        """Convert log10 values to z-normalized space (DIFFERENTIABLE)."""
        return (log10_vals - self.log_means) / self.log_stds

    def clamp_z_in_log10_space(
        self,
        z: torch.Tensor,
        log10_min: float = -30.0,
        log10_max: float = 10.0,
    ) -> torch.Tensor:
        """Clamp z values in log10 space to avoid nonphysical values."""
        log10_vals = self.z_to_log10(z)
        log10_clamped = torch.clamp(log10_vals, min=log10_min, max=log10_max)
        return self.log10_to_z(log10_clamped)

    def project_z_to_simplex(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project z onto the simplex in physical space (sum_i y_i = 1).

        Operates in log10 space:
            log10(y') = log10(y) - log10(sum_i 10**log10(y_i))
        """
        log10_vals = self.z_to_log10(z).to(torch.float32)
        ln10 = self.ln10.to(device=log10_vals.device, dtype=torch.float32)
        ln_raw = log10_vals * ln10
        ln_norm = ln_raw - torch.logsumexp(ln_raw, dim=-1, keepdim=True)
        log10_proj = ln_norm / ln10
        z_proj = self.log10_to_z(log10_proj)
        return z_proj.to(dtype=z.dtype, device=z.device)

    def add_noise_in_log10_space(self, z: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """
        Add Gaussian noise in log10 space for robust autoregressive training.

        Forces the model to recover from prediction errors that accumulate during inference rollouts.
        """
        log10_vals = self.z_to_log10(z)
        noise = torch.randn_like(log10_vals) * noise_std
        log10_noisy = log10_vals + noise
        return self.log10_to_z(log10_noisy)

    def forward(
        self,
        pred_z: torch.Tensor,
        true_z: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        diff_z_f = (pred_z - true_z).to(torch.float32)
        loss_z_f = diff_z_f * diff_z_f

        pred_log = self.z_to_log10(pred_z)
        true_log = self.z_to_log10(true_z)
        abs_log_f = (pred_log - true_log).abs().to(torch.float32)

        w_species_f = self.w_species.to(dtype=abs_log_f.dtype)
        loss_phys_f = abs_log_f * w_species_f

        denom_f = torch.tensor(loss_phys_f.numel(), dtype=loss_phys_f.dtype, device=loss_phys_f.device)

        mean_abs_log10 = abs_log_f.sum() / denom_f
        weighted_mean_abs_log10 = loss_phys_f.sum() / denom_f

        phys = self.lambda_phys * weighted_mean_abs_log10
        zstb = self.lambda_z * (loss_z_f.sum() / denom_f)
        total = phys + zstb

        if return_components:
            return {
                "total": total,
                "phys": phys,
                "z": zstb,
                "mean_abs_log10": mean_abs_log10,
                "weighted_mean_abs_log10": weighted_mean_abs_log10,
            }
        return total


# =============================================================================
# Epoch CSV Writer
# =============================================================================


class EpochCSVWriter:
    """Single, epoch-level CSV writer that appends across restarts."""

    COLUMNS = (
        "epoch",
        "step",
        "wall_time",
        "epoch_time_sec",
        "lr",
        "train_loss",
        "train_phys",
        "train_z",
        "train_mult_err_proxy",
        "val_loss",
        "val_phys",
        "val_z",
        "val_mult_err_proxy",
        "val_rollout_final_abslog",
        "val_rollout_final_mult_err",
    )

    def __init__(self, csv_path: Path, *, resume: bool) -> None:
        self.csv_path = Path(csv_path)
        self.resume = bool(resume)

    @staticmethod
    def wall_time_str() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _backup_existing(self) -> None:
        if not self.csv_path.exists():
            return
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        bak = self.csv_path.with_name(f"{self.csv_path.stem}.bak.{ts}{self.csv_path.suffix}")
        try:
            self.csv_path.replace(bak)
        except Exception:
            try:
                self.csv_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                return [dict(r) for r in reader]
        except Exception:
            return []

    def _write_rows_atomic(self, rows: list[dict[str, str]]) -> None:
        tmp = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.COLUMNS))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in self.COLUMNS})
        tmp.replace(self.csv_path)

    def ensure_initialized(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if (not self.resume) and self.csv_path.exists():
            self._backup_existing()

        if not self.csv_path.exists():
            self._write_rows_atomic([])
            return

        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            if header is None or tuple(header) != tuple(self.COLUMNS):
                self._backup_existing()
                self._write_rows_atomic([])
        except Exception:
            self._backup_existing()
            self._write_rows_atomic([])

    def write_epoch_row(self, epoch_1idx: int, row: dict[str, str]) -> None:
        rows = self._read_rows()
        kept: list[dict[str, str]] = []
        for r in rows:
            try:
                e = int(float(r.get("epoch", "0") or 0))
            except Exception:
                continue
            if e < epoch_1idx:
                kept.append(r)

        kept.append({k: row.get(k, "") for k in self.COLUMNS})
        self._write_rows_atomic(kept)


# =============================================================================
# Lightning Module
# =============================================================================


class ModelTask(pl.LightningModule):
    """
    LightningModule wrapping the model + sample-weighted epoch metrics.

    When rollout is enabled, both training *and validation* run the long-horizon autoregressive rollout.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        work_dir: Path,
        *,
        resume: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        # If the model predicts in physical log space and enforces mass conservation, project to simplex.
        self._enforce_simplex = bool(getattr(model, "predict_delta_log_phys", False))
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self._logger = logging.getLogger("trainer.task")

        self._resume = bool(resume)
        self._csv: Optional[EpochCSVWriter] = None
        self._epoch_t0: float | None = None
        self._printed_header: bool = False
        self._last_reported_epoch: int = -1
        self._last_train_metrics: dict[str, float] = {}
        self._last_val_metrics: dict[str, float] = {}
        self._last_train_epoch_idx: int | None = None
        self._last_val_epoch_idx: int | None = None

        tcfg = cfg.get("training", {})
        self.lr = float(tcfg.get("lr", DEF_LR))
        self.weight_decay = float(tcfg.get("weight_decay", DEF_WEIGHT_DECAY))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", DEF_WARMUP_EPOCHS))
        self.min_lr = float(tcfg.get("min_lr", DEF_MIN_LR))

        self._init_rollout_config(tcfg)

        self._target_idx = self._resolve_target_indices()
        self.criterion = self._build_loss()

        # dt normalization spec (needed for chunked temporal bundling)
        # Populated by _build_loss() from normalization.json.
        # These are floats (python scalars) and should be treated as read-only.
        self._dt_log_min: float
        self._dt_log_max: float
        self._dt_log_range: float
        self._dt_eps: float
        self._dt_phys_min: float
        self._dt_phys_max: float

        self.save_hyperparameters({
            "cfg_min": {
                "training": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "warmup_epochs": self.warmup_epochs,
                    "min_lr": self.min_lr,
                    "rollout": tcfg.get("rollout", {}),
                },
                "paths": {"processed_data_dir": cfg.get("paths", {}).get("processed_data_dir", "")},
            },
            "work_dir": str(work_dir),
        })

    # --------------------------- Rollout config --------------------------------

    def _init_rollout_config(self, tcfg: Dict[str, Any]) -> None:
        """Initialize rollout training configuration."""
        rollout_cfg = tcfg.get("rollout", {})

        self.rollout_enabled = bool(rollout_cfg.get("enabled", False))
        self.rollout_steps = int(rollout_cfg.get("steps", 4))

        # Curriculum learning
        curriculum_cfg = rollout_cfg.get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.curriculum_start_steps = int(curriculum_cfg.get("start_steps", 1))
        self.curriculum_end_steps = int(curriculum_cfg.get("end_steps", 8))
        self.curriculum_ramp_epochs = int(curriculum_cfg.get("ramp_epochs", 20))

        # Noise injection
        noise_cfg = rollout_cfg.get("noise", {})
        self.noise_enabled = bool(noise_cfg.get("enabled", False))
        self.noise_log10_std = float(noise_cfg.get("log10_std", 0.01))

        # Clipping
        clip_cfg = rollout_cfg.get("clip", {})
        self.clip_enabled = bool(clip_cfg.get("enabled", False))
        self.clip_log10_min = float(clip_cfg.get("log10_min", -30.0))
        self.clip_log10_max = float(clip_cfg.get("log10_max", 10.0))

        # Loss weighting across steps
        self.loss_weighting = str(rollout_cfg.get("loss_weighting", "uniform"))
        self.loss_discount = float(rollout_cfg.get("loss_discount", 0.9))

        # Truncated BPTT
        self.detach_every = int(rollout_cfg.get("detach_every", 0))

        # Optional training modes
        # 1) Pushforward loss mode: stopgrad through the propagated state each step (and optionally skip step-0 loss).
        push_cfg = rollout_cfg.get("pushforward", {})
        self.pushforward_enabled = bool(push_cfg.get("enabled", False))
        self.pushforward_skip_first_loss = bool(push_cfg.get("skip_first_step_loss", True))

        # Backward-compatible: if burn_in_steps is not provided, interpret
        # skip_first_step_loss as burn_in_steps=1. burn_in_steps controls
        # how many initial rollout steps contribute zero loss weight (but are
        # still propagated).
        self.pushforward_burn_in_steps = int(
            push_cfg.get("burn_in_steps", 1 if self.pushforward_skip_first_loss else 0)
        )
        if self.pushforward_burn_in_steps < 0:
            raise ValueError(
                f"rollout.pushforward.burn_in_steps must be >= 0; got {self.pushforward_burn_in_steps}"
            )

        # Whether to apply pushforward behavior during validation rollout.
        # Default: True (backward compatible). Set false if you want rollout
        # validation metrics to reflect the fully backpropagating rollout loss.
        self.pushforward_apply_in_validation = bool(push_cfg.get("apply_in_validation", True))

        # Pushforward implementation mode.
        #   - legacy: detach propagated state at each step/chunk and optionally mask early losses (backward compatible).
        #   - paper: two-step pushforward stability loss as in arXiv:2202.03376 (temporal bundling + stop-grad through the first unroll).
        self.pushforward_mode = str(push_cfg.get("mode", "legacy")).lower().strip()
        if self.pushforward_mode not in ("legacy", "paper"):
            raise ValueError(f"rollout.pushforward.mode must be \"legacy\" or \"paper\"; got {self.pushforward_mode!r}")

        # Relative weighting of the stability (pushforward) term in paper mode.
        self.pushforward_lambda_stability = float(push_cfg.get("lambda_stability", 1.0))
        if self.pushforward_lambda_stability < 0.0:
            raise ValueError("rollout.pushforward.lambda_stability must be >= 0")

        # 2) Chunked temporal bundling: predict multiple consecutive rollout targets in one model call.
        #    For a chunk of incremental dt, we build *cumulative* dt within the chunk, call the model once,
        #    compute per-step losses against the corresponding ground truth, then propagate only the last state.
        bund_cfg = rollout_cfg.get("bundling", {})
        self.bundling_enabled = bool(bund_cfg.get("enabled", False))
        self.bundling_chunk_size = int(bund_cfg.get("chunk_size", 1))
        if self.bundling_chunk_size < 1:
            raise ValueError(f"rollout.bundling.chunk_size must be >= 1; got {self.bundling_chunk_size}")

        # Validation rollout
        val_rollout_cfg = rollout_cfg.get("validation", {})
        # User request: if rollout is enabled, validation should also be long-rollout.
        self.val_rollout_enabled = bool(self.rollout_enabled or val_rollout_cfg.get("enabled", False))
        self.val_rollout_steps = int(val_rollout_cfg.get("steps", self.rollout_steps))

    # --------------------------- Data / loss setup -----------------------------

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

        # --- dt normalization spec (used by chunked temporal bundling) ---
        dt_spec = manifest.get("dt", None)
        if not isinstance(dt_spec, dict):
            raise RuntimeError("normalization.json is missing top-level 'dt' spec (expected dict with log_min/log_max).")
        try:
            self._dt_log_min = float(dt_spec["log_min"])
            self._dt_log_max = float(dt_spec["log_max"])
        except Exception as e:
            raise RuntimeError("Bad dt spec in normalization.json; expected {'log_min': <float>, 'log_max': <float>}.") from e

        self._dt_log_range = float(self._dt_log_max - self._dt_log_min)
        if not (self._dt_log_range > 0.0):
            raise RuntimeError(f"Bad dt spec: log_max must be > log_min; got log_min={self._dt_log_min}, log_max={self._dt_log_max}")

        # Match dataset.py behavior: dt_epsilon may live under dataset.dt_epsilon or normalization.epsilon.
        self._dt_eps = float(self.cfg.get("dataset", {}).get("dt_epsilon", self.cfg.get("normalization", {}).get("epsilon", 1e-30)))
        self._dt_phys_min = max((10.0 ** self._dt_log_min), self._dt_eps)
        self._dt_phys_max = (10.0 ** self._dt_log_max)

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

    # --------------------------- Forward helpers -------------------------------

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.model(y_i, dt_norm, g)

    def _unpack(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) != 4:
            raise ValueError(f"Expected batch to be a 4-tuple (y_i, dt_norm, y_j, g); got len={len(batch)}")
        y_i, dt_norm, y_j, g = batch

        if self._target_idx is not None:
            idx = self._target_idx.to(y_j.device)
            y_j = y_j.index_select(dim=-1, index=idx)
        return y_i, dt_norm, y_j, g

    def _get_rollout_steps_for_epoch(self) -> int:
        """Get number of rollout steps for current epoch (with curriculum)."""
        if not self.curriculum_enabled:
            return self.rollout_steps

        return get_curriculum_steps(
            epoch=self.current_epoch,
            start_steps=self.curriculum_start_steps,
            end_steps=self.curriculum_end_steps,
            ramp_epochs=self.curriculum_ramp_epochs,
        )

    def _apply_state_constraints(self, state: torch.Tensor, apply_noise: bool) -> torch.Tensor:
        """Apply clipping, simplex projection, and optional noise to a state in z-space."""
        if apply_noise and self.noise_enabled and self.training:
            state = self.criterion.add_noise_in_log10_space(state, noise_std=self.noise_log10_std)

        if self.clip_enabled:
            state = self.criterion.clamp_z_in_log10_space(
                state, log10_min=self.clip_log10_min, log10_max=self.clip_log10_max
            )

        if self._enforce_simplex:
            state = self.criterion.project_z_to_simplex(state)

        return state

    # --------------------------- dt conversion helpers (for bundling) ---------

    def _dt_norm_to_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized dt in [0,1] to physical dt (seconds). Always returns float32."""
        dt = dt_norm
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        if dt.ndim != 2:
            raise ValueError(f"dt_norm_to_phys expects [B,K] or [B,K,1]; got {tuple(dt_norm.shape)}")

        dt_f32 = dt.to(torch.float32)
        dt_f32 = torch.clamp(dt_f32, 0.0, 1.0)
        dt_log = self._dt_log_min + dt_f32 * self._dt_log_range
        dt_phys = torch.pow(10.0, dt_log)
        return torch.clamp(dt_phys, min=self._dt_eps)

    def _dt_phys_to_norm(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """Convert physical dt (seconds) to normalized dt in [0,1]. Always returns float32."""
        dt = dt_phys
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        if dt.ndim != 2:
            raise ValueError(f"dt_phys_to_norm expects [B,K] or [B,K,1]; got {tuple(dt_phys.shape)}")

        dt_f32 = dt.to(torch.float32)
        dt_clamped = dt_f32.clamp(min=self._dt_phys_min, max=self._dt_phys_max)
        dt_log = torch.log10(dt_clamped)
        dt_norm = (dt_log - self._dt_log_min) / self._dt_log_range
        return torch.clamp(dt_norm, 0.0, 1.0)

    # --------------------------- Loss paths ------------------------------------

    def _onestep_forward_loss(
        self,
        y_i: torch.Tensor,
        dt_in: torch.Tensor,
        y_j: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Standard one-step forward pass and loss computation over all K in one call."""
        pred = self(y_i, dt_in, g)

        if pred.shape != y_j.shape:
            raise RuntimeError(f"Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")

        comps = self.criterion(pred, y_j, return_components=True)
        comps["num_steps"] = torch.tensor(y_j.shape[1], device=y_j.device, dtype=torch.long)
        return comps["total"], comps


    def _paper_pushforward_rollout_forward_loss(
        self,
        *,
        y_i: torch.Tensor,
        dt_in_3: torch.Tensor,
        y_j: torch.Tensor,
        g: torch.Tensor,
        num_steps: int,
        record_step_abslog: bool,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Paper-faithful pushforward + temporal bundling (arXiv:2202.03376).

        Implements the *two-step* pushforward stability objective:
          L_total = 0.5 * ( L_one_step + lambda * L_stability )

        where:
          - L_one_step compares predictions from the true state distribution (teacher-forced)
          - L_stability compares predictions from the pushforward distribution, using the last
            bundled prediction from the first call as the next input, with stop-grad through that input.

        This path is intentionally strict:
          - requires num_steps == 2 * bundle_size
          - requires burn_in_steps == bundle_size (to avoid ambiguous configs)
          - ignores loss weighting/discounting (uniform, matching the paper)
        """
        B, K, _S = y_j.shape
        device = y_j.device

        bundle_size = int(self.bundling_chunk_size) if (self.bundling_enabled and self.bundling_chunk_size > 1) else 1
        if bundle_size < 1:
            raise ValueError(f"Invalid bundle_size={bundle_size}")

        expected_steps = 2 * bundle_size
        if int(num_steps) != int(expected_steps):
            raise ValueError(
                f"pushforward.mode='paper' requires rollout.steps == 2 * bundling.chunk_size "
                f"(expected {expected_steps}, got {num_steps})."
            )

        # burn_in_steps is a legacy knob; in paper mode, it must match exactly.
        if int(self.pushforward_burn_in_steps) != int(bundle_size):
            raise ValueError(
                f"pushforward.mode='paper' requires pushforward.burn_in_steps == bundling.chunk_size "
                f"(expected {bundle_size}, got {self.pushforward_burn_in_steps})."
            )

        if dt_in_3.shape[0] != B or dt_in_3.shape[1] < expected_steps or dt_in_3.shape[-1] != 1:
            raise ValueError(
                f"dt_in must have shape [B,>= {expected_steps},1] for paper pushforward; got {tuple(dt_in_3.shape)}"
            )

        def _build_dt_cum_norm_3(start: int, L: int) -> torch.Tensor:
            dt_inc_norm = dt_in_3[:, start : start + L, :]      # [B,L,1]
            dt_inc_phys = self._dt_norm_to_phys(dt_inc_norm)    # [B,L]
            dt_cum_phys = torch.cumsum(dt_inc_phys, dim=1)      # [B,L]
            dt_cum_norm = self._dt_phys_to_norm(dt_cum_phys)    # [B,L]
            return dt_cum_norm.unsqueeze(-1)                    # [B,L,1]

        lam = float(self.pushforward_lambda_stability)

        # ---- One-step bundled loss on the true state distribution ----
        dt1_cum_norm_3 = _build_dt_cum_norm_3(0, bundle_size)
        true1 = y_j[:, 0:bundle_size, :]
        pred1 = self(y_i, dt1_cum_norm_3, g)
        if pred1.shape != true1.shape:
            raise RuntimeError(f"paper pushforward: pred1/true1 shape mismatch {tuple(pred1.shape)} vs {tuple(true1.shape)}")
        comps1 = self.criterion(pred1, true1, return_components=True)

        # Propagate only the last state (constrained) to build the pushforward distribution.
        last_pred1 = pred1[:, -1, :]
        state1 = self._apply_state_constraints(last_pred1, apply_noise=False)

        # ---- Stability bundled loss on the pushforward distribution (stop-grad through state1) ----
        dt2_cum_norm_3 = _build_dt_cum_norm_3(bundle_size, bundle_size)
        true2 = y_j[:, bundle_size : expected_steps, :]
        pred2 = self(state1.detach(), dt2_cum_norm_3, g)
        if pred2.shape != true2.shape:
            raise RuntimeError(f"paper pushforward: pred2/true2 shape mismatch {tuple(pred2.shape)} vs {tuple(true2.shape)}")
        comps2 = self.criterion(pred2, true2, return_components=True)

        # Average across the two solver calls so logging/weighting matches num_steps.
        total_loss = 0.5 * (comps1["total"] + lam * comps2["total"])
        total_phys = 0.5 * (comps1["phys"] + lam * comps2["phys"])
        total_z = 0.5 * (comps1["z"] + lam * comps2["z"])
        total_abslog = 0.5 * (comps1["mean_abs_log10"] + lam * comps2["mean_abs_log10"])

        comps = {
            "total": total_loss,
            "phys": total_phys,
            "z": total_z,
            "mean_abs_log10": total_abslog,
            "num_steps": torch.tensor(int(expected_steps), device=device, dtype=torch.long),
        }

        step_abslog_vec = None
        final_abslog = None
        if record_step_abslog:
            step_abslog_vec = torch.zeros((expected_steps,), device=device, dtype=torch.float32)

            # Per-step unweighted abs-log10 (mean over batch + species).
            pred1_log = self.criterion.z_to_log10(pred1)
            true1_log = self.criterion.z_to_log10(true1)
            abs1 = (pred1_log - true1_log).abs().to(torch.float32)              # [B,L,S]
            step_abslog_vec[0:bundle_size] = abs1.mean(dim=(0, 2))

            pred2_log = self.criterion.z_to_log10(pred2)
            true2_log = self.criterion.z_to_log10(true2)
            abs2 = (pred2_log - true2_log).abs().to(torch.float32)
            step_abslog_vec[bundle_size:expected_steps] = abs2.mean(dim=(0, 2))

            # Final-step abslog uses the CONSTRAINED state that is actually propagated.
            last_pred2 = pred2[:, -1, :]
            state2 = self._apply_state_constraints(last_pred2, apply_noise=False)
            final_pred_log = self.criterion.z_to_log10(state2)
            final_true_log = self.criterion.z_to_log10(y_j[:, expected_steps - 1, :])
            final_abslog = (final_pred_log - final_true_log).abs().mean().to(torch.float32)

        return total_loss, comps, step_abslog_vec, final_abslog

    def _rollout_forward_loss(
        self,
        y_i: torch.Tensor,
        dt_in: torch.Tensor,
        y_j: torch.Tensor,
        g: torch.Tensor,
        *,
        num_steps_req: int,
        apply_noise: bool,
        record_step_abslog: bool = False,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout forward pass.

        - Feeds model output as next input (autoregressive)
        - Optionally injects noise in log10 space (train only)
        - Applies inference-time constraints (clip/simplex) at every step
        - Accumulates a weighted rollout loss across steps

        Returns:
          loss, comps, step_abslog[B? aggregated], final_abslog (optional, validation-style)
        """
        B, K, _S = y_j.shape
        device = y_j.device

        num_steps = min(int(num_steps_req), int(K))
        if num_steps <= 0:
            raise ValueError(f"Invalid rollout steps: requested={num_steps_req} K={K}")

        # Normalize dt_in to [B,K,1] for consistent slicing.
        if dt_in.ndim == 2:
            dt_in_3 = dt_in.unsqueeze(-1)
        else:
            dt_in_3 = dt_in
        if dt_in_3.ndim != 3 or dt_in_3.shape[-1] != 1:
            raise ValueError(f"dt_in must be [B,K,1] or [B,K]; got {tuple(dt_in.shape)}")


        # Decide whether to apply pushforward behavior in this pass.
        apply_pushforward = bool(self.pushforward_enabled and (self.training or self.pushforward_apply_in_validation))
        if apply_pushforward and self.pushforward_mode == "paper":
            if apply_noise:
                raise ValueError(
                    "rollout.noise.enabled is incompatible with rollout.pushforward.mode='paper' (paper uses pushforward instead of Gaussian noise)."
                )
            return self._paper_pushforward_rollout_forward_loss(
                y_i=y_i,
                dt_in_3=dt_in_3,
                y_j=y_j,
                g=g,
                num_steps=num_steps,
                record_step_abslog=record_step_abslog,
            )
        step_weights = compute_rollout_loss_weights(
            num_steps=num_steps,
            weighting=self.loss_weighting,
            discount=self.loss_discount,
            device=device,
            dtype=torch.float32,
        )

        # Pushforward mode (paper-inspired, backward-compatible):
        #   - stopgrad through the propagated state (implemented by detaching the *input state*
        #     for steps/chunks > 0)
        #   - optional burn-in region: first N steps contribute zero loss weight, but are still
        #     propagated
        #
        # By default, burn_in_steps=1 when skip_first_step_loss=True, matching prior behavior.
        burn_in_steps = int(self.pushforward_burn_in_steps) if apply_pushforward else 0
        if burn_in_steps < 0:
            raise ValueError(f"Invalid pushforward burn_in_steps={burn_in_steps}")
        if burn_in_steps >= num_steps:
            raise ValueError(
                f"pushforward.burn_in_steps={burn_in_steps} must be < rollout steps={num_steps} "
                f"(requested={num_steps_req}, K={K})"
            )
        if apply_pushforward and burn_in_steps > 0:
            step_weights = step_weights.clone()
            step_weights[:burn_in_steps] = 0.0
            if not (float(step_weights.sum().item()) > 0.0):
                raise ValueError("Internal error: pushforward step_weights sum became 0 after burn-in masking.")
            step_weights = step_weights / step_weights.sum()

        state = y_i

        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_phys = torch.zeros((), device=device, dtype=torch.float32)
        total_z = torch.zeros((), device=device, dtype=torch.float32)
        total_abslog = torch.zeros((), device=device, dtype=torch.float32)

        step_abslog_vec = torch.zeros((num_steps,), device=device, dtype=torch.float32) if record_step_abslog else None

        # ---- Chunked temporal bundling (one model call per chunk) ----
        if self.bundling_enabled and self.bundling_chunk_size > 1:
            step = 0
            while step < num_steps:
                L = min(int(self.bundling_chunk_size), int(num_steps - step))

                # dt_in is incremental in rollout mode. Build cumulative dt within this chunk.
                dt_inc_norm = dt_in_3[:, step : step + L, :]              # [B,L,1]
                dt_inc_phys = self._dt_norm_to_phys(dt_inc_norm)          # [B,L]
                dt_cum_phys = torch.cumsum(dt_inc_phys, dim=1)            # [B,L]
                dt_cum_norm = self._dt_phys_to_norm(dt_cum_phys)          # [B,L]
                dt_cum_norm_3 = dt_cum_norm.unsqueeze(-1)                 # [B,L,1]

                # Pushforward: stopgrad through propagated state across *chunk boundaries*.
                # Within a chunk, we do not propagate intermediate states, so there is no
                # notion of per-step pushforward inside the chunk.
                state_in = state.detach() if (apply_pushforward and step > 0) else state
                pred_chunk = self(state_in, dt_cum_norm_3, g)             # [B,L,S]
                if pred_chunk.ndim != 3 or pred_chunk.shape[1] != L:
                    raise RuntimeError(f"Chunked rollout expected pred [B,{L},S]; got {tuple(pred_chunk.shape)}")

                # Losses: computed per-step to support step-wise weights.
                for j in range(L):
                    step_idx = step + j
                    pred = pred_chunk[:, j, :]
                    true = y_j[:, step_idx, :]
                    step_comps = self.criterion(pred.unsqueeze(1), true.unsqueeze(1), return_components=True)

                    w = step_weights[step_idx]
                    total_loss = total_loss + w * step_comps["total"]
                    total_phys = total_phys + w * step_comps["phys"]
                    total_z = total_z + w * step_comps["z"]
                    total_abslog = total_abslog + w * step_comps["mean_abs_log10"]

                    if step_abslog_vec is not None:
                        step_abslog_vec[step_idx] = step_comps["mean_abs_log10"].detach().to(torch.float32)

                # Propagate only the last predicted state in the chunk.
                last_pred = pred_chunk[:, -1, :]
                next_state = self._apply_state_constraints(last_pred, apply_noise=apply_noise)

                # Truncated BPTT can only be applied at chunk boundaries (we do not have propagated intermediate states).
                if (not apply_pushforward) and self.detach_every > 0 and (step + L) % self.detach_every == 0:
                    next_state = next_state.detach()

                state = next_state
                step += L

        # ---- Standard step-wise rollout (one model call per step) ----
        else:
            for step_idx in range(num_steps):
                dt_step = dt_in_3[:, step_idx : step_idx + 1, :]

                # Pushforward: stopgrad through propagated state.
                state_in = state.detach() if (apply_pushforward and step_idx > 0) else state

                pred = self(state_in, dt_step, g).squeeze(1)
                true = y_j[:, step_idx, :]

                step_comps = self.criterion(pred.unsqueeze(1), true.unsqueeze(1), return_components=True)

                w = step_weights[step_idx]
                total_loss = total_loss + w * step_comps["total"]
                total_phys = total_phys + w * step_comps["phys"]
                total_z = total_z + w * step_comps["z"]
                total_abslog = total_abslog + w * step_comps["mean_abs_log10"]

                if step_abslog_vec is not None:
                    step_abslog_vec[step_idx] = step_comps["mean_abs_log10"].detach().to(torch.float32)

                # Prepare for next step (this is the *propagated* state)
                next_state = self._apply_state_constraints(pred, apply_noise=apply_noise)

                # Optional truncated BPTT (ignored in pushforward mode, which already enforces stopgrad-by-input)
                if (not apply_pushforward) and self.detach_every > 0 and (step_idx + 1) % self.detach_every == 0:
                    next_state = next_state.detach()

                state = next_state

        comps = {
            "total": total_loss,
            "phys": total_phys,
            "z": total_z,
            "mean_abs_log10": total_abslog,
            "num_steps": torch.tensor(num_steps, device=device, dtype=torch.long),
        }

        final_abslog = None
        if record_step_abslog:
            # Final-step abslog uses the CONSTRAINED state that is actually propagated.
            final_pred_log = self.criterion.z_to_log10(state)
            final_true_log = self.criterion.z_to_log10(y_j[:, num_steps - 1, :])
            final_abslog = (final_pred_log - final_true_log).abs().mean().to(torch.float32)

        return total_loss, comps, step_abslog_vec, final_abslog

    # --------------------------- Optimizer / Scheduler --------------------------

    def configure_optimizers(self):
        tcfg = self.cfg.get("training", {})
        opt_name = str(tcfg.get("optimizer", "adamw")).lower().strip()

        if opt_name == "adamw":
            try:
                has_fused_flag = "fused" in inspect.signature(torch.optim.AdamW).parameters
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

        scheds = []
        if self.warmup_epochs > 0:
            scheds.append(
                LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=max(1, self.warmup_epochs))
            )

        max_epochs = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        t_max = max(1, max_epochs - self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.min_lr)

        scheduler = cosine if not scheds else SequentialLR(
            optimizer, schedulers=[scheds[0], cosine], milestones=[self.warmup_epochs]
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": CKPT_MONITOR}]

    # --------------------------- Fit hooks -------------------------------------

    def on_fit_start(self) -> None:
        tr = getattr(self, "trainer", None)
        if tr is None or (not tr.is_global_zero) or getattr(tr, "sanity_checking", False):
            return
        self._csv = EpochCSVWriter(self.work_dir / "metrics.csv", resume=self._resume)
        self._csv.ensure_initialized()

    # --------------------------- Training --------------------------------------

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.time()
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device
        self._t_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_wsum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_z_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._t_abslog_sum = torch.zeros((), dtype=torch.float64, device=dev)

    def training_step(self, batch, batch_idx: int):
        y_i, dt_in, y_j, g = self._unpack(batch)

        if self.rollout_enabled:
            num_steps = self._get_rollout_steps_for_epoch()
            # Paper pushforward mode (K*2 unroll with stop-grad on the pushforward state)
            # is incompatible with extra noise injection; the pushforward distribution is the perturbation.
            apply_noise = True
            if self.pushforward_enabled and getattr(self, 'pushforward_mode', 'legacy') == 'paper':
                apply_noise = False

            loss, comps, _step_abslog, _final_abslog = self._rollout_forward_loss(
                y_i, dt_in, y_j, g,
                num_steps_req=num_steps,
                apply_noise=apply_noise,
                record_step_abslog=False,
            )
        else:
            loss, comps = self._onestep_forward_loss(y_i, dt_in, y_j, g)

        if not torch.isfinite(loss):
            raise ValueError(f"[train] Non-finite total loss at batch {batch_idx}")
        if not torch.isfinite(comps["phys"]):
            raise ValueError(f"[train] Non-finite phys loss at batch {batch_idx}")
        if not torch.isfinite(comps["z"]):
            raise ValueError(f"[train] Non-finite z loss at batch {batch_idx}")
        if "mean_abs_log10" in comps and (not torch.isfinite(comps["mean_abs_log10"])):
            raise ValueError(f"[train] Non-finite mean_abs_log10 at batch {batch_idx}")

        with torch.no_grad():
            B = int(y_j.shape[0])
            S = int(y_j.shape[-1])
            used_steps = int(comps.get("num_steps", torch.tensor(y_j.shape[1], device=y_j.device)).item())
            valid_pairs = B * used_steps
            if valid_pairs > 0:
                weight = torch.tensor(valid_pairs * S, device=y_j.device, dtype=torch.float64)
                self._t_sum += comps["total"].detach().to(torch.float64) * weight
                self._t_phys_sum += comps["phys"].detach().to(torch.float64) * weight
                self._t_z_sum += comps["z"].detach().to(torch.float64) * weight
                if "mean_abs_log10" in comps:
                    self._t_abslog_sum += comps["mean_abs_log10"].detach().to(torch.float64) * weight
                self._t_wsum += weight

        return loss

    def on_train_epoch_end(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            for t in (self._t_sum, self._t_phys_sum, self._t_z_sum, self._t_abslog_sum, self._t_wsum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if float(self._t_wsum.item()) <= 0.0:
            for name in ("_t_sum", "_t_phys_sum", "_t_z_sum", "_t_abslog_sum", "_t_wsum"):
                setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        w = self._t_wsum
        train_loss = (self._t_sum / w).to(torch.float32)
        train_phys = (self._t_phys_sum / w).to(torch.float32)
        train_z = (self._t_z_sum / w).to(torch.float32)

        mean_abslog = (self._t_abslog_sum / w).to(torch.float64)
        mean_abslog = torch.clamp(mean_abslog, min=0.0)
        train_mult_err_proxy = torch.expm1(mean_abslog * LN10).to(torch.float32)

        self._last_train_metrics = {
            "train_loss": float(train_loss.detach().cpu().item()),
            "train_phys": float(train_phys.detach().cpu().item()),
            "train_z": float(train_z.detach().cpu().item()),
            "train_mult_err_proxy": float(train_mult_err_proxy.detach().cpu().item()),
        }
        self._last_train_epoch_idx = int(self.current_epoch)

        self.log("train_loss", train_loss, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_phys", train_phys, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_z", train_z, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("train_mult_err_proxy", train_mult_err_proxy, prog_bar=False, on_epoch=True, sync_dist=False)

        self._maybe_report_epoch()

        for name in ("_t_sum", "_t_phys_sum", "_t_z_sum", "_t_abslog_sum", "_t_wsum"):
            setattr(self, name, torch.zeros_like(getattr(self, name)))

    # --------------------------- Validation ------------------------------------

    def on_validation_epoch_start(self) -> None:
        dev = self.device if hasattr(self, "device") else next(self.parameters()).device
        self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)
        self._v_abslog_sum = torch.zeros((), dtype=torch.float64, device=dev)

        if self.val_rollout_enabled:
            K = int(self.val_rollout_steps)
            self._vr_step_abslog_sum = torch.zeros((K,), dtype=torch.float64, device=dev)
            self._vr_step_count = torch.zeros((K,), dtype=torch.float64, device=dev)
            self._vr_final_abslog_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._vr_final_count = torch.zeros((), dtype=torch.float64, device=dev)

    def validation_step(self, batch, batch_idx: int):
        y_i, dt_in, y_j, g = self._unpack(batch)

        if self.val_rollout_enabled:
            loss, comps, step_abslog_vec, final_abslog = self._rollout_forward_loss(
                y_i, dt_in, y_j, g,
                num_steps_req=self.val_rollout_steps,
                apply_noise=False,
                record_step_abslog=True,
            )

            if step_abslog_vec is None or final_abslog is None:
                raise RuntimeError("Internal error: rollout validation expected step/final metrics.")

            # Accumulate per-step abslog (raw per-step prediction quality)
            B = int(y_j.shape[0])
            used_steps = int(comps["num_steps"].item())
            for step_idx in range(min(used_steps, int(self.val_rollout_steps))):
                self._vr_step_abslog_sum[step_idx] += float(step_abslog_vec[step_idx].item()) * B
                self._vr_step_count[step_idx] += B

            # Final-step abslog (constrained propagated state)
            self._vr_final_abslog_sum += float(final_abslog.item()) * B
            self._vr_final_count += B

        else:
            # One-step validation (legacy)
            pred = self(y_i, dt_in, g)
            if pred.shape != y_j.shape:
                raise RuntimeError(f"[val] Pred/GT shape mismatch: {pred.shape} vs {y_j.shape}")
            comps = self.criterion(pred, y_j, return_components=True)
            loss = comps["total"]
            comps["num_steps"] = torch.tensor(y_j.shape[1], device=y_j.device, dtype=torch.long)

        total = comps["total"].detach()
        phys = comps["phys"].detach()
        zstb = comps["z"].detach()
        abslog = comps.get("mean_abs_log10")
        if abslog is not None:
            abslog = abslog.detach()

        if not torch.isfinite(total):
            raise ValueError(f"[val] Non-finite total loss at batch {batch_idx}")
        if not torch.isfinite(phys):
            raise ValueError(f"[val] Non-finite phys loss at batch {batch_idx}")
        if not torch.isfinite(zstb):
            raise ValueError(f"[val] Non-finite z loss at batch {batch_idx}")
        if abslog is not None and (not torch.isfinite(abslog)):
            raise ValueError(f"[val] Non-finite mean_abs_log10 at batch {batch_idx}")

        B = int(y_j.shape[0])
        S = int(y_j.shape[-1])
        used_steps = int(comps.get("num_steps", torch.tensor(y_j.shape[1], device=y_j.device)).item())
        valid_pairs = B * used_steps

        if valid_pairs <= 0:
            raise ValueError(f"[val] Zero valid pairs in batch {batch_idx}")

        weight = torch.tensor(valid_pairs * S, device=y_j.device, dtype=torch.float64)

        self._v_sum += total.to(torch.float64) * weight
        self._v_phys_sum += phys.to(torch.float64) * weight
        self._v_z_sum += zstb.to(torch.float64) * weight
        if abslog is not None:
            self._v_abslog_sum += abslog.to(torch.float64) * weight
        self._v_wsum += weight

        return total

    def on_validation_epoch_end(self) -> None:
        tr = getattr(self, "trainer", None)
        if tr is not None and getattr(tr, "sanity_checking", False):
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_abslog_sum", "_v_wsum"):
                if hasattr(self, name):
                    setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        import torch.distributed as dist

        if not hasattr(self, "_v_wsum"):
            dev = self.device if hasattr(self, "device") else next(self.parameters()).device
            self._v_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_phys_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_z_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_abslog_sum = torch.zeros((), dtype=torch.float64, device=dev)
            self._v_wsum = torch.zeros((), dtype=torch.float64, device=dev)

        if dist.is_available() and dist.is_initialized():
            for t in (self._v_sum, self._v_phys_sum, self._v_z_sum, self._v_abslog_sum, self._v_wsum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if float(self._v_wsum.item()) <= 0.0:
            for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_abslog_sum", "_v_wsum"):
                setattr(self, name, torch.zeros_like(getattr(self, name)))
            return

        w = self._v_wsum
        val_loss = (self._v_sum / w).to(torch.float32)
        val_phys = (self._v_phys_sum / w).to(torch.float32)
        val_z = (self._v_z_sum / w).to(torch.float32)

        mean_abslog = (self._v_abslog_sum / w).to(torch.float64)
        mean_abslog = torch.clamp(mean_abslog, min=0.0)
        val_mult_err_proxy = torch.expm1(mean_abslog * LN10).to(torch.float32)

        self._last_val_metrics = {
            "val_loss": float(val_loss.detach().cpu().item()),
            "val_phys": float(val_phys.detach().cpu().item()),
            "val_z": float(val_z.detach().cpu().item()),
            "val_mult_err_proxy": float(val_mult_err_proxy.detach().cpu().item()),
        }
        self._last_val_epoch_idx = int(self.current_epoch)

        # When rollout is enabled, val_loss/val_phys/val_z are the rollout metrics.
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_phys", val_phys, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_z", val_z, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_mult_err_proxy", val_mult_err_proxy, prog_bar=False, on_epoch=True, sync_dist=False)

        # Rollout-specific validation logs (per-step and final) if enabled.
        final_abslog = None
        final_mult_err = None

        if self.val_rollout_enabled and hasattr(self, "_vr_step_count"):
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self._vr_step_abslog_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._vr_step_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._vr_final_abslog_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._vr_final_count, op=dist.ReduceOp.SUM)

            for step_idx in range(int(self.val_rollout_steps)):
                count = self._vr_step_count[step_idx]
                if float(count.item()) > 0.0:
                    step_abslog = (self._vr_step_abslog_sum[step_idx] / count).to(torch.float32)
                    step_mult_err = torch.expm1(step_abslog * LN10)
                    self.log(
                        f"val_rollout_step{step_idx}_abslog",
                        step_abslog,
                        prog_bar=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    self.log(
                        f"val_rollout_step{step_idx}_mult_err",
                        step_mult_err,
                        prog_bar=False,
                        on_epoch=True,
                        sync_dist=False,
                    )

            if float(self._vr_final_count.item()) > 0.0:
                final_abslog = (self._vr_final_abslog_sum / self._vr_final_count).to(torch.float32)
                final_mult_err = torch.expm1(final_abslog * LN10).to(torch.float32)
                self.log("val_rollout_final_abslog", final_abslog, prog_bar=False, on_epoch=True, sync_dist=False)
                self.log("val_rollout_final_mult_err", final_mult_err, prog_bar=True, on_epoch=True, sync_dist=False)

                self._last_val_metrics["val_rollout_final_abslog"] = float(final_abslog.detach().cpu().item())
                self._last_val_metrics["val_rollout_final_mult_err"] = float(final_mult_err.detach().cpu().item())

        self._maybe_report_epoch()

        for name in ("_v_sum", "_v_phys_sum", "_v_z_sum", "_v_abslog_sum", "_v_wsum"):
            setattr(self, name, torch.zeros_like(getattr(self, name)))

    # --------------------------- Epoch Reporting --------------------------------

    def _maybe_report_epoch(self) -> None:
        """
        Report once per epoch, but only after the metrics for this epoch are available.

        Lightning hook ordering can vary across versions/strategies; this guards against
        printing/writing rows with stale train/val values.
        """
        tr = getattr(self, "trainer", None)
        if tr is None:
            return

        cur_epoch_idx = int(self.current_epoch)
        have_train = (self._last_train_epoch_idx == cur_epoch_idx)
        have_val = (self._last_val_epoch_idx == cur_epoch_idx)

        # If there is no validation loop, allow reporting with train-only.
        try:
            num_val_batches = getattr(tr, "num_val_batches", 0)
            has_val_loop = bool(num_val_batches) and (sum(num_val_batches) if isinstance(num_val_batches, (list, tuple)) else int(num_val_batches)) > 0
        except Exception:
            has_val_loop = True

        if have_train and (have_val or (not has_val_loop)):
            self._report_epoch()

    def _report_epoch(self) -> None:
        tr = getattr(self, "trainer", None)
        if tr is None or (not tr.is_global_zero) or getattr(tr, "sanity_checking", False):
            return
        if self._csv is None:
            self._csv = EpochCSVWriter(self.work_dir / "metrics.csv", resume=self._resume)
            self._csv.ensure_initialized()

        epoch_1idx = int(self.current_epoch) + 1
        step = int(getattr(tr, "global_step", 0) or 0)

        epoch_time = float("nan") if self._epoch_t0 is None else (time.time() - float(self._epoch_t0))

        lr: Optional[float] = None
        try:
            if tr.optimizers and tr.optimizers[0].param_groups:
                lr = float(tr.optimizers[0].param_groups[0].get("lr", None))
        except Exception:
            lr = None

        if lr is not None:
            self.log("lr", float(lr), prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)

        cur_epoch_idx = int(self.current_epoch)
        train_m = self._last_train_metrics if self._last_train_epoch_idx == cur_epoch_idx else {}
        val_m = self._last_val_metrics if self._last_val_epoch_idx == cur_epoch_idx else {}

        def fmt(x: Optional[float]) -> str:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return ""
            return f"{x:.6e}"

        row: dict[str, str] = {
            "epoch": str(epoch_1idx),
            "step": str(step),
            "wall_time": self._csv.wall_time_str(),
            "epoch_time_sec": "" if (not math.isfinite(epoch_time)) else f"{epoch_time:.6f}",
            "lr": "" if lr is None else f"{lr:.6e}",
            "train_loss": fmt(train_m.get("train_loss")),
            "train_phys": fmt(train_m.get("train_phys")),
            "train_z": fmt(train_m.get("train_z")),
            "train_mult_err_proxy": fmt(train_m.get("train_mult_err_proxy")),
            "val_loss": fmt(val_m.get("val_loss")),
            "val_phys": fmt(val_m.get("val_phys")),
            "val_z": fmt(val_m.get("val_z")),
            "val_mult_err_proxy": fmt(val_m.get("val_mult_err_proxy")),
            "val_rollout_final_abslog": fmt(val_m.get("val_rollout_final_abslog")),
            "val_rollout_final_mult_err": fmt(val_m.get("val_rollout_final_mult_err")),
        }
        self._csv.write_epoch_row(epoch_1idx, row)

        if epoch_1idx == self._last_reported_epoch:
            return

        def fmt_sci(x: Optional[float], width: int = 11) -> str:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return f"{'—':>{width}}"
            return f"{x:>{width}.3e}"

        def fmt_time(seconds: float) -> str:
            if not math.isfinite(seconds) or seconds < 0:
                seconds = 0.0
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:d}:{m:02d}:{s:02d}"
            return f"{m:d}:{s:02d}"

        if not self._printed_header:
            if self.val_rollout_enabled:
                self._logger.info("EPOCH | train(loss)  | val(loss)    | val(final_mult) | lr         | time")
                self._logger.info("----- | -----------  | -----------  | --------------- | ---------- | --------")
            else:
                self._logger.info("EPOCH | train       | val         | mult_err_proxy | lr         | time")
                self._logger.info("----- | ----------- | ----------- | -------------- | ---------- | --------")
            self._printed_header = True

        tr_loss = train_m.get("train_loss")
        va_loss = val_m.get("val_loss")
        va_final_mult = val_m.get("val_rollout_final_mult_err") if self.val_rollout_enabled else val_m.get("val_mult_err_proxy")

        epoch_str = f"E{epoch_1idx:04d}"
        line = (
            f"{epoch_str} | "
            f"{fmt_sci(tr_loss)} | "
            f"{fmt_sci(va_loss)} | "
            f"{fmt_sci(va_final_mult, width=14)} | "
            f"{fmt_sci(lr)} | "
            f"{fmt_time(epoch_time)}"
        )
        self._logger.info(line)
        self._last_reported_epoch = epoch_1idx


# =============================================================================
# Trainer Wrapper
# =============================================================================


class Trainer:
    """Compatibility wrapper: `from trainer import Trainer` -> .train() returns best monitored metric."""

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
        pl_precision_override: Any = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.pl_precision_override = pl_precision_override

        self.logger = logger or logging.getLogger("trainer")

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

        self.deterministic = bool(syscfg.get("deterministic", False))
        self.cudnn_benchmark = bool(syscfg.get("cudnn_benchmark", True)) and not self.deterministic

        self.use_swa = bool(tcfg.get("use_swa", False))
        self.swa_lrs = tcfg.get("swa_lrs", None)
        self.swa_epoch_start = tcfg.get("swa_epoch_start", 0.8)
        self.swa_annealing_epochs = int(tcfg.get("swa_annealing_epochs", 10))
        self.swa_annealing_strategy = str(tcfg.get("swa_annealing_strategy", "cos"))

        self.compile_backend = tcfg.get("torch_compile_backend", COMPILE_BACKEND)
        self.compile_mode = tcfg.get("torch_compile_mode", COMPILE_MODE)
        self.compile_dynamic = bool(tcfg.get("compile_dynamic", COMPILE_DYNAMIC))
        self.compile_fullgraph = bool(tcfg.get("compile_fullgraph", COMPILE_FULLGRAPH))

        self.limit_train_batches = int(tcfg.get("max_train_batches", DEF_MAX_TRAIN_BATCHES) or 0)
        self.limit_val_batches = int(tcfg.get("max_val_batches", DEF_MAX_VAL_BATCHES) or 0)

        # Log rollout config
        rollout_cfg = tcfg.get("rollout", {})
        if rollout_cfg.get("enabled", False):
            val_rollout = rollout_cfg.get("validation", {})
            self.logger.info(
                f"Rollout enabled: steps={rollout_cfg.get('steps', 4)}, "
                f"noise={rollout_cfg.get('noise', {}).get('enabled', False)}, "
                f"clip={rollout_cfg.get('clip', {}).get('enabled', False)}, "
                f"curriculum={rollout_cfg.get('curriculum', {}).get('enabled', False)}, "
                f"val_rollout_steps={val_rollout.get('steps', rollout_cfg.get('steps', 4))}"
            )

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

        env_resume = os.environ.get("RESUME", "").strip()
        if env_resume and env_resume.lower() != "auto":
            p = Path(env_resume)
            return str(p) if p.is_file() else None

        if env_resume.lower() == "auto" or bool(tcfg.get("auto_resume", True)):
            p = self.work_dir / "last.ckpt"
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
        ckpt_path = self._resolve_resume_ckpt()
        self.logger.info(f"resume: {ckpt_path if ckpt_path else 'fresh'}")

        self.model = self._maybe_compile(self.model)
        task = ModelTask(self.model, self.cfg, self.work_dir, resume=bool(ckpt_path))

        # --------------------- CHECKPOINT MONITOR SELECTION ---------------------
        # When rollout is enabled, checkpoint on final-step rollout abslog (true inference behavior).
        # Otherwise fall back to one-step val_loss.
        tcfg = self.cfg.get("training", {}) if isinstance(self.cfg, dict) else {}
        rollout_cfg = tcfg.get("rollout", {}) if isinstance(tcfg, dict) else {}
        use_rollout_ckpt = bool(rollout_cfg.get("enabled", False))
        ckpt_monitor = "val_rollout_final_abslog" if use_rollout_ckpt else CKPT_MONITOR

        self.logger.info(
            f"checkpoint monitor: {ckpt_monitor} "
            f"({'rollout' if use_rollout_ckpt else 'one-step'})"
        )
        # -----------------------------------------------------------------------

        ckpt_cb = ModelCheckpoint(
            dirpath=str(self.work_dir),
            filename=CKPT_FILENAME_BEST,
            monitor=ckpt_monitor,
            mode=CKPT_MODE,
            save_top_k=1,
            save_last=True,
            verbose=False,
        )
        callbacks = [ckpt_cb]

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

        trainer_kwargs: Dict[str, Any] = {}
        if self.limit_train_batches > 0:
            trainer_kwargs["limit_train_batches"] = self.limit_train_batches
        if self.limit_val_batches > 0:
            trainer_kwargs["limit_val_batches"] = self.limit_val_batches

        world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
        externally_launched = world_size > 1

        devices_env = os.getenv("PL_DEVICES", "").strip()
        if not externally_launched:
            devices = 1
            strategy = "auto"
            if devices_env:
                try:
                    req = int(devices_env)
                    if req != 1:
                        self.logger.warning(
                            "PL_DEVICES=%s requested, but non-DDP multi-device is not supported here; forcing devices=1. "
                            "Use torchrun/mpirun/srun for multi-GPU.",
                            devices_env,
                        )
                except Exception:
                    pass
        else:
            devices = 1
            strategy = "ddp"
            try:
                num_nodes = int(os.getenv("SLURM_NNODES", os.getenv("NUM_NODES", "1")))
            except Exception:
                num_nodes = 1
            if num_nodes > 1:
                trainer_kwargs["num_nodes"] = num_nodes

        trainer_kwargs.update(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
        )

        base_kwargs: Dict[str, Any] = dict(
            max_epochs=self.epochs,
            accumulate_grad_batches=max(1, self.accumulate),
            gradient_clip_val=float(self.grad_clip),
            logger=False,
            callbacks=callbacks,
            enable_checkpointing=ENABLE_CHECKPOINTING,
            enable_progress_bar=ENABLE_PROGRESS_BAR,
            enable_model_summary=ENABLE_MODEL_SUMMARY,
            detect_anomaly=DETECT_ANOMALY,
            inference_mode=INFERENCE_MODE,
            benchmark=self.cudnn_benchmark,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            # Avoid Lightning sanity-check issues with a rollout-only monitor.
            num_sanity_val_steps=0 if use_rollout_ckpt else 2,
            **trainer_kwargs,
        )

        trainer_params = set(inspect.signature(pl.Trainer).parameters.keys())
        filtered = {k: v for k, v in base_kwargs.items() if k in trainer_params}
        if "deterministic" in trainer_params:
            filtered["deterministic"] = self.deterministic

        pl_trainer = pl.Trainer(**filtered)

        pl_trainer.fit(
            task,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=ckpt_path,
        )

        best = ckpt_cb.best_model_score
        return float(best.item()) if best is not None else float("nan")
