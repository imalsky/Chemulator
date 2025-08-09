#!/usr/bin/env python3
# src/hyperparameter_tuning.py
"""
Aggressive Optuna HPO for LiLaN models, callable from main.py (--tune).
- No circular imports (does NOT import main.py).
- Preprocesses data on-demand (using DataPreprocessor) if needed.
- Every search dimension is gated by a GLOBAL at the top: comment out to disable searching that space.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch

from utils.utils import (
    ensure_directories,
    load_json_config,
    save_json,
    load_json,
    seed_everything,
)
from utils.hardware import setup_device
from data.preprocessor import DataPreprocessor
from data.dataset import SequenceDataset
from data.normalizer import NormalizationHelper
from models.model import create_model
from training.trainer import Trainer

# =============================================================================
# GLOBAL HPO SETTINGS (comment out a variable to disable that search dimension)
# =============================================================================

# ---- Study / sampler / pruner ----
HPO_SEED: int = 42
HPO_STUDY_PREFIX: str = "lilan_hpo"

# Hyperband resource bounds (fallbacks if training.hpo_* not in config)
HPO_DEFAULT_MIN_EPOCHS: int = 5
HPO_DEFAULT_MAX_EPOCHS: int = 50
HPO_REDUCTION_FACTOR: int = 4  # aggressive down-selection

# TPE exploration settings
HPO_TPE_STARTUP: int = 20
HPO_TPE_EI_CANDIDATES: int = 64

# ---- Search spaces (comment out to freeze at config value) ----
MODEL_TYPE_CHOICES = ["linear_latent", "linear_latent_mixture"]

LATENT_CHOICES = [32, 64, 128, 256]

# Encoder/decoder depth ranges (inclusive)
ENC_DEPTH_RANGE: Tuple[int, int] = (2, 6)
DEC_DEPTH_RANGE: Tuple[int, int] = (2, 6)

# Per-layer base widths and optional growth factors
BASE_WIDTH_CHOICES = [128, 256, 512, 1024, 2048]
GROWTH_RANGE: Tuple[float, float] = (0.3, 0.8)  # if removed -> constant width per layer

# Low-rank generator
GEN_RANK_CHOICES = [4, 8, 12, 16, 24, 32, 48, 64]

# Mixture-of-experts
MIXTURE_K_CHOICES = [2, 4, 8]
MIXTURE_TEMPERATURE_RANGE: Tuple[float, float] = (0.1, 3.0)  # log-uniform

# Time warp
TIME_WARP_ENABLE_CHOICES = [True, False]
J_TERMS_CHOICES = [2, 4, 8]
WARP_HIDDEN_CHOICES = [64, 128, 256, 512]

# Activations / dropout
ACTIVATION_CHOICES = ["gelu", "tanh"]
DROPOUT_RANGE: Tuple[float, float, float] = (0.0, 0.2, 0.05)  # (min, max, step)

# Training hyperparams
#LEARNING_RATE_RANGE: Tuple[float, float] = (1e-6, 3e-3)    # log-uniform
#WEIGHT_DECAY_RANGE: Tuple[float, float] = (1e-8, 1e-2)     # log-uniform
#OPTIMIZER_CHOICES = ["adamw", "lion"]

# Batch size as power of 2
#BATCH_POWER_RANGE: Tuple[int, int] = (8, 13)  # -> 2**8 .. 2**13 (256..8192)

# Mixture regularizers (only if mixture type used)
LAMBDA_ENTROPY_RANGE: Tuple[float, float] = (1e-6, 1e-2)   # log-uniform
LAMBDA_DIVERSITY_RANGE: Tuple[float, float] = (1e-6, 1e-2) # log-uniform

# Gate temperature schedule (for PrunableTrainer)
T_START_RANGE: Tuple[float, float] = (0.5, 2.0)
T_END_RANGE: Tuple[float, float] = (0.1, 1.0)
T_ANNEAL_FRAC_RANGE: Tuple[float, float] = (0.3, 0.9)

# Early stopping (if Trainer respects it)
#EARLY_STOP_PATIENCE_RANGE: Tuple[int, int] = (10, 40)

# Loss function (leave as ["mse"] unless you actually support others)
#LOSS_CHOICES = ["mse"]

# =============================================================================
# Internal utilities
# =============================================================================

def _existing(name: str) -> Any:
    """Return a global by name if present; None otherwise. Lets you 'disable by commenting out'."""
    return globals().get(name, None)

def _seq_dir_from_config(cfg: Dict[str, Any]) -> Path:
    base = Path(cfg["paths"]["processed_data_dir"])
    tag = "seq_mode" if cfg["data"].get("sequence_mode", False) else "row_mode"
    return base / tag

def _ensure_preprocessed(cfg: Dict[str, Any], processed_dir: Path, logger: logging.Logger) -> None:
    """Preprocess if shard index or normalization is missing."""
    shard_index = processed_dir / "shard_index.json"
    norm_json = processed_dir / "normalization.json"
    if shard_index.exists() and norm_json.exists():
        logger.info(f"Found preprocessed data at {processed_dir}.")
        return

    # Validate raw files first
    missing = [str(p) for p in cfg["paths"]["raw_data_files"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw data files: {missing}")

    logger.info("Preprocessing data for HPO...")
    dp = DataPreprocessor(
        raw_files=[Path(p) for p in cfg["paths"]["raw_data_files"]],
        output_dir=processed_dir,
        config=cfg,
    )
    dp.process_to_npy_shards()

def _read_norm_stats(processed_dir: Path) -> Dict[str, Any]:
    p = processed_dir / "normalization.json"
    if not p.exists():
        raise FileNotFoundError(f"Normalization stats missing: {p}")
    return load_json(p)

# =============================================================================
# Trainer with pruning hook
# =============================================================================

class _PrunableTrainer(Trainer):
    """Trainer that supports Optuna pruning callbacks."""
    def __init__(self, *args, epoch_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = epoch_callback

    def _run_training_loop(self):
        best_train_loss = float("inf")

        mix_sched = self.train_config.get("mixture_temperature_schedule", {})
        t_start = float(mix_sched.get("start", 1.0))
        t_end = float(mix_sched.get("end", 0.3))
        t_anneal_frac = float(mix_sched.get("anneal_frac", 0.6))

        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Update gate temperature if model supports it
            if hasattr(self.model, "set_gate_temperature") and getattr(self.model, "K", 1) > 1:
                progress = min(1.0, (epoch - 1) / max(1, int(self.train_config["epochs"] * t_anneal_frac)))
                temp = t_start + (t_end - t_start) * progress
                self.model.set_gate_temperature(temp)

            # Train + validate
            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._validate()

            # Scheduler
            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)

            # Prune?
            if self.epoch_callback:
                loss_for_pruning = val_loss if self.has_validation else train_loss
                if self.epoch_callback(epoch, loss_for_pruning):
                    self.logger.info(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

            # Early stopping
            if self.has_validation:
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if train_loss < (best_train_loss - self.min_delta):
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                    self.best_epoch = epoch
                    self._save_best_model()

# =============================================================================
# Suggestion helpers (respect "disable by commenting globals out")
# =============================================================================

def _maybe_cat(trial: optuna.Trial, name: str, choices: Optional[List[Any]], current: Any) -> Any:
    if choices:
        return trial.suggest_categorical(name, choices)
    return current

def _maybe_int(trial: optuna.Trial, name: str, rng: Optional[Tuple[int, int]], current: int) -> int:
    if rng:
        low, high = rng
        return trial.suggest_int(name, low, high)
    return current

def _maybe_float(trial: optuna.Trial, name: str,
                 rng: Optional[Tuple[float, float]],
                 current: float, *, log: bool = False,
                 step: Optional[float] = None) -> float:
    if rng:
        low, high = rng
        if step is not None:
            return trial.suggest_float(name, low, high, step=step)
        return trial.suggest_float(name, low, high, log=log)
    return current

# =============================================================================
# The public entry point (called by main.py)
# =============================================================================

def optimize_hyperparameters(
    config_path: Path,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    use_hyperband: bool = True,
) -> optuna.Study:
    """
    Run Optuna HPO. Safe to call from main.py's --tune path.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Load and (if needed) preprocess data ---
    base_config = load_json_config(config_path)
    processed_dir = _seq_dir_from_config(base_config)
    ensure_directories(processed_dir)

    _ensure_preprocessed(base_config, processed_dir, logger)
    norm_stats = _read_norm_stats(processed_dir)

    # --- Device + normalization helper ---
    device = setup_device()
    norm_helper = NormalizationHelper(stats=norm_stats, device=device, config=base_config)

    # --- Epoch budgets & pruner ---
    min_epochs = int(base_config.get("training", {}).get("hpo_min_epochs", _existing("HPO_DEFAULT_MIN_EPOCHS") or 10))
    max_epochs = int(base_config.get("training", {}).get("hpo_max_epochs", _existing("HPO_DEFAULT_MAX_EPOCHS") or 200))
    pruner = None
    if use_hyperband:
        pruner = HyperbandPruner(
            min_resource=min_epochs,
            max_resource=max_epochs,
            reduction_factor=int(_existing("HPO_REDUCTION_FACTOR") or 4),
        )
        logger.info(f"Hyperband: min={min_epochs}, max={max_epochs}, rf={_existing('HPO_REDUCTION_FACTOR') or 4}")

    # --- Sampler ---
    sampler = TPESampler(
        seed=int(_existing("HPO_SEED") or 42),
        multivariate=True,
        group=True,                         # co-sample related params
        constant_liar=True,
        warn_independent_sampling=False,    # suppress fallback warnings
        n_startup_trials=int(_existing("HPO_TPE_STARTUP") or 10),
        n_ei_candidates=int(_existing("HPO_TPE_EI_CANDIDATES") or 24),
        consider_endpoints=True,
    )

    # --- Study ---
    if study_name is None:
        study_name = f"{_existing('HPO_STUDY_PREFIX') or 'hpo'}_{time.strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
    )

    # --- Objective (one trial) ---
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_config)

        # Mark this run as HPO so Trainer skips per-epoch export and exports once at end.
        cfg.setdefault("optuna", {})["enabled"] = True

        # Ensure export is allowed (Trainer will export once at end for HPO).
        cfg.setdefault("system", {}).setdefault("use_torch_export", True)

        # ----- Model type -----
        cfg["model"]["type"] = _maybe_cat(trial, "model_type", _existing("MODEL_TYPE_CHOICES"), cfg["model"]["type"])

        # ----- Core dims -----
        cfg["model"]["latent_dim"] = _maybe_cat(trial, "latent_dim", _existing("LATENT_CHOICES"), cfg["model"].get("latent_dim", 64))

        # ----- Depths -----
        enc_depth = _maybe_int(trial, "encoder_depth", _existing("ENC_DEPTH_RANGE"), len(cfg["model"].get("encoder_layers", [])) or 3)
        dec_depth = _maybe_int(trial, "decoder_depth", _existing("DEC_DEPTH_RANGE"), len(cfg["model"].get("decoder_layers", [])) or 3)

        # ----- Widths -----
        base_width = _maybe_cat(trial, "base_width", _existing("BASE_WIDTH_CHOICES"), (cfg["model"].get("encoder_layers") or [256])[0])
        growth_rng = _existing("GROWTH_RANGE")

        if growth_rng:
            enc_widths: List[int] = []
            for i in range(enc_depth):
                g = trial.suggest_float(f"enc_growth_{i}", growth_rng[0], growth_rng[1])
                enc_widths.append(min(int(base_width * (1 + i * g)), 2048))
            dec_widths: List[int] = []
            for i in range(dec_depth):
                g = trial.suggest_float(f"dec_growth_{i}", growth_rng[0], growth_rng[1])
                dec_widths.append(min(int(base_width * (1 + (dec_depth - i - 1) * g)), 2048))
        else:
            enc_widths = [int(base_width)] * enc_depth
            dec_widths = [int(base_width)] * dec_depth

        cfg["model"]["encoder_layers"] = enc_widths
        cfg["model"]["decoder_layers"] = dec_widths

        # ----- Generator rank -----
        cfg["model"]["generator"] = {
            "rank": _maybe_cat(trial, "generator_rank", _existing("GEN_RANK_CHOICES"), (cfg["model"].get("generator", {}) or {}).get("rank", 8))
        }

        # ----- Mixture settings -----
        if cfg["model"]["type"] == "linear_latent_mixture":
            K = _maybe_cat(trial, "mixture_K", _existing("MIXTURE_K_CHOICES"), (cfg["model"].get("mixture", {}) or {}).get("K", 4))
            cfg["model"]["mixture"] = {"K": K}

            # Gate temperature (model-time temperature, not the schedule)
            temp_rng = _existing("MIXTURE_TEMPERATURE_RANGE")
            if temp_rng:
                cfg["model"]["mixture"]["temperature"] = trial.suggest_float("mixture_temperature", temp_rng[0], temp_rng[1], log=True)

            # Regularizers
            le_rng = _existing("LAMBDA_ENTROPY_RANGE")
            ld_rng = _existing("LAMBDA_DIVERSITY_RANGE")
            if le_rng and ld_rng:
                cfg["training"]["regularization"] = {
                    "lambda_entropy": trial.suggest_float("lambda_entropy", le_rng[0], le_rng[1], log=True),
                    "lambda_diversity": trial.suggest_float("lambda_diversity", ld_rng[0], ld_rng[1], log=True),
                }
            else:
                cfg["training"]["regularization"] = {"lambda_entropy": 0.0, "lambda_diversity": 0.0}
        else:
            cfg["model"].pop("mixture", None)
            cfg["training"]["regularization"] = {"lambda_entropy": 0.0, "lambda_diversity": 0.0}

        # ----- Time warp -----
        tw_enable_choices = _existing("TIME_WARP_ENABLE_CHOICES")
        if tw_enable_choices is not None:
            use_warp = trial.suggest_categorical("use_time_warp", tw_enable_choices)
            if use_warp:
                cfg["model"]["time_warp"] = {
                    "enabled": True,
                    "J_terms": _maybe_cat(trial, "J_terms", _existing("J_TERMS_CHOICES"), (cfg["model"].get("time_warp", {}) or {}).get("J_terms", 3)),
                    "hidden_dim": _maybe_cat(trial, "warp_hidden_dim", _existing("WARP_HIDDEN_CHOICES"), (cfg["model"].get("time_warp", {}) or {}).get("hidden_dim", 128)),
                }
            else:
                cfg["model"]["time_warp"] = {"enabled": False}
        # else: leave whatever is in the config as-is

        # ----- Activation / dropout / loss -----
        cfg["model"]["activation"] = _maybe_cat(trial, "activation", _existing("ACTIVATION_CHOICES"), cfg["model"].get("activation", "gelu"))
        dr = _existing("DROPOUT_RANGE")
        cfg["model"]["dropout"] = _maybe_float(
            trial, "dropout", (dr[0], dr[1]) if dr else None, cfg["model"].get("dropout", 0.0),
            step=(dr[2] if dr else None)
        )
        cfg["training"]["loss"] = _maybe_cat(trial, "loss", _existing("LOSS_CHOICES"), cfg["training"].get("loss", "mse"))

        # ----- Optimizer / LR / WD / Batch -----
        cfg["training"]["optimizer"] = _maybe_cat(trial, "optimizer", _existing("OPTIMIZER_CHOICES"), cfg["training"].get("optimizer", "adamw"))
        cfg["training"]["learning_rate"] = _maybe_float(trial, "learning_rate", _existing("LEARNING_RATE_RANGE"), cfg["training"].get("learning_rate", 1e-4), log=True)
        cfg["training"]["weight_decay"] = _maybe_float(trial, "weight_decay", _existing("WEIGHT_DECAY_RANGE"), cfg["training"].get("weight_decay", 1e-6), log=True)

        bp_rng = _existing("BATCH_POWER_RANGE")
        if bp_rng:
            batch_power = trial.suggest_int("batch_power", bp_rng[0], bp_rng[1])
            cfg["training"]["batch_size"] = 2 ** batch_power

        # ----- Gate temperature schedule -----
        ts_rng = _existing("T_START_RANGE")
        te_rng = _existing("T_END_RANGE")
        taf_rng = _existing("T_ANNEAL_FRAC_RANGE")
        sched = cfg["training"].get("mixture_temperature_schedule", {"start": 1.0, "end": 0.3, "anneal_frac": 0.6})
        sched["start"] = _maybe_float(trial, "t_start", ts_rng, sched.get("start", 1.0))
        sched["end"] = _maybe_float(trial, "t_end", te_rng, sched.get("end", 0.3))
        sched["anneal_frac"] = _maybe_float(trial, "t_anneal_frac", taf_rng, sched.get("anneal_frac", 0.6))
        cfg["training"]["mixture_temperature_schedule"] = sched

        # ----- Early stopping (if used) -----
        cfg["training"]["early_stopping_patience"] = _maybe_int(
            trial, "early_stopping_patience", _existing("EARLY_STOP_PATIENCE_RANGE"), cfg["training"].get("early_stopping_patience", 30)
        )

        # ----- Epoch budget (pruner decides early exit) -----
        cfg["training"]["epochs"] = max_epochs

        # ----- Seed everything for reproducibility -----
        seed_everything(cfg.get("system", {}).get("seed", int(_existing("HPO_SEED") or 42)))

        # ----- Save dir for this trial -----
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}"
        save_dir = Path(base_config["paths"]["model_save_dir"]) / "optuna" / f"{timestamp}_{trial_id}"
        ensure_directories(save_dir)

        # ----- Datasets + model -----
        model = create_model(cfg, device)
        train_ds = SequenceDataset(processed_dir, "train", cfg, device, norm_stats)
        val_ds = SequenceDataset(processed_dir, "validation", cfg, device, norm_stats)

        # ----- Trainer with pruning -----
        min_ep = min_epochs
        pruning_cb = lambda epoch, val_loss: (trial.report(val_loss, epoch) or (epoch >= min_ep and trial.should_prune()))
        # Use dedicated subclass to keep your Trainer clean:
        prunable = _PrunableTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=None,
            config=cfg,
            save_dir=save_dir,
            device=device,
            norm_helper=NormalizationHelper(stats=norm_stats, device=device, config=cfg),
            epoch_callback=pruning_cb,
        )

        # Train
        best_val = prunable.train()

        # Persist trial info
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("best_epoch", prunable.best_epoch)
        save_json(cfg, save_dir / "config.json")

        return best_val

    # --- Run optimization ---
    logger.info(f"Starting HPO: trials={n_trials}, jobs={n_jobs}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    # --- Save results ---
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)

    best_cfg = study.best_trial.user_attrs.get("config", {})
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    results = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": best_cfg,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_file = results_dir / f"{study_name}_results.json"
    save_json(results, out_file)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {results['best_value']:.6f}")
    print(f"Best trial: #{results['best_trial_number']}")
    print(f"Completed: {results['n_trials_completed']}, Pruned: {results['n_trials_pruned']}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {out_file}")

    return study