#!/usr/bin/env python3
"""
Hyperparameter tuning for chemical kinetics models using Optuna.
Aggressive search for LiLaN (linear_latent / linear_latent_mixture).
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch

from utils.hardware import setup_device
from data.dataset import SequenceDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper
from utils.utils import (
    seed_everything,
    ensure_directories,
    load_json_config,
    save_json,
    load_json,
)

# =========================
# HPO GLOBAL DEFAULTS
# =========================
HPO_SEED: int = 42
HPO_STUDY_PREFIX: str = "lilan_hpo"

# Hyperband fallbacks if not provided in config["training"]
HPO_DEFAULT_MIN_EPOCHS: int = 10
HPO_DEFAULT_MAX_EPOCHS: int = 200
HPO_REDUCTION_FACTOR: int = 4  # more aggressive down-selection

# TPE sampler settings for broader exploration
HPO_TPE_STARTUP: int = 20
HPO_TPE_EI_CANDIDATES: int = 64

# Aggressive search space constants
LATENT_CHOICES = [32, 48, 64, 80, 96, 112, 128, 160]
BASE_WIDTH_CHOICES = [128, 256, 512, 768, 1024, 1536, 2048]
GEN_RANK_CHOICES = [4, 8, 12, 16, 24, 32, 48, 64]
MIXTURE_K_CHOICES = [2, 3, 4, 5, 6, 8]
WARP_HIDDEN_CHOICES = [64, 128, 256, 384, 512]
J_TERMS_CHOICES = [2, 3, 4, 5, 6, 8]
ENC_DEPTH_RANGE = (2, 6)  # inclusive
DEC_DEPTH_RANGE = (2, 6)  # inclusive
BATCH_POWER_RANGE = (8, 13)  # 2**8 .. 2**13  (256 .. 8192)


class OptunaPruningCallback:
    """Report intermediate values to Optuna for pruning."""
    def __init__(self, trial: optuna.Trial, min_epochs: int = 10):
        self.trial = trial
        self.min_epochs = min_epochs

    def __call__(self, epoch: int, val_loss: float) -> bool:
        self.trial.report(val_loss, epoch)
        if epoch < self.min_epochs:
            return False
        return self.trial.should_prune()


class HyperparameterTuner:
    """Manages hyperparameter optimization for chemical kinetics models."""
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.base_config = load_json_config(self.config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)

        # Preprocess once
        self.processed_dir = self._prepare_data()

        # Load normalization stats
        norm_stats_path = self.processed_dir / "normalization.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
        self.norm_stats = load_json(norm_stats_path)
        self.norm_helper = NormalizationHelper(
            stats=self.norm_stats,
            device=self.device,
            config=self.base_config,
        )

        # Ensure sequence mode
        self._assert_sequence_mode()

    def _prepare_data(self) -> Path:
        """Preprocess data if needed and return processed directory."""
        from main import ChemicalKineticsPipeline
        self.logger.info("Preparing data for hyperparameter optimization...")
        pipeline = ChemicalKineticsPipeline(self.base_config)
        pipeline.preprocess_data()
        return pipeline.processed_dir

    def _assert_sequence_mode(self) -> None:
        index_path = self.processed_dir / "shard_index.json"
        with open(index_path) as f:
            shard_index = json.load(f)
        if not bool(shard_index.get("sequence_mode", False)):
            raise ValueError("Hyperparameter tuning requires sequence mode data")

    def suggest_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate an aggressive trial configuration for LiLaN models."""
        cfg = copy.deepcopy(self.base_config)

        # ----- MODEL TYPE -----
        model_type = trial.suggest_categorical(
            "model_type", ["linear_latent", "linear_latent_mixture"]
        )
        cfg["model"]["type"] = model_type

        # ----- CORE DIMENSIONS -----
        cfg["model"]["latent_dim"] = trial.suggest_categorical(
            "latent_dim", LATENT_CHOICES
        )

        # ----- ENCODER / DECODER DEPTH + WIDTHS -----
        enc_depth = trial.suggest_int("encoder_depth", *ENC_DEPTH_RANGE)
        dec_depth = trial.suggest_int("decoder_depth", *DEC_DEPTH_RANGE)

        base_width = trial.suggest_categorical("base_width", BASE_WIDTH_CHOICES)

        # Progressive widths with per-layer growth factors; capped at 2048
        enc_widths = [
            min(
                int(base_width * (1 + i * trial.suggest_float(f"enc_growth_{i}", 0.3, 0.8))),
                2048,
            )
            for i in range(enc_depth)
        ]
        dec_widths = [
            min(
                int(base_width * (1 + (dec_depth - i - 1) * trial.suggest_float(f"dec_growth_{i}", 0.3, 0.8))),
                2048,
            )
            for i in range(dec_depth)
        ]

        cfg["model"]["encoder_layers"] = enc_widths
        cfg["model"]["decoder_layers"] = dec_widths

        # ----- GENERATOR RANK -----
        cfg["model"]["generator"] = {
            "rank": trial.suggest_categorical("generator_rank", GEN_RANK_CHOICES)
        }

        # ----- MIXTURE SETTINGS (if enabled) -----
        if model_type == "linear_latent_mixture":
            cfg["model"]["mixture"] = {
                "K": trial.suggest_categorical("mixture_K", MIXTURE_K_CHOICES),
                "temperature": trial.suggest_float("mixture_temperature", 0.1, 3.0, log=True),
            }
            # Open meaningful ranges for regularizers
            cfg["training"]["regularization"] = {
                "lambda_entropy": trial.suggest_float("lambda_entropy", 1e-6, 1e-2, log=True),
                "lambda_diversity": trial.suggest_float("lambda_diversity", 1e-6, 1e-2, log=True),
            }
        else:
            cfg["model"].pop("mixture", None)
            cfg["training"]["regularization"] = {"lambda_entropy": 0.0, "lambda_diversity": 0.0}

        # ----- TIME WARP -----
        use_warp = trial.suggest_categorical("use_time_warp", [True, False])
        if use_warp:
            cfg["model"]["time_warp"] = {
                "enabled": True,
                "J_terms": trial.suggest_categorical("J_terms", J_TERMS_CHOICES),
                "hidden_dim": trial.suggest_categorical("warp_hidden_dim", WARP_HIDDEN_CHOICES),
            }
        else:
            cfg["model"]["time_warp"] = {"enabled": False}

        # ----- ACTIVATION / DROPOUT -----
        cfg["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "tanh"])
        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.40, step=0.05)

        # ----- TRAINING HYPERPARAMETERS -----
        cfg["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 3e-3, log=True)
        cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        cfg["training"]["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "lion"])

        batch_power = trial.suggest_int("batch_power", *BATCH_POWER_RANGE)  # 256 .. 8192
        cfg["training"]["batch_size"] = 2 ** batch_power

        # Mixture temperature schedule (used by PrunableTrainer)
        cfg["training"]["mixture_temperature_schedule"] = {
            "start": trial.suggest_float("t_start", 0.5, 2.0),
            "end": trial.suggest_float("t_end", 0.1, 1.0),
            "anneal_frac": trial.suggest_float("t_anneal_frac", 0.3, 0.9),
        }

        # Early stopping (if Trainer reads it)
        cfg["training"]["early_stopping_patience"] = trial.suggest_int(
            "early_stopping_patience", 10, 40
        )

        return cfg

    def run_trial(self, trial: optuna.Trial) -> float:
        """Execute a single trial and return validation loss."""
        config = self.suggest_config(trial)

        # Save dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}"
        save_dir = Path(self.base_config["paths"]["model_save_dir"]) / "optuna" / f"{timestamp}_{trial_id}"
        ensure_directories(save_dir)

        try:
            # Seed
            seed_everything(config.get("system", {}).get("seed", HPO_SEED))

            # Model
            model = create_model(config, self.device)

            # Epoch budget: run to hpo_max_epochs; pruning will stop early
            base_train = self.base_config.get("training", {})
            max_epochs = int(base_train.get("hpo_max_epochs", HPO_DEFAULT_MAX_EPOCHS))
            min_epochs = int(base_train.get("hpo_min_epochs", HPO_DEFAULT_MIN_EPOCHS))
            config["training"]["epochs"] = max_epochs

            pruning_callback = OptunaPruningCallback(trial, min_epochs)

            # Fresh datasets per trial (config-dependent)
            train_dataset = SequenceDataset(self.processed_dir, "train", config, self.device, self.norm_stats)
            val_dataset = SequenceDataset(self.processed_dir, "validation", config, self.device, self.norm_stats)

            # Log trial summary
            self.logger.info(
                f"Trial {trial.number}: type={config['model']['type']}, "
                f"latent={config['model']['latent_dim']}, "
                f"lr={config['training']['learning_rate']:.2e}, "
                f"batch={config['training']['batch_size']}, "
                f"epochs={config['training']['epochs']}"
            )

            trainer = PrunableTrainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=None,
                config=config,
                save_dir=save_dir,
                device=self.device,
                norm_helper=self.norm_helper,
                epoch_callback=pruning_callback,
            )

            best_val_loss = trainer.train()

            # Persist trial info
            trial.set_user_attr("config", config)
            trial.set_user_attr("best_epoch", trainer.best_epoch)
            save_json(config, save_dir / "config.json")

            self.logger.info(
                f"Trial {trial.number} completed: loss={best_val_loss:.6f}, best_epoch={trainer.best_epoch}"
            )
            return best_val_loss

        except optuna.TrialPruned:
            self.logger.info(f"Trial {trial.number} pruned")
            raise

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return float("inf")

        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class PrunableTrainer(Trainer):
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
            if hasattr(self.model, "set_gate_temperature") and hasattr(self.model, "K") and self.model.K > 1:
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


def optimize_hyperparameters(
    config_path: Path,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    use_hyperband: bool = True,
) -> optuna.Study:
    """Run hyperparameter optimization."""
    logger = logging.getLogger(__name__)

    if study_name is None:
        study_name = f"{HPO_STUDY_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Initialize tuner (preprocess + checks)
    tuner = HyperparameterTuner(config_path)

    # Read epoch budget from config (with sane defaults)
    base_config = load_json_config(config_path)
    min_epochs = int(base_config.get("training", {}).get("hpo_min_epochs", HPO_DEFAULT_MIN_EPOCHS))
    max_epochs = int(base_config.get("training", {}).get("hpo_max_epochs", HPO_DEFAULT_MAX_EPOCHS))

    # Pruner
    pruner = None
    if use_hyperband:
        pruner = HyperbandPruner(
            min_resource=min_epochs,
            max_resource=max_epochs,
            reduction_factor=HPO_REDUCTION_FACTOR,
        )
        logger.info(f"Using Hyperband pruner: min={min_epochs}, max={max_epochs}, rf={HPO_REDUCTION_FACTOR}")

    # Sampler
    sampler = TPESampler(
        seed=HPO_SEED,
        multivariate=True,
        constant_liar=True,          # better for parallel workers
        n_startup_trials=HPO_TPE_STARTUP,
        n_ei_candidates=HPO_TPE_EI_CANDIDATES,
        consider_endpoints=True,
    )

    # Study
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
    )

    logger.info(f"Starting optimization: n_trials={n_trials}, n_jobs={n_jobs}")
    study.optimize(tuner.run_trial, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    # Results
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)

    best_config = study.best_trial.user_attrs.get("config", {})

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    results = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": best_config,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_file = results_dir / f"{study_name}_results.json"
    save_json(results, results_file)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {results['best_value']:.6f}")
    print(f"Best trial: #{results['best_trial_number']}")
    print(f"Completed: {results['n_trials_completed']}, Pruned: {results['n_trials_pruned']}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {results_file}")

    return study


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Aggressive Optuna HPO for LiLaN")
    p.add_argument("config", type=Path, help="Path to JSON config")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--no-hyperband", action="store_true")
    args = p.parse_args()

    optimize_hyperparameters(
        config_path=args.config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        use_hyperband=not args.no_hyperband,
    )
