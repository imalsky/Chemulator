#!/usr/bin/env python3
"""
Hyperparameter optimization module for LiLaN chemical kinetics models.

Provides Optuna-based hyperparameter search for optimizing model architecture
and training parameters. Designed to be called through main.py --tune.
"""

import copy
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from utils.utils import seed_everything, load_json, load_json_config
from data.preprocessor import DataPreprocessor
from data.dataset import SequenceDataset
from data.normalizer import NormalizationHelper
from models.model import create_model
from training.trainer import Trainer
from utils.hardware import setup_device, optimize_hardware

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Model Architecture Search Space
MODEL_SEARCH_SPACE = {
    "latent_dim": {"low": 16, "high": 256, "step": 16},
    "tau_mode": ["integral", "direct"],
    "encoder_layers": {
        "min_layers": 2,
        "max_layers": 7,
        "layer_size_min": 32,
        "layer_size_max": 1024,
        "layer_size_step": 64
    },
    "decoder_layers": {
        "min_layers": 2,
        "max_layers": 7,
        "layer_size_min": 32,
        "layer_size_max": 1024,
        "layer_size_step": 64
    },
    "activation": ["tanh", "gelu", "relu", "silu"],
    # "dropout": {"low": 0.0, "high": 0.2, "step": 0.05},
}

# Training Hyperparameter Search Space
TRAINING_SEARCH_SPACE = {
    # "optimizer": ["AdamW", "Lion"],
    # "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
    # "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
    # "beta1": {"low": 0.85, "high": 0.95},
    # "beta2": {"low": 0.95, "high": 0.999},
    # "batch_power": {"low": 8, "high": 12},
    # "use_amp": [True, False],
    # "gradient_clip": {"low": 0.5, "high": 5.0, "step": 0.5},
    # "scheduler": ["cosine", "plateau", "none"],
    # "cosine_T0": {"low": 10, "high": 100, "step": 10},
    # "cosine_Tmult": {"low": 1, "high": 2},
    # "amp_dtype": ["bfloat16", "float16"],
}

# Fixed Training Parameters for HPO
HPO_TRAINING_CONFIG = {
    "epochs": 100,
    "early_stopping_patience": 20,
    "min_delta": 1e-6,
}

# Default/Baseline Configuration for first trial
DEFAULT_TRIAL = {
    "latent_dim": 128,
    "n_encoder_layers": 4,
    "encoder_layer_0": 512,
    "encoder_layer_1": 512,
    "encoder_layer_2": 512,
    "encoder_layer_3": 512,
    "n_decoder_layers": 4,
    "decoder_layer_0": 192,
    "decoder_layer_1": 256,
    "decoder_layer_2": 256,
    "decoder_layer_3": 256,
    "activation": "silu",
}


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressCallback:
    """Tracks and logs optimization progress."""

    def __init__(self, logger, total_trials: Optional[int] = None):
        self.logger = logger
        self.total_trials = total_trials
        self.start_time = time.time()
        self.trial_times = []
        self._trial_start = None

    def on_trial_start(self, study, trial):
        """Log trial start."""
        self._trial_start = time.time()
        elapsed = self._trial_start - self.start_time

        self.logger.info("=" * 80)
        if self.total_trials:
            self.logger.info(f"TRIAL {trial.number} STARTING (planned total: {self.total_trials})")
        else:
            self.logger.info(f"TRIAL {trial.number} STARTING")
        self.logger.info(f"Total elapsed: {elapsed / 60:.1f} min")

        if self.trial_times:
            avg_time = float(np.mean(self.trial_times))
            self.logger.info(f"Average trial time: {avg_time:.1f}s")
            if self.total_trials:
                completed = len(self.trial_times)
                remaining = max(0, self.total_trials - completed)
                eta = avg_time * remaining / 60
                self.logger.info(f"Estimated remaining: {eta:.1f} min")
        self.logger.info("=" * 80)

    def on_trial_end(self, study, trial):
        """Log trial completion."""
        if self._trial_start is None:
            return

        trial_time = time.time() - self._trial_start
        self.trial_times.append(trial_time)

        self.logger.info("-" * 80)
        self.logger.info(f"TRIAL {trial.number} COMPLETED in {trial_time:.1f}s")

        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"Validation loss: {trial.value:.6f}")

            try:
                if hasattr(study, 'best_trial') and study.best_trial is not None:
                    self.logger.info(f"Current best: Trial {study.best_trial.number} "
                                     f"(loss: {study.best_value:.6f})")
                    if trial.value <= study.best_value * 1.001:
                        improvement = (study.best_value - trial.value) / abs(study.best_value) * 100
                        self.logger.info(f"IMPROVEMENT: {improvement:.2f}% better")
            except Exception as e:
                self.logger.debug(f"Could not compare with best trial: {e}")

        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.info("Trial PRUNED (early stopped)")
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.logger.info("Trial FAILED")
        else:
            self.logger.info(f"Trial ended with state: {trial.state}")

        self.logger.info("-" * 80 + "\n")


# ============================================================================
# HYPERPARAMETER TUNER
# ============================================================================

class HyperparameterTuner:
    """Orchestrates hyperparameter optimization using Optuna."""

    def __init__(
            self,
            base_config: Dict[str, Any],
            data_dir: Path,
            output_dir: Path,
            study_name: str = "lilan_hpo"
    ):
        """Initialize tuner with configuration and data."""
        self.base_config = base_config
        self.data_dir = Path(data_dir)
        self.study_name = study_name
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING HYPERPARAMETER OPTIMIZATION")
        self.logger.info("=" * 80)

        # Setup device
        self.device = setup_device()
        optimize_hardware(base_config.get("system", {}), self.device)
        self._log_device_info()

        # Setup directories at same level as model_save_dir
        self.study_dir = output_dir / "optuna_studies" / study_name

        # Check if study already exists
        if self.study_dir.exists():
            self.logger.error(f"Study directory already exists: {self.study_dir}")
            self.logger.error("Please use a different study name or remove the existing directory.")
            raise FileExistsError(
                f"Study directory already exists: {self.study_dir}\n"
                f"Use --study-name to specify a different name or remove the existing directory."
            )

        self.study_dir.mkdir(parents=True, exist_ok=False)
        self.logger.info(f"Study directory: {self.study_dir}")

        # Load normalization stats
        norm_path = self.data_dir / "normalization.json"
        self.norm_stats = load_json(norm_path) if norm_path.exists() else {}

        # Create progress callback
        self.progress_callback = ProgressCallback(self.logger)

        self.logger.info("Initialization complete\n")

    def _log_device_info(self):
        """Log device information."""
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            self.logger.info(f"Device: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.logger.info(f"Device: {self.device}")

    def suggest_model_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest model architecture parameters."""
        cfg = {}
        search = MODEL_SEARCH_SPACE

        # Latent dimension
        if "latent_dim" in search:
            cfg["latent_dim"] = trial.suggest_int(
                "latent_dim",
                search["latent_dim"]["low"],
                search["latent_dim"]["high"],
                step=search["latent_dim"]["step"]
            )
        else:
            cfg["latent_dim"] = self.base_config["model"].get("latent_dim", 128)

        if "tau_mode" in search:
            cfg["tau_mode"] = trial.suggest_categorical("tau_mode", search["tau_mode"])
        else:
            cfg["tau_mode"] = self.base_config["model"].get("tau_mode", "integral")

        # Encoder layers
        if "encoder_layers" in search:
            enc_search = search["encoder_layers"]
            n_enc = trial.suggest_int("n_encoder_layers", enc_search["min_layers"], enc_search["max_layers"])
            cfg["encoder_layers"] = [
                trial.suggest_int(
                    f"encoder_layer_{i}",
                    enc_search["layer_size_min"],
                    enc_search["layer_size_max"],
                    step=enc_search["layer_size_step"]
                ) for i in range(n_enc)
            ]
        else:
            cfg["encoder_layers"] = self.base_config["model"].get("encoder_layers", [512, 512, 512, 512])

        # Decoder layers
        if "decoder_layers" in search:
            dec_search = search["decoder_layers"]
            n_dec = trial.suggest_int("n_decoder_layers", dec_search["min_layers"], dec_search["max_layers"])
            cfg["decoder_layers"] = [
                trial.suggest_int(
                    f"decoder_layer_{i}",
                    dec_search["layer_size_min"],
                    dec_search["layer_size_max"],
                    step=dec_search["layer_size_step"]
                ) for i in range(n_dec)
            ]
        else:
            cfg["decoder_layers"] = self.base_config["model"].get("decoder_layers", [192, 256, 256, 256])

        # Activation
        if "activation" in search:
            cfg["activation"] = trial.suggest_categorical("activation", search["activation"])
        else:
            cfg["activation"] = self.base_config["model"].get("activation", "silu")

        # Dropout
        if "dropout" in search:
            cfg["dropout"] = trial.suggest_float(
                "dropout",
                search["dropout"]["low"],
                search["dropout"]["high"],
                step=search["dropout"]["step"]
            )
        else:
            cfg["dropout"] = self.base_config["model"].get("dropout", 0.0)

        # Copy over tau_layers from base config (not tuned)
        cfg["tau_layers"] = self.base_config["model"].get("tau_layers", [128, 128, 128, 128])

        return cfg

    def suggest_training_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest training hyperparameters."""
        cfg = {}
        search = TRAINING_SEARCH_SPACE

        # Optimizer
        if "optimizer" in search:
            cfg["optimizer"] = trial.suggest_categorical("optimizer", search["optimizer"])
        else:
            cfg["optimizer"] = self.base_config["training"].get("optimizer", "AdamW")

        # Learning rate
        if "learning_rate" in search:
            cfg["learning_rate"] = trial.suggest_float(
                "learning_rate",
                search["learning_rate"]["low"],
                search["learning_rate"]["high"],
                log=search["learning_rate"].get("log", False)
            )
        else:
            cfg["learning_rate"] = self.base_config["training"].get("learning_rate", 1e-4)

        # Weight decay
        if "weight_decay" in search:
            cfg["weight_decay"] = trial.suggest_float(
                "weight_decay",
                search["weight_decay"]["low"],
                search["weight_decay"]["high"],
                log=search["weight_decay"].get("log", False)
            )
        else:
            cfg["weight_decay"] = self.base_config["training"].get("weight_decay", 5e-5)

        # Betas for Adam/AdamW
        if "beta1" in search and "beta2" in search:
            beta1 = trial.suggest_float("beta1", search["beta1"]["low"], search["beta1"]["high"])
            beta2 = trial.suggest_float("beta2", search["beta2"]["low"], search["beta2"]["high"])
            # Ensure beta2 > beta1 for optimizer stability
            if beta2 <= beta1:
                beta2 = min(0.999, beta1 + 0.01)
            cfg["betas"] = [beta1, beta2]
        else:
            cfg["betas"] = self.base_config["training"].get("betas", [0.85, 0.95])

        # Batch size
        if "batch_power" in search:
            batch_power = trial.suggest_int(
                "batch_power",
                search["batch_power"]["low"],
                search["batch_power"]["high"]
            )
            cfg["batch_size"] = 2 ** batch_power
        else:
            cfg["batch_size"] = self.base_config["training"].get("batch_size", 256)

        # Gradient clipping
        if "gradient_clip" in search:
            cfg["gradient_clip"] = trial.suggest_float(
                "gradient_clip",
                search["gradient_clip"]["low"],
                search["gradient_clip"]["high"],
                step=search["gradient_clip"].get("step", 0.1)
            )
        else:
            cfg["gradient_clip"] = self.base_config["training"].get("gradient_clip", 2.0)

        # Scheduler
        if "scheduler" in search:
            cfg["scheduler"] = trial.suggest_categorical("scheduler", search["scheduler"])
        else:
            cfg["scheduler"] = self.base_config["training"].get("scheduler", "cosine")

        # Scheduler params (only if using cosine and params are in search space)
        if cfg["scheduler"] == "cosine":
            scheduler_params = {}
            if "cosine_T0" in search:
                scheduler_params["T_0"] = trial.suggest_int(
                    "cosine_T0",
                    search["cosine_T0"]["low"],
                    search["cosine_T0"]["high"],
                    step=search["cosine_T0"].get("step", 10)
                )
            else:
                scheduler_params["T_0"] = self.base_config["training"].get("scheduler_params", {}).get("T_0", 100)

            if "cosine_Tmult" in search:
                scheduler_params["T_mult"] = trial.suggest_int(
                    "cosine_Tmult",
                    search["cosine_Tmult"]["low"],
                    search["cosine_Tmult"]["high"]
                )
            else:
                scheduler_params["T_mult"] = self.base_config["training"].get("scheduler_params", {}).get("T_mult", 2)

            scheduler_params["eta_min"] = self.base_config["training"].get("scheduler_params", {}).get("eta_min", 1e-8)
            cfg["scheduler_params"] = scheduler_params
        elif cfg["scheduler"] == "plateau":
            # Use base config params for plateau scheduler
            cfg["scheduler_params"] = self.base_config["training"].get("scheduler_params", {
                "factor": 0.5,
                "patience": 10,
                "min_lr": 1e-7
            })
        else:
            cfg["scheduler_params"] = {}

        # Mixed precision training
        if "use_amp" in search:
            cfg["use_amp"] = trial.suggest_categorical("use_amp", search["use_amp"])
        else:
            cfg["use_amp"] = self.base_config["training"].get("use_amp", True)

        if cfg["use_amp"]:
            if "amp_dtype" in search:
                cfg["amp_dtype"] = trial.suggest_categorical("amp_dtype", search["amp_dtype"])
            else:
                cfg["amp_dtype"] = self.base_config["training"].get("amp_dtype", "bfloat16")

        # Add fixed HPO training config
        cfg.update(HPO_TRAINING_CONFIG)

        return cfg

    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Try to call progress callback start
        try:
            self.progress_callback.on_trial_start(trial.study, trial)
        except Exception as e:
            self.logger.debug(f"Progress callback start failed: {e}")

        # Set seed
        seed = self.base_config.get("system", {}).get("seed", 42)
        seed_everything(seed + trial.number)

        # Create configuration with proper deep copy
        config = copy.deepcopy(self.base_config)

        # Get hyperparameters from trial
        model_params = self.suggest_model_params(trial)
        train_params = self.suggest_training_params(trial)

        # Update config with trial parameters
        config["model"].update(model_params)
        config["training"].update(train_params)

        # Ensure use_fraction is consistently set from base config
        # This should never be tuned as it affects data preprocessing
        base_use_fraction = self.base_config.get("training", {}).get("use_fraction", 1.0)
        config["training"]["use_fraction"] = base_use_fraction

        # Mark this as an optuna trial for the trainer
        config["optuna"] = {"enabled": True}

        # Log hyperparameters
        self._log_trial_params(model_params, train_params)

        # Create trial directory
        trial_dir = self.study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save the actual config used for this trial
        with open(trial_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize resources for this trial
        train_dataset, val_dataset, test_dataset = None, None, None
        trainer, model = None, None

        try:
            # Create fresh datasets for this trial to prevent memory leaks
            self.logger.info("Loading datasets for trial %d...", trial.number)
            train_dataset = SequenceDataset(
                self.data_dir, "train", config, self.device, self.norm_stats
            )
            val_dataset = SequenceDataset(
                self.data_dir, "validation", config, self.device, self.norm_stats
            )
            try:
                test_dataset = SequenceDataset(
                    self.data_dir, "test", config, self.device, self.norm_stats
                )
            except Exception:
                test_dataset = None

            # Create and train model
            model = create_model(config, self.device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created: {n_params:,} parameters")

            norm_helper = NormalizationHelper(self.norm_stats, self.device, config) if self.norm_stats else None

            self.logger.info("Starting training...")
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                config=config,
                save_dir=trial_dir,
                device=self.device,
                norm_helper=norm_helper,
                processed_dir=self.data_dir
            )
            trainer.trial = trial  # For pruning

            train_start = time.time()
            best_val_loss = trainer.train()
            train_time = time.time() - train_start

            self.logger.info(f"Training complete in {train_time / 60:.1f} min")
            self.logger.info(f"Best validation loss: {best_val_loss:.6f}")

            if np.isfinite(best_val_loss):
                trial.set_user_attr("best_epoch", trainer.best_epoch)
                trial.set_user_attr("model_params", n_params)
                trial.set_user_attr("training_time", train_time)
                trial.set_user_attr("use_fraction", base_use_fraction)

            try:
                self.progress_callback.on_trial_end(trial.study, trial)
            except Exception as e:
                self.logger.debug(f"Progress callback end failed: {e}")

            return best_val_loss

        except optuna.TrialPruned:
            self.logger.info("Trial pruned")
            try:
                self.progress_callback.on_trial_end(trial.study, trial)
            except Exception as e:
                self.logger.debug(f"Progress callback end failed: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            try:
                self.progress_callback.on_trial_end(trial.study, trial)
            except Exception as e2:
                self.logger.debug(f"Progress callback end failed: {e2}")
            return float('inf')

        finally:
            # Comprehensive cleanup to prevent memory leaks
            for dataset in [train_dataset, val_dataset, test_dataset]:
                if dataset is not None:
                    try:
                        dataset.close()
                    except Exception as e:
                        self.logger.debug("Error closing dataset: %s", e)
            del trainer, model, train_dataset, val_dataset, test_dataset
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _log_trial_params(self, model_params: Dict, train_params: Dict):
        """Log key trial parameters."""
        self.logger.info(f"  Model: latent_dim={model_params['latent_dim']}, "
                         f"activation={model_params['activation']}")
        self.logger.info(f"  Training: optimizer={train_params['optimizer']}, "
                         f"lr={train_params['learning_rate']:.2e}, "
                         f"batch={train_params['batch_size']}")

    def run(
            self,
            n_trials: int = 100,
            n_jobs: int = 1,
            study_name: Optional[str] = None,
            use_hyperband: bool = True,
            timeout: Optional[int] = None,
    ) -> optuna.Study:
        """Run hyperparameter optimization."""
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING OPTIMIZATION: {study_name or self.study_name}")
        self.logger.info("=" * 80)

        if study_name and study_name != self.study_name:
            new_study_dir = self.study_dir.parent / study_name
            if new_study_dir.exists():
                raise FileExistsError(
                    f"Cannot change to study '{study_name}' - directory already exists: {new_study_dir}"
                )
            self.study_name = study_name
            self.study_dir = new_study_dir
            self.study_dir.mkdir(parents=True, exist_ok=False)

        self.logger.info(f"Configuration:")
        self.logger.info(f"  Trials: {n_trials}")
        self.logger.info(f"  Parallel jobs: {n_jobs}")
        self.logger.info(f"  Timeout: {timeout or 'None'}")
        self.logger.info(f"  Pruning: {'Hyperband' if use_hyperband else 'Disabled'}")

        # Setup sampler and pruner
        sampler = TPESampler(
            seed=self.base_config.get("system", {}).get("seed", 42),
            n_startup_trials=10,
            n_ei_candidates=24,
        )

        pruner = HyperbandPruner(
            min_resource=10,
            max_resource=HPO_TRAINING_CONFIG["epochs"],
            reduction_factor=3,
        ) if use_hyperband else optuna.pruners.NopPruner()

        # Create/load study
        self.logger.info(f"\nStudy database: {self.study_dir}/study.db")
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.study_dir}/study.db",
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=True,
        )

        # Set total trials for progress callback
        self.progress_callback.total_trials = n_trials

        # Seed with baseline if new
        if len(study.trials) == 0:
            self.logger.info("Seeding with baseline configuration")
            study.enqueue_trial(DEFAULT_TRIAL)
        else:
            self.logger.info(f"Resuming from {len(study.trials)} existing trials")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("OPTIMIZATION RUNNING")
        self.logger.info("=" * 80 + "\n")

        start_time = time.time()
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            gc_after_trial=True,
        )

        total_time = time.time() - start_time
        self.logger.info(f"\nOPTIMIZATION COMPLETE in {total_time / 60:.1f} minutes")

        self._report_results(study)
        return study

    def _materialize_model_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Convert trial params to model config."""
        cfg = {}

        # Handle each parameter with fallback to base config
        cfg["latent_dim"] = p.get("latent_dim", self.base_config["model"].get("latent_dim", 128))
        cfg["activation"] = p.get("activation", self.base_config["model"].get("activation", "silu"))
        cfg["dropout"] = p.get("dropout", self.base_config["model"].get("dropout", 0.0))

        # Add tau_mode handling
        cfg["tau_mode"] = p.get("tau_mode", self.base_config["model"].get("tau_mode", "integral"))

        # Handle encoder layers
        if "n_encoder_layers" in p:
            n_enc = p["n_encoder_layers"]
            cfg["encoder_layers"] = [p[f"encoder_layer_{i}"] for i in range(n_enc)]
        else:
            cfg["encoder_layers"] = self.base_config["model"].get("encoder_layers", [512, 512, 512, 512])

        # Handle decoder layers
        if "n_decoder_layers" in p:
            n_dec = p["n_decoder_layers"]
            cfg["decoder_layers"] = [p[f"decoder_layer_{i}"] for i in range(n_dec)]
        else:
            cfg["decoder_layers"] = self.base_config["model"].get("decoder_layers", [192, 256, 256, 256])

        # Copy tau_layers from base config (not tuned) - ONLY ONCE
        cfg["tau_layers"] = self.base_config["model"].get("tau_layers", [128, 128, 128, 128])

        return cfg

    def _materialize_training_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Convert trial params to training config."""
        cfg = {}

        # Use get() with defaults for all parameters since they might not be in trial params
        cfg["optimizer"] = p.get("optimizer", self.base_config["training"].get("optimizer", "AdamW"))
        cfg["learning_rate"] = p.get("learning_rate", self.base_config["training"].get("learning_rate", 1e-4))
        cfg["weight_decay"] = p.get("weight_decay", self.base_config["training"].get("weight_decay", 5e-5))

        # Handle betas carefully
        if "beta1" in p and "beta2" in p:
            cfg["betas"] = [p["beta1"], p["beta2"]]
        else:
            cfg["betas"] = self.base_config["training"].get("betas", [0.85, 0.95])

        # Handle batch size
        if "batch_power" in p:
            cfg["batch_size"] = 2 ** int(p["batch_power"])
        else:
            cfg["batch_size"] = self.base_config["training"].get("batch_size", 256)

        cfg["gradient_clip"] = p.get("gradient_clip", self.base_config["training"].get("gradient_clip", 2.0))
        cfg["scheduler"] = p.get("scheduler", self.base_config["training"].get("scheduler", "cosine"))

        # Handle scheduler params
        if cfg["scheduler"] == "cosine":
            if "cosine_T0" in p or "cosine_Tmult" in p:
                cfg["scheduler_params"] = {
                    "T_0": int(
                        p.get("cosine_T0", self.base_config["training"].get("scheduler_params", {}).get("T_0", 100))),
                    "T_mult": int(p.get("cosine_Tmult",
                                        self.base_config["training"].get("scheduler_params", {}).get("T_mult", 2))),
                    "eta_min": self.base_config["training"].get("scheduler_params", {}).get("eta_min", 1e-8),
                }
            else:
                cfg["scheduler_params"] = self.base_config["training"].get("scheduler_params", {})
        else:
            cfg["scheduler_params"] = self.base_config["training"].get("scheduler_params", {})

        cfg["use_amp"] = p.get("use_amp", self.base_config["training"].get("use_amp", True))
        if cfg["use_amp"]:
            cfg["amp_dtype"] = p.get("amp_dtype", self.base_config["training"].get("amp_dtype", "bfloat16"))

        # Never tune use_fraction - always use base config
        cfg["use_fraction"] = self.base_config.get("training", {}).get("use_fraction", 1.0)

        # Add fixed HPO config
        cfg.update(HPO_TRAINING_CONFIG)
        return cfg

    def _report_results(self, study: optuna.Study):
        """Report and save study results."""
        self.logger.info("=" * 80)
        self.logger.info("HYPERPARAMETER OPTIMIZATION RESULTS")
        self.logger.info("=" * 80)

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        total = len(study.trials)
        if total == 0:
            self.logger.info("No trials were executed.")
            return

        self.logger.info(f"\nStudy Statistics:")
        self.logger.info(f"  Total trials: {total}")
        if total > 0:
            self.logger.info(f"  Completed: {len(complete)} ({len(complete) / total * 100:.1f}%)")
            self.logger.info(f"  Pruned: {len(pruned)} ({len(pruned) / total * 100:.1f}%)")
            self.logger.info(f"  Failed: {len(failed)} ({len(failed) / total * 100:.1f}%)")

        if not complete:
            self.logger.info("\nNo completed trials. Skipping best configuration export.")
            return

        # Best trial info
        best_trial = study.best_trial
        self.logger.info(f"\nBest Trial: #{best_trial.number}")
        self.logger.info(f"  Validation Loss: {best_trial.value:.6f}")

        if "best_epoch" in best_trial.user_attrs:
            self.logger.info(f"  Best Epoch: {best_trial.user_attrs['best_epoch']}")
        if "model_params" in best_trial.user_attrs:
            self.logger.info(f"  Parameters: {best_trial.user_attrs['model_params']:,}")

        # Top parameters
        self.logger.info("\nBest Hyperparameters:")
        for i, (key, value) in enumerate(list(best_trial.params.items())[:15]):
            if isinstance(value, float):
                value_str = f"{value:.2e}" if value < 1e-3 or value > 1e3 else f"{value:.4f}"
            else:
                value_str = str(value)
            self.logger.info(f"  {key}: {value_str}")

        if len(best_trial.params) > 15:
            self.logger.info(f"  ... and {len(best_trial.params) - 15} more")

        # Save best configuration
        best_config = copy.deepcopy(self.base_config)
        best_config["model"].update(self._materialize_model_from_params(best_trial.params))
        best_config["training"].update(self._materialize_training_from_params(best_trial.params))

        best_config_path = self.study_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        self.logger.info(f"\nBest configuration saved: {best_config_path}")

        # Copy best model
        best_trial_dir = self.study_dir / f"trial_{best_trial.number:04d}"
        if (best_trial_dir / "best_model.pt").exists():
            shutil.copy2(
                best_trial_dir / "best_model.pt",
                self.study_dir / "best_model.pt"
            )
            self.logger.info(f"Best model saved: {self.study_dir / 'best_model.pt'}")

        # Loss statistics
        if complete:
            values = [t.value for t in complete if np.isfinite(t.value)]
            if values:
                self.logger.info(f"\nLoss Statistics:")
                self.logger.info(f"  Best: {np.min(values):.6f}")
                self.logger.info(f"  Mean: {np.mean(values):.6f} +/- {np.std(values):.6f}")
                self.logger.info(f"  Median: {np.median(values):.6f}")

        self.logger.info("=" * 80)


def optimize_hyperparameters(
        config_path: Union[str, Path],
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        use_hyperband: bool = True,
) -> optuna.Study:
    """Entry point for hyperparameter optimization from main.py."""
    cfg = load_json_config(Path(config_path))

    # Setup data directory
    data_dir = Path(cfg["paths"]["processed_data_dir"])
    model_save_dir = Path(cfg["paths"]["model_save_dir"])
    out_root = model_save_dir.parent  # Gets parent of "data/models" -> "data"
    out_root.mkdir(parents=True, exist_ok=True)

    # Preprocess if needed
    if not (data_dir / "normalization.json").exists():
        print("\nPreprocessed data not found. Running preprocessing first...")
        dp = DataPreprocessor(
            raw_files=[Path(p) for p in cfg["paths"]["raw_data_files"]],
            output_dir=data_dir,
            config=cfg,
        )
        dp.process_to_npy_shards()

    # Run optimization
    try:
        tuner = HyperparameterTuner(
            base_config=cfg,
            data_dir=data_dir,
            output_dir=out_root,
            study_name=study_name or "lilan_hpo",
        )
    except FileExistsError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    return tuner.run(
        n_trials=n_trials,
        n_jobs=n_jobs,
        study_name=study_name,
        use_hyperband=use_hyperband,
    )