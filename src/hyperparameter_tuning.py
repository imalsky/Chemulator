#!/usr/bin/env python3
"""
Hyperparameter optimization module for LiLaN chemical kinetics models.

Provides Optuna-based hyperparameter search for optimizing model architecture 
and training parameters. Designed to be called through main.py --tune.
"""

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
    "encoder_layers": {
        "min_layers": 2,
        "max_layers": 7,
        "layer_size_min": 32,
        "layer_size_max": 1024,
        "layer_size_step": 64
    },
    "decoder_layers": {
        "min_layers": 2,
        "max_layers": 5,
        "layer_size_min": 64,
        "layer_size_max": 1024,
        "layer_size_step": 64
    },
    "activation": ["tanh", "gelu", "relu", "silu"],
    "dropout": {"low": 0.0, "high": 0.2, "step": 0.05},
}

# Training Hyperparameter Search Space  
TRAINING_SEARCH_SPACE = {
    "optimizer": ["AdamW"],
    "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
    "beta1": {"low": 0.85, "high": 0.95},
    "beta2": {"low": 0.95, "high": 0.999},
    "batch_power": {"low": 8, "high": 12},
    "gradient_accumulation": {"low": 1, "high": 8},
    "use_amp": [True, False],
    "gradient_clip": {"low": 0.5, "high": 5.0, "step": 0.5},
    "scheduler": ["cosine", "plateau", "none"],
    "cosine_T0": {"low": 10, "high": 100, "step": 10},
    "cosine_Tmult": {"low": 1, "high": 2},
    "use_fraction": {"low": 0.8, "high": 1.0, "step": 0.1},
    "amp_dtype": ["bfloat16"]
}

# Fixed Training Parameters for HPO
HPO_TRAINING_CONFIG = {
    "epochs": 100,
    "early_stopping_patience": 20,
    "min_delta": 1e-6,
}

# Default/Baseline Configuration
DEFAULT_TRIAL = {
    "latent_dim": 32,
    "n_encoder_layers": 4,
    "encoder_layer_0": 256,
    "encoder_layer_1": 256,
    "encoder_layer_2": 256,
    "encoder_layer_3": 256,
    "n_decoder_layers": 4,
    "decoder_layer_0": 256,
    "decoder_layer_1": 256,
    "decoder_layer_2": 256,
    "decoder_layer_3": 256,
    "activation": "tanh",
    "dropout": 0.0,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    "batch_power": 10,
    "grad_accum": 4,
    "gradient_clip": 2.0,
    "scheduler": "cosine",
    "cosine_T0": 50,
    "cosine_Tmult": 2,
    "use_amp": False,
    "use_fraction": 1.0,
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
        
        self.logger.info("="*80)
        if self.total_trials:
            self.logger.info(f"TRIAL {trial.number} STARTING (planned total: {self.total_trials})")
        else:
            self.logger.info(f"TRIAL {trial.number} STARTING")
        self.logger.info(f"Total elapsed: {elapsed/60:.1f} min")
        
        if self.trial_times:
            avg_time = float(np.mean(self.trial_times))
            self.logger.info(f"Average trial time: {avg_time:.1f}s")
            if self.total_trials:
                completed = len(self.trial_times)
                remaining = max(0, self.total_trials - completed)
                eta = avg_time * remaining / 60
                self.logger.info(f"Estimated remaining: {eta:.1f} min")
        self.logger.info("="*80)
    
    def on_trial_end(self, study, trial):
        """Log trial completion."""
        if self._trial_start is None:
            return
            
        trial_time = time.time() - self._trial_start
        self.trial_times.append(trial_time)
        
        self.logger.info("-"*80)
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
        
        self.logger.info("-"*80 + "\n")


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
        
        self.logger.info("="*80)
        self.logger.info("INITIALIZING HYPERPARAMETER OPTIMIZATION")
        self.logger.info("="*80)
        
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
        
        # Initialize datasets
        self.logger.info("\nLoading datasets...")
        self._init_datasets()
        
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
    
    def _init_datasets(self):
        """Initialize reusable datasets."""
        start = time.time()
        
        self.train_dataset = SequenceDataset(
            self.data_dir, "train", self.base_config,
            self.device, self.norm_stats
        )
        
        self.val_dataset = SequenceDataset(
            self.data_dir, "validation", self.base_config,
            self.device, self.norm_stats
        )
        
        try:
            self.test_dataset = SequenceDataset(
                self.data_dir, "test", self.base_config,
                self.device, self.norm_stats
            )
        except Exception:
            self.test_dataset = None
        
        load_time = time.time() - start
        self.logger.info(
            f"Datasets loaded in {load_time:.1f}s - "
            f"Train: {len(self.train_dataset):,} | "
            f"Val: {len(self.val_dataset):,}"
        )
    
    def suggest_model_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest model architecture parameters."""
        cfg = {}
        search = MODEL_SEARCH_SPACE
        
        cfg["latent_dim"] = trial.suggest_int(
            "latent_dim",
            search["latent_dim"]["low"],
            search["latent_dim"]["high"],
            step=search["latent_dim"]["step"]
        )
        
        # Encoder layers
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
        
        # Decoder layers
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
        
        cfg["activation"] = trial.suggest_categorical("activation", search["activation"])
        cfg["dropout"] = trial.suggest_float(
            "dropout",
            search["dropout"]["low"],
            search["dropout"]["high"],
            step=search["dropout"]["step"]
        )
        
        return cfg
    
    def suggest_training_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest training hyperparameters."""
        cfg = {}
        search = TRAINING_SEARCH_SPACE
        
        cfg["optimizer"] = trial.suggest_categorical("optimizer", search["optimizer"])
        cfg["learning_rate"] = trial.suggest_float(
            "learning_rate",
            search["learning_rate"]["low"],
            search["learning_rate"]["high"],
            log=search["learning_rate"]["log"]
        )
        cfg["weight_decay"] = trial.suggest_float(
            "weight_decay",
            search["weight_decay"]["low"],
            search["weight_decay"]["high"],
            log=search["weight_decay"]["log"]
        )
        
        # Sample the betas
        beta1 = trial.suggest_float("beta1", search["beta1"]["low"], search["beta1"]["high"])
        beta2 = trial.suggest_float("beta2", search["beta2"]["low"], search["beta2"]["high"])
        
        # Ensure beta2 > beta1 for optimizer stability
        if beta2 <= beta1:
            beta2 = min(0.999, beta1 + 0.01)
        
        cfg["betas"] = [beta1, beta2]
        
        batch_power = trial.suggest_int("batch_power", search["batch_power"]["low"], search["batch_power"]["high"])
        cfg["batch_size"] = 2 ** batch_power
        cfg["gradient_accumulation_steps"] = trial.suggest_int(
            "grad_accum",
            search["gradient_accumulation"]["low"],
            search["gradient_accumulation"]["high"]
        )
        
        cfg["gradient_clip"] = trial.suggest_float(
            "gradient_clip",
            search["gradient_clip"]["low"],
            search["gradient_clip"]["high"],
            step=search["gradient_clip"]["step"]
        )
        
        cfg["scheduler"] = trial.suggest_categorical("scheduler", search["scheduler"])
        if cfg["scheduler"] == "cosine":
            cfg["scheduler_params"] = {
                "T_0": trial.suggest_int("cosine_T0", search["cosine_T0"]["low"], search["cosine_T0"]["high"]),
                "T_mult": trial.suggest_int("cosine_Tmult", search["cosine_Tmult"]["low"], search["cosine_Tmult"]["high"]),
                "eta_min": 1e-8
            }
        
        cfg["use_amp"] = trial.suggest_categorical("use_amp", search["use_amp"])
        if cfg["use_amp"]:
            cfg["amp_dtype"] = trial.suggest_categorical("amp_dtype", search["amp_dtype"])
        
        cfg["use_fraction"] = trial.suggest_float(
            "use_fraction",
            search["use_fraction"]["low"],
            search["use_fraction"]["high"],
            step=search["use_fraction"]["step"]
        )
        
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
        
        # Create configuration
        config = json.loads(json.dumps(self.base_config))
        model_params = self.suggest_model_params(trial)
        train_params = self.suggest_training_params(trial)
        config["model"].update(model_params)
        config["training"].update(train_params)
        config["optuna"] = {"enabled": True}
        
        # Log hyperparameters
        self._log_trial_params(model_params, train_params)
        
        # Create trial directory
        trial_dir = self.study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        with open(trial_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Create and train model
            model = create_model(config, self.device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created: {n_params:,} parameters")
            
            norm_helper = NormalizationHelper(self.norm_stats, self.device, config) if self.norm_stats else None
            
            self.logger.info("Starting training...")
            trainer = Trainer(
                model=model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                config=config,
                save_dir=trial_dir,
                device=self.device,
                norm_helper=norm_helper
            )
            trainer.trial = trial  # For pruning
            
            train_start = time.time()
            best_val_loss = trainer.train()
            train_time = time.time() - train_start
            
            self.logger.info(f"Training complete in {train_time/60:.1f} min")
            self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
            
            if np.isfinite(best_val_loss):
                trial.set_user_attr("best_epoch", trainer.best_epoch)
                trial.set_user_attr("model_params", n_params)
                trial.set_user_attr("training_time", train_time)
            
            # Try to call progress callback end
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
            # Cleanup to free memory
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
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
        self.logger.info("="*80)
        self.logger.info(f"STARTING OPTIMIZATION: {study_name or self.study_name}")
        self.logger.info("="*80)
        
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
        
        self.logger.info("\n" + "="*80)
        self.logger.info("OPTIMIZATION RUNNING")
        self.logger.info("="*80 + "\n")
        
        start_time = time.time()
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            gc_after_trial=True,
        )
        
        total_time = time.time() - start_time
        self.logger.info(f"\nOPTIMIZATION COMPLETE in {total_time/60:.1f} minutes")
        
        self._report_results(study)
        return study
    
    def _materialize_model_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Convert trial params to model config."""
        cfg = {
            "latent_dim": p["latent_dim"],
            "activation": p["activation"],
            "dropout": p["dropout"],
        }
        
        n_enc = p["n_encoder_layers"]
        cfg["encoder_layers"] = [p[f"encoder_layer_{i}"] for i in range(n_enc)]
        
        n_dec = p["n_decoder_layers"]
        cfg["decoder_layers"] = [p[f"decoder_layer_{i}"] for i in range(n_dec)]
        
        return cfg
    
    def _materialize_training_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Convert trial params to training config."""
        cfg = {
            "optimizer": p["optimizer"],
            "learning_rate": p["learning_rate"],
            "weight_decay": p["weight_decay"],
            "betas": [p["beta1"], p["beta2"]],
            "batch_size": 2 ** int(p["batch_power"]),
            "gradient_accumulation_steps": int(p["grad_accum"]),
            "gradient_clip": p["gradient_clip"],
            "scheduler": p["scheduler"],
            "use_fraction": p["use_fraction"],
        }
        
        if cfg["scheduler"] == "cosine":
            cfg["scheduler_params"] = {
                "T_0": int(p["cosine_T0"]),
                "T_mult": int(p["cosine_Tmult"]),
                "eta_min": 1e-8,
            }
        
        cfg["use_amp"] = p["use_amp"]
        if cfg["use_amp"]:
            cfg["amp_dtype"] = p.get("amp_dtype", "bfloat16")
        
        cfg.update(HPO_TRAINING_CONFIG)
        return cfg
    
    def _report_results(self, study: optuna.Study):
        """Report and save study results."""
        self.logger.info("="*80)
        self.logger.info("HYPERPARAMETER OPTIMIZATION RESULTS")
        self.logger.info("="*80)
        
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
            self.logger.info(f"  Completed: {len(complete)} ({len(complete)/total*100:.1f}%)")
            self.logger.info(f"  Pruned: {len(pruned)} ({len(pruned)/total*100:.1f}%)")
            self.logger.info(f"  Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
        
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
        best_config = json.loads(json.dumps(self.base_config))
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
        
        self.logger.info("="*80)


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