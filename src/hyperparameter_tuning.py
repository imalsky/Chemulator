#!/usr/bin/env python3
"""
Hyperparameter optimization module for LiLaN chemical kinetics models.

This module provides Optuna-based hyperparameter search for optimizing
model architecture and training parameters. It is designed to be called
exclusively through main.py with the --mode tune argument.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from data.dataset import SequenceDataset
from data.normalizer import NormalizationHelper
from model import create_model
from trainer import Trainer
from utils.hardware import setup_device, optimize_hardware
from utils.utils import seed_everything, load_json


# ============================================================================
# HYPERPARAMETER SEARCH CONFIGURATION
# ============================================================================

# Study Configuration
STUDY_SETTINGS = {
    "n_trials": 100,                    # Number of optimization trials
    "timeout": None,                     # Timeout in seconds (None = no limit)
    "pruning": True,                     # Enable early stopping of bad trials
    "pruning_warmup_steps": 10,         # Steps before pruning can occur
    "pruning_warmup_trials": 5,         # Trials before pruning starts
}

# Model Architecture Search Space
MODEL_SEARCH_SPACE = {
    # Model variant
    "variant": ["full"],   # Architecture type, full, , "independent"
    
    # Latent space dimension (key parameter from paper)
    "latent_dim": {
        "low": 16, 
        "high": 128, 
        "step": 16
    },
    
    # Encoder/Decoder depth
    "encoder_layers": {
        "min_layers": 2,
        "max_layers": 5,
        "layer_size_min": 64,
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
    
    # Activation and regularization
    "activation": ["tanh", "gelu", "relu", "silu"],
    "dropout": {"low": 0.0, "high": 0.2, "step": 0.05},
    
    # Mixture of experts
    "use_mixture": [True, False],
    "mixture_K": {"low": 1, "high": 4},           # Number of experts
    "diversity_mode": ["per_sample", "batch_mean"],
    
    # Time warping (key innovation)
    "use_time_warp": [True, False],
    "warp_J_terms": {"low": 2, "high": 8},        # Expansion terms
    "warp_hidden": {"low": 32, "high": 128, "step": 32},
}

# Training Hyperparameter Search Space  
TRAINING_SEARCH_SPACE = {
    # Optimizer
    "optimizer": ["AdamW", "Lion"], # AdamW, Lion
    "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
    "beta1": {"low": 0.85, "high": 0.95},
    "beta2": {"low": 0.95, "high": 0.999},
    
    # Batch size (as powers of 2)
    "batch_power": {"low": 8, "high": 12},  # 256 to 4096
    "gradient_accumulation": {"low": 1, "high": 8},
    
    # Gradient clipping
    "gradient_clip": {"low": 0.0, "high": 10.0, "step": 0.5},
    
    # Learning rate scheduling
    "scheduler": ["cosine"], #"plateau", "none"
    "cosine_T0": {"low": 25, "high": 50},
    "cosine_Tmult": {"low": 1, "high": 4},
    "plateau_factor": {"low": 0.3, "high": 0.8},
    "plateau_patience": {"low": 5, "high": 20},
    
    # Regularization
    "lambda_entropy": {"low": 0.0, "high": 0.1, "step": 0.01},
    "lambda_diversity": {"low": 0.0, "high": 0.1, "step": 0.01},
    
    # Temperature annealing
    "temp_start": {"low": 0.5, "high": 2.0},
    "temp_end": {"low": 0.1, "high": 0.5},
    "temp_anneal_frac": {"low": 0.3, "high": 0.8},
    
    # Mixed precision
    "use_amp": [True, False],
    "amp_dtype": ["float16", "bfloat16"],
    
    # Data efficiency
    "use_fraction": {"low": 0.5, "high": 1.0, "step": 0.1},
}

# Fixed Training Parameters for HPO
HPO_TRAINING_CONFIG = {
    "epochs": 50,                       # Reduced for faster trials
    "early_stopping_patience": 10,      # Stop if no improvement
    "min_delta": 1e-6,                  # Minimum change to be improvement
    "val_fraction": 0.15,                # Validation split
    "test_fraction": 0.15,               # Test split  
}

# Default/Baseline Configuration
DEFAULT_TRIAL = {
    "model_variant": "independent",
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
    "use_mixture": False,
    "use_time_warp": True,
    "warp_J_terms": 5,
    "warp_hidden": 64,
    "warp_use_features": True,
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
    "lambda_entropy": 0.0,
    "lambda_diversity": 0.0,
    "temp_start": 1.0,
    "temp_end": 0.3,
    "temp_anneal_frac": 0.6,
    "use_amp": False,
    "use_fraction": 1.0,
}


# ============================================================================
# HYPERPARAMETER TUNER CLASS
# ============================================================================

class HyperparameterTuner:
    """
    Orchestrates hyperparameter optimization for LiLaN models using Optuna.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        data_dir: Path,
        output_dir: Path,
        study_name: str = "lilan_hpo"
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            base_config: Base configuration to modify during search
            data_dir: Directory containing preprocessed data shards
            output_dir: Directory for study outputs
            study_name: Name for the Optuna study
        """
        self.base_config = base_config
        self.data_dir = Path(data_dir)
        self.study_name = study_name
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = setup_device()
        optimize_hardware(base_config["system"], self.device)
        
        # Setup study directory
        self.study_dir = output_dir / "optuna_studies" / study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Load normalization statistics
        norm_path = self.data_dir / "normalization.json"
        self.norm_stats = load_json(norm_path) if norm_path.exists() else {}
        
        # Initialize datasets once for reuse
        self._init_datasets()
        
        self.logger.info(f"HPO initialized: {STUDY_SETTINGS['n_trials']} trials")
    
    def _init_datasets(self) -> None:
        """Initialize datasets that will be reused across trials."""
        self.logger.info("Loading datasets for HPO...")
        
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
        
        self.logger.info(
            f"Datasets loaded - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}"
        )
    
    def suggest_model_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest model architecture parameters."""
        cfg = {}
        search = MODEL_SEARCH_SPACE
        
        # Model variant
        cfg["variant"] = trial.suggest_categorical("model_variant", search["variant"])
        
        # Latent dimension
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
        
        # Activation and dropout
        cfg["activation"] = trial.suggest_categorical("activation", search["activation"])
        cfg["dropout"] = trial.suggest_float(
            "dropout",
            search["dropout"]["low"],
            search["dropout"]["high"],
            step=search["dropout"]["step"]
        )
        
        # Mixture of experts
        use_mixture = trial.suggest_categorical("use_mixture", search["use_mixture"])
        if use_mixture:
            cfg["mixture"] = {
                "K": trial.suggest_int("mixture_K", search["mixture_K"]["low"], search["mixture_K"]["high"]),
                "temperature": 1.0,
                "diversity_mode": trial.suggest_categorical("diversity_mode", search["diversity_mode"]),
                "use_encoder_features": trial.suggest_categorical("gate_use_features", [True, False])
            }
        else:
            cfg["mixture"] = {"K": 1}
        
        # Time warp
        cfg["time_warp"] = {
            "enabled": trial.suggest_categorical("use_time_warp", search["use_time_warp"]),
            "J_terms": trial.suggest_int("warp_J_terms", search["warp_J_terms"]["low"], search["warp_J_terms"]["high"]),
            "hidden_dim": trial.suggest_int(
                "warp_hidden",
                search["warp_hidden"]["low"],
                search["warp_hidden"]["high"],
                step=search["warp_hidden"]["step"]
            ),
            "use_encoder_features": trial.suggest_categorical("warp_use_features", [True, False])
        }
        
        return cfg
    
    def suggest_training_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest training hyperparameters."""
        cfg = {}
        search = TRAINING_SEARCH_SPACE
        
        # Optimizer
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
        
        beta1 = trial.suggest_float("beta1", search["beta1"]["low"], search["beta1"]["high"])
        beta2 = trial.suggest_float("beta2", search["beta2"]["low"], search["beta2"]["high"])
        cfg["betas"] = [beta1, beta2]
        
        # Batch size and accumulation
        batch_power = trial.suggest_int("batch_power", search["batch_power"]["low"], search["batch_power"]["high"])
        cfg["batch_size"] = 2 ** batch_power
        cfg["gradient_accumulation_steps"] = trial.suggest_int(
            "grad_accum",
            search["gradient_accumulation"]["low"],
            search["gradient_accumulation"]["high"]
        )
        
        # Gradient clipping
        cfg["gradient_clip"] = trial.suggest_float(
            "gradient_clip",
            search["gradient_clip"]["low"],
            search["gradient_clip"]["high"],
            step=search["gradient_clip"]["step"]
        )
        
        # Scheduler
        cfg["scheduler"] = trial.suggest_categorical("scheduler", search["scheduler"])
        if cfg["scheduler"] == "cosine":
            cfg["scheduler_params"] = {
                "T_0": trial.suggest_int("cosine_T0", search["cosine_T0"]["low"], search["cosine_T0"]["high"]),
                "T_mult": trial.suggest_int("cosine_Tmult", search["cosine_Tmult"]["low"], search["cosine_Tmult"]["high"]),
                "eta_min": 1e-8
            }
        elif cfg["scheduler"] == "plateau":
            cfg["scheduler_params"] = {
                "factor": trial.suggest_float("plateau_factor", search["plateau_factor"]["low"], search["plateau_factor"]["high"]),
                "patience": trial.suggest_int("plateau_patience", search["plateau_patience"]["low"], search["plateau_patience"]["high"]),
                "min_lr": 1e-8
            }
        
        # Regularization
        cfg["regularization"] = {
            "lambda_entropy": trial.suggest_float(
                "lambda_entropy",
                search["lambda_entropy"]["low"],
                search["lambda_entropy"]["high"],
                step=search["lambda_entropy"]["step"]
            ),
            "lambda_diversity": trial.suggest_float(
                "lambda_diversity",
                search["lambda_diversity"]["low"],
                search["lambda_diversity"]["high"],
                step=search["lambda_diversity"]["step"]
            )
        }
        
        # Temperature annealing
        cfg["mixture_temperature_schedule"] = {
            "start": trial.suggest_float("temp_start", search["temp_start"]["low"], search["temp_start"]["high"]),
            "end": trial.suggest_float("temp_end", search["temp_end"]["low"], search["temp_end"]["high"]),
            "anneal_frac": trial.suggest_float("temp_anneal_frac", search["temp_anneal_frac"]["low"], search["temp_anneal_frac"]["high"])
        }
        
        # Mixed precision
        cfg["use_amp"] = trial.suggest_categorical("use_amp", search["use_amp"])
        if cfg["use_amp"]:
            cfg["amp_dtype"] = trial.suggest_categorical("amp_dtype", search["amp_dtype"])
        
        # Data fraction
        cfg["use_fraction"] = trial.suggest_float(
            "use_fraction",
            search["use_fraction"]["low"],
            search["use_fraction"]["high"],
            step=search["use_fraction"]["step"]
        )
        
        # Add fixed HPO parameters
        cfg.update(HPO_TRAINING_CONFIG)
        
        return cfg
    
    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Set seed for reproducibility
        seed = self.base_config.get("system", {}).get("seed", 42)
        seed_everything(seed + trial.number)
        
        # Create trial configuration
        config = json.loads(json.dumps(self.base_config))
        config["model"].update(self.suggest_model_params(trial))
        config["training"].update(self.suggest_training_params(trial))
        config["optuna"] = {"enabled": True}
        
        # Create trial directory
        trial_dir = self.study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(trial_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Create model
            model = create_model(config, self.device)
            
            # Create normalization helper
            norm_helper = None
            if self.norm_stats:
                norm_helper = NormalizationHelper(self.norm_stats, self.device, config)
            
            # Create trainer with trial attached
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
            trainer.trial = trial  # Attach for pruning
            
            # Train model
            best_val_loss = trainer.train()
            
            self.logger.info(f"Trial {trial.number}: val_loss={best_val_loss:.6f}")
            
            # Save metadata
            if np.isfinite(best_val_loss):
                trial.set_user_attr("best_epoch", trainer.best_epoch)
                trial.set_user_attr("model_params", sum(p.numel() for p in model.parameters()))
            
            return best_val_loss
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
    
    def run(self) -> optuna.Study:
        """Run the hyperparameter optimization study."""
        self.logger.info(f"Starting HPO study: {self.study_name}")
        
        # Create sampler
        sampler = TPESampler(
            seed=self.base_config.get("system", {}).get("seed", 42),
            n_startup_trials=10,
            n_ei_candidates=24
        )
        
        # Create pruner
        if STUDY_SETTINGS["pruning"]:
            pruner = HyperbandPruner(
                min_resource=STUDY_SETTINGS["pruning_warmup_steps"],
                max_resource=HPO_TRAINING_CONFIG["epochs"],
                reduction_factor=3
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.study_dir}/study.db",
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=True
        )
        
        # Add default trial if new study
        if len(study.trials) == 0:
            study.enqueue_trial(DEFAULT_TRIAL)
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=STUDY_SETTINGS["n_trials"],
            timeout=STUDY_SETTINGS["timeout"],
            gc_after_trial=True
        )
        
        # Report results
        self._report_results(study)
        
        return study


    def _materialize_model_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {}
        cfg["variant"] = p["model_variant"]
        cfg["latent_dim"] = p["latent_dim"]

        n_enc = p["n_encoder_layers"]
        cfg["encoder_layers"] = [p[f"encoder_layer_{i}"] for i in range(n_enc)]

        n_dec = p["n_decoder_layers"]
        cfg["decoder_layers"] = [p[f"decoder_layer_{i}"] for i in range(n_dec)]

        cfg["activation"] = p["activation"]
        cfg["dropout"]   = p["dropout"]

        use_mixture = p["use_mixture"]
        if use_mixture:
            cfg["mixture"] = {
                "K": p["mixture_K"],
                "temperature": 1.0,
                "diversity_mode": p["diversity_mode"],
                "use_encoder_features": p["gate_use_features"],
            }
        else:
            cfg["mixture"] = {"K": 1}

        cfg["time_warp"] = {
            "enabled": p["use_time_warp"],
            "J_terms": p["warp_J_terms"],
            "hidden_dim": p["warp_hidden"],
            "use_encoder_features": p["warp_use_features"],
        }
        return cfg

    def _materialize_training_from_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {}
        cfg["optimizer"] = p["optimizer"]
        cfg["learning_rate"] = p["learning_rate"]
        cfg["weight_decay"] = p["weight_decay"]
        cfg["betas"] = [p["beta1"], p["beta2"]]

        cfg["batch_size"] = 2 ** int(p["batch_power"])
        cfg["gradient_accumulation_steps"] = int(p["grad_accum"])

        cfg["gradient_clip"] = p["gradient_clip"]

        cfg["scheduler"] = p["scheduler"]
        if cfg["scheduler"] == "cosine":
            cfg["scheduler_params"] = {
                "T_0": int(p["cosine_T0"]),
                "T_mult": int(p["cosine_Tmult"]),
                "eta_min": 1e-8,
            }
        elif cfg["scheduler"] == "plateau":
            cfg["scheduler_params"] = {
                "factor": p["plateau_factor"],
                "patience": int(p["plateau_patience"]),
                "min_lr": 1e-8,
            }

        cfg["regularization"] = {
            "lambda_entropy": p["lambda_entropy"],
            "lambda_diversity": p["lambda_diversity"],
        }

        cfg["mixture_temperature_schedule"] = {
            "start": p["temp_start"],
            "end": p["temp_end"],
            "anneal_frac": p["temp_anneal_frac"],
        }

        cfg["use_amp"] = p["use_amp"]
        if cfg["use_amp"]:
            cfg["amp_dtype"] = p["amp_dtype"]

        cfg["use_fraction"] = p["use_fraction"]

        # Fixed HPO settings:
        cfg.update(HPO_TRAINING_CONFIG)
        return cfg

    def _report_results(self, study: optuna.Study) -> None:
        """Report and save study results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("HYPERPARAMETER OPTIMIZATION RESULTS")
        self.logger.info("="*60)
        
        # Best trial
        best_trial = study.best_trial
        self.logger.info(f"Best trial: {best_trial.number}")
        self.logger.info(f"Best value: {best_trial.value:.6f}")
        
        # Best parameters (top 10)
        self.logger.info("\nTop parameters:")
        for i, (key, value) in enumerate(best_trial.params.items()):
            if i >= 10:
                self.logger.info(f"  ... and {len(best_trial.params) - 10} more")
                break
            self.logger.info(f"  {key}: {value}")
        
        # Save best configuration
        best_config = json.loads(json.dumps(self.base_config))
        best_params = best_trial.params
        best_config["model"].update(self._materialize_model_from_params(best_params))
        best_config["training"].update(self._materialize_training_from_params(best_params))
        
        best_config_path = self.study_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        self.logger.info(f"\nBest configuration saved to: {best_config_path}")
        
        # Copy best model
        best_trial_dir = self.study_dir / f"trial_{best_trial.number:04d}"
        if (best_trial_dir / "best_model.pt").exists():
            shutil.copy2(
                best_trial_dir / "best_model.pt",
                self.study_dir / "best_model.pt"
            )
            self.logger.info(f"Best model saved to: {self.study_dir / 'best_model.pt'}")
        
        # Statistics
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        self.logger.info(f"\nStatistics:")
        self.logger.info(f"  Completed: {len(complete)}/{len(study.trials)}")
        
        if complete:
            values = [t.value for t in complete if np.isfinite(t.value)]
            if values:
                self.logger.info(f"  Mean: {np.mean(values):.6f} ± {np.std(values):.6f}")
        
        self.logger.info("="*60)


def run_hpo(config: Dict[str, Any], data_dir: Path, output_dir: Path) -> None:
    """
    Entry point for hyperparameter optimization from main.py.
    
    Args:
        config: Base configuration dictionary
        data_dir: Directory containing preprocessed data
        output_dir: Output directory for results
    """
    tuner = HyperparameterTuner(
        base_config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        study_name="lilan_hpo"
    )
    
    study = tuner.run()
    
    # Log final summary
    logger = logging.getLogger(__name__)
    logger.info(f"\nOptimization complete. Best loss: {study.best_value:.6f}")
    logger.info(f"Results saved to: {tuner.study_dir}")