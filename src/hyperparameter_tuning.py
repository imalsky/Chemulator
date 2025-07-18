#!/usr/bin/env python3
"""Hyperparameter tuning for chemical kinetics models using Optuna."""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch

from utils.hardware import setup_device, optimize_hardware
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json
from data.dataset import NPYDataset
from models.model import create_model
from training.trainer import Trainer


class OptunaPipeline:
    """Reusable pipeline for Optuna trials with shared datasets."""
    
    def __init__(self, base_config_path: Path):
        self.base_config = load_json_config(base_config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.processed_dir = Path(self.base_config["paths"]["processed_data_dir"])
        self.model_save_root = Path(self.base_config["paths"]["model_save_dir"])
        
        # Load datasets once
        self._load_datasets()
        
    def _load_datasets(self):
        """Load datasets once for reuse across trials."""
        self.logger.info("Loading datasets for Optuna optimization...")
        
        # Load indices
        train_indices = np.load(self.processed_dir / "train_indices.npy")
        val_indices = np.load(self.processed_dir / "val_indices.npy")
        test_indices = np.load(self.processed_dir / "test_indices.npy")
        
        # Create datasets
        self.train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=train_indices,
            config=self.base_config,
            device=self.device,
            split_name="train"
        )
        
        self.val_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=val_indices,
            config=self.base_config,
            device=self.device,
            split_name="validation"
        ) if len(val_indices) > 0 else None
        
        self.test_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=test_indices,
            config=self.base_config,
            device=self.device,
            split_name="test"
        ) if len(test_indices) > 0 else None
        
        self.logger.info(f"Datasets loaded: train={len(self.train_dataset)}, "
                        f"val={len(self.val_dataset) if self.val_dataset else 0}, "
                        f"test={len(self.test_dataset) if self.test_dataset else 0}")
    
    def run_trial(self, config: Dict[str, Any], trial: optuna.Trial) -> float:
        """Run a single trial with the given config."""
        # Create unique directory for this trial
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}"
        save_dir = self.model_save_root / f"optuna_{timestamp}_{trial_id}"
        ensure_directories(save_dir)
        
        # Setup logging for this trial
        log_file = save_dir / "trial.log"
        trial_logger = logging.getLogger(f"trial_{trial.number}")
        
        try:
            # Set random seed
            seed_everything(config["system"]["seed"])
            
            # Apply hardware optimizations
            optimize_hardware(config["system"], self.device)
            
            # Create model
            model = create_model(config, self.device)
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trial_logger.info(f"Model parameters: {total_params:,}")
            
            # Create trainer with shared datasets
            trainer = Trainer(
                model=model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                config=config,
                save_dir=save_dir,
                device=self.device
            )
            
            # Train model with reduced epochs for hyperparameter search
            original_epochs = config["training"]["epochs"]
            config["training"]["epochs"] = min(50, original_epochs)  # Cap at 50 for search
            
            best_val_loss = trainer.train()
            
            # Save trial config
            save_json(config, save_dir / "config.json")
            
            # Report intermediate values for pruning
            for epoch_data in trainer.training_history["epochs"]:
                trial.report(epoch_data["val_loss"], epoch_data["epoch"])
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
            
        except optuna.TrialPruned:
            trial_logger.info(f"Trial {trial.number} pruned")
            raise
            
        except Exception as e:
            trial_logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            raise optuna.TrialPruned(f"Trial failed: {str(e)}")
            
        finally:
            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def suggest_model_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest model architecture and training hyperparameters."""
    config = copy.deepcopy(base_config)
    
    # Model type and prediction mode
    model_type = trial.suggest_categorical("model_type", ["deeponet", "siren"])
    config["model"]["type"] = model_type
    
    # Prediction mode (SIREN only supports absolute)
    if model_type == "deeponet":
        config["prediction"]["mode"] = trial.suggest_categorical("prediction_mode", ["absolute", "delta"])
    else:
        config["prediction"]["mode"] = "absolute"
    
    # Common parameters
    config["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    
    # Model-specific architecture
    if model_type == "deeponet":
        # Branch network
        n_branch = trial.suggest_int("n_branch_layers", 2, 5)
        branch_layers = [
            trial.suggest_int(f"branch_layer_{i}", 64, 512, step=64) 
            for i in range(n_branch)
        ]
        config["model"]["branch_layers"] = branch_layers
        
        # Trunk network
        n_trunk = trial.suggest_int("n_trunk_layers", 2, 4)
        trunk_layers = [
            trial.suggest_int(f"trunk_layer_{i}", 32, 256, step=32) 
            for i in range(n_trunk)
        ]
        config["model"]["trunk_layers"] = trunk_layers
        config["model"]["basis_dim"] = trial.suggest_int("basis_dim", 32, 128, step=16)
        
    else:  # SIREN
        n_layers = trial.suggest_int("n_hidden_layers", 3, 6)
        hidden_dims = [
            trial.suggest_int(f"hidden_dim_{i}", 128, 512, step=64) 
            for i in range(n_layers)
        ]
        config["model"]["hidden_dims"] = hidden_dims
        config["model"]["omega_0"] = trial.suggest_float("omega_0", 10.0, 50.0)
    
    # FiLM layers
    if trial.suggest_categorical("use_film", [True, False]):
        config["film"]["enabled"] = True
        n_film = trial.suggest_int("film_n_layers", 1, 3)
        film_dims = [
            trial.suggest_int(f"film_layer_{i}", 64, 256, step=32) 
            for i in range(n_film)
        ]
        config["film"]["hidden_dims"] = film_dims
    else:
        config["film"]["enabled"] = False
    
    # Training hyperparameters
    config["training"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config["training"]["batch_size"] = trial.suggest_int("batch_size", 2048, 16384, step=2048)
    
    # Loss function
    loss_type = trial.suggest_categorical("loss", ["mse", "mae", "huber"])
    config["training"]["loss"] = loss_type
    if loss_type == "huber":
        config["training"]["huber_delta"] = trial.suggest_float("huber_delta", 0.1, 1.0)
    
    # Scheduler
    scheduler = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
    config["training"]["scheduler"] = scheduler
    if scheduler == "cosine":
        config["training"]["scheduler_params"] = {
            "T_0": trial.suggest_int("cosine_T_0", 1, 10),
            "T_mult": trial.suggest_int("cosine_T_mult", 1, 2),
        }
    
    return config


def optimize(config_path: Path, n_trials: int = 100, n_jobs: int = 1, 
             study_name: str = "chemical_kinetics_opt", pruner: Optional[optuna.pruners.BasePruner] = None):
    """Run Optuna optimization with dataset reuse."""
    # Create pipeline with shared datasets
    pipeline = OptunaPipeline(config_path)
    
    def objective(trial):
        # Get suggested config
        config = suggest_model_config(trial, pipeline.base_config)
        
        # Run trial with shared datasets
        return pipeline.run_trial(config, trial)
    
    # Default pruner if not specified
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=pruner,
        study_name=study_name
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Save results
    results_dir = Path("optuna_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    study_path = results_dir / f"{study_name}_{timestamp}.pkl"
    
    # Save study
    import pickle
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    # Save best config
    best_config = suggest_model_config(study.best_trial, pipeline.base_config)
    best_results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": best_config,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "study_file": str(study_path)
    }
    
    save_json(best_results, results_dir / f"best_config_{timestamp}.json")
    
    return study