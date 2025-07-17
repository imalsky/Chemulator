#!/usr/bin/env python3
"""
Hyperparameter optimization using Optuna for chemical kinetics models.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import copy

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import torch
import numpy as np

from models.model import create_model
from training.trainer import Trainer
from data.dataset import NPYDataset, create_dataloader
from utils.hardware import setup_device, optimize_hardware
from utils.utils import seed_everything, save_json, load_json


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna for chemical kinetics models.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        processed_dir: Path,
        save_dir: Path,
        device: torch.device
    ):
        self.base_config = base_config
        self.processed_dir = processed_dir
        self.save_dir = save_dir
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Get optuna configuration
        self.optuna_config = base_config["optuna"]
        
        # Create study directory
        self.study_dir = save_dir / "optuna_study"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Load normalization stats for dataset creation
        norm_path = self.processed_dir / "normalization.json"
        if norm_path.exists():
            self.norm_stats = load_json(norm_path)
        else:
            self.logger.warning("Normalization stats not found")
            self.norm_stats = None
    
    def _parse_search_space(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Parse search space configuration and suggest values."""
        suggested_params = {}
        
        for param_path, param_config in search_space.items():
            param_type = param_config["type"]
            
            if param_type == "fixed":
                value = param_config["value"]
            elif param_type == "float":
                value = trial.suggest_float(
                    param_path,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                value = trial.suggest_int(
                    param_path,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "categorical":
                value = trial.suggest_categorical(
                    param_path,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            suggested_params[param_path] = value
        
        return suggested_params
    
    def _update_config_with_params(self, config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration dictionary with suggested parameters."""
        config = copy.deepcopy(config)
        
        for param_path, value in params.items():
            # Split path (e.g., "model.learning_rate" -> ["model", "learning_rate"])
            keys = param_path.split(".")
            
            # Navigate to the correct location in config
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
        
        return config
    
    def _create_datasets(self, config: Dict[str, Any]) -> tuple:
        """Create datasets for training."""
        train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=np.load(self.processed_dir / "train_indices.npy"),
            config=config,
            device=self.device,
            split_name="train"
        )
        
        val_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=np.load(self.processed_dir / "val_indices.npy"),
            config=config,
            device=self.device,
            split_name="validation"
        )
        
        # For hyperparameter search, we typically don't use test set
        # to avoid overfitting hyperparameters to test performance
        return train_dataset, val_dataset
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss to minimize
        """
        # Log trial start
        self.logger.info(f"Starting trial {trial.number}")
        trial_start = time.time()
        
        # Suggest hyperparameters
        suggested_params = self._parse_search_space(
            trial, 
            self.optuna_config["search_space"]
        )
        
        # Update configuration
        trial_config = self._update_config_with_params(
            self.base_config,
            suggested_params
        )
        
        # Set seed for reproducibility
        seed = self.base_config["system"]["seed"] + trial.number
        seed_everything(seed)
        
        # Create model
        model = create_model(trial_config, self.device)
        
        # Create datasets
        train_dataset, val_dataset = self._create_datasets(trial_config)
        
        # Create trial-specific save directory
        trial_save_dir = self.study_dir / f"trial_{trial.number:04d}"
        trial_save_dir.mkdir(exist_ok=True)
        
        # Save trial configuration
        save_json(trial_config, trial_save_dir / "config.json")
        save_json(suggested_params, trial_save_dir / "suggested_params.json")
        
        # Initialize trainer with pruning callback
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=None,  # Don't use test set during hyperparameter search
            config=trial_config,
            save_dir=trial_save_dir,
            device=self.device
        )
        
        # Training with pruning
        best_val_loss = float('inf')
        pruned = False
        
        try:
            # Modified training loop to support pruning
            for epoch in range(1, trial_config["training"]["epochs"] + 1):
                trainer.current_epoch = epoch
                
                # Train one epoch
                train_loss, train_metrics = trainer._train_epoch()
                
                # Validate
                val_loss, val_metrics = trainer._validate()
                
                # Update scheduler
                if not trainer.scheduler_step_on_batch:
                    trainer.scheduler.step(val_loss)
                
                # Track best loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Report to Optuna and check for pruning
                trial.report(val_loss, epoch)
                
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    pruned = True
                    raise optuna.TrialPruned()
                
                # Early stopping check
                if hasattr(trainer, 'patience_counter'):
                    if trainer.patience_counter >= trainer.early_stopping_patience:
                        self.logger.info(f"Trial {trial.number} early stopped at epoch {epoch}")
                        break
        
        except optuna.TrialPruned:
            pruned = True
            raise
        
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            raise
        
        finally:
            # Save trial results
            trial_time = time.time() - trial_start
            results = {
                "trial_number": trial.number,
                "best_val_loss": best_val_loss,
                "pruned": pruned,
                "suggested_params": suggested_params,
                "trial_time": trial_time
            }
            save_json(results, trial_save_dir / "results.json")
            
            # Clean up GPU memory
            del model
            del trainer
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return best_val_loss
    
    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Returns:
            Optuna study object
        """
        self.logger.info("="*80)
        self.logger.info("Starting Optuna hyperparameter optimization")
        self.logger.info(f"Study name: {self.optuna_config['study_name']}")
        self.logger.info(f"N trials: {self.optuna_config['n_trials']}")
        self.logger.info(f"Direction: {self.optuna_config['direction']}")
        
        # Create sampler
        sampler_type = self.optuna_config.get("sampler", "TPE")
        if sampler_type == "TPE":
            sampler = TPESampler(seed=self.base_config["system"]["seed"])
        elif sampler_type == "Random":
            sampler = RandomSampler(seed=self.base_config["system"]["seed"])
        elif sampler_type == "Grid":
            sampler = GridSampler(self._create_grid_search_space())
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}")
        
        # Create pruner
        pruner_type = self.optuna_config.get("pruner", "MedianPruner")
        if pruner_type == "MedianPruner":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_type == "HyperbandPruner":
            pruner = HyperbandPruner()
        elif pruner_type == "None" or pruner_type is None:
            pruner = None
        else:
            raise ValueError(f"Unknown pruner: {pruner_type}")
        
        # Create or load study
        study_path = self.study_dir / "study.db"
        storage = f"sqlite:///{study_path}"
        
        study = optuna.create_study(
            study_name=self.optuna_config["study_name"],
            direction=self.optuna_config["direction"],
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )
        
        # Run optimization
        optimization_start = time.time()
        
        study.optimize(
            self.objective,
            n_trials=self.optuna_config["n_trials"],
            n_jobs=self.optuna_config.get("n_jobs", 1),
            gc_after_trial=True
        )
        
        optimization_time = time.time() - optimization_start
        
        # Log results
        self.logger.info("="*80)
        self.logger.info("Optimization completed!")
        self.logger.info(f"Total time: {optimization_time/3600:.2f} hours")
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_value:.6f}")
        self.logger.info("Best parameters:")
        for param, value in study.best_params.items():
            self.logger.info(f"  {param}: {value}")
        
        # Save study results
        study_results = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "optimization_time": optimization_time
        }
        save_json(study_results, self.study_dir / "study_results.json")
        
        # Save best configuration
        best_config = self._update_config_with_params(
            self.base_config,
            study.best_params
        )
        save_json(best_config, self.study_dir / "best_config.json")
        
        return study
    
    def _create_grid_search_space(self) -> Dict[str, Any]:
        """Create grid search space from configuration."""
        grid_space = {}
        
        for param_path, param_config in self.optuna_config["search_space"].items():
            if param_config["type"] == "categorical":
                grid_space[param_path] = param_config["choices"]
            elif param_config["type"] == "fixed":
                grid_space[param_path] = [param_config["value"]]
            else:
                self.logger.warning(
                    f"Grid search only supports categorical parameters, "
                    f"skipping {param_path}"
                )
        
        return grid_space