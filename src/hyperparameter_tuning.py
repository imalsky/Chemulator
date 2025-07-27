#!/usr/bin/env python3
"""
Hyperparameter tuning for chemical kinetics models using Optuna.
Optimized for ~40 hour runtime with aggressive but smart pruning.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch

from main import ChemicalKineticsPipeline
from utils.hardware import setup_device, optimize_hardware
from data.dataset import NPYDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json, load_json


class OptunaPruningCallback:
    """Callback to report intermediate values to Optuna for pruning."""
    def __init__(self, trial: optuna.Trial, min_epochs: int = 10):
        self.trial = trial
        self.min_epochs = min_epochs  # Don't prune before this epoch
        
    def __call__(self, epoch: int, val_loss: float) -> bool:
        """
        Report intermediate value to Optuna and check if should prune.
        Only allows pruning after min_epochs to avoid conflict with cosine warmup.
        """
        self.trial.report(val_loss, epoch)
        
        # Don't prune during warmup period
        if epoch < self.min_epochs:
            return False
            
        if self.trial.should_prune():
            return True
        return False


class OptunaTrialRunner:
    """Manages the execution of a single Optuna trial."""
    def __init__(self, base_config_path: Path):
        self.base_config_path = base_config_path
        self.base_config = load_json_config(base_config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        self._pipelines = {}
        
        # Preprocess data for all possible modes upfront
        self._prepare_all_modes()

    def _prepare_all_modes(self):
        """Ensure data is preprocessed for all possible prediction modes."""
        # For absolute-only config, just prepare absolute mode
        mode = self.base_config["prediction"]["mode"]
        self.logger.info(f"Preparing data for '{mode}' mode...")
        
        pipeline = ChemicalKineticsPipeline(self.base_config)
        pipeline.normalize_only()
        
        self._pipelines[mode] = OptunaPipeline(self.base_config, pipeline.processed_dir)

    def run_trial(self, trial: optuna.Trial) -> float:
        """Configures and runs a single trial."""
        config = suggest_model_config(trial, self.base_config)
        prediction_mode = config["prediction"]["mode"]
        pipeline = self._pipelines[prediction_mode]
        return pipeline.execute_trial(config, trial)


class OptunaPipeline:
    """Holds datasets and executes the training for a specific prediction mode."""
    def __init__(self, config: Dict[str, Any], processed_dir: Path):
        self.config = config
        self.device = setup_device()
        self.logger = logging.getLogger(f"OptunaPipeline_{config['prediction']['mode']}")
        
        self.processed_dir = processed_dir
        self.model_save_root = Path(self.config["paths"]["model_save_dir"])
        
        norm_stats_path = self.processed_dir / "normalization.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found in {norm_stats_path}")
        norm_stats = load_json(norm_stats_path)
        
        self.norm_helper = NormalizationHelper(
            stats=norm_stats, device=self.device,
            species_vars=self.config["data"]["species_variables"],
            global_vars=self.config["data"]["global_variables"],
            time_var=self.config["data"]["time_variable"],
            config=self.config
        )
        self._load_datasets()

    def _load_datasets(self):
        """Loads datasets from the mode-specific directory."""
        self.logger.info(f"Loading datasets from: {self.processed_dir}")
        train_indices = np.load(self.processed_dir / "train_indices.npy")
        val_indices = np.load(self.processed_dir / "val_indices.npy")
        
        self.train_dataset = NPYDataset(self.processed_dir, train_indices, self.config, self.device, "train")
        self.val_dataset = NPYDataset(self.processed_dir, val_indices, self.config, self.device, "validation")
        self.logger.info(f"Datasets loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

    def execute_trial(self, config: Dict[str, Any], trial: optuna.Trial) -> float:
        """Runs a single trial's training and evaluation with pruning."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}_{config['prediction']['mode']}"
        save_dir = self.model_save_root / "optuna" / f"{timestamp}_{trial_id}"
        ensure_directories(save_dir)
        
        try:
            seed_everything(config["system"]["seed"])
            optimize_hardware(config["system"], self.device)
            model = create_model(config, self.device)
            
            # Use dynamic epoch allocation from Hyperband
            n_epochs = trial.user_attrs.get("n_epochs", config["training"]["hpo_max_epochs"])
            config["training"]["epochs"] = n_epochs
            
            # Create pruning callback with minimum epochs
            min_epochs = config["training"]["hpo_min_epochs"]
            pruning_callback = OptunaPruningCallback(trial, min_epochs)
            
            # Log trial configuration
            self.logger.info(f"Trial {trial.number}: {n_epochs} epochs allocated by Hyperband")
            self.logger.info(f"Learning rate: {config['training']['learning_rate']:.2e}")
            
            trainer = PrunableTrainer(
                model=model, train_dataset=self.train_dataset,
                val_dataset=self.val_dataset, test_dataset=None,
                config=config, save_dir=save_dir, device=self.device,
                norm_helper=self.norm_helper, epoch_callback=pruning_callback
            )
            
            best_val_loss = trainer.train()

            trial.set_user_attr("full_config", config)
            trial.set_user_attr("final_lr", trainer.optimizer.param_groups[0]['lr'])
            save_json(config, save_dir / "config.json")
            
            self.logger.info(f"Trial {trial.number} completed. Best loss: {best_val_loss:.6f}, "
                           f"Final LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
            
            return best_val_loss
            
        except optuna.TrialPruned:
            self.logger.info(f"Trial {trial.number} pruned.")
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return float("inf")
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class PrunableTrainer(Trainer):
    """Extended Trainer that supports epoch callbacks for Optuna pruning."""
    def __init__(self, *args, epoch_callback: Optional[Callable[[int, float], bool]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = epoch_callback
        
    def _run_training_loop(self):
        """Main training loop with pruning support."""
        best_train_loss = float("inf")
        
        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._validate()

            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)

            if self.epoch_callback:
                loss_for_pruning = val_loss if self.has_validation and val_loss != float("inf") else train_loss
                
                if self.epoch_callback(epoch, loss_for_pruning):
                    self.logger.info(f"Trial pruned at epoch {epoch} with loss {loss_for_pruning:.6f}")
                    raise optuna.TrialPruned()

            if self.has_validation:
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                if train_loss < (best_train_loss - self.min_delta):
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                    self.best_epoch = epoch
                    self._save_best_model()


def suggest_model_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggests a valid model and training configuration for a trial."""
    config = copy.deepcopy(base_config)

    # For absolute-only mode
    config["prediction"]["mode"] = "absolute"
    
    # Model architecture choice
    model_type = trial.suggest_categorical("model_type", ["deeponet", "siren"])
    config["model"]["type"] = model_type
    
    # Common hyperparameters
    config["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
    config["training"]["learning_rate"] = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Gradient accumulation adjustment based on batch size
    if config["training"]["batch_size"] <= 2048:
        config["training"]["gradient_accumulation_steps"] = 4
    elif config["training"]["batch_size"] <= 4096:
        config["training"]["gradient_accumulation_steps"] = 2
    else:
        config["training"]["gradient_accumulation_steps"] = 1

    if model_type == "deeponet":
        # Simplified architecture search
        n_branch = trial.suggest_int("n_branch_layers", 2, 4)
        branch_width = trial.suggest_categorical("branch_width", [128, 256, 384])
        config["model"]["branch_layers"] = [branch_width] * n_branch
        
        n_trunk = trial.suggest_int("n_trunk_layers", 2, 3)
        trunk_width = trial.suggest_categorical("trunk_width", [64, 128, 192])
        config["model"]["trunk_layers"] = [trunk_width] * n_trunk
        
        config["model"]["basis_dim"] = trial.suggest_categorical("basis_dim", [64, 128, 192])
        
    else:  # SIREN
        n_layers = trial.suggest_int("n_hidden_layers", 3, 5)
        layer_width = trial.suggest_categorical("layer_width", [128, 256, 384])
        config["model"]["hidden_dims"] = [layer_width] * n_layers
        config["model"]["omega_0"] = trial.suggest_float("omega_0", 20.0, 40.0)

    # FiLM configuration
    use_film = trial.suggest_categorical("use_film", [True, False])
    config["film"]["enabled"] = use_film
    if use_film:
        film_layers = trial.suggest_int("film_layers", 1, 2)
        film_width = trial.suggest_categorical("film_width", [32, 64, 128])
        config["film"]["hidden_dims"] = [film_width] * film_layers

    # Loss function
    config["training"]["loss"] = trial.suggest_categorical("loss", ["mse", "huber"])
    if config["training"]["loss"] == "huber":
        config["training"]["huber_delta"] = trial.suggest_float("huber_delta", 0.1, 1.0)

    return config


def optimize(config_path: Path, n_trials: int = 25, n_jobs: int = 1,
             study_name: str = "chemulator_hpo", pruner: Optional[optuna.pruners.BasePruner] = None):
    """
    Main function to run Optuna optimization with Hyperband for 40-hour runtime.
    """
    logger = logging.getLogger(__name__)
    base_config = load_json_config(config_path)
    
    # Create trial runner which will prepare data
    trial_runner = OptunaTrialRunner(config_path)
    objective = trial_runner.run_trial

    # Use Hyperband pruner for efficient resource allocation
    if pruner is None:
        min_resource = base_config["training"]["hpo_min_epochs"]
        max_resource = base_config["training"]["hpo_max_epochs"]
        
        pruner = HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=3
        )
        
        logger.info(f"Using Hyperband pruner: min={min_resource} epochs, max={max_resource} epochs")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True
    )

    # Log expected runtime
    logger.info(f"Starting optimization with target {n_trials} trials")
    logger.info(f"Expected runtime: ~40 hours with aggressive pruning")

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Save Results
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    best_config = study.best_trial.user_attrs.get("full_config", {})
    if not best_config:
        logger.warning("Could not retrieve full config from user_attrs.")

    # Compute statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    epoch_distribution = {}
    for t in completed_trials + pruned_trials:
        n_epochs = t.user_attrs.get("n_epochs", "unknown")
        epoch_distribution[n_epochs] = epoch_distribution.get(n_epochs, 0) + 1

    best_results = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_config": best_config,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "epoch_distribution": epoch_distribution,
        "best_trial_final_lr": study.best_trial.user_attrs.get("final_lr", "unknown"),
        "study_db": f"{study_name}.db"
    }
    
    save_json(best_results, results_dir / f"best_config_{study_name}_{timestamp}.json")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_results['best_value']:.6f}")
    print(f"Best trial final LR: {best_results['best_trial_final_lr']}")
    print(f"Trials: {best_results['n_trials_completed']} completed, {best_results['n_trials_pruned']} pruned")
    print("\nEpoch distribution:")
    for epochs, count in sorted(epoch_distribution.items()):
        print(f"  {epochs} epochs: {count} trials")
    print("\nBest parameters:")
    for key, value in best_results['best_params'].items():
        print(f"  {key}: {value}")
    
    return study