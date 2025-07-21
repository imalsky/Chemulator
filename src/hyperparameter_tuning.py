#!/usr/bin/env python3
"""
Hyperparameter tuning for chemical kinetics models using Optuna.
This version includes fixes for:
1. Effective pruning during training via callbacks.
2. Robust reconstruction of the best trial's configuration for saving.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
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
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        
    def __call__(self, epoch: int, val_loss: float) -> bool:
        """
        Report intermediate value to Optuna and check if should prune.
        
        Args:
            epoch: Current epoch number.
            val_loss: Validation loss for this epoch.
            
        Returns:
            True if the trial should be pruned, False otherwise.
        """
        self.trial.report(val_loss, epoch)
        
        if self.trial.should_prune():
            return True
        return False


class OptunaTrialRunner:
    """Manages the execution of a single Optuna trial."""
    def __init__(self, base_config_path: Path, mode_to_dir: Dict[str, Path]):
        self.base_config_path = base_config_path
        self.base_config = load_json_config(base_config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        self.mode_to_dir = mode_to_dir
        self._pipelines = {}

    def _get_pipeline_for_mode(self, mode: str) -> 'OptunaPipeline':
        if mode not in self._pipelines:
            self.logger.info(f"Loading pipeline for '{mode}' mode from preprocessed data.")
            mode_config = copy.deepcopy(self.base_config)
            mode_config["prediction"]["mode"] = mode
            mode_config["paths"]["processed_data_dir"] = str(self.mode_to_dir[mode])
            self._pipelines[mode] = OptunaPipeline(mode_config)
        return self._pipelines[mode]

    def run_trial(self, trial: optuna.Trial) -> float:
        """Configures and runs a single trial."""
        config = suggest_model_config(trial, self.base_config)
        prediction_mode = config["prediction"]["mode"]
        pipeline = self._get_pipeline_for_mode(prediction_mode)
        return pipeline.execute_trial(config, trial)


class OptunaPipeline:
    """Holds datasets and executes the training for a specific prediction mode."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = setup_device()
        self.logger = logging.getLogger(f"OptunaPipeline_{config['prediction']['mode']}")
        
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
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
            
            pruning_callback = OptunaPruningCallback(trial)
            
            trainer = PrunableTrainer(
                model=model, train_dataset=self.train_dataset,
                val_dataset=self.val_dataset, test_dataset=None,
                config=config, save_dir=save_dir, device=self.device,
                norm_helper=self.norm_helper, epoch_callback=pruning_callback
            )

            config["training"]["epochs"] = min(
                config["training"].get("hpo_epochs", 50), config["training"]["epochs"]
            )
            
            best_val_loss = trainer.train()

            trial.set_user_attr("full_config", config)
            save_json(config, save_dir / "config.json")
            
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

            if self.epoch_callback and self.has_validation:
                if self.epoch_callback(epoch, val_loss):
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

    prediction_mode = trial.suggest_categorical("prediction_mode", ["absolute", "ratio"])
    config["prediction"]["mode"] = prediction_mode

    if prediction_mode == "ratio":
        model_type = "deeponet"
    else:
        model_type = trial.suggest_categorical("model_type", ["deeponet", "siren"])
    config["model"]["type"] = model_type
    
    config["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
    config["training"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])

    if model_type == "deeponet":
        n_branch = trial.suggest_int("n_branch_layers", 2, 5)
        config["model"]["branch_layers"] = [trial.suggest_int(f"branch_layer_{i}", 64, 512, step=64) for i in range(n_branch)]
        n_trunk = trial.suggest_int("n_trunk_layers", 2, 4)
        config["model"]["trunk_layers"] = [trial.suggest_int(f"trunk_layer_{i}", 64, 256, step=32) for i in range(n_trunk)]
        config["model"]["basis_dim"] = trial.suggest_categorical("basis_dim", [64, 128, 256])
    else:  # SIREN
        n_layers = trial.suggest_int("n_hidden_layers", 3, 7)
        config["model"]["hidden_dims"] = [trial.suggest_int(f"hidden_dim_{i}", 128, 512, step=64) for i in range(n_layers)]
        config["model"]["omega_0"] = trial.suggest_float("omega_0", 20.0, 40.0)

    if trial.suggest_categorical("use_film", [True, False]):
        config["film"]["enabled"] = True
        n_film = trial.suggest_int("film_n_layers", 1, 3)
        config["film"]["hidden_dims"] = [trial.suggest_int(f"film_layer_{i}", 64, 256, step=32) for i in range(n_film)]
    else:
        config["film"]["enabled"] = False

    return config


def _reconstruct_config_from_params(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs the full config dictionary from a flat dictionary of Optuna parameters.
    This serves as a robust fallback for saving the best trial's configuration.
    """
    config = copy.deepcopy(base_config)
    
    # Simple direct mappings
    config["prediction"]["mode"] = params.get("prediction_mode", config["prediction"]["mode"])
    config["model"]["type"] = params.get("model_type", config["model"]["type"])
    config["model"]["activation"] = params.get("activation", config["model"]["activation"])
    config["training"]["learning_rate"] = params.get("lr", config["training"]["learning_rate"])
    config["training"]["batch_size"] = params.get("batch_size", config["training"]["batch_size"])
    config["film"]["enabled"] = params.get("use_film", config["film"]["enabled"])

    # Conditional logic for model type
    if config["prediction"]["mode"] == "ratio":
        config["model"]["type"] = "deeponet"

    # DeepONet specific parameters
    if config["model"]["type"] == "deeponet":
        if "n_branch_layers" in params:
            n = params["n_branch_layers"]
            config["model"]["branch_layers"] = [params[f"branch_layer_{i}"] for i in range(n)]
        if "n_trunk_layers" in params:
            n = params["n_trunk_layers"]
            config["model"]["trunk_layers"] = [params[f"trunk_layer_{i}"] for i in range(n)]
        if "basis_dim" in params:
            config["model"]["basis_dim"] = params["basis_dim"]

    # SIREN specific parameters
    elif config["model"]["type"] == "siren":
        if "n_hidden_layers" in params:
            n = params["n_hidden_layers"]
            config["model"]["hidden_dims"] = [params[f"hidden_dim_{i}"] for i in range(n)]
        if "omega_0" in params:
            config["model"]["omega_0"] = params["omega_0"]

    # FiLM specific parameters
    if config["film"]["enabled"]:
        if "film_n_layers" in params:
            n = params["film_n_layers"]
            config["film"]["hidden_dims"] = [params[f"film_layer_{i}"] for i in range(n)]
            
    return config


def optimize(config_path: Path, n_trials: int = 100, n_jobs: int = 1,
             study_name: str = "chemulator_hpo", pruner: Optional[optuna.pruners.BasePruner] = None):
    """
    Main function to run Optuna optimization with fixed pruning and result saving.
    """
    logger = logging.getLogger(__name__)
    base_config = load_json_config(config_path)
    
    possible_modes = ["absolute", "ratio"]
    mode_to_dir = {}
    for mode in possible_modes:
        mode_config = copy.deepcopy(base_config)
        mode_config["prediction"]["mode"] = mode
        processed_dir = Path(mode_config["paths"]["processed_data_dir"]) / f"mode_{mode}"
        mode_config["paths"]["processed_data_dir"] = str(processed_dir)
        
        pipeline = ChemicalKineticsPipeline(config_path)
        pipeline.config = mode_config
        pipeline.setup_paths()
        pipeline.processed_dir = processed_dir
        pipeline.preprocess_data()
        mode_to_dir[mode] = processed_dir

    trial_runner = OptunaTrialRunner(config_path, mode_to_dir)
    objective = trial_runner.run_trial

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10, interval_steps=2)

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=42), pruner=pruner,
        study_name=study_name, storage=f"sqlite:///{study_name}.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # --- Save Results ---
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Retrieve the exact config, using the new reconstruction method as a fallback
    best_config = study.best_trial.user_attrs.get("full_config", {})
    if not best_config:
        logger.warning("Could not retrieve full config from user_attrs. Reconstructing from best_params.")
        best_config = _reconstruct_config_from_params(base_config, study.best_trial.params)

    best_results = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_config": best_config,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "study_db": f"{study_name}.db"
    }
    
    save_json(best_results, results_dir / f"best_config_{study_name}_{timestamp}.json")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_results['best_value']:.6f}")
    print(f"Trials: {best_results['n_trials_completed']} completed, {best_results['n_trials_pruned']} pruned")
    print("\nBest parameters:")
    for key, value in best_results['best_params'].items():
        print(f"  {key}: {value}")
    
    return study