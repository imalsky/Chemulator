#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the MLP model.
This version tunes the network architecture and dropout rate.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import optuna
from optuna import Trial
from optuna.exceptions import TrialPruned

from utils import save_json

logger = logging.getLogger(__name__)

def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new trial-specific configuration by suggesting MLP architecture
    and dropout hyperparameters.
    """
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg.get("optuna_hyperparam_search_space", {})

    # --- MLP Architecture Hyperparameters ---
    
    # 1. Suggest the number of hidden layers
    num_layers_range = search_space.get("num_hidden_layers_range", [2, 6])
    num_layers = trial.suggest_int("num_hidden_layers", *num_layers_range)

    # 2. Suggest the size of the hidden layers (all will be the same size)
    hidden_dim_choices = search_space.get("hidden_dim_size", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim_size", hidden_dim_choices)
    
    # Create the list of hidden dimensions for the model config
    cfg["hidden_dims"] = [hidden_dim] * num_layers

    # 3. Suggest dropout rate
    dropout_range = search_space.get("dropout_range", [0.0, 0.3])
    cfg["dropout"] = trial.suggest_float("dropout", *dropout_range)

    return cfg

def run_hyperparameter_search(
    base_config: Dict[str, Any], 
    data_dir_root: Path,
    norm_data_dir: Path,
    train_model_func: Callable[[optuna.Trial, Dict[str, Any], Any, Path, Path], float],
    setup_device_func: Callable[[], Any], 
    save_config_func: Callable[[Dict[str, Any], Path], bool],
) -> Optional[Dict[str, Any]]:
    """Executes the Optuna hyperparameter search process for the MLP model."""

    def objective(trial: Trial) -> float:
        trial_cfg = _suggest_hyperparams(trial, base_config)
        
        device = setup_device_func()
        tuning_dir = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number}"

        try:
            # The train function now takes the data directory and handles its own dataset creation.
            return train_model_func(trial, trial_cfg, device, norm_data_dir, trial_dir)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed critically: {e}", exc_info=True)
            raise TrialPruned("Training function raised a critical error.") from e

    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", "mlp-arch-tuning")
    output_folder = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
    output_folder.mkdir(parents=True, exist_ok=True)
    storage_path = output_folder / f"{study_name}.db"

    study = optuna.create_study(
        study_name=study_name, 
        storage=f"sqlite:///{storage_path}",
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=3),
        load_if_exists=True,
    )

    try:
        study.optimize(
            objective, 
            n_trials=optuna_cfg.get("num_trials", 50),
            gc_after_trial=True, 
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted by user.")
    
    if not study.best_trial:
        logger.error("No trials completed successfully. Cannot determine best configuration.")
        return None

    logger.info(f"Hyperparameter search complete. Best trial: {study.best_trial.number}")
    logger.info(f"  - Best Value (val_loss): {study.best_trial.value:.6f}")
    logger.info(f"  - Best Parameters: {study.best_trial.params}")
    
    best_config = copy.deepcopy(base_config)
    best_params = study.best_trial.params
    
    # Reconstruct the config from the best trial's parameters
    num_layers = best_params.pop("num_hidden_layers")
    hidden_dim = best_params.pop("hidden_dim_size")
    best_config["hidden_dims"] = [hidden_dim] * num_layers
    best_config.update(best_params)

    if save_config_func(best_config, output_folder / "best_config.json"):
         logger.info(f"Best overall configuration saved to {output_folder / 'best_config.json'}")
         
    return best_config

__all__ = ["run_hyperparameter_search"]