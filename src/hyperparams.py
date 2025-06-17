#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the MLP model.
This version is driven dynamically by a descriptive search space in the
configuration file.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna import Trial
from optuna.exceptions import TrialPruned

logger = logging.getLogger(__name__)

def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new trial-specific configuration by suggesting hyperparameters
    based on the descriptive search space defined in the config file.
    """
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg["optuna_hyperparam_search_space"]

    # --- 1. Special Case: Network Architecture ---
    arch_space = search_space["architecture"]
    num_layers_config = arch_space["num_hidden_layers"]
    hidden_dim_config = arch_space["hidden_dim"]
    
    num_layers = trial.suggest_int("num_hidden_layers", num_layers_config["low"], num_layers_config["high"])
    
    hidden_dims = [
        trial.suggest_categorical(f"hidden_dim_l{i+1}", hidden_dim_config["choices"])
        for i in range(num_layers)
    ]
    cfg["hidden_dims"] = hidden_dims

    # --- 2. Generic Loop for Independent Hyperparameters ---
    generic_params = search_space["hyperparameters"]
    for name, params in generic_params.items():
        param_type = params["type"]
        if param_type == "float":
            is_log = params.get("log", False)
            cfg[name] = trial.suggest_float(name, params["low"], params["high"], log=is_log)
        elif param_type == "int":
            cfg[name] = trial.suggest_int(name, params["low"], params["high"])
        elif param_type == "categorical":
            cfg[name] = trial.suggest_categorical(name, params["choices"])

    # --- 3. Explicit Handling for Conditional Hyperparameters ---
    cond_params = search_space["conditional_hyperparameters"]
    
    # ============================ CHANGE: START ============================
    # If time embedding is used, suggest its dimension.
    if cfg["use_time_embedding"]:
        time_emb_config = cond_params["time_embedding_dim"]
        cfg["time_embedding_dim"] = trial.suggest_categorical("time_embedding_dim", time_emb_config["choices"])
    # ============================= CHANGE: END =============================
    
    # If loss is 'huber', suggest a delta.
    if cfg["loss_function"] == "huber":
        huber_config = cond_params["huber_delta"]
        cfg["huber_delta"] = trial.suggest_float("huber_delta", huber_config["low"], huber_config["high"])
        
    # If scheduler is 'plateau', suggest its specific parameters.
    if cfg["scheduler_choice"] == "plateau":
        patience_config = cond_params["lr_patience"]
        factor_config = cond_params["lr_factor"]
        cfg["lr_patience"] = trial.suggest_int("lr_patience", patience_config["low"], patience_config["high"])
        cfg["lr_factor"] = trial.suggest_float("lr_factor", factor_config["low"], factor_config["high"])
        
    # If scheduler is 'cosine', suggest its specific parameters.
    elif cfg["scheduler_choice"] == "cosine":
        cosine_config = cond_params["cosine_T_0"]
        cfg["cosine_T_0"] = trial.suggest_int("cosine_T_0", cosine_config["low"], cosine_config["high"])
        
    return cfg

def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir_root: Path,
    norm_data_dir: Path,
    train_model_func: Callable[[optuna.Trial, Dict[str, Any], Any, Path, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Path], bool],
) -> Optional[Dict[str, Any]]:
    """Executes the Optuna hyperparameter search process."""

    def objective(trial: Trial) -> float:
        trial_cfg = _suggest_hyperparams(trial, base_config)
        device = setup_device_func()
        tuning_dir = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number}"
        try:
            return train_model_func(trial, trial_cfg, device, norm_data_dir, trial_dir)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed critically: {e}", exc_info=True)
            raise TrialPruned("Training function raised a critical error.") from e

    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", "mlp-dynamic-search-v1")
    output_folder = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
    output_folder.mkdir(parents=True, exist_ok=True)
    storage_path = output_folder / f"{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
        load_if_exists=True,
    )

    try:
        study.optimize(
            objective,
            n_trials=optuna_cfg.get("num_trials", 500),
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
    
    # --- Reconstruct the best configuration from the winning trial's parameters ---
    best_config = copy.deepcopy(base_config)
    
    # Update all simple key-value hyperparameters from the best trial
    for key, value in study.best_trial.params.items():
        if not key.startswith("hidden_dim_l"):
            best_config[key] = value

    # Rebuild the `hidden_dims` list from the individual layer parameters
    num_layers = study.best_trial.params.get("num_hidden_layers")
    if num_layers is not None:
        best_hidden_dims = [
            study.best_trial.params.get(f"hidden_dim_l{i+1}")
            for i in range(num_layers)
        ]
        best_config["hidden_dims"] = best_hidden_dims

    if save_config_func(best_config, output_folder / "best_config.json"):
         logger.info(f"Best overall configuration saved to {output_folder / 'best_config.json'}")
         
    return best_config

__all__ = ["run_hyperparameter_search"]