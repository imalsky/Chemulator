#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the chemical kinetics model.

This module provides functionality for automated hyperparameter optimization using Optuna.
The search space is dynamically configured through the main configuration file, allowing
for flexible experimentation with network architecture and training parameters.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
from optuna import Study, Trial
from optuna.exceptions import TrialPruned
from optuna.trial import FrozenTrial

# --- Constants ---
DEFAULT_STUDY_NAME = "chemical-kinetics-study"
DEFAULT_NUM_TRIALS = 100
PRUNER_WARMUP_STEPS = 5
PRUNER_STARTUP_TRIALS = 5
DEFAULT_GC_AFTER_TRIAL = True
DEFAULT_SHOW_PROGRESS_BAR = True

logger = logging.getLogger(__name__)


def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a trial-specific configuration by suggesting hyperparameters from the search space.
    This function delegates to specialized helpers for different parameter types.
    """
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg.get("optuna_hyperparam_search_space", {})

    _suggest_network_architecture(trial, search_space.get("architecture", {}), cfg)
    _suggest_independent_hyperparams(trial, search_space.get("hyperparameters", {}), cfg)
    _suggest_conditional_hyperparams(trial, search_space.get("conditional_hyperparameters", {}), cfg)

    return cfg


def _suggest_network_architecture(trial: Trial, arch_space: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Suggests network architecture (number of layers and their dimensions)."""
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        num_layers_cfg = arch_space["num_hidden_layers"]
        hidden_dim_cfg = arch_space["hidden_dim"]
        
        num_layers = trial.suggest_int("num_hidden_layers", num_layers_cfg["low"], num_layers_cfg["high"])
        
        # Suggest a dimension for each hidden layer individually
        hidden_dims = [
            trial.suggest_categorical(f"hidden_dim_l{i+1}", hidden_dim_cfg["choices"])
            for i in range(num_layers)
        ]
        cfg["hidden_dims"] = hidden_dims


def _suggest_independent_hyperparams(trial: Trial, hyperparam_space: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Suggests independent hyperparameters (float, int, categorical)."""
    for name, params in hyperparam_space.items():
        param_type = params.get("type")
        if param_type == "float":
            cfg[name] = trial.suggest_float(name, params["low"], params["high"], log=params.get("log", False))
        elif param_type == "int":
            cfg[name] = trial.suggest_int(name, params["low"], params["high"])
        elif param_type == "categorical":
            cfg[name] = trial.suggest_categorical(name, params["choices"])


def _suggest_conditional_hyperparams(trial: Trial, cond_space: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Suggests hyperparameters whose relevance depends on other suggested values."""
    # Example: time_embedding_dim is only relevant if use_time_embedding is True
    if cfg.get("use_time_embedding") and "time_embedding_dim" in cond_space:
        cfg["time_embedding_dim"] = trial.suggest_categorical(
            "time_embedding_dim", cond_space["time_embedding_dim"]["choices"]
        )
    
    # Suggest dropout, which could be considered conditional on the model type
    if "dropout" in cond_space:
        cfg["dropout"] = trial.suggest_float("dropout", cond_space["dropout"]["low"], cond_space["dropout"]["high"])

    # SIREN-specific parameters
    if cfg.get("model_type") == "siren":
        if "siren_w0_initial" in cond_space:
            cfg["siren_w0_initial"] = trial.suggest_float("siren_w0_initial", **cond_space["siren_w0_initial"])
        if "siren_w0_hidden" in cond_space:
            cfg["siren_w0_hidden"] = trial.suggest_categorical("siren_w0_hidden", **cond_space["siren_w0_hidden"])


def _log_best_trial_callback(study: Study, trial: FrozenTrial) -> None:
    """Callback to log the best trial's results after each trial finishes."""
    if study.best_trial:
        logger.info(f"Current best trial is #{study.best_trial.number} with value: {study.best_trial.value:.6f}")


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir_root: Path,
    norm_data_dir: Path,  # This now points to the RAW data directory
    train_model_func: Callable[[optuna.Trial, Dict[str, Any], Any, Path, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Path], bool],
) -> Optional[Dict[str, Any]]:
    """
    Executes the complete Optuna hyperparameter search process.

    This function orchestrates the entire workflow:
    1. Defines an objective function that wraps the model training process.
    2. Sets up an Optuna study with persistence (SQLite) and a pruner.
    3. Runs the optimization loop for a configured number of trials.
    4. Identifies, assembles, and saves the best performing configuration.
    """
    def objective(trial: Trial) -> float:
        """Objective function for Optuna, called for each trial."""
        trial_cfg = _suggest_hyperparams(trial, base_config)
        device = setup_device_func()
        
        tuning_dir = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number}"
        
        try:
            # The training function now takes the raw data directory
            return train_model_func(trial, trial_cfg, device, norm_data_dir, trial_dir)
        except TrialPruned:
            raise  # Re-raise to let Optuna handle pruning
        except Exception as e:
            logger.error(f"Trial {trial.number} failed critically: {e}", exc_info=True)
            # Prune the trial to avoid crashing the entire study
            raise TrialPruned("Training function raised a critical error.") from e

    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", DEFAULT_STUDY_NAME)
    
    output_folder = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
    output_folder.mkdir(parents=True, exist_ok=True)
    storage_path = output_folder / f"{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="minimize",  # We want to minimize validation loss
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=optuna_cfg.get("pruner_warmup_steps", PRUNER_WARMUP_STEPS),
            n_startup_trials=optuna_cfg.get("pruner_startup_trials", PRUNER_STARTUP_TRIALS)
        ),
        load_if_exists=True,
    )

    try:
        study.optimize(
            objective,
            n_trials=optuna_cfg.get("num_trials", DEFAULT_NUM_TRIALS),
            gc_after_trial=optuna_cfg.get("gc_after_trial", DEFAULT_GC_AFTER_TRIAL),
            show_progress_bar=optuna_cfg.get("show_progress_bar", DEFAULT_SHOW_PROGRESS_BAR),
            callbacks=[_log_best_trial_callback],
        )
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted by user.")
    
    if not study.best_trial:
        logger.error("No trials completed successfully. Cannot determine best configuration.")
        return None

    logger.info(f"Hyperparameter search complete. Best trial: #{study.best_trial.number}")
    logger.info(f"  - Best Value (val_loss): {study.best_trial.value:.6f}")
    logger.info(f"  - Best Parameters: {study.best_trial.params}")
    
    best_config = _assemble_best_config(base_config, study.best_trial)
    
    if save_config_func(best_config, output_folder / "best_config.json"):
        logger.info(f"Best overall configuration saved to {output_folder / 'best_config.json'}")
         
    return best_config


def _assemble_best_config(base_config: Dict[str, Any], best_trial: FrozenTrial) -> Dict[str, Any]:
    """Assembles the final best configuration from the base config and best trial parameters."""
    best_config = copy.deepcopy(base_config)
    
    # Update config with all non-architecture parameters from the best trial
    for key, value in best_trial.params.items():
        if not key.startswith("hidden_dim_l"):
            best_config[key] = value

    # Reconstruct the hidden_dims list from the individual layer parameters
    if "num_hidden_layers" in best_trial.params:
        num_layers = best_trial.params["num_hidden_layers"]
        best_hidden_dims = [
            best_trial.params[f"hidden_dim_l{i+1}"] for i in range(num_layers)
        ]
        best_config["hidden_dims"] = best_hidden_dims

    return best_config


__all__ = ["run_hyperparameter_search"]