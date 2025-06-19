#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the MLP model.

This module provides functionality for automated hyperparameter optimization using Optuna.
The search space is dynamically configured through the configuration file, allowing for
flexible experimentation with different parameter combinations.

Key Features:
- Dynamic search space configuration via config file
- Support for independent and conditional hyperparameters
- Special handling for network architecture parameters
- Robust error handling with trial pruning
- SQLite-based study persistence
- Comprehensive logging of optimization progress
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna import Study, Trial
from optuna.exceptions import TrialPruned
from optuna.trial import FrozenTrial

# Constants
DEFAULT_STUDY_NAME = "mlp-dynamic-search-v1"
DEFAULT_NUM_TRIALS = 500
PRUNER_WARMUP_STEPS = 5
PRUNER_STARTUP_TRIALS = 5

logger = logging.getLogger(__name__)


def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a trial-specific configuration by suggesting hyperparameters.
    
    This function handles three types of hyperparameter optimization:
    1. Network architecture (special case for correlated parameters)
    2. Independent hyperparameters (generic optimization)
    3. Conditional hyperparameters (dependent on other parameter values)
    
    Args:
        trial: Optuna trial object for parameter suggestion
        base_cfg: Base configuration dictionary containing search space definitions
        
    Returns:
        Deep copy of base configuration with trial-specific parameter values
        
    Note:
        If a parameter is not in the search space, its default value from base_cfg is used.
    """
    # Create a deep copy to avoid modifying the original configuration
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg.get("optuna_hyperparam_search_space", {})

    # --- 1. Special Case: Network Architecture ---
    # Handle correlated parameters (num_layers + hidden_dims) together
    _suggest_network_architecture(trial, search_space, cfg)

    # --- 2. Generic Loop for Independent Hyperparameters ---
    # Handle parameters that don't depend on other parameter values
    _suggest_independent_hyperparams(trial, search_space, cfg)

    # --- 3. Explicit Handling for Conditional Hyperparameters ---
    # Handle parameters whose valid ranges depend on other parameter values
    _suggest_conditional_hyperparams(trial, search_space, cfg)

    return cfg


def _suggest_network_architecture(
    trial: Trial, 
    search_space: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> None:
    """
    Suggests network architecture parameters (num_hidden_layers and hidden_dims).
    
    This is handled as a special case because the number of hidden layers determines
    how many hidden dimension parameters need to be suggested.
    
    Args:
        trial: Optuna trial object
        search_space: Search space configuration
        cfg: Configuration dictionary to modify in-place
    """
    arch_space = search_space.get("architecture", {})
    
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        # Get configuration for number of layers and hidden dimensions
        num_layers_config = arch_space["num_hidden_layers"]
        hidden_dim_config = arch_space["hidden_dim"]
        
        # Suggest number of hidden layers
        num_layers = trial.suggest_int(
            "num_hidden_layers", 
            num_layers_config["low"], 
            num_layers_config["high"]
        )
        
        # Suggest hidden dimension for each layer
        hidden_dims = [
            trial.suggest_categorical(f"hidden_dim_l{i+1}", hidden_dim_config["choices"])
            for i in range(num_layers)
        ]
        
        cfg["hidden_dims"] = hidden_dims


def _suggest_independent_hyperparams(
    trial: Trial, 
    search_space: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> None:
    """
    Suggests independent hyperparameters that don't depend on other parameters.
    
    Supports three parameter types:
    - float: Continuous parameters with optional log scale
    - int: Integer parameters
    - categorical: Discrete choice parameters
    
    Args:
        trial: Optuna trial object
        search_space: Search space configuration
        cfg: Configuration dictionary to modify in-place
    """
    generic_params = search_space.get("hyperparameters", {})
    
    for name, params in generic_params.items():
        param_type = params["type"]
        
        if param_type == "float":
            is_log = params.get("log", False)
            cfg[name] = trial.suggest_float(
                name, params["low"], params["high"], log=is_log
            )
        elif param_type == "int":
            cfg[name] = trial.suggest_int(name, params["low"], params["high"])
        elif param_type == "categorical":
            cfg[name] = trial.suggest_categorical(name, params["choices"])


def _suggest_conditional_hyperparams(
    trial: Trial, 
    search_space: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> None:
    """
    Suggests conditional hyperparameters whose valid ranges depend on other parameters.
    
    Handles the following conditional dependencies:
    - time_embedding_dim: Only relevant if use_time_embedding is True
    - huber_delta: Only relevant if loss_function is 'huber'
    - lr_patience, lr_factor: Only relevant if scheduler_choice is 'plateau'
    - cosine_T_0: Only relevant if scheduler_choice is 'cosine'
    
    Args:
        trial: Optuna trial object
        search_space: Search space configuration
        cfg: Configuration dictionary to modify in-place
    """
    cond_params = search_space.get("conditional_hyperparameters", {})
    
    # Time embedding dimension (conditional on use_time_embedding)
    if cfg.get("use_time_embedding") and "time_embedding_dim" in cond_params:
        time_emb_config = cond_params["time_embedding_dim"]
        cfg["time_embedding_dim"] = trial.suggest_categorical(
            "time_embedding_dim", time_emb_config["choices"]
        )
    
    # Huber loss delta (conditional on loss_function == 'huber')
    if cfg.get("loss_function") == "huber" and "huber_delta" in cond_params:
        huber_config = cond_params["huber_delta"]
        cfg["huber_delta"] = trial.suggest_float(
            "huber_delta", huber_config["low"], huber_config["high"]
        )
    
    # Learning rate scheduler parameters
    _suggest_scheduler_params(trial, cond_params, cfg)


def _suggest_scheduler_params(
    trial: Trial, 
    cond_params: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> None:
    """
    Suggests scheduler-specific parameters based on the chosen scheduler.
    
    Args:
        trial: Optuna trial object
        cond_params: Conditional hyperparameters configuration
        cfg: Configuration dictionary to modify in-place
    """
    scheduler_choice = cfg.get("scheduler_choice")
    
    if scheduler_choice == "plateau":
        # ReduceLROnPlateau parameters
        if "lr_patience" in cond_params:
            patience_config = cond_params["lr_patience"]
            cfg["lr_patience"] = trial.suggest_int(
                "lr_patience", patience_config["low"], patience_config["high"]
            )
        
        if "lr_factor" in cond_params:
            factor_config = cond_params["lr_factor"]
            cfg["lr_factor"] = trial.suggest_float(
                "lr_factor", factor_config["low"], factor_config["high"]
            )
    
    elif scheduler_choice == "cosine" and "cosine_T_0" in cond_params:
        # CosineAnnealingWarmRestarts parameters
        cosine_config = cond_params["cosine_T_0"]
        cfg["cosine_T_0"] = trial.suggest_int(
            "cosine_T_0", cosine_config["low"], cosine_config["high"]
        )


def _log_best_trial_callback(study: Study, trial: FrozenTrial) -> None:
    """
    Callback function to log the current best trial after each trial completion.
    
    This provides real-time feedback on optimization progress, showing which
    trial is currently performing best and its validation loss value.
    
    Args:
        study: Optuna study object containing optimization history
        trial: The trial that just completed
    """
    if study.best_trial:
        logger.info(
            f"Current best trial is #{study.best_trial.number} "
            f"with value: {study.best_trial.value:.6f}"
        )


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir_root: Path,
    norm_data_dir: Path,
    train_model_func: Callable[[optuna.Trial, Dict[str, Any], Any, Path, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Path], bool],
) -> Optional[Dict[str, Any]]:
    """
    Executes the complete Optuna hyperparameter search process.
    
    This function orchestrates the entire hyperparameter optimization workflow:
    1. Creates an objective function that wraps the training process
    2. Sets up Optuna study with persistence and pruning
    3. Runs the optimization loop
    4. Extracts and saves the best configuration
    
    Args:
        base_config: Base configuration containing search space and default values
        data_dir_root: Root directory for saving optimization results
        norm_data_dir: Directory containing normalized training data
        train_model_func: Function to train model for a given configuration
        setup_device_func: Function to set up compute device (CPU/GPU)
        save_config_func: Function to save configuration to JSON file
    
    Returns:
        Best configuration found during optimization, or None if no trials succeeded
    
    Raises:
        Various exceptions may be raised by the training function or Optuna internals
    """
    def objective(trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        This function is called for each trial and must return a metric to minimize.
        It handles both successful trials and various failure modes.
        
        Args:
            trial: Optuna trial object for this optimization attempt
            
        Returns:
            Validation loss value to minimize
            
        Raises:
            TrialPruned: If trial should be terminated early or failed critically
        """
        # Generate trial-specific configuration
        trial_cfg = _suggest_hyperparams(trial, base_config)
        
        # Set up compute device for this trial
        device = setup_device_func()
        
        # Create trial-specific output directory
        tuning_dir = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number}"
        
        try:
            # Execute training with trial configuration
            return train_model_func(trial, trial_cfg, device, norm_data_dir, trial_dir)
        except TrialPruned:
            # Re-raise pruning signals (expected behavior)
            raise
        except Exception as e:
            # Handle unexpected critical errors during training
            logger.error(
                f"Trial {trial.number} failed critically: {e}", 
                exc_info=True
            )
            # Mark trial as invalid by raising TrialPruned
            raise TrialPruned("Training function raised a critical error.") from e

    # --- Study Configuration ---
    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", DEFAULT_STUDY_NAME)
    
    # Set up output directory for optimization results
    output_folder = data_dir_root / base_config["output_paths_config"]["tuning_results_foldername"]
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Configure persistent storage for study
    storage_path = output_folder / f"{study_name}.db"

    # --- Study Creation ---
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="minimize",  # Minimize validation loss
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=PRUNER_WARMUP_STEPS, 
            n_startup_trials=PRUNER_STARTUP_TRIALS
        ),
        load_if_exists=True,  # Resume existing study if available
    )

    # --- Optimization Execution ---
    try:
        study.optimize(
            objective,
            n_trials=optuna_cfg.get("num_trials", DEFAULT_NUM_TRIALS),
            gc_after_trial=True,  # Clean up memory after each trial
            show_progress_bar=True,
            callbacks=[_log_best_trial_callback]
        )
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted by user.")
    
    # --- Results Processing ---
    if not study.best_trial:
        logger.error("No trials completed successfully. Cannot determine best configuration.")
        return None

    # Log optimization results
    logger.info(f"Hyperparameter search complete. Best trial: {study.best_trial.number}")
    logger.info(f"  - Best Value (val_loss): {study.best_trial.value:.6f}")
    logger.info(f"  - Best Parameters: {study.best_trial.params}")
    
    # --- Best Configuration Assembly ---
    best_config = _assemble_best_config(base_config, study.best_trial)
    
    # Save best configuration
    if save_config_func(best_config, output_folder / "best_config.json"):
        logger.info(f"Best overall configuration saved to {output_folder / 'best_config.json'}")
         
    return best_config


def _assemble_best_config(
    base_config: Dict[str, Any], 
    best_trial: FrozenTrial
) -> Dict[str, Any]:
    """
    Assembles the best configuration from base config and optimal trial parameters.
    
    This function handles the special case of network architecture parameters,
    where multiple layer-specific parameters need to be combined into a single
    hidden_dims list.
    
    Args:
        base_config: Original base configuration
        best_trial: Best trial from Optuna optimization
        
    Returns:
        Complete configuration with optimal parameter values
    """
    best_config = copy.deepcopy(base_config)
    
    # Copy standard parameters (excluding layer-specific hidden_dim parameters)
    for key, value in best_trial.params.items():
        if not key.startswith("hidden_dim_l"):
            best_config[key] = value

    # Handle network architecture parameters specially
    num_layers = best_trial.params.get("num_hidden_layers")
    if num_layers is not None:
        # Reconstruct hidden_dims list from layer-specific parameters
        best_hidden_dims = [
            best_trial.params[f"hidden_dim_l{i+1}"]
            for i in range(num_layers)
        ]
        best_config["hidden_dims"] = best_hidden_dims

    return best_config


__all__ = ["run_hyperparameter_search"]