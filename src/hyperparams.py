#!/usr/bin/env python3
"""
hyperparams.py – Simplified Optuna-based hyperparameter tuning.
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

from hardware import setup_device
from train import ModelTrainer
from utils import save_json
from normalizer import DataNormalizer

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STUDY_NAME = "chemical-kinetics-study"
DEFAULT_NUM_TRIALS = 100
PRUNER_WARMUP_STEPS = 5
PRUNER_STARTUP_TRIALS = 5
DEFAULT_GC_AFTER_TRIAL = True
DEFAULT_SHOW_PROGRESS_BAR = True


def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a trial-specific configuration by suggesting hyperparameters.
    
    This function modifies a copy of the base configuration with
    hyperparameter values suggested by Optuna for the current trial.
    """
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg.get("optuna_hyperparam_search_space", {})
    model_params = cfg["model_hyperparameters"]
    train_params = cfg["training_hyperparameters"]

    # Suggest network architecture
    arch_space = search_space.get("architecture", {})
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        num_layers_cfg = arch_space["num_hidden_layers"]
        hidden_dim_cfg = arch_space["hidden_dim"]
        
        # Suggest number of layers
        num_layers = trial.suggest_int(
            "num_hidden_layers", 
            num_layers_cfg["low"], 
            num_layers_cfg["high"]
        )
        
        # Suggest dimension for each layer
        model_params["hidden_dims"] = [
            trial.suggest_categorical(f"hidden_dim_l{i+1}", hidden_dim_cfg["choices"])
            for i in range(num_layers)
        ]

    # Suggest independent hyperparameters
    for name, params in search_space.get("hyperparameters", {}).items():
        # Skip removed features
        if name in ("use_ema", "ema_decay", "optimizer"):
            continue
            
        suggested_val = None
        
        if params["type"] == "categorical":
            suggested_val = trial.suggest_categorical(name, params["choices"])
        elif params["type"] == "float":
            suggested_val = trial.suggest_float(
                name, params["low"], params["high"], log=params.get("log", False)
            )
        elif params["type"] == "int":
            if params.get("log", False):
                # Log-scale integer by using float and converting
                suggested_val = int(trial.suggest_float(
                    name, params["low"], params["high"], log=True
                ))
            else:
                suggested_val = trial.suggest_int(
                    name, params["low"], params["high"]
                )

        if suggested_val is not None:
            # Place the parameter in the correct section
            if name in train_params:
                train_params[name] = suggested_val
            elif name in model_params:
                model_params[name] = suggested_val
            else:
                cfg[name] = suggested_val
                
    return cfg


def _reconstruct_config_from_trial(
    trial: FrozenTrial, base_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reconstructs a configuration dictionary from a completed Optuna trial.
    
    This is used to recreate the best configuration after hyperparameter search.
    """
    cfg = copy.deepcopy(base_cfg)
    params = trial.params
    search_space = cfg.get("optuna_hyperparam_search_space", {})
    model_params = cfg["model_hyperparameters"]
    train_params = cfg["training_hyperparameters"]

    # Reconstruct architecture
    arch_space = search_space.get("architecture", {})
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        if "num_hidden_layers" in params:
            num_layers = params["num_hidden_layers"]
            model_params["hidden_dims"] = [
                params[f"hidden_dim_l{i+1}"] for i in range(num_layers)
            ]

    # Reconstruct other parameters
    for name, value in params.items():
        # Skip architecture parameters (already handled)
        if name.startswith("hidden_dim_l") or name == "num_hidden_layers":
            continue

        # Place parameter in correct section
        if name in train_params:
            train_params[name] = value
        elif name in model_params:
            model_params[name] = value
        else:
            # Check if it's a valid hyperparameter
            if name in search_space.get("hyperparameters", {}):
                cfg[name] = value
    
    return cfg


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_root_dir: Path,
    h5_path: Path,
    splits: Dict[str, List[int]],
    collate_fn: Callable,
) -> Optional[Dict[str, Any]]:
    """
    Executes the complete Optuna hyperparameter search process.
    
    Args:
        base_config: Base configuration dictionary
        data_root_dir: Root directory for data and outputs
        h5_path: Path to HDF5 dataset
        splits: Train/val/test split indices
        collate_fn: Collation function for DataLoader
        
    Returns:
        Best configuration found, or None if no trials succeeded
    """
    # Pre-calculate normalization stats ONCE for efficiency
    logger.info("Pre-calculating normalization statistics for the entire tuning study...")
    device = setup_device()
    normalizer = DataNormalizer(config_data=base_config)
    pre_calculated_norm_metadata = normalizer.calculate_stats(h5_path, splits['train'])
    logger.info("Normalization statistics pre-calculation complete.")

    def objective(trial: Trial) -> float:
        """
        Objective function for Optuna, called for each trial.
        
        Returns the validation loss to minimize.
        """
        # Generate trial configuration
        trial_cfg = _suggest_hyperparams(trial, base_config)
        
        # Create trial directory
        tuning_dir = data_root_dir / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial configuration
        save_json(trial_cfg, trial_dir / "run_config.json")
        
        try:
            # Create trainer with pre-calculated normalization
            trainer = ModelTrainer(
                config=trial_cfg,
                device=device,
                save_dir=trial_dir,
                h5_path=h5_path,
                splits=splits,
                collate_fn=collate_fn,
                optuna_trial=trial,
                norm_metadata=pre_calculated_norm_metadata,  # Pass pre-calculated stats
            )
            
            # Train and return best validation loss
            best_val_loss = trainer.train()
            return best_val_loss
            
        except TrialPruned:
            # Trial was pruned by Optuna
            raise
        except Exception as e:
            # Log the error and prune the trial
            logger.error(f"Trial {trial.number} failed critically: {e}", exc_info=True)
            raise TrialPruned(f"Training failed with error: {e}") from e

    # Get Optuna configuration
    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", DEFAULT_STUDY_NAME)
    
    # Create output directory
    output_folder = data_root_dir / base_config["output_paths_config"]["tuning_results_foldername"]
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Setup study storage
    storage_path = output_folder / f"{study_name}.db"
    
    # Create study
    logger.info(f"Creating/loading Optuna study: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=optuna_cfg.get("pruner_warmup_steps", PRUNER_WARMUP_STEPS),
            n_startup_trials=optuna_cfg.get("pruner_startup_trials", PRUNER_STARTUP_TRIALS)
        ),
        load_if_exists=True,
    )

    # Run optimization
    try:
        # Extract relevant arguments for optimize()
        optimize_kwargs = {}
        if "n_trials" in optuna_cfg:
            optimize_kwargs["n_trials"] = optuna_cfg["n_trials"]
        if "timeout" in optuna_cfg:
            optimize_kwargs["timeout"] = optuna_cfg["timeout"]
        if "callbacks" in optuna_cfg:
            optimize_kwargs["callbacks"] = optuna_cfg["callbacks"]
        if "catch" in optuna_cfg:
            optimize_kwargs["catch"] = optuna_cfg["catch"]
        if "show_progress_bar" in optuna_cfg:
            optimize_kwargs["show_progress_bar"] = optuna_cfg["show_progress_bar"]
        
        logger.info(f"Starting hyperparameter search with {optimize_kwargs.get('n_trials', 'unlimited')} trials...")
        study.optimize(objective, **optimize_kwargs)
        
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted by user.")
    
    # Check if we have any completed trials
    if not study.best_trial:
        logger.error("No trials completed successfully. Cannot determine best config.")
        return None

    # Log best trial info
    logger.info(f"Best trial: #{study.best_trial.number} with value: {study.best_trial.value:.6f}")
    logger.info(f"Best parameters: {study.best_trial.params}")
    
    # Reconstruct best configuration
    best_config = _reconstruct_config_from_trial(study.best_trial, base_config)
    
    # Save best configuration
    best_config_path = output_folder / "best_config.json"
    save_json(best_config, best_config_path)
    logger.info(f"Best configuration saved to {best_config_path}")
    
    # Save study summary
    summary = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }
    save_json(summary, output_folder / "study_summary.json")
    
    return best_config


__all__ = ["run_hyperparameter_search"]