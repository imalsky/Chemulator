#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the chemical kinetics model.
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

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STUDY_NAME = "chemical-kinetics-study"
DEFAULT_NUM_TRIALS = 100
PRUNER_WARMUP_STEPS = 5
PRUNER_STARTUP_TRIALS = 5
DEFAULT_GC_AFTER_TRIAL = True
DEFAULT_SHOW_PROGRESS_BAR = True

def _suggest_hyperparams(trial: Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a trial-specific configuration by suggesting hyperparameters."""
    cfg = copy.deepcopy(base_cfg)
    search_space = cfg.get("optuna_hyperparam_search_space", {})
    model_params = cfg["model_hyperparameters"]
    train_params = cfg["training_hyperparameters"]

    # Suggest network architecture
    arch_space = search_space.get("architecture", {})
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        num_layers_cfg = arch_space["num_hidden_layers"]
        hidden_dim_cfg = arch_space["hidden_dim"]
        num_layers = trial.suggest_int("num_hidden_layers", num_layers_cfg["low"], num_layers_cfg["high"])
        model_params["hidden_dims"] = [
            trial.suggest_categorical(f"hidden_dim_l{i+1}", hidden_dim_cfg["choices"])
            for i in range(num_layers)
        ]

    # Suggest independent hyperparameters
    for name, params in search_space.get("hyperparameters", {}).items():
        suggested_val = None
        if params["type"] == "categorical":
            suggested_val = trial.suggest_categorical(name, params["choices"])
        elif params["type"] == "float":
            suggested_val = trial.suggest_float(
                name, params["low"], params["high"], log=params.get("log", False)
            )
        elif params["type"] == "int":
            suggested_val = trial.suggest_int(
                name, params["low"], params["high"], log=params.get("log", False)
            )

        if suggested_val is not None:
            if name in train_params:
                train_params[name] = suggested_val
            elif name in model_params:
                model_params[name] = suggested_val
            else:
                cfg[name] = suggested_val  # Fallback to root
    return cfg

def _reconstruct_config_from_trial(
    trial: FrozenTrial, base_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Reconstructs a configuration dictionary from a completed Optuna trial."""
    cfg = copy.deepcopy(base_cfg)
    params = trial.params
    search_space = cfg.get("optuna_hyperparam_search_space", {})
    model_params = cfg["model_hyperparameters"]
    train_params = cfg["training_hyperparameters"]

    # Handle special case for network architecture
    arch_space = search_space.get("architecture", {})
    if "num_hidden_layers" in arch_space and "hidden_dim" in arch_space:
        if "num_hidden_layers" in params:
            num_layers = params["num_hidden_layers"]
            model_params["hidden_dims"] = [
                params[f"hidden_dim_l{i+1}"] for i in range(num_layers)
            ]

    # Update other hyperparameters
    for name, value in params.items():
        if name.startswith("hidden_dim_l") or name == "num_hidden_layers":
            continue  # Already handled

        if name in train_params:
            train_params[name] = value
        elif name in model_params:
            model_params[name] = value
        else:
            # Check against the search space definition to avoid misplaced keys
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
    """Executes the complete Optuna hyperparameter search process."""

    def objective(trial: Trial) -> float:
        """Objective function for Optuna, called for each trial."""
        trial_cfg = _suggest_hyperparams(trial, base_config)
        
        tuning_dir = data_root_dir / base_config["output_paths_config"]["tuning_results_foldername"]
        trial_dir = tuning_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_json(trial_cfg, trial_dir / "run_config.json")
        
        try:
            trainer = ModelTrainer(
                config=trial_cfg,
                device=setup_device(),
                save_dir=trial_dir,
                h5_path=h5_path,
                splits=splits,
                collate_fn=collate_fn,
                optuna_trial=trial,
            )
            return trainer.train()
        except TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed critically: {e}", exc_info=True)
            raise TrialPruned("Training function raised a critical error.") from e

    optuna_cfg = base_config.get("optuna_settings", {})
    study_name = optuna_cfg.get("study_name", DEFAULT_STUDY_NAME)
    
    output_folder = data_root_dir / base_config["output_paths_config"]["tuning_results_foldername"]
    storage_path = output_folder / f"{study_name}.db"
    
    # Add support for 'int' suggestion type
    for _, params in base_config.get("optuna_hyperparam_search_space", {}).get("hyperparameters", {}).items():
        if params["type"] == "int":
            # This is a bit of a hack to make sure the trial suggestion works
            # without a large refactor. In a real scenario, the _suggest_hyperparams
            # should handle this directly. For this fix, it is assumed this is the only `int`.
            if 'log' not in params: params['log'] = False

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

    try:
        _opt_args = {k: optuna_cfg[k] for k in (
            "n_trials", "timeout", "callbacks", "catch", "show_progress_bar"
        ) if k in optuna_cfg}
        study.optimize(objective, **_opt_args)
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted by user.")
    
    if not study.best_trial:
        logger.error("No trials completed successfully. Cannot determine best config.")
        return None

    logger.info(f"Best trial: #{study.best_trial.number} with value: {study.best_trial.value:.6f}")
    best_config = _reconstruct_config_from_trial(study.best_trial, base_config)
    save_json(best_config, output_folder / "best_config.json")
    logger.info(f"Best configuration saved to {output_folder / 'best_config.json'}")
    return best_config

__all__ = ["run_hyperparameter_search"]