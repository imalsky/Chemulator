#!/usr/bin/env python3
"""
main.py - Main entry point and orchestrator for the training pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
import torch

from dataset import collate_fn
from hardware import setup_device
from hyperparams import run_hyperparameter_search
from train import ModelTrainer
from utils import (
    ensure_dirs,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
)

# --- Constants ---
DEFAULT_CONFIG_PATH = Path("inputs/config.jsonc")
DEFAULT_DATA_DIR_PATH = Path("data")
DEFAULT_RANDOM_SEED = 42
MATMUL_PRECISION = "high"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# --- Config Section and Key Names ---
DATA_PATHS_SECTION = "data_paths_config"
OUTPUT_PATHS_SECTION = "output_paths_config"
RAW_PROFILES_KEY = "raw_profiles_foldername"
FIXED_MODEL_KEY = "fixed_model_foldername"
TUNING_RESULTS_KEY = "tuning_results_foldername"
RANDOM_SEED_KEY = "random_seed"

# --- Global PyTorch & Warning Settings ---
torch.set_float32_matmul_precision(MATMUL_PRECISION)
warnings.filterwarnings("ignore", ".*torch.compile for Metal is an early protoype.*")

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""
    pass


def _get_config_value(config: Dict[str, Any], section: str, key: str, op_desc: str) -> str:
    """Safely extracts a string value from the configuration with descriptive errors."""
    if section not in config or not isinstance(config[section], dict):
        raise PipelineError(f"Config section '{section}' missing or invalid for {op_desc}.")
    val = config[section].get(key)
    if not isinstance(val, str) or not val.strip():
        raise PipelineError(f"Config key '{key}' in '{section}' missing or empty for {op_desc}.")
    return val.strip()


def _validate_directory_exists(directory: Path, description: str) -> None:
    """Validates that a directory exists, raising a PipelineError if not."""
    if not directory.is_dir():
        raise PipelineError(f"{description} directory not found: '{directory.resolve()}'")


def _execute_model_training(
    optuna_trial: Optional[optuna.Trial],
    train_config: Dict[str, Any],
    compute_device: torch.device,
    raw_data_dir: Path,
    model_save_dir: Path,
) -> float:
    """
    Core function to execute a single model training run. It orchestrates the
    ModelTrainer, which now handles the entire process from raw data to a trained model.

    Args:
        optuna_trial: An Optuna trial object if part of a hyperparameter search.
        train_config: The configuration for this specific training run.
        compute_device: The torch.device to train on.
        raw_data_dir: Path to the directory with raw, unnormalized data profiles.
        model_save_dir: Path to the directory where model artifacts will be saved.

    Returns:
        The best validation loss achieved during training.
    """
    ensure_dirs(model_save_dir)
    save_json(train_config, model_save_dir / "run_config.json")
    logger.info(f"Saved run configuration to {model_save_dir / 'run_config.json'}")

    try:
        trainer = ModelTrainer(
            config=train_config,
            device=compute_device,
            save_dir=model_save_dir,
            data_dir=raw_data_dir,  
            collate_fn=collate_fn,
            optuna_trial=optuna_trial,
        )
        best_loss = trainer.train()
        logger.info(f"Training completed with best validation loss: {best_loss:.6e}")
        return best_loss
    except Exception as e:
        # Catch errors from the trainer (e.g., data normalization issues) and wrap them.
        raise PipelineError(f"Model training failed in '{model_save_dir}': {e}") from e


def _run_fixed_training(config: Dict[str, Any], data_root_dir: Path) -> None:
    """Runs training with a single, fixed configuration."""
    logger.info("Starting model training with fixed configuration...")
    raw_folder = _get_config_value(config, DATA_PATHS_SECTION, RAW_PROFILES_KEY, "fixed training")
    model_folder = _get_config_value(config, OUTPUT_PATHS_SECTION, FIXED_MODEL_KEY, "fixed training")
    
    raw_data_dir = data_root_dir / raw_folder
    model_save_dir = data_root_dir / model_folder
    
    _validate_directory_exists(raw_data_dir, "Raw data")
    
    compute_device = setup_device()
    _execute_model_training(
        optuna_trial=None,
        train_config=config,
        compute_device=compute_device,
        raw_data_dir=raw_data_dir,
        model_save_dir=model_save_dir,
    )
    logger.info("Fixed configuration training completed successfully.")


def _run_hyperparameter_tuning(config: Dict[str, Any], data_root_dir: Path) -> None:
    """Manages the Optuna hyperparameter tuning process."""
    logger.info("Starting hyperparameter search...")
    raw_folder = _get_config_value(config, DATA_PATHS_SECTION, RAW_PROFILES_KEY, "hyperparameter tuning")
    raw_data_dir = data_root_dir / raw_folder
    _validate_directory_exists(raw_data_dir, "Raw data")

    try:
        best_config = run_hyperparameter_search(
            base_config=config,
            data_dir_root=data_root_dir,
            norm_data_dir=raw_data_dir,  # This argument now points to RAW data
            train_model_func=_execute_model_training,
            setup_device_func=setup_device,
            save_config_func=save_json,
        )
        if best_config is None:
            raise PipelineError("Hyperparameter search returned no valid configuration.")
        logger.info("Hyperparameter tuning completed successfully.")
    except Exception as e:
        raise PipelineError(f"Hyperparameter tuning failed: {e}") from e


def _parse_command_line_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Chemical reaction predictor training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the main configuration file.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR_PATH, help="Path to the root data directory.")
    
    # The --normalize action is removed, as it's now an internal part of training.
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--train", action="store_true", help="Train a model with a fixed configuration from raw data.")
    action_group.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search from raw data.")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the pipeline."""
    try:
        args = _parse_command_line_arguments()
        
        ensure_dirs(args.data_dir)
        log_file = args.data_dir / f"run_{datetime.now().strftime(LOG_TIMESTAMP_FORMAT)}.log"
        setup_logging(log_file=log_file)
        
        logger.info(f"Pipeline started with config: {args.config.resolve()}")
        config = load_config(args.config)
        
        seed = config.get(RANDOM_SEED_KEY, DEFAULT_RANDOM_SEED)
        seed_everything(seed)
        
        action_name = "Unknown"
        if args.train:
            action_name = "Training"
            _run_fixed_training(config, args.data_dir)
        elif args.tune:
            action_name = "Hyperparameter Tuning"
            _run_hyperparameter_tuning(config, args.data_dir)
        
        logger.info(f"{action_name} process completed successfully.")
        return EXIT_SUCCESS
        
    except (PipelineError, FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Pipeline error: {e}", exc_info=False)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user (Ctrl+C).")
        return EXIT_INTERRUPTED
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())