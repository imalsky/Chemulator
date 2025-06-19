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
from normalizer import DataNormalizer
from train import ModelTrainer
from utils import (
    ensure_dirs,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
)

# Constants
DEFAULT_CONFIG_PATH = Path("inputs/model_input_params.jsonc")
DEFAULT_DATA_DIR_PATH = Path("data")
DEFAULT_RANDOM_SEED = 42
DEFAULT_EPSILON = 1e-9
MATMUL_PRECISION = "high"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Config section and key names
DATA_PATHS_SECTION = "data_paths_config"
OUTPUT_PATHS_SECTION = "output_paths_config"
NORMALIZATION_SECTION = "normalization"
RAW_PROFILES_KEY = "raw_profiles_foldername"
NORMALIZED_PROFILES_KEY = "normalized_profiles_foldername"
FIXED_MODEL_KEY = "fixed_model_foldername"
EPSILON_KEY = "epsilon"
RANDOM_SEED_KEY = "random_seed"

# Configure PyTorch and warnings
torch.set_float32_matmul_precision(MATMUL_PRECISION)
warnings.filterwarnings("ignore", ".*non-base types in `__init__`.*")
warnings.filterwarnings("ignore", message=".*torch.compile for Metal is an early protoype.*")

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""
    pass


def _get_config_path(
    config: Dict[str, Any], 
    section: str, 
    key: str, 
    operation: str
) -> str:
    """
    Safely extracts a path from configuration with descriptive error messages.
    
    Args:
        config: Configuration dictionary
        section: Configuration section name
        key: Key within the section
        operation: Description of the operation (for error messages)
        
    Returns:
        Path string from configuration
        
    Raises:
        PipelineError: If path is missing or invalid
    """
    if section not in config or not isinstance(config[section], dict):
        raise PipelineError(
            f"Configuration section '{section}' is missing or invalid. "
            f"Required for {operation}."
        )
    
    path_value = config[section].get(key)
    if not isinstance(path_value, str) or not path_value.strip():
        raise PipelineError(
            f"Configuration key '{key}' in section '{section}' is missing "
            f"or empty. Required for {operation}."
        )
    
    return path_value.strip()


def _validate_directory_exists(directory: Path, description: str) -> None:
    """
    Validates that a directory exists.
    
    Args:
        directory: Path to validate
        description: Description for error messages
        
    Raises:
        PipelineError: If directory doesn't exist
    """
    if not directory.is_dir():
        raise PipelineError(
            f"{description} directory '{directory}' not found."
        )


def _normalize_data(config: Dict[str, Any], data_root_dir: Path) -> None:
    """
    Performs data normalization based on configuration.
    
    Args:
        config: Main configuration dictionary
        data_root_dir: Root directory for data files
        
    Raises:
        PipelineError: If normalization fails
    """
    logger.info("Starting data normalization...")
    
    # Get paths from config
    raw_folder = _get_config_path(
        config, DATA_PATHS_SECTION, RAW_PROFILES_KEY, "normalization"
    )
    norm_folder = _get_config_path(
        config, DATA_PATHS_SECTION, NORMALIZED_PROFILES_KEY, "normalization"
    )
    
    # Set up directories
    raw_dir = data_root_dir / raw_folder
    norm_dir = data_root_dir / norm_folder
    
    _validate_directory_exists(raw_dir, "Raw profiles")
    ensure_dirs(norm_dir)
    
    # Extract normalization parameters
    norm_config = config.get(NORMALIZATION_SECTION, {})
    epsilon = norm_config.get(EPSILON_KEY, DEFAULT_EPSILON)
    
    try:
        # Initialize normalizer
        normalizer = DataNormalizer(
            input_folder=raw_dir,
            output_folder=norm_dir,
            config_data=config,
            epsilon=epsilon
        )
        
        # Calculate statistics and normalize
        logger.info(f"Calculating normalization statistics from {raw_dir}")
        stats = normalizer.calculate_global_stats()
        
        if not stats:
            raise PipelineError("Failed to calculate normalization statistics")
        
        logger.info(f"Processing profiles to {norm_dir}")
        normalizer.process_profiles(stats)
        
        logger.info("Data normalization completed successfully")
        
    except Exception as e:
        raise PipelineError(f"Normalization failed: {e}") from e


def _execute_model_training(
    optuna_trial: Optional[optuna.Trial], 
    train_config: Dict[str, Any], 
    compute_device: torch.device,
    data_dir: Path,
    model_save_dir: Path
) -> float:
    """
    Executes model training with the given configuration.
    
    Args:
        optuna_trial: Optional Optuna trial for hyperparameter optimization
        train_config: Training configuration
        compute_device: Device to train on
        data_dir: Directory containing normalized data
        model_save_dir: Directory to save model artifacts
        
    Returns:
        Best validation loss achieved
        
    Raises:
        PipelineError: If training fails
    """
    ensure_dirs(model_save_dir)
    
    # Save run configuration
    config_path = model_save_dir / "run_config.json"
    if not save_json(train_config, config_path):
        raise PipelineError(f"Failed to save run configuration to {config_path}")
    
    logger.info(f"Saved run configuration to {config_path}")
    
    try:
        trainer = ModelTrainer(
            config=train_config, 
            device=compute_device, 
            save_dir=model_save_dir,
            data_dir=data_dir,
            collate_fn=collate_fn, 
            optuna_trial=optuna_trial
        )
        
        best_loss = trainer.train()
        logger.info(f"Training completed with best validation loss: {best_loss:.6e}")
        return best_loss
        
    except Exception as e:
        raise PipelineError(
            f"Model training failed in {model_save_dir}: {e}"
        ) from e


def _run_hyperparameter_tuning(config: Dict[str, Any], data_root_dir: Path) -> None:
    """
    Manages hyperparameter tuning process.
    
    Args:
        config: Main configuration dictionary
        data_root_dir: Root directory for data files
        
    Raises:
        PipelineError: If hyperparameter tuning fails
    """
    logger.info("Starting hyperparameter search...")
    
    # Get normalized data directory
    norm_folder = _get_config_path(
        config, DATA_PATHS_SECTION, NORMALIZED_PROFILES_KEY, "hyperparameter tuning"
    )
    norm_data_dir = data_root_dir / norm_folder
    _validate_directory_exists(norm_data_dir, "Normalized data")

    try:
        best_config = run_hyperparameter_search(
            base_config=config,
            data_dir_root=data_root_dir,
            norm_data_dir=norm_data_dir,
            train_model_func=_execute_model_training,
            setup_device_func=setup_device,
            save_config_func=save_json
        )
        
        if best_config is None:
            raise PipelineError("Hyperparameter search returned no valid configuration")
        
        logger.info("Hyperparameter tuning completed successfully")
        
    except Exception as e:
        raise PipelineError(f"Hyperparameter tuning failed: {e}") from e


def _run_fixed_training(config: Dict[str, Any], data_root_dir: Path) -> None:
    """
    Runs training with a fixed configuration.
    
    Args:
        config: Main configuration dictionary
        data_root_dir: Root directory for data files
        
    Raises:
        PipelineError: If training fails
    """
    logger.info("Starting model training with fixed configuration...")
    
    # Get directories
    norm_folder = _get_config_path(
        config, DATA_PATHS_SECTION, NORMALIZED_PROFILES_KEY, "training"
    )
    model_folder = _get_config_path(
        config, OUTPUT_PATHS_SECTION, FIXED_MODEL_KEY, "training"
    )
    
    norm_data_dir = data_root_dir / norm_folder
    model_save_dir = data_root_dir / model_folder
    
    _validate_directory_exists(norm_data_dir, "Normalized data")
    
    # Set up device and execute training
    compute_device = setup_device()
    _execute_model_training(
        optuna_trial=None,
        train_config=config,
        compute_device=compute_device,
        data_dir=norm_data_dir,
        model_save_dir=model_save_dir
    )
    
    logger.info("Fixed configuration training completed successfully")


def _validate_config_structure(config: Dict[str, Any]) -> None:
    """
    Validates basic configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        PipelineError: If configuration structure is invalid
    """
    if OUTPUT_PATHS_SECTION in config:
        logger.info("Found output path configuration section")
    else:
        logger.warning(
            f"No '{OUTPUT_PATHS_SECTION}' section found in config. "
            f"Using default paths."
        )


def _setup_logging_and_directories(data_dir: Path) -> None:
    """
    Sets up logging and ensures required directories exist.
    
    Args:
        data_dir: Root data directory
    """
    ensure_dirs(data_dir)
    
    timestamp = datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
    log_file = data_dir / f"run_{timestamp}.log"
    
    setup_logging(log_file=log_file)


def _parse_command_line_arguments() -> argparse.Namespace:
    """
    Parses and validates command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Chemical reaction predictor training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=Path, 
        default=DEFAULT_CONFIG_PATH,
        help="Path to the main configuration file"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=DEFAULT_DATA_DIR_PATH,
        help="Path to the root data directory"
    )
    
    # Mutually exclusive action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--normalize", 
        action="store_true",
        help="Calculate statistics and normalize all profiles"
    )
    action_group.add_argument(
        "--train", 
        action="store_true",
        help="Train a model with fixed configuration"
    )
    action_group.add_argument(
        "--tune", 
        action="store_true",
        help="Run Optuna hyperparameter search"
    )
    
    return parser.parse_args()


def _execute_pipeline_action(
    args: argparse.Namespace, 
    config: Dict[str, Any]
) -> str:
    """
    Executes the requested pipeline action.
    
    Args:
        args: Parsed command line arguments
        config: Main configuration dictionary
        
    Returns:
        Name of the executed action
        
    Raises:
        PipelineError: If the action fails
    """
    if args.normalize:
        _normalize_data(config, args.data_dir)
        return "Normalization"
    
    elif args.train:
        _run_fixed_training(config, args.data_dir)
        return "Training"
    
    elif args.tune:
        _run_hyperparameter_tuning(config, args.data_dir)
        return "Hyperparameter Tuning"
    
    else:
        raise PipelineError("No valid action specified")


def main() -> int:
    """
    Main entry point for the pipeline.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command line arguments
        args = _parse_command_line_arguments()
        
        # Set up logging and directories
        _setup_logging_and_directories(args.data_dir)
        
        logger.info(f"Pipeline started with config: {args.config.resolve()}")
        
        # Load and validate configuration
        config = load_config(args.config)
        if config is None:
            logger.error("Failed to load configuration file")
            return EXIT_FAILURE
        
        _validate_config_structure(config)
        
        # Set random seed for reproducibility
        random_seed = config.get(RANDOM_SEED_KEY, DEFAULT_RANDOM_SEED)
        seed_everything(random_seed)
        
        # Execute the requested action
        action_name = _execute_pipeline_action(args, config)
        
        logger.info(f"{action_name} process completed successfully")
        return EXIT_SUCCESS
        
    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return EXIT_FAILURE
    
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user (Ctrl+C)")
        return EXIT_INTERRUPTED
    
    except Exception as e:
        logger.critical(
            f"Unhandled exception in pipeline: {e}", 
            exc_info=True
        )
        return EXIT_FAILURE


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if e.code != EXIT_SUCCESS:
            logger.info(f"Pipeline exited with code {e.code}")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(EXIT_FAILURE)