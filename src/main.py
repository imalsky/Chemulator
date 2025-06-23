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
from typing import Any, Dict

import torch

from dataset import collate_fn
from hardware import setup_device
from hyperparams import run_hyperparameter_search
from train import ModelTrainer
from utils import (
    ensure_dirs,
    get_config_str,
    load_config,
    load_or_generate_splits,
    save_json,
    seed_everything,
    setup_logging,
)

# --- Constants ---
DEFAULT_CONFIG_PATH = Path("inputs/config.jsonc")
DEFAULT_DATA_DIR_PATH = Path("data")
DEFAULT_RANDOM_SEED = 42
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# --- Config Section and Key Names ---
DATA_PATHS_SECTION = "data_paths_config"
OUTPUT_PATHS_SECTION = "output_paths_config"
HDF5_DATASET_KEY = "hdf5_dataset_filename"
FIXED_MODEL_KEY = "fixed_model_foldername"
TUNING_RESULTS_KEY = "tuning_results_foldername"
MISC_SECTION = "miscellaneous_settings"
RANDOM_SEED_KEY = "random_seed"

# --- Global PyTorch & Warning Settings ---
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*torch.compile for Metal is an early protoype.*")
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""
    pass


def _get_h5_path(config: Dict[str, Any], data_root_dir: Path) -> Path:
    """Get and validate HDF5 dataset path."""
    h5_filename = get_config_str(config, DATA_PATHS_SECTION, HDF5_DATASET_KEY, "HDF5 data loading")
    h5_path = data_root_dir / h5_filename
    
    if not h5_path.is_file():
        raise PipelineError(f"HDF5 dataset file not found: '{h5_path}'")
    
    return h5_path


def _run_fixed_training(config: Dict[str, Any], data_root_dir: Path) -> None:
    """Runs training with a single, fixed configuration."""
    logger.info("Starting model training with fixed configuration...")
    model_folder = get_config_str(config, OUTPUT_PATHS_SECTION, FIXED_MODEL_KEY, "fixed training")
    model_save_dir = data_root_dir / model_folder
    ensure_dirs(model_save_dir)
    save_json(config, model_save_dir / "run_config.json")
    
    # Get HDF5 path
    h5_path = _get_h5_path(config, data_root_dir)
    
    # Load or generate splits
    splits, splits_path = load_or_generate_splits(config, data_root_dir, h5_path)
    
    # Save splits info to model directory
    save_json({"splits_file": str(splits_path)}, model_save_dir / "splits_info.json")
    
    try:
        trainer = ModelTrainer(
            config=config, 
            device=setup_device(), 
            save_dir=model_save_dir,
            h5_path=h5_path,
            splits=splits,
            collate_fn=collate_fn
        )
        trainer.train()
        logger.info("Fixed configuration training completed successfully.")
    except Exception as e:
        raise PipelineError(f"Fixed model training failed: {e}") from e


def _run_hyperparameter_tuning(config: Dict[str, Any], data_root_dir: Path) -> None:
    """Manages the Optuna hyperparameter tuning process."""
    logger.info("Starting hyperparameter search...")
    
    # Get HDF5 path
    h5_path = _get_h5_path(config, data_root_dir)
    
    # Load or generate splits
    splits, splits_path = load_or_generate_splits(config, data_root_dir, h5_path)
    
    # Save splits info to tuning directory
    tuning_folder = get_config_str(config, OUTPUT_PATHS_SECTION, TUNING_RESULTS_KEY, "tuning results")
    tuning_dir = data_root_dir / tuning_folder
    ensure_dirs(tuning_dir)
    save_json({"splits_file": str(splits_path)}, tuning_dir / "splits_info.json")

    try:
        best_config = run_hyperparameter_search(
            base_config=config,
            data_root_dir=data_root_dir,
            h5_path=h5_path,
            splits=splits,
            collate_fn=collate_fn,
        )
        if best_config is None:
            raise PipelineError("Hyperparameter search returned no valid configuration.")
        logger.info("Hyperparameter tuning completed successfully.")
    except Exception as e:
        raise PipelineError(f"Hyperparameter tuning failed: {e}") from e


def _parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chemical reaction predictor training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, 
        help="Path to the main configuration file."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR_PATH, 
        help="Path to the root data directory."
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--train", action="store_true", 
        help="Train a model with a fixed configuration."
    )
    action_group.add_argument(
        "--tune", action="store_true", 
        help="Run Optuna hyperparameter search."
    )
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
        seed = config.get(MISC_SECTION, {}).get(RANDOM_SEED_KEY, DEFAULT_RANDOM_SEED)
        seed_everything(seed)
        
        action_name = "Training" if args.train else "Hyperparameter Tuning"
        if args.train:
            _run_fixed_training(config, args.data_dir)
        elif args.tune:
            _run_hyperparameter_tuning(config, args.data_dir)
        
        logger.info(f"{action_name} process completed successfully.")
        return EXIT_SUCCESS
        
    except (PipelineError, FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Pipeline error: {e}", exc_info=False)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user (Ctrl+C).")
        return 130
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())