#!/usr/bin/env python3
"""
main.py - Simplified main entry point for training pipeline

This module serves as the main entry point for the chemical kinetics
prediction model training pipeline. It handles configuration loading,
environment setup, and orchestrates the training process.
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
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# --- Config Section and Key Names ---
DATA_PATHS_SECTION = "data_paths_config"
OUTPUT_PATHS_SECTION = "output_paths_config"
HDF5_DATASET_KEY = "hdf5_dataset_filename"
FIXED_MODEL_KEY = "fixed_model_foldername"
MISC_SECTION = "miscellaneous_settings"
RANDOM_SEED_KEY = "random_seed"
NUM_CONSTANTS_SECTION = "numerical_constants"

# --- Global PyTorch & Warning Settings ---
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*torch.compile for Metal is an early protoype.*")
warnings.filterwarnings("ignore", ".*Torch was not compiled with flash attention.*")
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""
    pass


def _get_h5_path(config: Dict[str, Any], data_root_dir: Path) -> Path:
    """
    Get and validate HDF5 dataset path.
    
    Args:
        config: Configuration dictionary
        data_root_dir: Root directory for data files
        
    Returns:
        Path to HDF5 file
        
    Raises:
        PipelineError: If HDF5 file not found
    """
    h5_filename = get_config_str(config, DATA_PATHS_SECTION, HDF5_DATASET_KEY, "HDF5 data loading")
    h5_path = data_root_dir / h5_filename
    
    if not h5_path.is_file():
        raise PipelineError(f"HDF5 dataset file not found: '{h5_path.resolve()}'")
    
    return h5_path


def _run_training(config: Dict[str, Any], data_root_dir: Path) -> None:
    """
    Runs training with the given configuration.
    
    Args:
        config: Configuration dictionary
        data_root_dir: Root directory for data files
        
    Raises:
        PipelineError: If training fails
    """
    logger.info("Starting model training...")
    
    # Get HDF5 path
    h5_path = _get_h5_path(config, data_root_dir)
    
    model_folder = get_config_str(config, OUTPUT_PATHS_SECTION, FIXED_MODEL_KEY, "model training")
    model_save_dir = data_root_dir / model_folder
    # Directory will be created by save_json if needed
    save_json(config, model_save_dir / "run_config.json")
    
    # Load or generate splits - now passing model_save_dir
    splits, splits_path = load_or_generate_splits(config, data_root_dir, h5_path, model_save_dir)
    
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
        logger.info("Training completed successfully.")
    except Exception as e:
        raise PipelineError(f"Model training failed: {e}") from e


def _parse_command_line_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Chemical reaction predictor training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR_PATH, 
        help="Path to the root data directory."
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the pipeline.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = _parse_command_line_arguments()
    
    # Ensure root data directory exists before setting up logging
    ensure_dirs(args.data_dir)
    log_file = args.data_dir / f"run_{datetime.now().strftime(LOG_TIMESTAMP_FORMAT)}.log"
    setup_logging(log_file=log_file)
    
    try:
        logger.info("Pipeline started")
        logger.info(f"Using config: {args.config.resolve()}")
        config = load_config(args.config)
        
        # Get seed from config with proper fallback chain
        num_constants = config.get(NUM_CONSTANTS_SECTION, {})
        default_seed = num_constants.get("default_seed", 42)
        seed = config.get(MISC_SECTION, {}).get(RANDOM_SEED_KEY, default_seed)
        seed_everything(seed)
        
        _run_training(config, args.data_dir)
        
        logger.info("Pipeline process completed successfully.")
        return EXIT_SUCCESS
        
    except PipelineError as e:
        # Expected pipeline errors (e.g., file not found, bad config)
        logger.error(f"A pipeline error occurred: {e}", exc_info=False)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user (Ctrl+C).")
        return 130 # Standard exit code for Ctrl+C
    except Exception as e:
        # Unexpected errors
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())