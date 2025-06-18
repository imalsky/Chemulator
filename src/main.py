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
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import optuna

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

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", ".*non-base types in `__init__`.*")

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("inputs/model_input_params.jsonc")
DEFAULT_DATA_DIR_PATH = Path("data")

def _get_path_from_config(config: Dict[str, Any], group: str, key: str, action: str) -> Optional[str]:
    if group not in config or not isinstance(config[group], dict):
        logger.error(f"Config error: '{group}' section is missing. Required for {action}."); return None
    path = config[group].get(key)
    if not isinstance(path, str) or not path:
        logger.error(f"Config error: '{key}' in '{group}' is missing. Required for {action}."); return None
    return path

def _normalize_data(config: Dict[str, Any], data_root_dir: Path) -> bool:
    raw_folder = _get_path_from_config(config, "data_paths_config", "raw_profiles_foldername", "normalization")
    norm_folder = _get_path_from_config(config, "data_paths_config", "normalized_profiles_foldername", "normalization")
    if not raw_folder or not norm_folder: return False
    
    raw_dir, norm_dir = data_root_dir / raw_folder, data_root_dir / norm_folder
    if not raw_dir.is_dir():
        logger.error(f"Raw profiles directory '{raw_dir}' not found."); return False

    ensure_dirs(norm_dir)
    try:
        normalizer = DataNormalizer(
            input_folder=raw_dir, output_folder=norm_dir, config_data=config,
            epsilon=config.get("normalization", {}).get("epsilon", 1e-9)
        )
        stats = normalizer.calculate_global_stats()
        if not stats: logger.error("Normalization statistics calculation failed."); return False
        normalizer.process_profiles(stats)
    except Exception as e:
        logger.error(f"An unexpected error occurred during normalization: {e}", exc_info=True); return False
    return True

def _execute_model_training(
    optuna_trial: Optional[optuna.Trial], 
    train_config: Dict[str, Any], 
    compute_device: torch.device,
    data_dir: Path,
    model_save_dir: Path
) -> float:
    ensure_dirs(model_save_dir)
    # --- ADDED: Save the configuration used for this specific run ---
    save_json(train_config, model_save_dir / "run_config.json")
    logger.info(f"Saved run configuration to {model_save_dir / 'run_config.json'}")
    # --- END ADDITION ---
    try:
        # The trainer now handles its own dataset creation internally.
        trainer = ModelTrainer(
            config=train_config, 
            device=compute_device, 
            save_dir=model_save_dir,
            data_dir=data_dir,
            collate_fn=collate_fn, 
            optuna_trial=optuna_trial
        )
        return trainer.train()
    except Exception:
        logger.error(f"Critical error during model training in {model_save_dir}.", exc_info=True)
        raise

def _initiate_hyperparameter_tuning(
    base_config: Dict[str, Any], data_root_dir: Path
) -> bool:
    """Manages hyperparameter tuning."""
    logger.info("Starting hyperparameter search...")
    # The normalization folder is where the trainer will look for data.
    norm_folder = _get_path_from_config(base_config, "data_paths_config", "normalized_profiles_foldername", "tuning")
    if not norm_folder: return False
    norm_data_dir = data_root_dir / norm_folder
    if not norm_data_dir.is_dir():
        logger.error(f"Normalized data directory '{norm_data_dir}' not found. Please run --normalize first."); return False

    try:
        best_config = run_hyperparameter_search(
            base_config=base_config,
            data_dir_root=data_root_dir,
            norm_data_dir=norm_data_dir,
            train_model_func=_execute_model_training,
            setup_device_func=setup_device,
            save_config_func=save_json
        )
        return best_config is not None
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed unexpectedly: {e}", exc_info=True)
        return False

def _parse_command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State-Evolution Predictor pipeline.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the main configuration file.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR_PATH, help="Path to the root data directory.")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--normalize", action="store_true", help="Calculate stats and normalize all profiles.")
    action_group.add_argument("--train", action="store_true", help="Train a model with a fixed config.")
    action_group.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search.")
    return parser.parse_args()

def main() -> int:
    cli_args = _parse_command_line_args()
    log_file = cli_args.data_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ensure_dirs(cli_args.data_dir)
    setup_logging(log_file=log_file)

    logger.info("Pipeline started with config: %s", cli_args.config.resolve())
    main_config = load_config(cli_args.config)
    if not main_config: return 1
    
    # --- ADDED: Update output paths from the newly provided config ---
    if "output_paths_config" in main_config:
        logger.info("Applying output path configuration from file.")
    else:
        logger.warning("No 'output_paths_config' found in config file. Using defaults.")
    # --- END ADDITION ---

    seed_everything(main_config.get("random_seed", 42))
    action_successful = False
    action_name = ""

    if cli_args.normalize:
        action_name = "Normalization"
        action_successful = _normalize_data(main_config, cli_args.data_dir)
    
    elif cli_args.train or cli_args.tune:
        # Check for normalized data before starting train or tune.
        norm_folder = _get_path_from_config(main_config, "data_paths_config", "normalized_profiles_foldername", "train/tune")
        if not norm_folder: return 1
        norm_data_dir = cli_args.data_dir / norm_folder
        if not norm_data_dir.is_dir():
            logger.error(f"Normalized data directory '{norm_data_dir}' not found. Please run --normalize first."); return 1

        if cli_args.train:
            action_name = "Training"
            logger.info("Starting model training with fixed configuration...")
            model_folder = _get_path_from_config(main_config, "output_paths_config", "fixed_model_foldername", action_name)
            if not model_folder: return 1
            try:
                _execute_model_training(
                    optuna_trial=None, 
                    train_config=main_config, 
                    compute_device=setup_device(),
                    data_dir=norm_data_dir,
                    model_save_dir=cli_args.data_dir / model_folder
                )
                action_successful = True
            except Exception:
                action_successful = False
        
        elif cli_args.tune:
            action_name = "Hyperparameter Tuning"
            # The tuning function now manages its own data loading via the trainer
            action_successful = _initiate_hyperparameter_tuning(main_config, cli_args.data_dir)

    if action_successful:
        logger.info(f"{action_name} process finished successfully."); return 0
    else:
        logger.error(f"{action_name} process failed."); return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if e.code != 0:
            logger.info(f"Pipeline exited with error code {e.code}.")
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main pipeline: {e}", exc_info=True)
        sys.exit(1)