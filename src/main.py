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

from dataset import ChemicalDataset, collate_fn
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

def _initialize_dataset_and_collate(config: Dict[str, Any], data_root_dir: Path) -> Optional[Tuple[ChemicalDataset, Callable]]:
    norm_folder = _get_path_from_config(config, "data_paths_config", "normalized_profiles_foldername", "dataset init")
    if not norm_folder: return None
    
    norm_path = data_root_dir / norm_folder
    if not norm_path.is_dir():
        logger.error(f"Normalized data directory '{norm_path}' not found. Please run --normalize first."); return None

    try:
        dataset = ChemicalDataset(
            data_folder=norm_path,
            species_variables=config["species_variables"],
            global_variables=config["global_variables"],
            all_variables=config["all_variables"],
            validate_profiles=config.get("validate_profiles", True)
        )
        return dataset, collate_fn
    except (KeyError, ValueError, FileNotFoundError) as e:
        logger.error(f"Dataset initialization failed: {e}", exc_info=True); return None

def _execute_model_training(
    optuna_trial: Optional[optuna.Trial], train_config: Dict[str, Any], compute_device: torch.device,
    dataset: ChemicalDataset, collate_fn: Callable, model_save_dir: Path
) -> float:
    ensure_dirs(model_save_dir)
    try:
        trainer = ModelTrainer(
            config=train_config, device=compute_device, save_dir=model_save_dir,
            dataset=dataset, collate_fn=collate_fn, optuna_trial=optuna_trial
        )
        return trainer.train()
    except Exception:
        logger.error(f"Critical error during model training in {model_save_dir}.", exc_info=True)
        raise

def _initiate_hyperparameter_tuning(
    base_config: Dict[str, Any], data_root_dir: Path, dataset: ChemicalDataset, collate_fn: Callable
) -> bool:
    """Manages hyperparameter tuning with a pre-loaded dataset."""
    logger.info("Starting hyperparameter search...")
    try:
        best_config = run_hyperparameter_search(
            base_config=base_config,
            data_dir_root=data_root_dir,
            dataset_and_collate=(dataset, collate_fn),
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
    # FIX: Add the --tune argument back
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

    seed_everything(main_config.get("random_seed", 42))
    action_successful = False
    action_name = ""

    if cli_args.normalize:
        action_name = "Normalization"
        action_successful = _normalize_data(main_config, cli_args.data_dir)
    
    # FIX: Handle --train and --tune, which both require a dataset
    elif cli_args.train or cli_args.tune:
        dataset_setup = _initialize_dataset_and_collate(main_config, cli_args.data_dir)
        if dataset_setup is None:
            logger.critical("Failed to initialize dataset. Terminating."); return 1
        dataset, collate_fn = dataset_setup

        if cli_args.train:
            action_name = "Training"
            logger.info("Starting model training with fixed configuration...")
            model_folder = _get_path_from_config(main_config, "output_paths_config", "fixed_model_foldername", action_name)
            if not model_folder: return 1
            try:
                _execute_model_training(
                    optuna_trial=None, train_config=main_config, compute_device=setup_device(),
                    dataset=dataset, collate_fn=collate_fn, model_save_dir=cli_args.data_dir / model_folder
                )
                action_successful = True
            except Exception:
                action_successful = False
        
        elif cli_args.tune:
            action_name = "Hyperparameter Tuning"
            action_successful = _initiate_hyperparameter_tuning(
                main_config, cli_args.data_dir, dataset, collate_fn
            )

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