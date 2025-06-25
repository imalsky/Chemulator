#!/usr/bin/env python3
"""
utils.py - Optimized helper functions without atom parsing.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import sys
import h5py

try:
    import json5 as _json_backend
    _HAS_JSON5 = True
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
DEFAULT_SEED = 42
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """Sets up logging to console and an optional file, avoiding duplicate handlers."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding=UTF8_ENCODING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to console and file: {log_path.resolve()}")
        except Exception as e:
            logger.error(f"File logging setup failed for {log_file}: {e}. Continuing with console only.")
    else:
        logger.info("Logging to console only.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a configuration file (supports JSON with comments via JSON5)."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        config_text = config_path.read_text(encoding=UTF8_SIG_ENCODING)
        config_dict = _json_backend.loads(config_text)
        validate_config(config_dict)
        backend = "JSON5" if _HAS_JSON5 else "JSON"
        logger.info(f"Successfully loaded and validated {backend} config from {config_path}.")
        return config_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate config {config_path}: {e}") from e


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the nested structure and key values of the configuration dictionary."""
    data_spec = config.get("data_specification")
    if not isinstance(data_spec, dict):
        raise ValueError("Config section 'data_specification' is missing or not a dictionary.")
    if not isinstance(data_spec.get("species_variables"), list) or not data_spec.get("species_variables"):
        raise ValueError("Config key 'species_variables' in 'data_specification' must be a non-empty list.")
    if not isinstance(data_spec.get("all_variables"), list) or not data_spec.get("all_variables"):
        raise ValueError("Config key 'all_variables' in 'data_specification' must be a non-empty list.")
    model_params = config.get("model_hyperparameters")
    if not isinstance(model_params, dict):
        raise ValueError("Config section 'model_hyperparameters' is missing or not a dictionary.")
    if not isinstance(model_params.get("hidden_dims"), list) or not model_params.get("hidden_dims"):
        raise ValueError("Config key 'hidden_dims' in 'model_hyperparameters' must be a non-empty list.")
    dropout = model_params.get("dropout")
    if dropout is not None and not (0.0 <= dropout < 1.0):
        raise ValueError("'dropout' in 'model_hyperparameters' must be a float in the range [0.0, 1.0).")


def ensure_dirs(*paths: Union[str, Path]) -> bool:
    """Creates one or more directories if they do not already exist."""
    try:
        for path in paths: 
            Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for handling numpy, torch, Path, and set objects."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """Saves a dictionary to a JSON file with pretty printing and robust serialization."""
    try:
        json_path = Path(path)
        if not ensure_dirs(json_path.parent):
            return False
        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            json.dump(data, f, indent=2, default=_json_serializer, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}", exc_info=True)
        return False


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to {seed}.")


def generate_dataset_splits(
    h5_path: Path, 
    val_frac: float = 0.15, 
    test_frac: float = 0.15,
    random_seed: int = DEFAULT_SEED
) -> Dict[str, List[int]]:
    """
    Generate train/validation/test splits from HDF5 dataset.
    
    Args:
        h5_path: Path to the HDF5 dataset
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for test
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' keys containing indices
    """
    # Validate fractions
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")
    
    # Get number of profiles from HDF5
    with h5py.File(h5_path, 'r') as hf:
        # Find any dataset to get the first dimension (number of profiles)
        first_key = next(iter(hf.keys()))
        n_profiles = hf[first_key].shape[0]
    
    if n_profiles == 0:
        raise ValueError(f"No profiles found in {h5_path}")
    
    # Calculate split sizes
    n_val = int(round(n_profiles * val_frac))
    n_test = int(round(n_profiles * test_frac))
    n_train = n_profiles - n_val - n_test
    
    # Generate and shuffle indices
    indices = list(range(n_profiles))
    rng = random.Random(random_seed)
    rng.shuffle(indices)
    
    # Create splits
    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    test_idx = sorted(indices[n_train + n_val:])
    
    splits = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx
    }
    
    logger.info(f"Generated splits from {n_profiles} profiles: "
                f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return splits


def get_config_str(config: Dict[str, Any], section: str, key: str, op_desc: str) -> str:
    """Safely extracts a path string from configuration."""
    if section not in config or not isinstance(config[section], dict):
        raise ValueError(f"Config section '{section}' missing or invalid for {op_desc}.")
    path_val = config[section].get(key)
    if not isinstance(path_val, str) or not path_val.strip():
        raise ValueError(f"Config key '{key}' in '{section}' missing or empty for {op_desc}.")
    return path_val.strip()


def load_or_generate_splits(
    config: Dict[str, Any], 
    data_root_dir: Path, 
    h5_path: Path
) -> Tuple[Dict[str, List[int]], Path]:
    """
    Load existing splits or generate new ones based on config.
    
    Returns:
        Tuple of (splits dict, splits file path)
    """
    # Check if splits file is specified and exists
    try:
        splits_filename = get_config_str(
            config, "data_paths_config", "dataset_splits_filename", "dataset splits"
        )
        splits_path = data_root_dir / splits_filename
        
        if splits_path.is_file():
            # Load existing splits
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            required = {"train", "validation", "test"}
            if not required.issubset(splits.keys()):
                raise ValueError(f"Splits file must contain keys: {required}")
            logger.info(f"Loaded existing dataset splits from {splits_path}")
            logger.info(f"Split sizes: {len(splits['train'])} train, "
                       f"{len(splits['validation'])} val, {len(splits['test'])} test.")
            return splits, splits_path
    except (KeyError, ValueError) as e:
        logger.info(f"No valid splits file found, will generate new splits. Reason: {e}")
    
    # Generate new splits
    logger.info("Generating new dataset splits...")
    
    # Get split parameters from config
    train_params = config.get("training_hyperparameters", {})
    val_frac = train_params.get("val_frac", 0.15)
    test_frac = train_params.get("test_frac", 0.15)
    
    # Get random seed from config
    misc_settings = config.get("miscellaneous_settings", {})
    random_seed = misc_settings.get("random_seed", DEFAULT_SEED)
    
    # Generate splits
    splits = generate_dataset_splits(
        h5_path=h5_path,
        val_frac=val_frac,
        test_frac=test_frac,
        random_seed=random_seed
    )
    
    # Save splits
    splits_path = h5_path.with_name(h5_path.stem + "_splits.json")
    save_json(splits, splits_path)
    logger.info(f"Saved generated splits to {splits_path}")
    
    return splits, splits_path


__all__ = [
    "setup_logging", "load_config", "validate_config", "ensure_dirs", 
    "save_json", "seed_everything", "generate_dataset_splits", 
    "get_config_str", "load_or_generate_splits"
]