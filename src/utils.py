#!/usr/bin/env python3
"""
utils.py - Optimized helper functions for configuration, logging, and data handling.

This module provides utility functions for:
- Configuration file loading and validation
- Logging setup
- Random seed management
- Dataset split generation
- JSON serialization with custom handlers
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch

# Try to import json5 for JSONC support (comments in JSON)
try:
    import json5 as _json_backend
    _HAS_JSON5 = True
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False

# Constants
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
DEFAULT_SEED = 42
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Set up logging configuration for console and optional file output.
    
    This function configures the root logger with appropriate handlers and formatters.
    It ensures no duplicate handlers are created if called multiple times.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file for persistent logging
    """
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to prevent duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler - always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - optional
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding=UTF8_ENCODING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Logging to console and file: {log_path.resolve()}")
        except Exception as e:
            print(f"Error: File logging setup failed for {log_file}: {e}. Continuing with console only.", file=sys.stderr)
    else:
        print("Logging to console only.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate a configuration file with JSONC support.
    
    This function loads JSON configuration files and supports JSON5 format
    which allows comments and trailing commas for better readability.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        RuntimeError: If configuration is invalid or cannot be parsed
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Use UTF-8-sig encoding to handle potential BOM
        config_text = config_path.read_text(encoding=UTF8_SIG_ENCODING)
        config_dict = _json_backend.loads(config_text)
        validate_config(config_dict)
        
        backend = "JSON5" if _HAS_JSON5 else "JSON"
        logger.info(f"Successfully loaded and validated {backend} config from {config_path}.")
        return config_dict
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate config {config_path}: {e}") from e


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the structure and content of a configuration dictionary.
    
    This function ensures all required sections and keys are present
    and have valid values according to the expected schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate data specification
    data_spec = config.get("data_specification")
    if not isinstance(data_spec, dict):
        raise ValueError("Config section 'data_specification' is missing or not a dictionary.")
    
    if not isinstance(data_spec.get("species_variables"), list) or not data_spec.get("species_variables"):
        raise ValueError("Config key 'species_variables' in 'data_specification' must be a non-empty list.")
    
    if not isinstance(data_spec.get("all_variables"), list) or not data_spec.get("all_variables"):
        raise ValueError("Config key 'all_variables' in 'data_specification' must be a non-empty list.")
    
    # Validate model parameters
    model_params = config.get("model_hyperparameters")
    if not isinstance(model_params, dict):
        raise ValueError("Config section 'model_hyperparameters' is missing or not a dictionary.")
    
    if not isinstance(model_params.get("hidden_dims"), list) or not model_params.get("hidden_dims"):
        raise ValueError("Config key 'hidden_dims' in 'model_hyperparameters' must be a non-empty list.")
    
    # Validate model type (should be SIREN)
    model_type = model_params.get("model_type", "siren").lower()
    if model_type != "siren":
        logger.warning(f"Model type '{model_type}' specified, but only SIREN is implemented.")
    
    # Validate numeric parameters
    dropout = model_params.get("dropout")
    if dropout is not None and not (0.0 <= dropout < 1.0):
        raise ValueError("'dropout' in 'model_hyperparameters' must be a float in the range [0.0, 1.0).")


def ensure_dirs(*paths: Union[str, Path]) -> bool:
    """
    Create one or more directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths to create
        
    Returns:
        True if all directories were created or already exist, False on error
    """
    try:
        for path in paths: 
            Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for handling special types.
    
    This function handles serialization of numpy arrays, torch tensors,
    Path objects, and sets, converting them to JSON-compatible formats.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """
    Save a dictionary to a JSON file with pretty printing.
    
    This function handles custom types using the _json_serializer and
    ensures the output directory exists before writing.
    
    Args:
        data: Dictionary to save
        path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = Path(path)
        ensure_dirs(json_path.parent)
        
        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            json.dump(data, f, indent=2, default=_json_serializer, ensure_ascii=False)
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}", exc_info=True)
        return False


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for Python's random module, NumPy,
    and PyTorch to ensure reproducible results.
    
    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Note: The following settings can impact performance
        # Uncomment only if strict reproducibility is required:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    logger.info(f"Global random seed set to {seed}.")


def generate_dataset_splits(
    h5_path: Path, 
    val_frac: float = 0.15, 
    test_frac: float = 0.15,
    random_seed: int = DEFAULT_SEED
) -> Dict[str, List[int]]:
    """
    Generate train/validation/test splits from HDF5 dataset.
    
    This function creates random splits of profile indices for training,
    validation, and testing, ensuring no overlap between sets.
    
    Args:
        h5_path: Path to the HDF5 dataset
        val_frac: Fraction of data for validation (0-1)
        test_frac: Fraction of data for test (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' keys containing sorted indices
        
    Raises:
        ValueError: If fractions are invalid or dataset is empty
    """
    # Validate fractions
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")
    
    # Get number of profiles from HDF5
    try:
        with h5py.File(h5_path, 'r') as hf:
            if not hf.keys():
                raise ValueError(f"HDF5 file is empty: {h5_path}")
            
            # Use the first dataset to determine number of profiles
            first_key = next(iter(hf.keys()))
            n_profiles = hf[first_key].shape[0]
            
    except Exception as e:
        raise ValueError(f"Failed to read HDF5 file {h5_path}: {e}") from e
    
    if n_profiles == 0:
        raise ValueError(f"No profiles found in {h5_path}")
    
    # Calculate split sizes
    n_val = int(round(n_profiles * val_frac))
    n_test = int(round(n_profiles * test_frac))
    
    # Ensure at least one sample in each split
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    
    n_train = n_profiles - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Training split has {n_train} samples. "
            f"Reduce validation ({val_frac}) or test ({test_frac}) fractions."
        )
    
    # Generate and shuffle indices
    indices = list(range(n_profiles))
    rng = random.Random(random_seed)
    rng.shuffle(indices)
    
    # Create splits
    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train : n_train + n_val])
    test_idx = sorted(indices[n_train + n_val :])
    
    splits = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx
    }
    
    logger.info(
        f"Generated splits from {n_profiles} profiles: "
        f"train={len(train_idx)} ({len(train_idx)/n_profiles:.1%}), "
        f"val={len(val_idx)} ({len(val_idx)/n_profiles:.1%}), "
        f"test={len(test_idx)} ({len(test_idx)/n_profiles:.1%})"
    )
    
    return splits


def get_config_str(config: Dict[str, Any], section: str, key: str, op_desc: str) -> str:
    """
    Safely extract a non-empty string value from configuration.
    
    This function provides clear error messages when configuration
    values are missing or invalid.
    
    Args:
        config: Configuration dictionary
        section: Section name in config
        key: Key name within section
        op_desc: Description of operation for error message
        
    Returns:
        Trimmed string value
        
    Raises:
        ValueError: If section/key is missing or value is empty
    """
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
    Load existing dataset splits or generate new ones.
    
    This function first attempts to load splits from a specified file.
    If that fails, it generates new splits based on configuration parameters.
    
    Args:
        config: Configuration dictionary
        data_root_dir: Root directory for data files
        h5_path: Path to HDF5 dataset
        
    Returns:
        Tuple of (splits dictionary, path to splits file)
    """
    splits_path = None
    
    # Try to load existing splits
    try:
        splits_filename = get_config_str(
            config, "data_paths_config", "dataset_splits_filename", "dataset splits"
        )
        splits_path = data_root_dir / splits_filename
        
        logger.info(f"Attempting to load dataset splits from: {splits_path}")
        
        # Verify file exists before trying to load
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_path}")
        
        with open(splits_path, 'r', encoding=UTF8_ENCODING) as f:
            splits = json.load(f)
        
        # Validate splits structure
        required_keys = {"train", "validation", "test"}
        if not required_keys.issubset(splits.keys()):
            raise ValueError(f"Splits file must contain keys: {required_keys}")
        
        # Validate that splits contain valid indices
        for key in required_keys:
            if not isinstance(splits[key], list) or not splits[key]:
                raise ValueError(f"Split '{key}' must be a non-empty list")
        
        logger.info(f"Successfully loaded dataset splits from {splits_path}")
        logger.info(
            f"Split sizes: {len(splits['train'])} train, "
            f"{len(splits['validation'])} val, {len(splits['test'])} test."
        )
        return splits, splits_path
    
    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"Could not load specified splits file. Reason: {e}. Will generate new splits.")
    
    # Generate new splits if loading failed
    logger.info("Generating new dataset splits...")
    
    # Get split parameters from config
    train_params = config.get("training_hyperparameters", {})
    val_frac = train_params.get("val_frac", 0.15)
    test_frac = train_params.get("test_frac", 0.15)
    
    misc_settings = config.get("miscellaneous_settings", {})
    random_seed = misc_settings.get("random_seed", DEFAULT_SEED)
    
    # Generate splits
    splits = generate_dataset_splits(
        h5_path=h5_path,
        val_frac=val_frac,
        test_frac=test_frac,
        random_seed=random_seed
    )
    
    # Save generated splits
    new_splits_path = h5_path.with_name(h5_path.stem + "_splits_generated.json")
    if save_json(splits, new_splits_path):
        logger.info(f"Saved newly generated splits to {new_splits_path}")
    else:
        logger.warning("Failed to save generated splits to file")
    
    return splits, new_splits_path


__all__ = [
    "setup_logging", "load_config", "validate_config", "ensure_dirs", 
    "save_json", "seed_everything", "generate_dataset_splits", 
    "get_config_str", "load_or_generate_splits"
]