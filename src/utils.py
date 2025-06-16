#!/usr/bin/env python3
"""
utils.py - A collection of shared helper functions for the project.
"""
from __future__ import annotations
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

try:
    import json5 as _json_backend
except ImportError:
    _json_backend = json

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    while root_logger.handlers: root_logger.handlers.pop().close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to console and file: {log_file}")
        except Exception as exc:
            logger.error(f"File logging setup failed for {log_file}: {exc}. Console only.")

def load_config(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    config_path = Path(path)
    if not config_path.is_file():
        logger.critical(f"Configuration file not found: {config_path}"); return None
    try:
        config_dict = _json_backend.loads(config_path.read_text(encoding="utf-8-sig"))
        validate_config(config_dict)
        logger.info("Successfully loaded and validated configuration from %s.", config_path)
        return config_dict
    except Exception as e:
        logger.critical(f"Failed to load or validate config {config_path}: {e}", exc_info=True)
        return None

def _validate_parameter(
    config: Dict[str, Any], key: str, expected_type: Union[type, tuple[type, ...]],
    required: bool = True, non_empty: bool = False, min_value: Optional[Union[int, float]] = None
) -> Any:
    if key not in config:
        if required: raise ValueError(f"Required configuration key '{key}' is missing.")
        return None
    value = config[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"Configuration key '{key}' must be {expected_type}, got {type(value)}.")
    if non_empty and isinstance(value, (list, dict)) and not value:
        raise ValueError(f"Configuration key '{key}' must be non-empty.")
    if min_value is not None and isinstance(value, (int, float)) and value < min_value:
        raise ValueError(f"'{key}' ({value}) must be >= {min_value}.")
    return value

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the configuration dictionary. It checks for common parameters
    and then conditionally validates model-specific hyperparameters.
    """
    # Common parameters for all models
    _validate_parameter(config, "species_variables", list, required=True, non_empty=True)
    _validate_parameter(config, "global_variables", list, required=True) # Can be empty
    _validate_parameter(config, "all_variables", list, required=True, non_empty=True)
    
    dropout = config.get("dropout", 0.1)
    if not isinstance(dropout, (float, int)) or not (0.0 <= dropout < 1.0):
        raise ValueError("'dropout' must be a float or int in the range [0.0, 1.0).")
    config["dropout"] = float(dropout)

    # --- Conditional Validation based on model type ---
    # If "hidden_dims" is present, we assume it's an MLP config.
    if "hidden_dims" in config:
        _validate_parameter(config, "hidden_dims", list, required=True, non_empty=True)
    # Otherwise, assume it's a Transformer config.
    else:
        d_model = _validate_parameter(config, "d_model", int, required=True, min_value=1)
        nhead = _validate_parameter(config, "nhead", int, required=True, min_value=1)
        if d_model % nhead != 0:
            raise ValueError(f"'d_model' ({d_model}) must be divisible by 'nhead' ({nhead}).")
        _validate_parameter(config, "num_encoder_layers", int, required=True, min_value=1)
        _validate_parameter(config, "num_query_layers", int, required=True, min_value=1)
        _validate_parameter(config, "dim_feedforward", int, required=True, min_value=1)
        _validate_parameter(config, "max_sequence_length", int, required=True, min_value=1)

def ensure_dirs(*paths: Union[str, Path]) -> bool:
    try:
        for p in paths: Path(p).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {paths}: {e}"); return False

def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if torch.is_tensor(obj): return obj.detach().cpu().tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, set): return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    try:
        ensure_dirs(Path(path).parent)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_serializer, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.error(f"Failed to save JSON to {path}: {exc}", exc_info=True); return False

def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to {seed}.")

__all__ = ["setup_logging", "load_config", "validate_config", "ensure_dirs", "save_json", "seed_everything"]