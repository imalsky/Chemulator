#!/usr/bin/env python3
"""
General utility functions for the chemical kinetics pipeline.

Provides helpers for:
- Configuration management with JSON5 support
- Logging setup
- Random seed control
- File I/O operations
- Data validation
"""

import json
import logging
import os
import random
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Custom format string for log messages
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    
    # Remove all existing handlers from all loggers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Also clean up any child logger handlers to prevent duplication
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
    
    # Configure root logger
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        try:
            # Ensure directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
            
            print(f"Logging to file: {log_file}")
        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # Only set CUDA seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # For better reproducibility (optional - may impact performance)
    # torch.use_deterministic_algorithms(True)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed}")


def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON configuration file with validation.
    
    Supports both standard JSON and JSON with comments (using json5 if available).
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
        ImportError: If json5 is needed but not installed
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Try json5 first for comment support
    try:
        import json5
        with open(path, 'r', encoding='utf-8') as f:
            config = json5.load(f)
    except ImportError:
        # Check if file likely contains comments
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if '//' in content or '/*' in content:
                raise ImportError(
                    "Configuration file appears to contain comments but json5 is not installed.\n"
                    "Install it with: pip install json5"
                )
        
        # Fallback to standard json
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Validate configuration
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required top-level sections
    required_sections = ["paths", "data", "normalization", "model", "training", "system"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")
    
    # Validate paths
    paths = config["paths"]
    if not isinstance(paths.get("raw_data_files"), list):
        raise ValueError("'paths.raw_data_files' must be a list")
    
    # Validate data specification
    data = config["data"]
    if not isinstance(data.get("species_variables"), list) or not data["species_variables"]:
        raise ValueError("'data.species_variables' must be a non-empty list")
    
    if not isinstance(data.get("global_variables"), list):
        raise ValueError("'data.global_variables' must be a list")
    
    # Enhanced validation: Check variable names
    all_variables = data["species_variables"] + data["global_variables"] + [data.get("time_variable", "t_time")]
    duplicates = []
    seen = set()
    
    for var in all_variables:
        # Check for duplicates
        if var in seen:
            duplicates.append(var)
        seen.add(var)
    
    if duplicates:
        raise ValueError(f"Duplicate variable names found: {duplicates}")
    
    # Validate model configuration
    model = config["model"]
    if model.get("type") not in ["siren", "resnet", "deeponet"]:
        raise ValueError("'model.type' must be either 'siren', 'resnet', or 'deeponet'")
    
    # Validate model-specific parameters
    model_type = model.get("type")
    
    if model_type in ["siren", "resnet"]:
        # These models require hidden_dims
        if not isinstance(model.get("hidden_dims"), list) or not model["hidden_dims"]:
            raise ValueError("'model.hidden_dims' must be a non-empty list for siren and resnet models")
    
    elif model_type == "deeponet":
        # DeepONet requires branch_layers, trunk_layers, and basis_dim
        if not isinstance(model.get("branch_layers"), list) or not model["branch_layers"]:
            raise ValueError("'model.branch_layers' must be a non-empty list for deeponet model")
        
        if not isinstance(model.get("trunk_layers"), list) or not model["trunk_layers"]:
            raise ValueError("'model.trunk_layers' must be a non-empty list for deeponet model")
        
        if not isinstance(model.get("basis_dim"), int) or model["basis_dim"] <= 0:
            raise ValueError("'model.basis_dim' must be a positive integer for deeponet model")
        
        # Validate activation function if specified
        if "activation" in model:
            valid_activations = {"gelu", "relu", "tanh"}
            if model["activation"] not in valid_activations:
                raise ValueError(f"'model.activation' must be one of {valid_activations}")
    
    # Validate training parameters
    training = config["training"]
    
    # Check numeric parameters
    numeric_params = {
        "batch_size": (1, None),
        "epochs": (1, None),
        "learning_rate": (0, 1),
        "gradient_clip": (0, None),
        "val_fraction": (0, 1),
        "test_fraction": (0, 1)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        value = training.get(param)
        if value is None:
            raise ValueError(f"Missing training parameter: '{param}'")
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{param}' must be numeric")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"'{param}' must be >= {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"'{param}' must be <= {max_val}")
    
    # Check split fractions sum
    if training["val_fraction"] + training["test_fraction"] >= 1.0:
        raise ValueError("Sum of val_fraction and test_fraction must be < 1.0")
    
    # Validate normalization methods
    norm_config = config["normalization"]
    valid_methods = {"standard", "log-standard", "log-min-max", "symlog", "none"}
    
    default_method = norm_config.get("default_method")
    if default_method not in valid_methods:
        raise ValueError(f"Invalid default normalization method: {default_method}")
    
    # Check per-variable methods
    methods = norm_config.get("methods", {})
    for var, method in methods.items():
        if method not in valid_methods:
            raise ValueError(f"Invalid normalization method for '{var}': {method}")


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save dictionary to JSON file with pretty printing.
    
    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom encoder for special types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    # Write file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directories(*paths: Union[str, Path]) -> None:
    """
    Create directories if they don't exist with thread safety.
    
    Args:
        *paths: Variable number of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)