#!/usr/bin/env python3
"""
Utility functions for the DeepONet chemical kinetics pipeline.
Provides logging, reproducibility, and I/O utilities.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def setup_logging(level: int = logging.INFO, log_file: Path = None) -> None:
    """
    Configure logging for the application.

    Sets up both console and file logging with consistent formatting.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file for persistent logging
    """
    format_string = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter with consistent timestamp format
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    """
    Set random seeds for complete reproducibility.

    Seeds all random number generators:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Integer seed for all RNGs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed} for reproducibility")


def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON configuration file with optional JSON5 support.

    Attempts to use JSON5 for comment support, falls back to standard JSON.

    Args:
        path: Path to configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Try json5 first for comment support in config files
    try:
        import json5
        with open(path, 'r', encoding='utf-8') as f:
            config = json5.load(f)
    except ImportError:
        # Fallback to standard json if json5 not available
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    return config


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file with proper type handling.

    Automatically handles conversion of numpy/torch types to JSON-serializable formats.

    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation level for readability
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        """Custom encoder for scientific computing types."""

        def default(self, obj):
            # Handle numpy types
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle torch types
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            # Handle Path objects
            elif isinstance(obj, Path):
                return str(obj)
            # Default to parent class handling
            return super().default(obj)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directories(*paths: Union[str, Path]) -> None:
    """
    Create directories if they don't exist.

    Args:
        *paths: Variable number of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def format_number(num: float, precision: int = 3) -> str:
    """
    Format number for display with appropriate precision.

    Uses scientific notation for very large/small numbers.

    Args:
        num: Number to format
        precision: Number of significant digits

    Returns:
        Formatted string representation
    """
    if abs(num) < 1e-3 or abs(num) > 1e3:
        return f"{num:.{precision}e}"
    else:
        return f"{num:.{precision}f}"


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculate model size statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts and memory usage
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate memory in MB (assuming float32)
    param_memory_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": param_memory_mb
    }