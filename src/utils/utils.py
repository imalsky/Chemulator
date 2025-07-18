#!/usr/bin/env python3
"""
Simplified utility functions for the chemical kinetics pipeline.
- Removed excessive validation
- Simplified configuration loading
- Focused on essential utilities
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
    """Configure logging for the application."""
    format_string = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed}")


def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON configuration file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Try json5 first for comment support
    try:
        import json5
        with open(path, 'r', encoding='utf-8') as f:
            config = json5.load(f)
    except ImportError:
        # Fallback to standard json
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return config


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom encoder for numpy/torch types
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
            return super().default(obj)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directories(*paths: Union[str, Path]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)