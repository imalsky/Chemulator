#!/usr/bin/env python3
"""
utils.py - Core utilities for the flow-map training pipeline.

This module provides common utilities used across the codebase:
    - Reproducible seeding for random number generators
    - Configuration loading and validation
    - Atomic file operations for safe writes
    - Directory management

These utilities ensure consistent behavior across different components
and provide safety features like atomic writes that prevent data
corruption from interrupted operations.

Usage:
    from utils import seed_everything, load_json_config, ensure_dir

    seed_everything(1234, deterministic=True)
    cfg = load_json_config("config.json")
    output_dir = ensure_dir("models/run")
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for Python's random module, NumPy, and PyTorch (if available).
    Can optionally enable fully deterministic mode for PyTorch, which
    disables non-deterministic CUDA operations.

    Args:
        seed: Integer seed value for all random number generators.
            Should be in range [0, 2^32 - 1].
        deterministic: If True, enable fully deterministic mode in PyTorch.
            This disables cuDNN benchmark and uses deterministic algorithms.
            May significantly slow down training but ensures reproducibility.

    Note:
        Deterministic mode may not be fully supported for all operations.
        PyTorch will use warn_only=True to avoid crashes on unsupported ops.

    Example:
        >>> seed_everything(42, deterministic=True)
        >>> random.random()  # Reproducible
        >>> np.random.rand()  # Reproducible
        >>> torch.rand(1)  # Reproducible
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Disable cuDNN benchmark for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Use deterministic algorithms where available
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            # Enable cuDNN benchmark for performance
            # This may cause slight non-determinism
            torch.backends.cudnn.benchmark = True

    except ImportError:
        # PyTorch not installed, skip torch-specific seeding
        pass


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """
    Create directory if it doesn't exist.

    Creates the directory and all necessary parent directories.
    Safe to call on existing directories.

    Args:
        path: Directory path to create

    Returns:
        Path object for the directory

    Example:
        >>> output_dir = ensure_dir("models/experiment1/checkpoints")
        >>> output_dir.exists()  # True
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Reads and parses a JSON file, returning the parsed content as a
    Python dictionary. Provides clear error messages for common issues.

    Args:
        path: Path to the JSON config file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If JSON parsing fails (includes line number if available)

    Example:
        >>> cfg = load_json_config("config.json")
        >>> print(cfg["training"]["batch_size"])
        256
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {file_path}. "
            "Ensure the file exists and the path is correct."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {file_path} at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e


def atomic_write_json(
    path: Union[str, os.PathLike], obj: Any, *, indent: int = 2
) -> None:
    """
    Atomically write JSON to file using write-to-temp-then-rename pattern.

    This ensures that the file is either fully written or not modified at all,
    preventing corruption from interrupted writes (e.g., power loss, Ctrl+C).

    The operation:
    1. Writes to a temporary file (path.tmp)
    2. Renames temp file to target path (atomic on most filesystems)
    3. Cleans up temp file if rename fails

    Args:
        path: Destination file path
        obj: JSON-serializable Python object
        indent: JSON indentation level for pretty-printing (default: 2)

    Raises:
        TypeError: If obj is not JSON-serializable

    Example:
        >>> atomic_write_json("config.json", {"key": "value"})
        # File is either fully written or unchanged
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        # Serialize with sorted keys for reproducible output
        text = json.dumps(obj, indent=indent, sort_keys=True)

        # Write to temp file
        tmp.write_text(text + "\n", encoding="utf-8")

        # Atomic rename (on most filesystems)
        tmp.replace(path)

    finally:
        # Clean up temp file if it still exists (rename failed)
        if tmp.exists():
            tmp.unlink()


def now_ts() -> str:
    """
    Return current timestamp as formatted string.

    Format: YYYY-MM-DD HH:MM:SS

    Returns:
        Formatted timestamp string

    Example:
        >>> print(now_ts())
        2024-01-15 14:30:45
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")
