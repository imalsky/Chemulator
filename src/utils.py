#!/usr/bin/env python3
"""
utils.py - Core utilities for the flow-map training pipeline.

Provides:
    - Reproducible seeding with optional deterministic mode
    - Configuration loading and validation
    - Atomic file operations
    - Directory management
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

    Args:
        seed: Integer seed value for all RNGs.
        deterministic: If True, enable fully deterministic mode (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If JSON parsing fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e


def atomic_write_json(
    path: Union[str, os.PathLike], obj: Any, *, indent: int = 2
) -> None:
    """
    Atomically write JSON to file (write to temp, then rename).

    Args:
        path: Destination file path.
        obj: JSON-serializable object.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        text = json.dumps(obj, indent=indent, sort_keys=True)
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def now_ts() -> str:
    """Return current timestamp as formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
