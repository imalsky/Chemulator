#!/usr/bin/env python3
"""
Helper functions and classes for data preprocessing pipeline.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


TIME_DECREASE_TOLERANCE = 0.0
ALLOW_EQUAL_TIMEPOINTS = False

DEFAULT_HDF5_CHUNK_SIZE = 0  # 0 means use dataset native chunk
SHARD_FILENAME_FORMAT = "shard_{split}_{filetag}_{idx:05d}.npz"


def format_bytes(num_bytes: int | float) -> str:
    """Format byte count as human-readable string."""
    num_bytes = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_config_value(
    config: Dict[str, Any],
    path: Sequence[str],
    required: bool = False,
    default=None
):
    """
    Extract nested value from configuration dictionary.
    """
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            if required:
                raise KeyError(f"Missing required config key: {'.'.join(path)}")
            return default
        current = current[key]
    return current


def get_storage_dtype(config: Dict[str, Any]) -> np.dtype:
    """Get numpy dtype for storage from configuration."""
    system_config = config.get("system", {})
    dtype_str = str(system_config.get("io_dtype", system_config.get("dtype", "float32"))).lower()
    
    if dtype_str not in {"float32", "float64"}:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Use 'float32' or 'float64'.")
    
    return np.float32 if dtype_str == "float32" else np.float64


def deterministic_hash(text: str, seed: int) -> float:
    """Generate deterministic hash value in [0, 1]."""
    encoded = f"{seed}:{text}".encode("utf-8")
    hash_bytes = hashlib.sha256(encoded).digest()
    hash_int = int.from_bytes(hash_bytes[:8], "big", signed=False)
    return hash_int / float(1 << 64)


class WelfordAccumulator:
    """
    Statistics accumulator using Welford's algorithm.
    """
    
    __slots__ = ("count", "mean", "M2", "min_val", "max_val")
    
    def __init__(self):
        """Initialize accumulator."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = math.inf
        self.max_val = -math.inf
    
    def update(self, array: np.ndarray) -> None:
        """
        Update statistics with new data.
        """
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        
        # Batch statistics
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_var = float(values.var(ddof=0))
        batch_M2 = batch_var * batch_count
        
        # Combine with running statistics
        total_count = self.count + batch_count
        if total_count == 0:
            return
        
        delta = batch_mean - self.mean
        combined_mean = self.mean + delta * (batch_count / total_count)
        combined_M2 = self.M2 + batch_M2 + delta * delta * (self.count * batch_count / total_count)
        
        self.count = total_count
        self.mean = combined_mean
        self.M2 = combined_M2
        self.min_val = min(self.min_val, float(values.min()))
        self.max_val = max(self.max_val, float(values.max()))
    
    def finalize(self, min_std: float) -> Tuple[float, float, float, float]:
        """
        Get final statistics.
        """
        if self.count <= 0:
            raise RuntimeError("No data accumulated in Welford accumulator")
        
        variance = max(0.0, self.M2 / self.count)
        std = math.sqrt(variance)
        return (
            float(self.mean),
            float(max(std, min_std)),
            float(self.min_val),
            float(self.max_val)
        )


class RunningStatistics:
    """
    Accumulates statistics in both raw and log domains as needed.
    """
    
    def __init__(
        self,
        need_mean_std: bool,
        need_min_max: bool,
        need_log: bool,
        epsilon: float
    ):
        """
        Initialize statistics accumulator.
        """
        self.need_mean_std = bool(need_mean_std)
        self.need_min_max = bool(need_min_max)
        self.need_log = bool(need_log)
        self.epsilon = float(epsilon)
        
        self.raw = WelfordAccumulator() if (need_mean_std or need_min_max) else None
        self.log = WelfordAccumulator() if need_log else None
    
    def update(self, array: np.ndarray) -> None:
        """Update statistics with new data."""
        array = np.asarray(array)
        if array.size == 0:
            return
        
        if self.raw is not None:
            self.raw.update(array)
        
        if self.log is not None:
            log_values = np.log10(np.clip(array, self.epsilon, None))
            self.log.update(log_values)
    
    def to_manifest(self, min_std: float) -> Dict[str, float]:
        """
        Export statistics to manifest format.
        """
        output = {}
        
        if self.raw is not None:
            mean, std, min_val, max_val = self.raw.finalize(min_std)
            if self.need_mean_std:
                output["mean"] = mean
                output["std"] = std
            if self.need_min_max:
                output["min"] = min_val
                output["max"] = max_val
        
        if self.log is not None:
            log_mean, log_std, log_min, log_max = self.log.finalize(min_std)
            output["log_mean"] = log_mean
            output["log_std"] = log_std
            output["log_min"] = log_min
            output["log_max"] = log_max
        
        return output

def get_normalization_flags(method: str) -> Tuple[bool, bool, bool]:
    """
    Get flags for which statistics are needed for a normalization method.
    """
    method = str(method)
    
    if method == "standard":
        return True, False, False
    elif method == "min-max":
        return False, True, False
    elif method == "log-standard":
        return True, False, True
    elif method == "log-min-max":
        return False, True, True
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")