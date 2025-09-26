#!/usr/bin/env python3
"""
Preprocessing Utilities Module
==============================
Helper functions and classes for data preprocessing pipeline.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


# Configuration constants
TQDM_MININTERVAL = 0.25
TQDM_SMOOTHING = 0.1
TQDM_LEAVE_OUTER = True
TQDM_LEAVE_INNER = False

PARALLEL_SCAN_TIMEOUT_PER_FILE = 300  # 5 minutes per file
PARALLEL_SCAN_OVERHEAD = 120  # 2 minutes overhead

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
    
    Args:
        config: Configuration dictionary
        path: Sequence of keys for nested access
        required: Whether to raise error if not found
        default: Default value if not found
        
    Returns:
        Configuration value or default
        
    Raises:
        KeyError: If required and not found
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
    Online statistics accumulator using Welford's algorithm.
    
    Computes mean, variance, min, and max in a single pass with
    numerical stability for large datasets.
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
        
        Args:
            array: Data array to incorporate
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
        
        Args:
            min_std: Minimum standard deviation floor
            
        Returns:
            Tuple of (mean, std, min, max)
            
        Raises:
            RuntimeError: If no data was accumulated
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
        
        Args:
            need_mean_std: Whether to compute mean and std
            need_min_max: Whether to compute min and max
            need_log: Whether to compute log-domain statistics
            epsilon: Floor value for log operations
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
        
        Args:
            min_std: Minimum standard deviation floor
            
        Returns:
            Dictionary of statistics
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
    
    Args:
        method: Normalization method name
        
    Returns:
        Tuple of (need_mean_std, need_min_max, need_log)
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


def scan_hdf5_file_worker(
    path_str: str,
    species_vars: List[str],
    global_vars: List[str],
    time_key: str,
    min_value_threshold: float,
    epsilon: float,
    chunk_size: int,
) -> Tuple[Dict[str, Any], List[str], Optional[np.ndarray], Dict[str, int]]:
    """
    Worker function for parallel HDF5 file scanning with detailed progress tracking.
    
    Returns:
        Tuple of (file_report, valid_groups, time_candidate, progress_stats)
    """
    if h5py is None:
        raise RuntimeError("h5py is required but not available")
    
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    
    total_groups = 0
    valid_groups_list = []
    num_nan = 0
    num_below_threshold = 0
    time_candidate = None
    
    # Progress tracking
    progress_stats = {
        "filename": path.name,
        "groups_processed": 0,
        "groups_valid": 0,
        "groups_dropped": 0,
    }
    
    with h5py.File(path, "r") as hdf:
        group_names = list(hdf.keys())
        total_groups_in_file = len(group_names)
        progress_stats["total_groups"] = total_groups_in_file
        
        for group_idx, group_name in enumerate(group_names, 1):
            group = hdf[group_name]
            total_groups += 1
            progress_stats["groups_processed"] = group_idx
            
            # Validate time variable
            if time_key not in group:
                num_nan += 1
                continue
            
            time_data = np.array(group[time_key], dtype=np.float64, copy=False).reshape(-1)
            
            if time_data.size < 2 or not np.all(np.isfinite(time_data)):
                num_nan += 1
                continue
            
            # Check time monotonicity
            time_diffs = np.diff(time_data)
            if ALLOW_EQUAL_TIMEPOINTS:
                if np.any(time_diffs < -abs(TIME_DECREASE_TOLERANCE)):
                    num_nan += 1
                    continue
            else:
                if not np.all(time_diffs > 0.0):
                    num_nan += 1
                    continue
            
            # Check time grid consistency within file
            if time_candidate is None:
                time_candidate = time_data.copy()
            else:
                if not np.array_equal(time_data, time_candidate):
                    raise ValueError(
                        f"{path}:{group_name}: Time grid differs within file. "
                        f"All trajectories must have identical time grids."
                    )
            
            # Validate global variables
            try:
                global_values = np.array(
                    [float(group.attrs[key]) for key in global_vars],
                    dtype=np.float64
                )
                if not np.all(np.isfinite(global_values)):
                    num_nan += 1
                    continue
            except Exception:
                num_nan += 1
                continue
            
            # Validate species variables
            T = int(time_data.shape[0])
            has_nan = False
            below_threshold = False
            
            for species_name in species_vars:
                if species_name not in group:
                    has_nan = True
                    break
                
                dataset = group[species_name]
                if dataset.shape[0] != T:
                    has_nan = True
                    break
                
                # Read in chunks for efficiency
                chunk_len = chunk_size if chunk_size > 0 else T
                for start_idx in range(0, T, chunk_len):
                    end_idx = min(T, start_idx + chunk_len)
                    chunk_data = np.array(
                        dataset[start_idx:end_idx],
                        dtype=np.float64,
                        copy=False
                    ).reshape(-1)
                    
                    if not np.isfinite(chunk_data).all():
                        has_nan = True
                        break
                    
                    if (chunk_data < min_value_threshold).any():
                        below_threshold = True
                        break
                
                if has_nan or below_threshold:
                    break
            
            if has_nan:
                num_nan += 1
                continue
            elif below_threshold:
                num_below_threshold += 1
                continue
            
            valid_groups_list.append(group_name)
    
    progress_stats["groups_valid"] = len(valid_groups_list)
    progress_stats["groups_dropped"] = total_groups - len(valid_groups_list)
    
    file_report = {
        "path": str(path),
        "n_total": total_groups,
        "n_valid": len(valid_groups_list),
        "n_dropped": total_groups - len(valid_groups_list),
        "n_nan": num_nan,
        "n_below_threshold": num_below_threshold,
    }
    
    return file_report, valid_groups_list, time_candidate, progress_stats