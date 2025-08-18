#!/usr/bin/env python3
"""
This script transforms raw chemical kinetics simulation data from HDF5 format into a
set of sharded NumPy (`.npz`) files suitable for efficient training of neural
networks. It is optimized for speed and correctness, performing all necessary
steps in a single pass over the input data.

Core Functionality:
1.  **Single-Pass Processing**: It reads each trajectory from the source HDF5 files
    only once. During this single pass, it performs data validation, updates
    streaming normalization statistics, deterministically splits the data into
    train/validation/test sets, and writes the processed data to output shards.
    This architecture significantly reduces I/O overhead compared to a multi-pass
    approach.

2.  **Data Validation**: Each trajectory undergoes validation. It is
    dropped if it contains missing keys, non-finite values (NaNs, infs),
    non-monotonic time steps, or species concentrations below a configurable
    minimum threshold. This script requires fixed-length trajectories; all
    valid trajectories across all input files must have the same number of
    time steps.

3.  **Streaming Normalization Statistics**: The script calculates normalization
    statistics (mean, standard deviation, min, max) for all variables on the fly
    using Welford's algorithm. This avoids the need for a preliminary pass and
    ensures that the final `normalization.json` file contains accurate global
    statistics for the entire dataset.

4.  **Deterministic Data Splitting**: Trajectories are assigned to 'train',
    'validation', or 'test' splits using a stable hashing function (SHA256).
    This ensures that preprocessing runs are reproducible across different
    machines and executions, a critical requirement for consistent model
    evaluation.

5.  **Efficient Sharding**: Validated and transformed data is written to
    compressed `.npz` files. Sharding allows for efficient
    loading of large datasets by breaking them into smaller, manageable chunks.
    Shard names are also generated deterministically for reproducibility.

6.  **Parallel Processing**: The script can leverage multiple CPU cores to process
    HDF5 files in parallel, significantly speeding up the workflow for large
    collections of input files. Statistics from each worker process are correctly
    merged to produce accurate global results.

Workflow:
- The `DataPreprocessor` class orchestrates the entire process.
- It initializes a `SinglePassPreprocessor` for each input file (either in serial
  or in parallel via `_process_file_worker`).
- The `SinglePassPreprocessor` reads trajectories, validates them using
  `_extract_trajectory`, updates a `DataStatisticsLogger` instance, and writes
  to a `SequenceShardWriter`.
- After all files are processed, the main class finalizes the normalization
  statistics, merges metadata from all files, and saves three key artifacts:
    1.  `shards/`: Directories containing the processed `.npz` data shards.
    2.  `normalization.json`: Contains statistics for normalizing data during training.
    3.  `shard_index.json`: A manifest file describing the dataset structure and
        the location of all shards.
"""

import logging
import math
import time
import hashlib
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm

from utils.utils import save_json


# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MIN_VALUE_THRESHOLD = 1e-30
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10


# ============================================================================
# DATA STATISTICS TRACKING
# ============================================================================

@dataclass
class DataStatistics:
    """Track comprehensive preprocessing statistics."""
    
    # Trajectory counts
    total_groups: int = 0
    valid_trajectories: int = 0
    
    # Drop reasons
    dropped_missing_keys: int = 0
    dropped_non_finite: int = 0
    dropped_below_threshold: int = 0
    dropped_use_fraction: int = 0
    
    # Data statistics
    total_time_points: int = 0
    species_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    global_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range: Tuple[float, float] = (float('inf'), float('-inf'))
    
    # Split distribution
    split_distribution: Dict[str, int] = field(default_factory=lambda: {
        "train": 0, "validation": 0, "test": 0
    })
    
    # Performance metrics
    processing_times: Dict[str, float] = field(default_factory=dict)
    file_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def merge_inplace(self, other: 'DataStatistics') -> None:
        """Merge another statistics object into this one."""
        self.total_groups += other.total_groups
        self.valid_trajectories += other.valid_trajectories
        self.dropped_missing_keys += other.dropped_missing_keys
        self.dropped_non_finite += other.dropped_non_finite
        self.dropped_below_threshold += other.dropped_below_threshold
        self.dropped_use_fraction += other.dropped_use_fraction
        self.total_time_points += other.total_time_points
        
        # Merge ranges
        for k, (mn, mx) in other.species_min_max.items():
            if k in self.species_min_max:
                omn, omx = self.species_min_max[k]
                self.species_min_max[k] = (min(omn, mn), max(omx, mx))
            else:
                self.species_min_max[k] = (mn, mx)
        
        for k, (mn, mx) in other.global_min_max.items():
            if k in self.global_min_max:
                omn, omx = self.global_min_max[k]
                self.global_min_max[k] = (min(omn, mn), max(omx, mx))
            else:
                self.global_min_max[k] = (mn, mx)
        
        self.time_range = (
            min(self.time_range[0], other.time_range[0]),
            max(self.time_range[1], other.time_range[1]),
        )
        
        for split, cnt in other.split_distribution.items():
            self.split_distribution[split] += cnt
        
        self.processing_times.update(other.processing_times)
        self.file_stats.update(other.file_stats)


@dataclass
class StreamingStatistics:
    """Streaming statistics accumulator using Welford's algorithm."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    log_min: float = float('inf')
    log_max: float = float('-inf')


class DataStatisticsLogger:
    """Logger for comprehensive data statistics during preprocessing."""
    
    def __init__(self, output_dir: Path):
        """Initialize statistics logger."""
        self.output_dir = output_dir
        self.stats = DataStatistics()
        self.logger = logging.getLogger(__name__)
        # Streaming accumulators for normalization stats
        self.accumulators: Dict[str, StreamingStatistics] = {}
    
    def update_accumulator(self, var: str, data: np.ndarray, log_floor: float, method: str) -> None:
        """Update streaming statistics for a variable."""
        if var not in self.accumulators:
            self.accumulators[var] = StreamingStatistics()
        
        acc = self.accumulators[var]
        
        # Update min/max
        acc.min_val = min(acc.min_val, float(data.min()))
        acc.max_val = max(acc.max_val, float(data.max()))
        
        # Log-domain min/max
        log_data = np.log10(np.maximum(data, log_floor))
        acc.log_min = min(acc.log_min, float(log_data.min()))
        acc.log_max = max(acc.log_max, float(log_data.max()))
        
        # Welford's algorithm for mean/variance
        data_for_stats = log_data if "log" in method else data
        n0 = acc.count
        n1 = int(data_for_stats.size)
        if n1 == 0:
            return
        
        c_mean = float(data_for_stats.mean())
        c_m2 = float(((data_for_stats - c_mean) ** 2).sum())
        
        n = n0 + n1
        if n0 == 0:
            mean, m2 = c_mean, c_m2
        else:
            delta = c_mean - acc.mean
            mean = acc.mean + delta * (n1 / n)
            m2 = acc.m2 + c_m2 + (delta * delta) * (n0 * n1 / n)
        
        acc.count, acc.mean, acc.m2 = n, mean, m2
    
    def update_species_range(self, var: str, data: np.ndarray) -> None:
        """Update min/max range for a species variable."""
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        
        vmin, vmax = float(finite.min()), float(finite.max())
        if var in self.stats.species_min_max:
            omin, omax = self.stats.species_min_max[var]
            self.stats.species_min_max[var] = (min(omin, vmin), max(omax, vmax))
        else:
            self.stats.species_min_max[var] = (vmin, vmax)
    
    def update_global_range(self, var: str, value: float) -> None:
        """Update min/max range for a global variable."""
        if var in self.stats.global_min_max:
            omin, omax = self.stats.global_min_max[var]
            self.stats.global_min_max[var] = (min(omin, value), max(omax, value))
        else:
            self.stats.global_min_max[var] = (value, value)
    
    def update_time_range(self, times: np.ndarray) -> None:
        """Update time range statistics."""
        if times.size == 0:
            return
        
        tmin, tmax = float(times.min()), float(times.max())
        self.stats.time_range = (
            min(self.stats.time_range[0], tmin),
            max(self.stats.time_range[1], tmax),
        )
        self.stats.total_time_points += int(times.size)
    
    def finalize_normalization_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize accumulated statistics into normalization format."""
        norm_cfg = config["normalization"]
        default_method = norm_cfg.get("default_method", "log-standard")
        methods_override = norm_cfg.get("methods", {})
        min_std = float(norm_cfg.get("min_std", DEFAULT_MIN_STD))
        
        data_cfg = config["data"]
        species_vars = list(data_cfg["species_variables"])
        target_species_vars = list(data_cfg.get("target_species_variables", species_vars))
        time_var = str(data_cfg.get("time_variable", "t_time"))
        
        stats = {
            "per_key_stats": {},
            "normalization_methods": {},
            "already_logged_vars": sorted(set(species_vars) | set(target_species_vars))
        }
        
        for var, acc in self.accumulators.items():
            if acc.count == 0:
                continue
            
            var_method = methods_override.get(var, default_method)
            stats["normalization_methods"][var] = var_method
            
            variance = acc.m2 / max(1, acc.count - 1)
            std = math.sqrt(max(0.0, variance))
            
            block = {
                "method": var_method,
                "min": float(acc.min_val) if math.isfinite(acc.min_val) else 0.0,
                "max": float(acc.max_val) if math.isfinite(acc.max_val) else 1.0,
                "log_min": float(acc.log_min) if math.isfinite(acc.log_min) else -30.0,
                "log_max": float(acc.log_max) if math.isfinite(acc.log_max) else 0.0,
            }
            
            if "standard" in var_method:
                if "log" in var_method:
                    block["log_mean"] = float(acc.mean)
                    block["log_std"] = max(std, min_std)
                else:
                    block["mean"] = float(acc.mean)
                    block["std"] = max(std, min_std)
            
            stats["per_key_stats"][var] = block
        
        # Time normalization
        if time_var in self.accumulators:
            acc = self.accumulators[time_var]
            time_method = methods_override.get(time_var, default_method)
            
            if acc.count > 0:
                tn = {
                    "tmin_raw": float(acc.min_val),
                    "tmax_raw": float(acc.max_val),
                    "time_transform": time_method,
                }
                if time_method == "time-norm":
                    tau0 = float(norm_cfg.get("tau0", 1.0))
                    tn["tau0"] = tau0
                    tn["tmin"] = math.log1p(tn["tmin_raw"] / tau0)
                    tn["tmax"] = math.log1p(tn["tmax_raw"] / tau0)
                stats["time_normalization"] = tn
        
        return stats
    
    def save_summary(self) -> None:
        """Save preprocessing summary to JSON file."""
        summary_path = self.output_dir / "preprocessing_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_groups_processed": self.stats.total_groups,
            "valid_trajectories": self.stats.valid_trajectories,
            "dropped_counts": {
                "missing_keys": self.stats.dropped_missing_keys,
                "non_finite": self.stats.dropped_non_finite,
                "below_threshold": self.stats.dropped_below_threshold,
                "use_fraction": self.stats.dropped_use_fraction,
            },
            "data_ranges": {
                "species": self.stats.species_min_max,
                "globals": self.stats.global_min_max,
                "time": {"min": self.stats.time_range[0], "max": self.stats.time_range[1]},
            },
            "split_distribution": self.stats.split_distribution,
            "total_time_points": self.stats.total_time_points,
            "processing_times": self.stats.processing_times,
            "file_statistics": self.stats.file_stats,
        }
        save_json(summary, summary_path)
        self.logger.info(f"Statistics summary saved to {summary_path}")


# ============================================================================
# OPTIMIZED HDF5 READING
# ============================================================================

class OptimizedHDF5Reader:
    """Memory-efficient HDF5 reader with intelligent caching."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """Initialize optimized reader."""
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        self._cache = {}  # Cache for recently read datasets
    
    def read_dataset(
        self,
        group: h5py.Group,
        var_name: str,
        indices: Optional[np.ndarray] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """Read dataset with caching to avoid redundant reads."""
        if var_name not in group:
            raise KeyError(f"Variable {var_name} not found in group")
        
        cache_key = f"{id(group)}_{var_name}"
        
        if use_cache and cache_key in self._cache:
            data = self._cache[cache_key]
            return data[indices] if indices is not None else data
        
        dset = group[var_name]
        
        if indices is not None:
            data = dset[indices]
        else:
            total = dset.shape[0]
            if total <= self.chunk_size:
                data = dset[:]
            else:
                # Read in chunks
                chunks = []
                for i in range(0, total, self.chunk_size):
                    end = min(i + self.chunk_size, total)
                    chunks.append(dset[i:end])
                data = np.concatenate(chunks)
        
        if use_cache:
            self._cache[cache_key] = data
        
        return data
    
    def clear_cache(self):
        """Clear the cache to free memory."""
        self._cache.clear()


# ============================================================================
# SHARD WRITER
# ============================================================================

class SequenceShardWriter:
    """Efficient writer for trajectory sequence shards."""
    
    def __init__(
        self,
        output_dir: Path,
        trajectories_per_shard: int,
        shard_idx_base: str,
        n_species: int,
        n_globals: int,
        dtype: np.dtype,
        compressed: bool = True,
    ):
        """Initialize shard writer."""
        self.output_dir = output_dir
        self.trajectories_per_shard = max(1, int(trajectories_per_shard))
        self.shard_idx_base = shard_idx_base
        self.n_species = n_species
        self.n_globals = n_globals
        self.dtype = dtype
        self.compressed = compressed
        
        self.buffer: List[Dict[str, np.ndarray]] = []
        self.shard_id = 0
        self.shard_metadata: List[Dict[str, Any]] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._exp_M = None
        self._exp_n_targets = None
    
    def add_trajectory(
        self,
        x0_log: np.ndarray,
        globals_vec: np.ndarray,
        t_vec: np.ndarray,
        y_mat_log: np.ndarray
    ) -> None:
        """Add trajectory to buffer, writing shard when full."""
        # Validate
        if x0_log.ndim != 1 or x0_log.shape[0] != self.n_species:
            raise ValueError(f"x0_log shape {x0_log.shape} must be ({self.n_species},)")
        
        if globals_vec.ndim != 1 or globals_vec.shape[0] != self.n_globals:
            raise ValueError(f"globals shape {globals_vec.shape} must be ({self.n_globals},)")
        
        if t_vec.ndim != 1:
            raise ValueError(f"t_vec must be 1D, got shape {t_vec.shape}")
        
        if y_mat_log.ndim != 2:
            raise ValueError(f"y_mat must be 2D [M, n_targets], got shape {y_mat_log.shape}")
        
        M, n_targets = y_mat_log.shape
        if t_vec.shape[0] != M:
            raise ValueError(f"t_vec length {t_vec.shape[0]} must equal y_mat rows {M}")
        
        # Enforce consistency
        if self._exp_M is None:
            self._exp_M = M
            self._exp_n_targets = n_targets
        else:
            if M != self._exp_M or n_targets != self._exp_n_targets:
                raise ValueError(
                    f"Shape mismatch in shard {self.shard_id}: "
                    f"expected (M={self._exp_M}, n_targets={self._exp_n_targets}), "
                    f"got (M={M}, n_targets={n_targets})"
                )
        
        self.buffer.append({
            "x0_log": x0_log.astype(self.dtype, copy=False),
            "globals": globals_vec.astype(self.dtype, copy=False),
            "t_vec": t_vec.astype(self.dtype, copy=False),
            "y_mat": y_mat_log.astype(self.dtype, copy=False),
        })
        
        if len(self.buffer) >= self.trajectories_per_shard:
            self._write_shard()
    
    def _write_shard(self) -> None:
        """Write buffered trajectories to NPZ file."""
        if not self.buffer:
            return
        
        # Stack all trajectories
        x0_log_arr = np.stack([t["x0_log"] for t in self.buffer])
        globals_arr = np.stack([t["globals"] for t in self.buffer])
        t_vec_arr = np.stack([t["t_vec"] for t in self.buffer])
        y_mat_arr = np.stack([t["y_mat"] for t in self.buffer])
        
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        
        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(
            filepath,
            x0_log=x0_log_arr.astype(self.dtype),
            globals=globals_arr.astype(self.dtype),
            t_vec=t_vec_arr.astype(self.dtype),
            y_mat=y_mat_arr.astype(self.dtype),
        )
        
        self.shard_metadata.append({
            "filename": filename,
            "n_trajectories": len(self.buffer),
        })
        
        self.buffer = []
        self.shard_id += 1
    
    def flush(self) -> None:
        """Write any remaining trajectories in buffer."""
        if self.buffer:
            self._write_shard()


# ============================================================================
# SINGLE-PASS PREPROCESSOR
# ============================================================================

class SinglePassPreprocessor:
    """Optimized single-pass preprocessor that collects stats while writing shards.""" 
    def __init__(
        self,
        config: Dict[str, Any],
        stats_logger: DataStatisticsLogger,
    ):
        """Initialize preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stats_logger = stats_logger
        
        # Extract config
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]
        
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup preprocessing parameters."""
        # Chunking
        self.chunk_size = int(self.proc_cfg.get("hdf5_chunk_size", DEFAULT_CHUNK_SIZE))
        
        # Data type
        dtype_str = self.system_cfg.get("dtype", "float32")
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32
        
        # Variables
        self.species_vars = list(self.data_cfg["species_variables"])
        self.target_species_vars = list(
            self.data_cfg.get("target_species_variables", self.species_vars)
        )
        self.global_vars = list(self.data_cfg["global_variables"])
        self.time_var = str(self.data_cfg.get("time_variable", "t_time"))
        
        # Dimensions
        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        
        # Thresholds
        self.min_value_threshold = float(
            self.proc_cfg.get("min_value_threshold", DEFAULT_MIN_VALUE_THRESHOLD)
        )
        self.epsilon = float(self.norm_cfg.get("epsilon", DEFAULT_EPSILON))
        self.log_floor = max(self.min_value_threshold, self.epsilon)
        
        # Trajectories per shard
        self.trajectories_per_shard = int(self.proc_cfg["trajectories_per_shard"])
        
        # Normalization methods
        default_method = self.norm_cfg.get("default_method", "log-standard")
        methods_override = self.norm_cfg.get("methods", {})
        self.norm_methods = {}
        
        for var in self.species_vars + self.target_species_vars + self.global_vars + [self.time_var]:
            self.norm_methods[var] = methods_override.get(var, default_method)

        # Split fractions (validated once)
        self.test_frac = float(self.train_cfg.get("test_fraction", 0.0))
        self.val_frac = float(self.train_cfg.get("val_fraction", 0.0))
        if not (0.0 <= self.test_frac <= 1.0 and 0.0 <= self.val_frac <= 1.0):
            raise ValueError("test_fraction and val_fraction must be in [0, 1].")
        if self.test_frac + self.val_frac > 1.0:
            raise ValueError("test_fraction + val_fraction must be \u2264 1.")
        
        self.use_fraction = float(self.train_cfg.get("use_fraction", 1.0))
        if not (0.0 <= self.use_fraction <= 1.0):
            raise ValueError("use_fraction must be in [0, 1].")
    
    def _fast_hash(self, s: str, seed: int) -> float:
        """Stable 64-bit hash in [0,1), independent of PYTHONHASHSEED."""
        b = f"{seed}:{s}".encode("utf-8")
        h = hashlib.sha256(b).digest()
        u64 = int.from_bytes(h[:8], "big", signed=False)
        return u64 / float(1 << 64)
    
    def _should_use_trajectory(self, gname: str, seed: int) -> bool:
        """Check if trajectory should be used based on validated use_fraction."""
        if self.use_fraction >= 1.0:
            return True
        return self._fast_hash(f"{gname}:use", seed) < self.use_fraction
    
    def _determine_split(self, gname: str, seed: int) -> str:
        """Determine train/val/test split for trajectory (validated in _setup_parameters)."""
        split_hash = self._fast_hash(f"{gname}:split", seed)
        if split_hash < self.test_frac:
            return "test"
        elif split_hash < self.test_frac + self.val_frac:
            return "validation"
        else:
            return "train"
    
    def process_file(
        self,
        file_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process single HDF5 file with single-pass stats collection and shard writing."""
        start_time = time.time()

        # Snapshot global counters to compute per-file deltas
        groups_before = self.stats_logger.stats.total_groups
        valid_before = self.stats_logger.stats.valid_trajectories

        # Generate unique shard prefix
        hname = hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()[:12]
        shard_base_name = f"{file_path.stem}_{hname}"
        
        # Initialize shard writers
        compressed_npz = bool(self.proc_cfg.get("npz_compressed", True))
        writers = {}
        for split in ("train", "validation", "test"):
            writers[split] = SequenceShardWriter(
                output_dir / split,
                self.trajectories_per_shard,
                shard_base_name,
                self.n_species,
                self.n_globals,
                self.np_dtype,
                compressed=compressed_npz,
            )
        
        # Process trajectories
        split_counts = {"train": 0, "validation": 0, "test": 0}
        reader = OptimizedHDF5Reader(self.chunk_size)
        seed = int(self.system_cfg.get("seed", 42))
        
        local_expected_M = None
        
        with h5py.File(file_path, "r") as f:
            keys = sorted(f.keys())
            for gname in tqdm(keys, desc=f"Processing {file_path.name}", unit="traj"):
                grp = f[gname]
                self.stats_logger.stats.total_groups += 1
                
                # Apply use_fraction filter
                if not self._should_use_trajectory(gname, seed):
                    self.stats_logger.stats.dropped_use_fraction += 1
                    continue
                
                # Extract and validate trajectory
                result = self._extract_trajectory(grp, gname, reader)
                if result is None:
                    reader.clear_cache()
                    continue
                
                time_data, x0, globals_vec, species_mat = result
                
                # Check M consistency
                M = int(time_data.shape[0])
                if local_expected_M is None:
                    local_expected_M = M
                elif M != local_expected_M:
                    raise ValueError(f"Group {gname}: time length {M} != expected {local_expected_M}")
                
                self.stats_logger.stats.valid_trajectories += 1

                # Count time stats only for validated trajectories
                self.stats_logger.update_time_range(time_data)

                # Ranges: only from validated trajectories
                for j, var in enumerate(self.target_species_vars):
                    self.stats_logger.update_species_range(var, species_mat[:, j])
                for i, var in enumerate(self.global_vars):
                    self.stats_logger.update_global_range(var, float(globals_vec[i]))
                
                # Update streaming statistics for normalization (validated only)
                self._update_statistics(time_data, x0, globals_vec, species_mat)
                
                # Apply log transformation
                x0_log = np.log10(np.maximum(x0, self.log_floor))
                y_mat_log = np.log10(np.maximum(species_mat, self.log_floor))
                
                # Determine split and write
                split = self._determine_split(gname, seed)
                writers[split].add_trajectory(
                    x0_log.astype(self.np_dtype),
                    globals_vec.astype(self.np_dtype),
                    time_data.astype(self.np_dtype),
                    y_mat_log.astype(self.np_dtype)
                )
                split_counts[split] += 1
                self.stats_logger.stats.split_distribution[split] += 1
                
                # Clear reader cache periodically
                reader.clear_cache()
        
        # Flush all writers
        for w in writers.values():
            w.flush()
        
        # Record processing time
        elapsed = time.time() - start_time
        self.stats_logger.stats.processing_times[str(file_path)] = elapsed
        self.stats_logger.stats.file_stats[str(file_path)] = {
            "groups_processed": self.stats_logger.stats.total_groups - groups_before,
            "valid_trajectories": self.stats_logger.stats.valid_trajectories - valid_before,
            "processing_time": elapsed,
        }
        
        # Build metadata
        metadata = {
            "expected_M": int(local_expected_M) if local_expected_M is not None else 0,
            "splits": {}
        }
        
        for split in ("train", "validation", "test"):
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
            }
        
        return metadata
    
    def _extract_trajectory(
        self,
        group: h5py.Group,
        gname: str,
        reader: OptimizedHDF5Reader
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Extract and validate trajectory data efficiently."""
        # Check required globals
        missing = [k for k in self.global_vars if k not in group.attrs]
        if missing:
            self.logger.debug(f"Group {gname}: Missing globals {missing}")
            self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Check time variable
        if self.time_var not in group:
            self.logger.debug(f"Group {gname}: Missing time variable {self.time_var}")
            self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Load time data
        time_data = reader.read_dataset(group, self.time_var)
        if time_data.size == 0:
            self.logger.debug(f"Group {gname}: Empty time array")
            self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        if not np.all(np.isfinite(time_data)):
            self.logger.debug(f"Group {gname}: Non-finite time values")
            self.stats_logger.stats.dropped_non_finite += 1
            return None
        
        if np.any(np.diff(time_data) <= 0):
            self.logger.debug(f"Group {gname}: Non-monotonic time array")
            self.stats_logger.stats.dropped_non_finite += 1
            return None
                
        n_times = time_data.shape[0]
        
        # Build species data efficiently
        x0 = np.empty(self.n_species, dtype=self.np_dtype)
        species_mat = np.empty((n_times, self.n_target_species), dtype=self.np_dtype)
        
        # Cache for species data that appear in both x0 and targets
        species_cache = {}
        
        # Process target species (most will also be in x0)
        for j, var in enumerate(self.target_species_vars):
            if var not in group:
                self.logger.debug(f"Group {gname}: Missing target species {var}")
                self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            # Read once and cache
            arr = reader.read_dataset(group, var, use_cache=True)
            species_cache[var] = arr
            
            if arr.shape[0] != n_times:
                self.logger.debug(f"Group {gname}: Length mismatch for {var}")
                self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            if not np.all(np.isfinite(arr)):
                self.logger.debug(f"Group {gname}: Non-finite values in {var}")
                self.stats_logger.stats.dropped_non_finite += 1
                return None
            
            if np.any(arr <= self.min_value_threshold):
                self.logger.debug(f"Group {gname}: Values <= threshold in {var}")
                self.stats_logger.stats.dropped_below_threshold += 1
                return None
            
            species_mat[:, j] = arr
        
        # Extract initial conditions, reusing cached data
        for i, var in enumerate(self.species_vars):
            if var in species_cache:
                # Reuse cached data
                v0 = float(species_cache[var][0])
            else:
                # Species not in targets, only need first value
                if var not in group:
                    self.logger.debug(f"Group {gname}: Missing species {var}")
                    self.stats_logger.stats.dropped_missing_keys += 1
                    return None

                dset = group[var]
                # Ensure non-empty dataset before indexing [0]
                if dset.shape[0] == 0:
                    self.logger.debug(f"Group {gname}: Empty dataset for {var}")
                    self.stats_logger.stats.dropped_missing_keys += 1
                    return None

                # Only read first value for efficiency
                v0 = float(dset[0])

                if not np.isfinite(v0):
                    self.logger.debug(f"Group {gname}: Non-finite initial value for {var}")
                    self.stats_logger.stats.dropped_non_finite += 1
                    return None

                # Enforce threshold on initial condition for non-target species
                if v0 <= self.min_value_threshold:
                    self.logger.debug(f"Group {gname}: Initial {var} <= threshold: {v0}")
                    self.stats_logger.stats.dropped_below_threshold += 1
                    return None

            x0[i] = v0
        
        # Extract globals
        globals_vec = np.empty(self.n_globals, dtype=self.np_dtype)
        for i, var in enumerate(self.global_vars):
            value = float(group.attrs[var])
            if not np.isfinite(value):
                self.logger.debug(f"Group {gname}: Non-finite global {var}")
                self.stats_logger.stats.dropped_non_finite += 1
                return None
            globals_vec[i] = value
            
    
        for j, var in enumerate(self.target_species_vars):
            self.stats_logger.update_species_range(var, species_mat[:, j])

        for i, var in enumerate(self.global_vars):
            self.stats_logger.update_global_range(var, float(globals_vec[i]))

        return time_data, x0, globals_vec, species_mat
    
    def _update_statistics(
        self,
        time_data: np.ndarray,
        x0: np.ndarray,
        globals_vec: np.ndarray,
        species_mat: np.ndarray
    ) -> None:
        """Update streaming statistics for normalization."""
        # Update time stats
        self.stats_logger.update_accumulator(
            self.time_var,
            time_data,
            self.log_floor,
            self.norm_methods[self.time_var]
        )
        
        # Update species stats
        for i, var in enumerate(self.species_vars):
            if var in self.target_species_vars:
                # Use full evolution data
                idx = self.target_species_vars.index(var)
                data = species_mat[:, idx]
            else:
                # Only initial condition
                data = np.array([x0[i]])
            
            self.stats_logger.update_accumulator(
                var,
                data,
                self.log_floor,
                self.norm_methods[var]
            )
        
        # Update global stats
        for i, var in enumerate(self.global_vars):
            self.stats_logger.update_accumulator(
                var,
                np.array([globals_vec[i]]),
                self.log_floor,
                self.norm_methods[var]
            )

def _process_file_worker(
    file_path_str: str,
    output_dir_str: str,
    config_pickle: bytes,
) -> Dict[str, Any]:
    """Worker function for parallel processing using pickle for efficiency."""
    import pickle
    
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    config = pickle.loads(config_pickle)
    
    # Create local stats logger
    stats_logger = DataStatisticsLogger(output_dir)
    processor = SinglePassPreprocessor(config, stats_logger)
    
    # Process file
    meta = processor.process_file(file_path, output_dir)
    
    # Attach stats for merging
    meta["_worker_stats"] = asdict(stats_logger.stats)
    meta["_worker_accumulators"] = {
        k: asdict(v) for k, v in stats_logger.accumulators.items()
    }
    
    return meta


class DataPreprocessor:
    """Optimized main preprocessor with single-pass processing."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """Initialize main preprocessor."""
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_logger = DataStatisticsLogger(output_dir)
    
    def process_to_npy_shards(self) -> None:
        """Execute optimized single-pass preprocessing pipeline."""
        self.logger.info("Starting optimized data preprocessing")
        
        # Process files (collects stats during processing)
        all_metadata = self._process_files()
        
        # Finalize normalization stats from accumulated data
        norm_stats = self.stats_logger.finalize_normalization_stats(self.config)
        
        # Save normalization stats
        save_json(norm_stats, self.output_dir / "normalization.json")
        self.logger.info(f"Saved normalization statistics")
        
        # Save shard index
        self._save_shard_index(all_metadata, norm_stats)
        
        # Save statistics summary
        self.stats_logger.save_summary()
        
        # Log summary
        self.logger.info(
            f"Preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )
    
    def _process_files(self) -> Dict[str, Any]:
        """Process all files with single-pass stats collection."""
        all_metadata = {
            "expected_M": None,
            "splits": {
                "train": {"shards": [], "n_trajectories": 0},
                "validation": {"shards": [], "n_trajectories": 0},
                "test": {"shards": [], "n_trajectories": 0},
            }
        }
        
        num_workers = int(self.config.get("preprocessing", {}).get("num_workers", 0))
        
        if num_workers > 0:
            # Parallel processing with pickle for efficiency
            import pickle
            self.logger.info(f"Parallel preprocessing with {num_workers} workers")
            cfg_pickle = pickle.dumps(self.config)
            
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn")
            ) as ex:
                futures = [
                    ex.submit(
                        _process_file_worker,
                        str(file_path),
                        str(self.processed_dir),
                        cfg_pickle,
                    )
                    for file_path in self.raw_files
                ]
                
                with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
                    for fut in as_completed(futures):
                        metadata = fut.result()
                        self._merge_metadata(all_metadata, metadata)
                        
                        # Merge worker stats
                        if "_worker_stats" in metadata:
                            worker_stats = DataStatistics(**metadata["_worker_stats"])
                            self.stats_logger.stats.merge_inplace(worker_stats)
                        
                        # Merge accumulators
                        if "_worker_accumulators" in metadata:
                            for var, acc_dict in metadata["_worker_accumulators"].items():
                                if var not in self.stats_logger.accumulators:
                                    self.stats_logger.accumulators[var] = StreamingStatistics()
                                # Merge accumulator stats
                                self._merge_accumulators(
                                    self.stats_logger.accumulators[var],
                                    StreamingStatistics(**acc_dict)
                                )
                        
                        pbar.update(1)
        else:
            # Serial processing
            for file_path in tqdm(self.raw_files, desc="Processing files", unit="file"):
                self.logger.info(f"Processing file: {file_path}")
                processor = SinglePassPreprocessor(self.config, self.stats_logger)
                metadata = processor.process_file(file_path, self.processed_dir)
                self._merge_metadata(all_metadata, metadata)
        
        return all_metadata
    
    def _merge_accumulators(self, acc1: StreamingStatistics, acc2: StreamingStatistics) -> None:
        """Merge two streaming statistics accumulators."""
        if acc2.count == 0:
            return
        if acc1.count == 0:
            acc1.count = acc2.count
            acc1.mean = acc2.mean
            acc1.m2 = acc2.m2
            acc1.min_val = acc2.min_val
            acc1.max_val = acc2.max_val
            acc1.log_min = acc2.log_min
            acc1.log_max = acc2.log_max
        else:
            # Merge using parallel algorithm
            n1, n2 = acc1.count, acc2.count
            n = n1 + n2
            delta = acc2.mean - acc1.mean
            acc1.mean = acc1.mean + delta * (n2 / n)
            acc1.m2 = acc1.m2 + acc2.m2 + (delta * delta) * (n1 * n2 / n)
            acc1.count = n
            acc1.min_val = min(acc1.min_val, acc2.min_val)
            acc1.max_val = max(acc1.max_val, acc2.max_val)
            acc1.log_min = min(acc1.log_min, acc2.log_min)
            acc1.log_max = max(acc1.log_max, acc2.log_max)
    
    def _save_shard_index(
        self,
        all_metadata: Dict[str, Any],
        norm_stats: Dict[str, Any]
    ) -> None:
        """Save shard index with all metadata."""
        if not all_metadata.get("expected_M"):
            raise ValueError("Could not infer M_per_sample; no valid trajectories processed.")
        
        shard_index = {
            "sequence_mode": True,
            "variable_length": False,
            "M_per_sample": int(all_metadata["expected_M"]),
            "n_input_species": len(self.config["data"]["species_variables"]),
            "n_target_species": len(
                self.config["data"].get(
                    "target_species_variables",
                    self.config["data"]["species_variables"]
                )
            ),
            "n_globals": len(self.config["data"]["global_variables"]),
            "compression": "npz",
            "splits": all_metadata["splits"],
            "time_normalization": norm_stats.get("time_normalization", {}),
            "already_logged_vars": sorted(set(
                self.config["data"]["species_variables"]
            ) | set(self.config["data"].get(
                "target_species_variables",
                self.config["data"]["species_variables"]
            ))),
        }
        save_json(shard_index, self.output_dir / "shard_index.json")
        self.logger.info(f"Saved shard index")
    
    @staticmethod
    def _merge_metadata(all_meta: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Merge file metadata into combined metadata."""
        for split in ("train", "validation", "test"):
            all_meta["splits"][split]["shards"].extend(meta["splits"][split]["shards"])
            all_meta["splits"][split]["n_trajectories"] += meta["splits"][split]["n_trajectories"]
        
        # Validate M consistency
        m = meta.get("expected_M", None)
        if m:
            m = int(m)
            if all_meta.get("expected_M") is None:
                all_meta["expected_M"] = m
            elif all_meta["expected_M"] != m:
                raise ValueError(
                    f"M_per_sample mismatch across files: {all_meta['expected_M']} vs {m}"
                )