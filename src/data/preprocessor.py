#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor for LiLaN.

This module preprocesses raw HDF5 trajectory data for training LiLaN models:

1. Data Loading and Validation:
   - Loads trajectories from HDF5 files
   - Validates data integrity (no NaNs, finite values, positive concentrations)
   - Checks for monotonic time arrays
   - Drops trajectories with any invalid data points

2. Data Transformation:
   - Applies log10 transformation to species concentrations
   - Preserves original time grids from HDF5 (no interpolation)
   - Maintains trajectory structure for sequence learning

3. Normalization Statistics:
   - Collects per-variable statistics (mean, std, min, max)
   - Supports configurable normalization methods per variable
   - Saves statistics for use during training/inference

4. Data Sharding:
   - Splits data into train/validation/test sets
   - Creates efficient NPZ shards for fast loading
   - Supports parallel processing for large datasets

Key Features:
- No interpolation - preserves original simulation grids
- Strict validation - drops entire trajectory on any bad value
- Configurable normalization per variable
- Efficient sharding for large-scale training
- Comprehensive logging and statistics
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from utils.utils import save_json


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
    dropped_non_monotonic_time: int = 0
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
    
    def merge_inplace(self, other: DataStatistics) -> None:
        """Merge another statistics object into this one."""
        # Merge counts
        self.total_groups += other.total_groups
        self.valid_trajectories += other.valid_trajectories
        self.dropped_missing_keys += other.dropped_missing_keys
        self.dropped_non_finite += other.dropped_non_finite
        self.dropped_below_threshold += other.dropped_below_threshold
        self.dropped_non_monotonic_time += other.dropped_non_monotonic_time
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
        
        # Merge distributions and metadata
        for split, cnt in other.split_distribution.items():
            self.split_distribution[split] = self.split_distribution.get(split, 0) + cnt
        
        self.processing_times.update(other.processing_times)
        self.file_stats.update(other.file_stats)


class DataStatisticsLogger:
    """Logger for comprehensive data statistics during preprocessing."""
    
    def __init__(self, output_dir: Path):
        """Initialize statistics logger."""
        self.output_dir = output_dir
        self.stats = DataStatistics()
        self.logger = logging.getLogger(__name__)
    
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
                "non_monotonic_time": self.stats.dropped_non_monotonic_time,
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
# HDF5 READING UTILITIES
# ============================================================================

class ChunkedHDF5Reader:
    """
    Memory-efficient HDF5 reader with chunked loading.
    
    Reads large datasets in chunks to minimize memory usage.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize chunked reader.
        
        Args:
            chunk_size: Maximum chunk size for reading
        """
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def read_dataset_chunked(
        self,
        group: h5py.Group,
        var_name: str,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Read dataset in chunks to minimize memory usage.
        
        Args:
            group: HDF5 group containing the dataset
            var_name: Variable name to read
            indices: Optional specific indices to read
            
        Returns:
            Concatenated array from all chunks
            
        Raises:
            KeyError: If variable not found in group
        """
        if var_name not in group:
            raise KeyError(f"Variable {var_name} not found in group")
        
        dset = group[var_name]
        
        if indices is not None:
            return dset[indices]
        
        total = dset.shape[0]
        if total <= self.chunk_size:
            return dset[:]
        
        # Read in chunks and concatenate
        chunks = []
        for i in range(0, total, self.chunk_size):
            end = min(i + self.chunk_size, total)
            chunks.append(dset[i:end])
        
        return np.concatenate(chunks)


# ============================================================================
# DATA WRITING UTILITIES
# ============================================================================

class SequenceShardWriter:
    """
    Efficient writer for trajectory sequence shards.
    
    Buffers trajectories and writes them in batches to NPZ files.
    """
    
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
        """
        Initialize shard writer.
        
        Args:
            output_dir: Directory to write shards
            trajectories_per_shard: Number of trajectories per shard file
            shard_idx_base: Base name for shard files
            n_species: Number of species variables
            n_globals: Number of global variables
            dtype: Data type for arrays
            compressed: Whether to use compression
        """
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
    
    def add_trajectory(
        self,
        x0_log: np.ndarray,
        globals_vec: np.ndarray,
        t_vec: np.ndarray,
        y_mat_log: np.ndarray
    ) -> None:
        """
        Add trajectory to buffer, writing shard when full.
        
        Args:
            x0_log: Log-transformed initial conditions [n_species]
            globals_vec: Global parameters [n_globals]
            t_vec: Time points [n_timepoints]
            y_mat_log: Log-transformed species evolution [n_timepoints, n_species]
        """
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
        
        # Create lists to handle variable-length trajectories
        x0_log_list = [t["x0_log"] for t in self.buffer]
        globals_list = [t["globals"] for t in self.buffer]
        t_vec_list = [t["t_vec"] for t in self.buffer]
        y_mat_list = [t["y_mat"] for t in self.buffer]
        
        # Write to file with object arrays for variable-length data
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        
        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(
            filepath,
            x0_log=np.array(x0_log_list, dtype=object),
            globals=np.array(globals_list, dtype=object),
            t_vec=np.array(t_vec_list, dtype=object),
            y_mat=np.array(y_mat_list, dtype=object),
        )
        
        # Record metadata
        self.shard_metadata.append({
            "filename": filename,
            "n_trajectories": len(self.buffer),
            "time_points": [len(t["t_vec"]) for t in self.buffer]
        })
        
        # Clear buffer for next shard
        self.buffer = []
        self.shard_id += 1
    
    def flush(self) -> None:
        """Write any remaining trajectories in buffer."""
        if self.buffer:
            self._write_shard()


# ============================================================================
# CORE PREPROCESSING LOGIC
# ============================================================================

class CorePreprocessor:
    """
    Core preprocessing logic for chemical kinetics data.
    
    Handles trajectory extraction, validation, and transformation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        stats_logger: Optional[DataStatisticsLogger] = None,
    ):
        """
        Initialize core preprocessor.
        
        Args:
            config: Configuration dictionary
            stats_logger: Optional statistics logger
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stats_logger = stats_logger
        
        # Extract config sections
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]
        
        # Setup parameters
        self._setup_parameters()
    
    def _setup_parameters(self) -> None:
        """Setup preprocessing parameters from config."""
        # Chunking
        self.chunk_size = int(self.proc_cfg.get("hdf5_chunk_size", 10000))
        
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
        self.min_value_threshold = float(self.proc_cfg.get("min_value_threshold", 1e-30))
        self.epsilon = float(self.norm_cfg.get("epsilon", 1e-30))
    
    def _extract_and_validate_trajectory(
        self,
        group: h5py.Group,
        gname: str,
        reader: ChunkedHDF5Reader
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract and validate trajectory data from HDF5 group.
        
        Performs comprehensive validation:
        - Checks for missing variables
        - Validates finite values (no NaNs or Infs)
        - Ensures positive concentrations above threshold
        - Verifies monotonic time array
        
        Args:
            group: HDF5 group containing trajectory
            gname: Group name (for logging)
            reader: Chunked reader instance
            
        Returns:
            Tuple of (time_data, x0, globals_vec, species_mat) or None if invalid
        """
        # Check for required global variables
        missing = [k for k in self.global_vars if k not in group.attrs]
        if missing:
            self.logger.debug(f"Group {gname}: Missing global variables {missing}")
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Check time variable exists
        if self.time_var not in group:
            self.logger.debug(f"Group {gname}: Missing time variable {self.time_var}")
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Load and validate time data
        time_data = reader.read_dataset_chunked(group, self.time_var)
        if time_data.size == 0:
            self.logger.debug(f"Group {gname}: Empty time array")
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Check for finite time values
        if not np.all(np.isfinite(time_data)):
            self.logger.debug(f"Group {gname}: Non-finite time values")
            if self.stats_logger:
                self.stats_logger.stats.dropped_non_finite += 1
            return None
        
        # Check for monotonic time (strictly increasing)
        if np.any(np.diff(time_data) <= 0):
            self.logger.error(f"Group {gname}: Non-monotonic time array detected!")
            if self.stats_logger:
                self.stats_logger.stats.dropped_non_monotonic_time += 1
            raise ValueError(f"Non-monotonic time array in group {gname}. Time must be strictly increasing.")
        
        if self.stats_logger:
            self.stats_logger.update_time_range(time_data)
        
        # Extract initial conditions (x0)
        x0 = np.empty(self.n_species, dtype=self.np_dtype)
        for i, var in enumerate(self.species_vars):
            if var not in group:
                self.logger.debug(f"Group {gname}: Missing species {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            # Get initial value
            v0 = float(group[var][0])
            
            # Check finite and above threshold
            if not np.isfinite(v0):
                self.logger.debug(f"Group {gname}: Non-finite initial value for {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_non_finite += 1
                return None
            
            if v0 <= self.min_value_threshold:
                self.logger.debug(f"Group {gname}: Initial {var} below threshold: {v0}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None
            
            x0[i] = v0
        
        # Extract full species time series
        n_times = time_data.shape[0]
        species_mat = np.empty((n_times, self.n_target_species), dtype=self.np_dtype)
        
        for j, var in enumerate(self.target_species_vars):
            if var not in group:
                self.logger.debug(f"Group {gname}: Missing target species {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            # Read full time series
            arr = reader.read_dataset_chunked(group, var)
            
            # Validate length matches time array
            if arr.shape[0] != n_times:
                self.logger.debug(f"Group {gname}: Length mismatch for {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            # Check for any non-finite values
            if not np.all(np.isfinite(arr)):
                self.logger.debug(f"Group {gname}: Non-finite values in {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_non_finite += 1
                return None
            
            # Check all values above threshold
            if np.any(arr <= self.min_value_threshold):
                self.logger.debug(f"Group {gname}: Values below threshold in {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None
            
            species_mat[:, j] = arr
            
            if self.stats_logger:
                self.stats_logger.update_species_range(var, arr)
        
        # Extract global variables
        globals_vec = np.empty(self.n_globals, dtype=self.np_dtype)
        for i, var in enumerate(self.global_vars):
            value = float(group.attrs[var])
            
            if not np.isfinite(value):
                self.logger.debug(f"Group {gname}: Non-finite global {var}")
                if self.stats_logger:
                    self.stats_logger.stats.dropped_non_finite += 1
                return None
            
            globals_vec[i] = value
            
            if self.stats_logger:
                self.stats_logger.update_global_range(var, value)
        
        return time_data, x0, globals_vec, species_mat
    
    def process_file_for_shards(
        self,
        file_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process single HDF5 file and write trajectory shards.
        
        Args:
            file_path: Path to input HDF5 file
            output_dir: Directory for output shards
            
        Returns:
            Metadata dictionary with shard information
        """
        start_time = time.time()
        
        # Calculate shard size
        trajectories_per_shard = self._calculate_shard_size()
        compressed_npz = bool(self.proc_cfg.get("npz_compressed", True))
        
        # Initialize shard writers for each split
        writers = self._init_shard_writers(
            output_dir, file_path.stem, trajectories_per_shard, compressed_npz
        )
        
        # Process trajectories
        split_counts = {"train": 0, "validation": 0, "test": 0}
        reader = ChunkedHDF5Reader(self.chunk_size)
        seed = int(self.system_cfg.get("seed", 42))
        
        # Local statistics
        local_stats = DataStatistics()
        
        with h5py.File(file_path, "r") as f:
            keys = sorted(f.keys())
            for gname in tqdm(keys, desc=f"Processing {file_path.name}", unit="traj"):
                grp = f[gname]
                local_stats.total_groups += 1
                if self.stats_logger:
                    self.stats_logger.stats.total_groups += 1
                
                # Apply use_fraction filter
                if not self._should_use_trajectory(gname, seed):
                    local_stats.dropped_use_fraction += 1
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_use_fraction += 1
                    continue
                
                # Extract and validate trajectory
                extracted = self._extract_and_validate_trajectory(grp, gname, reader)
                if extracted is None:
                    continue
                
                time_data, x0, globals_vec, species_mat = extracted
                
                # Valid trajectory - increment counter
                local_stats.valid_trajectories += 1
                if self.stats_logger:
                    self.stats_logger.stats.valid_trajectories += 1
                
                # Apply log transformation to species data
                x0_log = np.log10(np.maximum(x0, self.epsilon))
                y_mat_log = np.log10(np.maximum(species_mat, self.epsilon))
                
                # Determine split
                split = self._determine_split(gname, seed)
                
                # Add to appropriate writer
                writers[split].add_trajectory(
                    x0_log.astype(self.np_dtype),
                    globals_vec.astype(self.np_dtype),
                    time_data.astype(self.np_dtype),
                    y_mat_log.astype(self.np_dtype)
                )
                split_counts[split] += 1
                
                local_stats.split_distribution[split] += 1
                if self.stats_logger:
                    self.stats_logger.stats.split_distribution[split] += 1
        
        # Flush all writers
        for w in writers.values():
            w.flush()
        
        # Record processing time
        elapsed = time.time() - start_time
        local_stats.processing_times[str(file_path)] = elapsed
        local_stats.file_stats[str(file_path)] = {
            "groups_processed": local_stats.total_groups,
            "valid_trajectories": local_stats.valid_trajectories,
            "processing_time": elapsed,
        }
        
        if self.stats_logger:
            self.stats_logger.stats.processing_times[str(file_path)] = elapsed
            self.stats_logger.stats.file_stats[str(file_path)] = local_stats.file_stats[str(file_path)]
        
        # Build metadata
        metadata = self._build_metadata(writers, split_counts)
        metadata["file_stats"] = asdict(local_stats)
        
        return metadata
    
    def _calculate_shard_size(self) -> int:
        """Calculate optimal trajectories per shard based on target file size."""
        trajectories_per_shard = self.proc_cfg.get("trajectories_per_shard", None)
        
        if trajectories_per_shard is None:
            # Use a reasonable default
            trajectories_per_shard = 100
        
        return max(1, int(trajectories_per_shard))
    
    def _init_shard_writers(
        self,
        output_dir: Path,
        file_stem: str,
        trajectories_per_shard: int,
        compressed: bool
    ) -> Dict[str, SequenceShardWriter]:
        """Initialize shard writers for each split."""
        writers = {}
        for split in ("train", "validation", "test"):
            writers[split] = SequenceShardWriter(
                output_dir / split,
                trajectories_per_shard,
                file_stem,
                self.n_species,
                self.n_globals,
                self.np_dtype,
                compressed=compressed,
            )
        return writers
    
    def _should_use_trajectory(self, gname: str, seed: int) -> bool:
        """Check if trajectory should be used based on use_fraction."""
        use_fraction = float(self.train_cfg.get("use_fraction", 1.0))
        if use_fraction >= 1.0:
            return True
        
        # Deterministic hash-based sampling
        hash_val = int(hashlib.sha256(f"{seed}:{gname}:use".encode()).hexdigest()[:8], 16)
        return (hash_val / 0xFFFFFFFF) < use_fraction
    
    def _determine_split(self, gname: str, seed: int) -> str:
        """Determine train/val/test split for trajectory."""
        # Deterministic hash-based splitting
        hash_val = int(hashlib.sha256(f"{seed}:{gname}:split".encode()).hexdigest()[:8], 16)
        split_hash = hash_val / 0xFFFFFFFF
        
        test_frac = float(self.train_cfg.get("test_fraction", 0.0))
        val_frac = float(self.train_cfg.get("val_fraction", 0.0))
        
        if split_hash < test_frac:
            return "test"
        elif split_hash < test_frac + val_frac:
            return "validation"
        else:
            return "train"
    
    def _build_metadata(
        self,
        writers: Dict[str, SequenceShardWriter],
        split_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Build metadata dictionary for processed file."""
        metadata: Dict[str, Any] = {"splits": {}}
        
        for split in ("train", "validation", "test"):
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
            }
        
        return metadata


# ============================================================================
# STATISTICS COLLECTION
# ============================================================================

def _collect_normalization_stats(
    raw_files: List[Path],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Collect normalization statistics for all variables.
    
    Args:
        raw_files: List of HDF5 files to process
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary with per-variable statistics and methods
    """
    logger.info("Collecting normalization statistics...")
    
    stats: Dict[str, Any] = {"per_key_stats": {}, "normalization_methods": {}}
    accumulators: Dict[str, Dict[str, Any]] = {}
    
    # Get configuration
    norm_cfg = config["normalization"]
    default_method = norm_cfg.get("default_method", "log-standard")
    methods_override = norm_cfg.get("methods", {})
    
    # Get all variables
    species_vars = config["data"]["species_variables"]
    global_vars = config["data"]["global_variables"]
    all_vars = species_vars + global_vars
    
    # Initialize accumulators
    for var in all_vars:
        accumulators[var] = {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "log_min": float("inf"),
            "log_max": float("-inf"),
        }
    
    # Add time variable
    time_var = config["data"].get("time_variable", "t_time")
    accumulators[time_var] = {
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "log_min": float("inf"),
        "log_max": float("-inf"),
    }
    
    chunk_size = int(config["preprocessing"].get("hdf5_chunk_size", 10000))
    epsilon = float(norm_cfg.get("epsilon", 1e-30))
    
    # Process all files
    for file_path in tqdm(raw_files, desc="Collecting stats", unit="file"):
        with h5py.File(file_path, "r") as f:
            for gname in f.keys():
                grp = f[gname]
                
                # Process time variable
                if time_var in grp:
                    times = grp[time_var][:]
                    if np.all(np.isfinite(times)) and times.size > 0:
                        acc = accumulators[time_var]
                        acc["count"] += times.size
                        acc["min"] = min(acc["min"], float(times.min()))
                        acc["max"] = max(acc["max"], float(times.max()))
                        
                        # Log statistics for time
                        log_times = np.log10(np.maximum(times, epsilon))
                        acc["log_min"] = min(acc["log_min"], float(log_times.min()))
                        acc["log_max"] = max(acc["log_max"], float(log_times.max()))
                        
                        # Running mean and variance
                        for t in times:
                            delta = t - acc["mean"]
                            acc["mean"] += delta / acc["count"]
                            acc["m2"] += delta * (t - acc["mean"])
                
                # Process species variables
                for var in species_vars:
                    if var not in grp:
                        continue
                    
                    ds = grp[var]
                    for start in range(0, ds.shape[0], chunk_size):
                        end = min(start + chunk_size, ds.shape[0])
                        arr = ds[start:end]
                        
                        # Filter valid values
                        valid = arr[np.isfinite(arr)]
                        if valid.size == 0:
                            continue
                        
                        acc = accumulators[var]
                        var_method = methods_override.get(var, default_method)
                        
                        # Update count and min/max
                        acc["count"] += valid.size
                        acc["min"] = min(acc["min"], float(valid.min()))
                        acc["max"] = max(acc["max"], float(valid.max()))
                        
                        # Log statistics
                        if "log" in var_method or var_method == "log10":
                            log_vals = np.log10(np.maximum(valid, epsilon))
                            acc["log_min"] = min(acc["log_min"], float(log_vals.min()))
                            acc["log_max"] = max(acc["log_max"], float(log_vals.max()))
                            
                            # Use log values for mean/std
                            data_for_stats = log_vals
                        else:
                            data_for_stats = valid
                            log_vals = np.log10(np.maximum(valid, epsilon))
                            acc["log_min"] = min(acc["log_min"], float(log_vals.min()))
                            acc["log_max"] = max(acc["log_max"], float(log_vals.max()))
                        
                        # Update running mean and variance
                        for val in data_for_stats:
                            delta = val - acc["mean"]
                            acc["mean"] += delta / acc["count"]
                            acc["m2"] += delta * (val - acc["mean"])
                
                # Process global variables
                for var in global_vars:
                    if var in grp.attrs:
                        value = float(grp.attrs[var])
                        if not np.isfinite(value):
                            continue
                        
                        acc = accumulators[var]
                        var_method = methods_override.get(var, default_method)
                        
                        acc["count"] += 1
                        acc["min"] = min(acc["min"], value)
                        acc["max"] = max(acc["max"], value)
                        
                        # Log statistics
                        if value > 0:
                            log_val = math.log10(max(value, epsilon))
                            acc["log_min"] = min(acc["log_min"], log_val)
                            acc["log_max"] = max(acc["log_max"], log_val)
                        
                        # Value for statistics
                        if "log" in var_method or var_method == "log10":
                            stat_val = math.log10(max(value, epsilon))
                        else:
                            stat_val = value
                        
                        # Update running mean and variance
                        delta = stat_val - acc["mean"]
                        acc["mean"] += delta / acc["count"]
                        acc["m2"] += delta * (stat_val - acc["mean"])
    
    # Finalize statistics
    for var, acc in accumulators.items():
        if acc["count"] == 0:
            continue
        
        var_method = methods_override.get(var, default_method)
        
        # Special handling for time variable
        if var == time_var:
            var_method = norm_cfg.get("methods", {}).get(time_var, "log-min-max")
        
        stats["normalization_methods"][var] = var_method
        
        # Compute standard deviation
        variance = acc["m2"] / max(1, acc["count"] - 1)
        std = math.sqrt(max(0, variance))
        min_std = float(norm_cfg.get("min_std", 1e-10))
        
        # Build statistics block
        block = {
            "method": var_method,
            "min": float(acc["min"]) if math.isfinite(acc["min"]) else 0.0,
            "max": float(acc["max"]) if math.isfinite(acc["max"]) else 1.0,
            "log_min": float(acc["log_min"]) if math.isfinite(acc["log_min"]) else -30.0,
            "log_max": float(acc["log_max"]) if math.isfinite(acc["log_max"]) else 0.0,
        }
        
        # Add mean/std for methods that need them
        if "standard" in var_method:
            if "log" in var_method:
                block["log_mean"] = float(acc["mean"])
                block["log_std"] = max(std, min_std)
            else:
                block["mean"] = float(acc["mean"])
                block["std"] = max(std, min_std)
        
        stats["per_key_stats"][var] = block
    
    # Mark which variables are stored in log space
    stats["already_logged_vars"] = species_vars
    
    # Add time statistics
    if time_var in accumulators and accumulators[time_var]["count"] > 0:
        acc = accumulators[time_var]
        stats["time_normalization"] = {
            "tmin_raw": float(acc["min"]),
            "tmax_raw": float(acc["max"]),
            "time_transform": var_method,
        }
    
    logger.info("Normalization statistics collection complete.")
    return stats


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def _process_one_file_worker(
    file_path_str: str,
    output_dir_str: str,
    config_json: str,
) -> Dict[str, Any]:
    """
    Worker function for parallel file processing.
    
    Args:
        file_path_str: Path to input file as string
        output_dir_str: Output directory as string
        config_json: Serialized config
        
    Returns:
        Metadata dictionary with processing results
    """
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    config = json.loads(config_json)
    
    # Create local statistics logger
    stats_logger = DataStatisticsLogger(output_dir)
    processor = CorePreprocessor(config, stats_logger)
    
    # Process file
    meta = processor.process_file_for_shards(file_path, output_dir)
    
    # Attach stats for merging
    meta["_worker_stats"] = asdict(stats_logger.stats)
    
    return meta


# ============================================================================
# MAIN PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """
    Main preprocessor orchestrating the full pipeline.
    
    Coordinates:
    1. Normalization statistics collection
    2. Parallel/serial trajectory processing
    3. Metadata generation and saving
    """
    
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """
        Initialize main preprocessor.
        
        Args:
            raw_files: List of input HDF5 files
            output_dir: Directory for processed output
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_logger = DataStatisticsLogger(output_dir)
    
    def process_to_npy_shards(self) -> None:
        """
        Execute the full preprocessing pipeline.
        
        Processes raw HDF5 files into normalized NPZ shards ready for training.
        """
        self.logger.info("Starting data preprocessing for LiLaN")
        
        # Collect normalization statistics
        norm_stats = _collect_normalization_stats(
            self.raw_files, self.config, self.logger
        )
        
        # Save normalization stats
        save_json(norm_stats, self.output_dir / "normalization.json")
        self.logger.info(f"Saved normalization statistics to {self.output_dir / 'normalization.json'}")
        
        # Process files
        all_metadata = self._process_files()
        
        # Save shard index
        self._save_shard_index(all_metadata, norm_stats)
        
        # Save statistics summary
        self.stats_logger.save_summary()
        
        # Log summary
        self.logger.info(
            "Preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )
    
    def _process_files(self) -> Dict[str, Any]:
        """Process all files either in parallel or serial."""
        all_metadata = {
            "splits": {
                "train": {"shards": [], "n_trajectories": 0},
                "validation": {"shards": [], "n_trajectories": 0},
                "test": {"shards": [], "n_trajectories": 0},
            }
        }
        
        num_workers = int(self.config.get("preprocessing", {}).get("num_workers", 0))
        
        if num_workers > 0:
            # Parallel processing
            self.logger.info(f"Parallel preprocessing with {num_workers} workers...")
            cfg_json = json.dumps(self.config)
            
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as ex:
                futures = [
                    ex.submit(
                        _process_one_file_worker,
                        str(file_path),
                        str(self.processed_dir),
                        cfg_json,
                    )
                    for file_path in self.raw_files
                ]
                
                with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
                    for fut in as_completed(futures):
                        metadata = fut.result()
                        self._merge_metadata(all_metadata, metadata)
                        if "_worker_stats" in metadata:
                            worker_stats = DataStatistics(**metadata["_worker_stats"])
                            self.stats_logger.stats.merge_inplace(worker_stats)
                        pbar.update(1)
        else:
            # Serial processing
            for file_path in tqdm(self.raw_files, desc="Processing files", unit="file"):
                self.logger.info(f"Processing file: {file_path}")
                processor = CorePreprocessor(self.config, self.stats_logger)
                metadata = processor.process_file_for_shards(file_path, self.processed_dir)
                self._merge_metadata(all_metadata, metadata)
        
        return all_metadata
    
    def _save_shard_index(
        self,
        all_metadata: Dict[str, Any],
        norm_stats: Dict[str, Any]
    ) -> None:
        """Save shard index with all metadata."""
        shard_index = {
            "variable_length": True,  # Trajectories have different lengths
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
            "already_logged_vars": list(self.config["data"]["species_variables"]),
        }
        save_json(shard_index, self.output_dir / "shard_index.json")
        self.logger.info(f"Saved shard index to {self.output_dir / 'shard_index.json'}")
    
    @staticmethod
    def _merge_metadata(all_meta: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Merge file metadata into combined metadata."""
        for split in ("train", "validation", "test"):
            all_meta["splits"][split]["shards"].extend(meta["splits"][split]["shards"])
            all_meta["splits"][split]["n_trajectories"] += meta["splits"][split]["n_trajectories"]