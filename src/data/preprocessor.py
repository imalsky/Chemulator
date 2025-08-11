#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with sequence mode support for LiLaN.

This module handles the preprocessing pipeline:
1. Loads raw HDF5 trajectory data
2. Applies quality filters and validation
3. Interpolates trajectories onto a fixed global time grid
4. Collects normalization statistics
5. Saves processed data as NPZ shards for efficient loading

Key features:
- Fixed log-spaced time grid shared across all trajectories
- Config-driven normalization methods
- Parallel processing support
- Streaming statistics collection for large datasets
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from utils.utils import save_json


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
    dropped_insufficient_time: int = 0
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
        self.dropped_insufficient_time += other.dropped_insufficient_time
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
                "insufficient_time": self.stats.dropped_insufficient_time,
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


class ChunkedHDF5Reader:
    """
    Memory-efficient HDF5 reader with chunked loading.
    
    Reads large datasets in chunks to minimize memory usage
    during preprocessing.
    """
    
    def __init__(self, file_path: Optional[Path], chunk_size: int = 10000):
        """
        Initialize chunked reader.
        
        Args:
            file_path: Path to HDF5 file
            chunk_size: Maximum chunk size for reading
        """
        self.file_path = file_path
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


class SequenceShardWriter:
    """
    Efficient writer for trajectory sequence shards.
    
    Buffers trajectories and writes them in batches to NPZ files
    for efficient storage and loading.
    """
    
    def __init__(
        self,
        output_dir: Path,
        trajectories_per_shard: int,
        shard_idx_base: str,
        M: int,
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
            M: Number of time points per trajectory
            n_species: Number of species variables
            n_globals: Number of global variables
            dtype: Data type for arrays
            compressed: Whether to use compression
        """
        self.output_dir = output_dir
        self.trajectories_per_shard = max(1, int(trajectories_per_shard))
        self.shard_idx_base = shard_idx_base
        self.M = M
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
        y_mat: np.ndarray
    ) -> None:
        """Add trajectory to buffer, writing shard when full."""
        self.buffer.append({
            "x0_log": x0_log.astype(self.dtype, copy=False),
            "globals": globals_vec.astype(self.dtype, copy=False),
            "t_vec": t_vec.astype(self.dtype, copy=False),
            "y_mat": y_mat.astype(self.dtype, copy=False),
        })
        
        if len(self.buffer) >= self.trajectories_per_shard:
            self._write_shard()
    
    def _write_shard(self) -> None:
        """Write buffered trajectories to NPZ file."""
        if not self.buffer:
            return
        
        # Stack all trajectories
        x0_log = np.stack([t["x0_log"] for t in self.buffer])
        globals_vec = np.stack([t["globals"] for t in self.buffer])
        t_vec = np.stack([t["t_vec"] for t in self.buffer])
        y_mat = np.stack([t["y_mat"] for t in self.buffer])
        
        # Write to file
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        
        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(filepath, x0_log=x0_log, globals=globals_vec, t_vec=t_vec, y_mat=y_mat)
        
        # Record metadata
        self.shard_metadata.append({
            "filename": filename,
            "n_trajectories": len(self.buffer)
        })
        
        # Clear buffer for next shard
        self.buffer = []
        self.shard_id += 1
    
    def flush(self) -> None:
        """Write any remaining trajectories in buffer."""
        if self.buffer:
            self._write_shard()


class CorePreprocessor:
    """
    Core preprocessing logic with fixed global time grid.
    
    Handles trajectory extraction, validation, interpolation,
    and normalization for chemical kinetics data.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        norm_stats: Optional[Dict[str, Any]] = None,
        stats_logger: Optional[DataStatisticsLogger] = None,
    ):
        """
        Initialize core preprocessor.
        
        Args:
            config: Configuration dictionary
            norm_stats: Optional pre-computed normalization statistics
            stats_logger: Optional statistics logger
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stats_logger = stats_logger
        self.norm_stats = norm_stats or {}
        
        # Extract config sections
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]
        
        # Setup parameters
        self._setup_parameters()
        
        # Initialize fixed time grid
        self.fixed_grid = self._init_fixed_time_grid()
        self.fixed_grid_cast = self.fixed_grid.astype(self.np_dtype)
        
        # Pre-compute log grid for interpolation
        eps = float(self.norm_cfg.get("epsilon", 1e-30))
        self.log_tq = np.log10(np.maximum(self.fixed_grid, eps))
    
    def _setup_parameters(self) -> None:
        """Setup preprocessing parameters from config."""
        # Chunking
        self.chunk_size = int(self.proc_cfg.get("hdf5_chunk_size", 10000))
        
        # Sequence mode
        self.M_per_sample = int(self.data_cfg.get("M_per_sample", 16))
        
        # Data type
        dtype_str = self.system_cfg.get("dtype", "float32")
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32
        
        # Variables
        self.species_vars = list(self.data_cfg["species_variables"])
        self.target_species_vars = list(
            self.data_cfg.get("target_species_variables", self.species_vars)
        )
        self.global_vars = list(self.data_cfg["global_variables"])
        self.time_var = str(self.data_cfg["time_variable"])
        
        # Dimensions
        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        
        # Thresholds
        self.min_value_threshold = float(self.proc_cfg.get("min_value_threshold", 1e-30))
    
    def _init_fixed_time_grid(self) -> np.ndarray:
        """
        Initialize global log-spaced time grid.
        
        Returns:
            Fixed time grid with M_per_sample points
        """
        # Get time range from config or stats
        cfg_range = self.data_cfg.get("fixed_time_range", None)
        if cfg_range is not None:
            tmin = float(cfg_range.get("min"))
            tmax = float(cfg_range.get("max"))
        else:
            # Fallback to stats
            tn = self.norm_stats.get("time_normalization", {})
            if "tmin_raw" in tn and "tmax_raw" in tn:
                tmin = float(tn["tmin_raw"])
                tmax = float(tn["tmax_raw"])
            else:
                raise ValueError("Time range not specified in config or stats")
        
        # Ensure positive values for geomspace
        eps = 1e-12 * max(1.0, abs(tmin))
        a = max(tmin, eps)
        b = max(a * (1.0 + 1e-12), tmax)
        
        # Create log-spaced grid with endpoints
        grid = np.geomspace(a, b, int(self.M_per_sample), endpoint=True, dtype=np.float64)
        return grid
    
    def _extract_trajectory_chunked(
        self,
        group: h5py.Group,
        gname: str,
        reader: ChunkedHDF5Reader
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract and validate trajectory data from HDF5 group.
        
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
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        # Load and validate time data
        time_data = reader.read_dataset_chunked(group, self.time_var)
        if not np.all(np.isfinite(time_data)) or time_data.size == 0:
            if self.stats_logger:
                self.stats_logger.stats.dropped_non_finite += 1
            return None
        
        if self.stats_logger:
            self.stats_logger.update_time_range(time_data)
        
        # Extract initial conditions (x0)
        x0 = np.empty(self.n_species, dtype=self.np_dtype)
        for i, var in enumerate(self.species_vars):
            if var not in group:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            v0 = float(group[var][0])
            if not np.isfinite(v0) or v0 <= self.min_value_threshold:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None
            x0[i] = v0
        
        # Extract full species time series
        species_mat = np.empty((time_data.shape[0], self.n_target_species), dtype=self.np_dtype)
        for j, var in enumerate(self.target_species_vars):
            if var not in group:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            
            arr = reader.read_dataset_chunked(group, var)
            if not np.all(np.isfinite(arr)) or np.any(arr <= self.min_value_threshold):
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None
            
            species_mat[:, j] = arr
            if self.stats_logger:
                self.stats_logger.update_species_range(var, arr)
        
        # Extract global variables
        globals_vec = np.array(
            [float(group.attrs[k]) for k in self.global_vars],
            dtype=self.np_dtype
        )
        
        if self.stats_logger:
            for i, var in enumerate(self.global_vars):
                self.stats_logger.update_global_range(var, globals_vec[i])
        
        return time_data, x0, globals_vec, species_mat
    
    def process_file_for_sequence_shards(
        self,
        file_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process single file and write sequence shards.
        
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
        reader = ChunkedHDF5Reader(file_path, self.chunk_size)
        seed = int(self.config["system"].get("seed", 42))
        
        # Local statistics
        local_stats = DataStatistics()
        local_time_points = 0
        
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                grp = f[gname]
                local_stats.total_groups += 1
                if self.stats_logger:
                    self.stats_logger.stats.total_groups += 1
                
                # Basic validation
                if self.time_var not in grp:
                    self._update_drop_stats(local_stats, "dropped_missing_keys")
                    continue
                
                n_t = grp[self.time_var].shape[0]
                local_time_points += int(n_t)
                
                # Need at least 2 points for interpolation
                if n_t < 2:
                    self._update_drop_stats(local_stats, "dropped_insufficient_time")
                    continue
                
                # Apply use_fraction filter
                if not self._should_use_trajectory(gname, seed):
                    self._update_drop_stats(local_stats, "dropped_use_fraction")
                    continue
                
                # Extract trajectory data
                extracted = self._extract_trajectory_chunked(grp, gname, reader)
                if extracted is None:
                    continue
                
                time_data, x0, globals_vec, species_mat = extracted
                
                # Check time coverage
                if not self._check_time_coverage(time_data):
                    self._update_drop_stats(local_stats, "dropped_insufficient_time")
                    continue
                
                # Valid trajectory
                local_stats.valid_trajectories += 1
                if self.stats_logger:
                    self.stats_logger.stats.valid_trajectories += 1
                
                # Interpolate onto fixed grid
                y_mat_log = self._interpolate_to_grid(time_data, species_mat)
                x0_log = np.log10(np.maximum(x0, self.norm_cfg.get("epsilon", 1e-30)))
                x0_log = x0_log.astype(self.np_dtype, copy=False)
                
                # Determine split
                split = self._determine_split(gname, seed)
                
                # Add to appropriate writer
                writers[split].add_trajectory(
                    x0_log, globals_vec, self.fixed_grid_cast, y_mat_log
                )
                split_counts[split] += 1
                
                if self.stats_logger:
                    self.stats_logger.stats.split_distribution[split] += 1
        
        # Flush all writers
        for w in writers.values():
            w.flush()
        
        # Record processing time
        elapsed = time.time() - start_time
        local_stats.total_time_points += local_time_points
        local_stats.processing_times[str(file_path)] = elapsed
        local_stats.file_stats[str(file_path)] = {
            "groups_processed": local_stats.total_groups,
            "valid_trajectories": local_stats.valid_trajectories,
            "processing_time": elapsed,
        }
        
        if self.stats_logger:
            self.stats_logger.stats.processing_times[str(file_path)] = elapsed
            self.stats_logger.stats.file_stats[str(file_path)] = dict(
                local_stats.file_stats[str(file_path)]
            )
        
        # Build metadata
        metadata = self._build_metadata(writers, split_counts)
        metadata["file_stats"] = asdict(local_stats)
        
        return metadata
    
    def _calculate_shard_size(self) -> int:
        """Calculate optimal trajectories per shard."""
        trajectories_per_shard = self.proc_cfg.get("trajectories_per_shard", None)
        
        if trajectories_per_shard is None:
            # Calculate based on target size
            target_bytes = int(self.proc_cfg.get("target_shard_bytes", 200 * 1024 * 1024))
            bytes_per_val = 8 if self.np_dtype == np.float64 else 4
            
            vals_per_traj = (
                self.n_species +  # x0
                self.n_globals +  # globals
                self.M_per_sample +  # time
                self.M_per_sample * self.n_target_species  # targets
            )
            
            traj_bytes = max(bytes_per_val * vals_per_traj, 1)
            trajectories_per_shard = max(1, target_bytes // traj_bytes)
        
        return trajectories_per_shard
    
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
                self.M_per_sample,
                self.n_species,
                self.n_globals,
                self.np_dtype,
                compressed=compressed,
            )
        return writers
    
    def _update_drop_stats(self, stats: DataStatistics, reason: str) -> None:
        """Update drop statistics."""
        setattr(stats, reason, getattr(stats, reason) + 1)
        if self.stats_logger:
            setattr(self.stats_logger.stats, reason, 
                   getattr(self.stats_logger.stats, reason) + 1)
    
    def _should_use_trajectory(self, gname: str, seed: int) -> bool:
        """Check if trajectory should be used based on use_fraction."""
        use_fraction = float(self.train_cfg.get("use_fraction", 1.0))
        if use_fraction >= 1.0:
            return True
        
        # Deterministic hash-based sampling
        hash_val = int(hashlib.sha256(f"{seed}:{gname}:use".encode()).hexdigest()[:8], 16)
        return (hash_val / 0xFFFFFFFF) < use_fraction
    
    def _check_time_coverage(self, time_data: np.ndarray) -> bool:
        """Check if trajectory time span covers the fixed grid."""
        raw_min = float(np.min(time_data))
        raw_max = float(np.max(time_data))
        gmin = float(self.fixed_grid[0])
        gmax = float(self.fixed_grid[-1])
        
        # Relative tolerance for coverage check
        rtol = float(self.proc_cfg.get("grid_coverage_rtol", 1e-6))
        
        return (raw_min <= gmin * (1.0 + rtol)) and (raw_max >= gmax * (1.0 - rtol))
    
    def _interpolate_to_grid(
        self,
        time_data: np.ndarray,
        species_mat: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate species data onto fixed time grid using log-log interpolation.
        
        Args:
            time_data: Original time points
            species_mat: Species concentrations [Nt, n_species]
            
        Returns:
            Interpolated data on fixed grid [M, n_species]
        """
        eps = float(self.norm_cfg.get("epsilon", 1e-30))
        
        # Convert to log space
        log_t = np.log10(np.maximum(time_data, eps))
        log_y = np.log10(np.maximum(species_mat, eps))

        # Enforce strictly increasing and unique times (required by np.interp)
        if not np.all(np.isfinite(log_t)):
            raise ValueError("Non-finite time values encountered after epsilon clamp.")
        if np.any(np.diff(log_t) <= 0):
            raise ValueError("Time array must be strictly increasing with unique values for interpolation.")
        
        # Interpolate each species
        y_mat_log = np.empty((len(self.fixed_grid), self.n_target_species), dtype=self.np_dtype)
        for j in range(self.n_target_species):
            y_mat_log[:, j] = np.interp(self.log_tq, log_t, log_y[:, j])
            y_mat_log[:, j] = y_mat_log[:, j].astype(self.np_dtype, copy=False)
        
        return y_mat_log
    
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
                "total_samples": split_counts[split],
            }
        
        return metadata
    
    def collect_time_stats(self, file_paths: List[Path]) -> Dict[str, float]:
        """
        Collect time statistics using streaming histogram approach.
        
        Computes:
        - tau0: 5th percentile of positive times
        - Time ranges for normalization
        
        Args:
            file_paths: List of HDF5 files to process
            
        Returns:
            Dictionary with time normalization parameters
        """
        if not file_paths:
            raise ValueError("No files provided")
        
        # Pass 1: Find global min/max
        global_min, global_max, total_count = self._find_time_range(file_paths)
        
        if not math.isfinite(global_min) or not math.isfinite(global_max) or total_count == 0:
            raise ValueError("No valid time data found")
        
        if not (global_max > global_min):
            global_max = global_min * (1.0 + 1e-12)
        
        # Pass 2: Build histogram for percentile calculation
        tau0 = self._compute_tau0_percentile(file_paths, global_min, global_max)
        
        # Compute normalization bounds
        tau_min = float(np.log(1.0 + global_min / tau0))
        tau_max = float(np.log(1.0 + global_max / tau0))
        
        # Get configured time method
        time_method = self.config.get("normalization", {}).get("methods", {}).get(
            self.time_var, "log-min-max"
        )
        
        return {
            "tau0": tau0,
            "tmin": tau_min,  # tau-space min
            "tmax": tau_max,  # tau-space max
            "tmin_raw": global_min,  # raw time min
            "tmax_raw": global_max,  # raw time max
            "time_transform": time_method,
        }
    
    def _find_time_range(
        self,
        file_paths: List[Path]
    ) -> Tuple[float, float, int]:
        """Find global time range across all files."""
        global_min = float("inf")
        global_max = float("-inf")
        total_count = 0
        
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var not in grp:
                        continue
                    
                    ds = grp[self.time_var]
                    for start in range(0, ds.shape[0], self.chunk_size):
                        end = min(start + self.chunk_size, ds.shape[0])
                        chunk = ds[start:end]
                        
                        # Filter positive times
                        chunk = chunk[chunk > 1e-10]
                        if chunk.size == 0:
                            continue
                        
                        total_count += int(chunk.size)
                        cmin, cmax = float(chunk.min()), float(chunk.max())
                        global_min = min(global_min, cmin)
                        global_max = max(global_max, cmax)
        
        return global_min, global_max, total_count
    
    def _compute_tau0_percentile(
        self,
        file_paths: List[Path],
        global_min: float,
        global_max: float
    ) -> float:
        """Compute tau0 as 5th percentile using histogram."""
        num_bins = int(self.proc_cfg.get("time_hist_bins", 4096))
        edges = np.linspace(global_min, global_max, num_bins + 1, dtype=np.float64)
        hist = np.zeros(num_bins, dtype=np.int64)
        
        # Build histogram
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var not in grp:
                        continue
                    
                    ds = grp[self.time_var]
                    for start in range(0, ds.shape[0], self.chunk_size):
                        end = min(start + self.chunk_size, ds.shape[0])
                        chunk = ds[start:end]
                        chunk = chunk[chunk > 1e-10]
                        
                        if chunk.size == 0:
                            continue
                        
                        h, _ = np.histogram(chunk, bins=edges)
                        hist += h
        
        # Find 5th percentile
        target_rank = max(0, int(0.05 * hist.sum()))
        cumsum = np.cumsum(hist)
        bin_idx = int(np.searchsorted(cumsum, target_rank, side="left"))
        bin_idx = max(0, min(bin_idx, num_bins - 1))
        
        # Linear interpolation within bin
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        prev_cum = 0 if bin_idx == 0 else int(cumsum[bin_idx - 1])
        in_bin_rank = max(0, target_rank - prev_cum)
        bin_count = int(hist[bin_idx]) if hist[bin_idx] > 0 else 1
        frac = min(1.0, in_bin_rank / bin_count)
        
        return float(left + (right - left) * frac)


def _process_one_file_worker(
    file_path_str: str,
    output_dir_str: str,
    config_json: str,
    norm_stats_json: Optional[str],
) -> Dict[str, Any]:
    """
    Worker function for parallel file processing.
    
    Args:
        file_path_str: Path to input file as string
        output_dir_str: Output directory as string
        config_json: Serialized config
        norm_stats_json: Serialized normalization stats
        
    Returns:
        Metadata dictionary with processing results
    """
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    config = json.loads(config_json)
    norm_stats = json.loads(norm_stats_json) if norm_stats_json else None
    
    # Create local statistics logger
    stats_logger = DataStatisticsLogger(output_dir)
    processor = CorePreprocessor(config, norm_stats, stats_logger)
    
    # Process file
    meta = processor.process_file_for_sequence_shards(file_path, output_dir)
    
    # Attach stats for merging
    meta["_worker_stats"] = asdict(stats_logger.stats)
    
    return meta


class DataPreprocessor:
    """
    Main preprocessor orchestrating the full pipeline.
    
    Coordinates:
    1. Time statistics collection
    2. Normalization statistics collection  
    3. Parallel/serial trajectory processing
    4. Metadata generation and saving
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
        self.sequence_mode = True
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_logger = DataStatisticsLogger(output_dir)
    
    def _welford_merge(
        self,
        acc_count: int,
        acc_mean: float,
        acc_m2: float,
        chunk: np.ndarray
    ) -> Tuple[int, float, float, float, float]:
        """
        Merge chunk into running statistics using Welford's algorithm.
        
        Args:
            acc_count: Accumulated count
            acc_mean: Accumulated mean
            acc_m2: Accumulated sum of squared differences
            chunk: New data chunk
            
        Returns:
            Updated (count, mean, m2, min, max)
        """
        d = chunk[np.isfinite(chunk)]
        if d.size == 0:
            return acc_count, acc_mean, acc_m2, float("inf"), float("-inf")
        
        # Chunk statistics
        n1 = int(d.size)
        c_min = float(d.min())
        c_max = float(d.max())
        c_mean = float(d.mean())
        c_m2 = float(((d - c_mean) ** 2).sum())
        
        if acc_count == 0:
            return n1, c_mean, c_m2, c_min, c_max
        
        # Merge using Chan-Welford algorithm
        n0 = acc_count
        delta = c_mean - acc_mean
        n = n0 + n1
        mean = acc_mean + delta * (n1 / n)
        m2 = acc_m2 + c_m2 + (delta * delta) * (n0 * n1 / n)
        
        return n, mean, m2, c_min, c_max
    
    def _collect_normalization_stats(self) -> Dict[str, Any]:
        """
        Collect normalization statistics for all variables.
        
        Returns:
            Dictionary with per-variable statistics and methods
        """
        self.logger.info("Collecting normalization statistics...")
        
        stats: Dict[str, Any] = {"per_key_stats": {}, "normalization_methods": {}}
        accumulators: Dict[str, Dict[str, float]] = {}
        
        # Get configuration
        norm_cfg = self.config["normalization"]
        default_method = norm_cfg.get("default_method", "log-standard")
        methods_override = norm_cfg.get("methods", {})
        all_vars = self.config["data"]["species_variables"] + self.config["data"]["global_variables"]
        
        # Initialize accumulators
        for var in all_vars:
            accumulators[var] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "lin_min": float("inf"),
                "lin_max": float("-inf"),
                "log_min": float("inf"),
                "log_max": float("-inf"),
            }
        
        chunk_size = int(self.config["preprocessing"].get("hdf5_chunk_size", 10000))
        
        # Process all files
        for file_path in self.raw_files:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    
                    # Process species datasets
                    for var in self.config["data"]["species_variables"]:
                        if var not in grp:
                            continue
                        
                        ds = grp[var]
                        var_method = methods_override.get(var, default_method)
                        
                        for start in range(0, ds.shape[0], chunk_size):
                            end = min(start + chunk_size, ds.shape[0])
                            arr_raw = ds[start:end]
                            
                            # Update species range
                            if self.stats_logger:
                                self.stats_logger.update_species_range(var, arr_raw)
                            
                            # Choose domain for statistics
                            if "log" in var_method or var_method == "log10":
                                eps = norm_cfg.get("epsilon", 1e-30)
                                arr = np.log10(np.maximum(arr_raw, eps))
                            else:
                                arr = arr_raw
                            
                            # Update statistics
                            a = accumulators[var]
                            n, m, m2, _, _ = self._welford_merge(
                                a["count"], a["mean"], a["m2"], arr
                            )
                            a["count"], a["mean"], a["m2"] = n, m, m2
                            
                            # Update min/max in both domains
                            finite_lin = arr_raw[np.isfinite(arr_raw)]
                            if finite_lin.size:
                                a["lin_min"] = min(a["lin_min"], float(finite_lin.min()))
                                a["lin_max"] = max(a["lin_max"], float(finite_lin.max()))
                            
                            eps = float(norm_cfg.get("epsilon", 1e-30))
                            raw_pos = np.maximum(arr_raw, eps)
                            finite_log = raw_pos[np.isfinite(raw_pos)]
                            if finite_log.size:
                                log_vals = np.log10(finite_log)
                                a["log_min"] = min(a["log_min"], float(log_vals.min()))
                                a["log_max"] = max(a["log_max"], float(log_vals.max()))
                    
                    # Process global attributes
                    for var in self.config["data"]["global_variables"]:
                        if var in grp.attrs:
                            value = float(grp.attrs[var])
                            var_method = methods_override.get(var, default_method)
                            a = accumulators[var]
                            
                            # Accumulate for standardization methods
                            if "standard" in var_method:
                                if "log" in var_method or var_method == "log10":
                                    v = math.log10(max(value, norm_cfg.get("epsilon", 1e-30)))
                                else:
                                    v = value
                                
                                n, m, m2, _, _ = self._welford_merge(
                                    a["count"], a["mean"], a["m2"],
                                    np.array([v], dtype=np.float64),
                                )
                                a["count"], a["mean"], a["m2"] = n, m, m2
                            
                            # Track min/max
                            if math.isfinite(value):
                                a["lin_min"] = min(a["lin_min"], value)
                                a["lin_max"] = max(a["lin_max"], value)
                                
                                eps = float(norm_cfg.get("epsilon", 1e-30))
                                v_log = math.log10(max(value, eps))
                                a["log_min"] = min(a["log_min"], v_log)
                                a["log_max"] = max(a["log_max"], v_log)
                            
                            if self.stats_logger:
                                self.stats_logger.update_global_range(var, value)
        
        # Finalize statistics
        for var, a in accumulators.items():
            if not (math.isfinite(a["lin_min"]) and math.isfinite(a["lin_max"])):
                continue
            
            var_method = methods_override.get(var, default_method)
            stats["normalization_methods"][var] = var_method
            
            # Build stats block
            block = {
                "method": var_method,
                "min": float(a["lin_min"]),
                "max": float(a["lin_max"]),
                "log_min": float(a["log_min"]),
                "log_max": float(a["log_max"]),
            }
            
            # Add mean/std for standardization methods
            if "standard" in var_method:
                variance = a["m2"] / (a["count"] - 1) if a["count"] > 1 else 0.0
                std = float(np.sqrt(max(variance, 0.0)))
                min_std = float(norm_cfg.get("min_std", 1e-10))
                
                key_prefix = "log_" if ("log" in var_method or var_method == "log10") else ""
                block[f"{key_prefix}mean"] = float(a["mean"])
                block[f"{key_prefix}std"] = float(max(std, min_std))
            
            stats["per_key_stats"][var] = block
        
        # Add time method
        time_var = self.config["data"]["time_variable"]
        stats["normalization_methods"][time_var] = (
            self.config.get("normalization", {})
                .get("methods", {})
                .get(time_var, "log-min-max")
        )
        
        self.logger.info("Normalization statistics collection complete.")
        return stats
    
    def process_to_npy_shards(self) -> None:
        """
        Execute the full preprocessing pipeline.
        
        Processes raw HDF5 files into normalized NPZ shards
        ready for training.
        """
        self.logger.info("Processing data in SEQUENCE MODE for LiLaN")
        
        # Collect time statistics
        processor = CorePreprocessor(self.config, stats_logger=self.stats_logger)
        time_stats = processor.collect_time_stats(self.raw_files)
        
        # Determine fixed grid bounds
        cfg_range = self.config.get("data", {}).get("fixed_time_range")
        if cfg_range is not None:
            grid_min = float(cfg_range.get("min"))
            grid_max = float(cfg_range.get("max"))
        else:
            grid_min = float(time_stats["tmin_raw"])
            grid_max = float(time_stats["tmax_raw"])
        
        # Collect normalization statistics
        norm_stats = self._collect_normalization_stats()
        norm_stats["time_normalization"] = time_stats
        norm_stats["fixed_time_grid"] = {
            "min": grid_min,
            "max": grid_max,
            "M": int(self.config["data"]["M_per_sample"]),
            "endpoint": True,
            "spacing": "log",
        }
        norm_stats["already_logged_vars"] = list(self.config["data"]["species_variables"])
        
        # Save normalization stats
        save_json(norm_stats, self.output_dir / "normalization.json")
        
        # Process files
        all_metadata = self._process_files(norm_stats)
        
        # Save shard index
        self._save_shard_index(all_metadata, time_stats, grid_min, grid_max)
        
        # Save statistics summary
        self.stats_logger.save_summary()
        
        # Log summary
        self.logger.info(
            "Sequence mode preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )
    
    def _process_files(self, norm_stats: Dict[str, Any]) -> Dict[str, Any]:
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
            norm_json = json.dumps(norm_stats)
            
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as ex:
                futures = [
                    ex.submit(
                        _process_one_file_worker,
                        str(file_path),
                        str(self.processed_dir),
                        cfg_json,
                        norm_json,
                    )
                    for file_path in self.raw_files
                ]
                
                for fut in as_completed(futures):
                    metadata = fut.result()
                    self._merge_metadata(all_metadata, metadata)
                    
                    # Merge worker stats
                    if "_worker_stats" in metadata:
                        worker_stats = DataStatistics(**metadata["_worker_stats"])
                        self.stats_logger.stats.merge_inplace(worker_stats)
        else:
            # Serial processing
            for file_path in self.raw_files:
                self.logger.info(f"Processing file: {file_path}")
                cproc = CorePreprocessor(self.config, norm_stats, self.stats_logger)
                metadata = cproc.process_file_for_sequence_shards(file_path, self.processed_dir)
                self._merge_metadata(all_metadata, metadata)
        
        return all_metadata
    
    def _save_shard_index(
        self,
        all_metadata: Dict[str, Any],
        time_stats: Dict[str, float],
        grid_min: float,
        grid_max: float
    ) -> None:
        """Save shard index with all metadata."""
        shard_index = {
            "sequence_mode": True,
            "M_per_sample": self.config["data"]["M_per_sample"],
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
            "time_normalization": time_stats,
            "time_grid": {
                "type": "logspace",
                "min": grid_min,
                "max": grid_max,
                "M": int(self.config["data"]["M_per_sample"]),
                "endpoint": True,
            },
            "already_logged_vars": list(self.config["data"]["species_variables"]),
        }
        save_json(shard_index, self.output_dir / "shard_index.json")
    
    @staticmethod
    def _merge_metadata(all_meta: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Merge file metadata into combined metadata."""
        for split in ("train", "validation", "test"):
            all_meta["splits"][split]["shards"].extend(meta["splits"][split]["shards"])
            all_meta["splits"][split]["n_trajectories"] += meta["splits"][split]["n_trajectories"]