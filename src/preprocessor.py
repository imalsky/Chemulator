#!/usr/bin/env python3
"""
Data preprocessing module for chemical kinetics simulation data.

Transforms HDF5 simulation data into sharded NumPy arrays for efficient training.
Performs validation, deterministic splitting, and calculates normalization statistics
in a single pass over the data to minimize I/O overhead.
"""

import hashlib
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm

from utils import save_json


# Configuration constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MIN_VALUE_THRESHOLD = 1e-30
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10


@dataclass
class StreamingStatistics:
    """Accumulator for computing statistics in a streaming fashion using Welford's algorithm."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, values: np.ndarray) -> None:
        """Update statistics with new batch of values."""
        if values.size == 0:
            return
        
        # Update min/max
        self.min_val = min(self.min_val, float(values.min()))
        self.max_val = max(self.max_val, float(values.max()))
        
        # Welford's algorithm for mean and variance
        n0 = self.count
        n1 = int(values.size)
        n = n0 + n1
        
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        
        if n0 == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (n1 / n)
            self.m2 = self.m2 + batch_m2 + (delta * delta) * (n0 * n1 / n)
        
        self.count = n
    
    def merge(self, other: 'StreamingStatistics') -> None:
        """Merge statistics from another accumulator."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.min_val = other.min_val
            self.max_val = other.max_val
        else:
            n1, n2 = self.count, other.count
            n = n1 + n2
            delta = other.mean - self.mean
            self.mean = self.mean + delta * (n2 / n)
            self.m2 = self.m2 + other.m2 + (delta * delta) * (n1 * n2 / n)
            self.count = n
            self.min_val = min(self.min_val, other.min_val)
            self.max_val = max(self.max_val, other.max_val)
    
    def get_stats(self, min_std: float = DEFAULT_MIN_STD) -> Dict[str, float]:
        """Get final statistics."""
        if self.count == 0:
            return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
        
        variance = self.m2 / max(1, self.count - 1)
        std = max(math.sqrt(max(0.0, variance)), min_std)
        
        return {
            "mean": self.mean,
            "std": std,
            "min": self.min_val if math.isfinite(self.min_val) else 0.0,
            "max": self.max_val if math.isfinite(self.max_val) else 1.0
        }


@dataclass
class DataStatistics:
    """Track preprocessing statistics."""
    # Counts
    total_groups: int = 0
    valid_trajectories: int = 0
    dropped_missing_keys: int = 0
    dropped_non_finite: int = 0
    dropped_below_threshold: int = 0
    dropped_use_fraction: int = 0
    
    # Data characteristics
    total_time_points: int = 0
    species_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    global_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range: Tuple[float, float] = (float('inf'), float('-inf'))
    
    # Split distribution
    split_counts: Dict[str, int] = field(default_factory=lambda: {
        "train": 0, "validation": 0, "test": 0
    })
    
    # Performance
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    # Normalization accumulators (for training data only)
    accumulators: Dict[str, StreamingStatistics] = field(default_factory=dict)
    
    def merge(self, other: 'DataStatistics') -> None:
        """Merge statistics from another instance."""
        self.total_groups += other.total_groups
        self.valid_trajectories += other.valid_trajectories
        self.dropped_missing_keys += other.dropped_missing_keys
        self.dropped_non_finite += other.dropped_non_finite
        self.dropped_below_threshold += other.dropped_below_threshold
        self.dropped_use_fraction += other.dropped_use_fraction
        self.total_time_points += other.total_time_points
        
        # Merge ranges
        for k, (mn, mx) in other.species_ranges.items():
            if k in self.species_ranges:
                omn, omx = self.species_ranges[k]
                self.species_ranges[k] = (min(omn, mn), max(omx, mx))
            else:
                self.species_ranges[k] = (mn, mx)
        
        for k, (mn, mx) in other.global_ranges.items():
            if k in self.global_ranges:
                omn, omx = self.global_ranges[k]
                self.global_ranges[k] = (min(omn, mn), max(omx, mx))
            else:
                self.global_ranges[k] = (mn, mx)
        
        self.time_range = (
            min(self.time_range[0], other.time_range[0]),
            max(self.time_range[1], other.time_range[1])
        )
        
        for split, cnt in other.split_counts.items():
            self.split_counts[split] += cnt
        
        self.processing_times.update(other.processing_times)
        
        # Merge normalization accumulators
        for var, acc in other.accumulators.items():
            if var not in self.accumulators:
                self.accumulators[var] = StreamingStatistics()
            self.accumulators[var].merge(acc)


class ShardWriter:
    """Efficient writer for trajectory data shards."""
    
    def __init__(
        self,
        output_dir: Path,
        trajectories_per_shard: int,
        shard_prefix: str,
        n_species: int,
        n_globals: int,
        n_targets: int,
        dtype: np.dtype,
        compressed: bool = True
    ):
        self.output_dir = output_dir
        self.trajectories_per_shard = max(1, trajectories_per_shard)
        self.shard_prefix = shard_prefix
        self.n_species = n_species
        self.n_globals = n_globals
        self.n_targets = n_targets
        self.dtype = dtype
        self.compressed = compressed
        
        self.buffer = []
        self.shard_id = 0
        self.shard_metadata = []
        self.expected_M = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_trajectory(
        self,
        x0: np.ndarray,
        globals_vec: np.ndarray,
        t_vec: np.ndarray,
        y_mat: np.ndarray
    ) -> None:
        """Add trajectory to buffer and write shard when full."""
        # Validate dimensions
        M = t_vec.shape[0]
        if self.expected_M is None:
            self.expected_M = M
        elif M != self.expected_M:
            raise ValueError(f"Inconsistent trajectory length: {M} vs {self.expected_M}")
        
        self.buffer.append({
            "x0": x0.astype(self.dtype),
            "globals": globals_vec.astype(self.dtype),
            "t_vec": t_vec.astype(self.dtype),
            "y_mat": y_mat.astype(self.dtype)
        })
        
        if len(self.buffer) >= self.trajectories_per_shard:
            self._write_shard()
    
    def _write_shard(self) -> None:
        """Write buffered trajectories to disk."""
        if not self.buffer:
            return
        
        # Stack trajectories
        data = {
            "x0": np.stack([t["x0"] for t in self.buffer]),
            "globals": np.stack([t["globals"] for t in self.buffer]),
            "t_vec": np.stack([t["t_vec"] for t in self.buffer]),
            "y_mat": np.stack([t["y_mat"] for t in self.buffer])
        }
        
        filename = f"shard_{self.shard_prefix}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        
        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(filepath, **data)
        
        self.shard_metadata.append({
            "filename": filename,
            "n_trajectories": len(self.buffer)
        })
        
        self.buffer = []
        self.shard_id += 1
    
    def flush(self) -> None:
        """Write any remaining trajectories."""
        if self.buffer:
            self._write_shard()


class SingleFileProcessor:
    """Process a single HDF5 file with single-pass statistics collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.prep_cfg = config["preprocessing"]
        self.train_cfg = config["training"]
        self.sys_cfg = config["system"]
        
        self._setup_parameters()
        self.stats = DataStatistics()
    
    def _setup_parameters(self) -> None:
        """Initialize processing parameters."""
        # Variables
        self.species_vars = self.data_cfg["species_variables"]
        self.target_vars = self.data_cfg.get("target_species_variables", self.species_vars)
        self.global_vars = self.data_cfg["global_variables"]
        self.time_var = self.data_cfg.get("time_variable", "t_time")
        
        # Dimensions
        self.n_species = len(self.species_vars)
        self.n_targets = len(self.target_vars)
        self.n_globals = len(self.global_vars)
        
        # Data type
        dtype_str = self.sys_cfg.get("dtype", "float32")
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32
        
        # Thresholds
        self.min_threshold = float(self.prep_cfg.get("min_value_threshold", DEFAULT_MIN_VALUE_THRESHOLD))
        self.epsilon = float(self.norm_cfg.get("epsilon", DEFAULT_EPSILON))
        self.log_floor = max(self.min_threshold, self.epsilon)
        self.min_std = float(self.norm_cfg.get("min_std", DEFAULT_MIN_STD))
        
        self.trajectories_per_shard = int(self.prep_cfg["trajectories_per_shard"])
        
        # Split fractions
        self.test_frac = float(self.train_cfg.get("test_fraction", 0.0))
        self.val_frac = float(self.train_cfg.get("val_fraction", 0.0))
        self.use_fraction = float(self.train_cfg.get("use_fraction", 1.0))
        
        # Normalization methods
        self.default_method = self.norm_cfg.get("default_method", "log-standard")
        self.methods = self.norm_cfg.get("methods", {})
        
        if not (0 <= self.test_frac <= 1 and 0 <= self.val_frac <= 1):
            raise ValueError("Split fractions must be in [0, 1]")
        if self.test_frac + self.val_frac > 1:
            raise ValueError("test_fraction + val_fraction must be <= 1")
    
    def _get_method(self, var: str) -> str:
        """Get normalization method for variable."""
        return self.methods.get(var, self.default_method).lower()
    
    def _apply_transform_for_stats(self, values: np.ndarray, method: str) -> np.ndarray:
            if method == "time-norm":
                tau0 = self.norm_cfg.get("tau0", 1.0)
                if tau0 <= 0:
                    raise ValueError(f"tau0 must be positive for 'time-norm', but got {tau0}")
                return np.log1p(values / tau0)  # Collect stats in tau space
            elif "log" in method:
                return np.log10(np.maximum(values, self.log_floor))
            return values
    
    def _deterministic_hash(self, s: str, seed: int) -> float:
        """Generate deterministic hash value in [0, 1)."""
        b = f"{seed}:{s}".encode("utf-8")
        h = hashlib.sha256(b).digest()
        return int.from_bytes(h[:8], "big", signed=False) / (1 << 64)
    
    def _determine_split(self, group_name: str, seed: int) -> Optional[str]:
        """Determine data split for trajectory."""
        # Check use_fraction filter
        if self.use_fraction < 1.0:
            if self._deterministic_hash(f"{group_name}:use", seed) >= self.use_fraction:
                self.stats.dropped_use_fraction += 1
                return None
        
        # Determine split
        split_hash = self._deterministic_hash(f"{group_name}:split", seed)
        if split_hash < self.test_frac:
            return "test"
        elif split_hash < self.test_frac + self.val_frac:
            return "validation"
        else:
            return "train"
    
    def _update_normalization_stats(
            self,
            x0: np.ndarray,
            globals_vec: np.ndarray,
            time_data: np.ndarray,
            y_mat: np.ndarray
        ) -> None:
            """Update normalization statistics for training data only."""
            # Use full trajectory for input species that are also targets.
            
            # Process input species variables
            for i, var in enumerate(self.species_vars):
                method = self._get_method(var)
                
                # Check if this input species is also a target to decide data source
                if var in self.target_vars:
                    # If it's a target, use the full trajectory for more representative stats
                    try:
                        j = self.target_vars.index(var)
                        values = y_mat[:, j]
                    except ValueError:
                        # Should not happen if config is valid, but as a fallback:
                        values = np.array([x0[i]])
                else:
                    # If it's an input-only species, stats must be based on initial condition
                    values = np.array([x0[i]])

                transformed = self._apply_transform_for_stats(values, method)
                if var not in self.stats.accumulators:
                    self.stats.accumulators[var] = StreamingStatistics()
                self.stats.accumulators[var].update(transformed)
            
            # Process target variables that are NOT also input species
            for j, var in enumerate(self.target_vars):
                if var not in self.species_vars:
                    method = self._get_method(var)
                    values = y_mat[:, j]
                    
                    transformed = self._apply_transform_for_stats(values, method)
                    if var not in self.stats.accumulators:
                        self.stats.accumulators[var] = StreamingStatistics()
                    self.stats.accumulators[var].update(transformed)
            
            # Process global variables
            for i, var in enumerate(self.global_vars):
                method = self._get_method(var)
                values = np.array([globals_vec[i]])
                transformed = self._apply_transform_for_stats(values, method)
                
                if var not in self.stats.accumulators:
                    self.stats.accumulators[var] = StreamingStatistics()
                self.stats.accumulators[var].update(transformed)
            
            # Process time variable
            method = self._get_method(self.time_var)
            transformed = self._apply_transform_for_stats(time_data, method)
            
            if self.time_var not in self.stats.accumulators:
                self.stats.accumulators[self.time_var] = StreamingStatistics()
            self.stats.accumulators[self.time_var].update(transformed)
    
    def _extract_trajectory(
            self,
            group: h5py.Group,
            group_name: str
        ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            """Extract and validate trajectory data."""
            # Check for required data
            missing = [v for v in self.global_vars if v not in group.attrs]
            if missing or self.time_var not in group:
                self.stats.dropped_missing_keys += 1
                return None
            
            # Load time data
            time_data = group[self.time_var][:]
            if time_data.size == 0 or not np.all(np.isfinite(time_data)):
                self.stats.dropped_non_finite += 1
                return None
            
            # Use a tolerance-based check consistent with the model,
            # allowing duplicate time points (diff=0) but rejecting decreasing time.
            time_diffs = np.diff(time_data)
            if np.any(time_diffs < -1e-9):
                self.stats.dropped_non_finite += 1
                return None
            
            n_times = time_data.shape[0]
            
            # Extract initial conditions
            x0 = np.empty(self.n_species, dtype=self.np_dtype)
            for i, var in enumerate(self.species_vars):
                if var not in group:
                    self.stats.dropped_missing_keys += 1
                    return None
                
                dset = group[var]
                if dset.shape[0] == 0:
                    self.stats.dropped_missing_keys += 1
                    return None
                
                value = float(dset[0])
                if not np.isfinite(value) or value <= self.min_threshold:
                    self.stats.dropped_below_threshold += 1
                    return None
                
                x0[i] = value
            
            # Extract target species evolution
            y_mat = np.empty((n_times, self.n_targets), dtype=self.np_dtype)
            for j, var in enumerate(self.target_vars):
                if var not in group:
                    self.stats.dropped_missing_keys += 1
                    return None
                
                arr = group[var][:]
                if arr.shape[0] != n_times:
                    self.stats.dropped_missing_keys += 1
                    return None
                
                if not np.all(np.isfinite(arr)) or np.any(arr <= self.min_threshold):
                    self.stats.dropped_below_threshold += 1
                    return None
                
                y_mat[:, j] = arr
            
            # Extract global parameters
            globals_vec = np.empty(self.n_globals, dtype=self.np_dtype)
            for i, var in enumerate(self.global_vars):
                value = float(group.attrs[var])
                if not np.isfinite(value):
                    self.stats.dropped_non_finite += 1
                    return None
                globals_vec[i] = value
            
            # Update data ranges
            for j, var in enumerate(self.target_vars):
                vmin, vmax = float(y_mat[:, j].min()), float(y_mat[:, j].max())
                if var in self.stats.species_ranges:
                    old_min, old_max = self.stats.species_ranges[var]
                    self.stats.species_ranges[var] = (min(old_min, vmin), max(old_max, vmax))
                else:
                    self.stats.species_ranges[var] = (vmin, vmax)
            
            for i, var in enumerate(self.global_vars):
                val = float(globals_vec[i])
                if var in self.stats.global_ranges:
                    old_min, old_max = self.stats.global_ranges[var]
                    self.stats.global_ranges[var] = (min(old_min, val), max(old_max, val))
                else:
                    self.stats.global_ranges[var] = (val, val)
            
            tmin, tmax = float(time_data.min()), float(time_data.max())
            self.stats.time_range = (
                min(self.stats.time_range[0], tmin),
                max(self.stats.time_range[1], tmax)
            )
            self.stats.total_time_points += n_times
            
            return x0, globals_vec, time_data, y_mat
    
    def process(
        self,
        file_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process single HDF5 file with single-pass statistics collection."""
        start_time = time.time()
        
        # Generate unique shard prefix
        file_hash = hashlib.sha256(file_path.name.encode()).hexdigest()[:12]
        shard_prefix = f"{file_path.stem}_{file_hash}"
        
        # Initialize writers for each split
        compressed = bool(self.prep_cfg.get("npz_compressed", True))
        writers = {}
        for split in ("train", "validation", "test"):
            writers[split] = ShardWriter(
                output_dir / split,
                self.trajectories_per_shard,
                shard_prefix,
                self.n_species,
                self.n_globals,
                self.n_targets,
                self.np_dtype,
                compressed
            )
        
        # Process trajectories
        seed = int(self.sys_cfg.get("seed", 42))
        
        with h5py.File(file_path, "r") as f:
            for group_name in tqdm(f.keys(), desc=f"Processing {file_path.name}"):
                self.stats.total_groups += 1
                
                # Determine split
                split = self._determine_split(group_name, seed)
                if split is None:
                    continue
                
                # Extract trajectory
                result = self._extract_trajectory(f[group_name], group_name)
                if result is None:
                    continue
                
                x0, globals_vec, time_data, y_mat = result
                self.stats.valid_trajectories += 1
                self.stats.split_counts[split] += 1
                
                # Update normalization statistics (training data only)
                if split == "train":
                    self._update_normalization_stats(x0, globals_vec, time_data, y_mat)
                
                # Write raw data to appropriate shard
                writers[split].add_trajectory(x0, globals_vec, time_data, y_mat)
        
        # Flush all writers
        for writer in writers.values():
            writer.flush()
        
        # Record processing time
        self.stats.processing_times[str(file_path)] = time.time() - start_time
        
        # Build metadata
        metadata = {
            "splits": {}
        }
        for split in ("train", "validation", "test"):
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": self.stats.split_counts[split]
            }
        
        # Store expected M if any valid trajectories were found
        if self.stats.valid_trajectories > 0:
            metadata["expected_M"] = writers["train"].expected_M or writers["validation"].expected_M or writers["test"].expected_M
        
        return metadata


def _process_file_worker(
    file_path_str: str,
    output_dir_str: str,
    config_pickle: bytes
) -> Dict[str, Any]:
    """Worker function for parallel processing."""
    import pickle
    
    processor = SingleFileProcessor(pickle.loads(config_pickle))
    metadata = processor.process(Path(file_path_str), Path(output_dir_str))
    metadata["_stats"] = asdict(processor.stats)
    return metadata


class DataPreprocessor:
    """Main preprocessor orchestrating data transformation and statistics calculation."""
    
    def __init__(
        self,
        raw_files: List[Path],
        output_dir: Path,
        config: Dict[str, Any]
    ):
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_to_npy_shards(self) -> None:
        """Execute preprocessing pipeline with single-pass statistics collection."""
        self.logger.info("Starting data preprocessing")
        
        # Process all files and collect statistics
        all_metadata = self._process_files()
        
        # Compile and save normalization statistics
        norm_stats = self._compile_normalization_stats(all_metadata)
        save_json(norm_stats, self.output_dir / "normalization.json")
        self.logger.info("Saved normalization statistics")
        
        # Save shard index
        self._save_shard_index(all_metadata)
        
        # Save preprocessing summary
        self._save_summary(all_metadata)
        
        self.logger.info(
            "Preprocessing complete - Train: %d, Val: %d, Test: %d trajectories",
            all_metadata["splits"]["train"]["n_trajectories"],
            all_metadata["splits"]["validation"]["n_trajectories"],
            all_metadata["splits"]["test"]["n_trajectories"]
        )
    
    def _process_files(self) -> Dict[str, Any]:
            """Process all HDF5 files with single-pass statistics collection."""
            combined_metadata = {
                "expected_M": None,
                "splits": {
                    "train": {"shards": [], "n_trajectories": 0},
                    "validation": {"shards": [], "n_trajectories": 0},
                    "test": {"shards": [], "n_trajectories": 0}
                }
            }
            
            combined_stats = DataStatistics()
            num_workers = int(self.config.get("preprocessing", {}).get("num_workers", 0))
            
            if num_workers > 0:
                # Parallel processing
                import pickle
                import multiprocessing as mp
                
                self.logger.info(f"Processing with {num_workers} workers")
                config_pickle = pickle.dumps(self.config)
                
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as ex:
                    futures = [
                        ex.submit(_process_file_worker, str(f), str(self.output_dir), config_pickle)
                        for f in self.raw_files
                    ]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                        metadata = future.result()
                        self._merge_metadata(combined_metadata, metadata)
                        
                        if "_stats" in metadata:
                            stats_dict = metadata["_stats"]
                            # Reconstruct StreamingStatistics objects from dictionaries
                            if "accumulators" in stats_dict:
                                stats_dict["accumulators"] = {
                                    var: StreamingStatistics(**acc_dict)
                                    for var, acc_dict in stats_dict["accumulators"].items()
                                }
                            file_stats = DataStatistics(**stats_dict)
                            combined_stats.merge(file_stats)
            else:
                # Serial processing
                for file_path in tqdm(self.raw_files, desc="Processing files"):
                    processor = SingleFileProcessor(self.config)
                    metadata = processor.process(file_path, self.output_dir)
                    self._merge_metadata(combined_metadata, metadata)
                    combined_stats.merge(processor.stats)
            
            # Store combined statistics
            combined_metadata["statistics"] = combined_stats
            return combined_metadata
    
    def _compile_normalization_stats(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compile normalization statistics from accumulated data."""
        combined_stats = metadata["statistics"]
        norm_cfg = self.config["normalization"]
        data_cfg = self.config["data"]
        
        default_method = norm_cfg.get("default_method", "log-standard")
        methods = norm_cfg.get("methods", {})
        min_std = float(norm_cfg.get("min_std", DEFAULT_MIN_STD))
        
        stats = {
            "per_key_stats": {},
            "normalization_methods": {}
        }
        
        # Process each variable
        all_vars = (set(data_cfg["species_variables"]) | 
                   set(data_cfg.get("target_species_variables", data_cfg["species_variables"])) |
                   set(data_cfg["global_variables"]) | 
                   {data_cfg.get("time_variable", "t_time")})
        
        for var in all_vars:
            if var not in combined_stats.accumulators:
                continue
            
            method = methods.get(var, default_method).lower()
            stats["normalization_methods"][var] = method
            
            var_stats = combined_stats.accumulators[var].get_stats(min_std)
            
            # Build statistics based on method
            if method == "log-standard":
                stats["per_key_stats"][var] = {
                    "method": method,
                    "log_mean": var_stats["mean"],
                    "log_std": var_stats["std"],
                    "log_min": var_stats["min"],
                    "log_max": var_stats["max"]
                }
            elif method == "standard":
                stats["per_key_stats"][var] = {
                    "method": method,
                    "mean": var_stats["mean"],
                    "std": var_stats["std"],
                    "min": var_stats["min"],
                    "max": var_stats["max"]
                }
            elif method == "log-min-max":
                stats["per_key_stats"][var] = {
                    "method": method,
                    "log_min": var_stats["min"],
                    "log_max": var_stats["max"]
                }
            elif method == "min-max":
                stats["per_key_stats"][var] = {
                    "method": method,
                    "min": var_stats["min"],
                    "max": var_stats["max"]
                }
            elif method == "log10":
                stats["per_key_stats"][var] = {
                    "method": method,
                    "log_min": var_stats["min"],
                    "log_max": var_stats["max"]
                }
            else:  # "none"
                stats["per_key_stats"][var] = {"method": method}
        
        # Special handling for time normalization
        time_var = data_cfg.get("time_variable", "t_time")
        if time_var in stats["per_key_stats"]:
            time_method = methods.get(time_var, default_method).lower()
            
            if time_method == "time-norm":
                # Tau-space normalization
                tau0 = float(norm_cfg.get("tau0", 1.0))
                raw_stats = combined_stats.accumulators[time_var].get_stats()
                stats["time_normalization"] = {
                    "time_transform": "time-norm",
                    "tau0": tau0,
                    "tmin": math.log1p(raw_stats["min"] / tau0),
                    "tmax": math.log1p(raw_stats["max"] / tau0),
                    "tmin_raw": raw_stats["min"],
                    "tmax_raw": raw_stats["max"]
                }
            elif time_method == "log-min-max":
                time_stats = stats["per_key_stats"][time_var]
                stats["time_normalization"] = {
                    "time_transform": "log-min-max",
                    "tmin_raw": 10 ** time_stats["log_min"],
                    "tmax_raw": 10 ** time_stats["log_max"]
                }
        
        return stats
    
    def _merge_metadata(self, target: Dict, source: Dict) -> None:
        """Merge file metadata into combined metadata."""
        for split in ("train", "validation", "test"):
            target["splits"][split]["shards"].extend(source["splits"][split]["shards"])
            target["splits"][split]["n_trajectories"] += source["splits"][split]["n_trajectories"]
        
        # Validate M consistency
        if "expected_M" in source and source["expected_M"]:
            if target["expected_M"] is None:
                target["expected_M"] = source["expected_M"]
            elif target["expected_M"] != source["expected_M"]:
                raise ValueError(f"Inconsistent M across files: {target['expected_M']} vs {source['expected_M']}")
    
    def _save_shard_index(self, metadata: Dict[str, Any]) -> None:
        """Save shard index file."""
        if not metadata.get("expected_M"):
            raise ValueError("No valid trajectories processed")
        
        shard_index = {
            "sequence_mode": True,
            "variable_length": False,
            "M_per_sample": int(metadata["expected_M"]),
            "n_input_species": len(self.config["data"]["species_variables"]),
            "n_target_species": len(self.config["data"].get(
                "target_species_variables",
                self.config["data"]["species_variables"]
            )),
            "n_globals": len(self.config["data"]["global_variables"]),
            "compression": "npz",
            "splits": metadata["splits"]
        }
        
        save_json(shard_index, self.output_dir / "shard_index.json")
        self.logger.info("Saved shard index")
    
    def _save_summary(self, metadata: Dict[str, Any]) -> None:
        """Save preprocessing summary."""
        stats = metadata["statistics"]
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_groups_processed": stats.total_groups,
            "valid_trajectories": stats.valid_trajectories,
            "dropped_counts": {
                "missing_keys": stats.dropped_missing_keys,
                "non_finite": stats.dropped_non_finite,
                "below_threshold": stats.dropped_below_threshold,
                "use_fraction": stats.dropped_use_fraction
            },
            "data_ranges": {
                "species": stats.species_ranges,
                "globals": stats.global_ranges,
                "time": {
                    "min": stats.time_range[0] if math.isfinite(stats.time_range[0]) else 0,
                    "max": stats.time_range[1] if math.isfinite(stats.time_range[1]) else 0
                }
            },
            "split_distribution": stats.split_counts,
            "total_time_points": stats.total_time_points,
            "processing_times": stats.processing_times
        }
        
        save_json(summary, self.output_dir / "preprocessing_summary.json")
        self.logger.info("Saved preprocessing summary")