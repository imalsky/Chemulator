#!/usr/bin/env python3
"""
Data Preprocessing Module
=========================
Converts HDF5 trajectory data to NPZ shards with normalization statistics.

This module handles the complete preprocessing pipeline:
1. Scans HDF5 files to identify valid trajectories
2. Validates time grid consistency across all data
3. Writes NPZ shards with deterministic train/validation/test splits
4. Computes normalization statistics from training data only
5. Generates centralized dt normalization specification

Output Structure:
- NPZ shards: Organized by split (train/validation/test)
- normalization.json: Statistics and methods for all variables
- preprocess_report.json: Processing metrics and timing
- shard_index.json: Shard metadata for dataset loading
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

# Configure multiprocessing start method
try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=False)
except Exception:
    pass

# Import all utilities from preprocessor_utils
from preprocessor_utils import (
    TQDM_MININTERVAL, TQDM_SMOOTHING, TQDM_LEAVE_OUTER, TQDM_LEAVE_INNER,
    PARALLEL_SCAN_TIMEOUT_PER_FILE, PARALLEL_SCAN_OVERHEAD,
    TIME_DECREASE_TOLERANCE, ALLOW_EQUAL_TIMEPOINTS,
    DEFAULT_HDF5_CHUNK_SIZE, SHARD_FILENAME_FORMAT,
    format_bytes, ensure_directory, load_config_value, get_storage_dtype,
    deterministic_hash, WelfordAccumulator, RunningStatistics,
    get_normalization_flags, scan_hdf5_file_worker
)


class DataPreprocessor:
    """
    Main preprocessing pipeline for converting HDF5 to NPZ shards.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.cfg = config
        self.logger = logger or logging.getLogger("preprocessor")
        
        # Extract configuration
        self._load_configuration()
        self._validate_configuration()
        
        # Initialize state
        self._drop_report = {
            "files": [],
            "overall": {
                "n_total": 0,
                "n_valid": 0,
                "n_dropped": 0,
                "n_nan": 0,
                "n_below_threshold": 0
            }
        }
        self._valid_group_names = {}
        self._canonical_time = None
    
    def _load_configuration(self) -> None:
        """Load configuration parameters."""
        # Paths
        self.raw_files = list(load_config_value(
            self.cfg, ["paths", "raw_data_files"], required=True
        ))
        self.processed_dir = Path(load_config_value(
            self.cfg, ["paths", "processed_data_dir"], required=True
        ))
        
        # Data schema
        data_cfg = self.cfg.get("data", {})
        self.species_vars = list(load_config_value(
            data_cfg, ["species_variables"], required=True
        ))
        self.target_species_vars = list(
            data_cfg.get("target_species_variables", self.species_vars)
        )
        self.global_vars = list(load_config_value(
            data_cfg, ["global_variables"], required=True
        ))
        self.time_key = str(load_config_value(
            data_cfg, ["time_variable"], required=True
        ))
        
        # Normalization
        norm_cfg = self.cfg.get("normalization", {})
        self.default_method = str(norm_cfg["default_method"])
        self.methods = dict(norm_cfg["methods"])
        self.epsilon = float(norm_cfg["epsilon"])
        self.min_std = float(norm_cfg["min_std"])
        self.clamp_value = float(norm_cfg["clamp_value"])
        
        # Preprocessing
        preproc_cfg = self.cfg.get("preprocessing", {})
        self.npz_compressed = bool(load_config_value(
            preproc_cfg, ["npz_compressed"], required=True
        ))
        self.traj_per_shard = int(load_config_value(
            preproc_cfg, ["trajectories_per_shard"], required=True
        ))
        self.hdf5_chunk_size = int(load_config_value(
            preproc_cfg, ["hdf5_chunk_size"], default=DEFAULT_HDF5_CHUNK_SIZE
        ))
        self.min_value_threshold = float(load_config_value(
            preproc_cfg, ["min_value_threshold"], required=True
        ))
        
        # Worker configuration
        requested_workers = int(load_config_value(
            preproc_cfg, ["num_workers"], default=0
        ))
        
        if len(self.raw_files) <= 2:
            self.num_workers = 0
            if requested_workers > 0:
                self.logger.info(
                    f"Using serial mode for {len(self.raw_files)} files "
                    f"(parallel overhead not justified)"
                )
        else:
            if requested_workers > 0:
                self.num_workers = min(requested_workers, len(self.raw_files))
            else:
                cpu_count = os.cpu_count() or 1
                self.num_workers = min(4, cpu_count, len(self.raw_files))
        
        # Training configuration
        train_cfg = self.cfg.get("training", {})
        self.val_fraction = float(load_config_value(
            train_cfg, ["val_fraction"], required=True
        ))
        self.test_fraction = float(load_config_value(
            train_cfg, ["test_fraction"], required=True
        ))
        self.use_fraction = float(train_cfg.get("use_fraction", 1.0))
        self.min_steps = int(load_config_value(
            train_cfg, ["min_steps"], required=True
        ))
        self.max_steps = int(load_config_value(
            train_cfg, ["max_steps"], required=True
        ))
        
        # Storage dtype
        self.storage_dtype = get_storage_dtype(self.cfg)
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if len(self.raw_files) == 0:
            raise ValueError("No raw data files specified")
        
        if self.processed_dir.exists():
            if any(self.processed_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {self.processed_dir}"
                )
            else:
                self.logger.warning(f"Output directory exists but is empty: {self.processed_dir}")
        else:
            ensure_directory(self.processed_dir)
        
        if not (0.0 <= self.use_fraction <= 1.0):
            raise ValueError("use_fraction must be in [0, 1]")
        
        if not (0.0 <= self.val_fraction <= 1.0):
            raise ValueError("val_fraction must be in [0, 1]")
        
        if not (0.0 <= self.test_fraction <= 1.0):
            raise ValueError("test_fraction must be in [0, 1]")
        
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")
        
        if "default_method" not in self.cfg["normalization"]:
            raise KeyError("normalization.default_method is required")
        
        if "methods" not in self.cfg["normalization"]:
            raise KeyError("normalization.methods is required")
    
    def run(self) -> None:
        """Execute the preprocessing pipeline."""
        if h5py is None:
            raise RuntimeError("h5py is required but not installed")
        
        start_time = time.time()
        self.logger.info(f"Starting preprocessing of {len(self.raw_files)} HDF5 files")
        self.logger.info(f"Parallel scan workers: {self.num_workers}")
        
        # Phase 1: Scan and validate files
        scan_start = time.time()
        self._scan_files()
        scan_end = time.time()
        
        if self._canonical_time is None:
            raise RuntimeError("No valid trajectories found")
        
        # Log memory estimate
        self._log_memory_estimate()
        
        # Phase 2: Compute dt specification
        dt_start = time.time()
        dt_spec = self._compute_dt_specification(self._canonical_time)
        dt_end = time.time()
        
        # Phase 3: Write shards and collect statistics
        shard_start = time.time()
        train_stats = self._write_shards_and_collect_stats()
        shard_end = time.time()
        
        # Phase 4: Write normalization manifest
        manifest_start = time.time()
        manifest = self._finalize_manifest(train_stats, dt_spec)
        manifest_path = self.processed_dir / "normalization.json"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Wrote normalization manifest: {manifest_path}")
        manifest_end = time.time()
        
        # Phase 5: Write reports
        self._write_reports(
            start_time, scan_start, scan_end,
            dt_start, dt_end, shard_start,
            shard_end, manifest_start, manifest_end
        )
    
    def _scan_files(self) -> None:
        """Scan HDF5 files to identify valid trajectories."""
        time_candidates = []
        
        if self.num_workers > 1:
            # Parallel scanning
            self._scan_files_parallel(time_candidates)
        else:
            # Serial scanning
            self._scan_files_serial(time_candidates)
        
        # Validate canonical time grid
        self._validate_canonical_time(time_candidates)
    
    def _scan_files_parallel(self, time_candidates: List[np.ndarray]) -> None:
        """Scan files in parallel using process pool with detailed progress reporting."""
        total_timeout = len(self.raw_files) * PARALLEL_SCAN_TIMEOUT_PER_FILE + PARALLEL_SCAN_OVERHEAD
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                self.logger.info(f"Starting parallel scan with {self.num_workers} workers")
                
                # Submit all jobs
                futures = {}
                for file_idx, path in enumerate(self.raw_files, 1):
                    self.logger.info(f"Submitting file {file_idx}/{len(self.raw_files)}: {Path(path).name}")
                    future = executor.submit(
                        scan_hdf5_file_worker,
                        path,
                        self.species_vars,
                        self.global_vars,
                        self.time_key,
                        self.min_value_threshold,
                        self.epsilon,
                        self.hdf5_chunk_size,
                    )
                    futures[future] = path
                
                # Process completed files
                completed = 0
                for future in as_completed(futures, timeout=total_timeout):
                    file_path = futures[future]
                    file_name = Path(file_path).name
                    completed += 1
                    
                    try:
                        file_report, valid_groups, time_candidate, progress_stats = future.result(timeout=None)
                        
                        # Log detailed results for this file
                        self.logger.info(
                            f"Completed {completed}/{len(self.raw_files)}: {file_name} - "
                            f"{progress_stats['total_groups']} groups scanned"
                        )
                        self.logger.info(
                            f"  Results: {progress_stats['groups_valid']} valid, "
                            f"{progress_stats['groups_dropped']} dropped "
                            f"({file_report['n_nan']} invalid data, "
                            f"{file_report['n_below_threshold']} below threshold)"
                        )
                        
                        # Update global statistics
                        self._drop_report["files"].append(file_report)
                        for key in ("n_total", "n_valid", "n_dropped", "n_nan", "n_below_threshold"):
                            self._drop_report["overall"][key] += file_report[key]
                        
                        self._valid_group_names[str(file_path)] = valid_groups
                        
                        if time_candidate is not None:
                            time_candidates.append(time_candidate)
                        
                        # Log running total
                        self.logger.info(
                            f"  Running total: {self._drop_report['overall']['n_valid']} valid / "
                            f"{self._drop_report['overall']['n_total']} total trajectories"
                        )
                        
                    except TimeoutError:
                        self.logger.error(f"Timeout processing {file_name}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Error processing {file_name}: {e}")
                        raise
                
                if completed < len(self.raw_files):
                    raise RuntimeError(f"Only processed {completed}/{len(self.raw_files)} files")
                
                self.logger.info("Parallel scan completed successfully")
                
        except Exception as e:
            self.logger.warning(f"Parallel processing failed ({e}), falling back to serial mode")
            # Reset and retry serially
            self._drop_report["files"].clear()
            self._drop_report["overall"] = {
                "n_total": 0, "n_valid": 0, "n_dropped": 0,
                "n_nan": 0, "n_below_threshold": 0
            }
            self._valid_group_names.clear()
            time_candidates.clear()
            self.num_workers = 0
            self._scan_files_serial(time_candidates)
    
    def _scan_files_serial(self, time_candidates: List[np.ndarray]) -> None:
        """Scan files serially with detailed progress reporting."""
        for file_idx, path in enumerate(self.raw_files, 1):
            path_obj = Path(path)
            self.logger.info(f"Processing file {file_idx}/{len(self.raw_files)}: {path_obj.name}")
            
            # Initialize counters for this file
            file_total = 0
            file_valid = 0
            file_nan = 0
            file_below = 0
            time_candidate = None
            valid_groups = []
            
            # Open and process HDF5 file
            with h5py.File(path, "r") as hdf:
                group_names = list(hdf.keys())
                total_groups = len(group_names)
                
                self.logger.info(f"  Found {total_groups} groups in {path_obj.name}")
                
                # Process each group with progress reporting
                for group_idx, group_name in enumerate(group_names, 1):
                    group = hdf[group_name]
                    file_total += 1
                    
                    # Log progress every 10% or every 10000 groups, whichever is smaller
                    log_interval = min(10000, max(1, total_groups // 10))
                    if group_idx % log_interval == 0 or group_idx == total_groups:
                        self.logger.info(
                            f"  Processing groups: {group_idx}/{total_groups} "
                            f"({100.0 * group_idx / total_groups:.1f}%) - "
                            f"valid: {len(valid_groups)}, dropped: {file_total - len(valid_groups)}"
                        )
                    
                    # Validate time variable
                    if self.time_key not in group:
                        file_nan += 1
                        continue
                    
                    time_data = np.array(group[self.time_key], dtype=np.float64, copy=False).reshape(-1)
                    
                    if time_data.size < 2 or not np.all(np.isfinite(time_data)):
                        file_nan += 1
                        continue
                    
                    # Check time monotonicity
                    time_diffs = np.diff(time_data)
                    if ALLOW_EQUAL_TIMEPOINTS:
                        if np.any(time_diffs < -abs(TIME_DECREASE_TOLERANCE)):
                            file_nan += 1
                            continue
                    else:
                        if not np.all(time_diffs > 0.0):
                            file_nan += 1
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
                            [float(group.attrs[key]) for key in self.global_vars],
                            dtype=np.float64
                        )
                        if not np.all(np.isfinite(global_values)):
                            file_nan += 1
                            continue
                    except Exception:
                        file_nan += 1
                        continue
                    
                    # Validate species variables
                    T = int(time_data.shape[0])
                    has_nan = False
                    below_threshold = False
                    
                    for species_name in self.species_vars:
                        if species_name not in group:
                            has_nan = True
                            break
                        
                        dataset = group[species_name]
                        if dataset.shape[0] != T:
                            has_nan = True
                            break
                        
                        # Read in chunks for efficiency
                        chunk_len = self.hdf5_chunk_size if self.hdf5_chunk_size > 0 else T
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
                            
                            if (chunk_data < self.min_value_threshold).any():
                                below_threshold = True
                                break
                        
                        if has_nan or below_threshold:
                            break
                    
                    if has_nan:
                        file_nan += 1
                        continue
                    elif below_threshold:
                        file_below += 1
                        continue
                    
                    file_valid += 1
                    valid_groups.append(group_name)
            
            # Log file summary
            self.logger.info(
                f"  File summary: {file_valid} valid, {file_total - file_valid} dropped "
                f"({file_nan} invalid data, {file_below} below threshold)"
            )
            
            # Update global report
            file_report = {
                "path": str(path_obj),
                "n_total": file_total,
                "n_valid": file_valid,
                "n_dropped": file_total - file_valid,
                "n_nan": file_nan,
                "n_below_threshold": file_below,
            }
            
            self._drop_report["files"].append(file_report)
            for key in ("n_total", "n_valid", "n_dropped", "n_nan", "n_below_threshold"):
                self._drop_report["overall"][key] += file_report[key]
            
            self._valid_group_names[str(path)] = valid_groups
            
            if time_candidate is not None:
                time_candidates.append(time_candidate)
    
    def _validate_canonical_time(self, time_candidates: List[np.ndarray]) -> None:
        """Validate that all files share the same time grid."""
        time_candidates = [t for t in time_candidates if t is not None and t.size > 0]
        
        if len(time_candidates) == 0:
            self._canonical_time = None
        else:
            canonical = time_candidates[0]
            for time_grid in time_candidates[1:]:
                if not np.array_equal(time_grid, canonical):
                    raise ValueError(
                        "Time grid differs across files. "
                        "All trajectories must have identical time grids."
                    )
            self._canonical_time = canonical.copy()
        
        self.logger.info(
            f"Scan complete: {self._drop_report['overall']['n_valid']} valid trajectories "
            f"out of {self._drop_report['overall']['n_total']} total"
        )
    
    def _log_memory_estimate(self) -> None:
        """Log estimated memory usage for processing."""
        try:
            N = int(self._drop_report["overall"]["n_valid"])
            T = int(self._canonical_time.shape[0])
            S = int(len(self.species_vars))
            itemsize = int(np.dtype(self.storage_dtype).itemsize)
            
            estimated_memory = int(self._canonical_time.nbytes) + N * T * S * itemsize
            self.logger.info(f"Estimated memory for sharding: {format_bytes(estimated_memory)}")
        except Exception:
            pass
    
    def _compute_dt_specification(self, time_grid: np.ndarray) -> Dict[str, float]:
        """
        Compute centralized dt normalization specification.
        
        Args:
            time_grid: Canonical time grid
            
        Returns:
            Dictionary with dt normalization parameters
        """
        time_grid = np.asarray(time_grid, dtype=np.float64).reshape(-1)
        T = int(time_grid.shape[0])
        
        if T < 2:
            raise ValueError("Time grid must have at least 2 points")
        
        diffs = np.diff(time_grid)
        if not np.all(diffs > 0.0):
            raise ValueError("Time grid must be strictly increasing")
        
        min_steps = max(1, self.min_steps)
        max_steps = max(min_steps, min(self.max_steps, T - 1))
        
        log_min_all = math.inf
        log_max_all = -math.inf
        
        for k in range(min_steps, max_steps + 1):
            dt_values = time_grid[k:] - time_grid[:-k]
            
            if np.any(dt_values <= 0.0):
                raise ValueError("Non-positive dt encountered")
            
            dt_min = float(np.min(dt_values))
            dt_max = float(np.max(dt_values))
            
            dt_min_clipped = max(dt_min, self.epsilon)
            dt_max_clipped = max(dt_max, self.epsilon)
            
            log_min_all = min(log_min_all, float(np.log10(dt_min_clipped)))
            log_max_all = max(log_max_all, float(np.log10(dt_max_clipped)))
        
        if not np.isfinite(log_min_all) or not np.isfinite(log_max_all):
            raise RuntimeError("Failed to compute valid dt bounds")
        
        if log_max_all <= log_min_all:
            raise RuntimeError("Invalid dt bounds (log_max <= log_min)")
        
        return {
            "method": "log-min-max",
            "log_min": float(log_min_all),
            "log_max": float(log_max_all)
        }
    
    def _write_shards_and_collect_stats(self) -> Dict[str, RunningStatistics]:
        """
        Write NPZ shards and collect training statistics.
        
        Returns:
            Dictionary of statistics for each variable
        """
        seed = int(self.cfg.get("system", {}).get("seed", 42))
        
        # Initialize statistics accumulators for training data only
        all_keys = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        need_flags = {
            key: get_normalization_flags(self.methods.get(key, self.default_method))
            for key in all_keys
        }
        
        train_stats = {
            key: RunningStatistics(
                need_mean_std=flags[0],
                need_min_max=flags[1],
                need_log=flags[2],
                epsilon=self.epsilon
            )
            for key, flags in need_flags.items()
        }
        
        # Create split directories
        split_dirs = {
            "train": self.processed_dir / "train",
            "validation": self.processed_dir / "validation",
            "test": self.processed_dir / "test",
        }
        for directory in split_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        shard_counts = {"train": 0, "validation": 0, "test": 0}
        split_counts = {"train": 0, "validation": 0, "test": 0}
        shards_metadata = {"train": [], "validation": [], "test": []}
        
        # Process each file
        outer_progress = tqdm(
            total=len(self.raw_files),
            desc="Writing shards",
            unit="file",
            smoothing=TQDM_SMOOTHING,
            mininterval=TQDM_MININTERVAL,
            leave=TQDM_LEAVE_OUTER
        )
        
        for path_str in self.raw_files:
            path = Path(path_str)
            valid_groups = self._valid_group_names.get(str(path), [])
            
            if not valid_groups:
                outer_progress.update(1)
                continue
            
            # Process file
            self._process_file_to_shards(
                path, valid_groups, seed,
                split_dirs, shard_counts, split_counts,
                shards_metadata, train_stats
            )
            
            outer_progress.update(1)
        
        outer_progress.close()
        
        # Write metadata files
        self._write_shard_index(shards_metadata)
        self._write_preprocessing_summary(split_counts)
        
        self.logger.info(
            f"Wrote shards - train: {split_counts['train']}, "
            f"validation: {split_counts['validation']}, "
            f"test: {split_counts['test']}"
        )
        
        return train_stats
    
    def _process_file_to_shards(
        self,
        path: Path,
        valid_groups: List[str],
        seed: int,
        split_dirs: Dict[str, Path],
        shard_counts: Dict[str, int],
        split_counts: Dict[str, int],
        shards_metadata: Dict[str, List],
        train_stats: Dict[str, RunningStatistics]
    ) -> None:
        """Process single HDF5 file and write to shards."""
        # Deterministic shuffling
        rng = np.random.default_rng(seed ^ (hash(path.name) & 0xFFFFFFFF))
        order = rng.permutation(len(valid_groups))
        groups_ordered = [valid_groups[i] for i in order]
        
        # Progress bar for groups
        group_progress = tqdm(
            total=len(groups_ordered),
            desc=f"{path.name}",
            unit="group",
            smoothing=TQDM_SMOOTHING,
            mininterval=TQDM_MININTERVAL,
            leave=TQDM_LEAVE_INNER
        )
        
        # Buffers for batching
        buffers = {
            "train": {"x0": [], "g": [], "t": [], "y": []},
            "validation": {"x0": [], "g": [], "t": [], "y": []},
            "test": {"x0": [], "g": [], "t": [], "y": []},
        }
        
        with h5py.File(path, "r") as hdf:
            for group_name in groups_ordered:
                group = hdf[group_name]
                
                # Determine split
                hash_val = deterministic_hash(f"{group_name}:split", seed)
                if hash_val < self.test_fraction:
                    split = "test"
                elif hash_val < self.test_fraction + self.val_fraction:
                    split = "validation"
                else:
                    split = "train"
                
                # Apply use_fraction filter
                if self.use_fraction < 1.0:
                    if deterministic_hash(f"{group_name}:use", seed) >= self.use_fraction:
                        group_progress.update(1)
                        continue
                
                # Load and validate data
                time_data = np.array(group[self.time_key], dtype=np.float64, copy=False).reshape(-1)
                T = int(time_data.shape[0])
                
                # Load species matrix
                species_matrix = self._read_species_matrix(group, T)
                
                if not np.isfinite(species_matrix).all():
                    group_progress.update(1)
                    continue
                
                if (species_matrix < self.min_value_threshold).any():
                    group_progress.update(1)
                    continue
                
                # Load globals
                global_values = np.array(
                    [float(group.attrs[key]) for key in self.global_vars],
                    dtype=np.float64
                )
                
                if not np.isfinite(global_values).all():
                    group_progress.update(1)
                    continue
                
                # Initial condition
                x0 = species_matrix[0, :].astype(self.storage_dtype, copy=False)
                
                # Add to buffer
                buffers[split]["x0"].append(x0)
                buffers[split]["g"].append(global_values.astype(self.storage_dtype, copy=False))
                buffers[split]["t"].append(time_data.astype(self.storage_dtype, copy=False))
                buffers[split]["y"].append(species_matrix.astype(self.storage_dtype, copy=False))
                split_counts[split] += 1
                
                # Update training statistics
                if split == "train":
                    self._update_training_stats(
                        train_stats, time_data, global_values, species_matrix
                    )
                
                # Flush buffers if full
                for split_name in ("train", "validation", "test"):
                    buffer = buffers[split_name]
                    if len(buffer["x0"]) >= self.traj_per_shard:
                        self._write_shard(
                            split_dirs[split_name],
                            buffer,
                            path.stem[:30],
                            split_name,
                            shard_counts[split_name]
                        )
                        shards_metadata[split_name].append({
                            "filename": f"shard_{split_name}_{path.stem[:30]}_{shard_counts[split_name]:05d}.npz",
                            "n_trajectories": self.traj_per_shard
                        })
                        shard_counts[split_name] += 1
                        for key in buffer.keys():
                            buffer[key].clear()
                
                group_progress.update(1)
            
            # Flush remaining buffers
            for split_name in ("train", "validation", "test"):
                buffer = buffers[split_name]
                if buffer["x0"]:
                    self._write_shard(
                        split_dirs[split_name],
                        buffer,
                        path.stem[:30],
                        split_name,
                        shard_counts[split_name]
                    )
                    shards_metadata[split_name].append({
                        "filename": f"shard_{split_name}_{path.stem[:30]}_{shard_counts[split_name]:05d}.npz",
                        "n_trajectories": len(buffer["x0"])
                    })
                    shard_counts[split_name] += 1
                    for key in buffer.keys():
                        buffer[key].clear()
        
        group_progress.close()
    
    def _read_species_matrix(self, group, T: int) -> np.ndarray:
        """Read species data into matrix form."""
        S = len(self.species_vars)
        matrix = np.empty((T, S), dtype=np.float64)
        
        for species_idx, species_name in enumerate(self.species_vars):
            dataset = group[species_name]
            
            if int(dataset.shape[0]) != T:
                raise ValueError("Species length mismatch")
            
            # Read in chunks
            chunk_len = self.hdf5_chunk_size if self.hdf5_chunk_size > 0 else T
            position = 0
            
            for start_idx in range(0, T, chunk_len):
                end_idx = min(T, start_idx + chunk_len)
                chunk = np.array(dataset[start_idx:end_idx], dtype=np.float64, copy=False).reshape(-1)
                matrix[start_idx:end_idx, species_idx] = chunk
                position = end_idx
            
            if position != T:
                raise RuntimeError("Incomplete read of species data")
        
        return matrix
    
    def _update_training_stats(
        self,
        train_stats: Dict[str, RunningStatistics],
        time_data: np.ndarray,
        global_values: np.ndarray,
        species_matrix: np.ndarray
    ) -> None:
        """Update training statistics with new data."""
        # Time statistics
        train_stats[self.time_key].update(time_data[None, :])
        
        # Global statistics
        for global_idx, global_name in enumerate(self.global_vars):
            value = np.array([global_values[global_idx]], dtype=np.float64)
            train_stats[global_name].update(value)
        
        # Species statistics
        for species_idx, species_name in enumerate(self.species_vars):
            data = species_matrix[:, species_idx]
            train_stats[species_name].update(data)
    
    def _write_shard(
        self,
        output_dir: Path,
        buffer: Dict[str, List[np.ndarray]],
        file_tag: str,
        split_name: str,
        shard_idx: int
    ) -> None:
        """Write buffer contents to NPZ shard."""
        # Stack arrays
        x0 = np.stack(buffer["x0"], axis=0)
        globals_array = (
            np.stack(buffer["g"], axis=0) if self.global_vars
            else np.zeros((len(buffer["x0"]), 0), dtype=self.storage_dtype)
        )
        y = np.stack(buffer["y"], axis=0)
        
        # Validate time consistency
        if len(buffer["t"]) == 0:
            raise RuntimeError("Empty buffer")
        
        time_ref = np.asarray(buffer["t"][0])
        for idx, time_array in enumerate(buffer["t"][1:], start=1):
            if not np.array_equal(time_ref, time_array):
                # Find first difference
                diff_indices = np.where(time_ref != time_array)[0]
                first_diff = int(diff_indices[0]) if diff_indices.size else -1
                raise RuntimeError(
                    f"Time grid mismatch in shard: trajectory 0 vs {idx} "
                    f"differ at index {first_diff}"
                )
        
        # Use 1D time vector
        time_1d = time_ref
        
        # Write NPZ
        output_path = output_dir / SHARD_FILENAME_FORMAT.format(
            split=split_name,
            filetag=file_tag,
            idx=shard_idx
        )
        
        if self.npz_compressed:
            np.savez_compressed(
                output_path,
                x0=x0,
                globals=globals_array,
                t_vec=time_1d,
                y_mat=y
            )
        else:
            np.savez(
                output_path,
                x0=x0,
                globals=globals_array,
                t_vec=time_1d,
                y_mat=y
            )
    
    def _finalize_manifest(
        self,
        train_stats: Dict[str, RunningStatistics],
        dt_spec: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create final normalization manifest."""
        per_key_stats = {}
        for key, accumulator in train_stats.items():
            per_key_stats[key] = accumulator.to_manifest(self.min_std)
        
        methods = {}
        all_keys = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        for key in all_keys:
            methods[key] = self.methods.get(key, self.default_method)
        
        return {
            "per_key_stats": per_key_stats,
            "normalization_methods": methods,
            "epsilon": self.epsilon,
            "min_std": self.min_std,
            "clamp_value": self.clamp_value,
            "dt": dt_spec
        }
    
    def _write_shard_index(self, shards_metadata: Dict[str, List]) -> None:
        """Write shard index file."""
        T = int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0
        
        shard_index = {
            "sequence_mode": True,
            "variable_length": False,
            "M_per_sample": T,
            "n_input_species": len(self.species_vars),
            "n_target_species": len(self.target_species_vars),
            "n_globals": len(self.global_vars),
            "compression": "npz",
            "splits": {
                key: {
                    "shards": value,
                    "n_trajectories": sum(x["n_trajectories"] for x in value)
                }
                for key, value in shards_metadata.items()
            }
        }
        
        with open(self.processed_dir / "shard_index.json", "w", encoding="utf-8") as f:
            json.dump(shard_index, f, indent=2)
        
        self.logger.info("Wrote shard_index.json")
    
    def _write_preprocessing_summary(self, split_counts: Dict[str, int]) -> None:
        """Write preprocessing summary file."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "valid_trajectories": {
                "train": int(split_counts["train"]),
                "validation": int(split_counts["validation"]),
                "test": int(split_counts["test"]),
            },
            "overall_from_scan": self._drop_report["overall"],
            "time_grid_len": int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0
        }
        
        with open(self.processed_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Wrote preprocessing_summary.json")
    
    def _write_reports(
        self,
        start_time: float,
        scan_start: float,
        scan_end: float,
        dt_start: float,
        dt_end: float,
        shard_start: float,
        shard_end: float,
        manifest_start: float,
        manifest_end: float
    ) -> None:
        """Write final preprocessing report."""
        report = {
            "min_value_threshold": self.min_value_threshold,
            "overall_from_scan": self._drop_report["overall"],
            "timings_sec": {
                "scan_phase": round(scan_end - scan_start, 3),
                "dt_computation": round(dt_end - dt_start, 3),
                "shard_writing": round(shard_end - shard_start, 3),
                "manifest_writing": round(manifest_end - manifest_start, 3),
                "total": round(manifest_end - start_time, 3),
            },
            "split_fractions": {
                "train": float(1.0 - self.val_fraction - self.test_fraction),
                "validation": float(self.val_fraction),
                "test": float(self.test_fraction),
            },
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(manifest_end))
        }
        
        with open(self.processed_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Wrote preprocess_report.json")