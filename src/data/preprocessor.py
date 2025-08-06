#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with sequence mode support for LiLaN.
Improved with chunked HDF5 reading and comprehensive data statistics logging.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

import h5py
import numpy as np
import torch
import json

from data.normalizer import NormalizationHelper
from utils.utils import save_json, load_json


@dataclass
class DataStatistics:
    """Track preprocessing statistics."""
    total_groups: int = 0
    valid_trajectories: int = 0
    dropped_missing_keys: int = 0
    dropped_non_finite: int = 0
    dropped_below_threshold: int = 0
    dropped_insufficient_time: int = 0
    dropped_use_fraction: int = 0
    total_time_points: int = 0
    species_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    global_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range: Tuple[float, float] = (float('inf'), float('-inf'))
    split_distribution: Dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0, "test": 0})
    processing_times: Dict[str, float] = field(default_factory=dict)
    file_stats: Dict[str, Dict] = field(default_factory=dict)


class DataStatisticsLogger:
    """Log comprehensive data statistics during preprocessing."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stats = DataStatistics()
        self.logger = logging.getLogger(__name__)
        
    def update_species_range(self, var: str, data: np.ndarray):
        """Update min/max for species variable."""
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            vmin, vmax = float(finite_data.min()), float(finite_data.max())
            if var in self.stats.species_min_max:
                old_min, old_max = self.stats.species_min_max[var]
                self.stats.species_min_max[var] = (min(old_min, vmin), max(old_max, vmax))
            else:
                self.stats.species_min_max[var] = (vmin, vmax)
    
    def update_global_range(self, var: str, value: float):
        """Update min/max for global variable."""
        if var in self.stats.global_min_max:
            old_min, old_max = self.stats.global_min_max[var]
            self.stats.global_min_max[var] = (min(old_min, value), max(old_max, value))
        else:
            self.stats.global_min_max[var] = (value, value)
    
    def update_time_range(self, times: np.ndarray):
        """Update time range statistics."""
        if len(times) > 0:
            tmin, tmax = float(times.min()), float(times.max())
            self.stats.time_range = (
                min(self.stats.time_range[0], tmin),
                max(self.stats.time_range[1], tmax)
            )
            self.stats.total_time_points += len(times)
    
    def save_summary(self):
        """Save statistics summary to file."""
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
        
        # Log key statistics
        self.logger.info(f"Preprocessing Summary:")
        self.logger.info(f"  Total groups: {self.stats.total_groups}")
        self.logger.info(f"  Valid trajectories: {self.stats.valid_trajectories}")
        self.logger.info(f"  Dropped: {sum(v for k, v in summary['dropped_counts'].items())}")
        self.logger.info(f"  Split distribution: {self.stats.split_distribution}")


class ChunkedHDF5Reader:
    """Read HDF5 data in chunks to minimize memory usage."""
    def __init__(self, file_path: Path, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
    def read_dataset_chunked(self, group: h5py.Group, var_name: str, 
                            indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Read dataset in chunks, optionally at specific indices."""
        if var_name not in group:
            raise KeyError(f"Variable {var_name} not found in group")
            
        dataset = group[var_name]
        total_size = dataset.shape[0]
        
        if indices is not None:
            # Direct fancy indexing for small selections
            if len(indices) < self.chunk_size:
                return dataset[indices]
            
            # Chunked reading for large selections
            result = np.empty(len(indices), dtype=dataset.dtype)
            for i in range(0, len(indices), self.chunk_size):
                chunk_indices = indices[i:i + self.chunk_size]
                result[i:i + len(chunk_indices)] = dataset[chunk_indices]
            return result
        
        # Read full dataset in chunks
        if total_size <= self.chunk_size:
            return dataset[:]
        
        chunks = []
        for start in range(0, total_size, self.chunk_size):
            end = min(start + self.chunk_size, total_size)
            chunks.append(dataset[start:end])
        return np.concatenate(chunks)
    
    def stream_group_data(self, group: h5py.Group, variables: List[str], 
                         chunk_size: Optional[int] = None):
        """Stream data from group in chunks."""
        if not variables:
            return
            
        chunk_size = chunk_size or self.chunk_size
        first_var = next((v for v in variables if v in group), None)
        if not first_var:
            return
            
        total_size = group[first_var].shape[0]
        
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            chunk_data = {}
            for var in variables:
                if var in group:
                    chunk_data[var] = group[var][start:end]
            yield start, end, chunk_data


class SequenceShardWriter:
    """Writes trajectory sequences to NPZ shards with improved buffering."""
    def __init__(self, output_dir: Path, trajectories_per_shard: int,
                 shard_idx_base: str, M: int, n_species: int, n_globals: int,
                 dtype: np.dtype, compressed: bool = True):
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
    
    def add_trajectory(self, x0_log: np.ndarray, globals_vec: np.ndarray,
                      t_vec: np.ndarray, y_mat: np.ndarray):
        """Add trajectory to buffer."""
        trajectory = {
            "x0_log": x0_log.astype(self.dtype, copy=False),
            "globals": globals_vec.astype(self.dtype, copy=False),
            "t_vec": t_vec.astype(self.dtype, copy=False),
            "y_mat": y_mat.astype(self.dtype, copy=False),
        }
        self.buffer.append(trajectory)
        
        if len(self.buffer) >= self.trajectories_per_shard:
            self._write_shard()
    
    def _write_shard(self):
        """Write buffered trajectories to NPZ file."""
        if not self.buffer:
            return
        
        # Stack all trajectories
        x0_log = np.stack([t["x0_log"] for t in self.buffer])
        globals_vec = np.stack([t["globals"] for t in self.buffer])
        t_vec = np.stack([t["t_vec"] for t in self.buffer])
        y_mat = np.stack([t["y_mat"] for t in self.buffer])
        
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        
        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(filepath, x0_log=x0_log, globals=globals_vec, t_vec=t_vec, y_mat=y_mat)
        
        self.shard_metadata.append({
            "filename": filename,
            "n_samples": len(self.buffer),
        })
        
        self.buffer = []
        self.shard_id += 1
    
    def flush(self):
        """Write any remaining trajectories."""
        if self.buffer:
            self._write_shard()


class CorePreprocessor:
    """Core preprocessing logic with chunked reading support."""
    def __init__(self, config: Dict[str, Any], norm_stats: Optional[Dict[str, Any]] = None,
                 stats_logger: Optional[DataStatisticsLogger] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stats_logger = stats_logger
        
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]
        
        # Chunking parameters
        self.chunk_size = self.proc_cfg.get("hdf5_chunk_size", 10000)
        
        # Sequence mode settings
        self.M_per_sample = self.data_cfg.get("M_per_sample", 16)
        self.time_sampling = self.data_cfg.get("time_sampling", {"uniform": 1.0})
        
        # Get dtype
        dtype_str = self.system_cfg.get("dtype", "float32")
        self.np_dtype = np.float32 if dtype_str == "float32" else np.float64
        
        # Variables
        self.species_vars = self.data_cfg["species_variables"]
        self.target_species_vars = self.data_cfg.get("target_species_variables", self.species_vars)
        self.global_vars = self.data_cfg["global_variables"]
        self.time_var = self.data_cfg["time_variable"]
        
        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        
        self.min_value_threshold = self.proc_cfg.get("min_value_threshold", 1e-30)
        self.norm_stats = norm_stats or {}
        
        self._create_index_mappings()
        
        if norm_stats:
            self.norm_helper = NormalizationHelper(
                norm_stats, torch.device("cpu"), self.species_vars,
                self.global_vars, self.time_var, config
            )
    

    def _create_index_mappings(self):
            """Create index mappings for robust variable access."""
            # This mapping from variable name to column index is correct.
            self.var_to_idx = {var: i for i, var in enumerate(self.species_vars + self.global_vars + [self.time_var])}
            self.species_indices = [self.var_to_idx[var] for var in self.species_vars]
            self.target_species_indices = [self.var_to_idx[var] for var in self.target_species_vars]
    
    def _sample_times(self, t_min: float, t_max: float) -> np.ndarray:
        """Sample M time points according to time_sampling config."""
        if isinstance(self.time_sampling, dict):
            uniform_frac = self.time_sampling.get("uniform", 0.5)
            log_frac = self.time_sampling.get("log_spaced", 0.5)
            
            total_frac = max(1e-12, uniform_frac + log_frac)
            uniform_frac = uniform_frac / total_frac
            
            n_uniform = int(self.M_per_sample * uniform_frac)
            n_log = self.M_per_sample - n_uniform
            
            times_list = []
            
            if n_uniform > 0:
                times_list.append(np.linspace(t_min, t_max, n_uniform))
            
            if n_log > 0:
                t_min_log = max(t_min, 1e-10)
                if t_max > t_min_log:
                    log_min, log_max = np.log10(t_min_log), np.log10(t_max)
                    times_list.append(np.logspace(log_min, log_max, n_log))
                else:
                    times_list.append(np.linspace(t_min, t_max, n_log))
            
            if times_list:
                times = np.unique(np.sort(np.concatenate(times_list)))
                if len(times) > self.M_per_sample:
                    indices = np.linspace(0, len(times) - 1, self.M_per_sample, dtype=int)
                    times = times[indices]
                elif len(times) < self.M_per_sample:
                    extra = self.M_per_sample - len(times)
                    t_extra = np.linspace(t_min, t_max, extra + 2)[1:-1]
                    times = np.sort(np.concatenate([times, t_extra]))[:self.M_per_sample]
            else:
                times = np.linspace(t_min, t_max, self.M_per_sample)
        else:
            times = np.linspace(t_min, t_max, self.M_per_sample)
        
        return times
    
    def _extract_trajectory_chunked(self, group: h5py.Group, gname: str,
                                   reader: ChunkedHDF5Reader) -> Optional[Tuple[np.ndarray, ...]]:
        """Extract trajectory data using chunked reading."""
        # Check globals
        missing = [k for k in self.global_vars if k not in group.attrs]
        if missing:
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None
        
        try:
            # Read time data
            time_data = reader.read_dataset_chunked(group, self.time_var)
            if not np.all(np.isfinite(time_data)) or time_data.size == 0:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_non_finite += 1
                return None
            
            # Update statistics
            if self.stats_logger:
                self.stats_logger.update_time_range(time_data)
            
            # Read species at t0 for x0
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
            
            # Read target species efficiently
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
                
                # Update statistics
                if self.stats_logger:
                    self.stats_logger.update_species_range(var, arr)
            
            # Globals
            globals_vec = np.array([float(group.attrs[k]) for k in self.global_vars], dtype=self.np_dtype)
            
            # Update global statistics
            if self.stats_logger:
                for i, var in enumerate(self.global_vars):
                    self.stats_logger.update_global_range(var, globals_vec[i])
            
            return time_data, x0, globals_vec, species_mat
            
        except Exception as e:
            self.logger.error(f"Failed to read profile from group '{gname}': {e}")
            return None
    
    def process_file_for_sequence_shards(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process file and write sequence-mode shards with chunked reading."""
        start_time = time.time()
        
        # Calculate trajectories per shard
        trajectories_per_shard = self.proc_cfg.get("trajectories_per_shard", None)
        if trajectories_per_shard is None:
            target_bytes = int(self.proc_cfg.get("target_shard_bytes", 200 * 1024 * 1024))
            bytes_per_val = 4 if self.np_dtype == np.float32 else 8
            vals_per_traj = (self.n_species + self.n_globals + self.M_per_sample + 
                           self.M_per_sample * self.n_target_species)
            traj_bytes = max(bytes_per_val * vals_per_traj, 1)
            trajectories_per_shard = max(1, target_bytes // traj_bytes)
        
        compressed_npz = bool(self.proc_cfg.get("npz_compressed", True))
        
        # Create shard writers
        writers = {}
        for split in ["train", "validation", "test"]:
            writers[split] = SequenceShardWriter(
                output_dir / split, trajectories_per_shard, file_path.stem,
                self.M_per_sample, self.n_species, self.n_globals,
                self.np_dtype, compressed=compressed_npz
            )
        
        split_counts = {"train": 0, "validation": 0, "test": 0}
        reader = ChunkedHDF5Reader(file_path, self.chunk_size)
        
        file_groups_processed = 0
        file_valid = 0
        
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                grp = f[gname]
                file_groups_processed += 1
                
                if self.stats_logger:
                    self.stats_logger.stats.total_groups += 1
                
                # Quick checks
                if self.time_var not in grp:
                    self.logger.debug(f"Skipping group '{gname}': missing time")
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_missing_keys += 1
                    continue
                
                n_t = grp[self.time_var].shape[0]
                if n_t <= self.M_per_sample:
                    self.logger.debug(f"Skipping group '{gname}': insufficient time points")
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_insufficient_time += 1
                    continue
                
                # Use fraction check
                if self.train_cfg.get("use_fraction", 1.0) < 1.0:
                    hash_val = int(hashlib.sha256(gname.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if hash_val >= self.train_cfg["use_fraction"]:
                        if self.stats_logger:
                            self.stats_logger.stats.dropped_use_fraction += 1
                        continue
                
                # Extract trajectory with chunked reading
                extracted = self._extract_trajectory_chunked(grp, gname, reader)
                if extracted is None:
                    continue
                
                time_data, x0, globals_vec, species_mat = extracted
                file_valid += 1
                
                if self.stats_logger:
                    self.stats_logger.stats.valid_trajectories += 1
                
                # Sample times
                t_min, t_max = float(time_data.min()), float(time_data.max())
                sampled_times = self._sample_times(t_min, t_max)
                
                # Find nearest indices
                pos = np.searchsorted(time_data, sampled_times, side="left")
                pos = np.clip(pos, 1, len(time_data) - 1)
                prev = pos - 1
                choose_prev = (np.abs(sampled_times - time_data[prev]) <= 
                             np.abs(time_data[pos] - sampled_times))
                time_indices = np.where(choose_prev, prev, pos).astype(np.int64)
                
                # Prepare data
                eps = self.norm_cfg.get("epsilon", 1e-30)
                x0_log = np.log10(np.maximum(x0, eps))
                y_mat = species_mat[time_indices, :]
                y_mat_log = np.log10(np.maximum(y_mat, eps))
                t_vec = time_data[time_indices]
                
                # Determine split
                split_hash = int(hashlib.sha256((gname + "_split").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                test_frac = self.train_cfg.get("test_fraction", 0.0)
                val_frac = self.train_cfg.get("val_fraction", 0.0)
                
                if split_hash < test_frac:
                    split = "test"
                elif split_hash < test_frac + val_frac:
                    split = "validation"
                else:
                    split = "train"
                
                writers[split].add_trajectory(x0_log, globals_vec, t_vec, y_mat_log)
                split_counts[split] += 1
                
                if self.stats_logger:
                    self.stats_logger.stats.split_distribution[split] += 1
        
        # Flush writers
        for writer in writers.values():
            writer.flush()
        
        # Record file statistics
        if self.stats_logger:
            elapsed = time.time() - start_time
            self.stats_logger.stats.processing_times[str(file_path)] = elapsed
            self.stats_logger.stats.file_stats[str(file_path)] = {
                "groups_processed": file_groups_processed,
                "valid_trajectories": file_valid,
                "processing_time": elapsed
            }
        
        # Collect metadata
        metadata = {"splits": {}}
        for split in ["train", "validation", "test"]:
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
                "total_samples": split_counts[split],
            }
        
        return metadata
    
    def collect_time_stats(self, file_paths: List[Path]) -> Dict[str, float]:
        """Collect global time statistics for normalization using chunked reading."""
        all_times = []
        reader = ChunkedHDF5Reader(file_paths[0] if file_paths else None, self.chunk_size)
        
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var in grp:
                        times = reader.read_dataset_chunked(grp, self.time_var)
                        times = times[times > 1e-10]
                        if len(times) > 0:
                            all_times.append(times)
        
        if not all_times:
            raise ValueError("No valid time data found")
        
        all_times = np.concatenate(all_times)
        tau0 = np.percentile(all_times, 5)
        tau = np.log(1 + all_times / tau0)
        
        return {
            "tau0": float(tau0),
            "tmin": float(tau.min()),
            "tmax": float(tau.max()),
            "time_transform": "log-min-max",
        }


class DataPreprocessor:
    """Main preprocessor for sequence mode with statistics tracking."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        self.sequence_mode = True
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistics logger
        self.stats_logger = DataStatisticsLogger(output_dir)
    
    def process_to_npy_shards(self) -> None:
        """Process data in sequence mode for LiLaN."""
        self.logger.info("Processing data in SEQUENCE MODE for LiLaN multi-time supervision")
        
        # Collect time statistics
        processor = CorePreprocessor(self.config, stats_logger=self.stats_logger)
        time_stats = processor.collect_time_stats(self.raw_files)
        self.logger.info(f"Time normalization stats: tau0={time_stats['tau0']:.3e}, "
                        f"tmin={time_stats['tmin']:.3f}, tmax={time_stats['tmax']:.3f}")
        
        # Collect normalization statistics
        norm_stats = self._collect_normalization_stats()
        norm_stats["time_normalization"] = time_stats
        save_json(norm_stats, self.output_dir / "normalization.json")
        
        # Process files to sequence shards
        all_metadata = {
            "splits": {
                "train": {"shards": [], "n_trajectories": 0},
                "validation": {"shards": [], "n_trajectories": 0},
                "test": {"shards": [], "n_trajectories": 0},
            }
        }
        
        for file_path in self.raw_files:
            self.logger.info(f"Processing file: {file_path}")
            processor = CorePreprocessor(self.config, norm_stats, self.stats_logger)
            metadata = processor.process_file_for_sequence_shards(file_path, self.processed_dir)
            
            # Aggregate metadata
            for split in ["train", "validation", "test"]:
                all_metadata["splits"][split]["shards"].extend(metadata["splits"][split]["shards"])
                all_metadata["splits"][split]["n_trajectories"] += metadata["splits"][split]["n_trajectories"]
        
        # Save shard index
        shard_index = {
            "sequence_mode": True,
            "M_per_sample": self.config["data"]["M_per_sample"],
            "n_input_species": len(self.config["data"]["species_variables"]),
            "n_target_species": len(self.config["data"].get("target_species_variables", 
                                                            self.config["data"]["species_variables"])),
            "n_globals": len(self.config["data"]["global_variables"]),
            "compression": "npz",
            "splits": all_metadata["splits"],
            "time_normalization": time_stats,
        }
        save_json(shard_index, self.output_dir / "shard_index.json")
        
        # Save comprehensive statistics
        self.stats_logger.save_summary()
        
        self.logger.info(
            "Sequence mode preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )
    
    def _collect_normalization_stats(self) -> Dict[str, Any]:
        """Collect normalization statistics using chunked reading."""
        stats: Dict[str, Any] = {"per_key_stats": {}, "normalization_methods": {}}
        accumulators: Dict[str, Dict[str, float]] = {}
        
        for var in self.config["data"]["species_variables"] + self.config["data"]["global_variables"]:
            accumulators[var] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        
        reader = ChunkedHDF5Reader(self.raw_files[0] if self.raw_files else None, 
                                  self.config["preprocessing"].get("hdf5_chunk_size", 10000))
        
        # Process each file
        for file_path in self.raw_files:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    
                    # Species variables
                    for var in self.config["data"]["species_variables"]:
                        if var in grp:
                            data = reader.read_dataset_chunked(grp, var)
                            log_data = np.log10(np.maximum(data, 1e-30))
                            self._update_accumulator(accumulators[var], log_data)
                    
                    # Global variables
                    for var in self.config["data"]["global_variables"]:
                        if var in grp.attrs:
                            val = float(grp.attrs[var])
                            self._update_accumulator(accumulators[var], np.array([val]))
        
        # Finalize statistics
        for var, acc in accumulators.items():
            if acc["count"] > 0:
                mean = acc["mean"]
                variance = acc["m2"] / (acc["count"] - 1) if acc["count"] > 1 else 0.0
                std = float(np.sqrt(variance))
                
                if var in self.config["data"]["species_variables"]:
                    stats["normalization_methods"][var] = "log-standard"
                    stats["per_key_stats"][var] = {
                        "method": "log-standard",
                        "log_mean": float(mean),
                        "log_std": float(max(std, 1e-10)),
                        "min": float(acc["min"]),
                        "max": float(acc["max"]),
                    }
                else:
                    stats["normalization_methods"][var] = "standard"
                    stats["per_key_stats"][var] = {
                        "method": "standard",
                        "mean": float(mean),
                        "std": float(max(std, 1e-10)),
                        "min": float(acc["min"]),
                        "max": float(acc["max"]),
                    }
        
        stats["normalization_methods"][self.config["data"]["time_variable"]] = "log-min-max"
        
        return stats
    
    def _update_accumulator(self, acc: Dict[str, float], data: np.ndarray):
        """Update accumulator with Welford's algorithm for numerical stability."""
        for val in data.flatten():
            if np.isfinite(val):
                n = acc["count"]
                acc["count"] = n + 1
                delta = val - acc["mean"]
                acc["mean"] += delta / acc["count"]
                delta2 = val - acc["mean"]
                acc["m2"] += delta * delta2
                acc["min"] = min(acc["min"], float(val))
                acc["max"] = max(acc["max"], float(val))