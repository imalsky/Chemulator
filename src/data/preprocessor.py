#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with sequence mode support for LiLaN.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import h5py
import numpy as np
import torch
import json
import sys

from .normalizer import DataNormalizer, NormalizationHelper
from utils.utils import save_json, load_json

DEFAULT_EPSILON_MIN = 1e-38
DEFAULT_EPSILON_MAX = 1e38


class SequenceShardWriter:
    """Writes trajectory sequences to NPZ shards."""
    def __init__(self, output_dir: Path, shard_size: int, shard_idx_base: str, 
                 M: int, n_species: int, n_globals: int, dtype: np.dtype):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx_base = shard_idx_base
        self.M = M
        self.n_species = n_species
        self.n_globals = n_globals
        self.dtype = dtype
        self.buffer = []
        self.shard_id = 0
        self.shard_metadata = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_trajectory(self, x0_log: np.ndarray, globals: np.ndarray, 
                      t_vec: np.ndarray, y_mat: np.ndarray):
        """Add a trajectory to buffer."""
        trajectory = {
            'x0_log': x0_log.astype(self.dtype),
            'globals': globals.astype(self.dtype),
            't_vec': t_vec.astype(self.dtype),
            'y_mat': y_mat.astype(self.dtype)
        }
        self.buffer.append(trajectory)
        
        if len(self.buffer) >= self.shard_size:
            self._write_shard()
    
    def _write_shard(self):
        """Write buffered trajectories to NPZ file."""
        if not self.buffer:
            return
            
        # Stack all trajectories
        x0_log = np.stack([t['x0_log'] for t in self.buffer])
        globals = np.stack([t['globals'] for t in self.buffer])
        t_vec = np.stack([t['t_vec'] for t in self.buffer])
        y_mat = np.stack([t['y_mat'] for t in self.buffer])
        
        # Write NPZ
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename
        np.savez_compressed(
            filepath,
            x0_log=x0_log,
            globals=globals,
            t_vec=t_vec,
            y_mat=y_mat
        )
        
        self.shard_metadata.append({
            "filename": filename,
            "n_samples": len(self.buffer)
        })
        
        self.buffer = []
        self.shard_id += 1
    
    def flush(self):
        """Write any remaining trajectories."""
        if self.buffer:
            self._write_shard()


class CorePreprocessor:
    """Core preprocessing logic with sequence mode support."""
    def __init__(self, config: Dict[str, Any], norm_stats: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.pred_cfg = config.get("prediction", {})
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]
        
        # Get sequence mode settings
        self.sequence_mode = self.data_cfg.get("sequence_mode", False)
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
        self.var_order = self.species_vars + self.global_vars + [self.time_var]
        
        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        self.n_vars = self.n_species + self.n_globals + 1
        
        self.min_value_threshold = self.proc_cfg.get("min_value_threshold", 1e-30)
        self.prediction_mode = self.pred_cfg.get("mode", "absolute")
        
        self.normalizer = DataNormalizer(config)
        self.norm_stats = norm_stats or {}
        
        self._create_index_mappings()
        
        if norm_stats:
            self.norm_helper = NormalizationHelper(
                norm_stats, torch.device("cpu"),
                self.species_vars, self.global_vars, self.time_var, config
            )
    
    def _create_index_mappings(self):
        """Create index mappings for robust variable access."""
        self.var_to_idx = {var: i for i, var in enumerate(self.var_order)}
        self.species_indices = [self.var_to_idx[var] for var in self.species_vars]
        self.target_species_indices = [self.var_to_idx[var] for var in self.target_species_vars]
        self.global_indices = [self.var_to_idx[var] for var in self.global_vars]
        self.time_idx = self.var_to_idx[self.time_var]
    
    def _sample_times(self, t_min: float, t_max: float) -> np.ndarray:
        """Sample M time points according to time_sampling config."""
        if isinstance(self.time_sampling, dict):
            # Mixed sampling
            uniform_frac = self.time_sampling.get("uniform", 0.5)
            log_frac = self.time_sampling.get("log_spaced", 0.5)
            
            n_uniform = int(self.M_per_sample * uniform_frac)
            n_log = self.M_per_sample - n_uniform
            
            # Uniform samples
            t_uniform = np.linspace(t_min, t_max, n_uniform)
            
            # Log-spaced samples  
            if n_log > 0:
                log_min = np.log10(max(t_min, 1e-10))
                log_max = np.log10(t_max)
                t_log = np.logspace(log_min, log_max, n_log)
            else:
                t_log = np.array([])
            
            # Combine and sort
            times = np.concatenate([t_uniform, t_log])
            times = np.unique(np.sort(times))
            
            # Ensure we have exactly M samples
            if len(times) > self.M_per_sample:
                times = times[:self.M_per_sample]
            elif len(times) < self.M_per_sample:
                # Pad with linspace
                extra = self.M_per_sample - len(times)
                t_extra = np.linspace(t_min, t_max, extra + 2)[1:-1]
                times = np.sort(np.concatenate([times, t_extra]))
                
        else:
            # Simple uniform sampling
            times = np.linspace(t_min, t_max, self.M_per_sample)
            
        return times
    
    def _is_profile_valid(self, group: h5py.Group) -> Tuple[bool, str]:
        """Check if a profile is valid."""
        required_keys = self.species_vars + [self.time_var]
        if not set(required_keys).issubset(group.keys()):
            return False, "missing_keys"
        
        for var in required_keys:
            try:
                data = group[var][:]
            except Exception:
                return False, "read_error"
            
            if not np.all(np.isfinite(data)):
                return False, "non_finite"
                
            if np.any(data <= self.min_value_threshold):
                return False, "below_threshold"
                
        return True, "valid"
    
    def _extract_profile(self, group: h5py.Group, gname: str, n_t: int) -> Optional[np.ndarray]:
        """Extract a full data profile from an HDF5 group."""
        config_globals = set(self.global_vars)
        group_attrs = set(group.attrs.keys())
        
        if 'SEED' in group_attrs:
            group_attrs.remove('SEED')
        
        globals_missing = config_globals - group_attrs
        attrs_extra = group_attrs - config_globals
        
        if globals_missing or attrs_extra:
            self.logger.critical(f"Global variable mismatch in group '{gname}'")
            sys.exit(1)
        
        globals_dict = {key: float(group.attrs[key]) for key in self.global_vars}
        profile = np.empty((n_t, self.n_vars), dtype=self.np_dtype)
        
        try:
            for i, var_name in enumerate(self.var_order):
                if var_name in group:
                    profile[:, i] = group[var_name][:]
                elif var_name in globals_dict:
                    profile[:, i] = globals_dict[var_name]
                else:
                    self.logger.error(f"Variable '{var_name}' not found")
                    sys.exit(1)
        except Exception as e:
            self.logger.error(f"Failed to read profile: {e}")
            return None
            
        return profile
    
    def process_file_for_sequence_shards(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process file and write sequence-mode shards."""
        # Create shard writers for each split
        writers = {}
        for split in ["train", "validation", "test"]:
            writers[split] = SequenceShardWriter(
                output_dir / split,
                self.proc_cfg["shard_size"] // self.M_per_sample,  # Adjust for sequences
                file_path.stem,
                self.M_per_sample,
                self.n_species,
                self.n_globals,
                self.np_dtype
            )
        
        split_counts = {"train": 0, "validation": 0, "test": 0}
        
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                # Validation
                is_valid, _ = self._is_profile_valid(f[gname])
                if not is_valid:
                    continue
                
                # Use fraction check
                if self.train_cfg["use_fraction"] < 1.0:
                    hash_val = int(hashlib.sha256(gname.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if hash_val >= self.train_cfg["use_fraction"]:
                        continue
                
                # Extract profile
                grp = f[gname]
                n_t = grp[self.time_var].shape[0]
                if n_t <= self.M_per_sample:
                    continue
                    
                profile = self._extract_profile(grp, gname, n_t)
                if profile is None:
                    continue
                
                # Sample M time points
                time_data = profile[:, self.time_idx]
                t_min, t_max = time_data.min(), time_data.max()
                sampled_times = self._sample_times(t_min, t_max)
                
                # Find indices for sampled times
                time_indices = []
                for t in sampled_times:
                    idx = np.argmin(np.abs(time_data - t))
                    time_indices.append(idx)
                time_indices = np.array(time_indices)
                
                # Extract data at sampled times
                x0 = profile[0, self.species_indices]
                globals_vec = profile[0, self.global_indices]
                
                # Log transform species
                x0_log = np.log10(np.maximum(x0, self.norm_cfg.get("epsilon", 1e-30)))
                
                # Get targets at sampled times
                y_mat = profile[time_indices][:, self.target_species_indices]
                y_mat_log = np.log10(np.maximum(y_mat, self.norm_cfg.get("epsilon", 1e-30)))
                
                # Normalize time globally (will be done after collecting stats)
                t_vec = profile[time_indices, self.time_idx]
                
                # Determine split
                split_hash = int(hashlib.sha256((gname + "_split").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                test_frac = self.train_cfg["test_fraction"]
                val_frac = self.train_cfg["val_fraction"]
                
                if split_hash < test_frac:
                    split = "test"
                elif split_hash < test_frac + val_frac:
                    split = "validation"
                else:
                    split = "train"
                
                # Add to writer
                writers[split].add_trajectory(x0_log, globals_vec, t_vec, y_mat_log)
                split_counts[split] += 1
        
        # Flush all writers
        for writer in writers.values():
            writer.flush()
        
        # Collect metadata
        metadata = {
            "splits": {}
        }
        for split in ["train", "validation", "test"]:
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
                "total_samples": split_counts[split]  # In sequence mode, 1 trajectory = 1 sample
            }
            
        return metadata
    
    def collect_time_stats(self, file_paths: List[Path]) -> Dict[str, float]:
        """Collect global time statistics for normalization."""
        all_times = []
        
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var in grp:
                        times = grp[self.time_var][:]
                        # Filter out very small times
                        times = times[times > 1e-10]
                        if len(times) > 0:
                            all_times.append(times)
        
        if not all_times:
            raise ValueError("No valid time data found")
            
        all_times = np.concatenate(all_times)
        
        # Compute tau0 (5th percentile of non-zero times)
        tau0 = np.percentile(all_times, 5)
        
        # Apply log transform
        tau = np.log(1 + all_times / tau0)
        
        return {
            "tau0": float(tau0),
            "tmin": float(tau.min()),
            "tmax": float(tau.max()),
            "time_transform": "log_min_max"
        }


class DataPreprocessor:
    """Main preprocessor with sequence mode support."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        
        self.sequence_mode = config["data"].get("sequence_mode", False)
        self.normalizer = DataNormalizer(config)
        self.num_workers = config["preprocessing"].get("num_workers", 1)
        self.parallel = self.num_workers > 1 and len(self.raw_files) > 1
        
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_to_npy_shards(self) -> None:
        """Process data in sequence mode if enabled."""
        if self.sequence_mode:
            self._process_sequence_mode()
        else:
            # Original row-wise processing
            self._process_legacy_mode()
    
    def _process_sequence_mode(self):
        """Process data in sequence mode for LiLaN."""
        self.logger.info("Processing data in SEQUENCE MODE for multi-time supervision")
        
        # First collect time statistics
        processor = CorePreprocessor(self.config)
        time_stats = processor.collect_time_stats(self.raw_files)
        self.logger.info(f"Time normalization stats: tau0={time_stats['tau0']:.3e}, "
                        f"tmin={time_stats['tmin']:.3f}, tmax={time_stats['tmax']:.3f}")
        
        # Collect normalization statistics (for species and globals)
        norm_stats = self._collect_normalization_stats()
        norm_stats["time_normalization"] = time_stats
        save_json(norm_stats, self.output_dir / "normalization.json")
        
        # Process files to sequence shards
        all_metadata = {"splits": {"train": {"shards": [], "n_trajectories": 0},
                                  "validation": {"shards": [], "n_trajectories": 0},
                                  "test": {"shards": [], "n_trajectories": 0}}}
        
        for file_path in self.raw_files:
            processor = CorePreprocessor(self.config, norm_stats)
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
            "prediction_mode": self.config.get("prediction", {}).get("mode", "absolute"),
            "splits": all_metadata["splits"],
            "time_normalization": time_stats
        }
        save_json(shard_index, self.output_dir / "shard_index.json")
        
        self.logger.info(f"Sequence mode preprocessing complete. "
                        f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
                        f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
                        f"Test: {all_metadata['splits']['test']['n_trajectories']}")
    
    def _collect_normalization_stats(self) -> Dict[str, Any]:
        """Collect normalization statistics for species and globals."""
        # Simplified - just collect min/max/mean/std for each variable
        stats = {"per_key_stats": {}, "normalization_methods": {}}
        
        # Initialize accumulators
        accumulators = {}
        for var in self.config["data"]["species_variables"] + self.config["data"]["global_variables"]:
            accumulators[var] = {
                "count": 0, "mean": 0.0, "m2": 0.0,
                "min": float("inf"), "max": float("-inf")
            }
        
        # Process each file
        for file_path in self.raw_files:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    
                    # Species variables (use log transform)
                    for var in self.config["data"]["species_variables"]:
                        if var in grp:
                            data = grp[var][:]
                            # Apply log transform
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
                variance = acc["m2"] / (acc["count"] - 1) if acc["count"] > 1 else 0
                std = np.sqrt(variance)
                
                if var in self.config["data"]["species_variables"]:
                    # Species use log-standard normalization
                    stats["normalization_methods"][var] = "log-standard"
                    stats["per_key_stats"][var] = {
                        "method": "log-standard",
                        "log_mean": mean,
                        "log_std": max(std, 1e-10),
                        "min": acc["min"],
                        "max": acc["max"]
                    }
                else:
                    # Globals use standard normalization
                    stats["normalization_methods"][var] = "standard"
                    stats["per_key_stats"][var] = {
                        "method": "standard", 
                        "mean": mean,
                        "std": max(std, 1e-10),
                        "min": acc["min"],
                        "max": acc["max"]
                    }
        
        # Time will use special normalization
        stats["normalization_methods"][self.config["data"]["time_variable"]] = "log_min_max"
        
        return stats
    
    def _update_accumulator(self, acc: Dict, data: np.ndarray):
        """Update accumulator with Welford's algorithm."""
        for val in data.flatten():
            if np.isfinite(val):
                n = acc["count"]
                acc["count"] += 1
                delta = val - acc["mean"]
                acc["mean"] += delta / acc["count"]
                delta2 = val - acc["mean"]
                acc["m2"] += delta * delta2
                acc["min"] = min(acc["min"], val)
                acc["max"] = max(acc["max"], val)
    
    def _process_legacy_mode(self):
        """Original row-wise processing (fallback)."""
        # Import original implementation
        from .preprocessor import DataPreprocessor as LegacyPreprocessor
        legacy = LegacyPreprocessor(self.raw_files, self.output_dir, self.config)
        legacy.process_to_npy_shards()