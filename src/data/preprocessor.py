#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with sequence mode support for LiLaN.

This module builds sequence-mode NPZ shards containing, per trajectory:
  - x0_log:   [n_species]          (log10 of initial species at t0)
  - globals:  [n_globals]
  - t_vec:    [M]                   (raw times; dataset layer will normalize)
  - y_mat:    [M, n_target_species] (log10 of species at sampled times)

Key improvements:
  - Avoids building large [n_t, n_vars] profile matrices (lower memory/IO).
  - Vectorized nearest-time index selection via np.searchsorted (faster).
  - Configurable shard sizing (by count or target bytes).
  - Optional NPZ compression toggle (defaults to previous behavior: compressed).
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

from data.normalizer import NormalizationHelper
from utils.utils import save_json, load_json


class SequenceShardWriter:
    """Writes trajectory sequences to NPZ shards."""
    def __init__(
        self,
        output_dir: Path,
        trajectories_per_shard: int,
        shard_idx_base: str,
        M: int,
        n_species: int,
        n_globals: int,
        dtype: np.dtype,
        compressed: bool = True,  # keep prior default behavior
    ):
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
        y_mat: np.ndarray,
    ):
        """Add a trajectory to buffer."""
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

        # Write NPZ
        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename

        if self.compressed:
            np.savez_compressed(
                filepath,
                x0_log=x0_log,
                globals=globals_vec,
                t_vec=t_vec,
                y_mat=y_mat,
            )
        else:
            np.savez(
                filepath,
                x0_log=x0_log,
                globals=globals_vec,
                t_vec=t_vec,
                y_mat=y_mat,
            )

        self.shard_metadata.append(
            {
                "filename": filename,
                "n_samples": len(self.buffer),
            }
        )

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
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]

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
        self.var_order = self.species_vars + self.global_vars + [self.time_var]

        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        self.n_vars = self.n_species + self.n_globals + 1

        self.min_value_threshold = self.proc_cfg.get("min_value_threshold", 1e-30)

        self.norm_stats = norm_stats or {}

        self._create_index_mappings()

        if norm_stats:
            self.norm_helper = NormalizationHelper(
                norm_stats,
                torch.device("cpu"),
                self.species_vars,
                self.global_vars,
                self.time_var,
                config,
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

            # Normalize fractions
            total_frac = max(1e-12, uniform_frac + log_frac)
            uniform_frac = uniform_frac / total_frac
            log_frac = log_frac / total_frac

            n_uniform = int(self.M_per_sample * uniform_frac)
            n_log = self.M_per_sample - n_uniform

            times_list: List[np.ndarray] = []

            # Uniform samples
            if n_uniform > 0:
                t_uniform = np.linspace(t_min, t_max, n_uniform)
                times_list.append(t_uniform)

            # Log-spaced samples
            if n_log > 0:
                # Ensure positive values for log spacing
                t_min_log = max(t_min, 1e-10)
                if t_max > t_min_log:
                    log_min = np.log10(t_min_log)
                    log_max = np.log10(t_max)
                    t_log = np.logspace(log_min, log_max, n_log)
                    times_list.append(t_log)
                else:
                    # Fallback to uniform if log spacing not possible
                    t_log = np.linspace(t_min, t_max, n_log)
                    times_list.append(t_log)

            if times_list:
                times = np.concatenate(times_list)
                times = np.unique(np.sort(times))
                # Ensure we have exactly M samples
                if len(times) > self.M_per_sample:
                    indices = np.linspace(0, len(times) - 1, self.M_per_sample, dtype=int)
                    times = times[indices]
                elif len(times) < self.M_per_sample:
                    extra = self.M_per_sample - len(times)
                    t_extra = np.linspace(t_min, t_max, extra + 2)[1:-1]
                    times = np.sort(np.concatenate([times, t_extra]))[: self.M_per_sample]
            else:
                times = np.linspace(t_min, t_max, self.M_per_sample)
        else:
            # Simple uniform sampling
            times = np.linspace(t_min, t_max, self.M_per_sample)

        return times

    # --- Legacy validation retained (not used in the new fast path) ---
    def _is_profile_valid(self, group: h5py.Group) -> Tuple[bool, str]:
        """Check if a profile is valid. (Legacy; superseded by _extract_minimal)"""
        required_keys = self.species_vars + [self.time_var]
        if not set(required_keys).issubset(group.keys()):
            return False, "missing_keys"

        for var in required_keys:
            try:
                data = group[var][:]
                if not np.all(np.isfinite(data)):
                    return False, "non_finite"
                if np.any(data <= self.min_value_threshold):
                    return False, "below_threshold"
            except Exception:
                return False, "read_error"

        return True, "valid"

    def _extract_minimal(
        self, group: h5py.Group, gname: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Efficiently extract only what we need:
          - time_data: [N]
          - x0 (species at t0): [n_species]
          - globals_vec: [n_globals]
          - species_mat: [N, n_target_species]
        """
        attrs = group.attrs
        missing = [k for k in self.global_vars if k not in attrs]
        if missing:
            self.logger.error(f"Missing global variables in group '{gname}': {missing}")
            return None

        try:
            # Times
            time_data = group[self.time_var][:].astype(self.np_dtype, copy=False)
            if not np.all(np.isfinite(time_data)) or time_data.size == 0:
                return None

            # Species matrix for targets
            species_mat = np.empty((time_data.shape[0], self.n_target_species), dtype=self.np_dtype)
            for j, var in enumerate(self.target_species_vars):
                if var not in group:
                    self.logger.error(f"Species '{var}' not in group '{gname}'")
                    return None
                arr = group[var][:].astype(self.np_dtype, copy=False)
                if not np.all(np.isfinite(arr)) or np.any(arr <= self.min_value_threshold):
                    return None
                species_mat[:, j] = arr

            # x0 vector from full species set at t0
            x0 = np.empty(self.n_species, dtype=self.np_dtype)
            for i, var in enumerate(self.species_vars):
                if var not in group:
                    self.logger.error(f"Species '{var}' not in group '{gname}'")
                    return None
                v0 = float(group[var][0])
                if not np.isfinite(v0) or v0 <= self.min_value_threshold:
                    return None
                x0[i] = v0

            # Globals
            globals_vec = np.array([float(attrs[k]) for k in self.global_vars], dtype=self.np_dtype)

            return time_data, x0, globals_vec, species_mat

        except Exception as e:
            self.logger.error(f"Failed to read profile from group '{gname}': {e}")
            return None

    def process_file_for_sequence_shards(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process file and write sequence-mode shards."""
        # Determine trajectories_per_shard
        # Priority: explicit count -> target bytes -> legacy heuristic
        trajectories_per_shard = self.proc_cfg.get("trajectories_per_shard", None)
        if trajectories_per_shard is None:
            target_bytes = int(self.proc_cfg.get("target_shard_bytes", 200 * 1024 * 1024))  # ~200MB
            bytes_per_val = 4 if self.np_dtype == np.float32 else 8
            vals_per_traj = (
                self.n_species                 # x0_log
                + self.n_globals               # globals
                + self.M_per_sample            # t_vec
                + self.M_per_sample * self.n_target_species  # y_mat
            )
            traj_bytes = max(bytes_per_val * vals_per_traj, 1)
            trajectories_per_shard = max(1, target_bytes // traj_bytes)

        compressed_npz = bool(self.proc_cfg.get("npz_compressed", True))

        # Create shard writers for each split
        writers = {}
        for split in ["train", "validation", "test"]:
            writers[split] = SequenceShardWriter(
                output_dir / split,
                trajectories_per_shard,
                file_path.stem,
                self.M_per_sample,
                self.n_species,
                self.n_globals,
                self.np_dtype,
                compressed=compressed_npz,
            )

        split_counts = {"train": 0, "validation": 0, "test": 0}

        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                grp = f[gname]

                # Minimal checks before heavy reads
                if self.time_var not in grp:
                    self.logger.debug(f"Skipping group '{gname}': missing time")
                    continue
                n_t = grp[self.time_var].shape[0]
                if n_t <= self.M_per_sample:
                    self.logger.debug(
                        f"Skipping group '{gname}': insufficient time points ({n_t} <= {self.M_per_sample})"
                    )
                    continue

                # Fractional usage
                if self.train_cfg.get("use_fraction", 1.0) < 1.0:
                    hash_val = int(hashlib.sha256(gname.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if hash_val >= self.train_cfg["use_fraction"]:
                        continue

                # Fast extraction
                extracted = self._extract_minimal(grp, gname)
                if extracted is None:
                    continue
                time_data, x0, globals_vec, species_mat = extracted

                # Sample M time points
                t_min, t_max = float(time_data.min()), float(time_data.max())
                sampled_times = self._sample_times(t_min, t_max)

                # Vectorized nearest indices via searchsorted (assumes time is non-decreasing)
                pos = np.searchsorted(time_data, sampled_times, side="left")
                pos = np.clip(pos, 1, len(time_data) - 1)
                prev = pos - 1
                choose_prev = (
                    np.abs(sampled_times - time_data[prev]) <= np.abs(time_data[pos] - sampled_times)
                )
                time_indices = np.where(choose_prev, prev, pos).astype(np.int64)

                # Assemble output arrays
                eps = self.norm_cfg.get("epsilon", 1e-30)
                x0_log = np.log10(np.maximum(x0, eps))
                y_mat = species_mat[time_indices, :]
                y_mat_log = np.log10(np.maximum(y_mat, eps))
                t_vec = time_data[time_indices]

                # Determine split
                split_hash = int(
                    hashlib.sha256((gname + "_split").encode()).hexdigest()[:8], 16
                ) / 0xFFFFFFFF
                test_frac = self.train_cfg.get("test_fraction", 0.0)
                val_frac = self.train_cfg.get("val_fraction", 0.0)

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
        metadata = {"splits": {}}
        for split in ["train", "validation", "test"]:
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
                "total_samples": split_counts[split],  # sequence mode: 1 trajectory = 1 sample
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
            "time_transform": "log-min-max",
        }


class DataPreprocessor:
    """Main preprocessor for sequence mode."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config

        # LiLaN always uses sequence mode
        self.sequence_mode = True

        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_to_npy_shards(self) -> None:
        """Process data in sequence mode for LiLaN."""
        self.logger.info("Processing data in SEQUENCE MODE for LiLaN multi-time supervision")

        # First collect time statistics
        processor = CorePreprocessor(self.config)
        time_stats = processor.collect_time_stats(self.raw_files)
        self.logger.info(
            f"Time normalization stats: tau0={time_stats['tau0']:.3e}, "
            f"tmin={time_stats['tmin']:.3f}, tmax={time_stats['tmax']:.3f}"
        )

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
            "n_target_species": len(
                self.config["data"].get("target_species_variables", self.config["data"]["species_variables"])
            ),
            "n_globals": len(self.config["data"]["global_variables"]),
            "compression": "npz",
            "splits": all_metadata["splits"],
            "time_normalization": time_stats,
        }
        save_json(shard_index, self.output_dir / "shard_index.json")

        self.logger.info(
            "Sequence mode preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )

    def _collect_normalization_stats(self) -> Dict[str, Any]:
        """Collect normalization statistics for species and globals."""
        stats: Dict[str, Any] = {"per_key_stats": {}, "normalization_methods": {}}

        # Initialize accumulators
        accumulators: Dict[str, Dict[str, float]] = {}
        for var in self.config["data"]["species_variables"] + self.config["data"]["global_variables"]:
            accumulators[var] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
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
                variance = acc["m2"] / (acc["count"] - 1) if acc["count"] > 1 else 0.0
                std = float(np.sqrt(variance))

                if var in self.config["data"]["species_variables"]:
                    # Species use log-standard normalization
                    stats["normalization_methods"][var] = "log-standard"
                    stats["per_key_stats"][var] = {
                        "method": "log-standard",
                        "log_mean": float(mean),
                        "log_std": float(max(std, 1e-10)),
                        "min": float(acc["min"]),
                        "max": float(acc["max"]),
                    }
                else:
                    # Globals use standard normalization
                    stats["normalization_methods"][var] = "standard"
                    stats["per_key_stats"][var] = {
                        "method": "standard",
                        "mean": float(mean),
                        "std": float(max(std, 1e-10)),
                        "min": float(acc["min"]),
                        "max": float(acc["max"]),
                    }

        # Time uses special normalization (handled separately)
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
