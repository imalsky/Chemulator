#!/usr/bin/env python3
"""
Preprocessor for chemical kinetics data.
Converts raw HDF5 files directly to normalized NPY shards.
"""

import logging
import random
import time
import re
from pathlib import Path
from typing import Dict, List, Any

import h5py
import numpy as np
import torch
from utils.utils import save_json

from .normalizer import DataNormalizer, NormalizationHelper

DEFAULT_TOSS = 1e-25  # Minimum threshold for species values

class DataPreprocessor:
    """
    Preprocess raw HDF5 files to normalized NPY shards.
    """
    
    def __init__(
        self,
        raw_files: List[Path],
        output_dir: Path,
        config: Dict[str, Any]
    ):
        self.raw_files = raw_files
        self.output_dir = output_dir
        self.config = config
        
        self.logger = logging.getLogger(__name__)
        
        self.data_config = config["data"]
        self.species_vars = self.data_config["species_variables"]
        self.global_vars = self.data_config["global_variables"]
        self.time_var = self.data_config["time_variable"]
        self.var_order = self.species_vars + self.global_vars + [self.time_var]
        
        self.n_vars = len(self.var_order)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        
        self.shard_size = self.data_config.get("shard_size", 1000000)  # Default if not specified
        self.chunk_size = self.data_config["chunk_size"]
        
        # Initialize summary counters
        self.total_groups = 0
        self.skipped_fraction = 0
        self.skipped_missing = 0
        self.skipped_pattern = 0
        self.skipped_nonfinite = 0
        self.total_nonfinite_values = 0

        self._init_shard_index()  # Initialize shard_index structure
    
    def _init_shard_index(self) -> None:
        """Initialize an empty shard-index structure."""
        self.shard_index: Dict[str, Any] = {
            "format": "npy_shards_v1",
            "n_species": self.n_species,
            "n_globals": self.n_globals,
            "samples_per_shard": self.shard_size,
            "n_shards": 0,
            "total_samples": 0,
            "shards": [],
            "split_files": {
                "train": "train_indices.npy",
                "validation": "val_indices.npy",
                "test": "test_indices.npy"
            }
        }
    
    def process_to_npy_shards(self) -> Dict[str, Any]:
        """
        Two-stage pipeline:
          1. Single streaming sweep to accumulate statistics and count rows.
          2. Second streaming sweep to write normalised NPY shards.

        Returns:
            Dictionary with train_indices, val_indices, test_indices, and misc metadata.
        """
        self.logger.info("─" * 80)
        self.logger.info("Stage 1 – collecting normalisation statistics")

        ########################
        # Pass 1 – statistics  #
        ########################
        pass1_start = time.time()
        max_timesteps = 0
        accumulators  = self._initialize_accumulators()

        for raw_file in self.raw_files:
            file_start = time.time()
            self.logger.info(f"Processing stats from {raw_file}")
            
            with h5py.File(raw_file, "r") as f:
                for gname in f.keys():
                    self.total_groups += 1  # Track all attempted groups

                    grp = f[gname]
                    if not self._accept_group(grp, gname):
                        continue

                    n_t = grp[self.time_var].shape[0]
                    if n_t > 10000:
                        self.logger.debug(f"Processing large group {gname} with {n_t:,} timesteps")
                    
                    max_timesteps = max(max_timesteps, n_t)
                    profile_np    = self._group_to_profile(grp, gname, n_t)
                    # Reshape to 3D (n_profiles=1, n_t, n_vars) for normalizer compatibility
                    profile_3d = profile_np.reshape(1, n_t, self.n_vars)
                    self.normalizer._update_accumulators(
                        profile_3d, accumulators, n_t
                    )
            
            file_time = time.time() - file_start
            self.logger.info(f"Processed stats for file {raw_file} in {file_time:.1f}s")
        
        pass1_time = time.time() - pass1_start
        self.logger.info(f"Stage 1 completed in {pass1_time:.1f}s")

        norm_stats = self.normalizer._finalize_statistics(accumulators)
        try:
            save_json(norm_stats, self.output_dir / "normalization.json")
        except OSError as e:
            self.logger.error(f"Failed to write normalization.json: {e}")
            raise

        ########################
        # Pass 2 – shard write #
        ########################
        self.logger.info("Stage 2 – writing normalised shards")
        pass2_start = time.time()
        
        helper       = NormalizationHelper(
            norm_stats, torch.device("cpu"),
            self.species_vars, self.global_vars, self.time_var, self.config
        )
        shard_id     = 0
        shard_chunks   = []  # list of 2D np arrays
        splits       = {"train": [], "validation": [], "test": []}
        val_f, test_f = self.config["training"]["val_fraction"], self.config["training"]["test_fraction"]
        global_idx   = 0

        for raw_file in self.raw_files:
            file_start = time.time()
            self.logger.info(f"Processing {raw_file} to shards")
            
            with h5py.File(raw_file, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if not self._accept_group(grp, gname):
                        continue

                    n_t = grp[self.time_var].shape[0]
                    if n_t > 10000:
                        group_start = time.time()
                        self.logger.debug(f"Starting processing for large group {gname} with {n_t:,} timesteps")
                    
                    # Get and normalize profile
                    profile = self._group_to_profile(grp, gname, n_t)
                    profile_tensor = torch.from_numpy(profile)
                    normalized_profile = helper.normalize_profile(profile_tensor).numpy()
                    
                    # Convert to samples
                    samples = self._group_to_samples(normalized_profile, n_t)   # 2D np.ndarray or None
                    if samples is not None:
                        shard_chunks.append(samples)

                    # flush if needed
                    current_size = sum(chunk.shape[0] for chunk in shard_chunks)
                    while current_size >= self.shard_size:
                        # Collect data for one shard
                        to_write = []
                        remaining = self.shard_size
                        new_chunks = []
                        for chunk in shard_chunks:
                            if remaining == 0:
                                new_chunks.append(chunk)
                                continue
                            if chunk.shape[0] <= remaining:
                                to_write.append(chunk)
                                remaining -= chunk.shape[0]
                            else:
                                to_write.append(chunk[:remaining])
                                new_chunks.append(chunk[remaining:])
                                remaining = 0
                        shard_chunks = new_chunks

                        write_data = np.concatenate(to_write, axis=0) if len(to_write) > 1 else to_write[0]
                        self._write_shard(
                            shard_id, write_data, self.shard_index
                        )
                        shard_id += 1

                        current_size = sum(chunk.shape[0] for chunk in shard_chunks)

                    # split bookkeeping
                    for _ in range(n_t - 1):
                        r = random.random()
                        tgt = ("test"        if r < test_f else
                               "validation"  if r < test_f + val_f else
                               "train")
                        splits[tgt].append(global_idx)
                        global_idx += 1
                    
                    if n_t > 10000:
                        group_time = time.time() - group_start
                        self.logger.debug(f"Completed processing for large group {gname} in {group_time:.1f}s")
            
            file_time = time.time() - file_start
            self.logger.info(f"Sharded file {raw_file} in {file_time:.1f}s")

        # final partial shard
        if shard_chunks:
            write_data = np.concatenate(shard_chunks, axis=0) if len(shard_chunks) > 1 else shard_chunks[0]
            self._write_shard(shard_id, write_data, self.shard_index)

        # Update total_samples
        self.shard_index["total_samples"] = global_idx
        save_json(self.shard_index, self.output_dir / "shard_index.json")
        
        pass2_time = time.time() - pass2_start
        self.logger.info(f"Stage 2 completed in {pass2_time:.1f}s")

        # Save split indices as NPY
        # Use the split_files mapping so 'validation' → 'val_indices.npy'
        for split_name, idx_list in splits.items():
            idx_arr = np.array(idx_list, dtype=np.int64)
            filename = self.shard_index["split_files"].get(
                split_name,
                f"{split_name}_indices.npy"
            )
            np.save(self.output_dir / filename, idx_arr)
            self.logger.info(f"{split_name} split: {len(idx_list):,} samples (saved to {filename})")

        # Log skipped summary
        skipped_total = self.skipped_fraction + self.skipped_missing + self.skipped_pattern + self.skipped_nonfinite
        self.logger.info(f"Preprocessing summary: Total groups attempted: {self.total_groups}, "
                         f"Processed: {self.total_groups - skipped_total}, "
                         f"Skipped: {skipped_total} ({skipped_total / self.total_groups * 100 if self.total_groups else 0:.1f}%) "
                         f"[fraction: {self.skipped_fraction}, missing keys: {self.skipped_missing}, "
                         f"pattern mismatch: {self.skipped_pattern}, non-finite: {self.skipped_nonfinite}]")

        if global_idx == 0:
            self.logger.error("No valid data processed. Exiting with warning.")
            raise SystemExit("Preprocessing failed: No valid data found in raw files. Check input data for issues like missing variables or non-finite values.")

        return {f"{k}_indices": splits[k] for k in ["train", "validation", "test"]}
    
    def _initialize_accumulators(self):
        """
        Initialize accumulators for stats collection.
        First sweep to discover the true max_timesteps, then init normalizer.
        """
        # 1) Find true maximum timesteps across all groups
        max_timesteps = 0
        for raw_file in self.raw_files:
            with h5py.File(raw_file, "r") as f:
                for gname in f.keys():
                    n_t = f[gname][self.time_var].shape[0]
                    max_timesteps = max(max_timesteps, n_t)

        # 2) Initialize normalizer now that we know max_timesteps
        self.normalizer = DataNormalizer(
            self.config,
            actual_timesteps=max_timesteps
        )
        return self.normalizer._initialize_accumulators(max_timesteps)
    
    def _accept_group(self, group: h5py.Group, gname: str) -> bool:
        """Combined acceptance check: validate + pattern match + use_fraction."""
        if random.random() > self.config["training"]["use_fraction"]:
            self.skipped_fraction += 1
            return False
        
        # Check all required variables exist
        required_keys = set(self.species_vars + [self.time_var])
        missing_vars = required_keys - set(group.keys())
        if missing_vars:
            if not hasattr(self, '_warned_missing_vars'):
                self._warned_missing_vars = True
                self.logger.error(f"Missing variables in HDF5: {missing_vars}")
            self.skipped_missing += 1
            return False
        
        if not self._validate_group(group):
            self.skipped_nonfinite += 1
            return False
        
        match = re.match(r"run_T_(?P<T_init>[\d.eE+-]+)_P_(?P<P_init>[\d.eE+-]+)_SEED_\d+", gname)
        if not match:
            self.skipped_pattern += 1
            return False
        
        return True
    
    def _group_to_profile(self, group: h5py.Group, gname: str, n_t: int) -> np.ndarray:
        """Extract profile from group using chunked reading to handle large n_t."""
        match = re.match(r"run_T_(?P<T_init>[\d.eE+-]+)_P_(?P<P_init>[\d.eE+-]+)_SEED_\d+", gname)
        try:
            T_init = float(match.group('T_init'))
            P_init = float(match.group('P_init'))
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid T_init or P_init in group {gname}: {e}")
            raise ValueError(f"Cannot parse initial conditions from {gname}")
        
        profile = np.zeros((n_t, self.n_vars), dtype=np.float32)
        
        # Chunked load for variables from HDF5
        for start in range(0, n_t, self.chunk_size):
            end = min(start + self.chunk_size, n_t)
            chunk_size = end - start
            if chunk_size > 10000:
                self.logger.debug(f"Loading chunk {start}-{end} for group {gname}")
            
            for i, var in enumerate(self.var_order):
                if var in self.species_vars or var == self.time_var:
                    profile[start:end, i] = group[var][start:end]
                elif var == "T_init":
                    profile[start:end, i] = T_init
                elif var == "P_init":
                    profile[start:end, i] = P_init
        
        return profile
    
    def _group_to_samples(self, normalized_profile: np.ndarray, n_t: int) -> np.ndarray:
        """Flatten normalized profile to samples (exclude t=0) using vectorized operations."""
        if n_t <= 1:
            return None
        
        # Log processing for large profiles
        if n_t > 10000:
            self.logger.info(f"Converting profile with {n_t:,} timesteps to {n_t-1:,} training samples")
        
        # Initial species (repeat for all samples)
        init_species = np.repeat(normalized_profile[0, :self.n_species][np.newaxis, :], n_t - 1, axis=0)
        
        # Globals (from t=1 to end)
        globals_t = normalized_profile[1:, self.n_species:self.n_species + self.n_globals]
        
        # Time (from t=1 to end)
        time_t = normalized_profile[1:, -1][:, np.newaxis]
        
        # Targets (species from t=1 to end)
        targets = normalized_profile[1:, :self.n_species]
        
        # Concatenate along columns
        samples_array = np.concatenate([init_species, globals_t, time_t, targets], axis=1)
        
        return samples_array
    
    def _write_shard(
        self,
        shard_idx: int,
        data: np.ndarray,
        shard_index: Dict[str, Any]
    ) -> None:
        """
        Write a single NPY shard (data already normalized).
        Updates shard_index with correct start/end offsets.
        """
        shard_start = time.time()
        shard_path = self.output_dir / f"shard_{shard_idx:04d}.npy"

        # Serialize data to disk
        np.save(shard_path, data)

        # Measure file size
        file_size = shard_path.stat().st_size

        # Compute correct start/end indices from running total_samples
        start_idx = shard_index.get(
            "total_samples",
            shard_idx * shard_index["samples_per_shard"]
        )
        end_idx = start_idx + data.shape[0]

        # Record shard metadata
        shard_info = {
            "shard_idx": shard_idx,
            "filename": shard_path.name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_samples": data.shape[0],
            "file_size": int(file_size),
        }
        shard_index["shards"].append(shard_info)
        shard_index["n_shards"] += 1

        # Update total_samples for the next shard
        shard_index["total_samples"] = end_idx

        # Debug log
        shard_time = time.time() - shard_start
        self.logger.debug(
            f"Wrote shard {shard_idx}: {data.shape[0]:,} samples, "
            f"{file_size/1e6:.1f} MB in {shard_time:.1f}s"
        )
    
    def _validate_group(self, group: h5py.Group) -> bool:
        """
        Validate group data and log rejection reasons.
        Returns False if any NaN/Inf is found or if any species value is 
        below threshold.
        """
        required_keys = self.species_vars + [self.time_var]
        is_valid = True
        n_t = group[self.time_var].shape[0]
        min_threshold = DEFAULT_TOSS
        
        for key in required_keys:
            bad_count = 0  # For non-finites
            below_threshold_count = 0  # For values below threshold
            
            for start in range(0, n_t, self.chunk_size):
                end = min(start + self.chunk_size, n_t)
                data_chunk = group[key][start:end]
                
                # Check for non-finites
                bad_mask = ~np.isfinite(data_chunk)
                bad_count += int(bad_mask.sum())
                
                # Check for species values below threshold
                if key in self.species_vars:
                    below_mask = data_chunk < min_threshold
                    below_threshold_count += int(below_mask.sum())
                    
                    if np.any(below_mask):
                        is_valid = False
            
            self.total_nonfinite_values += bad_count
            if bad_count > 0:
                is_valid = False
        
        return is_valid