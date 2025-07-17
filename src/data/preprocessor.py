#!/usr/bin/env python3
"""
Preprocessor for chemical kinetics data.
Converts raw HDF5 files to normalized chunked HDF5 with train/val/test splits.
"""

import logging
import sys
import random
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

import h5py
import numpy as np
import torch
from utils.utils import save_json

from .normalizer import DataNormalizer, NormalizationHelper

DEFAULT_TOSS = 1e-25

class DataPreprocessor:
    """
    Preprocess raw HDF5 files to normalized chunked HDF5 with splits.
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
        
        self.chunk_size = self.data_config["chunk_size"]
        self.compression = self.data_config.get("compression", "gzip")
        self.compression_level = self.data_config.get("compression_level", 4)
        
        # Output HDF5 path
        self.output_hdf5 = self.output_dir / config["paths"].get("processed_hdf5_file", "preprocessed_data.h5")
        
        # Initialize summary counters
        self.total_groups = 0
        self.skipped_fraction = 0
        self.skipped_missing = 0
        self.skipped_pattern = 0
        self.skipped_nonfinite = 0
        self.total_nonfinite_values = 0
    
    def process_to_hdf5(self) -> Dict[str, Any]:
        """
        Two-stage pipeline:
        1. Single streaming sweep to accumulate statistics and count rows.
        2. Second streaming sweep to write normalized HDF5 with splits.

        Returns:
            Dictionary with dataset metadata and split information.
            
        Raises:
            SystemExit: If no valid data is found
        """
        self.logger.info("─" * 80)
        self.logger.info("Stage 1 – collecting normalisation statistics")

        ########################
        # Pass 1 – statistics  #
        ########################
        pass1_start = time.time()
        max_timesteps = 0
        total_samples = 0
        accumulators = self._initialize_accumulators()

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
                    total_samples += (n_t - 1)  # Exclude t=0
                    
                    profile_np = self._group_to_profile(grp, gname, n_t)
                    # Reshape to 3D (n_profiles=1, n_t, n_vars) for normalizer compatibility
                    profile_3d = profile_np.reshape(1, n_t, self.n_vars)
                    self.normalizer._update_accumulators(
                        profile_3d, accumulators, n_t
                    )
            
            file_time = time.time() - file_start
            self.logger.info(f"Processed stats for file {raw_file} in {file_time:.1f}s")
        
        pass1_time = time.time() - pass1_start
        self.logger.info(f"Stage 1 completed in {pass1_time:.1f}s")
        self.logger.info(f"Total samples to process: {total_samples:,}")

        if total_samples == 0:
            self.logger.error("No valid samples found in data!")
            self.logger.error("Check that:")
            self.logger.error("1. Raw data files contain expected variables")
            self.logger.error("2. Data values are within acceptable ranges")
            self.logger.error("3. Group names match expected pattern")
            sys.exit(1)  # Exit with error instead of returning error dict

        # Continue with rest of processing...
        norm_stats = self.normalizer._finalize_statistics(accumulators)
        save_json(norm_stats, self.output_dir / "normalization.json")

        ########################
        # Pass 2 – HDF5 write  #
        ########################
        self.logger.info("Stage 2 – writing normalized HDF5 with splits")
        pass2_start = time.time()
        
        helper = NormalizationHelper(
            norm_stats, torch.device("cpu"),
            self.species_vars, self.global_vars, self.time_var, self.config
        )
        
        # Calculate split sizes
        val_f = self.config["training"]["val_fraction"]
        test_f = self.config["training"]["test_fraction"]
        
        # Create output HDF5 file
        self.logger.info(f"Creating HDF5 file: {self.output_hdf5}")
        
        with h5py.File(self.output_hdf5, 'w') as out_f:
            # Create split groups
            splits = {
                "train": self._create_split_group(out_f, "train", self.chunk_size),
                "validation": self._create_split_group(out_f, "validation", self.chunk_size),
                "test": self._create_split_group(out_f, "test", self.chunk_size)
            }
            
            # Add global metadata
            out_f.attrs['n_species'] = self.n_species
            out_f.attrs['n_globals'] = self.n_globals
            out_f.attrs['format_version'] = "1.0"
            out_f.attrs['created'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Process all files and write to splits
            split_indices = {"train": 0, "validation": 0, "test": 0}
            global_idx = 0
            
            for raw_file in self.raw_files:
                file_start = time.time()
                self.logger.info(f"Processing {raw_file} to HDF5")
                
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
                        inputs_arr, targets_arr = self._profile_to_samples(normalized_profile, n_t)
                        num_samples = len(inputs_arr)
                        
                        # Assign split for the entire profile
                        r = random.random()
                        if r < test_f:
                            split_name = "test"
                        elif r < test_f + val_f:
                            split_name = "validation"
                        else:
                            split_name = "train"
                        
                        # Write directly to HDF5
                        if num_samples > 0:
                            start_idx = split_indices[split_name]
                            end_idx = start_idx + num_samples
                            
                            inputs_dset = splits[split_name]["inputs"]
                            targets_dset = splits[split_name]["targets"]
                            
                            # Dynamically resize if necessary
                            current_size = inputs_dset.shape[0]
                            if end_idx > current_size:
                                new_size = max(end_idx, current_size * 2)
                                inputs_dset.resize((new_size, self.n_species + self.n_globals + 1))
                                targets_dset.resize((new_size, self.n_species))
                            
                            inputs_dset[start_idx:end_idx] = inputs_arr
                            targets_dset[start_idx:end_idx] = targets_arr
                            
                            split_indices[split_name] = end_idx
                            global_idx += num_samples
                        
                        if n_t > 10000:
                            group_time = time.time() - group_start
                            self.logger.debug(f"Completed processing for large group {gname} in {group_time:.1f}s")
                
                file_time = time.time() - file_start
                self.logger.info(f"Processed file {raw_file} in {file_time:.1f}s")
            
            # Resize datasets to actual sizes and update metadata
            for split_name, split_data in splits.items():
                actual_size = split_indices[split_name]
                split_data["inputs"].resize((actual_size, self.n_species + self.n_globals + 1))
                split_data["targets"].resize((actual_size, self.n_species))
                split_data["group"].attrs['n_samples'] = actual_size
                
                self.logger.info(f"{split_name} split: {actual_size:,} samples")
        
        pass2_time = time.time() - pass2_start
        self.logger.info(f"Stage 2 completed in {pass2_time:.1f}s")
        
        # Log summary
        skipped_total = self.skipped_fraction + self.skipped_missing + self.skipped_pattern + self.skipped_nonfinite
        self.logger.info(
            f"Preprocessing summary: Total groups attempted: {self.total_groups}, "
            f"Processed: {self.total_groups - skipped_total}, "
            f"Skipped: {skipped_total} ({skipped_total / self.total_groups * 100 if self.total_groups else 0:.1f}%) "
            f"[fraction: {self.skipped_fraction}, missing keys: {self.skipped_missing}, "
            f"pattern mismatch: {self.skipped_pattern}, non-finite: {self.skipped_nonfinite}]"
        )
        
        # Return metadata
        return {
            "output_file": str(self.output_hdf5),
            "total_samples": global_idx,
            "splits": {
                split_name: split_indices[split_name] 
                for split_name in ["train", "validation", "test"]
            },
            "normalization_file": str(self.output_dir / "normalization.json")
        }
    
    def _create_split_group(self, hdf5_file: h5py.File, split_name: str, chunk_size: int) -> Dict[str, Any]:
        """Create a split group in the HDF5 file with chunked datasets."""
        grp = hdf5_file.create_group(split_name)
        
        # Create datasets with chunking and compression
        # Start with a reasonable size, will resize later
        initial_size = 1000000
        
        inputs_dset = grp.create_dataset(
            "inputs",
            shape=(initial_size, self.n_species + self.n_globals + 1),
            maxshape=(None, self.n_species + self.n_globals + 1),
            dtype=np.float32,
            chunks=(chunk_size, self.n_species + self.n_globals + 1),
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        targets_dset = grp.create_dataset(
            "targets", 
            shape=(initial_size, self.n_species),
            maxshape=(None, self.n_species),
            dtype=np.float32,
            chunks=(chunk_size, self.n_species),
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        # Add metadata
        grp.attrs['n_species'] = self.n_species
        grp.attrs['n_globals'] = self.n_globals
        grp.attrs['n_samples'] = 0  # Will be updated
        
        return {
            "group": grp,
            "inputs": inputs_dset,
            "targets": targets_dset
        }
    
    def _initialize_accumulators(self):
        """Initialize accumulators for stats collection."""
        self.normalizer = DataNormalizer(self.config, actual_timesteps=1)  # Timesteps updated per group
        return self.normalizer._initialize_accumulators(0)
    
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
    
    def _profile_to_samples(self, normalized_profile: np.ndarray, n_t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized profile to input and target arrays.
        Raises error if insufficient timesteps instead of returning empty arrays.
        """
        if n_t <= 1:
            self.logger.error(f"Insufficient timesteps for training: n_t={n_t}. Need at least 2 timesteps.")
            raise ValueError(f"Cannot create training samples from {n_t} timesteps. Need at least 2.")
        
        # Log processing for large profiles
        if n_t > 10000:
            self.logger.info(f"Converting profile with {n_t:,} timesteps to {n_t-1:,} training samples")
        
        # Initial species (repeat for all samples)
        init_species = np.repeat(normalized_profile[0, :self.n_species][np.newaxis, :], n_t - 1, axis=0)
        
        # Globals (from t=1 to end)
        globals_t = normalized_profile[1:, self.n_species:self.n_species + self.n_globals]
        
        # Time (from t=1 to end)
        time_t = normalized_profile[1:, -1][:, np.newaxis]
        
        # Inputs: init_species + globals_t + time_t
        inputs_arr = np.concatenate([init_species, globals_t, time_t], axis=1)
        
        # Targets (species from t=1 to end)
        targets_arr = normalized_profile[1:, :self.n_species]
        
        return inputs_arr, targets_arr
        
    def _validate_group(self, group: h5py.Group) -> bool:
        """
        Validate group data and log rejection reasons.
        Returns False if any NaN/Inf is found or if any species value is 
        below threshold.
        """
        required_keys = self.species_vars + [self.time_var]
        is_valid = True
        n_t = group[self.time_var].shape[0]
        chunk_size = self.chunk_size
        min_threshold = DEFAULT_TOSS
        
        for key in required_keys:
            bad_count = 0  # For non-finites
            below_threshold_count = 0  # For values below threshold
            
            for start in range(0, n_t, chunk_size):
                end = min(start + chunk_size, n_t)
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