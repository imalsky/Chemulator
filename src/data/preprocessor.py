#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with corrected normalization statistics.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import torch

from .normalizer import DataNormalizer, NormalizationHelper
from utils.utils import save_json


class DataPreprocessor:
    """Preprocess HDF5 raw data to normalized NPY shards."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        # Sort the raw_files list to ensure a deterministic processing order.
        self.raw_files = sorted(raw_files)
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data configuration
        data_config = config["data"]
        self.species_vars = data_config["species_variables"]
        self.global_vars = data_config["global_variables"]
        self.time_var = data_config["time_variable"]
        self.var_order = self.species_vars + self.global_vars + [self.time_var]
        
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        self.n_vars = self.n_species + self.n_globals + 1
        
        # Preprocessing config
        preprocess_config = config["preprocessing"]
        self.shard_size = preprocess_config["shard_size"]
        self.min_threshold = preprocess_config["min_species_threshold"]
        self.compression = preprocess_config.get("compression", None)
        self.num_workers = preprocess_config.get("num_workers", 16)
        self.parallel_enabled = preprocess_config.get("parallel_enabled", True)
        
        # Prediction mode
        self.prediction_mode = config.get("prediction", {}).get("mode", "absolute")
        
        # Initialize statistics tracking
        self.preprocessing_stats = {
            "total_groups_examined": 0,
            "groups_dropped_subsampling": 0,
            "groups_dropped_validation": 0,
            "groups_dropped_no_future": 0,
            "groups_processed": 0,
            "total_samples_generated": 0,
            "per_file_stats": {}
        }
        
        self._init_shard_index()
    
    def _init_shard_index(self) -> None:
        """Initialize shard index structure."""
        self.shard_index = {
            "n_species": self.n_species,
            "n_globals": self.n_globals,
            "samples_per_shard": self.shard_size,
            "n_shards": 0,
            "total_samples": 0,
            "compression": self.compression,
            "shards": [],
            "split_files": {
                "train": "train_indices.npy",
                "validation": "val_indices.npy",
                "test": "test_indices.npy"
            },
            "prediction_mode": self.prediction_mode,
        }
        
    def process_to_npy_shards(self) -> Dict[str, Any]:
        """Two-pass processing: collect statistics then write normalized shards."""
        start_time = time.time()
        self.logger.info(f"Starting preprocessing with {len(self.raw_files)} files")
        self.logger.info(f"Prediction mode: {self.prediction_mode}")
        
        # Pass 1: Collect statistics
        self.logger.info("Pass 1: Collecting normalization statistics")
        norm_stats = self._collect_statistics()
        
        # Pass 2: Write normalized shards
        self.logger.info("Pass 2: Writing normalized NPY shards")
        split_indices = self._write_normalized_shards(norm_stats)
        
        # Generate data report
        self._generate_data_report(norm_stats, split_indices)
        
        self.logger.info(f"Preprocessing completed in {time.time() - start_time:.1f}s")
        return split_indices
        
    def _collect_statistics(self) -> Dict[str, Any]:
        """First pass: collect normalization statistics with mode-specific handling."""
        self.normalizer = DataNormalizer(self.config)
        
        # Initialize accumulators for input variables
        # In ratio mode: species and globals use only t=0, time uses t>0
        # In absolute mode: all variables use all timesteps except time uses t>0
        accumulators = self.normalizer._initialize_accumulators()
        
        # In ratio mode, create per-species accumulators for the log-ratios
        if self.prediction_mode == "ratio":
            ratio_accumulators = {
                var: {"count": 0, "mean": 0.0, "m2": 0.0, "min": float("inf"), "max": float("-inf")}
                for var in self.species_vars
            }

        for raw_file in self.raw_files:
            if self.prediction_mode == "ratio":
                self._process_file_statistics_ratio(raw_file, accumulators, ratio_accumulators)
            else:
                self._process_file_statistics_absolute(raw_file, accumulators)

        # Finalize statistics for the input variables
        norm_stats = self.normalizer._finalize_statistics(accumulators)

        # If in ratio mode, finalize the per-species ratio statistics
        if self.prediction_mode == "ratio":
            norm_stats["ratio_stats"] = {}
            for var, acc in ratio_accumulators.items():
                if acc["count"] > 1:
                    variance = acc["m2"] / (acc["count"] - 1)
                    std = max(np.sqrt(variance), self.config["normalization"]["min_std"])
                else:
                    std = 1.0

                norm_stats["ratio_stats"][var] = {
                    "mean": acc["mean"],
                    "std": std,
                    "min": acc["min"],
                    "max": acc["max"],
                    "count": acc["count"]
                }
            self.logger.info("Per-species ratio statistics calculated.")

        save_json(norm_stats, self.output_dir / "normalization.json")
        return norm_stats

    def _process_file_statistics_absolute(self, raw_file: Path, accumulators: Dict[str, Any]):
        """Process statistics for absolute mode - species/globals use all timesteps, time uses t>0."""
        groups_processed = 0
        with h5py.File(raw_file, "r") as f:
            for gname in f.keys():
                grp = f[gname]
                
                use_fraction = self.config["training"]["use_fraction"]
                if use_fraction < 1.0:
                    gname_hash = hashlib.sha256(gname.encode('utf-8')).hexdigest()
                    hash_float = int(gname_hash[:8], 16) / 0xFFFFFFFF
                    if hash_float >= use_fraction:
                        continue
                
                if not self._validate_group_simple(grp):
                    continue

                n_t = grp[self.time_var].shape[0]
                profile = self._extract_profile(grp, gname, n_t)

                if profile is None or n_t <= 1:
                    continue

                # For absolute mode:
                # - Species and globals: use all timesteps
                # - Time: use only t>0
                self._update_accumulators_selective(profile, accumulators, n_t, 
                                                   exclude_t0_for_time=True)

                groups_processed += 1
        self.logger.info(f"  Processed {groups_processed} groups from {raw_file.name}")

    def _process_file_statistics_ratio(self, raw_file: Path, accumulators: Dict[str, Any],
                                    ratio_accumulators: Dict[str, Dict[str, Any]]):
        """Process statistics for ratio mode - species/globals use only t=0, time uses t>0."""
        groups_processed = 0
        with h5py.File(raw_file, "r") as f:
            for gname in f.keys():
                grp = f[gname]
                
                use_fraction = self.config["training"]["use_fraction"]
                if use_fraction < 1.0:
                    gname_hash = hashlib.sha256(gname.encode('utf-8')).hexdigest()
                    hash_float = int(gname_hash[:8], 16) / 0xFFFFFFFF
                    if hash_float >= use_fraction:
                        continue
                
                if not self._validate_group_simple(grp):
                    continue

                n_t = grp[self.time_var].shape[0]
                profile = self._extract_profile(grp, gname, n_t)

                if profile is None or n_t <= 1:
                    continue

                # For ratio mode:
                # - Species and globals: use only t=0 (initial conditions)
                # - Time: use only t>0 (actual input times)
                self._update_accumulators_ratio_mode(profile, accumulators, n_t)

                # Calculate and accumulate log-ratios for each species
                initial_species = profile[0, :self.n_species]
                future_species = profile[1:, :self.n_species]
                
                epsilon = self.config["normalization"]["epsilon"]
                ratios = future_species / np.maximum(initial_species[None, :], epsilon)
                log_ratios = np.log10(np.maximum(ratios, epsilon))

                # Update ratio statistics per species
                for i, var_name in enumerate(self.species_vars):
                    acc = ratio_accumulators[var_name]
                    vec = log_ratios[:, i]
                    
                    if vec.size > 0:
                        finite_mask = np.isfinite(vec)
                        vec = vec[finite_mask]
                        if vec.size == 0:
                            continue

                        n_a = acc["count"]
                        n_b = vec.size

                        mean_a = acc["mean"]
                        mean_b = float(vec.mean())
                        m2_b = float(((vec - mean_b) ** 2).sum())

                        delta = mean_b - mean_a
                        n_ab = n_a + n_b

                        if n_ab > 0:
                            acc["mean"] = (n_a * mean_a + n_b * mean_b) / n_ab
                            acc["m2"] += m2_b + (delta**2 * n_a * n_b) / n_ab
                            acc["count"] = n_ab
                            acc["min"] = min(acc["min"], float(vec.min()))
                            acc["max"] = max(acc["max"], float(vec.max()))

                groups_processed += 1
        self.logger.info(f"  Processed {groups_processed} groups from {raw_file.name} for ratio stats")

    def _update_accumulators_selective(self, profile: np.ndarray, accumulators: Dict[str, Any], 
                                     n_t: int, exclude_t0_for_time: bool = True):
        """Update accumulators with selective timestep handling."""
        profile_3d = profile.reshape(1, n_t, self.n_vars)
        
        for var, acc in accumulators.items():
            idx = acc["index"]
            method = acc["method"]
            
            if var == self.time_var and exclude_t0_for_time and n_t > 1:
                # For time variable, exclude t=0
                vec = profile_3d[0, 1:, idx].astype(np.float64)
            else:
                # For other variables, use all timesteps
                vec = profile_3d[0, :, idx].ravel().astype(np.float64)
            
            # Rest of the update logic remains the same
            finite_mask = np.isfinite(vec)
            if (~finite_mask).sum() > 0:
                if (~finite_mask).sum() / vec.size > 0.01:
                    self.logger.warning(f"Variable {var} has {(~finite_mask).sum()}/{vec.size} non-finite values")
                vec = vec[finite_mask]
            
            if vec.size == 0:
                self.logger.warning(f"Variable {var} has no finite values, skipping")
                continue

            if method in {"log-standard", "log-min-max"}:
                below_epsilon = vec < self.normalizer.epsilon
                if below_epsilon.any():
                    self.logger.warning(
                        f"Variable {var} has {below_epsilon.sum()} values below epsilon. "
                        f"Min value: {vec.min():.2e}"
                    )
                vec = np.log10(np.maximum(vec, self.normalizer.epsilon))

            n_b = vec.size
            mean_b = float(vec.mean())
            m2_b = float(((vec - mean_b) ** 2).sum()) if n_b > 1 else 0.0

            acc["min"] = min(acc["min"], float(vec.min()))
            acc["max"] = max(acc["max"], float(vec.max()))

            # Chan's parallel mean/variance update
            n_a = acc["count"]
            delta = mean_b - acc["mean"]
            n_ab = n_a + n_b

            acc["mean"] += delta * n_b / n_ab
            acc["m2"] += m2_b + delta**2 * n_a * n_b / n_ab
            acc["count"] = n_ab

    def _update_accumulators_ratio_mode(self, profile: np.ndarray, accumulators: Dict[str, Any], n_t: int):
        """Update accumulators for ratio mode: species/globals use t=0, time uses t>0."""
        for var, acc in accumulators.items():
            idx = acc["index"]
            method = acc["method"]
            
            if var in self.species_vars or var in self.global_vars:
                # For species and globals in ratio mode, use only t=0
                vec = np.array([profile[0, idx]], dtype=np.float64)
            elif var == self.time_var:
                # For time, use only t>0
                if n_t > 1:
                    vec = profile[1:, idx].astype(np.float64)
                else:
                    continue
            else:
                # Should not happen
                continue
            
            # Apply log transform if needed
            if method in {"log-standard", "log-min-max"}:
                vec = np.log10(np.maximum(vec, self.normalizer.epsilon))
            
            # Update statistics
            n_b = vec.size
            mean_b = float(vec.mean())
            m2_b = float(((vec - mean_b) ** 2).sum()) if n_b > 1 else 0.0

            acc["min"] = min(acc["min"], float(vec.min()))
            acc["max"] = max(acc["max"], float(vec.max()))

            # Chan's update
            n_a = acc["count"]
            delta = mean_b - acc["mean"]
            n_ab = n_a + n_b

            acc["mean"] += delta * n_b / n_ab
            acc["m2"] += m2_b + delta**2 * n_a * n_b / n_ab
            acc["count"] = n_ab
        
    def _write_normalized_shards(self, norm_stats: Dict[str, Any]) -> Dict[str, List[int]]:
        """Second pass – write normalized data to NPY shards."""
        # Setup helper and writer
        helper = NormalizationHelper(
            norm_stats,
            torch.device("cpu"),
            self.species_vars,
            self.global_vars,
            self.time_var,
            self.config,
        )

        shard_writer = ShardWriter(
            self.output_dir,
            self.shard_size,
            self.shard_index,
            compression=self.compression,
        )

        val_f  = self.config["training"]["val_fraction"]
        test_f = self.config["training"]["test_fraction"]
        splits = {"train": [], "validation": [], "test": []}

        global_idx = 0
        profiles_written = 0
        profiles_skipped = 0

        # Get full ratio stats dict if in ratio mode
        ratio_stats = norm_stats.get("ratio_stats", {})

        # Process files
        if self.parallel_enabled and self.num_workers > 1 and len(self.raw_files) > 1:
            # Parallel processing with fixed index handling
            results = self._parallel_write_shards_fixed(
                norm_stats, helper, val_f, test_f, 
                ratio_stats, global_idx
            )
            
            # Aggregate results - indices are now absolute, no adjustment needed
            for file_samples, file_splits, written, skipped, file_stats in results:
                for samples in file_samples:
                    shard_writer.add_samples(samples)
                
                # Directly extend splits with the absolute indices
                for split_name, indices in file_splits.items():
                    splits[split_name].extend(indices)
                
                profiles_written += written
                profiles_skipped += skipped
                
                # Update preprocessing stats
                for key, value in file_stats.items():
                    if key in self.preprocessing_stats:
                        self.preprocessing_stats[key] += value
            
            # Calculate total samples from all splits
            global_idx = sum(len(indices) for indices in splits.values())
        
        else:
            # Sequential processing
            for raw_file in self.raw_files:
                file_stats = {
                    "total_groups_examined": 0,
                    "groups_dropped_subsampling": 0,
                    "groups_dropped_validation": 0,
                    "groups_dropped_no_future": 0,
                    "groups_processed": 0,
                }
                
                with h5py.File(raw_file, "r") as f:
                    for gname in sorted(f.keys()):
                        file_stats["total_groups_examined"] += 1
                        
                        result = self._process_single_group(
                            f[gname], gname, helper, 
                            val_f, test_f, global_idx,
                            ratio_stats, file_stats
                        )
                        
                        if result is None:
                            profiles_skipped += 1
                            continue
                        
                        samples, split_key, n_written = result
                        shard_writer.add_samples(samples)
                        
                        start_idx = global_idx
                        global_idx += n_written
                        splits[split_key].extend(range(start_idx, global_idx))
                        profiles_written += 1
                        file_stats["groups_processed"] += 1
                
                # Update global stats
                for key, value in file_stats.items():
                    self.preprocessing_stats[key] += value
                self.preprocessing_stats["per_file_stats"][str(raw_file.name)] = file_stats

        # Finalize
        shard_writer.flush()
        self.shard_index["total_samples"] = global_idx
        self.preprocessing_stats["total_samples_generated"] = global_idx
        save_json(self.shard_index, self.output_dir / "shard_index.json")

        # Log statistics
        self.logger.info(f"Profiles written: {profiles_written}, skipped: {profiles_skipped}")

        for split_name, idxs in splits.items():
            if idxs:
                fname = self.shard_index["split_files"][split_name]
                np.save(self.output_dir / fname, np.array(idxs, dtype=np.int64))
                self.logger.info(f"{split_name} split: {len(idxs):,} samples")

        return splits
        
    def _parallel_write_shards_fixed(self, norm_stats, helper, val_f, test_f, 
                                    ratio_stats, start_global_idx):
        """Process files in parallel with fixed index handling."""
        results = []
        
        # Pre-calculate the number of valid samples per file
        self.logger.info("Pre-calculating sample counts for parallel processing...")
        file_sample_counts = []
        for raw_file in self.raw_files:
            count = self._count_valid_samples_in_file(raw_file)
            file_sample_counts.append(count)
            self.logger.debug(f"{raw_file.name}: {count} samples")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks with deterministic starting indices
            futures = []
            current_idx = start_global_idx
            
            for i, (raw_file, sample_count) in enumerate(zip(self.raw_files, file_sample_counts)):
                future = executor.submit(
                    self._process_file_for_shards_fixed,
                    raw_file, norm_stats, val_f, test_f,
                    current_idx, ratio_stats
                )
                futures.append((future, raw_file))
                current_idx += sample_count  # Increment by exact count
            
            # Collect results
            for future, raw_file in futures:
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed processing {raw_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {raw_file}: {e}")
                    raise
        
        return results
    
    def _count_valid_samples_in_file(self, raw_file: Path) -> int:
        """Count the exact number of samples that will be generated from a file."""
        count = 0
        use_fraction = self.config["training"]["use_fraction"]
        
        with h5py.File(raw_file, "r") as f:
            for gname in sorted(f.keys()):
                # Apply subsampling deterministically
                if use_fraction < 1.0:
                    h = hashlib.sha256(gname.encode("utf-8")).hexdigest()
                    if int(h[:8], 16) / 0xFFFFFFFF >= use_fraction:
                        continue
                
                # Check if group is valid
                if not self._validate_group_simple(f[gname]):
                    continue
                
                # Count samples (n_timesteps - 1)
                n_t = f[gname][self.time_var].shape[0]
                if n_t > 1:
                    count += (n_t - 1)
        
        return count
    
    def _process_file_for_shards_fixed(self, raw_file, norm_stats, val_f, test_f, 
                                      start_global_idx, ratio_stats):
        """Process a single file with fixed global indexing."""
        # Re-create helper in subprocess
        helper = NormalizationHelper(
            norm_stats,
            torch.device("cpu"),
            self.species_vars,
            self.global_vars,
            self.time_var,
            self.config,
        )
        
        file_samples = []
        file_splits = {"train": [], "validation": [], "test": []}
        profiles_written = 0
        profiles_skipped = 0
        global_idx = start_global_idx
        
        file_stats = {
            "total_groups_examined": 0,
            "groups_dropped_subsampling": 0,
            "groups_dropped_validation": 0,
            "groups_dropped_no_future": 0,
            "groups_processed": 0,
        }
        
        with h5py.File(raw_file, "r") as f:
            for gname in sorted(f.keys()):
                file_stats["total_groups_examined"] += 1
                
                result = self._process_single_group(
                    f[gname], gname, helper,
                    val_f, test_f, global_idx,
                    ratio_stats, file_stats
                )
                
                if result is None:
                    profiles_skipped += 1
                    continue
                
                samples, split_key, n_written = result
                file_samples.append(samples)
                
                # Use absolute indices
                start_idx = global_idx
                global_idx += n_written
                file_splits[split_key].extend(range(start_idx, global_idx))
                profiles_written += 1
                file_stats["groups_processed"] += 1
        
        return file_samples, file_splits, profiles_written, profiles_skipped, file_stats
        
    def _process_single_group(self, grp, gname, helper, val_f, test_f, 
                            global_idx, ratio_stats, file_stats=None):
        """Process a single group and return samples."""
        # Deterministic subsampling
        use_fraction = self.config["training"]["use_fraction"]
        if use_fraction < 1.0:
            h = hashlib.sha256(gname.encode("utf-8")).hexdigest()
            if int(h[:8], 16) / 0xFFFFFFFF >= use_fraction:
                if file_stats:
                    file_stats["groups_dropped_subsampling"] += 1
                return None
        
        if not self._validate_group_simple(grp):
            if file_stats:
                file_stats["groups_dropped_validation"] += 1
            return None
        
        # Extract & normalize
        n_t = grp[self.time_var].shape[0]
        if n_t <= 1:
            if file_stats:
                file_stats["groups_dropped_no_future"] += 1
            return None
            
        profile = self._extract_profile(grp, gname, n_t)
        if profile is None:
            if file_stats:
                file_stats["groups_dropped_validation"] += 1
            return None
        
        # Get samples based on prediction mode
        if self.prediction_mode == "ratio":
            samples = self._profile_to_samples_ratio(
                profile, n_t, helper, ratio_stats
            )
        else:
            profile_t = torch.from_numpy(profile)
            norm_prof = helper.normalize_profile(profile_t).numpy()
            samples = self._profile_to_samples(norm_prof, n_t)
        
        if samples is None:
            return None
        
        # Determine split
        split_h = hashlib.sha256((gname + "_split").encode("utf-8")).hexdigest()
        p = int(split_h[:8], 16) / 0xFFFFFFFF
        
        split_key = (
            "test" if p < test_f
            else "validation" if p < test_f + val_f
            else "train"
        )
        
        return samples, split_key, samples.shape[0]
    
    def _validate_group_simple(self, group: h5py.Group) -> bool:
        """Simplified validation - only check essentials."""
        # Check required variables exist
        required_vars = set(self.species_vars + [self.time_var])
        if not required_vars.issubset(group.keys()):
            return False
        
        # Check for minimum threshold violations and non-finite values
        for var in self.species_vars:
            try:
                var_data = group[var][:]
                if not np.all(np.isfinite(var_data)) or np.any(var_data < self.min_threshold):
                    return False
                # Additional check for ratio mode: ensure no exact zeros at t=0
                if self.prediction_mode == "ratio" and var_data[0] < self.config["normalization"]["epsilon"]:
                    self.logger.debug(f"Skipping group {group.name}: zero initial condition for {var}")
                    return False
            except Exception:
                return False
        
        # FIXED: Always verify globals are constant (remove conditional)
        for var in self.global_vars:
            if var in group:
                data = group[var][:]
                if not np.allclose(data, data[0], rtol=1e-10):
                    self.logger.warning(f"Global variable {var} is not constant in group {group.name}")
                    return False
        
        return True
    
    def _extract_profile(self, group: h5py.Group, gname: str, n_t: int) -> Optional[np.ndarray]:
        """Extract profile data from group."""
        import re
        
        # Define labels from global_vars by stripping "_init"
        global_labels = [var.replace("_init", "") for var in self.global_vars]
        globals_dict = {}
        
        # Find all _label_value patterns before SEED
        # More robust regex to handle scientific notation and spaces
        matches = re.findall(r"_(\w+)_([\d.eE+-]+)", gname)
        for label, value in matches:
            if label in global_labels:
                var = label + "_init"
                if var in self.global_vars:
                    try:
                        globals_dict[var] = float(value)
                    except ValueError:
                        self.logger.warning(f"Failed to parse value '{value}' for {var} in {gname}")
                        return None
        
        # Check if all global_vars were found
        if set(globals_dict.keys()) != set(self.global_vars):
            missing = set(self.global_vars) - set(globals_dict.keys())
            self.logger.debug(f"Missing global variables {missing} in {gname}")
            return None
        
        # Pre-allocate buffer
        profile = np.empty((n_t, self.n_vars), dtype=np.float32)
        
        # Fill data
        try:
            for i, var in enumerate(self.var_order):
                if var in self.species_vars or var == self.time_var:
                    profile[:, i] = group[var][:].astype(np.float32)
                elif var in self.global_vars:
                    profile[:, i] = globals_dict[var]
        except Exception as e:
            self.logger.warning(f"Failed to extract profile from {gname}: {e}")
            return None
        
        return profile
    
    def _profile_to_samples(self, normalized_profile: np.ndarray, n_t: int) -> Optional[np.ndarray]:
        """Convert profile to input-output samples with fixed time alignment."""
        if n_t <= 1:
            return None
        
        n_samples = n_t - 1
        n_features = self.n_species + self.n_globals + 1 + self.n_species
        
        samples = np.empty((n_samples, n_features), dtype=np.float32)
        
        # Initial species at t=0
        samples[:, :self.n_species] = normalized_profile[0, :self.n_species]
        
        # Initial globals at t=0
        initial_globals = normalized_profile[0, self.n_species:self.n_species + self.n_globals]
        samples[:, self.n_species:self.n_species + self.n_globals] = initial_globals
        
        # Time at t+1
        samples[:, self.n_species + self.n_globals] = normalized_profile[1:, -1]
        
        # Target species at t+1
        samples[:, -self.n_species:] = normalized_profile[1:, :self.n_species]
        
        return samples
    
    def _profile_to_samples_ratio(self, raw_profile: np.ndarray, n_t: int,
                                    helper: NormalizationHelper,
                                    ratio_stats: Dict[str, Dict[str, float]]) -> Optional[np.ndarray]:
            """Convert profile to samples for ratio mode with per-species standardization."""
            if n_t <= 1:
                return None

            # --- Prepare inputs ---
            n_samples = n_t - 1
            n_inputs = self.n_species + self.n_globals + 1
            n_targets = self.n_species
            samples = np.empty((n_samples, n_inputs + n_targets), dtype=np.float32)

            # Normalize the full profile to get normalized inputs
            profile_t = torch.from_numpy(raw_profile)
            norm_profile = helper.normalize_profile(profile_t).numpy()
            
            # Use initial state at t=0 for branch network inputs
            samples[:, :self.n_species + self.n_globals] = norm_profile[0, :self.n_species + self.n_globals]
            # Use time at t > 0 for trunk network input
            samples[:, self.n_species + self.n_globals] = norm_profile[1:, -1]

            # --- Prepare targets (Standardized Per-Species Log-Ratios) ---
            initial_species_raw = raw_profile[0, :self.n_species]
            future_species_raw = raw_profile[1:, :self.n_species]

            epsilon = self.config["normalization"]["epsilon"]
            ratios = future_species_raw / np.maximum(initial_species_raw[None, :], epsilon)
            log_ratios = np.log10(np.maximum(ratios, epsilon))

            # Get per-species mean and std for standardization from the dictionary
            ratio_means = np.array([ratio_stats[var]["mean"] for var in self.species_vars], dtype=np.float32)
            ratio_stds = np.array([ratio_stats[var]["std"] for var in self.species_vars], dtype=np.float32)
            
            # Standardize each column (species) with its own mean and std
            standardized_log_ratios = (log_ratios - ratio_means) / ratio_stds
            samples[:, -n_targets:] = standardized_log_ratios
            
            return samples

    def _generate_data_report(self, norm_stats: Dict[str, Any], split_indices: Dict[str, List[int]]):
        """Generate a comprehensive data preprocessing report."""
        report_path = Path(self.config["paths"]["log_dir"]) / "data_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA PREPROCESSING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction Mode: {self.prediction_mode}\n")
            f.write("\n")
            
            # File information
            f.write("INPUT FILES\n")
            f.write("-" * 40 + "\n")
            for raw_file in self.raw_files:
                f.write(f"  {raw_file.name}\n")
            f.write(f"Total files: {len(self.raw_files)}\n")
            f.write("\n")
            
            # Overall statistics
            f.write("PREPROCESSING STATISTICS\n")
            f.write("-" * 40 + "\n")
            total_examined = self.preprocessing_stats["total_groups_examined"]
            f.write(f"Total groups examined: {total_examined:,}\n")
            f.write(f"Groups dropped (subsampling): {self.preprocessing_stats['groups_dropped_subsampling']:,} "
                   f"({100 * self.preprocessing_stats['groups_dropped_subsampling'] / max(1, total_examined):.1f}%)\n")
            f.write(f"Groups dropped (validation): {self.preprocessing_stats['groups_dropped_validation']:,} "
                   f"({100 * self.preprocessing_stats['groups_dropped_validation'] / max(1, total_examined):.1f}%)\n")
            f.write(f"Groups dropped (no future): {self.preprocessing_stats['groups_dropped_no_future']:,} "
                   f"({100 * self.preprocessing_stats['groups_dropped_no_future'] / max(1, total_examined):.1f}%)\n")
            f.write(f"Groups processed: {self.preprocessing_stats['groups_processed']:,}\n")
            f.write(f"Total samples generated: {self.preprocessing_stats['total_samples_generated']:,}\n")
            f.write(f"Use fraction: {self.config['training']['use_fraction']}\n")
            f.write("\n")
            
            # Per-file breakdown
            if self.preprocessing_stats.get("per_file_stats"):
                f.write("PER-FILE BREAKDOWN\n")
                f.write("-" * 40 + "\n")
                for filename, stats in self.preprocessing_stats["per_file_stats"].items():
                    f.write(f"\n{filename}:\n")
                    f.write(f"  Groups examined: {stats['total_groups_examined']:,}\n")
                    f.write(f"  Groups processed: {stats['groups_processed']:,}\n")
                    f.write(f"  Dropped (subsampling): {stats['groups_dropped_subsampling']:,}\n")
                    f.write(f"  Dropped (validation): {stats['groups_dropped_validation']:,}\n")
                    f.write(f"  Dropped (no future): {stats['groups_dropped_no_future']:,}\n")
                f.write("\n")
            
            # Split information
            f.write("DATA SPLITS\n")
            f.write("-" * 40 + "\n")
            total_samples = sum(len(indices) for indices in split_indices.values())
            for split_name, indices in split_indices.items():
                n_samples = len(indices)
                percentage = 100 * n_samples / max(1, total_samples)
                f.write(f"{split_name:12s}: {n_samples:10,} samples ({percentage:5.1f}%)\n")
            f.write(f"{'Total':12s}: {total_samples:10,} samples\n")
            f.write("\n")
            
            # Normalization statistics summary
            f.write("NORMALIZATION STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write("Variable normalization methods:\n")
            for var, method in norm_stats["normalization_methods"].items():
                f.write(f"  {var:20s}: {method}\n")
            f.write("\n")
            
            # Summarize key statistics
            f.write("Key statistics per variable:\n")
            for var, stats in norm_stats.get("per_key_stats", {}).items():
                f.write(f"\n  {var}:\n")
                method = stats.get("method", "unknown")
                if method == "standard":
                    f.write(f"    Mean: {stats['mean']:.6e}, Std: {stats['std']:.6e}\n")
                elif method == "log-standard":
                    f.write(f"    Log Mean: {stats['log_mean']:.6f}, Log Std: {stats['log_std']:.6f}\n")
                elif method in ["min-max", "log-min-max"]:
                    f.write(f"    Min: {stats['min']:.6e}, Max: {stats['max']:.6e}\n")
            
            # Ratio mode statistics
            if self.prediction_mode == "ratio" and "ratio_stats" in norm_stats:
                f.write("\nRatio mode statistics (per-species log-ratios):\n")
                for var, stats in norm_stats["ratio_stats"].items():
                    f.write(f"\n  {var}:\n")
                    f.write(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}\n")
                    f.write(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}\n")
                    f.write(f"    Count: {stats['count']:,}\n")
            
            # Configuration summary
            f.write("\nCONFIGURATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Species variables: {', '.join(self.species_vars)}\n")
            f.write(f"Global variables: {', '.join(self.global_vars)}\n")
            f.write(f"Time variable: {self.time_var}\n")
            f.write(f"Shard size: {self.shard_size:,}\n")
            f.write(f"Min species threshold: {self.min_threshold:.2e}\n")
            f.write(f"Epsilon: {self.config['normalization']['epsilon']:.2e}\n")
            f.write(f"Min std: {self.config['normalization']['min_std']:.2e}\n")
            
        self.logger.info(f"Data report written to {report_path}")


class ShardWriter:
    """Efficient shard writer with buffering."""
    
    def __init__(self, output_dir: Path, shard_size: int, shard_index: Dict, compression: str = None):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_index = shard_index
        self.compression = compression
        
        self.buffer = []
        self.buffer_size = 0
        self.shard_id = 0
    
    def add_samples(self, samples: np.ndarray):
        """Add samples to buffer and flush if needed."""
        self.buffer.append(samples)
        self.buffer_size += samples.shape[0]
        
        while self.buffer_size >= self.shard_size:
            self._write_shard()
    
    def flush(self):
        """Write any remaining samples."""
        if self.buffer_size > 0:
            self._write_shard()
    
    def _write_shard(self):
        """Write a single shard to disk."""
        # Collect samples for one shard
        shard_data = []
        remaining = self.shard_size
        
        new_buffer = []
        for chunk in self.buffer:
            if remaining <= 0:
                new_buffer.append(chunk)
                continue
            
            if chunk.shape[0] <= remaining:
                shard_data.append(chunk)
                remaining -= chunk.shape[0]
            else:
                shard_data.append(chunk[:remaining])
                new_buffer.append(chunk[remaining:])
                remaining = 0
        
        # Concatenate shard data
        data = np.vstack(shard_data) if len(shard_data) > 1 else shard_data[0]
        
        # Write shard
        shard_path = self.output_dir / f"shard_{self.shard_id:04d}.npy"
        
        if self.compression == 'npz':
            save_path = shard_path.with_suffix('.npz')
            np.savez_compressed(save_path, data=data)
        else:
            save_path = shard_path
            np.save(save_path, data)
        
        # Update shard info with clarified indexing
        start_idx = self.shard_index.get("total_samples", 0)
        end_idx = start_idx + data.shape[0]
        shard_info = {
            "shard_idx": self.shard_id,
            "filename": save_path.name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_samples": data.shape[0],
        }
        
        self.shard_index["shards"].append(shard_info)
        self.shard_index["total_samples"] = end_idx
        self.shard_index["n_shards"] += 1
        self.shard_id += 1
        
        # Update buffer
        self.buffer = new_buffer
        self.buffer_size = sum(chunk.shape[0] for chunk in self.buffer)