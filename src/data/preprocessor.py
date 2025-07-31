#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor. This runs before the rest of the training.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import torch
import os
import re
import sys

from .normalizer import DataNormalizer, NormalizationHelper
from utils.utils import save_json, load_json

DEFAULT_EPSILON_MIN = 1e-38
DEFAULT_EPSILON_MAX = 1e38

class CorePreprocessor:
    """A lightweight helper class containing only the logic needed within a worker."""
    def __init__(self, config: Dict[str, Any], norm_stats: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)

        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.pred_cfg = config.get("prediction", {})
        self.proc_cfg = config["preprocessing"]

        self.species_vars = self.data_cfg["species_variables"]
        self.global_vars = self.data_cfg["global_variables"]
        self.time_var = self.data_cfg["time_variable"]
        self.var_order = self.species_vars + self.global_vars + [self.time_var]
        
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)
        self.n_vars = self.n_species + self.n_globals + 1
        
        self.min_value_threshold = self.proc_cfg.get("min_value_threshold", 1e-30)
        
        self.prediction_mode = self.pred_cfg.get("mode", "absolute")
        self.normalizer = DataNormalizer(config)
        self.norm_stats = norm_stats or {}
        
        # Create index mappings for robust variable ordering
        self._create_index_mappings()
        
        if norm_stats:
            self.norm_helper = NormalizationHelper(norm_stats, torch.device("cpu"), 
                                                   self.species_vars,self.global_vars, self.time_var, config)
    
    def _create_index_mappings(self):
        """Create index mappings for robust variable access"""
        self.var_to_idx = {var: i for i, var in enumerate(self.var_order)}
        self.species_indices = [self.var_to_idx[var] for var in self.species_vars]
        self.global_indices = [self.var_to_idx[var] for var in self.global_vars]
        self.time_idx = self.var_to_idx[self.time_var]

    def _is_profile_valid(self, group: h5py.Group) -> Tuple[bool, str]:
        """
        Checks if a profile is valid according to criteria in config.
        Returns (is_valid, reason_for_failure_or_success).
        """
        # Validate every variable (species + globals + time)
        required_keys = self.species_vars + [self.time_var]
        if not set(required_keys).issubset(group.keys()):
            return False, "missing_keys"

        # Check each dataset for NaNs, Infs, and value thresholds
        for var in required_keys:
            try:
                data = group[var][:]
            except Exception:
                return False, "read_error"

            if not np.all(np.isfinite(data)):
                return False, "non_finite"

            # Drop profile if any value is less than or equal to threshold
            if np.any(data <= self.min_value_threshold):
                return False, "below_threshold"

        return True, "valid"
    
    def process_file_for_stats(self, file_path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict], int, Dict]:
        """Get a report of the raw data"""
        accumulators = self.normalizer._initialize_accumulators()

        ratio_accumulators: Dict[str, Dict[str, Any]] = {}
        if self.prediction_mode == "ratio":
            for v in self.species_vars:
                raw_method = self.normalizer._get_method(v)
                ratio_method = raw_method[4:] if raw_method.startswith("log-") else raw_method
                ratio_accumulators[v] = {
                    "method": ratio_method,
                    "count": 0,
                    "mean": 0.0,
                    "m2":   0.0,
                    "min":  float("inf"),
                    "max":  float("-inf"),
                }

        valid_sample_count = 0
        report = {
            "total_profiles":   0,
            "profiles_kept":    0,
            "dropped_reasons":  defaultdict(int),
        }

        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                report["total_profiles"] += 1
                grp = f[gname]

                # dataset‑level validation
                is_ok, reason = self._is_profile_valid(grp)
                if not is_ok:
                    report["dropped_reasons"][reason] += 1
                    continue

                # deterministic down‑sampling
                # check if whole dataset is being used
                if self.train_cfg["use_fraction"] < 1.0:
                    h = int(hashlib.sha256(gname.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if h >= self.train_cfg["use_fraction"]:
                        continue

                n_t = grp[self.time_var].shape[0]
                if n_t <= 1:
                    report["dropped_reasons"]["too_few_timesteps"] += 1
                    continue

                # assemble full profile (species + globals + time)
                profile = self._extract_profile(grp, gname, n_t)
                if profile is None:
                    report["dropped_reasons"]["extract_profile_failed"] += 1
                    continue

                # final profile‑level check
                if (np.any(~np.isfinite(profile)) or
                    np.any(profile <= self.min_value_threshold)):
                    report["dropped_reasons"]["below_threshold"] += 1
                    continue

                # update statistics
                report["profiles_kept"] += 1
                valid_sample_count += (n_t - 1)
                self._update_stats_for_profile(
                    profile, n_t,
                    accumulators,
                    ratio_accumulators,
                )

        return accumulators, ratio_accumulators, valid_sample_count, report

    def process_file_for_shards(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process file and write to split-specific shard directories."""
        # Create separate shard writers for each split
        shard_writers = {
            "train": ShardWriter(
                output_dir / "train", 
                self.proc_cfg["shard_size"], 
                file_path.stem
            ),
            "validation": ShardWriter(
                output_dir / "validation",
                self.proc_cfg["shard_size"],
                file_path.stem
            ),
            "test": ShardWriter(
                output_dir / "test",
                self.proc_cfg["shard_size"], 
                file_path.stem
            )
        }
        
        # Track samples per split
        split_counts = {"train": 0, "validation": 0, "test": 0}
        
        # Process file
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                # Validation checks
                is_valid, _ = self._is_profile_valid(f[gname])
                if not is_valid:
                    continue
                
                # Use fraction check
                use_fraction = self.train_cfg["use_fraction"]
                if use_fraction < 1.0:
                    hash_val = int(hashlib.sha256(gname.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if hash_val >= use_fraction:
                        continue
                
                # Process profile
                grp = f[gname]
                n_t = grp[self.time_var].shape[0]
                if n_t <= 1:
                    continue
                    
                profile = self._extract_profile(grp, gname, n_t)
                if profile is None:
                    continue

                if (np.any(~np.isfinite(profile)) or
                    np.any(profile <= self.min_value_threshold)):
                    continue

                # Determine split
                p = int(hashlib.sha256((gname + "_split").encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
                test_frac = self.train_cfg["test_fraction"]
                val_frac = self.train_cfg["val_fraction"]
                
                if p < test_frac:
                    split_key = "test"
                elif p < test_frac + val_frac:
                    split_key = "validation"
                else:
                    split_key = "train"
                
                # Convert profile to samples
                if self.prediction_mode == "ratio":
                    samples = self._profile_to_samples_ratio(profile, n_t, self.norm_stats.get("ratio_stats", {}))
                else:
                    norm_prof = self.norm_helper.normalize_profile(torch.from_numpy(profile)).numpy()
                    samples = self._profile_to_samples(norm_prof, n_t)
                
                if samples is not None:
                    # Add to appropriate writer - no need to track indices
                    shard_writers[split_key].add_samples(samples)
                    split_counts[split_key] += samples.shape[0]
        
        # Flush all writers
        for writer in shard_writers.values():
            writer.flush()
        
        # Return metadata
        return {
            "splits": {
                "train": {
                    "shards": shard_writers["train"].get_shard_metadata(),
                    "samples_written": split_counts["train"]
                },
                "validation": {
                    "shards": shard_writers["validation"].get_shard_metadata(),
                    "samples_written": split_counts["validation"]
                },
                "test": {
                    "shards": shard_writers["test"].get_shard_metadata(),
                    "samples_written": split_counts["test"]
                }
            }
        }

    def _update_stats_for_profile(self, profile, n_t, accumulators, ratio_accumulators):
        """Consistent normalization in both modes."""
        if self.prediction_mode == "ratio":
            # Use full profiles for all variables in ratio mode
            for var, acc in accumulators.items():
                idx = acc["index"]
                method = acc["method"]
                
                # Use full profile data for all variables (not just initial timestep)
                if var == self.time_var and n_t > 1:
                    vec = profile[1:, idx]
                else:
                    vec = profile[:, idx]
                
                if vec.size > 0:
                    self.normalizer._update_single_accumulator(acc, vec, var)
            
            # Compute ratio statistics correctly with proper indices
            initial = profile[0, self.species_indices]
            future = profile[1:, self.species_indices]
            
            ratios = future / np.maximum(initial[None, :], self.normalizer.epsilon)
            ratios = np.clip(ratios, -DEFAULT_EPSILON_MAX, DEFAULT_EPSILON_MAX)
            log_ratios = np.sign(ratios) * np.log10( np.clip(np.abs(ratios), DEFAULT_EPSILON_MIN, DEFAULT_EPSILON_MAX))


            for i, var_name in enumerate(self.species_vars):
                self.normalizer._update_single_accumulator(ratio_accumulators[var_name], log_ratios[:, i], var_name)
        else:
            # Absolute mode
            for var, acc in accumulators.items():
                idx = acc["index"]
                vec = profile[1:, idx] if (var == self.time_var and n_t > 1) else profile[:, idx]
                if vec.size > 0:
                    self.normalizer._update_single_accumulator(acc, vec, var)
    


    def _extract_profile(self, group: h5py.Group, gname: str, n_t: int) -> Optional[np.ndarray]:
        """
        Extracts a full data profile from an HDF5 group with strict validation.
        """

        config_globals = set(self.global_vars)
        group_attrs = set(group.attrs.keys())
        
        if 'SEED' in group_attrs:
            group_attrs.remove('SEED')

        globals_missing_from_attrs = config_globals - group_attrs
        attrs_not_in_globals = group_attrs - config_globals
        
        if globals_missing_from_attrs or attrs_not_in_globals:
            self.logger.critical(f"MISMATCH in HDF5 attributes for group: '{gname}'")
            if globals_missing_from_attrs:
                self.logger.critical(f"CONFIG ERROR: The following variables" 
                                     f"are in your config but MISSING from the HDF5"
                                     f"attributes: {list(globals_missing_from_attrs)}")
            if attrs_not_in_globals:
                self.logger.critical(f"DATA ERROR: The following HDF5 attributes are"
                                     f"UNUSED in your config's 'global_variables': {list(attrs_not_in_globals)}")
            self.logger.critical("  > Program will now terminate to enforce data integrity.")
            sys.exit(1) 

        globals_dict = {key: float(group.attrs[key]) for key in self.global_vars}
        profile = np.empty((n_t, self.n_vars), dtype=np.float64)

        try:
            for i, var_name in enumerate(self.var_order):
                if var_name in group:
                    profile[:, i] = group[var_name][:]
                elif var_name in globals_dict:
                    profile[:, i] = globals_dict[var_name]
                else:
                    self.logger.error(f"FATAL LOGIC ERROR for group '{gname}'"
                                      f"Variable '{var_name}' was not found."
                                      f"Please check the preprocessor code.")
                    sys.exit(1)

        except Exception as e:
            self.logger.error(f"Failed to read dataset while assembling profile for group '{gname}': {e}")
            return None

        return profile

    def _profile_to_samples(self, norm_prof, n_t):
        """Build (n_t‑1) samples for absolute mode."""
        if n_t <= 1:
            return None
        
        n_inputs = self.n_species + self.n_globals + 1
        samples = np.empty((n_t - 1, n_inputs + self.n_species), dtype=np.float64)
        
        # Initial species
        samples[:, :self.n_species] = norm_prof[0, self.species_indices] 

        # Globals
        samples[:, self.n_species:self.n_species + self.n_globals] = norm_prof[0, self.global_indices]

        # Time
        samples[:, n_inputs - 1] = norm_prof[1:, self.time_idx]

        # Target species
        samples[:, n_inputs:] = norm_prof[1:, self.species_indices]
        
        return samples

    def _profile_to_samples_ratio(
        self, raw_prof: np.ndarray, n_t: int, ratio_stats: Dict[str, Dict]
    ) -> Optional[np.ndarray]:
        """
        Build (n_t‑1) samples for ratio‑prediction mode.
        raw_prof is **unnormalised** profile array  shape = (n_t, n_vars)
        ratio_stats contains mean/std/min/max for log‑ratios (already computed)
        """
        logger = logging.getLogger(__name__)

        if n_t <= 1:
            return None
        
        n_inputs  = self.n_species + self.n_globals + 1
        samples   = np.empty((n_t - 1, n_inputs + self.n_species), dtype=np.float64)

        norm_prof = self.norm_helper.normalize_profile(torch.from_numpy(raw_prof)).numpy()
        samples[:, :self.n_species]                   = norm_prof[0, self.species_indices]
        samples[:, self.n_species:self.n_species+self.n_globals] = norm_prof[0, self.global_indices]
        samples[:, n_inputs - 1]                      = norm_prof[1:, self.time_idx]

        initial = raw_prof[0, self.species_indices]
        future  = raw_prof[1:, self.species_indices]

        ratios = future / np.maximum(initial[None, :], self.norm_cfg["epsilon"])
        ratios = np.clip(ratios, -DEFAULT_EPSILON_MAX, DEFAULT_EPSILON_MAX)
        log_ratios = np.sign(ratios) * np.log10(np.clip(np.abs(ratios), DEFAULT_EPSILON_MIN, DEFAULT_EPSILON_MAX))

        methods_cfg = self.norm_cfg.get("methods", {})
        default_m   = self.norm_cfg.get("default_method", "standard")
        clamp_val   = self.norm_cfg.get("clamp_value", 50.0)
        min_std     = self.norm_cfg.get("min_std", 1e-10)

        normd = np.empty_like(log_ratios, dtype=np.float64)

        for i, var in enumerate(self.species_vars):
            method = methods_cfg.get(var, default_m)
            if method.startswith("log-"):
                method = method[4:]

            stats = ratio_stats[var]

            if method == "min-max":
                rng = max(stats["max"] - stats["min"], self.norm_cfg["epsilon"])
                normd[:, i] = (log_ratios[:, i] - stats["min"]) / rng
            else:
                std = max(stats["std"], min_std)
                normd[:, i] = (log_ratios[:, i] - stats["mean"]) / std

        if np.any(np.abs(normd) > clamp_val):
            n_clamped = np.sum(np.abs(normd) > clamp_val)
            logger.warning(f"Clamping {n_clamped} normalised log‑ratio values to ±{clamp_val}")

        samples[:, n_inputs:] = np.clip(normd, -clamp_val, clamp_val)
        return samples

def stats_worker(file_path, config):
    return CorePreprocessor(config).process_file_for_stats(Path(file_path))

class DataPreprocessor:
    """Main parent class to orchestrate parallel data preprocessing."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)

        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir

        self.processed_dir = self.output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        
        self.normalizer = DataNormalizer(config)
        self.num_workers = config["preprocessing"].get("num_workers", 1)
        self.parallel = self.num_workers > 1 and len(self.raw_files) > 1

    def process_to_npy_shards(self) -> None:
        """Main entry point - creates split-specific shard directories."""
        start_time = time.time()
        self.logger.info(f"Starting core data preprocessing with {len(self.raw_files)} files...")
        
        # Collect statistics
        norm_stats, file_sample_counts, summary_report = self._collect_stats_and_counts()
        save_json(norm_stats, self.output_dir / "normalization.json")

        # Write split-specific shards
        split_metadata = self._write_normalized_shards(norm_stats, file_sample_counts)
        
        # Save split-aware shard index
        shard_index = {
            "n_species": len(self.config["data"]["species_variables"]),
            "n_globals": len(self.config["data"]["global_variables"]),
            "samples_per_shard": self.config["preprocessing"]["shard_size"],
            "compression": self.config["preprocessing"].get("compression"),
            "prediction_mode": self.config.get("prediction", {}).get("mode", "absolute"),
            "splits": split_metadata,
            "total_samples": sum(meta["total_samples"] for meta in split_metadata.values())
        }
        save_json(shard_index, self.output_dir / "shard_index.json")

        self._write_summary_log(summary_report, shard_index["total_samples"])
        self.logger.info(f"Core data preprocessing completed in {time.time() - start_time:.1f}s")
        

    def _write_normalized_shards(self, norm_stats, file_sample_counts) -> Dict[str, List[Dict]]:
        """Second pass: write split-specific shards to separate directories."""
        self.logger.info("Writing split-specific shards...")
        
        # Create split directories
        split_dirs = {
            "train": self.processed_dir / "train",
            "validation": self.processed_dir / "validation", 
            "test": self.processed_dir / "test"
        }
        for split_name, dir_path in split_dirs.items():
            dir_path.mkdir(exist_ok=True)
            self.logger.info(f"Created {split_name} directory: {dir_path}")
        
        # Initialize split metadata
        split_metadata = {
            "train": {"shards": [], "total_samples": 0},
            "validation": {"shards": [], "total_samples": 0},
            "test": {"shards": [], "total_samples": 0}
        }
        
        # Process each file
        with ProcessPoolExecutor(max_workers=self.num_workers if self.parallel else 1) as exe:
            futures = []
            
            for file_path in self.raw_files:
                future = exe.submit(
                    shard_worker_split_aware,
                    file_path,
                    self.config,
                    norm_stats,
                    self.processed_dir
                )
                futures.append((future, file_path))
            
            # Collect results
            for future, file_path in futures:
                result = future.result()
                
                # Aggregate metadata by split
                for split_name in ["train", "validation", "test"]:
                    split_meta = result["splits"][split_name]
                    split_metadata[split_name]["shards"].extend(split_meta["shards"])
                    split_metadata[split_name]["total_samples"] += split_meta["samples_written"]
        
        # Sort shards within each split and assign global indices
        for split_name, meta in split_metadata.items():
            # Sort shards by filename
            meta["shards"].sort(key=lambda x: x["filename"])
            
            # Assign sequential start/end indices
            current_idx = 0
            for shard in meta["shards"]:
                shard["start_idx"] = current_idx
                shard["end_idx"] = current_idx + shard["n_samples"]
                current_idx = shard["end_idx"]
            
            self.logger.info(f"{split_name}: {len(meta['shards'])} shards, {meta['total_samples']:,} samples")
        
        return split_metadata
     
    def _collect_stats_and_counts(self) -> Tuple[Dict, Dict, Dict]:
        self.logger.info("Pass 1: Collecting statistics and sample counts...")
        prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
        final_accs = self.normalizer._initialize_accumulators()
        final_ratio_accs = {var: {"count": 0,"mean": 0.0,
                                  "m2": 0.0,"min": float('inf'),
                                  "max": float('-inf')} for var in self.config["data"]["species_variables"]} if prediction_mode == "ratio" else {}
        file_counts = {}
        
        total_report = {
            "total_profiles": 0,
            "profiles_kept": 0,
            "dropped_reasons": defaultdict(int)
        }

        with ProcessPoolExecutor(max_workers=self.num_workers if self.parallel else 1) as executor:
            futures = {executor.submit(stats_worker, fp, self.config): fp for fp in self.raw_files}
            for fut in as_completed(futures):
                accs, ratio_accs, count, worker_report = fut.result()
                
                # Aggregate results
                self.normalizer._merge_accumulators(final_accs, accs)
                if ratio_accs: self.normalizer._merge_accumulators(final_ratio_accs, ratio_accs)
                file_counts[futures[fut].name] = count
                
                # Aggregate the reports ---
                total_report["total_profiles"] += worker_report["total_profiles"]
                total_report["profiles_kept"] += worker_report["profiles_kept"]
                for reason, num in worker_report["dropped_reasons"].items():
                    total_report["dropped_reasons"][reason] += num

        norm_stats = self.normalizer._finalize_statistics(final_accs)
        if final_ratio_accs:
            norm_stats["ratio_stats"] = self.normalizer._finalize_statistics(final_ratio_accs, is_ratio=True)

        file_counts = {Path(k).stem: v for k, v in file_counts.items()}

        return norm_stats, file_counts, total_report

    def _write_summary_log(self, report: Dict, total_samples: int):
        """Writes a human-readable summary of the preprocessing results."""
        log_dir = Path(self.config["paths"]["log_dir"])
        log_dir.mkdir(exist_ok=True)
        summary_path = log_dir / f"preprocessing_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        dropped_count = report["total_profiles"] - report["profiles_kept"]
        
        reason_map = {
            "missing_keys": "Required dataset keys were missing",
            "non_finite": "Contained NaN or Infinity values",
            "below_threshold": f"A species value was below the threshold ({self.config['preprocessing']['min_value_threshold']:.1e})",
            "too_few_timesteps": "Contained 1 or fewer time steps",
            "extract_profile_failed": "Failed to extract global variables from name",
            "read_error": "Could not read a dataset from the HDF5 group"
        }

        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("      Data Preprocessing Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw Files Processed: {len(self.raw_files)}\n\n")

            f.write("--- Profile Filtering --- \n")
            f.write(f"Total Profiles Found:     {report['total_profiles']:,}\n")
            f.write(f"Profiles Kept:            {report['profiles_kept']:,}\n")
            f.write(f"Profiles Dropped:         {dropped_count:,}\n\n")
            
            if dropped_count > 0:
                f.write("--- Reasons for Dropped Profiles ---\n")
                for reason, count in sorted(report["dropped_reasons"].items()):
                    f.write(f"  - {count:>10,} : {reason_map.get(reason, reason)}\n")
                f.write("\n")

            f.write("--- Final Sample Count ---\n")
            f.write(f"Total Usable Samples:     {total_samples:,}\n")
            f.write("(Train/Val/Test splits generated separately)\n")

        self.logger.info(f"Preprocessing summary saved to: {summary_path}")


class ShardWriter:
    """Writes numpy arrays to shard files, handling buffering and file naming."""
    def __init__(self, output_dir: Path, shard_size: int, shard_idx_base: str):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx_base = shard_idx_base
        self.buffer: List[np.ndarray] = []
        self.buffer_size = 0
        self.local_shard_id = 0
        self.shard_metadata: List[Dict] = []
        
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_samples(self, samples: np.ndarray):
        """Simplified add_samples without tracking global indices."""
        if samples.dtype != np.float64:
            samples = samples.astype(np.float64)
        
        self.buffer.append(samples)
        self.buffer_size += samples.shape[0]
        
        while self.buffer_size >= self.shard_size:
            self._write_shard()

    def _write_shard(self) -> None:
        """Write one shard of exactly shard_size samples (or less if flushing)."""
        if not self.buffer:
            return

        rows_to_write = []
        size_so_far = 0

        # Collect arrays until we have enough for a shard
        while self.buffer and size_so_far < self.shard_size:
            arr = self.buffer.pop(0)
            needed = self.shard_size - size_so_far
            
            if arr.shape[0] <= needed:
                rows_to_write.append(arr)
                size_so_far += arr.shape[0]
            else:
                # Split the array
                rows_to_write.append(arr[:needed])
                self.buffer.insert(0, arr[needed:])
                size_so_far += needed
        
        # Update buffer size
        self.buffer_size = sum(arr.shape[0] for arr in self.buffer)

        # Write shard
        data = np.concatenate(rows_to_write).astype(np.float64, copy=False)
        
        final_path = self.output_dir / f"shard_{self.shard_idx_base}_{self.local_shard_id:04d}.npy"
        tmp_path = final_path.with_suffix(".tmp.npy")

        np.save(tmp_path, data, allow_pickle=False)
        os.replace(tmp_path, final_path)

        self.shard_metadata.append({
            "filename": final_path.name,
            "n_samples": data.shape[0],
        })
        
        self.local_shard_id += 1

    def flush(self) -> None:
        """Write out any rows left in the buffer (< shard_size)."""
        if self.buffer_size == 0:
            return
        
        # write whatever is left as a final shard
        rows_to_write = self.buffer
        self.buffer = []
        self.buffer_size = 0

        data = np.concatenate(rows_to_write).astype(np.float64, copy=False)
        final_path = self.output_dir / f"shard_{self.shard_idx_base}_{self.local_shard_id:04d}.npy"
        tmp_path   = final_path.with_suffix(".tmp.npy")
        np.save(tmp_path, data, allow_pickle=False)
        os.replace(tmp_path, final_path)

        self.shard_metadata.append({
            "filename": final_path.name,
            "n_samples": data.shape[0],
        })
        self.local_shard_id += 1

    def get_shard_metadata(self) -> List[Dict]:
        """Return metadata collected for all shards."""
        return list(self.shard_metadata)
    
def shard_worker_split_aware(file_path, config, norm_stats, output_dir):
    """Worker function that creates split-specific shards."""
    processor = CorePreprocessor(config, norm_stats)
    return processor.process_file_for_shards(Path(file_path), Path(output_dir))   