#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor.
This version uses a highly efficient, parallelized, two-pass process with an
architecture that minimizes inter-process communication (IPC) and memory overhead.

--- UPDATED FEATURES ---
- Stricter Filtering: Drops entire profiles if they contain any non-finite values (NaN/inf)
  or any species value below a configurable 'min_value_threshold'.
- Summary Reporting: Generates a human-readable summary log detailing how many
  profiles were processed, kept, and dropped (with reasons).
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

from .normalizer import DataNormalizer, NormalizationHelper
from utils.utils import save_json, load_json


# ##############################################################################
# LIGHTWEIGHT WORKER-SIDE IMPLEMENTATION
# ##############################################################################

class CorePreprocessor:
    """A lightweight helper class containing only the logic needed within a worker."""
    def __init__(self, config: Dict[str, Any], norm_stats: Optional[Dict[str, Any]] = None):
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
        
        # Create index mappings for robust variable ordering (Bug 3 fix)
        self._create_index_mappings()
        
        if norm_stats:
            self.norm_helper = NormalizationHelper(
                norm_stats, torch.device("cpu"), self.species_vars,
                self.global_vars, self.time_var, config
            )
    
    def _create_index_mappings(self):
        """Create index mappings for robust variable access (Bug 3 fix)."""
        self.var_to_idx = {var: i for i, var in enumerate(self.var_order)}
        self.species_indices = [self.var_to_idx[var] for var in self.species_vars]
        self.global_indices = [self.var_to_idx[var] for var in self.global_vars]
        self.time_idx = self.var_to_idx[self.time_var]

    def _is_profile_valid(self, group: h5py.Group) -> Tuple[bool, str]:
        """
        Checks if a profile is valid according to strict criteria.
        Returns (is_valid, reason_for_failure_or_success).
        """
        # 1. Check for missing datasets
        required_keys = self.species_vars + [self.time_var]
        if not set(required_keys).issubset(group.keys()):
            return False, "missing_keys"

        # 2. Check each dataset for NaNs, Infs, and value thresholds
        for var in required_keys:
            try:
                data = group[var][:]
            except Exception:
                return False, "read_error"

            if not np.all(np.isfinite(data)):
                return False, "non_finite"

            if var in self.species_vars:
                if np.any(data < self.min_value_threshold):
                    return False, "below_threshold"
        
        return True, "valid"
    
    def process_file_for_stats(self, file_path: Path) -> Tuple[Dict, Dict, int, Dict]:
        """Worker logic for Pass 1: compute stats, counts, and a validation report."""
        accumulators = self.normalizer._initialize_accumulators()
        ratio_accumulators = {}
        if self.prediction_mode == "ratio":
            for var in self.species_vars:
                # Get the normalization method for this specific variable
                method = self.normalizer._get_method(var)
                
                # Only create an accumulator if the method is NOT "none"
                if method != "none":
                    ratio_accumulators[var] = {
                        "method": method,  # The CRITICAL fix: Add the method key
                        "count": 0,
                        "mean": 0.0,
                        "m2": 0.0,
                        "min": float("inf"),
                        "max": float("-inf")
                    }
        
        valid_sample_count = 0
        
        # --- NEW: Reporting dictionary for this worker ---
        report = {
            "total_profiles": 0,
            "profiles_kept": 0,
            "dropped_reasons": defaultdict(int)
        }
        
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                report["total_profiles"] += 1
                grp = f[gname]
                
                # --- NEW: Use the stricter validation function ---
                is_valid, reason = self._is_profile_valid(grp)
                if not is_valid:
                    report["dropped_reasons"][reason] += 1
                    continue

                use_fraction = self.train_cfg["use_fraction"]
                if use_fraction < 1.0 and int(hashlib.sha256(gname.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF >= use_fraction:
                    continue

                n_t = grp[self.time_var].shape[0]
                if n_t <= 1:
                    report["dropped_reasons"]["too_few_timesteps"] += 1
                    continue
                
                profile = self._extract_profile(grp, gname, n_t)
                if profile is None:
                    report["dropped_reasons"]["extract_profile_failed"] += 1
                    continue
                
                report["profiles_kept"] += 1
                valid_sample_count += (n_t - 1)
                self._update_stats_for_profile(profile, n_t, accumulators, ratio_accumulators)

        return accumulators, ratio_accumulators, valid_sample_count, report

    # --- UPDATED: Pass 2 now also uses the strict validation ---
    def process_file_for_shards(self, file_path: Path, output_dir: Path, start_idx: int) -> Dict[str, Any]:
        """Worker logic for Pass 2: process a file, write shards, and return metadata."""
        shard_idx_base = f"{file_path.stem}_{start_idx}"
        shard_writer = ShardWriter(output_dir, self.proc_cfg["shard_size"], shard_idx_base)
        
        splits = {"train": [], "validation": [], "test": []}
        current_idx = start_idx
        ratio_stats = self.norm_stats.get("ratio_stats", {})
        
        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                # --- NEW: Re-validate to ensure consistency between passes ---
                is_valid, _ = self._is_profile_valid(f[gname])
                if not is_valid:
                    continue
                
                use_fraction = self.train_cfg["use_fraction"]
                if use_fraction < 1.0 and int(hashlib.sha256(gname.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF >= use_fraction:
                    continue
                
                result = self._process_single_group(f[gname], gname, ratio_stats)
                if result is None:
                    continue
                
                samples, split_key = result
                n_written = samples.shape[0]
                shard_writer.add_samples(samples, current_idx)
                
                splits[split_key].extend(range(current_idx, current_idx + n_written))
                current_idx += n_written

        shard_writer.flush()
        return {
            "shards": shard_writer.get_shard_metadata(),
            "splits": splits,
            "rows_written": current_idx - start_idx,
        }

    def _update_stats_for_profile(self, profile, n_t, accumulators, ratio_accumulators):
        """Updated method with Bug 1 fix: consistent normalization in both modes."""
        import logging
        logger = logging.getLogger(__name__)
        
        if self.prediction_mode == "ratio":
            # Bug 1 Fix: Use full profiles for all variables in ratio mode
            for var, acc in accumulators.items():
                idx = acc["index"]
                method = acc["method"]
                
                # Use full profile data for all variables (not just initial timestep)
                if var == self.time_var and n_t > 1:
                    vec = profile[1:, idx]  # Time starts from t=1
                else:
                    vec = profile[:, idx]   # Full profile for species and globals
                
                if vec.size > 0:
                    if method.startswith("log-"):
                        vec = np.log10(np.maximum(vec, self.normalizer.epsilon))
                    self.normalizer._update_single_accumulator(acc, vec, var)
            
            # Compute ratio statistics correctly with proper indices (Bug 3 fix)
            initial = profile[0, self.species_indices]
            future = profile[1:, self.species_indices]
            
            ratios = future / np.maximum(initial[None, :], self.normalizer.epsilon)
            
            # Bug 5 Fix: Add logging for extreme values
            if np.any(ratios < self.normalizer.epsilon):
                n_below = np.sum(ratios < self.normalizer.epsilon)
                logger.warning(f"Found {n_below} ratio values below epsilon {self.normalizer.epsilon}")
            
            log_ratios = np.log10(np.maximum(ratios, self.normalizer.epsilon))
            
            # Bug 5 Fix: Log if clipping is needed
            if np.any(np.abs(log_ratios) > self.norm_cfg.get("clamp_value", 50.0)):
                n_clamped = np.sum(np.abs(log_ratios) > self.norm_cfg.get("clamp_value", 50.0))
                logger.warning(f"Clamping {n_clamped} extreme log-ratio values")
            
            for i, var_name in enumerate(self.species_vars):
                self.normalizer._update_single_accumulator(
                    ratio_accumulators[var_name], log_ratios[:, i], var_name
                )
        else:
            # Absolute mode - existing logic is correct
            for var, acc in accumulators.items():
                idx = acc["index"]
                vec = profile[1:, idx] if (var == self.time_var and n_t > 1) else profile[:, idx]
                if vec.size > 0:
                    self.normalizer._update_single_accumulator(acc, vec, var)

    def _process_single_group(self, grp, gname, ratio_stats) -> Optional[Tuple[np.ndarray, str]]:
        # The stricter validation is now done before this function is called.
        n_t = grp[self.time_var].shape[0]
        if n_t <= 1: return None
        profile = self._extract_profile(grp, gname, n_t)
        if profile is None: return None
        
        if self.prediction_mode == "ratio": samples = self._profile_to_samples_ratio(profile, n_t, ratio_stats)
        else: samples = self._profile_to_samples(self.norm_helper.normalize_profile(torch.from_numpy(profile)).numpy(), n_t)
        if samples is None: return None
        
        p = int(hashlib.sha256((gname + "_split").encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        split_key = "test" if p < self.train_cfg["test_fraction"] else "validation" if p < self.train_cfg["test_fraction"] + self.train_cfg["val_fraction"] else "train"
        return samples, split_key

    
    def _extract_profile(self, group: h5py.Group, gname: str, n_t: int) -> Optional[np.ndarray]:
        import re
        globals_dict = {f"{lbl}_init": float(val) for lbl, val in re.findall(r"_([A-Z])_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", gname) if f"{lbl}_init" in self.global_vars}
        if len(globals_dict) != len(self.global_vars): return None
        profile = np.empty((n_t, self.n_vars), dtype=np.float32)
        try:
            for i, var in enumerate(self.var_order):
                profile[:, i] = group[var][:] if var in group else globals_dict[var]
        except Exception: return None
        return profile

    def _profile_to_samples(self, norm_prof, n_t):
        """Updated method with Bug 3 fix: use proper indices."""
        if n_t <= 1:
            return None
        
        n_inputs = self.n_species + self.n_globals + 1
        samples = np.empty((n_t - 1, n_inputs + self.n_species), dtype=np.float32)
        
        # Use proper variable ordering (Bug 3 fix)
        samples[:, :self.n_species] = norm_prof[0, self.species_indices]  # Initial species
        samples[:, self.n_species:self.n_species + self.n_globals] = norm_prof[0, self.global_indices]  # Globals
        samples[:, n_inputs - 1] = norm_prof[1:, self.time_idx]  # Time
        samples[:, n_inputs:] = norm_prof[1:, self.species_indices]  # Target species
        
        return samples

    def _profile_to_samples_ratio(self, raw_prof, n_t, ratio_stats):
        """Updated method with Bug 3 fix: use proper indices."""
        import logging
        logger = logging.getLogger(__name__)
        
        if n_t <= 1:
            return None
        
        n_inputs = self.n_species + self.n_globals + 1
        samples = np.empty((n_t - 1, n_inputs + self.n_species), dtype=np.float32)
        
        # Normalize the profile
        norm_prof = self.norm_helper.normalize_profile(torch.from_numpy(raw_prof)).numpy()
        
        # Use proper variable ordering (Bug 3 fix)
        samples[:, :self.n_species] = norm_prof[0, self.species_indices]  # Initial species
        samples[:, self.n_species:self.n_species + self.n_globals] = norm_prof[0, self.global_indices]  # Globals
        samples[:, n_inputs - 1] = norm_prof[1:, self.time_idx]  # Time
        
        # Compute log-ratios with proper indices
        initial = raw_prof[0, self.species_indices]
        future = raw_prof[1:, self.species_indices]
        ratios = future / np.maximum(initial[None, :], self.norm_cfg["epsilon"])
        
        # Bug 5 Fix: Log extreme values before processing
        if np.any(ratios == 0):
            logger.warning(f"Found {np.sum(ratios == 0)} zero ratios - will be clamped to epsilon")
        
        log_ratios = np.log10(np.clip(ratios, 1e-38, 1e38))
        
        # Standardize log-ratios
        means = np.array([ratio_stats[v]["mean"] for v in self.species_vars], dtype=np.float32)
        stds = np.array([ratio_stats[v]["std"] for v in self.species_vars], dtype=np.float32)
        std_log_ratios = (log_ratios - means) / np.maximum(stds, self.norm_cfg["min_std"])
        
        # Bug 5 Fix: Log if clamping occurs
        clamp_val = self.norm_cfg.get("clamp_value", 50.0)
        if np.any(np.abs(std_log_ratios) > clamp_val):
            n_clamped = np.sum(np.abs(std_log_ratios) > clamp_val)
            logger.warning(f"Clamping {n_clamped} standardized log-ratio values to [-{clamp_val}, {clamp_val}]")
        
        samples[:, n_inputs:] = np.clip(std_log_ratios, -clamp_val, clamp_val)
        
        return samples

def stats_worker(file_path, config):
    return CorePreprocessor(config).process_file_for_stats(Path(file_path))

def shard_worker(file_path, config, norm_stats, start_idx, output_dir):
    return CorePreprocessor(config, norm_stats).process_file_for_shards(Path(file_path), Path(output_dir), start_idx)

# ##############################################################################
# MAIN PARENT PREPROCESSOR CLASS
# ##############################################################################

class DataPreprocessor:
    """Main parent class to orchestrate parallel data preprocessing."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.normalizer = DataNormalizer(config)
        self.num_workers = config["preprocessing"].get("num_workers", 1)
        self.parallel = self.num_workers > 1 and len(self.raw_files) > 1

    def process_to_npy_shards(self) -> None:
        """## MODIFIED ## Main entry point. Now ONLY creates core data (shards, normalization, index)."""
        start_time = time.time()
        self.logger.info(f"Starting core data preprocessing with {len(self.raw_files)} files...")
        
        norm_stats, file_sample_counts, summary_report = self._collect_stats_and_counts()
        save_json(norm_stats, self.output_dir / "normalization.json")

        all_shards = self._write_normalized_shards(norm_stats, file_sample_counts)
        
        total_samples = sum(s['n_samples'] for s in all_shards)
        shard_index = {
            "n_species": len(self.config["data"]["species_variables"]),
            "n_globals": len(self.config["data"]["global_variables"]),
            "samples_per_shard": self.config["preprocessing"]["shard_size"],
            "compression": self.config["preprocessing"].get("compression"),
            "prediction_mode": self.config.get("prediction", {}).get("mode", "absolute"),
            "shards": sorted(all_shards, key=lambda x: x['start_idx']),
            "n_shards": len(all_shards),
            "total_samples": total_samples,
            "split_files": { "train": "train_indices.npy", "validation": "val_indices.npy", "test": "test_indices.npy" }
        }
        save_json(shard_index, self.output_dir / "shard_index.json")

        self._write_summary_log(summary_report, total_samples)
        self.logger.info(f"Core data preprocessing completed in {time.time() - start_time:.1f}s")
    
    ## NEW FUNCTION ##
    def generate_split_indices(self) -> None:
        """Generates train/val/test split indices from an existing shard_index.json. This is a very fast operation."""
        self.logger.info("Generating new train/val/test split indices...")
        shard_index_path = self.output_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Cannot generate splits: shard_index.json not found in {self.output_dir}")
        
        shard_index = load_json(shard_index_path)
        total_samples = shard_index["total_samples"]
        indices = np.arange(total_samples)
        
        seed = self.config.get("system", {}).get("seed", 42)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        use_fraction = self.config["training"].get("use_fraction", 1.0)
        if use_fraction < 1.0:
            indices = indices[:int(total_samples * use_fraction)]
        
        n = len(indices)
        test_frac = self.config["training"]["test_fraction"]
        val_frac = self.config["training"]["val_fraction"]
        
        test_split_idx = int(n * test_frac)
        val_split_idx = test_split_idx + int(n * val_frac)
        
        split_data = {
            "test": np.sort(indices[:test_split_idx]).astype(np.int64),
            "validation": np.sort(indices[test_split_idx:val_split_idx]).astype(np.int64),
            "train": np.sort(indices[val_split_idx:]).astype(np.int64)
        }
        
        for name, idx_array in split_data.items():
            path = self.output_dir / shard_index["split_files"][name]
            np.save(path, idx_array)
            self.logger.info(f"Saved {name} indices to {path} ({len(idx_array)} samples)")

    def _write_normalized_shards(self, norm_stats, file_sample_counts) -> List[Dict]:
        """
        Second pass: write float32 shards + gather metadata.
        """
        self.logger.info("Writing shards …")
        all_meta = []
        current_start = 0  # Renamed to avoid shadowing

        with ProcessPoolExecutor(max_workers=self.num_workers if self.parallel else 1) as exe:
            # Precompute cumulative starts in O(n) time
            cumulative_starts = []
            for fp in self.raw_files:
                cumulative_starts.append(current_start)
                # Use fp.stem as key (assuming file_sample_counts uses stems; confirm in _collect_stats_and_counts)
                current_start += file_sample_counts.get(fp.stem, 0)

            # Submit all jobs with precomputed starts
            futures = [
                exe.submit(
                    shard_worker,
                    self.raw_files[i],
                    self.config,
                    norm_stats,
                    cumulative_starts[i],
                    self.output_dir
                )
                for i in range(len(self.raw_files))
            ]

            # Process results as they complete
            for fut in as_completed(futures):
                meta = fut.result()
                all_meta.extend(meta["shards"])

        return all_meta

    def _collect_stats_and_counts(self) -> Tuple[Dict, Dict, Dict]:
            self.logger.info("Pass 1: Collecting statistics and sample counts...")
            prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
            final_accs = self.normalizer._initialize_accumulators()
            final_ratio_accs = {var: {"count": 0,"mean": 0.0,"m2": 0.0,"min": float('inf'),"max": float('-inf')} for var in self.config["data"]["species_variables"]} if prediction_mode == "ratio" else {}
            file_counts = {}
            
            # --- NEW: Aggregate report dictionary ---
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
                    
                    # --- NEW: Aggregate the reports ---
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
        self.buffer: List[Tuple[np.ndarray, int]] = []
        self.buffer_size = 0
        self.local_shard_id = 0
        self.shard_metadata: List[Dict] = []
        # Ensure the output directory exists from within the worker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_samples(self, samples: np.ndarray, global_start_idx: int):
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        self.buffer.append((samples, global_start_idx))
        self.buffer_size += samples.shape[0]
        while self.buffer_size >= self.shard_size:
            self._write_shard()

    def flush(self):
        # Important: Use _write_shard in a loop to handle remaining data
        # that might still be larger than one shard.
        while self.buffer_size > 0:
            self._write_shard()
    
    def get_shard_metadata(self) -> List[Dict]:
        return self.shard_metadata

    def _write_shard(self) -> None:
        """
        Assembles and writes ONE shard of exactly `shard_size` (or less if flushing the remainder).
        This is the correct, original logic that respects shard boundaries.
        """
        if not self.buffer:
            return

        rows_to_write = []
        size_so_far = 0
        first_global_idx = self.buffer[0][1]

        # Collect arrays until we have enough for a shard
        while self.buffer and size_so_far < self.shard_size:
            arr, start_idx = self.buffer.pop(0)
            needed = self.shard_size - size_so_far
            
            if arr.shape[0] <= needed:
                # Take the whole array
                rows_to_write.append(arr)
                size_so_far += arr.shape[0]
            else:
                # Split the array
                rows_to_write.append(arr[:needed])
                # Put the remainder back at the front of the buffer
                self.buffer.insert(0, (arr[needed:], start_idx + needed))
                size_so_far += needed
        
        # Update the total buffer size
        self.buffer_size = sum(arr.shape[0] for arr, _ in self.buffer)

        # Concatenate and write the data for this shard
        data = np.concatenate(rows_to_write).astype(np.float32, copy=False)
        
        final_path = self.output_dir / f"shard_{self.shard_idx_base}_{self.local_shard_id:04d}.npy"
        tmp_path = final_path.with_suffix(".tmp.npy")

        np.save(tmp_path, data, allow_pickle=False)
        os.replace(tmp_path, final_path)

        self.shard_metadata.append({
            "filename": final_path.name,
            "start_idx": first_global_idx,
            "end_idx": first_global_idx + data.shape[0],
            "n_samples": data.shape[0],
        })
        self.local_shard_id += 1