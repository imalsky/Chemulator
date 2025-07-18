#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import h5py
import numpy as np
import torch


from .normalizer import DataNormalizer, NormalizationHelper
from utils.utils import save_json


class DataPreprocessor:
    """Preprocess HDF5 raw data to normalized NPY shards."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        # Sort the raw_files list to ensure a deterministic processing order.
        # This is critical for reproducible train/validation/test splits.
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
        }
        
    def process_to_npy_shards(self) -> Dict[str, Any]:
        """Two-pass processing: collect statistics then write normalized shards."""
        start_time = time.time()
        self.logger.info(f"Starting preprocessing with {len(self.raw_files)} files")
        
        # Pass 1: Collect statistics
        self.logger.info("Pass 1: Collecting normalization statistics")
        norm_stats = self._collect_statistics()
        
        # Pass 2: Write normalized shards
        self.logger.info("Pass 2: Writing normalized NPY shards")
        split_indices = self._write_normalized_shards(norm_stats)
        
        self.logger.info(f"Preprocessing completed in {time.time() - start_time:.1f}s")
        return split_indices
        
    def _collect_statistics(self) -> Dict[str, Any]:
        """First pass: collect normalization statistics."""
        self.normalizer = DataNormalizer(self.config)
        accumulators = self.normalizer._initialize_accumulators()
        
        for raw_file in self.raw_files:
            self._process_file_statistics(raw_file, accumulators)
        
        # Finalize statistics
        norm_stats = self.normalizer._finalize_statistics(accumulators)
        save_json(norm_stats, self.output_dir / "normalization.json")
        
        return norm_stats
    
    def _process_file_statistics(self, raw_file: Path, accumulators: Dict[str, Any]):
        """Process a single file for statistics collection."""
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
                
                # Validate group
                if not self._validate_group_simple(grp):
                    continue
                
                # Extract profile data
                n_t = grp[self.time_var].shape[0]
                profile = self._extract_profile(grp, gname, n_t)
                
                if profile is None:
                    continue
                
                # Update statistics
                profile_3d = profile.reshape(1, n_t, self.n_vars)
                self.normalizer._update_accumulators(profile_3d, accumulators, n_t)
                groups_processed += 1
        
        self.logger.info(f"  Processed {groups_processed} groups from {raw_file.name}")
        
    def _write_normalized_shards(self, norm_stats: Dict[str, Any]) -> Dict[str, List[int]]:
        """Second pass – write normalized data to NPY shards with fixed splitting."""
        # ─── setup ────────────────────────────────────────────────────────────────
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

        # ─── iterate over raw HDF5 files ─────────────────────────────────────────
        for raw_file in self.raw_files:
            with h5py.File(raw_file, "r") as f:
                for gname in sorted(f.keys()):
                    grp = f[gname]

                    # deterministic subsampling – must match statistics‑pass logic
                    use_fraction = self.config["training"]["use_fraction"]
                    if use_fraction < 1.0:
                        h = hashlib.sha256(gname.encode("utf-8")).hexdigest()
                        if int(h[:8], 16) / 0xFFFFFFFF >= use_fraction:
                            continue

                    if not self._validate_group_simple(grp):
                        continue

                    # extract & normalise
                    n_t     = grp[self.time_var].shape[0]
                    profile = self._extract_profile(grp, gname, n_t)
                    if profile is None:
                        continue

                    profile_t = torch.from_numpy(profile)
                    norm_prof = helper.normalize_profile(profile_t).numpy()

                    # convert to fixed‑alignment samples
                    samples = self._profile_to_samples(norm_prof, n_t)
                    if samples is None:
                        profiles_skipped += 1
                        continue

                    # BUG FIX: Only update indices for successfully written samples
                    n_written = samples.shape[0]
                    
                    # Add samples to writer
                    shard_writer.add_samples(samples)

                    # ─ split bookkeeping ─────────────────────────────────────────
                    split_h   = hashlib.sha256((gname + "_split").encode("utf-8")).hexdigest()
                    p         = int(split_h[:8], 16) / 0xFFFFFFFF

                    split_key = (
                        "test" if p < test_f
                        else "validation" if p < test_f + val_f
                        else "train"
                    )

                    start_idx = global_idx
                    global_idx += n_written
                    splits[split_key].extend(range(start_idx, global_idx))
                    profiles_written += 1

        # ─── finalise ────────────────────────────────────────────────────────────
        shard_writer.flush()
        self.shard_index["total_samples"] = global_idx
        save_json(self.shard_index, self.output_dir / "shard_index.json")

        # Log statistics
        self.logger.info(f"Profiles written: {profiles_written}, skipped: {profiles_skipped}")

        for split_name, idxs in splits.items():
            if idxs:
                fname = self.shard_index["split_files"][split_name]
                np.save(self.output_dir / fname, np.array(idxs, dtype=np.int64))
                self.logger.info(f"{split_name} split: {len(idxs):,} samples")

        return splits
    
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
            except Exception:
                return False
                
        return True
    
    def _extract_profile(self, group: h5py.Group, gname: str, n_t: int) -> Optional[np.ndarray]:
        """Extract profile data from group."""
        import re
        
        # Define labels from global_vars by stripping "_init"
        global_labels = [var.replace("_init", "") for var in self.global_vars]
        globals_dict = {}
        
        # Find all _label_value patterns before SEED
        matches = re.findall(r"_(\w+)_([\d.eE+-]+)", gname)
        for label, value in matches:
            if label in global_labels:
                var = label + "_init"
                if var in self.global_vars:
                    try:
                        globals_dict[var] = float(value)
                    except ValueError:
                        return None
        
        # Check if all global_vars were found
        if set(globals_dict.keys()) != set(self.global_vars):
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
        except Exception:
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
        
        # Update shard info
        shard_info = {
            "shard_idx": self.shard_id,
            "filename": save_path.name,
            "start_idx": self.shard_index.get("total_samples", 0),
            "end_idx": self.shard_index.get("total_samples", 0) + data.shape[0],
            "n_samples": data.shape[0],
        }
        
        self.shard_index["shards"].append(shard_info)
        self.shard_index["total_samples"] = shard_info["end_idx"]
        self.shard_index["n_shards"] += 1
        self.shard_id += 1
        
        # Update buffer
        self.buffer = new_buffer
        self.buffer_size = sum(chunk.shape[0] for chunk in self.buffer)