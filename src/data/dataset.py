#!/usr/bin/env python3
"""
High-performance dataset implementation with sequence mode support for LiLaN.
This version uses a config-driven NormalizationHelper to correctly process
both inputs and targets into a standardized space.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math

# Import the corrected NormalizationHelper
from data.normalizer import NormalizationHelper


class SequenceDataset(Dataset):
    """
    Dataset for sequence mode (trajectory-based) data that correctly applies
    a config-driven normalization scheme to both inputs and targets.
    """
    def __init__(self, shard_dir: Path, split_name: str, config: Dict[str, Any],
                 device: torch.device, norm_stats: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Get dtype from config
        dtype_str = self.config["system"].get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)
        self.np_dtype = np.float32 if dtype_str == "float32" else np.float64

        self.norm_stats = norm_stats or {}

        # Load metadata from the master index file
        shard_index_path = self.shard_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
        with open(shard_index_path) as f:
            self.shard_index = json.load(f)

        if not self.shard_index.get("sequence_mode", False):
            raise ValueError("Expected sequence mode data for SequenceDataset")

        # Extract dimensions and variable lists from config
        self.split_info = self.shard_index["splits"][split_name]
        self.M = self.shard_index["M_per_sample"]
        self.species_vars = self.config["data"]["species_variables"]
        self.global_vars = self.config["data"]["global_variables"]
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)

        # Time normalization parameters (handled separately)
        self.time_norm = self.shard_index.get("time_normalization", {})
        self.tau0 = self.time_norm.get("tau0", 1.0)
        self.tmin = self.time_norm.get("tmin", 0.0)
        self.tmax = self.time_norm.get("tmax", 1.0)

        # Build shard info and lookup table
        self.shards = self.split_info.get("shards", [])
        self.n_total_samples = self.split_info.get("n_trajectories", 0)

        if self.n_total_samples == 0:
            self.logger.warning(f"No samples found in '{split_name}' split.")
        
        self._build_shard_lookup()

        # --- CRITICAL CHANGE: Instantiate the NormalizationHelper ---
        self.norm_helper = None
        if self.norm_stats and self.norm_stats.get("per_key_stats"):
            # Instantiate on CPU. It will be used for pre-caching data before moving to GPU.
            self.norm_helper = NormalizationHelper(self.norm_stats, torch.device("cpu"), self.config)
            self.logger.info(f"NormalizationHelper initialized for '{split_name}' dataset.")
        else:
            self.logger.warning(f"No normalization stats found. '{split_name}' dataset will use raw data.")

        # Caching setup
        self.gpu_cache = None
        self._shard_cache = {"name": None, "data": None} # Per-worker cache for CPU fallback
        self._try_gpu_cache()

    def _build_shard_lookup(self):
        """Build lookup arrays for shard access. Correct as is."""
        self.shard_starts = [0]
        cumsum = 0
        for shard in self.shards:
            cumsum += shard["n_samples"]
            self.shard_starts.append(cumsum)
        # Use int64 for robust searchsorted math, especially with large datasets
        self.shard_starts = np.array(self.shard_starts[:-1], dtype=np.int64)

    def _normalize_time(self, t: np.ndarray) -> np.ndarray:
        """Apply global log-min-max time normalization. Correct as is."""
        tau = np.log(1 + t / self.tau0)
        return np.clip((tau - self.tmin) / max(self.tmax - self.tmin, 1e-10), 0, 1)

    def _try_gpu_cache(self):
        """
        Corrected method. Tries to load, fully normalize, and cache the entire
        dataset on the GPU. Normalizes both inputs AND targets.
        """
        if self.n_total_samples == 0:
            return

        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False or self.device.type != "cuda":
            return

        # --- Memory budget calculation (correct as is) ---
        bytes_per_float = 8 if self.dtype == torch.float64 else 4
        # Note: This is an estimate of the *final* data size. Raw data may be larger.
        bytes_needed = self.n_total_samples * (
            self.n_species + self.n_globals + self.M + self.n_species * self.M
        ) * bytes_per_float

        try:
            idx = 0 if self.device.index is None else self.device.index
            total_mem = torch.cuda.get_device_properties(idx).total_memory
            tcfg = self.config.get("training", {})
            max_frac = float(tcfg.get("gpu_cache_max_fraction", 0.5))
            reserve_bytes = int(tcfg.get("gpu_cache_reserved_bytes", 2e9))
            budget = max(0, int(total_mem * max_frac) - reserve_bytes)

            if bytes_needed > budget:
                self.logger.warning(
                    f"GPU cache for '{self.split_name}' disabled: "
                    f"need {bytes_needed/1e9:.2f} GB > budget {budget/1e9:.2f} GB"
                )
                return
        except Exception as e:
            self.logger.warning(f"Could not assess GPU memory for caching, disabling. Error: {e}")
            return

        # --- Data Loading and Normalization ---
        self.logger.info(f"Loading and normalizing '{self.split_name}' data for GPU cache ({bytes_needed/1e9:.1f} GB)...")
        
        all_x0_norm, all_g_norm, all_t_norm, all_y_norm = [], [], [], []

        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                # Load raw data from shards and convert to CPU tensors
                x0_log = torch.from_numpy(data["x0_log"].astype(self.np_dtype))
                g_vec = torch.from_numpy(data["globals"].astype(self.np_dtype))
                t_vec_raw = data["t_vec"]
                y_mat_log = torch.from_numpy(data["y_mat"].astype(self.np_dtype))

                # Normalize time separately
                t_vec_norm = torch.from_numpy(self._normalize_time(t_vec_raw).astype(self.np_dtype))
                
                # Normalize all other features using the helper
                if self.norm_helper:
                    x0_norm = self.norm_helper.normalize(x0_log, self.species_vars)
                    g_norm = self.norm_helper.normalize(g_vec, self.global_vars)
                    y_norm = self.norm_helper.normalize(y_mat_log, self.species_vars)
                else:
                    x0_norm, g_norm, y_norm = x0_log, g_vec, y_mat_log

                all_x0_norm.append(x0_norm)
                all_g_norm.append(g_norm)
                all_t_norm.append(t_vec_norm)
                all_y_norm.append(y_norm)
        
        # Concatenate all normalized tensors on the CPU
        x0_full = torch.cat(all_x0_norm, dim=0)
        g_full = torch.cat(all_g_norm, dim=0)
        t_full = torch.cat(all_t_norm, dim=0)
        y_full = torch.cat(all_y_norm, dim=0)

        # Pre-build final inputs and targets tensors
        inputs_full = torch.cat([x0_full, g_full, t_full], dim=1)

        # Move final, fully-processed tensors to GPU in the correct dtype
        self.gpu_cache = {
            "inputs": inputs_full.to(device=self.device, dtype=self.dtype, non_blocking=True),
            "targets": y_full.to(device=self.device, dtype=self.dtype, non_blocking=True),
        }
        self.logger.info(f"GPU cache for '{self.split_name}' created successfully.")

    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrected method for CPU fallback. Loads a single item from a shard
        and applies the full, correct normalization to both inputs and targets.
        """
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]
        shard_info = self.shards[shard_idx]

        # LRU-1 cache for raw shard data to reduce disk reads by workers
        if self._shard_cache.get("name") != shard_info["filename"]:
            with np.load(self.split_dir / shard_info["filename"], allow_pickle=False) as data:
                self._shard_cache = {"name": shard_info["filename"], "data": dict(data)}
        
        raw_data = self._shard_cache["data"]

        # Extract raw data for the specific index and convert to CPU tensors
        x0_log = torch.from_numpy(raw_data['x0_log'][local_idx].astype(self.np_dtype))
        g_vec = torch.from_numpy(raw_data['globals'][local_idx].astype(self.np_dtype))
        t_vec_raw = raw_data['t_vec'][local_idx]
        y_mat_log = torch.from_numpy(raw_data['y_mat'][local_idx].astype(self.np_dtype))

        # Normalize time
        t_vec_norm = torch.from_numpy(self._normalize_time(t_vec_raw).astype(self.np_dtype))

        # Normalize all other features using the helper
        if self.norm_helper:
            x0_norm = self.norm_helper.normalize(x0_log, self.species_vars)
            g_norm = self.norm_helper.normalize(g_vec, self.global_vars)
            y_norm = self.norm_helper.normalize(y_mat_log, self.species_vars)
        else:
            x0_norm, g_norm, y_norm = x0_log, g_vec, y_mat_log

        # Build final input and target tensors in the correct data type
        inputs_tensor = torch.cat([x0_norm, g_norm, t_vec_norm]).to(self.dtype)
        targets_tensor = y_norm.to(self.dtype)
        
        return inputs_tensor, targets_tensor

    def __len__(self) -> int:
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < self.n_total_samples:
            raise IndexError(f"Index {idx} is out of range for dataset with size {self.n_total_samples}")

        if self.gpu_cache is not None:
            return self.gpu_cache["inputs"][idx], self.gpu_cache["targets"][idx]
        else:
            return self._load_from_disk(idx)

# This function is correct and does not need changes.
def create_dataloader(dataset: Dataset,
                      config: Dict[str, Any],
                      shuffle: bool = True,
                      device: Optional[torch.device] = None,
                      drop_last: bool = True,
                      **_) -> Optional[DataLoader]:
    """Create dataloader for sequence dataset."""
    log = logging.getLogger(__name__)

    if dataset is None or len(dataset) == 0:
        log.warning("Cannot create DataLoader for empty dataset")
        return None

    tcfg = config["training"]
    bs = tcfg["batch_size"]

    log.info(f"DataLoader[{dataset.split_name}] sequence mode: bs={bs}, samples={len(dataset)}")

    # If data is GPU cached, use simple batching with no workers
    if hasattr(dataset, 'gpu_cache') and dataset.gpu_cache is not None:
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=drop_last,
        )

    # CPU fallback - use workers for loading
    workers = tcfg.get("num_workers", 0)

    kwargs = dict(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device is not None and device.type == "cuda" and workers > 0),
        drop_last=drop_last,
        persistent_workers=(workers > 0),
    )
    if workers > 0 and "prefetch_factor" in tcfg:
        kwargs["prefetch_factor"] = tcfg["prefetch_factor"]

    return DataLoader(**kwargs)