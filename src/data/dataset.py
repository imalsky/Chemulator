#!/usr/bin/env python3
"""
High-performance dataset implementation with sequence mode support for LiLaN.

Improvements over baseline:
- CPU fallback now caches the currently-used shard in RAM (LRU-1) to avoid
  reopening and re-decompressing the .npz on every sample.
- GPU cache prebuilds `inputs=[x0_log, globals, t_vec]` and `targets=y_mat`
  once, removing per-item device concatenations in the hot path.
- Shard start indices stored as int64 to keep searchsorted robust.
- Safer np.load with allow_pickle=False; fewer redundant copies when moving to device.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math


class SequenceDataset(Dataset):
    """Dataset for sequence mode (trajectory-based) data."""
    def __init__(self, shard_dir: Path, split_name: str, config: Dict[str, Any], 
                device: torch.device, norm_stats: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Get dtype
        dtype_str = self.config["system"].get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)
        self.np_dtype = np.float32 if dtype_str == "float32" else np.float64

        # Normalization stats (for standardizing x0_log & globals)
        self.norm_stats = norm_stats or {}
        self._use_input_norm = False  # set true by helper if stats present
        
        # Load metadata
        shard_index_path = self.shard_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
        
        with open(shard_index_path) as f:
            self.shard_index = json.load(f)
        
        # Verify this is sequence mode data
        if not self.shard_index.get("sequence_mode", False):
            raise ValueError("Expected sequence mode data for SequenceDataset")
        
        self.split_info = self.shard_index["splits"][split_name]
        self.M = self.shard_index["M_per_sample"]
        self.n_species = self.shard_index["n_input_species"]
        self.n_target_species = self.shard_index["n_target_species"]
        self.n_globals = self.shard_index["n_globals"]
        
        # Time normalization parameters
        self.time_norm = self.shard_index["time_normalization"]
        self.tau0 = self.time_norm["tau0"]
        self.tmin = self.time_norm["tmin"]
        self.tmax = self.time_norm["tmax"]

        self.logger.info(
            f"Time normalization (global): tau0={self.tau0:.6g}, tmin={self.tmin:.6g}, tmax={self.tmax:.6g}, "
            f"M={self.M}, n_species={self.n_species}, n_targets={self.n_target_species}, n_globals={self.n_globals}"
        )
        
        # Build shard info
        self.shards = self.split_info["shards"]
        self.n_shards = len(self.shards)
        self.n_total_samples = self.split_info["n_trajectories"]
        
        if self.n_total_samples == 0:
            self.logger.warning(f"No samples in {split_name} split")
        
        # Build lookup for shard boundaries
        self._build_shard_lookup()

        # Precompute input normalization tensors (species: log stats; globals: linear stats)
        self._prepare_input_norm_tensors()
        
        # Caching
        self.gpu_cache = None
        self.cpu_fallback = False
        self._try_gpu_cache()


        # Per-worker (process) single-entry shard cache for CPU fallback
        self._shard_cache = {"name": None, "data": None}


    def _prepare_input_norm_tensors(self) -> None:
        """
        Precompute per-variable mean/std for input standardization.
        Species: use log stats (log_mean/log_std); Globals: linear (mean/std).
        """
        if not self.norm_stats:
            self._use_input_norm = False
            return

        pks = self.norm_stats.get("per_key_stats", {})
        species = self.config["data"]["species_variables"]
        globals_ = self.config["data"]["global_variables"]

        eps = float(self.norm_stats.get("min_std", 1e-10))

        s_means, s_stds = [], []
        for v in species:
            st = pks.get(v, {})
            s_means.append(float(st.get("log_mean", 0.0)))
            s_stds.append(max(float(st.get("log_std", 1.0)), eps))

        g_means, g_stds = [], []
        for v in globals_:
            st = pks.get(v, {})
            g_means.append(float(st.get("mean", 0.0)))
            g_stds.append(max(float(st.get("std", 1.0)), eps))

        # Numpy for CPU path
        self._s_mean_np = np.asarray(s_means, dtype=self.np_dtype)
        self._s_std_np  = np.asarray(s_stds,  dtype=self.np_dtype)
        self._g_mean_np = np.asarray(g_means, dtype=self.np_dtype)
        self._g_std_np  = np.asarray(g_stds,  dtype=self.np_dtype)

        # Torch tensors for CPU normalization (used in _try_gpu_cache)
        self._s_mean_cpu = torch.tensor(s_means, dtype=self.dtype, device="cpu")
        self._s_std_cpu  = torch.tensor(s_stds,  dtype=self.dtype, device="cpu")
        self._g_mean_cpu = torch.tensor(g_means, dtype=self.dtype, device="cpu")
        self._g_std_cpu  = torch.tensor(g_stds,  dtype=self.dtype, device="cpu")

        # Torch tensors on target device (kept for potential other use)
        self._s_mean_t = torch.tensor(s_means, dtype=self.dtype, device=self.device)
        self._s_std_t  = torch.tensor(s_stds,  dtype=self.dtype, device=self.device)
        self._g_mean_t = torch.tensor(g_means, dtype=self.dtype, device=self.device)
        self._g_std_t  = torch.tensor(g_stds,  dtype=self.dtype, device=self.device)

        self._use_input_norm = True

    def _build_shard_lookup(self):
        """Build lookup arrays for shard access."""
        self.shard_starts = [0]
        cumsum = 0
        for shard in self.shards:
            cumsum += shard["n_samples"]
            self.shard_starts.append(cumsum)
        # Start indices for each shard; int64 for robust searchsorted math
        self.shard_starts = np.array(self.shard_starts[:-1], dtype=np.int64)

    def _normalize_time(self, t: np.ndarray) -> np.ndarray:
        """Apply global log-min-max time normalization."""
        # Apply log transform with tau0
        tau = np.log(1 + t / self.tau0)
        # Normalize to [0, 1]
        return np.clip((tau - self.tmin) / max(self.tmax - self.tmin, 1e-10), 0, 1)

    def _try_gpu_cache(self):
        """Try to cache all data on GPU and standardize x0_log & globals if stats are available."""
        if self.n_total_samples == 0:
            self.cpu_fallback = True
            return

        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False or self.device.type != "cuda":
            self.cpu_fallback = True
            return

        bytes_per_float = 4 if self.dtype == torch.float32 else 8
        bytes_needed = self.n_total_samples * (
            self.n_species +                 # x0_log
            self.n_globals +                 # globals
            self.M +                         # t_vec
            self.M * self.n_target_species   # y_mat
        ) * bytes_per_float

        idx = 0 if self.device.index is None else self.device.index
        total_mem = torch.cuda.get_device_properties(idx).total_memory

        tcfg = self.config.get("training", {})
        max_frac = float(tcfg.get("gpu_cache_max_fraction", 0.5))
        reserve_bytes = int(tcfg.get("gpu_cache_reserved_bytes", 2_000_000_000))
        budget = max(0, int(total_mem * max_frac) - reserve_bytes)

        if bytes_needed > budget:
            self.logger.warning(
                f"GPU cache disabled: need {bytes_needed/1e9:.2f} GB > budget {budget/1e9:.2f} GB "
                f"(total {total_mem/1e9:.2f} GB, max_frac={max_frac}, reserve={reserve_bytes/1e9:.2f} GB)"
            )
            self.cpu_fallback = True
            return

        # Load all data on CPU first
        self.logger.info(f"Loading {self.split_name} sequence data to GPU ({bytes_needed/1e9:.1f} GB)...")

        all_x0_log, all_globals, all_t_vec, all_y_mat = [], [], [], []

        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

            # Use allow_pickle=False for safety
            with np.load(shard_path, allow_pickle=False) as data:
                # Normalize time -> [0,1] (numpy)
                t_vec_norm = self._normalize_time(data["t_vec"])

                # To torch (CPU, correct dtype)
                x0_log = torch.from_numpy(data["x0_log"].astype(self.np_dtype, copy=False)).to(dtype=self.dtype, device="cpu")
                g_vec  = torch.from_numpy(data["globals"].astype(self.np_dtype, copy=False)).to(dtype=self.dtype, device="cpu")
                t_vec  = torch.from_numpy(t_vec_norm.astype(self.np_dtype, copy=False)).to(dtype=self.dtype, device="cpu")
                y_mat  = torch.from_numpy(data["y_mat"].astype(self.np_dtype, copy=False)).to(dtype=self.dtype, device="cpu")

                # --- Standardize inputs on CPU if we have stats ---
                if self._use_input_norm:
                    x0_log = (x0_log - self._s_mean_cpu) / self._s_std_cpu
                    g_vec  = (g_vec  - self._g_mean_cpu) / self._g_std_cpu

                all_x0_log.append(x0_log)
                all_globals.append(g_vec)
                all_t_vec.append(t_vec)
                all_y_mat.append(y_mat)

        # Concatenate on CPU
        x0_log_full = torch.cat(all_x0_log, dim=0)
        globals_full = torch.cat(all_globals, dim=0)
        t_vec_full = torch.cat(all_t_vec, dim=0)
        y_mat_full = torch.cat(all_y_mat, dim=0)

        # Pre-build inputs (CPU)
        inputs_full = torch.cat([x0_log_full, globals_full, t_vec_full], dim=1)

        # Move once to GPU
        self.gpu_cache = {
            "inputs": inputs_full.to(self.device, non_blocking=True),
            "targets": y_mat_full.to(self.device, non_blocking=True),
        }

        self.logger.info("GPU sequence cache loaded successfully")


    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single trajectory from disk and standardize x0_log & globals if stats available."""
        # Which shard?
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]

        shard_path = self.split_dir / self.shards[shard_idx]["filename"]
        shard_name = shard_path.name

        # LRU-1 cache: keep arrays of the last shard in RAM
        cache = self._shard_cache
        if cache["name"] != shard_name or cache["data"] is None:
            with np.load(shard_path, allow_pickle=False) as npz:
                shard_data = {
                    "x0_log": npz["x0_log"].astype(self.np_dtype, copy=False),
                    "globals": npz["globals"].astype(self.np_dtype, copy=False),
                    "t_vec":   npz["t_vec"].astype(self.np_dtype, copy=False),
                    "y_mat":   npz["y_mat"].astype(self.np_dtype, copy=False),
                }
            self._shard_cache = {"name": shard_name, "data": shard_data}
            cache = self._shard_cache

        data = cache["data"]

        # Extract trajectory
        x0_log = data['x0_log'][local_idx]       # [S], np
        globals_vec = data['globals'][local_idx] # [G], np
        t_vec_raw = data['t_vec'][local_idx]     # [M], np
        y_mat = data['y_mat'][local_idx]         # [M, T], np

        # Normalize time -> [0,1]
        t_vec = self._normalize_time(t_vec_raw)

        # Standardize (numpy path)
        if self._use_input_norm:
            x0_log = (x0_log - self._s_mean_np) / self._s_std_np
            globals_vec = (globals_vec - self._g_mean_np) / self._g_std_np

        # Build inputs
        inputs = np.concatenate([x0_log, globals_vec, t_vec])  # [S+G+M]

        # To tensors (CPU; DataLoader can pin/move later)
        inputs_tensor = torch.from_numpy(inputs).to(self.dtype)
        targets_tensor = torch.from_numpy(y_mat.astype(self.np_dtype, copy=False)).to(self.dtype)
        return inputs_tensor, targets_tensor


    def __len__(self) -> int:
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a trajectory sample."""
        if idx >= self.n_total_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_total_samples} samples")

        if self.gpu_cache is not None:
            # Fast GPU path: direct indexing
            return self.gpu_cache["inputs"][idx], self.gpu_cache["targets"][idx]
        else:
            # CPU fallback
            return self._load_from_disk(idx)

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
            num_workers=0,       # Data already on GPU
            pin_memory=False,
            drop_last=drop_last,
        )

    # CPU fallback - use workers for loading
    workers = tcfg.get("num_workers", 0)

    # Build kwargs to avoid passing prefetch_factor when workers==0
    kwargs = dict(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device is not None and device.type == "cuda"),
        drop_last=drop_last,
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)
