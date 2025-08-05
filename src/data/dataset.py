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
                 device: torch.device):
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

        # Caching
        self.gpu_cache = None
        self.cpu_fallback = False
        self._try_gpu_cache()

        # Per-worker (process) single-entry shard cache for CPU fallback
        self._shard_cache = {"name": None, "data": None}

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
        """Try to cache all data on GPU."""
        if self.n_total_samples == 0:
            self.cpu_fallback = True
            return

        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False or self.device.type != "cuda":
            self.cpu_fallback = True
            return

        bytes_per_float = 4 if self.dtype == torch.float32 else 8
        bytes_needed = self.n_total_samples * (
            self.n_species +            # x0_log
            self.n_globals +            # globals
            self.M +                    # t_vec
            self.M * self.n_target_species  # y_mat
        ) * bytes_per_float

        idx = 0 if self.device.index is None else self.device.index
        total_mem = torch.cuda.get_device_properties(idx).total_memory

        # Configurable cap on fraction of TOTAL VRAM (not "free")
        tcfg = self.config.get("training", {})
        max_frac = float(tcfg.get("gpu_cache_max_fraction", 0.5))            # default 50%
        reserve_bytes = int(tcfg.get("gpu_cache_reserved_bytes", 2_000_000_000))  # ~2 GB
        budget = max(0, int(total_mem * max_frac) - reserve_bytes)

        if bytes_needed > budget:
            self.logger.warning(
                f"GPU cache disabled: need {bytes_needed/1e9:.2f} GB > budget {budget/1e9:.2f} GB "
                f"(total {total_mem/1e9:.2f} GB, max_frac={max_frac}, reserve={reserve_bytes/1e9:.2f} GB)"
            )
            self.cpu_fallback = True
            return

        # Load all data to CPU once
        self.logger.info(f"Loading {self.split_name} sequence data to GPU ({bytes_needed/1e9:.1f} GB)...")

        all_x0_log, all_globals, all_t_vec, all_y_mat = [], [], [], []
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as z:
                t_vec_norm = self._normalize_time(z['t_vec'])
                all_x0_log.append(torch.from_numpy(z['x0_log'].astype(self.np_dtype)))
                all_globals.append(torch.from_numpy(z['globals'].astype(self.np_dtype)))
                all_t_vec.append(torch.from_numpy(t_vec_norm.astype(self.np_dtype)))
                all_y_mat.append(torch.from_numpy(z['y_mat'].astype(self.np_dtype)))

        # Concatenate and move to GPU (dtype+device in one step)
        x0_log = torch.cat(all_x0_log, dim=0).to(device=self.device, dtype=self.dtype)     # [N, S]
        globals_ = torch.cat(all_globals, dim=0).to(device=self.device, dtype=self.dtype)  # [N, G]
        t_vec = torch.cat(all_t_vec, dim=0).to(device=self.device, dtype=self.dtype)       # [N, M]
        y_mat = torch.cat(all_y_mat, dim=0).to(device=self.device, dtype=self.dtype)       # [N, M, T]

        # Prebuild inputs for fast indexing on device
        inputs = torch.cat([x0_log, globals_, t_vec], dim=1)                                # [N, S+G+M]

        self.gpu_cache = {"inputs": inputs, "targets": y_mat}
        self.logger.info("GPU sequence cache loaded successfully")

    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single trajectory from disk with per-worker shard caching."""
        # Find which shard
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]

        shard_info = self.shards[shard_idx]
        shard_path = self.split_dir / shard_info["filename"]

        # Load and cache shard arrays (LRU-1)
        if self._shard_cache["name"] != shard_info["filename"]:
            with np.load(shard_path, allow_pickle=False) as z:
                cached = {
                    "x0_log": z["x0_log"].astype(self.np_dtype, copy=False),
                    "globals": z["globals"].astype(self.np_dtype, copy=False),
                    # Cache normalized time to avoid recomputing per sample
                    "t_vec_norm": self._normalize_time(z["t_vec"]).astype(self.np_dtype, copy=False),
                    "y_mat": z["y_mat"].astype(self.np_dtype, copy=False),
                }
            self._shard_cache = {"name": shard_info["filename"], "data": cached}

        arrs = self._shard_cache["data"]

        # Extract trajectory
        x0_log = arrs['x0_log'][local_idx]
        globals_vec = arrs['globals'][local_idx]
        t_vec = arrs['t_vec_norm'][local_idx]
        y_mat = arrs['y_mat'][local_idx]

        # Combine inputs
        inputs = np.concatenate([x0_log, globals_vec, t_vec])

        # Convert to tensors (stay on CPU; DataLoader will move to device in training step)
        inputs_tensor = torch.from_numpy(inputs).to(self.dtype)
        targets_tensor = torch.from_numpy(y_mat).to(self.dtype)

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
    if workers == 0:
        workers = min(4, os.cpu_count() or 1)

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
