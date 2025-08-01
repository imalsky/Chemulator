#!/usr/bin/env python3
"""
High-performance dataset implementation for chemical kinetics training.
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


class NPYDataset(Dataset):
    """
    PyTorch Dataset with GPU caching and CPU fallback.
    """
    def __init__(self, shard_dir: Path, split_name: str, config: Dict[str, Any], 
                 device: torch.device):
        """Initialize dataset for a specific split."""
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

        # Verify split directory exists
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Load shard index metadata
        shard_index_path = self.shard_dir / "shard_index.json"
        with open(shard_index_path) as f:
            full_index = json.load(f)
        
        self.shard_index = full_index
        self.split_info = full_index["splits"][split_name]
        
        # Extract dimensions
        self.n_input_species = self.shard_index["n_input_species"]
        self.n_target_species = self.shard_index["n_target_species"]
        self.n_species = self.n_input_species 
        self.n_globals = self.shard_index["n_globals"]
        self.samples_per_shard = self.shard_index["samples_per_shard"]
        self.prediction_mode = self.shard_index.get("prediction_mode", "absolute")
        self.n_features = self.n_input_species + self.n_globals + 1 + self.n_target_species
        self.n_inputs = self.n_input_species + self.n_globals + 1
        
        # Get shard info for this split
        self.shards = self.split_info["shards"]
        self.n_shards = len(self.shards)
        self.n_total_samples = self.split_info["total_samples"]
        
        # Build lookup arrays
        self._shard_starts = np.array([s["start_idx"] for s in self.shards])
        self._shard_ends = np.array([s["end_idx"] for s in self.shards])
        
        # Memory info - use configured dtype size
        bytes_per_element = 4 if dtype_str == "float32" else 8
        self.bytes_per_sample = self.n_features * bytes_per_element
        self.total_bytes = self.n_total_samples * self.bytes_per_sample
        
        # Initialize caching
        self.gpu_cache = None
        self.cpu_fallback = False
        self._try_gpu_cache()
        
    def _try_gpu_cache(self):
        """Load the entire split to GPU memory; fall back to CPU if needed."""
        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False:
            self.logger.info(f"GPU caching disabled for {self.split_name}")
            self.cpu_fallback = True
            return

        if self.device.type != "cuda":
            self.logger.info(f"GPU caching not available on {self.device.type}")
            self.cpu_fallback = True
            return

        free_mem, _ = torch.cuda.mem_get_info(self.device.index)
        needed_gb = self.total_bytes / 1024 ** 3
        free_gb   = free_mem       / 1024 ** 3
        if needed_gb > free_gb * 0.85:     
            self.logger.warning(
                f"Insufficient GPU memory for {self.split_name}: "
                f"need {needed_gb:.1f} GB, have {free_gb:.1f} GB.  Falling back to CPU."
            )
            self.cpu_fallback = True
            return

        self.logger.info(f"Loading {self.split_name} dataset to GPU ({needed_gb:.1f} GB)…")
        start = time.time()

        self.gpu_cache = torch.empty(
            (self.n_total_samples, self.n_features),
            dtype=self.dtype,
            device=self.device,
        )

        cur = 0
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            if self.shard_index.get("compression") == "npz":
                with np.load(shard_path) as zf:
                    data = zf["data"].astype(self.np_dtype, copy=False)
            else:
                data = np.load(shard_path).astype(self.np_dtype, copy=False)

            n = data.shape[0]
            self.gpu_cache[cur:cur + n] = torch.from_numpy(data).pin_memory().to(self.device, non_blocking=True)
            cur += n
            del data                  

        torch.cuda.synchronize()
        t = time.time() - start
        self.logger.info(f"GPU cache loaded in {t:.1f}s ({needed_gb/t:.1f} GB/s)")

    def _load_shard_cpu(self, shard_idx: int) -> np.ndarray:
        """Load a shard from disk (CPU fallback)."""
        
        # Check if the cache exists on this worker process
        if not hasattr(self, '_shard_cache'):
            # functools.lru_cache is a simpler way to do this
            from functools import lru_cache
            # Cache up to 2 shards per worker, a common scenario
            self._shard_cache = lru_cache(maxsize=2)(self._load_shard_from_disk)
        
        return self._shard_cache(shard_idx)

    def _load_shard_from_disk(self, shard_idx: int) -> np.ndarray:
        """Helper for the LRU cache that actually hits the disk."""
        shard_info = self.shards[shard_idx]
        shard_path = self.split_dir / shard_info["filename"]
        
        if self.shard_index.get("compression") == "npz":
            with np.load(shard_path) as zf:
                return zf["data"].astype(self.np_dtype, copy=False)
        else:
            return np.load(shard_path).astype(self.np_dtype, copy=False)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index."""
        if self.gpu_cache is not None:
            # Fast path: Direct GPU slicing, no change needed here.
            row = self.gpu_cache[idx]
            return row[:self.n_inputs], row[self.n_inputs:]
        else:
            # CPU fallback path
            if not self.cpu_fallback:
                raise RuntimeError("Dataset not properly initialized")
            
            # Find which shard contains this index
            shard_idx = np.searchsorted(self._shard_starts, idx, side='right') - 1
            local_idx = idx - self._shard_starts[shard_idx]
            
            # Load shard and get sample
            shard_data = self._load_shard_cpu(shard_idx)
            row = shard_data[local_idx]
            
            # Return CPU tensors with configured dtype
            input_tensor = torch.from_numpy(row[:self.n_inputs].copy()).to(self.dtype)
            target_tensor = torch.from_numpy(row[self.n_inputs:].copy()).to(self.dtype)
            
            return input_tensor, target_tensor

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics with consistent structure."""
        if self.gpu_cache is not None:
            return {
                "type": "gpu",
                "size_gb": self.total_bytes / 1024**3,
                "device": str(self.device),
                "status": "active"
            }
        elif self.cpu_fallback:
            return {
                "type": "cpu",
                "size_gb": 0,
                "device": str(self.device),
                "status": "fallback",
                "message": "Using CPU loading due to insufficient GPU memory"
            }
        else:
            return {
                "type": "none",
                "size_gb": 0,
                "device": str(self.device),
                "status": "error",
                "message": "Dataset caching failed"
            }

def create_dataloader(dataset: Dataset,
                      config: Dict[str, Any],
                      shuffle: bool = True,
                      device: Optional[torch.device] = None,
                      drop_last: bool = True,
                      **_) -> DataLoader:
    """
    Build a DataLoader
    """
    if dataset is None or len(dataset) == 0:
        logging.getLogger(__name__).warning("Cannot create DataLoader for empty dataset")
        return None

    log  = logging.getLogger(__name__)
    tcfg = config["training"]
    bs   = tcfg["batch_size"]

    is_gpu_cached   = getattr(dataset, "gpu_cache", None) is not None

    if is_gpu_cached:
        log.info(f"DataLoader[{dataset.split_name}] GPU‑batch mode: bs={bs}")

        class GPUBatchDataset(torch.utils.data.Dataset):
            def __init__(self, gpu_tensor: torch.Tensor, n_inputs: int,
                         batch_size: int, shuffle_batches: bool):
                self.gpu_tensor  = gpu_tensor
                self.n_inputs    = n_inputs
                self.batch_size  = batch_size
                self.shuffle     = shuffle_batches
                self.total       = gpu_tensor.size(0)
                self.n_batches   = math.ceil(self.total / batch_size)
                self.permutation = None

            def __len__(self):
                return self.n_batches

            def __getitem__(self, batch_idx: int):
                if self.shuffle:
                    if self.permutation is None or batch_idx == 0:
                        self.permutation = torch.randperm(
                            self.total, device=self.gpu_tensor.device
                        )
                    idx = self.permutation
                else:
                    idx = None

                start = batch_idx * self.batch_size
                end   = min(start + self.batch_size, self.total)

                if idx is None:
                    batch = self.gpu_tensor[start:end]
                else:
                    batch = self.gpu_tensor.index_select(0, idx[start:end])

                return batch[:, :self.n_inputs], batch[:, self.n_inputs:]

        gpu_ds = GPUBatchDataset(
            dataset.gpu_cache, dataset.n_inputs, bs, shuffle
        )
        return DataLoader(
            gpu_ds,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    # Cpu fallback
    log.warning(f"DataLoader[{dataset.split_name}] CPU fallback mode: bs={bs}")
    workers = tcfg.get("num_workers") or min(16, (os.cpu_count() or 1))
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device and device.type == "cuda"),
        drop_last=drop_last,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
    )