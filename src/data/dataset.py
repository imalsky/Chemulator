#!/usr/bin/env python3
"""
High-performance dataset implementation for chemical kinetics training.
OPTIMIZED VERSION for A100 GPU with full GPU caching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        self.n_species = self.shard_index["n_species"]
        self.n_globals = self.shard_index["n_globals"]
        self.samples_per_shard = self.shard_index["samples_per_shard"]
        self.prediction_mode = self.shard_index.get("prediction_mode", "absolute")
        self.n_features = self.n_species * 2 + self.n_globals + 1
        self.n_inputs = self.n_species + self.n_globals + 1
        
        # Get shard info for this split
        self.shards = self.split_info["shards"]
        self.n_shards = len(self.shards)
        self.n_total_samples = self.split_info["total_samples"]
        
        # Build lookup arrays
        self._shard_starts = np.array([s["start_idx"] for s in self.shards])
        self._shard_ends = np.array([s["end_idx"] for s in self.shards])
        
        # Memory info
        self.bytes_per_sample = self.n_features * 4  # float32
        self.total_bytes = self.n_total_samples * self.bytes_per_sample
        
        # Initialize caching
        self.gpu_cache = None
        self.cpu_fallback = False
        self._try_gpu_cache()
        
    def _try_gpu_cache(self):
        """Try to load entire dataset to GPU memory with fallback."""
        # Check if GPU caching is enabled
        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        
        if gpu_cache_setting == False:
            self.logger.info(f"GPU caching disabled for {self.split_name}")
            self.cpu_fallback = True
            return
            
        # Only attempt on CUDA devices
        if self.device.type != "cuda":
            self.logger.info(f"GPU caching not available on {self.device.type}")
            self.cpu_fallback = True
            return
            
        # Check available memory
        free_mem, total_mem = torch.cuda.mem_get_info(self.device.index)
        needed_gb = self.total_bytes / 1024**3
        free_gb = free_mem / 1024**3
        
        # Need some buffer for operations (using 85% is a safe value)
        if needed_gb > free_gb * 0.85:
            self.logger.warning(
                f"Insufficient GPU memory for {self.split_name}: "
                f"need {needed_gb:.1f}GB, have {free_gb:.1f}GB free. "
                f"Falling back to CPU loading."
            )
            self.cpu_fallback = True
            return
            
        # Try to load to GPU
        self.logger.info(f"Loading {self.split_name} dataset to GPU ({needed_gb:.1f}GB)...")
        start_time = time.time()
        
        try:
            # Pre-allocate GPU tensor
            self.gpu_cache = torch.empty(
                (self.n_total_samples, self.n_features),
                dtype=torch.float32,
                device=self.device
            )
            
            # Load each shard
            current_idx = 0
            for shard_info in self.shards:
                shard_path = self.split_dir / shard_info["filename"]
                
                # Load shard
                if self.shard_index.get("compression") == "npz":
                    with np.load(shard_path) as zf:
                        data = zf["data"].astype(np.float32, copy=False)
                else:
                    data = np.load(shard_path).astype(np.float32, copy=False)
                
                # Copy to GPU
                n_samples = data.shape[0]
                
                # =============================================================
                # FIXED: Added .pin_memory() to enable true non_blocking copy
                # =============================================================
                pinned_tensor = torch.from_numpy(data).pin_memory()
                self.gpu_cache[current_idx:current_idx + n_samples] = pinned_tensor.to(
                    self.device, non_blocking=True
                )
                current_idx += n_samples
            
            # Ensure all transfers complete
            torch.cuda.synchronize()
            
            load_time = time.time() - start_time
            throughput = needed_gb / load_time
            self.logger.info(
                f"GPU cache loaded successfully in {load_time:.1f}s ({throughput:.1f} GB/s)"
            )
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.warning(f"GPU caching failed: {e}. Using CPU fallback.")
            self.gpu_cache = None
            self.cpu_fallback = True
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _load_shard_cpu(self, shard_idx: int) -> np.ndarray:
        """Load a shard from disk (CPU fallback)."""
        # ====================================================================
        # FIXED: Added simple per-worker LRU cache to avoid re-reading files
        # ====================================================================
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
                return zf["data"].astype(np.float32, copy=False)
        else:
            return np.load(shard_path).astype(np.float32, copy=False)

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
            
            # Return CPU tensors. The batch will be moved to the GPU
            # all at once by the dataloader / training loop.
            input_tensor = torch.from_numpy(row[:self.n_inputs].copy())
            target_tensor = torch.from_numpy(row[self.n_inputs:].copy())
            
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
                      **kwargs) -> DataLoader:
    """
    Create DataLoader with proper handling of GPU cache and CPU fallback.
    
    Args:
        dataset: The dataset to load from
        config: Configuration dictionary
        shuffle: Whether to shuffle the data
        device: PyTorch device
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    if dataset is None or len(dataset) == 0:
        logging.getLogger(__name__).warning("Cannot create DataLoader for empty dataset")
        return None

    log = logging.getLogger(__name__)
    tcfg = config["training"]
    bs = tcfg["batch_size"]
    
    # Check if using GPU cache
    is_gpu_cached = hasattr(dataset, 'gpu_cache') and dataset.gpu_cache is not None
    is_cpu_fallback = hasattr(dataset, 'cpu_fallback') and dataset.cpu_fallback
    
    if is_gpu_cached:
        # GPU mode: no workers needed
        log.info(f"DataLoader[{dataset.split_name}] GPU mode: bs={bs}")
        
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,  # MUST be 0 for GPU cache
            pin_memory=False,  # Already on GPU
            drop_last=drop_last,
        )
    elif is_cpu_fallback:
        # CPU fallback mode
        log.warning(f"DataLoader[{dataset.split_name}] CPU fallback mode: bs={bs}")
        
        # ==============================================================
        # FIXED: Correct logic for determining number of workers.
        # Use a sane default if not specified in the config.
        # ==============================================================
        num_workers_from_config = tcfg.get("num_workers")
        if num_workers_from_config is None:
            # If the key doesn't exist, default to a reasonable number.
            import os
            num_workers = min(16, os.cpu_count() or 1)
        else:
            # Use the value from the config file if it exists.
            num_workers = num_workers_from_config
        
        log.info(f"Using {num_workers} workers for CPU fallback data loading.")

        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device and device.type == "cuda"),
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        # This shouldn't happen but handle it
        raise RuntimeError(
            f"Dataset {dataset.split_name if hasattr(dataset, 'split_name') else 'unknown'} "
            "is neither GPU-cached nor in CPU fallback mode. Check dataset initialization."
        )