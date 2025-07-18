#!/usr/bin/env python3
"""
Dataset for chemical kinetics data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import gc
import psutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import collections


class NPYDataset(Dataset):
    """High-performance dataset using memory-mapped NPY shards."""
    
    def __init__(self, shard_dir: Path, indices: np.ndarray, config: Dict[str, Any],
                 device: torch.device, split_name: Optional[str] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.config = config
        # The device parameter is no longer needed here but is kept for API consistency.
        self.device = device
        self.split_name = split_name
        self.logger = logging.getLogger(__name__)

        # Load shard index
        with open(self.shard_dir / "shard_index.json") as f:
            self.shard_index = json.load(f)

        # Extract metadata
        self.n_species = self.shard_index["n_species"]
        self.n_globals = self.shard_index["n_globals"]
        self.samples_per_shard = self.shard_index["samples_per_shard"]
        
        # Store indices
        self.sample_indices = indices
        self.n_total_samples = len(indices) if indices is not None else self.shard_index["total_samples"]

        # Dynamic cache sizing based on available memory
        self._setup_cache()
        
        # Pre-build shard lookup for efficiency
        self._build_shard_lookup()

        self.logger.info(
            f"NPYDataset initialized: {self.n_total_samples:,} samples, "
            f"cache size: {self._max_cache_size} shards"
        )
    
    def _setup_cache(self):
        """Setup cache with dynamic sizing based on available memory."""
        # Get available memory
        try:
            available_memory = psutil.virtual_memory().available
        except:
            available_memory = 4 * 1024**3  # Default to 4GB if psutil fails
        
        # Estimate shard size
        n_features = self.n_species * 2 + self.n_globals + 1
        bytes_per_sample = n_features * 4  # float32
        bytes_per_shard = self.samples_per_shard * bytes_per_sample
        
        # Use up to 25% of available memory for cache
        max_cache_memory = available_memory * 0.25
        self._max_cache_size = max(1, min(
            int(max_cache_memory / bytes_per_shard),
            self.config["training"].get("dataset_cache_shards", 4)
        ))
        
        # Use an OrderedDict for LRU behavior
        self._shard_cache = collections.OrderedDict()
    
    def _build_shard_lookup(self):
        """Pre-build lookup table for shard indices."""
        self._shard_starts = np.array([s["start_idx"] for s in self.shard_index["shards"]])
        self._shard_ends = np.array([s["end_idx"] for s in self.shard_index["shards"]])
        
    def _get_shard_data(self, shard_idx: int) -> np.ndarray:
        """Get shard data with LRU caching and memory management."""
        if shard_idx in self._shard_cache:
            # Move the accessed item to the end to mark it as most recently used
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        # Load shard data
        shard_info = self.shard_index["shards"][shard_idx]
        shard_path = self.shard_dir / shard_info["filename"]

        try:
            if self.shard_index.get("compression") == "npz":
                with np.load(shard_path) as npz_file:
                    shard_data = npz_file['data'].copy()  # Copy to ensure it's in memory
            else:
                # Use memory-mapping for efficient read-only access to .npy files
                shard_data = np.load(shard_path, mmap_mode='r')
        except Exception as e:
            self.logger.error(f"Failed to load shard {shard_idx} from {shard_path}: {e}")
            raise

        # Evict the least recently used item if the cache is full
        if len(self._shard_cache) >= self._max_cache_size:
            # popitem(last=False) removes the first item added (the LRU item)
            evicted_key, evicted_data = self._shard_cache.popitem(last=False)
            # For memory-mapped arrays, delete reference to close the file
            del evicted_data
            # Force garbage collection for large arrays
            if self._max_cache_size < 10:
                gc.collect()

        # Add the newly loaded shard to the cache
        self._shard_cache[shard_idx] = shard_data
        
        return shard_data
    
    def _find_shard_idx(self, global_idx: int) -> Tuple[int, int]:
        """Find shard index and local index using vectorized search."""
        # Vectorized search
        valid_shards = (self._shard_starts <= global_idx) & (global_idx < self._shard_ends)
        shard_indices = np.where(valid_shards)[0]
        
        if len(shard_indices) == 0:
            raise IndexError(f"Sample index {global_idx} not found in any shard.")
        
        shard_idx = shard_indices[0]
        local_idx = global_idx - self._shard_starts[shard_idx]
        
        return shard_idx, local_idx
    
    def __len__(self) -> int:
        return self.n_total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample as a CPU tensor with improved error handling."""
        # Validate index
        if idx < 0 or idx >= self.n_total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")
        
        try:
            # Step 1: Get the true global index from the split-specific indices
            global_idx = self.sample_indices[idx]

            # Step 2: Find the correct shard and local index
            shard_idx, local_idx = self._find_shard_idx(global_idx)
            
            # Step 3: Get the shard data using the LRU cache
            shard_data = self._get_shard_data(shard_idx)

            # Validate local index
            if local_idx >= shard_data.shape[0]:
                raise IndexError(
                    f"Local index {local_idx} out of bounds for shard {shard_idx} "
                    f"with size {shard_data.shape[0]}"
                )
            
            row = shard_data[local_idx]
            
            # Step 4: Extract input and target and convert to tensors
            n_input = self.n_species + self.n_globals + 1
            input_arr = row[:n_input]
            target_arr = row[n_input:]
            
            # .copy() is important to avoid PyTorch errors with read-only memory-mapped arrays
            input_tensor = torch.from_numpy(input_arr.copy())
            target_tensor = torch.from_numpy(target_arr.copy())
            
            return input_tensor, target_tensor
            
        except Exception as e:
            self.logger.error(f"Error accessing sample {idx} (global {global_idx}): {e}")
            raise


def create_dataloader(dataset: Dataset, config: Dict[str, Any], shuffle: bool = True,
                     device: Optional[torch.device] = None, drop_last: bool = True) -> DataLoader:
    """Create an optimized standard DataLoader."""
    if dataset is None or len(dataset) == 0:
        return None
        
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    
    num_workers = train_cfg.get("num_workers", 0)
    
    # Adjust workers based on dataset size
    if len(dataset) < batch_size * 10:
        num_workers = min(2, num_workers)  # Reduce workers for small datasets
    
    # Use pin_memory for faster CPU-to-GPU transfers, handled by the trainer.
    pin_memory = train_cfg.get("pin_memory", False) and device and device.type == "cuda"
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None
    )