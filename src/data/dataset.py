#!/usr/bin/env python3
"""
Dataset for chemical kinetics data with simplified caching.
Fixed issues:
1. Enforce float32 dtype to match model weights
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import psutil


class NPYDataset(Dataset):
    """High-performance dataset using NPY shards with LRU caching."""
    
    def __init__(self, shard_dir: Path, indices: np.ndarray, config: Dict[str, Any],
                 device: torch.device, split_name: Optional[str] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.config = config
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
        self.prediction_mode = self.shard_index.get("prediction_mode", "absolute")
        
        # Store indices
        self.sample_indices = indices
        self.n_total_samples = len(indices) if indices is not None else self.shard_index["total_samples"]

        # Dynamic cache sizing based on available memory
        self._setup_cache()
        
        # Pre-build shard lookup for efficiency
        self._build_shard_lookup()

        self.logger.info(
            f"NPYDataset initialized: {self.n_total_samples:,} samples, "
            f"cache size: {self._max_cache_size} shards, "
            f"prediction mode: {self.prediction_mode}"
        )
    
    def _setup_cache(self):
        """Setup cache with dynamic sizing based on available memory."""
        # Get available memory
        try:
            available_memory = psutil.virtual_memory().available
        except:
            available_memory = 4 * 1024**3
        
        # Estimate shard size
        n_features = self.n_species * 2 + self.n_globals + 1
        bytes_per_sample = n_features * 4  # float32
        bytes_per_shard = self.samples_per_shard * bytes_per_sample
        
        # Use up to 25% of available memory for cache
        max_cache_memory = available_memory * 0.25
        self._max_cache_size = max(1, min(
            int(max_cache_memory / bytes_per_shard),
            self.config["training"].get("dataset_cache_shards", 256)
        ))
        
        # Set up LRU cache for shard loading
        self._get_shard_data = lru_cache(maxsize=self._max_cache_size)(self._get_shard_data_impl)
    
    def _build_shard_lookup(self):
        """Pre-build lookup table for shard indices with binary search support."""
        self._shard_starts = np.array([s["start_idx"] for s in self.shard_index["shards"]])
        self._shard_ends = np.array([s["end_idx"] for s in self.shard_index["shards"]])
        
        # Verify shards are contiguous and sorted
        assert np.all(self._shard_starts[1:] == self._shard_ends[:-1]), "Shards must be contiguous"
        assert np.all(self._shard_starts[:-1] < self._shard_starts[1:]), "Shards must be sorted"
    
    def _get_shard_data_impl(self, shard_idx: int) -> np.ndarray:
        """Load shard data from disk."""
        shard_info = self.shard_index["shards"][shard_idx]
        shard_path = self.shard_dir / shard_info["filename"]

        try:
            if self.shard_index.get("compression") == "npz":
                with np.load(shard_path) as npz_file:
                    return npz_file['data'].copy()  # Copy to ensure it's in memory
            else:
                return np.load(shard_path)
        except Exception as e:
            self.logger.error(f"Failed to load shard {shard_idx} from {shard_path}: {e}")
            raise
    
    def _find_shard_idx(self, global_idx: int) -> Tuple[int, int]:
        """Find shard index and local index using binary search."""
        # Use binary search instead of linear scan
        shard_idx = np.searchsorted(self._shard_starts, global_idx, side='right') - 1
        
        # Validate the result
        if shard_idx < 0 or shard_idx >= len(self._shard_starts):
            raise IndexError(f"Sample index {global_idx} not found in any shard.")
        
        local_idx = global_idx - self._shard_starts[shard_idx]
        
        # Double-check the bounds
        if not (0 <= local_idx < (self._shard_ends[shard_idx] - self._shard_starts[shard_idx])):
            raise IndexError(f"Sample index {global_idx} not in shard {shard_idx} bounds.")
        
        return shard_idx, local_idx
    
    def __len__(self) -> int:
        return self.n_total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample as a CPU tensor."""
        # Validate index
        if idx < 0 or idx >= self.n_total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")
        
        try:
            # Get the true global index from the split-specific indices
            global_idx = self.sample_indices[idx]

            # Find the correct shard and local index
            shard_idx, local_idx = self._find_shard_idx(global_idx)
            
            # Get the shard data using the LRU cache
            shard_data = self._get_shard_data(shard_idx)

            # Validate local index
            if local_idx >= shard_data.shape[0]:
                raise IndexError(
                    f"Local index {local_idx} out of bounds for shard {shard_idx} "
                    f"with size {shard_data.shape[0]}"
                )
            
            row = shard_data[local_idx]
            
            # Extract input and target
            n_input = self.n_species + self.n_globals + 1
            input_arr = row[:n_input]
            target_arr = row[n_input:]
            
            # CORRECTED: Enforce float32 dtype to match model weights
            input_tensor = torch.from_numpy(input_arr.copy()).to(dtype=torch.float32)
            target_tensor = torch.from_numpy(target_arr.copy()).to(dtype=torch.float32)
            
            return input_tensor, target_tensor
            
        except Exception as e:
            self.logger.error(f"Error accessing sample {idx} (global {global_idx if 'global_idx' in locals() else 'unknown'}): {e}")
            raise


def create_dataloader(dataset: Dataset, config: Dict[str, Any], shuffle: bool = True,
                     device: Optional[torch.device] = None, drop_last: bool = True) -> DataLoader:
    """Create an optimized DataLoader."""
    if dataset is None or len(dataset) == 0:
        return None
        
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    
    num_workers = train_cfg.get("num_workers", 0)
    
    # Adjust workers based on dataset size
    if len(dataset) < batch_size * 10:
        num_workers = min(2, num_workers)  # Reduce workers for small datasets
    
    # Use pin_memory for faster CPU-to-GPU transfers
    pin_memory = train_cfg.get("pin_memory", False) and device and device.type == "cuda"
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=drop_last,
        prefetch_factor=train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    )