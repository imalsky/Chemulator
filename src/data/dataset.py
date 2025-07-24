#!/usr/bin/env python3
"""
Dataset and DataLoader implementation to resolve memory accumulation issues.
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
    """High-performance dataset using NPY shards with memory-aware LRU caching."""

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

        # Defer cache setup to be pickle-friendly for multiprocessing
        self.cache_is_setup = False
        self._determine_cache_size()

        # Pre-build shard lookup for efficiency
        self._build_shard_lookup()

        self.logger.info(
            f"NPYDataset '{self.split_name}' initialized: {self.n_total_samples:,} samples, "
            f"target cache size: {self._max_cache_size} shards, "
            f"prediction mode: {self.prediction_mode}"
        )

    def _determine_cache_size(self):
        """Determine cache size based on available memory and config - VERSION."""
        try:
            available_memory = psutil.virtual_memory().available
        except Exception:
            available_memory = 4 * 1024**3  # Fallback to 4GB

        # Estimate shard size
        n_features = self.n_species * 2 + self.n_globals + 1
        bytes_per_sample = n_features * 4  # float32
        bytes_per_shard = self.samples_per_shard * bytes_per_sample

        max_cache_memory = available_memory * 0.25
        
        # Get num_workers from config
        num_workers = self.config["training"].get("num_workers", 1)
        
        if num_workers > 0:
            # Each worker gets a smaller share to prevent memory explosion
            max_cache_memory = max_cache_memory / max(4, num_workers)
            self.logger.debug(f"Adjusting cache size for {num_workers} workers")
        
        # For large datasets (50M samples per shard), limit to 2-4 shards max
        dataset_size_gb = (self.n_total_samples * bytes_per_sample) / (1024**3)
        if dataset_size_gb > 10:  # Large dataset
            max_shards = 2
        elif dataset_size_gb > 5:  # Medium dataset
            max_shards = 4
        else:  # Small dataset
            max_shards = 8
        
        self._max_cache_size = max(1, min(
            int(max_cache_memory / max(1, bytes_per_shard)),
            self.config["training"].get("dataset_cache_shards", 256),
            max_shards  # Apply our conservative limit
        ))

    def _setup_worker_cache(self):
        """Initializes the LRU cache within a worker process."""
        if not self.cache_is_setup:
            self._get_shard_data = lru_cache(maxsize=self._max_cache_size)(self._get_shard_data_impl)
            self.cache_is_setup = True

    def _build_shard_lookup(self):
        """Pre-build lookup table for shard indices with binary search support."""
        self._shard_starts = np.array([s["start_idx"] for s in self.shard_index["shards"]])
        self._shard_ends = np.array([s["end_idx"] for s in self.shard_index["shards"]])

        # Verify shards are contiguous and sorted
        if len(self._shard_starts) > 1:
            assert np.all(self._shard_starts[1:] == self._shard_ends[:-1]), "Shards must be contiguous"
            assert np.all(self._shard_starts[:-1] < self._shard_starts[1:]), "Shards must be sorted"

    def _get_shard_data_impl(self, shard_idx: int) -> np.ndarray:
        """Load shard data from disk. This method is what gets cached."""
        shard_info = self.shard_index["shards"][shard_idx]
        shard_path = self.shard_dir / shard_info["filename"]

        try:
            if self.shard_index.get("compression") == "npz":
                with np.load(shard_path) as npz_file:
                    return npz_file['data'].copy()  # Copy to ensure it's in memory
            else:
                return np.load(shard_path, mmap_mode='r')
        except Exception as e:
            self.logger.error(f"Failed to load shard {shard_idx} from {shard_path}: {e}")
            raise

    def _find_shard_idx(self, global_idx: int) -> Tuple[int, int]:
        """Find shard index and local index using binary search."""
        shard_idx = np.searchsorted(self._shard_starts, global_idx, side='right') - 1

        if not (0 <= shard_idx < len(self._shard_starts)):
            raise IndexError(f"Sample index {global_idx} not found in any shard.")

        local_idx = global_idx - self._shard_starts[shard_idx]

        if not (0 <= local_idx < (self._shard_ends[shard_idx] - self._shard_starts[shard_idx])):
            raise IndexError(f"Sample index {global_idx} not in shard {shard_idx} bounds.")

        return shard_idx, local_idx

    def __len__(self) -> int:
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample, with lazy cache initialization for workers."""
        # Lazily initialize the cache on first access within a worker process
        if not self.cache_is_setup:
            self._setup_worker_cache()

        if not (0 <= idx < self.n_total_samples):
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")

        global_idx = -1  # For better error reporting
        try:
            # Get the true global index from the split-specific indices
            global_idx = self.sample_indices[idx]

            # Find the correct shard and local index
            shard_idx, local_idx = self._find_shard_idx(global_idx)

            # Get the shard data using the now-initialized LRU cache
            shard_data = self._get_shard_data(shard_idx)

            if not (0 <= local_idx < shard_data.shape[0]):
                raise IndexError(
                    f"Local index {local_idx} out of bounds for shard {shard_idx} "
                    f"with size {shard_data.shape[0]}"
                )

            row = shard_data[local_idx]

            # Extract input and target
            n_input = self.n_species + self.n_globals + 1
            input_arr = row[:n_input]
            target_arr = row[n_input:]

            if isinstance(input_arr, np.memmap):
                input_tensor = torch.from_numpy(input_arr.copy()).to(dtype=torch.float32)
                target_tensor = torch.from_numpy(target_arr.copy()).to(dtype=torch.float32)
            else:
                input_tensor = torch.from_numpy(input_arr.copy()).to(dtype=torch.float32)
                target_tensor = torch.from_numpy(target_arr.copy()).to(dtype=torch.float32)

            return input_tensor, target_tensor

        except Exception as e:
            self.logger.error(f"Error accessing sample idx={idx} (global_idx={global_idx}): {e}", exc_info=True)
            raise


def create_dataloader(dataset: Dataset, config: Dict[str, Any], shuffle: bool = True,
                     device: Optional[torch.device] = None, drop_last: bool = True) -> DataLoader:
    """Create an optimized DataLoader - for large datasets."""
    if dataset is None or len(dataset) == 0:
        return None

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]

    # Get dataset size estimate
    dataset_size = len(dataset) * dataset.n_species * 2 * 4 / (1024**3)  # Rough GB estimate
    
    if dataset_size > 50:  # Large dataset (>10GB)
        # Force single-threaded loading for large datasets
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
        
        logging.getLogger(__name__).info(
            f"Large dataset detected ({dataset_size:.1f}GB). Using single-threaded loading for stability."
        )
    elif dataset_size > 20:  # Medium dataset (5-10GB)
        # Use minimal workers
        num_workers = min(32, train_cfg.get("num_workers", 0))
        pin_memory = False
        persistent_workers = False
        prefetch_factor = 2 if num_workers > 0 else None
    else:  # Small dataset (<5GB)
        # Use configured settings but cap workers
        num_workers = min(16, train_cfg.get("num_workers", 0))
        pin_memory = train_cfg.get("pin_memory", False) and device and device.type == "cuda"
        persistent_workers = train_cfg.get("persistent_workers", False) and num_workers > 0
        prefetch_factor = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

    # Adjust workers based on dataset size
    if len(dataset) < batch_size * 10:
        num_workers = min(2, num_workers)  # Reduce workers for small datasets

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor
    )