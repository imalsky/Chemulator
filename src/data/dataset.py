#!/usr/bin/env python3
"""
Optimized dataset for chemical kinetics data.
Uses NPYDataset for high-performance memory-mapped data loading on pre-normalized shards.
"""

import logging
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NPYDataset(Dataset):
    """
    High-performance dataset using memory-mapped NPY shards.
    Assumes pre-normalized data for zero runtime overhead.
    """

    def __init__(
        self,
        shard_dir: Path,
        indices: List[int],
        config: Dict[str, Any],
        device: torch.device,
        split_name: Optional[str] = None,
    ):
        self.shard_dir = Path(shard_dir)
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device_type = device.type
        self.split_name = split_name

        # Load shard index
        with open(self.shard_dir / "shard_index.json") as jf:
            self.shard_index = json.load(jf)
        if self.shard_index.get("format") != "npy_shards_v1":
            raise ValueError(f"{self.shard_dir}: unsupported shard format "
                             f"{self.shard_index.get('format')}")

        # Store frequently-used constants
        self.n_species = self.shard_index["n_species"]
        self.n_globals = self.shard_index["n_globals"]
        data_cfg = self.config["data"]
        self.species_vars = data_cfg["species_variables"]
        self.global_vars = data_cfg["global_variables"]
        self.time_var = data_cfg["time_variable"]

        # Split handling - prioritize passed indices
        if indices is not None:
            self.sample_indices = np.asarray(indices, dtype=np.int64)
            self.n_total_samples = int(self.sample_indices.shape[0])
            self.logger.info(f"{split_name or 'custom'} split: {self.n_total_samples:,} samples (indices passed in)")
        else:
            split_file = (self.shard_dir /
                          self.shard_index["split_files"].get(split_name or "", ""))
            if split_file.exists():
                self.sample_indices = np.load(split_file, mmap_mode="r")
                self.n_total_samples = int(self.sample_indices.shape[0])
                self.logger.info(f"{split_name.capitalize()} split: {self.n_total_samples:,} samples (indices mmap-loaded)")
            else:
                self.sample_indices = None
                self.n_total_samples = int(self.shard_index["total_samples"])

        # Worker-local cache using instance attributes (per-process after pickling)
        self.samples_per_shard = int(self.shard_index["samples_per_shard"])
        self._current_shard_idx = -1
        self._current_shard_data = None
        self._shard_cache = {}

        self.logger.info(f"NPYDataset ready: {self.n_total_samples:,} samples, "
                         f"{self.shard_index['n_shards']} shards (pre-normalized)")
    
    def _get_shard_data(self, shard_idx: int) -> np.ndarray:
        """
        Get memory-mapped shard data with caching.
        Uses instance attributes for multiprocess compatibility.
        """
        # Check if we already have this shard loaded
        if shard_idx == self._current_shard_idx and self._current_shard_data is not None:
            return self._current_shard_data
        
        # Check cache
        if shard_idx in self._shard_cache:
            self._current_shard_idx = shard_idx
            self._current_shard_data = self._shard_cache[shard_idx]
            return self._current_shard_data
        
        # Load new shard
        shard_info = self.shard_index["shards"][shard_idx]
        shard_path = self.shard_dir / shard_info["filename"]
        
        # Memory-map the file for zero-copy access
        shard_data = np.load(shard_path, mmap_mode='r')
        
        # Cache management - keep more shards for better sequential access
        max_cache_size = 8
        self._shard_cache[shard_idx] = shard_data
        if len(self._shard_cache) > max_cache_size:
            # Remove least recently used shard
            oldest_idx = min(self._shard_cache.keys())
            if oldest_idx != shard_idx:
                del self._shard_cache[oldest_idx]
        
        self._current_shard_idx = shard_idx
        self._current_shard_data = shard_data
        
        return shard_data
    
    def _global_to_shard_idx(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global sample index to (shard_idx, local_idx).
        """
        samples_per_shard = self.shard_index["samples_per_shard"]
        shard_idx = global_idx // samples_per_shard
        local_idx = global_idx % samples_per_shard
        return shard_idx, local_idx
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with efficient memory-mapped access.
        Assumes pre-normalized data - direct tensor creation.
        """
        # Map to actual sample index if using split indices
        if self.sample_indices is not None:
            if idx >= len(self.sample_indices):
                raise IndexError(f"Index {idx} out of range for split with {len(self.sample_indices)} samples")
            global_idx = self.sample_indices[idx]
        else:
            global_idx = idx
        
        # Find which shard contains this sample
        shard_idx, local_idx = self._global_to_shard_idx(global_idx)
        
        # Get shard data (memory-mapped)
        shard_data = self._get_shard_data(shard_idx)
        
        # Extract row (this is a view, not a copy)
        row = shard_data[local_idx]
        
        # Split into input and target
        n_input = self.n_species + self.n_globals + 1
        
        # Direct tensor creation without any processing
        input_arr = row[:n_input]
        target_arr = row[n_input:]
        
        # Pin memory if target device is CUDA
        pin = self.device_type == "cuda"
        input_tensor = torch.from_numpy(input_arr).pin_memory() if pin else torch.from_numpy(input_arr)
        target_tensor = torch.from_numpy(target_arr).pin_memory() if pin else torch.from_numpy(target_arr)
        
        return input_tensor, target_tensor
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about dataset structure for logging."""
        return {
            "format": "npy_shards",
            "n_shards": self.shard_index["n_shards"],
            "samples_per_shard": self.shard_index["samples_per_shard"],
            "total_samples": self.n_total_samples,
            "split": self.split_name
        }


def worker_init_fn(worker_id: int):
    """Initialize worker with proper random seed."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    *,
    shuffle: bool = True,
    device: Optional[torch.device] = None
) -> DataLoader:
    """
    Construct a DataLoader whose defaults adapt to the host hardware.
    """
    train_cfg   = config["training"]
    device_type = (device or torch.device("cuda")).type

    from utils.hardware import optimize_dataloader_settings
    opts = optimize_dataloader_settings(
        batch_size=train_cfg["batch_size"],
        device_type=device_type,
        num_workers=train_cfg.get("num_workers"),
    )

    opts["num_workers"]   = train_cfg.get("num_workers", min(16, opts["num_workers"]))
    opts["prefetch_factor"] = None if opts["num_workers"] == 0 else min(
        4, max(2, opts["num_workers"] // 4)
    )

    if opts["num_workers"] > 0:
        import torch.multiprocessing as mp
        mp.set_start_method("forkserver", force=True)

    kwargs = dict(
        dataset=dataset,
        batch_size=opts["batch_size"],
        shuffle=shuffle,
        num_workers=opts["num_workers"],
        pin_memory=opts["pin_memory"],
        persistent_workers=opts["persistent_workers"] if opts["num_workers"] > 0 else False,
        drop_last=True,
        worker_init_fn=worker_init_fn if opts["num_workers"] > 0 else None,
    )
    if opts["prefetch_factor"] is not None:
        kwargs["prefetch_factor"] = int(opts["prefetch_factor"])

    # Guard the CUDA-only call
    if opts["pin_memory"] and device_type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except AttributeError:       # older PyTorch
            pass

    logging.getLogger(__name__).info(
        "DataLoader with %d worker(s), prefetch=%s",
        opts["num_workers"],
        kwargs.get("prefetch_factor", "N/A"),
    )
    return DataLoader(**kwargs)