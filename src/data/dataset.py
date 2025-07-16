#!/usr/bin/env python3
"""
Optimized dataset for chemical kinetics data using HDF5 format.
Uses chunked HDF5 files for efficient data loading with built-in compression.
"""

import logging
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HDF5Dataset(Dataset):
    """
    High-performance dataset using chunked HDF5 format.
    Assumes pre-normalized data for zero runtime overhead.
    """

    def __init__(
        self,
        hdf5_path: Path,
        split_name: str,  # "train", "validation", or "test"
        config: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.hdf5_path = Path(hdf5_path)
        self.split_name = split_name
        self.config = config
        self.device_type = device.type
        
        # Load metadata
        start_time = time.time()
        self.logger.info(f"Opening HDF5 file for {split_name} split...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get split group
            if split_name not in f:
                raise ValueError(f"Split '{split_name}' not found in HDF5 file")
            
            split_group = f[split_name]
            
            # Get metadata
            self.n_samples = split_group.attrs['n_samples']
            self.n_species = split_group.attrs['n_species']
            self.n_globals = split_group.attrs['n_globals']
            
            # Verify data exists
            if 'inputs' not in split_group or 'targets' not in split_group:
                raise ValueError(f"Missing 'inputs' or 'targets' dataset in {split_name} split")
        
        # Setup for efficient access
        self._file_handle = None
        self._inputs_dset = None
        self._targets_dset = None
        
        # Cache configuration
        self.cache_size = 32  # Number of chunks to cache
        self._chunk_cache = {}
        self._cache_order = []
        
        load_time = time.time() - start_time
        self.logger.info(
            f"HDF5Dataset ready: {split_name} split with {self.n_samples:,} samples "
            f"(loaded metadata in {load_time:.2f}s)"
        )
    
    def _ensure_file_open(self):
        """Ensure HDF5 file is open with proper error handling."""
        if self._file_handle is None:
            try:
                # Use SWMR mode for concurrent access safety
                self._file_handle = h5py.File(self.hdf5_path, 'r', swmr=True)
                
                # Verify split exists
                if self.split_name not in self._file_handle:
                    raise ValueError(f"Split '{self.split_name}' not found in HDF5 file")
                    
                split_group = self._file_handle[self.split_name]
                
                # Verify datasets exist
                if 'inputs' not in split_group:
                    raise ValueError(f"'inputs' dataset missing in {self.split_name} split")
                if 'targets' not in split_group:
                    raise ValueError(f"'targets' dataset missing in {self.split_name} split")
                    
                self._inputs_dset = split_group['inputs']
                self._targets_dset = split_group['targets']
                
                # Verify dataset shapes
                if len(self._inputs_dset) != len(self._targets_dset):
                    raise ValueError(
                        f"Mismatched dataset sizes: inputs={len(self._inputs_dset)}, "
                        f"targets={len(self._targets_dset)}"
                    )
                
                # Log chunk info for debugging
                self.logger.debug(
                    f"HDF5 chunks - inputs: {self._inputs_dset.chunks}, "
                    f"targets: {self._targets_dset.chunks}"
                )
                
            except OSError as e:
                self.logger.error(f"Failed to open HDF5 file: {self.hdf5_path}")
                self.logger.error(f"Error: {e}")
                raise RuntimeError(f"Cannot open HDF5 file: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error opening HDF5 file: {e}")
                raise
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with efficient HDF5 access.
        Optimized to avoid unnecessary copies for A100 GPU performance.
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for split with {self.n_samples} samples")
        
        # Ensure file is open
        self._ensure_file_open()
        
        # Get chunk index and local index
        chunk_size = self._inputs_dset.chunks[0] if self._inputs_dset.chunks else 4096
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        
        # Try to get from cache
        if chunk_idx in self._chunk_cache:
            inputs_chunk, targets_chunk = self._chunk_cache[chunk_idx]
            # Update cache order (LRU)
            self._cache_order.remove(chunk_idx)
            self._cache_order.append(chunk_idx)
        else:
            # Load chunk
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, self.n_samples)
            
            # Read full chunk (more efficient than single row)
            inputs_chunk = self._inputs_dset[chunk_start:chunk_end]
            targets_chunk = self._targets_dset[chunk_start:chunk_end]
            
            # Add to cache
            self._chunk_cache[chunk_idx] = (inputs_chunk, targets_chunk)
            self._cache_order.append(chunk_idx)
            
            # Evict oldest chunk if cache is full
            if len(self._chunk_cache) > self.cache_size:
                oldest_chunk = self._cache_order.pop(0)
                del self._chunk_cache[oldest_chunk]
        
        # Extract the specific sample
        input_arr = inputs_chunk[local_idx]
        target_arr = targets_chunk[local_idx]
        
        # Convert to tensors efficiently
        # For contiguous arrays, torch.from_numpy doesn't copy
        if input_arr.flags['C_CONTIGUOUS']:
            input_tensor = torch.from_numpy(input_arr)
        else:
            input_tensor = torch.from_numpy(np.ascontiguousarray(input_arr))
            
        if target_arr.flags['C_CONTIGUOUS']:
            target_tensor = torch.from_numpy(target_arr)
        else:
            target_tensor = torch.from_numpy(np.ascontiguousarray(target_arr))
        
        # Pin memory only for CUDA devices
        if self.device_type == "cuda":
            input_tensor = input_tensor.pin_memory()
            target_tensor = target_tensor.pin_memory()
        
        return input_tensor, target_tensor
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about dataset structure for logging."""
        return {
            "format": "hdf5_chunked",
            "file": str(self.hdf5_path),
            "split": self.split_name,
            "n_samples": self.n_samples,
            "n_species": self.n_species,
            "n_globals": self.n_globals
        }
    
    def __del__(self):
        """Clean up HDF5 file handle."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except:
                pass


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
    Fixed multiprocessing initialization to avoid conflicts.
    """
    train_cfg = config["training"]
    device_type = (device or torch.device("cuda")).type
    
    from utils.hardware import optimize_dataloader_settings
    opts = optimize_dataloader_settings(
        batch_size=train_cfg["batch_size"],
        device_type=device_type,
        num_workers=train_cfg.get("num_workers"),
    )
    
    # Override with config values if specified
    opts["num_workers"] = train_cfg.get("num_workers", min(16, opts["num_workers"]))
    opts["prefetch_factor"] = None if opts["num_workers"] == 0 else min(
        4, max(2, opts["num_workers"] // 4)
    )
    
    # Setup multiprocessing safely
    if opts["num_workers"] > 0:
        import torch.multiprocessing as mp
        try:
            # Only set if not already set
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("forkserver")
        except RuntimeError:
            # Already set, which is fine
            pass
    
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
    
    # Set CUDA memory fraction if using pinned memory
    if opts["pin_memory"] and device_type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except AttributeError:  # older PyTorch
            pass
    
    logging.getLogger(__name__).info(
        f"DataLoader with {opts['num_workers']} worker(s), "
        f"prefetch={kwargs.get('prefetch_factor', 'N/A')}"
    )
    
    return DataLoader(**kwargs)