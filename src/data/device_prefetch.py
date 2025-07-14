#!/usr/bin/env python3
"""
Device prefetching utilities for overlapping data transfer with computation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Iterator, Tuple


class DevicePrefetchLoader:
    """
    Wraps a DataLoader to prefetch batches to GPU in a separate CUDA stream.
    This overlaps data transfer with model computation for better GPU utilization.
    """
    
    def __init__(self, loader: DataLoader, device: torch.device):
        """
        Initialize the prefetch loader.
        
        Args:
            loader: The DataLoader to wrap
            device: Target device (should be cuda)
        """
        self.loader = loader
        self.device = device
        
        # Only use CUDA streams for CUDA devices
        if device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=device)
        else:
            self.stream = None
        
        self.preload_batch = None
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        # Start the preload
        loader_iter = iter(self.loader)
        self._preload(loader_iter)
        
        while self.preload_batch is not None:
            # Wait for the preload to complete and get the batch
            if self.stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
            
            batch = self.preload_batch
            
            # Start preloading the next batch
            self._preload(loader_iter)
            
            yield batch
    
    def _preload(self, loader_iter: Iterator):
        """Preload the next batch in a separate stream."""
        try:
            data = next(loader_iter)
        except StopIteration:
            self.preload_batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                # Move batch to device asynchronously
                self.preload_batch = self._move_to_device(data)
        else:
            # No CUDA stream available, just move normally
            self.preload_batch = self._move_to_device(data)
    
    def _move_to_device(self, data):
        """Move data to device with non_blocking."""
        if isinstance(data, (tuple, list)):
            return tuple(
                d.to(self.device, non_blocking=True) if isinstance(d, torch.Tensor) else d
                for d in data
            )
        elif isinstance(data, dict):
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        else:
            return data
        
    # ──────────────────────────────────────────────────────────────
    def __getattr__(self, name):
        """
        Forward attribute look-ups that this wrapper does not define to the
        underlying DataLoader. This exposes .num_workers, .batch_size, etc.,
        so downstream code can treat DevicePrefetchLoader just like a regular
        DataLoader.

        NOTE: magic (dunder) names are excluded to avoid infinite recursion.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.loader, name)