#!/usr/bin/env python3
"""
High-performance dataset implementation with sequence mode support.
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
        with open(shard_index_path) as f:
            self.shard_index = json.load(f)
        
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
        
        # Build shard info
        self.shards = self.split_info["shards"]
        self.n_shards = len(self.shards)
        self.n_total_samples = self.split_info["n_trajectories"]
        
        # Build lookup for shard boundaries
        self._build_shard_lookup()
        
        # Caching
        self.gpu_cache = None
        self.cpu_fallback = False
        self._try_gpu_cache()
    
    def _build_shard_lookup(self):
        """Build lookup arrays for shard access."""
        self.shard_starts = [0]
        cumsum = 0
        for shard in self.shards:
            cumsum += shard["n_samples"]
            self.shard_starts.append(cumsum)
        self.shard_starts = np.array(self.shard_starts[:-1])
    
    def _normalize_time(self, t: np.ndarray) -> np.ndarray:
        """Apply global log-min-max time normalization."""
        tau = np.log(1 + t / self.tau0)
        return (tau - self.tmin) / (self.tmax - self.tmin)
    
    def _try_gpu_cache(self):
        """Try to cache all data on GPU."""
        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False or self.device.type != "cuda":
            self.cpu_fallback = True
            return
        
        # Calculate memory needed
        bytes_per_float = 4 if self.dtype == torch.float32 else 8
        bytes_needed = self.n_total_samples * (
            self.n_species +  # x0_log
            self.n_globals +  # globals
            self.M +  # t_vec
            self.M * self.n_target_species  # y_mat
        ) * bytes_per_float
        
        free_mem, _ = torch.cuda.mem_get_info(self.device.index)
        if bytes_needed > free_mem * 0.85:
            self.logger.warning(f"Insufficient GPU memory for sequence caching. Using CPU fallback.")
            self.cpu_fallback = True
            return
        
        # Load all data
        self.logger.info(f"Loading {self.split_name} sequence data to GPU ({bytes_needed/1e9:.1f} GB)...")
        
        all_x0_log = []
        all_globals = []
        all_t_vec = []
        all_y_mat = []
        
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            data = np.load(shard_path)
            
            # Normalize time
            t_vec_norm = self._normalize_time(data['t_vec'])
            
            all_x0_log.append(torch.from_numpy(data['x0_log']).to(self.dtype))
            all_globals.append(torch.from_numpy(data['globals']).to(self.dtype))
            all_t_vec.append(torch.from_numpy(t_vec_norm).to(self.dtype))
            all_y_mat.append(torch.from_numpy(data['y_mat']).to(self.dtype))
        
        # Concatenate and move to GPU
        self.gpu_cache = {
            'x0_log': torch.cat(all_x0_log).to(self.device),
            'globals': torch.cat(all_globals).to(self.device),
            't_vec': torch.cat(all_t_vec).to(self.device),
            'y_mat': torch.cat(all_y_mat).to(self.device)
        }
        
        self.logger.info(f"GPU sequence cache loaded successfully")
    
    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single trajectory from disk."""
        # Find which shard
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]
        
        # Load shard
        shard_path = self.split_dir / self.shards[shard_idx]["filename"]
        data = np.load(shard_path)
        
        # Extract trajectory
        x0_log = data['x0_log'][local_idx]
        globals_vec = data['globals'][local_idx]
        t_vec = self._normalize_time(data['t_vec'][local_idx])
        y_mat = data['y_mat'][local_idx]
        
        # Combine inputs
        inputs = np.concatenate([x0_log, globals_vec, t_vec])
        
        # Convert to tensors
        inputs_tensor = torch.from_numpy(inputs).to(self.dtype)
        targets_tensor = torch.from_numpy(y_mat).to(self.dtype)
        
        return inputs_tensor, targets_tensor
    
    def __len__(self) -> int:
        return self.n_total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a trajectory sample."""
        if self.gpu_cache is not None:
            # Fast GPU path
            x0_log = self.gpu_cache['x0_log'][idx]
            globals_vec = self.gpu_cache['globals'][idx]
            t_vec = self.gpu_cache['t_vec'][idx]
            y_mat = self.gpu_cache['y_mat'][idx]
            
            # Combine inputs
            inputs = torch.cat([x0_log, globals_vec, t_vec])
            
            return inputs, y_mat
        else:
            # CPU fallback
            return self._load_from_disk(idx)


class NPYDataset(Dataset):
    """Original row-wise dataset for backward compatibility."""
    # Keep original implementation for non-sequence mode
    def __init__(self, shard_dir: Path, split_name: str, config: Dict[str, Any], 
                 device: torch.device):
        # Check if sequence mode
        shard_index_path = shard_dir / "shard_index.json"
        if shard_index_path.exists():
            with open(shard_index_path) as f:
                index = json.load(f)
                if index.get("sequence_mode", False):
                    raise ValueError("This is sequence mode data. Use SequenceDataset instead.")
        
        # Original implementation continues...
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
    """Create appropriate dataloader based on dataset type."""
    if dataset is None or len(dataset) == 0:
        logging.getLogger(__name__).warning("Cannot create DataLoader for empty dataset")
        return None
    
    log = logging.getLogger(__name__)
    tcfg = config["training"]
    bs = tcfg["batch_size"]
    
    # Check if sequence mode
    is_sequence = isinstance(dataset, SequenceDataset)
    
    if is_sequence:
        # Sequence mode - simple batching
        log.info(f"DataLoader[{dataset.split_name}] sequence mode: bs={bs}")
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,  # Already on GPU
            pin_memory=False,
            drop_last=drop_last
        )
    else:
        # Original implementation for row-wise data
        is_gpu_cached = getattr(dataset, "gpu_cache", None) is not None

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