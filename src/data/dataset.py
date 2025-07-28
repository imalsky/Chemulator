#!/usr/bin/env python3
"""
High-performance dataset implementation for chemical kinetics training.

This module provides efficient data loading from numpy shard files with:
- Intelligent memory-based caching with LRU eviction
- Zero-copy tensor creation for optimal performance
- Multi-worker support with proper memory management
- Binary search for O(log n) sample lookups
- Conservative memory allocation to prevent OOM issues
- Shard-aware sampling to maximize cache efficiency
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Iterator
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import psutil
import os


class ShardAwareSampler(Sampler):
    """
    A sampler that generates indices in a shard-aware manner to maximize cache efficiency.
    
    This sampler:
    1. Groups indices by their shard
    2. Shuffles shards randomly
    3. Within each shard, shuffles indices randomly
    4. Yields batches that primarily come from 1-2 shards at a time
    
    This dramatically reduces cache misses and disk I/O.
    """
    def __init__(self, dataset: 'NPYDataset', batch_size: int, drop_last: bool = True, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        
        # Group indices by shard
        self.shard_to_indices = self._group_indices_by_shard()
        self.total_samples = len(dataset)
        
    def _group_indices_by_shard(self) -> Dict[int, List[int]]:
            """
            Group dataset indices by their shard using robust binary search.
            This correctly handles shards of variable sizes.
            """
            shard_groups = {}
            if not len(self.dataset):
                return shard_groups

            # Get the start indices of all shards in this split
            shard_starts = self.dataset._shard_starts
            
            # Use numpy's highly optimized searchsorted to find the
            # shard index for every sample in the dataset in one go.
            all_shard_indices = np.searchsorted(
                shard_starts, 
                np.arange(len(self.dataset)), 
                side='right'
            ) - 1
            
            # Now, group the dataset indices by their calculated shard index
            for dataset_idx, shard_idx in enumerate(all_shard_indices):
                # The key must be an integer for dictionary access
                shard_idx_int = int(shard_idx)
                if shard_idx_int not in shard_groups:
                    shard_groups[shard_idx_int] = []
                shard_groups[shard_idx_int].append(dataset_idx)
                
            return shard_groups
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices in a cache-friendly order."""
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.seed + torch.utils.data.get_worker_info().id 
                                     if torch.utils.data.get_worker_info() else self.seed)
        
        # Shuffle shard order
        shard_order = list(self.shard_to_indices.keys())
        rng.shuffle(shard_order)
        
        # Collect all indices in shard-aware order
        all_indices = []
        for shard_idx in shard_order:
            # Get indices for this shard and shuffle them
            shard_indices = self.shard_to_indices[shard_idx].copy()
            rng.shuffle(shard_indices)
            all_indices.extend(shard_indices)
        
        # Yield batches
        for i in range(0, len(all_indices) - self.batch_size + 1, self.batch_size):
            yield all_indices[i:i + self.batch_size]
            
        # Handle last batch
        if not self.drop_last and len(all_indices) % self.batch_size != 0:
            yield all_indices[-(len(all_indices) % self.batch_size):]
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class NPYDataset(Dataset):
    """
    PyTorch Dataset for loading chemical kinetics data from numpy shard files.
    
    This dataset efficiently handles large-scale data by:
    - Loading entire shards into memory for fast access (no mmap overhead)
    - Caching frequently accessed shards with LRU eviction
    - Creating PyTorch tensors without data copying
    - Supporting train/validation/test splits via index arrays
    - Conservative memory allocation to prevent OOM issues
    - Providing shard-aware sampling for cache efficiency
    
    Args:
        shard_dir: Directory containing shard files and metadata
        indices: Array of global sample indices for this split
        config: Training configuration dictionary
        device: PyTorch device (used for logging, not data loading)
        split_name: Name of this split (train/val/test) for logging
    """
    def __init__(self, shard_dir: Path, split_name: str, config: Dict[str, Any], device: torch.device):
        """Initialize dataset for a specific split - much simpler!"""
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
        
        # Get shard info for this split
        self.shards = self.split_info["shards"]
        self.n_shards = len(self.shards)
        self.n_total_samples = self.split_info["total_samples"]
        
        # Build lookup arrays for O(log n) access
        self._shard_starts = np.array([s["start_idx"] for s in self.shards])
        self._shard_ends = np.array([s["end_idx"] for s in self.shards])
        
        # Memory calculations
        self.bytes_per_sample = self.n_features * 4  # float32
        self.bytes_per_shard = self.samples_per_shard * self.bytes_per_sample
        
        # Initialize caching
        self.cache_is_setup = False
        self._determine_cache_size()
        
        # Log initialization
        self.logger.info(
            f"NPYDataset '{self.split_name}' initialized: "
            f"{self.n_total_samples:,} samples across {self.n_shards} shards "
            f"({self.bytes_per_shard / 1024**2:.1f} MB/shard), "
            f"cache capacity: {self._max_cache_size} shards"
        )


    def _determine_cache_size(self):
        """
        Calculate optimal shard cache size based on available system memory.
        Uses conservative estimates to prevent OOM issues when multiple
        datasets are running simultaneously.
        """
        # Query available system memory
        try:
            mem_info = psutil.virtual_memory()
            available_memory = mem_info.available
            total_memory = mem_info.total
            memory_percent_free = (available_memory / total_memory) * 100
            
            self.logger.debug(
                f"System memory: {total_memory / 1024**3:.1f} GB total, "
                f"{available_memory / 1024**3:.1f} GB available ({memory_percent_free:.1f}% free)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to query system memory: {e}. Using 4GB fallback.")
            available_memory = 4 * 1024**3  # Conservative 4GB fallback

        # Conservative memory allocation (30% of available)
        cache_memory_fraction = 0.3
        total_cache_memory = available_memory * cache_memory_fraction
        
        # Account for multiple datasets running concurrently
        # Assume up to 3 datasets (train/val/test) may be active
        num_concurrent_datasets = 3
        cache_memory_per_dataset = total_cache_memory / num_concurrent_datasets
        
        # Account for workers in this dataset
        num_workers = self.config["training"].get("num_workers", 1)
        
        if num_workers > 0:
            max_cache_memory_per_worker = cache_memory_per_dataset / num_workers
            
            self.logger.debug(
                f"Allocating {cache_memory_per_dataset / 1024**3:.1f} GB cache memory "
                f"for {self.split_name} dataset across {num_workers} workers "
                f"({max_cache_memory_per_worker / 1024**3:.2f} GB each)"
            )
        else:
            max_cache_memory_per_worker = cache_memory_per_dataset

        # Account for memory overhead and fragmentation
        # Each shard needs ~2x its size due to Python overhead, fragmentation, etc.
        memory_overhead_factor = 2.0
        effective_shard_size = self.bytes_per_shard * memory_overhead_factor
        
        # Calculate how many shards fit in allocated memory per worker
        memory_based_shards = int(max_cache_memory_per_worker / max(1, effective_shard_size))
        
        config_limit = self.config["training"].get("dataset_cache_shards", 16)  # Changed default
        
        if num_workers >= 16:
            practical_limit = 64
        elif num_workers >= 8:
            practical_limit = 256
        elif num_workers >= 4:
            practical_limit = 512
        else:
            practical_limit = 1024
        
        # Use the minimum of all constraints, with a floor of 1 shard
        self._max_cache_size = max(1, min(
            memory_based_shards,
            config_limit,
            practical_limit
        ))
        
        # Calculate and log expected memory usage
        expected_cache_gb = (self._max_cache_size * effective_shard_size) / 1024**3
        total_expected_gb = expected_cache_gb * num_workers
        
        self.logger.info(
            f"Cache size for '{self.split_name}': "
            f"{self._max_cache_size} shards per worker "
            f"(memory_based={memory_based_shards}, config={config_limit}, practical={practical_limit}). "
            f"Expected memory: {expected_cache_gb:.1f} GB per worker, "
            f"{total_expected_gb:.1f} GB total for {num_workers} workers"
        )
        
        # Warn if memory usage seems high
        if total_expected_gb > total_memory / 1024**3 * 0.5:
            self.logger.warning(
                f"High memory usage warning: Expected cache memory ({total_expected_gb:.1f} GB) "
                f"exceeds 50% of system RAM. Consider reducing num_workers or dataset_cache_shards."
            )

    def _setup_worker_cache(self) -> None:
        """
        Build an independent LRU cache in each worker process.
        This is called lazily when the dataset is first accessed in a worker.
        """
        if getattr(self, "_cache_is_ready", False):
            return

        # Create the LRU-cached version of the shard loader
        self._get_shard_data = lru_cache(maxsize=self._max_cache_size)(self._get_shard_data_impl)

        # Run memory check once per worker
        if not getattr(self, "_ram_guard_ran", False):
            self.check_memory_requirements()
            self._ram_guard_ran = True

        # Mark cache as ready
        self.cache_is_setup = True
        self._cache_is_ready = True


    def _get_shard_data_impl(self, shard_idx: int) -> np.ndarray:
        """Load a shard from the split-specific directory."""
        shard_info = self.shards[shard_idx]
        shard_path = self.split_dir / shard_info["filename"]
        exp_samples = shard_info["n_samples"]

        self.logger.debug(f"Loading shard {shard_idx} from {self.split_name}: {shard_path}")

        t0 = time.time()
        
        # Load data
        if self.shard_index.get("compression") == "npz":
            with np.load(shard_path) as zf:
                data = zf["data"].astype(np.float32, copy=False)
        else:
            arr = np.load(shard_path)
            data = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

        self.logger.debug(f"Shard loaded in {(time.time()-t0):.3f}s")

        # Validate
        if data.shape[0] != exp_samples:
            raise ValueError(f"Shard sample mismatch ({data.shape[0]} vs {exp_samples})")
        if data.shape[1] != self.n_features:
            raise ValueError(f"Feature dim mismatch ({data.shape[1]} vs {self.n_features})")

        return data

    def __len__(self) -> int:
        """Return the total number of samples in this dataset split."""
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single sample - much simpler without index translation!"""
        # Initialize cache on first access
        if not self.cache_is_setup:
            self._setup_worker_cache()

        # Validate index
        if not (0 <= idx < self.n_total_samples):
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")

        try:
            # Find shard using binary search
            shard_idx = np.searchsorted(self._shard_starts, idx, side='right') - 1
            
            # Calculate local index within shard
            local_idx = idx - self._shard_starts[shard_idx]
            
            # Load shard data (may hit cache)
            shard_data = self._get_shard_data(shard_idx)
            
            # Extract the sample row
            row = shard_data[local_idx]
            
            # Split into input and target
            n_input = self.n_species + self.n_globals + 1
            
            # Create tensors (zero-copy)
            input_tensor = torch.from_numpy(row[:n_input])
            target_tensor = torch.from_numpy(row[n_input:])
            
            return input_tensor, target_tensor

        except Exception as e:
            self.logger.error(
                f"Error accessing sample idx={idx} in {self.split_name}: {e}",
                exc_info=True
            )
            raise

    def check_memory_requirements(self) -> Dict[str, float]:
        """
        Pre-flight check: Estimate memory requirements and validate against available RAM.
        Accounts for the fact that multiple datasets may be running concurrently.
        Returns dict with memory estimates.
        """
        import psutil
        
        # Get current memory state
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / 1024**3
        total_gb = mem_info.total / 1024**3
        
        # Calculate per-component memory usage
        num_workers = self.config["training"]["num_workers"]
        cache_shards = self._max_cache_size
        batch_size = self.config["training"]["batch_size"]
        prefetch = self.config["training"].get("prefetch_factor", 2)
        
        # Memory calculations (in GB)
        shard_gb = self.bytes_per_shard / 1024**3
        batch_gb = (batch_size * self.n_features * 4) / 1024**3  # float32
        
        # Component breakdown
        memory_breakdown = {
            "shard_cache_per_worker_gb": cache_shards * shard_gb * 2,  # 2x for overhead
            "prefetch_per_worker_gb": prefetch * batch_gb,
            "num_workers": num_workers,
            "python_overhead_gb": 1.0,  # Base Python/PyTorch overhead
            "dataloader_overhead_gb": num_workers * 0.5,  # Per-worker overhead
        }
        
        # Total expected usage for this dataset
        total_expected_gb = (
            num_workers * memory_breakdown["shard_cache_per_worker_gb"] +
            num_workers * memory_breakdown["prefetch_per_worker_gb"] +
            memory_breakdown["python_overhead_gb"] +
            memory_breakdown["dataloader_overhead_gb"]
        )
        
        memory_breakdown["total_expected_gb"] = total_expected_gb
        memory_breakdown["available_gb"] = available_gb
        memory_breakdown["total_system_gb"] = total_gb
        memory_breakdown["usage_percent"] = (total_expected_gb / available_gb) * 100
        
        # Log detailed breakdown
        #self.logger.info(f"\n{'='*60}")
        #self.logger.info(f"Memory Requirements Check for '{self.split_name}' Dataset")
        #self.logger.info(f"{'='*60}")
        #self.logger.info(f"System Memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        #self.logger.info(f"Configuration: {num_workers} workers, {cache_shards} cached shards, "
        #                f"{batch_size} batch size")
        #self.logger.info(f"\nMemory Breakdown:")
        #self.logger.info(f"  - Shard cache: {memory_breakdown['shard_cache_per_worker_gb']:.2f} GB/worker")
        #self.logger.info(f"  - Prefetch buffer: {memory_breakdown['prefetch_per_worker_gb']:.2f} GB/worker")
        #self.logger.info(f"  - Python overhead: {memory_breakdown['python_overhead_gb']:.2f} GB")
        #self.logger.info(f"  - Worker overhead: {memory_breakdown['dataloader_overhead_gb']:.2f} GB")
        #self.logger.info(f"\nTotal Expected: {total_expected_gb:.1f} GB "
        #                f"({memory_breakdown['usage_percent']:.0f}% of available)")
        #self.logger.info(f"{'='*60}\n")
        
        # Validate memory requirements
        safety_factor = 0.8  # Don't use more than 80% of available memory
        if total_expected_gb > available_gb * safety_factor:
            error_msg = (
                f"Insufficient memory for {self.split_name} dataset: "
                f"need {total_expected_gb:.1f} GB, "
                f"but only {available_gb * safety_factor:.1f} GB safely available. "
                f"Reduce num_workers (currently {num_workers}) or "
                f"dataset_cache_shards (currently {cache_shards})."
            )
            self.logger.error(error_msg)
            raise MemoryError(error_msg)
        
        return memory_breakdown


def create_dataloader(dataset: Dataset,
                      config: Dict[str, Any],
                      shuffle: bool = True,
                      device: Optional[torch.device] = None,
                      drop_last: bool = True,
                      use_shard_aware_sampling: bool = True) -> DataLoader:
    """
    Build a DataLoader with safe, high‑performance defaults.
    
    Args:
        dataset: The dataset to load from
        config: Configuration dictionary
        shuffle: Whether to shuffle the data (ignored if using shard-aware sampling)
        device: PyTorch device for memory checks
        drop_last: Whether to drop the last incomplete batch
        use_shard_aware_sampling: Whether to use shard-aware sampling for better cache efficiency
        
    Returns:
        Configured DataLoader or None if dataset is empty
    """
    if dataset is None or len(dataset) == 0:
        logging.getLogger(__name__).warning("Cannot create DataLoader for empty dataset")
        return None

    log     = logging.getLogger(__name__)
    tcfg    = config["training"]
    bs      = tcfg["batch_size"]
    workers = min(32, tcfg.get("num_workers", 0))

    # Pin memory only if using CUDA and workers
    pin   = (tcfg.get("pin_memory", False) and workers > 0
             and device is not None and device.type == "cuda")
    
    # Persistent workers only if pinning memory
    pers  = tcfg.get("persistent_workers", False) and pin
    
    # Prefetch factor only matters with workers
    pre   = tcfg.get("prefetch_factor", 2) if workers > 0 else 1

    # Validate batch and prefetch settings for GPU memory
    _validate_batch_prefetch(bs, pre, dataset.n_features, device)

    # Determine if we should use shard-aware sampling
    if use_shard_aware_sampling and shuffle and isinstance(dataset, NPYDataset):
        # Use shard-aware sampler for training data
        batch_sampler = ShardAwareSampler(
            dataset=dataset,
            batch_size=bs,
            drop_last=drop_last,
            seed=config.get("system", {}).get("seed", 42)
        )
        
        log.info(f"DataLoader[{dataset.split_name}] using ShardAwareSampler: "
                 f"bs={bs}  workers={workers}  pin={pin}  pers={pers}  prefetch={pre}")
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,  # Use our shard-aware sampler
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=pers,
            prefetch_factor=pre,
            worker_init_fn=_worker_init_fn,
        )
    else:
        # Use standard DataLoader for validation/test or when shard-aware is disabled
        log.info(f"DataLoader[{dataset.split_name}]  bs={bs}  workers={workers}  "
                 f"pin={pin}  pers={pers}  prefetch={pre}")

        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=pers,
            prefetch_factor=pre,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
        )


def _worker_init_fn(worker_id: int):
    """
    Initialization code run once per DataLoader worker process.
    - Disable CUDA in workers (they should only load data)
    - Limit OpenBLAS/MKL threads to prevent oversubscription
    - Set deterministic but distinct RNG seeds
    - Light stagger to avoid simultaneous disk hits
    """
    import os, time, numpy as np, torch, random

    # Prevent workers from using CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Limit CPU threads to prevent oversubscription
    torch.set_num_threads(1)

    # Stagger worker startup to avoid disk contention
    time.sleep(0.25 * worker_id)

    # Set unique but deterministic seeds for each worker
    seed = int(time.time()) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _validate_batch_prefetch(batch_size: int,
                             prefetch_factor: int,
                             feature_dim: int,
                             device: Optional[torch.device]) -> None:
    """
    Validate that prefetched batches won't exceed GPU memory.
    Raises RuntimeError if the configuration would cause OOM.
    """
    if device is None or device.type != "cuda":
        return
    
    import torch
    
    # Calculate memory needed for prefetched batches
    bytes_needed = batch_size * max(prefetch_factor, 1) * feature_dim * 4  # float32
    
    # Get available GPU memory
    free_bytes = torch.cuda.mem_get_info(device.index)[0]
    
    # Check if we'd use more than 80% of free GPU memory
    if bytes_needed > free_bytes * 0.80:
        human = bytes_needed / 1024**3
        raise RuntimeError(
            f"prefetch_factor×batch_size would pre‑queue ≈{human:.1f} GB "
            "→ exceeds safe free GPU memory. Reduce batch_size or prefetch_factor."
        )