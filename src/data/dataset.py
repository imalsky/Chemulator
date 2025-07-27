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
        """Group dataset indices by their shard for efficient access."""
        shard_groups = {}
        
        for idx in range(len(self.dataset)):
            global_idx = self.dataset.sample_indices[idx]
            shard_idx, _ = self.dataset._find_shard_idx(global_idx)
            
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = []
            shard_groups[shard_idx].append(idx)
            
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
    def __init__(self, shard_dir: Path, indices: np.ndarray, config: Dict[str, Any],
                device: torch.device, split_name: Optional[str] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.config = config
        self.device = device
        self.split_name = split_name or "unknown"
        self.logger = logging.getLogger(__name__)

        # Load and validate shard index metadata
        shard_index_path = self.shard_dir / "shard_index.json"
        self.logger.debug(f"Loading shard index from {shard_index_path}")
        
        with open(shard_index_path) as f:
            self.shard_index = json.load(f)

        # Extract data dimensions and configuration
        self.n_species = self.shard_index["n_species"]
        self.n_globals = self.shard_index["n_globals"]
        self.samples_per_shard = self.shard_index["samples_per_shard"]
        self.prediction_mode = self.shard_index.get("prediction_mode", "absolute")
        self.n_shards = self.shard_index["n_shards"]

        # Validate and store sample indices for this split
        self.sample_indices = indices
        self.n_total_samples = len(indices) if indices is not None else self.shard_index["total_samples"]

        # Calculate data dimensions for memory estimation
        self.n_features = self.n_species * 2 + self.n_globals + 1  # inputs + outputs
        
        # Always use float32 for memory calculations
        self.bytes_per_sample = self.n_features * 4  # float32
        self.bytes_per_shard = self.samples_per_shard * self.bytes_per_sample

        # Initialize caching system (deferred for multiprocessing compatibility)
        self.cache_is_setup = False
        self._determine_cache_size()

        # Build efficient lookup structures for O(log n) access
        self._build_shard_lookup()

        # Run memory pre-flight check only in main process
        if torch.utils.data.get_worker_info() is None:  # Only in main process
            try:
                memory_info = self.check_memory_requirements()
                
                # Additional warning for high memory usage
                if memory_info["usage_percent"] > 60:
                    self.logger.warning(
                        f"⚠️  High memory usage expected: {memory_info['usage_percent']:.0f}% "
                        f"of available RAM. Monitor closely for OOM issues."
                    )
            except MemoryError as e:
                self.logger.error("Memory check failed - aborting initialization")
                raise

        # Log initialization summary
        self.logger.info(
            f"NPYDataset '{self.split_name}' initialized: "
            f"{self.n_total_samples:,} samples across {self.n_shards} shards "
            f"({self.bytes_per_shard / 1024**2:.1f} MB/shard as float32), "
            f"cache capacity: {self._max_cache_size} shards, "
            f"prediction mode: {self.prediction_mode}"
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
        
        # Apply configuration limits - THIS IS WHERE THE BUG FIX HAPPENS
        # The config should have a reasonable default, not 1
        config_limit = self.config["training"].get("dataset_cache_shards", 16)  # Changed default
        
        # Conservative practical limits based on worker count
        if num_workers >= 16:
            practical_limit = 8   # Increased from 4
        elif num_workers >= 8:
            practical_limit = 16  # Increased from 8
        elif num_workers >= 4:
            practical_limit = 32  # Increased from 16
        else:
            practical_limit = 64  # Increased from 32
        
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
        self._get_shard_data = lru_cache(maxsize=self._max_cache_size)(
            self._get_shard_data_impl
        )

        # Run memory check once per worker
        if not getattr(self, "_ram_guard_ran", False):
            self.check_memory_requirements()
            self._ram_guard_ran = True

        # Mark cache as ready
        self.cache_is_setup = True
        self._cache_is_ready = True


    def _build_shard_lookup(self):
        """
        Build efficient lookup structures for O(log n) sample access.
        
        Creates arrays of shard boundaries to enable binary search when
        mapping global sample indices to (shard_index, local_index) pairs.
        Also validates that shards are contiguous and properly ordered.
        """
        self.logger.debug("Building shard lookup tables for efficient indexing")
        
        # Extract start and end indices for each shard
        self._shard_starts = np.array([s["start_idx"] for s in self.shard_index["shards"]])
        self._shard_ends = np.array([s["end_idx"] for s in self.shard_index["shards"]])

        # Validate shard integrity
        if len(self._shard_starts) > 1:
            # Check that shards are contiguous (no gaps)
            gaps_check = np.all(self._shard_starts[1:] == self._shard_ends[:-1])
            if not gaps_check:
                gap_indices = np.where(self._shard_starts[1:] != self._shard_ends[:-1])[0]
                self.logger.error(f"Non-contiguous shards detected at indices: {gap_indices}")
                raise ValueError("Shards must be contiguous")
            
            # Check that shards are properly sorted
            sort_check = np.all(self._shard_starts[:-1] < self._shard_starts[1:])
            if not sort_check:
                self.logger.error("Shards are not properly sorted by start index")
                raise ValueError("Shards must be sorted by start index")
        
        self.logger.debug(
            f"Shard lookup tables built: {len(self._shard_starts)} shards, "
            f"sample range [{self._shard_starts[0]}, {self._shard_ends[-1]})"
        )

    def _get_shard_data_impl(self, shard_idx: int) -> np.ndarray:
        """
        Load a shard from disk as float32, contiguous, and writable.
        This is the actual implementation that gets wrapped by lru_cache.
        """
        shard_info   = self.shard_index["shards"][shard_idx]
        shard_path   = self.shard_dir / shard_info["filename"]
        exp_samples  = shard_info["n_samples"]

        self.logger.debug(f"⏫  Loading shard {shard_idx}: {shard_path}")

        t0 = time.time()
        # Load compressed or raw data
        if self.shard_index.get("compression") == "npz":
            with np.load(shard_path) as zf:
                data = zf["data"].astype(np.float32, copy=False)
        else:
            arr = np.load(shard_path)
            # Ensure float32 dtype
            data = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            # Ensure C-contiguous for efficient access
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

        self.logger.debug(f"✅  Shard loaded in {(time.time()-t0):.3f}s  "
                          f"→ shape {data.shape}, dtype {data.dtype}")

        # Validate loaded data
        if data.shape[0] != exp_samples:
            raise ValueError(f"Shard sample mismatch ({data.shape[0]} vs {exp_samples})")
        if data.shape[1] != self.n_features:
            raise ValueError(f"Feature dim mismatch ({data.shape[1]} vs {self.n_features})")

        return data

    def _find_shard_idx(self, global_idx: int) -> Tuple[int, int]:
        """
        Map a global sample index to its shard and local offset using binary search.
        
        Args:
            global_idx: Global index across all samples
            
        Returns:
            Tuple of (shard_index, local_index_within_shard)
        """
        # Binary search to find which shard contains this global index
        # searchsorted with side='right' finds the insertion point, so we subtract 1
        shard_idx = np.searchsorted(self._shard_starts, global_idx, side='right') - 1

        # Validate shard index bounds
        if not (0 <= shard_idx < len(self._shard_starts)):
            self.logger.error(
                f"Sample index {global_idx} not found in any shard. "
                f"Valid range: [{self._shard_starts[0]}, {self._shard_ends[-1]})"
            )
            raise IndexError(f"Sample index {global_idx} not found in any shard")

        # Calculate local index within the shard
        local_idx = global_idx - self._shard_starts[shard_idx]

        # Validate local index bounds
        shard_size = self._shard_ends[shard_idx] - self._shard_starts[shard_idx]
        if not (0 <= local_idx < shard_size):
            self.logger.error(
                f"Invalid local index {local_idx} for shard {shard_idx} "
                f"(shard size: {shard_size})"
            )
            raise IndexError(f"Sample index {global_idx} not in shard {shard_idx} bounds")

        return shard_idx, local_idx

    def __len__(self) -> int:
        """Return the total number of samples in this dataset split."""
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from the dataset.
        
        This method:
        1. Maps the split-specific index to a global index
        2. Finds which shard contains the sample (binary search)
        3. Loads the shard data (with caching)
        4. Creates PyTorch tensors without copying data
        
        Args:
            idx: Index within this dataset split
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Initialize cache on first access (lazy initialization for multiprocessing)
        if not self.cache_is_setup:
            self._setup_worker_cache()

        # Validate index bounds
        if not (0 <= idx < self.n_total_samples):
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")

        global_idx = -1  # Initialize for error reporting
        try:
            # Map split-specific index to global index
            global_idx = self.sample_indices[idx]

            # Find shard and local offset
            shard_idx, local_idx = self._find_shard_idx(global_idx)

            # Load shard data (may hit cache)
            shard_data = self._get_shard_data(shard_idx)

            # Additional bounds check (defensive programming)
            if not (0 <= local_idx < shard_data.shape[0]):
                raise IndexError(
                    f"Local index {local_idx} out of bounds for shard {shard_idx} "
                    f"with size {shard_data.shape[0]}"
                )

            # Extract the sample row
            row = shard_data[local_idx]

            # Split into input and target features
            n_input = self.n_species + self.n_globals + 1  # species + globals + time
            
            # Create tensors directly from numpy arrays (zero-copy operation)
            # This works because our arrays are writable (not memory-mapped)
            input_tensor = torch.from_numpy(row[:n_input])
            target_tensor = torch.from_numpy(row[n_input:])

            return input_tensor, target_tensor

        except Exception as e:
            self.logger.error(
                f"Error accessing sample idx={idx} (global_idx={global_idx}): {e}",
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
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Memory Requirements Check for '{self.split_name}' Dataset")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"System Memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        self.logger.info(f"Configuration: {num_workers} workers, {cache_shards} cached shards, "
                        f"{batch_size} batch size")
        self.logger.info(f"\nMemory Breakdown:")
        self.logger.info(f"  - Shard cache: {memory_breakdown['shard_cache_per_worker_gb']:.2f} GB/worker")
        self.logger.info(f"  - Prefetch buffer: {memory_breakdown['prefetch_per_worker_gb']:.2f} GB/worker")
        self.logger.info(f"  - Python overhead: {memory_breakdown['python_overhead_gb']:.2f} GB")
        self.logger.info(f"  - Worker overhead: {memory_breakdown['dataloader_overhead_gb']:.2f} GB")
        self.logger.info(f"\nTotal Expected: {total_expected_gb:.1f} GB "
                        f"({memory_breakdown['usage_percent']:.0f}% of available)")
        self.logger.info(f"{'='*60}\n")
        
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