#!/usr/bin/env python3
"""
High-performance dataset implementation for chemical kinetics training.

This module provides efficient data loading from numpy shard files with:
- Intelligent memory-based caching with LRU eviction
- Zero-copy tensor creation for optimal performance
- Multi-worker support with proper memory management
- Binary search for O(log n) sample lookups
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import psutil
import os

class NPYDataset(Dataset):
    """
    PyTorch Dataset for loading chemical kinetics data from numpy shard files.
    
    This dataset efficiently handles large-scale data by:
    - Loading entire shards into memory for fast access (no mmap overhead)
    - Caching frequently accessed shards with LRU eviction
    - Creating PyTorch tensors without data copying
    - Supporting train/validation/test splits via index arrays
    
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
        
        # FIX: Always use float32 for calculations
        self.bytes_per_sample = self.n_features * 4  # float32
        self.bytes_per_shard = self.samples_per_shard * self.bytes_per_sample

        # Initialize caching system (deferred for multiprocessing compatibility)
        self.cache_is_setup = False
        self._determine_cache_size()

        # Build efficient lookup structures for O(log n) access
        self._build_shard_lookup()

        # FIX: Run memory pre-flight check
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
        FIXED: More accurate memory calculations and conservative defaults.
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

        # FIX: More conservative memory allocation (30% instead of 40%)
        cache_memory_fraction = 0.3
        total_cache_memory = available_memory * cache_memory_fraction
        
        # FIX: Account for actual worker distribution
        # Each DataLoader instance (train/val/test) has its own workers
        num_workers = self.config["training"].get("num_workers", 1)
        # Don't multiply by num_loaders here - this is per dataset instance
        
        if num_workers > 0:
            max_cache_memory_per_worker = total_cache_memory / num_workers
            
            self.logger.debug(
                f"Allocating {total_cache_memory / 1024**3:.1f} GB total cache memory "
                f"across {num_workers} workers ({max_cache_memory_per_worker / 1024**3:.2f} GB each) "
                f"for {self.split_name} dataset"
            )
        else:
            max_cache_memory_per_worker = total_cache_memory

        # FIX: Account for memory overhead and fragmentation
        # Each shard needs ~2x its size due to Python overhead, fragmentation, etc.
        memory_overhead_factor = 2.0
        effective_shard_size = self.bytes_per_shard * memory_overhead_factor
        
        # Calculate how many shards fit in allocated memory per worker
        memory_based_shards = int(max_cache_memory_per_worker / max(1, effective_shard_size))
        
        # Apply configuration limits
        config_limit = self.config["training"].get("dataset_cache_shards", 256)
        
        # FIX: More conservative practical limits
        if num_workers >= 16:
            practical_limit = 4   # Very conservative for many workers
        elif num_workers >= 8:
            practical_limit = 8   # Moderate for medium worker count
        elif num_workers >= 4:
            practical_limit = 16  # Still conservative
        else:
            practical_limit = 32  # More generous for few workers
        
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

    def _setup_worker_cache(self):
        """
        Build the LRU cache **once** in the parent; worker processes reuse it.
        This keeps only one copy of each shard in host RAM regardless of
        `num_workers`.
        """
        if self.cache_is_setup:
            return

        # Parent process (get_worker_info() is None) builds the cache once
        if torch.utils.data.get_worker_info() is None:
            NPYDataset._global_cache = lru_cache(maxsize=self._max_cache_size)(
                self._get_shard_data_impl
            )

        # Every process—parent or worker—uses the same callable
        self._get_shard_data = NPYDataset._global_cache
        self.cache_is_setup = True

        NPYDataset._cache_owner = os.getpid() 



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
        Load a shard file from disk into memory as a writable numpy array.
        FIXED: Force float32 and avoid unnecessary copies.
        """
        shard_info = self.shard_index["shards"][shard_idx]
        shard_path = self.shard_dir / shard_info["filename"]
        expected_samples = shard_info["n_samples"]

        self.logger.debug(
            f"Loading shard {shard_idx} from {shard_path} "
            f"({expected_samples} samples, {self.bytes_per_shard / 1024**2:.1f} MB)"
        )

        try:
            load_start_time = time.time() if self.logger.isEnabledFor(logging.DEBUG) else 0
            
            if self.shard_index.get("compression") == "npz":
                # Handle compressed shards
                with np.load(shard_path) as npz_file:
                    # Force float32 during load
                    data = npz_file['data'].astype(np.float32, copy=False)
                    self.logger.debug(f"Loaded compressed shard in {time.time() - load_start_time:.3f}s")
            else:
                # FIX: Load and immediately convert to float32 without extra copies
                loaded_data = np.load(shard_path)
                
                # Only convert if necessary
                if loaded_data.dtype != np.float32:
                    self.logger.debug(f"Converting from {loaded_data.dtype} to float32")
                    data = loaded_data.astype(np.float32)
                else:
                    data = loaded_data
                
                # Only make contiguous if necessary (avoiding copy when possible)
                if not data.flags['C_CONTIGUOUS']:
                    self.logger.debug("Converting shard to C-contiguous layout")
                    # If we already converted dtype, data is a copy, so we can use copy=False
                    if data is loaded_data:
                        data = np.ascontiguousarray(data, dtype=np.float32)
                    else:
                        # Already a copy from dtype conversion
                        data = np.ascontiguousarray(data)
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    load_time = time.time() - load_start_time
                    throughput_mbps = (self.bytes_per_shard / 1024**2) / load_time
                    actual_size_mb = data.nbytes / 1024**2
                    self.logger.debug(
                        f"Loaded shard in {load_time:.3f}s ({throughput_mbps:.1f} MB/s), "
                        f"actual size: {actual_size_mb:.1f} MB, "
                        f"dtype: {data.dtype}, "
                        f"C_CONTIGUOUS: {data.flags['C_CONTIGUOUS']}"
                    )
            
            # Validate loaded data
            if data.shape[0] != expected_samples:
                raise ValueError(
                    f"Shard shape mismatch: expected {expected_samples} samples, "
                    f"got {data.shape[0]}"
                )
            
            if data.shape[1] != self.n_features:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features}, "
                    f"got {data.shape[1]}"
                )
                
            return data
            
        except Exception as e:
            self.logger.error(
                f"Failed to load shard {shard_idx} from {shard_path}: {e}",
                exc_info=True
            )
            raise

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
            # This works because our arrays are now writable (not memory-mapped)
            input_tensor = torch.from_numpy(row[:n_input])
            target_tensor = torch.from_numpy(row[n_input:])

            return input_tensor, target_tensor

        except Exception as e:
            self.logger.error(
                f"Error accessing sample idx={idx} (global_idx={global_idx}): {e}",
                exc_info=True
            )
            raise

def _worker_init_fn(worker_id: int):
    """
    Called once in every new DataLoader worker.
    - Sleeps a little so only ~1‑2 workers start I/O simultaneously.
    - Seeds NumPy / PyTorch RNGs so shuffling differs per worker.
    """
    import time, numpy as np, torch, random
    # < 1 s total extra start‑up time even with 4 workers
    time.sleep(0.25 * worker_id)

    seed = int(time.time()) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

def create_dataloader(dataset: Dataset, config: Dict[str, Any], shuffle: bool = True,
                     device: Optional[torch.device] = None, drop_last: bool = True) -> DataLoader:
    """
    Create a high-performance PyTorch DataLoader with optimal settings.
    
    This function configures the DataLoader with:
    - Multi-worker data loading for parallelism
    - Pinned memory for fast GPU transfers (when using CUDA)
    - Persistent workers to avoid process restart overhead
    - Appropriate prefetch factor for smooth data pipeline
    
    Args:
        dataset: PyTorch Dataset instance
        config: Training configuration dictionary
        shuffle: Whether to shuffle data (typically True for training)
        device: PyTorch device for determining GPU features
        drop_last: Whether to drop incomplete final batch
        
    Returns:
        Configured DataLoader instance or None if dataset is empty
    """
    if dataset is None or len(dataset) == 0:
        logging.getLogger(__name__).warning("Cannot create DataLoader for empty dataset")
        return None

    logger = logging.getLogger(__name__)
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    
    # Determine number of worker processes
    # Cap at 32 to prevent system instability from too many processes
    requested_workers = train_cfg.get("num_workers", 0)
    num_workers = min(32, requested_workers)
    
    if num_workers != requested_workers:
        logger.warning(
            f"Capped num_workers at {num_workers} (requested: {requested_workers}) "
            f"to prevent system instability"
        )

    # Determine if we can use performance features
    # These require CUDA GPU and multiple workers
    can_use_performance_features = (num_workers > 0 and device and device.type == "cuda")

    # Configure pin_memory for fast CPU-to-GPU transfers
    # This is one of the most important settings for GPU training performance
    pin_memory = train_cfg.get("pin_memory", False) and can_use_performance_features
    
    # Configure persistent_workers to avoid worker restart overhead between epochs
    # This significantly reduces epoch transition time
    persistent_workers = train_cfg.get("persistent_workers", False) and can_use_performance_features
    
    # Configure prefetch_factor (number of batches each worker loads in advance)
    # Only applicable when using multiple workers
    prefetch_factor = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

    # Log the effective DataLoader configuration
    logger.info(
        f"Creating DataLoader for '{dataset.split_name}' split: "
        f"batch_size={batch_size}, shuffle={shuffle}, drop_last={drop_last}"
    )
    logger.info(
        f"Performance settings: workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}"
    )
    
    if can_use_performance_features:
        logger.debug("GPU-optimized features enabled (pin_memory, persistent_workers)")
    else:
        reasons = []
        if num_workers == 0:
            reasons.append("no workers")
        if not device or device.type != "cuda":
            reasons.append("no CUDA device")
        logger.debug(f"GPU-optimized features disabled: {', '.join(reasons)}")

    # Create the DataLoader with optimized settings
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn,
        drop_last=drop_last
    )


def check_memory_requirements(self) -> Dict[str, float]:
    """
    Pre-flight check: Estimate memory requirements and validate against available RAM.
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
        "shard_cache_per_worker_gb": cache_shards * shard_gb * 2,  # 2x for boundary crossing
        "prefetch_per_worker_gb": prefetch * batch_gb,
        "num_workers": num_workers,
        "python_overhead_gb": 1.0,  # Base Python/PyTorch overhead
        "dataloader_overhead_gb": num_workers * 0.5,  # Per-worker overhead
    }
    
    # Total expected usage
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