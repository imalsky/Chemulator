#!/usr/bin/env python3
"""
High-performance dataset implementation for chemical kinetics models.

Provides efficient data loading with:
- Config-driven normalization for inputs and targets
- GPU caching for small datasets
- Chunked loading for large datasets
- Support for multiple data types (float32/float64)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.normalizer import NormalizationHelper


# Constants to avoid magic numbers
GPU_MEMORY_RESERVE_BYTES = 3_000_000_000  # 3GB
GPU_MEMORY_OVERHEAD_FACTOR = 1.15  # 15% overhead
DEFAULT_GPU_CACHE_FRACTION = 0.5
BYTES_PER_FLOAT = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


class SequenceDataset(Dataset):
    """
    Dataset for sequence-mode (trajectory-based) data with normalization.
    
    Efficiently loads preprocessed trajectory data and applies
    config-driven normalization to both inputs and targets.
    """
    
    def __init__(
        self,
        shard_dir: Path,
        split_name: str,
        config: Dict[str, Any],
        device: torch.device,
        norm_stats: Optional[Dict[str, Any]] = None,
        disable_cache: bool = False
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            shard_dir: Directory containing preprocessed shards
            split_name: Split name ('train', 'validation', 'test')
            config: Configuration with data and normalization settings
            device: Target device for tensors
            norm_stats: Optional normalization statistics
            disable_cache: Explicitly disable shard caching (for multi-worker safety)
        """
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.norm_stats = norm_stats or {}
        self._in_worker_process = disable_cache
        self._closed = False

        # Configure data types
        self._setup_dtypes()
        
        # Load and validate metadata
        self._load_metadata()
        
        # Extract configuration parameters
        self._setup_dimensions()
        
        # Build shard lookup table
        self._build_shard_lookup()

        # Auto-load normalization stats if not provided
        if not self.norm_stats or not self.norm_stats.get("per_key_stats"):
            norm_path = self.shard_dir / "normalization.json"
            if norm_path.exists():
                with open(norm_path, 'r', encoding='utf-8') as f:
                    self.norm_stats = json.load(f)
            else:
                raise ValueError(f"Normalization stats required but not found.")
    
        # Initialize normalization helper
        self._setup_normalization()
        
        # Initialize caching
        self.gpu_cache = None
        self._shard_cache = {"name": None, "data": None}
        
        # Try to cache entire dataset on GPU if possible
        self._try_gpu_cache()

    def _setup_dtypes(self) -> None:
        """Configure data types from config."""
        dtype_str = self.config["system"].get("dtype", "float32")
        try:
            self.dtype = getattr(torch, dtype_str)
        except AttributeError:
            self.logger.warning("Unknown dtype '%s', using float32", dtype_str)
            self.dtype = torch.float32
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32

    def _load_metadata(self) -> None:
        """Load and validate shard metadata."""
        shard_index_path = self.shard_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
            
        with open(shard_index_path, 'r', encoding='utf-8') as f:
            self.shard_index = json.load(f)
            
        if not self.shard_index.get("sequence_mode", False):
            raise ValueError("Expected sequence mode data for SequenceDataset")
        
        # Check for variable-length sequences and error if found
        if self.shard_index.get("variable_length", False):
            raise ValueError("Variable-length sequences detected.")
        
        self.split_info = self.shard_index["splits"][self.split_name]
        self.M = int(self.shard_index.get("M_per_sample", 0))
        
        if self.M <= 0:
            raise ValueError(f"Invalid M_per_sample: {self.M}")

    def _setup_dimensions(self) -> None:
        """Extract dimensions and variable lists from config."""
        data_config = self.config["data"]
        
        self.species_vars = data_config["species_variables"]
        self.target_vars = data_config.get("target_species_variables", self.species_vars)
        self.global_vars = data_config["global_variables"]
        self.time_var = data_config["time_variable"]

        # Compute counts
        self.n_species = len(self.species_vars)
        self.n_targets = len(self.target_vars)
        self.n_globals = len(self.global_vars)
        
        # Verify counts against metadata
        si = self.shard_index
        if (self.n_species != int(si["n_input_species"]) or
            self.n_targets != int(si["n_target_species"]) or
            self.n_globals != int(si["n_globals"])):
            raise ValueError(f"Variable count mismatch with shard_index.json: ")

    def _build_shard_lookup(self) -> None:
        """Build efficient lookup arrays for shard access."""
        self.shards = self.split_info.get("shards", [])
        self.n_total_samples = self.split_info.get("n_trajectories", 0)
        
        if self.n_total_samples == 0:
            self.logger.warning("No samples found in '%s' split", self.split_name)
            return
        
        # Build cumulative sum for fast shard indexing
        cumsum = 0
        self.shard_starts = []
        
        for shard in self.shards:
            if "n_trajectories" not in shard:
                raise KeyError(f"Shard missing n_trajectories: {shard}")
            self.shard_starts.append(cumsum)
            cumsum += int(shard["n_trajectories"])
        
        self.shard_starts = np.array(self.shard_starts, dtype=np.int64)
        self.n_total_samples = cumsum

    def _setup_normalization(self) -> None:
        """Initialize normalization helper with statistics."""
        if self.norm_stats and self.norm_stats.get("per_key_stats"):
            self.norm_helper = NormalizationHelper(
                self.norm_stats, 
                self.device,
                self.config
            )
            self.logger.info("Normalization enabled for '%s' dataset", self.split_name)
        else:
            raise RuntimeError(f"Cannot initialize normalization helper for '{self.split_name}' dataset.")

    def _normalize_time_tensor(self, t_np: np.ndarray) -> torch.Tensor:
        """
        Normalize time values and return as tensor on the correct device.
        """
        if not self.norm_helper:
            raise RuntimeError(
                f"Cannot normalize time values: normalization helper not initialized. "
                f"Ensure normalization.json exists at {self.shard_dir / 'normalization.json'} "
                f"or provide norm_stats when creating the dataset."
            )
        
        # Convert to tensor (initially on CPU)
        t_tensor = torch.from_numpy(t_np.astype(self.np_dtype, copy=False))
        
        # Normalize (this moves to self.device internally and returns on self.device)
        return self.norm_helper.normalize_time(t_tensor)

    def _try_gpu_cache(self) -> None:
        """Attempt to cache entire dataset on GPU if memory permits."""
        if self.n_total_samples == 0:
            return
        
        # Check GPU cache configuration
        gpu_cache_setting = self.config.get("training", {}).get("gpu_cache_dataset", "auto")
        if gpu_cache_setting is False or self.device.type != "cuda":
            return
        
        # Estimate memory requirements
        bytes_needed = self._estimate_memory_requirement()
        budget = self._calculate_memory_budget()
        
        if bytes_needed > budget:
            self.logger.info(
                "GPU cache disabled for '%s': need %.2f GB > budget %.2f GB",
                self.split_name, bytes_needed/1e9, budget/1e9
            )
            return
        
        # Load and normalize all data
        self.logger.info(
            "Loading '%s' data for GPU cache (%.1f GB)...",
            self.split_name, bytes_needed/1e9
        )
        
        try:
            all_inputs, all_targets = self._load_all_data()
            
            # Transfer to GPU with correct dtype
            self.gpu_cache = {
                "inputs": all_inputs.to(device=self.device, dtype=self.dtype, non_blocking=True),
                "targets": all_targets.to(device=self.device, dtype=self.dtype, non_blocking=True),
            }
            self.logger.info("GPU cache created for '%s'", self.split_name)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.warning("Failed to create GPU cache for '%s': %s", self.split_name, e)
            self.gpu_cache = None
            # Clear any partial allocations
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _estimate_memory_requirement(self) -> int:
        """Calculate bytes needed for GPU cache including intermediate buffers."""
        bytes_per_float = BYTES_PER_FLOAT.get(self.dtype, 4)
        
        # Total elements: inputs + targets
        total_elements = self.n_total_samples * (
            self.n_species + self.n_globals + self.M + self.n_targets * self.M
        )
        
        # Base memory for final tensors
        base_memory = int(total_elements * bytes_per_float)
        
        # Account for intermediate tensors during normalization
        intermediate_memory = int(self.n_total_samples * self.M * bytes_per_float)
        
        # Add buffer for concatenation and other operations (50% overhead)
        overhead = int(base_memory * 0.5)
        
        return base_memory + intermediate_memory + overhead

    def _calculate_memory_budget(self) -> int:
        """Calculate available GPU memory budget."""
        try:
            if self.device.index is None:
                idx = torch.cuda.current_device()
            else:
                idx = self.device.index
            
            # Get actual free memory
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
            
            tcfg = self.config.get("training", {})
            max_frac = float(tcfg.get("gpu_cache_max_fraction", DEFAULT_GPU_CACHE_FRACTION))
            reserve_bytes = int(tcfg.get("gpu_cache_reserved_bytes", GPU_MEMORY_RESERVE_BYTES))
            overhead = float(tcfg.get("gpu_cache_overhead_factor", GPU_MEMORY_OVERHEAD_FACTOR))
            
            # Calculate budget from free memory
            budget = max(0, int(free_mem * max_frac) - reserve_bytes)
            return int(budget / overhead)
            
        except Exception as e:
            self.logger.warning("Could not assess GPU memory: %s", e)
            return 0

    def _load_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize all data for GPU caching."""
        if not self.norm_helper:
            raise RuntimeError("Cannot load data without normalization helper")
            
        all_x0_norm, all_g_norm, all_t_norm, all_y_norm = [], [], [], []
        
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
                
            with np.load(shard_path, allow_pickle=False) as data:
                # Load raw data (already log-transformed for species)
                x0_log = torch.from_numpy(data["x0_log"].astype(self.np_dtype))
                g_vec = torch.from_numpy(data["globals"].astype(self.np_dtype))
                t_vec_np = data["t_vec"]
                y_log = torch.from_numpy(data["y_mat"].astype(self.np_dtype))
                
                # Normalize time directly to tensor (avoiding numpy round-trip)
                t_norm = self._normalize_time_tensor(t_vec_np)
                
                # Apply normalization (these return tensors on self.device)
                x0_n = self.norm_helper.normalize(x0_log, self.species_vars)
                g_n = self.norm_helper.normalize(g_vec, self.global_vars)
                y_n = self.norm_helper.normalize(y_log, self.target_vars)
                
                all_x0_norm.append(x0_n)
                all_g_norm.append(g_n)
                all_t_norm.append(t_norm)
                all_y_norm.append(y_n)
        
        # Concatenate all data (all tensors are on the same device now)
        x0_full = torch.cat(all_x0_norm, dim=0).to(self.dtype)
        g_full = torch.cat(all_g_norm, dim=0).to(self.dtype)
        t_full = torch.cat(all_t_norm, dim=0).to(self.dtype)
        y_full = torch.cat(all_y_norm, dim=0).to(self.dtype)
        
        # Combine inputs: [x0, globals, times]
        inputs_full = torch.cat([x0_full, g_full, t_full], dim=1)
        
        return inputs_full, y_full

    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single trajectory from disk with normalization."""
        if self._closed:
            raise RuntimeError("Cannot load data from closed dataset")
            
        # Find shard containing this index
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]
        shard_info = self.shards[shard_idx]
        
        # Disable caching when using multiple workers
        use_cache = not self._in_worker_process
        
        if use_cache and self._shard_cache["name"] == shard_info["filename"]:
            raw = self._shard_cache["data"]
        else:
            # Load data from disk
            shard_path = self.split_dir / shard_info["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
                
            with np.load(shard_path, allow_pickle=False) as data:
                raw = dict(data)
                
                # Cache if safe to do so
                if use_cache:
                    # Clear old cache before loading new
                    if self._shard_cache["data"] is not None:
                        del self._shard_cache["data"]
                        
                    self._shard_cache = {
                        "name": shard_info["filename"],
                        "data": raw
                    }
        
        # Extract trajectory data
        x0_log_np = raw["x0_log"][local_idx].astype(self.np_dtype, copy=False)
        g_np = raw["globals"][local_idx].astype(self.np_dtype, copy=False)
        t_np = raw["t_vec"][local_idx].astype(self.np_dtype, copy=False)
        y_log_np = raw["y_mat"][local_idx].astype(self.np_dtype, copy=False)
        
        # Convert to tensors
        x0_log = torch.from_numpy(x0_log_np)
        g_vec = torch.from_numpy(g_np)
        y_log = torch.from_numpy(y_log_np)
        
        # Normalize time directly to tensor
        t_norm = self._normalize_time_tensor(t_np)
        
        # Apply normalization (returns tensors on self.device)
        x0_n = self.norm_helper.normalize(x0_log.unsqueeze(0), self.species_vars).squeeze(0)
        g_n = self.norm_helper.normalize(g_vec.unsqueeze(0), self.global_vars).squeeze(0)
        y_n = self.norm_helper.normalize(y_log, self.target_vars)
        
        # Combine inputs (all tensors are on self.device now)
        inputs_tensor = torch.cat([x0_n, g_n, t_norm], dim=0).to(self.dtype)
        targets_tensor = y_n.to(self.dtype)
        
        return inputs_tensor, targets_tensor

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single trajectory by index."""
        if not 0 <= idx < self.n_total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")
        
        if self.gpu_cache is not None:
            return self.gpu_cache["inputs"][idx], self.gpu_cache["targets"][idx]
        else:
            return self._load_from_disk(idx)

def create_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    shuffle: bool = True,
    device: Optional[torch.device] = None,
    drop_last: bool = True,
    **kwargs
) -> Optional[DataLoader]:
    """
    Create optimized DataLoader for sequence dataset.
    
    Automatically configures based on GPU caching status to minimize
    data transfer overhead and maximize throughput.
    
    Args:
        dataset: The dataset to load from
        config: Configuration dictionary containing training parameters
        shuffle: Whether to shuffle data each epoch
        device: Target device for pin_memory optimization
        drop_last: Whether to drop incomplete last batch
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured DataLoader or None if dataset is empty
    """
    log = logging.getLogger(__name__)
    
    if dataset is None or len(dataset) == 0:
        log.warning("Cannot create DataLoader for empty dataset")
        return None
    
    tcfg = config["training"]
    batch_size = tcfg["batch_size"]
    
    log.info(
        "DataLoader[%s]: batch_size=%d, samples=%d",
        dataset.split_name, batch_size, len(dataset)
    )
    
    # GPU-cached data: simple batching without workers
    if hasattr(dataset, 'gpu_cache') and dataset.gpu_cache is not None:
        if kwargs.get("num_workers", 0) != 0:
            raise ValueError("GPU-cached dataset must use num_workers=0")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=drop_last,
            **kwargs
        )
    
    # CPU loading: use workers for parallel data loading
    num_workers = tcfg.get("num_workers", 0)
    
    # Mark dataset for worker process handling
    if hasattr(dataset, '_in_worker_process') and num_workers > 0:
        dataset._in_worker_process = True
        log.info("Shard caching disabled for multi-worker DataLoader")
    
    # Configure DataLoader parameters
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda" and num_workers > 0),
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        **kwargs
    )
    
    # Add prefetch factor if specified
    if num_workers > 0 and "prefetch_factor" in tcfg:
        loader_kwargs["prefetch_factor"] = tcfg["prefetch_factor"]
    
    return DataLoader(**loader_kwargs)