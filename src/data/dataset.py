#!/usr/bin/env python3
"""
Dataset implementation for chemical kinetics models.

Provides efficient data loading with GPU caching and normalization.
Loads raw data from preprocessed shards and applies transformations via NormalizationHelper.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.normalizer import NormalizationHelper


# Memory management constants
GPU_MEMORY_RESERVE_BYTES = 3_000_000_000  # 3GB
GPU_MEMORY_OVERHEAD_FACTOR = 1.15
DEFAULT_GPU_CACHE_FRACTION = 0.5
BYTES_PER_FLOAT = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2
}


class SequenceDataset(Dataset):
    """
    Dataset for sequence-mode trajectory data with normalization.
    
    Efficiently loads preprocessed data and applies normalization
    transformations based on configuration.
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
        Initialize dataset.
        
        Args:
            shard_dir: Directory containing preprocessed shards
            split_name: Split name ('train', 'validation', 'test')
            config: Configuration dictionary
            device: Target device for tensors
            norm_stats: Normalization statistics
            disable_cache: Disable shard caching for multi-worker loading
        """
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.norm_stats = norm_stats or {}
        self._disable_cache = disable_cache
        self._closed = False
        
        # Setup data type
        self._setup_dtype()
        
        # Load metadata
        self._load_metadata()
        
        # Setup dimensions
        self._setup_dimensions()
        
        # Build shard lookup
        self._build_shard_lookup()
        
        # Load normalization stats if not provided
        if not self.norm_stats or not self.norm_stats.get("per_key_stats"):
            norm_path = self.shard_dir / "normalization.json"
            if norm_path.exists():
                from utils.utils import load_json
                self.norm_stats = load_json(norm_path)
            else:
                raise ValueError(f"Normalization statistics required but not found at {norm_path}")
        
        # Initialize normalization helper
        self.norm_helper = NormalizationHelper(self.norm_stats, self.device, self.config)
        
        # Initialize caching
        self.gpu_cache = None
        self._shard_cache = {"name": None, "data": None}
        
        # Try GPU caching if possible
        self._try_gpu_cache()
    
    def _setup_dtype(self) -> None:
        """Configure data types."""
        dtype_str = self.config["system"].get("dtype", "float32")
        try:
            self.dtype = getattr(torch, dtype_str)
        except AttributeError:
            self.logger.warning(f"Unknown dtype '{dtype_str}', using float32")
            self.dtype = torch.float32
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32
    
    def _load_metadata(self) -> None:
        """Load and validate shard metadata."""
        from utils.utils import load_json
        
        shard_index_path = self.shard_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
        
        self.shard_index = load_json(shard_index_path)
        
        if not self.shard_index.get("sequence_mode", False):
            raise ValueError("Expected sequence mode data")
        
        if self.shard_index.get("variable_length", False):
            raise ValueError("Variable-length sequences not supported")
        
        self.split_info = self.shard_index["splits"][self.split_name]
        self.M = int(self.shard_index.get("M_per_sample", 0))
        
        if self.M <= 0:
            raise ValueError(f"Invalid M_per_sample: {self.M}")
    
    def _setup_dimensions(self) -> None:
        """Extract dimensions from configuration."""
        data_cfg = self.config["data"]
        
        self.species_vars = data_cfg["species_variables"]
        self.target_vars = data_cfg.get("target_species_variables", self.species_vars)
        self.global_vars = data_cfg["global_variables"]
        self.time_var = data_cfg["time_variable"]
        
        self.n_species = len(self.species_vars)
        self.n_targets = len(self.target_vars)
        self.n_globals = len(self.global_vars)
        
        # Verify against metadata
        si = self.shard_index
        if (self.n_species != si["n_input_species"] or
            self.n_targets != si["n_target_species"] or
            self.n_globals != si["n_globals"]):
            raise ValueError("Variable count mismatch with shard_index.json")
    
    def _build_shard_lookup(self) -> None:
        """Build lookup arrays for efficient shard access."""
        self.shards = self.split_info.get("shards", [])
        self.n_total_samples = self.split_info.get("n_trajectories", 0)
        
        if self.n_total_samples == 0:
            self.logger.warning(f"No samples found in '{self.split_name}' split")
            return
        
        # Build cumulative sum for indexing
        cumsum = 0
        self.shard_starts = []
        
        for shard in self.shards:
            if "n_trajectories" not in shard:
                raise KeyError(f"Shard missing n_trajectories: {shard}")
            self.shard_starts.append(cumsum)
            cumsum += int(shard["n_trajectories"])
        
        self.shard_starts = np.array(self.shard_starts, dtype=np.int64)
        self.n_total_samples = cumsum
    
    def _try_gpu_cache(self) -> None:
        """Attempt to cache entire dataset on GPU."""
        if self.n_total_samples == 0:
            return
        
        # Check configuration
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
        
        # Load all data
        self.logger.info(
            "Loading '%s' data for GPU cache (%.1f GB)...",
            self.split_name, bytes_needed/1e9
        )
        
        try:
            all_inputs, all_targets = self._load_all_data()
            
            # Transfer to GPU
            self.gpu_cache = {
                "inputs": all_inputs.to(device=self.device, dtype=self.dtype, non_blocking=True),
                "targets": all_targets.to(device=self.device, dtype=self.dtype, non_blocking=True)
            }
            self.logger.info("GPU cache created for '%s'", self.split_name)
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.warning("Failed to create GPU cache: %s", e)
            self.gpu_cache = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
    
    def _estimate_memory_requirement(self) -> int:
        """Calculate bytes needed for GPU cache."""
        bytes_per_float = BYTES_PER_FLOAT.get(self.dtype, 4)
        
        # Total elements
        total_elements = self.n_total_samples * (
            self.n_species + self.n_globals + self.M + self.n_targets * self.M
        )
        
        # Base memory plus overhead
        base_memory = int(total_elements * bytes_per_float)
        intermediate_memory = int(self.n_total_samples * self.M * bytes_per_float)
        overhead = int(base_memory * 0.5)
        
        return base_memory + intermediate_memory + overhead
    
    def _calculate_memory_budget(self) -> int:
        """Calculate available GPU memory budget."""
        try:
            if self.device.index is None:
                idx = torch.cuda.current_device()
            else:
                idx = self.device.index
            
            free_mem, _ = torch.cuda.mem_get_info(idx)
            
            tcfg = self.config.get("training", {})
            max_frac = float(tcfg.get("gpu_cache_max_fraction", DEFAULT_GPU_CACHE_FRACTION))
            reserve = int(tcfg.get("gpu_cache_reserved_bytes", GPU_MEMORY_RESERVE_BYTES))
            overhead = float(tcfg.get("gpu_cache_overhead_factor", GPU_MEMORY_OVERHEAD_FACTOR))
            
            budget = max(0, int(free_mem * max_frac) - reserve)
            return int(budget / overhead)
            
        except Exception as e:
            self.logger.warning("Could not assess GPU memory: %s", e)
            return 0
    
    def _load_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize all data for GPU caching."""
        all_inputs = []
        all_targets = []
        
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            
            with np.load(shard_path, allow_pickle=False) as data:
                # Load raw data
                x0 = torch.from_numpy(data["x0"].astype(self.np_dtype))
                g_vec = torch.from_numpy(data["globals"].astype(self.np_dtype))
                t_vec = torch.from_numpy(data["t_vec"].astype(self.np_dtype))
                y_mat = torch.from_numpy(data["y_mat"].astype(self.np_dtype))
                
                # Apply normalization
                x0_norm = self.norm_helper.normalize(x0, self.species_vars)
                g_norm = self.norm_helper.normalize(g_vec, self.global_vars)
                
                # Normalize time (handle batched data)
                batch_size = t_vec.shape[0]
                t_flat = t_vec.reshape(-1, 1)
                t_norm_flat = self.norm_helper.normalize(t_flat, [self.time_var])
                t_norm = t_norm_flat.reshape(batch_size, self.M)
                
                # Normalize targets (handle batched data)
                y_mat_flat = y_mat.reshape(-1, self.n_targets)
                y_norm_flat = self.norm_helper.normalize(y_mat_flat, self.target_vars)
                y_norm = y_norm_flat.reshape(batch_size, self.M, self.n_targets)
                
                # Combine inputs: [x0, globals, times]
                inputs = torch.cat([x0_norm, g_norm, t_norm], dim=1)
                
                all_inputs.append(inputs)
                all_targets.append(y_norm)
        
        # Concatenate all data
        return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)
    
    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load single trajectory from disk with normalization."""
        if self._closed:
            raise RuntimeError("Cannot load data from closed dataset")
        
        # Find shard containing this index
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]
        shard_info = self.shards[shard_idx]
        
        # Check cache
        use_cache = not self._disable_cache
        
        if use_cache and self._shard_cache["name"] == shard_info["filename"]:
            raw = self._shard_cache["data"]
        else:
            # Load from disk
            shard_path = self.split_dir / shard_info["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            
            with np.load(shard_path, allow_pickle=False) as data:
                raw = dict(data)
                
                if use_cache:
                    # Update cache
                    if self._shard_cache["data"] is not None:
                        del self._shard_cache["data"]
                    
                    self._shard_cache = {
                        "name": shard_info["filename"],
                        "data": raw
                    }
        
        # Extract trajectory (raw data)
        x0 = torch.from_numpy(raw["x0"][local_idx].astype(self.np_dtype))
        g_vec = torch.from_numpy(raw["globals"][local_idx].astype(self.np_dtype))
        t_vec = torch.from_numpy(raw["t_vec"][local_idx].astype(self.np_dtype))
        y_mat = torch.from_numpy(raw["y_mat"][local_idx].astype(self.np_dtype))
        
        # Apply normalization (includes log10 transformation where needed)
        x0_norm = self.norm_helper.normalize(x0.unsqueeze(0), self.species_vars).squeeze(0)
        g_norm = self.norm_helper.normalize(g_vec.unsqueeze(0), self.global_vars).squeeze(0)
        
        # Normalize time
        t_norm = self.norm_helper.normalize(t_vec.unsqueeze(1), [self.time_var]).squeeze(1)
        
        # Normalize targets
        y_norm = self.norm_helper.normalize(y_mat, self.target_vars)
        
        # Combine inputs: [x0, globals, times]
        inputs_tensor = torch.cat([x0_norm, g_norm, t_norm], dim=0).to(self.dtype)
        targets_tensor = y_norm.to(self.dtype)
        
        return inputs_tensor, targets_tensor
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single trajectory by index."""
        if not 0 <= idx < self.n_total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_total_samples})")
        
        if self.gpu_cache is not None:
            return self.gpu_cache["inputs"][idx], self.gpu_cache["targets"][idx]
        else:
            return self._load_from_disk(idx)
    
    def close(self) -> None:
        """Clean up resources."""
        self._closed = True
        self._shard_cache = {"name": None, "data": None}
        self.gpu_cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def create_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    shuffle: bool = True,
    device: Optional[torch.device] = None,
    drop_last: bool = True,
    **kwargs
) -> Optional[DataLoader]:
    """
    Create optimized DataLoader.
    
    Args:
        dataset: Dataset to load from
        config: Configuration dictionary
        shuffle: Whether to shuffle data
        device: Target device for pin_memory
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
    
    # GPU-cached data: no workers needed
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
    
    # CPU loading: use workers
    num_workers = tcfg.get("num_workers", 0)
    
    # Mark dataset for worker process handling
    if hasattr(dataset, '_disable_cache') and num_workers > 0:
        dataset._disable_cache = True
        log.info("Shard caching disabled for multi-worker DataLoader")
    
    # Configure DataLoader
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