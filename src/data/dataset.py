#!/usr/bin/env python3
"""
High-performance dataset implementation with sequence mode support for LiLaN.

This module provides efficient data loading with:
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


class SequenceDataset(Dataset):
    """
    Dataset for sequence-mode (trajectory-based) data with normalization.
    
    This dataset efficiently loads preprocessed trajectory data and applies
    config-driven normalization to both inputs and targets. Supports GPU
    caching for datasets that fit in memory.
    
    Attributes:
        shard_dir: Directory containing preprocessed data shards
        split_name: Dataset split ('train', 'validation', 'test')
        config: Configuration dictionary
        device: Target device for tensors
        norm_stats: Normalization statistics from preprocessing
    """
    
    def __init__(
        self,
        shard_dir: Path,
        split_name: str,
        config: Dict[str, Any],
        device: torch.device,
        norm_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            shard_dir: Directory containing preprocessed shards
            split_name: Split name ('train', 'validation', 'test')
            config: Configuration with data, normalization, and system settings
            device: Target device for tensors
            norm_stats: Optional normalization statistics
            
        Raises:
            FileNotFoundError: If shard index is missing
            ValueError: If data is not in sequence mode
        """
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.norm_stats = norm_stats or {}

        # Configure data types
        self._setup_dtypes()
        
        # Load and validate metadata
        self._load_metadata()
        
        # Extract configuration parameters
        self._setup_dimensions()
        
        # Validate time normalization consistency
        self._validate_time_normalization()
        
        # Build shard lookup table
        self._build_shard_lookup()
        
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
        self.dtype = getattr(torch, dtype_str)
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32

    def _load_metadata(self) -> None:
        """Load and validate shard metadata."""
        shard_index_path = self.shard_dir / "shard_index.json"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Shard index not found: {shard_index_path}")
            
        with open(shard_index_path) as f:
            self.shard_index = json.load(f)
            
        if not self.shard_index.get("sequence_mode", False):
            raise ValueError("Expected sequence mode data for SequenceDataset")
        
        self.split_info = self.shard_index["splits"][self.split_name]

    def _setup_dimensions(self) -> None:
        """Extract dimensions and variable lists from config."""
        data_config = self.config["data"]
        
        self.M = self.shard_index["M_per_sample"]
        self.species_vars = data_config["species_variables"]
        self.target_vars = data_config.get("target_species_variables", self.species_vars)
        self.global_vars = data_config["global_variables"]

        # Compute counts
        self.n_species = len(self.species_vars)
        self.n_targets = len(self.target_vars)
        self.n_globals = len(self.global_vars)
        
        # Verify counts against shard_index.json
        si = self.shard_index
        if (self.n_species != int(si["n_input_species"]) or
            self.n_targets != int(si["n_target_species"]) or
            self.n_globals != int(si["n_globals"])):
            raise ValueError(
                "Mismatch between config variable counts and shard_index.json "
                f"(species {self.n_species} vs {si['n_input_species']}, "
                f"targets {self.n_targets} vs {si['n_target_species']}, "
                f"globals {self.n_globals} vs {si['n_globals']})"
            )

    def _validate_time_normalization(self) -> None:
        """Check for consistency between saved and configured time normalization."""
        self.time_var = self.config["data"]["time_variable"]
        
        saved_methods = self.norm_stats.get("normalization_methods", {})
        self.saved_time_method = saved_methods.get(self.time_var)
        
        config_methods = self.config.get("normalization", {}).get("methods", {})
        self.config_time_method = config_methods.get(self.time_var, "log-min-max")
        
        if self.saved_time_method and self.saved_time_method != self.config_time_method:
            self.logger.warning(
                "Time normalization differs: stats=%s, config=%s. Using runtime normalization.",
                self.saved_time_method, self.config_time_method
            )

    def _build_shard_lookup(self) -> None:
        """Build efficient lookup arrays for shard access."""
        self.shards = self.split_info.get("shards", [])
        self.n_total_samples = self.split_info.get("n_trajectories", 0)
        
        if self.n_total_samples == 0:
            self.logger.warning(f"No samples found in '{self.split_name}' split.")
            return
        
        # Build cumulative sum for fast shard indexing
        cumsum = 0
        self.shard_starts = []
        
        for shard in self.shards:
            if "n_trajectories" not in shard:
                raise KeyError(f"Shard metadata missing n_trajectories: {shard}")
            self.shard_starts.append(cumsum)
            cumsum += int(shard["n_trajectories"])
        
        self.shard_starts = np.array(self.shard_starts, dtype=np.int64)
        self.n_total_samples = cumsum  # Use derived total to avoid mismatches

    def _setup_normalization(self) -> None:
        """Initialize normalization helper if stats are available."""
        if self.norm_stats and self.norm_stats.get("per_key_stats"):
            self.norm_helper = NormalizationHelper(
                self.norm_stats, 
                torch.device("cpu"),  # CPU for preprocessing
                self.config
            )
            self.logger.info(f"NormalizationHelper initialized for '{self.split_name}' dataset.")
        else:
            self.norm_helper = None
            self.logger.warning(f"No normalization stats found. '{self.split_name}' will use raw data.")

    def _normalize_time(self, t: np.ndarray) -> np.ndarray:
        """
        Normalize time values according to the configured method.

        - "time-norm" / "log-min-max": returns values in [0, 1]
        - "none": returns raw time unchanged

        Args:
            t: Raw time values

        Returns:
            Time values normalized per config (or raw if method is "none").
        """
        if self.norm_helper:
            # Use helper for consistent normalization
            t_tensor = torch.from_numpy(t.astype(self.np_dtype, copy=False))
            t_norm = self.norm_helper.normalize_time(t_tensor)
            return t_norm.cpu().numpy().astype(self.np_dtype, copy=False)
        
        # Legacy fallback if helper unavailable
        norm_cfg = self.config.get("normalization", {})
        min_scale = float(norm_cfg.get("min_std", 1e-12))
        time_norm = self.shard_index.get("time_normalization", {})
        time_method = self.config_time_method
        
        if time_method == "time-norm":
            # Paper's tau-space normalization
            tau0 = float(time_norm.get("tau0", 1.0))
            tmin = float(time_norm.get("tmin", 0.0))
            tmax = float(time_norm.get("tmax", 1.0))
            
            tau = np.log1p(t / tau0)
            denom = max(tmax - tmin, min_scale)
            z = (tau - tmin) / denom
            return np.clip(z, 0.0, 1.0)
        elif time_method == "log-min-max":
            # Log-space min-max normalization
            tmin_raw = float(time_norm.get("tmin_raw", 0.0))
            tmax_raw = float(time_norm.get("tmax_raw", 1.0))
            eps = float(norm_cfg.get("epsilon", 1e-30))
            
            lo = np.log10(max(tmin_raw, eps))
            hi = np.log10(max(tmax_raw, tmin_raw + eps))
            denom = max(hi - lo, min_scale)
            
            z = (np.log10(np.maximum(t, eps)) - lo) / denom
            return np.clip(z, 0.0, 1.0)
        elif time_method in ("none", None):
            # Pass-through
            return t.astype(self.np_dtype, copy=False)
        else:
            raise ValueError(f"Unknown time normalization method: {time_method}")

    def _try_gpu_cache(self) -> None:
        """
        Attempt to cache entire dataset on GPU if memory permits.
        
        Loads all data, applies normalization, and transfers to GPU
        for fast access during training.
        """
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
            self.logger.warning(
                f"GPU cache for '{self.split_name}' disabled: "
                f"need {bytes_needed/1e9:.2f} GB > budget {budget/1e9:.2f} GB"
            )
            return
        
        # Load and normalize all data
        self.logger.info(
            f"Loading '{self.split_name}' data for GPU cache ({bytes_needed/1e9:.1f} GB)..."
        )
        
        all_inputs, all_targets = self._load_all_data()
        
        # Transfer to GPU
        self.gpu_cache = {
            "inputs": all_inputs.to(device=self.device, dtype=self.dtype, non_blocking=True),
            "targets": all_targets.to(device=self.device, dtype=self.dtype, non_blocking=True),
        }
        self.logger.info(f"GPU cache for '{self.split_name}' created successfully.")

    def _estimate_memory_requirement(self) -> int:
        """Calculate bytes needed for GPU cache."""
        bytes_per_float = {
            torch.float64: 8,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(self.dtype, 4)
        
        # Total elements: inputs + targets
        total_elements = self.n_total_samples * (
            self.n_species + self.n_globals + self.M +  # inputs
            self.n_targets * self.M  # targets
        )
        
        return total_elements * bytes_per_float

    def _calculate_memory_budget(self) -> int:
        """Calculate available GPU memory budget."""
        try:
            idx = 0 if self.device.index is None else self.device.index
            total_mem = torch.cuda.get_device_properties(idx).total_memory
            
            tcfg = self.config.get("training", {})
            max_frac = float(tcfg.get("gpu_cache_max_fraction", 0.5))
            reserve_bytes = int(tcfg.get("gpu_cache_reserved_bytes", 3_000_000_000))  # 3GB
            overhead = float(tcfg.get("gpu_cache_overhead_factor", 1.15))  # +15%
            
            budget = max(0, int(total_mem * max_frac) - reserve_bytes)
            return int(budget / overhead)  # Account for overhead
            
        except Exception as e:
            self.logger.warning(f"Could not assess GPU memory: {e}")
            return 0

    def _load_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize all data for GPU caching."""
        all_x0_norm, all_g_norm, all_t_norm, all_y_norm = [], [], [], []
        
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            
            with np.load(shard_path, allow_pickle=False) as data:
                # Load raw data
                x0_log = torch.from_numpy(data["x0_log"].astype(self.np_dtype))
                g_vec = torch.from_numpy(data["globals"].astype(self.np_dtype))
                t_vec_np = data["t_vec"]
                y_log = torch.from_numpy(data["y_mat"].astype(self.np_dtype))
                
                # Normalize time
                t_norm_np = self._normalize_time(t_vec_np).astype(self.np_dtype)
                t_norm = torch.from_numpy(t_norm_np)
                
                # Apply normalization if available
                if self.norm_helper:
                    x0_n = self.norm_helper.normalize(x0_log, self.species_vars)
                    g_n = self.norm_helper.normalize(g_vec, self.global_vars)
                    y_n = self.norm_helper.normalize(y_log, self.target_vars)
                else:
                    x0_n, g_n, y_n = x0_log, g_vec, y_log
                
                all_x0_norm.append(x0_n)
                all_g_norm.append(g_n)
                all_t_norm.append(t_norm)
                all_y_norm.append(y_n)
        
        # Concatenate all data
        x0_full = torch.cat(all_x0_norm, dim=0).to(self.dtype)
        g_full = torch.cat(all_g_norm, dim=0).to(self.dtype)
        t_full = torch.cat(all_t_norm, dim=0).to(self.dtype)
        y_full = torch.cat(all_y_norm, dim=0).to(self.dtype)
        
        # Combine inputs: [x0, globals, times]
        inputs_full = torch.cat([x0_full, g_full, t_full], dim=1)
        
        return inputs_full, y_full

    def _load_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single trajectory from disk with normalization.
        
        Args:
            idx: Global index of the trajectory
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        # Find shard containing this index
        shard_idx = np.searchsorted(self.shard_starts, idx, side='right') - 1
        local_idx = idx - self.shard_starts[shard_idx]
        shard_info = self.shards[shard_idx]
        
        # Load shard data (with simple LRU cache)
        if self._shard_cache.get("name") != shard_info["filename"]:
            with np.load(self.split_dir / shard_info["filename"], allow_pickle=False) as data:
                self._shard_cache = {
                    "name": shard_info["filename"],
                    "data": dict(data)
                }
        
        raw = self._shard_cache["data"]
        
        # Extract trajectory data
        x0_log_np = raw["x0_log"][local_idx].astype(self.np_dtype, copy=False)
        g_np = raw["globals"][local_idx].astype(self.np_dtype, copy=False)
        t_np = raw["t_vec"][local_idx]
        y_log_np = raw["y_mat"][local_idx].astype(self.np_dtype, copy=False)
        
        # Convert to tensors
        x0_log = torch.from_numpy(x0_log_np)
        g_vec = torch.from_numpy(g_np)
        t_norm = torch.from_numpy(self._normalize_time(t_np).astype(self.np_dtype, copy=False))
        y_log = torch.from_numpy(y_log_np)
        
        # Apply normalization
        if self.norm_helper:
            x0_n = self.norm_helper.normalize(x0_log.unsqueeze(0), self.species_vars).squeeze(0)
            g_n = self.norm_helper.normalize(g_vec.unsqueeze(0), self.global_vars).squeeze(0)
            y_n = self.norm_helper.normalize(y_log, self.target_vars)
        else:
            x0_n, g_n, y_n = x0_log, g_vec, y_log
        
        # Combine inputs and convert to target dtype
        inputs_tensor = torch.cat([x0_n, g_n, t_norm], dim=0).to(self.dtype)
        targets_tensor = y_n.to(self.dtype)
        
        return inputs_tensor, targets_tensor

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.n_total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single trajectory by index.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Tuple of (inputs, targets) tensors
            
        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= idx < self.n_total_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_total_samples} samples")
        
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
    
    Automatically configures the DataLoader based on whether the dataset
    is GPU-cached or requires CPU loading with workers.
    
    Args:
        dataset: The dataset to load from
        config: Configuration dictionary
        shuffle: Whether to shuffle data
        device: Target device (used for pin_memory decision)
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
    
    log.info(f"DataLoader[{dataset.split_name}]: batch_size={batch_size}, samples={len(dataset)}")
    
    # GPU-cached data: simple batching, no workers needed
    if hasattr(dataset, 'gpu_cache') and dataset.gpu_cache is not None:
        # Guard: GPU-cached dataset must not use worker processes
        if kwargs.get("num_workers", 0) != 0:
            raise ValueError("GPU-cached dataset must use num_workers=0.")
        if getattr(dataset, "device", None) is not None and dataset.device.type != "cuda":
            raise RuntimeError("GPU-cached dataset present but dataset.device is not CUDA.")
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
    
    # Add prefetch factor if specified and using workers
    if num_workers > 0 and "prefetch_factor" in tcfg:
        loader_kwargs["prefetch_factor"] = tcfg["prefetch_factor"]
    
    return DataLoader(**loader_kwargs)