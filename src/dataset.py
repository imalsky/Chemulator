#!/usr/bin/env python3
"""
dataset.py - Data loading for chemical kinetics.

"""
from __future__ import annotations

import h5py
import logging
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from normalizer import DataNormalizer

TIME_KEY = "t_time"
DTYPE = torch.float32
UNLIMITED_CACHE = -1
MIN_TIME_STEPS = 2

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU (Least Recently Used) cache for raw data profiles."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: Dict[str, Any]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class ChemicalDataset(TorchDataset):
    """
    Optimized PyTorch Dataset for chemical reaction profiles.
    Creates training samples from HDF5 with efficient normalization.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: List[int],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        *,
        cache_size: int = UNLIMITED_CACHE,
        batch_read_profiles: int = 10,  # Read multiple profiles at once
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.indices = indices
        # PERFORMANCE FIX: Create a mapping from profile index to its list position for O(1) lookup.
        self.profile_idx_to_list_pos = {p_idx: i for i, p_idx in enumerate(self.indices)}
        self.h5_file_handle: Optional[h5py.File] = None
        self.batch_read_profiles = batch_read_profiles

        self.norm_metadata = normalization_metadata
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        # Define variable order
        self.input_var_order = self.species_vars + self.global_vars + [TIME_KEY]
        self.target_var_order = self.species_vars
        
        self.all_vars = set(self.input_var_order)

        # Validate and build index
        self.num_time_steps: Optional[int] = None
        self.flat_index: List[Tuple[int, int]] = []
        self._validate_and_build_flat_index()

        # Setup cache
        self.cache_capacity = len(self.indices) if cache_size == UNLIMITED_CACHE else cache_size
        self.profile_cache = LRUCache(self.cache_capacity) if self.cache_capacity > 0 else None

        # Pre-compute normalization tensors
        self._setup_vectorized_normalization()

        logger.info(
            f"Dataset for {len(self.indices)} profiles initialized, creating "
            f"{len(self)} on-the-fly samples. Cache: {self.cache_capacity}."
        )

    def _setup_vectorized_normalization(self) -> None:
        """Pre-compute normalization parameters for efficient operations."""
        # Create normalization tensors for all variables at once
        self.norm_params = {"input": {}, "target": {}}
        
        for vector_type, var_order in [("input", self.input_var_order), ("target", self.target_var_order)]:
            n_vars = len(var_order)
            
            # Initialize default values
            means = torch.zeros(n_vars, dtype=DTYPE)
            stds = torch.ones(n_vars, dtype=DTYPE)
            log_means = torch.zeros(n_vars, dtype=DTYPE)
            log_stds = torch.ones(n_vars, dtype=DTYPE)
            mins = torch.zeros(n_vars, dtype=DTYPE)
            maxs = torch.ones(n_vars, dtype=DTYPE)
            
            # Collect normalization methods
            methods = []
            
            for i, var in enumerate(var_order):
                method = self.norm_metadata["normalization_methods"].get(var, "none")
                methods.append(method)
                
                stats = self.norm_metadata["per_key_stats"].get(var, {})
                
                if method == "standard":
                    means[i] = stats.get("mean", 0.0)
                    stds[i] = stats.get("std", 1.0)
                elif method == "log-standard":
                    log_means[i] = stats.get("log_mean", 0.0)
                    log_stds[i] = stats.get("log_std", 1.0)
                elif method == "log-min-max":
                    mins[i] = stats.get("min", 0.0)
                    maxs[i] = stats.get("max", 1.0)
            
            self.norm_params[vector_type] = {
                "methods": methods,
                "means": means,
                "stds": stds,
                "log_means": log_means,
                "log_stds": log_stds,
                "mins": mins,
                "maxs": maxs
            }

    def _get_h5_handle(self) -> h5py.File:
        """Opens an HDF5 file handle, specific to each DataLoader worker."""
        if self.h5_file_handle is None:
            self.h5_file_handle = h5py.File(self.h5_path, 'r', swmr=True)
        return self.h5_file_handle

    def _validate_and_build_flat_index(self) -> None:
        """Validates timesteps and builds flat index."""
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            num_time_steps = hf[TIME_KEY].shape[1]
            
            if num_time_steps < MIN_TIME_STEPS:
                raise ValueError(f"Profiles must have at least {MIN_TIME_STEPS} time steps, got {num_time_steps}.")
            
            self.num_time_steps = num_time_steps

            # Build flat index for all samples
            for h5_profile_idx in self.indices:
                for time_step_idx in range(1, self.num_time_steps):
                    self.flat_index.append((h5_profile_idx, time_step_idx))

        if not self.flat_index:
            raise ValueError("Dataset created with zero valid samples.")
        
        logger.info(f"All profiles validated with {self.num_time_steps} timesteps each.")

    def _read_profiles_batch(self, profile_indices: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
        """Batch read multiple profiles from HDF5 for efficiency."""
        h5_file = self._get_h5_handle()
        batch_data = {}
        
        # Read all variables for the batch of profiles at once
        for var in self.all_vars:
            if var not in h5_file:
                logger.warning(f"Variable '{var}' not found in HDF5 file.")
                continue
            
            # Read batch of profiles for this variable
            var_data = h5_file[var][profile_indices]
            
            # Store in individual profile dictionaries
            for i, profile_idx in enumerate(profile_indices):
                if profile_idx not in batch_data:
                    batch_data[profile_idx] = {}
                batch_data[profile_idx][var] = var_data[i]
        
        return batch_data

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Optimized sample loading with batch reading and fast normalization."""
        if not (0 <= idx < len(self.flat_index)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        profile_idx, query_time_idx = self.flat_index[idx]

        # Check cache first
        profile = self.profile_cache.get(profile_idx) if self.profile_cache else None
        
        if profile is None:
            # Determine batch of profiles to read
            # PERFORMANCE FIX: Use O(1) dictionary lookup instead of O(n) list.index() search.
            flat_idx_in_indices = self.profile_idx_to_list_pos[profile_idx]
            batch_start = (flat_idx_in_indices // self.batch_read_profiles) * self.batch_read_profiles
            batch_end = min(batch_start + self.batch_read_profiles, len(self.indices))
            batch_indices = self.indices[batch_start:batch_end]
            
            # Batch read profiles
            batch_profiles = self._read_profiles_batch(batch_indices)
            
            # Cache all profiles in the batch
            if self.profile_cache is not None:
                for p_idx, p_data in batch_profiles.items():
                    self.profile_cache.put(p_idx, p_data)
            
            profile = batch_profiles[profile_idx]

        # Build raw vectors
        raw_input = self._build_raw_input(profile, query_time_idx)
        raw_target = self._build_raw_target(profile, query_time_idx)

        # Fast vectorized normalization
        input_vector = self._normalize_vector_optimized(raw_input, "input")
        target_vector = self._normalize_vector_optimized(raw_target, "target")
        
        return input_vector, target_vector

    def _normalize_vector_optimized(self, raw_vector: np.ndarray, vector_type: str) -> Tensor:
        """Highly optimized vectorized normalization."""
        tensor = torch.from_numpy(raw_vector).to(dtype=DTYPE)
        params = self.norm_params[vector_type]
        
        # Create normalized tensor
        normalized = torch.zeros_like(tensor)
        
        # Process each normalization method
        for i, method in enumerate(params["methods"]):
            if method == "none":
                normalized[i] = tensor[i]
            elif method == "standard":
                normalized[i] = (tensor[i] - params["means"][i]) / params["stds"][i]
            elif method == "log-standard":
                # Use 1e-40 as minimum to avoid log(0)
                log_val = torch.log10(torch.clamp(tensor[i], min=1e-40))
                normalized[i] = (log_val - params["log_means"][i]) / params["log_stds"][i]
            elif method == "log-min-max":
                log_val = torch.log10(torch.clamp(tensor[i], min=1e-40))
                range_val = params["maxs"][i] - params["mins"][i]
                normalized[i] = torch.clamp((log_val - params["mins"][i]) / range_val, 0.0, 1.0)
            else:
                # Fallback for complex methods
                var_name = self.input_var_order[i] if vector_type == "input" else self.target_var_order[i]
                stats = self.norm_metadata["per_key_stats"].get(var_name)
                if stats:
                    normalized[i] = DataNormalizer.normalize_tensor(
                        tensor[i].unsqueeze(0), method, stats
                    ).squeeze(0)
                else:
                    normalized[i] = tensor[i]
        
        return normalized

    def _build_raw_input(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Constructs the raw input vector."""
        input_data = []
        
        # Initial species (t=0)
        for var in self.species_vars:
            input_data.append(profile.get(var, np.array([0.0]))[0])
    
        # Global variables
        for var in self.global_vars:
            raw = profile.get(var, 0.0)
            input_data.append(raw.item() if hasattr(raw, 'item') and np.ndim(raw) == 0 else raw)

        # Query time
        input_data.append(profile.get(TIME_KEY, np.array([0.0]))[query_time_idx])
        
        return np.array(input_data, dtype=np.float32)

    def _build_raw_target(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Constructs the raw target vector."""
        target_data = [profile.get(var, np.array([0.0]))[query_time_idx] for var in self.species_vars]
        return np.array(target_data, dtype=np.float32)


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """Optimized collate function without conservation loss."""
    valid_batch = [b for b in batch if b is not None]
    if not valid_batch:
        raise RuntimeError("Collate function received an empty batch.")

    input_vectors, target_vectors = zip(*valid_batch)
    
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)

    return model_inputs, target_batch


__all__ = ["ChemicalDataset", "collate_fn"]