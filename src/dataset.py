#!/usr/bin/env python3
"""
dataset.py - Data loading and on-the-fly normalization for chemical kinetics.

FIXED ISSUES:
- Validates all profiles have same timesteps
- Efficient atom matrix device handling
- Only creates conservation tensors when needed
- Better error handling and logging
- Correctly handles tensor device placement (CPU in, GPU out via training loop)
- Simplified and corrected timestep validation logic
"""
from __future__ import annotations

import h5py
import logging
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    A PyTorch Dataset that loads chemical reaction profiles from an HDF5 file
    and creates fixed-vector training samples, normalizing them on-the-fly.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: List[int],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        atom_matrix: Optional[Tensor] = None,
        *,
        cache_size: int = UNLIMITED_CACHE,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.indices = indices
        self.h5_file_handle: Optional[h5py.File] = None
        self.device = torch.device('cpu') # Data is always loaded to CPU first

        self.norm_metadata = normalization_metadata
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        self.all_vars = set(self.species_vars + self.global_vars + [TIME_KEY])

        # For conservation loss - atom_matrix is passed on the correct training device
        self.use_conservation = atom_matrix is not None
        if self.use_conservation and atom_matrix is not None:
            self.atom_matrix = atom_matrix
        else:
            self.atom_matrix = None

        # Validate timesteps and build flat index
        self.num_time_steps: Optional[int] = None
        self.flat_index: List[Tuple[int, int]] = []
        self._validate_and_build_flat_index()

        self.cache_capacity = len(self.indices) if cache_size == UNLIMITED_CACHE else cache_size
        self.profile_cache = LRUCache(self.cache_capacity) if self.cache_capacity > 0 else None

        logger.info(
            f"Dataset for {len(self.indices)} profiles initialized, creating "
            f"{len(self)} on-the-fly samples. Cache: {self.cache_capacity}. "
            f"Conservation Loss: {self.use_conservation}."
        )

    def _get_h5_handle(self) -> h5py.File:
        """Opens an HDF5 file handle, specific to each DataLoader worker."""
        if self.h5_file_handle is None:
            self.h5_file_handle = h5py.File(self.h5_path, 'r', swmr=True)
        return self.h5_file_handle

    def _validate_and_build_flat_index(self) -> None:
        """Validates timesteps consistency and builds flat index."""
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            # The data preparation script guarantees all profiles have the same length.
            # We just need to read this value once.
            num_time_steps = hf[TIME_KEY].shape[1]
            
            if num_time_steps < MIN_TIME_STEPS:
                raise ValueError(f"Profiles must have at least {MIN_TIME_STEPS} time steps, got {num_time_steps}.")
            
            self.num_time_steps = num_time_steps

            # Build flat index
            for h5_profile_idx in self.indices:
                for time_step_idx in range(1, self.num_time_steps):
                    self.flat_index.append((h5_profile_idx, time_step_idx))

        if not self.flat_index:
            raise ValueError("Dataset created with zero valid samples. Check HDF5 data and indices.")
        
        logger.info(f"All profiles validated with {self.num_time_steps} timesteps each.")

    def _read_profile_from_h5(self, profile_idx: int) -> Dict[str, np.ndarray]:
        """Reads a full raw data profile for a given index from the HDF5 file."""
        h5_file = self._get_h5_handle()
        profile_data = {}
        
        for var in self.all_vars:
            if var not in h5_file:
                logger.warning(f"Variable '{var}' not found in HDF5 file, skipping.")
                continue
            profile_data[var] = h5_file[var][profile_idx]
        
        return profile_data

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Loads, normalizes, and returns a single training sample as CPU tensors."""
        if not (0 <= idx < len(self.flat_index)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        profile_idx, query_time_idx = self.flat_index[idx]

        # Get profile from cache or load it
        profile = self.profile_cache.get(profile_idx) if self.profile_cache else None
        if profile is None:
            profile = self._read_profile_from_h5(profile_idx)
            if self.profile_cache is not None:
                self.profile_cache.put(profile_idx, profile)

        # Build normalized input and target
        input_vector = self._build_normalized_input(profile, query_time_idx)
        target_vector = self._build_normalized_target(profile, query_time_idx)

        # Pre-calculate initial atoms for conservation loss
        if self.use_conservation and self.atom_matrix is not None:
            initial_species_raw = torch.tensor(
                [profile[var][0] for var in self.species_vars], 
                dtype=DTYPE, 
                device=self.device # CPU
            )
            # Matmul happens on CPU here. initial_species is on CPU, and atom_matrix is moved to CPU.
            # This is efficient enough as it's a small operation per sample.
            initial_atoms = torch.matmul(initial_species_raw, self.atom_matrix.to(self.device))
        else:
            # Return empty tensor
            initial_atoms = torch.zeros(0, dtype=DTYPE, device=self.device)
        
        return input_vector, target_vector, initial_atoms

    def _normalize_value(self, raw_value: Union[float, np.number], var_name: str) -> float:
        """Normalizes a single scalar value using the provided statistics."""
        tensor_val = torch.tensor([raw_value], dtype=DTYPE)
        normalized_tensor = DataNormalizer.normalize_tensor(
            tensor_val, 
            self.norm_metadata["normalization_methods"][var_name],
            self.norm_metadata["per_key_stats"][var_name]
        )
        return normalized_tensor.item()

    def _build_normalized_input(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> Tensor:
        """Constructs the complete normalized input vector for the model."""
        input_data = []
        
        # Initial state of species (at time t=0)
        for var in self.species_vars:
            if var in profile:
                input_data.append(self._normalize_value(profile[var][0], var))
            else:
                logger.warning(f"Species variable '{var}' not in profile, using 0.0")
                input_data.append(0.0)
        
        # Global variables (time-independent)
        for var in self.global_vars:
            if var in profile:
                raw = profile[var]
                raw_scalar = raw if np.ndim(raw) == 0 else raw.flat[0]
                input_data.append(self._normalize_value(raw_scalar, var))
            else:
                logger.warning(f"Global variable '{var}' not in profile, using 0.0")
                input_data.append(0.0)
        
        # Query time
        if TIME_KEY in profile:
            input_data.append(self._normalize_value(profile[TIME_KEY][query_time_idx], TIME_KEY))
        else:
            logger.warning(f"Time key '{TIME_KEY}' not in profile, using 0.0")
            input_data.append(0.0)
        
        return torch.tensor(input_data, dtype=DTYPE, device=self.device)

    def _build_normalized_target(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> Tensor:
        """Constructs the normalized target vector (species at query time)."""
        target_data = []
        for var in self.species_vars:
            if var in profile:
                target_data.append(self._normalize_value(profile[var][query_time_idx], var))
            else:
                logger.warning(f"Species variable '{var}' not in profile for target, using 0.0")
                target_data.append(0.0)
        return torch.tensor(target_data, dtype=DTYPE, device=self.device)


def collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Collates a batch of (input, target, initial_atoms) tensors."""
    valid_batch = [b for b in batch if b is not None]
    if not valid_batch:
        raise RuntimeError("Collate function received an empty or fully invalid batch.")

    input_vectors, target_vectors, initial_atoms_vectors = zip(*valid_batch)
    
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)
    
    # All items in a batch will either have atoms or not, based on dataset config.
    if initial_atoms_vectors[0].numel() > 0:
        initial_atoms_batch = torch.stack(initial_atoms_vectors, dim=0)
    else:
        # If they don't have atoms, return an empty tensor with the correct batch dim.
        batch_size = len(initial_atoms_vectors)
        initial_atoms_batch = torch.zeros(
            batch_size, 0, 
            dtype=DTYPE, device=initial_atoms_vectors[0].device
        )

    return model_inputs, target_batch, initial_atoms_batch


__all__ = ["ChemicalDataset", "collate_fn"]