#!/usr/bin/env python3
"""
dataset.py - Data loading and on-the-fly normalization for chemical kinetics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from normalizer import DataNormalizer

TIME_KEY = "t_time"
UTF8_ENCODING = "utf-8-sig"
DTYPE = torch.float32
UNLIMITED_CACHE = -1
MIN_TIME_STEPS = 2

logger = logging.getLogger(__name__)


class ChemicalDataset(TorchDataset):
    """
    A PyTorch Dataset that loads raw chemical reaction profiles and creates
    fixed-vector training samples, normalizing them on-the-fly.
    """

    def __init__(
        self,
        data_folder: Union[str, Path],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        *,
        profile_paths: Optional[List[Path]] = None,
        cache_size: int = UNLIMITED_CACHE,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_folder)
        self._validate_data_directory()

        self.norm_metadata = normalization_metadata
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)

        self.flat_index: List[Tuple[int, int]] = []
        self.profile_data_cache: Dict[int, Dict[str, Any]] = {}

        self.profile_paths = self._get_profile_paths(profile_paths)
        if self.profile_paths:
            self._load_profiles_and_build_index(cache_size)
            logger.info(
                f"Dataset initialized with {len(self.profile_data_cache)} cached raw profiles, "
                f"creating {len(self)} on-the-fly normalized training samples."
            )
        else:
            logger.warning("Dataset initialized with zero profiles.")

    def _validate_data_directory(self) -> None:
        """Validates that the raw data directory exists."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Raw data directory not found: {self.data_dir}")

    def _get_profile_paths(self, profile_paths: Optional[List[Path]]) -> List[Path]:
        """Gets the list of profile files to use, either from the provided list or by discovery."""
        if profile_paths is not None:
            # CORRECTED: Handle empty list gracefully.
            if not profile_paths:
                logger.warning("Empty 'profile_paths' list provided. No profiles will be loaded for this dataset split.")
                return []
            return profile_paths
        
        discovered_paths = sorted(p for p in self.data_dir.glob("*.json"))
        if not discovered_paths:
            raise FileNotFoundError(f"No profile files ('*.json') found in {self.data_dir}")
        return discovered_paths

    def _load_profiles_and_build_index(self, cache_size: int) -> None:
        """Loads raw profiles into cache and builds the flat index for all samples."""
        profiles_to_load = self.profile_paths
        if cache_size != UNLIMITED_CACHE:
            num_to_load = max(0, min(cache_size, len(self.profile_paths)))
            profiles_to_load = self.profile_paths[:num_to_load]
        
        logger.info(f"Attempting to load {len(profiles_to_load)} raw profiles into memory...")

        loaded_count = 0
        for profile_idx, path in enumerate(profiles_to_load):
            try:
                profile_data = json.loads(path.read_text(encoding=UTF8_ENCODING))
                self._validate_profile_data(profile_data, path)
                self.profile_data_cache[profile_idx] = profile_data
                self._add_profile_to_index(profile_idx, profile_data)
                loaded_count += 1
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Skipping profile {path.name} due to error: {e}")

        if not self.flat_index:
            raise ValueError("Dataset created with zero valid samples. Check profile data and config.")
        logger.info(f"Successfully loaded {loaded_count} profiles.")

    def _validate_profile_data(self, profile_data: Dict[str, Any], path: Path) -> None:
        """Validates that a raw profile contains required variables and has a consistent structure."""
        all_vars_to_check = self.species_vars + self.global_vars + [TIME_KEY]
        missing_vars = [v for v in all_vars_to_check if v not in profile_data]
        if missing_vars:
            raise ValueError(f"Profile {path.name} is missing required variables: {missing_vars}")

        time_data = profile_data[TIME_KEY]
        if not isinstance(time_data, list) or len(time_data) < MIN_TIME_STEPS:
            raise ValueError(f"Profile {path.name} has < {MIN_TIME_STEPS} time points.")
        
        num_time_steps = len(time_data)
        for var in self.species_vars:
            if not isinstance(profile_data[var], list) or len(profile_data[var]) != num_time_steps:
                raise ValueError(f"In profile {path.name}, variable '{var}' length mismatch. Expected {num_time_steps}, got {len(profile_data[var])}.")

    def _add_profile_to_index(self, profile_idx: int, profile_data: Dict[str, Any]) -> None:
        """Adds all valid time steps from a single profile to the flat dataset index."""
        for time_idx in range(1, len(profile_data[TIME_KEY])):
            self.flat_index.append((profile_idx, time_idx))

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Loads, normalizes, and returns a single training sample."""
        if not (0 <= idx < len(self.flat_index)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        profile_idx, query_time_idx = self.flat_index[idx]
        profile = self.profile_data_cache.get(profile_idx)
        if profile is None:
            raise RuntimeError(f"Raw profile with index {profile_idx} not found in cache.")

        input_vector = self._build_normalized_input(profile, query_time_idx)
        target_vector = self._build_normalized_target(profile, query_time_idx)
        return input_vector, target_vector

    def _normalize_value(self, raw_value: float, var_name: str) -> float:
        """Normalizes a single scalar value using the provided statistics."""
        tensor_val = torch.tensor([raw_value], dtype=DTYPE)
        normalized_tensor = DataNormalizer.normalize_tensor(
            tensor_val, self.norm_metadata["normalization_methods"][var_name],
            self.norm_metadata["per_key_stats"][var_name]
        )
        return normalized_tensor.item()

    def _build_normalized_input(self, profile: Dict[str, Any], query_time_idx: int) -> Tensor:
        """Constructs the complete normalized input vector for the model."""
        input_data = [self._normalize_value(profile[var][0], var) for var in self.species_vars]
        input_data.extend([self._normalize_value(profile[var], var) for var in self.global_vars])
        input_data.append(self._normalize_value(profile[TIME_KEY][query_time_idx], TIME_KEY))
        return torch.tensor(input_data, dtype=DTYPE)

    def _build_normalized_target(self, profile: Dict[str, Any], query_time_idx: int) -> Tensor:
        """Constructs the normalized target vector (species at query time)."""
        target_data = [self._normalize_value(profile[var][query_time_idx], var) for var in self.species_vars]
        return torch.tensor(target_data, dtype=DTYPE)

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """Collates a batch of (input, target) tensors into a format suitable for the model."""
    valid_batch = [b for b in batch if b is not None]
    if not valid_batch:
        return {"x": torch.empty(0)}, torch.empty(0)
    input_vectors, target_vectors = zip(*valid_batch)
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)
    return model_inputs, target_batch

__all__ = ["ChemicalDataset", "collate_fn"]