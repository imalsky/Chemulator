#!/usr/bin/env python3
"""
dataset.py - Data loading functionality
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

# Constants
NORMALIZATION_METADATA_FILE = "normalization_metadata.json"
TIME_KEY = "t_time"
UTF8_ENCODING = "utf-8"
DTYPE = torch.float32
UNLIMITED_CACHE = -1
MIN_TIME_STEPS = 2

logger = logging.getLogger(__name__)


class ChemicalDataset(TorchDataset):
    """
    A PyTorch Dataset that loads pre-normalized chemical reaction profiles.
    
    Args:
        data_folder: Directory containing normalized profile JSON files
        species_variables: List of species variable names to use
        global_variables: List of global variable names to use
        profile_paths: Optional specific list of profile files to use
        cache_size: Number of profiles to load into memory (-1 for all)
    
    The dataset creates training samples by using:
    - Input: [initial_species_values, global_conditions, query_time]
    - Target: [species_values_at_query_time]
    """
    
    def __init__(
        self,
        data_folder: Union[str, Path],
        species_variables: List[str],
        global_variables: List[str],
        *,
        profile_paths: Optional[List[Path]] = None,
        cache_size: int = UNLIMITED_CACHE,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_folder)
        self._validate_data_directory()
        
        # Store variable names (sorted for consistency)
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        # Determine which profile files to use
        self.profile_paths = self._get_profile_paths(profile_paths)
        
        # Initialize data structures
        self.flat_index: List[Tuple[int, int]] = []
        self.profile_data_cache: Dict[int, Dict[str, Any]] = {}
        
        # Load profiles and build training sample index
        self._load_profiles_and_build_index(cache_size)
        
        logger.info(
            f"Dataset initialized with {len(self.profile_data_cache)} cached profiles, "
            f"creating {len(self)} training samples."
        )

    def _validate_data_directory(self) -> None:
        """Validates that the data directory exists."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Normalized data directory not found: {self.data_dir}"
            )

    def _get_profile_paths(self, profile_paths: Optional[List[Path]]) -> List[Path]:
        """Gets the list of profile files to use."""
        if profile_paths is None:
            # Discover all JSON files except metadata
            discovered_paths = sorted([
                p for p in self.data_dir.glob("*.json") 
                if p.name != NORMALIZATION_METADATA_FILE
            ])
            if not discovered_paths:
                raise FileNotFoundError(
                    f"No profile files found in {self.data_dir}"
                )
            return discovered_paths
        else:
            if not profile_paths:
                raise ValueError("Empty profile_paths list provided")
            return profile_paths

    def _load_profiles_and_build_index(self, cache_size: int) -> None:
        """Loads profiles into cache and builds the flat index for training samples."""
        # Determine how many profiles to load
        if cache_size == UNLIMITED_CACHE:
            profiles_to_load = self.profile_paths
        else:
            cache_size = max(1, min(cache_size, len(self.profile_paths)))
            profiles_to_load = self.profile_paths[:cache_size]
        
        logger.info(f"Loading {len(profiles_to_load)} profiles into memory...")
        
        for profile_idx, path in enumerate(profiles_to_load):
            try:
                profile_data = self._load_single_profile(path)
                self._validate_profile_data(profile_data, path)
                
                # Use the original profile_idx for the cache key
                self.profile_data_cache[profile_idx] = profile_data
                
                # Build index for this profile using the original profile_idx
                self._add_profile_to_index(profile_idx, profile_data)
                
            except Exception as e:
                logger.warning(
                    f"Skipping profile {path.name} due to error: {e}"
                )

        if not self.flat_index:
            raise ValueError(
                "Dataset created with zero valid samples. "
                "Check profile files and variable names."
            )

    def _load_single_profile(self, path: Path) -> Dict[str, Any]:
        """Loads a single profile from disk."""
        try:
            with path.open("r", encoding=UTF8_ENCODING) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to load profile {path.name}: {e}")

    def _validate_profile_data(
        self, profile_data: Dict[str, Any], path: Path
    ) -> None:
        """Validates that a profile contains required variables and structure."""
        # Check for time data
        if TIME_KEY not in profile_data:
            raise ValueError(f"Profile {path.name} missing '{TIME_KEY}' key")
        
        time_data = profile_data[TIME_KEY]
        if not isinstance(time_data, list) or len(time_data) < MIN_TIME_STEPS:
            raise ValueError(
                f"Profile {path.name} needs at least {MIN_TIME_STEPS} time points"
            )
        
        # Check for required species variables
        missing_species = [
            var for var in self.species_vars 
            if var not in profile_data
        ]
        if missing_species:
            raise ValueError(
                f"Profile {path.name} missing species variables: {missing_species}"
            )
        
        # Check for required global variables
        missing_globals = [
            var for var in self.global_vars 
            if var not in profile_data
        ]
        if missing_globals:
            raise ValueError(
                f"Profile {path.name} missing global variables: {missing_globals}"
            )
        
        # Validate data length consistency
        num_time_steps = len(time_data)
        for var in self.species_vars:
            var_data = profile_data[var]
            if not isinstance(var_data, list) or len(var_data) != num_time_steps:
                raise ValueError(
                    f"Profile {path.name} variable '{var}' length mismatch"
                )

    def _add_profile_to_index(
        self, profile_idx: int, profile_data: Dict[str, Any]
    ) -> None:
        """Adds training samples from a profile to the flat index."""
        num_time_steps = len(profile_data[TIME_KEY])
        
        # Create training samples for each time step (except t=0 which is input)
        for time_idx in range(1, num_time_steps):
            self.flat_index.append((profile_idx, time_idx))

    def __len__(self) -> int:
        """Returns the total number of training samples."""
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Gets a training sample by index.
        
        Args:
            idx: Index of the training sample
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        if idx >= len(self.flat_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Map the flat index to the specific profile and time step
        profile_idx, query_time_idx = self.flat_index[idx]
        
        # Retrieve the profile data from the cache
        profile = self.profile_data_cache.get(profile_idx)
        if profile is None:
            raise RuntimeError(
                f"Profile data for index {profile_idx} not found in cache. "
                f"This indicates the profile failed to load or a dataset initialization error."
            )

        # Build input vector: [initial_species, global_vars, query_time]
        input_vector = self._build_input_vector(profile, query_time_idx)
        
        # Build target vector: [species_at_query_time]
        target_vector = self._build_target_vector(profile, query_time_idx)
        
        return input_vector, target_vector

    def _build_input_vector(self, profile: Dict[str, Any], query_time_idx: int) -> Tensor:
        """Builds the input vector for the model."""
        # Initial species values (at t=0)
        initial_species = [profile[var][0] for var in self.species_vars]
        
        # Global conditions (constant across time)
        global_conditions = [profile[var] for var in self.global_vars]
        
        # Query time
        query_time = profile[TIME_KEY][query_time_idx]
        
        # Combine all inputs
        input_data = initial_species + global_conditions + [query_time]
        return torch.tensor(input_data, dtype=DTYPE)

    def _build_target_vector(self, profile: Dict[str, Any], query_time_idx: int) -> Tensor:
        """Builds the target vector for the model."""
        target_species = [profile[var][query_time_idx] for var in self.species_vars]
        return torch.tensor(target_species, dtype=DTYPE)

    def get_profile_filenames_by_indices(self, indices: List[int]) -> List[str]:
        """
        Retrieves unique profile filenames corresponding to dataset indices.
        
        Args:
            indices: List of flat dataset indices
            
        Returns:
            List of unique profile filenames
        """
        if not indices:
            return []
        
        # Validate indices
        max_idx = max(indices)
        if max_idx >= len(self.flat_index):
            raise IndexError(f"Index {max_idx} out of range for dataset")
        
        # Get unique profile indices from the flat index
        unique_profile_indices = {
            self.flat_index[i][0] for i in indices 
            if i < len(self.flat_index)
        }
        
        # Map to original profile paths
        # Since profile_idx corresponds to position in profiles_to_load,
        # and profiles_to_load is a slice of self.profile_paths,
        # the mapping is direct
        filenames = []
        for profile_idx in sorted(unique_profile_indices):
            if profile_idx < len(self.profile_paths):
                filenames.append(self.profile_paths[profile_idx].name)
        
        return filenames


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    Collates a batch of samples into tensors suitable for model training.
    
    Args:
        batch: List of (input_tensor, target_tensor) tuples
        
    Returns:
        Tuple of (model_inputs_dict, target_batch_tensor)
    """
    if not batch:
        return {"x": torch.empty(0)}, torch.empty(0)
    
    input_vectors, target_vectors = zip(*batch)
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)
    
    return model_inputs, target_batch


__all__ = ["ChemicalDataset", "collate_fn"]