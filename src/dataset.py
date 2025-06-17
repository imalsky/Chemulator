#!/usr/bin/env python3
"""
dataset.py - Data loading for the State-Evolution Predictor Model (MLP).
This version loads pre-normalized data and constructs samples for the MLP.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ChemicalDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-normalized chemical profiles.
    
    cache_size=-1 (default): Load all profiles into memory
    cache_size=N: Load only first N profiles into memory
    """
    def __init__(
        self,
        data_folder: Union[str, Path],
        species_variables: List[str],
        global_variables: List[str],
        *,
        profile_paths: Optional[List[Path]] = None,
        cache_size: int = -1,  # Default: load all
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Normalized data directory not found: {self.data_dir}.")
        
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        # If specific profile paths are not provided, discover them
        if profile_paths is None:
            self.profile_paths = sorted([p for p in self.data_dir.glob("*.json") if p.name != "normalization_metadata.json"])
        else:
            self.profile_paths = profile_paths
            
        if not self.profile_paths:
            raise FileNotFoundError(f"No profiles found to build dataset from in {self.data_dir}.")
        
        # Determine how many profiles to load
        if cache_size == -1:
            profiles_to_load = self.profile_paths
        else:
            profiles_to_load = self.profile_paths[:cache_size]
        
        # Load profiles and build index
        self.flat_index: List[Tuple[int, int]] = []
        self.profile_data_cache: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"Loading {len(profiles_to_load)} profiles into memory...")
        for profile_idx, path in enumerate(profiles_to_load):
            try:
                with path.open("r", encoding="utf-8-sig") as f:
                    profile_data = json.load(f)
                    
                # Cache the loaded data
                self.profile_data_cache[profile_idx] = profile_data
                
                # Build index
                num_time_steps = len(profile_data["t_time"])
                for time_idx in range(1, num_time_steps):
                    self.flat_index.append((profile_idx, time_idx))
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping profile {path.name} due to loading error: {e}")

        if not self.flat_index:
            raise ValueError("Dataset created with zero valid samples. Check profile files.")

        logger.info(f"ChemicalDataset initialized with {len(profiles_to_load)} profiles, creating {len(self)} total training samples.")

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Map the flat index to the specific profile and time step
        profile_idx, query_time_idx = self.flat_index[idx]
        
        # Retrieve the profile data from the cache
        profile = self.profile_data_cache.get(profile_idx)
        if profile is None:
            raise RuntimeError(f"Profile data for index {profile_idx} was not cached.")

        # Assemble the model input from initial state (t=0)
        initial_species = [profile[key][0] for key in self.species_vars]
        global_conds = [profile[key] for key in self.global_vars]
        time_query = profile["t_time"][query_time_idx]
        input_vector = torch.tensor(initial_species + global_conds + [time_query], dtype=torch.float32)

        # Assemble the target output at queried time
        target_species = [profile[key][query_time_idx] for key in self.species_vars]
        target_vector = torch.tensor(target_species, dtype=torch.float32)
        
        return input_vector, target_vector

    def get_profile_filenames_by_indices(self, indices: List[int]) -> List[str]:
        """
        Retrieves the unique profile filenames corresponding to a list of
        flattened dataset indices.
        """
        unique_profile_indices = {self.flat_index[i][0] for i in indices}
        return [self.profile_paths[i].name for i in sorted(list(unique_profile_indices))]

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    if not batch: 
        return {}, torch.empty(0)
    input_vectors, target_vectors = zip(*batch)
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    return model_inputs, torch.stack(target_vectors, dim=0)

__all__ = ["ChemicalDataset", "collate_fn"]