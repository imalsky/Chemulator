#!/usr/bin/env python3
"""
dataset.py - Data loading for the State-Evolution Predictor Model (MLP).
This version loads pre-normalized data and constructs samples for the MLP.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ChemicalDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-normalized chemical profiles.

    This Dataset implements a "flattening" strategy. Instead of treating a
    profile as a single item, it treats every valid time-point within every
    profile as a unique training example. This ensures that all data points
    are utilized during training.

    An optional `profile_paths` list can be provided to initialize the dataset
    with only a specific subset of profiles (e.g., for train/val/test splits).
    """
    def __init__(
        self,
        data_folder: Union[str, Path],
        species_variables: List[str],
        global_variables: List[str],
        *,
        profile_paths: Optional[List[Path]] = None
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Normalized data directory not found: {self.data_dir}.")
        
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        # If specific profile paths are not provided, discover them from the data_folder.
        # This is used to create specific datasets for train, val, and test sets.
        if profile_paths is None:
            self.profile_paths = sorted([p for p in self.data_dir.glob("*.json") if p.name != "normalization_metadata.json"])
        else:
            self.profile_paths = profile_paths
            
        if not self.profile_paths:
            raise FileNotFoundError(f"No profiles found to build dataset from in {self.data_dir}.")
            
        # --- Flattening Logic ---
        # Instead of a dataset of N profiles, we create a dataset of M time-points.
        # self.flat_index will store tuples of (profile_index, time_step_index)
        # for every valid sample.
        self.flat_index: List[Tuple[int, int]] = []
        self.profile_data_cache: List[Optional[Dict[str, Any]]] = [None] * len(self.profile_paths)
        
        logger.info(f"Building dataset index from {len(self.profile_paths)} profiles...")
        for profile_idx, path in enumerate(self.profile_paths):
            try:
                with path.open("r", encoding="utf-8-sig") as f:
                    profile_data = json.load(f)
                    # Cache the loaded data to avoid repeated file I/O
                    self.profile_data_cache[profile_idx] = profile_data
                    num_time_steps = len(profile_data["t_time"])
                    # Create a sample for each time step except the first one (t=0)
                    for time_idx in range(1, num_time_steps):
                        self.flat_index.append((profile_idx, time_idx))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping profile {path.name} due to loading error: {e}")

        if not self.flat_index:
            raise ValueError("Dataset created with zero valid samples. Check profile files.")

        logger.info(f"ChemicalDataset initialized with {len(self.profile_paths)} profiles, creating {len(self)} total training samples.")

    def __len__(self) -> int:
        # The length of the dataset is the total number of individual time-points.
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Map the flat index to the specific profile and time step.
        profile_idx, query_time_idx = self.flat_index[idx]
        
        # Retrieve the profile data from the cache (loaded during __init__).
        profile = self.profile_data_cache[profile_idx]
        if profile is None:
            # This should not happen with the current logic, but is a safeguard.
            raise RuntimeError(f"Profile data for index {profile_idx} was not cached.")

        # --- Assemble the model input ---
        # The input is ALWAYS constructed from the initial state (t=0).
        initial_species = [profile[key][0] for key in self.species_vars]
        global_conds = [profile[key] for key in self.global_vars]
        time_query = profile["t_time"][query_time_idx]
        input_vector = torch.tensor(initial_species + global_conds + [time_query], dtype=torch.float32)

        # --- Assemble the target output ---
        # The target is the state at the queried time step.
        target_species = [profile[key][query_time_idx] for key in self.species_vars]
        target_vector = torch.tensor(target_species, dtype=torch.float32)
        
        return input_vector, target_vector

    # FIX: Add this method back in for the trainer to use
    def get_profile_filenames_by_indices(self, indices: List[int]) -> List[str]:
        """
        Retrieves the unique profile filenames corresponding to a list of
        flattened dataset indices.
        """
        # 1. For each flat index, find the original profile_idx it belongs to.
        # 2. Use a set to get only the unique profile indices.
        # 3. Map these unique indices back to their filenames.
        unique_profile_indices = {self.flat_index[i][0] for i in indices}
        return [self.profile_paths[i].name for i in sorted(list(unique_profile_indices))]

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    if not batch: return {}, torch.empty(0)
    input_vectors, target_vectors = zip(*batch)
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    return model_inputs, torch.stack(target_vectors, dim=0)

__all__ = ["ChemicalDataset", "collate_fn"]