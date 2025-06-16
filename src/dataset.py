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
from typing import Any, Dict, List, Tuple, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ChemicalDataset(Dataset):
    def __init__(
        self, data_folder: Union[str, Path], species_variables: List[str],
        global_variables: List[str], all_variables: List[str], *, validate_profiles: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir(): raise FileNotFoundError(f"Normalized data directory not found: {self.data_dir}.")
        
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        self.profile_paths = sorted([p for p in self.data_dir.glob("*.json") if p.name != "normalization_metadata.json"])
        if not self.profile_paths: raise FileNotFoundError(f"No profiles found in {self.data_dir}.")
        logger.info(f"ChemicalDataset initialized with {len(self)} profiles.")

    def __len__(self) -> int:
        return len(self.profile_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        with self.profile_paths[idx].open("r", encoding="utf-8-sig") as f:
            profile = json.load(f)

        query_idx = random.randint(1, len(profile["t_time"]) - 1)

        initial_species = [profile[key][0] for key in self.species_vars]
        global_conds = [profile[key] for key in self.global_vars]
        time_query = profile["t_time"][query_idx]
        input_vector = torch.tensor(initial_species + global_conds + [time_query], dtype=torch.float32)

        target_species = [profile[key][query_idx] for key in self.species_vars]
        target_vector = torch.tensor(target_species, dtype=torch.float32)
        
        return input_vector, target_vector

    # FIX: Add this method back in for the trainer to use
    def get_profile_filenames_by_indices(self, indices: List[int]) -> List[str]:
        """Retrieves the filenames for a given list of dataset indices."""
        return [self.profile_paths[i].name for i in indices]

def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    if not batch: return {}, torch.empty(0)
    input_vectors, target_vectors = zip(*batch)
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    return model_inputs, torch.stack(target_vectors, dim=0)

__all__ = ["ChemicalDataset", "collate_fn"]