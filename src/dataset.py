#!/usr/bin/env python3
"""
dataset.py - Data loading (REFACTORED for performance)

Key improvements:
- Fixed missing random import
- Added proper worker initialization for HDF5
- Improved error handling for multi-worker scenarios
- Added memory-efficient chunking with dynamic sizing
"""
from __future__ import annotations

import h5py
import logging
import numpy as np
import random  # FIX: Added missing import
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from normalizer import DataNormalizer

TIME_KEY = "t_time"
DTYPE = torch.float32
MIN_TIME_STEPS = 2

logger = logging.getLogger(__name__)


class ChemicalDataset(IterableDataset):
    """
    Optimized PyTorch IterableDataset for chemical reaction profiles.
    
    Streams data by reading large chunks of profiles, shuffling them in memory,
    and yielding individual samples. This avoids random disk I/O bottlenecks.
    
    Key features:
    - Efficient chunk-based reading for HDF5
    - Proper multi-worker support with HDF5
    - Memory-aware chunking
    - Robust error handling
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: List[int],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        *,
        profiles_per_chunk: int = 2048,
        max_memory_gb: float = 2.0,  # Maximum memory per worker
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            h5_path: Path to HDF5 file
            indices: Profile indices to use
            species_variables: List of species variable names
            global_variables: List of global variable names
            normalization_metadata: Pre-calculated normalization statistics
            profiles_per_chunk: Number of profiles to read at once
            max_memory_gb: Maximum memory to use per worker (in GB)
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.indices = indices
        self.profiles_per_chunk = profiles_per_chunk
        self.max_memory_gb = max_memory_gb
        
        # Store file handle per worker
        self.h5_file_handle: Optional[h5py.File] = None
        self.worker_id: Optional[int] = None

        self.norm_metadata = normalization_metadata
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)
        
        # Define variable order
        self.input_var_order = self.species_vars + self.global_vars + [TIME_KEY]
        self.target_var_order = self.species_vars
        
        self.all_vars = set(self.input_var_order)

        # Validate HDF5 structure and get metadata
        self.num_time_steps: Optional[int] = None
        self._validate_h5_structure()
        
        # Calculate total samples for progress tracking
        self.total_samples = len(self.indices) * (self.num_time_steps - 1)
        
        # Adjust chunk size based on available memory
        self._adjust_chunk_size()

        # Setup vectorized normalization
        self._setup_vectorized_normalization()

        logger.info(
            f"ChemicalDataset initialized: {len(self.indices)} profiles, "
            f"~{self.total_samples:,} samples, chunk_size={self.profiles_per_chunk}"
        )

    def _adjust_chunk_size(self) -> None:
        """Dynamically adjust chunk size based on memory constraints."""
        # Estimate memory per profile (rough calculation)
        num_vars = len(self.all_vars)
        bytes_per_element = 4  # float32
        elements_per_profile = num_vars * self.num_time_steps
        mb_per_profile = (elements_per_profile * bytes_per_element) / (1024 * 1024)
        
        # Calculate maximum profiles that fit in memory
        max_profiles = int((self.max_memory_gb * 1024) / mb_per_profile)
        
        # Adjust chunk size if needed
        if self.profiles_per_chunk > max_profiles:
            old_size = self.profiles_per_chunk
            self.profiles_per_chunk = max(1, max_profiles)
            logger.warning(
                f"Reduced chunk size from {old_size} to {self.profiles_per_chunk} "
                f"to fit in {self.max_memory_gb}GB memory limit"
            )

    def _setup_vectorized_normalization(self) -> None:
        """Pre-compute normalization parameters for efficient operations."""
        self.norm_params = {"input": {}, "target": {}}
        
        for vector_type, var_order in [("input", self.input_var_order), ("target", self.target_var_order)]:
            n_vars = len(var_order)
            
            # Initialize tensors for stats
            means = torch.zeros(n_vars, dtype=DTYPE)
            stds = torch.ones(n_vars, dtype=DTYPE)
            log_means = torch.zeros(n_vars, dtype=DTYPE)
            log_stds = torch.ones(n_vars, dtype=DTYPE)
            mins = torch.zeros(n_vars, dtype=DTYPE)
            maxs = torch.ones(n_vars, dtype=DTYPE)
            
            methods = []
            var_to_idx = {var: i for i, var in enumerate(var_order)}
            
            for i, var in enumerate(var_order):
                method = self.norm_metadata["normalization_methods"].get(var, "none")
                methods.append(method)
                
                stats = self.norm_metadata["per_key_stats"].get(var)
                
                if not stats or method == "none":
                    continue
                
                # Populate stats based on method
                if method == "standard":
                    means[i] = stats.get("mean", 0.0)
                    stds[i] = stats.get("std", 1.0)
                elif method == "log-standard":
                    log_means[i] = stats.get("log_mean", 0.0)
                    log_stds[i] = stats.get("log_std", 1.0)
                elif method == "log-min-max":
                    mins[i] = stats.get("min", 0.0)
                    maxs[i] = stats.get("max", 1.0)
            
            # Create masks for each method for vectorized application
            methods_np = np.array(methods)
            masks = {
                method: torch.from_numpy(methods_np == method)
                for method in set(methods) if method != "none"
            }
            
            self.norm_params[vector_type] = {
                "methods": methods,
                "masks": masks,
                "means": means,
                "stds": stds,
                "log_means": log_means,
                "log_stds": log_stds,
                "mins": mins,
                "maxs": maxs,
                "var_to_idx": var_to_idx,
            }

    def _get_h5_handle(self) -> h5py.File:
        """
        Opens an HDF5 file handle, specific to each DataLoader worker.
        
        This is crucial for multi-worker support with HDF5.
        Each worker must have its own file handle to avoid conflicts.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        # Create new handle if worker changed or handle doesn't exist
        if self.h5_file_handle is None or self.worker_id != worker_id:
            if self.h5_file_handle is not None:
                self.h5_file_handle.close()
            
            # Open with SWMR mode for multi-process safety
            # libver='latest' enables better concurrent access
            self.h5_file_handle = h5py.File(
                self.h5_path, 'r', 
                swmr=True,
                libver='latest'
            )
            self.worker_id = worker_id
            
            if worker_id >= 0:
                logger.debug(f"Worker {worker_id} opened HDF5 file handle")
        
        return self.h5_file_handle

    def _validate_h5_structure(self) -> None:
        """Validates HDF5 structure and extracts metadata."""
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            num_time_steps = hf[TIME_KEY].shape[1]
            
            if num_time_steps < MIN_TIME_STEPS:
                raise ValueError(
                    f"Profiles must have at least {MIN_TIME_STEPS} time steps, got {num_time_steps}."
                )
            
            # Validate that all required variables exist
            missing_vars = self.all_vars - set(hf.keys())
            if missing_vars:
                logger.warning(f"Variables not found in HDF5: {missing_vars}")
            
            self.num_time_steps = num_time_steps
        
        logger.info(f"HDF5 validated: {self.num_time_steps} timesteps per profile")

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Main iterator method for the IterableDataset.
        
        Handles:
        - Multi-worker data splitting
        - Chunk-based reading for efficiency
        - In-memory shuffling
        - Robust error handling
        """
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Split indices among workers
        indices_per_worker = len(self.indices) // num_workers
        extra_indices = len(self.indices) % num_workers
        
        # Calculate this worker's share
        start_idx = worker_id * indices_per_worker + min(worker_id, extra_indices)
        end_idx = start_idx + indices_per_worker + (1 if worker_id < extra_indices else 0)
        
        worker_indices = self.indices[start_idx:end_idx]
        
        if not worker_indices:
            logger.warning(f"Worker {worker_id} has no indices to process")
            return
        
        logger.debug(
            f"Worker {worker_id}/{num_workers} processing {len(worker_indices)} profiles "
            f"(indices {start_idx}-{end_idx})"
        )
        
        # Shuffle indices for this epoch
        # Use worker_id as part of seed for different shuffling per worker
        epoch_seed = random.randint(0, 2**31)
        random.seed(epoch_seed + worker_id)
        random.shuffle(worker_indices)
        
        # Get file handle for this worker
        h5_file = self._get_h5_handle()
        
        # Process profiles in chunks
        for chunk_start in range(0, len(worker_indices), self.profiles_per_chunk):
            chunk_end = min(chunk_start + self.profiles_per_chunk, len(worker_indices))
            chunk_profile_indices = worker_indices[chunk_start:chunk_end]
            
            if not chunk_profile_indices:
                continue
            
            try:
                # Read chunk data efficiently
                chunk_data = self._read_chunk_data(h5_file, chunk_profile_indices)
                
                # Generate samples from chunk
                yield from self._generate_samples_from_chunk(chunk_data, chunk_profile_indices)
                
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} error processing chunk "
                    f"{chunk_start}-{chunk_end}: {e}"
                )
                # Continue with next chunk instead of failing completely
                continue

    def _read_chunk_data(
        self, h5_file: h5py.File, chunk_indices: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Efficiently read a chunk of data from HDF5.
        
        Uses sorted fancy indexing for optimal HDF5 performance.
        """
        # Sort indices for efficient HDF5 access
        sorted_indices = sorted(chunk_indices)
        index_map = {idx: i for i, idx in enumerate(sorted_indices)}
        
        chunk_data = {}
        
        for var in self.all_vars:
            if var not in h5_file:
                logger.debug(f"Variable '{var}' not in HDF5, skipping")
                continue
            
            try:
                # Read data in sorted order (most efficient for HDF5)
                data = h5_file[var][sorted_indices]
                
                # Reorder to match original chunk order
                reordered_data = np.empty_like(data)
                for orig_idx, chunk_idx in enumerate(chunk_indices):
                    reordered_data[orig_idx] = data[index_map[chunk_idx]]
                
                chunk_data[var] = reordered_data
                
            except Exception as e:
                logger.error(f"Error reading variable '{var}': {e}")
                # Return zeros as fallback
                shape = (len(chunk_indices),) + h5_file[var].shape[1:]
                chunk_data[var] = np.zeros(shape, dtype=np.float32)
        
        return chunk_data

    def _generate_samples_from_chunk(
        self, chunk_data: Dict[str, np.ndarray], chunk_indices: List[int]
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate shuffled samples from a chunk of data."""
        # Create flat index for all samples in chunk
        chunk_samples = []
        for i, profile_idx in enumerate(chunk_indices):
            for time_idx in range(1, self.num_time_steps):
                chunk_samples.append((i, profile_idx, time_idx))
        
        # Shuffle samples within chunk
        random.shuffle(chunk_samples)
        
        # Generate samples
        for chunk_pos, profile_idx, time_idx in chunk_samples:
            try:
                # Extract profile data
                profile_data = {
                    var: data[chunk_pos] for var, data in chunk_data.items()
                }
                
                # Build input and target
                raw_input = self._build_raw_input(profile_data, time_idx)
                raw_target = self._build_raw_target(profile_data, time_idx)
                
                # Normalize
                input_vector = self._normalize_vector_optimized(raw_input, "input")
                target_vector = self._normalize_vector_optimized(raw_target, "target")
                
                yield input_vector, target_vector
                
            except Exception as e:
                logger.debug(
                    f"Error generating sample for profile {profile_idx}, "
                    f"time {time_idx}: {e}"
                )
                # Skip this sample
                continue

    def _normalize_vector_optimized(self, raw_vector: np.ndarray, vector_type: str) -> Tensor:
        """Highly optimized vectorized normalization with proper error handling."""
        tensor = torch.from_numpy(raw_vector).to(dtype=DTYPE)
        params = self.norm_params[vector_type]
        
        normalized = tensor.clone()
        
        try:
            # Apply vectorized normalization for each method
            if "standard" in params["masks"]:
                mask = params["masks"]["standard"]
                normalized[mask] = (tensor[mask] - params["means"][mask]) / params["stds"][mask]
            
            if "log-standard" in params["masks"]:
                mask = params["masks"]["log-standard"]
                safe_tensor = torch.clamp(tensor[mask], min=1e-30)
                log_tensor = torch.log10(safe_tensor)
                normalized[mask] = (log_tensor - params["log_means"][mask]) / params["log_stds"][mask]

            if "log-min-max" in params["masks"]:
                mask = params["masks"]["log-min-max"]
                safe_tensor = torch.clamp(tensor[mask], min=1e-30)
                log_tensor = torch.log10(safe_tensor)
                denom = params["maxs"][mask] - params["mins"][mask]
                denom = torch.clamp(denom, min=1e-6)  # Prevent division by zero
                normed = (log_tensor - params["mins"][mask]) / denom
                normalized[mask] = torch.clamp(normed, 0.0, 1.0)
            
            # Handle non-vectorized methods
            for i, method in enumerate(params["methods"]):
                if method not in {"standard", "log-standard", "log-min-max", "none"}:
                    var_name = (self.input_var_order if vector_type == "input" 
                               else self.target_var_order)[i]
                    stats = self.norm_metadata["per_key_stats"].get(var_name, {})
                    if stats:
                        normalized[i] = DataNormalizer.normalize_tensor(
                            tensor[i].unsqueeze(0), method, stats
                        ).squeeze(0)

        except Exception as e:
            logger.error(
                f"Normalization failed for {vector_type}: {e}. Using raw values."
            )
            return tensor
        
        return normalized

    def _build_raw_input(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Constructs the raw input vector with proper error handling."""
        input_data = []
        
        # Species at initial time (t=0)
        for var in self.species_vars:
            if var in profile:
                input_data.append(profile[var][0])
            else:
                logger.debug(f"Missing species variable '{var}', using 0.0")
                input_data.append(0.0)
        
        # Global variables
        for var in self.global_vars:
            if var in profile:
                raw = profile[var]
                # Handle scalar vs array
                if np.ndim(raw) == 0 or (hasattr(raw, 'shape') and raw.shape == ()):
                    input_data.append(float(raw))
                else:
                    input_data.append(raw.item() if hasattr(raw, 'item') else float(raw))
            else:
                logger.debug(f"Missing global variable '{var}', using 0.0")
                input_data.append(0.0)
        
        # Time value
        if TIME_KEY in profile:
            input_data.append(profile[TIME_KEY][query_time_idx])
        else:
            logger.debug(f"Missing time variable, using index {query_time_idx}")
            input_data.append(float(query_time_idx))
        
        return np.array(input_data, dtype=np.float32)

    def _build_raw_target(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Constructs the raw target vector with proper error handling."""
        target_data = []
        
        # Species at query time
        for var in self.species_vars:
            if var in profile:
                target_data.append(profile[var][query_time_idx])
            else:
                logger.debug(f"Missing target variable '{var}' at time {query_time_idx}, using 0.0")
                target_data.append(0.0)
        
        return np.array(target_data, dtype=np.float32)

    def __del__(self):
        """Cleanup: close HDF5 file handle if open."""
        self.close()

    def close(self) -> None:
        """Close HDF5 file handle if open."""
        if self.h5_file_handle is not None:
            try:
                self.h5_file_handle.close()
                logger.debug(f"Closed HDF5 file handle for worker {self.worker_id}")
            except Exception as e:
                logger.debug(f"Error closing HDF5 file: {e}")
            finally:
                self.h5_file_handle = None
                self.worker_id = None


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    Optimized collate function with validation.
    
    Filters out invalid samples and provides informative warnings.
    """
    # Filter valid samples
    valid_batch = []
    for i, sample in enumerate(batch):
        if sample is None:
            continue
        
        if not isinstance(sample, tuple) or len(sample) != 2:
            logger.debug(f"Invalid sample format at index {i}")
            continue
        
        inputs, targets = sample
        if torch.isfinite(inputs).all() and torch.isfinite(targets).all():
            valid_batch.append(sample)
        else:
            logger.debug(f"Non-finite values in sample at index {i}")
    
    if not valid_batch:
        raise RuntimeError("Collate function received an empty or all-invalid batch.")
    
    if len(valid_batch) < len(batch):
        logger.debug(f"Dropped {len(batch) - len(valid_batch)} invalid samples from batch")
    
    # Stack valid samples
    input_vectors, target_vectors = zip(*valid_batch)
    
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)
    
    return model_inputs, target_batch


__all__ = ["ChemicalDataset", "collate_fn"]