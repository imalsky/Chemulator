#!/usr/bin/env python3
"""
dataset.py - Efficient data loading with caching and streaming support.

This module provides two dataset implementations:
1. CachedChemicalDataset: Loads all data into memory for fastest training
2. ChemicalDataset: Streams data from disk for memory-efficient training
"""
from __future__ import annotations

import gc
import h5py
import logging
import numpy as np
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset

from normalizer import DataNormalizer

TIME_KEY = "t_time"
DTYPE = torch.float32
MIN_TIME_STEPS = 2

logger = logging.getLogger(__name__)


class CachedChemicalDataset(Dataset):
    """
    Pre-loads and caches all data in memory for fastest possible training.
    
    This dataset loads all data once, normalizes it, and builds all input/target
    tensors in memory. This makes __getitem__ extremely fast (just tensor slicing)
    at the cost of higher memory usage.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: List[int],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        raw_data_for_caching: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Initialize and cache the entire dataset in memory.
        
        Args:
            h5_path: Path to HDF5 file
            indices: Profile indices to use
            species_variables: List of species variable names
            global_variables: List of global variable names
            normalization_metadata: Pre-calculated normalization statistics
            raw_data_for_caching: Optional pre-loaded raw data to avoid re-reading
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        self.indices = indices
        self.norm_metadata = normalization_metadata
        self.species_vars = sorted(species_variables)
        self.global_vars = sorted(global_variables)

        # Define variable order for consistent tensor structure
        self.input_var_order = self.species_vars + self.global_vars + [TIME_KEY]
        self.target_var_order = self.species_vars

        # Load, normalize, and build tensors
        logger.info(f"Pre-processing and caching {len(indices)} profiles into memory...")
        start_time = time.time()
        
        if raw_data_for_caching:
            logger.info("Using pre-loaded raw data for caching.")
            self._build_tensors(raw_data_for_caching)
        else:
            logger.info(f"Reading raw data from {self.h5_path} for caching.")
            self._load_and_build_tensors()

        load_time = time.time() - start_time
        memory_mb = (self.input_tensors.nbytes + self.target_tensors.nbytes) / (1024 * 1024)
        
        logger.info(
            f"Fully cached {len(self.indices)} profiles ({self.total_samples:,} samples) "
            f"in {load_time:.1f}s, using ~{memory_mb:.1f}MB of memory."
        )

    def _load_and_build_tensors(self) -> None:
        """Load data from HDF5 and build cached tensors."""
        raw_data = {}
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            num_time_steps = hf[TIME_KEY].shape[1]
            all_vars_to_load = set(self.input_var_order) | set(self.target_var_order)
            
            for var in all_vars_to_load:
                if var in hf:
                    raw_data[var] = hf[var][self.indices]
                else:
                    logger.warning(f"Variable '{var}' not in HDF5, using zeros.")
                    if var in self.global_vars:
                        shape = (len(self.indices),)
                    else:
                        shape = (len(self.indices), num_time_steps)
                    raw_data[var] = np.zeros(shape, dtype=np.float32)
                    
        self._build_tensors(raw_data)

    def _build_tensors(self, raw_data: Dict[str, np.ndarray]) -> None:
        """
        Build normalized input/target tensors using memory-efficient strategy.
        
        This method normalizes data and constructs samples directly into
        pre-allocated tensors to minimize peak memory usage.
        
        Args:
            raw_data: Dictionary of raw numpy arrays
        """
        logger.debug("Starting tensor building...")
        start_time = time.time()
        
        # Determine dimensions
        time_var_data = raw_data.get(TIME_KEY)
        if time_var_data is None:
            raise ValueError(f"'{TIME_KEY}' is missing from raw data.")
        
        self.num_time_steps = time_var_data.shape[1]
        if self.num_time_steps < MIN_TIME_STEPS:
            raise ValueError(f"Profiles must have at least {MIN_TIME_STEPS} time steps.")
        
        logger.debug(f"Determined {self.num_time_steps} timesteps.")

        # Normalize all data
        logger.debug("Normalizing all raw data...")
        normalized_data = {}
        for var, data in raw_data.items():
            method = self.norm_metadata["normalization_methods"].get(var, "none")
            stats = self.norm_metadata["per_key_stats"].get(var, {})
            tensor = torch.from_numpy(data)
            normalized_data[var] = DataNormalizer.normalize_tensor(tensor, method, stats)
        
        # Free raw data memory
        del raw_data
        gc.collect()
        logger.debug(f"Normalization complete. Time: {time.time() - start_time:.2f}s")

        # Pre-allocate final tensors
        num_profiles = len(self.indices)
        num_samples_per_profile = self.num_time_steps - 1
        self.total_samples = num_profiles * num_samples_per_profile

        num_input_features = len(self.species_vars) + len(self.global_vars) + 1
        num_target_features = len(self.species_vars)
        
        logger.debug(f"Pre-allocating tensors for {self.total_samples:,} samples...")
        self.input_tensors = torch.empty((self.total_samples, num_input_features), dtype=DTYPE)
        self.target_tensors = torch.empty((self.total_samples, num_target_features), dtype=DTYPE)

        # Fill tensors efficiently using vectorized operations
        logger.debug("Filling pre-allocated tensors...")
        for i in range(num_profiles):
            start_idx = i * num_samples_per_profile
            end_idx = start_idx + num_samples_per_profile
            
            # Extract data for this profile
            initial_species = torch.stack([normalized_data[var][i, 0] for var in self.species_vars], dim=0)
            global_vars = torch.tensor([normalized_data[var][i] for var in self.global_vars], dtype=DTYPE)
            time_vals = normalized_data[TIME_KEY][i, 1:]
            
            # Fill input tensor using broadcasting
            self.input_tensors[start_idx:end_idx, :len(self.species_vars)] = initial_species
            self.input_tensors[start_idx:end_idx, len(self.species_vars):-1] = global_vars
            self.input_tensors[start_idx:end_idx, -1] = time_vals
            
            # Fill target tensor
            target_species = torch.stack(
                [normalized_data[var][i, 1:] for var in self.target_var_order], dim=1
            )
            self.target_tensors[start_idx:end_idx, :] = target_species
        
        # Clean up normalized data
        del normalized_data
        gc.collect()
        logger.debug(f"Tensor building complete. Time: {time.time() - start_time:.2f}s")

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample (extremely fast - just tensor slicing).
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input tensor, target tensor)
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        return self.input_tensors[idx], self.target_tensors[idx]


class ChemicalDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for chemical reaction profiles.
    
    This dataset reads data in chunks from HDF5, making it suitable for
    large datasets that don't fit in memory. It supports multi-worker
    loading with proper HDF5 file handle management.
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
        max_memory_gb: float = 2.0,
    ) -> None:
        """
        Initialize the streaming dataset.
        
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

        # Setup vectorized normalization for efficiency
        self._setup_vectorized_normalization()

        logger.info(
            f"ChemicalDataset initialized: {len(self.indices)} profiles, "
            f"~{self.total_samples:,} samples, chunk_size={self.profiles_per_chunk}"
        )

    def _adjust_chunk_size(self) -> None:
        """Dynamically adjust chunk size based on memory constraints."""
        # Estimate memory per profile
        num_vars = len(self.all_vars)
        bytes_per_element = 4  # float32
        elements_per_profile = num_vars * self.num_time_steps
        mb_per_profile = (elements_per_profile * bytes_per_element) / (1024 * 1024)
        
        if mb_per_profile == 0:
            return

        # Calculate maximum profiles that fit in memory limit
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
        """Pre-compute normalization parameters for efficient batch operations."""
        self.norm_params = {"input": {}, "target": {}}
        
        for vector_type, var_order in [("input", self.input_var_order), ("target", self.target_var_order)]:
            n_vars = len(var_order)
            
            # Initialize tensors for statistics
            means = torch.zeros(n_vars, dtype=DTYPE)
            stds = torch.ones(n_vars, dtype=DTYPE)
            log_means = torch.zeros(n_vars, dtype=DTYPE)
            log_stds = torch.ones(n_vars, dtype=DTYPE)
            mins = torch.zeros(n_vars, dtype=DTYPE)
            maxs = torch.ones(n_vars, dtype=DTYPE)
            
            methods = []
            var_to_idx = {var: i for i, var in enumerate(var_order)}
            
            # Populate statistics for each variable
            for i, var in enumerate(var_order):
                method = self.norm_metadata["normalization_methods"].get(var, "none")
                methods.append(method)
                
                stats = self.norm_metadata["per_key_stats"].get(var)
                if not stats or method == "none":
                    continue
                
                # Populate stats based on normalization method
                if method == "standard":
                    means[i] = stats.get("mean", 0.0)
                    stds[i] = stats.get("std", 1.0)
                elif method == "log-standard":
                    log_means[i] = stats.get("log_mean", 0.0)
                    log_stds[i] = stats.get("log_std", 1.0)
                elif method == "log-min-max":
                    mins[i] = stats.get("min", 0.0)
                    maxs[i] = stats.get("max", 1.0)
            
            # Create masks for vectorized application
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
        Get HDF5 file handle with proper multi-worker support.
        
        Each worker maintains its own file handle to avoid conflicts.
        
        Returns:
            HDF5 file handle
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        # Create new handle if worker changed or handle doesn't exist
        if self.h5_file_handle is None or self.worker_id != worker_id:
            if self.h5_file_handle is not None:
                self.h5_file_handle.close()
            
            # Open with SWMR mode for multi-process safety
            try:
                self.h5_file_handle = h5py.File(
                    self.h5_path, 'r', 
                    swmr=True,
                    libver='latest'
                )
            except (OSError, TypeError):
                # Fallback for systems that don't support SWMR
                self.h5_file_handle = h5py.File(self.h5_path, 'r')
                
            self.worker_id = worker_id
            
            if worker_id >= 0:
                logger.debug(f"Worker {worker_id} opened HDF5 file handle")
        
        return self.h5_file_handle

    def _validate_h5_structure(self) -> None:
        """Validate HDF5 file structure and extract metadata."""
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            num_time_steps = hf[TIME_KEY].shape[1]
            
            if num_time_steps < MIN_TIME_STEPS:
                raise ValueError(
                    f"Profiles must have at least {MIN_TIME_STEPS} time steps, got {num_time_steps}."
                )
            
            # Validate that required variables exist
            missing_vars = self.all_vars - set(hf.keys())
            if missing_vars:
                logger.warning(f"Variables not found in HDF5: {missing_vars}")
            
            self.num_time_steps = num_time_steps
        
        logger.info(f"HDF5 validated: {self.num_time_steps} timesteps per profile")

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Main iterator for streaming data.
        
        Handles multi-worker data splitting, chunk-based reading,
        and in-memory shuffling for randomness.
        
        Yields:
            Tuple of (input tensor, target tensor) for each sample
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
            logger.debug(f"Worker {worker_id} has no indices to process")
            return
        
        logger.debug(
            f"Worker {worker_id}/{num_workers} processing {len(worker_indices)} profiles "
            f"(indices {start_idx}-{end_idx})"
        )
        
        # Shuffle indices for this epoch
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
                # Continue with next chunk instead of failing
                continue

    def _read_chunk_data(
        self, h5_file: h5py.File, chunk_indices: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Efficiently read a chunk of data from HDF5.
        
        Uses sorted indexing for optimal HDF5 performance.
        
        Args:
            h5_file: HDF5 file handle
            chunk_indices: List of profile indices to read
            
        Returns:
            Dictionary of numpy arrays with chunk data
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
                # Read data in sorted order
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
        """
        Generate shuffled samples from a chunk of data.
        
        Args:
            chunk_data: Dictionary of numpy arrays
            chunk_indices: List of profile indices in chunk
            
        Yields:
            Tuple of (input tensor, target tensor) for each sample
        """
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
                
                # Normalize using vectorized operations
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
        """
        Apply normalization using pre-computed parameters for efficiency.
        
        Args:
            raw_vector: Raw numpy array
            vector_type: "input" or "target"
            
        Returns:
            Normalized tensor
        """
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
                # Use larger epsilon for numerical stability
                safe_tensor = torch.clamp(tensor[mask], min=1e-8)
                log_tensor = torch.log10(safe_tensor)
                normalized[mask] = (log_tensor - params["log_means"][mask]) / params["log_stds"][mask]

            if "log-min-max" in params["masks"]:
                mask = params["masks"]["log-min-max"]
                safe_tensor = torch.clamp(tensor[mask], min=1e-8)
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
        """
        Construct raw input vector for a specific time point.
        
        Args:
            profile: Dictionary of profile data
            query_time_idx: Time index to query
            
        Returns:
            Raw input array
        """
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
        """
        Construct raw target vector for a specific time point.
        
        Args:
            profile: Dictionary of profile data
            query_time_idx: Time index
            
        Returns:
            Raw target array
        """
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


def create_dataset(
    h5_path: Union[str, Path],
    indices: List[int],
    species_variables: List[str],
    global_variables: List[str],
    normalization_metadata: Dict[str, Any],
    cache_dataset: bool = False,
    profiles_per_chunk: int = 2048,
    max_memory_gb: float = 2.0,
    raw_data_for_caching: Optional[Dict[str, np.ndarray]] = None,
) -> Union[CachedChemicalDataset, ChemicalDataset]:
    """
    Factory function to create either cached or streaming dataset.
    
    Args:
        h5_path: Path to HDF5 file
        indices: Profile indices to use
        species_variables: List of species variable names
        global_variables: List of global variable names
        normalization_metadata: Pre-calculated normalization statistics
        cache_dataset: Whether to cache entire dataset in memory
        profiles_per_chunk: Number of profiles to read at once (for streaming)
        max_memory_gb: Maximum memory to use per worker (for streaming)
        raw_data_for_caching: Optional pre-loaded data for CachedChemicalDataset
        
    Returns:
        Either CachedChemicalDataset or ChemicalDataset instance
    """
    if cache_dataset:
        logger.info("Creating cached dataset (loading all data into memory)...")
        return CachedChemicalDataset(
            h5_path=h5_path,
            indices=indices,
            species_variables=species_variables,
            global_variables=global_variables,
            normalization_metadata=normalization_metadata,
            raw_data_for_caching=raw_data_for_caching,
        )
    else:
        logger.info("Creating streaming dataset...")
        return ChemicalDataset(
            h5_path=h5_path,
            indices=indices,
            species_variables=species_variables,
            global_variables=global_variables,
            normalization_metadata=normalization_metadata,
            profiles_per_chunk=profiles_per_chunk,
            max_memory_gb=max_memory_gb,
        )


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    Collate function with validation and error handling.
    
    Filters out invalid samples and provides informative warnings.
    
    Args:
        batch: List of (input, target) tuples
        
    Returns:
        Tuple of (input dict, target tensor)
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
        # Check for finite values
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


__all__ = ["ChemicalDataset", "CachedChemicalDataset", "create_dataset", "collate_fn"]