"""
dataset.py - Efficient streaming data loader for chemical kinetics data.

Provides streaming dataset implementation that reads data from HDF5 files
in chunks for memory-efficient training.
"""
from __future__ import annotations

import h5py
import logging
import numpy as np
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from normalizer import DataNormalizer

# Constants
TIME_KEY = "t_time"
DTYPE = torch.float32

logger = logging.getLogger(__name__)


class ChemicalDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for chemical reaction profiles.
    
    Reads data in chunks from HDF5 files with proper multi-worker support.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: List[int],
        species_variables: List[str],
        global_variables: List[str],
        normalization_metadata: Dict[str, Any],
        profiles_per_chunk: int,
        config: Dict[str, Any],
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
            config: Full configuration dictionary
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.indices = indices
        self.profiles_per_chunk = profiles_per_chunk
        self.config = config
        
        # Get numerical constants from config
        self.num_constants = config.get("numerical_constants", {})
        self.epsilon = self.num_constants.get("epsilon", 1e-10)
        self.normalized_value_clamp = self.num_constants.get("normalized_value_clamp", 10.0)
        self.min_time_steps = self.num_constants.get("min_time_steps", 2)
        
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
        
        # Calculate total samples
        self.total_samples = len(self.indices) * (self.num_time_steps - 1)

        # Setup vectorized normalization for efficiency
        self._setup_vectorized_normalization()
        
        logger.info(
            f"ChemicalDataset initialized: {len(self.indices)} profiles, "
            f"~{self.total_samples:,} samples, chunk_size={self.profiles_per_chunk}"
        )

    def _validate_h5_structure(self) -> None:
        """Validate HDF5 file structure and extract metadata."""
        with h5py.File(self.h5_path, 'r') as hf:
            if TIME_KEY not in hf:
                raise ValueError(f"HDF5 file must contain dataset '{TIME_KEY}'.")
            
            num_time_steps = hf[TIME_KEY].shape[1]
            
            if num_time_steps < self.min_time_steps:
                raise ValueError(
                    f"Profiles must have at least {self.min_time_steps} time steps, got {num_time_steps}."
                )
            
            # Validate that required variables exist
            missing_vars = self.all_vars - set(hf.keys())
            if missing_vars:
                logger.warning(f"Variables not found in HDF5: {missing_vars}")
            
            self.num_time_steps = num_time_steps
        
        logger.info(f"HDF5 validated: {self.num_time_steps} timesteps per profile")

    def _setup_vectorized_normalization(self) -> None:
        """
        Pre-compute per-variable means / scale factors so that each
        batch can be normalised with simple vector ops.

        Any variable that is **not** listed in `normalization_methods`
        is treated as `"none"` (i.e. passed through unchanged).  This
        prevents KeyErrors when new keys appear in the HDF5 file.
        """
        self.norm_params: Dict[str, Dict[str, Any]] = {"input": {}, "target": {}}
        
        # Create a temporary normalizer instance to access normalize_tensor method
        self.normalizer = DataNormalizer(config_data=self.config)

        for vector_type, var_order in [
            ("input", self.input_var_order),
            ("target", self.target_var_order),
        ]:
            n_vars = len(var_order)

            # Default = identity transform
            means = torch.zeros(n_vars, dtype=DTYPE)
            stds = torch.ones(n_vars, dtype=DTYPE)
            methods: List[str] = []

            for i, var in enumerate(var_order):
                # ---------- safe lookup with fallback -------------------
                method: str = self.norm_metadata["normalization_methods"].get(var, "none")
                methods.append(method)

                # Skip identity / boolean keys early
                if method in ("none", "bool"):
                    continue

                # ---------- stats lookup (may be missing) ---------------
                stats: Dict[str, float] = self.norm_metadata["per_key_stats"].get(var, {})
                if not stats:      # no pre-computed stats ⇒ leave defaults
                    logger.warning("Stats not found for %s; keeping defaults.", var)
                    continue

                # ---------- populate mean / scale based on method -------
                if method == "standard":
                    means[i], stds[i] = stats["mean"], stats["std"]
                elif method == "log-standard":
                    means[i], stds[i] = stats["log_mean"], stats["log_std"]
                elif method == "log-min-max":
                    min_val, max_val = stats["min"], stats["max"]
                    means[i] = min_val
                    stds[i] = max_val - min_val if max_val > min_val else 1.0
                elif method == "max-out":
                    means[i] = 0.0
                    stds[i] = stats["max_val"]
                elif method == "iqr":
                    means[i], stds[i] = stats["median"], stats["iqr"]
                elif method == "signed-log":
                    means[i], stds[i] = stats["mean"], stats["std"]
                elif method == "symlog":
                    # symlog normalisation handled on-the-fly; leave defaults
                    pass
                else:
                    raise ValueError(f"Unknown normalisation method: {method}")

            # ---------- stash tensors & metadata for fast access ----------
            self.norm_params[vector_type] = {
                "means": means,
                "stds": stds,
                "methods": methods,
                "stats_dict": self.norm_metadata["per_key_stats"],
            }

    def _get_h5_handle(self) -> h5py.File:
        """Get HDF5 file handle with proper multi-worker support."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        if self.h5_file_handle is None or self.worker_id != worker_id:
            if self.h5_file_handle is not None:
                self.h5_file_handle.close()
            
            try:
                self.h5_file_handle = h5py.File(
                    self.h5_path, 'r', swmr=True, libver='latest',
                    rdcc_nbytes=100*1024*1024, rdcc_nslots=10007
                )
            except (OSError, TypeError):
                self.h5_file_handle = h5py.File(self.h5_path, 'r')
                
            self.worker_id = worker_id
            if worker_id >= 0:
                logger.debug(f"Worker {worker_id} opened HDF5 file handle")
        
        return self.h5_file_handle

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Main iterator for streaming data."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        
        indices_per_worker = len(self.indices) // num_workers
        extra_indices = len(self.indices) % num_workers
        
        start_idx = worker_id * indices_per_worker + min(worker_id, extra_indices)
        end_idx = start_idx + indices_per_worker + (1 if worker_id < extra_indices else 0)
        
        worker_indices = self.indices[start_idx:end_idx]
        if not worker_indices:
            return iter([])
        
        # Use a reproducible seed for each worker
        random.seed(torch.initial_seed() + worker_id)
        random.shuffle(worker_indices)
        
        h5_file = self._get_h5_handle()
        
        for chunk_start in range(0, len(worker_indices), self.profiles_per_chunk):
            chunk_end = min(chunk_start + self.profiles_per_chunk, len(worker_indices))
            chunk_profile_indices = worker_indices[chunk_start:chunk_end]
            if not chunk_profile_indices:
                continue
            
            try:
                chunk_data = self._read_chunk_data(h5_file, chunk_profile_indices)
                yield from self._generate_samples_from_chunk(chunk_data)
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing chunk: {e}", exc_info=True)
                continue

    def _read_chunk_data(
        self, h5_file: h5py.File, chunk_indices: List[int]
    ) -> Dict[str, np.ndarray]:
        """Efficiently read a chunk of data from HDF5 by sorting indices first."""
        chunk_data = {}
        # Use efficient numpy argsort-based reordering to avoid large copies and python loops
        chunk_indices_np = np.array(chunk_indices, dtype=np.int64)
        sort_order = np.argsort(chunk_indices_np)
        inverse_sort_order = np.argsort(sort_order)
        
        # Read data in sorted order for HDF5 efficiency
        sorted_indices_for_read = chunk_indices_np[sort_order]

        for var in self.all_vars:
            if var not in h5_file:
                continue
            
            try:
                dataset = h5_file[var]
                data_sorted = dataset[list(sorted_indices_for_read)]
                # Reorder the sorted data back to the original shuffled order
                chunk_data[var] = data_sorted[inverse_sort_order]
            except Exception as e:
                logger.error(f"Error reading variable '{var}': {e}")
                raise
        
        return chunk_data

    def _generate_samples_from_chunk(
        self, chunk_data: Dict[str, np.ndarray]
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate shuffled samples from a chunk of data."""
        num_profiles_in_chunk = next(iter(chunk_data.values())).shape[0]
        chunk_samples = []
        for i in range(num_profiles_in_chunk):
            for time_idx in range(1, self.num_time_steps):
                chunk_samples.append((i, time_idx))
        
        random.shuffle(chunk_samples)
        
        for chunk_pos, time_idx in chunk_samples:
            try:
                profile_data = {var: data[chunk_pos] for var, data in chunk_data.items()}
                raw_input = self._build_raw_input(profile_data, time_idx)
                raw_target = self._build_raw_target(profile_data, time_idx)
                input_vector = self._normalize_vector(raw_input, self.input_var_order)
                target_vector = self._normalize_vector(raw_target, self.target_var_order)
                yield input_vector, target_vector
            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue

    def _build_raw_input(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Construct raw input vector for a specific time point."""
        input_data = []
        
        for var in self.species_vars:
            if var not in profile:
                raise KeyError(f"Required species variable '{var}' not found in profile data.")
            input_data.append(profile[var][0])
        
        for var in self.global_vars:
            if var not in profile:
                raise KeyError(f"Required global variable '{var}' not found in profile data.")
            
            value = profile[var]
            if not np.isscalar(value):
                 raise ValueError(f"Global variable '{var}' must be a scalar, but got shape {getattr(value, 'shape', 'N/A')}.")
            input_data.append(value)
        
        if TIME_KEY not in profile:
            raise KeyError(f"Required time key '{TIME_KEY}' not found in profile data.")
        input_data.append(profile[TIME_KEY][query_time_idx])
        
        return np.array(input_data, dtype=np.float32)

    def _build_raw_target(self, profile: Dict[str, np.ndarray], query_time_idx: int) -> np.ndarray:
        """Construct raw target vector for a specific time point."""
        target_data = []
        
        for var in self.species_vars:
            if var not in profile:
                raise KeyError(f"Required target species variable '{var}' not found in profile data.")
            target_data.append(profile[var][query_time_idx])
        
        return np.array(target_data, dtype=np.float32)

    def _normalize_vector(self, raw_vector: np.ndarray, var_order: List[str]) -> Tensor:
        """Apply normalization using pre-computed vectorized operations."""
        tensor = torch.from_numpy(raw_vector).to(dtype=DTYPE)
        
        vector_type = "input" if var_order == self.input_var_order else "target"
        params = self.norm_params[vector_type]
        
        normalized = tensor.clone()
        methods = params["methods"]
        
        for i, method in enumerate(methods):
            var_name = var_order[i]
            stats = params["stats_dict"].get(var_name, {})
            
            # Use the normalizer's normalize_tensor method for consistency
            if method != "none" and stats:
                normalized[i] = self.normalizer.normalize_tensor(
                    tensor[i].unsqueeze(0), method, stats
                ).squeeze(0)
        
        return normalized

    def __del__(self):
        self.close()

    def close(self) -> None:
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
    profiles_per_chunk: int,
    config: Dict[str, Any],
) -> ChemicalDataset:
    """Factory function to create streaming dataset."""
    logger.info("Creating streaming dataset...")
    return ChemicalDataset(
        h5_path=h5_path,
        indices=indices,
        species_variables=species_variables,
        global_variables=global_variables,
        normalization_metadata=normalization_metadata,
        profiles_per_chunk=profiles_per_chunk,
        config=config,
    )


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Optional[Tuple[Dict[str, Tensor], Tensor]]:
    """
    Collate function with validation and error handling.
    If the entire batch is invalid, it returns None.
    """

    valid_batch = [
        sample for sample in batch 
        if sample is not None and 
           isinstance(sample, tuple) and len(sample) == 2 and
           torch.isfinite(sample[0]).all() and torch.isfinite(sample[1]).all()
    ]
    
    if not valid_batch:
        return None
    
    if len(valid_batch) < len(batch):
        logger.debug(f"Dropped {len(batch) - len(valid_batch)} invalid samples from batch")
    
    input_vectors, target_vectors = zip(*valid_batch)
    
    model_inputs = {"x": torch.stack(input_vectors, dim=0)}
    target_batch = torch.stack(target_vectors, dim=0)
    
    return model_inputs, target_batch


__all__ = ["ChemicalDataset", "create_dataset", "collate_fn"]