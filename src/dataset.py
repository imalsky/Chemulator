#!/usr/bin/env python3
"""
GPU-optimized dataset implementations for AE-DeepONet training.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from normalizer import NormalizationHelper
from utils import load_json


def check_gpu_memory_available(required_gb: float, device: torch.device) -> bool:
    """Check if sufficient GPU memory is available."""
    if device.type != "cuda":
        return False

    free_mem, total_mem = torch.cuda.mem_get_info(device.index or 0)
    free_gb = free_mem / 1e9

    logger = logging.getLogger(__name__)
    logger.info(f"GPU memory: {free_gb:.1f}GB free, {required_gb:.1f}GB required")

    return free_gb >= required_gb


class GPUSequenceDataset(Dataset):
    """
    GPU-resident dataset for autoencoder pretraining with relative time sampling.
    """

    def __init__(
            self,
            shard_dir: Path,
            split_name: str,
            config: Dict[str, Any],
            device: torch.device,
            norm_stats: Optional[Dict[str, Any]] = None,
            use_relative_time: bool = False,
            window_size: int = 100
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Force GPU usage
        if device.type != "cuda":
            raise RuntimeError(
                "GPUSequenceDataset requires CUDA device. "
                "CPU training not supported for performance reasons."
            )

        self.device = device
        self.shard_dir = Path(shard_dir)
        self.split_dir = self.shard_dir / split_name
        self.split_name = split_name
        self.config = config
        self.use_relative_time = use_relative_time
        self.window_size = window_size

        # Load normalization helper
        if norm_stats is None:
            norm_path = self.shard_dir / "normalization.json"
            norm_stats = load_json(norm_path)

        self.norm_helper = NormalizationHelper(norm_stats, device, config)

        # Load shard index
        shard_index = load_json(self.shard_dir / "shard_index.json")
        self.split_info = shard_index["splits"][split_name]
        self.shards = self.split_info["shards"]

        # Setup dimensions
        data_cfg = config["data"]
        self.species_vars = data_cfg["species_variables"]
        self.global_vars = data_cfg["global_variables"]
        self.time_var = data_cfg["time_variable"]

        # Validate global variables
        expected_globals = data_cfg.get("expected_globals", None)
        if expected_globals and self.global_vars != expected_globals:
            raise ValueError(
                f"Global variables mismatch! Config has {self.global_vars}, "
                f"expected {expected_globals}. Check your data."
            )

        # Preload all data to GPU
        self._preload_to_gpu()

    def _estimate_memory_gb(self) -> float:
        """Estimate memory required in GB."""
        n_total = self.split_info["n_trajectories"]
        n_species = len(self.species_vars)
        n_globals = len(self.global_vars)

        # Estimate based on float32
        M = 100  # Approximate trajectory length
        bytes_per_float = 4

        total_floats = n_total * (
                n_species +  # x0
                n_globals +  # globals
                M +  # times
                M * n_species  # y_mat
        )

        # Add 20% overhead
        total_gb = (total_floats * bytes_per_float * 1.2) / 1e9

        return total_gb

    def _preload_to_gpu(self):
        """Preload all data to GPU memory."""
        self.logger.info(f"Preloading {self.split_name} dataset to GPU...")

        # Estimate memory requirement
        required_gb = self._estimate_memory_gb()

        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for {self.split_name} dataset. "
                "Consider reducing batch size or using smaller data splits."
            )

        self.trajectories = []

        # Load and process all shards
        for shard_info in self.shards:
            shard_path = self.split_dir / shard_info["filename"]

            with np.load(shard_path, allow_pickle=False) as data:
                # Load raw data
                x0 = torch.from_numpy(data["x0"].astype(np.float32))
                globals_vec = torch.from_numpy(data["globals"].astype(np.float32))
                t_vec = torch.from_numpy(data["t_vec"].astype(np.float32))
                y_mat = torch.from_numpy(data["y_mat"].astype(np.float32))

                # Apply normalization
                x0_norm = self.norm_helper.normalize(x0, self.species_vars)
                globals_norm = self.norm_helper.normalize(globals_vec, self.global_vars)

                # Fix time normalization shape
                if t_vec.dim() == 1:
                    # Shared time grid
                    t_norm = self.norm_helper.normalize(
                        t_vec.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)
                    t_norm = t_norm.unsqueeze(0).expand(x0.shape[0], -1)
                else:
                    # Per-trajectory times
                    t_norm = self.norm_helper.normalize(
                        t_vec.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)

                y_norm = self.norm_helper.normalize(y_mat, self.species_vars)

                # Store complete trajectories
                for i in range(x0.shape[0]):
                    traj = {
                        'x0': x0_norm[i].to(self.device),
                        'globals': globals_norm[i].to(self.device),
                        'times': t_norm[i].to(self.device),
                        'y': y_norm[i].to(self.device)
                    }
                    self.trajectories.append(traj)

        # Create sample indices for relative time sampling
        if self.use_relative_time:
            self._create_relative_samples()
        else:
            self._create_standard_samples()

        # Force synchronization
        torch.cuda.synchronize(self.device)

        # Log memory usage
        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(self.trajectories)} trajectories to GPU. "
            f"Memory allocated: {allocated_gb:.2f}GB. "
            f"Created {len(self.samples)} training samples."
        )

    def _create_standard_samples(self):
        """Create standard samples (full trajectories from t=0)."""
        self.samples = []
        for traj in self.trajectories:
            # Standard: predict full trajectory from initial condition
            self.samples.append({
                'inputs': torch.cat([traj['x0'], traj['globals'], traj['times']]),
                'targets': traj['y']
            })

    def _create_relative_samples(self):
        """Create relative time samples (sliding windows)."""
        self.samples = []
        for traj in self.trajectories:
            M = traj['times'].shape[0]

            # Create multiple samples per trajectory with different starting points
            n_windows = max(1, (M - self.window_size) // 10)  # Sample every 10 timesteps

            for start_idx in range(0, M - self.window_size, max(1, (M - self.window_size) // n_windows)):
                end_idx = min(start_idx + self.window_size, M)

                # Extract window
                x_start = traj['y'][start_idx]  # Use state at start_idx as initial condition
                times_window = traj['times'][start_idx:end_idx] - traj['times'][start_idx]  # Relative times
                y_window = traj['y'][start_idx:end_idx]

                self.samples.append({
                    'inputs': torch.cat([x_start, traj['globals'], times_window]),
                    'targets': y_window
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns data directly from GPU memory."""
        sample = self.samples[idx]
        return sample['inputs'], sample['targets']


class GPULatentDataset(Dataset):
    """
    GPU-resident dataset for DeepONet training on latent space.
    """

    def __init__(
            self,
            latent_dir: Path,
            split_name: str,
            config: Dict[str, Any],
            device: torch.device
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Force GPU usage
        if device.type != "cuda":
            raise RuntimeError(
                "GPULatentDataset requires CUDA device. "
                "CPU training not supported for performance reasons."
            )

        self.device = device
        self.config = config
        self.latent_dir = Path(latent_dir)
        self.split_dir = self.latent_dir / split_name
        self.split_name = split_name

        # Load latent shard index
        shard_index = load_json(self.latent_dir / "latent_shard_index.json")
        self.shards = shard_index["splits"][split_name]["shards"]
        self.latent_dim = shard_index.get("latent_dim", config["model"]["latent_dim"])

        # Preload to GPU
        self._preload_to_gpu()

    def _estimate_memory_gb(self) -> float:
        """Estimate memory required for latent data."""
        # Count total samples
        n_total = sum(shard["n_trajectories"] for shard in self.shards)

        # Use actual latent dimension from index
        M = len(self.config["model"].get("trunk_times", [0.25, 0.5, 0.75, 1.0]))
        n_globals = len(self.config["data"]["global_variables"])

        bytes_per_float = 4
        floats_per_sample = (
                self.latent_dim + n_globals +  # inputs
                M * self.latent_dim  # targets
        )

        total_gb = (n_total * floats_per_sample * bytes_per_float * 1.2) / 1e9
        return total_gb

    def _preload_to_gpu(self):
        """Preload all latent data to GPU."""
        self.logger.info(f"Preloading {self.split_name} latent dataset to GPU...")

        # Check memory
        required_gb = self._estimate_memory_gb()

        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for latent dataset. "
                "Consider reducing batch size or using smaller splits."
            )

        all_inputs = []
        all_targets = []

        # Load all shards
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]

            with np.load(shard_path, allow_pickle=False) as data:
                inputs = torch.from_numpy(data["latent_inputs"].astype(np.float32))
                targets = torch.from_numpy(data["latent_targets"].astype(np.float32))

                all_inputs.append(inputs)
                all_targets.append(targets)

        # Concatenate and move to GPU
        self.inputs = torch.cat(all_inputs, dim=0).to(self.device, non_blocking=True)
        self.targets = torch.cat(all_targets, dim=0).to(self.device, non_blocking=True)

        # Force synchronization
        torch.cuda.synchronize(self.device)

        # Log memory usage
        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(self.inputs)} latent trajectories to GPU. "
            f"Memory allocated: {allocated_gb:.2f}GB"
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns latent data directly from GPU memory."""
        return self.inputs[idx], self.targets[idx]


def create_gpu_dataloader(
        dataset: Dataset,
        config: Dict[str, Any],
        shuffle: bool = True,
        **kwargs
) -> DataLoader:
    """Create DataLoader for GPU-resident datasets."""
    batch_size = config["training"]["batch_size"]

    # Force num_workers=0 for GPU datasets
    if hasattr(dataset, 'device') and dataset.device.type == 'cuda':
        if 'num_workers' in kwargs:
            kwargs.pop('num_workers')
        num_workers = 0
        pin_memory = False
    else:
        num_workers = config["training"].get("num_workers", 0)
        pin_memory = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs
    )