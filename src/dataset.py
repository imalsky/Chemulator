#!/usr/bin/env python3
"""
GPU-optimized dataset implementations for AE-DeepONet training.
ENHANCED: Added per-trajectory batching option for AE training
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

    idx = device.index if device.index is not None else torch.cuda.current_device()
    free_mem, total_mem = torch.cuda.mem_get_info(idx)
    free_gb = free_mem / 1e9

    logger = logging.getLogger(__name__)
    logger.info(f"GPU memory: {free_gb:.1f}GB free, {required_gb:.1f}GB required")

    return free_gb >= required_gb


class GPUSequenceDataset(Dataset):
    """
    GPU-resident dataset for autoencoder pretraining.
    Enhanced with optional per-trajectory batching for better efficiency.
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

        # Check if we should use per-trajectory batching
        self.ae_per_trajectory = bool(config.get("training", {}).get("ae_per_trajectory", False))

        # Preload all data to GPU efficiently
        self._preload_to_gpu_optimized()

    def _estimate_memory_gb(self) -> float:
        """Estimate memory based on actual data dimensions."""
        # Probe first shard for real dimensions
        if not self.shards:
            raise RuntimeError(f"No shards found for split '{self.split_name}'")

        first_shard = self.split_dir / self.shards[0]["filename"]
        with np.load(first_shard, allow_pickle=False) as d:
            y_shape = d["y_mat"].shape
            if len(y_shape) == 3:
                _, M, S = y_shape
            else:
                raise ValueError(f"Unexpected y_mat shape: {y_shape}")
            G = d["globals"].shape[-1]

            # Check if time is shared or per-sample
            t_shape = d["t_vec"].shape
            has_shared_time = (len(t_shape) == 1)  # [M] means shared, [N, M] means per-sample

        n_total = self.split_info["n_trajectories"]
        bytes_per_float = 4

        # Calculate actual memory needed based on storage strategy
        # We store: ae_flat (all y matrices flattened), all_x0, all_globals, times

        # Flattened y matrices: N * M * S
        y_floats = n_total * M * S

        # Initial conditions and globals: N * S + N * G
        metadata_floats = n_total * (S + G)

        # Time storage depends on whether it's shared or per-sample
        if has_shared_time:
            # Single shared time vector
            time_floats = M
        else:
            # Per-sample time vectors
            time_floats = n_total * M

        # Trajectory offsets (stored as long integers, 8 bytes each)
        offset_bytes = n_total * 8

        # Total memory calculation
        total_floats = y_floats + metadata_floats + time_floats
        total_bytes = (total_floats * bytes_per_float) + offset_bytes

        # Add 20% overhead for PyTorch internals and fragmentation
        total_gb = (total_bytes * 1.2) / 1e9

        self.logger.debug(
            f"Memory estimate: {n_total} trajectories, "
            f"{'shared' if has_shared_time else 'per-sample'} time, "
            f"{total_gb:.2f} GB required"
        )

        return total_gb

    def _preload_to_gpu_optimized(self):
        """Preload data to GPU with optimized memory layout and pinned transfers."""
        self.logger.info(f"Preloading {self.split_name} sequence dataset to GPU...")

        required_gb = self._estimate_memory_gb()
        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for sequence dataset. "
                "Reduce batch size or use smaller splits."
            )

        # Collect all y matrices for flattened storage
        all_y_list = []
        trajectory_info = []
        current_idx = 0

        # For metadata storage (minimal)
        all_x0_list = []
        all_globals_list = []

        # Shared time grid (if applicable)
        self.shared_times = None
        self.per_sample_times = []

        for shard_info in self.shards:
            shard_path = self.split_dir / shard_info["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                # Create pinned CPU tensors for non-blocking transfer
                x0_cpu = torch.from_numpy(data["x0"].astype(np.float32)).pin_memory()
                g_cpu = torch.from_numpy(data["globals"].astype(np.float32)).pin_memory()
                y_cpu = torch.from_numpy(data["y_mat"].astype(np.float32)).pin_memory()
                t_np = data["t_vec"].astype(np.float32)

                N = x0_cpu.shape[0]

                # Move to GPU and normalize there (more efficient)
                x0_gpu = x0_cpu.to(self.device, non_blocking=True)
                g_gpu = g_cpu.to(self.device, non_blocking=True)
                y_gpu = y_cpu.to(self.device, non_blocking=True)

                # Normalize on GPU
                x0_norm = self.norm_helper.normalize(x0_gpu, self.species_vars)
                g_norm = self.norm_helper.normalize(g_gpu, self.global_vars)
                y_norm = self.norm_helper.normalize(y_gpu, self.species_vars)

                # Handle time normalization
                if t_np.ndim == 1:
                    # Shared time grid - only normalize once
                    if self.shared_times is None:
                        t_cpu = torch.from_numpy(t_np).pin_memory()
                        t_gpu = t_cpu.to(self.device, non_blocking=True)
                        self.shared_times = self.norm_helper.normalize(
                            t_gpu.unsqueeze(-1), [self.time_var]
                        ).squeeze(-1)
                    per_sample_times = False
                else:
                    # Per-sample times
                    t_cpu = torch.from_numpy(t_np).pin_memory()
                    t_gpu = t_cpu.to(self.device, non_blocking=True)
                    t_norm = self.norm_helper.normalize(
                        t_gpu.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)
                    per_sample_times = True

                    for n in range(N):
                        self.per_sample_times.append(t_norm[n])

                # Store normalized data
                for n in range(N):
                    all_x0_list.append(x0_norm[n])
                    all_globals_list.append(g_norm[n])
                    all_y_list.append(y_norm[n])  # [M, S]
                    M = y_norm[n].shape[0]
                    trajectory_info.append((current_idx, M))
                    current_idx += M

        # Create single contiguous tensor for all AE samples
        self.ae_flat = torch.cat(all_y_list, dim=0).contiguous()  # [total_timepoints, S]
        self.trajectory_info = torch.tensor(trajectory_info, dtype=torch.long, device=self.device)

        # Store metadata (much smaller)
        self.all_x0 = torch.stack(all_x0_list)
        self.all_globals = torch.stack(all_globals_list)

        # Don't synchronize here - let pipeline continue

        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(all_x0_list)} trajectories ({self.ae_flat.shape[0]} samples) to GPU. "
            f"Memory allocated: {allocated_gb:.2f} GB. "
            f"Per-trajectory mode: {self.ae_per_trajectory}"
        )

    def __len__(self) -> int:
        """Return dataset size based on AE training mode."""
        if self.ae_per_trajectory:
            # Return number of trajectories for per-trajectory batching
            return int(self.trajectory_info.size(0))
        else:
            # Return total number of flattened samples for per-timepoint batching
            return self.ae_flat.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return data based on AE training mode."""
        if self.ae_per_trajectory:
            # Per-trajectory mode: return entire trajectory
            start, length = self.trajectory_info[idx].tolist()
            if length <= 0:
                raise RuntimeError(f"Trajectory {idx} has non-positive length {length}")
            y_mat = self.ae_flat[start:start + length].contiguous()  # [M, S]
            return y_mat, y_mat
        else:
            # Per-timepoint mode: return single sample
            vec = self.ae_flat[idx]  # [S]
            return vec, vec


class GPULatentDataset(Dataset):
    """
    GPU-resident dataset for DeepONet training on latent space.
    """

    def __init__(
            self,
            latent_dir: Path,
            split_name: str,
            config: Dict[str, Any],
            device: torch.device,
            time_sampling_mode: Optional[str] = None,
            n_time_points: Optional[int] = None
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if device.type != "cuda":
            raise RuntimeError("GPULatentDataset requires CUDA device.")
        self.device = device

        self.latent_dir = Path(latent_dir)
        self.split_name = split_name
        self.config = config

        # Load latent index and trunk_times
        latent_index_path = self.latent_dir / "latent_shard_index.json"
        latent_index = load_json(latent_index_path)
        trunk_times_list = latent_index.get("trunk_times")
        if not trunk_times_list:
            raise KeyError(f"'trunk_times' missing in {latent_index_path}")

        self.trunk_times = torch.tensor(trunk_times_list, dtype=torch.float32, device=self.device)
        self.total_time_points = int(self.trunk_times.numel())

        self.logger.info(f"Loaded shared time grid with {self.total_time_points} points")

        # Shards for this split
        split_info = latent_index["splits"][split_name]
        self.shards = split_info["shards"]
        self.split_dir = self.latent_dir / split_name

        # Memory estimate and load
        self._estimate_and_load_memory(split_info)

        # Time sampling configuration
        self._setup_time_sampling(time_sampling_mode, n_time_points)

    def _estimate_and_load_memory(self, split_info: Dict[str, Any]):
        """Estimate memory and load data to GPU efficiently."""
        if not self.shards:
            raise RuntimeError(f"No shards found for split '{self.split_name}'")

        # Probe first shard for dimensions
        first_shard = self.split_dir / self.shards[0]["filename"]
        with np.load(first_shard, allow_pickle=False) as probe:
            li_shape = probe["latent_inputs"].shape
            lt_shape = probe["latent_targets"].shape

        _, in_dim = li_shape
        _, M_total_probe, L_probe = lt_shape

        if M_total_probe != self.total_time_points:
            raise ValueError(
                f"Latent targets time dimension ({M_total_probe}) doesn't match trunk_times length "
                f"({self.total_time_points}). Regenerate latent dataset."
            )

        N_total = int(split_info["n_trajectories"])
        bytes_per_float = 4

        # Accurate memory calculation
        floats_inputs = N_total * in_dim
        floats_targets = N_total * self.total_time_points * L_probe
        est_gb = (floats_inputs + floats_targets) * bytes_per_float * 1.2 / 1e9  # 20% overhead

        if not check_gpu_memory_available(est_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{est_gb:.1f}GB for latent dataset. "
                f"Reduce batch size or regenerate with fewer time points."
            )

        # Load all shards to GPU with pinned memory
        all_inputs = []
        all_targets = []

        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                # Create pinned tensors for non-blocking transfer
                li_cpu = torch.from_numpy(data["latent_inputs"].astype(np.float32)).pin_memory()
                lt_cpu = torch.from_numpy(data["latent_targets"].astype(np.float32)).pin_memory()

                all_inputs.append(li_cpu)
                all_targets.append(lt_cpu)

        # Concatenate on CPU with pinned memory, then single transfer to GPU
        inputs_concat = torch.cat(all_inputs, dim=0)
        targets_concat = torch.cat(all_targets, dim=0)

        self.inputs = inputs_concat.to(self.device, non_blocking=True)
        self.targets = targets_concat.to(self.device, non_blocking=True)

        # Sanity check
        if int(self.targets.shape[1]) != self.total_time_points:
            raise ValueError(
                f"Latent targets time dimension ({self.targets.shape[1]}) != trunk_times length "
                f"({self.total_time_points}). Regenerate latent dataset."
            )

        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(self.inputs)} latent trajectories to GPU. "
            f"Memory allocated: {allocated_gb:.2f} GB. "
            f"Shared time grid: {self.total_time_points} points"
        )

    def _setup_time_sampling(self, time_sampling_mode: Optional[str], n_time_points: Optional[int]):
        """Setup time sampling configuration."""
        train_cfg = self.config["training"]

        if time_sampling_mode is None:
            self.time_sampling_mode = train_cfg.get("train_time_sampling", "random") if self.split_name == "train" \
                else train_cfg.get("val_time_sampling", "fixed")
        else:
            self.time_sampling_mode = time_sampling_mode

        self.randomize_time_points = (self.time_sampling_mode == "random")
        self.fixed_indices = None

        if not self.randomize_time_points:
            # Fixed time sampling
            if self.split_name == "train":
                fixed_times = train_cfg.get("train_fixed_times", None)
                default_count = int(train_cfg.get("train_time_points", 10))
            else:
                fixed_times = train_cfg.get("val_fixed_times", None)
                default_count = int(train_cfg.get("val_time_points", 50))

            # CHANGED: Handle "all" as a special value
            if fixed_times == "all":
                # Use all available time points
                self.fixed_indices = torch.arange(self.total_time_points, device=self.device)
            elif fixed_times is not None and fixed_times != "all":
                # Find nearest neighbor indices for specified times
                req = torch.tensor([float(t) for t in fixed_times], dtype=torch.float32, device=self.device)
                diffs = (self.trunk_times.unsqueeze(0) - req.unsqueeze(1)).abs()
                idx = diffs.argmin(dim=1)
                self.fixed_indices = idx.sort().values
            else:
                # Evenly spaced subset without duplicates
                n_points = int(n_time_points) if n_time_points is not None else default_count
                n_points = max(1, min(n_points, self.total_time_points))

                if n_points == self.total_time_points:
                    self.fixed_indices = torch.arange(self.total_time_points, device=self.device)
                else:
                    # Use linspace and ensure unique
                    self.fixed_indices = torch.linspace(
                        0, self.total_time_points - 1,
                        steps=n_points,
                        device=self.device
                    ).round().long()
                    # Ensure strictly unique and sorted
                    self.fixed_indices = torch.unique(self.fixed_indices, sorted=True)
        else:
            # Random time sampling configuration
            if self.split_name == "train":
                self.min_time_points = int(train_cfg.get("train_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("train_max_time_points", 32))
            else:
                self.min_time_points = int(train_cfg.get("val_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("val_max_time_points", 32))

            # Clamp to availability
            self.max_time_points = max(1, min(self.max_time_points, self.total_time_points))
            self.min_time_points = max(1, min(self.min_time_points, self.max_time_points))

            # Set seed for reproducibility if configured
            seed = self.config.get("system", {}).get("seed", None)
            if seed is not None:
                self.generator = torch.Generator(device=self.device)
                self.generator.manual_seed(seed + hash(self.split_name))
            else:
                self.generator = None

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns data with on-the-fly time sampling."""
        if self.randomize_time_points:
            # Generate indices on the fly (no memory overhead)
            if self.min_time_points == self.max_time_points:
                n_points = self.min_time_points
            else:
                n_points = int(torch.randint(
                    self.min_time_points,
                    self.max_time_points + 1,
                    (1,),
                    device=self.device,
                    generator=self.generator
                ).item())

            # Generate sorted random indices
            indices = torch.randperm(
                self.total_time_points,
                device=self.device,
                generator=self.generator
            )[:n_points].sort().values
        else:
            indices = self.fixed_indices

        # Efficient indexing
        selected_targets = self.targets[idx, indices]
        time_values = self.trunk_times[indices]

        return self.inputs[idx], selected_targets, time_values


def create_gpu_dataloader(
        dataset: Dataset,
        config: Dict[str, Any],
        shuffle: bool = True,
        **kwargs
) -> DataLoader:
    """Create DataLoader for GPU-resident datasets with optimized settings."""
    batch_size = config["training"]["batch_size"]

    # Custom collate function for AE per-trajectory batching
    def ae_collate_fn(batch):
        """Collate for per-trajectory AE training requiring shared M."""
        # batch is list of (y_mat, y_mat) where y_mat is [M, S]
        inputs = [item[0] for item in batch]

        # Verify all have same M dimension
        Ms = [y.shape[0] for y in inputs]
        if len(set(Ms)) != 1:
            raise RuntimeError(
                f"Per-trajectory AE training requires all trajectories in batch to have same time dimension. "
                f"Got M values: {sorted(set(Ms))}. Ensure preprocessing generates consistent time grids."
            )

        # Stack into [B, M, S]
        inputs_tensor = torch.stack(inputs)
        return inputs_tensor, inputs_tensor

    # Custom collate function for variable-length time sequences (latent)
    def latent_collate_fn(batch):
        if len(batch[0]) == 3:  # Latent dataset with time values
            inputs, targets, times = zip(*batch)

            # Stack inputs (all same size)
            inputs_tensor = torch.stack(inputs)

            # Check if all time tensors are identical (fast path)
            if all(torch.equal(times[0], t) for t in times[1:]):
                # Stack targets and return single time tensor
                targets_tensor = torch.stack(targets)
                return inputs_tensor, targets_tensor, times[0]
            else:
                # Variable length: return lists for trainer to handle
                return inputs_tensor, targets, times
        else:
            # Standard dataset
            return torch.utils.data.dataloader.default_collate(batch)

    # Force num_workers=0 for GPU datasets
    if hasattr(dataset, 'device') and dataset.device.type == 'cuda':
        kwargs.pop('num_workers', None)
        num_workers = 0
        pin_memory = False

        # Use appropriate collate function
        if isinstance(dataset, GPUSequenceDataset) and dataset.ae_per_trajectory:
            kwargs['collate_fn'] = ae_collate_fn
        elif hasattr(dataset, 'randomize_time_points'):
            kwargs['collate_fn'] = latent_collate_fn
    else:
        num_workers = config["training"].get("num_workers", 0)
        pin_memory = True

    # Only drop_last for training
    is_validation = not shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(not is_validation),
        **kwargs
    )