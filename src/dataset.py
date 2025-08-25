#!/usr/bin/env python3
"""
GPU-optimized dataset implementations for AE-DeepONet training.
UPDATED: Support for flexible time point sampling during training.
UPDATED: Fixed drop_last for validation; clarified shared time grid requirement.
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
        """Preload sequence (AE-pretrain) data to GPU and build samples (all NORMALIZED)."""
        self.logger.info(f"Preloading {self.split_name} sequence dataset to GPU...")

        required_gb = self._estimate_memory_gb()
        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for sequence dataset. "
                "Reduce batch size or use smaller splits."
            )

        self.trajectories = []

        # Load each shard of the split
        for shard_info in self.shards:
            shard_path = self.split_dir / shard_info["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                # Expected keys from preprocessor: "x0", "globals", "t_vec", "y_mat"
                x0_np = data["x0"].astype(np.float32)  # [N, S]
                g_np = data["globals"].astype(np.float32)  # [N, G]
                t_np = data["t_vec"].astype(np.float32)  # [M] or [N, M]
                y_np = data["y_mat"].astype(np.float32)  # [N, M, S]

                N = x0_np.shape[0]
                if y_np.shape[0] != N:
                    raise ValueError(f"'y_mat' first dim must equal N; got {y_np.shape[0]} vs {N}")

                # Detect time layout
                if t_np.ndim == 1:
                    M = t_np.shape[0]
                    per_sample_times = False
                elif t_np.ndim == 2:
                    N2, M = t_np.shape
                    if N2 != N:
                        raise ValueError(f"'t_vec' 2D must be [N, M]; got {t_np.shape}, expected {(N, M)}")
                    per_sample_times = True
                else:
                    raise ValueError(f"'t_vec' must be 1D [M] or 2D [N, M]; got shape {t_np.shape}")

                # ---- Normalize everything on the configured device ----
                # Species: x0 and trajectory y(t)
                x0_t = torch.from_numpy(x0_np)
                y_t = torch.from_numpy(y_np)  # [N, M, S]
                x0_norm = self.norm_helper.normalize(x0_t, self.species_vars)  # [N, S]
                y_norm = self.norm_helper.normalize(y_t, self.species_vars)  # [N, M, S]

                # Globals: [P, T]
                g_t = torch.from_numpy(g_np)
                g_norm = self.norm_helper.normalize(g_t, self.global_vars)  # [N, G]

                # Time: normalize with method configured for time_variable
                if not per_sample_times:
                    t_shared = torch.from_numpy(t_np)  # [M]
                    t_shared_norm = self.norm_helper.normalize(
                        t_shared.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)  # [M] in [0,1]
                    for n in range(N):
                        self.trajectories.append({
                            "x0": x0_norm[n].to(self.device, non_blocking=True),  # [S]
                            "globals": g_norm[n].to(self.device, non_blocking=True),  # [G]
                            "times": t_shared_norm.to(self.device, non_blocking=True),  # [M]
                            "y": y_norm[n].to(self.device, non_blocking=True),  # [M, S]
                        })
                else:
                    t_all = torch.from_numpy(t_np)  # [N, M]
                    t_all_norm = self.norm_helper.normalize(
                        t_all.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)  # [N, M] in [0,1]
                    for n in range(N):
                        self.trajectories.append({
                            "x0": x0_norm[n].to(self.device, non_blocking=True),
                            "globals": g_norm[n].to(self.device, non_blocking=True),
                            "times": t_all_norm[n].to(self.device, non_blocking=True),
                            "y": y_norm[n].to(self.device, non_blocking=True),
                        })

        # Build samples
        if self.use_relative_time:
            self._create_relative_samples()
        else:
            self._create_standard_samples()

    def _create_standard_samples(self):
        """Create AE-pretrain samples: reconstruct species vectors y(t) from themselves (normalized)."""
        self.samples = []
        for traj in self.trajectories:
            y_mat = traj['y']  # [M, S], already normalized
            M = y_mat.shape[0]
            # One sample per timepoint: (y(t), y(t))
            for m in range(M):
                y_vec = y_mat[m]  # [S]
                self.samples.append({
                    'inputs': y_vec,  # AE input: species vector
                    'targets': y_vec  # AE target: same vector
                })

    def _create_relative_samples(self):
        """Create AE-pretrain samples from windows but still as vector→vector reconstruction."""
        self.samples = []
        for traj in self.trajectories:
            y_mat = traj['y']  # [M, S], normalized
            M = y_mat.shape[0]
            if M <= 0:
                continue

            # Choose a step so we don't explode the count; sample roughly every 10 steps.
            step = max(1, min(10, M // max(1, M // 10)))
            # Use sliding windows to *subsample* time indices; still vector→vector
            for start_idx in range(0, max(1, M - self.window_size + 1), step):
                end_idx = min(M, start_idx + self.window_size)
                for m in range(start_idx, end_idx):
                    y_vec = y_mat[m]  # [S]
                    self.samples.append({
                        'inputs': y_vec,
                        'targets': y_vec
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

    IMPORTANT: This dataset assumes all trajectories share an identical time grid.
    The latent generation stage enforces this requirement. If you have per-sample
    time grids, the latent generation will fail with an error.

    Supports fixed/random time point sampling over ALL available times saved in the latent set.
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

        # Load latent index and trunk_times (the ACTUAL normalized time grid used)
        latent_index_path = self.latent_dir / "latent_shard_index.json"
        latent_index = load_json(latent_index_path)
        trunk_times_list = latent_index.get("trunk_times")
        if not trunk_times_list:
            raise KeyError(f"'trunk_times' missing or empty in {latent_index_path}")

        # IMPORTANT: This is the shared time grid for ALL trajectories
        self.trunk_times = torch.tensor(trunk_times_list, dtype=torch.float32, device=self.device)  # [M_total]
        self.total_time_points = int(self.trunk_times.numel())

        self.logger.info(f"Loaded shared time grid with {self.total_time_points} points")

        # Shards for this split
        split_info = latent_index["splits"][split_name]
        self.shards = split_info["shards"]
        self.split_dir = self.latent_dir / split_name

        # ---------- Memory estimate (based on latent shapes) ----------
        if not self.shards:
            raise RuntimeError(f"No shards found for split '{split_name}' in latent dataset.")
        first_shard = self.split_dir / self.shards[0]["filename"]
        with np.load(first_shard, allow_pickle=False) as probe:
            li_shape = probe["latent_inputs"].shape  # [N_probe, L+G]
            lt_shape = probe["latent_targets"].shape  # [N_probe, M_total, L]
        _, in_dim = li_shape
        _, M_total_probe, L_probe = lt_shape
        if M_total_probe != self.total_time_points:
            raise ValueError(
                f"Latent targets time dimension ({M_total_probe}) doesn't match trunk_times length "
                f"({self.total_time_points}). Regenerate latent dataset."
            )

        N_total = int(split_info["n_trajectories"])
        bytes_per_float = 4
        floats_inputs = N_total * in_dim
        floats_targets = N_total * self.total_time_points * L_probe
        est_gb = (floats_inputs + floats_targets) * bytes_per_float * 1.2 / 1e9  # 20% overhead

        if not check_gpu_memory_available(est_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{est_gb:.1f}GB for latent dataset. "
                f"Reduce batch size or regenerate with fewer time points."
            )

        # ---------- Load all shards to GPU ----------
        all_inputs, all_targets = [], []
        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                li = torch.from_numpy(data["latent_inputs"].astype(np.float32))  # [N, L+G]
                lt = torch.from_numpy(data["latent_targets"].astype(np.float32))  # [N, M_total, L]
                all_inputs.append(li)
                all_targets.append(lt)

        self.inputs = torch.cat(all_inputs, dim=0).to(self.device, non_blocking=True)  # [N_tot, L+G]
        self.targets = torch.cat(all_targets, dim=0).to(self.device, non_blocking=True)  # [N_tot, M,   L]

        # Sanity: M must match trunk_times
        if int(self.targets.shape[1]) != self.total_time_points:
            raise ValueError(
                f"Latent targets time dimension ({self.targets.shape[1]}) != trunk_times length "
                f"({self.total_time_points}). Regenerate latent dataset."
            )

        torch.cuda.synchronize(self.device)
        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(self.inputs)} latent trajectories to GPU. "
            f"Memory allocated: {allocated_gb:.2f} GB. "
            f"Shared time grid: {self.total_time_points} points"
        )

        # ---------- Time sampling configuration ----------
        train_cfg = self.config["training"]
        if time_sampling_mode is None:
            self.time_sampling_mode = train_cfg.get("train_time_sampling", "random") if split_name == "train" \
                else train_cfg.get("val_time_sampling", "fixed")
        else:
            self.time_sampling_mode = time_sampling_mode

        self.randomize_time_points = (self.time_sampling_mode == "random")
        self.fixed_indices = None

        if not self.randomize_time_points:
            # Prefer explicit fixed time lists
            if split_name == "train":
                fixed_times = train_cfg.get("train_fixed_times", None)
                default_count = int(train_cfg.get("train_time_points", 10))
            else:
                fixed_times = train_cfg.get("val_fixed_times", None)
                default_count = int(train_cfg.get("val_time_points", 50))

            if fixed_times is not None:
                req = torch.tensor([float(t) for t in fixed_times], dtype=torch.float32, device=self.device)  # [K]
                diffs = (self.trunk_times.unsqueeze(0) - req.unsqueeze(1)).abs()  # [K, M_total]
                idx = diffs.argmin(dim=1)  # [K]
                self.fixed_indices = idx.sort().values
            else:
                # Evenly spaced subset by count
                n_points = int(n_time_points) if n_time_points is not None else default_count
                n_points = max(1, min(n_points, self.total_time_points))
                if n_points == self.total_time_points:
                    self.fixed_indices = torch.arange(self.total_time_points, device=self.device)
                else:
                    step = (self.total_time_points - 1) / float(n_points - 1)
                    self.fixed_indices = torch.round(torch.arange(n_points, device=self.device) * step).long()
        else:
            # Random time sampling configuration
            if split_name == "train":
                self.min_time_points = int(train_cfg.get("train_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("train_max_time_points", 32))
            else:
                self.min_time_points = int(train_cfg.get("val_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("val_max_time_points", 32))
            # Clamp to availability
            self.max_time_points = max(1, min(self.max_time_points, self.total_time_points))
            self.min_time_points = max(1, min(self.min_time_points, self.max_time_points))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs  : [L+G]
            targets : [M_i, L]    (selected latent targets)
            times   : [M_i]       (the actual normalized trunk times from shared grid)
        """
        if self.randomize_time_points:
            if self.min_time_points == self.max_time_points:
                n_points = self.min_time_points
            else:
                n_points = int(torch.randint(self.min_time_points,
                                             self.max_time_points + 1,
                                             (1,), device=self.device).item())
            # Unique, sorted random indices across ALL available M
            indices = torch.randperm(self.total_time_points, device=self.device)[:n_points].sort().values
        else:
            indices = self.fixed_indices
            n_points = int(indices.numel())

        selected_targets = self.targets[idx, indices]  # [M_i, L]
        time_values = self.trunk_times[indices]  # [M_i] from shared grid
        return self.inputs[idx], selected_targets, time_values

def create_gpu_dataloader(
        dataset: Dataset,
        config: Dict[str, Any],
        shuffle: bool = True,
        **kwargs
) -> DataLoader:
    """Create DataLoader for GPU-resident datasets.

    UPDATED: Fixed drop_last behavior - only drop for training, not validation.
    """
    batch_size = config["training"]["batch_size"]

    # Custom collate function for variable-length time sequences
    def collate_fn(batch):
        if len(batch[0]) == 3:  # Latent dataset with time values
            inputs, targets, times = zip(*batch)

            # Stack inputs (all same size)
            inputs_tensor = torch.stack(inputs)

            # For variable length targets/times, we need to handle differently
            # Option 1: Pad to max length in batch (not ideal for DeepONet)
            # Option 2: Keep them separate and handle in training loop
            # We'll go with option 2 - return lists

            return inputs_tensor, targets, times
        else:
            # Standard dataset
            return torch.utils.data.dataloader.default_collate(batch)

    # Force num_workers=0 for GPU datasets
    if hasattr(dataset, 'device') and dataset.device.type == 'cuda':
        if 'num_workers' in kwargs:
            kwargs.pop('num_workers')
        num_workers = 0
        pin_memory = False

        # Use custom collate for latent dataset
        if hasattr(dataset, 'randomize_time_points'):
            kwargs['collate_fn'] = collate_fn
    else:
        num_workers = config["training"].get("num_workers", 0)
        pin_memory = True

    # FIXED: Only drop_last for training, not validation
    # Determine if this is validation based on shuffle=False (validation never shuffles)
    is_validation = not shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(not is_validation),  # Only drop for training
        **kwargs
    )