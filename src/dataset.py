#!/usr/bin/env python3
"""
GPU-optimized dataset implementations for AE-DeepONet training.
FIXED: Race condition in random index generation, division by zero guard
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

    # Ensure we query the intended device (handles device='cuda' with no index)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    free_mem, total_mem = torch.cuda.mem_get_info(idx)
    free_gb = free_mem / 1e9

    logger = logging.getLogger(__name__)
    logger.info(f"GPU memory: {free_gb:.1f}GB free, {required_gb:.1f}GB required")

    return free_gb >= required_gb


class GPUSequenceDataset(Dataset):
    """GPU-resident dataset for autoencoder pretraining with efficient memory usage."""

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

        if norm_stats is None:
            norm_path = self.shard_dir / "normalization.json"
            norm_stats = load_json(norm_path)

        self.norm_helper = NormalizationHelper(norm_stats, device, config)

        shard_index = load_json(self.shard_dir / "shard_index.json")
        self.split_info = shard_index["splits"][split_name]
        self.shards = self.split_info["shards"]

        data_cfg = config["data"]
        self.species_vars = data_cfg["species_variables"]
        self.global_vars = data_cfg["global_variables"]
        self.time_var = data_cfg["time_variable"]

        expected_globals = data_cfg.get("expected_globals", None)
        if expected_globals and self.global_vars != expected_globals:
            raise ValueError(
                f"Global variables mismatch! Config has {self.global_vars}, "
                f"expected {expected_globals}. Check your data."
            )

        self._preload_to_gpu()

    def _estimate_memory_gb(self) -> float:
        """Accurate memory estimation including all overheads."""
        n_total = self.split_info["n_trajectories"]
        n_species = len(self.species_vars)
        n_globals = len(self.global_vars)

        M = 100
        bytes_per_float = 4

        data_bytes = n_total * (
                n_species +
                n_globals +
                M +
                M * n_species
        ) * bytes_per_float

        pytorch_overhead = 1.3

        if self.use_relative_time:
            max_samples = n_total * M * 2
        else:
            max_samples = n_total * M
        index_bytes = max_samples * 8

        loading_overhead = 1.2

        total_data_gb = (data_bytes + index_bytes) * pytorch_overhead * loading_overhead / 1e9

        model_reserve_gb = 3.0

        return total_data_gb + model_reserve_gb

    def _preload_to_gpu(self):
        """Preload data to GPU using efficient contiguous tensor storage."""
        self.logger.info(f"Preloading {self.split_name} sequence dataset to GPU...")

        required_gb = self._estimate_memory_gb()
        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for sequence dataset. "
                "Reduce batch size or use smaller splits."
            )

        all_x0_list = []
        all_globals_list = []
        all_times_list = []
        all_y_list = []
        trajectory_info = []

        current_idx = 0

        for shard_info in self.shards:
            shard_path = self.split_dir / shard_info["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                x0_np = data["x0"].astype(np.float32)
                g_np = data["globals"].astype(np.float32)
                t_np = data["t_vec"].astype(np.float32)
                y_np = data["y_mat"].astype(np.float32)

                N = x0_np.shape[0]

                if t_np.ndim == 1:
                    M = t_np.shape[0]
                    per_sample_times = False
                elif t_np.ndim == 2:
                    N2, M = t_np.shape
                    if N2 != N:
                        raise ValueError(f"Time dimension mismatch: {N2} vs {N}")
                    per_sample_times = True
                else:
                    raise ValueError(f"Invalid time shape: {t_np.shape}")

                x0_t = torch.from_numpy(x0_np)
                y_t = torch.from_numpy(y_np)
                g_t = torch.from_numpy(g_np)

                x0_norm = self.norm_helper.normalize(x0_t, self.species_vars)
                y_norm = self.norm_helper.normalize(y_t, self.species_vars)
                g_norm = self.norm_helper.normalize(g_t, self.global_vars)

                if not per_sample_times:
                    t_shared = torch.from_numpy(t_np)
                    t_shared_norm = self.norm_helper.normalize(
                        t_shared.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)

                    for n in range(N):
                        all_x0_list.append(x0_norm[n])
                        all_globals_list.append(g_norm[n])
                        all_times_list.append(t_shared_norm)
                        all_y_list.append(y_norm[n])
                        trajectory_info.append((current_idx, M))
                        current_idx += M
                else:
                    t_all = torch.from_numpy(t_np)
                    t_all_norm = self.norm_helper.normalize(
                        t_all.unsqueeze(-1), [self.time_var]
                    ).squeeze(-1)

                    for n in range(N):
                        all_x0_list.append(x0_norm[n])
                        all_globals_list.append(g_norm[n])
                        all_times_list.append(t_all_norm[n])
                        all_y_list.append(y_norm[n])
                        trajectory_info.append((current_idx, M))
                        current_idx += M

        self.all_x0 = torch.stack(all_x0_list).to(self.device, non_blocking=True)
        self.all_globals = torch.stack(all_globals_list).to(self.device, non_blocking=True)
        self.all_y_flat = torch.cat(all_y_list, dim=0).to(self.device, non_blocking=True)
        self.trajectory_info = torch.tensor(trajectory_info, dtype=torch.long, device=self.device)

        if len(set(t.shape[0] for t in all_times_list)) == 1:
            self.all_times = torch.stack(all_times_list).to(self.device, non_blocking=True)
        else:
            self.all_times = [t.to(self.device, non_blocking=True) for t in all_times_list]

        torch.cuda.synchronize(self.device)

        self._create_sample_indices()

        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(all_x0_list)} trajectories ({self.n_samples} samples) to GPU. "
            f"Memory allocated: {allocated_gb:.2f} GB"
        )

    def _create_sample_indices(self):
        if self.use_relative_time:
            self._create_relative_indices()
        else:
            self._create_standard_indices()

    def _create_standard_indices(self):
        self.n_samples = self.all_y_flat.shape[0]

    def _create_relative_indices(self):
        indices = []

        # Convert trajectory_info to CPU ints to avoid tensor/int mixing
        traj_info_cpu = self.trajectory_info.detach().cpu().tolist()
        for traj_idx, (start_idx, length) in enumerate(traj_info_cpu):
            if length <= 0:
                continue

            step = max(1, min(10, length // 10))

            for window_start in range(0, max(1, length - self.window_size + 1), step):
                window_end = min(length, window_start + self.window_size)
                for local_idx in range(window_start, window_end):
                    global_idx = start_idx + local_idx
                    indices.append(global_idx)

        self.sample_indices = torch.tensor(indices, dtype=torch.long, device=self.device) if indices else \
            torch.empty(0, dtype=torch.long, device=self.device)
        self.n_samples = len(indices)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_relative_time:
            actual_idx = self.sample_indices[idx]
        else:
            actual_idx = idx

        y_vec = self.all_y_flat[actual_idx]

        return y_vec, y_vec


class GPULatentDataset(Dataset):
    """GPU-resident dataset for DeepONet training on latent space."""

    def __init__(
            self,
            latent_dir: Path,
            split_name: str,
            config: Dict[str, Any],
            device: torch.device,
            time_sampling_mode: Optional[str] = None,
            n_time_points: Optional[int] = None,
            use_fp16_targets: bool = True
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if device.type != "cuda":
            raise RuntimeError("GPULatentDataset requires CUDA device.")
        self.device = device
        self.use_fp16_targets = use_fp16_targets

        self.latent_dir = Path(latent_dir)
        self.split_name = split_name
        self.config = config

        latent_index_path = self.latent_dir / "latent_shard_index.json"
        latent_index = load_json(latent_index_path)
        trunk_times_list = latent_index.get("trunk_times")
        if not trunk_times_list:
            raise KeyError(f"'trunk_times' missing in {latent_index_path}")

        self.trunk_times = torch.tensor(trunk_times_list, dtype=torch.float32, device=self.device)
        self.total_time_points = int(self.trunk_times.numel())

        self.logger.info(f"Loaded shared time grid with {self.total_time_points} points")

        split_info = latent_index["splits"][split_name]
        self.shards = split_info["shards"]
        self.split_dir = self.latent_dir / split_name
        self.n_trajectories = split_info["n_trajectories"]

        self._estimate_and_validate_memory(split_info)
        self._load_to_gpu()
        self._setup_time_sampling(time_sampling_mode, n_time_points)

        if self.randomize_time_points:
            self._pregenerate_random_indices()

    def _estimate_and_validate_memory(self, split_info: Dict[str, Any]):
        if not self.shards:
            raise RuntimeError(f"No shards found for split '{self.split_name}'")

        first_shard = self.split_dir / self.shards[0]["filename"]
        with np.load(first_shard, allow_pickle=False) as probe:
            li_shape = probe["latent_inputs"].shape
            lt_shape = probe["latent_targets"].shape

        _, in_dim = li_shape
        _, M_total_probe, L_probe = lt_shape

        if M_total_probe != self.total_time_points:
            raise ValueError(
                f"Time dimension mismatch: {M_total_probe} vs {self.total_time_points}"
            )

        self.latent_dim = L_probe
        self.input_dim = in_dim

        N_total = int(split_info["n_trajectories"])

        target_bytes_per_element = 2 if self.use_fp16_targets else 4
        input_bytes_per_element = 4

        data_bytes = (
                N_total * in_dim * input_bytes_per_element +
                N_total * self.total_time_points * L_probe * target_bytes_per_element
        )

        pytorch_overhead = 1.3
        model_reserve_gb = 3.0

        if self.split_name == "train":
            index_bank_gb = 0.1
        else:
            index_bank_gb = 0.0

        required_gb = (data_bytes * pytorch_overhead / 1e9) + model_reserve_gb + index_bank_gb

        if not check_gpu_memory_available(required_gb, self.device):
            raise RuntimeError(
                f"Insufficient GPU memory. Need ~{required_gb:.1f}GB for latent dataset. "
                f"Consider enabling FP16 targets or reducing batch size."
            )

        self.logger.info(
            f"Memory estimate: {required_gb:.1f}GB "
            f"(using {'FP16' if self.use_fp16_targets else 'FP32'} for targets)"
        )

    def _load_to_gpu(self):
        all_inputs = []
        all_targets = []

        for shard in self.shards:
            shard_path = self.split_dir / shard["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                li = torch.from_numpy(data["latent_inputs"].astype(np.float32))

                if self.use_fp16_targets:
                    lt = torch.from_numpy(data["latent_targets"].astype(np.float16))
                else:
                    lt = torch.from_numpy(data["latent_targets"].astype(np.float32))

                all_inputs.append(li)
                all_targets.append(lt)

        self.inputs = torch.cat(all_inputs, dim=0).to(self.device, non_blocking=True)
        self.targets = torch.cat(all_targets, dim=0).to(self.device, non_blocking=True)

        if int(self.targets.shape[1]) != self.total_time_points:
            raise ValueError(
                f"Targets time dimension ({self.targets.shape[1]}) != trunk_times ({self.total_time_points})"
            )

        torch.cuda.synchronize(self.device)

        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        self.logger.info(
            f"Loaded {len(self.inputs)} latent trajectories to GPU. "
            f"Memory allocated: {allocated_gb:.2f} GB "
            f"(targets: {'FP16' if self.use_fp16_targets else 'FP32'})"
        )

    def _setup_time_sampling(self, time_sampling_mode: Optional[str], n_time_points: Optional[int]):
        train_cfg = self.config["training"]

        if time_sampling_mode is None:
            if self.split_name == "train":
                self.time_sampling_mode = train_cfg.get("train_time_sampling", "random")
            else:
                self.time_sampling_mode = train_cfg.get("val_time_sampling", "fixed")
        else:
            self.time_sampling_mode = time_sampling_mode

        self.randomize_time_points = (self.time_sampling_mode == "random")
        self.fixed_indices = None

        if not self.randomize_time_points:
            if self.split_name == "train":
                fixed_times = train_cfg.get("train_fixed_times", None)
                default_count = int(train_cfg.get("train_time_points", 10))
            else:
                fixed_times = train_cfg.get("val_fixed_times", None)
                default_count = int(train_cfg.get("val_time_points", 50))

            if fixed_times is not None:
                req = torch.tensor([float(t) for t in fixed_times], dtype=torch.float32, device=self.device)
                diffs = (self.trunk_times.unsqueeze(0) - req.unsqueeze(1)).abs()
                idx = diffs.argmin(dim=1)
                self.fixed_indices = idx.sort().values
            else:
                n_points = int(n_time_points) if n_time_points is not None else default_count
                n_points = max(1, min(n_points, self.total_time_points))

                if n_points == self.total_time_points:
                    self.fixed_indices = torch.arange(self.total_time_points, device=self.device)
                elif n_points == 1:
                    # FIXED: Guard against division by zero
                    self.fixed_indices = torch.tensor([self.total_time_points // 2], device=self.device)
                else:
                    step = (self.total_time_points - 1) / float(n_points - 1)
                    self.fixed_indices = torch.round(torch.arange(n_points, device=self.device) * step).long()

            self.logger.info(f"Fixed time sampling: {len(self.fixed_indices)} points")
        else:
            if self.split_name == "train":
                self.min_time_points = int(train_cfg.get("train_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("train_max_time_points", 32))
            else:
                self.min_time_points = int(train_cfg.get("val_min_time_points", 8))
                self.max_time_points = int(train_cfg.get("val_max_time_points", 32))

            self.max_time_points = max(1, min(self.max_time_points, self.total_time_points))
            self.min_time_points = max(1, min(self.min_time_points, self.max_time_points))

            self.logger.info(
                f"Random time sampling: [{self.min_time_points}, {self.max_time_points}] points"
            )

    def _pregenerate_random_indices(self, n_batches_per_epoch: Optional[int] = None):
        if n_batches_per_epoch is None:
            batch_size = self.config["training"].get("batch_size", 128)
            n_batches_per_epoch = max(100, (self.n_trajectories + batch_size - 1) // batch_size * 2)

        self.logger.info(f"Pre-generating {n_batches_per_epoch} random index sets on CPU")

        self.random_index_bank = []
        self.sample_n_points = []

        cpu_generator = torch.Generator(device='cpu')
        cpu_generator.manual_seed(self.config.get("system", {}).get("seed", 42) + hash(self.split_name))

        for _ in range(n_batches_per_epoch * self.config["training"].get("batch_size", 128)):
            if self.min_time_points == self.max_time_points:
                n_points = self.min_time_points
            else:
                n_points = int(torch.randint(
                    self.min_time_points,
                    self.max_time_points + 1,
                    (1,),
                    generator=cpu_generator,
                    device='cpu'
                ).item())

            indices = torch.randperm(
                self.total_time_points,
                generator=cpu_generator,
                device='cpu'
            )[:n_points].sort().values

            self.random_index_bank.append(indices)
            self.sample_n_points.append(n_points)

        self.random_index_counter = 0
        self.logger.info(f"Pre-generated {len(self.random_index_bank)} random index sets")

    def reset_random_indices(self):
        """FIXED: Reset properly at epoch boundaries."""
        if self.randomize_time_points and hasattr(self, 'random_index_bank'):
            self.random_index_counter = 0
            # Shuffle the bank for variety across epochs
            import random
            random.shuffle(self.random_index_bank)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FIXED: Proper modular arithmetic for random index access."""
        if self.randomize_time_points:
            # Use modular arithmetic to wrap around the bank
            bank_idx = (idx + self.random_index_counter) % len(self.random_index_bank)
            indices = self.random_index_bank[bank_idx]

            if indices.device != self.device:
                indices = indices.to(self.device, non_blocking=True)
        else:
            indices = self.fixed_indices

        selected_targets = self.targets[idx, indices]

        if selected_targets.dtype == torch.float16:
            selected_targets = selected_targets.float()

        time_values = self.trunk_times[indices]

        return self.inputs[idx], selected_targets, time_values

    def get_memory_usage(self) -> Dict[str, float]:
        input_bytes = self.inputs.element_size() * self.inputs.numel()
        target_bytes = self.targets.element_size() * self.targets.numel()
        time_bytes = self.trunk_times.element_size() * self.trunk_times.numel()

        if self.randomize_time_points and hasattr(self, 'random_index_bank'):
            index_bytes = sum(
                idx.element_size() * idx.numel()
                for idx in self.random_index_bank[:100]
            ) * len(self.random_index_bank) / 100
        else:
            index_bytes = 0 if self.fixed_indices is None else \
                self.fixed_indices.element_size() * self.fixed_indices.numel()

        return {
            "inputs_gb": input_bytes / 1e9,
            "targets_gb": target_bytes / 1e9,
            "times_gb": time_bytes / 1e9,
            "indices_gb": index_bytes / 1e9,
            "total_gb": (input_bytes + target_bytes + time_bytes + index_bytes) / 1e9,
            "target_dtype": str(self.targets.dtype),
            "input_dtype": str(self.inputs.dtype)
        }


def create_gpu_dataloader(
        dataset: Dataset,
        config: Dict[str, Any],
        shuffle: bool = True,
        **kwargs
) -> DataLoader:
    """Create DataLoader for GPU-resident datasets."""
    batch_size = config["training"]["batch_size"]

    def collate_fn(batch):
        if len(batch[0]) == 3:
            inputs, targets, times = zip(*batch)
            inputs_tensor = torch.stack(inputs)
            return inputs_tensor, targets, times
        else:
            return torch.utils.data.dataloader.default_collate(batch)

    if hasattr(dataset, 'device') and dataset.device.type == 'cuda':
        if 'num_workers' in kwargs:
            kwargs.pop('num_workers')
        num_workers = 0
        pin_memory = False

        if isinstance(dataset, GPULatentDataset):
            kwargs['collate_fn'] = collate_fn
    else:
        num_workers = config["training"].get("num_workers", 0)
        pin_memory = True

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