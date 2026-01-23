#!/usr/bin/env python3
"""
dataset.py - Rollout dataset for autoregressive training.

Provides time-series windows from pre-normalized trajectory data.
Data is assumed to already be in z-space (log10-standardized) from preprocessing.

Directory Layout (processed_data_dir):
    normalization.json          # Normalization parameters
    train/shard_*.npz          # Training shards
    validation/shard_*.npz     # Validation shards
    test/shard_*.npz           # Test shards (optional)

Each NPZ Shard Contains:
    y_mat   : [N, T, S] float32  # Trajectories in z-space
    globals : [N, G]    float32  # Global parameters in z-space
    t_vec   : [T]       float64  # Shared relative time vector (0, dt, 2dt, ...)

Dataset Returns (per sample):
    y0      : [S]       # Initial state at window start
    dt_norm : scalar    # Normalized time step (constant across all windows)
    y_seq   : [K, S]    # Target states for K rollout steps
    g       : [G]       # Global parameters
"""

from __future__ import annotations

import json
import warnings
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from normalizer import NormalizationHelper


@dataclass
class ShardIndex:
    """Metadata for a single data shard."""

    path: Path
    n: int  # Number of trajectories in this shard


class FlowMapRolloutDataset(Dataset):
    """
    Dataset that samples random time windows from trajectory shards.

    Efficiently loads data from sharded NPZ files with optional preloading
    and an LRU cache for frequently accessed shards.
    """

    def __init__(
        self,
        processed_dir: Path,
        split: str,
        *,
        total_steps: int,
        windows_per_trajectory: int = 1,
        seed: int = 1234,
        preload_to_device: bool = False,
        device: torch.device = torch.device("cpu"),
        storage_dtype: torch.dtype = torch.float32,
        shard_cache_size: int = 2,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            processed_dir: Path to directory containing processed shards.
            split: Data split name ('train', 'validation', 'test').
            total_steps: Number of rollout steps per window (K).
            windows_per_trajectory: Random windows sampled per trajectory per epoch.
            seed: Random seed for window sampling.
            preload_to_device: If True, load all data to device memory.
            device: Target device for preloaded data.
            storage_dtype: Data type for stored tensors.
            shard_cache_size: Number of shards to keep in LRU cache.
        """
        super().__init__()

        # Validate inputs
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if windows_per_trajectory <= 0:
            raise ValueError(
                f"windows_per_trajectory must be positive, got {windows_per_trajectory}"
            )

        self.processed_dir = Path(processed_dir)
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.preload_to_device = bool(preload_to_device)
        self.device = device
        self.storage_dtype = storage_dtype
        self.shard_cache_size = max(1, int(shard_cache_size))

        # Load normalization manifest
        self._load_manifest()

        # Index all shards and validate consistency
        self._index_shards()

        # Validate constant dt and compute normalized dt
        self._validate_and_compute_dt()

        # Setup caching
        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._rng_cache: Dict[int, np.random.Generator] = {}
        self._npz_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )

        if self.preload_to_device:
            self._preload_all()

    def _load_manifest(self) -> None:
        """Load and parse the normalization manifest."""
        manifest_path = self.processed_dir / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run preprocessing first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict = json.load(f)
        self.norm = NormalizationHelper(self.manifest)

    def _index_shards(self) -> None:
        """Index all shards in the split directory."""
        split_dir = self.processed_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        shard_paths = sorted(split_dir.glob("shard_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.npz found in {split_dir}")

        # Get dimensions from first shard
        with np.load(shard_paths[0]) as first:
            y0, g0, t0 = first["y_mat"], first["globals"], first["t_vec"]

        if y0.ndim != 3 or g0.ndim != 2 or t0.ndim != 1:
            raise ValueError("Expected y_mat[N,T,S], globals[N,G], t_vec[T]")

        self.T = int(y0.shape[1])
        self.S = int(y0.shape[2])
        self.G = int(g0.shape[1])
        self.t_vec = np.asarray(t0, dtype=np.float64)

        # Index all shards with validation
        self.shards: List[ShardIndex] = []
        for p in shard_paths:
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float32)
                g = np.asarray(z["globals"], dtype=np.float32)
                t = np.asarray(z["t_vec"], dtype=np.float64)

            if y.shape[1:] != (self.T, self.S):
                raise ValueError(f"Shard {p} has inconsistent y_mat shape")
            if g.shape[1] != self.G:
                raise ValueError(f"Shard {p} has inconsistent globals shape")
            if not np.allclose(t, self.t_vec, rtol=0.0, atol=0.0):
                raise ValueError(f"Shard {p} has different t_vec")

            self.shards.append(ShardIndex(path=p, n=int(y.shape[0])))

        # Build prefix sums for O(log N) shard lookup
        self._shard_offsets = [0]
        for si in self.shards:
            self._shard_offsets.append(self._shard_offsets[-1] + si.n)
        self._total_traj = self._shard_offsets[-1]

    def _validate_and_compute_dt(self) -> None:
        """Validate constant dt and compute normalized dt value."""
        if self.T < 2:
            raise ValueError("t_vec must have at least 2 time points")

        t64 = np.asarray(self.t_vec, dtype=np.float64)
        dt_phys = float(t64[1] - t64[0])
        diffs = np.diff(t64)

        atol = 1e-12 * max(1.0, abs(dt_phys))
        if not np.allclose(diffs, dt_phys, rtol=1e-6, atol=atol):
            raise ValueError("t_vec must have constant dt spacing")

        self.dt_phys = dt_phys
        dt_norm = self.norm.normalize_dt_from_phys(
            torch.tensor([dt_phys], dtype=torch.float64)
        )[0]
        self.dt_norm = dt_norm.to(dtype=self.storage_dtype)

        # Precompute dt on device if preloading
        if self.preload_to_device:
            self.dt_norm = self.dt_norm.to(device=self.device)

        # Compute allowable window start range
        self.max_start = self.T - self.total_steps - 1
        if self.max_start < 0:
            raise ValueError(
                f"T={self.T} too short for total_steps={self.total_steps}. "
                f"Need T >= total_steps + 2."
            )

    def _preload_all(self) -> None:
        """Load all shards into device memory."""
        self._shard_cache.clear()
        for si in self.shards:
            with np.load(si.path) as z:
                y = torch.tensor(
                    z["y_mat"], dtype=self.storage_dtype, device=self.device
                )
                g = torch.tensor(
                    z["globals"], dtype=self.storage_dtype, device=self.device
                )
            self._shard_cache.append((y, g))
        self._preloaded = True

    def _get_shard_arrays(self, shard_i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get shard arrays with LRU caching."""
        if shard_i in self._npz_cache:
            # Move to end (most recently used)
            item = self._npz_cache.pop(shard_i)
            self._npz_cache[shard_i] = item
            return item

        with np.load(self.shards[shard_i].path) as z:
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)

        self._npz_cache[shard_i] = (y_all, g_all)

        # Evict LRU entries
        while len(self._npz_cache) > self.shard_cache_size:
            self._npz_cache.popitem(last=False)

        return y_all, g_all

    def _get_worker_rng(self) -> np.random.Generator:
        """Get or create RNG for current worker, seeded deterministically."""
        info = get_worker_info()
        wid = 0 if info is None else int(info.id)

        if wid not in self._rng_cache:
            # Incorporate torch seed for epoch-dependent randomness
            try:
                torch_seed = torch.initial_seed() % (2**32)
            except RuntimeError:
                torch_seed = 0
            combined_seed = self.seed + 1000 * wid + torch_seed
            self._rng_cache[wid] = np.random.default_rng(combined_seed)

        return self._rng_cache[wid]

    def __len__(self) -> int:
        """Total samples = trajectories x windows_per_trajectory."""
        return self._total_traj * self.windows_per_trajectory

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            y0: [S] initial state
            dt_norm: scalar normalized time step
            y_seq: [K, S] target sequence
            g: [G] global parameters
        """
        rng = self._get_worker_rng()

        # Map index to trajectory
        traj_idx = idx // self.windows_per_trajectory
        if traj_idx < 0 or traj_idx >= self._total_traj:
            raise IndexError(f"Index {idx} out of bounds")

        # Find shard via binary search on prefix sums
        shard_i = bisect_right(self._shard_offsets, traj_idx) - 1
        local_i = traj_idx - self._shard_offsets[shard_i]

        # Random window start
        start = int(rng.integers(0, self.max_start + 1))
        end = start + self.total_steps + 1

        # Load trajectory data
        if self._preloaded:
            y_mat, g_mat = self._shard_cache[shard_i]
            y_traj, g = y_mat[local_i], g_mat[local_i]
        else:
            y_all, g_all = self._get_shard_arrays(shard_i)
            y_np, g_np = y_all[local_i], g_all[local_i]

            if self.storage_dtype == torch.float32:
                y_traj = torch.from_numpy(y_np)
                g = torch.from_numpy(g_np)
            else:
                y_traj = torch.as_tensor(y_np, dtype=self.storage_dtype)
                g = torch.as_tensor(g_np, dtype=self.storage_dtype)

        # Extract window
        y0 = y_traj[start, :]
        y_seq = y_traj[start + 1 : end, :]

        return y0, self.dt_norm, y_seq, g


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    Create DataLoader with smart defaults.

    Args:
        dataset: Source dataset.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle samples.
        num_workers: Parallel data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches to prefetch per worker.
        drop_last: Drop incomplete final batch (auto-determined if None).

    Returns:
        Configured DataLoader instance.

    Raises:
        ValueError: If dataset is empty or configuration is invalid.
    """
    # Validate device compatibility
    preload = getattr(dataset, "preload_to_device", False)
    if preload and num_workers != 0:
        raise ValueError("preload_to_device=True requires num_workers=0")

    # Preloaded data is already on device; pinning is invalid
    if preload and pin_memory:
        warnings.warn(
            "pin_memory=True is incompatible with preload_to_device=True. Disabling.",
            UserWarning,
            stacklevel=2,
        )
        pin_memory = False

    if num_workers == 0:
        persistent_workers = False

    # Check dataset size
    try:
        ds_len = len(dataset)
    except TypeError:
        ds_len = None

    if ds_len is not None and ds_len == 0:
        raise ValueError("Dataset is empty - cannot create DataLoader")

    # Determine drop_last behavior
    if drop_last is None:
        drop_last = shuffle and ds_len is not None and ds_len >= batch_size
    elif drop_last and ds_len is not None and ds_len < batch_size:
        warnings.warn(
            f"drop_last=True with dataset size {ds_len} < batch_size {batch_size} "
            "would yield 0 batches; forcing drop_last=False",
            UserWarning,
            stacklevel=2,
        )
        drop_last = False

    # Build DataLoader kwargs
    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dl_kwargs)
