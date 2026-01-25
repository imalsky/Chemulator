#!/usr/bin/env python3
"""
dataset.py - Rollout dataset for autoregressive training.

This module provides the FlowMapRolloutDataset class for loading and serving
time-series windows from pre-normalized trajectory data. The data is expected
to already be in z-space (normalized) from preprocessing.

Data Format:
    The processed data directory should contain:
        normalization.json          # Normalization manifest
        train/shard_*.npz          # Training shards
        validation/shard_*.npz     # Validation shards
        test/shard_*.npz           # Test shards (optional)

    Each NPZ shard contains:
        y_mat       : [N, T, S]    float32  # z-space trajectories
        globals     : [N, G]       float32  # z-space globals
        dt_norm_mat : [N, T-1]     float32  # normalized dt per step

Dataset Output:
    Each sample provides a window for autoregressive training as a dict
    compatible with the PyTorch default collate_fn:
        y  : [K+1, S]  Full trajectory window including initial state y0
        dt : [K]       Per-step normalized dt for the window
        g  : [G]       Global parameters

Windowing Convention:
    For a trajectory of length T and window size K:
    - A random start index s is chosen such that s + K + 1 <= T
    - y0 = y[s]                   (initial state)
    - y_seq = y[s+1 : s+1+K]      (K target states)
    - dt_seq = dt[s : s+K]        (K timesteps)

Performance Features:
    - Shard-level LRU caching (configurable size)
    - Optional memory-mapped loading for large datasets
    - Optional full preloading to device memory
    - Proper per-worker random seeding for multi-process loading

Usage:
    dataset = FlowMapRolloutDataset(
        processed_dir="data/processed",
        split="train",
        total_steps=100,
        windows_per_trajectory=1,
    )
    dataloader = create_dataloader(dataset, batch_size=256, shuffle=True, ...)
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
    """
    Metadata for a single data shard.

    Attributes:
        path: Path to the NPZ shard file
        n: Number of trajectories in this shard
    """

    path: Path
    n: int


class FlowMapRolloutDataset(Dataset):
    """
    Dataset that samples random time windows from trajectory shards.

    This dataset loads normalized trajectories from NPZ shards and provides
    random windows suitable for autoregressive flow-map training. Each window
    contains an initial state, a sequence of target states, and corresponding
    timesteps.

    The dataset supports several loading strategies:
        1. On-demand loading: Load shard data when accessed (default)
        2. Cached loading: LRU cache for recently accessed shards
        3. Memory-mapped loading: Memory-efficient for large datasets
        4. Full preloading: Load all data to device memory (fast but memory-intensive)

    Args:
        processed_dir: Path to processed data directory containing shards
        split: Data split name ("train", "validation", or "test")
        total_steps: Number of steps K in each window (determines window size)
        windows_per_trajectory: Number of random windows to sample per trajectory
            per epoch. Higher values provide more data augmentation but may
            increase correlation between samples.
        seed: Random seed for reproducible window selection
        preload_to_device: If True, load all data to device memory at init.
            Fast iteration but high memory usage. Requires num_workers=0.
        device: Target device for preloaded data (only used if preload_to_device=True)
        storage_dtype: Data type for stored tensors (default: float32)
        shard_cache_size: Maximum number of shards to keep in memory cache.
            Larger values reduce disk I/O but increase memory usage.
        use_mmap: If True, use memory-mapped file access for shards. This is
            memory-efficient for large datasets but may be slower for random access.

    Attributes:
        T: Number of timesteps per trajectory
        S: State dimension (number of species)
        G: Global parameter dimension
        norm: NormalizationHelper instance for optional denormalization

    Raises:
        FileNotFoundError: If processed_dir or split directory doesn't exist
        ValueError: If total_steps is incompatible with trajectory length

    Example:
        >>> dataset = FlowMapRolloutDataset(
        ...     "data/processed", "train",
        ...     total_steps=100,
        ...     windows_per_trajectory=2,
        ... )
        >>> y0, dt_seq, y_seq, g = dataset[0]
        >>> print(y0.shape, dt_seq.shape, y_seq.shape, g.shape)
        torch.Size([12]) torch.Size([100]) torch.Size([100, 12]) torch.Size([2])
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
        use_mmap: bool = False,
    ) -> None:
        super().__init__()

        # Validate arguments
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if windows_per_trajectory <= 0:
            raise ValueError(
                f"windows_per_trajectory must be positive, got {windows_per_trajectory}"
            )

        # Store configuration
        self.processed_dir = Path(processed_dir)
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.preload_to_device = bool(preload_to_device)
        self.device = device
        self.storage_dtype = storage_dtype
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.use_mmap = bool(use_mmap) and not preload_to_device

        # Load manifest and index shards
        self._load_manifest()
        self._index_shards()

        # Initialize caching structures
        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._npz_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )

        # Per-worker RNG cache - will be lazily initialized per worker
        # We store the base seed; each worker will create its own RNG
        # Cache keyed by worker identifier. For multi-worker loaders this is (worker_id, torch_seed).
        self._rng_cache: Dict[object, np.random.Generator] = {}

        # Optionally preload all data
        if self.preload_to_device:
            self._preload_all()

    def _load_manifest(self) -> None:
        """Load normalization manifest from processed data directory."""
        manifest_path = self.processed_dir / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run preprocessing first to generate "
                "normalized data and normalization manifest."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict = json.load(f)

        self.norm = NormalizationHelper(self.manifest)

    def _index_shards(self) -> None:
        """
        Index all shards in the split directory.

        Validates shard consistency and builds lookup structures for efficient
        sample indexing.
        """
        split_dir = self.processed_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}. "
                f"Available splits: {[d.name for d in self.processed_dir.iterdir() if d.is_dir()]}"
            )

        shard_paths = sorted(split_dir.glob("shard_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.npz files found in {split_dir}. "
                "Run preprocessing to generate data shards."
            )

        # Use first shard to establish expected dimensions
        with np.load(shard_paths[0]) as first:
            y0 = np.asarray(first["y_mat"], dtype=np.float32)
            g0 = np.asarray(first["globals"], dtype=np.float32)
            dt0 = np.asarray(first["dt_norm_mat"], dtype=np.float32)

        if y0.ndim != 3 or g0.ndim != 2 or dt0.ndim != 2:
            raise ValueError(
                f"Expected y_mat[N,T,S], globals[N,G], dt_norm_mat[N,T-1]. "
                f"Got shapes: y={y0.shape}, g={g0.shape}, dt={dt0.shape}"
            )

        self.T = int(y0.shape[1])  # Trajectory length
        self.S = int(y0.shape[2])  # State dimension
        self.G = int(g0.shape[1])  # Global parameter dimension

        if dt0.shape[1] != self.T - 1:
            raise ValueError(
                f"dt_norm_mat must have T-1 columns. "
                f"Got dt columns={dt0.shape[1]}, T={self.T}"
            )

        # Index all shards and validate consistency
        self.shards: List[ShardIndex] = []
        for p in shard_paths:
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float32)
                g = np.asarray(z["globals"], dtype=np.float32)
                dt = np.asarray(z["dt_norm_mat"], dtype=np.float32)

            # Validate shape consistency
            if y.ndim != 3 or y.shape[1:] != (self.T, self.S):
                raise ValueError(
                    f"Shard {p.name} has inconsistent y_mat shape {y.shape}. "
                    f"Expected [N, {self.T}, {self.S}]"
                )
            if g.ndim != 2 or g.shape[1] != self.G:
                raise ValueError(
                    f"Shard {p.name} has inconsistent globals shape {g.shape}. "
                    f"Expected [N, {self.G}]"
                )
            if dt.ndim != 2 or dt.shape[1] != self.T - 1:
                raise ValueError(
                    f"Shard {p.name} has inconsistent dt_norm_mat shape {dt.shape}. "
                    f"Expected [N, {self.T - 1}]"
                )
            if y.shape[0] != g.shape[0] or y.shape[0] != dt.shape[0]:
                raise ValueError(
                    f"Shard {p.name} has inconsistent N across arrays: "
                    f"y={y.shape[0]}, g={g.shape[0]}, dt={dt.shape[0]}"
                )

            self.shards.append(ShardIndex(path=p, n=int(y.shape[0])))

        # Build prefix sums for O(log N) sample-to-shard lookup
        self._shard_offsets = [0]
        for si in self.shards:
            self._shard_offsets.append(self._shard_offsets[-1] + si.n)
        self._total_traj = int(self._shard_offsets[-1])

        # Validate window can fit in trajectory
        # Need at least total_steps + 1 timesteps (1 for y0, total_steps for y_seq)
        self.max_start = self.T - (self.total_steps + 1)
        if self.max_start < 0:
            raise ValueError(
                f"Trajectory length T={self.T} is too short for total_steps={self.total_steps}. "
                f"Need T >= total_steps + 1 = {self.total_steps + 1}. "
                "Either reduce total_steps in training config or increase n_steps in preprocessing."
            )

    def _preload_all(self) -> None:
        """
        Load all shards into device memory.

        This provides fastest iteration but uses significant memory.
        Should only be used with num_workers=0.
        """
        self._shard_cache.clear()

        for si in self.shards:
            with np.load(si.path) as z:
                y = torch.tensor(
                    z["y_mat"], dtype=self.storage_dtype, device=self.device
                )
                g = torch.tensor(
                    z["globals"], dtype=self.storage_dtype, device=self.device
                )
                dt = torch.tensor(
                    z["dt_norm_mat"], dtype=self.storage_dtype, device=self.device
                )
            self._shard_cache.append((y, g, dt))

        self._preloaded = True

    def _get_shard_arrays(
        self, shard_i: int, local_i: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get shard arrays with LRU caching and optional memory mapping.

        Args:
            shard_i: Index of the shard to load
            local_i: If provided and use_mmap=True, only load this trajectory

        Returns:
            Tuple of (y_mat, globals, dt_norm_mat) numpy arrays
        """
        # Check cache first
        if shard_i in self._npz_cache:
            # Move to end (most recently used)
            item = self._npz_cache.pop(shard_i)
            self._npz_cache[shard_i] = item
            return item

        shard_path = self.shards[shard_i].path

        if self.use_mmap and local_i is not None:
            # Memory-mapped loading: only read the specific trajectory
            # This is more memory-efficient for large shards with random access
            with np.load(shard_path, mmap_mode="r") as z:
                # Copy just the needed trajectory to get a contiguous array
                y_traj = np.array(z["y_mat"][local_i], dtype=np.float32)
                g_traj = np.array(z["globals"][local_i], dtype=np.float32)
                dt_traj = np.array(z["dt_norm_mat"][local_i], dtype=np.float32)

            # For mmap mode, we don't cache the full shard
            # Return expanded arrays that match the expected interface
            return (
                y_traj[np.newaxis, ...],
                g_traj[np.newaxis, ...],
                dt_traj[np.newaxis, ...],
            )

        # Standard loading: read entire shard
        with np.load(shard_path) as z:
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)
            dt_all = np.asarray(z["dt_norm_mat"], dtype=np.float32)

        # Add to cache
        self._npz_cache[shard_i] = (y_all, g_all, dt_all)

        # Evict oldest if cache is full
        while len(self._npz_cache) > self.shard_cache_size:
            self._npz_cache.popitem(last=False)

        return y_all, g_all, dt_all

    def _get_worker_rng(self) -> np.random.Generator:
        """
        Get or create a random number generator for the current worker.

        Uses numpy's SeedSequence for proper entropy spawning, which avoids
        potential seed collisions that can occur with simple arithmetic
        (seed + worker_id) approaches.

        Each worker gets an independent RNG stream derived from the base seed,
        ensuring reproducibility while avoiding correlation between workers.

        The correct pattern is to spawn ALL worker sequences at once and then
        index by worker id:
            spawn(num_workers)[worker_id]

        This ensures each worker gets a unique, independent stream.

        For epoch-to-epoch variation while maintaining reproducibility within
        each epoch, we incorporate torch.initial_seed() into the entropy.

        Returns:
            numpy random Generator for the current worker
        """
        info = get_worker_info()

        if info is None:
            # Single-process loading (num_workers=0)
            #
            # We derive the RNG stream from torch.initial_seed() so that, if the
            # surrounding training loop re-seeds between runs (or across processes),
            # we do not accidentally reuse stale RNG state. The generator state will
            # advance across __getitem__ calls, providing window variation over time.
            torch_seed = int(torch.initial_seed()) % (2**32)
            cache_key = (0, torch_seed)
            if cache_key not in self._rng_cache:
                # Remove prior single-process entries for worker 0 (if any)
                for k in list(self._rng_cache.keys()):
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == 0:
                        del self._rng_cache[k]
                ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(torch_seed,))
                self._rng_cache[cache_key] = np.random.default_rng(ss)
            return self._rng_cache[cache_key]

        # Multi-process loading: need unique RNG per worker per epoch
        # Include torch.initial_seed() for epoch variation
        # torch.initial_seed() changes each epoch in PyTorch DataLoader
        torch_seed = int(torch.initial_seed()) % (2**32)

        # Create a cache key that includes both worker id and epoch seed
        # This ensures we get a new RNG each epoch while keeping workers independent
        cache_key = (info.id, torch_seed)

        if cache_key not in self._rng_cache:
            # Clear old entries for this worker (from previous epochs)
            keys_to_remove = [k for k in self._rng_cache if isinstance(k, tuple) and k[0] == info.id]
            for k in keys_to_remove:
                del self._rng_cache[k]

            # Spawn num_workers child sequences and take the one for this worker
            # This is the correct pattern: spawn(n) returns n independent sequences
            # and we index by worker id to get this worker's unique sequence
            base_ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(torch_seed,))
            child_sequences = base_ss.spawn(info.num_workers)
            worker_ss = child_sequences[info.id]

            self._rng_cache[cache_key] = np.random.default_rng(worker_ss)

        return self._rng_cache[cache_key]

    def __len__(self) -> int:
        """
        Return total number of samples in the dataset.

        Total samples = num_trajectories * windows_per_trajectory
        """
        return self._total_traj * self.windows_per_trajectory

    def __getitem__(
        self, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Samples a random window from the trajectory corresponding to idx.
        The same trajectory may yield different windows due to random
        start position selection.

        Args:
            idx: Sample index in [0, len(dataset))

        Returns:
            Dictionary with:
                y: Full trajectory window [K+1, S] including initial state y0
                dt: Timestep sequence [K]
                g: Global parameters [G]

        Raises:
            IndexError: If idx is out of bounds
        """
        rng = self._get_worker_rng()

        # Map sample index to trajectory index
        traj_idx = idx // self.windows_per_trajectory
        if traj_idx < 0 or traj_idx >= self._total_traj:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {len(self)} samples"
            )

        # Find which shard contains this trajectory
        shard_i = bisect_right(self._shard_offsets, traj_idx) - 1
        local_i = int(traj_idx - self._shard_offsets[shard_i])

        # Sample random window start position
        start = int(rng.integers(0, self.max_start + 1))
        end = start + self.total_steps + 1

        if self._preloaded:
            # Use preloaded device tensors
            y_mat, g_mat, dt_mat = self._shard_cache[shard_i]
            y_traj = y_mat[local_i]
            g = g_mat[local_i]
            dt_traj = dt_mat[local_i]
        else:
            # Load from disk (with caching)
            if self.use_mmap:
                # Memory-mapped: load single trajectory
                y_all, g_all, dt_all = self._get_shard_arrays(shard_i, local_i)
                y_np = y_all[0]  # Already indexed
                g_np = g_all[0]
                dt_np = dt_all[0]
            else:
                # Standard: load full shard (cached)
                y_all, g_all, dt_all = self._get_shard_arrays(shard_i)
                y_np = y_all[local_i]
                g_np = g_all[local_i]
                dt_np = dt_all[local_i]

            # Convert to tensors
            if self.storage_dtype == torch.float32:
                y_traj = torch.from_numpy(y_np)
                g = torch.from_numpy(g_np)
                dt_traj = torch.from_numpy(dt_np)
            else:
                y_traj = torch.as_tensor(y_np, dtype=self.storage_dtype)
                g = torch.as_tensor(g_np, dtype=self.storage_dtype)
                dt_traj = torch.as_tensor(dt_np, dtype=self.storage_dtype)

        # Extract window
        y0 = y_traj[start, :]                         # Initial state [S]
        y_seq = y_traj[start + 1 : end, :]            # Target sequence [K, S]
        dt_seq = dt_traj[start : start + self.total_steps]  # Timesteps [K]

        # Validate shapes (should always pass if shards are consistent)
        if dt_seq.shape[0] != y_seq.shape[0]:
            raise RuntimeError(
                f"Window shape mismatch: dt_seq={dt_seq.shape}, y_seq={y_seq.shape}, "
                f"start={start}, end={end}. This indicates a data inconsistency."
            )

        # Combine y0 and y_seq into a full trajectory window.
        # This produces the shape expected by FlowMapRolloutModule: [K+1, S].
        y_full = torch.cat([y0.unsqueeze(0), y_seq], dim=0)

        return {"y": y_full, "dt": dt_seq, "g": g}


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
    Create a DataLoader with safe defaults for this codebase.

    Handles edge cases and incompatible configurations:
        - Disables pin_memory when data is preloaded to device
        - Disables persistent_workers when num_workers=0
        - Warns if drop_last would result in zero batches

    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples each epoch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        drop_last: Whether to drop incomplete final batch.
            If None, automatically set based on shuffle and dataset size.

    Returns:
        Configured DataLoader

    Raises:
        ValueError: If configuration is invalid (e.g., preload with workers)
    """
    # Check for incompatible configurations
    preload = bool(getattr(dataset, "preload_to_device", False))
    if preload and num_workers != 0:
        raise ValueError(
            "preload_to_device=True requires num_workers=0. "
            "Preloaded data cannot be shared across worker processes."
        )

    # Disable pin_memory for preloaded data (already on device)
    if preload and pin_memory:
        warnings.warn(
            "pin_memory=True is incompatible with preload_to_device=True. "
            "Data is already on device. Disabling pin_memory.",
            UserWarning,
            stacklevel=2,
        )
        pin_memory = False

    # persistent_workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False

    # Get dataset size for drop_last logic
    try:
        ds_len = len(dataset)
    except TypeError:
        ds_len = None

    if ds_len is not None and ds_len == 0:
        raise ValueError(
            "Dataset is empty - cannot create DataLoader. "
            "Check that preprocessing generated data for this split."
        )

    # Auto-configure drop_last
    if drop_last is None:
        # Drop last only when shuffling and dataset is large enough
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

    # prefetch_factor is only valid with num_workers > 0
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)
