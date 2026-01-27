#!/usr/bin/env python3
"""
dataset.py - Rollout dataset for autoregressive training.

This module provides datasets and dataloader helpers for autoregressive
flow-map training on pre-normalized trajectory shards.

Key design goals:
- Correctness and reproducibility (worker-safe RNG handling).
- Practical performance at scale (shard caching, mmap, optional GPU preload).
- Efficient GPU-resident training (vectorized batch assembly when preloaded).

Data format (per shard NPZ):
    y_mat       : [N, T, S] float32  (z-space trajectories)
    globals     : [N, G]    float32  (z-space globals)
    dt_norm_mat : [N, T-1]  float32  (normalized dt per step)

Sample format (map-style dataset):
    {
        "y":  [K+1, S],   # includes the initial state at index 0
        "dt": [K],
        "g":  [G],
    }

Batch format (default collate on map-style dataset):
    {
        "y":  [B, K+1, S],
        "dt": [B, K],
        "g":  [B, G],
    }

Performance note (important for GH200-class GPUs):
If `preload_to_device=True`, using a map-style dataset with large `batch_size`
forces Python to call `__getitem__` B times per step, producing thousands of
tiny GPU indexing ops. This typically results in very low utilization.
To avoid that, this module automatically switches to a vectorized, GPU-native
IterableDataset that yields fully assembled batches with only a few large GPU
kernels per step.
"""

from __future__ import annotations

import json
import warnings
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from normalizer import NormalizationHelper


@dataclass(frozen=True)
class ShardIndex:
    """Metadata for a single data shard."""
    path: Path
    n: int


class FlowMapRolloutDataset(Dataset):
    """
    Map-style dataset that samples random windows from trajectory shards.

    This dataset is convenient and works well with multi-worker loading when
    data is not preloaded to GPU. If `preload_to_device=True`, prefer the
    vectorized batch stream created by `create_dataloader()` (automatic).
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        split: str,
        *,
        total_steps: int,
        windows_per_trajectory: int = 1,
        seed: int = 1234,
        random_windows: Optional[bool] = None,
        preload_to_device: bool = False,
        device: torch.device = torch.device("cpu"),
        storage_dtype: torch.dtype = torch.float32,
        shard_cache_size: int = 2,
        use_mmap: bool = False,
    ) -> None:
        super().__init__()

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
        self.use_mmap = bool(use_mmap) and not preload_to_device

        # Window sampling policy:
        # - train: random windows by default
        # - val/test: deterministic windows by default (stable metrics)
        if random_windows is None:
            random_windows = str(split).lower().strip() == "train"
        self.random_windows = bool(random_windows)

        self._load_manifest()
        self._index_shards()

        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._npz_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )

        # Per-worker RNG cache (numpy Generator). Keyed by (worker_id, torch_seed).
        self._rng_cache: Dict[object, np.random.Generator] = {}

        if self.preload_to_device:
            self._preload_all()

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _load_manifest(self) -> None:
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

        with np.load(shard_paths[0]) as first:
            y0 = np.asarray(first["y_mat"], dtype=np.float32)
            g0 = np.asarray(first["globals"], dtype=np.float32)
            dt0 = np.asarray(first["dt_norm_mat"], dtype=np.float32)

        if y0.ndim != 3:
            raise ValueError(f"y_mat must be 3D [N,T,S], got shape {y0.shape}")
        if g0.ndim != 2:
            raise ValueError(f"globals must be 2D [N,G], got shape {g0.shape}")
        if dt0.ndim != 2:
            raise ValueError(f"dt_norm_mat must be 2D [N,T-1], got shape {dt0.shape}")

        self.T = int(y0.shape[1])
        self.S = int(y0.shape[2])
        self.G = int(g0.shape[1])

        if dt0.shape[1] != self.T - 1:
            raise ValueError(
                f"dt_norm_mat second dim must be T-1={self.T-1}, got {dt0.shape[1]}"
            )

        if self.total_steps + 1 > self.T:
            raise ValueError(
                f"total_steps={self.total_steps} requires window length K+1={self.total_steps+1} "
                f"but trajectories have length T={self.T}."
            )

        self.max_start = int(self.T - (self.total_steps + 1))
        if self.max_start < 0:
            raise ValueError(
                f"Invalid max_start {self.max_start}. Check total_steps and trajectory length."
            )

        self.shards: List[ShardIndex] = []
        for p in shard_paths:
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float32)
            n = int(y.shape[0])
            if n <= 0:
                raise ValueError(f"Shard {p} has no trajectories.")
            self.shards.append(ShardIndex(path=p, n=n))

        self._total_traj = int(sum(si.n for si in self.shards))

        # Prefix offsets in trajectory space (global trajectory id -> shard).
        self._shard_offsets: List[int] = [0]
        running = 0
        for si in self.shards:
            running += int(si.n)
            self._shard_offsets.append(running)

    def _preload_all(self) -> None:
        """Load all shards into device memory (requires num_workers=0)."""
        self._shard_cache.clear()

        for si in self.shards:
            with np.load(si.path) as z:
                y = torch.as_tensor(z["y_mat"], dtype=self.storage_dtype, device=self.device)
                g = torch.as_tensor(z["globals"], dtype=self.storage_dtype, device=self.device)
                dt = torch.as_tensor(z["dt_norm_mat"], dtype=self.storage_dtype, device=self.device)
            self._shard_cache.append((y, g, dt))

        self._preloaded = True

    # -------------------------------------------------------------------------
    # On-demand shard loading (CPU)
    # -------------------------------------------------------------------------

    def _get_shard_arrays(
        self, shard_i: int, local_i: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get shard arrays with LRU caching and optional memory mapping."""
        if shard_i in self._npz_cache:
            item = self._npz_cache.pop(shard_i)
            self._npz_cache[shard_i] = item
            return item

        shard_path = self.shards[shard_i].path

        if self.use_mmap and local_i is not None:
            with np.load(shard_path, mmap_mode="r") as z:
                y_traj = np.array(z["y_mat"][local_i], dtype=np.float32)
                g_traj = np.array(z["globals"][local_i], dtype=np.float32)
                dt_traj = np.array(z["dt_norm_mat"][local_i], dtype=np.float32)
            return (
                y_traj[np.newaxis, ...],
                g_traj[np.newaxis, ...],
                dt_traj[np.newaxis, ...],
            )

        with np.load(shard_path) as z:
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)
            dt_all = np.asarray(z["dt_norm_mat"], dtype=np.float32)

        self._npz_cache[shard_i] = (y_all, g_all, dt_all)
        while len(self._npz_cache) > self.shard_cache_size:
            self._npz_cache.popitem(last=False)

        return y_all, g_all, dt_all

    # -------------------------------------------------------------------------
    # RNG utilities
    # -------------------------------------------------------------------------

    def _get_worker_rng(self) -> np.random.Generator:
        """
        Return a numpy RNG suitable for __getitem__.

        For multi-worker loaders, each worker gets its own independent stream.
        We additionally incorporate torch.initial_seed() to avoid accidental
        reuse when the outer training loop changes seeds.
        """
        info = get_worker_info()

        if info is None:
            torch_seed = int(torch.initial_seed()) % (2**32)
            cache_key = (0, torch_seed)
            if cache_key not in self._rng_cache:
                for k in list(self._rng_cache.keys()):
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == 0:
                        del self._rng_cache[k]
                ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(torch_seed,))
                self._rng_cache[cache_key] = np.random.default_rng(ss)
            return self._rng_cache[cache_key]

        torch_seed = int(torch.initial_seed()) % (2**32)
        cache_key = (info.id, torch_seed)

        if cache_key not in self._rng_cache:
            keys_to_remove = [
                k for k in self._rng_cache.keys() if isinstance(k, tuple) and len(k) == 2 and k[0] == info.id
            ]
            for k in keys_to_remove:
                del self._rng_cache[k]
            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(info.id, torch_seed))
            self._rng_cache[cache_key] = np.random.default_rng(ss)

        return self._rng_cache[cache_key]

    # -------------------------------------------------------------------------
    # Window selection
    # -------------------------------------------------------------------------

    def _deterministic_start(self, window_id: int) -> int:
        """Deterministically choose a window start index for a given window id."""
        if self.max_start <= 0 or self.windows_per_trajectory <= 1:
            return 0
        w = int(window_id) % int(self.windows_per_trajectory)
        frac = w / float(self.windows_per_trajectory - 1)
        start = int(round(frac * self.max_start))
        # Safety clamp against rounding edge-cases
        if start < 0:
            return 0
        if start > self.max_start:
            return int(self.max_start)
        return start

    def __len__(self) -> int:
        return self._total_traj * self.windows_per_trajectory

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._get_worker_rng()

        traj_idx = idx // self.windows_per_trajectory
        if traj_idx < 0 or traj_idx >= self._total_traj:
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self)} samples")

        shard_i = bisect_right(self._shard_offsets, traj_idx) - 1
        local_i = int(traj_idx - self._shard_offsets[shard_i])

        if self.random_windows:
            start = int(rng.integers(0, self.max_start + 1))
        else:
            # Use a deterministic set of window start positions per trajectory.
            w = int(idx % self.windows_per_trajectory)
            start = self._deterministic_start(w)
        end = start + self.total_steps + 1  # slice end for y window

        if self._preloaded:
            y_mat, g_mat, dt_mat = self._shard_cache[shard_i]
            y_traj = y_mat[local_i]
            g = g_mat[local_i]
            dt_traj = dt_mat[local_i]
        else:
            if self.use_mmap:
                y_all, g_all, dt_all = self._get_shard_arrays(shard_i, local_i)
                y_np, g_np, dt_np = y_all[0], g_all[0], dt_all[0]
            else:
                y_all, g_all, dt_all = self._get_shard_arrays(shard_i)
                y_np, g_np, dt_np = y_all[local_i], g_all[local_i], dt_all[local_i]

            y_traj = torch.as_tensor(y_np, dtype=self.storage_dtype)
            g = torch.as_tensor(g_np, dtype=self.storage_dtype)
            dt_traj = torch.as_tensor(dt_np, dtype=self.storage_dtype)

        # Efficient window extraction: a single slice gives [K+1, S].
        y_full = y_traj[start:end, :]
        dt_seq = dt_traj[start : start + self.total_steps]

        if y_full.shape[0] != self.total_steps + 1:
            raise RuntimeError(
                f"Bad y window shape {y_full.shape}; expected [{self.total_steps + 1}, {self.S}]."
            )
        if dt_seq.shape[0] != self.total_steps:
            raise RuntimeError(
                f"Bad dt window shape {dt_seq.shape}; expected [{self.total_steps}]."
            )

        return {"y": y_full, "dt": dt_seq, "g": g}


class _GPUPreloadedBatchStream(IterableDataset):
    """
    Vectorized batch stream for GPU-preloaded FlowMapRolloutDataset.

    This is specifically designed to address the main utilization bottleneck:
    calling __getitem__ B times per step and doing thousands of tiny GPU slices.

    The stream yields already-batched tensors directly on the target device.
    """

    def __init__(
        self,
        base: FlowMapRolloutDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not getattr(base, "_preloaded", False):
            raise ValueError("_GPUPreloadedBatchStream requires a preloaded base dataset")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.base = base
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.device = base.device
        self.dtype = base.storage_dtype
        self.K = int(base.total_steps)
        self.T = int(base.T)
        self.S = int(base.S)
        self.G = int(base.G)

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

        counts = torch.tensor([si.n for si in base.shards], device=self.device, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError("Encountered a shard with non-positive trajectory count.")
        self._shard_weights = counts / counts.sum()

        self._seed = int(seed if seed is not None else base.seed)

        # RNG handling:
        # - Prefer a device generator so random index generation stays on-GPU.
        # - If unavailable (older torch), fall back to a CPU generator and generate indices on CPU,
        #   then transfer only the small index tensors to GPU. This avoids device-mismatch errors.
        self._use_cpu_rng = False
        self._shard_weights_cpu: Optional[torch.Tensor] = None

        try:
            self._gen = torch.Generator(device=self.device)
        except TypeError:
            self._use_cpu_rng = True
            self._gen = torch.Generator()
            warnings.warn(
                "torch.Generator(device=...) is not available; using a CPU generator and CPU-sampled indices. "
                "Throughput may be slightly reduced.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._gen.manual_seed(self._seed)
        # Keep a CPU copy for multinomial in CPU-RNG fallback.
        if self._use_cpu_rng:
            self._shard_weights_cpu = self._shard_weights.detach().to("cpu")

    def __len__(self) -> int:
        total = len(self.base)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        total_samples = len(self.base)
        num_batches = len(self)

        for b in range(num_batches):
            remaining = total_samples - b * self.batch_size
            B = self.batch_size if (self.drop_last or remaining >= self.batch_size) else remaining
            if B <= 0:
                break

            # Select a shard and sample (trajectory, start) indices.
            if self._use_cpu_rng:
                assert self._shard_weights_cpu is not None
                shard_i = int(torch.multinomial(self._shard_weights_cpu, 1, generator=self._gen).item())
            else:
                shard_i = int(torch.multinomial(self._shard_weights, 1, generator=self._gen).item())

            y_mat, g_mat, dt_mat = self.base._shard_cache[shard_i]
            N = int(y_mat.shape[0])

            if self._use_cpu_rng:
                traj = torch.randint(0, N, (B,), device="cpu", generator=self._gen, dtype=torch.long).to(self.device)
                start = torch.randint(
                    0,
                    self.base.max_start + 1,
                    (B,),
                    device="cpu",
                    generator=self._gen,
                    dtype=torch.long,
                ).to(self.device)
            else:
                traj = torch.randint(0, N, (B,), device=self.device, generator=self._gen, dtype=torch.long)
                start = torch.randint(
                    0,
                    self.base.max_start + 1,
                    (B,),
                    device=self.device,
                    generator=self._gen,
                    dtype=torch.long,
                )

            y_btS = y_mat.index_select(0, traj)
            idx_y = (start[:, None] + self._t_y[None, :]).unsqueeze(-1).expand(-1, -1, self.S)
            y_win = torch.gather(y_btS, dim=1, index=idx_y)

            dt_bT = dt_mat.index_select(0, traj)
            idx_dt = (start[:, None] + self._t_dt[None, :])
            dt_win = torch.gather(dt_bT, dim=1, index=idx_dt)

            g = g_mat.index_select(0, traj)

            yield {"y": y_win, "dt": dt_win, "g": g}


class _GPUPreloadedDeterministicBatchStream(IterableDataset):
    """
    Deterministic, exhaustive batch stream for GPU-preloaded FlowMapRolloutDataset.

    This yields the same samples as the map-style dataset (no replacement),
    but avoids calling __getitem__ B times per step. It is intended for
    validation/test when preload_to_device=True and shuffle=False.
    """

    def __init__(
        self,
        base: FlowMapRolloutDataset,
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        super().__init__()

        if not getattr(base, "_preloaded", False):
            raise ValueError(
                "_GPUPreloadedDeterministicBatchStream requires a preloaded base dataset"
            )

        if getattr(base, "random_windows", False):
            raise ValueError(
                "_GPUPreloadedDeterministicBatchStream requires random_windows=False "
                "(validation/test deterministic windows)."
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.base = base
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        self.device = base.device
        self.dtype = base.storage_dtype
        self.K = int(base.total_steps)
        self.S = int(base.S)
        self.G = int(base.G)
        self.windows_per_trajectory = int(base.windows_per_trajectory)

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

        starts = [int(base._deterministic_start(w)) for w in range(self.windows_per_trajectory)]
        self._starts_w = torch.tensor(starts, device=self.device, dtype=torch.long)

        # Shard sample offsets in "sample space" (trajectory-major, then window_id).
        wpt = self.windows_per_trajectory
        shard_ns = [int(y.shape[0]) for (y, _, _) in base._shard_cache]
        shard_samples = [n * wpt for n in shard_ns]

        offsets = [0]
        total = 0
        for ss in shard_samples:
            total += int(ss)
            offsets.append(total)

        self._shard_sample_offsets = offsets  # length = num_shards + 1
        self._total_samples = int(total)

    def __len__(self) -> int:
        if self.drop_last:
            return self._total_samples // self.batch_size
        return (self._total_samples + self.batch_size - 1) // self.batch_size

    def _gather_from_shard(
        self, shard_i: int, shard_sid_start: int, B: int
    ) -> Dict[str, torch.Tensor]:
        wpt = self.windows_per_trajectory
        y_mat, g_mat, dt_mat = self.base._shard_cache[shard_i]

        sid = torch.arange(
            shard_sid_start,
            shard_sid_start + B,
            device=self.device,
            dtype=torch.long,
        )
        local_traj = sid // wpt
        w = sid - local_traj * wpt
        start = self._starts_w.index_select(0, w)

        y_btS = y_mat.index_select(0, local_traj)
        idx_y = (start[:, None] + self._t_y[None, :]).unsqueeze(-1).expand(-1, -1, self.S)
        y_win = torch.gather(y_btS, dim=1, index=idx_y)

        dt_bT = dt_mat.index_select(0, local_traj)
        idx_dt = (start[:, None] + self._t_dt[None, :])
        dt_win = torch.gather(dt_bT, dim=1, index=idx_dt)

        g = g_mat.index_select(0, local_traj)

        return {"y": y_win, "dt": dt_win, "g": g}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.drop_last and self._total_samples < self.batch_size:
            return iter(())

        total = self._total_samples
        bs = self.batch_size
        offsets = self._shard_sample_offsets

        # Global sample cursor over the entire split.
        sid_global = 0
        while sid_global < total:
            remaining = total - sid_global
            B = bs if (self.drop_last or remaining >= bs) else remaining
            if B <= 0:
                break

            # Find shard for the first sample in this batch (sid_global).
            shard_i = bisect_right(offsets, sid_global) - 1
            shard_start = offsets[shard_i]
            shard_end = offsets[shard_i + 1]

            # Number of samples we can take from this shard without crossing boundary.
            take0 = min(B, shard_end - sid_global)
            batch0 = self._gather_from_shard(shard_i, sid_global - shard_start, int(take0))

            if take0 == B:
                yield batch0
            else:
                # Batch crosses into next shard (at most one boundary, since ordering is sequential).
                shard_i2 = shard_i + 1
                if shard_i2 >= len(self.base._shard_cache):
                    raise RuntimeError("Shard boundary logic failed: ran past last shard.")

                take1 = int(B - take0)
                batch1 = self._gather_from_shard(shard_i2, 0, take1)

                yield {
                    "y": torch.cat([batch0["y"], batch1["y"]], dim=0),
                    "dt": torch.cat([batch0["dt"], batch1["dt"]], dim=0),
                    "g": torch.cat([batch0["g"], batch1["g"]], dim=0),
                }

            sid_global += B


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

    Performance/correctness notes for preload_to_device=True:
    - Training (shuffle=True): use a stochastic GPU batch stream that samples windows with replacement.
      This maximizes throughput and is appropriate for SGD.
    - Validation/test (shuffle=False): use a deterministic GPU batch stream that iterates the dataset
      exhaustively (no replacement), matching map-style semantics while avoiding per-sample __getitem__.
    """
    preload = bool(getattr(dataset, "preload_to_device", False))
    if preload and num_workers != 0:
        raise ValueError(
            "preload_to_device=True requires num_workers=0. "
            "Preloaded data cannot be shared across worker processes."
        )

    # Dataset size for drop_last logic
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
        drop_last = bool(shuffle and ds_len is not None and ds_len >= batch_size)
    elif drop_last and ds_len is not None and ds_len < batch_size:
        warnings.warn(
            f"drop_last=True with dataset size {ds_len} < batch_size {batch_size} "
            "would yield 0 batches; forcing drop_last=False",
            UserWarning,
            stacklevel=2,
        )
        drop_last = False

    # Preloaded data is already on device; pinning host memory is meaningless.
    if preload and pin_memory:
        warnings.warn(
            "pin_memory=True is incompatible with preload_to_device=True. "
            "Data is already on device. Disabling pin_memory.",
            UserWarning,
            stacklevel=2,
        )
        pin_memory = False

    if num_workers == 0:
        persistent_workers = False

    # ---- Optimized GPU streams for preloaded FlowMapRolloutDataset ----
    if preload and isinstance(dataset, FlowMapRolloutDataset) and getattr(dataset, "_preloaded", False):
        if shuffle:
            stream: IterableDataset = _GPUPreloadedBatchStream(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=bool(drop_last),
                seed=getattr(dataset, "seed", 1234),
            )
        else:
            # For validation/test we must not drop samples; enforce drop_last=False.
            if drop_last:
                warnings.warn(
                    "drop_last=True for validation/test would discard samples; forcing drop_last=False.",
                    UserWarning,
                    stacklevel=2,
                )
                drop_last = False
            stream = _GPUPreloadedDeterministicBatchStream(
                dataset,
                batch_size=batch_size,
                drop_last=False,
            )

        return DataLoader(
            dataset=stream,
            batch_size=None,  # stream yields already-batched tensors
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    # Standard map-style loader (used for non-preloaded cases)
    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=bool(drop_last),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)
