#!/usr/bin/env python3
"""
dataset.py - Rollout dataset for autoregressive training.

This module provides datasets and dataloader helpers for autoregressive
flow-map training on *pre-normalized* trajectory shards.

What the model expects (per model.py and trainer.py):
- y is already in z-space (normalized). For log-standardized species this means:
      z = (log10(y_phys) - log_mean) / log_std
  and log10(y_phys) can be recovered via:
      log10(y_phys) = z * log_std + log_mean

- dt is already normalized to [0, 1] using log10 + min-max (see normalizer.py):
      dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

- g (globals) is already normalized (method depends on your preprocessing/manifest).

Accordingly, this dataset does NOT apply NormalizationHelper during sampling.
The normalization manifest is still required for training/inference elsewhere
(e.g., loss conversion to log10-space, and post-hoc denormalization).

Data format (per shard NPZ):
    y_mat       : [N, T, S] float32  (z-space trajectories)
    globals     : [N, G]    float32  (normalized globals)
    dt_norm_mat : [N, T-1]  float32  (normalized dt per step in [0,1])

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

Performance note:
If `preload_to_device=True`, using a map-style dataset with large `batch_size`
forces Python to call `__getitem__` B times per step, producing thousands of
tiny device indexing ops. To avoid that, `create_dataloader()` automatically
switches to a vectorized, GPU-native IterableDataset that yields pre-batched
tensors with a few large kernels per step.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


_DT_PROBE_MAX = 1024
_DT_RANGE_EPS = 1e-6


@dataclass(frozen=True)
class ShardIndex:
    """Metadata for a single data shard."""
    path: Path
    n: int


class FlowMapRolloutDataset(Dataset):
    """
    Map-style dataset that samples random windows from trajectory shards.

    This dataset works well with multi-worker CPU loading. If `preload_to_device=True`,
    prefer the vectorized batch stream created by `create_dataloader()` (automatic).
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
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
        validate_manifest: bool = True,
        validate_dt_range: bool = True,
    ) -> None:
        """
        Args:
            processed_dir: Root processed data directory containing normalization.json and split subdirs.
            split: "train" | "validation" | "test" (or any subdir name under processed_dir).
            total_steps: K. The dataset returns y windows of length K+1 and dt windows of length K.
            windows_per_trajectory: Controls *epoch size* by repeating each trajectory this many times.
                                  Sampling is still random each time.
            seed: Base seed for per-worker RNG.
            preload_to_device: If True, all shards are loaded to `device` and a GPU-native batch stream
                              can be used by create_dataloader().
            device: Target device for preload_to_device.
            storage_dtype: dtype for tensors returned by the dataset (float32 recommended).
            shard_cache_size: LRU size for cached shards (CPU full-load cache) or open NPZ handles (mmap).
            use_mmap: If True, use mmap_mode="r" NPZ handles and avoid loading full shards into RAM.
            validate_manifest: If True, load and store normalization.json (for sanity checks / downstream access).
            validate_dt_range: If True, verify dt_norm_mat is within [0,1] on the first shard (fast check).
        """
        super().__init__()

        processed_dir = Path(processed_dir).expanduser().resolve()
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if windows_per_trajectory <= 0:
            raise ValueError(f"windows_per_trajectory must be positive, got {windows_per_trajectory}")
        if shard_cache_size <= 0:
            raise ValueError(f"shard_cache_size must be positive, got {shard_cache_size}")

        self.processed_dir = processed_dir
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.preload_to_device = bool(preload_to_device)
        self.device = device
        self.storage_dtype = storage_dtype
        self.shard_cache_size = int(shard_cache_size)
        self.use_mmap = bool(use_mmap)

        self.manifest: Optional[Dict[str, object]] = None
        if validate_manifest:
            mpath = processed_dir / "normalization.json"
            if not mpath.exists():
                raise FileNotFoundError(f"Missing normalization.json at {mpath}")
            with open(mpath, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

        shard_paths = sorted(split_dir.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard NPZ files found in {split_dir}")

        # Peek first shard for basic schema + dimensions.
        with np.load(shard_paths[0]) as z:
            y0 = z["y_mat"]
            g0 = z["globals"]
            dt0 = z["dt_norm_mat"]

        if y0.ndim != 3:
            raise ValueError(f"Expected y_mat to be 3D [N,T,S], got {y0.shape}")
        if g0.ndim != 2:
            raise ValueError(f"Expected globals to be 2D [N,G], got {g0.shape}")
        if dt0.ndim != 2:
            raise ValueError(f"Expected dt_norm_mat to be 2D [N,T-1], got {dt0.shape}")

        N0, T0, S0 = y0.shape
        if dt0.shape[1] != T0 - 1:
            raise ValueError(f"dt_norm_mat second dim must be T-1={T0-1}, got {dt0.shape[1]}")
        if g0.shape[0] != N0:
            raise ValueError(f"globals first dim must match y_mat N={N0}, got {g0.shape[0]}")

        if T0 < self.total_steps + 1:
            raise ValueError(f"Trajectories too short for total_steps={self.total_steps}.")

        self.T = int(T0)
        self.S = int(S0)
        self.G = int(g0.shape[1])

        self.max_start = int(self.T - (self.total_steps + 1))
        if self.max_start < 0:
            raise ValueError(f"max_start < 0; T={self.T} total_steps={self.total_steps}")
        # Strict dtypes: avoid silent CPU copies / perf cliffs.
        if y0.dtype != np.float32:
            raise ValueError(f"y_mat dtype must be float32, got {y0.dtype}. Regenerate preprocessing outputs.")
        if g0.dtype != np.float32:
            raise ValueError(f"globals dtype must be float32, got {g0.dtype}. Regenerate preprocessing outputs.")
        if dt0.dtype != np.float32:
            raise ValueError(f"dt_norm_mat dtype must be float32, got {dt0.dtype}. Regenerate preprocessing outputs.")

        # Fast dt range sanity check on a small slice (avoids reading full array).
        if validate_dt_range:
            dt_probe = np.asarray(dt0[0, : min(_DT_PROBE_MAX, dt0.shape[1])])
            if np.any(dt_probe < -_DT_RANGE_EPS) or np.any(dt_probe > 1.0 + _DT_RANGE_EPS):
                raise ValueError(
                    "dt_norm_mat contains values outside [0,1]. The training code expects dt already normalized. "
                    "Regenerate preprocessing outputs."
                )

        # Build shard index (trajectory counts per shard)
        self.shards: List[ShardIndex] = []
        for p in shard_paths:
            with np.load(p) as z:
                n = int(z["y_mat"].shape[0])
            if n <= 0:
                raise ValueError(f"Shard {p} has no trajectories.")
            self.shards.append(ShardIndex(path=p, n=n))

        self._total_traj = int(sum(si.n for si in self.shards))

        # Precompute O(1) global-trajectory → (shard, local) lookup tables.
        # This avoids per-sample Python binary search in __getitem__.
        self._traj_to_shard = np.empty(self._total_traj, dtype=np.int32)
        self._traj_to_local = np.empty(self._total_traj, dtype=np.int32)
        cursor = 0
        for shard_i, si in enumerate(self.shards):
            n = int(si.n)
            self._traj_to_shard[cursor : cursor + n] = shard_i
            self._traj_to_local[cursor : cursor + n] = np.arange(n, dtype=np.int32)
            cursor += n

        # GPU preload cache
        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # CPU caches
        self._npz_cache_full: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        self._npz_handle_cache: "OrderedDict[int, np.lib.npyio.NpzFile]" = OrderedDict()

        # RNG cache per worker/seed
        self._rng_cache: Dict[Tuple[int, int], np.random.Generator] = {}

        if self.preload_to_device:
            if self.device.type == "cpu":
                raise ValueError("preload_to_device=True requires a CUDA/MPS device (data should live on accelerator).")
            self._preload_all()

    # -------------------------------------------------------------------------
    # Preload
    # -------------------------------------------------------------------------

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
    # On-demand loading (CPU)
    # -------------------------------------------------------------------------

    def _get_full_shard_arrays(self, shard_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load an entire shard into RAM with LRU caching."""
        if shard_i in self._npz_cache_full:
            item = self._npz_cache_full.pop(shard_i)
            self._npz_cache_full[shard_i] = item
            return item

        shard_path = self.shards[shard_i].path
        with np.load(shard_path) as z:
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)
            dt_all = np.asarray(z["dt_norm_mat"], dtype=np.float32)

        self._npz_cache_full[shard_i] = (y_all, g_all, dt_all)
        while len(self._npz_cache_full) > self.shard_cache_size:
            self._npz_cache_full.popitem(last=False)
        return y_all, g_all, dt_all

    def _get_mmap_handle(self, shard_i: int) -> "np.lib.npyio.NpzFile":
        """Get an mmap-mode handle for a shard, with LRU caching."""
        if shard_i in self._npz_handle_cache:
            h = self._npz_handle_cache.pop(shard_i)
            self._npz_handle_cache[shard_i] = h
            return h

        shard_path = self.shards[shard_i].path
        h = np.load(shard_path, mmap_mode="r")
        self._npz_handle_cache[shard_i] = h
        while len(self._npz_handle_cache) > self.shard_cache_size:
            _, old = self._npz_handle_cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return h

    def _get_traj_arrays(self, shard_i: int, local_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single trajectory, using mmap handles or full-shard caching."""
        if self.use_mmap:
            h = self._get_mmap_handle(shard_i)
            y_traj = np.asarray(h["y_mat"][local_i], dtype=np.float32)
            g = np.asarray(h["globals"][local_i], dtype=np.float32)
            dt_traj = np.asarray(h["dt_norm_mat"][local_i], dtype=np.float32)

            if y_traj.dtype != np.float32:
                y_traj = y_traj.astype(np.float32, copy=False)
            if g.dtype != np.float32:
                g = g.astype(np.float32, copy=False)
            if dt_traj.dtype != np.float32:
                dt_traj = dt_traj.astype(np.float32, copy=False)

            return y_traj, g, dt_traj

        y_all, g_all, dt_all = self._get_full_shard_arrays(shard_i)
        return y_all[local_i], g_all[local_i], dt_all[local_i]

    def close(self) -> None:
        """Close any cached open NPZ handles (primarily useful in long-running processes)."""
        for _, h in list(self._npz_handle_cache.items()):
            try:
                h.close()
            except Exception:
                pass
        self._npz_handle_cache.clear()

    def __enter__(self) -> "FlowMapRolloutDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # RNG utilities
    # -------------------------------------------------------------------------

    def _get_worker_rng(self) -> np.random.Generator:
        """
        Return a NumPy RNG suitable for __getitem__.

        - In the main process (num_workers=0), we key the RNG off torch.initial_seed().
        - In DataLoader workers (num_workers>0), we prefer the per-worker seed provided
          by PyTorch (get_worker_info().seed) so behavior is deterministic under
          Lightning / seed_everything(workers=True).
        """
        info = get_worker_info()

        if info is None:
            worker_id = 0
            worker_seed = int(torch.initial_seed()) % (2**32)
        else:
            worker_id = int(info.id)
            # PyTorch sets a unique seed per worker process; prefer that when available.
            worker_seed = int(getattr(info, "seed", torch.initial_seed())) % (2**32)

        cache_key = (worker_id, worker_seed)
        if cache_key not in self._rng_cache:
            # Drop any stale entries for this worker id.
            for k in list(self._rng_cache.keys()):
                if isinstance(k, tuple) and len(k) == 2 and k[0] == worker_id:
                    del self._rng_cache[k]

            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker_id, worker_seed))
            self._rng_cache[cache_key] = np.random.default_rng(ss)

        return self._rng_cache[cache_key]

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        # windows_per_trajectory controls how many samples you get per trajectory per epoch.
        return self._total_traj * self.windows_per_trajectory

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._get_worker_rng()

        traj_idx = idx // self.windows_per_trajectory
        if traj_idx < 0 or traj_idx >= self._total_traj:
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self)} samples")

        shard_i = int(self._traj_to_shard[traj_idx])
        local_i = int(self._traj_to_local[traj_idx])

        # Stochastic window start for every sample.
        start = int(rng.integers(0, self.max_start + 1))
        end = start + self.total_steps + 1  # slice end for y window

        if self._preloaded:
            y_mat, g_mat, dt_mat = self._shard_cache[shard_i]
            y_traj = y_mat[local_i]
            g = g_mat[local_i]
            dt_traj = dt_mat[local_i]
        else:
            y_np, g_np, dt_np = self._get_traj_arrays(shard_i, local_i)
            y_traj = torch.as_tensor(y_np, dtype=self.storage_dtype)
            g = torch.as_tensor(g_np, dtype=self.storage_dtype)
            dt_traj = torch.as_tensor(dt_np, dtype=self.storage_dtype)

        # Single slice yields [K+1, S].
        y_full = y_traj[start:end, :]
        dt_seq = dt_traj[start : start + self.total_steps]

        if int(y_full.shape[0]) != self.total_steps + 1:
            raise RuntimeError(
                f"Bad y window shape {tuple(y_full.shape)}; expected [{self.total_steps + 1}, {self.S}]."
            )
        if int(dt_seq.shape[0]) != self.total_steps:
            raise RuntimeError(
                f"Bad dt window shape {tuple(dt_seq.shape)}; expected [{self.total_steps}]."
            )

        return {"y": y_full, "dt": dt_seq, "g": g}


class _GPUPreloadedStochasticBatchStream(IterableDataset):
    """
    Vectorized stochastic batch stream for GPU-preloaded FlowMapRolloutDataset.

    This addresses the utilization bottleneck of calling __getitem__ B times per step.
    The stream yields already-batched tensors directly on the target device.

    Sampling is with replacement (SGD-style).
    """

    def __init__(
        self,
        base: FlowMapRolloutDataset,
        *,
        batch_size: int,
        drop_last: bool,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not getattr(base, "_preloaded", False):
            raise ValueError("_GPUPreloadedStochasticBatchStream requires a preloaded base dataset")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.base = base
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        self.device = base.device
        self.dtype = base.storage_dtype
        
        self.K = int(base.total_steps)
        self.S = int(base.S)
        self.T = int(base.T)
        self.T_dt = int(base.T - 1)

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)


        counts = torch.tensor([si.n for si in base.shards], device="cpu", dtype=torch.float32)
        if int((counts <= 0).sum().item()) != 0:
            raise ValueError("Encountered a shard with non-positive trajectory count.")
        self._shard_weights_cpu = counts / counts.sum()

        self._seed = int(seed if seed is not None else base.seed)
        # RNG: shard selection is done on CPU to avoid device→host sync from .item().
        self._gen_cpu = torch.Generator()
        self._gen_cpu.manual_seed(self._seed)

        # Trajectory/start indices are sampled on-device.
        try:
            self._gen_dev = torch.Generator(device=self.device)
        except TypeError as e:
            raise RuntimeError(
                "preload_to_device=True requires torch.Generator(device=...) support. "
                "Upgrade PyTorch or set dataset.preload_to_device=false."
            ) from e
        self._gen_dev.manual_seed(self._seed + 1)


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
            # Select a shard (CPU) and sample (trajectory, start) indices.
            shard_i = int(torch.multinomial(self._shard_weights_cpu, 1, generator=self._gen_cpu).item())

            y_mat, g_mat, dt_mat = self.base._shard_cache[shard_i]
            N = int(y_mat.shape[0])
            traj = torch.randint(0, N, (B,), device=self.device, generator=self._gen_dev, dtype=torch.long)
            start = torch.randint(
                0,
                self.base.max_start + 1,
                (B,),
                device=self.device,
                generator=self._gen_dev,
                dtype=torch.long,
            )

            # Gather windows without materializing [B, T, S] intermediates.
            # y_mat: [N, T, S]  dt_mat: [N, T-1]
            t_y = start[:, None] + self._t_y[None, :]  # [B, K+1]
            lin_y = (traj[:, None] * self.T + t_y).reshape(-1)
            y_flat = y_mat.reshape(-1, self.S)  # [N*T, S]
            y_win = y_flat.index_select(0, lin_y).reshape(B, self.K + 1, self.S)

            t_dt = start[:, None] + self._t_dt[None, :]  # [B, K]
            lin_dt = (traj[:, None] * self.T_dt + t_dt).reshape(-1)
            dt_flat = dt_mat.reshape(-1)  # [N*(T-1)]
            dt_win = dt_flat.index_select(0, lin_dt).reshape(B, self.K)

            g = g_mat.index_select(0, traj)

            yield {"y": y_win, "dt": dt_win, "g": g}



def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    drop_last: bool,
) -> DataLoader:
    """
    Create a DataLoader with safe defaults for this codebase.

    Notes for preload_to_device=True:
    - The loader will switch to a GPU-native stochastic batch stream (sampling with replacement).
    """
    preload = bool(getattr(dataset, "preload_to_device", False))
    if preload and num_workers != 0:
        raise ValueError(
            "preload_to_device=True requires num_workers=0. "
            "Preloaded data cannot be shared across worker processes."
        )
    # Dataset size checks
    try:
        ds_len = len(dataset)
    except TypeError:
        ds_len = None

    if ds_len is not None and ds_len == 0:
        raise ValueError("Dataset is empty - check preprocessing outputs.")

    if drop_last and ds_len is not None and ds_len < batch_size:
        raise ValueError(
            f"drop_last=True would yield 0 batches (dataset size {ds_len} < batch_size {batch_size})."
        )

    if preload and pin_memory:
        raise ValueError("pin_memory=True is incompatible with preload_to_device=True (data is already on device).")

    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers=True requires num_workers>0.")

    if num_workers == 0 and prefetch_factor is not None:
        raise ValueError("prefetch_factor requires num_workers>0.")

    # ---- Optimized GPU stream for preloaded FlowMapRolloutDataset ----
    if preload and isinstance(dataset, FlowMapRolloutDataset) and getattr(dataset, "_preloaded", False):
        stream: IterableDataset = _GPUPreloadedStochasticBatchStream(
            dataset,
            batch_size=batch_size,
            drop_last=bool(drop_last),
            seed=getattr(dataset, "seed", 1234),
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

    # prefetch_factor is only meaningful when using worker processes. Some PyTorch versions
    # error if it is passed when num_workers==0, so only forward it when applicable.
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)
