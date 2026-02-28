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

import zipfile
from numpy.lib import format as npy_format

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


_DT_PROBE_MAX = 1024
_DT_RANGE_EPS = 1e-6

# -----------------------------------------------------------------------------
# NPZ header helpers (avoid decompressing full arrays)
# -----------------------------------------------------------------------------

def _npz_read_array_header(path: Path, *, key: str) -> Tuple[Tuple[int, ...], np.dtype, bool]:
    """Return (shape, dtype, fortran_order) for `key` inside an .npz.

    NumPy's np.load(...)[key] materializes the full array (and therefore fully
    decompresses compressed shards). For dataset indexing we often only need
    the header metadata (shape/dtype). This helper reads the embedded .npy
    header directly from the zip member and avoids touching the bulk data.
    """
    inner = f"{key}.npy"
    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(inner) as f:
                ver = npy_format.read_magic(f)
                if ver == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(f)
                elif ver == (2, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(f)
                else:
                    # Private helper supports additional header versions (e.g., 3.0) without forcing
                    # us to depend on newer NumPy public APIs.
                    shape, fortran_order, dtype = npy_format._read_array_header(f, ver)
        shape_t = tuple(int(s) for s in shape)
        return shape_t, np.dtype(dtype), bool(fortran_order)
    except Exception:
        # Fallback for unusual containers: materialize via NumPy (slow).
        with np.load(path) as z:
            arr = z[key]
        return tuple(int(s) for s in arr.shape), arr.dtype, bool(arr.flags["F_CONTIGUOUS"])


def _npz_read_row0_prefix(path: Path, *, key: str, max_elems: int) -> np.ndarray:
    """Read up to `max_elems` elements from row 0 of a 2D float array in an .npz.

    Used for fast dt range validation without inflating a full shard into RAM.
    """
    inner = f"{key}.npy"
    take = int(max(0, max_elems))
    if take == 0:
        return np.empty((0,), dtype=np.float32)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(inner) as f:
                ver = npy_format.read_magic(f)
                if ver == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(f)
                elif ver == (2, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(f)
                else:
                    # Private helper supports additional header versions (e.g., 3.0) without forcing
                    # us to depend on newer NumPy public APIs.
                    shape, fortran_order, dtype = npy_format._read_array_header(f, ver)

                shape_t = tuple(int(s) for s in shape)
                if len(shape_t) != 2:
                    raise ValueError(f"{key} expected 2D array, got shape {shape_t}")
                n0, m0 = int(shape_t[0]), int(shape_t[1])
                if n0 <= 0 or m0 <= 0:
                    return np.empty((0,), dtype=np.dtype(dtype))

                if bool(fortran_order):
                    # Uncommon; fall back to NumPy materialization.
                    raise ValueError("fortran_order")

                dt = np.dtype(dtype)
                take_eff = int(min(take, m0))
                nbytes = int(take_eff * dt.itemsize)
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise ValueError(f"Truncated member {inner} in {path}")
                return np.frombuffer(buf, dtype=dt, count=take_eff)
    except Exception:
        with np.load(path) as z:
            arr = z[key]
        if arr.ndim != 2:
            raise ValueError(f"{key} expected 2D array, got shape {arr.shape}")
        take_eff = int(min(int(arr.shape[1]), take))
        return np.asarray(arr[0, :take_eff])


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
            shard_cache_size: LRU size for cached shards when loading on-demand on CPU.
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

        # Peek first shard for basic schema + dimensions (header-only; avoids full decompression).
        y_shape0, y_dtype0, _ = _npz_read_array_header(shard_paths[0], key="y_mat")
        g_shape0, g_dtype0, _ = _npz_read_array_header(shard_paths[0], key="globals")
        dt_shape0, dt_dtype0, _ = _npz_read_array_header(shard_paths[0], key="dt_norm_mat")

        if len(y_shape0) != 3:
            raise ValueError(f"Expected y_mat to be 3D [N,T,S], got {y_shape0}")
        if len(g_shape0) != 2:
            raise ValueError(f"Expected globals to be 2D [N,G], got {g_shape0}")
        if len(dt_shape0) != 2:
            raise ValueError(f"Expected dt_norm_mat to be 2D [N,T-1], got {dt_shape0}")

        N0, T0, S0 = (int(y_shape0[0]), int(y_shape0[1]), int(y_shape0[2]))
        if int(dt_shape0[1]) != T0 - 1:
            raise ValueError(f"dt_norm_mat second dim must be T-1={T0-1}, got {int(dt_shape0[1])}")
        if int(g_shape0[0]) != N0:
            raise ValueError(f"globals first dim must match y_mat N={N0}, got {int(g_shape0[0])}")
        if int(dt_shape0[0]) != N0:
            raise ValueError(f"dt_norm_mat first dim must match y_mat N={N0}, got {int(dt_shape0[0])}")

        if T0 < self.total_steps + 1:
            raise ValueError(f"Trajectories too short for total_steps={self.total_steps}.")

        self.T = int(T0)
        self.S = int(S0)
        self.G = int(g_shape0[1])

        self.max_start = int(self.T - (self.total_steps + 1))
        if self.max_start < 0:
            raise ValueError(f"max_start < 0; T={self.T} total_steps={self.total_steps}")
        # Strict dtypes: avoid silent CPU copies / perf cliffs.
        if y_dtype0 != np.float32:
            raise ValueError(f"y_mat dtype must be float32, got {y_dtype0}. Regenerate preprocessing outputs.")
        if g_dtype0 != np.float32:
            raise ValueError(f"globals dtype must be float32, got {g_dtype0}. Regenerate preprocessing outputs.")
        if dt_dtype0 != np.float32:
            raise ValueError(f"dt_norm_mat dtype must be float32, got {dt_dtype0}. Regenerate preprocessing outputs.")

        # Fast dt range sanity check on a small slice of the first trajectory.
        if validate_dt_range:
            dt_probe = _npz_read_row0_prefix(shard_paths[0], key="dt_norm_mat", max_elems=_DT_PROBE_MAX)
            if np.any(dt_probe < -_DT_RANGE_EPS) or np.any(dt_probe > 1.0 + _DT_RANGE_EPS):
                raise ValueError(
                    "dt_norm_mat contains values outside [0,1]. The training code expects dt already normalized. "
                    "Regenerate preprocessing outputs."
                )

        # Build shard index (trajectory counts per shard) using header-only reads.
        self.shards: List[ShardIndex] = []
        for p in shard_paths:
            y_shape, y_dtype, _ = _npz_read_array_header(p, key="y_mat")
            g_shape, g_dtype, _ = _npz_read_array_header(p, key="globals")
            dt_shape, dt_dtype, _ = _npz_read_array_header(p, key="dt_norm_mat")

            if len(y_shape) != 3:
                raise ValueError(f"Shard {p} y_mat must be 3D [N,T,S], got {y_shape}")
            if len(g_shape) != 2:
                raise ValueError(f"Shard {p} globals must be 2D [N,G], got {g_shape}")
            if len(dt_shape) != 2:
                raise ValueError(f"Shard {p} dt_norm_mat must be 2D [N,T-1], got {dt_shape}")

            n, t, s = (int(y_shape[0]), int(y_shape[1]), int(y_shape[2]))
            if t != self.T or s != self.S:
                raise ValueError(f"Shard {p} y_mat shape is {y_shape}; expected [N,{self.T},{self.S}].")
            if y_dtype != np.float32:
                raise ValueError(f"Shard {p} y_mat dtype must be float32, got {y_dtype}.")
            if g_dtype != np.float32:
                raise ValueError(f"Shard {p} globals dtype must be float32, got {g_dtype}.")
            if dt_dtype != np.float32:
                raise ValueError(f"Shard {p} dt_norm_mat dtype must be float32, got {dt_dtype}.")

            if int(g_shape[0]) != n or int(g_shape[1]) != self.G:
                raise ValueError(f"Shard {p} globals shape is {g_shape}; expected [{n},{self.G}].")
            if int(dt_shape[0]) != n or int(dt_shape[1]) != self.T - 1:
                raise ValueError(f"Shard {p} dt_norm_mat shape is {dt_shape}; expected [{n},{self.T - 1}].")

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
        self._preloaded_all: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        # CPU caches
        self._npz_cache_full: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

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
        """Load all shards into device memory (requires num_workers=0).

        Implementation note:
          We build a single contiguous [N,T,S] / [N,G] / [N,T-1] buffer and then keep
          per-shard *views* into it. This keeps the public (shard-indexed) interface
          intact while enabling efficient uniform sampling across the full dataset.
        """
        self._shard_cache.clear()
        self._preloaded_all = None

        total = int(self._total_traj)
        if total <= 0:
            raise RuntimeError("no trajectories to preload")

        # Allocate contiguous buffers on the target device.
        y_all = torch.empty((total, self.T, self.S), device=self.device, dtype=self.storage_dtype)
        g_all = torch.empty((total, self.G), device=self.device, dtype=self.storage_dtype)
        dt_all = torch.empty((total, self.T - 1), device=self.device, dtype=self.storage_dtype)

        cursor = 0
        for si in self.shards:
            with np.load(si.path) as z:
                # Force float32 on CPU; dtype conversion (if any) happens during copy_ into the destination.
                y_np = np.asarray(z["y_mat"], dtype=np.float32)
                g_np = np.asarray(z["globals"], dtype=np.float32)
                dt_np = np.asarray(z["dt_norm_mat"], dtype=np.float32)

            n = int(y_np.shape[0])
            if n != int(si.n):
                raise ValueError(f"Shard {si.path} count mismatch: index says {si.n}, file has {n}.")
            if cursor + n > total:
                raise RuntimeError("preload overflow (bug)")

            y_all[cursor : cursor + n].copy_(torch.from_numpy(y_np), non_blocking=False)
            g_all[cursor : cursor + n].copy_(torch.from_numpy(g_np), non_blocking=False)
            dt_all[cursor : cursor + n].copy_(torch.from_numpy(dt_np), non_blocking=False)

            # Per-shard views (no extra allocation).
            self._shard_cache.append(
                (y_all[cursor : cursor + n], g_all[cursor : cursor + n], dt_all[cursor : cursor + n])
            )
            cursor += n

        if cursor != total:
            raise RuntimeError(f"preload size mismatch: filled {cursor}, expected {total}")
        self._preloaded_all = (y_all, g_all, dt_all)
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

    def _get_traj_arrays(self, shard_i: int, local_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays for a single trajectory.

        NPZ is a compressed container format; NumPy cannot memory-map individual
        arrays out of it. For CPU loading this dataset therefore caches whole
        shards in RAM (LRU) and slices per-trajectory views from those arrays.
        """
        y_all, g_all, dt_all = self._get_full_shard_arrays(shard_i)
        return y_all[local_i], g_all[local_i], dt_all[local_i]

    def close(self) -> None:
        """Drop cached shard arrays."""
        self._npz_cache_full.clear()

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
        if base._preloaded_all is None:
            raise ValueError("_GPUPreloadedStochasticBatchStream requires base._preloaded_all (contiguous preload)")
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

        self._y_all, self._g_all, self._dt_all = base._preloaded_all
        self.N = int(self._y_all.shape[0])

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

        self._seed = int(seed if seed is not None else base.seed)

        # Sample indices on-device for efficiency.
        try:
            self._gen_dev = torch.Generator(device=self.device)
        except TypeError as e:
            raise RuntimeError(
                "preload_to_device=True requires torch.Generator(device=...) support. "
                "Upgrade PyTorch or set dataset.preload_to_device=false."
            ) from e
        # Use +1 to preserve the historical separation between CPU and device RNG streams.
        self._gen_dev.manual_seed(self._seed + 1)

    def __len__(self) -> int:
        total = len(self.base)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        total_samples = len(self.base)
        num_batches = len(self)

        # Views (no allocation).
        y_flat = self._y_all.reshape(-1, self.S)  # [N*T, S]
        dt_flat = self._dt_all.reshape(-1)        # [N*(T-1)]
        g_mat = self._g_all                       # [N, G]

        for b in range(num_batches):
            remaining = total_samples - b * self.batch_size
            B = self.batch_size if (self.drop_last or remaining >= self.batch_size) else remaining
            if B <= 0:
                break

            traj = torch.randint(0, self.N, (B,), device=self.device, generator=self._gen_dev, dtype=torch.long)
            start = torch.randint(
                0,
                self.base.max_start + 1,
                (B,),
                device=self.device,
                generator=self._gen_dev,
                dtype=torch.long,
            )

            # Gather windows without materializing [B, T, S] intermediates.
            t_y = start[:, None] + self._t_y[None, :]  # [B, K+1]
            lin_y = (traj[:, None] * self.T + t_y).reshape(-1)
            y_win = y_flat.index_select(0, lin_y).reshape(B, self.K + 1, self.S)

            t_dt = start[:, None] + self._t_dt[None, :]  # [B, K]
            lin_dt = (traj[:, None] * self.T_dt + t_dt).reshape(-1)
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
