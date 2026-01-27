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

Change log vs the prior version of this repo:
- Deterministic sampling (deterministic window positions and deterministic
  GPU batch stream) has been removed. All sampling is stochastic.
- `use_mmap=True` no longer re-opens the NPZ file per sample. Instead, each
  worker keeps an LRU cache of open NPZ handles (mmap_mode="r"), eliminating
  per-sample file-open overhead.
- The dataset no longer constructs an unused NormalizationHelper instance.
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
        random_windows: Optional[bool] = None,
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
            random_windows: Deprecated. Deterministic sampling was removed; windows are always random.
                            If provided and False, a warning is emitted and random sampling is used.
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

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if windows_per_trajectory <= 0:
            raise ValueError(f"windows_per_trajectory must be positive, got {windows_per_trajectory}")

        self.processed_dir = Path(processed_dir)
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.preload_to_device = bool(preload_to_device)
        self.device = device
        self.storage_dtype = storage_dtype
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.use_mmap = bool(use_mmap) and not self.preload_to_device

        # Deterministic windows were removed. Keep the parameter for backward compatibility only.
        if random_windows is not None and not bool(random_windows):
            warnings.warn(
                "random_windows=False is no longer supported (deterministic sampling removed). "
                "Proceeding with stochastic window sampling.",
                UserWarning,
                stacklevel=2,
            )

        # Normalization manifest is required by the overall pipeline; dataset does not apply it.
        self.manifest: Optional[Dict] = None
        if validate_manifest:
            self._load_manifest()

        self._index_shards(validate_dt_range=validate_dt_range)

        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # CPU path caches:
        # - _npz_cache_full caches fully-loaded shard arrays (y_all, g_all, dt_all)
        # - _npz_handle_cache caches open NPZ handles when use_mmap=True (avoids per-sample np.load)
        self._npz_cache_full: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        self._npz_handle_cache: "OrderedDict[int, np.lib.npyio.NpzFile]" = OrderedDict()

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
            self.manifest = json.load(f)

    def _index_shards(self, *, validate_dt_range: bool) -> None:
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

        # Probe the first shard for shape/dtype checks.
        with np.load(shard_paths[0]) as first:
            y0 = first["y_mat"]
            g0 = first["globals"]
            dt0 = first["dt_norm_mat"]

        # Shape checks
        if y0.ndim != 3:
            raise ValueError(f"y_mat must be 3D [N,T,S], got shape {y0.shape}")
        if g0.ndim != 2:
            raise ValueError(f"globals must be 2D [N,G], got shape {g0.shape}")
        if dt0.ndim != 2:
            raise ValueError(f"dt_norm_mat must be 2D [N,T-1], got shape {dt0.shape}")

        self.T = int(y0.shape[1])
        self.S = int(y0.shape[2])
        self.G = int(g0.shape[1])

        if int(dt0.shape[1]) != self.T - 1:
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

        # Dtype checks: strongly prefer float32 in preprocessing to avoid per-sample copies.
        if y0.dtype != np.float32:
            warnings.warn(
                f"y_mat dtype is {y0.dtype}; preprocessing should store float32 to avoid copies.",
                UserWarning,
                stacklevel=2,
            )
        if g0.dtype != np.float32:
            warnings.warn(
                f"globals dtype is {g0.dtype}; preprocessing should store float32 to avoid copies.",
                UserWarning,
                stacklevel=2,
            )
        if dt0.dtype != np.float32:
            warnings.warn(
                f"dt_norm_mat dtype is {dt0.dtype}; preprocessing should store float32 to avoid copies.",
                UserWarning,
                stacklevel=2,
            )

        # Fast dt range sanity check on a small slice (avoids reading full array).
        if validate_dt_range:
            # sample up to 1024 values from the first trajectory
            dt_probe = np.asarray(dt0[0, : min(1024, dt0.shape[1])])
            if np.any(dt_probe < -1e-6) or np.any(dt_probe > 1.0 + 1e-6):
                warnings.warn(
                    "dt_norm_mat appears to contain values outside [0,1]. "
                    "The model expects dt normalized to [0,1] (log10 + min-max). "
                    "Check preprocessing.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build shard index (trajectory counts per shard)
        self.shards = []
        for p in shard_paths:
            with np.load(p) as z:
                n = int(z["y_mat"].shape[0])
            if n <= 0:
                raise ValueError(f"Shard {p} has no trajectories.")
            self.shards.append(ShardIndex(path=p, n=n))

        self._total_traj = int(sum(si.n for si in self.shards))

        # Prefix offsets in trajectory space (global trajectory id -> shard).
        self._shard_offsets = [0]
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
            # np.asarray(dtype=np.float32) will copy if stored dtype mismatches;
            # preprocessing should store float32 to keep this zero-copy.
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)
            dt_all = np.asarray(z["dt_norm_mat"], dtype=np.float32)

        self._npz_cache_full[shard_i] = (y_all, g_all, dt_all)
        while len(self._npz_cache_full) > self.shard_cache_size:
            self._npz_cache_full.popitem(last=False)
        return y_all, g_all, dt_all

    def _get_mmap_npz(self, shard_i: int) -> np.lib.npyio.NpzFile:
        """
        Get an open NPZ handle with mmap_mode="r" from an LRU cache.

        This eliminates the per-sample np.load(...) overhead that the previous implementation had.
        Note: true OS-level mmap depends on how the NPZ was produced; for best results, store
        float32 arrays and avoid compression.
        """
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
        """
        Return numpy arrays for a single trajectory: (y_traj[T,S], g[G], dt_traj[T-1]).
        """
        if self.use_mmap:
            h = self._get_mmap_npz(shard_i)
            y_traj = h["y_mat"][local_i]
            g = h["globals"][local_i]
            dt_traj = h["dt_norm_mat"][local_i]

            # Avoid copies when possible; enforce float32 only if needed.
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

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

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
                # Drop any stale entries for worker_id=0
                for k in list(self._rng_cache.keys()):
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == 0:
                        del self._rng_cache[k]
                ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(torch_seed,))
                self._rng_cache[cache_key] = np.random.default_rng(ss)
            return self._rng_cache[cache_key]

        torch_seed = int(torch.initial_seed()) % (2**32)
        cache_key = (info.id, torch_seed)

        if cache_key not in self._rng_cache:
            # Drop any stale entries for this worker id
            keys_to_remove = [
                k
                for k in self._rng_cache.keys()
                if isinstance(k, tuple) and len(k) == 2 and k[0] == info.id
            ]
            for k in keys_to_remove:
                del self._rng_cache[k]
            ss = np.random.SeedSequence(entropy=self.seed, spawn_key=(info.id, torch_seed))
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

        shard_i = bisect_right(self._shard_offsets, traj_idx) - 1
        local_i = int(traj_idx - self._shard_offsets[shard_i])

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

    Sampling is with replacement (SGD-style). This is used for both train and validation/test
    after removing deterministic sampling from this repo.
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

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

        counts = torch.tensor([si.n for si in base.shards], device=self.device, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError("Encountered a shard with non-positive trajectory count.")
        self._shard_weights = counts / counts.sum()

        self._seed = int(seed if seed is not None else base.seed)

        # RNG handling:
        # Prefer a device generator so random index generation stays on-GPU.
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

            # Gather windows
            y_btS = y_mat.index_select(0, traj)
            idx_y = (start[:, None] + self._t_y[None, :]).unsqueeze(-1).expand(-1, -1, self.S)
            y_win = torch.gather(y_btS, dim=1, index=idx_y)

            dt_bT = dt_mat.index_select(0, traj)
            idx_dt = (start[:, None] + self._t_dt[None, :])
            dt_win = torch.gather(dt_bT, dim=1, index=idx_dt)

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
    prefetch_factor: int,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader with safe defaults for this codebase.

    Notes for preload_to_device=True:
    - The loader will switch to a GPU-native stochastic batch stream (sampling with replacement).
    - Deterministic/exhaustive sampling was removed from this repo per request.
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
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)
