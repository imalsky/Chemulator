# dataset.py
#!/usr/bin/env python3
"""
dataset.py

Rollout dataset for constant-Δt, long-horizon autoregressive training.

This dataset assumes preprocessing has already produced *normalized* (z-space)
arrays and a normalization manifest. No preprocessing happens here.

Directory layout (processed_data_dir):
  normalization.json
  preprocessing_summary.json
  train/shard_*.npz
  validation/shard_*.npz
  test/shard_*.npz (optional)

Each shard NPZ:
  - y_mat   : [N, T, S] float32 (z-space)
  - globals : [N, G]    float32 (z-space)
  - t_vec   : [T]       float64 (shared relative time vector: 0, dt, 2dt, ...)

Dataset returns (per sample):
  y0      : [S]         z-space at time index t
  dt_norm : scalar      normalized dt (same for all steps; constant dt)
  y_seq   : [K, S]      z-space targets for next K steps (t+1..t+K)
  g       : [G]         z-space globals
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bisect import bisect_right
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from normalizer import NormalizationHelper


@dataclass
class ShardIndex:
    path: Path
    n: int


class FlowMapRolloutDataset(Dataset):
    """
    Samples random windows from trajectories stored in NPZ shards.
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
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.preload_to_device = bool(preload_to_device)
        self.device = device
        self.storage_dtype = storage_dtype

        # Load normalization manifest
        man_path = self.processed_dir / "normalization.json"
        if not man_path.exists():
            raise FileNotFoundError(f"Missing {man_path}. Run preprocessing first.")
        with open(man_path, "r") as f:
            self.manifest: Dict = json.load(f)
        self.norm = NormalizationHelper(self.manifest)

        # Load shard indices
        self.shards: List[ShardIndex] = []
        split_dir = self.processed_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir {split_dir}")

        shard_paths = sorted(split_dir.glob("shard_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.npz found in {split_dir}")

        # Determine dimensions and time vector from the first shard
        with np.load(shard_paths[0]) as first:
            y0 = first["y_mat"]
            g0 = first["globals"]
            t0 = first["t_vec"]

        if y0.ndim != 3:
            raise ValueError("y_mat must be [N,T,S]")
        if g0.ndim != 2:
            raise ValueError("globals must be [N,G]")
        if t0.ndim != 1:
            raise ValueError("t_vec must be [T]")

        self.N_per_shard = int(y0.shape[0])
        self.T = int(y0.shape[1])
        self.S = int(y0.shape[2])
        self.G = int(g0.shape[1])

        # Shared t_vec (relative)
        self.t_vec = np.asarray(t0, dtype=np.float64)

        # Index all shards and verify shared t_vec
        self._load_shards(shard_paths)

        # Prefix sums for O(log N_shards) shard lookup (avoids linear scan in __getitem__).
        self._shard_offsets: List[int] = [0]
        for si in self.shards:
            self._shard_offsets.append(self._shard_offsets[-1] + int(si.n))
        self._total_traj = int(self._shard_offsets[-1])

        # Validate constant dt and compute dt_norm
        if self.T < 2:
            raise ValueError("t_vec length must be >= 2")

        # Use float64 for robust Δt checks over long horizons.
        t64 = np.asarray(self.t_vec, dtype=np.float64)
        dt_phys = float(t64[1] - t64[0])
        diffs = np.diff(t64)
        atol = 1e-12 * max(1.0, abs(dt_phys))
        if not np.allclose(diffs, dt_phys, rtol=1e-6, atol=atol):
            raise ValueError(
                "t_vec is not constant-Δt (within tolerance). Preprocessing must output a shared constant-Δt relative grid."
            )

        self.dt_phys = dt_phys
        dt_norm = self.norm.normalize_dt_from_phys(torch.as_tensor([dt_phys], dtype=torch.float64))[0]
        self.dt_norm = dt_norm.to(dtype=self.storage_dtype)

        # Precompute allowable start range for windows
        self.max_start = self.T - self.total_steps - 1
        if self.max_start < 0:
            raise ValueError(
                f"Not enough time steps T={self.T} for total_steps={self.total_steps}. Need T >= total_steps+2."
            )

        # Optional preloading
        self._preloaded = False
        self._shard_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Worker-local RNG streams (one per DataLoader worker id). Each worker process
        # owns its dataset copy, so this cache is process-local.
        self._rng_cache: Dict[int, np.random.Generator] = {}

        # LRU cache of decoded NPZ shard arrays (per worker process).
        # This avoids re-reading entire shards from disk for each sample.
        self._npz_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._npz_cache_max = 2
        if self.preload_to_device:
            self._preload_all()

    def _load_shards(self, shard_paths: List[Path]) -> None:
        t_ref = None
        for p in shard_paths:
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float32)
                g = np.asarray(z["globals"], dtype=np.float32)
                t = np.asarray(z["t_vec"], dtype=np.float64)

            if y.shape[1] != self.T or y.shape[2] != self.S:
                raise ValueError(f"Shard {p} has inconsistent y_mat shape {y.shape}, expected (*,{self.T},{self.S})")
            if g.shape[1] != self.G:
                raise ValueError(f"Shard {p} has inconsistent globals shape {g.shape}, expected (*,{self.G})")
            if t.shape[0] != self.T:
                raise ValueError(f"Shard {p} has inconsistent t_vec length {t.shape[0]}, expected {self.T}")

            if t_ref is None:
                t_ref = t
            else:
                # Must be identical across shards
                if not np.allclose(t, t_ref, rtol=0.0, atol=0.0):
                    raise ValueError(f"Shard {p} has a different t_vec. All shards must share the same time grid.")

            self.shards.append(ShardIndex(path=p, n=int(y.shape[0])))

        if t_ref is None:
            raise ValueError("No shards loaded")
        self.t_vec = t_ref

    def _preload_all(self) -> None:
        # NOTE: Preloading onto a device should only be done with num_workers=0.
        self._shard_cache.clear()
        for si in self.shards:
            with np.load(si.path) as z:
                y = torch.tensor(z["y_mat"], dtype=self.storage_dtype, device=self.device)
                g = torch.tensor(z["globals"], dtype=self.storage_dtype, device=self.device)
            self._shard_cache.append((y, g))
        self._preloaded = True

    def _get_npz_shard_arrays(self, shard_i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_mat, globals) numpy arrays for a shard, using a small LRU cache."""
        if self._preloaded:
            raise RuntimeError("_get_npz_shard_arrays should not be called when preloaded")
        if shard_i in self._npz_cache:
            y_all, g_all = self._npz_cache.pop(shard_i)
            self._npz_cache[shard_i] = (y_all, g_all)  # move to MRU
            return y_all, g_all
        with np.load(self.shards[shard_i].path) as z:
            y_all = np.asarray(z["y_mat"], dtype=np.float32)
            g_all = np.asarray(z["globals"], dtype=np.float32)
        self._npz_cache[shard_i] = (y_all, g_all)
        # evict LRU
        while len(self._npz_cache) > self._npz_cache_max:
            self._npz_cache.popitem(last=False)
        return y_all, g_all

    def __len__(self) -> int:
        # Each trajectory yields `windows_per_trajectory` random windows per epoch.
        return self._total_traj * self.windows_per_trajectory

    def _rng_for_worker(self) -> np.random.Generator:
        info = get_worker_info()
        wid = 0 if info is None else int(info.id)
        rng = self._rng_cache.get(wid)
        if rng is None:
            # Each worker gets a deterministic but different stream.
            rng = np.random.default_rng(self.seed + 1000 * wid)
            self._rng_cache[wid] = rng
        return rng

    def __getitem__(self, idx: int):
        rng = self._rng_for_worker()

        # Map idx to a trajectory (ignoring "window id" on purpose; windows are random)
        traj_idx = idx // self.windows_per_trajectory
        if traj_idx < 0 or traj_idx >= self._total_traj:
            raise IndexError("Index out of bounds")

        # Find which shard contains this traj_idx via prefix sums (O(log N_shards))
        shard_i = bisect_right(self._shard_offsets, int(traj_idx)) - 1
        local_i = int(traj_idx) - int(self._shard_offsets[shard_i])

        # Choose a random start
        start = int(rng.integers(0, self.max_start + 1))
        end = start + self.total_steps + 1  # inclusive end index for targets slice

        if self._preloaded:
            y_mat, g_mat = self._shard_cache[shard_i]
            y_traj = y_mat[local_i]           # [T,S]
            g = g_mat[local_i]                # [G]
        else:
            y_all, g_all = self._get_npz_shard_arrays(shard_i)
            y_np = y_all[local_i]  # [T,S] float32
            g_np = g_all[local_i]  # [G]   float32
            if self.storage_dtype == torch.float32:
                y_traj = torch.from_numpy(y_np)
                g = torch.from_numpy(g_np)
            else:
                y_traj = torch.as_tensor(y_np, dtype=self.storage_dtype)
                g = torch.as_tensor(g_np, dtype=self.storage_dtype)

        # Build window
        y0 = y_traj[start, :]                             # [S]
        y_seq = y_traj[start + 1:end, :]                  # [K,S] where K=total_steps
        dt_norm = self.dt_norm                            # scalar tensor

        # If not preloaded, optionally move to device here
        if self.preload_to_device and (not self._preloaded):
            y0 = y0.to(self.device, non_blocking=True)
            y_seq = y_seq.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)
            dt_norm = dt_norm.to(self.device, non_blocking=True)

        return y0, dt_norm, y_seq, g


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
    """Create a DataLoader with sane defaults.

    Important:
      - If dataset.preload_to_device=True, you usually want num_workers=0 to avoid
        each worker separately loading shards onto GPU/CPU device memory.
      - torch DataLoader does not accept prefetch_factor when num_workers==0 on
        some versions; only pass it when num_workers>0.
      - Do NOT default drop_last=True purely because shuffle=True. If the dataset is
        smaller than batch_size, drop_last=True yields *zero* batches and Lightning
        will stop with: "Trainer.fit stopped: No training batches".
    """
    if getattr(dataset, "preload_to_device", False) and num_workers != 0:
        raise ValueError("preload_to_device=True is incompatible with num_workers>0; set num_workers=0.")

    if num_workers == 0:
        persistent_workers = False

    # Fail fast for empty datasets.
    try:
        ds_len = len(dataset)
    except Exception:
        ds_len = None

    if ds_len is not None and int(ds_len) == 0:
        split = getattr(dataset, "split", None)
        processed_dir = getattr(dataset, "processed_dir", None)
        hint = ""
        if processed_dir is not None and split is not None:
            hint = f" (processed_dir={processed_dir}, split={split})"
        raise ValueError(
            "Dataset length is zero. This will produce zero DataLoader batches and training cannot run." + hint
        )

    # Resolve drop_last behavior.
    if drop_last is None:
        # Only drop the last incomplete batch if we have at least one full batch.
        if shuffle and ds_len is not None and int(ds_len) >= int(batch_size):
            drop_last_val = True
        else:
            drop_last_val = False
    else:
        drop_last_val = bool(drop_last)
        # Guardrail: never allow drop_last=True to silently create 0 batches.
        if drop_last_val and ds_len is not None and int(ds_len) < int(batch_size):
            import warnings

            warnings.warn(
                f"drop_last=True with len(dataset)={ds_len} < batch_size={batch_size} would yield 0 batches; "
                "forcing drop_last=False.",
                UserWarning,
                stacklevel=2,
            )
            drop_last_val = False

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        drop_last=bool(drop_last_val),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)
