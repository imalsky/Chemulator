#!/usr/bin/env python3
"""
Dataset for flow-map DeepONet in Δt (autonomous) mode.

- Loads NPZ shards from preprocessor.py
- Normalizes globals, absolute time, and targets via NormalizationHelper on the target device
- Randomly samples (i, j) with j > i; trunk input is normalized Δt = t_j − t_i
- Vectorized, GPU-friendly batching; GPU-resident dataset without host-RAM spikes
"""

from __future__ import annotations

import logging
import time
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from normalizer import NormalizationHelper
from utils import load_json


# ------------------------------- Small utils --------------------------------

def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


# ------------------------------- I/O helpers --------------------------------

def _discover_split_files(root: Path, split: str) -> List[Path]:
    shard_dir = root / split
    files = sorted(map(Path, glob(str(shard_dir / "*.npz"))))
    if not files:
        raise FileNotFoundError(f"No NPZ shards found under: {shard_dir}")
    return files


def _load_npz_keys(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Force materialization (avoid memmap lifetime issues)
    with np.load(path, mmap_mode=None) as d:
        x0 = np.array(d["x0"], copy=True)         # [N, S_in] (unused here)
        g = np.array(d["globals"], copy=True)     # [N, G]
        t = np.array(d["t_vec"], copy=True)       # [N, T] (PHYSICAL absolute time)
        y = np.array(d["y_mat"], copy=True)       # [N, T, S]
    return x0, g, t, y


# --------------------------- SplitMix64 on CPU (numpy) ----------------------

_MASK64 = np.uint64((1 << 64) - 1)
_C1 = np.uint64(0xbf58476d1ce4e5b9)
_C2 = np.uint64(0x94d049bb133111eb)


def _mix64_np(x: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(x, np.right_shift(x, np.uint64(30)))
    x = (x * _C1) & _MASK64
    x = np.bitwise_xor(x, np.right_shift(x, np.uint64(27)))
    x = (x * _C2) & _MASK64
    x = np.bitwise_xor(x, np.right_shift(x, np.uint64(31)))
    return x


def _u32_from_array(seed: int, a: np.ndarray) -> np.ndarray:
    """Single 32-bit stream from seed and array."""
    s = np.uint64(seed) & _MASK64
    a64 = (np.uint64(a) & _MASK64)
    x = s ^ (a64 << np.uint64(1))
    r = _mix64_np(x)
    return np.uint32(r & np.uint64(0xFFFFFFFF))


# ------------------------------- Dataset class ------------------------------

class FlowMapPairsDataset(Dataset):
    """
    Dataset that samples arbitrary (i, j>i) pairs per trajectory.
    Trunk input is Δt (normalized with log-min-max bounds derived from the first shard’s grid).
    """

    def __init__(
        self,
        processed_root: Path | str,
        split: str,
        config: dict,
        pairs_per_traj: int = 64,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        preload_to_gpu: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        log_every_files: int = 5,
    ):
        """
        Autonomous flow-map dataset (Δt-only), GPU-resident without host-RAM spikes.

        • Pass #1: scan shard shapes ONLY to compute N_total and grab the first time grid.
        • Pre-allocate final tensors (G, t, Y) directly on target device (GPU if given).
        • Pass #2: stream each shard, normalize on device, and write into the pre-alloc tensors.
        • Δt bounds derived once from the FIRST shard's grid over [min_steps, max_steps].
          Assumes all shards share the same time grid (warns once).
        • Honors training.use_fraction for the train split (optional downsampling).
        """
        super().__init__()
        self.root = Path(processed_root)
        self.split = split
        self.cfg = config
        self.base_seed = int(seed)
        self.log = logging.getLogger("dataset")

        # ---- Normalizer & devices ----
        t0_all = time.perf_counter()
        norm_path = self.root / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")
        norm_stats = load_json(norm_path)

        self.preload_to_gpu = bool(preload_to_gpu)
        self._target_device = device if (self.preload_to_gpu and device is not None) else torch.device("cpu")
        self._norm_device = self._target_device
        self.norm = NormalizationHelper(stats=norm_stats, device=self._norm_device, config=self.cfg)

        data_cfg = self.cfg.get("data", {})
        self.target_vars = list(data_cfg.get("target_species_variables", data_cfg.get("species_variables", [])))
        self.global_vars = list(data_cfg.get("global_variables", []))
        self.time_var = data_cfg.get("time_variable", "t_time")

        self.epoch = 0  # for epoch-fresh sampling

        # ---- Discover shards; optionally subsample for training ----
        self.log.info(f"[{split}] discovering shards under {self.root / split}")
        files = _discover_split_files(self.root, split)

        use_fraction = float(self.cfg.get("training", {}).get("use_fraction", 1.0))
        if split == "train" and 0.0 < use_fraction < 1.0:
            rng = np.random.RandomState(self.base_seed)
            idx = np.arange(len(files))
            rng.shuffle(idx)
            k = max(1, int(round(use_fraction * len(files))))
            files = [files[i] for i in idx[:k]]
            self.log.warning("[train] training.use_fraction=%.3f -> using %d/%d shards",
                             use_fraction, k, len(idx))

        self.log.info(f"[{split}] found {len(files)} shard(s)")

        # ---- PASS 1: shapes only + Δt-bounds from first shard ----
        bytes_on_disk = 0
        N_total = 0
        T_global = None
        G_dim = None
        S_dim = None
        first_time_grid = None
        t0 = time.perf_counter()

        for k, f in enumerate(files, 1):
            with np.load(f, mmap_mode=None) as d:
                y_shape = d["y_mat"].shape   # [N_shard, T, S]
                g_shape = d["globals"].shape # [N_shard, G]
                t_shape = d["t_vec"].shape   # [N_shard, T]
                N_shard, T_shard, S_shard = y_shape
                G_shard = g_shape[1]
                if T_global is None:
                    T_global = T_shard
                    S_dim = S_shard
                    G_dim = G_shard
                    # Grid for Δt stats (first trajectory of first shard), float64 for safe subtraction
                    first_time_grid = np.array(d["t_vec"][0], copy=True).astype(np.float64).reshape(-1)
                else:
                    if T_shard != T_global or S_shard != S_dim or G_shard != G_dim or t_shape[1] != T_global:
                        raise ValueError(
                            f"Shard {f} has inconsistent shapes: "
                            f"T={T_shard} vs {T_global}, S={S_shard} vs {S_dim}, G={G_shard} vs {G_dim}"
                        )
                N_total += N_shard
                try:
                    bytes_on_disk += f.stat().st_size
                except Exception:
                    pass

            if (k % max(1, log_every_files)) == 0 or k == len(files):
                self.log.info(
                    f"[{split}] scanned {k}/{len(files)} shards "
                    f"({_fmt_bytes(bytes_on_disk)} so far) "
                    f"in {time.perf_counter() - t0:.1f}s"
                )

        if N_total <= 0 or T_global is None:
            raise RuntimeError(f"[{split}] No data found.")

        # Δt bounds from first grid over allowed horizons
        ms = int(min_steps)
        M = int(max_steps) if max_steps is not None else (T_global - 1)
        M = min(max(ms, 1), T_global - 1)
        if ms < 1 or ms > M:
            raise ValueError(f"Invalid step bounds: min_steps={ms}, max_steps={M}, T={T_global}")

        eps = float(getattr(self.norm, "epsilon", 1e-30))
        dt_min_vec = first_time_grid[ms:] - first_time_grid[:-ms]
        dt_min_vec = dt_min_vec[dt_min_vec > eps]
        dt_max_vec = first_time_grid[M:] - first_time_grid[:-M]
        dt_max_vec = dt_max_vec[dt_max_vec > eps]

        if dt_min_vec.size == 0 or dt_max_vec.size == 0:
            self.dt_log_min = float(self.norm.per_key_stats.get(self.time_var, {}).get("log_min", -3.0))
            self.dt_log_max = float(self.norm.per_key_stats.get(self.time_var, {}).get("log_max", 8.0))
            self.log.warning(
                "[%s] Could not derive Δt stats from first shard; falling back to t_time stats: "
                "log_min=%.6g, log_max=%.6g", split, self.dt_log_min, self.dt_log_max
            )
        else:
            lo = float(np.percentile(np.log10(dt_min_vec), 0.5))
            hi = float(np.percentile(np.log10(dt_max_vec), 99.5))
            if hi <= lo:
                hi = lo + 1.0
            self.dt_log_min, self.dt_log_max = lo, hi
            self.log.warning(
                "[%s] Using Δt stats from FIRST shard grid (assumes ALL shards share this grid): "
                "log10(Δt)_min=%.6g, log10(Δt)_max=%.6g", split, lo, hi
            )
        self.dt_log_denom = (self.dt_log_max - self.dt_log_min) if (self.dt_log_max > self.dt_log_min) else 1.0

        # ---- Pre-allocate final tensors on target device ----
        self.dtype = dtype
        self.N, self.T, self.S = N_total, T_global, S_dim

        # Note: keep t as float32. G and Y use requested dtype.
        self.G = torch.empty((self.N, G_dim), device=self._target_device, dtype=dtype)
        self.t = torch.empty((self.N, self.T), device=self._target_device, dtype=torch.float32)
        self.Y = torch.empty((self.N, self.T, self.S), device=self._target_device, dtype=dtype)

        # Shared physical grid on device; float64 for safe subtraction
        self.time_grid_phys = torch.from_numpy(first_time_grid.astype(np.float64)).to(
            device=self._target_device, non_blocking=True
        )
        self._dt_eps = eps

        # ---- PASS 2: stream shards into GPU tensors ----
        write_ptr = 0
        bytes_on_disk = 0
        t0 = time.perf_counter()

        for k, f in enumerate(files, 1):
            x0_np, g_np, t_np, y_np = _load_npz_keys(f)  # one shard in host RAM
            N_shard = y_np.shape[0]
            sl = slice(write_ptr, write_ptr + N_shard)

            # Globals
            if self.global_vars:
                g_tensor = torch.from_numpy(g_np.astype(np.float32)).to(self._norm_device, non_blocking=True)
                g_norm = self.norm.normalize(g_tensor, self.global_vars)  # [N_shard, G]
                self.G[sl] = g_norm.to(self._target_device, dtype=dtype, non_blocking=True)
                del g_tensor, g_norm
            else:
                self.G[sl].zero_()

            # Time (absolute normalized; Δt is computed per-batch)
            t_tensor = torch.from_numpy(t_np.astype(np.float32)).unsqueeze(-1).to(self._norm_device, non_blocking=True)
            t_norm = self.norm.normalize(t_tensor, [self.time_var]).squeeze(-1)  # [N_shard, T]
            self.t[sl] = t_norm.to(self._target_device, dtype=torch.float32, non_blocking=True)
            del t_tensor, t_norm

            # Targets
            y_tensor = torch.from_numpy(y_np.astype(np.float32)).reshape(-1, self.S).to(self._norm_device, non_blocking=True)
            y_norm = self.norm.normalize(y_tensor, self.target_vars).reshape(N_shard, self.T, self.S)
            self.Y[sl] = y_norm.to(self._target_device, dtype=dtype, non_blocking=True)
            del y_tensor, y_norm

            write_ptr += N_shard
            try:
                bytes_on_disk += f.stat().st_size
            except Exception:
                pass
            if (k % max(1, log_every_files)) == 0 or k == len(files):
                self.log.info(
                    f"[{split}] loaded {k}/{len(files)} shards into GPU "
                    f"({_fmt_bytes(bytes_on_disk)} read) in {time.perf_counter() - t0:.1f}s"
                )

        # ---- GPU memory report ----
        if self._target_device.type == "cuda":
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            self.log.info(
                f"[{split}] staged to GPU: "
                f"G={_fmt_bytes(self.G.numel() * self.G.element_size())}, "
                f"t={_fmt_bytes(self.t.numel() * self.t.element_size())}, "
                f"Y={_fmt_bytes(self.Y.numel() * self.Y.element_size())} | "
                f"GPU mem used {_fmt_bytes(used)} / {_fmt_bytes(total)}"
            )

        # ---- Pair sampling params ----
        self.pairs_per_traj = int(pairs_per_traj)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps) if max_steps is not None else self.T - 1
        if self.min_steps < 1:
            raise ValueError("min_steps must be at least 1")
        if self.max_steps > self.T - 1:
            self.max_steps = self.T - 1
        if self.min_steps > self.max_steps:
            raise ValueError(f"min_steps ({self.min_steps}) > max_steps ({self.max_steps})")

        self._length = int(self.N) * int(self.pairs_per_traj)
        self.log.info(
            f"[{split}] ready (Δt mode): length={self._length:,} "
            f"(pairs_per_traj={self.pairs_per_traj}, "
            f"min_steps={self.min_steps}, max_steps={self.max_steps}) "
            f"| init time={time.perf_counter() - t0_all:.1f}s"
        )

    # ------------------------------- Map-style API ---------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch so (i,j) sampling changes deterministically each epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        return int(idx)

    # ------------------------------ Batched gather ---------------------------

    def _gather_batch(self, idx_list: List[int] | torch.Tensor):
        """
        Vectorized batch fetch for autonomous flow-map (Δt mode ONLY).

        Returns tuple:
          (y_i[B,S], dt_norm[B,1], y_j[B,S], g[B,G], ij[B,2])
        where:
          - y_i, y_j, g are already normalized,
          - dt_norm is Δt normalized ONCE using in-memory dt_log_min/max derived from first shard,
            or falls back to t_time bounds if unavailable,
          - ij = (i, j) integer indices for reference.
        """
        device = self.Y.device
        B = len(idx_list)

        # ----- 1) Compute trajectory index n -----
        if isinstance(idx_list, torch.Tensor):
            idx_cpu = idx_list.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            idx_cpu = np.asarray(idx_list, dtype=np.int64)

        n_cpu = idx_cpu // int(self.pairs_per_traj)

        # ----- 2) Sample arbitrary (i, j>i) with epoch-fresh deterministic RNG -----
        # IMPORTANT: make i depend on idx_cpu so it varies per sample (not per trajectory).
        r_i = _u32_from_array(self.base_seed + 1315423911 * (self.epoch + 1), idx_cpu ^ 0x9E3779B9)
        r_j = _u32_from_array(self.base_seed + 2654435761 * (self.epoch + 1), idx_cpu)

        i_max = max(0, self.T - 1 - self.min_steps)  # i ∈ [0, T-1-min_steps]
        i_cpu = (r_i.astype(np.int64) % (i_max + 1))

        j_lo = i_cpu + self.min_steps  # j ∈ [i+min_steps, min(T-1, i+max_steps)]
        j_hi = np.minimum(self.T - 1, i_cpu + self.max_steps)
        span = np.maximum(j_hi - j_lo + 1, 1)
        j_cpu = j_lo + (r_j.astype(np.int64) % span)

        # ----- 3) Move indices to GPU and gather normalized tensors -----
        n = torch.from_numpy(n_cpu).to(device=device, dtype=torch.long)
        i = torch.from_numpy(i_cpu).to(device=device, dtype=torch.long)
        j = torch.from_numpy(j_cpu).to(device=device, dtype=torch.long)

        y_i = self.Y[n, i]  # [B, S] normalized
        y_j = self.Y[n, j]  # [B, S] normalized
        g   = self.G[n]     # [B, G] normalized

        # ----- 4) Build Δt from the SHARED physical grid, then normalize ONCE -----
        # time_grid_phys: [T] on device (float64); assume shared grid (warned once at init)
        t_i_phys = self.time_grid_phys[i].to(dtype=torch.float64)  # [B]
        t_j_phys = self.time_grid_phys[j].to(dtype=torch.float64)  # [B]
        dt_phys  = torch.clamp(t_j_phys - t_i_phys, min=self._dt_eps)

        # log-min-max using in-memory dt_log_min/max
        dt_log = torch.log10(dt_phys)  # [B], float64
        x = (dt_log - self.dt_log_min) / self.dt_log_denom
        dt_norm = torch.clamp(x, 0.0, 1.0).to(dtype=torch.float32).unsqueeze(-1)  # [B,1]

        ij = torch.stack([i, j], dim=-1)  # [B, 2]

        return y_i, dt_norm, y_j, g, ij


# ---------------------------- DataLoader helper -----------------------------

def create_dataloader(
    dataset: FlowMapPairsDataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Build a DataLoader with optimized settings for GPU-resident data.
    Validation/Test do not shuffle by default.
    """
    target_device = getattr(dataset, "_target_device", torch.device("cpu"))
    on_cuda = isinstance(target_device, torch.device) and target_device.type == "cuda"

    def _collate_indices(batch_indices: List[int]):
        return dataset._gather_batch(batch_indices)

    if on_cuda:
        num_workers = 0
        pin_memory = False
        persistent_workers = False

    shuffle = (getattr(dataset, "split", "train") == "train")

    # Build kwargs dict
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
        collate_fn=_collate_indices,
    )

    # Only add prefetch_factor if we have workers
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)
