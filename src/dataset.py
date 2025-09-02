#!/usr/bin/env python3
"""
Dataset for autonomous Flow-map DeepONet with multi-time-per-anchor support.

This dataset samples (trajectory n, anchor step i) and then, optionally,
K strictly-later steps j_k > i for the *same* anchor, returning tensors:

    y_i      : [B, S]
    dt_norm  : [B, K]           (normalized Δt = t[j_k] - t[i])
    y_j      : [B, K, S]
    g        : [B, G]
    ij_index : [B, K, 2] int32  (for logging/debugging)

When K=1 it degrades to the classic pairwise interface.

Key features
------------
- Deterministic, stateless GPU/CPU sampling per-epoch (no RNG state carried)
- Supports shards with t_vec saved as [T] *or* [B, T]
- Uses a shared Δt lookup table only when a truly shared time grid is detected
- Preloading to GPU (optional) for high throughput
- Batch-level option to share step offsets across anchors
"""

from __future__ import annotations

import logging
import time
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from normalizer import NormalizationHelper
from utils import load_json


# ------------------------------- Small utils --------------------------------

def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def _load_npz_keys(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (x0, globals, t_vec, y_mat) arrays from a shard.
    Required shapes (flexible for t_vec):
      - x0:   [N, S]
      - g:    [N, G]
      - t:    [T] or [N, T]
      - y:    [N, T, S]
    """
    with np.load(path, allow_pickle=False, mmap_mode="r") as f:
        x0 = f["x0"]
        g = f["globals"]
        t = f["t_vec"]
        y = f["y_mat"]
    return x0, g, t, y


# -------------------------- Stateless hash "random" --------------------------
# CPU-safe 64-bit mixer using only int64 ops. We emulate logical rshift.

def _urshift64(x: torch.Tensor, n: int) -> torch.Tensor:
    return (x >> n) & ((1 << (64 - n)) - 1)

def _mix64_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Murmur-inspired 64-bit mixer using int64 with wraparound semantics.
    Input/Output: torch.int64 (bit pattern only matters).
    """
    # Signed int64 equivalents of 0xff51afd7ed558ccd and 0xc4ceb9fe1a85ec53
    C1 = torch.tensor(-49064778989728563, dtype=torch.int64, device=x.device)
    C2 = torch.tensor(-4265267296055464877, dtype=torch.int64, device=x.device)
    x = x ^ _urshift64(x, 33)
    x = x * C1
    x = x ^ _urshift64(x, 33)
    x = x * C2
    x = x ^ _urshift64(x, 33)
    return x

def _u32_from_array_torch(seed: int, a: torch.Tensor) -> torch.Tensor:
    """
    Deterministic 32-bit hash per element of `a` (any integer-ish tensor).
    Returns an int64 tensor whose values are in [0, 2^32-1].
    Handles arbitrarily large Python ints by folding into signed int64
    two's-complement first (avoids Overflow when creating the tensor).
    """
    # fold seed into 64-bit, then map to signed int64 range
    seed_u64 = seed & 0xFFFFFFFFFFFFFFFF
    if seed_u64 >= (1 << 63):
        seed_i64 = seed_u64 - (1 << 64)
    else:
        seed_i64 = seed_u64

    s = torch.as_tensor(seed_i64, dtype=torch.int64, device=a.device)
    x = a.to(torch.int64) ^ s
    y = _mix64_torch(x)
    return _urshift64(y, 32) & 0xFFFFFFFF


# --------------------------------- Dataset ----------------------------------

class FlowMapPairsDataset(Dataset):
    """
    Samples (n, i) anchors and K strictly-later times j_k for training flow-map DeepONet.

    Output of __getitem__ is just an integer index; DataLoader collates indices and
    calls _gather_batch to build tensors on-device.
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
        super().__init__()
        self.root = Path(processed_root)
        self.split = str(split)
        self.cfg = config
        self.base_seed = int(seed)
        self.log = logging.getLogger("dataset")

        ds_cfg = self.cfg.get("dataset", {})
        self.require_dt_stats: bool = bool(ds_cfg.get("require_dt_stats", True))
        self.precompute_dt_table_cfg: bool = bool(ds_cfg.get("precompute_dt_table", True))

        # Multi-time-per-anchor
        self.multi_time: bool = bool(ds_cfg.get("multi_time_per_anchor", False))
        self.K: int = int(ds_cfg.get("times_per_anchor", 1))
        if not self.multi_time:
            self.K = 1
        self.share_offsets_across_batch: bool = bool(ds_cfg.get("share_times_across_batch", False))

        # Normalization manifest
        norm_path = self.root / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")
        manifest = load_json(norm_path)

        # Staging device / dtype
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self._runtime_dtype = dtype
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Enforce presence of centralized Δt spec when required
        if self.require_dt_stats:
            dt_spec = manifest.get("dt", None)
            if not isinstance(dt_spec, dict) or "log_min" not in dt_spec or "log_max" not in dt_spec:
                raise RuntimeError(
                    "normalization.json must include 'dt' with {'method','log_min','log_max'}; "
                    "re-run preprocessing to write centralized Δt stats."
                )
            if str(dt_spec.get("method", "")).lower() != "log-min-max":
                raise RuntimeError("Currently only 'log-min-max' is supported for Δt.")

        # Discover shards
        pattern = str(self.root / self.split / "*.npz")
        files = sorted(glob(pattern))
        if not files:
            raise RuntimeError(f"[{self.split}] no NPZ shards found under {pattern}")

        # Pass 1: scan shapes and detect if a truly shared time grid exists
        N_total = 0
        T_global: Optional[int] = None
        S_global: Optional[int] = None
        G_global: Optional[int] = None

        bytes_on_disk = 0
        shared_grid_possible = True
        shared_grid_ref: Optional[np.ndarray] = None  # float64 copy for robust comparison
        any_per_row_t = False

        t0_scan = time.perf_counter()
        for k, fpath in enumerate(files, 1):
            f = Path(fpath)
            _, g_np, t_np, y_np = _load_npz_keys(f)

            # Validate shapes and record global dims
            if t_np.ndim == 1:
                T_shard = int(t_np.shape[0])
                if y_np.shape[1] != T_shard:
                    raise RuntimeError(f"[{self.split}] {f.name}: y_mat.shape[1]!=len(t_vec) ({y_np.shape} vs {t_np.shape})")
            elif t_np.ndim == 2:
                any_per_row_t = True
                T_shard = int(t_np.shape[1])
                if y_np.shape[0] != t_np.shape[0] or y_np.shape[1] != T_shard:
                    raise RuntimeError(f"[{self.split}] {f.name}: y_mat and t_vec shapes incompatible ({y_np.shape} vs {t_np.shape})")
            else:
                raise RuntimeError(f"[{self.split}] invalid t_vec ndim in {f.name}: {t_np.shape}")

            N_shard, _, S_shard = int(y_np.shape[0]), int(y_np.shape[1]), int(y_np.shape[2])
            G_shard = int(g_np.shape[1])

            # Detect shared grid: require that every shard has identical grid bytes to the first shard
            # For [B,T], require all rows be identical to row 0 within this shard.
            if shared_grid_possible:
                if t_np.ndim == 1:
                    grid = np.asarray(t_np, dtype=np.float64)
                else:
                    row0 = np.asarray(t_np[0], dtype=np.float64)
                    if not np.all(t_np == t_np[0]):  # fast exact check; dtype preserved in file
                        shared_grid_possible = False
                        grid = None
                    else:
                        grid = row0
                if grid is not None:
                    if shared_grid_ref is None:
                        shared_grid_ref = grid.copy()
                    else:
                        if not np.array_equal(shared_grid_ref, grid):
                            shared_grid_possible = False

            N_total += N_shard
            if T_global is None:
                T_global, S_global, G_global = T_shard, S_shard, G_shard
            else:
                if T_global != T_shard or S_global != S_shard or G_global != G_shard:
                    raise RuntimeError(f"[{self.split}] heterogeneous shard dims detected; T/S/G must match across shards.")

            try:
                bytes_on_disk += f.stat().st_size
            except Exception:
                pass

            if (k % max(1, log_every_files)) == 0 or k == len(files):
                self.log.info(f"[{self.split}] scanned {k}/{len(files)} shards "
                              f"({_fmt_bytes(bytes_on_disk)} so far) in {time.perf_counter() - t0_scan:.1f}s")

        if N_total <= 0 or T_global is None or S_global is None or G_global is None:
            raise RuntimeError(f"[{self.split}] No data found or invalid shard shapes.")

        # Step bounds (STRICT)
        ms = int(min_steps)
        M = int(max_steps) if max_steps is not None else (T_global - 1)
        if ms < 1 or M < ms or M > T_global - 1:
            raise ValueError(f"Invalid step bounds: min_steps={ms}, max_steps={M}, T={T_global}")

        self.N = int(N_total)
        self.T = int(T_global)
        self.S = int(S_global)
        self.G_dim = int(G_global)
        self.min_steps = ms
        self.max_steps = M
        self.pairs_per_traj = int(pairs_per_traj)

        # Allocate staging buffers (host or device depending on preload_to_gpu)
        self.G = torch.empty((self.N, self.G_dim), device=self._stage_device, dtype=torch.float32)
        self.Y = torch.empty((self.N, self.T, self.S), device=self._stage_device, dtype=self._runtime_dtype)

        # Time grid storage
        self.shared_time_grid: Optional[torch.Tensor] = None
        self.time_grid_per_row: Optional[torch.Tensor] = None
        self.has_shared_grid = bool(shared_grid_possible and shared_grid_ref is not None)

        if self.has_shared_grid:
            self.shared_time_grid = torch.from_numpy(shared_grid_ref.astype(np.float64)).to(self._stage_device)
        else:
            # Per-row storage [N, T] float64
            self.time_grid_per_row = torch.empty((self.N, self.T), device=self._stage_device, dtype=torch.float64)

        # Pass 2: stream shards, normalize on-the-fly, and stage
        write_ptr = 0
        t0_load = time.perf_counter()
        for k, fpath in enumerate(files, 1):
            f = Path(fpath)
            _, g_np, t_np, y_np = _load_npz_keys(f)
            N_shard = int(y_np.shape[0])
            sl = slice(write_ptr, write_ptr + N_shard)

            # Globals → float32
            g_tensor = torch.from_numpy(g_np.astype(np.float32, copy=False))
            self.G[sl] = g_tensor.to(self._stage_device, dtype=torch.float32, non_blocking=True)

            # Species → normalize columns per variables
            y_tensor = torch.from_numpy(y_np)  # storage dtype
            y_norm = self.norm.normalize(
                y_tensor, self.cfg["data"].get("target_species_variables", self.cfg["data"]["species_variables"])
            ).reshape(N_shard, self.T, self.S)
            self.Y[sl] = y_norm.to(self._stage_device, dtype=self._runtime_dtype, non_blocking=True)

            # Time grids
            if not self.has_shared_grid:
                if t_np.ndim == 1:
                    # broadcast to all rows in this shard
                    self.time_grid_per_row[sl] = torch.from_numpy(t_np.astype(np.float64, copy=False)).to(
                        self._stage_device
                    ).view(1, self.T).expand(N_shard, self.T)
                else:
                    self.time_grid_per_row[sl] = torch.from_numpy(t_np.astype(np.float64, copy=False)).to(
                        self._stage_device
                    )

            write_ptr += N_shard

            if (k % max(1, log_every_files)) == 0 or k == len(files):
                self.log.info(
                    f"[{self.split}] loaded {k}/{len(files)} shards into "
                    f"{'GPU' if self._stage_device.type == 'cuda' else 'CPU'} "
                    f"in {time.perf_counter() - t0_load:.1f}s"
                )

        # Flattened view for fast (n, t) gathers
        self.Y_flat = self.Y.reshape(self.N * self.T, self.S)

        # Optional Δt normalization table (float32, on staging device) only if shared grid
        self.dt_table: Optional[torch.Tensor] = None
        if self.has_shared_grid and self.precompute_dt_table_cfg:
            self.dt_table = self.norm.make_dt_norm_table(
                time_grid_phys=self.shared_time_grid, min_steps=self.min_steps, max_steps=self.max_steps
            ).to(device=self._stage_device, dtype=torch.float32)
        elif self.precompute_dt_table_cfg and not self.has_shared_grid:
            self.log.info(f"[{self.split}] multiple time grids detected; disabling precompute_dt_table.")

        # Dataset length is anchors-per-traj × N
        self._length = int(self.N * self.pairs_per_traj)

        # Epoch counter used for deterministic sampling
        self.epoch = 0

        # Memory report
        if self._stage_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            self.log.info(
                f"[{self.split}] staged: G={_fmt_bytes(self.G.numel()*self.G.element_size())}, "
                f"Y={_fmt_bytes(self.Y.numel()*self.Y.element_size())} | "
                f"GPU mem used {used / (1024 ** 3):.1f} GiB / {total / (1024 ** 3):.1f} GiB"
            )

        self.log.info(
            f"[{self.split}] ready: N={self.N}, T={self.T}, S={self.S}, G={self.G_dim} | "
            f"pairs_per_traj={self.pairs_per_traj}, min_steps={self.min_steps}, max_steps={self.max_steps} | "
            f"multi_time={self.multi_time}, K={self.K}, share_offsets_across_batch={self.share_offsets_across_batch}"
        )

    # ----------------------------- Epoch control -----------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch so stateless sampling changes deterministically."""
        self.epoch = int(epoch)

    # ------------------------------ Iteration API ----------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> int:
        # DataLoader will batch integer indices and we build tensors in _gather_batch
        return int(idx)

    # ------------------------------ Batched gather ---------------------------

    def _gather_batch(self, idx_list):
        """
        Construct a batch on the dataset's staging device.
        Returns (y_i, dt_norm, y_j, g, ij) with shapes in the module docstring.

        Variable-K handling near the end:
        - We take all available later steps (span = j_hi-j_lo+1).
        - If span < K, we pad with j=i (trainer can mask with (j>i)).
        """
        dev = self.Y.device
        B = len(idx_list)
        K = self.K

        # Indices to device
        if isinstance(idx_list, torch.Tensor):
            idx = idx_list.to(device=dev, dtype=torch.int64, non_blocking=True)
        else:
            idx = torch.as_tensor(idx_list, device=dev, dtype=torch.int64)

        # Decode sample index → trajectory id
        n = torch.div(idx, self.pairs_per_traj, rounding_mode='floor')  # [B]

        # Stateless RNG for anchors (i)
        seed_i = self.base_seed + 1315423911 * (self.epoch + 1)
        r_i_u32 = _u32_from_array_torch(seed_i, idx)               # uint32
        i_max = max(0, self.T - 1 - self.min_steps)
        i = (r_i_u32.remainder(i_max + 1)).to(torch.int64)         # [B]

        # Bounds for j per anchor
        j_lo = i + self.min_steps                                  # [B]
        j_hi = torch.minimum(torch.full_like(i, self.T - 1),
                             i + self.max_steps)                    # [B]
        span = torch.clamp(j_hi - j_lo + 1, min=0)                 # [B] (#valid later steps)

        # Build j indices with "all possible" when span<K, otherwise K without-replacement
        j_out = torch.empty((B, K), dtype=torch.int64, device=dev)
        j_out[:] = i.view(B, 1)  # pad with j=i by default (trainer can mask)

        if K == 1:
            # Fast path: choose one j uniformly in [j_lo, j_hi]
            seed_j = self.base_seed ^ 0x9e3779b97f4a7c15 ^ (self.epoch + 7)
            r_j = _u32_from_array_torch(seed_j, idx)
            # guard span==0 case → keep j=i (already padded)
            choose = (span > 0)
            # compute offsets safely where span>0
            off = torch.zeros_like(i)
            if torch.any(choose):
                off[choose] = (r_j[choose].remainder(span[choose])).to(torch.int64)
            j = j_lo + off
            j_out[:, 0] = torch.where(choose, j, i)
        else:
            if self.share_offsets_across_batch:
                # Shared random permutation over [0, span_max)
                span_max = int(torch.max(span).item()) if B > 0 else 0
                if span_max > 0:
                    seed_off = self.base_seed ^ 0xD2B74407B1CE6E93 ^ (self.epoch + 101)
                    offs = torch.arange(span_max, device=dev, dtype=torch.int64)
                    ranks = _u32_from_array_torch(seed_off, offs)  # [span_max]
                    perm = torch.argsort(ranks)                    # ascending
                    shared = offs[perm]                            # permutation
                    for b in range(B):
                        sb = int(span[b].item())
                        used_k = min(K, sb)
                        if used_k > 0:
                            j_sel = j_lo[b].item() + shared[:used_k]
                            j_out[b, :used_k] = j_sel
            else:
                # Independent per-row permutations limited to span[b]
                seed_off = self.base_seed ^ 0xC3A5C85C97CB3127 ^ (self.epoch + 19)
                for b in range(B):
                    sb = int(span[b].item())
                    if sb <= 0:
                        continue  # keep padding (j=i)
                    used_k = min(K, sb)
                    offs = torch.arange(sb, device=dev, dtype=torch.int64)
                    # Mix idx[b] into the ranks to differ per row deterministically
                    ranks = _u32_from_array_torch(seed_off ^ int(idx[b].item()), offs)
                    perm = torch.argsort(ranks)
                    sel = offs[perm[:used_k]]
                    j_out[b, :used_k] = int(j_lo[b].item()) + sel

        # Gather tensors
        # y_i: [B,S]
        lin_i = n * self.T + i
        y_i = self.Y_flat.index_select(0, lin_i)  # [B,S]

        # y_j: [B,K,S]
        lin_j = n.view(B, 1) * self.T + j_out    # [B,K]
        y_j = self.Y_flat.index_select(0, lin_j.reshape(-1)).reshape(B, K, self.S)

        # g: [B,G]
        g = self.G.index_select(0, n)            # [B,G]

        # Δt normalization
        # Mask for padded positions (j==i)
        pad_mask = (j_out == i.view(B, 1))

        if self.dt_table is not None:
            # Shared grid table
            dt_norm = self.dt_table[i.view(B, 1), j_out].to(dtype=self._runtime_dtype)  # [B,K]
            # Force padded to 0 so trainers can optionally ignore even if they forget to mask
            dt_norm = torch.where(pad_mask, torch.zeros_like(dt_norm), dt_norm)
        else:
            # Compute from physical times per-row or shared vector
            if self.time_grid_per_row is not None:
                # Per-row grids: advanced indexing
                t_i = self.time_grid_per_row[n, i]                                  # [B]
                t_j = self.time_grid_per_row[n.view(B, 1).expand(-1, K), j_out]     # [B,K]
            else:
                # Shared vector
                vec = self.shared_time_grid  # [T]
                t_i = vec.index_select(0, i)                                       # [B]
                t_j = vec.index_select(0, j_out.reshape(-1)).view(B, K)            # [B,K]
            dt_phys = t_j - t_i.view(B, 1)
            eps = float(self.norm.epsilon)
            dt_phys = torch.where(dt_phys > 0, dt_phys,
                                  torch.as_tensor(eps, dtype=dt_phys.dtype, device=dt_phys.device))
            dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(dtype=self._runtime_dtype)
            # Zero out padded for niceness
            dt_norm = torch.where(pad_mask, torch.zeros_like(dt_norm), dt_norm)

        # ij indices (int32): [B,K,2]
        ij = torch.stack((i.view(B, 1).expand(B, K).to(torch.int32),
                          j_out.to(torch.int32)), dim=-1)

        return y_i, dt_norm, y_j, g, ij


# ------------------------------- Dataloader ----------------------------------

def create_dataloader(
    dataset: FlowMapPairsDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for this dataset.
    The dataset performs its own stateless sampling; DataLoader shuffling is redundant.
    """
    # If dataset tensors live on GPU, keep workers=0 and pin_memory=False
    on_cuda = (dataset.Y.device.type == "cuda")
    if on_cuda:
        num_workers = 0
        pin_memory = False
        shuffle = False

    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        drop_last=False,
        collate_fn=lambda idxs: dataset._gather_batch(idxs),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return torch.utils.data.DataLoader(**kwargs)
