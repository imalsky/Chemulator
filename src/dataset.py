#!/usr/bin/env python3
"""
Flow-map DeepONet Dataset (Δt trunk input, dt-spec normalization)

Vectorized dataset for flow-map prediction with random anchors and multiple
future targets j > i.

Key points:
- Trunk input is Δt = t_j - t_i in physical units, normalized via dt-spec (log-min-max).
- Outputs (y_i, y_j, g, dt) are returned in the **runtime dtype** selected from config
  (see utils.resolve_precision_and_dtype). This is passed in at construction time.
- Internal time grids (t) are always kept in float32 to ensure broad backend support
  (notably MPS does not support float64).
- Supports shared time grid (all rows share one canonical t) or per-row time grids.

Returned batch (matches Trainer expectations):
    y_i : [B, S]          (normalized species at anchor, runtime dtype)
    dt  : [B, K, 1]       (normalized Δt, runtime dtype)
    y_j : [B, K, S]       (normalized species at target, runtime dtype)
    g   : [B, G]          (normalized globals, runtime dtype)
    aux : {'i':[B], 'j':[B,K]}
    k_mask : [B, K]       (boolean; True = valid target; False = clamped-to-last)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _as_path(p: Any) -> Path:
    return Path(p) if isinstance(p, Path) else Path(p)


def _to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.device != device or x.dtype != dtype:
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return x


def _canonical_split(name: str) -> str:
    n = str(name).strip().lower()
    if n in ("train", "training"):
        return "train"
    if n in ("val", "valid", "validation"):
        return "validation"
    if n in ("test", "testing"):
        return "test"
    raise ValueError(f"Unknown split: {name}")


@dataclass
class _Index:
    row: int


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class FlowMapPairsDataset(Dataset):
    """
    Flow-map dataset with K future targets per anchor.

    Contract:
      __getitem__ returns a lightweight index; heavy sampling happens in collate().

    Construction parameters:
      processed_root : directory with preprocessed artifacts
      split          : 'train' | 'validation' | 'test'
      config         : full (hydrated) config dict
      pairs_per_traj : anchors per trajectory per epoch
      min_steps      : minimum offset (j - i) >= 1
      max_steps      : maximum offset (None = T-1)
      preload_to_gpu : whether to stage tensors to device at init
      device         : stage device if preload_to_gpu=True else ignored (CPU used)
      dtype          : **runtime** dtype for outputs (e.g., torch.bfloat16 / torch.float16 / torch.float32)
      seed           : base seed; epoch seeding handled via set_epoch(epoch)
      logger         : optional logger
    """

    # ------------------------------- Init ------------------------------------

    def __init__(
            self,
            processed_root: Path | str,
            split: str,
            config: Dict[str, Any],
            pairs_per_traj: int,
            min_steps: int,
            max_steps: Optional[int],
            preload_to_gpu: bool,
            device: Optional[torch.device],
            dtype: torch.dtype,
            seed: int = 42,
            logger: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.root = _as_path(processed_root)
        self.split = _canonical_split(split)
        self.cfg = config
        self.logger = logger

        # Sampling controls
        self.pairs_per_traj = int(max(1, pairs_per_traj))
        self.min_steps = max(1, int(min_steps))
        self.max_steps = int(max_steps) if (max_steps is not None) else None

        dcfg = dict(self.cfg.get("dataset", {}))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", True))
        self.K = int(dcfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dcfg.get("share_times_across_batch", False))
        self.uniform_offset_sampling = bool(dcfg.get("uniform_offset_sampling", False))

        # Stage device and dtypes
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype  # driven by config/mixed precision
        self._time_dtype = torch.float32  # robust across backends

        # Δt clamp floor: dataset override wins; else use normalization epsilon; else tiny
        self.dt_epsilon = float(dcfg.get("dt_epsilon", self.cfg.get("normalization", {}).get("epsilon", 1e-30)))

        # Normalization helper (requires manifest)
        manifest_path = self.root / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Keys/order (hydrated in main.py)
        dcfg_data = dict(self.cfg.get("data", {}))
        self.species_vars: List[str] = list(dcfg_data.get("species_variables", []))
        self.target_species: List[str] = list(dcfg_data.get("target_species", [])) or list(self.species_vars)
        self.global_vars: List[str] = list(dcfg_data.get("global_variables", []))
        self.time_key: str = str(dcfg_data.get("time_variable", "t_time"))

        if not self.species_vars:
            raise RuntimeError("cfg.data.species_variables is empty (ensure hydration from artifacts)")
        if any(k not in manifest.get("normalization_methods", {}) for k in
               (self.species_vars + self.global_vars + [self.time_key])):
            raise RuntimeError("Normalization methods missing for one or more variables in normalization.json")

        # RNG & epoch state
        self._base_seed = int(self.cfg.get("system", {}).get("seed", seed))
        self._epoch = 0
        self._rng = torch.Generator(device="cpu")
        self._shared_offsets: Optional[torch.Tensor] = None

        # Load shards
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        shard_paths = sorted(p for p in split_dir.glob("*.npz") if p.is_file())
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {split_dir}")

        # Accumulate across shards
        y_list: List[torch.Tensor] = []
        g_list: List[torch.Tensor] = []
        t_ref: Optional[torch.Tensor] = None
        t_rows: List[torch.Tensor] = []

        S_expected = None
        G_expected = None
        T_expected = None

        for p in shard_paths:
            g_np, t_np, y_np = _read_npz_triplet(p)

            # Validate shapes
            if y_np.ndim != 3:
                raise ValueError(f"{p.name}: expected y_mat with shape [N, T, S], got {y_np.shape}")
            n_i, T_i, S_i = y_np.shape
            G_i = g_np.shape[-1] if g_np is not None else 0

            if S_expected is None:
                S_expected, G_expected, T_expected = S_i, G_i, T_i
            else:
                if (S_i != S_expected) or (G_i != G_expected) or (T_i != T_expected):
                    raise ValueError(
                        f"{p.name}: inconsistent shapes across shards; "
                        f"got (T={T_i}, S={S_i}, G={G_i}) expected "
                        f"(T={T_expected}, S={S_expected}, G={G_expected})"
                    )

            # Normalize species & globals
            y_t = torch.from_numpy(y_np.astype(np.float32, copy=False))  # [n_i, T, S] float32
            y_t = self.norm.normalize(y_t, self.species_vars)  # normalized float32
            y_list.append(y_t)

            if G_i > 0:
                g_t = torch.from_numpy(g_np.astype(np.float32, copy=False))  # [n_i, G] float32
                g_t = self.norm.normalize(g_t, self.global_vars)  # normalized float32
                g_list.append(g_t)

            # Time: support shared 1-D or per-row 2-D
            if t_np.ndim == 1:
                t_vec = torch.from_numpy(t_np.astype(np.float32, copy=False)).reshape(-1)  # [T]
                if t_ref is None:
                    t_ref = t_vec
                else:
                    if not torch.allclose(t_vec, t_ref, rtol=0.0, atol=0.0):
                        # retroactively expand prior rows to per-row using the previous shared t_ref
                        if not t_rows:
                            if t_ref is None:
                                raise RuntimeError("Internal error: t_ref missing when building per-row grids")
                            prev_rows = sum(x.shape[0] for x in y_list[:-1])  # rows from all but current shard
                            t_rows.extend([t_ref.clone() for _ in range(prev_rows)])
                        # Now add current shard rows with its own vector
                        t_rows.extend([t_vec.clone() for _ in range(n_i)])
                        t_ref = None
            elif t_np.ndim == 2:
                if t_rows is None:
                    t_rows = []
                for r in range(n_i):
                    t_rows.append(torch.from_numpy(t_np[r].astype(np.float32, copy=False)).reshape(-1))
                t_ref = None
            else:
                raise ValueError(f"{p.name}: invalid t_vec dimensionality {t_np.ndim}")

        # Stack species/globals
        self.y = torch.cat(y_list, dim=0).contiguous()  # [N, T, S] float32
        if g_list:
            self.g = torch.cat(g_list, dim=0).contiguous()  # [N, G] float32
        else:
            self.g = torch.empty((self.y.shape[0], 0), device=self._stage_device, dtype=self._runtime_dtype)

        # Shapes
        self.N, self.T, self.S = self.y.shape
        self.G = self.g.shape[-1] if self.g.numel() > 0 else 0

        # Time grids: shared vs per-row
        if t_ref is not None and len(t_rows) == 0:
            # Shared
            self._shared_time_grid = True
            self.t_shared = t_ref.to(device=self._stage_device, dtype=self._time_dtype)  # [T]
            self.time_grid_per_row = None
        else:
            # Per-row
            self._shared_time_grid = False
            if len(t_rows) == 0:
                # Prior shards shared; replicate shared for all rows
                if t_ref is None:
                    raise RuntimeError("Internal error: missing t_ref when building per-row time grid")
                tr = t_ref.to(dtype=self._time_dtype)
                self.time_grid_per_row = tr.unsqueeze(0).expand(self.N, -1).contiguous()
            else:
                if len(t_rows) != self.N:
                    raise RuntimeError("Internal error: per-row time list length mismatch")
                self.time_grid_per_row = torch.stack([t.to(dtype=self._time_dtype) for t in t_rows], dim=0)

            self.time_grid_per_row = _to_device_dtype(self.time_grid_per_row, self._stage_device, self._time_dtype)
            self.t_shared = None

        # Stage Y/G to device and cast to runtime dtype
        self.y = _to_device_dtype(self.y, self._stage_device, self._runtime_dtype)
        self.g = _to_device_dtype(self.g, self._stage_device, self._runtime_dtype)

        # Optional dt lookup table for shared grid
        self.dt_table: Optional[torch.Tensor] = None
        if self._shared_time_grid and self.precompute_dt_table:
            # Build an upper-triangular dt table (future-only); mask j <= i to dt_min_phys
            t = self.t_shared  # [T] float32 on stage device
            T = int(t.shape[0])
            t_i = t.view(T, 1).expand(T, T)
            t_j = t.view(1, T).expand(T, T)

            dt_phys = (t_j - t_i)  # [T, T]
            # mask lower triangle + diagonal to dt_min_phys so normalizer never sees invalid pairs
            dt_min_phys = torch.tensor(self.norm.dt_min_phys, dtype=t.dtype, device=t.device)
            lower_or_diag = torch.ones((T, T), dtype=torch.bool, device=t.device).tril()  # j <= i
            dt_phys = torch.where(lower_or_diag, dt_min_phys, dt_phys)
            dt_phys = dt_phys.clamp_min(float(self.dt_epsilon))  # final safety

            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)  # [T, T] float32
            # Keep lookup in float32 for broad backend compatibility
            self.dt_table = dt_norm.to(dtype=self._time_dtype).contiguous()

        # Derived
        self.length = self.N * self.pairs_per_traj

    # ---------------------------- PyTorch Dataset API ----------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> _Index:
        # Map to row id; anchor sampling happens in collate for vectorization
        row = int(idx % self.N)
        return _Index(row=row)

    # ---------------------------- Sampling Utilities ----------------------------

    def set_epoch(self, epoch: int) -> None:
        """To be called by the Trainer at each epoch boundary."""
        self._epoch = int(epoch)
        self._reset_rng()
        self._shared_offsets = None  # re-sample shared offsets next collate

    def _reset_rng(self) -> None:
        seed_everything(self._base_seed + self._epoch, workers=False)
        self._rng.manual_seed(self._base_seed + self._epoch)

    def _sample_anchor_indices(self, B: int) -> torch.Tensor:
        """
        Sample anchor indices i per row.

        Draw with the CPU generator and move to the stage device to avoid
        device-mismatch errors on MPS.
        """
        upper = max(0, self.T - 1 - self.min_steps)
        if upper <= 0:
            # Degenerate case; rely on downstream masking
            return torch.zeros(B, dtype=torch.long, device=self._stage_device)
        idx_cpu = torch.randint(
            low=0,
            high=upper + 1,  # randint high is exclusive
            size=(B,),
            generator=self._rng,
            device="cpu",
            dtype=torch.long,
        )
        return idx_cpu.to(self._stage_device, non_blocking=True)

    def _sample_offsets(self, B: int) -> torch.Tensor:
        """
        Sample j - i offsets.

        All random draws use the CPU generator, then are moved to the stage device.
        Shared-offsets are cached on the stage device until the next epoch.
        """
        k = int(self.K)
        max_global = (self.T - 1) if (self.max_steps is None) else min(self.max_steps, self.T - 1)
        low = int(self.min_steps)
        high = int(max(low + 1, max_global + 1))  # randint high is exclusive

        if self.share_offsets_across_batch:
            if self._shared_offsets is None:
                off_cpu = torch.randint(
                    low=low, high=high, size=(k,),
                    generator=self._rng, device="cpu", dtype=torch.long,
                )
                self._shared_offsets = off_cpu.to(self._stage_device, non_blocking=True)
            return self._shared_offsets

        if self.uniform_offset_sampling:
            off_cpu = torch.randint(
                low=low, high=high, size=(B, k),
                generator=self._rng, device="cpu", dtype=torch.long,
            )
            return off_cpu.to(self._stage_device, non_blocking=True)
        else:
            # Same global-range sampling; downstream masking handles out-of-bounds j
            off_cpu = torch.randint(
                low=low, high=high, size=(B, k),
                generator=self._rng, device="cpu", dtype=torch.long,
            )
            return off_cpu.to(self._stage_device, non_blocking=True)

    # ---------------------------- Collate (vectorized) ----------------------------

    def collate_batch(self, batch: Sequence[_Index]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Vectorized sampling for a batch of row indices.

        Returns:
            (y_i, dt, y_j, g, aux, k_mask) as documented at top.
        """
        B = len(batch)
        rows = torch.tensor([b.row for b in batch], device=self._stage_device, dtype=torch.long)  # [B]

        # Sample anchors i and offsets off per row
        i_idx = self._sample_anchor_indices(B)                                    # [B]
        offs = self._sample_offsets(B)                                            # [B,K] or [K]
        if offs.ndim == 1:  # share across batch
            offs = offs.unsqueeze(0).expand(B, -1)
        # Targets and validity mask
        j_raw = i_idx.unsqueeze(1) + offs                                         # [B, K]
        j_valid = j_raw < self.T                                                  # [B, K] bool
        j_idx = torch.where(j_valid, j_raw, (self.T - 1))                         # [B, K]

        # Gather y_i and y_j
        y_btS = self.y.index_select(0, rows)                                      # [B, T, S] runtime dtype
        y_i = y_btS.gather(dim=1, index=i_idx.view(B, 1, 1).expand(-1, 1, self.S)).squeeze(1)  # [B, S]
        y_j = y_btS.gather(dim=1, index=j_idx.view(B, self.K, 1).expand(-1, -1, self.S))       # [B, K, S]

        # Globals
        if self.G > 0:
            g = self.g.index_select(0, rows)                                      # [B, G]
        else:
            g = torch.empty((B, 0), device=self._stage_device, dtype=self._runtime_dtype)

        # Δt normalized
        if self.dt_table is not None:
            # tabulated for shared grid
            dt_norm = self.dt_table.index_select(0, i_idx).gather(dim=1, index=j_idx)          # [B, K]
        else:
            # compute from time grids (float32 path)
            if self._shared_time_grid:
                t = self.t_shared                                                                 # [T] float32
                t_i = t.index_select(0, i_idx)                                                    # [B]
                t_j = t.index_select(0, j_idx.reshape(-1)).view(B, self.K)                        # [B, K]
            else:
                t_rows = self.time_grid_per_row.index_select(0, rows)                             # [B, T] float32
                t_i = t_rows.gather(1, i_idx.view(B, 1)).squeeze(1)                               # [B]
                t_j = t_rows.gather(1, j_idx)                                                     # [B, K]

            # Fill invalid pairs with dt_min_phys before normalization to avoid range warnings
            dt_phys = (t_j - t_i.unsqueeze(1))                                                    # [B, K] float32
            dt_min_phys = torch.tensor(self.norm.dt_min_phys, dtype=dt_phys.dtype, device=dt_phys.device)
            dt_phys = torch.where(j_valid, dt_phys, dt_min_phys)                                  # fill invalids
            dt_phys = dt_phys.clamp_min(float(self.dt_epsilon))                                   # final safety

            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)                                   # [B, K] float32

        # Cast dt to runtime dtype and add trailing singleton dim
        dt = dt_norm.to(dtype=self._runtime_dtype).unsqueeze(-1)                                   # [B, K, 1]

        aux = {"i": i_idx, "j": j_idx}
        k_mask = j_valid

        return y_i, dt, y_j, g, aux, k_mask


# --------------------------------------------------------------------------------------
# DataLoader helper
# --------------------------------------------------------------------------------------

def collate_flowmap(batch: Sequence[_Index], dataset: FlowMapPairsDataset):
    return dataset.collate_batch(batch)


def create_dataloader(
    dataset: FlowMapPairsDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        pin_memory=bool(pin_memory),
        prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
        drop_last=False,
        collate_fn=lambda b: collate_flowmap(b, dataset),
    )
    # Torch DataLoader disallows prefetch_factor when num_workers=0
    if kwargs["prefetch_factor"] is None:
        kwargs.pop("prefetch_factor", None)
    return DataLoader(dataset, **kwargs)


# --------------------------------------------------------------------------------------
# NPZ reader
# --------------------------------------------------------------------------------------

def _read_npz_triplet(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a preprocessed NPZ shard.

    Expected keys:
      - 'globals' or 'g_vec' : [N, G]
      - 't_vec'              : [T] or [N, T]
      - 'y_mat' or 'y_arr'   : [N, T, S]
    """
    with np.load(path) as npz:
        if "globals" in npz:
            g_vec = npz["globals"]
        elif "g_vec" in npz:
            g_vec = npz["g_vec"]
        else:
            g_vec = None

        if "t_vec" not in npz:
            raise KeyError(f"{path.name}: missing 't_vec'")
        t_vec = npz["t_vec"]

        if "y_mat" in npz:
            y_mat = npz["y_mat"]
        elif "y_arr" in npz:
            y_mat = npz["y_arr"]
        else:
            raise KeyError(f"{path.name}: missing species array ('y_mat' or 'y_arr')")

    return g_vec, t_vec, y_mat
