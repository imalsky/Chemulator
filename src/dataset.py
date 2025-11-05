#!/usr/bin/env python3
"""
dataset.py — Flow-map DeepONet Dataset (Δt trunk input, dt-spec normalization)

Adds:
- mmap NPZ loading (no compression assumed).
- Fast-path toggles: skip_scan, skip_validate_grids, assume_shared_grid.
- Formalized: precompute_dt_table, share_times_across_batch.
- Option to keep the **entire** dataset on GPU at init (preload_to_gpu).
- Optional "uniform_offset_sampling_strict": sample K offsets uniformly, then
  choose an anchor so all K targets fit (the old behavior you asked for).

Batch contract (matches Trainer):
    y_i : [B, S]          (runtime dtype)
    dt  : [B, K, 1]       (dt-spec normalized, runtime dtype)
    y_j : [B, K, S]       (runtime dtype)
    g   : [B, G]          (runtime dtype)
    aux : {'i':[B], 'j':[B,K]}
    k_mask : [B, K]       (True where j is valid)
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

    __getitem__ returns a lightweight index; actual sampling happens in collate_batch().
    """

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
        self.cfg = dict(config)
        self.pairs_per_traj = int(pairs_per_traj)
        self.min_steps = int(min_steps)
        self.max_steps = None if (max_steps is None) else int(max_steps)
        self._base_seed = int(seed)
        self._epoch = 0
        self._rng = torch.Generator(device="cpu").manual_seed(self._base_seed)
        self._log = logger

        # Dataset config knobs
        dcfg = dict(self.cfg.get("dataset", {}))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", True))
        self.K = int(dcfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dcfg.get("share_times_across_batch", False))

        # Strict anchor-fit mode
        self.uniform_offset_sampling_strict = bool(dcfg.get("uniform_offset_sampling_strict", False))

        # New fast-paths / IO
        self.skip_scan = bool(dcfg.get("skip_scan", False))
        self.skip_validate_grids = bool(dcfg.get("skip_validate_grids", False))
        self.assume_shared_grid = bool(dcfg.get("assume_shared_grid", False))
        self._mmap_mode = str(dcfg.get("mmap_mode", "r")) if dcfg.get("mmap_mode", "r") else None  # None disables memmap

        # Stage device and dtypes
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype  # returned tensors dtype
        self._time_dtype = torch.float32  # internal time dtype for broad backend support

        # Δt clamp floor: dataset override wins; else use normalization epsilon; else tiny
        self.dt_epsilon = float(dcfg.get("dt_epsilon", self.cfg.get("normalization", {}).get("epsilon", 1e-30)))

        # Normalization helper (requires manifest)
        manifest_path = self.root / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Keys/order (hydrated earlier)
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

        # ----------------------------------------------------------------------------------
        # Load shards (mmap) and accumulate
        # ----------------------------------------------------------------------------------
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        shard_paths = sorted(p for p in split_dir.glob("*.npz") if p.is_file())
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {split_dir}")

        y_list: List[torch.Tensor] = []
        g_list: List[torch.Tensor] = []
        t_ref: Optional[torch.Tensor] = None
        t_rows: List[torch.Tensor] = []

        S_expected = None
        G_expected = None
        T_expected = None

        for p in shard_paths:
            g_np, t_np, y_np = _read_npz_triplet(p, mmap_mode=self._mmap_mode)

            # --- basic sanity (unless skip_scan) ---
            if not self.skip_scan:
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
            else:
                # Derive minimal metadata from first shard lazily
                n_i, T_i, S_i = y_np.shape
                G_i = g_np.shape[-1] if g_np is not None else 0
                if S_expected is None:
                    S_expected, G_expected, T_expected = S_i, G_i, T_i

            # Normalize species & globals (float32 inside helper)
            y_t = torch.from_numpy(np.asarray(y_np, dtype=np.float32))  # [n_i, T, S]
            y_t = self.norm.normalize(y_t, self.species_vars)           # normalized float32
            y_list.append(y_t)

            if G_i > 0:
                g_t = torch.from_numpy(np.asarray(g_np, dtype=np.float32))  # [n_i, G]
                g_t = self.norm.normalize(g_t, self.global_vars)            # normalized float32
                g_list.append(g_t)

            # --- Time handling ---
            if self.assume_shared_grid:
                # Force shared grid: take the first time vector we encounter
                if t_ref is None:
                    if t_np.ndim == 1:
                        t_ref = torch.from_numpy(np.asarray(t_np, dtype=np.float32)).reshape(-1)
                    elif t_np.ndim == 2:
                        t_ref = torch.from_numpy(np.asarray(t_np[0], dtype=np.float32)).reshape(-1)
                    else:
                        raise ValueError(f"{p.name}: invalid t_vec dimensionality {t_np.ndim}")
                continue  # do not collect per-row grids in forced-shared mode

            # Normal path (auto-detect shared vs per-row)
            if t_np.ndim == 1:
                t_vec = torch.from_numpy(np.asarray(t_np, dtype=np.float32)).reshape(-1)  # [T]
                if t_ref is None:
                    t_ref = t_vec
                else:
                    if not self.skip_validate_grids:
                        if not torch.allclose(t_vec, t_ref, rtol=0.0, atol=0.0):
                            # Start tracking per-row grids: replicate previous rows with t_ref
                            if not t_rows:
                                prev_rows = sum(x.shape[0] for x in y_list[:-1])
                                t_rows.extend([t_ref.clone() for _ in range(prev_rows)])
                            t_rows.extend([t_vec.clone() for _ in range(n_i)])
                            t_ref = None
            elif t_np.ndim == 2:
                for r in range(n_i):
                    t_rows.append(torch.from_numpy(np.asarray(t_np[r], dtype=np.float32)).reshape(-1))
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
        if self.assume_shared_grid:
            if t_ref is None:
                raise RuntimeError("assume_shared_grid=True but no time vector was found.")
            self._shared_time_grid = True
            self.t_shared = t_ref.to(device=self._stage_device, dtype=self._time_dtype)  # [T]
            self.time_grid_per_row = None
        elif t_ref is not None and len(t_rows) == 0:
            # Shared
            self._shared_time_grid = True
            self.t_shared = t_ref.to(device=self._stage_device, dtype=self._time_dtype)  # [T]
            self.time_grid_per_row = None
        else:
            # Per-row
            self._shared_time_grid = False
            if len(t_rows) == 0:
                if t_ref is None:
                    raise RuntimeError("Internal error: missing t_ref when building per-row time grid")
                tr = t_ref.to(dtype=self._time_dtype)
                self.time_grid_per_row = tr.unsqueeze(0).expand(self.N, -1).contiguous()
            else:
                if len(t_rows) != self.N:
                    # Conservative pad/truncate
                    t_rows = t_rows[:self.N]
                self.time_grid_per_row = torch.stack(t_rows, dim=0).contiguous()  # [N, T]
            # Stage to device if requested
            self.time_grid_per_row = _to_device_dtype(
                self.time_grid_per_row, self._stage_device, self._time_dtype
            )
            self.t_shared = None

        # Max steps (if None, permit full T-1)
        if self.max_steps is None:
            self.max_steps = max(1, self.T - 1)

        # Stage Y/G to device and cast to runtime dtype
        self.y = _to_device_dtype(self.y, self._stage_device, self._runtime_dtype)
        self.g = _to_device_dtype(self.g, self._stage_device, self._runtime_dtype)

        # Optional dt lookup table for shared grid
        self.dt_table: Optional[torch.Tensor] = None
        if self._shared_time_grid and self.precompute_dt_table:
            t = self.t_shared  # [T] float32 on stage device
            T = int(t.shape[0])
            t_i = t.view(T, 1).expand(T, T)
            t_j = t.view(1, T).expand(T, T)

            dt_phys = (t_j - t_i)  # [T, T]
            dt_min_phys = torch.tensor(self.norm.dt_min_phys, dtype=t.dtype, device=t.device)
            lower_or_diag = torch.ones((T, T), dtype=torch.bool, device=t.device).tril()  # j <= i
            dt_phys = torch.where(lower_or_diag, dt_min_phys, dt_phys)
            dt_phys = dt_phys.clamp_min(float(self.dt_epsilon))  # safety

            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)  # [T, T] float32
            self.dt_table = dt_norm.to(dtype=self._time_dtype).contiguous()

        # Derived length
        self.length = self.N * self.pairs_per_traj

    # ---------------------------- PyTorch Dataset API ----------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> _Index:
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
        Sample anchor indices i per row for the non-strict modes.
        """
        upper = max(0, self.T - 1 - self.min_steps)
        if upper <= 0:
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
        Sample offsets o = j - i.

        - If share_offsets_across_batch=True, sample [K] once per batch.
        - Else sample [B,K].
        - If uniform_offset_sampling=True, sample from [min_steps, max_steps] uniformly.
        - Else sample from the same range; downstream masking handles out-of-bounds j.

        All random draws use the CPU generator, then move to the stage device.
        """
        k = int(self.K)
        low = int(self.min_steps)
        high = int(self.max_steps) + 1  # exclusive

        if self.share_offsets_across_batch:
            if getattr(self, "_shared_offsets", None) is None:
                off_cpu = torch.randint(
                    low=low, high=high, size=(k,),
                    generator=self._rng, device="cpu", dtype=torch.long,
                )
                self._shared_offsets = off_cpu.to(self._stage_device, non_blocking=True)
            return self._shared_offsets

        # Per-row offsets
        off_cpu = torch.randint(
            low=low, high=high, size=(B, k),
            generator=self._rng, device="cpu", dtype=torch.long,
        )
        return off_cpu.to(self._stage_device, non_blocking=True)

    # ---------------------------- Collate (Vectorized) ----------------------------

    def collate_batch(self, batch: Sequence[_Index]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Vectorized sampling for a batch of row indices.

        Returns:
            (y_i, dt, y_j, g, aux, k_mask)
        """
        B = len(batch)
        rows = torch.tensor([b.row for b in batch], device=self._stage_device, dtype=torch.long)  # [B]

        # Strict uniform-offset mode (old behavior): choose i_used so all K offsets fit.
        if self.uniform_offset_sampling_strict:
            # Offsets
            offs = self._sample_offsets(B)  # [B,K] or [K]
            if offs.ndim == 1:
                # Shared across batch
                o_max = offs.max()  # scalar
                i_max = (self.T - 1 - o_max).clamp_min(0)
                # Sample i_used per row ∈ [0, i_max]
                i_used_cpu = torch.randint(
                    low=0, high=int(i_max.item()) + 1,
                    size=(B,), generator=self._rng, device="cpu", dtype=torch.long,
                )
                i_used = i_used_cpu.to(self._stage_device, non_blocking=True)
                j_idx = i_used.unsqueeze(1) + offs.unsqueeze(0).expand(B, -1)  # [B,K]
            else:
                # Per-row offsets
                o_max = offs.max(dim=1).values  # [B]
                i_max = (self.T - 1 - o_max).clamp_min(0)  # [B]
                # Each row samples its anchor upper bound
                i_used_cpu = torch.floor(
                    torch.rand((B,), generator=self._rng, device="cpu") * (i_max.to(torch.float32, copy=True).cpu() + 1.0)
                ).to(torch.long)
                i_used = i_used_cpu.to(self._stage_device, non_blocking=True)
                j_idx = i_used.unsqueeze(1) + offs  # [B,K]
            k_mask = torch.ones((B, self.K), dtype=torch.bool, device=self._stage_device)

        else:
            # Non-strict modes: sample anchor first, then offsets; mask invalid j
            i_idx = self._sample_anchor_indices(B)                                    # [B]
            offs = self._sample_offsets(B)                                            # [B,K] or [K]
            if offs.ndim == 1:
                offs = offs.unsqueeze(0).expand(B, -1)
            j_raw = i_idx.unsqueeze(1) + offs                                         # [B, K]
            k_mask = (j_raw < self.T)                                                 # [B, K]
            j_idx = torch.where(k_mask, j_raw, (self.T - 1))                          # [B, K]
            i_used = i_idx                                                            # [B]

        # Gather y_i and y_j
        y_btS = self.y.index_select(0, rows)                                          # [B, T, S]
        y_i = y_btS.gather(dim=1, index=i_used.view(B, 1, 1).expand(-1, 1, self.S)).squeeze(1)  # [B, S]
        y_j = y_btS.gather(dim=1, index=j_idx.view(B, self.K, 1).expand(-1, -1, self.S))        # [B, K, S]

        # Globals
        if self.G > 0:
            g = self.g.index_select(0, rows)                                          # [B, G]
        else:
            g = torch.empty((B, 0), device=self._stage_device, dtype=self._runtime_dtype)

        # Δt normalized (float32 internal, then cast)
        if self.dt_table is not None:
            # tabulated for shared grid
            dt_norm = self.dt_table.index_select(0, i_used).gather(dim=1, index=j_idx)          # [B, K]
        else:
            if self._shared_time_grid:
                t = self.t_shared                                                                   # [T]
                t_i = t.index_select(0, i_used)                                                     # [B]
                t_j = t.index_select(0, j_idx.reshape(-1)).view(B, self.K)                         # [B, K]
            else:
                t_rows = self.time_grid_per_row.index_select(0, rows)                               # [B, T]
                t_i = t_rows.gather(dim=1, index=i_used.view(B, 1)).squeeze(1)                      # [B]
                t_j = t_rows.gather(dim=1, index=j_idx)                                             # [B, K]
            dt_phys = (t_j - t_i.unsqueeze(1)).clamp_min(float(self.dt_epsilon))                    # [B, K]
            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)                                      # [B, K] float32

        dt = dt_norm.to(dtype=self._runtime_dtype).unsqueeze(-1)                                     # [B, K, 1]

        aux = {"i": i_used, "j": j_idx}
        return y_i, dt, y_j, g, aux, k_mask

# ------------------------------ Collate adapter ------------------------------

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
    if kwargs["prefetch_factor"] is None:
        kwargs.pop("prefetch_factor", None)
    return DataLoader(dataset, **kwargs)

# --------------------------------------------------------------------------------------
# NPZ reader
# --------------------------------------------------------------------------------------

def _read_npz_triplet(path: Path, mmap_mode: Optional[str] = "r") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a preprocessed NPZ shard with optional memory mapping.

    Expected keys:
      - 'globals' or 'g_vec' : [N, G]
      - 't_vec'              : [T] or [N, T]
      - 'y_mat' or 'y_arr'   : [N, T, S]
    """
    with np.load(path, mmap_mode=mmap_mode) as npz:
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
