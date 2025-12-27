#!/usr/bin/env python3
"""
Batch variables and structures:
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
from torch.utils.data import Dataset, DataLoader

from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper


def _as_path(p: Any) -> Path:
    return Path(p) if isinstance(p, Path) else Path(p)


def _to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.device == device and x.dtype == dtype:
        return x
    return x.to(device=device, dtype=dtype, non_blocking=(device.type == "cuda"))


def _canonical_split(split: str) -> str:
    name = str(split).strip().lower()
    if name in ("train", "training"):
        return "train"
    if name in ("val", "valid", "validation"):
        return "validation"  # keep in sync with preprocessor output
    if name in ("test", "testing"):
        return "test"
    raise ValueError(f"Unknown split: {split!r}")


@dataclass
class _Index:
    row: int  # global trajectory index in [0, N)
    is_first: bool  # first of the pairs_per_traj samples for that row


class FlowMapPairsDataset(Dataset):
    """
    High-performance in-memory dataset with K future targets per anchor.

    Design:
      * All normalized trajectories (y, g, time) for a split are preloaded
        once into a single set of contiguous tensors on the staging device
        (CPU by default; optionally GPU if preload_to_gpu=True).
      * __getitem__ returns lightweight indices, and collate_batch performs
        fully vectorized gathers over the contiguous tensors.
      * Randomness comes from anchor/offset sampling, not from DataLoader
        shuffle, so DataLoader can use sequential sampling without becoming
        an I/O bottleneck.
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
        self._log = logger

        # One-time logging flags
        self._did_log_init_summary: bool = False
        self._did_log_first_batch_checks: bool = False

        # Dataset config knobs
        dcfg = dict(self.cfg.get("dataset", {}))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", True))
        self.K = int(dcfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dcfg.get("share_times_across_batch", False))

        # Enforce i=0 for the first of the pairs_per_traj samples (training only)
        self.use_first_anchor = bool(dcfg.get("use_first_anchor", False))

        # Time-grid / IO options
        self.skip_scan = bool(dcfg.get("skip_scan", False))
        self.skip_validate_grids = bool(dcfg.get("skip_validate_grids", False))
        self.assume_shared_grid = bool(dcfg.get("assume_shared_grid", False))
        self._mmap_mode = (
            str(dcfg.get("mmap_mode", "r")) if dcfg.get("mmap_mode", "r") else None
        )

        # Stage device and dtypes
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype
        self._time_dtype = torch.float32

        # Δt clamp floor
        self.dt_epsilon = float(
            dcfg.get("dt_epsilon", self.cfg.get("normalization", {}).get("epsilon", 1e-30))
        )

        # Normalization helper (requires manifest)
        manifest_path = self.root / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Keys/order
        dcfg_data = dict(self.cfg.get("data", {}))
        self.species_vars: List[str] = list(dcfg_data.get("species_variables", []))
        self.target_species: List[str] = list(dcfg_data.get("target_species", [])) or list(
            self.species_vars
        )
        self.global_vars: List[str] = list(dcfg_data.get("global_variables", []))
        self.time_key: str = str(dcfg_data.get("time_variable", "t_time"))

        if not self.species_vars:
            raise RuntimeError("cfg.data.species_variables is empty (ensure hydration from artifacts)")

        norm_methods = manifest.get("normalization_methods", {})
        for key in (self.species_vars + self.global_vars + [self.time_key]):
            if key not in norm_methods:
                raise RuntimeError(
                    f"Normalization method missing for variable {key!r} in normalization.json"
                )

        # Enumerate shard files for this split
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        shard_paths = sorted(p for p in split_dir.glob("*.npz") if p.is_file())
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {split_dir}")
        self.shard_paths: List[Path] = shard_paths

        # ------------------------------------------------------------------
        # First pass: shapes, counts, and shared-grid detection
        # ------------------------------------------------------------------
        shard_sizes: List[int] = []
        S_expected: Optional[int] = None
        G_expected: Optional[int] = None
        T_expected: Optional[int] = None

        t_ref: Optional[torch.Tensor] = None
        shared_time_grid: bool = True

        for p in shard_paths:
            g_np, t_np, y_np = _read_npz_triplet(p, mmap_mode=self._mmap_mode)

            if y_np.ndim != 3:
                raise ValueError(f"{p.name}: expected y_mat with shape [N, T, S], got {y_np.shape}")
            n_i, T_i, S_i = y_np.shape
            G_i = g_np.shape[-1] if g_np is not None else 0

            shard_sizes.append(int(n_i))

            if S_expected is None:
                S_expected, G_expected, T_expected = S_i, G_i, T_i
            else:
                if not self.skip_scan:
                    if (S_i != S_expected) or (G_i != G_expected) or (T_i != T_expected):
                        raise ValueError(
                            f"{p.name}: inconsistent shapes across shards: "
                            f"got (N=?, T={T_i}, S={S_i}, G={G_i}) "
                            f"expected (N=?, T={T_expected}, S={S_expected}, G={G_expected})"
                        )

            # Time-grid handling
            if t_np.ndim == 1:
                if t_ref is None:
                    t_ref = torch.from_numpy(np.asarray(t_np, dtype=np.float32)).reshape(-1)
                elif not self.skip_validate_grids and not self.assume_shared_grid:
                    t_vec = torch.from_numpy(np.asarray(t_np, dtype=np.float32)).reshape(-1)
                    if not torch.allclose(t_vec, t_ref, rtol=0.0, atol=0.0):
                        shared_time_grid = False
            elif t_np.ndim == 2:
                shared_time_grid = False
                if t_ref is None:
                    t_ref = torch.from_numpy(np.asarray(t_np[0], dtype=np.float32)).reshape(-1)
            else:
                raise ValueError(f"{p.name}: invalid t_vec dimensionality {t_np.ndim}")

        if S_expected is None or T_expected is None or G_expected is None:
            raise RuntimeError("Failed to infer dataset shapes from shards.")

        self.N = int(sum(shard_sizes))
        self.T = int(T_expected)
        self.S = int(S_expected)
        self.G = int(G_expected)

        # Shard offsets: prefix sums so that we can still locate rows by shard if needed
        offsets = [0]
        for n_i in shard_sizes:
            offsets.append(offsets[-1] + int(n_i))
        self._shard_offsets = torch.tensor(offsets, dtype=torch.long, device=torch.device("cpu"))

        # Shared vs per-row time grids
        if t_ref is None:
            raise RuntimeError("No time vector found in any shard.")

        if self.assume_shared_grid:
            self._shared_time_grid = True
        else:
            self._shared_time_grid = bool(shared_time_grid)

        # Allocate global tensors (contiguous, normalized runtime dtype)
        self.y = torch.empty(
            (self.N, self.T, self.S),
            device=self._stage_device,
            dtype=self._runtime_dtype,
        )
        if self.G > 0:
            self.g = torch.empty(
                (self.N, self.G),
                device=self._stage_device,
                dtype=self._runtime_dtype,
            )
        else:
            self.g = torch.empty(
                (self.N, 0),
                device=self._stage_device,
                dtype=self._runtime_dtype,
            )

        if self._shared_time_grid:
            self.t_shared = t_ref.to(device=self._stage_device, dtype=self._time_dtype)
            self.time_grid_per_row: Optional[torch.Tensor] = None
        else:
            self.t_shared = None
            self.time_grid_per_row = torch.empty(
                (self.N, self.T),
                device=self._stage_device,
                dtype=self._time_dtype,
            )

        # ------------------------------------------------------------------
        # Second pass: load + normalize into the global tensors
        # ------------------------------------------------------------------
        for shard_idx, p in enumerate(shard_paths):
            g_np, t_np, y_np = _read_npz_triplet(p, mmap_mode=self._mmap_mode)
            n_i = int(y_np.shape[0])
            start = int(self._shard_offsets[shard_idx].item())
            end = start + n_i

            # Species: normalize then cast to runtime dtype
            y_t = torch.from_numpy(np.asarray(y_np, dtype=np.float32))  # [n_i, T, S]
            y_t = self.norm.normalize(y_t, self.species_vars)  # log/std etc.
            y_t = _to_device_dtype(y_t, self._stage_device, self._runtime_dtype)
            self.y[start:end] = y_t

            # Globals
            if self.G > 0:
                if g_np is None:
                    raise RuntimeError(
                        f"{p.name}: global variables missing but dataset.G={self.G} > 0"
                    )
                g_t = torch.from_numpy(np.asarray(g_np, dtype=np.float32))  # [n_i, G]
                g_t = self.norm.normalize(g_t, self.global_vars)
                g_t = _to_device_dtype(g_t, self._stage_device, self._runtime_dtype)
                self.g[start:end] = g_t

            # Time grids
            if not self._shared_time_grid:
                if t_np.ndim == 1:
                    t_block = torch.from_numpy(np.asarray(t_np, dtype=np.float32)).view(1, -1)
                    t_block = t_block.expand(n_i, -1)
                elif t_np.ndim == 2:
                    t_block = torch.from_numpy(np.asarray(t_np, dtype=np.float32))
                else:
                    raise ValueError(f"{p.name}: invalid t_vec dimensionality {t_np.ndim}")
                t_block = _to_device_dtype(t_block, self._stage_device, self._time_dtype)
                assert self.time_grid_per_row is not None
                self.time_grid_per_row[start:end] = t_block

        # dt lookup table for shared grid (if requested)
        self.dt_table: Optional[torch.Tensor] = None
        if self._shared_time_grid and self.precompute_dt_table:
            t = self.t_shared
            if t is None:
                raise RuntimeError("Shared time grid enabled but t_shared is None.")
            T = int(t.shape[0])
            t_i = t.view(T, 1).expand(T, T)
            t_j = t.view(1, T).expand(T, T)

            dt_phys = (t_j - t_i)  # [T,T]
            dt_min_phys = torch.tensor(
                float(self.norm.dt_min_phys),
                dtype=t.dtype,
                device=t.device,
            )
            lower_or_diag = torch.ones((T, T), dtype=torch.bool, device=t.device).tril()
            dt_phys = torch.where(lower_or_diag, dt_min_phys, dt_phys)
            dt_phys = dt_phys.clamp_min(float(self.dt_epsilon))

            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)  # [T,T] float32
            self.dt_table = dt_norm.to(dtype=self._time_dtype).contiguous()

        # Derived length
        self.length = self.N * self.pairs_per_traj

        # Minimal one-time init logging
        self._log_init_summary_once(num_shards=len(self.shard_paths), shard_sizes=shard_sizes)

    # ------------------------------------------------------------------ #
    # Basic Dataset protocol                                             #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> _Index:
        row = int(idx % self.N)
        is_first = (idx // self.N) == 0
        return _Index(row=row, is_first=is_first)

    # =========================== RNG / sampling helpers ===========================

    def set_epoch(self, epoch: int) -> None:
        """Hook for external schedulers; Lightning callback calls this each epoch."""
        self._epoch = int(epoch)
        seed_everything(self._base_seed + self._epoch)

    def _sample_anchor_indices(self, B: int) -> torch.Tensor:
        """
        Sample anchor indices per row.
        """
        upper = max(0, self.T - 1 - self.min_steps)
        if upper <= 0:
            return torch.zeros(B, dtype=torch.long, device=self._stage_device)
        return torch.randint(
            low=0,
            high=upper + 1,  # randint high is exclusive
            size=(B,),
            device=self._stage_device,
            dtype=torch.long,
        )

    def _sample_offsets_conditioned(self, i_used: torch.Tensor) -> torch.Tensor:
        """
        Sample offsets o = j - i conditioned on the chosen anchor i so that j is always valid.

        Returns:
            offs : [B, K] long, with low <= offs <= min(max_steps, T-1-i_used)
        """
        if i_used.ndim != 1:
            raise ValueError(f"i_used must be 1D [B], got shape {tuple(i_used.shape)}")

        B = int(i_used.shape[0])
        k = int(self.K)
        dev = self._stage_device

        low = int(self.min_steps)
        max_off_global = int(self.max_steps) if (self.max_steps is not None) else (int(self.T) - 1)
        max_off_global = min(max_off_global, int(self.T) - 1)

        if max_off_global < low:
            raise ValueError(
                f"Invalid offset range: max_steps={self.max_steps} < min_steps={self.min_steps}"
            )

        # Per-row maximum offset so that j = i + o <= T-1
        max_off_row = torch.minimum(
            torch.full((B,), max_off_global, device=dev, dtype=torch.long),
            (self.T - 1 - i_used),
        )  # [B]

        # Defensive: should never happen if anchors are sampled with upper = T-1-min_steps
        if torch.any(max_off_row < low):
            max_off_row = torch.maximum(max_off_row, torch.full_like(max_off_row, low))

        if self.share_offsets_across_batch:
            # Preserve the meaning of "share offsets": same integer offsets for every row.
            # Use the batch-min max_off so all rows can use the shared offsets without masking.
            max_off_batch = int(max_off_row.min().item())
            width = max(1, max_off_batch - low + 1)

            offs_shared_cpu = torch.randint(
                low=0,
                high=width,
                size=(k,),
                device=torch.device("cpu"),
                dtype=torch.long,
            ) + low  # [K]
            offs = offs_shared_cpu.to(dev, non_blocking=True).view(1, k).expand(B, -1)  # [B,K]
            return offs

        # Per-row offsets with per-row ranges (vectorized via rand/floor on CPU)
        width_row = (max_off_row - low + 1).clamp_min(1)  # [B]
        width_cpu_f = width_row.to(torch.float32).cpu()  # [B] on CPU for fast rand/floor

        u = torch.rand((B, k), device="cpu")  # [B,K]
        offs_cpu = (torch.floor(u * width_cpu_f.view(B, 1)).to(torch.long) + low)  # [B,K]
        return offs_cpu.to(dev, non_blocking=True)

    # =============================== Collation ==================================

    def collate_batch(
            self, batch: Sequence[_Index]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Vectorized sampling for a batch of row indices.

        Returns:
            (y_i, dt, y_j, g, aux, k_mask)
        """
        B = len(batch)
        dev = self._stage_device

        rows = torch.tensor([b.row for b in batch], device=dev, dtype=torch.long)  # [B]

        # If enabled, force i=0 for the "first" sample of each trajectory (training only)
        force_first = self.use_first_anchor
        first_mask: Optional[torch.Tensor] = None
        if force_first and self.split != "test":
            first_mask = torch.tensor(
                [b.is_first for b in batch],
                device=dev,
                dtype=torch.bool,
            )

        # ------------------------------------------------------------------
        # Choose anchors i and offsets o = j - i
        #
        # Simplified behavior:
        #   - Always sample anchor i first.
        #   - Then sample offsets conditioned on i so that j = i + o is always valid.
        #   - No masking/clamping required; k_mask is kept for API compatibility.
        # ------------------------------------------------------------------
        i_used = self._sample_anchor_indices(B)  # [B]

        if first_mask is not None and first_mask.any():
            # Keep prior behavior: forced-first anchors apply only on non-test splits.
            i_used = torch.where(first_mask, torch.zeros_like(i_used), i_used)

        offs = self._sample_offsets_conditioned(i_used)  # [B,K]
        j_idx = i_used.unsqueeze(1) + offs  # [B,K], guaranteed < T

        # keep mask for API compatibility (now always True)
        k_mask = torch.ones((B, int(self.K)), device=dev, dtype=torch.bool)

        # ------------------------------------------------------------------
        # Gather species/globals/time for this batch from in-memory tensors
        # ------------------------------------------------------------------
        y_btS = self.y.index_select(0, rows)  # [B,T,S]
        g = self.g.index_select(0, rows) if self.G > 0 else self.g.new_zeros((B, 0))

        # y_i, y_j
        y_i = y_btS.gather(
            dim=1, index=i_used.view(B, 1, 1).expand(-1, 1, self.S)
        ).squeeze(1)  # [B,S]
        y_j = y_btS.gather(
            dim=1, index=j_idx.view(B, self.K, 1).expand(-1, -1, self.S)
        )  # [B,K,S]

        # Δt normalized -> [B,K,1]
        if self.dt_table is not None:
            dt_norm = self.dt_table.index_select(0, i_used).gather(dim=1, index=j_idx)  # [B,K]
        else:
            if self._shared_time_grid:
                t = self.t_shared
                if t is None:
                    raise RuntimeError("Shared time grid enabled but t_shared is None.")
                t_i = t.index_select(0, i_used)  # [B]
                t_j = t.index_select(0, j_idx.reshape(-1)).view(B, self.K)  # [B,K]
            else:
                if self.time_grid_per_row is None:
                    raise RuntimeError("Per-row time grids required but time_grid_per_row is None.")
                t_rows = self.time_grid_per_row.index_select(0, rows)  # [B,T]
                t_i = t_rows.gather(dim=1, index=i_used.view(B, 1)).squeeze(1)  # [B]
                t_j = t_rows.gather(dim=1, index=j_idx)  # [B,K]
            dt_phys = (t_j - t_i.unsqueeze(1)).clamp_min(float(self.dt_epsilon))  # [B,K]
            dt_norm = self.norm.normalize_dt_from_phys(dt_phys)  # [B,K] float32

        dt = dt_norm.to(dtype=self._runtime_dtype).unsqueeze(-1)  # [B,K,1]
        aux = {"i": i_used, "j": j_idx}

        # Minimal one-time batch sanity checks logged via logger
        self._log_first_batch_checks_once(y_i, dt, y_j, g, k_mask)

        return y_i, dt, y_j, g, aux, k_mask

    # ============================= Logging helpers ==============================

    def _log_init_summary_once(self, num_shards: int, shard_sizes: List[int]) -> None:
        if self._log is None or self._did_log_init_summary:
            return
        try:
            total_rows = self.N
            min_shard = min(shard_sizes) if shard_sizes else 0
            max_shard = max(shard_sizes) if shard_sizes else 0
            self._log.info(
                "FlowMapPairsDataset init: split=%s, shards=%d, total_rows=%d, "
                "T=%d, S=%d, G=%d, shared_time_grid=%s, dt_table_precomputed=%s, "
                "shard_size_range=[%d, %d], stage_device=%s, runtime_dtype=%s",
                self.split,
                num_shards,
                total_rows,
                self.T,
                self.S,
                self.G,
                bool(self._shared_time_grid),
                bool(self.dt_table is not None),
                min_shard,
                max_shard,
                str(self._stage_device),
                str(self._runtime_dtype),
            )
        finally:
            self._did_log_init_summary = True

    def _log_first_batch_checks_once(
            self,
            y_i: torch.Tensor,
            dt: torch.Tensor,
            y_j: torch.Tensor,
            g: torch.Tensor,
            k_mask: torch.Tensor,
    ) -> None:
        if self._log is None or self._did_log_first_batch_checks:
            return
        try:
            yi_finite = bool(torch.isfinite(y_i).all().item())
            yj_finite = bool(torch.isfinite(y_j).all().item())
            dt_finite = bool(torch.isfinite(dt).all().item())
            if g.numel() > 0:
                g_finite = bool(torch.isfinite(g).all().item())
            else:
                g_finite = True

            dt_min = float(dt.min().item())
            dt_max = float(dt.max().item())
            k_valid_frac = float(k_mask.float().mean().item())

            self._log.info(
                "FlowMapPairsDataset first batch checks: "
                "y_i finite=%s, y_j finite=%s, dt finite=%s, g finite=%s, "
                "dt_range=[%g, %g], k_mask_valid_frac=%.3f, "
                "batch_size=%d, K=%d, S=%d, G=%d",
                yi_finite,
                yj_finite,
                dt_finite,
                g_finite,
                dt_min,
                dt_max,
                k_valid_frac,
                int(y_i.shape[0]),
                int(y_j.shape[1]) if y_j.ndim >= 2 else 0,
                int(y_i.shape[1]) if y_i.ndim == 2 else 0,
                int(g.shape[1]) if g.ndim == 2 else 0,
            )
        finally:
            self._did_log_first_batch_checks = True


def collate_flowmap(batch: Sequence[_Index], dataset: FlowMapPairsDataset):
    return dataset.collate_batch(batch)


def create_dataloader(
        dataset: "FlowMapPairsDataset",
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        prefetch_factor: int,
) -> DataLoader:
    """
    Wrap torch.utils.data.DataLoader with the right collate_fn and drop_last.

    Notes on performance:
      * Shuffling at the DataLoader level is not strictly required for good
        training stochasticity, since anchors/offsets are resampled every
        epoch. Keeping shuffle=True for train is still fine, but this dataset
        is fully in-memory, so CPU/I/O are not the bottleneck.
    """
    num_workers = int(num_workers)
    is_train = getattr(dataset, "split", "") == "train"

    if getattr(dataset, "_stage_device", None) is not None and getattr(dataset, "_stage_device").type == "cuda":
        if num_workers > 0:
            log = getattr(dataset, "_log", None)
            msg = (
                "[dl] preload_to_gpu=True detected; forcing num_workers=0, "
                "pin_memory=False, persistent_workers=False to avoid CUDA-in-fork "
                "errors in DataLoader workers."
            )
            if log is not None:
                log.warning(msg)
            else:
                print(msg)

        num_workers = 0
        persistent_workers = False
        pin_memory = False

    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) and is_train,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers) and num_workers > 0,
        pin_memory=bool(pin_memory),
        drop_last=is_train,
        collate_fn=dataset.collate_batch,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**kwargs)


def _read_npz_triplet(path: Path, mmap_mode: Optional[str] = "r") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a preprocessed NPZ shard with optional memory mapping.

    Expected keys:
        - 'globals' or 'g_vec' or 'g_arr' : [N, G] global features (optional; may be missing)
        - 't_vec'                         : [T] or [N, T] time grid(s)
        - 'y_mat' or 'y_arr'              : [N, T, S] species trajectories
    """
    path = _as_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ shard not found: {path}")

    with np.load(path, mmap_mode=mmap_mode) as npz:
        # Globals
        if "globals" in npz:
            g_vec = npz["globals"]
        elif "g_vec" in npz:
            g_vec = npz["g_vec"]
        elif "g_arr" in npz:
            g_vec = npz["g_arr"]
        else:
            g_vec = None

        # Time
        if "t_vec" not in npz:
            raise KeyError(f"{path.name}: missing 't_vec'")
        t_vec = npz["t_vec"]

        # Species
        if "y_mat" in npz:
            y_mat = npz["y_mat"]
        elif "y_arr" in npz:
            y_mat = npz["y_arr"]
        else:
            raise KeyError(f"{path.name}: missing species array ('y_mat' or 'y_arr')")

    return g_vec, t_vec, y_mat
