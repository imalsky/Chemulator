#!/usr/bin/env python3
"""dataset.py

Batch variables and structures:
    y_i : [B, S]          (runtime dtype)
    dt  : [B, K, 1]       (dt-spec normalized, runtime dtype)
    y_j : [B, K, S]       (runtime dtype)
    g   : [B, G]          (runtime dtype)
    aux : {'i':[B], 'j':[B,K]}
    k_mask : [B, K]       (True where j is valid)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from normalizer import NormalizationHelper
from utils import load_json_config as load_json


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


def _preview(items: Sequence[str], max_items: int = 10) -> str:
    items = list(items)
    if len(items) <= max_items:
        return "[" + ", ".join(repr(x) for x in items) + "]"
    head = ", ".join(repr(x) for x in items[:max_items])
    return f"[{head}, ...] (+{len(items) - max_items} more)"


def _check_duplicates(items: Sequence[str], label: str) -> None:
    items = list(items)
    if len(items) != len(set(items)):
        dupes = [x for x in dict.fromkeys(items) if items.count(x) > 1]
        raise ValueError(f"{label} contains duplicate entries: {dupes}")


@dataclass(frozen=True)
class _Index:
    row: int        # global trajectory index in [0, N)
    is_first: bool  # True for the first of the pairs_per_traj samples for that row


class FlowMapPairsDataset(Dataset):
    """
    High-performance in-memory dataset with K future targets per anchor.

    Key behavior:
      - This dataset does not precompute (i, j) pairs.
      - Each batch samples anchors/offsets on the fly inside collate_batch().
      - With DataLoader workers, randomness lives inside the worker process; therefore
        reproducibility depends on DataLoader's worker seeding, not on dataset.set_epoch().
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

        self._log = logger if logger is not None else logging.getLogger(__name__)
        log = self._log

        # One-time logging flags
        self._did_log_init_summary: bool = False
        self._did_log_first_batch_checks: bool = False

        # Dataset config knobs
        dcfg = dict(self.cfg.get("dataset", {}))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", True))
        self.K = int(dcfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dcfg.get("share_times_across_batch", False))

        # Enforce i=0 for the first of the pairs_per_traj samples (TRAIN + VAL).
        if "use_first_anchor" in dcfg:
            self.use_first_anchor = bool(dcfg.get("use_first_anchor", True))
        else:
            self.use_first_anchor = (self.split != "test")

        # Time-grid / IO options
        self.skip_scan = bool(dcfg.get("skip_scan", False))
        self.skip_validate_grids = bool(dcfg.get("skip_validate_grids", False))
        self.assume_shared_grid = bool(dcfg.get("assume_shared_grid", False))
        self._mmap_mode = (str(dcfg.get("mmap_mode", "r")) if dcfg.get("mmap_mode", "r") else None)

        # Stage device and dtypes (if preload_to_gpu=True, DataLoader workers must be disabled)
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype
        self._time_dtype = torch.float32

        # Δt clamp floor
        self.dt_epsilon = float(dcfg.get("dt_epsilon", self.cfg.get("normalization", {}).get("epsilon", 1e-30)))

        # Normalization helper (requires manifest)
        manifest_path = self.root / "normalization.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Processed meta (authoritative column order for shards)
        meta = manifest.get("meta", {}) or {}
        processed_species_all: List[str] = list(meta.get("species_variables", []) or [])
        if not processed_species_all:
            raise RuntimeError(
                "normalization.json is missing meta.species_variables; cannot validate or index species columns."
            )
        processed_species_index = {name: i for i, name in enumerate(processed_species_all)}

        processed_globals_all: List[str] = list(meta.get("global_variables", []) or [])
        processed_globals_index = {name: i for i, name in enumerate(processed_globals_all)}

        # Config data section
        dcfg_data = dict(self.cfg.get("data", {}))
        self.time_key: str = str(dcfg_data.get("time_variable", meta.get("time_variable", "t_time")))

        cfg_species_raw = dcfg_data.get("species_variables", []) or []
        cfg_targets_raw = dcfg_data.get("target_species", []) or []
        cfg_globals_raw = dcfg_data.get("global_variables", []) or []

        cfg_species: List[str] = list(cfg_species_raw) if cfg_species_raw else []
        cfg_targets: List[str] = list(cfg_targets_raw) if cfg_targets_raw else []
        cfg_globals: List[str] = list(cfg_globals_raw) if cfg_globals_raw else []

        # Resolve species selection:
        # - Empty list means "unspecified": use all processed species.
        # - Non-empty list must be a subset of processed species.
        if cfg_species:
            _check_duplicates(cfg_species, "cfg.data.species_variables")
            missing = [s for s in cfg_species if s not in processed_species_index]
            if missing:
                raise ValueError(
                    "cfg.data.species_variables contains species not present in processed artifacts: "
                    f"{missing}. Processed species={_preview(processed_species_all)}"
                )
            selected_species = list(cfg_species)
            if selected_species != processed_species_all:
                log.info(
                    "Dataset species selection: using cfg.data.species_variables (%d of %d processed): %s",
                    len(selected_species),
                    len(processed_species_all),
                    _preview(selected_species),
                )
            else:
                log.info(
                    "Dataset species selection: cfg.data.species_variables matches processed list (%d species).",
                    len(selected_species),
                )
        else:
            selected_species = list(processed_species_all)
            log.info(
                "Dataset species selection: cfg.data.species_variables is empty; using all %d processed species.",
                len(selected_species),
            )

        # Resolve target selection:
        # - If specified, must be subset of processed species and identical to species_variables.
        # - If empty, defaults to species_variables.
        if cfg_targets:
            _check_duplicates(cfg_targets, "cfg.data.target_species")
            missing = [s for s in cfg_targets if s not in processed_species_index]
            if missing:
                raise ValueError(
                    "cfg.data.target_species contains species not present in processed artifacts: "
                    f"{missing}. Processed species={_preview(processed_species_all)}"
                )
            if cfg_targets != selected_species:
                raise ValueError(
                    "cfg.data.target_species must be identical to cfg.data.species_variables. "
                    f"species_variables={_preview(selected_species)}; target_species={_preview(cfg_targets)}"
                )
            selected_targets = list(cfg_targets)
        else:
            selected_targets = list(selected_species)
            log.info(
                "Dataset target selection: cfg.data.target_species is empty; using the same list as species_variables (%d).",
                len(selected_targets),
            )

        # Resolve globals selection:
        # - If processed has globals, empty config means use all processed globals.
        # - If specified, must be subset of processed globals (and will be selected in that order).
        if processed_globals_all:
            if cfg_globals:
                _check_duplicates(cfg_globals, "cfg.data.global_variables")
                missing = [g for g in cfg_globals if g not in processed_globals_index]
                if missing:
                    raise ValueError(
                        "cfg.data.global_variables contains globals not present in processed artifacts: "
                        f"{missing}. Processed globals={_preview(processed_globals_all)}"
                    )
                selected_globals = list(cfg_globals)
                if selected_globals != processed_globals_all:
                    log.info(
                        "Dataset globals selection: using cfg.data.global_variables (%d of %d processed): %s",
                        len(selected_globals),
                        len(processed_globals_all),
                        _preview(selected_globals),
                    )
                else:
                    log.info(
                        "Dataset globals selection: cfg.data.global_variables matches processed list (%d).",
                        len(selected_globals),
                    )
            else:
                selected_globals = list(processed_globals_all)
                log.info(
                    "Dataset globals selection: cfg.data.global_variables is empty; using all %d processed globals: %s",
                    len(selected_globals),
                    _preview(selected_globals),
                )
        else:
            # No processed globals; require config to be empty.
            if cfg_globals:
                raise ValueError(
                    "Processed artifacts contain no global_variables, but cfg.data.global_variables is non-empty: "
                    f"{_preview(cfg_globals)}"
                )
            selected_globals = []

        # Persist resolved lists on the dataset instance.
        self.species_vars = list(selected_species)
        self.target_species = list(selected_targets)
        self.global_vars = list(selected_globals)

        self._processed_species_all = list(processed_species_all)
        self._processed_globals_all = list(processed_globals_all)

        self._species_idx = [processed_species_index[s] for s in self.species_vars]
        self._globals_idx = [processed_globals_index[g] for g in self.global_vars] if self.global_vars else []

        # Normalization methods must exist for all used keys.
        norm_methods = manifest.get("normalization_methods", {})
        for key in (self.species_vars + self.global_vars + [self.time_key]):
            if key not in norm_methods:
                raise RuntimeError(f"Normalization method missing for variable {key!r} in normalization.json")

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
        G_expected_raw: Optional[int] = None
        T_expected: Optional[int] = None

        t_ref: Optional[torch.Tensor] = None
        shared_time_grid: bool = True

        for p in shard_paths:
            g_np, t_np, y_np = _read_npz_triplet(p, mmap_mode=self._mmap_mode)

            if y_np.ndim != 3:
                raise ValueError(f"{p.name}: expected y_mat with shape [N, T, S], got {y_np.shape}")
            n_i, T_i, S_i = y_np.shape
            G_i = int(g_np.shape[-1]) if g_np is not None else 0

            shard_sizes.append(int(n_i))

            if S_expected is None:
                S_expected, G_expected_raw, T_expected = S_i, G_i, T_i
            else:
                if not self.skip_scan:
                    if (S_i != S_expected) or (G_i != G_expected_raw) or (T_i != T_expected):
                        raise ValueError(
                            f"{p.name}: inconsistent shapes across shards: "
                            f"got (N=?, T={T_i}, S={S_i}, G={G_i}) "
                            f"expected (N=?, T={T_expected}, S={S_expected}, G={G_expected_raw})"
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

        if S_expected is None or T_expected is None or G_expected_raw is None:
            raise RuntimeError("Failed to infer dataset shapes from shards.")

        # Validate processed shard dimensions against manifest meta.
        if int(S_expected) != len(self._processed_species_all):
            raise RuntimeError(
                "Processed shard species dimension does not match normalization.json meta.species_variables. "
                f"Shards report S={int(S_expected)} but meta.species_variables has {len(self._processed_species_all)} entries. "
                "This usually indicates mixed artifacts or an out-of-sync processed directory."
            )
        if self._processed_globals_all and int(G_expected_raw) != len(self._processed_globals_all):
            raise RuntimeError(
                "Processed shard globals dimension does not match normalization.json meta.global_variables. "
                f"Shards report G={int(G_expected_raw)} but meta.global_variables has {len(self._processed_globals_all)} entries. "
                "This usually indicates mixed artifacts or an out-of-sync processed directory."
            )
        if (not self._processed_globals_all) and int(G_expected_raw) != 0:
            raise RuntimeError(
                "Processed shards contain globals (G>0) but normalization.json meta.global_variables is empty. "
                "This usually indicates inconsistent artifacts."
            )

        self.N = int(sum(shard_sizes))
        self.T = int(T_expected)
        self.S_raw = int(S_expected)
        self.S = int(len(self.species_vars))
        self.G_raw = int(G_expected_raw)
        self.G = int(len(self.global_vars))

        # Shared time grid flag
        self._shared_time_grid = bool(shared_time_grid) if not self.assume_shared_grid else True

        # ------------------------------------------------------------------
        # Second pass: preload + normalization
        # ------------------------------------------------------------------
        self.g_table: Optional[torch.Tensor] = None
        self.dt_table: Optional[torch.Tensor] = None
        self.t_shared: Optional[torch.Tensor] = None
        self.time_grid_per_row: Optional[torch.Tensor] = None
        self.y_table: Optional[torch.Tensor] = None

        # Preallocate on stage device
        self.g = torch.empty((self.N, self.G), device=self._stage_device, dtype=self._runtime_dtype)
        self.y = torch.empty((self.N, self.T, self.S), device=self._stage_device, dtype=self._runtime_dtype)

        if not self._shared_time_grid:
            self.time_grid_per_row = torch.empty((self.N, self.T), device=self._stage_device, dtype=self._time_dtype)
        else:
            if t_ref is None:
                raise RuntimeError("Shared time grid enabled but failed to load a reference time vector.")
            self.t_shared = _to_device_dtype(t_ref, self._stage_device, self._time_dtype).reshape(-1)

        # Load shards into preallocated tensors
        shard_offsets: List[int] = [0]
        for n_i in shard_sizes:
            shard_offsets.append(shard_offsets[-1] + int(n_i))

        for shard_idx, p in enumerate(shard_paths):
            start = shard_offsets[shard_idx]
            end = shard_offsets[shard_idx + 1]

            g_np, t_np, y_np = _read_npz_triplet(p, mmap_mode=self._mmap_mode)

            # Globals
            if self.G > 0:
                if g_np is None:
                    raise RuntimeError(
                        f"{p.name}: missing globals array, but cfg.data.global_variables is non-empty ({_preview(self.global_vars)})"
                    )
                if g_np.ndim != 2:
                    raise ValueError(f"{p.name}: expected globals with shape [N, G], got {g_np.shape}")

                g_np_sel = np.asarray(g_np[:, self._globals_idx], dtype=np.float32) if self._globals_idx else np.asarray(g_np, dtype=np.float32)
                g_t = torch.from_numpy(g_np_sel).to(device=self._stage_device, dtype=torch.float32, non_blocking=True)
                g_t = self.norm.normalize(g_t, self.global_vars)
                self.g[start:end].copy_(g_t.to(dtype=self._runtime_dtype), non_blocking=True)
            else:
                # No globals used.
                pass

            # Species: select requested columns (cfg order), normalize, then cast
            y_np_sel = np.asarray(y_np[..., self._species_idx], dtype=np.float32)  # [n_i, T, S_selected]
            y_t = torch.from_numpy(y_np_sel).to(device=self._stage_device, dtype=torch.float32, non_blocking=True)
            y_t = self.norm.normalize(y_t, self.species_vars)
            self.y[start:end].copy_(y_t.to(dtype=self._runtime_dtype), non_blocking=True)

            # Time grids
            if not self._shared_time_grid:
                if t_np.ndim == 1:
                    t_row = np.asarray(t_np, dtype=np.float32).reshape(1, -1)
                    t_row = np.repeat(t_row, repeats=(end - start), axis=0)
                else:
                    t_row = np.asarray(t_np, dtype=np.float32)
                t_t = torch.from_numpy(t_row).to(device=self._stage_device, dtype=self._time_dtype, non_blocking=True)
                assert self.time_grid_per_row is not None
                self.time_grid_per_row[start:end].copy_(t_t, non_blocking=True)

        # Precompute dt table if requested and feasible
        if self.precompute_dt_table:
            if self._shared_time_grid:
                # dt_table is indexed by time indices: dt_table[i, j] corresponds to dt(t_i -> t_j)
                assert self.t_shared is not None
                t = self.t_shared.to(dtype=self._time_dtype)
                dt_phys = t.view(1, self.T) - t.view(self.T, 1)  # [T, T], dt_phys[i, j] = t[j] - t[i]

                # For safety, force non-positive dt values to the dt_min floor.
                dt_min_phys = torch.tensor(float(self.norm.dt_min_phys), device=self._stage_device, dtype=self._time_dtype)
                lower_or_diag = torch.tril(torch.ones((self.T, self.T), device=self._stage_device, dtype=torch.bool))
                dt_phys = torch.where(lower_or_diag, dt_min_phys, dt_phys)

                dt_phys = dt_phys.clamp_min(float(self.dt_epsilon))
                dt_norm = self.norm.normalize_dt_from_phys(dt_phys)  # float32
                self.dt_table = dt_norm.to(device=self._stage_device, dtype=self._time_dtype).contiguous()
            else:
                # Per-row dt_table would be O(N*T^2) memory and is not feasible for typical runs.
                log.info(
                    "dt_table precompute requested but disabled because per-row time grids are in use (shared_time_grid=False). "
                    "Δt will be computed on the fly."
                )
                self.dt_table = None

        self.length = int(self.N * self.pairs_per_traj)
        self._log_init_summary_once(num_shards=len(shard_paths), shard_sizes=shard_sizes)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> _Index:
        row = int(idx % self.N)
        is_first = (idx // self.N) == 0
        return _Index(row=row, is_first=is_first)

    # =========================== RNG / sampling helpers ===========================

    def _sample_anchor_indices(self, B: int) -> torch.Tensor:
        """
        Sample anchor indices per row.

        Sampling is performed on CPU so it works in DataLoader workers even if
        staging tensors live on GPU (workers are disabled in that case).
        """
        upper = max(0, self.T - 1 - self.min_steps)
        if upper <= 0:
            out = torch.zeros(B, dtype=torch.long, device="cpu")
        else:
            out = torch.randint(
                low=0,
                high=upper + 1,  # randint high is exclusive
                size=(B,),
                device="cpu",
                dtype=torch.long,
            )

        dev = self._stage_device
        if dev.type == "cpu":
            return out
        return out.to(dev, non_blocking=True)

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
            raise ValueError(f"Invalid offset range: max_steps={self.max_steps} < min_steps={self.min_steps}")

        # Per-row maximum offset so that j = i + o <= T-1
        max_off_row = torch.minimum(
            torch.full((B,), max_off_global, device=dev, dtype=torch.long),
            (self.T - 1 - i_used),
        )  # [B]

        if torch.any(max_off_row < low):
            raise RuntimeError("Invalid sampling state: max_off_row < min_steps")

        if self.share_offsets_across_batch:
            # Same offsets for every row; choose offsets using batch-min to keep all rows valid.
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

        # Per-row offsets with per-row ranges; rand/floor on CPU for speed.
        width_row = (max_off_row - low + 1).clamp_min(1)  # [B]
        width_cpu_f = width_row.to(torch.float32).cpu()  # [B]

        u = torch.rand((B, k), device="cpu")  # [B,K]
        offs_cpu = (torch.floor(u * width_cpu_f.view(B, 1)).to(torch.long) + low)  # [B,K]
        return offs_cpu.to(dev, non_blocking=True)

    # =============================== Collation ==================================

    def collate_batch(
        self, batch: Sequence[_Index]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Vectorized sampling for a batch of row indices.

        Returns:
            (y_i, dt, y_j, g, aux, k_mask)
        """
        B = len(batch)
        dev = self._stage_device

        rows = torch.tensor([b.row for b in batch], device=dev, dtype=torch.long)  # [B]

        first_mask: Optional[torch.Tensor] = None
        if self.use_first_anchor and self.split != "test":
            first_mask = torch.tensor([b.is_first for b in batch], device=dev, dtype=torch.bool)

        # Choose anchors i and offsets o=j-i (conditioned so j is always valid)
        i_used = self._sample_anchor_indices(B)  # [B]
        if first_mask is not None and bool(first_mask.any().item()):
            i_used = torch.where(first_mask, torch.zeros_like(i_used), i_used)

        offs = self._sample_offsets_conditioned(i_used)  # [B,K]
        j_idx = i_used.unsqueeze(1) + offs  # [B,K], guaranteed < T

        # Keep mask for API compatibility (always True because j is always valid)
        k_mask = torch.ones((B, int(self.K)), device=dev, dtype=torch.bool)

        # Gather species/globals/time
        y_btS = self.y.index_select(0, rows)  # [B,T,S]
        g = self.g.index_select(0, rows) if self.G > 0 else self.y.new_zeros((B, 0))

        y_i = y_btS.gather(dim=1, index=i_used.view(B, 1, 1).expand(-1, 1, self.S)).squeeze(1)  # [B,S]
        y_j = y_btS.gather(dim=1, index=j_idx.view(B, self.K, 1).expand(-1, -1, self.S))  # [B,K,S]

        # Δt normalized -> [B,K,1]
        if self.dt_table is not None:
            # dt_table is [T,T]; index_select rows by i, then gather cols by j
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
                "T=%d, S=%d (raw=%d), G=%d (raw=%d), shared_time_grid=%s, dt_table_precomputed=%s, "
                "shard_size_range=[%d, %d], stage_device=%s, runtime_dtype=%s",
                self.split,
                num_shards,
                total_rows,
                self.T,
                self.S,
                self.S_raw,
                self.G,
                self.G_raw,
                bool(self._shared_time_grid),
                bool(self.dt_table is not None),
                min_shard,
                max_shard,
                str(self._stage_device),
                str(self._runtime_dtype),
            )
            self._log.info(
                "FlowMapPairsDataset variables: species_variables=%s; global_variables=%s; time_variable=%r",
                _preview(self.species_vars),
                _preview(self.global_vars),
                self.time_key,
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
            g_finite = bool(torch.isfinite(g).all().item()) if g.numel() > 0 else True

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


def create_dataloader(
    dataset: FlowMapPairsDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    """
    Create a DataLoader with reproducible per-epoch worker seeding.

    Notes:
      - A dedicated torch.Generator isolates DataLoader seeding from unrelated global RNG use.
      - If staging tensors live on CUDA, DataLoader workers are forced to 0 (CUDA-in-fork is unsafe).
    """
    num_workers = int(num_workers)
    is_train = getattr(dataset, "split", "") == "train"

    # If staging tensors live on CUDA, do not use workers (CUDA-in-fork issues).
    if getattr(dataset, "_stage_device", None) is not None and getattr(dataset, "_stage_device").type == "cuda":
        if num_workers > 0:
            log = getattr(dataset, "_log", None)
            msg = (
                "[dl] preload_to_gpu=True detected; forcing num_workers=0, "
                "pin_memory=False, persistent_workers=False to avoid CUDA-in-fork issues."
            )
            if log is not None:
                log.warning(msg)
            else:
                print(msg)

        num_workers = 0
        persistent_workers = False
        pin_memory = False

    base_seed = int(getattr(dataset, "_base_seed", 0))

    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())

    gen = torch.Generator(device="cpu")
    gen.manual_seed(base_seed + 1_000_003 * rank)

    def _worker_init_fn(worker_id: int) -> None:
        seed32 = int(torch.initial_seed() % (2**32))
        np.random.seed(seed32)

    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) and is_train,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers) and num_workers > 0,
        pin_memory=bool(pin_memory),
        drop_last=is_train,
        collate_fn=dataset.collate_batch,
        generator=gen,
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["worker_init_fn"] = _worker_init_fn

    return DataLoader(**kwargs)


def _read_npz_triplet(
    path: Path,
    mmap_mode: Optional[str] = "r",
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
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
