#!/usr/bin/env python3
"""dataset.py

Dataset for training a flow-map model on trajectory data.

Each preprocessed shard is an NPZ with:
  - y_mat:   [N, T, S] species values in physical space
  - globals: [N, G]    global scalar inputs in physical space
  - t_vec:   [T]       shared time grid (identical across all shards)

This dataset returns training pairs in *z-space*:
  - y_i:     [S]
  - dt_norm: [K]
  - y_j:     [K, S]
  - g:       [G]

Sampling contract:
  - Pair sampling is stochastic per access using a worker-local RNG stream.
  - Each worker seeds that stream once from the DataLoader worker seed.
  - Sampling is not a pure deterministic function of (epoch, idx).
  - `set_epoch()` is intentionally a no-op in this contract.

Key invariant:
  - All trajectories (and all shards) must share an identical time grid.
    If not, we raise immediately.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.normalizer import NormalizationHelper

# PyTorch RNG seeds are 32-bit unsigned integers.
_RNG_MODULUS = 2**32


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Torch RNGs in each DataLoader worker process."""
    _ = int(worker_id)
    worker_seed = torch.initial_seed() % _RNG_MODULUS
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class FlowMapPairsDataset(torch.utils.data.Dataset):
    """Random pair sampler over trajectories with a shared time grid."""

    def __init__(
        self,
        *,
        processed_root: Path,
        split: str,
        config: Dict[str, Any],
        pairs_per_traj: int,
        min_steps: int,
        max_steps: Optional[int],
        preload_to_gpu: bool,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()

        self.processed_root = Path(processed_root)
        self.split = str(split)
        self.cfg = config
        self.pairs_per_traj = int(pairs_per_traj)
        self.min_steps = int(min_steps)
        self.max_steps_cfg = int(max_steps) if max_steps is not None else None
        self.preload_to_gpu = bool(preload_to_gpu)
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)

        self.logger = logger or logging.getLogger(__name__)

        if self.pairs_per_traj <= 0:
            raise ValueError("pairs_per_traj must be > 0")
        if self.min_steps <= 0:
            raise ValueError("min_steps must be > 0")

        # Config-driven sampling behavior.
        dcfg = self.cfg.get("dataset", {})

        # This codebase intentionally does not support memory-mapped NPZ loading.
        # Shards are loaded into RAM.
        if "mmap_mode" in dcfg:
            raise KeyError("Unsupported config key: dataset.mmap_mode")

        # This knob is not implemented (and would silently no-op). Fail fast.
        if "share_times_across_batch" in dcfg:
            raise KeyError("Unsupported config key: dataset.share_times_across_batch")

        for key in ("multi_time_per_anchor", "times_per_anchor", "use_first_anchor"):
            if key not in dcfg:
                raise KeyError(f"Missing required dataset config key: dataset.{key}")

        self.multi_time_per_anchor = bool(dcfg["multi_time_per_anchor"])
        self.times_per_anchor = int(dcfg["times_per_anchor"])
        self.use_first_anchor = bool(dcfg["use_first_anchor"])

        if self.multi_time_per_anchor and self.times_per_anchor <= 0:
            raise ValueError("times_per_anchor must be > 0")

        # Precision for normalization math.
        precision_cfg = self.cfg.get("precision")
        if not isinstance(precision_cfg, dict) or "normalize_dtype" not in precision_cfg:
            raise KeyError("Missing config: precision.normalize_dtype")
        normalize_dtype_str = str(precision_cfg["normalize_dtype"]).lower()
        if normalize_dtype_str not in {"float32", "float64", "fp32", "fp64"}:
            raise ValueError("Unsupported precision.normalize_dtype")
        self.normalize_dtype = torch.float64 if "64" in normalize_dtype_str else torch.float32

        # Load normalization manifest.
        norm_path = self.processed_root / "normalization.json"
        self.norm = NormalizationHelper(json.loads(norm_path.read_text(encoding="utf-8")))

        # Load shards and build in-memory tensors.
        split_dir = self.processed_root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")

        shard_paths = sorted(split_dir.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in: {split_dir}")

        # First pass: scan shapes and validate time grids.
        shard_shapes: list[tuple[int, ...]] = []
        g_shapes: list[tuple[int, ...]] = []
        t_ref: Optional[np.ndarray] = None

        for p in shard_paths:
            with np.load(p, allow_pickle=False) as npz:
                y_shape = npz["y_mat"].shape
                g_shape = npz["globals"].shape
                t = np.asarray(npz["t_vec"]).reshape(-1)
            if t_ref is None:
                t_ref = t
            elif not np.array_equal(t_ref, t):
                raise ValueError("Time grids are not identical")
            shard_shapes.append(y_shape)
            g_shapes.append(g_shape)

        if t_ref is None:
            raise RuntimeError("No time grid")

        if len(shard_shapes[0]) != 3:
            raise ValueError("y_mat must be [N,T,S]")

        self.n_traj = int(sum(s[0] for s in shard_shapes))
        self.T = int(shard_shapes[0][1])
        self.S = int(shard_shapes[0][2])
        self.G = int(g_shapes[0][1]) if g_shapes and len(g_shapes[0]) == 2 else 0

        # Resolve max_steps.
        max_steps_global = self.T - 1
        if self.max_steps_cfg is not None:
            max_steps_global = min(max_steps_global, self.max_steps_cfg)
        if self.min_steps > max_steps_global:
            raise ValueError("min_steps > max_steps")

        self.max_steps = int(max_steps_global)

        # Anchors are restricted so that all offsets up to max_steps are valid.
        self.max_anchor = self.T - 1 - self.max_steps
        if self.max_anchor < 0:
            raise ValueError("Time grid too short")

        # Second pass: preallocate and fill (avoids list + concatenate peak memory).
        y_phys = torch.empty(self.n_traj, self.T, self.S, dtype=self.normalize_dtype)
        g_phys = torch.empty(self.n_traj, self.G, dtype=self.normalize_dtype)
        offset = 0
        for p, ys in zip(shard_paths, shard_shapes):
            n = ys[0]
            with np.load(p, allow_pickle=False) as npz:
                y_phys[offset:offset + n] = torch.from_numpy(np.asarray(npz["y_mat"])).to(self.normalize_dtype)
                g_phys[offset:offset + n] = torch.from_numpy(np.asarray(npz["globals"])).to(self.normalize_dtype)
            offset += n

        # Normalize to z-space (species) and normalized globals.
        species_vars = list(self.cfg["data"]["species_variables"])
        global_vars = list(self.cfg["data"]["global_variables"])
        if len(species_vars) != self.S:
            raise ValueError("species_variables mismatch")
        if len(global_vars) != self.G:
            raise ValueError("global_variables mismatch")

        y_z = self.norm.normalize(y_phys, species_vars).to(self.dtype)
        g_z = self.norm.normalize(g_phys, global_vars).to(self.dtype) if self.G > 0 else g_phys.to(self.dtype)
        # y_phys/g_phys are large staging tensors; free them before dt-table work.
        del y_phys, g_phys

        # Shared time grid and a compact Δt table.
        #
        # We only ever sample anchors i in [0, max_anchor] and offsets in
        # [min_steps, max_steps]. Precompute exactly that band:
        #   dt_table[i, off] = normalized(t[i + (min_steps + off)] - t[i])
        # Compute dt normalization in float64 to avoid spurious out-of-range
        # failures from float32 roundoff at interval endpoints.
        t_torch = torch.from_numpy(t_ref).to(torch.float64)

        anchor_idx = torch.arange(0, self.max_anchor + 1, dtype=torch.long)
        offsets = torch.arange(self.min_steps, self.max_steps + 1, dtype=torch.long)
        j_idx = anchor_idx[:, None] + offsets[None, :]

        dt_phys = t_torch[j_idx] - t_torch[anchor_idx[:, None]]  # [A, O]
        dt_norm_f64 = self.norm.normalize_dt_from_phys(dt_phys)
        try:
            self.norm.validate_dt_norm(dt_norm_f64)
        except ValueError as e:
            t_min = float(np.min(t_ref))
            t_max = float(np.max(t_ref))
            dt_min = float(dt_phys.min())
            dt_max = float(dt_phys.max())
            raise ValueError(
                f"{e} [split={self.split} t_dtype={t_ref.dtype} T={int(t_ref.shape[0])} "
                f"t_range=[{t_min:.6g}, {t_max:.6g}] dt_phys_range=[{dt_min:.6g}, {dt_max:.6g}] "
                f"min_steps={self.min_steps} max_steps={self.max_steps}]"
            ) from e
        dt_norm = dt_norm_f64.to(self.dtype)

        # Keep only what we need.
        self.y = y_z
        self.g = g_z
        self.dt_table = dt_norm  # [A, O]

        # Optionally move all tensors to GPU (requires num_workers=0 in DataLoader).
        if self.preload_to_gpu:
            self.y = self.y.to(self.device, non_blocking=False)
            self.g = self.g.to(self.device, non_blocking=False)
            self.dt_table = self.dt_table.to(self.device, non_blocking=False)

        self.logger.info(
            "Loaded %s split: N=%d T=%d S=%d G=%d dtype=%s",
            self.split,
            self.n_traj,
            self.T,
            self.S,
            self.G,
            str(self.dtype).replace("torch.", ""),
        )

        # Worker-local generator seeded lazily on first __getitem__.
        # This avoids expensive per-sample reseeding.
        self._gen = torch.Generator()
        self._gen_seeded = False

    def set_epoch(self, epoch: int) -> None:
        """No-op: sampling uses continuous worker-local RNG streams."""
        _ = int(epoch)

    def _ensure_worker_rng_seeded(self) -> None:
        if self._gen_seeded:
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = int(self.seed) % _RNG_MODULUS
        else:
            # Deterministic, worker-unique seed set by DataLoader worker init.
            seed = int(worker_info.seed) % _RNG_MODULUS

        self._gen.manual_seed(seed)
        self._gen_seeded = True

    def __len__(self) -> int:
        return self.n_traj * self.pairs_per_traj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_worker_rng_seeded()

        # Map index to trajectory.
        traj = int(idx) // self.pairs_per_traj

        # Anchor selection.
        if self.use_first_anchor:
            i = 0
        else:
            i = int(torch.randint(low=0, high=self.max_anchor + 1, size=(1,), generator=self._gen).item())

        # Offsets.
        K = self.times_per_anchor if self.multi_time_per_anchor else 1

        offsets = torch.randint(low=self.min_steps, high=self.max_steps + 1, size=(K,), generator=self._gen)
        j = i + offsets

        dev = self.device if self.preload_to_gpu else torch.device("cpu")
        j_dev = j.to(device=dev, dtype=torch.long)

        y_i = self.y[traj, i]  # [S]
        y_j = self.y[traj, j_dev]  # [K,S]
        g = self.g[traj]  # [G]

        # dt_norm from precomputed band table indexed by (anchor_i, offset)
        dt_norm = self.dt_table[i, offsets.to(dtype=torch.long, device=dev) - self.min_steps]  # [K]

        return y_i, dt_norm, y_j, g


# -----------------------------------------------------------------------------
# DataLoader helper
# -----------------------------------------------------------------------------


def create_dataloader(
    *,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    prefetch_factor: int,
) -> torch.utils.data.DataLoader:
    """Create a seeded DataLoader for deterministic index shuffling."""

    if getattr(dataset, "preload_to_gpu", False) and num_workers != 0:
        raise ValueError("preload_to_gpu requires num_workers=0")

    # The DataLoader generator controls shuffle deterministically.
    seed = getattr(dataset, "seed", 0)
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    nw = int(num_workers)
    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=nw,
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=gen,
        worker_init_fn=_seed_worker if nw > 0 else None,
        persistent_workers=bool(persistent_workers) if nw > 0 else False,
    )
    if nw > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return torch.utils.data.DataLoader(**kwargs)
