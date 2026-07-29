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
  - `pairs_per_traj` counts ANCHORS per trajectory, not pairs. One dataset item
    is one anchor and emits `dataset.times_per_anchor` targets, so a trajectory
    contributes pairs_per_traj * times_per_anchor pairs per epoch.
  - With `dataset.use_first_anchor`, the FIRST of the pairs_per_traj items for a
    trajectory is pinned to t0 and the remaining anchors are drawn uniformly
    over the grid. Offsets are drawn per anchor and clipped so the target index
    stays on the grid; the anchor range does not depend on max_steps.
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

# Normalized dt can miss [0, 1] by a float ULP at the endpoints: the manifest dt
# bounds are derived from the float32 time grid, so the grid increment that IS
# the bound normalizes to a few 1e-8 outside it once the log is taken in
# float64. Only excursions of that size are absorbed; anything larger is real
# extrapolation and must still fail validation.
_DT_NORM_EDGE_TOL = 1e-6


def _snap_dt_norm_edges(dt_norm: torch.Tensor) -> torch.Tensor:
    """Snap sub-tolerance excursions at the normalized-dt bounds onto [0, 1]."""
    below = (dt_norm < 0.0) & (dt_norm >= -_DT_NORM_EDGE_TOL)
    above = (dt_norm > 1.0) & (dt_norm <= 1.0 + _DT_NORM_EDGE_TOL)
    return dt_norm.masked_fill(below, 0.0).masked_fill(above, 1.0)


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

        # Anchor range is independent of max_steps. Anchors run over the whole
        # grid (any i that admits at least min_steps of room) and the offset is
        # clipped per anchor in __getitem__ so that j = i + offset stays on the
        # grid. Deriving max_anchor from max_steps instead would collapse every
        # anchor onto t0 whenever max_steps = T-1, which is the usual setting.
        self.max_anchor = self.T - 1 - self.min_steps
        if self.max_anchor < 0:
            raise ValueError("Time grid too short")

        # Rollout (autoregressive) training mode: opt-in via training.rollout.
        # In this mode a sample is a random anchor plus H CONSECUTIVE grid steps
        # (anchor -> anchor+1 -> ... -> anchor+H); the trainer feeds the model's
        # own prediction forward. This trains the deployment stepper directly
        # (vs the default one-step-from-anchor flow map). Every anchor must be
        # random, so dataset.use_first_anchor must be false.
        rollout_cfg = (self.cfg.get("training", {}) or {}).get("rollout", {}) or {}
        self.rollout_enabled = bool(rollout_cfg.get("enabled", False))
        self.rollout_horizon = 1
        if self.rollout_enabled:
            self.rollout_horizon = int(rollout_cfg.get("horizon", 0))
            if self.rollout_horizon < 1:
                raise ValueError("training.rollout.horizon must be >= 1")
            if self.use_first_anchor:
                raise ValueError(
                    "training.rollout.enabled=true requires dataset.use_first_anchor=false "
                    "(consecutive rollout training samples random anchors over the grid)"
                )
            self.rollout_max_anchor = self.T - 1 - self.rollout_horizon
            if self.rollout_max_anchor < 0:
                raise ValueError("Time grid too short for training.rollout.horizon")

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

        # Shared time grid and the Δt table.
        #
        # Compute dt normalization in float64 to avoid spurious out-of-range
        # failures from float32 roundoff at interval endpoints.
        t_torch = torch.from_numpy(t_ref).to(torch.float64)

        # Consecutive-step Δt vector for rollout training: dt_consec[k] =
        # normalized(t[k+1] - t[k]) for k in [0, T-2]. Computed in float64 and
        # clamped to [0,1] — the smallest grid increment sits at (or, by a float
        # ULP, just below) the manifest dt lower bound, which is a pure
        # boundary-roundoff, not real extrapolation. Always built (cheap); only
        # used when training.rollout.enabled.
        dt_consec_phys = t_torch[1:] - t_torch[:-1]
        dt_consec_f64 = self.norm.normalize_dt_from_phys(dt_consec_phys).clamp_(0.0, 1.0)
        dt_consec = dt_consec_f64.to(self.dtype)  # [T-1]

        # One-step table dt_table[i, j] = normalized(t[j] - t[i]) for the default
        # one-step-from-anchor sampling. Anchors and offsets are drawn
        # independently (the offset is clipped per anchor), so the reachable set
        # is the whole upper triangle j > i and a band table is not enough. The
        # table is [T, T], the same O(T^2) as the widest band. Entries with
        # j <= i are never sampled; they are filled with the manifest dt lower
        # bound so the log10 stays defined, then zeroed. Skipped entirely in
        # rollout mode (the rollout sampler uses dt_consec, not this table).
        dt_norm = None
        if not self.rollout_enabled:
            unreachable = torch.ones((self.T, self.T), dtype=torch.bool).tril()  # j <= i
            dt_phys = t_torch[None, :] - t_torch[:, None]  # [T, T], dt_phys[i,j] = t[j]-t[i]
            dt_min_phys = 10.0 ** float(self.norm.dt_spec.log_min)
            dt_phys = torch.where(unreachable, torch.full_like(dt_phys, dt_min_phys), dt_phys)

            # Same boundary-roundoff treatment as the consecutive-step vector
            # above: the smallest grid increment sits one float32 ULP below the
            # manifest bound, which is rounding, not extrapolation.
            dt_norm_f64 = _snap_dt_norm_edges(self.norm.normalize_dt_from_phys(dt_phys))
            dt_norm_f64.masked_fill_(unreachable, 0.0)
            try:
                self.norm.validate_dt_norm(dt_norm_f64)
            except ValueError as e:
                reachable = ~unreachable
                t_min = float(np.min(t_ref))
                t_max = float(np.max(t_ref))
                dt_min = float(dt_phys[reachable].min())
                dt_max = float(dt_phys[reachable].max())
                raise ValueError(
                    f"{e} [split={self.split} t_dtype={t_ref.dtype} T={int(t_ref.shape[0])} "
                    f"t_range=[{t_min:.6g}, {t_max:.6g}] dt_phys_range=[{dt_min:.6g}, {dt_max:.6g}] "
                    f"min_steps={self.min_steps} max_steps={self.max_steps}]"
                ) from e
            dt_norm = dt_norm_f64.to(self.dtype)

        # Keep only what we need.
        self.y = y_z
        self.g = g_z
        self.dt_table = dt_norm  # [T, T] or None in rollout mode
        self.dt_consec = dt_consec  # [T-1]

        # Optionally move all tensors to GPU (requires num_workers=0 in DataLoader).
        if self.preload_to_gpu:
            self.y = self.y.to(self.device, non_blocking=False)
            self.g = self.g.to(self.device, non_blocking=False)
            if self.dt_table is not None:
                self.dt_table = self.dt_table.to(self.device, non_blocking=False)
            self.dt_consec = self.dt_consec.to(self.device, non_blocking=False)

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
        if self.rollout_enabled:
            return self._getitem_rollout(idx)

        self._ensure_worker_rng_seeded()

        # Map index to trajectory. The pairs_per_traj items of a trajectory are
        # the consecutive indices [traj * pairs_per_traj, (traj+1) * pairs_per_traj).
        traj = int(idx) // self.pairs_per_traj
        item = int(idx) % self.pairs_per_traj

        # Anchor selection: with use_first_anchor, the first of the
        # pairs_per_traj items for this trajectory is pinned to t0 and the rest
        # are uniform over [0, max_anchor]. Every epoch therefore sees each
        # trajectory once from t0 and pairs_per_traj-1 times from a random state.
        if self.use_first_anchor and item == 0:
            i = 0
        else:
            i = int(torch.randint(low=0, high=self.max_anchor + 1, size=(1,), generator=self._gen).item())

        # Offsets, clipped to the room left after the anchor so that
        # j = i + offset stays on the grid. i <= T-1-min_steps guarantees
        # max_off >= min_steps.
        K = self.times_per_anchor if self.multi_time_per_anchor else 1
        max_off = min(self.max_steps, self.T - 1 - i)

        offsets = torch.randint(low=self.min_steps, high=max_off + 1, size=(K,), generator=self._gen)
        j = i + offsets

        dev = self.device if self.preload_to_gpu else torch.device("cpu")
        j_dev = j.to(device=dev, dtype=torch.long)

        y_i = self.y[traj, i]  # [S]
        y_j = self.y[traj, j_dev]  # [K,S]
        g = self.g[traj]  # [G]

        # dt_norm from the precomputed table indexed by (anchor_i, target_j)
        dt_norm = self.dt_table[i, j_dev]  # [K]

        return y_i, dt_norm, y_j, g

    def _getitem_rollout(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout sample: random anchor + H consecutive grid steps.

        Returns (y_anchor [S], dt_steps [H], y_targets [H, S], g [G]) where
        dt_steps[s] is the normalized Δt from step (anchor+s) to (anchor+s+1) and
        y_targets[s] is the true state at (anchor+s+1). The trainer unrolls these
        H steps, feeding the model's own prediction forward (optionally detached,
        optionally with input noise). Shapes mirror the one-step contract
        (K -> H), but the SEMANTICS differ: these are consecutive, fed-forward
        steps, not K independent targets from one anchor.
        """
        self._ensure_worker_rng_seeded()
        traj = int(idx) // self.pairs_per_traj
        H = self.rollout_horizon
        a = int(torch.randint(low=0, high=self.rollout_max_anchor + 1, size=(1,), generator=self._gen).item())

        dev = self.device if self.preload_to_gpu else torch.device("cpu")
        tgt_idx = torch.arange(a + 1, a + H + 1, dtype=torch.long, device=dev)

        y_anchor = self.y[traj, a]            # [S]
        y_targets = self.y[traj, tgt_idx]     # [H, S]
        dt_steps = self.dt_consec[a:a + H]    # [H]
        g = self.g[traj]                      # [G]
        return y_anchor, dt_steps, y_targets, g


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
