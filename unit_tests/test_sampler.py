#!/usr/bin/env python3
"""Tests for the one-step anchor/offset sampler in src/dataset.py.

Pins the sampling contract described in the paper (Sec. 2.3): with
`dataset.use_first_anchor`, the first of the `pairs_per_traj` items for a
trajectory is anchored at t0 and the remaining anchors are drawn uniformly over
the grid, with the offset clipped per anchor so the target stays on the grid.
The anchor range must NOT depend on `training.max_steps`.

Also covers the normalized-dt boundary rounding: the smallest grid increment can
land a float ULP below the manifest dt lower bound, which must be snapped rather
than reported as extrapolation, while a genuinely out-of-range dt must still be
a hard error.

Self-contained: builds a tiny synthetic processed dir, so no real data is needed.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import FlowMapPairsDataset

_SPECIES = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution",
]
_GLOBALS = ["P", "T"]
_S, _G = len(_SPECIES), len(_GLOBALS)
_N = 32
# Strictly increasing grid spanning many decades. Like the real grid, the
# smallest increment is NOT the first one (t[2]-t[1] = 3e-4 < t[1]-t[0] = 1e-3),
# so an anchor-agnostic dt table must reach below t[1]-t[0].
_T_VEC = np.array(
    [1e-3, 2e-3, 2.3e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3, 1e5, 1e8], dtype=np.float64
).astype(np.float32)
_T = int(_T_VEC.shape[0])

# Manifest dt bounds, derived the same way the dataset reads the grid (float64
# arithmetic on the stored float32 values) so the unperturbed manifest is exact.
_T64 = _T_VEC.astype(np.float64)
_DT_LOG_MIN = math.log10(float(np.min(_T64[1:] - _T64[:-1])))
_DT_LOG_MAX = math.log10(float(_T64[-1] - _T64[0]))


def _write_processed(root: Path, *, dt_log_min_offset: float = 0.0) -> None:
    """Write a minimal processed dir: manifest + summary + train shard.

    `dt_log_min_offset` raises the manifest dt lower bound above the smallest
    achievable grid increment, which is exactly what float32 storage of the time
    grid does to a bound computed from the raw float64 times.
    """
    per_key = {s: {"log_mean": 0.0, "log_std": 1.0} for s in _SPECIES}
    per_key["P"] = {"log_min": -6.0, "log_max": 4.0}
    per_key["T"] = {"mean": 1500.0, "std": 500.0}
    manifest = {
        "per_key_stats": per_key,
        "normalization_methods": {
            **{s: "log-standard" for s in _SPECIES}, "P": "log-min-max", "T": "standard",
        },
        "dt": {"log_min": _DT_LOG_MIN + dt_log_min_offset, "log_max": _DT_LOG_MAX},
        "epsilon": 1e-30, "min_std": 1e-10,
        "meta": {"species_variables": _SPECIES, "global_variables": _GLOBALS,
                 "time_variable": "t_time"},
    }
    (root / "normalization.json").write_text(json.dumps(manifest))
    (root / "preprocessing_summary.json").write_text(json.dumps({
        "species_variables": _SPECIES, "global_variables": _GLOBALS,
        "time_variable": "t_time", "time_grid_len": _T,
    }))
    (root / "shard_index.json").write_text(json.dumps({"splits": {"train": 1}}))

    rng = np.random.default_rng(0)
    d = root / "train"
    d.mkdir(parents=True, exist_ok=True)
    # Distinct values everywhere so a sampled state identifies its time index.
    y = (10.0 ** rng.uniform(-8.0, -1.0, size=(_N, _T, _S))).astype("float32")
    g = np.stack([10.0 ** rng.uniform(-2.0, 2.0, size=_N),
                  rng.uniform(600.0, 2400.0, size=_N)], axis=1).astype("float32")
    np.savez(d / "shard_train_0.npz", y_mat=y, globals=g, t_vec=_T_VEC)


def _cfg(*, times_per_anchor: int = 4, use_first_anchor: bool = True) -> dict:
    return {
        "precision": {"normalize_dtype": "float64"},
        "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS),
                 "time_variable": "t_time"},
        "dataset": {"multi_time_per_anchor": True, "times_per_anchor": times_per_anchor,
                    "use_first_anchor": use_first_anchor},
    }


def _dataset(root: Path, *, pairs_per_traj: int = 8, min_steps: int = 1,
             max_steps: int = _T - 1, times_per_anchor: int = 4,
             use_first_anchor: bool = True) -> FlowMapPairsDataset:
    return FlowMapPairsDataset(
        processed_root=root, split="train",
        config=_cfg(times_per_anchor=times_per_anchor, use_first_anchor=use_first_anchor),
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=False, device=torch.device("cpu"), dtype=torch.float32,
        seed=0, logger=logging.getLogger("t"),
    )


def _time_index(ds: FlowMapPairsDataset, traj: int, state: torch.Tensor) -> int:
    """Recover the grid index a sampled state came from (states are distinct)."""
    hits = torch.nonzero((ds.y[traj] == state.reshape(1, -1)).all(dim=1)).reshape(-1)
    if hits.numel() != 1:
        raise AssertionError(f"state did not match exactly one grid index (got {hits.numel()})")
    return int(hits[0].item())


class AnchorSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_processed(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_anchor_range_independent_of_max_steps(self) -> None:
        """max_steps = T-1 must still allow anchors across the whole grid."""
        for max_steps in (_T - 1, _T // 2, 1):
            ds = _dataset(self.root, max_steps=max_steps)
            self.assertEqual(ds.max_anchor, _T - 1 - ds.min_steps)

    def test_first_item_of_each_trajectory_is_t0(self) -> None:
        ppt = 8
        ds = _dataset(self.root, pairs_per_traj=ppt)
        for traj in range(_N):
            y_i, _dt, _y_j, _g = ds[traj * ppt]
            self.assertEqual(_time_index(ds, traj, y_i), 0)

    def test_remaining_items_sample_interior_anchors(self) -> None:
        ppt = 8
        ds = _dataset(self.root, pairs_per_traj=ppt)
        anchors = set()
        for traj in range(_N):
            for item in range(1, ppt):
                y_i, _dt, _y_j, _g = ds[traj * ppt + item]
                anchors.add(_time_index(ds, traj, y_i))
        # Every anchor in [0, max_anchor] is reachable; require broad coverage
        # rather than exhaustiveness so the test does not depend on RNG luck.
        self.assertGreater(len(anchors), (ds.max_anchor + 1) // 2)
        self.assertLessEqual(max(anchors), ds.max_anchor)

    def test_use_first_anchor_false_pins_nothing(self) -> None:
        ppt = 8
        ds = _dataset(self.root, pairs_per_traj=ppt, use_first_anchor=False)
        anchors = [
            _time_index(ds, idx // ppt, ds[idx][0]) for idx in range(_N * ppt)
        ]
        self.assertGreater(len(set(anchors)), 1)
        # t0 must no longer be over-represented: it is one of max_anchor+1 draws.
        self.assertLess(anchors.count(0), len(anchors) // 4)

    def test_targets_stay_on_grid_and_respect_step_bounds(self) -> None:
        ppt, min_steps, max_steps = 8, 2, 4
        ds = _dataset(self.root, pairs_per_traj=ppt, min_steps=min_steps, max_steps=max_steps)
        for idx in range(_N * ppt):
            traj = idx // ppt
            y_i, _dt, y_j, _g = ds[idx]
            i = _time_index(ds, traj, y_i)
            for k in range(y_j.shape[0]):
                j = _time_index(ds, traj, y_j[k])
                self.assertLessEqual(j, _T - 1)
                self.assertGreaterEqual(j - i, min_steps)
                self.assertLessEqual(j - i, max_steps)

    def test_returned_dt_matches_the_sampled_pair(self) -> None:
        ppt = 8
        ds = _dataset(self.root, pairs_per_traj=ppt, use_first_anchor=False)
        for idx in range(0, _N * ppt, 3):
            traj = idx // ppt
            y_i, dt_norm, y_j, _g = ds[idx]
            i = _time_index(ds, traj, y_i)
            for k in range(y_j.shape[0]):
                j = _time_index(ds, traj, y_j[k])
                want = float(ds.dt_table[i, j].item())
                self.assertAlmostEqual(float(dt_norm[k].item()), want, places=6)


class DtTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_table_covers_the_full_upper_triangle(self) -> None:
        _write_processed(self.root)
        ds = _dataset(self.root)
        self.assertEqual(tuple(ds.dt_table.shape), (_T, _T))

        rng_log = _DT_LOG_MAX - _DT_LOG_MIN
        for i in range(_T):
            for j in range(i + 1, _T):
                want = (math.log10(float(_T64[j] - _T64[i])) - _DT_LOG_MIN) / rng_log
                self.assertAlmostEqual(float(ds.dt_table[i, j].item()), want, places=6)
        # The smallest reachable dt is a mid-grid increment, not t[1]-t[0].
        self.assertEqual(float(ds.dt_table[1, 2].item()), 0.0)
        self.assertLess(float(ds.dt_table[1, 2].item()), float(ds.dt_table[0, 1].item()))

    def test_ulp_below_manifest_bound_is_snapped(self) -> None:
        """A dt one float ULP under the bound is rounding, not extrapolation."""
        # 1.2e-7 in log space is ~1e-8 normalized, the magnitude the float32
        # time grid produces against a bound computed from float64 raw times.
        _write_processed(self.root, dt_log_min_offset=1.2e-7)
        ds = _dataset(self.root)
        self.assertEqual(float(ds.dt_table.min().item()), 0.0)

    def test_real_extrapolation_still_raises(self) -> None:
        _write_processed(self.root, dt_log_min_offset=0.5)
        with self.assertRaises(ValueError) as ctx:
            _dataset(self.root)
        self.assertIn("out of range", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
