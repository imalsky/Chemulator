#!/usr/bin/env python3
"""Tests for the rollout (autoregressive) training regime.

Covers the wiring for `training.rollout` (the campaign-validated recipe, see
EXPERIMENTS.md / local_ab/RESULTS_2step.md):
  - dataset.py: consecutive-window sampling (shapes, consecutive dt, guards),
  - main.py: validate_rollout_config fail-fast cases,
  - trainer.py: real rollout epochs (detached γ-discount unroll, pushforward
    warm-steps, horizon curriculum, EMA, input-noise off by default),

Self-contained: builds a tiny synthetic processed dir (manifest + summary +
shards) so no real data is needed.

Scope note: this file exercises the successor (latent-linear, autoregressive)
regime, not the published model. The paper's FlowMapAutoencoder path is covered
by unit_tests/test_paper_model.py.
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.dataset import FlowMapPairsDataset, create_dataloader
from src.main import validate_rollout_config
from src.model import create_model
from src.normalizer import NormalizationHelper
from src.trainer import Trainer
from src.utils import resolve_precision_policy

_SPECIES = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution",
]
_GLOBALS = ["P", "T"]
_S, _G = len(_SPECIES), len(_GLOBALS)
_T, _N = 8, 16
_DT_LOG_MIN, _DT_LOG_MAX = -3.0, 8.0
# Strictly increasing grid; consecutive dt = 1,2,4,8,16,32,64.
_T_VEC = np.array([0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0], dtype=np.float64)


def _write_processed(root: Path) -> None:
    """Write a minimal processed dir: manifest + summary + 3 split shards."""
    per_key = {s: {"log_mean": 0.0, "log_std": 1.0} for s in _SPECIES}
    per_key["P"] = {"log_min": -6.0, "log_max": 4.0}
    per_key["T"] = {"mean": 1500.0, "std": 500.0}
    manifest = {
        "per_key_stats": per_key,
        "normalization_methods": {
            **{s: "log-standard" for s in _SPECIES}, "P": "log-min-max", "T": "standard",
        },
        "dt": {"log_min": _DT_LOG_MIN, "log_max": _DT_LOG_MAX},
        "epsilon": 1e-30, "min_std": 1e-10,
        "meta": {"species_variables": _SPECIES, "global_variables": _GLOBALS,
                 "time_variable": "t_time"},
    }
    (root / "normalization.json").write_text(json.dumps(manifest))
    (root / "preprocessing_summary.json").write_text(json.dumps({
        "species_variables": _SPECIES, "global_variables": _GLOBALS,
        "time_variable": "t_time", "time_grid_len": _T,
    }))
    (root / "shard_index.json").write_text(json.dumps({"splits": {"train": 1, "validation": 1, "test": 1}}))

    rng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        y = (10.0 ** rng.uniform(-8.0, -1.0, size=(_N, _T, _S))).astype("float32")
        g = np.stack([10.0 ** rng.uniform(-2.0, 2.0, size=_N),
                      rng.uniform(600.0, 2400.0, size=_N)], axis=1).astype("float32")
        np.savez(d / f"shard_{split}_0.npz", y_mat=y, globals=g, t_vec=_T_VEC.astype("float32"))


def _cfg(root: Path, *, horizon=3, detach=True, gamma=0.9, skip=1, noise=0.0,
         cstart=1, ramp=0, ema=False, ema_decay=0.5, use_first_anchor=False) -> dict:
    cfg = {
        "paths": {"processed_data_dir": str(root), "work_dir": str(root / "work"),
                  "raw_data_files": ["x.h5"], "overwrite": True},
        "system": {"seed": 0, "deterministic": False, "cudnn_benchmark": False, "omp_num_threads": 1},
        "precision": {"amp": "off", "dataset_dtype": "float32", "io_dtype": "float32",
                      "time_io_dtype": "float32", "normalize_dtype": "float32", "tf32": False},
        "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS),
                 "time_variable": "t_time"},
        "normalization": {"min_std": 1e-10},
        "dataset": {"batch_size_train": 8, "multi_time_per_anchor": False, "times_per_anchor": 1,
                    "use_first_anchor": use_first_anchor, "num_workers": 0,
                    "persistent_workers": False, "pin_memory": False, "prefetch_factor": 2,
                    "preload_to_gpu": False},
        "model": {"architecture": "latent_linear", "latent_dim": 8, "encoder_hidden": [16],
                  "decoder_mode": "mlp", "decoder_hidden": [16], "rate_clamp": 3.0,
                  "rate_init": [-5.0, 5.0], "init_seed": 0, "activation": "silu", "dropout": 0.0,
                  "predict_delta": True, "predict_delta_log_phys": False, "softmax_head": False},
        "training": {
            "epochs": 1, "gradient_clip": 1.0, "optimizer": "adamw", "torch_compile": False,
            "lr": 3e-4, "min_lr": 5e-7, "warmup_epochs": 0, "warmup_start_factor": 0.1,
            "weight_decay": 1e-4, "max_train_steps_per_epoch": 0, "max_val_batches": 0,
            "resume": None, "min_steps": 1, "max_steps": 1, "pairs_per_traj": 4,
            "rollout": {"enabled": True, "horizon": horizon, "detach_intermediate": detach,
                        "discount_gamma": gamma, "pushforward_skip": skip,
                        "input_noise_std": noise, "curriculum_start": cstart,
                        "curriculum_ramp_epochs": ramp},
            "adaptive_stiff_loss": {"lambda_phys": 1.0, "lambda_z": 0.0, "use_weighting": False,
                                    "weight_power": 0.5, "w_min": 0.5, "w_max": 2.0},
        },
    }
    if ema:
        cfg["training"]["ema"] = {"enabled": True, "decay": ema_decay}
    return cfg


def _dataset(root: Path, cfg: dict, split="train") -> FlowMapPairsDataset:
    return FlowMapPairsDataset(
        processed_root=root, split=split, config=cfg, pairs_per_traj=4,
        min_steps=1, max_steps=1, preload_to_gpu=False, device=torch.device("cpu"),
        dtype=torch.float32, seed=0, logger=logging.getLogger("t"),
    )


def _make_trainer(root: Path, cfg: dict) -> Trainer:
    device = torch.device("cpu")
    work = Path(cfg["paths"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    loaders = {s: create_dataloader(dataset=_dataset(root, cfg, sp), batch_size=8,
                                    shuffle=(s == "train"), num_workers=0,
                                    persistent_workers=False, pin_memory=False, prefetch_factor=2)
               for s, sp in (("train", "train"), ("val", "validation"))}
    model = create_model(cfg)
    policy = resolve_precision_policy(cfg, device)
    return Trainer(model=model, train_loader=loaders["train"], val_loader=loaders["val"],
                   cfg=cfg, work_dir=work, device=device, logger=logging.getLogger("tr"),
                   precision_policy=policy)


class RolloutDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_processed(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_window_shapes(self) -> None:
        ds = _dataset(self.root, _cfg(self.root, horizon=3))
        y0, dt_steps, y_tgt, g = ds[0]
        self.assertEqual(tuple(y0.shape), (_S,))
        self.assertEqual(tuple(dt_steps.shape), (3,))
        self.assertEqual(tuple(y_tgt.shape), (3, _S))
        self.assertEqual(tuple(g.shape), (_G,))

    def test_consecutive_dt_matches_grid(self) -> None:
        ds = _dataset(self.root, _cfg(self.root, horizon=3))
        helper = NormalizationHelper(json.loads((self.root / "normalization.json").read_text()))
        t = torch.from_numpy(_T_VEC)
        want = helper.normalize_dt_from_phys(t[1:] - t[:-1]).clamp_(0.0, 1.0).to(torch.float32)
        torch.testing.assert_close(ds.dt_consec, want)

    def test_use_first_anchor_true_raises(self) -> None:
        with self.assertRaises(ValueError):
            _dataset(self.root, _cfg(self.root, use_first_anchor=True))

    def test_horizon_too_long_raises(self) -> None:
        with self.assertRaises(ValueError):
            _dataset(self.root, _cfg(self.root, horizon=_T, cstart=1))

    def test_disabled_falls_back_to_one_step(self) -> None:
        cfg = _cfg(self.root, horizon=3)
        cfg["training"]["rollout"]["enabled"] = False
        cfg["dataset"]["use_first_anchor"] = True
        ds = _dataset(self.root, cfg)
        y_i, dt_norm, y_j, g = ds[0]
        self.assertEqual(tuple(y_i.shape), (_S,))
        self.assertEqual(tuple(y_j.shape)[-1], _S)
        self.assertEqual(tuple(g.shape), (_G,))

        # use_first_anchor with min_steps=max_steps=1 pins the pair to (t0, t1),
        # so the sample must be that exact state pair with that exact dt.
        helper = NormalizationHelper(json.loads((self.root / "normalization.json").read_text()))
        want_dt = helper.normalize_dt_from_phys(
            torch.from_numpy(_T_VEC[1:2] - _T_VEC[0:1])
        ).to(torch.float32)
        torch.testing.assert_close(dt_norm, want_dt)
        torch.testing.assert_close(y_i, ds.y[0, 0])
        torch.testing.assert_close(y_j, ds.y[0, 1:2])


class ValidateRolloutConfigTests(unittest.TestCase):
    def _base(self) -> dict:
        return {"training": {"rollout": {"enabled": True, "horizon": 10,
                "detach_intermediate": True, "discount_gamma": 0.9, "pushforward_skip": 2,
                "input_noise_std": 0.0, "curriculum_start": 1, "curriculum_ramp_epochs": 30}},
                "dataset": {"use_first_anchor": False}}

    def test_ok(self) -> None:
        validate_rollout_config(self._base())

    def test_absent_block_ok(self) -> None:
        validate_rollout_config({"training": {}})

    def test_disabled_ok_without_keys(self) -> None:
        validate_rollout_config({"training": {"rollout": {"enabled": False}}})

    def test_missing_key_raises(self) -> None:
        c = self._base()
        del c["training"]["rollout"]["discount_gamma"]
        with self.assertRaises(KeyError):
            validate_rollout_config(c)

    def test_bad_gamma_raises(self) -> None:
        c = self._base()
        c["training"]["rollout"]["discount_gamma"] = 1.5
        with self.assertRaises(ValueError):
            validate_rollout_config(c)

    def test_curriculum_start_out_of_range_raises(self) -> None:
        c = self._base()
        c["training"]["rollout"]["curriculum_start"] = 11  # > horizon
        with self.assertRaises(ValueError):
            validate_rollout_config(c)

    def test_use_first_anchor_true_raises(self) -> None:
        c = self._base()
        c["dataset"]["use_first_anchor"] = True
        with self.assertRaises(ValueError):
            validate_rollout_config(c)

    def test_negative_noise_raises(self) -> None:
        c = self._base()
        c["training"]["rollout"]["input_noise_std"] = -0.1
        with self.assertRaises(ValueError):
            validate_rollout_config(c)


class RolloutCurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_processed(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_horizon_ramps(self) -> None:
        tr = _make_trainer(self.root, _cfg(self.root, horizon=5, cstart=1, ramp=10))
        self.assertEqual(tr._current_horizon(0), 1)     # start
        self.assertEqual(tr._current_horizon(10), 5)    # fully ramped
        self.assertEqual(tr._current_horizon(100), 5)   # clamped at K
        mid = tr._current_horizon(5)
        self.assertTrue(1 <= mid <= 5)

    def test_no_curriculum_is_constant(self) -> None:
        tr = _make_trainer(self.root, _cfg(self.root, horizon=4, cstart=1, ramp=0))
        self.assertEqual(tr._current_horizon(0), 4)
        self.assertEqual(tr._current_horizon(50), 4)


class RolloutTrainerEpochTests(unittest.TestCase):
    """End-to-end: real Trainer rollout epochs run and checkpoint."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_processed(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, **kw) -> Trainer:
        tr = _make_trainer(self.root, _cfg(self.root, **kw))
        best = tr.train()
        self.assertTrue(np.isfinite(best))
        return tr

    def test_pushforward_detached_runs(self) -> None:
        self._run(horizon=3, detach=True, gamma=0.9, skip=1, noise=0.0)
        self.assertTrue((Path(self.root) / "work" / "last.ckpt").is_file())

    def test_bptt_runs(self) -> None:
        self._run(horizon=3, detach=False, gamma=1.0, skip=0, noise=0.0)

    def test_curriculum_epoch_runs(self) -> None:
        cfg = _cfg(self.root, horizon=4, cstart=1, ramp=3)
        cfg["training"]["epochs"] = 2
        tr = _make_trainer(self.root, cfg)
        self.assertTrue(np.isfinite(tr.train()))

    def test_ema_best_differs_from_raw(self) -> None:
        """With EMA on, best.ckpt holds EMA weights ≠ last.ckpt raw weights."""
        cfg = _cfg(self.root, horizon=3, ema=True, ema_decay=0.5)
        cfg["training"]["epochs"] = 2
        tr = _make_trainer(self.root, cfg)
        tr.train()
        self.assertIsNotNone(tr.ema_shadow)
        work = Path(self.root) / "work"
        best = torch.load(work / "best.ckpt", map_location="cpu", weights_only=False)["model_state"]
        last = torch.load(work / "last.ckpt", map_location="cpu", weights_only=False)["model_state"]
        # At least one trainable parameter should differ (EMA lags the raw weights).
        diffs = [float((best[k] - last[k]).abs().max()) for k in best
                 if best[k].dtype.is_floating_point and k in last]
        self.assertTrue(max(diffs) > 0.0)


if __name__ == "__main__":
    unittest.main()
