"""Unit tests for the latent-linear flow-map architecture.

Covers the LatentLinearFlowMap class contract (forward shapes,
K-vectorization, dt-range enforcement, closed-form decay monotonicity,
seeded init determinism) and the create_model dispatch / config validation
for model.architecture="latent_linear".

The create_model tests build a tiny synthetic normalization manifest in a
temp dir so they do not depend on the (multi-GB) processed dataset.

Scope note: latent_linear is the successor architecture, not the published
model. The paper's FlowMapAutoencoder path is covered by
unit_tests/test_paper_model.py.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.model import LatentLinearFlowMap, create_model


# Manifest dt bounds matching the production dataset (log10 seconds).
_DT_LOG_MIN = -3.530287171715986
_DT_LOG_MAX = 8.0


def _build_model(**overrides) -> LatentLinearFlowMap:
    """Construct a small latent-linear model with sane test defaults."""
    kwargs = dict(
        state_dim=12,
        global_dim=2,
        latent_dim=16,
        encoder_hidden=[32, 32],
        decoder_mode="mlp",
        decoder_hidden=[32],
        dt_log_min=_DT_LOG_MIN,
        dt_log_max=_DT_LOG_MAX,
        predict_delta=True,
    )
    kwargs.update(overrides)
    return LatentLinearFlowMap(**kwargs)


class LatentLinearForwardTests(unittest.TestCase):
    def test_forward_shape_and_finite(self) -> None:
        m = _build_model().eval()
        y = torch.rand(4, 12)
        dt = torch.rand(4, 3)
        g = torch.rand(4, 2)
        out = m(y, dt, g)
        self.assertEqual(tuple(out.shape), (4, 3, 12))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_k_vectorization_equivalence(self) -> None:
        """A forward over K targets must equal K independent K=1 calls."""
        m = _build_model().eval()
        y = torch.rand(4, 12)
        dt = torch.rand(4, 3)
        g = torch.rand(4, 2)
        with torch.no_grad():
            full = m(y, dt, g)  # [4,3,12]
            singles = torch.stack(
                [m(y, dt[:, k : k + 1], g)[:, 0, :] for k in range(3)], dim=1
            )
        self.assertTrue(torch.allclose(full, singles, atol=1e-5))

    def test_dt_norm_out_of_range_raises(self) -> None:
        m = _build_model().eval()
        y = torch.rand(4, 12)
        g = torch.rand(4, 2)
        for bad in (torch.full((4, 1), 1.5), torch.full((4, 1), -0.1)):
            with self.assertRaises((RuntimeError, ValueError)):
                m(y, bad, g)

    def test_global_dim_zero(self) -> None:
        m = _build_model(global_dim=0).eval()
        y = torch.rand(4, 12)
        dt = torch.rand(4, 3)
        g = torch.empty(4, 0)
        out = m(y, dt, g)
        self.assertEqual(tuple(out.shape), (4, 3, 12))

    def test_decay_monotonic_toward_equilibrium(self) -> None:
        """For a fixed anchor state, the latent h(dt) must move monotonically
        toward the equilibrium latent as dt increases (closed-form relaxation:
        ||h(dt) - h_eq|| = decay(dt) * ||h0 - h_eq|| with decay non-increasing).
        """
        m = _build_model().eval()
        y = torch.rand(1, 12)
        g = torch.rand(1, 2)
        with torch.no_grad():
            h0, h_eq, log10_k = m._encode_heads(y, g)
            dt_grid = torch.linspace(0.0, 1.0, 25).view(1, -1)  # [1, 25]
            h_dt = m._latent_state(h0, h_eq, log10_k, dt_grid)  # [1,25,L]
            dist = (h_dt - h_eq.unsqueeze(1)).norm(dim=-1).view(-1)  # [25]
        diffs = dist[1:] - dist[:-1]
        # Non-increasing (allow tiny fp slack).
        self.assertTrue(bool((diffs <= 1e-5).all()), f"distances not monotone: {dist}")

    def test_init_seed_determinism(self) -> None:
        """Same init_seed => identical rate-head biases; different => different."""
        a = _build_model(init_seed=7)
        b = _build_model(init_seed=7)
        c = _build_model(init_seed=8)
        self.assertTrue(torch.equal(a.head_rate.bias, b.head_rate.bias))
        self.assertFalse(torch.equal(a.head_rate.bias, c.head_rate.bias))

    def test_rate_init_spread(self) -> None:
        """Rate-head bias must span the configured U(low, high) range."""
        m = _build_model(latent_dim=256, rate_init=[-5.0, 5.0], init_seed=0)
        bias = m.head_rate.bias.detach()
        self.assertGreaterEqual(float(bias.min()), -5.0)
        self.assertLessEqual(float(bias.max()), 5.0)
        # With 256 modes the draw should populate a wide span (not collapsed).
        self.assertGreater(float(bias.max() - bias.min()), 6.0)

    def test_rate_clamp_keeps_large_kdt_finite(self) -> None:
        """Even at the largest dt with fast modes, decay path stays finite."""
        m = _build_model(rate_init=[4.0, 5.0]).eval()  # fast modes
        y = torch.rand(8, 12)
        dt = torch.ones(8, 1)  # dt_norm=1 => log10(dt_phys)=dt_log_max
        g = torch.rand(8, 2)
        out = m(y, dt, g)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_constructor_rejects_linear_with_hidden(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(decoder_mode="linear", decoder_hidden=[32])

    def test_constructor_rejects_mlp_without_hidden(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(decoder_mode="mlp", decoder_hidden=[])

    def test_constructor_rejects_empty_encoder(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(encoder_hidden=[])

    def test_constructor_rejects_bad_dt_bounds(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(dt_log_min=8.0, dt_log_max=8.0)

    def test_linear_decoder_forward(self) -> None:
        m = _build_model(decoder_mode="linear", decoder_hidden=[], latent_dim=64).eval()
        y = torch.rand(4, 12)
        dt = torch.rand(4, 3)
        g = torch.rand(4, 2)
        out = m(y, dt, g)
        self.assertEqual(tuple(out.shape), (4, 3, 12))


# ---------------------------------------------------------------------------
# create_model dispatch + config validation (synthetic manifest)
# ---------------------------------------------------------------------------

_SPECIES = ["A_evolution", "B_evolution", "C_evolution"]
_GLOBALS = ["P", "T"]


def _write_manifest(processed_dir: Path) -> None:
    """Write a minimal valid normalization.json with a dt spec."""
    per_key = {}
    for s in _SPECIES:
        per_key[s] = {"log_mean": 0.0, "log_std": 1.0}
    per_key["P"] = {"log_min": -6.0, "log_max": 4.0}
    per_key["T"] = {"mean": 1500.0, "std": 500.0}
    manifest = {
        "per_key_stats": per_key,
        "normalization_methods": {
            **{s: "log-standard" for s in _SPECIES},
            "P": "log-min-max",
            "T": "standard",
        },
        "dt": {"log_min": _DT_LOG_MIN, "log_max": _DT_LOG_MAX},
        "epsilon": 1e-30,
        "min_std": 1e-10,
        "meta": {
            "species_variables": _SPECIES,
            "global_variables": _GLOBALS,
            "time_variable": "t_time",
        },
    }
    (processed_dir / "normalization.json").write_text(json.dumps(manifest), encoding="utf-8")


def _base_latent_linear_cfg(processed_dir: Path) -> dict:
    return {
        "paths": {"processed_data_dir": str(processed_dir)},
        "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS)},
        "normalization": {"min_std": 1e-10},
        "model": {
            "architecture": "latent_linear",
            "latent_dim": 16,
            "encoder_hidden": [32],
            "decoder_mode": "mlp",
            "decoder_hidden": [32],
            "rate_clamp": 3.0,
            "rate_init": [-5.0, 5.0],
            "init_seed": 0,
            "activation": "silu",
            "dropout": 0.0,
            "predict_delta": True,
            "predict_delta_log_phys": False,
            "softmax_head": False,
        },
    }


class CreateModelLatentLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.processed = Path(self._tmp.name)
        _write_manifest(self.processed)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_latent_linear(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        m = create_model(cfg)
        self.assertIsInstance(m, LatentLinearFlowMap)
        # dt bounds must be read from the manifest, not hardcoded.
        self.assertAlmostEqual(float(m.dt_log_min), _DT_LOG_MIN, places=6)
        self.assertAlmostEqual(
            float(m.dt_log_min) + float(m.dt_log_range), _DT_LOG_MAX, places=6
        )

    def test_architecture_and_mlp_only_together_raises(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["mlp_only"] = False
        with self.assertRaises(KeyError):
            create_model(cfg)

    def test_missing_latent_linear_key_raises(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        del cfg["model"]["rate_clamp"]
        with self.assertRaises(KeyError):
            create_model(cfg)

    def test_linear_decoder_with_hidden_raises(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["decoder_mode"] = "linear"
        cfg["model"]["decoder_hidden"] = [32]
        with self.assertRaises(ValueError):
            create_model(cfg)

    def test_forbidden_dynamics_keys_raise(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["dynamics_hidden"] = [32]
        with self.assertRaises(KeyError):
            create_model(cfg)

    def test_unknown_architecture_raises(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["architecture"] = "transformer"
        with self.assertRaises(ValueError):
            create_model(cfg)

    def test_softmax_head_with_residual_raises(self) -> None:
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["softmax_head"] = True
        # predict_delta still True => mutual-exclusivity error.
        with self.assertRaises(ValueError):
            create_model(cfg)

    def test_legacy_mlp_only_still_dispatches(self) -> None:
        """A config without architecture but with mlp_only must still work."""
        cfg = {
            "paths": {"processed_data_dir": str(self.processed)},
            "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS)},
            "normalization": {"min_std": 1e-10},
            "model": {
                "mlp_only": True,
                "latent_dim": 16,
                "encoder_hidden": [32],
                "dynamics_hidden": [32],
                "decoder_hidden": [32],
                "dynamics_residual": True,
                "activation": "silu",
                "dropout": 0.0,
                "predict_delta": True,
                "predict_delta_log_phys": False,
                "softmax_head": False,
            },
        }
        m = create_model(cfg)
        self.assertEqual(type(m).__name__, "FlowMapMLP")

    def test_latent_linear_softmax_head_builds(self) -> None:
        """softmax_head (no residual) must build and load species norm buffers."""
        cfg = _base_latent_linear_cfg(self.processed)
        cfg["model"]["softmax_head"] = True
        cfg["model"]["predict_delta"] = False
        m = create_model(cfg)
        y = torch.rand(4, 3)
        dt = torch.rand(4, 2)
        g = torch.rand(4, 2)
        out = m(y, dt, g)
        self.assertEqual(tuple(out.shape), (4, 2, 3))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
