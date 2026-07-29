#!/usr/bin/env python3
"""Unit tests for the published (paper) model path.

The published emulator is the residual encoder / latent-operator / decoder
``FlowMapAutoencoder``, trained with the log-ratio + 0.5 * MSE-z objective and
shipped through ``PhysicalModelWrapper``. The other unit-test files cover the
later latent-linear and rollout work, so the assertions here are written
against the manuscript equations rather than against the implementation.
Equations are cited by LaTeX label, not by number, because the numbering
shifts as the manuscript is revised:

  - AdaptiveStiffLoss vs eq:loss (log-ratio + 0.5 * MSE-z), including the
    reduction order over B, K and N_s,
  - NormalizationHelper physical -> z -> physical round trip,
  - FlowMapAutoencoder forward vs eq:flowmap_noresiduals and
    eq:flowmap_both_residuals, composed by hand,
  - export/train equivalence: PhysicalModelWrapper and the exported program
    must reproduce the training forward pass,
  - the released artifacts under models/final_model (skipped when absent: the
    small text records of that run are tracked, but the checkpoints and the
    exported binary these tests need ship in the Zenodo record).

Self-contained: the synthetic manifest mirrors the production one (12
log-standard species, log-min-max P, standard T), so no processed data is
required for anything except the released-artifact tests.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

from src.model import FlowMapAutoencoder, create_model
from src.normalizer import NormalizationHelper
from src.trainer import AdaptiveStiffLoss
from src.utils import load_json_config
# Export helpers are exercised directly: the exported artifact is the object
# the paper's speed and accuracy numbers were measured on.
from testing.export import PhysicalModelWrapper, _build_group_tensors, _load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reconstructed config for the published run.
_PAPER_CONFIG = REPO_ROOT / "config" / "config_paper.jsonc"

# Released artifacts (gitignored; distributed via Zenodo).
_MODEL_DIR = REPO_ROOT / "models" / "final_model"
_RELEASED_CONFIG = _MODEL_DIR / "config.json"
_RELEASED_CKPT = _MODEL_DIR / "best.ckpt"
_RELEASED_PT2 = _MODEL_DIR / "physical_model_k1_cpu.pt2"
_RELEASED_META = _MODEL_DIR / "physical_model_metadata.json"
_PROCESSED_MANIFEST = REPO_ROOT / "data" / "processed" / "normalization.json"

# Parameter count of the published model (paper quotes 11.6M).
_PAPER_PARAM_COUNT = 11_577_356

_SPECIES = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution",
]
_GLOBALS = ["P", "T"]
_S, _G = len(_SPECIES), len(_GLOBALS)

# Production dt band (log10 seconds), matching data/processed/normalization.json.
_DT_LOG_MIN = -3.530287171715986
_DT_LOG_MAX = 8.0

# Per-species log10 stats: wide, uneven ranges like the real manifest.
_LOG_MEAN = [-9.0, -4.5, -8.0, -5.0, -3.5, -0.1, -11.0, -7.0, -1.2, -6.5, -13.0, -12.0]
_LOG_STD = [4.0, 2.5, 3.5, 2.0, 1.5, 0.2, 5.0, 3.0, 0.5, 2.8, 5.5, 4.5]


def _species_stats() -> dict:
    """Per-key stats for the 12 log-standard species (log_min/log_max included
    so the optional loss weighting can be built)."""
    out = {}
    for name, lm, ls in zip(_SPECIES, _LOG_MEAN, _LOG_STD):
        out[name] = {
            "log_mean": lm,
            "log_std": ls,
            "log_min": lm - 3.0 * ls,
            "log_max": lm + 3.0 * ls,
        }
    return out


def _paper_manifest() -> dict:
    """Synthetic manifest shaped like the production one."""
    per_key = _species_stats()
    per_key["P"] = {"log_min": -6.0, "log_max": 4.0}
    per_key["T"] = {"mean": 1500.0, "std": 500.0}
    return {
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
            "species_variables": list(_SPECIES),
            "global_variables": list(_GLOBALS),
            "time_variable": "t_time",
        },
    }


def _build_loss(
    *,
    lambda_phys: float = 1.0,
    lambda_z: float = 0.5,
    use_weighting: bool = False,
    weight_power: float = 0.5,
    w_min: float = 0.5,
    w_max: float = 2.0,
) -> AdaptiveStiffLoss:
    """AdaptiveStiffLoss with the published weights (1.0, 0.5)."""
    manifest = _paper_manifest()
    stats = manifest["per_key_stats"]
    log_min = log_max = None
    if use_weighting:
        log_min = torch.tensor([float(stats[s]["log_min"]) for s in _SPECIES])
        log_max = torch.tensor([float(stats[s]["log_max"]) for s in _SPECIES])
    return AdaptiveStiffLoss(
        norm_helper=NormalizationHelper(manifest),
        species_keys=list(_SPECIES),
        species_log_min=log_min,
        species_log_max=log_max,
        lambda_phys=lambda_phys,
        lambda_z=lambda_z,
        use_weighting=use_weighting,
        weight_power=weight_power,
        w_min=w_min,
        w_max=w_max,
    )


# ---------------------------------------------------------------------------
# eq:loss -- L = L_log-ratio + 0.5 * L_mse
# ---------------------------------------------------------------------------


class PaperLossTests(unittest.TestCase):
    """AdaptiveStiffLoss against the manuscript loss written out by hand."""

    @staticmethod
    def _reference_total(pred_z: torch.Tensor, true_z: torch.Tensor, lambda_z: float) -> float:
        """eq:loss from the manuscript, in float64, from physical abundances.

        a_hat / a are recovered from z with the log-standard stats, so this
        path never reuses the loss module's own conversion code.
        """
        lm = torch.tensor(_LOG_MEAN, dtype=torch.float64)
        ls = torch.tensor(_LOG_STD, dtype=torch.float64)
        a_hat = torch.pow(10.0, pred_z.to(torch.float64) * ls + lm)
        a = torch.pow(10.0, true_z.to(torch.float64) * ls + lm)
        l_log_ratio = torch.log10(a_hat / a).abs().mean()
        x_hat = (torch.log10(a_hat) - lm) / ls
        x = (torch.log10(a) - lm) / ls
        l_mse = ((x_hat - x) ** 2).mean()
        return float(l_log_ratio + lambda_z * l_mse)

    def test_matches_paper_equation(self) -> None:
        loss = _build_loss()
        gen = torch.Generator().manual_seed(0)
        pred = torch.randn(6, 4, _S, generator=gen)
        true = torch.randn(6, 4, _S, generator=gen)
        got = float(loss(pred, true))
        want = self._reference_total(pred, true, 0.5)
        self.assertAlmostEqual(got, want, delta=1e-5 * max(1.0, abs(want)))

    def test_reduction_is_mean_over_b_k_and_species(self) -> None:
        """A single perturbed scalar pins the denominator to B * K * N_s.

        If the loss summed over K (or averaged over species only) this value
        would be off by exactly a factor of K, B or B*K.
        """
        loss = _build_loss()
        B, K = 3, 5
        true = torch.zeros(B, K, _S)
        pred = true.clone()
        delta = 0.25
        pred[1, 2, 3] = delta

        n = B * K * _S
        want = (abs(delta) * _LOG_STD[3]) / n + 0.5 * (delta ** 2) / n
        self.assertAlmostEqual(float(loss(pred, true)), want, places=9)

    def test_batch_loss_equals_mean_of_per_target_losses(self) -> None:
        """eq:loss is stated for one target; the batch loss is its mean over
        the B * K targets."""
        loss = _build_loss()
        gen = torch.Generator().manual_seed(1)
        pred = torch.randn(4, 3, _S, generator=gen)
        true = torch.randn(4, 3, _S, generator=gen)

        total = float(loss(pred, true))
        singles = [
            float(loss(pred[b:b + 1, k:k + 1], true[b:b + 1, k:k + 1]))
            for b in range(4)
            for k in range(3)
        ]
        self.assertAlmostEqual(total, sum(singles) / len(singles), places=6)

    def test_components_use_published_weights(self) -> None:
        loss = _build_loss(lambda_phys=1.0, lambda_z=0.5)
        gen = torch.Generator().manual_seed(2)
        pred = torch.randn(2, 2, _S, generator=gen)
        true = torch.randn(2, 2, _S, generator=gen)
        comps = loss(pred, true, return_components=True)

        mse_z = float(((pred - true) ** 2).mean())
        self.assertAlmostEqual(float(comps["z"]), 0.5 * mse_z, places=6)
        # lambda_phys = 1.0 and weighting is off, so the log-ratio term is the
        # unweighted mean |log10| error.
        self.assertAlmostEqual(float(comps["phys"]), float(comps["mean_abs_log10"]), places=9)
        self.assertAlmostEqual(
            float(comps["total"]), float(comps["phys"]) + float(comps["z"]), places=9
        )

    def test_weighting_off_leaves_weights_at_one(self) -> None:
        """The published run has use_weighting=false, so every species carries
        weight 1."""
        loss = _build_loss(use_weighting=False)
        self.assertTrue(torch.equal(loss.w_species, torch.ones(_S)))
        gen = torch.Generator().manual_seed(3)
        pred = torch.randn(2, 3, _S, generator=gen)
        true = torch.randn(2, 3, _S, generator=gen)
        comps = loss(pred, true, return_components=True)
        want = float((pred - true).abs().mul(torch.tensor(_LOG_STD)).mean())
        self.assertAlmostEqual(float(comps["mean_abs_log10"]), want, places=6)

    def test_weighting_on_follows_sqrt_range_formula(self) -> None:
        """w_i proportional to sqrt(log range), normalized to unit mean."""
        stats = _species_stats()
        rng = torch.tensor([stats[s]["log_max"] - stats[s]["log_min"] for s in _SPECIES])
        want = torch.sqrt(rng)
        want = want / want.mean()
        # Widen the allowed band so the (deliberately uneven) synthetic ranges
        # do not trip the fail-fast bounds check.
        loss = _build_loss(use_weighting=True, w_min=0.1, w_max=10.0)
        torch.testing.assert_close(loss.w_species, want)

    def test_weights_outside_bounds_are_a_hard_error(self) -> None:
        with self.assertRaises(ValueError):
            _build_loss(use_weighting=True, w_min=0.99, w_max=1.01)

    def test_shape_mismatch_raises(self) -> None:
        loss = _build_loss()
        with self.assertRaises(ValueError):
            loss(torch.zeros(2, 3, _S), torch.zeros(2, 4, _S))


# ---------------------------------------------------------------------------
# Normalization round trip
# ---------------------------------------------------------------------------


class NormalizerRoundTripTests(unittest.TestCase):
    """physical -> z -> physical, to tight tolerance, on every method."""

    def setUp(self) -> None:
        self.helper = NormalizationHelper(_paper_manifest())

    def test_species_round_trip_float64(self) -> None:
        """The 12 log-standard species (single-method fast path)."""
        lm = torch.tensor(_LOG_MEAN, dtype=torch.float64)
        ls = torch.tensor(_LOG_STD, dtype=torch.float64)
        gen = torch.Generator().manual_seed(0)
        # Cover +-3 sigma in log space, i.e. tens of decades of dynamic range.
        u = torch.rand(64, _S, generator=gen, dtype=torch.float64) * 6.0 - 3.0
        y = torch.pow(10.0, lm + ls * u)

        z = self.helper.normalize(y, _SPECIES)
        back = self.helper.denormalize(z, _SPECIES)
        rel = ((back - y).abs() / y).max()
        self.assertLess(float(rel), 1e-12)
        # z must be the standardized log10 abundance x_i of eq:loss.
        torch.testing.assert_close(z, u, rtol=1e-12, atol=1e-12)

    def test_species_round_trip_float32(self) -> None:
        lm = torch.tensor(_LOG_MEAN)
        ls = torch.tensor(_LOG_STD)
        gen = torch.Generator().manual_seed(1)
        u = torch.rand(64, _S, generator=gen) * 6.0 - 3.0
        y = torch.pow(10.0, lm + ls * u)

        back = self.helper.denormalize(self.helper.normalize(y, _SPECIES), _SPECIES)
        rel = ((back - y).abs() / y).max()
        self.assertLess(float(rel), 1e-4)

    def test_mixed_method_round_trip(self) -> None:
        """P (log-min-max) and T (standard) go through the per-column path."""
        gen = torch.Generator().manual_seed(2)
        p = torch.pow(10.0, torch.rand(128, 1, generator=gen, dtype=torch.float64) * 10.0 - 6.0)
        t = 600.0 + 1800.0 * torch.rand(128, 1, generator=gen, dtype=torch.float64)
        g = torch.cat([p, t], dim=1)

        back = self.helper.denormalize(self.helper.normalize(g, _GLOBALS), _GLOBALS)
        rel = ((back - g).abs() / g).max()
        self.assertLess(float(rel), 1e-12)

    def test_all_four_methods_round_trip(self) -> None:
        manifest = _paper_manifest()
        manifest["per_key_stats"]["A"] = {"mean": 0.5, "std": 0.25}
        manifest["per_key_stats"]["B"] = {"min": -2.0, "max": 7.0}
        manifest["per_key_stats"]["C"] = {"log_mean": -8.0, "log_std": 3.0}
        manifest["per_key_stats"]["D"] = {"log_min": -20.0, "log_max": 2.0}
        manifest["normalization_methods"].update(
            {"A": "standard", "B": "min-max", "C": "log-standard", "D": "log-min-max"}
        )
        helper = NormalizationHelper(manifest)
        keys = ["A", "B", "C", "D"]

        gen = torch.Generator().manual_seed(3)
        a = torch.randn(256, 1, generator=gen, dtype=torch.float64)
        b = -2.0 + 9.0 * torch.rand(256, 1, generator=gen, dtype=torch.float64)
        c = torch.pow(10.0, -20.0 + 20.0 * torch.rand(256, 1, generator=gen, dtype=torch.float64))
        d = torch.pow(10.0, -20.0 + 22.0 * torch.rand(256, 1, generator=gen, dtype=torch.float64))
        x = torch.cat([a, b, c, d], dim=1)

        back = helper.denormalize(helper.normalize(x, keys), keys)
        rel = ((back - x).abs() / x.abs()).max()
        self.assertLess(float(rel), 1e-12)

    def test_dt_round_trip(self) -> None:
        gen = torch.Generator().manual_seed(4)
        u = torch.rand(256, generator=gen, dtype=torch.float64)
        dt = torch.pow(10.0, _DT_LOG_MIN + u * (_DT_LOG_MAX - _DT_LOG_MIN))

        dt_norm = self.helper.normalize_dt_from_phys(dt)
        self.assertGreaterEqual(float(dt_norm.min()), 0.0)
        self.assertLessEqual(float(dt_norm.max()), 1.0)
        back = self.helper.denormalize_dt_to_phys(dt_norm)
        self.assertLess(float(((back - dt).abs() / dt).max()), 1e-12)

    def test_non_positive_species_value_is_a_hard_error(self) -> None:
        y = torch.full((2, _S), 1e-6)
        y[0, 3] = 0.0
        with self.assertRaises(ValueError):
            self.helper.normalize(y, _SPECIES)

    def test_dt_out_of_band_is_a_hard_error(self) -> None:
        too_big = torch.full((2,), 10.0 ** (_DT_LOG_MAX + 1.0), dtype=torch.float64)
        with self.assertRaises(ValueError):
            self.helper.validate_dt_norm(self.helper.normalize_dt_from_phys(too_big))


# ---------------------------------------------------------------------------
# eq:flowmap_noresiduals / eq:flowmap_both_residuals -- forward composition
# ---------------------------------------------------------------------------


def _build_autoencoder(**overrides) -> FlowMapAutoencoder:
    kwargs = dict(
        state_dim=_S,
        global_dim=_G,
        latent_dim=16,
        encoder_hidden=[32, 32],
        dynamics_hidden=[32, 32],
        decoder_hidden=[32],
        activation_name="silu",
        dropout=0.0,
        dynamics_residual=True,
        predict_delta=True,
    )
    kwargs.update(overrides)
    return FlowMapAutoencoder(**kwargs)


class FlowMapAutoencoderForwardTests(unittest.TestCase):
    """The published architecture, checked against the manuscript equations."""

    def setUp(self) -> None:
        gen = torch.Generator().manual_seed(0)
        self.y = torch.randn(5, _S, generator=gen)
        self.g = torch.randn(5, _G, generator=gen)
        self.dt = torch.rand(5, 4, generator=gen)

    def test_forward_shape(self) -> None:
        out = _build_autoencoder().eval()(self.y, self.dt, self.g)
        self.assertEqual(tuple(out.shape), (5, 4, _S))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_matches_both_residuals_equation(self) -> None:
        """y_j = y_i + D(E(y_i,g) + F(E(y_i,g), dt, g)) composed by hand."""
        m = _build_autoencoder(dynamics_residual=True, predict_delta=True).eval()
        B, K = self.y.shape[0], self.dt.shape[1]

        with torch.no_grad():
            enc = m.encoder(self.y, self.g)                               # E(y_i, g)
            enc_k = enc.unsqueeze(1).expand(B, K, -1)
            g_k = self.g.unsqueeze(1).expand(B, K, -1)
            dyn_in = torch.cat([enc_k, self.dt.unsqueeze(-1), g_k], dim=-1)
            latent = enc_k + m.dynamics.network(dyn_in)                   # E + F(E, dt, g)
            want = self.y.unsqueeze(1) + m.decoder(latent)                # y_i + D(...)
            got = m(self.y, self.dt, self.g)

        torch.testing.assert_close(got, want)

    def test_matches_no_residuals_equation(self) -> None:
        """y_j = D(F(E(y_i,g), dt, g)) when both residual paths are off."""
        m = _build_autoencoder(dynamics_residual=False, predict_delta=False).eval()
        B, K = self.y.shape[0], self.dt.shape[1]

        with torch.no_grad():
            enc = m.encoder(self.y, self.g)
            enc_k = enc.unsqueeze(1).expand(B, K, -1)
            g_k = self.g.unsqueeze(1).expand(B, K, -1)
            dyn_in = torch.cat([enc_k, self.dt.unsqueeze(-1), g_k], dim=-1)
            want = m.decoder(m.dynamics.network(dyn_in))
            got = m(self.y, self.dt, self.g)

        torch.testing.assert_close(got, want)

    def test_residual_flags_change_the_output(self) -> None:
        """Guards against a wiring bug that silently ignores a residual flag."""
        m_both = _build_autoencoder(dynamics_residual=True, predict_delta=True).eval()
        m_none = _build_autoencoder(dynamics_residual=False, predict_delta=False).eval()
        m_none.load_state_dict(m_both.state_dict())
        with torch.no_grad():
            a = m_both(self.y, self.dt, self.g)
            b = m_none(self.y, self.dt, self.g)
        self.assertFalse(bool(torch.allclose(a, b)))

    def test_k_vectorization_equivalence(self) -> None:
        """One call over K targets equals K independent K=1 calls."""
        m = _build_autoencoder().eval()
        with torch.no_grad():
            full = m(self.y, self.dt, self.g)
            singles = torch.stack(
                [m(self.y, self.dt[:, k:k + 1], self.g)[:, 0, :] for k in range(self.dt.shape[1])],
                dim=1,
            )
        torch.testing.assert_close(full, singles)

    def test_prediction_depends_on_dt_and_globals(self) -> None:
        """A flow map that ignored dt (or g) would still produce plausible
        shapes and finite values, so assert both actually reach the output."""
        m = _build_autoencoder().eval()
        with torch.no_grad():
            slow = m(self.y, torch.full((5, 1), 0.1), self.g)
            fast = m(self.y, torch.full((5, 1), 0.9), self.g)
            other_g = m(self.y, torch.full((5, 1), 0.1), self.g + 1.0)
        self.assertFalse(bool(torch.allclose(slow, fast)))
        self.assertFalse(bool(torch.allclose(slow, other_g)))

    def test_dt_out_of_range_raises(self) -> None:
        m = _build_autoencoder().eval()
        for bad in (torch.full((5, 1), 1.5), torch.full((5, 1), -0.25)):
            with self.assertRaises((RuntimeError, ValueError)):
                m(self.y, bad, self.g)

    def test_create_model_builds_the_autoencoder_for_the_legacy_config(self) -> None:
        """The published config has no model.architecture, only mlp_only=false."""
        cfg = {
            "paths": {"processed_data_dir": "data"},
            "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS)},
            "normalization": {"min_std": 1e-10},
            "model": {
                "mlp_only": False,
                "latent_dim": 32,
                "encoder_hidden": [64, 64],
                "dynamics_hidden": [64],
                "decoder_hidden": [64],
                "dynamics_residual": True,
                "activation": "silu",
                "dropout": 0.0,
                "predict_delta": True,
                "predict_delta_log_phys": False,
                "softmax_head": False,
            },
        }
        m = create_model(cfg)
        self.assertIsInstance(m, FlowMapAutoencoder)
        self.assertEqual((m.S, m.G, m.Z), (_S, _G, 32))
        self.assertTrue(m.dynamics.residual)
        self.assertTrue(m.predict_delta)

    def test_softmax_head_needs_a_manifest(self) -> None:
        """Non-residual heads read species stats; a missing manifest is fatal."""
        cfg = {
            "paths": {"processed_data_dir": "does_not_exist"},
            "data": {"species_variables": list(_SPECIES), "global_variables": list(_GLOBALS)},
            "normalization": {"min_std": 1e-10},
            "model": {
                "mlp_only": False,
                "latent_dim": 8,
                "encoder_hidden": [16],
                "dynamics_hidden": [16],
                "decoder_hidden": [16],
                "dynamics_residual": True,
                "activation": "silu",
                "dropout": 0.0,
                "predict_delta": False,
                "predict_delta_log_phys": False,
                "softmax_head": True,
            },
        }
        with self.assertRaises(FileNotFoundError):
            create_model(cfg)


class LogPhysicalHeadTests(unittest.TestCase):
    """The z <-> log10(physical) conversions behind the non-default heads."""

    @staticmethod
    def _species_norm(methods: list[str]) -> dict:
        codes = {"standard": 0, "min-max": 1, "log-standard": 2, "log-min-max": 3}
        n = len(methods)
        return {
            "method_code": [codes[m] for m in methods],
            "mean": [1.0] * n,
            "std": [0.5] * n,
            "min": [0.5] * n,
            "max": [4.0] * n,
            "log_mean": list(_LOG_MEAN[:n]),
            "log_std": list(_LOG_STD[:n]),
            "log_min": [-20.0] * n,
            "log_max": [2.0] * n,
        }

    def test_z_to_log10_phys_round_trip(self) -> None:
        methods = ["log-standard"] * 4 + ["log-min-max"] * 4 + ["standard"] * 2 + ["min-max"] * 2
        m = _build_autoencoder(
            predict_delta=False,
            predict_delta_log_phys=True,
            species_norm=self._species_norm(methods),
            min_std=1e-10,
        ).eval()
        gen = torch.Generator().manual_seed(0)
        z = torch.rand(8, len(methods), generator=gen) * 0.5 + 0.25
        back = m._log10_phys_to_z(m._z_to_log10_phys(z))
        torch.testing.assert_close(back, z, rtol=1e-5, atol=1e-6)

    def test_non_positive_physical_value_is_a_hard_error(self) -> None:
        """Only the standard / min-max branches can produce a non-positive
        abundance; the log branches are affine in log space."""
        methods = ["standard"] * _S
        m = _build_autoencoder(
            predict_delta=False,
            predict_delta_log_phys=True,
            species_norm=self._species_norm(methods),
            min_std=1e-10,
        ).eval()
        # mean=1.0, std=0.5 => z=-4 maps to a physical value of -1.
        with self.assertRaises(ValueError):
            m._z_to_log10_phys(torch.full((2, _S), -4.0))


# ---------------------------------------------------------------------------
# Export / training equivalence
# ---------------------------------------------------------------------------


def _wrap_for_export(base: FlowMapAutoencoder, manifest: dict) -> PhysicalModelWrapper:
    methods = manifest["normalization_methods"]
    stats = manifest["per_key_stats"]
    min_std = float(manifest["min_std"])
    return PhysicalModelWrapper(
        base_model=base,
        species_keys=_SPECIES,
        global_keys=_GLOBALS,
        species_tensors=_build_group_tensors(
            keys=_SPECIES, methods=methods, per_key_stats=stats, min_std=min_std
        ),
        global_tensors=_build_group_tensors(
            keys=_GLOBALS, methods=methods, per_key_stats=stats, min_std=min_std
        ),
        min_std=min_std,
        dt_log_min=float(manifest["dt"]["log_min"]),
        dt_log_max=float(manifest["dt"]["log_max"]),
    ).eval()


def _physical_inputs(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """In-distribution physical inputs (species, dt seconds, globals)."""
    gen = torch.Generator().manual_seed(seed)
    lm = torch.tensor(_LOG_MEAN)
    ls = torch.tensor(_LOG_STD)
    u = torch.rand(n, _S, generator=gen) * 2.0 - 1.0
    y = torch.pow(10.0, lm + 0.5 * ls * u)
    dt = torch.pow(10.0, _DT_LOG_MIN + torch.rand(n, 1, generator=gen) * (_DT_LOG_MAX - _DT_LOG_MIN))
    p = torch.pow(10.0, torch.rand(n, generator=gen) * 6.0 - 4.0)
    t = 800.0 + 1400.0 * torch.rand(n, generator=gen)
    return y, dt, torch.stack([p, t], dim=1)


class ExportEquivalenceTests(unittest.TestCase):
    """The physical-I/O artifact must reproduce the training forward pass."""

    def setUp(self) -> None:
        self.manifest = _paper_manifest()
        self.helper = NormalizationHelper(self.manifest)
        torch.manual_seed(0)
        self.base = _build_autoencoder().eval()
        self.wrapper = _wrap_for_export(self.base, self.manifest)
        self.y, self.dt, self.g = _physical_inputs(6, seed=7)

    def test_wrapper_matches_training_normalization_path(self) -> None:
        """wrapper(phys) == denorm(model(norm(phys))) using the training-side
        NormalizationHelper, i.e. the exported normalization is not a
        re-derivation that could drift from training."""
        with torch.no_grad():
            got = self.wrapper(self.y, self.dt, self.g)
            y_z = self.helper.normalize(self.y, _SPECIES)
            g_z = self.helper.normalize(self.g, _GLOBALS)
            dt_norm = self.helper.normalize_dt_from_phys(self.dt)
            self.helper.validate_dt_norm(dt_norm)
            want = self.helper.denormalize(self.base(y_z, dt_norm, g_z), _SPECIES)

        self.assertTrue(bool(torch.isfinite(got).all()))
        # Compare in log10 space: species span tens of decades.
        torch.testing.assert_close(torch.log10(got), torch.log10(want), rtol=0.0, atol=1e-5)

    def test_exported_program_matches_eager_wrapper(self) -> None:
        batch = torch.export.Dim("batch", min=1, max=4096)
        y2, dt2, g2 = _physical_inputs(2, seed=1)
        exported = torch.export.export(
            self.wrapper,
            (y2, dt2, g2),
            dynamic_shapes=({0: batch}, {0: batch}, {0: batch}),
            strict=False,
        ).module()
        with torch.no_grad():
            got = exported(self.y, self.dt, self.g)
            want = self.wrapper(self.y, self.dt, self.g)
        torch.testing.assert_close(got, want)

    def test_species_order_is_the_configured_order(self) -> None:
        """Permuting the input columns must change the prediction: the wrapper
        does not sort or otherwise reorder species."""
        perm = list(range(_S))
        perm[0], perm[1] = perm[1], perm[0]
        with torch.no_grad():
            straight = self.wrapper(self.y, self.dt, self.g)
            swapped = self.wrapper(self.y[:, perm], self.dt, self.g)
        self.assertFalse(bool(torch.allclose(straight, swapped)))

    def test_dt_outside_the_trained_band_raises(self) -> None:
        for bad_log in (_DT_LOG_MIN - 1.0, _DT_LOG_MAX + 1.0):
            dt = torch.full((6, 1), float(10.0 ** bad_log))
            with self.assertRaises((RuntimeError, ValueError)):
                self.wrapper(self.y, dt, self.g)

    def test_non_positive_physical_input_raises(self) -> None:
        y = self.y.clone()
        y[0, 0] = 0.0
        with self.assertRaises((RuntimeError, ValueError)):
            self.wrapper(y, self.dt, self.g)


# ---------------------------------------------------------------------------
# Repository config for the published run
# ---------------------------------------------------------------------------


@unittest.skipUnless(_PAPER_CONFIG.is_file(), f"missing {_PAPER_CONFIG}")
class PaperConfigTests(unittest.TestCase):
    """config/config_paper.jsonc must rebuild the published model."""

    def setUp(self) -> None:
        self.cfg = load_json_config(_PAPER_CONFIG)

    def test_builds_the_published_architecture_and_size(self) -> None:
        model = create_model(self.cfg)
        self.assertIsInstance(model, FlowMapAutoencoder)
        self.assertEqual(sum(p.numel() for p in model.parameters()), _PAPER_PARAM_COUNT)
        # Both residual paths of eq:flowmap_both_residuals must be on.
        self.assertTrue(model.dynamics.residual)
        self.assertTrue(model.predict_delta)
        self.assertFalse(model.softmax_head)
        self.assertFalse(model.predict_delta_log_phys)

    def test_carries_the_published_loss_weights(self) -> None:
        loss_cfg = self.cfg["training"]["adaptive_stiff_loss"]
        self.assertEqual(float(loss_cfg["lambda_phys"]), 1.0)
        self.assertEqual(float(loss_cfg["lambda_z"]), 0.5)
        self.assertFalse(bool(loss_cfg["use_weighting"]))


# ---------------------------------------------------------------------------
# Released artifacts (models/ is gitignored; distributed via Zenodo)
# ---------------------------------------------------------------------------


_HAVE_RELEASED = all(
    p.is_file() for p in (_RELEASED_CONFIG, _RELEASED_CKPT, _RELEASED_PT2, _RELEASED_META, _PROCESSED_MANIFEST)
)
_SKIP_REASON = f"released artifacts not present under {_MODEL_DIR}"


@unittest.skipUnless(_HAVE_RELEASED, _SKIP_REASON)
class ReleasedModelTests(unittest.TestCase):
    """Ties the published numbers to the code that produced them."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads(_RELEASED_CONFIG.read_text(encoding="utf-8"))
        cls.manifest = json.loads(_PROCESSED_MANIFEST.read_text(encoding="utf-8"))
        cls.meta = json.loads(_RELEASED_META.read_text(encoding="utf-8"))
        cls.species = [str(s) for s in cls.cfg["data"]["species_variables"]]
        cls.globals = [str(s) for s in cls.cfg["data"]["global_variables"]]

    def test_config_builds_the_paper_model_with_the_paper_parameter_count(self) -> None:
        model = create_model(self.cfg)
        self.assertIsInstance(model, FlowMapAutoencoder)
        self.assertEqual(sum(p.numel() for p in model.parameters()), _PAPER_PARAM_COUNT)

    def test_config_uses_the_published_loss_weights(self) -> None:
        loss_cfg = self.cfg["training"]["adaptive_stiff_loss"]
        self.assertEqual(float(loss_cfg["lambda_phys"]), 1.0)
        self.assertEqual(float(loss_cfg["lambda_z"]), 0.5)
        self.assertFalse(bool(loss_cfg["use_weighting"]))

    def test_metadata_dt_bounds_match_the_manifest(self) -> None:
        bounds = self.meta["dt_bounds_sec"]
        self.assertAlmostEqual(
            math.log10(float(bounds["min"])), float(self.manifest["dt"]["log_min"]), places=9
        )
        self.assertAlmostEqual(
            math.log10(float(bounds["max"])), float(self.manifest["dt"]["log_max"]), places=9
        )
        self.assertEqual(list(self.meta["species_order"]), self.species)
        self.assertEqual(list(self.meta["globals_order"]), self.globals)

    def test_exported_artifact_matches_the_checkpoint_forward_pass(self) -> None:
        """The shipped .pt2 must equal checkpoint weights + training-side
        normalization. This is the artifact the paper's numbers were measured
        on, so a mismatch would invalidate them."""
        base = create_model(self.cfg).cpu().eval()
        _load_weights(base, _RELEASED_CKPT)
        base.eval()
        wrapper = _wrap_for_export(base, self.manifest)

        stats = self.manifest["per_key_stats"]
        gen = torch.Generator().manual_seed(11)
        lm = torch.tensor([float(stats[s]["log_mean"]) for s in self.species])
        ls = torch.tensor([float(stats[s]["log_std"]) for s in self.species])
        u = torch.rand(8, len(self.species), generator=gen) * 2.0 - 1.0
        y = torch.pow(10.0, lm + 0.25 * ls * u)
        dt = torch.pow(
            10.0,
            float(self.manifest["dt"]["log_min"])
            + torch.rand(8, 1, generator=gen)
            * (float(self.manifest["dt"]["log_max"]) - float(self.manifest["dt"]["log_min"])),
        )
        p = torch.pow(10.0, torch.rand(8, generator=gen) * 3.0 - 4.0)
        t = 800.0 + 1400.0 * torch.rand(8, generator=gen)
        g = torch.stack([p, t], dim=1)

        exported = torch.export.load(str(_RELEASED_PT2)).module()
        with torch.no_grad():
            got = exported(y, dt, g)
            want = wrapper(y, dt, g)

        self.assertEqual(tuple(got.shape), (8, 1, len(self.species)))
        self.assertTrue(bool(torch.isfinite(got).all()))
        torch.testing.assert_close(got, want)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
