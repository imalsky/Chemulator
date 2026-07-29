#!/usr/bin/env python3
"""
model.py

Flow-map models:
  (y_i, dt_norm, g) -> y_j  in z-space (normalized)

Three architectures share this contract (selected via model.architecture, or
the legacy model.mlp_only switch when model.architecture is absent):
  - "autoencoder":   Encoder -> LatentDynamics -> Decoder (FlowMapAutoencoder)
  - "mlp":           single MLP over [y_i, dt_norm, g]    (FlowMapMLP)
  - "latent_linear": Encoder -> closed-form diagonal latent relaxation
                     -> Decoder (LatentLinearFlowMap; campaign winner of the
                     June 2026 Robertson sandbox study — see spec.md §10.1)

All architectures support the same prediction heads:
  - Residual in z-space (predict_delta)
  - Residual in physical log10 space (predict_delta_log_phys)
  - Optional simplex head (softmax_head) to output log10 probabilities, then z-normalize

Notes:
  - y is always z-space (normalized per manifest method) throughout training.
  - "physical" means log10(y_phys) space (before standardization).
  - predict_delta_log_phys and softmax_head support all manifest species normalization methods.
  - Head semantics are implemented once in _SpeciesNormHeadsMixin so they are
    guaranteed identical across architectures (required for fair A/B runs).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve config paths relative to repository root when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


# ----------------------------- Validation -------------------------------------


def _runtime_assert(cond: torch.Tensor, message: str) -> None:
    """Runtime assert that survives torch.export tracing.

    Export runs on CPU, where ``aten::_assert_async.msg`` is implemented and is
    captured as a graph node (so the dt-range guard ships inside the exported
    artifact). That op is NOT implemented for the MPS backend in torch 2.8, so
    on MPS-eager (local dev) we fall back to ``torch._assert`` — a Python-level
    check that works on every backend. The CPU/CUDA/export behavior is
    unchanged; only the otherwise-crashing MPS-eager path is rescued.
    """
    if cond.device.type == "mps":
        torch._assert(cond, message)
        return
    if hasattr(torch.ops.aten, "_assert_async") and hasattr(torch.ops.aten._assert_async, "msg"):
        torch.ops.aten._assert_async.msg(cond, message)
        return
    torch._assert(cond, message)


def _validate_dt_norm_range(dt_norm: torch.Tensor) -> None:
    """Hard error if any normalized dt values are outside [0, 1]."""
    finite_ok = torch.all(torch.isfinite(dt_norm))
    range_ok = torch.all((dt_norm >= 0.0) & (dt_norm <= 1.0))
    _runtime_assert(finite_ok, "Normalized dt contains non-finite values")
    _runtime_assert(
        range_ok,
        "Normalized dt out of range [0, 1]. This indicates dt extrapolation beyond the trained range.",
    )


# ----------------------------- Small helpers ----------------------------------

# Scale factor for gentle decoder weight init on residual heads.
# Small values near zero make the initial residual prediction ~0,
# so the model starts close to identity before learning corrections.
_GENTLE_INIT_SCALE = 0.1

ACTIVATION_ALIASES = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


_NORM_CODE_STANDARD = 0
_NORM_CODE_MIN_MAX = 1
_NORM_CODE_LOG_STANDARD = 2
_NORM_CODE_LOG_MIN_MAX = 3

_NORM_METHOD_TO_CODE = {
    "standard": _NORM_CODE_STANDARD,
    "min-max": _NORM_CODE_MIN_MAX,
    "log-standard": _NORM_CODE_LOG_STANDARD,
    "log-min-max": _NORM_CODE_LOG_MIN_MAX,
}


def get_activation(name: str) -> nn.Module:
    n = str(name).lower().strip()
    if n not in ACTIVATION_ALIASES:
        raise ValueError(f"Unknown activation '{name}'. Choices={sorted(ACTIVATION_ALIASES)}")
    return ACTIVATION_ALIASES[n]()


def _dt_norm_to_2d(dt_norm: torch.Tensor) -> torch.Tensor:
    """Canonicalize normalized dt to [B, K].

    Accepted input shapes (the shared forward contract):
      [B, K]    -> unchanged
      [B, K, 1] -> trailing singleton squeezed
      [B]       -> treated as K=1
    """
    if dt_norm.ndim == 3:
        return dt_norm.squeeze(-1)
    if dt_norm.ndim == 1:
        return dt_norm.unsqueeze(1)
    return dt_norm


def _fresh_activation(act: nn.Module) -> nn.Module:
    # Create a new instance of the same activation type (avoid sharing state if any)
    return act.__class__()


def _resolve_species_norm_tensors(
    *,
    state_dim: int,
    species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]],
    min_std: float,
) -> Dict[str, torch.Tensor]:
    """Resolve per-species normalization metadata for log-physical heads."""

    state_dim_i = int(state_dim)
    if state_dim_i <= 0:
        raise ValueError("state_dim must be > 0")
    min_std_f = float(min_std)
    if (not math.isfinite(min_std_f)) or min_std_f <= 0.0:
        raise ValueError("Invalid min_std for species normalization metadata")

    def _as_float_tensor(name: str, seq: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(list(seq), dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(f"species_norm['{name}'] must be 1D")
        if tensor.numel() != state_dim_i:
            raise ValueError(
                f"species_norm['{name}'] has length {tensor.numel()}, expected {state_dim_i}"
            )
        return tensor

    if species_norm is None:
        raise ValueError("species_norm metadata is required for log-physical or softmax heads")

    method_code = torch.as_tensor(list(species_norm["method_code"]), dtype=torch.long)
    if method_code.ndim != 1:
        raise ValueError("species_norm['method_code'] must be 1D")
    if method_code.numel() != state_dim_i:
        raise ValueError(
            f"species_norm['method_code'] has length {method_code.numel()}, expected {state_dim_i}"
        )
    if torch.any((method_code < _NORM_CODE_STANDARD) | (method_code > _NORM_CODE_LOG_MIN_MAX)):
        raise ValueError("Invalid species normalization method code")

    mean = _as_float_tensor("mean", species_norm["mean"])
    std = _as_float_tensor("std", species_norm["std"])
    vmin = _as_float_tensor("min", species_norm["min"])
    vmax = _as_float_tensor("max", species_norm["max"])
    log_mean_t = _as_float_tensor("log_mean", species_norm["log_mean"])
    log_std_t = _as_float_tensor("log_std", species_norm["log_std"])
    log_min = _as_float_tensor("log_min", species_norm["log_min"])
    log_max = _as_float_tensor("log_max", species_norm["log_max"])

    if torch.any((method_code == _NORM_CODE_STANDARD) & (std <= 0.0)):
        raise ValueError("Invalid standard normalization stats: std must be > 0")
    if torch.any((method_code == _NORM_CODE_STANDARD) & (std < min_std_f)):
        raise ValueError("Invalid standard normalization stats: std below min_std")
    if torch.any((method_code == _NORM_CODE_MIN_MAX) & ((vmax - vmin) <= 0.0)):
        raise ValueError("Invalid min-max normalization stats: max must be > min")
    if torch.any((method_code == _NORM_CODE_LOG_STANDARD) & (log_std_t <= 0.0)):
        raise ValueError("Invalid log-standard normalization stats: log_std must be > 0")
    if torch.any((method_code == _NORM_CODE_LOG_STANDARD) & (log_std_t < min_std_f)):
        raise ValueError("Invalid log-standard normalization stats: log_std below min_std")
    if torch.any((method_code == _NORM_CODE_LOG_MIN_MAX) & ((log_max - log_min) <= 0.0)):
        raise ValueError("Invalid log-min-max normalization stats: log_max must be > log_min")

    return {
        "method_code": method_code,
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "log_mean": log_mean_t,
        "log_std": log_std_t,
        "log_min": log_min,
        "log_max": log_max,
    }


def _z_to_log10_phys(
    z: torch.Tensor,
    *,
    method_code: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    vmin: torch.Tensor,
    vmax: torch.Tensor,
    log_mean: torch.Tensor,
    log_std: torch.Tensor,
    log_min: torch.Tensor,
    log_max: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized species z [..., S] to log10(physical) [..., S]."""
    out = torch.empty_like(z)
    dtype = z.dtype
    codes = method_code

    m_standard = (codes == _NORM_CODE_STANDARD)
    if bool(torch.any(m_standard)):
        y = z[..., m_standard] * std[m_standard].to(dtype=dtype) + mean[m_standard].to(dtype=dtype)
        if torch.any(y <= 0):
            min_val = float(y.min())
            raise ValueError(
                f"Non-positive species value in z->log10(phys) conversion: min={min_val:.6g}"
            )
        out[..., m_standard] = torch.log10(y)

    m_minmax = (codes == _NORM_CODE_MIN_MAX)
    if bool(torch.any(m_minmax)):
        y = z[..., m_minmax] * (vmax[m_minmax] - vmin[m_minmax]).to(dtype=dtype) + vmin[m_minmax].to(dtype=dtype)
        if torch.any(y <= 0):
            min_val = float(y.min())
            raise ValueError(
                f"Non-positive species value in z->log10(phys) conversion: min={min_val:.6g}"
            )
        out[..., m_minmax] = torch.log10(y)

    m_log_standard = (codes == _NORM_CODE_LOG_STANDARD)
    if bool(torch.any(m_log_standard)):
        out[..., m_log_standard] = (
            z[..., m_log_standard] * log_std[m_log_standard].to(dtype=dtype)
            + log_mean[m_log_standard].to(dtype=dtype)
        )

    m_log_minmax = (codes == _NORM_CODE_LOG_MIN_MAX)
    if bool(torch.any(m_log_minmax)):
        out[..., m_log_minmax] = (
            z[..., m_log_minmax]
            * (log_max[m_log_minmax] - log_min[m_log_minmax]).to(dtype=dtype)
            + log_min[m_log_minmax].to(dtype=dtype)
        )

    return out


def _log10_phys_to_z(
    log10_phys: torch.Tensor,
    *,
    method_code: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    vmin: torch.Tensor,
    vmax: torch.Tensor,
    log_mean: torch.Tensor,
    log_std: torch.Tensor,
    log_min: torch.Tensor,
    log_max: torch.Tensor,
    min_std: float = 0.0,
) -> torch.Tensor:
    """Convert log10(physical) species [..., S] to normalized z [..., S]."""
    out = torch.empty_like(log10_phys)
    dtype = log10_phys.dtype
    codes = method_code
    ten = log10_phys.new_tensor(10.0)

    m_standard = (codes == _NORM_CODE_STANDARD)
    if bool(torch.any(m_standard)):
        std_sel = std[m_standard]
        if min_std > 0.0 and bool(torch.any(std_sel < min_std)):
            raise ValueError("std below min_std in log10_phys_to_z (standard path)")
        phys = torch.pow(ten, log10_phys[..., m_standard])
        out[..., m_standard] = (
            (phys - mean[m_standard].to(dtype=dtype)) / std_sel.to(dtype=dtype)
        )

    m_minmax = (codes == _NORM_CODE_MIN_MAX)
    if bool(torch.any(m_minmax)):
        phys = torch.pow(ten, log10_phys[..., m_minmax])
        out[..., m_minmax] = (
            (phys - vmin[m_minmax].to(dtype=dtype))
            / (vmax[m_minmax] - vmin[m_minmax]).to(dtype=dtype)
        )

    m_log_standard = (codes == _NORM_CODE_LOG_STANDARD)
    if bool(torch.any(m_log_standard)):
        log_std_sel = log_std[m_log_standard]
        if min_std > 0.0 and bool(torch.any(log_std_sel < min_std)):
            raise ValueError("log_std below min_std in log10_phys_to_z (log-standard path)")
        out[..., m_log_standard] = (
            (log10_phys[..., m_log_standard] - log_mean[m_log_standard].to(dtype=dtype))
            / log_std_sel.to(dtype=dtype)
        )

    m_log_minmax = (codes == _NORM_CODE_LOG_MIN_MAX)
    if bool(torch.any(m_log_minmax)):
        out[..., m_log_minmax] = (
            (log10_phys[..., m_log_minmax] - log_min[m_log_minmax].to(dtype=dtype))
            / (log_max[m_log_minmax] - log_min[m_log_minmax]).to(dtype=dtype)
        )

    return out


# ----------------------------- Shared prediction heads ------------------------


class _SpeciesNormHeadsMixin:
    """Shared prediction-head logic for all flow-map architectures.

    Provides three pieces used identically by every model class:
      - ``_init_prediction_heads(...)``: validates the head flags and registers
        the per-species normalization buffers required by the log-physical and
        simplex heads (or sets the attributes to ``None`` when unused).
      - ``_z_to_log10_phys`` / ``_log10_phys_to_z``: per-species z-space <->
        log10(physical) conversions driven by the registered buffers.
      - ``_apply_prediction_head(...)``: the three output branches
        (softmax_head, predict_delta_log_phys, predict_delta).

    Implementing the head semantics once is a hard contract: architecture A/B
    comparisons are only meaningful if every architecture maps its raw decoder
    output to z-space predictions in exactly the same way.

    Classes must inherit as ``class Model(_SpeciesNormHeadsMixin, nn.Module)``
    and call ``_init_prediction_heads`` after ``nn.Module.__init__`` (buffer
    registration requires an initialized module).
    """

    def _init_prediction_heads(
        self,
        *,
        state_dim: int,
        predict_delta: bool,
        predict_delta_log_phys: bool,
        softmax_head: bool,
        species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]],
        min_std: float,
    ) -> None:
        """Validate head flags and register species-normalization metadata."""
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")

        # Normalization metadata for log-phys/simplex heads.
        if self.predict_delta_log_phys or self.softmax_head:
            norm_tensors = _resolve_species_norm_tensors(
                state_dim=int(state_dim),
                species_norm=species_norm,
                min_std=min_std,
            )
            self.register_buffer("species_method_code", norm_tensors["method_code"], persistent=True)
            self.register_buffer("species_mean", norm_tensors["mean"], persistent=True)
            self.register_buffer("species_std", norm_tensors["std"], persistent=True)
            self.register_buffer("species_min", norm_tensors["min"], persistent=True)
            self.register_buffer("species_max", norm_tensors["max"], persistent=True)
            self.register_buffer("species_log_mean", norm_tensors["log_mean"], persistent=True)
            self.register_buffer("species_log_std", norm_tensors["log_std"], persistent=True)
            self.register_buffer("species_log_min", norm_tensors["log_min"], persistent=True)
            self.register_buffer("species_log_max", norm_tensors["log_max"], persistent=True)
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32), persistent=True)
            self._logsoftmax = nn.LogSoftmax(dim=-1) if self.softmax_head else None
            self._species_min_std = float(min_std)
        else:
            self.species_method_code = None
            self.species_mean = None
            self.species_std = None
            self.species_min = None
            self.species_max = None
            self.species_log_mean = None
            self.species_log_std = None
            self.species_log_min = None
            self.species_log_max = None
            self.ln10 = None
            self._logsoftmax = None
            self._species_min_std = 0.0

    def _z_to_log10_phys(self, z: torch.Tensor) -> torch.Tensor:
        """Convert normalized species z [..., S] to log10(physical) [..., S]."""
        return _z_to_log10_phys(
            z,
            method_code=self.species_method_code,
            mean=self.species_mean,
            std=self.species_std,
            vmin=self.species_min,
            vmax=self.species_max,
            log_mean=self.species_log_mean,
            log_std=self.species_log_std,
            log_min=self.species_log_min,
            log_max=self.species_log_max,
        )

    def _log10_phys_to_z(self, log10_phys: torch.Tensor) -> torch.Tensor:
        """Convert log10(physical) species [..., S] to normalized z [..., S]."""
        return _log10_phys_to_z(
            log10_phys,
            method_code=self.species_method_code,
            mean=self.species_mean,
            std=self.species_std,
            vmin=self.species_min,
            vmax=self.species_max,
            log_mean=self.species_log_mean,
            log_std=self.species_log_std,
            log_min=self.species_log_min,
            log_max=self.species_log_max,
            min_std=self._species_min_std,
        )

    def _apply_prediction_head(self, y_pred: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        """Map raw decoder output [B,K,S] to z-space predictions [B,K,S].

        ``y_i`` is the (z-space) anchor state [B,S], used by the residual heads.
        """
        if self.softmax_head:
            # Stable: log-softmax -> log10 -> normalize to active species z-space.
            log_p = self._logsoftmax(y_pred)  # [B,K,S]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            return self._log10_phys_to_z(log10_p)

        if self.predict_delta_log_phys:
            base_log = self._z_to_log10_phys(y_i)        # [B,S]
            y_pred_log = base_log.unsqueeze(1) + y_pred  # [B,K,S]
            return self._log10_phys_to_z(y_pred_log)

        if self.predict_delta:
            return y_pred + y_i.unsqueeze(1)

        return y_pred


# ----------------------------- Core blocks ------------------------------------


class MLP(nn.Module):
    """Simple MLP with optional dropout."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_fresh_activation(activation))
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(p=float(dropout_p)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    Residual MLP with skip connections across each hidden layer.

    This is used for the MLP-only model to provide a "dynamics-style" residual
    (analogous in spirit to LatentDynamics(residual=True) in the autoencoder),
    while still allowing an explicit y + dy head via predict_delta.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = bool(residual)

        h = [int(x) for x in hidden_dims]
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.linears = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        dims = [self.input_dim] + h
        for i in range(len(h)):
            in_d = dims[i]
            out_d = dims[i + 1]

            self.linears.append(nn.Linear(in_d, out_d))
            self.acts.append(_fresh_activation(activation))

            if dropout_p and dropout_p > 0:
                self.dropouts.append(nn.Dropout(p=float(dropout_p)))
            else:
                self.dropouts.append(nn.Identity())

            if self.residual:
                if in_d == out_d:
                    self.proj.append(nn.Identity())
                else:
                    # Projection for skip path when dimensions differ.
                    self.proj.append(nn.Linear(in_d, out_d, bias=False))
            else:
                self.proj.append(nn.Identity())

        last_dim = dims[-1]
        self.out = nn.Linear(last_dim, self.output_dim)

    def last_linear(self) -> nn.Linear:
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin, act, do, proj in zip(self.linears, self.acts, self.dropouts, self.proj):
            y = lin(h)
            y = act(y)
            y = do(y)
            if self.residual:
                y = y + proj(h)
            h = y
        return self.out(h)


class Encoder(nn.Module):
    """Encoder: (y, g) -> z."""
    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.global_dim = global_dim
        self.latent_dim = latent_dim
        self.network = MLP(
            input_dim=state_dim + global_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
        return self.network(x)


class LatentDynamics(nn.Module):
    """Latent dynamics: (z, dt_norm, g) -> z_j (vectorized over K)."""
    def __init__(
        self,
        latent_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
        dropout_p: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_dim = global_dim
        self.residual = bool(residual)
        self.network = MLP(
            input_dim=latent_dim + 1 + global_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        z: [B,Z]
        dt_norm: [B,K] or [B,K,1] or [B]
        g: [B,G]
        -> z_j: [B,K,Z]
        """
        dt = _dt_norm_to_2d(dt_norm)  # [B,K]

        B = z.shape[0]
        K = dt.shape[1]

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        # All tensors are expected to share the same dtype (controlled by cfg.precision.dataset_dtype).
        x = torch.cat([z_exp, dt.unsqueeze(-1), g_exp], dim=-1)  # [B,K,Z+1+G]
        dz = self.network(x)  # [B,K,Z]
        return (z_exp + dz) if self.residual else dz


class Decoder(nn.Module):
    """Decoder: z -> y (broadcast over K)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        state_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.network = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Accepts [B,K,Z] (or [B,Z]); nn.Linear handles leading dims.
        return self.network(z)


class FlowMapAutoencoder(_SpeciesNormHeadsMixin, nn.Module):
    """
    Flow-map AE: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Encoder -> LatentDynamics -> Decoder

    Heads (shared, see _SpeciesNormHeadsMixin):
      - predict_delta (z-space residual)
      - predict_delta_log_phys (delta in log10-physical space, then re-standardize)
      - softmax_head (simplex probabilities, then normalized by active species method)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        dynamics_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation_name: str = "gelu",
        dropout: float = 0.0,
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]] = None,
        softmax_head: bool = False,
        min_std: float = 1e-10,
    ):
        super().__init__()

        # Dimensions
        self.S = int(state_dim)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        # Head flags + species normalization buffers (shared contract).
        self._init_prediction_heads(
            state_dim=self.S,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            softmax_head=softmax_head,
            species_norm=species_norm,
            min_std=min_std,
        )

        # Activation
        act = get_activation(activation_name)

        # Submodules
        self.encoder = Encoder(
            state_dim=self.S,
            global_dim=self.G,
            hidden_dims=list(encoder_hidden),
            latent_dim=self.Z,
            activation=act,
            dropout_p=float(dropout),
        )
        self.dynamics = LatentDynamics(
            latent_dim=self.Z,
            global_dim=self.G,
            hidden_dims=list(dynamics_hidden),
            activation=act,
            dropout_p=float(dropout),
            residual=dynamics_residual,
        )
        self.decoder = Decoder(
            latent_dim=self.Z,
            hidden_dims=list(decoder_hidden),
            state_dim=self.S,
            activation=act,
            dropout_p=float(dropout),
        )

        # Gentle decoder init for residual heads
        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                lin_out: nn.Linear = self.decoder.network.network[-1]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(_GENTLE_INIT_SCALE)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions in z-space (normalized).
        Shapes:
          y_i: [B,S]
          dt_norm: [B,K] or [B,K,1]
          g: [B,G]
          -> y_pred_z: [B,K,S]
        """
        _validate_dt_norm_range(dt_norm)
        # Encode once
        z_i = self.encoder(y_i, g)                        # [B,Z]
        # Vectorized latent evolution over K
        z_j = self.dynamics(z_i, dt_norm, g)              # [B,K,Z]
        # Decode
        y_pred = self.decoder(z_j)                        # [B,K,S]
        # Shared head branches (softmax / log-phys residual / z residual)
        return self._apply_prediction_head(y_pred, y_i)


class FlowMapMLP(_SpeciesNormHeadsMixin, nn.Module):
    """
    MLP-only flow-map: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Single MLP over concatenated [y_i, dt_norm, g], vectorized over K.

    Heads (shared, see _SpeciesNormHeadsMixin):
      - predict_delta (z-space residual)
      - predict_delta_log_phys (delta in log10-physical space, then re-standardize)
      - softmax_head (simplex probabilities, then normalized by active species method)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        activation_name: str = "gelu",
        dropout: float = 0.0,
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]] = None,
        softmax_head: bool = False,
        min_std: float = 1e-10,
    ):
        super().__init__()

        # Dimensions
        self.S = int(state_dim)
        self.G = int(global_dim)

        # "Dynamics-style" residual connections inside the MLP (default ON).
        # This is distinct from the explicit y + dy head (predict_delta).
        self.dynamics_residual = bool(dynamics_residual)

        # Head flags + species normalization buffers (shared contract).
        self._init_prediction_heads(
            state_dim=self.S,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            softmax_head=softmax_head,
            species_norm=species_norm,
            min_std=min_std,
        )

        # Main MLP
        act = get_activation(activation_name)
        inp = self.S + 1 + self.G

        if self.dynamics_residual:
            self.network = ResidualMLP(
                input_dim=inp,
                hidden_dims=list(hidden_dims),
                output_dim=self.S,
                activation=act,
                dropout_p=float(dropout),
                residual=True,
            )
        else:
            self.network = MLP(
                input_dim=inp,
                hidden_dims=list(hidden_dims),
                output_dim=self.S,
                activation=act,
                dropout_p=float(dropout),
            )

        # Gentle decoder init for residual heads
        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                if isinstance(self.network, ResidualMLP):
                    lin_out = self.network.last_linear()
                else:
                    lin_out = self.network.network[-1]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(_GENTLE_INIT_SCALE)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions in z-space (normalized).
        Shapes:
          y_i: [B,S]
          dt_norm: [B,K] or [B,K,1] or [B]
          g: [B,G]
          -> y_pred_z: [B,K,S]
        """
        _validate_dt_norm_range(dt_norm)
        dt = _dt_norm_to_2d(dt_norm)  # [B,K]

        B = y_i.shape[0]
        K = dt.shape[1]

        y_exp = y_i.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        # All tensors are expected to share the same dtype (controlled by cfg.precision.dataset_dtype).
        x = torch.cat([y_exp, dt.unsqueeze(-1), g_exp], dim=-1)  # [B,K,S+1+G]
        y_pred = self.network(x)  # [B,K,S]
        # Shared head branches (softmax / log-phys residual / z residual)
        return self._apply_prediction_head(y_pred, y_i)


class LatentLinearFlowMap(_SpeciesNormHeadsMixin, nn.Module):
    """
    Latent-linear flow map: (y_i, dt_norm, g) -> y_j in z-space.

    Architecture (winner of the June 2026 Robertson sandbox campaign; see
    spec.md §10.1 and the umbrella-level EXPERIMENTS.md / IMPLEMENTATION_PLAN.md):

        feats   = trunk([y_i, g])              # shared encoder trunk (activated)
        h0      = head_h0(feats)               # [B,L] initial latent state
        h_eq    = head_heq(feats)              # [B,L] target ("equilibrium") latent
        log10_k = head_rate(feats)             # [B,L] per-mode log10 decay rates
        h(dt)   = h_eq + exp(-k*dt_phys) * (h0 - h_eq)   # exact diagonal linear-ODE solution
        out     = decoder(h(dt))               # decoder_mode "mlp" or "linear"

    Each latent mode relaxes toward a state-dependent target with a
    state-dependent timescale — the natural inductive bias for stiff
    chemistry, evaluated in closed form so any dt costs one forward pass.
    Both the rates AND the equilibrium are state-dependent (encoder heads):
    the campaign's 2x2 factorial showed fixed/global operators are strictly
    worse (up to 8.5x on the discriminator metric), with h_eq
    state-dependence mattering most.

    dt handling: the forward contract passes normalized dt in [0,1]; the model
    reconstructs log10(dt_phys) = dt_log_min + dt_norm * (dt_log_max - dt_log_min).
    The bounds MUST come from the same normalization manifest the dataset uses
    (create_model loads them; never hardcode) and are stored as persistent
    buffers, so they ride along in checkpoints and exported artifacts.

    Numerical safety:
      - log10(k*dt) is clamped from ABOVE at ``rate_clamp`` (default 3.0:
        exp(-10^3) == 0 in fp32, so the mode is exactly "fully relaxed";
        without the clamp, 10**x can overflow to inf and poison backward).
        No lower clamp — underflow to decay == 1 (no relaxation yet) is exact.
      - The 10**x -> exp(-x) relaxation path always runs in fp32, independent
        of any AMP policy (mirrors how other precision-sensitive paths are
        handled in this codebase).

    decoder_mode:
      - "mlp":    MLP(latent_dim -> decoder_hidden -> S). The all-round primary.
      - "linear": single Linear(latent_dim, S) — the "lindec" deployment
        variant (the model output becomes an interpretable mixture of
        exponentials; a linear readout cannot nonlinearly amplify
        off-manifold latent drift). Pair with a larger latent_dim to match
        parameter counts. decoder_hidden must be [] in this mode.

    Init: Xavier-uniform weights + zero biases everywhere; the rate-head bias
    is then drawn from U(rate_init) using a generator seeded with
    ``init_seed`` so builds are reproducible. The uniform spread is
    load-bearing — it covers slow-to-fast modes at init (the deterministic
    "ladder" alternative tested worse on the discriminator metric). The
    decoder output layer gets the standard gentle residual-head init.

    Heads (shared, see _SpeciesNormHeadsMixin):
      - predict_delta (z-space residual; the campaign-recommended default)
      - predict_delta_log_phys (delta in log10-physical space, then re-standardize)
      - softmax_head (simplex probabilities; the conservation-flavored A/B arm)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        decoder_mode: str,
        decoder_hidden: Sequence[int],
        dt_log_min: float,
        dt_log_max: float,
        rate_clamp: float = 3.0,
        rate_init: Sequence[float] = (-5.0, 5.0),
        init_seed: int = 0,
        activation_name: str = "silu",
        dropout: float = 0.0,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]] = None,
        softmax_head: bool = False,
        min_std: float = 1e-10,
    ):
        super().__init__()

        # Dimensions
        self.S = int(state_dim)
        self.G = int(global_dim)
        self.L = int(latent_dim)
        if self.S <= 0:
            raise ValueError("state_dim must be > 0")
        if self.L <= 0:
            raise ValueError("latent_dim must be > 0")

        # The encoder trunk is what makes rates/equilibrium state-dependent;
        # an empty trunk would collapse the model to a fixed affine operator
        # (the campaign's worst formulation). Hard error.
        encoder_hidden = [int(x) for x in encoder_hidden]
        if len(encoder_hidden) == 0:
            raise ValueError("latent_linear requires non-empty encoder_hidden")

        decoder_hidden = [int(x) for x in decoder_hidden]
        self.decoder_mode = str(decoder_mode)
        if self.decoder_mode not in ("mlp", "linear"):
            raise ValueError(
                f"Unknown decoder_mode {decoder_mode!r}. Choices=['mlp', 'linear']"
            )
        if self.decoder_mode == "linear" and len(decoder_hidden) > 0:
            raise ValueError(
                "decoder_mode='linear' requires decoder_hidden=[] (the lindec variant "
                "is a single Linear readout by definition)"
            )
        if self.decoder_mode == "mlp" and len(decoder_hidden) == 0:
            raise ValueError(
                "decoder_mode='mlp' requires non-empty decoder_hidden "
                "(use decoder_mode='linear' for a bare Linear readout)"
            )

        # dt reconstruction constants (from the normalization manifest).
        dt_log_min_f = float(dt_log_min)
        dt_log_max_f = float(dt_log_max)
        if not (math.isfinite(dt_log_min_f) and math.isfinite(dt_log_max_f)):
            raise ValueError("dt_log_min/dt_log_max must be finite")
        if dt_log_max_f <= dt_log_min_f:
            raise ValueError("dt_log_max must be > dt_log_min")
        self.register_buffer(
            "dt_log_min", torch.tensor(dt_log_min_f, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "dt_log_range",
            torch.tensor(dt_log_max_f - dt_log_min_f, dtype=torch.float32),
            persistent=True,
        )

        # Rate-path constants.
        self.rate_clamp = float(rate_clamp)
        if not math.isfinite(self.rate_clamp):
            raise ValueError("rate_clamp must be finite")
        rate_init_list = [float(x) for x in rate_init]
        if len(rate_init_list) != 2:
            raise ValueError("rate_init must be [low, high]")
        rate_lo, rate_hi = rate_init_list
        if not (math.isfinite(rate_lo) and math.isfinite(rate_hi)) or rate_lo >= rate_hi:
            raise ValueError("rate_init must satisfy low < high (finite)")

        # Head flags + species normalization buffers (shared contract).
        self._init_prediction_heads(
            state_dim=self.S,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            softmax_head=softmax_head,
            species_norm=species_norm,
            min_std=min_std,
        )

        # Shared encoder trunk: every hidden layer is activated so the three
        # heads read a nonlinear feature vector (matches the validated
        # sandbox formulation).
        act = get_activation(activation_name)
        trunk_layers: list[nn.Module] = []
        in_dim = self.S + self.G
        for width in encoder_hidden:
            trunk_layers.append(nn.Linear(in_dim, width))
            trunk_layers.append(_fresh_activation(act))
            if dropout and float(dropout) > 0:
                trunk_layers.append(nn.Dropout(p=float(dropout)))
            in_dim = width
        self.encoder_trunk = nn.Sequential(*trunk_layers)

        # Per-mode heads off the shared trunk (all state-dependent).
        self.head_h0 = nn.Linear(in_dim, self.L)
        self.head_heq = nn.Linear(in_dim, self.L)
        self.head_rate = nn.Linear(in_dim, self.L)

        # Decoder: latent state -> raw species output.
        if self.decoder_mode == "linear":
            self.decoder: nn.Module = nn.Linear(self.L, self.S)
        else:
            self.decoder = MLP(
                input_dim=self.L,
                hidden_dims=decoder_hidden,
                output_dim=self.S,
                activation=act,
                dropout_p=float(dropout),
            )

        self._reset_parameters(rate_lo=rate_lo, rate_hi=rate_hi, init_seed=int(init_seed))

    def _decoder_output_linear(self) -> nn.Linear:
        """Return the decoder's final Linear layer (both decoder modes)."""
        if self.decoder_mode == "linear":
            assert isinstance(self.decoder, nn.Linear)
            return self.decoder
        return self.decoder.network[-1]

    def _reset_parameters(self, *, rate_lo: float, rate_hi: float, init_seed: int) -> None:
        """Xavier/zero-bias init + seeded uniform rate spread + gentle decoder out.

        The rate-head bias draw uses a dedicated seeded generator so two
        builds with the same ``init_seed`` start from identical mode
        timescales regardless of global RNG state (the sandbox left this
        unseeded — flagged in the campaign audit; do not regress).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        with torch.no_grad():
            gen = torch.Generator()
            gen.manual_seed(int(init_seed))
            # Spread initial log10-rates over many decades: at init the modes
            # must cover slow-to-fast timescales for the relaxation to train.
            self.head_rate.bias.uniform_(rate_lo, rate_hi, generator=gen)

            # Gentle decoder init for residual heads (mirrors the other
            # architectures: near-zero initial residual => near-identity map).
            if self.predict_delta or self.predict_delta_log_phys:
                lin_out = self._decoder_output_linear()
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(_GENTLE_INIT_SCALE)

    def _encode_heads(
        self, y_i: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the anchor state once: (h0, h_eq, log10_k), each [B,L]."""
        x = torch.cat([y_i, g], dim=-1) if g.numel() > 0 else y_i
        feats = self.encoder_trunk(x)
        return self.head_h0(feats), self.head_heq(feats), self.head_rate(feats)

    def _latent_state(
        self,
        h0: torch.Tensor,
        h_eq: torch.Tensor,
        log10_k: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form latent relaxation h(dt) = h_eq + exp(-k*dt)(h0 - h_eq).

        Inputs: h0/h_eq/log10_k [B,L]; dt normalized [B,K].
        Output: [B,K,L], always fp32 (precision-sensitive path; see class doc).
        """
        # Reconstruct log10(dt_phys) from normalized dt using manifest bounds.
        log10_dt = self.dt_log_min + dt.to(torch.float32) * self.dt_log_range  # [B,K]
        # log10(k * dt) = log10_k + log10_dt, clamped from above only.
        log10_kdt = log10_k.to(torch.float32).unsqueeze(1) + log10_dt.unsqueeze(-1)  # [B,K,L]
        log10_kdt = torch.clamp(log10_kdt, max=self.rate_clamp)
        decay = torch.exp(-torch.pow(10.0, log10_kdt))  # in (0, 1], [B,K,L]

        h0_f = h0.to(torch.float32)
        h_eq_f = h_eq.to(torch.float32)
        return h_eq_f.unsqueeze(1) + decay * (h0_f - h_eq_f).unsqueeze(1)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions in z-space (normalized).
        Shapes:
          y_i: [B,S]
          dt_norm: [B,K] or [B,K,1] or [B]
          g: [B,G]
          -> y_pred_z: [B,K,S]
        """
        _validate_dt_norm_range(dt_norm)
        dt = _dt_norm_to_2d(dt_norm)  # [B,K]

        # Encode once per anchor; relax in closed form; decode over K.
        h0, h_eq, log10_k = self._encode_heads(y_i, g)   # each [B,L]
        h_dt = self._latent_state(h0, h_eq, log10_k, dt)  # [B,K,L] fp32
        y_pred = self.decoder(h_dt)                       # [B,K,S]
        # Shared head branches (softmax / log-phys residual / z residual)
        return self._apply_prediction_head(y_pred, y_i)


def create_model(config: Dict[str, Any], logger: Optional["logging.Logger"] = None) -> nn.Module:
    """
    Build model from config.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    data_cfg = config.get("data", {})
    if "target_species" in data_cfg:
        raise KeyError(
            "Unsupported config key: data.target_species (outputs always match data.species_variables)"
        )

    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    S = len(species_vars)
    G = len(global_vars)

    mcfg = config.get("model")
    if not isinstance(mcfg, dict):
        raise KeyError("Missing config: model")

    if "vae_mode" in mcfg and bool(mcfg["vae_mode"]):
        raise ValueError("vae_mode is not supported")

    if "allow_partial_simplex" in mcfg:
        raise KeyError("Unsupported config key: model.allow_partial_simplex")

    # ------------------------- Architecture resolution -------------------------
    # New-style configs select via model.architecture; legacy configs use the
    # model.mlp_only switch. The two are mutually exclusive (fail fast rather
    # than guess which one the user meant).
    _VALID_ARCHITECTURES = ("autoencoder", "mlp", "latent_linear")
    _COMMON_MODEL_KEYS = (
        "activation",
        "dropout",
        "predict_delta",
        "predict_delta_log_phys",
        "softmax_head",
    )

    architecture: Optional[str]
    if "architecture" in mcfg:
        if "mlp_only" in mcfg:
            raise KeyError(
                "model.architecture and model.mlp_only are mutually exclusive; "
                "remove the legacy model.mlp_only switch when model.architecture is set"
            )
        architecture = str(mcfg["architecture"])
        if architecture not in _VALID_ARCHITECTURES:
            raise ValueError(
                f"Unknown model.architecture {architecture!r}. Choices={list(_VALID_ARCHITECTURES)}"
            )
    else:
        architecture = None  # legacy dispatch via mlp_only, resolved below

    if architecture == "latent_linear":
        required_model_keys = (
            "latent_dim",
            "encoder_hidden",
            "decoder_mode",
            "decoder_hidden",
            "rate_clamp",
            "rate_init",
            "init_seed",
        ) + _COMMON_MODEL_KEYS
        # Keys that belong to the other architectures would be silent no-ops
        # here; reject them so configs always say what they mean.
        forbidden = [k for k in ("dynamics_hidden", "dynamics_residual") if k in mcfg]
        if forbidden:
            raise KeyError(
                f"model key(s) {forbidden} are not used by architecture='latent_linear'; "
                "remove them (this codebase forbids silent no-op keys)"
            )
    elif architecture in ("autoencoder", "mlp"):
        # `latent_dim` is required here for symmetry with the legacy branch
        # below, even though FlowMapMLP has no bottleneck and ignores it.
        required_model_keys = (
            "latent_dim",
            "encoder_hidden",
            "dynamics_hidden",
            "decoder_hidden",
            "dynamics_residual",
        ) + _COMMON_MODEL_KEYS
        # Mirror of the latent_linear check above: latent-linear-only keys are
        # silent no-ops here, so reject them too.
        forbidden = [
            k for k in ("decoder_mode", "rate_clamp", "rate_init", "init_seed") if k in mcfg
        ]
        if forbidden:
            raise KeyError(
                f"model key(s) {forbidden} are not used by architecture={architecture!r}; "
                "remove them (this codebase forbids silent no-op keys)"
            )
    else:
        # Legacy configs: same key set as before, including mlp_only.
        required_model_keys = (
            "latent_dim",
            "encoder_hidden",
            "dynamics_hidden",
            "decoder_hidden",
            "dynamics_residual",
        ) + _COMMON_MODEL_KEYS + ("mlp_only",)

    missing = [k for k in required_model_keys if k not in mcfg]
    if missing:
        raise KeyError(f"Missing model config key(s): {missing}")

    if architecture is None:
        architecture = "mlp" if bool(mcfg["mlp_only"]) else "autoencoder"

    # ------------------------------ Common keys --------------------------------
    latent_dim = int(mcfg["latent_dim"])
    encoder_hidden = list(mcfg["encoder_hidden"])
    decoder_hidden = list(mcfg["decoder_hidden"])

    activation = str(mcfg["activation"])
    dropout = float(mcfg["dropout"])

    predict_delta = bool(mcfg["predict_delta"])
    predict_delta_log_phys = bool(mcfg["predict_delta_log_phys"])
    softmax_head = bool(mcfg["softmax_head"])

    if softmax_head:
        if predict_delta or predict_delta_log_phys:
            raise ValueError("softmax_head=True requires predict_delta=False and predict_delta_log_phys=False")

    # ----------------------- Normalization manifest (lazy) ---------------------
    # Needed for (a) species stats when a log-physical/simplex head is active,
    # (b) dt bounds when architecture='latent_linear'. Loaded at most once.
    need_species_norm = predict_delta_log_phys or softmax_head
    need_manifest = need_species_norm or architecture == "latent_linear"
    manifest: Optional[Dict[str, Any]] = None
    if need_manifest:
        norm_path = _resolve_repo_path(config["paths"]["processed_data_dir"]) / "normalization.json"
        if not norm_path.is_file():
            raise FileNotFoundError(
                f"Missing normalization manifest: {norm_path}. "
                "It is required for log-physical/simplex heads and for "
                "architecture='latent_linear' (dt bounds); run preprocessing first."
            )
        with open(norm_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]] = None
    species_min_std = float(config.get("normalization", {}).get("min_std", 1e-10))
    if need_species_norm:
        stats = manifest["per_key_stats"]
        methods = manifest.get("normalization_methods")
        if not isinstance(methods, dict):
            raise KeyError("normalization_methods missing from normalization manifest")
        if "min_std" not in manifest:
            raise KeyError("min_std missing from normalization manifest")
        species_min_std = float(manifest["min_std"])
        if (not math.isfinite(species_min_std)) or species_min_std <= 0.0:
            raise ValueError("Invalid min_std in normalization manifest")

        method_codes: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        mins: list[float] = []
        maxs: list[float] = []
        log_means: list[float] = []
        log_stds: list[float] = []
        log_mins: list[float] = []
        log_maxs: list[float] = []

        for name in species_vars:
            if name not in stats:
                raise KeyError(f"Species '{name}' not in normalization stats")
            s = stats[name]

            method_name = str(methods.get(name, ""))
            if method_name not in _NORM_METHOD_TO_CODE:
                raise ValueError(
                    f"Unsupported normalization method for species '{name}': {method_name!r}"
                )
            code = _NORM_METHOD_TO_CODE[method_name]
            method_codes.append(code)

            if code == _NORM_CODE_STANDARD:
                for k in ("mean", "std"):
                    if k not in s:
                        raise KeyError(f"Species '{name}' stats missing {k}")
            elif code == _NORM_CODE_MIN_MAX:
                for k in ("min", "max"):
                    if k not in s:
                        raise KeyError(f"Species '{name}' stats missing {k}")
            elif code == _NORM_CODE_LOG_STANDARD:
                for k in ("log_mean", "log_std"):
                    if k not in s:
                        raise KeyError(f"Species '{name}' stats missing {k}")
            elif code == _NORM_CODE_LOG_MIN_MAX:
                for k in ("log_min", "log_max"):
                    if k not in s:
                        raise KeyError(f"Species '{name}' stats missing {k}")
            else:
                raise ValueError("Unexpected normalization method code")

            means.append(float(s.get("mean", 0.0)))
            stds.append(float(s.get("std", 1.0)))
            mins.append(float(s.get("min", 0.0)))
            maxs.append(float(s.get("max", 1.0)))
            log_means.append(float(s.get("log_mean", 0.0)))
            log_stds.append(float(s.get("log_std", 1.0)))
            log_mins.append(float(s.get("log_min", 0.0)))
            log_maxs.append(float(s.get("log_max", 1.0)))

        species_norm = {
            "method_code": method_codes,
            "mean": means,
            "std": stds,
            "min": mins,
            "max": maxs,
            "log_mean": log_means,
            "log_std": log_stds,
            "log_min": log_mins,
            "log_max": log_maxs,
        }
        method_summary = sorted({str(methods.get(name, "")) for name in species_vars})
        log.info(
            "Loaded species normalization metadata for %d species (methods=%s)",
            len(species_vars),
            method_summary,
        )

    # ------------------------------- Dispatch ----------------------------------
    if architecture == "latent_linear":
        # dt bounds come from the SAME manifest the dataset normalizes with so
        # the model's dt reconstruction is consistent end-to-end. Never
        # hardcode them (they are dataset properties, not model constants).
        dt_spec = manifest.get("dt") if manifest is not None else None
        if not isinstance(dt_spec, dict) or "log_min" not in dt_spec or "log_max" not in dt_spec:
            raise KeyError(
                "normalization manifest is missing the dt spec (dt.log_min / dt.log_max) "
                "required by architecture='latent_linear'"
            )
        dt_log_min = float(dt_spec["log_min"])
        dt_log_max = float(dt_spec["log_max"])
        decoder_mode = str(mcfg["decoder_mode"])

        model = LatentLinearFlowMap(
            state_dim=S,
            global_dim=G,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_mode=decoder_mode,
            decoder_hidden=decoder_hidden,
            dt_log_min=dt_log_min,
            dt_log_max=dt_log_max,
            rate_clamp=float(mcfg["rate_clamp"]),
            rate_init=list(mcfg["rate_init"]),
            init_seed=int(mcfg["init_seed"]),
            activation_name=activation,
            dropout=dropout,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            species_norm=species_norm,
            softmax_head=softmax_head,
            min_std=species_min_std,
        )
        log.info(
            "Built latent_linear flow map: latent_dim=%d decoder_mode=%s dt_log=[%.4f, %.4f]",
            latent_dim,
            decoder_mode,
            dt_log_min,
            dt_log_max,
        )
        return model

    dynamics_hidden = list(mcfg["dynamics_hidden"])
    dynamics_residual = bool(mcfg["dynamics_residual"])

    if architecture == "mlp":
        # MLP hidden sizes are specified by the existing encoder/dynamics/decoder lists.
        mlp_hidden = list(encoder_hidden) + list(dynamics_hidden) + list(decoder_hidden)
        if len(mlp_hidden) == 0:
            raise ValueError(
                "MLP architecture (model.architecture='mlp' or model.mlp_only=true) "
                "requires non-empty encoder_hidden/dynamics_hidden/decoder_hidden"
            )

        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=mlp_hidden,
            activation_name=activation,
            dropout=dropout,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            species_norm=species_norm,
            softmax_head=softmax_head,
            min_std=species_min_std,
        )

    return FlowMapAutoencoder(
        state_dim=S,
        global_dim=G,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        dynamics_hidden=dynamics_hidden,
        decoder_hidden=decoder_hidden,
        activation_name=activation,
        dropout=dropout,
        dynamics_residual=dynamics_residual,
        predict_delta=predict_delta,
        predict_delta_log_phys=predict_delta_log_phys,
        species_norm=species_norm,
        softmax_head=softmax_head,
        min_std=species_min_std,
    )
