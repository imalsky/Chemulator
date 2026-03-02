#!/usr/bin/env python3
"""
model.py

Flow-map autoencoder model:
  (y_i, dt_norm, g) -> y_j  in z-space (normalized)

Supports:
  - Residual in z-space (predict_delta)
  - Residual in physical log10 space (predict_delta_log_phys)
  - Optional simplex head (softmax_head) to output log10 probabilities, then z-normalize

Notes:
  - y is always z-space (normalized per manifest method) throughout training.
  - "physical" means log10(y_phys) space (before standardization).
  - predict_delta_log_phys and softmax_head support all manifest species normalization methods.
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
    """Export-friendly runtime assert that survives torch.export tracing."""
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
        if dt_norm.ndim == 3:
            dt = dt_norm.squeeze(-1)  # [B,K] expected
        elif dt_norm.ndim == 1:
            dt = dt_norm.unsqueeze(1)  # [B,1]
        else:
            dt = dt_norm

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


class FlowMapAutoencoder(nn.Module):
    """
    Flow-map AE: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Encoder -> LatentDynamics -> Decoder

    Heads:
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

        # Config
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")

        # Normalization metadata for log-phys/simplex heads.
        if self.predict_delta_log_phys or self.softmax_head:
            norm_tensors = _resolve_species_norm_tensors(
                state_dim=self.S,
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

    def _z_to_log10_phys(self, z: torch.Tensor) -> torch.Tensor:
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

        # Heads
        if self.softmax_head:
            # Stable: log-softmax -> log10 -> normalize to active species z-space.
            log_p = self._logsoftmax(y_pred)  # [B,K,S]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            y_pred = self._log10_phys_to_z(log10_p)
            return y_pred

        if self.predict_delta_log_phys:
            base_log = self._z_to_log10_phys(y_i)      # [B,S]
            y_pred_log = base_log.unsqueeze(1) + y_pred # [B,K,S]
            y_pred = self._log10_phys_to_z(y_pred_log)
            return y_pred

        if self.predict_delta:
            y_pred = y_pred + y_i.unsqueeze(1)
            return y_pred

        return y_pred


class FlowMapMLP(nn.Module):
    """
    MLP-only flow-map: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Single MLP over concatenated [y_i, dt_norm, g], vectorized over K.

    Heads:
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

        # Config
        self.dynamics_residual = bool(dynamics_residual)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")

        # Normalization metadata for log-phys/simplex heads.
        if self.predict_delta_log_phys or self.softmax_head:
            norm_tensors = _resolve_species_norm_tensors(
                state_dim=self.S,
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

        # Main MLP
        act = get_activation(activation_name)
        inp = self.S + 1 + self.G

        # "Dynamics-style" residual connections inside the MLP (default ON).
        # This is distinct from the explicit y + dy head (predict_delta).
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

    def _z_to_log10_phys(self, z: torch.Tensor) -> torch.Tensor:
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
        if dt_norm.ndim == 3:
            dt = dt_norm.squeeze(-1)
        elif dt_norm.ndim == 1:
            dt = dt_norm.unsqueeze(1)
        else:
            dt = dt_norm

        B = y_i.shape[0]
        K = dt.shape[1]

        y_exp = y_i.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        # All tensors are expected to share the same dtype (controlled by cfg.precision.dataset_dtype).
        x = torch.cat([y_exp, dt.unsqueeze(-1), g_exp], dim=-1)  # [B,K,S+1+G]
        y_pred = self.network(x)  # [B,K,S]

        # Heads (must match FlowMapAutoencoder exactly)
        if self.softmax_head:
            log_p = self._logsoftmax(y_pred)  # [B,K,S]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            y_pred = self._log10_phys_to_z(log10_p)
            return y_pred

        if self.predict_delta_log_phys:
            base_log = self._z_to_log10_phys(y_i)       # [B,S]
            y_pred_log = base_log.unsqueeze(1) + y_pred  # [B,K,S]
            y_pred = self._log10_phys_to_z(y_pred_log)
            return y_pred

        if self.predict_delta:
            y_pred = y_pred + y_i.unsqueeze(1)
            return y_pred

        return y_pred


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

    required_model_keys = (
        "latent_dim",
        "encoder_hidden",
        "dynamics_hidden",
        "decoder_hidden",
        "dynamics_residual",
        "activation",
        "dropout",
        "predict_delta",
        "predict_delta_log_phys",
        "softmax_head",
        "mlp_only",
    )
    missing = [k for k in required_model_keys if k not in mcfg]
    if missing:
        raise KeyError(f"Missing model config key(s): {missing}")

    latent_dim = int(mcfg["latent_dim"])
    encoder_hidden = list(mcfg["encoder_hidden"])
    dynamics_hidden = list(mcfg["dynamics_hidden"])
    decoder_hidden = list(mcfg["decoder_hidden"])
    dynamics_residual = bool(mcfg["dynamics_residual"])

    activation = str(mcfg["activation"])
    dropout = float(mcfg["dropout"])

    predict_delta = bool(mcfg["predict_delta"])
    predict_delta_log_phys = bool(mcfg["predict_delta_log_phys"])
    softmax_head = bool(mcfg["softmax_head"])

    if softmax_head:
        if predict_delta or predict_delta_log_phys:
            raise ValueError("softmax_head=True requires predict_delta=False and predict_delta_log_phys=False")

    need_species_norm = predict_delta_log_phys or softmax_head
    species_norm: Optional[Dict[str, Sequence[float] | Sequence[int]]] = None
    species_min_std = float(config.get("normalization", {}).get("min_std", 1e-10))
    if need_species_norm:
        norm_path = _resolve_repo_path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
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

    use_mlp_only = bool(mcfg["mlp_only"])

    if use_mlp_only:
        # MLP hidden sizes are specified by the existing encoder/dynamics/decoder lists.
        mlp_hidden = list(encoder_hidden) + list(dynamics_hidden) + list(decoder_hidden)
        if len(mlp_hidden) == 0:
            raise ValueError("model.mlp_only=True requires non-empty encoder_hidden/dynamics_hidden/decoder_hidden")

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
