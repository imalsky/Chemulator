#!/usr/bin/env python3
"""
model.py - Model architectures for flow-map rollout prediction.

This module defines the neural networks used by the trainer. The core task is
one-step evolution under variable timestep:

    y_{t+1} = F(y_t, dt_t, g)

where:
  - y_t: state vector at time t (shape [B, S])
  - dt_t: normalized timestep for that step (shape [B] or [B, 1])
  - g: global conditioning parameters (shape [B, G])

Two model families are provided:

  1) FlowMapMLP
     Direct MLP that predicts y(t+dt) from concatenated (y(t), dt, g).

  2) FlowMapAutoencoder
     Encoder -> LatentDynamics -> Decoder. The dynamics operate in latent space,
     then decode back to the original state space.

Design goals:
  - Deterministic behavior: no silent fallbacks.
  - Strict interfaces: ambiguous shapes/config values are rejected early.
  - Stable rollouts: optional delta prediction and conservative head init.
  - Export-safe: no int() on dynamic tensor dimensions; uses torch._check for asserts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

ActivationFactory = Callable[[], nn.Module]

# Small output init for delta heads. This reduces early rollout blow-ups.
_DELTA_OUT_INIT_STD = 1e-3


# ==============================================================================
# Shape utilities (strict, unambiguous, export-safe)
# ==============================================================================

def normalize_dt_shape(
        dt: torch.Tensor,
        batch_size: torch.SymInt,
        seq_len: torch.SymInt,
        *,
        context: str,
) -> torch.Tensor:
    """Return dt as shape [B, K] using strict, unambiguous rules.

    Accepted shapes:
      - [B, K]     canonical
      - [B, K, 1]  squeezed to [B, K]
      - [K]        allowed only when B == 1 (treated as [1, K])

    Anything else is an error. This avoids silent broadcasting that can mask
    dataset / collation bugs.

    Note: batch_size and seq_len can be symbolic (torch.SymInt) for export compatibility.
    """
    B = batch_size
    K = seq_len

    if dt.ndim == 2:
        torch._check(dt.shape[0] == B)
        torch._check(dt.shape[1] == K)
        return dt

    if dt.ndim == 3 and dt.shape[-1] == 1:
        torch._check(dt.shape[0] == B)
        torch._check(dt.shape[1] == K)
        return dt[..., 0]

    if dt.ndim == 1:
        torch._check(B == 1)
        torch._check(dt.shape[0] == K)
        return dt.reshape(1, K)

    # For unsupported shapes, we need a concrete check (this path shouldn't be hit during export)
    raise ValueError(f"{context}: unsupported dt shape {dt.shape}.")


def normalize_dt_step(dt_norm: torch.Tensor, batch_size: torch.SymInt, *, context: str) -> torch.Tensor:
    """Return a single-step dt as shape [B, 1].

    Accepted shapes:
      - [B]
      - [B, 1]
      - scalar [] only when B == 1

    Note: batch_size can be symbolic (torch.SymInt) for export compatibility.
    """
    B = batch_size

    if dt_norm.ndim == 1:
        torch._check(dt_norm.shape[0] == B)
        return dt_norm.reshape(-1, 1)

    if dt_norm.ndim == 2:
        torch._check(dt_norm.shape[0] == B)
        torch._check(dt_norm.shape[1] == 1)
        return dt_norm

    if dt_norm.ndim == 0:
        torch._check(B == 1)
        return dt_norm.reshape(1, 1)

    raise ValueError(f"{context}: unsupported dt shape {dt_norm.shape}.")


def infer_seq_len(dt_norm: torch.Tensor, batch_size: torch.SymInt, *, context: str) -> torch.SymInt:
    """Infer K from dt_norm using the same strict shape rules as normalize_dt_shape.

    Note: Returns torch.SymInt for export compatibility.
    """
    B = batch_size

    if dt_norm.ndim == 2:
        torch._check(dt_norm.shape[0] == B)
        return dt_norm.shape[1]

    if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
        torch._check(dt_norm.shape[0] == B)
        return dt_norm.shape[1]

    if dt_norm.ndim == 1:
        torch._check(B == 1)
        return dt_norm.shape[0]

    raise ValueError(f"{context}: unsupported dt shape {dt_norm.shape}.")


def _validate_g_shape(g: torch.Tensor, batch_size: torch.SymInt, global_dim: int, context: str) -> None:
    """Validate g shape is exactly [B, G].

    Note: batch_size can be symbolic (torch.SymInt) for export compatibility.
    global_dim is always a concrete int (model attribute).
    """
    torch._check(g.ndim == 2)
    torch._check(g.shape[0] == batch_size)
    torch._check(g.shape[1] == global_dim)


# ==============================================================================
# Config coercion (strict)
# ==============================================================================

def _as_int_list(value: Any, *, name: str) -> List[int]:
    """Coerce a config value into list[int].

    Supported input types:
      - int: interpreted as a single hidden layer
      - list/tuple of ints

    Any other type is an error.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: expected int or list[int].")
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name}: must be non-empty.")
        out: List[int] = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name}: must contain ints.")
            out.append(int(v))
        return out
    raise TypeError(f"{name}: expected int or list[int].")


# ==============================================================================
# Building blocks
# ==============================================================================

def get_activation(name: str) -> ActivationFactory:
    """Return an activation factory by name."""
    key = str(name).lower().strip()
    factories: Dict[str, ActivationFactory] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "elu": nn.ELU,
    }
    if key not in factories:
        raise ValueError(f"activation unsupported: {key}")
    return factories[key]


def _init_linear_xavier(linear: nn.Linear, *, scale: float = 1.0) -> None:
    """Variance-preserving init; optionally scales weights."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(float(scale))


class MLP(nn.Module):
    """Plain MLP: Linear -> (LayerNorm) -> Act -> (Dropout) -> ... -> Linear."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
        use_ln = bool(layer_norm)

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())
            layers.append(activation_factory())
            if float(dropout_p) > 0.0:
                layers.append(nn.Dropout(p=float(dropout_p)))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear_xavier(m)

    def last_linear(self) -> nn.Linear:
        # Final layer in the Sequential is always the output Linear.
        out = self.network[-1]
        if not isinstance(out, nn.Linear):
            raise RuntimeError("MLP: last layer is not Linear.")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualMLP(nn.Module):
    """Residual MLP with per-layer skip connections.

    Per hidden layer:
        h = LN( proj(h) + Dropout(Act(Linear(h))) )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        dims = [int(input_dim)] + [int(d) for d in hidden_dims]
        h_dims = [int(d) for d in hidden_dims]
        use_ln = bool(layer_norm)

        self.linears = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(h_dims)):
            in_d, out_d = dims[i], dims[i + 1]
            self.linears.append(nn.Linear(in_d, out_d))
            self.acts.append(activation_factory())
            self.drops.append(nn.Dropout(p=float(dropout_p)) if float(dropout_p) > 0.0 else nn.Identity())
            self.projs.append(nn.Identity() if in_d == out_d else nn.Linear(in_d, out_d, bias=False))
            self.norms.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())

        self.out = nn.Linear(dims[-1], int(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        # Xavier init throughout + conservative scaling on the residual branch.
        n_layers = max(1, len(self.linears))
        residual_scale = 1.0 / math.sqrt(float(n_layers))

        for lin in self.linears:
            _init_linear_xavier(lin, scale=residual_scale)

        for proj in self.projs:
            if isinstance(proj, nn.Linear):
                _init_linear_xavier(proj)

        _init_linear_xavier(self.out)

    def last_linear(self) -> nn.Linear:
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin, act, drop, proj, norm in zip(self.linears, self.acts, self.drops, self.projs, self.norms):
            h = proj(h) + drop(act(lin(h)))
            h = norm(h)
        return self.out(h)


class Encoder(nn.Module):
    """Encoder: (y, g) -> z."""

    def __init__(
            self,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            latent_dim: int,
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            residual: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        net_cls = ResidualMLP if bool(residual) else MLP
        self.network = net_cls(
            int(state_dim) + int(global_dim),
            hidden_dims,
            int(latent_dim),
            activation_factory,
            dropout_p,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, g], dim=-1)
        return self.network(x)


class LatentDynamics(nn.Module):
    """Latent-space dynamics: (z, dt, g) -> z_next or dz."""

    def __init__(
            self,
            latent_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            residual: bool = True,
            mlp_residual: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.residual = bool(residual)

        net_cls = ResidualMLP if bool(mlp_residual) else MLP
        self.network = net_cls(
            int(latent_dim) + 1 + int(global_dim),
            hidden_dims,
            int(latent_dim),
            activation_factory,
            dropout_p,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    def forward_step(self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]  # Keep as symbolic
        dt = normalize_dt_step(dt_norm, B, context="LatentDynamics.forward_step")
        x = torch.cat([z, dt.to(z.dtype), g], dim=-1)
        dz = self.network(x)
        return z + dz if self.residual else dz

    def forward(
            self,
            z: torch.Tensor,
            dt_norm: torch.Tensor,
            g: torch.Tensor,
            *,
            seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Vectorized per-step prediction (not autoregressive)."""
        torch._check(z.ndim == 2)
        B = z.shape[0]  # Keep as symbolic

        K = infer_seq_len(dt_norm, B, context="LatentDynamics.forward")
        if seq_len is not None:
            torch._check(K == seq_len)

        dt = normalize_dt_shape(dt_norm, B, K, context="LatentDynamics.forward")

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)
        x = torch.cat([z_exp, dt.unsqueeze(-1).to(z_exp.dtype), g_exp], dim=-1)

        dz = self.network(x)
        return z_exp + dz if self.residual else dz


class Decoder(nn.Module):
    """Decoder: z -> y."""

    def __init__(
            self,
            latent_dim: int,
            hidden_dims: Sequence[int],
            state_dim: int,
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            residual: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        net_cls = ResidualMLP if bool(residual) else MLP
        self.network = net_cls(
            int(latent_dim),
            hidden_dims,
            int(state_dim),
            activation_factory,
            dropout_p,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    def last_linear(self) -> nn.Linear:
        return self.network.last_linear()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# ==============================================================================
# Full models
# ==============================================================================

class FlowMapAutoencoder(nn.Module):
    """Flow-map autoencoder: Encoder -> LatentDynamics -> Decoder."""

    def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            latent_dim: int,
            encoder_hidden: Sequence[int] | int,
            dynamics_hidden: Sequence[int] | int,
            decoder_hidden: Sequence[int] | int,
            activation_name: str = "gelu",
            dropout: float = 0.0,
            residual: bool = True,
            dynamics_residual: bool = True,
            predict_delta: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.S, self.G, self.Z = int(state_dim), int(global_dim), int(latent_dim)
        self.predict_delta = bool(predict_delta)

        # Hidden dims accept either an int (one layer) or a list[int].
        enc_h = _as_int_list(encoder_hidden, name="encoder_hidden")
        dyn_h = _as_int_list(dynamics_hidden, name="dynamics_hidden")
        dec_h = _as_int_list(decoder_hidden, name="decoder_hidden")

        act_factory = get_activation(activation_name)

        self.encoder = Encoder(
            self.S,
            self.G,
            enc_h,
            self.Z,
            act_factory,
            float(dropout),
            residual=bool(residual),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )
        self.dynamics = LatentDynamics(
            self.Z,
            self.G,
            dyn_h,
            act_factory,
            float(dropout),
            residual=bool(dynamics_residual),
            mlp_residual=bool(residual),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = Decoder(
            self.Z,
            dec_h,
            self.S,
            act_factory,
            float(dropout),
            residual=bool(residual),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

        if self.predict_delta:
            self._init_delta_head()

    def _init_delta_head(self) -> None:
        """Small-output init for stable delta rollouts."""
        with torch.no_grad():
            out_layer = self.decoder.last_linear()
            nn.init.zeros_(out_layer.bias)
            nn.init.normal_(out_layer.weight, mean=0.0, std=_DELTA_OUT_INIT_STD)

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = y_i.shape[0]  # Keep as symbolic
        _validate_g_shape(g, B, self.G, context="FlowMapAutoencoder.forward_step")
        z_i = self.encoder(y_i, g)
        z_j = self.dynamics.forward_step(z_i, dt_norm, g)
        y_pred = self.decoder(z_j)
        return y_pred + y_i if self.predict_delta else y_pred

    def forward(
            self,
            y_i: torch.Tensor,
            dt_norm: torch.Tensor,
            g: torch.Tensor,
            *,
            seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for K steps."""
        torch._check(y_i.ndim == 2)

        B = y_i.shape[0]  # Keep as symbolic
        S = y_i.shape[1]
        _validate_g_shape(g, B, self.G, context="FlowMapAutoencoder.forward")

        K = infer_seq_len(dt_norm, B, context="FlowMapAutoencoder.forward")
        if seq_len is not None:
            torch._check(K == seq_len)

        dt_seq = normalize_dt_shape(dt_norm, B, K, context="FlowMapAutoencoder.forward")

        y_pred = torch.empty(B, K, S, device=y_i.device, dtype=y_i.dtype)
        state = y_i
        for t in range(K):
            state = self.forward_step(state, dt_seq[:, t], g)
            y_pred[:, t, :] = state
        return y_pred


class FlowMapMLP(nn.Module):
    """Direct MLP flow-map: (y, dt, g) -> y_next."""

    def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            activation_name: str = "gelu",
            dropout: float = 0.0,
            residual: bool = True,
            predict_delta: bool = True,
            layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.S, self.G = int(state_dim), int(global_dim)
        self.predict_delta = bool(predict_delta)

        act_factory = get_activation(activation_name)
        input_dim = self.S + 1 + self.G

        net_cls = ResidualMLP if bool(residual) else MLP
        self.network = net_cls(
            input_dim,
            [int(d) for d in hidden_dims],
            self.S,
            act_factory,
            float(dropout),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

        if self.predict_delta:
            self._init_delta_head()

    def _init_delta_head(self) -> None:
        """Small-output init for stable delta rollouts."""
        with torch.no_grad():
            out = self.network.last_linear()
            nn.init.zeros_(out.bias)
            nn.init.normal_(out.weight, mean=0.0, std=_DELTA_OUT_INIT_STD)

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = y_i.shape[0]  # Keep as symbolic
        _validate_g_shape(g, B, self.G, context="FlowMapMLP.forward_step")

        dt = normalize_dt_step(dt_norm, B, context="FlowMapMLP.forward_step")
        x = torch.cat([y_i, dt.to(y_i.dtype), g], dim=-1)

        y_pred = self.network(x)
        return y_pred + y_i if self.predict_delta else y_pred

    def forward(
            self,
            y_i: torch.Tensor,
            dt_norm: torch.Tensor,
            g: torch.Tensor,
            *,
            seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for K steps."""
        torch._check(y_i.ndim == 2)

        B = y_i.shape[0]  # Keep as symbolic
        S = y_i.shape[1]
        _validate_g_shape(g, B, self.G, context="FlowMapMLP.forward")

        K = infer_seq_len(dt_norm, B, context="FlowMapMLP.forward")
        if seq_len is not None:
            torch._check(K == seq_len)

        dt_seq = normalize_dt_shape(dt_norm, B, K, context="FlowMapMLP.forward")

        y_pred = torch.empty(B, K, S, device=y_i.device, dtype=y_i.dtype)
        state = y_i
        for t in range(K):
            state = self.forward_step(state, dt_seq[:, t], g)
            y_pred[:, t, :] = state
        return y_pred


# ==============================================================================
# Factory
# ==============================================================================

def create_model(
        config: Dict[str, Any],
        *,
        state_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """Construct a model from config.

    Notes:
      - Model selection is driven ONLY by model.type.
      - model.mlp_only is explicitly rejected (deprecated).
      - Unknown model.type is a hard error.
    """
    log = logger or logging.getLogger(__name__)

    if not isinstance(config, dict):
        raise TypeError("config must be dict.")

    if state_dim is None or global_dim is None:
        data_cfg = config.get("data")
        if not isinstance(data_cfg, dict):
            raise KeyError("data required.")

        species_vars = data_cfg.get("species_variables")
        if not isinstance(species_vars, list) or not species_vars:
            raise KeyError("data.species_variables required.")
        global_vars = data_cfg.get("global_variables", [])
        if global_vars is None:
            global_vars = []
        if not isinstance(global_vars, list):
            raise TypeError("data.global_variables must be list.")

        if state_dim is None:
            state_dim = len(species_vars)
        if global_dim is None:
            global_dim = len(global_vars)

    S, G = int(state_dim), int(global_dim)
    log.info("Model dims: S=%d, G=%d", S, G)

    mcfg = config.get("model")
    if not isinstance(mcfg, dict):
        raise KeyError("model required.")
    if "mlp_only" in mcfg:
        raise ValueError("model.mlp_only unsupported.")

    model_type_raw = mcfg.get("type")
    if model_type_raw is None:
        raise KeyError("model.type required.")
    model_type = str(model_type_raw).lower().strip()

    activation = str(mcfg.get("activation", "gelu")).lower().strip()
    dropout = float(mcfg.get("dropout", 0.0))
    predict_delta = bool(mcfg.get("predict_delta", True))
    layer_norm = bool(mcfg.get("layer_norm", True))
    layer_norm_eps = float(mcfg.get("layer_norm_eps", 1e-5))

    if model_type == "mlp":
        mlp_cfg = mcfg.get("mlp")
        if not isinstance(mlp_cfg, dict):
            raise KeyError("model.mlp required.")
        hidden_dims = _as_int_list(mlp_cfg.get("hidden_dims"), name="model.mlp.hidden_dims")
        residual = bool(mlp_cfg.get("residual", True))
        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout=dropout,
            residual=residual,
            predict_delta=predict_delta,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    if model_type == "autoencoder":
        ae_cfg = mcfg.get("autoencoder")
        if not isinstance(ae_cfg, dict):
            raise KeyError("model.autoencoder required.")

        latent_dim_raw = ae_cfg.get("latent_dim")
        if latent_dim_raw is None:
            raise KeyError("model.autoencoder.latent_dim required.")
        latent_dim = int(latent_dim_raw)

        encoder_hidden = _as_int_list(ae_cfg.get("encoder_hidden"), name="model.autoencoder.encoder_hidden")
        dynamics_hidden = _as_int_list(ae_cfg.get("dynamics_hidden"), name="model.autoencoder.dynamics_hidden")
        decoder_hidden = _as_int_list(ae_cfg.get("decoder_hidden"), name="model.autoencoder.decoder_hidden")
        residual = bool(ae_cfg.get("residual", True))
        dynamics_residual = bool(ae_cfg.get("dynamics_residual", True))

        return FlowMapAutoencoder(
            state_dim=S,
            global_dim=G,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            dynamics_hidden=dynamics_hidden,
            decoder_hidden=decoder_hidden,
            activation_name=activation,
            dropout=dropout,
            residual=residual,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    raise ValueError(f"model.type unsupported: {model_type}")