#!/usr/bin/env python3
"""
model.py - Model architectures for flow-map rollout prediction.

This module provides neural network architectures for learning time evolution
of chemical state vectors under variable timestep dt. The core learning task is:
    y_{t+1} = F(y_t, dt_t, g)

where:
    y_t: state vector at time t (species concentrations in z-space)
    dt_t: timestep (normalized to [0,1])
    g: global conditioning parameters (e.g., pressure, temperature)

Supports two model types:
    1. FlowMapMLP: Direct MLP that predicts y(t+dt) from (y(t), dt, g)
    2. FlowMapAutoencoder: Encodes to latent space, evolves dynamics there, decodes back

Both models provide:
    - forward_step(): Single-step prediction for efficient autoregressive training
    - forward(): Multi-step rollout for inference

Architecture features:
    - Residual connections for stable gradient flow
    - Configurable activation functions (SiLU recommended for smooth dynamics)
    - LayerNorm (enabled by default) for scale stabilization
    - Delta prediction mode (predict change rather than absolute state)
    - Kinetics-friendly initialization (variance-preserving + small delta head)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

ActivationFactory = Callable[[], nn.Module]

_DELTA_OUT_INIT_STD = 1e-3

def normalize_dt_shape(
    dt: torch.Tensor,
    batch_size: int,
    seq_len: int,
    context: str = "forward",
) -> torch.Tensor:
    """Canonicalize dt to shape [B, K].

    Accepted shapes (strict):
      - scalar: []  -> broadcast to [B, K]
      - [B]         -> per-sample dt, broadcast over time
      - [K]         -> per-step dt schedule, broadcast over batch
      - [B, K]      -> canonical
      - [B, K, 1]   -> squeezed to [B, K]

    dt is assumed to be already normalized by preprocessing (typically to [0,1]).
    """
    B = int(batch_size)
    K = int(seq_len)

    if dt.ndim == 0:
        return dt.reshape(1, 1).expand(B, K)

    if dt.ndim == 1:
        n = int(dt.shape[0])
        if n == B:
            return dt.reshape(B, 1).expand(B, K)
        if n == K:
            return dt.reshape(1, K).expand(B, K)
        raise ValueError(f"[{context}] dt must be scalar, [B], [K], or [B,K]. Got {tuple(dt.shape)} for B={B}, K={K}.")

    if dt.ndim == 2:
        if dt.shape != (B, K):
            raise ValueError(f"[{context}] dt must have shape [B,K]=({B},{K}). Got {tuple(dt.shape)}.")
        return dt

    if dt.ndim == 3 and dt.shape[-1] == 1:
        dt2 = dt[..., 0]
        if dt2.shape != (B, K):
            raise ValueError(f"[{context}] dt must have shape [B,K,1]=({B},{K},1). Got {tuple(dt.shape)}.")
        return dt2

    raise ValueError(f"[{context}] dt must be scalar, [B], [K], [B,K], or [B,K,1]. Got {tuple(dt.shape)}.")


def normalize_dt_step(dt_norm: torch.Tensor, batch_size: int, *, context: str) -> torch.Tensor:
    """Normalize a single-step dt to shape [B, 1]."""
    B = int(batch_size)

    if dt_norm.ndim == 0:
        return dt_norm.reshape(1, 1).expand(B, 1)

    if dt_norm.ndim == 1:
        if dt_norm.shape[0] != B:
            raise ValueError(f"[{context}] Expected dt_norm shape [{B}], got {tuple(dt_norm.shape)}.")
        return dt_norm.reshape(B, 1)

    if dt_norm.ndim == 2 and dt_norm.shape == (B, 1):
        return dt_norm

    if dt_norm.ndim == 3 and dt_norm.shape == (B, 1, 1):
        return dt_norm.squeeze(-1)

    raise ValueError(f"[{context}] Unsupported dt_norm shape for single-step: {tuple(dt_norm.shape)}. ")


def _validate_g_shape(g: torch.Tensor, batch_size: int, global_dim: int, context: str) -> None:
    if g.ndim != 2 or g.shape[0] != batch_size or g.shape[1] != global_dim:
        raise ValueError(f"[{context}] Expected g shape [B={batch_size}, G={global_dim}], got {tuple(g.shape)}.")


def get_activation(name: str) -> ActivationFactory:
    """
    Get activation function factory by name.

    Returns a callable that creates a new activation instance each time.
    """
    name = name.lower().strip()

    factories: Dict[str, ActivationFactory] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "elu": nn.ELU,
    }

    if name not in factories:
        raise ValueError(f"Unknown activation: '{name}'. Available options: {list(factories.keys())}")
    return factories[name]


def _init_linear_xavier(linear: nn.Linear, *, scale: float = 1.0) -> None:
    """Variance-preserving init (good default for smooth dynamics and residual rollouts)."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(float(scale))


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron with configurable architecture.

    Architecture:
        input -> [Linear -> LayerNorm? -> Activation -> Dropout?] x N -> Linear -> output
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

        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
        use_ln = bool(layer_norm)

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())
            layers.append(activation_factory())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=float(dropout_p)))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        # Xavier init is a robust default for smooth activations (SiLU/GELU) and regression.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear_xavier(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def last_linear(self) -> nn.Linear:
        return self.network[-1]


class ResidualMLP(nn.Module):
    """
    MLP with residual (skip) connections between hidden layers.

    Per layer:
        h = LayerNorm?( proj(h) + Dropout(Act(Linear(h))) )
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
            self.drops.append(nn.Dropout(p=float(dropout_p)) if dropout_p > 0 else nn.Identity())
            self.projs.append(nn.Identity() if in_d == out_d else nn.Linear(in_d, out_d, bias=False))
            self.norms.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())

        self.out = nn.Linear(dims[-1], int(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        # Xavier init throughout + conservative residual-branch scaling for stable rollouts.
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
    """Encoder network: maps (state, globals) to latent representation."""

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
        net_cls = ResidualMLP if residual else MLP
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
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
        return self.network(x)


class LatentDynamics(nn.Module):
    """Dynamics network in latent space."""

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

        net_cls = ResidualMLP if mlp_residual else MLP
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
        B = z.shape[0]
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
        """
        Vectorized forward over K time steps (for inference).

        Note: This computes independent single-step predictions for each timestep,
        NOT autoregressive rollout. Use forward_step in a loop for true rollout behavior.
        """
        B = z.shape[0]

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(f"Ambiguous dt_norm shape {tuple(dt_norm.shape)}.")
                K = int(dt_norm.shape[0])
            elif dt_norm.ndim == 2:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            else:
                raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for sequence.")

        if K < 1:
            raise ValueError(f"seq_len must be >= 1, got {K}")

        dt = normalize_dt_shape(dt_norm, B, K, context="LatentDynamics.forward")

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        x = torch.cat([z_exp, dt.unsqueeze(-1).to(z_exp.dtype), g_exp], dim=-1)
        dz = self.network(x)
        return z_exp + dz if self.residual else dz


class Decoder(nn.Module):
    """Decoder network: maps latent representation back to state space."""

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
        net_cls = ResidualMLP if residual else MLP
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

class FlowMapAutoencoder(nn.Module):
    """Flow-map autoencoder: Encoder -> LatentDynamics -> Decoder."""

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
        residual: bool = True,
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.S, self.G, self.Z = int(state_dim), int(global_dim), int(latent_dim)
        self.predict_delta = bool(predict_delta)

        act_factory = get_activation(activation_name)

        self.encoder = Encoder(
            self.S,
            self.G,
            list(encoder_hidden),
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
            list(dynamics_hidden),
            act_factory,
            float(dropout),
            residual=bool(dynamics_residual),
            mlp_residual=bool(residual),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder = Decoder(
            self.Z,
            list(decoder_hidden),
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
        """Small-output init for stable delta rollouts (kinetics-friendly default)."""
        try:
            with torch.no_grad():
                out_layer = self.decoder.last_linear()
                nn.init.zeros_(out_layer.bias)
                nn.init.normal_(out_layer.weight, mean=0.0, std=_DELTA_OUT_INIT_STD)
        except (AttributeError, IndexError):
            pass

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        _validate_g_shape(g, y_i.shape[0], self.G, context="FlowMapAutoencoder.forward_step")
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
        if y_i.ndim != 2:
            raise ValueError(f"Expected y_i shape [B, S], got {tuple(y_i.shape)}")

        B, S = y_i.shape
        _validate_g_shape(g, B, self.G, context="FlowMapAutoencoder.forward")

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(f"Ambiguous dt_norm shape {tuple(dt_norm.shape)}.")
                K = int(dt_norm.shape[0])
            elif dt_norm.ndim == 2:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            else:
                raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for rollout.")

        if K < 1:
            raise ValueError(f"seq_len must be >= 1, got {K}")
        dt_seq = normalize_dt_shape(dt_norm, B, K, context="FlowMapAutoencoder.forward")

        y_pred = torch.empty(B, K, S, device=y_i.device, dtype=y_i.dtype)
        state = y_i
        for t in range(K):
            state = self.forward_step(state, dt_seq[:, t], g)
            y_pred[:, t, :] = state
        return y_pred


class FlowMapMLP(nn.Module):
    """Direct MLP flow-map: predicts next state from (state, dt, globals)."""

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

        net_cls = ResidualMLP if residual else MLP
        self.network = net_cls(
            input_dim,
            list(hidden_dims),
            self.S,
            act_factory,
            float(dropout),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

        if self.predict_delta:
            self._init_delta_head()

    def _init_delta_head(self) -> None:
        """Small-output init for stable delta rollouts (kinetics-friendly default)."""
        try:
            with torch.no_grad():
                out = self.network.last_linear()
                nn.init.zeros_(out.bias)
                nn.init.normal_(out.weight, mean=0.0, std=_DELTA_OUT_INIT_STD)
        except (AttributeError, IndexError):
            pass

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = y_i.shape[0]
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
        if y_i.ndim != 2:
            raise ValueError(f"Expected y_i shape [B, S], got {tuple(y_i.shape)}")

        B, S = y_i.shape
        _validate_g_shape(g, B, self.G, context="FlowMapMLP.forward")

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(f"Ambiguous dt_norm shape {tuple(dt_norm.shape)}.")
                K = int(dt_norm.shape[0])
            elif dt_norm.ndim == 2:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = int(dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0])
            else:
                raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for rollout.")

        if K < 1:
            raise ValueError(f"seq_len must be >= 1, got {K}")
        dt_seq = normalize_dt_shape(dt_norm, B, K, context="FlowMapMLP.forward")

        y_pred = torch.empty(B, K, S, device=y_i.device, dtype=y_i.dtype)
        state = y_i
        for t in range(K):
            state = self.forward_step(state, dt_seq[:, t], g)
            y_pred[:, t, :] = state
        return y_pred

def create_model(
    config: Dict[str, Any],
    *,
    state_dim: Optional[int] = None,
    global_dim: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """Build model from configuration dictionary."""
    log = logger or logging.getLogger(__name__)

    if state_dim is None or global_dim is None:
        data_cfg = config.get("data", {})
        species_vars = list(data_cfg.get("species_variables") or [])
        global_vars = list(data_cfg.get("global_variables", []))

        if not species_vars:
            raise KeyError("data.species_variables must be non-empty. This defines the state dimension S.")

        state_dim = state_dim or len(species_vars)
        global_dim = global_dim if global_dim is not None else len(global_vars)

    S, G = int(state_dim), int(global_dim)
    log.info(f"Model dimensions: S={S} species, G={G} globals")

    mcfg = config.get("model", {})
    model_type = str(mcfg.get("type", "mlp")).lower().strip()
    if mcfg.get("mlp_only", False):
        model_type = "mlp"

    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))
    predict_delta = bool(mcfg.get("predict_delta", True))

    layer_norm = bool(mcfg.get("layer_norm", True))
    layer_norm_eps = float(mcfg.get("layer_norm_eps", 1e-5))

    if model_type == "mlp":
        mlp_cfg = mcfg.get("mlp", {})
        hidden_dims = list(mlp_cfg.get("hidden_dims", [512, 512, 512, 512]))
        residual = bool(mlp_cfg.get("residual", True))

        log.info(
            "Creating FlowMapMLP: "
            f"hidden={hidden_dims}, residual={residual}, predict_delta={predict_delta}, "
            f"layer_norm={layer_norm}"
        )

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
        ae_cfg = mcfg.get("autoencoder", {})
        latent_dim = int(ae_cfg.get("latent_dim", 256))
        encoder_hidden = list(ae_cfg.get("encoder_hidden", [512, 512]))
        dynamics_hidden = list(ae_cfg.get("dynamics_hidden", [512, 512]))
        decoder_hidden = list(ae_cfg.get("decoder_hidden", [512, 512]))
        residual = bool(ae_cfg.get("residual", True))
        dynamics_residual = bool(ae_cfg.get("dynamics_residual", True))

        log.info(
            "Creating FlowMapAutoencoder: "
            f"latent={latent_dim}, encoder={encoder_hidden}, dynamics={dynamics_hidden}, "
            f"decoder={decoder_hidden}, layer_norm={layer_norm}"
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
            residual=residual,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

    raise ValueError(f"Unknown model.type: '{model_type}'. Supported types: 'mlp', 'autoencoder'.")
