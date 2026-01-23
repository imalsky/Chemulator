#!/usr/bin/env python3
"""
model.py - Model architectures for flow-map rollout prediction.

Supports two model types:
    1. Direct MLP flow-map: predicts y(t+dt) from y(t), dt, and globals
    2. Autoencoder flow-map: encodes to latent, evolves in latent space, decodes back

All models support both:
    - Single-step forward (forward_step): efficient for autoregressive training
    - Vectorized forward (forward): batched over K rollout steps for inference
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ==============================================================================
# Activation Factory
# ==============================================================================


def get_activation(name: str) -> Callable[[], nn.Module]:
    """
    Get activation function factory by name.

    Returns a callable that creates a new activation instance each time,
    preserving any parameters (e.g., LeakyReLU negative_slope).
    """
    name = name.lower().strip()

    factories: Dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "elu": nn.ELU,
    }

    if name not in factories:
        raise ValueError(f"Unknown activation: {name}. Options: {list(factories.keys())}")

    return factories[name]


# ==============================================================================
# Core Networks
# ==============================================================================


class MLP(nn.Module):
    """Standard MLP with configurable layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout_p: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_factory())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def last_linear(self) -> nn.Linear:
        """Return the final linear layer for initialization."""
        return self.network[-1]


class ResidualMLP(nn.Module):
    """MLP with residual connections between layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout_p: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        h = list(hidden_dims)

        self.linears = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()

        for i in range(len(h)):
            in_d, out_d = dims[i], dims[i + 1]
            self.linears.append(nn.Linear(in_d, out_d))
            self.acts.append(activation_factory())
            self.drops.append(
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
            )
            self.projs.append(
                nn.Identity() if in_d == out_d else nn.Linear(in_d, out_d, bias=False)
            )

        self.out = nn.Linear(dims[-1], output_dim)

    def last_linear(self) -> nn.Linear:
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin, act, drop, proj in zip(
            self.linears, self.acts, self.drops, self.projs
        ):
            h = drop(act(lin(h))) + proj(h)
        return self.out(h)


# ==============================================================================
# Autoencoder Components
# ==============================================================================


class Encoder(nn.Module):
    """Encode (y, g) -> latent z."""

    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout_p: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        net_cls = ResidualMLP if residual else MLP
        self.network = net_cls(
            state_dim + global_dim, hidden_dims, latent_dim, activation_factory, dropout_p
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
        return self.network(x)


class LatentDynamics(nn.Module):
    """
    Evolve latent state: (z, dt_norm, g) -> z_next.

    Supports both single-step (forward_step) and vectorized (forward) modes.
    """

    def __init__(
        self,
        latent_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        activation_factory: Callable[[], nn.Module],
        dropout_p: float = 0.0,
        residual: bool = True,
        mlp_residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        net_cls = ResidualMLP if mlp_residual else MLP
        self.network = net_cls(
            latent_dim + 1 + global_dim,
            hidden_dims,
            latent_dim,
            activation_factory,
            dropout_p,
        )

    def forward_step(
        self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-step forward for autoregressive training.

        Args:
            z: [B, Z] latent state
            dt_norm: [B] or scalar normalized time step
            g: [B, G] global parameters

        Returns:
            z_next: [B, Z] next latent state
        """
        B = z.shape[0]

        # Ensure dt has correct shape
        if dt_norm.ndim == 0:
            dt = dt_norm.expand(B, 1)
        elif dt_norm.ndim == 1:
            dt = dt_norm.view(B, 1)
        else:
            dt = dt_norm.view(B, 1)

        # Only disable autocast for the dt concatenation, not the matmuls
        x = torch.cat([z, dt.to(z.dtype), g], dim=-1)
        dz = self.network(x)

        return z + dz if self.residual else dz

    def forward(
        self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward over K time steps.

        Args:
            z: [B, Z] initial latent state
            dt_norm: [B, K] or [B, K, 1] normalized time steps
            g: [B, G] global parameters

        Returns:
            z_seq: [B, K, Z] sequence of latent states
        """
        dt = self._normalize_dt_shape(dt_norm)
        B, K = z.shape[0], dt.shape[1]

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        x = torch.cat([z_exp, dt.unsqueeze(-1).to(z_exp.dtype), g_exp], dim=-1)
        dz = self.network(x)

        return z_exp + dz if self.residual else dz

    @staticmethod
    def _normalize_dt_shape(dt: torch.Tensor) -> torch.Tensor:
        if dt.ndim == 3:
            return dt.squeeze(-1)
        if dt.ndim == 1:
            return dt.unsqueeze(1)
        return dt


class Decoder(nn.Module):
    """Decode latent z -> state y."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        state_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout_p: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        net_cls = ResidualMLP if residual else MLP
        self.network = net_cls(
            latent_dim, hidden_dims, state_dim, activation_factory, dropout_p
        )

    def last_linear(self) -> nn.Linear:
        return self.network.last_linear()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# ==============================================================================
# Full Models
# ==============================================================================


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
    ):
        super().__init__()
        self.S, self.G, self.Z = state_dim, global_dim, latent_dim
        self.predict_delta = predict_delta

        act_factory = get_activation(activation_name)

        self.encoder = Encoder(
            state_dim,
            global_dim,
            list(encoder_hidden),
            latent_dim,
            act_factory,
            dropout,
            residual=residual,
        )
        self.dynamics = LatentDynamics(
            latent_dim,
            global_dim,
            list(dynamics_hidden),
            act_factory,
            dropout,
            residual=dynamics_residual,
            mlp_residual=residual,
        )
        self.decoder = Decoder(
            latent_dim,
            list(decoder_hidden),
            state_dim,
            act_factory,
            dropout,
            residual=residual,
        )

        if predict_delta:
            self._gentle_init_decoder()

    def _gentle_init_decoder(self) -> None:
        """Initialize decoder output near zero for stable delta predictions."""
        try:
            with torch.no_grad():
                out_layer = self.decoder.last_linear()
                nn.init.zeros_(out_layer.bias)
                out_layer.weight.mul_(0.1)
        except (AttributeError, IndexError):
            pass

    def forward_step(
        self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-step forward for autoregressive training.

        Args:
            y_i: [B, S] current state
            dt_norm: [B] or scalar normalized time step
            g: [B, G] global parameters

        Returns:
            y_next: [B, S] predicted next state
        """
        z_i = self.encoder(y_i, g)
        z_j = self.dynamics.forward_step(z_i, dt_norm, g)
        y_pred = self.decoder(z_j)
        return y_pred + y_i if self.predict_delta else y_pred

    def forward(
        self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward over K time steps.

        Args:
            y_i: [B, S] initial state
            dt_norm: [B, K] or [B, K, 1] normalized time steps
            g: [B, G] global parameters

        Returns:
            y_pred: [B, K, S] predicted sequence
        """
        z_i = self.encoder(y_i, g)
        z_j = self.dynamics(z_i, dt_norm, g)
        y_pred = self.decoder(z_j)
        return y_pred + y_i.unsqueeze(1) if self.predict_delta else y_pred


class FlowMapMLP(nn.Module):
    """Direct MLP flow-map: (y_i, dt, g) -> y_j."""

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
    ):
        super().__init__()
        self.S, self.G = state_dim, global_dim
        self.predict_delta = predict_delta

        act_factory = get_activation(activation_name)
        input_dim = state_dim + 1 + global_dim  # y + dt + g

        if residual:
            self.network = ResidualMLP(
                input_dim, list(hidden_dims), state_dim, act_factory, dropout
            )
        else:
            self.network = MLP(
                input_dim, list(hidden_dims), state_dim, act_factory, dropout
            )

        if predict_delta:
            self._gentle_init_output()

    def _gentle_init_output(self) -> None:
        """Initialize output near zero for stable delta predictions."""
        try:
            with torch.no_grad():
                out = self.network.last_linear()
                nn.init.zeros_(out.bias)
                out.weight.mul_(0.1)
        except (AttributeError, IndexError):
            pass

    def forward_step(
        self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-step forward for autoregressive training.

        Args:
            y_i: [B, S] current state
            dt_norm: [B] or scalar normalized time step
            g: [B, G] global parameters

        Returns:
            y_next: [B, S] predicted next state
        """
        B = y_i.shape[0]

        # Ensure dt has correct shape
        if dt_norm.ndim == 0:
            dt = dt_norm.expand(B, 1)
        elif dt_norm.ndim == 1:
            dt = dt_norm.view(B, 1)
        else:
            dt = dt_norm.view(B, 1)

        x = torch.cat([y_i, dt.to(y_i.dtype), g], dim=-1)
        y_pred = self.network(x)

        return y_pred + y_i if self.predict_delta else y_pred

    def forward(
        self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward over K time steps.

        Args:
            y_i: [B, S] initial state
            dt_norm: [B, K] or [B, K, 1] normalized time steps
            g: [B, G] global parameters

        Returns:
            y_pred: [B, K, S] predicted sequence
        """
        dt = self._normalize_dt_shape(dt_norm)
        B, K = y_i.shape[0], dt.shape[1]

        y_exp = y_i.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        x = torch.cat([y_exp, dt.unsqueeze(-1).to(y_exp.dtype), g_exp], dim=-1)
        y_pred = self.network(x)

        return y_pred + y_i.unsqueeze(1) if self.predict_delta else y_pred

    @staticmethod
    def _normalize_dt_shape(dt: torch.Tensor) -> torch.Tensor:
        if dt.ndim == 3:
            return dt.squeeze(-1)
        if dt.ndim == 1:
            return dt.unsqueeze(1)
        return dt


# ==============================================================================
# Model Factory
# ==============================================================================


def create_model(
    config: Dict[str, Any],
    *,
    state_dim: Optional[int] = None,
    global_dim: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """
    Build model from configuration.

    Args:
        config: Configuration dictionary.
        state_dim: Override state dimension (uses config if None).
        global_dim: Override global dimension (uses config if None).
        logger: Optional logger for info messages.

    Config structure:
        data.species_variables: List[str]  # Determines state dimension S
        data.global_variables: List[str]   # Determines global dimension G

        model.type: "mlp" or "autoencoder"
        model.activation: str
        model.dropout: float
        model.predict_delta: bool

        model.mlp.hidden_dims: List[int]
        model.mlp.residual: bool

        model.autoencoder.latent_dim: int
        model.autoencoder.encoder_hidden: List[int]
        model.autoencoder.dynamics_hidden: List[int]
        model.autoencoder.decoder_hidden: List[int]
        model.autoencoder.residual: bool
        model.autoencoder.dynamics_residual: bool
    """
    log = logger or logging.getLogger(__name__)

    # Get dimensions
    if state_dim is None or global_dim is None:
        data_cfg = config.get("data", {})
        species_vars = list(data_cfg.get("species_variables") or [])
        global_vars = list(data_cfg.get("global_variables", []))

        if not species_vars:
            raise KeyError("data.species_variables must be non-empty")

        state_dim = state_dim or len(species_vars)
        global_dim = global_dim if global_dim is not None else len(global_vars)

    S, G = state_dim, global_dim
    log.info(f"Model dimensions: S={S} species, G={G} globals")

    # Model config
    mcfg = config.get("model", {})
    model_type = str(mcfg.get("type", "mlp")).lower().strip()

    # Handle legacy "mlp_only" flag
    if mcfg.get("mlp_only", False):
        model_type = "mlp"

    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))
    predict_delta = bool(mcfg.get("predict_delta", True))

    if model_type == "mlp":
        mlp_cfg = mcfg.get("mlp", {})
        hidden_dims = list(mlp_cfg.get("hidden_dims", [512, 512, 512, 512]))
        residual = bool(mlp_cfg.get("residual", True))

        log.info(f"Creating FlowMapMLP: hidden={hidden_dims}, residual={residual}")

        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout=dropout,
            residual=residual,
            predict_delta=predict_delta,
        )

    elif model_type == "autoencoder":
        ae_cfg = mcfg.get("autoencoder", {})
        latent_dim = int(ae_cfg.get("latent_dim", 256))
        encoder_hidden = list(ae_cfg.get("encoder_hidden", [512, 512]))
        dynamics_hidden = list(ae_cfg.get("dynamics_hidden", [512, 512]))
        decoder_hidden = list(ae_cfg.get("decoder_hidden", [512, 512]))
        residual = bool(ae_cfg.get("residual", True))
        dynamics_residual = bool(ae_cfg.get("dynamics_residual", True))

        log.info(
            f"Creating FlowMapAutoencoder: latent={latent_dim}, "
            f"encoder={encoder_hidden}, dynamics={dynamics_hidden}, decoder={decoder_hidden}"
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
        )

    else:
        raise ValueError(
            f"Unknown model.type: '{model_type}'. Use 'mlp' or 'autoencoder'."
        )
