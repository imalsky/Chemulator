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
    - Delta prediction mode (predict change rather than absolute state)
    - Gentle output initialization for stable early training
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ==============================================================================
# dt Shape Utilities
# ==============================================================================


def normalize_dt_shape(
    dt: torch.Tensor,
    batch_size: int,
    seq_len: int,
    context: str = "forward",
) -> torch.Tensor:
    """
    Normalize dt tensor to canonical shape [B, K] with explicit validation.

    This utility centralizes the dt shape handling logic that was previously
    duplicated across multiple model methods. It handles various input formats
    that users might provide and validates the result.

    Args:
        dt: Input dt tensor. Supported shapes:
            - scalar: constant dt for all samples and steps
            - [B]: constant dt per sample, broadcast across steps
            - [K]: constant dt schedule, broadcast across batch
            - [B, K]: per-sample per-step dt (canonical form)
            - [B, K, 1]: same as [B, K], squeezed
        batch_size: Expected batch dimension B
        seq_len: Expected sequence length K
        context: String describing calling context for error messages

    Returns:
        dt_seq: Tensor of shape [B, K]

    Raises:
        ValueError: If dt shape cannot be normalized to [B, K] unambiguously

    Examples:
        >>> dt = torch.tensor(0.5)  # scalar
        >>> normalize_dt_shape(dt, B=32, K=100)  # -> [32, 100] filled with 0.5

        >>> dt = torch.randn(32, 100)  # already canonical
        >>> normalize_dt_shape(dt, B=32, K=100)  # -> [32, 100] unchanged
    """
    B, K = batch_size, seq_len

    if dt.ndim == 0:
        return dt.view(1, 1).expand(B, K)

    if dt.ndim == 1:
        if dt.shape[0] == B and B != K:
            return dt.view(B, 1).expand(B, K)
        elif dt.shape[0] == K and B != K:
            return dt.view(1, K).expand(B, K)
        elif dt.shape[0] == B == K:
            warnings.warn(
                f"[{context}] Ambiguous 1D dt of length {B} where B==K. "
                "Interpreting as per-step schedule [K] (broadcast across batch). "
                "To disambiguate, pass dt with shape [B, 1] for per-sample constants "
                "or [1, K] / [B, K] for schedules.",
                RuntimeWarning,
                stacklevel=2,
            )
            return dt.view(1, K).expand(B, K)
        else:
            raise ValueError(
                f"[{context}] Cannot normalize 1D dt with shape {dt.shape} "
                f"to [B={B}, K={K}]. Expected length B or K."
            )

    if dt.ndim == 2:
        if dt.shape == (B, K):
            return dt
        elif dt.shape == (K, B):
            return dt.t()
        else:
            raise ValueError(
                f"[{context}] Cannot normalize 2D dt with shape {dt.shape} "
                f"to [B={B}, K={K}]. Shape mismatch."
            )

    if dt.ndim == 3 and dt.shape[-1] == 1:
        squeezed = dt.squeeze(-1)
        if squeezed.shape == (B, K):
            return squeezed
        raise ValueError(
            f"[{context}] Cannot normalize 3D dt with shape {dt.shape} "
            f"to [B={B}, K={K}]. After squeeze: {squeezed.shape}."
        )

    raise ValueError(
        f"[{context}] Unsupported dt shape: {dt.shape}. "
        f"Expected scalar, [B], [K], [B,K], or [B,K,1]."
    )


# ==============================================================================
# Activation Factory
# ==============================================================================


def get_activation(name: str) -> Callable[[], nn.Module]:
    """
    Get activation function factory by name.

    Returns a callable that creates a new activation instance each time,
    which is necessary for proper module registration in nn.Sequential.

    Args:
        name: Activation function name (case-insensitive). Options:
            - "relu": Standard ReLU
            - "gelu": Gaussian Error Linear Unit
            - "silu" / "swish": Sigmoid Linear Unit (recommended for dynamics)
            - "tanh": Hyperbolic tangent
            - "leaky_relu": LeakyReLU with slope 0.1
            - "elu": Exponential Linear Unit

    Returns:
        Factory callable that creates activation module instances

    Raises:
        ValueError: If activation name is not recognized
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
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Available options: {list(factories.keys())}"
        )

    return factories[name]


# ==============================================================================
# Core Networks
# ==============================================================================


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron with configurable architecture.

    A feedforward network with hidden layers, activation functions,
    and optional dropout. No residual connections.

    Architecture:
        input -> [Linear -> Activation -> Dropout?] x N -> Linear -> output

    Args:
        input_dim: Dimension of input features
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Dimension of output features
        activation_factory: Callable that creates activation modules
        dropout_p: Dropout probability (0 = no dropout)
    """

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
        """Forward pass through the MLP."""
        return self.network(x)

    def last_linear(self) -> nn.Linear:
        """Return the final linear layer for custom initialization."""
        return self.network[-1]


class ResidualMLP(nn.Module):
    """
    MLP with residual (skip) connections between layers.

    Residual connections help with gradient flow in deeper networks
    and allow the network to learn identity mappings more easily.

    Architecture for each hidden layer:
        h_{i+1} = activation(Linear(h_i)) + proj(h_i)

    where proj is either identity (if dims match) or a linear projection.

    Args:
        input_dim: Dimension of input features
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Dimension of output features
        activation_factory: Callable that creates activation modules
        dropout_p: Dropout probability (0 = no dropout)
    """

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
        """Return the final linear layer for custom initialization."""
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
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
    """
    Encoder network: maps (state, globals) to latent representation.

    Args:
        state_dim: Dimension of state vector S
        global_dim: Dimension of global parameters G
        hidden_dims: Hidden layer dimensions for encoder MLP
        latent_dim: Dimension of latent space Z
        activation_factory: Callable that creates activation modules
        dropout_p: Dropout probability
        residual: If True, use ResidualMLP; otherwise standard MLP
    """

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
        """
        Encode state and globals to latent representation.

        Args:
            y: State vector [B, S]
            g: Global parameters [B, G]

        Returns:
            z: Latent representation [B, Z]
        """
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
        return self.network(x)


class LatentDynamics(nn.Module):
    """
    Dynamics network in latent space: evolves latent state forward in time.

    Given current latent state z, timestep dt, and globals g, predicts
    the next latent state. Supports residual dynamics (predict delta z).

    Args:
        latent_dim: Dimension of latent space Z
        global_dim: Dimension of global parameters G
        hidden_dims: Hidden layer dimensions
        activation_factory: Callable that creates activation modules
        dropout_p: Dropout probability
        residual: If True, predict Δz and add to input (z_next = z + Δz)
        mlp_residual: If True, use ResidualMLP internally
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
        Single-step forward for efficient autoregressive training.

        Args:
            z: Current latent state [B, Z]
            dt_norm: Normalized timestep [B] or scalar
            g: Global parameters [B, G]

        Returns:
            z_next: Next latent state [B, Z]
        """
        B = z.shape[0]

        if dt_norm.ndim == 0:
            dt = dt_norm.expand(B, 1)
        elif dt_norm.ndim == 1:
            dt = dt_norm.view(B, 1)
        else:
            dt = dt_norm.view(B, 1)

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

        Note: This computes independent single-step predictions for each
        timestep, NOT autoregressive rollout. Use forward_step in a loop
        for true autoregressive behavior.

        Args:
            z: Initial latent state [B, Z]
            dt_norm: Timestep schedule (scalar/[B] require seq_len for K>1)
            g: Global parameters [B, G]
            seq_len: Override sequence length

        Returns:
            z_seq: Latent states [B, K, Z] for each timestep
        """
        B = z.shape[0]

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(
                        "Ambiguous dt shape [B]. Provide seq_len explicitly, or reshape dt to "
                        "[B, 1] for per-sample constants, or [K] for a schedule."
                    )
                K = dt_norm.shape[0]
            elif dt_norm.ndim == 2:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            else:
                raise ValueError(
                    f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for sequence."
                )
        if K < 1:
            raise ValueError(f"seq_len must be >= 1, got {K}")
        dt = normalize_dt_shape(dt_norm, B, K, context="LatentDynamics.forward")

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        x = torch.cat([z_exp, dt.unsqueeze(-1).to(z_exp.dtype), g_exp], dim=-1)
        dz = self.network(x)

        return z_exp + dz if self.residual else dz


class Decoder(nn.Module):
    """
    Decoder network: maps latent representation back to state space.

    Args:
        latent_dim: Dimension of latent space Z
        hidden_dims: Hidden layer dimensions
        state_dim: Dimension of output state S
        activation_factory: Callable that creates activation modules
        dropout_p: Dropout probability
        residual: If True, use ResidualMLP
    """

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
        """Return the final linear layer for custom initialization."""
        return self.network.last_linear()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to state.

        Args:
            z: Latent representation [B, Z] or [B, K, Z]

        Returns:
            y: Decoded state [B, S] or [B, K, S]
        """
        return self.network(z)


# ==============================================================================
# Full Models
# ==============================================================================


class FlowMapAutoencoder(nn.Module):
    """
    Flow-map autoencoder: Encoder -> LatentDynamics -> Decoder.

    This architecture projects the state to a latent space where the
    dynamics may be simpler to learn, evolves the latent state, then
    decodes back to physical state space.

    Args:
        state_dim: Dimension of state vector S (number of species)
        global_dim: Dimension of global parameters G
        latent_dim: Dimension of latent space Z
        encoder_hidden: Hidden dims for encoder
        dynamics_hidden: Hidden dims for latent dynamics
        decoder_hidden: Hidden dims for decoder
        activation_name: Activation function name
        dropout: Dropout probability
        residual: Use residual connections in encoder/decoder
        dynamics_residual: Use residual connection for dynamics (z + Δz)
        predict_delta: If True, decoder output is added to input state
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
        """Initialize decoder output layer near zero for stable delta predictions."""
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
        Single-step forward for efficient autoregressive training.

        Pipeline: y_i -> encode -> evolve latent -> decode -> y_next

        Args:
            y_i: Current state [B, S]
            dt_norm: Normalized timestep [B] or scalar
            g: Global parameters [B, G]

        Returns:
            y_next: Predicted next state [B, S]
        """
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
        """
        Autoregressive rollout over K time steps.

        Args:
            y_i: Initial state [B, S]
            dt_norm: Timestep schedule (various shapes supported)
            g: Global parameters [B, G]
            seq_len: Override sequence length

        Returns:
            y_pred: Predicted sequence [B, K, S]
        """
        if y_i.ndim != 2:
            raise ValueError(f"Expected y_i shape [B, S], got {tuple(y_i.shape)}")

        B, S = y_i.shape

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(
                        "Ambiguous dt shape [B]. Provide seq_len explicitly, or reshape dt to "
                        "[B, 1] for per-sample constants, or [K] for a schedule."
                    )
                K = dt_norm.shape[0]
            elif dt_norm.ndim == 2:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            else:
                raise ValueError(
                    f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for rollout."
                )
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
    """
    Direct MLP flow-map: predicts next state from (state, dt, globals).

    This is the simpler architecture that directly learns the mapping
    y_{t+1} = F(y_t, dt, g) without an intermediate latent space.

    Input at each step: concatenation of [state y (S), dt (1), globals g (G)]
    Output: next state y_{t+1} (S) or delta Δy (S) if predict_delta=True

    Args:
        state_dim: Dimension of state vector S (number of species)
        global_dim: Dimension of global parameters G
        hidden_dims: Hidden layer dimensions
        activation_name: Activation function name
        dropout: Dropout probability
        residual: Use ResidualMLP (recommended for deep networks)
        predict_delta: If True, predict Δy and compute y_next = y + Δy
    """

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
        input_dim = state_dim + 1 + global_dim

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
        """Initialize output layer near zero for stable delta predictions."""
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
        Single-step forward for efficient autoregressive training.

        Args:
            y_i: Current state [B, S]
            dt_norm: Normalized timestep [B] or scalar in [0, 1]
            g: Global parameters [B, G]

        Returns:
            y_next: Predicted next state [B, S]
        """
        B = y_i.shape[0]

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
        self,
        y_i: torch.Tensor,
        dt_norm: torch.Tensor,
        g: torch.Tensor,
        *,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout over K time steps.

        Args:
            y_i: Initial state [B, S]
            dt_norm: Timestep schedule (various shapes supported)
            g: Global parameters [B, G]
            seq_len: Override sequence length

        Returns:
            y_pred: Predicted sequence [B, K, S]
        """
        if y_i.ndim != 2:
            raise ValueError(f"Expected y_i shape [B, S], got {tuple(y_i.shape)}")

        B, S = y_i.shape

        if seq_len is not None:
            K = int(seq_len)
        else:
            if dt_norm.ndim == 0:
                K = 1
            elif dt_norm.ndim == 1:
                if dt_norm.shape[0] == B and B > 1:
                    raise ValueError(
                        "Ambiguous dt shape [B]. Provide seq_len explicitly, or reshape dt to "
                        "[B, 1] for per-sample constants, or [K] for a schedule."
                    )
                K = dt_norm.shape[0]
            elif dt_norm.ndim == 2:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                K = dt_norm.shape[1] if dt_norm.shape[0] == B else dt_norm.shape[0]
            else:
                raise ValueError(
                    f"Unsupported dt_norm shape {tuple(dt_norm.shape)} for rollout."
                )
        if K < 1:
            raise ValueError(f"seq_len must be >= 1, got {K}")
        dt_seq = normalize_dt_shape(dt_norm, B, K, context="FlowMapMLP.forward")

        y_pred = torch.empty(B, K, S, device=y_i.device, dtype=y_i.dtype)

        state = y_i
        for t in range(K):
            state = self.forward_step(state, dt_seq[:, t], g)
            y_pred[:, t, :] = state

        return y_pred


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
    Build model from configuration dictionary.

    Factory function that creates either FlowMapMLP or FlowMapAutoencoder
    based on config settings.

    Args:
        config: Configuration dictionary with model and data sections
        state_dim: Override state dimension (uses config if None)
        global_dim: Override global dimension (uses config if None)
        logger: Optional logger for info messages

    Returns:
        Configured model (FlowMapMLP or FlowMapAutoencoder)

    Raises:
        KeyError: If required config keys are missing
        ValueError: If model type is not recognized
    """
    log = logger or logging.getLogger(__name__)

    if state_dim is None or global_dim is None:
        data_cfg = config.get("data", {})
        species_vars = list(data_cfg.get("species_variables") or [])
        global_vars = list(data_cfg.get("global_variables", []))

        if not species_vars:
            raise KeyError(
                "data.species_variables must be non-empty. "
                "This defines the state dimension S."
            )

        state_dim = state_dim or len(species_vars)
        global_dim = global_dim if global_dim is not None else len(global_vars)

    S, G = state_dim, global_dim
    log.info(f"Model dimensions: S={S} species, G={G} globals")

    mcfg = config.get("model", {})
    model_type = str(mcfg.get("type", "mlp")).lower().strip()

    if mcfg.get("mlp_only", False):
        model_type = "mlp"

    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))
    predict_delta = bool(mcfg.get("predict_delta", True))

    if model_type == "mlp":
        mlp_cfg = mcfg.get("mlp", {})
        hidden_dims = list(mlp_cfg.get("hidden_dims", [512, 512, 512, 512]))
        residual = bool(mlp_cfg.get("residual", True))

        log.info(
            f"Creating FlowMapMLP: hidden={hidden_dims}, "
            f"residual={residual}, predict_delta={predict_delta}"
        )

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
            f"encoder={encoder_hidden}, dynamics={dynamics_hidden}, "
            f"decoder={decoder_hidden}"
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
            f"Unknown model.type: '{model_type}'. "
            f"Supported types: 'mlp', 'autoencoder'."
        )
