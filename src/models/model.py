#!/usr/bin/env python3
"""
Linear Latent Network (LiLaN) model implementation with monotone time-warp.


Formulation:
    y(t) = E(x0, p) + τ(t, x0, p) * C(x0, p)   # Linear latent dynamics
    x(t) = D(y(t))                             # Decode to physical space
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _get_activation_factory(name: str):
    """Return a factory that constructs an activation module for a given name."""
    name = (name or "gelu").lower()
    activations = {
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(),
        "tanh": lambda: nn.Tanh(),
        "relu": lambda: nn.ReLU(),
        "elu": lambda: nn.ELU(),
    }
    if name not in activations:
        logging.getLogger(__name__).warning(f"Unknown activation '{name}', defaulting to GELU")
        return lambda: nn.GELU()
    return activations[name]


def _get_xavier_gain(activation_name: str) -> float:
    """Gain to use with Xavier initialization for the given activation."""
    gain_map = {
        "relu": nn.init.calculate_gain("relu"),
        "tanh": nn.init.calculate_gain("tanh"),
        # No direct calculate_gain support; use 1.0
        "gelu": 1.0,
        "silu": 1.0,
        "elu": 1.0,
    }
    return gain_map.get(activation_name.lower(), 1.0)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: List[int],
    activation_factory,
    activation_name: str,
    dropout_p: float = 0.0,
    use_layernorm: bool = True,
    final_layer_scale: float = 1.0,
) -> nn.Sequential:
    """
    Build an MLP with blocks: Linear -> (LayerNorm) -> Activation -> (Dropout) for hidden layers.
    The final layer is Linear only. Xavier init is used throughout; the final layer can be downscaled.
    """
    layers = [input_dim] + hidden_layers + [output_dim]
    modules: List[nn.Module] = []

    xavier_gain = _get_xavier_gain(activation_name)

    for i in range(len(layers) - 1):
        linear = nn.Linear(layers[i], layers[i + 1])

        # Initialize weights/bias
        if i < len(layers) - 2:
            nn.init.xavier_uniform_(linear.weight, gain=xavier_gain)
        else:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            if final_layer_scale != 1.0:
                with torch.no_grad():
                    linear.weight.mul_(final_layer_scale)
        nn.init.zeros_(linear.bias)
        modules.append(linear)

        # Hidden layers receive LN + Act + (Dropout)
        if i < len(layers) - 2:
            if use_layernorm:
                ln = nn.LayerNorm(layers[i + 1])
                nn.init.ones_(ln.weight)
                nn.init.zeros_(ln.bias)
                modules.append(ln)

            modules.append(activation_factory())

            if dropout_p > 0.0:
                modules.append(nn.Dropout(dropout_p))

    return nn.Sequential(*modules)


class LinearLatentNetwork(nn.Module):
    """
    Linear Latent Network (LiLaN) with a monotone time-warp implemented
    as the cumulative integral of a strictly positive speed.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Data dimensions
        data_cfg = config["data"]
        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.num_targets = len(data_cfg.get("target_species_variables", data_cfg["species_variables"]))

        # Model hyperparameters
        model_cfg = config["model"]
        self.latent_dim = int(model_cfg.get("latent_dim", 64))
        if self.latent_dim < self.num_targets:
            raise ValueError(f"latent_dim ({self.latent_dim}) must be >= num_targets ({self.num_targets})")

        activation_name = str(model_cfg.get("activation", "gelu"))
        self.activation_factory = _get_activation_factory(activation_name)
        self.activation_name = activation_name
        self.dropout_p = float(model_cfg.get("dropout", 0.0))
        use_layernorm = bool(model_cfg.get("use_layernorm", True))
        final_scale = float(model_cfg.get("final_layer_scale", 0.5))
        
        # Store max_tau as instance variable from config
        self.max_tau = float(model_cfg.get("max_tau", 50.0))

        encoder_layers = list(model_cfg.get("encoder_layers", [256, 256, 128]))
        tau_layers = list(model_cfg.get("tau_layers", [256, 256, 128]))
        decoder_layers = list(model_cfg.get("decoder_layers", [128, 256, 256]))

        # Input dims
        encoder_input_dim = self.num_species + self.num_globals           # for E and C
        tau_input_dim = 1 + self.num_species + self.num_globals           # for speed net: [t, x0, p]
        decoder_input_dim = self.latent_dim

        # Networks
        # E: (x0, p) -> y0
        self.encoder_E = _build_mlp(
            encoder_input_dim, self.latent_dim, encoder_layers,
            self.activation_factory, self.activation_name,
            self.dropout_p, use_layernorm, final_scale
        )

        # C: (x0, p) -> c
        self.encoder_C = _build_mlp(
            encoder_input_dim, self.latent_dim, encoder_layers,
            self.activation_factory, self.activation_name,
            self.dropout_p, use_layernorm, final_scale
        )

        # Speed net for τ': (t, x0, p) -> speed >= 0 (enforced in forward via softplus)
        self.tau_speed_net = _build_mlp(
            tau_input_dim, self.latent_dim, tau_layers,
            self.activation_factory, self.activation_name,
            self.dropout_p, use_layernorm, final_scale
        )

        # Decoder: y -> targets
        self.decoder_D = _build_mlp(
            decoder_input_dim, self.num_targets, decoder_layers,
            self.activation_factory, self.activation_name,
            self.dropout_p, use_layernorm, final_scale
        )

        self.logger.info(
            "Created LiLaN model: latent_dim=%d, activation=%s, dropout=%.3f, max_tau=%.1f, n_params=%s",
            self.latent_dim, activation_name, self.dropout_p, self.max_tau,
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        inputs: [B, S + G + M]
            - first S entries: species initial conditions
            - next  G entries: global parameters
            - final M entries: time points (first is anchor)
        returns:
            predictions: [B, M, T]
            aux: {}
        """
        B, M = self._validate_inputs(inputs)

        # Parse inputs
        x0 = inputs[:, :self.num_species]                                  # [B, S]
        p = inputs[:, self.num_species:self.num_species + self.num_globals]# [B, G]
        t = inputs[:, self.num_species + self.num_globals:]                # [B, M]

        # Encoders
        # h := [x0, p]
        h = torch.cat([x0, p], dim=1)                                      # [B, S+G]
        y0 = self.encoder_E(h)                                             # [B, m]
        c  = self.encoder_C(h)                                             # [B, m]

        # Monotone time-warp τ via cumulative integral of positive speed
        # Prepare features for the speed network at each time point
        t_flat = t.reshape(B * M, 1)                                       # [B*M, 1]
        h_exp = h.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)         # [B*M, S+G]
        speed_in = torch.cat([t_flat, h_exp], dim=1)                       # [B*M, 1+S+G]

        raw_speed = self.tau_speed_net(speed_in).view(B, M, self.latent_dim)  # [B, M, m]
        speed = torch.nn.functional.softplus(raw_speed) + 1e-8             # strictly positive

        # Trapezoidal integration along time axis; anchor τ(t0) = 0
        dt = t[:, 1:] - t[:, :-1]                                          # [B, M-1]
        incr = 0.5 * (speed[:, 1:, :] + speed[:, :-1, :]) * dt.unsqueeze(-1)  # [B, M-1, m]
        tau = torch.zeros(B, 1, self.latent_dim, device=speed.device, dtype=speed.dtype)
        cumsum_tau = torch.cumsum(incr, dim=1)
        
        # Clamp tau to prevent numerical overflow using configured max_tau
        cumsum_tau = torch.clamp(cumsum_tau, max=self.max_tau)
        tau = torch.cat([tau, cumsum_tau], dim=1)                          # [B, M, m]

        # Latent trajectory and decoding
        y = y0.unsqueeze(1) + tau * c.unsqueeze(1)                         # [B, M, m]
        x = self.decoder_D(y.reshape(B * M, self.latent_dim))              # [B*M, T]
        predictions = x.view(B, M, self.num_targets)                       # [B, M, T]

        return predictions, {}

    def _validate_inputs(self, inputs: torch.Tensor) -> Tuple[int, int]:
        if inputs.dim() != 2:
            raise ValueError(f"Expected 2D input [B, S+G+M], got {inputs.shape}")
        expected_dim = self.num_species + self.num_globals
        if inputs.size(1) < expected_dim + 2:
            raise ValueError(f"Input dim {inputs.size(1)} < minimum required {expected_dim + 2} ")
        B = inputs.size(0)
        M = inputs.size(1) - expected_dim
        return B, M


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Factory that builds, dtypes, moves, and optionally compiles the model."""
    model = LinearLatentNetwork(config)

    # Dtype handling
    dtype_str = str(config.get("system", {}).get("dtype", "float32")).lower()
    if device.type == "cpu" and dtype_str in ("float16", "bfloat16"):
        logging.getLogger(__name__).warning("Half/bfloat16 on CPU is not recommended; using float32")
        dtype_str = "float32"

    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16" and device.type == "cuda":
        model = model.half()
    elif dtype_str == "bfloat16" and device.type == "cuda":
        model = model.bfloat16()

    model = model.to(device)

    # Optional torch.compile
    sys_cfg = config.get("system", {})
    if sys_cfg.get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            mode = sys_cfg.get("compile_mode", "default")
            model = torch.compile(model, mode=mode)
            logging.getLogger(__name__).info("Model compilation successful (mode=%s)", mode)
        except Exception as e:
            logging.getLogger(__name__).warning("Model compilation failed: %s. Running in eager mode.", e)

    return model
