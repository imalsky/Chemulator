#!/usr/bin/env python3
"""
Flow-map Autoencoder Model Architecture
========================================
Replaces DeepONet with an autoencoder approach for flow-map prediction.

Architecture Components:
- Encoder: Maps initial state + globals to latent space
- LatentDynamics: Evolves latent state forward in time (conditioned on dt and globals)
- Decoder: Maps from latent space back to physical space

Features:
- Supports VAE mode with KL divergence regularization
- Maintains compatibility with existing delta/softmax heads
- Handles multi-time predictions (K>1)
- Preserves normalization statistics for loss computation
"""

from __future__ import annotations

import math
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Activations ----------------------------------

ACTIVATION_ALIASES = {
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "swish": "SiLU",
    "tanh": "Tanh",
}


def get_activation(name: str) -> nn.Module:
    """Create activation function from name."""
    name_lower = name.lower()
    class_name = ACTIVATION_ALIASES.get(name_lower, name)

    activation_class = getattr(nn, class_name, None)

    if activation_class is None or not issubclass(activation_class, nn.Module):
        supported = sorted(set(ACTIVATION_ALIASES.keys()) |
                           {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "ELU"})
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Supported activations: {', '.join(supported)}"
        )

    if class_name == "LeakyReLU":
        return activation_class(negative_slope=0.01)

    return activation_class()


# --------------------------------- MLP ---------------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron with dropout support."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: nn.Module,
            dropout_p: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation.__class__())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------- Encoder Network --------------------------------


class Encoder(nn.Module):
    """
    Encoder network: [y_i, g] -> z (latent representation)

    Supports both standard AE and VAE modes.
    """

    def __init__(
            self,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            latent_dim: int,
            activation: nn.Module,
            dropout_p: float = 0.0,
            vae_mode: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.global_dim = global_dim
        self.latent_dim = latent_dim
        self.vae_mode = vae_mode

        input_dim = state_dim + global_dim
        output_dim = latent_dim * 2 if vae_mode else latent_dim

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(
            self,
            y: torch.Tensor,
            g: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            y: State tensor [B, S]
            g: Global parameters [B, G]

        Returns:
            z: Latent representation [B, Z]
            kl_loss: KL divergence loss (None if not VAE)
        """
        x = torch.cat([y, g], dim=-1)
        out = self.network(x)

        if not self.vae_mode:
            return out, None

        # VAE: split into mean and log-variance
        mu, logvar = torch.chunk(out, 2, dim=-1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # KL divergence loss (per batch, will be averaged by trainer)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return z, kl_loss


# ------------------------- Latent Dynamics Network ---------------------------


class LatentDynamics(nn.Module):
    """
    Dynamics network in latent space: (z_i, dt, g) -> z_j

    Learns how latent states evolve over time.
    """

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
        self.residual = residual

        # Input: [z, dt, g]
        input_dim = latent_dim + 1 + global_dim
        output_dim = latent_dim

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

        # Optional skip connection
        if residual:
            # Small initialization for residual branch
            with torch.no_grad():
                self.network.network[-1].weight.mul_(0.1)
                self.network.network[-1].bias.zero_()

    def forward(
            self,
            z: torch.Tensor,
            dt: torch.Tensor,
            g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: Initial latent state [B, Z]
            dt: Time differences [B, K] or [B, K, 1]
            g: Global parameters [B, G]

        Returns:
            z_future: Future latent states [B, K, Z]
        """
        B = z.shape[0]

        # Handle dt shape variations
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)  # [B, K, 1] -> [B, K]
        elif dt.ndim == 1:
            dt = dt.unsqueeze(1)  # [B] -> [B, 1]

        K = dt.shape[1]

        # Expand inputs to handle K time points
        z_exp = z.unsqueeze(1).expand(B, K, -1)  # [B, K, Z]
        g_exp = g.unsqueeze(1).expand(B, K, -1)  # [B, K, G]
        dt_exp = dt.unsqueeze(-1)  # [B, K, 1]

        # Concatenate inputs
        x = torch.cat([z_exp, dt_exp, g_exp], dim=-1)  # [B, K, Z+1+G]

        # Apply dynamics network
        delta_z = self.network(x)  # [B, K, Z]

        if self.residual:
            return z_exp + delta_z
        else:
            return delta_z


# ---------------------------- Decoder Network --------------------------------


class Decoder(nn.Module):
    """
    Decoder network: z -> y (physical space)
    """

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
        """
        Args:
            z: Latent representation [B, K, Z]

        Returns:
            y: Decoded states [B, K, S]
        """
        return self.network(z)


# -------------------------- Main Autoencoder Model ---------------------------


class FlowMapAutoencoder(nn.Module):
    """
    Flow-map Autoencoder: (y_i, g, dt_norm) -> y_j

    Architecture:
    - Encoder: [y_i, g] -> z_i
    - Dynamics: (z_i, dt, g) -> z_j
    - Decoder: z_j -> y_j

    Maintains compatibility with existing training pipeline by supporting:
    - Target index selection
    - Delta prediction modes
    - SoftMax head for conservation
    - Log-space normalization statistics
    """

    def __init__(
            self,
            *,
            state_dim_in: int,
            state_dim_out: int,
            global_dim: int,
            latent_dim: int,
            encoder_hidden: Sequence[int],
            dynamics_hidden: Sequence[int],
            decoder_hidden: Sequence[int],
            activation_name: str = "gelu",
            dropout: float = 0.0,
            vae_mode: bool = False,
            dynamics_residual: bool = True,
            # Compatibility with existing pipeline
            predict_delta: bool = True,
            predict_delta_log_phys: bool = False,
            target_idx: Optional[torch.Tensor] = None,
            target_log_mean: Optional[Sequence[float]] = None,
            target_log_std: Optional[Sequence[float]] = None,
            # SoftMax head parameters
            softmax_head: bool = False,
            allow_partial_simplex: bool = False,
    ):
        super().__init__()

        # Store dimensions
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        # Store configuration
        self.vae_mode = bool(vae_mode)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        # Validation (same as original DeepONet)
        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError(
                "softmax_head=True is incompatible with residual modes"
            )

        if self.softmax_head and (self.S_out != self.S_in) and not self.allow_partial_simplex:
            raise RuntimeError(
                f"softmax_head=True with S_out != S_in will renormalize "
                f"within subset. Set allow_partial_simplex=True to accept this."
            )

        # Register target indices
        if target_idx is None:
            self.target_idx = None
        else:
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx)

        # Register log statistics if needed
        if self.predict_delta_log_phys or self.softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError(
                    "target_log_mean and target_log_std required for "
                    "predict_delta_log_phys or softmax_head"
                )

            log_mean = torch.tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.tensor(target_log_std, dtype=torch.float32)
            self.register_buffer("log_mean", log_mean)
            self.register_buffer("log_std", torch.clamp(log_std, min=1e-10))
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None

        # Create activation
        act = get_activation(activation_name)

        # Build encoder
        self.encoder = Encoder(
            state_dim=self.S_in,
            global_dim=self.G,
            hidden_dims=list(encoder_hidden),
            latent_dim=self.Z,
            activation=act,
            dropout_p=float(dropout),
            vae_mode=self.vae_mode,
        )

        # Build dynamics network
        self.dynamics = LatentDynamics(
            latent_dim=self.Z,
            global_dim=self.G,
            hidden_dims=list(dynamics_hidden),
            activation=act,
            dropout_p=float(dropout),
            residual=dynamics_residual,
        )

        # Build decoder
        self.decoder = Decoder(
            latent_dim=self.Z,
            hidden_dims=list(decoder_hidden),
            state_dim=self.S_out,
            activation=act,
            dropout_p=float(dropout),
        )

        # Initialize decoder for residual connections
        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                # Small decoder output for residual
                self.decoder.network.network[-1].weight.mul_(0.1)
                self.decoder.network.network[-1].bias.zero_()

        # Initialize KL loss tracker
        self.kl_loss = None

    def forward(
            self,
            y_i: torch.Tensor,  # [B,S_in]
            dt_norm: torch.Tensor,  # [B,K] or [B,K,1]
            g: torch.Tensor,  # [B,G]
    ) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Returns predictions in z-space (log-standard normalized).
        """
        B = g.shape[0]

        # Encode initial state
        z_i, self.kl_loss = self.encoder(y_i, g)  # [B, Z], kl_loss

        # Evolve in latent space
        z_j = self.dynamics(z_i, dt_norm, g)  # [B, K, Z]

        # Decode to physical space
        y_pred = self.decoder(z_j)  # [B, K, S_out]

        # Apply output heads (same logic as original DeepONet)
        if self.softmax_head:
            # SoftMax head for conservation
            log_p = F.log_softmax(y_pred, dim=-1)
            ln10 = self.ln10.to(dtype=log_p.dtype)
            log_mean = self.log_mean.to(dtype=log_p.dtype)
            log_std = self.log_std.to(dtype=log_p.dtype)
            log10_p = log_p / ln10
            y_pred = (log10_p - log_mean) / log_std

        elif self.predict_delta_log_phys:
            # Residual in log-physical space - FIX: proper dtype casting
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base_z = y_i.index_select(1, self.target_idx)
            else:
                base_z = y_i

            # Cast stats to match base_z dtype
            lm = self.log_mean.to(dtype=base_z.dtype)
            ls = self.log_std.to(dtype=base_z.dtype)

            # Convert to log10 space
            base_log = base_z * ls + lm  # [B, S_out]

            # Add network's predicted Δlog10(y)
            y_pred_log = base_log.unsqueeze(1) + y_pred  # [B, K, S_out]

            # Back to z-space (cast stats to match y_pred_log dtype)
            lm2 = self.log_mean.to(dtype=y_pred_log.dtype)
            ls2 = self.log_std.to(dtype=y_pred_log.dtype)
            y_pred = (y_pred_log - lm2) / ls2  # [B, K, S_out]

        elif self.predict_delta:
            # Standard residual in z-space
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base = y_i.index_select(1, self.target_idx)
            else:
                base = y_i

            y_pred = y_pred + base.unsqueeze(1)

        return y_pred

    def get_latent_representation(self, y_i: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Get latent encoding of a state (useful for analysis)."""
        z, _ = self.encoder(y_i, g)
        return z

    @torch.no_grad()
    def validate_conservation(self, y_pred_z: torch.Tensor) -> tuple:
        """
        Check if predictions satisfy conservation after denormalization.

        Args:
            y_pred_z: Predictions in z-space [B,K,S] or [B,S]

        Returns:
            (max_sum_error, min_value): Maximum deviation from sum=1 and minimum value
        """
        if not self.softmax_head:
            return float('nan'), float('nan')

        # z -> log10 -> physical
        log10_y = (y_pred_z.to(self.log_mean.dtype) * self.log_std) + self.log_mean
        y_phys = torch.pow(torch.tensor(10.0, dtype=log10_y.dtype, device=log10_y.device), log10_y)

        # Check sum-to-1
        sums = y_phys.sum(dim=-1)
        sum_error = (sums - 1.0).abs().max().item()

        # Check non-negativity
        min_val = y_phys.min().item()

        return sum_error, min_val

    @torch.no_grad()
    def check_stat_consistency(self, loss_log_mean: torch.Tensor, loss_log_std: torch.Tensor) -> None:
        """Verify model and loss normalization stats match (compatibility method)."""
        if not (getattr(self, "softmax_head", False) or getattr(self, "predict_delta_log_phys", False)):
            return

        if self.log_mean is None or self.log_std is None:
            raise RuntimeError("Model configured to use stats but buffers missing")

        # Check consistency (same as original)
        m_mu = self.log_mean.detach().cpu().reshape(-1)
        m_sig = self.log_std.detach().cpu().reshape(-1)
        l_mu = loss_log_mean.detach().cpu().reshape(-1)
        l_sig = loss_log_std.detach().cpu().reshape(-1)

        rtol, atol = 1e-6, 1e-9

        if not torch.allclose(m_mu, l_mu, rtol=rtol, atol=atol):
            diff = (m_mu - l_mu).abs()
            idx = int(diff.argmax())
            raise ValueError(f"log_mean mismatch at {idx}")

        if not torch.allclose(m_sig, l_sig, rtol=rtol, atol=atol):
            diff = (m_sig - l_sig).abs()
            idx = int(diff.argmax())
            raise ValueError(f"log_std mismatch at {idx}")


# ------------------------------ Factory --------------------------------------


def create_model(config: dict) -> FlowMapAutoencoder:
    """
    Build FlowMapAutoencoder from configuration dictionary.

    Maintains compatibility with existing configuration structure.
    """
    import json
    from pathlib import Path

    # Extract data configuration (same as original)
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))

    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    target_vars = list(data_cfg.get("target_species") or species_vars)

    # Create index mapping
    name_to_idx = {name: i for i, name in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    # Dimensions
    state_dim_in = len(species_vars)
    state_dim_out = len(target_vars)
    global_dim = len(global_vars)

    # Model configuration
    mcfg = config.get("model", {})

    # Autoencoder-specific parameters (with defaults)
    latent_dim = int(mcfg.get("latent_dim", 32))
    encoder_hidden = mcfg.get("encoder_hidden", [256, 128])
    dynamics_hidden = mcfg.get("dynamics_hidden", [256, 256])
    decoder_hidden = mcfg.get("decoder_hidden", [128, 256])
    vae_mode = bool(mcfg.get("vae_mode", False))
    dynamics_residual = bool(mcfg.get("dynamics_residual", True))

    # Common parameters
    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))

    # Head configuration (preserved from original)
    predict_delta = bool(mcfg.get("predict_delta", True))
    predict_delta_log_phys = bool(mcfg.get("predict_delta_log_phys", False))
    softmax_head = bool(mcfg.get("softmax_head", False))
    allow_partial_simplex = bool(mcfg.get("allow_partial_simplex", False))

    # Validation (same as original)
    if softmax_head:
        if predict_delta or predict_delta_log_phys:
            raise ValueError(
                "softmax_head=True requires predict_delta=False and "
                "predict_delta_log_phys=False"
            )
        if state_dim_out != state_dim_in and not allow_partial_simplex:
            raise ValueError(
                "softmax_head=True with subset species requires "
                "allow_partial_simplex=True"
            )

    # Load normalization statistics if needed
    need_stats = predict_delta_log_phys or softmax_head
    target_log_mean = None
    target_log_std = None

    if need_stats:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r") as f:
            manifest = json.load(f)
        stats = manifest["per_key_stats"]

        target_log_mean = []
        target_log_std = []
        for name in target_vars:
            if name not in stats:
                raise KeyError(f"Target species '{name}' not in normalization stats")
            s = stats[name]
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded normalization stats for {len(target_vars)} species")

    # Create model
    return FlowMapAutoencoder(
        state_dim_in=state_dim_in,
        state_dim_out=state_dim_out,
        global_dim=global_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        dynamics_hidden=dynamics_hidden,
        decoder_hidden=decoder_hidden,
        activation_name=activation,
        dropout=dropout,
        vae_mode=vae_mode,
        dynamics_residual=dynamics_residual,
        predict_delta=predict_delta,
        predict_delta_log_phys=predict_delta_log_phys,
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        softmax_head=softmax_head,
        allow_partial_simplex=allow_partial_simplex,
    )