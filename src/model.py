#!/usr/bin/env python3
"""
model.py

Flow-map autoencoder model:
  (y_i, dt_norm, g) -> y_j  in z-space (log-standard normalized)

Supports:
  - Residual in z-space (predict_delta)
  - Residual in physical log10 space (predict_delta_log_phys)
  - Optional simplex head (softmax_head) to output log10 probabilities, then z-normalize

Notes:
  - y is always z-space (log-standard normalized) throughout training.
  - "physical" means log10(y_phys) space (before standardization).
  - S_in == S_out is enforced: models always predict all species.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ----------------------------- Small helpers ----------------------------------

ACTIVATION_ALIASES = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    n = str(name).lower().strip()
    if n not in ACTIVATION_ALIASES:
        raise ValueError(f"Unknown activation '{name}'. Choices={sorted(ACTIVATION_ALIASES)}")
    return ACTIVATION_ALIASES[n]()


def _fresh_activation(act: nn.Module) -> nn.Module:
    """Create a new instance of the same activation type (avoid sharing state)."""
    return act.__class__()


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
        """Return the final linear layer (used for gentle initialization)."""
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

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state and globals to latent space.
        
        Returns (z, kl) where kl is always 0 (placeholder for future VAE support).
        """
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
        z = self.network(x)
        kl = torch.zeros((), device=z.device, dtype=z.dtype)
        return z, kl


class LatentDynamics(nn.Module):
    """Latent dynamics: (z, dt_norm, g) -> z_j (vectorized over K time steps)."""

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
        Evolve latent state z forward by dt_norm.
        
        Args:
            z: [B, Z] latent state
            dt_norm: [B, K] or [B, K, 1] or [B] normalized time steps
            g: [B, G] global parameters
            
        Returns:
            z_j: [B, K, Z] evolved latent states
        """
        # Normalize dt_norm shape to [B, K]
        if dt_norm.ndim == 3:
            if dt_norm.shape[-1] != 1:
                raise ValueError(f"dt_norm expected last dim=1 when 3D; got {tuple(dt_norm.shape)}")
            dt = dt_norm.squeeze(-1)
        elif dt_norm.ndim == 2:
            dt = dt_norm
        elif dt_norm.ndim == 1:
            dt = dt_norm.unsqueeze(1)
        else:
            raise ValueError(f"dt_norm must be 1D/2D/3D; got shape {tuple(dt_norm.shape)}")

        B = z.shape[0]
        K = dt.shape[1]

        # Expand z and g to match K time steps
        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        # Disable autocast for dt-conditioned dynamics to preserve precision
        device_type = z.device.type
        autocast_off = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in ("cuda", "cpu")
            else nullcontext()
        )

        with autocast_off:
            x = torch.cat(
                [
                    z_exp.to(torch.float32),
                    dt.to(torch.float32).unsqueeze(-1),
                    g_exp.to(torch.float32),
                ],
                dim=-1,
            )  # [B, K, Z+1+G] in FP32
            dz = self.network(x)  # [B, K, Z] in FP32

        # Cast back to original dtype
        dz = dz.to(dtype=z_exp.dtype)

        return (z_exp + dz) if self.residual else dz


class Decoder(nn.Module):
    """Decoder: z -> y (broadcast over K time steps)."""

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
        """Decode latent to state space. Accepts [B, K, Z] or [B, Z]."""
        return self.network(z)


class FlowMapAutoencoder(nn.Module):
    """
    Flow-map autoencoder: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Encoder -> LatentDynamics -> Decoder

    Output heads (mutually exclusive):
      - predict_delta: residual in z-space (y_pred = decoder_out + y_i)
      - predict_delta_log_phys: residual in log10-physical space, then re-standardize
      - softmax_head: simplex probabilities converted to log10 and standardized
      
    Note: S_in == S_out is enforced (all species predicted).
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
        log_mean: Optional[Sequence[float]] = None,
        log_std: Optional[Sequence[float]] = None,
        softmax_head: bool = False,
    ):
        super().__init__()

        # Dimensions (S_in == S_out enforced)
        self.S = int(state_dim)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        # Output head configuration
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)

        # Validate head configuration
        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise ValueError("softmax_head=True is incompatible with residual modes")

        # Stats for log-phys or softmax heads
        if self.predict_delta_log_phys or self.softmax_head:
            if log_mean is None or log_std is None:
                raise ValueError("log_mean/log_std required for log-phys or softmax head")
            self.register_buffer("log_mean", torch.as_tensor(log_mean, dtype=torch.float32))
            self.register_buffer(
                "log_std",
                torch.clamp(torch.as_tensor(log_std, dtype=torch.float32), min=1e-10),
            )
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
            self._logsoftmax = nn.LogSoftmax(dim=-1) if self.softmax_head else None
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None
            self._logsoftmax = None

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

        # Gentle initialization for residual heads (helps stability at start of training)
        if self.predict_delta or self.predict_delta_log_phys:
            self._gentle_init_decoder()

    def _gentle_init_decoder(self) -> None:
        """Initialize decoder output layer with small weights for residual learning."""
        try:
            with torch.no_grad():
                lin_out: nn.Linear = self.decoder.network.network[-1]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(0.1)
        except (AttributeError, IndexError) as e:
            logging.getLogger(__name__).debug(f"Gentle init skipped: {e}")

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict future state in z-space.
        
        Args:
            y_i: [B, S] current state in z-space
            dt_norm: [B, K] or [B, K, 1] normalized time steps
            g: [B, G] global parameters
            
        Returns:
            y_pred: [B, K, S] predicted states in z-space
        """
        # Encode current state
        z_i, _ = self.encoder(y_i, g)  # [B, Z]
        
        # Evolve in latent space
        z_j = self.dynamics(z_i, dt_norm, g)  # [B, K, Z]
        
        # Decode to state space
        y_pred = self.decoder(z_j)  # [B, K, S]

        # Apply output head
        if self.softmax_head:
            return self._apply_softmax_head(y_pred)
        elif self.predict_delta_log_phys:
            return self._apply_log_phys_residual(y_pred, y_i)
        elif self.predict_delta:
            return y_pred + y_i.unsqueeze(1)
        else:
            return y_pred

    def _apply_softmax_head(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert raw outputs to z-space via log-softmax -> log10 -> standardize."""
        log_p = self._logsoftmax(y_pred)  # [B, K, S]
        dtype = log_p.dtype
        log10_p = log_p / self.ln10.to(dtype)
        return (log10_p - self.log_mean.to(dtype)) / self.log_std.to(dtype)

    def _apply_log_phys_residual(self, y_pred: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        """Apply residual in log10-physical space, then re-standardize to z-space."""
        lm = self.log_mean.to(y_i.dtype)
        ls = self.log_std.to(y_i.dtype)
        
        # Convert base state from z-space to log10-physical
        base_log = y_i * ls + lm  # [B, S]
        
        # Add residual in log10-physical space
        y_pred_log = base_log.unsqueeze(1) + y_pred  # [B, K, S]
        
        # Re-standardize to z-space
        return (y_pred_log - lm) / ls


class FlowMapMLP(nn.Module):
    """
    MLP-only flow-map: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture:
      Single MLP over concatenated [y_i, dt_norm, g], vectorized over K time steps.

    Output heads (mutually exclusive):
      - predict_delta: residual in z-space
      - predict_delta_log_phys: residual in log10-physical space
      - softmax_head: simplex probabilities converted to log10 and standardized
      
    Note: S_in == S_out is enforced (all species predicted).
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
        log_mean: Optional[Sequence[float]] = None,
        log_std: Optional[Sequence[float]] = None,
        softmax_head: bool = False,
    ):
        super().__init__()

        # Dimensions (S_in == S_out enforced)
        self.S = int(state_dim)
        self.G = int(global_dim)

        # Output head configuration
        self.dynamics_residual = bool(dynamics_residual)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)

        # Validate head configuration
        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise ValueError("softmax_head=True is incompatible with residual modes")

        # Stats for log-phys or softmax heads
        if self.predict_delta_log_phys or self.softmax_head:
            if log_mean is None or log_std is None:
                raise ValueError("log_mean/log_std required for log-phys or softmax head")
            self.register_buffer("log_mean", torch.as_tensor(log_mean, dtype=torch.float32))
            self.register_buffer(
                "log_std",
                torch.clamp(torch.as_tensor(log_std, dtype=torch.float32), min=1e-10),
            )
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
            self._logsoftmax = nn.LogSoftmax(dim=-1) if self.softmax_head else None
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None
            self._logsoftmax = None

        # Main MLP
        act = get_activation(activation_name)
        inp = self.S + 1 + self.G  # state + dt + globals

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

        # Gentle initialization for residual heads
        if self.predict_delta or self.predict_delta_log_phys:
            self._gentle_init_output()

    def _gentle_init_output(self) -> None:
        """Initialize output layer with small weights for residual learning."""
        try:
            with torch.no_grad():
                if isinstance(self.network, ResidualMLP):
                    lin_out = self.network.last_linear()
                else:
                    lin_out = self.network.network[-1]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(0.1)
        except (AttributeError, IndexError) as e:
            logging.getLogger(__name__).debug(f"Gentle init skipped: {e}")

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict future state in z-space.
        
        Args:
            y_i: [B, S] current state in z-space
            dt_norm: [B, K] or [B, K, 1] or [B] normalized time steps
            g: [B, G] global parameters
            
        Returns:
            y_pred: [B, K, S] predicted states in z-space
        """
        # Normalize dt_norm shape to [B, K]
        if dt_norm.ndim == 3:
            if dt_norm.shape[-1] != 1:
                raise ValueError(f"dt_norm expected last dim=1 when 3D; got {tuple(dt_norm.shape)}")
            dt = dt_norm.squeeze(-1)
        elif dt_norm.ndim == 2:
            dt = dt_norm
        elif dt_norm.ndim == 1:
            dt = dt_norm.unsqueeze(1)
        else:
            raise ValueError(f"dt_norm must be 1D/2D/3D; got shape {tuple(dt_norm.shape)}")

        B = y_i.shape[0]
        K = dt.shape[1]

        # Expand inputs to match K time steps
        y_exp = y_i.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)

        # Disable autocast for dt-conditioned MLP to preserve precision
        device_type = y_i.device.type
        autocast_off = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in ("cuda", "cpu")
            else nullcontext()
        )

        with autocast_off:
            x = torch.cat(
                [
                    y_exp.to(torch.float32),
                    dt.to(torch.float32).unsqueeze(-1),
                    g_exp.to(torch.float32),
                ],
                dim=-1,
            )  # [B, K, S+1+G] in FP32
            y_pred = self.network(x)  # [B, K, S] in FP32

        # Keep FP32 base for residual computations
        y_i_f = y_i.to(torch.float32)

        # Apply output head
        if self.softmax_head:
            return self._apply_softmax_head(y_pred)
        elif self.predict_delta_log_phys:
            return self._apply_log_phys_residual(y_pred, y_i_f)
        elif self.predict_delta:
            return y_pred + y_i_f.unsqueeze(1)
        else:
            return y_pred

    def _apply_softmax_head(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert raw outputs to z-space via log-softmax -> log10 -> standardize."""
        if self._logsoftmax is None:
            raise RuntimeError("softmax_head=True requires _logsoftmax to be initialized")
        log_p = self._logsoftmax(y_pred)
        dtype = log_p.dtype
        log10_p = log_p / self.ln10.to(dtype)
        return (log10_p - self.log_mean.to(dtype)) / self.log_std.to(dtype)

    def _apply_log_phys_residual(self, y_pred: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        """Apply residual in log10-physical space, then re-standardize to z-space."""
        lm = self.log_mean.to(y_i.dtype)
        ls = self.log_std.to(y_i.dtype)
        
        # Convert base state from z-space to log10-physical
        base_log = y_i * ls + lm  # [B, S]
        
        # Add residual in log10-physical space
        y_pred_log = base_log.unsqueeze(1) + y_pred  # [B, K, S]
        
        # Re-standardize to z-space
        return (y_pred_log - lm) / ls


def create_model(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    Build model from config.
    
    Config requirements:
      - data.species_variables: list of species names (determines S)
      - data.global_variables: list of global parameter names (determines G)
      - model.*: architecture hyperparameters
      
    Note: data.target_species is not supported; S_in == S_out always.
    """
    import json
    from pathlib import Path

    log = logger if logger is not None else logging.getLogger(__name__)

    # Reject unsupported config keys (defense-in-depth; main.py also checks)
    data_cfg = config.get("data", {})
    if data_cfg.get("target_species"):
        raise ValueError(
            "data.target_species is no longer supported. "
            "This codebase enforces S_in == S_out (all species predicted). "
            "Remove data.target_species from your config."
        )
    
    model_cfg = config.get("model", {})
    if model_cfg.get("allow_partial_simplex"):
        raise ValueError(
            "model.allow_partial_simplex is no longer supported. "
            "Remove it from your config."
        )

    # Extract data configuration
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    # Dimensions: S_in == S_out (no subset prediction)
    S = len(species_vars)
    G = len(global_vars)

    # Model configuration
    mcfg = config.get("model", {})

    latent_dim = int(mcfg.get("latent_dim", 32))
    encoder_hidden = list(mcfg.get("encoder_hidden", [256, 128]))
    dynamics_hidden = list(mcfg.get("dynamics_hidden", [256, 256]))
    decoder_hidden = list(mcfg.get("decoder_hidden", [128, 256]))
    dynamics_residual = bool(mcfg.get("dynamics_residual", True))

    activation = str(mcfg.get("activation", "gelu"))
    dropout = float(mcfg.get("dropout", 0.0))

    predict_delta = bool(mcfg.get("predict_delta", True))
    predict_delta_log_phys = bool(mcfg.get("predict_delta_log_phys", False))
    softmax_head = bool(mcfg.get("softmax_head", False))

    # Validate head configuration
    if softmax_head and (predict_delta or predict_delta_log_phys):
        raise ValueError("softmax_head=True requires predict_delta=False and predict_delta_log_phys=False")

    # Load normalization stats if needed for log-phys or softmax heads
    log_mean = None
    log_std = None
    if predict_delta_log_phys or softmax_head:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r") as f:
            manifest = json.load(f)
        stats = manifest.get("per_key_stats") or manifest.get("stats")
        if stats is None:
            raise KeyError("normalization manifest missing per_key_stats/stats")
        
        log_mean = []
        log_std = []
        for name in species_vars:
            if name not in stats:
                raise KeyError(f"Species '{name}' not in normalization stats")
            s = stats[name]
            log_mean.append(float(s.get("log_mean", 0.0)))
            log_std.append(float(s.get("log_std", 1.0)))
        log.info(f"Loaded normalization stats for {len(species_vars)} species")

    # Choose model architecture
    use_mlp_only = bool(mcfg.get("mlp_only", False))

    if use_mlp_only:
        # Configurable MLP for mlp_only mode.
        # Preferred (new): model.mlp = { "num_layers": N, "width": W } or { "widths": [..] } / { "hidden_dims": [..] }
        mlp_hidden: List[int] = []

        mlp_cfg = mcfg.get("mlp", None)
        if isinstance(mlp_cfg, dict) and mlp_cfg:
            if "hidden_dims" in mlp_cfg:
                mlp_hidden = [int(x) for x in (mlp_cfg.get("hidden_dims") or [])]
            elif "widths" in mlp_cfg:
                mlp_hidden = [int(x) for x in (mlp_cfg.get("widths") or [])]
            elif "layers" in mlp_cfg:
                mlp_hidden = [int(x) for x in (mlp_cfg.get("layers") or [])]
            else:
                n_layers = int(mlp_cfg.get("num_layers", 0))
                width = mlp_cfg.get("width", mlp_cfg.get("hidden_dim", None))
                if n_layers > 0 and width is not None:
                    if isinstance(width, (list, tuple)):
                        widths = [int(x) for x in width]
                        if len(widths) != n_layers:
                            raise ValueError(
                                f"model.mlp.width as a list must have length num_layers ({n_layers}), got {len(widths)}"
                            )
                        mlp_hidden = widths
                    else:
                        w = int(width)
                        if w <= 0:
                            raise ValueError(f"model.mlp.width must be > 0, got {w}")
                        mlp_hidden = [w] * n_layers

        # Backward-compatible (legacy): concatenate encoder/dynamics/decoder hidden lists.
        if not mlp_hidden:
            mlp_hidden = list(encoder_hidden) + list(dynamics_hidden) + list(decoder_hidden)

        if len(mlp_hidden) == 0:
            raise ValueError("model.mlp_only=True requires non-empty MLP hidden layer config")

        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=mlp_hidden,
            activation_name=activation,
            dropout=dropout,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            log_mean=log_mean,
            log_std=log_std,
            softmax_head=softmax_head,
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
        log_mean=log_mean,
        log_std=log_std,
        softmax_head=softmax_head,
    )
