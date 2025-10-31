#!/usr/bin/env python3
"""
Stable Autoregressive Autoencoder for Chemical Kinetics
========================================================

A neural dynamics autoencoder designed for long-term stable autoregressive rollout
of chemical kinetics systems. Features:

- Softmax output head ensuring exact conservation (species sum to 1)
- Optional contractive latent dynamics for guaranteed convergence
- Optional VAE encoder for regularized latent space
- Optional logit-delta composition for incremental updates
- Normalized log-space operations for numerical stability

The model operates in normalized log10 space where species concentrations are:
    z = (log10(x) - log_mean) / log_std

Key design choices for stability:
1. Contractive dynamics: z' = z + α(g,dt) * (z_eq(g) - z) where α ∈ (0, α_max]
2. Softmax head: Ensures positivity and conservation
3. Log-space operations: Avoids numerical underflow
"""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Helper Components
# -------------------------------------------------------------------------

def get_activation(name: Union[str, nn.Module]) -> nn.Module:
    """
    Returns an activation module by name.

    Supports: relu, gelu, silu/swish, mish, elu, tanh, identity
    """
    if isinstance(name, nn.Module):
        return name

    key = str(name).strip().lower()

    activations = {
        'relu': lambda: nn.ReLU(inplace=True),
        'gelu': lambda: nn.GELU(),
        'silu': lambda: nn.SiLU(inplace=True),
        'swish': lambda: nn.SiLU(inplace=True),
        'mish': lambda: nn.Mish(),
        'elu': lambda: nn.ELU(inplace=True),
        'tanh': lambda: nn.Tanh(),
        'identity': lambda: nn.Identity(),
        'none': lambda: nn.Identity(),
    }

    if key in activations:
        return activations[key]()

    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation and dropout.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dims: Sequence[int],
            out_dim: int,
            activation: Union[str, nn.Module] = "silu",
            dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        dims = [in_dim] + list(hidden_dims)

        # Build hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer (no activation)
        layers.append(nn.Linear(dims[-1], out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------------------
# Core Model Components
# -------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder: [y, g] -> z (latent representation)

    Supports both deterministic and VAE modes:
    - Deterministic: outputs z directly
    - VAE: outputs (z, kl_divergence) via reparameterization trick
    """

    def __init__(
            self,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            latent_dim: int,
            activation: Union[str, nn.Module] = "silu",
            dropout: float = 0.0,
            vae_mode: bool = False,
    ):
        super().__init__()

        self.vae_mode = vae_mode
        self.latent_dim = latent_dim

        # VAE outputs both mean and log-variance
        out_dim = latent_dim * 2 if vae_mode else latent_dim

        self.net = MLP(
            state_dim + global_dim,
            list(hidden_dims),
            out_dim,
            activation,
            dropout
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            y: [B, S] - normalized species concentrations
            g: [B, G] - global parameters (P, T)

        Returns:
            z: [B, Z] - latent representation
            kl: Optional scalar KL divergence (VAE mode only)
        """
        x = torch.cat([y, g], dim=-1)
        h = self.net(x)

        if not self.vae_mode:
            return h, None

        # VAE: split into mean and log-variance
        mu, logvar = torch.chunk(h, 2, dim=-1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # KL divergence from N(0,1) prior (per-batch scalar)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl = kl.mean()  # Average over batch

        return z, kl


class LatentDynamics(nn.Module):
    """
    Latent space dynamics: z_t -> z_{t+dt}

    Two modes:
    1. Residual MLP: z' = z + f(z, dt, g) - flexible but may drift
    2. Contractive: z' = z + α(g,dt) * (z_eq(g) - z) - guaranteed convergence

    Contractive mode ensures global stability by pulling the state toward
    a learned equilibrium z_eq(g) with rate α ∈ (0, α_max].
    """

    def __init__(
            self,
            latent_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            activation: Union[str, nn.Module] = "silu",
            dropout: float = 0.0,
            dt_stats: Optional[Dict[str, float]] = None,
            contractive: bool = False,
            contractive_alpha_max: float = 0.95,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.global_dim = global_dim
        self.contractive = contractive

        # Validate contractive parameters
        if contractive:
            if not (0.0 < contractive_alpha_max < 1.0):
                raise ValueError(
                    f"contractive_alpha_max must be in (0,1) for stability, got {contractive_alpha_max}"
                )

        self.alpha_max = contractive_alpha_max

        # Standard residual dynamics network (always created for compatibility)
        self.net = MLP(
            latent_dim + 1 + global_dim,  # [z, dt_norm, g]
            list(hidden_dims),
            latent_dim,
            activation,
            dropout
        )

        # Contractive mode components
        if contractive:
            # Equilibrium predictor: g -> z_eq
            self.eq_head = nn.Linear(global_dim, latent_dim, bias=True)

            # Convergence rate predictor: [g, dt] -> α ∈ (0, α_max]
            self.alpha_head = nn.Linear(global_dim + 1, 1, bias=True)

        # Time normalization statistics
        self.dt_stats = dt_stats or {}
        self.register_buffer("dt_log_min", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(0.0, dtype=torch.float32))

        # Validate and set dt bounds
        log_min = self.dt_stats.get("log_min", -9.0)
        log_max = self.dt_stats.get("log_max", 9.0)

        if not (math.isfinite(log_min) and math.isfinite(log_max) and log_max > log_min):
            warnings.warn(
                f"Invalid dt log bounds ({log_min}, {log_max}), using defaults (-9, 9)",
                RuntimeWarning
            )
            log_min, log_max = -9.0, 9.0

        self.dt_log_min.fill_(log_min)
        self.dt_log_max.fill_(log_max)

    def forward(self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Multi-step dynamics (vectorized over K timesteps).

        Args:
            z: [B, Z] - current latent state
            dt_norm: [B, K] - normalized timesteps
            g: [B, G] - global parameters

        Returns:
            z_next: [B, K, Z] - predicted states at each timestep
        """
        B, K = dt_norm.shape

        if not self.contractive:
            # Standard residual dynamics
            z_rep = z.unsqueeze(1).expand(B, K, self.latent_dim)
            dt_flat = dt_norm.reshape(B * K, 1)
            z_flat = z_rep.reshape(B * K, self.latent_dim)
            g_flat = g.unsqueeze(1).expand(B, K, self.global_dim).reshape(B * K, self.global_dim)

            x = torch.cat([z_flat, dt_flat, g_flat], dim=-1)
            dz = self.net(x).reshape(B, K, self.latent_dim)

            return z_rep + dz  # Residual connection

        # Contractive dynamics: z' = z + α * (z_eq - z)
        # Compute equilibrium once (independent of dt)
        z_eq = self.eq_head(g)  # [B, Z]
        z_eq_expanded = z_eq.unsqueeze(1).expand(B, K, self.latent_dim)

        # Compute per-step convergence rates
        g_expanded = g.unsqueeze(1).expand(B, K, self.global_dim)  # view
        dt_expanded = dt_norm.unsqueeze(-1)  # [B, K, 1]

        alpha_input = torch.cat([
            g_expanded.reshape(B * K, self.global_dim),
            dt_expanded.reshape(B * K, 1)
        ], dim=-1)

        alpha = torch.sigmoid(self.alpha_head(alpha_input)) * self.alpha_max  # [BK, 1]
        # Avoid accidental freeze at exactly zero
        alpha = alpha.clamp_min(1e-6)
        alpha = alpha.reshape(B, K, 1)  # [B, K, 1] for broadcasting

        # Apply contraction
        z_current = z.unsqueeze(1).expand(B, K, self.latent_dim)
        z_next = z_current + alpha * (z_eq_expanded - z_current)

        return z_next

    def step(self, z: torch.Tensor, dt_step_norm: Union[torch.Tensor, float], g: torch.Tensor) -> torch.Tensor:
        """
        Single-step dynamics.

        Args:
            z: [B, Z] - current latent state
            dt_step_norm: scalar or [B] or [B, 1] - normalized timestep
            g: [B, G] - global parameters

        Returns:
            z_next: [B, Z] - next latent state
        """
        # Robustly coerce dt_step_norm to [B,1]
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.tensor(dt_step_norm, dtype=z.dtype, device=z.device)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(z.shape[0]).unsqueeze(-1)  # [B,1]
        elif dt_step_norm.ndim == 1:
            # If shape is [B] or [1], make [B,1] by broadcasting
            if dt_step_norm.shape[0] == 1 and z.shape[0] > 1:
                dt_step_norm = dt_step_norm.expand(z.shape[0]).unsqueeze(-1)
            else:
                dt_step_norm = dt_step_norm.unsqueeze(-1)
        # else assume already [B,1]

        if not self.contractive:
            # Standard residual dynamics
            x = torch.cat([z, dt_step_norm, g], dim=-1)
            dz = self.net(x)
            return z + dz  # Residual connection

        # Contractive dynamics
        z_eq = self.eq_head(g)  # [B, Z]
        alpha_input = torch.cat([g, dt_step_norm], dim=-1)
        alpha = torch.sigmoid(self.alpha_head(alpha_input)) * self.alpha_max  # [B, 1]
        alpha = alpha.clamp_min(1e-6)
        return z + alpha * (z_eq - z)


class Decoder(nn.Module):
    """
    Decoder: z -> logits over species

    Maps latent representation to unnormalized log-probabilities (logits)
    which will be processed by a softmax head.
    """

    def __init__(
            self,
            latent_dim: int,
            hidden_dims: Sequence[int],
            out_dim: int,
            activation: Union[str, nn.Module] = "silu",
            dropout: float = 0.0,
    ):
        super().__init__()

        self.net = MLP(latent_dim, list(hidden_dims), out_dim, activation, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [..., Z] - latent representation(s)

        Returns:
            logits: [..., S] - unnormalized log-probabilities
        """
        return self.net(z)


# -------------------------------------------------------------------------
# Main Autoencoder Model
# -------------------------------------------------------------------------

class FlowMapAutoencoder(nn.Module):
    """
    Complete autoencoder for chemical kinetics with stable rollout.

    Architecture:
        Encoder: (y, g) -> z
        Dynamics: z_t -> z_{t+dt}
        Decoder: z -> logits
        Head: softmax -> log10 -> normalize

    Key features:
    - Softmax output ensures conservation (species sum to 1)
    - Optional contractive dynamics for guaranteed convergence
    - Optional VAE regularization
    - Optional logit-delta mode for incremental updates
    - All operations in log-space for numerical stability
    """

    def __init__(
            self,
            state_dim_in: int,
            state_dim_out: int,
            global_dim: int,
            latent_dim: int,
            encoder_hidden: Sequence[int],
            dynamics_hidden: Sequence[int],
            decoder_hidden: Sequence[int],
            activation: Union[str, nn.Module] = "silu",
            dropout: float = 0.0,
            target_idx: Optional[torch.Tensor] = None,
            target_log_mean: Optional[Sequence[float]] = None,
            target_log_std: Optional[Sequence[float]] = None,
            dt_stats: Optional[Dict[str, float]] = None,
            predict_logit_delta: bool = False,
            contractive_dynamics: bool = False,
            contractive_alpha_max: float = 0.95,
            vae_mode: bool = False,
    ):
        super().__init__()

        self.S_in = state_dim_in
        self.S_out = state_dim_out
        self.global_dim = global_dim
        self.latent_dim = latent_dim
        self.predict_logit_delta = predict_logit_delta
        self.vae_mode = vae_mode
        self.kl_loss: Optional[torch.Tensor] = None  # set during encode/forward/step when VAE

        # Validate configuration
        if self.S_out != self.S_in:
            raise ValueError(
                f"Softmax head requires S_out ({self.S_out}) == S_in ({self.S_in}) for conservation"
            )

        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for normalization")

        # Register normalization statistics as buffers
        self.register_buffer("log_mean", torch.tensor(target_log_mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.clamp(torch.tensor(target_log_std, dtype=torch.float32), min=1e-10))
        self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        self.register_buffer("ln10_inv", torch.tensor(1.0 / math.log(10.0), dtype=torch.float32))

        if target_idx is not None:
            self.register_buffer("target_idx", target_idx)
        else:
            self.target_idx = None  # vestigial; kept for compatibility

        # Initialize model components
        self.encoder = Encoder(
            self.S_in, self.global_dim, encoder_hidden,
            self.latent_dim, activation, dropout, vae_mode
        )

        self.dynamics = LatentDynamics(
            self.latent_dim, self.global_dim, dynamics_hidden,
            activation, dropout, dt_stats,
            contractive_dynamics, contractive_alpha_max
        )

        self.decoder = Decoder(
            self.latent_dim, decoder_hidden,
            self.S_out, activation, dropout
        )

    # -------------------------------------------------------------------------
    # Normalization helpers (for numerical stability)
    # -------------------------------------------------------------------------

    def _softmax_head_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to normalized concentrations via softmax.

        Pipeline: logits -> log_softmax -> log10 -> normalize
        All operations in log-space for stability.
        """
        # Compute log probabilities with numerical stability
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_p = torch.clamp(log_p, min=-50.0)  # Floor at ~1e-22 probability

        # Convert to log10 space
        log10_p = log_p * self.ln10_inv

        # Apply normalization
        z_normalized = (log10_p - self.log_mean) / self.log_std

        return z_normalized.to(dtype=logits.dtype)

    def _head_from_logprobs(self, log_p: torch.Tensor) -> torch.Tensor:
        """
        Convert natural log probabilities to normalized concentrations.
        """
        log_p = torch.clamp(log_p.float(), min=-50.0)
        log10_p = log_p * self.ln10_inv
        z_normalized = (log10_p - self.log_mean) / self.log_std
        return z_normalized.to(dtype=log_p.dtype)

    def _denorm_to_logp(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized concentrations back to natural log probabilities.
        """
        log10_p = y_norm.float() * self.log_std + self.log_mean
        return log10_p * self.ln10

    # -------------------------------------------------------------------------
    # Public interfaces
    # -------------------------------------------------------------------------

    def encode(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Encode state into latent representation.

        NOTE (compat): Returns ONLY z so downstream code does not change.
        When vae_mode=True, also sets self.kl_loss for the trainer to consume.

        Args:
            y: [B, S] - normalized species concentrations
            g: [B, G] - global parameters

        Returns:
            z: [B, Z] - latent representation
        """
        z, kl = self.encoder(y, g)
        # Maintain backward compatibility: stash KL if present
        self.kl_loss = kl
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to logits.

        Args:
            z: [..., Z] - latent representation

        Returns:
            logits: [..., S] - unnormalized log-probabilities
        """
        return self.decoder(z)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Predict states at multiple future timesteps (training mode).

        Args:
            y_i: [B, S] - initial normalized state
            dt_norm: [B, K] - normalized timestep offsets (absolute offsets from anchor)
            g: [B, G] - global parameters

        Returns:
            y_pred: [B, K, S] - predicted normalized states
        """
        # Encode initial state (also sets self.kl_loss if VAE)
        z0 = self.encode(y_i, g)  # [B, Z]

        # ---- cache for rollout loss (trainer will read & clear) ----
        self._cached_rollout_z0 = z0
        self._cached_rollout_film = None  # populate here if you later add FiLM
        # ------------------------------------------------------------

        # Vectorized latent propagation to each absolute offset
        Z_seq = self.dynamics(z0, dt_norm, g)  # [B, K, Z]

        if not self.predict_logit_delta:
            # Direct prediction mode: decode each latent independently
            logits = self.decoder(Z_seq)  # [B, K, S]
            return self._softmax_head_from_logits(logits)

        # Logit-delta mode: independent deltas relative to the same anchor y_i
        base_logp = self._denorm_to_logp(y_i).float()  # [B, S]

        outs = []
        K = dt_norm.shape[1]
        for k in range(K):
            z_k = Z_seq[:, k, :]  # [B, Z]
            logits_k = self.decoder(z_k)  # [B, S]
            log_q = F.log_softmax(logits_k.float(), dim=-1).clamp_min(-50.0)  # [B, S]

            combined = base_logp + log_q
            log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)  # [B, S]
            outs.append(self._head_from_logprobs(log_p))  # normalized [B, S]

        return torch.stack(outs, dim=1)  # [B, K, S]

    def step(self, y: torch.Tensor, dt_step_norm: Union[torch.Tensor, float], g: torch.Tensor) -> torch.Tensor:
        """
        Single autoregressive step (teacher forcing).

        Args:
            y: [B, S] - current normalized state
            dt_step_norm: timestep to next state
            g: [B, G] - global parameters

        Returns:
            y_next: [B, S] - predicted next state
        """
        # Encode current state (also sets self.kl_loss if VAE)
        z = self.encode(y, g)

        # Single dynamics step
        z = self.dynamics.step(z, dt_step_norm, g)

        # Decode
        logits = self.decoder(z)  # [B, S]

        if not self.predict_logit_delta:
            return self._softmax_head_from_logits(logits)

        # Logit-delta mode
        base_logp = self._denorm_to_logp(y)
        log_q = F.log_softmax(logits.float(), dim=-1)
        log_q = torch.clamp(log_q, min=-50.0)

        combined = base_logp + log_q
        log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)

        return self._head_from_logprobs(log_p)

    def rollout_vectorized(
            self,
            y0: torch.Tensor,
            g: torch.Tensor,
            dt_steps_norm: torch.Tensor,
            z0: Optional[torch.Tensor] = None,
            film_cache: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Gradient-carrying latent rollout over INCREMENTAL steps (for rollout loss).

        Args:
            y0:            [B, S]   - initial normalized state (anchor)
            g:             [B, G]   - globals
            dt_steps_norm: [B, H]   - per-step normalized increments (NOT absolute offsets)
            z0:            [B, Z]   - optional cached latent from forward()
            film_cache:    (gamma:[B,Z], beta:[B,Z], alpha:float) applied to latent before decode (optional)

        Returns:
            y_pred:        [B, H, S] - predicted normalized states along the rollout
        """
        B, H = dt_steps_norm.shape

        # Use cached latent if provided; otherwise encode once
        z = z0 if z0 is not None else self.encode(y0, g)  # [B, Z]

        # Step through latent dynamics sequentially to build the trajectory
        z_list = []
        for k in range(H):
            z = self.dynamics.step(z, dt_steps_norm[:, k], g)  # [B, Z]
            z_list.append(z)
        Z_seq = torch.stack(z_list, dim=1)  # [B, H, Z]

        # Optional FiLM on latent before decoding (no-op unless you add such a module)
        if film_cache is not None:
            gamma, beta, alpha = film_cache  # [B,Z], [B,Z], scalar
            # Apply the same affine to each time step
            Z_seq = (1.0 + alpha * torch.tanh(gamma)).unsqueeze(1) * Z_seq + beta.unsqueeze(1)

        logits = self.decoder(Z_seq)  # [B, H, S]

        if not self.predict_logit_delta:
            return self._softmax_head_from_logits(logits)  # [B, H, S]

        # True autoregressive composition in probability space
        traj = []
        current_logp = self._denorm_to_logp(y0).float()  # [B, S]
        for k in range(H):
            log_q = F.log_softmax(logits[:, k, :].float(), dim=-1).clamp_min(-50.0)  # [B, S]
            combined = current_logp + log_q
            current_logp = combined - torch.logsumexp(combined, dim=-1, keepdim=True)  # [B, S]
            traj.append(self._head_from_logprobs(current_logp))  # normalized [B, S]

        return torch.stack(traj, dim=1)  # [B, H, S]


# -------------------------------------------------------------------------
# Factory function
# -------------------------------------------------------------------------

def create_model(config: dict) -> FlowMapAutoencoder:
    """
    Create model from configuration dictionary.

    Expected config structure:
    - data.species_variables: list of species names
    - data.global_variables: list of global variable names (e.g., ["P", "T"])
    - data.target_species: optional subset of species to predict
    - model.latent_dim: latent space dimension
    - model.encoder_hidden: list of hidden layer sizes for encoder
    - model.dynamics_hidden: list of hidden layer sizes for dynamics
    - model.decoder_hidden: list of hidden layer sizes for decoder
    - model.activation: activation function name
    - model.dropout: dropout probability
    - model.predict_logit_delta: whether to use incremental updates
    - model.contractive_dynamics: whether to use contractive dynamics
    - model.contractive_alpha_max: maximum contraction rate
    - model.vae_mode: whether to use VAE encoder
    """
    # Parse data configuration
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables is required")

    global_vars = list(data_cfg.get("global_variables", []))
    target_vars = list(data_cfg.get("target_species", species_vars))

    # Map species names to indices
    name_to_idx = {name: i for i, name in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown species: {e.args[0]}")

    # Parse model configuration
    model_cfg = config.get("model", {})
    state_dim_in = len(species_vars)
    state_dim_out = len(target_vars)
    global_dim = len(global_vars)

    # Architecture parameters
    latent_dim = int(model_cfg.get("latent_dim", 128))
    encoder_hidden = model_cfg.get("encoder_hidden", [256, 512, 256])
    dynamics_hidden = model_cfg.get("dynamics_hidden", [256, 512, 256])
    decoder_hidden = model_cfg.get("decoder_hidden", [256, 512, 256])
    activation = model_cfg.get("activation", "silu")
    dropout = float(model_cfg.get("dropout", 0.0))

    # Model behavior flags
    predict_logit_delta = bool(model_cfg.get("predict_logit_delta", False))
    contractive_dynamics = bool(model_cfg.get("contractive_dynamics", False))
    contractive_alpha_max = float(model_cfg.get("contractive_alpha_max", 0.95))
    vae_mode = bool(model_cfg.get("vae_mode", False))

    # Load normalization statistics
    paths_cfg = config.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    norm_path = processed_dir / "normalization.json"

    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.json not found at {norm_path}")

    with open(norm_path, "r") as f:
        manifest = json.load(f)

    # Extract time statistics
    dt_stats = None
    if "dt" in manifest:
        dt_stats = {
            "log_min": float(manifest["dt"]["log_min"]),
            "log_max": float(manifest["dt"]["log_max"])
        }

    # Extract species normalization statistics
    per_key_stats = manifest.get("per_key_stats", {})
    target_log_mean = []
    target_log_std = []

    for name in target_vars:
        if name not in per_key_stats:
            raise KeyError(f"Species '{name}' not found in normalization statistics")

        stats = per_key_stats[name]
        target_log_mean.append(float(stats.get("log_mean", 0.0)))
        target_log_std.append(float(stats.get("log_std", 1.0)))

    # Create model
    return FlowMapAutoencoder(
        state_dim_in=state_dim_in,
        state_dim_out=state_dim_out,
        global_dim=global_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        dynamics_hidden=dynamics_hidden,
        decoder_hidden=decoder_hidden,
        activation=activation,
        dropout=dropout,
        target_idx=torch.tensor(target_idx, dtype=torch.long) if target_idx else None,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        dt_stats=dt_stats,
        predict_logit_delta=predict_logit_delta,
        contractive_dynamics=contractive_dynamics,
        contractive_alpha_max=contractive_alpha_max,
        vae_mode=vae_mode,
    )


__all__ = ["FlowMapAutoencoder", "create_model"]
