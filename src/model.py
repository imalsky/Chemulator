#!/usr/bin/env python3
"""
model.py

Flow-map models for stiff chemistry in z-space (log-standard normalized):

  (y_i, dt_norm, g) -> y_j  in z-space

Supports:
  - Residual in z-space (predict_delta)
  - Residual in physical log10 space (predict_delta_log_phys)
  - Simplex projection in physical space when predict_delta_log_phys is enabled

Architecture options:
  - FlowMapAutoencoder: Encoder -> LatentDynamics -> Decoder
  - FlowMapMLP: Direct MLP with configurable layers

IMPORTANT RESTRICTION (by design):
  - Subset target species is NOT supported.
  - The model always predicts the full state: S_out == S_in.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# =============================================================================
# Constants
# =============================================================================

LN10 = math.log(10.0)
MIN_LOG_STD = 1e-10
RESIDUAL_INIT_SCALE = 0.1

# LayerNorm epsilon: keep default-ish; too small can be unstable in bf16
LN_EPS = 1e-5

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


# =============================================================================
# Helpers
# =============================================================================


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    key = str(name).lower().strip()
    if key not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Choices={sorted(ACTIVATION_REGISTRY)}")
    return ACTIVATION_REGISTRY[key]()


def _fresh_activation(act: nn.Module) -> nn.Module:
    """Create a fresh activation instance, preserving constructor parameters.

    NOTE: Some activations (e.g., LeakyReLU, ELU) have constructor args stored on the module.
    Using act.__class__() would silently drop those args and change behavior.

    copy.deepcopy() creates a new module with the same attributes/parameters, without sharing state.
    """
    return copy.deepcopy(act)


def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast tensor to match reference dtype."""
    if x.dtype == ref.dtype:
        return x
    return x.to(dtype=ref.dtype)


def _project_log10_simplex(log10_vals: torch.Tensor, ln10: torch.Tensor) -> torch.Tensor:
    """
    Project log10-values onto the simplex in physical space (sum_i y_i = 1).

    Given log10(y_raw), returns log10(y_raw / sum_i y_raw). This guarantees that
    the corresponding linear-space values sum to 1, while staying in log10 space.
    """
    ln10_t = ln10.to(dtype=log10_vals.dtype, device=log10_vals.device)
    ln_raw = log10_vals * ln10_t
    ln_norm = ln_raw - torch.logsumexp(ln_raw, dim=-1, keepdim=True)
    return ln_norm / ln10_t


def _init_residual_output(linear: nn.Linear) -> None:
    """Initialize output layer for residual prediction (small weights, zero bias)."""
    with torch.no_grad():
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        linear.weight.mul_(RESIDUAL_INIT_SCALE)


def _normalize_dt_shape(dt_norm: torch.Tensor) -> torch.Tensor:
    """Normalize dt_norm to [B, K] shape."""
    if dt_norm.ndim == 3:
        if dt_norm.shape[-1] != 1:
            raise ValueError(f"dt_norm expected last dim=1 when 3D; got {tuple(dt_norm.shape)}")
        return dt_norm.squeeze(-1)
    elif dt_norm.ndim == 2:
        return dt_norm
    elif dt_norm.ndim == 1:
        return dt_norm.unsqueeze(1)
    else:
        raise ValueError(f"dt_norm must be 1D/2D/3D; got shape {tuple(dt_norm.shape)}")


def _cat_y_dt_g(
    y_exp: torch.Tensor,  # [B,K,Dy]
    dt: torch.Tensor,     # [B,K]
    g_exp: torch.Tensor,  # [B,K,Dg] or empty
) -> torch.Tensor:
    """
    Fast-ish concat helper:
      - keeps dt in the same dtype/device as y_exp to avoid FP32 upcast during concat
      - skips g if empty
    """
    dtc = dt.to(device=y_exp.device, dtype=y_exp.dtype).unsqueeze(-1)  # [B,K,1]
    if g_exp.numel() == 0:
        return torch.cat([y_exp, dtc], dim=-1)
    return torch.cat([y_exp, dtc, g_exp], dim=-1)


# =============================================================================
# Core Network Blocks
# =============================================================================


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
            if dropout_p > 0:
                layers.append(nn.Dropout(p=float(dropout_p)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    Residual MLP with skip connections across each hidden layer.

    Pre-norm LayerNorm variant:
      h -> LN(h) -> Linear -> Act -> Dropout -> + skip(proj(h))

    Notes:
      - LN is per-feature over the last dimension (works for [B,D] and [B,K,D]).
      - This is typically more stable for long rollouts, but it can cost throughput.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
        residual: bool = True,
        layernorm_eps: float = LN_EPS,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        h = [int(x) for x in hidden_dims]
        dims = [self.input_dim] + h

        self.norms = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(h)):
            in_d, out_d = dims[i], dims[i + 1]

            self.norms.append(nn.LayerNorm(in_d, eps=float(layernorm_eps)))

            self.linears.append(nn.Linear(in_d, out_d))
            self.acts.append(_fresh_activation(activation))
            self.dropouts.append(nn.Dropout(p=float(dropout_p)) if dropout_p > 0 else nn.Identity())

            if self.residual:
                self.proj.append(nn.Identity() if in_d == out_d else nn.Linear(in_d, out_d, bias=False))
            else:
                self.proj.append(nn.Identity())

        self.out = nn.Linear(dims[-1], self.output_dim)

    def last_linear(self) -> nn.Linear:
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for norm, lin, act, do, proj in zip(self.norms, self.linears, self.acts, self.dropouts, self.proj):
            h_norm = norm(h)
            y = do(act(lin(h_norm)))
            if self.residual:
                y = y + proj(h)
            h = y
        return self.out(h)


# =============================================================================
# Autoencoder Components
# =============================================================================


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
        self.state_dim = int(state_dim)
        self.global_dim = int(global_dim)
        self.latent_dim = int(latent_dim)

        self.network = MLP(
            input_dim=self.state_dim + self.global_dim,
            hidden_dims=hidden_dims,
            output_dim=self.latent_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if g.numel() > 0:
            x = torch.cat([y, g], dim=-1)
        else:
            x = y
        z = self.network(x)
        kl = torch.zeros((), device=z.device, dtype=z.dtype)
        return z, kl


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
        self.latent_dim = int(latent_dim)
        self.global_dim = int(global_dim)
        self.residual = bool(residual)

        self.network = MLP(
            input_dim=self.latent_dim + 1 + self.global_dim,
            hidden_dims=list(hidden_dims),
            output_dim=self.latent_dim,
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
        dt = _normalize_dt_shape(dt_norm)  # [B,K]
        B, K = z.shape[0], dt.shape[1]

        z_exp = z.unsqueeze(1).expand(B, K, -1)

        if g.numel() > 0:
            g_exp = g.unsqueeze(1).expand(B, K, -1)
        else:
            g_exp = z_exp.new_empty((B, K, 0))

        x = _cat_y_dt_g(z_exp, dt, g_exp)  # [B,K,Z+1+G] or [B,K,Z+1]
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
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)

        self.network = MLP(
            input_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            output_dim=self.state_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# =============================================================================
# Prediction Heads Mixin
# =============================================================================


class PredictionHeadMixin:
    """Mixin providing prediction head logic for both model types."""

    predict_delta: bool
    predict_delta_log_phys: bool
    log_mean: Optional[torch.Tensor]
    log_std: Optional[torch.Tensor]
    ln10: Optional[torch.Tensor]
    S: int

    def _apply_prediction_head(self, y_pred: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [B, K, S] raw network output
            y_i: [B, S] input state
        Returns:
            y_pred_z: [B, K, S] predictions in z-space
        """
        if self.predict_delta_log_phys:
            # Keep FP32 for the "important params" / arithmetic that are most sensitive.
            # This is where stiffness + simplex projection can make bf16/fp16 fragile.
            y_i_f = y_i.to(torch.float32)
            lm = self.log_mean.to(torch.float32)
            ls = self.log_std.to(torch.float32)

            base_log10 = y_i_f * ls + lm
            delta_out = y_pred.to(torch.float32)
            delta_log10 = delta_out * ls if self.predict_delta else delta_out

            y_pred_log10 = base_log10.unsqueeze(1) + delta_log10
            y_pred_log10 = _project_log10_simplex(y_pred_log10, self.ln10)

            y_pred_z = (y_pred_log10 - lm) / ls
            # Return in y_i dtype (often float32), which is typically what you want for loss stability.
            return _cast_like(y_pred_z, y_i)

        if self.predict_delta:
            # IMPORTANT CHANGE:
            # - Do the residual add in model dtype (y_pred dtype) to avoid a huge FP32 cast.
            # - But return in y_i dtype (often float32) for accuracy/stability in the loss pipeline.
            y = y_pred + y_i.to(dtype=y_pred.dtype).unsqueeze(1)
            return _cast_like(y, y_i)

        # Non-residual head: return in y_i dtype (often float32) for accuracy/stability.
        return _cast_like(y_pred, y_i)

    def _init_log_stats(
        self,
        target_log_mean: Optional[Sequence[float]],
        target_log_std: Optional[Sequence[float]],
    ) -> None:
        """Initialize log statistics buffers for predict_delta_log_phys mode."""
        if self.predict_delta_log_phys:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("target_log_mean/std required for predict_delta_log_phys")
            log_mean = torch.as_tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.clamp(torch.as_tensor(target_log_std, dtype=torch.float32), min=MIN_LOG_STD)
            if log_mean.numel() != self.S or log_std.numel() != self.S:
                raise ValueError(
                    f"target_log_mean/std must have length {self.S} (full state); "
                    f"got mean={log_mean.numel()} std={log_std.numel()}"
                )
            self.register_buffer("log_mean", log_mean, persistent=True)
            self.register_buffer("log_std", log_std, persistent=True)
            self.register_buffer("ln10", torch.tensor(LN10, dtype=torch.float32), persistent=True)
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None


# =============================================================================
# Full Models
# =============================================================================


class FlowMapAutoencoder(nn.Module, PredictionHeadMixin):
    """
    Flow-map AE: (y_i, g, dt_norm) -> y_j in z-space.

    Architecture: Encoder -> LatentDynamics -> Decoder
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
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        self.S = int(state_dim)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)

        self._init_log_stats(target_log_mean, target_log_std)

        if (self.predict_delta or self.predict_delta_log_phys) and (not dynamics_residual):
            logging.getLogger(__name__).warning(
                "FlowMapAutoencoder: predict_delta*/predict_delta_log_phys enabled while dynamics_residual=False. "
                "This is allowed, but it often makes training harder / less intuitive because the head is residual-style."
            )


        act = get_activation(activation_name)

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
            residual=bool(dynamics_residual),
        )
        self.decoder = Decoder(
            latent_dim=self.Z,
            hidden_dims=list(decoder_hidden),
            state_dim=self.S,
            activation=act,
            dropout_p=float(dropout),
        )

        if self.predict_delta or self.predict_delta_log_phys:
            _init_residual_output(self.decoder.network.network[-1])

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
          y_i: [B,S]
          dt_norm: [B,K] or [B,K,1] or [B]
          g: [B,G]
          -> y_pred_z: [B,K,S]
        """
        z_i, _ = self.encoder(y_i, g)         # [B,Z]
        z_j = self.dynamics(z_i, dt_norm, g)  # [B,K,Z]
        y_pred = self.decoder(z_j)            # [B,K,S]
        return self._apply_prediction_head(y_pred, y_i)


class FlowMapMLP(nn.Module, PredictionHeadMixin):
    """
    MLP-only flow-map: (y_i, g, dt_norm) -> y_j in z-space.

    If dynamics_residual=True: uses ResidualMLP (with pre-norm LayerNorm).
    Else: uses plain MLP.
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
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        self.S = int(state_dim)
        self.G = int(global_dim)

        self.dynamics_residual = bool(dynamics_residual)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)

        self._init_log_stats(target_log_mean, target_log_std)

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
                layernorm_eps=LN_EPS,
            )
        else:
            self.network = MLP(
                input_dim=inp,
                hidden_dims=list(hidden_dims),
                output_dim=self.S,
                activation=act,
                dropout_p=float(dropout),
            )

        if self.predict_delta or self.predict_delta_log_phys:
            try:
                lin_out = (
                    self.network.last_linear()
                    if isinstance(self.network, ResidualMLP)
                    else self.network.network[-1]
                )
                if not isinstance(lin_out, nn.Linear):
                    raise TypeError(f"Expected output layer nn.Linear, got {type(lin_out)}")
                _init_residual_output(lin_out)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to init residual output layer (continuing without residual init): %s",
                    str(e),
                    exc_info=True,
                )

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
          y_i: [B,S]
          dt_norm: [B,K] or [B,K,1] or [B]
          g: [B,G]
          -> y_pred_z: [B,K,S]
        """
        dt = _normalize_dt_shape(dt_norm)  # [B,K]
        B, K = y_i.shape[0], dt.shape[1]

        y_exp = y_i.unsqueeze(1).expand(B, K, -1)

        if g.numel() > 0:
            g_exp = g.unsqueeze(1).expand(B, K, -1)
        else:
            g_exp = y_exp.new_empty((B, K, 0))

        x = _cat_y_dt_g(y_exp, dt, g_exp)  # [B,K,S+1+G] or [B,K,S+1]
        y_pred = self.network(x)           # [B,K,S]
        return self._apply_prediction_head(y_pred, y_i)


# =============================================================================
# Model Factory
# =============================================================================


def _load_log_stats(norm_path: Path, species_vars: Sequence[str], logger: logging.Logger) -> Tuple[list, list]:
    """Load log statistics from normalization manifest."""
    with open(norm_path, "r") as f:
        manifest = json.load(f)
    stats = manifest["per_key_stats"]
    log_mean, log_std = [], []
    for name in species_vars:
        if name not in stats:
            raise KeyError(f"Species '{name}' not in normalization stats")
        s = stats[name]
        log_mean.append(float(s.get("log_mean", 0.0)))
        log_std.append(float(s.get("log_std", 1.0)))
    logger.info(f"Loaded normalization stats for {len(species_vars)} species")
    return log_mean, log_std


def _parse_hidden_dims(mcfg: Dict[str, Any]) -> Sequence[int]:
    """
    Parse hidden layer dimensions from config.

    Supports two formats:
      1) Explicit list: model.mlp_hidden = [512, 512, 512]
      2) Shorthand: model.mlp_num_layers = 4, model.mlp_hidden_dim = 512
    """
    if "mlp_hidden" in mcfg:
        return list(mcfg["mlp_hidden"])

    num_layers = mcfg.get("mlp_num_layers")
    hidden_dim = mcfg.get("mlp_hidden_dim")
    if num_layers is not None and hidden_dim is not None:
        return [int(hidden_dim)] * int(num_layers)

    return (
        list(mcfg.get("encoder_hidden", [256, 128]))
        + list(mcfg.get("dynamics_hidden", [256, 256]))
        + list(mcfg.get("decoder_hidden", [128, 256]))
    )


def create_model(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    Build model from config.

    Enforces:
      - Full-state prediction only: target_species must be absent or identical to species_variables.

    MLP configuration options (when mlp_only=True):
      - model.mlp_hidden: Explicit list of hidden layer widths
      - model.mlp_num_layers + model.mlp_hidden_dim: Shorthand for uniform layers
      - Fallback: concatenates encoder_hidden + dynamics_hidden + decoder_hidden
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    target_vars_cfg = list(data_cfg.get("target_species") or [])
    if target_vars_cfg and target_vars_cfg != species_vars:
        raise ValueError(
            "Subset target_species is not supported. "
            "Either remove data.target_species or set it equal to data.species_variables."
        )

    S = len(species_vars)
    G = len(global_vars)

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

    target_log_mean = None
    target_log_std = None
    if predict_delta_log_phys:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        target_log_mean, target_log_std = _load_log_stats(norm_path, species_vars, log)

    use_mlp_only = bool(mcfg.get("mlp_only", False))
    if use_mlp_only:
        mlp_hidden = _parse_hidden_dims(mcfg)
        if not mlp_hidden:
            raise ValueError("model.mlp_only=True requires non-empty hidden dimensions")

        log.info(f"Creating FlowMapMLP with hidden_dims={mlp_hidden}")
        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=mlp_hidden,
            activation_name=activation,
            dropout=dropout,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
        )

    log.info(
        f"Creating FlowMapAutoencoder with latent_dim={latent_dim}, "
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
        dynamics_residual=dynamics_residual,
        predict_delta=predict_delta,
        predict_delta_log_phys=predict_delta_log_phys,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
    )
