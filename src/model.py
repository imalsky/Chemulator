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
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

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
    n = str(name).lower().strip()
    if n not in ACTIVATION_ALIASES:
        raise ValueError(f"Unknown activation '{name}'. Choices={sorted(ACTIVATION_ALIASES)}")
    return ACTIVATION_ALIASES[n]()


def _fresh_activation(act: nn.Module) -> nn.Module:
    # Create a new instance of the same activation type (avoid sharing state if any)
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
        # For future expansion: return (z, kl) to support VAE-like encoder; for now kl=0.
        x = torch.cat([y, g], dim=-1) if g.numel() > 0 else y
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
            if dt_norm.shape[-1] != 1:
                raise ValueError(f"dt_norm expected last dim=1 when 3D; got {tuple(dt_norm.shape)}")
            dt = dt_norm.squeeze(-1)  # [B,K]
        elif dt_norm.ndim == 2:
            dt = dt_norm
        elif dt_norm.ndim == 1:
            dt = dt_norm.unsqueeze(1)  # [B,1]
        else:
            raise ValueError(f"dt_norm must be 1D/2D/3D; got shape {tuple(dt_norm.shape)}")

        B = z.shape[0]
        K = dt.shape[1]

        z_exp = z.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)
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
      - softmax_head (simplex probabilities, then log10 standardized)
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
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        target_idx: Optional[torch.Tensor] = None,
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        softmax_head: bool = False,
        allow_partial_simplex: bool = False,
    ):
        super().__init__()

        # Dimensions
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.Z = int(latent_dim)

        # Config
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")
        if self.softmax_head and (self.S_out != self.S_in) and not self.allow_partial_simplex:
            raise RuntimeError("softmax_head=True with S_out!=S_in requires allow_partial_simplex=True")

        # Target indices
        if target_idx is None:
            self.target_idx = None
        else:
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx, persistent=True)

        # Stats for log-phys/simplex heads
        if self.predict_delta_log_phys or self.softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("target_log_mean/std required for log-phys or softmax head")
            log_mean = torch.as_tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.clamp(torch.as_tensor(target_log_std, dtype=torch.float32), min=1e-10)
            self.register_buffer("log_mean", log_mean, persistent=True)
            self.register_buffer("log_std", log_std, persistent=True)
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32), persistent=True)
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
            state_dim=self.S_in,
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
            state_dim=self.S_out,
            activation=act,
            dropout_p=float(dropout),
        )

        # Gentle decoder init for residual heads
        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                lin_out: nn.Linear = self.decoder.network.network[-1]
                nn.init.zeros_(lin_out.bias)
                lin_out.weight.mul_(0.1)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions in z-space (log-standard normalized).
        Shapes:
          y_i: [B,S_in]
          dt_norm: [B,K] or [B,K,1]
          g: [B,G]
          -> y_pred_z: [B,K,S_out]
        """
        # Encode once
        z_i, _kl_unused = self.encoder(y_i, g)            # [B,Z]
        # Vectorized latent evolution over K
        z_j = self.dynamics(z_i, dt_norm, g)              # [B,K,Z]
        # Decode
        y_pred = self.decoder(z_j)                        # [B,K,S_out]

        # Heads
        if self.softmax_head:
            # Stable: log-softmax -> log10 -> standardize to z-space
            log_p = self._logsoftmax(y_pred)  # [B,K,S_out]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            y_pred = (log10_p - self.log_mean.to(dtype)) / self.log_std.to(dtype)
            return y_pred

        if self.predict_delta_log_phys:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base_z = y_i.index_select(1, self.target_idx)
            else:
                base_z = y_i
            lm = self.log_mean.to(base_z.dtype)
            ls = self.log_std.to(base_z.dtype)
            base_log = base_z * ls + lm                 # [B,S_out]
            y_pred_log = base_log.unsqueeze(1) + y_pred # [B,K,S_out]
            y_pred = (y_pred_log - lm.to(y_pred.dtype)) / ls.to(y_pred.dtype)
            return y_pred

        if self.predict_delta:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base = y_i.index_select(1, self.target_idx)
            else:
                base = y_i
            y_pred = y_pred + base.unsqueeze(1)
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
      - softmax_head (simplex probabilities, then log10 standardized)
    """

    def __init__(
        self,
        *,
        state_dim_in: int,
        state_dim_out: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        activation_name: str = "gelu",
        dropout: float = 0.0,
        dynamics_residual: bool = True,
        predict_delta: bool = True,
        predict_delta_log_phys: bool = False,
        target_idx: Optional[torch.Tensor] = None,
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        softmax_head: bool = False,
        allow_partial_simplex: bool = False,
    ):
        super().__init__()

        # Dimensions
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)

        # Config
        self.dynamics_residual = bool(dynamics_residual)
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError("softmax_head=True is incompatible with residual modes")
        if self.softmax_head and (self.S_out != self.S_in) and not self.allow_partial_simplex:
            raise RuntimeError("softmax_head=True with S_out!=S_in requires allow_partial_simplex=True")

        # Target indices
        if target_idx is None:
            self.target_idx = None
        else:
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx, persistent=True)

        # Stats for log-phys/simplex heads (match FlowMapAutoencoder behavior)
        if self.predict_delta_log_phys or self.softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("target_log_mean/std required for log-phys or softmax head")
            log_mean = torch.as_tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.clamp(torch.as_tensor(target_log_std, dtype=torch.float32), min=1e-10)
            self.register_buffer("log_mean", log_mean, persistent=True)
            self.register_buffer("log_std", log_std, persistent=True)
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32), persistent=True)
            self._logsoftmax = nn.LogSoftmax(dim=-1) if self.softmax_head else None
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None
            self._logsoftmax = None

        # Main MLP
        act = get_activation(activation_name)
        inp = self.S_in + 1 + self.G

        # "Dynamics-style" residual connections inside the MLP (default ON).
        # This is distinct from the explicit y + dy head (predict_delta).
        if self.dynamics_residual:
            self.network = ResidualMLP(
                input_dim=inp,
                hidden_dims=list(hidden_dims),
                output_dim=self.S_out,
                activation=act,
                dropout_p=float(dropout),
                residual=True,
            )
        else:
            self.network = MLP(
                input_dim=inp,
                hidden_dims=list(hidden_dims),
                output_dim=self.S_out,
                activation=act,
                dropout_p=float(dropout),
            )

        # Gentle init on residual-style heads for stability at start
        if self.predict_delta or self.predict_delta_log_phys:
            try:
                with torch.no_grad():
                    lin_out: nn.Linear
                    if isinstance(self.network, ResidualMLP):
                        lin_out = self.network.last_linear()
                    else:
                        lin_out = self.network.network[-1]  # type: ignore[assignment]
                    nn.init.zeros_(lin_out.bias)
                    lin_out.weight.mul_(0.1)
            except Exception:
                pass

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Returns predictions in z-space (log-standard normalized).
        Shapes:
          y_i: [B,S_in]
          dt_norm: [B,K] or [B,K,1] or [B]
          g: [B,G]
          -> y_pred_z: [B,K,S_out]
        """
        if dt_norm.ndim == 3:
            # [B,K,1] -> [B,K]
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

        y_exp = y_i.unsqueeze(1).expand(B, K, -1)
        g_exp = g.unsqueeze(1).expand(B, K, -1)
        x = torch.cat([y_exp, dt.unsqueeze(-1), g_exp], dim=-1)  # [B,K,S_in+1+G]

        y_pred = self.network(x)  # [B,K,S_out]

        # Heads (must match FlowMapAutoencoder exactly)
        if self.softmax_head:
            if self._logsoftmax is None:
                raise RuntimeError("softmax_head=True requires _logsoftmax to be initialized")
            # Stable: log-softmax -> log10 -> standardize to z-space
            log_p = self._logsoftmax(y_pred)  # [B,K,S_out]
            dtype = log_p.dtype
            ln10 = self.ln10.to(dtype=dtype)
            log10_p = log_p / ln10
            y_pred = (log10_p - self.log_mean.to(dtype)) / self.log_std.to(dtype)
            return y_pred

        if self.predict_delta_log_phys:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base_z = y_i.index_select(1, self.target_idx)
            else:
                base_z = y_i
            lm = self.log_mean.to(base_z.dtype)
            ls = self.log_std.to(base_z.dtype)
            base_log = base_z * ls + lm                 # [B,S_out]
            y_pred_log = base_log.unsqueeze(1) + y_pred # [B,K,S_out]
            y_pred = (y_pred_log - lm.to(y_pred.dtype)) / ls.to(y_pred.dtype)
            return y_pred

        if self.predict_delta:
            if self.S_out != self.S_in:
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError("target_idx required when S_out != S_in")
                base = y_i.index_select(1, self.target_idx)
            else:
                base = y_i
            y_pred = y_pred + base.unsqueeze(1)
            return y_pred

        return y_pred


def create_model(config: Dict[str, Any], logger: Optional["logging.Logger"] = None) -> nn.Module:
    """
    Build model from config.
    """
    import json
    from pathlib import Path
    import logging

    log = logger if logger is not None else logging.getLogger(__name__)

    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")

    target_vars = list(data_cfg.get("target_species") or species_vars)

    # map targets to indices
    name_to_idx = {n: i for i, n in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[n] for n in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    S_in = len(species_vars)
    S_out = len(target_vars)
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
    softmax_head = bool(mcfg.get("softmax_head", False))
    allow_partial_simplex = bool(mcfg.get("allow_partial_simplex", False))

    if softmax_head:
        if predict_delta or predict_delta_log_phys:
            raise ValueError("softmax_head=True requires predict_delta=False and predict_delta_log_phys=False")
        if S_out != S_in:
            # When predicting only a subset with a simplex head, force allow_partial_simplex
            if not allow_partial_simplex:
                raise ValueError("softmax_head=True with subset species requires allow_partial_simplex=True")

    need_stats = predict_delta_log_phys or softmax_head
    target_log_mean = None
    target_log_std = None
    if need_stats:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r") as f:
            manifest = json.load(f)
        stats = manifest["per_key_stats"]
        target_log_mean, target_log_std = [], []
        for name in target_vars:
            if name not in stats:
                raise KeyError(f"Target species '{name}' not in normalization stats")
            s = stats[name]
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))
        log.info(f"Loaded normalization stats for {len(target_vars)} species")

    use_mlp_only = bool(mcfg.get("mlp_only", False))

    if use_mlp_only:
        # MLP hidden sizes are specified by the existing encoder/dynamics/decoder lists.
        mlp_hidden = list(encoder_hidden) + list(dynamics_hidden) + list(decoder_hidden)
        if len(mlp_hidden) == 0:
            raise ValueError("model.mlp_only=True requires non-empty encoder_hidden/dynamics_hidden/decoder_hidden")

        return FlowMapMLP(
            state_dim_in=S_in,
            state_dim_out=S_out,
            global_dim=G,
            hidden_dims=mlp_hidden,
            activation_name=activation,
            dropout=dropout,
            dynamics_residual=dynamics_residual,
            predict_delta=predict_delta,
            predict_delta_log_phys=predict_delta_log_phys,
            target_idx=torch.tensor(target_idx, dtype=torch.long),
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
            softmax_head=softmax_head,
            allow_partial_simplex=allow_partial_simplex,
        )

    return FlowMapAutoencoder(
        state_dim_in=S_in,
        state_dim_out=S_out,
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
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        softmax_head=softmax_head,
        allow_partial_simplex=allow_partial_simplex,
    )
