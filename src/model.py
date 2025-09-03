#!/usr/bin/env python3
"""
Flow-map DeepONet Model Architecture
=====================================
Implements a DeepONet architecture for flow-map prediction with support for
multiple target times per anchor point.

Architecture Components:
- Branch network: Processes state and global features to produce basis coefficients
- Trunk network: Maps normalized time differences to temporal basis functions
- Output layer: Combines branch and trunk outputs to predict state evolution

Features:
- Supports both delta (residual) and direct prediction modes
- Configurable dropout in branch and trunk MLPs
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


# ------------------------------ Activations ----------------------------------


def get_activation(name: str) -> nn.Module:
    """
    Create activation function from name.

    Args:
        name: Activation function name

    Returns:
        Activation module

    Raises:
        ValueError: If activation name is not supported
    """
    name_lower = name.lower()

    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    }

    if name_lower not in activation_map:
        supported = ", ".join(sorted(activation_map.keys()))
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Supported activations: {supported}"
        )

    return activation_map[name_lower]


# --------------------------------- MLP ---------------------------------------


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module,
    *,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    """
    Construct a multi-layer perceptron with optional dropout after each hidden activation.

    Args:
        input_dim:  Input feature dimension.
        hidden_dims: List of hidden layer widths.
        output_dim: Output feature dimension.
        activation: Activation module instance (e.g., GELU()).
        dropout_p:  Dropout probability in [0, 1). Applied after each hidden activation.

    Returns:
        nn.Sequential implementing: Linear -> Act -> (Dropout) x L  -> Linear_out
    """
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in map(int, hidden_dims):
        layers.append(nn.Linear(prev, h))
        layers.append(activation)
        if dropout_p and dropout_p > 0.0:
            layers.append(nn.Dropout(p=float(dropout_p)))
        # Use a fresh activation instance for the next block (avoids shared state)
        activation = type(activation)()
        prev = h
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


# -------------------------------- Branch -------------------------------------


class BranchNet(nn.Module):
    """
    Branch network: processes concatenated [y_i, g] -> φ ∈ R^p.
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        activation: nn.Module,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        Args:
            input_dim:  S + G.
            width:      Hidden width for each layer.
            depth:      Number of hidden layers (>=1).
            output_dim: p (basis dimension).
            activation: Activation module (e.g., GELU()).
            dropout_p:  Dropout prob after each hidden activation.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.depth = max(1, int(depth))
        self.output_dim = int(output_dim)

        hidden_dims = [self.width] * self.depth
        self.network = build_mlp(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # [B, p]


# -------------------------------- Trunk --------------------------------------


class TrunkNet(nn.Module):
    """
    Trunk network: maps normalized time input (Δt_norm) to ψ ∈ R^p.
    Accepts t with shapes [B], [B,1], [B,K], or [B,K,1].
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
        *,
        dropout_p: float = 0.0,
    ) -> None:
        """
        Args:
            output_dim: p (basis dimension).
            hidden_dims: list of hidden widths.
            activation: Activation module.
            dropout_p:  Dropout prob after each hidden activation.
        """
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.network = build_mlp(
            input_dim=1,  # scalar time input
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize shapes to [N,1], run MLP, then reshape back to [..., p].
        if t.ndim == 1:          # [B]
            B = t.shape[0]
            out = self.network(t.view(B, 1))
            return out.view(B, 1, self.output_dim)    # [B,1,p]
        if t.ndim == 2:          # [B,K] or [B,1]
            B, K = t.shape
            out = self.network(t.view(B * K, 1))
            return out.view(B, K, self.output_dim)    # [B,K,p]
        if t.ndim == 3:          # [B,K,1]
            B, K, C = t.shape
            if C != 1:
                raise ValueError(f"TrunkNet.forward expects last dim==1, got {t.shape}")
            out = self.network(t.view(B * K, 1))
            return out.view(B, K, self.output_dim)    # [B,K,p]
        raise ValueError(f"Unsupported time tensor shape: {tuple(t.shape)}")


# ------------------------------ DeepONet Core --------------------------------


class FlowMapDeepONet(nn.Module):
    """
    Flow-map DeepONet: y_i, g, Δt_norm -> y_j (direct or residual).
    Always returns [B,K,S] (K=1 is kept as singleton for consistency).
    """

    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        basis_dim: int,
        branch_width: int,
        branch_depth: int,
        trunk_layers: Sequence[int],
        *,
        predict_delta: bool = True,
        trunk_dedup: bool = False,  # reserved for possible future optimization
        activation_name: str = "gelu",
        branch_dropout: float = 0.0,
        trunk_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.S = int(state_dim)
        self.G = int(global_dim)
        self.p = int(basis_dim)
        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = bool(trunk_dedup)

        act = get_activation(activation_name)

        # Branch: [y_i, g] -> φ ∈ R^p
        self.branch = BranchNet(
            input_dim=self.S + self.G,
            width=int(branch_width),
            depth=int(branch_depth),
            output_dim=self.p,
            activation=act,
            dropout_p=float(branch_dropout),
        )

        # Trunk: Δt_norm -> ψ ∈ R^p
        self.trunk = TrunkNet(
            output_dim=self.p,
            hidden_dims=[int(h) for h in trunk_layers],
            activation=act,
            dropout_p=float(trunk_dropout),
        )

        # Final linear maps elementwise φ ⊙ ψ to state dimension
        self.out = nn.Linear(self.p, self.S)

    def forward(
        self,
        y_i: torch.Tensor,     # [B,S] or [B,K,S]
        dt_norm: torch.Tensor, # [B], [B,1], [B,K], or [B,K,1]
        g: torch.Tensor        # [B,G]
    ) -> torch.Tensor:
        """
        Returns predictions as [B,K,S] (K=1 kept as singleton).
        """
        # --------- validate g -----------
        if g.ndim != 2:
            raise ValueError(f"Expected g as [B,G], got {tuple(g.shape)}")
        B = g.shape[0]
        if getattr(self, "G", None) is not None and g.shape[1] != self.G:
            raise ValueError(f"Expected G={self.G}, got {g.shape[1]}")

        # --------- normalize y_i to [B,S] -----------
        if y_i.ndim == 2:
            if y_i.shape[0] != B:
                raise ValueError(f"Batch mismatch: y_i has B={y_i.shape[0]}, g has B={B}")
            y_i_base = y_i  # [B,S]
        elif y_i.ndim == 3:
            if y_i.shape[0] != B:
                raise ValueError(f"Batch mismatch: y_i has B={y_i.shape[0]}, g has B={B}")
            y_i_base = y_i[:, 0, :]  # [B,S]
        else:
            raise ValueError(f"Expected y_i as [B,S] or [B,K,S], got {tuple(y_i.shape)}")
        if y_i_base.shape[1] != self.S:
            raise ValueError(f"Expected state dim S={self.S}, got {y_i_base.shape[1]}")

        # --------- determine K from dt_norm -----------
        if dt_norm.ndim == 1:               # [B] -> K=1
            K = 1
        elif dt_norm.ndim == 2:             # [B,K] or [B,1]
            if dt_norm.shape[0] != B:
                raise ValueError(f"Batch mismatch: dt_norm has B={dt_norm.shape[0]}, g has B={B}")
            K = dt_norm.shape[1]
        elif dt_norm.ndim == 3:             # [B,K,1]
            if dt_norm.shape[-1] != 1:
                raise ValueError(f"Expected dt_norm last dim 1 when 3D, got {tuple(dt_norm.shape)}")
            if dt_norm.shape[0] != B:
                raise ValueError(f"Batch mismatch: dt_norm has B={dt_norm.shape[0]}, g has B={B}")
            K = dt_norm.shape[1]
        else:
            raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)}")

        # --------- branch: [B,S+G] -> [B,p] -----------
        branch_input = torch.cat([y_i_base, g], dim=-1)   # [B, S+G]
        phi = self.branch(branch_input)                   # [B, p]

        # --------- trunk: dt_norm -> [B,K,p] -----------
        psi = self.trunk(dt_norm)                         # [B,1,p] or [B,K,p]
        if psi.ndim == 2:                                 # safety if trunk ever returns [B,p]
            psi = psi.unsqueeze(1)                        # -> [B,1,p]
            K = 1

        # --------- combine & project -----------
        combined = phi.unsqueeze(1) * psi                 # [B,K,p]
        y_pred = self.out(combined)                       # [B,K,S]

        # --------- residual add (if enabled) -----------
        if self.predict_delta:
            if y_i.ndim == 3:
                y_pred = y_pred + y_i                     # [B,K,S]
            else:
                y_pred = y_pred + y_i_base.unsqueeze(1)   # [B,1,S] -> [B,K,S]

        return y_pred                                     # [B,K,S]


# ------------------------------ Factory --------------------------------------


def create_model(config: dict) -> FlowMapDeepONet:
    """
    Factory for FlowMapDeepONet using config["data"] and config["model"].
    Supports model-level "dropout" and optional overrides
    "branch_dropout", "trunk_dropout".
    """
    # --- Resolve data dims ---
    data_cfg = config.get("data", {})
    species_vars = data_cfg.get("target_species_variables") or data_cfg.get("species_variables")
    global_vars = data_cfg.get("global_variables", [])
    if not species_vars:
        raise KeyError("data.species_variables or data.target_species_variables must be set")
    if global_vars is None:
        raise KeyError("data.global_variables must be set (use [] for none)")

    state_dim = len(species_vars)
    global_dim = len(global_vars)

    # --- Model hyperparams ---
    mcfg = config.get("model", {})
    basis_dim     = int(mcfg.get("p", 128))
    branch_width  = int(mcfg.get("branch_width", 512))
    branch_depth  = int(mcfg.get("branch_depth", 3))
    trunk_layers  = [int(h) for h in mcfg.get("trunk_layers", [512, 512])]
    predict_delta = bool(mcfg.get("predict_delta", True))
    trunk_dedup   = bool(mcfg.get("trunk_dedup", False))
    activation    = str(mcfg.get("activation", "gelu"))

    # --- Dropout (single knob, with per-subnet overrides) ---
    dropout        = float(mcfg.get("dropout", 0.0))
    branch_dropout = float(mcfg.get("branch_dropout", dropout))
    trunk_dropout  = float(mcfg.get("trunk_dropout",  dropout))

    return FlowMapDeepONet(
        state_dim=state_dim,
        global_dim=global_dim,
        basis_dim=basis_dim,
        branch_width=branch_width,
        branch_depth=branch_depth,
        trunk_layers=trunk_layers,
        predict_delta=predict_delta,
        trunk_dedup=trunk_dedup,
        activation_name=activation,
        branch_dropout=branch_dropout,
        trunk_dropout=trunk_dropout,
    )
