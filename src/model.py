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

from typing import List, Sequence, Optional

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
    Flow-map DeepONet: (y_i, g, Δt_norm) -> y_j
    - Branch takes [y_i, g]
    - Trunk takes Δt only
    - Supports output subset via target_idx
    """

    def __init__(
            self,
            *,
            state_dim_in: int,
            state_dim_out: int,
            global_dim: int,
            basis_dim: int,
            branch_width: int,
            branch_depth: int,
            trunk_layers: Sequence[int],
            predict_delta: bool,
            trunk_dedup: bool,
            activation_name: str,
            branch_dropout: float = 0.0,
            trunk_dropout: float = 0.0,
            target_idx: Optional[torch.Tensor] = None,
            input_idx: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.p = int(basis_dim)

        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = bool(trunk_dedup)

        # Indices
        if target_idx is not None and not isinstance(target_idx, torch.Tensor):
            target_idx = torch.tensor(target_idx, dtype=torch.long)
        if input_idx is not None and not isinstance(input_idx, torch.Tensor):
            input_idx = torch.tensor(input_idx, dtype=torch.long)

        self.register_buffer("target_idx", target_idx if target_idx is not None else None)
        self.register_buffer("input_idx", input_idx if input_idx is not None else None)

        # Activations
        act = get_activation(activation_name)

        # Branch: [y_i_subset, g] -> φ ∈ R^p
        self.branch = BranchNet(
            input_dim=self.S_in + self.G,
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

        # Projection from basis to targets
        self.out = nn.Linear(self.p, self.S_out)

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_i:     [B,S_full] or [B,1,S_full] current state (FULL species dimension)
            dt_norm: [B,K] or [B,K,1] normalized Δt (dt-spec)
            g:       [B,G] globals
        Returns:
            y_pred:  [B,K,S_out]
        """
        B = y_i.shape[0]

        # Normalize shapes
        if y_i.ndim == 3:
            y_i_base = y_i[:, 0, :]  # [B,S_full]
        elif y_i.ndim == 2:
            y_i_base = y_i
        else:
            raise ValueError(f"Unexpected y_i shape {tuple(y_i.shape)}")

        if dt_norm.ndim == 1:
            dt_in = dt_norm.view(B, 1)  # [B,1]
        elif dt_norm.ndim == 2:
            dt_in = dt_norm  # [B,K]
        elif dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_in = dt_norm.squeeze(-1)  # [B,K]
        else:
            raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)}")
        K = dt_in.shape[1]

        # Select branch inputs
        if self.input_idx is not None:
            y_in = y_i_base.index_select(1, self.input_idx)  # [B,S_in]
        else:
            y_in = y_i_base

        # Branch and trunk
        phi = self.branch(torch.cat([y_in, g], dim=-1))  # [B,p]
        psi = self.trunk(dt_in)  # [B,K,p]

        # Combine & project
        combined = phi.unsqueeze(1) * psi  # [B,K,p]
        y_pred = self.out(combined)  # [B,K,S_out]

        # Optional residual add (still in normalized space)
        if self.predict_delta:
            if (self.S_out == y_i_base.shape[-1]) and (self.target_idx is None):
                base = y_i_base.unsqueeze(1)  # [B,1,S_full] -> [B,K,S_full]
                if base.shape[1] != K:
                    base = base.expand(-1, K, -1)
                y_pred = y_pred + base
            else:
                if self.target_idx is None:
                    raise RuntimeError("predict_delta=True requires target_idx when S_out != S_full")
                base_subset = y_i_base.index_select(1, self.target_idx)  # [B,S_out]
                base_subset = base_subset.unsqueeze(1)  # [B,1,S_out] -> [B,K,S_out]
                if base_subset.shape[1] != K:
                    base_subset = base_subset.expand(-1, K, -1)
                y_pred = y_pred + base_subset

        return y_pred


# ------------------------------ Factory --------------------------------------


def create_model(config: dict) -> FlowMapDeepONet:
    """
    Build FlowMapDeepONet with optional input/target subsets.

    Config keys (under data):
      - species_variables: full ordered list of species in shards (REQUIRED)
      - global_variables : list of global variable names (may be [])
      - input_species    : optional list of species names to FEED INTO THE BRANCH
                           (defaults to species_variables i.e., all inputs)
      - target_species   : optional list of species names to PREDICT
                           (defaults to species_variables i.e., all targets)
    """
    import torch

    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    species_vars = list(data_cfg.get("species_variables") or [])
    if not species_vars:
        raise KeyError("config.data.species_variables must be set and non-empty")
    global_vars = list(data_cfg.get("global_variables", []))

    # Optional subsets
    input_vars  = list(data_cfg.get("input_species")  or species_vars)
    target_vars = list(data_cfg.get("target_species") or species_vars)

    # Map names -> indices in the FULL species list
    name_to_idx = {name: i for i, name in enumerate(species_vars)}

    try:
        input_idx  = [name_to_idx[name] for name in input_vars]
    except KeyError as e:
        raise KeyError(f"config.data.input_species contains unknown name: {e.args[0]!r} "
                       f"(not found in species_variables)") from None

    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(f"config.data.target_species contains unknown name: {e.args[0]!r} "
                       f"(not found in species_variables)") from None

    # Dimensions
    state_dim_in  = len(input_idx)
    state_dim_out = len(target_idx)
    global_dim    = len(global_vars)

    # Build model
    return FlowMapDeepONet(
        state_dim_in=state_dim_in,
        state_dim_out=state_dim_out,
        global_dim=global_dim,
        basis_dim=int(model_cfg.get("p", 128)),
        branch_width=int(model_cfg.get("branch_width", 512)),
        branch_depth=int(model_cfg.get("branch_depth", 3)),
        trunk_layers=[int(h) for h in model_cfg.get("trunk_layers", [512, 512])],
        predict_delta=bool(model_cfg.get("predict_delta", True)),
        trunk_dedup=bool(model_cfg.get("trunk_dedup", False)),
        activation_name=str(model_cfg.get("activation", "gelu")),
        branch_dropout=float(model_cfg.get("branch_dropout", model_cfg.get("dropout", 0.0))),
        trunk_dropout=float(model_cfg.get("trunk_dropout",  model_cfg.get("dropout", 0.0))),
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        input_idx=torch.tensor(input_idx, dtype=torch.long),
    )