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

The model supports both delta (residual) and direct prediction modes.
"""

from __future__ import annotations

from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module
) -> nn.Sequential:
    """
    Construct a multi-layer perceptron.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Dimensions of hidden layers
        output_dim: Output feature dimension
        activation: Activation function module
        
    Returns:
        Sequential MLP network
    """
    layers: List[nn.Module] = []
    current_dim = input_dim
    
    # Add hidden layers with activations
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(activation)
        current_dim = int(hidden_dim)
    
    # Add final output layer (no activation)
    layers.append(nn.Linear(current_dim, output_dim))
    
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """
    Branch network of DeepONet.
    
    Processes the concatenated state vector and global parameters to produce
    basis coefficients phi in R^p.
    
    Architecture:
        Input: [y_i, g] with dimension S + G
        Hidden layers: Specified width and depth
        Output: phi with dimension p
    """
    
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        activation: nn.Module
    ):
        """
        Initialize branch network.
        
        Args:
            input_dim: Dimension of input features (S + G)
            width: Width of hidden layers
            depth: Number of hidden layers
            output_dim: Dimension of output basis coefficients (p)
            activation: Activation function module
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.width = width
        self.depth = max(1, int(depth))
        self.output_dim = output_dim
        
        # Build network architecture
        if self.depth == 1:
            # Single hidden layer: input -> hidden -> output
            self.network = nn.Sequential(
                nn.Linear(input_dim, width),
                activation,
                nn.Linear(width, output_dim)
            )
        else:
            # Multiple hidden layers
            hidden_dims = [width] * (self.depth - 1)
            self.network = build_mlp(
                input_dim, hidden_dims, output_dim, activation
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through branch network.
        
        Args:
            x: Input tensor of shape [B, S+G]
            
        Returns:
            Basis coefficients phi of shape [B, p]
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """
    Trunk network of DeepONet.
    
    Maps normalized time differences to temporal basis functions psi in R^p.
    Handles both single and multiple time inputs by reshaping.
    
    Architecture:
        Input: dt_norm (scalar per time point)
        Hidden layers: Configurable MLP
        Output: psi with dimension p
    """
    
    def __init__(
        self,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module
    ):
        """
        Initialize trunk network.
        
        Args:
            output_dim: Dimension of output basis functions (p)
            hidden_dims: Dimensions of hidden layers
            activation: Activation function module
        """
        super().__init__()
        
        # Default architecture if not specified
        if len(hidden_dims) == 0:
            hidden_dims = [256, 256]
        
        self.output_dim = output_dim
        self.hidden_dims = [int(d) for d in hidden_dims]
        
        # Build MLP from scalar input to p-dimensional output
        self.network = build_mlp(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            activation=activation
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk network.
        
        Args:
            t: Normalized time differences of shape [B, K] or [B]
            
        Returns:
            Temporal basis functions psi of shape [B, K, p] or [B, p]
        """
        # Ensure input has feature dimension for linear layer
        if t.ndim == 1:
            t_input = t.unsqueeze(-1)  # [B] -> [B, 1]
        elif t.ndim == 2:
            original_shape = t.shape
            t_input = t.reshape(-1, 1)  # [B, K] -> [B*K, 1]
        else:
            t_input = t.unsqueeze(-1) if t.shape[-1] != 1 else t
        
        # Process through network
        psi = self.network(t_input)
        
        # Reshape back to original batch structure if needed
        if t.ndim == 2:
            B, K = original_shape
            psi = psi.reshape(B, K, self.output_dim)
        
        return psi


class FlowMapDeepONet(nn.Module):
    """
    DeepONet for flow-map prediction.
    
    Combines branch and trunk networks to predict state evolution:
        1. Branch network: phi = Branch([y_i, g]) -> [B, p]
        2. Trunk network: psi = Trunk(dt_norm) -> [B, K, p]
        3. Combination: f = phi * psi (element-wise product)
        4. Output: y_pred = Linear(f) -> [B, K, S]
        5. Optional residual: y_pred += y_i (if predict_delta=True)
    
    The model always returns shape [B, K, S] even when K=1 for consistency.
    The trainer handles shape harmonization with targets.
    """
    
    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        basis_dim: int = 256,
        branch_width: int = 1024,
        branch_depth: int = 3,
        trunk_layers: Sequence[int] = (512, 512),
        predict_delta: bool = True,
        trunk_dedup: bool = False,
        activation_name: str = "gelu",
    ):
        """
        Initialize Flow-map DeepONet.
        
        Args:
            state_dim: Dimension of state variables (S)
            global_dim: Dimension of global parameters (G)
            basis_dim: Dimension of basis functions (p)
            branch_width: Width of branch network hidden layers
            branch_depth: Depth of branch network
            trunk_layers: Hidden layer dimensions for trunk network
            predict_delta: Whether to predict residual (y_j - y_i) or direct y_j
            trunk_dedup: Flag for deduplication (NOT IMPLEMENTED)
            activation_name: Name of activation function 
                            ('relu', 'gelu', 'silu', 'tanh', 'leakyrelu')
        
        Raises:
            NotImplementedError: If trunk_dedup is set to True
        """
        super().__init__()
        
        # Check for unimplemented features
        if trunk_dedup:
            raise NotImplementedError("trunk_dedup functionality is not implemented. ")
        
        # Store dimensions
        self.S = int(state_dim)
        self.G = int(global_dim)
        self.p = int(basis_dim)
        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = False  # Always False since not implemented
        
        # Get activation function
        activation = get_activation(activation_name)
        
        # Initialize subnetworks
        self.branch = BranchNet(
            input_dim=self.S + self.G,
            width=branch_width,
            depth=branch_depth,
            output_dim=self.p,
            activation=activation
        )
        
        self.trunk = TrunkNet(
            output_dim=self.p,
            hidden_dims=trunk_layers,
            activation=activation
        )
        
        # Output projection (shared across all time points)
        self.output_layer = nn.Linear(self.p, self.S)
    
    def forward(
        self,
        y_i: torch.Tensor,
        dt_norm: torch.Tensor,
        g: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Flow-map DeepONet.
        
        Args:
            y_i: State at anchor time, shape [B, S]
            dt_norm: Normalized time differences, shape [B, K] or [B] or [B, 1]
            g: Global parameters, shape [B, G]
            
        Returns:
            Predicted states at target times, shape [B, K, S]
            
        Raises:
            AssertionError: If input shapes are invalid
        """
        batch_size = y_i.shape[0]
        
        # Validate input shapes
        assert y_i.dim() == 2 and y_i.shape[1] == self.S, \
            f"Expected y_i shape [B, {self.S}], got {tuple(y_i.shape)}"
        assert g.dim() == 2 and g.shape[1] == self.G, \
            f"Expected g shape [B, {self.G}], got {tuple(g.shape)}"
        
        # Process branch network
        branch_input = torch.cat([y_i, g], dim=-1)  # [B, S+G]
        phi = self.branch(branch_input)  # [B, p]
        
        # Normalize dt_norm shape
        if dt_norm.dim() == 1:
            dt_norm = dt_norm.view(batch_size, 1)
        elif dt_norm.dim() == 2:
            pass
        else:
            raise ValueError(f"dt_norm should be [B] or [B, K], got {tuple(dt_norm.shape)}")
        
        batch_size_dt, K = dt_norm.shape
        assert batch_size_dt == batch_size, \
            f"Batch dimension mismatch: y_i has {batch_size}, dt_norm has {batch_size_dt}"
        
        # Process trunk network
        psi = self.trunk(dt_norm)  # [B, K, p]
        
        # Combine branch and trunk outputs
        # phi: [B, p] -> [B, 1, p] for broadcasting
        # psi: [B, K, p]
        combined = phi.unsqueeze(1) * psi  # [B, K, p]
        
        # Generate output predictions
        y_pred = self.output_layer(combined)  # [B, K, S]
        
        # Add residual connection if configured
        if self.predict_delta:
            y_pred = y_pred + y_i.unsqueeze(1)  # [B, K, S] + [B, 1, S]
        
        return y_pred


def create_model(config: dict) -> FlowMapDeepONet:
    """
    Factory function to create Flow-map DeepONet from configuration.
    
    Args:
        config: Configuration dictionary containing:
            - data.species_variables or data.target_species_variables
            - data.global_variables
            - model.* parameters (optional)
            
    Returns:
        Configured FlowMapDeepONet instance
        
    Raises:
        KeyError: If required configuration keys are missing
    """
    # Extract data dimensions
    data_config = config.get("data", {})
    species_vars = list(
        data_config.get("target_species_variables",
                        data_config.get("species_variables", []))
    )
    global_vars = list(data_config.get("global_variables", []))
    
    if not species_vars:
        raise KeyError(
            "Configuration must provide data.species_variables "
            "or data.target_species_variables"
        )
    if global_vars is None:
        raise KeyError("Configuration must provide data.global_variables")
    
    state_dim = len(species_vars)
    global_dim = len(global_vars)
    
    # Extract model hyperparameters with defaults
    model_config = config.get("model", {})
    basis_dim = int(model_config.get("p", 256))
    branch_width = int(model_config.get("branch_width", 1024))
    branch_depth = int(model_config.get("branch_depth", 3))
    trunk_layers = list(model_config.get("trunk_layers", [512, 512]))
    predict_delta = bool(model_config.get("predict_delta", True))
    trunk_dedup = bool(model_config.get("trunk_dedup", False))
    activation_name = str(model_config.get("activation", "gelu"))
    
    return FlowMapDeepONet(
        state_dim=state_dim,
        global_dim=global_dim,
        basis_dim=basis_dim,
        branch_width=branch_width,
        branch_depth=branch_depth,
        trunk_layers=trunk_layers,
        predict_delta=predict_delta,
        trunk_dedup=trunk_dedup,
        activation_name=activation_name,
    )