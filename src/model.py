#!/usr/bin/env python3
"""
model.py - State-evolution predictors using MLP and SIREN architectures.
Enhanced for chemical kinetics accuracy.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding to better capture temporal dynamics.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: Tensor) -> Tensor:
        # t shape: [batch_size] or [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        device = t.device
        half_dim = self.dim // 2
        
        # Correctly implement the positional encoding frequency basis
        # emb = 1 / (10000^(2i/dim))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        
        # Apply frequencies to time, then sin/cos
        emb = t * inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Pad with zero if dim is odd
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
            
        return emb

class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),  # Expansion
            nn.SiLU(),  # Better for smooth functions
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)

class StateEvolutionPredictor(nn.Module):
    """
    Standard MLP-based predictor.
    
    CRITICAL FIX: The forward pass now implements a proper residual connection.
    The model learns the CHANGE (delta) from the initial conditions, rather than
    being forced into a hardcoded average. This is essential for learning
    systems that evolve significantly over time.
    """
    def __init__(
        self, 
        num_species: int, 
        num_global_vars: int, 
        hidden_dims: List[int], 
        dropout: float,
        use_time_embedding: bool = True,
        time_embedding_dim: int = 32,
        use_residual: bool = True,
        output_activation: str = "identity" # Should learn an unbounded delta
    ):
        super().__init__()
        
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        self.use_residual = use_residual
        
        # Calculate input dimension
        input_dim = num_species + num_global_vars
        if use_time_embedding:
            input_dim += time_embedding_dim
            self.time_embed = TimeEmbedding(time_embedding_dim)
        else:
            input_dim += 1  # Raw time value
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Build MLP layers
        layers = []
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            
            # Add layer norm before each block
            if i > 0: layers.append(nn.LayerNorm(in_dim))

            if use_residual and in_dim == out_dim:
                layers.append(ResidualBlock(in_dim, dropout))
            else:
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                ])
        
        self.mlp_body = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)
        
        # Output activation for the learned delta
        if output_activation == "sigmoid": self.output_act = nn.Sigmoid()
        elif output_activation == "softplus": self.output_act = nn.Softplus()
        elif output_activation == "tanh": self.output_act = nn.Tanh()
        else: self.output_act = nn.Identity()
        
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('silu'))
                if module.bias is not None: nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.output_proj.weight.mul_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        if self.use_time_embedding:
            time, global_vars = global_and_time[..., -1:], global_and_time[..., :-1]
            time_emb = self.time_embed(time.squeeze(-1))
            features = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        else:
            features = x
        
        hidden = self.input_proj(features)
        hidden = self.mlp_body(hidden)
        hidden = self.output_norm(hidden)
        delta = self.output_proj(hidden)
        
        # The network predicts the CHANGE from the initial state.
        delta = self.output_act(delta)
        
        # CRITICAL FIX: Add the learned change to the initial state.
        # This is a proper residual connection.
        final_output = initial_species + delta
            
        return final_output

# --- NEW, SUPERIOR ARCHITECTURE ---
class SineLayer(nn.Module):
    """Linear layer followed by a sine activation. For SIREN models."""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    """Sinusoidal Representation Network - ideal for coordinate-based tasks."""
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int],
                 use_time_embedding: bool = True, time_embedding_dim: int = 32,
                 w0_initial: float = 30.0, w0_hidden: float = 1.0,
                 output_activation: str = "identity"):
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding

        input_dim = num_species + num_global_vars
        if use_time_embedding:
            input_dim += time_embedding_dim
            self.time_embed = TimeEmbedding(time_embedding_dim)
        else:
            input_dim += 1
            
        net = []
        current_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            is_first = (i == 0)
            w0 = w0_initial if is_first else w0_hidden
            net.append(SineLayer(current_dim, dim, is_first=is_first, omega_0=w0))
            current_dim = dim
        
        self.net = nn.Sequential(*net)
        self.output_linear = nn.Linear(current_dim, num_species)
        
        if output_activation == "tanh": self.output_act = nn.Tanh()
        else: self.output_act = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        if self.use_time_embedding:
            time, global_vars = global_and_time[..., -1:], global_and_time[..., :-1]
            time_emb = self.time_embed(time.squeeze(-1))
            features = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        else:
            features = x
        
        coords = self.net(features)
        delta = self.output_linear(coords)
        delta = self.output_act(delta)
        
        return initial_species + delta

def create_prediction_model(
    config: Dict[str, Any], 
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Factory function to create the appropriate model based on config.
    Dispatches to MLP or SIREN creation functions.
    """
    model_type = config.get("model_type", "mlp").lower()
    logger.info(f"Creating model of type: '{model_type}'")
    
    if model_type == "siren":
        model_class = SIREN
        model_config = {
            "w0_initial": config.get("siren_w0_initial", 30.0),
            "w0_hidden": config.get("siren_w0_hidden", 1.0),
        }
    elif model_type == "mlp":
        model_class = StateEvolutionPredictor
        model_config = {
            "use_residual": config.get("use_residual", True),
            "dropout": config.get("dropout", 0.1),
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in configuration.")
    
    # Common parameters for both models
    model_config.update({
        "num_species": len(config["species_variables"]),
        "num_global_vars": len(config["global_variables"]),
        "hidden_dims": config.get("hidden_dims", [256, 256, 256]),
        "use_time_embedding": config.get("use_time_embedding", True),
        "time_embedding_dim": config.get("time_embedding_dim", 32),
        "output_activation": config.get("output_activation", "identity"),
    })

    model = model_class(**model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"{model_class.__name__} created with {trainable_params:,} trainable parameters "
        f"(total: {total_params:,})"
    )
    logger.info(f"Model config: {model_config}")
    
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model

__all__ = ["StateEvolutionPredictor", "SIREN", "create_prediction_model"]