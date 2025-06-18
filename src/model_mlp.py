#!/usr/bin/env python3
"""
model.py - A state-evolution predictor using a Multi-Layer Perceptron (MLP).
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
    """Sinusoidal time embedding to better capture temporal dynamics."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: Tensor) -> Tensor:
        # t shape: [batch_size] or [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        arange_tensor = torch.arange(half_dim, device=device)
        emb = torch.exp(arange_tensor * -emb)
        # ============================= FIX: END =============================

        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
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
    def __init__(
        self, 
        num_species: int, 
        num_global_vars: int, 
        hidden_dims: List[int], 
        dropout: float,
        use_time_embedding: bool = True,
        time_embedding_dim: int = 32,
        use_residual: bool = True,
        output_activation: str = "sigmoid"  # For normalized outputs [0,1]
    ):
        super().__init__()
        
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        self.use_residual = use_residual
        self.output_activation = output_activation
        
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
            if i > 0:
                # Transition between dimensions
                if hidden_dims[i] != hidden_dims[i-1]:
                    layers.extend([
                        nn.LayerNorm(hidden_dims[i-1]),
                        nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                        nn.SiLU(),
                        nn.Dropout(dropout)
                    ])
                elif use_residual:
                    # Use residual block when dimensions match
                    layers.append(ResidualBlock(hidden_dims[i], dropout))
                else:
                    # Standard block
                    layers.extend([
                        nn.LayerNorm(hidden_dims[i]),
                        nn.Linear(hidden_dims[i], hidden_dims[i]),
                        nn.SiLU(),
                        nn.Dropout(dropout)
                    ])
        
        self.mlp_body = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)
        
        # Output activation for chemical constraints
        if output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_activation == "softplus":
            self.output_act = nn.Softplus()
        else:
            self.output_act = nn.Identity()
        
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Careful initialization for chemical kinetics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier for most layers
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Special initialization for output layer (near identity)
        # This helps the model learn residuals from initial conditions
        with torch.no_grad():
            self.output_proj.weight.mul_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        # Extract components
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        # Separate time from global variables
        if self.use_time_embedding:
            time = global_and_time[..., -1:]
            global_vars = global_and_time[..., :-1]
            
            # Create time embedding
            time_emb = self.time_embed(time.squeeze(-1))
            
            # Concatenate all inputs
            features = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        else:
            features = x
        
        # Forward through network
        hidden = self.input_proj(features)
        hidden = self.mlp_body(hidden)
        hidden = self.output_norm(hidden)
        output = self.output_proj(hidden)
        
        # Apply output activation
        output = self.output_act(output)
        
        # Skip connection from initial conditions
        # This helps model learn the change rather than absolute values
        # Using a weighted combination that can be learned
        output = output * 0.9 + initial_species * 0.1
            
        return output

def create_prediction_model(
    config: Dict[str, Any], 
    device: Optional[Union[str, torch.device]] = None
) -> StateEvolutionPredictor:
    logger.info("Configuration for MLP model loaded.")
    
    # Get model configuration with sensible defaults for chemical kinetics
    model_config = {
        "use_time_embedding": config.get("use_time_embedding", True),
        "time_embedding_dim": config.get("time_embedding_dim", 32),
        "use_residual": config.get("use_residual", True),
        "output_activation": config.get("output_activation", "sigmoid"),
    }
    
    model = StateEvolutionPredictor(
        num_species=len(config["species_variables"]),
        num_global_vars=len(config["global_variables"]),
        hidden_dims=config.get("hidden_dims", [512, 512, 512, 256]),  # Deeper default
        dropout=config.get("dropout", 0.1),
        **model_config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"StateEvolutionPredictor created with {trainable_params:,} trainable parameters "
        f"(total: {total_params:,})"
    )
    
    # Log model configuration
    logger.info(f"Model config: hidden_dims={config.get('hidden_dims')}, {model_config}")
    
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model

__all__ = ["StateEvolutionPredictor", "create_prediction_model"]