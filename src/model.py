#!/usr/bin/env python3
"""
model.py - State-evolution predictors using MLP and SIREN architectures.
Enhanced for chemical kinetics accuracy with Feature-wise Linear Modulation (FiLM).
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

# --- Core Utility Modules ---

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding to better capture temporal dynamics.
    JIT-compatible.
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
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        
        # Apply frequencies to time, then sin/cos
        emb = t * inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Pad with zero if dim is odd
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), "constant", 0)
            
        return emb

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    This layer takes a conditioning vector and uses it to apply an affine
    transformation (scale and shift) to a feature map. JIT-compatible.
    """
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        # This linear layer generates the scale (gamma) and shift (beta) parameters
        self.generator = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        # Generate gamma and beta from the conditioning vector
        gamma, beta = torch.chunk(self.generator(condition), 2, dim=-1)
        
        # Apply the modulation: y = gamma * x + beta
        return gamma * features + beta

class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow. JIT-compatible."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),  # Expansion
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)

# --- Standard MLP Predictor ---

class StateEvolutionPredictor(nn.Module):
    """
    Standard MLP-based predictor. The model learns the CHANGE (delta)
    from the initial conditions.
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
        output_activation: str = "identity"
    ):
        super().__init__()
        self.num_species = num_species
        
        # Calculate input dimension by simple concatenation
        input_dim = num_species + num_global_vars
        if use_time_embedding:
            input_dim += time_embedding_dim
            self.time_embed: Optional[TimeEmbedding] = TimeEmbedding(time_embedding_dim)
        else:
            input_dim += 1  # Raw time value
            self.time_embed = None
        
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dims[0])]
        
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            
            if i > 0: layers.append(nn.LayerNorm(in_dim))

            if use_residual and in_dim == out_dim:
                layers.append(ResidualBlock(in_dim, dropout))
            else:
                layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Dropout(dropout)])
        
        self.mlp_body = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)
        
        self.output_act: nn.Module = nn.Identity()
        if output_activation == "sigmoid": self.output_act = nn.Sigmoid()
        elif output_activation == "softplus": self.output_act = nn.Softplus()
        elif output_activation == "tanh": self.output_act = nn.Tanh()

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
        
        if self.time_embed is not None:
            global_and_time = x[..., self.num_species:]
            time, global_vars = global_and_time[..., -1:], global_and_time[..., :-1]
            time_emb = self.time_embed(time.squeeze(-1))
            features = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        else:
            features = x
        
        hidden = self.mlp_body(features)
        hidden = self.output_norm(hidden)
        delta = self.output_proj(hidden)
        delta = self.output_act(delta)
        
        final_output = initial_species + delta
        return final_output

# --- NEW FiLM-ENHANCED MLP PREDICTOR ---

class FiLMPredictor(nn.Module):
    """
    An MLP-based predictor that uses a FiLM layer to condition the species
    evolution on global variables and time. This provides a strong inductive
    bias for learning multiplicative interactions.
    """
    def __init__(
        self, 
        num_species: int, 
        num_global_vars: int, 
        hidden_dims: List[int], 
        dropout: float,
        use_time_embedding: bool = True,
        time_embedding_dim: int = 32,
        condition_dim: int = 128,
        use_residual: bool = True,
        output_activation: str = "identity"
    ):
        super().__init__()
        self.num_species = num_species
        
        # --- 1. Conditioning Network ---
        # This network processes global variables and time into a single conditioning vector.
        condition_input_dim = num_global_vars
        if use_time_embedding:
            self.time_embed: Optional[TimeEmbedding] = TimeEmbedding(time_embedding_dim)
            condition_input_dim += time_embedding_dim
        else:
            self.time_embed = None
            condition_input_dim += 1
            
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )

        # --- 2. Main Processing Network ---
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0]) # Apply FiLM here
        
        body_layers: List[nn.Module] = []
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            
            body_layers.append(nn.LayerNorm(in_dim))
            
            if use_residual and in_dim == out_dim:
                body_layers.append(ResidualBlock(in_dim, dropout))
            else:
                body_layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Dropout(dropout)])

        self.mlp_body = nn.Sequential(*body_layers)
        
        # --- 3. Output Projection ---
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)
        
        self.output_act: nn.Module = nn.Identity()
        if output_activation == "tanh": self.output_act = nn.Tanh()

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
        
        # --- Create Conditioning Vector ---
        if self.time_embed is not None:
            time, global_vars = global_and_time[..., -1:], global_and_time[..., :-1]
            time_emb = self.time_embed(time.squeeze(-1))
            condition_input = torch.cat([global_vars, time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        # --- Main Path with FiLM ---
        hidden = self.input_proj(initial_species)
        hidden = self.film_layer(hidden, condition_vector) # Modulate the state
        hidden = self.mlp_body(hidden)
        
        # --- Final Output ---
        hidden = self.output_norm(hidden)
        delta = self.output_proj(hidden)
        delta = self.output_act(delta)
        
        return initial_species + delta

# --- SIREN Architectures ---

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
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

class FiLMSIREN(nn.Module):
    """
    SIREN conditioned via a FiLM layer. This is a powerful combination for
    learning the implicit neural representation of a system's dynamics as a
    function of its initial and global conditions.
    """
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int],
                 use_time_embedding: bool = True, time_embedding_dim: int = 32,
                 condition_dim: int = 128, w0_initial: float = 30.0,
                 w0_hidden: float = 1.0, output_activation: str = "identity"):
        super().__init__()
        self.num_species = num_species

        # --- 1. Conditioning Network ---
        condition_input_dim = num_global_vars
        if use_time_embedding:
            self.time_embed: Optional[TimeEmbedding] = TimeEmbedding(time_embedding_dim)
            condition_input_dim += time_embedding_dim
        else:
            self.time_embed = None
            condition_input_dim += 1
            
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # --- 2. Main SIREN Network ---
        siren_input_dim = num_species
        self.input_proj = nn.Linear(siren_input_dim, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        
        net: List[nn.Module] = []
        current_dim = hidden_dims[0]
        for i, dim in enumerate(hidden_dims):
            is_first = (i == 0) # This sine layer is the first in the sequence
            w0 = w0_initial if is_first else w0_hidden
            net.append(SineLayer(current_dim, dim, is_first=is_first, omega_0=w0))
            current_dim = dim
        
        self.net = nn.Sequential(*net)
        self.output_linear = nn.Linear(current_dim, num_species)
        
        self.output_act: nn.Module = nn.Identity()
        if output_activation == "tanh": self.output_act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        # --- Create Conditioning Vector ---
        if self.time_embed is not None:
            time, global_vars = global_and_time[..., -1:], global_and_time[..., :-1]
            time_emb = self.time_embed(time.squeeze(-1))
            condition_input = torch.cat([global_vars, time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)

        # --- Main SIREN Path with FiLM ---
        coords = self.input_proj(initial_species)
        coords = self.film_layer(coords, condition_vector) # Modulate the state
        coords = self.net(coords)
        
        # --- Final Output ---
        delta = self.output_linear(coords)
        delta = self.output_act(delta)
        
        return initial_species + delta

# --- Factory Function ---

def create_prediction_model(
    config: Dict[str, Any], 
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Factory function to create the appropriate model based on config.
    Dispatches to MLP or SIREN, with or without FiLM.
    """
    model_type = config.get("model_type", "mlp").lower()
    use_film = config.get("use_film", True) # Default to using FiLM
    
    logger.info(f"Creating model of type: '{model_type}' with FiLM: {use_film}")

    model_class: type[nn.Module]
    
    if model_type == "siren":
        model_class = FiLMSIREN if use_film else SIREN # SIREN is now legacy
        model_config = {
            "w0_initial": config.get("siren_w0_initial", 30.0),
            "w0_hidden": config.get("siren_w0_hidden", 1.0),
        }
    elif model_type == "mlp":
        model_class = FiLMPredictor if use_film else StateEvolutionPredictor
        model_config = {"dropout": config.get("dropout", 0.1)}
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in configuration.")

    # Common parameters for all models
    model_config.update({
        "num_species": len(config["species_variables"]),
        "num_global_vars": len(config["global_variables"]),
        "hidden_dims": config.get("hidden_dims", [256, 256, 256]),
        "use_time_embedding": config.get("use_time_embedding", True),
        "time_embedding_dim": config.get("time_embedding_dim", 32),
        "output_activation": config.get("output_activation", "identity"),
    })

    # Add FiLM-specific parameters if needed
    if use_film:
        model_config.update({"condition_dim": config.get("condition_dim", 128)})
    
    # Add MLP-specific parameters if not using FiLM for original model
    if model_type == "mlp" or (model_type == "mlp_film" and "use_residual" in config):
        model_config.update({"use_residual": config.get("use_residual", True)})
        
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

__all__ = ["StateEvolutionPredictor", "FiLMPredictor", "FiLMSIREN", "create_prediction_model"]