#!/usr/bin/env python3
"""
model.py - SIREN neural network model for chemical evolution prediction.

This module implements a Sinusoidal Representation Network (SIREN) with 
Feature-wise Linear Modulation (FiLM) for conditional predictions.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --- Constants ---
DEFAULT_TIME_EMBEDDING_DIM = 32
DEFAULT_CONDITION_DIM = 128
DEFAULT_OMEGA_0 = 30.0
MATMUL_PRECISION = "high"
SIREN_INIT_SCALE = 1.0

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision(MATMUL_PRECISION)


# --- Core Helper Modules ---

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding with learnable scaling and normalization.
    
    Creates position embeddings similar to those used in Transformers,
    allowing the model to understand temporal relationships.
    """
    
    def __init__(self, dim: int, learnable_scale: bool = True, max_period: float = 10000.0) -> None:
        """
        Initialize time embedding module.
        
        Args:
            dim: Dimension of the embedding
            learnable_scale: Whether to use learnable scale and shift parameters
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Pre-compute frequency components for efficiency
        half_dim = self.dim // 2
        denominator = max(half_dim - 1.0, 1.0)
        freqs = torch.arange(0, half_dim).float() / denominator
        self.register_buffer('inv_freq', 1.0 / (self.max_period ** freqs))
        
        # Learnable scaling and shift for better adaptation
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
            self.shift = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('shift', torch.zeros(1))

    def forward(self, t: Tensor) -> Tensor:
        """
        Apply sinusoidal encoding to time values.
        
        Args:
            t: Time tensor of shape (batch,) or (batch, 1)
            
        Returns:
            Time embeddings of shape (batch, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        if self.dim < 2:
            return t.expand(-1, self.dim)
        
        # Apply scaling and compute embeddings
        t_scaled = (t.float() + self.shift) * self.scale
        emb = t_scaled * self.inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0.0)
        
        # Normalize to prevent extreme values
        emb = emb / math.sqrt(self.dim)
        
        return emb.to(t.dtype)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    FiLM allows the network to modulate features based on conditioning information,
    enabling the model to adapt its behavior based on global conditions.
    """
    
    def __init__(self, condition_dim: int, feature_dim: int, 
                 use_norm: bool = True, residual: bool = True) -> None:
        """
        Initialize FiLM layer.
        
        Args:
            condition_dim: Dimension of conditioning vector
            feature_dim: Dimension of features to modulate
            use_norm: Whether to normalize features before modulation
            residual: Whether to use residual connection
        """
        super().__init__()
        self.use_norm = use_norm
        self.residual = residual
        
        # Generate gamma (scale) and beta (shift) parameters
        self.generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        # Initialize last layer to be near-identity transform at the beginning
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        
        if use_norm:
            self.norm = nn.LayerNorm(feature_dim)
        else:
            self.norm = nn.Identity()
    
    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        """
        Apply feature-wise modulation.
        
        Args:
            features: Features to modulate (batch, feature_dim)
            condition: Conditioning vector (batch, condition_dim)
            
        Returns:
            Modulated features (batch, feature_dim)
        """
        if self.use_norm:
            features_normed = self.norm(features)
        else:
            features_normed = features
        
        # Generate modulation parameters
        params = self.generator(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        
        # Apply affine transformation: (1 + gamma) * features + beta
        modulated = (1 + gamma) * features_normed + beta
        
        if self.residual:
            return features + modulated
        else:
            return modulated


# --- SIREN-Specific Modules ---
class SineLayer(nn.Module):
    """
    SIREN sine activation layer with improved initialization and numerical stability.
    
    This layer implements the sine activation function with proper weight initialization
    as described in the SIREN paper (Sitzmann et al., 2020).
    """
    
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 is_first: bool = False,
                 omega_0: float = DEFAULT_OMEGA_0) -> None:
        """
        Initialize SIREN sine layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
            is_first: Whether this is the first layer (affects initialization)
            omega_0: Frequency scaling factor
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper recommendations."""
        with torch.no_grad():
            if self.is_first:
                # More conservative initialization for first layer
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Account for sine activation variance
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sine activation after linear transformation.
        
        Args:
            x: Input tensor
            
        Returns:
            sin(omega_0 * (Wx + b))
        """
        return torch.sin(self.omega_0 * self.linear(x))


class FiLM_SIREN(nn.Module):
    """
    SIREN model with FiLM conditioning for chemical evolution prediction.
    
    This model combines the expressiveness of SIREN's periodic activations
    with FiLM's conditional modulation for accurate chemical species prediction
    under varying conditions.
    """
    
    def __init__(
        self, num_species: int, num_global_vars: int, hidden_dims: List[int],
        use_time_embedding: bool, time_embedding_dim: int, condition_dim: int,
        w0_initial: float, w0_hidden: float, final_activation: bool = True, **kwargs
    ) -> None:
        """
        Initialize FiLM-SIREN model.
        
        Args:
            num_species: Number of chemical species
            num_global_vars: Number of global variables (e.g., temperature, pressure)
            hidden_dims: List of hidden layer dimensions
            use_time_embedding: Whether to use sinusoidal time embedding
            time_embedding_dim: Dimension of time embedding
            condition_dim: Dimension of conditioning vector
            w0_initial: Omega_0 for first SIREN layer
            w0_hidden: Omega_0 for hidden SIREN layers
            final_activation: Whether to apply tanh to final output
            **kwargs: Additional unused arguments for compatibility
        """
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        
        # Time embedding module
        self.time_embed = TimeEmbedding(
            time_embedding_dim, learnable_scale=True
        ) if use_time_embedding else None
        
        # Conditioning network - processes global variables and time
        time_dim = time_embedding_dim if use_time_embedding else 1
        condition_input_dim = num_global_vars + time_dim
        self.conditioning_net = nn.Sequential(
            nn.LayerNorm(condition_input_dim),
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Input projection and FiLM modulation
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0], use_norm=False, residual=False)
        
        # SIREN layers
        siren_layers = []
        in_dim = hidden_dims[0]
        
        for i, out_dim in enumerate(hidden_dims):
            is_first = (i == 0)
            omega = w0_initial if is_first else w0_hidden
            siren_layers.append(SineLayer(in_dim, out_dim, is_first=is_first, omega_0=omega))
            in_dim = out_dim
        
        self.net = nn.Sequential(*siren_layers)
        
        # Output layer
        self.output_linear = nn.Linear(in_dim, num_species)
        if final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, num_species + num_global_vars + 1)
               where the last dimension contains:
               - initial species concentrations
               - global variables (temperature, pressure, etc.)
               - time value
               
        Returns:
            Predicted species concentrations at the query time
        """
        # Split input into components
        initial_species = x[:, :self.num_species]
        global_and_time = x[:, self.num_species:]
        
        # Process conditioning information
        if self.use_time_embedding and self.time_embed is not None:
            time_emb = self.time_embed(global_and_time[:, -1])
            condition_input = torch.cat([global_and_time[:, :-1], time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        # Process through SIREN with FiLM modulation
        coords = self.input_proj(initial_species)
        coords = self.film_layer(coords, condition_vector)
        coords = self.net(coords)
        delta = self.output_linear(coords)
        delta = self.final_activation(delta)
        
        # Residual connection - predict change from initial state
        return initial_species + delta


# --- Factory Function ---

def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Factory function to create and initialize the SIREN prediction model.
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Optional device to place model on
        
    Returns:
        Initialized SIREN model
    """
    model_params = config["model_hyperparameters"]
    data_spec = config["data_specification"]
    
    # Verify model type is SIREN
    model_type = model_params.get("model_type", "siren").lower()
    if model_type != "siren":
        logger.warning(f"Model type '{model_type}' requested but only SIREN is implemented. Using SIREN.")
    
    logger.info("Creating SIREN model")
    
    # Model arguments
    model_args = {
        "num_species": len(data_spec["species_variables"]),
        "num_global_vars": len(data_spec["global_variables"]),
        "hidden_dims": model_params["hidden_dims"],
        "use_time_embedding": model_params.get("use_time_embedding", True),
        "time_embedding_dim": model_params.get("time_embedding_dim", DEFAULT_TIME_EMBEDDING_DIM),
        "condition_dim": model_params.get("condition_dim", DEFAULT_CONDITION_DIM),
        "w0_initial": model_params.get("siren_w0_initial", DEFAULT_OMEGA_0),
        "w0_hidden": model_params.get("siren_w0_hidden", DEFAULT_OMEGA_0),
        "final_activation": model_params.get("final_activation", True),
    }
    
    # Create model
    model = FiLM_SIREN(**model_args)
    
    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"FiLM_SIREN created with {trainable_params:,} trainable parameters."
    )
    logger.debug(f"Model config: {model_args}")
    
    # Move to device if specified
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model


def export_model_jit(
    model: nn.Module,
    example_input: Tensor,
    save_path: Union[str, Path],
    optimize: bool = True
) -> None:
    """
    Export model as TorchScript JIT module for optimized inference.
    
    Args:
        model: Model to export
        example_input: Example input tensor for tracing
        save_path: Path to save the exported model
        optimize: Whether to apply TorchScript optimizations
    """
    save_path = Path(save_path)
    
    # Set eval mode for inference
    model.eval()
    
    with torch.no_grad():
        # Try scripting first, fall back to tracing if needed
        try:
            scripted_model = torch.jit.script(model)
            logger.info("Model successfully scripted.")
        except Exception as e:
            logger.warning(f"JIT script failed: {e}. Trying trace...")
            scripted_model = torch.jit.trace(model, example_input)
        
        # Apply optimizations if requested
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        # Verify output matches original model
        test_output = scripted_model(example_input)
        original_output = model(example_input)
        
        # Use relaxed tolerance for JIT comparison
        if not torch.allclose(test_output, original_output, rtol=1e-4, atol=1e-5):
            logger.warning("JIT output differs from original model!")
    
    # Save the exported model
    scripted_model.save(str(save_path))
    logger.info(f"JIT model saved to: {save_path}")


__all__ = ["FiLM_SIREN", "create_prediction_model", "export_model_jit"]