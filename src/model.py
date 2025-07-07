#!/usr/bin/env python3
"""
model.py - Neural network models for chemical evolution prediction.

This module implements two architectures:
1. SIREN (Sinusoidal Representation Network) with FiLM conditioning
2. A deep residual MLP for robust vector-to-vector regression.

Both models predict chemical species concentrations at arbitrary time points
given initial conditions and are fully JIT-compatible.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants
MATMUL_PRECISION = "high"

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision(MATMUL_PRECISION)


# =============================================================================
# Common Helper Modules
# =============================================================================

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


# =============================================================================
# SIREN-Specific Modules
# =============================================================================

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
            nn.GELU(),
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


class SineLayer(nn.Module):
    """
    SIREN sine activation layer with improved initialization and numerical stability.
    
    This layer implements the sine activation function with proper weight initialization
    as described in the SIREN paper (Sitzmann et al., 2020).
    """
    
    def __init__(self, in_features: int,
                 out_features: int,
                 omega_0: float,
                 *,
                 bias: bool = True,
                 is_first: bool = False,
                 siren_init_scale: float = 1.0) -> None:
        """
        Initialize SIREN sine layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            omega_0: Frequency scaling factor
            bias: Whether to use bias
            is_first: Whether this is the first layer (affects initialization)
            siren_init_scale: Scale factor for initialization
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.siren_init_scale = siren_init_scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper recommendations."""
        with torch.no_grad():
            if self.is_first:
                # More conservative initialization for first layer
                bound = self.siren_init_scale / self.linear.in_features
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


# =============================================================================
# MLP/ResNet-Specific Modules
# =============================================================================

class FourierFeatures(nn.Module):
    """
    Projects a scalar input vector into a higher-dimensional space using
    random Fourier features, allowing the model to learn high-frequency functions.
    """
    def __init__(self, in_features: int, out_features: int, scale: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be an even number.")
        # Ensure scale is reasonable to prevent numerical instability
        scale = min(scale, 10.0)
        self.register_buffer('freqs', torch.randn(in_features, out_features // 2) * scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate Fourier features.
        
        Args:
            x: Input tensor of shape (batch, in_features)
            
        Returns:
            Fourier features of shape (batch, out_features)
        """
        x_proj = x @ self.freqs  # Shape: [batch, out_features // 2]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FeedForwardBlock(nn.Module):
    """
    A standard feed-forward block (MLP block) for vector processing.
    This is the core building block of the ChemicalResNet model.
    """
    def __init__(self, channels: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward transformation with residual connection.
        
        Args:
            x: Input tensor of shape (batch, channels)
            
        Returns:
            Output tensor of shape (batch, channels)
        """
        return self.norm(x + self.net(x))


# =============================================================================
# Main Models
# =============================================================================

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
        w0_initial: float, w0_hidden: float, siren_init_scale: float, **kwargs
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
            siren_init_scale: Scale factor for SIREN initialization
            **kwargs: Additional unused arguments for compatibility
        """
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        self.time_embedding_dim = time_embedding_dim if use_time_embedding else 1
        
        # Time embedding module
        self.time_embed = TimeEmbedding(
            self.time_embedding_dim, learnable_scale=True
        )
        
        # Conditioning network - processes global variables and time
        condition_input_dim = num_global_vars + self.time_embedding_dim
        self.conditioning_net = nn.Sequential(
            nn.LayerNorm(condition_input_dim),
            nn.Linear(condition_input_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # FiLM modulation for initial species coordinates
        self.film_layer = FiLMLayer(condition_dim, num_species, use_norm=False, residual=False)
        
        # SIREN layers
        siren_layers = []
        in_dim = num_species
        
        for i, out_dim in enumerate(hidden_dims):
            is_first = (i == 0)
            omega = w0_initial if is_first else w0_hidden
            siren_layers.append(SineLayer(
                in_dim, out_dim, is_first=is_first, 
                omega_0=omega, siren_init_scale=siren_init_scale
            ))
            in_dim = out_dim
        
        self.net = nn.Sequential(*siren_layers)
        
        # Output layer
        self.output_linear = nn.Linear(in_dim, num_species)

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
        time_raw = global_and_time[:, -1]
        time_emb = self.time_embed(time_raw)
        
        # Use appropriate time representation based on configuration
        if not self.use_time_embedding:
            time_emb = time_raw.unsqueeze(-1)
        
        condition_input = torch.cat([global_and_time[:, :-1], time_emb], dim=-1)
        condition_vector = self.conditioning_net(condition_input)
        
        # Process through SIREN with FiLM modulation
        coords = self.film_layer(initial_species, condition_vector)
        coords = self.net(coords)
        delta = self.output_linear(coords)
        
        prediction = initial_species + delta
        return torch.clamp_min(prediction, 0.0)


class ChemicalResNet(nn.Module):
    """
    A deep residual MLP for predicting chemical state evolution.
    
    This model uses Fourier features and feed-forward blocks for
    vector-to-vector regression, suitable for chemical kinetics problems.
    """
    
    def __init__(
        self, num_species: int, num_global_vars: int, hidden_dims: List[int],
        use_time_embedding: bool, time_embedding_dim: int,
        fourier_features: int = 256, fourier_scale: float = 1.0,
        mlp_ratio: float = 2.0, dropout: float = 0.1, 
        output_activation: str = "softplus", **kwargs
    ):
        """
        Initialize ChemicalResNet model.
        
        Args:
            num_species: Number of chemical species
            num_global_vars: Number of global variables
            hidden_dims: List of hidden layer dimensions
            use_time_embedding: Whether to use sinusoidal time embedding
            time_embedding_dim: Dimension of time embedding
            fourier_features: Number of random Fourier features
            fourier_scale: Scale for Fourier feature frequencies
            mlp_ratio: Expansion ratio for MLP blocks
            dropout: Dropout rate
            output_activation: Output activation function
            **kwargs: Additional unused arguments for compatibility
        """
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        self.time_embedding_dim = time_embedding_dim if use_time_embedding else 1
        
        # Time embedding module
        self.time_embed = TimeEmbedding(
            self.time_embedding_dim, learnable_scale=True
        )
        
        # Calculate input dimension
        input_dim = num_species + num_global_vars + self.time_embedding_dim
        
        # Fourier feature embedding
        self.fourier_embed = FourierFeatures(
            input_dim, fourier_features, scale=fourier_scale
        )
        
        # Lifting layer to project to hidden dimension
        self.lifting = nn.Sequential(
            nn.Linear(input_dim + fourier_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Processing blocks (corrected architecture)
        self.processing_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            if h_dim != current_dim:
                # Dimension change layer
                self.processing_blocks.append(
                    nn.Linear(current_dim, h_dim)
                )
                current_dim = h_dim
            # Add feed-forward block
            self.processing_blocks.append(
                FeedForwardBlock(h_dim, mlp_ratio, dropout)
            )
        
        # Projection to output
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * 2, num_species)
        )
        
        # Output activation
        if output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_activation == "softplus":
            self.output_act = nn.Softplus()
        else:
            self.output_act = nn.Identity()
        
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """Initialize parameters carefully for chemical kinetics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for projection layer
        with torch.no_grad():
            if isinstance(self.projection[-1], nn.Linear):
                self.projection[-1].weight.mul_(0.01)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, num_species + num_global_vars + 1)
               
        Returns:
            Predicted species concentrations at the query time
        """
        # Split input
        initial_species = x[:, :self.num_species]
        global_vars = x[:, self.num_species:-1]
        time_raw = x[:, -1]
        
        # Process time embedding
        time_emb = self.time_embed(time_raw)
        if not self.use_time_embedding:
            time_emb = time_raw.unsqueeze(-1)
        
        # Combine all inputs
        combined_input = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        
        # Generate Fourier features
        fourier_feats = self.fourier_embed(combined_input)
        x_enhanced = torch.cat([combined_input, fourier_feats], dim=-1)
        
        # Lift to hidden dimension
        hidden = self.lifting(x_enhanced)
        
        # Apply processing blocks
        for block in self.processing_blocks:
            hidden = block(hidden)
        
        # Project to output
        output_delta = self.projection(hidden)
        output_delta = self.output_act(output_delta)
        
        # Add residual connection from initial species
        output = initial_species + output_delta
        
        return torch.clamp_min(output, 0.0)


# =============================================================================
# Factory Functions
# =============================================================================

def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Factory function to create and initialize prediction models.
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Optional device to place model on
        
    Returns:
        Initialized model (SIREN or ResNet based on config)
    """
    model_params = config["model_hyperparameters"]
    data_spec = config["data_specification"]
    num_constants = config.get("numerical_constants", {})
    
    # Get model type from config
    model_type = model_params.get("model_type", "siren").lower()
    
    # Get common parameters
    num_species = len(data_spec["species_variables"])
    num_global_vars = len(data_spec["global_variables"])
    hidden_dims = model_params["hidden_dims"]
    
    # Get constants from config with fallbacks
    default_time_embedding_dim = num_constants.get("default_time_embedding_dim", 64)
    default_condition_dim = num_constants.get("default_condition_dim", 64)
    default_omega_0 = num_constants.get("default_omega_0", 15.0)
    siren_init_scale = num_constants.get("siren_init_scale", 1.0)
    
    # Common model arguments
    common_args = {
        "num_species": num_species,
        "num_global_vars": num_global_vars,
        "hidden_dims": hidden_dims,
        "use_time_embedding": model_params.get("use_time_embedding", True),
        "time_embedding_dim": model_params.get("time_embedding_dim", default_time_embedding_dim),
    }
    
    if model_type == "siren":
        logger.info("Creating SIREN model")
        
        model_args = {
            **common_args,
            "condition_dim": model_params.get("condition_dim", default_condition_dim),
            "w0_initial": model_params.get("siren_w0_initial", default_omega_0),
            "w0_hidden": model_params.get("siren_w0_hidden", default_omega_0),
            "siren_init_scale": siren_init_scale,
        }
        
        model = FiLM_SIREN(**model_args)
        
    elif model_type == "resnet":
        logger.info("Creating ChemicalResNet model")
        
        model_args = {
            **common_args,
            "fourier_features": model_params.get("fourier_features", 256),
            "fourier_scale": model_params.get("fourier_scale", 1.0),
            "mlp_ratio": model_params.get("mlp_ratio", 2.0),
            "dropout": model_params.get("dropout", 0.1),
            "output_activation": model_params.get("output_activation", "softplus"),
        }
        
        model = ChemicalResNet(**model_args)
        
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. Supported types: 'siren', 'resnet'"
        )
    
    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"{model.__class__.__name__} created with {trainable_params:,} trainable parameters."
    )
    logger.debug(f"Model config: {model_args}")
    
    # Store model type for directory naming
    model._model_type = model_type
    
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
        # Use tracing for both SIREN and ResNet models
        try:
            scripted_model = torch.jit.trace(model, example_input, strict=False)
            logger.info("Model successfully traced.")
        except Exception as e:
            logger.error(f"JIT trace failed: {e}. Model export skipped.")
            return
        
        # Apply optimizations if requested
        if optimize:
            try:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            except Exception as e:
                logger.warning(f"JIT optimization failed: {e}. Saving unoptimized model.")
        
        # Verify output matches original model
        test_output = scripted_model(example_input)
        original_output = model(example_input)
        
        # Use relaxed tolerance for JIT comparison
        if not torch.allclose(test_output, original_output, rtol=1e-4, atol=1e-5):
            logger.warning("JIT output differs from original model!")
    
    # Save the exported model
    try:
        scripted_model.save(str(save_path))
        logger.info(f"JIT model saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save JIT model: {e}")


__all__ = ["FiLM_SIREN", "ChemicalResNet", "create_prediction_model", "export_model_jit"]