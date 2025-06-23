#!/usr/bin/env python3
"""
model.py - Neural network architectures for chemical state prediction.

This module provides MLP, SIREN, and FNO architectures that predict chemical
system states at future times. All models use Feature-wise Linear Modulation
(FiLM) for conditioning on global parameters and time.

Key features:
- JIT compilation support
- Multiple initialization schemes
- Flexible activation functions
- Efficient implementations
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
DEFAULT_DROPOUT = 0.1
DEFAULT_OMEGA_0 = 30.0
MATMUL_PRECISION = "high"

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision(MATMUL_PRECISION)


# --- Activation Functions ---
def get_activation(name: str) -> nn.Module:
    """Return activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),  # Swish
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "tanh": nn.Tanh(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]


# --- Weight Initialization ---
def init_weights(module: nn.Module, init_type: str = "xavier_uniform") -> None:
    """Initialize network weights using specified method."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# --- Core Helper Modules ---

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learnable scaling."""
    def __init__(self, dim: int, learnable_scale: bool = False) -> None:
        super().__init__()
        self.dim = dim
        
        # Pre-compute frequency components
        half_dim = self.dim // 2
        denominator = max(half_dim - 1.0, 1.0)
        freqs = torch.arange(0, half_dim).float() / denominator
        self.register_buffer('inv_freq', 1.0 / (10000 ** freqs))
        
        # Optional learnable scaling
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        if self.dim < 2:
            return t.expand(-1, self.dim)
        
        # Apply scaling and compute embeddings
        t_scaled = t.float() * self.scale
        emb = t_scaled * self.inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0.0)
        
        return emb.to(t.dtype)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation with optional normalization."""
    def __init__(self, condition_dim: int, feature_dim: int, 
                 use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        
        # Generate gamma and beta parameters
        self.generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        
        if use_norm:
            self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        if self.use_norm:
            features = self.norm(features)
        
        params = self.generator(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return (1 + gamma) * features + beta  # (1 + gamma) for residual learning


class ResidualBlock(nn.Module):
    """Residual block with pre-activation and optional stochastic depth."""
    def __init__(self, dim: int, dropout: float = DEFAULT_DROPOUT,
                 activation: str = "silu", stochastic_depth: float = 0.0) -> None:
        super().__init__()
        self.stochastic_depth = stochastic_depth
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Stochastic depth (drop path)
        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return x
        
        return x + self.net(x)


# --- SIREN-Specific Modules ---
class SineLayer(nn.Module):
    """SIREN sine activation layer with careful initialization."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = DEFAULT_OMEGA_0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform initialization
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: account for sine activation
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


# --- FNO-Specific Modules (Corrected) ---
class SpectralConv1d(nn.Module):
    """1D Spectral convolution for function approximation."""
    def __init__(self, in_channels: int, out_channels: int, 
                 modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Complex weights for Fourier modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )
    
    def compl_mul1d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication for 1D signals."""
        # input: (batch, in_channel, x), weights: (in_channel, out_channel, x)
        # output: (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, channels, width)
        batchsize = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)
        
        # Apply spectral convolution on low frequencies
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], weights_complex
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock(nn.Module):
    """Fourier Neural Operator block with spectral convolution and skip connection."""
    def __init__(self, width: int, modes: int, activation: str = "gelu",
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, width, grid)
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        
        # Reshape for layer norm
        x = x.permute(0, 2, 1)  # (batch, grid, width)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (batch, width, grid)
        
        x = self.activation(x)
        x = self.dropout(x)
        return x


# --- Main Model Architectures ---

class FiLM_MLP(nn.Module):
    """MLP predictor with FiLM conditioning and modern improvements."""
    
    def __init__(
        self, num_species: int, num_global_vars: int, hidden_dims: List[int],
        dropout: float, use_time_embedding: bool, time_embedding_dim: int,
        condition_dim: int, use_residual: bool, activation: str = "silu",
        use_layer_norm: bool = True, init_type: str = "xavier_uniform",
        stochastic_depth: float = 0.0, **kwargs
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        
        # Time embedding
        self.time_embed = TimeEmbedding(
            time_embedding_dim, learnable_scale=True
        ) if use_time_embedding else None
        
        # Conditioning network with skip connection
        condition_input_dim = num_global_vars + (
            time_embedding_dim if use_time_embedding else 1
        )
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            get_activation(activation),
            nn.Dropout(dropout * 0.5),  # Less dropout in conditioning
            nn.Linear(condition_dim, condition_dim),
            get_activation(activation),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Input projection with layer norm
        self.input_norm = nn.LayerNorm(num_species) if use_layer_norm else nn.Identity()
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0], use_norm=True)
        
        # Main body with residual blocks
        layers = nn.ModuleList()
        in_dim = hidden_dims[0]
        
        for i, out_dim in enumerate(hidden_dims):
            if use_residual and in_dim == out_dim:
                # Stochastic depth increases with depth
                sd_prob = stochastic_depth * (i / len(hidden_dims))
                layers.append(ResidualBlock(
                    in_dim, dropout, activation, sd_prob
                ))
            else:
                # Dimension change block
                layers.append(nn.Sequential(
                    nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity(),
                    nn.Linear(in_dim, out_dim),
                    get_activation(activation),
                    nn.Dropout(dropout)
                ))
            in_dim = out_dim
        
        self.mlp_body = layers
        
        # Output projection with residual
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            get_activation(activation),
            nn.Linear(hidden_dims[-1] // 2, num_species)
        )
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, init_type))

    def forward(self, x: Tensor) -> Tensor:
        # Split input
        initial_species = x[:, :self.num_species]
        global_and_time = x[:, self.num_species:]
        
        # Process conditioning
        if self.use_time_embedding and self.time_embed is not None:
            time_emb = self.time_embed(global_and_time[:, -1])
            condition_input = torch.cat([global_and_time[:, :-1], time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        # Process species through network
        hidden = self.input_norm(initial_species)
        hidden = self.input_proj(hidden)
        hidden = self.film_layer(hidden, condition_vector)
        
        # Main processing
        for layer in self.mlp_body:
            hidden = layer(hidden)
        
        # Output
        hidden = self.output_norm(hidden)
        delta = self.output_proj(hidden)
        
        # Residual connection
        return initial_species + delta


class FiLM_SIREN(nn.Module):
    """SIREN model with FiLM conditioning."""
    
    def __init__(
        self, num_species: int, num_global_vars: int, hidden_dims: List[int],
        use_time_embedding: bool, time_embedding_dim: int, condition_dim: int,
        w0_initial: float, w0_hidden: float, use_batch_norm: bool = False,
        final_activation: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        
        # Time embedding
        self.time_embed = TimeEmbedding(
            time_embedding_dim, learnable_scale=True
        ) if use_time_embedding else None
        
        # Conditioning network
        condition_input_dim = num_global_vars + (
            time_embedding_dim if use_time_embedding else 1
        )
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        
        # SIREN layers
        siren_layers = nn.ModuleList()
        in_dim = hidden_dims[0]
        
        for i, out_dim in enumerate(hidden_dims):
            is_first = (i == 0)
            omega = w0_initial if is_first else w0_hidden
            
            layer = [SineLayer(in_dim, out_dim, is_first=is_first, omega_0=omega)]
            
            # Optional batch norm
            if use_batch_norm and not is_first:
                layer.append(nn.BatchNorm1d(out_dim))
            
            siren_layers.append(nn.Sequential(*layer))
            in_dim = out_dim
        
        self.net = siren_layers
        
        # Output
        if final_activation:
            self.output_linear = nn.Sequential(
                nn.Linear(in_dim, num_species),
                nn.Tanh()  # Bound output changes
            )
        else:
            self.output_linear = nn.Linear(in_dim, num_species)

    def forward(self, x: Tensor) -> Tensor:
        # Split input
        initial_species = x[:, :self.num_species]
        global_and_time = x[:, self.num_species:]
        
        # Process conditioning
        if self.use_time_embedding and self.time_embed is not None:
            time_emb = self.time_embed(global_and_time[:, -1])
            condition_input = torch.cat([global_and_time[:, :-1], time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        # Process through SIREN
        coords = self.input_proj(initial_species)
        coords = self.film_layer(coords, condition_vector)
        
        for layer in self.net:
            coords = layer(coords)
        
        delta = self.output_linear(coords)
        return initial_species + delta


class FiLM_FNO(nn.Module):
    """Fourier Neural Operator with FiLM conditioning for chemical kinetics."""
    
    def __init__(
        self, num_species: int, num_global_vars: int, hidden_dims: List[int],
        dropout: float, use_time_embedding: bool, time_embedding_dim: int,
        condition_dim: int, fno_spectral_modes: int, fno_seq_length: int,
        compression: float = 0.5, activation: str = "gelu", **kwargs
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.use_time_embedding = use_time_embedding
        self.seq_length = fno_seq_length
        
        # Time embedding
        self.time_embed = TimeEmbedding(
            time_embedding_dim, learnable_scale=True
        ) if use_time_embedding else None
        
        # Conditioning network
        condition_input_dim = num_global_vars + (
            time_embedding_dim if use_time_embedding else 1
        )
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            get_activation(activation),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Lifting layer - expand to function space
        self.lifting = nn.Sequential(
            nn.Linear(num_species + condition_dim, hidden_dims[0]),
            get_activation(activation)
        )
        
        # Position encoding for creating function representation
        self.register_buffer(
            'pos_encoding',
            self._create_position_encoding(self.seq_length, hidden_dims[0])
        )
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList()
        in_channels = hidden_dims[0]
        
        for out_channels in hidden_dims:
            if in_channels != out_channels:
                # Channel adjustment layer
                self.fno_blocks.append(nn.Conv1d(in_channels, out_channels, 1))
                in_channels = out_channels
            
            # FNO block
            self.fno_blocks.append(
                FNOBlock(out_channels, fno_spectral_modes, activation, dropout)
            )
        
        # Projection back to species space
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1] // 2, 1),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dims[-1] // 2, num_species, 1)
        )

    def _create_position_encoding(self, length: int, d_model: int) -> Tensor:
        """Create sinusoidal position encoding."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # Split input
        initial_species = x[:, :self.num_species]
        global_and_time = x[:, self.num_species:]
        batch_size = x.shape[0]
        
        # Process conditioning
        if self.use_time_embedding and self.time_embed is not None:
            time_emb = self.time_embed(global_and_time[:, -1])
            condition_input = torch.cat([global_and_time[:, :-1], time_emb], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        # Combine species and conditioning
        combined = torch.cat([initial_species, condition_vector], dim=-1)
        
        # Lift to function space
        lifted = self.lifting(combined)  # (batch, channels)
        
        # Create function representation using position encoding
        # Expand lifted features across sequence dimension
        func_repr = lifted.unsqueeze(-1).expand(-1, -1, self.seq_length)
        
        # Add position encoding
        pos_enc = self.pos_encoding.permute(0, 2, 1).expand(batch_size, -1, -1)
        func_repr = func_repr + pos_enc
        
        # Process through FNO blocks
        for block in self.fno_blocks:
            func_repr = block(func_repr)
        
        # Project back and pool
        output = self.projection(func_repr)
        
        # Global average pooling to get single prediction
        delta = output.mean(dim=-1)
        
        return initial_species + delta


# --- Factory Function ---

def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """Factory function to create and initialize prediction models."""
    model_params = config["model_hyperparameters"]
    data_spec = config["data_specification"]
    model_type = model_params.get("model_type", "mlp").lower()
    
    logger.info(f"Creating model of type: '{model_type}'")
    
    # Base arguments for all models
    base_args = {
        "num_species": len(data_spec["species_variables"]),
        "num_global_vars": len(data_spec["global_variables"]),
        "hidden_dims": model_params["hidden_dims"],
        "use_time_embedding": model_params.get("use_time_embedding", True),
        "time_embedding_dim": model_params.get("time_embedding_dim", DEFAULT_TIME_EMBEDDING_DIM),
        "condition_dim": model_params.get("condition_dim", DEFAULT_CONDITION_DIM),
    }
    
    # Model-specific arguments
    model_class: type[nn.Module]
    
    if model_type == "mlp":
        model_class = FiLM_MLP
        base_args.update({
            "dropout": model_params.get("dropout", DEFAULT_DROPOUT),
            "use_residual": model_params.get("use_residual", True),
            "activation": model_params.get("activation", "silu"),
            "use_layer_norm": model_params.get("use_layer_norm", True),
            "init_type": model_params.get("init_type", "xavier_uniform"),
            "stochastic_depth": model_params.get("stochastic_depth", 0.0),
        })
        
    elif model_type == "siren":
        model_class = FiLM_SIREN
        base_args.update({
            "w0_initial": model_params.get("siren_w0_initial", DEFAULT_OMEGA_0),
            "w0_hidden": model_params.get("siren_w0_hidden", DEFAULT_OMEGA_0),
            "use_batch_norm": model_params.get("use_batch_norm", False),
            "final_activation": model_params.get("final_activation", False),
        })
        
    elif model_type == "fno":
        model_class = FiLM_FNO
        fno_seq_length = model_params.get("fno_seq_length", 32)
        fno_spectral_modes = model_params.get("fno_spectral_modes", 16)
        
        # Validate modes
        max_modes = fno_seq_length // 2
        if fno_spectral_modes > max_modes:
            logger.warning(
                f"FNO modes ({fno_spectral_modes}) > Nyquist limit ({max_modes}). "
                f"Clamping to {max_modes}."
            )
            fno_spectral_modes = max_modes
        
        base_args.update({
            "dropout": model_params.get("dropout", DEFAULT_DROPOUT),
            "fno_spectral_modes": fno_spectral_modes,
            "fno_seq_length": fno_seq_length,
            "compression": model_params.get("fno_compression", 0.5),
            "activation": model_params.get("activation", "gelu"),
        })
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose from: ['mlp', 'siren', 'fno']."
        )
    
    # Create model
    model = model_class(**base_args)
    
    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"{model_class.__name__} created with {trainable_params:,} trainable parameters."
    )
    logger.debug(f"Model config: {base_args}")
    
    # Move to device
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
    """Export model as TorchScript JIT module."""
    save_path = Path(save_path)
    
    # Set eval mode
    model.eval()
    
    with torch.no_grad():
        # Script the model
        try:
            scripted_model = torch.jit.script(model)
            logger.info("Model successfully scripted.")
        except Exception as e:
            logger.warning(f"JIT script failed: {e}. Trying trace...")
            scripted_model = torch.jit.trace(model, example_input)
        
        # Optimize if requested
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        # Verify output matches
        test_output = scripted_model(example_input)
        original_output = model(example_input)
        
        if not torch.allclose(test_output, original_output, rtol=1e-5):
            logger.warning("JIT output differs from original model!")
    
    # Save
    scripted_model.save(str(save_path))
    logger.info(f"JIT model saved to: {save_path}")


__all__ = [
    "FiLM_MLP", "FiLM_SIREN", "FiLM_FNO",
    "create_prediction_model", "export_model_jit"
]