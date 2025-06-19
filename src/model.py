#!/usr/bin/env python3
"""
model.py - Predict the state at a certain time, given the initial state
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants
DEFAULT_OMEGA_0 = 30.0
DEFAULT_TIME_EMBEDDING_DIM = 32
DEFAULT_CONDITION_DIM = 128
DEFAULT_DROPOUT = 0.1
MATMUL_PRECISION = "high"

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision(MATMUL_PRECISION)


# --- Core Helper Modules (Used by multiple architectures) ---

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding. JIT-compatible."""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        device = t.device
        half_dim = self.dim // 2
        
        if half_dim == 0:
            return torch.zeros(t.shape[0], self.dim, device=device)

        denominator = (half_dim - 1.0) if half_dim > 1 else 1.0
        
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, device=device).float() / denominator)
        )
        
        emb = t * inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0.0)
        
        return emb


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) Layer. JIT-compatible."""
    
    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.generator = nn.Linear(condition_dim, feature_dim * 2)
    
    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = torch.chunk(self.generator(condition), 2, dim=-1)
        return gamma * features + beta


class ResidualBlock(nn.Module):
    """Standard MLP residual block. JIT-compatible."""
    
    def __init__(self, dim: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# --- FNO-Specific Helper Modules ---

class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer for FNO."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        
        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        
        weights = torch.complex(self.weights_real, self.weights_imag)
        
        if x_ft.size(-1) >= self.modes:
            out_ft[..., :self.modes] = torch.einsum(
                "bix,iox->box", x_ft[..., :self.modes], weights
            )
        
        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x_out


class FNOBlock(nn.Module):
    """An FNO-inspired block with spectral convolution and linear skip paths."""
    
    def __init__(
        self, 
        channels: int, 
        modes: int, 
        seq_length: int, 
        dropout: float = DEFAULT_DROPOUT
    ) -> None:
        super().__init__()
        self.conv = SpectralConv1d(channels, channels, modes)
        self.w = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        # Create a persistent positional encoding buffer
        pos = torch.linspace(0, 1, seq_length).unsqueeze(0).unsqueeze(0)
        self.register_buffer('pos_encoding', pos)

    def forward(self, x: Tensor) -> Tensor:
        x_seq = x.unsqueeze(-1).expand(-1, -1, self.pos_encoding.shape[-1])
        x_seq = x_seq + self.pos_encoding
        
        # Path 1: Spectral Convolution
        out_spectral = self.conv(x_seq).mean(dim=-1)

        # Path 2: Linear skip connection
        out_linear = self.w(x)

        # Combine, activate, normalize
        out = F.gelu(out_spectral + out_linear)
        out = self.norm(x + out)  # Residual connection from input
        out = self.dropout(out)
        return out


# --- SIREN-Specific Helper Modules ---

class SineLayer(nn.Module):
    """Linear layer followed by a sine activation for SIREN models."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        is_first: bool = False, 
        omega_0: float = DEFAULT_OMEGA_0
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


# --- Main Model Architectures ---

class FiLM_MLP(nn.Module):
    """An MLP-based predictor that uses a FiLM layer for conditioning."""
    
    def __init__(
        self, 
        num_species: int, 
        num_global_vars: int, 
        hidden_dims: List[int], 
        dropout: float,
        use_time_embedding: bool, 
        time_embedding_dim: int, 
        condition_dim: int, 
        use_residual: bool
    ) -> None:
        super().__init__()
        self.num_species = num_species
        
        self.time_embed: Optional[TimeEmbedding]
        
        condition_input_dim = num_global_vars
        if use_time_embedding:
            self.time_embed = TimeEmbedding(time_embedding_dim)
            condition_input_dim += time_embedding_dim
        else:
            self.time_embed = None
            condition_input_dim += 1

        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        
        body_layers: List[nn.Module] = []
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]
            
            body_layers.append(nn.LayerNorm(in_dim))

            if use_residual and in_dim == out_dim:
                body_layers.append(ResidualBlock(in_dim, dropout))
            else:
                body_layers.extend([
                    nn.Linear(in_dim, out_dim), 
                    nn.SiLU(), 
                    nn.Dropout(dropout)
                ])

        self.mlp_body = nn.Sequential(*body_layers)
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        if self.time_embed is not None:
            time = global_and_time[..., -1:]
            global_vars = global_and_time[..., :-1]
            condition_input = torch.cat([
                global_vars, 
                self.time_embed(time.squeeze(-1))
            ], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        hidden = self.input_proj(initial_species)
        hidden = self.film_layer(hidden, condition_vector)
        hidden = self.mlp_body(hidden)
        hidden = self.output_norm(hidden)
        delta = self.output_proj(hidden)
        
        return initial_species + delta


class FiLM_SIREN(nn.Module):
    """A SIREN model conditioned via a FiLM layer."""
    
    def __init__(
        self, 
        num_species: int,
        num_global_vars: int,
        hidden_dims: List[int],
        use_time_embedding: bool,
        time_embedding_dim: int,
        condition_dim: int, 
        w0_initial: float, 
        w0_hidden: float
    ) -> None:
        super().__init__()
        self.num_species = num_species
        
        self.time_embed: Optional[TimeEmbedding]

        condition_input_dim = num_global_vars
        if use_time_embedding:
            self.time_embed = TimeEmbedding(time_embedding_dim)
            condition_input_dim += time_embedding_dim
        else:
            self.time_embed = None
            condition_input_dim += 1
        
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim), 
            nn.SiLU(), 
            nn.Linear(condition_dim, condition_dim)
        )
        
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        
        siren_layers: List[nn.Module] = []
        current_dim = hidden_dims[0]
        
        for i, dim in enumerate(hidden_dims):
            siren_layers.append(SineLayer(
                current_dim, 
                dim, 
                is_first=(i == 0), 
                omega_0=(w0_initial if i == 0 else w0_hidden)
            ))
            current_dim = dim
        
        self.net = nn.Sequential(*siren_layers)
        self.output_linear = nn.Linear(current_dim, num_species)

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        if self.time_embed is not None:
            time = global_and_time[..., -1:]
            global_vars = global_and_time[..., :-1]
            condition_input = torch.cat([
                global_vars, 
                self.time_embed(time.squeeze(-1))
            ], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)
        
        coords = self.input_proj(initial_species)
        coords = self.film_layer(coords, condition_vector)
        coords = self.net(coords)
        delta = self.output_linear(coords)
        
        return initial_species + delta


class FiLM_FNO(nn.Module):
    """An FNO-inspired model conditioned via a FiLM layer."""
    
    def __init__(
        self, 
        num_species: int, 
        num_global_vars: int, 
        hidden_dims: List[int], 
        dropout: float,
        use_time_embedding: bool, 
        time_embedding_dim: int, 
        condition_dim: int, 
        fno_spectral_modes: int,
        fno_seq_length: int
    ) -> None:
        super().__init__()
        self.num_species = num_species
        
        self.time_embed: Optional[TimeEmbedding]

        condition_input_dim = num_global_vars
        if use_time_embedding:
            self.time_embed = TimeEmbedding(time_embedding_dim)
            condition_input_dim += time_embedding_dim
        else:
            self.time_embed = None
            condition_input_dim += 1
        
        self.conditioning_net = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim), 
            nn.SiLU(), 
            nn.Linear(condition_dim, condition_dim)
        )

        self.lifting = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])

        self.fno_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            prev_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            
            if dim != prev_dim:
                self.fno_blocks.append(nn.Linear(prev_dim, dim))
            
            valid_modes = min(fno_spectral_modes, fno_seq_length // 2)
            self.fno_blocks.append(FNOBlock(dim, valid_modes, fno_seq_length, dropout))

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * 2, num_species)
        )

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        global_and_time = x[..., self.num_species:]
        
        if self.time_embed is not None:
            time = global_and_time[..., -1:]
            global_vars = global_and_time[..., :-1]
            condition_input = torch.cat([
                global_vars, 
                self.time_embed(time.squeeze(-1))
            ], dim=-1)
        else:
            condition_input = global_and_time
        
        condition_vector = self.conditioning_net(condition_input)

        hidden = self.lifting(initial_species)
        hidden = self.film_layer(hidden, condition_vector)
        
        for block in self.fno_blocks:
            hidden = block(hidden)

        delta = self.projection(hidden)
        
        return initial_species + delta


# --- Factory Function ---

def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """Factory function to create the appropriate model based on config."""
    model_type = config.get("model_type", "mlp").lower()
    use_film = config.get("use_film", True)

    logger.info(f"Creating model of type: '{model_type}' with FiLM: {use_film}")

    if not use_film:
        raise NotImplementedError(
            "Modern architectures in this file require use_film=True."
        )

    model_class: type[nn.Module]
    model_config: Dict[str, Any] = {}

    if model_type == "siren":
        model_class = FiLM_SIREN
        model_config.update({
            "w0_initial": config.get("siren_w0_initial", DEFAULT_OMEGA_0),
            "w0_hidden": config.get("siren_w0_hidden", 1.0),
        })
    elif model_type == "mlp":
        model_class = FiLM_MLP
        model_config.update({
            "dropout": config.get("dropout", DEFAULT_DROPOUT),
            "use_residual": config.get("use_residual", True),
        })
    elif model_type == "fno":
        model_class = FiLM_FNO
        fno_seq_length = config.get("fno_seq_length", 32)
        fno_spectral_modes = config.get("fno_spectral_modes", 16)
        
        if fno_spectral_modes > fno_seq_length // 2:
            fno_spectral_modes = fno_seq_length // 2
            logger.warning(
                f"Adjusted fno_spectral_modes to {fno_spectral_modes} "
                f"to satisfy Nyquist limit for sequence length {fno_seq_length}"
            )
        
        model_config.update({
            "dropout": config.get("dropout", DEFAULT_DROPOUT),
            "fno_spectral_modes": fno_spectral_modes,
            "fno_seq_length": fno_seq_length,
        })
    else:
        available_types = ["mlp", "siren", "fno"]
        raise ValueError(
            f"Unknown model_type '{model_type}' in configuration. "
            f"Available types: {available_types}"
        )

    try:
        species_vars = config["species_variables"]
        global_vars = config["global_variables"]
    except KeyError as e:
        raise ValueError(f"Missing required config key: {e}")
    
    model_config.update({
        "num_species": len(species_vars),
        "num_global_vars": len(global_vars),
        "hidden_dims": config.get("hidden_dims", [256, 256, 256]),
        "use_time_embedding": config.get("use_time_embedding", True),
        "time_embedding_dim": config.get("time_embedding_dim", DEFAULT_TIME_EMBEDDING_DIM),
        "condition_dim": config.get("condition_dim", DEFAULT_CONDITION_DIM),
        # Note: "output_activation" is no longer passed to the models
    })
        
    model = model_class(**model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"{model_class.__name__} created with {trainable_params:,} trainable parameters "
        f"({total_params:,} total parameters)."
    )
    logger.info(f"Model config: {model_config}")
    
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model


__all__ = ["FiLM_MLP", "FiLM_SIREN", "FiLM_FNO", "create_prediction_model"]