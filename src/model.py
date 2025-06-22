#!/usr/bin/env python3
"""
model.py - Defines neural network architectures for chemical state prediction.

This module provides several model architectures (MLP, SIREN, FNO) that predict
the state of a chemical system at a future time, given an initial state and
global conditions. All models use a Feature-wise Linear Modulation (FiLM) layer
to inject conditioning information. A factory function, `create_prediction_model`,
is provided to instantiate the desired model based on a configuration dictionary.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
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


# --- Core Helper Modules (Used by multiple architectures) ---

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for a single scalar time value. JIT-compatible."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1: t = t.unsqueeze(-1)
        device, dtype = t.device, t.dtype
        
        # Corrected: Handle edge case where embedding dimension is too small.
        if self.dim < 2:
            return t.expand(-1, self.dim).to(dtype)
            
        half_dim = self.dim // 2
        denominator = (half_dim - 1.0) if half_dim > 1 else 1.0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / denominator))
        emb = t.float() * inv_freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1), "constant", 0.0)
        return emb.to(dtype)

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
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
    def forward(self, x: Tensor) -> Tensor: return x + self.net(x)

# --- SIREN-Specific Helper Modules ---
class SineLayer(nn.Module):
    """Linear layer followed by a sine activation, for use in SIREN models."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = DEFAULT_OMEGA_0) -> None:
        super().__init__()
        self.omega_0, self.is_first = omega_0, is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x: Tensor) -> Tensor: return torch.sin(self.omega_0 * self.linear(x))

# --- FNO-Specific Helper Modules (Restored) ---

class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer for Fourier Neural Operator."""
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        weights = torch.complex(self.weights_real, self.weights_imag)
        if x_ft.size(-1) >= self.modes:
            out_ft[..., :self.modes] = torch.einsum("bix,iox->box", x_ft[..., :self.modes], weights)
        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x_out

class FNOBlock(nn.Module):
    """An FNO-inspired block with spectral convolution and linear skip paths."""
    def __init__(self, channels: int, modes: int, seq_length: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.conv = SpectralConv1d(channels, channels, modes)
        self.w = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        pos = torch.linspace(0, 1, seq_length).unsqueeze(0).unsqueeze(0)
        self.register_buffer('pos_encoding', pos)

    def forward(self, x: Tensor) -> Tensor:
        # FNO expects a sequence, but our input is a single vector. We create a dummy sequence.
        x_seq = x.unsqueeze(-1).expand(-1, -1, self.pos_encoding.shape[-1])
        x_seq = x_seq + self.pos_encoding
        out_spectral = self.conv(x_seq).mean(dim=-1)
        out_linear = self.w(x)
        out = F.gelu(out_spectral + out_linear)
        out = self.norm(x + out)
        out = self.dropout(out)
        return out

# --- Main Model Architectures ---

class FiLM_MLP(nn.Module):
    """An MLP-based predictor that uses a FiLM layer for conditioning."""
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int], dropout: float,
                 use_time_embedding: bool, time_embedding_dim: int, condition_dim: int, use_residual: bool) -> None:
        super().__init__()
        self.num_species = num_species
        self.time_embed = TimeEmbedding(time_embedding_dim) if use_time_embedding else None
        condition_input_dim = num_global_vars + (time_embedding_dim if use_time_embedding else 1)
        self.conditioning_net = nn.Sequential(nn.Linear(condition_input_dim, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        body_layers: List[nn.Module] = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            body_layers.append(nn.LayerNorm(in_dim))
            if use_residual and in_dim == out_dim:
                body_layers.append(ResidualBlock(in_dim, dropout))
            else:
                body_layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Dropout(dropout)])
            in_dim = out_dim
        self.mlp_body = nn.Sequential(*body_layers)
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_species)

    def forward(self, x: Tensor) -> Tensor:
        initial_species, global_and_time = x[:, :self.num_species], x[:, self.num_species:]
        if self.time_embed:
            condition_input = torch.cat([global_and_time[:, :-1], self.time_embed(global_and_time[:, -1])], dim=-1)
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
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int],
                 use_time_embedding: bool, time_embedding_dim: int, condition_dim: int,
                 w0_initial: float, w0_hidden: float) -> None:
        super().__init__()
        self.num_species = num_species
        self.time_embed = TimeEmbedding(time_embedding_dim) if use_time_embedding else None
        condition_input_dim = num_global_vars + (time_embedding_dim if use_time_embedding else 1)
        self.conditioning_net = nn.Sequential(nn.Linear(condition_input_dim, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input_proj = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        siren_layers: List[nn.Module] = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            siren_layers.append(SineLayer(in_dim, out_dim, is_first=(in_dim == hidden_dims[0]), omega_0=(w0_initial if in_dim == hidden_dims[0] else w0_hidden)))
            in_dim = out_dim
        self.net = nn.Sequential(*siren_layers)
        self.output_linear = nn.Linear(in_dim, num_species)

    def forward(self, x: Tensor) -> Tensor:
        initial_species, global_and_time = x[:, :self.num_species], x[:, self.num_species:]
        if self.time_embed:
            condition_input = torch.cat([global_and_time[:, :-1], self.time_embed(global_and_time[:, -1])], dim=-1)
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
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int], dropout: float,
                 use_time_embedding: bool, time_embedding_dim: int, condition_dim: int, 
                 fno_spectral_modes: int, fno_seq_length: int) -> None:
        super().__init__()
        self.num_species = num_species
        self.time_embed = TimeEmbedding(time_embedding_dim) if use_time_embedding else None
        condition_input_dim = num_global_vars + (time_embedding_dim if use_time_embedding else 1)
        self.conditioning_net = nn.Sequential(nn.Linear(condition_input_dim, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.lifting = nn.Linear(num_species, hidden_dims[0])
        self.film_layer = FiLMLayer(condition_dim, hidden_dims[0])
        self.fno_blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            if in_dim != out_dim:
                self.fno_blocks.append(nn.Linear(in_dim, out_dim))
            valid_modes = min(fno_spectral_modes, fno_seq_length // 2)
            self.fno_blocks.append(FNOBlock(out_dim, valid_modes, fno_seq_length, dropout))
            in_dim = out_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * 2, num_species)
        )

    def forward(self, x: Tensor) -> Tensor:
        initial_species, global_and_time = x[:, :self.num_species], x[:, self.num_species:]
        if self.time_embed:
            condition_input = torch.cat([global_and_time[:, :-1], self.time_embed(global_and_time[:, -1])], dim=-1)
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
    """Factory function to build and initialize the appropriate prediction model."""
    model_params = config["model_hyperparameters"]
    data_spec = config["data_specification"]
    model_type = model_params.get("model_type", "mlp").lower()
    logger.info(f"Creating model of type: '{model_type}'")
    
    base_args = {
        "num_species": len(data_spec["species_variables"]),
        "num_global_vars": len(data_spec["global_variables"]),
        "hidden_dims": model_params["hidden_dims"],
        "use_time_embedding": model_params.get("use_time_embedding", True),
        "time_embedding_dim": model_params.get("time_embedding_dim", DEFAULT_TIME_EMBEDDING_DIM),
        "condition_dim": model_params.get("condition_dim", DEFAULT_CONDITION_DIM),
    }

    model_class: type[nn.Module]
    if model_type == "mlp":
        model_class = FiLM_MLP
        base_args.update({
            "dropout": model_params.get("dropout", DEFAULT_DROPOUT),
            "use_residual": model_params.get("use_residual", True),
        })
    elif model_type == "siren":
        model_class = FiLM_SIREN
        base_args.update({
            "w0_initial": model_params.get("siren_w0_initial", DEFAULT_OMEGA_0),
            "w0_hidden": model_params.get("siren_w0_hidden", DEFAULT_OMEGA_0),
        })
    elif model_type == "fno":
        model_class = FiLM_FNO
        fno_seq_length = model_params.get("fno_seq_length", 32)
        fno_spectral_modes = model_params.get("fno_spectral_modes", 16)
        if fno_spectral_modes > fno_seq_length // 2:
            logger.warning(f"FNO modes ({fno_spectral_modes}) > seq_len/2 ({fno_seq_length//2}). Clamping to Nyquist limit.")
            fno_spectral_modes = fno_seq_length // 2
        base_args.update({
            "dropout": model_params.get("dropout", DEFAULT_DROPOUT),
            "fno_spectral_modes": fno_spectral_modes,
            "fno_seq_length": fno_seq_length,
        })
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from ['mlp', 'siren', 'fno'].")
        
    model = model_class(**base_args)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model_class.__name__} created with {trainable_params:,} trainable parameters.")
    logger.debug(f"Model config: {base_args}")
    
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model

__all__ = ["FiLM_MLP", "FiLM_SIREN", "FiLM_FNO", "create_prediction_model"]