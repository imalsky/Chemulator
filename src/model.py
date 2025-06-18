#!/usr/bin/env python3
"""
model.py - FNO-inspired state-evolution predictor for chemical kinetics.
This model uses Fourier-based transformations while maintaining compatibility
with the existing input/output format.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


# --- Helper Modules for ChemicalFNO ---

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
        # x shape: [batch, in_features]
        # Project input onto random frequencies
        x_proj = x @ self.freqs  # Shape: [batch, out_features // 2]
        # Return sine and cosine components
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SpectralConv1d(nn.Module):
    """
    1D Spectral convolution layer. It performs a convolution in the Fourier
    domain by multiplying the lower-frequency modes with a set of learnable weights.
    
    This implementation stores weights as separate real and imaginary parts
    to maintain compatibility with PyTorch's gradient operations.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable weights for the first `modes` frequencies
        # Store as separate real and imaginary parts for gradient compatibility
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        self._init_spectral_weights()

    def _init_spectral_weights(self):
        # Custom initialization for stability
        with torch.no_grad():
            self.weights_real.data *= 0.1
            self.weights_imag.data *= 0.1

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch, channels, seq_len]
        batch_size = x.shape[0]

        # Compute 1D Real FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Initialize output tensor in Fourier domain
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)

        # Combine real and imaginary parts into complex weights
        weights = torch.complex(self.weights_real, self.weights_imag)
        
        # Apply learnable weights to the low-frequency modes
        if x_ft.size(-1) >= self.modes:
            out_ft[..., :self.modes] = torch.einsum("bix,iox->box", x_ft[..., :self.modes], weights)

        # Compute Inverse Real FFT to return to the signal domain
        x_out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x_out


class FNOBlock(nn.Module):
    """
    An FNO-inspired block. It expands a feature vector into a temporary
    sequence, applies a spectral convolution, and combines it with a
    linear skip connection.
    """
    def __init__(self, channels: int, modes: int, seq_length: int, dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.conv = SpectralConv1d(channels, channels, modes)
        self.w = nn.Linear(channels, channels)  # Linear path (acts like a 1x1 conv)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
        # Create a persistent positional encoding buffer
        self._create_positional_encoding()

    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding for the sequence."""
        pos = torch.linspace(0, 1, self.seq_length)
        # Use multiple frequencies for richer encoding
        freqs = torch.arange(1, 5, dtype=torch.float32)
        pos_enc = torch.sin(2 * math.pi * pos.unsqueeze(0) * freqs.unsqueeze(1)).mean(0)
        # Shape: [1, 1, seq_length]
        self.register_buffer('pos_encoding', pos_enc.unsqueeze(0).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, channels]
        
        # 1. Expand feature vector into a temporary sequence
        x_seq = x.unsqueeze(-1).expand(-1, -1, self.seq_length)
        # Add positional encoding to give the FFT a non-constant signal
        x_seq = x_seq + 0.1 * self.pos_encoding

        # 2. Path 1: Spectral Convolution
        out_spectral = self.conv(x_seq)
        out_spectral = out_spectral.mean(dim=-1)  # [B, C, S] -> [B, C]

        # 3. Path 2: Linear skip connection
        out_linear = self.w(x)  # [B, C]

        # 4. Combine, normalize, and activate
        out = self.norm(out_spectral + out_linear)
        out = F.gelu(out)
        out = self.dropout(out)

        return out


# --- Main Model ---

class ChemicalFNO(nn.Module):
    """
    An FNO-inspired model for predicting chemical state evolution. This model
    is designed as a drop-in replacement for the original MLP.
    """
    def __init__(
        self,
        num_species: int,
        num_global_vars: int,
        hidden_dims: List[int],
        dropout: float,
        fno_fourier_features: int = 256,
        fno_fourier_scale: float = 1.0,
        fno_spectral_modes: int = 16,
        fno_seq_length: int = 32,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.num_species = num_species

        input_dim = num_species + num_global_vars + 1
        self.fourier_embed = FourierFeatures(input_dim, fno_fourier_features, scale=fno_fourier_scale)

        self.lifting = nn.Sequential(
            nn.Linear(input_dim + fno_fourier_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fno_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # If dimensions change, insert a simple linear projection block
            if i > 0 and hidden_dims[i] != hidden_dims[i-1]:
                self.fno_blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.LayerNorm(hidden_dims[i]),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ))
            # Otherwise, add a standard FNO block
            else:
                dim = hidden_dims[i]
                # Ensure modes are valid for the given sequence length
                valid_modes = min(fno_spectral_modes, fno_seq_length // 2)
                self.fno_blocks.append(FNOBlock(dim, valid_modes, fno_seq_length, dropout))

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * 2, num_species)
        )
        
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
            # The last linear layer in projection
            if isinstance(self.projection[-1], nn.Linear):
                self.projection[-1].weight.mul_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        initial_species = x[..., :self.num_species]
        fourier_feats = self.fourier_embed(x)
        x_enhanced = torch.cat([x, fourier_feats], dim=-1)

        hidden = self.lifting(x_enhanced)

        # Apply FNO blocks with proper residual connections
        for block in self.fno_blocks:
            if isinstance(block, FNOBlock):
                hidden = hidden + block(hidden)  # Apply FNOBlock with residual
            else:
                hidden = block(hidden)  # Apply dimension-change block directly

        output_delta = self.projection(hidden)
        output_delta = self.output_act(output_delta)
        
        # Final skip connection: model predicts change from initial state
        output = initial_species + output_delta
            
        return output


def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None
) -> ChemicalFNO:
    """Factory function to create the FNO-inspired model."""
    logger.info("Creating FNO-inspired model (ChemicalFNO).")

    fno_seq_length = config.get("fno_seq_length", 32)
    fno_spectral_modes = config.get("fno_spectral_modes", 16)
    
    # Ensure spectral modes are valid for the given sequence length
    if fno_spectral_modes > fno_seq_length // 2:
        logger.warning(
            f"fno_spectral_modes ({fno_spectral_modes}) is too high for fno_seq_length ({fno_seq_length}). "
            f"Capping modes to {fno_seq_length // 2} (Nyquist limit)."
        )
        fno_spectral_modes = fno_seq_length // 2

    model_config = {
        "fno_fourier_features": config.get("fno_fourier_features", 256),
        "fno_fourier_scale": config.get("fno_fourier_scale", 1.0),
        "fno_spectral_modes": fno_spectral_modes,
        "fno_seq_length": fno_seq_length,
        "output_activation": config.get("output_activation", "sigmoid"),
    }

    model = ChemicalFNO(
        num_species=len(config["species_variables"]),
        num_global_vars=len(config["global_variables"]),
        hidden_dims=config.get("hidden_dims", [256, 256, 256, 256]),
        dropout=config.get("dropout", 0.1),
        **model_config
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ChemicalFNO created with {trainable_params:,} trainable parameters "
        f"(total: {total_params:,})"
    )
    logger.info(f"Model config: hidden_dims={config.get('hidden_dims')}, {model_config}")

    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")

    return model


# Alias for backward compatibility
StateEvolutionPredictor = ChemicalFNO

__all__ = ["ChemicalFNO", "StateEvolutionPredictor", "create_prediction_model"]