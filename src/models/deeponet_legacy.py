#!/usr/bin/env python3

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim


class TimeEncoding(nn.Module):
    """Sinusoidal time encoding for better temporal representation."""
    def __init__(self, d_model: int, max_time: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        
        # Pre-compute the division terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(self.max_time) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch_size, 1] tensor of time values
        Returns:
            [batch_size, d_model] tensor of time encodings
        """
        # Use empty for efficiency (fix #4)
        pe = torch.empty(time.size(0), self.d_model, device=time.device, dtype=time.dtype)
        
        # Apply sinusoidal encoding
        pe[:, 0::2] = torch.sin(time * self.div_term)
        pe[:, 1::2] = torch.cos(time * self.div_term)
        
        return pe


class ResidualBlock(nn.Module):
    """Residual block with layer normalization for stable training."""
    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0, 
                 next_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.next_dim = next_dim if next_dim is not None else dim
        
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.activation = activation
        self.fc2 = nn.Linear(dim * 4, self.next_dim)  # Fix #3: handle dimension changes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(self.next_dim)
        
        # Skip connection with projection if dimensions differ
        if dim != self.next_dim:
            self.skip_proj = nn.Linear(dim, self.next_dim)
        else:
            self.skip_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.ln2(x + residual)


class SimpleCrossAttention(nn.Module):
    """Fixed cross-attention between state and time representations."""
    def __init__(self, state_dim: int, time_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        # Fix #5: Add assertion
        assert state_dim % num_heads == 0, f"state_dim ({state_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use standard multi-head attention for simplicity (Fix #1)
        self.mha = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project time features to match state dimension
        self.time_proj = nn.Linear(time_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)
    
    def forward(self, state: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            time_features: [batch_size, time_dim]
        """
        # Project time to match state dimension
        time_proj = self.time_proj(time_features)
        
        # Reshape for multi-head attention (batch_first=True)
        # Query: state, Key/Value: time
        state = state.unsqueeze(1)  # [B, 1, D]
        time_proj = time_proj.unsqueeze(1)  # [B, 1, D]
        
        # Apply attention
        attn_out, _ = self.mha(state, time_proj, time_proj)
        attn_out = attn_out.squeeze(1)  # [B, D]
        
        # Residual connection and layer norm
        return self.ln(state.squeeze(1) + attn_out)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self,
                 condition_dim: int, 
                 feature_dim: int, 
                 hidden_dims: List[int], 
                 activation: Union[str, List[str]] = "gelu", 
                 use_beta: bool = True):
        super().__init__()
        
        self.use_beta = use_beta
        self.feature_dim = feature_dim
        out_multiplier = 2 if use_beta else 1
        
        # Handle activation specification
        if isinstance(activation, str):
            activations = [activation] * len(hidden_dims)
        else:
            if len(activation) != len(hidden_dims):
                raise ValueError(f"Number of activations ({len(activation)}) must match hidden_dims ({len(hidden_dims)})")
            activations = activation
        
        # Build FiLM MLP
        layers = []
        prev_dim = condition_dim
        
        # Hidden layers with per-layer activation
        for dim, act in zip(hidden_dims, activations):
            layers.extend([
                nn.Linear(prev_dim, dim),
                self._get_activation(act)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_multiplier * feature_dim))
        
        self.film_net = nn.Sequential(*layers)
        
        # Initialize to identity mapping
        self._initialize_identity()
    
    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True)
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name.lower()]
    
    def _initialize_identity(self):
        """Initialize FiLM to identity mapping."""
        with torch.no_grad():
            final_layer = self.film_net[-1]
            # Small weights
            final_layer.weight.data.normal_(0, 0.02)
            # Set bias for gamma=1, beta=0
            if self.use_beta:
                final_layer.bias.data[:self.feature_dim] = 1.0
                final_layer.bias.data[self.feature_dim:] = 0.0
            else:
                final_layer.bias.data.fill_(1.0)
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation"""
        # Generate parameters
        params = self.film_net(condition)
        
        if self.use_beta:
            gamma = params[:, :self.feature_dim]
            beta = params[:, self.feature_dim:]
        else:
            gamma = params
            beta = 0.0  # Fix #2: Use float instead of int
        
        # Handle different feature dimensions efficiently
        if features.dim() == 2:
            # Simple 2D case
            return features * gamma + beta
        else:
            # Reshape for broadcasting
            shape = [gamma.size(0)] + [1] * (features.dim() - 2) + [self.feature_dim]
            gamma = gamma.view(*shape)
            if self.use_beta:
                beta = beta.view(*shape)
            return features * gamma + beta


class ChaoticMLP(nn.Module):
    """Enhanced MLP designed for chaotic systems with special time handling."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # ─── Dimensions ────────────────────────────────────────────────
        self.num_input_species  = len(config["data"]["species_variables"])
        self.num_output_species = len(
            config["data"].get("target_species_variables",
                               config["data"]["species_variables"]))
        self.num_globals        = len(config["data"]["global_variables"])

        # ─── Architecture flags ───────────────────────────────────────
        self.hidden_dims        = config["model"]["hidden_dims"]
        self.use_residual       = config["model"].get("use_residual", True)
        self.use_layernorm      = config["model"].get("use_layernorm", True)
        self.use_time_attention = config["model"].get("use_time_attention", False)

        self.time_encoding_dim  = config["model"].get("time_encoding_dim", 128)
        self.activation         = self._get_activation(
            config["model"].get("activation", "gelu")
        )
        dropout_rate            = config["model"].get("dropout", 0.0)
        self.dropout            = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # ─── FiLM config ──────────────────────────────────────────────
        film_cfg   = config.get("film", {})
        self.use_film = film_cfg.get("enabled", True)
        film_hid   = film_cfg.get("hidden_dims", [64, 64])
        film_act   = film_cfg.get("activation", "relu")

        # ─── Encoders ─────────────────────────────────────────────────
        self.time_encoder    = TimeEncoding(self.time_encoding_dim)
        in_state_dim         = self.num_input_species + self.num_globals
        first_hidden_dim     = self.hidden_dims[0]

        self.state_projection = nn.Linear(in_state_dim, first_hidden_dim)
        self.time_projection  = nn.Sequential(
            nn.Linear(self.time_encoding_dim, first_hidden_dim),
            nn.LayerNorm(first_hidden_dim),
            self.activation,
            nn.Linear(first_hidden_dim, first_hidden_dim),
            nn.LayerNorm(first_hidden_dim)
        )

        # ─── Main blocks ──────────────────────────────────────────────
        self.blocks       = nn.ModuleList()
        self.film_layers  = nn.ModuleList() if self.use_film else None
        self.attn_layers  = nn.ModuleList() if self.use_time_attention else None

        current_dim = first_hidden_dim
        for idx, next_dim in enumerate(self.hidden_dims):

            # ---------- core block ----------
            if self.use_residual and idx > 0:
                self.blocks.append(
                    ResidualBlock(current_dim,
                                  self.activation,
                                  dropout_rate,
                                  next_dim=next_dim)
                )
            else:
                self.blocks.append(
                    self._make_ff_block(current_dim, next_dim,
                                        layer_norm = (self.use_layernorm and idx > 0),
                                        dropout    = self.dropout)
                )

            # ---------- FiLM ----------
            if self.use_film:
                self.film_layers.append(
                    FiLMLayer(condition_dim = self.num_globals,
                              feature_dim   = next_dim,
                              hidden_dims   = film_hid,
                              activation    = film_act,
                              use_beta      = True)
                )

            # ---------- cross-attention ----------
            if self.use_time_attention and idx % 2 == 0:
                self.attn_layers.append(
                    SimpleCrossAttention(state_dim = next_dim,
                                         time_dim  = first_hidden_dim,
                                         num_heads = 8,
                                         dropout    = dropout_rate)
                )

            current_dim = next_dim  # <-- update for next loop

        # ─── Output head ──────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, current_dim // 2),
            self.activation,
            nn.Linear(current_dim // 2, self.num_output_species)
        )

        self._initialize_weights()

    def _make_ff_block(self,
                       in_dim:   int,
                       out_dim:  int,
                       layer_norm: bool,
                       dropout: Optional[nn.Dropout]):
        layers: List[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.extend([
            nn.Linear(in_dim, out_dim),
            self.activation
        ])
        if dropout is not None:
            layers.append(dropout)
        return nn.Sequential(*layers)
    
    
    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True),
            "swish": nn.SiLU(inplace=True)
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _initialize_weights(self):
        """Careful, depth-wise init tuned for chaotic kinetics."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Smaller gain keeps activations in a reasonable range
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # The output head gets an even smaller gain
        for layer in self.output_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input
        species      = x[:, :self.num_input_species]
        global_vars  = x[:, self.num_input_species : self.num_input_species + self.num_globals]
        t_raw        = x[:, -1:]

        # Encoders
        t_features   = self.time_encoder(t_raw)
        t_encoded    = self.time_projection(t_features)

        state_vec    = torch.cat([species, global_vars], dim=-1)
        h            = self.state_projection(state_vec)

        # Fuse state + (scaled) time signal
        h = h + 0.1 * t_encoded

        # Iterate through blocks
        attn_idx = 0
        for i, block in enumerate(self.blocks):
            h = block(h)

            if self.use_film:
                h = self.film_layers[i](h, global_vars)

            if self.use_time_attention and i % 2 == 0:
                h = self.attn_layers[attn_idx](h, t_encoded)
                attn_idx += 1

        # Output + positivity clamp
        out = self.output_head(h).clamp_min(1e-40)
        return out

class FiLMSIREN(nn.Module):
    """SIREN with FiLM conditioning and enhanced time handling."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Extract dimensions
        self.num_input_species = len(config["data"]["species_variables"])
        self.num_output_species = len(config["data"].get("target_species_variables", config["data"]["species_variables"]))
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]

        # SIREN parameters
        self.omega_0 = config["model"].get("omega_0", 30.0)
        self.use_time_modulation = config["model"].get("use_time_modulation", True)
        
        # Dropout
        dropout_rate = config["model"].get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # FiLM configuration
        film_config = config.get("film", {})
        self.use_film = film_config.get("enabled", True)

        # Time encoding for modulation
        if self.use_time_modulation:
            self.time_encoder = TimeEncoding(64)
            self.time_modulator = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, len(self.hidden_dims))
            )

        # Input dimension
        input_dim = self.num_input_species + self.num_globals + 1

        # Build network layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList() if self.use_film else None

        prev_dim = input_dim
        condition_dim = self.num_globals if self.use_film else None

        for i, dim in enumerate(self.hidden_dims):
            # Main layer
            self.layers.append(nn.Linear(prev_dim, dim))

            # FiLM layer
            if self.use_film:
                self.film_layers.append(
                    FiLMLayer(
                        condition_dim=condition_dim,
                        feature_dim=dim,
                        hidden_dims=film_config.get("hidden_dims", [128, 128]),
                        activation=film_config.get("activation", "gelu"),
                        use_beta=True
                    )
                )

            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.num_output_species)

        # Initialize SIREN weights
        self._initialize_siren_weights()
    
    def _initialize_siren_weights(self):
        """Initialize weights following SIREN paper (Fix #8)."""
        with torch.no_grad():
            # First layer
            if len(self.layers) > 0:
                fan_in = self.layers[0].in_features
                self.layers[0].weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
                # Fix #8: First layer bias should also be uniform
                if self.layers[0].bias is not None:
                    self.layers[0].bias.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            
            # Hidden layers
            for layer in self.layers[1:]:
                fan_in = layer.in_features
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
                layer.weight.uniform_(-bound, bound)
                # Fix #8: Hidden layer biases should be zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            
            # Output layer
            fan_in = self.output_layer.in_features
            bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with time-modulated frequencies."""
        
        # Extract time for modulation
        time = x[:, -1:]
        
        # Get time-dependent frequency modulation
        if self.use_time_modulation:
            time_enc = self.time_encoder(time)
            freq_modulation = self.time_modulator(time_enc)
            freq_modulation = torch.sigmoid(freq_modulation) * 2.0  # Scale between 0 and 2
        
        if self.use_film and self.film_layers is not None:
            cond_start = self.num_input_species
            cond_end = self.num_input_species + self.num_globals
            global_condition = x[:, cond_start:cond_end]

        # Process through layers
        h = x
        for i, layer in enumerate(self.layers):
            # Linear transformation
            h = layer(h)
            
            # Apply FiLM before activation
            if self.use_film and self.film_layers is not None:
                h = self.film_layers[i](h, global_condition)
            
            # SIREN activation with time modulation
            if self.use_time_modulation:
                omega = self.omega_0 * freq_modulation[:, i:i+1]
            else:
                omega = self.omega_0
            
            h = torch.sin(omega * h)
            
            # Dropout (except last layer)
            if self.dropout is not None and i < len(self.layers) - 1:
                h = self.dropout(h)
        
        # Output
        output = self.output_layer(h)
        
        # Enforce positivity
        output = output.clamp_min(1e-40)
        
        return output


class FiLMDeepONet(nn.Module):
    """Deep Operator Network with enhanced time processing for chaotic systems."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Dimensions
        self.num_input_species = len(config["data"]["species_variables"])
        self.num_output_species = len(config["data"].get("target_species_variables", config["data"]["species_variables"]))
        self.num_globals = len(config["data"]["global_variables"])

        # Architecture parameters
        branch_layers = config["model"]["branch_layers"]
        trunk_layers = config["model"]["trunk_layers"]
        self.basis_dim = config["model"]["basis_dim"]
        self.activation = self._get_activation(config["model"].get("activation", "gelu"))
        self.output_scale = config["model"].get("output_scale", 1.0)
        
        # Enhanced time processing
        self.use_time_encoding = config["model"].get("use_time_encoding", True)
        self.time_encoding_dim = config["model"].get("time_encoding_dim", 128)

        # Regularization
        dropout_rate = config["model"].get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # FiLM configuration
        film_config = config.get("film", {})
        self.use_film = film_config.get("enabled", True)
        condition_dim = self.num_globals if self.use_film else None

        # Branch input dimension
        branch_input_dim = (
            self.num_input_species
            if self.use_film
            else self.num_input_species + self.num_globals
        )

        # Build BRANCH network
        self.branch_net = self._build_mlp_with_film(
            input_dim=branch_input_dim,
            hidden_layers=branch_layers,
            output_dim=self.basis_dim * self.num_output_species,
            condition_dim=condition_dim,
            film_config=film_config if self.use_film else None
        )

        # Build TRUNK network with enhanced time processing
        if self.use_time_encoding:
            self.time_encoder = TimeEncoding(self.time_encoding_dim)
            trunk_input_dim = self.time_encoding_dim
        else:
            trunk_input_dim = 1
            
        self.trunk_net = self._build_mlp_with_film(
            input_dim=trunk_input_dim,
            hidden_layers=trunk_layers,
            output_dim=self.basis_dim,
            condition_dim=None,
            film_config=None,
            bias=True
        )
        
        # Optional basis function normalization for stability
        self.basis_norm = nn.LayerNorm(self.basis_dim)

    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True)
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _build_mlp_with_film(self, input_dim: int, hidden_layers: List[int],
                            output_dim: int, condition_dim: Optional[int] = None,
                            film_config: Optional[Dict] = None, bias: bool = True) -> nn.Module:
        """Build MLP with optional FiLM."""
        if self.use_film and condition_dim is not None and film_config is not None:
            # Build with FiLM
            layers = nn.ModuleList()
            film_layers = nn.ModuleList()

            prev_dim = input_dim
            for i, dim in enumerate(hidden_layers):
                layers.append(nn.Linear(prev_dim, dim, bias=bias))
                film_layers.append(
                    FiLMLayer(
                        condition_dim=condition_dim,
                        feature_dim=dim,
                        hidden_dims=film_config.get("hidden_dims", [128, 128]),
                        activation=film_config.get("activation", "gelu"),
                        use_beta=True
                    )
                )
                prev_dim = dim

            output_layer = nn.Linear(prev_dim, output_dim, bias=bias)

            class MLPWithFiLM(nn.Module):
                def __init__(self, layers, film_layers, output_layer, activation, dropout):
                    super().__init__()
                    self.layers = layers
                    self.film_layers = film_layers
                    self.output_layer = output_layer
                    self.activation = activation
                    self.dropout = dropout

                def forward(self, x, condition):
                    h = x
                    for i, (layer, film_layer) in enumerate(zip(self.layers, self.film_layers)):
                        h = layer(h)
                        h = film_layer(h, condition)
                        h = self.activation(h)
                        if self.dropout is not None and i < len(self.layers) - 1:
                            h = self.dropout(h)
                    return self.output_layer(h)

            return MLPWithFiLM(layers, film_layers, output_layer, self.activation, self.dropout)

        else:
            # Build standard MLP
            layers = []
            prev_dim = input_dim

            for i, dim in enumerate(hidden_layers):
                layers.append(nn.Linear(prev_dim, dim, bias=bias))
                layers.append(self.activation)
                if self.dropout is not None and i < len(hidden_layers) - 1:
                    layers.append(self.dropout)
                prev_dim = dim

            layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
            return nn.Sequential(*layers)
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced time processing."""
        batch_size = inputs.size(0)

        # Extract inputs
        if self.use_film:
            branch_input = inputs[:, :self.num_input_species]
            cond_start = self.num_input_species
            cond_end = cond_start + self.num_globals
            global_condition = inputs[:, cond_start:cond_end]
        else:
            branch_input = inputs[:, :self.num_input_species + self.num_globals]
            global_condition = None

        # Process time
        time = inputs[:, -1:]
        if self.use_time_encoding:
            trunk_input = self.time_encoder(time)
        else:
            trunk_input = time

        # Process through networks
        trunk_out = self.trunk_net(trunk_input)
        trunk_out = self.basis_norm(trunk_out)  # Normalize basis functions
        
        if self.use_film:
            branch_out = self.branch_net(branch_input, global_condition)
        else:
            branch_out = self.branch_net(branch_input)

        # Combine outputs
        branch_out = branch_out.view(batch_size, self.num_output_species, self.basis_dim)
        output = torch.bmm(branch_out, trunk_out.unsqueeze(2)).squeeze(2)

        # Apply output scaling
        if self.output_scale != 1.0:
            output = output * self.output_scale
        
        # Enforce positivity
        output = output.clamp_min(1e-40)
            
        return output


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and compile model with validation."""
    model_type = config["model"]["type"].lower()
    prediction_mode = config.get("prediction", {}).get("mode", "absolute")
    
    # Validate model-mode compatibility
    if prediction_mode == "ratio" and model_type not in ["deeponet", "mlp"]:
        raise ValueError(
            f"Prediction mode 'ratio' is only compatible with model types 'deeponet' or 'mlp', "
            f"but '{model_type}' was specified. Either:\n"
            f"  1. Change model.type to 'deeponet' or 'mlp' in your config, or\n"
            f"  2. Change prediction.mode to 'absolute'"
        )
    
    if model_type == "mlp":
        model = ChaoticMLP(config)
    elif model_type == "siren":
        model = FiLMSIREN(config)
    elif model_type == "deeponet":
        model = FiLMDeepONet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created {model_type} model for {prediction_mode} mode")
    
    # Compile model for performance
    if config["system"].get("use_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config["system"].get("compile_mode", "default")
        logger.info(f"Compiling model with mode='{compile_mode}'...")
        
        try:
            # A100-optimized compilation
            compile_options = {
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": False,
            }
            
            # Fix #9: Try reduce-overhead for complex control flow
            if model_type == "mlp" and config["model"].get("use_time_attention", False):
                compile_options["mode"] = "reduce-overhead"
                logger.info("Using reduce-overhead mode for complex attention control flow")
            
            if compile_mode == "max-autotune":
                # Additional options for maximum performance
                compile_options["options"] = {
                    "triton.cudagraphs": True,
                    "triton.max_autotune": True,
                }
            
            model = torch.compile(model, **compile_options)
            logger.info("Model compilation successful")
            
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")

    # Convert to specified dtype
    dtype_str = config["system"].get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
        logger.info("Converted model to float64 precision")
    elif dtype_str == "float32":
        model = model.float()
        logger.info("Using float32 precision")
        
    return model


def export_model(model: nn.Module, example_input: torch.Tensor, save_path: Path):
    """Export model with robust unwrapping and dynamic batch support."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    # Safely handle compiled models with multiple fallback attempts
    original_model = model
    if hasattr(model, '_orig_mod'):
        logger.info("Extracting original model from compiled wrapper (_orig_mod)")
        model = model._orig_mod
    elif hasattr(model, '_module'):
        logger.info("Extracting original model from compiled wrapper (_module)")
        model = model._module
    elif hasattr(model, 'module'):
        logger.info("Extracting original model from DataParallel/DistributedDataParallel wrapper")
        model = model.module
    else:
        # Try to detect if it's a compiled model by checking for graph attributes
        if hasattr(model, '_graph') or hasattr(model, '_code'):
            logger.warning(
                "Model appears to be compiled but cannot find unwrapping attribute. "
                "Export may fail or produce suboptimal results."
            )
    
    with torch.no_grad():
        try:
            if hasattr(torch, 'export') and hasattr(torch.export, 'export'):
                # Dynamic batch dimension
                batch_dim = Dim("batch", min=1, max=131072)
                
                # Safely detect parameter name
                import inspect
                try:
                    sig = inspect.signature(model.forward)
                    param_names = [p for p in sig.parameters.keys() if p != 'self']
                    param_name = param_names[0] if param_names else 'x'
                except Exception:
                    # Fallback if signature inspection fails
                    param_name = 'x' if hasattr(model, 'forward') else 'input'
                    logger.warning(f"Could not inspect forward signature, using '{param_name}' as parameter name")
                
                logger.info(f"Detected forward method parameter name: '{param_name}'")
                
                dynamic_shapes = {param_name: {0: batch_dim}}
                
                # Export with error handling
                try:
                    exported_program = torch.export.export(
                        model, 
                        (example_input,),
                        dynamic_shapes=dynamic_shapes
                    )
                    torch.export.save(exported_program, str(save_path))
                    logger.info(f"Model exported with torch.export to {save_path}")
                except Exception as e:
                    logger.warning(f"torch.export failed: {e}. Falling back to torch.jit")
                    raise
            else:
                # Direct to JIT if torch.export not available
                raise AttributeError("torch.export not available")
                
        except Exception:
            # Fallback to JIT tracing
            try:
                # Try with the original model if unwrapping failed
                model_to_trace = model
                traced_model = torch.jit.trace(model_to_trace, example_input)
                torch.jit.save(traced_model, str(save_path))
                logger.info(f"Model exported with torch.jit to {save_path}")
                logger.warning("JIT export may not support dynamic batch sizes as well as torch.export")
            except Exception as jit_error:
                # Last resort: try with the original compiled model
                if model is not original_model:
                    logger.warning("Trying JIT export with original (possibly compiled) model")
                    try:
                        traced_model = torch.jit.trace(original_model, example_input)
                        torch.jit.save(traced_model, str(save_path))
                        logger.info(f"Model exported with torch.jit (compiled version) to {save_path}")
                    except Exception as final_error:
                        logger.error(f"All export methods failed. Last error: {final_error}")
                        raise
                else:
                    logger.error(f"JIT export failed: {jit_error}")
                    raise