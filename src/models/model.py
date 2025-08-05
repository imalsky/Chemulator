#!/usr/bin/env python3
"""
Linear Latent Network models with constant-velocity latent dynamics and time warping.
LiLaN form: z(t) = E(x0, p) + τ(t, x0, p) ∘ C(x0, p); outputs are y(t) = D([z(t), p]).
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeWarp(nn.Module):
    """Monotone time warping module for multi-dimensional time."""
    def __init__(self, n_species: int, n_globals: int, latent_dim: int, J_terms: int = 3):
        super().__init__()
        self.J_terms = J_terms
        self.latent_dim = latent_dim
        
        # Network to predict warp parameters for each latent dimension
        input_dim = n_species + n_globals
        hidden_dim = 64
        
        # Output: for each latent dimension, we need s, {a_j, b_j}
        output_dim = latent_dim * (1 + 2 * J_terms)
        
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize near identity
        with torch.no_grad():
            bias = self.param_net[-1].bias
            bias_view = bias.view(latent_dim, 1 + 2 * J_terms)
            bias_view[:, 0] = 1.0  # s around 1
            bias_view[:, 1:1+J_terms] = 0.01  # a_j small
            bias_view[:, 1+J_terms:] = 1.0  # b_j around 1
    
    def forward(self, t_norm: torch.Tensor, x0_log: torch.Tensor, g_params: torch.Tensor) -> torch.Tensor:
        """
        Apply monotone time warp to produce multi-dimensional time.

        Args:
            t_norm: Normalized time in [0, 1], shape [B, M]
            x0_log: Initial log-species, shape [B, n_species]
            g_params: Global parameters, shape [B, n_globals]
        Returns:
            Warped time tau(t), shape [B, M, latent_dim]
        """
        B = x0_log.size(0)
        M = t_norm.size(1)

        # Predict warp params
        features = torch.cat([x0_log, g_params], dim=1)
        params = self.param_net(features).view(B, self.latent_dim, 1 + 2 * self.J_terms)

        # Positive parameters via softplus
        s = F.softplus(params[:, :, 0:1])                       # [B, D, 1]
        a = F.softplus(params[:, :, 1:1+self.J_terms])          # [B, D, J]
        b = F.softplus(params[:, :, 1+self.J_terms:])           # [B, D, J]

        # Broadcast time
        t_norm_expanded = t_norm.unsqueeze(1).expand(B, self.latent_dim, M)  # [B, D, M]

        # tau(t) = s*t + Σ_j a_j * (1 - exp(-b_j * t))
        # Use expm1 for better precision when b*t is small: 1 - exp(-x) = -expm1(-x)
        tau = s.expand(B, self.latent_dim, M) * t_norm_expanded
        t_be = t_norm_expanded.unsqueeze(2)                     # [B, D, 1, M]
        a_be = a.unsqueeze(-1)                                  # [B, D, J, 1]
        b_be = b.unsqueeze(-1)                                  # [B, D, J, 1]
        tau = tau + (a_be * (-torch.expm1(-b_be * t_be))).sum(dim=2)  # sum over J

        # [B, M, D]
        return tau.transpose(1, 2)



class LinearLatentMixture(nn.Module):
    """
    Linear Latent Network with constant velocity dynamics.
    Key equation: z(t) = E(x0, p) + τ(t, x0, p) ◦ C(x0, p)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Dimensions
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.num_targets = len(config["data"].get("target_species_variables", 
                                                  config["data"]["species_variables"]))
        
        # Model config
        model_config = config["model"]
        self.latent_dim = model_config.get("latent_dim", 64)
        self.K = model_config.get("mixture", {}).get("K", 1)
        self.use_time_warp = model_config.get("time_warp", {}).get("enabled", False)
        
        # Architecture parameters
        self.encoder_layers = model_config.get("encoder_layers", [256, 256, 128])
        self.decoder_layers = model_config.get("decoder_layers", [128, 256, 256])
        
        # Activation and dropout
        self.activation = self._get_activation(model_config.get("activation", "gelu"))
        dropout_rate = model_config.get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Build encoder E: (x0_log, globals) -> y0 (initial latent state)
        encoder_input_dim = self.num_species + self.num_globals
        self.encoder_E = self._build_mlp(
            encoder_input_dim,
            self.encoder_layers,
            self.latent_dim * self.K,
            use_layernorm=True
        )
        
        # Build encoder C: (x0_log, globals) -> velocity (constant velocity)
        self.encoder_C = self._build_mlp(
            encoder_input_dim,
            self.encoder_layers,
            self.latent_dim * self.K,
            use_layernorm=True
        )
        
        # Mixture gate network
        if self.K > 1:
            self.gate_net = nn.Sequential(
                nn.Linear(encoder_input_dim, 64),
                self.activation,
                nn.Linear(64, 32),
                self.activation,
                nn.Linear(32, self.K)
            )
            mix_cfg = model_config.get("mixture", {})
            self.gate_temperature = float(mix_cfg.get("temperature", 1.0))  # default 1.0
        
        # Time warp module
        if self.use_time_warp:
            J_terms = model_config.get("time_warp", {}).get("J_terms", 3)
            self.time_warp = TimeWarp(self.num_species, self.num_globals, self.latent_dim, J_terms)
        
        # Decoder: (z(t), globals) -> log-species
        decoder_input_dim = self.latent_dim + self.num_globals
        self.decoder = self._build_mlp(
            decoder_input_dim,
            self.decoder_layers,
            self.num_targets,
            use_layernorm=True,
            final_activation=False
        )
        
        self._initialize_weights()

    def set_gate_temperature(self, temp: float):
        self.gate_temperature = float(max(1e-3, temp))
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "elu": nn.ELU(inplace=True)
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _build_mlp(self, input_dim: int, hidden_layers: List[int], 
                   output_dim: int, use_layernorm: bool = True,
                   final_activation: bool = True) -> nn.Sequential:
        """Build a multi-layer perceptron."""
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_layers):
            if i > 0 and use_layernorm:
                layers.append(nn.LayerNorm(prev_dim))
            
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            
            if self.dropout is not None:
                layers.append(self.dropout)
            
            prev_dim = dim
        
        # Output layer
        if use_layernorm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if final_activation:
            layers.append(self.activation)
        
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """Initialize weights for stability (encoders/decoder + TimeWarp if present)."""
        # Encoders/decoder
        for module in [self.encoder_E, self.encoder_C, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # TimeWarp (if enabled): Xavier on all linear layers, then set final bias via softplus-inverse
        if self.use_time_warp:
            # 1) Initialize all TimeWarp linear layers
            for m in self.time_warp.param_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # 2) Overwrite the LAST layer so output == bias at init (input-independent)
            last = self.time_warp.param_net[-1]
            if isinstance(last, nn.Linear):
                # Make the last layer produce only the bias initially
                nn.init.zeros_(last.weight)
                if last.bias is None:
                    last.bias = nn.Parameter(
                        torch.zeros(self.latent_dim * (1 + 2 * self.time_warp.J_terms),
                                    dtype=last.weight.dtype, device=last.weight.device)
                    )

                with torch.no_grad():
                    J = self.time_warp.J_terms

                    # Desired *post-softplus* values (near identity)
                    # s ≈ 1, a_j ≈ 0.01 (small), b_j ≈ 1
                    desired_s = torch.full((self.latent_dim, 1), 1.0,
                                        dtype=last.weight.dtype, device=last.weight.device)
                    desired_a = torch.full((self.latent_dim, J), 0.01,
                                        dtype=last.weight.dtype, device=last.weight.device)
                    desired_b = torch.full((self.latent_dim, J), 1.0,
                                        dtype=last.weight.dtype, device=last.weight.device)

                    # Stable inverse softplus: x = log(exp(y) - 1)
                    def softplus_inv(y: torch.Tensor) -> torch.Tensor:
                        return torch.where(
                            y > 20,
                            y,  # for large y, softplus ~ y
                            torch.log(torch.expm1(y))
                        )

                    bias_view = last.bias.view(self.latent_dim, 1 + 2 * J)
                    bias_view[:, 0:1]           = softplus_inv(desired_s)  # s pre-activations
                    bias_view[:, 1:1+J]         = softplus_inv(desired_a)  # a_j pre-activations
                    bias_view[:, 1+J:1+2*J]     = softplus_inv(desired_b)  # b_j pre-activations

            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: latent linear dynamics with optional time warp, then decode.
        Args:
            inputs: [B, n_species + n_globals + M] (last M are normalized times)
        Returns:
            outputs: [B, M, n_targets] (log-abundances)
        """
        B = inputs.size(0)

        # Parse inputs
        x0_log = inputs[:, :self.num_species]
        globals_vec = inputs[:, self.num_species:self.num_species + self.num_globals]
        t_norm = inputs[:, self.num_species + self.num_globals:]  # [B, M]
        M = t_norm.size(1)

        # Encoders
        encoder_input = torch.cat([x0_log, globals_vec], dim=1)
        y0_all = self.encoder_E(encoder_input)  # [B, K*D] or [B, D] if K==1
        c_all  = self.encoder_C(encoder_input)  # [B, K*D] or [B, D]

        # Time-warp (shared across mixture components)
        if self.use_time_warp:
            tau = self.time_warp(t_norm, x0_log, globals_vec)  # [B, M, D]
        else:
            tau = t_norm.unsqueeze(-1).expand(B, M, self.latent_dim)  # [B, M, D]

        if self.K > 1:
            # Reshape to [B, K, D]
            y0_all = y0_all.view(B, self.K, self.latent_dim)
            c_all  = c_all.view(B, self.K, self.latent_dim)

            # Gating weights p: [B, K]
            logits = self.gate_net(encoder_input)
            p = F.softmax(logits / self.gate_temperature, dim=-1)  # [B, K]

            # Vectorized trajectories for all K:
            # y0 -> [B, K, M, D]; c -> [B, K, M, D]; tau -> [B, 1, M, D]
            y0_exp = y0_all.unsqueeze(2).expand(B, self.K, M, self.latent_dim)
            c_exp  = c_all.unsqueeze(2).expand(B, self.K, M, self.latent_dim)
            tau_exp = tau.unsqueeze(1)  # [B, 1, M, D]
            z_t_all = y0_exp + tau_exp * c_exp  # [B, K, M, D]

            # Weighted sum over K
            z_t = (z_t_all * p.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, M, D]
        else:
            # Single component
            y0 = y0_all.view(B, self.latent_dim)  # [B, D]
            c  = c_all.view(B, self.latent_dim)   # [B, D]
            z_t = y0.unsqueeze(1) + tau * c.unsqueeze(1)  # [B, M, D]

        # Decode per time step
        globals_expanded = globals_vec.unsqueeze(1).expand(B, M, self.num_globals)  # [B, M, G]
        decoder_input = torch.cat([z_t, globals_expanded], dim=-1)                  # [B, M, D+G]
        output_flat = self.decoder(decoder_input.view(B * M, -1))                   # [B*M, T]
        return output_flat.view(B, M, self.num_targets)                              # [B, M, T]

    def get_regularization_losses(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute regularization losses for mixture model."""
        losses: Dict[str, torch.Tensor] = {}

        if self.K <= 1:
            return losses

        B = inputs.size(0)
        x0_log = inputs[:, :self.num_species]
        globals_vec = inputs[:, self.num_species:self.num_species + self.num_globals]
        encoder_input = torch.cat([x0_log, globals_vec], dim=1)

        # Gate entropy (maximize entropy -> add negative entropy to loss)
        logits = self.gate_net(encoder_input)                               # [B, K]
        p = F.softmax(logits / self.gate_temperature, dim=-1)               # [B, K]
        entropy = -(p * (p.clamp_min(1e-8)).log()).sum(dim=-1).mean()       # scalar
        losses["entropy_loss"] = -entropy

        # Velocity diversity across components (vectorized; optional sampling for large K)
        c_all = self.encoder_C(encoder_input).view(B, self.K, self.latent_dim)  # [B, K, D]
        if self.K <= 8:
            diffs = c_all[:, :, None, :] - c_all[:, None, :, :]   # [B, K, K, D]
            dists = torch.norm(diffs, p=2, dim=-1)                # [B, K, K]
            mask = torch.triu(torch.ones(self.K, self.K, device=c_all.device, dtype=torch.bool), 1)
            mean_dist = dists[:, mask].mean()
        else:
            S = min(8 * self.K, self.K * (self.K - 1) // 2)
            i = torch.randint(0, self.K, (S,), device=c_all.device)
            j = torch.randint(0, self.K, (S,), device=c_all.device)
            valid = i != j
            i, j = i[valid], j[valid]
            mean_dist = torch.norm(c_all[:, i, :] - c_all[:, j, :], p=2, dim=-1).mean()

        losses["generator_diversity"] = -mean_dist  # add to loss to push components apart
        return losses




def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_type = config["model"]["type"].lower()
    
    if model_type in ["linear_latent", "linear_latent_mixture"]:
        model = LinearLatentMixture(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set precision
    dtype_str = config["system"].get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    
    model = model.to(device)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created {model_type} model with constant velocity dynamics")
    logger.info(f"  Mixture components K: {model.K}")
    logger.info(f"  Time warping: {model.use_time_warp}")
    logger.info(f"  Latent dimension: {model.latent_dim}")
    
    # Compile if requested
    if config["system"].get("use_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config["system"].get("compile_mode", "default")
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    return model