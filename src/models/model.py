#!/usr/bin/env python3
"""
Linear Latent Network models with mixture of generators and time warping.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeWarp(nn.Module):
    """Monotone time warping module."""
    def __init__(self, n_species: int, n_globals: int, J_terms: int = 3):
        super().__init__()
        self.J_terms = J_terms
        
        # Network to predict warp parameters
        input_dim = n_species + n_globals
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1 + 2 * J_terms)  # s, {a_j, b_j}
        )
        
        # Initialize near identity
        with torch.no_grad():
            # Last layer bias: [s=1, a_j=small, b_j=1]
            bias = self.param_net[-1].bias
            bias[0] = 1.0  # s around 1
            bias[1:1+J_terms] = 0.01  # a_j small
            bias[1+J_terms:] = 1.0  # b_j around 1
    
    def forward(self, t_norm: torch.Tensor, x0_log: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        """
        Apply monotone time warp.
        
        Args:
            t_norm: Normalized time in [0, 1], shape [B] or [B, M]
            x0_log: Initial log-species, shape [B, n_species]
            globals: Global parameters, shape [B, n_globals]
            
        Returns:
            Warped time tau(t), same shape as t_norm
        """
        # Get warp parameters
        features = torch.cat([x0_log, globals], dim=1)
        params = self.param_net(features)
        
        # Extract parameters with softplus for positivity
        s = F.softplus(params[:, 0:1])  # [B, 1]
        a = F.softplus(params[:, 1:1+self.J_terms])  # [B, J]
        b = F.softplus(params[:, 1+self.J_terms:])  # [B, J]
        
        # Handle different time shapes
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(1)  # [B, 1]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute warp: tau(t) = s*t + sum_j a_j(1 - exp(-b_j*t))
        tau = s * t_norm  # [B, M]
        
        for j in range(self.J_terms):
            tau = tau + a[:, j:j+1] * (1 - torch.exp(-b[:, j:j+1] * t_norm))
        
        if squeeze_output:
            tau = tau.squeeze(1)
            
        return tau


class LinearLatentMixture(nn.Module):
    """
    Linear Latent Network with mixture of K generators and optional time warping.
    
    For sequence mode with multi-time supervision.
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
        self.generator_rank = model_config.get("generator", {}).get("rank", 8)
        
        # Activation and dropout
        self.activation = self._get_activation(model_config.get("activation", "gelu"))
        dropout_rate = model_config.get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Build encoder: (x0_log, globals) -> z0
        encoder_input_dim = self.num_species + self.num_globals
        self.encoder = self._build_mlp(
            encoder_input_dim,
            self.encoder_layers,
            self.latent_dim,
            use_layernorm=True
        )
        
        # Mixture components
        if self.K > 1:
            # Gate network: (x0_log, globals) -> p_k
            self.gate_net = nn.Sequential(
                nn.Linear(encoder_input_dim, 64),
                self.activation,
                nn.Linear(64, 32),
                self.activation,
                nn.Linear(32, self.K)
            )
            
            # K generator networks
            self.generator_nets = nn.ModuleList([
                self._build_generator_net() for _ in range(self.K)
            ])
        else:
            # Single generator
            self.generator_nets = nn.ModuleList([self._build_generator_net()])
        
        # Optional time warp
        if self.use_time_warp:
            J_terms = model_config.get("time_warp", {}).get("J_terms", 3)
            self.time_warp = TimeWarp(self.num_species, self.num_globals, J_terms)
        
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
    
    def _build_generator_net(self) -> nn.Module:
        """Build a stable generator network A(globals)."""
        class StableGenerator(nn.Module):
            def __init__(self, n_globals, latent_dim, rank, activation):
                super().__init__()
                self.latent_dim = latent_dim
                self.rank = rank
                
                # Network for symmetric negative definite part
                self.sym_net = nn.Sequential(
                    nn.Linear(n_globals, 64),
                    activation,
                    nn.Linear(64, latent_dim * rank)
                )
                
                # Networks for skew-symmetric part
                if rank > 0:
                    self.skew_u_net = nn.Sequential(
                        nn.Linear(n_globals, 64),
                        activation,
                        nn.Linear(64, latent_dim * rank)
                    )
                    self.skew_v_net = nn.Sequential(
                        nn.Linear(n_globals, 64),
                        activation,
                        nn.Linear(64, latent_dim * rank)
                    )
                
                # Stability margin
                self.alpha = 1.0
            
            def forward(self, globals: torch.Tensor) -> torch.Tensor:
                """Compute stable A matrix."""
                B = globals.size(0)
                d, r = self.latent_dim, self.rank
                
                # Symmetric negative definite: -C*C^T - alpha*I
                C = self.sym_net(globals).view(B, d, r)
                sym_part = -torch.bmm(C, C.transpose(1, 2))
                I = torch.eye(d, device=globals.device, dtype=globals.dtype).unsqueeze(0)
                sym_part = sym_part - self.alpha * I
                
                # Skew-symmetric part
                if r > 0 and hasattr(self, 'skew_u_net'):
                    U = self.skew_u_net(globals).view(B, d, r)
                    V = self.skew_v_net(globals).view(B, d, r)
                    skew = torch.bmm(U, V.transpose(1, 2)) - torch.bmm(V, U.transpose(1, 2))
                else:
                    skew = torch.zeros_like(sym_part)
                
                return sym_part + skew
        
        return StableGenerator(self.num_globals, self.latent_dim, 
                               self.generator_rank, self.activation)
    
    def _initialize_weights(self):
        """Initialize weights for stability."""
        for module in [self.encoder, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize generators with small values
        for gen in self.generator_nets:
            for m in gen.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence mode.
        
        Args:
            inputs: [B, n_species + n_globals + M] where last M dims are normalized times
            
        Returns:
            outputs: [B, M, n_targets] log-abundances at M time points
        """
        B = inputs.size(0)
        
        # Parse inputs
        x0_log = inputs[:, :self.num_species]
        globals = inputs[:, self.num_species:self.num_species + self.num_globals]
        t_norm = inputs[:, self.num_species + self.num_globals:]  # [B, M]
        M = t_norm.size(1)
        
        # Apply time warp if enabled
        if self.use_time_warp:
            t_eff = self.time_warp(t_norm, x0_log, globals)
        else:
            t_eff = t_norm
        
        # Encode initial condition
        encoder_input = torch.cat([x0_log, globals], dim=1)
        z0 = self.encoder(encoder_input)  # [B, latent_dim]
        
        if self.K > 1:
            # Mixture of generators
            # Get gate probabilities
            logits = self.gate_net(encoder_input)  # [B, K]
            p = F.softmax(logits / 0.1, dim=-1)  # Temperature scaling
            
            # Compute weighted evolution
            z_t_all = []
            for k in range(self.K):
                A_k = self.generator_nets[k](globals)  # [B, d, d]
                
                # Evolve for all times
                z_t_k = []
                for m in range(M):
                    At = A_k * t_eff[:, m:m+1].unsqueeze(-1)  # [B, d, d]
                    exp_At = torch.linalg.matrix_exp(At)  # [B, d, d]
                    z_m = torch.bmm(exp_At, z0.unsqueeze(-1)).squeeze(-1)  # [B, d]
                    z_t_k.append(z_m)
                
                z_t_k = torch.stack(z_t_k, dim=1)  # [B, M, d]
                z_t_all.append(z_t_k)
            
            # Weighted sum
            z_t_all = torch.stack(z_t_all, dim=1)  # [B, K, M, d]
            p_expanded = p.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            z_t = (z_t_all * p_expanded).sum(dim=1)  # [B, M, d]
            
        else:
            # Single generator
            A = self.generator_nets[0](globals)  # [B, d, d]
            
            # Evolve for all times
            z_t_list = []
            for m in range(M):
                At = A * t_eff[:, m:m+1].unsqueeze(-1)  # [B, d, d]
                exp_At = torch.linalg.matrix_exp(At)  # [B, d, d]
                z_m = torch.bmm(exp_At, z0.unsqueeze(-1)).squeeze(-1)  # [B, d]
                z_t_list.append(z_m)
            
            z_t = torch.stack(z_t_list, dim=1)  # [B, M, d]
        
        # Decode all time points
        globals_expanded = globals.unsqueeze(1).expand(B, M, self.num_globals)
        decoder_input = torch.cat([z_t, globals_expanded], dim=-1)  # [B, M, d+g]
        
        # Reshape for decoder
        decoder_input_flat = decoder_input.view(B * M, -1)
        output_flat = self.decoder(decoder_input_flat)  # [B*M, n_targets]
        outputs = output_flat.view(B, M, self.num_targets)  # [B, M, n_targets]
        
        return outputs
    
    def get_regularization_losses(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute regularization losses."""
        losses = {}
        
        if self.K > 1:
            # Gate entropy regularization
            x0_log = inputs[:, :self.num_species]
            globals = inputs[:, self.num_species:self.num_species + self.num_globals]
            encoder_input = torch.cat([x0_log, globals], dim=1)
            
            logits = self.gate_net(encoder_input)
            p = F.softmax(logits / 0.1, dim=-1)
            entropy = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
            losses['gate_entropy'] = -entropy  # Maximize entropy
            
            # Generator diversity (pairwise distance)
            if self.K > 1:
                diversity = 0.0
                count = 0
                for i in range(self.K):
                    for j in range(i+1, self.K):
                        A_i = self.generator_nets[i](globals)
                        A_j = self.generator_nets[j](globals)
                        diversity += torch.norm(A_i - A_j, p='fro', dim=(1,2)).mean()
                        count += 1
                if count > 0:
                    losses['generator_diversity'] = -diversity / count  # Maximize diversity
        
        return losses


# Keep LinearLatentDynamics for backward compatibility
class LinearLatentDynamics(LinearLatentMixture):
    """Alias for backward compatibility."""
    def __init__(self, config: Dict[str, Any]):
        # Force K=1 and no time warp for compatibility
        config = config.copy()
        config["model"]["mixture"] = {"K": 1}
        config["model"]["time_warp"] = {"enabled": False}
        super().__init__(config)


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and compile model."""
    model_type = config["model"]["type"].lower()
    
    # Map model types
    if model_type == "linear_latent":
        model = LinearLatentDynamics(config)
    elif model_type == "linear_latent_mixture":
        model = LinearLatentMixture(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created {model_type} model")
    
    # Log model details
    if hasattr(model, 'K'):
        logger.info(f"  Mixture components K: {model.K}")
    if hasattr(model, 'use_time_warp'):
        logger.info(f"  Time warping: {model.use_time_warp}")
    logger.info(f"  Latent dimension: {model.latent_dim}")
    
    # Compile model if requested
    if config["system"].get("use_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config["system"].get("compile_mode", "default")
        logger.info(f"Compiling model with mode='{compile_mode}'...")
        
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")
    
    # Set dtype
    dtype_str = config["system"].get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    return model


def export_model(model: nn.Module, example_input: torch.Tensor, save_path: Path):
    """Export model with torch.export."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    # Unwrap compiled models
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    with torch.no_grad():
        try:
            if hasattr(torch, 'export'):
                # Use torch.export for dynamic shapes
                from torch.export import export, save
                from torch.export import Dim
                
                # Dynamic batch and time dimensions
                batch_dim = Dim("batch", min=1, max=131072)
                time_dim = Dim("time", min=1, max=4096)
                
                # Adjust example for multi-time
                if example_input.dim() == 2:
                    # Add time dimension if needed
                    n_species = model.num_species
                    n_globals = model.num_globals
                    # Assume last dims are time
                    n_times = example_input.size(1) - n_species - n_globals
                    if n_times == 1:
                        # Expand to multiple times for export
                        base = example_input[:, :n_species + n_globals]
                        times = example_input[:, -1:].repeat(1, 3)
                        example_input = torch.cat([base, times], dim=1)
                
                exported = export(
                    model, 
                    (example_input,),
                    dynamic_shapes={"inputs": {0: batch_dim, -1: time_dim}}
                )
                save(exported, str(save_path))
                logger.info(f"Model exported with torch.export to {save_path}")
                
        except Exception as e:
            # Fallback to JIT
            logger.warning(f"torch.export failed: {e}. Using torch.jit")
            traced = torch.jit.trace(model, example_input)
            torch.jit.save(traced, str(save_path))
            logger.info(f"Model exported with torch.jit to {save_path}")