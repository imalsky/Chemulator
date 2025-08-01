#!/usr/bin/env python3
"""
Corrected chemical kinetics models with Linear Latent Dynamics.
All issues from the analysis have been addressed.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim


class LinearLatentDynamics(nn.Module):
    """
    Global-conditioned linear latent dynamics with analytic time evolution.
    
    Architecture:
    1. Encoder: (x0, g) -> z0
    2. Dynamics: z(t) = exp(A(g) * t) * z0, where A(g) is guaranteed stable
    3. Decoder: (z(t), g) -> x(t)
    
    Key fixes:
    - No output clamping in forward pass
    - Guaranteed stable A(g) using symmetric negative definite + skew symmetric decomposition
    - Support for multi-time queries per sample
    - Proper dtype handling for all operations
    - Deterministic time normalization behavior
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Dimensions
        self.num_input_species = len(config["data"]["species_variables"])
        self.num_output_species = len(config["data"].get("target_species_variables", 
                                                         config["data"]["species_variables"]))
        self.num_globals = len(config["data"]["global_variables"])
        
        # Model hyperparameters
        model_config = config["model"]
        self.latent_dim = model_config.get("latent_dim", 64)
        self.encoder_layers = model_config.get("encoder_layers", [256, 256, 128])
        self.decoder_layers = model_config.get("decoder_layers", [128, 256, 256])
        self.dynamics_rank = model_config.get("dynamics_rank", 8)
        self.use_time_normalization = model_config.get("use_time_normalization", True)
        
        # Stability margin (eigenvalues will be <= -alpha)
        self.alpha = model_config.get("alpha", 1.0)
        
        # Check if time is already normalized in preprocessing
        norm_config = config.get("normalization", {})
        time_norm_method = norm_config.get("methods", {}).get(config["data"]["time_variable"], "none")
        self.time_is_prenormalized = time_norm_method != "none"
        
        # Deterministic handling: disable time normalization if already normalized
        if self.time_is_prenormalized and self.use_time_normalization:
            self.logger.warning("Time is already normalized in preprocessing. Disabling use_time_normalization.")
            self.use_time_normalization = False
        
        # Activation and regularization
        self.activation = self._get_activation(model_config.get("activation", "gelu"))
        dropout_rate = model_config.get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Build encoder: (x0, g) -> z0
        encoder_input_dim = self.num_input_species + self.num_globals
        self.encoder = self._build_mlp(
            encoder_input_dim, 
            self.encoder_layers, 
            self.latent_dim,
            use_layernorm=True
        )
        
        # Optional learned time normalization
        if self.use_time_normalization:
            self.time_scale_net = nn.Sequential(
                nn.Linear(self.num_input_species + self.num_globals, 64),
                self.activation,
                nn.Linear(64, 32),
                self.activation,
                nn.Linear(32, 1),
                nn.Softplus()  # Ensure positive time scale
            )
        
        # Dynamics: A(g) = -C(g)C(g)^T - αI + (U(g)V(g)^T - V(g)U(g)^T)
        # Symmetric negative definite part
        self.dynamics_sym_c_net = nn.Sequential(
            nn.Linear(self.num_globals, 64),
            self.activation,
            nn.Linear(64, self.latent_dim * self.dynamics_rank)
        )
        
        # Skew-symmetric part (only adds imaginary eigenvalues)
        if self.dynamics_rank > 0:
            self.dynamics_skew_u_net = nn.Sequential(
                nn.Linear(self.num_globals, 64),
                self.activation,
                nn.Linear(64, self.latent_dim * self.dynamics_rank)
            )
            
            self.dynamics_skew_v_net = nn.Sequential(
                nn.Linear(self.num_globals, 64),
                self.activation,
                nn.Linear(64, self.latent_dim * self.dynamics_rank)
            )
        
        # Build decoder: (z(t), g) -> x(t)
        decoder_input_dim = self.latent_dim + self.num_globals
        self.decoder = self._build_mlp(
            decoder_input_dim,
            self.decoder_layers,
            self.num_output_species,
            use_layernorm=True,
            final_activation=False
        )
        
        # Initialize weights carefully
        self._initialize_weights()
        
    def _get_activation(self, name: str):
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
        """Initialize weights for stable dynamics."""
        # Initialize encoder/decoder with Xavier
        for module in [self.encoder, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize dynamics networks with small values
        for net in [self.dynamics_sym_c_net, 
                   getattr(self, 'dynamics_skew_u_net', None),
                   getattr(self, 'dynamics_skew_v_net', None)]:
            if net is not None:
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.uniform_(m.weight, -0.01, 0.01)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def compute_dynamics_matrix(self, global_vars: torch.Tensor) -> torch.Tensor:
        """
        Compute A(g) = -C(g)C(g)^T - αI + skew_symmetric_part
        
        This guarantees all eigenvalues have real part <= -α
        
        Args:
            global_vars: [batch_size, num_globals]
            
        Returns:
            A: [batch_size, latent_dim, latent_dim]
        """
        B, d, r = global_vars.size(0), self.latent_dim, self.dynamics_rank
        device, dtype = global_vars.device, global_vars.dtype
        
        # Symmetric negative definite part: -C*C^T - αI
        C = self.dynamics_sym_c_net(global_vars).view(B, d, r)
        symmetric_part = -torch.bmm(C, C.transpose(1, 2))  # [B, d, d]
        
        # Create identity with correct dtype
        I = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
        symmetric_part = symmetric_part - self.alpha * I
        
        # Skew-symmetric part (only affects imaginary eigenvalues)
        if r > 0:
            U = self.dynamics_skew_u_net(global_vars).view(B, d, r)
            V = self.dynamics_skew_v_net(global_vars).view(B, d, r)
            
            UV_T = torch.bmm(U, V.transpose(1, 2))
            VU_T = torch.bmm(V, U.transpose(1, 2))
            skew = UV_T - VU_T  # Guaranteed skew-symmetric
        else:
            # Use zero tensor with correct shape and dtype
            skew = torch.zeros_like(symmetric_part)
        
        return symmetric_part + skew
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with matrix exponential dynamics.
        
        Supports multi-time queries:
        - If time has shape [B, 1], returns [B, num_output_species]
        - If time has shape [B, M], returns [B, M, num_output_species]
        
        Args:
            inputs: [batch_size, num_input_species + num_globals + time_dims]
                   where time_dims is either 1 or M (multiple time points)
        """
        batch_size = inputs.size(0)
        
        # Extract inputs
        initial_species = inputs[:, :self.num_input_species]
        global_vars = inputs[:, self.num_input_species:self.num_input_species + self.num_globals]
        
        # Handle single or multiple time queries
        time_data = inputs[:, self.num_input_species + self.num_globals:]
        if time_data.dim() == 2 and time_data.size(1) == 1:
            # Single time per sample
            time = time_data  # [B, 1]
            multi_time = False
        else:
            # Multiple times per sample
            time = time_data  # [B, M]
            multi_time = True
            num_times = time.size(1)
        
        # Time normalization (deterministic based on preprocessing)
        if self.use_time_normalization and not self.time_is_prenormalized:
            encoder_input = torch.cat([initial_species, global_vars], dim=1)
            time_scale = self.time_scale_net(encoder_input).clamp_min(1e-3)  # [B, 1]
            
            # Optional: detach to prevent trivial solutions
            # time_scale = time_scale.detach()
            
            if multi_time:
                normalized_time = time / time_scale  # [B, M]
            else:
                normalized_time = time / time_scale  # [B, 1]
        else:
            normalized_time = time
        
        # Encode to latent space
        encoder_input = torch.cat([initial_species, global_vars], dim=1)
        z0 = self.encoder(encoder_input)  # [B, latent_dim]
        
        # Compute dynamics matrix A(g)
        A = self.compute_dynamics_matrix(global_vars)  # [B, latent_dim, latent_dim]
        
        if multi_time:
            # Multiple time queries
            # Reshape for batch matrix operations
            At = A.unsqueeze(1) * normalized_time.unsqueeze(-1).unsqueeze(-1)  # [B, M, d, d]
            At_flat = At.view(-1, self.latent_dim, self.latent_dim)  # [B*M, d, d]
            
            # Compute matrix exponential for all time points
            exp_At_flat = torch.linalg.matrix_exp(At_flat)  # [B*M, d, d]
            exp_At = exp_At_flat.view(batch_size, num_times, self.latent_dim, self.latent_dim)  # [B, M, d, d]
            
            # Apply to initial latent state
            z0_expanded = z0.unsqueeze(1).unsqueeze(-1)  # [B, 1, d, 1]
            z_t = torch.matmul(exp_At, z0_expanded).squeeze(-1)  # [B, M, d]
            
            # Decode with global conditioning
            global_vars_expanded = global_vars.unsqueeze(1).expand(batch_size, num_times, self.num_globals)
            decoder_input = torch.cat([z_t, global_vars_expanded], dim=-1)  # [B, M, d+g]
            decoder_input_flat = decoder_input.view(-1, self.latent_dim + self.num_globals)
            
            output_flat = self.decoder(decoder_input_flat)  # [B*M, num_output_species]
            output = output_flat.view(batch_size, num_times, self.num_output_species)  # [B, M, num_output_species]
            
        else:
            # Single time query
            At = A * normalized_time.unsqueeze(-1)  # [B, d, d]
            exp_At = torch.linalg.matrix_exp(At)  # [B, d, d]
            
            # Apply to initial latent state
            z_t = torch.bmm(exp_At, z0.unsqueeze(-1)).squeeze(-1)  # [B, d]
            
            # Decode with global conditioning
            decoder_input = torch.cat([z_t, global_vars], dim=1)
            output = self.decoder(decoder_input)  # [B, num_output_species]
        
        # NO CLAMPING - let the training handle the appropriate domain
        return output
    
    def get_dynamics_spectrum(self, global_vars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute eigenvalues of A(g) for analysis/regularization.
        
        Returns dict with 'eigenvalues' and 'max_real_part'
        """
        A = self.compute_dynamics_matrix(global_vars)
        eigenvalues = torch.linalg.eigvals(A)
        max_real_part = eigenvalues.real.max(dim=1)[0]
        
        return {
            'eigenvalues': eigenvalues,
            'max_real_part': max_real_part,
            'A_matrix': A
        }


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and compile model with validation."""
    model_type = config["model"]["type"].lower()
    
    # Map model types
    if model_type == "linear_latent":
        model = LinearLatentDynamics(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    logger = logging.getLogger(__name__)
    prediction_mode = config.get("prediction", {}).get("mode", "absolute")
    logger.info(f"Created {model_type} model for {prediction_mode} mode")
    
    # Log model details
    if model_type == "linear_latent":
        logger.info(f"  Latent dimension: {model.latent_dim}")
        logger.info(f"  Dynamics rank: {model.dynamics_rank}")
        logger.info(f"  Stability margin α: {model.alpha}")
        logger.info(f"  Time pre-normalized: {model.time_is_prenormalized}")
        logger.info(f"  Use time normalization: {model.use_time_normalization}")
    
    # Compile model for performance
    if config["system"].get("use_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config["system"].get("compile_mode", "default")
        logger.info(f"Compiling model with mode='{compile_mode}'...")
        
        try:
            compile_options = {
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": False,
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    return model


def add_spectral_regularization_loss(model: nn.Module, 
                                   inputs: torch.Tensor,
                                   lambda_reg: float = 0.01) -> torch.Tensor:
    """
    Add spectral regularization to encourage stable dynamics.
    
    Use this in your training loop:
    ```python
    # In trainer
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    if isinstance(model, LinearLatentDynamics):
        spectral_loss = add_spectral_regularization_loss(model, inputs, lambda_reg=0.01)
        loss = loss + spectral_loss
    ```
    
    Args:
        model: LinearLatentDynamics model
        inputs: Input batch
        lambda_reg: Regularization strength
        
    Returns:
        Spectral regularization loss
    """
    if not isinstance(model, LinearLatentDynamics):
        return torch.tensor(0.0, device=inputs.device)
    
    # Extract global variables
    global_vars = inputs[:, model.num_input_species:model.num_input_species + model.num_globals]
    
    # Get dynamics spectrum
    spectrum_info = model.get_dynamics_spectrum(global_vars)
    max_real_parts = spectrum_info['max_real_part']
    
    # Penalize positive real parts (eigenvalues > -alpha)
    # ReLU(max_real_part + alpha) penalizes when real part > -alpha
    penalty = torch.relu(max_real_parts + model.alpha).mean()
    
    return lambda_reg * penalty


def export_model(model: nn.Module, example_input: torch.Tensor, save_path: Path):
    """Export model with robust unwrapping and dynamic batch/time support."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    # Safely handle compiled models
    original_model = model
    if hasattr(model, '_orig_mod'):
        logger.info("Extracting original model from compiled wrapper (_orig_mod)")
        model = model._orig_mod
    elif hasattr(model, '_module'):
        logger.info("Extracting original model from compiled wrapper (_module)")
        model = model._module
    elif hasattr(model, 'module'):
        logger.info("Extracting original model from DataParallel wrapper")
        model = model.module
    
    with torch.no_grad():
        try:
            if hasattr(torch, 'export') and hasattr(torch.export, 'export'):
                # Dynamic dimensions for both batch and time
                batch_dim = Dim("batch", min=1, max=131072)
                time_dim = Dim("time", min=1, max=4096)
                
                # Detect parameter name
                import inspect
                try:
                    sig = inspect.signature(model.forward)
                    param_names = [p for p in sig.parameters.keys() if p != 'self']
                    param_name = param_names[0] if param_names else 'x'
                except Exception:
                    param_name = 'inputs'
                
                # Set dynamic shapes for both batch and time dimensions
                dynamic_shapes = {param_name: {0: batch_dim, -1: time_dim}}
                
                # Export with multi-time example to capture that path
                if example_input.size(-1) == model.num_input_species + model.num_globals + 1:
                    # Create a multi-time example
                    single_time = example_input[..., -1:]
                    multi_time = single_time.repeat(1, 3)  # 3 time points
                    example_input_multi = torch.cat([
                        example_input[..., :-1],
                        multi_time
                    ], dim=-1)
                else:
                    example_input_multi = example_input
                
                # Export
                exported_program = torch.export.export(
                    model, 
                    (example_input_multi,),
                    dynamic_shapes=dynamic_shapes
                )
                torch.export.save(exported_program, str(save_path))
                logger.info(f"Model exported with torch.export to {save_path} (supports variable batch and time dims)")
                
        except Exception as e:
            # Fallback to JIT
            logger.warning(f"torch.export failed: {e}. Using torch.jit")
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, str(save_path))
            logger.info(f"Model exported with torch.jit to {save_path}")