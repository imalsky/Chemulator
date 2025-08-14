#!/usr/bin/env python3
"""
Linear Latent Network (LiLaN) model implementation for stiff ODEs.

This module implements the LiLaN architecture from the paper "LiLaN: A Linear 
Latent Network Approach for Real-Time Solutions of Stiff Nonlinear Ordinary 
Differential Equations" by Nockolds et al.

Core idea:
    y(t) = E(x0, p) + τ(t, x0, p) ⊙ C(x0, p)   # Linear latent dynamics
    x(t) = D(y(t), p)                           # Decode to physical space

Key components:
- Encoder E: Maps (x0, p) to initial latent state
- Encoder C: Maps (x0, p) to constant latent velocity  
- Time transformation τ: Maps (t, x0, p) to monotonic time scaling per latent dimension
- Decoder D: Maps latent state (and optionally globals) to physical space
"""

import logging
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        PyTorch activation module
    """
    name = (name or "tanh").lower()
    activations = {
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
    }
    return activations.get(name, nn.Tanh())


def _build_mlp(
    layers: List[int],
    activation: nn.Module,
    dropout_p: float = 0.0,
    use_layernorm: bool = False,
    final_activation: bool = False,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron.
    
    Args:
        layers: List of layer dimensions [input, hidden1, hidden2, ..., output]
        activation: Activation function module
        dropout_p: Dropout probability
        use_layernorm: Whether to use layer normalization
        final_activation: Whether to apply activation after final layer
        
    Returns:
        Sequential MLP module
    """
    if len(layers) < 2:
        raise ValueError("MLP must have at least input and output dimensions")
    
    modules: List[nn.Module] = []
    
    for i in range(len(layers) - 1):
        # Add layer norm before linear (except first layer)
        if i > 0 and use_layernorm:
            modules.append(nn.LayerNorm(layers[i]))
        
        # Linear layer
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        
        # Activation (except last layer unless specified)
        if i < len(layers) - 2 or final_activation:
            modules.append(activation)
        
        # Dropout (except last layer)
        if dropout_p > 0 and i < len(layers) - 2:
            modules.append(nn.Dropout(dropout_p))
    
    return nn.Sequential(*modules)


# ============================================================================
# TIME TRANSFORMATION MODULE
# ============================================================================

class MonotonicTimeTransform(nn.Module):
    """
    Monotonic time transformation for each latent dimension.
    
    Implements per-dimension time warping following the paper's formulation:
        τ_d(t) = s_d * t + Σ_{j=1..J} a_{d,j} * (1 - exp(-b_{d,j} * t))
    
    where s, a, b are constrained positive via softplus to ensure monotonicity.
    This allows the model to "stretch" or "compress" time differently for each
    latent dimension, crucial for handling multiple timescales in stiff systems.
    
    The transformation is anchored so that τ(0) = 0 to match the paper's
    initial condition y(0) = E(x0, p).
    """
    
    def __init__(
        self, 
        context_dim: int, 
        latent_dim: int, 
        J_terms: int = 5,
        hidden_layers: List[int] = None
    ):
        """
        Initialize monotonic time transform.
        
        Args:
            context_dim: Dimension of context features (x0, p)
            latent_dim: Number of latent dimensions to transform
            J_terms: Number of expansion terms per dimension
            hidden_layers: Hidden layer sizes for parameter network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.J_terms = J_terms
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        # Network to predict transform parameters from context
        # Output: s (scale), a_j (amplitudes), b_j (rates) for each latent dim
        out_dim = latent_dim * (1 + 2 * J_terms)
        layers = [context_dim] + hidden_layers + [out_dim]
        
        self.param_network = _build_mlp(
            layers,
            nn.GELU(),
            dropout_p=0.0,
            use_layernorm=False,
            final_activation=False
        )
        
        # Initialize near identity transformation
        self._initialize_near_identity()
    
    def _initialize_near_identity(self):
        """Initialize the network to produce near-identity transformation."""
        # Find the last linear layer
        for module in reversed(list(self.param_network.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    with torch.no_grad():
                        # Reshape bias to access individual parameters
                        bias = module.bias.view(self.latent_dim, 1 + 2 * self.J_terms)
                        # Set s ≈ 1 (softplus(0.54) ≈ 1)
                        bias[:, 0] = 0.54
                        # Set a ≈ 0 (softplus(-5) ≈ 0)  
                        bias[:, 1:1 + self.J_terms] = -5
                        # Set b ≈ 1 (softplus(0.54) ≈ 1)
                        bias[:, 1 + self.J_terms:] = 0.54
                break
    
    def forward(self, t_batch: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply monotonic time transformation.
        
        Args:
            t_batch: Time values, shape [B*M, 1] (vectorized over time)
            context: Context features (x0, p), shape [B, context_dim]
            
        Returns:
            Transformed time values, shape [B*M, latent_dim]
        """
        B_times_M = t_batch.shape[0]
        B = context.shape[0]
        M = B_times_M // B
        
        # Predict transform parameters from context
        params = self.param_network(context)  # [B, latent_dim * (1 + 2*J)]
        params = params.view(B, self.latent_dim, 1 + 2 * self.J_terms)
        
        # Extract and constrain parameters to be positive
        s = F.softplus(params[:, :, 0])                        # [B, D]
        a = F.softplus(params[:, :, 1:1 + self.J_terms])       # [B, D, J]
        b = F.softplus(params[:, :, 1 + self.J_terms:])        # [B, D, J]
        
        # Expand for all time points
        s = s.unsqueeze(1).expand(B, M, self.latent_dim).reshape(B_times_M, self.latent_dim)
        a = a.unsqueeze(1).expand(B, M, self.latent_dim, self.J_terms).reshape(B_times_M, self.latent_dim, self.J_terms)
        b = b.unsqueeze(1).expand(B, M, self.latent_dim, self.J_terms).reshape(B_times_M, self.latent_dim, self.J_terms)
        
        # Expand time for each latent dimension
        t_exp = t_batch.expand(B_times_M, self.latent_dim)  # [B*M, D]
        
        # Linear component
        tau = s * t_exp  # [B*M, D]
        
        # Nonlinear components: sum over J terms
        # Using expm1 for numerical stability: 1 - exp(-x) = -expm1(-x)
        for j in range(self.J_terms):
            tau = tau + a[:, :, j] * (-torch.expm1(-b[:, :, j] * t_exp))
        
        return tau  # [B*M, latent_dim]


# ============================================================================
# MAIN LILAN MODEL
# ============================================================================

class LinearLatentNetwork(nn.Module):
    """
    Linear Latent Network (LiLaN) for solving stiff ODEs.
    
    Architecture from the paper:
    1. Encoder E maps (x0, p) to initial latent state y0
    2. Encoder C maps (x0, p) to constant latent velocity dy/dτ
    3. Time transformation τ maps (t, x0, p) to scaled time for each latent dimension
    4. Latent solution: y(t) = E(x0, p) + τ(t, x0, p) ⊙ C(x0, p)
    5. Decoder D maps latent state y(t) (and optionally p) to physical solution x(t)
    
    The key insight is that the latent dynamics are linear (constant velocity),
    allowing analytical solution without numerical integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LiLaN model.
        
        Args:
            config: Configuration dictionary with model/data settings
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        data_cfg = config["data"]
        model_cfg = config["model"]
        
        # Problem dimensions
        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.num_targets = len(data_cfg.get("target_species_variables", data_cfg["species_variables"]))
        
        # Model hyperparameters
        self.latent_dim = model_cfg.get("latent_dim", 32)
        activation_name = model_cfg.get("activation", "tanh")
        self.activation = _get_activation(activation_name)
        self.dropout_p = float(model_cfg.get("dropout", 0.0))
        use_layernorm = model_cfg.get("use_layernorm", True)
        
        # Whether to include globals in decoder (practical enhancement)
        self.decoder_use_globals = model_cfg.get("decoder_use_globals", True)
        
        # Time warp configuration
        self.use_monotonic_tau = model_cfg.get("use_monotonic_tau", True)
        self.tau_J_terms = model_cfg.get("tau_J_terms", 5)
        
        # Input dimension: initial conditions + global parameters
        input_dim = self.num_species + self.num_globals
        
        # ====== Build encoder networks ======
        
        # Encoder E: (x0, p) → y0 (initial latent state)
        encoder_E_layers = [input_dim] + list(model_cfg.get("encoder_layers", [256, 256, 128])) + [self.latent_dim]
        self.encoder_E = _build_mlp(
            encoder_E_layers,
            self.activation,
            self.dropout_p,
            use_layernorm,
            final_activation=False
        )
        
        # Encoder C: (x0, p) → dy/dτ (constant latent velocity)
        encoder_C_layers = [input_dim] + list(model_cfg.get("encoder_layers", [256, 256, 128])) + [self.latent_dim]
        self.encoder_C = _build_mlp(
            encoder_C_layers,
            self.activation,
            self.dropout_p,
            use_layernorm,
            final_activation=False
        )
        
        # Time transformation τ: (t, x0, p) → τ (scaled time per latent dimension)
        if self.use_monotonic_tau:
            # Use monotonic parameterization for stability
            self.tau_network = MonotonicTimeTransform(
                context_dim=input_dim,
                latent_dim=self.latent_dim,
                J_terms=self.tau_J_terms,
                hidden_layers=model_cfg.get("tau_hidden_layers", [128, 128])
            )
        else:
            # Use unconstrained neural network (as in paper, but less stable)
            time_input_dim = 1 + self.num_species + self.num_globals
            tau_layers = [time_input_dim] + list(model_cfg.get("tau_layers", [256, 256, 128])) + [self.latent_dim]
            self.tau_network = _build_mlp(
                tau_layers,
                self.activation,
                self.dropout_p,
                use_layernorm,
                final_activation=False
            )
        
        # ====== Build decoder network ======
        
        # Decoder D: y (+ optionally p) → x (latent to physical space)
        if self.decoder_use_globals:
            decoder_input_dim = self.latent_dim + self.num_globals
        else:
            decoder_input_dim = self.latent_dim
            
        decoder_layers = [decoder_input_dim] + list(model_cfg.get("decoder_layers", [128, 256, 256])) + [self.num_targets]
        self.decoder_D = _build_mlp(
            decoder_layers,
            self.activation,
            self.dropout_p,
            use_layernorm,
            final_activation=False
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Restore near-identity init for τ AFTER global init so it is not overwritten
        if self.use_monotonic_tau and hasattr(self.tau_network, "_initialize_near_identity"):
            self.tau_network._initialize_near_identity()
        
        self.logger.info(
            f"Created LiLaN model: latent_dim={self.latent_dim}, "
            f"activation={activation_name}, "
            f"monotonic_tau={self.use_monotonic_tau}, "
            f"decoder_use_globals={self.decoder_use_globals}, "
            f"n_params={sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the LiLaN model.
        
        Following the paper's formulation:
        1. Parse inputs into x0 (initial conditions), p (parameters), and t (times)
        2. Compute y0 = E(x0, p) and c = C(x0, p)
        3. Compute τ = τ(t, x0, p) for each time point (vectorized)
        4. Compute latent trajectory: y(t) = y0 + τ ⊙ c
        5. Decode to physical space: x(t) = D(y(t), p)
        
        Args:
            inputs: Tensor of shape [B, S + G + M] containing:
                - S species initial conditions (already log-transformed)
                - G global parameters (normalized)
                - M normalized time points in [0,1]
                
        Returns:
            predictions: Model outputs of shape [B, M, T]
            aux: Empty dictionary (for compatibility)
        """
        B = inputs.size(0)
        
        # Parse inputs
        x0_log = inputs[:, :self.num_species]                                      # [B, S]
        g_vec = inputs[:, self.num_species:self.num_species + self.num_globals]    # [B, G]
        t_norm = inputs[:, self.num_species + self.num_globals:]                   # [B, M]
        M = t_norm.size(1)
        
        # Combine initial conditions and parameters for encoders
        encoder_input = torch.cat([x0_log, g_vec], dim=1)  # [B, S+G]
        
        # Compute initial latent state and velocity
        y0 = self.encoder_E(encoder_input)  # [B, latent_dim]
        c = self.encoder_C(encoder_input)   # [B, latent_dim]
        
        # Compute time transformation (vectorized)
        if self.use_monotonic_tau:
            # Monotonic transform expects [B*M, 1] times and [B, context_dim] context
            t_flat = t_norm.reshape(B * M, 1)  # [B*M, 1]
            tau_flat = self.tau_network(t_flat, encoder_input)  # [B*M, latent_dim]
            tau = tau_flat.view(B, M, self.latent_dim)  # [B, M, latent_dim]
        else:
            # Unconstrained neural network
            # Prepare input: concatenate time with context for each time point
            t_flat = t_norm.reshape(B * M, 1)  # [B*M, 1]
            context_expanded = encoder_input.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)  # [B*M, S+G]
            tau_input = torch.cat([t_flat, context_expanded[:, :self.num_species], context_expanded[:, self.num_species:]], dim=1)  # [B*M, 1+S+G]
            tau_flat = self.tau_network(tau_input)  # [B*M, latent_dim]
            tau = tau_flat.view(B, M, self.latent_dim)  # [B, M, latent_dim]
        
        # Anchor tau so that tau(t_start) = 0 (enforces y(0) = E(x0, p))
        tau = tau - tau[:, :1, :]  # Subtract first time point
        
        # Compute latent trajectory: y(t) = y0 + τ ⊙ c
        # Expand y0 and c for broadcasting
        y0_expanded = y0.unsqueeze(1).expand(B, M, self.latent_dim)  # [B, M, latent_dim]
        c_expanded = c.unsqueeze(1).expand(B, M, self.latent_dim)    # [B, M, latent_dim]
        
        # Element-wise multiplication of tau and c, then add y0
        y_t = y0_expanded + tau * c_expanded  # [B, M, latent_dim]
        
        # Decode each time point
        if self.decoder_use_globals:
            # Include global parameters in decoder (practical enhancement)
            g_expanded = g_vec.unsqueeze(1).expand(B, M, self.num_globals)  # [B, M, G]
            decoder_input = torch.cat([y_t, g_expanded], dim=-1)  # [B, M, latent_dim + G]
            decoder_input_flat = decoder_input.reshape(B * M, -1)  # [B*M, latent_dim + G]
        else:
            # Decoder uses only latent state (strict paper interpretation)
            decoder_input_flat = y_t.reshape(B * M, self.latent_dim)  # [B*M, latent_dim]
        
        x_flat = self.decoder_D(decoder_input_flat)  # [B*M, num_targets]
        
        # Reshape back to [B, M, num_targets]
        predictions = x_flat.view(B, M, self.num_targets)
        
        # Return predictions and empty aux dict
        return predictions, {}


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory function to create LiLaN model.
    
    Args:
        config: Configuration dictionary with model specifications
        device: Device to place model on (cpu/cuda/mps)
        
    Returns:
        Configured and initialized model
    """
    # Create model
    model = LinearLatentNetwork(config)
    
    # Set precision
    if str(config.get("system", {}).get("dtype", "float32")).lower() == "float64":
        model = model.double()
    
    # Move to device
    model = model.to(device)
    
    # Optional torch.compile for performance
    logger = logging.getLogger(__name__)
    if config.get("system", {}).get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            compile_mode = config["system"].get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"Model compilation successful (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")
    
    return model