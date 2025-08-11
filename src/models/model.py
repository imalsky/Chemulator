#!/usr/bin/env python3
"""
Linear Latent Network (LiLaN) model implementation for stiff ODEs.

This module implements the LiLaN architecture from the paper "LiLaN: A Linear 
Latent Network Approach for Real-Time Solutions of Stiff Nonlinear Ordinary 
Differential Equations" by Nockolds et al.

Core idea:
    z(t) = E(x0, p) + τ(t, x0, p) ∘ C(x0, p)   # Linear latent dynamics
    y(t) = D(z(t), p)                          # Decode to physical space

Key innovations:
- Constant-velocity latent dynamics (analytically solvable)
- Learned time warping for handling multiple timescales
- Optional mixture-of-experts for complex dynamics

Supports two architectural variants:
- "full": Single model for all output dimensions (shared encoder/decoder)
- "independent": Separate model for each output dimension (better performance)
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional

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
        name: Activation function name (gelu, relu, tanh, silu, elu)
        
    Returns:
        PyTorch activation module
    """
    name = (name or "gelu").lower()
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(inplace=True),
        "elu": nn.ELU(inplace=True),
    }
    return activations.get(name, nn.GELU())


def _build_mlp(
    in_dim: int,
    hidden: List[int],
    out_dim: int,
    activation: nn.Module,
    dropout_p: float = 0.0,
    use_layernorm: bool = True,
    final_activation: bool = False,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron with optional normalization and dropout.
    
    Args:
        in_dim: Input dimension
        hidden: List of hidden layer dimensions
        out_dim: Output dimension
        activation: Activation function module
        dropout_p: Dropout probability (0 = no dropout)
        use_layernorm: Whether to use layer normalization
        final_activation: Whether to apply activation after final layer
        
    Returns:
        Sequential MLP module
    """
    layers: List[nn.Module] = []
    prev = in_dim
    
    # Build hidden layers
    for i, h in enumerate(hidden):
        # Add layer norm before linear (except first layer)
        if i > 0 and use_layernorm:
            layers.append(nn.LayerNorm(prev))
        
        # Linear layer + activation
        layers.extend([nn.Linear(prev, h), activation])
        
        # Optional dropout
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        
        prev = h
    
    # Final layer
    if hidden and use_layernorm:
        layers.append(nn.LayerNorm(prev))
    layers.append(nn.Linear(prev, out_dim))
    
    # Optional final activation
    if final_activation:
        layers.append(activation)
    
    return nn.Sequential(*layers)


# ============================================================================
# TIME WARP MODULE
# ============================================================================

class TimeWarp(nn.Module):
    """
    Learned monotonic time transformation for each latent dimension.
    
    Implements per-dimension time warping:
        τ_d(t) = s_d * t + Σ_{j=1..J} a_{d,j} * (1 - exp(-b_{d,j} * t))
    
    where s, a, b are constrained positive via softplus to ensure monotonicity.
    This allows the model to "stretch" or "compress" time differently for each
    latent dimension, crucial for handling multiple timescales in stiff systems.
    """
    
    def __init__(
        self, 
        context_dim: int, 
        latent_dim: int, 
        J_terms: int = 3, 
        hidden_dim: int = 64
    ):
        """
        Initialize time warp module.
        
        Args:
            context_dim: Dimension of context features (encoder output)
            latent_dim: Number of latent dimensions to warp
            J_terms: Number of expansion terms per dimension
            hidden_dim: Hidden layer size for parameter network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.J_terms = J_terms
        
        # Network to predict warp parameters from context
        # Output: s (scale), a_j (amplitudes), b_j (rates) for each latent dim
        out_dim = latent_dim * (1 + 2 * J_terms)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t_norm: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to normalized time values.
        
        Args:
            t_norm: Normalized time values in [0,1], shape [B, M]
            context: Context features from encoder, shape [B, context_dim]
            
        Returns:
            Warped time values, shape [B, M, D] where D is latent_dim
        """
        B, M = t_norm.shape
        
        # Predict warp parameters from context
        params = self.net(context).view(B, self.latent_dim, 1 + 2 * self.J_terms)
        
        # Extract and constrain parameters to be positive
        s = F.softplus(params[:, :, 0:1])                      # Scale: [B, D, 1]
        a = F.softplus(params[:, :, 1:1 + self.J_terms])       # Amplitudes: [B, D, J]
        b = F.softplus(params[:, :, 1 + self.J_terms:])        # Rates: [B, D, J]
        
        # Expand time for broadcasting: [B, D, M]
        t_exp = t_norm.unsqueeze(1).expand(B, self.latent_dim, M)
        
        # Linear component
        tau = s * t_exp
        
        # Nonlinear components: sum over J terms
        # Using expm1 for numerical stability: 1 - exp(-x) = -expm1(-x)
        exp_terms = (a.unsqueeze(-1) * (-torch.expm1(-b.unsqueeze(-1) * t_exp.unsqueeze(2)))).sum(dim=2)
        
        # Combine and transpose to [B, M, D] format
        return (tau + exp_terms).transpose(1, 2)


# ============================================================================
# MAIN LILAN MODEL
# ============================================================================

class LinearLatentMixture(nn.Module):
    """
    Linear Latent Network with optional mixture-of-experts and time warping.
    
    Architecture:
    1. Encoder networks (E, C) map initial conditions to latent initial state and velocity
    2. Time warp network τ transforms physical time to latent time
    3. Latent evolution: z(τ) = E(x0, p) + τ ∘ C(x0, p) (analytically solved)
    4. Decoder network D maps latent state back to physical space
    
    Inputs: [x0_log (S), globals (G), times (M)]
    Outputs: [B, M, T] predictions for target species
    """
    
    def __init__(self, config: Dict[str, Any], output_dim: Optional[int] = None):
        """
        Initialize LiLaN model.
        
        Args:
            config: Configuration dictionary with model/data/training settings
            output_dim: Optional override for number of outputs (for independent models)
        """
        super().__init__()
        self.log = logging.getLogger(__name__)
        
        data_cfg = config["data"]
        model_cfg = config["model"]

        # Extract dimensions from config
        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        
        # Allow output dimension override for independent architecture
        if output_dim is not None:
            self.num_targets = output_dim
        else:
            self.num_targets = len(data_cfg.get("target_species_variables", data_cfg["species_variables"]))
        
        self.latent_dim = int(model_cfg.get("latent_dim", 64))

        # Activation and regularization
        self.activation = _get_activation(model_cfg.get("activation", "gelu"))
        self.dropout_p = float(model_cfg.get("dropout", 0.0))

        # Mixture-of-experts configuration
        mix_cfg = model_cfg.get("mixture", {})
        self.K = int(mix_cfg.get("K", 1))  # Number of experts
        self.gate_temp = float(mix_cfg.get("temperature", 1.0))
        
        # Regularization settings for mixtures
        self.diversity_mode = str(mix_cfg.get("diversity_mode", "per_sample"))
        self.full_pair_threshold = int(mix_cfg.get("full_pair_threshold", 8))
        self.sample_factor = int(mix_cfg.get("sample_factor", 8))
        self.sample_pairs: Optional[int] = mix_cfg.get("sample_pairs", None)
        self.gate_use_features = bool(mix_cfg.get("use_encoder_features", True))

        # Time warp configuration
        tw_cfg = model_cfg.get("time_warp", {})
        self.use_time_warp = bool(tw_cfg.get("enabled", False))
        self.warp_use_features = bool(tw_cfg.get("use_encoder_features", True))
        self.warp_J = int(tw_cfg.get("J_terms", 3))
        self.warp_hidden = int(tw_cfg.get("hidden_dim", 64))

        # ====== Build network components ======
        
        # Encoder backbone: processes initial conditions and parameters
        enc_layers = list(model_cfg.get("encoder_layers", [256, 256, 128]))
        enc_in = self.num_species + self.num_globals
        enc_out = enc_layers[-1] if enc_layers else enc_in
        
        self.encoder = _build_mlp(
            in_dim=enc_in, 
            hidden=enc_layers, 
            out_dim=enc_out,
            activation=self.activation, 
            dropout_p=self.dropout_p,
            use_layernorm=True, 
            final_activation=False,
        )

        # Heads for latent initial state (E) and velocity (C)
        # Output K experts if using mixture
        self.y0_head = nn.Linear(enc_out, self.latent_dim * self.K)
        self.c_head = nn.Linear(enc_out, self.latent_dim * self.K)

        # Gating network for mixture-of-experts
        if self.K > 1:
            gate_layers = list(mix_cfg.get("gate_layers", [64, 32]))
            gate_input_dim = enc_out if self.gate_use_features else enc_in
            
            self.gate = _build_mlp(
                in_dim=gate_input_dim,
                hidden=gate_layers, 
                out_dim=self.K,
                activation=self.activation, 
                dropout_p=self.dropout_p,
                use_layernorm=bool(mix_cfg.get("gate_use_layernorm", False)),
                final_activation=False,
            )

        # Time warp module
        if self.use_time_warp:
            warp_context_dim = enc_out if self.warp_use_features else enc_in
            self.time_warp = TimeWarp(
                warp_context_dim, self.latent_dim, self.warp_J, self.warp_hidden
            )

        # Decoder: maps latent state + globals to target species
        dec_layers = list(model_cfg.get("decoder_layers", [128, 256, 256]))
        dec_in = self.latent_dim + self.num_globals
        
        self.decoder = _build_mlp(
            in_dim=dec_in, 
            hidden=dec_layers, 
            out_dim=self.num_targets,
            activation=self.activation, 
            dropout_p=self.dropout_p,
            use_layernorm=True, 
            final_activation=False,
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize network weights with sensible defaults.
        
        Uses Xavier initialization for most layers, with special handling for:
        - Gate network: initialized to uniform probabilities
        - Time warp: initialized close to identity transformation
        """
        # Standard Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Gate network: start with uniform expert probabilities
        if self.K > 1:
            for m in self.gate.modules():
                if isinstance(m, nn.Linear) and m.out_features == self.K:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                    break

        # Time warp: initialize close to identity (τ ≈ t)
        if self.use_time_warp:
            # Find last linear layer in time warp network
            last_linear = None
            for m in self.time_warp.net.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                with torch.no_grad():
                    J, D = self.time_warp.J_terms, self.latent_dim
                    bias = last_linear.bias.view(D, 1 + 2 * J)
                    bias[:, 0] = 0.54      # softplus(0.54) ≈ 1  => s ≈ 1
                    bias[:, 1:1 + J] = -5  # softplus(-5)  ≈ 0  => a ≈ 0
                    bias[:, 1 + J:] = 0.54 # softplus(0.54) ≈ 1  => b ≈ 1

    def set_gate_temperature(self, temperature: float) -> None:
        """
        Set temperature for mixture-of-experts gating.
        
        Args:
            temperature: Temperature for softmax (lower = more peaked distribution)
        """
        if self.K > 1:
            self.gate_temp = max(1e-3, float(temperature))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the LiLaN model.
        
        Args:
            inputs: Tensor of shape [B, S + G + M] containing:
                - S species initial conditions (already log-transformed)
                - G global parameters (normalized)
                - M normalized time points in [0,1]
                
        Returns:
            predictions: Model outputs of shape [B, M, T]
            aux: Dictionary with auxiliary outputs for regularization:
                - 'gate_p': Gating probabilities if using mixture
                - 'c_all': All velocity vectors for diversity loss
        """
        B = inputs.size(0)
        
        # Parse inputs
        x0_log = inputs[:, :self.num_species]
        g_vec = inputs[:, self.num_species:self.num_species + self.num_globals]
        t_norm = inputs[:, self.num_species + self.num_globals:]  # [B, M]
        M = t_norm.size(1)

        # Encode initial conditions and parameters
        enc_input = torch.cat([x0_log, g_vec], dim=1)  # [B, S+G]
        features = self.encoder(enc_input)              # [B, enc_out]

        # Predict latent initial state and velocity
        y0_all = self.y0_head(features)  # [B, K*D]
        c_all = self.c_head(features)    # [B, K*D]

        # Apply time warping if enabled
        if self.use_time_warp:
            warp_ctx = features if self.warp_use_features else enc_input
            tau = self.time_warp(t_norm, warp_ctx)  # [B, M, D]
        else:
            # No warping: use normalized time directly
            tau = t_norm.unsqueeze(-1).expand(B, M, self.latent_dim)

        # Auxiliary outputs for regularization
        aux: Dict[str, torch.Tensor] = {}

        if self.K > 1:
            # === Mixture-of-experts path ===
            
            # Reshape to separate experts
            y0_all = y0_all.view(B, self.K, self.latent_dim)  # [B, K, D]
            c_all = c_all.view(B, self.K, self.latent_dim)    # [B, K, D]

            # Compute gating probabilities
            gate_ctx = features if self.gate_use_features else enc_input
            logits = self.gate(gate_ctx)  # [B, K]
            probs = F.softmax(logits / max(self.gate_temp, 1e-3), dim=-1)  # [B, K]

            # Store for regularization
            aux["gate_p"] = probs
            aux["c_all"] = c_all

            # Compute latent trajectories for each expert
            # z_k(t) = y0_k + tau * c_k
            z_all = y0_all.unsqueeze(2) + tau.unsqueeze(1) * c_all.unsqueeze(2)  # [B, K, M, D]
            
            # Weighted sum over experts
            z_t = (z_all * probs.view(B, self.K, 1, 1)).sum(dim=1)  # [B, M, D]
        else:
            # === Single expert path ===
            y0 = y0_all.view(B, self.latent_dim)
            c = c_all.view(B, self.latent_dim)
            
            # Linear evolution in latent space
            z_t = y0.unsqueeze(1) + tau * c.unsqueeze(1)  # [B, M, D]

        # Decode latent trajectories to physical space
        # Replicate global parameters for each time step
        g_rep = g_vec.unsqueeze(1).expand(B, M, self.num_globals)  # [B, M, G]
        
        # Concatenate latent state with globals and decode
        dec_in = torch.cat([z_t, g_rep], dim=-1).reshape(B * M, -1)  # [B*M, D+G]
        out = self.decoder(dec_in).view(B, M, self.num_targets)       # [B, M, T]
        
        return out, aux

    def get_regularization_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for mixture-of-experts.
        
        Args:
            aux: Auxiliary outputs from forward pass
            
        Returns:
            Dictionary with unweighted regularization losses:
            - 'gate_kl_to_uniform': KL divergence of gate from uniform
            - 'generator_similarity': Similarity between expert generators
        """
        losses: Dict[str, torch.Tensor] = {}
        
        if not aux or self.K == 1:
            return losses

        # Gate entropy regularization: encourage uniform gate usage
        # KL(p || Uniform) = E_p[log p] + log K
        p = aux["gate_p"].clamp_min(1e-8)
        losses["gate_kl_to_uniform"] = (p * (p.log() + math.log(self.K))).sum(dim=-1).mean()

        # Generator diversity regularization: encourage diverse experts
        c_all = aux["c_all"]  # [B, K, D]
        
        if self.diversity_mode == "batch_mean":
            # Compute similarity on batch-averaged generators
            w = F.normalize(c_all.mean(dim=0), dim=-1)  # [K, D]
            sim = w @ w.T  # [K, K]
            
            # Extract upper triangular similarities
            i, j = torch.triu_indices(self.K, self.K, offset=1, device=sim.device)
            sims = sim[i, j].clamp_min(0)
            losses["generator_similarity"] = sims.mean() if sims.numel() else sim.new_tensor(0.0)
        else:
            # Per-sample diversity (default)
            w = F.normalize(c_all, dim=-1)  # [B, K, D]
            sim = torch.einsum("bkd,bld->bkl", w, w)  # [B, K, K]
            
            # Get upper triangular indices
            i, j = torch.triu_indices(self.K, self.K, offset=1, device=sim.device)
            num_pairs = (self.K * (self.K - 1)) // 2

            # Sample pairs if too many
            if self.K <= self.full_pair_threshold or num_pairs == 0:
                sims = sim[:, i, j]  # [B, P]
            else:
                # Subsample pairs for efficiency
                if self.sample_pairs is None:
                    S = min(self.sample_factor * self.K, num_pairs)
                else:
                    S = min(int(self.sample_pairs), num_pairs)
                
                choice = torch.randint(0, num_pairs, (S,), device=sim.device)
                sims = sim[:, i[choice], j[choice]]  # [B, S]

            sims = sims.clamp_min(0)
            per_sample = sims.mean(dim=1)  # [B]
            losses["generator_similarity"] = per_sample.mean()
        
        return losses


# ============================================================================
# INDEPENDENT ARCHITECTURE
# ============================================================================

class IndependentLinearLatentMixture(nn.Module):
    """
    Independent LiLaN architecture with separate models per output.
    
    Creates K independent LiLaN models, one for each target dimension.
    This typically achieves better performance than the full architecture
    at the cost of more parameters, as each output gets its own specialized
    encoder, latent dynamics, and decoder.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize independent architecture.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.log = logging.getLogger(__name__)
        
        data_cfg = config["data"]
        self.target_vars = data_cfg.get("target_species_variables", data_cfg["species_variables"])
        self.num_targets = len(self.target_vars)
        
        # Create independent model for each target dimension
        self.models = nn.ModuleList()
        for i in range(self.num_targets):
            # Each model predicts only one output dimension
            model = LinearLatentMixture(config, output_dim=1)
            self.models.append(model)
        
        self.log.info(f"Created independent architecture with {self.num_targets} separate models")
    
    def set_gate_temperature(self, temperature: float) -> None:
        """
        Set gate temperature for all sub-models.
        
        Args:
            temperature: Temperature for softmax gating
        """
        for model in self.models:
            model.set_gate_temperature(temperature)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run each model independently and concatenate outputs.
        
        Args:
            inputs: Input tensor of shape [B, S + G + M]
            
        Returns:
            predictions: Concatenated outputs [B, M, T]
            aux: Combined auxiliary data from all models
        """
        outputs = []
        all_aux = {}
        
        # Run each model independently
        for i, model in enumerate(self.models):
            out, aux = model(inputs)  # out: [B, M, 1]
            outputs.append(out)
            
            # Collect auxiliary data from each model
            for key, val in aux.items():
                if key not in all_aux:
                    all_aux[key] = []
                all_aux[key].append(val)
        
        # Concatenate outputs along target dimension
        predictions = torch.cat(outputs, dim=-1)  # [B, M, T]
        
        # Stack auxiliary tensors along model dimension
        for key in all_aux:
            if len(all_aux[key]) > 0:
                all_aux[key] = torch.stack(all_aux[key], dim=0)  # [T, ...]
        
        return predictions, all_aux
    
    def get_regularization_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses averaged over all sub-models.
        
        Args:
            aux: Stacked auxiliary outputs from all models
            
        Returns:
            Dictionary with averaged regularization losses
        """
        losses = {}
        
        # Gate KL regularization
        if "gate_p" in aux:
            gate_losses = []
            for model_idx in range(self.num_targets):
                if aux["gate_p"].size(0) > model_idx:
                    p = aux["gate_p"][model_idx].clamp_min(1e-8)
                    K = p.size(-1)
                    if K > 1:  # Only compute for mixture models
                        gate_loss = (p * (p.log() + math.log(K))).sum(dim=-1).mean()
                        gate_losses.append(gate_loss)
            
            if gate_losses:
                losses["gate_kl_to_uniform"] = torch.stack(gate_losses).mean()
        
        # Generator diversity regularization
        if "c_all" in aux:
            sim_losses = []
            for model_idx in range(self.num_targets):
                if aux["c_all"].size(0) > model_idx:
                    c = aux["c_all"][model_idx]
                    
                    # Check if this is a mixture model
                    if c.dim() == 3 and c.size(1) > 1:  # [B, K, D] with K > 1
                        w = F.normalize(c, dim=-1)
                        sim = torch.einsum("bkd,bld->bkl", w, w)
                        
                        K = c.size(1)
                        i_idx, j_idx = torch.triu_indices(K, K, offset=1, device=sim.device)
                        
                        if i_idx.numel() > 0:
                            sims = sim[:, i_idx, j_idx].clamp_min(0)
                            sim_losses.append(sims.mean())
            
            if sim_losses:
                losses["generator_similarity"] = torch.stack(sim_losses).mean()
        
        return losses


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory function to create LiLaN models.
    
    Supports:
    - Model variants: "full" (default) or "independent"
    - Model types: "linear_latent" or "linear_latent_mixture"
    - Optional torch.compile for optimization
    - Configurable precision (float32/float64)
    
    Args:
        config: Configuration dictionary with model specifications
        device: Device to place model on (cpu/cuda/mps)
        
    Returns:
        Configured and initialized model
        
    Raises:
        ValueError: If unknown model type or variant specified
    """
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "linear_latent_mixture").lower()
    model_variant = model_cfg.get("variant", "full").lower()
    
    # Validate model type
    if model_type not in {"linear_latent", "linear_latent_mixture"}:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Validate model variant
    if model_variant not in {"full", "independent"}:
        raise ValueError(f"Unknown model variant: {model_variant}. Choose 'full' or 'independent'")
    
    # Create model based on variant
    if model_variant == "independent":
        model = IndependentLinearLatentMixture(config)
        variant_str = "independent"
    else:
        model = LinearLatentMixture(config)
        variant_str = "full"
    
    # Set precision
    if str(config["system"].get("dtype", "float32")).lower() == "float64":
        model = model.double()
    
    # Move to device
    model = model.to(device)
    
    # Optional torch.compile for performance
    logger = logging.getLogger(__name__)
    if config["system"].get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            compile_mode = config["system"].get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"Model compilation successful (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")
    
    # Log model summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Created {variant_str} {model_type}: "
        f"{n_params:,} parameters, "
        f"latent_dim={model_cfg.get('latent_dim', 64)}"
    )
    
    return model