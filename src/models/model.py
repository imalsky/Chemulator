#!/usr/bin/env python3
"""
Model definitions for chemical kinetics prediction.

Supports:
- FiLM-SIREN
- ResNet
- DeepONet

With optional torch.compile for performance.
"""

import logging
import time
from typing import Dict, Any, List

import torch
import torch.nn as nn
from pathlib import Path

MIN_CONCENTRATION = 0.0  # Minimum value for clamping outputs in normalized space

# =============================================================================
# Common Utilities
# =============================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.scale = nn.Linear(in_dim, out_dim)
        self.shift = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * self.scale(cond) + self.shift(cond)

# =============================================================================
# FiLM-SIREN Model
# =============================================================================

class FiLM_SIREN(nn.Module):
    """
    SIREN network with FiLM conditioning for chemical kinetics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]
        self.omega_0 = 30.0  # SIREN frequency scaling
        
        # Input layer
        self.input_layer = nn.Linear(self.num_species + self.num_globals + 1, self.hidden_dims[0])
        
        # Hidden layers with FiLM
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            self.films.append(FiLM(1, self.hidden_dims[i+1]))  # Time-conditioned FiLM
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.num_species)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, num_species + num_globals + 1)
            
        Returns:
            Predicted species concentrations (batch_size, num_species)
        """
        time = x[:, -1].unsqueeze(-1)
        features = x[:, :-1]
        
        hidden = torch.sin(self.omega_0 * self.input_layer(x))
        
        for layer, film in zip(self.layers, self.films):
            hidden = film(torch.sin(self.omega_0 * layer(hidden)), time)
        
        delta = self.output_layer(hidden)
        
        # Residual connection with initial species
        prediction = features[:, :self.num_species] + delta
        return torch.clamp(prediction, min=MIN_CONCENTRATION)

# =============================================================================
# ResNet Model
# =============================================================================

class ChemicalResNet(nn.Module):
    """
    Residual network for chemical kinetics prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]
        self.activation = nn.SiLU()
        
        self.use_time_embedding = config["model"].get("use_time_embedding", True)
        self.time_embedding_dim = config["model"].get("time_embedding_dim", 256)
        
        if self.use_time_embedding:
            self.time_embed = nn.Sequential(
                nn.Linear(1, self.time_embedding_dim),
                self.activation,
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
            )
        
        base_dim = self.num_species + self.num_globals + (self.time_embedding_dim if self.use_time_embedding else 1)
        
        # Input projection (enhanced features)
        self.input_projection = nn.Linear(base_dim, base_dim * 2)
        
        # Input layer
        self.input_layer = nn.Linear(base_dim * 3, self.hidden_dims[0])
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        prev_dim = self.hidden_dims[0]
        for dim in self.hidden_dims[1:]:
            block = nn.Sequential(
                nn.Linear(prev_dim, dim),
                self.activation
            )
            if prev_dim != dim:
                proj = nn.Linear(prev_dim, dim)
            else:
                proj = None
            self.blocks.append(nn.ModuleList([block, proj]))
            prev_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.num_species)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (batch_size, num_species + num_globals + 1)
            
        Returns:
            Predicted species concentrations (batch_size, num_species)
        """
        initial_species = x[:, :self.num_species]
        global_vars = x[:, self.num_species:self.num_species + self.num_globals]
        time_raw = x[:, -1].unsqueeze(-1)
        
        if self.use_time_embedding:
            time_emb = self.time_embed(time_raw)
        else:
            time_emb = time_raw
        
        combined = torch.cat([initial_species, global_vars, time_emb], dim=-1)
        
        projected_feats = self.input_projection(combined)
        enhanced = torch.cat([combined, projected_feats], dim=-1)
        
        hidden = self.input_layer(enhanced)
        
        for block, proj in self.blocks:
            residual = hidden
            hidden = block(hidden)
            if proj is not None:
                residual = proj(residual)
            hidden = hidden + residual
        
        delta = self.output_layer(hidden)
        
        prediction = initial_species + delta
        return torch.clamp(prediction, min=MIN_CONCENTRATION)

# =============================================================================
# DeepONet Model
# =============================================================================

class ChemicalDeepONet(nn.Module):
    """
    Deep Operator Network for chemical kinetics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        
        branch_layers = config["model"]["branch_layers"]
        trunk_layers = config["model"]["trunk_layers"]
        self.basis_dim = config["model"]["basis_dim"]
        self.activation = self._get_activation(config["model"].get("activation", "gelu"))
        self.use_residual = config["model"].get("use_residual", True)
        self.output_scale = config["model"].get("output_scale", 1.0)
        
        # Branch net: processes initial conditions, outputs basis for each species
        self.branch_net = self._build_mlp(
            self.num_species + self.num_globals,
            branch_layers,
            self.basis_dim * self.num_species
        )
        
        # Trunk net: processes time
        self.trunk_net = self._build_mlp(1, trunk_layers, self.basis_dim)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1, self.num_species))
    
    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _build_mlp(self, input_dim: int, hidden_layers: List[int], output_dim: int) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                self.activation
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with corrected multi-output handling.
        
        Args:
            x: Input tensor (batch_size, num_species + num_globals + 1)
            
        Returns:
            Predicted species concentrations (batch_size, num_species)
        """
        initial_conditions = x[:, :-1]
        time = x[:, -1].unsqueeze(1)
        initial_species = x[:, :self.num_species]
        
        # Compute branch and trunk
        branch_out = self.branch_net(initial_conditions)  # (batch, basis_dim * num_species)
        trunk_out = self.trunk_net(time)  # (batch, basis_dim)
        
        # Reshape branch output
        branch_out = branch_out.view(-1, self.num_species, self.basis_dim)  # (batch, num_species, basis_dim)
        
        # Compute output for each species
        output = torch.einsum('bki,bi->bk', branch_out, trunk_out)  # (batch, num_species)
        output = output * self.output_scale + self.bias
        
        if self.use_residual:
            output = initial_species + output
            
        return torch.clamp(output, min=MIN_CONCENTRATION)

# =============================================================================
# Model Factory with Optional Compilation
# =============================================================================

def create_model(
    config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    Construct the requested network, move it to the target device and – if
    enabled – JIT-compile it in default mode.
    """
    kind = config["model"]["type"].lower()
    if kind == "siren":
        model = FiLM_SIREN(config)
    elif kind == "resnet":
        model = ChemicalResNet(config)
    elif kind == "deeponet":
        model = ChemicalDeepONet(config)
    else:
        raise ValueError(f"Unknown model type: {kind}")

    model = model.to(device)  # move first

    # -------------------------- optional compile -------------------------
    if not config["system"].get("use_torch_compile", False):
        return model

    # Only compile if supported
    if device.type != "cuda":
        logging.info("Compilation only supported on CUDA devices – running eager")
        return model

    # Timing
    compile_start = time.time()
    logging.info("Starting model compilation in default mode…")

    try:
        model = torch.compile(model, mode="default")
    except Exception as e:
        logging.warning(f"Compilation failed: {e} – falling back to eager mode")
        return model
    finally:
        logging.info(f"Compilation finished in {time.time() - compile_start:.1f}s")

    return model


def export_jit_model(
    model: nn.Module,
    example_input: torch.Tensor,
    save_path: Path
):
    """
    Export model as TorchScript.
    
    Args:
        model: Model to export
        example_input: Example input tensor
        save_path: Path to save JIT model
    """
    logger = logging.getLogger(__name__)
    
    # Set to eval mode
    model.eval()
    
    with torch.no_grad():
        try:
            # For better JIT compatibility, use torch.jit.trace
            traced = torch.jit.trace(model, example_input)
            
            # Optimize for inference
            traced = torch.jit.freeze(traced)
            traced = torch.jit.optimize_for_inference(traced)
            
            # Save
            torch.jit.save(traced, str(save_path))
            logger.info(f"JIT model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to export JIT model: {e}")
            raise