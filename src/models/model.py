#!/usr/bin/env python3
"""
Model definitions for chemical kinetics prediction.

Provides implementations of:
- FiLM-SIREN
- ResNet
- DeepONet

With optional torch.compile support and model export using torch.export (modern best practice replacing legacy JIT/TorchScript).
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn

# Constants
MIN_CONCENTRATION = 1e-30

# =============================================================================
# SIREN Model
# =============================================================================

class SIREN(nn.Module):
    """
    SIREN (Sinusoidal Representation Network) for chemical kinetics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]
        
        self.layers = nn.ModuleList()
        prev_dim = self.num_species + self.num_globals + 1  # inputs + time
        
        for dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
        self.output_layer = nn.Linear(prev_dim, self.num_species)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.sin(layer(x))
        return self.output_layer(x)

# =============================================================================
# ResNet Model
# =============================================================================

class ChemicalResNet(nn.Module):
    """
    ResNet for chemical kinetics prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]
        
        self.input_layer = nn.Linear(self.num_species + self.num_globals + 1, self.hidden_dims[0])
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dims[i+1], self.hidden_dims[i+1])
                )
            )
        
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.num_species)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.input_layer(x))
        
        for block in self.res_blocks:
            residual = x
            x = nn.functional.relu(block(x) + residual)
        
        return torch.clamp(self.output_layer(x), min=MIN_CONCENTRATION)

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
    enabled – compile it in default mode.
    """
    kind = config["model"]["type"].lower()
    if kind == "siren":
        model = SIREN(config)
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


def export_model(
    model: nn.Module,
    example_input: torch.Tensor,
    save_path: Path
):
    """
    Export model using torch.export with proper handling for compiled models.
    
    Args:
        model: Model to export (compiled or eager)
        example_input: Example input tensor
        save_path: Path to save exported model
    """
    logger = logging.getLogger(__name__)
    
    # Set to eval mode
    model.eval()
    
    # Check if model is compiled and get the original model if needed
    is_compiled = hasattr(model, '_orig_mod')
    if is_compiled:
        logger.info("Detected compiled model, extracting original model for export")
        original_model = model._orig_mod
    else:
        original_model = model
    
    with torch.no_grad():
        try:
            # Start export timer for large models
            export_start = time.time()
            logger.info("Starting model export...")
            
            # Use torch.export.export for modern export
            exported = torch.export.export(original_model, (example_input,))
            
            # Save the exported program
            torch.export.save(exported, str(save_path))
            
            export_time = time.time() - export_start
            logger.info(f"Model exported successfully to {save_path} in {export_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            logger.error("Common causes: dynamic shapes, unsupported ops, or compilation artifacts")
            logger.info("Try exporting the model before compilation if this persists")
            raise