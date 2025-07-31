#!/usr/bin/env python3

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.export import Dim


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
            beta = 0
        
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


class FiLMSIREN(nn.Module):
    """SIREN with FiLM conditioning - optimized for A100."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Extract dimensions
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.hidden_dims = config["model"]["hidden_dims"]

        # SIREN parameters
        self.omega_0 = config["model"].get("omega_0", 30.0)
        
        # Dropout
        dropout_rate = config["model"].get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # FiLM configuration
        film_config = config.get("film", {})
        self.use_film = film_config.get("enabled", True)

        # Input dimension
        input_dim = self.num_species + self.num_globals + 1

        # Build network layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList() if self.use_film else None

        prev_dim = input_dim
        condition_dim = self.num_species + self.num_globals

        # Get FiLM activations
        film_activations = film_config.get("activations", film_config.get("activation", "gelu"))
        if isinstance(film_activations, str):
            film_activations = [film_activations] * len(self.hidden_dims)

        for i, dim in enumerate(self.hidden_dims):
            # Main layer
            self.layers.append(nn.Linear(prev_dim, dim))

            # FiLM layer
            if self.use_film:
                layer_activation = film_activations[i] if isinstance(film_activations, list) else film_activations
                self.film_layers.append(
                    FiLMLayer(
                        condition_dim=condition_dim,
                        feature_dim=dim,
                        hidden_dims=film_config.get("hidden_dims", [128, 128]),
                        activation=layer_activation,
                        use_beta=True
                    )
                )

            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.num_species)

        # Initialize SIREN weights
        self._initialize_siren_weights()
    
    def _initialize_siren_weights(self):
        """Initialize weights following SIREN paper."""
        with torch.no_grad():
            # First layer
            if len(self.layers) > 0:
                fan_in = self.layers[0].in_features
                self.layers[0].weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            
            # Hidden layers
            for layer in self.layers[1:]:
                fan_in = layer.in_features
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
                layer.weight.uniform_(-bound, bound)
            
            # Output layer
            fan_in = self.output_layer.in_features
            bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - optimized."""
        # Extract initial conditions for FiLM
        initial_conditions = x[:, :-1]  # All but time
        
        # Process through layers
        h = x
        for i, layer in enumerate(self.layers):
            # Linear transformation
            h = layer(h)
            
            # Apply FiLM before activation
            if self.use_film and self.film_layers is not None:
                h = self.film_layers[i](h, initial_conditions)
            
            # SIREN activation
            h = torch.sin(self.omega_0 * h)
            
            # Dropout (except last layer)
            if self.dropout is not None and i < len(self.layers) - 1:
                h = self.dropout(h)
        
        # Output
        return self.output_layer(h)


class FiLMDeepONet(nn.Module):
    """Deep Operator Network with FiLM conditioning - optimized."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract dimensions
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])

        # Architecture parameters
        branch_layers = config["model"]["branch_layers"]
        trunk_layers = config["model"]["trunk_layers"]
        self.basis_dim = config["model"]["basis_dim"]
        self.activation = self._get_activation(config["model"].get("activation", "gelu"))
        self.output_scale = config["model"].get("output_scale", 1.0)
        
        # Dropout
        dropout_rate = config["model"].get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # FiLM configuration
        film_config = config.get("film", {})
        self.use_film = film_config.get("enabled", True)
        
        # Build networks
        self.branch_net = self._build_mlp_with_film(
            input_dim=self.num_species + self.num_globals,
            hidden_layers=branch_layers,
            output_dim=self.basis_dim * self.num_species,
            condition_dim=self.num_globals if self.use_film else None,
            film_config=film_config if self.use_film else None
        )
        
        self.trunk_net = self._build_mlp_with_film(
            input_dim=1,
            hidden_layers=trunk_layers,
            output_dim=self.basis_dim,
            condition_dim=self.num_globals if self.use_film else None,
            film_config=film_config if self.use_film else None,
            bias=True
        )
    
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
        """Build MLP with optional FiLM - optimized."""
        if self.use_film and condition_dim is not None and film_config is not None:
            # Build with FiLM
            layers = nn.ModuleList()
            film_layers = nn.ModuleList()

            # Get FiLM activations
            film_activations = film_config.get("activations", film_config.get("activation", "gelu"))
            if isinstance(film_activations, str):
                film_activations = [film_activations] * len(hidden_layers)

            prev_dim = input_dim
            for i, dim in enumerate(hidden_layers):
                layers.append(nn.Linear(prev_dim, dim, bias=bias))

                layer_activation = film_activations[i] if isinstance(film_activations, list) else film_activations
                film_layers.append(
                    FiLMLayer(
                        condition_dim=condition_dim,
                        feature_dim=dim,
                        hidden_dims=film_config.get("hidden_dims", [128, 128]),
                        activation=layer_activation,
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
                        
                        # Dropout (except last layer)
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
                
                # Dropout (except last layer)
                if self.dropout is not None and i < len(hidden_layers) - 1:
                    layers.append(self.dropout)
                    
                prev_dim = dim

            layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
            return nn.Sequential(*layers)
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = inputs.shape[0]

        # Split inputs
        branch_input = inputs[:, :self.num_species + self.num_globals]
        trunk_input = inputs[:, -1:]  # Time

        # Process through networks
        if self.use_film:
            # Isolate the global parameters for conditioning
            global_params_for_conditioning = inputs[:, self.num_species : self.num_species + self.num_globals]
            
            # Pass only globals as the condition
            branch_out = self.branch_net(branch_input, global_params_for_conditioning)
            trunk_out = self.trunk_net(trunk_input, global_params_for_conditioning)
        else:
            branch_out = self.branch_net(branch_input)
            trunk_out = self.trunk_net(trunk_input)

        # Reshape and combine efficiently
        branch_out = branch_out.view(batch_size, self.num_species, self.basis_dim)
        
        # Efficient matrix multiplication
        output = torch.bmm(branch_out, trunk_out.unsqueeze(2)).squeeze(2)

        # Optional output scaling
        if self.output_scale != 1.0:
            output = output * self.output_scale

        return output


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and compile model with validation."""
    model_type = config["model"]["type"].lower()
    prediction_mode = config.get("prediction", {}).get("mode", "absolute")
    
    # Validate model-mode compatibility
    if prediction_mode == "ratio" and model_type != "deeponet":
        raise ValueError(
            f"Prediction mode 'ratio' is only compatible with model type 'deeponet', "
            f"but '{model_type}' was specified. Either:\n"
            f"  1. Change model.type to 'deeponet' in your config, or\n"
            f"  2. Change prediction.mode to 'absolute'"
        )
    
    if model_type == "siren":
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

    if config["system"].get("dtype") == "float64":
        model = model.double()
        logger.info("Converted model to float64 precision")
        
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