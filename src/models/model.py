#!/usr/bin/env python3
"""
Model definitions for chemical kinetics neural networks.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, condition_dim: int, feature_dim: int, 
                 hidden_dims: List[int], activation: str = "gelu", use_beta: bool = True):
        super().__init__()
        
        self.use_beta = use_beta
        out_multiplier = 2 if use_beta else 1
        
        # Build FiLM MLP
        layers = []
        prev_dim = condition_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                self._get_activation(activation)
            ])
            prev_dim = dim
        
        # Output layer (2x feature_dim for gamma and beta)
        layers.append(nn.Linear(prev_dim, out_multiplier * feature_dim))
        
        self.film_net = nn.Sequential(*layers)
        self.feature_dim = feature_dim
    
    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU()
        }
        return activations.get(name.lower(), nn.GELU())
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation."""
        # Generate gamma and beta
        params = self.film_net(condition)
        if self.use_beta:
            gamma, beta = params.chunk(2, dim=-1)
        else:
            gamma = params
            beta = torch.zeros_like(gamma)
        
        # Reshape for broadcasting
        shape = [gamma.size(0)] + [1] * (features.dim() - 2) + [self.feature_dim]
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)
        
        # Apply modulation: gamma * features + beta
        return gamma * features + beta


class FiLMSIREN(nn.Module):
    """SIREN with FiLM conditioning for chemical kinetics."""
    def __init__(self, config: Dict[str, Any]):
                super().__init__()

                # Extract dimensions
                self.num_species = len(config["data"]["species_variables"])
                self.num_globals = len(config["data"]["global_variables"])
                self.hidden_dims = config["model"]["hidden_dims"]

                # SIREN parameters
                self.omega_0 = config["model"].get("omega_0", 30.0)

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

                for i, dim in enumerate(self.hidden_dims):
                    # Main layer
                    self.layers.append(nn.Linear(prev_dim, dim))

                    # FiLM layer with SIREN-compatible initialization
                    if self.use_film:
                        film_layer = FiLMLayer(
                            condition_dim=condition_dim,
                            feature_dim=dim,
                            hidden_dims=film_config.get("hidden_dims", [128, 128]),
                            activation=film_config.get("activation", "gelu")
                        )

                        with torch.no_grad():
                            final_layer = film_layer.film_net[-1]
                            final_layer.weight.data.zero_()
                            # Set bias for gamma part to 1
                            final_layer.bias.data[:dim] = 1.0
                            # Set bias for beta part to 0
                            final_layer.bias.data[dim:] = 0.0

                        self.film_layers.append(film_layer)

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
                nn.init.uniform_(self.layers[0].weight, -1.0 / fan_in, 1.0 / fan_in)
            
            # Hidden layers
            for layer in self.layers[1:]:
                fan_in = layer.in_features
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
                nn.init.uniform_(layer.weight, -bound, bound)
            
            # Output layer
            fan_in = self.output_layer.in_features
            bound = math.sqrt(6.0 / fan_in) / self.omega_0
            nn.init.uniform_(self.output_layer.weight, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning."""
        # Extract components
        initial_conditions = x[:, :-1]  # All but time
        
        # Process through layers
        h = x
        for i, layer in enumerate(self.layers):
            # Linear transformation
            h = layer(h)
            
            # Apply FiLM before activation
            if self.use_film and self.film_layers is not None:
                h = self.film_layers[i](h, initial_conditions)
            
            # SIREN activation (sine)
            h = torch.sin(self.omega_0 * h)
        
        # Output
        output = self.output_layer(h)
        
        return output


class FiLMDeepONet(nn.Module):
    """Deep Operator Network with FiLM conditioning (no bias for delta mode)."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract dimensions
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.prediction_mode = config.get("prediction", {}).get("mode", "absolute")

        # Architecture parameters
        branch_layers = config["model"]["branch_layers"]
        trunk_layers = config["model"]["trunk_layers"]
        self.basis_dim = config["model"]["basis_dim"]
        self.activation = self._get_activation(config["model"].get("activation", "gelu"))
        self.output_scale = config["model"].get("output_scale", 1.0)
        
        # FiLM configuration
        film_config = config.get("film", {})
        self.use_film = film_config.get("enabled", True)
        
        # Determine bias based on prediction mode (critical for delta mode to ensure zero at t=0)
        prediction_mode = config.get("prediction", {}).get("mode", "absolute")
        bias = (prediction_mode != "delta")
        
        # Build branch network (processes initial conditions)
        self.branch_net = self._build_mlp_with_film(
            input_dim=self.num_species + self.num_globals,
            hidden_layers=branch_layers,
            output_dim=self.basis_dim * self.num_species,
            condition_dim=self.num_species + self.num_globals if self.use_film else None,
            film_config=film_config if self.use_film else None
        )
        
        # Build trunk network (processes time)
        self.trunk_net = self._build_mlp_with_film(
            input_dim=1,
            hidden_layers=trunk_layers,
            output_dim=self.basis_dim,
            condition_dim=self.num_species + self.num_globals if self.use_film else None,
            film_config=film_config if self.use_film else None,
            bias=bias  # No bias in trunk for delta mode
        )
    
    def _get_activation(self, name: str):
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU()
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _build_mlp_with_film(self, input_dim: int, hidden_layers: List[int],
                                output_dim: int, condition_dim: Optional[int] = None,
                                film_config: Optional[Dict] = None, bias: bool = True) -> nn.Module:
            """
            Build an MLP with optional FiLM layers.

            Args:
                input_dim: Input dimension for the MLP.
                hidden_layers: List of hidden layer dimensions.
                output_dim: Output dimension for the MLP.
                condition_dim: Dimension of the conditioning vector for FiLM.
                film_config: Configuration dictionary for FiLM layers.
                bias: If True, adds a learnable bias to the linear layers. This is set
                    to False for the trunk network in delta mode to ensure output is
                    zero at t=0.
            """

            if self.use_film and condition_dim is not None and film_config is not None:
                # Build with FiLM
                layers = nn.ModuleList()
                film_layers = nn.ModuleList()

                prev_dim = input_dim
                for dim in hidden_layers:
                    layers.append(nn.Linear(prev_dim, dim, bias=bias))

                    film_layers.append(
                        FiLMLayer(
                            condition_dim=condition_dim,
                            feature_dim=dim,
                            hidden_dims=film_config.get("hidden_dims", [128, 128]),
                            activation=film_config.get("activation", "gelu"),
                            use_beta=True  # Beta is always needed for effective FiLM.
                        )
                    )
                    prev_dim = dim

                output_layer = nn.Linear(prev_dim, output_dim, bias=bias)

                class MLPWithFiLM(nn.Module):
                    def __init__(self, layers, film_layers, output_layer, activation):
                        super().__init__()
                        self.layers = layers
                        self.film_layers = film_layers
                        self.output_layer = output_layer
                        self.activation = activation

                    def forward(self, x, condition):
                        h = x
                        for layer, film_layer in zip(self.layers, self.film_layers):
                            h = layer(h)
                            h = film_layer(h, condition)
                            h = self.activation(h)
                        return self.output_layer(h)

                return MLPWithFiLM(layers, film_layers, output_layer, self.activation)

            else:
                # Build standard MLP
                layers = []
                prev_dim = input_dim

                for dim in hidden_layers:
                    layers.extend([nn.Linear(prev_dim, dim, bias=bias), self.activation])
                    prev_dim = dim

                layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
                return nn.Sequential(*layers)
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning."""
        # Split inputs
        branch_input = inputs[:, :self.num_species + self.num_globals]
        trunk_input = inputs[:, -1:]  # Time

        # Use branch input as condition for FiLM
        condition = branch_input if self.use_film else None

        # Process through networks
        if self.use_film:
            branch_out = self.branch_net(branch_input, condition)
            trunk_out = self.trunk_net(trunk_input, condition)
        else:
            branch_out = self.branch_net(branch_input)
            trunk_out = self.trunk_net(trunk_input)

        # Reshape branch output
        branch_out = branch_out.view(-1, self.num_species, self.basis_dim)

        # Combine with dot product (no bias!)
        if self.prediction_mode == "delta":
            trunk_out = trunk_out * trunk_input

        output = torch.einsum("bni,bi->bn", branch_out, trunk_out)

        # Optional output scaling
        if self.output_scale != 1.0:
            output = output * self.output_scale

        return output


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_type = config["model"]["type"].lower()
    
    prediction_mode = config.get("prediction", {}).get("mode", "absolute")
    if prediction_mode == "delta":
        norm_config = config["normalization"]
        default_method = norm_config["default_method"]
        methods = norm_config.get("methods", {})

        # 1. FATAL: Check for log-scaling on species variables in delta mode.
        # This is invalid because delta is an additive concept, while log-scaling is multiplicative.
        for var in config["data"]["species_variables"]:
            method = methods.get(var, default_method)
            if "log" in method:
                raise ValueError(
                    f"Delta prediction mode is incompatible with log-based normalization "
                    f"for species variables. Variable '{var}' uses '{method}'."
                )

        # 2. Validate time variable normalization for delta mode.
        time_var = config["data"]["time_variable"]
        time_method = methods.get(time_var, default_method)

        # FATAL: Standard scaling breaks the guarantee that pred(t=0) is zero.
        if "standard" in time_method:
            raise ValueError(
                f"Delta prediction mode requires a min-max style normalizer for the time "
                f"variable ('{time_var}') to ensure the prediction at t=0 is zero. "
                f"The current method '{time_method}' is not supported."
            )
        
        # WARNING: Log-scaling time is physically unintuitive for a delta model.
        if "log-min-max" in time_method:
            logging.warning(
                f"Using log-scaling ('{time_method}') for the time variable ('{time_var}') "
                f"in delta mode is not recommended. It forces the model to learn a "
                f"physically unintuitive mapping and may lead to poor performance."
            )

    if model_type == "siren":
        if prediction_mode == "delta":
            raise ValueError("SIREN model not supported in delta prediction mode due to activation incompatibility.")
        model = FiLMSIREN(config)
    elif model_type == "deeponet":
        model = FiLMDeepONet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Compile model if enabled and supported
    if config["system"].get("use_torch_compile", False) and device.type == "cuda":
        compile_mode = config["system"].get("compile_mode", "default")
        logging.info(f"Compiling model with mode='{compile_mode}'...")
        
        try:
            model = torch.compile(model, mode=compile_mode)
            logging.info("Model compilation successful")
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")
    
    return model


def export_model(model: nn.Module, example_input: torch.Tensor, save_path: Path):
    """Export model using torch.export."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    # Handle compiled models
    if hasattr(model, '_orig_mod'):
        logger.info("Extracting original model from compiled wrapper")
        model = model._orig_mod
    
    with torch.no_grad():
        try:
            # Export the model
            exported_program = torch.export.export(model, (example_input,))
            torch.export.save(exported_program, str(save_path))
            logger.info(f"Model exported to {save_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise