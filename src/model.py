#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation following Goswami et al. (2023).
This version allows network dimensions to be specified via a configuration file.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_glorot_normal(m: nn.Module):
    """
    Initialize weights with Glorot Normal (Xavier Normal) distribution.
    This initialization is applied recursively to all linear layers in a model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction.
    The architecture (number and size of hidden layers) is dynamically
    built based on the provided configuration.
    """

    def __init__(self, num_species: int, latent_dim: int,
                 encoder_hidden_layers: List[int], decoder_hidden_layers: List[int]):
        """
        Initializes the Autoencoder.
        Args:
            num_species: The dimensionality of the input and output data.
            latent_dim: The dimensionality of the bottleneck layer.
            encoder_hidden_layers: A list of integers defining the size of each hidden layer in the encoder.
            decoder_hidden_layers: A list of integers defining the size of each hidden layer in the decoder.
        """
        super().__init__()
        self.num_species = num_species
        self.latent_dim = latent_dim

        # --- Dynamically build encoder layers ---
        self.encoder_layers = nn.ModuleList()
        in_features = num_species
        # Loop through the hidden layer definitions from the config
        for hidden_units in encoder_hidden_layers:
            self.encoder_layers.append(nn.Linear(in_features, hidden_units))
            in_features = hidden_units
        # Add the final layer that maps to the latent dimension
        self.encoder_layers.append(nn.Linear(in_features, latent_dim))

        # --- Dynamically build decoder layers ---
        self.decoder_layers = nn.ModuleList()
        in_features = latent_dim
        # Loop through the hidden layer definitions from the config
        for hidden_units in decoder_hidden_layers:
            self.decoder_layers.append(nn.Linear(in_features, hidden_units))
            in_features = hidden_units
        # Add the final layer that reconstructs the original species data
        self.decoder_layers.append(nn.Linear(in_features, num_species))

        # Apply Glorot Normal initialization to all created layers
        self.apply(init_weights_glorot_normal)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor from the physical space to the latent space.
        """
        y = x
        for i, layer in enumerate(self.encoder_layers):
            y = layer(y)
            # LeakyReLU is applied to all layers except the final output layer
            if i < len(self.encoder_layers) - 1:
                y = F.leaky_relu(y, negative_slope=0.01)
        return y

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space representation back to the physical space.
        """
        y = z
        for i, layer in enumerate(self.decoder_layers):
            y = layer(y)
            # LeakyReLU is applied to all layers except the final output layer
            if i < len(self.decoder_layers) - 1:
                y = F.leaky_relu(y, negative_slope=0.01)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a full forward pass: encode -> decode.
        """
        z = self.encode(x)
        return self.decode(z)


class DeepONet(nn.Module):
    """
    DeepONet implementation following the paper's architecture.
    The branch and trunk networks are built dynamically based on the provided configuration.
    """

    def __init__(self, latent_dim: int, num_globals: int, p: int,
                 branch_hidden_layers: List[int], trunk_hidden_layers: List[int]):
        """
        Initializes the DeepONet.
        Args:
            latent_dim: The dimensionality of the latent space.
            num_globals: The number of global parameters (e.g., Pressure, Temperature).
            p: The number of basis functions per latent dimension.
            branch_hidden_layers: A list defining the hidden layer sizes for the branch network.
            trunk_hidden_layers: A list defining the hidden layer sizes for the trunk network.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_globals = num_globals
        self.p = p

        # --- Dynamically build branch network layers ---
        self.branch_layers = nn.ModuleList()
        in_features = latent_dim + num_globals
        for hidden_units in branch_hidden_layers:
            self.branch_layers.append(nn.Linear(in_features, hidden_units))
            self.branch_layers.append(nn.BatchNorm1d(hidden_units))
            self.branch_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units
        # Add the final output layer for the branch network
        self.branch_layers.append(nn.Linear(in_features, latent_dim * p))

        # --- Dynamically build trunk network layers ---
        self.trunk_layers = nn.ModuleList()
        in_features = 1  # Trunk network input is always 1D (time)
        for hidden_units in trunk_hidden_layers:
            self.trunk_layers.append(nn.Linear(in_features, hidden_units))
            self.trunk_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units
        # Add the final output layer for the trunk network
        self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

        # Apply Glorot Normal initialization to all Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the branch network."""
        y = x
        for layer in self.branch_layers:
            y = layer(y)
        return y

    def forward_trunk(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the trunk network."""
        y = t
        for layer in self.trunk_layers:
            y = layer(y)
        return y

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Performs the full DeepONet forward pass, combining branch and trunk outputs.

        Args:
            branch_input: Tensor of shape [B, latent_dim + num_globals] containing initial latent state and global parameters.
            trunk_input: Tensor of shape [M, 1] containing normalized time points.

        Returns:
            A tensor of shape [B, M, latent_dim] representing the predicted latent state evolution.
        """
        B = branch_input.size(0)
        M = trunk_input.size(0)

        # Get branch network output and reshape for dot product
        branch_out = self.forward_branch(branch_input)
        branch_out = branch_out.view(B, self.latent_dim, self.p)

        # Get trunk network output and reshape for dot product
        trunk_out = self.forward_trunk(trunk_input)
        trunk_out = trunk_out.view(M, self.latent_dim, self.p)

        # Compute the inner product between branch and trunk outputs
        # This is the core operation of the DeepONet
        output = torch.einsum('blp,mlp->bml', branch_out, trunk_out)  # [B, M, latent_dim]

        return output


class AEDeepONet(nn.Module):
    """
    The combined Autoencoder-DeepONet model.
    This class orchestrates the autoencoder and DeepONet components,
    reading their architectural specifications from a configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Extract configuration dictionaries
        data_cfg = config["data"]
        model_cfg = config["model"]

        # Set model parameters from config
        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.latent_dim = model_cfg["latent_dim"]
        self.p = model_cfg.get("p", 10)
        self.use_pou = model_cfg.get("use_pou", False)

        # Extract layer configurations for sub-models
        ae_encoder_layers = model_cfg["ae_encoder_layers"]
        ae_decoder_layers = model_cfg["ae_decoder_layers"]
        branch_layers = model_cfg["branch_layers"]
        trunk_layers = model_cfg["trunk_layers"]

        # --- Build model components using the specified architectures ---
        self.autoencoder = Autoencoder(
            num_species=self.num_species,
            latent_dim=self.latent_dim,
            encoder_hidden_layers=ae_encoder_layers,
            decoder_hidden_layers=ae_decoder_layers
        )
        self.deeponet = DeepONet(
            latent_dim=self.latent_dim,
            num_globals=self.num_globals,
            p=self.p,
            branch_hidden_layers=branch_layers,
            trunk_hidden_layers=trunk_layers
        )

        # --- Validation Checks ---
        if self.num_species != 12:
            self.logger.warning(
                f"Paper uses 12 species for syngas, but found {self.num_species}. "
                f"Species: {data_cfg['species_variables']}"
            )

        expected_globals = data_cfg.get("expected_globals", ["P", "T"])
        if data_cfg["global_variables"] != expected_globals:
            raise ValueError(
                f"Global variables mismatch: got {data_cfg['global_variables']}, "
                f"expected {expected_globals}."
            )

        # Register trunk evaluation times as a buffer (part of the model's state)
        trunk_times = model_cfg.get("trunk_times", [0.25, 0.5, 0.75, 1.0])
        if not all(0.0 <= t <= 1.0 for t in trunk_times):
            raise ValueError(f"trunk_times must be in [0,1], got {trunk_times}")

        self.register_buffer(
            "trunk_times",
            torch.tensor(trunk_times, dtype=torch.float32).view(-1, 1)
        )

        # Log model configuration details
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(
            f"AE-DeepONet initialized with {n_params:,} total parameters. "
            f"latent_dim={self.latent_dim}, p={self.p}, use_pou={self.use_pou}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """A convenience method to access the autoencoder's encoder."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """A convenience method to access the autoencoder's decoder."""
        return self.autoencoder.decode(z)

    def forward(
            self,
            inputs: torch.Tensor,
            decode: bool = True,
            return_trunk_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs the full forward pass through the combined AE-DeepONet model.

        Args:
            inputs: A tensor of shape [B, latent_dim + num_globals] containing the initial latent state (z0) and global parameters.
            decode: If True, the latent space predictions are decoded back to the physical species space.
            return_trunk_outputs: If True, includes trunk network outputs in the auxiliary dictionary (for PoU regularization).

        Returns:
            A tuple containing:
            - predictions: The final model output. Shape is [B, M, num_species] if decoded, otherwise [B, M, latent_dim].
            - aux_dict: A dictionary with auxiliary outputs like latent predictions ('z_pred').
        """
        # Validate input tensor dimension
        expected_dim = self.latent_dim + self.num_globals
        if inputs.size(1) != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim} ([z0, P, T]), but got {inputs.size(1)}"
            )

        # Get the pre-defined trunk time points
        trunk_in = self.trunk_times.to(inputs.device, inputs.dtype)  # [M, 1]

        # Evolve the latent state through the DeepONet
        z_pred = self.deeponet(inputs, trunk_in)  # [B, M, latent_dim]

        aux = {"z_pred": z_pred}

        # Optionally get trunk outputs for regularization
        if return_trunk_outputs:
            trunk_out = self.deeponet.forward_trunk(trunk_in)
            trunk_out = trunk_out.view(-1, self.latent_dim, self.p)
            aux["trunk_outputs"] = trunk_out

        # If decoding is not required, return the latent predictions
        if not decode:
            return z_pred, aux

        # Decode the latent predictions back to the physical space
        B, M, _ = z_pred.shape
        z_flat = z_pred.reshape(B * M, self.latent_dim)
        y_flat = self.decode(z_flat)
        y_pred = y_flat.view(B, M, self.num_species)

        return y_pred, aux

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error loss between predictions and true values.
        """
        return F.mse_loss(y_pred, y_true)

    def pou_regularization(self, trunk_outputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the Partition of Unity (PoU) regularization loss.
        This encourages the sum of trunk basis functions to be close to 1.
        """
        if not self.use_pou:
            return torch.tensor(0.0, device=trunk_outputs.device)

        # The sum of basis functions for each time point and latent dim should be 1
        trunk_sum = trunk_outputs.sum(dim=-1)  # [M, latent_dim]
        pou_loss = ((trunk_sum - 1.0) ** 2).mean()

        return pou_loss

    def ae_parameters(self):
        """Returns an iterator over the autoencoder's parameters for optimization."""
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        """Returns an iterator over the DeepONet's parameters for optimization."""
        return self.deeponet.parameters()


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory function to create, configure, and prepare the AE-DeepONet model.

    Args:
        config: The main configuration dictionary.
        device: The torch.device to move the model to.

    Returns:
        The fully configured and device-placed AE-DeepONet model.
    """
    if device.type != "cuda":
        raise RuntimeError("AE-DeepONet requires a CUDA device.")

    model = AEDeepONet(config)

    # Set the model's data type (e.g., float32, float16)
    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16":
        model = model.bfloat16()

    # Move model to the specified device
    model = model.to(device)

    # Optionally compile the model using torch.compile for performance (PyTorch 2.0+)
    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    # Optionally export the model for deployment (PyTorch 2.1+)
    if config.get("system", {}).get("use_torch_export", False):
        try:
            batch_size = 1
            example_input = torch.randn(
                batch_size,
                model.latent_dim + model.num_globals,
                device=device,
                dtype=torch.float32
            )
            exported_program = torch.export.export(
                model, args=(example_input,), kwargs={"decode": True, "return_trunk_outputs": False}
            )

            # Save the exported model artifact
            export_dir = Path(config["paths"].get("model_save_dir", "models"))
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / "ae_deeponet_exported.pt2"
            torch.export.save(exported_program, str(export_path))
            logging.getLogger(__name__).info(f"Model exported for deployment to {export_path}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Model export failed: {e}")

    return model