#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation following Goswami et al. (2023).
CORRECTED: Fixed syntax errors, proper indentation, consistent initialization
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_glorot_normal(m: nn.Module):
    """Initialize weights with Glorot Normal (Xavier Normal) distribution."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction with optimized forward pass."""

    def __init__(self, num_species: int, latent_dim: int,
                 encoder_hidden_layers: List[int], decoder_hidden_layers: List[int]):
        super().__init__()
        self.num_species = num_species
        self.latent_dim = latent_dim

        # Build encoder as sequential for efficiency
        encoder_modules = []
        in_features = num_species
        for hidden_units in encoder_hidden_layers:
            encoder_modules.extend([
                nn.Linear(in_features, hidden_units),
                nn.LeakyReLU(0.01, inplace=True)
            ])
            in_features = hidden_units
        encoder_modules.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder as sequential for efficiency
        decoder_modules = []
        in_features = latent_dim
        for hidden_units in decoder_hidden_layers:
            decoder_modules.extend([
                nn.Linear(in_features, hidden_units),
                nn.LeakyReLU(0.01, inplace=True)
            ])
            in_features = hidden_units
        decoder_modules.append(nn.Linear(in_features, num_species))
        self.decoder = nn.Sequential(*decoder_modules)

        # Apply weight initialization
        self.apply(init_weights_glorot_normal)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DeepONet(nn.Module):
    """DeepONet with dynamically built branch and trunk networks."""

    def __init__(self, latent_dim: int, num_globals: int, p: int,
                 branch_hidden_layers: List[int], trunk_hidden_layers: List[int],
                 trunk_basis: str = "linear"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_globals = num_globals
        self.p = p
        self.trunk_basis = trunk_basis

        # Branch network with LayerNorm
        self.branch_layers = nn.ModuleList()
        in_features = latent_dim + num_globals

        for hidden_units in branch_hidden_layers:
            self.branch_layers.append(nn.Linear(in_features, hidden_units))
            self.branch_layers.append(nn.LayerNorm(hidden_units))
            self.branch_layers.append(nn.LeakyReLU(0.01, inplace=True))
            in_features = hidden_units

        self.branch_layers.append(nn.Linear(in_features, latent_dim * p))

        # Trunk network (no normalization as per paper)
        self.trunk_layers = nn.ModuleList()
        in_features = 1
        for hidden_units in trunk_hidden_layers:
            self.trunk_layers.append(nn.Linear(in_features, hidden_units))
            self.trunk_layers.append(nn.LeakyReLU(0.01, inplace=True))
            in_features = hidden_units
        self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

        # Single consistent weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.branch_layers:
            y = layer(y)
        return y

    def forward_trunk(self, t: torch.Tensor) -> torch.Tensor:
        y = t
        for layer in self.trunk_layers:
            y = layer(y)
        return y

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both prediction and trunk outputs."""
        B = branch_input.size(0)
        M = trunk_input.size(0)

        # Compute branch and trunk outputs
        branch_out = self.forward_branch(branch_input).view(B, self.latent_dim, self.p)
        trunk_raw = self.forward_trunk(trunk_input).view(M, self.latent_dim, self.p)

        # Apply basis transformation if needed
        if self.trunk_basis == "softmax":
            trunk_out = F.softmax(trunk_raw, dim=-1)
        else:
            trunk_out = trunk_raw

        # Tensor contraction
        z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)

        return z_pred, trunk_out


class AEDeepONet(nn.Module):
    """
    Combined Autoencoder-DeepONet model.
    CORRECTED: Fixed indentation, consistent initialization, proper PoU handling
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Parse configuration
        data_cfg = config["data"]
        model_cfg = config["model"]

        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.latent_dim = model_cfg["latent_dim"]
        self.p = model_cfg.get("p", 10)
        self.trunk_basis = model_cfg.get("trunk_basis", "linear")

        # PoU only makes sense for linear basis (softmax already sums to 1)
        self.use_pou = model_cfg.get("use_pou", False) and self.trunk_basis == "linear"
        if model_cfg.get("use_pou", False) and self.trunk_basis == "softmax":
            self.logger.info("PoU regularization disabled for softmax basis (already sums to 1)")

        self.pou_weight = config["training"].get("pou_weight", 0.01)

        # Decoder output control
        self.decoder_output_mode = str(model_cfg.get("decoder_output_mode", "linear")).lower()
        self.output_clamp = model_cfg.get("output_clamp", None)

        # Validation
        norm_cfg = config.get("normalization", {})
        default_species_method = str(norm_cfg.get("default_method", "")).lower()
        valid_01 = default_species_method in {"min-max", "log-min-max"}
        if self.decoder_output_mode == "sigmoid01" and not valid_01:
            self.logger.warning(
                "decoder_output_mode='sigmoid01' requires species to be min-max or log-min-max normalized; "
                f"found default_method='{default_species_method}'. Forcing 'linear'."
            )
            self.decoder_output_mode = "linear"

        # Build submodules
        ae_encoder_layers = model_cfg["ae_encoder_layers"]
        ae_decoder_layers = model_cfg["ae_decoder_layers"]
        branch_layers = model_cfg["branch_layers"]
        trunk_layers = model_cfg["trunk_layers"]

        self.autoencoder = Autoencoder(
            num_species=self.num_species,
            latent_dim=self.latent_dim,
            encoder_hidden_layers=ae_encoder_layers,
            decoder_hidden_layers=ae_decoder_layers,
        )

        self.deeponet = DeepONet(
            latent_dim=self.latent_dim,
            num_globals=self.num_globals,
            p=self.p,
            branch_hidden_layers=branch_layers,
            trunk_hidden_layers=trunk_layers,
            trunk_basis=self.trunk_basis,
        )

        # Cache device
        self.register_buffer('_device_tensor', torch.zeros(1))

        # Initialize clamping tensors
        self.clamp_min = torch.tensor([], dtype=torch.float32)
        self.clamp_max = torch.tensor([], dtype=torch.float32)
        self._init_normalized_clamp()

        # Validation warnings
        if self.num_species != 12:
            self.logger.warning(
                f"Paper uses 12 species for syngas, but found {self.num_species}. "
                f"Species: {data_cfg['species_variables']}"
            )

        expected_globals = data_cfg.get("expected_globals", ["P", "T"])
        if data_cfg["global_variables"] != expected_globals:
            raise ValueError(
                f"Global variables mismatch: got {data_cfg['global_variables']}, expected {expected_globals}. "
                "Note: this implementation intentionally uses [P, T] instead of [φ₀, T₀]."
            )

        # Default trunk times
        trunk_times = model_cfg.get("trunk_times", [0.25, 0.5, 0.75, 1.0])
        if not all(0.0 <= float(t) <= 1.0 for t in trunk_times):
            raise ValueError(f"trunk_times must be in [0,1], got {trunk_times}")
        self.register_buffer(
            "default_trunk_times",
            torch.tensor(trunk_times, dtype=torch.float32).view(-1, 1),
            persistent=False
        )

        # Tracking lists
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

    @property
    def device(self):
        """Get the device of the model."""
        return self._device_tensor.device

    def compute_deeponet_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            trunk_times: Optional[torch.Tensor] = None,
            pou_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the full DeepONet loss."""
        # Forward in latent space
        z_pred, aux = self(
            inputs,
            decode=False,
            return_trunk_outputs=True,
            trunk_times=trunk_times
        )

        # Shape validation
        if z_pred.shape != targets.shape:
            raise ValueError(
                f"MSE shape mismatch: pred {tuple(z_pred.shape)} vs target {tuple(targets.shape)}"
            )

        # Compute losses
        mse_loss = F.mse_loss(z_pred, targets)

        trunk_outputs = aux.get("trunk_outputs", None)
        if self.use_pou and trunk_outputs is not None and pou_weight > 0:
            pou_loss = self.pou_regularization(trunk_outputs)
        else:
            pou_loss = torch.zeros((), device=self.device, dtype=z_pred.dtype)

        total = mse_loss + float(pou_weight) * pou_loss
        return total, {"mse": mse_loss, "pou": pou_loss}

    def _init_normalized_clamp(self):
        """Initialize clamping bounds in NORMALIZED space only."""
        if self.output_clamp is None:
            return

        if isinstance(self.output_clamp, (list, tuple)) and len(self.output_clamp) == 2:
            lo, hi = float(self.output_clamp[0]), float(self.output_clamp[1])
            if lo > hi:
                raise ValueError(f"Clamp lower bound > upper bound: {self.output_clamp}")
            self.clamp_min = torch.full((1, self.num_species), lo, dtype=torch.float32)
            self.clamp_max = torch.full((1, self.num_species), hi, dtype=torch.float32)
        elif isinstance(self.output_clamp, dict) and "min" in self.output_clamp and "max" in self.output_clamp:
            lo = self.output_clamp["min"]
            hi = self.output_clamp["max"]

            if isinstance(lo, (int, float)):
                self.clamp_min = torch.full((1, self.num_species), float(lo), dtype=torch.float32)
            elif isinstance(lo, (list, tuple)) and len(lo) == self.num_species:
                self.clamp_min = torch.tensor(lo, dtype=torch.float32).view(1, -1)
            else:
                raise ValueError("output_clamp['min'] must be scalar or list of length num_species")

            if isinstance(hi, (int, float)):
                self.clamp_max = torch.full((1, self.num_species), float(hi), dtype=torch.float32)
            elif isinstance(hi, (list, tuple)) and len(hi) == self.num_species:
                self.clamp_max = torch.tensor(hi, dtype=torch.float32).view(1, -1)
            else:
                raise ValueError("output_clamp['max'] must be scalar or list of length num_species")

            if torch.any(self.clamp_min > self.clamp_max):
                raise ValueError("Some per-species lower bounds exceed upper bounds")
        else:
            raise ValueError("output_clamp must be None, [min, max], or {'min': ..., 'max': ...}")

    def set_normalized_clamp(self, clamp_min, clamp_max):
        """Set clamping bounds in NORMALIZED space."""

        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=torch.float32)
            elif isinstance(x, (int, float)):
                return torch.full((1, self.num_species), float(x), dtype=torch.float32)
            elif isinstance(x, (list, tuple)):
                if len(x) == self.num_species:
                    return torch.tensor(x, dtype=torch.float32).view(1, -1)
                else:
                    raise ValueError(f"Clamp bounds must have length {self.num_species}")
            else:
                raise ValueError("Unsupported clamp bound type")

        self.clamp_min = _to_tensor(clamp_min)
        self.clamp_max = _to_tensor(clamp_max)

        if torch.any(self.clamp_min > self.clamp_max):
            raise ValueError("Clamp min exceeds max for at least one species")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def pou_regularization(self, trunk_outputs: Optional[torch.Tensor]) -> torch.Tensor:
        """Partition-of-Unity (PoU) regularization."""
        if not self.use_pou or trunk_outputs is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Sum over basis functions
        trunk_sum = trunk_outputs.sum(dim=-1)  # [M, L]

        # Penalize deviation from 1
        pou_loss = ((trunk_sum - 1.0) ** 2).mean()

        return pou_loss

    def forward(
            self,
            inputs: torch.Tensor,
            decode: bool = True,
            return_trunk_outputs: bool = False,
            trunk_times: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through AE-DeepONet.

        Args:
            inputs: [B, latent_dim + num_globals] = [z0, globals]
            decode: if True, decode latent z(t) to species space
            return_trunk_outputs: if True, include trunk basis in aux
            trunk_times: Optional custom trunk times [M] or [M, 1]

        Returns:
            predictions, aux_dict
        """
        expected_dim = self.latent_dim + self.num_globals
        if inputs.size(1) != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim} ([latent, globals]), got {inputs.size(1)}"
            )

        # Handle trunk times with proper shape guarantee
        if trunk_times is None:
            trunk_in = self.default_trunk_times.to(inputs.device, inputs.dtype)  # [M, 1]
        else:
            if trunk_times.dim() == 1:
                trunk_times = trunk_times.unsqueeze(-1)  # [M] -> [M, 1]
            trunk_in = trunk_times.to(inputs.device, inputs.dtype)

        # Single forward pass through DeepONet
        z_pred, trunk_out = self.deeponet(inputs, trunk_in)  # [B, M, L], [M, L, p]

        aux = {"z_pred": z_pred}
        if return_trunk_outputs or self.use_pou:
            aux["trunk_outputs"] = trunk_out

        if not decode:
            return z_pred, aux

        # Decode latent to species space
        B, M, L = z_pred.shape
        y_flat = self.decode(z_pred.reshape(B * M, L))  # [B*M, S]

        # Apply decoder output mode
        if self.decoder_output_mode == "sigmoid01":
            y_flat = torch.sigmoid(y_flat)
        elif self.decoder_output_mode != "linear":
            raise ValueError(f"Unsupported decoder_output_mode: {self.decoder_output_mode}")

        # Apply clamping in normalized space if configured
        if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
            y_flat = torch.max(y_flat, self.clamp_min.to(y_flat.device, y_flat.dtype))
            y_flat = torch.min(y_flat, self.clamp_max.to(y_flat.device, y_flat.dtype))

        y_pred = y_flat.view(B, M, self.num_species)
        return y_pred, aux

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Plain MSE loss for AE pretraining."""
        return F.mse_loss(y_pred, y_true)

    def ae_parameters(self):
        """Return autoencoder parameters."""
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        """Return DeepONet parameters."""
        return self.deeponet.parameters()


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Factory to create and place the AE-DeepONet model."""
    if device.type != "cuda":
        raise RuntimeError("AE-DeepONet requires a CUDA device.")

    model = AEDeepONet(config)

    # Set dtype (consider keeping fp32 and letting AMP handle mixed precision)
    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16":
        model = model.bfloat16()

    model = model.to(device)

    # Optional compilation
    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    return model