#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation following Goswami et al. (2023).

Key implementation notes:
- We use [P, T] as global variables (pressure, temperature) instead of [φ₀, T₀]
  (equivalence ratio, initial temperature) as in the paper. This is intentional
  as our data uses pressure/temperature conditions rather than equivalence ratio.
- Clamping is ONLY performed in normalized space. Physical space clamping has been
  removed to avoid confusion and ensure consistency.
- PoU regularization is properly integrated into the loss function.
- BatchNorm is handled flexibly to support batch_size=1.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================
# Utilities
# =====================================================================================

def init_weights_glorot_normal(m: nn.Module):
    """
    Initialize weights with Glorot Normal (Xavier Normal) distribution.
    Applied recursively to all Linear layers in a model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =====================================================================================
# Autoencoder
# =====================================================================================

class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction.
    The architecture (number and size of hidden layers) is built from the config.
    """

    def __init__(self, num_species: int, latent_dim: int,
                 encoder_hidden_layers: List[int], decoder_hidden_layers: List[int]):
        super().__init__()
        self.num_species = num_species
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_layers = nn.ModuleList()
        in_features = num_species
        for hidden_units in encoder_hidden_layers:
            self.encoder_layers.append(nn.Linear(in_features, hidden_units))
            in_features = hidden_units
        self.encoder_layers.append(nn.Linear(in_features, latent_dim))

        # Decoder
        self.decoder_layers = nn.ModuleList()
        in_features = latent_dim
        for hidden_units in decoder_hidden_layers:
            self.decoder_layers.append(nn.Linear(in_features, hidden_units))
            in_features = hidden_units
        self.decoder_layers.append(nn.Linear(in_features, num_species))

        self.apply(init_weights_glorot_normal)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for i, layer in enumerate(self.encoder_layers):
            y = layer(y)
            if i < len(self.encoder_layers) - 1:
                y = F.leaky_relu(y, negative_slope=0.01)
        return y

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = z
        for i, layer in enumerate(self.decoder_layers):
            y = layer(y)
            if i < len(self.decoder_layers) - 1:
                y = F.leaky_relu(y, negative_slope=0.01)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# =====================================================================================
# DeepONet
# =====================================================================================

class FlexibleBatchNorm1d(nn.Module):
    """
    BatchNorm1d wrapper that handles batch_size=1 gracefully.
    Falls back to identity when batch_size=1 during training.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.size(0) == 1 and self.training:
            # Skip batch norm for single sample during training
            return x
        else:
            return self.bn(x)


class DeepONet(nn.Module):
    """
    DeepONet with dynamically built branch and trunk networks.
    - Branch input: [z0, globals] with dimension (latent_dim + num_globals)
    - Trunk input: normalized time t in [0, 1], dimension 1
    - Output: latent trajectory z(t) with shape [B, M, latent_dim]

    Matches paper implementation with BatchNorm in branch network.
    """

    def __init__(self, latent_dim: int, num_globals: int, p: int,
                 branch_hidden_layers: List[int], trunk_hidden_layers: List[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_globals = num_globals
        self.p = p

        # Branch network with BatchNorm (as per paper)
        self.branch_layers = nn.ModuleList()
        in_features = latent_dim + num_globals

        # Following paper's architecture exactly
        for hidden_units in branch_hidden_layers:
            self.branch_layers.append(nn.Linear(in_features, hidden_units))
            self.branch_layers.append(FlexibleBatchNorm1d(hidden_units))  # Use flexible BN
            self.branch_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units

        # Final layer without activation
        self.branch_layers.append(nn.Linear(in_features, latent_dim * p))

        # Trunk network (no BatchNorm as per paper)
        self.trunk_layers = nn.ModuleList()
        in_features = 1
        for hidden_units in trunk_hidden_layers:
            self.trunk_layers.append(nn.Linear(in_features, hidden_units))
            self.trunk_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units
        self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

        # Weight init
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
        """
        branch_input: [B, latent_dim + num_globals]
        trunk_input : [M, 1]  (normalized times)
        returns     : (z_pred, trunk_out)
                     z_pred: [B, M, latent_dim]
                     trunk_out: [M, latent_dim, p]
        """
        B = branch_input.size(0)
        M = trunk_input.size(0)

        branch_out = self.forward_branch(branch_input).view(B, self.latent_dim, self.p)  # [B, L, p]
        trunk_out = self.forward_trunk(trunk_input).view(M, self.latent_dim, self.p)  # [M, L, p]

        # Tensor contraction over basis p
        z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)  # [B, M, L]

        return z_pred, trunk_out  # Return both to avoid recomputation


# =====================================================================================
# AE-DeepONet (Autoencoder + DeepONet)
# =====================================================================================

class AEDeepONet(nn.Module):
    """
    Combined Autoencoder-DeepONet model driven by a configuration dictionary.

    IMPORTANT:
    - Global variables are [P, T] (pressure, temperature), NOT [φ₀, T₀] as in the paper.
      This is intentional for our specific application.
    - Clamping is ONLY in normalized space. Physical space clamping removed.
    - PoU regularization is integrated into the loss function.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # ---- Parse configuration ----
        data_cfg: Dict[str, Any] = config["data"]
        model_cfg: Dict[str, Any] = config["model"]

        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.latent_dim = model_cfg["latent_dim"]
        self.p = model_cfg.get("p", 10)
        self.use_pou = model_cfg.get("use_pou", False)
        self.pou_weight = config["training"].get("pou_weight", 0.01)

        # Output-range controls (decoder space) - NORMALIZED SPACE ONLY
        self.decoder_output_mode: str = str(model_cfg.get("decoder_output_mode", "linear")).lower()

        # Simplified clamping - ALWAYS in normalized space
        self.output_clamp = model_cfg.get("output_clamp", None)

        # ---- Build submodules ----
        ae_encoder_layers = model_cfg["ae_encoder_layers"]
        ae_decoder_layers = model_cfg["ae_decoder_layers"]
        branch_layers = model_cfg["branch_layers"]
        trunk_layers = model_cfg["trunk_layers"]

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

        # ---- Validation ----
        if self.num_species != 12:
            self.logger.warning(
                f"Paper uses 12 species for syngas, but found {self.num_species}. "
                f"Species: {data_cfg['species_variables']}"
            )

        # NOTE: We intentionally use [P, T] instead of [φ₀, T₀] from the paper
        expected_globals = data_cfg.get("expected_globals", ["P", "T"])
        if data_cfg["global_variables"] != expected_globals:
            raise ValueError(
                f"Global variables mismatch: got {data_cfg['global_variables']}, "
                f"expected {expected_globals}. "
                f"Note: We use [P, T] (pressure, temperature) instead of "
                f"[φ₀, T₀] (equivalence ratio, initial temp) from the paper."
            )

        # ---- Trunk evaluation times (normalized) ----
        trunk_times = model_cfg.get("trunk_times", [0.25, 0.5, 0.75, 1.0])
        if not all(0.0 <= float(t) <= 1.0 for t in trunk_times):
            raise ValueError(f"trunk_times must be in [0,1], got {trunk_times}")
        self.register_buffer("trunk_times", torch.tensor(trunk_times, dtype=torch.float32).view(-1, 1))

        # ---- Optional clamp buffers (NORMALIZED SPACE ONLY) ----
        self.register_buffer("clamp_min", torch.empty(0), persistent=False)
        self.register_buffer("clamp_max", torch.empty(0), persistent=False)
        self._init_normalized_clamp()

        # ---- Log summary ----
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(
            f"AE-DeepONet initialized with {n_params:,} params | "
            f"latent_dim={self.latent_dim}, p={self.p}, use_pou={self.use_pou} | "
            f"decoder_output_mode={self.decoder_output_mode} | "
            f"clamp_active={self.clamp_min.numel() > 0} (normalized space only)"
        )

    def _init_normalized_clamp(self):
        """Initialize clamping bounds in NORMALIZED space only."""
        if self.output_clamp is None:
            return

        if isinstance(self.output_clamp, (list, tuple)) and len(self.output_clamp) == 2:
            # Simple [min, max] bounds
            lo, hi = float(self.output_clamp[0]), float(self.output_clamp[1])
            if lo > hi:
                raise ValueError(f"Clamp lower bound > upper bound: {self.output_clamp}")
            self.clamp_min = torch.full((1, self.num_species), lo, dtype=torch.float32)
            self.clamp_max = torch.full((1, self.num_species), hi, dtype=torch.float32)
        elif isinstance(self.output_clamp, dict) and "min" in self.output_clamp and "max" in self.output_clamp:
            # Dict with min/max
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
        """
        Set clamping bounds in NORMALIZED space.

        Args:
            clamp_min: Minimum values in normalized space
            clamp_max: Maximum values in normalized space
        """

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

    # -------------------------
    # Convenience wrappers
    # -------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def pou_regularization(self, trunk_outputs: torch.Tensor) -> torch.Tensor:
        """
        Partition-of-Unity (PoU) regularization.
        Encourages sum of basis functions to equal 1.
        """
        if not self.use_pou:
            return torch.tensor(0.0, device=trunk_outputs.device)

        # Sum over basis functions (last dimension)
        trunk_sum = trunk_outputs.sum(dim=-1)  # [M, L]

        # Penalize deviation from 1
        pou_loss = ((trunk_sum - 1.0) ** 2).mean()

        return pou_loss

    # -------------------------
    # Forward
    # -------------------------
    def forward(
            self,
            inputs: torch.Tensor,
            decode: bool = True,
            return_trunk_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through AE-DeepONet.

        Args:
            inputs: [B, latent_dim + num_globals] = [z0, P, T]
                   NOTE: We use [P, T] not [φ₀, T₀] as in the paper
            decode: if True, decode latent z(t) to species space
            return_trunk_outputs: if True, include trunk basis in aux (for PoU)

        Returns:
            predictions, aux_dict
              decode=True  -> predictions: [B, M, num_species]
              decode=False -> predictions: [B, M, latent_dim]
            aux_dict contains "z_pred" and optionally "trunk_outputs"
        """
        expected_dim = self.latent_dim + self.num_globals
        if inputs.size(1) != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim} ([z0, P, T]), got {inputs.size(1)}"
            )

        # Trunk times (normalized) as registered buffer
        trunk_in = self.trunk_times.to(inputs.device, inputs.dtype)  # [M, 1]

        # Single forward pass through DeepONet (avoids double computation)
        z_pred, trunk_out = self.deeponet(inputs, trunk_in)  # z_pred: [B, M, L], trunk_out: [M, L, p]

        aux: Dict[str, torch.Tensor] = {"z_pred": z_pred}
        if return_trunk_outputs or self.use_pou:
            aux["trunk_outputs"] = trunk_out

        if not decode:
            return z_pred, aux

        # Decode latent to species space
        B, M, L = z_pred.shape
        y_flat = self.decode(z_pred.reshape(B * M, L))  # [B*M, S]

        # Optional decoder output mode
        if self.decoder_output_mode == "sigmoid01":
            y_flat = torch.sigmoid(y_flat)
        elif self.decoder_output_mode != "linear":
            raise ValueError(f"Unsupported decoder_output_mode: {self.decoder_output_mode}")

        # Optional clamping in NORMALIZED space only
        if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
            y_flat = torch.max(y_flat, self.clamp_min.to(y_flat.device, y_flat.dtype))
            y_flat = torch.min(y_flat, self.clamp_max.to(y_flat.device, y_flat.dtype))

        y_pred = y_flat.view(B, M, self.num_species)
        return y_pred, aux

    # -------------------------
    # Loss
    # -------------------------
    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss (PoU is handled in trainer)."""
        return F.mse_loss(y_pred, y_true)

    # -------------------------
    # Parameter helpers
    # -------------------------
    def ae_parameters(self):
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        return self.deeponet.parameters()


# =====================================================================================
# Factory
# =====================================================================================

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory to create and place the AE-DeepONet model.
    """
    if device.type != "cuda":
        raise RuntimeError("AE-DeepONet requires a CUDA device.")

    model = AEDeepONet(config)

    # dtype
    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16":
        model = model.bfloat16()

    model = model.to(device)

    # Optional compile
    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    # Optional export - use model's dtype
    if config.get("system", {}).get("use_torch_export", False):
        try:
            model.eval()

            # Use model's actual dtype
            example_dtype = next(model.parameters()).dtype
            example_input = torch.randn(
                1,
                model.latent_dim + model.num_globals,
                device=device,
                dtype=example_dtype
            )

            exported_program = torch.export.export(
                model, args=(example_input,), kwargs={"decode": True, "return_trunk_outputs": False}
            )

            export_dir = Path(config["paths"].get("model_save_dir", "models"))
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / "ae_deeponet_exported.pt2"
            torch.export.save(exported_program, str(export_path))
            logging.getLogger(__name__).info(f"Model exported for deployment to {export_path}")

            model.train()

        except Exception as e:
            logging.getLogger(__name__).warning(f"Model export failed: {e}")

    return model