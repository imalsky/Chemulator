#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation following Goswami et al. (2023).
Complete implementation with trunk_times required at inference.
UPDATED: Support for bypassing autoencoder to work directly in species space.
UPDATED: Added global-conditioned time-warping for trunk network.
CORRECTED: Time-warp regularization on actual output b(g), not layer parameters.
CORRECTED: Proper PoU masking for variable-length sequences.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

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
                 trunk_basis: str = "linear", use_time_warp: bool = False,
                 time_warp_hidden_dim: int = 32, time_warp_bias_clamp: float = 0.25):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_globals = num_globals
        self.p = p
        self.trunk_basis = trunk_basis
        self.use_time_warp = use_time_warp
        self.time_warp_bias_clamp = time_warp_bias_clamp

        # Branch network with LayerNorm for stability
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
        in_features = 1  # Time is 1D input
        for hidden_units in trunk_hidden_layers:
            self.trunk_layers.append(nn.Linear(in_features, hidden_units))
            self.trunk_layers.append(nn.LeakyReLU(0.01, inplace=True))
            in_features = hidden_units
        self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

        # Global-conditioned time-warp network
        if self.use_time_warp:
            self.time_warp = nn.Sequential(
                nn.Linear(num_globals, time_warp_hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(time_warp_hidden_dim, 2)  # outputs [log_s, b]
            )

        # Initialize all weights with Xavier normal first
        self.apply(init_weights_glorot_normal)

        # THEN override time-warp to identity transformation if enabled
        if self.use_time_warp:
            nn.init.zeros_(self.time_warp[-1].weight)
            nn.init.zeros_(self.time_warp[-1].bias)

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch network."""
        y = x
        for layer in self.branch_layers:
            y = layer(y)
        return y

    def forward_trunk(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through trunk network."""
        y = t
        for layer in self.trunk_layers:
            y = layer(y)
        return y

    def get_time_warp_params(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time-warp parameters from global variables.
        Returns scale and bias for time transformation.
        """
        log_s, b = self.time_warp(g).chunk(2, dim=-1)  # [B, 1] each
        log_s = torch.clamp(log_s, -2.0, 2.0)  # Limit extreme scales

        # Optionally clamp bias to prevent extreme time shifts
        if self.time_warp_bias_clamp > 0:
            b = torch.clamp(b, -self.time_warp_bias_clamp, self.time_warp_bias_clamp)

        s = torch.exp(log_s)  # [B, 1]
        return s, b

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning prediction, trunk outputs, and time-warp bias.

        Args:
            branch_input: [B, latent_dim + num_globals]
            trunk_input: [M, 1] time points

        Returns:
            z_pred: [B, M, latent_dim] predictions
            trunk_out: [M, latent_dim, p] or [B, M, latent_dim, p] if time_warp enabled
            time_warp_b: [B, 1] time-warp bias (only if time_warp enabled)
        """
        B = branch_input.size(0)
        M = trunk_input.size(0)

        # Compute branch outputs
        branch_out = self.forward_branch(branch_input).view(B, self.latent_dim, self.p)

        time_warp_b = None

        if self.use_time_warp:
            # Global-conditioned time warp
            g = branch_input[:, self.latent_dim:]  # Extract globals [B, G]
            s, b = self.get_time_warp_params(g)
            time_warp_b = b  # Store for regularization

            # Apply time warping
            t = trunk_input.unsqueeze(0)  # [1, M, 1]
            t_warp = s.unsqueeze(1) * t + b.unsqueeze(1)  # [B, M, 1]

            # Compute trunk for each batch item
            t_flat = t_warp.reshape(B * M, 1)
            trunk_raw = self.forward_trunk(t_flat).view(B, M, self.latent_dim, self.p)

            # Apply basis transformation
            if self.trunk_basis == "softmax":
                trunk_out = F.softmax(trunk_raw, dim=-1)
            else:
                trunk_out = trunk_raw

            # Batch-dependent contraction
            z_pred = torch.einsum('blp,bmlp->bml', branch_out, trunk_out)
        else:
            # Original batch-independent trunk
            trunk_raw = self.forward_trunk(trunk_input).view(M, self.latent_dim, self.p)

            # Apply basis transformation
            if self.trunk_basis == "softmax":
                trunk_out = F.softmax(trunk_raw, dim=-1)
            else:
                trunk_out = trunk_raw

            # Standard contraction
            z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)

        return z_pred, trunk_out, time_warp_b


class AEDeepONet(nn.Module):
    """
    Combined Autoencoder-DeepONet model.

    Three-stage training (or direct DeepONet if bypass_autoencoder=True):
    1. Autoencoder pretraining for dimensionality reduction (skipped if bypassed)
    2. Latent dataset generation (skipped if bypassed)
    3. DeepONet training in latent/species space

    NOTE: trunk_times must always be provided at inference - no defaults.
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

        # Time-warp configuration
        self.use_time_warp = model_cfg.get("use_time_warp", False)
        self.time_warp_hidden_dim = model_cfg.get("time_warp_hidden_dim", 32)
        self.time_warp_bias_clamp = model_cfg.get("time_warp_bias_clamp", 0.25)
        self.time_warp_bias_reg = model_cfg.get("time_warp_bias_reg", 0.01)

        # Check if we're bypassing the autoencoder
        self.bypass_autoencoder = model_cfg.get("bypass_autoencoder", False)

        # Set working dimension based on bypass mode
        if self.bypass_autoencoder:
            self.working_dim = self.num_species
            self.logger.info(f"Bypassing autoencoder - working directly in {self.working_dim}-D species space")
        else:
            self.working_dim = self.latent_dim
            self.logger.info(f"Using autoencoder with {self.working_dim}-D latent space")

        if self.use_time_warp:
            self.logger.info(
                f"Global-conditioned time-warping enabled "
                f"(hidden_dim={self.time_warp_hidden_dim}, bias_clamp={self.time_warp_bias_clamp})"
            )

        # PoU (Partition of Unity) regularization settings
        # Only makes sense for linear basis (softmax already sums to 1)
        self.use_pou = model_cfg.get("use_pou", False) and self.trunk_basis == "linear"
        if model_cfg.get("use_pou", False) and self.trunk_basis == "softmax":
            self.logger.info("PoU regularization disabled for softmax basis (already sums to 1)")

        if self.use_pou and self.trunk_basis == "linear":
            self.logger.info("PoU enabled: enforcing affine partition (sum=1) for linear trunk basis")

        self.pou_weight = config["training"].get("pou_weight", 0.01)

        # Decoder output control
        self.decoder_output_mode = str(model_cfg.get("decoder_output_mode", "linear")).lower()
        self.output_clamp = model_cfg.get("output_clamp", None)

        # Validate decoder output mode against normalization
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

        # Only create autoencoder if not bypassing
        if self.bypass_autoencoder:
            self.autoencoder = None
        else:
            self.autoencoder = Autoencoder(
                num_species=self.num_species,
                latent_dim=self.latent_dim,
                encoder_hidden_layers=ae_encoder_layers,
                decoder_hidden_layers=ae_decoder_layers,
            )

        # DeepONet works with working_dim (either latent_dim or num_species)
        self.deeponet = DeepONet(
            latent_dim=self.working_dim,
            num_globals=self.num_globals,
            p=self.p,
            branch_hidden_layers=branch_layers,
            trunk_hidden_layers=trunk_layers,
            trunk_basis=self.trunk_basis,
            use_time_warp=self.use_time_warp,
            time_warp_hidden_dim=self.time_warp_hidden_dim,
            time_warp_bias_clamp=self.time_warp_bias_clamp,
        )

        # Cache device using buffer
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

        # Validate global variables (more flexible check)
        expected_globals = data_cfg.get("expected_globals", ["P", "T"])
        if set(data_cfg["global_variables"]) != set(expected_globals):
            self.logger.warning(
                f"Global variables mismatch: got {data_cfg['global_variables']}, "
                f"expected {expected_globals}. Ensure your data matches configuration."
            )
            # Only raise if strict checking is enabled
            if data_cfg.get("strict_globals_check", False):
                raise ValueError(
                    f"Global variables must be exactly {expected_globals} in order. "
                    f"Got: {data_cfg['global_variables']}"
                )

        # Tracking lists for training history
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

    @property
    def device(self):
        """Get the device of the model."""
        return self._device_tensor.device

    def compute_deeponet_loss(
            self,
            z_pred: torch.Tensor,  # [B, M, working_dim]
            z_true: torch.Tensor,  # [B, M, working_dim]
            mask: Optional[torch.Tensor] = None,  # [B, M] booleans or {0,1}
            trunk_outputs: Optional[torch.Tensor] = None,  # [M,L,P] or [B,M,L,P]
            pou_weight: float = 0.0,
            pou_mask: Optional[torch.Tensor] = None,  # [M] or [B,M]
            time_warp_b: Optional[torch.Tensor] = None  # [B,1] the OUTPUT shift b(g)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Working-space loss with proper masking for PoU and meaningful time-warp regularization.

        Args:
            z_pred: Predictions [B, M, working_dim]
            z_true: Targets [B, M, working_dim]
            mask: Optional mask for valid positions [B, M]
            trunk_outputs: Trunk basis evaluations [M,L,P] or [B,M,L,P]
            pou_weight: Weight for PoU regularization
            pou_mask: Mask for PoU computation [M] or [B,M]
            time_warp_b: Actual time-warp bias output [B,1] for regularization

        Returns:
            total_loss, metrics dictionary
        """
        # MSE with fp32 accumulation for better precision
        if mask is None:
            mse = F.mse_loss(z_pred, z_true, reduction="mean")
        else:
            sq = (z_pred - z_true).float().pow(2)
            m32 = mask.to(dtype=torch.float32, device=z_pred.device).unsqueeze(-1)  # [B,M,1]
            valid = m32.sum().clamp_min(1.0) * float(z_pred.shape[-1])
            mse = (sq * m32).sum() / valid
            mse = mse.to(z_pred.dtype)

        # PoU penalty with proper masking
        if pou_weight > 0.0 and trunk_outputs is not None:
            pou_loss = self.pou_regularization(trunk_outputs, pou_mask)
        else:
            pou_loss = torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)

        # Time-warp regularization on actual output b(g)
        time_warp_reg = torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)
        if self.use_time_warp and self.time_warp_bias_reg > 0 and time_warp_b is not None:
            # Penalize the actual predicted shift, not the layer bias parameter
            time_warp_reg = 0.5 * (time_warp_b.float().pow(2)).mean().to(z_pred.dtype)

        total = mse + pou_weight * pou_loss + self.time_warp_bias_reg * time_warp_reg
        stats = {
            "mse": float(mse.detach().cpu()),
            "pou": float(pou_loss.detach().cpu()),
            "time_warp_reg": float(time_warp_reg.detach().cpu())
        }
        return total, stats

    def pou_regularization(self,
                           trunk_outputs: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Partition-of-Unity penalty with proper masking for variable-length batches.

        Args:
            trunk_outputs: [M, L, P] (batch-independent) or [B, M, L, P] (batch-dependent)
            mask: [M] or [B,M]. Automatically adapted to match trunk_outputs shape.

        Returns:
            Scalar tensor with mean-squared deviation from unity
        """
        if trunk_outputs is None:
            dev = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
            return torch.tensor(0.0, device=dev, dtype=torch.float32)

        pen = (trunk_outputs.sum(dim=-1) - 1.0) ** 2  # -> [M,L] or [B,M,L]

        if mask is None:
            return pen.mean()

        # Move mask to the same device/dtype, then harmonize shapes
        m = mask.to(device=pen.device, dtype=pen.dtype)

        if pen.dim() == 2:  # [M, L]
            # Accept [M] or [B,M] and reduce to [M]
            if m.dim() == 2:
                m = m.any(dim=0)  # [B,M] -> [M]
            elif m.dim() != 1:
                raise ValueError("PoU mask must be [M] or [B,M] when trunk_outputs is [M,L,P].")
            m = m.unsqueeze(-1)  # [M,1] -> broadcast to [M,L]

        elif pen.dim() == 3:  # [B, M, L]
            # Accept [B,M] (preferred) or [M] (broadcast across B)
            if m.dim() == 1:  # [M]
                m = m.view(1, -1, 1).expand(pen.size(0), -1, 1)  # -> [B,M,1]
            elif m.dim() == 2:  # [B,M]
                m = m.unsqueeze(-1)  # -> [B,M,1]
            else:
                raise ValueError("PoU mask must be [B,M] or [M] when trunk_outputs is [B,M,L,P].")
        else:
            raise ValueError(f"Unexpected trunk_outputs rank (after sum over P): {pen.shape}")

        return (pen * m).sum() / m.sum().clamp_min(1.0)

    def _init_normalized_clamp(self):
        """
        Initialize clamping bounds in NORMALIZED space.
        Clamping is applied after decoding to prevent extreme values.
        """
        if self.output_clamp is None:
            return

        if isinstance(self.output_clamp, (list, tuple)) and len(self.output_clamp) == 2:
            # Uniform bounds for all species
            lo, hi = float(self.output_clamp[0]), float(self.output_clamp[1])
            if lo > hi:
                raise ValueError(f"Clamp lower bound > upper bound: {self.output_clamp}")
            self.clamp_min = torch.full((1, self.num_species), lo, dtype=torch.float32)
            self.clamp_max = torch.full((1, self.num_species), hi, dtype=torch.float32)

        elif isinstance(self.output_clamp, dict) and "min" in self.output_clamp and "max" in self.output_clamp:
            # Per-species bounds
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
        Dynamically set clamping bounds in NORMALIZED space.
        Useful for adjusting bounds during inference.
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode species concentrations to latent space (or identity if bypassed)."""
        if self.bypass_autoencoder:
            return x  # Identity mapping
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to species concentrations (or identity if bypassed)."""
        if self.bypass_autoencoder:
            return z  # Identity mapping
        return self.autoencoder.decode(z)

    def forward(
            self,
            inputs: torch.Tensor,
            decode: bool = True,
            return_trunk_outputs: bool = False,
            trunk_times: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through AE-DeepONet.

        Args:
            inputs: [B, working_dim + num_globals] = [z0/x0, globals]
            decode: If True, decode latent z(t) to species space (ignored if bypassed)
            return_trunk_outputs: If True, include trunk basis in aux
            trunk_times: REQUIRED - time points [M] or [M, 1] for evaluation

        Returns:
            predictions: [B, M, working_dim] if decode=False or bypassed, [B, M, num_species] if decode=True
            aux_dict: Dictionary with auxiliary outputs (z_pred, trunk_outputs, time_warp_b)

        Raises:
            ValueError: If trunk_times not provided
        """
        expected_dim = self.working_dim + self.num_globals
        if inputs.size(1) != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim} ([{'latent' if not self.bypass_autoencoder else 'species'}, globals]), got {inputs.size(1)}"
            )

        # trunk_times now always required - no defaults
        if trunk_times is None:
            raise ValueError(
                "trunk_times must be provided for forward pass. "
                "Specify the time points where you want predictions."
            )

        # Ensure trunk times are 2D [M, 1] for trunk network
        if trunk_times.dim() == 1:
            trunk_times = trunk_times.unsqueeze(-1)  # [M] -> [M, 1]
        trunk_in = trunk_times.to(inputs.device, inputs.dtype)

        # Single forward pass through DeepONet
        z_pred, trunk_out, time_warp_b = self.deeponet(inputs, trunk_in)

        # Store auxiliary outputs
        aux = {"z_pred": z_pred}
        if return_trunk_outputs or self.use_pou:
            aux["trunk_outputs"] = trunk_out
        if time_warp_b is not None:
            aux["time_warp_b"] = time_warp_b

        # If bypassing autoencoder, we're already in species space
        if self.bypass_autoencoder:
            # Apply output transformations if configured
            if self.decoder_output_mode == "sigmoid01":
                z_pred = torch.sigmoid(z_pred)

            # Apply clamping if configured
            if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
                z_pred = torch.max(z_pred, self.clamp_min.to(z_pred.device, z_pred.dtype))
                z_pred = torch.min(z_pred, self.clamp_max.to(z_pred.device, z_pred.dtype))

            return z_pred, aux

        # Return latent predictions if not decoding
        if not decode:
            return z_pred, aux

        # Decode from latent to species space (only when not bypassed)
        B, M, L = z_pred.shape
        y_flat = self.decode(z_pred.reshape(B * M, L))  # [B*M, S]

        # Apply decoder output mode
        if self.decoder_output_mode == "sigmoid01":
            # Sigmoid for [0,1] normalized outputs
            y_flat = torch.sigmoid(y_flat)
        elif self.decoder_output_mode != "linear":
            raise ValueError(f"Unsupported decoder_output_mode: {self.decoder_output_mode}")

        # Apply clamping in normalized space if configured
        if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
            y_flat = torch.max(y_flat, self.clamp_min.to(y_flat.device, y_flat.dtype))
            y_flat = torch.min(y_flat, self.clamp_max.to(y_flat.device, y_flat.dtype))

        # Reshape to trajectory format
        y_pred = y_flat.view(B, M, self.num_species)
        return y_pred, aux

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss for autoencoder pretraining."""
        return F.mse_loss(y_pred, y_true)

    def ae_parameters(self):
        """Return autoencoder parameters for stage 1 training."""
        if self.bypass_autoencoder:
            return iter([])  # Empty iterator
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        """Return DeepONet parameters for stage 3 training."""
        return self.deeponet.parameters()


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory function to create and configure the AE-DeepONet model.

    Args:
        config: Configuration dictionary
        device: Target device (must be CUDA unless bypassing autoencoder)

    Returns:
        Configured AE-DeepONet model

    Raises:
        RuntimeError: If device is not CUDA (unless bypassing autoencoder)
    """
    bypass_ae = config.get("model", {}).get("bypass_autoencoder", False)

    if not bypass_ae and device.type != "cuda":
        raise RuntimeError("AE-DeepONet with autoencoder requires a CUDA device for efficient training.")
    elif bypass_ae and device.type != "cuda":
        logging.getLogger(__name__).warning(
            "Running DeepONet without autoencoder on CPU - training may be slow"
        )

    model = AEDeepONet(config)

    # Set dtype (consider keeping fp32 and letting AMP handle mixed precision)
    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    # else keep float32

    model = model.to(device)

    # Optional compilation for performance
    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    return model