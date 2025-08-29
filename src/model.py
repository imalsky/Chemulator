#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation with Fourier trunk option.
Simplified version without time warping, focused on computational efficiency.
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
    """Autoencoder for dimensionality reduction."""

    def __init__(self, num_species: int, latent_dim: int,
                 encoder_hidden_layers: List[int], decoder_hidden_layers: List[int]):
        super().__init__()
        self.num_species = num_species
        self.latent_dim = latent_dim

        # Build encoder
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

        # Build decoder
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

        self.apply(init_weights_glorot_normal)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class LowRankMixer(nn.Module):
    """
    Lightweight coefficient mixer using low-rank updates.
    Allows globals to influence trajectory shape without touching trunk.
    """

    def __init__(self, num_globals: int, p: int, rank: int = 4, hidden: int = 64):
        super().__init__()
        self.p = p
        self.rank = rank

        if rank > 0:
            self.net = nn.Sequential(
                nn.Linear(num_globals, hidden),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.fc_u = nn.Linear(hidden, p * rank)
            self.fc_v = nn.Linear(hidden, rank * p)
            self.apply(init_weights_glorot_normal)

    def forward(self, g: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Mix coefficients based on globals.
        Args:
            g: [B, num_globals]
            C: [B, L, P] coefficients
        Returns:
            [B, L, P] mixed coefficients
        """
        if self.rank <= 0:
            return C

        B, L, P = C.shape
        h = self.net(g)  # [B, hidden]
        U = self.fc_u(h).view(B, P, self.rank)  # [B, P, rank]
        V = self.fc_v(h).view(B, self.rank, P)  # [B, rank, P]

        # Low-rank update: C' = C + (C @ U) @ V
        delta = torch.matmul(torch.matmul(C, U), V)
        return C + delta


class DeepONet(nn.Module):
    """DeepONet with MLP or Fourier trunk options."""

    def __init__(
            self,
            latent_dim: int,
            num_globals: int,
            p: int,
            branch_hidden_layers: List[int],
            trunk_hidden_layers: List[int],
            trunk_basis: str = "linear",
            trunk_type: str = "mlp",
            fourier_K: Optional[int] = None,
            mixer_rank: int = 0,
            mixer_hidden: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_globals = num_globals
        self.p = p
        self.trunk_basis = trunk_basis
        self.trunk_type = trunk_type.lower()

        # Branch network
        self.branch_layers = nn.ModuleList()
        in_features = latent_dim + num_globals
        for hidden_units in branch_hidden_layers:
            self.branch_layers.append(nn.Linear(in_features, hidden_units))
            self.branch_layers.append(nn.LayerNorm(hidden_units))
            self.branch_layers.append(nn.LeakyReLU(0.01, inplace=True))
            in_features = hidden_units
        self.branch_layers.append(nn.Linear(in_features, latent_dim * p))

        # Trunk network
        if self.trunk_type == "mlp":
            self.trunk_layers = nn.ModuleList()
            in_features = 1
            for hidden_units in trunk_hidden_layers:
                self.trunk_layers.append(nn.Linear(in_features, hidden_units))
                self.trunk_layers.append(nn.LeakyReLU(0.01, inplace=True))
                in_features = hidden_units
            self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

        elif self.trunk_type == "fourier":
            # Fourier trunk: p must equal 2*K (cos/sin pairs)
            K = fourier_K if fourier_K is not None else (p // 2)
            if 2 * K != p:
                raise ValueError(f"Fourier trunk requires p == 2*K; got p={p}, K={K}")
            self.fourier_K = K
            # Frequencies: [1, 2, ..., K] * 2π
            self.register_buffer('fourier_omega',
                                 2 * torch.pi * torch.arange(1, K + 1, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown trunk_type: {self.trunk_type}")

        # Coefficient mixer (optional)
        self.mixer = None
        if mixer_rank > 0:
            self.mixer = LowRankMixer(num_globals, p, rank=mixer_rank, hidden=mixer_hidden)

        # Initialize weights
        self.apply(init_weights_glorot_normal)

    def forward_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch network."""
        y = x
        for layer in self.branch_layers:
            y = layer(y)
        return y

    def forward_trunk(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk network.
        Returns [M, latent_dim * p] for both MLP and Fourier.
        """
        M = t.size(0)
        t = t.view(M, 1)

        if self.trunk_type == "mlp":
            y = t
            for layer in self.trunk_layers:
                y = layer(y)
            return y  # [M, L*P]

        else:  # fourier
            # ω: [K], t: [M,1] -> wt: [M,K]
            omega = self.fourier_omega.to(dtype=t.dtype, device=t.device).view(1, -1)
            wt = t @ omega  # [M, K]
            phi = torch.cat([torch.cos(wt), torch.sin(wt)], dim=-1)  # [M, 2K] = [M, P]

            # Expand over latent dimensions: [M, L, P] -> flatten to [M, L*P]
            phi_expanded = phi.unsqueeze(1).expand(M, self.latent_dim, self.p)
            return phi_expanded.reshape(M, self.latent_dim * self.p)

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DeepONet.
        Args:
            branch_input: [B, latent_dim + num_globals]
            trunk_input: [M, 1] time points
        Returns:
            z_pred: [B, M, latent_dim]
            trunk_out: [M, latent_dim, p]
        """
        B = branch_input.size(0)
        M = trunk_input.size(0)

        # Branch: [B, L*P] -> [B, L, P]
        branch_out = self.forward_branch(branch_input).view(B, self.latent_dim, self.p)

        # Apply coefficient mixer if enabled
        if self.mixer is not None:
            g = branch_input[:, self.latent_dim:]  # Extract globals [B, G]
            branch_out = self.mixer(g, branch_out)  # [B, L, P]

        # Trunk: [M, L*P] -> [M, L, P]
        trunk_raw = self.forward_trunk(trunk_input).view(M, self.latent_dim, self.p)

        # Apply basis transformation (only for MLP trunk)
        if self.trunk_type == "mlp" and self.trunk_basis == "softmax":
            trunk_out = F.softmax(trunk_raw, dim=-1)
        else:
            trunk_out = trunk_raw

        # Contraction: [B, L, P] × [M, L, P] -> [B, M, L]
        z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)

        return z_pred, trunk_out


class AEDeepONet(nn.Module):
    """
    Combined Autoencoder-DeepONet model.
    Simplified version without time warping and MAE calculations.
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
        self.trunk_type = model_cfg.get("trunk_type", "mlp").lower()

        # Check if bypassing autoencoder
        self.bypass_autoencoder = model_cfg.get("bypass_autoencoder", False)
        if self.bypass_autoencoder:
            self.working_dim = self.num_species
            self.logger.info(f"Bypassing autoencoder - working in {self.working_dim}-D species space")
        else:
            self.working_dim = self.latent_dim
            self.logger.info(f"Using autoencoder with {self.working_dim}-D latent space")

        # PoU settings - disabled for Fourier trunk
        if self.trunk_type == "fourier" and model_cfg.get("use_pou", False):
            raise RuntimeError(
                "PoU regularization is incompatible with Fourier trunk. "
                "Set use_pou=false or use trunk_type='mlp'."
            )
        self.use_pou = (
                model_cfg.get("use_pou", False)
                and self.trunk_basis == "linear"
                and self.trunk_type != "fourier"
        )
        if self.use_pou:
            self.logger.info("PoU enabled for linear MLP trunk basis")

        # Decoder settings
        self.decoder_output_mode = model_cfg.get("decoder_output_mode", "linear").lower()
        self.output_clamp = model_cfg.get("output_clamp", None)

        # Build networks
        ae_encoder_layers = model_cfg["ae_encoder_layers"]
        ae_decoder_layers = model_cfg["ae_decoder_layers"]
        branch_layers = model_cfg["branch_layers"]
        # Only needed for MLP trunks; for Fourier it's ignored.
        trunk_layers = model_cfg.get("trunk_layers", [])

        # Autoencoder (if not bypassed)
        if self.bypass_autoencoder:
            self.autoencoder = None
        else:
            self.autoencoder = Autoencoder(
                num_species=self.num_species,
                latent_dim=self.latent_dim,
                encoder_hidden_layers=ae_encoder_layers,
                decoder_hidden_layers=ae_decoder_layers,
            )

        # DeepONet
        self.deeponet = DeepONet(
            latent_dim=self.working_dim,
            num_globals=self.num_globals,
            p=self.p,
            branch_hidden_layers=branch_layers,
            trunk_hidden_layers=trunk_layers,  # [] for Fourier; safe
            trunk_basis=self.trunk_basis,
            trunk_type=self.trunk_type,
            fourier_K=model_cfg.get("fourier_K", None),
            mixer_rank=model_cfg.get("mixer_rank", 0),
            mixer_hidden=model_cfg.get("mixer_hidden", 64),
        )

        # Initialize clamping
        self.clamp_min = torch.tensor([], dtype=torch.float32)
        self.clamp_max = torch.tensor([], dtype=torch.float32)
        self._init_normalized_clamp()

        # Tracking lists
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        # Logging
        self.logger.info(f"Trunk type: {self.trunk_type}")
        if self.trunk_type == "fourier":
            K = model_cfg.get("fourier_K", self.p // 2)
            self.logger.info(f"Fourier trunk with K={K} frequencies (p={self.p})")
            if self.trunk_basis == "softmax":
                self.logger.info("Ignoring trunk_basis='softmax' for Fourier trunk (not applicable).")

    def compute_deeponet_loss(
            self,
            z_pred: torch.Tensor,
            z_true: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            trunk_outputs: Optional[torch.Tensor] = None,
            pou_weight: float = 0.0,
            pou_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss in working space (latent or species).
        """
        # MSE with optional masking
        if mask is None:
            mse = F.mse_loss(z_pred, z_true, reduction="mean")
        else:
            sq = (z_pred - z_true).float().pow(2)
            m32 = mask.to(dtype=torch.float32, device=z_pred.device).unsqueeze(-1)
            valid = m32.sum().clamp_min(1.0) * float(z_pred.shape[-1])
            mse = (sq * m32).sum() / valid
            mse = mse.to(z_pred.dtype)

        # PoU penalty (only for MLP trunk with linear basis)
        if pou_weight > 0.0 and trunk_outputs is not None and self.use_pou:
            pou_loss = self.pou_regularization(trunk_outputs, pou_mask)
        else:
            pou_loss = torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)

        total = mse + pou_weight * pou_loss
        stats = {
            "mse": float(mse.detach().cpu()),
            "pou": float(pou_loss.detach().cpu())
        }
        return total, stats

    def pou_regularization(self, trunk_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Partition of Unity regularization."""
        if trunk_outputs is None:
            return torch.zeros((), device=self.device, dtype=next(self.parameters()).dtype)

        pen = (trunk_outputs.sum(dim=-1) - 1.0) ** 2  # -> [M,L] or [B,M,L]

        if mask is None:
            return pen.mean()

        m = mask.to(device=pen.device, dtype=pen.dtype)

        # Handle different shapes
        if pen.dim() == 2:  # [M, L]
            if m.dim() == 2:
                m = m.any(dim=0)  # [B,M] -> [M]
            m = m.unsqueeze(-1)  # [M,1]
        elif pen.dim() == 3:  # [B, M, L]
            if m.dim() == 1:
                m = m.view(1, -1, 1).expand(pen.size(0), -1, 1)
            elif m.dim() == 2:
                m = m.unsqueeze(-1)

        return (pen * m).sum() / m.sum().clamp_min(1.0)

    def _init_normalized_clamp(self):
        """Initialize output clamping bounds."""
        if self.output_clamp is None:
            return

        if isinstance(self.output_clamp, (list, tuple)) and len(self.output_clamp) == 2:
            lo, hi = float(self.output_clamp[0]), float(self.output_clamp[1])
            self.clamp_min = torch.full((1, self.num_species), lo, dtype=torch.float32)
            self.clamp_max = torch.full((1, self.num_species), hi, dtype=torch.float32)
        elif isinstance(self.output_clamp, dict):
            lo = self.output_clamp.get("min", -float('inf'))
            hi = self.output_clamp.get("max", float('inf'))

            if isinstance(lo, (int, float)):
                self.clamp_min = torch.full((1, self.num_species), float(lo), dtype=torch.float32)
            else:
                self.clamp_min = torch.tensor(lo, dtype=torch.float32).view(1, -1)

            if isinstance(hi, (int, float)):
                self.clamp_max = torch.full((1, self.num_species), float(hi), dtype=torch.float32)
            else:
                self.clamp_max = torch.tensor(hi, dtype=torch.float32).view(1, -1)

    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        if self.bypass_autoencoder:
            return x
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        if self.bypass_autoencoder:
            return z
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
            inputs: [B, working_dim + num_globals]
            decode: Whether to decode to species space
            return_trunk_outputs: Whether to return trunk outputs
            trunk_times: Time points [M] or [M, 1]
        Returns:
            predictions: [B, M, dim] predictions
            aux: Auxiliary outputs dict
        """
        if trunk_times is None:
            raise ValueError("trunk_times must be provided")

        # Ensure trunk times are 2D
        if trunk_times.dim() == 1:
            trunk_times = trunk_times.unsqueeze(-1)
        trunk_in = trunk_times.to(inputs.device, inputs.dtype)

        # Guard input width once
        expected = self.working_dim + self.num_globals
        if inputs.size(1) != expected:
            raise ValueError(
                f"Expected inputs with last dim {expected} "
                f"([{'species' if self.bypass_autoencoder else 'latent'} + globals]), "
                f"got {inputs.size(1)}."
            )

        # Forward through DeepONet
        z_pred, trunk_out = self.deeponet(inputs, trunk_in)

        # Auxiliary outputs
        aux = {"z_pred": z_pred}
        if return_trunk_outputs or self.use_pou:
            aux["trunk_outputs"] = trunk_out

        # If bypassing autoencoder, apply output transformations
        if self.bypass_autoencoder:
            if self.decoder_output_mode == "sigmoid01":
                z_pred = torch.sigmoid(z_pred)

            if self.clamp_min.numel() > 0:
                z_pred = torch.max(z_pred, self.clamp_min.to(z_pred.device, z_pred.dtype))
                z_pred = torch.min(z_pred, self.clamp_max.to(z_pred.device, z_pred.dtype))

            return z_pred, aux

        # Return latent predictions if not decoding
        if not decode:
            return z_pred, aux

        # Decode to species space
        B, M, L = z_pred.shape
        y_flat = self.decode(z_pred.reshape(B * M, L))

        # Apply output transformations
        if self.decoder_output_mode == "sigmoid01":
            y_flat = torch.sigmoid(y_flat)

        if self.clamp_min.numel() > 0:
            y_flat = torch.max(y_flat, self.clamp_min.to(y_flat.device, y_flat.dtype))
            y_flat = torch.min(y_flat, self.clamp_max.to(y_flat.device, y_flat.dtype))

        y_pred = y_flat.view(B, M, self.num_species)
        return y_pred, aux

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss."""
        return F.mse_loss(y_pred, y_true)

    def ae_parameters(self):
        """Return autoencoder parameters."""
        if self.bypass_autoencoder:
            return iter([])
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        """Return DeepONet parameters."""
        return self.deeponet.parameters()


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and configure model."""
    model = AEDeepONet(config)

    # Set dtype
    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        model = model.bfloat16()

    model = model.to(device)

    # Optional compilation
    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Compilation failed: {e}")

    return model