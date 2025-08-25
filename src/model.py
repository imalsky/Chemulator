#!/usr/bin/env python3
"""
Autoencoder-DeepONet implementation following Goswami et al. (2023).
FIXED: PoU regularization gating, loss division bug, default trunk times validation
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
    """Autoencoder for dimensionality reduction."""

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
            # Apply LeakyReLU to ALL layers including the final one
            y = F.leaky_relu(y, negative_slope=0.01)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


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
            self.branch_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units

        self.branch_layers.append(nn.Linear(in_features, latent_dim * p))

        # Trunk network
        self.trunk_layers = nn.ModuleList()
        in_features = 1
        for hidden_units in trunk_hidden_layers:
            self.trunk_layers.append(nn.Linear(in_features, hidden_units))
            self.trunk_layers.append(nn.LeakyReLU(0.01))
            in_features = hidden_units
        self.trunk_layers.append(nn.Linear(in_features, latent_dim * p))

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
        B = branch_input.size(0)
        M = trunk_input.size(0)

        branch_out = self.forward_branch(branch_input).view(B, self.latent_dim, self.p)
        trunk_raw = self.forward_trunk(trunk_input).view(M, self.latent_dim, self.p)

        if self.trunk_basis == "softmax":
            trunk_out = F.softmax(trunk_raw, dim=-1)
        else:
            trunk_out = trunk_raw

        z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)

        return z_pred, trunk_out


class AEDeepONet(nn.Module):
    """
    Combined Autoencoder-DeepONet model.
    FIXED: PoU gating, loss normalization, validation of trunk times
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        data_cfg = config["data"]
        model_cfg = config["model"]

        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        self.latent_dim = model_cfg["latent_dim"]
        self.p = model_cfg.get("p", 10)
        self.trunk_basis = model_cfg.get("trunk_basis", "linear")

        # FIXED: Gate PoU only for softmax basis (otherwise not meaningful)
        self.use_pou = bool(model_cfg.get("use_pou", False)) and self.trunk_basis == "softmax"
        if bool(model_cfg.get("use_pou", False)) and self.trunk_basis != "softmax":
            self.logger.info("PoU disabled for trunk_basis!='softmax' (not meaningful).")

        self.pou_weight = config["training"].get("pou_weight", 0.01)

        self.decoder_output_mode = str(model_cfg.get("decoder_output_mode", "linear")).lower()
        self.output_clamp = model_cfg.get("output_clamp", None)

        norm_cfg = config.get("normalization", {})
        default_species_method = str(norm_cfg.get("default_method", "")).lower()
        valid_01 = default_species_method in {"min-max", "log-min-max"}
        if self.decoder_output_mode == "sigmoid01" and not valid_01:
            self.logger.warning(
                "decoder_output_mode='sigmoid01' requires species to be min-max or log-min-max normalized; "
                f"found default_method='{default_species_method}'. Forcing 'linear'."
            )
            self.decoder_output_mode = "linear"

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

        self.register_buffer('_device_tensor', torch.zeros(1))

        self.clamp_min = torch.tensor([], dtype=torch.float32)
        self.clamp_max = torch.tensor([], dtype=torch.float32)
        self._init_normalized_clamp()

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

        # Validate trunk times are in [0,1]
        trunk_times = model_cfg.get("trunk_times", [0.25, 0.5, 0.75, 1.0])
        if not all(0.0 <= float(t) <= 1.0 for t in trunk_times):
            self.logger.warning(
                f"trunk_times should be in [0,1] for normalized time, got {trunk_times}. "
                "This may cause issues if time normalization doesn't map to [0,1]."
            )
        self.register_buffer(
            "default_trunk_times",
            torch.tensor(trunk_times, dtype=torch.float32).view(-1, 1),
            persistent=False
        )

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

    @property
    def device(self):
        return self._device_tensor.device

    def compute_deeponet_loss_vectorized(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            trunk_times: torch.Tensor,
            pou_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Vectorized DeepONet loss computation for entire batch at once."""
        B = inputs.shape[0]
        M = targets.shape[1]

        if trunk_times.dim() == 1:
            trunk_times = trunk_times.unsqueeze(-1)

        branch_out = self.deeponet.forward_branch(inputs)
        branch_out = branch_out.view(B, self.latent_dim, self.p)

        trunk_out = self.deeponet.forward_trunk(trunk_times)
        trunk_out = trunk_out.view(M, self.latent_dim, self.p)

        if self.deeponet.trunk_basis == "softmax":
            trunk_out = torch.nn.functional.softmax(trunk_out, dim=-1)

        z_pred = torch.einsum('blp,mlp->bml', branch_out, trunk_out)

        mse_loss = torch.nn.functional.mse_loss(z_pred, targets)

        if self.use_pou and pou_weight > 0:
            pou_loss = self.pou_regularization(trunk_out)
        else:
            pou_loss = torch.zeros((), device=self.device, dtype=z_pred.dtype)

        total = mse_loss + float(pou_weight) * pou_loss

        return total, {"mse": mse_loss, "pou": pou_loss}

    def compute_deeponet_loss_padded(
            self,
            inputs: torch.Tensor,
            padded_targets: torch.Tensor,
            all_unique_times: torch.Tensor,
            times_list: List[torch.Tensor],
            mask: torch.Tensor,
            pou_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """FIXED: Proper loss normalization for padded batch."""
        B = inputs.shape[0]
        max_M = padded_targets.shape[1]
        L = padded_targets.shape[2]

        branch_out = self.deeponet.forward_branch(inputs)
        branch_out = branch_out.view(B, self.latent_dim, self.p)

        if all_unique_times.dim() == 1:
            all_unique_times = all_unique_times.unsqueeze(-1)

        # Ensure union times are sorted ascending for searchsorted
        if __debug__:
            if all_unique_times.numel() > 1:
                diffs = all_unique_times[1:, 0] - all_unique_times[:-1, 0]
                if torch.any(diffs < 0):
                    raise ValueError("all_unique_times must be sorted ascending.")

        trunk_out_all = self.deeponet.forward_trunk(all_unique_times)
        trunk_out_all = trunk_out_all.view(-1, self.latent_dim, self.p)

        if self.deeponet.trunk_basis == "softmax":
            trunk_out_all = torch.nn.functional.softmax(trunk_out_all, dim=-1)

        z_pred_list = []

        for i in range(B):
            sample_times = times_list[i].to(self.device)
            M_i = sample_times.shape[0]

            indices = torch.searchsorted(
                all_unique_times.squeeze(-1),
                sample_times.squeeze(-1) if sample_times.dim() > 1 else sample_times
            )

            trunk_out_i = trunk_out_all[indices]
            z_pred_i = torch.einsum('lp,mlp->ml', branch_out[i], trunk_out_i)

            if M_i < max_M:
                padding = torch.zeros(max_M - M_i, self.latent_dim, device=self.device, dtype=z_pred_i.dtype)
                z_pred_i = torch.cat([z_pred_i, padding], dim=0)

            z_pred_list.append(z_pred_i)

        z_pred = torch.stack(z_pred_list)

        mse_per_point = (z_pred - padded_targets) ** 2
        mse_masked = mse_per_point * mask.unsqueeze(-1)

        # Divide by total valid elements (not just valid time steps)
        valid_elements = mask.sum().clamp_min(1) * L
        mse_loss = mse_masked.sum() / valid_elements

        if self.use_pou and pou_weight > 0:
            pou_loss = self.pou_regularization(trunk_out_all)
        else:
            pou_loss = torch.zeros((), device=self.device, dtype=z_pred.dtype)

        total = mse_loss + float(pou_weight) * pou_loss

        return total, {"mse": mse_loss, "pou": pou_loss}

    def compute_deeponet_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            trunk_times: Optional[torch.Tensor] = None,
            pou_weight: float = 0.0,
            decode_for_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the full DeepONet loss."""
        z_pred, aux = self(
            inputs,
            decode=False,
            return_trunk_outputs=True,
            trunk_times=trunk_times
        )

        if decode_for_loss:
            B, M, L = z_pred.shape
            y_pred = self.decode(z_pred.reshape(B * M, L))

            if self.decoder_output_mode == "sigmoid01":
                y_pred = torch.sigmoid(y_pred)

            if self.decoder_output_mode == "linear" and self.output_clamp is not None:
                if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
                    clamp_min = self.clamp_min.to(y_pred.device, y_pred.dtype)
                    clamp_max = self.clamp_max.to(y_pred.device, y_pred.dtype)
                    y_pred = torch.clamp(y_pred, clamp_min, clamp_max)

            y_pred = y_pred.view(B, M, self.num_species)
            y_true = self.decode(targets.reshape(B * M, L)).view(B, M, self.num_species)
            mse_loss = F.mse_loss(y_pred, y_true)
        else:
            if z_pred.shape != targets.shape:
                raise ValueError(
                    f"MSE shape mismatch: pred {tuple(z_pred.shape)} vs target {tuple(targets.shape)}"
                )
            mse_loss = F.mse_loss(z_pred, targets)

        trunk_outputs = aux.get("trunk_outputs", None)
        pou_loss = self.pou_regularization(trunk_outputs) if trunk_outputs is not None else torch.zeros(
            (), device=self.device, dtype=z_pred.dtype
        )

        total = mse_loss + float(pou_weight) * pou_loss
        return total, {"mse": mse_loss, "pou": pou_loss}

    def _init_normalized_clamp(self):
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

        trunk_sum = trunk_outputs.sum(dim=-1)
        pou_loss = ((trunk_sum - 1.0) ** 2).mean()

        return pou_loss

    def forward(
            self,
            inputs: torch.Tensor,
            decode: bool = True,
            return_trunk_outputs: bool = False,
            trunk_times: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through AE-DeepONet."""
        expected_dim = self.latent_dim + self.num_globals
        if inputs.size(1) != expected_dim:
            raise ValueError(
                f"Expected input dimension {expected_dim} ([z0, P, T]), got {inputs.size(1)}"
            )

        if trunk_times is None:
            trunk_in = self.default_trunk_times.to(inputs.device, inputs.dtype)
        else:
            if trunk_times.dim() == 1:
                trunk_times = trunk_times.unsqueeze(-1)
            trunk_in = trunk_times.to(inputs.device, inputs.dtype)

        z_pred, trunk_out = self.deeponet(inputs, trunk_in)

        aux = {"z_pred": z_pred}
        if return_trunk_outputs or self.use_pou:
            aux["trunk_outputs"] = trunk_out

        if not decode:
            return z_pred, aux

        B, M, L = z_pred.shape
        y_flat = self.decode(z_pred.reshape(B * M, L))

        if self.decoder_output_mode == "sigmoid01":
            y_flat = torch.sigmoid(y_flat)
        elif self.decoder_output_mode != "linear":
            raise ValueError(f"Unsupported decoder_output_mode: {self.decoder_output_mode}")

        if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
            y_flat = torch.max(y_flat, self.clamp_min.to(y_flat.device, y_flat.dtype))
            y_flat = torch.min(y_flat, self.clamp_max.to(y_flat.device, y_flat.dtype))

        y_pred = y_flat.view(B, M, self.num_species)
        return y_pred, aux

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss with optional clamping in normalized space."""
        if self.decoder_output_mode == "linear" and self.output_clamp is not None:
            if self.clamp_min.numel() > 0 and self.clamp_max.numel() > 0:
                clamp_min = self.clamp_min.to(y_pred.device, y_pred.dtype)
                clamp_max = self.clamp_max.to(y_pred.device, y_pred.dtype)
                y_pred = torch.clamp(y_pred, clamp_min, clamp_max)

        return F.mse_loss(y_pred, y_true)

    def ae_parameters(self):
        return self.autoencoder.parameters()

    def deeponet_parameters(self):
        return self.deeponet.parameters()


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Factory to create and place the AE-DeepONet model."""
    if device.type != "cuda":
        raise RuntimeError("AE-DeepONet requires a CUDA device.")

    model = AEDeepONet(config)

    dtype_str = config.get("system", {}).get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()
    elif dtype_str == "float16":
        model = model.half()
    elif dtype_str == "bfloat16":
        model = model.bfloat16()

    model = model.to(device)

    if config.get("system", {}).get("use_torch_compile", False):
        try:
            compile_mode = config.get("system", {}).get("compile_mode", "default")
            model = torch.compile(model, mode=compile_mode)
            logging.getLogger(__name__).info(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            logging.getLogger(__name__).warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    return model