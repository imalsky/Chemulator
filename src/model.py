#!/usr/bin/env python3
"""
Flow-map DeepONet Model Architecture
=====================================
Implements a DeepONet architecture for flow-map prediction with support for
multiple target times per anchor point.

Architecture Components:
- Branch network: Processes state and global features to produce basis coefficients
- Trunk network: Maps normalized time differences to temporal basis functions
- Output layer: Combines branch and trunk outputs to predict state evolution

Features:
- Supports both delta (residual) and direct prediction modes
- Configurable dropout in branch and trunk MLPs
- Corrected residual connection in log-physical space
- Optional SoftMax head for enforcing non-negativity and sum-to-one constraints
"""

from __future__ import annotations

import math
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Activations ----------------------------------

# Map common aliases to PyTorch activation class names
ACTIVATION_ALIASES = {
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "swish": "SiLU",  # Common alias
    "tanh": "Tanh",
}


def get_activation(name: str) -> nn.Module:
    """
    Create activation function from name using PyTorch's built-in modules.

    Args:
        name: Activation function name

    Returns:
        Activation module instance

    Raises:
        ValueError: If activation name is not supported
    """
    # Normalize name and check aliases
    name_lower = name.lower()
    class_name = ACTIVATION_ALIASES.get(name_lower, name)

    # Try to get the activation class from torch.nn
    activation_class = getattr(nn, class_name, None)

    if activation_class is None or not issubclass(activation_class, nn.Module):
        # List supported activations for helpful error message
        supported = sorted(set(ACTIVATION_ALIASES.keys()) |
                           {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "ELU"})
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Supported activations: {', '.join(supported)}"
        )

    # Special handling for LeakyReLU to set negative_slope
    if class_name == "LeakyReLU":
        return activation_class(negative_slope=0.01)

    return activation_class()


# --------------------------------- MLP ---------------------------------------


class MLP(nn.Module):
    """
    Multi-layer perceptron with dropout support.

    This is a cleaner implementation using ModuleList that makes the
    architecture more explicit and easier to debug.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: nn.Module,
            dropout_p: float = 0.0,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer widths
            output_dim: Output feature dimension
            activation: Activation module instance (e.g., nn.GELU())
            dropout_p: Dropout probability after each hidden layer
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers with activation and optional dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Use the same activation class but create new instances
            layers.append(activation.__class__())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -------------------------------- Branch -------------------------------------


class BranchNet(nn.Module):
    """
    Branch network: processes concatenated [y_i, g] -> φ ∈ R^p.

    The branch network encodes the initial state and global parameters
    into a set of basis coefficients.
    """

    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: nn.Module,
            *,
            dropout_p: float = 0.0,
    ) -> None:
        """
        Args:
            input_dim: S + G (state dim + global dim)
            width: Hidden width for each layer
            depth: Number of hidden layers (>=1)
            output_dim: p (basis dimension)
            activation: Activation module (e.g., nn.GELU())
            dropout_p: Dropout probability after each hidden activation
        """
        super().__init__()

        # Store dimensions for debugging/inspection
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.depth = max(1, int(depth))
        self.output_dim = int(output_dim)

        # Create MLP with uniform width across hidden layers
        hidden_dims = [self.width] * self.depth
        self.network = MLP(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, S+G] -> [B, p]"""
        return self.network(x)


# -------------------------------- Trunk --------------------------------------


class TrunkNet(nn.Module):
    """
    Trunk network: maps normalized time input (Δt_norm) to ψ ∈ R^p.

    Simplified implementation with cleaner shape handling.
    Accepts time tensors with shapes [B], [B,1], [B,K], or [B,K,1].
    """

    def __init__(
            self,
            output_dim: int,
            hidden_dims: Sequence[int],
            activation: nn.Module,
            *,
            dropout_p: float = 0.0,
    ) -> None:
        """
        Args:
            output_dim: p (basis dimension)
            hidden_dims: List of hidden layer widths
            activation: Activation module
            dropout_p: Dropout probability after each hidden activation
        """
        super().__init__()

        self.output_dim = int(output_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]

        # MLP expects scalar time input
        self.network = MLP(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with simplified shape handling.

        Args:
            t: Time tensor with shape [B], [B,1], [B,K], or [B,K,1]

        Returns:
            Basis functions with shape [B,K,p] (K=1 if input was 1D)
        """
        # Remove trailing singleton dimension if present
        if t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)  # [B,K,1] -> [B,K]

        # Ensure at least 2D
        if t.ndim == 1:
            t = t.unsqueeze(1)  # [B] -> [B,1]

        # Now t is guaranteed to be [B,K]
        B, K = t.shape

        # Flatten batch and time dimensions for MLP processing
        t_flat = t.reshape(B * K, 1)  # [B*K, 1]

        # Apply MLP
        out = self.network(t_flat)  # [B*K, p]

        # Reshape to [B, K, p]
        return out.view(B, K, self.output_dim)


# ------------------------------ DeepONet Core --------------------------------


class FlowMapDeepONet(nn.Module):
    """
    Flow-map DeepONet: (y_i, g, Δt_norm) -> y_j

    Architecture:
    - Branch takes [y_i, g] and outputs basis coefficients φ
    - Trunk takes Δt_norm and outputs temporal basis functions ψ
    - Output combines via element-wise product and linear projection

    Features:
    - Supports output subset via target_idx
    - Corrected residual connection in log-physical space
    - Optional SoftMax head for conservation constraints
    - Proper weight initialization for stability
    """

    def __init__(
            self,
            *,
            state_dim_in: int,
            state_dim_out: int,
            global_dim: int,
            basis_dim: int,
            branch_width: int,
            branch_depth: int,
            trunk_layers: Sequence[int],
            predict_delta: bool = True,
            predict_delta_log_phys: bool = False,
            trunk_dedup: bool = False,  # Reserved for future use
            activation_name: str = "gelu",
            branch_dropout: float = 0.0,
            trunk_dropout: float = 0.0,
            target_idx: Optional[torch.Tensor] = None,
            target_log_mean: Optional[Sequence[float]] = None,
            target_log_std: Optional[Sequence[float]] = None,
            # SoftMax head parameters
            softmax_head: bool = False,
            allow_partial_simplex: bool = False,
    ) -> None:
        super().__init__()

        # Store dimensions
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.p = int(basis_dim)

        # Prediction mode flags
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.trunk_dedup = bool(trunk_dedup)  # Reserved

        # SoftMax head flags
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        # ============ Configuration Validation (Hard Exits) ============

        # SoftMax is incompatible with any residual mode
        if self.softmax_head and (self.predict_delta or self.predict_delta_log_phys):
            raise RuntimeError(
                "INVALID CONFIGURATION: softmax_head=True is incompatible with residual modes.\n"
                "SoftMax outputs must stay on the simplex (>=0, sum=1), but residuals break this.\n"
                "Set predict_delta=False and predict_delta_log_phys=False to use SoftMax head."
            )

        # SoftMax with subset outputs is physically wrong unless explicitly allowed
        if self.softmax_head and (self.S_out != self.S_in) and not self.allow_partial_simplex:
            raise RuntimeError(
                "INVALID CONFIGURATION: softmax_head=True with S_out != S_in will renormalize\n"
                "mass within a subset of species, which is physically incorrect.\n"
                "Either:\n"
                "  1) Predict all species (set target_species = species_variables)\n"
                "  2) Set allow_partial_simplex=True if you accept subset renormalization\n"
                "Current: S_in={}, S_out={}".format(self.S_in, self.S_out)
            )

        # Register target indices as a buffer (handle None properly)
        if target_idx is None:
            self.target_idx = None
        else:
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx)

        # Register log statistics for SoftMax head or corrected residual mode
        if self.predict_delta_log_phys or self.softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError(
                    "target_log_mean and target_log_std are required when:\n"
                    "  - predict_delta_log_phys=True (for corrected residual)\n"
                    "  - softmax_head=True (for probability -> z-space conversion)"
                )

            # Validate stats shape matches output dimension
            if len(target_log_mean) != self.S_out or len(target_log_std) != self.S_out:
                raise ValueError(
                    f"Normalization stats length mismatch: "
                    f"got {len(target_log_mean)} and {len(target_log_std)}, "
                    f"expected {self.S_out} (S_out)"
                )

            log_mean = torch.tensor(target_log_mean, dtype=torch.float32)
            log_std = torch.tensor(target_log_std, dtype=torch.float32)
            self.register_buffer("log_mean", log_mean)
            self.register_buffer("log_std", torch.clamp(log_std, min=1e-10))

            # Register ln(10) as buffer for efficient log base conversion
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))

            # Additional shape check after registration
            if self.softmax_head:
                if self.log_mean.numel() != self.S_out or self.log_std.numel() != self.S_out:
                    raise ValueError(
                        f"log_mean/log_std must have length S_out={self.S_out} for softmax_head"
                    )

            # Log the configuration for debugging
            import logging
            logger = logging.getLogger(__name__)
            if self.softmax_head:
                logger.info(
                    f"SoftMax head enabled: outputs will be constrained to >=0 and sum=1\n"
                    f"  S_out={self.S_out}, S_in={self.S_in}\n"
                    f"  allow_partial_simplex={self.allow_partial_simplex}"
                )
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None

        # Create activation instance
        act = get_activation(activation_name)

        # Branch network: [y_i, g] -> φ ∈ R^p
        self.branch = BranchNet(
            input_dim=self.S_in + self.G,
            width=int(branch_width),
            depth=int(branch_depth),
            output_dim=self.p,
            activation=act,
            dropout_p=float(branch_dropout),
        )

        # Trunk network: Δt_norm -> ψ ∈ R^p
        self.trunk = TrunkNet(
            output_dim=self.p,
            hidden_dims=[int(h) for h in trunk_layers],
            activation=act,
            dropout_p=float(trunk_dropout),
        )

        # Output projection: elementwise φ⊙ψ -> target dimension
        self.out = nn.Linear(self.p, self.S_out)

        # Initialize output layer
        if self.predict_delta or self.predict_delta_log_phys:
            # Small weights help when using residual connections
            nn.init.zeros_(self.out.bias)
            with torch.no_grad():
                self.out.weight.mul_(0.1)
        elif self.softmax_head:
            # Zero bias can stabilize early training with SoftMax
            nn.init.zeros_(self.out.bias)

    def forward(
            self,
            y_i: torch.Tensor,  # [B,S_in] or [B,K,S_in]
            dt_norm: torch.Tensor,  # [B], [B,1], [B,K], or [B,K,1]
            g: torch.Tensor  # [B,G]
    ) -> torch.Tensor:
        """
        Forward pass of the DeepONet.

        Returns predictions in z-space (log-standard normalized).

        Head selection (mutually exclusive):
          - softmax_head=True: logits -> log_softmax -> log10 -> z
          - predict_delta_log_phys=True: residual in log10-phys -> z
          - predict_delta=True: residual in z
          - else: direct z

        Args:
            y_i: Initial state (normalized)
            dt_norm: Normalized time differences
            g: Global features/parameters

        Returns:
            Predicted states at target times [B,K,S_out] in z-space
        """
        # Validate inputs
        if g.ndim != 2:
            raise ValueError(f"g must be 2D [B,G], got {tuple(g.shape)}")
        B = g.shape[0]

        # Check global dimension edge case
        if self.G == 0 and g.shape[1] != 0:
            raise ValueError(f"global_dim=0 but g has shape {tuple(g.shape)}")
        if self.G > 0 and g.shape[1] != self.G:
            raise ValueError(f"Expected G={self.G}, got {g.shape[1]}")

        # Extract base state from y_i (handle both [B,S] and [B,K,S])
        if y_i.ndim == 2:
            if y_i.shape[0] != B:
                raise ValueError("Batch mismatch y_i vs g")
            y_i_base = y_i
        elif y_i.ndim == 3:
            if y_i.shape[0] != B:
                raise ValueError("Batch mismatch y_i vs g")
            # Use first time point as base state
            y_i_base = y_i[:, 0, :]
        else:
            raise ValueError(f"y_i must be [B,S_in] or [B,K,S_in], got {tuple(y_i.shape)}")

        if y_i_base.shape[1] != self.S_in:
            raise ValueError(f"Expected S_in={self.S_in}, got {y_i_base.shape[1]}")

        # Determine number of output times K from dt_norm
        if dt_norm.ndim == 1:
            if dt_norm.shape[0] != B:
                raise ValueError("Batch mismatch dt_norm vs g")
            K = 1
        elif dt_norm.ndim == 2:
            if dt_norm.shape[0] != B:
                raise ValueError("Batch mismatch dt_norm vs g")
            K = dt_norm.shape[1]
        elif dt_norm.ndim == 3:
            if dt_norm.shape[-1] != 1:
                raise ValueError("dt_norm last dim must be 1")
            if dt_norm.shape[0] != B:
                raise ValueError("Batch mismatch dt_norm vs g")
            K = dt_norm.shape[1]
        else:
            raise ValueError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)}")

        # Branch network: encode state and globals
        branch_input = torch.cat([y_i_base, g], dim=-1)  # [B, S_in+G]
        phi = self.branch(branch_input)  # [B, p]

        # Trunk network: encode time differences
        psi = self.trunk(dt_norm)  # [B, K, p]

        # Verify trunk output shape (use exception, not assert)
        if not (psi.ndim == 3 and psi.shape[0] == B and psi.shape[-1] == self.p):
            raise ValueError(f"Trunk must return [B,K,{self.p}], got {tuple(psi.shape)}")

        # Combine basis functions via element-wise product
        # phi: [B, p] -> [B, 1, p] for broadcasting
        combined = phi.unsqueeze(1) * psi  # [B, K, p]

        # Project to output dimension
        logits = self.out(combined)  # [B, K, S_out] (these become logits for SoftMax)

        # ============ Apply Head Based on Configuration ============

        if self.softmax_head:
            # Use numerically stable log_softmax approach
            # logits -> log_softmax -> log10 -> z-space

            # Compute log probabilities (natural log)
            log_p = F.log_softmax(logits, dim=-1)  # [B, K, S_out]

            # Ensure dtype compatibility under autocast
            ln10 = self.ln10.to(dtype=log_p.dtype)
            log_mean = self.log_mean.to(dtype=log_p.dtype)
            log_std = self.log_std.to(dtype=log_p.dtype)

            # Convert from natural log to log10 using buffered constant
            log10_p = log_p / ln10  # [B, K, S_out]

            # Convert to z-space using log-standard normalization
            y_pred = (log10_p - log_mean) / log_std  # [B, K, S_out]

            return y_pred

        elif self.predict_delta_log_phys:
            # Corrected residual in log-physical space

            # Get base state for residual (subset if needed)
            if self.S_out != self.S_in:
                # Defensive check for target_idx
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError(
                        "target_idx must be a Tensor when S_out != S_in in predict_delta_log_phys mode"
                    )
                base_z = y_i_base.index_select(1, self.target_idx)  # [B, S_out]
            else:
                base_z = y_i_base  # [B, S_out]

            # Convert base from z-space to log10(y) space
            base_log = base_z * self.log_std + self.log_mean  # [B, S_out]

            # Add network output as Δlog10(y) in log-physical space
            y_pred_log = base_log.unsqueeze(1) + logits  # [B, K, S_out]

            # Convert back to z-space for consistency with rest of pipeline
            y_pred = (y_pred_log - self.log_mean) / self.log_std  # [B, K, S_out]

            return y_pred

        elif self.predict_delta:
            # Standard residual in normalized z-space

            # Get base state (subset if needed)
            if self.S_out != self.S_in:
                # Defensive check for target_idx
                if not isinstance(self.target_idx, torch.Tensor):
                    raise RuntimeError(
                        "target_idx must be a Tensor when S_out != S_in in predict_delta mode"
                    )
                base = y_i_base.index_select(1, self.target_idx)  # [B, S_out]
            else:
                base = y_i_base  # [B, S_out]

            # Add residual with broadcasting
            y_pred = logits + base.unsqueeze(1)  # [B, K, S_out]

            return y_pred

        else:
            # Direct prediction in z-space (no residual, no SoftMax)
            return logits

    @torch.no_grad()
    def validate_conservation(self, y_pred_z: torch.Tensor) -> tuple:
        """
        Check if predictions satisfy conservation after denormalization.

        Args:
            y_pred_z: Predictions in z-space [B,K,S] or [B,S]

        Returns:
            (max_sum_error, min_value): Maximum deviation from sum=1 and minimum value
        """
        if not self.softmax_head:
            return float('nan'), float('nan')

        # z -> log10 -> physical
        log10_y = (y_pred_z.to(self.log_mean.dtype) * self.log_std) + self.log_mean
        y_phys = torch.pow(torch.tensor(10.0, dtype=log10_y.dtype, device=log10_y.device), log10_y)

        # Check sum-to-1
        sums = y_phys.sum(dim=-1)
        sum_error = (sums - 1.0).abs().max().item()

        # Check non-negativity
        min_val = y_phys.min().item()

        return sum_error, min_val

    @torch.no_grad()
    def check_stat_consistency(self, loss_log_mean: torch.Tensor, loss_log_std: torch.Tensor) -> None:
        """
        Verify that model and loss normalization stats match exactly (value & length).
        Raises with clear diagnostics if not.

        Intended to be called once during trainer setup, after both model and loss
        have loaded stats from normalization.json.

        Args:
            loss_log_mean: Log means from the loss function
            loss_log_std: Log stds from the loss function

        Raises:
            RuntimeError: If model stats aren't registered
            ValueError: If statistics don't match
        """
        # Nothing to check if the model doesn't use stats
        if not (getattr(self, "softmax_head", False) or getattr(self, "predict_delta_log_phys", False)):
            return

        # Ensure model actually has stats
        if (getattr(self, "log_mean", None) is None) or (getattr(self, "log_std", None) is None):
            raise RuntimeError(
                "Model is configured to use normalization stats but log_mean/log_std buffers are missing."
            )

        # Normalize dtype + device for robust comparison (move to CPU, float32)
        m_mu = self.log_mean.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        m_sig = self.log_std.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        l_mu = loss_log_mean.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        l_sig = loss_log_std.detach().to(dtype=torch.float32, device="cpu").reshape(-1)

        # Optional: length sanity (helps catch species-order bugs early)
        if m_mu.numel() != l_mu.numel() or m_sig.numel() != l_sig.numel():
            raise ValueError(
                f"Normalization stats length mismatch: "
                f"model mean/std = ({m_mu.numel()},{m_sig.numel()}), "
                f"loss mean/std = ({l_mu.numel()},{l_sig.numel()})."
            )

        rtol, atol = 1e-6, 1e-9

        if not torch.allclose(m_mu, l_mu, rtol=rtol, atol=atol):
            diff = (m_mu - l_mu).abs()
            idx = int(diff.argmax())
            raise ValueError(
                f"log_mean mismatch at index {idx}: "
                f"model={m_mu[idx].item():.9g}, loss={l_mu[idx].item():.9g}, "
                f"max|Δ|={diff.max().item():.3e} (rtol={rtol}, atol={atol})."
            )

        if not torch.allclose(m_sig, l_sig, rtol=rtol, atol=atol):
            diff = (m_sig - l_sig).abs()
            idx = int(diff.argmax())
            raise ValueError(
                f"log_std mismatch at index {idx}: "
                f"model={m_sig[idx].item():.9g}, loss={l_sig[idx].item():.9g}, "
                f"max|Δ|={diff.max().item():.3e} (rtol={rtol}, atol={atol})."
            )


# ------------------------------ Factory --------------------------------------


def create_model(config: dict) -> FlowMapDeepONet:
    """
    Build FlowMapDeepONet from configuration dictionary.

    Handles:
    - Target species subset selection
    - Loading normalization statistics for SoftMax head or corrected residual mode
    - Mapping configuration keys to model parameters
    - Configuration validation for SoftMax constraints

    Args:
        config: Full configuration dictionary with 'data', 'model', and 'paths' sections

    Returns:
        Configured FlowMapDeepONet instance
    """
    import json
    from pathlib import Path

    # Extract data configuration
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))

    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")
    if global_vars is None:
        raise KeyError("data.global_variables must be set (use [] for none)")

    # Determine target species (default to all species if not specified)
    target_vars = list(data_cfg.get("target_species") or species_vars)

    # Create index mapping for target subset
    name_to_idx = {name: i for i, name in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    # Determine dimensions
    state_dim_in = len(species_vars)
    state_dim_out = len(target_vars)
    global_dim = len(global_vars)

    # Extract model configuration
    mcfg = config.get("model", {})

    # Validate SoftMax configuration at factory level
    if mcfg.get("softmax_head", False):
        # Check residual mode compatibility
        if mcfg.get("predict_delta", True) or mcfg.get("predict_delta_log_phys", False):
            raise ValueError(
                "Configuration error: softmax_head=True requires both "
                "predict_delta=False and predict_delta_log_phys=False"
            )

        # Check subset renormalization (mirror constructor behavior)
        if state_dim_out != state_dim_in and not mcfg.get("allow_partial_simplex", False):
            raise ValueError(
                f"Configuration error: softmax_head=True with subset of species "
                f"({state_dim_out}/{state_dim_in}) will renormalize within the subset, "
                f"which is physically incorrect. Either:\n"
                f"  1) Predict all species (set target_species = species_variables)\n"
                f"  2) Set allow_partial_simplex=True to explicitly accept this behavior"
            )

    # Check if we need normalization statistics
    need_stats = bool(
        mcfg.get("predict_delta_log_phys", False) or
        mcfg.get("softmax_head", False)
    )

    # Load normalization statistics if needed
    target_log_mean = None
    target_log_std = None
    if need_stats:
        norm_path = Path(config["paths"]["processed_data_dir"]) / "normalization.json"
        with open(norm_path, "r") as f:
            manifest = json.load(f)
        stats = manifest["per_key_stats"]

        # Extract statistics for target species
        target_log_mean = []
        target_log_std = []
        for name in target_vars:
            if name not in stats:
                raise KeyError(f"Target species '{name}' not found in normalization statistics")
            s = stats[name]
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))

        # Verify stats length matches output dimension
        if len(target_log_mean) != state_dim_out or len(target_log_std) != state_dim_out:
            raise ValueError(
                f"Normalization statistics length mismatch: "
                f"got {len(target_log_mean)} means and {len(target_log_std)} stds, "
                f"but state_dim_out={state_dim_out}"
            )

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded normalization statistics for {len(target_vars)} target species")

    # Create model with all configuration options
    return FlowMapDeepONet(
        state_dim_in=state_dim_in,
        state_dim_out=state_dim_out,
        global_dim=global_dim,
        basis_dim=int(mcfg.get("p", 128)),
        branch_width=int(mcfg.get("branch_width", 512)),
        branch_depth=int(mcfg.get("branch_depth", 3)),
        trunk_layers=[int(h) for h in mcfg.get("trunk_layers", [512, 512])],
        predict_delta=bool(mcfg.get("predict_delta", True)),
        predict_delta_log_phys=bool(mcfg.get("predict_delta_log_phys", False)),
        trunk_dedup=bool(mcfg.get("trunk_dedup", False)),
        activation_name=str(mcfg.get("activation", "gelu")),
        branch_dropout=float(mcfg.get("branch_dropout", mcfg.get("dropout", 0.0))),
        trunk_dropout=float(mcfg.get("trunk_dropout", mcfg.get("dropout", 0.0))),
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        # SoftMax head parameters
        softmax_head=bool(mcfg.get("softmax_head", False)),
        allow_partial_simplex=bool(mcfg.get("allow_partial_simplex", False)),
    )