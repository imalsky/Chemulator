#!/usr/bin/env python3
"""
Flow-map DeepONet Model Architecture (with optional AE bottlenecks)
==================================================================

Implements a DeepONet architecture for flow-map prediction with support for:
- Multiple target times per anchor (vectorized trunk over K time offsets)
- Delta (residual) or direct prediction modes
- Configurable dropout and activations for branch/trunk MLPs
- Optional trunk input de-duplication for speed (same Δt rows reused)
- Optional **joint-trained** autoencoder:
    * Encode input species before the branch to reduce branch input dimension
    * Predict in a compact latent and decode back to species before loss
    * Optional auxiliary reconstruction loss hook exposed via `compute_ae_loss`

The AE path is fully end-to-end (no separate pretraining). When disabled,
the model reduces to the original DeepONet behavior.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    """Map a string to a torch activation module."""
    key = (name or "").strip().lower()
    if key in ("relu",):
        return nn.ReLU(inplace=True)
    if key in ("lrelu", "leakyrelu", "leaky-relu"):
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if key in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if key in ("gelu",):
        return nn.GELU()
    if key in ("tanh",):
        return nn.Tanh()
    if key in ("elu",):
        return nn.ELU(inplace=True)
    if key in ("identity", "linear", ""):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name!r}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module,
    *,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    """
    Construct a multi-layer perceptron with optional dropout after each hidden activation.
    """
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_dims:
        h = int(h)
        layers.append(nn.Linear(prev, h))
        layers.append(activation.__class__() if not isinstance(activation, nn.Identity) else nn.Identity())
        if dropout_p and dropout_p > 0:
            layers.append(nn.Dropout(p=float(dropout_p)))
        prev = h
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Branch and Trunk subnets
# -----------------------------------------------------------------------------
class BranchNet(nn.Module):
    """[y_i (or z_i), g] -> φ ∈ R^p"""
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        activation: nn.Module,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = [int(width)] * int(depth)
        self.net = build_mlp(
            input_dim=int(input_dim),
            hidden_dims=hidden,
            output_dim=int(output_dim),
            activation=activation,
            dropout_p=float(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    """Δt̂ -> ψ ∈ R^p (applied elementwise over [B,K] grid)."""
    def __init__(
        self,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.net = build_mlp(
            input_dim=1,  # scalar Δt per element
            hidden_dims=[int(h) for h in hidden_dims],
            output_dim=self.output_dim,
            activation=activation,
            dropout_p=float(dropout_p),
        )

    def forward(self, dt_in: torch.Tensor, *, dedup: bool = False) -> torch.Tensor:
        """
        Args:
            dt_in: [B,K] or [B,K,1] normalized Δt.
            dedup: if True, evaluate unique Δt values once and gather back.
        Returns:
            ψ: [B,K,p]
        """
        if dt_in.ndim == 3 and dt_in.shape[-1] == 1:
            dt = dt_in.squeeze(-1)  # [B,K]
        elif dt_in.ndim == 2:
            dt = dt_in  # [B,K]
        else:
            raise ValueError(f"Trunk expects [B,K] or [B,K,1], got {tuple(dt_in.shape)}")

        B, K = dt.shape

        if not dedup:
            x = dt.reshape(B * K, 1)
            psi = self.net(x).reshape(B, K, self.output_dim)
            return psi

        # De-duplicate dt values across the batch for efficiency
        flat = dt.reshape(-1)  # [B*K]
        uniq, inv = torch.unique(flat, sorted=True, return_inverse=True)
        psi_u = self.net(uniq.view(-1, 1))        # [U,p]
        psi = psi_u.index_select(dim=0, index=inv).reshape(B, K, self.output_dim)  # [B,K,p]
        return psi


# -----------------------------------------------------------------------------
# FlowMapDeepONet with optional autoencoder bottlenecks
# -----------------------------------------------------------------------------
class FlowMapDeepONet(nn.Module):
    """
    Flow-map DeepONet with optional autoencoder bottlenecks.

    - Input AE (enc_in):    S_in -> L_in, reduces branch input size.
    - Output AE (enc_out/dec_out):
        * If predict_in_latent=True: project p -> L_out then decode to S_out.
        * If recon_weight>0: adds auxiliary reconstruction loss hook on y_j.

    Residual/add is performed in normalized species space. Shapes:
        y_i:     [B,S_full] or [B,1,S_full]
        dt_norm: [B,K] or [B,K,1]
        g:       [B,G]
        return:  [B,K,S_out] (normalized species space)
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
        predict_delta: bool,
        trunk_dedup: bool,
        activation_name: str,
        branch_dropout: float = 0.0,
        trunk_dropout: float = 0.0,
        target_idx: Optional[torch.Tensor] = None,
        input_idx: Optional[torch.Tensor] = None,
        ae_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # Core dims
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.G = int(global_dim)
        self.p = int(basis_dim)

        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = bool(trunk_dedup)

        # Optional index subsets (registered as buffers to follow device moves)
        if target_idx is not None and not isinstance(target_idx, torch.Tensor):
            target_idx = torch.tensor(target_idx, dtype=torch.long)
        if input_idx is not None and not isinstance(input_idx, torch.Tensor):
            input_idx = torch.tensor(input_idx, dtype=torch.long)
        self.register_buffer("target_idx", target_idx if target_idx is not None else None)
        self.register_buffer("input_idx",  input_idx  if input_idx  is not None else None)

        # Activations
        act = get_activation(activation_name)

        # --- AE configuration ---
        ae = dict(ae_cfg or {})
        self.ae_enabled            = bool(ae.get("enabled", False))
        self.ae_encode_inputs      = bool(ae.get("encode_inputs", True))
        self.ae_predict_in_latent  = bool(ae.get("predict_in_latent", True))
        self.ae_recon_weight       = float(ae.get("recon_weight", 0.0))
        ae_width                   = int(ae.get("width", 256))
        ae_depth                   = int(ae.get("depth", 2))
        ae_dropout                 = float(ae.get("dropout", 0.0))
        self.L_in                  = int(ae.get("latent_in",  min(self.S_in, 256)))
        self.L_out                 = int(ae.get("latent_out", min(self.S_out, 256)))

        # Encoders/decoders
        self.enc_in: Optional[nn.Sequential] = None       # S_in -> L_in (for branch)
        self.enc_out: Optional[nn.Sequential] = None      # S_out -> L_out (for recon/latent supervision)
        self.dec_out: Optional[nn.Sequential] = None      # L_out -> S_out

        if self.ae_enabled and self.ae_encode_inputs:
            self.enc_in = build_mlp(
                input_dim=self.S_in,
                hidden_dims=[ae_width] * ae_depth,
                output_dim=self.L_in,
                activation=act,
                dropout_p=ae_dropout,
            )

        if self.ae_enabled and (self.ae_predict_in_latent or self.ae_recon_weight > 0.0):
            self.enc_out = build_mlp(
                input_dim=self.S_out,
                hidden_dims=[ae_width] * ae_depth,
                output_dim=self.L_out,
                activation=act,
                dropout_p=ae_dropout,
            )
            self.dec_out = build_mlp(
                input_dim=self.L_out,
                hidden_dims=[ae_width] * ae_depth,
                output_dim=self.S_out,
                activation=act,
                dropout_p=ae_dropout,
            )

        # Branch: [encoded_or_raw(y_i), g] -> φ ∈ R^p
        branch_in_dim = (self.L_in if (self.ae_enabled and self.ae_encode_inputs) else self.S_in) + self.G
        self.branch = BranchNet(
            input_dim=branch_in_dim,
            width=int(branch_width),
            depth=int(branch_depth),
            output_dim=self.p,
            activation=act,
            dropout_p=float(branch_dropout),
        )

        # Trunk: Δt̂ -> ψ ∈ R^p (elementwise over K)
        self.trunk = TrunkNet(
            output_dim=self.p,
            hidden_dims=[int(h) for h in trunk_layers],
            activation=act,
            dropout_p=float(trunk_dropout),
        )

        # Projection: p -> S_out or L_out
        out_dim = self.L_out if (self.ae_enabled and self.ae_predict_in_latent) else self.S_out
        self.out = nn.Linear(self.p, out_dim)

    # ------------------------
    # Forward
    # ------------------------
    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_i:     [B,S_full] or [B,1,S_full] current state (normalized)
            dt_norm: [B,K] or [B,K,1] normalized Δt (dt-spec)
            g:       [B,G] globals
        Returns:
            y_pred:  [B,K,S_out] in normalized species space
        """
        B = y_i.shape[0]

        # y_i canonicalize to [B,S_full]
        if y_i.ndim == 3:
            y_i_base = y_i[:, 0, :]
        elif y_i.ndim == 2:
            y_i_base = y_i
        else:
            raise ValueError(f"Unexpected y_i shape {tuple(y_i.shape)}")

        # Select input subset if provided
        if self.input_idx is not None:
            y_in = y_i_base.index_select(1, self.input_idx)  # [B,S_in]
        else:
            y_in = y_i_base

        # Optional input encoding
        if self.ae_enabled and self.ae_encode_inputs and (self.enc_in is not None):
            y_in = self.enc_in(y_in)  # [B,L_in]

        # Branch φ
        phi = self.branch(torch.cat([y_in, g], dim=-1))  # [B,p]

        # Trunk ψ
        psi = self.trunk(dt_norm, dedup=self.trunk_dedup)  # [B,K,p]

        # Combine & project
        combined = phi.unsqueeze(1) * psi                  # [B,K,p]
        proj = self.out(combined)                          # [B,K,S_out] or [B,K,L_out]

        # Decode back to species if predicting in latent
        if self.ae_enabled and self.ae_predict_in_latent and (self.dec_out is not None):
            y_pred = self.dec_out(proj)                    # [B,K,S_out]
        else:
            y_pred = proj

        # Optional residual add (normalized species space)
        if self.predict_delta:
            if (self.S_out == y_i_base.shape[-1]) and (self.target_idx is None):
                base = y_i_base.unsqueeze(1)               # [B,1,S_full]
                if base.shape[1] != y_pred.shape[1]:
                    base = base.expand(-1, y_pred.shape[1], -1)
                y_pred = y_pred + base
            else:
                if self.target_idx is None:
                    raise RuntimeError("predict_delta=True requires target_idx when S_out != S_full")
                base_subset = y_i_base.index_select(1, self.target_idx)  # [B,S_out]
                base_subset = base_subset.unsqueeze(1)                    # [B,1,S_out]
                if base_subset.shape[1] != y_pred.shape[1]:
                    base_subset = base_subset.expand(-1, y_pred.shape[1], -1)
                y_pred = y_pred + base_subset

        return y_pred

    # ------------------------
    # AE auxiliary loss hook
    # ------------------------
    def compute_ae_loss(self, y_i: torch.Tensor, y_j_targets: torch.Tensor) -> torch.Tensor:
        """
        Optional reconstruction loss on targets y_j (normalized species space).
        Expected y_j_targets: [B,K,S_out] AFTER target slicing in the trainer.
        """
        if not (self.ae_enabled and self.ae_recon_weight > 0.0 and self.enc_out is not None and self.dec_out is not None):
            # Return a scalar zero tensor on the right device/dtype
            return y_j_targets.new_zeros(())

        B, K, S = y_j_targets.shape
        flat = y_j_targets.reshape(B * K, S)
        z = self.enc_out(flat)           # [B*K, L_out]
        rec = self.dec_out(z)            # [B*K, S_out]
        mse = (rec - flat).pow(2).mean()
        return mse * self.ae_recon_weight


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_model(config: dict) -> FlowMapDeepONet:
    """
    Build FlowMapDeepONet with optional input/target subsets and optional AE config.
    Expects:
      config["data"]["species_variables"] : list[str]
      config["data"]["global_variables"]  : list[str]
      (optional) config["data"]["input_species"], ["target_species"]
      config["model"] with keys:
        - p, branch_width, branch_depth, trunk_layers, activation, predict_delta
        - trunk_dedup (bool), branch_dropout, trunk_dropout
        - (optional) "ae": {enabled, encode_inputs, predict_in_latent, latent_in, latent_out,
                            width, depth, dropout, recon_weight}
    """
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    species_vars = list(data_cfg.get("species_variables") or [])
    if not species_vars:
        raise KeyError("config.data.species_variables must be set and non-empty")
    global_vars = list(data_cfg.get("global_variables", []))

    # Optional subsets (default to 'all')
    input_vars = list(data_cfg.get("input_species") or species_vars)
    target_vars = list(data_cfg.get("target_species") or species_vars)

    name_to_idx = {name: i for i, name in enumerate(species_vars)}
    try:
        input_idx = [name_to_idx[name] for name in input_vars]
    except KeyError as e:
        raise KeyError(
            f"config.data.input_species contains unknown name: {e.args[0]!r} "
            f"(not found in species_variables)"
        ) from None
    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(
            f"config.data.target_species contains unknown name: {e.args[0]!r} "
            f"(not found in species_variables)"
        ) from None

    # Defer torch import to keep module import light
    import torch

    return FlowMapDeepONet(
        state_dim_in=len(input_idx),
        state_dim_out=len(target_idx),
        global_dim=len(global_vars),
        basis_dim=int(model_cfg.get("p", 128)),
        branch_width=int(model_cfg.get("branch_width", 512)),
        branch_depth=int(model_cfg.get("branch_depth", 3)),
        trunk_layers=[int(h) for h in model_cfg.get("trunk_layers", [512, 512])],
        predict_delta=bool(model_cfg.get("predict_delta", True)),
        trunk_dedup=bool(model_cfg.get("trunk_dedup", False)),
        activation_name=str(model_cfg.get("activation", "gelu")),
        branch_dropout=float(model_cfg.get("branch_dropout", model_cfg.get("dropout", 0.0))),
        trunk_dropout=float(model_cfg.get("trunk_dropout", model_cfg.get("dropout", 0.0))),
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        input_idx=torch.tensor(input_idx, dtype=torch.long),
        ae_cfg=model_cfg.get("ae", None),
    )
