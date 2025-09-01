#!/usr/bin/env python3
"""
Flow-map DeepONet model using absolute normalized delta time.
The trunk now processes absolute time instead of time differences.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class PreLNResidual(nn.Module):
    def __init__(self, width: int, learnable_alpha: bool = False, alpha_init: float = 2 ** -0.5):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.fc = nn.Linear(width, width, bias=True)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.fc(self.act(self.ln(x)))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.1, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchNet(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int, learnable_alpha: bool):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.LeakyReLU(0.1, inplace=True)
        )
        self.blocks = nn.Sequential(*[PreLNResidual(width, learnable_alpha) for _ in range(max(0, depth))])
        self.proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.proj(x)


class FlowMapDeepONet(nn.Module):
    """
    DeepONet for flow-map prediction using absolute normalized time.

    The model predicts y(t) from (y_0, globals, t) where:
    - y_0: initial state (normalized)
    - globals: global parameters (normalized)
    - t: absolute normalized time
    """

    def __init__(
            self,
            species_dim: int,
            globals_dim: int,
            p: int = 256,
            branch_width: int = 256,
            branch_depth: int = 3,
            trunk_layers: Optional[List[int]] = None,
            predict_delta: bool = True,
            branch_residual_learnable: bool = False,
            trunk_dedup: bool = True,
    ):
        super().__init__()
        assert species_dim > 0 and p > 0

        self.species_dim = species_dim
        self.globals_dim = globals_dim
        self.p = p
        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = bool(trunk_dedup)

        # Branch: processes initial state and globals
        self.branch = BranchNet(
            in_dim=species_dim + globals_dim,
            width=branch_width,
            depth=max(0, branch_depth),
            out_dim=p,
            learnable_alpha=branch_residual_learnable,
        )

        # Trunk: processes absolute normalized time
        trunk_hidden = list(trunk_layers) if trunk_layers is not None else [128, 128, 128]
        self.trunk = MLP(in_dim=1, hidden=trunk_hidden, out_dim=p)

        # Head: combines branch and trunk outputs
        self.head = nn.Linear(p, species_dim)

    def _eval_trunk(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate trunk with optional deduplication for repeated time values."""
        t = t.reshape(-1, 1)
        if not self.trunk_dedup:
            return self.trunk(t)

        # Deduplicate identical time values for efficiency
        uniq, inv = torch.unique(t, dim=0, return_inverse=True)
        psi_uniq = self.trunk(uniq)
        return psi_uniq.index_select(0, inv)

    def forward(self, y_0: torch.Tensor, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (AUTONOMOUS flow-map).  t = Δt (normalized), shape [B, 1].
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Branch: [B, S+G] -> [B, p]
        b = torch.cat([y_0, g], dim=-1)
        b = self.branch(b)  # [B, p]

        # Trunk: [B, 1] -> [B, p]
        psi = self._eval_trunk(t)  # [B, p]

        # Combine in feature space and map p -> S
        phi = b * psi  # [B, p]
        out = self.head(phi)  # [B, S]

        return y_0 + out if self.predict_delta else out


def _infer_dim_from_config(config: dict, key: str) -> int:
    """Helper to infer dimensions from config."""
    try:
        data = config.get("data", {})
        seq = data.get(key, [])
        if isinstance(seq, list):
            return int(len(seq))
    except Exception:
        pass
    return 0


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create and initialize the model from config."""
    mcfg = dict(config.get("model", {}))

    # Infer dimensions
    species_dim = int(mcfg.get("species_dim", 0)) or _infer_dim_from_config(config, "target_species_variables") \
                  or _infer_dim_from_config(config, "species_variables")
    globals_dim = int(mcfg.get("globals_dim", 0)) or _infer_dim_from_config(config, "global_variables")

    # Create model
    model = FlowMapDeepONet(
        species_dim=species_dim,
        globals_dim=globals_dim,
        p=int(mcfg.get("p", 256)),
        branch_width=int(mcfg.get("branch_width", 256)),
        branch_depth=int(mcfg.get("branch_depth", 3)),
        trunk_layers=list(mcfg.get("trunk_layers", [128, 128, 128])),
        predict_delta=bool(mcfg.get("predict_delta", True)),
        branch_residual_learnable=bool(mcfg.get("branch_residual_learnable", False)),
        trunk_dedup=bool(mcfg.get("trunk_dedup", False)),
    )

    return model.to(device)