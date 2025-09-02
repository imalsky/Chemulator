#!/usr/bin/env python3
# model.py
from __future__ import annotations

from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- small utils ---------------------------------

def _mlp_layers(in_dim: int, hidden: Sequence[int], out_dim: int, *, act: nn.Module) -> nn.Sequential:
    """
    Build an MLP: Linear + act for each hidden layer, then Linear(out).
    """
    layers: List[nn.Module] = []
    dim_prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(dim_prev, int(h)))
        layers.append(act)
        dim_prev = int(h)
    layers.append(nn.Linear(dim_prev, out_dim))
    return nn.Sequential(*layers)


# --------------------------------- modules -----------------------------------

class BranchNet(nn.Module):
    """
    Branch(y_i, g) -> phi in R^p
    Input dim = S + G
    """
    def __init__(self, in_dim: int, width: int, depth: int, p: int, act: Optional[nn.Module] = None):
        super().__init__()
        act = act if act is not None else nn.GELU()
        # depth >= 1: first hidden of size `width`, then (depth-1) more hidden layers
        hidden = [int(width)] * max(1, int(depth))
        self.net = _mlp_layers(in_dim, hidden[:-1], width, act=act) if len(hidden) > 1 else nn.Sequential()
        self.head = nn.Sequential(*( [nn.Linear(in_dim, width), act] if len(hidden) == 1 else [] ))
        self.out = nn.Linear(width if len(hidden) >= 1 else in_dim, p)

        # If depth > 1, add the remaining hidden layers after the initial block
        if len(hidden) > 1:
            blocks: List[nn.Module] = []
            # first linear stack is in self.net; we now append remaining (width->width) layers
            for _ in range(len(hidden) - 1):
                blocks.append(nn.Linear(width, width))
                blocks.append(act)
            self.mid = nn.Sequential(*blocks)
        else:
            self.mid = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S+G]
        if isinstance(self.net, nn.Sequential) and len(self.net) > 0:
            y = self.net(x)
        else:
            y = self.head[0](x) if len(self.head) else x
            if len(self.head) == 2:  # Linear + act
                y = self.head[1](y)
        y = self.mid(y)
        phi = self.out(y)  # [B, p]
        return phi


class TrunkNet(nn.Module):
    """
    Trunk(dt_norm) -> psi in R^p
    Works on scalar inputs; handles [B,K] or [B] by reshaping to (...,1).
    """
    def __init__(self, p: int, layers: Sequence[int], act: Optional[nn.Module] = None):
        super().__init__()
        act = act if act is not None else nn.GELU()
        hidden = [int(h) for h in layers] if len(layers) > 0 else [256, 256]
        self.net = _mlp_layers(1, hidden, p, act=act)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,K] or [B]
        t_in = t.unsqueeze(-1) if t.ndim == 1 or t.shape[-1] != 1 else t  # (...,1)
        psi = self.net(t_in)  # [..., p]
        return psi


class FlowMapDeepONet(nn.Module):
    """
    DeepONet-style flow-map:
      phi = Branch([y_i, g])          -> [B, p]
      psi = Trunk(dt_norm)            -> [B, K, p] (or [B, 1, p] when K=1)
      f   = phi[:,None,:] * psi       -> [B, K, p]
      pred = Linear_p_to_S(f)         -> [B, K, S]
      if predict_delta: pred += y_i[:,None,:]

    Notes
    -----
    * Always returns [B, K, S] (K=1 gives [B, 1, S]). The Trainer you’re using
      harmonizes shapes if your target is [B, S].
    * `trunk_dedup` flag is accepted for config parity. If you later want true
      dedup of repeated dt across the batch, you can add it in forward.
    """
    def __init__(
        self,
        S: int,
        G: int,
        p: int = 256,
        branch_width: int = 1024,
        branch_depth: int = 3,
        trunk_layers: Sequence[int] = (512, 512),
        predict_delta: bool = True,
        trunk_dedup: bool = False,
        act_name: str = "gelu",
    ):
        super().__init__()
        act = nn.GELU() if act_name.lower() == "gelu" else nn.ReLU()

        self.S = int(S)
        self.G = int(G)
        self.p = int(p)
        self.predict_delta = bool(predict_delta)
        self.trunk_dedup = bool(trunk_dedup)

        # Modules
        self.branch = BranchNet(in_dim=S + G, width=branch_width, depth=branch_depth, p=p, act=act)
        self.trunk = TrunkNet(p=p, layers=trunk_layers, act=act)
        self.out = nn.Linear(p, S)  # shared across time positions

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        y_i:    [B, S]
        dt_norm:[B, K]  or [B] / [B,1]
        g:      [B, G]

        returns: [B, K, S]
        """
        B = y_i.shape[0]
        assert y_i.dim() == 2 and y_i.shape[1] == self.S, f"y_i expected [B,{self.S}] got {tuple(y_i.shape)}"
        assert g.dim() == 2 and g.shape[1] == self.G, f"g expected [B,{self.G}] got {tuple(g.shape)}"

        # Branch features
        b = torch.cat([y_i, g], dim=-1)  # [B, S+G]
        phi = self.branch(b)             # [B, p]

        # Trunk features
        if dt_norm.dim() == 1:
            dt_norm = dt_norm.view(B, 1)
        elif dt_norm.dim() == 2:
            pass
        else:
            raise ValueError(f"dt_norm should be [B] or [B,K], got {tuple(dt_norm.shape)}")

        B2, K = dt_norm.shape
        assert B2 == B, "Batch dimension mismatch between y_i and dt_norm"

        psi = self.trunk(dt_norm)        # [B, K, p]

        # Combine
        f = phi.unsqueeze(1) * psi       # [B, K, p]
        y_hat = self.out(f)              # [B, K, S]

        if self.predict_delta:
            y_hat = y_hat + y_i.unsqueeze(1)  # residual prediction

        return y_hat


# ------------------------------ factory function -----------------------------

def create_model(cfg) -> FlowMapDeepONet:
    """
    Build FlowMapDeepONet from config dict.
    Required keys:
      cfg["data"]["species_variables"] (or "target_species_variables")
      cfg["data"]["global_variables"]
    Optional keys under cfg["model"]:
      p, branch_width, branch_depth, trunk_layers (list), predict_delta, trunk_dedup
    """
    dcfg = cfg.get("data", {})
    species = list(dcfg.get("target_species_variables", dcfg.get("species_variables", [])))
    globals_ = list(dcfg.get("global_variables", []))
    if not species or globals_ is None:
        raise KeyError("Config must provide data.species_variables (or target_species_variables) and data.global_variables")

    S = len(species)
    G = len(globals_)

    mcfg = cfg.get("model", {})
    p = int(mcfg.get("p", 256))
    branch_width = int(mcfg.get("branch_width", 1024))
    branch_depth = int(mcfg.get("branch_depth", 3))
    trunk_layers = list(mcfg.get("trunk_layers", [512, 512]))
    predict_delta = bool(mcfg.get("predict_delta", True))
    trunk_dedup = bool(mcfg.get("trunk_dedup", False))
    act_name = str(mcfg.get("activation", "gelu"))

    return FlowMapDeepONet(
        S=S,
        G=G,
        p=p,
        branch_width=branch_width,
        branch_depth=branch_depth,
        trunk_layers=trunk_layers,
        predict_delta=predict_delta,
        trunk_dedup=trunk_dedup,
        act_name=act_name,
    )
