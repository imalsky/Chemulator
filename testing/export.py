#!/usr/bin/env python3
"""
Export DeepONet (standard or flow-map) for inference.

Interface (both modes):
    forward(x0_norm[B,S], globals_norm[B,G], t_or_dt[K]) -> y_pred[B,K,S]

- Standard mode: t_or_dt = absolute times
- Flow-map mode:  t_or_dt = Δt (physical, > 0)
"""

import os, sys, json
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.export import export as texport, save as tsave, Dim

# ---- Config ----
MODEL_STR   = "big_deepo"            # adjust as needed
MODEL_DIR   = Path("../models") / MODEL_STR
MODEL_PATH  = MODEL_DIR / "best_model.pt"
CONFIG_PATH = MODEL_DIR / "config.json"
EX_TRUNK_STEPS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Project import (assumes this file lives in a sibling dir to src/)
sys.path.append(str((Path(__file__).resolve().parent.parent / "src").resolve()))
from model import create_model  # uses your config to build DeepONet or FlowMapDeepONet

# ---- Checkpoint utilities ----
_PREFIXES = ("_orig_mod.", "module.", "model.", "_orig_mod.module.")

def _strip_prefix(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k

def _remap_state_dict(model: nn.Module, raw):
    want = set(model.state_dict().keys())
    out = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        # Fallback: torch.nn.Module.state_dict() object
        items = raw.state_dict().items()
    for k, v in items:
        k2 = _strip_prefix(k)
        if k2 in want:
            out[k2] = v
    return out

# ---- Inference wrappers ----
class StandardInferenceWrapper(nn.Module):
    """Wraps standard DeepONet: branch takes [x0||globals], trunk takes absolute times."""
    def __init__(self, model: nn.Module, s: int, g: int):
        super().__init__()
        self.model = model.eval()
        self.S = int(s); self.G = int(g)

    def forward(self, x0_norm: torch.Tensor, globals_norm: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # x0_norm: [B,S], globals_norm: [B,G], times: [K] or [K,1]
        if times.dim() == 1:
            times = times.unsqueeze(-1)           # [K,1]
        if x0_norm.dim() != 2 or globals_norm.dim() != 2:
            raise RuntimeError("x0_norm/globals_norm must be [B, S] / [B, G].")
        if x0_norm.shape[0] != globals_norm.shape[0]:
            raise RuntimeError("Batch size mismatch between x0_norm and globals_norm.")
        if x0_norm.shape[1] != self.S or globals_norm.shape[1] != self.G:
            raise RuntimeError(f"Feature mismatch: expected S={self.S}, G={self.G}, "
                               f"got S={x0_norm.shape[1]}, G={globals_norm.shape[1]}")
        branch = torch.cat([x0_norm, globals_norm], dim=-1)     # [B, S+G]
        y_pred = self.model(branch, times)                      # [B, K, S]
        return y_pred

class FlowMapInferenceWrapper(nn.Module):
    """Wraps FlowMapDeepONet to vectorize over K Δt values: returns [B,K,S]."""
    def __init__(self, model: nn.Module, s: int, g: int):
        super().__init__()
        self.model = model.eval()
        self.S = int(s); self.G = int(g)

    def forward(self, y0_norm: torch.Tensor, globals_norm: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        # y0_norm: [B,S], globals_norm: [B,G], dts: [K] or [K,1] (physical, > 0)
        if dts.dim() == 2 and dts.shape[1] == 1:
            dts = dts.squeeze(-1)                 # [K]
        if dts.dim() != 1:
            raise RuntimeError("dts must be [K] or [K,1].")
        if y0_norm.dim() != 2 or globals_norm.dim() != 2:
            raise RuntimeError("y0_norm/globals_norm must be [B, S] / [B, G].")
        if y0_norm.shape[0] != globals_norm.shape[0]:
            raise RuntimeError("Batch size mismatch between y0_norm and globals_norm.")
        if y0_norm.shape[1] != self.S or globals_norm.shape[1] != self.G:
            raise RuntimeError(f"Feature mismatch: expected S={self.S}, G={self.G}, "
                               f"got S={y0_norm.shape[1]}, G={globals_norm.shape[1]}")

        B, K = y0_norm.shape[0], dts.shape[0]
        # Tile over K in a single call: [B,K,S] -> [B*K,S], same for G and Δt
        y0_flat = y0_norm.unsqueeze(1).expand(B, K, self.S).reshape(B * K, self.S)
        g_flat  = globals_norm.unsqueeze(1).expand(B, K, self.G).reshape(B * K, self.G) if self.G > 0 else None
        dt_flat = dts.unsqueeze(0).expand(B, K).reshape(B * K)                             # [B*K]

        y1_flat = self.model(y0_flat, dt_flat, g_flat)                                     # [B*K, S]
        return y1_flat.view(B, K, self.S)                                                  # [B,K,S]

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and build model from your factory
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    # Create model on chosen device (your factory also applies dtype)
    model = create_model(cfg, DEVICE)
    model.eval()

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    raw_sd = None
    if isinstance(ckpt, dict):
        raw_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if raw_sd is None and hasattr(ckpt, "state_dict"):
        raw_sd = ckpt.state_dict()
    if raw_sd is None:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    clean_sd = _remap_state_dict(model, raw_sd)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

    # Shapes
    S = len(cfg["data"]["species_variables"])
    G = len(cfg["data"]["global_variables"])
    K = EX_TRUNK_STEPS

    # Pick wrapper by mode
    mode = cfg.get("model", {}).get("mode", "standard").lower()
    if mode == "flowmap":
        wrapper = FlowMapInferenceWrapper(model, S, G).to(DEVICE)
        # Example inputs (Δt in physical units)
        ex_x   = torch.randn(2, S, device=DEVICE, dtype=torch.float32)
        ex_g   = torch.randn(2, G, device=DEVICE, dtype=torch.float32) if G > 0 else torch.zeros(2, 0, device=DEVICE)
        ex_dt  = torch.linspace(1e-3, 1.0, steps=K, device=DEVICE, dtype=torch.float32)
        dyn = (
            {0: Dim("B", 1, 8192)},     # y0_norm
            {0: Dim("B", 1, 8192)},     # globals_norm
            {0: Dim("K", 1, 10000)},    # dts
        )
        example_args = (ex_x, ex_g, ex_dt)
    else:
        wrapper = StandardInferenceWrapper(model, S, G).to(DEVICE)
        # Example inputs (absolute, normalized times)
        ex_x   = torch.randn(2, S, device=DEVICE, dtype=torch.float32)
        ex_g   = torch.randn(2, G, device=DEVICE, dtype=torch.float32) if G > 0 else torch.zeros(2, 0, device=DEVICE)
        ex_t   = torch.linspace(0.0, 1.0, steps=K, device=DEVICE, dtype=torch.float32)
        dyn = (
            {0: Dim("B", 1, 8192)},     # x0_norm
            {0: Dim("B", 1, 8192)},     # globals_norm
            {0: Dim("K", 1, 10000)},    # times
        )
        example_args = (ex_x, ex_g, ex_t)

    # Export
    prog = texport(
        wrapper,
        args=example_args,
        dynamic_shapes=dyn,
        strict=False,
    )

    out_path = MODEL_DIR / "complete_model_exported.pt2"
    tsave(prog, str(out_path))
    print(f"[OK] Exported -> {out_path}")

    # Smoke test (DON'T call .eval() on exported program’s module)
    try:
        from torch.export import load as tload
        m = tload(str(out_path)).module()
        with torch.inference_mode():
            y = m(*example_args)
        print(f"[OK] Test call: out shape {tuple(y.shape)}")
    except Exception as e:
        print(f"[WARN] Smoke test failed: {e}")

if __name__ == "__main__":
    main()
