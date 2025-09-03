#!/usr/bin/env python3
"""
Export Flow-map DeepONet for inference (CPU-only), with dynamic batch size (B) and
dynamic number of query steps (K). Optionally emits an INT8-quantized variant and a
static-K=1 artifact optimized for single-jump inference on CPU.

Exported module interface (both artifacts):
    forward(y0_norm[B,S], globals_norm[B,G], dt_norm[K]) -> y_pred[B,K,S]

Notes
-----
- Trunk input is **normalized Δt** (dt-spec, same as training).
- Batch size B and K are dynamic in the primary export.
- For fastest single-jump inference on CPU, also exporting a static-K=1 model is helpful.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import json5
import torch
import torch.nn as nn
from torch.export import export as texport, save as tsave, Dim

# --------------------
# Paths / constants
# --------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parent

MODEL_STR   = "flowmap-deeponet"                 # adjust as needed
MODEL_DIR   = REPO_ROOT / "models" / MODEL_STR
MODEL_PATH  = MODEL_DIR / "best_model.pt"
CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"

# Example sizes only for tracing (keep EX_B > 1 so B isn't baked as 1)
EX_B = 2
EX_K = 64

# Toggles
APPLY_DYNAMIC_QUANT = False   # Set True to export an INT8-dynamically-quantized variant
ALSO_EXPORT_K1      = True    # Set True to additionally export a static K=1 artifact

DEVICE = torch.device("cpu")  # CPU-only export

# Project import (assumes src/ at repo root)
sys.path.append(str((REPO_ROOT / "src").resolve()))
from model import create_model  # builds FlowMapDeepONet from cfg

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --------------------
# Checkpoint helpers
# --------------------
_PREFIXES = ("_orig_mod.", "module.", "model.", "_orig_mod.module.")

def _strip_prefix(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k

def _looks_like_state_dict(d) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    seen = 0
    tensorish = 0
    for v in d.values():
        seen += 1
        if torch.is_tensor(v) or hasattr(v, "shape"):
            tensorish += 1
        if seen >= 20:
            break
    return tensorish >= max(1, seen // 2)

def _extract_state_dict(ckpt_obj):
    """
    Accepts many common formats:

    - {'model_state_dict': {...}} / {'state_dict': {...}}
    - {'model': nn.Module} / {'model': {'...': tensor}}
    - {'ema_state_dict': {...}} / {'module': {...}} / {'net': {...}} / {'weights': {...}}
    - raw state_dict mapping (param_name -> tensor)
    - an nn.Module with .state_dict()

    Returns a mapping suitable for load_state_dict, or None.
    """
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "ema_state_dict", "module", "net", "weights"):
            v = ckpt_obj.get(key)
            if v is None:
                continue
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v
            if hasattr(v, "state_dict"):
                return v.state_dict()
        v = ckpt_obj.get("model")
        if v is not None:
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v
            if hasattr(v, "state_dict"):
                return v.state_dict()
        if _looks_like_state_dict(ckpt_obj):
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    return None

def _remap_state_dict(model: nn.Module, raw):
    """Strip common prefixes and drop unknown keys."""
    want = set(model.state_dict().keys())
    out = {}
    items = raw.items() if isinstance(raw, dict) else raw.state_dict().items()
    for k, v in items:
        k2 = _strip_prefix(k)
        if k2 in want:
            out[k2] = v
    return out

# --------------------
# Flow-map wrapper
# --------------------
class FlowMapInferenceWrapper(nn.Module):
    """Vectorizes over K Δt values in one call; returns [B,K,S].
    Expects **normalized Δt** (dt-spec) as trunk input.
    """
    def __init__(self, model: nn.Module, s: int, g: int):
        super().__init__()
        self.model = model.eval()
        self.S = int(s)
        self.G = int(g)

    def forward(self, y0_norm: torch.Tensor, globals_norm: torch.Tensor, dt_norm: torch.Tensor) -> torch.Tensor:
        # y0_norm:      [B,S]
        # globals_norm: [B,G]
        # dt_norm:      [K] or [K,1] (normalized via dt-spec)
        if dt_norm.dim() == 2 and dt_norm.shape[1] == 1:
            dt_norm = dt_norm.squeeze(-1)  # [K]
        if dt_norm.dim() != 1:
            raise RuntimeError("dt_norm must be [K] or [K,1].")
        if y0_norm.dim() != 2 or globals_norm.dim() != 2:
            raise RuntimeError("y0_norm/globals_norm must be [B,S] / [B,G].")
        if y0_norm.shape[0] != globals_norm.shape[0]:
            raise RuntimeError("Batch size mismatch between y0_norm and globals_norm.")
        if y0_norm.shape[1] != self.S or globals_norm.shape[1] != self.G:
            raise RuntimeError(f"Expected S={self.S}, G={self.G}; got S={y0_norm.shape[1]}, G={globals_norm.shape[1]}")

        B, K = y0_norm.shape[0], dt_norm.shape[0]

        # Fast path for K=1 (your inference case): avoid even expand/view math
        if K == 1:
            dt_flat = dt_norm.expand(B)                     # [B]
            g_flat  = globals_norm if self.G > 0 else globals_norm.new_zeros((B, 0))
            y1_flat = self.model(y0_norm, dt_flat, g_flat)  # [B,S]
            return y1_flat.unsqueeze(1)                     # [B,1,S]

        # General path (vectorized across K without extra memory)
        y0_flat = y0_norm.unsqueeze(1).expand(B, K, self.S).reshape(B * K, self.S)
        g_flat  = (globals_norm.unsqueeze(1).expand(B, K, self.G).reshape(B * K, self.G)
                   if self.G > 0 else globals_norm.new_zeros((B * K, 0)))
        dt_flat = dt_norm.unsqueeze(0).expand(B, K).reshape(B * K)  # [B*K]
        y1_flat = self.model(y0_flat, dt_flat, g_flat)              # [B*K, S]
        return y1_flat.view(B, K, self.S)

def _maybe_dynamic_quantize(m: nn.Module) -> nn.Module:
    if not APPLY_DYNAMIC_QUANT:
        return m
    # Choose engine: "fbgemm" (x86) or "qnnpack" (ARM/Apple)
    try:
        import platform
        is_arm = any(s in platform.machine().lower() for s in ("arm", "aarch", "apple"))
        torch.backends.quantized.engine = "qnnpack" if is_arm else "fbgemm"
    except Exception:
        pass
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and build model (CPU)
    with open(CONFIG_PATH, "r") as f:
        cfg = json5.load(f)

    base_model = create_model(cfg).to(DEVICE).eval()

    # Load checkpoint (robust)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    raw_sd = _extract_state_dict(ckpt)
    if raw_sd is None:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else "n/a"
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)} (keys={keys})")

    clean_sd = _remap_state_dict(base_model, raw_sd)
    missing, unexpected = base_model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {sorted(missing)[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {sorted(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

    # Shapes
    S = len(cfg["data"]["species_variables"])
    G = len(cfg["data"]["global_variables"])

    # Optionally create a quantized copy for faster CPU inference
    model = _maybe_dynamic_quantize(base_model)

    # ---- Primary dynamic (B, K) export ----
    wrapper = FlowMapInferenceWrapper(model, S, G).to(DEVICE)

    ex_y0 = torch.randn(EX_B, S, device=DEVICE, dtype=torch.float32)
    ex_g  = torch.randn(EX_B, G, device=DEVICE, dtype=torch.float32) if G > 0 else torch.zeros(EX_B, 0)
    ex_dt = torch.linspace(0.0, 1.0, steps=EX_K, device=DEVICE, dtype=torch.float32)  # normalized dt

    # Dynamic shapes: B shared by y0 and g; K free for dt
    dyn = (
        {0: Dim("B")},  # y0_norm[B,S]
        {0: Dim("B")},  # globals_norm[B,G]
        {0: Dim("K")},  # dt_norm[K]
    )

    prog = texport(wrapper, args=(ex_y0, ex_g, ex_dt), dynamic_shapes=dyn, strict=False)

    out_path = MODEL_DIR / ("complete_model_exported_int8.pt2" if APPLY_DYNAMIC_QUANT else "complete_model_exported.pt2")
    tsave(prog, str(out_path))
    print(f"[OK] Exported dynamic B,K -> {out_path}")

    # Smoke test with two different (B,K)
    from torch.export import load as tload
    m = tload(str(out_path)).module()
    with torch.inference_mode():
        yA = m(torch.randn(1, S), torch.randn(1, G) if G > 0 else torch.zeros(1, 0), torch.linspace(0, 1, 1))
        yB = m(torch.randn(3, S), torch.randn(3, G) if G > 0 else torch.zeros(3, 0), torch.linspace(0, 1, 7))
    print(f"[OK] Dynamic test calls: shapes {tuple(yA.shape)} and {tuple(yB.shape)}")

    # ---- Optional static K=1 artifact (fastest single-jump CPU inference) ----
    if ALSO_EXPORT_K1:
        k1_dt = torch.tensor([0.5], dtype=torch.float32)  # any normalized dt; shape [1]
        dyn_k1 = (
            {0: Dim("B")},  # y0_norm[B,S]
            {0: Dim("B")},  # globals_norm[B,G]
            None,           # dt_norm fixed to length 1 (static)
        )
        prog_k1 = texport(wrapper, args=(ex_y0, ex_g, k1_dt), dynamic_shapes=dyn_k1, strict=False)
        out_path_k1 = MODEL_DIR / ("complete_model_exported_k1_int8.pt2" if APPLY_DYNAMIC_QUANT else "complete_model_exported_k1.pt2")
        tsave(prog_k1, str(out_path_k1))
        print(f"[OK] Exported static K=1 -> {out_path_k1}")
        mk1 = tload(str(out_path_k1)).module()
        with torch.inference_mode():
            yK1 = mk1(torch.randn(5, S), torch.randn(5, G) if G > 0 else torch.zeros(5, 0), torch.tensor([0.25]))
        print(f"[OK] K=1 test call: shape {tuple(yK1.shape)}")

if __name__ == "__main__":
    main()
