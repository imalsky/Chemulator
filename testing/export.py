#!/usr/bin/env python3
"""
FlowMap Model Export Script (minimal)
- CPU:  K=1, dynamic batch
- CUDA: dynamic B,K
- MPS:  dynamic B,K
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Paths / basic config
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WORK_DIR = ROOT / "models" / "v1_long"
CONFIG_PATH = WORK_DIR / "config.json"

CPU_OUT = WORK_DIR / "export_k1_cpu.pt2"
GPU_OUT = WORK_DIR / "export_bk_gpu.pt2"
MPS_OUT = WORK_DIR / "export_bk_mps.pt2"

MIN_BATCH, MAX_BATCH = 1, 4096
MIN_K, MAX_K = 1, 1024

os.chdir(ROOT)
sys.path.insert(0, str(SRC))

from model import create_model  # type: ignore

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def parse_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get(dtype_str.lower(), torch.float32)


def find_ckpt(directory: Path) -> Path:
    d = Path(directory)
    if (d / "best.ckpt").exists():
        return d / "best-v1.ckpt"
    if (d / "last.ckpt").exists():
        return d / "last.ckpt"
    ckpts = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint found in {d}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        state = (
            payload.get("state_dict")
            or payload.get("model_state_dict")
            or payload.get("model")
            or payload.get("ema_model")
            or {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        )
    else:
        state = payload
    clean = {}
    for k, v in state.items():
        kk = k
        for prefix in ("model.", "module.", "_orig_mod."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        clean[kk] = v
    model.load_state_dict(clean, strict=False)


def optimize_inference(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


# --------------------------------------------------------------------------------------
# Export routines
# --------------------------------------------------------------------------------------

def export_cpu_k1(base: nn.Module) -> None:
    print("\n" + "=" * 80)
    print("Exporting CPU (K=1, dynamic B)")
    print("=" * 80)

    device = "cpu"
    dtype = parse_dtype("float32")
    model = optimize_inference(base.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)

    B = 2
    y = torch.zeros(B, S_in, dtype=dtype, device=device)  # [B,S]
    dt = torch.zeros(B, 1, dtype=dtype, device=device)    # [B,1] (K=1)
    g = torch.zeros(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)

    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=(
            {0: Bdim},   # y: [B,S]
            {0: Bdim},   # dt: [B,1]
            {0: Bdim},   # g: [B,G]
        ),
    )

    CPU_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, CPU_OUT)
    print(f"  wrote {CPU_OUT}")


def export_device_dynBK(base: nn.Module, device: str, out_path: Path, dtype_str: str) -> None:
    pretty = "GPU (CUDA)" if device == "cuda" else "MPS (Apple Silicon)"
    print("\n" + "=" * 80)
    print(f"Exporting {pretty} (dynamic B,K)")
    print("=" * 80)

    dtype = parse_dtype(dtype_str)
    model = optimize_inference(base.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)
    Kdim = torch.export.Dim("K", min=MIN_K, max=MAX_K)

    B, K = 2, 2
    y = torch.randn(B, S_in, dtype=dtype, device=device)  # [B,S]
    dt = torch.randn(B, K, 1, dtype=dtype, device=device) # [B,K,1]
    g = torch.randn(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)

    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=(
            {0: Bdim},         # y: [B,S]
            {0: Bdim, 1: Kdim},# dt: [B,K,1]
            {0: Bdim},         # g: [B,G]
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("FlowMap Export: CPU K=1 + GPU/MPS dynamic-K")
    print("=" * 80)
    print(f"Config path: {CONFIG_PATH}")

    cfg_json = json.loads(CONFIG_PATH.read_text())
    base = create_model(cfg_json).eval().cpu()

    ckpt = find_ckpt(WORK_DIR)
    print(f"Loading checkpoint: {ckpt}")
    load_weights(base, ckpt)

    export_cpu_k1(base)

    if torch.cuda.is_available():
        export_device_dynBK(base, "cuda", GPU_OUT, "bfloat16")
    else:
        print("[note] CUDA not available; skipping GPU export")

    if torch.backends.mps.is_available():
        export_device_dynBK(base, "mps", MPS_OUT, "float32")
    else:
        print("[note] MPS not available; skipping MPS export")

    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
