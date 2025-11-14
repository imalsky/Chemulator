#!/usr/bin/env python3
"""
FlowMap Model Export Script (new-model compatible)
==================================================
Exports FlowMapAutoencoder models to:
- CPU: K=1, dynamic batch (for xi.py)
- GPU/MPS: Dynamic B and K (native model interface: y[B,S], dt[B,K,1], g[B,G])
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (may be used by Torch export)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

# Project root assumed one level above this file's parent (…/<repo>/{src,models,...})
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WORK_DIR = ROOT / "models" / "2_layer"
CONFIG_PATH = WORK_DIR / "config.json"

# Export artifacts
CPU_OUT = WORK_DIR / "export_k1_cpu.pt2"       # K=1, dynamic B
GPU_OUT = WORK_DIR / "export_bk_gpu.pt2"       # dynamic B,K
MPS_OUT = WORK_DIR / "export_bk_mps.pt2"       # dynamic B,K

GPU_AOTI_DIR = WORK_DIR / "export_bk_gpu.aoti"
MPS_AOTI_DIR = WORK_DIR / "export_bk_mps.aoti"

# Make src importable
os.chdir(ROOT)
sys.path.insert(0, str(SRC))

from model import create_model  # type: ignore


# Register safe globals for torch.load
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class ExportConfig:
    # Dynamic shapes
    min_batch: int = 1
    max_batch: int = 4096
    min_k: int = 1
    max_k: int = 1024

    # Example sizes for validation
    eg_batch: int = 256
    eg_k: int = 8

    # Dtypes per device
    cpu_dtype: str = "float32"
    cuda_dtype: str = "bfloat16"
    mps_dtype: str = "float32"

    # Validation / compile
    run_validation: bool = True
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"


CFG = ExportConfig()


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def parse_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get(dtype_str.lower(), torch.float32)


def find_ckpt(directory: Path) -> Path:
    """
    Find the best available checkpoint in `directory`.

    Priority:
      1) work_dir/best.ckpt
      2) work_dir/checkpoints/best.ckpt
      3) work_dir/best_model.pt (legacy)
      4) lowest val among epoch*.ckpt in work_dir/checkpoints
      5) work_dir/last.ckpt then checkpoints/last.ckpt
      6) most recent *.ckpt (work_dir or checkpoints/)
      7) most recent *.pt (work_dir)
    """
    d = Path(directory)

    def first_existing(paths):
        for p in paths:
            if p is not None and p.exists():
                return p
        return None

    # explicit best
    p = first_existing([d / "best.ckpt", d / "checkpoints" / "best.ckpt"])
    if p:
        return p

    # legacy explicit best
    legacy_best = d / "best_model.pt"
    if legacy_best.exists():
        return legacy_best

    # epoch*.ckpt with val in name
    ckdir = d / "checkpoints"
    epoch_ckpts = []
    if ckdir.exists():
        for q in ckdir.glob("epoch*.ckpt"):
            m = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", q.name)
            if m:
                epoch = int(m.group(1))
                try:
                    val = float(m.group(2))
                except Exception:
                    val = float("inf")
                epoch_ckpts.append((val, -epoch, q))
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda t: (t[0], t[1]))
        return epoch_ckpts[0][2]

    # last.ckpt fallbacks
    p = first_existing([d / "last.ckpt", ckdir / "last.ckpt"])
    if p:
        return p

    # any .ckpt (newest)
    any_ckpts = list(d.glob("*.ckpt")) + (list(ckdir.glob("*.ckpt")) if ckdir.exists() else [])
    if any_ckpts:
        any_ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return any_ckpts[0]

    # any .pt (newest)
    any_pts = sorted(d.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if any_pts:
        return any_pts[0]

    raise FileNotFoundError(f"No checkpoint found in {d}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    """
    Load weights handling common checkpoint layouts.
    """
    payload = torch.load(ckpt_path, map_location="cpu")
    state = (
        payload.get("state_dict")
        or payload.get("model_state_dict")
        or payload.get("model")
        or payload.get("ema_model")
        or {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    )
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
    # Make Dropout a hard no-op (eval already disables it; setting p=0 removes tiny overhead)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


def _ensure_autoencoder_only(model: nn.Module) -> None:
    """
    Guardrail: error out if the model is configured as a VAE.
    Checks common flags found in the Encoder and at top-level if present.
    """
    flags = []
    if hasattr(model, "vae_mode"):
        try:
            flags.append(bool(getattr(model, "vae_mode")))
        except Exception:
            pass
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "vae_mode"):
        try:
            flags.append(bool(getattr(enc, "vae_mode")))
        except Exception:
            pass
    if any(flags):
        raise RuntimeError("VAE mode detected. This exporter supports autoencoder-only. Disable 'vae_mode' in your config.")


# --------------------------------------------------------------------------------------
# AOTI support
# --------------------------------------------------------------------------------------

def emit_aoti_for(ep, device: str, example_inputs, target_dir: Path) -> None:
    """
    Try to create an AOTI package (CUDA/MPS only). Best-effort across versions.
    """
    if device not in ("cuda", "mps"):
        return
    try:
        mod = ep.module()
        from torch._inductor import aot_compile  # type: ignore[attr-defined]
        aoti_pkg = aot_compile(mod, example_inputs)

        for method_name in (
            "save", "save_packaged_artifact", "save_to_path",
            "write_to_dir", "write_to_file", "export", "dump",
        ):
            if hasattr(aoti_pkg, method_name):
                method = getattr(aoti_pkg, method_name)
                try:
                    method(str(target_dir))
                    print(f"  AOTI package saved: {target_dir}")
                    return
                except TypeError:
                    try:
                        method(path=str(target_dir))
                        print(f"  AOTI package saved: {target_dir}")
                        return
                    except Exception:
                        pass
                except Exception:
                    pass

        if isinstance(aoti_pkg, (str, os.PathLike)):
            src = Path(aoti_pkg)
            if src.exists():
                if src.is_file():
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target_dir)
                else:
                    shutil.copytree(src, target_dir, dirs_exist_ok=True)
                print(f"  AOTI package saved: {target_dir}")
                return

        for attr in ("path", "output_path", "artifact_path", "dir", "directory"):
            if hasattr(aoti_pkg, attr):
                src = Path(getattr(aoti_pkg, attr))
                if src.exists():
                    if src.is_file():
                        target_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, target_dir)
                    else:
                        shutil.copytree(src, target_dir, dirs_exist_ok=True)
                    print(f"  AOTI package saved: {target_dir}")
                    return

        print("  [warn] AOTI packaging: unrecognized object; skipped")
    except Exception as e:
        print(f"  [warn] AOTI packaging failed: {e}")


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate_ep(ep, device: str, dtype: torch.dtype, S_in: int, G: int, dyn_k: bool) -> None:
    """
    Validate the exported program with shapes that match how it was traced.

    NEW model interface during export:
      - CPU K=1 path was traced with dt of shape [B, 1] (2D)
      - GPU/MPS dyn-K path was traced with dt of shape [B, K, 1] (3D)
    """
    if not CFG.run_validation:
        return

    B = CFG.eg_batch
    K = CFG.eg_k if dyn_k else 1
    td = dtype

    # Inputs
    y = torch.randn(B, S_in, dtype=td, device=device)  # [B, S]
    if dyn_k:
        # Must match GPU/MPS export rank: [B, K, 1]
        dt = torch.randn(B, K, 1, dtype=td, device=device)  # [B, K, 1]
    else:
        # Must match CPU K=1 export rank: [B, 1]
        dt = torch.randn(B, 1, dtype=td, device=device)     # [B, 1]
    g = (
        torch.randn(B, G, dtype=td, device=device)
        if G > 0 else torch.empty(B, 0, dtype=td, device=device)
    )  # [B, G]

    # Optional compile
    mod = ep.module()
    try:
        mod = torch.compile(mod, mode=CFG.compile_mode)  # type: ignore[arg-type]
    except Exception as e:
        print(f"  [note] compile skipped: {e}")

    with torch.inference_mode():
        out = mod(y, dt, g)

    # Shape checks
    if not isinstance(out, torch.Tensor) or out.dim() != 3 or out.size(0) != B or out.size(1) != K:
        raise RuntimeError(
            f"Validation shape mismatch: expected [B={B}, K={K}, S_out], got {tuple(out.shape)}"
        )
    print(f"  Validation OK ({'dyn-K' if dyn_k else 'K=1'}): out shape = {tuple(out.shape)}")



# --------------------------------------------------------------------------------------
# Export routines (no monkey-patching; use the new model as-is)
# --------------------------------------------------------------------------------------

def export_cpu_k1(base: nn.Module) -> None:
    """
    Export K=1 CPU variant (dynamic batch). Uses dt shape [B,1].
    """
    print("\n" + "=" * 80)
    print("Exporting CPU (K=1, dynamic B)")
    print("=" * 80)

    dev = "cpu"
    td = parse_dtype(CFG.cpu_dtype)

    model = optimize_inference(base.to(dev))

    # Introspect dims
    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    # Dynamic batch symbol
    Bdim = torch.export.Dim("batch", min=CFG.min_batch, max=CFG.max_batch)

    # Example inputs
    y = torch.zeros(2, S_in, dtype=td, device=dev)   # [B,S]
    dt = torch.zeros(2, 1, dtype=td, device=dev)     # [B,1]  -> K=1
    g = torch.zeros(2, G, dtype=td, device=dev) if G > 0 else torch.empty(2, 0, dtype=td, device=dev)

    # Export with dynamic batch
    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=({0: Bdim}, {0: Bdim}, {0: Bdim}),
    )

    CPU_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, CPU_OUT)
    print(f"  wrote {CPU_OUT}")

    validate_ep(ep, dev, td, S_in, G, dyn_k=False)


def export_device_dynBK(base: nn.Module, device: str, out_path: Path, aoti_dir: Path, dtype_str: str) -> None:
    """
    Export GPU/MPS variant with dynamic B and K.
    NEW model interface: y[B,S], dt[B,K,1], g[B,G]
    """
    print("\n" + "=" * 80)
    pretty = "GPU (CUDA)" if device == "cuda" else "MPS (Apple Silicon)"
    print(f"Exporting {pretty} (dynamic B,K)")
    print("=" * 80)

    td = parse_dtype(dtype_str)
    model = optimize_inference(base.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    # Dynamic dims
    Bdim = torch.export.Dim("batch", min=CFG.min_batch, max=CFG.max_batch)
    Kdim = torch.export.Dim("K", min=CFG.min_k, max=CFG.max_k)

    # Example tensors
    B, K = CFG.eg_batch, CFG.eg_k
    y = torch.randn(B, S_in, dtype=td, device=device)             # [B,S]
    dt = torch.randn(B, K, 1, dtype=td, device=device)            # [B,K,1]
    g = torch.randn(B, G, dtype=td, device=device) if G > 0 else torch.empty(B, 0, dtype=td, device=device)  # [B,G]

    # Export with dynamic shapes on B and K (K only on dt)
    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=(
            {0: Bdim},            # y: [B,S]
            {0: Bdim, 1: Kdim},   # dt: [B,K,1]
            {0: Bdim},            # g: [B,G]
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  wrote {out_path}")

    # AOTI package (best-effort)
    try:
        emit_aoti_for(ep, device, (y, dt, g), aoti_dir)
    except Exception as e:
        print(f"  [warn] AOTI emission failed: {e}")

    validate_ep(ep, device, td, S_in, G, dyn_k=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("FlowMap Export: CPU K=1 + GPU/MPS dynamic-K (new-model)")
    print("=" * 80)
    print(f"Config path: {CONFIG_PATH}")

    # Load config and build model
    cfg_json = json.loads(CONFIG_PATH.read_text())
    base = create_model(cfg_json).eval().cpu()

    # Autoencoder-only guard (fail fast if VAE enabled)
    _ensure_autoencoder_only(base)

    # Find and load checkpoint
    ckpt = find_ckpt(WORK_DIR)
    print(f"Loading checkpoint: {ckpt}")
    load_weights(base, ckpt)

    # CPU export
    export_cpu_k1(base)

    # CUDA export
    if torch.cuda.is_available():
        try:
            export_device_dynBK(base, "cuda", GPU_OUT, GPU_AOTI_DIR, CFG.cuda_dtype)
        except Exception as e:
            print(f"[warn] CUDA export failed: {e}")
    else:
        print("[note] CUDA not available; skipping GPU export")

    # MPS export
    if torch.backends.mps.is_available():
        try:
            export_device_dynBK(base, "mps", MPS_OUT, MPS_AOTI_DIR, CFG.mps_dtype)
        except Exception as e:
            print(f"[warn] MPS export failed: {e}")
    else:
        print("[note] MPS not available; skipping MPS export")

    print("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
