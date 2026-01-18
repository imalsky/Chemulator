#!/usr/bin/env python3
"""
FlowMap Model Export Script

Exports trained FlowMap models to torch.export format for inference.
Configure the MODEL_DIR and DEVICES globals below, then run directly.

Exported models can be loaded with:
    ep = torch.export.load("export_cpu.pt2")
    model = ep.module()
    output = model(y_input, dt_input, g_input)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

# =============================================================================
# Configuration — Edit these values
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODEL_DIR = ROOT / "models" / "mlp_paper_pf_steps96"

# Devices to export: "cpu", "cuda", "mps"
DEVICES = ["cpu", "cuda", "mps"]

# Dynamic shape bounds
MIN_BATCH, MAX_BATCH = 1, 4096
MIN_K, MAX_K = 1, 1024

# =============================================================================
# Device-specific settings
# =============================================================================

DEVICE_SETTINGS = {
    "cpu": {
        "dtype": torch.float32,
        "filename": "export_cpu.pt2",
        "dynamic_k": False,  # K=1 for simpler CPU inference
    },
    "cuda": {
        "dtype": torch.bfloat16,
        "filename": "export_cuda.pt2",
        "dynamic_k": True,
    },
    "mps": {
        "dtype": torch.float32,  # MPS has limited bfloat16 support
        "filename": "export_mps.pt2",
        "dynamic_k": True,
    },
}


# =============================================================================
# Model Loading
# =============================================================================

def find_checkpoint(directory: Path) -> Path:
    """Find best available checkpoint in directory."""
    for name in ("best.ckpt", "last.ckpt"):
        if (directory / name).exists():
            return directory / name

    for pattern in ("*.ckpt", "*.pt"):
        candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"No checkpoint found in {directory}")


def load_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Load weights from checkpoint, handling various formats and prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict from checkpoint
    if isinstance(checkpoint, dict):
        state = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("model")
                or {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        )
    else:
        state = checkpoint

    # Strip wrapper prefixes (Lightning, DataParallel, torch.compile)
    cleaned = {}
    for key, value in state.items():
        for prefix in ("model.", "module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning - missing keys: {missing}")
    if unexpected:
        print(f"  Warning - unexpected keys: {unexpected}")


def prepare_for_inference(model: nn.Module) -> nn.Module:
    """Disable training-specific behavior."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


# =============================================================================
# Export
# =============================================================================

def export_model(model: nn.Module, device: str) -> Path | None:
    """Export model for a specific device. Returns output path or None if skipped."""

    # Check availability
    if device == "cuda" and not torch.cuda.is_available():
        print(f"[{device}] Skipping - CUDA not available")
        return None
    if device == "mps" and not torch.backends.mps.is_available():
        print(f"[{device}] Skipping - MPS not available")
        return None

    settings = DEVICE_SETTINGS[device]
    dtype = settings["dtype"]
    dynamic_k = settings["dynamic_k"]
    output_path = MODEL_DIR / settings["filename"]

    print(f"\n[{device}] Exporting (dtype={dtype}, dynamic_k={dynamic_k})")

    # Prepare model
    export_model = prepare_for_inference(model.to(device))
    S = int(getattr(export_model, "S"))
    G = int(getattr(export_model, "G", 0) or 0)

    # Create example inputs
    B, K = 2, 2
    y = torch.randn(B, S, dtype=dtype, device=device)
    g = torch.randn(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)

    if dynamic_k:
        dt = torch.randn(B, K, 1, dtype=dtype, device=device)
    else:
        dt = torch.randn(B, 1, dtype=dtype, device=device)

    # Define dynamic shapes
    B_dim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)

    if dynamic_k:
        K_dim = torch.export.Dim("K", min=MIN_K, max=MAX_K)
        shapes = ({0: B_dim}, {0: B_dim, 1: K_dim}, {0: B_dim})
    else:
        shapes = ({0: B_dim}, {0: B_dim}, {0: B_dim})

    # Export and save
    ep = torch.export.export(export_model, (y, dt, g), dynamic_shapes=shapes)
    torch.export.save(ep, output_path)

    print(f"[{device}] Saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Setup paths
    os.chdir(ROOT)
    sys.path.insert(0, str(SRC))

    # Allow Path objects in checkpoints
    try:
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    except AttributeError:
        pass

    print("=" * 60)
    print("FlowMap Export")
    print("=" * 60)
    print(f"Model dir: {MODEL_DIR}")

    from model import create_model

    # Load model
    config = json.loads((MODEL_DIR / "config.json").read_text())
    model = create_model(config)

    checkpoint = find_checkpoint(MODEL_DIR)
    print(f"Checkpoint: {checkpoint.name}")
    load_weights(model, checkpoint)

    # Export for each device
    exported = []
    for device in DEVICES:
        path = export_model(model, device)
        if path:
            exported.append(path)

    # Summary
    print("\n" + "=" * 60)
    if exported:
        print("Exported:")
        for p in exported:
            print(f"  {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("No models exported.")
    print("Done.")


if __name__ == "__main__":
    main()
