#!/usr/bin/env python3
"""
Flow-map Koopman Autoencoder Export Script
==========================================
Export trained models to torch.export format for deployment.

Generates:
- CPU export (fp32)
- GPU export (CUDA bf16/fp32 or MPS fp32) if available

Run:
    python testing/export.py --model koopman-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# ================================ Configuration ================================

DEFAULT_MODEL_NAME = "autoencoder-flowmap"
OUT_CPU = "export_k1_cpu.pt2"
OUT_GPU = "export_k1_gpu.pt2"
WRITE_META = True

# GPU dtype preferences
USE_BF16_ON_CUDA = True
FORCE_FP32_ON_MPS = True  # Required: MPS + fp16/bf16 causes issues

# Enable TF32 on CUDA for performance
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ================================ Path Setup ==================================

SCRIPT_PATH = Path(__file__).resolve()
TESTING_DIR = SCRIPT_PATH.parent
REPO_ROOT = TESTING_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Add src to path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set working directory
os.chdir(REPO_ROOT)

# Import after path setup
from model import create_model
from utils import load_json


# ================================ Helper Functions =============================

def write_metadata(artifact_path: Path, metadata: Dict[str, Any]) -> None:
    """Write export metadata as JSON."""
    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"  Metadata → {meta_path.name}")


def load_checkpoint(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load config and checkpoint from model directory."""
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    return config, checkpoint


def resolve_processed_dir(config: Dict[str, Any]) -> Path:
    """Get processed data directory from config."""
    paths = config.get("paths", {})
    processed_str = paths.get("processed_data_dir")

    if not processed_str:
        raise KeyError("config['paths']['processed_data_dir'] is missing")

    processed_dir = Path(processed_str).expanduser().resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    return processed_dir


def hydrate_data_config(config: Dict[str, Any], processed_dir: Path) -> None:
    """
    Ensure config['data'] contains species_variables and global_variables.
    Uses normalization.json from processed data as source of truth.
    """
    data_cfg = config.setdefault("data", {})
    species = list(data_cfg.get("species_variables") or [])
    globals_vars = list(data_cfg.get("global_variables") or [])

    # If missing, load from normalization.json
    if not species or not globals_vars:
        norm_path = processed_dir / "normalization.json"
        if norm_path.exists():
            manifest = load_json(norm_path)
            meta = manifest.get("meta", {})
            species = species or list(meta.get("species_variables") or [])
            globals_vars = globals_vars or list(meta.get("global_variables") or [])

    # Fallback to preprocessing_summary.json
    if not species or not globals_vars:
        summary_path = processed_dir / "preprocessing_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            species = species or list(summary.get("species_variables") or [])
            globals_vars = globals_vars or list(summary.get("global_variables") or [])

    if not species:
        raise KeyError("species_variables not found in config or processed artifacts")

    data_cfg["species_variables"] = species
    data_cfg["global_variables"] = globals_vars


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common wrapper prefixes from state dict keys."""

    def strip_prefix(key: str) -> str:
        for prefix in ("_orig_mod.", "module.", "model."):
            if key.startswith(prefix):
                return strip_prefix(key[len(prefix):])
        return key

    return {strip_prefix(k): v for k, v in state_dict.items()}


def build_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build and load trained model."""
    config, checkpoint = load_checkpoint(model_dir)
    processed_dir = resolve_processed_dir(config)
    hydrate_data_config(config, processed_dir)

    # Create model
    model = create_model(config)

    # Extract state dict from checkpoint
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

    if state_dict is None and isinstance(checkpoint, dict):
        # Assume checkpoint is the state dict
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Cannot find valid state_dict in checkpoint")

    # Load weights
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(device).eval()

    return model, config


# ================================ Export Functions =============================

def verify_export_available() -> None:
    """Check if torch.export is available."""
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError(
            "torch.export is not available. "
            "Please install PyTorch >= 2.1 (preferably >= 2.3)"
        )


def export_model(
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        output_path: Path
) -> Dict[str, Any]:
    """
    Export model for K=1 (single timestep) with dynamic batch size.

    Args:
        model: Trained model
        config: Configuration dict
        device: Target device
        dtype: Target dtype
        output_path: Output file path

    Returns:
        Export metadata dict
    """
    model = model.to(device=device, dtype=dtype).eval()

    # Extract dimensions
    data_cfg = config.get("data", {})
    S_in = getattr(model, "S_in", len(data_cfg.get("species_variables", [])))
    S_out = getattr(model, "S_out", S_in)
    G = getattr(model, "G_in", len(data_cfg.get("global_variables", [])))

    if S_in == 0:
        raise ValueError("Cannot determine S_in (species dimension)")

    # Create dummy inputs for export (K=1)
    batch_size = 2
    y_i = torch.randn(batch_size, S_in, device=device, dtype=dtype)
    dt_norm = torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)
    g = torch.randn(batch_size, G, device=device, dtype=dtype) if G > 0 else \
        torch.zeros(batch_size, 0, device=device, dtype=dtype)

    # Export with dynamic batch dimension
    from torch.export import export as torch_export, Dim

    batch_dim = Dim("batch")
    dynamic_shapes = (
        {0: batch_dim},  # y_i: [B, S_in]
        {0: batch_dim},  # dt_norm: [B, K]
        {0: batch_dim},  # g: [B, G]
    )

    print(f"  Exporting with dimensions: S_in={S_in}, S_out={S_out}, G={G}")
    exported_program = torch_export(model, (y_i, dt_norm, g), dynamic_shapes=dynamic_shapes)

    # Save exported program
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, output_path)

    # Build metadata
    metadata = {
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "S_in": int(S_in),
        "S_out": int(S_out),
        "G": int(G),
        "K": 1,
        "file": str(output_path.name),
        "pytorch_version": torch.__version__,
    }

    print(f"  ✓ Exported → {output_path.name}")
    print(f"    Device: {metadata['device']}, Dtype: {metadata['dtype']}")

    return metadata


# ================================ Main ========================================

def main() -> None:
    """Main export pipeline."""
    parser = argparse.ArgumentParser(
        description="Export Flow-map Koopman Autoencoder for deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Export CPU version only (skip GPU)"
    )
    args = parser.parse_args()

    # Verify torch.export availability
    verify_export_available()

    # Resolve model directory
    model_dir = (REPO_ROOT / "models" / args.model).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"\n{'=' * 70}")
    print(f"Exporting model: {args.model}")
    print(f"Model directory: {model_dir}")
    print(f"{'=' * 70}\n")

    # Load model on CPU
    print("Loading model...")
    base_model, config = build_model(model_dir, torch.device("cpu"))
    print("  ✓ Model loaded\n")

    # Export CPU version (fp32)
    print("Exporting CPU version (fp32)...")
    cpu_output = model_dir / OUT_CPU
    cpu_metadata = export_model(
        base_model,
        config,
        torch.device("cpu"),
        torch.float32,
        cpu_output
    )

    if WRITE_META:
        write_metadata(cpu_output, cpu_metadata)
    print()

    # Export GPU version if available and requested
    if not args.cpu_only:
        gpu_metadata = None

        if torch.cuda.is_available():
            print("Exporting CUDA version...")
            dtype = torch.bfloat16 if USE_BF16_ON_CUDA else torch.float32
            dtype_str = "bf16" if USE_BF16_ON_CUDA else "fp32"
            print(f"  Using dtype: {dtype_str}")

            gpu_output = model_dir / OUT_GPU
            gpu_metadata = export_model(
                base_model,
                config,
                torch.device("cuda"),
                dtype,
                gpu_output
            )

        elif torch.backends.mps.is_available():
            print("Exporting MPS version (fp32)...")
            # MPS requires fp32
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            gpu_output = model_dir / OUT_GPU
            gpu_metadata = export_model(
                base_model,
                config,
                torch.device("mps"),
                torch.float32,
                gpu_output
            )
        else:
            print("No GPU backend (CUDA/MPS) available - skipping GPU export")

        if gpu_metadata and WRITE_META:
            write_metadata(gpu_output, gpu_metadata)
            print()

    print(f"{'=' * 70}")
    print("Export complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()