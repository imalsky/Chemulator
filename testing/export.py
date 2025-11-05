#!/usr/bin/env python3
"""
FlowMap Model Export Script
============================
Exports FlowMapAutoencoder models to different formats:
- CPU: K=1, dynamic batch (for inference with xi.py)
- GPU/MPS: Dynamic B,K with optional AOTI compilation
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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Path Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WORK_DIR = ROOT / "models" / "v1"
CONFIG_PATH = WORK_DIR / "config.json"

# Export artifact paths
CPU_OUT = WORK_DIR / "export_k1_cpu.pt2"  # K=1, dynamic batch (xi.py uses this)
GPU_OUT = WORK_DIR / "export_k_dyn_gpu.pt2"  # Dynamic B,K
MPS_OUT = WORK_DIR / "export_k_dyn_mps.pt2"  # Dynamic B,K

GPU_AOTI_DIR = WORK_DIR / "export_k_dyn_gpu.aoti"
MPS_AOTI_DIR = WORK_DIR / "export_k_dyn_mps.aoti"

# Setup Python path to import model code
os.chdir(ROOT)
sys.path.insert(0, str(SRC))

from model import create_model, FlowMapAutoencoder  # type: ignore

# Register safe globals for checkpoint loading (required for torch.load)
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


# ============================================================================
# Export Configuration
# ============================================================================

@dataclass
class ExportConfig:
    """Configuration for model export with dynamic shapes and validation."""

    # Dynamic shape ranges for torch.export
    min_batch: int = 1
    max_batch: int = 2048
    min_k: int = 1
    max_k: int = 1024

    # Example sizes for tracing and validation
    eg_batch: int = 256
    eg_k: int = 8

    # Data types for each device
    cpu_dtype: str = "float32"
    cuda_dtype: str = "bfloat16"
    mps_dtype: str = "float32"

    # Compilation and validation settings
    run_validation: bool = True
    compile_mode: str = "default"  # Options: "default", "reduce-overhead", "max-autotune"


CFG = ExportConfig()


# ============================================================================
# Utility Functions
# ============================================================================

def parse_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def find_ckpt(directory: Path) -> Path:
    """
    Find the best available checkpoint in the given directory.

    Priority order:
    1. best_model.pt (explicit best model)
    2. Best checkpoint by validation loss in checkpoints/ directory
    3. last.ckpt in checkpoints/
    4. Most recent .pt file by modification time

    Args:
        directory: Directory to search for checkpoints

    Returns:
        Path to the best checkpoint

    Raises:
        FileNotFoundError: If no checkpoint is found
    """
    # Check for explicit best model
    best = directory / "best_model.pt"
    if best.exists():
        return best

    # Check checkpoints directory
    ckdir = directory / "checkpoints"
    if ckdir.exists():
        # Find checkpoint with lowest validation loss
        candidates = []
        for p in ckdir.glob("epoch*.ckpt"):
            # Parse epoch and validation loss from filename
            match = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", p.name)
            if match:
                epoch = int(match.group(1))
                try:
                    val_loss = float(match.group(2))
                except Exception:
                    val_loss = float("inf")
                # Store (val_loss, -epoch, path) - negative epoch for tie-breaking
                candidates.append((val_loss, -epoch, p))

        if candidates:
            # Sort by validation loss, then by most recent epoch
            candidates.sort(key=lambda t: (t[0], t[1]))
            return candidates[0][2]

        # Fallback to last.ckpt
        last = ckdir / "last.ckpt"
        if last.exists():
            return last

    # Fallback to most recent .pt file
    pts = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]

    raise FileNotFoundError(f"No checkpoint found in {directory}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    """
    Load model weights from checkpoint, handling various checkpoint formats.

    Strips common prefixes like 'model.', 'module.', '_orig_mod.' from state dict keys.
    This handles checkpoints from different training frameworks and wrappers.

    Args:
        model: Model to load weights into
        ckpt_path: Path to checkpoint file
    """
    # Load checkpoint with weights_only=False for compatibility
    payload = torch.load(ckpt_path, map_location="cpu")

    # Extract state dict from various checkpoint formats
    # Try different common checkpoint structures
    state = (
            payload.get("state_dict") or
            payload.get("model_state_dict") or
            payload.get("model") or
            payload.get("ema_model") or
            # Fallback: extract all tensors from checkpoint
            {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    )

    # Clean parameter names by removing common prefixes
    clean = {}
    for k, v in state.items():
        kk = k
        # Strip common wrapper prefixes
        for prefix in ("model.", "module.", "_orig_mod."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        clean[kk] = v

    # Load with strict=False to handle partial matches
    model.load_state_dict(clean, strict=False)


def optimize_inference(model: nn.Module) -> nn.Module:
    """
    Prepare model for inference by setting eval mode and freezing parameters.

    Args:
        model: Model to prepare

    Returns:
        The same model in eval mode with frozen gradients
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ============================================================================
# Export-Safe Forward Pass Methods
# ============================================================================
# These methods patch the FlowMapAutoencoder to ensure proper export behavior
# and compatibility with torch.export

def _softmax_head_export(self, logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to normalized latent representation (export-safe version).

    Steps:
    1. Apply log_softmax to logits
    2. Normalize using stored statistics (ln10_inv, log_mean, log_std)
    3. Return in the same dtype as input

    Args:
        logits: Raw output logits [B, K, S_out]

    Returns:
        Normalized latent representation [B, K, S_out]
    """
    log_p = F.log_softmax(logits, dim=-1)
    z = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z.to(dtype=logits.dtype)


def _head_from_logprobs_export(self, log_p: torch.Tensor) -> torch.Tensor:
    """
    Convert log probabilities to normalized latent representation (export-safe version).

    Similar to _softmax_head_export but takes log_probs instead of logits.

    Args:
        log_p: Log probabilities [B, K, S_out]

    Returns:
        Normalized latent representation [B, K, S_out]
    """
    z = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z.to(dtype=log_p.dtype)


def _forward_k1_export(
        self,
        y_i: torch.Tensor,  # [B, S_in] - Input state
        dt_norm: torch.Tensor,  # [B, 1] - Normalized time delta
        g: torch.Tensor  # [B, G] - Global conditioning vector
) -> torch.Tensor:
    """
    Export-safe forward pass for K=1 (single step prediction).

    This replaces the original forward method to ensure compatibility with
    torch.export and proper shape handling for xi.py inference.

    Args:
        y_i: Input state [B, S_in]
        dt_norm: Normalized time delta [B, 1]
        g: Global conditioning [B, G]

    Returns:
        Predicted state [B, 1, S_out] (matches xi.py expectations)
    """
    # Step 1: Encode input state with conditioning
    enc = self.encoder(y_i, g)
    if isinstance(enc, (tuple, list)):
        z_i, self.kl_loss = enc
    else:
        z_i, self.kl_loss = enc, None

    # Step 2: Apply dynamics model to predict future latent state
    z_k = self.dynamics(z_i, dt_norm, g)  # [B, 1, Z]

    # Step 3: Apply FiLM conditioning if enabled in config
    if getattr(self, "decoder_condition_on_g", False):
        z_k = self.film(z_k, g)  # [B, 1, Z]

    # Step 4: Decode latent state to output space
    logits = self.decoder(z_k)  # [B, 1, S_out]

    # Step 5: Convert to final output format
    # Two modes: standard prediction or delta prediction
    if not getattr(self, "predict_logit_delta", False):
        # Standard mode: direct softmax of logits
        return self._softmax_head_from_logits(logits)

    # Delta mode: combine base state with predicted delta
    # Select target indices if output dimension differs from input
    if self.S_out == self.S_in:
        base = y_i
    else:
        base = y_i.index_select(1, self.target_idx)  # [B, S_out]

    # Convert base state to log probabilities
    base_logp = self._denorm_to_logp(base)  # [B, S_out]

    # Get log probabilities of delta
    log_q = F.log_softmax(logits, dim=-1).float()  # [B, 1, S_out]

    # Combine base and delta in log space
    log_p = base_logp.unsqueeze(1) + log_q

    # Renormalize combined distribution
    log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)

    return self._head_from_logprobs(log_p)


# Apply patches to FlowMapAutoencoder class globally
# These patches ensure export-safe behavior
FlowMapAutoencoder._softmax_head_from_logits = _softmax_head_export
FlowMapAutoencoder._head_from_logprobs = _head_from_logprobs_export
FlowMapAutoencoder.forward = _forward_k1_export


# ============================================================================
# Multi-Step Wrapper (Dynamic B,K)
# ============================================================================

class ExportWrapperBK(nn.Module):
    """
    Wrapper for multi-step predictions with dynamic batch and K dimensions.

    This wrapper transforms batched multi-step inputs [B, K, ...] into
    flattened single-step inputs [B*K, ...], calls the base K=1 model,
    then reshapes outputs back to [B, K, S_out].

    This enables efficient batched multi-step inference on GPU/MPS.
    """

    def __init__(self, base: nn.Module):
        """
        Initialize wrapper with base K=1 model.

        Args:
            base: Base FlowMapAutoencoder model with K=1 forward pass
        """
        super().__init__()
        self.base = base
        # Extract dimensions from base model
        self.S_in = int(getattr(base, "S_in"))
        self.S_out = int(getattr(base, "S_out"))
        self.G = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)

    def forward(
            self,
            state: torch.Tensor,  # [B, K, S_in]
            dt: torch.Tensor,  # [B, K, 1]
            g: torch.Tensor  # [B, K, G]
    ) -> torch.Tensor:
        """
        Forward pass with dynamic batch and K dimensions.

        Process:
        1. Reshape [B, K, ...] to [B*K, ...]
        2. Call base model with flattened inputs
        3. Reshape outputs back to [B, K, S_out]

        Args:
            state: Input states [B, K, S_in]
            dt: Time deltas [B, K, 1]
            g: Global conditioning [B, K, G]

        Returns:
            Predicted states [B, K, S_out]
        """
        B, K, S_in = state.shape
        assert S_in == self.S_in, f"Expected S_in={self.S_in}, got {S_in}"

        # Flatten batch and K dimensions for base model
        state_flat = state.reshape(B * K, S_in)  # [BK, S_in]
        dt_flat = dt.reshape(B * K, dt.size(-1))  # [BK, 1]

        # Handle global conditioning
        if self.G > 0:
            g_flat = g.reshape(B * K, self.G)  # [BK, G]
        else:
            g_flat = state_flat.new_zeros((B * K, 0))  # [BK, 0]

        # Call base model with flattened inputs
        out_k1 = self.base(state_flat, dt_flat, g_flat)  # [BK, 1, S_out]

        # Handle different output formats from base model
        if isinstance(out_k1, (tuple, list)):
            out_k1 = out_k1[0]

        # Reshape to flat output
        if out_k1.dim() == 3 and out_k1.size(1) == 1:
            out_flat = out_k1[:, 0, :]  # [BK, S_out]
        elif out_k1.dim() == 2:
            out_flat = out_k1  # [BK, S_out]
        else:
            raise RuntimeError(f"Unexpected base output shape: {tuple(out_k1.shape)}")

        # Reshape back to [B, K, S_out]
        return out_flat.view(B, K, self.S_out)


# ============================================================================
# AOTI (Ahead-of-Time Compilation) Support
# ============================================================================

def emit_aoti_for(ep, device: str, example_inputs, target_dir: Path) -> None:
    """
    Compile and save AOTI package for GPU/MPS devices.

    AOTI (Ahead-of-Time Inductor) compiles the model to optimized code that
    loads faster than regular torch.export. CPU AOTI is intentionally skipped.

    This function tries multiple methods to save the AOTI package as the API
    may vary across PyTorch versions.

    Args:
        ep: Exported program from torch.export
        device: Target device ("cuda" or "mps")
        example_inputs: Tuple of example inputs for compilation
        target_dir: Directory to save AOTI artifacts
    """
    if device not in ("cuda", "mps"):
        return

    try:
        # Get the compiled module and run AOT compilation
        mod = ep.module()
        from torch._inductor import aot_compile
        aoti_pkg = aot_compile(mod, example_inputs)

        # Try various methods to save the AOTI package
        # Method 1: Try common saver methods
        for method_name in ("save", "save_packaged_artifact", "save_to_path",
                            "write_to_dir", "write_to_file", "export", "dump"):
            if hasattr(aoti_pkg, method_name):
                method = getattr(aoti_pkg, method_name)
                if callable(method):
                    try:
                        # Try calling with path as string
                        method(str(target_dir))
                        print(f"  AOTI package saved: {target_dir}")
                        return
                    except TypeError:
                        try:
                            # Try calling with path kwarg
                            method(path=str(target_dir))
                            print(f"  AOTI package saved: {target_dir}")
                            return
                        except Exception:
                            continue
                    except Exception:
                        continue

        # Method 2: Check if aoti_pkg is already a path
        if isinstance(aoti_pkg, (str, os.PathLike)):
            src = Path(aoti_pkg)
            if src.exists():
                if src.is_file():
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target_dir)
                    print(f"  AOTI package saved (copied): {target_dir}")
                    return
                elif src.is_dir():
                    shutil.copytree(src, target_dir, dirs_exist_ok=True)
                    print(f"  AOTI package saved (copied): {target_dir}")
                    return

        # Method 3: Check for common path attributes
        for attr in ("path", "output_path", "artifact_path", "dir", "directory"):
            if hasattr(aoti_pkg, attr):
                src = Path(getattr(aoti_pkg, attr))
                if src.exists():
                    if src.is_file():
                        target_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, target_dir)
                        print(f"  AOTI package saved (from .{attr}): {target_dir}")
                        return
                    elif src.is_dir():
                        shutil.copytree(src, target_dir, dirs_exist_ok=True)
                        print(f"  AOTI package saved (from .{attr}): {target_dir}")
                        return

        # If we get here, we couldn't figure out how to save
        print(f"  [warn] AOTI packaging: unrecognized object type ({type(aoti_pkg).__name__}); skipped")

    except Exception as e:
        print(f"  [warn] AOTI packaging failed: {e}")


# ============================================================================
# Validation
# ============================================================================

def validate_ep(
        ep,
        device: str,
        dtype: torch.dtype,
        S_in: int,
        G: int,
        dyn_k: bool
) -> None:
    """
    Validate exported model by running inference with example inputs.

    This ensures the exported model produces outputs with expected shapes
    and runs without errors.

    Args:
        ep: Exported program to validate
        device: Target device ("cpu", "cuda", or "mps")
        dtype: Data type for validation tensors
        S_in: Input state dimension
        G: Global conditioning dimension
        dyn_k: Whether to validate with dynamic K dimension (multi-step)
    """
    if not CFG.run_validation:
        return

    td = dtype
    B = CFG.eg_batch
    K = CFG.eg_k if dyn_k else 1

    # Create example inputs based on whether we're testing K=1 or dynamic K
    if dyn_k:
        # Multi-step inputs: [B, K, ...]
        state = torch.randn(B, K, S_in, dtype=td, device=device)
        dt = torch.randn(B, K, 1, dtype=td, device=device)
        g = (
            torch.randn(B, K, G, dtype=td, device=device)
            if G > 0
            else torch.empty(B, K, 0, dtype=td, device=device)
        )
    else:
        # Single-step inputs: [B, ...]
        state = torch.randn(B, S_in, dtype=td, device=device)
        dt = torch.randn(B, 1, dtype=td, device=device)
        g = (
            torch.randn(B, G, dtype=td, device=device)
            if G > 0
            else torch.empty(B, 0, dtype=td, device=device)
        )

    # Get the module and optionally compile it
    mod = ep.module()
    try:
        mod = torch.compile(mod, mode=CFG.compile_mode)
    except Exception as e:
        print(f"  [note] compile skipped: {e}")

    # Run inference
    with torch.inference_mode():
        out = mod(state, dt, g)

    # Validate output shape based on mode
    if dyn_k:
        # Multi-step output should be [B, K, S_out]
        if not (isinstance(out, torch.Tensor) and out.dim() == 3 and
                out.size(0) == B and out.size(1) == K):
            raise RuntimeError(f"Validation shape mismatch for dynamic-K: got {tuple(out.shape)}")
        print(f"  Validation OK (dyn-K): out shape = {tuple(out.shape)}")
    else:
        # Single-step output can be [B, 1, S] or [B, S]
        if isinstance(out, torch.Tensor) and out.dim() == 3 and out.size(1) == 1:
            print(f"  Validation OK (K=1): out shape = {tuple(out.shape)}")
        elif isinstance(out, torch.Tensor) and out.dim() == 2:
            print(f"  Validation OK (K=1, squeezed): out shape = {tuple(out.shape)}")
        else:
            raise RuntimeError(f"Validation shape mismatch for K=1: got {tuple(out.shape)}")


# ============================================================================
# Export Functions
# ============================================================================

def export_cpu_k1(base: nn.Module) -> None:
    """
    Export model for CPU inference with K=1 and dynamic batch dimension.

    This export is specifically designed for use with xi.py for single-step
    inference. It supports dynamic batch sizes while keeping K=1 fixed.

    Args:
        base: Base FlowMapAutoencoder model to export
    """
    print("\n" + "=" * 80)
    print("Exporting CPU (K=1, dynamic B)")
    print("=" * 80)

    dev = "cpu"
    model = optimize_inference(base.to(dev))

    # Define dynamic batch dimension with min/max constraints
    B = torch.export.Dim("batch", min=CFG.min_batch, max=CFG.max_batch)

    # Extract model dimensions
    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "global_dim", getattr(model, "G", 0)) or 0)

    # Create example inputs for tracing (batch size 2)
    y = torch.zeros(2, S_in, dtype=parse_dtype(CFG.cpu_dtype), device=dev)
    dt = torch.zeros(2, 1, dtype=parse_dtype(CFG.cpu_dtype), device=dev)
    g = torch.zeros(2, G, dtype=parse_dtype(CFG.cpu_dtype), device=dev)

    # Export with dynamic batch dimension
    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=({0: B}, {0: B}, {0: B})
    )

    # Save exported program
    CPU_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, CPU_OUT)
    print(f"  wrote {CPU_OUT}")

    # Validate the export
    validate_ep(ep, dev, parse_dtype(CFG.cpu_dtype), S_in, G, dyn_k=False)


def export_device_dynBK(
        base: nn.Module,
        device: str,
        out_path: Path,
        aoti_dir: Path,
        dtype_str: str
) -> None:
    """
    Export model for GPU/MPS with dynamic batch and K dimensions.

    This export supports batched multi-step inference with both dynamic batch
    size (B) and dynamic number of steps (K). Also generates AOTI artifacts
    for faster loading.

    Args:
        base: Base FlowMapAutoencoder model to export
        device: Target device ("cuda" or "mps")
        out_path: Path to save the exported program
        aoti_dir: Directory for AOTI artifacts
        dtype_str: String representation of dtype (e.g., "bfloat16")
    """
    print("\n" + "=" * 80)
    pretty = "GPU (CUDA)" if device == "cuda" else "MPS (Apple Silicon)"
    print(f"Exporting {pretty} (dynamic B,K)")
    print("=" * 80)

    td = parse_dtype(dtype_str)
    model = optimize_inference(base.to(device))

    # Wrap model to handle batched multi-step inputs
    wrapper = ExportWrapperBK(model).to(device)

    # Define dynamic dimensions for batch and K
    batch_dim = torch.export.Dim("batch", min=CFG.min_batch, max=CFG.max_batch)
    k_dim = torch.export.Dim("K", min=CFG.min_k, max=CFG.max_k)

    # Extract dimensions from wrapper
    S_in = wrapper.S_in
    G = wrapper.G

    # Create example inputs for tracing
    B, K = CFG.eg_batch, CFG.eg_k
    state = torch.randn(B, K, S_in, dtype=td, device=device)
    dt = torch.randn(B, K, 1, dtype=td, device=device)
    g = (
        torch.randn(B, K, G, dtype=td, device=device)
        if G > 0
        else torch.empty(B, K, 0, dtype=td, device=device)
    )

    # Export with dynamic shapes for both B and K
    ep = torch.export.export(
        wrapper,
        (state, dt, g),
        dynamic_shapes=(
            {0: batch_dim, 1: k_dim},  # state: dynamic B and K
            {0: batch_dim, 1: k_dim},  # dt: dynamic B and K
            {0: batch_dim, 1: k_dim},  # g: dynamic B and K
        ),
    )

    # Save exported program
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  wrote {out_path}")

    # Generate AOTI artifacts (only for GPU/MPS, not CPU)
    try:
        emit_aoti_for(ep, device, (state, dt, g), aoti_dir)
    except Exception as e:
        print(f"  [warn] AOTI emission failed: {e}")

    # Validate the export
    validate_ep(ep, device, td, S_in, G, dyn_k=True)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """
    Main export orchestration.

    Loads the model and exports it to three formats:
    1. CPU with K=1 and dynamic batch (for xi.py)
    2. GPU with dynamic B,K and AOTI (if CUDA available)
    3. MPS with dynamic B,K and AOTI (if MPS available)
    """
    print("=" * 80)
    print("FlowMap Export: CPU K=1 + GPU/MPS dynamic-K")
    print("=" * 80)
    print(f"Config path: {CONFIG_PATH}")

    # Load model configuration from JSON
    cfg_json = json.loads(CONFIG_PATH.read_text())
    base = create_model(cfg_json).eval().cpu()

    # Find and load best checkpoint
    ckpt = find_ckpt(WORK_DIR)
    print(f"Loading checkpoint: {ckpt}")
    load_weights(base, ckpt)

    # Export CPU version (K=1, dynamic batch) — for xi.py
    export_cpu_k1(base)

    # Export GPU version (dynamic B,K + AOTI) if CUDA is available
    if torch.cuda.is_available():
        try:
            export_device_dynBK(base, "cuda", GPU_OUT, GPU_AOTI_DIR, CFG.cuda_dtype)
        except Exception as e:
            print(f"[warn] CUDA export failed: {e}")
    else:
        print("[note] CUDA not available; skipping GPU export")

    # Export MPS version (dynamic B,K + AOTI) if MPS is available
    if torch.backends.mps.is_available():
        try:
            export_device_dynBK(base, "mps", MPS_OUT, MPS_AOTI_DIR, CFG.mps_dtype)
        except Exception as e:
            print(f"[warn] MPS export failed: {e}")
    else:
        print("[note] MPS not available; skipping MPS export")

    print("\nDone.")


if __name__ == "__main__":
    # Enable MPS fallback for operations that aren't natively supported
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
