#!/usr/bin/env python3
"""
PyTorch Model Export with Optimizations (Dynamic-Batch + Dynamic-K)

- Loads your trained model and patches export-incompatible bits.
- Wraps the model with an export wrapper to accept inputs shaped (B, K, *).
- Exports torch.export programs for CPU, CUDA (if available), and MPS (if available).
- Validates correctness vs. the original model with a main K>=2 case and a K=1 smoke test.
- Optional CPU dynamic INT8 quantization (Linear layers).
- Optional AOTInductor packaging so you can run without torch.compile later.

All toggles are here; the benchmark script does no compilation/modification.

Notes
-----
- The exported graph expects 3D inputs: state:(B,K,S_in), dt:(B,K,1), globals:(B,K,G or (B,K,0)).
- If dynamic K is enabled, we trace with K>=2 to avoid guard collapse while still allowing K=1 at runtime.
"""

from __future__ import annotations
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# GLOBAL TOGGLES (only here)
# ==============================================================================

CPU_QUANTIZE: bool = False        # CPU INT8 dynamic quant (Linear). Export-time only.
EMIT_AOTI: bool = True            # Produce ahead-of-time compiled package(s) in addition to raw .pt2
AOTI_SUFFIX: str = ".aoti.pt2"    # Compiled artifact filename suffix
AOTI_INDUCTOR_CONFIGS: Optional[dict] = None  # e.g., {"max_autotune": True}

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExportConfig:
    # Paths
    work_dir: str = "models/big"         # <<< change to match your model dir
    cpu_output: str = "export_k_dyn_cpu.pt2"
    gpu_output: str = "export_k_dyn_gpu.pt2"
    mps_output: str = "export_k_dyn_mps.pt2"

    # Which exports to create
    export_cpu: bool = True
    export_gpu: bool = True   # CUDA
    export_mps: bool = True   # Apple Silicon

    # Dynamic batch configuration
    enable_dynamic_batch: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 4096
    optimal_batch_size: int = 1024   # example input

    # Dynamic K configuration
    enable_dynamic_k: bool = True
    min_k: int = 1
    max_k: int = 100
    optimal_k: int = 1              # user may set 1; exporter will trace with K>=2 if dynamic

    # Compilation (runtime) note: we do NOT compile at inference in bench; AOTI covers that.
    recommend_compile: bool = True
    compile_mode: str = "default"   # used only in validation to mirror typical runtime

    # Precision
    cpu_dtype: str = "float32"
    gpu_dtype: str = "bfloat16"     # "float32", "float16", "bfloat16"
    mps_dtype: str = "float32"      # MPS prefers float32

    # Quantization (CPU only)
    cpu_quantize: bool = CPU_QUANTIZE

    # Optimizations
    optimize_for_inference: bool = True

    # Validation
    run_validation: bool = True

    # Metadata
    save_metadata: bool = True

    # AOTI packaging
    emit_aoti: bool = EMIT_AOTI
    aoti_suffix: str = AOTI_SUFFIX
    aoti_inductor_configs: Optional[dict] = AOTI_INDUCTOR_CONFIGS


CONFIG = ExportConfig()

# ==============================================================================
# Setup
# ==============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORK_DIR = ROOT / CONFIG.work_dir
CONFIG_PATH = WORK_DIR / "config.json"

os.chdir(ROOT)
sys.path.insert(0, str(SRC_DIR))

from model import create_model, FlowMapAutoencoder  # type: ignore

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ==============================================================================
# Checkpoint Loading
# ==============================================================================

def find_best_checkpoint(model_dir: Path) -> Path:
    best_model = model_dir / "best_model.pt"
    if best_model.exists():
        return best_model

    checkpoint_dir = model_dir / "checkpoints"
    if checkpoint_dir.exists():
        candidates = []
        for ckpt_path in checkpoint_dir.glob("epoch*.ckpt"):
            m = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", ckpt_path.name)
            if m:
                epoch_num = int(m.group(1))
                try:
                    val_score = float(m.group(2))
                except (ValueError, OverflowError):
                    val_score = float("inf")
                candidates.append((val_score, -epoch_num, ckpt_path))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2]

        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

    pt_files = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    for key in ["state_dict", "model_state_dict", "model", "ema_model"]:
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes_to_remove = ["model.", "module.", "_orig_mod."]
    cleaned = {}
    for key, value in state_dict.items():
        k = key
        for prefix in prefixes_to_remove:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = value
    return cleaned


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    cleaned_state = clean_state_dict_keys(state_dict)
    model.load_state_dict(cleaned_state, strict=False)

# ==============================================================================
# Model Patching for Export Compatibility
# ==============================================================================

def patch_softmax_head_export(self, logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    latents = (log_probs.float() * self.ln10_inv - self.log_mean) / self.log_std
    return latents.to(dtype=logits.dtype)

def patch_head_from_logprobs_export(self, log_probs: torch.Tensor) -> torch.Tensor:
    latents = (log_probs.float() * self.ln10_inv - self.log_mean) / self.log_std
    return latents.to(dtype=log_probs.dtype)

def patch_forward_export(self, y_input, dt_normalized, globals_vec):
    enc = self.encoder(y_input, globals_vec)
    if isinstance(enc, (tuple, list)):
        latent_current, self.kl_loss = enc
    else:
        latent_current = enc
        self.kl_loss = None

    latent_future = self.dynamics(latent_current, dt_normalized, globals_vec)

    if getattr(self, "decoder_condition_on_g", False):
        latent_future = self.film(latent_future, globals_vec)

    logits = self.decoder(latent_future)

    if not getattr(self, "predict_logit_delta", False):
        return self._softmax_head_from_logits(logits)

    if self.S_out == self.S_in:
        base_state = y_input
    else:
        base_state = y_input.index_select(1, self.target_idx)

    base_logprobs = self._denorm_to_logp(base_state)      # (B*, S_out)
    log_deltas = F.log_softmax(logits, dim=-1).float()    # (B*, S_out)
    log_probs = base_logprobs + log_deltas                # (B*, S_out)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return self._head_from_logprobs(log_probs)

def apply_export_patches():
    FlowMapAutoencoder._softmax_head_from_logits = patch_softmax_head_export
    FlowMapAutoencoder._head_from_logprobs = patch_head_from_logprobs_export
    FlowMapAutoencoder.forward = patch_forward_export

# ==============================================================================
# Export Wrapper (Dynamic K via flatten/unflatten)
# ==============================================================================

class ExportWrapper(nn.Module):
    """
    Accepts (B, K, *) and flattens to (B*K, *) for the base model, then reshapes back.
    Inputs:
      state:   (B, K, S_in)
      dt:      (B, K, 1)
      globals: (B, K, G) or (B, K, 0)
    Returns:
      out:     (B, K, S_out)
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.S_in = getattr(base, "S_in", None)
        self.S_out = getattr(base, "S_out", None)
        self.global_dim = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)

    def forward(self, state: torch.Tensor, dt: torch.Tensor, globals_vec: torch.Tensor):
        B, K, S_in = state.shape
        state_flat = state.reshape(B * K, S_in)
        dt_flat = dt.reshape(B * K, dt.shape[-1])
        if globals_vec.ndim == 3 and globals_vec.shape[-1] > 0:
            G = globals_vec.shape[-1]
            globals_flat = globals_vec.reshape(B * K, G)
        else:
            globals_flat = state_flat.new_zeros((B * K, 0))

        out_flat = self.base(state_flat, dt_flat, globals_flat)
        if isinstance(out_flat, (tuple, list)):
            out_flat = out_flat[0]
        S_out = out_flat.shape[-1]
        return out_flat.reshape(B, K, S_out)

# ==============================================================================
# Utilities / Optimization
# ==============================================================================

def parse_dtype_string(dtype_str: str) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get(dtype_str.lower(), torch.float32)

def convert_model_precision(model: torch.nn.Module, dtype: str) -> torch.nn.Module:
    td = parse_dtype_string(dtype)
    if td != torch.float32:
        model = model.to(dtype=td)
    return model

def apply_quantization_cpu_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )
    except Exception as e:
        print(f"  Warning: CPU quantization failed ({e}); using unquantized model")
        return model

def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ==============================================================================
# Export helpers
# ==============================================================================

def _unwrap_base(m: nn.Module) -> nn.Module:
    return m.base if isinstance(m, ExportWrapper) else m

def _get_dims_from_model(m: nn.Module) -> Tuple[int, int]:
    base = _unwrap_base(m)
    S_in = int(getattr(base, "S_in"))
    G = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)
    return S_in, G

def create_example_inputs(model: nn.Module, device: str, dtype: str):
    B = CONFIG.optimal_batch_size
    if CONFIG.enable_dynamic_k:
        K_for_export = max(2, CONFIG.optimal_k, CONFIG.min_k)  # avoid K=1 collapse
    else:
        K_for_export = max(1, CONFIG.optimal_k)

    base = model.base if isinstance(model, ExportWrapper) else model
    S_in = int(getattr(base, "S_in"))
    G = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)
    tdtype = parse_dtype_string(dtype)

    state = torch.randn(B, K_for_export, S_in, dtype=tdtype, device=device)
    dt = torch.randn(B, K_for_export, 1, dtype=tdtype, device=device)
    globals_vec = torch.randn(B, K_for_export, G, dtype=tdtype, device=device) if G > 0 \
                  else torch.empty(B, K_for_export, 0, dtype=tdtype, device=device)
    return state, dt, globals_vec

def export_model(model: nn.Module, example_inputs, output_path: Path):
    state_ex, dt_ex, g_ex = example_inputs
    dynamic_shapes = None
    if CONFIG.enable_dynamic_batch or CONFIG.enable_dynamic_k:
        batch_dim = torch.export.Dim("batch", min=CONFIG.min_batch_size, max=CONFIG.max_batch_size)
        k_dim = torch.export.Dim("K", min=CONFIG.min_k, max=CONFIG.max_k)
        dynamic_shapes = (
            {0: batch_dim, 1: k_dim},
            {0: batch_dim, 1: k_dim},
            {0: batch_dim, 1: k_dim},
        )

    ep = torch.export.export(
        model,
        (state_ex, dt_ex, g_ex),
        dynamic_shapes=dynamic_shapes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, output_path)
    return ep

def save_export_metadata(output_path: Path, model: nn.Module, device: str, dtype: str):
    base = _unwrap_base(model)
    S_in = int(getattr(base, "S_in"))
    G = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)

    metadata = {
        "device": device,
        "dtype": dtype,
        "export_config": {
            "dynamic_batch": CONFIG.enable_dynamic_batch,
            "min_batch_size": CONFIG.min_batch_size,
            "max_batch_size": CONFIG.max_batch_size,
            "optimal_batch_size": CONFIG.optimal_batch_size,
            "dynamic_k": CONFIG.enable_dynamic_k,
            "min_k": CONFIG.min_k,
            "max_k": CONFIG.max_k,
            "optimal_k": CONFIG.optimal_k,
            "recommend_compile": CONFIG.recommend_compile,
            "recommended_compile_mode": CONFIG.compile_mode,
            "cpu_quantize": CONFIG.cpu_quantize if device == "cpu" else False,
        },
        "model_info": {
            "input_dim": S_in,
            "global_dim": G,
            "model_class": type(base).__name__,
        },
    }
    try:
        commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
        if commit:
            metadata["git_commit"] = commit
    except Exception:
        pass

    meta_path = output_path.parent / f"{output_path.name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

# --- AOTInductor packaging -----------------------------------------------------

def _aoti_name_for(raw_export_path: Path) -> Path:
    # "export_k_dyn_cpu.pt2" -> "export_k_dyn_cpu.aoti.pt2"
    stem = raw_export_path.with_suffix("")  # strip ".pt2"
    return stem.with_name(stem.name + CONFIG.aoti_suffix)

def aoti_compile_and_package(ep, out_path: Path) -> Path:
    from torch._inductor import aoti_compile_and_package
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pkg_path = aoti_compile_and_package(
        ep,
        package_path=str(out_path),
        inductor_configs=CONFIG.aoti_inductor_configs,
    )
    return Path(pkg_path)

def save_aoti_metadata(aoti_path: Path, device: str, dtype: str, model: nn.Module):
    base = _unwrap_base(model)
    S_in = int(getattr(base, "S_in"))
    G = int(getattr(base, "global_dim", getattr(base, "G", 0)) or 0)
    meta = {
        "device": device,
        "dtype": dtype,
        "aoti": True,
        "export_config": {  # mirror the raw export block for bench convenience
            "dynamic_batch": CONFIG.enable_dynamic_batch,
            "min_batch_size": CONFIG.min_batch_size,
            "max_batch_size": CONFIG.max_batch_size,
            "dynamic_k": CONFIG.enable_dynamic_k,
            "min_k": CONFIG.min_k,
            "max_k": CONFIG.max_k,
        },
        "model_info": {"input_dim": S_in, "global_dim": G, "model_class": type(base).__name__},
        "notes": "AOTInductor-compiled package; no torch.compile needed to run.",
    }
    meta_path = aoti_path.parent / f"{aoti_path.name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

# ==============================================================================
# Validation
# ==============================================================================

def validate_export(original_model: nn.Module, exported_program, device: str, dtype: str):
    if not CONFIG.run_validation:
        return

    B = CONFIG.optimal_batch_size
    K_main = max(2, CONFIG.optimal_k, CONFIG.min_k) if CONFIG.enable_dynamic_k else max(1, CONFIG.optimal_k)
    S_in, G = _get_dims_from_model(original_model)
    tdtype = parse_dtype_string(dtype)

    def make_inputs(Bsz: int, Ksz: int):
        state = torch.randn(Bsz, Ksz, S_in, dtype=tdtype, device=device)
        dt = torch.randn(Bsz, Ksz, 1, dtype=tdtype, device=device)
        g = torch.randn(Bsz, Ksz, G, dtype=tdtype, device=device) if G > 0 \
            else torch.empty(Bsz, Ksz, 0, dtype=tdtype, device=device)
        return state, dt, g

    exp_mod = exported_program.module()
    if CONFIG.recommend_compile:
        try:
            exp_mod = torch.compile(exp_mod, mode=CONFIG.compile_mode)
        except Exception as e:
            print(f"    compile skipped: {e}")

    # Main validation with K_main
    s, d, g = make_inputs(B, K_main)
    with torch.inference_mode():
        ref = original_model(s, d, g); ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        out = exp_mod(s, d, g);        out = out[0] if isinstance(out, (tuple, list)) else out
    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    rel_err = (torch.norm(ref - out) / torch.clamp(torch.norm(ref), min=1e-8)).item()

    print(f"    Main validation (K={K_main})")
    print(f"      Max |Δ|: {max_diff:.2e}")
    print(f"      Mean |Δ|: {mean_diff:.2e}")
    print(f"      Rel L2 error: {rel_err:.2e}")

    tol = 1e-3 if parse_dtype_string(dtype) is torch.float32 else 2e-2
    print("      Validation passed" if max_diff < tol else "      Warning: difference exceeds tolerance")

    # K=1 smoke test
    s1, d1, g1 = make_inputs(B, 1)
    with torch.inference_mode():
        ref1 = original_model(s1, d1, g1); ref1 = ref1[0] if isinstance(ref1, (tuple, list)) else ref1
        out1 = exp_mod(s1, d1, g1);        out1 = out1[0] if isinstance(out1, (tuple, list)) else out1
    md1 = (ref1 - out1).abs().max().item()
    mm1 = (ref1 - out1).abs().mean().item()
    rk1 = (torch.norm(ref1 - out1) / torch.clamp(torch.norm(ref1), min=1e-8)).item()
    print(f"    K=1 smoke test")
    print(f"      Max |Δ|: {md1:.2e}")
    print(f"      Mean |Δ|: {mm1:.2e}")
    print(f"      Rel L2 error: {rk1:.2e}")
    print("      K=1 passed" if md1 < tol else "      K=1 warning: difference exceeds tolerance")

# ==============================================================================
# Device export pipeline
# ==============================================================================

def export_for_device(base_model: nn.Module,
                      device_name: str,
                      device_type: str,
                      dtype: str,
                      output_filename: str,
                      quantize_cpu: bool = False) -> Path:
    print(f"\n{'=' * 80}")
    print(f"Exporting for {device_name} ({device_type}, {dtype})")
    print(f"{'=' * 80}")

    import copy
    model = copy.deepcopy(base_model).eval()

    print(f"  Moving model to {device_type}...")
    model = model.to(device_type)

    if dtype != "float32":
        print(f"  Converting to {dtype}...")
        model = convert_model_precision(model, dtype)

    if CONFIG.optimize_for_inference:
        print("  Applying inference optimizations...")
        model = optimize_model_for_inference(model)

    if device_type == "cpu" and quantize_cpu:
        print("  Applying CPU dynamic INT8 quantization...")
        model = apply_quantization_cpu_dynamic(model)

    wrapped = ExportWrapper(model)

    example_inputs = create_example_inputs(wrapped, device_type, dtype)

    output_path = WORK_DIR / output_filename
    print(f"  Exporting to {output_filename}...")
    if CONFIG.enable_dynamic_batch or CONFIG.enable_dynamic_k:
        print(f"    Dynamic batch: [{CONFIG.min_batch_size}, {CONFIG.max_batch_size}]")
        print(f"    Dynamic K: [{CONFIG.min_k}, {CONFIG.max_k}]")
    exported = export_model(wrapped, example_inputs, output_path)
    print(f"  Export saved: {output_path}")

    if CONFIG.save_metadata:
        save_export_metadata(output_path, wrapped, device_type, dtype)
        print("  Metadata saved")

    if CONFIG.run_validation:
        print("  Validating...")
        validate_export(wrapped, exported, device_type, dtype)

    # AOTInductor package (optional)
    if CONFIG.emit_aoti:
        print("  AOTInductor: compiling & packaging...")
        try:
            aoti_out = _aoti_name_for(output_path)
            pkg_path = aoti_compile_and_package(exported, aoti_out)
            print(f"  AOTI package saved: {pkg_path}")
            save_aoti_metadata(pkg_path, device_type, dtype, model=wrapped)
        except Exception as e:
            print(f"  AOTI packaging failed: {e}")

    print(f"  {device_name} export complete")
    return output_path

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("PyTorch Model Export with Optimizations (Dynamic-Batch + Dynamic-K)")
    print("=" * 80)

    print(f"\nLoading configuration from: {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text())

    print("Creating model architecture...")
    base_model = create_model(cfg).eval().cpu()

    ckpt_path = find_best_checkpoint(WORK_DIR)
    print(f"Loading checkpoint: {ckpt_path.name}")
    load_model_weights(base_model, ckpt_path)

    print("Applying export patches...")
    apply_export_patches()

    exports_created = []

    if CONFIG.export_cpu:
        path = export_for_device(
            base_model, "CPU", "cpu", CONFIG.cpu_dtype,
            CONFIG.cpu_output, quantize_cpu=CONFIG.cpu_quantize
        ); exports_created.append(("CPU", path))

    if CONFIG.export_gpu and torch.cuda.is_available():
        path = export_for_device(
            base_model, "GPU (CUDA)", "cuda", CONFIG.gpu_dtype,
            CONFIG.gpu_output, quantize_cpu=False
        ); exports_created.append(("GPU", path))
    elif CONFIG.export_gpu:
        print("\nWarning: GPU export requested but CUDA not available, skipping")

    if CONFIG.export_mps and torch.backends.mps.is_available():
        path = export_for_device(
            base_model, "MPS (Apple Silicon)", "mps", CONFIG.mps_dtype,
            CONFIG.mps_output, quantize_cpu=False
        ); exports_created.append(("MPS", path))
    elif CONFIG.export_mps:
        print("\nWarning: MPS export requested but not available, skipping")

    print(f"\n{'=' * 80}")
    print("Export Complete")
    print(f"{'=' * 80}")
    print(f"\nCreated {len(exports_created)} export(s):")
    for device_name, path in exports_created:
        print(f"  {device_name}: {path}")

    print("\nUsage (Python):")
    print("  # Raw exported program")
    print("  ep = torch.export.load('.../export_k_dyn_*.pt2')")
    print("  model = ep.module()")
    print("  # OR load AOTI package (no torch.compile):")
    print("  compiled = torch._inductor.aoti_load_package('.../export_k_dyn_*.aoti.pt2')")
    print("  out = compiled(state, dt, globals)  # shapes: (B,K,S_in), (B,K,1), (B,K,G)")

if __name__ == "__main__":
    main()
