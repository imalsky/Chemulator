#!/usr/bin/env python3
"""
Export Koopman AE (K=1) with torch.export to CPU and, if available, GPU (CUDA or MPS).

Run:
  python testing/export_k1.py
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# ========================== CONFIG ============================================

MODEL_NAME: str = "koopman-v2"   # folder under ./models/
OUT_CPU: str   = "export_k1_cpu.pt2"
OUT_GPU: str   = "export_k1_gpu.pt2"  # written only if a GPU backend exists
WRITE_META: bool = True

# GPU dtype preferences
USE_BF16_ON_CUDA: bool = True   # CUDA export dtype (bf16 if True, else fp32)
FORCE_FP32_ON_MPS: bool = True  # MUST be True; MPS + fp16/bf16 is problematic

# Enable TF32 on CUDA (safe speed knob)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ==============================================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(REPO_ROOT)

from model import create_model  # repo's model factory


# ----------------------------- I/O Helpers ------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(artifact: Path, meta: Dict[str, Any]) -> None:
    mp = artifact.with_suffix(artifact.suffix + ".meta.json")
    with mp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {mp}")


# --------------------------- Config / Model Load ------------------------------

def _load_cfg_and_checkpoint(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")
    cfg = _load_json(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return cfg, ckpt


def _resolve_processed_dir(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths") or {}
    pd_str = paths.get("processed_data_dir")
    if not pd_str:
        raise KeyError("config['paths']['processed_data_dir'] is missing.")
    pd = Path(pd_str).expanduser().resolve()
    if not pd.exists():
        raise FileNotFoundError(f"Processed data dir not found: {pd}")
    return pd


def _rehydrate_data_section(cfg: Dict[str, Any], processed_dir: Path) -> None:
    """
    Ensure cfg['data'] contains species_variables / global_variables
    using processed artifacts (normalization.json preferred).
    """
    data = cfg.setdefault("data", {})
    species = list(data.get("species_variables") or [])
    globals_ = list(data.get("global_variables") or [])

    if not species or not globals_:
        # Try normalization.json first
        norm_path = processed_dir / "normalization.json"
        if norm_path.exists():
            manifest = _load_json(norm_path)
            meta = manifest.get("meta", {})
            species = species or list(meta.get("species_variables") or [])
            globals_ = globals_ or list(meta.get("global_variables") or [])

    if not species or not globals_:
        # Fallback: preprocessing_summary.json
        summary = processed_dir / "preprocessing_summary.json"
        if summary.exists():
            summ = _load_json(summary)
            species = species or list(summ.get("species_variables") or [])
            globals_ = globals_ or list(summ.get("global_variables") or [])

    if not species:
        raise KeyError("species_variables not found in processed artifacts.")

    data["species_variables"] = species
    if globals_ is not None:
        data["global_variables"] = globals_


def _unwrap_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrapper prefixes."""
    def strip(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for pref in ("_orig_mod.", "module."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    changed = True
        return k
    return {strip(k): v for k, v in state.items()}


def _build_and_load(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg, ckpt = _load_cfg_and_checkpoint(model_dir)
    processed_dir = _resolve_processed_dir(cfg)
    _rehydrate_data_section(cfg, processed_dir)

    model = create_model(cfg)

    # find a plausible state_dict in the checkpoint
    state = None
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "ema_model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
    if state is None and isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a usable state_dict.")

    model.load_state_dict(_unwrap_state_dict(state), strict=True)
    model.to(device).eval()
    return model, cfg


# ------------------------------- Export ---------------------------------------

def _require_export():
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("torch.export is unavailable. Install PyTorch >= 2.1 (preferably >= 2.3).")


def _infer_dims(model: nn.Module, cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    data_cfg = cfg.get("data", {})
    S_in  = getattr(model, "S_in",  len(data_cfg.get("species_variables", [])) or 1)
    S_out = getattr(model, "S_out", S_in)
    G     = getattr(model, "G_in",  len(data_cfg.get("global_variables", [])) or 0)
    return int(S_in), int(S_out), int(G)


def _export_variant(model: nn.Module, cfg: Dict[str, Any],
                    device: torch.device, dtype: torch.dtype, out_path: Path) -> Dict[str, Any]:
    """
    Export model.forward(y, dt, g) for K=1 with batch dynamic shape.
    The program expects shapes: y [B,S], dt [B,1,1] (or [B,1]), g [B,G].
    """
    model = model.to(device=device, dtype=dtype).eval()

    # Infer dims for dummy inputs
    S_in, S_out, G = _infer_dims(model, cfg)

    # Dummy inputs (K=1 → dt shape [B,1,1])
    ex_B = 2
    y  = torch.randn(ex_B, S_in, device=device, dtype=dtype)
    dt = torch.full((ex_B, 1, 1), 0.5, device=device, dtype=dtype)
    g  = torch.randn(ex_B, G, device=device, dtype=dtype) if G > 0 else torch.zeros(ex_B, 0, device=device, dtype=dtype)

    # Export with batch dynamic shape
    from torch.export import export as export_prog, Dim
    B = Dim("batch")
    ep = export_prog(model, (y, dt, g), dynamic_shapes=({0: B}, {0: B}, {0: B}))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)

    meta = {
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "S_in": int(S_in),
        "S_out": int(S_out),
        "G": int(G),
        "file": str(out_path),
    }
    print(f"Exported -> {out_path}  |  device={meta['device']}  dtype={meta['dtype']}  (S_in={S_in}, S_out={S_out}, G={G})")
    return meta


# --------------------------------- Main ---------------------------------------

def main() -> None:
    _require_export()

    model_dir = (REPO_ROOT / "models" / MODEL_NAME).resolve()
    cpu_device = torch.device("cpu")
    base_model, cfg = _build_and_load(model_dir, cpu_device)

    # CPU export (float32)
    cpu_art = model_dir / OUT_CPU
    cpu_meta = _export_variant(base_model, cfg, cpu_device, torch.float32, cpu_art)

    # GPU export (CUDA preferred, else MPS)
    gpu_meta = None
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if USE_BF16_ON_CUDA else torch.float32
        gpu_art = model_dir / OUT_GPU
        gpu_meta = _export_variant(base_model, cfg, torch.device("cuda"), dtype, gpu_art)
    elif torch.backends.mps.is_available():
        # Always fp32 on MPS
        if FORCE_FP32_ON_MPS:
            dtype = torch.float32
        else:
            dtype = torch.float32
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        gpu_art = model_dir / OUT_GPU
        gpu_meta = _export_variant(base_model, cfg, torch.device("mps"), dtype, gpu_art)
    else:
        print("No GPU backend (CUDA/MPS) available; GPU export skipped.")

    # Write meta.json files
    if WRITE_META:
        _write_meta(cpu_art, cpu_meta)
        if gpu_meta:
            _write_meta(gpu_art, gpu_meta)


if __name__ == "__main__":
    main()
