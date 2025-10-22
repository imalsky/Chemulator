#!/usr/bin/env python3
"""
Export Flow-map AE (K=1) with torch.export to CPU and, if available, GPU (CUDA or MPS).

Run:
  python export.py
"""

from __future__ import annotations
import os

# --- macOS OpenMP duplication workaround (safe, avoids abort) ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import pathlib

import torch
import torch.nn as nn

# ========================== CONFIG ============================================

MODEL_NAME: str = "v1"   # folder under ./models/
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

from model import create_model  # your repo's model factory


# ----------------------------- I/O Helpers ------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(artifact: Path, meta: Dict[str, Any]) -> None:
    mp = artifact.with_suffix(artifact.suffix + ".meta.json")
    with mp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {mp}")


# --------------------------- Checkpoint picking --------------------------------

def _pick_checkpoint(model_dir: Path) -> Path:
    """
    Prefer: models/<MODEL_NAME>/checkpoints/last.ckpt
    Fallback: newest *.ckpt in checkpoints/
    Fallback: best_model.pt
    Fallback: newest *.pt in the model dir
    """
    ckpt_dir = model_dir / "checkpoints"
    preferred = ckpt_dir / "last.ckpt"
    if preferred.exists():
        print(f"Using checkpoint: {preferred}")
        return preferred

    if ckpt_dir.exists():
        all_ckpts = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if all_ckpts:
            print(f"Using newest checkpoint in {ckpt_dir}: {all_ckpts[0]}")
            return all_ckpts[0]

    best_model = model_dir / "best_model.pt"
    if best_model.exists():
        print(f"Using best_model.pt: {best_model}")
        return best_model

    pts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        print(f"Using newest *.pt in {model_dir}: {pts[0]}")
        return pts[0]

    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. "
        f"Tried checkpoints/last.ckpt, checkpoints/*.ckpt, best_model.pt, and *.pt."
    )


# --------------------------- Config / Model Load ------------------------------

def _load_cfg_and_checkpoint(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = _load_json(cfg_path)

    ckpt_path = _pick_checkpoint(model_dir)

    # PyTorch 2.6: default weights_only=True + strict unpickler.
    # Allow-list pathlib classes and try weights_only=True first, then fallback.
    def _load_checkpoint_safely(p: Path) -> Dict[str, Any]:
        try:
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([
                    pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath
                ])
            if hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath]):
                    return torch.load(p, map_location="cpu")
            # Older torch (no safe_globals): load normally (may default to weights_only=False)
            return torch.load(p, map_location="cpu")
        except Exception as e1:
            # Trusted local file: allow full unpickling
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load checkpoint at {p}.\n"
                    f"First error (weights_only=True): {e1}\n"
                    f"Second error (weights_only=False): {e2}"
                )

    ckpt = _load_checkpoint_safely(ckpt_path)
    return cfg, ckpt, ckpt_path


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
        norm_path = processed_dir / "normalization.json"
        if norm_path.exists():
            manifest = _load_json(norm_path)
            meta = manifest.get("meta", {})
            species = species or list(meta.get("species_variables") or [])
            globals_ = globals_ or list(meta.get("global_variables") or [])

    if not species or not globals_:
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
            for pref in ("_orig_mod.", "module.", "model."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    changed = True
        return k
    return {strip(k): v for k, v in state.items()}


def _build_and_load(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg, ckpt, ckpt_path = _load_cfg_and_checkpoint(model_dir)
    processed_dir = _resolve_processed_dir(cfg)
    _rehydrate_data_section(cfg, processed_dir)

    model = create_model(cfg)

    # find a plausible state_dict in the checkpoint
    state: Optional[Dict[str, torch.Tensor]] = None

    # (A) Lightning .ckpt case
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]

    # (B) our best_model.pt format: {'model': <state_dict>, ...}
    if state is None and isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]

    # (C) raw state_dict
    if state is None and isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt

    if state is None:
        raise RuntimeError(
            f"Could not find a usable state_dict in checkpoint: {ckpt_path}\n"
            f"Available keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    model.load_state_dict(_unwrap_state_dict(state), strict=True)
    model.to(device).eval()
    print(f"Loaded weights from: {ckpt_path}")
    return model, cfg


# ------------------------------- Export ---------------------------------------

def _require_export():
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("torch.export is unavailable. Install PyTorch >= 2.1 (preferably >= 2.3).")


def _export_variant(model: nn.Module, cfg: Dict[str, Any],
                    device: torch.device, dtype: torch.dtype, out_path: Path) -> Dict[str, Any]:
    """
    Export the model.forward(y, dt, g) for K=1 with batch dynamic shape.
    We pass the base model directly; no wrapper needed.
    """
    model = model.to(device=device, dtype=dtype).eval()

    # Infer dims for dummy inputs from cfg (rehydrated earlier)
    data_cfg = cfg.get("data", {})
    species = list(data_cfg.get("species_variables", []))
    globals_ = list(data_cfg.get("global_variables", []))
    S_in  = len(species) if species else 1
    G     = len(globals_) if globals_ else 0
    S_out = S_in  # forward returns [B,K,S]; S equals input species count in this model

    # Dummy inputs (K=1 → dt shape [B,1])
    ex_B = 2
    y  = torch.randn(ex_B, S_in, device=device, dtype=dtype)
    dt = torch.full((ex_B, 1), 0.5, device=device, dtype=dtype)  # normalized Δt offset
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
        "timestamp": int(time.time()),
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
        dtype = torch.float32  # MPS → always fp32
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
