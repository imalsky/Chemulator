#!/usr/bin/env python3
"""
Export Flow-map AE (K=1) with torch.export to CPU and, if available, GPU (CUDA or MPS).

Run:
  python testing/export_k1.py
"""

from __future__ import annotations
import os
import sys
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================== CONFIG ============================================

MODEL_NAME: str = "v4_4"   # folder under ./models/
OUT_CPU: str   = "export_k1_cpu.pt2"
OUT_GPU: str   = "export_k1_gpu.pt2"      # written only if a GPU backend exists
WRITE_META: bool = True                   # write a .meta.json next to each artifact

# GPU dtype preferences
USE_BF16_ON_CUDA: bool = True             # CUDA export dtype
USE_FP16_ON_MPS: bool  = True             # MPS export dtype

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

# ----------------------------- Wrapper ----------------------------------------

class K1AEExport(nn.Module):
    """
    Minimal, export-friendly forward for K=1 that mirrors FlowMapAutoencoder.forward.
    Caches constants to avoid per-call casting/divisions. Robust to missing/None stats.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

        S_in  = getattr(base, "S_in", None)
        S_out = getattr(base, "S_out", S_in)
        if S_out is None:
            raise RuntimeError("Base model must define S_out or S_in.")

        # ---- pull normalization stats safely ----
        def _as_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        lm = _as_tensor(getattr(base, "log_mean", None))
        ls = _as_tensor(getattr(base, "log_std",  None))

        # If stats exist but are on wrong device/dtype/shape, fix them.
        if lm is not None:
            lm = lm.view(-1).to(torch.float32)
        if ls is not None:
            ls = ls.view(-1).to(torch.float32)

        # Map stats from S_in -> S_out if needed and we have target_idx
        if (lm is not None and ls is not None) and lm.numel() != S_out:
            tgt = getattr(base, "target_idx", None)
            if tgt is not None:
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.as_tensor(tgt, dtype=torch.long)
                # Common case: stats saved for S_in, outputs are a subset/order via target_idx
                if lm.numel() == getattr(base, "S_in", lm.numel()) and tgt.numel() == S_out:
                    lm = lm.index_select(0, tgt)
                    ls = ls.index_select(0, tgt)

        # Final fallback if still missing or wrong sized: zeros/ones with safe std
        if lm is None or ls is None or lm.numel() != S_out or ls.numel() != S_out:
            lm = torch.zeros(S_out, dtype=torch.float32)
            ls = torch.ones(S_out,  dtype=torch.float32)

        # Avoid divide-by-zero in inv std
        ls = torch.where(ls == 0, torch.ones_like(ls), ls)

        # ---- cache constants/buffers (f32 baseline) ----
        self.register_buffer("inv_ln10_f32",
                             torch.tensor(1.0 / math.log(10.0), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("log_mean_f32", lm.clone(), persistent=False)
        self.register_buffer("log_std_f32",  ls.clone(), persistent=False)
        self.register_buffer("inv_log_std_f32", (1.0 / ls).clone(), persistent=False)

        # Optionally cache bf16 variants for CUDA fast paths
        if torch.cuda.is_available():
            self.register_buffer("inv_ln10_bf16",    self.inv_ln10_f32.to(torch.bfloat16), persistent=False)
            self.register_buffer("log_mean_bf16",    self.log_mean_f32.to(torch.bfloat16), persistent=False)
            self.register_buffer("inv_log_std_bf16", self.inv_log_std_f32.to(torch.bfloat16), persistent=False)

    @staticmethod
    def _to_b11(dt_norm: torch.Tensor) -> torch.Tensor:
        d = dt_norm.dim()
        if d == 1:  return dt_norm.view(dt_norm.size(0), 1, 1)
        if d == 2:  return dt_norm.unsqueeze(-1)  # [B,1] -> [B,1,1]
        if d == 3:  return dt_norm
        raise ValueError("dt_norm must have rank 1..3")

    def _pick(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        # bf16 fast-path for CUDA; otherwise fall back to f32 (also fine for fp16 MPS)
        if dtype == torch.bfloat16 and hasattr(self, name + "_bf16"):
            return getattr(self, name + "_bf16")
        return getattr(self, name + "_f32")

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        z_i, _kl = self.base.encoder(y_i, g)                     # [B,Z]
        z_j = self.base.dynamics(z_i, self._to_b11(dt_norm), g)  # [B,1,Z]
        y_pred = self.base.decoder(z_j)                          # [B,1,S_out]

        # Denorm / delta handling
        if getattr(self.base, "softmax_head", False):
            # y_pred are logits; convert to log_e probs then to log10 and z-normalize
            log_p = F.log_softmax(y_pred, dim=-1)
            inv_ln10 = self._pick("inv_ln10", log_p.dtype)
            lmean    = self._pick("log_mean", log_p.dtype)
            inv_lstd = self._pick("inv_log_std", log_p.dtype)
            y_pred = (log_p * inv_ln10 - lmean) * inv_lstd

        elif getattr(self.base, "predict_delta_log_phys", False):
            if self.base.S_out != self.base.S_in:
                tgt = getattr(self.base, "target_idx", None)
                if tgt is None:
                    raise RuntimeError("target_idx required when S_out != S_in")
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.as_tensor(tgt, dtype=torch.long, device=y_i.device)
                base_z = y_i.index_select(1, tgt).contiguous()
            else:
                base_z = y_i
            lmean    = self._pick("log_mean", base_z.dtype)
            inv_lstd = self._pick("inv_log_std", base_z.dtype)
            base_log   = base_z * (1.0 / inv_lstd) + lmean
            y_pred_log = base_log.unsqueeze(1) + y_pred
            y_pred     = (y_pred_log - lmean) * inv_lstd

        elif getattr(self.base, "predict_delta", False):
            if self.base.S_out != self.base.S_in:
                tgt = getattr(self.base, "target_idx", None)
                if tgt is None:
                    raise RuntimeError("target_idx required when S_out != S_in")
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.as_tensor(tgt, dtype=torch.long, device=y_i.device)
                base = y_i.index_select(1, tgt).contiguous()
            else:
                base = y_i
            y_pred = y_pred + base.unsqueeze(1)

        return y_pred.squeeze(1)  # [B,S_out]


# ----------------------------- Utilities --------------------------------------

def _require_export():
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("torch.export is unavailable. Install PyTorch >= 2.1 (preferably >= 2.3).")

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_cfg_and_checkpoint(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")
    return _load_json(cfg_path), torch.load(ckpt_path, map_location="cpu")

def _resolve_processed_dir(cfg: Dict[str, Any]) -> Path:
    pd_str = (cfg.get("paths") or {}).get("processed_data_dir")
    if not pd_str:
        raise KeyError("config['paths']['processed_data_dir'] is missing.")
    pd = Path(pd_str).expanduser().resolve()
    if not pd.exists():
        raise FileNotFoundError(f"Processed data dir not found: {pd}")
    return pd

def _rehydrate_data_section(cfg: Dict[str, Any], processed_dir: Path) -> None:
    species = globals_ = None
    tvar = None
    norm_path = processed_dir / "normalization.json"
    if norm_path.exists():
        manifest = _load_json(norm_path)
        meta = manifest.get("meta", {})
        species = list(meta.get("species_variables") or [])
        globals_ = list(meta.get("global_variables") or [])
        tvar = meta.get("time_variable")
    summ = processed_dir / "preprocessing_summary.json"
    if (not species or not globals_) and summ.exists():
        s = _load_json(summ)
        species = species or list(s.get("species_variables") or [])
        globals_ = globals_ or list(s.get("global_variables") or [])
        tvar = tvar or s.get("time_variable")
    if not species:
        raise KeyError("species_variables not found in processed artifacts.")
    data = cfg.setdefault("data", {})
    data["species_variables"] = species
    if globals_ is not None:
        data["global_variables"] = globals_
    if tvar is not None:
        data["time_variable"] = tvar

def _unwrap_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

def _export_variant(base_model: nn.Module, cfg: Dict[str, Any],
                    device: torch.device, dtype: torch.dtype, out_path: Path) -> Dict[str, Any]:
    base = base_model.to(device=device, dtype=dtype)
    wrapped = K1AEExport(base).to(device).eval()

    S_in = getattr(base, "S_in", len(cfg.get("data", {}).get("species_variables", [])) or 1)
    G    = getattr(base, "G",    len(cfg.get("data", {}).get("global_variables", [])) or 0)
    S_out= getattr(base, "S_out", S_in)

    ex_B = 2
    ex_y  = torch.randn(ex_B, S_in, device=device, dtype=dtype)
    ex_dt = torch.full((ex_B, 1, 1), 0.5, device=device, dtype=dtype)
    ex_g  = torch.randn(ex_B, G, device=device, dtype=dtype) if G > 0 else torch.zeros(ex_B, 0, device=device, dtype=dtype)

    from torch.export import export as export_prog, Dim
    B = Dim("batch")
    ep = export_prog(wrapped, (ex_y, ex_dt, ex_g), dynamic_shapes=({0: B}, {0: B}, {0: B}))
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
    print(f"Exported -> {out_path}  |  device={meta['device']} dtype={meta['dtype']}  dims: S_in={S_in}, S_out={S_out}, G={G}")
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
        dtype = torch.float16 if USE_FP16_ON_MPS else torch.float32
        gpu_art = model_dir / OUT_GPU
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safe for later runtime
        gpu_meta = _export_variant(base_model, cfg, torch.device("mps"), dtype, gpu_art)
    else:
        print("No GPU backend (CUDA/MPS) available; GPU export skipped.")

    # Write meta.json files
    if WRITE_META:
        def _write_meta(art: Path, meta: Dict[str, Any]):
            mp = art.with_suffix(art.suffix + ".meta.json")
            with mp.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Wrote {mp}")
        _write_meta(cpu_art, cpu_meta)
        if gpu_meta:
            _write_meta(gpu_art, gpu_meta)

if __name__ == "__main__":
    main()
