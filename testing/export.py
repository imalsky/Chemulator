#!/usr/bin/env python3
"""
Export Stable LPV Koopman AE to a clean CPU torch.export artifact (.pt2).

- No argparse; globals only.
- Normalization: ONLY normalization.json (no fallback).
- Patches during export:
  * Replaces the ENTIRE torch.amp.autocast_mode (and torch.cuda.amp.autocast_mode)
    with a dummy namespace exposing no-op {autocast, _enter_autocast, _exit_autocast}.
  * Also overrides torch.autocast and torch.amp.autocast to a no-op.
  * Replaces dynamics._denorm_dt with a tensor-only, float32 version (no .item()).
- Verifies the exported graph has no autocast nodes.

Outputs:
  models/stable-lpv-koopman/model_ep_cpu.pt2
  models/stable-lpv-koopman/model_ep_cpu.pt2.meta.json
"""

from __future__ import annotations
import os, sys, json, types
from pathlib import Path
from typing import Any, Dict, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

# ========================== GLOBALS ===========================================

MODEL_NAME       = "stable-lpv-koopman"     # models/<MODEL_NAME>
OUT_CPU_FILENAME = "model_ep_cpu.pt2"
WRITE_META       = True

# ==============================================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(REPO_ROOT)

import model as model_mod
from model import create_model  # repo factory

# ----------------------------- Helpers ----------------------------------------

def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _cfg_and_ckpt(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")
    cfg = _load_json(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return cfg, ckpt

def _strict_norm_meta(cfg: Dict[str, Any]) -> Tuple[list, list, Path]:
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")
    norm_path = data_dir / "normalization.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing required file: {norm_path}")
    norm = _load_json(norm_path)
    meta = norm.get("meta") or {}
    species = list(meta.get("species_variables") or [])
    globals_ = list(meta.get("global_variables") or [])
    if len(species) == 0:
        raise ValueError(
            "normalization.json has empty 'meta.species_variables'. "
            "You must build that file correctly before export."
        )
    return species, globals_, data_dir

def _unwrap_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip(k: str) -> str:
        for pref in ("_orig_mod.", "module."):
            if k.startswith(pref):
                return k[len(pref):]
        return k
    return {strip(k): v for k, v in state.items()}

def _find_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "ema_model"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise RuntimeError("Checkpoint does not contain a usable state_dict.")

# ------------------------ Autocast killer -------------------------------------

@contextmanager
def _kill_autocast_everywhere():
    """
    Nuclear option: during export, replace the entire autocast modules with
    dummy namespaces providing no-op {autocast, _enter_autocast, _exit_autocast}
    so ANY of these access patterns are safe:

      torch.autocast(...)
      torch.amp.autocast(...)
      torch.amp.autocast_mode._enter_autocast(...)
      torch.cuda.amp.autocast(...)
      torch.cuda.amp.autocast_mode._enter_autocast(...)

    Everything is restored afterward.
    """
    saved: Dict[tuple, object] = {}

    class _DummyCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    def _noop_ctx(*args, **kwargs) -> _DummyCtx:
        return _DummyCtx()

    def _noop_enter(*args, **kwargs):
        return object()

    def _noop_exit(*args, **kwargs):
        return None

    def save_set(obj, name, value):
        if obj is None or not hasattr(obj, name):
            return
        key = (obj, name)
        saved[key] = getattr(obj, name)
        setattr(obj, name, value)

    # 1) torch.autocast & torch.amp.autocast
    save_set(torch, "autocast", _noop_ctx)
    if hasattr(torch, "amp"):
        save_set(torch.amp, "autocast", _noop_ctx)

    # 2) Replace torch.amp.autocast_mode with a dummy namespace
    dummy_acm = types.SimpleNamespace(
        autocast=_noop_ctx,
        _enter_autocast=_noop_enter,
        _exit_autocast=_noop_exit,
    )
    if hasattr(torch, "amp"):
        save_set(torch.amp, "autocast_mode", dummy_acm)

    # 3) Replace torch.cuda.amp.autocast & autocast_mode if the package exists
    try:
        import torch.cuda.amp as cuda_amp  # noqa: F401
        has_cuda_amp = True
    except Exception:
        has_cuda_amp = False

    if has_cuda_amp:
        save_set(torch.cuda.amp, "autocast", _noop_ctx)
        dummy_cuda_acm = types.SimpleNamespace(
            autocast=_noop_ctx,
            _enter_autocast=_noop_enter,
            _exit_autocast=_noop_exit,
        )
        save_set(torch.cuda.amp, "autocast_mode", dummy_cuda_acm)

    # 4) If your model module imported symbols directly, patch them too
    for name in ("autocast", "_enter_autocast", "_exit_autocast"):
        if hasattr(model_mod, name):
            save_set(model_mod, name, {"autocast": _noop_ctx,
                                       "_enter_autocast": _noop_enter,
                                       "_exit_autocast": _noop_exit}[name])

    try:
        yield
    finally:
        for (obj, name), val in saved.items():
            setattr(obj, name, val)

def _assert_no_autocast_nodes(ep: "torch.export.ExportedProgram") -> None:
    gm = ep.graph_module
    for n in gm.graph.nodes:
        tn = str(n.target)
        if ("autocast" in n.name or "autocast" in tn or
            "_enter_autocast" in tn or "_exit_autocast" in tn):
            raise RuntimeError(
                f"Autocast op found in export graph: name={n.name}, target={tn}. Export is tainted; aborting."
            )

# ------------------ dynamics._denorm_dt monkey-patch --------------------------

def _patch_denorm_dt_float32(dyn_mod: nn.Module) -> None:
    """
    Replace _denorm_dt to avoid .item() and float64. Pure tensor math in float32.
    Returns [B, K].
    """
    def _denorm_dt_export(self, dt_norm: torch.Tensor) -> torch.Tensor:
        x = dt_norm
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.ndim == 1:
            x = x.view(1, -1)
        elif x.ndim != 2:
            B = x.shape[0]
            K = int(x.numel() // B)
            x = x.reshape(B, K)
        x = x.to(dtype=torch.float32).clamp(0.0, 1.0)

        dt_range = (self.dt_log_max - self.dt_log_min).to(dtype=torch.float32)
        dt_log   = self.dt_log_min.to(torch.float32) + x * dt_range
        ln10 = torch.tensor(2.302585093, dtype=torch.float32, device=dt_log.device)
        dt_phys = torch.exp(dt_log * ln10)
        return torch.clamp(dt_phys, min=1e-30)

    dyn_mod._denorm_dt = types.MethodType(_denorm_dt_export, dyn_mod)

# --------------------------------- Export -------------------------------------

def _export_cpu(model: nn.Module, species: list, globals_: list, out_path: Path) -> Dict[str, Any]:
    model = model.to("cpu", dtype=torch.float32).eval()

    # Patch dt denorm before tracing
    if hasattr(model, "dynamics") and hasattr(model.dynamics, "_denorm_dt"):
        _patch_denorm_dt_float32(model.dynamics)

    # Infer dims
    S_in  = getattr(model, "S_in",  max(1, len(species)))
    S_out = getattr(model, "S_out", S_in)
    G     = getattr(model, "G_in",  max(0, len(globals_)))

    # Dummy inputs (B dynamic)
    B_ex = 4
    y  = torch.randn(B_ex, S_in, dtype=torch.float32)
    dt = torch.full((B_ex, 1, 1), 0.5, dtype=torch.float32)
    g  = torch.randn(B_ex, G, dtype=torch.float32) if G > 0 else torch.zeros(B_ex, 0, dtype=torch.float32)

    from torch.export import export as export_prog, Dim
    B = Dim("batch")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # HARD disable any autocast references reachable from model.forward
    with _kill_autocast_everywhere():
        ep = export_prog(
            model,
            (y, dt, g),
            dynamic_shapes=({0: B}, {0: B}, {0: B}),
        )

    _assert_no_autocast_nodes(ep)
    torch.export.save(ep, out_path)

    meta = {
        "device": "cpu",
        "dtype": "float32",
        "S_in": int(S_in),
        "S_out": int(S_out),
        "G": int(G),
        "pytorch": torch.__version__,
        "file": str(out_path),
    }
    print(f"[OK] torch.export saved -> {out_path}")
    return meta

def _write_meta(artifact: Path, meta: Dict[str, Any]) -> None:
    mp = artifact.with_suffix(artifact.suffix + ".meta.json")
    with mp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] wrote meta -> {mp}")

# --------------------------------- Main ---------------------------------------

def main():
    # Force CPU
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_default_device("cpu")

    model_dir = (REPO_ROOT / "models" / MODEL_NAME).resolve()
    cfg, ckpt = _cfg_and_ckpt(model_dir)

    # STRICT: normalization.json only; inject into cfg['data']
    species, globals_, _ = _strict_norm_meta(cfg)
    data_cfg = cfg.setdefault("data", {})
    data_cfg["species_variables"] = list(species)
    data_cfg["global_variables"]  = list(globals_)

    # Build & load
    model = create_model(cfg)
    state = _unwrap_state_dict(_find_state_dict(ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    out_cpu = model_dir / OUT_CPU_FILENAME
    try:
        if out_cpu.exists():
            out_cpu.unlink()
    except Exception:
        pass

    meta = _export_cpu(model, species, globals_, out_cpu)
    if WRITE_META:
        _write_meta(out_cpu, meta)

if __name__ == "__main__":
    main()
