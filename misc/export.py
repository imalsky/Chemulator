#!/usr/bin/env python3
"""
export.py - Export FlowMap as a 1-step autoregressive module via torch.export (.pt2).

What you get:
  1) ALWAYS: a batch-1 export that will work with predictions.py immediately.
     - export_cpu_b1_1step.pt2
     - export_mps_b1_1step.pt2 (if available)
     - export_cuda_b1_1step.pt2 (if available)

  2) OPTIONAL: a dynamic-batch export attempt (true dynamic B).
     - export_cpu_dynB_1step.pt2 (and mps/cuda variants)
     This requires the model forward path to be SymInt-safe (no int(...) shape guards).
     If it fails, we print why and keep the batch-1 exports.

Intended location: <repo_root>/testing/export.py
(where <repo_root>/src contains main.py, model.py, etc.)

Edit RUN_DIR at the top to choose which trained run/checkpoint to export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import create_model  # noqa: E402


# -----------------------------
# CHOOSE YOUR RUN HERE (no args, no prompts)
# -----------------------------
RUN_DIR = (ROOT / "models" / "v1").resolve()
PREFERRED_CKPT = None

# Export controls
EXPORT_DYNAMIC_BATCH = True
DYNB_MIN = 1
DYNB_MAX = 4096


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    return str(pth if pth.is_absolute() else (root / pth).resolve())


def load_config(run_dir: Path) -> dict:
    candidates = [
        run_dir / "config.resolved.json",
        run_dir / "config.json",
        ROOT / "config.json",
    ]
    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError("Could not find a config. Looked for: " + ", ".join(str(p) for p in candidates))

    cfg = json.loads(cfg_path.read_text())

    cfg_root = cfg_path.parent.resolve()
    paths = cfg.get("paths", {}) or {}
    if isinstance(paths, dict):
        paths = dict(paths)
        for k, v in list(paths.items()):
            if isinstance(v, str) and v.strip():
                paths[k] = _resolve_path(cfg_root, v)
        cfg["paths"] = paths

    return cfg


def find_checkpoint(run_dir: Path) -> Path:
    ckpt_roots = [run_dir / "checkpoints", run_dir]

    if PREFERRED_CKPT:
        for r in ckpt_roots:
            p = r / PREFERRED_CKPT
            if p.exists():
                return p
        raise FileNotFoundError(f"Preferred checkpoint not found: {PREFERRED_CKPT} (searched {ckpt_roots})")

    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last

    for r in ckpt_roots:
        ckpts = sorted(r.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            return ckpts[0]

    raise FileNotFoundError(f"No checkpoint found under {run_dir} (or {run_dir / 'checkpoints'}).")


def _strip_prefixes(key: str) -> str:
    prefixes = (
        "state_dict.",
        "model.",
        "module.",
        "_orig_mod.",
        "model._orig_mod.",
        "module.model.",
        "module._orig_mod.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    cleaned = {_strip_prefixes(k): v for k, v in state.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in cleaned.items() if k in model_sd}

    missing = [k for k in model_sd.keys() if k not in filtered]
    if missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} model keys (showing up to 20):\n  {preview}\nCheckpoint: {ckpt_path}"
        )

    model.load_state_dict(filtered, strict=True)


def prepare_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


class OneStepAR(nn.Module):
    """
    Pure single-step transition:
        y_next = base.forward_step(y, dt, g)

    Inputs:
      y : [B, S]
      dt: [B]   (normalized dt in [0,1])
      g : [B, G]
    Output:
      y_next: [B, S]
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.S = int(getattr(base, "S"))
        self.G = int(getattr(base, "G"))

    def forward(self, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.base.forward_step(y, dt, g)


def _make_inputs(step: OneStepAR, B: int, device: str, dtype: torch.dtype):
    S, G = step.S, step.G
    y = torch.randn(B, S, device=device, dtype=dtype)
    dt = torch.rand(B, device=device, dtype=dtype)  # normalized dt in [0,1]
    g = torch.randn(B, G, device=device, dtype=dtype) if G > 0 else torch.empty(B, 0, device=device, dtype=dtype)
    return y, dt, g


def export_pt2(step: OneStepAR, out_path: Path, device: str, dtype: torch.dtype, *, dynamic_batch: bool) -> None:
    step = prepare_model(step.to(device=device, dtype=dtype))

    # Use an example batch; if your model has int(...) guards, it will specialize to this.
    B_ex = 2
    y, dt, g = _make_inputs(step, B_ex, device, dtype)

    dyn = None
    if dynamic_batch:
        Bdim = torch.export.Dim("B", min=DYNB_MIN, max=DYNB_MAX)
        dyn = ({0: Bdim}, {0: Bdim}, {0: Bdim})

    ep = torch.export.export(step, (y, dt, g), dynamic_shapes=dyn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"Saved: {out_path}")


def try_export(step: OneStepAR, out_path: Path, device: str, preferred_dtype: torch.dtype, *, dynamic_batch: bool) -> None:
    try:
        export_pt2(step, out_path, device, preferred_dtype, dynamic_batch=dynamic_batch)
    except Exception as e:
        if device == "mps" and preferred_dtype != torch.float32:
            print(f"{device} export fp16 failed ({type(e).__name__}); retrying fp32...")
            export_pt2(step, out_path, device, torch.float32, dynamic_batch=dynamic_batch)
        else:
            raise


def _export_all_devices(step: OneStepAR, run_dir: Path, suffix: str, *, dynamic_batch: bool) -> None:
    # CPU
    try_export(step, run_dir / f"export_cpu_{suffix}.pt2", "cpu", torch.float32, dynamic_batch=dynamic_batch)

    # MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try_export(step, run_dir / f"export_mps_{suffix}.pt2", "mps", torch.float16, dynamic_batch=dynamic_batch)

    # CUDA
    if torch.cuda.is_available():
        try_export(step, run_dir / f"export_cuda_{suffix}.pt2", "cuda", torch.float16, dynamic_batch=dynamic_batch)


def main() -> None:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")

    cfg = load_config(run_dir)
    ckpt = find_checkpoint(run_dir)

    try:
        ckpt_disp = str(ckpt.relative_to(run_dir))
    except ValueError:
        ckpt_disp = str(ckpt)

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {ckpt_disp}")

    base = create_model(cfg)
    load_weights(base, ckpt)

    step = OneStepAR(base)
    print(f"Exporting 1-step AR | S={step.S}, G={step.G}")

    # 1) Always produce a working batch-1 export (matches your predictions.py usage)
    print("[info] Exporting batch-1 (guaranteed usable with B=1 inputs)...")
    # We force specialization to B=1 by exporting without dynamic shapes and using B_ex=1.
    # Do this by temporarily exporting with a tiny wrapper call.
    def export_b1(device: str, dtype: torch.dtype, out_path: Path) -> None:
        step1 = prepare_model(step.to(device=device, dtype=dtype))
        y, dt, g = _make_inputs(step1, 1, device, dtype)
        ep = torch.export.export(step1, (y, dt, g))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(ep, out_path)
        print(f"Saved: {out_path}")

    export_b1("cpu", torch.float32, run_dir / "export_cpu_b1_1step.pt2")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            export_b1("mps", torch.float16, run_dir / "export_mps_b1_1step.pt2")
        except Exception as e:
            print(f"[warn] mps fp16 b1 export failed ({type(e).__name__}); retrying fp32...")
            export_b1("mps", torch.float32, run_dir / "export_mps_b1_1step.pt2")
    if torch.cuda.is_available():
        try:
            export_b1("cuda", torch.float16, run_dir / "export_cuda_b1_1step.pt2")
        except Exception as e:
            print(f"[warn] cuda fp16 b1 export failed ({type(e).__name__}); retrying fp32...")
            export_b1("cuda", torch.float32, run_dir / "export_cuda_b1_1step.pt2")

    # 2) Attempt true dynamic-batch export
    if EXPORT_DYNAMIC_BATCH:
        print("[info] Attempting dynamic-batch export (requires SymInt-safe shape checks in model forward)...")
        try:
            _export_all_devices(step, run_dir, "dynB_1step", dynamic_batch=True)
        except Exception as e:
            print(f"[warn] dynamic-batch export failed: {type(e).__name__}: {e}")
            print("[note] This usually means your model forward path does int(...) / tuple(shape) guards on batch dims,")
            print("       which forces torch.export to specialize B. The batch-1 exports above are still valid.")


if __name__ == "__main__":
    main()
