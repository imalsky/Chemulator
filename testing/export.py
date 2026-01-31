#!/usr/bin/env python3
"""
export.py - Export FlowMap as a 1-step autoregressive module via torch.export (.pt2).

Intended location: <repo_root>/testing/export.py
(where <repo_root>/src contains main.py, model.py, etc.)

Edit RUN_DIR at the top to choose which trained run/checkpoint to export.
Outputs are written into RUN_DIR:
  - export_cpu_1step.pt2
  - export_mps_1step.pt2   (if available)
  - export_cuda_1step.pt2  (if available)
"""

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
RUN_DIR = (ROOT / "models" / "v4").resolve()
PREFERRED_CKPT = 'epochepoch=054-valval_loss=0.000476.ckpt'  # e.g. "last.ckpt" or "epoch005-val0.123456.ckpt"


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    return str(pth if pth.is_absolute() else (root / pth).resolve())


def load_config(run_dir: Path) -> dict:
    """
    Prefer the resolved config saved by training:
      RUN_DIR/config.resolved.json
    Fallbacks:
      RUN_DIR/config.json
      <repo_root>/config.json
    """
    candidates = [
        run_dir / "config.resolved.json",
        run_dir / "config.json",
        ROOT / "config.json",
    ]
    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "Could not find a config. Looked for: " + ", ".join(str(p) for p in candidates)
        )

    cfg = json.loads(cfg_path.read_text())

    # Resolve cfg["paths"] relative to the config file location.
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
    """
    Training writes checkpoints under RUN_DIR/checkpoints/ (Lightning).
    Prefer:
      - PREFERRED_CKPT if set
      - checkpoints/last.ckpt if present
      - newest *.ckpt under checkpoints/
    Also supports older layouts where ckpts were written directly into RUN_DIR.
    """
    ckpt_roots = [run_dir / "checkpoints", run_dir]

    if PREFERRED_CKPT:
        for r in ckpt_roots:
            p = r / PREFERRED_CKPT
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Preferred checkpoint not found: {PREFERRED_CKPT} (searched {ckpt_roots})"
        )

    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last

    for r in ckpt_roots:
        ckpts = sorted(r.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            return ckpts[0]

    raise FileNotFoundError(f"No checkpoint found under {run_dir} (or {run_dir / 'checkpoints'}).")


def _strip_prefixes(key: str) -> str:
    # Repeatedly strip common Lightning / DDP / torch.compile prefixes.
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

    # Drop anything not belonging to the bare model (e.g. criterion.*, optimizer.*, etc).
    model_sd = model.state_dict()
    filtered = {k: v for k, v in cleaned.items() if k in model_sd}

    missing = [k for k in model_sd.keys() if k not in filtered]
    if missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} model keys (showing up to 20):\n  {preview}\n"
            f"Checkpoint: {ckpt_path}"
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
    1-step autoregressive wrapper for export.

    Exports a pure single-step transition:
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


def export_pt2(step: nn.Module, out_path: Path, device: str, dtype: torch.dtype) -> None:
    step = prepare_model(step.to(device=device, dtype=dtype))

    S, G = step.S, step.G
    B_ex = 2  # exported batch will be specialized to this unless model is made symint-safe
    y = torch.randn(B_ex, S, device=device, dtype=dtype)
    dt = torch.rand(B_ex, device=device, dtype=dtype)  # normalized dt in [0,1]
    g = torch.randn(B_ex, G, device=device, dtype=dtype) if G > 0 else torch.empty(B_ex, 0, device=device, dtype=dtype)

    # IMPORTANT: model currently specializes batch, so use AUTO (or omit dynamic_shapes).
    B_dim = torch.export.Dim.AUTO
    dyn = ({0: B_dim}, {0: B_dim}, {0: B_dim})

    ep = torch.export.export(step, (y, dt, g), dynamic_shapes=dyn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"Saved: {out_path}")



def try_export(step: nn.Module, out_path: Path, device: str, preferred_dtype: torch.dtype) -> None:
    try:
        export_pt2(step, out_path, device, preferred_dtype)
    except Exception as e:
        if device == "mps" and preferred_dtype != torch.float32:
            print(f"{device} export fp16 failed ({type(e).__name__}); retrying fp32...")
            export_pt2(step, out_path, device, torch.float32)
        else:
            raise


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

    # CPU
    try_export(step, run_dir / "export_cpu_1step.pt2", "cpu", torch.float32)

    # MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try_export(step, run_dir / "export_mps_1step.pt2", "mps", torch.float16)

    # CUDA
    if torch.cuda.is_available():
        try_export(step, run_dir / "export_cuda_1step.pt2", "cuda", torch.float16)


if __name__ == "__main__":
    main()
