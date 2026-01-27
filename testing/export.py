#!/usr/bin/env python3
"""
export.py - Export FlowMap as a 1-step autoregressive module (K=1) via torch.export (.pt2).

Edit RUN_DIR at the top to choose which trained run/checkpoint to export.
Outputs written into RUN_DIR:
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

from model import create_model

# -----------------------------
# CHOOSE YOUR RUN HERE (no args, no prompts)
# -----------------------------
RUN_DIR = (ROOT / "models" / "v1").resolve()
PREFERRED_CKPT = None  # set to "best.ckpt" or "last.ckpt" to force; otherwise auto-pick


def load_config() -> dict:
    cfg = json.loads((ROOT / "config" / "config.json").read_text())
    paths = cfg.get("paths", {})
    for key in ("processed_data_dir", "model_dir", "work_dir"):
        if key in paths:
            p = Path(paths[key])
            paths[key] = str(p if p.is_absolute() else (ROOT / p).resolve())
    cfg["paths"] = paths
    return cfg


def find_checkpoint(run_dir: Path) -> Path:
    if PREFERRED_CKPT:
        p = run_dir / PREFERRED_CKPT
        if p.exists():
            return p
        raise FileNotFoundError(f"Preferred checkpoint not found: {p}")

    for name in ("best.ckpt", "last.ckpt"):
        p = run_dir / name
        if p.exists():
            return p

    ckpts = sorted(run_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    clean = {}
    for k, v in state.items():
        for prefix in ("model.", "module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    model.load_state_dict(clean, strict=False)


def prepare_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


def infer_dt_rank(base: nn.Module, device: str, dtype: torch.dtype) -> int:
    """
    Infer whether base.forward expects dt shaped [B,K], [B,K,1], or [B] (for K=1).
    Returns: 1, 2, or 3 (dt rank).
    """
    base = prepare_model(base.to(device))
    S, G = int(getattr(base, "S")), int(getattr(base, "G"))
    B = 2
    y = torch.randn(B, S, device=device, dtype=dtype)
    g = torch.randn(B, G, device=device, dtype=dtype) if G > 0 else torch.empty(B, 0, device=device, dtype=dtype)

    # Try in order: [B,1], [B,1,1], [B]
    candidates = [
        (2, torch.randn(B, 1, device=device, dtype=dtype)),
        (3, torch.randn(B, 1, 1, device=device, dtype=dtype)),
        (1, torch.randn(B, device=device, dtype=dtype)),
    ]
    for rank, dt in candidates:
        try:
            _ = base(y, dt, g)
            return rank
        except Exception:
            pass
    raise RuntimeError("Could not infer dt shape for model forward (tried [B,1], [B,1,1], [B]).")


class OneStepAR(nn.Module):
    """1-step autoregressive wrapper that fixes K=1 for export (avoids dynamic range(K) issues)."""

    def __init__(self, base: nn.Module, dt_rank: int):
        super().__init__()
        self.base = base
        self.dt_rank = int(dt_rank)
        self.S = int(getattr(base, "S"))
        self.G = int(getattr(base, "G"))

    def forward(self, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor):
        # Export signature uses dt as [B]; reshape to what base expects for K=1.
        if self.dt_rank == 3:
            dt_in = dt.view(-1, 1, 1)     # [B,1,1]
        elif self.dt_rank == 2:
            dt_in = dt.view(-1, 1)        # [B,1]
        else:
            dt_in = dt.view(-1)           # [B]
        return self.base(y, dt_in, g)


def export_pt2(step: nn.Module, out_path: Path, device: str, dtype: torch.dtype) -> None:
    step = prepare_model(step.to(device))

    S, G = step.S, step.G
    B_ex = 2
    y = torch.randn(B_ex, S, device=device, dtype=dtype)
    dt = torch.randn(B_ex, device=device, dtype=dtype)  # [B] always for the wrapper
    g = torch.randn(B_ex, G, device=device, dtype=dtype) if G > 0 else torch.empty(B_ex, 0, device=device, dtype=dtype)

    B_dim = torch.export.Dim("batch", min=1, max=4096)
    dyn = ({0: B_dim}, {0: B_dim}, {0: B_dim})

    ep = torch.export.export(step, (y, dt, g), dynamic_shapes=dyn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"Saved: {out_path}")


def try_export(step: nn.Module, out_path: Path, device: str, preferred_dtype: torch.dtype) -> None:
    try:
        export_pt2(step, out_path, device, preferred_dtype)
    except Exception as e:
        # MPS fp16 can be finicky; fall back to fp32.
        if device == "mps" and preferred_dtype != torch.float32:
            print(f"{device} export fp16 failed ({type(e).__name__}); retrying fp32...")
            export_pt2(step, out_path, device, torch.float32)
        else:
            raise


def main() -> None:
    cfg = load_config()
    run_dir = RUN_DIR
    ckpt = find_checkpoint(run_dir)

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {ckpt.name}")

    base = create_model(cfg)
    load_weights(base, ckpt)

    dt_rank = infer_dt_rank(base, "cpu", torch.float32)
    step = OneStepAR(base, dt_rank=dt_rank)
    print(f"Exporting 1-step AR | S={step.S}, G={step.G}, base_dt_rank={dt_rank}")

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
