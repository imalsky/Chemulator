#!/usr/bin/env python3
"""
export.py - Export trained FlowMap model to TorchScript.

Outputs written to work_dir:
  - export_cpu.pt2   : CPU, K=1, dynamic batch
  - export_gpu.pt2   : CUDA, dynamic B and K
  - export_mps.pt2   : MPS, dynamic B and K
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


def load_config() -> dict:
    cfg = json.loads((ROOT / "config" / "config.json").read_text())
    paths = cfg.get("paths", {})
    for key in ("processed_data_dir", "model_dir", "work_dir"):
        if key in paths:
            p = Path(paths[key])
            paths[key] = str(p if p.is_absolute() else (ROOT / p).resolve())
    cfg["paths"] = paths
    return cfg


def find_checkpoint(directory: Path) -> Path:
    for name in ("best.ckpt", "last.ckpt"):
        if (directory / name).exists():
            return directory / name
    ckpts = sorted(directory.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint in {directory}")


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


def export_model(model: nn.Module, out_path: Path, device: str, dtype: torch.dtype, dynamic_k: bool) -> None:
    model = prepare_model(model.to(device))
    S, G = model.S, model.G
    
    B_dim = torch.export.Dim("batch", min=1, max=4096)
    K_dim = torch.export.Dim("K", min=1, max=1024) if dynamic_k else None
    
    B, K = 2, 2 if dynamic_k else 1
    y = torch.randn(B, S, dtype=dtype, device=device)
    dt = torch.randn(B, K, 1, dtype=dtype, device=device) if dynamic_k else torch.randn(B, 1, dtype=dtype, device=device)
    g = torch.randn(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)
    
    shapes = ({0: B_dim}, {0: B_dim, 1: K_dim}, {0: B_dim}) if dynamic_k else ({0: B_dim}, {0: B_dim}, {0: B_dim})
    
    ep = torch.export.export(model, (y, dt, g), dynamic_shapes=shapes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  Saved: {out_path}")


def main() -> None:
    print("=" * 60)
    print("  FLOWMAP EXPORT")
    print("=" * 60)
    
    cfg = load_config()
    work_dir = Path(cfg["paths"].get("work_dir", ROOT / "models" / "run"))
    
    ckpt = find_checkpoint(work_dir)
    print(f"  Checkpoint: {ckpt.name}")
    
    model = create_model(cfg)
    load_weights(model, ckpt)
    print(f"  Model: S={model.S}, G={model.G}")
    
    print("\nExporting CPU (K=1)...")
    export_model(model, work_dir / "export_cpu.pt2", "cpu", torch.float32, dynamic_k=False)
    
    if torch.cuda.is_available():
        print("\nExporting CUDA (dynamic K)...")
        export_model(model, work_dir / "export_gpu.pt2", "cuda", torch.float16, dynamic_k=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("\nExporting MPS (dynamic K)...")
        export_model(model, work_dir / "export_mps.pt2", "mps", torch.float16, dynamic_k=True)
    
    print("\n" + "=" * 60)
    print("  Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()