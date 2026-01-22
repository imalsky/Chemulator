#!/usr/bin/env python3
"""
FlowMap Model Export Script (minimal)

Outputs are written into the resolved training work_dir (next to checkpoints):
  - CPU:  K=1, dynamic batch  -> export_k1_cpu.pt2
  - CUDA: dynamic B,K         -> export_bk_gpu.pt2
  - MPS:  dynamic B,K         -> export_bk_mps.pt2
"""

from __future__ import annotations


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "config" / "config.json"

MIN_BATCH, MAX_BATCH = 1, 4096
MIN_K, MAX_K = 1, 1024

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(ROOT)
sys.path.insert(0, str(SRC))

from model import create_model  # type: ignore  # noqa: E402


try:
    # Some checkpoints may contain Path objects.
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


def _resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def resolve_paths(cfg: Dict) -> Dict:
    cfg = dict(cfg)
    pcfg = dict(cfg.get("paths", {}))

    raw_dir = pcfg.get("raw_data_dir", "data/raw")
    processed_dir = pcfg.get("processed_data_dir", "data/processed")
    model_dir = pcfg.get("model_dir", "models")

    pcfg["raw_data_dir"] = str(_resolve_path(ROOT, str(raw_dir)))
    pcfg["processed_data_dir"] = str(_resolve_path(ROOT, str(processed_dir)))
    pcfg["model_dir"] = str(_resolve_path(ROOT, str(model_dir)))

    if "work_dir" in pcfg:
        pcfg["work_dir"] = str(_resolve_path(ROOT, str(pcfg["work_dir"])))

    cfg["paths"] = pcfg
    return cfg


def parse_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get(str(dtype_str).lower(), torch.float32)


def find_ckpt(directory: Path) -> Path:
    d = Path(directory)
    if (d / "best.ckpt").exists():
        return d / "best.ckpt"
    if (d / "last.ckpt").exists():
        return d / "last.ckpt"
    ckpts = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint found in {d}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        state = (
            payload.get("state_dict")
            or payload.get("model_state_dict")
            or payload.get("model")
            or payload.get("ema_model")
            or {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        )
    else:
        state = payload

    clean = {}
    for k, v in state.items():
        kk = k
        for prefix in ("model.", "module.", "_orig_mod."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        clean[kk] = v

    model.load_state_dict(clean, strict=False)


def optimize_inference(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


def export_cpu_k1(base: nn.Module, *, out_path: Path) -> None:
    print("\n" + "=" * 80)
    print("Exporting CPU (K=1, dynamic B)")
    print("=" * 80)

    device = "cpu"
    dtype = parse_dtype("float32")
    model = optimize_inference(base.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)

    B = 2
    y = torch.zeros(B, S_in, dtype=dtype, device=device)   # [B,S]
    dt = torch.zeros(B, 1, dtype=dtype, device=device)     # [B,1] (K=1)
    g = torch.zeros(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)

    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=(
            {0: Bdim},  # y:  [B,S]
            {0: Bdim},  # dt: [B,1]
            {0: Bdim},  # g:  [B,G]
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  wrote {out_path}")


def export_device_dynBK(base: nn.Module, *, device: str, out_path: Path, dtype_str: str) -> None:
    pretty = "GPU (CUDA)" if device == "cuda" else "MPS (Apple Silicon)"
    print("\n" + "=" * 80)
    print(f"Exporting {pretty} (dynamic B,K)")
    print("=" * 80)

    dtype = parse_dtype(dtype_str)
    model = optimize_inference(base.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", getattr(model, "global_dim", 0)) or 0)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)
    Kdim = torch.export.Dim("K", min=MIN_K, max=MAX_K)

    B, K = 2, 2
    y = torch.randn(B, S_in, dtype=dtype, device=device)        # [B,S]
    dt = torch.randn(B, K, 1, dtype=dtype, device=device)       # [B,K,1]
    g = torch.randn(B, G, dtype=dtype, device=device) if G > 0 else torch.empty(B, 0, dtype=dtype, device=device)

    ep = torch.export.export(
        model,
        (y, dt, g),
        dynamic_shapes=(
            {0: Bdim},          # y:  [B,S]
            {0: Bdim, 1: Kdim}, # dt: [B,K,1]
            {0: Bdim},          # g:  [B,G]
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    print(f"  wrote {out_path}")


def main() -> None:
    print("=" * 80)
    print("FlowMap Export")
    print("=" * 80)
    print(f"Config path: {CONFIG_PATH}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = resolve_paths(cfg)

    pcfg = cfg.get("paths", {})
    model_dir = Path(pcfg["model_dir"])
    run_name = str(cfg.get("run_name") or cfg.get("experiment_name") or cfg.get("name") or "run")
    work_dir = Path(pcfg.get("work_dir", model_dir / run_name))

    ckpt = find_ckpt(work_dir)
    print(f"Work dir: {work_dir}")
    print(f"Loading checkpoint: {ckpt}")

    base = create_model(cfg).eval().cpu()
    load_weights(base, ckpt)

    export_cpu_k1(base, out_path=work_dir / "export_k1_cpu.pt2")

    if torch.cuda.is_available():
        export_device_dynBK(base, device="cuda", out_path=work_dir / "export_bk_gpu.pt2", dtype_str="float16")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        export_device_dynBK(base, device="mps", out_path=work_dir / "export_bk_mps.pt2", dtype_str="float16")
    else:
        print("No CUDA or MPS detected; GPU/MPS export skipped.")


if __name__ == "__main__":
    main()
