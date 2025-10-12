#!/usr/bin/env python3
"""
Benchmark torch.export artifacts on BOTH GPU (CUDA or MPS) and CPU.

Looks for:
  models/<MODEL_NAME>/export_k1_gpu.pt2[.meta.json]
  models/<MODEL_NAME>/export_k1_cpu.pt2[.meta.json]

Run:
  python testing/bench_k1_cpu_gpu.py
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

# ------------------------- SETTINGS -------------------------------------------

MODEL_NAME: str = "koopman-v1"

MODEL_DIR      = Path("../models") / MODEL_NAME
GPU_BASENAME   = "export_k1_gpu.pt2"
CPU_BASENAME   = "export_k1_cpu.pt2"

GPU_EXPORT_PATH = MODEL_DIR / GPU_BASENAME
GPU_META_PATH   = MODEL_DIR / (GPU_BASENAME + ".meta.json")

CPU_EXPORT_PATH = MODEL_DIR / CPU_BASENAME
CPU_META_PATH   = MODEL_DIR / (CPU_BASENAME + ".meta.json")

# Batch sizes to test
B_LIST: Iterable[int] = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_WARMUP = 3
N_ITERS  = 5
DT_NORM_VALUE = 0.5

# For MPS, allow CPU fallback for unsupported ops (safer for ad-hoc benches)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ------------------------------------------------------------------------------

def _load_meta(path: Path, default_device: str, default_dtype: str = "float32",
               default_S_in: int = 12, default_G: int = 2) -> dict:
    """Load .meta.json if present, else return reasonable defaults."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        # Fill any missing fields with defaults
        m.setdefault("device", default_device)
        m.setdefault("dtype", default_dtype)
        m.setdefault("S_in", default_S_in)
        m.setdefault("G", default_G)
        return m
    return {"device": default_device, "dtype": default_dtype, "S_in": default_S_in, "G": default_G}

def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float16", "half", "fp16"):  return torch.float16
    if s in ("bfloat16", "bf16"):         return torch.bfloat16
    if s in ("float32", "fp32", "float"): return torch.float32
    raise ValueError(f"Unsupported dtype in meta: {s}")

def _device_from_meta(meta_dev: str) -> torch.device:
    meta_dev = meta_dev.lower()
    if meta_dev == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Meta expects CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if meta_dev == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Meta expects MPS, but MPS is not available.")
        return torch.device("mps")
    if meta_dev == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported meta device: {meta_dev}")

def _bench(module: torch.nn.Module, device: torch.device, dtype: torch.dtype,
           S_in: int, G: int, label: str) -> None:
    """
    Benchmark an ExportedProgram-backed module. Do NOT call .eval() or .to() on it.
    Inputs must be created on the same device/dtype as the exported params.
    """
    print(f"\n[{label}] device={device} dtype={dtype}")
    print("B\tms/iter\tus/sample\tsamples/s")
    with torch.inference_mode():
        for B in B_LIST:
            try:
                y  = torch.randn(B, S_in, device=device, dtype=dtype)
                dt = torch.full((B, 1, 1), DT_NORM_VALUE, device=device, dtype=dtype)
                g  = torch.randn(B, G, device=device, dtype=dtype) if G > 0 else torch.zeros(B, 0, device=device, dtype=dtype)

                # Warmups
                for _ in range(N_WARMUP):
                    _ = module(y, dt, g)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()

                # Timed
                t0 = time.perf_counter()
                for _ in range(N_ITERS):
                    _ = module(y, dt, g)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                t1 = time.perf_counter()

                ms_per_iter = (t1 - t0) * 1e3 / N_ITERS
                us_per_sample = (t1 - t0) * 1e6 / (N_ITERS * B)
                sps = 1e6 / us_per_sample
                print(f"{B}\t{ms_per_iter:.3f}\t{us_per_sample:.1f}\t{sps:,.0f}")

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "resource" in msg:
                    print(f"{B}\tOOM")
                    break
                raise

def _run_cpu() -> Optional[dict]:
    if not CPU_EXPORT_PATH.exists():
        print(f"[CPU] Missing artifact: {CPU_EXPORT_PATH}")
        return None
    meta = _load_meta(CPU_META_PATH, default_device="cpu")
    device = _device_from_meta(meta["device"])  # should be cpu
    if device.type != "cpu":
        print(f"[CPU] Warning: meta device is '{meta['device']}', forcing CPU anyway.")
        device = torch.device("cpu")
    dtype  = _dtype_from_str(meta["dtype"])
    if dtype not in (torch.float32,):
        # Exported CPU path is almost always fp32; if meta says otherwise, try but warn.
        print(f"[CPU] Note: dtype in meta is '{meta['dtype']}', proceeding.")

    ep = torch.export.load(str(CPU_EXPORT_PATH))
    mod = ep.module()  # executable
    _bench(mod, device, dtype, int(meta["S_in"]), int(meta["G"]), label="CPU bench")
    return meta

def _run_gpu() -> Optional[dict]:
    if not GPU_EXPORT_PATH.exists():
        print(f"[GPU] Missing artifact: {GPU_EXPORT_PATH}")
        return None
    meta = _load_meta(GPU_META_PATH, default_device=("cuda" if torch.cuda.is_available() else "mps"))
    device = _device_from_meta(meta["device"])
    dtype  = _dtype_from_str(meta["dtype"])

    ep = torch.export.load(str(GPU_EXPORT_PATH))
    mod = ep.module()
    _bench(mod, device, dtype, int(meta["S_in"]), int(meta["G"]), label="GPU bench")
    return meta

def main():
    any_run = False

    # GPU first (if present)
    try:
        meta_gpu = _run_gpu()
        any_run = any_run or (meta_gpu is not None)
    except SystemExit as e:
        print(f"[GPU] Skipped: {e}")
    except Exception as e:
        print(f"[GPU] Error: {e}")

    # CPU next
    try:
        meta_cpu = _run_cpu()
        any_run = any_run or (meta_cpu is not None)
    except SystemExit as e:
        print(f"[CPU] Skipped: {e}")
    except Exception as e:
        print(f"[CPU] Error: {e}")

    if not any_run:
        raise SystemExit("No artifacts found to benchmark. Export CPU/GPU artifacts first.")

if __name__ == "__main__":
    main()
