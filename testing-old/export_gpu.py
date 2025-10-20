#!/usr/bin/env python3
"""
Benchmark torch.export artifacts on BOTH GPU (CUDA or MPS) and CPU.

Artifacts expected:
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
from typing import Iterable

import torch

# ------------------------- SETTINGS -------------------------------------------

MODEL_NAME: str = "koopman-v2"

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
    """Load .meta.json if present; otherwise return defaults."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)
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
    if s in ("float64", "double", "fp64"):return torch.float64
    if s in ("float32", "fp32", "float"): return torch.float32
    raise ValueError(f"Unsupported dtype in meta: {s}")

def _device_from_meta(meta_dev: str) -> torch.device:
    d = meta_dev.lower()
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("Meta expects CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if d == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Meta expects MPS, but MPS is not available.")
        return torch.device("mps")
    if d == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported meta device: {meta_dev}")

def _enforce_mps_dtype_policy(device: torch.device, dtype: torch.dtype, where: str) -> bool:
    """
    On MPS, disallow fp16/bf16 for exported graphs (MPS NDArray matmul asserts).
    Returns True if allowed to proceed, else False.
    """
    if device.type != "mps":
        return True
    if dtype in (torch.float16, torch.bfloat16):
        print(f"[{where}] MPS backend: fp16/bf16 exports are not supported for NDArray matmul.")
        print("       Action: re-export your MPS artifact in float32 and set dtype:'float32' in the .meta.json.")
        return False
    return True

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

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
                _sync(device)

                # Timed
                t0 = time.perf_counter()
                for _ in range(N_ITERS):
                    _ = module(y, dt, g)
                _sync(device)
                t1 = time.perf_counter()

                ms_per_iter = (t1 - t0) * 1e3 / N_ITERS
                us_per_sample = (t1 - t0) * 1e6 / (N_ITERS * B)
                sps = 1e6 / us_per_sample
                print(f"{B}\t{ms_per_iter:.3f}\t{us_per_sample:.1f}\t{sps:,.0f}")

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "resource" in msg:
                    print(f"{B}\tOOM")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif device.type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    break
                raise

def _run_cpu() -> bool:
    if not CPU_EXPORT_PATH.exists():
        print(f"[CPU] Missing artifact: {CPU_EXPORT_PATH}")
        return False

    meta = _load_meta(CPU_META_PATH, default_device="cpu")
    device = torch.device("cpu")
    dtype  = _dtype_from_str(meta["dtype"])

    ep = torch.export.load(str(CPU_EXPORT_PATH))
    mod = ep.module()
    _bench(mod, device, dtype, int(meta["S_in"]), int(meta["G"]), label="CPU bench")
    return True

def _run_gpu() -> bool:
    if not GPU_EXPORT_PATH.exists():
        print(f"[GPU] Missing artifact: {GPU_EXPORT_PATH}")
        return False

    # Prefer CUDA if available, else MPS
    default_dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cuda")
    meta = _load_meta(GPU_META_PATH, default_device=default_dev)

    # Resolve/validate device from meta
    try:
        device = _device_from_meta(meta["device"])
    except SystemExit as e:
        print(f"[GPU] Skipped: {e}")
        return False

    dtype  = _dtype_from_str(meta["dtype"])

    # Enforce MPS 16-bit policy
    if not _enforce_mps_dtype_policy(device, dtype, where="GPU bench"):
        print("[GPU] Skipping run. Re-export in float32 for MPS.")
        return False

    ep = torch.export.load(str(GPU_EXPORT_PATH))
    mod = ep.module()
    _bench(mod, device, dtype, int(meta["S_in"]), int(meta["G"]), label="GPU bench")
    return True

def main():
    any_run = False

    # GPU first
    try:
        any_run |= _run_gpu()
    except Exception as e:
        print(f"[GPU] Error: {e}")

    # CPU next
    try:
        any_run |= _run_cpu()
    except Exception as e:
        print(f"[CPU] Error: {e}")

    if not any_run:
        raise SystemExit("No artifacts benchmarked. Export CPU/GPU artifacts first (and use float32 on MPS).")

if __name__ == "__main__":
    main()
