#!/usr/bin/env python3
"""
benchmark.py

CPU + GPU benchmark (B up to 8192), with VULCAN baseline.

- Uses REPO/models/final_model.
- CPU: export_k1_cpu.pt2 (K1, FP32)
- GPU: tries export_bk_gpu.pt2 (BF16); if warmup dtype mismatch, falls back to
       export_bk_gpu_fp32.pt2 (FP32).
- Adds gray baseline line at y=100 labeled "VULCAN, CPU".
- X limits autoset from curve data.
- Y limits fixed to [1, 1e7].
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

MODEL_SUBDIR = "models/final_model"
BENCH_K: int = 1

WARMUP_STEPS: int = 10
MEASURE_STEPS: int = 200

MAX_BATCH_CPU = 8192
MAX_BATCH_GPU = 8192

VULCAN_BASELINE_Y = 100.0
YMIN, YMAX = 10.0, 1.0e7

CPU_PT2 = "export_k1_cpu.pt2"
GPU_BF16_PT2 = "export_bk_gpu.pt2"
GPU_FP32_PT2 = "export_bk_gpu_fp32.pt2"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / MODEL_SUBDIR
NORM_PATH = ROOT / "data" / "processed" / "normalization.json"


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def _amp_ctx(dev: str, dtype: torch.dtype):
    if dev.startswith("cuda") and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _p2_batches(max_cap: int) -> List[int]:
    out: List[int] = []
    b = 1
    while b <= max_cap:
        out.append(b)
        b <<= 1
    return out


def _dims_from_normalization(norm_path: Path) -> Tuple[int, int]:
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing {norm_path}")
    j = json.loads(norm_path.read_text())
    meta = j.get("meta", j)
    species = meta.get("species_variables") or j.get("species_variables") or []
    gvars = meta.get("global_variables") or j.get("global_variables") or []
    if not species:
        raise RuntimeError("species_variables missing in normalization.json")
    return len(species), len(gvars or [])


def _make_inputs_k1(B: int, S_in: int, G: int, dtype: torch.dtype, device: str):
    y = torch.randn(B, S_in, dtype=dtype, device=device).contiguous()
    dt = torch.randn(B, 1, dtype=dtype, device=device).contiguous()
    g = (
        torch.randn(B, G, dtype=dtype, device=device).contiguous()
        if G > 0
        else torch.empty(B, 0, dtype=dtype, device=device)
    )
    return y, dt, g


def _make_inputs_bk(B: int, K: int, S_in: int, G: int, dtype: torch.dtype, device: str):
    y = torch.randn(B, S_in, dtype=dtype, device=device).contiguous()
    dt = torch.randn(B, K, 1, dtype=dtype, device=device).contiguous()
    g = (
        torch.randn(B, G, dtype=dtype, device=device).contiguous()
        if G > 0
        else torch.empty(B, 0, dtype=dtype, device=device)
    )
    return y, dt, g


def _load_pt2(path: Path):
    from torch.export import load as torch_export_load

    ep = torch_export_load(str(path))
    return ep.module()


def _is_dtype_mismatch_err(e: Exception) -> bool:
    msg = str(e)
    return (
        "must have the same dtype" in msg
        or ("Float" in msg and "BFloat16" in msg)
        or ("Tensor dtype mismatch" in msg)
    )


def _autoset_log_xlim(ax, curves: List[Tuple[str, List[int], List[float]]]) -> None:
    xs: List[float] = []
    for _, bs, _ in curves:
        xs.extend([float(b) for b in bs if b > 0])
    if not xs:
        return
    pad = 1.25
    ax.set_xlim(max(min(xs) / pad, 1e-6), max(xs) * pad)


def main() -> None:
    print("=" * 80)
    print(f"Benchmarking exported models in: {WORK_DIR}")
    print("=" * 80)
    print(f"__file__={Path(__file__).resolve()}")

    if not WORK_DIR.exists():
        print(f"Model directory does not exist: {WORK_DIR}")
        return

    try:
        S_in, G = _dims_from_normalization(NORM_PATH)
    except Exception as e:
        print(f"Failed to read dims from {NORM_PATH}: {e}")
        return

    curves: List[Tuple[str, List[int], List[float]]] = []
    png_out = WORK_DIR / f"plots/bench_throughput_k{BENCH_K}.png"

    # ---------------- CPU ----------------
    cpu_path = WORK_DIR / CPU_PT2
    if not cpu_path.exists():
        print(f"- CPU: missing {cpu_path}")
    else:
        model_cpu = _load_pt2(cpu_path)
        dev = "cpu"
        dtype = torch.float32
        batches = _p2_batches(MAX_BATCH_CPU)

        print(
            f"- CPU: using {cpu_path.name} (PT2), sig=K1, device={dev}, dtype={dtype}, "
            f"S_in={S_in}, G={G}, K=1, B∈{batches}"
        )

        try:
            inp = _make_inputs_k1(1, S_in, G, dtype, dev)
            with torch.inference_mode():
                _ = model_cpu(*inp)
        except Exception as e:
            print(f"  CPU warmup failed: {e}")
        else:
            thr: List[float] = []
            ok_batches: List[int] = []
            for B in batches:
                try:
                    inp = _make_inputs_k1(B, S_in, G, dtype, dev)

                    for _ in range(WARMUP_STEPS):
                        with torch.inference_mode():
                            _ = model_cpu(*inp)

                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(MEASURE_STEPS):
                            _ = model_cpu(*inp)
                    elapsed = time.perf_counter() - t0

                    sps = (MEASURE_STEPS * B) / max(elapsed, 1e-12)
                    thr.append(float(sps))
                    ok_batches.append(B)
                    print(f"    B={B:>5d} -> {sps:,.1f} samples/s")
                except Exception as e:
                    print(f"    B={B:>5d} failed: {e}")
                    break

            if thr:
                curves.append(("CPU", ok_batches, thr))

    # ---------------- GPU ----------------
    if not torch.cuda.is_available():
        print("- GPU: CUDA not available, skipping")
    else:
        dev = "cuda"
        batches = _p2_batches(MAX_BATCH_GPU)

        gpu_path_bf16 = WORK_DIR / GPU_BF16_PT2
        gpu_path_fp32 = WORK_DIR / GPU_FP32_PT2

        chosen_path: Path | None = None
        chosen_dtype: torch.dtype | None = None
        model_gpu = None

        # Try BF16 export first
        if gpu_path_bf16.exists():
            try:
                model_gpu = _load_pt2(gpu_path_bf16)
                chosen_path = gpu_path_bf16
                chosen_dtype = torch.bfloat16
                inp = _make_inputs_bk(1, BENCH_K, S_in, G, chosen_dtype, dev)
                with torch.inference_mode(), _amp_ctx(dev, chosen_dtype):
                    _ = model_gpu(*inp)
                _sync(dev)
            except Exception as e:
                if _is_dtype_mismatch_err(e) and gpu_path_fp32.exists():
                    print(f"- GPU: BF16 warmup failed ({e}) -> using {gpu_path_fp32.name} for GPU benchmark")
                    model_gpu = None
                    chosen_path = None
                    chosen_dtype = None
                else:
                    print(f"- GPU: warmup failed: {e}")
                    model_gpu = None

        # Fall back to FP32 export
        if model_gpu is None and gpu_path_fp32.exists():
            try:
                model_gpu = _load_pt2(gpu_path_fp32)
                chosen_path = gpu_path_fp32
                chosen_dtype = torch.float32
                inp = _make_inputs_bk(1, BENCH_K, S_in, G, chosen_dtype, dev)
                with torch.inference_mode(), _amp_ctx(dev, chosen_dtype):
                    _ = model_gpu(*inp)
                _sync(dev)
            except Exception as e:
                print(f"- GPU: FP32 warmup failed: {e}")
                model_gpu = None

        if model_gpu is None or chosen_path is None or chosen_dtype is None:
            print("- GPU: no usable artifact, skipping")
        else:
            print(
                f"- GPU: using {chosen_path.name} (PT2), sig=BK, device={dev}, dtype={chosen_dtype}, "
                f"S_in={S_in}, G={G}, K={BENCH_K}, B∈{batches}"
            )

            thr: List[float] = []
            ok_batches: List[int] = []
            amp_ctx = _amp_ctx(dev, chosen_dtype)

            for B in batches:
                try:
                    inp = _make_inputs_bk(B, BENCH_K, S_in, G, chosen_dtype, dev)

                    for _ in range(WARMUP_STEPS):
                        with torch.inference_mode(), amp_ctx:
                            _ = model_gpu(*inp)
                    _sync(dev)

                    t0 = time.perf_counter()
                    with torch.inference_mode(), amp_ctx:
                        for _ in range(MEASURE_STEPS):
                            _ = model_gpu(*inp)
                    _sync(dev)

                    elapsed = time.perf_counter() - t0
                    sps = (MEASURE_STEPS * B * BENCH_K) / max(elapsed, 1e-12)
                    thr.append(float(sps))
                    ok_batches.append(B)
                    print(f"    B={B:>5d} -> {sps:,.1f} samples/s")
                except Exception as e:
                    print(f"    B={B:>5d} failed: {e}")
                    break

            if thr:
                curves.append(("GPU", ok_batches, thr))

    if not curves:
        print("No curves to plot.")
        return

    try:
        plt.style.use("science.mplstyle")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, bs, th in curves:
        ax.plot(bs, th, marker="o", label=label)

    ax.axhline(y=VULCAN_BASELINE_Y, linewidth=2, color="gray", linestyle="--", label="VULCAN, CPU")

    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Throughput (samples/second)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Fixed y-limits per request
    ax.set_ylim(YMIN, YMAX)

    # X-limits from data
    _autoset_log_xlim(ax, curves)

    ax.legend(loc="best")
    ax.set_box_aspect(1)

    fig.tight_layout()
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_out), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {png_out}")


if __name__ == "__main__":
    main()