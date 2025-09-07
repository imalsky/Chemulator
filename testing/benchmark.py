#!/usr/bin/env python3
"""CPU-only benchmark: Flow-map DeepONet inference vs batch size (K=1), with torch.compile and 4 CPU threads."""
from __future__ import annotations

import os, sys, time, gc
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------- CPU env BEFORE importing torch ----------
CPU_THREADS = 1  # fixed to 4 threads
os.environ["CUDA_VISIBLE_DEVICES"]             = ""
os.environ["OMP_NUM_THREADS"]                  = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]                  = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]             = str(CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parent

import numpy as np
import torch
import matplotlib.pyplot as plt
import json5  # <-- supports JSONC (comments, trailing commas)

# ---------- Config ----------
MODEL_STR   = "flowmap-deeponet"
MODEL_DIR   = REPO_ROOT / "models" / MODEL_STR
MODEL_FILE  = "complete_model_exported_k1.pt2"  # or "complete_model_exported.pt2"
CFG_PATH    = REPO_ROOT / "config" / "config.jsonc"

# We benchmark one time jump per call (K=1):
NUM_TIME_POINTS = 1

# Batch sizes to scan:
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]

# Timings
WARMUP_CALLS           = 50
CALLS_PER_MEASUREMENT  = 2
REPEATS                = 5

# Use torch.compile on the loaded exported module (CPU / Inductor)
USE_COMPILE = True

PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def _set_torch_threads(n: int) -> None:
    torch.set_num_threads(int(n))
    torch.set_num_interop_threads(1)  # often better on CPU

def _safe_style():
    try:
        plt.style.use("science.mplstyle")
    except Exception:
        pass

def _load_exported_model() -> Tuple[Any, Dict[str, Any], torch.device, Any]:
    with open(CFG_PATH, "r") as f:
        cfg = json5.load(f)  # <-- parse jsonc

    device = torch.device("cpu")
    _set_torch_threads(CPU_THREADS)

    mp = MODEL_DIR / MODEL_FILE
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    from torch.export import load as tload
    prog = tload(str(mp))
    fn   = prog.module()
    if USE_COMPILE:
        try:
            fn = torch.compile(fn, backend="inductor", mode="max-autotune", fullgraph=True)
            print("[INFO] torch.compile enabled (inductor, CPU)")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    return prog, cfg, device, fn

def _create_batch(bs: int, cfg: dict, device: torch.device, num_t: int = NUM_TIME_POINTS) -> Dict[str, torch.Tensor]:
    S = len(cfg["data"]["species_variables"])
    G = len(cfg["data"]["global_variables"])
    if num_t != 1:
        raise ValueError("This benchmark is for K=1 only. Set NUM_TIME_POINTS = 1.")
    return {
        "y0_norm":      torch.randn(bs, S, device=device, dtype=torch.float32).contiguous(),
        "globals_norm": (torch.randn(bs, G, device=device, dtype=torch.float32).contiguous() if G > 0
                         else torch.zeros(bs, 0, device=device, dtype=torch.float32)),
        "dt_norm":      torch.tensor([0.5], device=device, dtype=torch.float32),  # one normalized jump
    }

@torch.inference_mode()
def _run_calls(fn, batch: Dict[str, torch.Tensor], calls: int) -> None:
    y0, g, dt = batch["y0_norm"], batch["globals_norm"], batch["dt_norm"]
    for _ in range(calls):
        _ = fn(y0, g, dt)

@torch.inference_mode()
def _bench_once(fn, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    # Warmup includes potential torch.compile first-run compile cost
    _run_calls(fn, batch, WARMUP_CALLS)
    xs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        _run_calls(fn, batch, CALLS_PER_MEASUREMENT)
        xs.append((time.perf_counter_ns() - t0) / 1e3 / CALLS_PER_MEASUREMENT)  # us/call
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def _plot(batch_sizes: List[int], mean_us: List[float], std_us: List[float], threads: int) -> None:
    _safe_style()
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Each call does B * K predictions; here K=1, so per-jump = mean_us / B
    per_jump_mean = [m / bs for m, bs in zip(mean_us, batch_sizes)]
    per_jump_std  = [s / bs for s, bs in zip(std_us,  batch_sizes)]
    ax1.errorbar(batch_sizes, per_jump_mean, yerr=per_jump_std, marker="o", markersize=6, capsize=4, linewidth=2)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=10)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Inference time per prediction (all species)")

    # Throughput in jumps/s
    #thr_jumps = [bs / (m / 1e6) for bs, m in zip(batch_sizes, mean_us)]
    #ax2.plot(batch_sizes, thr_jumps, marker="s", markersize=6, linewidth=2)
    #ax2.set_xscale("log", base=2); ax2.set_yscale("log", base=10)
    #ax2.set_xlabel("Batch size"); ax2.set_ylabel("Throughput (jumps/s)")
    #ax2.set_title("Inference throughput (K=1)")#

    fig.tight_layout()
    out_png = PLOT_DIR / "benchmark_cpu_k1.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")

def _print_model_summary(cfg: dict) -> None:
    mc, dc = cfg.get("model", {}), cfg.get("data", {})
    print("\nModel Configuration:")
    print(f"  Branch layers:    {mc.get('branch_layers', 'N/A')}")
    print(f"  Trunk layers:     {mc.get('trunk_layers', 'N/A')}")
    print(f"  Latent dim:       {mc.get('latent_dim', 'N/A')}")
    print("\nData Configuration:")
    print(f"  Species (S):      {len(dc.get('species_variables', []))}")
    print(f"  Globals (G):      {len(dc.get('global_variables', []))}")
    print(f"  K (jumps/call):   1")

def main():
    print("=" * 60)
    print("FLOW-MAP CPU INFERENCE BENCHMARK (K=1)")
    print("=" * 60)
    print(f"CPU threads (intra-op): {CPU_THREADS}")
    print(f"Warmup calls:           {WARMUP_CALLS}")
    print(f"Calls/measurement:      {CALLS_PER_MEASUREMENT}")
    print(f"Repeats:                {REPEATS}")

    gc.disable()
    torch.set_grad_enabled(False)

    try:
        _, cfg, dev, fn = _load_exported_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    _print_model_summary(cfg)

    print("\nPreallocating input batches...")
    batches = {bs: _create_batch(bs, cfg, dev, NUM_TIME_POINTS) for bs in BATCH_SIZES}

    means, stds = [], []
    print("\nRunning benchmarks (amortized timing, K=1)...")
    print("-" * 75)
    print(f"{'Batch':>8} | {'Mean (μs)':>12} | {'Std (μs)':>10} | {'μs/jump':>12} | {'Jumps/s':>12}")
    print("-" * 75)

    for bs in BATCH_SIZES:
        try:
            mean_us, std_us = _bench_once(fn, batches[bs])
            means.append(mean_us); stds.append(std_us)
            us_per_jump = mean_us / bs
            thr_jumps = bs / (mean_us / 1e6)
            print(f"{bs:>8} | {mean_us:>12.2f} | {std_us:>10.2f} | {us_per_jump:>12.3f} | {thr_jumps:>10.0f}")
        except Exception as e:
            print(f"{bs:>8} | Error: {e}")
            means.append(float('nan')); stds.append(float('nan'))

    print("-" * 75)

    valid = [(bs, m, s) for bs, m, s in zip(BATCH_SIZES, means, stds) if not np.isnan(m)]
    if valid:
        v_bs, v_m, v_s = zip(*valid)
        thr_jumps = [bs / (m / 1e6) for bs, m in zip(v_bs, v_m)]
        bidx = int(np.argmax(thr_jumps))
        print(f"\nOptimal batch size: {v_bs[bidx]}")
        print(f"Peak throughput:    {thr_jumps[bidx]:.0f} jumps/s")
        print(f"Time per jump:      {v_m[bidx] / v_bs[bidx]:.3f} μs")

        S = len(cfg["data"]["species_variables"])
        G = len(cfg["data"]["global_variables"])
        bytes_per_sample = 4 * (S + G + 1 * S)  # y0[S] + g[G] + (K=1)*S output payload-ish
        print(f"\nMemory/sample:      {bytes_per_sample / 1024:.1f} KB")
        print(f"Memory at peak BS:  {v_bs[bidx] * bytes_per_sample / (1024**2):.1f} MB")

        _plot(list(v_bs), list(v_m), list(v_s), CPU_THREADS)

    gc.enable()
    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()
