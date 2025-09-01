#!/usr/bin/env python3
"""CPU-only benchmark: AE-DeepONet model inference time vs batch size."""
from __future__ import annotations

import os, sys, time, gc, json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------- CPU env BEFORE importing torch ----------
CPU_THREADS = min(os.cpu_count() or 4, 6)
os.environ["CUDA_VISIBLE_DEVICES"]             = ""
os.environ["OMP_NUM_THREADS"]                  = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]                  = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]             = str(CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("ACCELERATE_MATMUL_MULTITHREADING", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- Config ----------
MODEL_STR   = "big_deep"
MODEL_DIR   = Path(f"../models/{MODEL_STR}")
MODEL_FILE  = "complete_model_exported.pt2"
CONFIG_FILE = "config.json"

BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
NUM_TIME_POINTS = 100

WARMUP_CALLS           = 20
CALLS_PER_MEASUREMENT  = 16
REPEATS                = 3

PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def _set_torch_threads(n: int) -> None:
    torch.set_num_threads(int(n))
    torch.set_num_interop_threads(max(1, int(n // 2)))

def _safe_style():
    try:
        plt.style.use("science.mplstyle")
    except Exception:
        pass

def _load_exported_model() -> Tuple[Any, Dict[str, Any], torch.device, Any]:
    cfg_path = MODEL_DIR / CONFIG_FILE
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    device = torch.device("cpu")
    _set_torch_threads(CPU_THREADS)

    mp = MODEL_DIR / MODEL_FILE
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    from torch.export import load as tload
    prog = tload(str(mp))
    fn   = prog.module()
    return prog, cfg, device, fn

def _create_batch(bs: int, cfg: dict, device: torch.device, num_t: int = NUM_TIME_POINTS) -> Dict[str, torch.Tensor]:
    S = len(cfg["data"]["species_variables"])
    G = len(cfg["data"]["global_variables"])
    return {
        "x0_norm":      torch.randn(bs, S, device=device, dtype=torch.float32).contiguous(),
        "globals_norm": torch.randn(bs, G, device=device, dtype=torch.float32).contiguous(),
        "trunk_times":  torch.linspace(0.0, 1.0, steps=num_t, device=device, dtype=torch.float32),
    }

@torch.inference_mode()
def _run_calls(fn, batch: Dict[str, torch.Tensor], calls: int) -> None:
    x0, g, t = batch["x0_norm"], batch["globals_norm"], batch["trunk_times"]
    for _ in range(calls):
        _ = fn(x0, g, t)

@torch.inference_mode()
def _bench_once(fn, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    _run_calls(fn, batch, WARMUP_CALLS)
    xs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        _run_calls(fn, batch, CALLS_PER_MEASUREMENT)
        xs.append((time.perf_counter_ns() - t0) / 1e3 / CALLS_PER_MEASUREMENT)  # us/call
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def _plot(batch_sizes: List[int], mean_us: List[float], std_us: List[float], threads: int, num_times: int) -> None:
    _safe_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    per_point_mean = [m / (bs * num_times) for m, bs in zip(mean_us, batch_sizes)]
    per_point_std  = [s / (bs * num_times) for s, bs in zip(std_us,  batch_sizes)]
    ax1.errorbar(batch_sizes, per_point_mean, yerr=per_point_std, marker="o", markersize=6, capsize=4, linewidth=2)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=10)
    ax1.set_xlabel("Batch size"); ax1.set_ylabel("Time per sample point (μs)")
    ax1.set_title("Inference time per sample point")

    thr_pts = [(bs * num_times) / (m / 1e6) for bs, m in zip(batch_sizes, mean_us)]
    ax2.plot(batch_sizes, thr_pts, marker="s", markersize=6, linewidth=2)
    ax2.set_xscale("log", base=2); ax2.set_yscale("log", base=10)
    ax2.set_xlabel("Batch size"); ax2.set_ylabel("Throughput (points/s)")
    ax2.set_title("Inference throughput")

    bidx = int(np.argmax(thr_pts))
    ax2.scatter(batch_sizes[bidx], thr_pts[bidx], s=160, marker="*", color="red",
                zorder=5, label=f"Peak: {thr_pts[bidx]:.0f} points/s")
    ax2.legend()

    fig.suptitle(f"AE-DeepONet CPU Inference Benchmark (threads={threads}, {num_times} points/trajectory)")
    fig.tight_layout()
    out_png = PLOT_DIR / "benchmark_ae_deeponet_cpu.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")

def _print_model_summary(cfg: dict) -> None:
    mc, dc = cfg.get("model", {}), cfg.get("data", {})
    print("\nModel Configuration:")
    print(f"  Latent dimension: {mc.get('latent_dim', 'N/A')}")
    print(f"  Branch layers:    {mc.get('branch_layers', 'N/A')}")
    print(f"  Trunk layers:     {mc.get('trunk_layers', 'N/A')}")
    print(f"  Bypass AE:        {mc.get('bypass_autoencoder', False)}")
    print("\nData Configuration:")
    print(f"  Species:          {len(dc.get('species_variables', []))}")
    print(f"  Globals:          {len(dc.get('global_variables', []))}")
    print(f"  Time points/traj: {NUM_TIME_POINTS}")

def main():
    print("=" * 60)
    print("AE-DeepONet CPU INFERENCE BENCHMARK")
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
    print("\nRunning benchmarks (amortized timing)...")
    print("-" * 75)
    print(f"{'Batch':>8} | {'Mean (μs)':>12} | {'Std (μs)':>10} | {'μs/point':>12} | {'Points/s':>12}")
    print("-" * 75)

    for bs in BATCH_SIZES:
        try:
            mean_us, std_us = _bench_once(fn, batches[bs])
            means.append(mean_us); stds.append(std_us)
            us_per_point = mean_us / (bs * NUM_TIME_POINTS)
            thr_pts = (bs * NUM_TIME_POINTS) / (mean_us / 1e6)
            print(f"{bs:>8} | {mean_us:>12.2f} | {std_us:>10.2f} | {us_per_point:>12.3f} | {thr_pts:>10.0f} p/s")
        except Exception as e:
            print(f"{bs:>8} | Error: {e}")
            means.append(float('nan')); stds.append(float('nan'))

    print("-" * 75)

    valid = [(bs, m, s) for bs, m, s in zip(BATCH_SIZES, means, stds) if not np.isnan(m)]
    if valid:
        v_bs, v_m, v_s = zip(*valid)
        thr_pts = [(bs * NUM_TIME_POINTS) / (m / 1e6) for bs, m in zip(v_bs, v_m)]
        bidx = int(np.argmax(thr_pts))
        print(f"\nOptimal batch size: {v_bs[bidx]}")
        print(f"Peak throughput:    {thr_pts[bidx]:.0f} points/s")
        print(f"Time per point:     {v_m[bidx] / (v_bs[bidx] * NUM_TIME_POINTS):.3f} μs")

        S = len(cfg["data"]["species_variables"])
        G = len(cfg["data"]["global_variables"])
        bytes_per_sample = 4 * (S + G + NUM_TIME_POINTS * S)
        print(f"\nMemory/sample:      {bytes_per_sample / 1024:.1f} KB")
        print(f"Memory at peak BS:  {v_bs[bidx] * bytes_per_sample / (1024**2):.1f} MB")

        _plot(list(v_bs), list(v_m), list(v_s), CPU_THREADS, NUM_TIME_POINTS)

    gc.enable()
    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()
