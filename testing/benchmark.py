#!/usr/bin/env python3
"""CPU-only benchmark: AE-DeepONet model inference time vs batch size."""

from __future__ import annotations
import os
import sys
import time
import gc
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# CPU threading & environment (set before importing torch)
CPU_THREADS = min(os.cpu_count() or 4, 6)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("ACCELERATE_MATMUL_MULTITHREADING", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib.pyplot as plt

# Configuration
MODEL_STR = "deepo"
MODEL_DIR = Path(f"../models/{MODEL_STR}")
MODEL_FILE = "complete_model_exported.pt2"
CONFIG_FILE = "config.json"

# Batch sizes to test
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Number of time points to predict (trunk dimension)
NUM_TIME_POINTS = 100

# Warmup/timing parameters
WARMUP_CALLS = 20
CALLS_PER_MEASUREMENT = 16
REPEATS = 3

# Plotting
PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def set_torch_threads(threads: int) -> None:
    """Configure torch threading for CPU inference."""
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(max(1, int(threads // 2)))


def safe_style() -> None:
    """Try to use science style if available."""
    try:
        plt.style.use("science.mplstyle")
    except Exception:
        pass


def load_exported_model() -> Tuple[Any, Dict[str, Any], torch.device, Any]:
    """Load exported model, config, device, and cache the callable."""
    cfg_path = MODEL_DIR / CONFIG_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = json.load(f)

    device = torch.device("cpu")
    set_torch_threads(CPU_THREADS)

    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading exported model: {model_path}")
    from torch.export import load as tload
    exported_prog = tload(str(model_path))

    # Materialize the callable once
    fn = exported_prog.module()
    return exported_prog, config, device, fn


def create_batch(
        bs: int,
        config: dict,
        device: torch.device,
        num_times: int = NUM_TIME_POINTS
) -> Dict[str, torch.Tensor]:
    """Create a batch of inputs for the AE-DeepONet model."""
    num_species = len(config["data"]["species_variables"])
    num_globals = len(config["data"]["global_variables"])

    # Create normalized inputs
    x0_norm = torch.randn(bs, num_species, device=device, dtype=torch.float32).contiguous()
    globals_norm = torch.randn(bs, num_globals, device=device, dtype=torch.float32).contiguous()
    trunk_times = torch.linspace(0.0, 1.0, steps=num_times, device=device, dtype=torch.float32)

    return {
        "x0_norm": x0_norm,
        "globals_norm": globals_norm,
        "trunk_times": trunk_times
    }


@torch.inference_mode()
def run_forwards(fn, batch: Dict[str, torch.Tensor], calls: int) -> None:
    """Run multiple forward passes."""
    for _ in range(calls):
        _ = fn(batch["x0_norm"], batch["globals_norm"], batch["trunk_times"])


@torch.inference_mode()
def benchmark_bs(fn, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """Benchmark a single batch size, return mean and std in microseconds."""
    # Warmup
    run_forwards(fn, batch, WARMUP_CALLS)

    # Timed measurements
    times_us = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        run_forwards(fn, batch, CALLS_PER_MEASUREMENT)
        dt_us = (time.perf_counter_ns() - t0) / 1e3  # Convert ns to us
        times_us.append(dt_us / CALLS_PER_MEASUREMENT)

    arr = np.asarray(times_us, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def plot_results(
        batch_sizes: List[int],
        mean_us: List[float],
        std_us: List[float],
        threads: int,
        num_times: int
) -> None:
    """Generate benchmark plots."""
    safe_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Time per sample point (microseconds)
    ax = ax1
    per_point_mean = [m / (bs * num_times) for m, bs in zip(mean_us, batch_sizes)]
    per_point_std = [s / (bs * num_times) for s, bs in zip(std_us, batch_sizes)]
    ax.errorbar(batch_sizes, per_point_mean, yerr=per_point_std,
                marker="o", markersize=6, capsize=4, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    #ax.set_ylim(0.1, 100)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per sample point (μs)")
    ax.set_title("Inference time per sample point")

    # 2. Throughput (points per second)
    ax = ax2
    throughput_points = [(bs * num_times) / (m / 1e6) for bs, m in zip(batch_sizes, mean_us)]
    ax.plot(batch_sizes, throughput_points, marker="s", markersize=6, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (points/s)")
    ax.set_title("Inference throughput (sample points per second)")

    best_idx = int(np.argmax(throughput_points))
    ax.scatter(batch_sizes[best_idx], throughput_points[best_idx],
               s=160, marker="*", color="red", zorder=5,
               label=f"Peak: {throughput_points[best_idx]:.0f} points/s")
    ax.legend()

    fig.suptitle(f"AE-DeepONet CPU Inference Benchmark (threads={threads}, {num_times} points/trajectory)")
    fig.tight_layout()

    out_png = PLOT_DIR / "benchmark_ae_deeponet_cpu.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")


def print_model_summary(config: dict) -> None:
    """Print model configuration summary."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    print(f"\nModel Configuration:")
    print(f"  Latent dimension: {model_cfg.get('latent_dim', 'N/A')}")
    print(f"  Branch layers: {model_cfg.get('branch_layers', 'N/A')}")
    print(f"  Trunk layers: {model_cfg.get('trunk_layers', 'N/A')}")
    print(f"  Bypass autoencoder: {model_cfg.get('bypass_autoencoder', False)}")
    print(f"\nData Configuration:")
    print(f"  Species variables: {len(data_cfg.get('species_variables', []))}")
    print(f"  Global variables: {len(data_cfg.get('global_variables', []))}")
    print(f"  Time points per trajectory: {NUM_TIME_POINTS}")


def main():
    print("=" * 60)
    print("AE-DeepONet CPU INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"CPU threads (intra-op): {CPU_THREADS}")
    print(f"Warmup calls: {WARMUP_CALLS}")
    print(f"Calls/measurement: {CALLS_PER_MEASUREMENT}")
    print(f"Repeats: {REPEATS}")

    # Reduce noise
    gc.disable()
    torch.set_grad_enabled(False)

    # Load model
    try:
        exported_prog, config, device, fn = load_exported_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Print model summary
    print_model_summary(config)

    # Preallocate batches
    print("\nPreallocating input batches...")
    batches: Dict[int, Dict[str, torch.Tensor]] = {
        bs: create_batch(bs, config, device, NUM_TIME_POINTS)
        for bs in BATCH_SIZES
    }

    # Run benchmarks
    means, stds = [], []
    print("\nRunning benchmarks (amortized timing)...")
    print("-" * 75)
    print(f"{'Batch':>8} | {'Mean (μs)':>12} | {'Std (μs)':>10} | "
          f"{'μs/point':>12} | {'Points/s':>12}")
    print("-" * 75)

    for bs in BATCH_SIZES:
        try:
            mean_us, std_us = benchmark_bs(fn, batches[bs])
            means.append(mean_us)
            stds.append(std_us)

            us_per_point = mean_us / (bs * NUM_TIME_POINTS)
            throughput_points = (bs * NUM_TIME_POINTS) / (mean_us / 1e6)

            print(f"{bs:>8} | {mean_us:>12.2f} | {std_us:>10.2f} | "
                  f"{us_per_point:>12.3f} | {throughput_points:>10.0f} p/s")
        except Exception as e:
            print(f"{bs:>8} | Error: {e}")
            means.append(float('nan'))
            stds.append(float('nan'))

    print("-" * 75)

    # Summary statistics
    if means and not all(np.isnan(means)):
        valid_means = [m for m in means if not np.isnan(m)]
        valid_bs = [bs for bs, m in zip(BATCH_SIZES, means) if not np.isnan(m)]

        if valid_means:
            throughput_points = [(bs * NUM_TIME_POINTS) / (m / 1e6)
                                 for bs, m in zip(valid_bs, valid_means)]
            best_idx = int(np.argmax(throughput_points))

            print(f"\nOptimal batch size: {valid_bs[best_idx]}")
            print(f"Peak throughput: {throughput_points[best_idx]:.0f} points/s")
            print(f"Time per point at peak: "
                  f"{valid_means[best_idx] / (valid_bs[best_idx] * NUM_TIME_POINTS):.3f} μs")

            # Memory estimate
            num_species = len(config["data"]["species_variables"])
            num_globals = len(config["data"]["global_variables"])
            bytes_per_sample = 4 * (num_species + num_globals + NUM_TIME_POINTS * num_species)

            print(f"\nMemory estimate per sample: {bytes_per_sample / 1024:.1f} KB")
            print(f"Memory at peak batch size: "
                  f"{valid_bs[best_idx] * bytes_per_sample / (1024 ** 2):.1f} MB")

            plot_results(valid_bs, valid_means,
                         [stds[i] for i, bs in enumerate(BATCH_SIZES) if bs in valid_bs],
                         CPU_THREADS, NUM_TIME_POINTS)

    gc.enable()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()