# testing/advanced_benchmark.py
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Set to True to enable torch.compile (slower on Apple Silicon, faster on server CPUs)
USE_TORCH_COMPILE = False
MODEL_DIR = Path(__file__).parent.parent / 'data' / 'models' / "deeponet_20250722_142346"

# Benchmark settings
WARMUP_RUNS = 100
LATENCY_BENCHMARK_RUNS = 1000
THROUGHPUT_BENCHMARK_RUNS = 500
BATCH_SIZES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# ---------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_config(model_dir):
    """Loads the configuration file."""
    return json.loads((model_dir / 'config.json').read_text())

def load_exported_model(model_dir, use_compile, device):
    """Loads the exported model and optionally compiles it."""
    exported_path = model_dir / 'exported_model.pt'
    if not exported_path.exists():
        raise FileNotFoundError(f"This script requires 'exported_model.pt'. File not found in {model_dir}")

    print(f"Loading exported model from: {exported_path}")
    ep = torch.export.load(str(exported_path))
    model = ep.module().to(device)

    if use_compile:
        print("Applying torch.compile...")
        model = torch.compile(model)
    else:
        print("Skipping torch.compile for local eager-mode benchmark.")
    
    return model

def benchmark_latency(model, input_dim, device):
    """Measures single-core, single-item inference latency."""
    print("\n--- Part 1: Single-Core Latency Benchmark ---")
    print("Forcing 1 CPU thread to measure the fastest possible single-request latency.")
    
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    
    dummy_input = torch.randn(1, input_dim, device=device)
    timings_us = []

    with torch.no_grad():
        for _ in range(LATENCY_BENCHMARK_RUNS):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            timings_us.append((end_time - start_time) * 1_000_000) # Convert to microseconds
    
    # Restore original thread count
    torch.set_num_threads(original_threads)

    print(f"\nLatency for a single prediction (Batch Size 1, Single Core):")
    print(f"  Average: {np.mean(timings_us):.2f} µs")
    print(f"  Median:  {np.median(timings_us):.2f} µs")
    print(f"  Min:     {np.min(timings_us):.2f} µs")
    print(f"  Max:     {np.max(timings_us):.2f} µs")
    
def benchmark_throughput(model, input_dim, device):
    """Measures multi-core throughput across various batch sizes."""
    print("\n--- Part 2: Multi-Core Throughput & Latency Benchmark ---")
    print(f"Using all {torch.get_num_threads()} CPU threads to measure max throughput.")

    results = []
    print("\n" + "-"*80)
    print(f"{'Batch Size':<12} | {'Avg Latency/Prediction (µs)':<30} | {'Throughput (predictions/sec)':<30}")
    print("-" * 80)

    for batch_size in BATCH_SIZES_TO_TEST:
        dummy_input = torch.randn(batch_size, input_dim, device=device)
        timings_s = []

        with torch.no_grad():
            for _ in range(THROUGHPUT_BENCHMARK_RUNS):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                timings_s.append(end_time - start_time)
        
        total_time_s = sum(timings_s)
        avg_batch_time_s = total_time_s / THROUGHPUT_BENCHMARK_RUNS
        
        # Latency per prediction in this batched context
        latency_per_pred_us = (avg_batch_time_s / batch_size) * 1_000_000
        
        # Total predictions per second
        throughput = (batch_size * THROUGHPUT_BENCHMARK_RUNS) / total_time_s
        
        results.append(latency_per_pred_us)
        print(f"{batch_size:<12} | {latency_per_pred_us:<30.2f} | {throughput:,.0f}")
    
    print("-" * 80)
    return results

def plot_results(batch_sizes, latencies_us, save_path):
    """Plots latency vs. batch size with results in microseconds."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(batch_sizes, latencies_us, marker='o', linestyle='-', color='b')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size (log scale)')
    ax.set_ylabel('Latency per Prediction (µs, log scale)')
    ax.set_title(f'Model Inference Latency vs. Batch Size (torch.compile: {USE_TORCH_COMPILE})')
    
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    ax.grid(True, which="both", ls="--")
    
    for i, txt in enumerate(latencies_us):
        ax.annotate(f'{txt:.1f} µs', (batch_sizes[i], latencies_us[i]), textcoords="offset points", xytext=(5,-15), ha='left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nBenchmark plot saved to: {save_path}")
    plt.show()

def main():
    if not MODEL_DIR.exists():
        raise ValueError(f"Model directory not found: {MODEL_DIR}")
    
    # --- 1. System and Model Setup ---
    # Apply low-level optimizations first
    torch.set_flush_denormal(True)
    
    config = load_config(MODEL_DIR)
    device = torch.device('cpu')
    
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f"Using {num_threads} CPU threads for multi-core tests.")

    model = load_exported_model(MODEL_DIR, USE_TORCH_COMPILE, device)
    
    input_dim = len(config['data']['species_variables']) + len(config['data']['global_variables']) + 1

    # --- 2. Comprehensive Warmup ---
    print(f"\n--- Performing {WARMUP_RUNS} Warmup Runs ---")
    start_warmup = time.perf_counter()
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            dummy_input = torch.randn(BATCH_SIZES_TO_TEST[-1], input_dim, device=device)
            _ = model(dummy_input)
    print(f"Warmup complete in {time.perf_counter() - start_warmup:.2f} seconds.")

    # --- 3. Run Benchmarks ---
    benchmark_latency(model, input_dim, device)
    throughout_results = benchmark_throughput(model, input_dim, device)
    
    # --- 4. Plot Throughput Results ---
    plots_dir = MODEL_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f'advanced_benchmark_compile_{USE_TORCH_COMPILE}.png'
    plot_results(BATCH_SIZES_TO_TEST, throughout_results, save_path)


if __name__ == '__main__':
    main()