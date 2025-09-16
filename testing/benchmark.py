#!/usr/bin/env python3
"""
CPU-only benchmark: Flow-map DeepONet inference vs batch size (K=1),
with optional torch.compile and automatic schema discovery (species/globals).

What changed vs your old script
--------------------------------
- **Auto-parses S and G** in this priority order:
  1) `MODEL_DIR/config.snapshot.json`
  2) `PROCESSED_DIR/normalization.json` (uses `meta.{species_variables, global_variables}`)
  3) `CFG_PATH` (json/jsonc)
- Accepts env overrides for `MODEL_DIR`, `PROCESSED_DIR`, `CFG_PATH`.
- Tries multiple export filenames: `complete_model_exported_k1.int8.pt2`, `complete_model_exported_k1.pt2`, `complete_model_exported.pt2`.
- Clean failure if schema can’t be resolved.

Usage
-----
$ python benchmark_cpu_k1_autoparse.py

Optional env vars:
  MODEL_DIR, PROCESSED_DIR, CFG_PATH, CPU_THREADS, USE_COMPILE=0/1
"""
from __future__ import annotations

import os, sys, time, gc, json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ---------- CPU env BEFORE importing torch ----------
CPU_THREADS = int(os.environ.get("CPU_THREADS", "1"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parent

import numpy as np
import torch
import matplotlib.pyplot as plt

# json5 is nice-to-have; fall back to std json
try:
    import json5  # type: ignore
    _json5_load = json5.load
except Exception:  # pragma: no cover
    _json5_load = json.load

# ---------- Paths & Config ----------
MODEL_STR   = os.environ.get("MODEL_STR", "flowmap-deeponet")
MODEL_DIR   = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / MODEL_STR)).resolve()
CFG_PATH    = Path(os.environ.get("CONFIG_PATH", REPO_ROOT / "config" / "config.jsonc")).resolve()
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed")).resolve()

EXPORT_CANDIDATES = [
    MODEL_DIR / "complete_model_exported_k1_int8.pt2",
    MODEL_DIR / "complete_model_exported_k1.pt2",
    MODEL_DIR / "complete_model_exported.pt2",
]

# One time jump per call (K=1)
NUM_TIME_POINTS = 1

# Batch sizes to scan (log2 up to 1024)
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Timings
WARMUP_CALLS           = 50
CALLS_PER_MEASUREMENT  = 2
REPEATS                = 5

# Use torch.compile on the loaded exported module (CPU / Inductor)
USE_COMPILE = bool(int(os.environ.get("USE_COMPILE", "1")))

PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------- small utils ----------------------------

def _set_torch_threads(n: int) -> None:
    torch.set_num_threads(int(n))
    # Inter-op parallelism rarely helps here
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def _safe_style():
    try:
        plt.style.use("science.mplstyle")
    except Exception:
        pass


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Try json5-compatible loader first
            try:
                return _json5_load(f)
            except Exception:
                f.seek(0)
                return json.load(f)
    except Exception:
        return None


# ------------------- schema discovery (S, G) -------------------

def _resolve_schema() -> tuple[list[str], list[str]]:
    """Return (species_variables, global_variables) using robust priority.

    Priority:
      1) MODEL_DIR/config.snapshot.json
      2) PROCESSED_DIR/normalization.json -> meta.*
      3) CFG_PATH (json/jsonc)
    """
    # 1) snapshot near model
    snap = MODEL_DIR / "config.snapshot.json"
    if snap.exists():
        cfg = _load_json(snap) or {}
        data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        sv = list(data.get("species_variables", []) or [])
        gv = list(data.get("global_variables", []) or [])
        if sv:
            print(f"[schema] From snapshot: S={len(sv)}, G={len(gv)} -> {snap}")
            return sv, gv

    # 2) normalization manifest in processed dir
    norm = PROCESSED_DIR / "normalization.json"
    if norm.exists():
        man = _load_json(norm) or {}
        meta = (man.get("meta", {}) if isinstance(man, dict) else {})
        sv = list(meta.get("species_variables", []) or [])
        gv = list(meta.get("global_variables", []) or [])
        if sv:
            print(f"[schema] From normalization.json: S={len(sv)}, G={len(gv)} -> {norm}")
            return sv, gv

    # 3) project config
    if CFG_PATH.exists():
        cfg = _load_json(CFG_PATH) or {}
        data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        sv = list(data.get("species_variables", []) or [])
        gv = list(data.get("global_variables", []) or [])
        if sv:
            print(f"[schema] From config: S={len(sv)}, G={len(gv)} -> {CFG_PATH}")
            return sv, gv

    raise KeyError(
        "Unable to determine species/global variables. Checked:\n"
        f"  1) {snap}\n  2) {norm}\n  3) {CFG_PATH}\n"
        "Ensure one of these contains data.species_variables."
    )


# ------------------- exported model loading -------------------

def _load_exported_model() -> Tuple[Any, torch.device, Any]:
    device = torch.device("cpu")
    _set_torch_threads(CPU_THREADS)

    mp = _first_existing(EXPORT_CANDIDATES)
    if mp is None:
        tried = "\n  ".join(str(p) for p in EXPORT_CANDIDATES)
        raise FileNotFoundError("No exported model found. Tried:\n  " + tried)

    from torch.export import load as tload
    prog = tload(str(mp))
    fn   = prog.module()

    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            fn = torch.compile(fn, backend="inductor", mode="max-autotune", fullgraph=True)
            print("[INFO] torch.compile enabled (inductor, CPU)")
        except Exception as e:  # pragma: no cover
            print(f"[WARN] torch.compile failed: {e}")

    return prog, device, fn


# ------------------- batch creation & bench -------------------

def _create_batch(bs: int, species: list[str], globals_v: list[str], device: torch.device, num_t: int = NUM_TIME_POINTS) -> Dict[str, torch.Tensor]:
    S = len(species)
    G = len(globals_v)
    if num_t != 1:
        raise ValueError("This benchmark is for K=1 only. Set NUM_TIME_POINTS = 1.")
    return {
        "y0_norm":      torch.randn(bs, S, device=device, dtype=torch.float32).contiguous(),
        "globals_norm": (torch.randn(bs, G, device=device, dtype=torch.float32).contiguous() if G > 0 else torch.zeros(bs, 0, device=device, dtype=torch.float32)),
        "dt_norm":      torch.tensor([0.5], device=device, dtype=torch.float32),  # center of [0,1]
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

    fig.tight_layout()
    out_png = PLOT_DIR / "benchmark_cpu_k1.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")


def _print_summary(species: list[str], globals_v: list[str]) -> None:
    print("\nModel/Data Schema:")
    print(f"  Species (S):      {len(species)}")
    print(f"  Globals (G):      {len(globals_v)}")
    print(f"  K (jumps/call):   1")


# ------------------- main -------------------

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

    # Resolve schema early (fixes your KeyError)
    species, globals_v = _resolve_schema()
    _print_summary(species, globals_v)

    # Load exported model
    try:
        _, dev, fn = _load_exported_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nPreallocating input batches...")
    batches = {bs: _create_batch(bs, species, globals_v, dev, NUM_TIME_POINTS) for bs in BATCH_SIZES}

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

        # Rough memory accounting per sample: y0[S] + g[G] + output[S]
        bytes_per_sample = 4 * (len(species) + len(globals_v) + 1 * len(species))
        print(f"\nMemory/sample:      {bytes_per_sample / 1024:.1f} KB")
        print(f"Memory at peak BS:  {v_bs[bidx] * bytes_per_sample / (1024**2):.1f} MB")

        _plot(list(v_bs), list(v_m), list(v_s), CPU_THREADS)

    gc.enable()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
