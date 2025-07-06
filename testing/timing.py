#!/usr/bin/env python3
"""
cpu_bench.py — Measure TorchScript inference on *many-core* CPUs.

Highlights
----------
• Works with frozen / unfrozen ScriptModules.
• Lets you pick:
   – `--threads`  → intra-op (MKL-DNN / OpenMP) threads per op
   – `--workers`  → Python threads that *concurrently* call the model  
     (good when per-batch maths under-utilises cores).
• Keeps ≤200 source lines.
"""

from __future__ import annotations
import argparse, json, time, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

plt.style.use('science.mplstyle')

# ─────────────────── model utils ───────────────────
def load_ts(path: Path) -> torch.jit.ScriptModule:
    m = torch.jit.load(path / "best_model_jit.pt", map_location="cpu").eval()
    try:
        m = torch.jit.freeze(m)
    except Exception:
        pass
    try:
        m = torch.jit.optimize_for_inference(m)
    except Exception as e:
        warnings.warn(f"optimize_for_inference skipped: {e}")
    try:
        m = torch.compile(m, backend="inductor", mode="default")
    except Exception:
        pass
    return m


# ────────────── benchmarking core ────────────────
def run_one(model, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x)


def bench(model, dim_in: int, batches: list[int], workers: int, iters=100, warm=10):
    res = {"bs": [], "lat_ms": [], "thr": []}
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    big = torch.randn(max(batches), dim_in)
    for bs in batches:
        xs = big[:bs]
        # identical views for workers
        splits = list(xs.chunk(workers)) if workers > 1 else [xs]
        for _ in range(warm):
            if pool:
                _ = list(pool.map(run_one, (model,)*workers, splits))
            else:
                _ = run_one(model, xs)
        t = []
        for _ in range(iters):
            s = time.perf_counter()
            if pool:
                _ = list(pool.map(run_one, (model,)*workers, splits))
            else:
                _ = run_one(model, xs)
            t.append(time.perf_counter() - s)
        med = np.median(t)
        res["bs"].append(bs)
        res["lat_ms"].append(med*1e3)
        res["thr"].append(bs/med)
        print(f"bs={bs:<5}  lat={med*1e3:7.2f} ms  thr={bs/med:8.0f} s/s")
    if pool:
        pool.shutdown()
    return res


# ─────────────── plotting util ────────────────
def plot(base_dir, r):
    f,(a,b)=plt.subplots(1,2,figsize=(11,4))
    a.loglog(r["bs"],r["lat_ms"],'o-', linewidth=2)
    a.set(xlabel="batch",ylabel="Miliseconds for batch")
    b.loglog(r["bs"],1e6/np.asarray(r["thr"]),'o-', linewidth=2)
    b.set(xlabel="batch",ylabel="µs/sample",title="throughput")
    
    plt.tight_layout()
    plt.savefig(base_dir/"figures/cpu_bench.png",dpi=250)


# ─────────────────── main ──────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained_model_siren_v3")
    ap.add_argument("--max-batch", type=int, default=2**12)
    ap.add_argument("--threads", type=int, default=4, help="intra-op threads per op")
    ap.add_argument("--workers", type=int, default=8, help="Python threads that call the model")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)  # leave inter-op to us
    
    base_dir = Path(__file__).parent.parent / "data" / args.model


    config_path = base_dir / "run_config.json"

    with open(config_path, "r") as f:
        cfg = json.load(f)

    d_in = len(cfg["data_specification"]["species_variables"])+len(cfg["data_specification"]["global_variables"])+1
    batches = [1]
    while batches[-1] < args.max_batch: batches.append(batches[-1]*2)
    batches[-1]=min(batches[-1],args.max_batch)

    model = load_ts(base_dir)
    res = bench(model, d_in, batches, args.workers)
    json.dump(res, open("cpu_bench.json","w"), indent=2)

    os.makedirs(base_dir/"figures", exist_ok=True)

    plot(base_dir, res)


if __name__ == "__main__":
    main()
