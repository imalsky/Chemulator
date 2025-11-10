#!/usr/bin/env python3
"""
Benchmark exported models across powers-of-two batch sizes for a fixed K.

- Finds artifacts in REPO/models/big (matches your export.py).
- Infers S_in and G from data/processed/normalization.json (robust).
- Prefers GPU AOTI; uses PT2 on CPU/MPS by default. Falls back automatically.
- Detects BK (dynamic B,K) vs K1 (dynamic B only) from filename.
- Throughput = (iters * B * K_eff) / elapsed. Saves plots/bench_throughput_k{K}.png.
"""

from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

# ====== GLOBALS ======
MODEL_SUBDIR = "models/4"     # exporter target dir
BENCH_K: int = 1                # used for BK exports; CPU K1 ignores

WARMUP_STEPS: int = 10
MEASURE_STEPS: int = 200

# Caps for power-of-two batches
MAX_POW2_BATCH_BY_TAG: Dict[str, Optional[int]] = {
    "CPU": 1024,
    "MPS": 4096,
    "GPU": 8192,
}

# Prefer AOTI only on GPU (Apple MPS AOTI is brittle with addmm/as_strided)
PREFER_AOTI: Dict[str, bool] = {"CPU": False, "GPU": True, "MPS": False}

# Match your export.py filenames
RAW_EXPORT = {
    "CPU": "export_k1_cpu.pt2",     # K1
    "GPU": "export_bk_gpu.pt2",     # BK
    "MPS": "export_bk_mps.pt2",     # BK
}
AOTI_EXPORT = {
    "GPU": "export_bk_gpu.aoti",    # BK
    "MPS": "export_bk_mps.aoti",    # BK
}

# MPS safety
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("AOTI_RUNTIME_CHECK_INPUTS", "1")

# ====== PATHS ======
ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / MODEL_SUBDIR
PROCESSED_DIR = ROOT / "data" / "processed"
NORM_PATH = PROCESSED_DIR / "normalization.json"

# ====== Helpers ======
def _device_ok(tag: str) -> bool:
    return True if tag == "CPU" else (torch.cuda.is_available() if tag == "GPU" else torch.backends.mps.is_available())

def _device_str(tag: str) -> str:
    return "cpu" if tag == "CPU" else ("cuda" if tag == "GPU" else "mps")

def _sync(dev: str):
    if dev.startswith("cuda"): torch.cuda.synchronize()
    elif dev == "mps" and hasattr(torch, "mps"): torch.mps.synchronize()

def _dtype_for(tag: str) -> torch.dtype:
    # Mirrors your export defaults
    return torch.bfloat16 if tag == "GPU" else torch.float32

def _p2_batches(max_cap: int, min_b: int = 1) -> List[int]:
    b = 1
    while b < min_b: b <<= 1
    out = []
    while b <= max_cap:
        out.append(b); b <<= 1
    return out or [min_b]

def _artifact_candidates(tag: str) -> List[Path]:
    prefer_aoti = PREFER_AOTI.get(tag, False)
    names = []
    if prefer_aoti and tag in AOTI_EXPORT: names.append(AOTI_EXPORT[tag])
    names.append(RAW_EXPORT[tag])
    if not prefer_aoti and tag in AOTI_EXPORT: names.append(AOTI_EXPORT[tag])  # fallback
    return [WORK_DIR / n for n in names if (WORK_DIR / n).exists()]

def _is_aoti(p: Path) -> bool:
    return p.suffix == "" and p.name.endswith(".aoti")

def _load_callable(p: Path):
    if _is_aoti(p):
        try:
            from torch._inductor import aot_load_package
        except Exception as e:
            raise RuntimeError(f"AOTI loader unavailable: {e}")
        return aot_load_package(str(p)), True
    from torch.export import load as torch_export_load
    ep = torch_export_load(str(p))
    return ep.module(), False

def _signature_from_name(p: Path) -> str:
    nm = p.name.lower()
    return "K1" if "k1" in nm else "BK"

def _dims_from_normalization(norm_path: Path) -> Tuple[int, int]:
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing {norm_path}")
    j = json.loads(norm_path.read_text())
    meta = j.get("meta", j)
    species = meta.get("species_variables") or j.get("species_variables") or []
    gvars = meta.get("global_variables") or j.get("global_variables") or []
    if not species:
        raise RuntimeError("species_variables missing in normalization.json")
    S_in = len(species)
    G = len(gvars or [])
    return S_in, G

def _make_inputs(sig: str, B: int, K: int, S_in: int, G: int, dtype: torch.dtype, device: str):
    """
    Inputs must match how the module was exported:

    - K1 export  -> y[B,S], dt[B,1],   g[B,G]
    - BK export  -> y[B,S], dt[B,K,1], g[B,G]   (y has no K dim; K lives in dt)

    (Your export.py uses the new model directly; no BK wrapper.)
    """
    y  = torch.randn(B, S_in, dtype=dtype, device=device).contiguous()                 # [B,S]
    g  = (torch.randn(B, G, dtype=dtype, device=device).contiguous()                   # [B,G]
          if G > 0 else torch.empty(B, 0, dtype=dtype, device=device))

    if sig == "K1":
        dt = torch.randn(B, 1, dtype=dtype, device=device).contiguous()                # [B,1]
    else:
        dt = torch.randn(B, K, 1, dtype=dtype, device=device).contiguous()             # [B,K,1]

    return (y, dt, g)


# ====== Main ======
def main():
    print("=" * 80)
    print(f"Benchmarking exported models in: {WORK_DIR}")
    print("=" * 80)
    if not WORK_DIR.exists():
        print(f"Model directory does not exist: {WORK_DIR}")
        return

    try:
        S_in, G = _dims_from_normalization(NORM_PATH)
    except Exception as e:
        print(f"Failed to read dims from {NORM_PATH}: {e}")
        return

    TAGS = ["CPU", "GPU", "MPS"]
    curves: List[Tuple[str, List[int], List[float]]] = []
    png_out = WORK_DIR / f"plots/bench_throughput_k{BENCH_K}.png"

    for tag in TAGS:
        if not _device_ok(tag):
            print(f"- {tag}: device not available, skipping")
            continue

        cands = _artifact_candidates(tag)
        if not cands:
            print(f"- {tag}: no artifacts found in {WORK_DIR}, skipping")
            continue

        dev = _device_str(tag)
        dtype = _dtype_for(tag)

        # Try preferred, then fallback
        model = None; is_aoti = False; sig = "BK"; art = None
        for candidate in cands:
            try:
                m, is_a = _load_callable(candidate)
                model, is_aoti, art = m, is_a, candidate
                sig = _signature_from_name(candidate)
                break
            except Exception as e:
                print(f"- {tag}: failed to load {candidate.name}: {e}")
        if model is None:
            print(f"- {tag}: cannot load any artifact, skipping")
            continue

        # Batch schedule
        max_cap = MAX_POW2_BATCH_BY_TAG.get(tag) or 16384
        batches = _p2_batches(max_cap=max_cap, min_b=1)

        print(f"- {tag}: using {art.name} ({'AOTI' if is_aoti else 'PT2'}), sig={sig}, device={dev}, dtype={dtype}, "
              f"S_in={S_in}, G={G}, K={BENCH_K}, B∈{batches}")

        # Validate at B=1; if AOTI explodes, fall back to PT2 if possible
        def run_once(B: int):
            inp = _make_inputs(sig, B, BENCH_K, S_in, G, dtype, dev)
            with torch.inference_mode():
                return model(*inp)

        try:
            run_once(1); _sync(dev)
        except Exception as e:
            if is_aoti:
                print(f"  AOTI warmup failed: {e}\n  -> Falling back to PT2 for {tag}")
                pt2 = WORK_DIR / RAW_EXPORT[tag]
                if not pt2.exists():
                    print("  PT2 export not available; skipping this tag")
                    continue
                try:
                    model, is_aoti = _load_callable(pt2)
                    sig = _signature_from_name(pt2)
                    run_once(1); _sync(dev)
                except Exception as e2:
                    print(f"  PT2 warmup also failed: {e2}; skipping this tag")
                    continue
            else:
                print(f"  Warmup failed: {e}; skipping this tag")
                continue

        # Measure
        thr: List[float] = []
        for B in batches:
            try:
                inp = _make_inputs(sig, B, BENCH_K, S_in, G, dtype, dev)
                for _ in range(WARMUP_STEPS):
                    with torch.inference_mode():
                        _ = model(*inp)
                _sync(dev)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(MEASURE_STEPS):
                        _ = model(*inp)
                _sync(dev)
                elapsed = time.perf_counter() - t0
                K_eff = (BENCH_K if sig == "BK" else 1)
                sps = (MEASURE_STEPS * B * K_eff) / max(elapsed, 1e-12)
                thr.append(sps)
                print(f"    B={B:>5d} → {sps:,.1f} samples/s")
            except Exception as e:
                print(f"    B={B:>5d} failed: {e}")
                break

        if thr:
            curves.append((tag, batches[:len(thr)], thr))

    if not curves:
        print("No curves to plot. Nothing was benchmarked.")
        return

    # Plot
    plt.style.use("science.mplstyle")
    plt.figure(figsize=(8, 5))
    for label, bs, th in curves:
        plt.plot(bs, th, marker="o", label=label)
    plt.xlabel("Batch size (B)")
    plt.ylabel("Throughput (samples/second)")
    plt.yscale("log"); plt.xscale("log")
    plt.legend()
    png_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(png_out), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {png_out}")

if __name__ == "__main__":
    main()
