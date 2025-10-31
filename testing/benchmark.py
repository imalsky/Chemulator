#!/usr/bin/env python3
"""
Benchmark exported models across powers-of-two batch sizes for a fixed K.

- Uses SAME path logic as export.py:
    ROOT = Path(__file__).resolve().parents[1]
    WORK_DIR = ROOT / MODEL_SUBDIR
- Prefers AOTI on CPU/GPU, but uses RAW .pt2 on MPS by default (due to MPS AOTI addmm/as_strided bug).
- Reads .meta.json to get dims (S_in, G) and dynamic ranges for batch/K.
- Measures throughput = (iters * B * K) / elapsed.
- Saves plot bench_throughput_k{K}.png in WORK_DIR.
- On runtime errors with AOTI, automatically falls back to RAW export for that tag.

No argparse. Tweak GLOBALS below.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

# ==============================================================================
# GLOBALS
# ==============================================================================

MODEL_SUBDIR = "models/big"     # must match export config
BENCH_K: int = 1                # must be within [min_k, max_k] of the export
WARMUP_STEPS: int = 10
MEASURE_STEPS: int = 200

# Global cap for batch sizes (None = use export max). You can also set per-tag below.
MAX_POW2_BATCH_DEFAULT: Optional[int] = None
# Safer cap for MPS to avoid AOTI as_strided failures at huge B*K
MAX_POW2_BATCH_BY_TAG: Dict[str, Optional[int]] = {"CPU": 1024,
                                                   "MPS": 4096,
                                                   "GPU": 8
                                                   }

# Prefer AOTI on CPU/GPU; prefer RAW on MPS.
PREFER_AOTI: Dict[str, bool] = {"CPU": True, "GPU": True, "MPS": False}

# Environment for MPS convenience / guard messages
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("AOTI_RUNTIME_CHECK_INPUTS", "1")  # better guard errors instead of OOB

# Filenames matching export.py
RAW_EXPORT = {
    "CPU": "export_k_dyn_cpu.pt2",
    "GPU": "export_k_dyn_gpu.pt2",
    "MPS": "export_k_dyn_mps.pt2",
}
AOTI_EXPORT = {
    "CPU": "export_k_dyn_cpu.aoti.pt2",
    "GPU": "export_k_dyn_gpu.aoti.pt2",
    "MPS": "export_k_dyn_mps.aoti.pt2",
}

# ==============================================================================
# PATHS (same scheme as exporter)
# ==============================================================================

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / MODEL_SUBDIR

# ==============================================================================
# Helpers
# ==============================================================================

def _device_ok(tag: str) -> bool:
    if tag == "CPU":
        return True
    if tag == "GPU":
        return torch.cuda.is_available()
    if tag == "MPS":
        return torch.backends.mps.is_available()
    return True

def _device_str(tag: str) -> str:
    return "cpu" if tag == "CPU" else ("cuda" if tag == "GPU" else "mps")

def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def _parse_dtype(s: str) -> torch.dtype:
    m = {
        "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return m.get(str(s).lower(), torch.float32)

def _read_meta(path: Path) -> Dict:
    meta = path.parent / f"{path.name}.meta.json"
    if not meta.exists():
        raise FileNotFoundError(f"Missing metadata: {meta}")
    with open(meta, "r") as f:
        return json.load(f)

def _choose_pow2_batches(min_b: int, max_b: int, cap: Optional[int]) -> List[int]:
    if cap is not None:
        max_b = min(max_b, cap)
    batches = []
    b = 1
    while b < min_b:
        b <<= 1
    while b <= max_b:
        batches.append(b)
        b <<= 1
    if not batches:
        # Fallback: nearest pow2 >= min_b (or clamp to max_b)
        b = 1
        while b < min_b:
            b <<= 1
        batches = [b] if b <= max_b else [max_b]
    return batches

def _make_inputs(B: int, K: int, S_in: int, G: int, dtype: torch.dtype, device: str):
    state = torch.randn(B, K, S_in, dtype=dtype, device=device).contiguous()
    dt = torch.randn(B, K, 1, dtype=dtype, device=device).contiguous()
    g = (torch.randn(B, K, G, dtype=dtype, device=device).contiguous()
         if G > 0 else torch.empty(B, K, 0, dtype=dtype, device=device))
    return state, dt, g

def _try_load_aoti(tag: str, base_dir: Path):
    name = AOTI_EXPORT.get(tag)
    if not name:
        return None
    p = base_dir / name
    if not p.exists():
        return None
    try:
        return torch._inductor.aoti_load_package(str(p))
    except Exception as e:
        print(f"  failed to load AOTI ({p.name}): {e}")
        return None

def _try_load_raw(tag: str, base_dir: Path):
    name = RAW_EXPORT.get(tag)
    if not name:
        return None
    p = base_dir / name
    if not p.exists():
        return None
    try:
        ep = torch.export.load(str(p))
        return ep.module()
    except Exception as e:
        print(f"  failed to load raw export ({p.name}): {e}")
        return None

def _load_model(tag: str, base_dir: Path):
    """
    Returns: (callable_model, is_aoti: bool, artifact_path: Path)
    Respects PREFER_AOTI[tag], but falls back to whichever exists.
    """
    prefer_aoti = PREFER_AOTI.get(tag, True)
    aoti_model = _try_load_aoti(tag, base_dir) if prefer_aoti else None
    raw_model  = _try_load_raw(tag, base_dir) if not prefer_aoti else None

    if prefer_aoti:
        if aoti_model is not None:
            return aoti_model, True, base_dir / AOTI_EXPORT[tag]
        raw_model = _try_load_raw(tag, base_dir)
        if raw_model is not None:
            print("  using RAW export (AOTI unavailable or failed to load)")
            return raw_model, False, base_dir / RAW_EXPORT[tag]
    else:
        if raw_model is not None:
            return raw_model, False, base_dir / RAW_EXPORT[tag]
        aoti_model = _try_load_aoti(tag, base_dir)
        if aoti_model is not None:
            print("  using AOTI export (RAW unavailable)")
            return aoti_model, True, base_dir / AOTI_EXPORT[tag]

    return None, False, None

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print(f"Benchmarking exported models in: {WORK_DIR}")
    print("=" * 80)
    if not WORK_DIR.exists():
        print(f"Model directory does not exist: {WORK_DIR}")
        return

    TAGS = ["CPU", "GPU", "MPS"]
    curves: List[Tuple[str, List[int], List[float]]] = []
    png_out = WORK_DIR / f"plots/bench_throughput_k{BENCH_K}.png"

    for tag in TAGS:
        if not _device_ok(tag):
            print(f"- {tag}: device not available, skipping")
            continue

        # Decide which artifact name we will read meta from (prefer chosen type)
        prefer_aoti = PREFER_AOTI.get(tag, True)
        candidate_name = (AOTI_EXPORT.get(tag) if prefer_aoti else RAW_EXPORT.get(tag)) or ""
        other_name = (RAW_EXPORT.get(tag) if prefer_aoti else AOTI_EXPORT.get(tag)) or ""

        chosen_file = WORK_DIR / candidate_name
        if not chosen_file.exists():
            chosen_file = WORK_DIR / other_name

        if not chosen_file.exists():
            print(f"- {tag}: missing both AOTI and RAW artifacts; skipping")
            continue

        # Read metadata
        try:
            meta = _read_meta(chosen_file)
        except Exception as e:
            print(f"- {tag}: failed to read meta: {e}; skipping")
            continue

        device = _device_str(tag)
        dtype = _parse_dtype(meta.get("dtype", "float32"))
        info = meta.get("model_info", {})
        exp_cfg = meta.get("export_config", {})

        try:
            S_in = int(info["input_dim"])
            G = int(info.get("global_dim", 0))
        except KeyError as e:
            print(f"- {tag}: missing model_info key {e}; skipping")
            continue

        mb = int(exp_cfg.get("min_batch_size", 1))
        xb = int(exp_cfg.get("max_batch_size", 64))
        mk = int(exp_cfg.get("min_k", 1))
        xk = int(exp_cfg.get("max_k", max(1, BENCH_K)))

        if not (mk <= BENCH_K <= xk):
            print(f"- {tag}: BENCH_K={BENCH_K} not in export range [{mk}, {xk}], skipping")
            continue

        cap = MAX_POW2_BATCH_BY_TAG.get(tag, MAX_POW2_BATCH_DEFAULT)
        batches = _choose_pow2_batches(mb, xb, cap=cap)
        if not batches:
            print(f"- {tag}: no valid batch sizes within [{mb}, {min(xb, cap or xb)}], skipping")
            continue

        # Load model callable
        model, is_aoti, art_path = _load_model(tag, WORK_DIR)
        if model is None:
            print(f"- {tag}: failed to load model, skipping")
            continue

        print(f"- {tag}: device={device}, dtype={dtype}, S_in={S_in}, G={G}, "
              f"K={BENCH_K}, B∈{batches}  ({'AOTI' if is_aoti else 'RAW'})")

        thr: List[float] = []
        for B in batches:
            state, dt, g = _make_inputs(B, BENCH_K, S_in, G, dtype, device)

            # Warmup with automatic fallback if AOTI explodes
            def run_once():
                with torch.inference_mode():
                    return model(state, dt, g)

            try:
                _ = run_once()
            except Exception as e:
                if is_aoti:
                    print(f"  AOTI failure at B={B}: {e}\n  -> Falling back to RAW export for {tag}")
                    # Reload RAW and retry
                    raw_model = _try_load_raw(tag, WORK_DIR)
                    if raw_model is None:
                        print("  RAW export not available or failed to load; skipping this tag")
                        thr = []
                        break
                    model, is_aoti = raw_model, False
                    try:
                        _ = run_once()
                    except Exception as e2:
                        print(f"  RAW export also failed at B={B}: {e2}; skipping this tag")
                        thr = []
                        break
                else:
                    print(f"  Runtime failure at B={B}: {e}; skipping this tag")
                    thr = []
                    break

            _sync(device)

            # Measure
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(MEASURE_STEPS):
                    _ = model(state, dt, g)
            _sync(device)
            elapsed = time.perf_counter() - t0

            samples = MEASURE_STEPS * B * BENCH_K
            sps = samples / elapsed
            thr.append(sps)
            print(f"    B={B:>4d} -> {sps:,.1f} samples/s")

        if thr:
            curves.append((tag, batches[:len(thr)], thr))

    if not curves:
        print("No curves to plot. Nothing was benchmarked.")
        return

    # Plot
    plt.style.use("science.mplstyle")
    plt.figure(figsize=(8, 5))
    colors=['red', 'blue']; i=0
    for label, bs, th in curves:
        plt.plot(bs, th, marker="o", label=label, color=colors[i])
        i = i + 1
    plt.xlabel("Batch size (B)")
    plt.ylabel("Throughput (samples/second)")
    #plt.grid(True, which="both", axis="both")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    try:
        png_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(png_out), dpi=150, bbox_inches="tight")
        print(f"\nSaved plot: {png_out}")
    except Exception as e:
        print(f"\nFailed to save plot {png_out}: {e}")

if __name__ == "__main__":
    main()
