#!/usr/bin/env python3
"""
ONNX Runtime CPU-only optimization + multi-thread benchmark for Flow-map DeepONet (K=1).

- Loads flowmap_deeponet_k1.onnx (B dynamic, K=1 static)
- CPUExecutionProvider only
- Graph opt = ORT_ENABLE_ALL; saves optimized model
- Tunable intra_op threads; inter_op=1
- Correct IO binding via OrtValue to reuse output buffers
- Optional dynamic INT8 quantization (per-tensor) with ORT
- Reports µs/sample and throughput for a grid of (B, threads)
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# -------- Paths --------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parent

MODEL_STR = "flowmap-deeponet"
MODEL_DIR = REPO_ROOT / "models" / MODEL_STR
ONNX_K1   = MODEL_DIR / "flowmap_deeponet_k1.onnx"
CONFIG    = REPO_ROOT / "config" / "config.jsonc"

# -------- Benchmark settings --------
BATCH_GRID = [128, 256, 512, 1024]   # adjust as needed
INTRA_GRID = None                    # None -> derive from CPU count; or set like [4, 8, 12]
WARMUP     = 20
RUNS       = 100
DO_QUANT   = True                    # dynamic INT8 pass (optional)

# Add src/ for config loader
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json_config

def _cpu_count() -> int:
    try:
        import multiprocessing as mp
        return mp.cpu_count()
    except Exception:
        return os.cpu_count() or 1

def _choose_intra_grid() -> List[int]:
    if INTRA_GRID:
        return [int(x) for x in INTRA_GRID]
    cores = _cpu_count()
    # representative sweep
    grid = sorted(set([
        max(1, cores // 4),
        max(1, cores // 2),
        cores
    ]))
    return grid

def build_session_cpu(model_path: Path,
                      intra_threads: int,
                      inter_threads: int = 1,
                      optimized_out: Optional[Path] = None,
                      enable_mem_arena: bool = True,
                      enable_mem_pattern: bool = True,
                      parallel_exec: bool = False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, int(intra_threads))
    so.inter_op_num_threads = max(1, int(inter_threads))
    so.enable_cpu_mem_arena = bool(enable_mem_arena)
    so.enable_mem_pattern   = bool(enable_mem_pattern)
    so.enable_mem_reuse     = True
    if optimized_out is not None:
        so.optimized_model_filepath = str(optimized_out)
    so.execution_mode = (ort.ExecutionMode.ORT_PARALLEL
                         if parallel_exec else ort.ExecutionMode.ORT_SEQUENTIAL)
    providers = ["CPUExecutionProvider"]  # CPU only
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    return sess

def dynamic_quantize(model_path: Path) -> Optional[Path]:
    if not DO_QUANT:
        return None
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qpath = model_path.with_name(model_path.stem + "_int8.onnx")
        # Be compatible with various ORT versions: call with minimal args
        try:
            quantize_dynamic(str(model_path), str(qpath), weight_type=QuantType.QInt8)
        except TypeError:
            # Some older builds use positional args only
            quantize_dynamic(str(model_path), str(qpath))
        print(f"[OK] Wrote dynamic-quantized model: {qpath.name}")
        return qpath
    except Exception as e:
        print(f"[WARN] Dynamic quantization skipped: {e}")
        return None

def load_dims(cfg_path: Path) -> Tuple[int, int]:
    cfg = load_json_config(str(cfg_path))
    data = cfg["data"]
    species_vars = data.get("target_species_variables", data["species_variables"])
    S = len(species_vars)
    G = len(data["global_variables"])
    return S, G

def run_once_with_iobinding(sess, y0: np.ndarray, g: np.ndarray, dt: np.ndarray,
                            out_buf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Correct IO binding for CPU: pre-bind output using OrtValue (if provided) to reuse memory.
    Otherwise, bind output to CPU and copy once.
    """
    import onnxruntime as ort
    io = sess.io_binding()

    io.bind_cpu_input("y0_norm",      y0)
    io.bind_cpu_input("globals_norm", g)
    io.bind_cpu_input("dt_norm",      dt)

    out_name = sess.get_outputs()[0].name
    if out_buf is not None:
        ov = ort.OrtValue.ortvalue_from_numpy(out_buf, "cpu", 0)
        io.bind_ortvalue_output(out_name, ov)
        sess.run_with_iobinding(io)
        return out_buf
    else:
        io.bind_output(out_name, "cpu")
        sess.run_with_iobinding(io)
        return io.copy_outputs()[0]  # one copy

def bench_model(model_path: Path, S: int, G: int, intra_list: List[int]):
    # Save an optimized graph once (faster to load in production)
    opt_path = model_path.with_name(model_path.stem + ".opt.onnx")
    sess0 = build_session_cpu(model_path, intra_threads=intra_list[-1], optimized_out=opt_path)
    sess0 = None  # release

    # Optional INT8 variant
    q_path = dynamic_quantize(model_path)
    variants = [("fp32", model_path)]
    if q_path is not None:
        variants.append(("int8", q_path))

    for tag, path in variants:
        print(f"\n[{tag}] Using model: {path.name}")
        for intra in intra_list:
            sess = build_session_cpu(path, intra_threads=intra, inter_threads=1)
            print(f"  threads={intra:>2d} | providers={sess.get_providers()}")

            for B in BATCH_GRID:
                # Inputs for K=1 model
                y0 = np.random.randn(B, S).astype(np.float32)
                g  = (np.random.randn(B, G).astype(np.float32)
                      if G > 0 else np.zeros((B, 0), dtype=np.float32))
                dt = np.random.randn(B, 1).astype(np.float32)

                out_buf = np.empty((B, 1, S), dtype=np.float32)

                # Warm-up
                for _ in range(WARMUP):
                    _ = run_once_with_iobinding(sess, y0, g, dt, out_buf)

                # Timed runs
                t = []
                for _ in range(RUNS):
                    t0 = time.perf_counter()
                    _ = run_once_with_iobinding(sess, y0, g, dt, out_buf)
                    t.append(time.perf_counter() - t0)
                t = np.asarray(t)

                mean_batch = float(t.mean())
                us_per_sample = (mean_batch / B) * 1e6
                thr = B / mean_batch

                print(f"    B={B:<5d}  mean={us_per_sample:7.2f} µs/sample   "
                      f"throughput={thr:,.0f} samples/s   "
                      f"min={(t.min()/B)*1e6:7.2f} µs   med={(np.median(t)/B)*1e6:7.2f} µs")

def main():
    if not ONNX_K1.exists():
        raise FileNotFoundError(f"Missing K=1 ONNX: {ONNX_K1}")

    S, G = load_dims(CONFIG)
    print(f"[INFO] Dims: S={S}, G={G}")

    # CPU thread hints (reduce latency jitter on some libomp builds)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")

    intra_list = _choose_intra_grid()
    bench_model(ONNX_K1, S, G, intra_list)

if __name__ == "__main__":
    main()
