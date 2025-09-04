#!/usr/bin/env python3
"""
flowmap_ort_cpu.py — ONNX Runtime CPU runner for Flow-map DeepONet (K=1)

Features
- CPUExecutionProvider only
- Fixed, tuned threading (defaults: intra=6, inter=1)
- Single session reused across calls
- Correct IO binding with reusable OrtValues for zero-copy in/out
- Works with FP32 or INT8 (dynamic-quantized) K=1 model
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

# ---------------- Config ----------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "flowmap-deeponet"
MODEL_FP32 = MODEL_DIR / "flowmap_deeponet_k1.onnx"
MODEL_INT8 = MODEL_DIR / "flowmap_deeponet_k1_int8.onnx"  # optional

# Tuned for your results on M-series
INTRA_THREADS = 6
INTER_THREADS = 1

# Optional: reduce spin/jitter on some libomp builds
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")


class FlowmapOrtRunner:
    def __init__(self,
                 onnx_path: Path,
                 intra_threads: int = INTRA_THREADS,
                 inter_threads: int = INTER_THREADS):
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(self.onnx_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, int(intra_threads))
        so.inter_op_num_threads = max(1, int(inter_threads))
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern   = True
        so.enable_mem_reuse     = True
        # Save optimized graph alongside the model (faster reloads next time)
        opt_path = self.onnx_path.with_suffix("").with_name(self.onnx_path.stem + ".opt.onnx")
        so.optimized_model_filepath = str(opt_path)

        self.sess = ort.InferenceSession(str(self.onnx_path),
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        outs = self.sess.get_outputs()
        assert len(outs) == 1, "Expected single output"
        self.out_name = outs[0].name

        # IO binding + reusable OrtValues
        self.binding = self.sess.io_binding()
        self.ov_y0: Optional[ort.OrtValue] = None
        self.ov_g:  Optional[ort.OrtValue] = None
        self.ov_dt: Optional[ort.OrtValue] = None
        self.ov_out: Optional[ort.OrtValue] = None
        self.out_buf: Optional[np.ndarray] = None
        self.last_shapes: Tuple[int, int, int] = (-1, -1, -1)

        # Warmup with a tiny batch to trigger prepacking/fusions once
        self._warmup()

    def _ensure_buffers(self, B: int, S: int, G: int):
        """(Re)allocate OrtValues if shape changed."""
        if self.last_shapes == (B, S, G) and self.ov_out is not None:
            return
        self.last_shapes = (B, S, G)

        # Allocate new host buffers (float32)
        # Inputs are user-provided; we just create OrtValues that will wrap them at bind time.
        self.ov_y0 = None
        self.ov_g  = None
        self.ov_dt = None

        # Output buffer (B,1,S), reused across calls
        self.out_buf = np.empty((B, 1, S), dtype=np.float32)
        self.ov_out  = ort.OrtValue.ortvalue_from_numpy(self.out_buf, "cpu", 0)

    def _bind_inputs(self, y0: np.ndarray, g: np.ndarray, dt: np.ndarray):
        """Bind inputs/outputs for one run. Inputs can be re-bound each call without realloc."""
        b = self.binding
        b.clear_binding_inputs()
        b.clear_binding_outputs()

        # Use OrtValue wrapping to ensure zero-copy
        ov_y0 = ort.OrtValue.ortvalue_from_numpy(y0, "cpu", 0)
        ov_g  = ort.OrtValue.ortvalue_from_numpy(g,  "cpu", 0)
        ov_dt = ort.OrtValue.ortvalue_from_numpy(dt, "cpu", 0)

        b.bind_ortvalue_input("y0_norm", ov_y0)
        b.bind_ortvalue_input("globals_norm", ov_g)
        b.bind_ortvalue_input("dt_norm", ov_dt)
        b.bind_ortvalue_output(self.out_name, self.ov_out)

    def _warmup(self):
        # Attempt a minimal warmup if dimensions are known; otherwise no-op.
        try:
            # Try to infer S and G from model metadata; fall back to a small guess
            # Most models have input shapes with -1 for B; second dim is S/G; K=1 for dt.
            S = int(self.sess.get_inputs()[0].shape[1])
            G = int(self.sess.get_inputs()[1].shape[1])
            B = 32
        except Exception:
            B, S, G = 32, 16, 0

        y0 = np.zeros((B, S), dtype=np.float32)
        g  = np.zeros((B, G), dtype=np.float32)
        dt = np.zeros((B, 1), dtype=np.float32)

        self._ensure_buffers(B, S, G)
        self._bind_inputs(y0, g, dt)
        self.sess.run_with_iobinding(self.binding)

    def predict_k1(self, y0_norm: np.ndarray, globals_norm: np.ndarray, dt_norm: np.ndarray) -> np.ndarray:
        """
        Inference for K=1 model.
        y0_norm:      [B, S] float32
        globals_norm: [B, G] float32 (use shape [B,0] if no globals)
        dt_norm:      [B, 1] float32
        Returns:      [B, 1, S] float32 (view into internal buffer; copy if you need to keep it)
        """
        if y0_norm.dtype != np.float32 or globals_norm.dtype != np.float32 or dt_norm.dtype != np.float32:
            raise TypeError("All inputs must be float32 numpy arrays.")
        if y0_norm.ndim != 2 or globals_norm.ndim != 2 or dt_norm.ndim != 2 or dt_norm.shape[1] != 1:
            raise ValueError("Shapes must be [B,S], [B,G], [B,1].")

        B, S = y0_norm.shape
        G = globals_norm.shape[1]
        if globals_norm.shape[0] != B or dt_norm.shape[0] != B:
            raise ValueError("Batch sizes must match across inputs.")

        self._ensure_buffers(B, S, G)
        self._bind_inputs(y0_norm, globals_norm, dt_norm)
        self.sess.run_with_iobinding(self.binding)
        return self.out_buf  # (B,1,S)

# ---------------- Example usage ----------------
if __name__ == "__main__":
    import time

    # Choose INT8 if available
    onnx_path = MODEL_INT8 if MODEL_INT8.exists() else MODEL_FP32
    runner = FlowmapOrtRunner(onnx_path, intra_threads=INTRA_THREADS, inter_threads=INTER_THREADS)

    # Demo batch
    B, S, G = 512, 12, 2
    y0 = np.random.randn(B, S).astype(np.float32)
    g  = np.random.randn(B, G).astype(np.float32)
    dt = np.random.randn(B, 1).astype(np.float32)

    # Warm measurement
    n = 100
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        y = runner.predict_k1(y0, g, dt)
        t.append(time.perf_counter() - t0)
    t = np.asarray(t)
    print(f"mean: {(t.mean()/B)*1e6:6.2f} µs/sample | "
          f"median: {(np.median(t)/B)*1e6:6.2f} µs | "
          f"min: {(t.min()/B)*1e6:6.2f} µs | "
          f"throughput: {B/t.mean():,.0f} samples/s")
