#!/usr/bin/env python3
"""
Plot Flow-map DeepONet predictions vs ground truth using an **INT8-quantized ONNX** model.

- Assumes you have an INT8 K=1 export at:
    models/flowmap-deeponet/flowmap_deeponet_k1_int8.onnx
  (If a dynamic INT8 model exists, this script will also use it.)
- Trunk input is **normalized Δt** (dt-spec).
- We evaluate from the first physical time t0 of a chosen trajectory:
    Δt_k = t_phys[k] - t0  (clamped ≥ 0), then normalize via manifest dt-spec.

Usage: python predictions_onnx_int8.py
"""

# =========================
#          CONFIG
# =========================
import os, sys
from pathlib import Path

MODEL_STR     = "flowmap-deeponet"
REPO_ROOT     = Path(__file__).resolve().parent.parent
MODEL_DIR     = REPO_ROOT / "models" / MODEL_STR
CONFIG_PATH   = REPO_ROOT / "config" / "config.jsonc"
PROCESSED_DIR = REPO_ROOT / "data" / "processed-flowmap"

# Preferred INT8 models (first existing is used)
ONNX_PREFER = [
    MODEL_DIR / "completefsgdfgsdfg_dynamic_int8.onnx",  # dynamic K,B (if you quantized the dynamic model)
    MODEL_DIR / "flowmap_sdft8.onnx",     # static K=1 INT8 (most common in your runs)
]

# Optional fallbacks (FP32) if INT8 not found
ONNX_FALLBACK = [
    MODEL_DIR / "complete_model_dynamic.onnx",
    MODEL_DIR / "flowmap_deeponet_k1.onnx",
]

SAMPLE_INDEX = 7           # which trajectory to plot
OUTPUT_DIR   = None        # None -> <model_dir>/plots
SEED         = 42

# Query time selection on the PHYSICAL grid (normalization comes after)
Q_COUNT = 100              # None or 0 for full dense grid
Q_MODE  = "uniform"        # Options: uniform, random, log_uniform

# Plot styling
CONNECT_LINES = True
MARKER_EVERY  = 5

# =========================
#     ENV & IMPORTS
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
import onnxruntime as ort
import json5

# Add src to path
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass


# =========================
#     CORE FUNCTIONS
# =========================
def _pick_onnx_model() -> Path:
    for p in ONNX_PREFER + ONNX_FALLBACK:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"No ONNX model found. Tried: {ONNX_PREFER + ONNX_FALLBACK}")

def _is_k1_model(sess: ort.InferenceSession) -> bool:
    """
    Heuristic: if dt_norm input has shape [B,1] (rank 2 with second dim == 1),
    treat as static-K=1 model. If rank 1 (e.g., [K]) treat as dynamic-K model.
    """
    for i in sess.get_inputs():
        if i.name == "dt_norm":
            shp = i.shape
            # Examples: ['batch', 1] or [-1, 1] -> K=1; ['K'] or [-1] -> dynamic-K
            if isinstance(shp, (list, tuple)) and len(shp) == 2:
                try:
                    return int(shp[1]) == 1
                except Exception:
                    return True
            return len(shp) != 1
    # Default to K=1 if not sure (safer path)
    return True

def _load_ort_session(path: Path) -> tuple[ort.InferenceSession, bool]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, os.cpu_count() // 2 or 1)  # decent default on M-series
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern   = True
    so.enable_mem_reuse     = True
    opt_path = path.with_suffix("").with_name(path.stem + ".opt.onnx")
    so.optimized_model_filepath = str(opt_path)

    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess, _is_k1_model(sess)

def _load_single_test_sample(data_dir: Path, sample_idx: int | None):
    test_shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards in {data_dir}/test")

    with np.load(test_shards[0], allow_pickle=False) as d:
        x0   = d["x0"].astype(np.float32)          # [N,S]
        y    = d["y_mat"].astype(np.float32)       # [N,M,S]
        g    = d["globals"].astype(np.float32)     # [N,G]
        tvec = d["t_vec"]                          # [M] or [N,M]
    N = x0.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "y0":     y[sample_idx:sample_idx+1, 0, :],                # [1,S]
        "y_true": y[sample_idx],                                   # [M,S]
        "t_phys": tvec[sample_idx] if tvec.ndim == 2 else tvec,    # [M]
        "globals": g[sample_idx:sample_idx+1],                     # [1,G]
        "sample_idx": int(sample_idx),
    }

def _normalize_dt(norm: NormalizationHelper, dt_phys: np.ndarray | torch.Tensor, device: torch.device):
    if not isinstance(dt_phys, torch.Tensor):
        dt = torch.as_tensor(dt_phys, dtype=torch.float32, device=device)
    else:
        dt = dt_phys.to(device=device, dtype=torch.float32)
    dt = torch.clamp(dt, min=float(getattr(norm, "epsilon", 1e-25)))
    dt_norm = norm.normalize_dt_from_phys(dt.view(-1))  # [K] in [0,1]
    return dt_norm

def _select_query_indices(mode: str, count: int | None, t_phys: np.ndarray):
    M = int(t_phys.size)
    if not count or count >= M:
        return np.arange(M, dtype=int)

    rng = np.random.default_rng(SEED)
    if mode == "uniform":
        idx = np.linspace(0, M - 1, count).round().astype(int)
    elif mode == "random":
        idx = np.sort(rng.choice(M, size=count, replace=False))
    elif mode == "log_uniform":
        pos = t_phys[t_phys > 0]
        if pos.size == 0:
            idx = np.linspace(0, M - 1, count).round().astype(int)
        else:
            tmin, tmax = float(pos.min()), float(t_phys.max())
            grid = np.logspace(np.log10(tmin), np.log10(tmax), count)
            idx  = np.unique(np.array([np.abs(t_phys - g).argmin() for g in grid], dtype=int))
            if idx.size < count:
                rest = np.setdiff1d(np.arange(M), idx)
                idx  = np.sort(np.r_[idx, rest[:max(0, count - idx.size)]])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return idx

def _ort_predict_dynamicK(sess: ort.InferenceSession,
                          y0n: torch.Tensor, gn: torch.Tensor, dt_norm_sel: torch.Tensor) -> np.ndarray:
    """
    Dynamic-K model: inputs shapes are [B,S], [B,G], [K]; use B=1.
    Returns [K,S] numpy.
    """
    y0_np = y0n.detach().cpu().numpy().astype(np.float32)      # [1,S]
    g_np  = gn.detach().cpu().numpy().astype(np.float32)       # [1,G]
    dt_np = dt_norm_sel.detach().cpu().numpy().astype(np.float32)  # [K]

    feeds = {
        "y0_norm":      y0_np,
        "globals_norm": g_np,
        "dt_norm":      dt_np,
    }
    y = sess.run(None, feeds)[0]  # [1,K,S]
    return np.squeeze(y, axis=0)

def _ort_predict_k1(sess: ort.InferenceSession,
                    y0n: torch.Tensor, gn: torch.Tensor, dt_norm_sel: torch.Tensor) -> np.ndarray:
    """
    Static K=1 model: expects [B,S], [B,G], [B,1].
    We vectorize over K by setting B=K and tiling (y0, g).
    Returns [K,S] numpy.
    """
    K = int(dt_norm_sel.numel())
    y0_np_1 = y0n.detach().cpu().numpy().astype(np.float32)  # [1,S]
    g_np_1  = gn.detach().cpu().numpy().astype(np.float32)   # [1,G]

    y0_np = np.repeat(y0_np_1, K, axis=0)                    # [K,S]
    g_np  = np.repeat(g_np_1,  K, axis=0)                    # [K,G] or [K,0]
    dt_np = dt_norm_sel.detach().cpu().numpy().astype(np.float32).reshape(K, 1)  # [K,1]

    feeds = {
        "y0_norm":      y0_np,
        "globals_norm": g_np,
        "dt_norm":      dt_np,
    }
    y = sess.run(None, feeds)[0]  # [K,1,S]
    return np.squeeze(y, axis=1)  # [K,S]

def _predict_with_onnx(sess: ort.InferenceSession, is_k1: bool,
                       y0n: torch.Tensor, gn: torch.Tensor, dt_norm_sel: torch.Tensor,
                       norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    if is_k1:
        y_norm = _ort_predict_k1(sess, y0n, gn, dt_norm_sel)          # [K,S] normalized
    else:
        y_norm = _ort_predict_dynamicK(sess, y0n, gn, dt_norm_sel)    # [K,S] normalized
    # Denormalize to physical
    y_norm_t = torch.from_numpy(y_norm).to(dtype=torch.float32)
    y_phys_t = norm.denormalize(y_norm_t, species)  # [K,S] torch
    return y_phys_t.cpu().numpy()

def _compute_errors(y_pred: np.ndarray, y_true: np.ndarray, idx, species: list[str]):
    y_true_aligned = y_true[idx, :]
    rel = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)

    print("\n[ERROR] Mean relative error per species:")
    means = rel.mean(axis=0)
    for sp, e in zip(species, means):
        print(f"  {sp:15s}: {e:.3e}")
    print(f"\n[ERROR] Overall mean={rel.mean():.3e}, max={rel.max():.3e}")
    return rel

def _plot(t_phys, y_true, t_phys_sel, y_pred, sample_idx, species, q_mode, q_count,
          connect_lines, marker_every, out_path: Path):
    eps = 1e-30
    t_plot = np.clip(t_phys, eps, None)
    y_t    = np.clip(y_true, eps, None)
    y_p    = np.clip(y_pred, eps, None)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    for i in range(len(species)):
        ax.loglog(t_plot, y_t[:, i], '-',  color=colors[i], lw=2.0, alpha=0.9)
    if connect_lines and len(t_phys_sel) > 1:
        for i in range(len(species)):
            ax.loglog(t_phys_sel, y_p[:, i], '--', color=colors[i], lw=1.6, alpha=0.85)
    for i in range(len(species)):
        ax.loglog(t_phys_sel[::marker_every], y_p[::marker_every, i], 'o', color=colors[i], ms=5, alpha=0.9)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='black', lw=2.0, ls='-',  label='True (dense)'),
        Line2D([0],[0], color='black', lw=1.6, ls='--', label='Predicted (interp)'),
        Line2D([0],[0], color='black', marker='o', lw=0, ms=5, label='Predicted (queries)'),
    ], loc='lower left', fontsize=10)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Species Abundance")
    ax.set_title(f"Flow-map DeepONet vs Ground Truth (Sample {sample_idx}) — {q_mode} K={q_count}")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")


# =========================
#           MAIN
# =========================
def main():
    seed_everything(SEED)

    cfg        = json5.load(open(CONFIG_PATH, "r"))
    species    = cfg["data"]["species_variables"]
    globals_v  = cfg["data"]["global_variables"]

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (MODEL_DIR / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = _pick_onnx_model()
    print(f"[INFO] Using ONNX model: {onnx_path.name}")
    sess, is_k1 = _load_ort_session(onnx_path)
    print(f"[INFO] Detected model type: {'K=1 static' if is_k1 else 'Dynamic-K'}")

    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))

    data = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)

    # Normalize inputs (torch for NormalizationHelper)
    device = torch.device("cpu")
    y0 = torch.from_numpy(data["y0"]).to(device).contiguous()          # [1,S]
    g  = torch.from_numpy(data["globals"]).to(device).contiguous()      # [1,G]
    y0n = norm.normalize(y0, species)
    gn  = norm.normalize(g,  globals_v)

    # Dense physical time grid for this trajectory
    t_phys_dense = data["t_phys"].astype(np.float64)
    t0 = float(t_phys_dense[0])

    # Select query indices on PHYSICAL grid
    M   = int(t_phys_dense.size)
    idx = np.arange(M, dtype=int) if (not Q_COUNT or Q_COUNT >= M) else None
    if idx is None:
        rng = np.random.default_rng(SEED)
        if Q_MODE == "uniform":
            idx = np.linspace(0, M - 1, Q_COUNT).round().astype(int)
        elif Q_MODE == "random":
            idx = np.sort(rng.choice(M, size=Q_COUNT, replace=False))
        elif Q_MODE == "log_uniform":
            pos = t_phys_dense[t_phys_dense > 0]
            if pos.size == 0:
                idx = np.linspace(0, M - 1, Q_COUNT).round().astype(int)
            else:
                tmin, tmax = float(pos.min()), float(t_phys_dense.max())
                grid = np.logspace(np.log10(tmin), np.log10(tmax), Q_COUNT)
                idx  = np.unique(np.array([np.abs(t_phys_dense - g).argmin() for g in grid], dtype=int))
                if idx.size < Q_COUNT:
                    rest = np.setdiff1d(np.arange(M), idx)
                    idx  = np.sort(np.r_[idx, rest[:max(0, Q_COUNT - idx.size)]])
        else:
            raise ValueError(f"Unsupported Q_MODE: {Q_MODE}")

    t_phys_sel  = t_phys_dense[idx]
    dt_phys_sel = np.maximum(t_phys_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, device)  # torch [K]

    print(f"[INFO] Flow-map inference: Query mode={Q_MODE}, K={int(dt_norm_sel.numel())}")

    # Predict with ONNX (INT8 if chosen)
    y_pred = _predict_with_onnx(sess, is_k1, y0n, gn, dt_norm_sel, norm, species)  # [K,S] np

    # Plot against ground truth at the selected PHYSICAL times
    out_png = out_dir / f"predictions_onnxINT8_K{int(dt_norm_sel.numel())}_{Q_MODE}_sample_{data['sample_idx']}.png"
    _plot(t_phys_dense, data["y_true"], t_phys_sel, y_pred,
          data["sample_idx"], species, Q_MODE, int(dt_norm_sel.numel()),
          CONNECT_LINES, MARKER_EVERY, out_png)

    # Error metrics (align by selected indices)
    _ = _compute_errors(y_pred, data["y_true"], idx, species)

if __name__ == "__main__":
    main()
