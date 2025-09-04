#!/usr/bin/env python3
"""
Plot Flow-map DeepONet predictions vs ground truth using a PT2-exported model.

- Trunk input is **normalized Δt** (dt-spec).
- We evaluate from the first physical time t0 of a chosen trajectory:
    Δt_k = t_phys[k] - t0  (clamped ≥ 0), then normalize via manifest dt-spec.
- Excludes predictions at t0 (where Δt=0) since the model wasn't trained on this case.

Usage: python predictions.py
"""

# =========================
#          CONFIG
# =========================
import os, sys
from pathlib import Path

MODEL_STR = "flowmap-deeponet"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / MODEL_STR
CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
PROCESSED_DIR = REPO_ROOT / "data" / "processed-flowmap"

EXPORT_PATHS = [
    MODEL_DIR / "complete_model_exported.pt2",
    MODEL_DIR / "final_model_exported.pt2",  # fallback
]

SAMPLE_INDEX = 3  # which trajectory to plot
OUTPUT_DIR = None  # None -> <model_dir>/plots
SEED = 42

# Query time selection on the PHYSICAL grid (normalization comes after)
Q_COUNT = 100  # None or 0 for full dense grid
Q_MODE = "uniform"  # Options: uniform, random, log_uniform

# Plot styling
CONNECT_LINES = True
MARKER_EVERY = 5

# =========================
#     ENV & IMPORTS
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)
import matplotlib.pyplot as plt
from torch.export import load as torch_load
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
def _find_exported_model():
    for p in EXPORT_PATHS:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError(f"No exported model found. Tried: {EXPORT_PATHS}")


def _load_exported_model(path: str):
    prog = torch_load(path)
    mod = prog.module()
    return mod, torch.device("cpu")


def _load_single_test_sample(data_dir: Path, sample_idx: int | None):
    test_shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards in {data_dir}/test")

    with np.load(test_shards[0], allow_pickle=False) as d:
        x0 = d["x0"].astype(np.float32)  # [N,S]
        y = d["y_mat"].astype(np.float32)  # [N,M,S]
        g = d["globals"].astype(np.float32)  # [N,G]
        tvec = d["t_vec"]  # [M] or [N,M]
    N = x0.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "y0": y[sample_idx:sample_idx + 1, 0, :],  # [1,S] (state at first time)
        "y_true": y[sample_idx],  # [M,S]
        "t_phys": tvec[sample_idx] if tvec.ndim == 2 else tvec,  # [M]
        "globals": g[sample_idx:sample_idx + 1],  # [1,G]
        "sample_idx": int(sample_idx),
    }


def _normalize_dt(norm: NormalizationHelper, dt_phys: np.ndarray | torch.Tensor, device: torch.device):
    """Flow-map: normalize Δt with the dt-spec (e.g., log-min-max)."""
    if not isinstance(dt_phys, torch.Tensor):
        dt = torch.as_tensor(dt_phys, dtype=torch.float32, device=device)
    else:
        dt = dt_phys.to(device=device, dtype=torch.float32)
    dt = torch.clamp(dt, min=float(getattr(norm, "epsilon", 1e-25)))
    dt_norm = norm.normalize_dt_from_phys(dt.view(-1))  # [K] in [0,1]
    return dt_norm


def _select_query_indices(mode: str, count: int | None, t_phys: np.ndarray, exclude_first: bool = True):
    """
    Select indices into the dense physical time grid for evaluation.

    Args:
        mode: Selection mode (uniform, random, log_uniform)
        count: Number of points to select
        t_phys: Physical time array
        exclude_first: If True, exclude index 0 (t0) from selection
    """
    M = int(t_phys.size)
    start_idx = 1 if exclude_first else 0  # Start from 1 to exclude t0

    if not count or count >= (M - start_idx):
        return np.arange(start_idx, M, dtype=int)

    rng = np.random.default_rng(SEED)

    if mode == "uniform":
        # Uniform spacing starting AFTER t0
        idx = np.linspace(start_idx, M - 1, count).round().astype(int)
    elif mode == "random":
        # Random selection excluding t0
        idx = np.sort(rng.choice(np.arange(start_idx, M), size=count, replace=False))
    elif mode == "log_uniform":
        # Log-uniform spacing naturally excluding t0
        pos = t_phys[start_idx:]  # Exclude t0
        if pos.size == 0 or np.any(pos <= 0):
            # Fallback to uniform if log scale not applicable
            idx = np.linspace(start_idx, M - 1, count).round().astype(int)
        else:
            tmin, tmax = float(pos.min()), float(pos.max())
            grid = np.logspace(np.log10(tmin), np.log10(tmax), count)
            # Find closest indices, offset by start_idx
            idx = np.unique(np.array([start_idx + np.abs(pos - g).argmin() for g in grid], dtype=int))
            if idx.size < count:  # pad if collisions
                available = np.setdiff1d(np.arange(start_idx, M), idx)
                idx = np.sort(np.r_[idx, available[:max(0, count - idx.size)]])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return idx


@torch.inference_mode()
def _predict(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_sel: torch.Tensor,
             norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    """Call exported module and denormalize species back to physical space."""
    y_norm = fn(y0_norm, g_norm, dt_norm_sel).squeeze(0)  # [K,S]
    return norm.denormalize(y_norm, species).cpu().numpy()


def _compute_errors(y_pred: np.ndarray, y_true: np.ndarray, idx, species: list[str]):
    """Compare predictions to ground truth at selected indices."""
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
    y_t = np.clip(y_true, eps, None)
    y_p = np.clip(y_pred, eps, None)
    t_sel = np.clip(t_phys_sel, eps, None)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    for i in range(len(species)):
        ax.loglog(t_plot, y_t[:, i], '-', color=colors[i], lw=2.0, alpha=0.9)
    if connect_lines and len(t_sel) > 1:
        for i in range(len(species)):
            ax.loglog(t_sel, y_p[:, i], '--', color=colors[i], lw=1.6, alpha=0.85)
    for i in range(len(species)):
        ax.loglog(t_sel[::marker_every], y_p[::marker_every, i], 'o', color=colors[i], ms=5, alpha=0.9)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='True (dense)'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Predicted (interp)'),
        Line2D([0], [0], color='black', marker='o', lw=0, ms=5, label='Predicted (queries)'),
    ], loc='lower left', fontsize=10)

    ax.set_ylim(1e-5, 2)
    ax.set_xlim(1e-3, 1e8)

    ax.set_xlabel("Time (s)");
    ax.set_ylabel("Species Abundance")
    ax.set_title(f"Flow-map DeepONet vs Ground Truth (Sample {sample_idx}) — {q_mode} K={q_count}")
    plt.tight_layout();
    plt.savefig(out_path, dpi=150, bbox_inches="tight");
    plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")


# =========================
#           MAIN
# =========================
def main():
    seed_everything(SEED)

    cfg = json5.load(open(CONFIG_PATH, "r"))
    species = cfg["data"]["species_variables"]
    globals_v = cfg["data"]["global_variables"]

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (MODEL_DIR / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_exported_model()
    fn, model_dev = _load_exported_model(model_path)

    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))

    data = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)

    # Normalize inputs
    y0 = torch.from_numpy(data["y0"]).to(model_dev).contiguous()  # [1,S]
    g = torch.from_numpy(data["globals"]).to(model_dev).contiguous()  # [1,G]
    y0n = norm.normalize(y0, species)
    gn = norm.normalize(g, globals_v)

    # Dense physical time grid for this trajectory
    t_phys_dense = data["t_phys"].astype(np.float64)
    t0 = float(t_phys_dense[0])

    # Select query indices on the PHYSICAL time grid, EXCLUDING t0
    M = int(t_phys_dense.size)

    # Adjust Q_COUNT if it would include all points
    effective_q_count = Q_COUNT if (Q_COUNT and Q_COUNT < M - 1) else M - 1

    # Select indices, excluding the first one (t0)
    idx = _select_query_indices(Q_MODE, effective_q_count, t_phys_dense, exclude_first=True)

    # Ensure we never include index 0
    idx = idx[idx > 0] if len(idx) > 0 else np.array([1], dtype=int)

    t_phys_sel = t_phys_dense[idx]
    dt_phys_sel = np.maximum(t_phys_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, model_dev)

    print(f"[INFO] Flow-map inference: Query mode={Q_MODE}, K={int(dt_norm_sel.numel())}")
    print(f"[INFO] Excluding t0={t0:.3e}s from predictions (Δt=0 not in training regime)")
    print(f"[INFO] First prediction at t={t_phys_sel[0]:.3e}s (Δt={dt_phys_sel[0]:.3e}s)")

    # Predict (fn expects normalized Δt)
    y_pred = _predict(fn, y0n, gn, dt_norm_sel, norm, species)

    # Plot against ground truth at the selected PHYSICAL times
    out_png = out_dir / f"predictions_K{int(dt_norm_sel.numel())}_{Q_MODE}_sample_{data['sample_idx']}.png"
    _plot(t_phys_dense, data["y_true"], t_phys_sel, y_pred,
          data["sample_idx"], species, Q_MODE, int(dt_norm_sel.numel()),
          CONNECT_LINES, MARKER_EVERY, out_png)

    # Error metrics (align by selected indices)
    _ = _compute_errors(y_pred, data["y_true"], idx, species)


if __name__ == "__main__":
    main()