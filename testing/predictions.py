#!/usr/bin/env python3
"""
Plot Flow-map DeepONet predictions vs ground truth using a PT2-exported model (K=1).

- Trunk input is **normalized Δt** (dt-spec).
- We evaluate from the first physical time t0 of a chosen trajectory:
    Δt_k = t_phys[k] - t0  (clamped ≥ 0), then normalize via manifest dt-spec.
- Excludes predictions at t0 (where Δt=0) since the model wasn't trained on this case.
- Uses a K=1-exported model and calls it once per query time.
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
    MODEL_DIR / "complete_model_exported_k1.pt2",
]

SAMPLE_INDEX = 51  # which trajectory to plot
OUTPUT_DIR = None  # None -> <model_dir>/plots
SEED = 42

# Query time selection on the PHYSICAL grid (normalization comes after)
Q_COUNT = 100  # None or 0 = use all times after t0

# Plot styling
CONNECT_LINES = True
MARKER_EVERY = 2000

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


def _select_query_indices(count: int | None, t_phys: np.ndarray, exclude_first: bool = True):
    """
    Uniformly select indices into the physical time grid.

    Args:
        count: Number of points to select (None/0 = all after t0)
        t_phys: Physical time array
        exclude_first: If True, exclude index 0 (t0) from selection
    """
    M = int(t_phys.size)
    start_idx = 1 if exclude_first else 0  # exclude t0 by default
    if M - start_idx <= 0:
        raise ValueError("Time grid must contain at least two points (t0 and a later time).")

    if not count or count >= (M - start_idx):
        return np.arange(start_idx, M, dtype=int)

    # Uniform spacing over [start_idx, M-1]
    return np.linspace(start_idx, M - 1, count).round().astype(int)


@torch.inference_mode()
def _predict_one(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_scalar: torch.Tensor,
                 norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    """
    Call exported K=1 module once for a single Δt and return [S] in physical space.
    Accepts model outputs shaped [S], [1,S], [1,1,S], etc., and collapses them to [S].
    """
    # Ensure dt is 1-D shape [1] to satisfy the export constraint
    if not isinstance(dt_norm_scalar, torch.Tensor):
        dt1 = torch.tensor([float(dt_norm_scalar)], dtype=torch.float32, device=y0_norm.device)  # [1]
    else:
        dt1 = dt_norm_scalar.to(device=y0_norm.device, dtype=torch.float32).reshape(-1)  # [1]
    if dt1.numel() != 1:
        raise ValueError(f"dt_norm_scalar must be a scalar or length-1 tensor, got shape {tuple(dt1.shape)}")

    # Run the exported graph
    out = fn(y0_norm, g_norm, dt1)  # shapes seen in practice: [1,S] or [1,1,S]

    # Coerce to tensor and ensure last dim = S
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=y0_norm.device)
    S = y0_norm.shape[-1]
    if out.shape[-1] != S:
        raise RuntimeError(f"Exported model output last dim != S: got {out.shape}, expected last={S}")

    # Collapse ALL leading dims to 1 row, then squeeze to [S]
    out_2d = out.reshape(-1, S)  # e.g., [1,S] from [1,1,S] or [1,S]
    if out_2d.shape[0] != 1:
        out_2d = out_2d[:1, :]  # defensive; K=1 export should yield one row
    y_phys = norm.denormalize(out_2d, species)  # [1,S]
    return y_phys.squeeze(0).detach().cpu().numpy()  # -> [S]


@torch.inference_mode()
def _predict_many(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_vec: torch.Tensor,
                  norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    """
    Loop over K query times with a K=1-exported model.
    Ensures each per-call dt has shape [1].
    Returns: [K,S]
    """
    preds = []
    K = int(dt_norm_vec.numel())
    for k in range(K):
        dt1 = dt_norm_vec[k:k + 1]  # 1-D slice -> shape [1]
        preds.append(_predict_one(fn, y0_norm, g_norm, dt1, norm, species))
    return np.stack(preds, axis=0)  # [K,S]


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


def _plot(t_phys, y_true, t_phys_sel, y_pred, sample_idx, species,
          connect_lines, marker_every, out_path: Path):
    """Plot with correct color-species matching and legend sorted by max concentration."""
    eps = 1e-30
    t_plot = np.clip(t_phys, eps, None)
    y_t = np.clip(y_true, eps, None)
    y_p = np.clip(y_pred, eps, None)
    t_sel = np.clip(t_phys_sel, eps, None)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Generate colors for each species (in original order)
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    # Calculate maximum values for each species and sort
    max_values = np.max(y_t, axis=0)  # Maximum concentration for each species
    sorted_indices = np.argsort(max_values)[::-1]  # Sort descending (largest first)

    # First plot all lines using original indices to maintain correct color assignment
    for i in range(len(species)):
        # Plot true values
        ax.loglog(t_plot, y_t[:, i], '-', color=colors[i], lw=2.0, alpha=0.9)

        # Plot predicted values connected
        if connect_lines and len(t_sel) > 1:
            ax.loglog(t_sel, y_p[:, i], '--', color=colors[i], lw=1.6, alpha=0.85)

        # Plot predicted markers
        ax.loglog(t_sel[::marker_every], y_p[::marker_every, i], 'o',
                  color=colors[i], ms=5, alpha=0.9, mfc='none')

    # Now create legend handles in sorted order
    legend_handles = []
    legend_labels = []

    for idx in sorted_indices:
        # Create a line handle with the correct color for this species
        from matplotlib.lines import Line2D
        line_handle = Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9)
        legend_handles.append(line_handle)
        legend_labels.append(species[idx])

    # Set axis limits and labels
    ax.set_xlim(1e-3, 1e8)
    ax.set_ylim(1e-3, 2)  # Show full range
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Species Abundance", fontsize=12)

    # NO GRID as requested
    ax.grid(False)

    # Create first legend for species - sorted by max concentration with correct colors
    n_cols = 1
    legend1 = ax.legend(handles=legend_handles,
                        labels=legend_labels,
                        loc='center left',
                        bbox_to_anchor=(1.01, 0.6),
                        title='Species',
                        fontsize=10,
                        title_fontsize=11,
                        ncol=n_cols,
                        borderaxespad=0)

    # Add the first legend manually to the axes
    ax.add_artist(legend1)

    # Create second legend for line styles
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Model Prediction'),
        #Line2D([0], [0], color='black', marker='o', lw=0, ms=5, mfc='none', label='Query Points'),
    ]

    # Position second legend below the species legend
    legend2 = ax.legend(handles=style_handles,
                        loc='center left',
                        bbox_to_anchor=(1.01, 0.2),
                        fontsize=10,
                        title_fontsize=11,
                        borderaxespad=0)

    # Save figure
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    effective_q_count = Q_COUNT if (Q_COUNT and Q_COUNT < M - 1) else M - 1
    idx = _select_query_indices(effective_q_count, t_phys_dense, exclude_first=True)  # uniform
    idx = idx[idx > 0] if len(idx) > 0 else np.array([1], dtype=int)  # defensive

    t_phys_sel = t_phys_dense[idx]
    dt_phys_sel = np.maximum(t_phys_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, model_dev)  # [K]

    print(f"[INFO] Flow-map inference: K={int(dt_norm_sel.numel())}")
    print(f"[INFO] Excluding t0={t0:.3e}s from predictions (Δt=0 not in training regime)")
    print(f"[INFO] First prediction at t={t_phys_sel[0]:.3e}s (Δt={dt_phys_sel[0]:.3e}s)")

    # Predict with a K=1 model by looping over dt_norm_sel
    y_pred = _predict_many(fn, y0n, gn, dt_norm_sel, norm, species)  # [K,S]

    # Plot against ground truth at the selected PHYSICAL times
    out_png = out_dir / f"predictions_K{int(dt_norm_sel.numel())}_sample_bigy_{data['sample_idx']}.png"
    _plot(t_phys_dense, data["y_true"], t_phys_sel, y_pred,
          data["sample_idx"], species, CONNECT_LINES, MARKER_EVERY, out_png)

    # Error metrics (align by selected indices)
    _ = _compute_errors(y_pred, data["y_true"], idx, species)


if __name__ == "__main__":
    main()