#!/usr/bin/env python3
"""
Simplified Flow-map AE prediction vs ground truth (no simplex ops).
- Solid lines  = ground truth from dataset
- Dashed lines = model predictions
- Select species via PLOT_SPECIES (match by base name; *_evolution suffix ignored)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('science.mplstyle')

# ---------------- Paths & settings ----------------
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/delta"
EP_FILENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper

SAMPLE_IDX = 1
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# Choose species by base names; empty list = plot all
PLOT_SPECIES: List[str] = ['H2', 'H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2']


# ---------------- Utils ----------------
def load_data(data_dir: Path, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load first test shard and extract one sample trajectory."""
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        y = d["y_mat"][sample_idx].astype(np.float32)     # [T, S]
        g = d["globals"][sample_idx].astype(np.float32)   # [G]
        t_phys = d["t_vec"]
        if t_phys.ndim > 1:
            t_phys = t_phys[sample_idx]
        t_phys = t_phys.astype(np.float32)                # [T]
    return y, g, t_phys


def prepare_batch(y0: np.ndarray,
                  g: np.ndarray,
                  t_phys: np.ndarray,
                  q_count: int,
                  norm: NormalizationHelper,
                  species: List[str],
                  globals_: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Prepare normalized (y, dt, g) for K=1 export."""
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)
    t_sel = t_phys[q_idx]                                 # absolute seconds
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species).float()      # [1,S]
    if globals_:
        g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_).float()   # [1,G]
    else:
        g_norm = torch.from_numpy(g[None, :]).float()

    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec)).view(-1, 1).float()  # [K,1]

    # Batch repeat anchor for each query Δt
    K = dt_norm.shape[0]
    y_batch = y0_norm.repeat(K, 1)  # [K,S]
    g_batch = g_norm.repeat(K, 1)   # [K,G]
    return y_batch, dt_norm, g_batch, q_idx, t_sel


@torch.inference_mode()
def run_inference(model, y_batch: torch.Tensor, dt_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    """Call exported program: expects output [K, 1, S]. Returns [K, S] (normalized)."""
    out = model(y_batch, dt_batch, g_batch)
    if out.dim() != 3 or out.size(1) != 1:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)} (expected [K,1,S])")
    return out[:, 0, :]


def plot_results(t_phys: np.ndarray,
                 y_true: np.ndarray,
                 t_pred: np.ndarray,
                 y_pred: np.ndarray,
                 species_all: List[str],
                 plot_species: List[str],
                 out_path: Path) -> None:
    """Log–log plot: solid=truth, dashed=prediction; only selected species."""
    # Map names to base names (strip *_evolution)
    base_all = [n[:-10] if n.endswith("_evolution") else n for n in species_all]
    if plot_species:
        keep = [i for i, b in enumerate(base_all) if b in plot_species]
    else:
        keep = list(range(len(base_all)))

    if not keep:
        raise RuntimeError("No requested species found in manifest.")

    # Slice arrays
    y_true = y_true[:, keep]
    y_pred = y_pred[:, keep]
    labels = [base_all[i] for i in keep]

    # Clip to numeric floor just for plotting
    tiny = 1e-35
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pr = (t_pred >= XMIN) & (t_pred <= XMAX)

    t_gt_plot = t_phys[m_gt]
    y_gt_plot = np.clip(y_true[m_gt], tiny, None)
    t_pr_plot = t_pred[m_pr]
    y_pr_plot = np.clip(y_pred[m_pr], tiny, None)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(keep)))

    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.95, color=col)
        if t_gt_plot.size:
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=col, ms=5)
        if t_pr_plot.size:
            ax.loglog(t_pr_plot, y_pr_plot[:, i], '--', lw=1.5, alpha=0.9, color=col)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")

    # Legends
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1]
    species_handles = [Line2D([0], [0], color=colors[i], lw=2.0) for i in order]
    species_labels = [labels[i] for i in order]
    leg1 = ax.legend(handles=species_handles, labels=species_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2), fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    os.chdir(REPO)
    seed_everything(42)

    # Load config + manifest
    try:
        cfg = load_json(MODEL_DIR / "config.json")
    except FileNotFoundError:
        cfg = load_json(MODEL_DIR / "trial_config.final.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    manifest = load_json(data_dir / "normalization.json")

    meta = manifest.get("meta", {})
    species = list(meta.get("species_variables", []))
    globals_ = list(meta.get("global_variables", []))
    if not species:
        raise RuntimeError("No species_variables in normalization.json")

    # Load exported program
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    # Normalizer
    norm = NormalizationHelper(manifest)

    # Load sample
    y, g, t_phys = load_data(data_dir, SAMPLE_IDX)  # y:[T,S], g:[G], t:[T]
    y0 = y[0]                                        # anchor at t0

    # Prepare batch (normalized)
    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(y0, g, t_phys, Q_COUNT, norm, species, globals_)

    # Predict, then denormalize
    y_pred_norm = run_inference(model, y_batch, dt_batch, g_batch)     # [K,S]
    y_pred = norm.denormalize(y_pred_norm, species).cpu().numpy()      # [K,S]
    y_true_sel = y[q_idx, :len(species)]                               # [K,S]

    # Sanity logs
    print(f"Pred sum (min,max): {y_pred.sum(1).min():.4e}, {y_pred.sum(1).max():.4e}")
    print(f"NaN in pred? {np.isnan(y_pred).any()}")

    # Plot solid vs dashed for selected species
    out_png = MODEL_DIR / "plots" / f"pred_vs_gt_{SAMPLE_IDX}.png"
    plot_results(t_phys, y[:, :len(species)], t_sel, y_pred, species, PLOT_SPECIES, out_png)

    # Simple error metric on sampled points
    rel_err = np.abs(y_pred - y_true_sel) / (np.abs(y_true_sel) + 1e-12)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()
