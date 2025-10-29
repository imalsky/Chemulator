#!/usr/bin/env python3
"""
Simplified Flow-map AE prediction and plotting.
Loads exported K=1 model, runs inference, and plots predictions vs ground truth.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paths
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/2"
EP_FILENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

# Settings
SAMPLE_IDX = 3
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8


def load_data(data_dir: Path, sample_idx: int):
    """Load first test shard and extract sample trajectory."""
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        y = d["y_mat"][sample_idx].astype(np.float32)
        g = d["globals"][sample_idx].astype(np.float32)
        t_phys = d["t_vec"]
        if t_phys.ndim > 1:
            t_phys = t_phys[sample_idx]

    return y, g, t_phys


def prepare_batch(y0, g, t_phys, q_count, norm, species, globals_):
    """Prepare normalized batch inputs for inference."""
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)
    t_sel = t_phys[q_idx].astype(np.float32)
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    # Normalize
    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species).float()
    g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_).float() if globals_ else torch.from_numpy(
        g[None, :]).float()
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec)).view(-1, 1).float()

    # Batch
    K = dt_norm.shape[0]
    y_batch = y0_norm.repeat(K, 1)
    g_batch = g_norm.repeat(K, 1)

    return y_batch, dt_norm, g_batch, q_idx, t_sel


def run_inference(model, y_batch, dt_batch, g_batch):
    """Run model inference and extract predictions."""
    with torch.inference_mode():
        out = model(y_batch, dt_batch, g_batch)

    if out.dim() != 3 or out.size(1) != 1:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")

    return out[:, 0, :]


def plot_results(t_phys, y_true, t_pred, y_pred, species, output_path):
    """Create log-log plot of predictions vs ground truth."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))
    tiny = 1e-35

    # Filter by x-range
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pred = (t_pred >= XMIN) & (t_pred <= XMAX)

    t_gt_plot = t_phys[m_gt]
    y_gt_plot = np.clip(y_true[m_gt], tiny, None)
    t_pred_plot = t_pred[m_pred]
    y_pred_plot = np.clip(y_pred[m_pred], tiny, None)

    # Plot all species
    for i, (species_name, color) in enumerate(zip(species, colors)):
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.9, color=color)
        if t_gt_plot.size:
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=color, ms=5)
        if t_pred_plot.size:
            ax.loglog(t_pred_plot, y_pred_plot[:, i], '--', lw=1.4, alpha=0.85, color=color)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.grid(False)

    # Legends
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1]
    species_handles = [Line2D([0], [0], color=colors[i], lw=2.0) for i in order]
    species_labels = [species[i] for i in order]
    leg1 = ax.legend(handles=species_handles, labels=species_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10, ncol=1)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2), fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {output_path}")


def main():
    os.chdir(REPO)
    seed_everything(42)

    # Load configuration
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

    # Load model
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    # Initialize normalizer
    norm = NormalizationHelper(manifest)

    # Load data
    y, g, t_phys = load_data(data_dir, SAMPLE_IDX)

    # Prepare batch
    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(
        y[0], g, t_phys, Q_COUNT, norm, species, globals_
    )

    # Run inference
    y_pred_norm = run_inference(model, y_batch, dt_batch, g_batch)

    # Denormalize
    y_pred = norm.denormalize(y_pred_norm, species).cpu().numpy()
    y_true_sel = y[q_idx, :len(species)]

    # Sanity checks
    print(f"Prediction sum: min={y_pred.sum(1).min():.4e}, max={y_pred.sum(1).max():.4e}")
    print(f"Contains NaN: {np.isnan(y_pred).any()}")

    # Plot
    plot_results(t_phys, y[:, :len(species)], t_sel, y_pred, species,
                 MODEL_DIR / "plots" / f"pred_vs_gt_{SAMPLE_IDX}.png")

    # Error metrics
    rel_err = np.abs(y_pred - y_true_sel) / (np.abs(y_true_sel) + 1e-10)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()