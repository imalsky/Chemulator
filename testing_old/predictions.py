#!/usr/bin/env python3
"""Minimal Flow-map DeepONet prediction + plotting."""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paths
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/v1"
DATA_DIR = REPO / "data/processed_medium"

# Add src to path
sys.path.insert(0, str(REPO / "src"))
from normalizer import NormalizationHelper
from utils import load_json, seed_everything
from model import create_model

# Settings
SAMPLE_IDX = 0
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# Optional plot style
try:
    plt.style.use("science.mplstyle")
except:
    pass


def main():
    seed_everything(42)

    # Load model checkpoint
    ckpt = torch.load(MODEL_DIR / "export_eager.pt", map_location="cpu")
    cfg = ckpt["config"]

    # Get species and globals from normalization file
    norm_data = load_json(DATA_DIR / "normalization.json")
    meta = norm_data["meta"]
    species = meta["species_variables"]
    globals_v = meta["global_variables"]
    target_species = cfg["data"].get("target_species", species)

    # Build and load model (set CWD for relative paths in create_model)
    import os
    os.chdir(REPO)
    model = create_model(cfg).eval()
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Load normalization helper
    norm = NormalizationHelper(norm_data)

    # Load test sample
    shards = sorted((DATA_DIR / "test").glob("shard_*.npz"))
    print(f"Found {len(shards)} test shards")

    with np.load(shards[0]) as d:
        y_all = d["y_mat"].astype(np.float32)
        print(f"Total samples: {len(y_all)}")
        y = y_all[SAMPLE_IDX]  # [M, S]
        g = d["globals"].astype(np.float32)[SAMPLE_IDX]  # [G]
        t_phys = d["t_vec"] if d["t_vec"].ndim == 1 else d["t_vec"][SAMPLE_IDX]  # [M]

    # Map target species indices
    target_idx = [species.index(s) for s in target_species]

    # Prepare inputs
    y0 = torch.from_numpy(y[0:1])  # [1, S]
    g_tensor = torch.from_numpy(g[None, :])  # [1, G]
    y0_norm = norm.normalize(y0, species)
    g_norm = norm.normalize(g_tensor, globals_v)

    # Select query times
    M = len(t_phys)
    idx = np.linspace(1, M - 1, min(Q_COUNT, M - 1)).round().astype(int)
    t_sel = t_phys[idx]
    dt_phys = np.maximum(t_sel - t_phys[0], 0.0)
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_phys)).unsqueeze(0)  # [1, K]

    # Run model
    with torch.inference_mode():
        out = model(y0_norm.float(), dt_norm.float(), g_norm.float())  # [1, K, S_out]
        y_pred = norm.denormalize(out[0], target_species).cpu().numpy()  # [K, S_out]

    # Ground truth for target species
    y_true = y[:, target_idx]  # [M, S_out]

    # Plot with strict x-limits
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(target_species)))

    # Filter by x-limits
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pred = (t_sel >= XMIN) & (t_sel <= XMAX)
    t_gt_plot = t_phys[m_gt]
    y_gt_plot = np.clip(y_true[m_gt], 1e-30, None)
    t_pred_plot = t_sel[m_pred]
    y_pred_plot = np.clip(y_pred[m_pred], 1e-30, None)

    # Plot each species
    for i, name in enumerate(target_species):
        c = colors[i]
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.9, color=c)
        if len(t_gt_plot):
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=c, ms=5)
        if len(t_pred_plot) > 1:
            ax.loglog(t_pred_plot, y_pred_plot[:, i], '--', lw=1.4, alpha=0.85, color=c)

    # Configure axes
    ax.set_xlim(XMIN, XMAX)
    #ax.set_ylim(1e-15, 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Species Abundance")
    ax.grid(False)

    # Legend: species ordered by max abundance
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1]
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in order]
    legend_labels = [target_species[idx] for idx in order]
    leg1 = ax.legend(handles=legend_handles, labels=legend_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10, title_fontsize=11, ncol=1)
    ax.add_artist(leg1)

    # Legend: line styles
    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Model Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11)

    # Save
    fig.tight_layout()
    out_path = MODEL_DIR / "plots" / f"predictions_K{len(idx)}_sample_{SAMPLE_IDX}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # Simple error metrics
    y_true_at_sel = y_true[idx]
    rel_err = np.abs(y_pred - y_true_at_sel) / (np.abs(y_true_at_sel) + 1e-10)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()