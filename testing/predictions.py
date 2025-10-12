#!/usr/bin/env python3
"""Minimal Flow-map AE (exported .pt2) prediction + plotting."""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------- Paths & Imports ---------------------------------

REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/koopman-v1"   # <- change if needed
EP_FILENAME = "export_k1_cpu.pt2"                     # required by your request

# Add src to path
sys.path.insert(0, str(REPO / "src"))
from normalizer import NormalizationHelper
from utils import load_json, seed_everything

plt.style.use('science.mplstyle')

# ------------------------------ Settings -------------------------------------

SAMPLE_IDX = 1
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# Optional plot style
try:
    plt.style.use("science.mplstyle")
except Exception:
    pass

# -----------------------------------------------------------------------------

def main():
    # Ensure repo-relative paths (some helpers assume this)
    os.chdir(REPO)
    seed_everything(42)

    # ---------- Load config & normalization ----------
    cfg = load_json(MODEL_DIR / "config.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")

    norm_data = load_json(data_dir / "normalization.json")
    meta = norm_data["meta"]
    species = meta["species_variables"]
    globals_v = meta["global_variables"]
    target_species = cfg.get("data", {}).get("target_species", species)

    # ---------- Load exported program ----------
    from torch.export import load as load_exported
    ep_path = MODEL_DIR / EP_FILENAME
    if not ep_path.exists():
        raise FileNotFoundError(f"Exported program not found: {ep_path}")
    ep = load_exported(ep_path)
    gm = ep.module()    # GraphModule; do NOT call .eval() (exported modules don’t support it)

    # ---------- Normalizer ----------
    norm = NormalizationHelper(norm_data)

    # ---------- Load one test shard ----------
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found in {data_dir / 'test'}")
    with np.load(shards[0]) as d:
        # Shapes expected from your pipeline: y_mat: [N, M, S], globals: [N, G], t_vec: [M] or [N, M]
        y_all = d["y_mat"].astype(np.float32)
        g_all = d["globals"].astype(np.float32)
        t_phys_all = d["t_vec"]

    print(f"Using shard: {shards[0].name}  |  total samples: {len(y_all)}")
    y = y_all[SAMPLE_IDX]                              # [M, S]
    g = g_all[SAMPLE_IDX]                              # [G]
    t_phys = t_phys_all if t_phys_all.ndim == 1 else t_phys_all[SAMPLE_IDX]  # [M]
    M = len(t_phys)

    # ---------- Prepare anchor (t0), globals, and queries ----------
    # Normalize to z-space (species, globals)
    y0 = torch.from_numpy(y[0:1])                      # [1, S]
    g_tensor = torch.from_numpy(g[None, :])            # [1, G]
    y0_norm = norm.normalize(y0, species).to(torch.float32)
    g_norm  = norm.normalize(g_tensor, globals_v).to(torch.float32)

    # Pick query indices (exclude anchor at idx=0)
    idx = np.linspace(1, M - 1, min(Q_COUNT, M - 1)).round().astype(int)
    t_sel = t_phys[idx].astype(np.float32)
    dt_phys = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    # Normalize Δt and shape as [K,1,1]; batch y0 and g to [K,*]
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_phys)).to(torch.float32)  # [K]
    dt_norm_b11 = dt_norm.view(-1, 1, 1)                                                # [K,1,1]
    K = dt_norm_b11.shape[0]
    y_batch = y0_norm.repeat(K, 1)                                                      # [K, S]
    g_batch = g_norm.repeat(K, 1)                                                       # [K, G]

    # ---------- Inference (K = dynamic batch) ----------
    with torch.inference_mode():
        # Exported model returns z-space predictions [K, S_out]; K=1 is enforced inside the graph.
        y_pred_z = gm(y_batch, dt_norm_b11, g_batch)                                    # [K, S_out]

    # ---------- Denormalize to physical space ----------
    # Map target species indices and gather ground truth
    target_idx = [species.index(s) for s in target_species]
    y_true = y[:, target_idx]                                                           # [M, S_out]

    # Use the same target key order for denormalization
    y_pred = norm.denormalize(y_pred_z, target_species).cpu().numpy()                   # [K, S_out]

    # ------------------------------- Plot -------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(target_species)))

    # Filter by x-limits
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pred = (t_sel >= XMIN) & (t_sel <= XMAX)
    t_gt_plot = t_phys[m_gt]
    y_gt_plot = np.clip(y_true[m_gt], 1e-30, None)
    t_pred_plot = t_sel[m_pred]
    y_pred_plot = np.clip(y_pred[m_pred], 1e-30, None)

    for i, name in enumerate(target_species):
        c = colors[i]
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.9, color=c)
        if len(t_gt_plot):
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=c, ms=5)
        if len(t_pred_plot) > 1:
            ax.loglog(t_pred_plot, y_pred_plot[:, i], '--', lw=1.4, alpha=0.85, color=c)

    ax.set_xlim(XMIN, XMAX)
    #ax.set_ylim(1e-10, 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.grid(False)

    # Legend: species ordered by max abundance
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1]
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in order]
    legend_labels = [target_species[idx][:-10] for idx in order]
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

    fig.tight_layout()
    out_path = MODEL_DIR / "plots" / f"predictions_K{len(idx)}_sample_{SAMPLE_IDX}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # -------------------------- Simple error metrics --------------------------
    y_true_at_sel = y_true[idx]
    rel_err = np.abs(y_pred - y_true_at_sel) / (np.abs(y_true_at_sel) + 1e-10)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()
