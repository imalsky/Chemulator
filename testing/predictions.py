#!/usr/bin/env python3
"""
Minimal Flow-map AE prediction + plotting against a test shard.

Assumes an exported program saved as models/autoencoder/export_k1_cpu.pt2
with K=1 baked into the graph. So we pass dt as [B,1] where B == #queries.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------- Paths -------------------------------------------
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/autoencoder"
EP_FILENAME = "export_k1_cpu.pt2"   # change if you named it differently

sys.path.insert(0, str(REPO / "src"))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

# -------------------------- User settings ------------------------------------
SAMPLE_IDX = 2           # which trajectory in first test shard to plot
Q_COUNT    = 100          # how many query times between t1..tM
XMIN, XMAX = 1e-3, 1e8    # x-limits for log-log plot

# -----------------------------------------------------------------------------

def main():
    os.chdir(REPO)
    seed_everything(42)

    # ------- Load config + manifest (strict, no rehydration games) ------------
    cfg = load_json(MODEL_DIR / "config.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")

    manifest = load_json(data_dir / "normalization.json")
    meta     = manifest.get("meta", {})
    species  = list(meta.get("species_variables", []))
    globals_ = list(meta.get("global_variables", []))
    if not species:
        raise RuntimeError("normalization.json is missing meta.species_variables")

    # ------- Load exported program (K=1 graph) --------------------------------
    from torch.export import load as load_exported
    ep_path = MODEL_DIR / EP_FILENAME
    if not ep_path.exists():
        raise FileNotFoundError(f"Exported program not found: {ep_path}")
    ep = load_exported(ep_path)
    gm = ep.module()   # params/buffers are bound; call with (y, dt, g)

    norm = NormalizationHelper(manifest)

    # ------- Grab first test shard and one sample -----------------------------
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found in {data_dir / 'test'}")
    with np.load(shards[0]) as d:
        y_all      = d["y_mat"].astype(np.float32)   # [N, M, S]
        g_all      = d["globals"].astype(np.float32) # [N, G]
        t_phys_all = d["t_vec"]                      # [M] or [N, M]


    y      = y_all[SAMPLE_IDX]                                   # [M, S]
    g      = g_all[SAMPLE_IDX]                                   # [G]
    t_phys = t_phys_all if t_phys_all.ndim == 1 else t_phys_all[SAMPLE_IDX]  # [M]
    M      = len(t_phys)
    if M < 2:
        raise RuntimeError("Trajectory has <2 time points")

    # ------- Build anchor and queries -----------------------------------------
    y0        = torch.from_numpy(y[0:1])                         # [1, S]
    g_tensor  = torch.from_numpy(g[None, :])                     # [1, G]
    y0_norm   = norm.normalize(y0, species).to(torch.float32)    # [1, S]
    g_norm    = norm.normalize(g_tensor, globals_).to(torch.float32) if globals_ else g_tensor.to(torch.float32)

    # Query times: indices in [1..M-1]
    qn     = max(1, min(Q_COUNT, M - 1))
    q_idx  = np.linspace(1, M - 1, qn).round().astype(int)
    t_sel  = t_phys[q_idx].astype(np.float32)
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)   # offsets from t0 (seconds)

    # Normalize dt -> [0,1] (log-min-max per manifest.dt)
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec))   # [K]
    dt_b1   = dt_norm.view(-1, 1).to(torch.float32)                   # [K,1]  **IMPORTANT** K=1 baked in graph

    # Tile y0 and g to match B=K
    K       = dt_b1.shape[0]
    y_batch = y0_norm.repeat(K, 1)                                    # [K, S]
    g_batch = g_norm.repeat(K, 1)                                     # [K, G]

    # ------- Inference (normalized z) -----------------------------------------
    with torch.inference_mode():
        out = gm(y_batch, dt_b1, g_batch)   # exported forward returns [B,1,S_out] with K=1
    if out.dim() != 3 or out.size(1) != 1:
        raise RuntimeError(f"Unexpected export output shape: {tuple(out.shape)} (expected [B,1,S])")
    y_pred_z = out[:, 0, :]                                         # [K, S_out]

    # Basic shape sanity
    S_out = y_pred_z.shape[1]
    if S_out != len(species):
        raise RuntimeError(f"S_out ({S_out}) != #species from manifest ({len(species)}). "
                           f"Don’t guess; re-export or fix your manifest.")

    # ------- Denormalize to physical space ------------------------------------
    y_pred = norm.denormalize(y_pred_z, species).cpu().numpy()       # [K, S]
    y_true = y[:, :S_out]                                            # [M, S] (ordered like species)
    y_true_sel = y_true[q_idx]                                       # [K, S]

    # ------- Quick sanity checks ----------------------------------------------
    tiny = 1e-35
    sums_pred = y_pred.sum(axis=1)
    print(f"[check] pred row-sum: min={sums_pred.min():.4e}, max={sums_pred.max():.4e}")
    print(f"[check] any NaN in pred? {np.isnan(y_pred).any()}")

    # ------- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, S_out))

    # Restrict to the visible x-range
    m_gt   = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pred = (t_sel   >= XMIN) & (t_sel   <= XMAX)

    t_gt_plot   = t_phys[m_gt]
    y_gt_plot   = np.clip(y_true[m_gt], tiny, None)
    t_pred_plot = t_sel[m_pred]
    y_pred_plot = np.clip(y_pred[m_pred], tiny, None)

    for i in range(S_out):
        c = colors[i]
        # GT
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.9, color=c)
        if t_gt_plot.size:
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=c, ms=5)
        # Pred
        if t_pred_plot.size:
            ax.loglog(t_pred_plot, y_pred_plot[:, i], '--', lw=1.4, alpha=0.85, color=c)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.grid(False)

    # Legend ordered by GT max
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1]
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in order]
    legend_labels  = [species[idx] for idx in order]
    leg1 = ax.legend(handles=legend_handles, labels=legend_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10, title_fontsize=11, ncol=1)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-',  label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Model Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11)

    fig.tight_layout()
    out_path = MODEL_DIR / "plots" / f"pred_vs_gt_{SAMPLE_IDX}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # ------- Simple error metrics (on the same timestamps) --------------------
    y_true_eval = np.clip(y_true_sel, tiny, None)
    y_pred_eval = np.clip(y_pred,     tiny, None)
    rel_err = np.abs(y_pred_eval - y_true_eval) / (np.abs(y_true_eval) + 1e-10)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")

if __name__ == "__main__":
    main()
