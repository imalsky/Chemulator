#!/usr/bin/env python3
"""Minimal Flow-map AE (exported .pt2) prediction + plotting (handles [K,1,S] outputs)."""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------- Paths & Imports ---------------------------------

REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models/koopman-v2"        # <- change if needed
EP_FILENAME = "export_k1_cpu.pt2"             # choose your artifact

# Add src to path
sys.path.insert(0, str(REPO / "src"))
from normalizer import NormalizationHelper
from utils import load_json, seed_everything

# ------------------------------ Settings -------------------------------------

SAMPLE_IDX = 0
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass

# -----------------------------------------------------------------------------

def _load_meta_json(pt2_path: Path) -> dict:
    meta_path = pt2_path.with_suffix(pt2_path.suffix + ".meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    os.chdir(REPO)
    seed_everything(42)

    # ---------- Load config & normalization ----------
    cfg = load_json(MODEL_DIR / "config.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")

    norm_data = load_json(data_dir / "normalization.json")
    meta = norm_data["meta"]
    species = list(meta["species_variables"])
    globals_v = list(meta.get("global_variables", []))
    target_species_cfg = list(cfg.get("data", {}).get("target_species", species))

    # ---------- Load exported program ----------
    from torch.export import load as load_exported
    ep_path = MODEL_DIR / EP_FILENAME
    if not ep_path.exists():
        raise FileNotFoundError(f"Exported program not found: {ep_path}")
    ep = load_exported(ep_path)
    gm = ep.module()    # GraphModule; do NOT call .eval()

    # Optional meta info
    meta_json = _load_meta_json(ep_path)
    s_out_meta = int(meta_json["S_out"]) if "S_out" in meta_json else None

    # ---------- Normalizer ----------
    norm = NormalizationHelper(norm_data)

    # ---------- Load one test shard ----------
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found in {data_dir / 'test'}")
    with np.load(shards[0]) as d:
        y_all = d["y_mat"].astype(np.float32)   # [N, M, S]
        g_all = d["globals"].astype(np.float32) # [N, G]
        t_phys_all = d["t_vec"]                 # [M] or [N, M]

    print(f"Using shard: {shards[0].name}  |  total samples: {len(y_all)}")
    y = y_all[SAMPLE_IDX]                              # [M, S]
    g = g_all[SAMPLE_IDX]                              # [G]
    t_phys = t_phys_all if t_phys_all.ndim == 1 else t_phys_all[SAMPLE_IDX]  # [M]
    M = len(t_phys)

    # ---------- Prepare anchor (t0), globals, and queries ----------
    y0 = torch.from_numpy(y[0:1])                      # [1, S]
    g_tensor = torch.from_numpy(g[None, :])            # [1, G]
    y0_norm = norm.normalize(y0, species).to(torch.float32)
    g_norm  = norm.normalize(g_tensor, globals_v).to(torch.float32)

    qn = max(1, min(Q_COUNT, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)
    t_sel = t_phys[q_idx].astype(np.float32)
    dt_phys = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_phys)).to(torch.float32)  # [K]
    dt_norm_b11 = dt_norm.view(-1, 1, 1)                                                # [K,1,1]
    K = dt_norm_b11.shape[0]
    y_batch = y0_norm.repeat(K, 1)                                                      # [K, S]
    g_batch = g_norm.repeat(K, 1)                                                       # [K, G]

    # ---------- Inference ----------
    with torch.inference_mode():
        out = gm(y_batch, dt_norm_b11, g_batch)  # could be [K,1,S], [1,K,S], or [K,S]

    # Squeeze singleton step dim if present
    if out.dim() == 3:
        if out.size(1) == 1:
            y_pred_z = out[:, 0, :]      # [K, S_out]
        elif out.size(0) == 1:
            y_pred_z = out[0, :, :]      # [K, S_out]
        else:
            # Unexpected extra step dimension; flatten it
            y_pred_z = out.reshape(out.size(0) * out.size(1), out.size(2))
            # Align K-dependent arrays to match the new K
            K = y_pred_z.shape[0]
            # Rebuild q_idx/t_sel to the same length if needed
            if len(q_idx) != K:
                q_idx = np.linspace(1, M - 1, K).round().astype(int)
                t_sel = t_phys[q_idx].astype(np.float32)
                dt_phys = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)
    else:
        y_pred_z = out  # [K, S_out]

    S_out_pred = y_pred_z.shape[1]
    if s_out_meta is not None and s_out_meta != S_out_pred:
        print(f"[warn] meta S_out={s_out_meta} but export returned S_out={S_out_pred}; using {S_out_pred}.")

    # Align target species to predicted columns
    if S_out_pred != len(target_species_cfg):
        print(f"[info] Model predicts {S_out_pred} species; truncating/aligning target list (was {len(target_species_cfg)}).")
    target_species_used = target_species_cfg[:S_out_pred]

    # Ground truth slice in the same order
    try:
        target_idx = [species.index(s) for s in target_species_used]
    except ValueError as e:
        missing = str(e).split("'")[1]
        raise KeyError(f"Target species '{missing}' not found in species_variables.") from None

    y_true = y[:, target_idx]                        # [M, S_out_pred]

    # ---------- Denormalize predictions to physical space ----------
    # y_pred_z must be 2D [K, S_out_used] and keys length must match columns
    y_pred = norm.denormalize(y_pred_z, target_species_used).cpu().numpy()  # [K, S_out_pred]

    # ------------------------------- Plot -------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(target_species_used)))

    # Filter by x-limits
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pred = (t_sel >= XMIN) & (t_sel <= XMAX)
    t_gt_plot = t_phys[m_gt]
    y_gt_plot = np.clip(y_true[m_gt], 1e-30, None)
    t_pred_plot = t_sel[m_pred]
    y_pred_plot = np.clip(y_pred[m_pred], 1e-30, None)

    n_cols = min(y_gt_plot.shape[1], y_pred_plot.shape[1], len(target_species_used))
    for i in range(n_cols):
        name = target_species_used[i]
        c = colors[i]
        ax.loglog(t_gt_plot, y_gt_plot[:, i], '-', lw=1.8, alpha=0.9, color=c)
        if t_gt_plot.size:
            ax.loglog([t_gt_plot[0]], [y_gt_plot[0, i]], 'o', mfc='none', color=c, ms=5)
        if t_pred_plot.size > 0:
            ax.loglog(t_pred_plot, y_pred_plot[:, i], '--', lw=1.4, alpha=0.85, color=c)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.grid(False)

    # Legend: order by GT max
    order = np.argsort(np.max(y_gt_plot, axis=0))[::-1][:n_cols]
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in order]
    legend_labels = [target_species_used[idx] for idx in order]
    leg1 = ax.legend(handles=legend_handles, labels=legend_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10, title_fontsize=11, ncol=1)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Model Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11)

    fig.tight_layout()
    out_path = MODEL_DIR / "plots" / f"predictions_K{K}_sample_{SAMPLE_IDX}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # -------------------------- Simple error metrics --------------------------
    y_true_at_sel = y_true[q_idx]
    y_true_at_sel = y_true_at_sel[:, :n_cols]
    y_pred_eval = y_pred[:, :n_cols]
    rel_err = np.abs(y_pred_eval - y_true_at_sel) / (np.abs(y_true_at_sel) + 1e-10)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")

if __name__ == "__main__":
    main()
