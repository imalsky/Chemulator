#!/usr/bin/env python3
"""
Plot raw (denormalized) test profiles exactly as stored in the processed dataset.

- Loads processed data + normalization.json from the model's config.json path.
- Denormalizes species trajectories and plots EVERY recorded point (no interpolation or resampling).
- Does not use or load any model artifacts.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================== Globals / Settings ===============================

# Repo + model layout
REPO = Path(__file__).parent.parent
MODEL_NAME = "stable-lpv-koopman"                   # <- your model folder name
MODEL_DIR = REPO / "models" / MODEL_NAME            # contains config.json

# What to plot
SAMPLE_IDX = 14                                      # which test sample to show
XMIN, XMAX = None, None                             # set to floats (seconds) or leave None
PLOT_LOGLOG = True                                  # True => log-log axes; zeros/negatives hidden
SCATTER_MARKER = 'o'                                # marker only (no lines => no interpolation)
SCATTER_SIZE = 2
SCATTER_ALPHA = 0.9

# Output
OUT_DIR = MODEL_DIR / "plots"
OUT_NAME = f"test_profile_sample_{SAMPLE_IDX}.png"

# Optional MPL style (ignored if missing)
try:
    plt.style.use("science.mplstyle")
except Exception:
    pass

# Make src importable
sys.path.insert(0, str(REPO / "src"))
from normalizer import NormalizationHelper  # uses only normalization.json
from utils import load_json                 # small helper


# ================================ Helpers =====================================

def _load_meta_json_from_norm(norm_data: dict) -> dict:
    meta = norm_data.get("meta", {})
    if not meta:
        raise KeyError("normalization.json missing 'meta' section.")
    return meta


def _load_processed_paths() -> tuple[Path, dict]:
    cfg = load_json(MODEL_DIR / "config.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")
    norm_path = data_dir / "normalization.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.json not found: {norm_path}")
    norm_data = load_json(norm_path)
    return data_dir, norm_data


# ================================ Main ========================================

def main():
    os.chdir(REPO)

    # ---------- Locate processed data + normalization ----------
    data_dir, norm_data = _load_processed_paths()
    meta = _load_meta_json_from_norm(norm_data)
    species = list(meta["species_variables"])               # order used in y_mat columns

    # ---------- Build normalizer (JSON-only) ----------
    norm = NormalizationHelper(norm_data)

    # ---------- Load ONE test shard (raw points; no resampling) ----------
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found under {data_dir / 'test'}")
    shard_path = shards[0]

    with np.load(shard_path) as d:
        y_all = d["y_mat"].astype(np.float32)              # [N, M, S]
        t_all = d["t_vec"]                                 # [M] or [N, M]

    print(f"Using shard: {shard_path.name} | total samples: {len(y_all)}")

    if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(y_all):
        raise IndexError(f"SAMPLE_IDX={SAMPLE_IDX} out of range [0, {len(y_all)-1}]")

    # Slice one sample's trajectory (raw underlying points)
    y_norm = y_all[SAMPLE_IDX]                             # [M, S] (normalized)
    if t_all.ndim == 1:
        t_phys = t_all.astype(np.float64)                  # [M]
    else:
        t_phys = t_all[SAMPLE_IDX].astype(np.float64)      # [M]

    # ---------- Denormalize species to physical space ----------
    # Keep every point; do NOT filter, smooth, or interpolate.
    y_norm_t = torch.from_numpy(y_norm)                    # [M, S] torch
    y_phys_t = norm.denormalize(y_norm_t, species)         # [M, S] torch
    y_phys = y_phys_t.cpu().numpy()                        # numpy

    # Optional time range mask (still "exactly the stored points" within range)
    if XMIN is not None or XMAX is not None:
        xmin = -np.inf if XMIN is None else float(XMIN)
        xmax = +np.inf if XMAX is None else float(XMAX)
        m = (t_phys >= xmin) & (t_phys <= xmax)
        t_plot = t_phys[m]
        y_plot = y_phys[m]
    else:
        t_plot = t_phys
        y_plot = y_phys

    # If using log axes, drop non-positive points (cannot display on log scale).
    # We DO NOT modify values (no clipping); we simply omit invalid points per species.
    if PLOT_LOGLOG:
        valid_mask = t_plot > 0
        t_plot = t_plot[valid_mask]
        y_plot = y_plot[valid_mask, :]

    if t_plot.size == 0:
        raise RuntimeError("No points remain to plot after applying axis/mask constraints.")

    # ---------- Plot (scatter ONLY; no lines => no interpolation) ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20(np.linspace(0, 0.95, y_plot.shape[1]))

    # For each species, scatter ALL points exactly once
    for i, name in enumerate(species[:y_plot.shape[1]]):
        if PLOT_LOGLOG:
            # Drop non-positive y values on log scale (can't display)
            mask_y = y_plot[:, i] > 0
            tp = t_plot[mask_y]
            yp = y_plot[mask_y, i]
        else:
            tp = t_plot
            yp = y_plot[:, i]

        if tp.size == 0:
            continue

        if PLOT_LOGLOG:
            ax.loglog(tp, yp, linestyle='none',
                      marker=SCATTER_MARKER, ms=SCATTER_SIZE,
                      alpha=SCATTER_ALPHA, color=colors[i],
                      label=name if i == 0 else None)
        else:
            ax.plot(tp, yp, linestyle='none',
                    marker=SCATTER_MARKER, ms=SCATTER_SIZE,
                    alpha=SCATTER_ALPHA, color=colors[i],
                    label=name if i == 0 else None)

    # Axis labels/grid
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance (phys)")
    ax.grid(False)

    # Legend by GT max (same ordering trick as before)
    # Compute per-species max on the actually-plotted points to avoid bias.
    col_max = []
    for i in range(y_plot.shape[1]):
        if PLOT_LOGLOG:
            vals = y_plot[:, i]
            vals = vals[vals > 0]
        else:
            vals = y_plot[:, i]
        col_max.append(vals.max() if vals.size else -np.inf)

    order = np.argsort(np.array(col_max))[::-1]
    legend_handles = [Line2D([0], [0], color=colors[idx], marker=SCATTER_MARKER,
                             linestyle='none', markersize=SCATTER_SIZE, alpha=SCATTER_ALPHA)
                      for idx in order[:min(len(order), y_plot.shape[1])]]
    legend_labels = [species[idx] for idx in order[:min(len(order), y_plot.shape[1])]]
    leg1 = ax.legend(handles=legend_handles, labels=legend_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.6),
                     title='Species', fontsize=10, title_fontsize=11, ncol=1)
    ax.add_artist(leg1)

    # Style key for clarity (still no lines drawn)
    style_handles = [
        Line2D([0], [0], color='black', marker=SCATTER_MARKER, linestyle='none',
               markersize=SCATTER_SIZE, alpha=1.0, label='Raw Points'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved -> {out_path}")


if __name__ == "__main__":
    main()
