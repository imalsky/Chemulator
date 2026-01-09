#!/usr/bin/env python3
"""
Simplified Flow-map AE prediction vs ground truth (no simplex ops).
- Solid lines  = ground truth from dataset
- Dashed lines = model predictions
- Select species via PLOT_SPECIES (match by base name; *_evolution suffix ignored)

Key fix:
- Slice shard y/g columns to match *config.json* (species_variables / target_species / global_variables)
  instead of blindly using all columns from normalization.json.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("science.mplstyle")

# ---------------- Paths & settings ----------------
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models" / "big_mlp"
EP_FILENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper

SAMPLE_IDX = 5
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# Choose species by base names; empty list = plot all
PLOT_SPECIES: List[str] = []


# ---------------- Utils ----------------
def load_data(data_dir: Path, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load first test shard and extract one sample trajectory."""
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        y = d["y_mat"][sample_idx].astype(np.float32)    # [T, S_all]
        g = d["globals"][sample_idx].astype(np.float32)  # [G_all]
        t_phys = d["t_vec"]
        if t_phys.ndim > 1:
            t_phys = t_phys[sample_idx]
        t_phys = t_phys.astype(np.float32)               # [T]
    return y, g, t_phys


def _indices(all_names: List[str], chosen: List[str]) -> List[int]:
    m = {n: i for i, n in enumerate(all_names)}
    return [m[n] for n in chosen]


def prepare_batch(
    y0: np.ndarray,
    g: np.ndarray,
    t_phys: np.ndarray,
    q_count: int,
    norm: NormalizationHelper,
    species_in: List[str],
    globals_used: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Prepare normalized (y, dt, g) for CPU K=1 export."""
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)

    t_sel = t_phys[q_idx]
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species_in).float()  # [1,S_in]
    if globals_used:
        g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_used).float()  # [1,G]
    else:
        g_norm = torch.from_numpy(g[None, :]).float()

    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec)).view(-1, 1).float()  # [K,1]

    K = dt_norm.shape[0]
    y_batch = y0_norm.repeat(K, 1)  # [K,S_in]
    g_batch = g_norm.repeat(K, 1)   # [K,G]
    return y_batch, dt_norm, g_batch, q_idx, t_sel


@torch.inference_mode()
def run_inference(model, y_batch: torch.Tensor, dt_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    """Call exported program. CPU K=1 export returns [B,1,S_out]; we return [B,S_out]."""
    return model(y_batch, dt_batch, g_batch)[:, 0, :]


def plot_results(
    t_phys: np.ndarray,
    y_true_full: np.ndarray,
    t_pred: np.ndarray,
    y_pred: np.ndarray,
    species_out: List[str],
    plot_species: List[str],
    out_path: Path,
) -> None:
    """Log–log plot: solid=truth, dashed=prediction; only selected species.

    Color map: viridis
    Ordering: by max ground-truth abundance (descending)
    """
    base_all = [n[:-10] if n.endswith("_evolution") else n for n in species_out]
    keep = [i for i, b in enumerate(base_all) if (not plot_species) or (b in plot_species)]
    labels = [base_all[i] for i in keep]

    y_true_full = y_true_full[:, keep]
    y_pred = y_pred[:, keep]

    tiny = 1e-35
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pr = (t_pred >= XMIN) & (t_pred <= XMAX)

    t_gt = t_phys[m_gt]
    y_gt = np.clip(y_true_full[m_gt], tiny, None)
    t_pr = t_pred[m_pr]
    y_pr = np.clip(y_pred[m_pr], tiny, None)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Order species by max ground-truth abundance (descending)
    if y_gt.size:
        max_gt = np.max(y_gt, axis=0)  # [N_keep]
        order = np.argsort(max_gt)[::-1]
    else:
        order = np.arange(len(labels))

    # Viridis colors assigned by abundance rank: max -> brightest
    n = len(order)
    if n > 0:
        color_vals = np.linspace(0.15, 0.95, n)  # avoid extremes; nice contrast
        colors = plt.cm.plasma(color_vals[::-1])  # max abundance gets ~0.95 (bright)
    else:
        colors = np.empty((0, 4))

    for rank, idx in enumerate(order):
        lab = labels[idx]
        col = colors[rank]

        ax.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.3, color=col)
        if t_gt.size:
            ax.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        ax.loglog(t_pr, y_pr[:, idx], "--", lw=3, alpha=1.0, color=col)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.set_box_aspect(1)

    # Species legend (already ordered by max ground-truth abundance)
    species_handles = [Line2D([0], [0], color=colors[r], lw=2.0) for r in range(n)]
    species_labels = [labels[i] for i in order]
    leg1 = ax.legend(handles=species_handles, labels=species_labels, loc="best", title="Species", ncol=3)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="Ground Truth"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")
    ax.set_ylim(1e-28, 3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main() -> None:
    os.chdir(REPO)
    seed_everything(42)

    cfg = load_json(MODEL_DIR / "config.json")
    data_cfg = cfg.get("data", {}) or {}
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    manifest = load_json(data_dir / "normalization.json")
    meta = manifest.get("meta", {}) or {}

    species_all = list(meta.get("species_variables", []) or [])
    globals_all = list(meta.get("global_variables", []) or [])

    # Use model config lists (these define the exported program’s expected dims)
    species_in = list(data_cfg.get("species_variables") or species_all)
    species_out = list(data_cfg.get("target_species") or species_in)
    globals_used = list(data_cfg.get("global_variables") or globals_all)

    idx_in = _indices(species_all, species_in)
    idx_out = _indices(species_all, species_out)
    idx_g = _indices(globals_all, globals_used) if globals_used else []

    # Exported program
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    norm = NormalizationHelper(manifest)

    # Load sample (full processed columns), then slice to match config
    y_all, g_all, t_phys = load_data(data_dir, SAMPLE_IDX)
    y_in = y_all[:, idx_in]        # [T, S_in]
    y_out = y_all[:, idx_out]      # [T, S_out]
    g = g_all[idx_g] if idx_g else np.empty((0,), dtype=np.float32)

    y0 = y_in[0]

    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(
        y0=y0,
        g=g,
        t_phys=t_phys,
        q_count=Q_COUNT,
        norm=norm,
        species_in=species_in,
        globals_used=globals_used,
    )

    y_pred_norm = run_inference(model, y_batch, dt_batch, g_batch)          # [K, S_out]
    y_pred = norm.denormalize(y_pred_norm, species_out).cpu().numpy()       # [K, S_out]
    y_true_sel = y_out[q_idx, :]                                            # [K, S_out]

    out_png = MODEL_DIR / "plots" / f"pred_{SAMPLE_IDX}.png"
    plot_results(t_phys, y_out, t_sel, y_pred, species_out, PLOT_SPECIES, out_png)

    rel_err = np.abs(y_pred - y_true_sel) / (np.abs(y_true_sel) + 1e-12)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()
