#!/usr/bin/env python3
"""
Flow-map AE: prediction vs ground-truth overlay plot.

Solid lines  = ground truth (from test shard)
Dashed lines = model predictions (from exported program)

Species are selected via PLOT_SPECIES (matched by base name; any
*_evolution suffix is stripped automatically).  Leave empty to plot all.

Column slicing uses the model's config.json (species_variables,
target_species, global_variables) rather than the full normalization
manifest, so shard columns always match the exported program's dims.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Graceful style fallback — avoids a hard crash if the file is missing.
try:
    plt.style.use("science.mplstyle")
except OSError:
    warnings.warn("science.mplstyle not found; falling back to default style.")

# ────────────────────────── paths & settings ──────────────────────────
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models" / "final_model"
EP_FILENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json  # noqa: E402
from normalizer import NormalizationHelper          # noqa: E402

SAMPLE_IDX = 3          # which trajectory from the first test shard
Q_COUNT = 100           # number of query time-steps to evaluate
XMIN, XMAX = 1e-3, 1e8  # log-log x-axis limits

# Base species names to plot; empty list ⇒ plot every species.
PLOT_SPECIES: List[str] = []

# ────────────────── qualitative colour palette ────────────────────────
# 20 perceptually distinct hues (tab20 without the faint halves).
# Falls back to tab20 directly when >20 species are needed.
_TAB20_STRONG = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _species_colors(n: int) -> np.ndarray:
    """Return (n, 4) RGBA array with maximally distinct colours.

    Uses a hand-picked 20-colour list for small n, and falls back to
    the full tab20 colourmap (cycled) for larger species counts.
    """
    if n <= len(_TAB20_STRONG):
        import matplotlib.colors as mcolors
        return np.array([mcolors.to_rgba(c) for c in _TAB20_STRONG[:n]])
    cmap = plt.cm.tab20
    return cmap(np.linspace(0, 1, n))


# ──────────────────────────── data I/O ────────────────────────────────
def load_data(
    data_dir: Path, sample_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the first test shard and return one sample trajectory.

    Returns
    -------
    y : ndarray [T, S_all]   species abundances (all columns in shard)
    g : ndarray [G_all]      global parameters for this sample
    t : ndarray [T]           physical times (seconds)
    """
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        y = d["y_mat"][sample_idx].astype(np.float32)
        g = d["globals"][sample_idx].astype(np.float32)
        t = d["t_vec"]
        if t.ndim > 1:
            t = t[sample_idx]
        t = t.astype(np.float32)
    return y, g, t


def _indices(all_names: List[str], chosen: List[str]) -> List[int]:
    """Map a list of *chosen* names to their positions in *all_names*."""
    lookup = {n: i for i, n in enumerate(all_names)}
    return [lookup[n] for n in chosen]


# ──────────────────────── batch preparation ───────────────────────────
def prepare_batch(
    y0: np.ndarray,
    g: np.ndarray,
    t_phys: np.ndarray,
    q_count: int,
    norm: NormalizationHelper,
    species_in: List[str],
    globals_used: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Build a normalised (y, dt, g) batch for the CPU K=1 export.

    Selects *q_count* query indices evenly spaced from t[1] to t[-1],
    normalises y₀ and Δt, and repeats them into shape [K, …].

    Returns
    -------
    y_batch  : Tensor [K, S_in]
    dt_batch : Tensor [K, 1]
    g_batch  : Tensor [K, G]
    q_idx    : ndarray [K]      integer indices into the original t_phys
    t_sel    : ndarray [K]      physical times at those indices
    """
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)

    t_sel = t_phys[q_idx]
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species_in).float()
    if globals_used:
        g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_used).float()
    else:
        g_norm = torch.from_numpy(g[None, :]).float()
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec)).view(-1, 1).float()

    K = dt_norm.shape[0]
    y_batch = y0_norm.repeat(K, 1)
    g_batch = g_norm.repeat(K, 1)
    return y_batch, dt_norm, g_batch, q_idx, t_sel


# ──────────────────────────── inference ───────────────────────────────
@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    y_batch: torch.Tensor,
    dt_batch: torch.Tensor,
    g_batch: torch.Tensor,
) -> torch.Tensor:
    """Run the exported program and return [B, S_out] predictions.

    The CPU K=1 export produces shape [B, 1, S_out]; we squeeze dim-1.
    """
    return model(y_batch, dt_batch, g_batch)[:, 0, :]


# ──────────────────────────── plotting ────────────────────────────────
def plot_results(
    t_phys: np.ndarray,
    y_true_full: np.ndarray,
    t_pred: np.ndarray,
    y_pred: np.ndarray,
    species_out: List[str],
    plot_species: List[str],
    out_path: Path,
) -> None:
    """Log–log overlay: solid = truth, dashed = prediction.

    Species are ordered by peak ground-truth abundance (descending) and
    coloured with a qualitative palette so each is visually distinct.
    """
    # Strip "_evolution" suffixes for display labels.
    base_all = [
        n.removesuffix("_evolution") for n in species_out
    ]
    keep = [
        i for i, b in enumerate(base_all)
        if (not plot_species) or (b in plot_species)
    ]
    labels = [base_all[i] for i in keep]
    y_true_full = y_true_full[:, keep]
    y_pred = y_pred[:, keep]

    # Mask to the visible x-range.
    tiny = 1e-35
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pr = (t_pred >= XMIN) & (t_pred <= XMAX)

    t_gt = t_phys[m_gt]
    y_gt = np.clip(y_true_full[m_gt], tiny, None)
    t_pr = t_pred[m_pr]
    y_pr = np.clip(y_pred[m_pr], tiny, None)

    # Sort species by peak ground-truth abundance (descending).
    if y_gt.size:
        order = np.argsort(np.max(y_gt, axis=0))[::-1]
    else:
        order = np.arange(len(labels))

    n = len(order)
    colors = _species_colors(n)

    fig, ax = plt.subplots(figsize=(7, 7))

    for rank, idx in enumerate(order):
        col = colors[rank]
        ax.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.35, color=col)
        if t_gt.size:
            ax.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        ax.loglog(t_pr, y_pr[:, idx], "--", lw=2.5, alpha=1.0, color=col)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(1e-28, 3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.set_box_aspect(1)

    # Species legend (abundance-descending order).
    species_handles = [
        Line2D([0], [0], color=colors[r], lw=2.0) for r in range(n)
    ]
    species_labels = [labels[i] for i in order]
    leg_species = ax.legend(
        handles=species_handles, labels=species_labels,
        loc="best", title="Species", ncol=3,
    )
    ax.add_artist(leg_species)

    # Style legend (solid vs dashed).
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="Ground Truth"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ────────────────────────────── main ──────────────────────────────────
def main() -> None:
    # NOTE: os.chdir is a global side-effect; relative paths in configs
    # depend on it, so we keep it, but be aware during testing/imports.
    os.chdir(REPO)

    cfg = load_json(MODEL_DIR / "config.json")
    data_cfg = cfg.get("data", {}) or {}
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    manifest = load_json(data_dir / "normalization.json")
    meta = manifest.get("meta", {}) or {}

    species_all = list(meta.get("species_variables", []) or [])
    globals_all = list(meta.get("global_variables", []) or [])

    # Model config determines the exported program's expected dimensions.
    species_in = list(data_cfg.get("species_variables") or species_all)
    species_out = list(data_cfg.get("target_species") or species_in)
    globals_used = list(data_cfg.get("global_variables") or globals_all)

    idx_in = _indices(species_all, species_in)
    idx_out = _indices(species_all, species_out)
    idx_g = _indices(globals_all, globals_used) if globals_used else []

    # Load exported program.
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    norm = NormalizationHelper(manifest)

    # Load one sample; slice shard columns to match config.
    y_all, g_all, t_phys = load_data(data_dir, SAMPLE_IDX)
    y_in = y_all[:, idx_in]
    y_out = y_all[:, idx_out]
    g = g_all[idx_g] if idx_g else np.empty((0,), dtype=np.float32)

    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(
        y0=y_in[0],
        g=g,
        t_phys=t_phys,
        q_count=Q_COUNT,
        norm=norm,
        species_in=species_in,
        globals_used=globals_used,
    )

    y_pred_norm = run_inference(model, y_batch, dt_batch, g_batch)
    y_pred = norm.denormalize(y_pred_norm, species_out).cpu().numpy()
    y_true_sel = y_out[q_idx, :]

    out_png = MODEL_DIR / "plots" / f"pred_{SAMPLE_IDX}.png"
    plot_results(t_phys, y_out, t_sel, y_pred, species_out, PLOT_SPECIES, out_png)

    rel_err = np.abs(y_pred - y_true_sel) / (np.abs(y_true_sel) + 1e-12)
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")


if __name__ == "__main__":
    main()