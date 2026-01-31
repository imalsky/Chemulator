#!/usr/bin/env python3
"""
predictions.py - Autoregressive 1-step evaluation on test data (constant dt).

Evaluates an exported 1-step autoregressive model (.pt2) on preprocessed test shards,
and plots ground truth (lines) vs predictions (marker-only, no connecting line).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# Resolve repo root and make <repo>/src importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Optional plotting style (will error if the style file isn't present).
plt.style.use("science.mplstyle")


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Where the exported model and plots live.
    run_dir: Path = ROOT / "models" / "v4"
    export_name: str = "export_cpu_1step.pt2"

    # Where processed test shards + normalization live.
    processed_dir: Path = ROOT / "data" / "processed"

    # Which test trajectory to load and how much of it to roll out.
    sample_idx: int = 3
    start_index: int = 5
    n_steps: int = 490

    # Plot settings.
    y_range: tuple = (1e-30, 3)

    # Ground-truth line styling.
    true_lw: float = 3.0
    true_alpha: float = 1.0

    # Prediction marker-only styling (NO line).
    pred_alpha: float = 1.0
    pred_marker: str = "o"
    pred_ms: float = 3.5          # marker size
    pred_mew: float = 0.9         # marker edge width

    # Plot every Nth prediction marker (1 => plot all markers).
    pred_marker_every: int = 10


CFG = Config()


# =============================================================================
# Data loading
# =============================================================================

def load_test_trajectory(processed_dir: Path, idx: int):
    """
    Loads a single test trajectory from shard_0.

    Returns:
      y_z:     [T, S]   normalized (z-space) species trajectory
      g_z:     [G]      normalized globals
      dt_norm: [T-1]    normalized dt for each transition
      species: list[str] species keys, aligned with S
      manifest: normalization metadata dict
    """
    shard_path = sorted((processed_dir / "test").glob("shard_*.npz"))[0]
    with np.load(shard_path) as f:
        y_all = f["y_mat"].astype(np.float32)        # [N, T, S]
        g_all = f["globals"].astype(np.float32)      # [N, G]
        dt_all = f["dt_norm_mat"].astype(np.float32) # [N, T-1]

    manifest = json.loads((processed_dir / "normalization.json").read_text())
    species = list(manifest["species_variables"])

    return y_all[idx], g_all[idx], dt_all[idx], species, manifest


# =============================================================================
# Normalization helpers
# =============================================================================

def dt_to_seconds(dt_norm: float, manifest: dict) -> float:
    """
    Converts normalized dt back to seconds.
    """
    log_min = manifest["dt"]["log_min"]
    log_max = manifest["dt"]["log_max"]
    log_dt = dt_norm * (log_max - log_min) + log_min
    return 10.0 ** log_dt


def denormalize_species(y_z: np.ndarray, species: list, manifest: dict) -> np.ndarray:
    """
    Converts normalized (z-space) species back to physical abundances.
    """
    stats = manifest["per_key_stats"]
    mu = np.array([stats[s]["log_mean"] for s in species], dtype=np.float64)
    sd = np.array([stats[s]["log_std"] for s in species], dtype=np.float64)
    return 10.0 ** (y_z * sd + mu)


# =============================================================================
# Inference
# =============================================================================

def _step_model(model, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Runs one model step and normalizes output shape to [B, S].
    """
    out = model(y, dt, g)
    return out[:, 0, :] if out.ndim == 3 else out


@torch.inference_mode()
def rollout(model, y0_z: np.ndarray, g_z: np.ndarray, dt_norm: float, n_steps: int,
            device: str, dtype: torch.dtype) -> np.ndarray:
    """
    Autoregressive rollout with constant dt, returning [n_steps, S] in z-space.
    """
    y = torch.from_numpy(y0_z).to(device=device, dtype=dtype).unsqueeze(0)  # [1, S]
    g = torch.from_numpy(g_z).to(device=device, dtype=dtype).unsqueeze(0)   # [1, G]
    dt = torch.tensor([float(dt_norm)], device=device, dtype=dtype)         # [1]

    ys = []
    for _ in range(n_steps):
        y = _step_model(model, y, dt, g)   # [1, S]
        ys.append(y[0])                   # [S]

    return torch.stack(ys, dim=0).cpu().numpy()  # [n_steps, S]


# =============================================================================
# Plotting / Metrics
# =============================================================================

def _distinct_colors(n: int):
    """
    Produces a list of visually distinct RGB tuples for categorical plotting.
    """
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    cols = []
    for name in cmaps:
        cmap = plt.get_cmap(name)
        cols.extend(list(getattr(cmap, "colors", [])))

    if n > len(cols):
        hs = np.linspace(0, 1, n, endpoint=False)
        cols = [tuple(mcolors.hsv_to_rgb((h, 0.85, 0.95))) for h in hs]

    return [tuple(mcolors.to_rgb(c)) for c in cols[:n]]


def plot_results(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, species: list, out_path: Path) -> None:
    """
    Plots ground truth as lines and predictions as marker-only (subsampled by CFG.pred_marker_every).
    """
    labels = [s.replace("_evolution", "") for s in species]

    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = _distinct_colors(len(order))

    # Indices of timesteps to draw markers for (1 => all, 2 => every other, etc).
    me = int(CFG.pred_marker_every)
    marker_idx = np.arange(0, len(t), me)

    fig, ax = plt.subplots(figsize=(7, 7))

    for r, i in enumerate(order):
        c = colors[r]

        # Ground truth: continuous line at all timesteps.
        ax.plot(
            t,
            y_true[:, i],
            linestyle="-",
            lw=CFG.true_lw,
            alpha=CFG.true_alpha,
            color=c,
        )

        # Predictions: markers only at subsampled timesteps.
        ax.plot(
            t[marker_idx],
            y_pred[marker_idx, i],
            linestyle="None",
            marker=CFG.pred_marker,
            ms=CFG.pred_ms,
            alpha=CFG.pred_alpha,
            color=c,
            markerfacecolor="none",
            markeredgecolor=c,
            markeredgewidth=CFG.pred_mew,
        )

    ax.set_yscale("log")
    ax.set(
        xlim=(t[0], t[-1]),
        ylim=CFG.y_range,
        xlabel="Time (s)",
        ylabel="Relative Abundance",
        title=f"Autoregressive (constant dt): {len(t)} steps",
    )
    ax.set_box_aspect(1)

    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(
        species_handles,
        [labels[i] for i in order],
        loc="lower left",
        title="Species",
        ncol=2,
        fontsize=8,
    )
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=CFG.true_lw, alpha=CFG.true_alpha, label="Ground Truth"),
        Line2D(
            [0], [0],
            color="k",
            linestyle="None",
            marker=CFG.pred_marker,
            ms=CFG.pred_ms,
            alpha=CFG.pred_alpha,
            markerfacecolor="none",
            markeredgecolor="k",
            markeredgewidth=CFG.pred_mew,
            label=f"Prediction (every {me} step{'s' if me != 1 else ''})",
        ),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_errors(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Prints simple physical-space and log-space errors over the whole rollout.
    """
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    mae = np.mean(np.abs(y_pred - y_true))
    log_err = np.mean(np.abs(np.log10(y_pred) - np.log10(y_true)))

    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    processed_dir = CFG.processed_dir
    export_path = (CFG.run_dir / CFG.export_name).resolve()

    device = "cuda" if "cuda" in CFG.export_name.lower() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    model = torch.export.load(export_path).module().to(device=device)

    y_z, g_z, dt_vec, species, manifest = load_test_trajectory(processed_dir, CFG.sample_idx)

    dt0 = float(dt_vec[CFG.start_index])
    dt_sec = dt_to_seconds(dt0, manifest)

    y0_z = y_z[CFG.start_index]
    y_true_z = y_z[CFG.start_index + 1: CFG.start_index + 1 + CFG.n_steps]

    y_pred_z = rollout(model, y0_z, g_z, dt0, CFG.n_steps, device=device, dtype=dtype)

    y_true = denormalize_species(y_true_z, species, manifest)
    y_pred = denormalize_species(y_pred_z, species, manifest)

    t_eval = dt_sec * np.arange(1, CFG.n_steps + 1)

    out_path = CFG.run_dir / "plots" / f"autoregressive_constdt_{CFG.sample_idx}_t{CFG.start_index}.png"
    plot_results(t_eval, y_true, y_pred, species, out_path)
    print_errors(y_true, y_pred)


if __name__ == "__main__":
    main()
