#!/usr/bin/env python3
"""
predictions.py - Autoregressive constant-timestep evaluation on test data.

This script is intentionally specialized for the "constant dt" setting:
    - The test shard provides a single shared t_vec with uniform spacing.
    - dt is inferred from t_vec and normalized using normalization.json.
    - Predictions are produced autoregressively:
          y[t+1] = model(y[t], dt_norm, g)
    - Ground truth vs predictions are compared at matching discrete time indices.

Ground truth = thick solid lines (medium alpha).
Predictions  = dotted lines with dot markers (high alpha).
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import load_json_config
from normalizer import NormalizationHelper

# ============================================================================
# Settings - EDIT THESE
# ============================================================================

MODEL_DIR = ROOT / "models" / "rollout_run"

SAMPLE_IDX = 1            # Trajectory index within the first test shard
START_INDEX = 0           # Time index in the trajectory to start autoregression from
N_STEPS = 5000              # Number of autoregressive steps to evaluate

PLOT_SPECIES: List[str] = []  # Empty = all species (labels without "_evolution")
YMIN, YMAX = 1e-28, 3

# Styling
TRUE_LW = 3.0
TRUE_ALPHA = 0.45

PRED_LW = 2.0
PRED_ALPHA = 1.0
PRED_LS = ":"                 # dotted
PRED_MARKER = "o"
PRED_MS = 3


# ============================================================================
# Data Loading (Constant dt)
# ============================================================================

def load_test_trajectory(
    processed_dir: Path,
    idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one test trajectory from the first test shard (already in z-space).

    Returns:
        y_z: [T, S] float32
        g_z: [G] float32
        t_vec: [T] float64
    """
    shards = sorted((processed_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {processed_dir / 'test'}")

    with np.load(shards[0]) as f:
        y_all = np.asarray(f["y_mat"], dtype=np.float32)
        g_all = np.asarray(f["globals"], dtype=np.float32)
        t_vec = np.asarray(f["t_vec"], dtype=np.float64)

    if y_all.ndim != 3 or g_all.ndim != 2 or t_vec.ndim != 1:
        raise ValueError("Expected y_mat[N,T,S], globals[N,G], t_vec[T]")

    if idx < 0 or idx >= y_all.shape[0]:
        raise IndexError(f"SAMPLE_IDX={idx} out of range for shard with N={y_all.shape[0]} trajectories")

    return y_all[idx], g_all[idx], t_vec


def infer_constant_dt_seconds(t_vec: np.ndarray) -> float:
    """Infer constant dt from t_vec and validate uniform spacing."""
    if t_vec.size < 2:
        raise ValueError("t_vec must have at least 2 points")
    diffs = np.diff(t_vec)
    dt = float(diffs[0])
    atol = 1e-12 * max(1.0, abs(dt))
    if not np.allclose(diffs, dt, rtol=1e-6, atol=atol):
        raise ValueError("t_vec is not uniformly spaced; this script assumes constant dt")
    return dt


# ============================================================================
# Normalization Helpers
# ============================================================================

def denormalize_matrix(z: torch.Tensor, keys: List[str], norm: NormalizationHelper) -> torch.Tensor:
    """Denormalize last dimension of z using corresponding keys."""
    out = torch.empty_like(z)
    for i, key in enumerate(keys):
        out[..., i] = norm.denormalize(z[..., i], key)
    return out


# ============================================================================
# Autoregressive Inference (Constant dt)
# ============================================================================

@torch.inference_mode()
def run_autoregressive_constant_dt(
    model,
    y0_z: np.ndarray,
    g_z: np.ndarray,
    dt_norm: torch.Tensor,
    n_steps: int,
) -> np.ndarray:
    """
    Run constant-dt autoregressive steps in z-space.

    Important: export_cpu.pt2 expects dt input shape [B, 1] (not [B,1,1]).
    """
    if dt_norm.ndim != 0:
        dt_norm = dt_norm.view(())

    state = torch.from_numpy(y0_z).float().unsqueeze(0)     # [B=1, S]
    g_batch = torch.from_numpy(g_z).float().unsqueeze(0)    # [B=1, G]

    B = state.shape[0]
    dt_batch = dt_norm.to(dtype=torch.float32).view(1, 1).expand(B, 1)  # [B, 1]

    preds: List[torch.Tensor] = []
    for _ in range(int(n_steps)):
        next_state = model(state, dt_batch, g_batch)[:, 0, :]  # [B, S]
        preds.append(next_state)
        state = next_state

    y_pred_z = torch.cat(preds, dim=0)  # [n_steps, S] since B=1
    return y_pred_z.cpu().numpy()


# ============================================================================
# Plotting
# ============================================================================

def plot_results(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species: List[str],
    plot_species: List[str],
    out_path: Path,
) -> None:
    """Plot ground truth vs autoregressive predictions (constant dt)."""
    labels = [s.replace("_evolution", "") for s in species]

    if plot_species:
        keep = [i for i, lab in enumerate(labels) if lab in plot_species]
        if not keep:
            raise ValueError(f"PLOT_SPECIES did not match any species labels: {plot_species}")
        labels = [labels[i] for i in keep]
        y_true = y_true[:, keep]
        y_pred = y_pred[:, keep]

    # Use AR extent for x-limits (exactly the evaluated horizon)
    x_min = float(t[0])
    x_max = float(t[-1])

    # Clip for numerical stability on log scale
    y_true = np.clip(y_true, 1e-35, None)
    y_pred = np.clip(y_pred, 1e-35, None)

    # Order by max ground truth abundance
    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(order))[::-1])

    fig, ax = plt.subplots(figsize=(7, 7))

    for rank, idx in enumerate(order):
        c = colors[rank]
        # Ground truth: thick solid line, medium alpha
        ax.loglog(
            t,
            y_true[:, idx],
            linestyle="-",
            linewidth=TRUE_LW,
            alpha=TRUE_ALPHA,
            color=c,
        )
        # Prediction: dotted + dots, high alpha
        ax.loglog(
            t,
            y_pred[:, idx],
            linestyle=PRED_LS,
            linewidth=PRED_LW,
            marker=PRED_MARKER,
            markersize=PRED_MS,
            alpha=PRED_ALPHA,
            color=c,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.set_title(f"Autoregressive (constant dt): {len(t)} steps")
    ax.set_box_aspect(1)

    # Legends
    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(
        species_handles,
        [labels[i] for i in order],
        loc="upper left",
        title="Species",
        ncol=2,
        fontsize=8,
    )
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="black", lw=TRUE_LW, ls="-", alpha=TRUE_ALPHA, label="Ground Truth"),
        Line2D([0], [0], color="black", lw=PRED_LW, ls=PRED_LS, marker=PRED_MARKER, markersize=PRED_MS, alpha=PRED_ALPHA, label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("  AUTOREGRESSIVE CONSTANT-DT EVALUATION")
    print("=" * 60)

    cfg = load_json_config(MODEL_DIR / "config.json")

    data_dir = Path(cfg["paths"]["processed_data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    data_cfg = cfg.get("data", {})
    species = list(data_cfg.get("species_variables", []))
    if not species:
        raise KeyError("data.species_variables must be present in config.json")

    manifest = load_json_config(data_dir / "normalization.json")
    norm = NormalizationHelper(manifest)

    export_path = MODEL_DIR / "export_cpu.pt2"
    if not export_path.exists():
        raise FileNotFoundError(f"Missing exported model: {export_path}. Run export.py first.")

    model = torch.export.load(export_path).module()
    print(f"  Model: {export_path.name}")

    y_traj_z, g_z, t_vec = load_test_trajectory(data_dir, SAMPLE_IDX)
    T = int(t_vec.shape[0])

    dt_seconds = infer_constant_dt_seconds(t_vec)
    dt_norm = norm.normalize_dt_from_phys(float(dt_seconds)).to(dtype=torch.float32).view(())

    print(f"  Test shard: {data_dir / 'test'} (first shard)")
    print(f"  Trajectory index: {SAMPLE_IDX}")
    print(f"  Trajectory length: T={T}")
    print(f"  Inferred dt: {dt_seconds:.6g} seconds")

    if START_INDEX < 0 or START_INDEX >= T - 1:
        raise ValueError(f"START_INDEX={START_INDEX} must satisfy 0 <= START_INDEX <= T-2 (T={T})")

    max_steps = (T - 1) - START_INDEX
    n_steps = min(int(N_STEPS), int(max_steps))
    if n_steps <= 0:
        raise ValueError("N_STEPS must be >= 1 within the available trajectory length")

    print(f"  Start index: {START_INDEX}")
    print(f"  Evaluating steps: {n_steps} (requested {N_STEPS}, available {max_steps})")
    print(f"  Total horizon: {n_steps * dt_seconds:.3e} seconds")

    y0_z = y_traj_z[START_INDEX]
    y_true_z = y_traj_z[START_INDEX + 1: START_INDEX + 1 + n_steps]
    t_eval = t_vec[START_INDEX + 1: START_INDEX + 1 + n_steps]

    y_pred_z = run_autoregressive_constant_dt(model, y0_z, g_z, dt_norm, n_steps)

    y_true = denormalize_matrix(torch.from_numpy(y_true_z), species, norm).numpy()
    y_pred = denormalize_matrix(torch.from_numpy(y_pred_z), species, norm).numpy()

    out_path = MODEL_DIR / "plots" / f"autoregressive_constdt_{SAMPLE_IDX}_t{START_INDEX}.png"
    plot_results(t_eval, y_true, y_pred, species, PLOT_SPECIES, out_path)

    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    mae = np.mean(np.abs(y_pred - y_true))
    print("\n  Error summary (physical space):")
    print(f"    rel_err mean = {rel_err.mean():.3e}")
    print(f"    rel_err max  = {rel_err.max():.3e}")
    print(f"    MAE          = {mae:.3e}")

    y_true_clip = np.clip(y_true, 1e-35, None)
    y_pred_clip = np.clip(y_pred, 1e-35, None)
    log10_err = np.mean(np.abs(np.log10(y_pred_clip) - np.log10(y_true_clip)))
    print("\n  Error summary (log10 space):")
    print(f"    mean |Δlog10| = {log10_err:.3f} (orders of magnitude)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
