#!/usr/bin/env python3
"""
predictions.py - Autoregressive 1-step evaluation on test data (constant dt).

Evaluates an exported 1-step autoregressive model on preprocessed test shards,
comparing predictions against ground truth in both physical and log space.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
plt.style.use("science.mplstyle")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Evaluation settings."""
    # Paths
    run_dir: Path = ROOT / "models" / "v1"
    processed_dir: Path = ROOT / "data" / "processed"
    export_name: str = "export_cpu_1step.pt2"

    # Evaluation
    sample_idx: int = 1
    start_index: int = 0
    n_steps: int = 5

    # Plot settings
    plot_species: list = None  # None or [] => all species
    y_range: tuple = (1e-15, 3)

    # Line styles
    true_lw: float = 3.0
    true_alpha: float = 0.45
    pred_lw: float = 2.0
    pred_alpha: float = 1.0
    pred_ls: tuple = (8, 3)  # long dashes: 8pt on, 3pt off
    pred_marker: str = "o"
    pred_ms: float = 3


CFG = Config()


# ============================================================================
# Data Loading
# ============================================================================

def load_test_trajectory(processed_dir: Path, idx: int):
    """Load a single test trajectory from preprocessed shards.

    Returns: (y_z, g_z, dt_norm, species_list)
    """
    shards = sorted((processed_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {processed_dir / 'test'}")

    with np.load(shards[0]) as f:
        y_all = f["y_mat"].astype(np.float32)  # [N, T, S]
        g_all = f["globals"].astype(np.float32)  # [N, G]
        dt_all = f["dt_norm_mat"].astype(np.float32)  # [N, T-1]

    if idx < 0 or idx >= y_all.shape[0]:
        raise IndexError(f"Sample index {idx} out of range (N={y_all.shape[0]})")

    manifest = json.loads((processed_dir / "normalization.json").read_text())
    species = list(manifest.get("species_variables", []))

    return y_all[idx], g_all[idx], dt_all[idx], species


# ============================================================================
# Normalization Utilities
# ============================================================================

def dt_to_seconds(dt_norm: float, manifest: dict) -> float:
    """Convert normalized dt back to seconds."""
    log_min, log_max = manifest["dt"]["log_min"], manifest["dt"]["log_max"]
    log_dt = dt_norm * (log_max - log_min) + log_min
    return 10.0 ** log_dt


def denormalize_species(y_z: np.ndarray, species: list, manifest: dict) -> np.ndarray:
    """Convert z-normalized species values back to physical space."""
    stats = manifest["per_key_stats"]
    mu = np.array([stats[s]["log_mean"] for s in species])
    sd = np.array([stats[s]["log_std"] for s in species])
    return 10.0 ** (y_z * sd + mu)


# ============================================================================
# Model Inference
# ============================================================================

@torch.inference_mode()
def rollout(model, y0_z: np.ndarray, g_z: np.ndarray, dt_norm: float, n_steps: int) -> np.ndarray:
    """Autoregressive rollout with constant dt. Returns predictions in z-space."""
    state = torch.from_numpy(y0_z).float().unsqueeze(0)
    g = torch.from_numpy(g_z).float().unsqueeze(0)
    dt = torch.tensor([dt_norm], dtype=torch.float32)

    preds = []
    for _ in range(n_steps):
        out = model(state, dt, g)
        state = out[:, 0, :] if out.ndim == 3 else out
        preds.append(state)

    return torch.cat(preds, dim=0).numpy()


# ============================================================================
# Visualization
# ============================================================================

def plot_results(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 species: list, out_path: Path) -> None:
    """Generate comparison plot of ground truth vs predictions."""
    labels = [s.replace("_evolution", "") for s in species]

    # Filter species if specified
    if CFG.plot_species:
        keep = [i for i, lab in enumerate(labels) if lab in CFG.plot_species]
        if not keep:
            raise ValueError(f"No matching species: {CFG.plot_species}")
        labels, y_true, y_pred = [labels[i] for i in keep], y_true[:, keep], y_pred[:, keep]

    # Clip for log scale
    y_true = np.clip(y_true, 1e-35, None)
    y_pred = np.clip(y_pred, 1e-35, None)

    # Sort by max abundance (descending)
    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(order))[::-1])

    fig, ax = plt.subplots(figsize=(7, 7))

    for rank, idx in enumerate(order):
        c = colors[rank]
        ax.semilogy(t, y_true[:, idx], "-", lw=CFG.true_lw, alpha=CFG.true_alpha, color=c)
        ax.semilogy(t, y_pred[:, idx], lw=CFG.pred_lw, dashes=CFG.pred_ls,
                    marker=CFG.pred_marker, ms=CFG.pred_ms, alpha=CFG.pred_alpha, color=c)

    ax.set(xlim=(t[0], t[-1]), ylim=CFG.y_range,
           xlabel="Time (s)", ylabel="Relative Abundance",
           title=f"Autoregressive (constant dt): {len(t)} steps")
    ax.set_box_aspect(1)

    # Legends
    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(species_handles, [labels[i] for i in order],
                     loc="lower left", title="Species", ncol=2, fontsize=8)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=CFG.true_lw, alpha=CFG.true_alpha, label="Ground Truth"),
        Line2D([0], [0], color="k", lw=CFG.pred_lw, dashes=CFG.pred_ls,
               marker=CFG.pred_marker, ms=CFG.pred_ms, alpha=CFG.pred_alpha, label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_errors(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print error summary statistics."""
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    mae = np.mean(np.abs(y_pred - y_true))

    y_true_c, y_pred_c = np.clip(y_true, 1e-35, None), np.clip(y_pred, 1e-35, None)
    log_err = np.mean(np.abs(np.log10(y_pred_c) - np.log10(y_true_c)))

    print(f"\nPhysical space:  rel_err={rel_err.mean():.3e} (mean), {rel_err.max():.3e} (max), MAE={mae:.3e}")
    print(f"Log10 space:     mean |Δlog10| = {log_err:.3f} orders of magnitude")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("AUTOREGRESSIVE CONSTANT-DT EVALUATION")
    print("=" * 60)

    # Load manifest and model
    manifest_path = CFG.processed_dir / "normalization.json"
    manifest = json.loads(manifest_path.read_text())

    export_path = CFG.run_dir / CFG.export_name
    model = torch.export.load(export_path).module()
    print(f"Model: {export_path.name}")

    # Load test data
    y_z, g_z, dt_vec, species = load_test_trajectory(CFG.processed_dir, CFG.sample_idx)
    T, S = y_z.shape
    print(f"Sample {CFG.sample_idx}: T={T}, S={S}")

    # Validate and compute steps
    max_steps = (T - 1) - CFG.start_index
    n_steps = min(CFG.n_steps, max_steps)
    if n_steps <= 0:
        raise ValueError("Invalid start_index or n_steps")

    # Verify constant dt
    dt_slice = dt_vec[CFG.start_index: CFG.start_index + n_steps]
    dt0 = float(dt_slice[0])
    if not np.allclose(dt_slice, dt0, rtol=1e-5):
        raise ValueError("Non-constant dt over selected horizon")

    dt_sec = dt_to_seconds(dt0, manifest)
    print(f"Start={CFG.start_index}, steps={n_steps}, dt≈{dt_sec:.3e}s, horizon≈{n_steps * dt_sec:.3e}s")

    # Run inference
    y0_z = y_z[CFG.start_index]
    y_true_z = y_z[CFG.start_index + 1: CFG.start_index + 1 + n_steps]
    y_pred_z = rollout(model, y0_z, g_z, dt0, n_steps)

    # Denormalize
    y_true = denormalize_species(y_true_z, species, manifest)
    y_pred = denormalize_species(y_pred_z, species, manifest)

    # Plot and report
    t_eval = dt_sec * np.arange(1, n_steps + 1)
    out_path = CFG.run_dir / "plots" / f"autoregressive_constdt_{CFG.sample_idx}_t{CFG.start_index}.png"
    plot_results(t_eval, y_true, y_pred, species, out_path)
    print_errors(y_true, y_pred)

    print("=" * 60)


if __name__ == "__main__":
    main()
