#!/usr/bin/env python3
"""
predictions.py - Autoregressive 1-step evaluation on test data (constant dt).

Assumptions for the NEW codebase:
  - Test shards are produced by preprocessing.py and contain:
      y_mat        : [N, T, S]  (z-space; species are log-standardized)
      globals      : [N, G]     (already normalized)
      dt_norm_mat  : [N, T-1]   (dt normalized to [0,1] via log10 + min-max)
  - normalization.json (in processed_data_dir) contains:
      per_key_stats[species]["log_mean"], ["log_std"]
      dt["log_min"], dt["log_max"]
  - Exported model is a 1-step autoregressive wrapper (K=1) saved as .pt2, and
    accepts inputs: (y: [B,S], dt: [B], g: [B,G]).
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


# ============================================================================
# SETTINGS (edit these)
# ============================================================================

RUN_DIR = (ROOT / "models" / "run").resolve()                 # where export_*.pt2 lives
PROCESSED_DIR = (ROOT / "data" / "processed").resolve()       # where shards + normalization.json live
EXPORT_NAME = "export_cpu_1step.pt2"                          # exported 1-step model (CPU)

SAMPLE_IDX = 1
START_INDEX = 0
N_STEPS = 10

PLOT_SPECIES: List[str] = []  # labels without "_evolution"; [] => all
YMIN, YMAX = 1e-28, 3

TRUE_LW = 3.0
TRUE_ALPHA = 0.45
PRED_LW = 2.0
PRED_ALPHA = 1.0
PRED_LS = ":"
PRED_MARKER = "o"
PRED_MS = 3


# ============================================================================
# Helpers
# ============================================================================

def load_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_trajectory(processed_dir: Path, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      y_z:        [T, S] float32 (z-space)
      g_z:        [G] float32    (normalized globals)
      dt_norm:    [T-1] float32  (normalized dt per step)
      species:    list[str]      (from normalization.json manifest)
    """
    shards = sorted((processed_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {processed_dir / 'test'}")

    with np.load(shards[0]) as f:
        y_all = np.asarray(f["y_mat"], dtype=np.float32)            # [N,T,S]
        g_all = np.asarray(f["globals"], dtype=np.float32)          # [N,G]
        dt_all = np.asarray(f["dt_norm_mat"], dtype=np.float32)     # [N,T-1]

    if y_all.ndim != 3 or g_all.ndim != 2 or dt_all.ndim != 2:
        raise ValueError("Expected y_mat[N,T,S], globals[N,G], dt_norm_mat[N,T-1]")

    if idx < 0 or idx >= y_all.shape[0]:
        raise IndexError(f"SAMPLE_IDX={idx} out of range for shard with N={y_all.shape[0]}")

    manifest = load_json(processed_dir / "normalization.json")
    species = list(manifest.get("species_variables", []))
    if not species:
        raise KeyError("normalization.json missing species_variables")

    return y_all[idx], g_all[idx], dt_all[idx], species


def dt_norm_to_seconds(dt_norm_scalar: float, manifest: dict) -> float:
    """
    Invert preprocessing dt normalization:
      dt_norm = (log10(dt) - log_min) / (log_max - log_min)
    """
    log_min = float(manifest["dt"]["log_min"])
    log_max = float(manifest["dt"]["log_max"])
    rng = max(log_max - log_min, 1e-12)
    logdt = float(dt_norm_scalar) * rng + log_min
    return float(10.0 ** logdt)


def denormalize_species_z(y_z: np.ndarray, species: List[str], manifest: dict) -> np.ndarray:
    """
    Species are normalized as:
      z = (log10(max(y, eps)) - log_mean) / log_std
    Invert:
      y = 10 ** (z * log_std + log_mean)
    """
    stats = manifest["per_key_stats"]
    S = len(species)
    mu = np.array([stats[s]["log_mean"] for s in species], dtype=np.float64).reshape(1, S)
    sd = np.array([stats[s]["log_std"] for s in species], dtype=np.float64).reshape(1, S)
    z = y_z.astype(np.float64, copy=False)
    return 10.0 ** (z * sd + mu)


def model_step_out_to_state(out: torch.Tensor) -> torch.Tensor:
    """Support either [B,S] or [B,1,S] outputs; return [B,S]."""
    if out.ndim == 3:
        return out[:, 0, :]
    if out.ndim == 2:
        return out
    raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")


@torch.inference_mode()
def run_autoregressive_constant_dt(model, y0_z: np.ndarray, g_z: np.ndarray, dt_norm_scalar: float, n_steps: int) -> np.ndarray:
    """
    Autoregressive rollout in z-space using a constant dt_norm scalar.

    Exported 1-step wrapper expects:
      y:  [B,S]
      dt: [B]
      g:  [B,G]
    """
    state = torch.from_numpy(y0_z).float().unsqueeze(0)   # [1,S]
    g = torch.from_numpy(g_z).float().unsqueeze(0)        # [1,G]
    dt = torch.tensor([float(dt_norm_scalar)], dtype=torch.float32)  # [1]

    preds = []
    for _ in range(int(n_steps)):
        out = model(state, dt, g)
        state = model_step_out_to_state(out)
        preds.append(state)

    y_pred = torch.cat(preds, dim=0)  # [n_steps, S]
    return y_pred.cpu().numpy()


def plot_results(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, species: List[str], plot_species: List[str], out_path: Path) -> None:
    labels = [s.replace("_evolution", "") for s in species]

    if plot_species:
        keep = [i for i, lab in enumerate(labels) if lab in plot_species]
        if not keep:
            raise ValueError(f"PLOT_SPECIES did not match any species labels: {plot_species}")
        labels = [labels[i] for i in keep]
        y_true = y_true[:, keep]
        y_pred = y_pred[:, keep]

    y_true = np.clip(y_true, 1e-35, None)
    y_pred = np.clip(y_pred, 1e-35, None)

    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(order))[::-1])

    fig, ax = plt.subplots(figsize=(7, 7))

    for rank, idx in enumerate(order):
        c = colors[rank]
        ax.loglog(t, y_true[:, idx], linestyle="-", linewidth=TRUE_LW, alpha=TRUE_ALPHA, color=c)
        ax.loglog(
            t, y_pred[:, idx],
            linestyle=PRED_LS, linewidth=PRED_LW,
            marker=PRED_MARKER, markersize=PRED_MS,
            alpha=PRED_ALPHA, color=c
        )

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("Time (s) (relative)")
    ax.set_ylabel("Relative Abundance")
    ax.set_title(f"Autoregressive (constant dt): {len(t)} steps")
    ax.set_box_aspect(1)

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
    print(f"Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("AUTOREGRESSIVE CONSTANT-DT EVALUATION (1-step export)")
    print("=" * 60)

    processed_dir = PROCESSED_DIR
    run_dir = RUN_DIR

    manifest_path = processed_dir / "normalization.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing normalization.json: {manifest_path}")
    manifest = load_json(manifest_path)

    export_path = run_dir / EXPORT_NAME
    if not export_path.exists():
        raise FileNotFoundError(f"Missing exported model: {export_path} (run export.py first)")

    model = torch.export.load(export_path).module()
    print(f"Model: {export_path.name}")

    y_traj_z, g_z, dt_norm_vec, species = load_test_trajectory(processed_dir, SAMPLE_IDX)
    T, S = y_traj_z.shape
    print(f"Test: {processed_dir / 'test'} (first shard) | sample={SAMPLE_IDX} | T={T} | S={S}")

    if START_INDEX < 0 or START_INDEX >= T - 1:
        raise ValueError(f"START_INDEX={START_INDEX} must satisfy 0 <= START_INDEX <= T-2 (T={T})")

    max_steps = (T - 1) - START_INDEX
    n_steps = min(int(N_STEPS), int(max_steps))
    if n_steps <= 0:
        raise ValueError("N_STEPS must be >= 1 within the available trajectory length")

    # Constant-dt check over the evaluated horizon
    dt_slice = dt_norm_vec[START_INDEX: START_INDEX + n_steps]
    dt0 = float(dt_slice[0])
    if not np.allclose(dt_slice, dt0, rtol=1e-5, atol=1e-6):
        raise ValueError("dt_norm_mat is not constant over the selected horizon; this script assumes constant dt.")

    dt_seconds = dt_norm_to_seconds(dt0, manifest)
    print(f"START_INDEX={START_INDEX} | steps={n_steps} | dt≈{dt_seconds:.6g}s | horizon≈{n_steps * dt_seconds:.3e}s")

    y0_z = y_traj_z[START_INDEX]
    y_true_z = y_traj_z[START_INDEX + 1: START_INDEX + 1 + n_steps]
    y_pred_z = run_autoregressive_constant_dt(model, y0_z, g_z, dt0, n_steps)

    y_true = denormalize_species_z(y_true_z, species, manifest)
    y_pred = denormalize_species_z(y_pred_z, species, manifest)

    t_eval = dt_seconds * np.arange(1, n_steps + 1, dtype=np.float64)

    out_path = run_dir / "plots" / f"autoregressive_constdt_{SAMPLE_IDX}_t{START_INDEX}.png"
    plot_results(t_eval, y_true, y_pred, species, PLOT_SPECIES, out_path)

    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    mae = np.mean(np.abs(y_pred - y_true))
    print("\nError summary (physical space):")
    print(f"  rel_err mean = {rel_err.mean():.3e}")
    print(f"  rel_err max  = {rel_err.max():.3e}")
    print(f"  MAE          = {mae:.3e}")

    y_true_clip = np.clip(y_true, 1e-35, None)
    y_pred_clip = np.clip(y_pred, 1e-35, None)
    log10_err = np.mean(np.abs(np.log10(y_pred_clip) - np.log10(y_true_clip)))
    print("\nError summary (log10 space):")
    print(f"  mean |Δlog10| = {log10_err:.3f} (orders of magnitude)")

    print("=" * 60)


if __name__ == "__main__":
    main()
