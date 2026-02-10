#!/usr/bin/env python3
"""
predictions.py - Autoregressive 1-step evaluation on test data (constant dt), PHYSICAL-space model.

Hard requirements:
- Uses exactly ONE exported artifact:
    export_cpu_dynB_1step_phys.pt2
- Runs on CPU only (simple + deterministic).
- Model expects PHYSICAL inputs:
    y_phys : [B, S]   (positive)
    dt_sec : [B]      (seconds)
    g_phys : [B, G]   (same units as preprocessing)
- We load normalization.json only to invert processed test shard (z-space -> physical)
  for inputs/GT + dt_seconds inversion for constant dt.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# Repo root and import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

plt.style.use("science.mplstyle")

# =============================================================================
# Globals (single export, CPU only)
# =============================================================================
RUN_DIR = (ROOT / "models" / "v1_done_1000_epochs").resolve()
EXPORT_PATH = (RUN_DIR / "export_cpu_dynB_1step_phys.pt2").resolve()

PROCESSED_DIR = (ROOT / "data" / "processed").resolve()

DEVICE = "cpu"
DTYPE = torch.float32


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    # Which test trajectory to load and how much of it to roll out.
    sample_idx: int = 2
    start_index: int = 1
    n_steps: int = 498

    # Plot settings.
    y_range: tuple = (1e-30, 3)

    # Ground-truth line styling.
    true_lw: float = 3.0
    true_alpha: float = 1.0

    # Prediction marker-only styling (NO line).
    pred_alpha: float = 1.0
    pred_marker: str = "o"
    pred_ms: float = 3.5
    pred_mew: float = 0.9

    # Plot every Nth prediction marker (1 => plot all markers).
    pred_marker_every: int = 10


CFG = Config()


# =============================================================================
# Data loading
# =============================================================================
def load_test_trajectory(processed_dir: Path, idx: int):
    """
    Loads a single test trajectory from the first test shard.

    Returns:
      y_z:     [T, S]   normalized (z-space) species trajectory
      g_z:     [G]      normalized globals
      dt_norm: [T-1]    normalized dt for each transition (0..1)
      species_keys: list[str] species keys aligned with S
      gvars: list[str] global variable names aligned with G
      manifest: normalization metadata dict
    """
    shard_path = sorted((processed_dir / "test").glob("shard_*.npz"))[0]
    with np.load(shard_path) as f:
        y_all = f["y_mat"].astype(np.float32)  # [N, T, S]
        g_all = f["globals"].astype(np.float32)  # [N, G]
        dt_all = f["dt_norm_mat"].astype(np.float32)  # [N, T-1]

    manifest = json.loads((processed_dir / "normalization.json").read_text())
    species_keys = list(manifest["species_variables"])
    gvars = list(manifest.get("global_variables") or manifest.get("meta", {}).get("global_variables") or [])

    return y_all[idx], g_all[idx], dt_all[idx], species_keys, gvars, manifest


# =============================================================================
# Normalization inversion (z -> physical) for processed shards
# =============================================================================
def dt_to_seconds(dt_norm: float, manifest: dict) -> float:
    a = float(manifest["dt"]["log_min"])
    b = float(manifest["dt"]["log_max"])
    log_dt = float(dt_norm) * (b - a) + a
    return 10.0 ** log_dt


def _method_for_key(key: str, manifest: dict) -> str:
    methods = (manifest.get("methods") or manifest.get("normalization_methods") or {})
    default = str(manifest.get("default_method", "standard"))
    return str(methods.get(key, default)).lower().replace("_", "-")


def denormalize_species(y_z: np.ndarray, species_keys: list[str], manifest: dict) -> np.ndarray:
    """
    Inverts species z-space -> physical. Assumes species are log-standard in manifest.
    """
    stats = manifest["per_key_stats"]
    mu = np.array([stats[k]["log_mean"] for k in species_keys], dtype=np.float64)
    sd = np.array([stats[k]["log_std"] for k in species_keys], dtype=np.float64)
    return (10.0 ** (y_z.astype(np.float64) * sd + mu)).astype(np.float64)


def denormalize_globals(g_z: np.ndarray, gvars: list[str], manifest: dict) -> np.ndarray:
    """
    Inverts globals z-space -> physical using each key's configured method.
    """
    stats = manifest["per_key_stats"]
    g_phys = np.zeros((len(gvars),), dtype=np.float64)

    for i, nm in enumerate(gvars):
        m = _method_for_key(nm, manifest)
        st = stats[nm]
        z = float(g_z[i])

        if ("min" in m and "max" in m) and ("standard" not in m):
            # min-max or log-min-max
            if "log" in m:
                a, b = float(st["log_min"]), float(st["log_max"])
                g_phys[i] = 10.0 ** (z * (b - a) + a)
            else:
                a, b = float(st["min"]), float(st["max"])
                g_phys[i] = z * (b - a) + a
        else:
            # standard or log-standard
            if "log" in m:
                a, b = float(st["log_mean"]), float(st["log_std"])
                g_phys[i] = 10.0 ** (z * b + a)
            else:
                a, b = float(st["mean"]), float(st["std"])
                g_phys[i] = z * b + a

    return g_phys


# =============================================================================
# Model + inference
# =============================================================================
def _step_model(model, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = model(y, dt, g)
    return out[:, 0, :] if out.ndim == 3 else out


@torch.inference_mode()
def rollout_phys(
    model,
    y0_phys: np.ndarray,
    g_phys: np.ndarray,
    dt_sec: float,
    n_steps: int,
) -> np.ndarray:
    """
    Autoregressive rollout with constant dt (seconds).
    Returns [n_steps, S] in physical space.
    """
    y = torch.from_numpy(y0_phys.astype(np.float32)).to(device=DEVICE, dtype=DTYPE).unsqueeze(0)  # [1,S]
    g = torch.from_numpy(g_phys.astype(np.float32)).to(device=DEVICE, dtype=DTYPE).unsqueeze(0)  # [1,G]
    dt = torch.tensor([float(dt_sec)], device=DEVICE, dtype=DTYPE)  # [1]

    ys = []
    for _ in range(int(n_steps)):
        y = _step_model(model, y, dt, g)  # [1,S]
        ys.append(y[0])
    return torch.stack(ys, dim=0).cpu().numpy().astype(np.float64)


# =============================================================================
# Plotting / metrics
# =============================================================================
def _distinct_colors(n: int):
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    cols = []
    for name in cmaps:
        cmap = plt.get_cmap(name)
        cols.extend(list(getattr(cmap, "colors", [])))
    if n > len(cols):
        hs = np.linspace(0, 1, n, endpoint=False)
        cols = [tuple(mcolors.hsv_to_rgb((h, 0.85, 0.95))) for h in hs]
    return [tuple(mcolors.to_rgb(c)) for c in cols[:n]]


def plot_results(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, species_keys: list[str], out_path: Path) -> None:
    labels = [s.replace("_evolution", "") for s in species_keys]
    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = _distinct_colors(len(order))

    me = int(CFG.pred_marker_every)
    marker_idx = np.arange(0, len(t), me)

    fig, ax = plt.subplots(figsize=(7, 7))

    for r, i in enumerate(order):
        c = colors[r]
        ax.plot(t, y_true[:, i], linestyle="-", lw=CFG.true_lw, alpha=CFG.true_alpha, color=c)
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
        title=f"Autoregressive (constant dt): {len(t)} steps (PHYS model, CPU export)",
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
    eps = 1e-30
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)

    rel_err = np.abs(yp - yt) / (np.abs(yt) + 1e-12)
    mae = np.mean(np.abs(yp - yt))
    log_err = np.mean(np.abs(np.log10(yp) - np.log10(yt)))

    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if not EXPORT_PATH.exists():
        raise FileNotFoundError(f"Missing export: {EXPORT_PATH}")

    export_path = EXPORT_PATH
    model = torch.export.load(export_path).module().to(device=DEVICE)
    #model.eval()
    print(f"[model] {export_path.name}  device={DEVICE} dtype=torch.float32")

    y_z, g_z, dt_vec, species_keys, gvars, manifest = load_test_trajectory(PROCESSED_DIR, CFG.sample_idx)

    dt0_norm = float(dt_vec[CFG.start_index])
    dt_sec = dt_to_seconds(dt0_norm, manifest)

    y0_z = y_z[CFG.start_index]
    y_true_z = y_z[CFG.start_index + 1 : CFG.start_index + 1 + CFG.n_steps]

    y0_phys = denormalize_species(y0_z, species_keys, manifest)
    y_true_phys = denormalize_species(y_true_z, species_keys, manifest)

    g_phys = denormalize_globals(g_z, gvars, manifest)

    y_pred_phys = rollout_phys(model, y0_phys, g_phys, dt_sec, CFG.n_steps)

    t_eval = dt_sec * np.arange(1, CFG.n_steps + 1, dtype=float)

    out_path = RUN_DIR / "plots" / f"autoregressive_constdt_phys_cpuexport_{CFG.sample_idx}_t{CFG.start_index}.png"
    plot_results(t_eval, y_true_phys, y_pred_phys, species_keys, out_path)
    print_errors(y_true_phys, y_pred_phys)


if __name__ == "__main__":
    main()
