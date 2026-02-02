#!/usr/bin/env python3
"""
xi_again.py — VULCAN → PHYSICAL inputs/globals → constant-dt autoregressive rollout → plot (PHYSICAL-space model).

This version uses the baked physical-space export:
  - Inputs: y_phys (species), g_phys (P,T), dt_sec (seconds)
  - Output: y_phys next-step

We still read normalization.json ONLY to get the canonical ordering of:
  - species_variables (model input/output channel order)
  - global_variables   (model globals channel order)

Plot behavior matches your current two-panel layout:
  - Left: after-anchor (relative time) with VULCAN solid interpolated line + raw dots + ML markers
  - Right: full VULCAN on log-x with anchor line
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------- GLOBALS -----------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RUN_DIR = ROOT / "models" / "v1"
OUT_DIR = RUN_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VULCAN_DIR = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar")
PROFILE = "2000K_1bar"

START_T_SEC = 1e0   # requested anchor (snapped to nearest VULCAN sample)
DT_SEC = 1e2         # constant dt (seconds)
N_STEPS = 1000       # ML steps (rollout length)

# Plot control: cap number of ML markers plotted (uniformly downsampled)
MAX_ML_PLOT_POINTS = 10

# How many points to use for the interpolated VULCAN solid line in the left panel
VULCAN_LEFT_LINE_POINTS = 2000

# If you still want to overlay raw VULCAN samples on the left, keep this True
PLOT_VULCAN_RAW_DOTS_LEFT = True

# Minimum number of raw VULCAN samples after anchor to include on the left (for the dots overlay)
N_FUTURE_PTS = 5

# IMPORTANT: on Apple MPS, prefer the CPU-exported phys model (fp32 graph) and run it on MPS.
EXPORT_NAME = "export_cpu_dynB_1step_phys.pt2"

PLOT_SPECIES = ["H2O", "CH4", "CO", "CO2", "NH3", "HCN", "N2", "C2H2"]
YMIN, YMAX = 1e-30, 2.0
# -------------------------------------------------------------------

PROFILES = {
    "2000K_1bar":  {"T_K": 2000.0, "P_bar": 1.0},
    "2000K_1mbar": {"T_K": 2000.0, "P_bar": 1e-3},
    "1000K_1bar":  {"T_K": 1000.0, "P_bar": 1.0},
    "1000K_1mbar": {"T_K": 1000.0, "P_bar": 1e-3},
}


def vulcan_path(profile: str) -> Path:
    T = int(PROFILES[profile]["T_K"])
    barye = float(PROFILES[profile]["P_bar"]) * 1e6
    return VULCAN_DIR / f"vul-T{T}KlogP{math.log10(barye):.1f}-NCHO-solar_hot_ini.vul"


def load_vulcan(profile: str):
    with open(vulcan_path(profile), "rb") as f:
        d = pickle.load(f)
    t = np.asarray(d["variable"]["t_time"], float)
    Y = np.asarray(d["variable"]["y_time"], float)  # [T, layer, S]
    names = list(d["variable"]["species"])
    idx = {n: i for i, n in enumerate(names)}
    MR = Y[:, 0, :] / np.maximum(Y[:, 0, :].sum(-1), 1e-30)[:, None]
    T_K = float(PROFILES[profile]["T_K"])
    barye = float(PROFILES[profile]["P_bar"]) * 1e6
    return t, MR.astype(np.float64), idx, T_K, barye


def _step(model, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = model(y, dt, g)
    return out[:, 0, :] if out.ndim == 3 else out


@torch.inference_mode()
def rollout_phys(model, y0_phys: np.ndarray, g_phys: np.ndarray, dt_sec: float, n: int, device: str) -> np.ndarray:
    """
    Physical-space rollout. Inputs/outputs are physical abundances.
    """
    y = torch.from_numpy(y0_phys.astype(np.float32)).to(device=device).unsqueeze(0)  # [1,S]
    g = torch.from_numpy(g_phys.astype(np.float32)).to(device=device).unsqueeze(0)   # [1,G]
    dt = torch.tensor([float(dt_sec)], device=device, dtype=torch.float32)           # [1]

    ys = []
    for _ in range(int(n)):
        y = _step(model, y, dt, g)
        ys.append(y[0])
    return torch.stack(ys, 0).cpu().numpy().astype(np.float64)


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=int)
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, int(max_points))).astype(int))


def _interp_vulcan_logtime(t_abs: np.ndarray, y: np.ndarray, tq_abs: np.ndarray) -> np.ndarray:
    """
    Super basic interpolation of y(t) onto tq_abs using linear interpolation in log10(t).
    Extrapolation clamps to endpoints. Requires positive times.
    """
    m = t_abs > 0.0
    t = t_abs[m]
    yy = y[m]
    if t.size == 0:
        return np.full_like(tq_abs, np.nan, dtype=float)
    if t.size == 1:
        return np.full_like(tq_abs, float(yy[0]), dtype=float)

    order = np.argsort(t)
    t = t[order]
    yy = yy[order]

    uniq = np.ones(t.shape[0], dtype=bool)
    uniq[1:] = t[1:] != t[:-1]
    t = t[uniq]
    yy = yy[uniq]
    if t.size == 1:
        return np.full_like(tq_abs, float(yy[0]), dtype=float)

    tq = np.asarray(tq_abs, float)
    tq_clip = np.clip(tq, float(t[0]), float(t[-1]))
    return np.interp(np.log10(tq_clip), np.log10(t), yy).astype(float)


def main():
    plt.style.use("science.mplstyle")

    # Canonical channel order comes from the manifest used during training/export.
    manifest = json.loads((PROCESSED_DIR / "normalization.json").read_text())
    species_keys = list(manifest["species_variables"])
    bases = [k[:-10] if k.endswith("_evolution") else k for k in species_keys]
    gvars = list(manifest.get("global_variables") or manifest.get("meta", {}).get("global_variables") or [])

    pt2 = (RUN_DIR / EXPORT_NAME).resolve()

    # Choose runtime device (run on best available).
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = torch.export.load(pt2).module().to(device=device)
    print(f"[model] {pt2.name}  device={device} dtype=torch.float32")

    t_all, MR_all, vidx, T_K, barye = load_vulcan(PROFILE)

    idx0 = int(np.argmin(np.abs(t_all - float(START_T_SEC))))
    t0 = float(t_all[idx0])

    # For log-x panel and log-time interpolation, use a positive anchor for those operations
    t_pos = t_all[t_all > 0.0]
    t0_pos = t0 if t0 > 0.0 else float(t_pos.min())

    # Build y0 in canonical model order (species_keys / bases).
    y0 = np.array([MR_all[idx0, vidx[b]] for b in bases], float)
    y0 = np.clip(y0, 1e-30, None)
    y0 = y0 / y0.sum()
    y0_phys = y0.astype(np.float64)

    # Build g in canonical order (gvars).
    g_phys = np.zeros((len(gvars),), dtype=np.float64)
    for i, nm in enumerate(gvars):
        g_phys[i] = barye if nm.strip().lower().startswith("p") else T_K

    # Roll out in physical space.
    y_pred = rollout_phys(model, y0_phys, g_phys, float(DT_SEC), int(N_STEPS), device=device)

    # Optional: keep predictions normalized as mixing ratios.
    y_pred = np.clip(y_pred, 1e-30, None)
    y_pred = y_pred / np.maximum(y_pred.sum(1, keepdims=True), 1e-30)

    pred_rel = float(DT_SEC) * np.arange(1, int(N_STEPS) + 1, dtype=float)

    # Downsample ML markers for plotting
    ml_idx = _downsample_indices(int(N_STEPS), int(MAX_ML_PLOT_POINTS))
    pred_rel_plot = pred_rel[ml_idx] if pred_rel.size else pred_rel
    y_pred_plot = y_pred[ml_idx] if y_pred.shape[0] else y_pred

    # Left-panel x extent
    x_left_max = float(pred_rel.max()) if pred_rel.size else 0.0

    # Dense x grid for interpolated VULCAN solid line (relative time)
    n_line = max(int(VULCAN_LEFT_LINE_POINTS), 2)
    t_rel_line = np.linspace(0.0, x_left_max, n_line)
    t_abs_line = t0 + t_rel_line
    t_abs_line_for_interp = np.maximum(t_abs_line, float(t_pos.min()))

    # Raw VULCAN points to overlay on left
    hi_min = min(len(t_all), idx0 + 1 + int(N_FUTURE_PTS))
    t_abs_minseg = t_all[idx0:hi_min]
    MR_minseg = MR_all[idx0:hi_min]
    m_range = (t_all >= t0) & (t_all <= (t0 + x_left_max))
    t_abs_range = t_all[m_range]
    MR_range = MR_all[m_range]
    if t_abs_range.size:
        t_abs_dots = np.concatenate([t_abs_minseg, t_abs_range], axis=0)
        MR_dots = np.concatenate([MR_minseg, MR_range], axis=0)
    else:
        t_abs_dots = t_abs_minseg
        MR_dots = MR_minseg
    t_rel_dots = t_abs_dots - t0

    # ----------------------------- COLORS (shared map) -----------------------------
    color_arr = plt.cm.tab20(np.linspace(0, 0.95, len(PLOT_SPECIES)))
    sp_color = {sp: color_arr[i] for i, sp in enumerate(PLOT_SPECIES)}

    # ----------------------------- PLOT (2 columns) -----------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax in (ax0, ax1):
        ax.set_yscale("log")
        ax.set_ylim(YMIN, YMAX)
        ax.set_ylabel("Relative Abundance")

    # Left: after-anchor view (relative time)
    ax0.set_xlim(0.0, x_left_max)
    ax0.set_xlabel("Time after anchor t0 (s)")
    ax0.set_title("After anchor (VULCAN + ML)")

    # Right: full VULCAN extent (absolute time) on LOG x-axis + anchor marker
    ax1.set_xscale("log")
    ax1.set_xlim(float(t_pos.min()), float(t_pos.max()))
    ax1.set_xlabel("Absolute time (s)")
    ax1.set_title("Full VULCAN (log x)")
    ax1.axvline(t0_pos, color="0.7", ls="--", lw=1.2, alpha=0.9)

    fig.suptitle(f"{PROFILE}  snapped t0={t0:.3e}s (idx {idx0})", y=1.02)

    # Map species name -> model channel index via bases list.
    for sp in PLOT_SPECIES:
        c = sp_color[sp]
        jv = vidx[sp]
        jm = bases.index(sp)

        # LEFT: VULCAN SOLID line (interpolated; labeled)
        y_line = _interp_vulcan_logtime(t_all, MR_all[:, jv], t_abs_line_for_interp)
        ax0.plot(
            t_rel_line,
            np.clip(y_line, YMIN, None),
            "-",
            lw=1.6,
            alpha=0.85,
            color=c,
            label=sp,
        )

        # LEFT: raw VULCAN dots (unlabeled)
        if PLOT_VULCAN_RAW_DOTS_LEFT and t_rel_dots.size:
            ax0.plot(
                t_rel_dots,
                np.clip(MR_dots[:, jv], YMIN, None),
                ".",
                ms=4.0,
                alpha=0.9,
                color=c,
            )

        # LEFT: ML markers (downsampled; unlabeled)
        ax0.plot(
            pred_rel_plot,
            np.clip(y_pred_plot[:, jm], YMIN, None),
            "o",
            ms=4.2,
            mfc="none",
            mec=c,
            mew=1.0,
            alpha=0.95,
        )

        # RIGHT: full VULCAN (line + dots; labeled on the line)
        ax1.plot(t_all, np.clip(MR_all[:, jv], YMIN, None), "-", lw=1.4, alpha=0.85, color=c, label=sp)
        ax1.plot(t_all, np.clip(MR_all[:, jv], YMIN, None), ".", ms=2.2, alpha=0.7, color=c)

    ax0.legend(loc="best", fontsize=9)
    ax1.legend(loc="best", fontsize=9)

    out = OUT_DIR / f"xi_again_2col_logx_phys_{PROFILE}_t0_{t0:.3e}_dt_{DT_SEC:.0e}_N{int(N_STEPS)}_after_anchor.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
