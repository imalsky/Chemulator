#!/usr/bin/env python3
"""
xi_again.py — VULCAN → normalized inputs/globals → constant-dt autoregressive rollout → plot.

Changes:
- Add a global param MAX_ML_PLOT_POINTS to cap how many ML markers are plotted (even if N_STEPS is large).
  (Rollout still runs for N_STEPS; we just downsample the plotted markers uniformly.)
- LEFT PANEL: VULCAN is plotted as a SOLID line obtained by basic interpolation of the VULCAN time series
  onto a dense set of times spanning the left-panel x-range. This guarantees the VULCAN curve is visible
  even when VULCAN has very few samples early on.
- Raw VULCAN points (true samples) are still overlaid as dots (unlabeled).
- Two-column figure:
    * Left: after-anchor (relative time) with VULCAN solid line (+ raw VULCAN dots) and ML markers
    * Right: full VULCAN (absolute time) on a LOG x-axis with a gray dashed vertical line at t0 (ML start)
  The two panels share the same y-axis limits/scale.
- Right panel x-limits are derived from the data itself (min/max of t_all where t_all>0).
- Colors are guaranteed to match between panels via an explicit species->color map.
- Legend is shown on BOTH panels.
"""

from __future__ import annotations

import json, math, pickle
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

EXPORT_NAME = "export_mps_dynB_1step.pt2"

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


def dt_norm(dt_sec: float, manifest: dict) -> float:
    a = float(manifest["dt"]["log_min"])
    b = float(manifest["dt"]["log_max"])
    return (math.log10(float(dt_sec)) - a) / (b - a)


def z_from_species(y_phys: np.ndarray, species_keys: list[str], stats: dict) -> np.ndarray:
    mu = np.array([stats[k]["log_mean"] for k in species_keys], float)
    sd = np.array([stats[k]["log_std"] for k in species_keys], float)
    return ((np.log10(y_phys) - mu) / sd).astype(np.float32)


def species_from_z(y_z: np.ndarray, species_keys: list[str], stats: dict) -> np.ndarray:
    mu = np.array([stats[k]["log_mean"] for k in species_keys], float)
    sd = np.array([stats[k]["log_std"] for k in species_keys], float)
    return (10.0 ** (y_z * sd + mu)).astype(np.float64)


def z_from_globals(T_K: float, barye: float, gvars: list[str], manifest: dict) -> np.ndarray:
    methods = (manifest.get("methods") or manifest.get("meta", {}).get("methods") or {})
    default = str(manifest.get("globals_default_method", "standard"))
    stats = manifest["per_key_stats"]
    z = np.zeros((len(gvars),), np.float32)

    for i, nm in enumerate(gvars):
        x = barye if nm.strip().lower().startswith("p") else T_K
        m = str(methods.get(nm, default)).lower().replace("_", "-")
        st = stats[nm]

        if ("min" in m and "max" in m) and ("standard" not in m):
            if "log" in m:
                x = math.log10(float(x))
                a, b = float(st["log_min"]), float(st["log_max"])
            else:
                a, b = float(st["min"]), float(st["max"])
            z[i] = (float(x) - a) / (b - a)
        else:
            if "log" in m:
                x = math.log10(float(x))
                a, b = float(st["log_mean"]), float(st["log_std"])
            else:
                a, b = float(st["mean"]), float(st["std"])
            z[i] = (float(x) - a) / b

    return z


def _step(model, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = model(y, dt, g)
    return out[:, 0, :] if out.ndim == 3 else out


@torch.inference_mode()
def rollout(model, y0_z: np.ndarray, g_z: np.ndarray, dt0: float, n: int, device: str, dtype: torch.dtype) -> np.ndarray:
    y = torch.from_numpy(y0_z).to(device=device, dtype=dtype).unsqueeze(0)  # [1,S]
    g = torch.from_numpy(g_z).to(device=device, dtype=dtype).unsqueeze(0)   # [1,G]
    dt = torch.tensor([float(dt0)], device=device, dtype=dtype)             # [1]
    ys = []
    for _ in range(n):
        y = _step(model, y, dt, g)
        ys.append(y[0])
    return torch.stack(ys, 0).cpu().numpy()


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=int)
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.round(np.linspace(0, n - 1, int(max_points))).astype(int))


def _interp_vulcan_logtime(t_abs: np.ndarray, y: np.ndarray, tq_abs: np.ndarray) -> np.ndarray:
    """
    Super basic interpolation of y(t) onto tq_abs using linear interpolation in log10(t).
    Extrapolation is "flat" (clamps to endpoints).
    Requires positive times.
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

    # Remove duplicate times (keep first occurrence)
    uniq = np.ones(t.shape[0], dtype=bool)
    uniq[1:] = t[1:] != t[:-1]
    t = t[uniq]
    yy = yy[uniq]
    if t.size == 1:
        return np.full_like(tq_abs, float(yy[0]), dtype=float)

    tq = np.asarray(tq_abs, float)
    tq_clip = np.clip(tq, float(t[0]), float(t[-1]))

    xv = np.log10(t)
    xq = np.log10(tq_clip)
    return np.interp(xq, xv, yy).astype(float)


def main():
    plt.style.use("science.mplstyle")

    manifest = json.loads((PROCESSED_DIR / "normalization.json").read_text())
    stats = manifest["per_key_stats"]
    species_keys = list(manifest["species_variables"])
    bases = [k[:-10] if k.endswith("_evolution") else k for k in species_keys]
    gvars = list(manifest.get("global_variables") or manifest.get("meta", {}).get("global_variables") or [])

    pt2 = (RUN_DIR / EXPORT_NAME).resolve()
    nn = pt2.name.lower()
    device = (
        "cuda"
        if ("cuda" in nn or "gpu" in nn) and torch.cuda.is_available()
        else ("mps" if "mps" in nn and torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = torch.export.load(pt2).module().to(device=device)
    print(f"[model] {pt2.name}  device={device}")

    t_all, MR_all, vidx, T_K, barye = load_vulcan(PROFILE)

    idx0 = int(np.argmin(np.abs(t_all - float(START_T_SEC))))
    t0 = float(t_all[idx0])

    # For log-x panel and log-time interpolation, use a positive anchor for those operations
    t_pos = t_all[t_all > 0.0]
    t0_pos = t0 if t0 > 0.0 else float(t_pos.min())

    y0 = np.array([MR_all[idx0, vidx[b]] for b in bases], float)
    y0 = np.clip(y0, 1e-30, None)
    y0 = y0 / y0.sum()
    y0_z = z_from_species(y0, species_keys, stats)

    g_z = z_from_globals(T_K, barye, gvars, manifest) if gvars else np.zeros((0,), np.float32)

    dt0 = dt_norm(DT_SEC, manifest)
    y_pred_z = rollout(model, y0_z, g_z, dt0, int(N_STEPS), device=device, dtype=dtype)
    y_pred = species_from_z(y_pred_z, species_keys, stats)
    y_pred = y_pred / np.maximum(y_pred.sum(1, keepdims=True), 1e-30)

    pred_rel = float(DT_SEC) * np.arange(1, int(N_STEPS) + 1, dtype=float)

    # Downsample ML markers for plotting
    ml_idx = _downsample_indices(int(N_STEPS), int(MAX_ML_PLOT_POINTS))
    pred_rel_plot = pred_rel[ml_idx] if pred_rel.size else pred_rel
    y_pred_plot = y_pred[ml_idx] if y_pred.shape[0] else y_pred

    # Left-panel x extent
    x_left_max = float(pred_rel.max()) if pred_rel.size else 0.0

    # Dense x grid for interpolated VULCAN solid line (relative time)
    n_line = int(VULCAN_LEFT_LINE_POINTS)
    n_line = max(n_line, 2)
    t_rel_line = np.linspace(0.0, x_left_max, n_line)
    t_abs_line = t0 + t_rel_line
    # Use positive times for log-time interpolation
    t_abs_line_for_interp = np.maximum(t_abs_line, t_pos.min())

    # Raw VULCAN points to overlay on left: at least N_FUTURE_PTS after anchor, and within left x-range if possible
    hi_min = min(len(t_all), idx0 + 1 + int(N_FUTURE_PTS))
    t_abs_minseg = t_all[idx0:hi_min]
    MR_minseg = MR_all[idx0:hi_min]
    # Also include any points in [t0, t0 + x_left_max] (could be empty if sampling is sparse)
    m_range = (t_all >= t0) & (t_all <= (t0 + x_left_max))
    t_abs_range = t_all[m_range]
    MR_range = MR_all[m_range]
    # Merge for plotting dots
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

    for sp in PLOT_SPECIES:
        c = sp_color[sp]
        jv = vidx[sp]
        jm = bases.index(sp)

        # LEFT: VULCAN SOLID line (interpolated from VULCAN; labeled for legend)
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

        # LEFT: raw VULCAN samples as dots (unlabeled)
        if PLOT_VULCAN_RAW_DOTS_LEFT and t_rel_dots.size:
            ax0.plot(
                t_rel_dots,
                np.clip(MR_dots[:, jv], YMIN, None),
                ".",
                ms=4.0,
                alpha=0.9,
                color=c,
            )

        # LEFT: ML markers (no label) — downsampled
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

        # RIGHT: full VULCAN (label line; dots unlabeled)
        ax1.plot(t_all, np.clip(MR_all[:, jv], YMIN, None), "-", lw=1.4, alpha=0.85, color=c, label=sp)
        ax1.plot(t_all, np.clip(MR_all[:, jv], YMIN, None), ".", ms=2.2, alpha=0.7, color=c)

    ax0.legend(loc="best", fontsize=9)
    ax1.legend(loc="best", fontsize=9)

    out = OUT_DIR / f"xi_again_2col_logx_{PROFILE}_t0_{t0:.3e}_dt_{DT_SEC:.0e}_N{int(N_STEPS)}_after_anchor.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
