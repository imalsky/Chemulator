#!/usr/bin/env python3
"""
Simplified Flow-map AE prediction vs ground truth (no simplex ops).

- Solid lines  = ground truth from dataset
- Dashed lines = model predictions (panel 1 only)
- Panel 2 shows ONLY square markers (no dashed/solid prediction line), colored per-species.

Two panels:
1) One jump: y(t0) -> y(t0 + Δt) in a single model call (vectorized across query times)
2) TRUE long autoregressive rollout using ONE global schedule, plotting ONLY the jump endpoints:
   - logspace mode: ~N log-spaced cumulative-time jumps from [t0, t_max] (may merge small dt)
   - constant mode: uses EXACT dt = ROLLOUT_CONST_DT_SEC (if provided), runs up to ROLLOUT_MAX_STEPS,
                    and then STOPS (i.e., does NOT try to reach the panel-1 t_max if it would take more steps)

Key fix:
- Slice shard y/g columns to match *config.json* (species_variables / target_species / global_variables)
  instead of blindly using all columns from normalization.json.

dt override:
- Uses DT_MIN_PHYS_OVERRIDE as the minimum physical dt for:
  (a) the rollout schedule lower bound (logspace mode), and
  (b) the dt normalization clamp min (panel 2)

Hard safety:
- This script raises immediately if any dt <= 0 is ever constructed or encountered.

Plot downsampling (panel 2):
- The rollout can compute ~1e4+ steps, but we downsample ONLY for plotting so you don’t get 10k dots.
- Downsampling is ~uniform in log10(t - t0), so the visual density per decade stays roughly constant.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("science.mplstyle")

# ---------------- Paths & settings ----------------
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models" / "big_mlp"
#MODEL_DIR = REPO / "models" / "big_big_big"
EP_FILENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper

SAMPLE_IDX = 6
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# ---------------- Panel 2 rollout schedule options ----------------
# Choose rollout schedule:
#   "logspace"  -> log-spaced cumulative time grid (existing behavior)
#   "constant"  -> constant dt steps with a maximum number of autoregressive steps (and STOP)
ROLLOUT_SCHEDULE_MODE = "constant"  # "logspace" or "constant"

# Logspace mode: target number of jumps total (can be reduced by min-dt merging)
ROLLOUT_N_JUMPS_TOTAL = 50

# Constant mode: maximum number of autoregressive steps
ROLLOUT_MAX_STEPS = ROLLOUT_N_JUMPS_TOTAL

# Constant mode: EXACT constant dt in seconds (if set). In constant mode, we run exactly
# min(ROLLOUT_MAX_STEPS, steps implied by the chosen horizon), but the horizon itself is defined by:
#   t_end = t0 + ROLLOUT_CONST_DT_SEC * ROLLOUT_MAX_STEPS
# so dt is never inflated; the rollout simply stops after max steps.
ROLLOUT_CONST_DT_SEC: Optional[float] = 1e3

# How to sample "truth at rollout times" for panel-2 error reporting:
#   "interp"  -> linear interpolation in time (uses your exact dt grid)
#   "nearest" -> nearest neighbor (old behavior)
PANEL2_TRUTH_SAMPLING = "interp"

# Physical min dt used for panel-2 dt normalization clamp min, and the logspace schedule start
DT_MIN_PHYS_OVERRIDE = 1e-3  # <-- fixed per your comment request

# Squares for panel-2 prediction points
P2_MARKER = "s"
P2_MS = 6

# Panel-2 plotting downsample:
# Plot at most this many rollout endpoints (still compute all steps and compute errors on all steps).
P2_MAX_PLOT_POINTS = 30

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
    missing = [n for n in chosen if n not in m]
    if missing:
        raise KeyError(f"Names not found in metadata: {missing}")
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
    """Prepare normalized (y, dt, g) for CPU K=1 export (vectorized over K query times)."""
    M = len(t_phys)
    if M < 2:
        raise ValueError(f"Need at least 2 time points, got {M}")

    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)

    t_sel = t_phys[q_idx].astype(np.float32, copy=False)
    dt_sec = (t_sel - np.float32(t_phys[0])).astype(np.float32, copy=False)

    # Hard crash on any non-positive dt (your request).
    if np.any(dt_sec <= 0.0):
        bad = np.where(dt_sec <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt encountered in panel-1 query batch.\n"
            f"t0={float(t_phys[0]):.6e}, example bad indices (within q batch)={bad.tolist()}, "
            f"dt values={dt_sec[bad].tolist()}"
        )

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


def _dt_norm_local(dt_sec: float, norm: NormalizationHelper) -> torch.Tensor:
    """
    Local dt normalization matching NormalizationHelper.normalize_dt_from_phys math, but using
    DT_MIN_PHYS_OVERRIDE as the physical clamp min.
    """
    dt_sec_f = float(dt_sec)
    if dt_sec_f <= 0.0:
        raise RuntimeError(f"Non-positive dt passed to dt normalizer: dt={dt_sec_f}")

    log_min = float(norm.dt_spec["log_min"])
    log_max = float(norm.dt_spec["log_max"])
    range_log = max(log_max - log_min, 1e-12)

    phys_min = float(max(norm.epsilon, DT_MIN_PHYS_OVERRIDE))
    phys_max = float(norm.dt_max_phys)

    dt_t = torch.tensor([dt_sec_f], dtype=torch.float32)
    dt_t = dt_t.clamp(min=phys_min, max=phys_max)
    dt_log = torch.log10(dt_t)

    dt_norm = (dt_log - log_min) / range_log
    dt_norm = dt_norm.clamp(0.0, 1.0).view(1, 1)
    return dt_norm


def _enforce_min_dt(dt: np.ndarray, min_dt: float) -> np.ndarray:
    """
    Merge too-small dt steps into neighbors so every dt >= min_dt (unless total < min_dt).
    Keeps total sum exactly (up to float rounding).
    """
    if dt.size == 0:
        return dt.astype(np.float32)

    dt64 = np.asarray(dt, dtype=np.float64).copy()
    total = float(np.sum(dt64, dtype=np.float64))

    if total <= 0.0:
        raise RuntimeError(f"Non-positive total dt encountered: total={total}")

    if total < float(min_dt):
        # Single step (still positive) smaller than min_dt is allowed by design here.
        # But dt must still be > 0.
        out = np.array([total], dtype=np.float64)
        if out[0] <= 0.0:
            raise RuntimeError(f"Non-positive dt after total<min_dt path: dt={out[0]}")
        return out.astype(np.float32)

    min_dt_f = float(min_dt)

    # Merge until the minimum dt is >= min_dt_f
    while dt64.size > 1 and float(dt64.min()) < min_dt_f:
        i = int(np.argmin(dt64))
        if i == 0:
            dt64[1] = dt64[1] + dt64[0]
            dt64 = dt64[1:]
        else:
            dt64[i - 1] = dt64[i - 1] + dt64[i]
            dt64 = np.delete(dt64, i)

    # Restore exact total by adjusting the last element by the float64 drift.
    if dt64.size:
        drift = total - float(np.sum(dt64, dtype=np.float64))
        dt64[-1] = dt64[-1] + drift

    # Crash if anything non-positive exists.
    if np.any(dt64 <= 0.0):
        bad = np.where(dt64 <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt produced after min-dt enforcement.\n"
            f"min_dt={min_dt_f}, total={total}, bad_idx={bad.tolist()}, bad_dt={dt64[bad].tolist()}"
        )

    return dt64.astype(np.float32)


def semigroup_dt_steps(dt_total: float, n_jumps: int, min_dt: float) -> np.ndarray:
    """
    Build ~n_jumps dt steps that sum to dt_total, using log-spaced *cumulative time* (0 -> dt_total),
    starting at tau_min = min_dt (your override), then enforce dt >= min_dt by merging too-small steps.

    Raises immediately if any dt <= 0 is constructed at any stage.
    """
    dt_total_f = float(dt_total)

    if n_jumps <= 0:
        raise ValueError(f"n_jumps must be >= 1, got {n_jumps}")
    if dt_total_f <= 0.0:
        raise ValueError(f"dt_total must be > 0, got {dt_total_f}")

    if n_jumps == 1:
        # Single positive step
        return np.array([np.float32(dt_total_f)], dtype=np.float32)

    tau_min = float(min_dt)
    if tau_min <= 0.0:
        raise ValueError(f"min_dt must be > 0, got {tau_min}")

    if tau_min >= dt_total_f:
        return np.array([np.float32(dt_total_f)], dtype=np.float32)

    # Use float64 internally to reduce rounding-caused zeros.
    tau = np.logspace(np.log10(tau_min), np.log10(dt_total_f), n_jumps, dtype=np.float64)  # length n_jumps
    tau[-1] = dt_total_f
    tau = np.concatenate([np.array([0.0], dtype=np.float64), tau])  # prepend 0

    # Ensure strictly increasing tau (crash if not).
    if np.any(np.diff(tau) <= 0.0):
        bad = np.where(np.diff(tau) <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-increasing cumulative-time grid produced for rollout schedule.\n"
            f"Example bad indices in diff(tau): {bad.tolist()}"
        )

    dt = np.diff(tau).astype(np.float64)

    if np.any(dt <= 0.0):
        bad = np.where(dt <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt produced when differencing cumulative schedule.\n"
            f"dt_total={dt_total_f}, min_dt={tau_min}, bad_idx={bad.tolist()}, bad_dt={dt[bad].tolist()}"
        )

    # Enforce min dt (merge) and re-check positivity.
    dt = _enforce_min_dt(dt, min_dt=tau_min).astype(np.float32)

    if np.any(dt <= 0.0):
        bad = np.where(dt <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt produced after enforcement (should not happen).\n"
            f"bad_idx={bad.tolist()}, bad_dt={dt[bad].tolist()}"
        )

    # Final sum check (sanity): allow tiny float error.
    s = float(np.sum(dt.astype(np.float64)))
    if not np.isfinite(s) or abs(s - dt_total_f) > 1e-6 * max(1.0, dt_total_f):
        raise RuntimeError(f"Rollout dt sum mismatch: sum(dt)={s} vs dt_total={dt_total_f}")

    return dt


def constant_dt_steps_exact(dt_step: float, n_steps: int) -> np.ndarray:
    """
    EXACT constant dt schedule: dt[k] == dt_step for k=0..n_steps-1.
    Hard crash if dt_step <= 0 or n_steps < 1.
    """
    dt_f = float(dt_step)
    if dt_f <= 0.0:
        raise ValueError(f"dt_step must be > 0, got {dt_f}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    dt = np.full((n_steps,), dt_f, dtype=np.float32)
    if np.any(dt <= 0.0):
        raise RuntimeError("Non-positive dt produced in constant_dt_steps_exact (should not happen).")
    return dt


def build_rollout_dt_steps(
    dt_total: float,
    mode: str,
    min_dt: float,
    n_jumps_total: int,
    max_steps: int,
    const_dt: Optional[float],
) -> np.ndarray:
    """
    Build rollout dt steps using the selected schedule mode.

    - logspace: uses dt_total and n_jumps_total (may merge small dt)
    - constant:
        * if const_dt is set: returns EXACT dt=const_dt for exactly max_steps steps (STOP behavior)
        * if const_dt is None: falls back to a constant schedule that reaches dt_total in <= max_steps
          (keeps older "reach t_max" style for the dt-None case)
    """
    mode_l = str(mode).strip().lower()
    if mode_l not in ("logspace", "constant"):
        raise ValueError(f"Unknown ROLLOUT_SCHEDULE_MODE={mode!r}; expected 'logspace' or 'constant'")

    if mode_l == "logspace":
        return semigroup_dt_steps(dt_total=dt_total, n_jumps=n_jumps_total, min_dt=min_dt)

    # constant
    if max_steps <= 0:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    if const_dt is not None:
        # EXACT dt, run max_steps and stop.
        return constant_dt_steps_exact(dt_step=float(const_dt), n_steps=max_steps)

    # dt not specified: choose constant dt to reach dt_total in <= max_steps (legacy-like)
    dt_total_f = float(dt_total)
    if dt_total_f <= 0.0:
        raise ValueError(f"dt_total must be > 0, got {dt_total_f}")
    dt_step = dt_total_f / float(max_steps)
    if dt_step <= 0.0:
        raise RuntimeError(f"Non-positive derived dt_step in constant mode: dt_step={dt_step}")
    dt = np.full((max_steps,), dt_step, dtype=np.float64)
    drift = dt_total_f - float(np.sum(dt, dtype=np.float64))
    dt[-1] = dt[-1] + drift
    if np.any(dt <= 0.0):
        bad = np.where(dt <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt produced in constant derived schedule.\n"
            f"dt_total={dt_total_f}, max_steps={max_steps}, bad_idx={bad.tolist()}, bad_dt={dt[bad].tolist()}"
        )
    return dt.astype(np.float32)


@torch.inference_mode()
def autoregressive_rollout_scheduled(
    model,
    y0_norm_1x: torch.Tensor,   # [1,S]
    g_norm_1x: torch.Tensor,    # [1,G]
    t0: float,
    t_max: float,
    norm: NormalizationHelper,
    mode: str,
    n_jumps_total: int,
    max_steps: int,
    const_dt: Optional[float],
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    TRUE long autoregressive rollout from t0 to t_max using ONE global schedule.

    Returns:
      t_roll: [N_eff] jump endpoint times
      y_roll_norm: [N_eff,S] states at those times (normalized)

    logspace: N_eff can be < n_jumps_total if min-dt enforcement merges steps.
    constant with const_dt set: N_eff == max_steps, and dt is EXACT const_dt (STOP behavior).

    Crashes if any dt <= 0 is produced or encountered.
    """
    t0_f = float(t0)
    t_max_f = float(t_max)
    if t_max_f <= t0_f:
        raise ValueError(f"Need t_max > t0 for rollout, got t0={t0_f}, t_max={t_max_f}")

    dt_total = t_max_f - t0_f
    if dt_total <= 0.0:
        raise ValueError(f"Non-positive rollout span: dt_total={dt_total}")

    min_dt = float(DT_MIN_PHYS_OVERRIDE)

    dt_steps = build_rollout_dt_steps(
        dt_total=dt_total,
        mode=mode,
        min_dt=min_dt,
        n_jumps_total=n_jumps_total,
        max_steps=max_steps,
        const_dt=const_dt,
    )

    # Hard crash if any dt <= 0.
    if np.any(dt_steps <= 0.0):
        bad = np.where(dt_steps <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive dt found in rollout step list (should not happen).\n"
            f"bad_idx={bad.tolist()}, bad_dt={dt_steps[bad].tolist()}"
        )

    t_roll = (t0_f + np.cumsum(dt_steps.astype(np.float64))).astype(np.float32)

    y_state = y0_norm_1x.contiguous().float().clone()  # [1,S]
    g_norm_1x = g_norm_1x.contiguous().float()

    outs: List[torch.Tensor] = []
    for k, dt_sec in enumerate(dt_steps):
        dt_sec_f = float(dt_sec)
        if dt_sec_f <= 0.0:
            raise RuntimeError(f"Non-positive dt at rollout step {k}: dt={dt_sec_f}")

        dt_norm = _dt_norm_local(dt_sec_f, norm)  # [1,1]
        y_state = model(y_state, dt_norm, g_norm_1x)[:, 0, :]  # [1,S]
        outs.append(y_state[0])

    y_roll_norm = torch.stack(outs, dim=0) if outs else torch.empty((0, y0_norm_1x.shape[1]), dtype=torch.float32)

    if t_roll.shape[0] != y_roll_norm.shape[0]:
        raise RuntimeError(
            "Internal mismatch: t_roll and y_roll_norm must have identical length.\n"
            f"len(t_roll)={t_roll.shape[0]} vs len(y_roll_norm)={y_roll_norm.shape[0]}"
        )

    return t_roll, y_roll_norm


# Backwards-compatible wrapper (kept): logspace schedule with ~N jumps total
@torch.inference_mode()
def autoregressive_rollout_fixed_N(
    model,
    y0_norm_1x: torch.Tensor,   # [1,S]
    g_norm_1x: torch.Tensor,    # [1,G]
    t0: float,
    t_max: float,
    norm: NormalizationHelper,
    n_jumps_total: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    return autoregressive_rollout_scheduled(
        model=model,
        y0_norm_1x=y0_norm_1x,
        g_norm_1x=g_norm_1x,
        t0=t0,
        t_max=t_max,
        norm=norm,
        mode="logspace",
        n_jumps_total=n_jumps_total,
        max_steps=n_jumps_total,
        const_dt=None,
    )


def nearest_indices(t_grid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Nearest-neighbor indices in t_grid for each t_query (both 1D)."""
    t_grid = np.asarray(t_grid, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    idx = np.searchsorted(t_grid, t_query, side="left")
    idx = np.clip(idx, 0, len(t_grid) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_grid) - 1)

    d0 = np.abs(t_query - t_grid[idx0])
    d1 = np.abs(t_query - t_grid[idx])
    out = np.where(d1 < d0, idx, idx0)
    return out.astype(int)


def truth_at_times_interp(t_grid: np.ndarray, y_grid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of y_grid(t) onto t_query for error reporting.
    Returns NaN for out-of-bounds queries.

    Hard safety: requires strictly increasing t_grid.
    """
    t = np.asarray(t_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64)

    if t.ndim != 1 or tq.ndim != 1:
        raise ValueError("truth_at_times_interp expects 1D time arrays.")
    if y.ndim != 2 or y.shape[0] != t.shape[0]:
        raise ValueError(f"truth_at_times_interp expects y_grid shape [T,S] with T=len(t_grid); got {y.shape}")

    if np.any(np.diff(t) <= 0.0):
        bad = np.where(np.diff(t) <= 0.0)[0][:10]
        raise RuntimeError(f"Non-increasing t_grid; cannot interpolate. Example bad diff indices: {bad.tolist()}")

    S = y.shape[1]
    out = np.empty((tq.shape[0], S), dtype=np.float64)

    # mask in-bounds; out-of-bounds -> NaN
    inb = (tq >= t[0]) & (tq <= t[-1])
    out[:] = np.nan

    if np.any(inb):
        tq_in = tq[inb]
        for s in range(S):
            out[inb, s] = np.interp(tq_in, t, y[:, s])
    return out.astype(np.float32)


def downsample_for_logtime_plot(
    t: np.ndarray,
    y: np.ndarray,
    n_plot: int,
    t0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample rollout endpoints ONLY for plotting so marker count stays reasonable.

    Picks indices so points are ~uniform in log10(t - t0), which keeps visual density roughly
    constant per decade on a log-x plot.

    Returns:
      t_ds, y_ds, idx_ds
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if t.ndim != 1:
        raise ValueError("downsample_for_logtime_plot expects t to be 1D.")
    if y.ndim != 2 or y.shape[0] != t.shape[0]:
        raise ValueError(f"y must be [N,S] with N=len(t); got y={y.shape}, t={t.shape}")
    if n_plot <= 0:
        raise ValueError(f"n_plot must be >= 1, got {n_plot}")

    N = t.shape[0]
    if N <= n_plot:
        idx = np.arange(N, dtype=int)
        return t, y, idx

    tau = t.astype(np.float64) - float(t0)
    if np.any(tau <= 0.0):
        bad = np.where(tau <= 0.0)[0][:10]
        raise RuntimeError(
            "Non-positive (t - t0) encountered in downsampling (should not happen if dt>0).\n"
            f"Example bad indices: {bad.tolist()}, tau={tau[bad].tolist()}"
        )

    tau_min = float(tau[0])
    tau_max = float(tau[-1])
    if not np.isfinite(tau_min) or not np.isfinite(tau_max) or tau_max <= tau_min:
        raise RuntimeError(f"Bad tau range for downsampling: tau_min={tau_min}, tau_max={tau_max}")

    tau_targets = np.logspace(np.log10(tau_min), np.log10(tau_max), n_plot, dtype=np.float64)

    idx = np.searchsorted(tau, tau_targets, side="left")
    idx = np.clip(idx, 0, N - 1)

    # Deduplicate while preserving order
    idx = np.unique(idx)

    # Force include endpoints
    if idx[0] != 0:
        idx = np.concatenate(([0], idx))
    if idx[-1] != N - 1:
        idx = np.concatenate((idx, [N - 1]))

    idx = idx.astype(int)
    return t[idx], y[idx], idx


def plot_results_two_panel(
    t_phys: np.ndarray,
    y_true_full: np.ndarray,
    t_pred1: np.ndarray,
    y_pred1: np.ndarray,
    t_pred2: np.ndarray,
    y_pred2: np.ndarray,
    species_out: List[str],
    plot_species: List[str],
    out_path: Path,
    n_jumps_total: int,
) -> None:
    """
    Log–log plot: solid=truth.
    Panel 1: dashed=one-jump prediction across many query times
    Panel 2: squares only (no line) = rollout prediction endpoints (possibly downsampled)
    """
    base_all = [n[:-10] if n.endswith("_evolution") else n for n in species_out]
    keep = [i for i, b in enumerate(base_all) if (not plot_species) or (b in plot_species)]
    labels = [base_all[i] for i in keep]

    y_true_full = y_true_full[:, keep]
    y_pred1 = y_pred1[:, keep]
    y_pred2 = y_pred2[:, keep]

    tiny = 1e-35
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    t_gt = t_phys[m_gt]
    y_gt = np.clip(y_true_full[m_gt], tiny, None)

    m_pr1 = (t_pred1 >= XMIN) & (t_pred1 <= XMAX)
    t_pr1 = t_pred1[m_pr1]
    y_pr1 = np.clip(y_pred1[m_pr1], tiny, None)

    m_pr2 = (t_pred2 >= XMIN) & (t_pred2 <= XMAX)
    t_pr2 = t_pred2[m_pr2]
    y_pr2 = np.clip(y_pred2[m_pr2], tiny, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # Order species by max ground-truth abundance (descending)
    if y_gt.size:
        max_gt = np.max(y_gt, axis=0)  # [N_keep]
        order = np.argsort(max_gt)[::-1]
    else:
        order = np.arange(len(labels))

    # Plasma colors assigned by abundance rank: max -> brightest
    n = len(order)
    if n > 0:
        color_vals = np.linspace(0.15, 0.95, n)  # avoid extremes; nice contrast
        colors = plt.cm.plasma(color_vals[::-1])  # max abundance gets ~0.95 (bright)
    else:
        colors = np.empty((0, 4))

    # Panel 1: truth + dashed one-jump prediction
    for rank, idx in enumerate(order):
        col = colors[rank]
        ax1.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.3, color=col)
        if t_gt.size:
            ax1.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        ax1.loglog(t_pr1, y_pr1[:, idx], "--", lw=3, alpha=1.0, color=col)

    # Panel 2: truth + rollout prediction as colored squares only (no line)
    for rank, idx in enumerate(order):
        col = colors[rank]
        ax2.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.3, color=col)
        if t_gt.size:
            ax2.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)

        # Prediction points ONLY
        if t_pr2.size:
            ax2.loglog(
                t_pr2,
                y_pr2[:, idx],
                linestyle="None",
                marker=P2_MARKER,
                markersize=P2_MS,
                markerfacecolor=col,
                markeredgecolor=col,
                alpha=1.0,
                zorder=15,
            )

    for ax in (ax1, ax2):
        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Abundance")
        ax.set_box_aspect(1)
        ax.set_ylim(1e-30, 3)

    # Species legend (ordered by max ground-truth abundance)
    species_handles = [Line2D([0], [0], color=colors[r], lw=2.0) for r in range(n)]
    species_labels = [labels[i] for i in order]
    leg1 = ax1.legend(handles=species_handles, labels=species_labels, loc="best", title="Species", ncol=3)
    ax1.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="VULCAN"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="One Shot Predictions"),
        Line2D([0], [0], color="black", lw=0.0, marker=P2_MARKER,
               markersize=P2_MS, label="Autoregressive Prediction"),
    ]
    ax2.legend(handles=style_handles, loc="best")

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

    species_in = list(data_cfg.get("species_variables") or species_all)
    species_out = list(data_cfg.get("target_species") or species_in)
    globals_used = list(data_cfg.get("global_variables") or globals_all)

    if species_in != species_out:
        raise ValueError(
            "Autoregressive rollout requires species_in == species_out so predictions can be fed back in.\n"
            f"Got len(species_in)={len(species_in)} vs len(species_out)={len(species_out)}.\n"
            "Set data.target_species to match data.species_variables for rollout testing."
        )

    idx_in = _indices(species_all, species_in)
    idx_out = _indices(species_all, species_out)
    idx_g = _indices(globals_all, globals_used) if globals_used else []

    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    norm = NormalizationHelper(manifest)

    y_all, g_all, t_phys = load_data(data_dir, SAMPLE_IDX)
    y_in = y_all[:, idx_in]
    y_out = y_all[:, idx_out]
    g = g_all[idx_g] if idx_g else np.empty((0,), dtype=np.float32)

    y0 = y_in[0]

    # ---------------- Panel 1: one-jump (dense) ----------------
    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(
        y0=y0,
        g=g,
        t_phys=t_phys,
        q_count=Q_COUNT,
        norm=norm,
        species_in=species_in,
        globals_used=globals_used,
    )

    y_pred1_norm = run_inference(model, y_batch, dt_batch, g_batch)
    y_pred1 = norm.denormalize(y_pred1_norm, species_out).cpu().numpy()

    # ---------------- Panel 2: long rollout (compute ALL steps; plot downsampled endpoints) ----------------
    y0_norm_1x = norm.normalize(torch.from_numpy(y0[None, :]), species_in).float()
    if globals_used:
        g_norm_1x = norm.normalize(torch.from_numpy(g[None, :]), globals_used).float()
    else:
        g_norm_1x = torch.from_numpy(g[None, :]).float()

    t0 = float(t_phys[0])

    # Default horizon (logspace intent): match panel-1 max query time
    t_max = float(np.max(t_sel)) if t_sel.size else float(t0)
    if t_max <= t0:
        raise RuntimeError(f"Rollout span is non-positive: t0={t0}, t_max={t_max}")

    # Constant mode STOP logic:
    # If dt is specified, define the rollout horizon purely by (dt * max_steps), so dt is EXACT
    # and the rollout just stops after max_steps.
    if ROLLOUT_SCHEDULE_MODE.strip().lower() == "constant" and ROLLOUT_CONST_DT_SEC is not None:
        dt_const = float(ROLLOUT_CONST_DT_SEC)
        if dt_const <= 0.0:
            raise ValueError(f"ROLLOUT_CONST_DT_SEC must be > 0, got {dt_const}")
        if ROLLOUT_MAX_STEPS <= 0:
            raise ValueError(f"ROLLOUT_MAX_STEPS must be >= 1, got {ROLLOUT_MAX_STEPS}")
        t_max = t0 + dt_const * float(ROLLOUT_MAX_STEPS)
        if t_max <= t0:
            raise RuntimeError(f"Constant-mode computed t_max is non-positive: t0={t0}, t_max={t_max}")

    t_roll, y_roll_norm = autoregressive_rollout_scheduled(
        model=model,
        y0_norm_1x=y0_norm_1x,
        g_norm_1x=g_norm_1x,
        t0=t0,
        t_max=t_max,
        norm=norm,
        mode=ROLLOUT_SCHEDULE_MODE,
        n_jumps_total=ROLLOUT_N_JUMPS_TOTAL,
        max_steps=ROLLOUT_MAX_STEPS,
        const_dt=ROLLOUT_CONST_DT_SEC,
    )
    y_roll = norm.denormalize(y_roll_norm, species_out).cpu().numpy()

    # Downsample ONLY for plotting so you can run ~1e4 steps but not draw 1e4 squares.
    if t_roll.size and y_roll.shape[0]:
        t_roll_plot, y_roll_plot, idx_roll_plot = downsample_for_logtime_plot(
            t=t_roll,
            y=y_roll,
            n_plot=int(P2_MAX_PLOT_POINTS),
            t0=t0,
        )
    else:
        t_roll_plot = t_roll
        y_roll_plot = y_roll
        idx_roll_plot = np.empty((0,), dtype=int)

    # Truth at rollout times (for error reporting only) -- uses FULL rollout, not downsampled.
    if t_roll.size:
        mode_truth = str(PANEL2_TRUTH_SAMPLING).strip().lower()
        if mode_truth == "nearest":
            idx_nn = nearest_indices(t_phys, t_roll)
            y_true_roll = y_out[idx_nn, :]
        elif mode_truth == "interp":
            y_true_roll = truth_at_times_interp(t_phys, y_out, t_roll)
        else:
            raise ValueError(f"Unknown PANEL2_TRUTH_SAMPLING={PANEL2_TRUTH_SAMPLING!r}; use 'nearest' or 'interp'")
    else:
        y_true_roll = np.empty((0, y_out.shape[1]), dtype=np.float32)

    out_png = MODEL_DIR / "plots" / f"pred_{SAMPLE_IDX}.png"
    plot_results_two_panel(
        t_phys=t_phys,
        y_true_full=y_out,
        t_pred1=t_sel,
        y_pred1=y_pred1,
        t_pred2=t_roll_plot,   # <- downsampled for plotting
        y_pred2=y_roll_plot,   # <- downsampled for plotting
        species_out=species_out,
        plot_species=PLOT_SPECIES,
        out_path=out_png,
        n_jumps_total=ROLLOUT_N_JUMPS_TOTAL,
    )

    # Errors
    y_true_sel = y_out[q_idx, :]
    rel_err_1 = np.abs(y_pred1 - y_true_sel) / (np.abs(y_true_sel) + 1e-12)

    if y_roll.shape[0] and y_true_roll.shape[0]:
        # If interp mode, ignore out-of-bounds NaNs
        finite = np.isfinite(y_true_roll).all(axis=1) if y_true_roll.size else np.array([], dtype=bool)
        if finite.size and not np.all(finite):
            y_r = y_roll[finite]
            y_t = y_true_roll[finite]
        else:
            y_r = y_roll
            y_t = y_true_roll

        if y_r.shape[0] == 0:
            rel_err_R_mean = float("nan")
            rel_err_R_max = float("nan")
        else:
            rel_err_R = np.abs(y_r - y_t) / (np.abs(y_t) + 1e-12)
            rel_err_R_mean = float(rel_err_R.mean())
            rel_err_R_max = float(rel_err_R.max())

        print(f"ROLLOUT_SCHEDULE_MODE = {ROLLOUT_SCHEDULE_MODE}")
        print(f"PANEL2_TRUTH_SAMPLING = {PANEL2_TRUTH_SAMPLING}")
        print(f"DT_MIN_PHYS_OVERRIDE = {DT_MIN_PHYS_OVERRIDE:.3e} s")
        if ROLLOUT_SCHEDULE_MODE.strip().lower() == "logspace":
            print(f"Logspace requested jumps: {ROLLOUT_N_JUMPS_TOTAL} (can be fewer if min-dt merging happens)")
        else:
            print(f"Constant max steps:      {ROLLOUT_MAX_STEPS}")
            print(f"Constant dt (exact):     {ROLLOUT_CONST_DT_SEC}")
            if t_roll.size:
                print(f"Constant rollout starts: dt1={float(t_roll[0] - t0):.3e} s (should equal const dt)")

        print(f"Panel-2 plot cap points: {int(P2_MAX_PLOT_POINTS)}")
        print(f"One-jump relative error:   mean={rel_err_1.mean():.3e}, max={rel_err_1.max():.3e}")
        print(f"Rollout endpoints rel err: mean={rel_err_R_mean:.3e}, max={rel_err_R_max:.3e}")
        print(f"Rollout computed points: {y_roll.shape[0]}")
        print(f"Rollout plotted points:  {y_roll_plot.shape[0]}")
        if t_roll.size:
            print(f"Rollout last time: t_end={float(t_roll[-1]):.6e} s")
    else:
        print(f"ROLLOUT_SCHEDULE_MODE = {ROLLOUT_SCHEDULE_MODE}")
        print(f"DT_MIN_PHYS_OVERRIDE = {DT_MIN_PHYS_OVERRIDE:.3e} s")
        print(f"Panel-2 plot cap points: {int(P2_MAX_PLOT_POINTS)}")
        print(f"One-jump relative error:   mean={rel_err_1.mean():.3e}, max={rel_err_1.max():.3e}")
        print("Rollout produced no points (check dt_total and schedule constraints).")


if __name__ == "__main__":
    main()
