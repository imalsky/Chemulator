#!/usr/bin/env python3
"""
Flow-map prediction vs ground truth with proper inference-time constraints.

- Solid lines  = ground truth from dataset
- Dashed lines = model predictions (panel 1 only)
- Panel 2 shows square markers for autoregressive rollout endpoints

Two panels:
1) One-shot: y(t0) -> y(t0 + Δt) in a single model call (vectorized across query times)
2) Autoregressive rollout with constraints matching training:
   - Clipping in log10 space to prevent nonphysical values
   - Simplex projection to enforce mass conservation

CRITICAL: The rollout loop applies the same constraints used during training.
Without these, errors compound rapidly after ~50-100 steps.
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

# =============================================================================
# Configuration
# =============================================================================

REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / "models" / "mlp_paper_pf_steps96"
EP_FILENAME = "export_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper

# Sample and plotting settings
SAMPLE_IDX = 1
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8

# Rollout schedule: "logspace" or "constant"
ROLLOUT_SCHEDULE_MODE = "constant"
ROLLOUT_N_JUMPS_TOTAL = 1000
ROLLOUT_MAX_STEPS = 1000
ROLLOUT_CONST_DT_SEC: Optional[float] = 1e3

# Truth sampling for error computation: "interp" or "nearest"
PANEL2_TRUTH_SAMPLING = "interp"

# Minimum physical dt for normalization and schedule
DT_MIN_PHYS_OVERRIDE = 1e-3

# Panel 2 plotting
P2_MARKER = "s"
P2_MS = 5
P2_MAX_PLOT_POINTS = 10

# Inference-time constraints (should match training config)
CLIP_ENABLED = True
CLIP_LOG10_MIN = -30.0
CLIP_LOG10_MAX = 10.0
SIMPLEX_ENABLED = True  # Set True if model.predict_delta_log_phys = True

# Species filter (empty = plot all)
PLOT_SPECIES: List[str] = []


# =============================================================================
# Inference Constraints (must match training)
# =============================================================================

def clamp_z_in_log10_space(
        z: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        log10_min: float = -30.0,
        log10_max: float = 10.0,
) -> torch.Tensor:
    """
    Clamp z-normalized values in log10 space to prevent nonphysical drift.

    This matches AdaptiveStiffLoss.clamp_z_in_log10_space from trainer.py.

    Args:
        z: [B, S] normalized state
        log_mean: [S] per-species log10 mean from normalization
        log_std: [S] per-species log10 std from normalization
        log10_min: lower bound in log10 space
        log10_max: upper bound in log10 space

    Returns:
        z_clamped: [B, S] clamped normalized state
    """
    # z -> log10
    log10_vals = z * log_std + log_mean
    # Clamp
    log10_clamped = torch.clamp(log10_vals, min=log10_min, max=log10_max)
    # log10 -> z
    return (log10_clamped - log_mean) / log_std


def project_z_to_simplex(
        z: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
) -> torch.Tensor:
    """
    Project z-normalized values onto simplex in physical space (sum_i y_i = 1).

    This matches AdaptiveStiffLoss.project_z_to_simplex from trainer.py.
    Operates via: log10(y') = log10(y) - log10(sum_i 10^log10(y_i))

    Args:
        z: [B, S] normalized state
        log_mean: [S] per-species log10 mean from normalization
        log_std: [S] per-species log10 std from normalization

    Returns:
        z_proj: [B, S] projected normalized state
    """
    LN10 = 2.302585092994046

    # z -> log10
    log10_vals = (z * log_std + log_mean).to(torch.float32)

    # log10 -> ln, then logsumexp for normalization
    ln10 = torch.tensor(LN10, dtype=torch.float32, device=z.device)
    ln_raw = log10_vals * ln10
    ln_norm = ln_raw - torch.logsumexp(ln_raw, dim=-1, keepdim=True)

    # ln -> log10 -> z
    log10_proj = ln_norm / ln10
    z_proj = (log10_proj - log_mean) / log_std

    return z_proj.to(dtype=z.dtype)


def apply_inference_constraints(
        z: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        clip_enabled: bool,
        clip_log10_min: float,
        clip_log10_max: float,
        simplex_enabled: bool,
) -> torch.Tensor:
    """
    Apply all inference-time constraints in the same order as training.

    Order matches trainer.py _apply_state_constraints:
    1. Clipping (if enabled)
    2. Simplex projection (if enabled)
    """
    if clip_enabled:
        z = clamp_z_in_log10_space(z, log_mean, log_std, clip_log10_min, clip_log10_max)

    if simplex_enabled:
        z = project_z_to_simplex(z, log_mean, log_std)

    return z


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: Path, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load first test shard and extract one sample trajectory."""
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        n_samples = d["y_mat"].shape[0]
        if sample_idx >= n_samples:
            raise IndexError(f"sample_idx {sample_idx} >= shard size {n_samples}")

        y = d["y_mat"][sample_idx].astype(np.float32)
        g = d["globals"][sample_idx].astype(np.float32)
        t_phys = d["t_vec"]
        if t_phys.ndim > 1:
            t_phys = t_phys[sample_idx]
        t_phys = t_phys.astype(np.float32)

    return y, g, t_phys


def _indices(all_names: List[str], chosen: List[str]) -> List[int]:
    """Get indices of chosen names within all_names."""
    m = {n: i for i, n in enumerate(all_names)}
    missing = [n for n in chosen if n not in m]
    if missing:
        raise KeyError(f"Names not found in metadata: {missing}")
    return [m[n] for n in chosen]


def get_log_stats(norm: NormalizationHelper, species: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract log_mean and log_std tensors for species list."""
    stats = norm.per_key_stats
    log_mean = torch.tensor([stats[s]["log_mean"] for s in species], dtype=torch.float32)
    log_std = torch.tensor([stats[s]["log_std"] for s in species], dtype=torch.float32)
    log_std = torch.clamp(log_std, min=1e-10)  # Match training MIN_LOG_STD
    return log_mean, log_std


# =============================================================================
# Panel 1: One-Shot Prediction
# =============================================================================

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

    t_sel = t_phys[q_idx].astype(np.float32)
    dt_sec = (t_sel - np.float32(t_phys[0])).astype(np.float32)

    if np.any(dt_sec <= 0.0):
        bad = np.where(dt_sec <= 0.0)[0][:10]
        raise RuntimeError(
            f"Non-positive dt in panel-1 batch: t0={t_phys[0]:.6e}, bad_idx={bad.tolist()}"
        )

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


@torch.inference_mode()
def run_inference(model, y_batch: torch.Tensor, dt_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    """Call exported model. CPU K=1 export returns [B,1,S]; we return [B,S]."""
    return model(y_batch, dt_batch, g_batch)[:, 0, :]


# =============================================================================
# Panel 2: Autoregressive Rollout
# =============================================================================

def _dt_norm_local(dt_sec: float, norm: NormalizationHelper) -> torch.Tensor:
    """Normalize a single dt value, using DT_MIN_PHYS_OVERRIDE as clamp min."""
    if dt_sec <= 0.0:
        raise RuntimeError(f"Non-positive dt: {dt_sec}")

    log_min = float(norm.dt_spec["log_min"])
    log_max = float(norm.dt_spec["log_max"])
    range_log = max(log_max - log_min, 1e-12)

    phys_min = max(norm.epsilon, DT_MIN_PHYS_OVERRIDE)
    phys_max = norm.dt_max_phys

    dt_t = torch.tensor([dt_sec], dtype=torch.float32)
    dt_t = dt_t.clamp(min=phys_min, max=phys_max)
    dt_log = torch.log10(dt_t)
    dt_norm = ((dt_log - log_min) / range_log).clamp(0.0, 1.0)

    return dt_norm.view(1, 1)


def build_rollout_schedule(
        dt_total: float,
        mode: str,
        min_dt: float,
        n_jumps: int,
        max_steps: int,
        const_dt: Optional[float],
) -> np.ndarray:
    """Build dt steps for rollout."""
    mode = mode.strip().lower()

    if mode == "constant":
        if const_dt is not None:
            if const_dt <= 0:
                raise ValueError(f"const_dt must be > 0, got {const_dt}")
            return np.full(max_steps, const_dt, dtype=np.float32)
        else:
            dt_step = dt_total / max_steps
            return np.full(max_steps, dt_step, dtype=np.float32)

    elif mode == "logspace":
        if dt_total <= 0:
            raise ValueError(f"dt_total must be > 0, got {dt_total}")
        if n_jumps <= 1:
            return np.array([dt_total], dtype=np.float32)

        tau_min = max(min_dt, 1e-12)
        if tau_min >= dt_total:
            return np.array([dt_total], dtype=np.float32)

        tau = np.logspace(np.log10(tau_min), np.log10(dt_total), n_jumps, dtype=np.float64)
        tau[-1] = dt_total
        tau = np.concatenate([[0.0], tau])
        dt = np.diff(tau)

        # Merge steps smaller than min_dt
        dt = _enforce_min_dt(dt, min_dt)
        return dt.astype(np.float32)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _enforce_min_dt(dt: np.ndarray, min_dt: float) -> np.ndarray:
    """Merge too-small dt steps into neighbors."""
    dt = np.asarray(dt, dtype=np.float64).copy()
    total = dt.sum()

    if total < min_dt:
        return np.array([total], dtype=np.float64)

    while len(dt) > 1 and dt.min() < min_dt:
        i = int(np.argmin(dt))
        if i == 0:
            dt[1] += dt[0]
            dt = dt[1:]
        else:
            dt[i - 1] += dt[i]
            dt = np.delete(dt, i)

    # Fix floating point drift
    dt[-1] += total - dt.sum()
    return dt


@torch.inference_mode()
def autoregressive_rollout(
        model,
        y0_norm: torch.Tensor,
        g_norm: torch.Tensor,
        t0: float,
        dt_steps: np.ndarray,
        norm: NormalizationHelper,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        clip_enabled: bool = True,
        clip_log10_min: float = -30.0,
        clip_log10_max: float = 10.0,
        simplex_enabled: bool = True,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Autoregressive rollout with proper inference-time constraints.

    CRITICAL: Applies the same constraints used during training after each step.
    Without these, errors compound rapidly.

    Args:
        model: Exported model
        y0_norm: [1, S] initial state (normalized)
        g_norm: [1, G] global variables (normalized)
        t0: Initial time
        dt_steps: [N] array of dt values for each step
        norm: NormalizationHelper for dt normalization
        log_mean: [S] per-species log10 mean
        log_std: [S] per-species log10 std
        clip_enabled: Whether to clip in log10 space
        clip_log10_min: Lower bound for clipping
        clip_log10_max: Upper bound for clipping
        simplex_enabled: Whether to project to simplex

    Returns:
        t_roll: [N] times at each step
        y_roll: [N, S] states at each step (normalized)
    """
    if np.any(dt_steps <= 0):
        bad = np.where(dt_steps <= 0)[0][:5]
        raise RuntimeError(f"Non-positive dt at indices {bad.tolist()}")

    # Compute cumulative times
    t_roll = (t0 + np.cumsum(dt_steps.astype(np.float64))).astype(np.float32)

    # Initialize state
    y_state = y0_norm.clone().float()
    g_norm = g_norm.float()

    # Move log stats to same device
    log_mean = log_mean.to(y_state.device)
    log_std = log_std.to(y_state.device)

    outputs = []

    for k, dt_sec in enumerate(dt_steps):
        # Normalize dt
        dt_norm = _dt_norm_local(float(dt_sec), norm)

        # Model prediction
        y_state = model(y_state, dt_norm, g_norm)[:, 0, :]

        # === APPLY CONSTRAINTS (matches training) ===
        y_state = apply_inference_constraints(
            z=y_state,
            log_mean=log_mean,
            log_std=log_std,
            clip_enabled=clip_enabled,
            clip_log10_min=clip_log10_min,
            clip_log10_max=clip_log10_max,
            simplex_enabled=simplex_enabled,
        )

        outputs.append(y_state[0].clone())

    if outputs:
        y_roll = torch.stack(outputs, dim=0)
    else:
        y_roll = torch.empty((0, y0_norm.shape[1]), dtype=torch.float32)

    return t_roll, y_roll


# =============================================================================
# Evaluation Helpers
# =============================================================================

def nearest_indices(t_grid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Find nearest indices in t_grid for each t_query."""
    idx = np.searchsorted(t_grid, t_query, side="left")
    idx = np.clip(idx, 0, len(t_grid) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_grid) - 1)

    d0 = np.abs(t_query - t_grid[idx0])
    d1 = np.abs(t_query - t_grid[idx])

    return np.where(d1 < d0, idx, idx0).astype(int)


def truth_at_times_interp(t_grid: np.ndarray, y_grid: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of y_grid onto t_query times."""
    t = np.asarray(t_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64)

    S = y.shape[1]
    out = np.full((len(tq), S), np.nan, dtype=np.float64)

    inb = (tq >= t[0]) & (tq <= t[-1])
    if np.any(inb):
        for s in range(S):
            out[inb, s] = np.interp(tq[inb], t, y[:, s])

    return out.astype(np.float32)


def downsample_log_uniform(
        t: np.ndarray,
        y: np.ndarray,
        n_plot: int,
        t0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample to ~uniform density in log10(t - t0) for plotting."""
    N = len(t)
    if N <= n_plot:
        return t, y, np.arange(N)

    tau = t.astype(np.float64) - t0
    tau = np.maximum(tau, 1e-30)  # Avoid log(0)

    tau_targets = np.logspace(np.log10(tau[0]), np.log10(tau[-1]), n_plot)
    idx = np.searchsorted(tau, tau_targets, side="left")
    idx = np.clip(idx, 0, N - 1)
    idx = np.unique(idx)

    # Ensure endpoints
    if idx[0] != 0:
        idx = np.concatenate([[0], idx])
    if idx[-1] != N - 1:
        idx = np.concatenate([idx, [N - 1]])

    return t[idx], y[idx], idx


# =============================================================================
# Plotting
# =============================================================================

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
) -> None:
    """Two-panel plot: one-shot (left) and autoregressive rollout (right)."""

    # Filter species
    base_names = [n[:-10] if n.endswith("_evolution") else n for n in species_out]
    keep = [i for i, b in enumerate(base_names) if (not plot_species) or (b in plot_species)]
    labels = [base_names[i] for i in keep]

    y_true_full = y_true_full[:, keep]
    y_pred1 = y_pred1[:, keep]
    y_pred2 = y_pred2[:, keep]

    # Apply plotting bounds
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

    # Color by abundance rank
    n = len(keep)
    if y_gt.size and n > 0:
        max_gt = np.max(y_gt, axis=0)
        order = np.argsort(max_gt)[::-1]
        color_vals = np.linspace(0.15, 0.95, n)
        colors = plt.cm.plasma(color_vals[::-1])
    else:
        order = np.arange(n)
        colors = plt.cm.plasma(np.linspace(0.15, 0.95, max(n, 1)))

    # Panel 1: One-shot
    for rank, idx in enumerate(order):
        col = colors[rank]
        ax1.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.3, color=col)
        if t_gt.size:
            ax1.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        ax1.loglog(t_pr1, y_pr1[:, idx], "--", lw=3, alpha=1.0, color=col)

    # Panel 2: Autoregressive rollout
    for rank, idx in enumerate(order):
        col = colors[rank]
        ax2.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.3, color=col)
        if t_gt.size:
            ax2.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        if t_pr2.size:
            ax2.loglog(t_pr2, y_pr2[:, idx], linestyle="None", marker=P2_MARKER,
                       markersize=P2_MS, color=col, alpha=1.0, zorder=15)

    for ax in (ax1, ax2):
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(1e-30, 3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Abundance")
        ax.set_box_aspect(1)

    # Legends
    species_handles = [Line2D([0], [0], color=colors[r], lw=2.0) for r in range(n)]
    species_labels = [labels[i] for i in order]
    ax1.legend(handles=species_handles, labels=species_labels, loc="best", title="Species", ncol=3)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="Ground Truth"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="One-Shot"),
        Line2D([0], [0], color="black", lw=0, marker=P2_MARKER, ms=P2_MS, label="Autoregressive"),
    ]
    ax2.legend(handles=style_handles, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    os.chdir(REPO)
    seed_everything(42)

    # Load config and metadata
    cfg = load_json(MODEL_DIR / "config.json")
    data_cfg = cfg.get("data", {}) or {}
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    manifest = load_json(data_dir / "normalization.json")
    meta = manifest.get("meta", {}) or {}

    # Resolve species and globals
    species_all = list(meta.get("species_variables", []) or [])
    globals_all = list(meta.get("global_variables", []) or [])

    species_in = list(data_cfg.get("species_variables") or species_all)
    species_out = list(data_cfg.get("target_species") or species_in)
    globals_used = list(data_cfg.get("global_variables") or globals_all)

    if species_in != species_out:
        raise ValueError("Autoregressive rollout requires species_in == species_out")

    idx_in = _indices(species_all, species_in)
    idx_out = _indices(species_all, species_out)
    idx_g = _indices(globals_all, globals_used) if globals_used else []

    # Load model
    print(f"Loading model from {MODEL_DIR / EP_FILENAME}")
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    # Setup normalization
    norm = NormalizationHelper(manifest)
    log_mean, log_std = get_log_stats(norm, species_in)

    # Load data
    y_all, g_all, t_phys = load_data(data_dir, SAMPLE_IDX)
    y_in = y_all[:, idx_in]
    y_out = y_all[:, idx_out]
    g = g_all[idx_g] if idx_g else np.empty((0,), dtype=np.float32)
    y0 = y_in[0]
    t0 = float(t_phys[0])

    print(f"Sample {SAMPLE_IDX}: T={len(t_phys)}, S={len(species_in)}, G={len(globals_used)}")
    print(f"Time range: [{t_phys[0]:.3e}, {t_phys[-1]:.3e}]")
    print(f"Constraints: clip={CLIP_ENABLED} [{CLIP_LOG10_MIN}, {CLIP_LOG10_MAX}], simplex={SIMPLEX_ENABLED}")

    # === Panel 1: One-shot prediction ===
    y_batch, dt_batch, g_batch, q_idx, t_sel = prepare_batch(
        y0=y0, g=g, t_phys=t_phys, q_count=Q_COUNT,
        norm=norm, species_in=species_in, globals_used=globals_used,
    )

    y_pred1_norm = run_inference(model, y_batch, dt_batch, g_batch)
    y_pred1 = norm.denormalize(y_pred1_norm, species_out).cpu().numpy()

    # === Panel 2: Autoregressive rollout ===
    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species_in).float()
    if globals_used:
        g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_used).float()
    else:
        g_norm = torch.from_numpy(g[None, :]).float()

    # Compute rollout horizon
    if ROLLOUT_SCHEDULE_MODE.lower() == "constant" and ROLLOUT_CONST_DT_SEC is not None:
        t_max = t0 + ROLLOUT_CONST_DT_SEC * ROLLOUT_MAX_STEPS
    else:
        t_max = float(t_sel.max()) if t_sel.size else t0 + 1.0

    dt_total = t_max - t0
    print(f"Rollout: mode={ROLLOUT_SCHEDULE_MODE}, t_max={t_max:.3e}, dt_total={dt_total:.3e}")

    # Build schedule
    dt_steps = build_rollout_schedule(
        dt_total=dt_total,
        mode=ROLLOUT_SCHEDULE_MODE,
        min_dt=DT_MIN_PHYS_OVERRIDE,
        n_jumps=ROLLOUT_N_JUMPS_TOTAL,
        max_steps=ROLLOUT_MAX_STEPS,
        const_dt=ROLLOUT_CONST_DT_SEC,
    )
    print(f"Rollout steps: {len(dt_steps)}, dt range: [{dt_steps.min():.3e}, {dt_steps.max():.3e}]")

    # Run rollout with constraints
    t_roll, y_roll_norm = autoregressive_rollout(
        model=model,
        y0_norm=y0_norm,
        g_norm=g_norm,
        t0=t0,
        dt_steps=dt_steps,
        norm=norm,
        log_mean=log_mean,
        log_std=log_std,
        clip_enabled=CLIP_ENABLED,
        clip_log10_min=CLIP_LOG10_MIN,
        clip_log10_max=CLIP_LOG10_MAX,
        simplex_enabled=SIMPLEX_ENABLED,
    )
    y_roll = norm.denormalize(y_roll_norm, species_out).cpu().numpy()

    # Downsample for plotting
    t_roll_plot, y_roll_plot, _ = downsample_log_uniform(t_roll, y_roll, P2_MAX_PLOT_POINTS, t0)
    print(f"Rollout computed: {len(t_roll)} points, plotting: {len(t_roll_plot)} points")

    # === Plot ===
    out_png = MODEL_DIR / "plots" / f"pred_{SAMPLE_IDX}.png"
    plot_results_two_panel(
        t_phys=t_phys,
        y_true_full=y_out,
        t_pred1=t_sel,
        y_pred1=y_pred1,
        t_pred2=t_roll_plot,
        y_pred2=y_roll_plot,
        species_out=species_out,
        plot_species=PLOT_SPECIES,
        out_path=out_png,
    )

    # === Error metrics ===
    y_true_sel = y_out[q_idx, :]
    rel_err_1 = np.abs(y_pred1 - y_true_sel) / (np.abs(y_true_sel) + 1e-12)
    print(f"\nOne-shot error: mean={rel_err_1.mean():.3e}, max={rel_err_1.max():.3e}")

    if t_roll.size:
        if PANEL2_TRUTH_SAMPLING == "nearest":
            idx_nn = nearest_indices(t_phys, t_roll)
            y_true_roll = y_out[idx_nn, :]
        else:
            y_true_roll = truth_at_times_interp(t_phys, y_out, t_roll)

        finite = np.isfinite(y_true_roll).all(axis=1)
        if finite.any():
            rel_err_R = np.abs(y_roll[finite] - y_true_roll[finite]) / (np.abs(y_true_roll[finite]) + 1e-12)
            print(f"Rollout error:  mean={rel_err_R.mean():.3e}, max={rel_err_R.max():.3e}")

        print(f"Rollout final time: {t_roll[-1]:.3e}")


if __name__ == "__main__":
    main()