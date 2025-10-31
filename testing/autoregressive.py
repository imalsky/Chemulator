#!/usr/bin/env python3
"""
Prediction Viewer for Flow-map AE (torch.export .pt2)
====================================================

Plots the ENTIRE ground-truth (GT) profile and overlays predictions in one of 3 modes:

  MODE A: "full_direct"      Direct-from-anchor at each GT timestamp (batched).
  MODE B: "const_direct"     Direct-from-anchor on a constant-Δt grid (batched).
  MODE C: "profile_autoreg"  Autoregressive, step sizes match the GT profile (with optional per-step bounds).

Auto-selects CPU/GPU artifact (and its dtype), keeps tensors on that device, and
prints total time, preds, and time/pred. Colors are per-species and consistent
for GT/anchor/predictions.

NOTE: Normalizer.denormalize uses float64; MPS lacks float64, so we denormalize on CPU.

IMPORTANT SHAPE CONTRACT:
The exported model (gm) expects:
    y:  [B, S_in]
    dt: [B, 1]        <-- 2D, NOT [B,1,1]
    g:  [B, G]
We enforce that here.

Δt bounds:
You can constrain prediction step sizes with DT_MIN_USER / DT_MAX_USER at the top.
If None, dataset/normalization bounds are used.
"""

from __future__ import annotations
import os, sys, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal
from time import perf_counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Allow MPS fallback like in your bench
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# =============================================================================
# Repo discovery / imports
# =============================================================================

def _find_repo(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "src").is_dir() and (p / "models").is_dir():
            return p
    return start.resolve().parent.parent  # fallback

REPO = _find_repo(Path(__file__).resolve())
MODEL_DIR = REPO / "models" / "big"   # <- adjust if needed

# Artifact names
GPU_BASENAME = "export_k1_gpu.pt2"
CPU_BASENAME = "export_k1_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from normalizer import NormalizationHelper
from utils import load_json, seed_everything

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass

# =============================================================================
# ----------------------------- GLOBAL KNOBS ----------------------------------
# =============================================================================

# Trajectory to visualize
SAMPLE_IDX: int = 5

# Prediction mode
PLOT_MODE: Literal["full_direct", "const_direct", "profile_autoreg"] = "profile_autoreg"

# Device preference: "auto" | "mps" | "cuda" | "cpu"
DEVICE_STR: Literal["auto","mps","cuda","cpu"] = "auto"

# Numerical guards
CLIP_MIN_FEED: float = 1e-30   # floor for any physical value fed into model
DT_EPS_NORM: float = 1e-3      # clamp normalized dt away from exact 0/1
Z_CLIP_FOR_FB: Tuple[float, float] | None = None  # e.g., (-8, 8) to clamp latent pre-denorm

# User-overridable per-step Δt bounds (seconds); None => dataset/normalization bounds
DT_MIN_USER: float = None
DT_MAX_USER: float = None

# Mode B grid
CONST_START_DT: float = 0.0
CONST_END_DT: float   = 1.0e4
CONST_STEP_DT: float  = 1.0e-1   # will be clamped into effective [dt_min, dt_max]

# Mode C segment (None => full profile)
AR_START_INDEX: int | None = None
AR_END_INDEX:   int | None = None

# Plot limits (Δt axis is log-log). None => auto
PLOT_XMIN: float | None = 1e-3
PLOT_XMAX: float | None = None
PLOT_YMIN: float | None = None
PLOT_YMAX: float | None = 3

# Warmup forwards for timing stability
WARMUP_STEPS: int = 5

# =============================================================================
# Device & artifact helpers
# =============================================================================

def pick_host_pref_device(pref: str) -> str:
    if pref == "mps":  return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":  return "cpu"
    # auto
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():        return "cuda"
    return "cpu"

def _load_meta(path: Path, default_device: str, default_dtype: str = "float32",
               default_S_in: int = 12, default_G: int = 2) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        m.setdefault("device", default_device)
        m.setdefault("dtype", default_dtype)
        m.setdefault("S_in", default_S_in)
        m.setdefault("G", default_G)
        return m
    return {"device": default_device, "dtype": default_dtype, "S_in": default_S_in, "G": default_G}

def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float16","half","fp16"):  return torch.float16
    if s in ("bfloat16","bf16"):        return torch.bfloat16
    if s in ("float32","fp32","float"): return torch.float32
    raise ValueError(f"Unsupported dtype in meta: {s}")

@dataclass
class Artifacts:
    export_path: Path
    meta_path: Path
    device: torch.device
    dtype: torch.dtype
    S_in: int
    G: int
    label: str  # "gpu" or "cpu"

def select_artifact() -> Artifacts:
    host_pref = pick_host_pref_device(DEVICE_STR)
    gpu_path = MODEL_DIR / GPU_BASENAME
    cpu_path = MODEL_DIR / CPU_BASENAME
    gpu_meta = MODEL_DIR / (GPU_BASENAME + ".meta.json")
    cpu_meta = MODEL_DIR / (CPU_BASENAME + ".meta.json")

    if host_pref in ("mps","cuda") and gpu_path.exists():
        meta = _load_meta(gpu_meta, default_device=host_pref)
        dev = meta["device"].lower()
        if dev not in ("cuda","mps"):
            dev = host_pref
        device = torch.device(dev)
        return Artifacts(
            export_path=gpu_path, meta_path=gpu_meta,
            device=device, dtype=_dtype_from_str(meta["dtype"]),
            S_in=int(meta["S_in"]), G=int(meta["G"]), label="gpu"
        )

    # Fallback to CPU artifact
    meta = _load_meta(cpu_meta, default_device="cpu")
    return Artifacts(
        export_path=cpu_path, meta_path=cpu_meta,
        device=torch.device("cpu"), dtype=_dtype_from_str(meta["dtype"]),
        S_in=int(meta["S_in"]), G=int(meta["G"]), label="cpu"
    )

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class LoadedData:
    in_names: List[str]
    globals_names: List[str]
    out_names: List[str]
    out_to_in: np.ndarray
    identity_map: bool

    y_traj: np.ndarray       # [M, S_in] float32
    g_vec: np.ndarray        # [G]       float32
    t_phys: np.ndarray       # [M]       float64

    norm: NormalizationHelper
    gm: torch.nn.Module
    dt_min_phys: float
    dt_max_phys: float
    artifact: Artifacts

# =============================================================================
# Load model, normalization, shard
# =============================================================================

def load_everything() -> LoadedData:
    os.chdir(REPO)
    seed_everything(42)

    # Config & normalization
    cfg = load_json(MODEL_DIR / "config.json")
    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")

    norm_data = load_json(data_dir / "normalization.json")
    meta = norm_data.get("meta", {})
    in_names: List[str] = list(meta.get("species_variables") or [])
    globals_v: List[str] = list(meta.get("global_variables") or [])
    if not in_names or globals_v is None:
        raise RuntimeError("Missing species_variables or global_variables in normalization.json")

    # Select & load exported program
    art = select_artifact()
    if not art.export_path.exists():
        raise FileNotFoundError(f"No artifact found at {art.export_path} (and no GPU artifact).")

    ep = torch.export.load(str(art.export_path))
    gm = ep.module()  # executable; device/dtype fixed by export

    # Normalizer
    norm = NormalizationHelper(norm_data)
    dt_min_phys, dt_max_phys = _dt_bounds_from_manifest(norm)

    # One test shard
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found in {data_dir / 'test'}")

    with np.load(shards[0]) as d:
        y_all = d["y_mat"].astype(np.float32)      # [N, M, S]
        g_all = d["globals"].astype(np.float32)    # [N, G]
        t_vec = d["t_vec"]                         # [M] or [N, M]
    print(f"Using shard: {shards[0].name}  |  total samples: {len(y_all)}")

    y_traj = y_all[SAMPLE_IDX]          # [M, S_in]
    g_vec  = g_all[SAMPLE_IDX]          # [G]
    t_phys = t_vec if t_vec.ndim == 1 else t_vec[SAMPLE_IDX]
    t_phys = t_phys.astype(np.float64)

    # Probe S_out & mapping (build inputs on artifact device/dtype)
    in_index = {n: i for i, n in enumerate(in_names)}
    y0_probe = norm.normalize(
        torch.from_numpy(y_traj[0:1]),
        in_names
    ).to(art.device, art.dtype)

    g_probe = norm.normalize(
        torch.from_numpy(g_vec[None, :]),
        globals_v
    ).to(art.device, art.dtype)

    with torch.inference_mode():
        # IMPORTANT: dt must be [B,1] because export traced with shape [B,1]
        dt_probe = torch.tensor([1.0], device=art.device, dtype=art.dtype).view(1, 1)
        out_probe = gm(y0_probe, dt_probe, g_probe)
    out_probe = _coerce_pred_shape(out_probe)

    S_out = int(out_probe.shape[-1])
    out_names: List[str] = list(in_names[:S_out])
    out_to_in = np.array([in_index.get(n, -1) for n in out_names], dtype=int)
    identity_map = (S_out == len(in_names)) and np.all(out_to_in == np.arange(S_out))

    return LoadedData(
        in_names=in_names,
        globals_names=globals_v,
        out_names=out_names,
        out_to_in=out_to_in,
        identity_map=identity_map,
        y_traj=y_traj,
        g_vec=g_vec,
        t_phys=t_phys,
        norm=norm,
        gm=gm,
        dt_min_phys=dt_min_phys,
        dt_max_phys=dt_max_phys,
        artifact=art,
    )

# =============================================================================
# Utilities
# =============================================================================

@torch.no_grad()
def _coerce_pred_shape(y_pred) -> torch.Tensor:
    if isinstance(y_pred, (list, tuple)):
        y_pred = y_pred[0]
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)
    if y_pred.ndim == 1:
        return y_pred.view(1, -1)
    if y_pred.ndim == 3 and y_pred.shape[1] == 1:
        return y_pred[:, 0, :]
    if y_pred.ndim == 2:
        return y_pred
    raise RuntimeError(f"Unexpected export output shape: {tuple(y_pred.shape)}")

def _dt_bounds_from_manifest(norm: NormalizationHelper) -> tuple[float, float]:
    man = getattr(norm, "manifest", {}) or {}
    per = dict(man.get("per_key_stats", {}))
    methods = dict(man.get("normalization_methods", {}))
    if methods.get("dt") == "log-min-max":
        s = per.get("dt", {})
        return 10.0 ** float(s.get("log_min", -3.0)), 10.0 ** float(s.get("log_max", 8.0))
    if methods.get("t_time") == "log-min-max":
        s = per.get("t_time", {})
        return 10.0 ** float(s.get("log_min", -3.0)), 10.0 ** float(s.get("log_max", 8.0))
    return 1e-3, 1e8

def _effective_dt_bounds(data: LoadedData) -> Tuple[float, float]:
    """
    Returns the effective physical Δt bounds after applying user overrides,
    validated so min < max.
    """
    lo = float(data.dt_min_phys)
    hi = float(data.dt_max_phys)
    if DT_MIN_USER is not None:
        lo = max(lo, float(DT_MIN_USER))
    if DT_MAX_USER is not None:
        hi = min(hi, float(DT_MAX_USER))
    if not (lo < hi):
        raise ValueError(f"Invalid Δt bounds after overrides: min={lo}, max={hi}")
    return lo, hi

def _norm_globals(norm: NormalizationHelper, g_vec: np.ndarray, keys: List[str],
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return norm.normalize(
        torch.from_numpy(g_vec[None, :]),
        keys
    ).to(device, dtype)

def _norm_state(norm: NormalizationHelper, y_phys: np.ndarray, keys: List[str],
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return norm.normalize(
        torch.from_numpy(np.maximum(y_phys, CLIP_MIN_FEED)[None, :]).to(torch.float32),
        keys,
    ).to(device, dtype)

def _norm_dt(norm: NormalizationHelper, dt_phys: np.ndarray | float,
             device: torch.device, dtype: torch.dtype,
             clamp_bounds: Tuple[float, float] | None = None) -> torch.Tensor:
    """
    Normalize physical dt(s) to the model's expected normalized range.

    Returns shape [B,1] to match the export contract.
    If clamp_bounds is provided, clamp in *physical* space before normalization.
    """
    v = np.asarray(dt_phys, dtype=np.float32).reshape(-1)
    if clamp_bounds is not None:
        lo, hi = clamp_bounds
        np.clip(v, lo, hi, out=v)

    dt_hat = norm.normalize_dt_from_phys(torch.tensor(v, dtype=torch.float32))
    dt_hat = torch.clamp(dt_hat, DT_EPS_NORM, 1.0 - DT_EPS_NORM)
    dt_hat = dt_hat.to(device, dtype)
    return dt_hat.view(-1, 1)  # [B,1]

# =============================================================================
# Prediction helpers (batched/non-AR and stepwise/AR)
# =============================================================================

def batch_predict_from_anchor(
    data: LoadedData, y_anchor_phys: np.ndarray, dt_list: np.ndarray
) -> np.ndarray:
    """
    Non-autoregressive: predict y(t0 + Δt_k) for many Δt in parallel,
    always conditioning on the SAME anchor state y_anchor_phys at t0.
    """
    K = int(len(dt_list))
    if K == 0:
        return np.zeros((0, len(data.out_names)), dtype=np.float32)

    dev, dtp = data.artifact.device, data.artifact.dtype
    g_norm = _norm_globals(data.norm, data.g_vec, data.globals_names, dev, dtp).expand(K, -1)
    y_in_norm = _norm_state(data.norm, y_anchor_phys, data.in_names, dev, dtp).expand(K, -1)

    # Enforce physical dt bounds here
    dt_bounds = _effective_dt_bounds(data)
    dt_b = _norm_dt(data.norm, dt_list, dev, dtp, clamp_bounds=dt_bounds)  # [K,1]

    # warmup
    for _ in range(WARMUP_STEPS):
        with torch.inference_mode():
            _ = _coerce_pred_shape(data.gm(y_in_norm[:1], dt_b[:1], g_norm[:1]))

    with torch.inference_mode():
        z = _coerce_pred_shape(data.gm(y_in_norm, dt_b, g_norm))
        if Z_CLIP_FOR_FB is not None:
            lo, hi = Z_CLIP_FOR_FB
            z = torch.clamp(z, float(lo), float(hi))

    # ---- IMPORTANT: denormalize on CPU to avoid MPS float64 issue ----
    z_cpu = z.detach().to("cpu")
    y_phys = data.norm.denormalize(z_cpu, data.out_names).numpy()
    return y_phys  # [K,S_out]

def predict_next_from_current_AR(
    data: LoadedData, y_curr_phys: np.ndarray, g_norm: torch.Tensor, dt_phys: float
) -> np.ndarray:
    """
    Autoregressive 1-step: given CURRENT physical state y_curr_phys,
    predict y_next after dt_phys and return it in physical space.
    """
    dev, dtp = data.artifact.device, data.artifact.dtype
    y_in_norm = _norm_state(data.norm, y_curr_phys, data.in_names, dev, dtp)  # [1,S_in]
    dt_b = _norm_dt(data.norm, float(dt_phys), dev, dtp,
                    clamp_bounds=_effective_dt_bounds(data))                 # [1,1]
    with torch.inference_mode():
        z = _coerce_pred_shape(data.gm(y_in_norm, dt_b, g_norm))
        if Z_CLIP_FOR_FB is not None:
            lo, hi = Z_CLIP_FOR_FB
            z = torch.clamp(z, float(lo), float(hi))

    # ---- IMPORTANT: denormalize on CPU to avoid MPS float64 issue ----
    z_cpu = z.detach().to("cpu")
    y_next = data.norm.denormalize(z_cpu, data.out_names).numpy().reshape(-1)
    return y_next

# =============================================================================
# Three modes
# =============================================================================

def run_mode_full_direct(data: LoadedData) -> Tuple[np.ndarray, np.ndarray]:
    """
    MODE A:
    For each GT timestamp t[k], predict directly from y(t0) with Δt = t[k]-t[0],
    but only for Δt within the effective [min,max] bounds.
    """
    t = data.t_phys
    dt_all = (t[1:] - t[0]).astype(np.float64)
    lo, hi = _effective_dt_bounds(data)
    m = (dt_all >= lo) & (dt_all <= hi)
    dt_list = dt_all[m].astype(np.float32)

    y0 = data.y_traj[0].copy()
    y_pred = batch_predict_from_anchor(data, y0, dt_list)
    return dt_list.astype(np.float64), y_pred.astype(np.float64)

def run_mode_const_direct(data: LoadedData) -> Tuple[np.ndarray, np.ndarray]:
    """
    MODE B:
    Predict on a constant-Δt grid with step clamped to the effective [min,max] per-step bounds.
    The absolute Δt values (from the anchor) are truncated at the max bound.
    """
    lo, hi = _effective_dt_bounds(data)
    # Per-step size must lie within [lo, hi]
    step = float(CONST_STEP_DT)
    step = max(step, lo)
    step = min(step, hi)

    # Absolute Δt grid from anchor: start at 'step', end at min(CONST_END_DT, hi)
    start = max(float(CONST_START_DT), step)
    end   = min(float(CONST_END_DT), hi)
    if end < start + 1e-30:
        raise ValueError("CONST_END_DT too small after applying Δt bounds.")

    grid = np.arange(start, end + 1e-30, step, dtype=float)

    y0 = data.y_traj[0].copy()
    y_pred = batch_predict_from_anchor(data, y0, grid.astype(np.float32))
    return grid.astype(np.float64), y_pred.astype(np.float64)

def run_mode_profile_autoreg(data: LoadedData) -> Tuple[np.ndarray, np.ndarray]:
    """
    MODE C:
    Walk forward autoregressively with the GT *count* of steps, but use per-step
    Δt clamped to the effective [min,max]. The x-axis is cumulative sum of the
    *used* (clamped) Δt values (i.e., no longer forced to GT times).
    """
    t, y, g = data.t_phys, data.y_traj, data.g_vec
    i_start = 0 if AR_START_INDEX is None else int(AR_START_INDEX)
    i_end   = (len(t) - 1) if AR_END_INDEX is None else int(AR_END_INDEX)
    if not (0 <= i_start < len(t) - 1):
        raise ValueError("AR_START_INDEX must be within [0, M-2].")
    if not (i_start + 1 <= i_end < len(t)):
        raise ValueError("AR_END_INDEX must be within [i_start+1, M-1].")

    lo, hi = _effective_dt_bounds(data)
    g_norm = _norm_globals(data.norm, g, data.globals_names,
                           data.artifact.device, data.artifact.dtype)

    y_curr = np.maximum(y[i_start].copy(), CLIP_MIN_FEED)

    # warmup with first effective step
    dt_first = float(t[i_start + 1] - t[i_start])
    dt_first = min(max(dt_first, lo), hi)
    for _ in range(WARMUP_STEPS):
        _ = predict_next_from_current_AR(data, y_curr, g_norm, dt_first)

    pred_dt: List[float] = []
    pred_phys: List[np.ndarray] = []
    cum_t = 0.0

    for k in range(i_start, i_end):
        dt_raw = float(t[k + 1] - t[k])
        dt_used = min(max(dt_raw, lo), hi)

        y_next = predict_next_from_current_AR(data, y_curr, g_norm, dt_used)
        cum_t += dt_used

        pred_dt.append(cum_t)
        pred_phys.append(y_next)

        # Update current state for next step (autoregressive feedback)
        if data.identity_map:
            y_curr = np.maximum(y_next, CLIP_MIN_FEED)
        else:
            buf = y_curr.astype(np.float32, copy=True)
            for j_out, j_in in enumerate(data.out_to_in):
                if j_in >= 0:
                    buf[j_in] = max(y_next[j_out], CLIP_MIN_FEED)
            y_curr = buf

    return np.asarray(pred_dt, float), np.asarray(pred_phys, float)

# =============================================================================
# Plotting (fixed per-species colors)
# =============================================================================

def plot_all(data: LoadedData, pred_dt: np.ndarray, pred_phys: np.ndarray, mode_name: str) -> Path:
    in_index = {n: i for i, n in enumerate(data.in_names)}
    t, y, out_names = data.t_phys, data.y_traj, data.out_names

    base_colors = plt.cm.tab20(np.linspace(0, 0.95, len(out_names)))
    color_map = {nm: base_colors[i] for i, nm in enumerate(out_names)}

    dt_gt = (t - float(t[0])).astype(float)
    gt_full = np.zeros((len(t), len(out_names)), dtype=np.float32)
    for j_out, nm in enumerate(out_names):
        j_in = in_index.get(nm, None)
        if j_in is not None:
            gt_full[:, j_out] = y[:, j_in]
    gt_full = np.clip(gt_full, 1e-30, None)

    xmin = PLOT_XMIN if PLOT_XMIN is not None else (
        max(1e-12, dt_gt[1] * 0.5) if len(dt_gt) > 1 else 1e-12
    )
    xmax = PLOT_XMAX if PLOT_XMAX is not None else float(dt_gt[-1])
    ymin = PLOT_YMIN if PLOT_YMIN is not None else max(
        1e-30,
        float(np.nanmin(gt_full[gt_full > 0]) * 0.5)
    )
    ymax = PLOT_YMAX if PLOT_YMAX is not None else float(np.nanmax(gt_full) * 1.2)

    m_gt = (dt_gt >= xmin) & (dt_gt <= xmax)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Ground truth curves
    for nm in out_names:
        c = color_map[nm]
        j_in = in_index[nm]
        j_out = out_names.index(nm)
        ax.loglog(dt_gt[m_gt], gt_full[m_gt, j_out],
                  '-', lw=1.8, alpha=0.9, color=c)
        # If you want to mark the anchor value, uncomment:
        # start_val = float(np.clip(y[0, j_in], 1e-30, None))
        # ax.loglog([xmin], [start_val],
        #           marker='x', mec=c, mfc='none', mew=1.6, ms=7, linestyle='none')

    # Predictions
    if pred_dt.size and pred_phys.size:
        pred_phys = np.clip(pred_phys, 1e-30, None)
        for j_out, nm in enumerate(out_names):
            c = color_map[nm]
            ax.loglog(pred_dt, pred_phys[:, j_out],
                      'o-', lw=1.2, ms=5, mfc='none', mec=c, color=c)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Absolute time [s]")
    ax.set_ylabel("Species Abundance")
    ax.grid(False)

    # Legend by species, ordered by max abundance
    order = np.argsort(np.max(gt_full, axis=0))[::-1]
    legend_handles = [
        Line2D([0], [0], color=base_colors[idx], lw=2.0, alpha=0.9)
        for idx in order
    ]
    legend_labels = [out_names[idx] for idx in order]
    leg1 = ax.legend(
        legend_handles, legend_labels,
        loc='center left', bbox_to_anchor=(1.01, 0.6),
        title='Species', fontsize=10, title_fontsize=11
    )
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', marker='o', lw=1.2, label='Prediction'),
    ]
    ax.legend(
        handles=style_handles,
        loc='center left', bbox_to_anchor=(1.01, 0.2),
        fontsize=10, title_fontsize=11
    )

    fig.tight_layout()
    out_dir = MODEL_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"profile_{mode_name}_sample_{SAMPLE_IDX}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# =============================================================================
# Main
# =============================================================================

def main():
    data = load_everything()

    type_desc = {
        "full_direct": "direct-from-anchor (non-AR, full profile, batched)",
        "const_direct": "direct-from-anchor (non-AR, constant Δt grid, batched)",
        "profile_autoreg": "autoregressive (profile-matched steps with bounds)",
    }[PLOT_MODE]

    # Timed prediction pass
    t0 = perf_counter()
    if PLOT_MODE == "full_direct":
        pred_dt, pred_phys = run_mode_full_direct(data)
    elif PLOT_MODE == "const_direct":
        pred_dt, pred_phys = run_mode_const_direct(data)
    elif PLOT_MODE == "profile_autoreg":
        pred_dt, pred_phys = run_mode_profile_autoreg(data)
    else:
        raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}")

    # Ensure device work is finished before stopping timer
    if data.artifact.device.type == "cuda":
        torch.cuda.synchronize()
    elif data.artifact.device.type == "mps":
        torch.mps.synchronize()
    t1 = perf_counter()

    elapsed = t1 - t0
    K = int(pred_dt.size)
    ms_total = elapsed * 1e3
    ms_per = (ms_total / K) if K else float("nan")

    out_path = plot_all(data, pred_dt, pred_phys, PLOT_MODE)

    print(
        f"Mode: {PLOT_MODE}  |  Type: {type_desc}  |  Artifact: {data.artifact.label}"
        f"  |  Device: {data.artifact.device}  |  DType: {data.artifact.dtype}"
    )
    print(
        f"[TIMER] type={PLOT_MODE}, preds={K}, "
        f"elapsed_total={ms_total:.2f} ms, time_per_pred={ms_per:.3f} ms"
    )
    lo, hi = _effective_dt_bounds(data)
    print(f"dt_bounds_eff=[{lo:g}, {hi:g}] s  (dataset=[{data.dt_min_phys:g}, {data.dt_max_phys:g}])")
    if K:
        print(
            f"Pred Δt: min={pred_dt.min():.3g}, max={pred_dt.max():.3g}, K={K}"
        )
    print(f"Plot saved: {out_path}")

if __name__ == "__main__":
    main()
