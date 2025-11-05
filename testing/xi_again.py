#!/usr/bin/env python3
"""
MiniChem-anchored single-shot predictions at K log-spaced Δt (no autoregression),
with MiniChem truth interpolated on a log-time axis using a shape-preserving cubic (PCHIP),
and plotted on a Δt axis.

- Anchor y0 from MiniChem at absolute time T0 (INTERPOLATED at T0 with the same scheme)
- Δt grid = logspace(DT_MIN, T_FINAL - T0) with K_POINTS samples
- Single batched call to exported model (K=1 export signature)
- FEED_MIN merged with learned floor, then simplex projection before normalization
- Plot: solid = MiniChem truth (PCHIP-interpolated, renormalized to output-subset simplex, vs Δt)
        empty circles = predictions at Δt grid (same subset)
        hollow square = anchor at Δt = XMIN_DT
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json, pickle, sys, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("science.mplstyle")

# ---------- Paths ----------
REPO_ROOT     = Path("/Users/imalsky/Desktop/Chemulator")
SRC_DIR       = REPO_ROOT / "src"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR     = REPO_ROOT / "models" / "big"
VULCAN_PATH   = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar/vul-T1000KlogP3.0-NCHO-solar_hot_ini.vul")

# ---------- Time window (absolute) & Δt grid ----------
T0        = 1.0e-3                 # absolute anchor time (s)
T_FINAL   = T0 + 1.0e8             # absolute final time (s)
DT_MIN    = 1.0e-3                 # minimum Δt (s) for the logspace grid
K_POINTS  = 50                     # number of Δt samples (log-spaced)

# ---------- Δt-axis limits ----------
XMIN_DT   = DT_MIN                 # lower bound (Δt, s) on log axis (>0)
XMAX_DT   = 1e18                   # upper bound (Δt, s)

# ---------- Inputs & plotting knobs ----------
FEED_MIN   = 1e-15                 # min abundance fed into the model (merged with learned floor)
T_K        = 1000.0                # global T (K)
P_Pa       = 100.0                 # global P (Pa)
P_barye    = P_Pa * 10.0           # global P (barye)
PLOT_SPECIES: List[str] = ['H2','H2O','CH4','CO','CO2','NH3','HCN','N2','C2H2','H','CH3','OH','O']
PLOT_FLOOR = 1e-30

# ---------- Interpolation options ----------
# method: 'pchip' (shape-preserving cubic, linear abundance) | 'linear' | 'pchip_logy' (cubic on log10 abundance)
INTERP_METHOD = "pchip"
INTERP_EPS    = 1e-300   # floor to keep log-time/log-y safe

# ---------- Repo imports ----------
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from normalizer import NormalizationHelper  # type: ignore

# ---------- Helpers ----------
def read_json(p: Path) -> Dict:
    return json.loads(p.read_text())

def load_manifest_and_norm() -> Tuple[Dict, NormalizationHelper]:
    man_path = PROCESSED_DIR / "normalization.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}")
    manifest = read_json(man_path)
    return manifest, NormalizationHelper(manifest)

def get_meta_lists(manifest: Dict) -> Tuple[List[str], List[str], List[str]]:
    meta = manifest.get("meta", {})
    in_names: List[str] = list(meta.get("species_variables") or manifest.get("species_variables") or [])
    if not in_names:
        raise RuntimeError("species_variables missing in normalization.json")
    in_bases = [n[:-10] if n.endswith("_evolution") else n for n in in_names]
    gvars: List[str] = list(meta.get("global_variables") or manifest.get("global_variables") or [])
    return in_names, in_bases, gvars

def find_export(model_dir: Path) -> Path:
    for name in ("export_k1_cpu.pt2", "export_k1.pt2", "complete_model_exported_k1.pt2", "complete_model_exported.pt2"):
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No exported model found in {model_dir}")

def load_minichem(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"MiniChem/VULCAN file not found: {p}")
    with open(p, "rb") as h:
        d = pickle.load(h)
    t = np.asarray(d["variable"]["t_time"], dtype=float)     # [T]
    Y = np.asarray(d["variable"]["y_time"], dtype=float)     # [T, layer, S]
    names = list(d["variable"]["species"])
    den = np.maximum(Y[:, 0, :].sum(axis=-1), 1e-30)         # include He in denom
    MR  = Y[:, 0, :] / den[:, None]                          # mixing ratios
    return {"t": t, "MR": MR, "names": names}

def infer_training_floor(norm: NormalizationHelper, var_names: List[str]) -> np.ndarray:
    z = torch.zeros(1, len(var_names), dtype=torch.float32)
    eff = norm.denormalize(norm.normalize(z, var_names), var_names).numpy().reshape(-1)
    return np.maximum(np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0), 1e-30).astype(np.float64)

def map_from_dict_ordered(keys: List[str], val_by_key: Dict[str, float]) -> np.ndarray:
    return np.array([max(val_by_key.get(k, 0.0), 0.0) for k in keys], dtype=np.float64)

def project_to_training_simplex(vec: np.ndarray, floor_vec: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    v = np.maximum(v, floor_vec)
    s = float(v.sum())
    v = (np.full_like(v, 1.0/len(v)) if (not np.isfinite(s) or s <= 0) else (v / s))
    return v.astype(np.float32)

def safe_denorm(norm: NormalizationHelper, y_norm: torch.Tensor, names: List[str], manifest: Dict) -> np.ndarray:
    arr = norm.denormalize(y_norm, names).cpu().numpy()
    eps = float((manifest.get("normalization") or {}).get("epsilon", 1e-30))
    cap = float((manifest.get("normalization") or {}).get("clamp_value", 1e10))
    return np.clip(np.nan_to_num(arr, nan=eps, posinf=cap, neginf=eps), eps, cap)

# ---------- Robust preparation of MiniChem time series ----------
def _dedupe_and_sort_times(t: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure strictly increasing time grid; average values at duplicate times, sort by time."""
    t = np.asarray(t, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    order = np.argsort(t)
    t_sorted = t[order]
    Y_sorted = Y[order]

    uniq_vals, idx_start = np.unique(t_sorted, return_index=True)
    # average across duplicates
    t_out = []
    y_out = []
    for i, t_val in enumerate(uniq_vals):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(t_sorted)
        t_out.append(t_val)
        y_out.append(np.nanmean(Y_sorted[start:end, :], axis=0))
    t_out = np.asarray(t_out, dtype=np.float64)
    y_out = np.asarray(y_out, dtype=np.float64)
    # drop non-increasing remnants if any
    keep = np.diff(np.concatenate([[t_out[0]-1.0], t_out])) > 0.0
    return t_out[keep], y_out[keep, :]

# ---------- Shape-preserving cubic interpolation (Fritsch–Carlson, PCHIP) ----------
def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute tangents m_i for monotone cubic (per 1D array y)."""
    n = x.size
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)

    # interior slopes via weighted harmonic mean when same sign
    for i in range(1, n-1):
        if d[i-1] == 0.0 or d[i] == 0.0 or np.sign(d[i-1]) != np.sign(d[i]):
            m[i] = 0.0
        else:
            w1 = 2*h[i] + h[i-1]
            w0 = h[i] + 2*h[i-1]
            m[i] = (w0 + w1) / (w0/d[i-1] + w1/d[i])

    # endpoints (Fritsch–Butland style limiting)
    m0  = ((2*h[0] + h[1])*d[0] - h[0]*d[1]) / (h[0] + h[1]) if n > 2 else d[0]
    if np.sign(m0) != np.sign(d[0]): m0 = 0.0
    if (np.sign(d[0]) == np.sign(m0)) and (abs(m0) > 3*abs(d[0])): m0 = 3*d[0]
    m[n-1] = ((2*h[-1] + h[-2])*d[-1] - h[-1]*d[-2]) / (h[-1] + h[-2]) if n > 2 else d[-1]
    if np.sign(m[n-1]) != np.sign(d[-1]): m[n-1] = 0.0
    if (np.sign(d[-1]) == np.sign(m[n-1])) and (abs(m[n-1]) > 3*abs(d[-1])): m[n-1] = 3*d[-1]
    m[0] = m0
    return m

def _pchip_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate monotone cubic Hermite at xq (constant extrapolation outside domain)."""
    x0, xN = x[0], x[-1]
    out = np.empty_like(xq, dtype=np.float64)
    # left/right outside: constant
    left  = xq <= x0
    right = xq >= xN
    out[left]  = y[0]
    out[right] = y[-1]
    # interior
    mid_mask = ~(left | right)
    xm = xq[mid_mask]
    # interval indices
    k = np.searchsorted(x, xm) - 1
    k = np.clip(k, 0, x.size - 2)
    h  = x[k+1] - x[k]
    t  = (xm - x[k]) / h
    t2 = t * t
    t3 = t2 * t
    hmk = h
    yk  = y[k]
    yk1 = y[k+1]
    mk  = m[k]
    mk1 = m[k+1]
    out[mid_mask] = (
        (2*t3 - 3*t2 + 1)*yk
        + (t3 - 2*t2 + t)*hmk*mk
        + (-2*t3 + 3*t2)*yk1
        + (t3 - t2)*hmk*mk1
    )
    return out

def interp_MR_over_time(
    t_known: np.ndarray,
    MR_known: np.ndarray,     # [T, S] aligned with names
    t_query: np.ndarray,      # [Q]
    method: str = "pchip",
    use_log_time: bool = True,
    clip_floor: float = INTERP_EPS,
) -> np.ndarray:
    """
    Interpolate species mixing ratios over time.
    - x-axis: log10(t) if use_log_time else t
    - y-axis: linear abundance for 'linear' and 'pchip'; log10(y) for 'pchip_logy'
    - method: 'linear' (piecewise linear), 'pchip' (shape-preserving cubic), 'pchip_logy'
    """
    # Prepare x
    t_known = np.asarray(t_known, dtype=np.float64)
    MR_known = np.asarray(MR_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    # Keep strictly increasing time and average duplicates
    t_known, MR_known = _dedupe_and_sort_times(t_known, MR_known)

    if use_log_time:
        xk = np.log10(np.clip(t_known, clip_floor, None))
        xq = np.log10(np.clip(t_query, clip_floor, None))
    else:
        xk = t_known
        xq = t_query

    Q, S = len(t_query), MR_known.shape[1]
    out = np.empty((Q, S), dtype=np.float64)

    if method == "linear" or xk.size < 3:
        # fallback or insufficient points for cubic
        for j in range(S):
            yk = np.nan_to_num(MR_known[:, j], nan=0.0)
            out[:, j] = np.interp(xq, xk, yk, left=yk[0], right=yk[-1])
        return np.clip(out, clip_floor, None)

    if method == "pchip":
        for j in range(S):
            yk = np.nan_to_num(MR_known[:, j], nan=0.0)
            m  = _pchip_slopes(xk, yk)
            out[:, j] = _pchip_eval(xk, yk, m, xq)
        return np.clip(out, clip_floor, None)

    if method == "pchip_logy":
        for j in range(S):
            yk_lin = np.clip(np.nan_to_num(MR_known[:, j], nan=0.0), clip_floor, None)
            yk = np.log10(yk_lin)
            m  = _pchip_slopes(xk, yk)
            out[:, j] = np.power(10.0, _pchip_eval(xk, yk, m, xq))
        return np.clip(out, clip_floor, None)

    raise ValueError(f"Unknown interpolation method: {method}")

@torch.inference_mode()
def call_export(fn, y_b: torch.Tensor, dt_b: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
    out = fn(y_b, dt_b, g_b)  # expected [B,1,S] in some exports; [B,S] in others
    if isinstance(out, torch.Tensor):
        return out[:, 0, :] if (out.dim() == 3 and out.size(1) == 1) else out
    return torch.as_tensor(out)

# ---------- Main ----------
def main() -> None:
    torch.set_num_threads(1)

    manifest, norm = load_manifest_and_norm()
    in_names, in_bases, gvars = get_meta_lists(manifest)

    # Load MiniChem
    prof = load_minichem(VULCAN_PATH)
    t_all = prof["t"].astype(np.float64)      # [T]
    MR_all = prof["MR"].astype(np.float64)    # [T, S]
    names_all = prof["names"]
    name_to_idx = {n: i for i, n in enumerate(names_all)}

    # Globals (phys -> norm)
    if gvars:
        g_phys = np.zeros((1, len(gvars)), dtype=np.float32)
        for i, name in enumerate(gvars):
            n = name.strip().lower()
            g_phys[0, i] = P_barye if n.startswith("p") else (T_K if n.startswith("t") else 0.0)
        g_norm = norm.normalize(torch.from_numpy(g_phys), gvars).float()
    else:
        g_norm = torch.zeros(1, 0, dtype=torch.float32)

    # ---- Anchor from MiniChem @ T0 (INTERPOLATED with chosen method) ----
    MR_T0 = interp_MR_over_time(
        t_all, MR_all, np.array([T0], dtype=np.float64),
        method=INTERP_METHOD, use_log_time=True, clip_floor=INTERP_EPS
    )[0]  # [S]
    vul0_dict = {n: float(MR_T0[name_to_idx[n]]) for n in names_all}

    # Inputs: FEED_MIN merged with learned floor → simplex → normalize
    floor_train = infer_training_floor(norm, in_names)
    floor_total = np.maximum(floor_train, FEED_MIN)
    y0_inputs_phys = map_from_dict_ordered(in_bases, vul0_dict)           # [S_in]
    y0_simplex = project_to_training_simplex(y0_inputs_phys, floor_total)
    y0_norm = norm.normalize(torch.from_numpy(y0_simplex[None, :]), in_names).float()  # [1,S_in]

    # ---- Δt grids ----
    if not (T_FINAL > T0):
        raise ValueError("T_FINAL must be > T0.")
    req_span = float(T_FINAL - T0)

    # Predictions Δt grid (K points)
    dt_min_eff = max(DT_MIN, 1e-12)
    dt_pred = np.logspace(math.log10(dt_min_eff), math.log10(max(dt_min_eff, req_span)),
                          K_POINTS, dtype=np.float32)  # [K]
    t_pred_abs = T0 + dt_pred

    # Truth Δt grid covers entire MiniChem forward span (independent of predictions & axis limits)
    N_TRUTH = max(256, min(2048, 6 * K_POINTS))
    dt_truth_max = max(dt_min_eff, float(t_all.max() - T0))  # full MiniChem coverage
    dt_truth = np.logspace(math.log10(dt_min_eff), math.log10(dt_truth_max),
                           N_TRUTH, dtype=np.float64)  # [Q]
    t_truth_abs = T0 + dt_truth

    # ---- Exported model call (single shot at each Δt) ----
    from torch.export import load as torch_export_load
    ep = torch_export_load(str(find_export(MODEL_DIR)))
    fn = ep.module()
    dt_hat = norm.normalize_dt_from_phys(torch.from_numpy(dt_pred)).view(-1, 1).float()  # [K,1]
    y_pred_norm = call_export(fn, y0_norm.repeat(len(dt_hat), 1), dt_hat, g_norm.repeat(len(dt_hat), 1))  # [K,S_out]

    # Output names
    S_out = int(y_pred_norm.shape[-1])
    meta  = manifest.get("meta", {})
    cand  = list(meta.get("target_species_variables") or manifest.get("target_species_variables") or [])
    out_names = (cand if (cand and len(cand) == S_out)
                 else (in_names if len(in_names) == S_out else in_names[:S_out]))
    out_bases = [n[:-10] if n.endswith("_evolution") else n for n in out_names]

    # Denorm predictions
    y_pred_phys = safe_denorm(norm, y_pred_norm, out_names, manifest)  # [K,S_out]

    # ---- MiniChem truth interpolation on requested window (subset = outputs∩MiniChem) ----
    present_idx = [name_to_idx[b] for b in out_bases if b in name_to_idx]
    present_bases = [b for b in out_bases if b in name_to_idx]
    if not present_idx:
        raise RuntimeError("None of the model output species are present in the MiniChem profile.")

    MR_sub_truth = MR_all[:, np.array(present_idx, dtype=int)]  # [T, M]
    MR_truth_interp = interp_MR_over_time(
        t_all, MR_sub_truth, t_truth_abs,
        method=INTERP_METHOD, use_log_time=True, clip_floor=INTERP_EPS
    )  # [Q, M]
    print(f"[INFO] MiniChem truth interpolated with method='{INTERP_METHOD}' on log-time axis (shape-preserving if 'pchip').")

    # Renormalize both truth and preds to the subset simplex (no-He over outputs)
    truth_den = np.maximum(MR_truth_interp.sum(axis=1, keepdims=True), 1e-30)
    truth_simplex = MR_truth_interp / truth_den                                       # [Q, M]

    mask_out_sub = [b in present_bases for b in out_bases]
    pred_sub = y_pred_phys[:, np.where(mask_out_sub)[0]]                              # [K, M]
    pred_den = np.maximum(pred_sub.sum(axis=1, keepdims=True), 1e-30)
    pred_simplex = pred_sub / pred_den                                                # [K, M]

    # Species filter
    if PLOT_SPECIES:
        keep = [i for i, b in enumerate(present_bases) if b in PLOT_SPECIES]
    else:
        keep = list(range(len(present_bases)))
    if not keep:
        raise RuntimeError("Requested PLOT_SPECIES not present in outputs/MiniChem.")

    labels = [present_bases[i] for i in keep]
    truth_plot = truth_simplex[:, keep]                                               # [Q, m]
    pred_plot  = np.clip(pred_simplex[:, keep], PLOT_FLOOR, None)                     # [K, m]

    # Also compute y0 on outputs (at T0) for anchor marker
    in_idx_by_base = {b: j for j, b in enumerate(in_bases)}
    y0_on_outputs = np.array([y0_simplex[in_idx_by_base[b]] if b in in_idx_by_base else np.nan
                              for b in present_bases], dtype=np.float64)
    y0_on_outputs = y0_on_outputs[keep]

    # ---------- Plot on Δt axis (log–log) ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XMIN_DT, XMAX_DT)
    ax.set_xlabel("Δt since anchor (s)")
    ax.set_ylabel("Abundance (no-He simplex over model outputs)")

    colors = plt.cm.tab20(np.linspace(0, 0.95, len(keep)))

    # Solid truth vs Δt (dense grid)
    for i, col in enumerate(colors):
        ax.plot(dt_truth, np.clip(truth_plot[:, i], PLOT_FLOOR, None),
                '-', lw=1.8, alpha=0.95, color=col)

    # Anchor marker at Δt = XMIN_DT (hollow square)
    for i, col in enumerate(colors):
        y0v = y0_on_outputs[i]
        if np.isfinite(y0v):
            ax.plot([XMIN_DT], [max(y0v, PLOT_FLOOR)], marker='s', mfc='none', mec=col, ms=6, mew=1.2, linestyle='none')

    # Predictions as empty circles at Δt grid
    for i, col in enumerate(colors):
        ax.plot(dt_pred, pred_plot[:, i], linestyle='none',
                marker='o', mfc='none', mec=col, ms=4.0, mew=1.0, alpha=0.95)

    # Legends
    order = np.argsort(np.max(truth_plot, axis=0))[::-1]
    species_handles = [Line2D([0], [0], color=colors[i], lw=2.0) for i in order]
    species_labels  = [labels[i] for i in order]
    leg1 = ax.legend(handles=species_handles, labels=species_labels,
                     loc='center left', bbox_to_anchor=(1.01, 0.62),
                     title='Species', fontsize=10)
    ax.add_artist(leg1)
    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='MiniChem truth (interpolated)'),
        Line2D([0], [0], color='black', lw=0.0, ls='none', marker='o', mfc='none', mec='black', label='Prediction'),
        Line2D([0], [0], color='black', lw=0.0, ls='none', marker='s', mfc='none', mec='black', label='Anchor'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.18), fontsize=10)

    fig.tight_layout()
    out_png = MODEL_DIR / "plots" / "single_shot_minichem_deltat_interp_circles.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_png}")

if __name__ == "__main__":
    main()