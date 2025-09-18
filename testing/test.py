#!/usr/bin/env python3
"""
Flow-map DeepONet (K=1 export) — hardcode ALL inputs by NAME (no lists/loops in the edit section).

- Loads ONE test sample only for: time grid (x-axis) and ground truth curves.
- Hardcode globals and initial species with explicit set_* calls by name.
- Normalizes inputs via NormalizationHelper, runs exported K=1 model at selected times.
- Plots GT (solid) vs prediction (dashed + sparse 'x') on strict [XMIN, XMAX] log-log axes.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
REPO_ROOT     = Path(__file__).resolve().parent.parent
MODEL_DIR     = REPO_ROOT / "models" / "flowmap-deeponet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EXPORT_CANDIDATES = [
    MODEL_DIR / "complete_model_exported_k1.pt2",
    MODEL_DIR / "complete_model_exported_k1_int8.pt2",
]

# ============================================================
# RUNTIME / PLOTTING PARAMS
# ============================================================
SAMPLE_INDEX   = 0    # Which test sample to use for time grid + GT (+ defaults for unspecified species)
Q_COUNT        = 100  # Number of query times (<=0 -> all)
XMIN, XMAX     = 1e-3, 1e8  # strict x-axis bounds (seconds)
CONNECT_LINES  = True
MARKER_FREQ    = 5
SEED           = 42
assert XMIN > 0 and XMAX > XMIN, "Require 0 < XMIN < XMAX"

# ============================================================
# Repo imports
# ============================================================
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

# ============================================================
# Helpers
# ============================================================
def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n  ".join(map(str, paths))
    raise FileNotFoundError(f"No exported model found. Tried:\n  {tried}")

def _normkey(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _find_species_index(name: str, species_full: List[str]) -> Optional[int]:
    """Return index for species name (accepts optional '_evolution' suffix), else None."""
    key = _normkey(name)
    for i, n in enumerate(species_full):
        if _normkey(n) == key or _normkey(n.replace("_evolution", "")) == key:
            return i
    return None

def _find_global_index(name: str, globals_v: List[str]) -> Optional[int]:
    """Return index for a global by name (case/format-insensitive), else None."""
    key = _normkey(name)
    for i, n in enumerate(globals_v):
        if _normkey(n) == key:
            return i
    return None

def _load_species_globals() -> Tuple[List[str], List[str], List[str] | None]:
    """
    Return (species_full, globals_names, target_species_or_None).
    Priority: config.snapshot.json -> normalization.json -> empty lists.
    """
    snap = MODEL_DIR / "config.snapshot.json"
    if snap.exists():
        try:
            conf = json.load(open(snap, "r"))
            d = conf.get("data", {}) or {}
            species = list(d.get("species_variables", []) or [])
            globs   = list(d.get("global_variables", [])  or [])
            target  = list(d.get("target_species", [])    or []) or None
            if species:
                return species, globs, target
        except Exception:
            pass

    norm_path = PROCESSED_DIR / "normalization.json"
    if norm_path.exists():
        manifest = load_json(norm_path)
        meta = (manifest.get("meta", {}) or {})
        species = list(meta.get("species_variables", []) or [])
        globs   = list(meta.get("global_variables", [])  or [])
        return species, globs, None

    return [], [], None

def _load_exported_model(path: Path) -> torch.nn.Module:
    from torch.export import load as torch_export_load
    prog = torch_export_load(str(path))
    return prog.module()

def _load_single_test_sample(data_dir: Path, sample_idx: int | None) -> dict:
    test_dir = data_dir / "test"
    shards = sorted(test_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No test shards in {test_dir}")
    with np.load(shards[0], allow_pickle=False) as d:
        y = d["y_mat"].astype(np.float32)    # [N,M,S]
        g = d["globals"].astype(np.float32)  # [N,G]
        tvec = d["t_vec"]                    # [M] or [N,M]
    N = y.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "y0":       y[sample_idx:sample_idx + 1, 0, :],                   # [1,S_full]
        "y_true":   y[sample_idx],                                        # [M,S_full]
        "t_phys":   tvec[sample_idx] if tvec.ndim == 2 else tvec,         # [M]
        "globals":  g[sample_idx:sample_idx + 1],                          # [1,G]
        "sample_idx": int(sample_idx),
    }

def _select_query_indices(count: int | None, t_phys: np.ndarray, exclude_first: bool = True) -> np.ndarray:
    M = int(t_phys.size)
    start = 1 if exclude_first else 0
    if M - start <= 0:
        raise ValueError("Time grid must have at least two points.")
    if not count or count <= 0 or count >= (M - start):
        return np.arange(start, M, dtype=int)
    return np.linspace(start, M - 1, int(count)).round().astype(int)

def _normalize_dt(norm: NormalizationHelper, dt_phys: np.ndarray, device: torch.device) -> torch.Tensor:
    eps = float(getattr(norm, "epsilon", 1e-25))
    dt = torch.as_tensor(dt_phys, dtype=torch.float32, device=device).clamp(min=eps)
    return norm.normalize_dt_from_phys(dt.view(-1))

@torch.inference_mode()
def _predict_many(
    fn,
    y0_norm: torch.Tensor,   # [1,S_full_normed]
    g_norm:  torch.Tensor,   # [1,G_normed]
    dt_norm_vec: torch.Tensor,  # [K]
    norm: NormalizationHelper,
    species_out: List[str],
) -> np.ndarray:
    preds = []
    for k in range(int(dt_norm_vec.numel())):
        dt1 = dt_norm_vec[k:k+1].reshape(-1)  # [1]
        out = fn(y0_norm, g_norm, dt1)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, device=y0_norm.device)
        out_2d = out.reshape(-1, out.shape[-1])  # [1, S_out]
        y_phys = norm.denormalize(out_2d, species_out)  # [1,S_out]
        preds.append(y_phys.squeeze(0).detach().cpu().numpy())
    return np.stack(preds, axis=0)  # [K,S_out]

def _plot_strict(
    t_phys: np.ndarray, y_true: np.ndarray,
    t_sel: np.ndarray,  y_pred: np.ndarray,
    species: List[str], out_path: Path,
    connect_lines: bool, xmin: float, xmax: float
) -> None:
    m_gt   = (t_phys >= xmin) & (t_phys <= xmax)
    m_pred = (t_sel  >= xmin) & (t_sel  <= xmax)
    t_gt, Y_gt = t_phys[m_gt],  y_true[m_gt, :]
    t_pd, Y_pd = t_sel[m_pred], y_pred[m_pred, :]

    eps_y = 1e-30
    Y_gt = np.clip(Y_gt, eps_y, None)
    Y_pd = np.clip(Y_pd, eps_y, None)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    # order legend by species max
    max_vals = np.max(Y_gt, axis=0)
    order = np.argsort(max_vals)[::-1]

    stride = max(1, len(t_pd) // max(1, MARKER_FREQ))

    for i in range(len(species)):
        c = colors[i]
        ax.loglog(t_gt, Y_gt[:, i], '-', color=c, lw=1.8, alpha=0.9)
        if t_gt.size:
            ax.loglog([t_gt[0]], [Y_gt[0, i]], 'o', color=c, ms=5, mfc='none')
        if connect_lines and len(t_pd) > 1:
            ax.loglog(t_pd, Y_pd[:, i], '--', color=c, lw=1.4, alpha=0.85)
        if len(t_pd):
            ax.loglog(t_pd[::stride], Y_pd[::stride, i], 'x', color=c, ms=5, alpha=0.9)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Species Abundance")
    ax.grid(False)

    from matplotlib.lines import Line2D
    leg_handles = [Line2D([0],[0], color=colors[j], lw=2.0, alpha=0.9) for j in order]
    leg_labels  = [species[j] for j in order]
    legend1 = ax.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1.01, 0.6),
                        title='Species', fontsize=10, title_fontsize=11, ncol=1, borderaxespad=0)
    ax.add_artist(legend1)

    style_handles = [
        Line2D([0],[0], color='black', lw=2.0, ls='-',  label='Ground Truth'),
        Line2D([0],[0], color='black', lw=1.6, ls='--', label='Model Prediction'),
    ]
    ax.legend(style_handles, [h.get_label() for h in style_handles],
              loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11, borderaxespad=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")

# ============================================================
# Main
# ============================================================
def main() -> None:
    seed_everything(SEED)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # Names
    species_full, globals_v, target_species = _load_species_globals()
    if not species_full:
        raise RuntimeError("Could not resolve species/global variables from snapshot/normalization.")
    species_out = target_species if (target_species and len(target_species) > 0) else species_full

    # Model + Normalization
    model_path = _first_existing(EXPORT_CANDIDATES)
    fn   = _load_exported_model(model_path)
    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))

    # Sample (for time grid + GT + defaults)
    data          = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)
    y0_phys_full  = data["y0"].copy()       # [1,S_full]
    g_phys        = data["globals"].copy()  # [1,G]
    t_phys        = data["t_phys"].astype(np.float64)  # [M]
    y_true_full   = data["y_true"]          # [M,S_full]
    sample_idx    = data["sample_idx"]

    # --------------------------------------------------------
    # HARD-CODED INPUTS — EDIT ONLY BELOW
    # No lists/loops here. Each assignment is by NAME.
    # If a name does not exist in the dataset, a warning is printed and nothing is changed.
    # Pressure target: 1 mbar = 100 Pa = 1e-3 bar = 1e3 dyn/cm^2.
    # Temperature target: 1000 K.
    # --------------------------------------------------------

    def set_global(name: str, value: float) -> None:
        idx = _find_global_index(name, globals_v)
        if idx is None:
            print(f"[WARN] Global '{name}' not found; ignoring.")
            return
        g_phys[0, idx] = float(value)

    def set_species(name: str, value: float) -> None:
        idx = _find_species_index(name, species_full)
        if idx is None:
            print(f"[WARN] Species '{name}' not found; ignoring.")
            return
        y0_phys_full[0, idx] = float(value)

    # ---- Globals (choose the appropriate names that exist in your dataset) ----
    # Temperature (Kelvin)
    set_global("T",   1.625354e+03)
    set_global("P",     6.818810e+07)   # if P is stored as Pa

    # ---- Species at t0 (physical). Your requested 1000 K, 1 mbar values: ----
    #set_species("H2",   9.975331e-01)
    #set_species("H2O",  1.074060e-03)
    #set_species("H",    0.000000e+00)
    #set_species("CH4",  5.902400e-04)
    #set_species("CO",   0.000000e+00)
    #set_species("CO2",  0.000000e+00)
    #set_species("N2",   0.000000e+00)
    #set_species("NH3",  1.415900e-04)
    #set_species("He",   1.679000e-01)

    set_species("C2H2", 8.913668e-10)
    set_species("C2H2", 1e-15)

    set_species("CH3", 1.034118e-08)
    set_species("CH3", 1e-15)

    set_species("CH4", 1.168895e-11)
    set_species("CH4", 1e-15)


    set_species("CO2", 3.010446e-05)
    set_species("CO2", 1e-15)

    set_species("CO", 4.093728e-04)
    set_species("CO", 4.093728e-04)


    set_species("H2O", 2.481025e-07)
    set_species("H2", 9.886430e-01)
    set_species("HCN", 5.410422e-12)
    set_species("H", 6.946361e-07)
    set_species("N2", 2.452540e-10)
    set_species("NH3", 1.018076e-03)
    set_species("OH", 9.898477e-03)

    set_species("O", 5.335861e-11)
    set_species("O", 1e-40)



    # --------------------------------------------------------
    # END HARD-CODED INPUTS
    # --------------------------------------------------------

    # Report the actual inputs used
    print("\n=== Inputs used (physical) ===")
    for i, name in enumerate(globals_v):
        print(f"{name:>16s} : {float(g_phys[0, i]):.6e}")

    print("\n=== Initial y0 (species_full order) ===")
    for i, name in enumerate(species_full):
        print(f"{name:>24s} : {float(y0_phys_full[0, i]):.6e}")

    # Normalize inputs
    y0n = norm.normalize(torch.from_numpy(y0_phys_full).to(device), species_full)  # [1,S_full_normed]
    gn  = norm.normalize(torch.from_numpy(g_phys).to(device),        globals_v)    # [1,G_normed]

    # Query times / dt
    t0 = float(t_phys[0])
    idx = _select_query_indices(Q_COUNT, t_phys, exclude_first=True)
    idx = idx[idx > 0] if len(idx) > 0 else np.array([1], dtype=int)
    t_sel = t_phys[idx]
    dt_phys_sel = np.maximum(t_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, device)

    print(f"\n[INFO] Sample index = {sample_idx}")
    print(f"[INFO] Flow-map inference: K={int(dt_norm_sel.numel())}")
    print(f"[INFO] Excluding t0={t0:.3e}s from predictions (Δt=0 not in training set)")
    if len(t_sel):
        print(f"[INFO] First prediction at t={t_sel[0]:.3e}s (Δt={dt_phys_sel[0]:.3e}s)")

    # Predict (physical)
    y_pred = _predict_many(fn, y0n, gn, dt_norm_sel, norm, species_out)   # [K,S_out]

    # Align GT to target species
    spec_index = { n: i for i, n in enumerate(species_full) }
    target_idx = [spec_index[n] for n in species_out]
    y_true_subset = y_true_full[:, target_idx]  # [M,S_out]

    # Plot
    out_png = MODEL_DIR / "plots" / f"predictions_strict_xlim_K{int(dt_norm_sel.numel())}_sample_{sample_idx}.png"
    _plot_strict(t_phys, y_true_subset, t_sel, y_pred, species_out, out_png,
                 connect_lines=CONNECT_LINES, xmin=XMIN, xmax=XMAX)

    # Relative error
    y_true_aligned = y_true_subset[idx, :]
    rel = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)
    print(f"[ERROR] rel-mean={rel.mean():.3e}, rel-max={rel.max():.3e}")

if __name__ == "__main__":
    main()
