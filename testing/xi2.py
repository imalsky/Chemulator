#!/usr/bin/env python3
"""
Minimal autoregressive chemical evolution (globals-only configuration) + DEBUG.

What you get:
  • Hard-cap per-substep Δt at DT_MAX_HARDCAP.
  • Replace-any-<threshold with REPLACE_WITH, then optional renorm (your policy).
  • Uses ONLY norm.normalize_dt_from_phys() for time.
  • Steps along the VULCAN timeline; chunk any Δt bigger than allowed.
  • Clear debug on requested Δt, chunking, and substep counts.

Blunt note:
  If you set REPLACE_WITH to something huge and keep renormalizing, you’ll
  dominate the composition each time. That’s not a bug—just math.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import json
import pickle
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# ========================= USER CONFIGURATION =========================

# Repo root inferred from this file location: .../Chemulator/testing/xi2.py
ROOT = Path(__file__).resolve().parents[1]  # .../Chemulator
SRC  = ROOT / "src"                         # to import normalizer.py

# Exact files to use
NORM_PATH   = ROOT / "data" / "processed" / "normalization.json"
MODEL_PATH  = ROOT / "models" / "v4_2" / "export_k1_cpu.pt2"   # ← pick the exact model here
VULCAN_PATH = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar/vul-T1000KlogP3.0-NCHO-solar_hot_ini.vul")

# Physical globals for this run (units must match your training setup)
T_K  = 1000.0   # Kelvin
P_Pa = 100.0    # Pascals

# Time window drawn from the VULCAN profile
T0      = 1.0e0     # start time [s]
T_FINAL = 1e11     # end time [s]

# Plot controls
PLOT_SPECIES = ['H2', 'H2O', 'CH4', 'NH3', 'CO', 'CO2', 'HCN', 'N2', 'C2H2']
YLIM         = (1e-15, 2.0)
PLOT_FLOOR   = 1e-30  # display-only floor (not fed to model)

# Replacement/flooring policy (applied when seeding and before each model call)
REPLACE_THRESH      = 1e-15    # if value < this
REPLACE_WITH        = 1e-15    # replace by this
RENORMALIZE_SIMPLEX = True     # then renormalize to sum=1

# Δt cap (absolute upper bound for a physical substep)
DT_MAX_HARDCAP = 5e7  # seconds

# Torch threading (optional)
TORCH_NUM_THREADS = 1

# ---- DEBUG knobs ------------------------------------------------------
DEBUG_ENABLE = True
# Show per-step details for the first N coarse steps (set 0 to silence)
DEBUG_LIST_FIRST_N_STEPS = 10
# Also emit a one-line summary every K steps (useful when N is small)
DEBUG_SUMMARY_EVERY_K_STEPS = 50

# Optional plotting style
try:
    plt.style.use((ROOT / "testing" / "science.mplstyle").as_posix())
except Exception:
    pass

# ======================= END USER CONFIGURATION =======================

# Make src/ importable
if not SRC.is_dir():
    raise RuntimeError(f"Expected src dir at {SRC}; adjust ROOT if layout differs.")
sys.path.insert(0, str(SRC))
from normalizer import NormalizationHelper  # noqa: E402

# ------------------------------- Utilities -------------------------------

def debug(msg: str):
    if DEBUG_ENABLE:
        print(msg)

def nearest_index(a: np.ndarray, x: float) -> int:
    """Index i minimizing |a[i]-x| (1D array)."""
    return int(np.argmin(np.abs(a - x)))

def dt_chunk(dt_phys: float, dt_max: float) -> List[float]:
    """
    Split a physical Δt into chunks of size ≤ dt_max:
      [dt_max, dt_max, ..., remainder (≤ dt_max)].
    If dt_phys ≤ dt_max, returns [dt_phys].
    """
    dt_phys = float(dt_phys)
    dt_max  = float(dt_max)
    if dt_phys <= dt_max:
        return [dt_phys]
    n_full    = int(dt_phys // dt_max)
    remainder = dt_phys - n_full * dt_max
    out = [dt_max] * n_full
    if remainder > 0.0:
        out.append(remainder)
    return out

def strip_suffix(name: str, suffix: str = "_evolution") -> str:
    return name[:-len(suffix)] if name.endswith(suffix) else name

def replace_small_then_renorm(y_phys: np.ndarray,
                              thresh: float = REPLACE_THRESH,
                              replacement: float = REPLACE_WITH,
                              renorm: bool = RENORMALIZE_SIMPLEX) -> np.ndarray:
    """
    Replace any entry < thresh with `replacement`; optionally renormalize to sum=1.
    """
    y = np.asarray(y_phys, dtype=float).copy()
    y[y < float(thresh)] = float(replacement)
    if renorm:
        s = y.sum()
        if not np.isfinite(s) or s <= 0.0:
            y[:] = 1.0 / len(y)
        else:
            y /= s
    return y

def infer_dt_max_phys(norm_manifest: Dict) -> float:
    """
    Reconstruct the physical training Δt max from normalization.json.
      1) If stats.dt.max exists, use it.
      2) Else, if top-level "dt" has "log_max" (log-min-max), return 10**log_max.
      3) Clamp to 'clamp_value' if present.
      4) Finally, enforce the hard-cap DT_MAX_HARDCAP.
    """
    stats = norm_manifest.get("stats", {})
    dt_stats = stats.get("dt", {})
    if isinstance(dt_stats, dict) and "max" in dt_stats:
        dt_max = float(dt_stats["max"])
    else:
        dt_cfg = norm_manifest.get("dt", {})
        if "log_max" not in dt_cfg:
            raise ValueError("Cannot infer dt_max: neither stats.dt.max nor dt.log_max present.")
        log_max = float(dt_cfg["log_max"])
        dt_max = 10.0 ** log_max

    clamp_val = norm_manifest.get("clamp_value", None)
    if clamp_val is not None:
        dt_max = min(dt_max, float(clamp_val))

    # Enforce hard cap
    dt_max = min(dt_max, float(DT_MAX_HARDCAP))
    return dt_max

# --------------------------------- I/O ----------------------------------

def load_vulcan(path: Path) -> Dict:
    """
    Load a VULCAN 0-D pickle with fields:
      d["variable"]["t_time"]   -> [M] seconds
      d["variable"]["y_time"]   -> [M, 1, S]
      d["variable"]["species"]  -> list[str] length S
    Returns { "t": [M], "MR": [M,S] row-normalized, "names": list[str] }
    """
    with open(path, "rb") as h:
        d = pickle.load(h)
    t = np.asarray(d["variable"]["t_time"], dtype=float)        # [M]
    Y = np.asarray(d["variable"]["y_time"], dtype=float)        # [M, 1, S]
    names = list(d["variable"]["species"])                      # [S]
    den = np.maximum(Y[:, 0, :].sum(axis=-1), 1e-300)
    MR = Y[:, 0, :] / den[:, None]                              # [M, S]
    return {"t": t, "MR": MR, "names": names}

# ------------------------- Normalization prep --------------------------

def build_globals_tensor(norm: NormalizationHelper,
                         global_vars: List[str],
                         T_K: float,
                         P_Pa: float) -> torch.Tensor:
    """
    Create [1, G] physical globals in the same order used in training,
    then normalize via norm.normalize(..., global_vars).
    Assumes your provided T_K, P_Pa are already in the training units.
    """
    g = np.zeros((1, len(global_vars)), dtype=np.float32)
    for i, name in enumerate(global_vars):
        lname = name.lower()
        if lname == 'p' or 'press' in lname:
            g[0, i] = float(P_Pa)
        elif lname == 't' or 'temp' in lname:
            g[0, i] = float(T_K)
        else:
            g[0, i] = 0.0  # set other globals as needed
    g_z = norm.normalize(torch.from_numpy(g), global_vars).float()
    return g_z

def get_species_and_globals(norm_manifest: Dict) -> Tuple[List[str], List[str]]:
    meta = norm_manifest.get("meta", {})
    species_vars = list(meta.get("species_variables", []))
    global_vars  = list(meta.get("global_variables", []))
    if not species_vars:
        raise ValueError("No 'species_variables' in normalization.json meta.")
    return species_vars, global_vars

# -------------------------- Autoregressive loop ------------------------

@torch.inference_mode()
def run_autoreg(
    model,
    norm: NormalizationHelper,
    y0_phys: np.ndarray,             # [S]
    t_ref: np.ndarray,               # [M] strictly increasing seconds
    species_vars: List[str],         # training species order
    g_z: torch.Tensor,               # [1, G]
    dt_max_phys: float,              # training+cap max Δt [s]
    plot_floor: float,
) -> np.ndarray:
    """
    Advance y along t_ref using ONLY model's time normalization and
    chunking by dt_max_phys. Returns [M,S] in physical space.
    Also prints detailed debug about Δt and chunking.
    """
    device = torch.device("cpu")
    dtype  = torch.float32

    # Precompute all requested coarse Δt
    dt_req = np.diff(t_ref).astype(float)  # [M-1]
    num_coarse = dt_req.size
    if num_coarse <= 0:
        raise ValueError("Need at least two time points in t_ref.")

    # Summarize Δt stats
    dt_min = float(np.min(dt_req))
    dt_max = float(np.max(dt_req))
    dt_med = float(np.median(dt_req))
    debug(f"[DEBUG] Coarse time jumps: {num_coarse} "
          f"| Δt_req min/median/max = {dt_min:.3e} / {dt_med:.3e} / {dt_max:.3e} s "
          f"| dt_max_phys (cap) = {dt_max_phys:.3e} s")

    # For early steps, show the exact chunking
    if DEBUG_LIST_FIRST_N_STEPS > 0:
        n_list = min(DEBUG_LIST_FIRST_N_STEPS, num_coarse)
        for k in range(n_list):
            ch = dt_chunk(dt_req[k], dt_max_phys)
            debug(f"[DEBUG] step {k+1:4d}/{num_coarse}: Δt_req={dt_req[k]:.3e} s "
                  f"→ chunks={len(ch)} {['%.3e' % c for c in ch]}")

    # Compute total substeps (without mutating state)
    chunk_counts = [len(dt_chunk(float(d), dt_max_phys)) for d in dt_req]
    total_substeps = int(np.sum(chunk_counts))
    num_chunked_steps = int(np.sum([c > 1 for c in chunk_counts]))
    debug(f"[DEBUG] Steps needing chunking: {num_chunked_steps}/{num_coarse} "
          f"| Total model calls (substeps): {total_substeps}")

    # Normalize initial state (order matches species_vars)
    y_z = norm.normalize(torch.from_numpy(y0_phys[None, :]).to(dtype), species_vars).to(device, dtype)

    M = t_ref.size
    S = y0_phys.size
    out = np.zeros((M, S), dtype=np.float64)
    out[0, :] = np.maximum(y0_phys, plot_floor)

    for k in range(1, M):
        dt_phys = float(t_ref[k] - t_ref[k - 1])
        if dt_phys <= 0.0 or not np.isfinite(dt_phys):
            out[k, :] = out[k - 1, :]
            continue

        chunks = dt_chunk(dt_phys, dt_max_phys)

        # Optional periodic summary line
        if DEBUG_SUMMARY_EVERY_K_STEPS and (k % DEBUG_SUMMARY_EVERY_K_STEPS == 0):
            debug(f"[DEBUG] step {k:4d}/{num_coarse}: Δt_req={dt_phys:.3e} s | substeps={len(chunks)}")

        for chunk in chunks:
            # Replacement/flooring before each model call
            y_phys_curr = norm.denormalize(y_z, species_vars).cpu().numpy().reshape(-1)
            y_phys_curr = replace_small_then_renorm(y_phys_curr)
            y_z = norm.normalize(torch.from_numpy(y_phys_curr[None, :]).to(dtype), species_vars).to(device, dtype)

            # Normalize Δt and step
            dt_norm = norm.normalize_dt_from_phys(torch.tensor([chunk], dtype=dtype)).view(1, 1, 1)
            y_z = model(y_z, dt_norm, g_z)
            if y_z.dim() == 3:  # some exports emit [1,1,S]
                y_z = y_z.squeeze(1)

        # Record the physical state at this coarse time
        y_phys_out = norm.denormalize(y_z, species_vars).cpu().numpy().reshape(-1)
        y_phys_out = replace_small_then_renorm(y_phys_out)
        y_z = norm.normalize(torch.from_numpy(y_phys_out[None, :]).to(dtype), species_vars).to(device, dtype)
        out[k, :] = np.maximum(y_phys_out, plot_floor)

    debug("[DEBUG] Autoregression complete.")
    return out

# -------------------------------- Plotting --------------------------------

def plot_model_vs_vulcan(times: np.ndarray,
                         model_phys: np.ndarray,      # [M, S]
                         species_vars: List[str],
                         vul: Dict,
                         plot_species: List[str],
                         y_limit: Tuple[float, float]):
    """
    Overlay model vs VULCAN, both normalized by the sum over the tracked subset
    (species your model actually evolves, in training order).
    """
    tracked = [strip_suffix(s) for s in species_vars]
    name_to_idx_v = {n: i for i, n in enumerate(vul["names"]) if n in tracked}
    tracked_idx   = [name_to_idx_v[n] for n in tracked if n in name_to_idx_v]

    # Normalize model per time by tracked-subset sum
    plot_floor = 1e-30
    sums_model = np.maximum(model_phys.sum(axis=1, keepdims=True), plot_floor)
    model_norm = model_phys / sums_model

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    c_map  = {sp: colors[i % 10] for i, sp in enumerate(plot_species)}

    # Model curves
    for j, sname in enumerate(species_vars):
        sp = strip_suffix(sname)
        if sp in plot_species:
            y = np.maximum(model_norm[:, j], plot_floor)
            ax.loglog(times, y, "--", lw=2, alpha=0.9, color=c_map[sp], label=f"{sp} (model)")

    # VULCAN curves normalized over tracked subset
    if tracked_idx:
        vt = vul["t"]
        vMR = vul["MR"]
        v_sum = np.maximum(vMR[:, tracked_idx].sum(axis=1, keepdims=True), plot_floor)
        mask = (vt >= times[0]) & (vt <= times[-1])
        if mask.any():
            for sp in plot_species:
                if sp in name_to_idx_v:
                    jv = name_to_idx_v[sp]
                    vy = np.maximum(vMR[:, jv] / v_sum[:, 0], plot_floor)
                    ax.loglog(vt[mask], vy[mask], "-", lw=1.6, alpha=0.55, color=c_map[sp], label=f"{sp} (VULCAN)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Abundance (subset-normalized)")
    ax.set_ylim(y_limit)
    ax.grid(True, which="both", alpha=0.25)

    # dedupe legend
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

# --------------------------------- Main ----------------------------------

def main():
    torch.set_num_threads(TORCH_NUM_THREADS)

    # Load normalization + model
    with open(NORM_PATH, "r") as f:
        norm_manifest = json.load(f)
    norm = NormalizationHelper(norm_manifest)
    species_vars, global_vars = get_species_and_globals(norm_manifest)
    dt_max_phys = infer_dt_max_phys(norm_manifest)  # already min(..., DT_MAX_HARDCAP)

    from torch.export import load as torch_export_load
    exported = torch_export_load(str(MODEL_PATH))
    model = exported.module()  # DO NOT call .eval() on torch.export programs

    # Load VULCAN and select time window
    vul = load_vulcan(VULCAN_PATH)
    t_all = vul["t"]
    if t_all.ndim != 1 or t_all.size < 2:
        raise ValueError("VULCAN time array invalid (need ≥2 samples).")

    i0 = nearest_index(t_all, T0)
    i1 = nearest_index(t_all, T_FINAL)
    lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
    t_win = t_all[lo:hi + 1]
    if t_win.size < 2:
        raise ValueError("Selected time window has fewer than 2 samples.")

    debug(f"[DEBUG] Time window indices: i0={i0}, i1={i1}, inclusive range [{lo}, {hi}] "
          f"→ {t_win.size} points, {t_win.size-1} coarse jumps")
    debug(f"[DEBUG] Start={t_win[0]:.3e}s End={t_win[-1]:.3e}s | dt_max_phys (cap)={dt_max_phys:.3e}s")

    # Initial state from VULCAN at T0 (nearest sample) → apply replacement rule
    names_vul = vul["names"]
    MR_vul    = vul["MR"]
    base_species = [strip_suffix(s) for s in species_vars]
    name_to_idx_v = {n: i for i, n in enumerate(names_vul)}

    y0 = np.zeros(len(base_species), dtype=np.float64)
    for j, sp in enumerate(base_species):
        y0[j] = MR_vul[i0, name_to_idx_v[sp]] if sp in name_to_idx_v else 0.0
    y0 = replace_small_then_renorm(y0)

    # Globals tensor (assumed-correct units)
    g_z = build_globals_tensor(norm, global_vars, T_K=T_K, P_Pa=P_Pa)

    # Run autoregressive evolution along the VULCAN timeline (with in-loop debug)
    model_phys = run_autoreg(
        model=model,
        norm=norm,
        y0_phys=y0,
        t_ref=t_win,
        species_vars=species_vars,
        g_z=g_z,
        dt_max_phys=dt_max_phys,
        plot_floor=PLOT_FLOOR,
    )

    # Plot model vs VULCAN
    plot_model_vs_vulcan(t_win, model_phys, species_vars, vul, PLOT_SPECIES, YLIM)

    print(f"[DONE] Steps: {t_win.size-1} | dt_max_used={dt_max_phys:.3e} s "
          f"| Start={t_win[0]:.3e}s End={t_win[-1]:.3e}s | Model={MODEL_PATH.name}")

if __name__ == "__main__":
    main()
