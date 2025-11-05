#!/usr/bin/env python3
"""
DeepONet vs VULCAN (debug) — constant He re-introduction that PRESERVES SHAPE

What this does:

• Model inputs:
  - Panel 1 uses VULCAN at the first t >= 1 s to build y0 for the 13 target species:
      * with-He source (VULCAN mixing ratios), map to targets
      * missing -> 1e-25; floor <1e-25 -> 1e-25
      * renormalize ACROSS the 13 targets to sum=1 (NO He) -> feed to model
      * hollow circle shows the with-He initial value from VULCAN (floored only for plotting)
  - Panel 2 uses your Case-2 (NO He) map:
      * missing -> 1e-25; floor; renorm across the 13 targets to sum=1 (NO He) -> feed to model

• Model evaluation times:
  - Panel 1: 1e0 .. 1e8
  - Panel 2: 1e-3 .. 1e8

• Axes:
  - Panel 1 x-axis: 10**-0.1 .. 1e18 (VULCAN overlays 1e0..1e18)
  - Panel 2 x-axis: 1e-3 .. 1e8

• Helium (Panel 2 ONLY, SHAPE-PRESERVING):
  - After denormalizing model outputs to physical (no-He space),
    multiply EVERYTHING by a single constant factor: SCALE = (1 - He_mixing_ratio)
    — NO per-time renorm. Shapes are preserved exactly.
  - We print min/max of (after / before) to prove it’s constant.

• Plotting hygiene:
  - We DO NOT cap the top at 1.0 (that can change apparent shapes). We only apply a small bottom floor for log plots.
  - VULCAN zeros are plotted as gaps (NaN), not masked to tiny.

Units: P to model in barye (1 Pa = 10 barye). Titles show both.
"""

from __future__ import annotations
import os, sys, json, pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

# ---------------- Paths / env ----------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "1"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed"))

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Repo import
sys.path.append(str((REPO_ROOT / "src").resolve()))
from normalizer import NormalizationHelper  # type: ignore

# ---------------- Constants ----------------
viridis = matplotlib.colormaps.get_cmap("viridis")
He_mixing_ratio = 1.679000e-01     # << He fraction you want to "add back"
HE_SCALE = 1.0 - He_mixing_ratio   # = 0.8321

# 13 target species (model order)
TARGET_SPECIES: List[str] = [
    "C2H2_evolution","CH3_evolution","CH4_evolution","CO2_evolution","CO_evolution",
    "H2O_evolution","H2_evolution","HCN_evolution","H_evolution","N2_evolution",
    "NH3_evolution","OH_evolution","O_evolution",
]
BASE_NAMES = [s[:-10] if s.endswith("_evolution") else s for s in TARGET_SPECIES]
BASE_TO_IDX = {BASE_NAMES[i]: i for i in range(len(TARGET_SPECIES))}

# Case 2 (NO He) -> Panel 2 model y0
CASE2_Y0_MAP: Dict[str, float] = {
    "C2H2_evolution": 8.913668e-10, "CH3_evolution": 1.034118e-08, "CH4_evolution": 1.168895e-11,
    "CO2_evolution": 3.010446e-05, "CO_evolution": 4.093728e-04, "H2O_evolution": 2.481025e-07,
    "H2_evolution": 9.886430e-01, "HCN_evolution": 5.410422e-12, "H_evolution": 6.946361e-07,
    "N2_evolution": 2.452540e-10, "NH3_evolution": 1.018076e-03, "OH_evolution": 9.898477e-03,
    "O_evolution": 5.335861e-11,
}
CASE2_P_PA = 68_188_104.0
CASE2_T_K  = 1625.354248046875

# Panel 1: VULCAN overlay path & species shown
VULCAN_PATH = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar/vul-T1000KlogP3.0-NCHO-solar_hot_ini.vul")
PLOT_SPEC = ['H2O','CH4','CO','CO2','NH3','HCN','N2','C2H2']  # subset drawn on Panel 1

# Panel 1 thermodynamics (Pa -> barye for model)
CASE1_P_PA = 100.0      # 1 mbar
CASE1_T_K  = 1000.0

# Time ranges
# Panel 1: model 1e0..1e8; VULCAN 1e0..1e18; xlim 10**-0.1..1e18
T1_MODEL_MIN, T1_MODEL_MAX = 1e0, 1e8
T1_AXIS_MIN,  T1_AXIS_MAX  = 10**(-0.1), 1e18
# Panel 2: model 1e-3..1e8
T2_MODEL_MIN, T2_MODEL_MAX = 1e-3, 1e8

# ---------------- Helpers ----------------
def _suffix(n: str) -> str:
    return n if n.endswith("_evolution") else f"{n}_evolution"

def pa_to_barye(p_pa: float) -> float:
    return float(p_pa) * 10.0

def _floor_min(arr: np.ndarray, minv: float = 1e-30) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[out < minv] = minv
    return out

def _floor_only_for_plot(y: np.ndarray, minv: float = 1e-30) -> np.ndarray:
    # Floor bottom for log-plot, don't cap top. Guard NaNs/±inf.
    y = np.nan_to_num(y, nan=minv, posinf=np.nanmax(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 1.0, neginf=minv)
    return np.maximum(y, minv)

def _load_manifest_and_model() -> tuple[NormalizationHelper, list[str], torch.nn.Module]:
    man_path = PROCESSED_DIR / "normalization.json"
    if not man_path.exists():
        raise FileNotFoundError(f"normalization.json not found: {man_path}")
    with open(man_path, "r") as f:
        manifest = json.load(f)
    norm = NormalizationHelper(manifest)
    meta = manifest.get("meta") or {}
    globals_v = list(meta.get("global_variables") or [])
    if not globals_v:
        raise RuntimeError("global_variables missing from normalization.json meta")
    from torch.export import load as torch_load
    tried = []
    for name in ("complete_model_exported_k1.pt2","complete_model_exported_k1_int8.pt2",
                 "complete_model_exported.pt2","complete_model_exported_int8.pt2"):
        p = MODEL_DIR / name
        tried.append(str(p))
        if p.exists():
            print(f"[INFO] Using exported model: {p}")
            mod = torch_load(str(p)).module()
            return norm, globals_v, mod
    raise FileNotFoundError("No exported K=1 model found. Tried:\n  " + "\n  ".join(tried))

def _load_vulcan_series(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"VULCAN pickle not found:\n{path}\n"
            "Fix VULCAN_PATH at the top of this script."
        )
    with open(path, "rb") as h:
        data = pickle.load(h)
    raw_species = data['variable']['species']
    species = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in raw_species]
    t_time = np.asarray(data['variable']['t_time'], dtype=float)     # [T]
    y_time = np.asarray(data['variable']['y_time'], dtype=float)     # [T, layer, S]
    if y_time.ndim != 3 or y_time.shape[1] < 1:
        raise ValueError("Unexpected VULCAN y_time shape (want [T, layer, S] with layer>=1)")
    total_all = np.maximum(y_time[:, 0, :].sum(axis=-1).astype(float), 1e-300)  # denom includes He
    y_ratio = y_time[:, 0, :] / total_all[:, None]                   # [T,S] mixing ratio incl He in denom
    return t_time, species, y_ratio

def _globals_barye(globals_v: list[str], P_barye: float, T_K: float) -> np.ndarray:
    g = np.zeros((1, len(globals_v)), dtype=np.float32)
    for i, name in enumerate(globals_v):
        n = name.strip().lower()
        if n.startswith("t"): g[0, i] = float(T_K)
        elif n.startswith("p"): g[0, i] = float(P_barye)
        else: g[0, i] = 0.0
    return g

def _make_y0_noHe_from_map(name_to_val: Dict[str, float], debug_tag: str) -> np.ndarray:
    """
    name_to_val: keys in TARGET_SPECIES form (e.g., 'H2O_evolution'); values are mixing ratios (with-He source OK).
    Returns y0_noHe[TARGET_SPECIES] after: fill-missing -> floor<1e-25 -> renorm to sum=1 across 13 targets.
    """
    print(f"\n[DEBUG] {_make_y0_noHe_from_map.__name__} | {debug_tag}")
    raw = np.array([name_to_val.get(name, 1e-25) for name in TARGET_SPECIES], dtype=np.float64)
    print(f"  raw y0 (first 10): {raw[:10]}")
    raw = np.where(raw < 1e-25, 1e-25, raw)
    print(f"  floored to >=1e-25 (first 10): {raw[:10]}")
    s_raw = float(raw.sum()); print(f"  sum(raw,floored) = {s_raw:.6e}")
    if not np.isfinite(s_raw) or s_raw <= 0: raise RuntimeError("Bad raw sum after floor.")
    y0_noHe = raw / s_raw
    print(f"  y0_noHe (renormed) first 10: {y0_noHe[:10]}")
    print(f"  sum(y0_noHe) = {float(y0_noHe.sum()):.6e} (should be 1)")
    assert np.isclose(y0_noHe.sum(), 1.0, rtol=1e-6, atol=1e-9)
    return y0_noHe.astype(np.float32)

@torch.inference_mode()
def _predict_series_k1(fn: torch.nn.Module, y0n: torch.Tensor, gn: torch.Tensor, t_hat: torch.Tensor) -> np.ndarray:
    preds = []
    for k in range(int(t_hat.numel())):
        out_n = fn(y0n, gn, t_hat[k:k+1])    # scalar time as shape [1]
        if not isinstance(out_n, torch.Tensor):
            out_n = torch.as_tensor(out_n)
        preds.append(out_n.reshape(1, -1))
    out = torch.cat(preds, dim=0).cpu().numpy()  # [K,S]
    return out

# ---------------- Main ----------------
def main() -> None:
    norm, globals_v, fn = _load_manifest_and_model()
    eps = 1e-25 if not hasattr(norm, "epsilon") else float(norm.epsilon)
    print(f"\n[DEBUG] epsilon used for time normalization clamp = {eps:g}")
    print(f"[DEBUG] globals_v = {globals_v}")
    print(f"[DEBUG] target species ({len(TARGET_SPECIES)}): {TARGET_SPECIES}")

    # ===== Panel 1: VULCAN-driven initial conditions (with-He source → 13-target no-He y0) =====
    t_vul, sp_vul, y_vul = _load_vulcan_series(VULCAN_PATH)
    print(f"\n[DEBUG] VULCAN: times range [{t_vul.min():.3e}, {t_vul.max():.3e}]  #points={t_vul.size}")
    print(f"[DEBUG] VULCAN species count={len(sp_vul)} (first 10) -> {sp_vul[:10]}")

    # Use first index with t >= 1 for initial snapshot
    mask_v = (t_vul >= 1.0)
    if not np.any(mask_v):
        raise RuntimeError("VULCAN file has no times ≥ 1e0; cannot set initial from VULCAN.")
    i0_vul = int(np.argmax(mask_v))
    print(f"[DEBUG] Panel1 start index in VULCAN = {i0_vul}, t_start={t_vul[i0_vul]:.3e}s  (>=1)")

    # Build with-He map at t_start for all targets
    full_withHe_target_map: Dict[str, float] = {}
    for base in BASE_NAMES:
        full_withHe_target_map[_suffix(base)] = float(y_vul[i0_vul, sp_vul.index(base)]) if base in sp_vul else 1e-25

    print("\n[DEBUG] Panel1 VULCAN with-He (selected bases):")
    for b in PLOT_SPEC:
        val = float(y_vul[i0_vul, sp_vul.index(b)]) if b in sp_vul else 1e-25
        print(f"  {b:>6s}: {val:.6e}")

    # Model input for Panel 1: renorm across targets (NO He)
    y0_1_noHe = _make_y0_noHe_from_map(full_withHe_target_map, "Panel1 from VULCAN (with-He source)")
    # Hollow-circle markers for Panel 1: with-He initial values (floored only for plotting)
    y0_1_withHe = _floor_min(np.array([full_withHe_target_map[name] for name in TARGET_SPECIES], dtype=np.float64), 1e-30)

    # Panel 1 model time grid
    t1_model = np.logspace(np.log10(T1_MODEL_MIN), np.log10(T1_MODEL_MAX), 99, dtype=np.float64)
    t1_hat = norm.normalize_dt_from_phys(torch.from_numpy(np.clip(t1_model, eps, None).astype(np.float32)))

    # Globals for Panel 1 (barye)
    g1 = _globals_barye(globals_v, pa_to_barye(CASE1_P_PA), CASE1_T_K)
    g1n = norm.normalize(torch.from_numpy(g1), globals_v)
    y0_1n = norm.normalize(torch.from_numpy(y0_1_noHe.reshape(1, -1)), TARGET_SPECIES)

    # Predict Panel 1
    ypred1_n = _predict_series_k1(fn, y0_1n, g1n, t1_hat)
    ypred1   = norm.denormalize(torch.from_numpy(ypred1_n), TARGET_SPECIES).numpy()
    print(f"[DEBUG] Panel1 pred stats: min={np.nanmin(ypred1):.3e}, max={np.nanmax(ypred1):.3e}, NaN? {np.isnan(ypred1).any()}")

    # ===== Panel 2: Case-2 (manual NO He) =====
    case2_map = {name: CASE2_Y0_MAP.get(name, 1e-25) for name in TARGET_SPECIES}
    y0_2_noHe = _make_y0_noHe_from_map(case2_map, "Panel2 manual sample (no He)")

    # Panel 2 time grid
    t2_model = np.logspace(np.log10(T2_MODEL_MIN), np.log10(T2_MODEL_MAX), 99, dtype=np.float64)
    t2_hat = norm.normalize_dt_from_phys(torch.from_numpy(np.clip(t2_model, eps, None).astype(np.float32)))

    # Globals for Panel 2 (barye)
    g2 = _globals_barye(globals_v, pa_to_barye(CASE2_P_PA), CASE2_T_K)
    g2n = norm.normalize(torch.from_numpy(g2), globals_v)
    y0_2n = norm.normalize(torch.from_numpy(y0_2_noHe.reshape(1, -1)), TARGET_SPECIES)

    # Predict Panel 2
    ypred2_n = _predict_series_k1(fn, y0_2n, g2n, t2_hat)
    ypred2   = norm.denormalize(torch.from_numpy(ypred2_n), TARGET_SPECIES).numpy()
    print(f"[DEBUG] Panel2 pred stats (pre-He-scale): min={np.nanmin(ypred2):.3e}, max={np.nanmax(ypred2):.3e}, NaN? {np.isnan(ypred2).any()}")

    # ===== Constant He scaling (Panel 2 ONLY, shape-preserving) =====
    ypred2_scaled = (ypred2 * HE_SCALE).astype(np.float64)

    # Ratio check: should be constant HE_SCALE wherever ypred2 > 0
    mask_pos = ypred2 > 0
    ratio = np.full_like(ypred2_scaled, np.nan, dtype=np.float64)
    ratio[mask_pos] = ypred2_scaled[mask_pos] / ypred2[mask_pos]
    rmin = np.nanmin(ratio); rmax = np.nanmax(ratio)
    print(f"[DEBUG] Panel2 scale ratio (after/before) min={rmin:.6f} max={rmax:.6f} (expect {HE_SCALE:.6f})")

    # ===== Plot prep =====
    # VULCAN plotted (t >= 1.0); zeros -> NaN (no tiny masking)
    mask_v_plot = (t_vul >= 1.0)
    t_vul_plot = t_vul[mask_v_plot]
    y_vul_plot = y_vul[mask_v_plot, :].copy()
    y_vul_plot[y_vul_plot <= 0.0] = np.nan

    # Floor-only for model curves to survive log-plot; NO top cap
    ypred1_plot = _floor_only_for_plot(ypred1, 1e-30)
    ypred2_plot = _floor_only_for_plot(ypred2_scaled, 1e-30)

    # ===== Figure =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)

    # Panel 1 (subset PLOT_SPEC; model dashed, VULCAN solid)
    color_by_base_subset = {sp: viridis(k / max(len(PLOT_SPEC) - 1, 1)) for k, sp in enumerate(PLOT_SPEC)}
    model_entries_legend = []
    vul_entries_legend   = []

    for base in PLOT_SPEC:
        if base not in BASE_TO_IDX:
            print(f"[WARN] Model missing '{base}', skipping in panel 1 model.")
            continue
        i = BASE_TO_IDX[base]
        col = color_by_base_subset[base]

        # Model (dashed)
        ax1.loglog(t1_model, ypred1_plot[:, i], ls='--', lw=1.9, color=col, zorder=3)

        # Hollow circle at first model time: with-He initial (floored only)
        ax1.loglog([t1_model[0]], [max(float(y0_1_withHe[i]), 1e-30)],
                   marker='o', mfc='none', mec=col, mew=1.6, ms=7, zorder=4)

        model_entries_legend.append((f"{base} (model)", col, '--', float(y0_1_noHe[i])))

        # VULCAN solid
        if base in sp_vul:
            j = sp_vul.index(base)
            ax1.loglog(t_vul_plot, y_vul_plot[:, j], ls='-', lw=2.1, color=col, zorder=2)
            v0 = float(y_vul[i0_vul, j]) if np.isfinite(y_vul[i0_vul, j]) else 0.0
            vul_entries_legend.append((f"{base} (VULCAN)", col, '-', v0))
        else:
            print(f"[WARN] VULCAN missing '{base}', skipping overlay.")

    ax1.set_title(
        f"VULCAN vs Model @ T={CASE1_T_K:.0f} K, P={CASE1_P_PA:.0f} Pa "
        f"({pa_to_barye(CASE1_P_PA):.0f} barye fed to model)",
        fontsize=13
    )
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mixing Ratio")
    ax1.set_xlim(T1_AXIS_MIN, T1_AXIS_MAX)
    ax1.axvline(1.0, color='0.85', lw=1.0, zorder=1)
    ax1.grid(False)

    # Legends: model start (no He) and VULCAN start (with He)
    model_entries_legend = sorted(model_entries_legend, key=lambda t: -t[3])
    handles_m = [Line2D([0],[0], color=col, lw=2.0, ls='--') for (_, col, _, _) in model_entries_legend]
    labels_m  = [f"{label}: {val:.2e}" for (label, _, _, val) in model_entries_legend]
    leg1 = ax1.legend(handles_m, labels_m, loc='upper right', fontsize=8,
                      title="Model (dashed), start mix (no He)", frameon=False)
    ax1.add_artist(leg1)

    vul_entries_legend = sorted(vul_entries_legend, key=lambda t: -t[3])
    handles_v = [Line2D([0],[0], color=col, lw=2.1, ls='-') for (_, col, _, _) in vul_entries_legend]
    labels_v  = [f"{label}: {val:.2e}" for (label, _, _, val) in vul_entries_legend]
    ax1.legend(handles_v, labels_v, loc='lower left', fontsize=8,
               title="VULCAN (solid), start mix (with He)", frameon=False)

    # Panel 2 (ALL species; model dashed, with-He = constant scale)
    colors_all = [viridis(i / max(len(TARGET_SPECIES) - 1, 1)) for i in range(len(TARGET_SPECIES))]
    model2_entries_legend = []

    for i, base in enumerate(BASE_NAMES):
        col = colors_all[i]
        ax2.loglog(t2_model, ypred2_plot[:, i], ls='--', lw=1.9, color=col, zorder=3)
        # Hollow circle at first model time (with-He initial = no-He * HE_SCALE)
        ax2.loglog([t2_model[0]], [max(float(y0_2_noHe[i] * HE_SCALE), 1e-30)],
                   marker='o', mfc='none', mec=col, mew=1.4, ms=6, zorder=4)
        model2_entries_legend.append((f"{base} (model ×{HE_SCALE:.3f})", col, '--', float(y0_2_noHe[i] * HE_SCALE)))

    ax2.set_title(
        f"Model Only (×{HE_SCALE:.3f} to add He={He_mixing_ratio:.3f}) @ T={CASE2_T_K:.0f} K, "
        f"P={CASE2_P_PA:.1e} Pa ({pa_to_barye(CASE2_P_PA):.0f} barye)",
        fontsize=13
    )
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mixing Ratio")
    ax2.set_xlim(T2_MODEL_MIN, T2_MODEL_MAX)
    ax2.axvline(T2_MODEL_MIN, color='0.85', lw=1.0, zorder=1)
    ax2.grid(False)

    model2_entries_legend = sorted(model2_entries_legend, key=lambda t: -t[3])
    handles2 = [Line2D([0],[0], color=col, lw=2.0, ls='--') for (_, col, _, _) in model2_entries_legend]
    labels2  = [f"{label}: {val:.2e}" for (label, _, _, val) in model2_entries_legend]
    ax2.legend(handles2, labels2, loc='best', fontsize=8,
               title=f"Model (dashed), start mix with He={He_mixing_ratio:.3f}", frameon=False)

    out_png = MODEL_DIR / "plots" / "predictions_dual_fixedHe_shape_preserved.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Plot saved to {out_png}")

if __name__ == "__main__":
    main()
