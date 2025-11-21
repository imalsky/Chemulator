#!/usr/bin/env python3
"""
Single-Shot Chemistry Prediction Script (No Interpolation, Multi-Profile)
=========================================================================
Generates *separate* plots for each selected T/P profile.

Key behavior:
- No interpolation anywhere.
- VULCAN is plotted on ABSOLUTE time (the full original profile).
- T0 ONLY selects the ML anchor: we snap to the nearest VULCAN sample at T0,
  feed that state into the ML model, and predict at absolute times t = t_anchor + Δt.
- ML predictions are evaluated on Δt ∈ [1e-3, 1e8] s (log-spaced) relative to the anchor.

Edit PROFILES_TO_PLOT and T0 below as needed.
"""
from __future__ import annotations

import json, math, os, pickle, sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

plt.style.use("science.mplstyle")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ============================================================================
# Profiles
# ============================================================================
# Valid options: "2000K_1bar", "2000K_1mbar", "1000K_1bar", "1000K_1mbar"
PROFILES: Dict[str, Dict[str, float]] = {
    "2000K_1bar":  {"T_K": 2000.0, "P_bar": 1.0},
    "2000K_1mbar": {"T_K": 2000.0, "P_bar": 1e-3},
    "1000K_1bar":  {"T_K": 1000.0, "P_bar": 1.0},
    "1000K_1mbar": {"T_K": 1000.0, "P_bar": 1e-3},
}

# Choose which profiles to render (separate figure per profile)
PROFILES_TO_PLOT: List[str] = [
    "1000K_1mbar",
    "1000K_1bar",
    "2000K_1mbar",
    "2000K_1bar",
]

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path("/Users/imalsky/Desktop/Chemulator")
SRC_DIR = REPO_ROOT / "src"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models" / "quarter"
VULCAN_DIR = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar")
OUT_DIR = MODEL_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Time / Plot Config
# ============================================================================

T0 = 1.0e5           # Requested anchor time (seconds). Controls ML anchor ONLY.
DT_MIN = 1e-3         # ML Δt range (relative to anchor)
DT_MAX = 1e8
K_POINTS = 50.0       # number of Δt samples (float for easy edits; cast to int)

YMAX = 2
YMIN = 1e-30
XMIN = 1e-3
XMAX = 1e18

# ============================================================================
# Species / Data Config
# ============================================================================

FEED_MIN = 1.0e-30
PLOT_SPECIES: List[str] = ['H2', 'H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2', 'C2H2', 'H', 'CH3', 'OH', 'O']
PLOT_SPECIES: List[str] = ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2', 'C2H2']


PLOT_FLOOR = 1.0e-30

# ============================================================================
# Python Path
# ============================================================================

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from normalizer import NormalizationHelper  # type: ignore


# ============================================================================
# Helpers
# ============================================================================

def load_manifest_and_normalizer() -> Tuple[dict, NormalizationHelper, List[str], List[str]]:
    manifest = json.loads((PROCESSED_DIR / "normalization.json").read_text())
    norm = NormalizationHelper(manifest)
    meta = manifest.get("meta", {})

    in_names = list(meta.get("species_variables") or manifest.get("species_variables") or [])
    assert in_names, "species_variables missing in normalization.json"
    in_bases = [n[:-10] if n.endswith("_evolution") else n for n in in_names]

    gvars = list(meta.get("global_variables") or manifest.get("global_variables") or [])
    return manifest, norm, in_names, gvars


def load_model() -> Tuple[torch.nn.Module, torch.device, bool]:
    """Load exported model once; prefer AOTI MPS, then PT2 variants."""
    fn, dev, use_dyn = None, torch.device("cpu"), True

    aoti_dir = MODEL_DIR / "export_k_dyn_mps.aoti"
    if torch.backends.mps.is_available() and aoti_dir.exists():
        try:
            from torch._inductor import aot_load_package
            fn = aot_load_package(str(aoti_dir))
            dev = torch.device("mps")
            print(f"[INFO] Using AOTI MPS: {aoti_dir}")
        except Exception as e:
            print(f"[WARN] AOTI load failed: {e}  → fallback to .pt2")

    if fn is None:
        from torch.export import load as torch_export_load
        pref = [
            "export_k_dyn_mps.pt2", "export_k_dyn_gpu.pt2", "export_k_dyn_cpu.pt2",
            "export_k1_mps.pt2",    "export_k1_gpu.pt2",    "export_k1_cpu.pt2",
            "complete_model_exported_k1.pt2", "complete_model_exported.pt2",
        ]
        pt2 = next((MODEL_DIR / p for p in pref if (MODEL_DIR / p).exists()), None)
        if pt2 is None:
            cands = sorted(MODEL_DIR.glob("*.pt2"), key=lambda p: p.stat().st_mtime, reverse=True)
            pt2 = cands[0] if cands else None
        assert pt2 is not None, "No exported model found (.aoti or .pt2)"

        ep = torch_export_load(str(pt2))
        fn = ep.module()
        n = pt2.name.lower()
        dev = (torch.device("mps") if ("mps" in n and torch.backends.mps.is_available())
               else (torch.device("cuda") if (("gpu" in n or "cuda" in n) and torch.cuda.is_available())
                     else torch.device("cpu")))
        use_dyn = ("dyn" in n)
        print(f"[INFO] Using PT2: {pt2.name} → device={dev.type} dynBK={use_dyn}")

    return fn, dev, use_dyn


def profile_to_vulcan_path(profile: str) -> Path:
    assert profile in PROFILES, f"Unknown PROFILE={profile}"
    T_K = float(PROFILES[profile]["T_K"])
    P_bar = float(PROFILES[profile]["P_bar"])
    barye = P_bar * 1e6
    logP = math.log10(barye)
    return VULCAN_DIR / f"vul-T{int(T_K)}KlogP{logP:.1f}-NCHO-solar_hot_ini.vul"


def load_vulcan(profile: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int], float, float]:
    vul_path = profile_to_vulcan_path(profile)
    assert vul_path.exists(), f"Missing VULCAN file: {vul_path}"
    with open(vul_path, "rb") as h:
        d = pickle.load(h)

    t_all = np.asarray(d["variable"]["t_time"], float)      # [T]
    Y     = np.asarray(d["variable"]["y_time"], float)      # [T, layer, S]
    names = list(d["variable"]["species"])
    name_to_idx = {n: i for i, n in enumerate(names)}

    # mixing ratios at layer 0
    den = np.maximum(Y[:, 0, :].sum(-1), 1e-30)
    MR_all = Y[:, 0, :] / den[:, None]  # [T, S]

    T_K = float(PROFILES[profile]["T_K"])
    P_bar = float(PROFILES[profile]["P_bar"])
    barye = P_bar * 1e6
    return t_all, MR_all, names, name_to_idx, T_K, barye


def make_gvars_tensor(gvars: List[str], T_K: float, barye: float, norm: NormalizationHelper) -> torch.Tensor:
    if not gvars:
        return torch.zeros(1, 0)
    g_phys = np.zeros((1, len(gvars)), np.float32)
    for i, nm in enumerate(gvars):
        nm_l = nm.strip().lower()
        g_phys[0, i] = (barye if nm_l.startswith("p") else (T_K if nm_l.startswith("t") else 0.0))
    return norm.normalize(torch.from_numpy(g_phys), gvars).float()


def prepare_anchor(T0_phys: float,
                   MR_all: np.ndarray, t_all: np.ndarray, in_bases: List[str],
                   norm: NormalizationHelper, in_names: List[str],
                   name_to_idx: Dict[str, int]) -> Tuple[int, float, np.ndarray, torch.Tensor]:
    """
    Pick the anchor by snapping to the nearest VULCAN sample to T0_phys
    and return the model-ready normalized state.
    """
    # nearest-neighbor snap in *physical* time
    idx_anchor = int(np.argmin(np.abs(t_all - T0_phys)))
    t_anchor = float(t_all[idx_anchor])

    MR_T0 = MR_all[idx_anchor, :]  # exact sample at snapped time

    # Map VULCAN species to model input order
    y0_inputs = np.array([max(MR_T0[name_to_idx[b]], 0.0) if b in name_to_idx else 0.0
                          for b in in_bases], dtype=float)

    # effective floor via roundtrip zeros
    eff = norm.denormalize(
        norm.normalize(torch.zeros(1, len(in_names)), in_names), in_names
    ).numpy().reshape(-1)
    floor = np.maximum(np.nan_to_num(eff, nan=0.0), FEED_MIN)
    v = np.maximum(y0_inputs, floor)
    s = v.sum()
    y0_simplex = (np.full_like(v, 1.0 / len(v)) if (not np.isfinite(s) or s <= 0)
                  else v / s).astype(np.float32)

    y0_norm = norm.normalize(torch.from_numpy(y0_simplex[None, :]), in_names).float()
    return idx_anchor, t_anchor, y0_simplex, y0_norm


def run_model(fn: torch.nn.Module, dev: torch.device, use_dyn: bool,
              y0_norm: torch.Tensor, g_norm: torch.Tensor,
              dt_pred: np.ndarray, norm: NormalizationHelper) -> torch.Tensor:
    K = len(dt_pred)
    dt_hat = norm.normalize_dt_from_phys(torch.from_numpy(dt_pred)).view(-1, 1).float()

    y0_norm_d = y0_norm.to(dev)
    g_norm_d = g_norm.to(dev)

    # BK signature: [1, K, S], K1 signature: [K, S]
    state_bk = y0_norm_d.expand(1, K, -1)
    dt_bk = dt_hat.to(dev).view(1, K, 1)
    g_bk = (g_norm_d.expand(1, K, -1) if g_norm_d.numel() else torch.empty(1, K, 0, device=dev))

    state_k1 = y0_norm_d.repeat(K, 1)
    dt_k1 = dt_hat.to(dev)
    g_k1 = (g_norm_d.repeat(K, 1) if g_norm_d.numel() else torch.empty(K, 0, device=dev))

    y_pred_norm = None
    for mode in (["BK", "K1"] if use_dyn else ["K1", "BK"]):
        try:
            with torch.inference_mode():
                if mode == "BK":
                    out = fn(state_bk, dt_bk, g_bk)
                    y_pred_norm = (out[0] if (isinstance(out, torch.Tensor) and out.dim() == 3 and out.size(0) == 1)
                                   else out).to("cpu")
                else:
                    out = fn(state_k1, dt_k1, g_k1)
                    y_pred_norm = (out[:, 0, :] if (isinstance(out, torch.Tensor) and out.dim() == 3 and out.size(1) == 1)
                                   else out).to("cpu")
            break
        except Exception as e:
            print(f"[note] {mode} call failed: {e}")
    assert y_pred_norm is not None, "Failed to run model with either BK or K1 signature"
    return y_pred_norm


def denorm_and_align(y_pred_norm: torch.Tensor, norm: NormalizationHelper,
                     in_names: List[str], meta: dict,
                     out_species_in_vulcan: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    S_out = int(y_pred_norm.shape[-1])

    cand = list(meta.get("target_species_variables") or norm.manifest.get("target_species_variables") or [])
    out_names = (cand if (cand and len(cand) == S_out)
                 else (in_names if len(in_names) == S_out else in_names[:S_out]))
    out_bases = [n[:-10] if n.endswith("_evolution") else n for n in out_names]

    # Denormalize to physical mixing ratios
    pred_phys = norm.denormalize(y_pred_norm, out_names).cpu().numpy()

    # Keep only species present in VULCAN
    present = [b for b in out_bases if b in out_species_in_vulcan]
    assert present, "No overlap between outputs and VULCAN species"

    mask = [b in present for b in out_bases]
    pred_sub = np.clip(pred_phys[:, np.where(mask)[0]], 1e-300, None)
    pred_sub /= np.maximum(pred_sub.sum(1, keepdims=True), 1e-30)

    return pred_sub, present


def min_positive(*arrays: np.ndarray, default: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        if a.size:
            ap = a[a > 0]
            if ap.size:
                vals.append(float(ap.min()))
    return min(vals) if vals else default


def plot_one_profile(profile: str,
                     manifest: dict, norm: NormalizationHelper, in_names: List[str], gvars: List[str],
                     fn: torch.nn.Module, dev: torch.device, use_dyn: bool,
                     in_bases: List[str], T0_phys: float) -> Path:
    # Load VULCAN (original samples only)
    t_all, MR_all, names, name_to_idx, T_K, barye = load_vulcan(profile)

    # Anchor and model inputs — T0_phys controls where we anchor (ML only)
    idx_anchor, t_anchor, y0_simplex, y0_norm = prepare_anchor(
        T0_phys, MR_all, t_all, in_bases, norm, in_names, name_to_idx
    )

    # Global variables
    g_norm = make_gvars_tensor(gvars, T_K, barye, norm)

    # Fixed Δt prediction grid (always same physical Δt range, relative to anchor)
    K = int(K_POINTS)
    dt_pred = np.logspace(math.log10(DT_MIN), math.log10(DT_MAX), K, dtype=np.float32)
    t_pred_abs = t_anchor + dt_pred  # ABSOLUTE prediction times for plotting

    # Inference from the anchor state
    y_pred_norm = run_model(fn, dev, use_dyn, y0_norm, g_norm, dt_pred, norm)

    # Align species, no interpolation
    meta = manifest.get("meta", {})
    pred_sub, present = denorm_and_align(y_pred_norm, norm, in_names, meta, name_to_idx)

    # Truth restricted to 'present' species (ABSOLUTE time), renormalized over that set
    idx = np.array([name_to_idx[b] for b in present], dtype=int)
    truth_full = np.clip(MR_all[:, idx], 1e-300, None)
    truth_full /= np.maximum(truth_full.sum(1, keepdims=True), 1e-30)

    # Species selection
    keep = [i for i, b in enumerate(present) if b in PLOT_SPECIES] or list(range(len(present)))
    labels = [present[i] for i in keep]
    truth_plot = truth_full[:, keep]
    pred_plot = np.clip(pred_sub[:, keep], PLOT_FLOOR, None)

    # Anchor markers for those species (from y0_simplex) at ABSOLUTE time t_anchor
    in_idx_by_base = {b: j for j, b in enumerate(in_bases)}
    y0_on_outputs = np.array([(y0_simplex[in_idx_by_base[b]] if b in in_idx_by_base else np.nan)
                              for b in present])[keep]

    # Filter non-positive times for log x-axis (VULCAN absolute)
    mask_pos = t_all > 0
    t_abs_vulcan = t_all[mask_pos]
    truth_plot = truth_plot[mask_pos, :]

    # Determine x-limits from union of VULCAN and prediction absolute times
    xmin = max(min_positive(t_abs_vulcan, t_pred_abs), XMIN)
    xmax = float(max(t_abs_vulcan.max() if t_abs_vulcan.size else DT_MAX,
                     t_pred_abs.max() if t_pred_abs.size else DT_MAX))
    xmax = min(XMAX, xmax)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Abudnance")

    colors = plt.cm.tab20(np.linspace(0, 0.95, len(keep)))

    # VULCAN truth (solid lines; ABSOLUTE time, original samples, no interpolation)
    for i, c in enumerate(colors):
        ax.plot(t_abs_vulcan, np.clip(truth_plot[:, i], PLOT_FLOOR, None),
                '-', lw=1.8, alpha=0.95, color=c)

    # Anchor (hollow squares) at ABSOLUTE time = t_anchor
    #for i, c in enumerate(colors):
    #    if np.isfinite(y0_on_outputs[i]) and t_anchor > 0:
    #        ax.plot([t_anchor], [max(y0_on_outputs[i], PLOT_FLOOR)],
    #                marker='s', mfc='none', mec=c, ms=6, mew=1.2, ls='none')

    # Predictions (hollow circles) at ABSOLUTE times t = t_anchor + Δt
    for i, c in enumerate(colors):
        ax.plot(t_pred_abs, pred_plot[:, i], ls='none', marker='o',
                mfc='none', mec=c, ms=4.0, mew=1.0, alpha=0.95)

    # Legends
    order = np.argsort(np.max(truth_plot, axis=0))[::-1]
    species_handles = [Line2D([0], [0], color=colors[i], lw=2.0) for i in order]
    species_labels = [labels[i] for i in order]

    # Species-only legend (names only)
    #ax.legend(species_handles, species_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.legend(species_handles, species_labels, loc='best')


    fig.tight_layout()
    out_png = OUT_DIR / f"single_shot_{profile}_abs_time_T0_{T0_phys:.3e}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {profile} → requested T0={T0_phys:.6g} s, anchor t={t_anchor:.6g} s (idx {idx_anchor}); "
          f"VULCAN samples plotted={t_abs_vulcan.size}; out={out_png}")
    return out_png


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    torch.set_num_threads(1)

    # Load once
    manifest, norm, in_names, gvars = load_manifest_and_normalizer()
    in_bases = [n[:-10] if n.endswith("_evolution") else n for n in in_names]
    fn, dev, use_dyn = load_model()

    # Fixed prediction grid setup note
    print(f"[INFO] ML Δt grid = [{DT_MIN:.1e}, {DT_MAX:.1e}] s with K={int(K_POINTS)}")
    print(f"[INFO] Requested anchor T0 = {T0:.6g} s (snap per profile; VULCAN plotted on ABSOLUTE time)")

    # Render each requested profile
    for profile in PROFILES_TO_PLOT:
        assert profile in PROFILES, f"Unknown profile in PROFILES_TO_PLOT: {profile}"
        plot_one_profile(profile, manifest, norm, in_names, gvars, fn, dev, use_dyn, in_bases, T0)


if __name__ == "__main__":
    main()
