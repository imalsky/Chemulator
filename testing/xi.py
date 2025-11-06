#!/usr/bin/env python3
"""
Single-Shot Chemistry Prediction Script
========================================
Makes one-step predictions at various time deltas from an anchor point,
comparing model predictions against VULCAN ground truth data.

This script:
1. Loads VULCAN chemistry simulation data for a chosen T/P profile
2. Sets an anchor point at time T0
3. Uses an exported FlowMap model to predict chemistry at T0 + Δt
4. Compares predictions to ground truth via linear interpolation
5. Generates a comparison plot
"""
from __future__ import annotations

import json, math, os, pickle, sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

plt.style.use("science.mplstyle")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ============================================================================
# Profile Selection
# ============================================================================
# Pick exactly one temperature/pressure profile to test
# Valid options: "2000K_1bar", "2000K_1mbar", "1000K_1bar", "1000K_1mbar"
#
# VULCAN filename format: vul-T{T}KlogP{log10(barye):.1f}-NCHO-solar_hot_ini.vul
# Pressure conversions: 1 bar = 1e6 barye → logP=6.0; 1 mbar = 1e3 barye → logP=3.0

PROFILE = "2000K_1bar"

PROFILES: Dict[str, Dict[str, float]] = {
    "2000K_1bar": {"T_K": 2000.0, "P_bar": 1.0},
    "2000K_1mbar": {"T_K": 2000.0, "P_bar": 1e-3},
    "1000K_1bar": {"T_K": 1000.0, "P_bar": 1.0},
    "1000K_1mbar": {"T_K": 1000.0, "P_bar": 1e-3},
}

# ============================================================================
# Path Configuration
# ============================================================================

REPO_ROOT = Path("/Users/imalsky/Desktop/Chemulator")
SRC_DIR = REPO_ROOT / "src"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models" / "delta"
VULCAN_DIR = Path("/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar")

# ============================================================================
# Time Configuration
# ============================================================================

# CHANGED: split scalars; keep as floats for easy editing
T0 = 1.0e-3
T_FINAL = T0 + 1.0e8

DT_MIN = 1e-3
K_POINTS = 50.0  # float by request; cast when used

# Plot x-axis limits for delta-t (floats, separate lines)
XMIN_DT = DT_MIN
XMAX_DT = 1.0e18

YMAX = 2
YMIN = 1e-30

# ============================================================================
# Data & Plotting Configuration
# ============================================================================

FEED_MIN = 1.0e-15  # Minimum abundance floor when feeding to model

# Species to plot (subset of all species for clarity)
PLOT_SPECIES: List[str] = ['H2', 'H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2', 'C2H2', 'H', 'CH3', 'OH', 'O']
PLOT_FLOOR = 1.0e-30  # Minimum abundance floor for plotting (y-axis)

# ============================================================================
# Setup Python Path
# ============================================================================

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from normalizer import NormalizationHelper  # type: ignore


# ============================================================================
# Main Script
# ============================================================================

def main() -> None:
    """Main execution: load data, run model, compare to truth, and plot."""
    torch.set_num_threads(1)

    # ========================================================================
    # Step 1: Profile Configuration
    # ========================================================================

    assert PROFILE in PROFILES, f"Unknown PROFILE={PROFILE}"
    T_K = float(PROFILES[PROFILE]["T_K"])
    P_bar = float(PROFILES[PROFILE]["P_bar"])

    # Convert pressure to barye and compute log for VULCAN filename
    barye = P_bar * 1e6
    logP = math.log10(barye)

    VULCAN_PATH = VULCAN_DIR / f"vul-T{int(T_K)}KlogP{logP:.1f}-NCHO-solar_hot_ini.vul"
    assert VULCAN_PATH.exists(), f"Missing VULCAN file: {VULCAN_PATH}"

    # ========================================================================
    # Step 2: Load Normalization Manifest
    # ========================================================================

    manifest = json.loads((PROCESSED_DIR / "normalization.json").read_text())
    norm = NormalizationHelper(manifest)
    meta = manifest.get("meta", {})

    # Extract species and global variable names
    in_names = list(meta.get("species_variables") or manifest.get("species_variables") or [])
    assert in_names, "species_variables missing in normalization.json"

    # Strip "_evolution" suffix to get base species names
    in_bases = [n[:-10] if n.endswith("_evolution") else n for n in in_names]

    # Global variables (T, P, etc.)
    gvars = list(meta.get("global_variables") or manifest.get("global_variables") or [])

    # ========================================================================
    # Step 3: Load VULCAN Ground Truth Data
    # ========================================================================

    with open(VULCAN_PATH, "rb") as h:
        d = pickle.load(h)

    t_all = np.asarray(d["variable"]["t_time"], float)  # [T] - Time points
    Y = np.asarray(d["variable"]["y_time"], float)  # [T, layer, S] - Number densities

    names = list(d["variable"]["species"])
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Compute mixing ratios at layer 0: MR = n_i / Σn_j
    den = np.maximum(Y[:, 0, :].sum(-1), 1e-30)
    MR_all = Y[:, 0, :] / den[:, None]  # [T, S]

    # Create log-time grid for interpolation
    xk = np.log10(np.clip(t_all, 1e-300, None))

    # ========================================================================
    # Step 4: Prepare Global Conditioning Variables
    # ========================================================================

    if gvars:
        g_phys = np.zeros((1, len(gvars)), np.float32)
        for i, nm in enumerate(gvars):
            nm_l = nm.strip().lower()
            g_phys[0, i] = (barye if nm_l.startswith("p") else
                            (T_K if nm_l.startswith("t") else 0.0))
        g_norm = norm.normalize(torch.from_numpy(g_phys), gvars).float()
    else:
        g_norm = torch.zeros(1, 0)

    # ========================================================================
    # Step 5: Extract Anchor State at T0 via Linear Interpolation
    # ========================================================================

    # Interpolate in log-time space (abundance interpolated linearly)
    xq0 = np.log10(np.array([T0], float))

    MR_T0 = np.array([
        np.interp(xq0, xk, MR_all[:, j], left=MR_all[0, j], right=MR_all[-1, j])[0]
        for j in range(MR_all.shape[1])
    ], float)

    # Map VULCAN species to model input species
    y0_inputs = np.array([
        max(MR_T0[name_to_idx[b]], 0.0) if b in name_to_idx else 0.0
        for b in in_bases
    ], float)

    # Compute effective floor from normalizer (roundtrip through zero)
    eff = norm.denormalize(
        norm.normalize(torch.zeros(1, len(in_names)), in_names), in_names
    ).numpy().reshape(-1)
    floor = np.maximum(np.nan_to_num(eff, nan=0.0), FEED_MIN)

    # Apply floor and normalize to simplex
    v = np.maximum(y0_inputs, floor)
    s = v.sum()
    y0_simplex = (np.full_like(v, 1.0 / len(v)) if (not np.isfinite(s) or s <= 0)
                  else v / s).astype(np.float32)

    # Normalize anchor state for model input
    y0_norm = norm.normalize(torch.from_numpy(y0_simplex[None, :]), in_names).float()

    # ========================================================================
    # Step 6: Prepare Time Delta Grids
    # ========================================================================

    assert T_FINAL > T0, "T_FINAL must be > T0"

    # Prediction grid: K points logarithmically spaced
    dt_pred = np.logspace(math.log10(max(DT_MIN, 1e-12)),
                          math.log10(max(DT_MIN, T_FINAL - T0)),
                          int(K_POINTS), dtype=np.float32)

    # Truth grid: Higher resolution for smooth curves; extend to full available truth
    dt_truth = np.logspace(math.log10(max(DT_MIN, 1e-12)),
                           math.log10(max(1e-12, float(t_all.max() - T0))),
                           max(256, min(2048, 6 * len(dt_pred))), dtype=np.float64)

    # Normalize time deltas for model input
    dt_hat = norm.normalize_dt_from_phys(torch.from_numpy(dt_pred)).view(-1, 1).float()

    # ========================================================================
    # Step 7: Load Exported Model
    # ========================================================================
    # Try to load in order: (1) AOTI MPS, (2) PT2 exports, (3) any PT2

    fn, dev, use_dyn = None, torch.device("cpu"), True

    # Try loading AOTI package first (MPS)
    aoti_dir = MODEL_DIR / "export_k_dyn_mps.aoti"
    if torch.backends.mps.is_available() and aoti_dir.exists():
        try:
            from torch._inductor import aot_load_package
            fn = aot_load_package(str(aoti_dir))
            dev = torch.device("mps")
            print(f"[INFO] Using AOTI MPS: {aoti_dir}")
        except Exception as e:
            print(f"[WARN] AOTI load failed: {e}  → fallback to .pt2")

    # Fallback to PT2 exports
    if fn is None:
        from torch.export import load as torch_export_load

        pref = ["export_k_dyn_mps.pt2", "export_k_dyn_gpu.pt2", "export_k_dyn_cpu.pt2",
                "export_k1_mps.pt2", "export_k1_gpu.pt2", "export_k1_cpu.pt2",
                "complete_model_exported_k1.pt2", "complete_model_exported.pt2"]

        pt2 = next((MODEL_DIR / p for p in pref if (MODEL_DIR / p).exists()), None)

        # If no known names found, use most recent PT2
        if pt2 is None:
            cands = sorted(MODEL_DIR.glob("*.pt2"), key=lambda p: p.stat().st_mtime, reverse=True)
            pt2 = cands[0] if cands else None

        assert pt2 is not None, "No exported model found (.aoti or .pt2)"

        ep = torch_export_load(str(pt2))
        fn = ep.module()
        n = pt2.name.lower()

        # Infer device and format from filename
        dev = (torch.device("mps") if ("mps" in n and torch.backends.mps.is_available())
               else (torch.device("cuda") if (("gpu" in n or "cuda" in n) and torch.cuda.is_available())
                     else torch.device("cpu")))
        use_dyn = ("dyn" in n)

        print(f"[INFO] Using PT2: {pt2.name} → device={dev.type} dynBK={use_dyn}")

    # ========================================================================
    # Step 8: Prepare Model Inputs for Both Signatures
    # ========================================================================

    K = len(dt_pred)
    y0_norm_d = y0_norm.to(dev)
    g_norm_d = g_norm.to(dev)

    # BK signature: [B=1, K, ...]
    state_bk = y0_norm_d.expand(1, K, -1)
    dt_bk = dt_hat.to(dev).view(1, K, 1)
    g_bk = (g_norm_d.expand(1, K, -1) if g_norm_d.numel()
            else torch.empty(1, K, 0, device=dev))

    # K1 signature: [K, ...]
    state_k1 = y0_norm_d.repeat(K, 1)
    dt_k1 = dt_hat.to(dev)
    g_k1 = (g_norm_d.repeat(K, 1) if g_norm_d.numel()
            else torch.empty(K, 0, device=dev))

    # ========================================================================
    # Step 9: Run Model Inference
    # ========================================================================
    # Try both signatures (BK and K1) to handle different export formats

    y_pred_norm, sig = None, None

    for mode in (["BK", "K1"] if use_dyn else ["K1", "BK"]):
        try:
            with torch.inference_mode():
                if mode == "BK":
                    out = fn(state_bk, dt_bk, g_bk)  # [1, K, S_out]
                    y_pred_norm = (out[0] if (isinstance(out, torch.Tensor) and
                                              out.dim() == 3 and out.size(0) == 1)
                                   else out).to("cpu")
                else:
                    out = fn(state_k1, dt_k1, g_k1)  # [K, 1, S_out] or [K, S_out]
                    y_pred_norm = (out[:, 0, :] if (isinstance(out, torch.Tensor) and
                                                    out.dim() == 3 and out.size(1) == 1)
                                   else out).to("cpu")
            sig = mode
            break
        except Exception as e:
            print(f"[note] {mode} call failed: {e}")

    assert y_pred_norm is not None, "Failed to run model with either BK or K1 signature"
    print(f"[INFO] Inference OK: device={dev.type} signature={sig} pred_shape={tuple(y_pred_norm.shape)}")

    # ========================================================================
    # Step 10: Denormalize Predictions
    # ========================================================================

    S_out = int(y_pred_norm.shape[-1])

    # Try to get target species names from manifest
    cand = list(meta.get("target_species_variables") or
                manifest.get("target_species_variables") or [])

    out_names = (cand if (cand and len(cand) == S_out)
                 else (in_names if len(in_names) == S_out else in_names[:S_out]))
    out_bases = [n[:-10] if n.endswith("_evolution") else n for n in out_names]

    # Denormalize to physical mixing ratios
    y_pred_phys = norm.denormalize(y_pred_norm, out_names).cpu().numpy()

    # ========================================================================
    # Step 11: Prepare Ground Truth via Interpolation
    # ========================================================================

    # Find species that exist in both model outputs and VULCAN
    present = [b for b in out_bases if b in name_to_idx]
    assert present, "No overlap between outputs and MiniChem species"

    idx = np.array([name_to_idx[b] for b in present], int)

    # Interpolate ground truth at T0 + dt_truth (in log-time space)
    xq = np.log10(T0 + dt_truth)
    MR_sub = MR_all[:, idx]
    truth = np.column_stack([
        np.interp(xq, xk, MR_sub[:, j], left=MR_sub[0, j], right=MR_sub[-1, j])
        for j in range(MR_sub.shape[1])
    ])

    # Normalize to simplex
    truth = np.clip(truth, 1e-300, None)
    truth /= np.maximum(truth.sum(1, keepdims=True), 1e-30)

    # ========================================================================
    # Step 12: Align Predictions with Ground Truth Species
    # ========================================================================

    mask = [b in present for b in out_bases]
    pred_sub = np.clip(y_pred_phys[:, np.where(mask)[0]], 1e-300, None)
    pred_sub /= np.maximum(pred_sub.sum(1, keepdims=True), 1e-30)

    # ========================================================================
    # Step 13: Select Species for Plotting
    # ========================================================================

    keep = [i for i, b in enumerate(present) if b in PLOT_SPECIES] or list(range(len(present)))
    labels = [present[i] for i in keep]
    truth_plot = truth[:, keep]
    pred_plot = np.clip(pred_sub[:, keep], PLOT_FLOOR, None)

    # Get anchor values for selected species
    in_idx_by_base = {b: j for j, b in enumerate(in_bases)}
    y0_on_outputs = np.array([
        (y0_simplex[in_idx_by_base[b]] if b in in_idx_by_base else np.nan)
        for b in present
    ])[keep]

    # ========================================================================
    # Step 14: Generate Comparison Plot
    # ========================================================================

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # CHANGED: let truth extend past predictions by auto-expanding the upper limit
    xmax_truth = float(dt_truth.max()) if dt_truth.size else XMAX_DT
    ax.set_xlim(XMIN_DT, max(XMAX_DT, xmax_truth))
    ax.set_ylim(YMIN, YMAX)

    ax.set_xlabel("Δt since anchor (s)")
    ax.set_ylabel("Abundance (no-He simplex over model outputs)")

    colors = plt.cm.tab20(np.linspace(0, 0.95, len(keep)))

    # Plot ground truth curves (solid lines)
    for i, c in enumerate(colors):
        ax.plot(dt_truth, np.clip(truth_plot[:, i], PLOT_FLOOR, None),
                '-', lw=1.8, alpha=0.95, color=c)

    # Plot anchor points (hollow squares)
    for i, c in enumerate(colors):
        if np.isfinite(y0_on_outputs[i]):
            ax.plot([XMIN_DT], [max(y0_on_outputs[i], PLOT_FLOOR)],
                    marker='s', mfc='none', mec=c, ms=6, mew=1.2, ls='none')

    # Plot predictions (hollow circles)
    for i, c in enumerate(colors):
        ax.plot(dt_pred, pred_plot[:, i], ls='none', marker='o',
                mfc='none', mec=c, ms=4.0, mew=1.0, alpha=0.95)

    # Create legend ordered by maximum abundance
    order = np.argsort(np.max(truth_plot, axis=0))[::-1]
    ax.legend([Line2D([0], [0], color=colors[i], lw=2.0) for i in order],
              [labels[i] for i in order], loc='best', fontsize=10)

    # Add generic legend for line/marker types
    ax.legend(handles=[
        Line2D([0], [0], color='black', lw=2.0, ls='-',
               label='MiniChem truth (linear interp, log-time)'),
        Line2D([0], [0], color='black', lw=0.0, ls='none', marker='o',
               mfc='none', mec='black', label='Prediction'),
        Line2D([0], [0], color='black', lw=0.0, ls='none', marker='s',
               mfc='none', mec='black', label='Anchor')
    ], fontsize=10)

    # Save figure
    fig.tight_layout()
    out_png = MODEL_DIR / "plots" / f"single_shot_{PROFILE}_deltat_linear_interp.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {out_png}")


if __name__ == "__main__":
    main()
