#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN vs Flow-map model (colors aligned) — model starts from *provided* initial abundances.

What this does:
1) Load VULCAN 0D pickle, compute/plot mixing-ratio time series (He included in denominator).
2) Build model y0 *from the provided TEST_SPECIES/VALUES*, map to model species (suffix _evolution),
   fill unspecified model species with DEFAULT_Y0, then renormalize over model species so sum=1.
   Feed T and P (converted to barye) to the exported K=1 model over a Δt grid up to 1e7 s.
3) Normalize → model → denormalize; plot model points with the SAME color per species as VULCAN lines.
"""

from __future__ import annotations
import os, sys, json, pickle, csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===================== USER EDITABLE PATHS =====================
# VULCAN input
VULCAN_BASE = "/Users/imalsky/Desktop/Chemistry_Project/Vulcan/0D_full_NCHO/solar"
VULCAN_FILE = "vul-T1000KlogP3.0-NCHO-solar_hot_ini.vul"

# Model repo layout (override with env if you prefer)
REPO_ROOT = Path("/Users/imalsky/Desktop/Goswami")            # repo root containing src/, data/, models/
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "1"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
SRC_DIR = Path(os.environ.get("SRC_DIR", REPO_ROOT / "src"))   # to import normalizer
sys.path.insert(0, str(SRC_DIR))

# Output
PLOT_DIR = Path("plot/NCHO_loop"); PLOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR  = Path("csv");            CSV_DIR.mkdir(parents=True, exist_ok=True)

# Case / conditions
T_K  = 1000.0
P_Pa = 100.0  # 1 mbar = 100 Pa
# Feed pressure to model in **barye** (cgs): 1 Pa = 10 barye
P_BARYE = float(os.environ.get("P_BARYE", P_Pa * 10.0))

# What to plot from VULCAN (order defines colors)
PLOT_SPEC = ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2', 'C2H2']

# Δt grid for model (seconds)
DT_GRID = np.logspace(-2, 7, 25, dtype=np.float64)  # 1e-2 .. 1e7

# ======= YOUR PROVIDED INITIAL MIX (will be renormalized to sum=1 over provided list) =======
TEST_SPECIES = ['H2', 'H2O', 'H', 'CH4', 'CO', 'CO2', 'N2', 'NH3', 'He']
TEST_VALUES  = [9.975331e-01, 1.074060e-03, 0.0, 5.902400e-04, 0.0, 0.0, 0.0, 1.415900e-04, 1.679000e-01]
DEFAULT_Y0   = 1e-25  # used for model species not present in the provided list
# ===============================================================

# ---------- helpers ----------
def _suffix(s: str) -> str:
    return s if s.endswith("_evolution") else f"{s}_evolution"

def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists(): return p
    raise FileNotFoundError("None of these exist:\n  " + "\n  ".join(map(str, paths)))

def _load_manifest() -> dict:
    for p in (MODEL_DIR/"normalization.json", PROCESSED_DIR/"normalization.json"):
        if p.exists(): return json.load(open(p, "r"))
    raise FileNotFoundError("normalization.json not found in MODEL_DIR or PROCESSED_DIR")

def _load_exported_model(md: Path):
    from torch.export import load as torch_load
    return torch_load(str(_first_existing([
        md/"complete_model_exported_k1.pt2",
        md/"complete_model_exported_k1_int8.pt2",
        md/"complete_model_exported.pt2",
        md/"complete_model_exported_int8.pt2",
    ]))).module()

def _normalize_vec_to_one(arr: np.ndarray, eps: float = 0.0) -> np.ndarray:
    arr = np.maximum(arr, eps)
    s = arr.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("Vector sum must be positive to normalize to 1.")
    return arr / s

# ---------- 1) Load VULCAN & write t=0 CSV (for reference & plotting) ----------
vul_path = Path(VULCAN_BASE) / VULCAN_FILE
if not vul_path.exists():
    raise FileNotFoundError(f"Required file not found: {vul_path}")

with open(vul_path, "rb") as h:
    data = pickle.load(h)

species_vul: List[str] = list(data['variable']['species'])
t_time  = np.asarray(data['variable']['t_time'], dtype=float)         # [T]
y_time  = np.asarray(data['variable']['y_time'], dtype=float)         # [T, layer, S]
total_all = y_time[:, 0, :].sum(axis=-1).astype(float)                # [T], includes He in denominator
total0 = float(max(total_all[0], 1e-30))

# VULCAN t=0 MR for every species (including He if present) — written for reference
y0_map_vul: Dict[str, float] = {sp: float(y_time[0, 0, species_vul.index(sp)] / total0) for sp in species_vul}
csv_y0 = CSV_DIR / "vulcan_y0_all_species.csv"
with open(csv_y0, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["species", "y0_mixing_ratio"])
    for sp in species_vul: w.writerow([sp, f"{y0_map_vul[sp]:.10e}"])
print(f"[OK] wrote VULCAN y0 CSV: {csv_y0}")

# ---------- 2) Build model inputs FROM PROVIDED MIX (NOT from VULCAN), then normalize/model/denormalize ----------
from normalizer import NormalizationHelper

manifest = _load_manifest()
meta = manifest.get("meta", {}) or {}
species_model: List[str] = list(meta.get("species_variables", []) or [])
globals_v:     List[str] = list(meta.get("global_variables", []) or [])
if not species_model:
    raise RuntimeError("No species_variables in normalization.json")

# Provided list → normalize to 1 over *provided entries* (He can be present; that's fine)
vals = np.asarray(TEST_VALUES, dtype=np.float64)
vals_norm = _normalize_vec_to_one(vals, eps=0.0)
provided_map = { _suffix(n): float(v) for n, v in zip(TEST_SPECIES, vals_norm) }

print("\n[INFO] Provided initial abundances (normalized to 1 over provided entries):")
for n, v in zip(TEST_SPECIES, vals_norm):
    print(f"  {n:>6s}: {v:.6e}")
print(f"  sum: {vals_norm.sum():.6e}")

# Map into model species order; default tiny for unspecified, then renormalize over model species
y0_vec = np.array([provided_map.get(name, DEFAULT_Y0) for name in species_model], dtype=np.float64)
y0_vec = _normalize_vec_to_one(y0_vec, eps=0.0).astype(np.float32)  # ensure sum(model y0)=1
y0_phys = torch.tensor([y0_vec], dtype=torch.float32)               # [1,S]

print("\n[INFO] y0 fed to model (model species order; sum=1 across model species):")
print(f"  sum(model y0)={float(y0_phys.sum()):.6e}")
for name, val in zip(species_model[:24], y0_vec[:24]):
    print(f"  {name:>24s}: {val:.6e}")
if len(species_model) > 24:
    print(f"  ... ({len(species_model)-24} more species filled / normalized)")

# Globals (Pressure in **barye**, Temperature in K) in model’s globals order
g_arr = np.zeros((1, len(globals_v)), dtype=np.float32)
for i, name in enumerate(globals_v):
    key = name.strip().lower()
    if key.startswith("p"):   g_arr[0, i] = P_BARYE
    elif key.startswith("t"): g_arr[0, i] = T_K
    else:                     g_arr[0, i] = 0.0
g_phys = torch.from_numpy(g_arr)

# Normalize → model → denormalize
norm = NormalizationHelper(manifest)
y0n = norm.normalize(y0_phys, species_model)
gn  = norm.normalize(g_phys,  globals_v)
dt_hat = norm.normalize_dt_from_phys(torch.tensor(DT_GRID, dtype=torch.float32))  # [K]

fn = _load_exported_model(MODEL_DIR)
preds = []
with torch.inference_mode():
    for k in range(int(dt_hat.numel())):
        out_n = fn(y0n, gn, dt_hat[k:k+1])                       # normalized prediction
        out_p = norm.denormalize(out_n.reshape(1, -1), species_model)  # physical mixing ratios
        preds.append(out_p.squeeze(0).cpu().numpy())
Y_model = np.stack(preds, axis=0)  # [K, S]

# Save model predictions (all species) for convenience
csv_pred = CSV_DIR / "model_predictions_over_dt_from_provided_y0.csv"
with open(csv_pred, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species"] + [f"dt_{i}_{DT_GRID[i]:.3e}_s" for i in range(len(DT_GRID))])
    for j, s in enumerate(species_model):
        w.writerow([s] + [f"{Y_model[i, j]:.10e}" for i in range(len(DT_GRID))])
print(f"[OK] wrote model predictions CSV: {csv_pred}")

# ---------- 3) Plot VULCAN curves + model points with MATCHED COLORS ----------
# Build a color map per species (by PLOT_SPEC order)
cmap = plt.get_cmap("tab20")
colors = [cmap(i / max(1, len(PLOT_SPEC) - 1)) for i in range(len(PLOT_SPEC))]
color_map = {sp: colors[i] for i, sp in enumerate(PLOT_SPEC)}

fig, ax = plt.subplots(1, 1, figsize=(9.5, 6))

# VULCAN curves (solid lines)
for sp in PLOT_SPEC:
    if sp not in species_vul:
        print(f"[warn] '{sp}' not found in VULCAN file; skipping VULCAN curve.")
        continue
    j = species_vul.index(sp)
    mr = y_time[:, 0, j] / np.maximum(total_all, 1e-30)
    ax.plot(t_time, np.clip(mr, 1e-30, None), lw=1.8, color=color_map[sp], label=f"{sp} (VULCAN)")

# Model points (same color, marker only) — starting from your PROVIDED y0 (not VULCAN)
for sp in PLOT_SPEC:
    s_model = _suffix(sp)
    if s_model not in species_model:
        continue
    j = species_model.index(s_model)
    ax.plot(DT_GRID, np.clip(Y_model[:, j], 1e-30, None), 'o', ms=4, color=color_map[sp], label=f"{sp} (model)")

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("Time since start, Δt (s)")
ax.set_ylabel("Mixing Ratio")
ax.set_title(f"VULCAN vs Model (model starts from PROVIDED y₀) @ T={T_K:.0f} K, P={P_Pa:.0f} Pa ({P_BARYE:.0f} barye)")
ax.set_xlim(1.0, max(float(t_time.max()*1.05), DT_GRID.max()*1.05, 1.0))
ax.set_ylim(5e-17, 1e-2)
ax.legend(frameon=False, ncol=2, fontsize=9)
fig.tight_layout()
png_path = PLOT_DIR / f"0D-compare-from-provided-y0-T{int(T_K)}K-P{int(P_Pa)}Pa.png"
fig.savefig(png_path, dpi=170)
plt.close(fig)
print(f"[OK] wrote plot: {png_path}")

# ---------- Console summary ----------
present = [s for s in map(_suffix, TEST_SPECIES) if s in species_model]
idx = np.array([species_model.index(s) for s in present], dtype=int)
print("\n=== Inputs (to model) ===")
print(f"T_K = {T_K:.1f},  P_Pa = {P_Pa:.1f},  P_barye (fed) = {P_BARYE:.1f}")
print(f"Δt grid: {DT_GRID[0]:.2e} .. {DT_GRID[-1]:.2e}  (K={len(DT_GRID)})")
print("\nStart -> End (for provided species that exist in the model):")
for sname, j in zip(present, idx):
    print(f"{sname:>18s}: {y0_phys[0, j].item():.6e} -> {Y_model[-1, j]:.6e}")

# ---------- Runtime knobs ----------
if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
