#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ======= ABSOLUTE BASE PATH TO YOUR VULCAN 0D FILES (hot case) =======
VULCAN_BASE = "/Users/imalsky/Desktop/Vulcan/0D_full_NCHO/solar"
# The file name for 1000 K, logP = 3.0 (1 mbar) hot initial condition:
VULCAN_FILE = "vul-T1000KlogP3.0-NCHO-solar_hot_ini.vul"

# ======= OUTPUT DIR (PNG only) =======
PLOT_DIR = "plot/NCHO_loop"
os.makedirs(PLOT_DIR, exist_ok=True)

# ======= CASE (SI units for printing) =======
T_K = 1000.0
P_Pa = 100.0  # 1 mbar = 100 Pa

# ======= SPECIES TO PLOT =======
PLOT_SPEC = ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'HCN', 'N2', 'C2H2']

# ======= COLLABORATOR TEST SET (will be renormalized to sum to 1) =======
TEST_SPECIES = ['H2', 'H2O', 'H', 'CH4', 'CO', 'CO2', 'N2', 'NH3', 'He']
TEST_VALUES  = [9.975331e-01, 1.074060e-03, 0.0, 5.902400e-04, 0.0, 0.0, 0.0, 1.415900e-04, 1.679000e-01]

# ======= LOAD VULCAN FILE =======
vul_path = os.path.join(VULCAN_BASE, VULCAN_FILE)
if not os.path.exists(vul_path):
    raise FileNotFoundError(f"Required file not found: {vul_path}")

with open(vul_path, "rb") as h:
    data = pickle.load(h)

species = list(data['variable']['species'])
t_time  = np.asarray(data['variable']['t_time'], dtype=float)     # [T]
y_time  = np.asarray(data['variable']['y_time'], dtype=float)     # [T, layer, S]
# We will normalize by the total number density including He at each time:
total_all = y_time[:, 0, :].sum(axis=-1).astype(float)            # [T]

# ======= PLOT (log–log, simple legend) =======
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for sp in PLOT_SPEC:
    if sp not in species:
        print(f"[warn] '{sp}' not found in file; skipping in plot.")
        continue
    j = species.index(sp)
    mr = y_time[:, 0, j] / np.maximum(total_all, 1e-30)  # mixing ratio including He in denominator
    ax.plot(t_time, np.clip(mr, 1e-30, None), lw=1.6, label=sp)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mixing Ratio")
ax.set_title("T = 1000 K, P = 100 Pa (1 mbar)")
ax.legend(frameon=False)
ax.set_xlim(1.0, max(t_time.max()*1.05, 1.0))
ax.set_ylim(5e-17, 1e-2)
fig.tight_layout()

png_path = os.path.join(PLOT_DIR, f"0D-singlecase-T{int(T_K)}K-P{int(P_Pa)}Pa.png")
fig.savefig(png_path, dpi=160)
plt.close(fig)
print(f"[OK] wrote: {png_path}")

# ======= PRINT INITIAL ABUNDANCES FOR UNION (PLOT_SPEC ∪ TEST_SPECIES) =======
# Build ordered union: keep PLOT_SPEC order, then append any TEST_SPECIES not already included
ordered_union = list(PLOT_SPEC) + [s for s in TEST_SPECIES if s not in PLOT_SPEC]

# Renormalize the provided test set to sum to 1
vals = np.array(TEST_VALUES, dtype=float)
sum_vals = vals.sum()
if sum_vals <= 0:
    raise ValueError("Sum of TEST_VALUES must be positive.")
vals_norm = vals / sum_vals
provided_map = {k: float(v) for k, v in zip(TEST_SPECIES, vals_norm)}

# File y0 (t=0) mixing ratios normalized by total (including He)
total0 = float(max(total_all[0], 1e-30))
file_y0_map = {}
for sp in ordered_union:
    if sp in species:
        j = species.index(sp)
        file_y0_map[sp] = float(y_time[0, 0, j] / total0)
    else:
        file_y0_map[sp] = 0.0  # not in file

# Print (SI units for T, P; mixing ratios are unitless)
print("\n=== Initial abundances (union of plotted + collaborator set) ===")
print(f"T_K : {T_K}")
print(f"P_Pa: {P_Pa}")
for sp in ordered_union:
    prov = provided_map.get(sp, 0.0)
    f0   = file_y0_map.get(sp, 0.0)
    print(f"{sp:>5} | provided_norm={prov:.6e} | file_y0={f0:.6e}")
