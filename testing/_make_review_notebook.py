"""Generate collaborator_review.ipynb from clean Python strings."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "collaborator_review.ipynb"


def md(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": src}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

C_INTRO = """\
# Collaborator Feedback — Notebook Response

Addressing the four points in the reviewer email:

| # | Topic | In notebook? |
|---|-------|--------------|
| 1 | Expanded intro paragraph on previous surrogate chemistry models | ❌ Paper text only |
| 2 | Mention Minichem in intro + availability check | ✅ See "Minichem" section |
| 3 | Compare Chemulator vs Minichem quantitatively | ⚠️  Partial — explanation + roadmap below |
| 4 | More individual-species trajectory plots (good & bad cases) | ✅ See "Trajectory Plots" section |\
"""

C_SETUP = """\
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# Works whether launched from testing/ or from Chemulator/
_cwd = Path().resolve()
REPO = _cwd if (_cwd / "models" / "final_version").exists() else _cwd.parent
TESTING_DIR = REPO / "testing"
MODEL_DIR = REPO / "models" / "final_version"
SHARD = REPO / "data" / "processed" / "test" / "shard_test_mix_r0_00003.npz"

try:
    plt.style.use(str(TESTING_DIR / "science.mplstyle"))
except OSError:
    warnings.warn("science.mplstyle not found; using matplotlib defaults.")

# Load exported physical-I/O model
ep = torch.export.load(MODEL_DIR / "physical_model_k1_cpu.pt2")
model = ep.module()

# Species metadata
meta = json.loads((MODEL_DIR / "physical_model_metadata.json").read_text())
SPECIES = meta["species_order"]          # e.g. ["C2H2_evolution", ...]
NAMES = [s.removesuffix("_evolution") for s in SPECIES]
N_SP = len(SPECIES)

# Test shard (physical mixing ratios — not normalized)
d = np.load(SHARD, allow_pickle=False)
y_mat = d["y_mat"].astype(np.float32)   # (N_traj, 100, 12) mixing ratios
g_mat = d["globals"].astype(np.float32) # (N_traj, 2)  [P_Pa, T_K]
t_vec = d["t_vec"].astype(np.float32)   # (100,)       shared time grid [s]
N_TRAJ, N_TIME, _ = y_mat.shape

print(f"Loaded {N_TRAJ:,} test trajectories × {N_TIME} timesteps × {N_SP} species")
print(f"P range : {g_mat[:,0].min():.1e} → {g_mat[:,0].max():.1e} Pa")
print(f"T range : {g_mat[:,1].min():.0f} → {g_mat[:,1].max():.0f} K")
print(f"t range : {t_vec[0]:.1e} → {t_vec[-1]:.1e} s")

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:N_SP]
TINY   = 1e-35\
"""

C_HELPERS = """\
def predict(y0: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
    \"\"\"One-shot predict all t[1:] from initial state y0 at t[0].
    Returns array of shape (len(t)-1, N_SP) in physical mixing-ratio space.
    \"\"\"
    dt = (t[1:] - t[0]).astype(np.float32)   # (99,)
    N  = len(dt)
    y_b  = torch.from_numpy(y0[None]).float().repeat(N, 1)  # (N, 12)
    g_b  = torch.from_numpy(g[None]).float().repeat(N, 1)   # (N,  2)
    dt_b = torch.from_numpy(dt).float().view(-1, 1)          # (N,  1)
    with torch.no_grad():
        return model(y_b, dt_b, g_b)[:, 0, :].numpy()       # (N, 12)


def make_traj_figure(traj_idx: int, suptitle: str = "") -> plt.Figure:
    \"\"\"3×4 grid: one subplot per species, VULCAN truth vs Chemulator prediction.\"\"\"
    y      = y_mat[traj_idx]                   # (100, 12) ground truth
    g      = g_mat[traj_idx]                   # (2,)
    y_pred = predict(y[0], g, t_vec)           # (99,  12)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)

    for sp_i, (ax, name) in enumerate(zip(axes.flat, NAMES)):
        col  = COLORS[sp_i]
        y_t  = np.clip(y[:, sp_i],      TINY, None)
        y_p  = np.clip(y_pred[:, sp_i], TINY, None)
        ax.loglog(t_vec,     y_t, "-",  color=col, lw=2.0, alpha=0.6)
        ax.loglog(t_vec[1:], y_p, "--", color=col, lw=1.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.set_ylabel("mixing ratio", fontsize=7)
        ax.tick_params(labelsize=6)

    title = (suptitle or f"Trajectory {traj_idx}") + f"\\nP={g[0]:.2e} Pa  T={g[1]:.0f} K"
    fig.suptitle(title, fontsize=10)

    handles = [
        Line2D([0], [0], color="k", lw=2.0, ls="-",  label="VULCAN (ground truth)"),
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="Chemulator (prediction)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    return fig\
"""

C_REPRESENTATIVE = """\
# Pick 4 trajectories that span the P-T parameter space quadrants.
rng   = np.random.default_rng(0)
logP  = np.log10(g_mat[:, 0])
T_K   = g_mat[:, 1]
logP_med = np.median(logP)
T_med    = np.median(T_K)

def pick(mask):
    idxs = np.where(mask)[0]
    return int(rng.choice(idxs)) if len(idxs) else int(rng.integers(N_TRAJ))

cases = [
    (pick((logP <  logP_med) & (T_K <  T_med)), "Low P, Low T"),
    (pick((logP <  logP_med) & (T_K >= T_med)), "Low P, High T"),
    (pick((logP >= logP_med) & (T_K <  T_med)), "High P, Low T"),
    (pick((logP >= logP_med) & (T_K >= T_med)), "High P, High T"),
]

print("Representative trajectories selected:")
for idx, label in cases:
    g = g_mat[idx]
    print(f"  {label:20s}  traj {idx:5d}  P={g[0]:.2e} Pa  T={g[1]:.0f} K")\
"""

C_PLOT_REPRESENTATIVE = """\
# Point 4 — good/representative cases
for idx, label in cases:
    fig = make_traj_figure(idx, label)
    plt.show()\
"""

C_FIND_WORST = """\
# Scan 500 random trajectories; rank by peak absolute log10 error
# on species with meaningful abundance (> 1e-15 mixing ratio).
N_SCAN   = 500
scan_idx = rng.choice(N_TRAJ, size=N_SCAN, replace=False)

def traj_max_err(i: int) -> float:
    y      = y_mat[i]
    y_pred = predict(y[0], g_mat[i], t_vec)  # (99, 12)
    y_true = y[1:]                            # (99, 12)
    sig    = y_true > 1e-15                   # ignore trace / numerical-zero
    if not sig.any():
        return 0.0
    log_err = np.abs(
        np.log10(np.clip(y_pred, TINY, None)) -
        np.log10(np.clip(y_true, TINY, None))
    )
    return float(log_err[sig].max())

print(f"Scanning {N_SCAN} trajectories …")
errs = np.array([traj_max_err(i) for i in scan_idx])

p50, p90, p99 = np.percentile(errs, [50, 90, 99])
print(f"Max log10-error distribution over {N_SCAN} trajectories:")
print(f"  median={p50:.2f} dex  |  p90={p90:.2f} dex  |  p99={p99:.2f} dex  |  max={errs.max():.2f} dex")

worst_idx = scan_idx[np.argsort(errs)[::-1][:4]]
print("\\nWorst-case trajectories:")
for rank, wi in enumerate(worst_idx, 1):
    g   = g_mat[wi]
    err = errs[scan_idx == wi][0]
    print(f"  Rank {rank}: traj {wi:5d}  P={g[0]:.2e} Pa  T={g[1]:.0f} K  → max err={err:.2f} dex")\
"""

C_PLOT_WORST = """\
# Point 4 — bad prediction cases
for rank, wi in enumerate(worst_idx, 1):
    err = errs[scan_idx == wi][0]
    fig = make_traj_figure(wi, f"Rank {rank} worst case  (max error {err:.2f} dex)")
    plt.show()\
"""

C_MINICHEM_INTRO = """\
## Points 2 & 3 — Minichem: What it is, availability, and comparison roadmap

**What Minichem is**

Minichem is a *reduced chemical network* for CHON exoplanet atmospheres, developed
to enable fast chemistry in 3-D GCM codes (e.g. canoe / Athena++).  It uses ~6 net
reactions and a relaxation-based integration scheme (Tsai et al. 2018;
Zahnle & Marley 2014) to approximate equilibrium chemistry ~10–100× faster than
VULCAN's full 300-reaction network.  The goal — accelerating chemistry in atmosphere
models — is the same as Chemulator's, which is why the reviewer wants a comparison.

**Is it pip-installable?**  No.  Minichem is a C++ component of the
[canoe](https://github.com/chengcli/canoe) atmospheric model, not a standalone
Python package.  There is no `pip install minichem`.

**Where it lives locally**\
"""

C_MINICHEM_CODE = '''\
CANOE_MINICHEM = Path(
    "/Users/imalsky/Documents/Astronomy-Software/canoe/examples/2024-XZhang-minichem"
)

print(f"canoe/minichem path: {CANOE_MINICHEM}")
print(f"Exists locally     : {CANOE_MINICHEM.exists()}\\n")

if CANOE_MINICHEM.exists():
    print("Files in the example directory:")
    for f in sorted(CANOE_MINICHEM.iterdir()):
        print(f"  {f.name}")

roadmap = """
How to run a Minichem comparison (roadmap - not done yet):
  1. cd canoe && cmake -B build -DCMAKE_BUILD_TYPE=Release
     cmake --build build --target minichem -j4
  2. For each (P, T) in the test set, write a minichem input file
     (dry_mini.inp style), run the binary, read species output.
  3. Align time grids with t_vec, compare mixing ratios for shared species.
  4. Plot: VULCAN vs Minichem vs Chemulator on the same axes.

Key caveats before interpreting any comparison:
  * Minichem uses a reduced network - may not track all 12 Chemulator species.
  * Chemulator is a one-shot predictor (t0 -> tN); Minichem time-steps incrementally.
  * Minichem targets thermochemical equilibrium; Chemulator was trained on
    arbitrary VULCAN time-evolution (including disequilibrium regimes).
  * Fair comparison requires: same P, T, initial abundances, time grid.
"""
print(roadmap)\
'''

C_POINT1_NOTE = """\
## Points 1 & 2 (Paper Introduction) — Not addressable in a notebook

**Point 1** asks for an expanded introduction paragraph surveying previous surrogate
chemistry models (0-D box models, 1-D diffusion-chemistry surrogates, Earth-atmosphere
ML work).  This is writing work for the paper draft, not something a notebook can produce.

**Point 2** asks to mention Minichem *in the paper introduction* before the ML section.
Again, this is paper text.  The notebook above covers the technical side (what Minichem
is, where it lives, how to run a comparison), but the prose needs to go into the LaTeX
source.

A suggested outline for the intro addition (for discussion):
1. Paragraph on classical accelerations: relaxation-based schemes (Zahnle & Marley 2014;
   Tsai et al. 2018) and reduced networks (Minichem / mini-chemical schemes).
2. Short paragraph on ML surrogates: prior work on 0-D box-model emulators
   (Earth atmosphere studies), then exoplanet-specific work.
3. Bridge sentence: "Here we present Chemulator, a one-shot neural emulator …"\
"""

C_SUMMARY = """\
## Summary

| Task | Status |
|------|--------|
| Point 1 — intro paragraph on surrogate chemistry models | ❌ Paper text — see outline in cell above |
| Point 2 — Minichem in intro | ❌ Paper text — outline above; technical check done |
| Point 2 — Can Minichem be downloaded? | ✅ It's C++ in local canoe repo, not pip-installable |
| Point 3 — Compare with Minichem | ⚠️  Roadmap in notebook; ~1 day of C++ build + scripting |
| Point 4 — More trajectory plots (representative cases) | ✅ Done — 4 P/T quadrant cases |
| Point 4 — Bad prediction cases | ✅ Done — worst 4 from 500-trajectory scan |\
"""

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

cells = [
    md(C_INTRO),
    md("## Setup"),
    code(C_SETUP),
    code(C_HELPERS),
    md("## Point 4 — Individual Species Trajectory Plots\n\n"
       "For each case below, solid lines = VULCAN ground truth, dashed = Chemulator "
       "one-shot prediction from the initial state.\n\n"
       "### Representative cases (one per P-T quadrant)"),
    code(C_REPRESENTATIVE),
    code(C_PLOT_REPRESENTATIVE),
    md("### Bad prediction cases\n\n"
       "We scan 500 random test trajectories, rank by peak absolute log₁₀ error "
       "on species with mixing ratio > 10⁻¹⁵, and show the four worst."),
    code(C_FIND_WORST),
    code(C_PLOT_WORST),
    md(C_MINICHEM_INTRO),
    code(C_MINICHEM_CODE),
    md(C_POINT1_NOTE),
    md(C_SUMMARY),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {OUT}")
