#!/usr/bin/env python3
"""Build the VULCAN / emulator / mini-chem comparison figure (compare.png).

The published figure was previously assembled by hand from two notebooks, which is
how it came to disagree with the numbers beside it. This script builds the whole
two-row figure from the corrected cache and writes it straight into the manuscript
directory, so the figure and the reported statistics always come from one run.

Top row:    emulator in single-jump mode, each output time predicted from t0.
Bottom row: emulator stepped autoregressively through the same 99 intervals.
Both rows:  VULCAN reference (solid) and mini-chem stepped sequentially (markers).

    python -m testing.make_compare_figure          # from Chemulator/
"""
from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTING = REPO / "testing"
MODEL_DIR = REPO / "models" / "final_model"
SHARD = REPO / "data" / "processed" / "test" / "shard_test_mix_r2_200015.npz"
PAPER_DIR = REPO.parent / "Chemulator_ApJ"
CACHE = TESTING / "cache" / "minichem_compare_n1000_s42.npz"

BARYE_TO_PA = 0.1
RG_T = (1000.0, 2500.0)
TINY = 1e-35

PLOT_SPECIES = ["CH4", "CO", "CO2", "H2O", "NH3", "HCN"]
PLOT_COLORS = {"CH4": "#EE6677", "CO": "#4477AA", "CO2": "#228833",
               "H2O": "#AA3377", "NH3": "#66CCEE", "HCN": "#CCBB44"}
MC_MARKER_EVERY = 4


def main() -> int:
    try:
        plt.style.use(str(TESTING / "science.mplstyle"))
    except OSError:
        warnings.warn("science.mplstyle not found; using matplotlib defaults.")

    d = np.load(SHARD, allow_pickle=False)
    y_mat = d["y_mat"].astype(np.float32)
    g_mat = d["globals"].astype(np.float32)
    t_vec = d["t_vec"].astype(np.float32)
    P_Pa_all = BARYE_TO_PA * g_mat[:, 0].astype(np.float64)

    z = np.load(CACHE)
    stat_idxs = z["stat_idxs"]
    y_truth, y_chem, y_mc = z["y_truth"], z["y_chem"], z["y_mc"]
    store = {int(t): (y_truth[i], y_chem[i], y_mc[i]) for i, t in enumerate(stat_idxs)}

    meta_species = [s.removesuffix("_evolution")
                    for s in __import__("json").loads(
                        (MODEL_DIR / "physical_model_metadata.json").read_text())["species_order"]]

    # Pick one TYPICAL trajectory from each pressure tercile rather than the extreme
    # corners of the sampled regime. Selection is on (log P, T) only, never on the
    # error being plotted, so the panels cannot be accused of cherry-picking. Taking
    # the corners instead lands the low-pressure panel on a ~99.7th-percentile case,
    # because mini-chem's agreement with VULCAN degrades toward low pressure.
    ids = np.array(sorted(int(t) for t in stat_idxs))
    lp = np.log10(P_Pa_all[ids])
    edges = np.quantile(lp, [0.0, 1 / 3, 2 / 3, 1.0])
    plot_idxs = []
    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        m = (lp >= lo) & (lp <= hi) if b == 2 else (lp >= lo) & (lp < hi)
        band = ids[m]
        blp, bt = np.log10(P_Pa_all[band]), g_mat[band, 1].astype(np.float64)
        # nearest to the band centre in standardized (log P, T)
        dist = (((blp - np.median(blp)) / (blp.std() or 1.0)) ** 2
                + ((bt - np.median(bt)) / (bt.std() or 1.0)) ** 2)
        plot_idxs.append(int(band[int(np.argmin(dist))]))
    labels = ["Low pressure", "Intermediate pressure", "High pressure"]

    # Autoregressive rollout of the exported model over the same 99 intervals.
    ep = torch.export.load(MODEL_DIR / "physical_model_k1_cpu.pt2")
    model = ep.module()

    def rollout(ti):
        y = torch.from_numpy(y_mat[ti, 0][None]).float()
        g = torch.from_numpy(g_mat[ti][None]).float()
        out = []
        with torch.no_grad():
            for i in range(1, len(t_vec)):
                dt = torch.tensor([[float(t_vec[i] - t_vec[i - 1])]], dtype=torch.float32)
                y = model(y, dt, g)[:, 0, :]
                out.append(y.numpy()[0].copy())
        return np.array(out, dtype=np.float32)

    ar = {ti: rollout(ti) for ti in plot_idxs}
    print("autoregressive rollouts done")

    sp_i = [meta_species.index(s) for s in PLOT_SPECIES]

    # Scale each column to its own data instead of one shared 28-decade axis.
    # Both rows of a column share limits so the single-jump and autoregressive
    # panels stay directly comparable, but a high-pressure column no longer has
    # to reserve twenty empty decades for a low-pressure one. The span is capped
    # at MAX_DECADES so one species diving to the abundance floor cannot squash
    # everything else; a curve that leaves the axis is doing so visibly.
    MAX_DECADES = 16.0
    ylims = []
    for ti in plot_idxs:
        y_true, _, y_mc_t = store[ti]
        v = np.concatenate([y_true[:, sp_i].ravel(), store[ti][1][:, sp_i].ravel(),
                            ar[ti][:, sp_i].ravel(), y_mc_t[:, sp_i].ravel()])
        v = np.log10(v[v > 0])
        # Round up to a half-decade, then add another half-decade of headroom so
        # the highest-abundance species never runs into the top of the frame.
        hi = np.ceil(v.max() * 2) / 2 + 0.5
        # Bound the floor by the 1st percentile rather than the absolute minimum:
        # a single brief excursion should not cost eight empty decades. Anything
        # below simply runs off the bottom of the panel, which reads as what it is.
        lo = np.floor(max(np.percentile(v, 1) - 1.0, hi - MAX_DECADES))
        ylims.append((10.0 ** lo, 10.0 ** hi))

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharey=False,
                             sharex=True, constrained_layout=True)

    for row, (pred_of, row_label) in enumerate((
            (lambda ti: store[ti][1], "single jump"),
            (lambda ti: ar[ti], "autoregressive"))):
        for col, (ti, lbl) in enumerate(zip(plot_idxs, labels)):
            ax = axes[row, col]
            y_true, _, y_mc_t = store[ti]
            y_pred = pred_of(ti)
            for k, name in zip(sp_i, PLOT_SPECIES):
                c = PLOT_COLORS[name]
                ax.loglog(t_vec, np.clip(y_true[:, k], TINY, None), "-",
                          color=c, lw=4.5, alpha=0.55)
                ax.loglog(t_vec[1:], np.clip(y_pred[:, k], TINY, None), "--",
                          color=c, lw=3.5, alpha=0.95)
                ax.loglog(t_vec[::MC_MARKER_EVERY],
                          np.clip(y_mc_t[::MC_MARKER_EVERY, k], TINY, None),
                          marker="o", linestyle="None", color=c, markersize=5.5,
                          markeredgecolor="white", markeredgewidth=0.7, alpha=0.95)
            ax.set_ylim(*ylims[col])
            ax.tick_params(labelleft=True)
            if row == 0:
                ax.set_title(f"{lbl}\nP={P_Pa_all[ti]:.2e} Pa, T={g_mat[ti,1]:.0f} K")
            if row == 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel(f"Relative Abundance\n({row_label})")

    axes[0, 0].legend(handles=[Line2D([0], [0], color=PLOT_COLORS[s], lw=1.5, label=s)
                               for s in PLOT_SPECIES],
                      title="Species", ncol=2, loc="lower left", handlelength=1.5,
                      handletextpad=0.5, columnspacing=1.0)
    axes[0, -1].legend(handles=[
        Line2D([0], [0], color="k", lw=4.5, alpha=0.6, label="VULCAN (truth)"),
        Line2D([0], [0], color="k", lw=3.5, ls="--", label="This work"),
        Line2D([0], [0], color="k", marker="o", linestyle="None", markersize=7,
               markeredgecolor="white", markeredgewidth=0.7,
               label="mini-chem (time-stepped)")],
        title="Method", loc="lower left")

    out = PAPER_DIR / "compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")
    for ti, lbl in zip(plot_idxs, labels):
        print(f"  {lbl:16s} traj {ti:6d}  P={P_Pa_all[ti]:.3e} Pa  T={g_mat[ti,1]:.0f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
