#!/usr/bin/env python3
"""
Flow-map AE: Predicted vs True Scatter (1:1 line)

- Loads exported CPU K=1 program (export_k1_cpu.pt2)
- Uses test shard(s) under processed_data_dir/test/shard_*.npz
- Samples N_SAMPLES trajectories and Q_COUNT times per trajectory
- Produces a log–log scatter of y_true vs y_pred with a 1:1 line
- Prints summary error metrics in two buckets:
  (1) Points where y_true > ABUND_BUCKET_THRESHOLD
  (2) All finite points together

Also prints how the point counts are constructed (trajectories, anchors, query times, species).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

plt.style.use("science.mplstyle")

# ---------------- Paths & settings ----------------
REPO = Path(__file__).parent.parent
#MODEL_DIR = REPO / "models" / "big_big_big"
MODEL_DIR = REPO / "models" / "big_flow"
EP_FILENAME = "export_cpu.pt2"

sys.path.insert(0, str(REPO / "src"))
from utils import load_json_config as load_json, seed_everything
from normalizer import NormalizationHelper

# Globals (no argparse)
N_SAMPLES: int = 84
Q_COUNT: int = 100
PLOT_SPECIES: List[str] = []

EPS_FRACTIONAL = 1e-20
PLOT_FLOOR = 5e-20

ABUND_BUCKET_THRESHOLD = 1e-10  # bucket boundary (defined on y_true)


# ---------------- Helpers ----------------
def load_first_test_shard(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the first test shard.

    Returns:
        y_mat : [N, T, S_all]
        g_all : [N, G_all]
        t_vec : [T] or [N, T]
    """
    test_dir = data_dir / "test"
    shards = sorted(test_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {test_dir}")

    shard_path = shards[0]
    with np.load(shard_path) as d:
        y_mat = d["y_mat"].astype(np.float32)      # [N, T, S_all]
        g_all = d["globals"].astype(np.float32)    # [N, G_all]
        t_vec = d["t_vec"].astype(np.float32)      # [T] or [N, T]
    return y_mat, g_all, t_vec


def get_time_vector_for_sample(t_vec: np.ndarray, idx: int) -> np.ndarray:
    if t_vec.ndim == 1:
        return t_vec
    if t_vec.ndim == 2:
        return t_vec[idx]
    raise ValueError(f"Invalid t_vec ndim={t_vec.ndim}")


def select_species_indices(species_all: List[str], plot_species: List[str]) -> Tuple[List[int], List[str]]:
    base_all = [n[:-10] if n.endswith("_evolution") else n for n in species_all]
    if plot_species:
        keep = [i for i, b in enumerate(base_all) if b in plot_species]
    else:
        keep = list(range(len(base_all)))
    if not keep:
        raise RuntimeError("No requested species found in selected species list.")
    labels = [base_all[i] for i in keep]
    return keep, labels


def _indices(all_names: List[str], chosen: List[str]) -> List[int]:
    m = {n: i for i, n in enumerate(all_names)}
    try:
        return [m[n] for n in chosen]
    except KeyError as e:
        raise ValueError(f"Name not found in normalization meta list: {e.args[0]}") from None


def prepare_batch(
    y0: np.ndarray,
    g: np.ndarray,
    t_phys: np.ndarray,
    q_count: int,
    norm: NormalizationHelper,
    species: List[str],
    globals_: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Prepare normalized (y, dt, g) for K=1 export for a single trajectory anchored at t0.

    Returns:
        y_batch : [K, S]  (normalized)
        dt_norm : [K, 1]  (normalized Δt)
        g_batch : [K, G]  (normalized)
        q_idx   : [K]     (indices into t_phys / y)
    """
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)

    t_sel = t_phys[q_idx]
    dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)

    # Normalize anchor and globals
    y0_norm = norm.normalize(torch.from_numpy(y0[None, :]), species).float()  # [1, S]
    if globals_:
        g_norm = norm.normalize(torch.from_numpy(g[None, :]), globals_).float()  # [1, G]
    else:
        g_norm = torch.from_numpy(g[None, :]).float()  # [1, 0] or [1, G]

    # Normalize Δt (seconds)
    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_sec)).view(-1, 1).float()  # [K, 1]

    K = dt_norm.shape[0]
    y_batch = y0_norm.repeat(K, 1)  # [K, S]
    g_batch = g_norm.repeat(K, 1)   # [K, G]
    return y_batch, dt_norm, g_batch, q_idx


@torch.inference_mode()
def run_inference(model, y_batch: torch.Tensor, dt_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        y_batch : [B, S]
        dt_batch: [B, 1]
        g_batch : [B, G]

    Output (exported CPU K=1):
        [B, 1, S] -> return [B, S]
    """
    out = model(y_batch, dt_batch, g_batch)
    return out[:, 0, :]


def plot_pred_vs_true_scatter(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, out_path: Path) -> None:
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = np.clip(y_true_flat[mask], PLOT_FLOOR, None)
    y_pred = np.clip(y_pred_flat[mask], PLOT_FLOOR, None)

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    lo = vmin * 0.8
    hi = vmax * 1.2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(y_true, y_pred, ".", alpha=0.3, markersize=3)
    ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1.5, label="1:1")

    ax.set_xlabel("True abundance")
    ax.set_ylabel("Predicted abundance")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="lower right")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(), numticks=0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(), numticks=0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=12, pad=2)
    ax.set_box_aspect(1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved: {out_path}")


def _stats_1d(x: np.ndarray) -> Tuple[int, float, int, int]:
    """Returns: (n, median, min, max) for integer-ish arrays (safe for empty)."""
    if x.size == 0:
        return 0, float("nan"), 0, 0
    return int(x.size), float(np.median(x)), int(np.min(x)), int(np.max(x))


def print_count_breakdown(context: Dict[str, object], points_after_filters: int, title: str) -> None:
    n_traj = int(context["n_traj"])
    n_species = int(context["n_species"])
    total_q = int(context["total_q"])
    raw_points = int(context["raw_points"])
    finite_points = int(context["finite_points"])
    q_req = int(context["q_requested"])
    t_stats = context["t_stats"]
    k_stats = context["k_stats"]
    ku_stats = context["ku_stats"]
    dup_total = int(context["dup_total"])
    dup_traj = int(context["dup_traj"])

    print(f"\n--- {title} ---")
    print(f"Trajectories used:            {n_traj}")
    print(f"Anchors:                     {n_traj} (1 anchor per trajectory at t0)")
    print(f"Requested Q_COUNT:           {q_req}")
    print(f"Trajectory lengths T (med/min/max): {t_stats[1]:.1f} / {t_stats[2]} / {t_stats[3]}")
    print(f"Query times K (med/min/max):        {k_stats[1]:.1f} / {k_stats[2]} / {k_stats[3]}")
    print(f"Unique query times Ku (med/min/max):{ku_stats[1]:.1f} / {ku_stats[2]} / {ku_stats[3]}")
    if dup_traj > 0:
        print(
            f"Note: duplicate q_idx present in {dup_traj}/{n_traj} trajectories "
            f"(total duplicate indices counted as separate points: {dup_total})."
        )

    print(f"Species plotted:             {n_species}")
    print(f"Total query times (sum K):   {total_q}")
    print(f"Raw points:                  {raw_points} = (sum K) {total_q} * (species) {n_species}")
    print(f"Finite points:               {finite_points} (after dropping non-finite y_true/y_pred)")
    print(f"Points (this bucket):        {points_after_filters}")


def print_error_stats(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, title: str) -> None:
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]

    n = int(y_true.size)
    if n == 0:
        print("\nError stats: none (no finite points in this bucket).")
        return

    frac_err = np.abs(y_pred - y_true) / (np.abs(y_true) + EPS_FRACTIONAL)
    mean_frac_err = float(np.mean(frac_err))
    max_frac_err = float(np.max(frac_err))

    diff = y_pred - y_true
    mae_lin = float(np.mean(np.abs(diff)))
    rmse_lin = float(np.sqrt(np.mean(diff ** 2)))

    y_true_clamped = np.maximum(y_true, PLOT_FLOOR)
    y_pred_clamped = np.maximum(y_pred, PLOT_FLOOR)
    log_err = np.log10(y_pred_clamped) - np.log10(y_true_clamped)

    mae_log = float(np.mean(np.abs(log_err)))
    rmse_log = float(np.sqrt(np.mean(log_err ** 2)))
    bias_log = float(np.mean(log_err))

    pcts = np.percentile(frac_err * 100, [50, 90, 95, 99, 99.9])

    print("\nError stats:")
    print(f"Points:           {n}")
    print(f"Average % error:  {100 * mean_frac_err:.3f}")
    print(f"Linear MAE:       {mae_lin:.4e}")
    print(f"Linear RMSE:      {rmse_lin:.4e}")
    print(f"Log10 MAE (dex):  {mae_log:.4f}")
    print(f"Log10 RMSE (dex): {rmse_log:.4f}")
    print(f"Log10 Bias (dex): {bias_log:.4f}")
    print(
        "Frac Err % (50/90/95/99/99.9): "
        f"{pcts[0]:.2f} / {pcts[1]:.2f} / {pcts[2]:.2f} / {pcts[3]:.2f} / {pcts[4]:.2f}"
    )
    print(f"Max Frac Error:   {100 * max_frac_err:.2f} %")


# ---------------- Main ----------------
def main() -> None:
    os.chdir(REPO)
    seed_everything(42)

    # Load config
    cfg = load_json(MODEL_DIR / "config.json")
    data_cfg = dict(cfg.get("data", {}))
    species_cfg = list(data_cfg.get("species_variables") or [])
    globals_cfg = list(data_cfg.get("global_variables") or [])

    data_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    # Load normalization manifest (authoritative processed column order)
    manifest = load_json(data_dir / "normalization.json")
    meta = manifest.get("meta", {}) or {}
    species_all = list(meta.get("species_variables", []) or [])
    globals_all = list(meta.get("global_variables", []) or [])

    # If config leaves lists empty, it means "use all processed"
    if not species_cfg:
        species_cfg = species_all
    if globals_all and not globals_cfg:
        globals_cfg = globals_all

    idx_species = _indices(species_all, species_cfg)
    idx_globals = _indices(globals_all, globals_cfg) if globals_cfg else []

    norm = NormalizationHelper(manifest)
    keep_idx, _labels = select_species_indices(species_cfg, PLOT_SPECIES)
    n_species_plotted = len(keep_idx)

    # Load exported program
    ep = torch.export.load(MODEL_DIR / EP_FILENAME)
    model = ep.module()

    # Load first test shard
    y_mat, g_all, t_vec = load_first_test_shard(data_dir)
    n_use = min(N_SAMPLES, y_mat.shape[0])

    all_true = []
    all_pred = []

    # For count breakdown
    traj_T = []
    traj_K = []
    traj_Ku = []
    dup_total = 0
    dup_traj = 0

    for i in range(n_use):
        y_traj = y_mat[i][:, idx_species]  # [T, S_model]
        g_vec = g_all[i][idx_globals] if idx_globals else np.empty((0,), dtype=np.float32)
        t_phys = get_time_vector_for_sample(t_vec, i)

        traj_T.append(int(len(t_phys)))

        y0 = y_traj[0]

        y_batch, dt_batch, g_batch, q_idx = prepare_batch(
            y0=y0,
            g=g_vec,
            t_phys=t_phys,
            q_count=Q_COUNT,
            norm=norm,
            species=species_cfg,
            globals_=globals_cfg,
        )

        K = int(q_idx.size)
        Ku = int(np.unique(q_idx).size)
        traj_K.append(K)
        traj_Ku.append(Ku)
        if Ku < K:
            dup_traj += 1
            dup_total += (K - Ku)

        y_pred_norm = run_inference(model, y_batch, dt_batch, g_batch)  # [K, S_model]
        y_pred = norm.denormalize(y_pred_norm, species_cfg).cpu().numpy()  # [K, S_model]

        y_true_sel = y_traj[q_idx, :]  # [K, S_model]

        y_true_sel = y_true_sel[:, keep_idx]
        y_pred_sel = y_pred[:, keep_idx]

        all_true.append(y_true_sel.reshape(-1))
        all_pred.append(y_pred_sel.reshape(-1))

    y_true_flat = np.concatenate(all_true, axis=0)
    y_pred_flat = np.concatenate(all_pred, axis=0)

    # Global count components
    traj_T_arr = np.asarray(traj_T, dtype=int)
    traj_K_arr = np.asarray(traj_K, dtype=int)
    traj_Ku_arr = np.asarray(traj_Ku, dtype=int)

    total_q = int(np.sum(traj_K_arr))
    raw_points = int(total_q * n_species_plotted)
    # This should match unless something changes in how we flatten/append.
    if raw_points != int(y_true_flat.size):
        print(
            "WARNING: raw_points != y_true_flat.size "
            f"({raw_points} vs {y_true_flat.size}). Using y_true_flat.size for reporting."
        )
        raw_points = int(y_true_flat.size)

    finite_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    finite_points = int(np.sum(finite_mask))

    context: Dict[str, object] = {
        "n_traj": int(n_use),
        "n_species": int(n_species_plotted),
        "q_requested": int(Q_COUNT),
        "total_q": int(total_q),
        "raw_points": int(raw_points),
        "finite_points": int(finite_points),
        "t_stats": _stats_1d(traj_T_arr),
        "k_stats": _stats_1d(traj_K_arr),
        "ku_stats": _stats_1d(traj_Ku_arr),
        "dup_total": int(dup_total),
        "dup_traj": int(dup_traj),
    }

    # Bucket 1: y_true > threshold (and finite)
    bucket_mask = finite_mask & (y_true_flat > ABUND_BUCKET_THRESHOLD)
    bucket_points = int(np.sum(bucket_mask))

    print_count_breakdown(
        context=context,
        points_after_filters=bucket_points,
        title=f"Bucket: y_true > {ABUND_BUCKET_THRESHOLD:.1e}",
    )
    print_error_stats(y_true_flat[bucket_mask], y_pred_flat[bucket_mask], title="(stats)")

    # Bucket 2: all finite points
    print_count_breakdown(
        context=context,
        points_after_filters=finite_points,
        title="Bucket: all finite points",
    )
    print_error_stats(y_true_flat[finite_mask], y_pred_flat[finite_mask], title="(stats)")

    out_png = MODEL_DIR / "plots" / f"accuracy.png"
    plot_pred_vs_true_scatter(y_true_flat, y_pred_flat, out_png)


if __name__ == "__main__":
    main()
