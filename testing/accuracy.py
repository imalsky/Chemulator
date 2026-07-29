#!/usr/bin/env python3
"""Scatter + error metrics on test split using physical-I/O model.

Sample size is configurable via --n-samples / --n-shards (or the environment
variables CHEMULATOR_N_SAMPLES / CHEMULATOR_N_SHARDS). Pass
--n-samples 84 --n-shards 1 to reproduce the published run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import prepare_platform_environment

prepare_platform_environment()

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator


REPO = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).with_name("science.mplstyle")
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_model"))
).expanduser().resolve()


try:
    plt.style.use(str(STYLE_PATH))
except OSError:
    warnings.warn("science.mplstyle not found; using matplotlib defaults.")

MODEL_PATH = MODEL_DIR / "physical_model_k1_cpu.pt2"
METADATA_PATH = MODEL_DIR / "physical_model_metadata.json"
CONFIG_PATH = MODEL_DIR / "config.json"

PROCESSED_DIR_ENV = "CHEMULATOR_PROCESSED_DIR"

# Sampling size. The published run used 84 trajectories from a single shard;
# pass --n-samples 84 --n-shards 1 to reproduce exactly. The default is larger
# so the tail percentiles (99, 99.9) rest on more than one shard; it costs
# tens of seconds on a laptop CPU.
DEFAULT_N_SAMPLES = 2048
DEFAULT_N_SHARDS = 8
Q_COUNT = 100
PLOT_SPECIES: List[str] = []

# Additive floor in the fractional-error denominator:
#   frac_err = |y_pred - y_true| / (|y_true| + EPS_FRACTIONAL)
# It keeps the ratio finite where y_true underflows, and it also means the
# reported fractional errors are deflated for |y_true| of order 1e-20 and
# below. The printed stats spell out the definition so the floor cannot be
# mistaken for a plain relative error.
EPS_FRACTIONAL = 1e-20

# Lower clamp applied to the scatter axes and to the log10 error statistics.
# Points below it are drawn (and scored) at the floor rather than at their true
# value, so the affected fraction is printed alongside the stats.
PLOT_FLOOR = 5e-20

ABUND_BUCKET_THRESHOLD = 1e-10


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cfg_path(path_like: str | os.PathLike[str], *, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _select_species(species_order: List[str], plot_species: List[str]) -> Tuple[List[int], List[str]]:
    labels = [n.removesuffix("_evolution") for n in species_order]
    keep = [i for i, b in enumerate(labels) if (not plot_species) or (b in plot_species)]
    if not keep:
        raise RuntimeError("No species selected for plotting")
    return keep, [labels[i] for i in keep]


def _resolve_processed_dir(cfg: dict) -> Path:
    """Resolve the processed-data root: env override first, then the config.

    No silent fallback. A hydrated config records paths.processed_data_dir as
    it was at training time, so it can be stale relative to the shards on disk;
    the error names the override and the likely alternative rather than
    guessing. Same env var as testing/export.py.
    """
    raw = os.getenv(PROCESSED_DIR_ENV)
    if raw:
        processed_dir = Path(raw).expanduser().resolve()
    else:
        processed_dir = _resolve_cfg_path(cfg["paths"]["processed_data_dir"], base_dir=REPO)

    if (processed_dir / "test").is_dir():
        return processed_dir

    alt = processed_dir / "processed"
    hint = ""
    if (alt / "test").is_dir():
        hint = f" A test split does exist at {alt}, so the configured path is likely stale."
    raise FileNotFoundError(
        f"No test split under processed data dir: {processed_dir}."
        f" Set {PROCESSED_DIR_ENV} to override paths.processed_data_dir.{hint}"
    )


def _load_test_trajectories(
    processed_dir: Path, *, n_samples: int, n_shards: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load up to n_samples trajectories spread over the first n_shards shards.

    Trajectories are taken in file order from each shard, so --n-shards 1
    --n-samples 84 reproduces the single-shard sample used for the published
    figure. Returns per-trajectory time rows, so all shards must share the
    time grid (a preprocessing invariant).
    """
    shard_paths = sorted((processed_dir / "test").glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No test shards found in {processed_dir / 'test'}")

    selected = shard_paths[:n_shards]
    per_shard = -(-n_samples // len(selected))

    y_parts: List[np.ndarray] = []
    g_parts: List[np.ndarray] = []
    t_parts: List[np.ndarray] = []
    used: List[str] = []
    remaining = n_samples

    for path in selected:
        if remaining <= 0:
            break
        with np.load(path, allow_pickle=False) as d:
            y_mat = np.asarray(d["y_mat"], dtype=np.float32)
            g_mat = np.asarray(d["globals"], dtype=np.float32)
            t_vec = np.asarray(d["t_vec"], dtype=np.float32)

        k = min(per_shard, remaining, y_mat.shape[0])
        if k <= 0:
            continue
        if t_vec.ndim == 1:
            t_rows = np.broadcast_to(t_vec, (k, t_vec.shape[0])).copy()
        elif t_vec.ndim == 2:
            if t_vec.shape[0] != y_mat.shape[0]:
                raise ValueError(
                    f"t_vec rows do not match trajectories in {path.name}: "
                    f"{t_vec.shape[0]} != {y_mat.shape[0]}"
                )
            t_rows = t_vec[:k]
        else:
            raise ValueError(f"Unsupported t_vec shape in {path.name}: {t_vec.shape}")

        y_parts.append(y_mat[:k])
        g_parts.append(g_mat[:k])
        t_parts.append(t_rows)
        used.append(path.name)
        remaining -= k

    if not y_parts:
        raise RuntimeError("No trajectories loaded from the test split")

    grid = t_parts[0].shape[1]
    if any(part.shape[1] != grid for part in t_parts):
        raise RuntimeError("Test shards disagree on the time-grid length")

    return (
        np.concatenate(y_parts, axis=0),
        np.concatenate(g_parts, axis=0),
        np.concatenate(t_parts, axis=0),
        used,
    )


def _time_for_sample(t_vec: np.ndarray, idx: int) -> np.ndarray:
    if t_vec.ndim == 1:
        return t_vec
    if t_vec.ndim == 2:
        return t_vec[idx]
    raise ValueError(f"Unsupported t_vec shape: {t_vec.shape}")


def _prepare_batch(y0: np.ndarray, g: np.ndarray, t_phys: np.ndarray, q_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    M = len(t_phys)
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)
    dt_sec = (t_phys[q_idx] - t_phys[0]).astype(np.float32)
    if np.any(dt_sec <= 0.0):
        raise RuntimeError("Non-positive dt encountered in sampled query points")

    y_batch = torch.from_numpy(y0[None, :]).float().repeat(qn, 1)
    g_batch = torch.from_numpy(g[None, :]).float().repeat(qn, 1)
    dt_batch = torch.from_numpy(dt_sec).float().view(-1, 1)
    return y_batch, dt_batch, g_batch, q_idx


def _plot_scatter(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, out_path: Path) -> None:
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = np.clip(y_true_flat[mask], PLOT_FLOOR, None)
    y_pred = np.clip(y_pred_flat[mask], PLOT_FLOOR, None)

    lo = min(y_true.min(), y_pred.min()) * 0.8
    hi = max(y_true.max(), y_pred.max()) * 1.2

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
    # Disable minor ticks explicitly; some matplotlib versions can error on LogLocator(numticks=0).
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_box_aspect(1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stats_1d(x: np.ndarray) -> Tuple[int, float, int, int]:
    if x.size == 0:
        return 0, float("nan"), 0, 0
    return int(x.size), float(np.median(x)), int(np.min(x)), int(np.max(x))


def _print_count_breakdown(context: Dict[str, object], points_after_filters: int, title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"Trajectories used:            {int(context['n_traj'])}")
    print(f"Anchors:                     {int(context['n_traj'])} (1 anchor per trajectory at t0)")
    print(f"Requested Q_COUNT:           {int(context['q_requested'])}")
    print(
        "Trajectory lengths T (med/min/max): "
        f"{context['t_stats'][1]:.1f} / {context['t_stats'][2]} / {context['t_stats'][3]}"
    )
    print(
        "Query times K (med/min/max):        "
        f"{context['k_stats'][1]:.1f} / {context['k_stats'][2]} / {context['k_stats'][3]}"
    )
    print(
        "Unique query times Ku (med/min/max):"
        f"{context['ku_stats'][1]:.1f} / {context['ku_stats'][2]} / {context['ku_stats'][3]}"
    )
    if int(context["dup_traj"]) > 0:
        print(
            f"Note: duplicate q_idx present in {int(context['dup_traj'])}/{int(context['n_traj'])} trajectories "
            f"(total duplicate indices counted as separate points: {int(context['dup_total'])})."
        )

    print(f"Test shards read:            {int(context['n_shards'])} ({', '.join(context['shards'])})")
    print(f"Species plotted:             {int(context['n_species'])}")
    print(f"Total query times (sum K):   {int(context['total_q'])}")
    print(
        f"Raw points:                  {int(context['raw_points'])} = "
        f"(sum K) {int(context['total_q'])} * (species) {int(context['n_species'])}"
    )
    print(f"Finite points:               {int(context['finite_points'])} (after dropping non-finite y_true/y_pred)")
    print(f"Points (this bucket):        {int(points_after_filters)}")


def _print_error_stats(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> None:
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true = y_true_flat[mask]
    y_pred = y_pred_flat[mask]
    if y_true.size == 0:
        print("\nError stats: none (no finite points in this bucket).")
        return

    frac_err = np.abs(y_pred - y_true) / (np.abs(y_true) + EPS_FRACTIONAL)
    diff = y_pred - y_true
    y_true_clamped = np.maximum(y_true, PLOT_FLOOR)
    y_pred_clamped = np.maximum(y_pred, PLOT_FLOOR)
    log_err = np.log10(y_pred_clamped) - np.log10(y_true_clamped)

    clamped = (y_true < PLOT_FLOOR) | (y_pred < PLOT_FLOOR)
    clamped_pct = 100.0 * float(np.mean(clamped))

    pcts = np.percentile(frac_err * 100, [50, 90, 95, 99, 99.9])
    print("\nError stats:")
    print(f"Definition:       frac err = |y_pred - y_true| / (|y_true| + {EPS_FRACTIONAL:.1e})")
    print(
        f"Log10 stats clamp both values at {PLOT_FLOOR:.1e}: "
        f"{clamped_pct:.2f} % of points in this bucket are affected"
    )
    print(f"Points:           {int(y_true.size)}")
    print(f"Average % error:  {100.0 * float(np.mean(frac_err)):.3f}")
    print(f"Linear MAE:       {float(np.mean(np.abs(diff))):.4e}")
    print(f"Linear RMSE:      {float(np.sqrt(np.mean(diff ** 2))):.4e}")
    print(f"Log10 MAE (dex):  {float(np.mean(np.abs(log_err))):.4f}")
    print(f"Log10 RMSE (dex): {float(np.sqrt(np.mean(log_err ** 2))):.4f}")
    print(f"Log10 Bias (dex): {float(np.mean(log_err)):.4f}")
    print(
        "Frac Err % (50/90/95/99/99.9): "
        f"{pcts[0]:.2f} / {pcts[1]:.2f} / {pcts[2]:.2f} / {pcts[3]:.2f} / {pcts[4]:.2f}"
    )
    print(f"Max Frac Error:   {100.0 * float(np.max(frac_err)):.2f} %")


def _print_floor_disclosure(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> None:
    """State how much of the scatter is clamped onto the plot floor.

    Both scatter axes clamp at PLOT_FLOOR, so points with true and predicted
    values below it are drawn at the same location, exactly on the 1:1 line.
    """
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    n = int(np.sum(mask))
    if n == 0:
        print("\nPlot floor: no finite points to report.")
        return

    below_true = y_true_flat[mask] < PLOT_FLOOR
    below_pred = y_pred_flat[mask] < PLOT_FLOOR
    both = below_true & below_pred

    print(f"\n--- Plot floor disclosure (PLOT_FLOOR = {PLOT_FLOOR:.1e}) ---")
    print(f"Finite points:               {n}")
    print(f"True below floor:            {int(below_true.sum())} ({100.0 * float(np.mean(below_true)):.2f} %)")
    print(f"Predicted below floor:       {int(below_pred.sum())} ({100.0 * float(np.mean(below_pred)):.2f} %)")
    print(
        f"Both below floor:            {int(both.sum())} ({100.0 * float(np.mean(both)):.2f} %) "
        "drawn at the floor, i.e. stacked on the 1:1 line"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(os.getenv("CHEMULATOR_N_SAMPLES", str(DEFAULT_N_SAMPLES))),
        help="Trajectories to evaluate, spread over the selected shards (env: CHEMULATOR_N_SAMPLES)",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=int(os.getenv("CHEMULATOR_N_SHARDS", str(DEFAULT_N_SHARDS))),
        help="Test shards to read, in sorted file order (env: CHEMULATOR_N_SHARDS)",
    )
    args = parser.parse_args(argv)
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    if args.n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_json(CONFIG_PATH)
    meta = _load_json(METADATA_PATH)
    ep = torch.export.load(MODEL_PATH)
    model = ep.module()

    processed_dir = _resolve_processed_dir(cfg)
    species_order = [str(x) for x in meta["species_order"]]
    globals_order = [str(x) for x in meta["globals_order"]]
    cfg_species = [str(x) for x in cfg["data"]["species_variables"]]
    cfg_globals = [str(x) for x in cfg["data"]["global_variables"]]

    if species_order != cfg_species:
        raise ValueError("metadata species_order must match cfg.data.species_variables")
    if globals_order != cfg_globals:
        raise ValueError("metadata globals_order must match cfg.data.global_variables")

    keep_idx, _ = _select_species(species_order, PLOT_SPECIES)
    n_species_plotted = len(keep_idx)

    y_mat, g_mat, t_vec, shards_used = _load_test_trajectories(
        processed_dir, n_samples=args.n_samples, n_shards=args.n_shards
    )
    n_use = int(y_mat.shape[0])

    all_true = []
    all_pred = []
    traj_T: List[int] = []
    traj_K: List[int] = []
    traj_Ku: List[int] = []
    dup_total = 0
    dup_traj = 0

    for i in range(n_use):
        y_traj = y_mat[i]
        g_vec = g_mat[i] if globals_order else np.empty((0,), dtype=np.float32)
        t_phys = _time_for_sample(t_vec, i)

        traj_T.append(int(len(t_phys)))
        y_batch, dt_batch, g_batch, q_idx = _prepare_batch(y_traj[0], g_vec, t_phys, Q_COUNT)

        K = int(q_idx.size)
        Ku = int(np.unique(q_idx).size)
        traj_K.append(K)
        traj_Ku.append(Ku)
        if Ku < K:
            dup_traj += 1
            dup_total += (K - Ku)

        y_pred = model(y_batch, dt_batch, g_batch)[:, 0, :].detach().cpu().numpy()
        y_true_sel = y_traj[q_idx, :]

        all_true.append(y_true_sel[:, keep_idx].reshape(-1))
        all_pred.append(y_pred[:, keep_idx].reshape(-1))

    y_true_flat = np.concatenate(all_true, axis=0)
    y_pred_flat = np.concatenate(all_pred, axis=0)

    total_q = int(np.sum(np.asarray(traj_K, dtype=int)))
    raw_points = total_q * n_species_plotted
    finite_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    finite_points = int(np.sum(finite_mask))

    context: Dict[str, object] = {
        "n_traj": int(n_use),
        "n_species": int(n_species_plotted),
        "q_requested": int(Q_COUNT),
        "total_q": int(total_q),
        "raw_points": int(raw_points),
        "finite_points": int(finite_points),
        "t_stats": _stats_1d(np.asarray(traj_T, dtype=int)),
        "k_stats": _stats_1d(np.asarray(traj_K, dtype=int)),
        "ku_stats": _stats_1d(np.asarray(traj_Ku, dtype=int)),
        "dup_total": int(dup_total),
        "dup_traj": int(dup_traj),
        "n_shards": int(len(shards_used)),
        "shards": list(shards_used),
    }

    bucket_mask = finite_mask & (y_true_flat > ABUND_BUCKET_THRESHOLD)
    bucket_points = int(np.sum(bucket_mask))

    _print_count_breakdown(context, bucket_points, f"Bucket: y_true > {ABUND_BUCKET_THRESHOLD:.1e}")
    _print_error_stats(y_true_flat[bucket_mask], y_pred_flat[bucket_mask])

    _print_count_breakdown(context, finite_points, "Bucket: all finite points")
    _print_error_stats(y_true_flat[finite_mask], y_pred_flat[finite_mask])

    _print_floor_disclosure(y_true_flat, y_pred_flat)

    out_png = MODEL_DIR / "plots" / "accuracy.png"
    _plot_scatter(y_true_flat, y_pred_flat, out_png)
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Processed data: {processed_dir}")
    print(f"Plot saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
