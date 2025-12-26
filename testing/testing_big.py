#!/usr/bin/env python3
"""
testing_big.py — FlowMap EXPORT Accuracy Benchmark Suite (no argparse)

Requirements implemented:
- NO runtime/time benchmarking (no perf_counter, no throughput, no synthetic perf).
- Loads data exactly like your working script:
    data_dir / SPLIT / shard_*.npz with keys: y_mat, globals, t_vec
- Uses <= 100 time points per trajectory
- Only evaluates times within [1e-3, 1e8] seconds
- Only accuracy / emulator-quality benchmarks + plots
- Saves all outputs to: MODEL_DIR / "plots"

Outputs (MODEL_DIR/plots):
  - export_acc_summary.json
  - export_acc_per_species.csv
  - export_acc_dt_bins.csv
  - export_acc_worst_cases.csv
  - export_plot_dt_curve.png
  - export_plot_pair_mae_hist.png
  - export_plot_species_mae_bar.png
  - export_plot_parity_scatter.png
  - export_plot_mass_error_hist.png
  - export_plot_neg_rate_by_dt.png
  - export_plot_gt_mass_hist.png
  - export_plot_pred_mass_hist.png
  - export_plot_pred_gt_mass_scatter.png
  - export_plot_gt_pred_sum_scatter.png
"""

from __future__ import annotations

import os
import sys
import csv
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================================
# GLOBALS (edit these; no argparse)
# ======================================================================================

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

MODEL_DIR = REPO / "models" / "big_big_big"
EXPORT_FILENAME = "export_k1_cpu.pt2"

SPLIT = "test"
SEED = 42

# Fast defaults (increase for more coverage)
MAX_SHARDS: int = 1                 # number of shard_*.npz files to evaluate
TRAJS_PER_SHARD: int = 12           # number of trajectories sampled per shard
SHUFFLE_TRAJS: bool = True          # random subset per shard (seeded)

# Query sampling (HARD CAPS required)
MAX_TIMEPOINTS: int = 100           # <= 100
TMIN: float = 1e-3                  # only evaluate times >= 1e-3 s
TMAX: float = 1e8                   # only evaluate times <= 1e8 s
QUERY_MODE: str = "lin"             # "lin" or "log" selection over available indices

# Inference batching (fewer model calls = faster)
INFER_BATCH: int = 4096             # number of (traj,time) pairs per forward call

# Metrics settings
EPS_PHYS: float = 1e-30             # floor for log10 and relative denom
DT_BINS: int = 20                   # fewer bins = faster & cleaner plots
DT_BINS_MIN: float = TMIN
DT_BINS_MAX: float = TMAX

# Device/dtype
DEVICE_STR: str = "auto"            # "auto", "cpu", "mps", "cuda"
DTYPE_STR: str = "auto"             # "auto", "float32", "bfloat16", "float16"

# Reservoir samples for plots (small memory)
RESERVOIR_PAIR_MAE: int = 20000      # histogram of per-pair MAE(log10)
RESERVOIR_PARITY_POINTS: int = 40000 # scatter points (across species)
RESERVOIR_MASS_ERR: int = 20000      # histogram of mass error

# Worst cases (written to CSV)
KEEP_WORST_CASES: int = 50           # keep top-N worst per-pair MAE(log10)

# Plots toggles
MAKE_PLOTS: bool = True
PLOT_TOPK_SPECIES: int = 24          # bar plot shows top-K by MAE(log10)


# ======================================================================================
# Imports from your codebase
# ======================================================================================

from utils import load_json_config, dump_json, seed_everything  # type: ignore
from normalizer import NormalizationHelper  # type: ignore


# ======================================================================================
# Small utilities
# ======================================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))

def _device_available(name: str) -> bool:
    name = str(name).lower()
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return False

def _pick_device_from_export_name(export_path: Path) -> Optional[str]:
    n = export_path.name.lower()
    if "cpu" in n:
        return "cpu"
    if "mps" in n:
        return "mps"
    if ("cuda" in n) or ("gpu" in n):
        return "cuda"
    return None

def _choose_device(export_path: Path, requested: str) -> torch.device:
    req = str(requested).strip().lower()
    hinted = _pick_device_from_export_name(export_path)

    if req != "auto":
        if not _device_available(req):
            raise RuntimeError(f"Requested DEVICE_STR={req!r} but it is not available.")
        return torch.device(req)

    # Prevent CPU-export-on-MPS/CUDA mismatch by respecting filename hint
    if hinted is not None and _device_available(hinted):
        return torch.device(hinted)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _resolve_dtype(device: torch.device, dtype_str: str) -> torch.dtype:
    s = str(dtype_str).strip().lower()
    if s in ("float32", "float", "fp32"):
        return torch.float32
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "half", "fp16"):
        return torch.float16
    # auto
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32

def _try_set_eval(mod: torch.nn.Module) -> None:
    try:
        mod.eval()
    except Exception:
        pass

def _load_exported_module(export_path: Path) -> torch.nn.Module:
    exported = torch.export.load(str(export_path))
    mod = exported.module()   # do NOT call ExportedProgram.eval()
    _try_set_eval(mod)
    return mod

def _find_cfg_path(model_dir: Path) -> Path:
    for p in [
        model_dir / "config.json",
        model_dir / "trial_config.final.json",
        model_dir / "config.final.json",
        model_dir / "config.used.json",
        model_dir / "config.snapshot.json",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find config.json / trial_config.final.json in MODEL_DIR.")

def _resolve_data_dir(cfg: Dict[str, Any]) -> Path:
    raw = cfg["paths"]["processed_data_dir"]
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO / p).resolve()

def _list_shards(data_dir: Path, split: str) -> List[Path]:
    shards = sorted((data_dir / split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found in {data_dir / split}")
    return shards

def _load_shard_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path) as d:
            y = d["y_mat"].astype(np.float32)   # [N,T,S]
            g = d["globals"].astype(np.float32) # [N,G]
            t = d["t_vec"].astype(np.float32)   # [T] or [N,T]
        return y, g, t
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"{path} is not a valid .npz zip: {e}") from e

def _get_time_for_traj(t_vec: np.ndarray, i: int) -> np.ndarray:
    return t_vec if t_vec.ndim == 1 else t_vec[i]

def _choose_query_indices_by_time(t_phys: np.ndarray) -> np.ndarray:
    """
    Select up to MAX_TIMEPOINTS indices j (j>0) such that TMIN<=t_phys[j]<=TMAX.
    Returns sorted indices.
    """
    T = int(t_phys.size)
    if T <= 1:
        return np.zeros((0,), dtype=np.int64)

    valid = np.where((t_phys >= float(TMIN)) & (t_phys <= float(TMAX)))[0]
    valid = valid[valid > 0]
    if valid.size == 0:
        return np.zeros((0,), dtype=np.int64)

    qn = min(int(MAX_TIMEPOINTS), int(valid.size))
    if qn <= 0:
        return np.zeros((0,), dtype=np.int64)

    if QUERY_MODE.lower() == "log":
        xs = np.geomspace(1, valid.size, qn)
        pick = np.unique(np.clip(np.rint(xs).astype(np.int64) - 1, 0, valid.size - 1))
        idx = valid[pick]
    else:
        pick = np.linspace(0, valid.size - 1, qn).round().astype(np.int64)
        pick = np.unique(np.clip(pick, 0, valid.size - 1))
        idx = valid[pick]

    idx = np.unique(np.clip(idx, 1, T - 1))
    return idx.astype(np.int64)

def _logspace_edges(xmin: float, xmax: float, n_bins: int) -> np.ndarray:
    xmin = max(float(xmin), 1e-300)
    xmax = max(float(xmax), xmin * (1.0 + 1e-12))
    return np.logspace(np.log10(xmin), np.log10(xmax), int(n_bins) + 1)

def _bin_indices_log(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx

def _safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 0.0, None) + float(EPS_PHYS))


# ======================================================================================
# Reservoir sampling
# ======================================================================================

class Reservoir:
    def __init__(self, k: int, seed: int) -> None:
        self.k = int(k)
        self.rng = random.Random(int(seed))
        self.n_seen = 0
        self.buf: List[Any] = []

    def add_many(self, xs: Sequence[Any]) -> None:
        for v in xs:
            self.n_seen += 1
            if len(self.buf) < self.k:
                self.buf.append(v)
            else:
                j = self.rng.randint(1, self.n_seen)
                if j <= self.k:
                    self.buf[j - 1] = v

    def as_array_float(self) -> np.ndarray:
        if not self.buf:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(self.buf, dtype=np.float64)

    def count(self) -> int:
        return int(self.n_seen)


# ======================================================================================
# Metrics accumulation
# ======================================================================================

@dataclass
class Accum:
    n_pairs: int
    n_species: int

    # log10-space metrics (per species)
    sum_abs_log: np.ndarray
    sum_sq_log: np.ndarray
    max_abs_log: np.ndarray
    sum_true_log: np.ndarray
    sum_true2_log: np.ndarray

    # relative metrics (per species, linear)
    sum_abs_rel: np.ndarray
    max_abs_rel: np.ndarray

    # sanity counts (per pair)
    n_pred_nan: int
    n_pred_inf: int
    n_pred_neg_any: int
    n_pred_gt1_any: int

    # dt bins
    dt_edges: np.ndarray
    dt_count: np.ndarray
    dt_sum_pair_mae_log: np.ndarray
    dt_neg_count: np.ndarray  # number of pairs in bin with any negative pred

    # sum(species) mass stats
    sum_true_mass_res: Reservoir
    sum_pred_mass_res: Reservoir

def _init_accum(S: int, dt_bins: int) -> Accum:
    edges = _logspace_edges(DT_BINS_MIN, DT_BINS_MAX, dt_bins)
    return Accum(
        n_pairs=0,
        n_species=S,
        sum_abs_log=np.zeros((S,), dtype=np.float64),
        sum_sq_log=np.zeros((S,), dtype=np.float64),
        max_abs_log=np.zeros((S,), dtype=np.float64),
        sum_true_log=np.zeros((S,), dtype=np.float64),
        sum_true2_log=np.zeros((S,), dtype=np.float64),
        sum_abs_rel=np.zeros((S,), dtype=np.float64),
        max_abs_rel=np.zeros((S,), dtype=np.float64),
        n_pred_nan=0,
        n_pred_inf=0,
        n_pred_neg_any=0,
        n_pred_gt1_any=0,
        dt_edges=edges,
        dt_count=np.zeros((dt_bins,), dtype=np.int64),
        dt_sum_pair_mae_log=np.zeros((dt_bins,), dtype=np.float64),
        dt_neg_count=np.zeros((dt_bins,), dtype=np.int64),
        sum_true_mass_res=Reservoir(k=min(RESERVOIR_MASS_ERR, 20000), seed=SEED + 33),
        sum_pred_mass_res=Reservoir(k=min(RESERVOIR_MASS_ERR, 20000), seed=SEED + 34),
    )

def _update_bins(acc: Accum, dt_phys: np.ndarray, pair_mae_log: np.ndarray, pred_neg_any: np.ndarray) -> None:
    idx = _bin_indices_log(dt_phys, acc.dt_edges)
    for b in range(acc.dt_count.size):
        m = (idx == b)
        if not np.any(m):
            continue
        c = int(np.sum(m))
        acc.dt_count[b] += c
        acc.dt_sum_pair_mae_log[b] += float(np.sum(pair_mae_log[m]))
        acc.dt_neg_count[b] += int(np.sum(pred_neg_any[m]))


# ======================================================================================
# Plot helpers
# ======================================================================================

def _plot_dt_curve(out_dir: Path, edges: np.ndarray, counts: np.ndarray, sum_mae: np.ndarray) -> None:
    centers = np.sqrt(edges[:-1] * edges[1:])
    mae = np.full_like(centers, np.nan, dtype=np.float64)
    m = counts > 0
    mae[m] = sum_mae[m] / counts[m]

    plt.figure()
    plt.xscale("log")
    plt.plot(centers, mae)
    plt.xlabel("Δt (s)")
    plt.ylabel("Pair MAE in log10(phys)")
    plt.title("Error vs Δt (binned)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_dt_curve.png", dpi=200)
    plt.close()

def _plot_pair_hist(out_dir: Path, pair_mae_res: Reservoir) -> None:
    x = pair_mae_res.as_array_float()
    if x.size == 0:
        return
    plt.figure()
    plt.hist(x, bins=80)
    plt.xlabel("Pair MAE in log10(phys)")
    plt.ylabel("Count")
    plt.title("Per-pair error distribution (reservoir sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_pair_mae_hist.png", dpi=200)
    plt.close()

def _plot_species_bar(out_dir: Path, species: List[str], mae_log: np.ndarray) -> None:
    order = np.argsort(mae_log)[::-1]
    topk = min(int(PLOT_TOPK_SPECIES), order.size)
    idx = order[:topk]
    labs = [species[i] for i in idx]
    vals = mae_log[idx]

    plt.figure(figsize=(10, max(3.0, 0.25 * topk)))
    y = np.arange(topk)
    plt.barh(y, vals)
    plt.yticks(y, labs)
    plt.gca().invert_yaxis()
    plt.xlabel("MAE in log10(phys)")
    plt.title(f"Top-{topk} species by MAE (log10)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_species_mae_bar.png", dpi=200)
    plt.close()

def _plot_parity(out_dir: Path, parity_res: Reservoir) -> None:
    if not parity_res.buf:
        return
    arr = np.asarray(parity_res.buf, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return
    x = arr[:, 0]
    y = arr[:, 1]

    plt.figure()
    plt.scatter(x, y, s=3, alpha=0.15)
    lo = np.nanpercentile(np.concatenate([x, y]), 1)
    hi = np.nanpercentile(np.concatenate([x, y]), 99)
    plt.plot([lo, hi], [lo, hi], lw=2)
    plt.xlabel("log10(true + eps)")
    plt.ylabel("log10(pred + eps)")
    plt.title("Parity plot (reservoir sample across species)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_parity_scatter.png", dpi=200)
    plt.close()

def _plot_mass_hists(out_dir: Path, true_sum: Reservoir, pred_sum: Reservoir) -> None:
    xt = true_sum.as_array_float()
    xp = pred_sum.as_array_float()
    if xt.size:
        plt.figure()
        plt.hist(xt, bins=80)
        plt.xlabel("sum(true) (phys)")
        plt.ylabel("Count")
        plt.title("Distribution of sum(true) across evaluated pairs (reservoir sample)")
        plt.tight_layout()
        plt.savefig(out_dir / "export_plot_gt_mass_hist.png", dpi=200)
        plt.close()
    if xp.size:
        plt.figure()
        plt.hist(xp, bins=80)
        plt.xlabel("sum(pred) (phys)")
        plt.ylabel("Count")
        plt.title("Distribution of sum(pred) across evaluated pairs (reservoir sample)")
        plt.tight_layout()
        plt.savefig(out_dir / "export_plot_pred_mass_hist.png", dpi=200)
        plt.close()

def _plot_mass_scatter(out_dir: Path, true_sum: Reservoir, pred_sum: Reservoir) -> None:
    xt = true_sum.as_array_float()
    xp = pred_sum.as_array_float()
    n = min(xt.size, xp.size)
    if n == 0:
        return
    x = xt[:n]
    y = xp[:n]
    plt.figure()
    plt.scatter(x, y, s=4, alpha=0.2)
    lo = np.nanpercentile(np.concatenate([x, y]), 1)
    hi = np.nanpercentile(np.concatenate([x, y]), 99)
    plt.plot([lo, hi], [lo, hi], lw=2)
    plt.xlabel("sum(true) (phys)")
    plt.ylabel("sum(pred) (phys)")
    plt.title("Sum(species) parity (reservoir sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_pred_gt_mass_scatter.png", dpi=200)
    plt.close()

def _plot_sum_error_scatter(out_dir: Path, true_sum: Reservoir, pred_sum: Reservoir) -> None:
    xt = true_sum.as_array_float()
    xp = pred_sum.as_array_float()
    n = min(xt.size, xp.size)
    if n == 0:
        return
    err = xp[:n] - xt[:n]
    plt.figure()
    plt.scatter(xt[:n], err, s=4, alpha=0.2)
    plt.xlabel("sum(true) (phys)")
    plt.ylabel("sum(pred) - sum(true) (phys)")
    plt.title("Mass error vs true mass (reservoir sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_gt_pred_sum_scatter.png", dpi=200)
    plt.close()

def _plot_neg_rate_by_dt(out_dir: Path, edges: np.ndarray, counts: np.ndarray, neg_count: np.ndarray) -> None:
    centers = np.sqrt(edges[:-1] * edges[1:])
    rate = np.full_like(centers, np.nan, dtype=np.float64)
    m = counts > 0
    rate[m] = neg_count[m] / counts[m]

    plt.figure()
    plt.xscale("log")
    plt.plot(centers, rate)
    plt.xlabel("Δt (s)")
    plt.ylabel("Fraction of pairs with any negative pred")
    plt.title("Negativity rate vs Δt (binned)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_neg_rate_by_dt.png", dpi=200)
    plt.close()


# ======================================================================================
# Core evaluation
# ======================================================================================

def _run_forward_batch(
    mod: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    norm: NormalizationHelper,
    species: List[str],
    globals_: List[str],
    y0_phys_list: List[np.ndarray],
    g_phys_list: List[np.ndarray],
    dt_phys_list: List[np.ndarray],
    y_true_phys_list: List[np.ndarray],
    shard_ids: List[int],
    traj_ids: List[int],
    time_ids: List[int],
    t_abs_list: List[np.ndarray],
    acc: Accum,
    pair_mae_res: Reservoir,
    parity_res: Reservoir,
    mass_err_res: Reservoir,
    worst: List[Tuple[float, int, int, float, int]],
    parity_cols: np.ndarray,
) -> None:
    if not y0_phys_list:
        return

    # Concatenate
    y0_phys = np.concatenate(y0_phys_list, axis=0).astype(np.float32)       # [B,S]
    g_phys = np.concatenate(g_phys_list, axis=0).astype(np.float32)         # [B,G]
    dt_phys = np.concatenate(dt_phys_list, axis=0).astype(np.float32)       # [B]
    y_true_phys = np.concatenate(y_true_phys_list, axis=0).astype(np.float32)  # [B,S]
    t_abs = np.concatenate(t_abs_list, axis=0).astype(np.float32)           # [B]

    B = int(y0_phys.shape[0])
    S = int(y0_phys.shape[1])

    # Normalize inputs exactly like your working script
    y0_norm = norm.normalize(torch.from_numpy(y0_phys), species).float()    # [B,S]
    if globals_:
        g_norm = norm.normalize(torch.from_numpy(g_phys), globals_).float() # [B,G]
    else:
        g_norm = torch.from_numpy(g_phys).float()

    dt_norm = norm.normalize_dt_from_phys(torch.from_numpy(dt_phys)).view(-1, 1).float()  # [B,1]

    y_in = y0_norm.to(device=device, dtype=dtype)
    g_in = g_norm.to(device=device, dtype=dtype)
    dt_in = dt_norm.to(device=device, dtype=dtype)

    with torch.inference_mode():
        out = mod(y_in, dt_in, g_in)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3 and out.shape[1] == 1:
            y_pred_norm = out[:, 0, :]
        elif out.ndim == 2:
            y_pred_norm = out
        else:
            raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")

    # Denormalize predictions to phys
    y_pred_phys = norm.denormalize(y_pred_norm.float(), keys=species).cpu().numpy().astype(np.float64)  # [B,S]
    y_true_phys64 = y_true_phys.astype(np.float64)

    # Sanity counts (per pair)
    pred_nan = np.isnan(y_pred_phys).any(axis=1)
    pred_inf = np.isinf(y_pred_phys).any(axis=1)
    pred_neg_any = (y_pred_phys < 0.0).any(axis=1)
    pred_gt1_any = (y_pred_phys > 1.0).any(axis=1)

    acc.n_pred_nan += int(np.sum(pred_nan))
    acc.n_pred_inf += int(np.sum(pred_inf))
    acc.n_pred_neg_any += int(np.sum(pred_neg_any))
    acc.n_pred_gt1_any += int(np.sum(pred_gt1_any))

    # Mass/sum-species diagnostics
    mass_true = y_true_phys64.sum(axis=1)
    mass_pred = y_pred_phys.sum(axis=1)
    mass_err = mass_pred - mass_true
    mass_err_res.add_many(mass_err.tolist())

    acc.sum_true_mass_res.add_many(mass_true.tolist())
    acc.sum_pred_mass_res.add_many(mass_pred.tolist())

    # Errors
    true_log = _safe_log10(y_true_phys64)
    pred_log = _safe_log10(y_pred_phys)

    err_log = pred_log - true_log
    abs_err_log = np.abs(err_log)
    sq_err_log = err_log * err_log

    denom = np.abs(y_true_phys64) + float(EPS_PHYS)
    abs_rel = np.abs(y_pred_phys - y_true_phys64) / denom

    # Accumulate per-species
    acc.n_pairs += B
    acc.sum_abs_log += abs_err_log.sum(axis=0)
    acc.sum_sq_log += sq_err_log.sum(axis=0)
    acc.max_abs_log = np.maximum(acc.max_abs_log, abs_err_log.max(axis=0))
    acc.sum_true_log += true_log.sum(axis=0)
    acc.sum_true2_log += (true_log * true_log).sum(axis=0)

    acc.sum_abs_rel += abs_rel.sum(axis=0)
    acc.max_abs_rel = np.maximum(acc.max_abs_rel, abs_rel.max(axis=0))

    # per-pair summaries
    pair_mae_log = abs_err_log.mean(axis=1)  # [B]
    pair_mae_res.add_many(pair_mae_log.tolist())

    # parity reservoir: (true_log, pred_log) sampled columns
    if RESERVOIR_PARITY_POINTS > 0 and parity_cols.size > 0:
        pts: List[Tuple[float, float]] = []
        for j in parity_cols.tolist():
            pts.extend([(float(true_log[i, j]), float(pred_log[i, j])) for i in range(B)])
        parity_res.add_many(pts)

    # dt bins update
    _update_bins(acc, dt_phys.astype(np.float64), pair_mae_log.astype(np.float64), pred_neg_any.astype(np.bool_))

    # worst cases
    for k in range(B):
        score = float(pair_mae_log[k])
        entry = (score, int(shard_ids[k]), int(traj_ids[k]), float(t_abs[k]), int(time_ids[k]))
        if len(worst) < int(KEEP_WORST_CASES):
            worst.append(entry)
            worst.sort(key=lambda x: x[0], reverse=True)
        else:
            if score > worst[-1][0]:
                worst[-1] = entry
                worst.sort(key=lambda x: x[0], reverse=True)

    # clear lists
    y0_phys_list.clear()
    g_phys_list.clear()
    dt_phys_list.clear()
    y_true_phys_list.clear()
    shard_ids.clear()
    traj_ids.clear()
    time_ids.clear()
    t_abs_list.clear()


def evaluate_export(
    mod: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    data_dir: Path,
    manifest: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    meta = manifest.get("meta", {})
    species: List[str] = list(meta.get("species_variables", []))
    globals_: List[str] = list(meta.get("global_variables", []))
    if not species:
        raise RuntimeError("normalization.json meta.species_variables is empty.")

    norm = NormalizationHelper(manifest)

    shards = _list_shards(data_dir, SPLIT)[: int(MAX_SHARDS)]
    acc = _init_accum(S=len(species), dt_bins=int(DT_BINS))

    pair_mae_res = Reservoir(int(RESERVOIR_PAIR_MAE), seed=SEED + 1)
    parity_res = Reservoir(int(RESERVOIR_PARITY_POINTS), seed=SEED + 2)
    mass_err_res = Reservoir(int(RESERVOIR_MASS_ERR), seed=SEED + 3)

    worst: List[Tuple[float, int, int, float, int]] = []

    # choose parity columns once
    rng = np.random.default_rng(SEED + 123)
    parity_cols = rng.choice(len(species), size=min(4, len(species)), replace=False).astype(np.int64)

    # streaming buffers
    y0_buf: List[np.ndarray] = []
    g_buf: List[np.ndarray] = []
    dt_buf: List[np.ndarray] = []
    ytrue_buf: List[np.ndarray] = []
    shard_id_buf: List[int] = []
    traj_id_buf: List[int] = []
    time_id_buf: List[int] = []
    tabs_buf: List[np.ndarray] = []

    total_trajs_seen = 0

    for shard_idx, shard_path in enumerate(shards):
        y_mat, g_mat, t_vec = _load_shard_npz(shard_path)  # y:[N,T,S] g:[N,G]
        if y_mat.ndim != 3:
            continue

        N, T, S_file = y_mat.shape
        if S_file != len(species):
            raise RuntimeError(f"{shard_path.name}: y_mat S={S_file} but manifest species S={len(species)}")
        if g_mat.ndim != 2 or g_mat.shape[0] != N:
            raise RuntimeError(f"{shard_path.name}: globals shape {g_mat.shape} inconsistent with N={N}")
        if g_mat.shape[1] != len(globals_):
            raise RuntimeError(f"{shard_path.name}: globals G={g_mat.shape[1]} but manifest globals G={len(globals_)}")

        idxs = np.arange(N, dtype=np.int64)
        if SHUFFLE_TRAJS:
            rng2 = np.random.default_rng(SEED + 999 + shard_idx)
            rng2.shuffle(idxs)
        idxs = idxs[: min(int(TRAJS_PER_SHARD), int(N))]

        for traj_i in idxs:
            total_trajs_seen += 1
            t_phys = _get_time_for_traj(t_vec, int(traj_i)).astype(np.float32)
            if t_phys.ndim != 1 or t_phys.size != T:
                continue

            q_idx = _choose_query_indices_by_time(t_phys)
            if q_idx.size == 0:
                continue

            # anchor at t0
            y0 = y_mat[int(traj_i), 0, :].astype(np.float32)            # [S]
            g = g_mat[int(traj_i), :].astype(np.float32)                # [G]

            # selected points
            t_sel = t_phys[q_idx].astype(np.float32)                    # [K]
            dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)  # [K]
            y_true = y_mat[int(traj_i), q_idx, :].astype(np.float32)     # [K,S]

            K = int(q_idx.size)

            y0_buf.append(np.repeat(y0[None, :], K, axis=0))             # [K,S]
            g_buf.append(np.repeat(g[None, :], K, axis=0))               # [K,G]
            dt_buf.append(dt_sec.reshape(-1))                            # [K]
            ytrue_buf.append(y_true)                                     # [K,S]
            tabs_buf.append(t_sel.reshape(-1))                           # [K]
            shard_id_buf.extend([int(shard_idx)] * K)
            traj_id_buf.extend([int(traj_i)] * K)
            time_id_buf.extend([int(x) for x in q_idx.tolist()])

            buffered = sum(x.shape[0] for x in dt_buf)
            if buffered >= int(INFER_BATCH):
                _run_forward_batch(
                    mod=mod,
                    device=device,
                    dtype=dtype,
                    norm=norm,
                    species=species,
                    globals_=globals_,
                    y0_phys_list=y0_buf,
                    g_phys_list=g_buf,
                    dt_phys_list=dt_buf,
                    y_true_phys_list=ytrue_buf,
                    shard_ids=shard_id_buf,
                    traj_ids=traj_id_buf,
                    time_ids=time_id_buf,
                    t_abs_list=tabs_buf,
                    acc=acc,
                    pair_mae_res=pair_mae_res,
                    parity_res=parity_res,
                    mass_err_res=mass_err_res,
                    worst=worst,
                    parity_cols=parity_cols,
                )

    # final flush
    _run_forward_batch(
        mod=mod,
        device=device,
        dtype=dtype,
        norm=norm,
        species=species,
        globals_=globals_,
        y0_phys_list=y0_buf,
        g_phys_list=g_buf,
        dt_phys_list=dt_buf,
        y_true_phys_list=ytrue_buf,
        shard_ids=shard_id_buf,
        traj_ids=traj_id_buf,
        time_ids=time_id_buf,
        t_abs_list=tabs_buf,
        acc=acc,
        pair_mae_res=pair_mae_res,
        parity_res=parity_res,
        mass_err_res=mass_err_res,
        worst=worst,
        parity_cols=parity_cols,
    )

    n = max(1, int(acc.n_pairs))

    mae_log = acc.sum_abs_log / n
    rmse_log = np.sqrt(acc.sum_sq_log / n)

    sst = np.maximum(acc.sum_true2_log - (acc.sum_true_log * acc.sum_true_log) / n, 0.0)
    r2_log = np.where(sst > 0.0, 1.0 - (acc.sum_sq_log / sst), np.nan)

    mae_rel = acc.sum_abs_rel / n

    # per-species CSV
    _safe_write_csv(
        out_dir / "export_acc_per_species.csv",
        header=["species", "mae_log10", "rmse_log10", "r2_log10", "max_abs_log10", "mae_rel", "max_rel"],
        rows=[
            [
                species[i],
                float(mae_log[i]),
                float(rmse_log[i]),
                (float(r2_log[i]) if np.isfinite(r2_log[i]) else ""),
                float(acc.max_abs_log[i]),
                float(mae_rel[i]),
                float(acc.max_abs_rel[i]),
            ]
            for i in range(len(species))
        ],
    )

    # dt bins CSV
    centers = np.sqrt(acc.dt_edges[:-1] * acc.dt_edges[1:])
    dt_rows = []
    for i in range(acc.dt_count.size):
        c = int(acc.dt_count[i])
        mae_i = (acc.dt_sum_pair_mae_log[i] / c) if c > 0 else ""
        neg_i = (acc.dt_neg_count[i] / c) if c > 0 else ""
        dt_rows.append([float(acc.dt_edges[i]), float(acc.dt_edges[i + 1]), float(centers[i]), c,
                        (float(mae_i) if mae_i != "" else ""),
                        (float(neg_i) if neg_i != "" else "")])
    _safe_write_csv(
        out_dir / "export_acc_dt_bins.csv",
        header=["dt_lo", "dt_hi", "dt_center", "count", "pair_mae_log10", "neg_pair_fraction"],
        rows=dt_rows,
    )

    # worst cases CSV
    worst_sorted = sorted(worst, key=lambda x: x[0], reverse=True)
    _safe_write_csv(
        out_dir / "export_acc_worst_cases.csv",
        header=["pair_mae_log10", "shard_idx", "traj_idx", "t_abs_s", "time_index"],
        rows=[[float(s), int(sh), int(tr), float(tabs), int(ti)] for (s, sh, tr, tabs, ti) in worst_sorted],
    )

    # plots
    if MAKE_PLOTS:
        _plot_dt_curve(out_dir, acc.dt_edges, acc.dt_count, acc.dt_sum_pair_mae_log)
        _plot_pair_hist(out_dir, pair_mae_res)
        _plot_species_bar(out_dir, species, mae_log)
        _plot_parity(out_dir, parity_res)

        # mass related plots
        _plot_mass_hists(out_dir, acc.sum_true_mass_res, acc.sum_pred_mass_res)
        _plot_mass_scatter(out_dir, acc.sum_true_mass_res, acc.sum_pred_mass_res)
        _plot_sum_error_scatter(out_dir, acc.sum_true_mass_res, acc.sum_pred_mass_res)
        _plot_neg_rate_by_dt(out_dir, acc.dt_edges, acc.dt_count, acc.dt_neg_count)

        # mass error histogram from mass_err_res
        x = mass_err_res.as_array_float()
        if x.size:
            plt.figure()
            plt.hist(x, bins=80)
            plt.xlabel("sum(pred) - sum(true) (phys)")
            plt.ylabel("Count")
            plt.title("Mass / sum-species error distribution (reservoir sample)")
            plt.tight_layout()
            plt.savefig(out_dir / "export_plot_mass_error_hist.png", dpi=200)
            plt.close()

    summary = {
        "repo": str(REPO),
        "model_dir": str(MODEL_DIR),
        "export": str(EXPORT_FILENAME),
        "split": str(SPLIT),
        "data_dir": str(data_dir),
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "coverage": {
            "max_shards": int(MAX_SHARDS),
            "trajs_per_shard": int(TRAJS_PER_SHARD),
            "max_timepoints_per_traj": int(MAX_TIMEPOINTS),
            "time_window_s": [float(TMIN), float(TMAX)],
            "infer_batch": int(INFER_BATCH),
        },
        "counts": {
            "pairs_evaluated": int(acc.n_pairs),
            "trajs_sampled_total": int(total_trajs_seen),
        },
        "sanity": {
            "pred_nan_pairs": int(acc.n_pred_nan),
            "pred_inf_pairs": int(acc.n_pred_inf),
            "pred_neg_any_pairs": int(acc.n_pred_neg_any),
            "pred_gt1_any_pairs": int(acc.n_pred_gt1_any),
        },
        "aggregate_metrics": {
            "mean_species_mae_log10": float(np.mean(mae_log)),
            "mean_species_rmse_log10": float(np.mean(rmse_log)),
            "mean_species_mae_rel": float(np.mean(mae_rel)),
        },
        "reservoir_seen": {
            "pair_mae_log10": int(pair_mae_res.count()),
            "parity_points": int(parity_res.count()),
            "mass_err": int(mass_err_res.count()),
        },
        "worst_cases_kept": int(len(worst_sorted)),
    }

    # IMPORTANT FIX: dump_json(path, obj)
    dump_json(out_dir / "export_acc_summary.json", summary)
    return summary


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    os.chdir(REPO)
    seed_everything(int(SEED))
    random.seed(int(SEED))
    np.random.seed(int(SEED))

    export_path = (MODEL_DIR / EXPORT_FILENAME).expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    cfg_path = _find_cfg_path(MODEL_DIR)
    cfg = load_json_config(cfg_path)

    data_dir = _resolve_data_dir(cfg)
    norm_path = data_dir / "normalization.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")

    manifest = load_json_config(norm_path)
    meta = manifest.get("meta", {})
    species = list(meta.get("species_variables", []))
    globals_ = list(meta.get("global_variables", []))
    if not species:
        raise RuntimeError("No species_variables in normalization.json meta.")

    device = _choose_device(export_path, DEVICE_STR)
    dtype = _resolve_dtype(device, DTYPE_STR)

    mod = _load_exported_module(export_path)
    _try_set_eval(mod)

    out_dir = _ensure_dir(MODEL_DIR / "plots")

    print("=" * 90)
    print("FlowMap EXPORT Accuracy Benchmark Suite (no argparse)")
    print("=" * 90)
    print(f"REPO:      {REPO}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"EXPORT:    {export_path.name}")
    print(f"SPLIT:     {SPLIT}")
    print(f"DEVICE:    {device.type}")
    print(f"DTYPE:     {str(dtype).replace('torch.', '')}")
    print(f"DATA_DIR:  {data_dir}")
    print(f"S={len(species)}  G={len(globals_)}")
    print(f"Time window: [{TMIN:.1e}, {TMAX:.1e}] s | max points/trajectory: {MAX_TIMEPOINTS}")
    print(f"Coverage: MAX_SHARDS={MAX_SHARDS}  TRAJS_PER_SHARD={TRAJS_PER_SHARD}")
    print(f"Outputs:  {out_dir}")
    print("-" * 90)

    summary = evaluate_export(
        mod=mod,
        device=device,
        dtype=dtype,
        data_dir=data_dir,
        manifest=manifest,
        out_dir=out_dir,
    )

    print("-" * 90)
    print("DONE.")
    print(f"Pairs evaluated: {summary['counts']['pairs_evaluated']}")
    print(f"Neg pairs: {summary['sanity']['pred_neg_any_pairs']} | NaN pairs: {summary['sanity']['pred_nan_pairs']} | Inf pairs: {summary['sanity']['pred_inf_pairs']}")
    print(f"Mean species MAE(log10): {summary['aggregate_metrics']['mean_species_mae_log10']:.4g}")
    print("=" * 90)


if __name__ == "__main__":
    main()
