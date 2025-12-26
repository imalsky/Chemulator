#!/usr/bin/env python3
"""
testing_big.py — FlowMap EXPORT Accuracy Benchmark Suite (no argparse)

Key requirements implemented:
- NO runtime/time benchmarking (no perf_counter, no throughput, no synthetic perf).
- Loads data the same way as your working script:
    (data_dir / SPLIT / shard_*.npz) with keys: y_mat, globals, t_vec
- Query times capped:
    - max 100 time points per trajectory
    - only times within [1e-3, 1e8] seconds
- Only accuracy / emulator-quality benchmarks:
    - log10-space MAE/RMSE/R2 per species
    - relative error (linear space) per species
    - error vs dt bins
    - parity (log10 true vs log10 pred) scatter
    - error-vs-time curve (binned)
    - mass conservation diagnostics (sum of species)
    - positivity/finite checks (NaN/Inf/negative rates)
    - top worst cases (largest per-pair MAE log10)

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
"""

from __future__ import annotations

import os
import sys
import json
import csv
import math
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

MODEL_DIR = REPO / "models" / "full_big"
EXPORT_FILENAME = "export_k1_cpu.pt2"

SPLIT = "test"
SEED = 42

# Fast defaults (edit if you want more coverage)
MAX_SHARDS: int = 1                 # number of shard_*.npz files to evaluate
TRAJS_PER_SHARD: int = 12           # number of trajectories sampled per shard
SHUFFLE_TRAJS: bool = True          # random subset per shard (seeded)

# Query sampling (HARD CAPS required by you)
MAX_TIMEPOINTS: int = 100           # <= 100
TMIN: float = 1e-3                  # only evaluate times >= 1e-3 s
TMAX: float = 1e8                   # only evaluate times <= 1e8 s
QUERY_MODE: str = "lin"             # "lin" or "log" selection over available indices

# Inference batching (keeps calls fewer/faster)
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
    # match your working behavior: os.chdir(REPO) then resolve
    return (REPO / p).resolve()

def _list_shards(data_dir: Path, split: str) -> List[Path]:
    shards = sorted((data_dir / split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found in {data_dir / split}")
    return shards

def _load_shard_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path) as d:
            y = d["y_mat"].astype(np.float32)                 # [N,T,S]
            g = d["globals"].astype(np.float32)               # [N,G]
            t = d["t_vec"].astype(np.float32)                 # [T] or [N,T]
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
        # pick log-spaced over the valid index list (still within [TMIN,TMAX])
        xs = np.geomspace(1, valid.size, qn)
        pick = np.unique(np.clip(np.rint(xs).astype(np.int64) - 1, 0, valid.size - 1))
        idx = valid[pick]
    else:
        # linear over valid list
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
    sum_abs_log: np.ndarray
    sum_sq_log: np.ndarray
    max_abs_log: np.ndarray
    sum_true_log: np.ndarray
    sum_true2_log: np.ndarray

    sum_abs_rel: np.ndarray   # mean absolute relative error (linear)
    max_abs_rel: np.ndarray

    # sanity counts
    n_pred_nan: int
    n_pred_inf: int
    n_pred_neg: int

    # dt bins
    dt_edges: np.ndarray
    dt_count: np.ndarray
    dt_sum_pair_mae_log: np.ndarray
    dt_sum_neg_frac: np.ndarray  # accumulate neg-rate per bin (sum of neg_fraction for each batch)

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
        n_pred_neg=0,
        dt_edges=edges,
        dt_count=np.zeros((dt_bins,), dtype=np.int64),
        dt_sum_pair_mae_log=np.zeros((dt_bins,), dtype=np.float64),
        dt_sum_neg_frac=np.zeros((dt_bins,), dtype=np.float64),
    )

def _update_bins(acc: Accum, dt_phys: np.ndarray, pair_mae_log: np.ndarray, neg_mask: np.ndarray) -> None:
    idx = _bin_indices_log(dt_phys, acc.dt_edges)
    for b in range(acc.dt_count.size):
        m = (idx == b)
        if not np.any(m):
            continue
        acc.dt_count[b] += int(np.sum(m))
        acc.dt_sum_pair_mae_log[b] += float(np.sum(pair_mae_log[m]))
        acc.dt_sum_neg_frac[b] += float(np.mean(neg_mask[m])) * float(np.sum(m))

def _safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 0.0, None) + float(EPS_PHYS))


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
    """
    parity_res stores tuples (true_log, pred_log).
    """
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

def _plot_mass_error(out_dir: Path, mass_res: Reservoir) -> None:
    x = mass_res.as_array_float()
    if x.size == 0:
        return
    plt.figure()
    plt.hist(x, bins=80)
    plt.xlabel("sum(pred) - sum(true)  (phys)")
    plt.ylabel("Count")
    plt.title("Mass / sum-species error distribution (reservoir sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "export_plot_mass_error_hist.png", dpi=200)
    plt.close()

def _plot_neg_rate_by_dt(out_dir: Path, edges: np.ndarray, counts: np.ndarray, sum_neg: np.ndarray) -> None:
    centers = np.sqrt(edges[:-1] * edges[1:])
    neg_rate = np.full_like(centers, np.nan, dtype=np.float64)
    m = counts > 0
    # sum_neg stores (neg_fraction * count) accumulated
    neg_rate[m] = sum_neg[m] / counts[m]

    plt.figure()
    plt.xscale("log")
    plt.plot(centers, neg_rate)
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
    mass_res: Reservoir,
    worst: List[Tuple[float, int, int, float, int]],
) -> None:
    """
    Inputs are lists of chunks; we normalize exactly like your working script and run export K=1.
    We update accumulators and reservoirs. Also update "worst cases".
    """
    if not y0_phys_list:
        return

    # Concatenate
    y0_phys = np.concatenate(y0_phys_list, axis=0).astype(np.float32)     # [B,S]
    g_phys = np.concatenate(g_phys_list, axis=0).astype(np.float32)       # [B,G]
    dt_phys = np.concatenate(dt_phys_list, axis=0).astype(np.float32)     # [B]
    y_true_phys = np.concatenate(y_true_phys_list, axis=0).astype(np.float32)  # [B,S]
    t_abs = np.concatenate(t_abs_list, axis=0).astype(np.float32)         # [B]

    B = int(y0_phys.shape[0])
    S = int(y0_phys.shape[1])
    G = int(g_phys.shape[1]) if g_phys.ndim == 2 else 0

    # Normalize inputs exactly like your working script
    y0_norm = norm.normalize(torch.from_numpy(y0_phys), species).float()   # [B,S]
    if globals_:
        g_norm = norm.normalize(torch.from_numpy(g_phys), globals_).float()  # [B,G]
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

    # Sanity
    pred_nan = np.isnan(y_pred_phys).any(axis=1)
    pred_inf = np.isinf(y_pred_phys).any(axis=1)
    pred_neg_any = (y_pred_phys < 0.0).any(axis=1)
    acc.n_pred_nan += int(np.sum(pred_nan))
    acc.n_pred_inf += int(np.sum(pred_inf))
    acc.n_pred_neg += int(np.sum(pred_neg_any))

    # Mass/sum-species diagnostic
    mass_true = y_true_phys64.sum(axis=1)
    mass_pred = y_pred_phys.sum(axis=1)
    mass_err = mass_pred - mass_true
    mass_res.add_many(mass_err.tolist())

    # Errors
    true_log = _safe_log10(y_true_phys64)
    pred_log = _safe_log10(y_pred_phys)

    err_log = pred_log - true_log
    abs_err_log = np.abs(err_log)
    sq_err_log = err_log * err_log

    # relative error in linear space
    denom = np.abs(y_true_phys64) + float(EPS_PHYS)
    rel = np.abs(y_pred_phys - y_true_phys64) / denom
    abs_rel = rel

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

    # parity sample across species: reservoir over flattened points
    # sample a few species per pair to avoid heavy list creation
    # (still reservoir-limited; this is fast)
    if RESERVOIR_PARITY_POINTS > 0:
        # take up to 4 random species columns per pair for parity reservoir
        rng = np.random.default_rng(SEED + 123)
        cols = rng.choice(S, size=min(4, S), replace=False)
        pts = [(float(true_log[i, j]), float(pred_log[i, j])) for i in range(B) for j in cols]
        parity_res.add_many(pts)

    # dt bins update (use absolute time since t0, but here dt_phys is already that)
    _update_bins(acc, dt_phys.astype(np.float64), pair_mae_log.astype(np.float64), pred_neg_any.astype(np.bool_))

    # worst cases tracking: keep tuples (score, shard_id, traj_id, t_abs, time_index)
    # score = pair_mae_log
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
    mass_res = Reservoir(int(RESERVOIR_MASS_ERR), seed=SEED + 3)

    worst: List[Tuple[float, int, int, float, int]] = []

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
            # if manifest has no globals, allow G=0
            if len(globals_) == 0 and g_mat.shape[1] == 0:
                pass
            else:
                raise RuntimeError(f"{shard_path.name}: globals G={g_mat.shape[1]} but manifest globals G={len(globals_)}")

        idxs = np.arange(N, dtype=np.int64)
        if SHUFFLE_TRAJS:
            rng = np.random.default_rng(SEED + 999 + shard_idx)
            rng.shuffle(idxs)
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
            y0 = y_mat[int(traj_i), 0, :].astype(np.float32)          # [S]
            g = g_mat[int(traj_i), :].astype(np.float32)              # [G]

            # selected points
            t_sel = t_phys[q_idx].astype(np.float32)                  # [K]
            dt_sec = np.maximum(t_sel - t_phys[0], 0.0).astype(np.float32)  # [K]
            y_true = y_mat[int(traj_i), q_idx, :].astype(np.float32)   # [K,S]

            K = int(q_idx.size)

            # Push into buffers as chunk arrays
            y0_buf.append(np.repeat(y0[None, :], K, axis=0))           # [K,S] repeated anchor
            g_buf.append(np.repeat(g[None, :], K, axis=0))             # [K,G]
            dt_buf.append(dt_sec.reshape(-1))                          # [K]
            ytrue_buf.append(y_true)                                   # [K,S]
            tabs_buf.append(t_sel.reshape(-1))                         # [K]
            shard_id_buf.extend([int(shard_idx)] * K)
            traj_id_buf.extend([int(traj_i)] * K)
            time_id_buf.extend([int(x) for x in q_idx.tolist()])

            # flush
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
                    mass_res=mass_res,
                    worst=worst,
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
        mass_res=mass_res,
        worst=worst,
    )

    n = max(1, int(acc.n_pairs))

    mae_log = acc.sum_abs_log / n
    rmse_log = np.sqrt(acc.sum_sq_log / n)

    sst = np.maximum(acc.sum_true2_log - (acc.sum_true_log * acc.sum_true_log) / n, 0.0)
    r2_log = np.where(sst > 0.0, 1.0 - (acc.sum_sq_log / sst), np.nan)

    mae_rel = acc.sum_abs_rel / n

    # Write per-species CSV
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
        neg_i = (acc.dt_sum_neg_frac[i] / c) if c > 0 else ""
        dt_rows.append([float(acc.dt_edges[i]), float(acc.dt_edges[i + 1]), float(centers[i]), c,
                        (float(mae_i) if mae_i != "" else ""),
                        (float(neg_i) if neg_i != "" else "")])
    _safe_write_csv(
        out_dir / "export_acc_dt_bins.csv",
        header=["dt_lo", "dt_hi", "dt_center", "count", "pair_mae_log10", "neg_pair_fraction"],
        rows=dt_rows,
    )

    # worst cases CSV
    # columns: score, shard, traj, t_abs, time_index
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
        _plot_mass_error(out_dir, mass_res)
        _plot_neg_rate_by_dt(out_dir, acc.dt_edges, acc.dt_count, acc.dt_sum_neg_frac)

    summary = {
        "repo": str(REPO),
        "model_dir": str(MODEL_DIR),
        "export": str(EXPORT_FILENAME),
        "split": str(SPLIT),
        "data_dir": str(data_dir),
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "max_shards": int(MAX_SHARDS),
        "trajs_per_shard": int(TRAJS_PER_SHARD),
        "max_timepoints_per_traj": int(MAX_TIMEPOINTS),
        "time_window_s": [float(TMIN), float(TMAX)],
        "n_pairs_evaluated": int(acc.n_pairs),
        "n_trajs_sampled_total": int(total_trajs_seen),
        "pred_nan_pairs": int(acc.n_pred_nan),
        "pred_inf_pairs": int(acc.n_pred_inf),
        "pred_neg_pairs": int(acc.n_pred_neg),
        "global_mean_species_mae_log10": float(np.mean(mae_log)),
        "global_mean_species_rmse_log10": float(np.mean(rmse_log)),
        "global_mean_species_mae_rel": float(np.mean(mae_rel)),
        "pair_mae_log10_reservoir_seen": int(pair_mae_res.count()),
        "parity_points_reservoir_seen": int(parity_res.count()),
        "mass_err_reservoir_seen": int(mass_res.count()),
        "worst_cases_kept": int(len(worst_sorted)),
    }

    dump_json(summary, out_dir / "export_acc_summary.json")
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
    print(f"Pairs evaluated: {summary['n_pairs_evaluated']}")
    print(f"Neg pairs: {summary['pred_neg_pairs']} | NaN pairs: {summary['pred_nan_pairs']} | Inf pairs: {summary['pred_inf_pairs']}")
    print(f"Mean species MAE(log10): {summary['global_mean_species_mae_log10']:.4g}")
    print("=" * 90)


if __name__ == "__main__":
    main()
