#!/usr/bin/env python3
"""
Preprocessor: HDF5 (grouped trajectories) → NPZ shards + normalization.json (+ indices/summaries)

Key changes in this version:
- STRICT intra-shard time check: all trajectories buffered into a shard must have
  IDENTICAL time grids; otherwise we raise with diagnostics.
- Store `t_vec` as 1-D [T] in each shard (instead of [N, T]). This avoids shape
  mismatches in downstream readers without relaxing any time-grid guarantees.
- Time-grid auditor accepts either [T] (new) or [N, T] (legacy) shards.

Other features preserved:
- Per-file tqdm (outer) and per-group tqdm (inner) progress bars.
- Deterministic hashing-based split assignment and optional use_fraction filter.
- Training-only normalization stats + centralized Δt spec computed from the
  single canonical grid discovered during scan.
- Detailed reports: normalization.json, preprocess_report.json, shard_index.json,
  preprocessing_summary.json.
"""

from __future__ import annotations

# ============================== GLOBAL CONSTANTS ==============================

# TQDM tuning
TQDM_MININTERVAL: float = 0.25
TQDM_SMOOTHING: float = 0.1
TQDM_LEAVE_OUTER: bool = True
TQDM_LEAVE_INNER: bool = False

# Parallel scan timeouts
PAR_SCAN_TIMEOUT_PER_FILE_S: int = 300     # 5 min per file
PAR_SCAN_OVERHEAD_S: int = 120             # +2 min overhead
PAR_FUTURE_RESULT_TIMEOUT_S: int = 0       # 0 → don't time out per-result after as_completed()

# Time grid validation
# REQUIRE strictly increasing grids to guarantee Δt>0 for k>=1 used by dt spec.
ALLOW_EQUAL_TIMEPOINTS: bool = False
TIME_DECREASE_TOL: float = 0.0

# HDF5 read defaults
DEFAULT_HDF5_CHUNK_FALLBACK: int = 0       # 0 → use dataset native chunk or full length

# NPZ shard naming
SHARD_FILENAME_FMT: str = "shard_{split}_{filetag}_{idx:05d}.npz"

# Misc
BYTES_PER_GIB: float = 1024.0 ** 3

# =============================================================================

import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=False)
except Exception:
    pass

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


# ------------------------------- Utilities -----------------------------------

def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_config_value(d: Dict[str, Any], path: Sequence[str], *, required: bool = False, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            if required:
                raise KeyError(f"Missing required config key: {'.'.join(path)}")
            return default
        cur = cur[k]
    return cur


def _dtype_from_config(cfg: Dict[str, Any]) -> np.dtype:
    sys = cfg.get("system", {})
    s = str(sys.get("io_dtype", sys.get("dtype", "float32"))).lower()
    if s not in {"float32", "float64"}:
        raise ValueError(f"Unsupported dtype '{s}'. Allowed: 'float32', 'float64'.")
    return np.float32 if s == "float32" else np.float64


def _hash01(s: str, seed: int) -> float:
    b = f"{seed}:{s}".encode("utf-8")
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], "big", signed=False) / float(1 << 64)


# ------------------------------- Welford stats -------------------------------

class _Welford1D:
    __slots__ = ("n", "mean", "M2", "min", "max")

    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.min: float = math.inf
        self.max: float = -math.inf

    def update_array(self, arr: np.ndarray) -> None:
        v = np.asarray(arr, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return
        n_b = int(v.size)
        mean_b = float(v.mean())
        var_b = float(v.var(ddof=0))
        M2_b = var_b * n_b

        n_a = self.n
        mean_a = self.mean
        M2_a = self.M2

        n = n_a + n_b
        if n == 0:
            return
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        M2 = M2_a + M2_b + delta * delta * (n_a * n_b / n)

        self.n = n
        self.mean = mean
        self.M2 = M2
        self.min = float(min(self.min, float(v.min())))
        self.max = float(max(self.max, float(v.max())))

    def finalize(self, min_std: float) -> Tuple[float, float, float, float]:
        if self.n <= 0:
            raise RuntimeError("Insufficient data for Welford finalize.")
        var = max(0.0, self.M2 / self.n)
        std = math.sqrt(var)
        return float(self.mean), float(max(std, float(min_std))), float(self.min), float(self.max)


class _RunningStats:
    """Per-key composite stats: raw and/or log domain depending on method flags."""

    def __init__(self, need_mean_std: bool, need_min_max: bool, need_log: bool, epsilon: float):
        self.need_mean_std = bool(need_mean_std)
        self.need_min_max = bool(need_min_max)
        self.need_log = bool(need_log)
        self.epsilon = float(epsilon)
        self.raw = _Welford1D() if (self.need_mean_std or self.need_min_max) else None
        self.log = _Welford1D() if self.need_log else None

    def update(self, arr: np.ndarray) -> None:
        a = np.asarray(arr)
        if a.size == 0:
            return
        if self.raw is not None:
            self.raw.update_array(a)
        if self.log is not None:
            a_log = np.log10(np.clip(a, self.epsilon, None))
            self.log.update_array(a_log)

    def to_manifest(self, min_std: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.raw is not None:
            mean, std, mn, mx = self.raw.finalize(min_std)
            if self.need_mean_std:
                out["mean"] = float(mean)
                out["std"] = float(std)
            if self.need_min_max:
                out["min"] = float(mn)
                out["max"] = float(mx)
        if self.log is not None:
            mean_l, std_l, log_min, log_max = self.log.finalize(min_std)
            out["log_mean"] = float(mean_l)
            out["log_std"] = float(std_l)
            out["log_min"] = float(log_min)
            out["log_max"] = float(log_max)
        return out


def _need_flags_for_method(method: str) -> Tuple[bool, bool, bool]:
    m = str(method)
    if m == "standard":
        return True, False, False
    if m == "min-max":
        return False, True, False
    if m == "log-standard":
        return True, False, True
    if m == "log-min-max":
        return False, True, True
    raise ValueError(f"Unknown normalization method: '{method}'.")


# ----------------------- Worker for process-based scan -----------------------

def _choose_chunk_len_static(dset, T: int, cfg_chunk_len: int) -> int:
    if cfg_chunk_len and cfg_chunk_len > 0:
        return max(1, min(int(cfg_chunk_len), T))
    try:
        if hasattr(dset, "chunks") and dset.chunks and len(dset.chunks) >= 1 and dset.chunks[0]:
            return max(1, min(int(dset.chunks[0]), T))
    except Exception:
        pass
    return T


def _scan_one_file_worker(
    path_str: str,
    species_vars: List[str],
    global_vars: List[str],
    time_key: str,
    min_value_threshold: float,
    epsilon: float,             # kept for symmetry; not used here
    cfg_chunk_len: int,
) -> Tuple[Dict[str, Any], List[str], Optional[np.ndarray]]:
    """
    Process-pool worker: scans a single HDF5 file.
    Returns (file_report, valid_groups, time_candidate).
    """
    if h5py is None:
        raise RuntimeError("h5py is required but not available in worker.")

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"HDF5 file not found: {p}")

    n_total = n_valid = n_nan = n_below = 0
    valid_groups: List[str] = []
    time_candidate: Optional[np.ndarray] = None

    with h5py.File(p, "r") as f:
        group_names = list(f.keys())
        for group_name in group_names:
            grp = f[group_name]
            n_total += 1

            # ---- time
            if time_key not in grp:
                n_nan += 1
                continue
            t = np.array(grp[time_key], dtype=np.float64, copy=False).reshape(-1)
            if t.size < 2 or not np.all(np.isfinite(t)):
                n_nan += 1
                continue
            diffs = np.diff(t)
            if ALLOW_EQUAL_TIMEPOINTS:
                if np.any(diffs < -abs(TIME_DECREASE_TOL)):
                    n_nan += 1
                    continue
            else:
                if not np.all(diffs > 0.0):
                    n_nan += 1
                    continue

            # within-file shared time
            if time_candidate is None:
                time_candidate = t.copy()
            else:
                if not np.array_equal(t, time_candidate):
                    raise ValueError(f"{p}:{group_name}: time grid differs within file; expected shared grid.")

            # ---- globals
            try:
                g_vec = np.array([float(grp.attrs[k]) for k in global_vars], dtype=np.float64)
                if not np.all(np.isfinite(g_vec)):
                    n_nan += 1
                    continue
            except Exception:
                n_nan += 1
                continue

            # ---- species first pass (validate, chunked)
            T = int(t.shape[0])
            had_nan_flag = False
            below_flag = False
            for sname in species_vars:
                dset = grp[sname]
                if dset.shape[0] != T:
                    had_nan_flag = True
                    break
                chunk_len = _choose_chunk_len_static(dset, T, cfg_chunk_len)
                for s0 in range(0, T, chunk_len):
                    s1 = min(T, s0 + chunk_len)
                    arr = np.array(dset[s0:s1], dtype=np.float64, copy=False).reshape(-1)
                    if not np.isfinite(arr).all():
                        had_nan_flag = True
                        break
                    if (arr < min_value_threshold).any():
                        below_flag = True
                        break
                if had_nan_flag or below_flag:
                    break

            if had_nan_flag or below_flag:
                if had_nan_flag:
                    n_nan += 1
                else:
                    n_below += 1
                continue

            n_valid += 1
            valid_groups.append(group_name)

    file_report = {
        "path": str(p),
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "n_dropped": int(n_total - n_valid),
        "n_nan": int(n_nan),
        "n_below_threshold": int(n_below),
    }
    return file_report, valid_groups, time_candidate


# ------------------------------- Preprocessor --------------------------------

class DataPreprocessor:
    """Convert grouped HDF5 files to NPZ shards and write normalization.json."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.cfg = config
        self.log = logger or logging.getLogger("preprocessor")

        # Paths
        self.raw_files: List[str] = list(_load_config_value(self.cfg, ["paths", "raw_data_files"], required=True))
        if len(self.raw_files) == 0:
            raise ValueError("paths.raw_data_files is empty.")
        self.processed_dir = Path(_load_config_value(self.cfg, ["paths", "processed_data_dir"], required=True))

        # Check output directory state
        if self.processed_dir.exists():
            if any(self.processed_dir.iterdir()):
                raise FileExistsError(f"Output directory already exists and is not empty: {self.processed_dir}")
            else:
                self.log.warning("Output directory exists but is empty: %s", self.processed_dir)
        else:
            _ensure_dir(self.processed_dir)

        # IO/storage dtype
        self.storage_dtype_np: np.dtype = _dtype_from_config(self.cfg)

        # Data schema
        dcfg = self.cfg.get("data", {})
        self.species_vars: List[str] = list(_load_config_value(dcfg, ["species_variables"], required=True))
        self.target_species_vars: List[str] = list(dcfg.get("target_species_variables", self.species_vars))
        self.global_vars: List[str] = list(_load_config_value(dcfg, ["global_variables"], required=True))
        self.time_key: str = str(_load_config_value(dcfg, ["time_variable"], required=True))

        # Normalization settings
        ncfg = self.cfg.get("normalization", {})
        if "default_method" not in ncfg or "methods" not in ncfg:
            raise KeyError("normalization.default_method and normalization.methods are required.")
        self.default_method: str = str(ncfg["default_method"])
        self.methods: Dict[str, str] = dict(ncfg["methods"])
        if "epsilon" not in ncfg or "min_std" not in ncfg or "clamp_value" not in ncfg:
            raise KeyError("normalization.epsilon, normalization.min_std, normalization.clamp_value are required.")
        self.epsilon: float = float(ncfg["epsilon"])
        self.min_std: float = float(ncfg["min_std"])
        self.clamp_value: float = float(ncfg["clamp_value"])

        # Sharding + drop policy
        pcfg = self.cfg.get("preprocessing", {})
        self.npz_compressed: bool = bool(_load_config_value(pcfg, ["npz_compressed"], required=True))
        self.traj_per_shard: int = int(_load_config_value(pcfg, ["trajectories_per_shard"], required=True))
        self.hdf5_chunk_size: int = int(_load_config_value(pcfg, ["hdf5_chunk_size"], required=False, default=DEFAULT_HDF5_CHUNK_FALLBACK))
        self.min_value_threshold: float = float(_load_config_value(pcfg, ["min_value_threshold"], required=True))

        # Worker configuration for PASS 1 (scan)
        requested_workers = int(_load_config_value(pcfg, ["num_workers"], required=False, default=0))
        if len(self.raw_files) <= 2:
            self.num_workers = 0
            if requested_workers > 0:
                self.log.info("Using serial mode for %d files (parallel overhead not worth it)", len(self.raw_files))
        else:
            if requested_workers > 0:
                self.num_workers = min(requested_workers, len(self.raw_files))
            else:
                cpu_count = os.cpu_count() or 1
                self.num_workers = min(4, cpu_count, len(self.raw_files))

        # Splits & selection
        scfg = self.cfg.get("training", {})
        self.val_fraction: float = float(_load_config_value(scfg, ["val_fraction"], required=True))
        self.test_fraction: float = float(_load_config_value(scfg, ["test_fraction"], required=True))
        self.use_fraction: float = float(scfg.get("use_fraction", 1.0))
        if not (0.0 <= self.use_fraction <= 1.0):
            raise ValueError("training.use_fraction must be in [0,1].")
        if not (0.0 <= self.val_fraction <= 1.0 and 0.0 <= self.test_fraction <= 1.0 and self.val_fraction + self.test_fraction < 1.0):
            raise ValueError("Require 0 ≤ val_fraction,test_fraction and val+test < 1.")

        # Δt step constraints (for dt spec)
        tcfg = self.cfg.get("training", {})
        self.min_steps: int = int(_load_config_value(tcfg, ["min_steps"], required=True))
        self.max_steps: int = int(_load_config_value(tcfg, ["max_steps"], required=True))
        if self.min_steps < 1:
            raise ValueError("training.min_steps must be ≥ 1.")
        if self.max_steps < self.min_steps:
            raise ValueError("training.max_steps must be ≥ training.min_steps.")

        # Accumulators and state
        self._drop_report: Dict[str, Any] = {
            "files": [],
            "overall": {"n_total": 0, "n_valid": 0, "n_dropped": 0, "n_nan": 0, "n_below_threshold": 0}
        }
        self._valid_group_names: Dict[str, List[str]] = {}
        self._canonical_time: Optional[np.ndarray] = None

    # ------------------------------ Entry point ------------------------------

    def run(self) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required to run the preprocessor but is not installed.")

        t_start = time.time()
        self.log.info("Starting preprocessing over %d HDF5 file(s)…", len(self.raw_files))
        self.log.info("Parallel scan workers (processes): %d", int(self.num_workers))

        # PASS 1: scan & validate
        t1 = time.time()
        self._scan_files_collect_valid_groups()
        t2 = time.time()
        if self._canonical_time is None:
            raise RuntimeError("No valid trajectories found (no canonical time grid established).")

        # Estimate memory footprint
        try:
            N = int(self._drop_report["overall"]["n_valid"])
            T = int(self._canonical_time.shape[0])
            S = int(len(self.species_vars))
            itemsize = int(np.dtype(self.storage_dtype_np).itemsize)
            estimated_mem = int(self._canonical_time.nbytes) + N * T * S * itemsize
            self.log.info("Estimated memory for sharding phase: %s", _fmt_bytes(estimated_mem))
        except Exception:
            pass

        # Centralized Δt spec from canonical grid
        dt_spec = self._compute_dt_spec_from_grid(self._canonical_time)
        t3 = time.time()

        # PASS 2: write shards (+ collect TRAINING stats for normalization)
        train_stats = self._write_shards_and_collect_training_stats()
        t4 = time.time()

        # Finalize manifest using TRAINING stats + centralized dt
        manifest = self._finalize_manifest(train_stats, dt_spec)
        manifest_path = self.processed_dir / "normalization.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self.log.info("Wrote normalization manifest: %s", manifest_path)
        t5 = time.time()

        # Write top-level reports (modern + legacy-style)
        self._write_reports(t_start, t1, t2, t3, t4, t5)

    # ----------------------------- Pass 1: scan ------------------------------

    def _scan_files_collect_valid_groups(self) -> None:
        """Parallel (or serial) scan of HDF5 files -> valid group names per file + canonical time."""
        per_file_time_candidates: List[np.ndarray] = []
        effective_workers = self.num_workers

        if effective_workers > 1:
            try:
                total_timeout = len(self.raw_files) * PAR_SCAN_TIMEOUT_PER_FILE_S + PAR_SCAN_OVERHEAD_S
                with ProcessPoolExecutor(max_workers=int(effective_workers)) as ex:
                    pbar = tqdm(total=len(self.raw_files), desc="Scanning HDF5 files (parallel)", unit="file",
                                smoothing=TQDM_SMOOTHING, mininterval=TQDM_MININTERVAL, leave=TQDM_LEAVE_OUTER)
                    futures = {}
                    for path in self.raw_files:
                        fut = ex.submit(
                            _scan_one_file_worker,
                            path,
                            self.species_vars,
                            self.global_vars,
                            self.time_key,
                            self.min_value_threshold,
                            self.epsilon,
                            int(self.hdf5_chunk_size),
                        )
                        futures[fut] = path

                    completed = 0
                    for fut in as_completed(futures, timeout=total_timeout):
                        file_path = futures[fut]
                        try:
                            file_report, valid_groups, time_candidate = fut.result(timeout=PAR_FUTURE_RESULT_TIMEOUT_S or None)
                            self._drop_report["files"].append(file_report)
                            for k in ("n_total","n_valid","n_dropped","n_nan","n_below_threshold"):
                                self._drop_report["overall"][k] += file_report[k]
                            self._valid_group_names[str(file_path)] = valid_groups
                            if time_candidate is not None:
                                per_file_time_candidates.append(time_candidate)
                            completed += 1
                            pbar.update(1)
                        except TimeoutError:
                            self.log.error("Timeout processing %s", file_path)
                            pbar.close()
                            raise
                        except Exception as e:
                            self.log.error("Error processing %s: %s", file_path, str(e))
                            pbar.close()
                            raise
                    pbar.close()
                    if completed < len(self.raw_files):
                        raise RuntimeError(f"Only processed {completed}/{len(self.raw_files)} files")
            except Exception as e:
                self.log.warning("Parallel processing failed (%s), falling back to serial mode", str(e))
                # Reset and fall back to serial
                self._drop_report["files"].clear()
                self._drop_report["overall"] = {"n_total": 0, "n_valid": 0, "n_dropped": 0, "n_nan": 0, "n_below_threshold": 0}
                self._valid_group_names.clear()
                per_file_time_candidates.clear()
                effective_workers = 0

        if effective_workers <= 1:
            for path in tqdm(self.raw_files, desc="Scanning HDF5 files (serial)", unit="file",
                             smoothing=TQDM_SMOOTHING, mininterval=TQDM_MININTERVAL, leave=TQDM_LEAVE_OUTER):
                file_report, valid_groups, time_candidate = _scan_one_file_worker(
                    path,
                    self.species_vars,
                    self.global_vars,
                    self.time_key,
                    self.min_value_threshold,
                    self.epsilon,
                    int(self.hdf5_chunk_size),
                )
                self._drop_report["files"].append(file_report)
                for k in ("n_total","n_valid","n_dropped","n_nan","n_below_threshold"):
                    self._drop_report["overall"][k] += file_report[k]
                self._valid_group_names[str(Path(path))] = valid_groups
                if time_candidate is not None:
                    per_file_time_candidates.append(time_candidate)

        # Canonical grid validation across files
        per_file_time_candidates = [t for t in per_file_time_candidates if t is not None and t.size > 0]
        if len(per_file_time_candidates) == 0:
            self._canonical_time = None
        else:
            canon = per_file_time_candidates[0]
            for t in per_file_time_candidates[1:]:
                if not np.array_equal(t, canon):
                    raise ValueError("Time grid differs across files; dataset expects a single shared grid.")
            self._canonical_time = canon.copy()

        self.log.info("Scan complete: %d valid trajectories out of %d total",
                      self._drop_report["overall"]["n_valid"],
                      self._drop_report["overall"]["n_total"])

    # ------------------------- Δt spec (centralized) -------------------------

    def _compute_dt_spec_from_grid(self, t: np.ndarray) -> Dict[str, float]:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        T = int(t.shape[0])
        if T < 2:
            raise ValueError("Encountered time grid with T<2 while computing Δt bounds.")

        diffs = np.diff(t)
        if not np.all(diffs > 0.0):
            raise ValueError("Time grid must be strictly increasing to define positive Δt.")

        min_steps = max(1, self.min_steps)
        max_steps = max(min_steps, min(self.max_steps, T - 1))

        log_min_all = math.inf
        log_max_all = -math.inf

        for k in range(min_steps, max_steps + 1):
            dt_vec = t[k:] - t[:-k]
            if np.any(dt_vec <= 0.0):
                raise ValueError("Non-positive Δt encountered; check time grid and step constraints.")
            vmin = float(np.min(dt_vec)); vmax = float(np.max(dt_vec))
            vmin_c = max(vmin, self.epsilon)
            vmax_c = max(vmax, self.epsilon)
            log_min_all = min(log_min_all, float(np.log10(vmin_c)))
            log_max_all = max(log_max_all, float(np.log10(vmax_c)))

        if not np.isfinite(log_min_all) or not np.isfinite(log_max_all) or log_max_all <= log_min_all:
            raise RuntimeError("Failed to compute valid Δt log bounds (log_max ≤ log_min).")

        return {"method": "log-min-max", "log_min": float(log_min_all), "log_max": float(log_max_all)}

    # ------------------------- PASS 2: write shards --------------------------

    def _write_shards_and_collect_training_stats(self) -> Dict[str, _RunningStats]:
        """
        Write NPZ shards per split using deterministic hashing, and collect
        TRAINING split statistics for normalization.

        Returns a dict of _RunningStats keyed by variable name.
        """
        seed = int(self.cfg.get("system", {}).get("seed", 42))

        # Prepare per-key stats containers for TRAINING ONLY
        keys_all = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        need_flags = {k: _need_flags_for_method(self.methods.get(k, self.default_method)) for k in keys_all}
        train_stats: Dict[str, _RunningStats] = {
            k: _RunningStats(need_mean_std=f[0], need_min_max=f[1], need_log=f[2], epsilon=self.epsilon)
            for k, f in need_flags.items()
        }

        split_dirs = {
            "train": self.processed_dir / "train",
            "validation": self.processed_dir / "validation",
            "test": self.processed_dir / "test",
        }
        for d in split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        shard_counts = {"train": 0, "validation": 0, "test": 0}
        split_counts = {"train": 0, "validation": 0, "test": 0}
        shards_meta = {"train": [], "validation": [], "test": []}

        # Outer file loop with progress
        outer = tqdm(total=len(self.raw_files), desc="Writing shards (by file)", unit="file",
                     smoothing=TQDM_SMOOTHING, mininterval=TQDM_MININTERVAL, leave=TQDM_LEAVE_OUTER)

        for path_str in self.raw_files:
            p = Path(path_str)
            valid_groups = self._valid_group_names.get(str(p), [])
            if not valid_groups:
                outer.update(1)
                continue

            # Deterministic permutation for reproducibility
            rng = np.random.default_rng(seed ^ (hash(p.name) & 0xFFFFFFFF))
            order = rng.permutation(len(valid_groups))
            groups_ordered = [valid_groups[i] for i in order]

            # Per-file progress across groups
            pbar = tqdm(total=len(groups_ordered), desc=f"{p.name}", unit="grp",
                        smoothing=TQDM_SMOOTHING, mininterval=TQDM_MININTERVAL, leave=TQDM_LEAVE_INNER)

            # Batch buffers per split
            buffers = {
                "train": {"x0": [], "g": [], "t": [], "y": []},
                "validation": {"x0": [], "g": [], "t": [], "y": []},
                "test": {"x0": [], "g": [], "t": [], "y": []},
            }

            with h5py.File(p, "r") as f:
                for gname in groups_ordered:
                    grp = f[gname]
                    # Determine split via deterministic hashing (legacy behavior)
                    h = _hash01(f"{gname}:split", seed)
                    if h < self.test_fraction:
                        split = "test"
                    elif h < self.test_fraction + self.val_fraction:
                        split = "validation"
                    else:
                        split = "train"

                    # Apply use_fraction filter (only on the decided split)
                    if self.use_fraction < 1.0:
                        if _hash01(f"{gname}:use", seed) >= self.use_fraction:
                            pbar.update(1)
                            continue

                    # Load arrays again and minimally validate
                    t = np.array(grp[self.time_key], dtype=np.float64, copy=False).reshape(-1)
                    T = int(t.shape[0])
                    # species matrix [T,S]
                    Y = self._read_species_matrix_chunked(grp, T=T)
                    if not np.isfinite(Y).all() or (Y < self.min_value_threshold).any():
                        pbar.update(1)
                        continue

                    # globals vector
                    g_vec = np.array([float(grp.attrs[k]) for k in self.global_vars], dtype=np.float64)
                    if not np.isfinite(g_vec).all():
                        pbar.update(1)
                        continue

                    # x0 [S]
                    x0 = Y[0, :].astype(self.storage_dtype_np, copy=False)

                    # Append to buffers
                    buffers[split]["x0"].append(x0)
                    buffers[split]["g"].append(g_vec.astype(self.storage_dtype_np, copy=False))
                    buffers[split]["t"].append(t.astype(self.storage_dtype_np, copy=False))  # [T], storage dtype
                    buffers[split]["y"].append(Y.astype(self.storage_dtype_np, copy=False))
                    split_counts[split] += 1

                    # TRAINING stats update
                    if split == "train":
                        # time (raw/log depending on method)
                        if train_stats[self.time_key].raw is not None:
                            train_stats[self.time_key].raw.update_array(t[None, :])
                        if train_stats[self.time_key].log is not None:
                            t_log = np.log10(np.clip(t[None, :], self.epsilon, None))
                            train_stats[self.time_key].log.update_array(t_log)

                        # globals (scalars)
                        for gi, gname in enumerate(self.global_vars):
                            val = np.array([g_vec[gi]], dtype=np.float64)
                            if train_stats[gname].raw is not None:
                                train_stats[gname].raw.update_array(val)
                            if train_stats[gname].log is not None:
                                train_stats[gname].log.update_array(np.log10(np.clip(val, self.epsilon, None)))

                        # species (full trajectory)
                        for si, sname in enumerate(self.species_vars):
                            arr = Y[:, si]
                            if train_stats[sname].raw is not None:
                                train_stats[sname].raw.update_array(arr)
                            if train_stats[sname].log is not None:
                                train_stats[sname].log.update_array(np.log10(np.clip(arr, self.epsilon, None)))

                    # Flush to shards if buffer is full
                    for split_name in ("train", "validation", "test"):
                        buf = buffers[split_name]
                        if len(buf["x0"]) >= self.traj_per_shard:
                            filetag = p.stem[:30]
                            out_path = split_dirs[split_name] / SHARD_FILENAME_FMT.format(
                                split=split_name, filetag=filetag, idx=shard_counts[split_name]
                            )
                            self._write_shard_from_buffer(out_path, buf)
                            shards_meta[split_name].append({"filename": out_path.name,
                                                            "n_trajectories": self.traj_per_shard})
                            shard_counts[split_name] += 1
                            for k in buf.keys():
                                buf[k].clear()

                    pbar.update(1)

                # flush tail buffers
                for split_name in ("train", "validation", "test"):
                    buf = buffers[split_name]
                    if buf["x0"]:
                        filetag = p.stem[:30]
                        out_path = split_dirs[split_name] / SHARD_FILENAME_FMT.format(
                            split=split_name, filetag=filetag, idx=shard_counts[split_name]
                        )
                        self._write_shard_from_buffer(out_path, buf)
                        shards_meta[split_name].append({"filename": out_path.name,
                                                        "n_trajectories": len(buf["x0"])})
                        shard_counts[split_name] += 1
                        for k in buf.keys():
                            buf[k].clear()

            pbar.close()
            outer.update(1)

        outer.close()

        # Persist legacy-style indices/summaries now
        self._write_shard_index(shards_meta)
        self._write_preprocessing_summary(split_counts)

        self.log.info("Wrote shards. Totals (trajectories): train=%d, val=%d, test=%d",
                      split_counts["train"], split_counts["validation"], split_counts["test"])

        return train_stats

    def _write_shard_from_buffer(self, out_path: Path, buf: Dict[str, List[np.ndarray]]) -> None:
        """
        Strictly verify the time grid is identical across all buffered trajectories,
        then write a single 1-D [T] `t_vec` per shard.
        """
        # Stack payloads
        x0 = np.stack(buf["x0"], axis=0)  # [N,S]
        g = (np.stack(buf["g"], axis=0) if self.global_vars
             else np.zeros((len(buf["x0"]), 0), dtype=self.storage_dtype_np))
        y = np.stack(buf["y"], axis=0)    # [N,T,S]

        # ---- STRICT time verification
        if len(buf["t"]) == 0:
            raise RuntimeError("Empty buffer passed to _write_shard_from_buffer (no time vectors).")
        t0 = np.asarray(buf["t"][0])  # already storage dtype, shape [T]
        for idx, t_i in enumerate(buf["t"][1:], start=1):
            if not np.array_equal(t0, t_i):
                # Find first diff for easier debugging
                a = np.asarray(t0).reshape(-1)
                b = np.asarray(t_i).reshape(-1)
                mism = np.where(a != b)[0]
                first = int(mism[0]) if mism.size else -1
                raise RuntimeError(
                    f"Intra-shard time mismatch while writing {out_path.name}: "
                    f"trajectory 0 vs {idx} differ at index {first} "
                    f"(t0={a[first] if first>=0 else 'n/a'}, t{idx}={b[first] if first>=0 else 'n/a'})."
                )
        t1d = t0  # 1-D [T], storage dtype

        # ---- Write NPZ with 1-D t_vec
        if self.npz_compressed:
            np.savez_compressed(out_path, x0=x0, globals=g, t_vec=t1d, y_mat=y)
        else:
            np.savez(out_path, x0=x0, globals=g, t_vec=t1d, y_mat=y)

    # --------------------------- Manifest finalize ---------------------------

    def _finalize_manifest(self, train_stats: Dict[str, _RunningStats], dt_spec: Dict[str, float]) -> Dict[str, Any]:
        per_key_stats: Dict[str, Dict[str, float]] = {}
        for key, acc in train_stats.items():
            per_key_stats[key] = acc.to_manifest(self.min_std)

        methods: Dict[str, str] = {}
        keys_all = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        for k in keys_all:
            methods[k] = self.methods.get(k, self.default_method)

        return {
            "per_key_stats": per_key_stats,
            "normalization_methods": methods,
            "epsilon": self.epsilon,
            "min_std": self.min_std,
            "clamp_value": self.clamp_value,
            "dt": dt_spec
        }

    # ------------------------------ Pass 2: IO -------------------------------

    def _choose_chunk_len(self, dset, T: int) -> int:
        cfg_len = int(getattr(self, "hdf5_chunk_size", DEFAULT_HDF5_CHUNK_FALLBACK) or 0)
        if cfg_len > 0:
            return max(1, min(cfg_len, T))
        try:
            if hasattr(dset, "chunks") and dset.chunks and len(dset.chunks) >= 1 and dset.chunks[0]:
                return max(1, min(int(dset.chunks[0]), T))
        except Exception:
            pass
        return T

    def _read_species_matrix_chunked(self, grp, T: int) -> np.ndarray:
        S = len(self.species_vars)
        Y = np.empty((T, S), dtype=np.float64)
        for si, sname in enumerate(self.species_vars):
            dset = grp[sname]
            if int(dset.shape[0]) != T:
                raise ValueError("species length mismatch vs time")
            chunk_len = self._choose_chunk_len(dset, T)
            pos = 0
            for s0 in range(0, T, chunk_len):
                s1 = min(T, s0 + chunk_len)
                sl = np.array(dset[s0:s1], dtype=np.float64, copy=False).reshape(-1)
                Y[s0:s1, si] = sl
                pos = s1
            if pos != T:
                raise RuntimeError("Chunked read did not cover full time window.")
        return Y

    # --------------------------- Reporting outputs ---------------------------

    def _write_shard_index(self, shards_meta: Dict[str, List[Dict[str, Any]]]) -> None:
        T = int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0
        shard_index = {
            "sequence_mode": True,
            "variable_length": False,
            "M_per_sample": T,
            "n_input_species": len(self.species_vars),
            "n_target_species": len(self.target_species_vars),
            "n_globals": len(self.global_vars),
            "compression": "npz",
            "splits": {k: {"shards": v, "n_trajectories": sum(x["n_trajectories"] for x in v)}
                       for k, v in shards_meta.items()}
        }
        with open(self.processed_dir / "shard_index.json", "w", encoding="utf-8") as f:
            json.dump(shard_index, f, indent=2)
        self.log.info("Wrote shard_index.json")

    def _write_preprocessing_summary(self, split_counts: Dict[str, int]) -> None:
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "valid_trajectories": {
                "train": int(split_counts["train"]),
                "validation": int(split_counts["validation"]),
                "test": int(split_counts["test"]),
            },
            "overall_from_scan": self._drop_report["overall"],
            "time_grid_len": int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0
        }
        with open(self.processed_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.log.info("Wrote preprocessing_summary.json")

    def _write_reports(self, t_start: float, t1: float, t2: float, t3: float, t4: float, t5: float) -> None:
        report = {
            "min_value_threshold": self.min_value_threshold,
            "overall_from_scan": self._drop_report["overall"],
            "timings_sec": {
                "pass1_scan": round(t2 - t1, 3),
                "dt_spec": round(t3 - t2, 3),
                "write_shards_plus_stats": round(t4 - t3, 3),
                "manifest_write": round(t5 - t4, 3),
                "total": round(t5 - t_start, 3),
            },
            "splits_fraction": {
                "train": float(1.0 - self.val_fraction - self.test_fraction),
                "validation": float(self.val_fraction),
                "test": float(self.test_fraction),
            },
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t5))
        }
        with open(self.processed_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.log.info("Wrote preprocess_report.json")


# ----------------------------- Time-grid auditor -----------------------------

def audit_time_grids(
    processed_dir: str | Path,
    *,
    round_decimals: Optional[int] = None,
    save_dirname: str = "timegrid_audit",
    save_npys: bool = True,
    max_examples_per_grid: int = 10,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Scan all shards under processed_dir/{train,validation,test} and collect unique time grids.

    - Computes a stable hash for each grid (shape + bytes), after canonicalizing to float64.
    - Optional rounding (round_decimals) lets you see whether tiny dtype/ulp differences are the cause.
    - Saves each unique grid to processed_dir/save_dirname/timegrid_<k>.npy.
    - Emits a JSON summary: processed_dir/save_dirname/summary.json.

    Accepts both the new [T] and legacy [N, T] `t_vec` encodings.
    """
    processed_dir = Path(processed_dir)
    out_dir = processed_dir / save_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = log or logging.getLogger("timegrid_audit")

    splits = ["train", "validation", "test"]
    shard_paths: List[Path] = []
    for sp in splits:
        shard_paths.extend(sorted((processed_dir / sp).glob("shard_*.npz")))

    def _canon(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if round_decimals is not None:
            a = np.around(a, decimals=int(round_decimals)).copy()
        else:
            a = np.ascontiguousarray(a)
        return a

    def _hash_grid(x: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(str(x.shape).encode("utf-8"))
        h.update(x.tobytes(order="C"))
        return h.hexdigest()

    summary: Dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "round_decimals": round_decimals,
        "total_shards": 0,
        "total_rows_checked": 0,
        "unique_grids": [],
        "intra_shard_mismatches": [],
    }

    registry: Dict[str, Dict[str, Any]] = {}

    lg.info("Auditing time grids in %s (%d shards detected)…", processed_dir, len(shard_paths))
    for shard_path in shard_paths:
        try:
            with np.load(shard_path, allow_pickle=False, mmap_mode="r") as z:
                t = z["t_vec"]
        except Exception as e:
            lg.error("Failed to open %s: %s", shard_path, e)
            continue

        summary["total_shards"] += 1

        if t.ndim == 1:
            # New encoding: a single grid per shard
            B, T = 1, int(t.shape[0])
            summary["total_rows_checked"] += B
            canon0 = _canon(t)
            key0 = _hash_grid(canon0)
            info = registry.get(key0)
            if info is None:
                preview = {
                    "T": T,
                    "dtype_seen": str(t.dtype),
                    "min": float(np.min(canon0)),
                    "max": float(np.max(canon0)),
                    "head": [float(x) for x in canon0[:min(5, T)]],
                    "tail": [float(x) for x in canon0[max(0, T-5):]],
                }
                registry[key0] = info = {
                    "key": key0,
                    "count_rows": 0,
                    "count_shards": 0,
                    "example_shards": [],
                    "preview": preview,
                    "npy_path": None
                }
            info["count_rows"] += B
            info["count_shards"] += 1
            if len(info["example_shards"]) < max_examples_per_grid:
                info["example_shards"].append(str(shard_path))
            if save_npys and info["npy_path"] is None:
                npy_path = out_dir / f"timegrid_{len(registry):04d}.npy"
                np.save(npy_path, canon0)
                info["npy_path"] = str(npy_path)

        elif t.ndim == 2:
            # Legacy encoding: one row per trajectory, should be identical
            B, T = int(t.shape[0]), int(t.shape[1])
            summary["total_rows_checked"] += B

            row0 = t[0]
            diffs = np.any(t != row0, axis=1)
            if np.any(diffs):
                shard_keys: Dict[str, int] = {}
                for b in range(B):
                    canon = _canon(t[b])
                    key = _hash_grid(canon)
                    shard_keys[key] = shard_keys.get(key, 0) + 1
                summary["intra_shard_mismatches"].append({
                    "shard": str(shard_path),
                    "unique_grids_in_shard": len(shard_keys),
                    "counts": shard_keys
                })
                lg.warning("Intra-shard mismatch: %s has %d unique time grids", shard_path.name, len(shard_keys))

            canon0 = _canon(row0)
            key0 = _hash_grid(canon0)
            info = registry.get(key0)
            if info is None:
                preview = {
                    "T": T,
                    "dtype_seen": str(t.dtype),
                    "min": float(np.min(canon0)),
                    "max": float(np.max(canon0)),
                    "head": [float(x) for x in canon0[:min(5, T)]],
                    "tail": [float(x) for x in canon0[max(0, T-5):]],
                }
                registry[key0] = info = {
                    "key": key0,
                    "count_rows": 0,
                    "count_shards": 0,
                    "example_shards": [],
                    "preview": preview,
                    "npy_path": None
                }

            info["count_rows"] += B
            info["count_shards"] += 1
            if len(info["example_shards"]) < max_examples_per_grid:
                info["example_shards"].append(str(shard_path))
            if save_npys and info["npy_path"] is None:
                npy_path = out_dir / f"timegrid_{len(registry):04d}.npy"
                np.save(npy_path, canon0)
                info["npy_path"] = str(npy_path)

        else:
            lg.error("Shard %s has t_vec ndim=%d (expected 1 or 2)", shard_path, t.ndim)
            continue

    summary["unique_grids"] = list(registry.values())

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lg.info("Time grid audit complete: %d unique grids across %d shards. Report: %s",
            len(registry), summary["total_shards"], json_path)

    if len(registry) > 1:
        lg.warning("Found %d unique time grids. See %s for details.", len(registry), json_path)

    return summary
