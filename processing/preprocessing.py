#!/usr/bin/env python3
"""
preprocessing.py

Resample adaptive-timestep HDF5 trajectories into fixed-length (n_steps) uniform-dt profiles,
shard to NPZ, compute normalization, and write final normalized shards.

Key properties:
- Sampling/interpolation core matches the fast regrid.py approach: precompute indices/weights once per chunk,
  then vectorized log-log interpolation for species.
- Keeps larger-codebase functionality: outputs uncompressed NPZ shards + normalization.json + preprocessing_summary.json.
- Robust to raw HDF5 schema:
  - Species/global datasets may be nested in subgroups.
  - Names may have optional evolve_ prefix (config can use either).
  - Globals may be scalars or time-series aligned with t_raw (reduced to scalar).
- Provides clear “stuck” diagnostics: attempt-logging and reject summaries.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm
import math


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


def load_json_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _name_candidates(base: str) -> List[str]:
    out = [base]
    if base.startswith("evolve_"):
        out.append(base[len("evolve_") :])
    else:
        out.append("evolve_" + base)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_leaf_index(grp: h5py.Group) -> Dict[str, List[str]]:
    """
    Build index: leaf_name -> list of relative dataset paths under grp.
    Example: if dataset path is "state/evolve_H", leaf is "evolve_H".
    """
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def get_time_array_recursive(grp: h5py.Group, time_keys: List[str], leaf_index: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Find time array by trying:
      1) direct grp[k]
      2) any nested dataset with leaf == k
    Must be 1D, finite, strictly increasing.
    Returns (path_or_key, t_raw) or (None, None)
    """
    # direct
    for k in time_keys:
        if k in grp and isinstance(grp[k], h5py.Dataset):
            t = np.asarray(grp[k][...], dtype=np.float64).reshape(-1)
            if t.size >= 2 and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0):
                return k, t

    # nested by leaf
    for k in time_keys:
        for p in leaf_index.get(k, []):
            ds = grp[p]
            if not isinstance(ds, h5py.Dataset):
                continue
            if ds.ndim != 1 or len(ds) < 2:
                continue
            t = np.asarray(ds[...], dtype=np.float64).reshape(-1)
            if t.size >= 2 and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0):
                return p, t

    return None, None


def find_dataset_path(
    grp: h5py.Group,
    base: str,
    leaf_index: Dict[str, List[str]],
    *,
    t_len: Optional[int] = None,
    prefer_time_aligned: bool = True,
) -> Optional[str]:
    """
    Resolve dataset by leaf name (exact), searching nested paths.
    If t_len is provided and prefer_time_aligned=True, prefer datasets with shape[0]==t_len.
    """
    for cand in _name_candidates(base):
        paths = leaf_index.get(cand, [])
        if not paths:
            continue
        if t_len is not None and prefer_time_aligned:
            aligned = []
            for p in paths:
                try:
                    ds = grp[p]
                    if isinstance(ds, h5py.Dataset) and ds.shape and int(ds.shape[0]) == int(t_len):
                        aligned.append(p)
                except Exception:
                    continue
            if aligned:
                return aligned[0]
        return paths[0]
    return None


def read_global_flexible_recursive(
    grp: h5py.Group,
    key_base: str,
    t_len: int,
    leaf_index: Dict[str, List[str]],
    warn_state: Dict[str, bool],
) -> float:
    """
    Read a “global” conditioning scalar.
    Supports:
      - scalar dataset/attr on grp or root
      - nested scalar dataset (leaf match) under grp
      - time-series dataset aligned with t_len (leaf match) under grp or root: reduce to scalar

    Reduction rule for time-series:
      - if nearly constant -> mean
      - else -> first value (warn once)
    """
    # 1) attrs on grp/root
    for cand in _name_candidates(key_base):
        if cand in grp.attrs:
            v = np.asarray(grp.attrs[cand]).reshape(-1)
            if v.size == 1 and np.isfinite(v[0]):
                return float(v[0])
        if cand in grp.file.attrs:
            v = np.asarray(grp.file.attrs[cand]).reshape(-1)
            if v.size == 1 and np.isfinite(v[0]):
                return float(v[0])

    # 2) direct datasets on grp/root
    for cand in _name_candidates(key_base):
        if cand in grp and isinstance(grp[cand], h5py.Dataset):
            arr = np.asarray(grp[cand][...], dtype=np.float64)
            if arr.size == 1 and np.isfinite(arr.reshape(-1)[0]):
                return float(arr.reshape(-1)[0])
        if cand in grp.file and isinstance(grp.file[cand], h5py.Dataset):
            arr = np.asarray(grp.file[cand][...], dtype=np.float64)
            if arr.size == 1 and np.isfinite(arr.reshape(-1)[0]):
                return float(arr.reshape(-1)[0])

    # 3) nested dataset under grp (leaf match)
    p = find_dataset_path(grp, key_base, leaf_index, t_len=t_len, prefer_time_aligned=False)
    if p is not None:
        ds = grp[p]
        arr = np.asarray(ds[...], dtype=np.float64)
        flat = arr.reshape(-1)
        if flat.size == 1 and np.isfinite(flat[0]):
            return float(flat[0])
        # time-series aligned?
        if arr.shape and int(arr.shape[0]) == int(t_len):
            x = arr.reshape(t_len, -1)[:, 0]
            if not np.all(np.isfinite(x)):
                raise KeyError(f"Global '{key_base}' found at '{p}' but not finite.")
            mu = float(np.mean(x))
            sd = float(np.std(x))
            if sd <= 1e-12 * max(abs(mu), 1.0):
                return mu
            if not warn_state.get(p, False):
                log(f"Warning: global '{key_base}' at '{p}' varies over time; using first value.")
                warn_state[p] = True
            return float(x[0])

    # 4) root nested: not typical; skip deep traversal to avoid huge cost
    raise KeyError(f"Missing/invalid global '{key_base}' (tried { _name_candidates(key_base) })")


def _prepare_log_interp(log_t_valid: np.ndarray, log_t_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(log_t_valid.shape[0])
    idx = np.searchsorted(log_t_valid, log_t_chunk, side="left")
    idx = np.clip(idx, 1, n - 1)
    i0 = (idx - 1).astype(np.int64)
    i1 = idx.astype(np.int64)
    t0 = log_t_valid[i0]
    t1 = log_t_valid[i1]
    denom = np.maximum(t1 - t0, 1e-300)
    w = (log_t_chunk - t0) / denom
    w = np.clip(w, 0.0, 1.0).astype(np.float64)
    return i0, i1, w


def interp_loglog_species(y_valid: np.ndarray, i0: np.ndarray, i1: np.ndarray, w: np.ndarray) -> np.ndarray:
    y0 = np.log10(np.maximum(y_valid[i0, :], 1e-300))
    y1 = np.log10(np.maximum(y_valid[i1, :], 1e-300))
    w2 = w.reshape(-1, 1)
    ylog = (1.0 - w2) * y0 + w2 * y1
    return 10.0 ** ylog


@dataclass
class PreCfg:
    raw_dir: Path
    processed_dir: Path

    dt: float
    n_steps: int
    t_min: float

    output_trajectories_per_file: int
    max_chunks_per_source_trajectory: int
    anchor_first_chunk: bool
    max_sampling_attempts_per_file: int

    drop_below: float
    time_keys: List[str]

    val_fraction: float
    test_fraction: float
    shard_size: int
    overwrite: bool

    global_variables: List[str]
    species_variables: List[str]

    epsilon: float
    min_std: float
    methods: Dict[str, str]
    default_method: str
    globals_default_method: str

    log_every_n_trajectories: int
    log_every_n_attempts: int
    seed: int
    pool_size: int

    # Balanced sampling (optional): sample a fixed number of chunks from a fixed number of source trajectories
    source_trajectories_per_file: int
    samples_per_source_trajectory: int
    max_source_selection_attempts: int
    max_chunk_attempts_per_source: int


def load_precfg(cfg: Dict) -> PreCfg:
    pcfg = cfg.get("paths", {})
    raw_dir = Path(pcfg.get("raw_data_dir", PROJECT_ROOT / "data" / "raw"))
    processed_dir = Path(pcfg.get("processed_data_dir", PROJECT_ROOT / "data" / "processed"))
    if not raw_dir.is_absolute():
        raw_dir = (PROJECT_ROOT / raw_dir).resolve()
    if not processed_dir.is_absolute():
        processed_dir = (PROJECT_ROOT / processed_dir).resolve()

    pr = cfg.get("preprocessing", {})
    dcfg = cfg.get("data", {})
    ncfg = cfg.get("normalization", {})

    return PreCfg(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dt=float(pr.get("dt", 100.0)),
        n_steps=int(pr.get("n_steps", 1000)),
        t_min=float(pr.get("t_min", 1e-3)),

        output_trajectories_per_file=int(pr.get("output_trajectories_per_file", 100)),
        max_chunks_per_source_trajectory=int(pr.get("max_chunks_per_source_trajectory", 0)),
        anchor_first_chunk=bool(pr.get("anchor_first_chunk", True)),
        max_sampling_attempts_per_file=int(pr.get("max_sampling_attempts_per_file", 5_000_000)),

        drop_below=float(pr.get("drop_below", 1e-35)),
        time_keys=list(pr.get("time_keys", ["t_time", "time", "t"])),

        shard_size=int(pr.get("shard_size", 1024)),
        overwrite=bool(pr.get("overwrite", True)),
        val_fraction=float(pr.get("val_fraction", 0.1)),
        test_fraction=float(pr.get("test_fraction", 0.1)),

        global_variables=list(dcfg.get("global_variables", ["P", "T"])),
        species_variables=list(dcfg.get("species_variables", [])),

        epsilon=float(ncfg.get("epsilon", 1e-30)),
        min_std=float(ncfg.get("min_std", 1e-12)),
        methods=dict(ncfg.get("methods", {})),
        default_method=str(ncfg.get("default_method", "log-standard")),
        globals_default_method=str(ncfg.get("globals_default_method", "standard")),

        log_every_n_trajectories=int(pr.get("log_every_n_trajectories", 50)),
        log_every_n_attempts=int(pr.get("log_every_n_attempts", 200_000)),
        seed=int(pr.get("seed", cfg.get("system", {}).get("seed", 1234))),
        pool_size=int(pr.get("pool_size", 20_000)),
        source_trajectories_per_file=int(pr.get("source_trajectories_per_file", 0)),
        samples_per_source_trajectory=int(pr.get("samples_per_source_trajectory", 0)),
        max_source_selection_attempts=int(pr.get("max_source_selection_attempts", 1_000_000)),
        max_chunk_attempts_per_source=int(pr.get("max_chunk_attempts_per_source", 200)),
    )


def _split_name(u: float, val_fraction: float, test_fraction: float) -> str:
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "validation"
    return "train"


def clean_tmp_dir(tmp_dir: Path, overwrite: bool) -> None:
    if tmp_dir.exists():
        if not overwrite:
            raise RuntimeError(f"Temp dir exists and overwrite=false: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def clean_final_outputs(processed_dir: Path, overwrite: bool) -> None:
    """
    Clean final outputs without touching _tmp_physical.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for split in ("train", "validation", "test"):
            if (processed_dir / split).exists():
                raise RuntimeError(f"Processed dir has existing split '{split}' and overwrite=false: {processed_dir}")
        for fn in ("normalization.json", "preprocessing_summary.json"):
            if (processed_dir / fn).exists():
                raise RuntimeError(f"Processed dir has existing '{fn}' and overwrite=false: {processed_dir}")
        return

    for split in ("train", "validation", "test"):
        d = processed_dir / split
        if d.exists():
            shutil.rmtree(d)

    for fn in ("normalization.json", "preprocessing_summary.json"):
        p = processed_dir / fn
        if p.exists():
            p.unlink()


def flush_shard(
    out_dir: Path,
    split: str,
    shard_idx: int,
    y_buf: List[np.ndarray],
    g_buf: List[np.ndarray],
    t_vec: np.ndarray,
    suffix: str,
) -> Tuple[int, int]:
    if not y_buf:
        return shard_idx, 0
    out_dir_split = out_dir / split
    out_dir_split.mkdir(parents=True, exist_ok=True)
    y_mat = np.stack(y_buf, axis=0)
    g_mat = np.stack(g_buf, axis=0)
    shard_path = out_dir_split / f"shard_{shard_idx:06d}{suffix}.npz"
    np.savez(shard_path, y_mat=y_mat, globals=g_mat, t_vec=t_vec.astype(np.float64, copy=False))
    return shard_idx + 1, int(y_mat.shape[0])


def iter_shards(root: Path, split: str, suffix: str) -> List[Path]:
    d = root / split
    if not d.exists():
        return []
    return sorted(d.glob(f"shard_*{suffix}.npz"))


def reservoir_sample(keys_iter, k: int, rng: np.random.Generator) -> List[str]:
    pool: List[str] = []
    for i, key in enumerate(keys_iter):
        if i < k:
            pool.append(key)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                pool[j] = key
    return pool


def sample_trajectories_from_file(
    file_path: Path,
    out_tmp: Path,
    cfg: PreCfg,
    rng: np.random.Generator,
    chunk_offsets: np.ndarray,
    t_vec_rel: np.ndarray,
    counts_total: Dict[str, int],
    shard_idx: Dict[str, int],
    last_log_count: int,
) -> Tuple[int, Dict[str, int]]:
    """
    Sample fixed-length trajectories from one HDF5 file and write physical shards.

    Two modes are supported:

    1) Default (legacy): random sampling of (traj, start_time) pairs until
       output_trajectories_per_file is reached or max_sampling_attempts_per_file is hit.

    2) Balanced per-trajectory sampling (recommended for "K samples per raw trajectory"):
       enabled when preprocessing.source_trajectories_per_file > 0 AND
       preprocessing.samples_per_source_trajectory > 0.

       The sampler will:
         - Select N "good" source trajectories (pass filters: time present, species present, all species >= drop_below, etc.)
         - Sample up to K chunks per selected trajectory (first chunk can be anchored if anchor_first_chunk=true)
         - Stop once output_trajectories_per_file samples are written (or N*K if output_trajectories_per_file<=0)

    Instrumentation:
      - Tracks reject reasons with counts AND a few concrete examples per reason.
      - Prints a ranked reject summary at end of file processing.
    """
    y_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    g_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}

    # Reasons must be stable keys (so you can diff runs).
    reject_stats: Dict[str, int] = {
        "not_group": 0,
        "no_time": 0,
        "no_positive_time": 0,
        "missing_species": 0,
        "missing_globals": 0,
        "drop_below": 0,
        "non_finite": 0,
        "chunk_no_span": 0,
        "chunk_out_of_bounds": 0,
        "interp_non_finite": 0,
        "cache_bad_traj": 0,
        "insufficient_good_traj": 0,
        "failed_to_sample_chunk": 0,
    }

    # Store a small number of examples per reject reason to see EXACTLY what failed.
    REJECT_EXAMPLES_PER_REASON = 6
    reject_examples: Dict[str, List[str]] = {k: [] for k in reject_stats.keys()}

    def reject(reason: str, detail: str = "") -> None:
        # Count
        if reason not in reject_stats:
            reject_stats[reason] = 0
            reject_examples[reason] = []
        reject_stats[reason] += 1
        # Collect examples (bounded)
        if detail and len(reject_examples[reason]) < REJECT_EXAMPLES_PER_REASON:
            reject_examples[reason].append(detail)

    warn_state: Dict[str, bool] = {}
    chunk_duration = float(chunk_offsets[-1])  # last offset = (n_steps-1)*dt

    per_source_traj_counts: Dict[str, int] = {}
    anchor_fail_logged: set = set()

    CACHE_MAX = 32
    traj_cache: Dict[str, Dict[str, np.ndarray]] = {}
    traj_cache_order: List[str] = []
    bad_traj: set = set()

    printed_schema_hint = False  # one-time debug print for missing species

    fixed_species_vars: List[str] = cfg.species_variables[:]  # preserve config order
    fixed_global_vars: List[str] = cfg.global_variables[:]

    def _cache_touch(name: str) -> None:
        if name in traj_cache_order:
            traj_cache_order.remove(name)
        traj_cache_order.append(name)

    def _cache_put(name: str, data: Dict[str, np.ndarray]) -> None:
        traj_cache[name] = data
        _cache_touch(name)
        if len(traj_cache_order) > CACHE_MAX:
            old = traj_cache_order.pop(0)
            traj_cache.pop(old, None)

    def _get_traj_data(traj_name: str, grp: h5py.Group) -> Optional[Dict[str, np.ndarray]]:
        nonlocal printed_schema_hint

        if traj_name in bad_traj:
            reject("cache_bad_traj", f"{traj_name}: previously marked bad")
            return None

        cached = traj_cache.get(traj_name)
        if cached is not None:
            _cache_touch(traj_name)
            return cached

        # Build leaf index once per trajectory load
        leaf_index = build_leaf_index(grp)

        # Time (recursive)
        time_path, t_raw = get_time_array_recursive(grp, cfg.time_keys, leaf_index)
        if t_raw is None or time_path is None:
            reject("no_time", f"{traj_name}: no time key in {cfg.time_keys}")
            bad_traj.add(traj_name)
            return None

        pos = t_raw > 0
        if not np.any(pos):
            reject("no_positive_time", f"{traj_name}: all t<=0 (min={t_raw.min():.3e}, max={t_raw.max():.3e})")
            bad_traj.add(traj_name)
            return None

        first_pos_time = float(t_raw[pos][0])
        t_lo = max(cfg.t_min, first_pos_time)

        valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
        if not np.any(valid):
            reject("no_positive_time", f"{traj_name}: no valid t after filter (t_lo={t_lo:.3e})")
            bad_traj.add(traj_name)
            return None

        t_valid = t_raw[valid]
        if t_valid.size < 2:
            reject("no_time", f"{traj_name}: too few valid time points ({t_valid.size})")
            bad_traj.add(traj_name)
            return None

        T = int(t_raw.shape[0])

        # Resolve species dataset paths (recursive, leaf match), prefer time-aligned
        species_paths: List[str] = []
        missing = []
        for s in fixed_species_vars:
            p = find_dataset_path(grp, s, leaf_index, t_len=T, prefer_time_aligned=True)
            if p is None:
                missing.append(s)
            else:
                species_paths.append(p)

        if missing:
            reject("missing_species", f"{traj_name}: missing={missing[:6]} (total_missing={len(missing)})")
            if not printed_schema_hint:
                printed_schema_hint = True
                leaves = sorted(list(leaf_index.keys()))
                preview = ", ".join(leaves[:120])
                log(
                    f"[{file_path.name}] missing species example traj='{traj_name}'. "
                    f"Available dataset leaf names (first 120): {preview}"
                )
            bad_traj.add(traj_name)
            return None

        # Load species matrix [T_raw, S]
        S = int(len(species_paths))
        y_raw = np.empty((T, S), dtype=np.float64)
        for j, p in enumerate(species_paths):
            ds = grp[p]
            arr = np.asarray(ds[...], dtype=np.float64)
            if not arr.shape or int(arr.shape[0]) != T:
                reject("missing_species", f"{traj_name}: path='{p}' not time-aligned (shape={arr.shape}, T={T})")
                bad_traj.add(traj_name)
                return None
            x = arr.reshape(T, -1)[:, 0]
            if not np.all(np.isfinite(x)):
                reject("non_finite", f"{traj_name}: non-finite in '{p}'")
                bad_traj.add(traj_name)
                return None
            y_raw[:, j] = x

        # Strict filter: reject any trajectory that dips below drop_below (including exact zeros).
        if np.any(y_raw < cfg.drop_below):
            mn = float(y_raw.min())
            reject("drop_below", f"{traj_name}: min_species={mn:.3e} < drop_below={cfg.drop_below:.3e}")
            bad_traj.add(traj_name)
            return None

        y_valid = y_raw[valid, :]
        if not np.all(np.isfinite(y_valid)):
            reject("non_finite", f"{traj_name}: non-finite after valid time filter")
            bad_traj.add(traj_name)
            return None

        log_t_valid = np.log10(t_valid)

        # Globals (recursive leaf match + scalar/time-series reduction)
        try:
            g_vec = np.array(
                [read_global_flexible_recursive(grp, gv, T, leaf_index, warn_state) for gv in fixed_global_vars],
                dtype=np.float64,
            )
        except Exception as e:
            reject("missing_globals", f"{traj_name}: globals missing/invalid ({e})")
            bad_traj.add(traj_name)
            return None

        if not np.all(np.isfinite(g_vec)):
            reject("non_finite", f"{traj_name}: non-finite globals")
            bad_traj.add(traj_name)
            return None

        data = {
            "t_raw": t_raw,
            "t_lo": np.float64(t_lo),
            "t_valid": t_valid,
            "log_t_valid": log_t_valid,
            "y_valid": y_valid,
            "g_vec": g_vec,
        }
        _cache_put(traj_name, data)
        return data

    def _feasible_start_range(td: Dict[str, np.ndarray]) -> Optional[Tuple[float, float]]:
        """Return (t_start_min, t_start_max) ensuring the full chunk fits in valid times."""
        t_raw = td["t_raw"]
        t_lo = float(td["t_lo"])
        t_valid = td["t_valid"]

        # Tightest constraints: must fit within BOTH raw and valid region.
        t_start_min = max(t_lo, float(t_valid[0]))
        t_start_max = min(float(t_raw[-1] - chunk_duration), float(t_valid[-1] - chunk_duration))

        if not (np.isfinite(t_start_min) and np.isfinite(t_start_max)):
            return None
        if t_start_max <= t_start_min:
            return None
        if t_start_min <= 0.0:
            # log-uniform sampling requires positive start times
            return None
        return float(t_start_min), float(t_start_max)

    with h5py.File(file_path, "r") as fin:
        pool = reservoir_sample(fin.keys(), cfg.pool_size, rng)
        if not pool:
            log(f"[{file_path.name}] no groups at root; skipping.")
            return last_log_count, reject_stats

        # ---------------- Balanced per-trajectory sampling mode ----------------
        if cfg.source_trajectories_per_file > 0 and cfg.samples_per_source_trajectory > 0:
            N = int(cfg.source_trajectories_per_file)
            K = int(cfg.samples_per_source_trajectory)

            target_total = int(cfg.output_trajectories_per_file)
            if target_total <= 0:
                target_total = N * K

            # If target_total is not divisible by K, the last trajectory will contribute fewer samples.
            if N * K < target_total:
                N = int(math.ceil(target_total / float(K)))

            log(
                f"[{file_path.name}] Balanced sampling enabled: "
                f"source_trajectories_per_file={N}, samples_per_source_trajectory={K}, target_total={target_total}"
            )

            # Select N good trajectories
            selected: List[str] = []
            seen: set = set()
            attempts_sel = 0
            max_sel = int(cfg.max_source_selection_attempts)

            # Random draws from the (potentially large) pool, without replacement.
            # This avoids materializing all keys for huge files.
            while len(selected) < N and attempts_sel < max_sel and len(seen) < len(pool):
                attempts_sel += 1
                traj_name = pool[int(rng.integers(0, len(pool)))]
                if traj_name in seen:
                    continue
                seen.add(traj_name)

                if traj_name not in fin:
                    continue
                grp = fin[traj_name]
                if not isinstance(grp, h5py.Group):
                    reject("not_group", f"{traj_name}: root item not a group")
                    continue

                td = _get_traj_data(traj_name, grp)
                if td is None:
                    continue

                span = _feasible_start_range(td)
                if span is None:
                    reject(
                        "chunk_no_span",
                        f"{traj_name}: insufficient span for a chunk of duration={chunk_duration:.3e}",
                    )
                    bad_traj.add(traj_name)
                    continue

                selected.append(traj_name)

            if len(selected) < N:
                reject(
                    "insufficient_good_traj",
                    f"found={len(selected)}/{N} after attempts_sel={attempts_sel}, pool_size={len(pool)}",
                )
                log(
                    f"[{file_path.name}] Warning: only found {len(selected)}/{N} usable source trajectories "
                    f"from pool_size={len(pool)} after {attempts_sel} selection attempts. "
                    f"Proceeding with {len(selected)}."
                )

            # Sample K chunks per selected trajectory until target_total reached
            written = 0
            for traj_name in selected:
                if written >= target_total:
                    break

                grp = fin[traj_name]
                td = _get_traj_data(traj_name, grp)
                if td is None:
                    continue  # should be rare, but keep safe

                span = _feasible_start_range(td)
                if span is None:
                    continue
                t_start_min, t_start_max = span

                t_valid = td["t_valid"]
                log_t_valid = td["log_t_valid"]
                y_valid = td["y_valid"]
                g_vec = td["g_vec"]

                remaining = int(target_total - written)
                chunks_to_take = min(int(K), remaining)

                ccount = int(per_source_traj_counts.get(traj_name, 0))

                for k in range(chunks_to_take):
                    # Honor the legacy cap as an additional safeguard.
                    if cfg.max_chunks_per_source_trajectory > 0 and ccount >= cfg.max_chunks_per_source_trajectory:
                        break

                    ok = False
                    max_chunk_tries = int(cfg.max_chunk_attempts_per_source)

                    for a in range(max_chunk_tries):
                        # Anchor first chunk per trajectory if requested, but do not get stuck if infeasible.
                        if cfg.anchor_first_chunk and ccount == 0 and k == 0 and a == 0:
                            t_start = t_start_min
                        else:
                            t_start = 10.0 ** float(rng.uniform(np.log10(t_start_min), np.log10(t_start_max)))

                        t_chunk = t_start + chunk_offsets

                        # Ensure chunk fits within valid time region
                        if t_chunk[0] < t_valid[0] or t_chunk[-1] > t_valid[-1]:
                            reject(
                                "chunk_out_of_bounds",
                                f"{traj_name}: t_start={t_start:.3e}, chunk=[{t_chunk[0]:.3e},{t_chunk[-1]:.3e}] "
                                f"valid=[{t_valid[0]:.3e},{t_valid[-1]:.3e}] "
                                f"(anchor={cfg.anchor_first_chunk and ccount==0 and k==0 and a==0})",
                            )
                            if cfg.anchor_first_chunk and ccount == 0 and k == 0 and a == 0 and traj_name not in anchor_fail_logged:
                                log(
                                    f"[{file_path.name}] anchor infeasible for traj='{traj_name}': "
                                    f"t_valid=[{t_valid[0]:.3e},{t_valid[-1]:.3e}], chunk_end={t_chunk[-1]:.3e}"
                                )
                                anchor_fail_logged.add(traj_name)
                            continue

                        # Fast interpolation (precompute indices/weights once)
                        log_t_chunk = np.log10(t_chunk)
                        i0, i1, w = _prepare_log_interp(log_t_valid, log_t_chunk)
                        y_new = interp_loglog_species(y_valid, i0, i1, w)

                        if y_new.shape[0] != cfg.n_steps or not np.all(np.isfinite(y_new)):
                            reject(
                                "interp_non_finite",
                                f"{traj_name}: y_new finite={bool(np.all(np.isfinite(y_new)))}, shape={y_new.shape}",
                            )
                            continue

                        # Success
                        ok = True
                        break

                    if not ok:
                        reject(
                            "failed_to_sample_chunk",
                            f"{traj_name}: failed to sample a valid chunk after {max_chunk_tries} tries",
                        )
                        # Do not permanently mark the whole traj as bad; just stop drawing from it.
                        break

                    split = _split_name(float(rng.random()), cfg.val_fraction, cfg.test_fraction)

                    y_buf[split].append(y_new.astype(np.float32, copy=False))
                    g_buf[split].append(g_vec.astype(np.float32, copy=False))
                    counts_total[split] += 1

                    if len(y_buf[split]) >= cfg.shard_size:
                        shard_idx[split], _ = flush_shard(
                            out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
                        )
                        y_buf[split].clear()
                        g_buf[split].clear()

                    written += 1
                    ccount += 1
                    per_source_traj_counts[traj_name] = ccount

                    total_written = int(counts_total["train"] + counts_total["validation"] + counts_total["test"])
                    if total_written - last_log_count >= cfg.log_every_n_trajectories:
                        log(
                            f"Progress: {total_written} trajectories written "
                            f"(train={counts_total['train']}, val={counts_total['validation']}, test={counts_total['test']})"
                        )
                        last_log_count = total_written

            # Flush leftovers
            for split in ("train", "validation", "test"):
                if y_buf[split]:
                    shard_idx[split], _ = flush_shard(
                        out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
                    )
                    y_buf[split].clear()
                    g_buf[split].clear()

            log(
                f"[{file_path.name}] sampled={written}/{target_total}, "
                f"unique_src={len(per_source_traj_counts)}, "
                f"min_per_src={(min(per_source_traj_counts.values()) if per_source_traj_counts else 0)}, "
                f"max_per_src={(max(per_source_traj_counts.values()) if per_source_traj_counts else 0)}"
            )

            total_rejects = int(sum(reject_stats.values()))
            if total_rejects > 0:
                nz = [(k, v) for k, v in reject_stats.items() if v > 0]
                nz.sort(key=lambda kv: kv[1], reverse=True)
                log(f"[{file_path.name}] Reject summary (total_rejects={total_rejects}, written={written}):")
                for k, v in nz[:12]:
                    log(f"  - {k}: {v}")
                    ex = reject_examples.get(k, [])
                    for e in ex[:REJECT_EXAMPLES_PER_REASON]:
                        log(f"      example: {e}")

            return last_log_count, reject_stats

        # ---------------- Legacy random sampling mode ----------------
        attempts = 0
        written = 0
        target = int(cfg.output_trajectories_per_file)

        while written < target and attempts < int(cfg.max_sampling_attempts_per_file):
            attempts += 1

            # Helpful when you're short of target: print rejects periodically.
            if attempts % int(cfg.log_every_n_attempts) == 0:
                top = sorted(((k, v) for k, v in reject_stats.items() if v > 0), key=lambda kv: kv[1], reverse=True)[:8]
                top_s = ", ".join([f"{k}={v}" for k, v in top]) if top else "none"
                log(f"[{file_path.name}] attempts={attempts}, written={written}, bad_traj={len(bad_traj)} | rejects: {top_s}")

            traj_name = pool[int(rng.integers(0, len(pool)))]
            if traj_name not in fin:
                continue
            grp = fin[traj_name]
            if not isinstance(grp, h5py.Group):
                reject("not_group", f"{traj_name}: root item not a group")
                continue

            ccount = per_source_traj_counts.get(traj_name, 0)
            if cfg.max_chunks_per_source_trajectory > 0 and ccount >= cfg.max_chunks_per_source_trajectory:
                # Not an error; skip without counting as reject (by design).
                continue

            td = _get_traj_data(traj_name, grp)
            if td is None:
                continue

            t_raw = td["t_raw"]
            t_lo = float(td["t_lo"])
            t_valid = td["t_valid"]
            log_t_valid = td["log_t_valid"]
            y_valid = td["y_valid"]
            g_vec = td["g_vec"]

            # Can we fit a full chunk of length (n_steps-1)*dt inside this trajectory?
            t_hi = float(t_raw[-1] - chunk_duration)
            if t_hi <= t_lo:
                reject(
                    "chunk_no_span",
                    f"{traj_name}: t_end={float(t_raw[-1]):.3e}, t_lo={t_lo:.3e}, "
                    f"chunk_dur={chunk_duration:.3e} => t_hi={t_hi:.3e} <= t_lo",
                )
                continue

            # Start time selection
            if cfg.anchor_first_chunk and ccount == 0:
                t_start = t_lo
            else:
                t_start = 10.0 ** float(rng.uniform(np.log10(t_lo), np.log10(t_hi)))

            t_chunk = t_start + chunk_offsets

            # Ensure chunk fits within valid time region
            if t_chunk[0] < t_valid[0] or t_chunk[-1] > t_valid[-1]:
                reject(
                    "chunk_out_of_bounds",
                    f"{traj_name}: t_start={t_start:.3e}, chunk=[{t_chunk[0]:.3e},{t_chunk[-1]:.3e}] "
                    f"valid=[{t_valid[0]:.3e},{t_valid[-1]:.3e}] "
                    f"(anchor={cfg.anchor_first_chunk and ccount==0})",
                )
                if cfg.anchor_first_chunk and ccount == 0 and traj_name not in anchor_fail_logged:
                    log(
                        f"[{file_path.name}] anchor infeasible for traj='{traj_name}': "
                        f"t_start={t_start:.3e}, t_valid=[{t_valid[0]:.3e},{t_valid[-1]:.3e}], "
                        f"chunk_end={t_chunk[-1]:.3e}"
                    )
                    anchor_fail_logged.add(traj_name)
                continue

            # Fast interpolation (precompute indices/weights once)
            log_t_chunk = np.log10(t_chunk)
            i0, i1, w = _prepare_log_interp(log_t_valid, log_t_chunk)
            y_new = interp_loglog_species(y_valid, i0, i1, w)

            if y_new.shape[0] != cfg.n_steps or not np.all(np.isfinite(y_new)):
                reject(
                    "interp_non_finite",
                    f"{traj_name}: y_new finite={bool(np.all(np.isfinite(y_new)))}, shape={y_new.shape}",
                )
                continue

            split = _split_name(float(rng.random()), cfg.val_fraction, cfg.test_fraction)

            y_buf[split].append(y_new.astype(np.float32, copy=False))
            g_buf[split].append(g_vec.astype(np.float32, copy=False))
            counts_total[split] += 1

            if len(y_buf[split]) >= cfg.shard_size:
                shard_idx[split], _ = flush_shard(
                    out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
                )
                y_buf[split].clear()
                g_buf[split].clear()

            written += 1
            per_source_traj_counts[traj_name] = ccount + 1

            total_written = int(counts_total["train"] + counts_total["validation"] + counts_total["test"])
            if total_written - last_log_count >= cfg.log_every_n_trajectories:
                log(
                    f"Progress: {total_written} trajectories written "
                    f"(train={counts_total['train']}, val={counts_total['validation']}, test={counts_total['test']})"
                )
                last_log_count = total_written

        # Flush leftovers
        for split in ("train", "validation", "test"):
            if y_buf[split]:
                shard_idx[split], _ = flush_shard(
                    out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
                )
                y_buf[split].clear()
                g_buf[split].clear()

        log(
            f"[{file_path.name}] sampled={written}/{target}, attempts={attempts}, "
            f"unique_src={len(per_source_traj_counts)}"
        )

        total_rejects = int(sum(reject_stats.values()))
        if total_rejects > 0:
            nz = [(k, v) for k, v in reject_stats.items() if v > 0]
            nz.sort(key=lambda kv: kv[1], reverse=True)

            log(f"[{file_path.name}] Reject summary (total_rejects={total_rejects}, attempts={attempts}, written={written}):")
            for k, v in nz[:12]:
                log(f"  - {k}: {v}")
                ex = reject_examples.get(k, [])
                for e in ex[:REJECT_EXAMPLES_PER_REASON]:
                    log(f"      example: {e}")

    return last_log_count, reject_stats
class RunningMeanVar:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.M2 = np.zeros((self.dim,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        x2 = x.reshape(-1, self.dim)
        m = int(x2.shape[0])
        if m == 0:
            return

        b_mean = np.mean(x2, axis=0)
        b_M2 = np.sum((x2 - b_mean) ** 2, axis=0)

        if self.n == 0:
            self.n = m
            self.mean = b_mean
            self.M2 = b_M2
            return

        n0 = float(self.n)
        m0 = float(m)
        delta = b_mean - self.mean
        n_new = n0 + m0
        self.mean = self.mean + delta * (m0 / n_new)
        self.M2 = self.M2 + b_M2 + (delta ** 2) * (n0 * m0 / n_new)
        self.n = int(n_new)

    def finalize(self, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            std = np.ones_like(self.mean)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, float(min_std) ** 2))
        return self.mean, std


def compute_train_stats_from_physical(out_tmp: Path, cfg: PreCfg, species_vars: List[str]) -> Tuple[Dict, Dict[str, str]]:
    eps = float(cfg.epsilon)

    methods: Dict[str, str] = {s: "log-standard" for s in species_vars}
    for g in cfg.global_variables:
        requested = str(cfg.methods.get(g, cfg.globals_default_method))
        if requested not in ("standard", "min-max", "identity", "minmax"):
            raise ValueError(f"Global '{g}' requested unsupported normalization method '{requested}'.")
        methods[g] = "min-max" if requested == "minmax" else requested
    methods["t_vec"] = "identity"

    train_shards = iter_shards(out_tmp, "train", suffix="_physical")
    shards = train_shards
    if not shards:
        # If train ended up empty due to random split with small N, compute from all physical shards.
        all_shards: List[Path] = []
        for sp in ("train", "validation", "test"):
            all_shards.extend(iter_shards(out_tmp, sp, suffix="_physical"))
        if not all_shards:
            raise RuntimeError("No physical shards found; cannot compute stats.")
        log("Warning: no TRAIN physical shards found; computing normalization stats from all splits.")
        shards = all_shards

    S = int(len(species_vars))
    G = int(len(cfg.global_variables))

    rms_s = RunningMeanVar(S)
    rms_g = RunningMeanVar(G)

    s_min = np.full((S,), np.inf, dtype=np.float64)
    s_max = np.full((S,), -np.inf, dtype=np.float64)
    g_min = np.full((G,), np.inf, dtype=np.float64)
    g_max = np.full((G,), -np.inf, dtype=np.float64)

    for p in tqdm(shards, desc="Computing normalization stats", leave=False):
        with np.load(p) as z:
            y = np.asarray(z["y_mat"], dtype=np.float64)   # [N,T,S] physical
            g = np.asarray(z["globals"], dtype=np.float64) # [N,G] physical

        y_log = np.log10(np.maximum(y, eps)).reshape(-1, S)
        rms_s.update(y_log)
        s_min = np.minimum(s_min, np.min(y_log, axis=0))
        s_max = np.maximum(s_max, np.max(y_log, axis=0))

        g2 = g.reshape(-1, G)
        rms_g.update(g2)
        g_min = np.minimum(g_min, np.min(g2, axis=0))
        g_max = np.maximum(g_max, np.max(g2, axis=0))

    mu_s, sd_s = rms_s.finalize(cfg.min_std)
    mu_g, sd_g = rms_g.finalize(cfg.min_std)

    per_key_stats: Dict = {}
    for i, s in enumerate(species_vars):
        per_key_stats[s] = {
            "log_mean": float(mu_s[i]),
            "log_std": float(sd_s[i]),
            "log_min": float(s_min[i]),
            "log_max": float(s_max[i]),
            "epsilon": eps,
        }
    for i, gv in enumerate(cfg.global_variables):
        per_key_stats[gv] = {
            "mean": float(mu_g[i]),
            "std": float(sd_g[i]),
            "min": float(g_min[i]),
            "max": float(g_max[i]),
        }
    per_key_stats["t_vec"] = {"method": "identity"}

    return per_key_stats, methods


def normalize_and_write_final(
    out_tmp: Path,
    out_final: Path,
    cfg: PreCfg,
    species_vars: List[str],
    per_key_stats: Dict,
    methods: Dict[str, str],
    t_vec_global: np.ndarray,
) -> None:
    out_final.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (out_final / split).mkdir(parents=True, exist_ok=True)

    eps = float(cfg.epsilon)
    S = int(len(species_vars))
    G = int(len(cfg.global_variables))

    mu_s = np.array([float(per_key_stats[s]["log_mean"]) for s in species_vars], dtype=np.float64).reshape(1, 1, S)
    sd_s = np.array([float(per_key_stats[s]["log_std"]) for s in species_vars], dtype=np.float64).reshape(1, 1, S)

    g_mean = np.array([float(per_key_stats[g]["mean"]) for g in cfg.global_variables], dtype=np.float64).reshape(1, G)
    g_std  = np.array([float(per_key_stats[g]["std"])  for g in cfg.global_variables], dtype=np.float64).reshape(1, G)
    g_min  = np.array([float(per_key_stats[g]["min"])  for g in cfg.global_variables], dtype=np.float64).reshape(1, G)
    g_max  = np.array([float(per_key_stats[g]["max"])  for g in cfg.global_variables], dtype=np.float64).reshape(1, G)
    g_methods = [str(methods.get(g, "standard")) for g in cfg.global_variables]

    def _normalize_globals(g: np.ndarray) -> np.ndarray:
        out = np.empty_like(g, dtype=np.float32)
        for j, m in enumerate(g_methods):
            if m == "identity":
                out[:, j] = g[:, j].astype(np.float32, copy=False)
            elif m == "standard":
                out[:, j] = ((g[:, j] - g_mean[0, j]) / g_std[0, j]).astype(np.float32, copy=False)
            elif m == "min-max":
                denom = max(float(g_max[0, j] - g_min[0, j]), 1e-12)
                out[:, j] = ((g[:, j] - g_min[0, j]) / denom).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unsupported global normalization method '{m}' for key '{cfg.global_variables[j]}'.")
        return out

    for split in ("train", "validation", "test"):
        physical_shards = iter_shards(out_tmp, split, suffix="_physical")
        if not physical_shards:
            continue

        shard_counter = 0
        y_carry: Optional[np.ndarray] = None
        g_carry: Optional[np.ndarray] = None

        for p in tqdm(physical_shards, desc=f"Normalizing {split}", leave=False):
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float64)
                g = np.asarray(z["globals"], dtype=np.float64)

            y_log = np.log10(np.maximum(y, eps))
            y_z = ((y_log - mu_s) / sd_s).astype(np.float32, copy=False)
            g_z = _normalize_globals(g)

            if y_carry is None:
                y_all = y_z
                g_all = g_z
            else:
                y_all = np.concatenate([y_carry, y_z], axis=0)
                g_all = np.concatenate([g_carry, g_z], axis=0)

            n_all = int(y_all.shape[0])
            start = 0
            while n_all - start >= cfg.shard_size:
                end = start + cfg.shard_size
                out_path = out_final / split / f"shard_{shard_counter:06d}.npz"
                np.savez(out_path, y_mat=y_all[start:end], globals=g_all[start:end], t_vec=t_vec_global)
                shard_counter += 1
                start = end

            if start < n_all:
                y_carry = y_all[start:]
                g_carry = g_all[start:]
            else:
                y_carry = None
                g_carry = None

        if y_carry is not None and y_carry.shape[0] > 0:
            out_path = out_final / split / f"shard_{shard_counter:06d}.npz"
            np.savez(out_path, y_mat=y_carry, globals=g_carry, t_vec=t_vec_global)

    log_dt = float(np.log10(cfg.dt))
    manifest = {
        "schema_version": 1,
        "species_variables": species_vars,
        "global_variables": cfg.global_variables,
        "epsilon": float(cfg.epsilon),
        "min_std": float(cfg.min_std),
        "dt": {"log_min": log_dt, "log_max": log_dt},
        "normalization_methods": methods,
        "per_key_stats": per_key_stats,
    }
    with open(out_final / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_summary(out_final: Path, cfg: PreCfg, counts_total: Dict[str, int], rejects: Dict[str, int]) -> None:
    summary = {
        "dt": cfg.dt,
        "n_steps": cfg.n_steps,
        "t_min": cfg.t_min,
        "output_trajectories_per_file": cfg.output_trajectories_per_file,
        "max_chunks_per_source_trajectory": cfg.max_chunks_per_source_trajectory,
        "anchor_first_chunk": cfg.anchor_first_chunk,
        "max_sampling_attempts_per_file": cfg.max_sampling_attempts_per_file,
        "drop_below": cfg.drop_below,
        "val_fraction": cfg.val_fraction,
        "test_fraction": cfg.test_fraction,
        "shard_size": cfg.shard_size,
        "global_variables": cfg.global_variables,
        "species_variables": cfg.species_variables,
        "counts_total": counts_total,
        "rejects_total": rejects,
        "raw_dir": str(cfg.raw_dir),
        "processed_dir": str(cfg.processed_dir),
    }
    with open(out_final / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    log(f"Loading config from: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    cfg_dict = load_json_config(CONFIG_PATH)
    cfg = load_precfg(cfg_dict)

    if cfg.output_trajectories_per_file <= 0:
        raise ValueError("preprocessing.output_trajectories_per_file must be > 0")
    if not cfg.species_variables:
        raise RuntimeError("data.species_variables is empty; must be set explicitly.")

    raw_files = sorted(cfg.raw_dir.glob("*.h5")) + sorted(cfg.raw_dir.glob("*.hdf5"))
    if not raw_files:
        raise FileNotFoundError(f"No HDF5 files found in raw_dir={cfg.raw_dir}")

    out_final = cfg.processed_dir
    out_tmp = cfg.processed_dir / "_tmp_physical"

    clean_tmp_dir(out_tmp, overwrite=cfg.overwrite)
    clean_final_outputs(out_final, overwrite=cfg.overwrite)

    chunk_offsets = np.arange(cfg.n_steps, dtype=np.float64) * cfg.dt
    t_vec_rel = chunk_offsets.copy()

    rng = np.random.default_rng(cfg.seed)

    counts_total = {"train": 0, "validation": 0, "test": 0}
    shard_idx = {"train": 0, "validation": 0, "test": 0}
    rejects_total: Dict[str, int] = {}

    log(f"Raw dir: {cfg.raw_dir}")
    log(f"Processed dir: {cfg.processed_dir}")
    log(f"Found {len(raw_files)} raw file(s)")
    log(
        f"Sampling target: {cfg.output_trajectories_per_file} trajectories per file "
        f"({cfg.n_steps} steps × dt={cfg.dt}); val_fraction={cfg.val_fraction}, test_fraction={cfg.test_fraction}"
    )
    log(
        f"Sampling controls: max_chunks_per_source_trajectory={cfg.max_chunks_per_source_trajectory}, "
        f"anchor_first_chunk={cfg.anchor_first_chunk}, max_sampling_attempts_per_file={cfg.max_sampling_attempts_per_file}"
    )
    log(f"Filtering: drop_below={cfg.drop_below}, time_keys={cfg.time_keys}")
    log("Schema resolution: recursive dataset search by leaf name; for X tries X and evolve_X")

    log("Phase 1: Sampling trajectories and writing physical shards...")
    last_log_count = 0
    for i, fp in enumerate(raw_files):
        log(f"Processing raw file {i + 1}/{len(raw_files)}: {fp.name}")
        last_log_count, rej = sample_trajectories_from_file(
            fp, out_tmp, cfg, rng, chunk_offsets, t_vec_rel, counts_total, shard_idx, last_log_count
        )
        for k, v in rej.items():
            rejects_total[k] = rejects_total.get(k, 0) + v

    total = int(counts_total["train"] + counts_total["validation"] + counts_total["test"])
    log(
        f"Sampling complete: {total} trajectories "
        f"(train={counts_total['train']}, val={counts_total['validation']}, test={counts_total['test']})"
    )

    if total == 0:
        top = sorted(rejects_total.items(), key=lambda kv: kv[1], reverse=True)
        top_s = ", ".join([f"{k}={v}" for k, v in top])
        raise RuntimeError(
            "No trajectories were sampled. Every candidate trajectory was rejected.\n"
            f"Reject summary: {top_s}\n"
            "The one-time log above prints available dataset leaf names; align data.species_variables to those."
        )

    log("Phase 2: Computing normalization statistics...")
    per_key_stats, methods = compute_train_stats_from_physical(out_tmp, cfg, cfg.species_variables)

    log("Phase 3: Normalizing and writing final shards...")
    normalize_and_write_final(out_tmp, out_final, cfg, cfg.species_variables, per_key_stats, methods, t_vec_rel)

    log("Phase 4: Writing preprocessing summary...")
    write_summary(out_final, cfg, counts_total, rejects_total)

    log("Done.")


if __name__ == "__main__":
    main()
