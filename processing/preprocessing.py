#!/usr/bin/env python3
"""
preprocessing.py

Resample adaptive-timestep HDF5 trajectories into fixed-length (n_steps) uniform-dt profiles,
shard to NPZ, compute normalization, and write final normalized shards.

Key properties:
- Vectorized core: precompute log-time interpolation indices/weights once per chunk, then log-log interpolate all species.
- Works with variable dt (dt_mode="per_chunk") and fixed dt (dt_mode="fixed").
- Outputs:
    processed_dir/
      normalization.json
      preprocessing_summary.json
      train/shard_*.npz
      validation/shard_*.npz
      test/shard_*.npz
  and uses a temp physical stage:
      _tmp_physical/<split>/shard_*_physical.npz

Schema tolerance:
- Species/global datasets can be nested under groups.
- Dataset names may optionally have an "evolve_" prefix (config can use either).
- Globals may be scalars or time-series aligned with t_raw (reduced to scalar).

Notes on "functionality stability":
- This version adds docstrings, comments, and more informative stdout logging (especially on drops/rejections).
- It does not change sampling, interpolation, sharding formats, normalization math, or file outputs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Repo/config discovery
# -----------------------------------------------------------------------------

def _default_repo_root() -> Path:
    """
    Heuristically locate the repository root.

    The function searches upward from this file's directory for either:
    - config/config.json
    - config.json

    If neither is found within a small parent range, it falls back to the directory
    containing this file.
    """
    here = Path(__file__).resolve().parent
    for p in (here, here.parent, here.parent.parent):
        if (p / "config.json").exists() or (p / "config.json").exists():
            return p
    return here


PROJECT_ROOT = _default_repo_root()


def _default_config_path(root: Path) -> Path:
    """
    Choose a default config path under the given root.

    Preference order:
    1) root/config/config.json
    2) root/config.json

    If neither exists, returns the first candidate (so the caller can fail loudly
    when attempting to open it).
    """
    cand = [root / "config" / "config.json", root / "config.json"]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


CONFIG_PATH = _default_config_path(PROJECT_ROOT)


def log(msg: str) -> None:
    """
    Print a timestamped log line to stdout.

    This is intentionally simple (no log levels, no external logger dependency) so
    behavior remains stable across environments. All new logging in this file uses
    this function to keep formatting consistent.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


def load_json_config(path: Path) -> Dict:
    """
    Load a JSON configuration file into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------

def _name_candidates(base: str) -> List[str]:
    """
    Return candidate dataset/attribute names for schema tolerance.

    If a name is provided with/without the "evolve_" prefix, we attempt both forms.
    This allows config lists to use either convention while still finding the dataset.
    """
    out = [base]
    if base.startswith("evolve_"):
        out.append(base[len("evolve_"):])
    else:
        out.append("evolve_" + base)
    seen: set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_leaf_index(grp: h5py.Group) -> Dict[str, List[str]]:
    """
    Build an index mapping leaf dataset names -> list of full paths within `grp`.

    Example:
      /foo/bar/T  and  /baz/T  both yield leaf "T" mapping to ["foo/bar/T", "baz/T"].

    This supports flexible lookup when datasets are nested under arbitrary groups.
    """
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def get_time_array_recursive(
    grp: h5py.Group, time_keys: List[str], leaf_index: Dict[str, List[str]]
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Locate and validate a monotonic increasing 1D time array under `grp`.

    Search order:
    1) Direct members of `grp` with keys in `time_keys`
    2) Any nested dataset whose leaf name matches an entry in `time_keys`

    Returns:
      (path_or_key_used, time_array) if found and valid; otherwise (None, None).

    Validation:
      - at least 2 samples
      - finite values
      - strictly increasing (diff > 0)
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
    Find a dataset path under `grp` matching `base` (with evolve_ tolerance).

    If `t_len` is provided and `prefer_time_aligned` is True, the function prefers
    datasets whose first dimension equals `t_len` (time-aligned series). Otherwise,
    it returns the first match by leaf name.
    """
    for cand in _name_candidates(base):
        paths = leaf_index.get(cand, [])
        if not paths:
            continue
        if t_len is not None and prefer_time_aligned:
            for p in paths:
                try:
                    ds = grp[p]
                    if isinstance(ds, h5py.Dataset) and ds.shape and int(ds.shape[0]) == int(t_len):
                        return p
                except Exception:
                    continue
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
    Read a "global" variable from an HDF5 trajectory group with schema tolerance.

    Supported locations/types (first match wins):
    1) Attribute on the trajectory group (`grp.attrs`)
    2) Attribute on the file (`grp.file.attrs`)
    3) Scalar dataset under the trajectory group (`grp[...]`)
    4) Scalar dataset at the file root (`grp.file[...]`)
    5) Any dataset whose leaf name matches (possibly nested), including time-series

    For time-series globals aligned with the trajectory time axis:
      - If nearly constant over time, returns the mean.
      - Otherwise logs a warning (once per dataset path per file) and returns the first value.

    Raises:
      KeyError if the global cannot be located or is invalid/unexpected in shape.
    """
    # attrs (group then file)
    for cand in _name_candidates(key_base):
        if cand in grp.attrs:
            v = np.asarray(grp.attrs[cand]).reshape(-1)
            if v.size == 1 and np.isfinite(v[0]):
                return float(v[0])
        if cand in grp.file.attrs:
            v = np.asarray(grp.file.attrs[cand]).reshape(-1)
            if v.size == 1 and np.isfinite(v[0]):
                return float(v[0])

    # scalar datasets (group then file root)
    for cand in _name_candidates(key_base):
        if cand in grp and isinstance(grp[cand], h5py.Dataset):
            arr = np.asarray(grp[cand][...], dtype=np.float64).reshape(-1)
            if arr.size == 1 and np.isfinite(arr[0]):
                return float(arr[0])
        if cand in grp.file and isinstance(grp.file[cand], h5py.Dataset):
            arr = np.asarray(grp.file[cand][...], dtype=np.float64).reshape(-1)
            if arr.size == 1 and np.isfinite(arr[0]):
                return float(arr[0])

    # any dataset leaf match
    p = find_dataset_path(grp, key_base, leaf_index, t_len=t_len, prefer_time_aligned=False)
    if p is None:
        raise KeyError(f"Missing/invalid global '{key_base}' (tried {_name_candidates(key_base)})")

    ds = grp[p]
    arr = np.asarray(ds[...], dtype=np.float64)
    flat = arr.reshape(-1)

    if flat.size == 1 and np.isfinite(flat[0]):
        return float(flat[0])

    # Time-aligned series: accept and reduce to a scalar
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

    raise KeyError(f"Global '{key_base}' found at '{p}' but not scalar or time-aligned.")


# -----------------------------------------------------------------------------
# Interpolation core (vectorized)
# -----------------------------------------------------------------------------

def _prepare_log_interp(
    log_t_valid: np.ndarray, log_t_chunk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute index pairs and interpolation weights for log-time interpolation.

    Inputs:
      log_t_valid: log10(time) for all "valid" source points, strictly increasing.
      log_t_chunk: log10(time) for target points (chunk grid).

    Outputs:
      i0, i1: int64 indices into log_t_valid for left/right bracketing points
      w:      float64 weights in [0, 1] for linear interpolation in log-time

    This is computed once per chunk and reused across all species columns.
    """
    n = int(log_t_valid.shape[0])
    idx = np.searchsorted(log_t_valid, log_t_chunk, side="left")
    idx = np.clip(idx, 1, n - 1)
    i0 = (idx - 1).astype(np.int64)
    i1 = idx.astype(np.int64)
    t0 = log_t_valid[i0]
    t1 = log_t_valid[i1]
    denom = np.maximum(t1 - t0, 1e-300)
    w = (log_t_chunk - t0) / denom
    return i0, i1, np.clip(w, 0.0, 1.0).astype(np.float64)


def interp_loglog_species(
    y_valid: np.ndarray, i0: np.ndarray, i1: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Log-log interpolate species values.

    This performs:
      y_new(t) = 10 ** lerp(log10(y_valid), w)

    where lerp happens between bracketing indices i0 and i1 and w is in [0,1].

    Notes:
    - Values are clipped to a tiny positive floor to avoid log10(0).
    - Interpolation is vectorized across species columns for performance.
    """
    y0 = np.log10(np.maximum(y_valid[i0, :], 1e-300))
    y1 = np.log10(np.maximum(y_valid[i1, :], 1e-300))
    w2 = w.reshape(-1, 1)
    return 10.0 ** ((1.0 - w2) * y0 + w2 * y1)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class PreCfg:
    """
    Strongly-typed preprocessing configuration.

    This is a direct, minimally-transformed view of config.json values used by this script.
    The fields are intentionally explicit to make it obvious which knobs affect output.

    Important:
    - Modifying defaults here is a functional change. This file only adds comments/logging.
    """
    raw_dir: Path
    processed_dir: Path
    raw_file_patterns: List[str]

    dt: float
    dt_mode: str               # "fixed" | "per_chunk"
    dt_min: float
    dt_max: float
    dt_sampling: str           # "uniform" | "loguniform"

    n_steps: int
    t_min: float

    output_trajectories_per_file: int
    shard_size: int
    overwrite: bool

    time_keys: List[str]
    time_key: Optional[str]
    species_group: Optional[str]
    globals_group: Optional[str]

    val_fraction: float
    test_fraction: float

    global_variables: List[str]
    species_variables: List[str]

    epsilon: float
    min_std: float
    methods: Dict[str, str]
    default_method: str
    globals_default_method: str

    seed: int
    pool_size: int
    samples_per_source_trajectory: int
    max_chunk_attempts_per_source: int
    drop_below: float


def load_precfg(cfg: Dict) -> PreCfg:
    """
    Parse a raw config dict (from JSON) into a PreCfg instance.

    This function performs only:
    - path resolution relative to PROJECT_ROOT
    - minor backward-compat mapping for dt_mode and split fractions
    - defaulting fields that may be missing

    It does not touch sampling/interpolation logic and is intended to keep
    backward compatibility with older config schema variants.
    """
    pcfg = cfg.get("paths", {}) or {}
    raw_dir = Path(pcfg.get("raw_data_dir", PROJECT_ROOT / "data" / "raw"))
    processed_dir = Path(pcfg.get("processed_data_dir", PROJECT_ROOT / "data" / "processed"))
    if not raw_dir.is_absolute():
        raw_dir = (PROJECT_ROOT / raw_dir).resolve()
    if not processed_dir.is_absolute():
        processed_dir = (PROJECT_ROOT / processed_dir).resolve()

    pr = cfg.get("preprocessing", {}) or {}
    dcfg = cfg.get("data", {}) or {}
    ncfg = cfg.get("normalization", {}) or {}

    dt = float(pr.get("dt", 100.0))
    dt_mode = str(pr.get("dt_mode", "fixed")).lower().strip()
    dt_min = float(pr.get("dt_min", dt))
    dt_max = float(pr.get("dt_max", dt))

    # Normalize dt_mode + validate dt range.
    if dt_mode in ("fixed", "constant"):
        dt_mode = "fixed"
        dt_min = dt
        dt_max = dt
    elif dt_mode in ("per_chunk", "chunk", "variable"):
        dt_mode = "per_chunk"
        dt_min = float(pr.get("dt_min", dt_min))
        dt_max = float(pr.get("dt_max", dt_max))
    else:
        raise ValueError(f"Unsupported dt_mode='{dt_mode}'.")

    if dt_min <= 0 or dt_max <= 0 or dt_max < dt_min:
        raise ValueError(f"Invalid dt range: dt_min={dt_min}, dt_max={dt_max}.")

    dt_sampling_raw = str(pr.get("dt_sampling", "loguniform")).lower().strip()
    dt_sampling = "loguniform" if "log" in dt_sampling_raw else "uniform"

    # Keep backward compatibility with both naming conventions.
    val_fraction = float(pr.get("val_fraction", pr.get("val_split", 0.1)))
    test_fraction = float(pr.get("test_fraction", pr.get("test_split", 0.1)))

    time_key = pr.get("time_key")
    time_keys = list(pr.get("time_keys", ["t_time", "time", "t"]))
    if time_key:
        time_keys = [str(time_key)] + [str(k) for k in time_keys if str(k) != str(time_key)]

    raw_pat = pr.get("raw_file_pattern", pr.get("raw_file_patterns", ["*.h5", "*.hdf5"]))
    raw_file_patterns = [raw_pat] if isinstance(raw_pat, str) else [str(x) for x in raw_pat]

    return PreCfg(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_file_patterns=raw_file_patterns,
        dt=dt,
        dt_mode=dt_mode,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_sampling=dt_sampling,
        n_steps=int(pr.get("n_steps", 1000)),
        t_min=float(pr.get("t_min", 1e-3)),
        drop_below=float(pr.get("drop_below", 1e-35)),
        output_trajectories_per_file=int(pr.get("output_trajectories_per_file", 100)),
        shard_size=int(pr.get("shard_size", 1024)),
        overwrite=bool(pr.get("overwrite", True)),
        time_keys=time_keys,
        time_key=str(time_key) if time_key else None,
        species_group=pr.get("species_group"),
        globals_group=pr.get("globals_group"),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        global_variables=list(dcfg.get("global_variables", ["P", "T"])),
        species_variables=list(dcfg.get("species_variables", [])),
        epsilon=float(ncfg.get("epsilon", 1e-30)),
        min_std=float(ncfg.get("min_std", 1e-12)),
        methods=dict(ncfg.get("methods", {})),
        default_method=str(ncfg.get("default_method", "log-standard")),
        globals_default_method=str(ncfg.get("globals_default_method", "standard")),
        seed=int(pr.get("seed", cfg.get("system", {}).get("seed", 1234))),
        pool_size=int(pr.get("pool_size", 1_000_000)),
        samples_per_source_trajectory=int(pr.get("samples_per_source_trajectory", 1)) or 1,
        max_chunk_attempts_per_source=int(pr.get("max_chunk_attempts_per_source", 200)),
    )


# -----------------------------------------------------------------------------
# File utilities
# -----------------------------------------------------------------------------

def list_raw_files(raw_dir: Path, patterns: List[str]) -> List[Path]:
    """
    List raw HDF5 files under `raw_dir` that match any pattern in `patterns`.
    """
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(raw_dir.glob(pat)))
    return sorted({p.resolve() for p in files})


def clean_tmp_dir(tmp_dir: Path, overwrite: bool) -> None:
    """
    Prepare the temporary physical output directory.

    If `overwrite` is False and the directory exists, this raises to avoid
    silently mixing outputs from multiple runs.
    """
    if tmp_dir.exists():
        if not overwrite:
            raise RuntimeError(f"Temp dir exists: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def clean_final_outputs(processed_dir: Path, overwrite: bool) -> None:
    """
    Remove existing split directories under the final processed output directory.

    This mirrors the original behavior:
    - If split dir exists and overwrite is False -> raise
    - Otherwise remove split dirs so the run is clean/reproducible
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        d = processed_dir / split
        if d.exists():
            if not overwrite:
                raise RuntimeError(f"Output split dir exists: {d}")
            shutil.rmtree(d)


def flush_shard(
    out_dir: Path,
    split: str,
    shard_idx: int,
    y_buf: List[np.ndarray],
    g_buf: List[np.ndarray],
    dt_buf: List[np.ndarray],
    suffix: str,
) -> Tuple[int, int]:
    """
    Write a shard from in-memory buffers to disk.

    Parameters:
      out_dir:   root directory for this stage (temp physical or final processed)
      split:     "train" | "validation" | "test"
      shard_idx: current shard index counter for the split
      y_buf:     list of [T,S] arrays
      g_buf:     list of [G] arrays
      dt_buf:    list of [T-1] arrays
      suffix:    filename suffix; "_physical" indicates the temp stage

    Returns:
      (new_shard_idx, num_samples_written)

    Note:
      This function preserves the original NPZ key names and shapes. The only
      additions in this file are comments and logging elsewhere.
    """
    if not y_buf:
        return shard_idx, 0
    out_dir_split = out_dir / split
    out_dir_split.mkdir(parents=True, exist_ok=True)

    y_mat = np.stack(y_buf, axis=0)
    g_mat = np.stack(g_buf, axis=0)
    dt_mat = np.stack(dt_buf, axis=0)

    shard_path = out_dir_split / f"shard_{shard_idx:06d}{suffix}.npz"
    if suffix.endswith("_physical"):
        np.savez(shard_path, y_mat=y_mat, globals=g_mat, dt_mat=dt_mat)
    else:
        np.savez(shard_path, y_mat=y_mat, globals=g_mat, dt_norm_mat=dt_mat)
    return shard_idx + 1, int(y_mat.shape[0])


def iter_shards(root: Path, split: str, suffix: str) -> List[Path]:
    """
    List shard files for a given split and suffix.
    """
    d = root / split
    return sorted(d.glob(f"shard_*{suffix}.npz")) if d.exists() else []


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def reservoir_sample(keys_iter, k: int, rng: np.random.Generator) -> List[str]:
    """
    Reservoir sample k keys from an iterator of unknown length.

    This yields a uniform sample without reading all keys into memory first.
    """
    pool: List[str] = []
    for i, key in enumerate(keys_iter):
        if i < k:
            pool.append(key)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                pool[j] = key
    return pool


def _split_for_trajectory(file_name: str, traj_name: str, seed: int, val_frac: float, test_frac: float) -> str:
    """
    Deterministically assign a (file, trajectory) pair to train/val/test.

    This is stable across runs given the same seed and names. It does not depend
    on traversal order.
    """
    key = f"{file_name}:{traj_name}"
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    u = (int.from_bytes(h[:8], "big") + 0.5) / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "validation"
    return "train"


def _sample_dt(cfg: PreCfg, rng: np.random.Generator) -> float:
    """
    Sample dt according to configuration.

    - fixed: always cfg.dt
    - per_chunk: sample from [dt_min, dt_max] using uniform or log-uniform
    """
    if cfg.dt_mode == "fixed":
        return float(cfg.dt)
    if cfg.dt_sampling == "loguniform":
        lo, hi = np.log10(cfg.dt_min), np.log10(cfg.dt_max)
        return float(10.0 ** rng.uniform(lo, hi))
    return float(rng.uniform(cfg.dt_min, cfg.dt_max))


def _pick_t_start(
    t_raw: np.ndarray,
    t_valid: np.ndarray,
    *,
    dt_s: float,
    n_steps: int,
    t_min: float,
    rng: np.random.Generator,
    anchor_first: bool,
    max_attempts: int,
) -> Optional[float]:
    """
    Choose a chunk start time t_start such that the resampled chunk fits in t_valid.

    Semantics intentionally match testing.py:
    - Work only with positive times (t_raw > 0)
    - Enforce a minimum start >= max(t_min, first_positive_time)
    - Sample t_start log-uniformly between feasible bounds

    If anchor_first is True, attempt t_start = t_lo first (deterministic first sample),
    then fall back to random sampling.
    """
    pos = t_raw > 0
    if not np.any(pos):
        return None
    first_pos_time = float(t_raw[pos][0])
    t_lo = max(float(t_min), first_pos_time)

    chunk_offsets = np.arange(n_steps, dtype=np.float64) * float(dt_s)
    chunk_duration = float(chunk_offsets[-1])

    t_end = float(t_raw[-1])
    t_hi = t_end - chunk_duration
    if t_hi <= t_lo:
        return None

    def fits(start: float) -> bool:
        t_chunk = start + chunk_offsets
        return (t_chunk[0] >= t_valid[0]) and (t_chunk[-1] <= t_valid[-1])

    if anchor_first:
        t0 = t_lo
        if fits(t0):
            return t0

    lo, hi = np.log10(t_lo), np.log10(t_hi)
    for _ in range(max_attempts):
        cand = float(10.0 ** rng.uniform(lo, hi))
        if fits(cand):
            return cand

    return None


def sample_trajectories_from_file(
    file_path: Path,
    out_tmp: Path,
    cfg: PreCfg,
    rng: np.random.Generator,
    counts_total: Dict[str, int],
    shard_idx: Dict[str, int],
) -> Dict[str, int]:
    """
    Sample (and resample) trajectories from a single HDF5 file and write "physical" shards.

    Physical shard contents (per sample):
      - y_mat:    [n_steps, n_species] in physical units (float32), post log-log interpolation
      - globals:  [n_globals] float32
      - dt_mat:   [n_steps-1] dt for the chunk (float32), constant per sample

    Rejection accounting:
      This function tracks drop reasons in `rejects` and returns them so the caller can
      aggregate across files. It also logs a per-file summary of what was dropped.
    """
    y_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    g_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    dt_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}

    rejects = {
        "not_group": 0,
        "no_time": 0,
        "too_few_valid": 0,
        "missing_species": 0,
        "missing_globals": 0,
        "drop_below": 0,
        "non_finite": 0,
        "no_fit_chunk": 0,
        "interp_non_finite": 0,
    }

    # Lightweight, capped examples for more informative logging (does not affect behavior).
    example_cap = 5
    examples: Dict[str, List[str]] = {k: [] for k in rejects.keys()}

    # Per-file counters by split, for informative logging.
    written_by_split = {"train": 0, "validation": 0, "test": 0}

    target = int(cfg.output_trajectories_per_file)
    written = 0

    with h5py.File(file_path, "r") as fin:
        # Reservoir sample trajectory groups from the file to avoid loading all keys.
        pool = reservoir_sample(fin.keys(), int(cfg.pool_size), rng)
        rng.shuffle(pool)

        for traj_name in pool:
            if written >= target:
                break

            grp = fin[traj_name]
            if not isinstance(grp, h5py.Group):
                rejects["not_group"] += 1
                if len(examples["not_group"]) < example_cap:
                    examples["not_group"].append(str(traj_name))
                continue

            leaf_index = build_leaf_index(grp)
            time_path, t_raw = get_time_array_recursive(grp, cfg.time_keys, leaf_index)
            if t_raw is None:
                rejects["no_time"] += 1
                if len(examples["no_time"]) < example_cap:
                    examples["no_time"].append(str(traj_name))
                continue

            # Validity mask (match testing.py)
            # - Only consider positive times
            # - Enforce that "valid" times are not too close to t=0 by using t_lo * 0.5 cutoff
            pos = t_raw > 0
            if not np.any(pos):
                rejects["too_few_valid"] += 1
                if len(examples["too_few_valid"]) < example_cap:
                    examples["too_few_valid"].append(f"{traj_name} (no positive times)")
                continue
            first_pos_time = float(t_raw[pos][0])
            t_lo = max(cfg.t_min, first_pos_time)
            valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
            if np.count_nonzero(valid) < 2:
                rejects["too_few_valid"] += 1
                if len(examples["too_few_valid"]) < example_cap:
                    examples["too_few_valid"].append(f"{traj_name} (valid<2)")
                continue

            t_valid = t_raw[valid]
            log_t_valid = np.log10(t_valid)

            # Resolve species dataset paths for time-aligned series.
            T = int(t_raw.shape[0])
            species_paths: List[str] = []
            missing_species_var: Optional[str] = None
            for s in cfg.species_variables:
                p = find_dataset_path(grp, s, leaf_index, t_len=T, prefer_time_aligned=True)
                if p is None:
                    species_paths = []
                    missing_species_var = s
                    break
                species_paths.append(p)
            if not species_paths:
                rejects["missing_species"] += 1
                if len(examples["missing_species"]) < example_cap:
                    if missing_species_var is None:
                        examples["missing_species"].append(str(traj_name))
                    else:
                        examples["missing_species"].append(f"{traj_name} (missing '{missing_species_var}')")
                continue

            # Read species into y_raw: shape [T,S]
            # Avoid intermediate stacks: read each species dataset into a column.
            S = len(species_paths)
            y_raw = np.empty((T, S), dtype=np.float64)
            ok = True
            for j, p in enumerate(species_paths):
                try:
                    arr = np.asarray(grp[p][...], dtype=np.float64)
                except Exception:
                    ok = False
                    break
                if arr.ndim == 0 or int(arr.shape[0]) != T:
                    ok = False
                    break
                col = arr.reshape(T, -1)[:, 0]
                if not np.all(np.isfinite(col)):
                    ok = False
                    break
                y_raw[:, j] = col
            if not ok:
                rejects["non_finite"] += 1
                if len(examples["non_finite"]) < example_cap:
                    examples["non_finite"].append(str(traj_name))
                continue

            # Drop trajectories with any species value below configured floor.
            # This matches prior semantics exactly (a single value triggers drop).
            if np.any(y_raw < cfg.drop_below):
                rejects["drop_below"] += 1
                if len(examples["drop_below"]) < example_cap:
                    examples["drop_below"].append(str(traj_name))
                continue

            y_valid = y_raw[valid, :]
            if np.any(~np.isfinite(y_valid)):
                rejects["non_finite"] += 1
                if len(examples["non_finite"]) < example_cap:
                    examples["non_finite"].append(f"{traj_name} (y_valid)")
                continue

            # Globals: read per-trajectory scalars (with tolerance for nested datasets).
            try:
                warn_state: Dict[str, bool] = {}
                g_vec = np.array(
                    [read_global_flexible_recursive(grp, gv, T, leaf_index, warn_state) for gv in cfg.global_variables],
                    dtype=np.float32,
                )
            except Exception as e:
                rejects["missing_globals"] += 1
                if len(examples["missing_globals"]) < example_cap:
                    # Keep messages short; avoid dumping large exception reprs.
                    examples["missing_globals"].append(f"{traj_name} ({type(e).__name__}: {str(e)[:120]})")
                continue

            split = _split_for_trajectory(file_path.name, traj_name, cfg.seed, cfg.val_fraction, cfg.test_fraction)

            # Potentially multiple samples per source trajectory.
            # Note: RNG consumption and semantics remain unchanged; only comments/logging were added.
            for sidx in range(int(cfg.samples_per_source_trajectory)):
                if written >= target:
                    break

                dt_s = _sample_dt(cfg, rng)
                t_start = _pick_t_start(
                    t_raw,
                    t_valid,
                    dt_s=dt_s,
                    n_steps=cfg.n_steps,
                    t_min=cfg.t_min,
                    rng=rng,
                    anchor_first=(sidx == 0),
                    max_attempts=int(cfg.max_chunk_attempts_per_source),
                )
                if t_start is None:
                    rejects["no_fit_chunk"] += 1
                    if len(examples["no_fit_chunk"]) < example_cap:
                        examples["no_fit_chunk"].append(str(traj_name))
                    continue

                t_chunk = t_start + np.arange(cfg.n_steps, dtype=np.float64) * dt_s
                log_t_chunk = np.log10(t_chunk)

                # Precompute interpolation indices/weights in log-time once per chunk.
                i0, i1, w = _prepare_log_interp(log_t_valid, log_t_chunk)
                y_new = interp_loglog_species(y_valid, i0, i1, w)
                if not np.all(np.isfinite(y_new)):
                    rejects["interp_non_finite"] += 1
                    if len(examples["interp_non_finite"]) < example_cap:
                        examples["interp_non_finite"].append(str(traj_name))
                    continue

                # Buffer the sample for shard flushing.
                y_buf[split].append(y_new.astype(np.float32, copy=False))
                g_buf[split].append(g_vec.astype(np.float32, copy=False))
                dt_buf[split].append(np.full(cfg.n_steps - 1, dt_s, dtype=np.float32))

                counts_total[split] += 1
                written_by_split[split] += 1
                written += 1

                # Flush shard when buffer reaches shard_size.
                if len(y_buf[split]) >= cfg.shard_size:
                    shard_idx[split], _ = flush_shard(
                        out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], dt_buf[split], "_physical"
                    )
                    y_buf[split].clear()
                    g_buf[split].clear()
                    dt_buf[split].clear()

        # Flush remainders (possibly empty; flush_shard handles that).
        for sp in ("train", "validation", "test"):
            shard_idx[sp], _ = flush_shard(out_tmp, sp, shard_idx[sp], y_buf[sp], g_buf[sp], dt_buf[sp], "_physical")

    # Per-file informative logging about outputs and drops.
    # This is stdout-only and does not change any artifacts or control flow.
    total_rejects = int(sum(rejects.values()))
    log(
        f"[FILE SUMMARY] {file_path.name} | time_key_used={time_path!s} | "
        f"written={written} (train={written_by_split['train']}, val={written_by_split['validation']}, test={written_by_split['test']}) | "
        f"rejected={total_rejects}"
    )
    if total_rejects > 0:
        # Order by descending count for readability.
        ranked = sorted(rejects.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ", ".join([f"{k}={v}" for k, v in ranked if v > 0])
        log(f"[DROPS] {file_path.name} | {top}")
        # Include brief examples for the most common reasons (capped) to aid debugging.
        top_reasons = [k for k, v in ranked if v > 0][:3]
        for k in top_reasons:
            ex = examples.get(k) or []
            if ex:
                log(f"[DROPS:EXAMPLES] {file_path.name} | {k}: " + "; ".join(ex))

        # Explicitly surface key thresholds in drop-related logs.
        if rejects.get("drop_below", 0) > 0:
            log(f"[DROP THRESHOLD] {file_path.name} | drop_below={cfg.drop_below:g}")

    return rejects


# -----------------------------------------------------------------------------
# Normalization (streaming)
# -----------------------------------------------------------------------------

class RunningMeanVar:
    """
    Streaming mean/variance accumulator (per-dimension) using a batch-wise Welford-style merge.

    This supports large datasets without holding all samples in memory.

    Stored state:
      - n:    number of samples seen
      - mean: running mean
      - M2:   running sum of squared deviations from the mean
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a batch of samples.

        Input:
          x: array that can be reshaped to [-1, dim]
        """
        x2 = x.reshape(-1, self.dim).astype(np.float64, copy=False)
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

        delta = b_mean - self.mean
        n_new = self.n + m
        self.mean = self.mean + delta * (m / n_new)
        self.M2 = self.M2 + b_M2 + (delta**2) * (self.n * m / n_new)
        self.n = n_new

    def finalize(self, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finalize mean and std (with a minimum std floor).
        """
        if self.n <= 1:
            return self.mean, np.ones_like(self.mean)
        var = self.M2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, float(min_std) ** 2))
        return self.mean, std


def _canonical_method(s: str) -> str:
    """
    Canonicalize normalization method aliases into a small supported set.
    """
    m = str(s).lower().strip()
    if m in ("minmax", "min_max", "min-max"):
        return "min-max"
    if m in ("logminmax", "log-minmax", "log_min_max", "log-min-max"):
        return "log-min-max"
    if m in ("log10-standard", "log10_standard"):
        return "log-standard"
    if m in ("none", ""):
        return "identity"
    return m


def compute_train_stats_from_physical(
    out_tmp: Path, cfg: PreCfg, species_vars: List[str]
) -> Tuple[Dict, Dict[str, str]]:
    """
    Compute normalization statistics from physical shards.

    Species:
      - Compute stats on log10(y) with epsilon floor, returning mean/std/min/max in log-space.

    Globals:
      - Compute stats in raw space (mean/std/min/max) and log-space (log_min/log_max) to
        support multiple normalization methods (standard, min-max, log-min-max).

    Data source:
      - Prefer train shards. If no train shards exist, fall back to validation shards.
      - Raises if neither exists (i.e., nothing to normalize).

    Returns:
      per_key_stats: dict keyed by variable name
      methods:       dict keyed by variable name, with canonical method strings
    """
    eps = float(cfg.epsilon)
    S = len(species_vars)
    G = len(cfg.global_variables)

    # Method map for manifest
    methods: Dict[str, str] = {}
    for s in species_vars:
        methods[s] = "log-standard"
    for g in cfg.global_variables:
        methods[g] = _canonical_method(cfg.methods.get(g, cfg.globals_default_method))

    # Prefer train; fallback to validation if train empty
    shards = iter_shards(out_tmp, "train", "_physical")
    if not shards:
        raise RuntimeError("No physical shards found (train/validation) to compute normalization.")

    rms_logy = RunningMeanVar(S)
    logy_min = np.full(S, np.inf, dtype=np.float64)
    logy_max = np.full(S, -np.inf, dtype=np.float64)

    rms_g = RunningMeanVar(G)
    g_min = np.full(G, np.inf, dtype=np.float64)
    g_max = np.full(G, -np.inf, dtype=np.float64)
    glog_min = np.full(G, np.inf, dtype=np.float64)
    glog_max = np.full(G, -np.inf, dtype=np.float64)

    for p in tqdm(shards, desc="Computing stats"):
        with np.load(p) as z:
            y = np.asarray(z["y_mat"], dtype=np.float64)   # [N,T,S]
            g = np.asarray(z["globals"], dtype=np.float64) # [N,G]

        ylog = np.log10(np.maximum(y, eps))
        y2 = ylog.reshape(-1, S)
        rms_logy.update(y2)
        logy_min = np.minimum(logy_min, np.min(y2, axis=0))
        logy_max = np.maximum(logy_max, np.max(y2, axis=0))

        g2 = g.reshape(-1, G)
        rms_g.update(g2)
        g_min = np.minimum(g_min, np.min(g2, axis=0))
        g_max = np.maximum(g_max, np.max(g2, axis=0))

        glog = np.log10(np.maximum(g2, eps))
        glog_min = np.minimum(glog_min, np.min(glog, axis=0))
        glog_max = np.maximum(glog_max, np.max(glog, axis=0))

    mu_s, sd_s = rms_logy.finalize(cfg.min_std)
    mu_g, sd_g = rms_g.finalize(cfg.min_std)

    per_key_stats: Dict[str, Dict] = {}
    for i, s in enumerate(species_vars):
        per_key_stats[s] = {
            "log_mean": float(mu_s[i]),
            "log_std": float(sd_s[i]),
            "log_min": float(logy_min[i]),
            "log_max": float(logy_max[i]),
            "epsilon": eps,
        }

    for i, gv in enumerate(cfg.global_variables):
        per_key_stats[gv] = {
            "mean": float(mu_g[i]),
            "std": float(sd_g[i]),
            "min": float(g_min[i]),
            "max": float(g_max[i]),
            "log_min": float(glog_min[i]),
            "log_max": float(glog_max[i]),
        }

    return per_key_stats, methods


def normalize_and_write_final(
    out_tmp: Path,
    out_final: Path,
    cfg: PreCfg,
    species_vars: List[str],
    per_key_stats: Dict,
    methods: Dict[str, str],
) -> None:
    """
    Normalize physical shards and write final shards, plus normalization.json.

    Species normalization:
      y_z = (log10(max(y, eps)) - mu_log) / sd_log

    Globals normalization:
      Vectorized per-variable methods, each in:
        - identity
        - standard
        - min-max
        - log-min-max

    dt normalization:
      dt_norm = clip((log10(dt) - log10(dt_min)) / (log10(dt_max) - log10(dt_min)), 0, 1)

    Important:
      This function preserves original output key names and directory layout.
    """
    eps = float(cfg.epsilon)
    S = len(species_vars)
    G = len(cfg.global_variables)

    out_final.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (out_final / split).mkdir(parents=True, exist_ok=True)

    mu_s = np.array([per_key_stats[s]["log_mean"] for s in species_vars], dtype=np.float64).reshape(1, 1, S)
    sd_s = np.array([per_key_stats[s]["log_std"] for s in species_vars], dtype=np.float64).reshape(1, 1, S)

    # Vectorized globals normalization (no per-sample loops)
    method_ids = np.zeros(G, dtype=np.int64)  # 0=identity, 1=standard, 2=min-max, 3=log-min-max
    g_mean = np.zeros(G, dtype=np.float64)
    g_inv_std = np.ones(G, dtype=np.float64)
    g_min = np.zeros(G, dtype=np.float64)
    g_inv_rng = np.ones(G, dtype=np.float64)
    glog_min = np.zeros(G, dtype=np.float64)
    glog_inv_rng = np.ones(G, dtype=np.float64)

    for j, gv in enumerate(cfg.global_variables):
        m = _canonical_method(methods[gv])
        st = per_key_stats[gv]
        if m == "identity":
            method_ids[j] = 0
        elif m == "standard":
            method_ids[j] = 1
            g_mean[j] = float(st["mean"])
            g_inv_std[j] = 1.0 / max(float(st["std"]), cfg.min_std)
        elif m == "min-max":
            method_ids[j] = 2
            g_min[j] = float(st["min"])
            rng = max(float(st["max"]) - g_min[j], 1e-12)
            g_inv_rng[j] = 1.0 / rng
        elif m == "log-min-max":
            method_ids[j] = 3
            glog_min[j] = float(st["log_min"])
            rng = max(float(st["log_max"]) - glog_min[j], 1e-12)
            glog_inv_rng[j] = 1.0 / rng
        else:
            raise ValueError(f"Unsupported global normalization method '{m}' for '{gv}'.")

    mid = method_ids.reshape(1, G)
    g_mean = g_mean.reshape(1, G)
    g_inv_std = g_inv_std.reshape(1, G)
    g_min = g_min.reshape(1, G)
    g_inv_rng = g_inv_rng.reshape(1, G)
    glog_min = glog_min.reshape(1, G)
    glog_inv_rng = glog_inv_rng.reshape(1, G)

    dt_log_min = float(np.log10(cfg.dt_min))
    dt_log_max = float(np.log10(cfg.dt_max))
    dt_log_rng = max(dt_log_max - dt_log_min, 1e-12)

    for split in ("train", "validation", "test"):
        physical_shards = iter_shards(out_tmp, split, "_physical")
        for i, p in enumerate(tqdm(physical_shards, desc=f"Normalizing {split}")):
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float64)      # [N,T,S]
                g = np.asarray(z["globals"], dtype=np.float64)    # [N,G]
                dt = np.asarray(z["dt_mat"], dtype=np.float64)    # [N,T-1]

            y_z = ((np.log10(np.maximum(y, eps)) - mu_s) / sd_s).astype(np.float32)

            # globals
            out_g = g.astype(np.float64, copy=False)
            g_std = (out_g - g_mean) * g_inv_std
            g_mm = (out_g - g_min) * g_inv_rng
            g_lmm = (np.log10(np.maximum(out_g, eps)) - glog_min) * glog_inv_rng
            g_z = np.where(mid == 1, g_std, out_g)
            g_z = np.where(mid == 2, g_mm, g_z)
            g_z = np.where(mid == 3, g_lmm, g_z)
            g_z = g_z.astype(np.float32, copy=False)

            # dt: log10 + min-max to [0,1]
            dt_norm = (np.log10(np.maximum(dt, eps)) - dt_log_min) / dt_log_rng
            dt_norm = np.clip(dt_norm, 0.0, 1.0).astype(np.float32)

            np.savez(out_final / split / f"shard_{i:06d}.npz", y_mat=y_z, globals=g_z, dt_norm_mat=dt_norm)

    manifest = {
        "normalization_methods": methods,
        "methods": methods,  # legacy alias
        "per_key_stats": per_key_stats,
        "epsilon": eps,
        "min_std": float(cfg.min_std),
        "dt": {"log_min": dt_log_min, "log_max": dt_log_max},
        "species_variables": species_vars,
        "global_variables": cfg.global_variables,
    }
    with open(out_final / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_summary(out_final: Path, cfg: PreCfg, counts_total: Dict[str, int], rejects_total: Dict[str, int]) -> None:
    """
    Write preprocessing_summary.json.

    Note:
      The schema is preserved exactly from the original code: adding fields here would
      be an output change. This function therefore remains structurally unchanged aside
      from this docstring.
    """
    summary = {
        "counts_total": counts_total,
        "rejects_total": rejects_total,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
    }
    with open(out_final / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end preprocessing pipeline.

    Steps:
      1) Discover raw HDF5 files.
      2) Sample and resample trajectories into temporary "physical" shards.
      3) Compute normalization stats from (train or validation) physical shards.
      4) Normalize all splits and write final shards + normalization.json.
      5) Write preprocessing_summary.json aggregating counts and drop reasons.

    Additional logging:
      - Per-file written counts and drop breakdown.
      - Final run totals and top drop reasons.
    """
    cfg = load_precfg(load_json_config(CONFIG_PATH))
    raw_files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
    if not raw_files:
        raise FileNotFoundError(f"No raw HDF5 files found under {cfg.raw_dir} for patterns {cfg.raw_file_patterns}")

    out_final = cfg.processed_dir
    out_tmp = cfg.processed_dir / "_tmp_physical"
    clean_tmp_dir(out_tmp, cfg.overwrite)
    clean_final_outputs(out_final, cfg.overwrite)

    counts_total = {"train": 0, "validation": 0, "test": 0}
    shard_idx = {"train": 0, "validation": 0, "test": 0}
    rejects_total: Dict[str, int] = {}

    rng = np.random.default_rng(cfg.seed)

    log(f"Starting preprocessing | raw_dir={cfg.raw_dir} | processed_dir={cfg.processed_dir}")
    log(
        "Config summary | "
        f"dt_mode={cfg.dt_mode}, dt=[{cfg.dt_min:g}, {cfg.dt_max:g}] ({cfg.dt_sampling}), "
        f"n_steps={cfg.n_steps}, t_min={cfg.t_min:g}, shard_size={cfg.shard_size}, "
        f"per_file_target={cfg.output_trajectories_per_file}, samples_per_source={cfg.samples_per_source_trajectory}, "
        f"drop_below={cfg.drop_below:g}"
    )

    for fp in raw_files:
        log(f"Processing {fp.name}")
        rej = sample_trajectories_from_file(fp, out_tmp, cfg, rng, counts_total, shard_idx)
        for k, v in rej.items():
            rejects_total[k] = rejects_total.get(k, 0) + int(v)

    total_written = counts_total["train"] + counts_total["validation"] + counts_total["test"]
    if total_written == 0:
        raise RuntimeError("No trajectories were written. Check rejects in preprocessing_summary.json.")

    # Log aggregate drop reasons (stdout only).
    log(
        f"[RUN TOTALS] written={total_written} "
        f"(train={counts_total['train']}, val={counts_total['validation']}, test={counts_total['test']})"
    )
    if rejects_total:
        ranked = sorted(rejects_total.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ", ".join([f"{k}={v}" for k, v in ranked if v > 0])
        if top:
            log(f"[RUN DROPS] {top}")

    per_key_stats, methods = compute_train_stats_from_physical(out_tmp, cfg, cfg.species_variables)
    normalize_and_write_final(out_tmp, out_final, cfg, cfg.species_variables, per_key_stats, methods)
    write_summary(out_final, cfg, counts_total, rejects_total)
    log("Done.")


if __name__ == "__main__":
    main()
