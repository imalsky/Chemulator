#!/usr/bin/env python3
"""
preprocessing.py

Create processed NPZ shards from raw HDF5 trajectories.

Pipeline:
1) Sample constant-Δt sub-trajectories ("chunks") from raw adaptive-time trajectories.
2) Write *physical* shards (y, globals, t_vec) as NPZ (uncompressed).
3) Compute normalization stats from TRAIN split physical shards.
4) Normalize physical shards -> z-space shards and write final NPZ shards.
5) Write normalization.json and preprocessing_summary.json

Config is strict JSON (no comments).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
# Config file location: one directory up from this script, then in the config folder
# e.g., if this script is at /project/src/preprocessing.py,
#       config is at /project/config/config.json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


# ---------------------------- Logging ----------------------------


def log(msg: str) -> None:
    """Log a timestamped message to stdout."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


# ---------------------------- JSON Loader ----------------------------


def load_json_config(path: Path) -> Dict:
    """Load a strict JSON config (no comments, no trailing commas)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON at {path}: {e}") from e


# ---------------------------- HDF5 Helpers ----------------------------


def first_group(fin: h5py.File) -> str:
    """Return the name of the first group in an HDF5 file."""
    for k in fin.keys():
        if isinstance(fin[k], h5py.Group):
            return k
    raise ValueError("No groups found in file")


def get_time_array(grp: h5py.Group, time_keys: List[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Try to find and return a time array from the group using candidate key names.
    
    Returns:
        (key_name, time_array) or (None, None) if not found
    """
    for k in time_keys:
        if k in grp and isinstance(grp[k], h5py.Dataset):
            t = np.asarray(grp[k][...], dtype=np.float64).reshape(-1)
            if t.size >= 2 and np.all(np.isfinite(t)):
                return k, t
    return None, None


def read_global(grp: h5py.Group, key: str) -> float:
    """
    Read a scalar global parameter from a group (dataset or attribute).
    
    Raises:
        KeyError: if the global is not found or invalid
    """
    # Try as dataset first
    if key in grp and isinstance(grp[key], h5py.Dataset):
        v = np.asarray(grp[key][...]).reshape(-1)
        if v.size == 1 and np.isfinite(v[0]):
            return float(v[0])
    # Try as attribute
    if key in grp.attrs:
        v = grp.attrs[key]
        vv = np.asarray(v).reshape(-1)
        if vv.size == 1 and np.isfinite(vv[0]):
            return float(vv[0])
    raise KeyError(f"Missing global '{key}' in group {grp.name}")


def detect_species_vars(grp: h5py.Group, t_len: int, time_key: str, global_vars: List[str]) -> List[str]:
    """
    Auto-detect species variables: 1D arrays of length t_len, excluding time and globals.
    """
    out: List[str] = []
    for k in grp.keys():
        if k == time_key:
            continue
        if k in global_vars:
            continue
        obj = grp[k]
        if isinstance(obj, h5py.Dataset):
            arr = np.asarray(obj[...])
            if arr.ndim == 1 and arr.shape[0] == t_len:
                out.append(k)
    out.sort()
    return out


# ---------------------------- Interpolation ----------------------------


def interp_loglog_precomputed(
    y_valid: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    w: np.ndarray,
    one_minus_w: np.ndarray,
) -> np.ndarray:
    """
    Perform log-log interpolation given precomputed neighbor indices and weights.

    Args:
        y_valid: [T_valid, S] physical values in linear space
        i0, i1: neighbor indices
        w, one_minus_w: interpolation weights
        
    Returns:
        [T_new, S] interpolated values in linear space
    """
    y0 = np.log(np.maximum(y_valid[i0, :], 1e-300))
    y1 = np.log(np.maximum(y_valid[i1, :], 1e-300))
    ylog = (one_minus_w[:, None] * y0) + (w[:, None] * y1)
    return np.exp(ylog)


def _prepare_log_interp(
    log_t_valid: np.ndarray, log_t_chunk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare neighbor indices and weights for log-time interpolation.

    Args:
        log_t_valid: [T_valid] strictly increasing log-times
        log_t_chunk: [T_new] target log-times within valid range
        
    Returns:
        (i0, i1, w, one_minus_w) arrays for interpolation
    """
    idx = np.searchsorted(log_t_valid, log_t_chunk, side="left")
    idx = np.clip(idx, 1, len(log_t_valid) - 1)
    i0 = idx - 1
    i1 = idx

    t0 = log_t_valid[i0]
    t1 = log_t_valid[i1]
    denom = np.maximum(t1 - t0, 1e-300)
    w = (log_t_chunk - t0) / denom
    w = np.clip(w, 0.0, 1.0)
    one_minus_w = 1.0 - w
    
    return (
        i0.astype(np.int64),
        i1.astype(np.int64),
        w.astype(np.float64),
        one_minus_w.astype(np.float64),
    )


# ---------------------------- Configuration ----------------------------


@dataclass
class PreCfg:
    """
    Preprocessing configuration container.
    
    The goal is to extract fixed-length trajectories (n_steps points, dt apart)
    from variable-length raw HDF5 trajectories.
    """
    raw_dir: Path
    processed_dir: Path

    # Output trajectory parameters
    # Each output trajectory has n_steps points spaced dt apart
    dt: float                  # Time step between points in output trajectory
    n_steps: int               # Number of points per output trajectory (e.g., 1000)
    t_min: float               # Minimum start time for sampling

    # Sampling controls
    # How many fixed-length trajectories to extract from the raw data
    output_trajectories_per_file: int      # Target number of output trajectories per raw file
    max_chunks_per_source_trajectory: int  # Max chunks to take from any single source trajectory
    anchor_first_chunk: bool               # If True, first chunk starts at t_min
    max_sampling_attempts_per_file: int    # Safety limit on sampling attempts
    
    # Filtering
    drop_below: float          # Reject data with values below this threshold
    time_keys: List[str]       # Candidate keys for time array in HDF5

    # Split and shard parameters
    val_fraction: float
    test_fraction: float
    shard_size: int
    overwrite: bool

    # Data keys
    global_variables: List[str]
    species_variables: List[str]

    # Normalization parameters
    epsilon: float
    min_std: float
    methods: Dict[str, str]
    default_method: str
    globals_default_method: str

    # Logging
    log_every_n_trajectories: int
    seed: int


def load_precfg(cfg: Dict) -> PreCfg:
    """
    Load preprocessing configuration from dictionary.
    
    Uses PROJECT_ROOT global to resolve relative paths.
    """
    pcfg = cfg.get("paths", {})
    raw_dir = Path(pcfg.get("raw_data_dir", PROJECT_ROOT / "data" / "raw"))
    processed_dir = Path(pcfg.get("processed_data_dir", PROJECT_ROOT / "data" / "processed"))

    # Resolve relative paths against project root
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
        # Output trajectory shape
        dt=float(pr.get("dt", 100.0)),
        n_steps=int(pr.get("n_steps", 1000)),
        t_min=float(pr.get("t_min", 1e-3)),
        # Sampling controls (support old and new key names for compatibility)
        output_trajectories_per_file=int(
            pr.get("output_trajectories_per_file", pr.get("profiles_per_file", 0))
        ),
        max_chunks_per_source_trajectory=int(
            pr.get("max_chunks_per_source_trajectory", pr.get("chunks_per_trajectory", 0))
        ),
        anchor_first_chunk=bool(
            pr.get("anchor_first_chunk", pr.get("anchor_first_chunk_per_trajectory", True))
        ),
        max_sampling_attempts_per_file=int(
            pr.get("max_sampling_attempts_per_file", pr.get("max_attempts_per_file", 5_000_000))
        ),
        # Filtering
        drop_below=float(pr.get("drop_below", 1e-35)),
        time_keys=list(pr.get("time_keys", ["t_time", "time", "t"])),
        # Split and shard
        val_fraction=float(pr.get("val_fraction", 0.02)),
        test_fraction=float(pr.get("test_fraction", 0.00)),
        shard_size=int(pr.get("shard_size", 1024)),
        overwrite=bool(pr.get("overwrite", True)),
        # Data keys
        global_variables=list(dcfg.get("global_variables", ["P", "T"])),
        species_variables=list(dcfg.get("species_variables", [])),
        # Normalization
        epsilon=float(ncfg.get("epsilon", 1e-30)),
        min_std=float(ncfg.get("min_std", 1e-12)),
        methods=dict(ncfg.get("methods", {})),
        default_method=str(ncfg.get("default_method", "log-standard")),
        globals_default_method=str(ncfg.get("globals_default_method", "standard")),
        # Logging and seed
        log_every_n_trajectories=int(pr.get("log_every_n_trajectories", 100)),
        seed=int(pr.get("seed", cfg.get("system", {}).get("seed", 1234))),
    )


# ---------------------------- Shard Writing ----------------------------


def _split_name(u: float, val_fraction: float, test_fraction: float) -> str:
    """Determine split name (train/validation/test) from random uniform value."""
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "validation"
    return "train"


def flush_shard(
    out_dir: Path,
    split: str,
    shard_idx: int,
    y_buf: List[np.ndarray],
    g_buf: List[np.ndarray],
    t_vec: np.ndarray,
    suffix: str,
) -> int:
    """
    Write accumulated samples to a shard file.
    
    Returns:
        Next shard index
    """
    if not y_buf:
        return shard_idx

    out_dir_split = out_dir / split
    out_dir_split.mkdir(parents=True, exist_ok=True)

    y_mat = np.stack(y_buf, axis=0)  # [N, T, S]
    g_mat = np.stack(g_buf, axis=0)  # [N, G]

    shard_path = out_dir_split / f"shard_{shard_idx:06d}{suffix}.npz"
    np.savez(shard_path, y_mat=y_mat, globals=g_mat, t_vec=t_vec.astype(np.float64, copy=False))
    return shard_idx + 1


def iter_shards(root: Path, split: str, suffix: str) -> List[Path]:
    """List all shard files for a given split and suffix."""
    d = root / split
    if not d.exists():
        return []
    return sorted(d.glob(f"shard_*{suffix}.npz"))


# ---------------------------- Sampling ----------------------------


def sample_trajectories_from_file(
    file_path: Path,
    out_tmp: Path,
    cfg: PreCfg,
    rng: np.random.Generator,
    t_vec_rel: np.ndarray,
    global_counts: Dict[str, int],
    last_log_count: int,
) -> int:
    """
    Sample fixed-length trajectories from one HDF5 file and write physical shards.
    
    Each output trajectory has cfg.n_steps points spaced cfg.dt apart, extracted
    from variable-length source trajectories via log-log interpolation.

    Args:
        file_path: Path to raw HDF5 file
        out_tmp: Directory for temporary physical shards
        cfg: Preprocessing configuration
        rng: Random number generator
        t_vec_rel: Relative time vector for output trajectories
        global_counts: Running counts per split (modified in place)
        last_log_count: Total count at last log message
        
    Returns:
        Updated last_log_count for next file
    """
    # Buffers per split
    y_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    g_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    shard_idx: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    # Duration of one output trajectory in time units
    chunk_duration = float((cfg.n_steps - 1) * cfg.dt)
    chunk_offsets = np.arange(cfg.n_steps, dtype=np.float64) * cfg.dt

    anchor_fail_logged: set = set()
    per_source_traj_counts: Dict[str, int] = {}

    # Per-file trajectory cache (LRU) to avoid re-reading same trajectories
    CACHE_MAX = 16
    traj_cache: Dict[str, Dict[str, np.ndarray]] = {}
    traj_cache_order: List[str] = []
    bad_traj: set = set()

    # Species ordering must be consistent; lock on first trajectory if not provided
    fixed_species_vars: Optional[List[str]] = cfg.species_variables[:] if cfg.species_variables else None

    def _cache_touch(name: str) -> None:
        """Move trajectory to end of LRU order (most recently used)."""
        if name in traj_cache_order:
            traj_cache_order.remove(name)
        traj_cache_order.append(name)

    def _cache_put(name: str, data: Dict[str, np.ndarray]) -> None:
        """Add trajectory data to cache, evicting oldest if full."""
        traj_cache[name] = data
        _cache_touch(name)
        if len(traj_cache_order) > CACHE_MAX:
            old = traj_cache_order.pop(0)
            traj_cache.pop(old, None)

    def _get_traj_data(traj_name: str, grp: h5py.Group) -> Optional[Dict[str, np.ndarray]]:
        """
        Load and validate trajectory data from HDF5 group.
        
        Returns cached data if available, otherwise loads from disk.
        Returns None if trajectory is invalid or has been marked bad.
        """
        nonlocal fixed_species_vars

        if traj_name in bad_traj:
            return None
        cached = traj_cache.get(traj_name)
        if cached is not None:
            _cache_touch(traj_name)
            return cached

        time_key, t_raw = get_time_array(grp, cfg.time_keys)
        if t_raw is None or time_key is None:
            bad_traj.add(traj_name)
            return None

        pos = t_raw > 0
        if not np.any(pos):
            bad_traj.add(traj_name)
            return None

        first_pos = float(t_raw[pos][0])
        t_lo = max(cfg.t_min, first_pos)

        valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
        if not np.any(valid):
            bad_traj.add(traj_name)
            return None

        t_valid = t_raw[valid]
        if t_valid.size < 2:
            bad_traj.add(traj_name)
            return None

        # Lock species ordering on first valid trajectory (critical for correctness)
        if fixed_species_vars is None:
            detected = detect_species_vars(grp, len(t_raw), time_key, cfg.global_variables)
            if not detected:
                bad_traj.add(traj_name)
                return None
            fixed_species_vars = detected
            log(f"[{file_path.name}] Locked species_variables from autodetect: {fixed_species_vars}")

        species_vars = fixed_species_vars

        # Load species matrix [T_raw, S]
        try:
            y_raw = np.stack(
                [
                    np.asarray(grp[s][...], dtype=np.float64).reshape(len(t_raw), -1)[:, 0]
                    for s in species_vars
                ],
                axis=1,
            )
        except Exception:
            bad_traj.add(traj_name)
            return None

        # Reject trajectories with values below threshold (numerical stability)
        if np.any(y_raw < cfg.drop_below):
            bad_traj.add(traj_name)
            return None
        if not np.all(np.isfinite(y_raw)):
            bad_traj.add(traj_name)
            return None

        y_valid = y_raw[valid, :]  # [T_valid, S]
        log_t_valid = np.log10(t_valid)

        # Load global parameters (e.g., pressure, temperature)
        try:
            g_vec = np.array([read_global(grp, gv) for gv in cfg.global_variables], dtype=np.float64)
        except Exception:
            bad_traj.add(traj_name)
            return None
        if not np.all(np.isfinite(g_vec)):
            bad_traj.add(traj_name)
            return None

        data = {
            "t_raw": t_raw,
            "t_lo": np.asarray([t_lo], dtype=np.float64),
            "t_valid": t_valid,
            "log_t_valid": log_t_valid,
            "y_valid": y_valid,
            "g_vec": g_vec,
        }
        _cache_put(traj_name, data)
        return data

    with h5py.File(file_path, "r") as fin:
        # Source trajectory pool = top-level groups in HDF5 file
        pool = [k for k in fin.keys() if isinstance(fin[k], h5py.Group)]
        if not pool:
            return last_log_count

        attempts = 0
        written = 0
        target = cfg.output_trajectories_per_file

        # Main sampling loop: randomly select source trajectories and extract chunks
        while written < target and attempts < cfg.max_sampling_attempts_per_file:
            attempts += 1

            traj_name = pool[int(rng.integers(0, len(pool)))]
            grp = fin.get(traj_name, None)
            if grp is None or not isinstance(grp, h5py.Group):
                continue

            # Limit chunks per source trajectory to ensure diversity
            ccount = per_source_traj_counts.get(traj_name, 0)
            if cfg.max_chunks_per_source_trajectory > 0 and ccount >= cfg.max_chunks_per_source_trajectory:
                continue

            td = _get_traj_data(traj_name, grp)
            if td is None:
                continue

            t_raw = td["t_raw"]
            t_lo = float(td["t_lo"][0])
            t_valid = td["t_valid"]
            log_t_valid = td["log_t_valid"]
            y_valid = td["y_valid"]
            g_vec = td["g_vec"]

            t_hi = float(t_raw[-1] - chunk_duration)
            if t_hi <= t_lo:
                continue

            # Choose start time: anchor first chunk or random log-uniform
            if cfg.anchor_first_chunk and ccount == 0:
                t_start = t_lo
            else:
                t_start = 10.0 ** float(rng.uniform(np.log10(t_lo), np.log10(t_hi)))

            t_chunk = t_start + chunk_offsets

            # Validate chunk is within valid time range
            if t_chunk[0] < t_valid[0] or t_chunk[-1] > t_valid[-1]:
                if cfg.anchor_first_chunk and ccount == 0 and traj_name not in anchor_fail_logged:
                    log(
                        f"[{file_path.name}] anchor infeasible for traj='{traj_name}': "
                        f"t_start={t_start:.3e}, t_valid=[{t_valid[0]:.3e},{t_valid[-1]:.3e}], "
                        f"chunk_end={t_chunk[-1]:.3e}"
                    )
                    anchor_fail_logged.add(traj_name)
                continue

            # Interpolate to uniform time grid using log-log interpolation
            log_t_chunk = np.log10(t_chunk)
            i0, i1, w, one_minus_w = _prepare_log_interp(log_t_valid, log_t_chunk)

            y_new = interp_loglog_precomputed(y_valid, i0, i1, w, one_minus_w).astype(np.float64, copy=False)
            if y_new.shape[0] != cfg.n_steps or not np.all(np.isfinite(y_new)):
                continue

            # Assign to train/validation/test split
            split = _split_name(float(rng.random()), cfg.val_fraction, cfg.test_fraction)

            y_buf[split].append(y_new.astype(np.float32, copy=False))
            g_buf[split].append(g_vec.astype(np.float32, copy=False))
            global_counts[split] += 1

            # Flush shard to disk when buffer is full
            if len(y_buf[split]) >= cfg.shard_size:
                shard_idx[split] = flush_shard(
                    out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
                )
                y_buf[split].clear()
                g_buf[split].clear()

            written += 1
            per_source_traj_counts[traj_name] = ccount + 1

            # Log progress based on total trajectories written
            total_written = sum(global_counts.values())
            if total_written - last_log_count >= cfg.log_every_n_trajectories:
                log(
                    f"Progress: {total_written} trajectories written "
                    f"(train={global_counts['train']}, val={global_counts['validation']}, test={global_counts['test']})"
                )
                last_log_count = total_written

        # Flush remaining samples in buffers
        for split in ("train", "validation", "test"):
            shard_idx[split] = flush_shard(
                out_tmp, split, shard_idx[split], y_buf[split], g_buf[split], t_vec_rel, suffix="_physical"
            )

    return last_log_count


# ---------------------------- Statistics Computation ----------------------------


class RunningMeanVar:
    """
    Welford's online algorithm for computing running mean and variance.
    
    Memory-efficient: processes data in batches without storing all values.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.M2 = np.zeros((dim,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        x2 = x.reshape(-1, self.dim)
        m = int(x2.shape[0])
        if m == 0:
            return

        # Compute batch statistics
        b_mean = np.mean(x2, axis=0)
        b_M2 = np.sum((x2 - b_mean) ** 2, axis=0)

        if self.n == 0:
            self.n = m
            self.mean = b_mean
            self.M2 = b_M2
            return

        # Combine with existing statistics using parallel algorithm
        n0 = float(self.n)
        m0 = float(m)
        delta = b_mean - self.mean
        n_new = n0 + m0
        self.mean = self.mean + delta * (m0 / n_new)
        self.M2 = self.M2 + b_M2 + (delta ** 2) * (n0 * m0 / n_new)
        self.n = int(n_new)

    def finalize(self, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) with minimum std floor for numerical stability."""
        if self.n < 2:
            std = np.ones_like(self.mean)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, min_std ** 2))
        return self.mean, std


def compute_train_stats_from_physical(
    out_tmp: Path,
    cfg: PreCfg,
    species_vars: List[str],
) -> Tuple[Dict, Dict[str, str]]:
    """
    Compute normalization statistics from TRAIN split physical shards.
    
    Species are always log-standardized (required by training/loss code).
    Globals can be: standard, min-max, or identity.

    Returns:
        (per_key_stats, methods) dictionaries
    """
    eps = float(cfg.epsilon)

    # Methods mapping: determines how each variable is normalized
    methods: Dict[str, str] = {}

    # Species are always log-standardized (required by training/loss code)
    for s in species_vars:
        requested = str(cfg.methods.get(s, cfg.default_method))
        if requested not in ("log-standard", "log_std", "logstandard"):
            raise ValueError(
                f"Species '{s}' requested normalization method '{requested}', but this codebase "
                "requires 'log-standard' for species z-space training."
            )
        methods[s] = "log-standard"

    # Globals: allow 'standard', 'min-max', or 'identity'
    for g in cfg.global_variables:
        requested = str(cfg.methods.get(g, cfg.globals_default_method))
        if requested not in ("standard", "min-max", "identity", "minmax"):
            raise ValueError(
                f"Global '{g}' requested normalization method '{requested}'. "
                "Supported: standard | min-max | identity."
            )
        methods[g] = "min-max" if requested == "minmax" else requested

    # Time grid uses identity (no normalization needed)
    methods["t_vec"] = "identity"

    per_key_stats: Dict = {}

    # Running stats for species (computed in log space)
    rms = {s: RunningMeanVar(1) for s in species_vars}
    s_min = {s: np.inf for s in species_vars}
    s_max = {s: -np.inf for s in species_vars}

    # Running stats for globals (computed in physical space)
    g_rms = {g: RunningMeanVar(1) for g in cfg.global_variables}
    g_min = {g: np.inf for g in cfg.global_variables}
    g_max = {g: -np.inf for g in cfg.global_variables}

    shards = iter_shards(out_tmp, "train", suffix="_physical")
    if not shards:
        raise RuntimeError("No train physical shards found; cannot compute stats.")

    # Iterate through all training shards to compute statistics
    for p in tqdm(shards, desc="Computing normalization stats", leave=False):
        with np.load(p) as z:
            y = np.asarray(z["y_mat"], dtype=np.float64)  # [N, T, S] physical
            g = np.asarray(z["globals"], dtype=np.float64)  # [N, G] physical

        # Species: compute log-space statistics
        for i, s in enumerate(species_vars):
            y_i = np.maximum(y[..., i], eps)
            y_log = np.log10(y_i).reshape(-1, 1)
            rms[s].update(y_log)
            s_min[s] = float(min(s_min[s], np.min(y_log)))
            s_max[s] = float(max(s_max[s], np.max(y_log)))

        # Globals: compute physical-space statistics
        for i, gv in enumerate(cfg.global_variables):
            g_i = g[:, i].reshape(-1, 1)
            g_rms[gv].update(g_i)
            g_min[gv] = float(min(g_min[gv], np.min(g_i)))
            g_max[gv] = float(max(g_max[gv], np.max(g_i)))

    # Finalize species stats
    for s in species_vars:
        mu, sd = rms[s].finalize(cfg.min_std)
        per_key_stats[s] = {
            "log_mean": float(mu[0]),
            "log_std": float(sd[0]),
            "log_min": float(s_min[s]),
            "log_max": float(s_max[s]),
            "epsilon": eps,
        }

    # Finalize global stats
    for gv in cfg.global_variables:
        mu, sd = g_rms[gv].finalize(cfg.min_std)
        per_key_stats[gv] = {
            "mean": float(mu[0]),
            "std": float(sd[0]),
            "min": float(g_min[gv]),
            "max": float(g_max[gv]),
        }

    # Time grid identity (no stats needed)
    per_key_stats["t_vec"] = {"method": "identity"}

    return per_key_stats, methods


# ---------------------------- Final Normalization ----------------------------


def normalize_and_write_final(
    out_tmp: Path,
    out_final: Path,
    cfg: PreCfg,
    species_vars: List[str],
    per_key_stats: Dict,
    methods: Dict[str, str],
    t_vec_global: np.ndarray,
) -> None:
    """
    Normalize physical shards to z-space and write final NPZ shards.
    
    Z-space transformation:
    - Species: z = (log10(y) - mean) / std
    - Globals: depends on method (standard, min-max, or identity)
    
    Args:
        out_tmp: Directory containing physical shards
        out_final: Directory for final z-space shards
        cfg: Preprocessing configuration
        species_vars: List of species variable names
        per_key_stats: Dictionary of per-key statistics
        methods: Dictionary of normalization methods
        t_vec_global: Global time vector (used for all shards)
    """
    out_final.mkdir(parents=True, exist_ok=True)
    eps = float(cfg.epsilon)

    # Create output split directories
    for split in ("train", "validation", "test"):
        (out_final / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ("train", "validation", "test"):
        physical_shards = iter_shards(out_tmp, split, suffix="_physical")
        if not physical_shards:
            continue

        # Accumulators for re-sharding (ensures consistent shard sizes)
        y_buf: List[np.ndarray] = []
        g_buf: List[np.ndarray] = []
        shard_counter = 0

        for p in tqdm(physical_shards, desc=f"Normalizing {split}", leave=False):
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float64)       # [N, T, S] physical
                g = np.asarray(z["globals"], dtype=np.float64)     # [N, G] physical

            # Species -> z-space via log-standard normalization
            y_z = np.empty_like(y, dtype=np.float32)
            for i, s in enumerate(species_vars):
                st = per_key_stats[s]
                mu = float(st["log_mean"])
                sd = float(st["log_std"])
                y_log = np.log10(np.maximum(y[..., i], eps))
                y_z[..., i] = ((y_log - mu) / sd).astype(np.float32, copy=False)

            # Globals -> z-space (configurable: standard | min-max | identity)
            g_z = np.empty_like(g, dtype=np.float32)
            for i, gv in enumerate(cfg.global_variables):
                st = per_key_stats[gv]
                method = str(methods.get(gv, "standard"))
                if method == "identity":
                    g_z[:, i] = g[:, i].astype(np.float32, copy=False)
                elif method == "standard":
                    mu = float(st["mean"])
                    sd = float(st["std"])
                    g_z[:, i] = ((g[:, i] - mu) / sd).astype(np.float32, copy=False)
                elif method == "min-max":
                    mn = float(st["min"])
                    mx = float(st["max"])
                    denom = max(mx - mn, 1e-12)
                    g_z[:, i] = ((g[:, i] - mn) / denom).astype(np.float32, copy=False)
                else:
                    raise ValueError(f"Unsupported global normalization method '{method}' for key '{gv}'.")

            # Accumulate samples
            for n in range(y_z.shape[0]):
                y_buf.append(y_z[n])
                g_buf.append(g_z[n])

                # Write shard when buffer is full
                if len(y_buf) >= cfg.shard_size:
                    out_path = out_final / split / f"shard_{shard_counter:06d}.npz"
                    y_mat = np.stack(y_buf, axis=0)
                    g_mat = np.stack(g_buf, axis=0)
                    np.savez(out_path, y_mat=y_mat, globals=g_mat, t_vec=t_vec_global)
                    shard_counter += 1
                    y_buf.clear()
                    g_buf.clear()

        # Flush remaining samples for this split
        if y_buf:
            out_path = out_final / split / f"shard_{shard_counter:06d}.npz"
            y_mat = np.stack(y_buf, axis=0)
            g_mat = np.stack(g_buf, axis=0)
            np.savez(out_path, y_mat=y_mat, globals=g_mat, t_vec=t_vec_global)

    # Write normalization manifest (required for training and inference)
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
    with open(out_final / "normalization.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_summary(out_final: Path, cfg: PreCfg, counts_total: Dict[str, int], species_vars: List[str]) -> None:
    """Write preprocessing summary JSON for reproducibility and debugging."""
    summary = {
        "dt": cfg.dt,
        "n_steps": cfg.n_steps,
        "t_min": cfg.t_min,
        "output_trajectories_per_file": cfg.output_trajectories_per_file,
        "max_chunks_per_source_trajectory": cfg.max_chunks_per_source_trajectory,
        "drop_below": cfg.drop_below,
        "val_fraction": cfg.val_fraction,
        "test_fraction": cfg.test_fraction,
        "shard_size": cfg.shard_size,
        "global_variables": cfg.global_variables,
        "species_variables": species_vars,
        "counts_total": counts_total,
    }
    with open(out_final / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------- Main ----------------------------


def main() -> None:
    """
    Main preprocessing entrypoint.
    
    Loads config from CONFIG_PATH global (one directory up, in config folder).
    No command-line arguments required.
    """
    # Load configuration from global path
    log(f"Loading config from: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    cfg_dict = load_json_config(CONFIG_PATH)
    cfg = load_precfg(cfg_dict)

    out_final = cfg.processed_dir
    out_tmp = cfg.processed_dir / "_tmp_physical"

    if out_final.exists() and not cfg.overwrite:
        raise RuntimeError(f"Processed dir exists and overwrite=false: {out_final}")

    # Clean temporary directory if overwriting
    if out_tmp.exists() and cfg.overwrite:
        for p in out_tmp.rglob("*"):
            if p.is_file():
                p.unlink()
    out_tmp.mkdir(parents=True, exist_ok=True)

    # Define shared relative time vector (used for all shards)
    # Each output trajectory has n_steps points, spaced dt apart
    t_vec_rel = np.arange(cfg.n_steps, dtype=np.float64) * cfg.dt

    # Enumerate raw HDF5 files
    raw_files = sorted(cfg.raw_dir.glob("*.h5")) + sorted(cfg.raw_dir.glob("*.hdf5"))
    if not raw_files:
        raise FileNotFoundError(f"No HDF5 files found in raw_dir={cfg.raw_dir}")

    rng = np.random.default_rng(cfg.seed)

    # Global counts across all files
    counts_total = {"train": 0, "validation": 0, "test": 0}

    log(f"Raw dir: {cfg.raw_dir}")
    log(f"Processed dir: {cfg.processed_dir}")
    log(f"Found {len(raw_files)} raw file(s)")
    log(f"Target: {cfg.output_trajectories_per_file} trajectories per file "
        f"({cfg.n_steps} steps × dt={cfg.dt})")

    # Phase 1: Sample from raw files and write physical shards
    log("Phase 1: Sampling trajectories from raw files...")
    last_log_count = 0
    for i, fp in enumerate(raw_files):
        log(f"Processing file {i+1}/{len(raw_files)}: {fp.name}")
        last_log_count = sample_trajectories_from_file(
            fp, out_tmp, cfg, rng, t_vec_rel, counts_total, last_log_count
        )

    # Final count log
    total = sum(counts_total.values())
    log(f"Sampling complete: {total} trajectories "
        f"(train={counts_total['train']}, val={counts_total['validation']}, test={counts_total['test']})")

    # Validate species variables are specified
    species_vars = cfg.species_variables[:]
    if not species_vars:
        raise RuntimeError(
            "data.species_variables is empty. Set it explicitly in config so preprocessing "
            "can record the exact species ordering."
        )

    # Phase 2: Compute statistics from training physical shards
    log("Phase 2: Computing normalization statistics from training data...")
    per_key_stats, methods = compute_train_stats_from_physical(out_tmp, cfg, species_vars)

    # Clean output directory if overwriting
    if out_final.exists() and cfg.overwrite:
        for p in out_final.rglob("*"):
            if p.is_file():
                p.unlink()
    out_final.mkdir(parents=True, exist_ok=True)

    # Phase 3: Normalize and write final z-space shards
    log("Phase 3: Normalizing and writing final shards...")
    normalize_and_write_final(out_tmp, out_final, cfg, species_vars, per_key_stats, methods, t_vec_rel)

    # Phase 4: Write summary for reproducibility
    log("Phase 4: Writing preprocessing summary...")
    write_summary(out_final, cfg, counts_total, species_vars)

    log("Done.")


if __name__ == "__main__":
    main()