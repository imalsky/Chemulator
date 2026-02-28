#!/usr/bin/env python3
"""preprocessing.py

Preprocess raw adaptive-timestep HDF5 trajectories into fixed-length uniform-dt
windows, shard to NPZ, compute normalization, and write final normalized shards.

Outputs (under paths.processed_dir):
  normalization.json
  preprocessing_summary.json
  train/shard_*.npz
  validation/shard_*.npz
  test/shard_*.npz

Temporary stage (under paths.processed_dir):
  _tmp_physical/<split>/shard_*_physical.npz

Shard schema:
  Physical shards (temporary):
    y_mat   : [N, n_steps, S] float32  (physical units)
    globals : [N, G]          float32  (physical units)
    dt_mat  : [N, n_steps-1]  float32  (physical dt, constant within each sample)

  Final shards:
    y_mat       : [N, n_steps, S] float32  (z-space, log-standard)
    globals     : [N, G]          float32  (normalized globals)
    dt_norm_mat : [N, n_steps-1]  float32  (normalized dt in [0, 1])

This script is intentionally strict:
- Configuration must use the canonical schema in config.json.
- Dataset names must match exactly (no implicit renaming).
- Globals must be scalar per trajectory (time-series globals are rejected).

Run:
  python -m preprocessing --config /path/to/config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper().strip(), None)
    if lvl is None:
        raise ValueError(f"Unsupported log level: {level}")
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Strict config parsing
# -----------------------------------------------------------------------------


def _require(mapping: Dict, key: str) -> object:
    if key not in mapping:
        raise KeyError(f"missing config key: {key}")
    return mapping[key]


def _require_dict(mapping: Dict, key: str) -> Dict:
    obj = _require(mapping, key)
    if not isinstance(obj, dict):
        raise TypeError(f"config.{key} must be an object")
    return obj


def _require_str(mapping: Dict, key: str) -> str:
    obj = _require(mapping, key)
    if not isinstance(obj, str) or not obj.strip():
        raise TypeError(f"config.{key} must be a non-empty string")
    return obj.strip()


def _require_int(mapping: Dict, key: str) -> int:
    obj = _require(mapping, key)
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise TypeError(f"config.{key} must be an int")
    return int(obj)


def _require_float(mapping: Dict, key: str) -> float:
    obj = _require(mapping, key)
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise TypeError(f"config.{key} must be a number")
    return float(obj)


def _require_bool(mapping: Dict, key: str) -> bool:
    obj = _require(mapping, key)
    if not isinstance(obj, bool):
        raise TypeError(f"config.{key} must be a bool")
    return bool(obj)


def _require_str_list(mapping: Dict, key: str) -> List[str]:
    obj = _require(mapping, key)
    if not isinstance(obj, list) or not obj:
        raise TypeError(f"config.{key} must be a non-empty list")
    out: List[str] = []
    for v in obj:
        if not isinstance(v, str) or not v.strip():
            raise TypeError(f"config.{key} must contain non-empty strings")
        out.append(v.strip())
    return out


def _resolve_path(root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class PreCfg:
    raw_dir: Path
    processed_dir: Path
    raw_file_patterns: List[str]

    dt: float
    dt_mode: str  # fixed | per_chunk
    dt_min: float
    dt_max: float
    dt_sampling: str  # uniform | loguniform

    n_steps: int
    t_min: float

    output_trajectories_per_file: int
    shard_size: int
    overwrite: bool

    time_key: str

    val_fraction: float
    test_fraction: float

    global_variables: List[str]
    species_variables: List[str]

    epsilon: float
    min_std: float
    methods: Dict[str, str]
    globals_default_method: str

    seed: int
    pool_size: int
    samples_per_source_trajectory: int
    max_chunk_attempts_per_source: int
    drop_below: float


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError("Config must be a JSON object")
    return obj


def parse_precfg(cfg: Dict, *, cfg_path: Path) -> PreCfg:
    root = cfg_path.parent.resolve()

    paths = _require_dict(cfg, "paths")
    raw_dir = _resolve_path(root, _require_str(paths, "raw_dir"))
    processed_dir = _resolve_path(root, _require_str(paths, "processed_dir"))

    data = _require_dict(cfg, "data")
    global_variables = _require_str_list(data, "global_variables")
    species_variables = _require_str_list(data, "species_variables")

    norm = _require_dict(cfg, "normalization")
    epsilon = _require_float(norm, "epsilon")
    min_std = _require_float(norm, "min_std")
    globals_default_method = _require_str(norm, "globals_default_method")
    methods_obj = _require(norm, "methods")
    if not isinstance(methods_obj, dict):
        raise TypeError("config.normalization.methods must be an object")
    methods: Dict[str, str] = {}
    for k, v in methods_obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("config.normalization.methods must map strings to strings")
        methods[k] = v

    pr = _require_dict(cfg, "preprocessing")
    raw_file_patterns = _require_str_list(pr, "raw_file_patterns")

    dt = _require_float(pr, "dt")
    dt_mode = _require_str(pr, "dt_mode").lower()
    if dt_mode not in ("fixed", "per_chunk"):
        raise ValueError("preprocessing.dt_mode must be 'fixed' or 'per_chunk'")

    dt_min = _require_float(pr, "dt_min")
    dt_max = _require_float(pr, "dt_max")
    if dt_min <= 0.0 or dt_max <= 0.0 or dt_max < dt_min:
        raise ValueError("Invalid dt range: require 0 < dt_min <= dt_max")

    if dt_mode == "fixed":
        if dt_min != dt or dt_max != dt:
            raise ValueError("dt_mode='fixed' requires dt_min == dt_max == dt")

    dt_sampling = _require_str(pr, "dt_sampling").lower()
    if dt_sampling not in ("uniform", "loguniform"):
        raise ValueError("preprocessing.dt_sampling must be 'uniform' or 'loguniform'")

    n_steps = _require_int(pr, "n_steps")
    if n_steps < 2:
        raise ValueError("preprocessing.n_steps must be >= 2")

    t_min = _require_float(pr, "t_min")
    if t_min < 0.0:
        raise ValueError("preprocessing.t_min must be >= 0")

    output_trajectories_per_file = _require_int(pr, "output_trajectories_per_file")
    shard_size = _require_int(pr, "shard_size")
    overwrite = _require_bool(pr, "overwrite")

    time_key = _require_str(pr, "time_key")

    val_fraction = _require_float(pr, "val_fraction")
    test_fraction = _require_float(pr, "test_fraction")
    if val_fraction < 0.0 or test_fraction < 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError("Require 0 <= val_fraction, 0 <= test_fraction, and val_fraction + test_fraction < 1")

    seed = _require_int(pr, "seed")
    pool_size = _require_int(pr, "pool_size")
    samples_per_source_trajectory = _require_int(pr, "samples_per_source_trajectory")
    max_chunk_attempts_per_source = _require_int(pr, "max_chunk_attempts_per_source")
    drop_below = _require_float(pr, "drop_below")

    if output_trajectories_per_file <= 0:
        raise ValueError("preprocessing.output_trajectories_per_file must be > 0")
    if shard_size <= 0:
        raise ValueError("preprocessing.shard_size must be > 0")
    if pool_size <= 0:
        raise ValueError("preprocessing.pool_size must be > 0")
    if samples_per_source_trajectory <= 0:
        raise ValueError("preprocessing.samples_per_source_trajectory must be > 0")
    if max_chunk_attempts_per_source <= 0:
        raise ValueError("preprocessing.max_chunk_attempts_per_source must be > 0")
    if epsilon <= 0.0:
        raise ValueError("normalization.epsilon must be > 0")
    if min_std <= 0.0:
        raise ValueError("normalization.min_std must be > 0")

    return PreCfg(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_file_patterns=raw_file_patterns,
        dt=dt,
        dt_mode=dt_mode,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_sampling=dt_sampling,
        n_steps=n_steps,
        t_min=t_min,
        output_trajectories_per_file=output_trajectories_per_file,
        shard_size=shard_size,
        overwrite=overwrite,
        time_key=time_key,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        global_variables=global_variables,
        species_variables=species_variables,
        epsilon=epsilon,
        min_std=min_std,
        methods=methods,
        globals_default_method=globals_default_method,
        seed=seed,
        pool_size=pool_size,
        samples_per_source_trajectory=samples_per_source_trajectory,
        max_chunk_attempts_per_source=max_chunk_attempts_per_source,
        drop_below=drop_below,
    )


# -----------------------------------------------------------------------------
# File utilities
# -----------------------------------------------------------------------------


def list_raw_files(raw_dir: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(raw_dir.glob(pat)))
    uniq = sorted({p.resolve() for p in files})
    return uniq


def clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_processed_outputs(processed_dir: Path, *, overwrite: bool) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        d = processed_dir / split
        if d.exists():
            if not overwrite:
                raise RuntimeError(f"Refusing to overwrite existing output split dir: {d}")
            shutil.rmtree(d)


def flush_shard(
    out_dir: Path,
    *,
    split: str,
    shard_idx: int,
    y_buf: List[np.ndarray],
    g_buf: List[np.ndarray],
    dt_buf: List[np.ndarray],
    suffix: str,
) -> Tuple[int, int]:
    if not y_buf:
        return shard_idx, 0

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    y_mat = np.stack(y_buf, axis=0)
    g_mat = np.stack(g_buf, axis=0)
    dt_mat = np.stack(dt_buf, axis=0)

    shard_path = split_dir / f"shard_{shard_idx:06d}{suffix}.npz"

    if suffix.endswith("_physical"):
        np.savez(shard_path, y_mat=y_mat, globals=g_mat, dt_mat=dt_mat)
    else:
        np.savez(shard_path, y_mat=y_mat, globals=g_mat, dt_norm_mat=dt_mat)

    return shard_idx + 1, int(y_mat.shape[0])


def iter_shards(root: Path, split: str, suffix: str) -> List[Path]:
    d = root / split
    return sorted(d.glob(f"shard_*{suffix}.npz")) if d.exists() else []


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


def reservoir_sample(keys: Iterable[str], k: int, rng: np.random.Generator) -> List[str]:
    pool: List[str] = []
    for i, key in enumerate(keys):
        if i < k:
            pool.append(key)
            continue
        j = int(rng.integers(0, i + 1))
        if j < k:
            pool[j] = key
    return pool


def split_for_trajectory(file_name: str, traj_name: str, *, seed: int, val_frac: float, test_frac: float) -> str:
    key = f"{file_name}:{traj_name}"
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    u = (int.from_bytes(h[:8], "big") + 0.5) / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "validation"
    return "train"


def sample_dt(cfg: PreCfg, rng: np.random.Generator) -> float:
    if cfg.dt_mode == "fixed":
        return float(cfg.dt)
    if cfg.dt_sampling == "loguniform":
        lo = np.log10(cfg.dt_min)
        hi = np.log10(cfg.dt_max)
        return float(10.0 ** rng.uniform(lo, hi))
    return float(rng.uniform(cfg.dt_min, cfg.dt_max))


def pick_t_start(
    *,
    t_raw: np.ndarray,
    t_valid: np.ndarray,
    dt_s: float,
    n_steps: int,
    t_min: float,
    rng: np.random.Generator,
    anchor_first: bool,
    max_attempts: int,
) -> Optional[float]:
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
        return (t_chunk[0] >= float(t_valid[0])) and (t_chunk[-1] <= float(t_valid[-1]))

    if anchor_first and fits(t_lo):
        return float(t_lo)

    lo = np.log10(t_lo)
    hi = np.log10(t_hi)

    for _ in range(max_attempts):
        cand = float(10.0 ** rng.uniform(lo, hi))
        if fits(cand):
            return cand

    return None


# -----------------------------------------------------------------------------
# HDF5 reading (strict)
# -----------------------------------------------------------------------------


def leaf_dataset_index(grp: h5py.Group) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def _unique_dataset_path(grp: h5py.Group, leaf_index: Dict[str, List[str]], key: str) -> str:
    if "/" in key:
        if key not in grp:
            raise KeyError(f"Dataset path not found in trajectory group: {key}")
        obj = grp[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Expected dataset at {key}")
        return key

    matches = leaf_index.get(key, [])
    if not matches:
        raise KeyError(f"Dataset not found in trajectory group: {key}")
    if len(matches) != 1:
        raise KeyError(f"Dataset name is ambiguous in trajectory group: {key} -> {matches}")
    return matches[0]


def read_time(grp: h5py.Group, *, time_key: str, leaf_index: Dict[str, List[str]]) -> np.ndarray:
    p = _unique_dataset_path(grp, leaf_index, time_key)

    t = np.asarray(grp[p][...], dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError(f"time dataset '{time_key}' has < 2 samples")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"time dataset '{time_key}' contains non-finite values")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"time dataset '{time_key}' is not strictly increasing")
    return t


def read_species_matrix(
    grp: h5py.Group,
    *,
    t_len: int,
    species_vars: Sequence[str],
    leaf_index: Dict[str, List[str]],
) -> np.ndarray:
    s = len(species_vars)
    y = np.empty((t_len, s), dtype=np.float64)

    for j, name in enumerate(species_vars):
        p = _unique_dataset_path(grp, leaf_index, name)
        arr = np.asarray(grp[p][...], dtype=np.float64)

        if arr.ndim == 0 or int(arr.shape[0]) != int(t_len):
            raise ValueError(f"Species dataset '{name}' must be time-aligned with length {t_len}, got {arr.shape}")

        col = arr.reshape(t_len, -1)[:, 0]
        if not np.all(np.isfinite(col)):
            raise ValueError(f"Species dataset '{name}' contains non-finite values")
        y[:, j] = col

    return y


def read_globals_vector(
    grp: h5py.Group,
    *,
    global_vars: Sequence[str],
    leaf_index: Dict[str, List[str]],
) -> np.ndarray:
    if not global_vars:
        return np.zeros((0,), dtype=np.float32)

    out = np.empty((len(global_vars),), dtype=np.float64)

    for j, name in enumerate(global_vars):
        # Prefer per-trajectory attribute.
        if name in grp.attrs:
            v = np.asarray(grp.attrs[name], dtype=np.float64).reshape(-1)
            if v.size != 1 or not np.isfinite(v[0]):
                raise ValueError(f"Global attribute '{name}' must be a finite scalar")
            out[j] = float(v[0])
            continue

        # Otherwise require a scalar dataset in the trajectory group.
        p = _unique_dataset_path(grp, leaf_index, name)
        ds = grp[p]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"Global '{name}' is not a dataset")

        arr = np.asarray(ds[...], dtype=np.float64).reshape(-1)
        if arr.size != 1 or not np.isfinite(arr[0]):
            raise ValueError(f"Global '{name}' must be a finite scalar dataset; got shape {ds.shape}")
        out[j] = float(arr[0])

    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Interpolation
# -----------------------------------------------------------------------------


def prepare_log_interp(log_t_valid: np.ndarray, log_t_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(log_t_valid.shape[0])
    idx = np.searchsorted(log_t_valid, log_t_target, side="left")
    idx = np.clip(idx, 1, n - 1)

    i0 = (idx - 1).astype(np.int64)
    i1 = idx.astype(np.int64)

    t0 = log_t_valid[i0]
    t1 = log_t_valid[i1]
    denom = t1 - t0
    if np.any(denom <= 0):
        raise ValueError("Invalid time grid: encountered non-positive interpolation denominator")

    w = (log_t_target - t0) / denom
    w = np.clip(w, 0.0, 1.0).astype(np.float64)
    return i0, i1, w


def interp_loglog(
    y_valid: np.ndarray,
    *,
    i0: np.ndarray,
    i1: np.ndarray,
    w: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    # log-log interpolation in physical space:
    #   log10(y) linear-interpolated in log10(t)
    y0 = np.log10(np.maximum(y_valid[i0, :], epsilon))
    y1 = np.log10(np.maximum(y_valid[i1, :], epsilon))
    w2 = w.reshape(-1, 1)
    return 10.0 ** ((1.0 - w2) * y0 + w2 * y1)


# -----------------------------------------------------------------------------
# Core preprocessing
# -----------------------------------------------------------------------------


def sample_file(
    file_path: Path,
    *,
    out_tmp: Path,
    cfg: PreCfg,
    rng: np.random.Generator,
    counts_total: Dict[str, int],
    shard_idx: Dict[str, int],
) -> Dict[str, int]:
    """Sample and resample trajectories from a single HDF5 file into physical shards."""

    y_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    g_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    dt_buf: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}

    rejects: Dict[str, int] = {
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

    written_by_split = {"train": 0, "validation": 0, "test": 0}

    target = int(cfg.output_trajectories_per_file)
    written = 0

    with h5py.File(file_path, "r") as fin:
        pool = reservoir_sample(fin.keys(), int(cfg.pool_size), rng)
        rng.shuffle(pool)

        for traj_name in pool:
            if written >= target:
                break

            grp_obj = fin[traj_name]
            if not isinstance(grp_obj, h5py.Group):
                rejects["not_group"] += 1
                continue

            grp: h5py.Group = grp_obj

            leaf_index = leaf_dataset_index(grp)

            # Time
            try:
                t_raw = read_time(grp, time_key=cfg.time_key, leaf_index=leaf_index)
            except (KeyError, ValueError, TypeError):
                rejects["no_time"] += 1
                continue

            T = int(t_raw.shape[0])

            # Species
            try:
                y_raw = read_species_matrix(grp, t_len=T, species_vars=cfg.species_variables, leaf_index=leaf_index)
            except KeyError:
                rejects["missing_species"] += 1
                continue
            except (ValueError, TypeError):
                rejects["non_finite"] += 1
                continue

            # Globals
            try:
                g_vec = read_globals_vector(grp, global_vars=cfg.global_variables, leaf_index=leaf_index)
            except KeyError:
                rejects["missing_globals"] += 1
                continue
            except (ValueError, TypeError):
                rejects["non_finite"] += 1
                continue

            # Validity mask (matches training assumptions):
            # - Only consider positive times.
            # - Require at least 2 valid samples.
            pos = t_raw > 0
            if not np.any(pos):
                rejects["too_few_valid"] += 1
                continue

            first_pos_time = float(t_raw[pos][0])
            t_lo = max(float(cfg.t_min), first_pos_time)
            valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
            if int(np.count_nonzero(valid)) < 2:
                rejects["too_few_valid"] += 1
                continue

            if np.any(y_raw < float(cfg.drop_below)):
                rejects["drop_below"] += 1
                continue

            t_valid = t_raw[valid]
            log_t_valid = np.log10(t_valid)
            y_valid = y_raw[valid, :]

            split = split_for_trajectory(
                file_path.name,
                str(traj_name),
                seed=cfg.seed,
                val_frac=cfg.val_fraction,
                test_frac=cfg.test_fraction,
            )

            for sidx in range(int(cfg.samples_per_source_trajectory)):
                if written >= target:
                    break

                dt_s = sample_dt(cfg, rng)
                t_start = pick_t_start(
                    t_raw=t_raw,
                    t_valid=t_valid,
                    dt_s=dt_s,
                    n_steps=cfg.n_steps,
                    t_min=cfg.t_min,
                    rng=rng,
                    anchor_first=(sidx == 0),
                    max_attempts=cfg.max_chunk_attempts_per_source,
                )
                if t_start is None:
                    rejects["no_fit_chunk"] += 1
                    continue

                t_chunk = float(t_start) + np.arange(cfg.n_steps, dtype=np.float64) * float(dt_s)
                log_t_chunk = np.log10(t_chunk)

                i0, i1, w = prepare_log_interp(log_t_valid, log_t_chunk)
                y_new = interp_loglog(y_valid, i0=i0, i1=i1, w=w, epsilon=cfg.epsilon)

                if not np.all(np.isfinite(y_new)):
                    rejects["interp_non_finite"] += 1
                    continue

                y_buf[split].append(y_new.astype(np.float32, copy=False))
                g_buf[split].append(g_vec.astype(np.float32, copy=False))
                dt_buf[split].append(np.full(cfg.n_steps - 1, dt_s, dtype=np.float32))

                counts_total[split] += 1
                written_by_split[split] += 1
                written += 1

                if len(y_buf[split]) >= cfg.shard_size:
                    shard_idx[split], _ = flush_shard(
                        out_tmp,
                        split=split,
                        shard_idx=shard_idx[split],
                        y_buf=y_buf[split],
                        g_buf=g_buf[split],
                        dt_buf=dt_buf[split],
                        suffix="_physical",
                    )
                    y_buf[split].clear()
                    g_buf[split].clear()
                    dt_buf[split].clear()

        for sp in ("train", "validation", "test"):
            shard_idx[sp], _ = flush_shard(
                out_tmp,
                split=sp,
                shard_idx=shard_idx[sp],
                y_buf=y_buf[sp],
                g_buf=g_buf[sp],
                dt_buf=dt_buf[sp],
                suffix="_physical",
            )

    log.info(
        "file=%s written=%d (train=%d val=%d test=%d) rejects=%s",
        file_path.name,
        written,
        written_by_split["train"],
        written_by_split["validation"],
        written_by_split["test"],
        rejects,
    )

    return rejects


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


class RunningMeanVar:
    """Streaming per-dimension mean/variance (Welford merge)."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
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

    def finalize(self, *, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 1:
            raise RuntimeError("Insufficient samples to compute variance")
        var = self.M2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, float(min_std) ** 2))
        return self.mean, std


def canonical_method(method: str) -> str:
    m = str(method).lower().strip()
    if m in ("none", ""):
        return "identity"
    if m in ("minmax", "min_max", "min-max"):
        return "min-max"
    if m in ("logminmax", "log-minmax", "log_min_max", "log-min-max"):
        return "log-min-max"
    if m in ("log10-standard", "log10_standard"):
        return "log-standard"
    return m


def compute_train_stats(out_tmp: Path, cfg: PreCfg) -> Tuple[Dict, Dict[str, str]]:
    """Compute normalization statistics from physical train shards."""

    shards = iter_shards(out_tmp, "train", "_physical")
    if not shards:
        raise RuntimeError("No physical train shards found to compute normalization")

    S = len(cfg.species_variables)
    G = len(cfg.global_variables)

    rms_logy = RunningMeanVar(S)
    logy_min = np.full(S, np.inf, dtype=np.float64)
    logy_max = np.full(S, -np.inf, dtype=np.float64)

    rms_g = RunningMeanVar(G) if G > 0 else None
    g_min = np.full(G, np.inf, dtype=np.float64)
    g_max = np.full(G, -np.inf, dtype=np.float64)
    glog_min = np.full(G, np.inf, dtype=np.float64)
    glog_max = np.full(G, -np.inf, dtype=np.float64)

    for p in tqdm(shards, desc="Computing stats"):
        with np.load(p) as z:
            y = np.asarray(z["y_mat"], dtype=np.float64)
            g = np.asarray(z["globals"], dtype=np.float64) if G > 0 else None

        ylog = np.log10(np.maximum(y, cfg.epsilon))
        y2 = ylog.reshape(-1, S)
        rms_logy.update(y2)
        logy_min = np.minimum(logy_min, np.min(y2, axis=0))
        logy_max = np.maximum(logy_max, np.max(y2, axis=0))

        if G > 0 and g is not None:
            g2 = g.reshape(-1, G)
            rms_g.update(g2)  # type: ignore[union-attr]
            g_min = np.minimum(g_min, np.min(g2, axis=0))
            g_max = np.maximum(g_max, np.max(g2, axis=0))

            glog = np.log10(np.maximum(g2, cfg.epsilon))
            glog_min = np.minimum(glog_min, np.min(glog, axis=0))
            glog_max = np.maximum(glog_max, np.max(glog, axis=0))

    mu_s, sd_s = rms_logy.finalize(min_std=cfg.min_std)

    mu_g: Optional[np.ndarray] = None
    sd_g: Optional[np.ndarray] = None
    if G > 0 and rms_g is not None:
        mu_g, sd_g = rms_g.finalize(min_std=cfg.min_std)

    per_key_stats: Dict[str, Dict] = {}
    for i, name in enumerate(cfg.species_variables):
        per_key_stats[name] = {
            "log_mean": float(mu_s[i]),
            "log_std": float(sd_s[i]),
            "log_min": float(logy_min[i]),
            "log_max": float(logy_max[i]),
            "epsilon": float(cfg.epsilon),
        }

    for i, name in enumerate(cfg.global_variables):
        if mu_g is None or sd_g is None:
            raise RuntimeError("Globals configured but global stats unavailable")
        per_key_stats[name] = {
            "mean": float(mu_g[i]),
            "std": float(sd_g[i]),
            "min": float(g_min[i]),
            "max": float(g_max[i]),
            "log_min": float(glog_min[i]),
            "log_max": float(glog_max[i]),
        }

    methods: Dict[str, str] = {}
    for s in cfg.species_variables:
        methods[s] = "log-standard"
    for g in cfg.global_variables:
        methods[g] = canonical_method(cfg.methods.get(g, cfg.globals_default_method))

    return per_key_stats, methods


def normalize_and_write(
    *,
    out_tmp: Path,
    out_final: Path,
    cfg: PreCfg,
    per_key_stats: Dict,
    methods: Dict[str, str],
) -> None:
    eps = float(cfg.epsilon)
    S = len(cfg.species_variables)
    G = len(cfg.global_variables)

    out_final.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        (out_final / split).mkdir(parents=True, exist_ok=True)

    mu_s = np.array([per_key_stats[s]["log_mean"] for s in cfg.species_variables], dtype=np.float64).reshape(1, 1, S)
    sd_s = np.array([per_key_stats[s]["log_std"] for s in cfg.species_variables], dtype=np.float64).reshape(1, 1, S)

    # Globals normalization vectors.
    method_ids = np.zeros(G, dtype=np.int64)  # 0=identity, 1=standard, 2=min-max, 3=log-min-max
    g_mean = np.zeros(G, dtype=np.float64)
    g_std = np.ones(G, dtype=np.float64)
    g_min = np.zeros(G, dtype=np.float64)
    g_rng = np.ones(G, dtype=np.float64)
    glog_min = np.zeros(G, dtype=np.float64)
    glog_rng = np.ones(G, dtype=np.float64)

    for j, name in enumerate(cfg.global_variables):
        m = canonical_method(methods[name])
        st = per_key_stats[name]

        if m == "identity":
            method_ids[j] = 0
            continue

        if m == "standard":
            method_ids[j] = 1
            g_mean[j] = float(st["mean"])
            g_std[j] = float(st["std"])
            if g_std[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive std")
            continue

        if m == "min-max":
            method_ids[j] = 2
            g_min[j] = float(st["min"])
            g_rng[j] = float(st["max"]) - g_min[j]
            if g_rng[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive range for min-max")
            continue

        if m == "log-min-max":
            method_ids[j] = 3
            glog_min[j] = float(st["log_min"])
            glog_rng[j] = float(st["log_max"]) - glog_min[j]
            if glog_rng[j] <= 0.0:
                raise ValueError(f"Global '{name}' has non-positive range for log-min-max")
            continue

        raise ValueError(f"Unsupported global normalization method '{m}' for '{name}'")

    # dt normalization: log10 + min-max to [0,1]
    dt_log_min = float(np.log10(cfg.dt_min))
    dt_log_max = float(np.log10(cfg.dt_max))
    dt_log_rng = dt_log_max - dt_log_min
    dt_is_constant = (dt_log_rng == 0.0)

    for split in ("train", "validation", "test"):
        physical_shards = iter_shards(out_tmp, split, "_physical")
        for i, p in enumerate(tqdm(physical_shards, desc=f"Normalizing {split}")):
            with np.load(p) as z:
                y = np.asarray(z["y_mat"], dtype=np.float64)
                g = np.asarray(z["globals"], dtype=np.float64) if G > 0 else None
                dt = np.asarray(z["dt_mat"], dtype=np.float64)

            y_z = ((np.log10(np.maximum(y, eps)) - mu_s) / sd_s).astype(np.float32)

            if G > 0 and g is not None:
                g2 = g.astype(np.float64, copy=False)

                g_out = g2

                # standard
                std_mask = method_ids == 1
                if np.any(std_mask):
                    g_out = np.where(std_mask.reshape(1, G), (g2 - g_mean.reshape(1, G)) / g_std.reshape(1, G), g_out)

                # min-max
                mm_mask = method_ids == 2
                if np.any(mm_mask):
                    g_out = np.where(mm_mask.reshape(1, G), (g2 - g_min.reshape(1, G)) / g_rng.reshape(1, G), g_out)

                # log-min-max
                lmm_mask = method_ids == 3
                if np.any(lmm_mask):
                    g_out = np.where(
                        lmm_mask.reshape(1, G),
                        (np.log10(np.maximum(g2, eps)) - glog_min.reshape(1, G)) / glog_rng.reshape(1, G),
                        g_out,
                    )

                g_z = g_out.astype(np.float32, copy=False)
            else:
                g_z = np.zeros((y.shape[0], 0), dtype=np.float32)

            if dt_is_constant:
                dt_norm = np.zeros_like(dt, dtype=np.float32)
            else:
                dt_norm = (np.log10(np.maximum(dt, eps)) - dt_log_min) / dt_log_rng
                dt_norm = np.clip(dt_norm, 0.0, 1.0).astype(np.float32)

            np.savez(out_final / split / f"shard_{i:06d}.npz", y_mat=y_z, globals=g_z, dt_norm_mat=dt_norm)

    manifest = {
        "normalization_methods": methods,
        "methods": methods,  # legacy alias
        "per_key_stats": per_key_stats,
        "epsilon": float(cfg.epsilon),
        "min_std": float(cfg.min_std),
        "dt": {"log_min": dt_log_min, "log_max": dt_log_max},
        "species_variables": list(cfg.species_variables),
        "global_variables": list(cfg.global_variables),
    }

    with open(out_final / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def write_summary(
    out_final: Path,
    *,
    cfg: PreCfg,
    counts_total: Dict[str, int],
    rejects_total: Dict[str, int],
) -> None:
    summary = {
        "counts_total": counts_total,
        "rejects_total": rejects_total,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
    }
    with open(out_final / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def default_config_path() -> Path:
    # Repository layout: <repo_root>/config.json and <repo_root>/src/preprocessing.py
    return Path(__file__).resolve().parents[1] / "config.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(default_config_path()), help="Path to config.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dict = load_json(cfg_path)

    sys_cfg = _require_dict(cfg_dict, "system")
    _configure_logging(_require_str(sys_cfg, "log_level"))

    cfg = parse_precfg(cfg_dict, cfg_path=cfg_path)

    raw_files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
    if not raw_files:
        raise FileNotFoundError(f"No raw HDF5 files found under {cfg.raw_dir} for patterns {cfg.raw_file_patterns}")

    out_final = cfg.processed_dir
    out_tmp = cfg.processed_dir / "_tmp_physical"

    clean_dir(out_tmp, overwrite=cfg.overwrite)
    clean_processed_outputs(out_final, overwrite=cfg.overwrite)

    counts_total = {"train": 0, "validation": 0, "test": 0}
    shard_idx = {"train": 0, "validation": 0, "test": 0}
    rejects_total: Dict[str, int] = {}

    rng = np.random.default_rng(cfg.seed)

    log.info(
        "Starting preprocessing raw_dir=%s processed_dir=%s files=%d",
        str(cfg.raw_dir),
        str(cfg.processed_dir),
        len(raw_files),
    )

    t0 = time.time()

    for fp in raw_files:
        log.info("Processing %s", fp.name)
        rej = sample_file(
            fp,
            out_tmp=out_tmp,
            cfg=cfg,
            rng=rng,
            counts_total=counts_total,
            shard_idx=shard_idx,
        )
        for k, v in rej.items():
            rejects_total[k] = rejects_total.get(k, 0) + int(v)

    total_written = counts_total["train"] + counts_total["validation"] + counts_total["test"]
    if total_written == 0:
        raise RuntimeError("No trajectories were written. Check rejects_total for the failure mode.")

    log.info(
        "Totals written=%d (train=%d val=%d test=%d)",
        total_written,
        counts_total["train"],
        counts_total["validation"],
        counts_total["test"],
    )

    per_key_stats, methods = compute_train_stats(out_tmp, cfg)
    normalize_and_write(out_tmp=out_tmp, out_final=out_final, cfg=cfg, per_key_stats=per_key_stats, methods=methods)
    write_summary(out_final, cfg=cfg, counts_total=counts_total, rejects_total=rejects_total)

    log.info("Done in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
