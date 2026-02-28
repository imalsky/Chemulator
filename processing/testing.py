#!/usr/bin/env python3
"""
testing.py

3x2 panel figure:
- 3 rows: 3 different randomly selected raw trajectories (groups)
- Left column: log-x + log-y (loglog), overlay all species (raw vs chunk)
- Right column: linear-x + log-y (semilog-y), overlay all species (raw vs chunk),
  with x-axis restricted to the extent of the blue (regridded) points.

Overlay style:
- raw: red circles ('ro', ms=8)
- chunk: blue squares ('bs', ms=5)
- alpha=0.35
- legend entries for Raw / Chunk

No argparse. Configure via globals below.
Outputs into: <repo_root>/figures/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS (EDIT THESE)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

# If None, selects first matching file under config.paths.raw_dir using
# config.preprocessing.raw_file_patterns.
RAW_FILE: Optional[Path] = None

# Number of rows (trajectories) and layout
N_ROWS = 3
N_COLS = 2

# Random selection seed for trajectories
SEED = 1

# Chunk generation semantics (mirrors preprocessing.py pick_t_start behavior)
USE_ANCHOR_FIRST_CHUNK = True
CHUNK_T_START_ATTEMPTS = 500  # random t_start attempts (after anchor)

# If set, overrides dt selection regardless of config preprocessing.dt_mode
DT_OVERRIDE: Optional[float] = None

# Whether to apply preprocessing.drop_below rejection (strictly matches preprocessing).
# If True, any trajectory with any raw y < drop_below is skipped.
APPLY_DROP_BELOW = True

# Output folder/file
FIGURES_DIR = PROJECT_ROOT / "figures"
OUT_PNG = FIGURES_DIR / "chunk_vs_raw_panels.png"
DPI = 250

# Safety: max groups to attempt while searching for N_ROWS valid ones
MAX_GROUP_TRIES = 5000


# =============================================================================
# Config + raw reading helpers (match preprocessing.py semantics)
# =============================================================================

def load_json_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a JSON object: {path}")
    return obj


def leaf_dataset_index(grp: h5py.Group) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def _unique_dataset_path(grp: h5py.Group, leaf_index: Dict[str, List[str]], key: str) -> str:
    """
    Mirror preprocessing._unique_dataset_path():
      - If key contains '/', treat as full path.
      - Else match by leaf name; require exactly one match.
    """
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
    species_vars: List[str],
    leaf_index: Dict[str, List[str]],
) -> np.ndarray:
    s = len(species_vars)
    y = np.empty((t_len, s), dtype=np.float64)

    for j, name in enumerate(species_vars):
        p = _unique_dataset_path(grp, leaf_index, name)
        arr = np.asarray(grp[p][...], dtype=np.float64)

        if arr.ndim == 0 or int(arr.shape[0]) != int(t_len):
            raise ValueError(
                f"Species dataset '{name}' must be time-aligned with length {t_len}, got {arr.shape}"
            )

        col = arr.reshape(t_len, -1)[:, 0]
        if not np.all(np.isfinite(col)):
            raise ValueError(f"Species dataset '{name}' contains non-finite values")
        y[:, j] = col

    return y


# =============================================================================
# Chunk placement + interpolation (match preprocessing.py semantics)
# =============================================================================

def prepare_log_interp(log_t_valid: np.ndarray, log_t_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror preprocessing.prepare_log_interp(): strict denom > 0.
    """
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
    """
    Mirror preprocessing.interp_loglog(): floor at epsilon (NOT 1e-300).
    """
    y0 = np.log10(np.maximum(y_valid[i0, :], epsilon))
    y1 = np.log10(np.maximum(y_valid[i1, :], epsilon))
    w2 = w.reshape(-1, 1)
    return 10.0 ** ((1.0 - w2) * y0 + w2 * y1)


def sample_dt(cfg: "PlotCfg", rng: np.random.Generator) -> float:
    if DT_OVERRIDE is not None:
        if DT_OVERRIDE <= 0:
            raise ValueError(f"DT_OVERRIDE must be > 0, got {DT_OVERRIDE}")
        return float(DT_OVERRIDE)

    if cfg.dt_mode == "fixed":
        return float(cfg.dt)

    # per_chunk
    if cfg.dt_sampling == "loguniform":
        lo = np.log10(cfg.dt_min)
        hi = np.log10(cfg.dt_max)
        return float(10.0 ** rng.uniform(lo, hi))

    # uniform
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
    """
    Mirror preprocessing.pick_t_start().
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


# =============================================================================
# Plot config
# =============================================================================

@dataclass(frozen=True)
class PlotCfg:
    raw_dir: Path
    raw_file_patterns: List[str]

    dt: float
    dt_mode: str
    dt_min: float
    dt_max: float
    dt_sampling: str

    n_steps: int
    t_min: float
    drop_below: float

    time_key: str
    species_variables: List[str]

    epsilon: float


def _resolve_path(root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_plotcfg(cfg: Dict, *, cfg_path: Path) -> PlotCfg:
    root = cfg_path.parent.resolve()

    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("config.paths must be an object")

    # Prefer current schema key raw_dir; fall back to raw_data_dir if present.
    raw_dir_key = None
    if "raw_dir" in paths:
        raw_dir_key = "raw_dir"
    elif "raw_data_dir" in paths:
        raw_dir_key = "raw_data_dir"
    if raw_dir_key is None:
        raise KeyError("config.paths must contain 'raw_dir' (or legacy 'raw_data_dir')")

    raw_dir = _resolve_path(root, str(paths[raw_dir_key]))

    data = cfg.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("config.data must be an object")
    species_variables = data.get("species_variables", [])
    if not isinstance(species_variables, list) or not all(isinstance(x, str) and x.strip() for x in species_variables):
        raise TypeError("config.data.species_variables must be a non-empty list of strings")
    species_variables = [str(x).strip() for x in species_variables]

    pr = cfg.get("preprocessing", {})
    if not isinstance(pr, dict):
        raise TypeError("config.preprocessing must be an object")

    raw_file_patterns = pr.get("raw_file_patterns", ["*.h5", "*.hdf5"])
    if isinstance(raw_file_patterns, str):
        raw_file_patterns = [raw_file_patterns]
    if (
        not isinstance(raw_file_patterns, list)
        or not raw_file_patterns
        or not all(isinstance(x, str) and x.strip() for x in raw_file_patterns)
    ):
        raise TypeError("config.preprocessing.raw_file_patterns must be a non-empty list of strings")
    raw_file_patterns = [str(x).strip() for x in raw_file_patterns]

    time_key = pr.get("time_key", None)
    if not isinstance(time_key, str) or not time_key.strip():
        # legacy: time_keys list
        time_keys = pr.get("time_keys", None)
        if isinstance(time_keys, list) and time_keys and isinstance(time_keys[0], str) and time_keys[0].strip():
            time_key = str(time_keys[0]).strip()
        else:
            raise KeyError("config.preprocessing.time_key is missing (and no usable legacy time_keys found)")
    time_key = str(time_key).strip()

    dt = float(pr.get("dt", 100.0))
    dt_mode = str(pr.get("dt_mode", "fixed")).lower().strip()
    dt_min = float(pr.get("dt_min", dt))
    dt_max = float(pr.get("dt_max", dt))
    dt_sampling = str(pr.get("dt_sampling", "uniform")).lower().strip()
    n_steps = int(pr.get("n_steps", 300))
    t_min = float(pr.get("t_min", 0.0))
    drop_below = float(pr.get("drop_below", 1e-35))

    norm = cfg.get("normalization", {})
    if not isinstance(norm, dict):
        raise TypeError("config.normalization must be an object")
    epsilon = float(norm.get("epsilon", 1e-30))
    if epsilon <= 0.0:
        raise ValueError(f"normalization.epsilon must be > 0, got {epsilon}")

    if dt <= 0.0:
        raise ValueError(f"preprocessing.dt must be > 0, got {dt}")
    if n_steps < 2:
        raise ValueError(f"preprocessing.n_steps must be >= 2, got {n_steps}")
    if dt_min <= 0.0 or dt_max <= 0.0 or dt_max < dt_min:
        raise ValueError(f"Invalid dt_min/dt_max: require 0 < dt_min <= dt_max (got {dt_min}, {dt_max})")
    if dt_mode not in ("fixed", "per_chunk"):
        raise ValueError("preprocessing.dt_mode must be 'fixed' or 'per_chunk'")
    if dt_sampling not in ("uniform", "loguniform"):
        raise ValueError("preprocessing.dt_sampling must be 'uniform' or 'loguniform'")

    return PlotCfg(
        raw_dir=raw_dir,
        raw_file_patterns=raw_file_patterns,
        dt=dt,
        dt_mode=dt_mode,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_sampling=dt_sampling,
        n_steps=n_steps,
        t_min=t_min,
        drop_below=drop_below,
        time_key=time_key,
        species_variables=species_variables,
        epsilon=epsilon,
    )


def pick_raw_file(raw_dir: Path, patterns: List[str]) -> Path:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(raw_dir.glob(pat)))
    uniq = sorted({p.resolve() for p in files})
    if not uniq:
        raise FileNotFoundError(f"No HDF5 files found under raw_dir={raw_dir} for patterns={patterns}")
    return uniq[0]


def list_group_names(f_raw: h5py.File) -> List[str]:
    return [k for k in f_raw.keys() if isinstance(f_raw[k], h5py.Group)]


def build_raw_and_chunk_from_preprocessing_logic(
    raw: h5py.Group,
    pcfg: PlotCfg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      t_raw   : [Traw]
      y_raw   : [Traw, S]
      t_chunk : [n_steps] absolute times
      y_chunk : [n_steps, S] interpolated (floored at epsilon in log space)
      dt_s    : scalar dt used for the chunk
    """
    leaf_index = leaf_dataset_index(raw)

    t_raw = read_time(raw, time_key=pcfg.time_key, leaf_index=leaf_index)
    T = int(t_raw.shape[0])

    y_raw = read_species_matrix(raw, t_len=T, species_vars=pcfg.species_variables, leaf_index=leaf_index)

    # Apply preprocessing.drop_below rejection (optional; default True to match preprocessing)
    if APPLY_DROP_BELOW:
        if np.any(y_raw < float(pcfg.drop_below)):
            raise RuntimeError(f"Rejected by drop_below: min(y_raw)={float(np.min(y_raw)):.3e} < {pcfg.drop_below:.3e}")

    # Valid mask (mirrors preprocessing.sample_file)
    pos = t_raw > 0
    if not np.any(pos):
        raise RuntimeError("Rejected: all times <= 0")

    first_pos_time = float(t_raw[pos][0])
    t_lo = max(float(pcfg.t_min), first_pos_time)

    valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
    if int(np.count_nonzero(valid)) < 2:
        raise RuntimeError(f"Rejected: too few valid samples after time filter (t_lo={t_lo:.3e})")

    t_valid = t_raw[valid]
    y_valid = y_raw[valid, :]

    # Sample dt (mirrors preprocessing.sample_dt unless DT_OVERRIDE is set)
    dt_s = sample_dt(pcfg, rng)

    # Choose chunk start
    t_start = pick_t_start(
        t_raw=t_raw,
        t_valid=t_valid,
        dt_s=dt_s,
        n_steps=pcfg.n_steps,
        t_min=pcfg.t_min,
        rng=rng,
        anchor_first=USE_ANCHOR_FIRST_CHUNK,
        max_attempts=CHUNK_T_START_ATTEMPTS,
    )
    if t_start is None:
        raise RuntimeError("Rejected: could not fit a chunk (no_fit_chunk)")

    t_chunk = float(t_start) + np.arange(pcfg.n_steps, dtype=np.float64) * float(dt_s)

    # Interpolate in log-time/log-y (mirrors preprocessing.interp_loglog)
    log_t_valid = np.log10(t_valid)
    log_t_chunk = np.log10(t_chunk)
    i0, i1, w = prepare_log_interp(log_t_valid, log_t_chunk)
    y_chunk = interp_loglog(y_valid, i0=i0, i1=i1, w=w, epsilon=pcfg.epsilon)

    if not np.all(np.isfinite(y_chunk)):
        raise RuntimeError("Rejected: interp produced non-finite values")

    # Defensive assertion: with epsilon flooring, chunk should never drop below epsilon
    if np.any(y_chunk < float(pcfg.epsilon)):
        mn = float(np.min(y_chunk))
        raise RuntimeError(f"BUG: y_chunk dropped below epsilon (min={mn:.3e} < eps={pcfg.epsilon:.3e})")

    return t_raw, y_raw, t_chunk, y_chunk, float(dt_s)


# =============================================================================
# Plotting
# =============================================================================

def style_axes(ax, *, xlog: bool) -> None:
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_yscale("log")
    ax.set_xscale("log" if xlog else "linear")
    ax.set_xlabel("Time", labelpad=6)
    ax.set_ylabel("Mixing ratio", labelpad=6)


def plot_overlay_all_species(
    ax,
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    t_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    legend: bool,
    species_count: int,
    dt_s: float,
) -> None:
    raw_pts_per_species = int(np.sum(t_raw > 0))
    chunk_pts = int(len(t_chunk))

    raw_label = f"Raw ({raw_pts_per_species} pts/species, N={species_count})"
    chunk_label = f"Chunk ({chunk_pts} pts, dt={dt_s:.0f})"

    did_raw_label = False
    did_chunk_label = False

    S = int(y_raw.shape[1])
    for j in range(S):
        yr = y_raw[:, j].astype(float, copy=False)
        yc = y_chunk[:, j].astype(float, copy=False)

        m_raw = (t_raw > 0) & (yr > 0)
        m_chunk = (t_chunk > 0) & (yc > 0)

        ax.plot(
            t_raw[m_raw],
            yr[m_raw],
            "ro",
            ms=8,
            alpha=0.35,
            label=(raw_label if (legend and not did_raw_label) else None),
            rasterized=True,
        )
        did_raw_label = True

        ax.plot(
            t_chunk[m_chunk],
            yc[m_chunk],
            "bs",
            ms=5,
            alpha=0.35,
            label=(chunk_label if (legend and not did_chunk_label) else None),
            rasterized=True,
        )
        did_chunk_label = True

    if legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    cfg = load_json_config(CONFIG_PATH)
    pcfg = load_plotcfg(cfg, cfg_path=CONFIG_PATH)
    if not pcfg.species_variables:
        raise RuntimeError("data.species_variables is empty; cannot plot species curves.")

    raw_file = RAW_FILE if RAW_FILE is not None else pick_raw_file(pcfg.raw_dir, pcfg.raw_file_patterns)
    if not raw_file.is_absolute():
        raw_file = (PROJECT_ROOT / raw_file).resolve()
    if not raw_file.exists():
        raise FileNotFoundError(f"RAW_FILE not found: {raw_file}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    with h5py.File(raw_file, "r") as f_raw:
        group_names = list_group_names(f_raw)
        if not group_names:
            raise RuntimeError(f"No groups found at root of raw file: {raw_file}")

        samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        used: set[str] = set()
        reject_reasons: Dict[str, int] = {}

        tries = 0
        while len(samples) < N_ROWS and tries < MAX_GROUP_TRIES:
            tries += 1
            gname = group_names[int(rng.integers(0, len(group_names)))]
            if gname in used:
                continue
            used.add(gname)

            raw_grp = f_raw[gname]
            try:
                t_raw, y_raw, t_chunk, y_chunk, dt_s = build_raw_and_chunk_from_preprocessing_logic(raw_grp, pcfg, rng)
            except Exception as e:
                key = str(e).split(":")[0].strip() if str(e).strip() else e.__class__.__name__
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                continue

            samples.append((gname, t_raw, y_raw, t_chunk, y_chunk, dt_s))

        if len(samples) < N_ROWS:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:12])
            raise RuntimeError(
                f"Could only find {len(samples)} valid trajectories (wanted {N_ROWS}) after {tries} tries. "
                f"Top reject reasons: {summary}"
            )

        fig, axes = plt.subplots(
            N_ROWS,
            N_COLS,
            figsize=(14.0, 4.2 * N_ROWS),
            squeeze=False,
        )

        for r, (gname, t_raw, y_raw, t_chunk, y_chunk, dt_s) in enumerate(samples):
            # Left: log-x (and log-y)
            axL = axes[r][0]
            style_axes(axL, xlog=True)
            axL.set_title(f"{gname} (log-x)", pad=10)
            plot_overlay_all_species(
                axL,
                t_raw,
                y_raw,
                t_chunk,
                y_chunk,
                legend=True,
                species_count=len(pcfg.species_variables),
                dt_s=dt_s,
            )

            # Right: linear-x (log-y) BUT x-range exactly equals blue-dot extent
            axR = axes[r][1]
            style_axes(axR, xlog=False)
            axR.set_title(f"{gname} (linear-x; xlim=chunk extent)", pad=10)
            plot_overlay_all_species(
                axR,
                t_raw,
                y_raw,
                t_chunk,
                y_chunk,
                legend=True,
                species_count=len(pcfg.species_variables),
                dt_s=dt_s,
            )
            axR.set_xlim(float(t_chunk[0]), float(t_chunk[-1]))

        fig.suptitle(
            f"Raw vs Generated Chunk Overlay (dt_mode={pcfg.dt_mode}, n_steps={pcfg.n_steps}, eps={pcfg.epsilon:.1e})\n"
            f"Raw file: {raw_file.name}",
            fontsize=11,
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.965])

        fig.savefig(OUT_PNG, dpi=DPI)
        plt.close(fig)

        print(f"Wrote: {OUT_PNG.resolve()}")
        print("Rows (trajectories):", [s[0] for s in samples])
        if reject_reasons:
            top = ", ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:10])
            print("Top reject reasons:", top)


if __name__ == "__main__":
    main()