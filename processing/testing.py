#!/usr/bin/env python3
"""
testing.py (or plot.py)

3x2 panel figure:
- 3 rows: 3 different randomly selected raw trajectories (groups)
- Left column: log-x + log-y (loglog), overlay all species (raw vs chunk)
- Right column: linear-x + log-y (semilog-y), overlay all species (raw vs chunk),
  but the x-axis is restricted EXACTLY to the extent of the blue (regridded) points.

Overlay style (matches original):
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
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

# If None, uses first *.h5/*.hdf5 under config.paths.raw_data_dir
RAW_FILE: Optional[Path] = None

# Number of rows (trajectories) and layout
N_ROWS = 3
N_COLS = 2

# Random selection seed for trajectories
SEED = 0

# Chunk generation semantics
USE_ANCHOR_FIRST_CHUNK = True      # match preprocessing: for "first chunk", try t_start = t_lo
CHUNK_T_START_ATTEMPTS = 500       # fallback attempts if anchor doesn't fit

# Output folder/file
FIGURES_DIR = PROJECT_ROOT / "figures"   # same level as src/
OUT_PNG = FIGURES_DIR / "chunk_vs_raw_panels.png"
DPI = 250


# =============================================================================
# Helpers copied to match the exact preprocessing.py you pasted
# =============================================================================

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
class PlotCfg:
    raw_dir: Path
    dt: float
    n_steps: int
    t_min: float
    drop_below: float
    time_keys: List[str]
    species_variables: List[str]


def load_plotcfg(cfg: Dict) -> PlotCfg:
    pr = cfg.get("preprocessing", {})
    dcfg = cfg.get("data", {})
    pcfg = cfg.get("paths", {})

    raw_dir = Path(pcfg.get("raw_data_dir", PROJECT_ROOT / "data" / "raw"))
    if not raw_dir.is_absolute():
        raw_dir = (PROJECT_ROOT / raw_dir).resolve()

    return PlotCfg(
        raw_dir=raw_dir,
        dt=float(pr.get("dt", 100.0)),
        n_steps=int(pr.get("n_steps", 1000)),
        t_min=float(pr.get("t_min", 1e-3)),
        drop_below=float(pr.get("drop_below", 1e-35)),
        time_keys=list(pr.get("time_keys", ["t_time", "time", "t"])),
        species_variables=list(dcfg.get("species_variables", [])),
    )


def pick_raw_file(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under raw_dir={raw_dir}")
    return files[0]


def list_group_names(f_raw: h5py.File) -> List[str]:
    return [k for k in f_raw.keys() if isinstance(f_raw[k], h5py.Group)]


def build_raw_and_chunk_from_preprocessing_logic(
    raw: h5py.Group,
    pcfg: PlotCfg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_raw   : [Traw]
      y_raw   : [Traw, S]
      t_chunk : [n_steps] (absolute)
      y_chunk : [n_steps, S]
    """
    leaf_index = build_leaf_index(raw)

    _, t_raw = get_time_array_recursive(raw, pcfg.time_keys, leaf_index)
    if t_raw is None:
        raise RuntimeError("No time array found (direct or nested).")
    T = int(t_raw.shape[0])

    # Resolve species paths in config order
    species_paths: List[str] = []
    for s in pcfg.species_variables:
        p = find_dataset_path(raw, s, leaf_index, t_len=T, prefer_time_aligned=True)
        if p is None:
            leaves = sorted(list(leaf_index.keys()))
            preview = ", ".join(leaves[:120])
            raise RuntimeError(
                f"Missing species '{s}'. Available dataset leaf names (first 120): {preview}"
            )
        species_paths.append(p)

    S = len(species_paths)
    y_raw = np.empty((T, S), dtype=np.float64)
    for j, p in enumerate(species_paths):
        arr = np.asarray(raw[p][...], dtype=np.float64)
        if not arr.shape or int(arr.shape[0]) != T:
            raise RuntimeError(f"Dataset '{p}' not time-aligned with t_raw (shape={arr.shape}, T={T})")
        y_raw[:, j] = arr.reshape(T, -1)[:, 0]

    # Match preprocessing validity logic
    pos = t_raw > 0
    if not np.any(pos):
        raise RuntimeError("All times are <= 0.")
    first_pos_time = float(t_raw[pos][0])
    t_lo = max(pcfg.t_min, first_pos_time)

    valid = (t_raw > 0) & (t_raw >= t_lo * 0.5)
    if not np.any(valid):
        raise RuntimeError(f"No valid times after filter (t_lo={t_lo}).")

    t_valid = t_raw[valid]
    y_valid = y_raw[valid, :]

    # Chunk offsets
    chunk_offsets = np.arange(pcfg.n_steps, dtype=np.float64) * pcfg.dt
    chunk_duration = float(chunk_offsets[-1])

    t_end = float(t_raw[-1])
    t_hi = t_end - chunk_duration
    if t_hi <= t_lo:
        raise RuntimeError(
            f"Too short to fit chunk: t_end={t_end:.3e}, t_lo={t_lo:.3e}, "
            f"chunk_dur={chunk_duration:.3e} => t_hi={t_hi:.3e} <= t_lo"
        )

    def fits(start: float) -> bool:
        t_chunk_local = start + chunk_offsets
        return (t_chunk_local[0] >= t_valid[0]) and (t_chunk_local[-1] <= t_valid[-1])

    # Try anchor first (if configured)
    if USE_ANCHOR_FIRST_CHUNK:
        t_start = t_lo
        if not fits(t_start):
            t_start = None
    else:
        t_start = None

    # Fallback: random log-uniform t_start until it fits
    if t_start is None:
        ok = False
        for _ in range(CHUNK_T_START_ATTEMPTS):
            cand = 10.0 ** float(rng.uniform(np.log10(t_lo), np.log10(t_hi)))
            if fits(cand):
                t_start = cand
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Could not find a valid t_start after {CHUNK_T_START_ATTEMPTS} attempts. "
                f"Valid t range: [{t_valid[0]:.3e}, {t_valid[-1]:.3e}]  t_lo={t_lo:.3e}  t_hi={t_hi:.3e}"
            )

    t_chunk = t_start + chunk_offsets

    # Interpolate in log-time/log-y (same as preprocessing core)
    log_t_valid = np.log10(t_valid)
    log_t_chunk = np.log10(t_chunk)
    i0, i1, w = _prepare_log_interp(log_t_valid, log_t_chunk)
    y_chunk = interp_loglog_species(y_valid, i0, i1, w)

    return t_raw, y_raw, t_chunk, y_chunk


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
) -> None:
    raw_pts_per_species = int(np.sum(t_raw > 0))
    chunk_pts = int(len(t_chunk))
    dt = (t_chunk[1] - t_chunk[0]) if len(t_chunk) > 1 else float("nan")

    raw_label = f"Raw ({raw_pts_per_species} pts/species, N={species_count})"
    chunk_label = f"Chunk ({chunk_pts} pts, dt={dt:.0f})"

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
    pcfg = load_plotcfg(cfg)
    if not pcfg.species_variables:
        raise RuntimeError("data.species_variables is empty; cannot plot species curves.")

    raw_file = RAW_FILE if RAW_FILE is not None else pick_raw_file(pcfg.raw_dir)
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

        # Choose N_ROWS groups that look minimally valid
        chosen: List[str] = []
        used = set()
        max_tries = max(10 * N_ROWS, 50)

        tries = 0
        while len(chosen) < N_ROWS and tries < max_tries:
            tries += 1
            gname = group_names[int(rng.integers(0, len(group_names)))]
            if gname in used:
                continue
            used.add(gname)

            try:
                raw = f_raw[gname]
                leaf_index = build_leaf_index(raw)
                _, t_raw0 = get_time_array_recursive(raw, pcfg.time_keys, leaf_index)
                if t_raw0 is None:
                    continue
                T0 = int(t_raw0.shape[0])
                p0 = find_dataset_path(raw, pcfg.species_variables[0], leaf_index, t_len=T0, prefer_time_aligned=True)
                if p0 is None:
                    continue
                chosen.append(gname)
            except Exception:
                continue

        if len(chosen) < N_ROWS:
            raise RuntimeError(f"Could only find {len(chosen)} valid trajectories (wanted {N_ROWS}).")

        fig, axes = plt.subplots(
            N_ROWS, N_COLS,
            figsize=(14.0, 4.2 * N_ROWS),
            squeeze=False,
        )

        for r, gname in enumerate(chosen):
            raw = f_raw[gname]

            try:
                t_raw, y_raw, t_chunk, y_chunk = build_raw_and_chunk_from_preprocessing_logic(raw, pcfg, rng)
            except Exception as e:
                raise RuntimeError(f"Failed building chunk for group '{gname}': {e}") from e

            # Left: log-x (and log-y)
            axL = axes[r][0]
            style_axes(axL, xlog=True)
            axL.set_title(f"{gname} (log-x)", pad=10)
            plot_overlay_all_species(
                axL, t_raw, y_raw, t_chunk, y_chunk,
                legend=True, species_count=len(pcfg.species_variables),
            )

            # Right: linear-x (log-y) BUT x-range exactly equals blue-dot extent
            axR = axes[r][1]
            style_axes(axR, xlog=False)
            axR.set_title(f"{gname} (linear-x; xlim=chunk extent)", pad=10)
            plot_overlay_all_species(
                axR, t_raw, y_raw, t_chunk, y_chunk,
                legend=True, species_count=len(pcfg.species_variables),
            )
            # Restrict x-axis EXACTLY to the chunk extent (blue dots)
            axR.set_xlim(float(t_chunk[0]), float(t_chunk[-1]))

        fig.suptitle(
            f"Raw vs Generated Chunk Overlay (dt={pcfg.dt:g}, n_steps={pcfg.n_steps})\n"
            f"Raw file: {raw_file.name}",
            fontsize=11,
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.965])

        fig.savefig(OUT_PNG, dpi=DPI)
        plt.close(fig)

        print(f"Wrote: {OUT_PNG.resolve()}")
        print(f"Rows (trajectories): {chosen}")


if __name__ == "__main__":
    main()
