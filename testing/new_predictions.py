#!/usr/bin/env python3
"""
testing/predictions.py

Autoregressive rollout evaluation using the exported 1-step *physical-space* model (.pt2).

The export produced by `testing/new_export.py` has signature:

    y_next_phys = model(y_phys, dt_seconds, g_phys)

with:
  - y_phys     : [B, S]  (species in physical units; positive)
  - dt_seconds : [B]     (seconds; positive)
  - g_phys     : [B, G]  (globals in physical units; shape can be [B, 0])

This script evaluates the export against the *processed* test split in `data/processed/`.
The processed shards store:
  - y_mat       : [N, T, S]  species in z-space (normalized log-space)
  - globals     : [N, G]     globals in z-space (normalized)
  - dt_norm_mat : [N, T-1]   per-step dt in normalized [0, 1] log10-minmax space

We invert normalization via `data/processed/normalization.json` to obtain physical-space
inputs/ground-truth, then run an open-loop (autoregressive) rollout using the export.

No CLI arguments. Adjust the global settings below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

# Matplotlib is only used for saving a PNG; use a non-interactive backend.
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


# =============================================================================
# User settings (globals only)
# =============================================================================

# Repo layout:
#   <root>/
#     src/
#     testing/   <-- this file
#     data/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Training run directory containing the exported artifact and where plots are saved.
RUN_DIR: Path = (ROOT / "models" / "v1").resolve()

# Export artifact produced by testing/new_export.py
EXPORT_NAME: str = "export_cpu_dynB_1step_phys.pt2"

# Split/shard/sample selection inside data/processed/<split>/shard_*.npz
SPLIT: str = "test"          # "test" | "validation" | "train" (train is usually not what you want here)
SHARD_INDEX: int = 0         # which shard file (sorted lexicographically)
SAMPLE_INDEX: int = 3        # which trajectory inside the shard (0 <= idx < N)
START_INDEX: int = 1         # starting state index inside the trajectory (0 <= idx < T-1)
N_STEPS: int = 498           # number of autoregressive 1-step predictions

# dt handling:
# - If True: use CONSTANT_DT_SECONDS for every step.
# - If False: use per-step dt from dt_norm_mat (recommended; works for fixed or varying dt).
USE_CONSTANT_DT: bool = False
CONSTANT_DT_SECONDS: float = 100.0

# Plot/output
PLOT_Y_LIM: Tuple[float, float] = (1e-30, 3.0)
PLOT_MARKER_EVERY: int = 10     # plot every Nth prediction marker (1 => all)
PLOT_STYLE: str | None = "science.mplstyle"  # None disables custom style

# Inference
DEVICE: torch.device = torch.device("cpu")
DTYPE: torch.dtype = torch.float32


# =============================================================================
# Small helpers
# =============================================================================

def _try_style(style: str | None) -> None:
    if not style:
        return

    # 1) Matplotlib named style
    try:
        plt.style.use(style)
        return
    except Exception:
        pass

    # 2) Style file relative to repo root (common patterns)
    for rel in (style, f"misc/{style}"):
        p = (ROOT / rel).resolve()
        if p.exists():
            try:
                plt.style.use(str(p))
                return
            except Exception:
                pass


def _load_export_module(export_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a torch.export artifact and its embedded metadata.json.

    Returns:
      (module, metadata_dict)
    """
    extra_files = {"metadata.json": ""}
    ep = torch.export.load(str(export_path), extra_files=extra_files)
    meta_raw = extra_files.get("metadata.json", "") or ""
    meta: Dict[str, Any] = json.loads(meta_raw) if meta_raw else {}

    module = ep.module().to(device=DEVICE)
    #module.eval()
    return module, meta


def _resolve_normalization_path(meta: Mapping[str, Any]) -> Path:
    """
    Best-effort location of data/processed/normalization.json.

    Primary: metadata["normalization_path"] (saved at export time).
    Fallback: <repo_root>/data/processed/normalization.json
    """
    cand: Path | None = None

    raw = meta.get("normalization_path")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw).expanduser()
        if not p.is_absolute():
            # Treat relative to repo root (portable across machines).
            p = (ROOT / p).resolve()
        if p.exists():
            cand = p

    if cand is None:
        p2 = (ROOT / "data" / "processed" / "normalization.json").resolve()
        if p2.exists():
            cand = p2

    if cand is None:
        raise FileNotFoundError(
            "Could not locate normalization.json. "
            "Tried export metadata normalization_path and ROOT/data/processed/normalization.json."
        )

    return cand


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path} ({type(e).__name__}: {e})") from e


def _pick_methods_map(manifest: Mapping[str, Any]) -> Mapping[str, str]:
    methods = manifest.get("methods") or manifest.get("normalization_methods") or {}
    if not isinstance(methods, Mapping):
        return {}
    return methods  # type: ignore[return-value]


def _canonical_method(method: str) -> str:
    m = str(method).lower().strip().replace("_", "-")
    if m in ("minmax", "min-max"):
        return "min-max"
    if m in ("logminmax", "log-min-max"):
        return "log-min-max"
    if m in ("log10-standard", "log10-standard"):
        return "log-standard"
    if m in ("none", "", "identity"):
        return "identity"
    if m == "standard":
        return "standard"
    return m


def _dt_norm_to_seconds(dt_norm: np.ndarray, manifest: Mapping[str, Any]) -> np.ndarray:
    dt = manifest.get("dt") or {}
    if not isinstance(dt, Mapping) or "log_min" not in dt or "log_max" not in dt:
        raise KeyError("normalization.json missing dt.log_min / dt.log_max")

    a = float(dt["log_min"])
    b = float(dt["log_max"])
    log_dt = dt_norm.astype(np.float64) * (b - a) + a
    return np.power(10.0, log_dt, dtype=np.float64)


def _denormalize_species(y_z: np.ndarray, species_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    """
    z-space -> physical for species.

    Supports:
      - log-standard (expected for this repo)
      - log-min-max  (supported defensively)
    """
    stats = manifest.get("per_key_stats") or {}
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")

    methods = _pick_methods_map(manifest)

    out = np.empty_like(y_z, dtype=np.float64)
    for j, key in enumerate(species_keys):
        st = stats[key]
        m = _canonical_method(str(methods.get(key, "log-standard")))

        zj = y_z[..., j].astype(np.float64)

        if m == "log-standard":
            mu = float(st["log_mean"])
            sd = float(st["log_std"])
            out[..., j] = np.power(10.0, zj * sd + mu, dtype=np.float64)
        elif m == "log-min-max":
            lo = float(st["log_min"])
            hi = float(st["log_max"])
            out[..., j] = np.power(10.0, zj * (hi - lo) + lo, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported species normalization method '{m}' for key '{key}'")

    return out


def _denormalize_globals(g_z: np.ndarray, global_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    """
    z-space -> physical for globals.

    Supports:
      - identity
      - standard
      - min-max
      - log-min-max
      - log-standard (rare; included for completeness)
    """
    if len(global_keys) == 0:
        return np.empty((0,), dtype=np.float64)

    stats = manifest.get("per_key_stats") or {}
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")

    methods = _pick_methods_map(manifest)

    out = np.empty((len(global_keys),), dtype=np.float64)
    for j, key in enumerate(global_keys):
        st = stats[key]
        m = _canonical_method(str(methods.get(key, manifest.get("default_method", "standard"))))
        z = float(g_z[j])

        if m == "identity":
            out[j] = float(z)

        elif m == "standard":
            mu = float(st.get("mean", 0.0))
            sd = float(st.get("std", 1.0))
            out[j] = z * sd + mu

        elif m == "min-max":
            lo = float(st.get("min", 0.0))
            hi = float(st.get("max", 1.0))
            out[j] = z * (hi - lo) + lo

        elif m == "log-min-max":
            lo = float(st.get("log_min", 0.0))
            hi = float(st.get("log_max", 1.0))
            out[j] = 10.0 ** (z * (hi - lo) + lo)

        elif m == "log-standard":
            mu = float(st.get("log_mean", 0.0))
            sd = float(st.get("log_std", 1.0))
            out[j] = 10.0 ** (z * sd + mu)

        else:
            raise ValueError(f"Unsupported global normalization method '{m}' for key '{key}'")

    return out


def _load_shard(processed_dir: Path, *, split: str, shard_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load y_z, g_z, dt_norm for a shard.

    Returns:
      y_mat       [N, T, S]  float32
      g_mat       [N, G]     float32
      dt_norm_mat [N, T-1]   float32
    """
    shard_dir = processed_dir / split
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {shard_dir}")

    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"SHARD_INDEX out of range: {shard_index} (found {len(shards)} shards)")

    p = shards[shard_index]
    with np.load(p) as f:
        y_mat = f["y_mat"].astype(np.float32, copy=False)
        g_mat = f["globals"].astype(np.float32, copy=False)
        dt_norm_mat = f["dt_norm_mat"].astype(np.float32, copy=False)

    return y_mat, g_mat, dt_norm_mat


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    *,
    y0_phys: np.ndarray,            # [S]
    g_phys: np.ndarray,             # [G]
    dt_seconds: np.ndarray,         # [K]
) -> np.ndarray:
    """
    Autoregressive rollout in physical space using per-step dt_seconds.

    Returns:
      y_pred_phys: [K, S] float64
    """
    y = torch.from_numpy(y0_phys.astype(np.float32, copy=False)).to(device=DEVICE, dtype=DTYPE).unsqueeze(0)  # [1,S]

    if g_phys.size == 0:
        g = torch.empty((1, 0), device=DEVICE, dtype=DTYPE)
    else:
        g = torch.from_numpy(g_phys.astype(np.float32, copy=False)).to(device=DEVICE, dtype=DTYPE).unsqueeze(0)  # [1,G]

    dt_all = torch.from_numpy(dt_seconds.astype(np.float32, copy=False)).to(device=DEVICE, dtype=DTYPE)  # [K]

    preds: List[torch.Tensor] = []
    for k in range(int(dt_all.shape[0])):
        dt = dt_all[k : k + 1]  # [1]
        y = model(y, dt, g)  # expected [1,S]
        preds.append(y[0].detach().cpu())

    return torch.stack(preds, dim=0).numpy().astype(np.float64, copy=False)


def _distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    cols: List[Tuple[float, float, float]] = []
    for name in cmaps:
        cmap = plt.get_cmap(name)
        cols.extend([tuple(mcolors.to_rgb(c)) for c in getattr(cmap, "colors", [])])

    if n > len(cols):
        hs = np.linspace(0.0, 1.0, n, endpoint=False)
        cols = [tuple(mcolors.hsv_to_rgb((h, 0.85, 0.95))) for h in hs]

    return cols[:n]


def _plot(
    *,
    t_sec: np.ndarray,            # [K]
    y_true: np.ndarray,           # [K,S]
    y_pred: np.ndarray,           # [K,S]
    species_keys: List[str],
    out_path: Path,
) -> None:
    labels = [s.replace("_evolution", "") for s in species_keys]

    # Order by peak abundance for a readable legend.
    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = _distinct_colors(len(order))

    me = max(1, int(PLOT_MARKER_EVERY))
    marker_idx = np.arange(0, len(t_sec), me)

    fig, ax = plt.subplots(figsize=(7, 7))

    for r, j in enumerate(order):
        c = colors[r]
        ax.plot(t_sec, y_true[:, j], linestyle="-", lw=2.5, alpha=1.0, color=c)
        ax.plot(
            t_sec[marker_idx],
            y_pred[marker_idx, j],
            linestyle="None",
            marker="o",
            ms=3.5,
            alpha=1.0,
            color=c,
            markerfacecolor="none",
            markeredgecolor=c,
            markeredgewidth=0.9,
        )

    ax.set_yscale("log")
    ax.set(
        xlim=(float(t_sec[0]), float(t_sec[-1])),
        ylim=PLOT_Y_LIM,
        xlabel="Time (s)",
        ylabel="Relative Abundance",
        title=f"Autoregressive rollout ({SPLIT}): {len(t_sec)} steps (exported 1-step PHYS model)",
    )
    ax.set_box_aspect(1)

    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(
        species_handles,
        [labels[int(i)] for i in order],
        loc="lower left",
        title="Species",
        ncol=2,
        fontsize=8,
    )
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.5, alpha=1.0, label="Ground Truth"),
        Line2D(
            [0], [0],
            color="k",
            linestyle="None",
            marker="o",
            ms=3.5,
            alpha=1.0,
            markerfacecolor="none",
            markeredgecolor="k",
            markeredgewidth=0.9,
            label=f"Prediction (every {me} step{'s' if me != 1 else ''})",
        ),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_errors(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    eps = 1e-30
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)

    rel_err = np.abs(yp - yt) / (np.abs(yt) + 1e-12)
    mae = np.mean(np.abs(yp - yt))
    log_err = np.mean(np.abs(np.log10(yp) - np.log10(yt)))

    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    _try_style(PLOT_STYLE)

    export_path = (RUN_DIR / EXPORT_NAME).resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Missing export: {export_path}")

    model, meta = _load_export_module(export_path)
    print(f"[model] {export_path.name}  device={DEVICE.type} dtype={DTYPE}")

    norm_path = _resolve_normalization_path(meta)
    processed_dir = norm_path.parent
    manifest = _load_json(norm_path)

    species_keys = list(manifest.get("species_variables") or [])
    if not species_keys:
        raise KeyError("normalization.json missing species_variables")
    global_keys = list(manifest.get("global_variables") or [])

    # Load one shard and pick one trajectory
    y_mat, g_mat, dt_norm_mat = _load_shard(processed_dir, split=SPLIT, shard_index=SHARD_INDEX)
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= int(y_mat.shape[0]):
        raise IndexError(f"SAMPLE_INDEX out of range: {SAMPLE_INDEX} (shard has N={y_mat.shape[0]})")

    y_z = y_mat[SAMPLE_INDEX]          # [T,S]
    g_z = g_mat[SAMPLE_INDEX]          # [G]
    dt_norm = dt_norm_mat[SAMPLE_INDEX]  # [T-1]

    T = int(y_z.shape[0])
    if START_INDEX < 0 or START_INDEX >= T - 1:
        raise IndexError(f"START_INDEX out of range: {START_INDEX} (trajectory has T={T})")

    max_steps = int((T - 1) - START_INDEX)
    if N_STEPS <= 0 or N_STEPS > max_steps:
        raise ValueError(f"N_STEPS must be in [1, {max_steps}], got {N_STEPS}")

    # Ground truth and per-step dt for the evaluation horizon
    y0_z = y_z[START_INDEX]  # [S]
    y_true_z = y_z[START_INDEX + 1 : START_INDEX + 1 + N_STEPS]  # [K,S]
    dt_seq_norm = dt_norm[START_INDEX : START_INDEX + N_STEPS]   # [K]

    y0_phys = _denormalize_species(y0_z[None, :], species_keys, manifest)[0]
    y_true_phys = _denormalize_species(y_true_z, species_keys, manifest)
    g_phys = _denormalize_globals(g_z, global_keys, manifest)

    if USE_CONSTANT_DT:
        dt_seq_seconds = np.full((N_STEPS,), float(CONSTANT_DT_SECONDS), dtype=np.float64)
    else:
        dt_seq_seconds = _dt_norm_to_seconds(dt_seq_norm, manifest)

    # Rollout
    y_pred_phys = _rollout(model, y0_phys=y0_phys, g_phys=g_phys, dt_seconds=dt_seq_seconds)

    # Time axis: cumulative sum of dt (t=dt_1, dt_1+dt_2, ...)
    t_eval = np.cumsum(dt_seq_seconds, dtype=np.float64)

    out_path = (
        RUN_DIR
        / "plots"
        / f"autoregressive_phys_export_{SPLIT}_sh{SHARD_INDEX}_i{SAMPLE_INDEX}_t{START_INDEX}_k{N_STEPS}.png"
    )
    _plot(t_sec=t_eval, y_true=y_true_phys, y_pred=y_pred_phys, species_keys=species_keys, out_path=out_path)
    print(f"Saved: {out_path}")
    _print_errors(y_true_phys, y_pred_phys)


if __name__ == "__main__":
    main()
