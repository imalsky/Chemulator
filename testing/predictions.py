#!/usr/bin/env python3
"""
predictions.py - Autoregressive 1-step evaluation on test data (constant dt).

Evaluates an exported 1-step autoregressive model (.pt2) on preprocessed test shards,
comparing predictions against ground truth in both physical and log space.

Assumes repo layout:
  Auto-Chem/
    src/
    testing/  (this file)
    data/processed/
    models/v1/
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    run_dir: Path = ROOT / "models" / "v2_auto"
    processed_dir: Path = ROOT / "data" / "processed"
    export_name: str = "export_cpu_1step.pt2"

    sample_idx: int = 1
    start_index: int = 1
    n_steps: int = 200

    plot_species: list | None = None  # None/[] => all
    y_range: tuple = (1e-30, 3)

    true_lw: float = 3.0
    true_alpha: float = 1.0
    pred_lw: float = 2.0
    pred_alpha: float = 1.0
    pred_ls: tuple = (8, 3)
    pred_marker: str = "o"
    pred_ms: float = 3


CFG = Config()


# =============================================================================
# Paths
# =============================================================================

def _resolve_processed_dir() -> Path:
    """
    Always prefer <repo>/data/processed.
    Only fall back to run_dir configs if they point to an existing local path.
    """
    fallback = CFG.processed_dir.resolve()
    if (fallback / "normalization.json").exists():
        return fallback

    for cfg_path in (CFG.run_dir / "config.resolved.json", CFG.run_dir / "config.json", ROOT / "config.json"):
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        paths = cfg.get("paths", {}) or {}
        p = paths.get("processed_dir") or paths.get("processed_data_dir")
        if not isinstance(p, str) or not p.strip():
            continue
        cand = Path(p).expanduser()
        if not cand.is_absolute():
            cand = (cfg_path.parent / cand).resolve()
        if (cand / "normalization.json").exists():
            return cand

    return fallback


def _infer_device_and_dtype(export_name: str) -> tuple[str, torch.dtype]:
    n = export_name.lower()
    if "cuda" in n and torch.cuda.is_available():
        return "cuda", torch.float16
    if "mps" in n and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# =============================================================================
# Data
# =============================================================================

def load_test_trajectory(processed_dir: Path, idx: int):
    """
    Returns (y_z, g_z, dt_norm, species)
      y_z: [T, S]
      g_z: [G]
      dt : [T-1]
    """
    shards = sorted((processed_dir / "test").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards in {processed_dir / 'test'}")

    # Assumes idx is contained in shard_0; extend if needed.
    with np.load(shards[0]) as f:
        y_all = f["y_mat"].astype(np.float32)        # [N, T, S]
        g_all = f["globals"].astype(np.float32)      # [N, G]
        dt_all = f["dt_norm_mat"].astype(np.float32) # [N, T-1]

    if idx < 0 or idx >= y_all.shape[0]:
        raise IndexError(f"Sample index {idx} out of range (N={y_all.shape[0]})")

    manifest = json.loads((processed_dir / "normalization.json").read_text())
    species = list(manifest.get("species_variables", []))
    return y_all[idx], g_all[idx], dt_all[idx], species


# =============================================================================
# Normalization
# =============================================================================

def dt_to_seconds(dt_norm: float, manifest: dict) -> float:
    log_min, log_max = manifest["dt"]["log_min"], manifest["dt"]["log_max"]
    log_dt = dt_norm * (log_max - log_min) + log_min
    return 10.0 ** log_dt


def denormalize_species(y_z: np.ndarray, species: list, manifest: dict) -> np.ndarray:
    stats = manifest["per_key_stats"]
    mu = np.array([stats[s]["log_mean"] for s in species], dtype=np.float64)
    sd = np.array([stats[s]["log_std"] for s in species], dtype=np.float64)
    return 10.0 ** (y_z * sd + mu)


# =============================================================================
# Inference
# =============================================================================

def _step_model(model, y: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = model(y, dt, g)
    return out[:, 0, :] if out.ndim == 3 else out


@torch.inference_mode()
def rollout(model, y0_z: np.ndarray, g_z: np.ndarray, dt_norm: float, n_steps: int,
            device: str, dtype: torch.dtype) -> np.ndarray:
    """
    Autoregressive rollout with constant dt. Returns [n_steps, S] in z-space.

    Handles batch-specialized exports (often B=2):
      try B=1; if export rejects it, retry B=2 by duplicating the sample and
      keeping predictions for the first element.
    """
    y0 = torch.from_numpy(y0_z).to(device=device, dtype=dtype)      # [S]
    g0 = torch.from_numpy(g_z).to(device=device, dtype=dtype)       # [G]
    dt0 = torch.tensor(float(dt_norm), device=device, dtype=dtype)  # scalar

    def run(B: int) -> torch.Tensor:
        y = y0.unsqueeze(0).expand(B, -1).contiguous()              # [B, S]
        g = g0.unsqueeze(0).expand(B, -1).contiguous()              # [B, G]
        dt = dt0.expand(B).contiguous()                             # [B]
        ys = []
        for _ in range(n_steps):
            y = _step_model(model, y, dt, g)                        # [B, S]
            ys.append(y[0])
        return torch.stack(ys, dim=0)                               # [n_steps, S]

    try:
        pred = run(1)
    except Exception:
        pred = run(2)

    return pred.detach().cpu().numpy()


# =============================================================================
# Plotting / Metrics
# =============================================================================

def _distinct_colors(n: int):
    """
    Return n visually distinct RGB colors.

    Uses several qualitative (categorical) Matplotlib palettes first (best contrast),
    then falls back to evenly-spaced HSV hues if n is large.
    """
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    cols = []

    for name in cmaps:
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):  # ListedColormap
            cols.extend(list(cmap.colors))
        else:
            cols.extend([cmap(i) for i in np.linspace(0, 1, getattr(cmap, "N", 256))])

    # De-duplicate (some palettes share colors)
    uniq = []
    seen = set()
    for c in cols:
        rgb = tuple(np.round(mcolors.to_rgb(c), 4))
        if rgb not in seen:
            seen.add(rgb)
            uniq.append(rgb)

    if n <= len(uniq):
        return uniq[:n]

    # Fallback: golden-ratio spaced hues (keeps separation as n grows)
    phi = 0.61803398875
    h = 0.0
    while len(uniq) < n:
        h = (h + phi) % 1.0
        rgb = mcolors.hsv_to_rgb((h, 0.85, 0.95))
        uniq.append(tuple(rgb))

    return uniq[:n]


def plot_results(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, species: list, out_path: Path) -> None:
    labels = [s.replace("_evolution", "") for s in species]

    if CFG.plot_species:
        keep = [i for i, lab in enumerate(labels) if lab in CFG.plot_species]
        if not keep:
            raise ValueError(f"No matching species: {CFG.plot_species}")
        labels = [labels[i] for i in keep]
        y_true = y_true[:, keep]
        y_pred = y_pred[:, keep]

    y_true = np.clip(y_true, 1e-35, None)
    y_pred = np.clip(y_pred, 1e-35, None)

    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = _distinct_colors(len(order))

    fig, ax = plt.subplots(figsize=(7, 7))

    for r, i in enumerate(order):
        c = colors[r]
        ax.plot(t,
                y_true[:, i],
                "-",
                lw=CFG.true_lw,
                alpha=CFG.true_alpha,
                color=c)
        ax.plot(
            t,
            y_pred[:, i],
            lw=CFG.pred_lw,
            dashes=CFG.pred_ls,
            marker=CFG.pred_marker,
            ms=CFG.pred_ms,
            alpha=CFG.pred_alpha,
            color=c,
            markerfacecolor="none",
            markeredgecolor=c,
            markeredgewidth=0.9,
        )

    ax.set_yscale("log")
    #ax.set_xscale("log")

    ax.set(xlim=(t[0], t[-1]), ylim=CFG.y_range,
           xlabel="Time (s)", ylabel="Relative Abundance",
           title=f"Autoregressive (constant dt): {len(t)} steps")
    ax.set_box_aspect(1)

    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(species_handles, [labels[i] for i in order],
                     loc="lower left", title="Species", ncol=2, fontsize=8)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=CFG.true_lw, alpha=CFG.true_alpha, label="Ground Truth"),
        Line2D([0], [0], color="k", lw=CFG.pred_lw, dashes=CFG.pred_ls,
               marker=CFG.pred_marker, ms=CFG.pred_ms, alpha=CFG.pred_alpha,
               markerfacecolor="none", markeredgecolor="k", markeredgewidth=0.9,
               label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_errors(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    mae = np.mean(np.abs(y_pred - y_true))

    y_true_c = np.clip(y_true, 1e-35, None)
    y_pred_c = np.clip(y_pred, 1e-35, None)
    log_err = np.mean(np.abs(np.log10(y_pred_c) - np.log10(y_true_c)))

    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    processed_dir = _resolve_processed_dir()
    manifest = json.loads((processed_dir / "normalization.json").read_text())

    export_path = (CFG.run_dir / CFG.export_name).resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    device, dtype = _infer_device_and_dtype(CFG.export_name)
    model = torch.export.load(export_path).module()
    try:
        model = model.to(device)
    except Exception:
        device, dtype = "cpu", torch.float32

    print(f"processed_dir: {processed_dir}")
    print(f"export:        {export_path.name} | device={device} dtype={str(dtype).replace('torch.', '')}")

    y_z, g_z, dt_vec, species = load_test_trajectory(processed_dir, CFG.sample_idx)
    T, S = y_z.shape
    print(f"sample={CFG.sample_idx} | T={T} S={S}")

    max_steps = (T - 1) - CFG.start_index
    n_steps = min(CFG.n_steps, max_steps)
    if n_steps <= 0:
        raise ValueError("Invalid start_index or n_steps")

    dt_slice = dt_vec[CFG.start_index: CFG.start_index + n_steps]
    dt0 = float(dt_slice[0])
    if not np.allclose(dt_slice, dt0, rtol=1e-5, atol=1e-7):
        raise ValueError("Non-constant dt over selected horizon")

    dt_sec = dt_to_seconds(dt0, manifest)
    print(f"start={CFG.start_index} steps={n_steps} dt≈{dt_sec:.3e}s horizon≈{n_steps * dt_sec:.3e}s")

    y0_z = y_z[CFG.start_index]
    y_true_z = y_z[CFG.start_index + 1: CFG.start_index + 1 + n_steps]
    y_pred_z = rollout(model, y0_z, g_z, dt0, n_steps, device=device, dtype=dtype)

    y_true = denormalize_species(y_true_z, species, manifest)
    y_pred = denormalize_species(y_pred_z, species, manifest)

    t_eval = dt_sec * np.arange(1, n_steps + 1)
    out_path = CFG.run_dir / "plots" / f"autoregressive_constdt_{CFG.sample_idx}_t{CFG.start_index}.png"
    plot_results(t_eval, y_true, y_pred, species, out_path)
    print_errors(y_true, y_pred)


if __name__ == "__main__":
    main()
