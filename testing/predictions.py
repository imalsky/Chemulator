#!/usr/bin/env python3
"""
Simple plot with strict x-axis: Flow-map DeepONet predictions (K=1 export) vs. ground truth.
- Plain Matplotlib (no external style files)
- Ground truth: solid line with the first plotted point over-marked by a circle
- Predictions: dashed line with sparse "x" markers
- X-axis is **fixed** to [XMIN, XMAX] (defaults 1e-3 .. 1e8)
- We **do not** clip time to tiny eps (which previously pulled xmin to ~1e-30 on log axes)

Env overrides (optional):
  MODEL_DIR, PROCESSED_DIR, CONFIG_PATH, SAMPLE_INDEX, Q_COUNT, XMIN, XMAX
"""
from __future__ import annotations

import os, sys, json
from pathlib import Path
from typing import List

# ---------------- Paths / Config ----------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "flowmap-deeponet"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", REPO_ROOT / "config" / "config.jsonc"))

EXPORT_CANDIDATES = [
    MODEL_DIR / "complete_model_exported_k1.pt2",
    MODEL_DIR / "complete_model_exported_k1_int8.pt2",
]

SAMPLE_INDEX = 0  #int(os.environ.get("SAMPLE_INDEX", "4"))
OUTPUT_DIR = None  # None -> <model_dir>/plots
SEED = 42
Q_COUNT = int(os.environ.get("Q_COUNT", "100"))  # if <=0 uses all times after t0
CONNECT_LINES = True
MARKER_FREQ = 5

# Strict x-axis bounds
XMIN = float(os.environ.get("XMIN", "1e-3"))
XMAX = float(os.environ.get("XMAX", "1e8"))
assert XMIN > 0 and XMAX > XMIN, "Require 0 < XMIN < XMAX"

# ---------------- Imports ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import json5
    _json5_load = json5.load
except Exception:
    _json5_load = json.load

torch.set_num_threads(1)

# Add src to path
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

plt.style.use("science.mplstyle")

# ---------------- Helpers ----------------

def _first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def _load_species_globals() -> tuple[list[str], list[str], list[str] | None]:
    """Resolve species/global names. Priority: snapshot -> normalization.json -> config.jsonc."""
    snap = MODEL_DIR / "config.snapshot.json"
    if snap.exists():
        try:
            conf = json.load(open(snap, "r"))
            data = conf.get("data", {})
            species = list(data.get("species_variables", []) or [])
            globals_v = list(data.get("global_variables", []) or [])
            target = list(data.get("target_species", []) or []) or None
            if species:
                return species, globals_v, target
        except Exception:
            pass

    norm_path = PROCESSED_DIR / "normalization.json"
    if norm_path.exists():
        try:
            manifest = json.load(open(norm_path, "r"))
        except Exception:
            manifest = _json5_load(open(norm_path, "r"))
        meta = (manifest or {}).get("meta", {}) or {}
        species = list(meta.get("species_variables", []) or [])
        globals_v = list(meta.get("global_variables", []) or [])
        return species, globals_v, None

    try:
        cfg = _json5_load(open(CONFIG_PATH, "r"))
        data = cfg.get("data", {})
        species = list(data.get("species_variables", []) or [])
        globals_v = list(data.get("global_variables", []) or [])
        target = list(data.get("target_species", []) or []) or None
        return species, globals_v, target
    except Exception:
        return [], [], None

def _find_exported_model() -> str:
    p = _first_existing(EXPORT_CANDIDATES)
    if p is None:
        tried = "\n  ".join(map(str, EXPORT_CANDIDATES))
        raise FileNotFoundError(f"No exported model found. Tried:\n  {tried}")
    return str(p)

def _load_exported_model(path: str) -> tuple[torch.nn.Module, torch.device]:
    from torch.export import load as torch_load
    prog = torch_load(path)
    mod = prog.module()
    return mod, torch.device("cpu")

def _load_single_test_sample(data_dir: Path, sample_idx: int | None):
    test_dir = data_dir / "test"
    test_shards = sorted(test_dir.glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards in {test_dir}")
    with np.load(test_shards[0], allow_pickle=False) as d:
        y = d["y_mat"].astype(np.float32)  # [N,M,S]
        g = d["globals"].astype(np.float32)  # [N,G]
        tvec = d["t_vec"]  # [M] or [N,M]
    N = y.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "y0": y[sample_idx:sample_idx + 1, 0, :],  # [1,S]
        "y_true": y[sample_idx],  # [M,S]
        "t_phys": tvec[sample_idx] if tvec.ndim == 2 else tvec,  # [M]
        "globals": g[sample_idx:sample_idx + 1],  # [1,G]
        "sample_idx": int(sample_idx),
    }

def _normalize_dt(norm: NormalizationHelper, dt_phys: np.ndarray, device: torch.device) -> torch.Tensor:
    dt = torch.as_tensor(dt_phys, dtype=torch.float32, device=device)
    dt = torch.clamp(dt, min=float(getattr(norm, "epsilon", 1e-25)))
    return norm.normalize_dt_from_phys(dt.view(-1))  # [K]

def _select_query_indices(count: int | None, t_phys: np.ndarray, exclude_first: bool = True) -> np.ndarray:
    M = int(t_phys.size)
    start_idx = 1 if exclude_first else 0
    if M - start_idx <= 0:
        raise ValueError("Time grid must contain at least two points (t0 and a later time).")
    if not count or count <= 0 or count >= (M - start_idx):
        return np.arange(start_idx, M, dtype=int)
    return np.linspace(start_idx, M - 1, int(count)).round().astype(int)

@torch.inference_mode()
def _predict_one(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_scalar: torch.Tensor,
                 norm: NormalizationHelper, species_out: list[str]) -> np.ndarray:
    dt1 = torch.as_tensor(dt_norm_scalar, dtype=torch.float32, device=y0_norm.device).reshape(-1)  # [1]
    if dt1.numel() != 1:
        raise ValueError(f"dt_norm_scalar must be scalar/len-1, got {tuple(dt1.shape)}")
    out = fn(y0_norm, g_norm, dt1)  # exported K=1 expects dt shape [1]
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=y0_norm.device)
    out_2d = out.reshape(-1, out.shape[-1])
    if out_2d.shape[-1] != len(species_out):
        raise RuntimeError(f"Exported model output last dim {out_2d.shape[-1]} != len(species_out) {len(species_out)}")
    if out_2d.shape[0] != 1:
        out_2d = out_2d[:1, :]
    y_phys = norm.denormalize(out_2d, species_out)  # [1,S_out]
    return y_phys.squeeze(0).detach().cpu().numpy()

@torch.inference_mode()
def _predict_many(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_vec: torch.Tensor,
                  norm: NormalizationHelper, species_out: list[str]) -> np.ndarray:
    preds = []
    for k in range(int(dt_norm_vec.numel())):
        preds.append(_predict_one(fn, y0_norm, g_norm, dt_norm_vec[k:k + 1], norm, species_out))
    return np.stack(preds, axis=0)  # [K,S_out]

# ---------------- Plotting (strict x-axis) ----------------

def _plot_strict(t_phys: np.ndarray, y_true: np.ndarray,
                 t_sel: np.ndarray, y_pred: np.ndarray,
                 species: list[str], out_path: Path,
                 connect_lines: bool = True,
                 xmin: float = XMIN, xmax: float = XMAX) -> None:
    m_gt = (t_phys >= xmin) & (t_phys <= xmax)
    m_pred = (t_sel >= xmin) & (t_sel <= xmax)

    t_gt = t_phys[m_gt]
    Y_gt = y_true[m_gt, :]

    t_pd = t_sel[m_pred]
    Y_pd = y_pred[m_pred, :]

    eps_y = 1e-30
    Y_gt = np.clip(Y_gt, eps_y, None)
    Y_pd = np.clip(Y_pd, eps_y, None)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    max_values = np.max(Y_gt, axis=0)
    sorted_indices = np.argsort(max_values)[::-1]

    stride = max(1, max(1, len(t_pd)) // MARKER_FREQ)

    for i in range(len(species)):
        color = colors[i]
        ax.loglog(t_gt, Y_gt[:, i], '-', color=color, linewidth=1.8, alpha=0.9)
        if t_gt.size:
            ax.loglog([t_gt[0]], [Y_gt[0, i]], 'o', color=color, markersize=5, mfc='none')
        if connect_lines and len(t_pd) > 1:
            ax.loglog(t_pd, Y_pd[:, i], '--', color=color, linewidth=1.4, alpha=0.85)
        if len(t_pd):
            ax.loglog(t_pd[::stride], Y_pd[::stride, i], 'x', color=color, markersize=5, alpha=0.9)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Species Abundance", fontsize=12)
    ax.grid(False)

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in sorted_indices]
    legend_labels = [species[idx] for idx in sorted_indices]

    legend1 = ax.legend(handles=legend_handles, labels=legend_labels,
                        loc='center left', bbox_to_anchor=(1.01, 0.6),
                        title='Species', fontsize=10, title_fontsize=11, ncol=1,
                        borderaxespad=0)
    ax.add_artist(legend1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2.0, ls='-', label='Ground Truth'),
        Line2D([0], [0], color='black', lw=1.6, ls='--', label='Model Prediction'),
    ]
    ax.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.01, 0.2),
              fontsize=10, title_fontsize=11, borderaxespad=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")

# ---------------- Main ----------------

def main() -> None:
    seed_everything(SEED)

    species_full, globals_v, target_species = _load_species_globals()
    if not species_full:
        raise RuntimeError("Could not resolve species/global variables from snapshot/normalization/config")
    species_out = target_species if (target_species and len(target_species) > 0) else species_full

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (MODEL_DIR / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_exported_model()
    fn, model_dev = _load_exported_model(model_path)

    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))
    data = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)

    # Map full->target species for printing and plotting
    norm_manifest = load_json(PROCESSED_DIR / "normalization.json")
    full_from_norm = list((norm_manifest.get("meta", {}) or {}).get("species_variables", []) or species_full)
    name_to_idx = {n: i for i, n in enumerate(full_from_norm)}
    target_idx = [name_to_idx[n] for n in species_out]

    # ---------- NEW: print P,T and y0 abundances (physical) ----------
    # Globals are physical values in the shards; try to extract T and P by name
    g_phys = data["globals"]  # [1,G]
    T_val = None; P_val = None
    for i, name in enumerate(globals_v):
        n = name.strip().lower()
        val = float(g_phys[0, i])
        if n in ("t", "temp", "temperature", "t_k", "temperature_k"):
            T_val = val
        elif n in ("p", "pressure", "p_pa", "pressure_pa", "p_bar", "p_cgs", "p_dyn_cm2", "pressure_cgs"):
            P_val = val
    print("\n=== Model inputs (physical) ===")
    if T_val is not None: print(f"T : {T_val}")
    if P_val is not None: print(f"P : {P_val}")
    print(f"Sample index: {data['sample_idx']}")
    y0_phys_for_out = data["y0"][0, target_idx]  # [S_out]
    print("\n=== Initial abundances y0 (species_out order) ===")
    for n, v in zip(species_out, y0_phys_for_out):
        print(f"{n:>8s} : {float(v):.6e}")
    # ---------------------------------------------------------------

    # Normalize inputs (unchanged)
    y0 = torch.from_numpy(data["y0"]).to(model_dev).contiguous()  # [1,S]
    g = torch.from_numpy(data["globals"]).to(model_dev).contiguous()  # [1,G]
    y0n = norm.normalize(y0, species_full)
    gn = norm.normalize(g, globals_v)

    # Physical times
    t_phys = data["t_phys"].astype(np.float64)
    t0 = float(t_phys[0])
    M = int(t_phys.size)

    effective_q = Q_COUNT if (Q_COUNT and Q_COUNT > 0 and Q_COUNT < M - 1) else (M - 1)
    idx = _select_query_indices(effective_q, t_phys, exclude_first=True)
    idx = idx[idx > 0] if len(idx) > 0 else np.array([1], dtype=int)

    t_sel = t_phys[idx]
    dt_phys_sel = np.maximum(t_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, model_dev)

    print(f"[INFO] Flow-map inference: K={int(dt_norm_sel.numel())}")
    print(f"[INFO] Excluding t0={t0:.3e}s from predictions (Δt=0 not in training regime)")
    print(f"[INFO] First prediction at t={t_sel[0]:.3e}s (Δt={dt_phys_sel[0]:.3e}s)")

    y_pred = _predict_many(fn, y0n, gn, dt_norm_sel, norm, species_out)  # [K,S_out]
    y_true_subset = data["y_true"][:, target_idx]  # [M, S_out]

    out_png = out_dir / f"predictions_strict_xlim_K{int(dt_norm_sel.numel())}_sample_{data['sample_idx']}.png"
    _plot_strict(t_phys, y_true_subset, t_sel, y_pred, species_out, out_png,
                 connect_lines=CONNECT_LINES, xmin=XMIN, xmax=XMAX)

    y_true_aligned = y_true_subset[idx, :]
    rel = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)
    print(f"[ERROR] rel-mean={rel.mean():.3e}, rel-max={rel.max():.3e}")

if __name__ == "__main__":
    main()
