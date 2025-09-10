#!/usr/bin/env python3
"""
Plot Flow-map DeepONet predictions vs ground truth using a PT2-exported model (K=1).
- Trunk input is normalized Δt (dt-spec).
- We evaluate from the first physical time t0 of a chosen trajectory.
"""

# =========================
#          CONFIG
# =========================
import os, sys, json
from pathlib import Path

# Prefer environment variables; fallback to repo-relative defaults
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "flowmap-deeponet_done"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", REPO_ROOT / "config" / "config.jsonc"))

# We’ll search these in order:
EXPORT_CANDIDATES = [
    MODEL_DIR / "complete_model_exported_k1.pt2",
    MODEL_DIR / "complete_model_exported_k1_int8.pt2",
]

SAMPLE_INDEX = 4            # which trajectory to plot
OUTPUT_DIR = None            # None -> <model_dir>/plots
SEED = 42
Q_COUNT = 100                # None or 0 = use all times after t0
CONNECT_LINES = True
MARKER_EVERY = 2000

# =========================
#     ENV & IMPORTS
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from torch.export import load as torch_load
import json5

# Add src to path
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass

# =========================
#     SMALL HELPERS
# =========================
def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def _load_species_globals():
    """
    Prefer MODEL_DIR/config.snapshot.json (the config that lives with the model).
    Fallback to PROCESSED_DIR/normalization.json meta.
    Last resort: CONFIG_PATH (json/jsonc).
    """
    snap = MODEL_DIR / "config.snapshot.json"
    if snap.exists():
        try:
            conf = json.load(open(snap, "r"))
            data = conf.get("data", {})
            species = list(data.get("species_variables", []))
            globals_v = list(data.get("global_variables", []))
            if species:
                return species, globals_v
        except Exception:
            pass

    # Fallback: normalization.json
    norm_path = PROCESSED_DIR / "normalization.json"
    if norm_path.exists():
        try:
            manifest = json.load(open(norm_path, "r"))
        except Exception:
            manifest = json5.load(open(norm_path, "r"))
        meta = (manifest or {}).get("meta", {}) or {}
        species = list(meta.get("species_variables", []) or [])
        globals_v = list(meta.get("global_variables", []) or [])
        if species:
            return species, globals_v

    # Last resort: original config.jsonc
    try:
        cfg = json5.load(open(CONFIG_PATH, "r"))
        data = cfg.get("data", {})
        return list(data.get("species_variables", [])), list(data.get("global_variables", []))
    except Exception:
        return [], []

def _find_exported_model():
    p = _first_existing(EXPORT_CANDIDATES)
    if p is None:
        raise FileNotFoundError(f"No exported model found. Tried:\n  " + "\n  ".join(map(str, EXPORT_CANDIDATES)))
    return str(p)

def _load_exported_model(path: str):
    prog = torch_load(path)
    mod = prog.module()
    return mod, torch.device("cpu")

def _load_single_test_sample(data_dir: Path, sample_idx: int | None):
    test_dir = data_dir / "test"
    test_shards = sorted(test_dir.glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards in {test_dir}")
    with np.load(test_shards[0], allow_pickle=False) as d:
        x0 = d["x0"].astype(np.float32)          # [N,S]
        y = d["y_mat"].astype(np.float32)        # [N,M,S]
        g = d["globals"].astype(np.float32)      # [N,G]
        tvec = d["t_vec"]                        # [M] or [N,M]
    N = x0.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "y0": y[sample_idx:sample_idx + 1, 0, :],                           # [1,S] (state at first time)
        "y_true": y[sample_idx],                                            # [M,S]
        "t_phys": tvec[sample_idx] if tvec.ndim == 2 else tvec,             # [M]
        "globals": g[sample_idx:sample_idx + 1],                            # [1,G]
        "sample_idx": int(sample_idx),
    }

def _normalize_dt(norm: NormalizationHelper, dt_phys, device):
    dt = torch.as_tensor(dt_phys, dtype=torch.float32, device=device)
    dt = torch.clamp(dt, min=float(getattr(norm, "epsilon", 1e-25)))
    return norm.normalize_dt_from_phys(dt.view(-1))  # [K]

def _select_query_indices(count: int | None, t_phys: np.ndarray, exclude_first: bool = True):
    M = int(t_phys.size)
    start_idx = 1 if exclude_first else 0
    if M - start_idx <= 0:
        raise ValueError("Time grid must contain at least two points (t0 and a later time).")
    if not count or count >= (M - start_idx):
        return np.arange(start_idx, M, dtype=int)
    return np.linspace(start_idx, M - 1, count).round().astype(int)

@torch.inference_mode()
def _predict_one(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_scalar: torch.Tensor,
                 norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    dt1 = torch.as_tensor(dt_norm_scalar, dtype=torch.float32, device=y0_norm.device).reshape(-1)  # [1]
    if dt1.numel() != 1:
        raise ValueError(f"dt_norm_scalar must be scalar/len-1, got {tuple(dt1.shape)}")
    out = fn(y0_norm, g_norm, dt1)  # exported K=1 expects dt shape [1]
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out, device=y0_norm.device)
    S = y0_norm.shape[-1]
    if out.shape[-1] != S:
        raise RuntimeError(f"Exported model output last dim != S: got {out.shape}, expected last={S}")
    out_2d = out.reshape(-1, S)
    if out_2d.shape[0] != 1:
        out_2d = out_2d[:1, :]
    y_phys = norm.denormalize(out_2d, species)  # [1,S]
    return y_phys.squeeze(0).detach().cpu().numpy()

@torch.inference_mode()
def _predict_many(fn, y0_norm: torch.Tensor, g_norm: torch.Tensor, dt_norm_vec: torch.Tensor,
                  norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    preds = []
    K = int(dt_norm_vec.numel())
    for k in range(K):
        preds.append(_predict_one(fn, y0_norm, g_norm, dt_norm_vec[k:k+1], norm, species))
    return np.stack(preds, axis=0)  # [K,S]

def _compute_errors(y_pred: np.ndarray, y_true: np.ndarray, idx, species: list[str]):
    y_true_aligned = y_true[idx, :]
    rel = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)
    print("\n[ERROR] Mean relative error per species:")
    means = rel.mean(axis=0)
    for sp, e in zip(species, means):
        print(f"  {sp:15s}: {e:.3e}")
    print(f"\n[ERROR] Overall mean={rel.mean():.3e}, max={rel.max():.3e}")
    return rel

def _plot(t_phys, y_true, t_phys_sel, y_pred, sample_idx, species,
          connect_lines, marker_every, out_path: Path):
    eps = 1e-30
    t_plot = np.clip(t_phys, eps, None)
    y_t = np.clip(y_true, eps, None)
    y_p = np.clip(y_pred, eps, None)
    t_sel = np.clip(t_phys_sel, eps, None)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    max_values = np.max(y_t, axis=0)
    sorted_indices = np.argsort(max_values)[::-1]

    for i in range(len(species)):
        ax.loglog(t_plot, y_t[:, i], '-', color=colors[i], lw=2.0, alpha=0.9)
        if connect_lines and len(t_sel) > 1:
            ax.loglog(t_sel, y_p[:, i], '--', color=colors[i], lw=1.6, alpha=0.85)
        ax.loglog(t_sel[::max(1, marker_every)], y_p[::max(1, marker_every), i], 'o',
                  color=colors[i], ms=5, alpha=0.9, mfc='none')

    # Legend (sorted by species prominence)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=colors[idx], lw=2.0, alpha=0.9) for idx in sorted_indices]
    legend_labels = [species[idx] for idx in sorted_indices]
    ax.set_xlim(1e-3, 1e8)
    #ax.set_ylim(1e-3, 2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Species Abundance", fontsize=12)
    ax.grid(False)

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

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")

# =========================
#           MAIN
# =========================
def main():
    seed_everything(SEED)

    # Prefer config that lives with the model
    species, globals_v = _load_species_globals()
    if not species:
        raise RuntimeError("Could not resolve species/global variables from model snapshot, normalization.json, or config.jsonc")

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (MODEL_DIR / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_exported_model()
    fn, model_dev = _load_exported_model(model_path)

    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))
    data = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)

    # Normalize inputs
    y0 = torch.from_numpy(data["y0"]).to(model_dev).contiguous()      # [1,S]
    g  = torch.from_numpy(data["globals"]).to(model_dev).contiguous() # [1,G]
    y0n = norm.normalize(y0, species)
    gn  = norm.normalize(g, globals_v)

    # Build PHYSICAL time grid (exclude t0)
    t_phys_dense = data["t_phys"].astype(np.float64)
    t0 = float(t_phys_dense[0])
    M = int(t_phys_dense.size)
    effective_q_count = Q_COUNT if (Q_COUNT and Q_COUNT < M - 1) else M - 1
    idx = _select_query_indices(effective_q_count, t_phys_dense, exclude_first=True)
    idx = idx[idx > 0] if len(idx) > 0 else np.array([1], dtype=int)

    t_phys_sel = t_phys_dense[idx]
    dt_phys_sel = np.maximum(t_phys_sel - t0, 0.0)
    dt_norm_sel = _normalize_dt(norm, dt_phys_sel, model_dev)  # [K]

    print(f"[INFO] Flow-map inference: K={int(dt_norm_sel.numel())}")
    print(f"[INFO] Excluding t0={t0:.3e}s from predictions (Δt=0 not in training regime)")
    print(f"[INFO] First prediction at t={t_phys_sel[0]:.3e}s (Δt={dt_phys_sel[0]:.3e}s)")

    # Loop over K with K=1 export
    y_pred = _predict_many(fn, y0n, gn, dt_norm_sel, norm, species)  # [K,S]

    # Plot + errors
    out_png = out_dir / f"predictions_K{int(dt_norm_sel.numel())}_sample_bigy_{data['sample_idx']}.png"
    _plot(t_phys_dense, data["y_true"], t_phys_sel, y_pred,
          data["sample_idx"], species, CONNECT_LINES, MARKER_EVERY, out_png)

    _ = _compute_errors(y_pred, data["y_true"], idx, species)

if __name__ == "__main__":
    main()
