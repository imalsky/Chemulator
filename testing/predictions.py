#!/usr/bin/env python3
"""
Plot AE-DeepONet predictions vs ground truth using a PT2-exported model.

Usage: python predictions.py
"""

# =========================
#          CONFIG
# =========================
MODEL_STR     = "big_deep"
MODEL_DIR     = f"../models/{MODEL_STR}"
CONFIG_PATH   = f"{MODEL_DIR}/config.json"
PROCESSED_DIR = "../data/processed-10-log-standard"
EXPORT_PATHS  = [
    f"{MODEL_DIR}/complete_model_exported.pt2",
    f"{MODEL_DIR}/final_model_exported.pt2",  # fallback
]

SAMPLE_INDEX = 1           # which trajectory to plot
OUTPUT_DIR   = None        # None -> <model_dir>/plots
SEED         = 42

# Query time selection
Q_COUNT = 100              # None or 0 for full dense grid
Q_MODE  = "uniform"        # Options: uniform, random, log_uniform, anchors

# Plot styling
CONNECT_LINES = True
MARKER_EVERY  = 5

# =========================
#     ENV & IMPORTS
# =========================
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]      = "1"

from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from torch.export import load as torch_load

# Add src to path
sys.path.append(str((Path(__file__).resolve().parent.parent / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass


# =========================
#     CORE FUNCTIONS
# =========================
def _find_exported_model():
    for p in EXPORT_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"No exported model found. Tried: {EXPORT_PATHS}")

def _infer_device_from_module(m: torch.nn.Module) -> torch.device:
    for t in list(m.parameters()) + list(m.buffers()):
        return t.device
    return torch.device("cpu")

def _load_exported_model(path: str):
    prog = torch_load(path)
    mod  = prog.module()
    dev  = _infer_device_from_module(mod)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Exported module is on CUDA but CUDA is unavailable.")
    return mod, dev

def _load_single_test_sample(data_dir: str, sample_idx: int | None):
    test_shards = sorted((Path(data_dir) / "test").glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards in {data_dir}/test")

    with np.load(test_shards[0], allow_pickle=False) as d:
        x0   = d["x0"].astype(np.float32)          # [N,S]
        y    = d["y_mat"].astype(np.float32)       # [N,M,S]
        g    = d["globals"].astype(np.float32)     # [N,G]
        tvec = d["t_vec"]                          # [M] or [N,M]
    N = x0.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.default_rng(SEED).integers(0, N)
    return {
        "x0":     x0[sample_idx:sample_idx+1],               # [1,S]
        "y_true": y[sample_idx],                             # [M,S]
        "t_phys": tvec[sample_idx] if tvec.ndim == 2 else tvec,  # [M]
        "globals": g[sample_idx:sample_idx+1],               # [1,G]
        "sample_idx": int(sample_idx),
    }

def _normalize_time(norm: NormalizationHelper, t_phys: np.ndarray, t_name: str, device: torch.device):
    t = torch.as_tensor(t_phys, dtype=torch.float32, device=device).view(-1, 1)
    t_norm = norm.normalize(t, [t_name]).view(-1)
    if not torch.isfinite(t_norm).all():
        raise RuntimeError("Normalized time contains non-finite values")
    return t_norm

def _denorm_time(norm: NormalizationHelper, t_norm: torch.Tensor, t_name: str):
    try:
        return norm.denormalize(t_norm.view(-1, 1), [t_name]).view(-1).cpu().numpy()
    except Exception:
        return t_norm.detach().cpu().numpy()

def _load_anchor_times(data_dir: str, q_count: int | None, norm: NormalizationHelper, t_name: str, device: torch.device):
    candidates = [
        Path(data_dir).parent / "latent_data" / "latent_shard_index.json",
        Path(data_dir) / "latent" / "latent_shard_index.json",
    ]
    for fp in candidates:
        if fp.exists():
            anchors = load_json(fp).get("trunk_times")
            if not anchors:
                break
            a = np.asarray(anchors, dtype=np.float32)
            if q_count and q_count < a.size:
                idx = np.linspace(0, a.size - 1, q_count).round().astype(int)
                a = a[idx]
            t_norm_sel = torch.tensor(a, dtype=torch.float32, device=device)
            t_phys_sel = _denorm_time(norm, t_norm_sel, t_name)
            return t_norm_sel, t_phys_sel, None
    return None

def _select_query_times(mode: str, count: int | None, t_phys: np.ndarray, t_norm_dense: torch.Tensor,
                        data_dir: str, norm: NormalizationHelper, t_name: str, device: torch.device):
    M = int(t_norm_dense.numel())
    if not count or count >= M:
        return t_norm_dense, t_phys, np.arange(M, dtype=int)

    if mode == "anchors":
        r = _load_anchor_times(data_dir, count, norm, t_name, device)
        if r is not None:
            return r
        mode = "uniform"

    rng = np.random.default_rng(SEED)
    if mode == "uniform":
        idx = np.linspace(0, M - 1, count).round().astype(int)
    elif mode == "random":
        idx = np.sort(rng.choice(M, size=count, replace=False))
    elif mode == "log_uniform":
        pos = t_phys[t_phys > 0]
        if pos.size == 0:
            idx = np.linspace(0, M - 1, count).round().astype(int)
        else:
            tmin, tmax = float(pos.min()), float(t_phys.max())
            grid = np.logspace(np.log10(tmin), np.log10(tmax), count)
            idx  = np.unique(np.array([np.abs(t_phys - g).argmin() for g in grid], dtype=int))
            if idx.size < count:  # pad if collisions
                rest = np.setdiff1d(np.arange(M), idx)
                idx  = np.sort(np.r_[idx, rest[:max(0, count - idx.size)]])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return t_norm_dense[idx], t_phys[idx], idx

@torch.inference_mode()
def _predict(fn, x0_norm: torch.Tensor, g_norm: torch.Tensor, t_norm_sel: torch.Tensor,
             norm: NormalizationHelper, species: list[str]) -> np.ndarray:
    y_norm = fn(x0_norm, g_norm, t_norm_sel).squeeze(0)  # [K,S]
    return norm.denormalize(y_norm, species).cpu().numpy()

@torch.inference_mode()
def _time_sensitivity(fn, x0_norm, g_norm, t_norm_sel):
    if t_norm_sel.numel() < 2:
        return 0.0
    d = (fn(x0_norm, g_norm, t_norm_sel[:1]) - fn(x0_norm, g_norm, t_norm_sel[-1:])).abs().mean()
    val = float(d.item())
    print(f"[CHECK] Mean |Δpred| (first vs last time) = {val:.3e}")
    return val

def _compute_errors(y_pred: np.ndarray, y_true: np.ndarray, idx, t_sel, t_dense, species: list[str]):
    if idx is None:
        td = t_dense.detach().cpu().numpy()
        idx = np.array([np.abs(td - float(t)).argmin() for t in t_sel.cpu()], dtype=int)
    y_true_aligned = y_true[idx, :]
    rel = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)

    print("\n[ERROR] Mean relative error per species:")
    means = rel.mean(axis=0)
    for sp, e in zip(species, means):
        print(f"  {sp:15s}: {e:.3e}")
    print(f"\n[ERROR] Overall mean={rel.mean():.3e}, max={rel.max():.3e}")
    return rel

def _plot(t_phys, y_true, t_phys_sel, y_pred, sample_idx, species, q_mode, q_count,
          connect_lines, marker_every, out_path: Path):
    eps = 1e-30
    t_plot = np.clip(t_phys, eps, None)
    y_t    = np.clip(y_true, eps, None)
    y_p    = np.clip(y_pred, eps, None)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    for i in range(len(species)):
        ax.loglog(t_plot, y_t[:, i], '-',  color=colors[i], lw=2.0, alpha=0.9)
    if connect_lines and len(t_phys_sel) > 1:
        for i in range(len(species)):
            ax.loglog(t_phys_sel, y_p[:, i], '--', color=colors[i], lw=1.6, alpha=0.85)
    for i in range(len(species)):
        ax.loglog(t_phys_sel[::marker_every], y_p[::marker_every, i], 'o', color=colors[i], ms=5, alpha=0.9)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='black', lw=2.0, ls='-',  label='True (dense)'),
        Line2D([0],[0], color='black', lw=1.6, ls='--', label='Predicted (interp)'),
        Line2D([0],[0], color='black', marker='o', lw=0, ms=5, label='Predicted (queries)'),
    ], loc='lower left', fontsize=10)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Species Abundance")
    ax.set_title(f"AE-DeepONet vs Ground Truth (Sample {sample_idx}) — {q_mode} K={q_count}")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Plot saved to {out_path}")

# =========================
#           MAIN
# =========================
def main():
    seed_everything(SEED)

    cfg        = load_json(Path(CONFIG_PATH))
    species    = cfg["data"]["species_variables"]
    globals_v  = cfg["data"]["global_variables"]
    time_name  = cfg["data"]["time_variable"]

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (Path(MODEL_DIR) / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path     = _find_exported_model()
    fn, model_dev  = _load_exported_model(model_path)

    norm = NormalizationHelper(load_json(Path(PROCESSED_DIR) / "normalization.json"),
                               model_dev, cfg)

    data = _load_single_test_sample(PROCESSED_DIR, SAMPLE_INDEX)

    x0 = torch.from_numpy(data["x0"]).to(model_dev).contiguous()
    g  = torch.from_numpy(data["globals"]).to(model_dev).contiguous()
    x0n = norm.normalize(x0, species)
    gn  = norm.normalize(g,  globals_v)

    t_norm_dense = _normalize_time(norm, data["t_phys"], time_name, model_dev)
    t_sel, t_phys_sel, idx = _select_query_times(
        Q_MODE, Q_COUNT, data["t_phys"], t_norm_dense, PROCESSED_DIR, norm, time_name, model_dev
    )

    print(f"[INFO] Query mode={Q_MODE}, count={int(t_sel.numel())}")
    y_pred = _predict(fn, x0n, gn, t_sel, norm, species)

    _time_sensitivity(fn, x0n, gn, t_sel)

    out_png = out_dir / f"predictions_K{int(t_sel.numel())}_{Q_MODE}_sample_{data['sample_idx']}.png"
    _plot(data["t_phys"], data["y_true"], t_phys_sel, y_pred,
          data["sample_idx"], species, Q_MODE, int(t_sel.numel()),
          CONNECT_LINES, MARKER_EVERY, out_png)

    _compute_errors(y_pred, data["y_true"], idx, t_sel, t_norm_dense, species)

if __name__ == "__main__":
    main()
