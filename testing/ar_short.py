#!/usr/bin/env python3
# Minimal AR viewer using torch.export artifact, no argparse.

from __future__ import annotations
import os, json, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path

# =======================
# Globals (edit here)
# =======================
SAMPLE_IDX = 2
DEVICE_PREF = "auto"          # "auto" | "cuda" | "mps" | "cpu"
MODEL_DIR = "models/big"
GPU_BASENAME = "export_k1_gpu.pt2"
CPU_BASENAME = "export_k1_cpu.pt2"
WARMUP_STEPS = 5
DT_EPS_NORM = 1e-3
CLIP_MIN_FEED = 1e-30
PLOT_XMIN, PLOT_XMAX = 1e-3, None
PLOT_YMIN, PLOT_YMAX = None, 3.0

# =======================
# Repo / imports
# =======================
def _repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "src").is_dir() and (p / "models").is_dir():
            return p
    return cur

HERE = Path(__file__).resolve()
ROOT = _repo_root(HERE)
MODEL_DIR = (ROOT / MODEL_DIR).resolve()
os.sys.path.insert(0, str(ROOT / "src"))

from normalizer import NormalizationHelper

# =======================
# Artifact helpers
# =======================
def _pick_device(pref: str) -> str:
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":  return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cpu":  return "cpu"
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    return {"float16":torch.float16,"half":torch.float16,"fp16":torch.float16,
            "bfloat16":torch.bfloat16,"bf16":torch.bfloat16,
            "float32":torch.float32,"fp32":torch.float32,"float":torch.float32}[s]

def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f: return json.load(f)

def select_artifact() -> tuple[Path, torch.device, torch.dtype]:
    host = _pick_device(DEVICE_PREF)
    gpu = MODEL_DIR / GPU_BASENAME
    cpu = MODEL_DIR / CPU_BASENAME
    if host in ("cuda","mps") and gpu.exists():
        meta_p = MODEL_DIR / (GPU_BASENAME + ".meta.json")
        meta = _load_json(meta_p) if meta_p.exists() else {"device":host,"dtype":"float32"}
        dev = torch.device(meta.get("device", host))
        return gpu, dev, _dtype_from_str(meta.get("dtype","float32"))
    meta_p = MODEL_DIR / (CPU_BASENAME + ".meta.json")
    meta = _load_json(meta_p) if meta_p.exists() else {"device":"cpu","dtype":"float32"}
    return cpu, torch.device("cpu"), _dtype_from_str(meta.get("dtype","float32"))

# =======================
# Data + bounds
# =======================
def dt_bounds_from_manifest(norm: NormalizationHelper) -> tuple[float,float]:
    man = getattr(norm, "manifest", {}) or {}
    per = dict(man.get("per_key_stats", {}))
    meth = dict(man.get("normalization_methods", {}))
    if meth.get("dt") == "log-min-max":
        s = per.get("dt", {}); return 10**float(s.get("log_min",-3)), 10**float(s.get("log_max",8))
    if meth.get("t_time") == "log-min-max":
        s = per.get("t_time", {}); return 10**float(s.get("log_min",-3)), 10**float(s.get("log_max",8))
    return 1e-3, 1e8

def _coerce_pred_shape(y):
    if isinstance(y,(list,tuple)): y = y[0]
    if not torch.is_tensor(y): y = torch.as_tensor(y)
    if y.ndim == 1: return y.view(1,-1)
    if y.ndim == 3 and y.shape[1] == 1: return y[:,0,:]
    if y.ndim == 2: return y
    raise RuntimeError(f"Unexpected export output shape: {tuple(y.shape)}")

def load_everything():
    cfg = _load_json(MODEL_DIR / "config.json")
    data_dir = (ROOT / cfg["paths"]["processed_data_dir"]).resolve()
    norm = NormalizationHelper(_load_json(data_dir / "normalization.json"))
    shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not shards: raise FileNotFoundError("No test shards in test/")
    with np.load(shards[0]) as d:
        y_all = d["y_mat"].astype(np.float32)   # [N,M,S]
        g_all = d["globals"].astype(np.float32) # [N,G]
        t_vec = d["t_vec"]
    y_traj = y_all[SAMPLE_IDX]
    g_vec  = g_all[SAMPLE_IDX]
    t_phys = (t_vec if t_vec.ndim==1 else t_vec[SAMPLE_IDX]).astype(np.float64)
    return norm, y_traj, g_vec, t_phys, data_dir

# =======================
# Normalize utilities
# =======================
def norm_state(norm: NormalizationHelper, y_phys: np.ndarray, keys, dev, dtp):
    y = np.maximum(y_phys, CLIP_MIN_FEED)[None,:].astype(np.float32)
    return norm.normalize(torch.from_numpy(y), keys).to(dev, dtp)

def norm_globals(norm: NormalizationHelper, g: np.ndarray, keys, dev, dtp):
    return norm.normalize(torch.from_numpy(g[None,:].astype(np.float32)), keys).to(dev, dtp)

def norm_dt(norm: NormalizationHelper, dt_phys: float|np.ndarray, dev, dtp):
    v = np.asarray(dt_phys, dtype=np.float32).reshape(-1)
    dt_hat = norm.normalize_dt_from_phys(torch.tensor(v, dtype=torch.float32))
    dt_hat = torch.clamp(dt_hat, DT_EPS_NORM, 1.0 - DT_EPS_NORM)
    return dt_hat.to(dev, dtp).view(-1,1)  # [B,1]

# =======================
# Main AR path
# =======================
def main():
    export_path, dev, dtp = select_artifact()
    ep = torch.export.load(str(export_path))
    gm = ep.module()

    # Data
    norm, y, g, t, _ = load_everything()
    meta = norm.manifest.get("meta", {})
    in_names = list(meta.get("species_variables") or [])
    glob_names = list(meta.get("global_variables") or [])
    if not in_names or glob_names is None: raise RuntimeError("Missing species/globals in manifest.")
    dt_lo, dt_hi = dt_bounds_from_manifest(norm)

    # Probe output mapping
    y0 = norm_state(norm, y[0], in_names, dev, dtp)
    g_norm = norm_globals(norm, g, glob_names, dev, dtp)
    with torch.inference_mode():
        out_probe = _coerce_pred_shape(gm(y0, torch.tensor([[0.5]], device=dev, dtype=dtp), g_norm))
    S_out = out_probe.shape[-1]
    out_names = in_names[:S_out]
    in_index = {n:i for i,n in enumerate(in_names)}
    out_to_in = np.array([in_index.get(n,-1) for n in out_names], dtype=int)
    identity = (S_out == len(in_names)) and np.all(out_to_in == np.arange(S_out))

    # Warmup (first effective step)
    dt0 = float(np.clip(t[1]-t[0], dt_lo, dt_hi))
    for _ in range(WARMUP_STEPS):
        _ = _coerce_pred_shape(gm(y0, norm_dt(norm, dt0, dev, dtp), g_norm))

    # AR rollout over GT step count (with clamped per-step Δt)
    cur = np.maximum(y[0].copy(), CLIP_MIN_FEED)
    pred_dt = []
    pred_phys = []
    cum = 0.0

    for k in range(len(t)-1):
        dtk = float(np.clip(t[k+1]-t[k], dt_lo, dt_hi))
        y_in = norm_state(norm, cur, in_names, dev, dtp)
        with torch.inference_mode():
            z = _coerce_pred_shape(gm(y_in, norm_dt(norm, dtk, dev, dtp), g_norm)).detach().to("cpu")
        y_next = norm.denormalize(z, out_names).numpy().reshape(-1)
        cum += dtk
        pred_dt.append(cum)
        pred_phys.append(y_next)
        if identity:
            cur = np.maximum(y_next, CLIP_MIN_FEED)
        else:
            buf = cur.astype(np.float32, copy=True)
            for j_out, j_in in enumerate(out_to_in):
                if j_in >= 0: buf[j_in] = max(y_next[j_out], CLIP_MIN_FEED)
            cur = buf

    pred_dt = np.asarray(pred_dt, float)
    pred_phys = np.asarray(pred_phys, float)

    # Build GT in out_names order
    gt_full = np.zeros((len(t), S_out), dtype=np.float32)
    for j_out, nm in enumerate(out_names):
        gt_full[:, j_out] = y[:, in_index[nm]]
    gt_full = np.clip(gt_full, 1e-30, None)
    dt_gt = (t - float(t[0])).astype(float)

    # Plot (matching colors, no legend)
    colors = plt.cm.tab20(np.linspace(0, 0.95, S_out))
    xmin = PLOT_XMIN if PLOT_XMIN is not None else (max(1e-12, dt_gt[1]*0.5) if len(dt_gt)>1 else 1e-12)
    xmax = PLOT_XMAX if PLOT_XMAX is not None else float(dt_gt[-1])
    ymin = PLOT_YMIN if PLOT_YMIN is not None else max(1e-30, float(np.nanmin(gt_full[gt_full>0])*0.5))
    ymax = PLOT_YMAX if PLOT_YMAX is not None else float(np.nanmax(gt_full)*1.2)
    m_gt = (dt_gt >= xmin) & (dt_gt <= xmax)

    fig, ax = plt.subplots(figsize=(12,6))
    for j in range(S_out):
        c = colors[j]
        ax.loglog(dt_gt[m_gt], gt_full[m_gt, j], '-', lw=1.8, alpha=0.9, color=c)
        if pred_dt.size and pred_phys.size:
            ax.loglog(pred_dt, np.clip(pred_phys[:, j], 1e-30, None), 'o-', lw=1.2, ms=5, mfc='none', mec=c, color=c)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Absolute time [s]"); ax.set_ylabel("Species Abundance")
    ax.grid(False)
    out_dir = (MODEL_DIR / "plots"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"profile_autoreg_sample_{SAMPLE_IDX}.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"Artifact: {export_path.name} | Device: {dev} | DType: {dtp}")
    print(f"dt_bounds=[{dt_lo:g}, {dt_hi:g}]  |  pred K={len(pred_dt)}")
    print(f"Plot: {out_path}")

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
