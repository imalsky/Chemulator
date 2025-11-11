#!/usr/bin/env python3
# Minimal HDF5→model “single-shot” plot using SEED_1 and the first timestep as the anchor.
from __future__ import annotations
import json, math, re, sys
from pathlib import Path
import numpy as np, torch, h5py, matplotlib.pyplot as plt

# --------------------------- CONFIG ---------------------------
REPO_ROOT = Path("/Users/imalsky/Desktop/Chemulator")
SRC_DIR   = REPO_ROOT / "src"
PROCESSED = REPO_ROOT / "data" / "processed"          # contains normalization.json
MODEL_DIR = REPO_ROOT / "models" / "4"                # contains *.pt2 export
H5_PATH   = Path("/Users/imalsky/Desktop/test.h5")    # HDF5 truth file

PLOT_SPECIES = ['H2O','CH4','CO','CO2','NH3','HCN','N2','C2H2']
DT_MIN, DT_MAX, K = 1e-3, 1e8, 50
YMIN, YMAX = 1e-30, 2.0
OUT_DIR = MODEL_DIR / "plots"; OUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------

if str(SRC_DIR) not in sys.path: sys.path.append(str(SRC_DIR))
from normalizer import NormalizationHelper  # type: ignore

def load_norm():
    manifest = json.loads((PROCESSED / "normalization.json").read_text())
    norm = NormalizationHelper(manifest)
    meta = manifest.get("meta", {})
    in_names = list(meta.get("species_variables") or manifest.get("species_variables") or [])
    gvars    = list(meta.get("global_variables")  or manifest.get("global_variables")  or [])
    assert in_names, "species_variables missing in normalization.json"
    in_bases = [n[:-10] if n.endswith("_evolution") else n for n in in_names]
    return norm, in_names, in_bases, gvars

def load_model():
    from torch.export import load as tload
    order = ["export_k_dyn_gpu.pt2","export_k_dyn_mps.pt2","export_k_dyn_cpu.pt2",
             "export_k1_gpu.pt2","export_k1_mps.pt2","export_k1_cpu.pt2"]
    cands = [MODEL_DIR/p for p in order if (MODEL_DIR/p).exists()]
    if not cands:
        cands = sorted(MODEL_DIR.glob("*.pt2"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert cands, "No .pt2 export found"
    pt2 = cands[0]
    mod = tload(str(pt2)).module()
    name = pt2.name.lower()
    if ("gpu" in name or "cuda" in name) and torch.cuda.is_available():
        device = torch.device("cuda")
    elif "mps" in name and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_dyn = ("dyn" in name)
    print(f"[model] {pt2.name}  device={device.type}  dynBK={use_dyn}")
    return mod, device, use_dyn

def pick_seed1_group(f: h5py.File) -> str:
    cands = [k for k in f.keys() if isinstance(f[k], h5py.Group) and "SEED_1" in k]
    assert cands, "No group with SEED_1 found in H5"
    return sorted(cands)[0]

def parse_TP_from_group(gname: str):
    # run_T_1p000000eP03_P_1p000000eP06_SEED_1 → T=1e+03 K, P=1e+06 barye
    mT = re.search(r"T_([0-9p]+eP[0-9+-]+)", gname)
    mP = re.search(r"P_([0-9p]+eP[0-9+-]+)", gname)
    parse = lambda s: float(s.replace("p",".").replace("eP","e+"))
    T_K = parse(mT.group(1)) if mT else 0.0
    P_barye = parse(mP.group(1)) if mP else 0.0
    return float(T_K), float(P_barye)

def load_truth_first_timestep(f: h5py.File, gname: str):
    g = f[gname]
    t = np.asarray(g["t_time"][...], float)                        # [T]
    species, cols = [], []
    for k, d in g.items():
        if k.endswith("_evolution") and k != "t_time":
            species.append(k[:-10])
            cols.append(np.asarray(d[...], float))                 # [T]
    idx = np.argsort(species); species = [species[i] for i in idx]
    MR = np.stack([cols[i] for i in idx], axis=1)                  # [T, S]
    y0 = MR[0].clip(min=0.0); s = y0.sum(); y0 = (y0/s) if s>0 else np.full_like(y0, 1.0/len(y0))
    return t, MR, species, y0

def make_inputs(y0_map, in_bases, norm, in_names):
    FEED_MIN = 1e-30
    eff = norm.denormalize(norm.normalize(torch.zeros(1, len(in_names)), in_names), in_names).numpy().reshape(-1)
    floor = np.maximum(np.nan_to_num(eff, nan=0.0), FEED_MIN)
    vec = np.array([y0_map.get(b, 0.0) for b in in_bases], float)
    vec = np.maximum(vec, floor); vec = vec/np.maximum(vec.sum(), 1e-30)
    y0_norm = norm.normalize(torch.from_numpy(vec[None, :]).float(), in_names)
    return vec.astype(np.float32), y0_norm

def make_gvars(gvars, T_K, P_barye, norm):
    if not gvars: return torch.zeros(1,0)
    g = np.zeros((1,len(gvars)), np.float32)
    for i,n in enumerate(gvars):
        nl = n.lower().strip()
        g[0,i] = P_barye if nl.startswith("p") else (T_K if nl.startswith("t") else 0.0)
    return norm.normalize(torch.from_numpy(g), gvars).float()

def run_model(fn, dev, use_dyn, y0_norm, g_norm, dt_phys, norm):
    dt_hat = norm.normalize_dt_from_phys(torch.from_numpy(dt_phys)).view(-1,1).float()
    with torch.inference_mode():
        if use_dyn:  # BK signature: [1,K,S], [1,K,1], [1,K,G]
            y = fn(y0_norm.to(dev).expand(1,len(dt_phys),-1),
                   dt_hat.to(dev).view(1,-1,1),
                   (g_norm.to(dev).expand(1,len(dt_phys),-1) if g_norm.numel() else torch.empty(1,len(dt_phys),0, device=dev)))
            if isinstance(y, torch.Tensor) and y.dim()==3 and y.size(0)==1: y = y[0]
        else:        # K1 signature: [K,S], [K,1] or [K], [K,G]
            y = fn(y0_norm.to(dev).repeat(len(dt_phys),1),
                   dt_hat.to(dev),
                   (g_norm.to(dev).repeat(len(dt_phys),1) if g_norm.numel() else torch.empty(len(dt_phys),0, device=dev)))
    return y.to("cpu")

def main():
    torch.set_num_threads(1)
    norm, in_names, in_bases, gvars = load_norm()
    fn, dev, use_dyn = load_model()

    with h5py.File(H5_PATH, "r") as f:
        gname = pick_seed1_group(f)
        T_K, P_barye = parse_TP_from_group(gname)
        t, MR_all, species_h5, y0 = load_truth_first_timestep(f, gname)

    y0_map = {s: y0[i] for i, s in enumerate(species_h5)}
    y0_simplex, y0_norm = make_inputs(y0_map, in_bases, norm, in_names)
    g_norm = make_gvars(gvars, T_K, P_barye, norm)

    dt = np.logspace(math.log10(DT_MIN), math.log10(DT_MAX), K, dtype=np.float32)
    y_pred_norm = run_model(fn, dev, use_dyn, y0_norm, g_norm, dt, norm)

    # --------- Align outputs to H5 species (robust for S_out == 1) ----------
    cand = list(norm.manifest.get("meta", {}).get("target_species_variables")
                or norm.manifest.get("target_species_variables") or in_names)
    out_names = cand  # don't slice yet
    pred_phys = norm.denormalize(y_pred_norm, out_names).cpu().numpy()
    pred_phys = pred_phys.reshape(pred_phys.shape[0], -1)  # [K, S_out]
    S_out = pred_phys.shape[1]
    out_bases = [(n[:-10] if n.endswith("_evolution") else n) for n in out_names[:S_out]]

    present = [b for b in out_bases if b in species_h5]
    if not present:
        raise RuntimeError("No overlap between model outputs and H5 species.")
    out_idx = [out_bases.index(b) for b in present]
    h5_idx  = [species_h5.index(b)  for b in present]

    pred_sub  = np.clip(pred_phys[:, out_idx], 1e-300, None)
    pred_sub /= np.maximum(pred_sub.sum(1, keepdims=True), 1e-30)
    truth_sub = np.clip(MR_all[:, h5_idx], 1e-300, None)
    truth_sub /= np.maximum(truth_sub.sum(1, keepdims=True), 1e-30)

    keep = [i for i,b in enumerate(present) if b in PLOT_SPECIES] or list(range(len(present)))
    labels = [present[i] for i in keep]
    truth_plot = truth_sub[:, keep]
    pred_plot  = pred_sub[:, keep]

    # ------------------------------- Plot -----------------------------------
    mask = t > 0.0
    t_abs = t[mask]; truth_plot = truth_plot[mask]
    xmin = float(max((t_abs.min() if t_abs.size else DT_MIN), dt.min(), 1e-3))
    xmax = float(max((t_abs.max() if t_abs.size else DT_MAX), dt.max()))

    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin, xmax); ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Abundance")

    colors = plt.cm.tab20(np.linspace(0, 0.95, len(keep)))
    for i, c in enumerate(colors):
        ax.plot(t_abs, truth_plot[:, i], '-', lw=1.8, alpha=0.95, color=c)
        ax.plot(dt,    pred_plot[:,  i], 'o',  ms=4,  mfc='none', mec=c, mew=1.0, alpha=0.95)

    ax.legend([plt.Line2D([0],[0], color=colors[i], lw=2.0) for i in range(len(labels))],
              labels, loc='best', fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"h5_single_shot_seed1_{gname.replace('/','_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] saved → {out}")

if __name__ == "__main__":
    main()
