#!/usr/bin/env python3
"""
Plot Vulcan 0D truth (solid) and optional NN predictions (dashed) for three cases.

- Toggle overlay of predictions with PLOT_PRED_OVERLAY.
- Uses hard paths you provided.
- If predictions require species not in the provided initial list, they are set to 0.
- Emulator globals fed from each case (T in K; P in dyn/cm^2).
- Fixed axes: x in [1e-3, 1e8], y in [1e-15, 1].

Outputs:
  <MODEL_DIR>/plots/inference_three_cases.png
  <MODEL_DIR>/plots/inference_<case>.png  (per-panel)
"""

from __future__ import annotations
import os, sys, json, pickle, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch

# ========= USER TOGGLE =========
PLOT_PRED_OVERLAY = True  # set False to plot only Vulcan truth

# ========= HARD PATHS ==========
VULCAN_DIRS: Dict[str, Path] = {
    "solar":     Path("/Users/imalsky/Desktop/Vulcan/0D_full_NCHO/solar"),
    "100Xsolar": Path("/Users/imalsky/Desktop/Vulcan/0D_full_NCHO/100Xsolar"),
    "500Xsolar": Path("/Users/imalsky/Desktop/Vulcan/0D_full_NCHO/500Xsolar"),
    "COeq2":     Path("/Users/imalsky/Desktop/Vulcan/0D_full_NCHO/solar_CO2"),  # C/O = 2
}

# ========= REPO/MODEL PATHS ===
REPO_ROOT     = Path(__file__).resolve().parent.parent
MODEL_DIR     = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "flowmap-deeponet")).resolve()
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed")).resolve()
CONFIG_PATH   = Path(os.environ.get("CONFIG_PATH", REPO_ROOT / "config" / "config.jsonc")).resolve()
PLOTS_DIR     = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CANDIDATES = [
    MODEL_DIR / "complete_model_exported_k1_int8.pt2",
    MODEL_DIR / "complete_model_exported_k1.pt2",
    MODEL_DIR / "complete_model_exported.pt2",
]

# ========= IMPORT PROJECT HELPERS =========
sys.path.append(str((REPO_ROOT / "src").resolve()))
from utils import load_json, seed_everything           # type: ignore
from normalizer import NormalizationHelper             # type: ignore

# ========= ENV =========
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
torch.set_num_threads(1)

try:
    import json5  # type: ignore
    _json_load = json5.load
except Exception:
    _json_load = json.load

# ========= PLOTTING LIMITS ========
XMIN, XMAX = 1e-3, 1e8
YMIN, YMAX = 1e-15, 1.0

# ========= CASES (y0 includes He; normalize to sum=1 first) =========
CASE_DEFS = [
    {
        "label": "1000 K, 1 mbar",
        "T_K": 1000.0,
        "P_bar": 1e-3,
        "time_case": "solar",
        "species": ['H2', 'H2O', 'H', 'CH4', 'CO', 'CO2', 'N2', 'NH3', 'He'],
        "values":  [9.975331e-01, 1.074060e-03, 0.0, 5.902400e-04, 0.0, 0.0, 0.0, 1.415900e-04, 1.679000e-01],
        "plot_species": ['H2O','CH4','CO','CO2','NH3','N2','H2'],
    },
    {
        "label": "1500 K, 1 mbar",
        "T_K": 1500.0,
        "P_bar": 1e-3,
        "time_case": "solar",
        "species": ['OH','H2','H2O','H','CH4','CO','CO2','N2','NH3','He'],
        "values":  [0.0, 9.975331e-01, 1.074060e-03, 0.0, 5.902400e-04, 0.0, 0.0, 0.0, 1.415900e-04, 1.679000e-01],
        "plot_species": ['H2O','CH4','CO','CO2','NH3','N2','H2','OH'],
    },
    {
        "label": "2000 K, 1 bar",
        "T_K": 2000.0,
        "P_bar": 1.0,
        "time_case": "solar",
        "species": ['OH','H2','H2O','H','CH3','CH4','CO','CO2','H2CO','HCO','HCN','NH2','N2','NH3','He'],
        "values":  [0.0, 9.975331e-01, 1.074060e-03, 0.0, 0.0, 5.902400e-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.415900e-04, 1.679000e-01],
        "plot_species": ['H2O','CH4','CO','CO2','NH3','N2','H2','HCN'],
    },
]

# ========= name normalization =========
SUFFIXES = ("_evolution", "_mix", "_mixing_ratio", "_mr")
PREFIXES = ("abund_", "y_", "x_")

def base_name(name: str) -> str:
    s = name
    for pre in PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

# ========= loaders & helpers =========
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            try:
                return _json_load(f)
            except Exception:
                f.seek(0)
                return json.load(f)
    except Exception:
        return None

def resolve_species_globals() -> tuple[list[str], list[str], list[str] | None]:
    snap = MODEL_DIR / "config.snapshot.json"
    if snap.exists():
        conf = _load_json(snap) or {}
        data = conf.get("data", {}) if isinstance(conf, dict) else {}
        sv = list(data.get("species_variables", []) or [])
        gv = list(data.get("global_variables",  []) or [])
        tv = list(data.get("target_species",    []) or []) or None
        if sv:
            return sv, gv, tv

    norm = PROCESSED_DIR / "normalization.json"
    if norm.exists():
        man = _load_json(norm) or {}
        meta = (man.get("meta", {}) if isinstance(man, dict) else {})
        sv = list(meta.get("species_variables", []) or [])
        gv = list(meta.get("global_variables",  []) or [])
        return sv, gv, None

    if CONFIG_PATH.exists():
        cfg = _load_json(CONFIG_PATH) or {}
        data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        sv = list(data.get("species_variables", []) or [])
        gv = list(data.get("global_variables",  []) or [])
        tv = list(data.get("target_species",    []) or []) or None
        return sv, gv, tv

    return [], [], None

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def load_exported_model() -> torch.nn.Module:
    mp = first_existing(EXPORT_CANDIDATES)
    if mp is None:
        tried = "\n  ".join(map(str, EXPORT_CANDIDATES))
        raise FileNotFoundError("No exported model found. Tried:\n  " + tried)
    from torch.export import load as tload
    prog = tload(str(mp))
    return prog.module()

def find_vulcan_file(case_key: str, T_K: float, P_bar: float) -> Path:
    root = VULCAN_DIRS.get(case_key)
    if root is None or not root.exists():
        raise FileNotFoundError(f"Vulcan directory missing for '{case_key}': {root}")
    logP_cgs = int(round(math.log10(max(P_bar * 1e6, 1e-30))))  # bar->dyn/cm^2
    T_tag = f"T{int(round(T_K))}K"
    P_tag = f"logP{logP_cgs}"
    candidates = sorted(root.glob("*.vul"))
    for fp in candidates:
        name = fp.name
        if T_tag in name and P_tag in name:
            return fp
    # fallback: first readable
    for fp in candidates:
        try:
            with open(fp, "rb") as h:
                _ = pickle.load(h)
            return fp
        except Exception:
            pass
    raise FileNotFoundError(f"No usable .vul file in {root}")

def load_vulcan_truth(fp: Path) -> tuple[np.ndarray, List[str], np.ndarray, float]:
    with open(fp, "rb") as handle:
        data = pickle.load(handle)
    species = list(data["variable"]["species"])
    t = np.asarray(data["variable"]["t_time"], dtype=float)
    y_time = np.asarray(data["variable"]["y_time"])  # [T, layer, S]
    n0_arr = np.asarray(data["atm"]["n_0"])
    n0 = float(n0_arr.item() if n0_arr.shape == () else np.array(n0_arr).reshape(-1)[0])
    if y_time.ndim != 3 or y_time.shape[1] < 1:
        raise ValueError("Unexpected y_time shape; expected [time, layer, species] with >=1 layer.")
    return t, species, y_time[:, 0, :], n0

def build_y0(species_full: list[str], provided_names: list[str], provided_vals: list[float]) -> np.ndarray:
    """Normalize provided (including He) to sum=1, then map to species_full by base name; missing -> 0."""
    arr = np.asarray(provided_vals, dtype=np.float64)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Provided y0 values sum to 0.")
    arr = arr / s
    mapping = {base_name(n): float(v) for n, v in zip(provided_names, arr)}
    y0 = np.zeros(len(species_full), dtype=np.float32)
    for i, sp in enumerate(species_full):
        b = base_name(sp)
        y0[i] = mapping.get(b, 0.0)  # if the model expects a species you didn't give -> 0
    return y0

def globals_vector(globals_v: list[str], T_K: float, P_bar: float) -> np.ndarray:
    out = np.zeros(len(globals_v), dtype=np.float32)
    for i, k in enumerate(globals_v):
        kk = k.strip().lower()
        if kk in ("t", "temp", "temperature", "t_k", "temperature_k"):
            out[i] = float(T_K)
        elif kk in ("p", "pressure", "p_bar", "p_cgs", "p_dyn_cm2", "pressure_cgs"):
            out[i] = float(P_bar * 1e6)  # dyn/cm^2
        elif kk in ("log10p", "log10_p", "log10p_bar"):
            out[i] = float(np.log10(max(P_bar, 1e-30)))
        else:
            out[i] = 0.0
    return out

@torch.inference_mode()
def predict_series(fn, norm: NormalizationHelper,
                   species_full: list[str], species_out: list[str],
                   y0_phys: np.ndarray, g_phys: np.ndarray,
                   t_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return t_sel (t[1:]) and predictions array [K, S_out]."""
    device = torch.device("cpu")
    y0 = torch.from_numpy(y0_phys[None, :]).to(device)
    g  = torch.from_numpy(g_phys[None, :]).to(device)
    y0n = norm.normalize(y0, species_full)
    gn  = norm.normalize(g,  globals_v)

    t = np.asarray(t_phys, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_phys must be 1D with >= 2 points")
    t0 = float(t[0])
    t_sel = t[1:].copy()
    dt_phys = np.maximum(t_sel - t0, 0.0)
    dt_norm = norm.normalize_dt_from_phys(torch.as_tensor(dt_phys, dtype=torch.float32))

    preds = []
    for k in range(dt_norm.numel()):
        out = fn(y0n, gn, dt_norm[k:k+1])
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        out2 = out.reshape(-1, out.shape[-1])[:1, :]
        y_phys = norm.denormalize(out2, species_out).squeeze(0).cpu().numpy()
        preds.append(y_phys)
    y_pred = np.stack(preds, axis=0)
    return t_sel, y_pred

# ========= LaTeX labels for legend =========
TEX = {
    'H2':'H$_2$', 'H2O':'H$_2$O', 'CH4':'CH$_4$', 'CO2':'CO$_2$', 'CO':'CO',
    'NH3':'NH$_3$', 'NH2':'NH$_2$', 'C2H2':'C$_2$H$_2$', 'C2H6':'C$_2$H$_6$',
    'H2CO':'H$_2$CO', 'HCO':'HCO', 'HCN':'HCN', 'N2':'N$_2$', 'H':'H', 'OH':'OH',
}

# ========= MAIN =========
if __name__ == "__main__":
    seed_everything(42)

    # Resolve species & globals metadata
    species_full, globals_v, target_species = resolve_species_globals()
    if not species_full:
        raise SystemExit("Could not resolve species/globals from snapshot, normalization.json, or config.jsonc")
    species_out = target_species if (target_species and len(target_species) > 0) else species_full

    # Load model/normalizer only if predictions enabled
    if PLOT_PRED_OVERLAY:
        fn = load_exported_model()
    norm = NormalizationHelper(load_json(PROCESSED_DIR / "normalization.json"))

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, case in zip(axes, CASE_DEFS):
        label   = case["label"]
        T_K     = case["T_K"]
        P_bar   = case["P_bar"]
        names   = case["species"]
        values  = case["values"]
        shown   = case["plot_species"]

        # --- Vulcan truth ---
        vul_fp = find_vulcan_file(case["time_case"], T_K, P_bar)
        t_phys, vul_species, vul_y, n0 = load_vulcan_truth(vul_fp)
        # mixing ratio
        mr = vul_y / max(n0, 1e-30)

        # Plot truth (solid)
        truth_plotted = False
        for sp in shown:
            if sp in vul_species:
                j = vul_species.index(sp)
                y = np.clip(mr[:, j], YMIN, None)  # lower clip for log scale
                ax.loglog(t_phys, y, "-", lw=2.0, alpha=0.9, label=TEX.get(sp, sp))
                truth_plotted = True

        # --- Predictions overlay (dashed) ---
        if PLOT_PRED_OVERLAY:
            y0_full = build_y0(species_full, names, values)
            g_vec   = globals_vector(globals_v, T_K, P_bar)
            t_sel, y_pred = predict_series(fn, norm, species_full, species_out, y0_full, g_vec, t_phys)

            # map shown species (base-name match) onto species_out
            for sp in shown:
                base = base_name(sp)
                idx = None
                for k, out_name in enumerate(species_out):
                    if base_name(out_name) == base:
                        idx = k
                        break
                if idx is None:
                    continue  # model doesn't have this species
                pred = np.clip(y_pred[:, idx], YMIN, None)
                ax.loglog(t_sel, pred, "--", lw=1.8, alpha=0.9)

        # --- axes/legend ---
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mixing Ratio")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        if truth_plotted:
            ax.legend(fontsize=9, frameon=False, ncol=1)

        # Save per-panel
        save_name = "inference_" + label.replace(" ", "").replace(",", "").replace("/", "-") + ".png"
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
        # truth
        for sp in shown:
            if sp in vul_species:
                j = vul_species.index(sp)
                y = np.clip(mr[:, j], YMIN, None)
                ax2.loglog(t_phys, y, "-", lw=2.0, alpha=0.9, label=TEX.get(sp, sp))
        # preds
        if PLOT_PRED_OVERLAY:
            for sp in shown:
                base = base_name(sp)
                idx = None
                for k, out_name in enumerate(species_out):
                    if base_name(out_name) == base:
                        idx = k; break
                if idx is not None:
                    pred = np.clip(y_pred[:, idx], YMIN, None)
                    ax2.loglog(t_sel, pred, "--", lw=1.8, alpha=0.9)
        ax2.set_title(label)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Mixing Ratio")
        ax2.set_xlim(XMIN, XMAX)
        ax2.set_ylim(YMIN, YMAX)
        if truth_plotted:
            ax2.legend(fontsize=9, frameon=False, ncol=1)
        fig2.tight_layout()
        fig2.savefig(PLOTS_DIR / save_name, dpi=150)
        plt.close(fig2)

    # Save combined figure
    fig.savefig(PLOTS_DIR / "inference_three_cases.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Wrote {PLOTS_DIR / 'inference_three_cases.png'}")
