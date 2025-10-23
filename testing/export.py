#!/usr/bin/env python3
from __future__ import annotations
import json, re, sys, pathlib
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
MODEL_DIR = ROOT / "models" / "autoencoder"
CONFIG = MODEL_DIR / "config.json"
OUT = MODEL_DIR / "export_k1_cpu.pt2"

sys.path.insert(0, str(SRC))
from model import create_model, FlowMapAutoencoder  # <- we’ll patch this class

# PyTorch 2.6 safe unpickler for Lightning checkpoints
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

def _jload(p: Path): return json.loads(p.read_text())

def _rehydrate_cfg(cfg: dict) -> dict:
    pdd = Path(cfg["paths"]["processed_data_dir"])
    if not pdd.is_absolute(): pdd = (ROOT / pdd).resolve()
    cfg["paths"]["processed_data_dir"] = str(pdd)
    norm = _jload(pdd / "normalization.json")
    meta = norm.get("meta", {})
    data = cfg.setdefault("data", {})
    def _need(x): return x is None or (isinstance(x, (list, tuple)) and len(x)==0)
    if _need(data.get("species_variables")):
        data["species_variables"] = meta.get("species_variables") or norm.get("species_variables") or []
    if _need(data.get("global_variables")):
        data["global_variables"] = meta.get("global_variables") or norm.get("global_variables") or []
    if not data.get("time_variable"):
        data["time_variable"] = meta.get("time_variable") or norm.get("time_variable") or "t_time"
    if _need(data.get("target_species")):
        data["target_species"] = list(data["species_variables"])
    # attach stats if needed
    m = cfg.setdefault("model", {})
    if m.get("softmax_head") or m.get("predict_delta_log_phys"):
        pks, tgt = norm["per_key_stats"], data["target_species"]
        m["target_log_mean"] = [float(pks[s]["log_mean"]) for s in tgt]
        m["target_log_std"]  = [float(pks[s]["log_std"])  for s in tgt]
    return cfg

def _pick_ckpt(d: Path) -> Path:
    best = d / "best_model.pt"
    if best.exists(): return best
    ckdir = d / "checkpoints"
    if ckdir.exists():
        bests = []
        for p in ckdir.glob("epoch*.ckpt"):
            m = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", p.name)
            if m:
                epoch = int(m.group(1))
                try: val = float(m.group(2))
                except: val = float("inf")
                bests.append((val, -epoch, p))
        if bests:
            bests.sort(key=lambda t:(t[0], t[1]))
            return bests[0][2]
        last = ckdir / "last.ckpt"
        if last.exists(): return last
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts: return pts[0]
    raise FileNotFoundError("No checkpoint found")

def _safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")  # weights_only=True in 2.6
    except Exception:
        # Trusted local file fallback:
        return torch.load(path, map_location="cpu", weights_only=False)

def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module.", "_orig_mod."):
        if k.startswith(pref): k = k[len(pref):]
    return k

def _load_state_dict(path: Path, model: torch.nn.Module) -> dict:
    payload = _safe_load(path)
    if not isinstance(payload, dict): raise RuntimeError(f"Bad checkpoint: {type(payload)}")
    raw = None
    for k in ("state_dict","model","model_state_dict","ema_model"):
        if isinstance(payload.get(k), dict): raw = payload[k]; break
    if raw is None: raw = {k:v for k,v in payload.items() if isinstance(v, torch.Tensor)}
    raw = { _strip_prefix(k): v for k,v in raw.items() }
    want = set(model.state_dict().keys())
    filt = { k:v for k,v in raw.items() if k in want }
    missing = [k for k in model.state_dict().keys() if k not in filt]
    if missing: raise RuntimeError(f"Missing {len(missing)} keys; e.g. {missing[0]}")
    return filt

# --- remove data-dependent branching from softmax head for export ---
def _smx_no_guard(self, logits: torch.Tensor) -> torch.Tensor:
    # branch-free: sanitize, softmax, convert to z-space
    logits = torch.nan_to_num(logits)
    log_p = F.log_softmax(logits, dim=-1)
    ln10 = self.ln10.to(dtype=log_p.dtype)
    z = (log_p / ln10 - self.log_mean.to(dtype=log_p.dtype)) / self.log_std.to(dtype=log_p.dtype)
    return z
FlowMapAutoencoder._softmax_head_from_logits = _smx_no_guard  # class-level patch

def main():
    cfg = _rehydrate_cfg(_jload(CONFIG))
    model = create_model(cfg).eval()
    sd = _load_state_dict(_pick_ckpt(MODEL_DIR), model)
    model.load_state_dict(sd, strict=True)

    B = torch.export.Dim("batch")
    S_in = model.S_in
    G = getattr(model, "G", getattr(model, "global_dim", 0))
    y  = torch.zeros(2, S_in, dtype=torch.float32)
    dt = torch.zeros(2, 1, 1, dtype=torch.float32)  # K=1
    g  = torch.zeros(2, G, dtype=torch.float32) if G > 0 else torch.zeros(2, 0, dtype=torch.float32)

    ep = torch.export.export(model, (y, dt, g), dynamic_shapes=({0:B}, {0:B}, {0:B}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, OUT)
    print(f"Exported → {OUT}")

if __name__ == "__main__":
    main()
