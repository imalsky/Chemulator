#!/usr/bin/env python3
from __future__ import annotations
import os, json, re, sys, pathlib
from pathlib import Path
import torch, torch.nn.functional as F

# --------------------------- Paths & Imports ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
MODEL_DIR = ROOT / "models" / "autoencoder"
CONFIG = MODEL_DIR / "config.json"
OUT = MODEL_DIR / "export_k1_cpu.pt2"

# Ensure relative paths in config (e.g., data/processed_*) resolve correctly
os.chdir(ROOT)

sys.path.insert(0, str(SRC))
from model import create_model, FlowMapAutoencoder  # type: ignore

# PyTorch 2.6 safe unpickler for Lightning checkpoints
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

# ------------------------------ Utils ----------------------------------------
def _jload(p: Path): return json.loads(p.read_text())

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
                try:
                    val = float(m.group(2))
                except Exception:
                    val = float("inf")
                bests.append((val, -epoch, p))
        if bests:
            bests.sort(key=lambda t: (t[0], t[1]))
            return bests[0][2]
        last = ckdir / "last.ckpt"
        if last.exists(): return last
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts: return pts[0]
    raise FileNotFoundError(f"No checkpoint found in {d}")

def _safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")  # weights_only=True also OK for many ckpts
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)

def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module.", "_orig_mod."):
        if k.startswith(pref): return k[len(pref):]
    return k

def _load_state_dict(path: Path, model: torch.nn.Module) -> dict:
    payload = _safe_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bad checkpoint payload type: {type(payload)}")
    raw = None
    for k in ("state_dict", "model", "model_state_dict", "ema_model"):
        if isinstance(payload.get(k), dict):
            raw = payload[k]
            break
    if raw is None:
        raw = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    raw = {_strip_prefix(k): v for k, v in raw.items()}

    want = set(model.state_dict().keys())
    filt = {k: v for k, v in raw.items() if k in want}
    missing = [k for k in model.state_dict().keys() if k not in filt]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} keys; e.g. {missing[0]}")
    return filt

# --------- Branch-free heads and forward (remove data-dependent guards) -------
def _softmax_head_no_guard(self, logits: torch.Tensor) -> torch.Tensor:
    # Softmax (ln), convert to log10, normalize with stored stats. Do all in tensor ops.
    log_p = F.log_softmax(logits, dim=-1)               # natural log
    z_f = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z_f.to(dtype=logits.dtype)

def _head_from_logprobs_no_guard(self, log_p: torch.Tensor) -> torch.Tensor:
    z_f = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z_f.to(dtype=log_p.dtype)

def _forward_no_guard(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Same math as your FlowMapAutoencoder.forward, but without any
    `if torch.isfinite(...).all()` Python branches that break torch.export.
    Shapes:
      y_i:    [B, S_in]   (normalized)
      dt_norm:[B, K]
      g:      [B, G]
    Returns:
      [B, K, S_out] (normalized)
    """
    z_i = self.encoder(y_i, g)                      # [B,Z]
    z_k = self.dynamics(z_i, dt_norm, g)            # [B,K,Z]
    if self.decoder_condition_on_g:
        z_k = self.film(z_k, g)                     # [B,K,Z]
    logits = self.decoder(z_k)                      # [B,K,S_out]

    if not self.predict_logit_delta:
        return self._softmax_head_from_logits(logits)

    # logit-delta path
    base = y_i if self.S_out == self.S_in else y_i.index_select(1, self.target_idx)
    base_logp = self._denorm_to_logp(base)          # [B,S_out]
    log_q = F.log_softmax(logits, dim=-1).float()   # [B,K,S_out]
    log_p = base_logp.unsqueeze(1) + log_q          # [B,K,S_out]
    log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
    return self._head_from_logprobs(log_p)

# Apply patches for tracing
FlowMapAutoencoder._softmax_head_from_logits = _softmax_head_no_guard
FlowMapAutoencoder._head_from_logprobs      = _head_from_logprobs_no_guard
FlowMapAutoencoder.forward                  = _forward_no_guard

# ---------------------------------- Main -------------------------------------
def main():
    cfg = _jload(CONFIG)                 # no rehydration; use config.json as-is
    model = create_model(cfg).eval().cpu()

    sd = _load_state_dict(_pick_ckpt(MODEL_DIR), model)
    model.load_state_dict(sd, strict=True)

    # Dynamic batch; fix K=1 for this artifact
    B = torch.export.Dim("batch")
    S_in = model.S_in
    G = getattr(model, "global_dim", getattr(model, "G", 0))

    y  = torch.zeros(2, S_in, dtype=torch.float32)      # [B,S]
    dt = torch.zeros(2, 1,    dtype=torch.float32)      # [B,1]  (K=1)
    g  = torch.zeros(2, G,    dtype=torch.float32)      # [B,G]

    ep = torch.export.export(
        model, (y, dt, g),
        dynamic_shapes=({0: B}, {0: B}, {0: B})         # only batch is dynamic
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, OUT)
    print(f"Exported → {OUT}")

if __name__ == "__main__":
    main()
