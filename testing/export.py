#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, re, pathlib
from pathlib import Path
import torch, torch.nn.functional as F

# ---------------- paths ----------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WORK_DIR = ROOT / "models" / "v1"
CONFIG_PATH = WORK_DIR / "config.json"
OUT_PATH = WORK_DIR / "export_k1_cpu.pt2"

os.chdir(ROOT)
sys.path.insert(0, str(SRC))
from model import create_model, FlowMapAutoencoder  # type: ignore

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

# -------------- checkpoint load --------------
def pick_ckpt(d: Path) -> Path:
    best = d / "best_model.pt"
    if best.exists():
        return best
    ckdir = d / "checkpoints"
    if ckdir.exists():
        cands = []
        for p in ckdir.glob("epoch*.ckpt"):
            m = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", p.name)
            if m:
                ep = int(m.group(1))
                try:
                    val = float(m.group(2))
                except Exception:
                    val = float("inf")
                cands.append((val, -ep, p))
        if cands:
            cands.sort(key=lambda t: (t[0], t[1]))
            return cands[0][2]
        last = ckdir / "last.ckpt"
        if last.exists():
            return last
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError("no checkpoint found")

def load_weights_into(model: torch.nn.Module, ckpt_path: Path):
    pay = torch.load(ckpt_path, map_location="cpu")
    state = (
        pay.get("state_dict")
        or pay.get("model_state_dict")
        or pay.get("model")
        or pay.get("ema_model")
        or {k: v for k, v in pay.items() if isinstance(v, torch.Tensor)}
    )
    clean = {}
    for k, v in state.items():
        for pref in ("model.", "module.", "_orig_mod."):
            if k.startswith(pref):
                k = k[len(pref):]
        clean[k] = v
    model.load_state_dict(clean, strict=False)

# --------- patch model to remove export-hostile branches ---------
def _softmax_head_export(self, logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)  # ln
    z = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z.to(dtype=logits.dtype)

def _head_from_logprobs_export(self, log_p: torch.Tensor) -> torch.Tensor:
    z = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
    return z.to(dtype=log_p.dtype)

def _forward_export(self, y_i, dt_norm, g):
    enc = self.encoder(y_i, g)
    if isinstance(enc, (tuple, list)):
        z_i, self.kl_loss = enc
    else:
        z_i, self.kl_loss = enc, None
    z_k = self.dynamics(z_i, dt_norm, g)          # [B,K,Z]
    if getattr(self, "decoder_condition_on_g", False):
        z_k = self.film(z_k, g)
    logits = self.decoder(z_k)                    # [B,K,S_out]
    if not getattr(self, "predict_logit_delta", False):
        return self._softmax_head_from_logits(logits)
    base = y_i if self.S_out == self.S_in else y_i.index_select(1, self.target_idx)
    base_logp = self._denorm_to_logp(base)        # [B,S_out]
    log_q = F.log_softmax(logits, dim=-1).float() # [B,K,S_out]
    log_p = base_logp.unsqueeze(1) + log_q
    log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
    return self._head_from_logprobs(log_p)

FlowMapAutoencoder._softmax_head_from_logits = _softmax_head_export
FlowMapAutoencoder._head_from_logprobs = _head_from_logprobs_export
FlowMapAutoencoder.forward = _forward_export

# ---------------- main ----------------
def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    model = create_model(cfg).eval().cpu()
    load_weights_into(model, pick_ckpt(WORK_DIR))

    B = torch.export.Dim("batch")
    S_in = model.S_in
    G = int(getattr(model, "global_dim", getattr(model, "G", 0)) or 0)

    y = torch.zeros(2, S_in, dtype=torch.float32)
    dt = torch.zeros(2, 1, dtype=torch.float32)
    g = torch.zeros(2, G, dtype=torch.float32)

    ep = torch.export.export(
        model, (y, dt, g),
        dynamic_shapes=({0: B}, {0: B}, {0: B}),
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, OUT_PATH)
    print(f"wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
