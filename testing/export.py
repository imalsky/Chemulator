#!/usr/bin/env python3
"""
Export Flow-map AE to torch.export (K=1), minimal version.

- Loads the exact training config from MODEL_DIR/config.json
- Instantiates the model with that config
- Loads weights (best_model.pt / epoch*.ckpt / last.ckpt / newest)
- Patches small control-flow bits for export
- Exports with dynamic batch dimension
"""

from __future__ import annotations
import json, math, re, sys, pathlib
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------- Paths (adjust MODEL_DIR if needed) ----------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODEL_DIR = ROOT / "models" / "autoencoder_big"
CONFIG = MODEL_DIR / "config.json"
OUT = MODEL_DIR / "export_k1_cpu.pt2"

sys.path.insert(0, str(SRC))
from model import create_model, FlowMapAutoencoder  # noqa: E402

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

# ---------------- Small helpers ----------------
def load_json(p: Path) -> dict:
    return json.loads(p.read_text())

def load_ckpt(p: Path) -> dict:
    try:
        return torch.load(p, map_location="cpu")
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)

def find_checkpoint(model_dir: Path) -> Path:
    best = model_dir / "best_model.pt"
    if best.exists(): return best
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        cands = []
        for q in ckpt_dir.glob("epoch*.ckpt"):
            m = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", q.name)
            if m:
                epoch = int(m.group(1))
                try: val = float(m.group(2))
                except Exception: val = float("inf")
                cands.append((val, -epoch, q))
        if cands:
            cands.sort()
            return cands[0][2]
        last = ckpt_dir / "last.ckpt"
        if last.exists(): return last
    files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.ckpt"))
    if files:
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")

def extract_state_dict(ckpt_path: Path, model: torch.nn.Module) -> dict:
    payload = load_ckpt(ckpt_path)
    raw = None
    if isinstance(payload, dict):
        for k in ("state_dict", "ema_model", "model_state_dict", "model"):
            v = payload.get(k)
            if isinstance(v, dict):
                raw = v
                break
        if raw is None:  # flat tensor dict?
            raw = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unrecognized checkpoint format: {type(payload)}")

    def strip(k: str) -> str:
        for pref in ("model.", "module.", "_orig_mod."):
            if k.startswith(pref): return k[len(pref):]
        return k

    raw = {strip(k): v for k, v in raw.items()}
    want = set(model.state_dict().keys())
    filt = {k: v for k, v in raw.items() if k in want}
    miss = [k for k in want if k not in filt]
    if miss:
        raise RuntimeError(f"Missing {len(miss)} keys in checkpoint, e.g. {miss[0]}")
    return filt

def patch_for_export(model_cls):
    ln10 = math.log(10.0)
    inv_ln10 = 1.0 / ln10

    def _softmax_head_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1).float()  # nat log
        log10_p = log_p * inv_ln10
        z = (log10_p - self.log_mean) / self.log_std
        return z.to(dtype=logits.dtype)

    def _head_from_logprobs(self, log_p: torch.Tensor) -> torch.Tensor:
        log10_p = log_p.float() * inv_ln10
        z = (log10_p - self.log_mean) / self.log_std
        return z.to(dtype=log_p.dtype)

    def forward_k1(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Expect dt_norm shape [B, 1]
        z_i = self.encoder(y_i, g)
        z_k = self.dynamics(z_i, dt_norm, g)
        if getattr(self, "decoder_condition_on_g", False):
            z_k = self.film(z_k, g)
        logits = self.decoder(z_k)
        if not getattr(self, "predict_logit_delta", False):
            return self._softmax_head_from_logits(logits)

        base = y_i if self.S_out == self.S_in else y_i.index_select(1, self.target_idx)
        base_logp = self._denorm_to_logp(base)  # natural log
        log_q = F.log_softmax(logits, dim=-1).float()
        log_p = base_logp.unsqueeze(1) + log_q
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return self._head_from_logprobs(log_p)

    model_cls._softmax_head_from_logits = _softmax_head_from_logits
    model_cls._head_from_logprobs = _head_from_logprobs
    model_cls.forward = forward_k1

# ---------------- Main ----------------
def main():
    import os
    os.chdir(ROOT)

    cfg = load_json(CONFIG)                     # exact training config (includes data.*)
    model = create_model(cfg).eval().cpu()      # must see data.species_variables, etc.

    ckpt = find_checkpoint(MODEL_DIR)
    state = extract_state_dict(ckpt, model)
    model.load_state_dict(state, strict=True)
    print(f"[export] loaded: {ckpt.name}")

    patch_for_export(FlowMapAutoencoder)

    B = torch.export.Dim("batch")
    y = torch.zeros(2, model.S_in, dtype=torch.float32)
    dt = torch.zeros(2, 1, dtype=torch.float32)  # K=1
    G = getattr(model, "global_dim", getattr(model, "G", 0))
    g = torch.zeros(2, G, dtype=torch.float32)

    ep = torch.export.export(
        model, (y, dt, g),
        dynamic_shapes=({0: B}, {0: B}, {0: B}),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, OUT)
    print(f"[export] saved → {OUT}")

if __name__ == "__main__":
    main()
