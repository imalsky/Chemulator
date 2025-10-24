#!/usr/bin/env python3
"""
Export Flow-map AE to torch.export format with K=1 baked in.
Patches model forward to remove control flow for tracing compatibility.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODEL_DIR = ROOT / "models" / "autoencoder"
CONFIG = MODEL_DIR / "config.json"
OUT = MODEL_DIR / "export_k1_cpu.pt2"

sys.path.insert(0, str(SRC))
from model import create_model, FlowMapAutoencoder

# Safe unpickler for Lightning checkpoints
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


def load_config(path: Path) -> dict:
    """Load JSON config file."""
    return json.loads(path.read_text())


def find_checkpoint(model_dir: Path) -> Path:
    """Find best available checkpoint in priority order."""
    # Priority 1: best_model.pt
    best = model_dir / "best_model.pt"
    if best.exists():
        return best

    # Priority 2: best validation checkpoint
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates = []
        for ckpt_path in ckpt_dir.glob("epoch*.ckpt"):
            match = re.match(r"epoch(\d+)-val([0-9eE+\-\.]+)\.ckpt$", ckpt_path.name)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2)) if match.group(2) else float("inf")
                candidates.append((val_loss, -epoch, ckpt_path))

        if candidates:
            candidates.sort()
            return candidates[0][2]

        # Fallback: last.ckpt
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last

    # Priority 3: most recent .pt file
    pt_files = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint with fallback for weights_only compatibility."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt_path: Path, model: torch.nn.Module) -> dict:
    """Extract and filter state dict from checkpoint."""
    payload = load_checkpoint(ckpt_path)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint format: {type(payload)}")

    # Find state dict in checkpoint
    raw_state = None
    for key in ("state_dict", "model", "model_state_dict", "ema_model"):
        if isinstance(payload.get(key), dict):
            raw_state = payload[key]
            break

    # Fallback: treat payload as state dict
    if raw_state is None:
        raw_state = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}

    # Strip common prefixes
    def strip_prefix(key: str) -> str:
        for prefix in ("model.", "module.", "_orig_mod."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    raw_state = {strip_prefix(k): v for k, v in raw_state.items()}

    # Filter to model keys
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in raw_state.items() if k in model_keys}

    missing = [k for k in model_keys if k not in filtered]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} keys in checkpoint, e.g. {missing[0]}")

    return filtered


def patch_model_for_export(model_class):
    """Patch model methods to remove control flow for torch.export compatibility."""

    def softmax_head_no_guard(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax head without data-dependent guards."""
        log_p = F.log_softmax(logits, dim=-1)
        z_f = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
        return z_f.to(dtype=logits.dtype)

    def head_from_logprobs_no_guard(self, log_p: torch.Tensor) -> torch.Tensor:
        """Log probability head without data-dependent guards."""
        z_f = (log_p.float() * self.ln10_inv - self.log_mean) / self.log_std
        return z_f.to(dtype=log_p.dtype)

    def forward_no_guard(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Forward pass without control flow branches."""
        z_i = self.encoder(y_i, g)
        z_k = self.dynamics(z_i, dt_norm, g)

        if self.decoder_condition_on_g:
            z_k = self.film(z_k, g)

        logits = self.decoder(z_k)

        if not self.predict_logit_delta:
            return self._softmax_head_from_logits(logits)

        # Logit-delta path
        base = y_i if self.S_out == self.S_in else y_i.index_select(1, self.target_idx)
        base_logp = self._denorm_to_logp(base)
        log_q = F.log_softmax(logits, dim=-1).float()
        log_p = base_logp.unsqueeze(1) + log_q
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return self._head_from_logprobs(log_p)

    model_class._softmax_head_from_logits = softmax_head_no_guard
    model_class._head_from_logprobs = head_from_logprobs_no_guard
    model_class.forward = forward_no_guard


def main():
    # Setup
    import os
    os.chdir(ROOT)

    # Load config and create model
    config = load_config(CONFIG)
    model = create_model(config).eval().cpu()

    # Load weights
    ckpt_path = find_checkpoint(MODEL_DIR)
    state_dict = extract_state_dict(ckpt_path, model)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {ckpt_path.name}")

    # Patch for export
    patch_model_for_export(FlowMapAutoencoder)

    # Create example inputs (K=1)
    B = torch.export.Dim("batch")
    S_in = model.S_in
    G = getattr(model, "global_dim", getattr(model, "G", 0))

    y_example = torch.zeros(2, S_in, dtype=torch.float32)
    dt_example = torch.zeros(2, 1, dtype=torch.float32)
    g_example = torch.zeros(2, G, dtype=torch.float32)

    # Export
    exported_program = torch.export.export(
        model,
        (y_example, dt_example, g_example),
        dynamic_shapes=({0: B}, {0: B}, {0: B})
    )

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, OUT)
    print(f"Exported → {OUT}")


if __name__ == "__main__":
    main()