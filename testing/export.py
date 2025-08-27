#!/usr/bin/env python3
"""
Export complete AE-DeepONet model including encoder for inference.
Creates a model that accepts normalized initial species concentrations and outputs predictions.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Configuration
MODEL_STR = "deepo"
MODEL_DIR = f"../models/{MODEL_STR}"
MODEL_PATH = f"{MODEL_DIR}/best_model.pt"
CONFIG_PATH = f"{MODEL_DIR}/config.json"
EX_TRUNK_STEPS = 64
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

import sys, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.export import export as texport, save as tsave, Dim

sys.path.append(str((Path(__file__).resolve().parent.parent / "src").resolve()))
from model import AEDeepONet

# State dict sanitizer
_PREFIXES = ("_orig_mod.", "module.", "model.")


def _strip_prefixes(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def _sanitize_state_for(model: nn.Module, sd: dict) -> dict:
    want = set(model.state_dict().keys())
    out, dropped = {}, []
    for k, v in sd.items():
        if k.endswith("._device_tensor") or k == "_device_tensor":
            dropped.append(k);
            continue
        k2 = _strip_prefixes(k)
        if k2 in want:
            out[k2] = v
        else:
            k3 = _strip_prefixes(k2)
            if k3 in want:
                out[k3] = v
            else:
                dropped.append(k)
    if dropped:
        print(f"[LOAD] dropped {len(dropped)} keys (first few): {dropped[:8]}")
    print(f"[LOAD] mapped {len(out)}/{len(sd)} keys into model")
    return out


class CompleteInferenceWrapper(nn.Module):
    """
    Complete inference wrapper that includes encoding.

    Takes normalized initial species + globals, encodes to latent,
    runs DeepONet, and decodes back to species.

    Interface:
      forward(x0_norm[B, S], globals_norm[B, G], trunk_times[K]) -> y_pred[B, K, S]
    """

    def __init__(self, model: AEDeepONet):
        super().__init__()
        self.model = model

    def forward(self, x0_norm: torch.Tensor, globals_norm: torch.Tensor, trunk_times: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass from initial conditions to predictions.

        Args:
            x0_norm: [B, S] normalized initial species concentrations
            globals_norm: [B, G] normalized global parameters
            trunk_times: [K] or [K,1] normalized time points

        Returns:
            y_pred: [B, K, S] predicted species concentrations (normalized)
        """
        # Encode initial conditions to latent space
        z0 = self.model.encode(x0_norm)  # [B, L]

        # Combine with globals to form branch input
        branch_input = torch.cat([z0, globals_norm], dim=-1)  # [B, L+G]

        # Ensure trunk times are 2D
        if trunk_times.dim() == 1:
            trunk_times = trunk_times.unsqueeze(-1)

        # Run DeepONet and decode
        y_pred, _ = self.model(branch_input, decode=True, trunk_times=trunk_times)

        return y_pred  # [B, K, S]


def main():
    device = torch.device(DEVICE)
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load config & build model
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    model = AEDeepONet(cfg).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        raw_sd = ckpt.state_dict()
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    clean_sd = _sanitize_state_for(model, raw_sd)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    # Ignore compile-only buffer if present
    missing = [k for k in missing if k != "_device_tensor"]
    unexpected = [k for k in unexpected if k != "_device_tensor"]
    if missing or unexpected:
        raise RuntimeError(
            "State_dict mismatch after sanitize.\n"
            f"Missing: {missing[:8]}\nUnexpected: {unexpected[:8]}"
        )

    # Check if bypass mode
    bypass = bool(cfg["model"].get("bypass_autoencoder", False))

    if bypass:
        print("[WARN] Model was trained with bypass_autoencoder=True")
        print("       The encoder is not used in this case.")

    # Get dimensions
    num_species = len(cfg["data"]["species_variables"])
    num_globals = len(cfg["data"]["global_variables"])
    latent_dim = cfg["model"]["latent_dim"]

    # Build wrapper
    wrapper = CompleteInferenceWrapper(model).to(device).eval()

    # --- Export with dynamic shapes ---
    print("[INFO] Exporting model with encoder...")

    # Example inputs
    ex_x0 = torch.randn(2, num_species, device=device, dtype=torch.float32)
    ex_globals = torch.randn(2, num_globals, device=device, dtype=torch.float32)
    ex_times = torch.linspace(0.0, 1.0, steps=EX_TRUNK_STEPS, device=device, dtype=torch.float32)

    # Dynamic dimensions
    Bdim = Dim("batch", min=1, max=8192)
    Kdim = Dim("K", min=1, max=10000)

    prog = texport(
        wrapper,
        args=(ex_x0, ex_globals, ex_times),
        dynamic_shapes=(
            {0: Bdim},  # x0_norm: dynamic batch
            {0: Bdim},  # globals_norm: dynamic batch
            {0: Kdim},  # trunk_times: dynamic K
        ),
        strict=False,
    )

    out_path = model_dir / "complete_model_exported.pt2"
    tsave(prog, str(out_path))
    print(f"[OK] Complete model exported -> {out_path}")

    # Test the exported model
    print("\n[INFO] Testing exported model...")
    try:
        from torch.export import load as tload
        loaded_prog = tload(str(out_path)).module()

        # Test with different shapes
        test_x0 = torch.randn(1, num_species, device=device)
        test_globals = torch.randn(1, num_globals, device=device)
        test_times = torch.linspace(0, 1, 32, device=device)

        out = loaded_prog(test_x0, test_globals, test_times)
        print(f"  Test: input [{test_x0.shape}, {test_globals.shape}, {test_times.shape}] -> output {out.shape}")
        print("[OK] Exported model works!")

    except Exception as e:
        print(f"[WARN] Could not test: {e}")


if __name__ == "__main__":
    main()