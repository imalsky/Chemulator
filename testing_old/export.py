#!/usr/bin/env python3
"""Export trained model to export_eager.pt with config."""

import json
import sys
import os
from pathlib import Path
import torch

# Configuration
MODEL_SUBDIR = "models/v1"
EXPORT_NAME = "export_eager.pt"

# Paths
REPO = Path(__file__).parent.parent
MODEL_DIR = REPO / MODEL_SUBDIR
CONFIG_FILE = MODEL_DIR / "config.json"
PREP_SUMMARY = REPO / "data/processed_medium/preprocessing_summary.json"


def main():
    # Load and patch config
    cfg = json.loads(CONFIG_FILE.read_text())

    # Ensure data section exists
    if "data" not in cfg:
        cfg["data"] = {}

    # Patch missing fields from preprocessing summary if needed
    if not cfg["data"].get("species_variables") or \
            "global_variables" not in cfg["data"] or \
            "time_variable" not in cfg["data"]:

        prep = json.loads(PREP_SUMMARY.read_text())

        # Fill in missing fields
        if not cfg["data"].get("species_variables"):
            cfg["data"]["species_variables"] = prep["species_variables"]
        if "global_variables" not in cfg["data"]:
            cfg["data"]["global_variables"] = prep.get("global_variables", [])
        if "time_variable" not in cfg["data"]:
            cfg["data"]["time_variable"] = prep["time_variable"]

    # Set working directory for relative paths in create_model
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))

    # Create model
    from model import create_model
    model = create_model(cfg).eval()

    # Try to load checkpoint (check multiple possible locations)
    state_dict = None

    # Try best_model.pt first
    if (MODEL_DIR / "best_model.pt").exists():
        obj = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu")
        if hasattr(obj, "state_dict"):
            state_dict = obj.state_dict()
        elif isinstance(obj, dict):
            # Direct state dict
            if all(hasattr(v, "shape") for v in obj.values()):
                state_dict = obj
            # Nested under various keys
            else:
                for key in ["state_dict", "model_state_dict", "weights", "model"]:
                    if key in obj:
                        val = obj[key]
                        state_dict = val.state_dict() if hasattr(val, "state_dict") else val
                        break

    # Try .ckpt files if needed
    if state_dict is None:
        for ckpt_file in ["best.ckpt", "last.ckpt"]:
            path = MODEL_DIR / ckpt_file
            if path.exists():
                obj = torch.load(path, map_location="cpu")
                if isinstance(obj, dict) and "state_dict" in obj:
                    state_dict = obj["state_dict"]
                    break

    # Clean up key names (remove common prefixes)
    prefixes = ["_orig_mod.", "module.", "model.", "_orig_mod.module.", "net."]
    model_keys = set(model.state_dict().keys())
    clean_dict = {}

    for k, v in state_dict.items():
        # Strip prefixes
        clean_k = k
        for prefix in prefixes:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                break

        # Keep only keys that exist in model
        if clean_k in model_keys:
            clean_dict[clean_k] = v

    # Load weights
    missing, unexpected = model.load_state_dict(clean_dict, strict=False)
    if missing:
        print(f"[export] Missing keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        print(f"[export] Unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    # Save exported model
    out_path = MODEL_DIR / EXPORT_NAME
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg
    }, str(out_path))

    print(f"[export] Saved to: {out_path}")


if __name__ == "__main__":
    main()