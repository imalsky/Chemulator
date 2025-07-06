#!/usr/bin/env python3
"""
quick_demo.py – run from test/ to sanity-check the JIT model.

Project layout assumed:
  your_repo/
    ├─ data/
    ├─ src/
    ├─ trained_model_siren/
    └─ test/
         └─ quick_demo.py   ← you are here
"""

import json
import random
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import torch

# ── locate repo root and patch sys.path so we can import from src/ ────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from normalizer import DataNormalizer, DTYPE  # now import works

# ──────────────────────────────────────────────────────────────────────────────
TRAINED_DIR  = ROOT / "data" / "trained_model_siren"
H5_DATASET   = ROOT / "data" / "chem_data" / "data.h5"
SPECIES_NAME = "H2O_evolution"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── grab metadata ---------------------------------------------------------
    with open(TRAINED_DIR / "training_metadata.json") as f:
        meta_train = json.load(f)
    test_indices = meta_train["test_set_indices"]
    profile_idx  = random.choice(test_indices)
    print(f"Chosen profile from test set: {profile_idx}")

    with open(TRAINED_DIR / "normalization_metadata.json") as f:
        norm_meta = json.load(f)

    species_vars = [k for k in norm_meta["normalization_methods"]
                    if k.endswith("_evolution")]
    global_vars  = ["P_init", "T_init"]

    # handy aliases
    norm   = DataNormalizer.normalize_tensor
    denorm = DataNormalizer.denormalize_tensor

    # ── load one profile ------------------------------------------------------
    with h5py.File(H5_DATASET, "r") as hf:
        times    = hf["t_time"][profile_idx]                          # (100,)
        globals_ = torch.tensor([hf[g][profile_idx] for g in global_vars],
                                 dtype=DTYPE)
        species  = torch.tensor([hf[s][profile_idx] for s in species_vars],
                                 dtype=DTYPE)                        # (S, 100)

    # ── build per-timestep feature vectors -----------------------------------
    feats = []
    for t, τ in enumerate(times):
        feats.append(torch.cat([species[:, t], globals_, torch.tensor([τ])]))
    feats = torch.stack(feats)                                       # (100, F)

    # normalise column-wise
    cols = []
    for i, var in enumerate(species_vars + global_vars + ["t_time"]):
        cols.append(norm(feats[:, i], norm_meta["normalization_methods"][var],
                         norm_meta["per_key_stats"][var]))
    model_in = torch.stack(cols, dim=1).to(DEVICE)

    # ── run model -------------------------------------------------------------
    model = torch.jit.load(TRAINED_DIR / "best_model.pt", map_location=DEVICE)
    model.eval()
    with torch.no_grad():
        pred_norm = model(model_in)

    # de-normalise predictions (species only)
    preds = []
    for i, var in enumerate(species_vars):
        preds.append(denorm(pred_norm[:, i].cpu(),
                            norm_meta["normalization_methods"][var],
                            norm_meta["per_key_stats"][var]))
    preds = torch.stack(preds, dim=1)                                # (100, S)

    # ── plot one species ------------------------------------------------------
    s_idx = species_vars.index(SPECIES_NAME)
    plt.figure(figsize=(7, 4))
    plt.plot(times, species[s_idx], label="True")
    plt.plot(times, preds[:, s_idx], "--", label="Predicted")
    plt.xlabel("Time")
    plt.ylabel(SPECIES_NAME)
    plt.title(f"Profile {profile_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
