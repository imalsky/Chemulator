#!/usr/bin/env python3
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Paths to exported model + metadata
MODEL_DIR  = Path("../models/subset_big")
MODEL_PATH = MODEL_DIR / "standalone_phys_k1_cpu.pt2"
META_PATH  = MODEL_DIR / "standalone_phys_metadata.json"

# Load exported standalone physical-I/O model
ep = torch.export.load(MODEL_PATH)
model = ep.module()  # (y_phys, dt_sec, g_phys) -> y_next_phys

# Load metadata: species/global order + means
with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

species_order = meta["species_order"]
species_mean  = meta["species_mean"]
globals_order = meta["globals_order"]
globals_mean  = meta["globals_mean"]

# Build inputs directly from the mean values in the metadata
species_vals = [float(species_mean[name]) for name in species_order]
globals_vals = [float(globals_mean[name]) for name in globals_order]

y_phys = torch.tensor([species_vals], dtype=torch.float32)   # [1, S]
g_phys = torch.tensor([globals_vals], dtype=torch.float32)   # [1, G]

# Vectorized over dt and species, plotting outputs
dts = torch.logspace(0.1, 5, 100)
B = dts.numel()
y_batch = y_phys.repeat(B, 1)
g_batch = g_phys.repeat(B, 1)
out = model(y_batch, dts.view(-1, 1), g_batch)[:, 0, :].detach().numpy()  # [B, S]

cmap = plt.get_cmap("tab20")
x = dts.numpy()
for i in range(out.shape[1]):
    plt.scatter(x, out[:, i], color=cmap(i / max(1, out.shape[1] - 1)))

plt.xscale("log")
plt.yscale("log")
plt.show()
