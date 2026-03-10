#!/usr/bin/env python3
"""
Example script for Xi

Put this script next to `model.pt2`, then run:

    python stand_alone_run.py

The exported model call is:

    y_next_phys = model(y_phys, dt_seconds, g_phys)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch._export.serde.serialize import SerializedArtifact, deserialize


EXPORT_PATH = Path(__file__).with_name("model.pt2")
PLOT_PATH = Path(__file__).with_name("test_plot_dt_1.png")
DEVICE = torch.device("cpu")
ROLLOUT_STEPS = 1000


def _parse_export_dtype(raw: Any) -> torch.dtype:
    if not isinstance(raw, str) or not raw.strip():
        raise KeyError("metadata.json is missing export_dtype")

    value = raw.strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported export_dtype in metadata: {raw!r}")


print("Step 1: load the exported model")
print(f"export: {EXPORT_PATH.resolve()}")


# Loading the model
with zipfile.ZipFile(EXPORT_PATH, "r") as zf:
    archive_root = zf.namelist()[0].split("/", 1)[0] + "/"
    metadata = json.loads(zf.read(f"{archive_root}extra/metadata.json").decode("utf-8"))
    infer_dtype = _parse_export_dtype(metadata.get("export_dtype"))
    artifact = SerializedArtifact(exported_program=zf.read(f"{archive_root}models/model.json"),
                                  state_dict=zf.read(f"{archive_root}data/weights/model.pt"),
                                  constants=zf.read(f"{archive_root}data/constants/model.pt"),
                                  example_inputs=zf.read(f"{archive_root}data/sample_inputs/model.pt"))

# You have to match the specific ordering
model = deserialize(artifact).module().to(device=DEVICE, dtype=infer_dtype)
species_names = list(metadata["species_variables"])
global_names = list(metadata["global_variables"])

print(f"species order: {species_names}")
print(f"global order:  {global_names}")
print(f"model dtype:   {str(infer_dtype).replace('torch.', '')}")
print()

print("Step 2: build one simple input")
# y_phys shape: (1, n_species)
y_phys = torch.tensor(
    [[
        3.4868813636618317e-21,
        2.5886661447151851e-07,
        7.0948297762895706e-05,
        8.7119720430653403e-04,
        3.3354580889405328e-06,
        9.9853693226383833e-01,
        2.2550379422768412e-07,
        2.0833886111604422e-09,
        5.1710011838340100e-04,
        2.0382214416051654e-10,
        4.0335057639479328e-16,
        8.4217173529958508e-25,
    ]],
    device=DEVICE,
    dtype=infer_dtype,
)

# dt_seconds shape: (1,)
dt_value = 1
dt_seconds = torch.tensor([dt_value], device=DEVICE, dtype=infer_dtype)

# g_phys shape: (1, n_globals)
g_phys = torch.zeros((1, len(global_names)), device=DEVICE, dtype=infer_dtype)
g_phys[0, global_names.index("P")] = 6.909815748102695e8
g_phys[0, global_names.index("T")] = 1127.748742423797

print("y_phys:")
print(y_phys)
print()
print("dt_seconds:")
print(dt_seconds)
print()
print("g_phys:")
print(g_phys)
print()


print("Step 3: run an explicit autoregressive rollout")
print(f"input shapes: y={tuple(y_phys.shape)} dt={tuple(dt_seconds.shape)} g={tuple(g_phys.shape)}")

times = []
rollout = []
y_current = y_phys

with torch.inference_mode():
    for step_number in range(1, ROLLOUT_STEPS + 1):
        # Autoregressive
        y_current = model(y_current, dt_seconds, g_phys)
        times.append(step_number * dt_value)
        rollout.append(y_current[0].cpu())

rollout_tensor = torch.stack(rollout, dim=0)
composition_tensor = torch.clamp(rollout_tensor, min=1.0e-30)
composition_tensor = composition_tensor / composition_tensor.sum(dim=1, keepdim=True)

print(f"rollout shape: {tuple(rollout_tensor.shape)}")
print(f"time shape:    ({len(times)},)")
print()

print("final autoregressive composition:")
for name, value in zip(species_names, composition_tensor[-1].tolist()):
    print(f"  {name:>20s} : {value:.6e}")
print()
print(sum(composition_tensor[-1].tolist()))



print("Step 4: save a simple plot")
figure, axis = plt.subplots(figsize=(5.5, 5.5))
colors = plt.cm.tab20.colors
plot_species_names = [name.removesuffix("_evolution") for name in species_names]
for species_index, species_name in enumerate(plot_species_names):
    axis.plot(
        times,
        composition_tensor[:, species_index].numpy(),
        color=colors[species_index % len(colors)],
        linewidth=2.0,
        label=species_name,
    )

axis.set_xlim(10, 1000)
axis.set_ylim(1e-30, 3)
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel("Time (s)")
axis.set_ylabel("Mixing Ratio")
axis.set_box_aspect(1)
axis.minorticks_on()
axis.tick_params(which="both", direction="in", top=True, right=True)
axis.legend(ncol=2, fontsize=9, frameon=False)
figure.tight_layout()
figure.savefig(PLOT_PATH, dpi=150)
plt.close(figure)

print(f"saved plot: {PLOT_PATH.resolve()}")
