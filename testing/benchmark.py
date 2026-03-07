#!/usr/bin/env python3
"""Benchmark physical-I/O model throughput."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import prepare_platform_environment

prepare_platform_environment()

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).with_name("science.mplstyle")
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_version"))
).expanduser().resolve()

CPU_MODEL = MODEL_DIR / "physical_model_k1_cpu.pt2"
GPU_MODEL = MODEL_DIR / "physical_model_k1_gpu.pt2"  # optional
META_PATH = MODEL_DIR / "physical_model_metadata.json"

WARMUP_STEPS = 10
MEASURE_STEPS = 200
# Keep benchmark probes within the export wrapper's declared dynamic batch range.
EXPORT_MAX_BATCH = 4096
MAX_BATCH_CPU = EXPORT_MAX_BATCH
MAX_BATCH_GPU = EXPORT_MAX_BATCH

VULCAN_BASELINE_Y = 100.0
YMIN, YMAX = 10.0, 1.0e7


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pt2(path: Path):
    ep = torch.export.load(path)
    return ep.module()


def _p2_batches(max_cap: int) -> List[int]:
    out: List[int] = []
    b = 1
    while b <= max_cap:
        out.append(b)
        b <<= 1
    return out


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def _make_inputs(
    *,
    B: int,
    species_mean: np.ndarray,
    globals_mean: np.ndarray,
    dt_min: float,
    dt_max: float,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    S = species_mean.shape[0]
    G = globals_mean.shape[0]

    noise_s = 0.01 * np.random.randn(B, S).astype(np.float32)
    noise_g = 0.01 * np.random.randn(B, G).astype(np.float32) if G > 0 else np.empty((B, 0), dtype=np.float32)

    y = np.maximum(species_mean[None, :] * (1.0 + noise_s), 1e-35).astype(np.float32)
    g = np.maximum(globals_mean[None, :] * (1.0 + noise_g), 1e-35).astype(np.float32) if G > 0 else noise_g

    log_min = np.log10(max(dt_min, 1e-12))
    log_max = np.log10(max(dt_max, dt_min * 1.0001))
    dt = np.power(10.0, np.random.uniform(log_min, log_max, size=(B, 1))).astype(np.float32)

    y_t = torch.from_numpy(y).to(device=device, dtype=dtype)
    dt_t = torch.from_numpy(dt).to(device=device, dtype=dtype)
    g_t = torch.from_numpy(g).to(device=device, dtype=dtype)
    return y_t, dt_t, g_t


def _bench_device(
    *,
    label: str,
    model_path: Path,
    device: str,
    dtype: torch.dtype,
    species_mean: np.ndarray,
    globals_mean: np.ndarray,
    dt_min: float,
    dt_max: float,
    max_batch: int,
) -> Tuple[List[int], List[float]]:
    model = _load_pt2(model_path)
    batches = _p2_batches(max_batch)
    throughput: List[float] = []
    ok_batches: List[int] = []

    print(
        f"- {label}: model={model_path.name} device={device} dtype={str(dtype).replace('torch.', '')} "
        f"B in {batches}"
    )

    for B in batches:
        try:
            inp = _make_inputs(
                B=B,
                species_mean=species_mean,
                globals_mean=globals_mean,
                dt_min=dt_min,
                dt_max=dt_max,
                device=device,
                dtype=dtype,
            )

            for _ in range(WARMUP_STEPS):
                with torch.inference_mode():
                    _ = model(*inp)
            _sync(device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(MEASURE_STEPS):
                    _ = model(*inp)
            _sync(device)

            elapsed = time.perf_counter() - t0
            sps = (MEASURE_STEPS * B) / max(elapsed, 1e-12)
            throughput.append(float(sps))
            ok_batches.append(B)
            print(f"    B={B:>5d} -> {sps:,.1f} samples/s")
        except Exception as e:
            print(f"    B={B:>5d} failed: {e}")
            break

    return ok_batches, throughput


def _autoset_log_xlim(ax, curves: List[Tuple[str, List[int], List[float]]]) -> None:
    xs: List[float] = []
    for _, bs, _ in curves:
        xs.extend(float(b) for b in bs if b > 0)
    if not xs:
        return
    pad = 1.25
    ax.set_xlim(max(min(xs) / pad, 1e-6), max(xs) * pad)


def main() -> int:
    if not CPU_MODEL.is_file():
        raise FileNotFoundError(f"Missing CPU model artifact: {CPU_MODEL}")

    meta = _load_json(META_PATH)
    species_order = list(meta["species_order"])
    globals_order = list(meta["globals_order"])
    species_mean = np.asarray([float(meta["species_mean"][k]) for k in species_order], dtype=np.float32)
    globals_mean = np.asarray([float(meta["globals_mean"][k]) for k in globals_order], dtype=np.float32)
    dt_bounds = dict(meta.get("dt_bounds_sec", {}))
    dt_min = float(dt_bounds.get("min", 1e-3))
    dt_max = float(dt_bounds.get("max", 1e8))

    curves: List[Tuple[str, List[int], List[float]]] = []
    png_out = MODEL_DIR / "plots" / "bench_throughput_k1.png"

    bs_cpu, thr_cpu = _bench_device(
        label="CPU",
        model_path=CPU_MODEL,
        device="cpu",
        dtype=torch.float32,
        species_mean=species_mean,
        globals_mean=globals_mean,
        dt_min=dt_min,
        dt_max=dt_max,
        max_batch=MAX_BATCH_CPU,
    )
    if thr_cpu:
        curves.append(("CPU", bs_cpu, thr_cpu))

    if torch.cuda.is_available() and GPU_MODEL.is_file():
        bs_gpu, thr_gpu = _bench_device(
            label="GPU",
            model_path=GPU_MODEL,
            device="cuda",
            dtype=torch.float32,
            species_mean=species_mean,
            globals_mean=globals_mean,
            dt_min=dt_min,
            dt_max=dt_max,
            max_batch=MAX_BATCH_GPU,
        )
        if thr_gpu:
            curves.append(("GPU", bs_gpu, thr_gpu))
    else:
        print("- GPU: skipped (CUDA unavailable or optional GPU artifact missing)")

    if not curves:
        print("No curves to plot.")
        return 0

    try:
        plt.style.use(str(STYLE_PATH))
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, bs, th in curves:
        ax.plot(bs, th, marker="o", label=label)
    ax.axhline(y=VULCAN_BASELINE_Y, linewidth=2, color="gray", linestyle="--", label="VULCAN, CPU")

    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Throughput (samples/second)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    _autoset_log_xlim(ax, curves)
    ax.legend(loc="best")
    ax.set_box_aspect(1)

    fig.tight_layout()
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
