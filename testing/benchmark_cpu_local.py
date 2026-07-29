#!/usr/bin/env python3
"""Local CPU latency/throughput benchmark for the paper.

Bypasses torch.export.load (which fails when the .pt2 was saved by a newer
torch than the local install) by reconstructing PhysicalModelWrapper from
best.ckpt + config.json + normalization.json. Same forward path as the
exported .pt2.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.runtime import prepare_platform_environment  # noqa: E402

prepare_platform_environment()

import os as _os
if _os.environ.get("BENCH_THREADS"):
    _n = int(_os.environ["BENCH_THREADS"])
    torch.set_num_threads(_n)
    try:
        torch.set_num_interop_threads(max(1, _n))
    except RuntimeError:
        pass

from src import model as model_module  # noqa: E402
from testing.export import (  # noqa: E402
    PhysicalModelWrapper,
    _build_group_tensors,
    _extract_state_dict,
    _find_checkpoint,
    _load_json,
    _require_list,
)

MODEL_DIR = REPO / "models" / "final_model"
WARMUP = 200
import os as _os2
_b = _os2.environ.get("BENCH_BATCHES")
BATCHES = [int(x) for x in _b.split(",")] if _b else [1, 4, 16, 64, 256, 1024, 4096]


def build_wrapper() -> tuple[PhysicalModelWrapper, list[str], list[str]]:
    cfg = _load_json(MODEL_DIR / "config.json")
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    if not processed_dir.is_absolute():
        processed_dir = (REPO / processed_dir).resolve()
    candidates = [
        processed_dir / "normalization.json",
        processed_dir / "processed" / "normalization.json",
        REPO / "data" / "processed" / "normalization.json",
    ]
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"normalization.json not found in: {candidates}")
    manifest = _load_json(manifest_path)

    data_cfg = cfg["data"]
    species_keys = _require_list(data_cfg, "species_variables")
    global_keys = _require_list(data_cfg, "global_variables")

    base = model_module.create_model(cfg).cpu().eval()
    ckpt_path = _find_checkpoint(MODEL_DIR)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(payload)
    missing, unexpected = base.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    for p in base.parameters():
        p.requires_grad_(False)

    per_key_stats = manifest["per_key_stats"]
    methods = manifest["normalization_methods"]
    min_std = float(manifest["min_std"])
    dt_spec = manifest["dt"]

    species_tensors = _build_group_tensors(
        keys=species_keys, methods=methods, per_key_stats=per_key_stats, min_std=min_std,
    )
    global_tensors = _build_group_tensors(
        keys=global_keys, methods=methods, per_key_stats=per_key_stats, min_std=min_std,
    )

    wrapped = PhysicalModelWrapper(
        base_model=base,
        species_keys=species_keys,
        global_keys=global_keys,
        species_tensors=species_tensors,
        global_tensors=global_tensors,
        min_std=min_std,
        dt_log_min=float(dt_spec["log_min"]),
        dt_log_max=float(dt_spec["log_max"]),
    ).cpu().eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)
    return wrapped, species_keys, global_keys


def make_inputs(B: int, meta: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    species_order = list(meta["species_order"])
    globals_order = list(meta["globals_order"])
    species_mean = np.asarray([meta["species_mean"][k] for k in species_order], dtype=np.float32)
    globals_mean = np.asarray([meta["globals_mean"][k] for k in globals_order], dtype=np.float32)
    dt_min = float(meta["dt_bounds_sec"]["min"])
    dt_max = float(meta["dt_bounds_sec"]["max"])

    rng = np.random.default_rng(0)
    noise_s = 0.01 * rng.standard_normal((B, species_mean.size)).astype(np.float32)
    noise_g = 0.01 * rng.standard_normal((B, globals_mean.size)).astype(np.float32)
    y = np.maximum(species_mean[None, :] * (1.0 + noise_s), 1e-35).astype(np.float32)
    g = np.maximum(globals_mean[None, :] * (1.0 + noise_g), 1e-35).astype(np.float32)
    log_min, log_max = np.log10(dt_min), np.log10(dt_max)
    dt = np.power(10.0, rng.uniform(log_min, log_max, size=(B, 1))).astype(np.float32)
    return torch.from_numpy(y), torch.from_numpy(dt), torch.from_numpy(g)


def time_call(model, inp, n_calls: int) -> list[float]:
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(n_calls):
            t0 = time.perf_counter()
            _ = model(*inp)
            times.append(time.perf_counter() - t0)
    return times


def bench(model, meta: dict, B: int, target_seconds: float = 5.0) -> dict:
    inp = make_inputs(B, meta)
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(*inp)
    # one timed call to estimate iteration count
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(*inp)
    one = max(time.perf_counter() - t0, 1e-7)
    n_calls = max(50, min(20000, int(target_seconds / one)))
    times = time_call(model, inp, n_calls)
    times_sorted = sorted(times)
    median = statistics.median(times_sorted)
    p10 = times_sorted[int(0.10 * len(times_sorted))]
    p90 = times_sorted[int(0.90 * len(times_sorted))]
    return {
        "B": B,
        "n_calls": n_calls,
        "median_s": median,
        "p10_s": p10,
        "p90_s": p90,
        "throughput_sps": B / median,
        "per_cell_us": (median / B) * 1e6,
    }


def main() -> int:
    print("== Host ==")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()}")
    print(f"  python   : {platform.python_version()}")
    print(f"  torch    : {torch.__version__}")
    print(f"  threads  : torch.get_num_threads()={torch.get_num_threads()}, "
          f"interop={torch.get_num_interop_threads()}")
    print()

    meta = _load_json(MODEL_DIR / "physical_model_metadata.json")
    print("== Model ==")
    print(f"  dir      : {MODEL_DIR.relative_to(REPO)}")
    print(f"  species  : {len(meta['species_order'])}")
    print(f"  globals  : {len(meta['globals_order'])}")
    dt_min = meta["dt_bounds_sec"]["min"]
    dt_max = meta["dt_bounds_sec"]["max"]
    print(f"  dt range : [{dt_min:.3e}, {dt_max:.3e}] s")
    print()

    print("Building wrapped model from checkpoint...")
    model, _species, _globals = build_wrapper()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")
    print()

    print(f"== CPU benchmark (threads={torch.get_num_threads()}) ==")
    print(f"{'B':>6} {'median_ms':>11} {'p10_ms':>10} {'p90_ms':>10} "
          f"{'samples/s':>14} {'µs / cell':>12} {'n_calls':>9}")
    rows = []
    for B in BATCHES:
        try:
            r = bench(model, meta, B)
        except RuntimeError as e:
            print(f"{B:>6}  failed: {e}")
            break
        rows.append(r)
        print(f"{r['B']:>6} {r['median_s']*1e3:>11.3f} {r['p10_s']*1e3:>10.3f} "
              f"{r['p90_s']*1e3:>10.3f} {r['throughput_sps']:>14,.1f} "
              f"{r['per_cell_us']:>12.3f} {r['n_calls']:>9d}")

    out = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "model_params": n_params,
        "results": rows,
    }
    out_path = MODEL_DIR / "reports" / f"cpu_bench_t{torch.get_num_threads()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
