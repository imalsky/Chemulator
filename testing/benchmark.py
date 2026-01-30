#!/usr/bin/env python3
"""Benchmark exported 1-step autoregressive module vs batch size (CPU vs MPS).

Measures a single autoregressive jump y_next = model(y, dt, g) for different batch sizes.
Prints amortized latency per sample (microseconds/sample) and saves a log-log plot.

Expected artifacts in CFG.run_dir (produced by testing/export.py):
  - export_cpu_1step.pt2
  - export_mps_1step.pt2 (optional)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# -------------------------
# User-tunable global knobs
# -------------------------
# Input dtype for benchmarking. Set to one of:
#   - "auto"     (recommended): infer from the loaded exported module
#   - "float32"  (fp32)
#   - "float16"  (fp16)
# NOTE: For torch.export artifacts, dtype is typically baked into the graph.
DTYPE: str = "auto"

# If False and backend==mps, override mismatched requested dtype to the module dtype.
# (Prevents MPSGraph hard-aborts from mixed dtypes.)
ALLOW_UNSAFE_DTYPE_MISMATCH_ON_MPS: bool = False


@dataclass
class Config:
    run_dir: Path = ROOT / "models" / "v4"
    processed_dir: Path = ROOT / "data" / "processed"

    export_cpu: str = "export_cpu_1step.pt2"
    export_mps: str = "export_mps_1step.pt2"

    warmup: int = 50
    iters: int = 200

    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


CFG = Config()


def _resolve_processed_dir() -> Path:
    fallback = CFG.processed_dir.resolve()
    if (fallback / "normalization.json").exists():
        return fallback

    for cfg_path in (CFG.run_dir / "config.resolved.json", CFG.run_dir / "config.json", ROOT / "config.json"):
        if not cfg_path.exists():
            continue
        try:
            import json

            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        paths = cfg.get("paths", {}) or {}
        p = paths.get("processed_dir") or paths.get("processed_data_dir")
        if not isinstance(p, str) or not p.strip():
            continue
        cand = Path(p).expanduser()
        if not cand.is_absolute():
            cand = (cfg_path.parent / cand).resolve()
        if (cand / "normalization.json").exists():
            return cand

    return fallback


def _infer_dims(processed_dir: Path) -> tuple[int, int]:
    import json

    man = json.loads((processed_dir / "normalization.json").read_text())
    species = list(man.get("species_variables", []) or [])
    globals_ = list(man.get("global_variables", []) or [])
    if not species:
        raise ValueError(f"No species_variables in {processed_dir / 'normalization.json'}")
    return len(species), len(globals_)


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass


def _load_export(export_path: Path, device: str) -> torch.nn.Module:
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    # torch.export.load(...) returns ExportedProgram; call .module() to get nn.Module.
    m = torch.export.load(export_path).module()

    # Some CPU exports may reject .to('mps'); keep best effort.
    try:
        m = m.to(device)
    except Exception:
        m = m.to("cpu")
    return m


def _infer_model_dtype(m: torch.nn.Module) -> torch.dtype:
    try:
        for p in m.parameters():
            return p.dtype
        for b in m.buffers():
            return b.dtype
    except Exception:
        pass
    return torch.float32


def _parse_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in {"fp32", "float32", "f32"}:
        return torch.float32
    if s in {"fp16", "float16", "f16", "half"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported DTYPE='{s}'. Use auto|float32|float16 (or fp32|fp16).")


def _select_input_dtype(model: torch.nn.Module, *, backend: str) -> torch.dtype:
    model_dtype = _infer_model_dtype(model)
    req = DTYPE.strip().lower()

    if req == "auto":
        return model_dtype

    dt = _parse_dtype(req)

    if backend == "mps":
        if dt == torch.bfloat16:
            raise ValueError("DTYPE=bfloat16 is not supported on MPS; use float16 or float32.")
        if (dt != model_dtype) and (not ALLOW_UNSAFE_DTYPE_MISMATCH_ON_MPS):
            print(
                f"mps: requested dtype={str(dt).replace('torch.', '')} but export/module dtype is {str(model_dtype).replace('torch.', '')}; "
                "using module dtype to avoid MPSGraph abort."
            )
            return model_dtype

    return dt


def _dtype_tag(dt: torch.dtype) -> str:
    if dt == torch.float32:
        return "fp32"
    if dt == torch.float16:
        return "fp16"
    if dt == torch.bfloat16:
        return "bf16"
    return str(dt).replace("torch.", "")


@torch.inference_mode()
def _supports_batch(model: torch.nn.Module, *, device: str, dtype: torch.dtype, B: int, S: int, G: int) -> bool:
    try:
        y = torch.randn(B, S, device=device, dtype=dtype)
        dt = torch.rand(B, device=device, dtype=dtype)
        g = torch.randn(B, G, device=device, dtype=dtype) if G > 0 else torch.empty(B, 0, device=device, dtype=dtype)
        _ = model(y, dt, g)
        _sync(device)
        return True
    except Exception:
        return False


@torch.inference_mode()
def _bench_one(model: torch.nn.Module, *, device: str, dtype: torch.dtype, B: int, S: int, G: int, iters: int, warmup: int) -> float:
    """Return average seconds per 1-step *call* (not per-sample)."""
    y = torch.randn(B, S, device=device, dtype=dtype)
    dt = torch.rand(B, device=device, dtype=dtype)
    g = torch.randn(B, G, device=device, dtype=dtype) if G > 0 else torch.empty(B, 0, device=device, dtype=dtype)

    for _ in range(max(1, warmup)):
        _ = model(y, dt, g)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        _ = model(y, dt, g)
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) / float(max(1, iters))


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return

    headers = ["backend", "dtype", "B", "us/sample"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        widths["backend"] = max(widths["backend"], len(str(r["backend"])))
        widths["dtype"] = max(widths["dtype"], len(str(r["dtype"])))
        widths["B"] = max(widths["B"], len(str(r["B"])))
        widths["us/sample"] = max(widths["us/sample"], len(f"{r['us_per_sample']:.3f}"))

    print(
        f"{headers[0]:<{widths['backend']}}  "
        f"{headers[1]:<{widths['dtype']}}  "
        f"{headers[2]:>{widths['B']}}  "
        f"{headers[3]:>{widths['us/sample']}}"
    )
    print("-" * (sum(widths.values()) + 6))

    for r in rows:
        print(
            f"{str(r['backend']):<{widths['backend']}}  "
            f"{str(r['dtype']):<{widths['dtype']}}  "
            f"{int(r['B']):>{widths['B']}}  "
            f"{r['us_per_sample']:>{widths['us/sample']}.3f}"
        )


def _try_science_style(plt) -> None:
    # Try repo style sheet.
    try:
        plt.style.use("science.mplystyle")
        return
    except Exception:
        pass

    try:
        plt.style.use("science.mplstyle")
        return
    except Exception:
        pass

    # Best-effort fallbacks.
    for rel in ("science.mplstyle", "misc/science.mplstyle", "science.mplystyle", "misc/science.mplystyle"):
        p = (ROOT / rel).resolve()
        if p.exists():
            try:
                plt.style.use(str(p))
                return
            except Exception:
                pass


def _save_plot(rows: list[dict], run_dir: Path) -> Path | None:
    if not rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plot")
        return None

    _try_science_style(plt)

    plots_dir = (run_dir / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((str(r["backend"]), str(r["dtype"])), []).append(r)

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda x: int(x["B"]))

    plt.figure()
    for (backend, dtype), rs in grouped.items():
        bs = [int(r["B"]) for r in rs]
        us = [float(r["us_per_sample"]) for r in rs]
        plt.plot(bs, us, marker="o", label=f"{backend} ({dtype})")

    plt.xlabel("batch size B")
    plt.ylabel("amortized latency (\u03bcs/sample)")
    plt.title("1-step AR benchmark (amortized per-sample)")

    # Log-log axes.
    try:
        plt.xscale("log", base=2)
    except TypeError:
        plt.xscale("log", basex=2)
    plt.yscale("log")

    #plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = DTYPE.strip().lower().replace("float", "fp")
    out = plots_dir / f"benchmark_1step_us_per_sample_{tag}_{ts}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main() -> None:
    processed_dir = _resolve_processed_dir()
    S, G = _infer_dims(processed_dir)

    print(f"processed_dir: {processed_dir}")
    print(f"run_dir:       {CFG.run_dir}")
    print(f"dims:          S={S} G={G}")
    print(f"dtype:         {DTYPE}")
    print("")

    rows: list[dict] = []

    # CPU
    cpu_export = (CFG.run_dir / CFG.export_cpu).resolve()
    model_cpu = _load_export(cpu_export, "cpu")
    cpu_dtype = _select_input_dtype(model_cpu, backend="cpu")

    for B in CFG.batch_sizes:
        if not _supports_batch(model_cpu, device="cpu", dtype=cpu_dtype, B=B, S=S, G=G):
            print(f"cpu: batch B={B} unsupported by export; skipping")
            continue

        sec_per_call = _bench_one(
            model_cpu,
            device="cpu",
            dtype=cpu_dtype,
            B=B,
            S=S,
            G=G,
            iters=CFG.iters,
            warmup=CFG.warmup,
        )
        rows.append(
            {
                "backend": "cpu",
                "dtype": _dtype_tag(cpu_dtype),
                "B": int(B),
                "us_per_sample": (1e6 * sec_per_call) / float(B),
            }
        )

    # MPS (optional)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_export = (CFG.run_dir / CFG.export_mps).resolve()
        if mps_export.exists():
            model_mps = _load_export(mps_export, "mps")
            mps_dtype = _select_input_dtype(model_mps, backend="mps")

            for B in CFG.batch_sizes:
                if not _supports_batch(model_mps, device="mps", dtype=mps_dtype, B=B, S=S, G=G):
                    print(f"mps: batch B={B} unsupported by export (dtype={_dtype_tag(mps_dtype)}); skipping")
                    continue

                sec_per_call = _bench_one(
                    model_mps,
                    device="mps",
                    dtype=mps_dtype,
                    B=B,
                    S=S,
                    G=G,
                    iters=CFG.iters,
                    warmup=CFG.warmup,
                )
                rows.append(
                    {
                        "backend": "mps",
                        "dtype": _dtype_tag(mps_dtype),
                        "B": int(B),
                        "us_per_sample": (1e6 * sec_per_call) / float(B),
                    }
                )
        else:
            print(f"MPS available, but export not found: {mps_export}")
    else:
        print("MPS not available; skipping MPS benchmark.")

    print("")
    _print_table(rows)

    out = _save_plot(rows, CFG.run_dir)
    if out is not None:
        print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
