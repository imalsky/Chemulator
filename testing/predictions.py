#!/usr/bin/env python3
"""
Autoregressive rollout evaluation for exported 1-step physical-space models.

Export signature:
    y_next_phys = model(y_phys, dt_seconds, g_phys)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------

# Model/artifact selection (main knobs you typically edit first).
MODEL_RUN_DIR: str = "models/v2_bigger_batch"  # Absolute path or repo-relative path
MODEL_EXPORT_FILE: str = "export_cpu_dynB_1step_phys.pt2"

# Inference settings.
INFER_DEVICE: torch.device = torch.device("cpu")
INFER_DTYPE: torch.dtype = torch.float32

# Data selection.
EVAL_SPLIT: str = "test"
EVAL_SHARD_INDEX: int | None = None  # None => randomly select a shard from the chosen split
EVAL_SHARD_RANDOM_SEED: int | None = None  # Optional reproducible seed; None => fresh randomness each run
EVAL_SAMPLE_INDEX: int | None = None  # None => randomly select a sample/chunk from the selected shard
EVAL_SAMPLE_RANDOM_SEED: int | None = None  # Optional reproducible seed; None => fresh randomness each run

# Rollout window.
ROLLOUT_START_INDEX: int = 1
ROLLOUT_STEPS: int = 200

# Plotting/output.
PLOTS_SUBDIR: str = "plots"  # Relative to MODEL_RUN_DIR
PLOT_TIME_LOG_SCALE: bool = False  # If True, use log-scale on time axis
PLOT_Y_LIM: Tuple[float, float] = (1e-30, 3.0)
PLOT_MARKER_EVERY: int = 10
PLOT_STYLE: str | None = str((ROOT / "testing" / "science.mplstyle").resolve())

REQUIRED_GLOBALS: Tuple[str, str] = ("P", "T")


def _resolve_repo_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p.resolve()


def _resolve_export_path(run_dir: Path) -> Path:
    return (run_dir / MODEL_EXPORT_FILE).resolve()


def _resolve_output_dir(run_dir: Path) -> Path:
    return (run_dir / PLOTS_SUBDIR).resolve()


def _load_export_module(export_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    extra_files = {"metadata.json": ""}
    ep = torch.export.load(str(export_path), extra_files=extra_files)
    meta_raw = extra_files.get("metadata.json", "")
    if not meta_raw:
        raise RuntimeError("export is missing embedded metadata.json")
    meta = json.loads(meta_raw)
    if not isinstance(meta, dict):
        raise TypeError("embedded metadata.json must be a JSON object")
    model = ep.module().to(device=INFER_DEVICE, dtype=INFER_DTYPE)
    return model, meta


def _resolve_normalization_path(meta: Mapping[str, Any]) -> Path:
    raw = meta.get("normalization_path")
    if not isinstance(raw, str) or not raw.strip():
        raise KeyError("export metadata missing normalization_path")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"normalization_path from export metadata not found: {p}")
    return p


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return obj


def _pick_methods_map(manifest: Mapping[str, Any]) -> Mapping[str, str]:
    methods = manifest.get("normalization_methods")
    if not isinstance(methods, Mapping):
        raise KeyError("normalization.json missing normalization_methods mapping")
    return methods  # type: ignore[return-value]


def _canonical_method(method: str) -> str:
    m = str(method).lower().strip()
    return m


def _dt_norm_to_seconds(dt_norm: np.ndarray, manifest: Mapping[str, Any]) -> np.ndarray:
    dt = manifest.get("dt")
    if not isinstance(dt, Mapping):
        raise KeyError("normalization.json missing dt")
    if "log_min" not in dt or "log_max" not in dt:
        raise KeyError("normalization.json missing dt.log_min/dt.log_max")

    a = float(dt["log_min"])
    b = float(dt["log_max"])
    return np.power(10.0, dt_norm.astype(np.float64) * (b - a) + a, dtype=np.float64)


def _denormalize_species(y_z: np.ndarray, species_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")
    methods = _pick_methods_map(manifest)

    for k in species_keys:
        m = _canonical_method(str(methods.get(k, "")))
        if m != "log-standard":
            raise ValueError(f"species normalization must be log-standard, got {m!r} for {k!r}")

    mu = np.array([float(stats[k]["log_mean"]) for k in species_keys], dtype=np.float64)
    sd = np.array([float(stats[k]["log_std"]) for k in species_keys], dtype=np.float64)
    return np.power(10.0, y_z.astype(np.float64) * sd + mu, dtype=np.float64)


def _denormalize_globals(g_z: np.ndarray, global_keys: List[str], manifest: Mapping[str, Any]) -> np.ndarray:
    if list(global_keys) != list(REQUIRED_GLOBALS):
        raise ValueError(f"global_variables must be exactly {list(REQUIRED_GLOBALS)}")

    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise KeyError("normalization.json missing per_key_stats")
    methods = _pick_methods_map(manifest)

    out = np.empty((len(global_keys),), dtype=np.float64)
    for i, key in enumerate(global_keys):
        st = stats[key]
        z = float(g_z[i])
        m = _canonical_method(str(methods.get(key, "")))

        if m == "min-max":
            lo = float(st["min"])
            hi = float(st["max"])
            out[i] = z * (hi - lo) + lo
            continue

        if m == "log-min-max":
            lo = float(st["log_min"])
            hi = float(st["log_max"])
            out[i] = 10.0 ** (z * (hi - lo) + lo)
            continue

        if m == "standard":
            mu = float(st["mean"])
            sd = float(st["std"])
            out[i] = z * sd + mu
            continue

        if m == "log-standard":
            mu = float(st["log_mean"])
            sd = float(st["log_std"])
            out[i] = 10.0 ** (z * sd + mu)
            continue

        raise ValueError(f"unsupported global normalization method {m!r} for {key!r}")

    return out


def _choose_shard(processed_dir: Path, *, split: str, shard_index: int | None) -> Tuple[int, Path]:
    shard_dir = processed_dir / split
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards found under {shard_dir}")

    if shard_index is None:
        rng = np.random.default_rng(EVAL_SHARD_RANDOM_SEED)
        picked = int(rng.integers(0, len(shards)))
        seed_msg = f", seed={EVAL_SHARD_RANDOM_SEED}" if EVAL_SHARD_RANDOM_SEED is not None else ""
        print(f"[data] randomly selected shard index {picked} of {len(shards)}{seed_msg}")
        return picked, shards[picked]

    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError(f"SHARD_INDEX out of range: {shard_index} (found {len(shards)} shards)")

    return shard_index, shards[shard_index]


def _load_shard(shard_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(shard_path) as f:
        y_mat = f["y_mat"].astype(np.float32, copy=False)
        g_mat = f["globals"].astype(np.float32, copy=False)
        dt_norm_mat = f["dt_norm_mat"].astype(np.float32, copy=False)
    return y_mat, g_mat, dt_norm_mat


def _choose_sample(n_samples: int, *, sample_index: int | None) -> int:
    if n_samples <= 0:
        raise ValueError(f"selected shard has no samples/chunks (N={n_samples})")

    if sample_index is None:
        rng = np.random.default_rng(EVAL_SAMPLE_RANDOM_SEED)
        picked = int(rng.integers(0, n_samples))
        seed_msg = f", seed={EVAL_SAMPLE_RANDOM_SEED}" if EVAL_SAMPLE_RANDOM_SEED is not None else ""
        print(f"[data] randomly selected sample index {picked} of {n_samples}{seed_msg}")
        return picked

    if sample_index < 0 or sample_index >= n_samples:
        raise IndexError(f"SAMPLE_INDEX out of range: {sample_index} (shard has N={n_samples})")

    return sample_index


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    *,
    y0_phys: np.ndarray,
    g_phys: np.ndarray,
    dt_seconds: np.ndarray,
) -> np.ndarray:
    if g_phys.shape != (len(REQUIRED_GLOBALS),):
        raise ValueError(f"g_phys must have shape ({len(REQUIRED_GLOBALS)},), got {g_phys.shape}")

    y = torch.from_numpy(y0_phys.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=INFER_DTYPE).unsqueeze(0)
    g = torch.from_numpy(g_phys.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=INFER_DTYPE).unsqueeze(0)
    dt_all = torch.from_numpy(dt_seconds.astype(np.float32, copy=False)).to(device=INFER_DEVICE, dtype=INFER_DTYPE)

    preds: List[torch.Tensor] = []
    for k in range(int(dt_all.shape[0])):
        y = model(y, dt_all[k : k + 1], g)
        preds.append(y[0].detach().cpu())
    return torch.stack(preds, dim=0).numpy().astype(np.float64, copy=False)


def _distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    cols: List[Tuple[float, float, float]] = []
    for name in cmaps:
        cmap = plt.get_cmap(name)
        cols.extend([tuple(mcolors.to_rgb(c)) for c in getattr(cmap, "colors", [])])
    if n > len(cols):
        hs = np.linspace(0.0, 1.0, n, endpoint=False)
        cols = [tuple(mcolors.hsv_to_rgb((h, 0.85, 0.95))) for h in hs]
    return cols[:n]


def _plot(*, t_sec: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, species_keys: List[str], out_path: Path) -> None:
    labels = [s.replace("_evolution", "") for s in species_keys]
    order = np.argsort(y_true.max(axis=0))[::-1]
    colors = _distinct_colors(len(order))

    me = max(1, int(PLOT_MARKER_EVERY))
    marker_idx = np.arange(0, len(t_sec), me)

    fig, ax = plt.subplots(figsize=(7, 7))
    for r, j in enumerate(order):
        c = colors[r]
        ax.plot(t_sec, y_true[:, j], linestyle="-", lw=2.5, alpha=1.0, color=c)
        ax.plot(
            t_sec[marker_idx],
            y_pred[marker_idx, j],
            linestyle="None",
            marker="o",
            ms=3.5,
            alpha=1.0,
            color=c,
            markerfacecolor="none",
            markeredgecolor=c,
            markeredgewidth=0.9,
        )

    if PLOT_TIME_LOG_SCALE:
        if float(np.min(t_sec)) <= 0.0:
            raise ValueError("PLOT_TIME_LOG_SCALE=True requires strictly positive evaluation times.")
        ax.set_xscale("log")

    ax.set_yscale("log")
    ax.set(
        xlim=(float(t_sec[0]), float(t_sec[-1])),
        ylim=PLOT_Y_LIM,
        xlabel="Time (s)",
        ylabel="Relative Abundance",
        title=f"Autoregressive rollout ({EVAL_SPLIT}): {len(t_sec)} steps (exported 1-step PHYS model)",
    )
    ax.set_box_aspect(1)

    species_handles = [Line2D([0], [0], color=colors[r], lw=2) for r in range(len(order))]
    leg1 = ax.legend(
        species_handles,
        [labels[int(i)] for i in order],
        loc="lower left",
        title="Species",
        ncol=2,
        fontsize=8,
    )
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.5, alpha=1.0, label="Ground Truth"),
        Line2D(
            [0],
            [0],
            color="k",
            linestyle="None",
            marker="o",
            ms=3.5,
            alpha=1.0,
            markerfacecolor="none",
            markeredgecolor="k",
            markeredgewidth=0.9,
            label=f"Prediction (every {me} step{'s' if me != 1 else ''})",
        ),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_errors(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    eps = 1e-30
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)

    rel_err = np.abs(yp - yt) / (np.abs(yt) + 1e-12)
    mae = np.mean(np.abs(yp - yt))
    log_err = np.mean(np.abs(np.log10(yp) - np.log10(yt)))

    print(f"Physical: rel_err(mean)={rel_err.mean():.3e}, rel_err(max)={rel_err.max():.3e}, MAE={mae:.3e}")
    print(f"Log10:   mean |Δlog10| = {log_err:.3f} orders")


def main() -> None:
    if PLOT_STYLE is not None:
        plt.style.use(PLOT_STYLE)

    run_dir = _resolve_repo_path(MODEL_RUN_DIR)
    export_path = _resolve_export_path(run_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"missing export: {export_path}")

    output_dir = _resolve_output_dir(run_dir)
    print(
        f"[config] export={export_path} run_dir={run_dir} "
        f"device={INFER_DEVICE.type} dtype={INFER_DTYPE} split={EVAL_SPLIT}"
    )

    model, meta = _load_export_module(export_path)
    print(f"[model] file={export_path.name} meta.export_device_tag={meta.get('export_device_tag')}")

    norm_path = _resolve_normalization_path(meta)
    processed_dir = norm_path.parent
    manifest = _load_json(norm_path)

    species_keys = list(manifest.get("species_variables") or [])
    if not species_keys:
        raise KeyError("normalization.json missing species_variables")
    global_keys = list(manifest.get("global_variables") or [])
    if global_keys != list(REQUIRED_GLOBALS):
        raise ValueError(f"normalization.json global_variables must be exactly {list(REQUIRED_GLOBALS)}")

    shard_index, shard_path = _choose_shard(processed_dir, split=EVAL_SPLIT, shard_index=EVAL_SHARD_INDEX)
    y_mat, g_mat, dt_norm_mat = _load_shard(shard_path)
    n_samples = int(y_mat.shape[0])
    sample_index = _choose_sample(n_samples, sample_index=EVAL_SAMPLE_INDEX)
    print(f"[data] split={EVAL_SPLIT} shard={shard_path.name} sample={sample_index} n_samples={n_samples}")

    y_z = y_mat[sample_index]
    g_z = g_mat[sample_index]
    dt_norm = dt_norm_mat[sample_index]

    T = int(y_z.shape[0])
    if ROLLOUT_START_INDEX < 0 or ROLLOUT_START_INDEX >= T - 1:
        raise IndexError(f"ROLLOUT_START_INDEX out of range: {ROLLOUT_START_INDEX} (trajectory has T={T})")

    max_steps = int((T - 1) - ROLLOUT_START_INDEX)
    if ROLLOUT_STEPS <= 0 or ROLLOUT_STEPS > max_steps:
        raise ValueError(f"ROLLOUT_STEPS must be in [1, {max_steps}], got {ROLLOUT_STEPS}")

    y0_z = y_z[ROLLOUT_START_INDEX]
    y_true_z = y_z[ROLLOUT_START_INDEX + 1 : ROLLOUT_START_INDEX + 1 + ROLLOUT_STEPS]
    dt_seq_norm = dt_norm[ROLLOUT_START_INDEX : ROLLOUT_START_INDEX + ROLLOUT_STEPS]

    y0_phys = _denormalize_species(y0_z[None, :], species_keys, manifest)[0]
    y_true_phys = _denormalize_species(y_true_z, species_keys, manifest)
    g_phys = _denormalize_globals(g_z, global_keys, manifest)
    dt_seq_seconds = _dt_norm_to_seconds(dt_seq_norm, manifest)

    y_pred_phys = _rollout(model, y0_phys=y0_phys, g_phys=g_phys, dt_seconds=dt_seq_seconds)
    t_eval = np.cumsum(dt_seq_seconds, dtype=np.float64)

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_path = (
        output_dir
        / (
            f"autoregressive_rollout_{EVAL_SPLIT}_{shard_path.stem}"
            f"_sample-{sample_index}_start-{ROLLOUT_START_INDEX}_steps-{ROLLOUT_STEPS}"
            f"_run-{run_stamp}.png"
        )
    )
    _plot(t_sec=t_eval, y_true=y_true_phys, y_pred=y_pred_phys, species_keys=species_keys, out_path=out_path)
    print(f"Saved: {out_path}")
    _print_errors(y_true_phys, y_pred_phys)


if __name__ == "__main__":
    main()
