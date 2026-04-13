#!/usr/bin/env python3
"""Prediction vs ground-truth overlay using physical-I/O model."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import prepare_platform_environment

prepare_platform_environment()

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).with_name("science.mplstyle")
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_version"))
).expanduser().resolve()


try:
    plt.style.use(str(STYLE_PATH))
except OSError:
    warnings.warn("science.mplstyle not found; using matplotlib defaults.")

MODEL_PATH = MODEL_DIR / "physical_model_k1_cpu.pt2"
METADATA_PATH = MODEL_DIR / "physical_model_metadata.json"
CONFIG_PATH = MODEL_DIR / "config.json"

SAMPLE_IDX = 5
Q_COUNT = 100
XMIN, XMAX = 1e-3, 1e8
PLOT_SPECIES: List[str] = []

_TAB20_STRONG = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cfg_path(path_like: str | os.PathLike[str], *, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _species_colors(n: int) -> np.ndarray:
    if n <= len(_TAB20_STRONG):
        import matplotlib.colors as mcolors

        return np.array([mcolors.to_rgba(c) for c in _TAB20_STRONG[:n]])
    return plt.cm.tab20(np.linspace(0, 1, n))


def _load_one_sample(data_dir: Path, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = sorted((data_dir / "test").glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found in {data_dir / 'test'}")

    with np.load(shards[0], allow_pickle=False) as d:
        y_mat = np.asarray(d["y_mat"], dtype=np.float32)
        g_mat = np.asarray(d["globals"], dtype=np.float32)
        t_vec = np.asarray(d["t_vec"], dtype=np.float32)

    y = y_mat[sample_idx]
    g = g_mat[sample_idx]
    t = t_vec[sample_idx] if t_vec.ndim == 2 else t_vec
    return y, g, np.asarray(t, dtype=np.float32)


def _prepare_inputs(
    y0: np.ndarray,
    g: np.ndarray,
    t_phys: np.ndarray,
    q_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    M = len(t_phys)
    if M < 2:
        raise ValueError(f"Need at least 2 time points, got {M}")
    qn = max(1, min(q_count, M - 1))
    q_idx = np.linspace(1, M - 1, qn).round().astype(int)

    t_sel = t_phys[q_idx]
    dt_sec = (t_sel - t_phys[0]).astype(np.float32)
    if np.any(dt_sec <= 0):
        raise RuntimeError("Non-positive dt encountered in test sample")

    y_batch = torch.from_numpy(y0[None, :]).float().repeat(qn, 1)
    g_batch = torch.from_numpy(g[None, :]).float().repeat(qn, 1)
    dt_batch = torch.from_numpy(dt_sec).float().view(-1, 1)
    return y_batch, dt_batch, g_batch, q_idx, t_sel


def _plot(
    *,
    t_phys: np.ndarray,
    y_true_full: np.ndarray,
    t_pred: np.ndarray,
    y_pred: np.ndarray,
    species_names: List[str],
    plot_species: List[str],
    out_path: Path,
) -> None:
    base_names = [n.removesuffix("_evolution") for n in species_names]
    keep = [i for i, b in enumerate(base_names) if (not plot_species) or (b in plot_species)]
    labels = [base_names[i] for i in keep]

    y_true = y_true_full[:, keep]
    y_pred = y_pred[:, keep]

    tiny = 1e-35
    m_gt = (t_phys >= XMIN) & (t_phys <= XMAX)
    m_pr = (t_pred >= XMIN) & (t_pred <= XMAX)
    t_gt = t_phys[m_gt]
    t_pr = t_pred[m_pr]
    y_gt = np.clip(y_true[m_gt], tiny, None)
    y_pr = np.clip(y_pred[m_pr], tiny, None)

    order = np.argsort(np.max(y_gt, axis=0))[::-1] if y_gt.size else np.arange(len(labels))
    colors = _species_colors(len(order))

    fig, ax = plt.subplots(figsize=(7, 7))
    for rank, idx in enumerate(order):
        col = colors[rank]
        ax.loglog(t_gt, y_gt[:, idx], "-", lw=3, alpha=0.35, color=col)
        if t_gt.size:
            ax.loglog([t_gt[0]], [y_gt[0, idx]], "o", mfc="none", color=col, ms=5)
        ax.loglog(t_pr, y_pr[:, idx], "--", lw=2.5, alpha=1.0, color=col)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(1e-28, 3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Abundance")
    ax.set_box_aspect(1)

    species_handles = [Line2D([0], [0], color=colors[r], lw=2.0) for r in range(len(order))]
    species_labels = [labels[i] for i in order]
    leg_species = ax.legend(species_handles, species_labels, loc="best", title="Species", ncol=3)
    ax.add_artist(leg_species)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, ls="-", label="Ground Truth"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="Prediction"),
    ]
    ax.legend(handles=style_handles, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg = _load_json(CONFIG_PATH)
    meta = _load_json(METADATA_PATH)
    ep = torch.export.load(MODEL_PATH)
    model = ep.module()

    species_order = [str(x) for x in meta["species_order"]]
    globals_order = [str(x) for x in meta["globals_order"]]

    processed_dir = _resolve_cfg_path(cfg["paths"]["processed_data_dir"], base_dir=REPO)
    cfg_species = [str(x) for x in cfg["data"]["species_variables"]]
    cfg_globals = [str(x) for x in cfg["data"]["global_variables"]]

    if species_order != cfg_species:
        raise ValueError("metadata species_order must match cfg.data.species_variables")
    if globals_order != cfg_globals:
        raise ValueError("metadata globals_order must match cfg.data.global_variables")

    y_all, g_all, t_phys = _load_one_sample(processed_dir, SAMPLE_IDX)
    y = y_all
    g = g_all if globals_order else np.empty((0,), dtype=np.float32)

    y_batch, dt_batch, g_batch, q_idx, t_sel = _prepare_inputs(y[0], g, t_phys, Q_COUNT)
    y_pred = model(y_batch, dt_batch, g_batch)[:, 0, :].detach().cpu().numpy()
    y_true_sel = y[q_idx, :]

    out_png = MODEL_DIR / "plots" / f"pred_{SAMPLE_IDX}.png"
    _plot(
        t_phys=t_phys,
        y_true_full=y,
        t_pred=t_sel,
        y_pred=y_pred,
        species_names=species_order,
        plot_species=PLOT_SPECIES,
        out_path=out_png,
    )

    rel_err = np.abs(y_pred - y_true_sel) / (np.abs(y_true_sel) + 1e-12)
    print(f"Model path: {MODEL_PATH}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Relative error: mean={rel_err.mean():.3e}, max={rel_err.max():.3e}")
    print(f"Plot saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
