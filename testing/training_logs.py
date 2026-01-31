#!/usr/bin/env python3
"""
plot_losses.py - Plot epoch-level training/validation losses from models/<run>/metrics.csv.

Assumes repo layout:
  Auto-Chem/
    src/
    testing/  (this file)
    science.mplstyle  (your style file, ideally at repo root)
    models/v3/metrics.csv

Saves:
  <run_dir>/plots/loss_curves.png
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    run_dir: Path = ROOT / "models" / "v4"
    metrics_name: str = "metrics.csv"

    out_name: str = "loss_curves.png"
    plot_components: bool = True   # also plot log10_mae and z_mse losses
    smoothing: int = 0             # 0 disables; otherwise moving-average window in epochs

    style_candidates: tuple[Path | str, ...] = (
        ROOT / "science.mplstyle",
        ROOT / "testing" / "science.mplstyle",
        "science.mplstyle",
    )


CFG = Config()


# =============================================================================
# Style
# =============================================================================

def apply_style() -> None:
    for s in CFG.style_candidates:
        try:
            if isinstance(s, Path):
                if s.exists():
                    plt.style.use(str(s))
                    return
            else:
                plt.style.use(s)
                return
        except Exception:
            pass


# =============================================================================
# IO
# =============================================================================

def _to_float(x: str):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_metrics_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {path}")

    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"No rows in: {path}")

    # Keep last occurrence per epoch (useful across resumes/restarts)
    by_epoch = {}
    for r in rows:
        e = _to_float(r.get("epoch", ""))
        if e is None:
            continue
        by_epoch[int(e)] = r

    epochs = np.array(sorted(by_epoch.keys()), dtype=np.int64)
    last_rows = [by_epoch[int(e)] for e in epochs]

    def col(name: str):
        return np.array([_to_float(r.get(name, "")) for r in last_rows], dtype=np.float64)

    return {
        "epoch": epochs,
        "train_loss": col("train_loss"),
        "val_loss": col("val_loss"),
        "train_loss_log10_mae": col("train_loss_log10_mae"),
        "val_loss_log10_mae": col("val_loss_log10_mae"),
        "train_loss_z_mse": col("train_loss_z_mse"),
        "val_loss_z_mse": col("val_loss_z_mse"),
    }


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    y2 = y.astype(np.float64, copy=True)
    y2[~np.isfinite(y2)] = np.nan
    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    ypad = np.pad(y2, (pad, w - 1 - pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def _sanitize_for_log(y: np.ndarray) -> np.ndarray:
    y2 = y.astype(np.float64, copy=True)
    y2[~np.isfinite(y2)] = np.nan
    y2[y2 <= 0.0] = np.nan
    return y2


# =============================================================================
# Plotting
# =============================================================================

def plot_losses(m, out_path: Path) -> None:
    epoch = m["epoch"]

    train = _sanitize_for_log(m["train_loss"])
    val = _sanitize_for_log(m["val_loss"])

    if CFG.smoothing and CFG.smoothing > 1:
        train = moving_average(train, CFG.smoothing)
        val = moving_average(val, CFG.smoothing)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(epoch, train, label="train_loss", color='black', linestyle="solid", linewidth=3, alpha=0.5)
    ax.plot(epoch, val, label="val_loss", color='black', linestyle="dashed", linewidth=3)

    if CFG.plot_components:
        tlm = _sanitize_for_log(m["train_loss_log10_mae"])
        vlm = _sanitize_for_log(m["val_loss_log10_mae"])
        tzm = _sanitize_for_log(m["train_loss_z_mse"])
        vzm = _sanitize_for_log(m["val_loss_z_mse"])

        #ax.plot(epoch, tlm, linestyle="solid", label="train_log10_mae", color='red', linewidth=5, alpha=0.8)
        #ax.plot(epoch, vlm, linestyle="dashed", label="val_log10_mae", color='red')

        #ax.plot(epoch, tzm, linestyle="solid", label="train_z_mse", color='blue', linewidth=5, alpha=0.8)
        #ax.plot(epoch, vzm, linestyle="dashed", label="val_z_mse", color='blue')

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves: {CFG.run_dir.name}")
    ax.set_box_aspect(1)  # square plot area

    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    apply_style()
    metrics_path = (CFG.run_dir / CFG.metrics_name).resolve()
    m = load_metrics_csv(metrics_path)
    out_path = CFG.run_dir / "plots" / CFG.out_name
    plot_losses(m, out_path)


if __name__ == "__main__":
    main()
