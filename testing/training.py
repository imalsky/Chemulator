#!/usr/bin/env python3
"""Plot training curves from trainer metrics.jsonl."""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).with_name("science.mplstyle")
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_version"))
).expanduser().resolve()

METRICS_PATH = MODEL_DIR / "metrics.jsonl"
OUTFILE = MODEL_DIR / "plots" / "training.png"


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _load_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    records: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path} line {lineno}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"metrics.jsonl line {lineno} must be a JSON object")
        if "epoch" not in obj:
            raise KeyError(f"metrics.jsonl line {lineno} missing key: epoch")
        records.append(obj)

    if not records:
        raise RuntimeError(f"No metrics records found in {path}")
    return records


def _collapse_last_per_epoch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    collapsed: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        epoch = int(rec["epoch"])
        collapsed[epoch] = rec
    return [collapsed[e] for e in sorted(collapsed.keys())]


def main() -> int:
    try:
        plt.style.use(str(STYLE_PATH))
    except OSError:
        warnings.warn("science.mplstyle not found; using matplotlib defaults.")

    rows = _collapse_last_per_epoch(_load_metrics_jsonl(METRICS_PATH))

    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    train_mult: List[float] = []
    val_mult: List[float] = []

    for rec in rows:
        ep = int(rec["epoch"]) + 1
        tr = rec.get("train")
        if not isinstance(tr, dict):
            continue

        tr_loss = _safe_float(tr.get("loss"))
        if not math.isfinite(tr_loss) or tr_loss <= 0.0:
            continue

        va = rec.get("val")
        va_loss = float("nan")
        va_mult_val = float("nan")
        if isinstance(va, dict):
            va_loss = _safe_float(va.get("loss"))
            if not (math.isfinite(va_loss) and va_loss > 0.0):
                va_loss = float("nan")
            va_mult_val = _safe_float(va.get("mult_err_proxy"))

        epochs.append(ep)
        train_loss.append(tr_loss)
        val_loss.append(va_loss)
        train_mult.append(_safe_float(tr.get("mult_err_proxy")))
        val_mult.append(va_mult_val)

    if not epochs:
        raise RuntimeError("No valid train loss values found in metrics.jsonl")

    x = np.asarray(epochs, dtype=float)
    y_train = np.asarray(train_loss, dtype=float)
    y_val = np.asarray(val_loss, dtype=float)
    y_train_mult = np.asarray(train_mult, dtype=float)
    y_val_mult = np.asarray(val_mult, dtype=float)

    fig, (ax_loss, ax_mult) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(x, y_train, label="Train loss")
    if np.isfinite(y_val).any():
        ax_loss.plot(x, y_val, label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="best")
    ax_loss.set_box_aspect(1)

    ax_mult.plot(x, y_train_mult, label="Train mult_err_proxy")
    if np.isfinite(y_val_mult).any():
        ax_mult.plot(x, y_val_mult, label="Val mult_err_proxy")
    ax_mult.set_xlabel("Epoch")
    ax_mult.set_ylabel("mult_err_proxy")
    ax_mult.set_yscale("log")
    ax_mult.legend(loc="best")
    ax_mult.set_box_aspect(1)

    fig.tight_layout()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded metrics: {METRICS_PATH}")
    print(f"Epochs plotted: {len(epochs)} (min={min(epochs)}, max={max(epochs)})")
    print(f"Saved plot: {OUTFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
