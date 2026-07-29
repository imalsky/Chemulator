#!/usr/bin/env python3
"""Plot training curves from trainer metrics.jsonl.

Pass --model-dir more than once to overlay several runs in one figure, which
is what a model-selection sweep needs; each run keeps its own metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).with_name("science.mplstyle")
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_model"))
).expanduser().resolve()

METRICS_NAME = "metrics.jsonl"


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


def _curves_for(model_dir: Path) -> Dict[str, np.ndarray]:
    """Read one run's metrics.jsonl into plottable arrays."""
    rows = _collapse_last_per_epoch(_load_metrics_jsonl(model_dir / METRICS_NAME))

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
        raise RuntimeError(f"No valid train loss values found in {model_dir / METRICS_NAME}")

    return {
        "epoch": np.asarray(epochs, dtype=float),
        "train_loss": np.asarray(train_loss, dtype=float),
        "val_loss": np.asarray(val_loss, dtype=float),
        "train_mult": np.asarray(train_mult, dtype=float),
        "val_mult": np.asarray(val_mult, dtype=float),
    }


def _parse_args(argv: Sequence[str] | None) -> Tuple[List[Path], List[str], Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        action="append",
        help="Directory holding metrics.jsonl; repeat to overlay runs "
             "(default: CHEMULATOR_MODEL_DIR, else models/final_model)",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Legend label for the matching --model-dir (default: directory name)",
    )
    parser.add_argument("--out", help="Output PNG path")
    args = parser.parse_args(argv)

    dirs = [Path(d).expanduser().resolve() for d in (args.model_dir or [MODEL_DIR])]
    labels = list(args.label or [])
    if labels and len(labels) != len(dirs):
        raise ValueError("--label must be given once per --model-dir")
    if not labels:
        labels = [d.name for d in dirs]

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        name = "training.png" if len(dirs) == 1 else "training_comparison.png"
        out_path = dirs[0] / "plots" / name
    return dirs, labels, out_path


def main(argv: Sequence[str] | None = None) -> int:
    model_dirs, labels, out_path = _parse_args(argv)

    try:
        plt.style.use(str(STYLE_PATH))
    except OSError:
        warnings.warn("science.mplstyle not found; using matplotlib defaults.")

    runs = [(label, model_dir, _curves_for(model_dir)) for label, model_dir in zip(labels, model_dirs)]
    single = len(runs) == 1

    fig, (ax_loss, ax_mult) = plt.subplots(1, 2, figsize=(12, 5))

    for i, (label, _model_dir, curve) in enumerate(runs):
        train_label = "Train loss" if single else f"{label} train"
        val_label = "Val loss" if single else f"{label} val"
        mult_train_label = "Train mult_err_proxy" if single else f"{label} train"
        mult_val_label = "Val mult_err_proxy" if single else f"{label} val"
        # Two colors per run, so the single-run figure looks as it did before.
        color_train = f"C{(2 * i) % 10}"
        color_val = f"C{(2 * i + 1) % 10}"

        ax_loss.plot(curve["epoch"], curve["train_loss"], color=color_train, label=train_label)
        if np.isfinite(curve["val_loss"]).any():
            ax_loss.plot(curve["epoch"], curve["val_loss"], color=color_val, label=val_label)

        ax_mult.plot(curve["epoch"], curve["train_mult"], color=color_train, label=mult_train_label)
        if np.isfinite(curve["val_mult"]).any():
            ax_mult.plot(curve["epoch"], curve["val_mult"], color=color_val, label=mult_val_label)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="best")
    ax_loss.set_box_aspect(1)

    ax_mult.set_xlabel("Epoch")
    ax_mult.set_ylabel("mult_err_proxy")
    ax_mult.set_yscale("log")
    ax_mult.legend(loc="best")
    ax_mult.set_box_aspect(1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for label, model_dir, curve in runs:
        epochs = curve["epoch"]
        print(
            f"Loaded metrics: {model_dir / METRICS_NAME} (label={label}, "
            f"epochs={epochs.size}, min={int(epochs.min())}, max={int(epochs.max())})"
        )
    print(f"Saved plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
