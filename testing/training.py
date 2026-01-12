#!/usr/bin/env python3
"""
training.py

Plot training curves from the epoch-level CSV written by trainer.py.

Expected (new) layout:
  MODEL_DIR/metrics.csv

Fallbacks (legacy Lightning layouts) if metrics.csv isn't found directly:
  MODEL_DIR/csv/version_*/metrics.csv
  MODEL_DIR/lightning_logs/version_*/metrics.csv

Behavior:
  - Loads metrics.csv (or merges versioned metrics.csv files)
  - Collapses duplicates per epoch by keeping the last row (resume-safe)
  - Drops epochs with missing train_loss (common for epoch 1 in your log)
  - Reindexes epochs from 0 for plotting
  - Saves MODEL_DIR/plots/training.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("science.mplstyle")
plt.rcParams.update({"mathtext.default": "regular"})  # regular font for exponents


# ---------------- Paths ----------------
REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "big_big_big"
# MODEL_DIR = REPO / "models" / "big_mlp"

PLOT_DIR = MODEL_DIR / "plots"
OUTFILE = PLOT_DIR / "training.png"


def _read_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize / coerce expected numeric columns (silently creates NaNs if missing/bad)
    for col in ("epoch", "step", "lr", "train_loss", "val_loss"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_metrics(model_dir: Path) -> pd.DataFrame:
    # New layout
    direct = model_dir / "metrics.csv"
    if direct.is_file():
        df = _read_one_csv(direct)
        df["_src"] = str(direct)
        return df

    # Legacy layouts: merge versioned CSVs in order
    candidates = []
    candidates += sorted((model_dir / "csv").glob("version_*/metrics.csv"))
    candidates += sorted((model_dir / "lightning_logs").glob("version_*/metrics.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find metrics.csv in:\n"
            f"  - {direct}\n"
            f"  - {model_dir/'csv'/'version_*/metrics.csv'}\n"
            f"  - {model_dir/'lightning_logs'/'version_*/metrics.csv'}"
        )

    dfs = []
    for p in candidates:
        dfi = _read_one_csv(p)
        dfi["_src"] = str(p)
        dfs.append(dfi)

    return pd.concat(dfs, ignore_index=True)


def format_sci(x, _pos):
    if x == 0:
        return "0"
    s = f"{x:.1e}"
    base, exponent = s.split("e")
    return r"${} \times 10^{{{}}}$".format(base, int(exponent))


def main() -> int:
    df = load_metrics(MODEL_DIR)

    # If you have repeated epochs (resumes/rewrites), keep the last record per epoch.
    if "epoch" not in df.columns:
        raise RuntimeError("metrics.csv is missing required column: 'epoch'")

    # Sort so "last" means latest-by-step if step exists, else latest row order
    sort_cols = ["epoch"]
    if "step" in df.columns:
        sort_cols.append("step")
    df = df.sort_values(sort_cols).groupby("epoch", as_index=False).last()

    # Drop rows where train_loss is missing (your epoch 1 has blank train_* fields)
    if "train_loss" not in df.columns:
        raise RuntimeError("metrics.csv is missing required column: 'train_loss'")

    df_plot = df.dropna(subset=["train_loss"]).reset_index(drop=True)

    if df_plot.empty:
        raise RuntimeError("After dropping NaN train_loss rows, nothing is left to plot.")

    # Reindex epochs from 0 for plotting (keeps original epoch in df_plot['epoch'])
    df_plot["epoch_plot"] = range(len(df_plot))

    # Guard for log-scale: remove non-positive losses if they somehow appear
    for c in ("train_loss", "val_loss"):
        if c in df_plot.columns:
            df_plot.loc[df_plot[c] <= 0, c] = float("nan")

    print(df_plot.head(10).to_string(index=False))

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(df_plot["epoch_plot"], df_plot["train_loss"], label="Train")
    if "val_loss" in df_plot.columns and df_plot["val_loss"].notna().any():
        ax.plot(df_plot["epoch_plot"], df_plot["val_loss"], label="Val")

    ax.set(
        xlabel="Epoch (reindexed)",
        ylabel="Loss",
        yscale="log",
        box_aspect=1,
    )
    ax.set_xlim(df_plot["epoch_plot"].min(), df_plot["epoch_plot"].max())
    ax.legend()

    # Secondary x-axis: steps (aligned to the plotted epochs)
    if "step" in df_plot.columns and df_plot["step"].notna().any():
        ax2 = ax.twiny()
        ax2.set_xlim(df_plot["step"].min(), df_plot["step"].max())
        ax2.set_xlabel("Step")
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_sci))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
