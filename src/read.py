#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_repo_root(start: Path) -> Path:
    """Walk up until we find a directory that contains 'models'."""
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").is_dir():
            return p
    return start.resolve()


def _latest_run_dir(repo_root: Path) -> Path:
    """Pick the run directory that contains the newest metrics.csv under models/**."""
    metrics_files = list((repo_root / "models").glob("**/metrics.csv"))
    if not metrics_files:
        raise SystemExit(f"No metrics.csv found under: {repo_root / 'models'}")
    metrics_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return metrics_files[0].parent


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields where possible
            rr: Dict[str, Any] = {}
            for k, v in (r or {}).items():
                if v is None:
                    rr[k] = None
                    continue
                vv = v.strip()
                if vv == "":
                    rr[k] = None
                    continue
                try:
                    rr[k] = float(vv) if k != "epoch" else int(float(vv))
                except Exception:
                    rr[k] = vv
            rows.append(rr)
    return rows


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    d: Any = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _best(rows: List[Dict[str, Any]], key: str, mode: str = "min") -> Optional[Tuple[int, float]]:
    vals: List[Tuple[int, float]] = []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)) and isinstance(r.get("epoch"), int):
            vals.append((r["epoch"], float(r[key])))
    if not vals:
        return None
    if mode == "max":
        return max(vals, key=lambda t: t[1])
    return min(vals, key=lambda t: t[1])


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        # compact, human-friendly float formatting
        return f"{v:.6g}"
    return str(v)


def main() -> int:
    repo_root = _find_repo_root(Path.cwd())

    # Accept either a run dir or a metrics.csv path; otherwise pick newest under models/**
    arg = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    if arg:
        arg = (repo_root / arg).resolve() if not arg.is_absolute() else arg.resolve()
        if arg.is_dir():
            run_dir = arg
        elif arg.is_file() and arg.name == "metrics.csv":
            run_dir = arg.parent
        else:
            raise SystemExit(f"Argument must be a run directory or metrics.csv file. Got: {arg}")
    else:
        run_dir = _latest_run_dir(repo_root)

    metrics_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "config.json"

    if not metrics_path.exists():
        raise SystemExit(f"Missing: {metrics_path}")

    cfg = _read_json(cfg_path)
    rows = _read_metrics_csv(metrics_path)
    if not rows:
        raise SystemExit(f"No rows found in: {metrics_path}")

    # Auto-grab model name/type
    model_type = _get(cfg, "model", "type", default=None) or _get(cfg, "model", "name", default=None)
    if not model_type:
        model_type = run_dir.name  # fallback

    # Pull a few “important” config fields if present
    max_epochs = _get(cfg, "training", "max_epochs", default=None)
    batch_size = _get(cfg, "training", "batch_size", default=None)
    lr = _get(cfg, "training", "lr", default=None)
    wd = _get(cfg, "training", "weight_decay", default=None)
    precision = _get(cfg, "training", "precision", default=None)
    devices = _get(cfg, "training", "devices", default=None)
    accelerator = _get(cfg, "training", "accelerator", default=None)

    # Last row / epoch
    last = rows[-1]
    last_epoch = last.get("epoch", "?")

    # Best metrics (if present)
    best_val_loss = _best(rows, "val_loss", "min")
    best_val_log10 = _best(rows, "val_log10_err", "min")

    # Build a “final metrics” line using common keys if present
    final_keys = ["train_loss", "val_loss", "train_log10_err", "val_log10_err", "train_loss_main", "train_burn_loss"]
    final_present = [(k, last.get(k)) for k in final_keys if k in last and isinstance(last.get(k), (int, float))]

    # Print summary
    print("\n" + "=" * 72)
    print("TRAINING RUN SUMMARY")
    print("=" * 72)
    print(f"Repo:      {repo_root}")
    print(f"Run dir:   {run_dir}")
    print(f"Model:     {model_type}")

    cfg_bits = []
    if precision is not None: cfg_bits.append(f"precision={precision}")
    if devices is not None: cfg_bits.append(f"devices={devices}")
    if accelerator is not None: cfg_bits.append(f"accel={accelerator}")
    if batch_size is not None: cfg_bits.append(f"batch={batch_size}")
    if lr is not None: cfg_bits.append(f"lr={_fmt(lr)}")
    if wd is not None: cfg_bits.append(f"wd={_fmt(wd)}")
    if max_epochs is not None: cfg_bits.append(f"max_epochs={max_epochs}")
    if cfg_bits:
        print("Config:    " + " | ".join(cfg_bits))

    print(f"Logged:    {len(rows)} epoch rows (last epoch={last_epoch})")

    if best_val_loss:
        print(f"Best val:  val_loss={_fmt(best_val_loss[1])} @ epoch {best_val_loss[0]}")
    if best_val_log10:
        print(f"Best val:  val_log10_err={_fmt(best_val_log10[1])} @ epoch {best_val_log10[0]}")

    if final_present:
        print("Final:     " + " | ".join([f"{k}={_fmt(v)}" for k, v in final_present]))

    # Print last N epochs as a compact table
    N = 10
    tail = rows[-N:] if len(rows) > N else rows
    cols = ["epoch"] + [k for k in ["train_loss", "val_loss", "train_log10_err", "val_log10_err"] if k in rows[0] or any(k in r for r in rows)]
    cols = [c for c in cols if any(c in r for r in tail)]

    if cols and len(tail) > 1:
        print("\nRecent epochs:")
        # Header
        print("  " + "  ".join(f"{c:>14}" for c in cols))
        for r in tail:
            line = []
            for c in cols:
                v = r.get(c, None)
                line.append(f"{_fmt(v):>14}" if v is not None else f"{'':>14}")
            print("  " + "  ".join(line))

    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
