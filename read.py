#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCI_DIGITS = 3  # digits after decimal in scientific notation (e.g., 2.975e-02)


# ----------------------------
# path resolution
# ----------------------------


def _infer_models_dir() -> Path:
    """
    Locate the models/ directory without deep searching.

    Works for either:
      - running from repo root (./models exists)
      - running from inside ./models
      - read.py placed in repo root or in ./models
    """
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    if (cwd / "models").is_dir():
        return (cwd / "models").resolve()
    if cwd.name == "models":
        return cwd

    if (script_dir / "models").is_dir():
        return (script_dir / "models").resolve()
    if script_dir.name == "models":
        return script_dir.resolve()

    raise SystemExit("Could not locate models/ directory (run from repo root or from ./models).")


def _resolve_run_dir(models_dir: Path, name: str) -> Path:
    """
    Accept:
      - absolute/relative path to a run dir (if it exists)
      - "v1" under models/
      - "models/v1" (tolerated)
    """
    raw = Path(name).expanduser()

    # If the user passed a real directory path, honor it (no confinement).
    if raw.is_dir():
        return raw.resolve()

    # Otherwise interpret under models/.
    p = raw
    if len(p.parts) >= 2 and p.parts[0] == "models":
        p = Path(*p.parts[1:])

    candidate = (models_dir / p).resolve()

    # Confine under models_dir (avoid ../../ escapes).
    if candidate != models_dir and models_dir not in candidate.parents:
        raise SystemExit(f"Run dir must be under models/: {name}")

    if not candidate.is_dir():
        raise SystemExit(f"Run dir not found: {candidate.as_posix()}")

    return candidate


def _resolve_metrics_path(run_dir: Path) -> Path:
    """
    Prefer run_dir/metrics.csv.
    If missing, try common Lightning CSVLogger layouts:
      - run_dir/lightning_logs/version_*/metrics.csv
      - run_dir/version_*/metrics.csv
    """
    direct = run_dir / "metrics.csv"
    if direct.exists():
        return direct

    # Common Lightning default layout
    ll = run_dir / "lightning_logs"
    if ll.is_dir():
        candidates = sorted(ll.glob("version_*/metrics.csv"))
        if candidates:
            return candidates[-1].resolve()

    # Sometimes versioned dirs are created directly under run_dir
    candidates2 = sorted(run_dir.glob("version_*/metrics.csv"))
    if candidates2:
        return candidates2[-1].resolve()

    raise SystemExit(f"Missing metrics.csv under: {run_dir.as_posix()}")


def _resolve_config_path(run_dir: Path) -> Optional[Path]:
    """
    Prefer config.resolved.json (this repo writes that).
    Fall back to config.json if present.
    """
    p1 = run_dir / "config.resolved.json"
    if p1.exists():
        return p1
    p2 = run_dir / "config.json"
    if p2.exists():
        return p2
    return None


# ----------------------------
# file parsing helpers
# ----------------------------


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rr: Dict[str, Any] = {}
            for k, v in (r or {}).items():
                if v is None:
                    rr[k] = None
                    continue
                s = v.strip()
                if not s:
                    rr[k] = None
                    continue
                # epoch is always int-ish; everything else is float-ish
                try:
                    rr[k] = int(float(s)) if k == "epoch" else float(s)
                except Exception:
                    rr[k] = s
            rows.append(rr)
    return rows


def _coalesce_by_epoch(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightning CSVLogger can write multiple rows per epoch (train-phase vs val-phase).
    Coalesce rows by epoch by taking the last non-null value per key.
    Output: one row per epoch (in epoch order of first appearance).
    """
    merged: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

    for r in rows:
        ep = r.get("epoch")
        if isinstance(ep, float):
            ep = int(ep)
        if not isinstance(ep, int):
            continue

        if ep not in merged:
            merged[ep] = {"epoch": ep}

        dst = merged[ep]
        for k, v in r.items():
            if k == "epoch":
                continue
            if v is None:
                continue
            dst[k] = v

    return list(merged.values())


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    d: Any = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _best(rows: Iterable[Dict[str, Any]], key: str, mode: str = "min") -> Optional[Tuple[int, float]]:
    vals: List[Tuple[int, float]] = []
    for r in rows:
        ep = r.get("epoch")
        v = r.get(key)
        if isinstance(ep, int) and isinstance(v, (int, float)):
            vals.append((ep, float(v)))
    if not vals:
        return None
    return max(vals, key=lambda t: t[1]) if mode == "max" else min(vals, key=lambda t: t[1])


def _fmt(v: Any, *, digits: int = SCI_DIGITS) -> str:
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}e}"
    return str(v)


def _pipes(items: Sequence[str]) -> str:
    return " | ".join([s for s in items if s])


def _first_existing(header_set: set, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in header_set:
            return k
    return None


# ----------------------------
# summaries (concise)
# ----------------------------


def _model_summary(cfg: Dict[str, Any], run_dir_name: str) -> str:
    m = _get(cfg, "model", default={}) or {}
    mtype = _get(m, "type", default=None) or run_dir_name
    parts: List[str] = [str(mtype)]

    act = _get(m, "activation", default=None)
    drop = _get(m, "dropout", default=None)
    delta = _get(m, "predict_delta", default=None)
    ln = _get(m, "layer_norm", default=None)

    if act is not None:
        parts.append(f"act={act}")
    if drop is not None:
        parts.append(f"drop={drop}")
    if delta is not None:
        parts.append(f"delta={delta}")
    if ln is not None:
        parts.append(f"ln={ln}")

    if str(mtype).lower() == "mlp":
        h = _get(m, "mlp", "hidden_dims", default=None)
        res = _get(m, "mlp", "residual", default=None)
        if h is not None:
            parts.append(f"h={h}")
        if res is not None:
            parts.append(f"res={res}")

    if str(mtype).lower() == "autoencoder":
        ae = _get(m, "autoencoder", default={}) or {}
        z = _get(ae, "latent_dim", default=None)
        if z is not None:
            parts.append(f"z={z}")

    species = _get(cfg, "data", "species_variables", default=None)
    globals_ = _get(cfg, "data", "global_variables", default=None)
    if isinstance(species, list):
        parts.append(f"S={len(species)}")
    if isinstance(globals_, list):
        parts.append(f"G={len(globals_)}")

    return " ".join(parts)


def _train_summary(cfg: Dict[str, Any]) -> str:
    t = _get(cfg, "training", default={}) or {}
    parts: List[str] = []

    def add(k: str, v: Any) -> None:
        if v is None:
            return
        parts.append(f"{k}={v}")

    add("batch", t.get("batch_size"))
    add("max_epochs", t.get("max_epochs"))
    add("rollout", t.get("rollout_steps"))

    opt = t.get("optimizer", {}) or {}
    if isinstance(opt, dict):
        add("opt", opt.get("name"))
        lr = opt.get("lr")
        wd = opt.get("weight_decay")
        add("lr", _fmt(float(lr)) if isinstance(lr, (int, float)) else lr)
        add("wd", _fmt(float(wd)) if isinstance(wd, (int, float)) else wd)
    else:
        lr = t.get("lr")
        wd = t.get("weight_decay")
        add("lr", _fmt(float(lr)) if isinstance(lr, (int, float)) else lr)
        add("wd", _fmt(float(wd)) if isinstance(wd, (int, float)) else wd)

    cur = t.get("curriculum", {}) or {}
    if isinstance(cur, dict) and bool(cur.get("enabled", False)):
        # Support both schemas:
        # - old: base_k/max_k/ramp_epochs
        # - new: start_steps/end_steps/ramp_epochs
        start = cur.get("base_k", cur.get("start_steps", None))
        end = cur.get("max_k", cur.get("end_steps", None))
        ramp = cur.get("ramp_epochs", None)
        if start is not None and end is not None and ramp is not None:
            parts.append(f"cur={start}->{end}@{ramp}e")

    ar = t.get("autoregressive_training", {}) or {}
    if isinstance(ar, dict) and bool(ar.get("enabled", False)):
        # Support both key names seen in this repo’s history.
        ng = ar.get("no_grad_steps", ar.get("skip_steps", None))
        bps = ar.get("backward_per_step", None)
        detach = ar.get("detach_between_steps", None)
        extras = []
        if ng is not None:
            extras.append(f"ng={ng}")
        if bps is not None:
            extras.append(f"bps={bps}")
        if detach is not None:
            extras.append(f"detach={detach}")
        parts.append(f"AR=on({','.join(extras)})" if extras else "AR=on")

    loss = t.get("loss", {}) or {}
    if isinstance(loss, dict):
        lam1 = loss.get("lambda_log10_mae", None)
        lam2 = loss.get("lambda_z_mse", None)
        if lam1 is not None or lam2 is not None:
            parts.append(f"loss=log10_mae*{lam1}+z_mse*{lam2}")

    return " | ".join([p for p in parts if p]) if parts else ""


# ----------------------------
# column selection
# ----------------------------


def _select_table_cols(header: Sequence[str], max_cols: int = 8) -> List[str]:
    header_set = set(header)

    # Prefer metric-first aliases when present (covers both old and new schemas).
    priority = [
        "epoch",
        "train_loss",
        "train_loss_log10_mae",
        "train_log10_mae",          # legacy
        "val_loss",
        "val_loss_log10_mae",
        "val_log10_mae",            # legacy
        "val_loss_z_mse",
        "val_z_mse",                # legacy
        "epoch_time_sec",
        "lr",
        "test_loss",
        "test_loss_log10_mae",
        "test_log10_mae",           # legacy
    ]

    cols = [c for c in priority if c in header_set]

    # Dedup while preserving order.
    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in seen and c in header_set:
            seen.add(c)
            out.append(c)

    # Always include epoch if present.
    if "epoch" in header_set and (not out or out[0] != "epoch"):
        out = ["epoch"] + [c for c in out if c != "epoch"]

    return out[:max_cols]


def _best_lines(rows: List[Dict[str, Any]], header: Sequence[str]) -> List[str]:
    header_set = set(header)

    key_val_loss = _first_existing(header_set, ["val_loss"])
    key_val_log10 = _first_existing(header_set, ["val_loss_log10_mae", "val_log10_mae"])
    key_val_z = _first_existing(header_set, ["val_loss_z_mse", "val_z_mse"])

    candidates: List[Tuple[str, Optional[str], str]] = [
        ("best_val_loss", key_val_loss, "min"),
        ("best_val_log10_mae", key_val_log10, "min"),
        ("best_val_z_mse", key_val_z, "min"),
    ]

    out: List[str] = []
    for label, key, mode in candidates:
        if not key:
            continue
        b = _best(rows, key, mode=mode)
        if b:
            out.append(f"{label}={_fmt(b[1])}@{b[0]}")
        if len(out) >= 3:
            break
    return out


def _union_header(rows: Sequence[Dict[str, Any]]) -> List[str]:
    # Keep epoch first if present, then deterministic order by first appearance.
    seen: OrderedDict[str, None] = OrderedDict()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen[k] = None
    keys = list(seen.keys())
    if "epoch" in keys:
        keys = ["epoch"] + [k for k in keys if k != "epoch"]
    return keys


# ----------------------------
# main
# ----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="read.py",
        description="Concise training run summary from metrics.csv (handles multi-row-per-epoch CSVLogger outputs).",
    )
    ap.add_argument(
        "name",
        help="Run directory (path) OR name under ./models (e.g., 'v4' or 'models/v4').",
    )
    ap.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Print last N epochs as a table. Default: 10 (0 disables).",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Do not coalesce; show raw CSV rows (Lightning CSVLogger may produce multiple rows per epoch).",
    )
    args = ap.parse_args()

    models_dir = _infer_models_dir()
    run_dir = _resolve_run_dir(models_dir, args.name)

    metrics_path = _resolve_metrics_path(run_dir)
    cfg_path = _resolve_config_path(run_dir)

    rows_raw = _read_metrics_csv(metrics_path)
    if not rows_raw:
        raise SystemExit(f"No rows found in: {metrics_path.as_posix()}")

    rows = rows_raw if args.raw else _coalesce_by_epoch(rows_raw)
    if not rows:
        raise SystemExit(f"No usable epoch rows found in: {metrics_path.as_posix()}")

    cfg = _read_json(cfg_path)

    header = _union_header(rows)
    cols = _select_table_cols(header, max_cols=8)

    last = rows[-1]

    print(_pipes([f"run={run_dir.name}", f"metrics={metrics_path.parent.name}/{metrics_path.name}"]))

    if cfg_path is not None:
        print(_pipes([f"config={cfg_path.name}"]))

    model_line = _model_summary(cfg, run_dir.name)
    if model_line:
        print(_pipes([f"model={model_line}"]))

    train_line = _train_summary(cfg)
    if train_line:
        print(_pipes([f"train={train_line}"]))

    last_bits = [f"{c}={_fmt(last.get(c))}" for c in cols if last.get(c) is not None]
    if last_bits:
        print(_pipes(["last"] + last_bits))

    best_bits = _best_lines(rows, header)
    if best_bits:
        print(_pipes(best_bits))

    if args.tail and args.tail > 0:
        tail = rows[-args.tail :] if len(rows) > args.tail else rows

        headers = cols if cols else header
        table = [[_fmt(r.get(c)) for c in headers] for r in tail]
        widths = [
            max(len(headers[i]), max((len(row[i]) for row in table), default=0))
            for i in range(len(headers))
        ]

        def fmt_row(items: List[str]) -> str:
            return " | ".join(items[i].rjust(widths[i]) for i in range(len(items)))

        print("\nrecent:")
        print(fmt_row(list(headers)))
        print("-+-".join("-" * w for w in widths))
        for r in table:
            print(fmt_row(r))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
