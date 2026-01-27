#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCI_DIGITS = 3  # digits after decimal in scientific notation (e.g., 2.975e-02)


# ----------------------------
# path resolution (robust to where you run it)
# ----------------------------


def _infer_models_dir() -> Path:
    """
    Locate the models/ directory without searching/globbing.

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
    # Allow passing "v1" (preferred) or "models/v1" (tolerated).
    p = Path(name)
    if len(p.parts) >= 2 and p.parts[0] == "models":
        p = Path(*p.parts[1:])

    candidate = (models_dir / p).resolve()

    # Confine under models_dir (avoid ../../ escapes).
    if candidate != models_dir and models_dir not in candidate.parents:
        raise SystemExit(f"Run dir must be under models/: {name}")

    if not candidate.is_dir():
        raise SystemExit(f"Run dir not found under models/: {p.as_posix()}")

    return candidate


# ----------------------------
# file parsing helpers
# ----------------------------


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
            rr: Dict[str, Any] = {}
            for k, v in (r or {}).items():
                if v is None:
                    rr[k] = None
                    continue
                s = v.strip()
                if not s:
                    rr[k] = None
                    continue
                # epoch is always int; everything else is float-ish
                try:
                    rr[k] = int(float(s)) if k == "epoch" else float(s)
                except Exception:
                    rr[k] = s
            rows.append(rr)
    return rows


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
    lr = t.get("lr")
    wd = t.get("weight_decay")
    add("lr", _fmt(float(lr)) if isinstance(lr, (int, float)) else lr)
    add("wd", _fmt(float(wd)) if isinstance(wd, (int, float)) else wd)
    add("max_epochs", t.get("max_epochs"))
    add("rollout", t.get("rollout_steps"))

    tf = t.get("teacher_forcing", {}) or {}
    if isinstance(tf, dict) and ("start" in tf or "end" in tf):
        tf_start = tf.get("start", None)
        tf_end = tf.get("end", None)
        tf_mode = tf.get("mode", None)
        if tf_start is not None and tf_end is not None:
            parts.append(f"tf={tf_start}->{tf_end}{'' if tf_mode is None else f'({tf_mode})'}")

    cur = t.get("curriculum", {}) or {}
    if isinstance(cur, dict) and bool(cur.get("enabled", False)):
        parts.append(f"cur={cur.get('start_k')}->{cur.get('max_k')}@{cur.get('ramp_epochs')}e")

    longr = t.get("long_rollout", {}) or {}
    if isinstance(longr, dict) and bool(longr.get("enabled", False)):
        parts.append(f"long={longr.get('long_rollout_steps')}@last{longr.get('final_epochs')}e")

    return " | ".join([p for p in parts if p]) if parts else ""


# ----------------------------
# column selection (new trainer schema)
# ----------------------------


def _select_table_cols(header: Sequence[str], max_cols: int = 8) -> List[str]:
    header_set = set(header)

    # New trainer default CSV schema (fixed fields).
    priority = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_loss_log10_mae",
        "val_loss_z_mse",
        "lr",
        "epoch_time_sec",
        "train_tf_prob",
        "train_rollout_steps",
    ]

    cols = [c for c in priority if c in header_set]
    return cols[:max_cols]


def _best_lines(rows: List[Dict[str, Any]], header: Sequence[str]) -> List[str]:
    header_set = set(header)
    candidates = [
        ("best_val_loss", "val_loss", "min"),
        ("best_val_log10_mae", "val_loss_log10_mae", "min"),
        ("best_val_z_mse", "val_loss_z_mse", "min"),
    ]

    out: List[str] = []
    for label, key, mode in candidates:
        if key not in header_set:
            continue
        b = _best(rows, key, mode=mode)
        if b:
            out.append(f"{label}={_fmt(b[1])}@{b[0]}")
        if len(out) >= 3:
            break
    return out


# ----------------------------
# main
# ----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="read.py",
        description="Concise training run summary from ./models/<name>/metrics.csv",
    )
    ap.add_argument(
        "name",
        help="Run directory name under ./models (e.g., 'v1'). You may also pass 'models/v1'.",
    )
    ap.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Print last N epochs as a table. Default: 10 (0 disables).",
    )
    args = ap.parse_args()

    models_dir = _infer_models_dir()
    run_dir = _resolve_run_dir(models_dir, args.name)

    metrics_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "config.json"

    if not metrics_path.exists():
        raise SystemExit(f"Missing: {run_dir.name}/metrics.csv")

    cfg = _read_json(cfg_path)
    rows = _read_metrics_csv(metrics_path)
    if not rows:
        raise SystemExit(f"No rows found in: {run_dir.name}/metrics.csv")

    header = list(rows[0].keys())
    cols = _select_table_cols(header, max_cols=8)

    last = rows[-1]

    print(_pipes([f"run={run_dir.name}"]))

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
