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

    # Optimizer schema (new): training.optimizer.{name,lr,weight_decay,...}
    opt = t.get("optimizer", {}) or {}
    if isinstance(opt, dict):
        add("opt", opt.get("name"))
        lr = opt.get("lr")
        wd = opt.get("weight_decay")
        add("lr", _fmt(float(lr)) if isinstance(lr, (int, float)) else lr)
        add("wd", _fmt(float(wd)) if isinstance(wd, (int, float)) else wd)
    else:
        # Legacy fallback: training.lr / training.weight_decay
        lr = t.get("lr")
        wd = t.get("weight_decay")
        add("lr", _fmt(float(lr)) if isinstance(lr, (int, float)) else lr)
        add("wd", _fmt(float(wd)) if isinstance(wd, (int, float)) else wd)

    # Teacher forcing schema (new): {mode, p0, p1, ramp_epochs}
    tf = t.get("teacher_forcing", {}) or {}
    if isinstance(tf, dict):
        if "p0" in tf or "p1" in tf:
            p0 = tf.get("p0", None)
            p1 = tf.get("p1", None)
            mode = tf.get("mode", None)
            ramp = tf.get("ramp_epochs", None)
            if p0 is not None and p1 is not None:
                extra = []
                if mode is not None:
                    extra.append(str(mode))
                if ramp is not None:
                    extra.append(f"{ramp}e")
                parts.append(f"tf={p0}->{p1}{'' if not extra else '(' + ','.join(extra) + ')'}")
        # Legacy fallback: {start,end,mode}
        elif "start" in tf or "end" in tf:
            tf_start = tf.get("start", None)
            tf_end = tf.get("end", None)
            tf_mode = tf.get("mode", None)
            if tf_start is not None and tf_end is not None:
                parts.append(f"tf={tf_start}->{tf_end}{'' if tf_mode is None else f'({tf_mode})'}")

    # Curriculum schema (new): {enabled, mode, base_k, max_k, ramp_epochs}
    cur = t.get("curriculum", {}) or {}
    if isinstance(cur, dict) and bool(cur.get("enabled", False)):
        base_k = cur.get("base_k", cur.get("start_k", None))
        max_k = cur.get("max_k", None)
        ramp = cur.get("ramp_epochs", None)
        if base_k is not None and max_k is not None and ramp is not None:
            parts.append(f"cur={base_k}->{max_k}@{ramp}e")

    # Long rollout schema: {enabled, long_rollout_steps, final_epochs, ...}
    longr = t.get("long_rollout", {}) or {}
    if isinstance(longr, dict) and bool(longr.get("enabled", False)):
        steps = longr.get("long_rollout_steps", None)
        fin = longr.get("final_epochs", None)
        if steps is not None and fin is not None:
            parts.append(f"long={steps}@last{fin}e")

    # Loss weights (useful when comparing across runs)
    loss = t.get("loss", {}) or {}
    if isinstance(loss, dict):
        lam1 = loss.get("lambda_log10_mae", None)
        lam2 = loss.get("lambda_z_mse", None)
        if lam1 is not None or lam2 is not None:
            parts.append(f"loss=log10_mae*{lam1}+z_mse*{lam2}")

    # Autoregressive training flag
    ar = t.get("autoregressive_training", {}) or {}
    if isinstance(ar, dict) and bool(ar.get("enabled", False)):
        ng = ar.get("no_grad_steps", None)
        bps = ar.get("backward_per_step", None)
        parts.append(f"AR=on(ng={ng},bps={bps})")

    return " | ".join([p for p in parts if p]) if parts else ""


# ----------------------------
# column selection (new trainer schema)
# ----------------------------


def _select_table_cols(header: Sequence[str], max_cols: int = 8) -> List[str]:
    header_set = set(header)

    # Prefer metric-first aliases when present.
    priority = [
        "epoch",
        "train_loss",
        "train_log10_mae",
        "val_loss",
        "val_log10_mae",
        "val_loss_log10_mae",
        "val_loss_z_mse",
        "epoch_time_sec",
        "lr",
        "test_loss",
        "test_log10_mae",
    ]

    cols = [c for c in priority if c in header_set]

    # If alias keys aren't present, fall back to component keys.
    if "train_log10_mae" not in header_set and "train_loss_log10_mae" in header_set and "train_loss_log10_mae" not in cols:
        cols.insert(2, "train_loss_log10_mae")
    if "val_log10_mae" not in header_set and "val_loss_log10_mae" in header_set and "val_loss_log10_mae" not in cols:
        # keep val_loss_log10_mae near val_loss
        try:
            i = cols.index("val_loss") + 1
        except ValueError:
            i = len(cols)
        cols.insert(i, "val_loss_log10_mae")

    # Dedup while preserving order.
    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in seen and c in header_set:
            seen.add(c)
            out.append(c)

    return out[:max_cols]


def _best_lines(rows: List[Dict[str, Any]], header: Sequence[str]) -> List[str]:
    header_set = set(header)

    key_val_loss = _first_existing(header_set, ["val_loss"])
    key_val_log10 = _first_existing(header_set, ["val_log10_mae", "val_loss_log10_mae"])
    key_val_z = _first_existing(header_set, ["val_z_mse", "val_loss_z_mse"])

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
