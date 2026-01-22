# utils.py
#!/usr/bin/env python3
"""
utils.py

Small utilities used across the codebase.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Union[str, os.PathLike], text: str) -> None:
    """
    Atomic-ish write: write to tmp then replace.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Union[str, os.PathLike], obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, sort_keys=True)
    atomic_write_text(path, text + "\n")


def setup_logging() -> None:
    # Keep it minimal: use print() with timestamps for now
    pass


def load_json_config(path: Union[str, os.PathLike]) -> dict:
    """
    Load a JSON configuration file.

    This codebase expects **strict JSON** (no comments, no trailing commas).
    """
    file_path = Path(path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON at {file_path}: {e}") from e

def dump_json(path: Union[str, os.PathLike], obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)
        f.write("\n")


def human_bytes(num_bytes: float) -> str:
    num = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}EB"


def dict_get_path(d: Dict[str, Any], *keys: str, default: Optional[Any] = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def coerce_path(p: Union[str, os.PathLike], *, root: Optional[Path] = None) -> Path:
    pp = Path(p)
    if not pp.is_absolute() and root is not None:
        pp = (root / pp).resolve()
    return pp


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b if abs(b) > eps else (eps if b >= 0 else -eps))


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_git_revision(repo_root: Optional[Path] = None) -> str:
    """
    Best-effort git hash for reproducibility. Returns "" if not available.
    """
    import subprocess

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def find_latest_checkpoint(run_dir: Union[str, os.PathLike]) -> Optional[Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    cands = sorted(run_dir.glob("*.ckpt"))
    if not cands:
        return None
    # newest by mtime
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=kk))
        else:
            out[kk] = v
    return out


def unflatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        parts = str(k).split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return out


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge, override wins.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def ensure_keys(d: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")


def pretty_dict(d: Dict[str, Any]) -> str:
    return json.dumps(d, indent=2, sort_keys=True)


def timed(msg: str):
    """
    Simple context manager for timing blocks.
    """
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            print(f"{now_ts()} | {msg} | {dt:.3f}s", flush=True)

    return _Timer()
