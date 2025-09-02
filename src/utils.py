#!/usr/bin/env python3
"""
utils.py — small, focused helpers with clear separation of concerns.

Provided APIs (used by the codebase):
- setup_logging(log_file: Optional[PathLike]) -> None
- seed_everything(seed: int) -> None
- load_json_config(path: PathLike) -> dict        # accepts .json or .jsonc (with comments)
- load_json(path: PathLike) -> dict               # strict JSON (also tolerates jsonc safely)
- dump_json(path: PathLike, obj: dict) -> None    # atomic write with UTF-8 encoding

No side effects at import time.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


PathLike = Union[str, os.PathLike]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(log_file: Optional[PathLike] = None, level: int = logging.INFO) -> None:
    """
    Configure root logging with a concise console formatter and optional file sink.
    Idempotent: avoids adding duplicate handlers when called multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers only if we haven't configured yet
    if not getattr(root, "_utils_logging_configured", False):
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"))
        root.addHandler(ch)

        # File sink (optional)
        if log_file is not None:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"))
            root.addHandler(fh)

        setattr(root, "_utils_logging_configured", True)
    else:
        # If already configured, optionally add/replace file handler
        if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"))
            root.addHandler(fh)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and (if available) PyTorch RNGs for reproducibility.
    Does not toggle framework determinism flags (those are handled elsewhere).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        except Exception:
            # Keep going; training can proceed without failing on some environments
            pass


# -----------------------------------------------------------------------------
# JSON / JSONC
# -----------------------------------------------------------------------------

_JSONC_LINE = re.compile(r"(?P<code>[^\"/\n]*)(?P<comment>//[^\n]*)")
_JSONC_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_jsonc(text: str) -> str:
    """
    Remove // line comments and /* block */ comments from JSONC while preserving
    string literals. Strategy:
      1) Remove block comments first.
      2) For each line, strip // comments that appear outside of strings.

    This is a light-weight parser sufficient for typical config files.
    """
    # 1) Strip block comments (/* ... */)
    text = _JSONC_BLOCK.sub("", text)

    # 2) Strip // comments outside strings
    out_lines = []
    for line in text.splitlines():
        buf = []
        in_str = False
        esc = False
        i = 0
        while i < len(line):
            ch = line[i]
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue

            # not inside a string
            if ch == '"':
                in_str = True
                buf.append(ch)
                i += 1
                continue

            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                # start of // comment outside string → discard rest of line
                break

            buf.append(ch)
            i += 1

        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def load_json_config(path: PathLike) -> dict:
    """
    Load a JSON or JSONC file from disk. If the contents contain comments, they
    are stripped prior to parsing.

    Raises on parsing errors with a helpful message.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        # Try strict JSON first (fast path)
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: strip comments and parse again
        cleaned = _strip_jsonc(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON/JSONC config at {p}: {e}") from e


def load_json(path: PathLike) -> dict:
    """
    Load a JSON file. Also tolerates JSONC by stripping comments if strict parse fails.
    Used for reading normalization manifests.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _strip_jsonc(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON at {p}: {e}") from e


# -----------------------------------------------------------------------------
# Filesystem
# -----------------------------------------------------------------------------

def _atomic_write_text(path: PathLike, data: str, encoding: str = "utf-8") -> None:
    """
    Write text atomically: write to a temp file in the same directory and rename.
    Ensures readers never see a partially written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with io.open(tmp, "w", encoding=encoding, newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def dump_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    """
    Serialize `obj` as JSON to `path` atomically (UTF-8). Minimal, stable formatting.
    """
    text = json.dumps(obj, indent=indent, sort_keys=True)
    _atomic_write_text(path, text, encoding="utf-8")
