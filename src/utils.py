#!/usr/bin/env python3
"""utils.py

Small, dependency-light utilities used across the codebase.

Design principles for this repository:
  - Fail fast with simple error messages.
  - Keep precision control centralized in the config.
  - Prefer explicit, readable code over clever fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Logging / reproducibility
# -----------------------------------------------------------------------------


_LOGGING_CONFIGURED = False


def setup_logging(*, log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger.

    The function is idempotent for the console handler.
    """

    global _LOGGING_CONFIGURED

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _LOGGING_CONFIGURED:
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(fmt)
        root.addHandler(h)
        _LOGGING_CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file).expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing_file_handler = False
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if Path(handler.baseFilename).resolve() == log_file:
                        handler.setLevel(level)
                        handler.setFormatter(fmt)
                        existing_file_handler = True
                        break
                except Exception:
                    continue
        if not existing_file_handler:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC.

    This parser is string-aware: comment markers inside JSON strings are
    preserved as literal text.
    """

    out: list[str] = []
    i = 0
    n = len(text)

    in_str = False
    esc = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        # Not in string/comment.
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def load_json_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load JSON or JSONC config."""

    p = Path(path).expanduser().resolve()
    raw = p.read_text(encoding="utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = strip_jsonc_comments(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse config: {p}") from e


def dump_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> None:
    """Write JSON with deterministic formatting (atomic replace)."""

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(p)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def cfg_get(cfg: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Get a nested config value; raise KeyError with a short message if missing."""

    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(f"Missing config: {'.'.join(path)}")
        cur = cur[key]
    return cur


# -----------------------------------------------------------------------------
# Precision policy (single source of truth)
# -----------------------------------------------------------------------------


_NP_DTYPE_MAP: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "fp32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "fp64": np.dtype("float64"),
}


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "fp64": torch.float64,
    "64": torch.float64,
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved precision settings used by preprocessing, dataset, and training."""

    amp_dtype: torch.dtype
    dataset_dtype: torch.dtype
    normalize_dtype: torch.dtype
    io_dtype: np.dtype
    time_io_dtype: np.dtype
    tf32: bool

    @property
    def use_amp(self) -> bool:
        """True when autocast/GradScaler should be enabled."""
        return self.amp_dtype != torch.float32

    @property
    def amp_mode(self) -> str:
        """Human-readable AMP mode string."""
        if self.amp_dtype == torch.bfloat16:
            return "bf16-mixed"
        if self.amp_dtype == torch.float16:
            return "16-mixed"
        return "off"


def _parse_np_dtype(name: str) -> np.dtype:
    key = str(name).strip().lower()
    if key not in _NP_DTYPE_MAP:
        raise ValueError("Unsupported precision.io_dtype")
    return _NP_DTYPE_MAP[key]


def _parse_torch_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in _TORCH_DTYPE_MAP:
        raise ValueError("Unsupported dtype")
    return _TORCH_DTYPE_MAP[key]


def resolve_precision_policy(cfg: Mapping[str, Any], device: torch.device) -> PrecisionPolicy:
    """Resolve a single precision policy from cfg["precision"].

    Required keys:
      precision.amp
      precision.dataset_dtype
      precision.io_dtype
      precision.normalize_dtype
      precision.time_io_dtype
      precision.tf32

    Notes:
      - "dataset_dtype" controls the in-memory dtype for y/g/dt batches.
      - "normalize_dtype" controls dtype used for normalization math.
      - "io_dtype" and "time_io_dtype" control NPZ on-disk dtypes.
      - We fail fast if the requested precision is incompatible with CPU/MPS.
    """

    p = cfg_get(cfg, ["precision"])
    if not isinstance(p, Mapping):
        raise ValueError("precision must be a mapping")

    amp_raw = str(cfg_get(p, ["amp"]))
    amp_key = amp_raw.strip().lower()

    if amp_key in {"bf16", "bfloat16", "bf16-mixed", "bfloat16-mixed"}:
        amp_dtype = torch.bfloat16
    elif amp_key in {"fp16", "float16", "16", "16-mixed", "fp16-mixed"}:
        amp_dtype = torch.float16
    elif amp_key in {"fp32", "float32", "32", "32-true", "none", "off"}:
        amp_dtype = torch.float32
    else:
        raise ValueError("Unsupported precision.amp")

    dataset_raw = str(cfg_get(p, ["dataset_dtype"]))
    dataset_key = dataset_raw.strip().lower()
    if dataset_key == "auto":
        dataset_dtype = amp_dtype
    else:
        dataset_dtype = _parse_torch_dtype(dataset_key)

    io_dtype = _parse_np_dtype(str(cfg_get(p, ["io_dtype"])))
    time_io_dtype = _parse_np_dtype(str(cfg_get(p, ["time_io_dtype"])))

    normalize_key = str(cfg_get(p, ["normalize_dtype"])).strip().lower()
    normalize_dtype = _parse_torch_dtype(normalize_key)
    if normalize_dtype not in {torch.float32, torch.float64}:
        raise ValueError("precision.normalize_dtype must be float32 or float64")

    tf32 = bool(cfg_get(p, ["tf32"]))

    # Device capability checks (fail fast instead of silently overriding).
    dev_type = device.type
    if dev_type in {"cpu", "mps"}:
        if amp_dtype != torch.float32 or dataset_dtype != torch.float32:
            raise ValueError("On CPU/MPS, use precision.amp=fp32 and precision.dataset_dtype=float32")

    return PrecisionPolicy(
        amp_dtype=amp_dtype,
        dataset_dtype=dataset_dtype,
        normalize_dtype=normalize_dtype,
        io_dtype=io_dtype,
        time_io_dtype=time_io_dtype,
        tf32=tf32,
    )
