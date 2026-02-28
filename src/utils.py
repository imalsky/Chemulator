#!/usr/bin/env python3
"""
utils.py - Small shared utilities.

This project intentionally keeps shared code minimal and explicit.

Included helpers:
  - ensure_dir: create directories idempotently
  - load_json_config: load JSON with helpful errors
  - atomic_write_json: write JSON safely (write temp + atomic rename)
  - PrecisionConfig / parse_precision_config: central dtype configuration for training

Note:
  preprocessing.py is intentionally self-contained and does not import this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load a JSON configuration file with good errors."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {file_path} at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e

    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a JSON object at top-level: {file_path}.")
    return obj


def _fsync_dir_best_effort(directory: Path) -> None:
    """Best-effort fsync of a directory entry (helps durability after rename)."""
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(directory), flags)
    except Exception:
        return

    try:
        os.fsync(dir_fd)
    except Exception:
        pass
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass


def atomic_write_json(path: Union[str, os.PathLike], obj: Any, *, indent: int = 2) -> None:
    """Atomically write JSON to disk using write-to-temp-then-rename."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=int(indent), sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        # Atomic on POSIX when src/dst are on same filesystem.
        os.replace(tmp, out_path)
        _fsync_dir_best_effort(out_path.parent)

    finally:
        # If anything failed before os.replace, clean up the temp file.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


# ==============================================================================
# Precision (centralized dtype configuration)
# ==============================================================================

_ALLOWED_DTYPES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
    "single": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
    "double": torch.float64,
}


def parse_torch_dtype(value: Any, *, key: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype.

    Supported (case-insensitive):
      - bfloat16 / bf16
      - float32 / fp32 / float
      - float64 / fp64 / double

    Note:
      float16 is intentionally unsupported in this codebase.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"bad type: {key} (expected non-empty string)")
    name = value.strip().lower()
    if name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"unsupported dtype for {key}: '{value}' (supported: bfloat16, float32, float64)"
        )
    return _ALLOWED_DTYPES[name]


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing: {key}")
    return mapping[key]


def _as_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"bad type: {key}")
    return value.strip()


@dataclass(frozen=True)
class PrecisionConfig:
    """Resolved precision configuration (torch dtypes + Lightning precision string)."""

    compute_dtype: torch.dtype
    amp_mode: str  # "mixed" | "true"
    model_dtype: torch.dtype
    input_dtype: torch.dtype
    dataset_dtype: torch.dtype
    preload_dtype: torch.dtype
    loss_dtype: torch.dtype

    @property
    def lightning_precision(self) -> str:
        """Lightning precision string derived from compute_dtype + amp_mode."""
        if self.compute_dtype == torch.bfloat16:
            return "bf16-mixed" if self.amp_mode == "mixed" else "bf16-true"
        if self.compute_dtype == torch.float32:
            return "32-true"
        if self.compute_dtype == torch.float64:
            return "64-true"
        raise ValueError(f"unsupported compute_dtype: {self.compute_dtype}")

    @property
    def uses_autocast(self) -> bool:
        """True when we expect Lightning AMP/autocast to be active."""
        return self.compute_dtype == torch.bfloat16 and self.amp_mode == "mixed"


def parse_precision_config(cfg: Mapping[str, Any]) -> PrecisionConfig:
    """Parse and validate cfg["precision"] into a PrecisionConfig."""
    pcfg_raw = cfg.get("precision", None)
    if not isinstance(pcfg_raw, Mapping):
        raise KeyError("missing/invalid: precision (expected object)")

    compute_dtype = parse_torch_dtype(_require(pcfg_raw, "compute_dtype"), key="precision.compute_dtype")
    amp_mode = _as_str(_require(pcfg_raw, "amp_mode"), "precision.amp_mode").lower()
    if amp_mode not in ("mixed", "true"):
        raise ValueError("precision.amp_mode must be 'mixed' or 'true'")

    model_dtype = parse_torch_dtype(_require(pcfg_raw, "model_dtype"), key="precision.model_dtype")
    input_dtype = parse_torch_dtype(_require(pcfg_raw, "input_dtype"), key="precision.input_dtype")
    dataset_dtype = parse_torch_dtype(_require(pcfg_raw, "dataset_dtype"), key="precision.dataset_dtype")
    preload_dtype = parse_torch_dtype(_require(pcfg_raw, "preload_dtype"), key="precision.preload_dtype")
    loss_dtype = parse_torch_dtype(_require(pcfg_raw, "loss_dtype"), key="precision.loss_dtype")

    # Cross-validation: avoid silent dtype mismatches.
    if compute_dtype in (torch.float32, torch.float64):
        if amp_mode != "true":
            raise ValueError("precision.amp_mode must be 'true' when compute_dtype is float32/float64")
        if model_dtype != compute_dtype:
            raise ValueError("precision.model_dtype must equal precision.compute_dtype for float32/float64 compute")
        if input_dtype != compute_dtype:
            raise ValueError("precision.input_dtype must equal precision.compute_dtype for float32/float64 compute")

    if compute_dtype == torch.bfloat16 and amp_mode == "true":
        # Pure bf16: model + inputs must be bf16 to avoid dtype mismatches.
        if model_dtype != torch.bfloat16:
            raise ValueError("precision.model_dtype must be bfloat16 when compute_dtype=bfloat16 and amp_mode=true")
        if input_dtype != torch.bfloat16:
            raise ValueError("precision.input_dtype must be bfloat16 when compute_dtype=bfloat16 and amp_mode=true")

    # loss_dtype is allowed to differ (common: float32 loss under bf16 AMP).
    return PrecisionConfig(
        compute_dtype=compute_dtype,
        amp_mode=amp_mode,
        model_dtype=model_dtype,
        input_dtype=input_dtype,
        dataset_dtype=dataset_dtype,
        preload_dtype=preload_dtype,
        loss_dtype=loss_dtype,
    )
