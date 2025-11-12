#!/usr/bin/env python3
"""
Utility Functions Module
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Optional, Union, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Module-level state for logging configuration
_logging_configured = False


def setup_logging(
        log_file: Optional[Union[str, os.PathLike]] = None,
        level: int = logging.INFO
) -> None:
    """
    Configure root logger with console and optional file output.
    """
    global _logging_configured
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Define consistent formatter for all handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure console handler once
    if not _logging_configured:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _logging_configured = True

    # Add file handler if requested and not already present
    if log_file is not None:
        # Check if file handler already exists for this path
        file_path = Path(log_file).resolve()
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and
            Path(h.baseFilename).resolve() == file_path
            for h in root_logger.handlers
        )

        if not has_file_handler:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set Python hash seed for consistent dictionary ordering
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    # Seed PyTorch if available
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def strip_jsonc_comments(text: str) -> str:
    """
    Remove comments from JSONC (JSON with Comments).
    """
    # Remove block comments first
    block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    text = block_comment_pattern.sub("", text)

    # Process line-by-line to handle // comments while preserving strings
    output_lines = []

    for line in text.splitlines():
        cleaned_chars = []
        in_string = False
        escaped = False
        i = 0

        while i < len(line):
            ch = line[i]

            # Handle string content
            if in_string:
                cleaned_chars.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            # Start of string
            if ch == '"':
                in_string = True
                cleaned_chars.append(ch)
                i += 1
                continue

            # Check for line comment outside strings
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break  # Ignore rest of line

            cleaned_chars.append(ch)
            i += 1

        output_lines.append("".join(cleaned_chars))

    return "\n".join(output_lines)


def load_json_config(path: Union[str, os.PathLike]) -> dict:
    """
    Load JSON or JSONC configuration file.
    """
    file_path = Path(path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Try standard JSON first
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try json5 library for proper JSONC support
    try:
        import json5
        return json5.loads(raw_text)
    except (ImportError, Exception):
        pass

    # Fallback to manual comment stripping
    cleaned_text = strip_jsonc_comments(raw_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON/JSONC at {file_path}: {e}") from e


def dump_json(path: Union[str, os.PathLike], obj: Any, indent: int = 2) -> None:
    """
    Write JSON with consistent formatting.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize with consistent formatting
    json_text = json.dumps(
        obj,
        indent=indent,
        sort_keys=True,
        ensure_ascii=False
    )

    # Use atomic write pattern
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_text)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        temp_path.replace(file_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _map_storage_dtype(name: Optional[str]) -> Optional['torch.dtype']:
    """
    Map dtype string to PyTorch dtype.
    """
    if torch is None or not name:
        return None

    dtype_map = {
        # bfloat16 variants
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        # float16 variants
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "16": torch.float16,
        # float32 variants
        "fp32": torch.float32,
        "float32": torch.float32,
        "32": torch.float32,
    }

    return dtype_map.get(str(name).strip().lower())


def resolve_precision_and_dtype(
        cfg: dict,
        device: 'torch.device',
        logger: Optional[logging.Logger] = None
) -> Tuple[Union[str, int], 'torch.dtype']:
    """
    Resolve Lightning precision and runtime dtype from configuration.
    """
    mp_cfg = cfg.get("mixed_precision", {}) or {}
    mode = (
        mp_cfg.get("mode")
        or mp_cfg.get("precision")
        or cfg.get("precision")
        or "bf16"  # default
    )
    mode = str(mode).lower().strip()

    # Lightning precision field and corresponding "AMP dtype" used for runtime dtype
    if mode in {"bf16", "bfloat16", "bf16-mixed", "bfloat16-mixed"}:
        pl_precision = "bf16-mixed"
        amp_dtype = torch.bfloat16
    elif mode in {"fp16", "float16", "16-mixed", "fp16-mixed"}:
        pl_precision = "16-mixed"
        amp_dtype = torch.float16
    elif mode in {"fp32", "float32", "32"}:
        pl_precision = 32
        amp_dtype = torch.float32
    else:
        # Unknown => safe default
        pl_precision = "bf16-mixed"
        amp_dtype = torch.bfloat16

    dataset_dtype_str = cfg.get("dataset", {}).get("storage_dtype")
    runtime_dtype = _map_storage_dtype(dataset_dtype_str) or amp_dtype

    dev_type = getattr(device, "type", str(device))

    # Force FP32 on CPU
    if dev_type == "cpu":
        runtime_dtype = torch.float32
        pl_precision = 32
        if logger:
            logger.info("CPU detected — forcing FP32 (Lightning=32, Dataset=float32).")

    # Force FP32 on Apple MPS (avoid BF16/FP16 matmul asserts)
    if dev_type == "mps":
        runtime_dtype = torch.float32
        pl_precision = 32
        if logger:
            logger.info("MPS detected — forcing FP32 (Lightning=32, Dataset=float32).")

    if logger:
        logger.info(f"Precision config: Lightning={pl_precision}, Dataset={runtime_dtype}")

    return pl_precision, runtime_dtype
