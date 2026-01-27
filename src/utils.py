#!/usr/bin/env python3
"""
utils.py - Small shared utilities.

This project only needs a few filesystem/config helpers:
  - ensure_dir: create directories idempotently
  - load_json_config: load JSON with helpful errors
  - atomic_write_json: write JSON safely (write temp + atomic rename)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """
    Create a directory (and parents) if it doesn't already exist.

    Args:
        path: Directory path.

    Returns:
        Path to the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_config(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON (includes line/column).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {file_path}. Ensure the path is correct."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {file_path} at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e


def _fsync_dir_best_effort(directory: Path) -> None:
    """
    Best-effort fsync of a directory entry (helps durability after rename).
    No-op on platforms/filesystems where this is unsupported.
    """
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


def atomic_write_json(
    path: Union[str, os.PathLike], obj: Any, *, indent: int = 2
) -> None:
    """
    Atomically write JSON to disk using write-to-temp-then-rename.

    This prevents partial/corrupted files if the process is interrupted.

    Args:
        path: Destination file path.
        obj: JSON-serializable Python object.
        indent: Indentation level for pretty printing.

    Raises:
        TypeError: If `obj` is not JSON-serializable.
        OSError: For underlying filesystem errors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        # Atomic on POSIX when src/dst are on same filesystem.
        os.replace(tmp, path)
        _fsync_dir_best_effort(path.parent)

    finally:
        # If anything failed before os.replace, clean up the temp file.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
