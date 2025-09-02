#!/usr/bin/env python3
"""
Utility Functions Module
========================
Provides common utility functions for logging, configuration, and file I/O.

This module contains standalone helper functions used throughout the codebase
with no side effects at import time.
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
    import torch
except ImportError:
    torch = None


# Type alias for path-like objects
PathLike = Union[str, os.PathLike]


def setup_logging(
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO
) -> None:
    """
    Configure root logger with console and optional file output.
    
    Sets up a consistent logging format across the application.
    Idempotent: avoids adding duplicate handlers on multiple calls.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Check if already configured to avoid duplicate handlers
    if not getattr(root_logger, "_utils_logging_configured", False):
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Set formatting
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_file is not None:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Mark as configured
        setattr(root_logger, "_utils_logging_configured", True)
    
    else:
        # Already configured - only add file handler if new and requested
        if log_file is not None:
            has_file_handler = any(
                isinstance(h, logging.FileHandler)
                for h in root_logger.handlers
            )
            
            if not has_file_handler:
                file_path = Path(log_file)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(file_path, encoding="utf-8")
                file_handler.setLevel(level)
                
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Seeds the random number generators for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    
    Note: This sets seeds but does not enable deterministic algorithms.
    For full determinism, additional framework-specific settings are needed.
    
    Args:
        seed: Random seed value
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Python hash seed for reproducible hashing
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    
    # PyTorch (if available)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            
            # CUDA seeds
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception:
            # Continue even if PyTorch seeding fails
            pass


def strip_jsonc_comments(text: str) -> str:
    """
    Remove comments from JSONC (JSON with Comments) text.
    
    Handles both line comments (//) and block comments (/* */).
    Preserves string literals that contain comment-like patterns.
    
    Args:
        text: JSONC text with potential comments
        
    Returns:
        JSON text with comments removed
    """
    # First, remove block comments /* ... */
    block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    text = block_comment_pattern.sub("", text)
    
    # Process line by line to handle // comments
    output_lines = []
    
    for line in text.splitlines():
        # Track whether we're inside a string literal
        cleaned_chars = []
        in_string = False
        escaped = False
        char_index = 0
        
        while char_index < len(line):
            char = line[char_index]
            
            if in_string:
                cleaned_chars.append(char)
                
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                
                char_index += 1
                continue
            
            # Not in string - check for comment start
            if char == '"':
                in_string = True
                cleaned_chars.append(char)
                char_index += 1
                continue
            
            # Check for line comment
            if char == "/" and char_index + 1 < len(line) and line[char_index + 1] == "/":
                # Found comment - ignore rest of line
                break
            
            cleaned_chars.append(char)
            char_index += 1
        
        output_lines.append("".join(cleaned_chars))
    
    return "\n".join(output_lines)


def load_json_config(path: PathLike) -> dict:
    """
    Load JSON or JSONC configuration file.
    
    Automatically handles both standard JSON and JSONC (with comments).
    First attempts to parse as strict JSON for performance, then falls
    back to comment stripping if needed.
    
    Args:
        path: Path to JSON/JSONC file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        ValueError: If file cannot be parsed as valid JSON
    """
    file_path = Path(path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Try parsing as standard JSON first (fast path)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    
    # Fall back to JSONC parsing (strip comments)
    cleaned_text = strip_jsonc_comments(raw_text)
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON/JSONC file at {file_path}: {e}"
        ) from e


def load_json(path: PathLike) -> dict:
    """
    Load JSON file with automatic JSONC support.
    
    This is an alias for load_json_config, provided for consistency
    with the existing API.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        ValueError: If file cannot be parsed
    """
    return load_json_config(path)


def dump_json(
    path: PathLike,
    obj: Any,
    indent: int = 2
) -> None:
    """
    Write object to JSON file atomically.
    
    Performs atomic write using a temporary file and rename to ensure
    readers never see a partially written file. Uses consistent formatting
    with sorted keys for reproducibility.
    
    Args:
        path: Output file path
        obj: Object to serialize
        indent: Indentation level for pretty printing
    """
    file_path = Path(path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize to JSON string
    json_text = json.dumps(
        obj,
        indent=indent,
        sort_keys=True,
        ensure_ascii=False
    )
    
    # Atomic write using temporary file
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    
    try:
        # Write to temporary file
        with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_text)
            f.write("\n")  # Add trailing newline
            f.flush()
            
            # Force write to disk
            os.fsync(f.fileno())
        
        # Atomic rename
        os.replace(temp_path, file_path)
        
    except Exception:
        # Clean up temporary file on error
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    
    if hours > 0:
        parts.append(f"{hours}h")
    
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    if secs > 0 or len(parts) == 0:
        # Always show seconds if no larger units, or if non-zero
        if secs == int(secs):
            parts.append(f"{int(secs)}s")
        else:
            parts.append(f"{secs:.1f}s")
    
    return " ".join(parts)


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO 8601 formatted timestamp string
    """
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_path(path: PathLike) -> Path:
    """
    Convert path-like object to Path and create parent directories.
    
    Args:
        path: Path-like object
        
    Returns:
        Path object with parent directories created
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj