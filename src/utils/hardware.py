#!/usr/bin/env python3
"""
Hardware detection and optimization utilities.
"""

import logging
import os
from typing import Dict, Any
import torch


def setup_device() -> torch.device:
    """Detect and configure the best available compute device."""
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
        try:
            torch.cuda.set_device(idx)  # make it current explicitly
        except Exception:
            # If set_device fails, we still return an indexed device
            pass
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_memory = torch.cuda.get_device_properties(idx).total_memory / 1e9
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_memory:.1f} GB)")
        return device

    # Apple Silicon path (CPU-only PyTorch builds may not have mps)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
        return device

    # Fallback: CPU
    device = torch.device("cpu")
    logger.info(f"Using CPU device ({os.cpu_count()} cores)")
    return device


def optimize_hardware(config: Dict[str, Any], device: torch.device) -> None:
    """Apply hardware-specific optimizations with safe feature detection."""
    logger = logging.getLogger(__name__)

    # Matmul precision hint (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision("high")
        logger.info("float32 matmul precision set to 'high'")
    except Exception:
        pass

    # CUDA optimizations
    if device.type == "cuda":
        # Enable TensorFloat-32 for faster matmul when default dtype is float32
        if config.get("tf32", True) and torch.get_default_dtype() == torch.float32:
            try:
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                    logger.info("TensorFloat-32 enabled for matmul")
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                    logger.info("TensorFloat-32 enabled for cuDNN")
            except Exception:
                pass

        # Enable cuDNN autotuner for faster convs on fixed shapes
        try:
            if config.get("cudnn_benchmark", True) and hasattr(torch.backends.cudnn, "benchmark"):
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                logger.info("cuDNN autotuner enabled")
        except Exception:
            pass

        # Set per-process CUDA memory fraction (requires an indexed device or int)
        memory_fraction = float(config.get("cuda_memory_fraction", 0.9))
        if memory_fraction < 1.0 and hasattr(torch.cuda, "set_per_process_memory_fraction"):
            try:
                idx = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=idx)  # type: ignore[arg-type]
                logger.info(f"CUDA memory fraction set to {memory_fraction} on device {idx}")
            except Exception as e:
                logger.warning(f"Could not set CUDA memory fraction: {e}")

    # Cap CPU-side thread count unless user specified OMP_NUM_THREADS
    if "OMP_NUM_THREADS" not in os.environ:
        try:
            torch.set_num_threads(min(32, os.cpu_count() or 1))
            logger.info(f"Using {torch.get_num_threads()} CPU threads (auto-set)")
        except Exception:
            pass
    else:
        logger.info(f"Using {os.environ['OMP_NUM_THREADS']} CPU threads (OMP_NUM_THREADS)")
