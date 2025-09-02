#!/usr/bin/env python3
"""
Hardware configuration helpers optimized for A100 training.
"""
import logging
import os
from typing import Dict, Any, Optional
import torch


def setup_device() -> torch.device:
    """Setup and return the compute device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    return device


def optimize_hardware(config: Dict[str, Any], device: torch.device, logger: Optional[logging.Logger] = None) -> None:
    """
    Optimize hardware settings for training.

    Args:
        config: System configuration dictionary
        device: Target device for computation
        logger: Optional logger instance
    """
    log = logger or logging.getLogger("hardware")

    if device.type == "cuda":
        try:
            # Enable TF32 for matrix operations on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            log.info("TF32 enabled; cuDNN benchmark enabled")
        except Exception as e:
            log.warning(f"Could not enable TF32/cuDNN benchmark: {e}")

    # Set OpenMP threads for CPU operations
    omp_threads = int(config.get("omp_threads", os.cpu_count() or 8))
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    log.info(f"Using OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    # Enable deterministic algorithms if requested (reduces performance)
    if bool(config.get("deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            log.info("Deterministic algorithms enabled (performance will drop)")
        except Exception as e:
            log.warning(f"Deterministic enable failed: {e}")