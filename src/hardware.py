#!/usr/bin/env python3
"""
Hardware Configuration Module
==============================

Optimizes low-level hardware settings for efficient training on NVIDIA GPUs
(Ampere/Hopper and newer) or CPU fallback.

Configurable via cfg["system"]:
  - tf32: bool              enable TensorFloat-32 for CUDA matmul/cuDNN (default: True)
  - cudnn_benchmark: bool   cuDNN autotune (disabled if deterministic=True)
  - deterministic: bool     force deterministic algorithms (slower)
  - omp_num_threads: int    sets OMP_NUM_THREADS and (best-effort) torch CPU threads
"""

from __future__ import annotations

import logging
import os
from typing import Dict
import torch


def setup_device() -> torch.device:
    """
    Select the compute device with proper priority handling.
    """
    # Check CUDA first (highest priority)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check Apple Silicon / Metal (second priority)
    if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_built"):
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")

    # Fallback to CPU
    return torch.device("cpu")

def optimize_hardware(system_cfg: Dict, device: torch.device, logger: logging.Logger) -> None:
    """
    Apply hardware optimizations with better error handling.
    """
    tf32 = bool(system_cfg.get("tf32", True))
    deterministic = bool(system_cfg.get("deterministic", False))
    cudnn_benchmark = bool(system_cfg.get("cudnn_benchmark", True)) and (not deterministic)

    # OMP threads with better error handling
    omp = system_cfg.get("omp_num_threads")
    if omp is not None:
        try:
            omp_int = int(omp)
            if omp_int > 0:
                os.environ["OMP_NUM_THREADS"] = str(omp_int)
                logger.info(f"Set OMP_NUM_THREADS={omp_int}")
                try:
                    torch.set_num_threads(omp_int)
                    logger.info(f"Set torch threads={omp_int}")
                except RuntimeError as e:
                    # Important failure - log at warning level
                    logger.warning(f"Failed to set torch.set_num_threads({omp_int}): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid omp_num_threads value '{omp}': {e}")

    # CUDA / cuDNN / TF32 settings
    if device.type == "cuda":
        # TF32 for matmul
        if hasattr(torch.backends, "cuda"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                logger.debug(f"Set cuda.matmul.allow_tf32={tf32}")
            except AttributeError as e:
                logger.warning(f"Cannot set cuda.matmul.allow_tf32: {e}")

        # TF32 for cuDNN
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.allow_tf32 = tf32
                logger.debug(f"Set cudnn.allow_tf32={tf32}")
            except AttributeError as e:
                logger.warning(f"Cannot set cudnn.allow_tf32: {e}")

        # Float32 matmul precision (PyTorch >= 2.0)
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high" if tf32 else "highest")
                logger.debug(f"Set float32_matmul_precision={'high' if tf32 else 'highest'}")
            except RuntimeError as e:
                logger.warning(f"Cannot set float32_matmul_precision: {e}")

    # cuDNN autotune / determinism
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if deterministic:
            torch.backends.cudnn.deterministic = True
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                    logger.info("Enabled deterministic algorithms")
                except RuntimeError as e:
                    logger.warning(f"Cannot enable deterministic algorithms: {e}")

    # Log final effective settings
    eff_bench = getattr(torch.backends.cudnn, "benchmark", None) if hasattr(torch.backends, "cudnn") else None
    eff_det = getattr(torch.backends.cudnn, "deterministic", None) if hasattr(torch.backends, "cudnn") else None
    logger.info(
        f"Hardware settings: TF32={tf32}, cudnn.benchmark={eff_bench}, "
        f"deterministic={deterministic}, cudnn.deterministic={eff_det}"
    )