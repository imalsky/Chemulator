#!/usr/bin/env python3
"""
Hardware Configuration Module
==============================
Optimizes hardware settings for efficient training on GPUs, particularly
targeting NVIDIA A100 and similar architectures.

Configures:
- CUDA device selection
- TensorFloat-32 (TF32) for Ampere GPUs
- cuDNN benchmarking and optimization
- OpenMP thread configuration
- Optional deterministic mode for reproducibility
"""

import logging
import os
from typing import Dict, Any, Optional

import torch


def setup_device() -> torch.device:
    """
    Setup and return the compute device.

    Selects CUDA if available, otherwise falls back to CPU.

    Returns:
        torch.device: Selected compute device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set default CUDA device
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    return device


def optimize_hardware(
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[logging.Logger] = None
) -> None:
    """
    Optimize hardware settings for training performance.

    Configures various hardware-specific optimizations based on the
    device type and configuration settings.

    Args:
        config: System configuration dictionary containing:
            - omp_threads: Number of OpenMP threads (optional, accepts int, "auto", or null)
            - deterministic: Whether to use deterministic algorithms (optional)
            - tf32: Whether to enable TF32 (optional, default True for CUDA)
            - cudnn_benchmark: Whether to enable cuDNN benchmarking (optional)
        device: Target compute device
        logger: Logger instance for status messages (optional)
    """
    if logger is None:
        logger = logging.getLogger("hardware")

    # Configure GPU-specific optimizations
    if device.type == "cuda":
        _configure_cuda_optimizations(config, logger)

    # Configure CPU threading
    _configure_cpu_threading(config, logger)

    # Configure deterministic mode if requested
    if bool(config.get("deterministic", False)):
        _enable_deterministic_mode(logger)

def _configure_linalg_backend(config: dict, logger: logging.Logger) -> None:
    """
    Optionally select CUDA linalg backend ('cusolver' or 'magma') to avoid driver/library issues.
    """
    try:
        import torch
        prefer = str(config.get("linalg_library", "")).strip().lower()
        if prefer in ("cusolver", "magma"):
            torch.backends.cuda.preferred_linalg_library(prefer)  # set
            logger.info(f"Preferred CUDA linalg backend set to: {prefer}")
    except Exception as e:
        logger.debug(f"Unable to set preferred linalg backend: {e}")

def _configure_cuda_optimizations(config: dict, logger: logging.Logger) -> None:
    """
    Configure CUDA-specific knobs:
      - TF32 (matmul + cuDNN), with optional float32 matmul precision hint
      - cuDNN benchmark autotuner
      - Optional deterministic mode (delegated to _enable_deterministic_mode)
    """
    try:
        import torch
    except Exception as e:
        logger.warning(f"PyTorch not available, skipping CUDA optimizations: {e}")
        return

    if not torch.cuda.is_available():
        logger.info("CUDA not available; skipping CUDA optimizations")
        return

    # Basic device info (best-effort)
    try:
        dev_idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev_idx)
        major, minor = torch.cuda.get_device_capability(dev_idx)
        logger.info(f"Using CUDA device {dev_idx}: {name} (CC {major}.{minor})")
    except Exception as e:
        logger.debug(f"Unable to query CUDA device info: {e}")

    # -------- TF32 controls --------
    enable_tf32 = bool(config.get("tf32", True))
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    except Exception as e:
        logger.debug(f"TF32 toggles not available in this build: {e}")

    if enable_tf32:
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cuDNN")

        # Optional: let PyTorch pick faster FP32 matmul kernels on Ampere/Hopper
        # Safe no-op on older PyTorch; keep inside try to avoid hard dependency.
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("float32 matmul precision set to 'high'")
        except Exception:
            # Present on PyTorch >= 2.0; silently skip otherwise
            pass
    else:
        logger.info("TF32 disabled")
    _configure_linalg_backend(config, logger)

    # -------- cuDNN autotuner --------
    enable_benchmark = bool(config.get("cudnn_benchmark", True))
    try:
        torch.backends.cudnn.benchmark = enable_benchmark
        logger.info(f"cuDNN benchmark mode {'enabled' if enable_benchmark else 'disabled'}")
    except Exception as e:
        logger.debug(f"Unable to set cuDNN benchmark mode: {e}")

    # -------- Deterministic mode (optional) --------
    if bool(config.get("deterministic", False)):
        _enable_deterministic_mode(logger)


def _configure_cpu_threading(
        config: Dict[str, Any],
        logger: logging.Logger
) -> None:
    """
    Configure CPU threading for optimal performance.

    Precedence for thread count:
      1) config["omp_threads"] if a positive int (strings like "auto"/"none"/null ignored)
      2) Environment hints: OMP_NUM_THREADS, SLURM_CPUS_PER_TASK, PBS_NCPUS, NSLOTS
      3) os.cpu_count() fallback

    Always sets torch.set_num_threads() and, if not already set, OMP_NUM_THREADS.
    Handles null, "auto", "none", and numeric string values gracefully.
    """

    def _parse_int(x: Any) -> Optional[int]:
        """Parse value to int, returning None for auto/null/invalid values."""
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("", "none", "null", "auto", "default"):
            return None
        try:
            v = int(s)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    # 1) Explicit config wins (if valid)
    num_threads = _parse_int(config.get("omp_threads", None))

    # 2) Scheduler / env hints (check if OMP_NUM_THREADS already set first)
    if num_threads is None:
        for env_key in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS"):
            env_val = _parse_int(os.environ.get(env_key))
            if env_val is not None:
                num_threads = env_val
                if env_key == "OMP_NUM_THREADS":
                    logger.info(f"Using OMP_NUM_THREADS={env_val} (pre-set)")
                else:
                    logger.info(f"Using {env_key}={env_val} for CPU threading")
                break

    # 3) Fallback to CPU count
    if num_threads is None:
        num_threads = os.cpu_count() or 8
        logger.info(f"No valid thread hint found; using os.cpu_count()={num_threads}")

    # Clamp to a sane minimum
    num_threads = max(1, int(num_threads))

    # Set environment variable only if not already set (respect external schedulers)
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        logger.info(f"Set OMP_NUM_THREADS={num_threads}")
    else:
        # Already set - log current value
        current_val = os.environ["OMP_NUM_THREADS"]
        logger.info(f"OMP_NUM_THREADS already set to {current_val}")
        # Use the environment value for PyTorch
        try:
            num_threads = int(current_val)
        except (ValueError, TypeError):
            pass  # Keep computed value if env var is invalid

    # Align PyTorch thread count
    try:
        torch.set_num_threads(num_threads)
        logger.info(f"PyTorch CPU threads set to {num_threads}")
    except Exception as e:
        logger.warning(f"torch.set_num_threads({num_threads}) failed: {e}")


def _enable_deterministic_mode(logger: logging.Logger) -> None:
    """
    Enable deterministic algorithms for reproducibility.

    Note: This significantly reduces performance and should only be
    used when exact reproducibility is required.

    Args:
        logger: Logger instance
    """
    try:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True)
        logger.info("Deterministic algorithms enabled")

        # Configure cuDNN for determinism
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("cuDNN deterministic mode enabled (benchmark disabled)")

        logger.warning(
            "Deterministic mode is active. "
            "This will significantly reduce performance."
        )

    except Exception as e:
        logger.warning(f"Could not fully enable deterministic mode: {e}")


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.

    Returns:
        Dictionary containing device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "cpu_count": os.cpu_count(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()

        if hasattr(torch.backends, "cudnn"):
            info["cudnn_version"] = torch.backends.cudnn.version()

        # Get information for each GPU
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024 ** 3),
                "multi_processor_count": props.multi_processor_count,
            }
            info["devices"].append(device_info)

    return info