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
            - omp_threads: Number of OpenMP threads (optional)
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


def _configure_cuda_optimizations(
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Configure CUDA-specific optimizations.
    
    Args:
        config: System configuration dictionary
        logger: Logger instance
    """
    try:
        # Enable TF32 for matrix operations on Ampere and newer GPUs
        # TF32 provides significant speedup with minimal accuracy impact
        enable_tf32 = bool(config.get("tf32", True))
        
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TensorFloat-32 (TF32) enabled for matrix operations")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            logger.info("TensorFloat-32 (TF32) disabled for matrix operations")
        
        # Configure cuDNN settings
        if hasattr(torch.backends, "cudnn"):
            # Enable TF32 for convolutions if requested
            if enable_tf32:
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for cuDNN operations")
            else:
                torch.backends.cudnn.allow_tf32 = False
            
            # Enable cuDNN autotuner for optimal convolution algorithms
            # This finds the fastest algorithms for the specific hardware
            enable_benchmark = bool(config.get("cudnn_benchmark", True))
            
            if enable_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark mode enabled (autotuner active)")
            else:
                torch.backends.cudnn.benchmark = False
                logger.info("cuDNN benchmark mode disabled")
        
        # Report GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            logger.info(
                f"Using GPU: {gpu_name} "
                f"(compute capability {gpu_capability[0]}.{gpu_capability[1]})"
            )
            
            # Log memory information
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU memory: {total_memory / (1024**3):.1f} GiB")
            
    except Exception as e:
        logger.warning(f"Could not configure CUDA optimizations: {e}")


def _configure_cpu_threading(
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Configure CPU threading for optimal performance.
    
    Args:
        config: System configuration dictionary
        logger: Logger instance
    """
    # Determine number of threads
    if "omp_threads" in config:
        num_threads = int(config["omp_threads"])
    else:
        # Default to CPU count
        num_threads = os.cpu_count() or 8
    
    # Set OpenMP environment variable if not already set
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        logger.info(f"Set OMP_NUM_THREADS={num_threads}")
    else:
        existing = os.environ["OMP_NUM_THREADS"]
        logger.info(f"OMP_NUM_THREADS already set to {existing}")
    
    # Also configure PyTorch threads for consistency
    torch.set_num_threads(num_threads)
    logger.info(f"PyTorch CPU threads set to {num_threads}")


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
                "total_memory_gb": props.total_memory / (1024**3),
                "multi_processor_count": props.multi_processor_count,
            }
            info["devices"].append(device_info)
    
    return info