#!/usr/bin/env python3
"""
Hardware detection and optimization utilities.

This module provides functions to:
- Detect and configure available compute devices
- Optimize settings for specific hardware (especially A100)
- Configure memory and computation settings
"""

import logging
import os
from typing import Dict, Any, Optional

import torch
import numpy as np

# Hardware constants
DEFAULT_CUDA_ALLOC_MB = 512
DEFAULT_THREAD_RATIO = 0.5
MIN_WORKERS = 4
MAX_WORKERS = 8
MIN_CPU_WORKERS = 2
MPS_WORKERS = 0
MEMORY_BUFFER_RATIO = 0.9
DEFAULT_PREFETCH_FACTOR = 2
MAX_PREFETCH_FACTOR = 16  # PyTorch hard limit
MIN_PREFETCH_FACTOR = 2   # PyTorch minimum

# Compute capability thresholds
AMPERE_COMPUTE_CAPABILITY = 8  # For TF32, BFloat16, Flash Attention


def setup_device() -> torch.device:
    """
    Detect and configure the best available compute device.
    
    Priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU
    
    Returns:
        Configured torch.device
    """
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        # CUDA available
        device = torch.device("cuda")
        
        # Log GPU information
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        logger.info(f"Using CUDA device: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        logger.info(f"Number of GPUs: {gpu_count}")
        
        # Set default GPU
        torch.cuda.set_device(current_device)
        
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
        logger.warning("MPS backend has limited operator support")
        logger.warning("torch.compile is disabled for MPS devices")
        
    else:
        # CPU fallback
        device = torch.device("cpu")
        logger.info("Using CPU device")
        logger.warning("Training will be significantly slower on CPU")
        
        # Log CPU info
        cpu_count = os.cpu_count()
        logger.info(f"CPU cores: {cpu_count}")
    
    return device


def optimize_hardware(config: Dict[str, Any], device: torch.device) -> None:
    """
    Apply hardware-specific optimizations based on configuration.
    
    Args:
        config: System configuration dictionary
        device: The device being used
    """
    logger = logging.getLogger(__name__)
    
    # Disable torch.compile for MPS devices
    if device.type == "mps" and config.get("use_torch_compile", False):
        logger.warning("Disabling torch.compile for MPS device due to compatibility issues")
        config["use_torch_compile"] = False
    
    # CUDA optimizations
    if torch.cuda.is_available():
        # Enable TensorFloat-32 on Ampere GPUs (A100, RTX 30xx)
        if config.get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TensorFloat-32 (TF32) enabled for matrix operations")
        
        # Enable cuDNN autotuner for optimal convolution algorithms
        if config.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN autotuner enabled")
        
        # Set CUDA memory allocator settings
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            # Use larger allocation blocks to reduce fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{DEFAULT_CUDA_ALLOC_MB}"
        
        # Disable CUDA synchronous operations for better performance
        if "CUDA_LAUNCH_BLOCKING" not in os.environ:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set number of threads for CPU operations
    if "OMP_NUM_THREADS" not in os.environ:
        # Use half the CPU cores for OpenMP threads
        cpu_count = os.cpu_count() or 1
        # For systems with E-cores, limit to physical cores
        omp_threads = max(1, min(int(cpu_count * DEFAULT_THREAD_RATIO), cpu_count // 2))
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    
    # PyTorch threading settings - must be set before any operations
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    
    # Enable anomaly detection if requested (debugging only)
    if config.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Anomaly detection enabled - this will slow down training")


def get_device_info(device: torch.device = None) -> Dict[str, Any]:
    """
    Get detailed information about the current compute device.
    
    Args:
        device: Device to query (defaults to current device)
        
    Returns:
        Dictionary containing device information
    """
    if device is None:
        device = setup_device()
        
    info = {
        "device_type": device.type,
        "device_name": None,
        "memory_gb": None,
        "compute_capability": None,
        "supports_tf32": False,
        "supports_bfloat16": False,
        "supports_flash_attention": False,
        "supports_compile": False
    }
    
    if device.type == "cuda":
        device_props = torch.cuda.get_device_properties(device.index or 0)
        
        info["device_name"] = device_props.name
        info["memory_gb"] = device_props.total_memory / 1e9
        info["compute_capability"] = f"{device_props.major}.{device_props.minor}"
        
        # Check capabilities based on compute capability
        compute_major = device_props.major
        compute_minor = device_props.minor
        
        # TF32 support (Ampere and newer, compute capability 8.0+)
        info["supports_tf32"] = compute_major >= AMPERE_COMPUTE_CAPABILITY
        
        # BFloat16 support (Ampere and newer)
        info["supports_bfloat16"] = compute_major >= AMPERE_COMPUTE_CAPABILITY
        
        # Flash Attention support (Ampere and newer with specific requirements)
        # Note: Ada (8.9) and Hopper (9.0) also support it
        info["supports_flash_attention"] = compute_major >= AMPERE_COMPUTE_CAPABILITY
        
        # Compilation support
        info["supports_compile"] = True
        
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon GPU"
        info["supports_compile"] = False  # MPS has limited compile support
        
    else:
        info["device_name"] = "CPU"
        info["supports_compile"] = True
        
    return info


def optimize_dataloader_settings(
    batch_size: int,
    device_type: str,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get optimized DataLoader settings based on hardware.
    
    Args:
        batch_size: Batch size for training
        device_type: Type of compute device (cuda/mps/cpu)
        num_workers: Override for number of workers
        
    Returns:
        Dictionary of DataLoader settings
    """
    settings = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None  # Will be set based on device
    }
    
    if device_type == "cuda":
        # Enable pinned memory for faster GPU transfer
        settings["pin_memory"] = True
        
        # Use multiple workers for data loading
        if num_workers is None:
            # Use 4-8 workers typically
            cpu_count = os.cpu_count() or 1
            settings["num_workers"] = min(MAX_WORKERS, max(MIN_WORKERS, cpu_count // 2))
        else:
            settings["num_workers"] = num_workers
        
        # Enable persistent workers to avoid recreation overhead
        if settings["num_workers"] > 0:
            settings["persistent_workers"] = True
            # Set prefetch factor within PyTorch limits
            settings["prefetch_factor"] = min(
                max(DEFAULT_PREFETCH_FACTOR, settings["num_workers"] // 2),
                MAX_PREFETCH_FACTOR
            )
        else:
            # When num_workers is 0, prefetch_factor must be None
            settings["prefetch_factor"] = None
            
    elif device_type == "mps":
        # MPS doesn't work well with multiprocessing
        settings["num_workers"] = MPS_WORKERS
        settings["pin_memory"] = False
        settings["persistent_workers"] = False
        # No prefetch when no workers
        settings["prefetch_factor"] = None
        
    elif device_type == "cpu":
        # For CPU, use fewer workers to avoid overhead
        if num_workers is None:
            settings["num_workers"] = min(MIN_CPU_WORKERS, os.cpu_count() or 1)
        else:
            settings["num_workers"] = num_workers
            
        # Set prefetch factor for CPU workers
        if settings["num_workers"] > 0:
            settings["persistent_workers"] = True
            settings["prefetch_factor"] = MIN_PREFETCH_FACTOR
        else:
            settings["prefetch_factor"] = None
    
    return settings