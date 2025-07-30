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
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_memory:.1f} GB)")
        
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
        
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU device ({os.cpu_count()} cores)")
    
    return device


def optimize_hardware(config: Dict[str, Any], device: torch.device) -> None:
    """Apply hardware-specific optimizations with safe feature detection."""
    logger = logging.getLogger(__name__)
    
    # CUDA optimizations
    if device.type == "cuda":
        # Enable TensorFloat-32 for faster matmul
        if config.get("tf32", True):
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("TensorFloat-32 enabled for matmul")
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TensorFloat-32 enabled for cuDNN")
        
        # Enable cuDNN autotuner
        if config.get("cudnn_benchmark", True) and hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN autotuner enabled")
        
        # Set memory fraction
        memory_fraction = config.get("cuda_memory_fraction", 0.9)
        if memory_fraction < 1.0 and hasattr(torch.cuda, "set_per_process_memory_fraction"):
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"CUDA memory fraction set to {memory_fraction}")
            except Exception as e:
                logger.warning(f"Could not set CUDA memory fraction: {e}")
    
    # Set number of threads for CPU operations
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    logger.info(f"Using {torch.get_num_threads()} CPU threads")