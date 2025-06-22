#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration.

This module provides utilities to detect the best available PyTorch device
(CUDA, MPS, CPU) and to configure device-specific DataLoader settings
for optimal performance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Selects and returns a `torch.device` object for the best available backend.
    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: A `torch.device` object (e.g., torch.device('cuda:0')).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info(f"Using CUDA device: {device_name}")
        except Exception:
            logger.info("Using CUDA device.") # Fallback message
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device.")
        logger.warning(
            "MPS backend detected. Note: Some operations may have limited support or "
            "performance. If you encounter errors, consider setting the environment "
            "variable PYTORCH_ENABLE_MPS_FALLBACK=1"
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    
    return device


def get_device_properties() -> Dict[str, Any]:
    """
    Retrieves a dictionary of properties for the selected backend.

    Returns:
        A dictionary containing properties of the selected device, such as
        type, name, memory (for CUDA), and AMP support.
    """
    device = setup_device()
    device_type = device.type
    
    properties: Dict[str, Any] = {
        "type": device_type,
        "supports_amp": device_type == "cuda"  # Currently, stable AMP is best on CUDA
    }

    if device_type == "cuda":
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            properties.update({
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 2),
                "capability": (props.major, props.minor),
            })
        except Exception as e:
            logger.warning(f"Could not read CUDA device properties: {e}")
    elif device_type == "mps":
        properties["name"] = "Apple Silicon GPU"
        
    return properties


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Returns recommended keyword arguments for a PyTorch DataLoader based on device type.
    This helps optimize data transfer to the GPU.

    Returns:
        A dictionary with recommended DataLoader settings like 'pin_memory'
        and 'persistent_workers'.
    """
    device_type = setup_device().type
    is_cuda = (device_type == "cuda")
    
    return {
        "pin_memory": is_cuda,          # Speeds up CPU-to-CUDA memory transfers
        "persistent_workers": is_cuda,  # Reduces worker startup overhead between epochs
    }


__all__ = [
    "setup_device",
    "get_device_properties", 
    "configure_dataloader_settings",
]