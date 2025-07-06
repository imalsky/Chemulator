#!/usr/bin/env python3
"""
hardware.py – Basic device detection and DataLoader configuration.

This module provides utilities to detect the best available PyTorch device
(CUDA, MPS, CPU) and basic DataLoader settings.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Select and return the best available PyTorch device.
    
    Priority order: CUDA > MPS (Apple Silicon) > CPU.
    
    Returns:
        torch.device: The selected device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info(f"Using CUDA device: {device_name}")
        except Exception:
            logger.info("Using CUDA device.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device.")
        logger.warning(
            "MPS backend detected. Note: Some operations may be slower than CUDA."
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    
    return device


def get_device_properties() -> Dict[str, Any]:
    """
    Retrieve basic properties of the selected device.
    
    Returns:
        Dictionary containing device properties
    """
    device = setup_device()
    device_type = device.type
    
    properties: Dict[str, Any] = {
        "type": device_type,
        "supports_amp": device_type == "cuda"
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
    Return basic DataLoader settings based on device type.
    
    Returns:
        Dictionary with DataLoader configuration
    """
    device_type = setup_device().type
    is_cuda = (device_type == "cuda")
    
    return {
        "pin_memory": is_cuda,
        "persistent_workers": True,
    }


__all__ = [
    "setup_device",
    "get_device_properties", 
    "configure_dataloader_settings",
]