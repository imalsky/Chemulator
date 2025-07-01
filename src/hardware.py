#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration with auto-optimization.

This module provides utilities to detect the best available PyTorch device
(CUDA, MPS, CPU) and to configure device-specific DataLoader settings
for optimal performance, with automatic configuration based on hardware.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import psutil
from typing import Any, Dict, Optional, Union

import torch
import h5py
from pathlib import Path

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
    Retrieve detailed properties of the selected device.
    
    Returns:
        Dictionary containing device properties including type, name, memory,
        compute capability, and AMP support
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
                "multi_processor_count": props.multi_processor_count,
            })
            
            # Determine GPU tier for optimization
            memory_gb = properties["memory_gb"]
            if "A100" in props.name or "H100" in props.name or memory_gb >= 40:
                properties["gpu_tier"] = "high"
            elif "V100" in props.name or "A10" in props.name or memory_gb >= 24:
                properties["gpu_tier"] = "medium"
            else:
                properties["gpu_tier"] = "low"
                
        except Exception as e:
            logger.warning(f"Could not read CUDA device properties: {e}")
            properties["gpu_tier"] = "unknown"
    elif device_type == "mps":
        properties["name"] = "Apple Silicon GPU"
        properties["gpu_tier"] = "medium"
    else:
        properties["gpu_tier"] = "cpu"
        
    # Add system memory info
    try:
        vm = psutil.virtual_memory()
        properties["system_memory_gb"] = round(vm.total / (1024**3), 2)
        properties["available_memory_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        properties["system_memory_gb"] = 0
        properties["available_memory_gb"] = 0
        
    # Add CPU info
    properties["cpu_count"] = multiprocessing.cpu_count()
    
    return properties


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Return recommended DataLoader settings based on device type.
    
    Returns:
        Dictionary with DataLoader configuration like 'pin_memory' and 'persistent_workers'
    """
    device_type = setup_device().type
    is_cuda = (device_type == "cuda")
    
    return {
        "pin_memory": is_cuda,
        "persistent_workers": True,
    }


def auto_configure_training_params(
    config: Dict[str, Any], h5_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Automatically configure training parameters based on detected hardware.
    
    This function replaces 'auto' values in the config with optimal values
    for the detected hardware and dataset characteristics.
    
    Args:
        config: The configuration dictionary
        h5_path: Optional path to HDF5 file for dataset-specific optimizations
        
    Returns:
        Modified configuration dictionary with auto values replaced
    """
    properties = get_device_properties()
    gpu_tier = properties.get("gpu_tier", "unknown")
    gpu_memory_gb = properties.get("memory_gb", 0)
    system_memory_gb = properties.get("system_memory_gb", 16)
    cpu_count = properties.get("cpu_count", 4)
    
    # Check if using HDF5
    is_hdf5 = h5_path and str(h5_path).lower().endswith(('.h5', '.hdf5'))
    
    logger.info(f"Auto-configuring for: GPU tier={gpu_tier}, "
                f"GPU memory={gpu_memory_gb}GB, "
                f"System memory={system_memory_gb}GB, "
                f"CPUs={cpu_count}, "
                f"HDF5={is_hdf5}")
    
    # Training hyperparameters
    train_params = config.get("training_hyperparameters", {})
    
    # Auto-configure batch size based on GPU memory
    if train_params.get("batch_size") == "auto":
        if gpu_tier == "high":  # A100, H100, etc.
            base_batch_size = 8192
        elif gpu_tier == "medium":  # V100, A10, etc.
            base_batch_size = 4096
        elif gpu_tier == "low":  # Consumer GPUs
            base_batch_size = 1024
        else:  # CPU or unknown
            base_batch_size = 256
            
        if gpu_memory_gb > 0:
            memory_based_batch = int(gpu_memory_gb * 800)
            train_params["batch_size"] = min(base_batch_size, memory_based_batch)
        else:
            train_params["batch_size"] = base_batch_size
            
        logger.info(f"Auto-configured batch_size: {train_params['batch_size']}")
    
    # Auto-configure gradient accumulation
    if train_params.get("gradient_accumulation_steps") == "auto":
        batch_size = train_params.get("batch_size", 4096)
        target_effective_batch = batch_size * 4
        train_params["gradient_accumulation_steps"] = max(1, target_effective_batch // batch_size)
        logger.info(f"Auto-configured gradient_accumulation_steps: {train_params['gradient_accumulation_steps']}")
    
    # Auto-configure mixed precision
    if train_params.get("use_amp") == "auto":
        if properties.get("type") == "cuda" and properties.get("capability", (0, 0))[0] >= 7:
            train_params["use_amp"] = True
        else:
            train_params["use_amp"] = False
        logger.info(f"Auto-configured use_amp: {train_params['use_amp']}")
    
    # Miscellaneous settings
    misc_settings = config.get("miscellaneous_settings", {})
    
    # Auto-configure number of dataloader workers
    if misc_settings.get("num_dataloader_workers") == "auto":
        logger.info(
            "Setting 'num_dataloader_workers' to 'auto'. "
            "Final value will be determined by the trainer based on caching state."
        )
    
    # Auto-configure memory settings
    if misc_settings.get("profiles_per_chunk") == "auto":
        # Estimate based on available system memory
        # Conservative estimate for chemical kinetics data
        bytes_per_profile = 15 * 1024
        # Use 25% of available memory for chunking
        available_for_chunks = int(system_memory_gb * 0.25 * 1024 * 1024 * 1024)
        profiles_per_chunk = min(50000, max(1024, available_for_chunks // bytes_per_profile))
        misc_settings["profiles_per_chunk"] = profiles_per_chunk
        logger.info(f"Auto-configured profiles_per_chunk: {profiles_per_chunk}")
    
    if misc_settings.get("max_memory_per_worker_gb") == "auto":
        # Safe default of 2GB per worker prevents memory issues
        safe_default_mem_gb = 2.0
        misc_settings["max_memory_per_worker_gb"] = safe_default_mem_gb
        logger.info(
            f"Auto-configured max_memory_per_worker_gb: {safe_default_mem_gb} "
            "(safe default)"
        )
            
    # Auto-configure torch compile
    if misc_settings.get("use_torch_compile") == "auto":
        # Enable for high-tier GPUs with PyTorch 2.0+
        if gpu_tier == "high" and hasattr(torch, "compile"):
            misc_settings["use_torch_compile"] = True
            misc_settings["torch_compile_mode"] = "max-autotune"
        else:
            misc_settings["use_torch_compile"] = False
        logger.info(f"Auto-configured use_torch_compile: {misc_settings['use_torch_compile']}")
    
    # Model-specific optimizations
    model_params = config.get("model_hyperparameters", {})
    if model_params.get("use_gradient_checkpointing") == "auto":
        # Enable for limited GPU memory
        if gpu_memory_gb > 0 and gpu_memory_gb < 16:
            model_params["use_gradient_checkpointing"] = True
        else:
            model_params["use_gradient_checkpointing"] = False
        logger.info(f"Auto-configured use_gradient_checkpointing: {model_params['use_gradient_checkpointing']}")
    
    # Auto-configure cache_dataset with robust error handling
    if misc_settings.get("cache_dataset") == "auto":
        estimated_dataset_gb = _estimate_dataset_size(config, h5_path)
        
        # Cache if we have enough memory (need 2.5x for safety)
        memory_threshold = 2.5
        if system_memory_gb > estimated_dataset_gb * memory_threshold:
            misc_settings["cache_dataset"] = True
            logger.info(
                f"Auto-configured cache_dataset: True "
                f"(dataset ~{estimated_dataset_gb:.2f}GB, system has {system_memory_gb:.2f}GB)"
            )
        else:
            misc_settings["cache_dataset"] = False
            logger.info(
                f"Auto-configured cache_dataset: False "
                f"(dataset ~{estimated_dataset_gb:.2f}GB needs {estimated_dataset_gb * memory_threshold:.2f}GB, "
                f"system has {system_memory_gb:.2f}GB)"
            )
    
    return config


def _estimate_dataset_size(config: Dict[str, Any], h5_path: Optional[Union[str, Path]]) -> float:
    """
    Estimate dataset size with robust error handling.
    
    Args:
        config: Configuration dictionary
        h5_path: Path to HDF5 file
        
    Returns:
        Estimated dataset size in GB
    """
    estimated_dataset_gb = 20.0
    
    if not h5_path or not Path(h5_path).exists():
        logger.warning(f"HDF5 path not provided or doesn't exist. Using default estimate: {estimated_dataset_gb}GB")
        return estimated_dataset_gb
    
    try:
        with h5py.File(h5_path, 'r') as hf:
            if not hf.keys():
                logger.warning("HDF5 file is empty. Using default estimate.")
                return estimated_dataset_gb
            
            all_vars = config.get("data_specification", {}).get("all_variables", [])
            if not all_vars:
                logger.warning("No variables specified in config. Using default estimate.")
                return estimated_dataset_gb
            
            total_bytes = 0
            vars_found = 0
            
            for var in all_vars:
                if var in hf:
                    try:
                        dataset = hf[var]
                        # Calculate size: total elements * bytes per element
                        var_bytes = dataset.size * dataset.dtype.itemsize
                        total_bytes += var_bytes
                        vars_found += 1
                    except Exception as e:
                        logger.warning(f"Error reading variable '{var}': {e}")
                        continue
            
            if vars_found == 0:
                logger.warning("No variables could be read from HDF5. Using default estimate.")
                return estimated_dataset_gb
            
            # If we only found some variables, extrapolate
            if vars_found < len(all_vars):
                logger.warning(
                    f"Only {vars_found}/{len(all_vars)} variables found. "
                    f"Extrapolating total size."
                )
                total_bytes = total_bytes * len(all_vars) / vars_found
            
            estimated_dataset_gb = total_bytes / (1024**3)
            
            # Sanity check - if estimate seems unreasonable, use default
            if estimated_dataset_gb < 0.001 or estimated_dataset_gb > 1000:
                logger.warning(
                    f"Estimated size {estimated_dataset_gb:.2f}GB seems unreasonable. "
                    f"Using default estimate: 8.0GB"
                )
                return 8.0
                
    except Exception as e:
        logger.warning(f"Could not estimate dataset size due to error: {e}. Using default estimate: {estimated_dataset_gb}GB")
    
    return estimated_dataset_gb


def get_optimal_device_config(device_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get optimal configuration recommendations for specific devices.
    
    Args:
        device_name: Optional device name override
        
    Returns:
        Dictionary of recommended settings for the device
    """
    if device_name is None:
        properties = get_device_properties()
        device_name = properties.get("name", "unknown")
    
    # Device-specific optimizations for SIREN models
    configs = {
        "A100": {
            "batch_size": 8192,
            "use_amp": True,
            "use_torch_compile": True,
            "torch_compile_mode": "max-autotune",
            "profiles_per_chunk": 50000,
            "cache_dataset": True,
        },
        "V100": {
            "batch_size": 4096,
            "use_amp": True,
            "use_torch_compile": False,
            "profiles_per_chunk": 20000,
            "cache_dataset": False,
        },
        "default": {
            "batch_size": 1024,
            "use_amp": False,
            "use_torch_compile": False,
            "profiles_per_chunk": 5000,
            "cache_dataset": False,
        }
    }
    
    # Find matching config
    for key, config in configs.items():
        if key in device_name:
            return config
    
    return configs["default"]


__all__ = [
    "setup_device",
    "get_device_properties", 
    "configure_dataloader_settings",
    "auto_configure_training_params",
    "get_optimal_device_config",
]