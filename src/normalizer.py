#!/usr/bin/env python3
"""
Normalization Module
=====================
Provides normalization and denormalization operations for flow-map DeepONet.

This module centralizes all normalization logic, ensuring consistent handling
of physical variables across the pipeline. All normalization parameters are
read from the preprocessed normalization.json manifest.

Supported normalization methods:
- standard: (x - mean) / std
- min-max: (x - min) / (max - min)
- log-standard: (log10(x) - log_mean) / log_std
- log-min-max: (log10(x) - log_min) / (log_max - log_min)
"""

from __future__ import annotations

from typing import Dict, Any, List, Sequence, Optional

import torch


def as_list(x: Sequence[str] | str) -> List[str]:
    """Convert string or sequence to list of strings."""
    if isinstance(x, str):
        return [x]
    return list(x)


def get_device(*tensors: torch.Tensor) -> torch.device:
    """Get device from first available tensor."""
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


def get_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Get dtype from first available tensor."""
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            return tensor.dtype
    return torch.float32


class NormalizationHelper:
    """
    Manages normalization operations using preprocessed statistics.
    
    This class loads normalization parameters from a manifest file and provides
    methods to normalize and denormalize data consistently across the pipeline.
    All numerical parameters (epsilon, min_std, clamp_value) and per-variable
    statistics are centralized here.
    """
    
    def __init__(
        self,
        manifest: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize normalization helper.
        
        Args:
            manifest: Dictionary from normalization.json containing:
                - per_key_stats: Statistics for each variable
                - normalization_methods: Method for each variable
                - epsilon: Floor value for log operations
                - min_std: Minimum standard deviation
                - clamp_value: Optional clamping threshold
                - dt: Optional dt normalization parameters
            device: Device for tensor operations (default: cpu)
        """
        self.manifest = manifest or {}
        self.per_key_stats = dict(self.manifest.get("per_key_stats", {}))
        self.methods = dict(self.manifest.get("normalization_methods", {}))
        
        # Numerical stability parameters
        self.epsilon = float(self.manifest.get("epsilon", 1e-30))
        self.min_std = float(self.manifest.get("min_std", 1e-10))
        
        # Optional output clamping
        self.clamp_value = self.manifest.get("clamp_value", None)
        if self.clamp_value is not None:
            self.clamp_value = float(self.clamp_value)
        
        # Centralized dt normalization specification
        self.dt_spec = self.manifest.get("dt", None)
        if self.dt_spec is not None:
            self.dt_spec = dict(self.dt_spec)
            self.dt_spec.setdefault("method", "log-min-max")
        
        self.device = device or torch.device("cpu")
    
    def normalize(
        self,
        x: torch.Tensor,
        keys: Sequence[str]
    ) -> torch.Tensor:
        """
        Normalize data using per-variable methods.
        
        Args:
            x: Input tensor of shape [..., len(keys)]
            keys: Variable names for each column
            
        Returns:
            Normalized tensor with same shape as input
        """
        if x.ndim == 1:
            # Handle 1D input as single column
            keys = as_list(keys)
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._normalize_columns(x.unsqueeze(-1), keys)[..., 0]
        
        return self._normalize_columns(x, keys)
    
    def denormalize(
        self,
        x: torch.Tensor,
        keys: Sequence[str]
    ) -> torch.Tensor:
        """
        Inverse normalization operation.
        
        Args:
            x: Normalized tensor of shape [..., len(keys)]
            keys: Variable names for each column
            
        Returns:
            Denormalized tensor with same shape as input
        """
        if x.ndim == 1:
            keys = as_list(keys)
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._denormalize_columns(x.unsqueeze(-1), keys)[..., 0]
        
        return self._denormalize_columns(x, keys)
    
    def normalize_time(
        self,
        t_abs: torch.Tensor,
        time_key: str = "t_time"
    ) -> torch.Tensor:
        """
        Normalize absolute time values.
        
        Args:
            t_abs: Absolute time tensor
            time_key: Name of time variable in manifest
            
        Returns:
            Normalized time tensor
        """
        return self.normalize(t_abs, [time_key])
    
    def normalize_dt_from_phys(
        self,
        dt_phys: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize physical time differences.
        
        Uses centralized dt specification from manifest if available,
        otherwise falls back to time variable bounds.
        
        Args:
            dt_phys: Physical time differences
            
        Returns:
            Normalized dt tensor in [0, 1]
        """
        device = get_device(dt_phys) or self.device
        dtype = get_dtype(dt_phys)
        
        # Get normalization bounds
        if self.dt_spec is not None and self.dt_spec.get("method") == "log-min-max":
            log_min = float(self.dt_spec.get("log_min", -3.0))
            log_max = float(self.dt_spec.get("log_max", 8.0))
        else:
            # Fallback to time variable statistics
            time_stats = self.per_key_stats.get("t_time", {})
            log_min = float(time_stats.get("log_min", -3.0))
            log_max = float(time_stats.get("log_max", 8.0))
        
        denominator = max(log_max - log_min, 1e-12)
        
        # Apply log-min-max normalization
        dt_phys = dt_phys.to(device=device, dtype=torch.float64)
        dt_clipped = torch.clamp(dt_phys, min=self.epsilon)
        dt_log = torch.log10(dt_clipped)
        dt_norm = (dt_log - log_min) / denominator
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        
        return dt_norm.to(dtype=dtype)
    
    @torch.no_grad()
    def make_dt_norm_table(
        self,
        time_grid_phys: torch.Tensor,
        min_steps: int,
        max_steps: int,
    ) -> torch.Tensor:
        """
        Build precomputed dt normalization lookup table.
        
        Creates a [T, T] table where entry [i, j] contains the normalized
        time difference for time steps i and j, valid only when
        min_steps <= (j - i) <= max_steps.
        
        Args:
            time_grid_phys: Physical time grid of shape [T]
            min_steps: Minimum allowed time steps
            max_steps: Maximum allowed time steps
            
        Returns:
            Lookup table of shape [T, T] with dtype float32
            
        Raises:
            ValueError: If step bounds are invalid
        """
        if min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {min_steps}")
        if max_steps < min_steps:
            raise ValueError(
                f"max_steps must be >= min_steps, got {max_steps} < {min_steps}"
            )
        
        device = get_device(time_grid_phys) or self.device
        t = time_grid_phys.to(device=device, dtype=torch.float64)
        T = int(t.numel())


        # Compute pairwise time differences
        t_i = t.view(T, 1)  # [T, 1]
        t_j = t.view(1, T)  # [1, T]
        dt = torch.clamp(t_j - t_i, min=self.epsilon)  # dt[i,j] = t[j] - t[i]
        
        # Normalize using centralized dt method
        dt_norm = self.normalize_dt_from_phys(dt)
        
        # Create validity mask
        steps = torch.arange(T, device=device, dtype=torch.int64)
        step_diff = steps.view(1, T) - steps.view(T, 1)  # j - i
        valid_mask = (step_diff >= min_steps) & (step_diff <= max_steps)





        # Zero out invalid entries
        dt_table = torch.where(valid_mask, dt_norm, torch.zeros_like(dt_norm))
        
        return dt_table.to(dtype=torch.float32)
    
    def _normalize_columns(
        self,
        x: torch.Tensor,
        keys: Sequence[str]
    ) -> torch.Tensor:
        """
        Normalize each column using its specified method.
        
        Args:
            x: Input tensor of shape [..., K] where K = len(keys)
            keys: Variable names for each column
            
        Returns:
            Normalized tensor
        """
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor for column-wise normalization")
        
        num_keys = len(keys)
        if x.shape[-1] != num_keys:
            raise ValueError(
                f"Shape mismatch: tensor has {x.shape[-1]} columns, "
                f"but {num_keys} keys provided"
            )
        
        result = x.clone()
        device = get_device(x) or self.device
        dtype = get_dtype(x)
        
        for col_idx, key in enumerate(keys):
            # Get method and statistics for this variable
            method = str(self.methods.get(key, self.manifest.get("default_method", "log-standard")))
            stats = self.per_key_stats.get(key, {})
            
            # Extract column and convert to float64 for precision
            col_data = result[..., col_idx].to(device=device, dtype=torch.float64)
            
            # Apply normalization method
            if method == "standard":
                mean = float(stats.get("mean", 0.0))
                std = max(float(stats.get("std", 1.0)), self.min_std)
                col_norm = (col_data - mean) / std
                
            elif method == "min-max":
                min_val = float(stats.get("min", 0.0))
                max_val = float(stats.get("max", 1.0))
                range_val = max(max_val - min_val, 1e-12)
                col_norm = (col_data - min_val) / range_val
                
            elif method == "log-standard":
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_log = torch.log10(torch.clamp(col_data, min=self.epsilon))
                col_norm = (col_log - log_mean) / log_std
                
            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                range_val = max(log_max - log_min, 1e-12)
                col_log = torch.log10(torch.clamp(col_data, min=self.epsilon))
                col_norm = (col_log - log_min) / range_val
                
            else:
                raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")
            
            # Convert back to original dtype and apply clamping if configured
            col_norm = col_norm.to(dtype=dtype)
            if self.clamp_value is not None and self.clamp_value > 0:
                col_norm = torch.clamp(col_norm, -self.clamp_value, self.clamp_value)
            
            result[..., col_idx] = col_norm
        
        return result
    
    def _denormalize_columns(
        self,
        x: torch.Tensor,
        keys: Sequence[str]
    ) -> torch.Tensor:
        """
        Denormalize each column using its specified method.
        
        Args:
            x: Normalized tensor of shape [..., K] where K = len(keys)
            keys: Variable names for each column
            
        Returns:
            Denormalized tensor
        """
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor for column-wise denormalization")
        
        num_keys = len(keys)
        if x.shape[-1] != num_keys:
            raise ValueError(
                f"Shape mismatch: tensor has {x.shape[-1]} columns, "
                f"but {num_keys} keys provided"
            )
        
        result = x.clone()
        device = get_device(x) or self.device
        dtype = get_dtype(x)
        
        for col_idx, key in enumerate(keys):
            # Get method and statistics for this variable
            method = str(self.methods.get(key, self.manifest.get("default_method", "log-standard")))
            stats = self.per_key_stats.get(key, {})
            
            # Extract column and convert to float64 for precision
            col_data = result[..., col_idx].to(device=device, dtype=torch.float64)
            
            # Apply inverse normalization
            if method == "standard":
                mean = float(stats.get("mean", 0.0))
                std = max(float(stats.get("std", 1.0)), self.min_std)
                col_denorm = col_data * std + mean
                
            elif method == "min-max":
                min_val = float(stats.get("min", 0.0))
                max_val = float(stats.get("max", 1.0))
                range_val = max(max_val - min_val, 1e-12)
                col_denorm = col_data * range_val + min_val
                
            elif method == "log-standard":
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_log = col_data * log_std + log_mean
                col_denorm = torch.pow(10.0, col_log)
                
            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                range_val = max(log_max - log_min, 1e-12)
                col_log = col_data * range_val + log_min
                col_denorm = torch.pow(10.0, col_log)
                
            else:
                raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")
            
            result[..., col_idx] = col_denorm.to(dtype=dtype)
        
        return result