#!/usr/bin/env python3
"""
Normalization module for chemical kinetics data.

Handles all data transformations including log10 and normalization in a unified pipeline.
Uses pre-computed statistics from the preprocessing stage to apply transformations efficiently.
"""

import logging
from typing import Dict, List, Any
import torch

# Constants
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP_VALUE = 50.0
MIN_DENOMINATOR = 1e-8


class NormalizationHelper:
    """
    Apply normalization transformations to data using pre-computed statistics.
    
    Supports multiple normalization strategies:
    - log-standard: log10(x) then standardize
    - standard: standardize (subtract mean, divide by std)
    - log10: just apply log10 transformation
    - min-max: scale to [0,1] using min/max
    - log-min-max: log10 then min-max scaling
    - none: pass-through without modification
    """
    
    def __init__(
        self,
        stats: Dict[str, Any],
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize normalization helper with pre-computed statistics.
        
        Args:
            stats: Pre-computed statistics from preprocessing
            device: Device for tensor operations
            config: Configuration with normalization settings
        """
        self.stats = stats
        self.device = device
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.data_cfg = config.get("data", {})
        self.norm_cfg = config.get("normalization", {})
        
        # Set up data type
        dtype_name = config.get("system", {}).get("dtype", "float32")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32
        
        # Extract methods and statistics
        self.methods = stats.get("normalization_methods", {})
        self.per_key_stats = stats.get("per_key_stats", {})
        self.time_norm = stats.get("time_normalization", {})
        
        # Parameters
        self.epsilon = float(self.norm_cfg.get("epsilon", DEFAULT_EPSILON))
        self.clamp_val = float(self.norm_cfg.get("clamp_value", DEFAULT_CLAMP_VALUE))
        self.min_std = float(self.norm_cfg.get("min_std", DEFAULT_MIN_STD))
        
        # Precompute constants
        self._setup_constants()
    
    def _setup_constants(self) -> None:
        """Initialize constant tensors for efficient computation."""
        self.epsilon_tensor = torch.tensor(self.epsilon, dtype=self.dtype, device=self.device)
        
        # Time normalization parameters
        if self.time_norm:
            time_method = self.time_norm.get("time_transform")
            
            if time_method == "time-norm":
                # Tau-space parameters
                self._tau0 = torch.tensor(
                    self.time_norm["tau0"], dtype=self.dtype, device=self.device
                )
                self._tau_min = torch.tensor(
                    self.time_norm["tmin"], dtype=self.dtype, device=self.device
                )
                self._tau_max = torch.tensor(
                    self.time_norm["tmax"], dtype=self.dtype, device=self.device
                )
                self._tau_range = torch.clamp(
                    self._tau_max - self._tau_min, min=MIN_DENOMINATOR
                )
            elif time_method == "log-min-max":
                # Log-space min-max parameters
                tmin = max(self.time_norm["tmin_raw"], self.epsilon)
                tmax = max(self.time_norm["tmax_raw"], tmin + self.epsilon)
                self._tlog_min = torch.log10(torch.tensor(tmin, dtype=self.dtype, device=self.device))
                self._tlog_max = torch.log10(torch.tensor(tmax, dtype=self.dtype, device=self.device))
                self._tlog_range = torch.clamp(
                    self._tlog_max - self._tlog_min, min=MIN_DENOMINATOR
                )
    
    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Apply normalization including log10 transformation where specified.
        
        Args:
            data: Input tensor of shape (..., D) where D = len(var_list)
            var_list: List of variable names for each dimension
            
        Returns:
            Normalized tensor clamped to [-clamp_val, clamp_val]
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data dimension {data.shape[-1]} != variable count {len(var_list)}")
        
        # Ensure correct device and dtype
        data = data.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(data)
        
        # Process each variable
        for i, var in enumerate(var_list):
            method = self.methods.get(var, "none")
            var_stats = self.per_key_stats.get(var, {})
            
            # Special handling for time variable
            if var == self.data_cfg.get("time_variable", "t_time"):
                out[..., i] = self._normalize_time(data[..., i], method)
            else:
                out[..., i] = self._normalize_variable(data[..., i], method, var_stats)
        
        return torch.clamp(out, -self.clamp_val, self.clamp_val)

    def _normalize_variable(
            self,
            values: torch.Tensor,
            method: str,
            stats: Dict[str, float]
    ) -> torch.Tensor:
        """Apply normalization to a single variable with validation."""
        if method == "log-standard":
            if "log_mean" not in stats or "log_std" not in stats:
                raise ValueError(f"Missing required 'log_mean'/'log_std' stats for method '{method}'. "
                                 f"Available stats: {list(stats.keys())}")
            # Apply log10 then standardize
            log_values = torch.log10(torch.clamp(values, min=self.epsilon))
            mean = stats["log_mean"]
            std = max(stats["log_std"], self.min_std)
            return (log_values - mean) / std

        elif method == "standard":
            if "mean" not in stats or "std" not in stats:
                raise ValueError(f"Missing required 'mean'/'std' stats for method '{method}'. "
                                 f"Available stats: {list(stats.keys())}")
            # Direct standardization
            mean = stats["mean"]
            std = max(stats["std"], self.min_std)
            return (values - mean) / std

        elif method == "log-min-max":
            if "log_min" not in stats or "log_max" not in stats:
                raise ValueError(f"Missing required 'log_min'/'log_max' stats for method '{method}'. "
                                 f"Available stats: {list(stats.keys())}")
            # Log10 then min-max scaling
            log_values = torch.log10(torch.clamp(values, min=self.epsilon))
            log_min = stats["log_min"]
            log_max = stats["log_max"]
            range_val = max(log_max - log_min, MIN_DENOMINATOR)
            return (log_values - log_min) / range_val

        elif method == "min-max":
            if "min" not in stats or "max" not in stats:
                raise ValueError(f"Missing required 'min'/'max' stats for method '{method}'. "
                                 f"Available stats: {list(stats.keys())}")
            # Direct min-max scaling
            min_val = stats["min"]
            max_val = stats["max"]
            range_val = max(max_val - min_val, MIN_DENOMINATOR)
            return (values - min_val) / range_val

        elif method == "log10":
            # Updated: log10 transformation with mean centering
            if "log_mean" not in stats:
                raise ValueError(f"Missing required 'log_mean' stats for method '{method}'. "
                                 f"Available stats: {list(stats.keys())}")
            log_values = torch.log10(torch.clamp(values, min=self.epsilon))
            mean = stats["log_mean"]
            return log_values - mean

        else:  # "none"
            return values
    
    def _normalize_time(self, t: torch.Tensor, method: str) -> torch.Tensor:
        """Normalize time values with special handling."""
        if method == "time-norm" and hasattr(self, '_tau0'):
            # Tau-space normalization: tau = log1p(t / tau0)
            tau = torch.log1p(t / self._tau0)
            return torch.clamp((tau - self._tau_min) / self._tau_range, 0.0, 1.0)
            
        elif method == "log-min-max" and hasattr(self, '_tlog_min'):
            # Log-space min-max normalization
            t_log = torch.log10(torch.clamp(t, min=self.epsilon))
            return torch.clamp((t_log - self._tlog_min) / self._tlog_range, 0.0, 1.0)
            
        else:
            # Fall back to standard variable normalization
            time_var = self.data_cfg.get("time_variable", "t_time")
            return self._normalize_variable(t, method, self.per_key_stats.get(time_var, {}))
    
    def denormalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Invert normalization to recover original values.
        
        Args:
            data: Normalized tensor of shape (..., D)
            var_list: List of variable names for each dimension
            
        Returns:
            Denormalized tensor in original space
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data dimension {data.shape[-1]} != variable count {len(var_list)}")
        
        data = data.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(data)
        
        for i, var in enumerate(var_list):
            method = self.methods.get(var, "none")
            var_stats = self.per_key_stats.get(var, {})
            
            # Special handling for time
            if var == self.data_cfg.get("time_variable", "t_time"):
                out[..., i] = self._denormalize_time(data[..., i], method)
            else:
                out[..., i] = self._denormalize_variable(data[..., i], method, var_stats)
        
        return out

    def _denormalize_variable(
            self,
            values: torch.Tensor,
            method: str,
            stats: Dict[str, float]
    ) -> torch.Tensor:
        """
        Invert normalization for a single variable.

        Includes overflow/underflow protection for exponential operations by clamping
        log values to a safe range before exponentiation. This prevents inf/zero
        values that would corrupt loss calculations.
        """
        # Safe exponent range to prevent float32 overflow/underflow
        # 10^38 is the upper limit, 10^-38 is the lower limit for float32
        SAFE_MAX_EXPONENT = 30.0
        SAFE_MIN_EXPONENT = -30.0

        if method == "log-standard":
            # Unstandardize then exp10
            mean = stats.get("log_mean", 0.0)
            std = max(stats.get("log_std", 1.0), self.min_std)
            log_values = values * std + mean
            # Clamp to prevent overflow/underflow before exponentiation
            log_values_safe = torch.clamp(log_values, min=SAFE_MIN_EXPONENT, max=SAFE_MAX_EXPONENT)
            return torch.pow(10.0, log_values_safe)

        elif method == "standard":
            # Direct unstandardization
            mean = stats.get("mean", 0.0)
            std = max(stats.get("std", 1.0), self.min_std)
            return values * std + mean

        elif method == "log-min-max":
            # Unscale then exp10
            log_min = stats.get("log_min", -30.0)
            log_max = stats.get("log_max", 0.0)
            log_values = values * (log_max - log_min) + log_min
            # Clamp to prevent overflow/underflow before exponentiation
            log_values_safe = torch.clamp(log_values, min=SAFE_MIN_EXPONENT, max=SAFE_MAX_EXPONENT)
            return torch.pow(10.0, log_values_safe)

        elif method == "min-max":
            # Direct unscaling
            min_val = stats.get("min", 0.0)
            max_val = stats.get("max", 1.0)
            return values * (max_val - min_val) + min_val

        elif method == "log10":
            # Updated: Add mean back then exp10 with overflow/underflow protection
            mean = stats.get("log_mean", 0.0)
            log_values = values + mean
            log_values_safe = torch.clamp(log_values, min=SAFE_MIN_EXPONENT, max=SAFE_MAX_EXPONENT)
            return torch.pow(10.0, log_values_safe)

        else:  # "none"
            return values

    def _denormalize_time(self, t_norm: torch.Tensor, method: str) -> torch.Tensor:
        """
        Denormalize time values with overflow/underflow protection.
        """
        # Safe exponent range to prevent float32 overflow/underflow
        SAFE_MAX_EXPONENT = 30.0
        SAFE_MIN_EXPONENT = -30.0

        if method == "time-norm" and hasattr(self, '_tau0'):
            # Inverse tau-space transformation
            t_norm_clamped = torch.clamp(t_norm, 0.0, 1.0)
            tau = t_norm_clamped * self._tau_range + self._tau_min
            # Clamp tau to prevent overflow/underflow in expm1
            tau_safe = torch.clamp(tau, min=SAFE_MIN_EXPONENT, max=SAFE_MAX_EXPONENT)
            return self._tau0 * torch.expm1(tau_safe)

        elif method == "log-min-max" and hasattr(self, '_tlog_min'):
            # Inverse log-space min-max
            t_norm_clamped = torch.clamp(t_norm, 0.0, 1.0)
            t_log = t_norm_clamped * self._tlog_range + self._tlog_min
            # Clamp to prevent overflow/underflow
            t_log_safe = torch.clamp(t_log, min=SAFE_MIN_EXPONENT, max=SAFE_MAX_EXPONENT)
            return torch.pow(10.0, t_log_safe)

        else:
            # Fall back to standard variable denormalization
            time_var = self.data_cfg.get("time_variable", "t_time")
            return self._denormalize_variable(t_norm, method, self.per_key_stats.get(time_var, {}))
