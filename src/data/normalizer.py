#!/usr/bin/env python3
"""
Flexible normalization module for chemical kinetics data.

Supports multiple normalization strategies:
- log-standard: log10(x) then standardize
- standard: standardize (subtract mean, divide by std)
- log10: just apply log10 transformation
- min-max: scale to [0,1] using min/max
- log-min-max: log10 then min-max scaling
- none: pass-through without modification

Special handling for time variables with configurable transformations.
"""

import logging
from typing import Dict, List, Any, Tuple
import torch


# Constants to avoid magic numbers
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP_VALUE = 50.0
MIN_DENOMINATOR = 1e-8
LOG_BASE = 10.0
DEFAULT_MIN_VALUE_THRESHOLD = 1e-30


class NormalizationHelper:
    """
    Applies pre-computed normalization statistics to torch tensors.
    
    Provides consistent normalization and denormalization across the pipeline,
    supporting various transformation methods configured per-variable.
    """
    
    def __init__(
        self,
        stats: Dict[str, Any],
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize the normalization helper.
        
        Args:
            stats: Pre-computed statistics from preprocessing
            device: Device for tensor operations
            config: Configuration with normalization settings
        """
        self.stats = stats
        self.device = device
        self.config = config
        self.norm_config = config.get("normalization", {})
        self.logger = logging.getLogger(__name__)
        
        # Set up data type
        dtype_name = config.get("system", {}).get("dtype", "float32")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32
        
        # Process normalization methods
        raw_methods = self.stats.get("normalization_methods", {})
        self.methods = {
            k: ("none" if v is None else str(v).lower())
            for k, v in raw_methods.items()
        }
        
        # Core statistics
        self.per_key_stats = self.stats.get("per_key_stats", {})

        self.time_norm = self.stats.get("time_normalization", None)
        self.time_var = config.get("data", {}).get("time_variable", "t_time")

        if self.time_norm and "time_transform" in self.time_norm:
            self.time_method = self.time_norm["time_transform"]
        else:
            # Fallback to getting it from the saved methods like other variables
            self.time_method = self.methods.get(self.time_var, "none")
        
        # Variables already in log space from preprocessing
        self._already_logged_vars = set(self.stats.get("already_logged_vars", []))
        
        # Validate configuration
        self._validate_config()
        
        # Set up constants
        self._setup_constants()
        
        # Set up time normalization parameters
        self._setup_time_normalization()
    
    def _validate_config(self) -> None:
        """Validate normalization configuration against available statistics."""
        # Check for double-log transformation or invalid linear-space methods
        for var in self._already_logged_vars:
            method = self.methods.get(var, "none")
            # These methods assume linear-space data
            invalid_methods = {"standard", "min-max"}
            if method in invalid_methods:
                raise ValueError(f"Cannot apply '{method}' to '{var}' - already in log10 space.")
        
        # Verify required statistics exist
        for var, method in self.methods.items():
            if var == self.time_var:
                self._validate_time_stats()
                continue
            
            required_stats = self._get_required_stats(method, var)
            missing = [k for k in required_stats if k not in self.per_key_stats.get(var, {})]
            if missing:
                raise ValueError(f"Missing statistics for '{var}': {missing}")
    
    def _validate_time_stats(self) -> None:
        """Validate time normalization statistics."""
        # Use the actual method from stats, not from config
        actual_time_method = self.time_norm.get("time_transform") if self.time_norm else None
        if actual_time_method == "time-norm":
            required = ["tau0", "tmin", "tmax"]
            if not self.time_norm or not all(k in self.time_norm for k in required):
                raise ValueError(f"Time stats missing: {required}")
        elif self.time_method == "log-min-max":
            required = ["tmin_raw", "tmax_raw"]
            if not self.time_norm or not all(k in self.time_norm for k in required):
                raise ValueError(f"Time stats missing: {required}")
        elif self.time_method not in ("none", None):
            raise ValueError(f"Unsupported time normalization: {self.time_method}")
    
    def _get_required_stats(self, method: str, var: str) -> List[str]:
        """Get required statistics keys for a normalization method."""
        if method == "log-standard":
            return ["log_mean", "log_std"]
        elif method == "standard":
            return ["mean", "std"]
        elif method == "min-max":
            if var in self._already_logged_vars:
                raise ValueError(f"Cannot use 'min-max' on log-space variable '{var}'")
            return ["min", "max"]
        elif method == "log-min-max":
            return ["log_min", "log_max"]
        elif method in ("log10", "none"):
            return []
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _setup_constants(self) -> None:
        """Initialize constant tensors used in normalization."""
        # Match preprocessor's floor logic
        min_threshold = float(self.config.get("preprocessing", {}).get(
            "min_value_threshold", DEFAULT_MIN_VALUE_THRESHOLD
        ))
        epsilon = float(self.norm_config.get("epsilon", DEFAULT_EPSILON))
        self.log_floor = max(min_threshold, epsilon)
        
        self.epsilon = torch.tensor(
            self.norm_config.get("epsilon", DEFAULT_EPSILON),
            dtype=self.dtype,
            device=self.device
        )
        self.clamp_val = float(self.norm_config.get("clamp_value", DEFAULT_CLAMP_VALUE))
        self.min_std = float(self.norm_config.get("min_std", DEFAULT_MIN_STD))
        self._ten = torch.tensor(LOG_BASE, dtype=self.dtype, device=self.device)
    
    def _setup_time_normalization(self) -> None:
        """Initialize time normalization parameters."""
        if not self.time_norm:
            return

        if self.time_method == "time-norm":
            # Tau-space parameters: tau = log1p(t / tau0)
            self._tau0 = torch.tensor(
                float(self.time_norm["tau0"]),
                dtype=self.dtype,
                device=self.device,
            )
            self._tau_min = torch.tensor(
                float(self.time_norm["tmin"]),
                dtype=self.dtype,
                device=self.device,
            )
            self._tau_max = torch.tensor(
                float(self.time_norm["tmax"]),
                dtype=self.dtype,
                device=self.device,
            )
            self._tau_range = torch.clamp(
                self._tau_max - self._tau_min, 
                min=MIN_DENOMINATOR
            )

        elif self.time_method == "log-min-max":
            # Log-space min-max normalization
            eps = float(self.norm_config.get("epsilon", DEFAULT_EPSILON))
            lo = max(float(self.time_norm["tmin_raw"]), eps)
            hi = max(float(self.time_norm["tmax_raw"]), lo + eps)
            self._tlog_min = torch.log10(torch.tensor(lo, dtype=self.dtype, device=self.device))
            self._tlog_max = torch.log10(torch.tensor(hi, dtype=self.dtype, device=self.device))
            self._tlog_range = torch.clamp(
                self._tlog_max - self._tlog_min, 
                min=MIN_DENOMINATOR
            )
    
    def _get_params(self, var_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Build parameter tensors aligned to variable list."""
        means, stds, methods = [], [], []
        
        for var in var_list:
            method = self.methods.get(var, "none")
            stats = self.per_key_stats.get(var, {})
            
            if method == "log-standard":
                mean = stats.get("log_mean", 0.0)
                std = stats.get("log_std", 1.0)
            elif method == "standard":
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
            else:
                mean = 0.0
                std = 1.0
            
            means.append(float(mean))
            stds.append(max(float(std), self.min_std))
            methods.append(method)
        
        means_t = torch.tensor(means, dtype=self.dtype, device=self.device)
        stds_t = torch.tensor(stds, dtype=self.dtype, device=self.device)
        
        return means_t, stds_t, methods
    
    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize time values to [0,1] using configured method."""
        if self.time_method in ("none", None):
            return t
        
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found")
        
        # Move to device FIRST before any operations
        t = t.to(device=self.device, dtype=self.dtype)
        
        if self.time_method == "time-norm":
            # Tau-space normalization: tau = log1p(t / tau0)
            # Ensure tau0 is on the same device
            if not hasattr(self, '_tau0'):
                self._setup_time_normalization()
            
            tau = torch.log1p(t / self._tau0)
            normalized = (tau - self._tau_min) / self._tau_range
            
            # Clamp to valid range
            return torch.clamp(normalized, 0.0, 1.0)
        
        elif self.time_method == "log-min-max":
            # Log-space min-max normalization
            # Use epsilon as a scalar value, not tensor, to avoid device issues
            epsilon_val = self.epsilon.item() if isinstance(self.epsilon, torch.Tensor) else self.epsilon
            
            # Clamp time to avoid log(0)
            t_clamped = torch.clamp(t, min=epsilon_val)
            
            # Apply log transformation
            tlog = torch.log10(t_clamped)
            
            # Normalize to [0,1]
            if not hasattr(self, '_tlog_min'):
                self._setup_time_normalization()
                
            normalized = (tlog - self._tlog_min) / self._tlog_range
            
            # Clamp to valid range
            return torch.clamp(normalized, 0.0, 1.0)
        
        elif self.time_method == "standard":
            # Standard normalization for time (if configured)
            if self.time_var in self.per_key_stats and "mean" in self.per_key_stats[self.time_var]:
                mean = torch.tensor(
                    self.per_key_stats[self.time_var]["mean"], 
                    dtype=self.dtype, 
                    device=self.device
                )
                std = torch.tensor(
                    max(self.per_key_stats[self.time_var]["std"], self.min_std),
                    dtype=self.dtype, 
                    device=self.device
                )
                return (t - mean) / std
        else:
            raise ValueError(f"Unknown time normalization method: {self.time_method}")
    
    def denormalize_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Invert time normalization back to raw values."""
        if self.time_method in ("none", None):
            return t_norm
        
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found")
        
        t_norm = t_norm.to(device=self.device, dtype=self.dtype)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
        
        if self.time_method == "time-norm":
            tau = t_norm * self._tau_range + self._tau_min
            return self._tau0 * torch.expm1(tau)
        
        if self.time_method == "log-min-max":
            tlog = t_norm * self._tlog_range + self._tlog_min
            return torch.pow(self._ten, tlog)
        
        raise ValueError(f"Unknown time normalization: {self.time_method}")
    
    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Apply normalization to data tensor.
        
        Args:
            data: Input tensor of shape (..., D) where D = len(var_list)
            var_list: List of variable names for each dimension
            
        Returns:
            Normalized tensor with same shape, clamped to [-clamp_val, clamp_val]
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(
                f"Data dimension {data.shape[-1]} != variable count {len(var_list)}"
            )
        
        # Ensure correct device and dtype
        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)
        
        means, stds, methods = self._get_params(var_list)
        out = data.clone()
        
        for i, (method, varname) in enumerate(zip(methods, var_list)):
            col = out[..., i]
            
            if varname == self.time_var and method in ("time-norm", "log-min-max"):
                # Special time handling
                col = self.normalize_time(col)
                col = torch.clamp(col, -self.clamp_val, self.clamp_val)
            elif method == "log-standard":
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.log_floor))
                col = (col - means[i]) / stds[i]
            elif method == "standard":
                col = (col - means[i]) / stds[i]
            elif method == "log10":
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.log_floor))
            elif method == "min-max":
                vmin = torch.tensor(
                    self.per_key_stats[varname]["min"],
                    dtype=self.dtype, device=self.device
                )
                vmax = torch.tensor(
                    self.per_key_stats[varname]["max"],
                    dtype=self.dtype, device=self.device
                )
                denom = torch.clamp(vmax - vmin, min=MIN_DENOMINATOR)
                col = torch.clamp((col - vmin) / denom, 0.0, 1.0)
            elif method == "log-min-max":
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.log_floor))
                lmin = torch.tensor(
                    self.per_key_stats[varname]["log_min"],
                    dtype=self.dtype, device=self.device
                )
                lmax = torch.tensor(
                    self.per_key_stats[varname]["log_max"],
                    dtype=self.dtype, device=self.device
                )
                denom = torch.clamp(lmax - lmin, min=MIN_DENOMINATOR)
                col = torch.clamp((col - lmin) / denom, 0.0, 1.0)
            
            out[..., i] = col
        
        return torch.clamp(out, -self.clamp_val, self.clamp_val)
    
    def denormalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Invert normalization to recover raw values.
        
        Args:
            data: Normalized tensor of shape (..., D)
            var_list: List of variable names for each dimension
            
        Returns:
            Denormalized tensor in raw value space
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(
                f"Data dimension {data.shape[-1]} != variable count {len(var_list)}"
            )
        
        # Ensure correct device and dtype
        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)
        
        means, stds, methods = self._get_params(var_list)
        out = data.clone()
        
        for i, (method, varname) in enumerate(zip(methods, var_list)):
            col = out[..., i]
            
            if varname == self.time_var and method in ("time-norm", "log-min-max"):
                col = self.denormalize_time(torch.clamp(col, 0.0, 1.0))
            elif method == "log-standard":
                col = torch.pow(self._ten, col * stds[i] + means[i])
            elif method == "standard":
                col = col * stds[i] + means[i]
            elif method == "log10":
                col = torch.pow(self._ten, col)
            elif method == "min-max":
                vmin = torch.tensor(
                    self.per_key_stats[varname]["min"],
                    dtype=self.dtype, device=self.device
                )
                vmax = torch.tensor(
                    self.per_key_stats[varname]["max"],
                    dtype=self.dtype, device=self.device
                )
                col = torch.clamp(col, 0.0, 1.0) * (vmax - vmin) + vmin
            elif method == "log-min-max":
                lmin = torch.tensor(
                    self.per_key_stats[varname]["log_min"],
                    dtype=self.dtype, device=self.device
                )
                lmax = torch.tensor(
                    self.per_key_stats[varname]["log_max"],
                    dtype=self.dtype, device=self.device
                )
                logx = torch.clamp(col, 0.0, 1.0) * (lmax - lmin) + lmin
                col = torch.pow(self._ten, logx)
            
            out[..., i] = col
        
        return out