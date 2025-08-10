#!/usr/bin/env python3
"""
Flexible normalization module for chemical kinetics data.

Supports multiple normalization strategies:
- log-standard: log10(x) then standardize
- standard: standardize (subtract mean, divide by std)
- log10: just apply log10 transformation
- time-norm: time-specific normalization (tau = ln(1 + t/tau0))
- min-max: scale to [0,1] using min/max
- log-min-max: log10 then min-max scaling
- none: pass-through without modification
"""

import logging
from typing import Dict, List, Any, Tuple
import torch


class NormalizationHelper:
    """
    Applies pre-computed normalization statistics to torch tensors.
    
    This helper provides consistent normalization and denormalization
    across the pipeline, supporting various transformation methods
    configured per-variable.
    
    Attributes:
        stats: Pre-computed normalization statistics
        device: Device for tensor operations
        config: Configuration dictionary
        dtype: Data type for tensors
        methods: Normalization method per variable
        per_key_stats: Statistics per variable
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
        self.time_var = config.get("data", {}).get("time_variable", "t")
        self.time_method = config.get("normalization", {}).get("methods", {}).get(
            self.time_var, "log-min-max"
        )
        
        # Identify variables already in log space from preprocessing
        self._already_logged_vars = set(self.stats.get("already_logged_vars", []))
        
        # Fallback for older shards
        if not self._already_logged_vars:
            data_cfg = config.get("data", {})
            if data_cfg.get("sequence_mode", False):
                self._already_logged_vars = set(data_cfg.get("species_variables", []))
        
        if self._already_logged_vars:
            self.logger.info(
                "NormalizationHelper: %d variables already in log10 space from shards.",
                len(self._already_logged_vars)
            )
        
        # Validate configuration
        self._validate_config()
        
        # Set up constants
        self._setup_constants()
        
        # Set up time normalization parameters
        self._setup_time_normalization()
    
    def _validate_config(self) -> None:
        """Validate normalization configuration against available statistics."""
        self._validate_no_double_log()
        self._validate_stats_for_methods()
    
    def _validate_no_double_log(self) -> None:
        """
        Ensure no double-log transformation for already-logged variables.
        
        Variables already log10-transformed in shards cannot use plain 'standard'
        normalization (which expects linear values).
        """
        for var in self._already_logged_vars:
            method = self.methods.get(var, "none")
            if method == "standard":
                raise ValueError(
                    f"Config requests 'standard' for '{var}', but shards store it in log10 space. "
                    f"Use 'log-standard' or change preprocessing."
                )
    
    def _validate_stats_for_methods(self) -> None:
        """Verify required statistics exist for each normalization method."""
        for var, method in self.methods.items():
            # Time variable uses special handling
            if var == self.time_var:
                if method in ("time-norm", "log-min-max"):
                    if not self.time_norm:
                        raise ValueError("Time normalization stats missing")
                    continue
                if method in ("none", None):
                    # pass-through time; no stats required
                    continue
                raise ValueError(f"Unsupported time normalization method for '{self.time_var}': {method}")

            
            # Check required stats for each method
            if method == "log-standard":
                self._require_keys(var, ["log_mean", "log_std"])
            elif method == "standard":
                self._require_keys(var, ["mean", "std"])
            elif method == "min-max":
                self._require_keys(var, ["min", "max"])
                if var in self._already_logged_vars:
                    raise ValueError(
                        f"'min-max' requested for '{var}', but shards are log10. Use 'log-min-max'."
                    )
            elif method == "log-min-max":
                self._require_keys(var, ["log_min", "log_max"])
            elif method == "log10":
                # log10 doesn't need stats, just applies transformation
                pass
            elif method in ("none",):
                pass
            else:
                raise ValueError(f"Unknown normalization method for '{var}': {method}")
    
    def _require_keys(self, var: str, needed: List[str]) -> None:
        """Verify required statistics keys exist for a variable."""
        stats = self.per_key_stats.get(var, {})
        missing = [k for k in needed if k not in stats]
        if missing:
            raise ValueError(
                f"Normalization stats missing for '{var}': need {missing} in per_key_stats[{var}]"
            )
    
    def _setup_constants(self) -> None:
        """Initialize constant tensors used in normalization."""
        self.epsilon = torch.tensor(
            self.norm_config.get("epsilon", 1e-30),
            dtype=self.dtype,
            device=self.device
        )
        self.clamp_val = float(self.norm_config.get("clamp_value", 50.0))
        self.min_std = float(self.norm_config.get("min_std", 1e-10))
        self._ten = torch.tensor(10.0, dtype=self.dtype, device=self.device)
    
    def _setup_time_normalization(self) -> None:
        """Initialize time normalization parameters if available."""
        if not self.time_norm:
            return
        
        # Paper's tau-space parameters
        self._tau0 = torch.tensor(
            float(self.time_norm["tau0"]),
            dtype=self.dtype,
            device=self.device
        )
        self._tau_min = torch.tensor(
            float(self.time_norm["tmin"]),
            dtype=self.dtype,
            device=self.device
        )
        self._tau_max = torch.tensor(
            float(self.time_norm["tmax"]),
            dtype=self.dtype,
            device=self.device
        )
        self._tau_range = torch.clamp(self._tau_max - self._tau_min, min=1e-12)
        
        # Log-min-max parameters (raw time space)
        tmin_raw = float(self.time_norm.get("tmin_raw", 0.0))
        tmax_raw = float(self.time_norm.get("tmax_raw", 1.0))
        eps = float(self.norm_config.get("epsilon", 1e-30))
        
        lo = max(tmin_raw, eps)
        hi = max(tmax_raw, lo + eps)
        
        self._tlog_min = torch.log10(torch.tensor(lo, dtype=self.dtype, device=self.device))
        self._tlog_max = torch.log10(torch.tensor(hi, dtype=self.dtype, device=self.device))
        self._tlog_range = torch.clamp(self._tlog_max - self._tlog_min, min=1e-12)
    
    def _get_params(self, var_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Build parameter tensors aligned to variable list.
        
        Args:
            var_list: List of variable names
            
        Returns:
            Tuple of (means, stds, methods) tensors/lists
        """
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
    
    def _time_to_unit(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw time to [0,1] using configured method.
        
        Args:
            t: Raw time values
            
        Returns:
            Normalized time in [0,1]
        """
        if self.time_method in ("none", None):
            return t  # pass-through

        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")

        if self.time_method == "time-norm":
            tau = torch.log1p(t / self._tau0)
            return (tau - self._tau_min) / self._tau_range

        if self.time_method == "log-min-max":
            tlog = torch.log10(torch.clamp(t, min=self.epsilon))
            return (tlog - self._tlog_min) / self._tlog_range

        raise ValueError(f"Unknown time normalization method: {self.time_method}")
    
    def _unit_to_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Invert time normalization back to raw values.
        
        Args:
            t_norm: Normalized time in [0,1]
            
        Returns:
            Raw time values
        """
        if self.time_method in ("none", None):
            return t_norm  # pass-through

        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")

        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        if self.time_method == "time-norm":
            tau = t_norm * self._tau_range + self._tau_min
            return self._tau0 * torch.expm1(tau)

        if self.time_method == "log-min-max":
            tlog = t_norm * self._tlog_range + self._tlog_min
            return torch.pow(self._ten, tlog)

        raise ValueError(f"Unknown time normalization method: {self.time_method}")
    
    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Apply normalization to data tensor.
        
        Transforms each dimension according to its configured method:
        - log-standard: log10 then standardize
        - standard: standardize
        - log10: just log10 transformation
        - time-norm: time-specific transform
        - min-max: scale to [0,1]
        - log-min-max: log10 then min-max
        - none: pass-through
        
        Args:
            data: Input tensor of shape (..., D) where D = len(var_list)
            var_list: List of variable names for each dimension
            
        Returns:
            Normalized tensor with same shape, clamped to [-clamp_val, clamp_val]
            
        Raises:
            ValueError: If data dimensions don't match var_list
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data last dim {data.shape[-1]} != var_list length {len(var_list)}")
        
        # Ensure correct device and dtype
        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)
        
        means, stds, methods = self._get_params(var_list)
        out = data.clone()
        
        for i, (method, varname) in enumerate(zip(methods, var_list)):
            col = out[..., i]
            
            if varname == self.time_var and method in ("time-norm", "log-min-max"):
                # Special handling for time variables
                col = self._time_to_unit(col)
                col = torch.clamp(col, 0.0, 1.0)
                
            elif method == "log-standard":
                # Log10 then standardize
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.epsilon))
                col = (col - means[i]) / stds[i]
                
            elif method == "standard":
                # Plain standardization (check for double-log)
                if varname in self._already_logged_vars:
                    raise ValueError(
                        f"'standard' requested for '{varname}', but shards are log10. "
                        f"Use 'log-standard'."
                    )
                col = (col - means[i]) / stds[i]
                
            elif method == "log10":
                # Just log10 transformation
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.epsilon))
                # If already logged, it's a no-op
                
            elif method == "min-max":
                # Linear min-max scaling
                vmin = torch.as_tensor(
                    self.per_key_stats[varname]["min"],
                    dtype=self.dtype,
                    device=self.device
                )
                vmax = torch.as_tensor(
                    self.per_key_stats[varname]["max"],
                    dtype=self.dtype,
                    device=self.device
                )
                denom = torch.clamp(vmax - vmin, min=1e-12)
                
                if varname in self._already_logged_vars:
                    raise ValueError(
                        f"'min-max' on '{varname}' invalid: shards are log10. Use 'log-min-max'."
                    )
                col = torch.clamp((col - vmin) / denom, 0.0, 1.0)
                
            elif method == "log-min-max":
                # Log10 then min-max scaling
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.epsilon))
                    
                lmin = torch.as_tensor(
                    self.per_key_stats[varname]["log_min"],
                    dtype=self.dtype,
                    device=self.device
                )
                lmax = torch.as_tensor(
                    self.per_key_stats[varname]["log_max"],
                    dtype=self.dtype,
                    device=self.device
                )
                denom = torch.clamp(lmax - lmin, min=1e-12)
                col = torch.clamp((col - lmin) / denom, 0.0, 1.0)
            
            # else: method == "none" or unknown, pass through
            
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
            
        Raises:
            ValueError: If data dimensions don't match var_list
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data last dim {data.shape[-1]} != var_list length {len(var_list)}")
        
        # Ensure correct device and dtype
        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)
        
        means, stds, methods = self._get_params(var_list)
        out = data.clone()
        
        for i, (method, varname) in enumerate(zip(methods, var_list)):
            col = out[..., i]
            
            if varname == self.time_var and method in ("time-norm", "log-min-max"):
                # Invert time normalization
                col = self._unit_to_time(torch.clamp(col, 0.0, 1.0))
                
            elif method == "log-standard":
                # Invert standardization then exp10
                col = torch.pow(self._ten, col * stds[i] + means[i])
                
            elif method == "standard":
                # Invert standardization
                col = col * stds[i] + means[i]
                
            elif method == "log10":
                # Invert log10
                col = torch.pow(self._ten, col)
                
            elif method == "min-max":
                # Invert min-max scaling
                vmin = torch.as_tensor(
                    self.per_key_stats[varname]["min"],
                    dtype=self.dtype,
                    device=self.device
                )
                vmax = torch.as_tensor(
                    self.per_key_stats[varname]["max"],
                    dtype=self.dtype,
                    device=self.device
                )
                col = torch.clamp(col, 0.0, 1.0) * (vmax - vmin) + vmin
                
            elif method == "log-min-max":
                # Invert min-max then exp10
                lmin = torch.as_tensor(
                    self.per_key_stats[varname]["log_min"],
                    dtype=self.dtype,
                    device=self.device
                )
                lmax = torch.as_tensor(
                    self.per_key_stats[varname]["log_max"],
                    dtype=self.dtype,
                    device=self.device
                )
                logx = torch.clamp(col, 0.0, 1.0) * (lmax - lmin) + lmin
                col = torch.pow(self._ten, logx)
            
            # else: method == "none" or unknown, pass through
            
            out[..., i] = col
        
        return out
    
    # Convenience methods for time normalization
    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalize time values to [0,1].
        
        Args:
            t: Raw time values
            
        Returns:
            Normalized time tensor
        """
        return self._time_to_unit(t.to(device=self.device, dtype=self.dtype))
    
    def denormalize_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize time values from [0,1] to raw values.
        
        Args:
            t_norm: Normalized time values
            
        Returns:
            Raw time tensor
        """
        return self._unit_to_time(t_norm.to(device=self.device, dtype=self.dtype))