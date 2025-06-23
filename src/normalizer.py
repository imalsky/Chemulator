#!/usr/bin/env python3
"""
normalizer.py - Compute and invert various normalization schemes using PyTorch.

This module calculates global statistics from training profiles in an HDF5 file.
All computations use PyTorch tensors with float32 precision for consistency
and GPU compatibility.
"""
from __future__ import annotations

import h5py
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# --- Constants ---
DTYPE = torch.float32
DEFAULT_EPSILON = 1e-9
DEFAULT_QUANTILE_MEMORY_LIMIT = 5_000_000
DEFAULT_SYMLOG_PERCENTILE = 0.5

# Type alias for Welford state
WelfordState = Tuple[int, Tensor, Tensor]  # (count, mean, M2)


def welford_update(state: WelfordState, new_value: Tensor) -> WelfordState:
    """Update Welford's algorithm state with a new value (PyTorch version)."""
    count, mean, m2 = state
    count += 1
    delta = new_value - mean
    mean = mean + delta / count
    delta2 = new_value - mean
    m2 = m2 + delta * delta2
    return count, mean, m2


def welford_finalize(state: WelfordState, eps: float = DEFAULT_EPSILON) -> Tuple[float, float]:
    """Finalize Welford's algorithm to get mean and standard deviation."""
    count, mean, m2 = state
    if count < 2:
        return mean.item(), 1.0
    variance = m2 / (count - 1)
    std = torch.sqrt(variance).clamp(min=eps)
    return mean.item(), std.item()


class DataNormalizer:
    """
    Handles data normalization by calculating statistics from training data
    and providing methods to apply or invert normalization using PyTorch.
    """
    
    METHODS = {
        "standard", "log-standard", "log-min-max", "iqr", "max-out",
        "scaled_signed_offset_log", "symlog", "bool", "none"
    }
    QUANTILE_METHODS = {"iqr", "symlog", "log-min-max"}

    def __init__(self, *, config_data: Dict[str, Any], device: torch.device = None):
        self.config = config_data
        self.device = device or torch.device('cpu')
        self.data_spec = self.config.get("data_specification", {})
        self.norm_cfg = self.config.get("normalization", {})
        
        self.eps = float(self.norm_cfg.get("epsilon", DEFAULT_EPSILON))
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """Parse config to determine which keys to process and their methods."""
        all_vars = set(self.data_spec.get("all_variables", []))
        user_key_methods = self.norm_cfg.get("key_methods", {})
        default_method = self.norm_cfg.get("default_method", "standard")

        key_methods = {
            key: user_key_methods.get(key, default_method).lower() 
            for key in all_vars
        }

        # Validate methods
        for key, method in key_methods.items():
            if method not in self.METHODS:
                raise ValueError(f"Unsupported method '{method}' for key '{key}'.")

        return all_vars, key_methods

    def calculate_stats(self, h5_path: Path, train_indices: List[int]) -> Dict[str, Any]:
        """Calculate global statistics for all variables from HDF5 training data."""
        logger.info(f"Calculating stats on {len(train_indices)} profiles from {h5_path.name}...")
        
        if not train_indices:
            raise ValueError("Cannot calculate statistics from empty training indices.")

        accumulators = self._initialize_accumulators()

        with h5py.File(h5_path, 'r') as hf:
            available_keys = set(hf.keys())
            
            for key in self.keys_to_process:
                if key not in available_keys:
                    logger.warning(f"Key '{key}' not found in HDF5 file, skipping.")
                    continue
                
                # Read data and convert to float32 tensor
                data_slice_np = hf[key][train_indices]
                data_slice = torch.from_numpy(data_slice_np).to(DTYPE).to(self.device)
                
                self._update_stats_for_key(key, data_slice, accumulators)

        computed_stats = self._finalize_stats(accumulators)
        metadata = {
            "normalization_methods": self.key_methods,
            "per_key_stats": computed_stats
        }
        
        logger.info("Global statistics calculation complete.")
        return metadata

    def _initialize_accumulators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data structures for statistics accumulation."""
        accumulators = {}
        
        for key, method in self.key_methods.items():
            if method in ("none", "bool"):
                continue
                
            key_stats: Dict[str, Any] = {
                "min": torch.tensor(float('inf'), dtype=DTYPE, device=self.device),
                "max": torch.tensor(float('-inf'), dtype=DTYPE, device=self.device),
            }
            
            if method in ("standard", "log-standard"):
                # Initialize Welford state with tensors
                key_stats["welford"] = (
                    0,  # count
                    torch.tensor(0.0, dtype=DTYPE, device=self.device),  # mean
                    torch.tensor(0.0, dtype=DTYPE, device=self.device)   # M2
                )
            
            if method in self.QUANTILE_METHODS:
                key_stats["values"] = []
                
            accumulators[key] = key_stats
            
        return accumulators

    def _update_stats_for_key(self, key: str, data_slice: Tensor, accumulators: Dict) -> None:
        """Update accumulators for a single key using a PyTorch tensor."""
        if key not in accumulators:
            return
            
        method = self.key_methods[key]
        key_acc = accumulators[key]

        # Filter out non-finite values
        valid_mask = torch.isfinite(data_slice)
        valid_data = data_slice[valid_mask]

        # Check for log-based methods
        if method.startswith("log-") and torch.any(valid_data <= 0):
            raise ValueError(
                f"Variable '{key}' has non-positive values but requires "
                f"log-based normalization ('{method}'). Check data or config."
            )

        if valid_data.numel() > 0:
            # Update min/max using PyTorch operations
            key_acc["min"] = torch.min(key_acc["min"], valid_data.min())
            key_acc["max"] = torch.max(key_acc["max"], valid_data.max())

        # Method-specific updates
        if method == "standard":
            for v in valid_data.flatten():
                key_acc["welford"] = welford_update(key_acc["welford"], v)
                
        elif method == "log-standard":
            log_data = torch.log10(valid_data)
            for v in log_data.flatten():
                key_acc["welford"] = welford_update(key_acc["welford"], v)
                
        elif method in self.QUANTILE_METHODS:
            # Store as tensor for later processing
            key_acc["values"].append(valid_data.flatten())

    def _finalize_stats(self, accumulators: Dict) -> Dict[str, Any]:
        """Finalize statistics from accumulators after the full data pass."""
        final_stats = {}
        
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {
                "method": method,
                "epsilon": self.eps
            }
            
            if key not in accumulators or method in ("none", "bool"):
                stats["method"] = "none"
                final_stats[key] = stats
                continue

            key_acc = accumulators[key]
            
            # Convert min/max to Python floats
            if "min" in key_acc:
                min_val = key_acc["min"].item()
                max_val = key_acc["max"].item()
            
            if method == "standard":
                mean, std = welford_finalize(key_acc["welford"], self.eps)
                stats.update({"mean": mean, "std": std})
                
            elif method == "log-standard":
                log_mean, log_std = welford_finalize(key_acc["welford"], self.eps)
                stats.update({"log_mean": log_mean, "log_std": log_std})
                
            elif method == "max-out":
                stats["max_val"] = max(abs(min_val), abs(max_val), self.eps)
                
            elif method == "scaled_signed_offset_log":
                m_pos = torch.log10(torch.tensor(max(0, max_val) + 1 + self.eps, dtype=DTYPE)).item()
                m_neg = torch.log10(torch.tensor(max(0, -min_val) + 1 + self.eps, dtype=DTYPE)).item()
                stats.update({"m": max(m_pos, m_neg, 1.0)})
                
            elif method in self.QUANTILE_METHODS:
                stats.update(self._finalize_quantile_stats(key, key_acc, method))
                
            final_stats[key] = stats
            
        return final_stats
    
    def _finalize_quantile_stats(self, key: str, key_acc: dict, method: str) -> dict:
        """Compute stats for methods requiring quantiles (all data needed)."""
        values_list = key_acc.get("values", [])
        if not values_list:
            # Return safe defaults
            return {
                "median": 0.0, "iqr": 1.0, "min": 0.0, "max": 1.0,
                "threshold": 1.0, "scale_factor": 1.0
            }

        # Concatenate all values into single tensor
        values = torch.cat(values_list, dim=0)
        
        # Memory safety check
        max_values = self.norm_cfg.get("quantile_max_values_in_memory", DEFAULT_QUANTILE_MEMORY_LIMIT)
        if values.numel() > max_values:
            logger.warning(
                f"Key '{key}' has {values.numel():,} values, exceeding limit. "
                f"Using random sample of {max_values:,}."
            )
            perm = torch.randperm(values.numel(), device=values.device)
            values = values[perm[:max_values]]
        
        stats: Dict[str, Any] = {}
        
        if method == "iqr":
            quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE, device=values.device)
            q_vals = torch.quantile(values, quantiles)
            q1, med, q3 = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
            stats.update({
                "median": med,
                "iqr": max(q3 - q1, self.eps)
            })
            
        elif method == "log-min-max":
            log_vals = torch.log10(values)  # Values already validated > 0
            min_v = log_vals.min().item()
            max_v = log_vals.max().item()
            stats.update({
                "min": min_v,
                "max": max(max_v, min_v + self.eps)
            })
            
        elif method == "symlog":
            percentile = self.norm_cfg.get("symlog_percentile", DEFAULT_SYMLOG_PERCENTILE)
            thr = torch.quantile(torch.abs(values), percentile).item()
            thr = max(thr, self.eps)
            
            abs_v = torch.abs(values)
            mask = abs_v > thr
            transformed = torch.zeros_like(values)
            transformed[mask] = torch.sign(values[mask]) * (torch.log10(abs_v[mask] / thr) + 1)
            transformed[~mask] = values[~mask] / thr
            
            sf = transformed.abs().max().item() if transformed.numel() > 0 else 1.0
            stats.update({
                "threshold": thr,
                "scale_factor": max(sf, 1.0)
            })
            
        return stats

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """Apply normalization to a PyTorch tensor (always float32)."""
        # Ensure input is float32
        x = x.to(DTYPE)
        eps = stats.get("epsilon", DEFAULT_EPSILON)
        
        if method in ("none", "bool"):
            return x
        
        if method == "standard":
            return (x - stats["mean"]) / stats["std"]
            
        elif method == "log-standard":
            x_safe = torch.clamp(x, min=eps)
            return (torch.log10(x_safe) - stats["log_mean"]) / stats["log_std"]
            
        elif method == "log-min-max":
            x_safe = torch.clamp(x, min=eps)
            log_x = torch.log10(x_safe)
            denom = stats["max"] - stats["min"]
            return torch.clamp((log_x - stats["min"]) / max(denom, eps), 0.0, 1.0)
            
        elif method == "max-out":
            return x / stats["max_val"]
            
        elif method == "iqr":
            return (x - stats["median"]) / stats["iqr"]
            
        elif method == "scaled_signed_offset_log":
            y = torch.sign(x) * torch.log10(torch.abs(x) + 1 + eps)
            return y / stats["m"]
            
        elif method == "symlog":
            thr = stats["threshold"]
            sf = stats["scale_factor"]
            abs_x = torch.abs(x)
            mask = abs_x > thr
            y = torch.zeros_like(x)
            y[mask] = torch.sign(x[mask]) * (torch.log10(abs_x[mask] / thr) + 1)
            y[~mask] = x[~mask] / thr
            return torch.clamp(y / sf, -1.0, 1.0)
            
        else:
            raise ValueError(f"Unsupported normalization method '{method}'")

    @staticmethod
    def denormalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """Inverse normalization for a PyTorch tensor (always float32)."""
        # Ensure input is float32
        x = x.to(DTYPE)
        
        if method in ("none", "bool"):
            return x
            
        if not stats:
            raise ValueError(f"No stats for denormalization method '{method}'")
        
        eps = stats.get("epsilon", DEFAULT_EPSILON)
        
        if method == "standard":
            return x * stats["std"] + stats["mean"]
            
        elif method == "log-standard":
            return torch.pow(10, x * stats["log_std"] + stats["log_mean"])
            
        elif method == "log-min-max":
            x_clamped = torch.clamp(x, 0, 1)
            log_val = x_clamped * (stats["max"] - stats["min"]) + stats["min"]
            return torch.pow(10, log_val)
            
        elif method == "max-out":
            return x * stats["max_val"]
            
        elif method == "iqr":
            return x * stats["iqr"] + stats["median"]
            
        elif method == "scaled_signed_offset_log":
            ytmp = x * stats["m"]
            abs_y = torch.abs(ytmp)
            return torch.sign(ytmp) * (torch.pow(10, abs_y) - 1 - eps)
            
        elif method == "symlog":
            unscaled = x * stats["scale_factor"]
            abs_unscaled = torch.abs(unscaled)
            mask = abs_unscaled > 1.0
            y = torch.zeros_like(x)
            y[mask] = torch.sign(unscaled[mask]) * stats["threshold"] * torch.pow(10, abs_unscaled[mask] - 1)
            y[~mask] = unscaled[~mask] * stats["threshold"]
            return y
            
        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")

    @staticmethod
    def denormalize(
        v: Union[Tensor, List, float, bool, None],
        metadata: Dict[str, Any],
        var_name: str
    ) -> Union[Tensor, List, float, bool, None]:
        """Convenience wrapper for denormalizing various input types."""
        if v is None:
            return None
            
        method = metadata["normalization_methods"].get(var_name, "none")
        if method in ("none", "bool"):
            return v
        
        stats = metadata["per_key_stats"].get(var_name)
        if not stats:
            raise ValueError(f"No stats for '{var_name}' in metadata")
        
        # Remember input type
        is_scalar = not isinstance(v, (torch.Tensor, list))
        is_list = isinstance(v, list)
        
        # Convert to float32 tensor
        if isinstance(v, torch.Tensor):
            original_shape = v.shape
            tensor_v = v.clone().detach().to(DTYPE)
        else:
            original_shape = torch.as_tensor(v).shape
            tensor_v = torch.as_tensor(v, dtype=DTYPE)

        # Denormalize
        denorm_tensor = DataNormalizer.denormalize_tensor(
            tensor_v.flatten(), method, stats
        )
        
        # Restore original type and shape
        if is_scalar:
            return denorm_tensor.item()
        elif is_list:
            return denorm_tensor.view(original_shape).tolist()
        else:
            return denorm_tensor.view(original_shape)


__all__ = ["DataNormalizer"]