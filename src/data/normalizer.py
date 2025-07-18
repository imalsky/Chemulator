#!/usr/bin/env python3
"""
Data normalization module for chemical kinetics datasets.
"""

import logging
import math
from typing import Dict, List, Any, Optional

import numpy as np
import torch


# Normalization constants
DEFAULT_EPSILON = 1e-37
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP = 50.0


class DataNormalizer:
    """Calculate normalization statistics from data during preprocessing."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_config = config["data"]
        self.norm_config = config["normalization"]

        # Variable lists
        self.species_vars = self.data_config["species_variables"]
        self.global_vars = self.data_config["global_variables"]
        self.time_var = self.data_config["time_variable"]
        self.all_vars = self.species_vars + self.global_vars + [self.time_var]

        # Numerical constants
        self.epsilon = self.norm_config.get("epsilon", DEFAULT_EPSILON)
        self.min_std = self.norm_config.get("min_std", DEFAULT_MIN_STD)

        self.logger = logging.getLogger(__name__)
        
    def _initialize_accumulators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize per-variable statistics accumulators."""
        accumulators = {}
        for i, var in enumerate(self.all_vars):
            method = self._get_method(var)
            if method == "none":
                continue
                
                
            acc = {
                "method": method,
                "index": i,
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
            accumulators[var] = acc
        return accumulators
    
    def _get_method(self, var: str) -> str:
        """Get normalization method for a variable."""
        methods = self.norm_config.get("methods", {})
        method = methods.get(var, self.norm_config["default_method"])
        return method
    
    def _update_accumulators(
        self,
        data: np.ndarray,
        accumulators: Dict[str, Dict[str, Any]],
        n_timesteps: int,
    ) -> None:
        """Vectorised Chan update of running mean/variance & min/max."""
        _, _, _ = data.shape  # n_profiles, n_t, n_vars

        for var, acc in accumulators.items():
            idx     = acc["index"]
            method  = acc["method"]

            if var in self.global_vars:
                value = float(data[0, 0, idx])
                if method in {"log-standard", "log-min-max"}:
                    value = np.log10(np.maximum(value, self.epsilon))
                
                # Treat each profile as one observation, regardless of timesteps, to avoid biasing towards longer profiles
                n_b    = 1  
                mean_b = value
                m2_b   = 0.0

                acc["min"] = min(acc["min"], value)
                acc["max"] = max(acc["max"], value)

            # ─ species / time variables – fully vectorised ──────────────────────
            else:
                vec = data[:, :, idx].ravel().astype(np.float64)
                vec = vec[np.isfinite(vec)]
                if vec.size == 0:
                    continue

                if method in {"log-standard", "log-min-max"}:
                    vec = np.log10(np.maximum(vec, self.epsilon))

                n_b    = vec.size
                mean_b = float(vec.mean())
                m2_b   = float(((vec - mean_b) ** 2).sum()) if n_b > 1 else 0.0

                acc["min"] = min(acc["min"], float(vec.min()))
                acc["max"] = max(acc["max"], float(vec.max()))

            # ─ Chan’s parallel mean/variance update ─────────────────────────────
            n_a  = acc["count"]
            delta = mean_b - acc["mean"]
            n_ab  = n_a + n_b

            acc["mean"] += delta * n_b / n_ab
            acc["m2"]   += m2_b + delta**2 * n_a * n_b / n_ab
            acc["count"] = n_ab
        
    def _finalize_statistics(self, accumulators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize statistics from accumulators."""
        stats = {
            "normalization_methods": {},
            "per_key_stats": {}
        }
        
        for var, acc in accumulators.items():
            method = acc["method"]
            stats["normalization_methods"][var] = method
            
            if method == "none":
                continue
            
            var_stats = {"method": method}
            
            # Calculate standard deviation
            if acc["count"] > 1:
                variance = acc["m2"] / (acc["count"] - 1)
                std = max(math.sqrt(variance), self.min_std)
            else:
                std = 1.0
            
            # Store statistics based on method
            if method == "standard":
                var_stats["mean"] = acc["mean"]
                var_stats["std"] = std
                
            elif method == "log-standard":
                var_stats["log_mean"] = acc["mean"]
                var_stats["log_std"] = std
                
            elif method == "log-min-max":
                var_stats["min"] = acc["min"]
                var_stats["max"] = acc["max"]
                if acc["max"] - acc["min"] < self.epsilon:
                    var_stats["max"] = acc["min"] + 1.0
            
            stats["per_key_stats"][var] = var_stats
        
        # Add methods for variables not in accumulators
        for var in self.all_vars:
            if var not in stats["normalization_methods"]:
                stats["normalization_methods"][var] = "none"
        
        stats["epsilon"] = self.epsilon
        stats["clamp_value"] = self.norm_config.get("clamp_value", DEFAULT_CLAMP)
        
        return stats


class NormalizationHelper:
    """Normalization helper for use during preprocessing."""
    
    def __init__(self, stats: Dict[str, Any], device: torch.device, 
                 species_vars: List[str], global_vars: List[str], 
                 time_var: str, config: Optional[Dict[str, Any]] = None):
        self.stats = stats
        self.device = device
        self.species_vars = species_vars
        self.global_vars = global_vars
        self.time_var = time_var

        self.n_species = len(species_vars)
        self.n_globals = len(global_vars)
        self.methods = stats["normalization_methods"]
        self.per_key_stats = stats["per_key_stats"]

        self.epsilon = stats.get("epsilon", DEFAULT_EPSILON)
        self.clamp_value = stats.get("clamp_value", DEFAULT_CLAMP)

        self.logger = logging.getLogger(__name__)

        # Pre-compute normalization parameters
        self._precompute_parameters()

    def _precompute_parameters(self):
        """Pre-compute normalization parameters for efficiency."""
        self.norm_params = {}

        # Group variables by normalization method
        self.method_groups = {
            "standard": [],
            "log-standard": [],
            "log-min-max": [],
            "none": []
        }

        # Create parameter tensors for each variable
        for var, method in self.methods.items():
            if method == "none" or var not in self.per_key_stats:
                self.method_groups["none"].append(var)
                continue

            var_stats = self.per_key_stats[var]
            params = {"method": method}

            if method == "standard":
                params["mean"] = torch.tensor(var_stats["mean"], dtype=torch.float32, device=self.device)
                params["std"] = torch.tensor(var_stats["std"], dtype=torch.float32, device=self.device)

            elif method == "log-standard":
                params["log_mean"] = torch.tensor(var_stats["log_mean"], dtype=torch.float32, device=self.device)
                params["log_std"] = torch.tensor(var_stats["log_std"], dtype=torch.float32, device=self.device)

            elif method == "log-min-max":
                params["min"] = torch.tensor(var_stats["min"], dtype=torch.float32, device=self.device)
                params["max"] = torch.tensor(var_stats["max"], dtype=torch.float32, device=self.device)

            self.norm_params[var] = params
            self.method_groups[method].append(var)

        # Pre-compute column indices
        self._compute_column_indices()

    def _compute_column_indices(self):
        """Pre-compute column indices for vectorized operations."""
        self.col_indices = {}

        all_vars = self.species_vars + self.global_vars + [self.time_var]
        var_to_col = {var: i for i, var in enumerate(all_vars)}

        for method, vars_list in self.method_groups.items():
            if vars_list:
                self.col_indices[method] = [var_to_col[var] for var in vars_list]

    def normalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Normalize a complete profile tensor using vectorized operations."""
        if profile.device != self.device:
            profile = profile.to(self.device)

        normalized = profile.clone()

        # Apply normalization for each method group
        for method, col_idxs in self.col_indices.items():
            if not col_idxs or method == "none":
                continue

            # Get columns for this method
            cols = normalized[:, col_idxs]

            if method == "standard":
                means = torch.stack([self.norm_params[var]["mean"] for var in self.method_groups[method]])
                stds = torch.stack([self.norm_params[var]["std"] for var in self.method_groups[method]])
                normalized[:, col_idxs] = torch.clamp(
                    (cols - means) / stds,
                    -self.clamp_value, self.clamp_value
                )

            elif method == "log-standard":
                log_means = torch.stack([self.norm_params[var]["log_mean"] for var in self.method_groups[method]])
                log_stds = torch.stack([self.norm_params[var]["log_std"] for var in self.method_groups[method]])
                log_data = torch.log10(torch.clamp(cols, min=self.epsilon))
                normalized[:, col_idxs] = torch.clamp(
                    (log_data - log_means) / log_stds,
                    -self.clamp_value, self.clamp_value
                )

            elif method == "log-min-max":
                mins = torch.stack([self.norm_params[var]["min"] for var in self.method_groups[method]])
                maxs = torch.stack([self.norm_params[var]["max"] for var in self.method_groups[method]])
                log_data = torch.log10(torch.clamp(cols, min=self.epsilon))
                ranges = maxs - mins
                ranges = torch.clamp(ranges, min=self.epsilon)
                normalized[:, col_idxs] = torch.clamp((log_data - mins) / ranges, 0.0, 1.0)

        return normalized

    def denormalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Denormalize a complete profile tensor."""
        if profile.device != self.device:
            profile = profile.to(self.device)

        denormalized = profile.clone()

        for method, col_idxs in self.col_indices.items():
            if not col_idxs or method == "none":
                continue

            cols = denormalized[:, col_idxs]

            if method == "standard":
                means = torch.stack([self.norm_params[var]["mean"] for var in self.method_groups[method]])
                stds = torch.stack([self.norm_params[var]["std"] for var in self.method_groups[method]])
                denormalized[:, col_idxs] = cols * stds + means

            elif method == "log-standard":
                log_means = torch.stack([self.norm_params[var]["log_mean"] for var in self.method_groups[method]])
                log_stds = torch.stack([self.norm_params[var]["log_std"] for var in self.method_groups[method]])
                log_data = cols * log_stds + log_means
                log_data = torch.clamp(log_data, min=-38.0, max=38.0)
                denormalized[:, col_idxs] = torch.pow(10.0, log_data)

            elif method == "log-min-max":
                mins = torch.stack([self.norm_params[var]["min"] for var in self.method_groups[method]])
                maxs = torch.stack([self.norm_params[var]["max"] for var in self.method_groups[method]])
                ranges = maxs - mins
                ranges = torch.clamp(ranges, min=self.epsilon)
                log_data = cols * ranges + mins
                log_data = torch.clamp(log_data, min=-38.0, max=38.0)
                denormalized[:, col_idxs] = torch.pow(10.0, log_data)

        return denormalized