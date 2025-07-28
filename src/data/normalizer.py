#!/usr/bin/env python3
"""
Data normalization module for chemical kinetics datasets.
This version preserves all data validation and logging and supports the
parallel preprocessing architecture with _merge_accumulators.
"""

import logging
import math
from typing import Dict, List, Any, Optional

import numpy as np
import torch

DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP = 50.0

class DataNormalizer:
    """Calculates normalization statistics with robust data validation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_config = config["data"]
        self.norm_config = config["normalization"]

        self.species_vars = self.data_config["species_variables"]
        self.global_vars = self.data_config["global_variables"]
        self.time_var = self.data_config["time_variable"]
        self.all_vars = self.species_vars + self.global_vars + [self.time_var]

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
                "method": method, "index": i, "count": 0, "mean": 0.0, "m2": 0.0,
                "min": float("inf"), "max": float("-inf"),
            }
            accumulators[var] = acc
        return accumulators
    
    def _get_method(self, var: str) -> str:
        """Get normalization method for a variable."""
        methods = self.norm_config.get("methods", {})
        return methods.get(var, self.norm_config["default_method"])
    
    def _update_single_accumulator(self, acc: Dict[str, Any], vec: np.ndarray, var_name: str):
        """
        Vectorized update for one accumulator, including all validation and logging.
        """
        if vec.size == 0:
            return

        # 1. Filter non-finite values and warn if there are many
        finite_mask = np.isfinite(vec)
        if not np.all(finite_mask):
            n_non_finite = (~finite_mask).sum()
            if n_non_finite / vec.size > 0.01:
                self.logger.warning(f"Variable '{var_name}' has {n_non_finite}/{vec.size} non-finite values.")
            vec = vec[finite_mask]
            if vec.size == 0:
                self.logger.warning(f"Variable '{var_name}' has no finite values after filtering, skipping.")
                return

        # 2. Handle log-transformations with validation checks
        if acc["method"].startswith("log-"):
            below_epsilon = vec < self.epsilon
            if np.any(below_epsilon):
                self.logger.warning(
                    f"Variable '{var_name}' has {below_epsilon.sum()} values below epsilon {self.epsilon}. "
                    f"Min value: {vec.min():.2e}"
                )
            vec = np.log10(np.maximum(vec, self.epsilon))
            
            if vec.min() < -30 or vec.max() > 30:
                self.logger.warning(
                    f"Variable '{var_name}' has extreme log values: [{vec.min():.1f}, {vec.max():.1f}]"
                )

        # 3. Perform Chan's parallel update for mean and variance
        n_b = vec.size
        mean_b = float(vec.mean())
        m2_b = float(((vec - mean_b) ** 2).sum()) if n_b > 1 else 0.0

        n_a = acc["count"]
        delta = mean_b - acc["mean"]
        n_ab = n_a + n_b

        if n_ab > 0:
            acc["mean"] = (n_a * acc["mean"] + n_b * mean_b) / n_ab
            acc["m2"] += m2_b + delta**2 * n_a * n_b / n_ab
        
        acc["count"] = n_ab
        acc["min"] = min(acc["min"], float(vec.min()))
        acc["max"] = max(acc["max"], float(vec.max()))

    def _merge_accumulators(
        self,
        main_accs: Dict[str, Dict[str, Any]],
        other_accs: Dict[str, Dict[str, Any]],
    ) -> None:
        """Merge statistics from another set of accumulators into the main one."""
        for var, other_acc in other_accs.items():
            if not other_acc: continue
            if var not in main_accs:
                main_accs[var] = other_acc
                continue

            main_acc = main_accs[var]
            
            n_a, mean_a, m2_a = main_acc["count"], main_acc["mean"], main_acc["m2"]
            n_b, mean_b, m2_b = other_acc["count"], other_acc["mean"], other_acc["m2"]
            
            n_ab = n_a + n_b
            if n_ab == 0: continue
                
            delta = mean_b - mean_a
            
            main_acc["mean"] = (n_a * mean_a + n_b * mean_b) / n_ab
            main_acc["m2"] = m2_a + m2_b + (delta**2 * n_a * n_b) / n_ab
            main_acc["count"] = n_ab
            main_acc["min"] = min(main_acc["min"], other_acc["min"])
            main_acc["max"] = max(main_acc["max"], other_acc["max"])
        
    def _finalize_statistics(self, accumulators: Dict[str, Dict[str, Any]], is_ratio: bool = False) -> Dict[str, Any]:
        """Finalize statistics from accumulators."""
        stats = { "per_key_stats": {} }
        if not is_ratio:
            stats["normalization_methods"] = {}

        for var, acc in accumulators.items():
            method = acc.get("method", "standard")
            if not is_ratio:
                stats["normalization_methods"][var] = method
            
            if method == "none": continue
            
            var_stats = {"method": method}
            
            if acc["count"] > 1:
                variance = acc["m2"] / (acc["count"] - 1)
                std = max(math.sqrt(variance), self.min_std)
            else:
                std = 1.0
            
            if "standard" in method:
                mean_key, std_key = ("log_mean", "log_std") if "log" in method else ("mean", "std")
                var_stats[mean_key], var_stats[std_key] = acc["mean"], std
            elif "min-max" in method:
                var_stats["min"], var_stats["max"] = acc["min"], acc["max"]
                if acc["max"] - acc["min"] < self.epsilon:
                    var_stats["max"] = acc["min"] + 1.0
            
            if is_ratio:
                stats[var] = {"mean": acc["mean"], "std": std, "min": acc["min"], "max": acc["max"], "count": acc["count"]}
            else:
                stats["per_key_stats"][var] = var_stats

        if not is_ratio:
            for var in self.all_vars:
                if var not in stats["normalization_methods"]:
                    stats["normalization_methods"][var] = "none"
            stats["epsilon"] = self.epsilon
            stats["clamp_value"] = self.norm_config.get("clamp_value", DEFAULT_CLAMP)
        
        return stats


class NormalizationHelper:
    """Applies pre-computed normalization statistics to data tensors."""
    
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
        
        self.ratio_stats = stats.get("ratio_stats", None)

        self.logger = logging.getLogger(__name__)
        self._precompute_parameters()

    def _precompute_parameters(self):
        """Pre-compute normalization parameters on the target device for efficiency."""
        self.norm_params = {}
        self.method_groups = { "standard": [], "log-standard": [], "min-max": [], "log-min-max": [], "none": [] }
        var_to_col = {var: i for i, var in enumerate(self.species_vars + self.global_vars + [self.time_var])}

        for var, method in self.methods.items():
            if method == "none" or var not in self.per_key_stats:
                self.method_groups["none"].append(var)
                continue

            var_stats = self.per_key_stats[var]
            params = {"method": method}

            if "standard" in method:
                mean_key, std_key = ("log_mean", "log_std") if "log" in method else ("mean", "std")
                params["mean"] = torch.tensor(var_stats[mean_key], dtype=torch.float32, device=self.device)
                params["std"] = torch.tensor(var_stats[std_key], dtype=torch.float32, device=self.device)
            elif "min-max" in method:
                params["min"] = torch.tensor(var_stats["min"], dtype=torch.float32, device=self.device)
                params["max"] = torch.tensor(var_stats["max"], dtype=torch.float32, device=self.device)

            self.norm_params[var] = params
            self.method_groups[method].append(var)

        self.col_indices = {method: [var_to_col[var] for var in v_list] for method, v_list in self.method_groups.items() if v_list}

    def normalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Normalize a complete profile tensor using vectorized operations."""
        if profile.device != self.device:
            profile = profile.to(self.device)
        normalized = profile.clone()

        for method, col_idxs in self.col_indices.items():
            if not col_idxs or method == "none": continue
            cols = normalized[:, col_idxs]
            if "standard" in method:
                means = torch.stack([self.norm_params[var]["mean"] for var in self.method_groups[method]])
                stds = torch.stack([self.norm_params[var]["std"] for var in self.method_groups[method]])
                data = torch.log10(torch.clamp(cols, min=self.epsilon)) if "log" in method else cols
                normalized[:, col_idxs] = torch.clamp((data - means) / stds, -self.clamp_value, self.clamp_value)
            elif "min-max" in method:
                mins = torch.stack([self.norm_params[var]["min"] for var in self.method_groups[method]])
                maxs = torch.stack([self.norm_params[var]["max"] for var in self.method_groups[method]])
                ranges = torch.clamp(maxs - mins, min=self.epsilon)
                data = torch.log10(torch.clamp(cols, min=self.epsilon)) if "log" in method else cols
                normalized[:, col_idxs] = torch.clamp((data - mins) / ranges, 0.0, 1.0)
        return normalized

    def denormalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Denormalize a complete profile tensor using vectorized operations."""
        if profile.device != self.device:
            profile = profile.to(self.device)
        denormalized = profile.clone()

        for method, col_idxs in self.col_indices.items():
            if not col_idxs or method == "none": continue
            cols = denormalized[:, col_idxs]
            if "standard" in method:
                means = torch.stack([self.norm_params[var]["mean"] for var in self.method_groups[method]])
                stds = torch.stack([self.norm_params[var]["std"] for var in self.method_groups[method]])
                raw_vals = cols * stds + means
                if "log" in method:
                    raw_vals = torch.pow(10.0, torch.clamp(raw_vals, min=-38.0, max=38.0))
                denormalized[:, col_idxs] = torch.clamp(raw_vals, min=-3.4e38, max=3.4e38)
            elif "min-max" in method:
                mins = torch.stack([self.norm_params[var]["min"] for var in self.method_groups[method]])
                maxs = torch.stack([self.norm_params[var]["max"] for var in self.method_groups[method]])
                ranges = torch.clamp(maxs - mins, min=self.epsilon)
                raw_vals = cols * ranges + mins
                if "log" in method:
                    raw_vals = torch.pow(10.0, torch.clamp(raw_vals, min=-38.0, max=38.0))
                denormalized[:, col_idxs] = raw_vals
        return denormalized
        
    def denormalize_ratio_predictions(self, standardized_log_ratios: torch.Tensor,
                                    initial_species: torch.Tensor) -> torch.Tensor:
        """Convert standardized log-ratio predictions back to absolute species values."""
        if self.ratio_stats is None:
            raise ValueError("Ratio statistics not available for denormalization.")

        device = standardized_log_ratios.device
        initial_species = initial_species.to(device)

        ratio_means = torch.tensor([self.ratio_stats[var]["mean"] for var in self.species_vars], device=device, dtype=torch.float32)
        ratio_stds = torch.tensor([self.ratio_stats[var]["std"] for var in self.species_vars], device=device, dtype=torch.float32)
        
        log_ratios = (standardized_log_ratios * ratio_stds) + ratio_means
        log_ratios = torch.clamp(log_ratios, min=-38.0, max=38.0)
        
        ratios = torch.pow(10.0, log_ratios)
        predicted_species = initial_species * ratios
        return predicted_species