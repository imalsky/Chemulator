#!/usr/bin/env python3
"""
Data normalization module for chemical kinetics datasets.

Provides efficient normalization with multiple methods
and numerical stability guarantees.
Used only during preprocessing; no runtime calls.
"""

import logging
import math
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Normalization constants
DEFAULT_EPSILON = 1e-37
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP = 100.0
DEFAULT_SYMLOG_PERCENTILE = 0.5
FIXED_RESERVOIR_SIZE = 1000000  # Fixed size for symlog quantile estimation

FLOAT32_MAX_EXPONENT = 38  # log10(float32_max) ≈ 38.5

class DataNormalizer:
    """
    Calculate normalization statistics from data during preprocessing.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 *,
                 actual_timesteps: int) -> None:
        """
        Initialize the normalizer.

        Args:
            config: Full configuration dictionary.
            actual_timesteps: Number of timesteps in the data (for reservoir sizing).
        """
        self.config        = config
        self.data_config   = config["data"]
        self.norm_config   = config["normalization"]

        # Variable lists
        self.species_vars  = self.data_config["species_variables"]
        self.global_vars   = self.data_config["global_variables"]
        self.time_var      = self.data_config["time_variable"]
        self.all_vars      = self.species_vars + self.global_vars + [self.time_var]

        # Numerical constants
        self.epsilon       = self.norm_config.get("epsilon", 1e-37)
        self.min_std       = self.norm_config.get("min_std", 1e-10)

        self._actual_timesteps = actual_timesteps

        self.logger = logging.getLogger(__name__)
        
    def _initialize_accumulators(self, n_profiles: int) -> Dict[str, Dict[str, Any]]:
        """
        Initialise per-variable statistics accumulators.
        Uses fixed reservoir size for simplicity and to avoid dynamic estimation issues.
        """
        self.logger.info(
            f"Using fixed reservoir size {FIXED_RESERVOIR_SIZE:,} for symlog quantile estimation"
        )

        accumulators: Dict[str, Dict[str, Any]] = {}
        for i, var in enumerate(self.all_vars):
            method = self._get_method(var)
            if method == "none":
                continue
            acc = dict(
                method     = method,
                index      = i,
                count      = 0,
                mean       = 0.0,
                m2         = 0.0,
                min        = float("inf"),
                max        = float("-inf"),
                reservoir  = ReservoirSampler(FIXED_RESERVOIR_SIZE, seed=self.config["system"]["seed"]) if method == "symlog" else None,
            )
            accumulators[var] = acc
        return accumulators
    
    def _get_method(self, var: str) -> str:
        """Get normalization method for a variable."""
        methods = self.norm_config.get("methods", {})
        return methods.get(var, self.norm_config["default_method"])
    
    def _update_accumulators(
        self, 
        data: np.ndarray,
        accumulators: Dict[str, Dict[str, Any]],
        n_timesteps: int
    ):
        """
        Update accumulators with a chunk of data using Chan et al.'s parallel algorithm.
        Assumes data shape: (n_profiles, n_timesteps, n_vars)
        """
        n_profiles, n_t_check, _ = data.shape
        if n_t_check != n_timesteps:
            raise ValueError(f"Mismatched timesteps: expected {n_timesteps}, got {n_t_check}")
        
        # Optional profiling if anomaly detection enabled
        if self.config["system"].get("detect_anomaly", False):
            with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
                self._perform_update_accumulators(data, accumulators, n_timesteps)
            prof.export_chrome_trace("update_accumulators_trace.json")
            self.logger.info("Profiling trace saved to update_accumulators_trace.json")
        else:
            self._perform_update_accumulators(data, accumulators, n_timesteps)

    def _perform_update_accumulators(
        self,
        data: np.ndarray,
        accumulators: Dict[str, Dict[str, Any]],
        n_timesteps: int
    ):
        """Helper for core update logic, separable for profiling."""
        for var, acc in accumulators.items():
            var_idx = acc["index"]
            method = acc["method"]
            
            # Extract variable data (3D assumed)
            if var in self.global_vars:
                # Globals: constant, repeat n_timesteps times per profile for consistent counting
                per_profile = data[:, 0, var_idx]  # shape (n_profiles,)
                var_data = np.repeat(per_profile, n_timesteps)  # shape (n_profiles * n_timesteps,)
            else:
                # Species/time: varying
                var_data = data[:, :, var_idx].flatten()  # shape (n_profiles * n_timesteps,)
            
            var_data = var_data.astype(np.float64)
            
            # Filter out invalid values
            valid_mask = np.isfinite(var_data)
            var_data = var_data[valid_mask]
            
            if len(var_data) == 0:
                continue
            
            # Apply transformations based on method
            if method in ["log-standard", "log-min-max"]:
                var_data = np.log10(np.maximum(var_data, self.epsilon))
            
            # Calculate statistics for this chunk
            n_b = len(var_data)
            mean_b = np.mean(var_data, dtype=np.float64)
            
            # Min/max
            acc["min"] = min(acc["min"], float(np.min(var_data)))
            acc["max"] = max(acc["max"], float(np.max(var_data)))
            
            # Update mean and variance using Chan et al.'s parallel algorithm
            if acc["count"] == 0:
                acc["count"] = n_b
                acc["mean"] = float(mean_b)
                if n_b > 1:
                    acc["m2"] = float(np.sum((var_data - mean_b) ** 2, dtype=np.float64))
            else:
                n_a = acc["count"]
                mean_a = acc["mean"]
                m2_a = acc["m2"]
                
                n_ab = n_a + n_b
                
                delta = mean_b - mean_a
                mean_ab = mean_a + delta * n_b / n_ab
                
                if n_b > 1:
                    m2_b = float(np.sum((var_data - mean_b) ** 2, dtype=np.float64))
                else:
                    m2_b = 0.0
                
                m2_ab = m2_a + m2_b + delta * delta * n_a * n_b / n_ab
                
                acc["count"] = n_ab
                acc["mean"] = float(mean_ab)
                acc["m2"] = float(m2_ab)
            
            # Efficient quantile sampling for symlog
            if method == "symlog" and acc["reservoir"] is not None:
                acc["reservoir"].add_samples(var_data)
    
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
                
            elif method == "symlog":
                if acc["reservoir"] and acc["reservoir"].size > 0:
                    values = acc["reservoir"].get_samples()
                    percentile = self.norm_config.get("symlog_percentile", DEFAULT_SYMLOG_PERCENTILE)
                    threshold = np.percentile(np.abs(values), percentile * 100)
                    var_stats["threshold"] = max(float(threshold), self.epsilon)
                    
                    abs_vals = np.abs(values)
                    linear_mask = abs_vals <= var_stats["threshold"]
                    transformed = np.zeros_like(values)
                    
                    transformed[linear_mask] = values[linear_mask] / var_stats["threshold"]
                    log_vals = np.maximum(abs_vals[~linear_mask] / var_stats["threshold"], self.epsilon)
                    transformed[~linear_mask] = np.sign(values[~linear_mask]) * (
                        np.log10(log_vals) + 1
                    )
                    
                    var_stats["scale_factor"] = max(float(np.max(np.abs(transformed))), 1.0)
                else:
                    var_stats["threshold"] = 1.0
                    var_stats["scale_factor"] = 1.0
            
            stats["per_key_stats"][var] = var_stats
        
        # Add methods for variables not in accumulators
        for var in self.all_vars:
            if var not in stats["normalization_methods"]:
                stats["normalization_methods"][var] = "none"
        
        stats["epsilon"] = self.epsilon
        stats["clamp_value"] = self.norm_config.get("clamp_value", DEFAULT_CLAMP)
        
        return stats


class ReservoirSampler:
    """Efficient reservoir sampling for quantile estimation with proper seeding."""
    
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.reservoir = []
        self.count = 0
        # Initialize RNG with fixed seed for reproducibility
        self.rng = np.random.RandomState(seed)
    
    def add_samples(self, samples: np.ndarray):
        """Add samples to the reservoir."""
        for sample in samples:
            self.count += 1
            if len(self.reservoir) < self.capacity:
                self.reservoir.append(float(sample))
            else:
                j = self.rng.randint(0, self.count)
                if j < self.capacity:
                    self.reservoir[j] = float(sample)
    
    @property
    def size(self) -> int:
        return len(self.reservoir)
    
    def get_samples(self) -> np.ndarray:
        return np.array(self.reservoir, dtype=np.float64)


class NormalizationHelper:
    """
    Normalization helper for use during preprocessing.
    """

    def __init__(self,
                 stats: Dict[str, Any],
                 device: torch.device,
                 species_vars: List[str],
                 global_vars: List[str],
                 time_var: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the normalization helper.

        Args:
            stats: Pre-computed normalization statistics
            device: Target device for operations (usually CPU)
            species_vars: List of species variable names
            global_vars: List of global variable names
            time_var: Time variable name
            config: Full configuration dictionary (optional)
        """
        self.stats = stats
        self.device = device
        self.species_vars = species_vars
        self.global_vars = global_vars
        self.time_var = time_var

        # Use lengths to compute column offsets
        self.n_species = len(species_vars)
        self.n_globals = len(global_vars)
        self.methods = stats["normalization_methods"]
        self.per_key_stats = stats["per_key_stats"]

        # Numerical constants from stats or config or defaults
        self.epsilon = stats.get("epsilon", DEFAULT_EPSILON)

        # Get clamp_value from stats first, then config, then default
        self.clamp_value = stats.get("clamp_value", DEFAULT_CLAMP)
        if config and "normalization" in config:
            self.clamp_value = config["normalization"].get("clamp_value", self.clamp_value)

        # Pre-compute normalization parameters on device
        self._precompute_parameters()

        # Pre-compute for sample batch normalization
        self._precompute_sample_columns()

    def _precompute_parameters(self):
        """Pre-compute normalization parameters for efficiency."""
        self.norm_params = {}

        # Group variables by normalization method for vectorized operations
        self.method_groups = {
            "standard": [],
            "log-standard": [],
            "log-min-max": [],
            "symlog": [],
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

            elif method == "symlog":
                params["threshold"] = torch.tensor(var_stats["threshold"], dtype=torch.float32, device=self.device)
                params["scale_factor"] = torch.tensor(var_stats["scale_factor"], dtype=torch.float32, device=self.device)

            self.norm_params[var] = params
            self.method_groups[method].append(var)

        # Pre-compute column indices for each method group
        self._compute_column_indices()

    def _compute_column_indices(self):
        """Pre-compute column indices for vectorized operations."""
        self.col_indices = {}

        all_vars = self.species_vars + self.global_vars + [self.time_var]
        var_to_col = {var: i for i, var in enumerate(all_vars)}

        for method, vars_list in self.method_groups.items():
            if vars_list:
                self.col_indices[method] = [var_to_col[var] for var in vars_list]

    def _precompute_sample_columns(self):
        """Pre-compute variable mapping and indices for sample batches."""
        # Sample columns: species_init + globals + time + species_target
        self.sample_col_vars = self.species_vars + self.global_vars + [self.time_var] + self.species_vars

        # Group sample columns by method
        self.sample_col_indices = {}
        for method in self.method_groups:
            self.sample_col_indices[method] = [i for i, var in enumerate(self.sample_col_vars) if self.methods.get(var, "none") == method]

    def normalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Normalize a complete profile tensor using vectorized operations.

        Args:
            profile: Tensor of shape (timesteps, n_species + n_globals + 1)
        Returns:
            Normalized profile tensor
        """
        # Ensure profile is on the same device as normalization parameters
        if profile.device != self.device:
            profile = profile.to(self.device)

        normalized = profile.clone()

        # Apply normalization for each method group (vectorized)
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

            elif method == "symlog":
                for i, var in enumerate(self.method_groups[method]):
                    params = self.norm_params[var]
                    col_idx = col_idxs[i]
                    normalized[:, col_idx] = self._symlog_transform(
                        profile[:, col_idx], 
                        params["threshold"], 
                        params["scale_factor"]
                    )

        return normalized

    def normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of samples using vectorized operations.

        Args:
            batch: Tensor of shape (batch_size, 2*n_species + n_globals + 1)
        Returns:
            Normalized batch tensor
        """
        if batch.device != self.device:
            batch = batch.to(self.device)

        normalized = batch.clone()

        # Apply normalization for each method group on sample columns (vectorized)
        for method, col_idxs in self.sample_col_indices.items():
            if not col_idxs or method == "none":
                continue

            cols = normalized[:, col_idxs]

            if method == "standard":
                var_list = [self.sample_col_vars[col] for col in col_idxs]
                means = torch.stack([self.norm_params[v]["mean"] for v in var_list])
                stds = torch.stack([self.norm_params[v]["std"] for v in var_list])
                normalized[:, col_idxs] = torch.clamp(
                    (cols - means) / stds,
                    -self.clamp_value, self.clamp_value
                )

            elif method == "log-standard":
                var_list = [self.sample_col_vars[col] for col in col_idxs]
                log_means = torch.stack([self.norm_params[v]["log_mean"] for v in var_list])
                log_stds = torch.stack([self.norm_params[v]["log_std"] for v in var_list])
                log_data = torch.log10(torch.clamp(cols, min=self.epsilon))
                normalized[:, col_idxs] = torch.clamp(
                    (log_data - log_means) / log_stds,
                    -self.clamp_value, self.clamp_value
                )

            elif method == "log-min-max":
                var_list = [self.sample_col_vars[col] for col in col_idxs]
                mins = torch.stack([self.norm_params[v]["min"] for v in var_list])
                maxs = torch.stack([self.norm_params[v]["max"] for v in var_list])
                log_data = torch.log10(torch.clamp(cols, min=self.epsilon))
                ranges = maxs - mins
                ranges = torch.clamp(ranges, min=self.epsilon)
                normalized[:, col_idxs] = torch.clamp((log_data - mins) / ranges, 0.0, 1.0)

            elif method == "symlog":
                for i, col_idx in enumerate(col_idxs):
                    var = self.sample_col_vars[col_idx]
                    params = self.norm_params[var]
                    normalized[:, col_idx] = self._symlog_transform(
                        batch[:, col_idx],
                        params["threshold"],
                        params["scale_factor"]
                    )

        return normalized

    def denormalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a complete profile tensor using vectorized operations.

        Args:
            profile: Normalized tensor of shape (timesteps, n_species + n_globals + 1)
        Returns:
            Denormalized profile tensor in original units
        """
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
                denormalized[:, col_idxs] = torch.pow(10.0, log_data)

            elif method == "log-min-max":
                mins = torch.stack([self.norm_params[var]["min"] for var in self.method_groups[method]])
                maxs = torch.stack([self.norm_params[var]["max"] for var in self.method_groups[method]])
                ranges = maxs - mins
                ranges = torch.clamp(ranges, min=self.epsilon)
                log_data = cols * ranges + mins
                denormalized[:, col_idxs] = torch.pow(10.0, log_data)

            elif method == "symlog":
                for i, var in enumerate(self.method_groups[method]):
                    params = self.norm_params[var]
                    col_idx = col_idxs[i]
                    denormalized[:, col_idx] = self._symlog_inverse(
                        profile[:, col_idx],
                        params["threshold"],
                        params["scale_factor"]
                    )

        return denormalized

    def _symlog_transform(self, data: torch.Tensor, threshold: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply symlog transformation with optimized vectorized operations."""
        abs_data = torch.abs(data)
        sign_data = torch.sign(data)

        linear_part = data / threshold
        log_arg = torch.clamp(abs_data / threshold, min=self.epsilon)
        log_part = sign_data * (torch.log10(log_arg) + 1.0)

        result = torch.where(
            abs_data <= threshold,
            linear_part,
            log_part
        ) / scale

        return torch.clamp(result, -1.0, 1.0)

    def _symlog_inverse(self, data: torch.Tensor, threshold: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse symlog transformation with overflow protection.
        """
        scaled_data = data * scale
        abs_scaled = torch.abs(scaled_data)
        sign_scaled = torch.sign(scaled_data)

        linear_part = scaled_data * threshold
        exponent = torch.clamp(abs_scaled - 1.0, max=FLOAT32_MAX_EXPONENT)
        log_part = sign_scaled * threshold * torch.pow(10.0, exponent)

        result = torch.where(
            abs_scaled <= 1.0,
            linear_part,
            log_part
        )

        return result