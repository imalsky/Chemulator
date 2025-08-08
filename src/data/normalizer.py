#!/usr/bin/env python3
"""
Config-driven data normalization module for chemical kinetics datasets.
This helper can normalize arbitrary subsets of variables and handle
multi-dimensional tensors, correctly interpreting the normalization
scheme defined in the project's config file.
"""

import logging
from typing import Dict, List, Any, Tuple

import torch


class NormalizationHelper:
    """
    Applies pre-computed normalization statistics to torch tensors based on a
    flexible, config-driven scheme.
    """
    def __init__(self,
                 stats: Dict[str, Any],
                 device: torch.device,
                 config: Dict[str, Any]):

        self.stats = stats
        self.device = device
        self.config = config
        self.norm_config = config.get("normalization", {})
        self.logger = logging.getLogger(__name__)

        dtype_name = config.get("system", {}).get("dtype", "float32")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32

        # Load normalization parameters from the stats dictionary
        self.methods = self.stats.get("normalization_methods", {})
        self.per_key_stats = self.stats.get("per_key_stats", {})

        # --- FUNCTIONALITY PRESERVED ---
        # This logic is critical for sequence mode, where the preprocessor has already
        # applied the initial log10 transform. This flag prevents a double-log transform.
        self.inputs_already_logged = bool(self.stats.get("time_normalization")) or \
                                     bool(self.config.get("data", {}).get("sequence_mode", False))
        if self.inputs_already_logged:
            self.logger.info("NormalizationHelper running in 'inputs_already_logged' mode.")

        # Load scalar values from the config for consistent behavior
        self.epsilon = torch.tensor(self.norm_config.get("epsilon", 1e-30), dtype=self.dtype, device=self.device)
        self.clamp_val = self.norm_config.get("clamp_value", 50.0)
        self.min_std = self.norm_config.get("min_std", 1e-10)
        
        # Cache tensor for pow(10, x) operations for performance
        self._ten = torch.tensor(10.0, dtype=self.dtype, device=self.device)

    def _get_params(self, var_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Dynamically builds mean, std, and method lists for a given set of variables.
        This allows the helper to be flexible and not tied to a fixed data layout.
        """
        means, stds, methods = [], [], []

        for var in var_list:
            method = self.methods.get(var, "none")
            s = self.per_key_stats.get(var, {})
            
            mean, std = 0.0, 1.0
            if method == "log-standard":
                mean = s.get("log_mean", 0.0)
                std = s.get("log_std", 1.0)
            elif method == "standard":
                mean = s.get("mean", 0.0)
                std = s.get("std", 1.0)
            
            means.append(mean)
            stds.append(std)
            methods.append(method)

        means_t = torch.tensor(means, dtype=self.dtype, device=self.device)
        stds_t = torch.tensor(stds, dtype=self.dtype, device=self.device).clamp_min(self.min_std)
        
        return means_t, stds_t, methods

    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Normalizes data according to the scheme for the given list of variables.
        Handles multi-dimensional tensors by operating on the last dimension.
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data's last dimension ({data.shape[-1]}) must match var_list length ({len(var_list)})")
        
        # Ensure data is on the correct device and dtype
        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)

        # Get normalization parameters for this specific list of variables
        means, stds, methods = self._get_params(var_list)
        
        # Start with a copy to avoid modifying the original tensor
        norm_data = data.clone()

        # Step 1: Apply log transform where specified, respecting 'inputs_already_logged'
        for i, method in enumerate(methods):
            if "log" in (method or "") and not self.inputs_already_logged:
                norm_data[..., i] = torch.log10(norm_data[..., i].clamp_min(self.epsilon))

        # Step 2: Apply standardization (vectorized for performance)
        norm_data = (norm_data - means) / stds
        
        # Step 3: Clamp the final values to prevent outliers
        return torch.clamp(norm_data, -self.clamp_val, self.clamp_val)

    def denormalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Denormalizes data according to the scheme for the given variables.
        Handles multi-dimensional tensors and respects 'inputs_already_logged'.
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data's last dimension ({data.shape[-1]}) must match var_list length ({len(var_list)})")

        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)

        means, stds, methods = self._get_params(var_list)
        
        # Step 1: Invert standardization (vectorized)
        denorm_data = data * stds + means

        # Step 2: Invert log transform where specified, respecting 'inputs_already_logged'
        for i, method in enumerate(methods):
            if "log" in (method or "") and not self.inputs_already_logged:
                denorm_data[..., i] = torch.pow(self._ten, denorm_data[..., i])
                
        return denorm_data