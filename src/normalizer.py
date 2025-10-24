#!/usr/bin/env python3
"""
Normalization Module
=====================
Centralized normalization.
Key focus: Consistent dt normalization to [0,1] using log-min-max.
"""

from __future__ import annotations

from typing import Dict, Any, List, Sequence, Optional

import torch


class NormalizationHelper:
    """
    Manages normalization using preprocessed statistics.
    All dt values normalized to [0,1] using log-min-max transform.
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
                - dt: dt normalization parameters (log_min, log_max)
                - epsilon: Floor value for log operations
                - min_std: Minimum standard deviation
        """
        self.manifest = manifest or {}
        self.per_key_stats = dict(self.manifest.get("per_key_stats", {}))
        self.methods = dict(self.manifest.get("normalization_methods", {}))

        # Numerical stability parameters
        self.epsilon = float(self.manifest.get("epsilon", 1e-30))
        self.min_std = float(self.manifest.get("min_std", 1e-10))

        # dt normalization spec (critical for model)
        self.dt_spec = self.manifest.get("dt", None)
        if self.dt_spec is None:
            raise ValueError(
                "dt normalization spec missing from manifest. "
                "Run dataset once to auto-generate or add manually."
            )

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
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._denormalize_columns(x.unsqueeze(-1), keys)[..., 0]

        return self._denormalize_columns(x, keys)

    def normalize_dt_from_phys(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize physical time differences to [0,1].
        Uses log-min-max transform from dt spec.

        Args:
            dt_phys: Physical time differences

        Returns:
            Normalized dt tensor in [0, 1]
        """
        if self.dt_spec is None:
            raise ValueError("dt normalization spec required but not found in manifest")

        # Always use dt spec (no fallback)
        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])

        # Log transform with floor
        dt_safe = torch.clamp(dt_phys, min=self.epsilon)
        dt_log = torch.log10(dt_safe)

        # Normalize to [0,1]
        denominator = max(log_max - log_min, 1e-12)
        dt_norm = (dt_log - log_min) / denominator

        return torch.clamp(dt_norm, 0.0, 1.0)

    def denormalize_dt_to_phys(
            self,
            dt_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert normalized dt back to physical units.
        Inverse of normalize_dt_from_phys.

        Args:
            dt_norm: Normalized dt in [0,1]

        Returns:
            Physical time differences
        """
        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])

        # Inverse transform
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        dt_log = log_min + dt_norm * (log_max - log_min)
        dt_phys = torch.pow(10.0, dt_log)

        return torch.clamp(dt_phys, min=self.epsilon)

    def _normalize_columns(
            self,
            x: torch.Tensor,
            keys: Sequence[str]
    ) -> torch.Tensor:
        """Normalize each column using its specified method."""
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor")

        num_keys = len(keys)
        if x.shape[-1] != num_keys:
            raise ValueError(f"Shape mismatch: {x.shape[-1]} cols vs {num_keys} keys")

        result = x.clone()

        for col_idx, key in enumerate(keys):
            method = str(self.methods.get(key, "log-standard"))
            stats = self.per_key_stats.get(key, {})

            col_data = result[..., col_idx]

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
                raise ValueError(f"Unknown method '{method}' for key '{key}'")

            result[..., col_idx] = col_norm

        return result

    def _denormalize_columns(
            self,
            x: torch.Tensor,
            keys: Sequence[str]
    ) -> torch.Tensor:
        """Denormalize each column using its specified method."""
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor")

        num_keys = len(keys)
        if x.shape[-1] != num_keys:
            raise ValueError(f"Shape mismatch: {x.shape[-1]} cols vs {num_keys} keys")

        result = x.clone()

        for col_idx, key in enumerate(keys):
            method = str(self.methods.get(key, "log-standard"))
            stats = self.per_key_stats.get(key, {})

            col_data = result[..., col_idx]

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
                raise ValueError(f"Unknown method '{method}' for key '{key}'")

            result[..., col_idx] = col_denorm

        return result