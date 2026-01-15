#!/usr/bin/env python3
"""
normalizer.py

Normalization utilities for chemical kinetics data.
Supports z-space (log-standard normalized) and log10-physical space conversions.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

# =============================================================================
# Constants
# =============================================================================

MIN_RANGE_EPSILON = 1e-12
R_TOLERANCE = 1e-6
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10


# =============================================================================
# Normalization Helper
# =============================================================================


class NormalizationHelper:
    """
    Manages normalization using preprocessed statistics.

    Provides utilities for:
    - Standard normalization (z-space)
    - Log10 space conversions for rollout training
    - Clipping and noise injection in log10 space
    """

    def __init__(
        self,
        manifest: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize normalization helper.

        Args:
            manifest: Dict loaded from normalization.json with keys:
              - per_key_stats: per-variable stats dicts
              - normalization_methods: per-variable method names
              - dt: {'log_min': float, 'log_max': float} for dt log10 bounds
              - epsilon: small positive floor used for log transforms
              - min_std: minimum std clamp for 'standard' normalization
            device: Optional torch.device for any tensors created internally
        """
        self.manifest = manifest or {}
        self.device = device

        self.per_key_stats = dict(self.manifest.get("per_key_stats", {}))
        self.methods = dict(self.manifest.get("normalization_methods", {}))

        self.epsilon = float(self.manifest.get("epsilon", DEFAULT_EPSILON))
        self.min_std = float(self.manifest.get("min_std", DEFAULT_MIN_STD))

        self.dt_spec = self.manifest.get("dt", None)
        if self.dt_spec is None:
            raise ValueError("dt normalization spec missing")

        try:
            log_min = float(self.dt_spec["log_min"])
            log_max = float(self.dt_spec["log_max"])
        except Exception as e:
            raise ValueError("Bad dt spec; expected {'log_min': <float>, 'log_max': <float>}.") from e

        self.dt_min_phys: float = max(10.0 ** log_min, float(self.epsilon))
        self.dt_max_phys: float = 10.0 ** log_max

        # Cache for log stats tensors (lazy init)
        self._log_stats_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # =========================================================================
    # Log10 Space Helpers (for rollout training)
    # =========================================================================

    def get_log_stats(
        self,
        keys: Sequence[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log10 means and stds for specified species.

        Returns:
            (log_means, log_stds): Tensors of shape [len(keys)]
        """
        cache_key = tuple(keys)
        if cache_key not in self._log_stats_cache:
            log_means = []
            log_stds = []
            for key in keys:
                stats = self.per_key_stats.get(key, {})
                log_means.append(float(stats.get("log_mean", 0.0)))
                log_stds.append(max(float(stats.get("log_std", 1.0)), self.min_std))

            self._log_stats_cache[cache_key] = (
                torch.tensor(log_means, dtype=torch.float32),
                torch.tensor(log_stds, dtype=torch.float32),
            )

        log_means, log_stds = self._log_stats_cache[cache_key]

        if device is not None:
            log_means = log_means.to(device=device)
            log_stds = log_stds.to(device=device)
        if dtype is not None:
            log_means = log_means.to(dtype=dtype)
            log_stds = log_stds.to(dtype=dtype)

        return log_means, log_stds

    def z_to_log10(
        self,
        z: torch.Tensor,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert z-normalized values to log10 space (DIFFERENTIABLE).

        Args:
            z: [..., S] z-normalized values
            log_means: [S] per-species log10 means
            log_stds: [S] per-species log10 stds

        Returns:
            log10_vals: [..., S] values in log10 space
        """
        return z * log_stds + log_means

    def log10_to_z(
        self,
        log10_vals: torch.Tensor,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert log10 values to z-normalized space (DIFFERENTIABLE).

        Args:
            log10_vals: [..., S] values in log10 space
            log_means: [S] per-species log10 means
            log_stds: [S] per-species log10 stds

        Returns:
            z: [..., S] z-normalized values
        """
        return (log10_vals - log_means) / log_stds

    def clamp_z_in_log10_space(
        self,
        z: torch.Tensor,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        log10_min: float = -30.0,
        log10_max: float = 10.0,
    ) -> torch.Tensor:
        """
        Clamp z values by converting to log10, clamping, and converting back.

        Useful for rollout training to prevent nonphysical values.

        Args:
            z: [..., S] z-normalized values
            log_means: [S] per-species log10 means
            log_stds: [S] per-species log10 stds
            log10_min: minimum allowed log10 value
            log10_max: maximum allowed log10 value

        Returns:
            z_clamped: [..., S] clamped z-normalized values
        """
        log10_vals = self.z_to_log10(z, log_means, log_stds)
        log10_clamped = torch.clamp(log10_vals, min=log10_min, max=log10_max)
        return self.log10_to_z(log10_clamped, log_means, log_stds)

    def add_noise_in_log10_space(
        self,
        z: torch.Tensor,
        log_means: torch.Tensor,
        log_stds: torch.Tensor,
        noise_std: float = 0.01,
    ) -> torch.Tensor:
        """
        Add Gaussian noise in log10 space, then convert back to z-space.

        Implements Kelp-style noise injection for robustness.

        Args:
            z: [..., S] z-normalized values
            log_means: [S] per-species log10 means
            log_stds: [S] per-species log10 stds
            noise_std: standard deviation of noise in log10 space

        Returns:
            z_noisy: [..., S] z-normalized values with noise added
        """
        log10_vals = self.z_to_log10(z, log_means, log_stds)
        noise = torch.randn_like(log10_vals) * noise_std
        log10_noisy = log10_vals + noise
        return self.log10_to_z(log10_noisy, log_means, log_stds)

    # =========================================================================
    # Standard Normalization Interface
    # =========================================================================

    def normalize(
        self,
        x: torch.Tensor,
        keys: Sequence[str],
    ) -> torch.Tensor:
        """Normalize data using per-variable methods."""
        if x.ndim == 1:
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._normalize_columns(x.unsqueeze(-1), keys)[..., 0]

        return self._normalize_columns(x, keys)

    def denormalize(
        self,
        x: torch.Tensor,
        keys: Sequence[str],
    ) -> torch.Tensor:
        """Inverse normalization operation."""
        if x.ndim == 1:
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._denormalize_columns(x.unsqueeze(-1), keys)[..., 0]

        return self._denormalize_columns(x, keys)

    def normalize_dt_from_phys(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize physical dt (seconds) to [0,1] using log10 + min-max.

        IMPORTANT: Always returns float32 to preserve precision for small dt.
        """
        if not torch.is_floating_point(dt_phys):
            dt_phys = dt_phys.float()

        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])
        range_log = max(log_max - log_min, MIN_RANGE_EPSILON)

        phys_min, phys_max = self.dt_min_phys, self.dt_max_phys

        # Tolerate tiny round-off near bounds
        phys_min_eff = phys_min * (1.0 - R_TOLERANCE)
        phys_max_eff = phys_max * (1.0 + R_TOLERANCE)

        ood_mask = (dt_phys < phys_min_eff) | (dt_phys > phys_max_eff)
        if torch.any(ood_mask):
            n_ood = int(ood_mask.sum().item())
            n_tot = int(dt_phys.numel())
            mn = float(torch.nanmin(dt_phys).item())
            mx = float(torch.nanmax(dt_phys).item())
            warnings.warn(
                f"[normalize_dt_from_phys] {n_ood}/{n_tot} ({100.0 * n_ood / n_tot:.1f}%) dt values "
                f"outside training range [{phys_min:.3e}, {phys_max:.3e}] seconds. "
                f"Found values in range [{mn:.3e}, {mx:.3e}]. Values will be clamped.",
                RuntimeWarning,
                stacklevel=2,
            )

        dt_clamped = dt_phys.clamp(min=max(self.epsilon, phys_min), max=phys_max)
        dt_log = torch.log10(dt_clamped)

        dt_norm = (dt_log - log_min) / range_log
        dt_norm = dt_norm.clamp_(0.0, 1.0)

        return dt_norm.to(torch.float32)

    def denormalize_dt_to_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized dt back to physical units."""
        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])

        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        dt_log = log_min + dt_norm * (log_max - log_min)
        dt_phys = torch.pow(10.0, dt_log)

        return torch.clamp(dt_phys, min=self.epsilon)

    def _normalize_columns(
        self,
        x: torch.Tensor,
        keys: Sequence[str],
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
                range_val = max(max_val - min_val, MIN_RANGE_EPSILON)
                col_norm = (col_data - min_val) / range_val

            elif method == "log-standard":
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_log = torch.log10(torch.clamp(col_data, min=self.epsilon))
                col_norm = (col_log - log_mean) / log_std

            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                range_val = max(log_max - log_min, MIN_RANGE_EPSILON)
                col_log = torch.log10(torch.clamp(col_data, min=self.epsilon))
                col_norm = (col_log - log_min) / range_val

            else:
                raise ValueError(f"Unknown method '{method}' for key '{key}'")

            result[..., col_idx] = col_norm

        return result

    def _denormalize_columns(
        self,
        x: torch.Tensor,
        keys: Sequence[str],
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
                range_val = max(max_val - min_val, MIN_RANGE_EPSILON)
                col_denorm = col_data * range_val + min_val

            elif method == "log-standard":
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_log = col_data * log_std + log_mean
                col_denorm = torch.pow(10.0, col_log)

            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                range_val = max(log_max - log_min, MIN_RANGE_EPSILON)
                col_log = col_data * range_val + log_min
                col_denorm = torch.pow(10.0, col_log)

            else:
                raise ValueError(f"Unknown method '{method}' for key '{key}'")

            result[..., col_idx] = col_denorm

        return result
