#!/usr/bin/env python3
"""
normalizer.py

Normalization utilities for chemical kinetics data.
Supports z-space (log-standard normalized) transformations.

Note: Log10-space helpers for rollout training live in trainer.py's
AdaptiveStiffLoss class to keep rollout logic co-located with training.
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
    - dt normalization using log10 + min-max scaling

    Performance: When all keys share the same normalization method (common case),
    uses vectorized operations instead of column-by-column loops.
    """

    def __init__(
            self,
            manifest: Dict[str, Any],
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
        """
        self.manifest = manifest or {}

        self.per_key_stats = dict(self.manifest.get("per_key_stats", {}))
        self.methods = dict(self.manifest.get("normalization_methods", {}))

        self.epsilon = float(self.manifest.get("epsilon", DEFAULT_EPSILON))
        self.min_std = float(self.manifest.get("min_std", DEFAULT_MIN_STD))

        self.dt_spec = self.manifest.get("dt", None)
        if self.dt_spec is None:
            raise ValueError("dt normalization spec missing from manifest")

        try:
            log_min = float(self.dt_spec["log_min"])
            log_max = float(self.dt_spec["log_max"])
        except Exception as e:
            raise ValueError("Bad dt spec; expected {'log_min': <float>, 'log_max': <float>}.") from e

        self.dt_min_phys: float = max(10.0 ** log_min, float(self.epsilon))
        self.dt_max_phys: float = 10.0 ** log_max

        # Cache for vectorized normalization tensors (built on first use per key-set)
        self._vectorized_cache: Dict[Tuple[str, ...], Optional[Dict[str, Any]]] = {}

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
        # Always work in float32 for numerical safety with small dt values
        dt_f32 = dt_phys.to(torch.float32) if dt_phys.dtype != torch.float32 else dt_phys

        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])
        range_log = max(log_max - log_min, MIN_RANGE_EPSILON)

        phys_min, phys_max = self.dt_min_phys, self.dt_max_phys

        # Tolerate tiny round-off near bounds
        phys_min_eff = phys_min * (1.0 - R_TOLERANCE)
        phys_max_eff = phys_max * (1.0 + R_TOLERANCE)

        ood_mask = (dt_f32 < phys_min_eff) | (dt_f32 > phys_max_eff)
        if torch.any(ood_mask):
            n_ood = int(ood_mask.sum().item())
            n_tot = int(dt_f32.numel())
            mn = float(torch.nanmin(dt_f32).item())
            mx = float(torch.nanmax(dt_f32).item())
            warnings.warn(
                f"[normalize_dt_from_phys] {n_ood}/{n_tot} ({100.0 * n_ood / n_tot:.1f}%) dt values "
                f"outside training range [{phys_min:.3e}, {phys_max:.3e}] seconds. "
                f"Found values in range [{mn:.3e}, {mx:.3e}]. Values will be clamped.",
                RuntimeWarning,
                stacklevel=2,
            )

        dt_clamped = dt_f32.clamp(min=max(self.epsilon, phys_min), max=phys_max)
        dt_log = torch.log10(dt_clamped)

        dt_norm = (dt_log - log_min) / range_log
        dt_norm = dt_norm.clamp_(0.0, 1.0)

        return dt_norm

    def denormalize_dt_to_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized dt back to physical units."""
        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])

        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        dt_log = log_min + dt_norm * (log_max - log_min)
        dt_phys = torch.pow(10.0, dt_log)

        return torch.clamp(dt_phys, min=self.epsilon)

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_key(self, key: str, method: str) -> Dict[str, float]:
        """
        Validate that a key exists and has required stats for its method.

        Raises:
            KeyError: If key is missing from methods or stats
            ValueError: If required stats are missing for the method
        """
        if key not in self.methods:
            raise KeyError(
                f"Normalization method not found for key '{key}'. "
                f"Available keys: {sorted(self.methods.keys())}"
            )

        if key not in self.per_key_stats:
            raise KeyError(
                f"Statistics not found for key '{key}'. "
                f"Available keys: {sorted(self.per_key_stats.keys())}"
            )

        stats = self.per_key_stats[key]

        # Validate required stats based on method
        if method == "standard":
            required = ["mean", "std"]
        elif method == "min-max":
            required = ["min", "max"]
        elif method == "log-standard":
            required = ["log_mean", "log_std"]
        elif method == "log-min-max":
            required = ["log_min", "log_max"]
        else:
            raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")

        missing = [r for r in required if r not in stats]
        if missing:
            raise ValueError(
                f"Missing required stats {missing} for key '{key}' with method '{method}'. "
                f"Available stats: {sorted(stats.keys())}"
            )

        return stats

    # =========================================================================
    # Vectorized Tensor Caching
    # =========================================================================

    def _get_vectorized_params(
            self, keys: Sequence[str], device: torch.device
    ) -> Optional[Dict[str, Any]]:
        """
        Build or retrieve cached vectorized parameters for a key set.

        Returns None if keys use mixed methods (fall back to column-wise).
        Returns dict with method-specific tensors if all keys share a method.
        """
        cache_key = tuple(keys)

        if cache_key in self._vectorized_cache:
            cached = self._vectorized_cache[cache_key]
            if cached is None:
                return None
            # Move tensors to correct device if needed
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in cached.items()
            }

        # Check if all keys use the same method
        methods_for_keys = [self.methods[k] for k in keys]
        unique_methods = set(methods_for_keys)

        if len(unique_methods) != 1:
            # Mixed methods - cache None to skip check next time
            self._vectorized_cache[cache_key] = None
            return None

        method = methods_for_keys[0]

        # Build vectorized parameter tensors
        if method == "standard":
            means = torch.tensor(
                [float(self.per_key_stats[k]["mean"]) for k in keys],
                dtype=torch.float32
            )
            stds = torch.tensor(
                [max(float(self.per_key_stats[k]["std"]), self.min_std) for k in keys],
                dtype=torch.float32
            )
            params: Dict[str, Any] = {"method": method, "means": means, "stds": stds}

        elif method == "min-max":
            mins = torch.tensor(
                [float(self.per_key_stats[k]["min"]) for k in keys],
                dtype=torch.float32
            )
            maxs = torch.tensor(
                [float(self.per_key_stats[k]["max"]) for k in keys],
                dtype=torch.float32
            )
            ranges = (maxs - mins).clamp(min=MIN_RANGE_EPSILON)
            params = {"method": method, "mins": mins, "ranges": ranges}

        elif method == "log-standard":
            log_means = torch.tensor(
                [float(self.per_key_stats[k]["log_mean"]) for k in keys],
                dtype=torch.float32
            )
            log_stds = torch.tensor(
                [max(float(self.per_key_stats[k]["log_std"]), self.min_std) for k in keys],
                dtype=torch.float32
            )
            params = {"method": method, "log_means": log_means, "log_stds": log_stds}

        elif method == "log-min-max":
            log_mins = torch.tensor(
                [float(self.per_key_stats[k]["log_min"]) for k in keys],
                dtype=torch.float32
            )
            log_maxs = torch.tensor(
                [float(self.per_key_stats[k]["log_max"]) for k in keys],
                dtype=torch.float32
            )
            ranges = (log_maxs - log_mins).clamp(min=MIN_RANGE_EPSILON)
            params = {"method": method, "log_mins": log_mins, "ranges": ranges}

        else:
            self._vectorized_cache[cache_key] = None
            return None

        self._vectorized_cache[cache_key] = params
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}

    # =========================================================================
    # Column-wise Implementation (with vectorized fast path)
    # =========================================================================

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

        # Validate all keys first
        for key in keys:
            method = self.methods.get(key)
            if method is None:
                raise KeyError(
                    f"Normalization method not found for key '{key}'. "
                    f"Available keys: {sorted(self.methods.keys())}"
                )
            self._validate_key(key, method)

        # Work in float32 for numerical stability
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # Try vectorized fast path
        params = self._get_vectorized_params(keys, x.device)

        if params is not None:
            result = self._normalize_vectorized(x_f32, params)
        else:
            result = self._normalize_columnwise(x_f32, keys)

        # Cast back to original dtype if needed
        if original_dtype != torch.float32:
            result = result.to(original_dtype)

        return result

    def _normalize_vectorized(
            self, x: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Vectorized normalization when all columns share the same method."""
        method = params["method"]
        eps_floor = max(self.epsilon, torch.finfo(torch.float32).tiny)

        if method == "standard":
            means = params["means"]
            stds = params["stds"]
            return (x - means) / stds

        elif method == "min-max":
            mins = params["mins"]
            ranges = params["ranges"]
            return (x - mins) / ranges

        elif method == "log-standard":
            log_means = params["log_means"]
            log_stds = params["log_stds"]
            x_log = torch.log10(x.clamp(min=eps_floor))
            return (x_log - log_means) / log_stds

        elif method == "log-min-max":
            log_mins = params["log_mins"]
            ranges = params["ranges"]
            x_log = torch.log10(x.clamp(min=eps_floor))
            return (x_log - log_mins) / ranges

        else:
            raise ValueError(f"Unknown method in vectorized path: {method}")

    def _normalize_columnwise(
            self, x: torch.Tensor, keys: Sequence[str]
    ) -> torch.Tensor:
        """Column-by-column normalization for mixed methods."""
        result = x.clone()
        eps_floor = max(self.epsilon, torch.finfo(torch.float32).tiny)

        for col_idx, key in enumerate(keys):
            method = self.methods[key]
            stats = self.per_key_stats[key]
            col_data = result[..., col_idx]

            if method == "standard":
                mean = float(stats["mean"])
                std = max(float(stats["std"]), self.min_std)
                col_norm = (col_data - mean) / std

            elif method == "min-max":
                min_val = float(stats["min"])
                max_val = float(stats["max"])
                range_val = max(max_val - min_val, MIN_RANGE_EPSILON)
                col_norm = (col_data - min_val) / range_val

            elif method == "log-standard":
                log_mean = float(stats["log_mean"])
                log_std = max(float(stats["log_std"]), self.min_std)
                col_log = torch.log10(col_data.clamp(min=eps_floor))
                col_norm = (col_log - log_mean) / log_std

            elif method == "log-min-max":
                log_min = float(stats["log_min"])
                log_max = float(stats["log_max"])
                range_val = max(log_max - log_min, MIN_RANGE_EPSILON)
                col_log = torch.log10(col_data.clamp(min=eps_floor))
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

        # Validate all keys first
        for key in keys:
            method = self.methods.get(key)
            if method is None:
                raise KeyError(
                    f"Normalization method not found for key '{key}'. "
                    f"Available keys: {sorted(self.methods.keys())}"
                )
            self._validate_key(key, method)

        # Work in float32 for numerical stability
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # Try vectorized fast path
        params = self._get_vectorized_params(keys, x.device)

        if params is not None:
            result = self._denormalize_vectorized(x_f32, params)
        else:
            result = self._denormalize_columnwise(x_f32, keys)

        # Cast back to original dtype if needed
        if original_dtype != torch.float32:
            result = result.to(original_dtype)

        return result

    def _denormalize_vectorized(
            self, x: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Vectorized denormalization when all columns share the same method."""
        method = params["method"]

        if method == "standard":
            means = params["means"]
            stds = params["stds"]
            return x * stds + means

        elif method == "min-max":
            mins = params["mins"]
            ranges = params["ranges"]
            return x * ranges + mins

        elif method == "log-standard":
            log_means = params["log_means"]
            log_stds = params["log_stds"]
            x_log = x * log_stds + log_means
            return torch.pow(10.0, x_log)

        elif method == "log-min-max":
            log_mins = params["log_mins"]
            ranges = params["ranges"]
            x_log = x * ranges + log_mins
            return torch.pow(10.0, x_log)

        else:
            raise ValueError(f"Unknown method in vectorized path: {method}")

    def _denormalize_columnwise(
            self, x: torch.Tensor, keys: Sequence[str]
    ) -> torch.Tensor:
        """Column-by-column denormalization for mixed methods."""
        result = x.clone()

        for col_idx, key in enumerate(keys):
            method = self.methods[key]
            stats = self.per_key_stats[key]
            col_data = result[..., col_idx]

            if method == "standard":
                mean = float(stats["mean"])
                std = max(float(stats["std"]), self.min_std)
                col_denorm = col_data * std + mean

            elif method == "min-max":
                min_val = float(stats["min"])
                max_val = float(stats["max"])
                range_val = max(max_val - min_val, MIN_RANGE_EPSILON)
                col_denorm = col_data * range_val + min_val

            elif method == "log-standard":
                log_mean = float(stats["log_mean"])
                log_std = max(float(stats["log_std"]), self.min_std)
                col_log = col_data * log_std + log_mean
                col_denorm = torch.pow(10.0, col_log)

            elif method == "log-min-max":
                log_min = float(stats["log_min"])
                log_max = float(stats["log_max"])
                range_val = max(log_max - log_min, MIN_RANGE_EPSILON)
                col_log = col_data * range_val + log_min
                col_denorm = torch.pow(10.0, col_log)

            else:
                raise ValueError(f"Unknown method '{method}' for key '{key}'")

            result[..., col_idx] = col_denorm

        return result
