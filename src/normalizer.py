#!/usr/bin/env python3
"""
normalizer.py
"""

from __future__ import annotations

from typing import Dict, Any, Sequence, Optional
import warnings
import torch

MIN_RANGE_EPSILON = 1e-12
R_TOLERANCE = 1e-6

class NormalizationHelper:
    """
    Manages normalization using preprocessed statistics.
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
              - per_key_stats: per-variable stats dicts (means, stds, mins, maxes, etc.)
              - normalization_methods: per-variable method names ('standard', 'min-max', 'log-standard', 'log-min-max')
              - dt: {'log_min': float, 'log_max': float} for Δt log10 bounds
              - epsilon: small positive floor used for log transforms (default 1e-30)
              - min_std: minimum std clamp for 'standard' normalization (default 1e-10)
            device: Optional torch.device for any tensors created internally
        """
        self.manifest = manifest or {}
        self.device = device

        # Per-variable normalization config
        self.per_key_stats = dict(self.manifest.get("per_key_stats", {}))
        self.methods = dict(self.manifest.get("normalization_methods", {}))

        # Numerical stability knobs
        self.epsilon = float(self.manifest.get("epsilon", 1e-30))
        self.min_std = float(self.manifest.get("min_std", 1e-10))

        # Δt
        self.dt_spec = self.manifest.get("dt", None)
        if self.dt_spec is None:
            raise ValueError("dt normalization spec missing")

        # Validate dt-spec and derive physical bounds used by the dataset
        try:
            log_min = float(self.dt_spec["log_min"])
            log_max = float(self.dt_spec["log_max"])
        except Exception as e:
            raise ValueError("Bad dt spec; expected {'log_min': <float>, 'log_max': <float>}.") from e

        # Derived physical Δt bounds (seconds). Clamp lower bound by epsilon to stay > 0 for logs.
        self.dt_min_phys: float = max(10.0 ** log_min, float(self.epsilon))
        self.dt_max_phys: float = 10.0 ** log_max

    def normalize(
            self,
            x: torch.Tensor,
            keys: Sequence[str]
    ) -> torch.Tensor:
        """
        Normalize data using per-variable methods.
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
        """
        if x.ndim == 1:
            if len(keys) != 1:
                raise ValueError(f"1D input requires single key, got {len(keys)}")
            return self._denormalize_columns(x.unsqueeze(-1), keys)[..., 0]

        return self._denormalize_columns(x, keys)

    def normalize_dt_from_phys(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize physical Δt (seconds) to [0,1] using log10 + min-max from the dt-spec.
        """
        if not torch.is_floating_point(dt_phys):
            dt_phys = dt_phys.float()

        # Spec
        log_min = float(self.dt_spec["log_min"])
        log_max = float(self.dt_spec["log_max"])
        range_log = max(log_max - log_min, MIN_RANGE_EPSILON)

        phys_min, phys_max = self.dt_min_phys, self.dt_max_phys

        # Tolerate tiny round-off near bounds
        rtol = R_TOLERANCE
        phys_min_eff, phys_max_eff = phys_min * (1.0 - rtol), phys_max * (1.0 + rtol)

        # OOD check (with tolerance)
        ood_mask = (dt_phys < phys_min_eff) | (dt_phys > phys_max_eff)
        if torch.any(ood_mask):
            n_ood = int(ood_mask.sum().item())
            n_tot = int(dt_phys.numel())
            mn = float(torch.nanmin(dt_phys).item())
            mx = float(torch.nanmax(dt_phys).item())
            warnings.warn(
                f"[normalize_dt_from_phys] {n_ood}/{n_tot} ({100.0 * n_ood / n_tot:.1f}%) Δt values "
                f"outside training range [{phys_min:.3e}, {phys_max:.3e}] seconds. "
                f"Found values in range [{mn:.3e}, {mx:.3e}]. Values will be clamped.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Clamp to strict spec bounds before log (not to the tolerant bounds)
        dt_clamped = dt_phys.clamp(min=max(self.epsilon, phys_min), max=phys_max)
        dt_log = torch.log10(dt_clamped)

        # Normalize to [0,1]
        dt_norm = (dt_log - log_min) / range_log
        dt_norm = dt_norm.clamp_(0.0, 1.0)  # 1-line FP guard

        return dt_norm

    def denormalize_dt_to_phys(
            self,
            dt_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert normalized dt back to physical units.
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