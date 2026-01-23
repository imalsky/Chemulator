#!/usr/bin/env python3
"""
normalizer.py - Normalization utilities for the flow-map training pipeline.

Handles bidirectional conversion between physical and normalized (z-space) values:
    - Species: log10-standardized: z = (log10(y_phys) - log_mean) / log_std
    - Time (dt): log10 + min-max: dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

Manifest Structure:
    normalization_methods: {variable_name: method_string}
    per_key_stats: {variable_name: {log_mean, log_std, log_min, log_max, ...}}
    dt: {log_min, log_max} or dt_seconds: scalar
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch

# Numerical stability constants
_RELATIVE_TOLERANCE = 1e-6
_MIN_RANGE_EPSILON = 1e-12
_DEFAULT_EPSILON = 1e-30
_DEFAULT_MIN_STD = 1e-12


@dataclass(frozen=True)
class DtSpec:
    """Specification for dt normalization bounds (in log10 seconds)."""

    log_min: float
    log_max: float

    @property
    def phys_min(self) -> float:
        """Minimum physical dt in seconds."""
        return 10.0**self.log_min

    @property
    def phys_max(self) -> float:
        """Maximum physical dt in seconds."""
        return 10.0**self.log_max

    @property
    def log_range(self) -> float:
        """Range in log10 space, with minimum epsilon for stability."""
        return max(self.log_max - self.log_min, _MIN_RANGE_EPSILON)


def parse_dt_spec(dt_obj: Any) -> DtSpec:
    """
    Parse dt specification from manifest.

    Accepts:
        - None: defaults to (0.0, 0.0)
        - Scalar (seconds): constant dt
        - Dict with log_min/log_max or dt_seconds
    """
    if dt_obj is None:
        return DtSpec(log_min=0.0, log_max=0.0)

    if isinstance(dt_obj, (int, float, np.floating, np.integer)):
        dt_s = float(dt_obj)
        if dt_s <= 0:
            raise ValueError(f"dt_seconds must be positive, got {dt_s}")
        lg = float(np.log10(dt_s))
        return DtSpec(log_min=lg, log_max=lg)

    if isinstance(dt_obj, Mapping):
        if "log_min" in dt_obj and "log_max" in dt_obj:
            return DtSpec(
                log_min=float(dt_obj["log_min"]), log_max=float(dt_obj["log_max"])
            )
        if "dt_seconds" in dt_obj:
            dt_s = float(dt_obj["dt_seconds"])
            if dt_s <= 0:
                raise ValueError(f"dt_seconds must be positive, got {dt_s}")
            lg = float(np.log10(dt_s))
            return DtSpec(log_min=lg, log_max=lg)

    raise TypeError(f"Unrecognized dt spec type: {type(dt_obj)}")


class NormalizationHelper:
    """
    Bidirectional normalization for species and time variables.

    Normalizes/denormalizes based on per-variable statistics stored in the manifest.
    Supports multiple normalization methods per variable.
    """

    def __init__(self, manifest: Dict[str, Any]) -> None:
        """
        Initialize from normalization manifest.

        Args:
            manifest: Dictionary containing normalization_methods, per_key_stats,
                     and dt specification.
        """
        self.manifest = dict(manifest)

        # Support both old and new key names
        self.methods: Dict[str, str] = dict(
            self.manifest.get("normalization_methods", self.manifest.get("methods", {}))
        )
        self.per_key_stats: Dict[str, Dict[str, Any]] = dict(
            self.manifest.get("per_key_stats", self.manifest.get("stats", {}))
        )

        self.epsilon = float(self.manifest.get("epsilon", _DEFAULT_EPSILON))
        self.min_std = float(self.manifest.get("min_std", _DEFAULT_MIN_STD))

        # Parse dt specification
        dt_obj = self.manifest.get("dt") or self.manifest.get("dt_seconds")
        self.dt_spec = parse_dt_spec(dt_obj)

    def normalize_dt_from_phys(
        self, dt_phys: Union[torch.Tensor, np.ndarray, float]
    ) -> torch.Tensor:
        """
        Normalize physical dt (seconds) to [0, 1] using log10 + min-max.

        Args:
            dt_phys: Physical time step(s) in seconds.

        Returns:
            Normalized dt tensor in [0, 1].
        """
        if not isinstance(dt_phys, torch.Tensor):
            dt_phys = torch.as_tensor(dt_phys)
        dt_phys = dt_phys.to(dtype=torch.float64)

        # Check for out-of-distribution values
        rtol = _RELATIVE_TOLERANCE
        phys_min_eff = self.dt_spec.phys_min * (1.0 - rtol)
        phys_max_eff = self.dt_spec.phys_max * (1.0 + rtol)

        ood_mask = (dt_phys < phys_min_eff) | (dt_phys > phys_max_eff)
        if torch.any(ood_mask):
            n_ood, n_tot = int(ood_mask.sum()), dt_phys.numel()
            warnings.warn(
                f"[normalize_dt] {n_ood}/{n_tot} dt values outside training range "
                f"[{self.dt_spec.phys_min:.3e}, {self.dt_spec.phys_max:.3e}]. Clamping.",
                RuntimeWarning,
                stacklevel=2,
            )

        dt_clamped = dt_phys.clamp(
            min=max(self.epsilon, self.dt_spec.phys_min), max=self.dt_spec.phys_max
        )
        dt_log = torch.log10(dt_clamped)
        dt_norm = (dt_log - self.dt_spec.log_min) / self.dt_spec.log_range

        return dt_norm.clamp(0.0, 1.0)

    def denormalize_dt_to_phys(
        self, dt_norm: Union[torch.Tensor, np.ndarray, float]
    ) -> torch.Tensor:
        """
        Convert normalized dt back to physical seconds.

        Args:
            dt_norm: Normalized time step(s) in [0, 1].

        Returns:
            Physical dt tensor in seconds.
        """
        if not isinstance(dt_norm, torch.Tensor):
            dt_norm = torch.as_tensor(dt_norm)
        dt_norm = dt_norm.to(dtype=torch.float64).clamp(0.0, 1.0)

        dt_log = self.dt_spec.log_min + dt_norm * self.dt_spec.log_range
        dt_phys = 10.0**dt_log

        return dt_phys.clamp(self.dt_spec.phys_min, self.dt_spec.phys_max)

    def _get_method_and_stats(self, key: str) -> tuple[str, Dict[str, Any]]:
        """Get normalization method and statistics for a variable."""
        method = str(self.methods.get(key, "standard") or "standard")
        stats = self.per_key_stats.get(key)
        if stats is None:
            raise KeyError(f"Missing stats for key '{key}' in normalization manifest.")
        return method, stats

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize a variable from physical to z-space.

        Args:
            x: Physical values tensor.
            key: Variable name for looking up normalization parameters.

        Returns:
            Normalized (z-space) tensor.
        """
        method, stats = self._get_method_and_stats(key)

        if method in ("identity", "none", ""):
            return x

        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return (x - mu) / sd

        if method == "min-max":
            mn, mx = float(stats["min"]), float(stats["max"])
            return (x - mn) / max(mx - mn, _MIN_RANGE_EPSILON)

        if method in ("log-standard", "log10-standard"):
            mu = float(stats["log_mean"])
            sd = max(float(stats["log_std"]), self.min_std)
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mu) / sd

        if method == "log-min-max":
            mn, mx = float(stats["log_min"]), float(stats["log_max"])
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mn) / max(mx - mn, _MIN_RANGE_EPSILON)

        raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")

    def denormalize(self, z: torch.Tensor, key: str) -> torch.Tensor:
        """
        Denormalize a variable from z-space to physical.

        Args:
            z: Normalized (z-space) tensor.
            key: Variable name for looking up normalization parameters.

        Returns:
            Physical values tensor.
        """
        method, stats = self._get_method_and_stats(key)

        if method in ("identity", "none", ""):
            return z

        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return z * sd + mu

        if method == "min-max":
            mn, mx = float(stats["min"]), float(stats["max"])
            return z * max(mx - mn, _MIN_RANGE_EPSILON) + mn

        if method in ("log-standard", "log10-standard"):
            mu = float(stats["log_mean"])
            sd = max(float(stats["log_std"]), self.min_std)
            return 10.0 ** (z * sd + mu)

        if method == "log-min-max":
            mn, mx = float(stats["log_min"]), float(stats["log_max"])
            return 10.0 ** (z * max(mx - mn, _MIN_RANGE_EPSILON) + mn)

        raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")
