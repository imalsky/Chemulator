#!/usr/bin/env python3
"""
normalizer.py

Normalization helper consistent with preprocessing.py and trainer.py.

Key points:
- Species z-space is log10-standardized:
    z = (log10(y_phys) - log_mean) / log_std
- dt is normalized with log10 + min-max from a dt-spec in the manifest:
    dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

Manifest compatibility:
- Preferred keys:
    normalization_methods: {key: method}
    per_key_stats: {key: {log_mean, log_std, log_min, log_max, ...}}
    dt: {log_min, log_max}   (log10 seconds)
- Backward-compatible aliases:
    methods -> normalization_methods
    stats   -> per_key_stats
- Also accepts dt_seconds as a scalar (interpreted as log_min=log_max=log10(dt_seconds)).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch

R_TOLERANCE = 1e-6
MIN_RANGE_EPSILON = 1e-12


@dataclass(frozen=True)
class DtSpec:
    log_min: float
    log_max: float


def _coerce_dt_spec(dt_obj: Any) -> DtSpec:
    """Accept dt spec as dict-like or as a scalar dt_seconds."""
    if dt_obj is None:
        return DtSpec(log_min=0.0, log_max=0.0)

    # scalar seconds
    if isinstance(dt_obj, (int, float, np.floating, np.integer)):
        dt_s = float(dt_obj)
        if dt_s <= 0:
            raise ValueError(f"dt_seconds must be >0, got {dt_s}")
        lg = float(np.log10(dt_s))
        return DtSpec(log_min=lg, log_max=lg)

    if isinstance(dt_obj, Mapping):
        if "log_min" in dt_obj and "log_max" in dt_obj:
            return DtSpec(log_min=float(dt_obj["log_min"]), log_max=float(dt_obj["log_max"]))
        if "dt_seconds" in dt_obj:
            dt_s = float(dt_obj["dt_seconds"])
            if dt_s <= 0:
                raise ValueError(f"dt_seconds must be >0, got {dt_s}")
            lg = float(np.log10(dt_s))
            return DtSpec(log_min=lg, log_max=lg)

    raise TypeError(f"Unrecognized dt spec type: {type(dt_obj)}")


class NormalizationHelper:
    def __init__(self, manifest: Dict[str, Any]) -> None:
        self.manifest = dict(manifest)

        # Backward-compatible key mapping
        self.methods: Dict[str, str] = dict(
            self.manifest.get("normalization_methods", self.manifest.get("methods", {}))
        )
        self.per_key_stats: Dict[str, Dict[str, Any]] = dict(
            self.manifest.get("per_key_stats", self.manifest.get("stats", {}))
        )

        self.epsilon = float(self.manifest.get("epsilon", 1e-30))
        self.min_std = float(self.manifest.get("min_std", 1e-12))

        # dt spec: prefer manifest["dt"], else manifest["dt_seconds"]
        dt_obj = self.manifest.get("dt", None)
        if dt_obj is None and "dt_seconds" in self.manifest:
            dt_obj = self.manifest["dt_seconds"]
        self.dt_spec = _coerce_dt_spec(dt_obj)

        # Precompute physical dt bounds from dt spec (log10 bounds)
        self.dt_min_phys = 10.0 ** float(self.dt_spec.log_min)
        self.dt_max_phys = 10.0 ** float(self.dt_spec.log_max)

    def normalize_dt_from_phys(self, dt_phys: Union[torch.Tensor, np.ndarray, float, int]) -> torch.Tensor:
        """Normalize physical Δt (seconds) to [0,1] using log10 + min-max."""
        if not isinstance(dt_phys, torch.Tensor):
            dt_phys = torch.as_tensor(dt_phys)
        if not torch.is_floating_point(dt_phys):
            dt_phys = dt_phys.float()
        dt_phys = dt_phys.to(dtype=torch.float64)

        log_min = float(self.dt_spec.log_min)
        log_max = float(self.dt_spec.log_max)
        range_log = max(log_max - log_min, MIN_RANGE_EPSILON)

        phys_min, phys_max = self.dt_min_phys, self.dt_max_phys

        # Tolerate tiny round-off near bounds
        rtol = R_TOLERANCE
        phys_min_eff, phys_max_eff = phys_min * (1.0 - rtol), phys_max * (1.0 + rtol)

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

        dt_clamped = dt_phys.clamp(min=max(self.epsilon, phys_min), max=phys_max)
        dt_log = torch.log10(dt_clamped)
        dt_norm = (dt_log - log_min) / range_log
        return dt_norm.clamp(0.0, 1.0)

    def denormalize_dt_to_phys(self, dt_norm: Union[torch.Tensor, np.ndarray, float, int]) -> torch.Tensor:
        """Convert normalized dt back to physical units (seconds)."""
        if not isinstance(dt_norm, torch.Tensor):
            dt_norm = torch.as_tensor(dt_norm)
        if not torch.is_floating_point(dt_norm):
            dt_norm = dt_norm.float()
        dt_norm = dt_norm.to(dtype=torch.float64).clamp(0.0, 1.0)

        log_min = float(self.dt_spec.log_min)
        log_max = float(self.dt_spec.log_max)
        dt_log = log_min + dt_norm * (log_max - log_min)
        dt_phys = 10.0 ** dt_log
        return dt_phys.clamp(self.dt_min_phys, self.dt_max_phys)

    def _get_method_and_stats(self, key: str) -> tuple[str, Dict[str, Any]]:
        method = str(self.methods.get(key, "standard") or "standard")
        stats = self.per_key_stats.get(key, None)
        if stats is None:
            raise KeyError(f"Missing stats for key '{key}' in normalization manifest.")
        return method, stats

    def _normalize_columns(self, x: torch.Tensor, key: str) -> torch.Tensor:
        method, stats = self._get_method_and_stats(key)

        if method in ("identity", "none", ""):
            return x

        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return (x - mu) / sd

        if method == "min-max":
            mn = float(stats["min"])
            mx = float(stats["max"])
            denom = max(mx - mn, MIN_RANGE_EPSILON)
            return (x - mn) / denom

        if method in ("log-standard", "log10-standard"):
            mu = float(stats["log_mean"])
            sd = max(float(stats["log_std"]), self.min_std)
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mu) / sd

        if method == "log-min-max":
            mn = float(stats["log_min"])
            mx = float(stats["log_max"])
            denom = max(mx - mn, MIN_RANGE_EPSILON)
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mn) / denom

        raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")

    def _denormalize_columns(self, z: torch.Tensor, key: str) -> torch.Tensor:
        method, stats = self._get_method_and_stats(key)

        if method in ("identity", "none", ""):
            return z

        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return z * sd + mu

        if method == "min-max":
            mn = float(stats["min"])
            mx = float(stats["max"])
            denom = max(mx - mn, MIN_RANGE_EPSILON)
            return z * denom + mn

        if method in ("log-standard", "log10-standard"):
            mu = float(stats["log_mean"])
            sd = max(float(stats["log_std"]), self.min_std)
            x_log = z * sd + mu
            return 10.0 ** x_log

        if method == "log-min-max":
            mn = float(stats["log_min"])
            mx = float(stats["log_max"])
            denom = max(mx - mn, MIN_RANGE_EPSILON)
            x_log = z * denom + mn
            return 10.0 ** x_log

        raise ValueError(f"Unknown normalization method '{method}' for key '{key}'")

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return self._normalize_columns(x, key)

    def denormalize(self, z: torch.Tensor, key: str) -> torch.Tensor:
        return self._denormalize_columns(z, key)
