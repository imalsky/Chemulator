#!/usr/bin/env python3
"""normalizer.py

Normalization is driven entirely by the preprocessing manifest (normalization.json).

Key design points:
  - Per-key normalization methods are allowed to vary (species included).
  - Supported methods: "standard", "min-max", "log-standard", "log-min-max".
  - Δt normalization is defined by manifest["dt"] with log10-minmax bounds.

This module is intentionally strict:
  - Missing stats or unknown methods raise immediately.
  - No silent warnings or heuristic tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class DtSpec:
    """Δt normalization specification."""

    log_min: float
    log_max: float

    @property
    def range_log(self) -> float:
        r = self.log_max - self.log_min
        if r <= 0.0:
            raise ValueError("Invalid dt spec")
        return r


class NormalizationHelper:
    """Applies normalization and inverse normalization using a manifest."""

    def __init__(self, manifest: Mapping[str, Any], device: Optional[torch.device] = None) -> None:
        self.manifest: Mapping[str, Any] = manifest
        self.device = device

        per_key_stats = manifest.get("per_key_stats")
        methods = manifest.get("normalization_methods")
        if not isinstance(per_key_stats, Mapping) or not isinstance(methods, Mapping):
            raise ValueError("Invalid normalization manifest")

        self.per_key_stats: Dict[str, Dict[str, float]] = {
            str(k): dict(v) for k, v in per_key_stats.items()  # type: ignore[arg-type]
        }
        self.methods: Dict[str, str] = {str(k): str(v) for k, v in methods.items()}

        raw_epsilon = manifest.get("epsilon")
        raw_min_std = manifest.get("min_std")
        if raw_epsilon is None:
            raise KeyError("Missing required manifest key: 'epsilon'")
        if raw_min_std is None:
            raise KeyError("Missing required manifest key: 'min_std'")
        self.epsilon = float(raw_epsilon)
        self.min_std = float(raw_min_std)
        if self.epsilon <= 0.0 or self.min_std <= 0.0:
            raise ValueError("Invalid normalization manifest")

        dt = manifest.get("dt")
        if not isinstance(dt, Mapping):
            raise ValueError("dt normalization spec missing")

        self.dt_spec = DtSpec(log_min=float(dt["log_min"]), log_max=float(dt["log_max"]))
        self._vec_cache: Dict[tuple[tuple[str, ...], str, str, torch.dtype], torch.Tensor] = {}

    def normalize(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """Normalize columns of x according to per-key methods."""

        if x.ndim == 1:
            if x.shape[0] != len(keys):
                raise ValueError("Shape mismatch")
            return self._normalize_columns(x.unsqueeze(0), keys)[0]

        return self._normalize_columns(x, keys)

    def denormalize(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """Inverse normalization."""

        if x.ndim == 1:
            if x.shape[0] != len(keys):
                raise ValueError("Shape mismatch")
            return self._denormalize_columns(x.unsqueeze(0), keys)[0]

        return self._denormalize_columns(x, keys)

    def normalize_dt_from_phys(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """Normalize physical Δt (seconds) to [0, 1] using log10-minmax."""

        if not torch.is_floating_point(dt_phys):
            dt_phys = dt_phys.to(torch.float32)

        self._require_strictly_positive(dt_phys, context="dt normalization")
        dt_log = torch.log10(dt_phys)
        dt_norm = (dt_log - self.dt_spec.log_min) / self.dt_spec.range_log
        return dt_norm

    def validate_dt_norm(self, dt_norm: torch.Tensor) -> None:
        """Hard error if any normalized dt values are outside [0, 1]."""
        finite_mask = torch.isfinite(dt_norm)
        if not finite_mask.all():
            bad_count = int((~finite_mask).sum().item())
            raise ValueError(f"Normalized dt contains non-finite values (count={bad_count})")

        below = dt_norm < 0.0
        above = dt_norm > 1.0
        bad_mask = below | above
        if torch.any(bad_mask):
            lo = float(dt_norm.min())
            hi = float(dt_norm.max())
            bad_flat = torch.nonzero(bad_mask.reshape(-1), as_tuple=False).reshape(-1)
            bad_count = int(bad_flat.numel())
            sample_n = min(5, bad_count)
            flat = dt_norm.reshape(-1)
            sample_vals = [float(flat[int(i)].item()) for i in bad_flat[:sample_n]]
            raise ValueError(
                f"Normalized dt out of range [0, 1]: min={lo:.6g}, max={hi:.6g}. "
                f"n_bad={bad_count}, sample_bad={sample_vals}. "
                "This indicates dt extrapolation beyond the trained range."
            )

    def denormalize_dt_to_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized Δt back to physical units (seconds)."""

        if not torch.is_floating_point(dt_norm):
            dt_norm = dt_norm.to(torch.float32)

        self.validate_dt_norm(dt_norm)

        dt_log = self.dt_spec.log_min + dt_norm * self.dt_spec.range_log
        return torch.pow(10.0, dt_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stat_vector(
        self,
        keys: Sequence[str],
        stat_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        keys_tuple = tuple(keys)
        cache_key = (keys_tuple, str(stat_name), str(device), dtype)
        cached = self._vec_cache.get(cache_key)
        if cached is not None:
            return cached

        values: list[float] = []
        for key in keys_tuple:
            stats = self.per_key_stats.get(key)
            if stats is None:
                raise KeyError(f"Missing normalization stats for key: {key}")
            if stat_name not in stats:
                raise KeyError(f"Missing normalization stat '{stat_name}' for key: {key}")
            values.append(float(stats[stat_name]))

        vec = torch.tensor(values, device=device, dtype=dtype)
        self._vec_cache[cache_key] = vec
        return vec

    @staticmethod
    def _require_strictly_positive(x: torch.Tensor, *, context: str) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite value encountered during {context}")
        if torch.any(x <= 0):
            min_val = float(x.min())
            raise ValueError(
                f"Non-positive value encountered during {context}: min={min_val:.6g}"
            )

    def _normalize_columns(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Expected 2D+")
        if x.shape[-1] != len(keys):
            raise ValueError("Shape mismatch")

        methods = [self.methods.get(k) for k in keys]
        if any(m is None for m in methods):
            raise KeyError("Missing normalization method")

        # Fast path: all columns share the same method.
        m0 = methods[0]
        if all(m == m0 for m in methods):
            if m0 == "standard":
                means = self._get_stat_vector(keys, "mean", device=x.device, dtype=x.dtype)
                stds = self._get_stat_vector(keys, "std", device=x.device, dtype=x.dtype)
                if torch.any(stds <= 0):
                    raise ValueError("Invalid std")
                if torch.any(stds < self.min_std):
                    raise ValueError("std below min_std")
                return (x - means) / stds

            if m0 == "min-max":
                vmins = self._get_stat_vector(keys, "min", device=x.device, dtype=x.dtype)
                vmaxs = self._get_stat_vector(keys, "max", device=x.device, dtype=x.dtype)
                rng = vmaxs - vmins
                if torch.any(rng <= 0):
                    raise ValueError("Invalid min-max range")
                return (x - vmins) / rng

            if m0 == "log-standard":
                log_means = self._get_stat_vector(keys, "log_mean", device=x.device, dtype=x.dtype)
                log_stds = self._get_stat_vector(keys, "log_std", device=x.device, dtype=x.dtype)
                if torch.any(log_stds <= 0):
                    raise ValueError("Invalid log_std")
                if torch.any(log_stds < self.min_std):
                    raise ValueError("log_std below min_std")
                self._require_strictly_positive(x, context="log-standard normalization")
                x_log = torch.log10(x)
                return (x_log - log_means) / log_stds

            if m0 == "log-min-max":
                log_mins = self._get_stat_vector(keys, "log_min", device=x.device, dtype=x.dtype)
                log_maxs = self._get_stat_vector(keys, "log_max", device=x.device, dtype=x.dtype)
                rng = log_maxs - log_mins
                if torch.any(rng <= 0):
                    raise ValueError("Invalid log-min-max range")
                self._require_strictly_positive(x, context="log-min-max normalization")
                x_log = torch.log10(x)
                return (x_log - log_mins) / rng

        # Fallback: per-column methods.
        out = x.clone()
        for col, key in enumerate(keys):
            method = self.methods.get(key)
            if method is None:
                raise KeyError(f"Missing normalization method for key: {key}")

            stats = self.per_key_stats.get(key)
            if stats is None:
                raise KeyError(f"Missing normalization stats for key: {key}")

            v = out[..., col]

            if method == "standard":
                mean = float(stats["mean"])
                std = float(stats["std"])
                if std <= 0.0:
                    raise ValueError("Invalid std")
                if std < self.min_std:
                    raise ValueError(f"std below min_std for key: {key}")
                out[..., col] = (v - mean) / std

            elif method == "min-max":
                vmin = float(stats["min"])
                vmax = float(stats["max"])
                rng = vmax - vmin
                if rng <= 0.0:
                    raise ValueError("Invalid min-max range")
                out[..., col] = (v - vmin) / rng

            elif method == "log-standard":
                log_mean = float(stats["log_mean"])
                log_std = float(stats["log_std"])
                if log_std <= 0.0:
                    raise ValueError("Invalid log_std")
                if log_std < self.min_std:
                    raise ValueError(f"log_std below min_std for key: {key}")
                self._require_strictly_positive(v, context=f"log-standard normalization for key '{key}'")
                v_log = torch.log10(v)
                out[..., col] = (v_log - log_mean) / log_std

            elif method == "log-min-max":
                log_min = float(stats["log_min"])
                log_max = float(stats["log_max"])
                rng = log_max - log_min
                if rng <= 0.0:
                    raise ValueError("Invalid log-min-max range")
                self._require_strictly_positive(v, context=f"log-min-max normalization for key '{key}'")
                v_log = torch.log10(v)
                out[..., col] = (v_log - log_min) / rng

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        return out

    def _denormalize_columns(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Expected 2D+")
        if x.shape[-1] != len(keys):
            raise ValueError("Shape mismatch")

        methods = [self.methods.get(k) for k in keys]
        if any(m is None for m in methods):
            raise KeyError("Missing normalization method")

        m0 = methods[0]
        if all(m == m0 for m in methods):
            if m0 == "standard":
                means = self._get_stat_vector(keys, "mean", device=x.device, dtype=x.dtype)
                stds = self._get_stat_vector(keys, "std", device=x.device, dtype=x.dtype)
                if torch.any(stds <= 0):
                    raise ValueError("Invalid std")
                if torch.any(stds < self.min_std):
                    raise ValueError("std below min_std")
                return x * stds + means

            if m0 == "min-max":
                vmins = self._get_stat_vector(keys, "min", device=x.device, dtype=x.dtype)
                vmaxs = self._get_stat_vector(keys, "max", device=x.device, dtype=x.dtype)
                rng = vmaxs - vmins
                if torch.any(rng <= 0):
                    raise ValueError("Invalid min-max range")
                return x * rng + vmins

            if m0 == "log-standard":
                log_means = self._get_stat_vector(keys, "log_mean", device=x.device, dtype=x.dtype)
                log_stds = self._get_stat_vector(keys, "log_std", device=x.device, dtype=x.dtype)
                if torch.any(log_stds <= 0):
                    raise ValueError("Invalid log_std")
                if torch.any(log_stds < self.min_std):
                    raise ValueError("log_std below min_std")
                log_v = x * log_stds + log_means
                return torch.pow(10.0, log_v)

            if m0 == "log-min-max":
                log_mins = self._get_stat_vector(keys, "log_min", device=x.device, dtype=x.dtype)
                log_maxs = self._get_stat_vector(keys, "log_max", device=x.device, dtype=x.dtype)
                rng = log_maxs - log_mins
                if torch.any(rng <= 0):
                    raise ValueError("Invalid log-min-max range")
                log_v = x * rng + log_mins
                return torch.pow(10.0, log_v)

        out = x.clone()
        for col, key in enumerate(keys):
            method = self.methods.get(key)
            if method is None:
                raise KeyError(f"Missing normalization method for key: {key}")

            stats = self.per_key_stats.get(key)
            if stats is None:
                raise KeyError(f"Missing normalization stats for key: {key}")

            v = out[..., col]

            if method == "standard":
                mean = float(stats["mean"])
                std = float(stats["std"])
                if std <= 0.0:
                    raise ValueError("Invalid std")
                if std < self.min_std:
                    raise ValueError(f"std below min_std for key: {key}")
                out[..., col] = v * std + mean

            elif method == "min-max":
                vmin = float(stats["min"])
                vmax = float(stats["max"])
                rng = vmax - vmin
                if rng <= 0.0:
                    raise ValueError("Invalid min-max range")
                out[..., col] = v * rng + vmin

            elif method == "log-standard":
                log_mean = float(stats["log_mean"])
                log_std = float(stats["log_std"])
                if log_std <= 0.0:
                    raise ValueError("Invalid log_std")
                if log_std < self.min_std:
                    raise ValueError(f"log_std below min_std for key: {key}")
                log_v = v * log_std + log_mean
                out[..., col] = torch.pow(10.0, log_v)

            elif method == "log-min-max":
                log_min = float(stats["log_min"])
                log_max = float(stats["log_max"])
                rng = log_max - log_min
                if rng <= 0.0:
                    raise ValueError("Invalid log-min-max range")
                log_v = v * rng + log_min
                out[..., col] = torch.pow(10.0, log_v)

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        return out
