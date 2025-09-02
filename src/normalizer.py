#!/usr/bin/env python3
"""
Normalization helpers.

Goals of this refactor (minimal API disruption):
- Keep the same core responsibilities: load persisted stats, normalize/denormalize tensors.
- Centralize time/Δt math so Dataset and Model do not implement normalization logic.
- Ensure *all* numeric decisions (epsilon, min_std, clamp_value, per-key methods/stats)
  are read from the persisted normalization.json produced by preprocessing.

Key ideas
---------
Let x be a physical-space variable. Normalization methods supported:
- "standard"
    x_norm = (x - mean) / max(std, min_std)
- "min-max"
    x_norm = (x - min) / max(max - min, tiny)
- "log-standard"
    x_norm = (log10(max(x, epsilon)) - log_mean) / max(log_std, min_std)
- "log-min-max"
    x_norm = (log10(max(x, epsilon)) - log_min) / max(log_max - log_min, tiny)

The inverse operations are implemented correspondingly.

Centralized time/Δt:
- Time (absolute) uses the per-key method configured for `time_variable` (e.g., "t_time").
- Δt uses method "log-min-max" with parameters written by preprocessing under the
  "dt" key (recommended). If these are absent, we *gracefully* fall back to the time
  variable's log bounds inferred from stats, preserving historical behavior.

Numerical stability:
- epsilon: lower floor before taking log10 to avoid log(0).
- min_std: lower floor on std for "standard" / "log-standard".
- clamp_value: (optional) clamp |x_norm| ≤ clamp_value to bound outliers (no-op if None).

Everything is vectorized and torch-native, respecting device/dtype where possible.
"""

from __future__ import annotations

from typing import Dict, Any, List, Sequence, Optional

import torch


# ------------------------------- Utilities -----------------------------------

def _as_list(x: Sequence[str] | str) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


def _device_of(*tensors: torch.Tensor) -> torch.device:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def _dtype_of(*tensors: torch.Tensor) -> torch.dtype:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.dtype
    return torch.float32


# ----------------------------- Normalizer Core -------------------------------

class NormalizationHelper:
    """
    Wraps a saved normalization manifest (normalization.json) and provides:
      - normalize(x, keys)
      - denormalize(x, keys)
      - normalize_time(t_abs)
      - normalize_dt_from_phys(dt_phys)
      - make_dt_norm_table(time_grid_phys, min_steps, max_steps)

    Assumes preprocessing wrote per-key stats and methods into the JSON manifest.
    """

    def __init__(self, manifest: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Args
        ----
        manifest: dict from normalization.json, with entries like:
            {
                "per_key_stats": {
                    "t_time": { "log_min": ..., "log_max": ... , "mean": ..., "std": ... },
                    "P": { "log_min": ..., "log_max": ... },
                    "T": { "mean": ..., "std": ... },
                    ...
                },
                "normalization_methods": {
                    "t_time": "log-min-max",
                    "P": "log-min-max",
                    "T": "standard",
                    ...
                },
                "epsilon": 1e-30,
                "min_std": 1e-10,
                "clamp_value": 50.0,
                // Optional Δt key written by preprocessing (recommended):
                "dt": { "method": "log-min-max", "log_min": ..., "log_max": ... }
            }
        device: torch device used for internal tensors during on-the-fly ops.
        """
        self.manifest = manifest or {}
        self.per_key_stats: Dict[str, Dict[str, float]] = dict(self.manifest.get("per_key_stats", {}))
        self.methods: Dict[str, str] = dict(self.manifest.get("normalization_methods", {}))

        # Global numeric knobs (single source of truth)
        self.epsilon: float = float(self.manifest.get("epsilon", 1e-30))
        self.min_std: float = float(self.manifest.get("min_std", 1e-10))
        self.clamp_value: Optional[float] = self.manifest.get("clamp_value", None)
        if self.clamp_value is not None:
            self.clamp_value = float(self.clamp_value)

        # Optional centralized Δt spec (preferred)
        self.dt_spec: Optional[Dict[str, float]] = self.manifest.get("dt", None)
        if self.dt_spec is not None:
            # Ensure expected fields exist for log-min-max
            self.dt_spec = dict(self.dt_spec)
            self.dt_spec.setdefault("method", "log-min-max")

        self.device = device or torch.device("cpu")

    # ------------------------- Generic normalize API -------------------------

    def normalize(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """
        Normalize a 2D matrix x [:, len(keys)] with per-key methods.
        The i-th column uses method for keys[i]. No inplace modification.
        """
        if x.ndim == 1:
            # Treat as single column vector
            keys = _as_list(keys)
            if len(keys) != 1:
                raise ValueError(f"1D input requires a single key, got {len(keys)}")
            return self._normalize_column(x.unsqueeze(-1), keys)[..., 0]

        return self._normalize_column(x, keys)

    def denormalize(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """
        Inverse of normalize: x is normalized matrix [:, len(keys)].
        """
        if x.ndim == 1:
            keys = _as_list(keys)
            if len(keys) != 1:
                raise ValueError(f"1D input requires a single key, got {len(keys)}")
            return self._denormalize_column(x.unsqueeze(-1), keys)[..., 0]

        return self._denormalize_column(x, keys)

    # ------------------------ Time / Δt convenience --------------------------

    def normalize_time(self, t_abs: torch.Tensor, time_key: str = "t_time") -> torch.Tensor:
        """
        Normalize absolute time using the configured method for `time_key`.
        Typically this is "log-min-max" with per-key stats for t_time.
        """
        return self.normalize(t_abs, [time_key])

    def normalize_dt_from_phys(self, dt_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize a physical Δt tensor using centralized Δt rules.

        Preferred path:
          - Use manifest["dt"] with method "log-min-max": log_min/log_max on Δt.
        Fallback path (for backward compatibility):
          - If no "dt" section exists, we infer log bounds from the time key stats
            (e.g., t_time.log_min/log_max). This preserves historical behavior.

        Returns
        -------
        dt_norm : same shape as dt_phys, dtype preserved where possible
        """
        dev = _device_of(dt_phys) or self.device
        dtype = _dtype_of(dt_phys)

        # Preferred: explicit Δt spec
        if self.dt_spec is not None and str(self.dt_spec.get("method", "log-min-max")) == "log-min-max":
            log_min = float(self.dt_spec.get("log_min", -3.0))
            log_max = float(self.dt_spec.get("log_max", 8.0))
        else:
            # Fallback: derive from time variable stats if present
            # We use the broadest possible bounds consistent with prior usage:
            t_stats = self.per_key_stats.get("t_time", {})
            log_min = float(t_stats.get("log_min", -3.0))
            log_max = float(t_stats.get("log_max", 8.0))

        denom = max(log_max - log_min, 1e-12)

        # log-min-max on Δt with epsilon floor before log
        dt_phys = dt_phys.to(device=dev, dtype=torch.float64)  # precise log10() domain
        dt_phys = torch.clamp(dt_phys, min=self.epsilon)
        dt_log = torch.log10(dt_phys)
        dt_norm = (dt_log - log_min) / denom
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0).to(dtype=dtype)
        return dt_norm

    @torch.no_grad()
    def make_dt_norm_table(
        self,
        time_grid_phys: torch.Tensor,
        min_steps: int,
        max_steps: int,
    ) -> torch.Tensor:
        """
        Build a [T, T] lookup table L where L[i, j] = normalized Δt for j > i,
        and L[i, j] = 0 for j ≤ i (unused), using the same Δt rules as
        `normalize_dt_from_phys`.

        Math:
          Δt_ij = max(t[j] - t[i], epsilon)
          L[i, j] = (log10(Δt_ij) - log_min) / (log_max - log_min)
                    clipped to [0, 1]; only valid for min_steps ≤ (j - i) ≤ max_steps.

        Notes:
          - This is a convenience to *precompute* Δt_norm for all (i, j), so the
            Dataset can simply index L[i, j] in the hot path with O(1) work.
          - We compute in float64 for robust log differences on stiff grids,
            then cast down to float32 at the end.
        """
        if min_steps < 1:
            raise ValueError("min_steps must be ≥ 1.")
        if max_steps < min_steps:
            raise ValueError("max_steps must be ≥ min_steps.")

        dev = _device_of(time_grid_phys) or self.device
        t = time_grid_phys.to(device=dev, dtype=torch.float64)  # [T] absolute physical time
        T = int(t.numel())

        i = t.view(1, T)               # [1, T]
        j = t.view(T, 1)               # [T, 1]
        dt = torch.clamp(j - i, min=self.epsilon)  # [T, T] Δt_ij

        # Use the centralized Δt normalization
        l = self.normalize_dt_from_phys(dt)        # [T, T] in [0,1], dtype preserved by normalize_dt_from_phys

        # Zero out invalid regions (below min_steps, above max_steps, and diagonal/j<i)
        # We strictly enforce i<j and min_steps ≤ (j - i) ≤ max_steps
        steps = torch.arange(T, device=dev, dtype=torch.int64)
        di = steps.view(1, T)
        dj = steps.view(T, 1)
        delta_idx = (dj - di)  # [T,T]
        mask_valid = (delta_idx >= min_steps) & (delta_idx <= max_steps)
        l = torch.where(mask_valid, l, torch.zeros_like(l))

        return l.to(dtype=torch.float32)

    # ----------------------- Internal column-wise ops ------------------------

    def _normalize_column(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """
        Normalize a matrix column-wise according to per-key methods/stats.
        x: [*, K] where K == len(keys)
        """
        if x.ndim < 2:
            raise ValueError("Expected a 2D tensor for column-wise normalization.")
        K = len(keys)
        if x.shape[-1] != K:
            raise ValueError(f"Mismatched last dim: x has {x.shape[-1]}, keys={K}")

        out = x.clone()
        dev = _device_of(x) or self.device
        dtype = _dtype_of(x)

        for col, key in enumerate(keys):
            method = str(self.methods.get(key, self.manifest.get("default_method", "log-standard")))
            stats = self.per_key_stats.get(key, {})
            col_x = out[..., col].to(device=dev, dtype=torch.float64)  # robust intermediate

            if method == "standard":
                mean = float(stats.get("mean", 0.0))
                std = max(float(stats.get("std", 1.0)), self.min_std)
                col_y = (col_x - mean) / std

            elif method == "min-max":
                mn = float(stats.get("min", 0.0))
                mx = float(stats.get("max", 1.0))
                denom = max(mx - mn, 1e-12)
                col_y = (col_x - mn) / denom

            elif method == "log-standard":
                # Fixed: use log_mean and log_std for log-standard method
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_y = (torch.log10(torch.clamp(col_x, min=self.epsilon)) - log_mean) / log_std

            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                denom = max(log_max - log_min, 1e-12)
                col_y = (torch.log10(torch.clamp(col_x, min=self.epsilon)) - log_min) / denom

            else:
                raise ValueError(f"Unknown normalization method for key '{key}': {method}")

            col_y = col_y.to(dtype=dtype)
            if self.clamp_value is not None and self.clamp_value > 0:
                col_y = torch.clamp(col_y, -self.clamp_value, self.clamp_value)

            out[..., col] = col_y

        return out

    def _denormalize_column(self, x: torch.Tensor, keys: Sequence[str]) -> torch.Tensor:
        """
        Denormalize a matrix column-wise according to per-key methods/stats.
        x: [*, K] where K == len(keys)
        """
        if x.ndim < 2:
            raise ValueError("Expected a 2D tensor for column-wise denormalization.")
        K = len(keys)
        if x.shape[-1] != K:
            raise ValueError(f"Mismatched last dim: x has {x.shape[-1]}, keys={K}")

        out = x.clone()
        dev = _device_of(x) or self.device
        dtype = _dtype_of(x)

        for col, key in enumerate(keys):
            method = str(self.methods.get(key, self.manifest.get("default_method", "log-standard")))
            stats = self.per_key_stats.get(key, {})
            col_x = out[..., col].to(device=dev, dtype=torch.float64)

            if method == "standard":
                mean = float(stats.get("mean", 0.0))
                std = max(float(stats.get("std", 1.0)), self.min_std)
                col_y = col_x * std + mean

            elif method == "min-max":
                mn = float(stats.get("min", 0.0))
                mx = float(stats.get("max", 1.0))
                denom = max(mx - mn, 1e-12)
                col_y = col_x * denom + mn

            elif method == "log-standard":
                # Fixed: use log_mean and log_std for log-standard method
                log_mean = float(stats.get("log_mean", 0.0))
                log_std = max(float(stats.get("log_std", 1.0)), self.min_std)
                col_log = col_x * log_std + log_mean
                col_y = torch.pow(10.0, col_log)  # invert log10

            elif method == "log-min-max":
                log_min = float(stats.get("log_min", -3.0))
                log_max = float(stats.get("log_max", 8.0))
                denom = max(log_max - log_min, 1e-12)
                col_log = col_x * denom + log_min
                col_y = torch.pow(10.0, col_log)

            else:
                raise ValueError(f"Unknown normalization method for key '{key}': {method}")

            out[..., col] = col_y.to(dtype=dtype)

        return out