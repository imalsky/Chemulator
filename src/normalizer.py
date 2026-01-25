#!/usr/bin/env python3
"""
normalizer.py - Normalization utilities for the flow-map training pipeline.

This module handles bidirectional conversion between physical and normalized
(z-space) values for chemical species concentrations and time variables.

Normalization Methods:
    - Species: log10-standardized
        z = (log10(y_phys) - log_mean) / log_std
        This transforms species concentrations (often spanning many orders of
        magnitude) into a standardized space suitable for neural network training.

    - Time (dt): log10 + min-max to [0, 1]
        dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)
        This normalizes variable timesteps to a bounded range while preserving
        the log-scale distribution typical of adaptive timestepping.

    - Globals: configurable (standard, min-max, or identity)

Manifest Structure:
    The normalization manifest (normalization.json) contains:
        - normalization_methods: {variable_name: method_string}
        - per_key_stats: {variable_name: {log_mean, log_std, log_min, log_max, ...}}
        - dt: {log_min, log_max} or dt_seconds: scalar

Usage:
    >>> manifest = json.load(open("normalization.json"))
    >>> helper = NormalizationHelper(manifest)
    >>> z = helper.normalize(y_physical, "H2O_evolution")
    >>> y_physical = helper.denormalize(z, "H2O_evolution")
    >>> dt_norm = helper.normalize_dt_from_phys(dt_seconds)

Why Log-Space Normalization?
    Chemical species concentrations often span 20+ orders of magnitude
    (e.g., 1e-30 to 1e-5 mole fractions). Working in log10 space:
    1. Compresses this huge dynamic range to ~25 log units
    2. Makes relative errors more meaningful (1 log unit ≈ factor of 10)
    3. Provides more uniform gradient magnitudes across species
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Union, List, Tuple, Optional

import numpy as np
import torch


# ==============================================================================
# Numerical Stability Constants
# ==============================================================================

# Relative tolerance for out-of-distribution warnings
_RELATIVE_TOLERANCE = 1e-6

# Minimum range to prevent division by zero
_MIN_RANGE_EPSILON = 1e-12

# Default minimum value before taking log10 (prevents -inf)
_DEFAULT_EPSILON = 1e-30

# Minimum standard deviation to prevent division by zero
_DEFAULT_MIN_STD = 1e-12


# ==============================================================================
# dt Specification
# ==============================================================================


@dataclass(frozen=True)
class DtSpec:
    """
    Specification for dt normalization bounds (in log10 seconds).

    The dt normalization maps physical timesteps to [0, 1] using:
        dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

    Attributes:
        log_min: Minimum log10(dt) in the training data
        log_max: Maximum log10(dt) in the training data

    Properties:
        phys_min: Minimum physical dt in seconds (10^log_min)
        phys_max: Maximum physical dt in seconds (10^log_max)
        log_range: Range in log10 space (for normalization denominator)

    Example:
        For dt_min=10s, dt_max=1000s:
        >>> spec = DtSpec(log_min=1.0, log_max=3.0)
        >>> spec.phys_min  # 10.0
        >>> spec.phys_max  # 1000.0
    """

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
        """Range in log10 space, with minimum epsilon for numerical stability."""
        return max(self.log_max - self.log_min, _MIN_RANGE_EPSILON)


def parse_dt_spec(dt_obj: Any) -> DtSpec:
    """
    Parse dt specification from normalization manifest.

    Supports multiple formats for backwards compatibility:
        - None: defaults to (0.0, 0.0) - effectively constant dt=1s
        - Scalar (seconds): constant dt, log_min = log_max = log10(scalar)
        - Dict with log_min/log_max: direct specification
        - Dict with dt_seconds: constant dt

    Args:
        dt_obj: dt specification from manifest (various formats)

    Returns:
        Parsed DtSpec with log_min and log_max

    Raises:
        ValueError: If dt_seconds is non-positive
        TypeError: If dt_obj format is not recognized
    """
    if dt_obj is None:
        return DtSpec(log_min=0.0, log_max=0.0)

    # Scalar: constant dt
    if isinstance(dt_obj, (int, float, np.floating, np.integer)):
        dt_s = float(dt_obj)
        if dt_s <= 0:
            raise ValueError(f"dt_seconds must be positive, got {dt_s}")
        lg = float(np.log10(dt_s))
        return DtSpec(log_min=lg, log_max=lg)

    # Dictionary: either log_min/log_max or dt_seconds
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

    raise TypeError(
        f"Unrecognized dt spec type: {type(dt_obj)}. "
        "Expected None, scalar, or dict with log_min/log_max or dt_seconds."
    )


# ==============================================================================
# Main Normalization Helper
# ==============================================================================



@dataclass(frozen=True)
class _VectorSpec:
    """
    Cached vectorized normalization parameters for an ordered set of keys.

    All tensor fields are stored on CPU and moved/cast to the input tensor's
    device/dtype on demand. Shapes are [S] where S=len(keys).
    """

    keys: Tuple[str, ...]
    method_id: torch.Tensor  # int64, [S]  0=identity,1=standard,2=min-max,3=log-standard,4=log-min-max
    mean: torch.Tensor
    inv_std: torch.Tensor
    min: torch.Tensor
    inv_range: torch.Tensor
    log_mean: torch.Tensor
    inv_log_std: torch.Tensor
    log_min: torch.Tensor
    inv_log_range: torch.Tensor
    has_log: bool

class NormalizationHelper:
    """
    Bidirectional normalization for species and time variables.

    This class provides methods to convert between physical values and
    normalized z-space values, using statistics computed during preprocessing.

    Supported normalization methods:
        - "identity" / "none": No transformation
        - "standard": (x - mean) / std
        - "min-max": (x - min) / (max - min) → [0, 1]
        - "log-standard" / "log10-standard": (log10(x) - log_mean) / log_std
        - "log-min-max": (log10(x) - log_min) / (log_max - log_min) → [0, 1]

    The appropriate method is determined by the manifest's normalization_methods
    dictionary, which maps variable names to method strings.

    Args:
        manifest: Normalization manifest dictionary containing:
            - normalization_methods: {var_name: method} or methods: {var_name: method}
            - per_key_stats: {var_name: stats_dict} or species_stats or stats
            - dt: {log_min, log_max} or dt_seconds
            - epsilon: (optional) minimum value before log
            - min_std: (optional) minimum std for division

    Attributes:
        manifest: Original manifest dictionary
        methods: Variable name → normalization method mapping
        per_key_stats: Variable name → statistics dictionary mapping
        epsilon: Minimum value before taking log10
        min_std: Minimum standard deviation for division
        dt_spec: DtSpec for timestep normalization

    Example:
        >>> helper = NormalizationHelper(manifest)
        >>> z = helper.normalize(torch.tensor([1e-10, 1e-5]), "H2O_evolution")
        >>> y = helper.denormalize(z, "H2O_evolution")
    """

    def __init__(self, manifest: Dict[str, Any]) -> None:
        """
        Initialize from normalization manifest.

        Args:
            manifest: Dictionary containing normalization methods, statistics,
                     and dt specification from preprocessing.
        """
        self.manifest = dict(manifest)

        # Support both newer and legacy key names for backwards compatibility
        self.methods: Dict[str, str] = dict(
            self.manifest.get(
                "normalization_methods", self.manifest.get("methods", {})
            )
        )

        # Support multiple legacy key names for per-key statistics
        # Priority: per_key_stats > species_stats > stats
        # This matches the fallback chain used in trainer.py's build_loss_buffers
        self.per_key_stats: Dict[str, Dict[str, Any]] = dict(
            self.manifest.get("per_key_stats")
            or self.manifest.get("species_stats")
            or self.manifest.get("stats", {})
        )

        # Numerical stability parameters
        self.epsilon = float(self.manifest.get("epsilon", _DEFAULT_EPSILON))
        self.min_std = float(self.manifest.get("min_std", _DEFAULT_MIN_STD))

        # Parse dt specification
        dt_obj = self.manifest.get("dt") or self.manifest.get("dt_seconds")
        self.dt_spec = parse_dt_spec(dt_obj)

        # Cache for vectorized multi-key normalization parameters
        self._vector_cache: Dict[Tuple[str, ...], _VectorSpec] = {}

    # ==========================================================================
    # dt Normalization
    # ==========================================================================

    def normalize_dt_from_phys(
        self, dt_phys: Union[torch.Tensor, np.ndarray, float]
    ) -> torch.Tensor:
        """
        Normalize physical timestep (seconds) to [0, 1] using log10 + min-max.

        The transformation is:
            dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

        Values outside the training range [dt_min, dt_max] are clamped and
        a warning is issued.

        Preserves floating-point dtype of tensor inputs; non-tensor inputs
        default to float32.

        Args:
            dt_phys: Physical time step(s) in seconds. Can be scalar, numpy
                    array, or torch tensor.

        Returns:
            Normalized dt tensor in [0, 1], same shape as input
        """
        input_dtype = dt_phys.dtype if isinstance(dt_phys, torch.Tensor) else None

        if not isinstance(dt_phys, torch.Tensor):
            dt_phys = torch.as_tensor(dt_phys)
        dt_phys64 = dt_phys.to(dtype=torch.float64)

        # Check for out-of-distribution values
        rtol = _RELATIVE_TOLERANCE
        phys_min_eff = self.dt_spec.phys_min * (1.0 - rtol)
        phys_max_eff = self.dt_spec.phys_max * (1.0 + rtol)

        ood_mask = (dt_phys64 < phys_min_eff) | (dt_phys64 > phys_max_eff)
        if torch.any(ood_mask):
            n_ood, n_tot = int(ood_mask.sum()), dt_phys64.numel()
            warnings.warn(
                f"[normalize_dt] {n_ood}/{n_tot} dt values outside training range "
                f"[{self.dt_spec.phys_min:.3e}, {self.dt_spec.phys_max:.3e}]. "
                "Clamping to range. This may affect model accuracy.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Clamp to valid range and compute normalized value
        dt_clamped = dt_phys64.clamp(
            min=max(self.epsilon, self.dt_spec.phys_min), max=self.dt_spec.phys_max
        )
        dt_log = torch.log10(dt_clamped)
        dt_norm = (dt_log - self.dt_spec.log_min) / self.dt_spec.log_range

        out = dt_norm.clamp(0.0, 1.0)
        if input_dtype is not None and input_dtype.is_floating_point:
            return out.to(dtype=input_dtype)
        return out.to(dtype=torch.float32)

    def denormalize_dt_to_phys(
        self, dt_norm: Union[torch.Tensor, np.ndarray, float]
    ) -> torch.Tensor:
        """
        Convert normalized dt back to physical seconds.

        The inverse transformation is:
            dt_phys = 10^(log_min + dt_norm * (log_max - log_min))

        Preserves floating-point dtype of tensor inputs; non-tensor inputs
        default to float32.

        Args:
            dt_norm: Normalized time step(s) in [0, 1]

        Returns:
            Physical dt tensor in seconds
        """
        input_dtype = dt_norm.dtype if isinstance(dt_norm, torch.Tensor) else None

        if not isinstance(dt_norm, torch.Tensor):
            dt_norm = torch.as_tensor(dt_norm)
        dt_norm64 = dt_norm.to(dtype=torch.float64).clamp(0.0, 1.0)

        dt_log = self.dt_spec.log_min + dt_norm64 * self.dt_spec.log_range
        dt_phys = 10.0**dt_log

        out = dt_phys.clamp(self.dt_spec.phys_min, self.dt_spec.phys_max)
        if input_dtype is not None and input_dtype.is_floating_point:
            return out.to(dtype=input_dtype)
        return out.to(dtype=torch.float32)

    # ==========================================================================
    # Variable Normalization
    # ==========================================================================

    def _get_method_and_stats(self, key: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get normalization method and statistics for a variable.

        Args:
            key: Variable name

        Returns:
            Tuple of (method_string, stats_dict)

        Raises:
            KeyError: If statistics for key are not found
        """
        method = str(self.methods.get(key, "standard") or "standard")
        stats = self.per_key_stats.get(key)

        if stats is None:
            raise KeyError(
                f"Missing stats for key '{key}' in normalization manifest. "
                f"Available keys: {list(self.per_key_stats.keys())[:10]}..."
            )

        return method, stats

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize a variable from physical to z-space.

        Applies the normalization method specified in the manifest for
        the given variable.

        Args:
            x: Physical values tensor
            key: Variable name for looking up normalization parameters

        Returns:
            Normalized (z-space) tensor

        Raises:
            KeyError: If key is not in the manifest
            ValueError: If normalization method is not recognized
        """
        method, stats = self._get_method_and_stats(key)

        # Identity: no transformation
        if method in ("identity", "none", ""):
            return x

        # Standard: (x - mean) / std
        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return (x - mu) / sd

        # Min-max: (x - min) / (max - min)
        if method == "min-max":
            mn, mx = float(stats["min"]), float(stats["max"])
            return (x - mn) / max(mx - mn, _MIN_RANGE_EPSILON)

        # Log-standard: (log10(x) - log_mean) / log_std
        if method in ("log-standard", "log10-standard"):
            mu_val = stats.get("log_mean", stats.get("log10_mean"))
            sd_val = stats.get("log_std", stats.get("log10_std"))
            if mu_val is None or sd_val is None:
                raise KeyError(
                    "Missing log statistics for normalization. Expected log_* or log10_* keys."
                )
            mu = float(mu_val)
            sd = max(float(sd_val), self.min_std)
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mu) / sd

        # Log-min-max: (log10(x) - log_min) / (log_max - log_min)
        if method == "log-min-max":
            mn_val = stats.get("log_min", stats.get("log10_min"))
            mx_val = stats.get("log_max", stats.get("log10_max"))
            if mn_val is None or mx_val is None:
                raise KeyError(
                    "Missing log-min-max statistics for normalization. Expected log_* or log10_* keys."
                )
            mn, mx = float(mn_val), float(mx_val)
            x_log = torch.log10(torch.clamp(x, min=self.epsilon))
            return (x_log - mn) / max(mx - mn, _MIN_RANGE_EPSILON)

        raise ValueError(
            f"Unknown normalization method '{method}' for key '{key}'. "
            "Supported: identity, standard, min-max, log-standard, log-min-max"
        )

    def denormalize(self, z: torch.Tensor, key: str) -> torch.Tensor:
        """
        Denormalize a variable from z-space to physical.

        Applies the inverse of the normalization method to convert
        back to physical units.

        Args:
            z: Normalized (z-space) tensor
            key: Variable name for looking up normalization parameters

        Returns:
            Physical values tensor

        Raises:
            KeyError: If key is not in the manifest
            ValueError: If normalization method is not recognized
        """
        method, stats = self._get_method_and_stats(key)

        # Identity: no transformation
        if method in ("identity", "none", ""):
            return z

        # Standard inverse: z * std + mean
        if method == "standard":
            mu = float(stats["mean"])
            sd = max(float(stats["std"]), self.min_std)
            return z * sd + mu

        # Min-max inverse: z * (max - min) + min
        if method == "min-max":
            mn, mx = float(stats["min"]), float(stats["max"])
            return z * max(mx - mn, _MIN_RANGE_EPSILON) + mn

        # Log-standard inverse: 10^(z * log_std + log_mean)
        if method in ("log-standard", "log10-standard"):
            mu_val = stats.get("log_mean", stats.get("log10_mean"))
            sd_val = stats.get("log_std", stats.get("log10_std"))
            if mu_val is None or sd_val is None:
                raise KeyError(
                    "Missing log statistics for inverse normalization. Expected log_* or log10_* keys."
                )
            mu = float(mu_val)
            sd = max(float(sd_val), self.min_std)
            return 10.0 ** (z * sd + mu)

        # Log-min-max inverse: 10^(z * (log_max - log_min) + log_min)
        if method == "log-min-max":
            mn_val = stats.get("log_min", stats.get("log10_min"))
            mx_val = stats.get("log_max", stats.get("log10_max"))
            if mn_val is None or mx_val is None:
                raise KeyError(
                    "Missing log-min-max statistics for inverse normalization. Expected log_* or log10_* keys."
                )
            mn, mx = float(mn_val), float(mx_val)
            return 10.0 ** (z * max(mx - mn, _MIN_RANGE_EPSILON) + mn)

        raise ValueError(
            f"Unknown normalization method '{method}' for key '{key}'. "
            "Supported: identity, standard, min-max, log-standard, log-min-max"
        )

    # ==========================================================================
    # Vectorized multi-key normalization (no per-key Python loops at call time)
    # ==========================================================================

    def _get_vector_spec(self, keys: List[str]) -> _VectorSpec:
        """Get or build cached vectorized parameters for an ordered key list."""
        key_tup = tuple(keys)
        spec = self._vector_cache.get(key_tup)
        if spec is not None:
            return spec

        S = len(keys)
        if S == 0:
            raise ValueError("keys must be non-empty")

        method_ids: List[int] = []
        means: List[float] = []
        inv_stds: List[float] = []
        mins: List[float] = []
        inv_ranges: List[float] = []
        log_means: List[float] = []
        inv_log_stds: List[float] = []
        log_mins: List[float] = []
        inv_log_ranges: List[float] = []

        has_log = False

        for k in keys:
            method, stats = self._get_method_and_stats(k)
            m = str(method).lower().strip()

            # Defaults (used for identity and as placeholders for unused params)
            mid = 0
            mean_k = 0.0
            inv_std_k = 1.0
            min_k = 0.0
            inv_range_k = 1.0
            log_mean_k = 0.0
            inv_log_std_k = 1.0
            log_min_k = 0.0
            inv_log_range_k = 1.0

            if m in ("identity", "none", ""):
                mid = 0
            elif m == "standard":
                mid = 1
                mean_k = float(stats["mean"])
                sd = max(float(stats["std"]), self.min_std)
                inv_std_k = 1.0 / sd
            elif m == "min-max":
                mid = 2
                min_k = float(stats["min"])
                mx = float(stats["max"])
                rng = max(mx - min_k, _MIN_RANGE_EPSILON)
                inv_range_k = 1.0 / rng
            elif m in ("log-standard", "log10-standard"):
                mid = 3
                has_log = True
                log_mean_val = stats.get("log_mean", stats.get("log10_mean"))
                log_std_val = stats.get("log_std", stats.get("log10_std"))
                if log_mean_val is None or log_std_val is None:
                    raise KeyError(
                        f"Missing log stats for key '{k}'. Expected log_* or log10_* entries."
                    )
                log_mean_k = float(log_mean_val)
                sd = max(float(log_std_val), self.min_std)
                inv_log_std_k = 1.0 / sd
            elif m == "log-min-max":
                mid = 4
                has_log = True
                log_min_val = stats.get("log_min", stats.get("log10_min"))
                log_max_val = stats.get("log_max", stats.get("log10_max"))
                if log_min_val is None or log_max_val is None:
                    raise KeyError(
                        f"Missing log-min-max stats for key '{k}'. Expected log_* or log10_* entries."
                    )
                log_min_k = float(log_min_val)
                mx = float(log_max_val)
                rng = max(mx - log_min_k, _MIN_RANGE_EPSILON)
                inv_log_range_k = 1.0 / rng
            else:
                raise ValueError(
                    f"Unknown normalization method '{method}' for key '{k}'. "
                    "Supported: identity, standard, min-max, log-standard, log-min-max"
                )

            method_ids.append(mid)
            means.append(mean_k)
            inv_stds.append(inv_std_k)
            mins.append(min_k)
            inv_ranges.append(inv_range_k)
            log_means.append(log_mean_k)
            inv_log_stds.append(inv_log_std_k)
            log_mins.append(log_min_k)
            inv_log_ranges.append(inv_log_range_k)

        spec = _VectorSpec(
            keys=key_tup,
            method_id=torch.tensor(method_ids, dtype=torch.int64),
            mean=torch.tensor(means, dtype=torch.float32),
            inv_std=torch.tensor(inv_stds, dtype=torch.float32),
            min=torch.tensor(mins, dtype=torch.float32),
            inv_range=torch.tensor(inv_ranges, dtype=torch.float32),
            log_mean=torch.tensor(log_means, dtype=torch.float32),
            inv_log_std=torch.tensor(inv_log_stds, dtype=torch.float32),
            log_min=torch.tensor(log_mins, dtype=torch.float32),
            inv_log_range=torch.tensor(inv_log_ranges, dtype=torch.float32),
            has_log=has_log,
        )

        self._vector_cache[key_tup] = spec
        return spec

    def normalize_many(
        self, x: Union[torch.Tensor, np.ndarray, float], keys: List[str]
    ) -> torch.Tensor:
        """Vectorized normalization for an ordered set of keys along the last dimension."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        squeeze = False
        if x.ndim == 0:
            if len(keys) != 1:
                raise ValueError(f"Scalar input requires exactly one key, got {len(keys)}")
            x = x.view(1)
            squeeze = True

        S = len(keys)
        if x.shape[-1] != S:
            raise ValueError(
                f"Expected last dimension size {S} for keys, got {x.shape[-1]}"
            )

        spec = self._get_vector_spec(keys)

        work_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        xw = x.to(dtype=work_dtype)

        view_shape = (1,) * (xw.ndim - 1) + (S,)
        mid = spec.method_id.to(device=xw.device).view(view_shape)
        mean = spec.mean.to(device=xw.device, dtype=work_dtype).view(view_shape)
        inv_std = spec.inv_std.to(device=xw.device, dtype=work_dtype).view(view_shape)
        mn = spec.min.to(device=xw.device, dtype=work_dtype).view(view_shape)
        inv_rng = spec.inv_range.to(device=xw.device, dtype=work_dtype).view(view_shape)
        log_mean = spec.log_mean.to(device=xw.device, dtype=work_dtype).view(view_shape)
        inv_log_std = spec.inv_log_std.to(device=xw.device, dtype=work_dtype).view(view_shape)
        log_min = spec.log_min.to(device=xw.device, dtype=work_dtype).view(view_shape)
        inv_log_rng = spec.inv_log_range.to(device=xw.device, dtype=work_dtype).view(view_shape)

        out = xw

        if torch.any(mid == 1):
            out = torch.where(mid == 1, (xw - mean) * inv_std, out)
        if torch.any(mid == 2):
            out = torch.where(mid == 2, (xw - mn) * inv_rng, out)

        if spec.has_log and (torch.any(mid == 3) or torch.any(mid == 4)):
            x_log = torch.log10(torch.clamp(xw, min=self.epsilon))
            if torch.any(mid == 3):
                out = torch.where(mid == 3, (x_log - log_mean) * inv_log_std, out)
            if torch.any(mid == 4):
                out = torch.where(mid == 4, (x_log - log_min) * inv_log_rng, out)

        out = out.to(dtype=x.dtype)
        if squeeze:
            return out.view(())
        return out

    def denormalize_many(
        self, z: Union[torch.Tensor, np.ndarray, float], keys: List[str]
    ) -> torch.Tensor:
        """Vectorized denormalization for an ordered set of keys along the last dimension."""
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)

        squeeze = False
        if z.ndim == 0:
            if len(keys) != 1:
                raise ValueError(f"Scalar input requires exactly one key, got {len(keys)}")
            z = z.view(1)
            squeeze = True

        S = len(keys)
        if z.shape[-1] != S:
            raise ValueError(
                f"Expected last dimension size {S} for keys, got {z.shape[-1]}"
            )

        spec = self._get_vector_spec(keys)

        work_dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
        zw = z.to(dtype=work_dtype)

        view_shape = (1,) * (zw.ndim - 1) + (S,)
        mid = spec.method_id.to(device=zw.device).view(view_shape)
        mean = spec.mean.to(device=zw.device, dtype=work_dtype).view(view_shape)
        inv_std = spec.inv_std.to(device=zw.device, dtype=work_dtype).view(view_shape)
        mn = spec.min.to(device=zw.device, dtype=work_dtype).view(view_shape)
        inv_rng = spec.inv_range.to(device=zw.device, dtype=work_dtype).view(view_shape)
        log_mean = spec.log_mean.to(device=zw.device, dtype=work_dtype).view(view_shape)
        inv_log_std = spec.inv_log_std.to(device=zw.device, dtype=work_dtype).view(view_shape)
        log_min = spec.log_min.to(device=zw.device, dtype=work_dtype).view(view_shape)
        inv_log_rng = spec.inv_log_range.to(device=zw.device, dtype=work_dtype).view(view_shape)

        out = zw

        if torch.any(mid == 1):
            out = torch.where(mid == 1, zw / inv_std + mean, out)
        if torch.any(mid == 2):
            out = torch.where(mid == 2, zw / inv_rng + mn, out)

        if spec.has_log and (torch.any(mid == 3) or torch.any(mid == 4)):
            log_val = torch.zeros_like(zw)
            if torch.any(mid == 3):
                log_val = torch.where(mid == 3, zw / inv_log_std + log_mean, log_val)
            if torch.any(mid == 4):
                log_val = torch.where(mid == 4, zw / inv_log_rng + log_min, log_val)
            out = torch.where((mid == 3) | (mid == 4), 10.0 ** log_val, out)

        out = out.to(dtype=z.dtype)
        if squeeze:
            return out.view(())
        return out
