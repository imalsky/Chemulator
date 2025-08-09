#!/usr/bin/env python3
import logging
from typing import Dict, List, Any, Tuple
import torch

class NormalizationHelper:
    """
    Applies pre-computed normalization statistics to torch tensors based on a
    flexible, config-driven scheme. Supports:
      - "log-standard"  : log10(x) -> standardize
      - "standard"      : standardize
      - "log-min-max"   : time only, tau = ln(1 + t/tau0), then min-max
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

        # Core stats
        self.methods: Dict[str, str] = self.stats.get("normalization_methods", {})
        self.per_key_stats: Dict[str, Dict[str, float]] = self.stats.get("per_key_stats", {})
        self.time_norm = self.stats.get("time_normalization", None)  # {"tau0","tmin","tmax","time_transform":"log-min-max"}

        # In sequence mode, species are already log10-transformed in the shards.
        self.inputs_already_logged = bool(self.time_norm) or bool(self.config.get("data", {}).get("sequence_mode", False))
        if self.inputs_already_logged:
            self.logger.info("NormalizationHelper: sequence-mode detected; species are already in log10 space.")

        # Scalars
        self.epsilon = torch.tensor(self.norm_config.get("epsilon", 1e-30), dtype=self.dtype, device=self.device)
        self.clamp_val = float(self.norm_config.get("clamp_value", 50.0))
        self.min_std = float(self.norm_config.get("min_std", 1e-10))

        # Constants
        self._ten = torch.tensor(10.0, dtype=self.dtype, device=self.device)

        # Time constants (if available)
        if self.time_norm:
            self._tau0 = torch.tensor(float(self.time_norm["tau0"]), dtype=self.dtype, device=self.device)
            # NOTE: stats use *tau* bounds: tau_min/tau_max = ln(1 + t/tau0)
            self._tau_min = torch.tensor(float(self.time_norm["tmin"]), dtype=self.dtype, device=self.device)
            self._tau_max = torch.tensor(float(self.time_norm["tmax"]), dtype=self.dtype, device=self.device)
            self._tau_range = torch.clamp(self._tau_max - self._tau_min, min=1e-12)

    # ---------- helpers ----------
    def _get_params(self, var_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Build mean, std, and methods tensors aligned to var_list."""
        means, stds, methods = [], [], []
        for var in var_list:
            method = self.methods.get(var, "none")
            s = self.per_key_stats.get(var, {})
            if method == "log-standard":
                mean = s.get("log_mean", 0.0)
                std  = s.get("log_std", 1.0)
            elif method == "standard":
                mean = s.get("mean", 0.0)
                std  = s.get("std", 1.0)
            else:
                mean = 0.0
                std  = 1.0
            means.append(float(mean))
            stds.append(max(float(std), self.min_std))
            methods.append(method)
        means_t = torch.tensor(means, dtype=self.dtype, device=self.device)
        stds_t  = torch.tensor(stds,  dtype=self.dtype, device=self.device)
        return means_t, stds_t, methods

    # ---------- time normalization ----------
    def _time_to_unit(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize raw time t -> [0,1] using tau = ln(1 + t/tau0), min/max over tau."""
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")
        tau = torch.log1p(t / self._tau0)  # natural log
        return (tau - self._tau_min) / self._tau_range

    def _unit_to_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of _time_to_unit."""
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")
        tau = t_norm * self._tau_range + self._tau_min
        return self._tau0 * torch.expm1(tau)

    # ---------- public API ----------
    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Normalize a tensor shaped (..., D) where D == len(var_list).
        Applies per-dimension transforms based on method:
          - "log-standard" : ( (log10 x) or pass-through if already logged ) -> standardize -> clamp
          - "standard"     : standardize -> clamp
          - "log-min-max"  : time transform (ln1p + min-max) -> clamp to [0,1] then to +/- clamp_val (no-op if <=1)
          - "none"         : pass-through (then clamp)
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data last dim {data.shape[-1]} != var_list length {len(var_list)}")

        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)

        means, stds, methods = self._get_params(var_list)
        out = data.clone()

        for i, method in enumerate(methods):
            col = out[..., i]
            if method == "log-standard":
                if not self.inputs_already_logged:
                    col = torch.log10(col.clamp_min(self.epsilon))
                col = (col - means[i]) / stds[i]
            elif method == "standard":
                col = (col - means[i]) / stds[i]
            elif method == "log-min-max":
                # time variable: expects *raw* t, never log10
                col = self._time_to_unit(col)
                # keep in [0,1]; afterwards global clamp is harmless
                col = torch.clamp(col, 0.0, 1.0)
            elif method == "none":
                # no transform
                pass
            else:
                # Unknown method -> no transform but warn once.
                # (Avoid accidental log10 on time due to substring matches.)
                self.logger.warning(f"Unknown normalization method '{method}' for var '{var_list[i]}'; passing through.")
            out[..., i] = col

        return torch.clamp(out, -self.clamp_val, self.clamp_val)

    def denormalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Invert normalize() for supported methods.
        NOTE: For species in sequence mode, this returns values in **log10 space** when method == "log-standard".
              If you need linear units, exponentiate with 10**x after calling this.
        """
        if data.shape[-1] != len(var_list):
            raise ValueError(f"Data last dim {data.shape[-1]} != var_list length {len(var_list)}")

        if data.device != self.device or data.dtype != self.dtype:
            data = data.to(device=self.device, dtype=self.dtype)

        means, stds, methods = self._get_params(var_list)
        out = data.clone()

        for i, method in enumerate(methods):
            col = out[..., i]
            if method == "log-standard":
                col = col * stds[i] + means[i]        # back to log10 space
                if not self.inputs_already_logged:
                    col = torch.pow(self._ten, col)   # back to linear units
            elif method == "standard":
                col = col * stds[i] + means[i]
            elif method == "log-min-max":
                # back to raw t
                col = torch.clamp(col, 0.0, 1.0)
                col = self._unit_to_time(col)
            elif method == "none":
                pass
            else:
                self.logger.warning(f"Unknown normalization method '{method}' for var '{var_list[i]}'; passing through.")
            out[..., i] = col

        return out

    # Convenience methods if you keep time separate in your dataset code
    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        return self._time_to_unit(t.to(device=self.device, dtype=self.dtype))

    def denormalize_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        return self._unit_to_time(t_norm.to(device=self.device, dtype=self.dtype))
