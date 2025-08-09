#!/usr/bin/env python3
import logging
from typing import Dict, List, Any, Tuple
import torch

class NormalizationHelper:
    """
    Applies pre-computed normalization statistics to torch tensors based on a
    flexible, config-driven scheme. Supports:
      - "log-standard"  : log10(x) -> standardize
      - "standard"      : standardize (subtract off mean, divide by std)
      - "time-norm"   : time only, tau = ln(1 + t/tau0), then min-max
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
        raw_methods = self.stats.get("normalization_methods", {})
        self.methods: Dict[str, str] = {k: ("none" if v is None else str(v).lower()) for k, v in raw_methods.items()}
        self.per_key_stats: Dict[str, Dict[str, float]] = self.stats.get("per_key_stats", {})
        self.time_norm = self.stats.get("time_normalization", None)
        self.time_var = config.get("data", {}).get("time_variable", "t")
        self.time_method = (config.get("normalization", {}).get("methods", {}).get(self.time_var, "log-min-max"))

        # In sequence mode, species are already log10-transformed in the shards.
        # Build a per-variable set so only species skip the extra log10; globals won't.
        data_cfg = config.get("data", {})
        seq_mode = bool(data_cfg.get("sequence_mode", False))
        species_vars = list(data_cfg.get("species_variables", []))
        self._already_logged_vars = set(species_vars) if seq_mode else set()

        self.inputs_already_logged = bool(self.time_norm) or seq_mode
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
            # paper params
            self._tau0 = torch.tensor(float(self.time_norm["tau0"]), dtype=self.dtype, device=self.device)
            self._tau_min = torch.tensor(float(self.time_norm["tmin"]), dtype=self.dtype, device=self.device)
            self._tau_max = torch.tensor(float(self.time_norm["tmax"]), dtype=self.dtype, device=self.device)
            self._tau_range = torch.clamp(self._tau_max - self._tau_min, min=1e-12)

            # log-min-max params (raw)
            tmin_raw = float(self.time_norm.get("tmin_raw", 0.0))
            tmax_raw = float(self.time_norm.get("tmax_raw", 1.0))
            eps = float(self.norm_config.get("epsilon", 1e-30))
            lo = max(tmin_raw, eps)
            hi = max(tmax_raw, lo + eps)

            self._tlog_min = torch.log10(torch.tensor(lo, dtype=self.dtype, device=self.device))
            self._tlog_max = torch.log10(torch.tensor(hi, dtype=self.dtype, device=self.device))
            self._tlog_range = torch.clamp(self._tlog_max - self._tlog_min, min=1e-12)


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
    

    def _time_to_unit(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize raw time to [0,1] per configured method."""
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")

        if self.time_method == "time-norm":
            tau = torch.log1p(t / self._tau0)
            return (tau - self._tau_min) / self._tau_range

        if self.time_method == "log-min-max":
            eps = self.epsilon
            tlog = torch.log10(torch.clamp(t, min=eps))
            return (tlog - self._tlog_min) / self._tlog_range

        raise ValueError(f"Unknown time normalization method: {self.time_method}")

    def _unit_to_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Invert _time_to_unit back to RAW time."""
        if self.time_norm is None:
            raise RuntimeError("Time normalization stats not found.")

        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        if self.time_method == "time-norm":
            tau = t_norm * self._tau_range + self._tau_min
            return self._tau0 * torch.expm1(tau)

        if self.time_method == "log-min-max":
            tlog = t_norm * self._tlog_range + self._tlog_min
            return torch.pow(self._ten, tlog)

        raise ValueError(f"Unknown time normalization method: {self.time_method}")


    def normalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Normalize a tensor shaped (..., D) where D == len(var_list).
        Applies per-dimension transforms based on method:
        - "log-standard" : (log10 x) unless already in log space -> standardize -> clamp
        - "standard"     : standardize -> clamp
        - "time-norm"    : time transform (ln1p + min-max) -> clamp to [0,1]
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
                varname = var_list[i]
                # Only skip log10 for variables that were pre-logged (species in sequence mode)
                if varname not in self._already_logged_vars:
                    col = torch.log10(col.clamp_min(self.epsilon))
                col = (col - means[i]) / stds[i]
            elif method == "standard":
                col = (col - means[i]) / stds[i]
            elif method == "time-norm":
                col = self._time_to_unit(col)
                col = torch.clamp(col, 0.0, 1.0)
            elif method == "none":
                pass
            else:
                self.logger.warning(f"Unknown normalization method '{method}' for var '{var_list[i]}'; passing through.")
            out[..., i] = col

        return torch.clamp(out, -self.clamp_val, self.clamp_val)

    def denormalize(self, data: torch.Tensor, var_list: List[str]) -> torch.Tensor:
        """
        Invert normalize() for all supported methods, returning RAW values:
        - log-standard -> linear space (10**(x*std+mean))
        - standard     -> linear space (x*std+mean)
        - time-norm  -> raw time (inverse tau min-max)
        - none         -> pass-through
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
                # back to log10 space, then ALWAYS exponentiate to linear space
                col = col * stds[i] + means[i]     # log10(x)
                col = torch.pow(self._ten, col)    # -> x (raw linear)
            elif method == "standard":
                col = col * stds[i] + means[i]     # -> raw linear
            elif method == "time-norm":
                # from [0,1] back to raw time
                col = torch.clamp(col, 0.0, 1.0)
                col = self._unit_to_time(col)      # -> raw time
            elif method == "none":
                pass                                # already raw
            else:
                self.logger.warning(f"Unknown normalization method '{method}' for var '{var_list[i]}'; passing through.")
            out[..., i] = col

        return out


    # Convenience methods if you keep time separate in your dataset code
    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        return self._time_to_unit(t.to(device=self.device, dtype=self.dtype))

    def denormalize_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        return self._unit_to_time(t_norm.to(device=self.device, dtype=self.dtype))
