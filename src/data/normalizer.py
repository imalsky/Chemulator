#!/usr/bin/env python3
"""
Data normalization module for chemical kinetics datasets.
Simplified version without ratio mode complexity.

Key points:
- In SEQUENCE MODE your pipeline already logs species once (x0_log, y_mat_log).
  This helper will NOT apply a second log/exp when `inputs_already_logged=True`.
- Time normalization is handled upstream (dataset). We skip time in sequence mode.
- Means/stds/min/max tensors are precomputed once and reused (no per-call sorting/stacking).
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch


DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP = 50.0


class NormalizationHelper:
    """Apply pre-computed normalization statistics to torch tensors."""
    def __init__(self,
                 stats: Dict[str, Any],
                 device: torch.device,
                 species_vars: List[str],
                 global_vars: List[str],
                 time_var: str,
                 config: Optional[Dict[str, Any]] = None):

        dtype_name = (config or {}).get("system", {}).get("dtype", "float32")
        self.torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32

        self.stats = stats
        self.device = device
        self.species_vars = species_vars
        self.global_vars = global_vars
        self.time_var = time_var

        self.n_species = len(species_vars)
        self.n_globals = len(global_vars)

        # From stats.json
        self.methods = stats["normalization_methods"]            # per-key method names
        self.per_key_stats = stats.get("per_key_stats", {})      # stats for species/globals

        # Sequence mode: species in shards are already log10; time is normalized upstream
        self.is_sequence_mode = bool(stats.get("time_normalization")) or \
                                bool((config or {}).get("data", {}).get("sequence_mode", False))
        # In sequence mode, inputs are already logged once; do NOT log again here.
        self.inputs_already_logged = self.is_sequence_mode

        # Scalars
        self.epsilon = torch.tensor(
            float(stats.get("epsilon", DEFAULT_EPSILON)),
            dtype=self.torch_dtype,
            device=self.device,
        )
        self.clamp_value = float(stats.get("clamp_value", DEFAULT_CLAMP))

        self.logger = logging.getLogger(__name__)

        # Will be filled by _precompute_parameters()
        self.norm_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.method_groups: Dict[str, List[str]] = {}
        self.col_indices: Dict[str, List[int]] = {}
        self.params_by_method: Dict[str, Dict[str, torch.Tensor]] = {}

        # Precompute per-method tensors and column orders
        self._precompute_parameters()

        # Cache constant 10.0 for pow10 operations
        self._ten = torch.tensor(10.0, dtype=self.torch_dtype, device=self.device)

    def _precompute_parameters(self) -> None:
        """Cache normalization parameters and column orders on device."""
        self.method_groups = {"standard": [], "log-standard": [],
                              "min-max": [], "log-min-max": [], "none": []}

        # Column positions in a profile: [species..., globals..., time]
        all_order = self.species_vars + self.global_vars + [self.time_var]
        var_to_col = {v: i for i, v in enumerate(all_order)}

        # Build var -> params on device
        for var, method in self.methods.items():
            # Skip time normalization inside the helper in sequence mode (handled upstream)
            if self.is_sequence_mode and var == self.time_var:
                self.method_groups["none"].append(var)
                continue

            # Stats for time are stored in stats["time_normalization"], not in per_key_stats
            # So if missing from per_key_stats, skip it here.
            if method == "none" or var not in self.per_key_stats:
                self.method_groups["none"].append(var)
                continue

            normalized = method.replace("_", "-")  # allow "log-standard" / "log-min-max"
            vs = self.per_key_stats[var]
            pars: Dict[str, torch.Tensor] = {"method": torch.tensor(0)}  # dummy holder

            if "standard" in normalized:
                k_mean = "log_mean" if "log" in normalized else "mean"
                k_std = "log_std" if "log" in normalized else "std"
                mean_t = torch.tensor(float(vs[k_mean]), dtype=self.torch_dtype, device=self.device)
                std_t = torch.tensor(float(vs[k_std]), dtype=self.torch_dtype, device=self.device).clamp_min(DEFAULT_MIN_STD)
                pars = {"mean": mean_t, "std": std_t}

            elif "min-max" in normalized:
                # Species/globals stats include min/max in per_key_stats (time uses separate dict; we skip time here)
                min_t = torch.tensor(float(vs["min"]), dtype=self.torch_dtype, device=self.device)
                max_t = torch.tensor(float(vs["max"]), dtype=self.torch_dtype, device=self.device)
                pars = {"min": min_t, "max": max_t}

            self.norm_params[var] = pars
            self.method_groups.setdefault(normalized, []).append(var)

        # Build sorted column indices per method and pre-stack parameters in column order
        self.col_indices = {}
        self.params_by_method = {}

        for method, var_list in self.method_groups.items():
            if not var_list or method == "none":
                continue

            cols = [var_to_col[v] for v in var_list]
            order = np.argsort(cols)
            cols_sorted = [cols[i] for i in order]
            vars_sorted = [var_list[i] for i in order]
            self.col_indices[method] = cols_sorted

            pack: Dict[str, torch.Tensor] = {}
            if "standard" in method:
                means = torch.stack([self.norm_params[v]["mean"] for v in vars_sorted])
                stds = torch.stack([self.norm_params[v]["std"] for v in vars_sorted]).clamp_min(DEFAULT_MIN_STD)
                pack["mean"] = means
                pack["std"] = stds

            elif "min-max" in method:
                mins = torch.stack([self.norm_params[v]["min"] for v in vars_sorted])
                maxs = torch.stack([self.norm_params[v]["max"] for v in vars_sorted])
                rng = (maxs - mins).clamp_min(self.epsilon)
                pack["min"] = mins
                pack["max"] = maxs
                pack["range"] = rng

            self.params_by_method[method] = pack

    def normalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Normalize a data profile.

        profile shape: [N, n_species + n_globals + 1]
        - species first (already log10 in sequence mode),
        - then globals,
        - last column is time (skipped here in sequence mode).
        """
        if profile.device != self.device or profile.dtype != self.torch_dtype:
            profile = profile.to(device=self.device, dtype=self.torch_dtype)

        norm = profile.clone()

        for method, cols in self.col_indices.items():
            if not cols:
                continue

            slice_ = norm[:, cols]  # [N, K]

            if "standard" in method:
                means = self.params_by_method[method]["mean"]  # [K]
                stds = self.params_by_method[method]["std"]    # [K]
                if "log" in method and not self.inputs_already_logged:
                    data = torch.log10(torch.clamp(slice_, min=self.epsilon))
                else:
                    data = slice_
                norm[:, cols] = torch.clamp((data - means) / stds,
                                            -self.clamp_value, self.clamp_value)

            elif "min-max" in method:
                mins = self.params_by_method[method]["min"]     # [K]
                rng = self.params_by_method[method]["range"]    # [K]
                if "log" in method and not self.inputs_already_logged:
                    data = torch.log10(torch.clamp(slice_, min=self.epsilon))
                else:
                    data = slice_
                norm[:, cols] = torch.clamp((data - mins) / rng, 0.0, 1.0)

        return norm

    def denormalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a data profile.

        In sequence mode, species/globals are often kept in log space end-to-end.
        This function will NOT exponentiate if inputs are already logged (to avoid double-exp).
        """
        if profile.device != self.device or profile.dtype != self.torch_dtype:
            profile = profile.to(device=self.device, dtype=self.torch_dtype)

        denorm = profile.clone()

        for method, cols in self.col_indices.items():
            if not cols:
                continue

            slice_ = denorm[:, cols]

            if "standard" in method:
                means = self.params_by_method[method]["mean"]
                stds = self.params_by_method[method]["std"].clamp_min(DEFAULT_MIN_STD)
                raw = slice_ * stds + means
                if "log" in method and not self.inputs_already_logged:
                    raw = torch.pow(self._ten, torch.clamp(raw, min=-38.0, max=38.0))
                denorm[:, cols] = torch.clamp(raw, min=-3.4e38, max=3.4e38)

            elif "min-max" in method:
                mins = self.params_by_method[method]["min"]
                rng = self.params_by_method[method]["range"]
                raw = slice_ * rng + mins
                if "log" in method and not self.inputs_already_logged:
                    raw = torch.pow(self._ten, torch.clamp(raw, min=-38.0, max=38.0))
                denorm[:, cols] = raw

        return denorm
