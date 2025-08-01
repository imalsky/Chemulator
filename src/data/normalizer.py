#!/usr/bin/env python3
"""
Data normalization module for chemical kinetics datasets.

• Compatible with both *row-wise* and *sequence-mode* pipelines.
• Guards against double-normalizing time when the SequenceDataset has
  already mapped t → [0, 1].
• Casts tensors to the configured dtype / device on entry, avoiding
  mixed-precision surprises.
• All original validation, logging and ratio helpers are preserved.
"""

import logging
import math
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_EPSILON = 1e-30
DEFAULT_MIN_STD = 1e-10
DEFAULT_CLAMP   = 50.0
SAFE_EPSILON    = 1e-38      # for log-transform safety
MIN_RANGE       = 1e-10


# ----------------------------------------------------------------------
# Data-wide statistics collector
# ----------------------------------------------------------------------
class DataNormalizer:
    """Calculates normalization statistics with robust data validation."""
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config       = config
        self.data_config  = config["data"]
        self.norm_config  = config["normalization"]

        self.species_vars = self.data_config["species_variables"]
        self.global_vars  = self.data_config["global_variables"]
        self.time_var     = self.data_config["time_variable"]
        self.all_vars     = self.species_vars + self.global_vars + [self.time_var]

        self.epsilon  = self.norm_config.get("epsilon", DEFAULT_EPSILON)
        self.min_std  = self.norm_config.get("min_std", DEFAULT_MIN_STD)
        self.logger   = logging.getLogger(__name__)

    # ----- accumulator helpers ------------------------------------------------
    def _initialize_accumulators(self) -> Dict[str, Dict[str, Any]]:
        """Return per-variable accumulator dicts."""
        accs: Dict[str, Dict[str, Any]] = {}
        for i, var in enumerate(self.all_vars):
            method = self._get_method(var)
            if method == "none":
                continue
            accs[var] = {
                "method": method,
                "index":  i,
                "count":  0,
                "mean":   0.0,
                "m2":     0.0,
                "min":    float("inf"),
                "max":    float("-inf"),
            }
        return accs

    def _get_method(self, var: str) -> str:
        return self.norm_config.get("methods", {}).get(var,
                                                      self.norm_config["default_method"])

    def _update_single_accumulator(self, acc: Dict[str, Any],
                                   vec: np.ndarray,
                                   var_name: str) -> None:
        """Fast (vectorised) accumulator update with safety checks."""
        if vec.size == 0:
            return

        # filter non-finite
        finite_mask = np.isfinite(vec)
        if not np.all(finite_mask):
            bad = (~finite_mask).sum()
            if bad / vec.size > 0.01:
                self.logger.warning(f"{var_name}: {bad}/{vec.size} non-finite")
            vec = vec[finite_mask]
            if vec.size == 0:
                self.logger.warning(f"{var_name}: nothing left after filtering")
                return

        # log-transform safeguards
        if acc["method"].startswith("log-"):
            below = vec < SAFE_EPSILON
            if np.any(below):
                self.logger.warning(f"{var_name}: {below.sum()} values < {SAFE_EPSILON:.1e}")
            vec = np.log10(np.maximum(vec, SAFE_EPSILON))

            if vec.min() < -25 or vec.max() > 25:
                self.logger.warning(f"{var_name}: extreme log range [{vec.min():.1f}, {vec.max():.1f}]")

        # Welford merge
        n_b  = vec.size
        mean = float(vec.mean())
        m2   = float(((vec - mean) ** 2).sum()) if n_b > 1 else 0.0

        n_a  = acc["count"]
        delta = mean - acc["mean"]
        n_ab = n_a + n_b
        if n_ab > 0:
            acc["mean"] = (n_a * acc["mean"] + n_b * mean) / n_ab
            acc["m2"]  += m2 + delta ** 2 * n_a * n_b / n_ab

        acc["count"] = n_ab
        acc["min"]   = min(acc["min"], float(vec.min()))
        acc["max"]   = max(acc["max"], float(vec.max()))

    # merge, finalise identical to original version ---------------------------
    def _merge_accumulators(self,
                            main_accs: Dict[str, Dict[str, Any]],
                            other_accs: Dict[str, Dict[str, Any]]) -> None:
        for var, ob in other_accs.items():
            if not ob:
                continue
            if var not in main_accs:
                main_accs[var] = ob
                continue
            oa = main_accs[var]
            n_a, mean_a, m2_a = oa["count"], oa["mean"], oa["m2"]
            n_b, mean_b, m2_b = ob["count"], ob["mean"], ob["m2"]
            n_ab = n_a + n_b
            if n_ab == 0:
                continue
            delta    = mean_b - mean_a
            oa["mean"] = (n_a * mean_a + n_b * mean_b) / n_ab
            oa["m2"]   = m2_a + m2_b + delta ** 2 * n_a * n_b / n_ab
            oa["count"] = n_ab
            oa["min"]   = min(oa["min"], ob["min"])
            oa["max"]   = max(oa["max"], ob["max"])

    def _finalize_statistics(self,
                             accs: Dict[str, Dict[str, Any]],
                             is_ratio: bool = False) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"per_key_stats": {}}
        if not is_ratio:
            stats["normalization_methods"] = {}

        for var, acc in accs.items():
            method = acc.get("method", "standard")
            if not is_ratio:
                stats["normalization_methods"][var] = method
            if method == "none":
                continue
            if acc["count"] == 0:
                self.logger.warning(f"{var}: no valid samples, set to 'none'")
                if not is_ratio:
                    stats["normalization_methods"][var] = "none"
                continue

            var_stats: Dict[str, Any] = {"method": method}
            if acc["count"] > 1:
                variance = acc["m2"] / (acc["count"] - 1)
                std      = max(math.sqrt(variance), self.min_std)
            else:
                std = 1.0

            if "standard" in method:
                k_mean, k_std = ("log_mean", "log_std") if "log" in method else ("mean", "std")
                var_stats[k_mean] = acc["mean"]
                var_stats[k_std]  = std
                if std < MIN_RANGE:
                    self.logger.warning(f"{var}: very small std {std:.1e}")

            elif "min-max" in method:
                var_stats["min"] = acc["min"]
                var_stats["max"] = acc["max"]
                rng = acc["max"] - acc["min"]
                if rng < MIN_RANGE:
                    self.logger.warning(f"{var}: tiny range {rng:.1e}, expanding")
                    c = (acc["max"] + acc["min"]) / 2
                    var_stats["min"] = c - MIN_RANGE / 2
                    var_stats["max"] = c + MIN_RANGE / 2

            if is_ratio:
                stats[var] = {**var_stats,
                              "min": acc["min"],
                              "max": acc["max"],
                              "count": acc["count"]}
            else:
                stats["per_key_stats"][var] = var_stats

        if not is_ratio:
            for var in self.all_vars:
                stats["normalization_methods"].setdefault(var, "none")
            stats["epsilon"]      = SAFE_EPSILON
            stats["clamp_value"]  = self.norm_config.get("clamp_value", DEFAULT_CLAMP)
        return stats


# ----------------------------------------------------------------------
# Runtime helper (vectorised normalise / denormalise)
# ----------------------------------------------------------------------
class NormalizationHelper:
    """Apply pre-computed statistics to torch tensors."""
    def __init__(self,
                 stats: Dict[str, Any],
                 device: torch.device,
                 species_vars: List[str],
                 global_vars: List[str],
                 time_var: str,
                 config: Optional[Dict[str, Any]] = None):

        dtype_name   = (config or {}).get("system", {}).get("dtype", "float32")
        self.torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32

        self.stats        = stats
        self.device       = device
        self.species_vars = species_vars
        self.global_vars  = global_vars
        self.time_var     = time_var

        self.n_species    = len(species_vars)
        self.n_globals    = len(global_vars)

        self.methods      = stats["normalization_methods"]
        self.per_key_stats= stats["per_key_stats"]

        # detect sequence mode (time already pre-scaled to [0,1])
        self.is_sequence_mode = bool(stats.get("time_normalization")) or \
                                bool((config or {}).get("data", {}).get("sequence_mode", False))

        self.epsilon     = torch.tensor(stats.get("epsilon", DEFAULT_EPSILON),
                                        dtype=self.torch_dtype,
                                        device=self.device)
        self.clamp_value = stats.get("clamp_value", DEFAULT_CLAMP)
        self.ratio_stats = stats.get("ratio_stats", None)

        self.logger = logging.getLogger(__name__)
        self._precompute_parameters()

    # ---------- internal precompute ------------------------------------------
    def _precompute_parameters(self) -> None:
        """Cache per-variable tensors on the right device/dtype."""
        self.norm_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.method_groups = {
            "standard": [], "log-standard": [],
            "min-max": [], "log-min-max": [], "none": []
        }

        var_to_col = {v: i for i, v in enumerate(self.species_vars +
                                                 self.global_vars + [self.time_var])}

        for var, method in self.methods.items():
            # In sequence mode the time column is already normalised -> skip
            if self.is_sequence_mode and var == self.time_var:
                self.method_groups["none"].append(var)
                continue

            if method == "none" or var not in self.per_key_stats:
                self.method_groups["none"].append(var)
                continue

            vs   = self.per_key_stats[var]
            pars = {"method": method}

            if "standard" in method:
                k_mean, k_std = ("log_mean", "log_std") if "log" in method else ("mean", "std")
                pars["mean"] = torch.tensor(vs[k_mean], dtype=self.torch_dtype, device=self.device)
                pars["std"]  = torch.tensor(vs[k_std],  dtype=self.torch_dtype, device=self.device)
            elif "min-max" in method:
                pars["min"] = torch.tensor(vs["min"], dtype=self.torch_dtype, device=self.device)
                pars["max"] = torch.tensor(vs["max"], dtype=self.torch_dtype, device=self.device)

            self.norm_params[var] = pars
            self.method_groups[method].append(var)

        # column indices grouped by method
        self.col_indices: Dict[str, List[int]] = {
            m: [var_to_col[v] for v in lst] for m, lst in self.method_groups.items() if lst
        }

    # ---------- normalise / denormalise --------------------------------------
    def normalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Return a new tensor with variables normalised."""
        if profile.device != self.device or profile.dtype != self.torch_dtype:
            profile = profile.to(device=self.device, dtype=self.torch_dtype)

        norm = profile.clone()

        for method, cols in self.col_indices.items():
            if not cols or method == "none":
                continue
            slice_ = norm[:, cols]
            if "standard" in method:
                means = torch.stack([self.norm_params[v]["mean"]
                                     for v in self.method_groups[method]])
                stds  = torch.stack([self.norm_params[v]["std"]
                                     for v in self.method_groups[method]])
                data  = torch.log10(torch.clamp(slice_, min=self.epsilon)) if "log" in method else slice_
                norm[:, cols] = torch.clamp((data - means) / stds,
                                            -self.clamp_value, self.clamp_value)

            elif "min-max" in method:
                mins   = torch.stack([self.norm_params[v]["min"]
                                      for v in self.method_groups[method]])
                maxs   = torch.stack([self.norm_params[v]["max"]
                                      for v in self.method_groups[method]])
                ranges = torch.clamp(maxs - mins, min=self.epsilon)
                data   = torch.log10(torch.clamp(slice_, min=self.epsilon)) if "log" in method else slice_
                norm[:, cols] = torch.clamp((data - mins) / ranges, 0.0, 1.0)
        return norm

    def denormalize_profile(self, profile: torch.Tensor) -> torch.Tensor:
        """Invert normalisation."""
        if profile.device != self.device or profile.dtype != self.torch_dtype:
            profile = profile.to(device=self.device, dtype=self.torch_dtype)

        denorm = profile.clone()

        for method, cols in self.col_indices.items():
            if not cols or method == "none":
                continue
            slice_ = denorm[:, cols]
            if "standard" in method:
                means = torch.stack([self.norm_params[v]["mean"]
                                     for v in self.method_groups[method]])
                stds  = torch.stack([self.norm_params[v]["std"]
                                     for v in self.method_groups[method]])
                raw   = slice_ * stds + means
                if "log" in method:
                    raw = torch.pow(torch.tensor(10.0, dtype=self.torch_dtype, device=self.device),
                                    torch.clamp(raw, min=-38.0, max=38.0))
                denorm[:, cols] = torch.clamp(raw, min=-3.4e38, max=3.4e38)

            elif "min-max" in method:
                mins   = torch.stack([self.norm_params[v]["min"]
                                      for v in self.method_groups[method]])
                maxs   = torch.stack([self.norm_params[v]["max"]
                                      for v in self.method_groups[method]])
                ranges = torch.clamp(maxs - mins, min=self.epsilon)
                raw    = slice_ * ranges + mins
                if "log" in method:
                    raw = torch.pow(torch.tensor(10.0, dtype=self.torch_dtype, device=self.device),
                                    torch.clamp(raw, min=-38.0, max=38.0))
                denorm[:, cols] = raw
        return denorm

    # ---------- ratio helper (unchanged) -------------------------------------
    def denormalize_ratio_predictions(self,
                                      standardized_log_ratios: torch.Tensor,
                                      initial_species: torch.Tensor) -> torch.Tensor:
        """Convert std-log-ratio predictions back to absolute values."""
        if self.ratio_stats is None:
            raise ValueError("Ratio statistics not available.")

        device  = standardized_log_ratios.device
        initial = initial_species.to(device=device, dtype=self.torch_dtype)

        ratio_means = torch.tensor([self.ratio_stats[v]["mean"] for v in self.species_vars],
                                   device=device, dtype=self.torch_dtype)
        ratio_stds  = torch.tensor([self.ratio_stats[v]["std"]  for v in self.species_vars],
                                   device=device, dtype=self.torch_dtype)

        log_ratios = standardized_log_ratios * ratio_stds + ratio_means
        log_ratios = torch.clamp(log_ratios, min=-38.0, max=38.0)
        ratios     = torch.pow(torch.tensor(10.0, dtype=self.torch_dtype, device=device), log_ratios)

        return initial * ratios
