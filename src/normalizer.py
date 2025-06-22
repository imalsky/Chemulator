#!/usr/bin/env python3
"""
normalizer.py -- Compute and invert various normalization schemes.

This module calculates global statistics from a given set of training profiles.
It uses a memory-efficient streaming approach for most methods (like z-score)
and a memory-safe approach for quantile-based methods. Crucially, it provides
a `ValueError` with detailed context if it encounters data incompatible with
a chosen log-based normalization method, enabling rapid debugging.
"""
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# --- Constants ---
DTYPE = torch.float32
DEFAULT_EPSILON = 1e-9
DEFAULT_QUANTILE_MEMORY_LIMIT = 5_000_000
DEFAULT_SYMLOG_PERCENTILE = 0.5
WelfordState = Tuple[int, float, float]  # (count, mean, M2)


def welford_update(state: WelfordState, new_value: float) -> WelfordState:
    """Performs a single update step of Welford's algorithm for online variance."""
    count, mean, m2 = state
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return count, mean, m2


def welford_finalize(state: WelfordState, eps: float) -> Tuple[float, float]:
    """Finalizes Welford's algorithm to get mean and standard deviation."""
    count, mean, m2 = state
    if count < 2:
        return mean, 1.0
    variance = m2 / (count - 1)
    return mean, max(math.sqrt(variance), eps)


class DataNormalizer:
    """
    Handles data normalization by calculating statistics from training data
    and providing static methods to apply or invert the normalization.
    """
    METHODS = {"standard", "log-standard", "log-min-max", "iqr", "max-out",
               "scaled_signed_offset_log", "symlog", "bool", "none"}
    QUANTILE_METHODS = {"iqr", "symlog", "log-min-max"}

    def __init__(self, *, config_data: Dict[str, Any]):
        self.config = config_data
        self.data_spec = self.config.get("data_specification", {})
        self.norm_cfg = self.config.get("normalization", {})
        
        self.eps = float(self.norm_cfg.get("epsilon", DEFAULT_EPSILON))
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """Parses config to determine which keys to process and with which method."""
        all_vars = set(self.data_spec.get("all_variables", []))
        user_key_methods = self.norm_cfg.get("key_methods", {})
        default_method = self.norm_cfg.get("default_method", "standard")
        
        key_methods = {key: user_key_methods.get(key, default_method).lower() for key in all_vars}
        
        for key, method in key_methods.items():
            if method not in self.METHODS:
                raise ValueError(f"Unsupported method '{method}' for key '{key}'.")
        
        return all_vars, key_methods

    def calculate_stats(self, profile_paths: List[Path]) -> Dict[str, Any]:
        """Calculates global statistics for all variables from a list of profiles."""
        logger.info(f"Starting statistics calculation on {len(profile_paths)} profiles...")
        if not profile_paths:
            logger.warning("Cannot calculate statistics from an empty list of profiles. Returning empty stats.")
            return {"normalization_methods": {}, "per_key_stats": {}}

        accumulators = self._initialize_accumulators()

        for fpath in profile_paths:
            try:
                data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                for key in self.keys_to_process:
                    if key in data and data[key] is not None:
                        self._update_stats_for_key(key, data[key], accumulators, fpath)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error processing {fpath.name}: {e}. Skipping.")

        computed_stats = self._finalize_stats(accumulators)
        metadata = {"normalization_methods": self.key_methods, "per_key_stats": computed_stats}
        logger.info("Global statistics calculation complete.")
        return metadata

    def _initialize_accumulators(self) -> Dict[str, Dict[str, Any]]:
        """Initializes data structures for statistics accumulation."""
        accumulators = {}
        for key, method in self.key_methods.items():
            if method in ("none", "bool"): continue
            key_stats = {"min": float('inf'), "max": float('-inf')}
            if method in ("standard", "log-standard"):
                key_stats["welford"] = (0, 0.0, 0.0)
            if method in self.QUANTILE_METHODS:
                key_stats["values"] = []
            accumulators[key] = key_stats
        return accumulators

    def _update_stats_for_key(self, key: str, value: Any, accumulators: Dict, fpath: Path) -> None:
        """Updates accumulators for a single key, failing fast on invalid data for log methods."""
        if key not in accumulators: return
        method = self.key_methods[key]
        key_acc = accumulators[key]
        
        values_to_process = value if isinstance(value, list) else [value]
        for v in values_to_process:
            if not isinstance(v, (int, float)) or not math.isfinite(v): continue
            
            # CORRECTED: The threshold for log-based methods should be 0, not epsilon.
            if method.startswith("log-") and v <= 0:
                raise ValueError(
                    f"Variable '{key}' in profile '{fpath.name}' has non-positive value ({v}) "
                    f"but requires a log-based normalization ('{method}'). Please check data or config."
                )

            key_acc["min"] = min(key_acc["min"], v)
            key_acc["max"] = max(key_acc["max"], v)

            if method == "standard":
                key_acc["welford"] = welford_update(key_acc["welford"], v)
            elif method == "log-standard":
                key_acc["welford"] = welford_update(key_acc["welford"], math.log10(v))
            elif method in self.QUANTILE_METHODS:
                key_acc["values"].append(v)

    def _finalize_stats(self, accumulators: Dict) -> Dict[str, Any]:
        """Finalizes statistics from accumulators after the full data pass."""
        final_stats = {}
        
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"method": method, "epsilon": self.eps}

            if key not in accumulators:
                if method not in ("none", "bool"):
                    logger.warning(
                        f"Variable '{key}' was specified in config but not found in any training profiles. "
                        f"Assigning 'none' normalization as a fallback."
                    )
                self.key_methods[key] = "none"
                stats["method"] = "none"
                final_stats[key] = stats
                continue
            
            key_acc = accumulators[key]
            
            if method == "standard":
                mean, std = welford_finalize(key_acc["welford"], self.eps)
                stats.update({"mean": mean, "std": std})
            elif method == "log-standard":
                log_mean, log_std = welford_finalize(key_acc["welford"], self.eps)
                stats.update({"log_mean": log_mean, "log_std": log_std})
            elif method == "max-out":
                stats["max_val"] = max(abs(key_acc["min"]), abs(key_acc["max"]), self.eps)
            elif method == "scaled_signed_offset_log":
                m_pos = math.log10(max(0, key_acc["max"]) + 1 + self.eps)
                m_neg = math.log10(max(0, -key_acc["min"]) + 1 + self.eps)
                stats.update({"m": max(m_pos, m_neg, 1.0)})
            elif method in self.QUANTILE_METHODS:
                stats.update(self._finalize_quantile_stats(key, key_acc, method, self.norm_cfg))
            
            final_stats[key] = stats
        return final_stats
    
    def _finalize_quantile_stats(self, key: str, key_acc: dict, method: str, norm_cfg: dict) -> dict:
        """Computes stats for methods requiring all data (quantiles) with memory safety."""
        values_list = key_acc.get("values", [])
        max_values = norm_cfg.get("quantile_max_values_in_memory", DEFAULT_QUANTILE_MEMORY_LIMIT)
        if len(values_list) > max_values:
            logger.warning(f"Key '{key}' has {len(values_list):,} values, exceeding limit of {max_values:,}. Using random sample.")
            values_list = random.sample(values_list, max_values)
        if not values_list:
            logger.warning(f"No valid data found for quantile method on key '{key}'. Using defaults.")
            return {"median": 0, "iqr": 1.0, "threshold": 1.0, "scale_factor": 1.0, "min": 0.0, "max": 1.0}
        values = torch.tensor(values_list, dtype=DTYPE)
        stats: Dict[str, Any] = {}
        if method == "iqr":
            q1, med, q3 = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75])).tolist()
            stats.update({"median": med, "iqr": max(q3 - q1, self.eps)})
        elif method == "log-min-max":
            log_vals = torch.log10(torch.clamp(values, min=self.eps))
            min_v, max_v = log_vals.min().item(), log_vals.max().item()
            stats.update({"min": min_v, "max": max(max_v, min_v + self.eps)})
        elif method == "symlog":
            thresholds = norm_cfg.get("symlog_thresholds", {})
            percentile = norm_cfg.get("symlog_percentile", DEFAULT_SYMLOG_PERCENTILE)
            thr = thresholds.get(key, torch.quantile(torch.abs(values), percentile).item())
            thr = max(thr, self.eps)
            abs_v = torch.abs(values)
            mask = abs_v > thr
            transformed = torch.zeros_like(values)
            transformed[mask] = torch.sign(values[mask]) * (torch.log10(abs_v[mask] / thr) + 1)
            transformed[~mask] = values[~mask] / thr
            sf = transformed.abs().max().item() if transformed.numel() > 0 else 1.0
            stats.update({"threshold": thr, "scale_factor": max(sf, 1.0)})
        return stats

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """Applies the specified normalization method to a tensor."""
        eps = stats.get("epsilon", DEFAULT_EPSILON)
        if method in ("none", "bool"): return x
        
        if method == "standard":
            return (x - stats["mean"]) / stats["std"]
        elif method == "log-standard":
            # CORRECTED: Added clamp for safety
            safe_x = torch.clamp(x, min=eps)
            return (torch.log10(safe_x) - stats["log_mean"]) / stats["log_std"]
        elif method == "log-min-max":
            # CORRECTED: Added clamp for safety
            safe_x = torch.clamp(x, min=eps)
            denom = stats["max"] - stats["min"]
            return torch.clamp((torch.log10(safe_x) - stats["min"]) / max(denom, eps), 0.0, 1.0)
        elif method == "max-out":
            return x / stats["max_val"]
        elif method == "iqr":
            return (x - stats["median"]) / stats["iqr"]
        elif method == "scaled_signed_offset_log":
            y = torch.sign(x) * torch.log10(torch.abs(x) + 1 + eps)
            return y / stats["m"]
        elif method == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            abs_x = torch.abs(x)
            mask = abs_x > thr
            y = torch.zeros_like(x)
            y[mask] = torch.sign(x[mask]) * (torch.log10(abs_x[mask] / thr) + 1)
            y[~mask] = x[~mask] / thr
            return torch.clamp(y / sf, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported normalization method '{method}'")

    @staticmethod
    def denormalize(v: Union[Tensor, List, float, bool, None], metadata: Dict[str, Any], var_name: str) -> Union[Tensor, List, float, bool, None]:
        """Applies the inverse normalization to a value, tensor, or list."""
        if v is None: return None
        method = metadata["normalization_methods"].get(var_name, "none")
        if method in ("none", "bool"): return v
        
        stats = metadata["per_key_stats"].get(var_name)
        if not stats: raise ValueError(f"No stats for '{var_name}' in metadata")

        is_scalar = not isinstance(v, (torch.Tensor, list))
        x = torch.tensor(v, dtype=DTYPE) if not isinstance(v, torch.Tensor) else v.clone().detach().to(DTYPE)
        
        y: Tensor
        if method == "standard":
            y = x * stats["std"] + stats["mean"]
        elif method == "log-standard":
            y = torch.pow(10, x * stats["log_std"] + stats["log_mean"])
        elif method == "log-min-max":
            y = torch.pow(10, torch.clamp(x, 0, 1) * (stats["max"] - stats["min"]) + stats["min"])
        elif method == "max-out":
            y = x * stats["max_val"]
        elif method == "iqr":
            y = x * stats["iqr"] + stats["median"]
        elif method == "scaled_signed_offset_log":
            ytmp = x * stats["m"]
            y = torch.sign(ytmp) * (torch.pow(10, torch.abs(ytmp)) - 1 - stats["epsilon"])
        elif method == "symlog":
            unscaled = x * stats["scale_factor"]
            abs_unscaled = torch.abs(unscaled)
            mask = abs_unscaled > 1.0
            y = torch.zeros_like(x)
            y[mask] = torch.sign(unscaled[mask]) * stats["threshold"] * torch.pow(10, abs_unscaled[mask] - 1)
            y[~mask] = unscaled[~mask] * stats["threshold"]
        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")

        if is_scalar: return y.item()
        return y.tolist() if isinstance(v, list) else y

__all__ = ["DataNormalizer"]