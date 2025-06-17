#!/usr/bin/env python3
"""
normalizer.py -- Creates self-contained, normalized data files.
"""
from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import torch
from torch import Tensor
from utils import save_json

logger = logging.getLogger(__name__)

WelfordState = Tuple[int, float, float]

def welford_update(existing_aggregate: WelfordState, new_value: float) -> WelfordState:
    count, mean, M2 = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2

def welford_finalize(existing_aggregate: WelfordState) -> Tuple[float, float]:
    count, mean, M2 = existing_aggregate
    if count < 2: return mean, 0.0
    variance = M2 / (count - 1)
    return mean, math.sqrt(variance)

class DataNormalizer:
    METHODS = {"standard", "log-min-max", "iqr"}

    def __init__(self, input_folder: Union[str, Path], output_folder: Union[str, Path], *, config_data: Dict[str, Any], epsilon: float = 1e-9):
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir(): raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config_data
        self.eps = float(epsilon)
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        keys_to_process = set(self.config.get("all_variables", []))
        norm_config = self.config.get("normalization", {})
        user_key_methods = norm_config.get("key_methods", {})
        default_method = norm_config.get("default_method", "standard")
        key_methods = {key: user_key_methods.get(key, default_method).lower() for key in keys_to_process}
        
        # Validate log-min-max is only used for appropriate variables
        for key, method in key_methods.items():
            if method == "log-min-max":
                logger.warning(f"Variable '{key}' uses log-min-max normalization. Ensure all values are positive!")
        
        return keys_to_process, key_methods

    def calculate_global_stats(self) -> Dict[str, Any]:
        logger.info("Starting calculation of global statistics via streaming...")
        json_files = [p for p in self.input_dir.glob("*.json") if p.name != "normalization_metadata.json"]
        if not json_files: raise FileNotFoundError(f"No JSON profiles found in {self.input_dir}")

        welford_stats = {k: (0, 0.0, 0.0) for k, m in self.key_methods.items() if m == "standard"}
        iqr_accumulators = {k: [] for k, m in self.key_methods.items() if m == "iqr"}
        log_min_max = {k: (float('inf'), float('-inf')) for k, m in self.key_methods.items() if m == "log-min-max"}
        
        # Track negative/zero values for log normalization
        log_invalid_counts = {k: 0 for k, m in self.key_methods.items() if m == "log-min-max"}

        for fpath in json_files:
            try:
                data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                for key in self.keys_to_process:
                    if key in data and data[key] is not None:
                        method = self.key_methods[key]
                        values = data[key] if isinstance(data[key], list) else [data[key]]
                        
                        if method == "standard":
                            for v in values:
                                if isinstance(v, (int, float)) and math.isfinite(v):
                                    welford_stats[key] = welford_update(welford_stats[key], v)
                                    
                        elif method == "iqr":
                            iqr_accumulators[key].extend(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
                            
                        elif method == "log-min-max":
                            current_min, current_max = log_min_max[key]
                            for v in values:
                                if isinstance(v, (int, float)) and math.isfinite(v):
                                    if v <= 0:
                                        log_invalid_counts[key] += 1
                                    else:
                                        log_v = math.log10(v)
                                        current_min = min(current_min, log_v)
                                        current_max = max(current_max, log_v)
                            log_min_max[key] = (current_min, current_max)
                            
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error processing {fpath.name}: {e}. Skipping.")

        # Check for invalid log-min-max usage
        for key, count in log_invalid_counts.items():
            if count > 0:
                logger.error(
                    f"Variable '{key}' has {count} non-positive values but uses log-min-max normalization! "
                    f"Consider using 'standard' or 'iqr' method instead."
                )

        computed_stats: Dict[str, Any] = {}
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"epsilon": self.eps, "method": method}
            
            if method == "standard":
                stats["mean"], stats["std"] = welford_finalize(welford_stats[key])
                if stats["std"] < self.eps: 
                    stats["std"] = 1.0  # Avoid division by zero
                    logger.warning(f"Variable '{key}' has zero variance. Setting std=1.0")
                    
            elif method == "log-min-max":
                stats["min"], stats["max"] = log_min_max[key]
                if stats["min"] == float('inf') or stats["max"] == float('-inf'):
                    # No valid positive values found
                    logger.error(f"Variable '{key}' has no positive values for log-min-max normalization!")
                    stats["min"], stats["max"] = 0.0, 1.0
                elif stats["max"] <= stats["min"]:
                    # All values are the same
                    logger.warning(f"Variable '{key}' has constant log values. Setting range to [min, min+1]")
                    stats["max"] = stats["min"] + 1.0
                    
            elif method == "iqr":
                values = iqr_accumulators.get(key, [])
                if len(values) < 4:  # Need at least 4 values for quartiles
                    logger.warning(f"Variable '{key}' has too few values ({len(values)}) for IQR. Using defaults.")
                    stats.update({"median": 0.0, "iqr": 1.0})
                else:
                    # ============================ FIX: START ============================
                    # The original used torch.float64, which is causing a persistent
                    # serialization issue on MPS. Using float32 is sufficient for
                    # this calculation and removes the problematic dtype.
                    q = torch.quantile(torch.tensor(values, dtype=torch.float32), torch.tensor([0.25, 0.50, 0.75]))
                    # ============================= FIX: END =============================
                    iqr = (q[2] - q[0]).item()
                    if iqr < self.eps:
                        logger.warning(f"Variable '{key}' has zero IQR. Using median ± 1")
                        iqr = 1.0
                    stats["median"] = q[1].item()
                    stats["iqr"] = iqr
                    
            computed_stats[key] = stats

        metadata = {"normalization_methods": self.key_methods, "per_key_stats": computed_stats}
        self._save_metadata(metadata)
        logger.info("Global statistics calculation complete.")
        return metadata

    def _save_metadata(self, metadata: Dict[str, Any]):
        save_json(metadata, self.output_dir / "normalization_metadata.json")

    @staticmethod
    def normalize_tensor(x: torch.Tensor, method: str, stats: Dict[str, Any]) -> torch.Tensor:
        eps = stats.get("epsilon", 1e-9)
        
        if method == "standard":
            return (x - stats["mean"]) / stats["std"]
            
        elif method == "iqr":
            return (x - stats["median"]) / stats["iqr"]
            
        elif method == "log-min-max":
            # Handle non-positive values gracefully
            x_positive = torch.where(x > eps, x, torch.full_like(x, eps))
            x_log = torch.log10(x_positive)
            normalized = (x_log - stats["min"]) / (stats["max"] - stats["min"])
            # Clip to [0, 1] range to handle numerical issues
            return torch.clamp(normalized, 0.0, 1.0)
            
        else:
            raise ValueError(f"Unsupported method '{method}'")

    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float], 
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float]:
        """
        Inverts the normalization for a given variable.
        """
        method = metadata["normalization_methods"][var_name]
        stats = metadata["per_key_stats"][var_name]
        
        is_scalar = not isinstance(v, (torch.Tensor, list))
        x = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.clone().detach()

        if method == "standard":
            y = x * stats["std"] + stats["mean"]
        elif method == "iqr":
            y = x * stats["iqr"] + stats["median"]
        elif method == "log-min-max":
            # Clip normalized values to valid range
            x_clipped = torch.clamp(x, 0.0, 1.0)
            log_val = x_clipped * (stats["max"] - stats["min"]) + stats["min"]
            y = torch.pow(10, log_val)
        else:
            raise ValueError(f"Unsupported method '{method}' for denormalization.")

        if is_scalar:
            return y.item()
        if isinstance(v, list):
            return y.tolist()
        return y

    def process_profiles(self, stats_metadata: Dict[str, Any]):
        """Normalizes all profiles from the input folder and saves them to the output folder."""
        logger.info(f"Normalizing profiles to: {self.output_dir}")
        methods = stats_metadata["normalization_methods"]
        stats = stats_metadata["per_key_stats"]
        
        processed_count = 0
        error_count = 0
        
        for fpath in self.input_dir.glob("*.json"):
            if fpath.name == "normalization_metadata.json": 
                continue
                
            try:
                profile_data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                output_profile = {}
                is_valid = True
                
                for key in self.keys_to_process:
                    if key not in profile_data:
                        logger.warning(f"Key '{key}' missing in {fpath.name}. Skipping file.")
                        is_valid = False
                        break
                        
                    value = profile_data[key]
                    
                    # Check for invalid values in log-min-max variables
                    if methods[key] == "log-min-max":
                        if isinstance(value, list):
                            if any(v <= 0 for v in value if isinstance(v, (int, float))):
                                logger.error(f"Profile {fpath.name} has non-positive values for log variable '{key}'")
                                is_valid = False
                                break
                        elif isinstance(value, (int, float)) and value <= 0:
                            logger.error(f"Profile {fpath.name} has non-positive value for log variable '{key}'")
                            is_valid = False
                            break
                    
                    # Normalize
                    tensor_val = torch.tensor(value, dtype=torch.float32)
                    norm_tensor = self.normalize_tensor(tensor_val, methods[key], stats[key])
                    output_profile[key] = norm_tensor.tolist() if isinstance(value, list) else norm_tensor.item()
                    
                if is_valid:
                    save_json(output_profile, self.output_dir / fpath.name)
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {fpath.name}: {e}")
                error_count += 1
                
        logger.info(f"Profile processing complete. Processed: {processed_count}, Errors: {error_count}")

__all__ = ["DataNormalizer"]