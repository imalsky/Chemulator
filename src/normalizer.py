#!/usr/bin/env python3
"""
normalizer.py -- Creates self-contained, normalized data files using robust methods.
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

# A tuple to hold the state for Welford's online algorithm: (count, mean, M2)
WelfordState = Tuple[int, float, float]

def welford_update(existing_aggregate: WelfordState, new_value: float) -> WelfordState:
    """Performs a single update step of Welford's algorithm."""
    count, mean, M2 = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2

def welford_finalize(existing_aggregate: WelfordState) -> Tuple[float, float]:
    """Finalizes Welford's algorithm to get mean and standard deviation."""
    count, mean, M2 = existing_aggregate
    if count < 2:
        return mean, 1.0
    variance = M2 / (count - 1)
    return mean, math.sqrt(variance)

class DataNormalizer:
    """
    Handles the normalization of chemical profile data.

    This class supports three main normalization methods:
    - 'standard': Z-score normalization (subtract mean, divide by std).
    - 'log-standard': Applies a log10 transform then Z-score normalizes.
    - 'log-min-max': Applies a log10 transform then scales to a [0, 1] range.
    
    It operates in a streaming fashion to handle datasets that may not fit in memory.
    """
    METHODS = {"standard", "log-standard", "log-min-max"}

    def __init__(self, input_folder: Union[str, Path], output_folder: Union[str, Path], *, config_data: Dict[str, Any], epsilon: float = 1e-9):
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config_data
        self.eps = float(epsilon)
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """Parses the configuration to determine which keys to process and with which method."""
        keys_to_process = set(self.config.get("all_variables", []))
        norm_config = self.config.get("normalization", {})
        user_key_methods = norm_config.get("key_methods", {})
        default_method = norm_config.get("default_method", "standard")
        
        key_methods = {key: user_key_methods.get(key, default_method).lower() for key in keys_to_process}
        
        for key, method in key_methods.items():
            if method not in self.METHODS:
                raise ValueError(f"Unsupported normalization method '{method}' for key '{key}'. Supported methods are: {self.METHODS}")
            if method.startswith("log-"):
                logger.info(f"Variable '{key}' will use '{method}' normalization. Ensure all values are positive.")
        
        return keys_to_process, key_methods

    def calculate_global_stats(self) -> Dict[str, Any]:
        """Calculates global statistics for all variables in a single pass over the data."""
        logger.info("Starting calculation of global statistics via streaming...")
        json_files = [p for p in self.input_dir.glob("*.json") if p.name != "normalization_metadata.json"]
        if not json_files:
            raise FileNotFoundError(f"No JSON profiles found in {self.input_dir}")

        # Initialize accumulators for each method type
        welford_accumulators = {k: (0, 0.0, 0.0) for k, m in self.key_methods.items() if m == "standard"}
        log_welford_accumulators = {k: (0, 0.0, 0.0) for k, m in self.key_methods.items() if m == "log-standard"}
        log_min_max_accumulators = {k: (float('inf'), float('-inf')) for k, m in self.key_methods.items() if m == "log-min-max"}
        
        log_invalid_counts = {k: 0 for k, m in self.key_methods.items() if m.startswith("log-")}

        for fpath in json_files:
            try:
                data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                for key in self.keys_to_process:
                    if key not in data or data[key] is None:
                        continue
                        
                    method = self.key_methods[key]
                    values = data[key] if isinstance(data[key], list) else [data[key]]
                    
                    for v in values:
                        if not (isinstance(v, (int, float)) and math.isfinite(v)):
                            continue

                        if method == "standard":
                            welford_accumulators[key] = welford_update(welford_accumulators[key], v)
                        elif method.startswith("log-"):
                            if v <= 0:
                                log_invalid_counts[key] += 1
                                continue
                            log_v = math.log10(v)

                            if method == "log-standard":
                                log_welford_accumulators[key] = welford_update(log_welford_accumulators[key], log_v)
                            elif method == "log-min-max":
                                current_min, current_max = log_min_max_accumulators[key]
                                log_min_max_accumulators[key] = (min(current_min, log_v), max(current_max, log_v))
                            
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error processing {fpath.name}: {e}. Skipping.")

        computed_stats: Dict[str, Any] = {}
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"epsilon": self.eps, "method": method}
            
            if log_invalid_counts.get(key, 0) > 0:
                logger.error(f"FATAL: Variable '{key}' has {log_invalid_counts[key]} non-positive values but requires a log-based normalization.")

            if method == "standard":
                mean, std = welford_finalize(welford_accumulators[key])
                if std < self.eps:
                    logger.warning(f"Variable '{key}' has near-zero variance. Setting std=1.0.")
                    std = 1.0
                stats.update({"mean": mean, "std": std})
            
            elif method == "log-standard":
                log_mean, log_std = welford_finalize(log_welford_accumulators[key])
                if log_std < self.eps:
                    logger.warning(f"Log-transformed variable '{key}' has near-zero variance. Setting log_std=1.0.")
                    log_std = 1.0
                stats.update({"log_mean": log_mean, "log_std": log_std})

            elif method == "log-min-max":
                min_val, max_val = log_min_max_accumulators[key]
                if min_val == float('inf') or max_val == float('-inf'):
                    logger.error(f"Variable '{key}' has no valid positive values for 'log-min-max'. Using defaults [0, 1].")
                    min_val, max_val = 0.0, 1.0
                elif (max_val - min_val) < self.eps:
                    logger.warning(f"Variable '{key}' has constant log values. Setting range to [min, min+1] for 'log-min-max'.")
                    max_val = min_val + 1.0
                stats.update({"min": min_val, "max": max_val})
            
            computed_stats[key] = stats
            
        metadata = {"normalization_methods": self.key_methods, "per_key_stats": computed_stats}
        self._save_metadata(metadata)
        logger.info("Global statistics calculation complete.")
        return metadata

    def _save_metadata(self, metadata: Dict[str, Any]):
        save_json(metadata, self.output_dir / "normalization_metadata.json")

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """Applies the specified normalization to a tensor."""
        eps = stats.get("epsilon", 1e-9)
        
        if method == "standard":
            return (x - stats["mean"]) / stats["std"]
            
        elif method == "log-standard":
            x_positive = torch.where(x > 0, x, torch.full_like(x, eps))
            x_log = torch.log10(x_positive)
            return (x_log - stats["log_mean"]) / stats["log_std"]
            
        elif method == "log-min-max":
            x_positive = torch.where(x > 0, x, torch.full_like(x, eps))
            x_log = torch.log10(x_positive)
            denominator = stats["max"] - stats["min"]
            if denominator < eps:
                return torch.zeros_like(x_log)
            return torch.clamp((x_log - stats["min"]) / denominator, 0.0, 1.0)
            
        else:
            # This branch should ideally not be reached if config is validated
            raise ValueError(f"Unsupported normalization method '{method}'")

    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float], 
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float]:
        """Inverts the normalization for a given variable, restoring its original scale."""
        method = metadata["normalization_methods"][var_name]
        stats = metadata["per_key_stats"][var_name]
        
        is_scalar = not isinstance(v, (torch.Tensor, list))
        x = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.clone().detach()

        y = None
        if method == "standard":
            y = x * stats["std"] + stats["mean"]
        elif method == "log-standard":
            log_val = x * stats["log_std"] + stats["log_mean"]
            y = torch.pow(10, log_val)
        elif method == "log-min-max":
            # Clamp normalized values to valid range before denormalizing
            x_clipped = torch.clamp(x, 0.0, 1.0)
            log_val = x_clipped * (stats["max"] - stats["min"]) + stats["min"]
            y = torch.pow(10, log_val)
        else:
            raise ValueError(f"Unsupported method '{method}' for denormalization.")

        # Return in the original format
        if is_scalar:
            return y.item()
        if isinstance(v, list):
            return y.tolist()
        return y

    def process_profiles(self, stats_metadata: Dict[str, Any]):
        """Normalizes all profiles from the input folder and saves them to the output folder."""
        logger.info(f"Normalizing all profiles and saving to: {self.output_dir}")
        methods = stats_metadata["normalization_methods"]
        stats = stats_metadata["per_key_stats"]
        
        processed_count, error_count = 0, 0
        
        for fpath in self.input_dir.glob("*.json"):
            if fpath.name == "normalization_metadata.json": 
                continue
                
            try:
                profile_data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                output_profile = {}
                is_valid = True
                
                for key in self.keys_to_process:
                    if key not in profile_data:
                        logger.warning(f"Key '{key}' missing in {fpath.name}. Skipping this file.")
                        is_valid = False
                        break
                        
                    value = profile_data[key]
                    
                    if methods[key].startswith("log-"):
                        values_to_check = value if isinstance(value, list) else [value]
                        if any(v <= 0 for v in values_to_check if isinstance(v, (int, float))):
                            logger.error(f"Profile {fpath.name} contains non-positive values for log-based variable '{key}'. Skipping file.")
                            is_valid = False
                            break
                    
                    tensor_val = torch.tensor(value, dtype=torch.float32)
                    norm_tensor = self.normalize_tensor(tensor_val, methods[key], stats[key])
                    output_profile[key] = norm_tensor.tolist() if isinstance(value, list) else norm_tensor.item()
                    
                if is_valid:
                    save_json(output_profile, self.output_dir / fpath.name)
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process profile {fpath.name}: {e}", exc_info=True)
                error_count += 1
                
        logger.info(f"Profile processing complete. Successfully processed: {processed_count}, Errors/Skipped: {error_count}")


__all__ = ["DataNormalizer", "denormalize"]