#!/usr/bin/env python3
"""
normalizer.py - Data normalization with improved numerical stability and efficiency.
"""
from __future__ import annotations

import h5py
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants
DTYPE = torch.float32
EPSILON = 1e-50
DEFAULT_QUANTILE_MEMORY_LIMIT = 50_000_000
DEFAULT_SYMLOG_PERCENTILE = 0.1
STATS_CHUNK_SIZE = 1024
NORMALIZED_VALUE_CLAMP = 50.0  


class DataNormalizer:
    """
    Handles data normalization with multiple methods and improved numerical stability.
    
    This class provides various normalization techniques optimized for both CPU and GPU,
    with special attention to numerical stability in log-based transformations.
    """
    
    METHODS = {
        "standard", "log-standard", "log-min-max", "iqr", "max-out",
        "scaled_signed_offset_log", "symlog", "signed-log", "bool", "none"
    }
    QUANTILE_METHODS = {"iqr", "symlog", "log-min-max"}

    def __init__(self, *, config_data: Dict[str, Any]):
        """
        Initialize the DataNormalizer with configuration.
        
        Args:
            config_data: Configuration dictionary containing normalization settings
        """
        self.config = config_data
        self.device = torch.device('cpu')
        self.data_spec = self.config.get("data_specification", {})
        self.norm_cfg = self.config.get("normalization", {})
        
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()
        # Track which keys use approximated quantiles
        self._approximated_quantile_keys = set()
        logger.info(f"DataNormalizer initialized on device '{self.device}'.")

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """
        Parse configuration to determine which keys to process and their normalization methods.
        
        Returns:
            Tuple of (set of keys to process, dict mapping keys to methods)
        """
        all_vars = set(self.data_spec.get("all_variables", []))
        user_key_methods = self.norm_cfg.get("key_methods", {})
        default_method = self.norm_cfg.get("default_method", "standard")

        key_methods = {
            key: user_key_methods.get(key, default_method).lower() 
            for key in all_vars
        }

        for key, method in key_methods.items():
            if method not in self.METHODS:
                raise ValueError(f"Unsupported method '{method}' for key '{key}'.")

        return all_vars, key_methods

    def calculate_stats(self, h5_path: Path, train_indices: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculate normalization statistics by streaming data in chunks for memory efficiency.
        
        This method reads data in chunks to avoid loading the entire dataset into memory,
        making it suitable for large datasets. It also returns the raw data for caching.
        
        Args:
            h5_path: Path to HDF5 file
            train_indices: List of training sample indices
            
        Returns:
            Tuple of (normalization metadata dict, raw training data dict)
        """
        logger.info(f"Calculating statistics from {len(train_indices)} training samples...")
        if not train_indices:
            raise ValueError("Cannot calculate statistics from empty training indices.")
        
        sorted_train_indices = sorted(train_indices)
        
        raw_train_data_chunks = {}
        accumulators = None
        
        with h5py.File(h5_path, 'r', swmr=True, libver='latest') as hf:
            available_keys = self.keys_to_process.intersection(hf.keys())
            if len(available_keys) != len(self.keys_to_process):
                missing = self.keys_to_process - available_keys
                logger.warning(f"Keys not found in HDF5 and will be skipped: {missing}")
            
            accumulators = self._initialize_accumulators(available_keys)
            for key in available_keys:
                raw_train_data_chunks[key] = []

            logger.info("Reading training data and calculating stats in chunks...")
            start_time = time.time()
            
            num_chunks = (len(sorted_train_indices) + STATS_CHUNK_SIZE - 1) // STATS_CHUNK_SIZE
            for i in tqdm(range(0, len(sorted_train_indices), STATS_CHUNK_SIZE), desc="Calculating Stats", total=num_chunks):
                chunk_indices = sorted_train_indices[i:i + STATS_CHUNK_SIZE]
                
                batch_of_tensors = {}
                for key in available_keys:
                    data_chunk_np = hf[key][chunk_indices]
                    raw_train_data_chunks[key].append(data_chunk_np)
                    batch_of_tensors[key] = torch.from_numpy(data_chunk_np)
                
                self._update_accumulators_with_batch(batch_of_tensors, accumulators)
            
            load_time = time.time() - start_time
            logger.info(f"Finished reading data and calculating chunked stats in {load_time:.2f}s.")

        computed_stats = self._finalize_stats(accumulators)
        metadata = {"normalization_methods": self.key_methods, "per_key_stats": computed_stats}
        
        logger.info("Concatenating data chunks for caching...")
        final_raw_data = {key: np.concatenate(chunks) for key, chunks in raw_train_data_chunks.items()}
        
        logger.info("Statistics calculation and data pre-loading complete.")
        return metadata, final_raw_data

    def _initialize_accumulators(self, keys_to_process: Set[str]) -> Dict[str, Dict[str, Any]]:
        """
        Initialize data structures for online statistics accumulation using Welford's algorithm.
        
        Args:
            keys_to_process: Set of variable keys to process
            
        Returns:
            Dictionary of accumulators for each key
        """
        accumulators = {}
        for key in keys_to_process:
            method = self.key_methods[key]
            if method in ("none", "bool"):
                continue

            acc: Dict[str, Any] = {}
            if method in ("standard", "log-standard", "signed-log"):
                acc.update({
                    "count": 0,
                    "mean": torch.tensor(0.0, dtype=DTYPE, device=self.device),
                    "m2": torch.tensor(0.0, dtype=DTYPE, device=self.device)
                })
            
            if method in self.QUANTILE_METHODS:
                acc["values"] = []
                # Add a counter for total values seen for this key
                acc["total_values_seen"] = 0
            
            if method in ("max-out", "scaled_signed_offset_log", "log-min-max"):
                 acc.update({
                    "min": torch.tensor(float('inf'), dtype=DTYPE, device=self.device),
                    "max": torch.tensor(float('-inf'), dtype=DTYPE, device=self.device)
                })

            if acc:
                accumulators[key] = acc
        return accumulators

    def _update_accumulators_with_batch(self, batch: Dict[str, Tensor], accumulators: Dict) -> None:
        """
        Update accumulators for each key using Welford's algorithm for numerical stability.
        
        Args:
            batch: Dictionary of tensors for current batch
            accumulators: Dictionary of accumulator states to update
        """
        # Get the quantile memory limit from config or use default
        quantile_max_values = self.norm_cfg.get("quantile_max_values_in_memory", DEFAULT_QUANTILE_MEMORY_LIMIT)

        for key, data_batch in batch.items():
            if key not in accumulators:
                continue
            
            data = data_batch.to(dtype=DTYPE, device=self.device).flatten()
            valid_data = data[torch.isfinite(data)]
            if valid_data.numel() == 0:
                continue
            
            method = self.key_methods[key]
            key_acc = accumulators[key]
            
            data_for_stats = valid_data
            if method == "log-standard":
                data_for_stats = torch.log10(torch.clamp(valid_data, min=EPSILON))
            elif method == "signed-log":
                data_for_stats = torch.sign(valid_data) * torch.log10(torch.abs(valid_data) + 1.0)
                
            if "count" in key_acc:
                n_new = data_for_stats.numel()
                if n_new > 0:
                    count_old = key_acc["count"]
                    mean_old = key_acc["mean"]
                    m2_old = key_acc["m2"]
                    count_new = count_old + n_new
                    delta = torch.mean(data_for_stats) - mean_old
                    mean_new = mean_old + delta * (n_new / count_new)
                    m2_new = m2_old + torch.sum((data_for_stats - mean_old) * (data_for_stats - mean_new))
                    key_acc["count"] = count_new
                    key_acc["mean"] = mean_new
                    key_acc["m2"] = m2_new
            
            if "values" in key_acc:
                key_acc["total_values_seen"] += valid_data.numel()
                current_stored_size = sum(t.numel() for t in key_acc["values"])

                # If we have space, just append
                if current_stored_size + valid_data.numel() <= quantile_max_values:
                    key_acc["values"].append(valid_data)
                else:
                    # Reservoir sampling: we need to make space
                    self._approximated_quantile_keys.add(key)
                    
                    # Combine existing and new data, then subsample to the limit
                    combined_data = torch.cat(key_acc["values"] + [valid_data])
                    perm = torch.randperm(combined_data.numel(), device=self.device)[:quantile_max_values]
                    
                    # Store as a single tensor to avoid future concatenations
                    key_acc["values"] = [combined_data[perm]]

            if "max" in key_acc:
                key_acc["min"] = torch.min(key_acc["min"], valid_data.min())
                key_acc["max"] = torch.max(key_acc["max"], valid_data.max())

    def _finalize_stats(self, accumulators: Dict) -> Dict[str, Any]:
        """
        Finalize statistics from accumulated values with numerical stability checks.
        
        Args:
            accumulators: Dictionary of accumulator states
            
        Returns:
            Dictionary of finalized statistics for each key
        """
        quantile_max_values = self.norm_cfg.get("quantile_max_values_in_memory", DEFAULT_QUANTILE_MEMORY_LIMIT)

        final_stats = {}
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"method": method}
            if key not in accumulators:
                if method not in ("none", "bool"):
                    stats["method"] = "none"
                final_stats[key] = stats
                continue

            key_acc = accumulators[key]

            if "count" in key_acc and key_acc["count"] > 1:
                mean = key_acc["mean"].item()
                std = math.sqrt(key_acc["m2"] / (key_acc["count"] - 1))
                std = max(std, 1e-6)
                if method == "standard":
                    stats.update({"mean": mean, "std": std})
                elif method == "log-standard":
                    stats.update({"log_mean": mean, "log_std": std})
                elif method == "signed-log":
                    stats.update({"mean": mean, "std": std})
            
            if "values" in key_acc and key_acc["values"]:
                if key in self._approximated_quantile_keys:
                    logger.info(
                        f"Approximating quantiles for '{key}' using a random sample of "
                        f"{quantile_max_values:,} values (out of {key_acc['total_values_seen']:,} total)."
                    )
                
                all_values = torch.cat(key_acc["values"])
                if all_values.numel() > 0:
                    stats.update(self._compute_quantile_stats(all_values, key, method))
            
            if method == "max-out":
                 max_val = max(abs(key_acc["min"].item()), abs(key_acc["max"].item()))
                 stats["max_val"] = max(max_val, 1e-10)
            
            elif method == "scaled_signed_offset_log":
                max_val, min_val = key_acc["max"].item(), key_acc["min"].item()
                m_pos = torch.log10(torch.tensor(max(0, max_val) + 1, dtype=DTYPE)).item()
                m_neg = torch.log10(torch.tensor(max(0, -min_val) + 1, dtype=DTYPE)).item()
                stats.update({"m": max(m_pos, m_neg, 1e-6)})

            final_stats[key] = stats
        return final_stats

    def _compute_quantile_stats(self, values: Tensor, key: str, method: str) -> dict:
        """
        Compute statistics for quantile-based normalization methods with stability checks.
        
        Args:
            values: Tensor of values for quantile computation
            key: Variable key name
            method: Normalization method name
            
        Returns:
            Dictionary of computed statistics
        """
        stats: Dict[str, float] = {}
        if method == "iqr":
            q_vals = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE, device=values.device))
            q1, med, q3 = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
            iqr = q3 - q1
            stats.update({"median": med, "iqr": max(iqr, 1e-6)})
            
        elif method == "log-min-max":
            log_vals = torch.log10(torch.clamp(values, min=EPSILON))
            min_v, max_v = log_vals.min().item(), log_vals.max().item()
            stats.update({"min": min_v, "max": max(max_v, min_v + 1e-6)})
            
        elif method == "symlog":
            percentile = self.norm_cfg.get("symlog_percentile", DEFAULT_SYMLOG_PERCENTILE)
            thr = torch.quantile(torch.abs(values), percentile).item()
            thr = max(thr, EPSILON)
            
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
        """
        Apply normalization to a tensor with numerical stability and clamping.
        
        Args:
            x: Input tensor to normalize
            method: Normalization method name
            stats: Statistics dictionary for the normalization
            
        Returns:
            Normalized tensor
        """
        x = x.to(DTYPE)
        
        if method in ("none", "bool") or not stats:
            return x
        
        result = x
        
        try:
            if method == "standard":
                result = (x - stats["mean"]) / stats["std"]
            elif method == "log-standard":
                x_safe = torch.log10(torch.clamp(x, min=EPSILON))
                result = (x_safe - stats["log_mean"]) / stats["log_std"]
            elif method == "signed-log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
                result = (y - stats["mean"]) / stats["std"]
            elif method == "log-min-max":
                log_x = torch.log10(torch.clamp(x, min=EPSILON))
                denom = stats["max"] - stats["min"]
                normed = (log_x - stats["min"]) / denom
                result = torch.clamp(normed, 0.0, 1.0)
            elif method == "max-out":
                result = x / stats["max_val"]
            elif method == "iqr":
                result = (x - stats["median"]) / stats["iqr"]
            elif method == "scaled_signed_offset_log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1)
                result = y / stats["m"]
            elif method == "symlog":
                thr, sf = stats["threshold"], stats["scale_factor"]
                abs_x = torch.abs(x)
                linear_mask = abs_x <= thr
                y = torch.zeros_like(x)
                y[linear_mask] = x[linear_mask] / thr
                y[~linear_mask] = torch.sign(x[~linear_mask]) * (torch.log10(abs_x[~linear_mask] / thr) + 1.0)
                result = torch.clamp(y / sf, -1.0, 1.0)
            else:
                logger.warning(f"Unsupported normalization method '{method}' passed to normalize_tensor.")
        except KeyError as e:
            logger.error(f"Missing stat '{e}' for method '{method}'. Returning raw tensor.")
            return x

        if method in ("standard", "log-standard", "signed-log", "iqr"):
            result = torch.clamp(result, -NORMALIZED_VALUE_CLAMP, NORMALIZED_VALUE_CLAMP)
        
        return result

    @staticmethod
    def denormalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """
        Apply inverse normalization to recover original scale values.
        
        Args:
            x: Normalized tensor
            method: Normalization method name
            stats: Statistics dictionary used for normalization
            
        Returns:
            Denormalized tensor in original scale
        """
        x = x.to(DTYPE)
        if method in ("none", "bool"):
            return x
        if not stats:
            raise ValueError(f"No stats for denormalization with method '{method}'")

        dtype, device = x.dtype, x.device

        def to_t(val: float) -> Tensor:
            return torch.as_tensor(val, dtype=dtype, device=device)

        if method == "standard":
            return x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))
        elif method == "log-standard":
            return 10**(x.mul(to_t(stats["log_std"])).add(to_t(stats["log_mean"])))
        elif method == "signed-log":
            unscaled_log = x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))
            return torch.sign(unscaled_log) * (10**(torch.abs(unscaled_log)) - 1.0)
        elif method == "log-min-max":
            unscaled = torch.clamp(x, 0, 1).mul(to_t(stats["max"]) - to_t(stats["min"])).add(to_t(stats["min"]))
            return 10**unscaled
        elif method == "max-out":
            return x.mul(to_t(stats["max_val"]))
        elif method == "iqr":
            return x.mul(to_t(stats["iqr"])).add(to_t(stats["median"]))
        elif method == "scaled_signed_offset_log":
            ytmp = x.mul(to_t(stats["m"]))
            return torch.sign(ytmp) * (10**(torch.abs(ytmp)) - 1)
        elif method == "symlog":
            unscaled = x.mul(to_t(stats["scale_factor"]))
            abs_unscaled = torch.abs(unscaled)
            linear_mask = abs_unscaled <= 1.0
            thr = to_t(stats["threshold"])
            y = torch.zeros_like(x)
            y[linear_mask] = unscaled[linear_mask].mul(thr)
            y[~linear_mask] = torch.sign(unscaled[~linear_mask]) * thr * (10**(abs_unscaled[~linear_mask] - 1.0))
            return y
        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")

    @staticmethod
    def denormalize(
        v: Union[Tensor, List, float, bool, None], 
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List, float, bool, None]:
        """
        Convenience wrapper for denormalization of various input types.
        
        Args:
            v: Value to denormalize (tensor, list, scalar, or None)
            metadata: Normalization metadata dictionary
            var_name: Variable name to look up normalization method
            
        Returns:
            Denormalized value in same type as input
        """
        if v is None:
            return None
        
        method = metadata["normalization_methods"].get(var_name, "none")
        if method in ("none", "bool"):
            return v
        
        stats = metadata["per_key_stats"].get(var_name)
        if not stats:
            raise ValueError(f"No stats for '{var_name}' in metadata.")

        is_scalar = not isinstance(v, (torch.Tensor, list))
        is_list = isinstance(v, list)

        if isinstance(v, torch.Tensor):
            tensor_v = v.to(DTYPE)
        else:
            tensor_v = torch.as_tensor(v, dtype=DTYPE)

        denorm_tensor = DataNormalizer.denormalize_tensor(tensor_v, method, stats)

        if is_scalar:
            return denorm_tensor.item()
        elif is_list:
            return denorm_tensor.tolist()
        else:
            return denorm_tensor


__all__ = ["DataNormalizer"]