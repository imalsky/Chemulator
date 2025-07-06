#!/usr/bin/env python3
"""
normalizer.py - Data normalization with improved numerical stability and efficiency.
This version is GPU-aware for accelerated statistics calculation.
"""
from __future__ import annotations

import h5py
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
import torch
from torch import Tensor
from tqdm import tqdm

from hardware import setup_device

logger = logging.getLogger(__name__)

# Constants
DTYPE = torch.float32
STATS_DTYPE = torch.float32


class DataNormalizer:
    """
    Handles data normalization with multiple methods and improved numerical stability.
    
    This class is GPU-aware and will use the best available device (CUDA/MPS/CPU)
    to accelerate tensor-based statistics calculations.
    """
    
    METHODS = {
        "standard", "log-standard", "log-min-max", "iqr", "max-out",
        "symlog", "signed-log", "bool", "none", "error"
    }
    QUANTILE_METHODS = {"iqr", "symlog", "log-min-max"}

    def __init__(self, *, config_data: Dict[str, Any]):
        """
        Initialize the DataNormalizer with configuration.
        
        Args:
            config_data: Configuration dictionary containing normalization settings
        """
        self.config = config_data
        self.device = setup_device()
        self.data_spec = self.config.get("data_specification", {})
        self.norm_cfg = self.config.get("normalization", {})
        
        # Get numerical constants from config
        self.num_constants = self.config.get("numerical_constants", {})
        self.epsilon = self.num_constants.get("epsilon", 1e-10)
        self.min_std = self.num_constants.get("min_std", 1e-6)
        self.normalized_value_clamp = self.num_constants.get("normalized_value_clamp", 10.0)
        self.stats_chunk_size = self.num_constants.get("stats_chunk_size", 8192)
        
        # Get normalization parameters
        self.default_quantile_memory_limit = self.norm_cfg.get(
            "quantile_max_values_in_memory", 10_000_000
        )
        self.default_symlog_percentile = self.norm_cfg.get("symlog_percentile", 0.5)
        
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()
        self._approximated_quantile_keys = set()
        logger.info(f"DataNormalizer initialized on device '{self.device}'.")

    # ────────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ────────────────────────────────────────────────────────────────────────────

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """
        Parse configuration to determine which keys to process and their normalization methods.
        
        Returns:
            Tuple of (set of keys to process, dict mapping keys to methods)
        """
        all_vars = set(self.data_spec.get("all_variables", []))
        user_key_methods = self.norm_cfg.get("key_methods", {})
        default_method = self.norm_cfg.get("default_method", "error")

        key_methods = {}
        for key in all_vars:
            if key in user_key_methods:
                method = user_key_methods[key].lower()
            else:
                if default_method == "error":
                    raise ValueError(
                        f"No normalization method specified for variable '{key}' "
                        f"and default_method is set to 'error'. Please specify a method "
                        f"for this variable in normalization.key_methods."
                    )
                method = default_method.lower()
            
            if method not in self.METHODS:
                raise ValueError(f"Unsupported normalization method '{method}' for key '{key}'.")
            
            key_methods[key] = method

        return all_vars, key_methods

    # ────────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────────

    def calculate_stats(self, h5_path: Path, train_indices: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculate normalization statistics by streaming data in chunks for memory efficiency.
        
        Note: The raw_data return value is kept for compatibility but can be ignored
        when using streaming datasets.
        
        Args:
            h5_path: Path to HDF5 file
            train_indices: List of training sample indices
            
        Returns:
            Tuple of (normalization metadata dict, empty dict for raw data)
        """
        logger.info(f"Calculating statistics from {len(train_indices)} training samples...")
        if not train_indices:
            raise ValueError("Cannot calculate statistics from empty training indices.")
        
        # Verify HDF5 file exists and is readable
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        # Sort indices for efficient HDF5 access
        sorted_train_indices = sorted(train_indices)
        
        accumulators = None
        
        try:
            with h5py.File(h5_path, 'r', swmr=True, libver='latest') as hf:
                available_keys = self.keys_to_process.intersection(hf.keys())
                if len(available_keys) != len(self.keys_to_process):
                    missing = self.keys_to_process - available_keys
                    logger.warning(f"Keys not found in HDF5 and will be skipped: {missing}")
                
                if not available_keys:
                    raise ValueError(f"No requested keys found in HDF5 file. Requested: {self.keys_to_process}, Available in HDF5: {set(hf.keys())}")
                
                logger.info(f"Processing {len(available_keys)} variables: {sorted(available_keys)}")
                
                accumulators = self._initialize_accumulators(available_keys)

                # Validate indices are within bounds
                for key in available_keys:
                    dataset_shape = hf[key].shape
                    max_index = max(train_indices)
                    if max_index >= dataset_shape[0]:
                        raise ValueError(
                            f"Index {max_index} out of bounds for dataset '{key}' "
                            f"with shape {dataset_shape}"
                        )
                
                logger.info("Reading training data and calculating stats in chunks...")
                start_time = time.time()
                
                # Process data in chunks for efficiency
                with tqdm(total=len(sorted_train_indices), desc="Calculating Stats") as pbar:
                    for chunk_start in range(0, len(sorted_train_indices), self.stats_chunk_size):
                        chunk_end = min(chunk_start + self.stats_chunk_size, len(sorted_train_indices))
                        chunk_indices = sorted_train_indices[chunk_start:chunk_end]
                        
                        if not chunk_indices:
                            continue
                        
                        batch_of_tensors = {}
                        for key in available_keys:
                            try:
                                # Read data for this chunk of indices
                                data_chunk_np = hf[key][chunk_indices]
                                batch_of_tensors[key] = torch.from_numpy(data_chunk_np)
                            except Exception as e:
                                logger.error(f"Error reading key '{key}' at indices {chunk_indices[:5]}... : {e}")
                                raise
                        
                        self._update_accumulators_with_batch(batch_of_tensors, accumulators)
                        pbar.update(len(chunk_indices))
                
                load_time = time.time() - start_time
                logger.info(f"Finished calculating stats in {load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Error during statistics calculation: {repr(e)}", exc_info=True)
            raise RuntimeError(f"Failed to calculate statistics: {repr(e)}") from e

        computed_stats = self._finalize_stats(accumulators)
        metadata = {"normalization_methods": self.key_methods, "per_key_stats": computed_stats}
        
        logger.info("Statistics calculation complete.")
        # Return empty dict for raw_data since we're using streaming
        return metadata, {}

    # ────────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────────

    def _initialize_accumulators(self, keys_to_process: Set[str]) -> Dict[str, Dict[str, Any]]:
        """
        Initialize data structures for online statistics accumulation using Welford's algorithm.
        """
        accumulators = {}
        for key in keys_to_process:
            method = self.key_methods[key]
            if method in ("none", "bool", "error"):
                continue

            acc: Dict[str, Any] = {}

            if method in ("standard", "log-standard", "signed-log"):
                acc.update({
                    "count": 0,
                    "mean": torch.tensor(0.0, dtype=STATS_DTYPE, device='cpu'),
                    "m2": torch.tensor(0.0, dtype=STATS_DTYPE, device='cpu')
                })

            if method in self.QUANTILE_METHODS:
                acc["values"] = []
                acc["total_values_seen"] = 0
            
            if method in ("max-out", "log-min-max"):
                acc.update({
                    "min": torch.tensor(float('inf'), dtype=DTYPE, device=self.device),
                    "max": torch.tensor(float('-inf'), dtype=DTYPE, device=self.device)
                })

            if acc:
                accumulators[key] = acc
        return accumulators

    # ---------------------------------------------------------------------------

    def _update_accumulators_with_batch(
        self,
        batch: Dict[str, torch.Tensor],
        accumulators: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Stream a mini-batch into the running statistics.
        """
        quantile_cap = self.default_quantile_memory_limit

        for key, data_batch in batch.items():
            if key not in accumulators:
                continue

            data = data_batch.to(device=self.device, dtype=DTYPE).flatten()
            valid_data = data[torch.isfinite(data)]
            if valid_data.numel() == 0:
                continue

            method = self.key_methods[key]
            key_acc = accumulators[key]

            # choose the space that matches the normalisation method
            if method == "log-standard":
                trans = torch.log10(torch.clamp(valid_data, min=self.epsilon))
            elif method == "signed-log":
                trans = torch.sign(valid_data) * torch.log10(torch.abs(valid_data) + 1.0)
            else:
                trans = valid_data
            data_cpu = trans.to(device="cpu", dtype=STATS_DTYPE)

            # ── online mean / variance (Welford) ────────────────────────────────
            if "count" in key_acc:
                n_new = data_cpu.numel()
                if n_new:
                    c_old, μ_old, m2_old = key_acc["count"], key_acc["mean"], key_acc["m2"]
                    c_new = c_old + n_new
                    δ = data_cpu.mean() - μ_old
                    μ_new = μ_old + δ * (n_new / c_new)
                    m2_new = m2_old + torch.sum((data_cpu - μ_old) * (data_cpu - μ_new))
                    key_acc.update({"count": c_new, "mean": μ_new, "m2": m2_new})

            # ── collect values for quantile-based methods ───────────────────────
            if "values" in key_acc:
                key_acc["total_values_seen"] += valid_data.numel()
                cur = sum(t.numel() for t in key_acc["values"])
                if cur + valid_data.numel() <= quantile_cap:
                    key_acc["values"].append(valid_data.cpu())   # save VRAM
                else:
                    self._approximated_quantile_keys.add(key)
                    combined = torch.cat(key_acc["values"] + [valid_data.cpu()])
                    perm = torch.randperm(combined.numel())[:quantile_cap]
                    key_acc["values"] = [combined[perm]]

            # ── running extrema for min–max / max-out families ─────────────────
            if "max" in key_acc:
                if method == "log-min-max":
                    log_vals = torch.log10(torch.clamp(valid_data, min=self.epsilon))
                    key_acc["min"] = torch.min(key_acc["min"], log_vals.min())
                    key_acc["max"] = torch.max(key_acc["max"], log_vals.max())
                else:
                    key_acc["min"] = torch.min(key_acc["min"], valid_data.min())
                    key_acc["max"] = torch.max(key_acc["max"], valid_data.max())

    # ---------------------------------------------------------------------------

    def _finalize_stats(self, accumulators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Turn the raw accumulators into a plain-Python stats dict that the
        (de)normalisation routines can consume.
        """
        quantile_cap = self.default_quantile_memory_limit
        final_stats: Dict[str, Any] = {}

        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"method": method}
            if key not in accumulators:
                if method not in ("none", "bool", "error"):
                    stats["method"] = "none"
                final_stats[key] = stats
                continue

            acc = accumulators[key]

            # mean / std for *-standard methods --------------------------------
            if "count" in acc and acc["count"] > 1:
                mean = acc["mean"].item()
                std = math.sqrt(acc["m2"].item() / (acc["count"] - 1))
                std = max(std, self.min_std)
                if method == "standard":
                    stats.update({"mean": mean, "std": std})
                elif method == "log-standard":
                    stats.update({"log_mean": mean, "log_std": std})
                elif method == "signed-log":
                    stats.update({"mean": mean, "std": std})

            # quantile-based helpers ------------------------------------------
            if "values" in acc and acc["values"]:
                if key in self._approximated_quantile_keys:
                    logger.info(
                        f"Approximating quantiles for '{key}' with "
                        f"{quantile_cap:,} samples (out of "
                        f"{acc['total_values_seen']:,})."
                    )
                all_vals = torch.cat(acc["values"])
                if all_vals.numel():
                    stats.update(self._compute_quantile_stats(all_vals, key, method))

            # simple extrema-based methods ------------------------------------
            if method == "max-out":
                max_val = max(abs(acc["min"].item()), abs(acc["max"].item()))
                stats["max_val"] = max(max_val, self.epsilon)

            final_stats[key] = stats

        return final_stats

    # ---------------------------------------------------------------------------

    def _compute_quantile_stats(self, values: Tensor, key: str, method: str) -> dict:
        """
        Compute statistics for quantile-based normalization methods with stability checks.
        """
        stats: Dict[str, float] = {}

        def _robust_quantile_computation(tensor: Tensor, q_values: Union[float, Tensor]) -> Tensor:
            """Internal helper to compute quantiles with a fallback to subsampling."""
            try:
                return torch.quantile(tensor, q_values)
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "too large" in error_msg or "out of memory" in error_msg:
                    fallback_subsample_size = 10_000_000
                    if tensor.numel() <= fallback_subsample_size:
                        logger.error(f"Quantile computation failed for '{key}' on an already small tensor ({tensor.numel()}).")
                        raise e
                    logger.warning(
                        f"Quantile computation failed for '{key}' on tensor of size {tensor.numel():,}. "
                        f"Retrying with a random subsample of {fallback_subsample_size:,}."
                    )
                    perm = torch.randperm(tensor.numel(), device=tensor.device)[:fallback_subsample_size]
                    subsampled_tensor = tensor.flatten()[perm]
                    return torch.quantile(subsampled_tensor, q_values)
                else:
                    raise e
        
        if method == "iqr":
            q_tensor = torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE, device=values.device)
            q_vals = _robust_quantile_computation(values, q_tensor)
            q1, med, q3 = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
            iqr = q3 - q1
            stats.update({"median": med, "iqr": max(iqr, self.min_std)})
            
        elif method == "log-min-max":
            log_vals = torch.log10(torch.clamp(values, min=self.epsilon))
            min_v, max_v = log_vals.min().item(), log_vals.max().item()
            stats.update({"min": min_v, "max": max(max_v, min_v + self.min_std)})
            
        elif method == "symlog":
            percentile = self.default_symlog_percentile
            thr = _robust_quantile_computation(torch.abs(values), percentile).item()
            thr = max(thr, self.epsilon)
            
            abs_v = torch.abs(values)
            mask = abs_v > thr
            transformed = torch.zeros_like(values)
            transformed[mask] = torch.sign(values[mask]) * (torch.log10(abs_v[mask] / thr) + 1)
            transformed[~mask] = values[~mask] / thr
            
            sf = transformed.abs().max().item() if transformed.numel() > 0 else 1.0
            stats.update({"threshold": thr, "scale_factor": max(sf, 1.0)})
        return stats

    # ---------------------------------------------------------------------------
    # Normalise / denormalise
    # ---------------------------------------------------------------------------

    def normalize_tensor(self, x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """
        Apply normalization to a tensor with numerical stability and clamping.
        """
        x = x.to(DTYPE)
        
        if method in ("none", "bool") or not stats:
            return x
        
        result = x
        
        try:
            if method == "standard":
                result = (x - stats["mean"]) / stats["std"]
            elif method == "log-standard":
                x_safe = torch.log10(torch.clamp(x, min=self.epsilon))
                result = (x_safe - stats["log_mean"]) / stats["log_std"]
            elif method == "signed-log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
                result = (y - stats["mean"]) / stats["std"]
            elif method == "log-min-max":
                log_x = torch.log10(torch.clamp(x, min=self.epsilon))
                denom = stats["max"] - stats["min"]
                normed = (log_x - stats["min"]) / (denom if denom > 0 else 1.0)
                result = torch.clamp(normed, 0.0, 1.0)
            elif method == "max-out":
                result = x / stats["max_val"]
            elif method == "iqr":
                result = (x - stats["median"]) / stats["iqr"]
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
            result = torch.clamp(result, -self.normalized_value_clamp, self.normalized_value_clamp)
        
        return result

    def denormalize_tensor(self, x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """
        Apply inverse normalization to recover original scale values.
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

    # ---------------------------------------------------------------------------

    @staticmethod
    def denormalize(
        v: Union[Tensor, List, float, bool, None], 
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List, float, bool, None]:
        """
        Convenience wrapper for denormalization of various input types.
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

        # Create a temporary normalizer instance to access the config
        temp_normalizer = DataNormalizer(config_data={"numerical_constants": {"epsilon": 1e-10}})
        denorm_tensor = temp_normalizer.denormalize_tensor(tensor_v, method, stats)

        if is_scalar:
            return denorm_tensor.item()
        elif is_list:
            return denorm_tensor.tolist()
        else:
            return denorm_tensor


__all__ = ["DataNormalizer"]