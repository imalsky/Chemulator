#!/usr/bin/env python3
"""
Generate latent dataset for DeepONet training (Stage 2 of paper).
Uses normalized time coordinates matching trunk network expectations.
FIXED: Time normalization validation, vectorized encoding for efficiency
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from utils import load_json, save_json
from normalizer import NormalizationHelper


def generate_latent_dataset(
        model: torch.nn.Module,
        input_dir: Path,
        output_dir: Path,
        config: Dict[str, Any],
        device: torch.device
) -> None:
    """Generate latent dataset after autoencoder pretraining.

    If config['latent_generation']['mode'] == 'all', we encode ALL available
    time points from the preprocessed shards (no interpolation). This mode
    REQUIRES all trajectories to share an identical time grid.

    Otherwise, we select fixed times (nearest neighbor to source grid).
    """
    logger = logging.getLogger(__name__)
    if device.type != "cuda":
        raise RuntimeError("Latent dataset generation requires CUDA device.")

    # Early exit if latent dataset already exists
    if output_dir.exists() and (output_dir / "latent_shard_index.json").exists():
        logger.warning(f"Latent dataset already exists at {output_dir}. Skipping generation.")
        return

    logger.info("Generating latent dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input metadata and normalization
    shard_index = load_json(input_dir / "shard_index.json")
    norm_stats = load_json(input_dir / "normalization.json")

    data_cfg = config["data"]
    time_var = data_cfg["time_variable"]

    # Log time normalization method but don't enforce specific method
    nm = norm_stats.get("normalization_methods", {})
    time_method = nm.get(time_var, "unknown")
    logger.info(f"Time variable '{time_var}' uses '{time_method}' normalization")

    # Setup helpers and variables
    norm_helper = NormalizationHelper(norm_stats, device, config)
    species_vars = data_cfg["species_variables"]
    global_vars = data_cfg["global_variables"]
    expected = data_cfg.get("expected_globals", ["P", "T"])
    if global_vars != expected:
        raise ValueError(f"Global variables mismatch: got {global_vars}, expected {expected}.")
    logger.info(f"Using global variables: {global_vars} (P=pressure, T=temperature)")

    # Determine latent time mode
    lg_cfg = config.get("latent_generation", {})
    lg_mode = (lg_cfg.get("mode") or "all").lower()
    if lg_mode not in ("all", "fixed"):
        raise ValueError(f"latent_generation.mode must be 'all' or 'fixed', got '{lg_mode}'")

    # If fixed, pick list; else will fill from source
    fixed_times_list = None
    if lg_mode == "fixed":
        fixed_times_list = lg_cfg.get("fixed_times") or config["model"].get("trunk_times")
        if not fixed_times_list:
            raise ValueError("latent_generation.mode='fixed' requires 'fixed_times' or model.trunk_times.")
        # Enforce sorted, unique, in-range fixed times
        fixed_times = sorted(float(t) for t in fixed_times_list)
        fixed_times_unique = []
        for t in fixed_times:
            if not fixed_times_unique or abs(t - fixed_times_unique[-1]) > 1e-12:
                fixed_times_unique.append(t)
        if not all(0.0 <= t <= 1.0 for t in fixed_times_unique):
            raise ValueError(f"Fixed times must be in [0,1], got {fixed_times_list}")
        fixed_times_list = fixed_times_unique

    # Build latent shard index
    new_shard_index = {
        "latent_mode": True,
        "latent_dim": int(config["model"]["latent_dim"]),
        "trunk_times": None,  # set later
        "splits": {}
    }

    model.eval()
    with torch.no_grad():
        shared_time_vec = None
        time_tolerance = 1e-7  # Tolerance for time grid comparison

        for split_name in ["train", "validation", "test"]:
            logger.info(f"Processing {split_name} split...")
            split_dir = input_dir / split_name
            output_split_dir = output_dir / split_name
            output_split_dir.mkdir(parents=True, exist_ok=True)

            split_info = shard_index["splits"][split_name]
            new_shards = []

            for shard_info in tqdm(split_info["shards"], desc=split_name):
                shard_path = split_dir / shard_info["filename"]

                # Load raw tensors (CPU), cast to float32
                with np.load(shard_path, allow_pickle=False) as data:
                    x0_np = data["x0"].astype(np.float32)  # [N, S]
                    y_np = data["y_mat"].astype(np.float32)  # [N, M, S]
                    g_np = data["globals"].astype(np.float32)  # [N, G]
                    t_np = data["t_vec"].astype(np.float32)  # [M] or [N, M]

                # Convert to tensors (CPU first)
                x0 = torch.from_numpy(x0_np)
                y_mat = torch.from_numpy(y_np)  # [N, M, S]
                globals_vec = torch.from_numpy(g_np)  # [N, G]

                # Normalize
                x0_norm = norm_helper.normalize(x0, species_vars).to(device)  # [N, S]
                y_norm = norm_helper.normalize(y_mat, species_vars).to(device)  # [N, M, S]
                globals_norm = norm_helper.normalize(globals_vec, global_vars)  # [N, G]

                # Normalize time to [0,1]
                if t_np.ndim == 1:
                    t_vec = torch.from_numpy(t_np)  # [M]
                    t_norm = norm_helper.normalize(t_vec.unsqueeze(-1), [time_var]).squeeze(-1).to(device)  # [M]
                    per_sample_times = False
                elif t_np.ndim == 2:
                    t_vec = torch.from_numpy(t_np)  # [N, M]
                    t_norm = norm_helper.normalize(t_vec.unsqueeze(-1), [time_var]).squeeze(-1).to(device)  # [N, M]
                    per_sample_times = True
                else:
                    raise ValueError(f"'t_vec' must be [M] or [N,M], got shape {t_np.shape}")

                # Validate normalized time is strictly in [0,1]
                t_min, t_max = t_norm.min().item(), t_norm.max().item()
                if t_min < -1e-12 or t_max > 1.0 + 1e-12:
                    raise ValueError(
                        f"Normalized times must be in [0,1]. Got range [{t_min:.6f}, {t_max:.6f}]. "
                        f"Check normalization for '{time_var}'."
                    )

                # Monotonicity checks
                if per_sample_times:
                    N, M = t_norm.shape
                    for n in range(N):
                        diffs = torch.diff(t_norm[n])
                        if (diffs < -1e-12).any():
                            raise ValueError(f"Trajectory {n} has non-monotonic normalized time.")
                else:
                    M = int(t_norm.shape[0])
                    diffs = torch.diff(t_norm)
                    if (diffs < -1e-12).any():
                        raise ValueError("Shared time grid must be non-decreasing.")

                # Select times and targets
                if lg_mode == "all":
                    if per_sample_times:
                        # STRICT requirement: all trajectories must share identical time grid
                        time_diff = (t_norm - t_norm[0].unsqueeze(0)).abs().max().item()
                        if time_diff > time_tolerance:
                            raise RuntimeError(
                                f"latent_generation.mode='all' requires identical time grids across all trajectories. "
                                f"Found maximum time difference of {time_diff:.2e} (tolerance: {time_tolerance:.2e}). "
                                f"Consider using mode='fixed' or ensuring consistent time grids in preprocessing."
                            )
                        times_used = t_norm[0]  # [M]
                        logger.debug(f"Per-sample times detected, verified identical (diff={time_diff:.2e})")
                    else:
                        times_used = t_norm  # [M]
                    y_selected = y_norm  # [N, M, S]
                    trunk_times_tensor = times_used.to(device)  # [M]
                else:
                    # Fixed: nearest-neighbor to requested normalized times
                    req = torch.tensor([float(t) for t in fixed_times_list],
                                       dtype=torch.float32, device=device)  # [K]

                    if per_sample_times:
                        # For fixed mode with per-sample times, still require shared grid
                        time_diff = (t_norm - t_norm[0].unsqueeze(0)).abs().max().item()
                        if time_diff > time_tolerance:
                            raise RuntimeError(
                                f"latent_generation.mode='fixed' with per-sample times requires identical grids. "
                                f"Found maximum difference of {time_diff:.2e}."
                            )
                        diffs = (t_norm[0].unsqueeze(0) - req.unsqueeze(1)).abs()  # [K, M]
                        idx = diffs.argmin(dim=1)  # [K]
                        y_selected = y_norm[:, idx, :]  # [N, K, S]
                        trunk_times_tensor = t_norm[0][idx]  # ACTUAL NN times
                    else:
                        diffs = (t_norm.unsqueeze(0) - req.unsqueeze(1)).abs()  # [K, M]
                        idx = diffs.argmin(dim=1)  # [K]
                        y_selected = y_norm[:, idx, :]  # [N, K, S]
                        trunk_times_tensor = t_norm[idx]  # ACTUAL NN times

                # Vectorized encoding with AMP and chunking to avoid OOM
                N = x0_norm.shape[0]
                latent_dim = int(config["model"]["latent_dim"])
                K = int(y_selected.size(1))

                # Encode initial conditions
                from torch.amp import autocast  # local import to avoid changing file-level imports
                with autocast('cuda', enabled=True):
                    z0 = model.encode(x0_norm)  # [N, L]

                # Encode all time points in chunks: [N*K, S] -> [N*K, L]
                y_flat = y_selected.reshape(N * K, -1)
                chunk = int(lg_cfg.get("encode_chunk_size", 65536))
                z_chunks = []
                with autocast('cuda', enabled=True):
                    for start in range(0, y_flat.size(0), chunk):
                        end = min(start + chunk, y_flat.size(0))
                        z_chunks.append(model.encode(y_flat[start:end]))
                z_flat = torch.cat(z_chunks, dim=0)
                z_targets = z_flat.reshape(N, K, latent_dim)  # [N, K, L]

                # Move to CPU for saving
                z0_cpu = z0.to(dtype=torch.float32, device='cpu')
                globals_cpu = globals_norm.to(dtype=torch.float32, device='cpu')
                z_targets_cpu = z_targets.to(dtype=torch.float32, device='cpu')
                latent_inputs = torch.cat([z0_cpu, globals_cpu], dim=1)  # [N, L+2]

                # Save compressed latent shard
                output_filename = f"latent_{shard_info['filename']}"
                output_path = output_split_dir / output_filename
                np.savez_compressed(
                    output_path,
                    latent_inputs=latent_inputs.numpy(),  # [N, L+2]
                    latent_targets=z_targets_cpu.numpy()  # [N, K, L]
                )

                # Record shard
                new_shards.append({
                    "filename": output_filename,
                    "n_trajectories": shard_info["n_trajectories"]
                })

                # Enforce consistent time vector across shards
                times_list = trunk_times_tensor.to('cpu', dtype=torch.float32).numpy().tolist()
                if shared_time_vec is None:
                    shared_time_vec = times_list
                    logger.info(f"Established shared time grid with {len(times_list)} points")
                else:
                    if len(shared_time_vec) != len(times_list):
                        raise RuntimeError(
                            f"Inconsistent time grid lengths: {len(shared_time_vec)} vs {len(times_list)}. "
                            f"All shards must have identical time grids."
                        )
                    max_diff = max(abs(a - b) for a, b in zip(shared_time_vec, times_list))
                    if max_diff > time_tolerance:
                        raise RuntimeError(
                            f"Inconsistent trunk_times across shards (max diff: {max_diff:.2e}). "
                            f"Verify preprocessing generates consistent time grids."
                        )

                torch.cuda.empty_cache()

            new_shard_index["splits"][split_name] = {
                "shards": new_shards,
                "n_trajectories": split_info["n_trajectories"]
            }

    # Finalize index and copy normalization
    if shared_time_vec is None:
        raise RuntimeError("No shards processed; cannot finalize latent index.")

    # Ensure shared grid is non-decreasing
    if any(shared_time_vec[i] > shared_time_vec[i + 1] for i in range(len(shared_time_vec) - 1)):
        raise RuntimeError("Shared trunk_times must be non-decreasing.")

    new_shard_index["trunk_times"] = shared_time_vec
    save_json(new_shard_index, output_dir / "latent_shard_index.json")
    save_json(norm_stats, output_dir / "normalization.json")
    _compute_latent_statistics(output_dir, new_shard_index, config)
    logger.info(f"Latent dataset saved to {output_dir} with {len(shared_time_vec)} shared time points")


def _compute_latent_statistics(output_dir: Path, shard_index: Dict, config: Dict) -> None:
    """
    Compute mean and std of latent space for optional standardization during DeepONet training.
    """
    logger = logging.getLogger(__name__)

    # Accumulate statistics over training data only
    latent_dim = shard_index["latent_dim"]
    n_timepoints = len(shard_index["trunk_times"])

    # Running statistics
    n_samples = 0
    sum_inputs = None
    sum_inputs_sq = None
    sum_targets = None
    sum_targets_sq = None

    train_dir = output_dir / "train"
    for shard_info in shard_index["splits"]["train"]["shards"]:
        shard_path = train_dir / shard_info["filename"]

        with np.load(shard_path, allow_pickle=False) as data:
            inputs = data["latent_inputs"].astype(np.float64)  # [N, latent_dim + 2]
            targets = data["latent_targets"].astype(np.float64)  # [N, M, latent_dim]

            batch_size = inputs.shape[0]
            n_samples += batch_size

            # Accumulate sums
            if sum_inputs is None:
                sum_inputs = np.sum(inputs, axis=0)
                sum_inputs_sq = np.sum(inputs ** 2, axis=0)
                sum_targets = np.sum(targets, axis=(0, 1))  # Sum over batch and time
                sum_targets_sq = np.sum(targets ** 2, axis=(0, 1))
            else:
                sum_inputs += np.sum(inputs, axis=0)
                sum_inputs_sq += np.sum(inputs ** 2, axis=0)
                sum_targets += np.sum(targets, axis=(0, 1))
                sum_targets_sq += np.sum(targets ** 2, axis=(0, 1))

    # Compute statistics
    if n_samples > 0:
        # Inputs statistics
        mean_inputs = sum_inputs / n_samples
        var_inputs = (sum_inputs_sq / n_samples) - (mean_inputs ** 2)
        std_inputs = np.sqrt(np.maximum(var_inputs, 1e-10))

        # Targets statistics (per latent dimension, averaged over time)
        n_target_samples = n_samples * n_timepoints
        mean_targets = sum_targets / n_target_samples
        var_targets = (sum_targets_sq / n_target_samples) - (mean_targets ** 2)
        std_targets = np.sqrt(np.maximum(var_targets, 1e-10))

        latent_stats = {
            "n_samples": int(n_samples),
            "latent_dim": int(latent_dim),
            "n_timepoints": int(n_timepoints),
            "input_stats": {
                "mean": mean_inputs.tolist(),
                "std": std_inputs.tolist()
            },
            "target_stats": {
                "mean": mean_targets.tolist(),
                "std": std_targets.tolist()
            }
        }

        # Save statistics
        save_json(latent_stats, output_dir / "latent_statistics.json")

        logger.info(f"Latent statistics computed from {n_samples} training samples")
        logger.info(f"  Input latent mean range: [{np.min(mean_inputs):.3f}, {np.max(mean_inputs):.3f}]")
        logger.info(f"  Input latent std range: [{np.min(std_inputs):.3f}, {np.max(std_inputs):.3f}]")
        logger.info(f"  Target latent mean range: [{np.min(mean_targets):.3f}, {np.max(mean_targets):.3f}]")
        logger.info(f"  Target latent std range: [{np.min(std_targets):.3f}, {np.max(std_targets):.3f}]")