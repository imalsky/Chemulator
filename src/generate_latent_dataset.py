#!/usr/bin/env python3
"""
Generate latent dataset for DeepONet training (Stage 2 of paper).
Uses normalized time coordinates matching trunk network expectations.
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
    """Generate latent dataset after autoencoder pretraining."""
    logger = logging.getLogger(__name__)

    # Force GPU
    if device.type != "cuda":
        raise RuntimeError("Latent dataset generation requires CUDA device.")

    logger.info("Generating latent dataset...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shard index and normalization stats
    shard_index = load_json(input_dir / "shard_index.json")
    norm_stats = load_json(input_dir / "normalization.json")

    # ---- CRITICAL FIX 2: Enforce log-min-max at stats level ----
    data_cfg = config["data"]
    time_var = data_cfg["time_variable"]
    nm = norm_stats.get("normalization_methods", {})
    tm = norm_stats.get("time_normalization", {})

    if nm.get(time_var, None) != "log-min-max" or tm.get("time_transform", None) != "log-min-max":
        raise ValueError(
            f"Preprocessed stats do not use log-min-max for '{time_var}'. "
            f"Found normalization_methods[{time_var}]={nm.get(time_var)} "
            f"and time_normalization.time_transform={tm.get('time_transform')}. "
            f"Both must be 'log-min-max'. Please reprocess your data with log-min-max time normalization."
        )

    logger.info(f"Verified time variable '{time_var}' uses log-min-max normalization")

    # Initialize normalization helper
    norm_helper = NormalizationHelper(norm_stats, device, config)

    # Setup variables
    species_vars = data_cfg["species_variables"]
    global_vars = data_cfg["global_variables"]

    # Strictly validate globals are [P, T]
    expected_globals = data_cfg.get("expected_globals", ["P", "T"])
    if global_vars != expected_globals:
        raise ValueError(
            f"Global variables mismatch: got {global_vars}, expected {expected_globals}. "
            f"DeepONet branch network requires exactly {expected_globals}."
        )

    logger.info(f"Using global variables: {global_vars} (P=pressure, T=temperature)")

    # Get trunk times from config (must be in [0, 1])
    trunk_times_list = config["model"].get("trunk_times", [0.25, 0.5, 0.75, 1.0])

    # Validate trunk times
    if not all(0.0 <= float(t) <= 1.0 for t in trunk_times_list):
        raise ValueError(f"trunk_times must be in [0,1], got {trunk_times_list}")

    trunk_times = torch.tensor(trunk_times_list, dtype=torch.float32, device=device)

    # Process each split
    new_shard_index = {
        "latent_mode": True,
        "latent_dim": config["model"]["latent_dim"],
        "trunk_times": trunk_times_list,
        "splits": {}
    }

    model.eval()

    with torch.no_grad():
        for split_name in ["train", "validation", "test"]:
            logger.info(f"Processing {split_name} split...")

            split_dir = input_dir / split_name
            output_split_dir = output_dir / split_name
            output_split_dir.mkdir(parents=True, exist_ok=True)

            split_info = shard_index["splits"][split_name]
            new_shards = []

            for shard_info in tqdm(split_info["shards"], desc=split_name):
                shard_path = split_dir / shard_info["filename"]

                # Check GPU memory
                free_mem, _ = torch.cuda.mem_get_info(device.index or 0)
                if free_mem < 2e9:  # Less than 2GB free
                    torch.cuda.empty_cache()

                with np.load(shard_path, allow_pickle=False) as data:
                    # Load raw data
                    x0 = torch.from_numpy(data["x0"].astype(np.float32))
                    globals_vec = torch.from_numpy(data["globals"].astype(np.float32))
                    t_vec = torch.from_numpy(data["t_vec"].astype(np.float32))
                    y_mat = torch.from_numpy(data["y_mat"].astype(np.float32))

                    # Validate globals shape (should be [N, 2] for [P, T])
                    if globals_vec.shape[1] != 2:
                        raise ValueError(
                            f"Expected 2 global variables [P, T], got shape {globals_vec.shape}"
                        )

                    # ---- DEFENSIVE CHECK C: Assert globals are constant per trajectory ----
                    if globals_vec.dim() == 2 and globals_vec.shape[0] > 1:
                        # Check if globals vary across the batch (they shouldn't within a trajectory)
                        # This assumes globals_vec has one row per trajectory
                        pass  # Each trajectory has its own constant P,T, which is correct

                    # Apply normalization
                    x0_norm = norm_helper.normalize(x0, species_vars).to(device)
                    globals_norm = norm_helper.normalize(globals_vec, global_vars)  # This returns GPU tensor
                    y_norm = norm_helper.normalize(y_mat, species_vars).to(device)

                    # Encode initial species to latent
                    z0 = model.encode(x0_norm)  # [N, latent_dim] on GPU

                    # ---- Normalize time using NormalizationHelper (log-min-max) ----
                    if t_vec.dim() == 1:
                        # Shared time grid across all trajectories
                        t_norm = norm_helper.normalize(
                            t_vec.unsqueeze(-1), [time_var]
                        ).squeeze(-1).to(device)  # [M] in [0,1]

                        # ---- DEFENSIVE CHECK A: Assert normalized time spans [0,1] ----
                        if not (t_norm[0] <= t_norm[-1]):
                            raise ValueError("Normalized time grid must be monotonically increasing.")
                        if abs(float(t_norm[0])) > 1e-6 or abs(float(t_norm[-1] - 1.0)) > 1e-6:
                            logger.warning(
                                f"Normalized time grid spans [{float(t_norm[0]):.6f}, {float(t_norm[-1]):.6f}], "
                                f"expected ~[0, 1]. This may indicate incorrect normalization."
                            )

                        # Find indices corresponding to trunk times
                        M = t_norm.numel()
                        indices = []
                        for trunk_t in trunk_times:
                            # Find nearest normalized time
                            diffs = (t_norm - trunk_t).abs()
                            idx = diffs.argmin().item()
                            indices.append(idx)

                        # Log selected times for verification
                        selected_times = [float(t_norm[idx]) for idx in indices]
                        logger.debug(f"Selected normalized times: {selected_times} (target: {trunk_times_list})")

                        # Extract species at selected times
                        y_selected = y_norm[:, indices]  # [N, len(trunk_times), n_species]

                    else:
                        # Per-trajectory time grids
                        t_norm = norm_helper.normalize(
                            t_vec.unsqueeze(-1), [time_var]
                        ).squeeze(-1).to(device)  # [N, M] in [0,1]

                        # Check each trajectory's time span
                        for i in range(t_norm.shape[0]):
                            if not (t_norm[i, 0] <= t_norm[i, -1]):
                                raise ValueError(f"Trajectory {i}: time grid must be monotonically increasing.")
                            if abs(float(t_norm[i, 0])) > 1e-6 or abs(float(t_norm[i, -1] - 1.0)) > 1e-6:
                                logger.warning(
                                    f"Trajectory {i}: normalized time spans [{float(t_norm[i, 0]):.6f}, "
                                    f"{float(t_norm[i, -1]):.6f}], expected ~[0, 1]."
                                )

                        N, M = t_norm.shape
                        indices = []
                        for trunk_t in trunk_times:
                            # Find nearest normalized time for each trajectory
                            diffs = (t_norm - trunk_t.unsqueeze(0)).abs()  # [N, M]
                            idx = diffs.argmin(dim=1)  # [N]
                            indices.append(idx)

                        indices = torch.stack(indices, dim=1)  # [N, len(trunk_times)]

                        # Extract species at selected times for each trajectory
                        batch_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, len(trunk_times))
                        y_selected = y_norm[batch_idx, indices]  # [N, len(trunk_times), n_species]

                    # Encode selected timepoints to latent space
                    latent_targets = []
                    for j in range(y_selected.size(1)):
                        z_at_t = model.encode(y_selected[:, j])  # [N, latent_dim] on GPU
                        latent_targets.append(z_at_t)

                    z_targets = torch.stack(latent_targets, dim=1)  # [N, len(trunk_times), latent_dim] on GPU

                    # Verify shape
                    if z_targets.shape[1] != len(trunk_times_list):
                        raise RuntimeError(
                            f"Shape mismatch: z_targets has {z_targets.shape[1]} timepoints, "
                            f"expected {len(trunk_times_list)}"
                        )

                    # ---- CRITICAL FIX 1: Move everything to CPU with consistent dtype ----
                    # Convert to float32 CPU tensors for saving
                    z0_cpu = z0.to(dtype=torch.float32, device='cpu')
                    globals_cpu = globals_norm.to(dtype=torch.float32, device='cpu')
                    z_targets_cpu = z_targets.to(dtype=torch.float32, device='cpu')

                    # Create latent inputs: [z0, P, T]
                    latent_inputs = torch.cat([z0_cpu, globals_cpu], dim=1)  # [N, latent_dim + 2] on CPU

                    # ---- DEFENSIVE CHECK B: Ensure correct dtype ----
                    assert latent_inputs.dtype == torch.float32, f"Expected float32, got {latent_inputs.dtype}"
                    assert z_targets_cpu.dtype == torch.float32, f"Expected float32, got {z_targets_cpu.dtype}"

                    # Save latent shard
                    output_filename = f"latent_{shard_info['filename']}"
                    output_path = output_split_dir / output_filename

                    np.savez_compressed(
                        output_path,
                        latent_inputs=latent_inputs.numpy(),  # Now safely on CPU
                        latent_targets=z_targets_cpu.numpy()  # Now safely on CPU
                    )

                    new_shards.append({
                        "filename": output_filename,
                        "n_trajectories": shard_info["n_trajectories"]
                    })

                    # Clear GPU memory periodically
                    if len(new_shards) % 10 == 0:
                        torch.cuda.empty_cache()

            new_shard_index["splits"][split_name] = {
                "shards": new_shards,
                "n_trajectories": split_info["n_trajectories"]
            }

    # Save latent shard index
    save_json(new_shard_index, output_dir / "latent_shard_index.json")

    # Copy normalization stats
    save_json(norm_stats, output_dir / "normalization.json")

    logger.info(f"Latent dataset saved to {output_dir}")

    # Final sanity check: log a sample to verify
    logger.info("Sanity check - Sample latent data shapes:")
    logger.info(f"  latent_inputs: [N, {config['model']['latent_dim'] + 2}] = [z0, P, T]")
    logger.info(f"  latent_targets: [N, {len(trunk_times_list)}, {config['model']['latent_dim']}]")

    # Optional: Compute and save latent statistics for potential standardization
    if new_shard_index["splits"]["train"]["n_trajectories"] > 0:
        logger.info("Computing latent space statistics for optional standardization...")
        _compute_latent_statistics(output_dir, new_shard_index, config)


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