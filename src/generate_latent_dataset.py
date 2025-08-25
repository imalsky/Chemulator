#!/usr/bin/env python3
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

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
    """
    Generate latent dataset after autoencoder pretraining with safety checks.

    Safe skip-if-exists:
      - Checks latent_dim matches current config
      - Ensures trunk_times in output match the *intended* trunk times for this run
        (for mode='fixed' they must match requested fixed times; for mode='all' they
         must match the normalized source time grid)
      - Verifies normalization.json used for source vs. existing latent set are identical
    """
    logger = logging.getLogger(__name__)
    if device.type != "cuda":
        raise RuntimeError("Latent dataset generation requires CUDA device.")

    # ---------- Helpers ----------
    def _json_fingerprint(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        with path.open("rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        # canonical dump → stable hash
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _compute_intended_trunk_times_for_all_mode(
        _input_dir: Path,
        _norm_stats: Dict[str, Any],
        _config: Dict[str, Any],
    ) -> List[float]:
        """Probe the FIRST shard to compute the normalized time grid in 'all' mode."""
        shard_index = load_json(_input_dir / "shard_index.json")
        # Prefer train split; fall back to whatever is available
        for split_name in ["train", "validation", "test"]:
            split = shard_index["splits"].get(split_name)
            if split and split["shards"]:
                first_shard = (_input_dir / split_name / split["shards"][0]["filename"])
                break
        else:
            raise RuntimeError("No shards available to probe time grid for 'all' mode.")

        data_cfg = _config["data"]
        time_var = data_cfg["time_variable"]
        helper = NormalizationHelper(_norm_stats, device, _config)

        with np.load(first_shard, allow_pickle=False) as d:
            t_np = d["t_vec"].astype(np.float32)

        t = torch.from_numpy(t_np)
        if t.ndim == 1:
            t_norm = helper.normalize(t.unsqueeze(-1), [time_var]).squeeze(-1)
        elif t.ndim == 2:
            # If per-sample times, use the first trajectory's time grid as canonical
            t_norm = helper.normalize(t.unsqueeze(-1), [time_var]).squeeze(-1)[0]
        else:
            raise ValueError(f"Unexpected t_vec shape {t.shape} in shard probe.")

        return t_norm.to(dtype=torch.float32, device="cpu").tolist()

    # ---------- Load config bits ----------
    data_cfg = config["data"]
    model_cfg = config["model"]
    lg_cfg = config.get("latent_generation", {})
    lg_mode = (lg_cfg.get("mode") or "all").lower()
    if lg_mode not in ("all", "fixed"):
        raise ValueError(f"latent_generation.mode must be 'all' or 'fixed', got '{lg_mode}'")

    fixed_times_list = None
    if lg_mode == "fixed":
        fixed_times_list = lg_cfg.get("fixed_times") or model_cfg.get("trunk_times")
        if not fixed_times_list:
            raise ValueError("latent_generation.mode='fixed' requires 'fixed_times' or model.trunk_times.")
        if not all(0.0 <= float(t) <= 1.0 for t in fixed_times_list):
            raise ValueError(f"Fixed times must be in [0,1], got {fixed_times_list}")

    # ---------- Safe skip-if-exists ----------
    index_path = output_dir / "latent_shard_index.json"
    if index_path.exists():
        try:
            existing = load_json(index_path)
            ok = True

            # 1) latent_dim must match
            ok &= int(existing.get("latent_dim", -1)) == int(model_cfg["latent_dim"])

            # 2) normalization fingerprints must match (source vs existing latent folder)
            src_norm_fp = _json_fingerprint(input_dir / "normalization.json")
            dst_norm_fp = _json_fingerprint(output_dir / "normalization.json")
            ok &= (src_norm_fp is not None and src_norm_fp == dst_norm_fp)

            # 3) intended trunk_times must match existing
            existing_tt = existing.get("trunk_times", None)
            if existing_tt is None:
                ok = False  # cannot trust
            else:
                if lg_mode == "fixed":
                    want = [float(t) for t in fixed_times_list]
                else:  # 'all'
                    norm_stats = load_json(input_dir / "normalization.json")
                    want = _compute_intended_trunk_times_for_all_mode(input_dir, norm_stats, config)

                if len(existing_tt) != len(want):
                    ok = False
                else:
                    tol = 1e-6
                    for a, b in zip(existing_tt, want):
                        if abs(float(a) - float(b)) > tol:
                            ok = False
                            break

            if ok:
                logger.warning("Latent dataset already exists and matches config. Skipping generation.")
                return
            else:
                logger.warning("Existing latent dataset is incompatible; regenerating.")
        except Exception as e:
            # If anything goes wrong, we regenerate
            logger.warning(f"Could not validate existing latent dataset ({e}); regenerating.")

    # ---------- Actual generation (unchanged in spirit) ----------
    logger.info("Generating latent dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input metadata and normalization
    shard_index = load_json(input_dir / "shard_index.json")
    norm_stats = load_json(input_dir / "normalization.json")

    time_var = data_cfg["time_variable"]
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

    new_shard_index = {
        "latent_mode": True,
        "latent_dim": int(model_cfg["latent_dim"]),
        "trunk_times": None,  # set later
        "splits": {}
    }

    model.eval()
    with torch.no_grad():
        shared_time_vec = None
        time_tolerance = 1e-6  # a bit looser for float32 stability

        for split_name in ["train", "validation", "test"]:
            split = shard_index["splits"].get(split_name)
            if not split:
                continue

            logger.info(f"Processing {split_name} split...")
            split_dir = input_dir / split_name
            out_split_dir = output_dir / split_name
            out_split_dir.mkdir(parents=True, exist_ok=True)

            new_shards = []

            for shard_info in tqdm(split["shards"], desc=split_name):
                shard_path = split_dir / shard_info["filename"]

                # Load raw tensors (CPU)
                with np.load(shard_path, allow_pickle=False) as data:
                    x0_np = data["x0"].astype(np.float32)      # [N, S]
                    y_np  = data["y_mat"].astype(np.float32)   # [N, M, S]
                    g_np  = data["globals"].astype(np.float32) # [N, G]
                    t_np  = data["t_vec"].astype(np.float32)   # [M] or [N, M]

                # CPU → GPU and normalize
                x0 = torch.from_numpy(x0_np)
                y  = torch.from_numpy(y_np)
                g  = torch.from_numpy(g_np)

                x0_norm = norm_helper.normalize(x0, species_vars).to(device)   # [N, S]
                y_norm  = norm_helper.normalize(y,  species_vars).to(device)   # [N, M, S]
                g_norm  = norm_helper.normalize(g,  global_vars)               # [N, G] (CPU ok)

                # Time normalization
                t = torch.from_numpy(t_np)
                if t.ndim == 1:
                    t_norm = norm_helper.normalize(t.unsqueeze(-1), [time_var]).squeeze(-1).to(device)  # [M]
                    per_sample_times = False
                elif t.ndim == 2:
                    t_norm = norm_helper.normalize(t.unsqueeze(-1), [time_var]).squeeze(-1).to(device)  # [N, M]
                    per_sample_times = True
                else:
                    raise ValueError(f"'t_vec' must be [M] or [N,M], got shape {t_np.shape}")

                # Range & monotonicity
                t_min, t_max = float(t_norm.min().item()), float(t_norm.max().item())
                if t_min < -1e-6 or t_max > 1.0 + 1e-6:
                    raise ValueError(
                        f"Normalized times must be in [0,1]. Got range [{t_min:.6f}, {t_max:.6f}]. "
                        f"Check normalization for '{time_var}'."
                    )
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
                        # STRICT: all trajectories must share identical time grid
                        time_diff = (t_norm - t_norm[0].unsqueeze(0)).abs().max().item()
                        if time_diff > time_tolerance:
                            raise RuntimeError(
                                f"mode='all' requires identical time grids across trajectories; "
                                f"max diff {time_diff:.2e} > tol {time_tolerance:.2e}."
                            )
                        times_used = t_norm[0]                     # [M]
                    else:
                        times_used = t_norm                         # [M]
                    y_selected = y_norm                              # [N, M, S]
                    trunk_times_tensor = times_used.to(device)
                else:
                    # Fixed: nearest-neighbor to requested normalized times
                    req = torch.tensor([float(ti) for ti in fixed_times_list],
                                       dtype=torch.float32, device=device)     # [K]
                    if per_sample_times:
                        time_diff = (t_norm - t_norm[0].unsqueeze(0)).abs().max().item()
                        if time_diff > time_tolerance:
                            raise RuntimeError(
                                f"mode='fixed' with per-sample times requires identical grids; "
                                f"max diff {time_diff:.2e}."
                            )
                        diffs = (t_norm[0].unsqueeze(0) - req.unsqueeze(1)).abs()  # [K, M]
                        idx = diffs.argmin(dim=1)                                   # [K]
                        y_selected = y_norm[:, idx, :]                              # [N, K, S]
                        trunk_times_tensor = t_norm[0][idx]                         # [K]
                    else:
                        diffs = (t_norm.unsqueeze(0) - req.unsqueeze(1)).abs()      # [K, M]
                        idx = diffs.argmin(dim=1)
                        y_selected = y_norm[:, idx, :]
                        trunk_times_tensor = t_norm[idx]

                # Encode to latent space (batched)
                N = x0_norm.shape[0]
                latent_dim = int(model_cfg["latent_dim"])
                z0 = model.encode(x0_norm)                                          # [N, L]

                K = int(y_selected.size(1))
                z_targets = model.encode(y_selected.reshape(N * K, y_selected.size(-1)))\
                                   .reshape(N, K, latent_dim)                       # [N, K, L]

                # Move to CPU for saving
                z0_cpu       = z0.to(dtype=torch.float32, device='cpu')
                g_cpu        = g_norm.to(dtype=torch.float32, device='cpu')
                z_targets_cpu= z_targets.to(dtype=torch.float32, device='cpu')
                latent_inputs= torch.cat([z0_cpu, g_cpu], dim=1)                    # [N, L+G]

                out_name = f"latent_{shard_info['filename']}"
                out_path = out_split_dir / out_name
                np.savez_compressed(
                    out_path,
                    latent_inputs=latent_inputs.numpy(),
                    latent_targets=z_targets_cpu.numpy()
                )

                new_shards.append({
                    "filename": out_name,
                    "n_trajectories": shard_info["n_trajectories"]
                })

                # Enforce consistent time vector across ALL shards/splits
                times_list = trunk_times_tensor.to('cpu', dtype=torch.float32).numpy().tolist()
                if shared_time_vec is None:
                    shared_time_vec = times_list
                    logger.info(f"Established shared time grid with {len(times_list)} points")
                else:
                    if len(shared_time_vec) != len(times_list):
                        raise RuntimeError(
                            f"Inconsistent time grid lengths: {len(shared_time_vec)} vs {len(times_list)}."
                        )
                    max_diff = max(abs(a - b) for a, b in zip(shared_time_vec, times_list))
                    if max_diff > time_tolerance:
                        raise RuntimeError(
                            f"Inconsistent trunk_times across shards (max diff: {max_diff:.2e})."
                        )

            new_shard_index["splits"][split_name] = {
                "shards": new_shards,
                "n_trajectories": split["n_trajectories"]
            }

    # Finalize index and copy normalization
    if shared_time_vec is None:
        raise RuntimeError("No shards processed; cannot finalize latent index.")

    new_shard_index["trunk_times"] = shared_time_vec
    save_json(new_shard_index, output_dir / "latent_shard_index.json")
    # keep the exact normalization used to build the latent dataset
    norm_stats = load_json(input_dir / "normalization.json")
    save_json(norm_stats, output_dir / "normalization.json")

    # Optional: compute latent stats if your pipeline expects them
    try:
        _compute_latent_statistics(output_dir, new_shard_index, config)  # defined elsewhere in your codebase
    except NameError:
        logger.info("No _compute_latent_statistics found; skipping stats computation.")

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