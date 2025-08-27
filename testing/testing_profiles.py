#!/usr/bin/env python3
"""
Plot a random trajectory from the training set.
Left: normalized species vs NORMALIZED time (t_norm).
Right: physical species vs physical time.
A secondary x-axis on the left shows the corresponding physical seconds.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
PROCESSED_DIR = "../data/processed-10-log-standard"
OUTPUT_DIR = "../plots"
SEED = None  # Set to None for random selection, or specific int for reproducibility
EPS = 1e-30  # numerical floor for logs

if SEED is not None:
    np.random.seed(SEED)


# -------------------------------
# Normalization helpers
# -------------------------------
def load_normalization_stats(norm_path):
    """Load and parse normalization statistics."""
    with open(norm_path, 'r') as f:
        norm_dict = json.load(f)

    stats = {}
    methods = norm_dict.get("normalization_methods", {})
    per_key = norm_dict.get("per_key_stats", {})

    for var, method in methods.items():
        var_stats = per_key.get(var, {})

        if method == "log-standard":
            stats[var] = {
                "method": method,
                "mean": var_stats.get("log_mean", 0.0),
                "std": max(var_stats.get("log_std", 1.0), 1e-12)
            }
        elif method == "standard":
            stats[var] = {
                "method": method,
                "mean": var_stats.get("mean", 0.0),
                "std": max(var_stats.get("std", 1.0), 1e-12)
            }
        elif method in ["log-min-max", "log10"]:
            # Expect log-space bounds
            lo = var_stats.get("log_min", var_stats.get("min", 0.0))
            hi = var_stats.get("log_max", var_stats.get("max", 1.0))
            stats[var] = {
                "method": method,
                "min": lo,
                "max": hi
            }
        else:
            stats[var] = {"method": "none"}

    return stats, methods


def normalize_species(data, species_names, stats):
    """Normalize species data [M, S] -> [M, S]."""
    normalized = np.zeros_like(data, dtype=np.float64)

    for i, species in enumerate(species_names):
        s = stats.get(species, {"method": "none"})
        method = s["method"]

        if method == "log-standard":
            log_data = np.log10(np.maximum(data[:, i], EPS))
            normalized[:, i] = (log_data - s["mean"]) / s["std"]
        elif method == "standard":
            normalized[:, i] = (data[:, i] - s["mean"]) / s["std"]
        elif method in ["log-min-max", "log10"]:
            log_data = np.log10(np.maximum(data[:, i], EPS))
            denom = (s["max"] - s["min"]) if (s["max"] > s["min"]) else 1.0
            normalized[:, i] = (log_data - s["min"]) / (denom + 1e-12)
        else:
            normalized[:, i] = data[:, i]

    return normalized


def denormalize_species(norm_data, species_names, stats):
    """Denormalize species data [M, S] -> [M, S] in physical units."""
    denormalized = np.zeros_like(norm_data, dtype=np.float64)

    for i, species in enumerate(species_names):
        s = stats.get(species, {"method": "none"})
        method = s["method"]

        if method == "log-standard":
            log_data = norm_data[:, i] * s["std"] + s["mean"]
            denormalized[:, i] = np.power(10.0, log_data)
        elif method == "standard":
            denormalized[:, i] = norm_data[:, i] * s["std"] + s["mean"]
        elif method in ["log-min-max", "log10"]:
            log_data = norm_data[:, i] * (s["max"] - s["min"]) + s["min"]
            denormalized[:, i] = np.power(10.0, log_data)
        else:
            denormalized[:, i] = norm_data[:, i]

    return denormalized


def normalize_time(t_vec, time_var, stats):
    """Normalize time using the method/stats recorded for time_var."""
    t = np.asarray(t_vec, dtype=np.float64)
    s = stats.get(time_var, {"method": "none"})
    method = s["method"]

    if method == "log-standard":
        log_t = np.log10(np.maximum(t, EPS))
        return (log_t - s["mean"]) / s["std"], method, s
    elif method == "standard":
        return (t - s["mean"]) / s["std"], method, s
    elif method in ["log-min-max", "log10"]:
        log_t = np.log10(np.maximum(t, EPS))
        denom = (s["max"] - s["min"]) if (s["max"] > s["min"]) else 1.0
        return (log_t - s["min"]) / (denom + 1e-12), method, s
    else:
        # Fallback: no recorded time normalization. Use raw time (warn in printout).
        return t, "none", {}


def inverse_time_from_norm(z, method, s):
    """Inverse transform from normalized time -> physical seconds."""
    z = np.asarray(z, dtype=np.float64)
    if method == "log-standard":
        log_t = z * s["std"] + s["mean"]
        return np.power(10.0, log_t)
    elif method == "standard":
        return z * s["std"] + s["mean"]
    elif method in ["log-min-max", "log10"]:
        log_t = z * (s["max"] - s["min"]) + s["min"]
        return np.power(10.0, log_t)
    else:
        # identity
        return z


# -------------------------------
# Main
# -------------------------------
def main():
    # Setup paths
    processed_dir = Path(PROCESSED_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Load configuration
    with open(processed_dir.parent.parent / "models" / "deepo-log-standard" / "config.json") as f:
        config = json.load(f)

    species_names = config["data"]["species_variables"]
    time_var = config["data"]["time_variable"]

    # Load normalization statistics
    norm_stats, methods = load_normalization_stats(processed_dir / "normalization.json")

    # Get list of training shards
    train_dir = processed_dir / "train"
    train_shards = sorted(train_dir.glob("shard_*.npz"))
    if not train_shards:
        print("No training shards found!")
        return

    # Select a random shard
    shard = np.random.choice(train_shards)
    print(f"Selected shard: {shard.name}")

    # Load shard data
    with np.load(shard) as data:
        x0 = data["x0"]        # [N, S]
        y_mat = data["y_mat"]  # [N, M, S]
        t_vec = data["t_vec"]  # [M] or [N, M]
        globals_arr = data["globals"]  # [N, G]

    N = x0.shape[0]
    traj_idx = np.random.randint(0, N)
    print(f"Selected trajectory index: {traj_idx}/{N}")

    # Extract arrays for the trajectory
    trajectory = y_mat[traj_idx]     # [M, S]
    initial = x0[traj_idx]           # [S]
    time = t_vec if t_vec.ndim == 1 else t_vec[traj_idx]  # [M]
    P = float(globals_arr[traj_idx, 0])
    T = float(globals_arr[traj_idx, 1])

    print(f"Trajectory conditions: P = {P:.2e} Pa, T = {T:.1f} K")
    print(f"Time points: {len(time)}, from {time[0]:.3e} to {time[-1]:.3e} s")

    # Normalize species and time
    trajectory_norm = normalize_species(trajectory, species_names, norm_stats)
    t_norm, t_method, t_stats = normalize_time(time, time_var, norm_stats)
    if t_method == "none":
        print(f"[WARN] No normalization stats found for time variable '{time_var}'. "
              f"Normalized panel will use raw time on x-axis.")

    # For verification, denormalize species back
    trajectory_denorm = denormalize_species(trajectory_norm, species_names, norm_stats)

    # -------------------------------
    # Plot
    # -------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species_names)))

    # Panel 1: Normalized trajectories vs NORMALIZED time
    ax1.set_xlabel(f"{time_var} (normalized)")
    ax1.set_ylabel("Normalized Abundance")
    ax1.set_title(f"Normalized Species (Trajectory {traj_idx})")
    ax1.grid(True, alpha=0.3)

    # If we truly have no time normalization, use log x-scale only if values are positive and span decades
    if t_method == "none" and np.all(t_norm > 0) and (np.max(t_norm) / max(np.min(t_norm), EPS) > 100):
        ax1.set_xscale("log")  # raw time case
    # Plot
    for i, species in enumerate(species_names):
        ax1.plot(t_norm, trajectory_norm[:, i], color=colors[i], label=species, lw=1.5, alpha=0.8)

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Add a secondary x-axis that shows physical seconds (works for all supported methods)
    try:
        def forward(z):
            # from normalized axis -> physical seconds (for tick labels)
            return inverse_time_from_norm(z, t_method, t_stats)

        def inverse(t_phys):
            # from physical seconds -> normalized axis coords
            z, _, _ = normalize_time(np.asarray(t_phys), time_var, norm_stats)
            return z

        secax = ax1.secondary_xaxis('top', functions=(lambda z: forward(z), lambda t: inverse(t)))
        secax.set_xlabel("Time (s)")
        # If method normalized log-time, put ticks at decades in physical seconds
        if t_method.startswith("log"):
            # choose a reasonable decade range from available times
            lo_dec = int(np.floor(np.log10(max(np.min(time), EPS))))
            hi_dec = int(np.ceil(np.log10(max(np.max(time), EPS))))
            decades = [10.0 ** d for d in range(lo_dec, hi_dec + 1)]
            secax.set_xticks(decades)
            secax.set_xscale("log")
    except Exception as e:
        print(f"[INFO] Secondary x-axis skipped: {e}")

    # Panel 2: Physical units
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Species Abundance")
    ax2.set_title(f"Physical Units (P={P:.1e} Pa, T={T:.0f} K)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    for i, species in enumerate(species_names):
        ax2.plot(np.maximum(time, EPS), np.maximum(trajectory[:, i], EPS),
                 color=colors[i], label=species, lw=1.5, alpha=0.8)

    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.suptitle(f"Trajectory from {Path(shard).name} (Sample {traj_idx})")
    plt.tight_layout()

    # Save
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"trajectory_{Path(shard).stem}_sample{traj_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # -------------------------------
    # Stats / QA
    # -------------------------------
    print("\n" + "=" * 50)
    print(f"Time normalization method for '{time_var}': {t_method}")
    if t_method != "none":
        print(f"  Example: t_phys[0]={time[0]:.3e} s  ->  t_norm[0]={t_norm[0]:.3f}")
    print("\nInitial conditions (physical units):")
    for i, species in enumerate(species_names):
        print(f"  {species:8s}: {initial[i]:.3e}")

    print("\nFinal values (physical units):")
    for i, species in enumerate(species_names):
        print(f"  {species:8s}: {trajectory[-1, i]:.3e}")

    print("\nNormalized value ranges:")
    for i, species in enumerate(species_names):
        vmin, vmax = trajectory_norm[:, i].min(), trajectory_norm[:, i].max()
        print(f"  {species:8s}: [{vmin:7.3f}, {vmax:7.3f}]")

    # Verify denormalization accuracy (species)
    max_error = float(np.abs(trajectory - trajectory_denorm).max())
    mean_error = float(np.abs(trajectory - trajectory_denorm).mean())
    print(f"\nDenormalization verification (species):")
    print(f"  Max absolute error: {max_error:.3e}")
    print(f"  Mean absolute error: {mean_error:.3e}")


if __name__ == "__main__":
    main()
