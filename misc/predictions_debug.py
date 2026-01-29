#!/usr/bin/env python3
"""
predictions_debug.py - Step-by-step diagnostic of model predictions.

Shows exact values at each step to understand what's happening.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# CONFIG - Edit these directly
# =============================================================================

RUN_DIR = ROOT / "models" / "v2"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPORT_NAME = "export_cpu_1step.pt2"

SAMPLE_IDX = 12
START_INDEX = 0
N_STEPS = 5  # Just a few steps for clarity


# =============================================================================
# Helpers
# =============================================================================

def load_manifest(processed_dir: Path) -> dict:
    return json.loads((processed_dir / "normalization.json").read_text())


def load_test_trajectory(processed_dir: Path, idx: int):
    """Returns (y_z, g_z, dt_norm, species)"""
    shards = sorted((processed_dir / "test").glob("shard_*.npz"))
    with np.load(shards[0]) as f:
        y_all = f["y_mat"].astype(np.float32)
        g_all = f["globals"].astype(np.float32)
        dt_all = f["dt_norm_mat"].astype(np.float32)

    manifest = load_manifest(processed_dir)
    species = list(manifest.get("species_variables", []))
    return y_all[idx], g_all[idx], dt_all[idx], species


def z_to_log10(y_z: np.ndarray, species: list, manifest: dict) -> np.ndarray:
    """Convert z-space to log10 physical."""
    stats = manifest["per_key_stats"]
    mu = np.array([stats[s]["log_mean"] for s in species], dtype=np.float64)
    sd = np.array([stats[s]["log_std"] for s in species], dtype=np.float64)
    return y_z * sd + mu


def z_to_physical(y_z: np.ndarray, species: list, manifest: dict) -> np.ndarray:
    """Convert z-space to physical."""
    return 10.0 ** z_to_log10(y_z, species, manifest)


def dt_to_seconds(dt_norm: float, manifest: dict) -> float:
    log_min, log_max = manifest["dt"]["log_min"], manifest["dt"]["log_max"]
    log_dt = dt_norm * (log_max - log_min) + log_min
    return 10.0 ** log_dt


def print_state(label: str, y_z: np.ndarray, species: list, manifest: dict, show_n: int = 5):
    """Print a state vector nicely."""
    y_log10 = z_to_log10(y_z, species, manifest)
    y_phys = 10.0 ** y_log10

    print(f"\n  {label}:")
    print(f"    z-space:  min={y_z.min():.4f}, max={y_z.max():.4f}, mean={y_z.mean():.4f}")
    print(f"    log10:    min={y_log10.min():.2f}, max={y_log10.max():.2f}")
    print(f"    physical: min={y_phys.min():.2e}, max={y_phys.max():.2e}")

    # Show first few species
    print(f"    First {show_n} species (z-space):")
    for i in range(min(show_n, len(species))):
        name = species[i].replace("_evolution", "")
        print(f"      {name:12s}: z={y_z[i]:+8.4f}  log10={y_log10[i]:+8.2f}  phys={y_phys[i]:.3e}")


def print_delta(label: str, delta_z: np.ndarray, species: list, manifest: dict, show_n: int = 5):
    """Print a delta (change) vector."""
    stats = manifest["per_key_stats"]
    sd = np.array([stats[s]["log_std"] for s in species], dtype=np.float64)
    delta_log10 = delta_z * sd  # Change in log10 space

    print(f"\n  {label}:")
    print(f"    Δz:      min={delta_z.min():.6f}, max={delta_z.max():.6f}, mean={delta_z.mean():.6f}")
    print(f"    |Δz|:    mean={np.abs(delta_z).mean():.6f}")
    print(f"    Δlog10:  min={delta_log10.min():.4f}, max={delta_log10.max():.4f}")

    # Show first few species
    print(f"    First {show_n} species:")
    for i in range(min(show_n, len(species))):
        name = species[i].replace("_evolution", "")
        print(f"      {name:12s}: Δz={delta_z[i]:+10.6f}  Δlog10={delta_log10[i]:+8.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("STEP-BY-STEP PREDICTION DIAGNOSTIC")
    print("=" * 80)

    processed_dir = PROCESSED_DIR.resolve()
    manifest = load_manifest(processed_dir)

    # Load data
    print(f"\nLoading sample {SAMPLE_IDX} from {processed_dir}")
    y_z, g_z, dt_norm, species = load_test_trajectory(processed_dir, SAMPLE_IDX)
    T, S = y_z.shape

    print(f"  Trajectory: T={T} timesteps, S={S} species")
    print(f"  Globals: {g_z}")

    # Check dt
    dt0 = float(dt_norm[START_INDEX])
    dt_sec = dt_to_seconds(dt0, manifest)
    print(f"\n  dt_norm = {dt0:.6f}")
    print(f"  dt_sec  = {dt_sec:.3e} seconds")

    # Load model
    export_path = RUN_DIR / EXPORT_NAME
    print(f"\nLoading model: {export_path}")
    model = torch.export.load(export_path).module()
    #model.eval()

    device = "cpu"
    dtype = torch.float32

    # Prepare tensors
    g_t = torch.from_numpy(g_z).to(device=device, dtype=dtype).unsqueeze(0)  # [1, G]
    dt_t = torch.tensor([dt0], device=device, dtype=dtype)  # [1]

    # ==========================================================================
    # STEP 0: Initial state
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 0: INITIAL STATE (t=0)")
    print("=" * 80)

    y0_z = y_z[START_INDEX]  # [S]
    print_state("y0 (input to model)", y0_z, species, manifest)

    # ==========================================================================
    # Ground truth for steps 1, 2, ...
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GROUND TRUTH STATES")
    print("=" * 80)

    for step in range(1, min(N_STEPS + 1, T - START_INDEX)):
        y_true_z = y_z[START_INDEX + step]
        delta_true = y_true_z - y_z[START_INDEX + step - 1]

        print(f"\n--- Step {step} (t = {step * dt_sec:.3e} s) ---")
        print_state(f"y_true[{step}]", y_true_z, species, manifest)
        print_delta(f"True delta (y[{step}] - y[{step - 1}])", delta_true, species, manifest)

    # ==========================================================================
    # Model predictions
    # ==========================================================================
    print("\n" + "=" * 80)
    print("MODEL PREDICTIONS (AUTOREGRESSIVE)")
    print("=" * 80)

    y_current_z = y0_z.copy()
    y_current_t = torch.from_numpy(y_current_z).to(device=device, dtype=dtype).unsqueeze(0)  # [1, S]

    for step in range(1, N_STEPS + 1):
        print(f"\n{'=' * 40}")
        print(f"PREDICTING STEP {step}")
        print(f"{'=' * 40}")

        # Show input
        print(f"\n  Input to model:")
        print(f"    y shape: {tuple(y_current_t.shape)}")
        print(f"    y range: [{y_current_t.min().item():.4f}, {y_current_t.max().item():.4f}]")
        print(f"    dt: {dt_t.item():.6f}")
        print(f"    g: {g_t.squeeze().numpy()}")

        # Forward pass
        with torch.no_grad():
            # The model expects batch_size=2, so we duplicate the input
            y_in = y_current_t.repeat(2, 1)
            dt_in = dt_t.repeat(2)
            g_in = g_t.repeat(2, 1)

            # Run model
            out_batch = model(y_in, dt_in, g_in)

            # Slice back to single sample (keep dims consistent with original script logic)
            out = out_batch[0:1]

        print(f"\n  Raw model output:")
        print(f"    shape: {tuple(out.shape)}")

        # Handle output shape
        if out.ndim == 3:
            y_pred_t = out[:, 0, :]
            print(f"    (squeezed from 3D)")
        else:
            y_pred_t = out

        y_pred_z = y_pred_t[0].cpu().numpy()

        print(f"    range: [{y_pred_z.min():.4f}, {y_pred_z.max():.4f}]")

        # Compare to ground truth
        y_true_z = y_z[START_INDEX + step]
        delta_pred = y_pred_z - y_current_z
        delta_true = y_true_z - y_current_z

        print_state(f"Predicted y[{step}]", y_pred_z, species, manifest)
        print_state(f"True y[{step}]", y_true_z, species, manifest)

        print_delta(f"Predicted delta", delta_pred, species, manifest)
        print_delta(f"True delta", delta_true, species, manifest)

        # Error analysis
        error_z = y_pred_z - y_true_z
        print(f"\n  ERROR (pred - true):")
        print(f"    z-space:  min={error_z.min():.6f}, max={error_z.max():.6f}, MAE={np.abs(error_z).mean():.6f}")

        # Is the model predicting near-zero delta?
        pred_delta_mag = np.abs(delta_pred).mean()
        true_delta_mag = np.abs(delta_true).mean()
        if true_delta_mag > 1e-8:
            ratio = pred_delta_mag / true_delta_mag
            print(f"\n  Delta magnitude ratio: {ratio:.4f}")
            if ratio < 0.1:
                print(f"    ⚠️  Model predicting {ratio * 100:.1f}% of true delta magnitude!")
                print(f"        This looks like the 'predict zero change' problem.")

        # Update for next iteration (autoregressive)
        y_current_z = y_pred_z
        y_current_t = y_pred_t

    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: IDENTITY BASELINE COMPARISON")
    print("=" * 80)

    # What if we just predicted y0 for all steps?
    print("\n  If model predicted y0 unchanged for all steps:")
    for step in range(1, min(N_STEPS + 1, T - START_INDEX)):
        y_true_z = y_z[START_INDEX + step]
        identity_error = np.abs(y0_z - y_true_z).mean()
        print(f"    Step {step}: MAE(z) = {identity_error:.6f}")

    print("\n  Actual model MAE(z) was shown above for each step.")
    print("  If model MAE ≈ identity MAE, model learned nothing useful.")


if __name__ == "__main__":
    main()
