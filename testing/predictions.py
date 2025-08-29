#!/usr/bin/env python3
"""
Plot AE-DeepONet predictions vs ground truth using a PT2-exported model.

Usage: python predictions.py
"""

# =========================
#          CONFIG
# =========================
MODEL_STR = "deepo"
MODEL_DIR = f"../models/{MODEL_STR}"
CONFIG_PATH = f"{MODEL_DIR}/config.json"
PROCESSED_DIR = "../data/processed-10-log-standard"

EXPORT_PATHS = [
    f"{MODEL_DIR}/complete_model_exported.pt2",
    f"{MODEL_DIR}/final_model_exported.pt2",  # fallback
]

SAMPLE_INDEX = 6  # which trajectory to plot
OUTPUT_DIR = None  # None -> <model_dir>/plots
SEED = 42

# Query time selection
Q_COUNT = 100  # None or 0 for full dense grid
Q_MODE = "uniform"  # Options: uniform, random, log_uniform, anchors

# Plot styling
CONNECT_LINES = True
MARKER_EVERY = 5

# =========================
#     ENV & IMPORTS
# =========================
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import numpy as np
import torch

torch.set_num_threads(1)
import matplotlib.pyplot as plt
from torch.export import load as torch_load

# Add src to path
sys.path.append(str((Path(__file__).resolve().parent.parent / "src").resolve()))
from utils import load_json, seed_everything
from normalizer import NormalizationHelper

plt.style.use("science.mplstyle")


# =========================
#     CORE FUNCTIONS
# =========================
def find_exported_model():
    """Find and return path to exported model."""
    for path in EXPORT_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"No exported model found. Tried: {EXPORT_PATHS}")


def load_model(export_path):
    """Load PT2 exported model and infer its device."""
    print(f"[INFO] Loading exported model: {export_path}")
    program = torch_load(export_path)
    model = program.module()

    # Infer device from model weights
    try:
        state_dict = model.state_dict()
        if state_dict:
            device = next(iter(state_dict.values())).device
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Model is on CUDA but CUDA is not available")

    return model, device


def load_test_data(data_dir, sample_idx=None):
    """Load test shard and return single trajectory data."""
    test_shards = sorted((Path(data_dir) / "test").glob("shard_*.npz"))
    if not test_shards:
        raise RuntimeError(f"No test shards found in {data_dir}/test")

    # Load first shard
    with np.load(test_shards[0], allow_pickle=False) as data:
        if "t_vec" not in data.files:
            raise RuntimeError("Shard missing 't_vec'")

        x0 = data["x0"].astype(np.float32)
        y_mat = data["y_mat"].astype(np.float32)
        globals_arr = data["globals"].astype(np.float32)
        t_vec = data["t_vec"]

    # Select sample
    N = x0.shape[0]
    if sample_idx is None or not (0 <= sample_idx < N):
        sample_idx = np.random.randint(0, N)

    print(f"[INFO] Using shard={test_shards[0].name}, sample={sample_idx}/{N}")

    # Extract single trajectory
    return {
        "x0": x0[sample_idx:sample_idx + 1],
        "y_true": y_mat[sample_idx],
        "t_phys": t_vec[sample_idx] if t_vec.ndim == 2 else t_vec,
        "globals": globals_arr[sample_idx:sample_idx + 1],
        "sample_idx": sample_idx
    }


def normalize_time(norm_helper, t_phys, t_name, device):
    """Normalize physical time array with validation."""
    assert t_phys.ndim == 1, "Expected 1D time array"

    t_tensor = torch.as_tensor(t_phys, dtype=torch.float32, device=device)
    t_norm = norm_helper.normalize(t_tensor.view(-1, 1), [t_name]).view(-1)

    # Validate
    if not torch.isfinite(t_norm).all():
        raise RuntimeError("Normalized time contains non-finite values")

    unique_vals = torch.unique(t_norm)
    time_range = float(t_norm.max() - t_norm.min())

    print(f"[DEBUG] t_norm: min={t_norm.min():.6f}, max={t_norm.max():.6f}, "
          f"range={time_range:.6f}, unique={unique_vals.numel()}")

    if unique_vals.numel() <= 2 or time_range == 0.0:
        raise RuntimeError("Normalized time is effectively constant")

    return t_norm


def denormalize_time(norm_helper, t_norm, t_name):
    """Safely denormalize time."""
    try:
        return norm_helper.denormalize(t_norm.view(-1, 1), [t_name]).view(-1).cpu().numpy()
    except Exception:
        return t_norm.detach().cpu().numpy()


def load_anchor_times(data_dir, q_count, norm_helper, t_name, device):
    """Try to load anchor times from latent data."""
    # Look for anchors file
    candidates = [
        Path(data_dir).parent / "latent_data" / "latent_shard_index.json",
        Path(data_dir) / "latent" / "latent_shard_index.json",
    ]

    anchor_file = None
    for path in candidates:
        if path.exists():
            anchor_file = path
            break

    if not anchor_file:
        return None

    # Load anchors
    meta = load_json(anchor_file)
    anchors = meta.get("trunk_times")

    if not anchors:
        return None

    anchors = np.array(anchors, dtype=np.float32)

    # Subsample if needed
    if q_count and q_count < len(anchors):
        indices = np.linspace(0, len(anchors) - 1, q_count).round().astype(int)
        anchors = anchors[indices]

    t_norm_sel = torch.tensor(anchors, dtype=torch.float32, device=device)
    t_phys_sel = denormalize_time(norm_helper, t_norm_sel, t_name)

    return t_norm_sel, t_phys_sel, None


def select_query_times(mode, count, t_phys, t_norm_dense, data_dir,
                       norm_helper, t_name, device, seed):
    """
    Select query times based on mode.

    Returns: (t_norm_selected, t_phys_selected, indices_in_dense_grid)
    """
    M = t_norm_dense.numel()

    # Use full dense grid if requested
    if count is None or count <= 0 or count >= M:
        return t_norm_dense, t_phys, np.arange(M, dtype=int)

    # Try anchors mode
    if mode == "anchors":
        result = load_anchor_times(data_dir, count, norm_helper, t_name, device)
        if result:
            return result
        print("[WARN] Anchors not found, falling back to uniform")
        mode = "uniform"

    # Select indices based on mode
    rng = np.random.default_rng(seed)

    if mode == "uniform":
        indices = np.linspace(0, M - 1, count).round().astype(int)

    elif mode == "random":
        indices = np.sort(rng.choice(M, size=count, replace=False))

    elif mode == "log_uniform":
        # Log-spaced in physical time
        t_positive = t_phys[t_phys > 0]
        if len(t_positive) == 0:
            indices = np.linspace(0, M - 1, count).round().astype(int)
        else:
            tmin, tmax = t_positive.min(), t_phys.max()
            log_grid = np.logspace(np.log10(tmin), np.log10(tmax), count)

            # Map to nearest indices
            indices = np.array([np.argmin(np.abs(t_phys - g)) for g in log_grid], dtype=int)
            indices = np.unique(indices)

            # Ensure we have enough points
            if indices.size < count:
                remaining = np.setdiff1d(np.arange(M), indices)
                take = min(count - indices.size, remaining.size)
                indices = np.sort(np.r_[indices, remaining[:take]])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return t_norm_dense[indices], t_phys[indices], indices


def run_prediction(model, x0_norm, g_norm, t_norm_sel, norm_helper, species):
    """Run model prediction and denormalize."""
    with torch.no_grad():
        y_pred_norm = model(x0_norm, g_norm, t_norm_sel)  # [1, K, S]

    y_pred = norm_helper.denormalize(y_pred_norm.squeeze(0), species).cpu().numpy()
    return y_pred


def check_time_sensitivity(model, x0_norm, g_norm, t_norm_sel):
    """Quick check of model's sensitivity to time changes."""
    if t_norm_sel.numel() < 2:
        return 0.0

    with torch.no_grad():
        y_first = model(x0_norm, g_norm, t_norm_sel[:1])
        y_last = model(x0_norm, g_norm, t_norm_sel[-1:])

    delta = (y_first - y_last).abs().mean().item()
    print(f"[CHECK] Mean |Δpred| between first and last times = {delta:.3e}")
    return delta


def compute_errors(y_pred, y_true, indices, t_norm_sel, t_norm_dense, species):
    """Compute relative errors between predictions and ground truth."""
    # Handle alignment
    if indices is not None:
        y_true_aligned = y_true[indices, :]
    else:
        # For anchors: find nearest dense times
        t_dense_np = t_norm_dense.detach().cpu().numpy()
        nearest = np.array([np.argmin(np.abs(t_dense_np - float(t)))
                            for t in t_norm_sel], dtype=int)
        y_true_aligned = y_true[nearest, :]

    # Relative error
    rel_error = np.abs(y_pred - y_true_aligned) / (np.abs(y_true_aligned) + 1e-10)

    # Print summary
    print("\n[ERROR] Mean relative error per species:")
    for sp, err in zip(species, rel_error.mean(axis=0)):
        print(f"  {sp:15s}: {err:.3e}")
    print(f"\n[ERROR] Overall mean={rel_error.mean():.3e}, max={rel_error.max():.3e}")

    return rel_error


def plot_results(t_phys, y_true, t_phys_sel, y_pred, sample_idx,
                 species, q_mode, q_count, connect_lines, marker_every, output_path):
    """Create and save the prediction plot."""
    eps = 1e-30

    # Prepare data for log plots
    t_plot = np.clip(t_phys, eps, None)
    y_true_plot = np.clip(y_true, eps, None)
    y_pred_plot = np.clip(y_pred, eps, None)

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))

    # Plot ground truth (full dense curves)
    for i, sp in enumerate(species):
        ax.loglog(t_plot, y_true_plot[:, i], '-',
                  color=colors[i], lw=2.2, alpha=0.9)

    # Plot predictions
    if connect_lines and len(t_phys_sel) > 1:
        for i, sp in enumerate(species):
            ax.loglog(t_phys_sel, y_pred_plot[:, i], '--',
                      color=colors[i], lw=1.8, alpha=0.85)

    # Add markers
    for i, sp in enumerate(species):
        ax.loglog(t_phys_sel[::marker_every], y_pred_plot[::marker_every, i],
                  'o', color=colors[i],  ms=5, alpha=0.9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2.2, ls='-', label='True (dense)'),
        Line2D([0], [0], color='black', lw=1.8, ls='--', label='Predicted (interp)'),
        Line2D([0], [0], marker='o', color='black', lw=0, ms=5, label='Predicted (queries)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    # Labels and formatting
    ax.set_xlabel("Time (s)")
    ax.set_ylim(1e-9, 1)
    ax.set_ylabel("Species Abundance")
    ax.set_title(f"AE-DeepONet vs Ground Truth (Sample {sample_idx}) — {q_mode} K={q_count}")

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot saved to {output_path}")


# =========================
#           MAIN
# =========================
def main():
    seed_everything(SEED)

    # Load configuration
    cfg = load_json(Path(CONFIG_PATH))
    species = cfg["data"]["species_variables"]
    global_vars = cfg["data"]["global_variables"]
    time_var = cfg["data"]["time_variable"]

    # Setup paths
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else (Path(MODEL_DIR) / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    export_path = find_exported_model()
    model, device = load_model(export_path)

    # Initialize normalizer
    norm = NormalizationHelper(
        load_json(Path(PROCESSED_DIR) / "normalization.json"),
        device,
        cfg
    )

    # Load test data
    data = load_test_data(PROCESSED_DIR, SAMPLE_INDEX)

    # Move to device and normalize inputs
    x0_tensor = torch.from_numpy(data["x0"]).to(device)
    g_tensor = torch.from_numpy(data["globals"]).to(device)

    x0_norm = norm.normalize(x0_tensor, species)
    g_norm = norm.normalize(g_tensor, global_vars)

    # Normalize time
    t_norm_dense = normalize_time(norm, data["t_phys"], time_var, device)

    # Select query times
    t_norm_sel, t_phys_sel, indices = select_query_times(
        Q_MODE, Q_COUNT, data["t_phys"], t_norm_dense,
        PROCESSED_DIR, norm, time_var, device, SEED
    )

    K = t_norm_sel.numel()
    print(f"[INFO] Query mode={Q_MODE}, count={K}")

    # Run prediction
    y_pred = run_prediction(model, x0_norm, g_norm, t_norm_sel, norm, species)

    # Check time sensitivity
    check_time_sensitivity(model, x0_norm, g_norm, t_norm_sel)

    # Plot results
    output_file = output_dir / f"predictions_K{K}_{Q_MODE}_sample_{data['sample_idx']}.png"
    plot_results(
        data["t_phys"], data["y_true"], t_phys_sel, y_pred,
        data["sample_idx"], species, Q_MODE, K,
        CONNECT_LINES, MARKER_EVERY, output_file
    )

    # Compute errors
    compute_errors(y_pred, data["y_true"], indices, t_norm_sel,
                   t_norm_dense, species)


if __name__ == "__main__":
    main()