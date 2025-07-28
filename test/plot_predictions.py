#!/usr/bin/env python3
"""
Plots model predictions against ground truth for a RANDOMLY SELECTED,
VERIFIED test case from the pre-defined test set.

This script robustly identifies a true test sample by:
1. Loading the test_indices.npy file.
2. Picking a random sample index from this file.
3. Using shard_index.json to trace the sample back to its original HDF5 file
   and profile name (gname) by matching initial conditions.
4. Loading the full ground truth profile and generating a comparison plot.
"""

import json
import re
import sys
from pathlib import Path
import logging
import random
import bisect
import hashlib

# Add the project root to the Python path to allow importing from 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# --- -------------------------------------------------- ---
# ---              USER CONFIGURATION AREA              ---
# --- -------------------------------------------------- ---

# SET THE NAME OF THE TRAINED MODEL FOLDER YOU WANT TO PLOT
MODEL_FOLDER_NAME = "deeponet"  # <-- CHANGE THIS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# --- Global Paths ---
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'data' / 'models' / MODEL_FOLDER_NAME

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')


def load_model_and_config(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict, 'NormalizationHelper']:
    """
    Loads the configuration, model, and normalization helper from a model directory.
    """
    from src.models.model import create_model
    from src.data.normalizer import NormalizationHelper

    # Load Configuration
    logging.info(f"Loading configuration from {model_dir}")
    config_path = model_dir / 'config.json'
    config = json.loads(config_path.read_text())

    # Disable compilation
    config["system"]["use_torch_compile"] = False

    # Load model from checkpoint
    logging.info("--> Loading model from standard checkpoint (best_model.pt)...")
    checkpoint_path = model_dir / 'best_model.pt'
    model = create_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Fix for loading state_dict from compiled model
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    logging.info("Model loaded and set to evaluation mode.")

    # Load Normalization Helper
    prediction_mode = config['prediction']['mode']
    processed_data_dir = ROOT / config['paths']['processed_data_dir']
    norm_stats_path = processed_data_dir / 'normalization.json'
    norm_stats = json.loads(norm_stats_path.read_text())
    norm_helper = NormalizationHelper(
        stats=norm_stats, device=device,
        species_vars=config['data']['species_variables'],
        global_vars=config['data']['global_variables'],
        time_var=config['data']['time_variable'], config=config
    )
    logging.info("Normalization helper created.")
    
    return model, config, norm_helper


def find_source_profile_info(sample_index: int, processed_data_dir: Path, norm_helper, config) -> tuple[Path, str]:
    """
    Traces a global sample index back to its source HDF5 file and group name (gname).
    """
    logging.info(f"Tracing origin of global sample index: {sample_index}...")
    
    shard_index_path = processed_data_dir / 'shard_index.json'
    shard_index = json.loads(shard_index_path.read_text())
    shards_meta = shard_index['shards']
    
    shard_starts = [s['start_idx'] for s in shards_meta]
    shard_idx = bisect.bisect_right(shard_starts, sample_index) - 1
    
    target_shard_meta = shards_meta[shard_idx]
    shard_filename = target_shard_meta['filename']
    logging.info(f"Sample found in shard: {shard_filename}")

    match = re.search(r"shard_(run\d+-result)_", shard_filename)
    
    raw_filename = f"{match.group(1)}.h5"
    raw_filepath = ROOT / 'data' / 'raw' / raw_filename
    logging.info(f"Inferred source HDF5 file: {raw_filepath}")

    shard_path = processed_data_dir / shard_filename
    shard_data = np.load(shard_path)
    
    local_index = sample_index - target_shard_meta['start_idx']
    sample_row = shard_data[local_index]
    
    n_species = shard_index['n_species']
    n_globals = shard_index['n_globals']
    initial_conditions_sample = sample_row[:n_species + n_globals]
    
    all_matching_indices = np.where(np.all(np.isclose(shard_data[:, :n_species + n_globals], initial_conditions_sample), axis=1))[0]
    
    p_init_norm, t_init_norm = initial_conditions_sample[n_species], initial_conditions_sample[n_species+1]

    # Denormalize p and t
    def denormalize_var(norm_val, var_name):
        method = norm_helper.methods[var_name]
        var_stats = norm_helper.per_key_stats[var_name]
        if method == 'standard':
            return norm_val * var_stats['std'] + var_stats['mean']
        elif method == 'log-standard':
            log_val = norm_val * var_stats['log_std'] + var_stats['log_mean']
            return 10 ** log_val
        elif method == 'min-max':
            return norm_val * (var_stats['max'] - var_stats['min']) + var_stats['min']
        elif method == 'log-min-max':
            log_val = norm_val * (var_stats['max'] - var_stats['min']) + var_stats['min']
            return 10 ** log_val

    p_init_raw = denormalize_var(p_init_norm, 'P_init')
    t_init_raw = denormalize_var(t_init_norm, 'T_init')

    with h5py.File(raw_filepath, 'r') as f:
        for gname in f.keys():
            match = re.search(r"_P_([0-9.eE+-]+)_T_([0-9.eE+-]+)", gname)
            if match:
                p_val_gname, t_val_gname = float(match.group(1)), float(match.group(2))
                if np.isclose(t_init_raw, t_val_gname, rtol=1e-3, atol=1e-2) and np.isclose(p_init_raw, p_val_gname, rtol=1e-3, atol=1e-2):
                     logging.info(f"Heuristic match found! gname: {gname}")
                     return raw_filepath, gname

    # Fallback
    with h5py.File(raw_filepath, 'r') as f:
        test_fraction = shard_index.get("test_fraction", 0.15)
        for gname in f.keys():
             split_hash = hashlib.sha256((gname + "_split").encode("utf-8")).hexdigest()
             p = int(split_hash[:8], 16) / 0xFFFFFFFF
             if p < test_fraction:
                 logging.warning(f"Could not find exact gname match. Falling back to first available test group: {gname}")
                 return raw_filepath, gname

    raise ValueError(f"Could not find a matching gname in {raw_filepath.name}")


def extract_full_profile(raw_filepath: Path, gname: str, config: dict) -> np.ndarray:
    """Extracts a complete data profile from a specific HDF5 file and group."""
    with h5py.File(raw_filepath, 'r') as f:
        h5_group = f[gname]
        
        species_vars = config['data']['species_variables']
        global_vars = config['data']['global_variables']
        time_var = config['data']['time_variable']

        globals_dict = {}
        matches = re.findall(r"_([A-Z])_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", gname)
        for label, value in matches:
            key = f"{label}_init"
            if key in global_vars:
                globals_dict[key] = float(value)

        var_order = species_vars + global_vars + [time_var]
        n_t = h5_group[time_var].shape[0]
        profile = np.zeros((n_t, len(var_order)), dtype=np.float32)

        for i, var in enumerate(var_order):
            profile[:, i] = h5_group.get(var, globals_dict.get(var))
            
    return profile


def make_predictions(model: torch.nn.Module, profile: np.ndarray, config: dict, norm_helper: 'NormalizationHelper') -> np.ndarray:
    """Generates model predictions for an entire profile using efficient batching."""
    device = next(model.parameters()).device
    n_species = len(config['data']['species_variables'])
    n_t = profile.shape[0]

    profile_t = torch.from_numpy(profile).to(device)
    norm_prof = norm_helper.normalize_profile(profile_t)
    
    pred_raw = np.zeros((n_t, n_species), dtype=np.float32)
    pred_raw[0] = profile[0, :n_species]

    initial_conditions_norm = norm_prof[0, :-1]
    times_to_predict_norm = norm_prof[1:, -1:]
    num_predictions = times_to_predict_norm.shape[0]

    predictions = torch.zeros(num_predictions, n_species, device=device)
    batch_size = 2
    for start in range(0, num_predictions, batch_size):
        end = min(start + batch_size, num_predictions)
        sub_times = times_to_predict_norm[start:end]
        sub_input = torch.cat([
            initial_conditions_norm.unsqueeze(0).expand(end - start, -1),
            sub_times
        ], dim=1)
        predictions[start:end] = model(sub_input)

    profile_pred_norm = torch.zeros_like(norm_prof)
    profile_pred_norm[0, :] = norm_prof[0, :]
    profile_pred_norm[1:, :n_species] = predictions
    profile_pred_norm[1:, n_species:] = norm_prof[1:, n_species:]
    pred_denorm = norm_helper.denormalize_profile(profile_pred_norm)
    pred_raw = pred_denorm[:, :n_species].detach().cpu().numpy()

    return pred_raw


def plot_profile(true_profile: np.ndarray, pred_species_raw: np.ndarray, config: dict, gname: str, save_path: Path):
    """Generates and saves a log-log plot of species evolution."""
    species_vars = config['data']['species_variables']
    times = true_profile[:, -1]
    true_species_raw = true_profile[:, :len(species_vars)]
    
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)
    colors = plt.cm.get_cmap('tab20', len(species_vars))

    for i, species in enumerate(species_vars):
        ax.plot(times, true_species_raw[:, i], color=colors(i), linestyle='-', linewidth=2.5, label=f'{species} (True)')
        ax.plot(times, pred_species_raw[:, i], color=colors(i), linestyle='--', linewidth=2, label=f'{species} (Pred)')
        ax.plot(times[0], true_species_raw[0, i], 'o', color=colors(i), markersize=8, markeredgecolor='black', zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Abundance', fontsize=14)
    ax.set_title(f'Prediction vs. Truth for Profile: {gname}', fontsize=16)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Plot saved successfully to {save_path}")


def main():
    """Main execution function."""
    device = torch.device('cpu')
    
    model, config, norm_helper = load_model_and_config(MODEL_DIR, device)
    
    prediction_mode = config['prediction']['mode']
    processed_data_dir = ROOT / config['paths']['processed_data_dir']

    test_indices_path = processed_data_dir / 'test_indices.npy'
    test_indices = np.load(test_indices_path)
    random_test_index = random.choice(test_indices)
    
    raw_filepath, gname = find_source_profile_info(random_test_index, processed_data_dir, norm_helper, config)
    
    logging.info(f"Extracting ground truth for '{gname}' from {raw_filepath.name}...")
    true_profile = extract_full_profile(raw_filepath, gname, config)

    logging.info(f"Generating predictions for profile '{gname}'...")
    predicted_species = make_predictions(model, true_profile, config, norm_helper)
    
    plots_dir = MODEL_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f'test_prediction_{gname}.png'
    
    plot_profile(true_profile, predicted_species, config, gname, save_path)


if __name__ == '__main__':
    main()