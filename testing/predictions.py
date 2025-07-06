#!/usr/bin/env python3
"""Plot predictions vs ground truth for test profiles."""
import json
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_jit_model(model_dir):
    """Load JIT model and metadata."""
    jit_path = Path(model_dir) / "best_model_jit.pt"
    config_path = Path(model_dir) / "run_config.json"
    norm_path = Path(model_dir) / "normalization_metadata.json"
    test_meta_path = Path(model_dir) / "training_metadata.json"
    
    model = torch.jit.load(jit_path, map_location='cpu')
    model.eval()
    
    with open(config_path) as f:
        config = json.load(f)
    with open(norm_path) as f:
        norm_meta = json.load(f)
    with open(test_meta_path) as f:
        test_indices = json.load(f)["test_set_indices"]
    
    return model, config, norm_meta, test_indices

def normalize_value(x, method, stats):
    """Apply normalization to a value."""
    if method == "log-min-max":
        log_x = np.log10(np.clip(x, 1e-37, None))
        return (log_x - stats["min"]) / (stats["max"] - stats["min"])
    elif method == "standard":
        return (x - stats["mean"]) / stats["std"]
    return x

def denormalize_value(x, method, stats):
    """Apply denormalization to a value."""
    if method == "log-min-max":
        unscaled = x * (stats["max"] - stats["min"]) + stats["min"]
        return 10**unscaled
    elif method == "standard":
        return x * stats["std"] + stats["mean"]
    return x

def predict_profile(model, config, norm_meta, h5_file, prof):
    """Generate predictions for entire time series."""
    species = config["data_specification"]["species_variables"]
    global_vars = config["data_specification"]["global_variables"]
    
    # Read data
    t_time = h5_file["t_time"][prof]
    n_steps = len(t_time)
    
    # Get initial conditions and globals
    initial_species = np.array([h5_file[sp][prof, 0] for sp in species])
    globals_data = np.array([h5_file[gv][prof] for gv in global_vars])
    
    predictions = np.zeros((n_steps, len(species)))
    predictions[0] = initial_species
    
    # Predict each timestep
    for i in range(1, n_steps):
        # Build input
        inp = np.concatenate([initial_species, globals_data, [t_time[i]]])
        
        # Normalize
        norm_inp = inp.copy()
        for j, var in enumerate(species + global_vars + ["t_time"]):
            method = norm_meta["normalization_methods"][var]
            stats = norm_meta["per_key_stats"][var]
            norm_inp[j] = normalize_value(inp[j], method, stats)
        
        # Predict
        with torch.no_grad():
            pred_norm = model(torch.tensor(norm_inp, dtype=torch.float32).unsqueeze(0))
        
        # Denormalize
        pred = pred_norm.squeeze().numpy()
        for j, sp in enumerate(species):
            method = norm_meta["normalization_methods"][sp]
            stats = norm_meta["per_key_stats"][sp]
            predictions[i, j] = denormalize_value(pred[j], method, stats)
    
    return t_time, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../data/trained_model_siren_v3", help="Model directory")
    parser.add_argument("--data-file", default="../data/chem_data/data.h5", help="HDF5 data file")
    parser.add_argument("--prof", type=int, default=0, help="Test profile index")
    args = parser.parse_args()
    
    # Load model
    model, config, norm_meta, test_indices = load_jit_model(args.model_dir)
    species = config["data_specification"]["species_variables"]
    
    # Get predictions
    with h5py.File(args.data_file, 'r') as hf:
        prof = test_indices[args.prof]
        print("Test profile index, num", args.prof, prof)
        t_time, predictions = predict_profile(model, config, norm_meta, hf, prof)
        
        # Get ground truth
        ground_truth = np.array([hf[sp][prof] for sp in species]).T
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for i, sp in enumerate(species):
        if i == 0:
            ax.loglog(t_time, ground_truth[:, i], 'k-', label='True', linewidth=2)
            ax.loglog(t_time, predictions[:, i], 'r--', label='Predicted', linewidth=2)
        else:
            ax.loglog(t_time, ground_truth[:, i], 'k-',linewidth=2)
            ax.loglog(t_time, predictions[:, i], 'r--', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel(sp.replace('_evolution', ''))
        ax.set_title(sp.replace('_evolution', ''))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)

if __name__ == "__main__":
    main()