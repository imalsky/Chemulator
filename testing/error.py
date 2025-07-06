#!/usr/bin/env python3
"""Analyze prediction errors across test set."""
import json
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

def compute_errors(model_dir, data_file, max_profiles=100):
    """Compute fractional errors for test set."""
    # Load model and metadata
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
        test_indices = json.load(f)["test_set_indices"][:max_profiles]
    
    species = config["data_specification"]["species_variables"]
    global_vars = config["data_specification"]["global_variables"]
    
    # Prepare for batch processing
    all_errors = {sp: [] for sp in species}
    
    with h5py.File(data_file, 'r') as hf:
        for profile_idx in tqdm(test_indices, desc="Processing profiles"):
            # Get data
            t_time = hf["t_time"][profile_idx]
            n_steps = len(t_time)
            
            # Build batch of inputs for all timesteps
            inputs = []
            targets = []
            
            for t_idx in range(1, n_steps):
                # Build input
                inp = []
                for sp in species:
                    inp.append(hf[sp][profile_idx, 0])  # Initial condition
                for gv in global_vars:
                    inp.append(hf[gv][profile_idx])
                inp.append(t_time[t_idx])
                
                # Normalize
                norm_inp = []
                for i, var in enumerate(species + global_vars + ["t_time"]):
                    method = norm_meta["normalization_methods"][var]
                    stats = norm_meta["per_key_stats"][var]
                    val = inp[i]
                    
                    if method == "log-min-max":
                        log_val = np.log10(np.clip(val, 1e-37, None))
                        norm_val = (log_val - stats["min"]) / (stats["max"] - stats["min"])
                    elif method == "standard":
                        norm_val = (val - stats["mean"]) / stats["std"]
                    else:
                        norm_val = val
                    
                    norm_inp.append(norm_val)
                
                inputs.append(norm_inp)
                
                # Get target
                tgt = [hf[sp][profile_idx, t_idx] for sp in species]
                targets.append(tgt)
            
            # Batch predict
            if inputs:
                inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
                with torch.no_grad():
                    pred_norm = model(inputs_tensor).numpy()
                
                # Compute errors for each species
                for sp_idx, sp in enumerate(species):
                    method = norm_meta["normalization_methods"][sp]
                    stats = norm_meta["per_key_stats"][sp]
                    
                    for i in range(len(inputs)):
                        # Denormalize prediction
                        pred_n = pred_norm[i, sp_idx]
                        if method == "log-min-max":
                            pred = 10**(pred_n * (stats["max"] - stats["min"]) + stats["min"])
                        elif method == "standard":
                            pred = pred_n * stats["std"] + stats["mean"]
                        else:
                            pred = pred_n
                        
                        # Compute fractional error
                        true_val = targets[i][sp_idx]
                        if true_val > 1e-10:  # Avoid division by tiny numbers
                            frac_error = abs(pred - true_val) / true_val
                            all_errors[sp].append(frac_error)
    
    return all_errors

def plot_errors(errors):
    """Plot error statistics."""
    species = list(errors.keys())
    mean_errors = [np.mean(errors[sp]) * 100 for sp in species]
    median_errors = [np.median(errors[sp]) * 100 for sp in species]
    p95_errors = [np.percentile(errors[sp], 95) * 100 for sp in species]
    
    # Sort by mean error
    sorted_idx = np.argsort(mean_errors)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(species))
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(x - width, [mean_errors[i] for i in sorted_idx], width, label='Mean', alpha=0.8)
    bars2 = ax.bar(x, [median_errors[i] for i in sorted_idx], width, label='Median', alpha=0.8)
    bars3 = ax.bar(x + width, [p95_errors[i] for i in sorted_idx], width, label='95th Percentile', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Species')
    ax.set_ylabel('Fractional Error (%)')
    ax.set_title('Prediction Error Analysis by Species')
    ax.set_xticks(x)
    ax.set_xticklabels([species[i].replace('_evolution', '') for i in sorted_idx], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label visible bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150)
    plt.show()
    
    # Print summary statistics
    print("\nError Summary (%):")
    print(f"{'Species':<15} {'Mean':>8} {'Median':>8} {'95th %':>8}")
    print("-" * 45)
    for i in sorted_idx:
        print(f"{species[i].replace('_evolution', ''):<15} "
              f"{mean_errors[i]:>8.2f} {median_errors[i]:>8.2f} {p95_errors[i]:>8.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../data/trained_model_siren_v3", help="Model directory")
    parser.add_argument("--data-file", default="../data/chem_data/data.h5", help="HDF5 data file")
    parser.add_argument("--max-profiles", type=int, default=100, help="Max test profiles to analyze")
    args = parser.parse_args()
    
    # Compute errors
    print("Computing prediction errors...")
    errors = compute_errors(args.model_dir, args.data_file, args.max_profiles)
    
    # Plot results
    plot_errors(errors)

if __name__ == "__main__":
    main()