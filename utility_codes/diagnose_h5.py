#!/usr/bin/env python3
"""
diagnose_h5_hardcoded.py - A simple utility to inspect a hardcoded HDF5 file.

Instructions:
1. Set the H5_FILE_PATH variable below to the path of your HDF5 file.
2. Run the script: python diagnose_h5_hardcoded.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np

# --- HARDCODED SETTINGS ---

# <<< SET THIS TO THE PATH OF YOUR HDF5 FILE >>>
H5_FILE_PATH = "../data/chem_data/Xi_chem_data.h5"

# Variables that are often log-transformed and should be positive.
# Add any other variable names from your config's 'species_variables' list here.
LOG_TRANSFORMED_VARS = {"H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2"}

# --- END OF SETTINGS ---


def analyze_h5_file(h5_path_str: str):
    """Reads an H5 file and prints detailed diagnostic information."""
    h5_path = Path(h5_path_str)
    if not h5_path.is_file():
        print(f"Error: File not found at the hardcoded path '{h5_path}'")
        print("Please update the H5_FILE_PATH variable in the script.")
        sys.exit(1)

    print(f"--- Analyzing HDF5 file: {h5_path.name} ---\n")

    try:
        with h5py.File(h5_path, 'r') as hf:
            keys = sorted(list(hf.keys()))
            if not keys:
                print("File is empty or contains no top-level datasets.")
                return

            print(f"Found {len(keys)} datasets: {keys}\n")
            
            num_profiles = -1
            num_timesteps = -1

            # First, try to establish baseline dimensions from 't_time'
            if 't_time' in hf:
                dset = hf['t_time']
                if dset.ndim == 2:
                    num_profiles, num_timesteps = dset.shape
                    print(f"Baseline Dimensions from 't_time':")
                    print(f"  - Number of Profiles: {num_profiles}")
                    print(f"  - Number of Timesteps: {num_timesteps}\n")
                else:
                    print("Warning: 't_time' is not 2D, cannot establish baseline dimensions.\n")

            # Analyze each dataset
            for key in keys:
                print(f"--- Details for key: '{key}' ---")
                dset = hf[key]
                data = dset[:]  # Load data into memory for analysis

                print(f"  - Shape: {dset.shape}")
                print(f"  - Dtype: {dset.dtype}")
                print(f"  - Dimensions: {dset.ndim}D")

                # Basic statistics
                print("  - Statistics:")
                if np.issubdtype(dset.dtype, np.number):
                    print(f"    - Min:  {np.min(data):.4g}")
                    print(f"    - Max:  {np.max(data):.4g}")
                    print(f"    - Mean: {np.mean(data):.4g}")
                    print(f"    - Std:  {np.std(data):.4g}")

                    # Check for invalid values
                    has_nan = np.isnan(data).any()
                    has_inf = np.isinf(data).any()
                    print(f"    - Contains NaN: {has_nan}")
                    print(f"    - Contains Inf: {has_inf}")

                    # Specific check for variables that need to be positive for log transforms
                    if key in LOG_TRANSFORMED_VARS and np.min(data) <= 0:
                        count_non_positive = np.sum(data <= 0)
                        print(f"    - !! WARNING !! Contains {count_non_positive} non-positive values, which will cause issues with log transforms.")
                else:
                    print("    - (Non-numeric data, skipping stats)")

                print("-" * (25 + len(key)))
                print()

            # Final Consistency Check
            print("--- Overall Consistency Check ---")
            if num_profiles == -1:
                print("Could not establish a baseline number of profiles. Skipping consistency checks.")
                return

            consistent = True
            for key in keys:
                shape = hf[key].shape
                if shape[0] != num_profiles:
                    print(f"!! INCONSISTENCY !! Key '{key}' has {shape[0]} profiles, expected {num_profiles}.")
                    consistent = False
                
                # Check timesteps for 2D arrays (like species)
                if hf[key].ndim == 2 and shape[1] != num_timesteps:
                     print(f"!! INCONSISTENCY !! Key '{key}' has {shape[1]} timesteps, expected {num_timesteps}.")
                     consistent = False
            
            if consistent:
                print("All datasets appear to be consistent in number of profiles (and timesteps where applicable).")

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")
        print("The file might be corrupted or not a valid HDF5 file.")
        sys.exit(1)


if __name__ == "__main__":
    analyze_h5_file(H5_FILE_PATH)