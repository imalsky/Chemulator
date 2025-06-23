#!/usr/bin/env python3
"""
prepare_data.py - A single-run script to prepare atmospheric simulation data.

This script performs two main sequential tasks:
1.  It searches specified directories for individual simulation profile files (.json),
    validates them, and consolidates them into a single, efficient HDF5 file.
    It identifies any corrupted or inconsistent files and logs them for review,
    excluding them from the final dataset.
2.  It then takes the newly created HDF5 file and generates deterministic,
    shuffled train/validation/test splits, saving the indices to a JSON file.

This prepares all necessary data artifacts for the main training pipeline.

Prerequisites:
- pip install numpy h5py tqdm
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

# ==============================================================================
# ---                           CONFIGURATION                            ---
# ==============================================================================

# 1. DEFINE YOUR LIST OF SOURCE DIRECTORIES.
SOURCE_DIRECTORIES = [
    "data/chem-profiles",
]

# 2. DEFINE THE OUTPUT PATHS.
H5_OUTPUT_PATH = "./chem_data.h5"
SPLITS_OUTPUT_PATH = "./dataset_splits.json"
BAD_FILES_LOG_PATH = "./bad_files.log"

# 3. CONFIGURE THE DATASET SPLIT SIZES AND RANDOM SEED.
VAL_SPLIT_SIZE = 0.15
TEST_SPLIT_SIZE = 0.15
RANDOM_SEED = 42

# 4. CONFIGURE HDF5 WRITING PARAMETERS.
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 4
HDF5_CHUNK_SIZE = 512

# 5. CONFIGURE LOGGING PROGRESS BAR
#    The progress bar will update at this interval (in seconds).
PROGRESS_BAR_UPDATE_INTERVAL = 60.0

# ==============================================================================
# ---                        UTILITY & LOGGING SETUP                       ---
# ==============================================================================

# Setup logging
logger = logging.getLogger(__name__)

def setup_logging():
    """Sets up a clean, formatted logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class TqdmLoggingHandler(logging.Handler):
    """A logging handler that redirects logging to tqdm.write()."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

# Redirect root logger to tqdm handler during long operations
tqdm_handler = TqdmLoggingHandler()
root_logger = logging.getLogger()
# Keep original handlers
original_handlers = root_logger.handlers[:]

def redirect_logging_to_tqdm():
    """Swaps console log handlers with a tqdm-friendly handler."""
    root_logger.handlers = [tqdm_handler]

def restore_logging_handlers():
    """Restores the original logging handlers."""
    root_logger.handlers = original_handlers

def format_bytes(byte_size):
    """Formats bytes into a human-readable string (KB, MB, GB)."""
    if byte_size is None or byte_size < 0:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while byte_size >= power and n < len(power_labels) -1 :
        byte_size /= power
        n += 1
    return f"{byte_size:.2f} {power_labels[n]}"

# ==============================================================================
# ---                      PART 1: JSON to HDF5 Conversion                   ---
# ==============================================================================

def create_hdf5_from_dirs(source_dirs: list[str], output_path: str, bad_files_log: str):
    """
    Finds, validates, and consolidates JSON files into a single HDF5 file.
    """
    # --- Step 1.1: Find all potential JSON files ---
    logger.info("Step 1.1: Finding all JSON files from specified directories...")
    all_json_paths = []

    if not isinstance(source_dirs, list) or not source_dirs:
        raise ValueError("SOURCE_DIRECTORIES must be a non-empty list of paths.")

    for directory in source_dirs:
        logger.info(f"  - Searching in: {directory}")
        if not os.path.isdir(directory):
            logger.warning(f"A specified source directory does not exist: {directory}. Skipping.")
            continue
        found_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
        all_json_paths.extend(found_files)

    unique_paths = sorted(list(set(all_json_paths)))
    total_raw_size = sum(os.path.getsize(p) for p in unique_paths)

    if not unique_paths:
        raise FileNotFoundError("No .json files were found across all specified directories. Please check SOURCE_DIRECTORIES.")

    logger.info(f"Found {len(unique_paths)} unique JSON files to process (Total raw size: {format_bytes(total_raw_size)}).")

    # --- Step 1.2: Discover schema from the first valid file ---
    logger.info("Step 1.2: Inspecting first file to determine data schema...")
    try:
        with open(unique_paths[0], 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    except Exception as e:
        raise IOError(f"Error reading the first JSON file '{unique_paths[0]}'. Cannot establish schema. Details: {e}")

    array_keys = sorted([k for k, v in sample_data.items() if isinstance(v, list)])
    scalar_keys = sorted([k for k, v in sample_data.items() if not isinstance(v, list)])
    required_keys = set(array_keys + scalar_keys)
    array_length = len(sample_data[array_keys[0]]) if array_keys else 0

    logger.info(f"Schema found: {len(array_keys)} array keys (length {array_length}), {len(scalar_keys)} scalar keys.")

    # --- Step 1.3: Validate all files against the schema (Dry Run) ---
    logger.info("Step 1.3: Validating all files for integrity and schema consistency...")
    valid_paths = []
    bad_files_info = []
    
    # Use tqdm for progress, but direct logging through it
    redirect_logging_to_tqdm()
    with tqdm(unique_paths, desc="Validating files", unit="file", mininterval=PROGRESS_BAR_UPDATE_INTERVAL) as pbar:
        for path in pbar:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not required_keys.issubset(data.keys()):
                    missing_keys = required_keys - set(data.keys())
                    raise KeyError(f"Missing required keys: {', '.join(missing_keys)}")
                valid_paths.append(path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                bad_files_info.append(f"File: {path}\nError: {e}\n")
    restore_logging_handlers()

    # --- Step 1.4: Log bad files and report summary ---
    logger.info("Step 1.4: Reporting validation results...")
    if bad_files_info:
        logger.warning(f"Found {len(bad_files_info)} invalid or inconsistent files. They will be excluded.")
        logger.info(f"A detailed report is being saved to: {bad_files_log}")
        try:
            with open(bad_files_log, 'w', encoding='utf-8') as f:
                f.write(f"Data Validation Report - {datetime.now().isoformat()}\n"
                        f"Total files checked: {len(unique_paths)}\n"
                        f"Valid files found: {len(valid_paths)}\n"
                        f"Invalid files found: {len(bad_files_info)}\n{'='*50}\n\n")
                for entry in bad_files_info:
                    f.write(entry + "-"*20 + "\n")
        except IOError as e:
            logger.error(f"Could not write to log file at {bad_files_log}. Error: {e}")
    else:
        logger.info("All files validated successfully!")

    num_samples = len(valid_paths)
    if num_samples == 0:
        raise ValueError("No valid files were found after validation. Halting.")
    
    # --- Step 1.5: Estimate final file size and create HDF5 ---
    logger.info("Step 1.5: Preparing to write HDF5 file...")
    bytes_per_sample = sum(np.dtype('float64').itemsize * array_length for _ in array_keys) + \
                       sum(np.dtype('bool' if isinstance(sample_data[k], bool) else 'float64').itemsize for k in scalar_keys)
    estimated_size_uncompressed = bytes_per_sample * num_samples
    logger.info(f"  - Valid samples to write: {num_samples}")
    logger.info(f"  - Estimated uncompressed size: {format_bytes(estimated_size_uncompressed)}")
    logger.info(f"  - Using '{HDF5_COMPRESSION}' compression. Final size will be smaller.")
    logger.info(f"Creating HDF5 file at '{output_path}'...")
    
    try:
        with h5py.File(output_path, 'w') as hf:
            for key in array_keys:
                hf.create_dataset(key, (num_samples, array_length), dtype='float64', chunks=(1, array_length), compression=HDF5_COMPRESSION, compression_opts=HDF5_COMPRESSION_OPTS)
            for key in scalar_keys:
                dtype = 'bool' if isinstance(sample_data[key], bool) else 'float64'
                hf.create_dataset(key, (num_samples,), dtype=dtype, chunks=(HDF5_CHUNK_SIZE,), compression=HDF5_COMPRESSION, compression_opts=HDF5_COMPRESSION_OPTS)

            redirect_logging_to_tqdm()
            with tqdm(enumerate(valid_paths), total=num_samples, desc="Writing profiles to HDF5", unit="file", mininterval=PROGRESS_BAR_UPDATE_INTERVAL) as pbar:
                for i, path in pbar:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for key in array_keys:
                        hf[key][i, :] = data[key]
                    for key in scalar_keys:
                        hf[key][i] = data[key]
            restore_logging_handlers()

    except Exception as e:
        raise IOError(f"A critical error occurred while writing the HDF5 file. Details: {e}")

    final_size = os.path.getsize(output_path)
    logger.info(f"Part 1 Complete. {num_samples} valid samples written.")
    logger.info(f"Final HDF5 file size: {format_bytes(final_size)} (Compression ratio: ~{(estimated_size_uncompressed/final_size):.2f}x)")


# ==============================================================================
# ---                   PART 2: Dataset Splitting                            ---
# ==============================================================================
def create_and_save_splits(h5_path: str, output_path: str, val_size: float, test_size: float, seed: int):
    """
    Reads an HDF5 file, creates train/val/test splits, and saves indices to JSON.
    """
    logger.info("Step 2.1: Validating split configuration...")
    if not (0.0 < val_size < 1.0 and 0.0 < test_size < 1.0 and (val_size + test_size) < 1.0):
        raise ValueError("VAL_SPLIT_SIZE/TEST_SPLIT_SIZE must be between 0.0 and 1.0, and their sum must be less than 1.0.")
    logger.info("Configuration is valid.")

    logger.info(f"Step 2.2: Reading input HDF5 file at '{h5_path}'...")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Input HDF5 file not found: {h5_path}")

    try:
        with h5py.File(h5_path, 'r') as hf:
            if not list(hf.keys()):
                raise ValueError("The HDF5 file is empty and contains no datasets.")
            first_dataset_name = list(hf.keys())[0]
            num_samples = len(hf[first_dataset_name])
            logger.info(f"File is valid. Found {num_samples} total samples based on dataset '{first_dataset_name}'.")

    except Exception as e:
        raise IOError(f"Could not read the HDF5 file. It may be corrupted. Details: {e}")

    logger.info("Step 2.3: Creating and shuffling indices for splits...")
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    val_count = int(np.floor(val_size * num_samples))
    test_count = int(np.floor(test_size * num_samples))

    if val_count == 0 or test_count == 0 or (num_samples - val_count - test_count) == 0:
        raise ValueError(
            f"Split sizes resulted in an empty set (train/val/test). "
            f"Dataset of size {num_samples} is too small for the specified fractions."
        )

    val_indices = sorted(indices[:val_count].tolist())
    test_indices = sorted(indices[val_count:val_count + test_count].tolist())
    train_indices = sorted(indices[val_count + test_count:].tolist())

    logger.info("--- Split Summary ---")
    logger.info(f"  Training samples:   {len(train_indices):>6}")
    logger.info(f"  Validation samples: {len(val_indices):>6}")
    logger.info(f"  Test samples:       {len(test_indices):>6}")
    logger.info(f"  Total:              {len(train_indices) + len(val_indices) + len(test_indices):>6}")

    logger.info(f"Step 2.4: Saving split indices to '{output_path}'...")
    try:
        splits_for_json = {"train": train_indices, "validation": val_indices, "test": test_indices}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(splits_for_json, f, indent=4)
        logger.info(f"Splits successfully saved to '{output_path}'.")
    except IOError as e:
        raise IOError(f"Could not write to output file '{output_path}'. Check permissions. Details: {e}")

    logger.info("Part 2 Complete. Dataset splits have been generated.")

# ==============================================================================
# ---                          MAIN ORCHESTRATOR                           ---
# ==============================================================================

def main():
    """Main function to run the entire data preparation pipeline."""
    setup_logging()
    
    start_time = time.time()
    
    try:
        logger.info("="*60)
        logger.info("--- STARTING DATA PREPARATION PIPELINE ---")
        logger.info("="*60 + "\n")

        logger.info("--- PART 1: CONVERTING JSON PROFILES TO HDF5 ---")
        create_hdf5_from_dirs(SOURCE_DIRECTORIES, H5_OUTPUT_PATH, BAD_FILES_LOG_PATH)

        logger.info("\n" + "="*60 + "\n")

        logger.info("--- PART 2: CREATING DATASET SPLITS ---")
        create_and_save_splits(
            h5_path=H5_OUTPUT_PATH,
            output_path=SPLITS_OUTPUT_PATH,
            val_size=VAL_SPLIT_SIZE,
            test_size=TEST_SPLIT_SIZE,
            seed=RANDOM_SEED
        )

        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info(f"--- DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}s ---")
        logger.info("="*60)
        return 0

    except (ValueError, FileNotFoundError, IOError) as e:
        logger.error(f"FATAL PIPELINE ERROR: {e}")
        return 1
    except Exception:
        logger.critical("AN UNEXPECTED FATAL ERROR OCCURRED:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())