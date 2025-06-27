#!/usr/bin/env python3
"""
Restructure flat HDF5 file to hierarchical group format expected by the ML pipeline.
Converts from flat arrays to profile groups with proper naming.
"""

import h5py
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Species expected by the pipeline
SPECIES_VARS = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution"
]


def check_flat_structure(h5_file: h5py.File) -> Dict[str, tuple]:
    """Check if the file has the expected flat structure."""
    structure = {}
    
    # Check for species data
    for species in SPECIES_VARS:
        if species in h5_file:
            structure[species] = h5_file[species].shape
        else:
            logger.warning(f"Missing species: {species}")
    
    # Check for T, P
    for var in ['T', 'P']:
        if var in h5_file:
            structure[var] = h5_file[var].shape
    
    # Check for time
    if 't_time' in h5_file:
        structure['t_time'] = h5_file['t_time'].shape
    
    return structure


def restructure_to_groups(
    input_file: str,
    output_file: str,
    compression: str = 'gzip',
    compression_level: int = 4,
    chunk_profiles: int = 1000,
    validate_data: bool = True
) -> None:
    """
    Restructure flat HDF5 to hierarchical group format.
    
    Args:
        input_file: Path to input flat HDF5 file
        output_file: Path to output grouped HDF5 file
        compression: Compression type ('gzip', 'lzf', or None)
        compression_level: Compression level for gzip (1-9)
        chunk_profiles: Number of profiles to process at once
        validate_data: Whether to check for and fix data issues
    """
    
    with h5py.File(input_file, 'r') as f_in:
        # Check structure
        structure = check_flat_structure(f_in)
        logger.info(f"Input file structure: {structure}")
        
        # Get dimensions
        if not SPECIES_VARS[0] in f_in:
            raise ValueError(f"Cannot find {SPECIES_VARS[0]} in input file")
        
        n_profiles = f_in[SPECIES_VARS[0]].shape[0]
        n_timesteps = f_in[SPECIES_VARS[0]].shape[1]
        
        logger.info(f"Found {n_profiles:,} profiles with {n_timesteps} timesteps each")
        
        # Create output file
        with h5py.File(output_file, 'w') as f_out:
            # Process in chunks for memory efficiency
            n_chunks = (n_profiles + chunk_profiles - 1) // chunk_profiles
            
            # Track statistics
            stats = {
                'profiles_processed': 0,
                'profiles_with_issues': 0,
                'nan_fixes': 0,
                'negative_fixes': 0,
                'zero_fixes': 0
            }
            
            # Add time data at root level (shared across all profiles)
            if 't_time' in f_in:
                time_data = f_in['t_time'][0, :]  # Assume all profiles have same time grid
                f_out.create_dataset('t_time', data=time_data, compression=compression)
                logger.info(f"Added time data: {time_data.shape}")
            
            # Process profiles in chunks
            for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
                start_idx = chunk_idx * chunk_profiles
                end_idx = min((chunk_idx + 1) * chunk_profiles, n_profiles)
                chunk_size = end_idx - start_idx
                
                # Load chunk data
                chunk_data = {}
                
                # Load T and P for this chunk
                T_chunk = f_in['T'][start_idx:end_idx]
                P_chunk = f_in['P'][start_idx:end_idx]
                
                # Load species data
                for species in SPECIES_VARS:
                    if species in f_in:
                        chunk_data[species] = f_in[species][start_idx:end_idx, :]
                
                # Process each profile in the chunk
                for i in range(chunk_size):
                    profile_idx = start_idx + i
                    T_val = T_chunk[i]
                    P_val = P_chunk[i]
                    
                    # Create group name matching expected pattern
                    # Using profile index as seed for uniqueness
                    group_name = f"run_T_{T_val:.6f}_P_{P_val:.6f}_SEED_{profile_idx}"
                    
                    # Skip if group already exists (shouldn't happen but be safe)
                    if group_name in f_out:
                        logger.warning(f"Group {group_name} already exists, skipping")
                        continue
                    
                    # Create group
                    grp = f_out.create_group(group_name)
                    
                    # Flag to track if this profile has issues
                    has_issues = False
                    
                    # Add species data to group
                    for species in SPECIES_VARS:
                        if species in chunk_data:
                            data = chunk_data[species][i, :].copy()
                            
                            if validate_data:
                                # Check and fix data quality issues
                                
                                # 1. Fix NaN values
                                nan_mask = np.isnan(data)
                                if nan_mask.any():
                                    stats['nan_fixes'] += nan_mask.sum()
                                    has_issues = True
                                    # Replace NaN with 0 (or interpolate if you prefer)
                                    data[nan_mask] = 0.0
                                
                                # 2. Fix negative values (species should be non-negative)
                                neg_mask = data < 0
                                if neg_mask.any():
                                    stats['negative_fixes'] += neg_mask.sum()
                                    has_issues = True
                                    data[neg_mask] = 0.0
                                
                                # 3. Replace exact zeros with small value (except at t=0)
                                # This prevents log(0) issues
                                zero_mask = (data == 0.0)
                                if zero_mask[1:].any():  # Check all except first timestep
                                    stats['zero_fixes'] += zero_mask[1:].sum()
                                    has_issues = True
                                    data[zero_mask] = 1e-50
                            
                            # Create dataset with compression
                            if compression:
                                if compression == 'gzip':
                                    grp.create_dataset(
                                        species, 
                                        data=data,
                                        compression=compression,
                                        compression_opts=compression_level,
                                        chunks=True  # Auto-chunk
                                    )
                                else:
                                    grp.create_dataset(
                                        species,
                                        data=data,
                                        compression=compression,
                                        chunks=True
                                    )
                            else:
                                grp.create_dataset(species, data=data, chunks=True)
                    
                    # Store time data in group if not at root
                    if 't_time' not in f_out and 't_time' in f_in:
                        time_data = f_in['t_time'][profile_idx, :]
                        grp.create_dataset('t_time', data=time_data)
                    
                    # Update statistics
                    stats['profiles_processed'] += 1
                    if has_issues:
                        stats['profiles_with_issues'] += 1
                
                # Log progress
                if (chunk_idx + 1) % 10 == 0:
                    logger.info(
                        f"Processed {stats['profiles_processed']:,}/{n_profiles:,} profiles. "
                        f"Issues found in {stats['profiles_with_issues']:,} profiles."
                    )
            
            # Also copy T and P as 1D datasets at root level
            logger.info("Adding T and P datasets at root level...")
            f_out.create_dataset('T', data=f_in['T'][...], compression=compression)
            f_out.create_dataset('P', data=f_in['P'][...], compression=compression)
            
            # Log final statistics
            logger.info("\n" + "="*60)
            logger.info("CONVERSION COMPLETE")
            logger.info("="*60)
            logger.info(f"Profiles processed: {stats['profiles_processed']:,}")
            logger.info(f"Profiles with issues: {stats['profiles_with_issues']:,}")
            if validate_data:
                logger.info(f"NaN values fixed: {stats['nan_fixes']:,}")
                logger.info(f"Negative values fixed: {stats['negative_fixes']:,}")
                logger.info(f"Zero values replaced: {stats['zero_fixes']:,}")
            
            # Estimate file size reduction
            if compression:
                logger.info(f"\nCompression: {compression} (level {compression_level if compression == 'gzip' else 'N/A'})")
                logger.info("Note: File size should be significantly reduced")


def verify_output(output_file: str, n_samples: int = 5) -> None:
    """Verify the output file structure."""
    logger.info("\n" + "="*60)
    logger.info("VERIFYING OUTPUT FILE")
    logger.info("="*60)
    
    with h5py.File(output_file, 'r') as f:
        # Count groups
        groups = [k for k in f.keys() if k.startswith('run_T_')]
        logger.info(f"Total profile groups: {len(groups):,}")
        
        # Check root datasets
        root_datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        logger.info(f"Root datasets: {root_datasets}")
        
        # Sample a few groups
        logger.info(f"\nSampling {n_samples} groups:")
        for i, group_name in enumerate(groups[:n_samples]):
            grp = f[group_name]
            datasets = list(grp.keys())
            logger.info(f"  {group_name}: {len(datasets)} datasets")
            
            # Check one species
            if SPECIES_VARS[0] in grp:
                shape = grp[SPECIES_VARS[0]].shape
                compression = grp[SPECIES_VARS[0]].compression
                logger.info(f"    {SPECIES_VARS[0]}: shape={shape}, compression={compression}")


def main():
    parser = argparse.ArgumentParser(
        description="Restructure flat HDF5 file to grouped format for ML pipeline"
    )
    parser.add_argument("input_file", help="Path to input flat HDF5 file")
    parser.add_argument("output_file", help="Path to output grouped HDF5 file")
    parser.add_argument(
        "--compression", 
        choices=['gzip', 'lzf', 'none'],
        default='gzip',
        help="Compression type (default: gzip)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(1, 10),
        help="Compression level for gzip (1-9, default: 4)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of profiles to process at once (default: 1000)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation and fixing"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing file structure"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_output(args.input_file)
    else:
        # Convert compression
        compression = None if args.compression == 'none' else args.compression
        
        # Check if output file exists
        if Path(args.output_file).exists():
            response = input(f"Output file {args.output_file} exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted.")
                return
        
        # Run restructuring
        restructure_to_groups(
            args.input_file,
            args.output_file,
            compression=compression,
            compression_level=args.compression_level,
            chunk_profiles=args.chunk_size,
            validate_data=not args.no_validate
        )
        
        # Verify output
        verify_output(args.output_file)


if __name__ == "__main__":
    main()