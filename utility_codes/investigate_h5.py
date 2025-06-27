#!/usr/bin/env python3
"""
HDF5 File Investigation Script
Analyzes structure, compression, and data integrity for chemical kinetics datasets
"""

import h5py
import numpy as np
import sys
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'h5_investigation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Expected pattern from your codebase
GROUP_PATTERN = re.compile(r"run_T_([\d.]+)_P_([\d.]+)_SEED_(.+)")

# Expected variables from your config
EXPECTED_SPECIES = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution"
]

EXPECTED_SCALARS = ["T", "P"]
TIME_KEY = "t_time"


def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def analyze_dataset(name: str, dataset: h5py.Dataset) -> Dict[str, Any]:
    """Analyze a single dataset"""
    info = {
        'name': name,
        'shape': dataset.shape,
        'dtype': str(dataset.dtype),
        'size_bytes': dataset.size * dataset.dtype.itemsize,
        'compression': dataset.compression,
        'compression_opts': dataset.compression_opts,
        'chunks': dataset.chunks,
        'fillvalue': dataset.fillvalue
    }
    
    # Check for data statistics
    try:
        # For large datasets, sample instead of reading all
        if dataset.size > 1e6:
            # Sample approach depends on dimensionality
            if dataset.ndim == 1:
                # For 1D, directly sample indices
                sample_indices = np.random.choice(dataset.shape[0], min(10000, dataset.shape[0]), replace=False)
                sample_data = dataset[sample_indices]
            elif dataset.ndim == 2:
                # For 2D (like time series), sample some profiles
                n_profiles = dataset.shape[0]
                sample_profiles = min(100, n_profiles)
                profile_indices = np.random.choice(n_profiles, sample_profiles, replace=False)
                sample_data = dataset[profile_indices, :].flatten()
            else:
                # For higher dimensions, just take a slice
                sample_data = dataset[..., :10].flatten()
        else:
            sample_data = dataset[...]
        
        # Ensure we have a 1D array for statistics
        sample_data = np.asarray(sample_data).flatten()
        finite_data = sample_data[np.isfinite(sample_data)]
        
        if len(finite_data) > 0:
            info['stats'] = {
                'min': float(np.min(finite_data)),
                'max': float(np.max(finite_data)),
                'mean': float(np.mean(finite_data)),
                'std': float(np.std(finite_data)),
                'nan_count': int(np.isnan(sample_data).sum()),
                'inf_count': int(np.isinf(sample_data).sum()),
                'zero_count': int((sample_data == 0).sum()),
                'negative_count': int((sample_data < 0).sum())
            }
        else:
            info['stats'] = {'error': 'No finite values found'}
            
    except Exception as e:
        info['stats'] = {'error': str(e)}
    
    return info


def check_compression_filter_availability():
    """Check which compression filters are available"""
    logger.info("Checking available HDF5 compression filters:")
    
    # For older h5py versions, we'll test by trying to create datasets
    filters = {}
    
    # Test common filters
    test_filters = ['gzip', 'lzf', 'szip']
    
    try:
        # Create a temporary file to test filters
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            with h5py.File(tmp.name, 'w') as test_file:
                test_data = np.arange(100)
                
                for filter_name in test_filters:
                    try:
                        if filter_name == 'gzip':
                            test_file.create_dataset(f'test_{filter_name}', data=test_data, compression='gzip', compression_opts=1)
                        else:
                            test_file.create_dataset(f'test_{filter_name}', data=test_data, compression=filter_name)
                        filters[filter_name] = 'Available'
                    except (ValueError, RuntimeError):
                        filters[filter_name] = 'Not available'
    except Exception as e:
        logger.warning(f"Could not test compression filters: {e}")
        # Assume standard filters are available
        filters = {
            'gzip': 'Probably available',
            'lzf': 'Unknown',
            'szip': 'Unknown'
        }
    
    for name, status in filters.items():
        logger.info(f"  {name}: {status}")
    
    return filters


def investigate_h5_file(filepath: str) -> None:
    """Main investigation function"""
    file_path = Path(filepath)
    
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return
    
    file_size = file_path.stat().st_size
    logger.info(f"\n{'='*80}")
    logger.info(f"HDF5 FILE INVESTIGATION: {filepath}")
    logger.info(f"File size: {format_bytes(file_size)}")
    logger.info(f"{'='*80}\n")
    
    # Check compression filters
    available_filters = check_compression_filter_availability()
    
    try:
        with h5py.File(filepath, 'r') as f:
            logger.info(f"Successfully opened HDF5 file")
            logger.info(f"HDF5 Library version: {h5py.version.hdf5_version}")
            logger.info(f"h5py version: {h5py.__version__}")
            
            # Add file driver info
            if hasattr(f, 'driver'):
                logger.info(f"File driver: {f.driver}")
            
            # 1. Check root-level structure
            logger.info(f"\n{'='*60}")
            logger.info("ROOT LEVEL ANALYSIS")
            logger.info(f"{'='*60}")
            
            root_keys = list(f.keys())
            logger.info(f"Total root items: {len(root_keys)}")
            
            # Categorize root items
            scalar_datasets = []
            profile_groups = []
            other_items = []
            
            for key in root_keys:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    scalar_datasets.append(key)
                elif isinstance(item, h5py.Group) and GROUP_PATTERN.match(key):
                    profile_groups.append(key)
                else:
                    other_items.append(key)
            
            logger.info(f"  Scalar datasets: {len(scalar_datasets)}")
            logger.info(f"  Profile groups: {len(profile_groups)}")
            logger.info(f"  Other items: {len(other_items)}")
            
            # 2. Check scalar datasets
            logger.info(f"\n{'='*60}")
            logger.info("SCALAR DATASETS ANALYSIS")
            logger.info(f"{'='*60}")
            
            for scalar in EXPECTED_SCALARS:
                if scalar in f:
                    ds_info = analyze_dataset(scalar, f[scalar])
                    logger.info(f"\n{scalar}:")
                    logger.info(f"  Shape: {ds_info['shape']}")
                    logger.info(f"  Dtype: {ds_info['dtype']}")
                    logger.info(f"  Size: {format_bytes(ds_info['size_bytes'])}")
                    logger.info(f"  Compression: {ds_info['compression']}")
                    if 'stats' in ds_info and 'error' not in ds_info['stats']:
                        logger.info(f"  Range: [{ds_info['stats']['min']:.6e}, {ds_info['stats']['max']:.6e}]")
                else:
                    logger.warning(f"  Missing expected scalar dataset: {scalar}")
            
            # 3. Analyze profile groups
            logger.info(f"\n{'='*60}")
            logger.info("PROFILE GROUPS ANALYSIS")
            logger.info(f"{'='*60}")
            
            if profile_groups:
                # Sample first few groups for detailed analysis
                sample_size = min(5, len(profile_groups))
                logger.info(f"\nAnalyzing {sample_size} sample profile groups...")
                
                group_structures = []
                
                for i, group_name in enumerate(profile_groups[:sample_size]):
                    match = GROUP_PATTERN.match(group_name)
                    if match:
                        T, P, seed = match.groups()
                        logger.info(f"\nGroup {i+1}: {group_name}")
                        logger.info(f"  T={T}, P={P}, SEED={seed}")
                        
                        group = f[group_name]
                        group_keys = set(group.keys())
                        group_structures.append(group_keys)
                        
                        # Check for expected keys
                        missing_keys = set(EXPECTED_SPECIES) - group_keys
                        extra_keys = group_keys - set(EXPECTED_SPECIES)
                        
                        if missing_keys:
                            logger.warning(f"  Missing keys: {missing_keys}")
                        if extra_keys:
                            logger.info(f"  Extra keys: {extra_keys}")
                        
                        # Check TIME_KEY if it should be in groups
                        if TIME_KEY in group:
                            time_data = analyze_dataset(TIME_KEY, group[TIME_KEY])
                            logger.info(f"  {TIME_KEY}: shape={time_data['shape']}, dtype={time_data['dtype']}")
                        
                        # Sample one species for detailed info
                        if EXPECTED_SPECIES[0] in group:
                            sample_species = group[EXPECTED_SPECIES[0]]
                            info = analyze_dataset(EXPECTED_SPECIES[0], sample_species)
                            logger.info(f"  Sample species ({EXPECTED_SPECIES[0]}):")
                            logger.info(f"    Shape: {info['shape']}")
                            logger.info(f"    Dtype: {info['dtype']}")
                            logger.info(f"    Compression: {info['compression']}")
                            logger.info(f"    Chunks: {info['chunks']}")
                
                # Check consistency across groups
                if len(set(map(tuple, group_structures))) == 1:
                    logger.info(f"\n✓ All sampled groups have consistent structure")
                else:
                    logger.warning(f"\n✗ Inconsistent structure across groups!")
            
            # 4. Check for TIME_KEY at root level
            if TIME_KEY in f:
                logger.info(f"\n{'='*60}")
                logger.info(f"TIME DATA ANALYSIS ({TIME_KEY})")
                logger.info(f"{'='*60}")
                
                time_info = analyze_dataset(TIME_KEY, f[TIME_KEY])
                logger.info(f"  Shape: {time_info['shape']}")
                logger.info(f"  Dtype: {time_info['dtype']}")
                logger.info(f"  Compression: {time_info['compression']}")
                
                # Check number of timesteps
                if len(time_info['shape']) > 1:
                    num_timesteps = time_info['shape'][1]
                    logger.info(f"  Number of timesteps: {num_timesteps}")
                    if num_timesteps < 2:
                        logger.error(f"  ✗ Insufficient timesteps! Need at least 2, found {num_timesteps}")
            else:
                logger.warning(f"  {TIME_KEY} not found at root level")
            
            # 5. Compression and performance analysis
            logger.info(f"\n{'='*60}")
            logger.info("COMPRESSION AND PERFORMANCE ANALYSIS")
            logger.info(f"{'='*60}")
            
            compression_types = {}
            chunk_info = []
            
            def analyze_compression(name, obj):
                if isinstance(obj, h5py.Dataset):
                    comp = obj.compression or 'none'
                    compression_types[comp] = compression_types.get(comp, 0) + 1
                    if obj.chunks:
                        chunk_info.append({
                            'name': name,
                            'shape': obj.shape,
                            'chunks': obj.chunks,
                            'compression': comp
                        })
            
            f.visititems(analyze_compression)
            
            logger.info("\nCompression summary:")
            for comp_type, count in compression_types.items():
                logger.info(f"  {comp_type}: {count} datasets")
            
            if chunk_info:
                logger.info("\nChunking examples:")
                for info in chunk_info[:5]:  # Show first 5
                    logger.info(f"  {info['name']}: shape={info['shape']}, chunks={info['chunks']}")
            
            # 6. Potential issues check
            logger.info(f"\n{'='*60}")
            logger.info("POTENTIAL ISSUES CHECK")
            logger.info(f"{'='*60}")
            
            issues = []
            
            # Check for required compression filters
            used_compressions = set(compression_types.keys()) - {'none'}
            for comp in used_compressions:
                if comp not in available_filters or available_filters[comp] != 'Available':
                    issues.append(f"Compression '{comp}' used but may not be available")
            
            # Check for consistent number of profiles
            if scalar_datasets:
                scalar_lengths = []
                for scalar in scalar_datasets[:5]:  # Check first few
                    if scalar in f:
                        scalar_lengths.append(f[scalar].shape[0])
                
                if len(set(scalar_lengths)) > 1:
                    issues.append(f"Inconsistent scalar dataset lengths: {scalar_lengths}")
            
            # Report issues
            if issues:
                logger.warning("\nFound potential issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("\n✓ No major structural issues detected")
            
            # 7. Summary statistics
            logger.info(f"\n{'='*60}")
            logger.info("SUMMARY")
            logger.info(f"{'='*60}")
            
            total_datasets = sum(1 for _ in f.visit(lambda x: isinstance(f.get(x), h5py.Dataset)))
            total_groups = sum(1 for _ in f.visit(lambda x: isinstance(f.get(x), h5py.Group)))
            
            logger.info(f"Total datasets: {total_datasets}")
            logger.info(f"Total groups: {total_groups}")
            logger.info(f"Valid profile groups: {len(profile_groups)}")
            
            # Save summary to JSON
            summary = {
                'file': str(filepath),
                'file_size_bytes': file_size,
                'total_datasets': total_datasets,
                'total_groups': total_groups,
                'profile_groups': len(profile_groups),
                'compression_types': compression_types,
                'available_filters': available_filters,
                'issues': issues
            }
            
            summary_file = f"h5_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as sf:
                json.dump(summary, sf, indent=2)
            logger.info(f"\nSummary saved to: {summary_file}")
            
    except Exception as e:
        logger.error(f"Error opening HDF5 file: {e}")
        logger.error("This could be due to:")
        logger.error("  - File corruption")
        logger.error("  - Missing compression filters")
        logger.error("  - Version mismatch")
        logger.error("  - File still being written")
        
        # Try to open with different drivers
        logger.info("\nAttempting to open with different drivers...")
        drivers = ['sec2', 'stdio', 'core']
        for driver in drivers:
            try:
                with h5py.File(filepath, 'r', driver=driver) as f:
                    logger.info(f"  ✓ Successfully opened with driver: {driver}")
                    break
            except Exception as e:
                logger.error(f"  ✗ Failed with driver {driver}: {e}")


def check_h5_compatibility(filepath: str) -> None:
    """Additional compatibility checks"""
    logger.info(f"\n{'='*60}")
    logger.info("COMPATIBILITY CHECKS")
    logger.info(f"{'='*60}")
    
    try:
        # Check if file is valid HDF5
        is_hdf5 = h5py.is_hdf5(filepath)
        logger.info(f"Is valid HDF5 file: {is_hdf5}")
        
        if is_hdf5:
            # Get file info
            with h5py.File(filepath, 'r') as f:
                # Check for libver bounds if available
                try:
                    libver_bounds = f.id.get_libver_bounds()
                    logger.info(f"HDF5 library version bounds: {libver_bounds}")
                except AttributeError:
                    logger.info("Library version bounds not available in this h5py version")
                
                # Check for SWMR capability
                if hasattr(f, 'swmr_mode'):
                    logger.info(f"SWMR mode: {f.swmr_mode}")
                else:
                    logger.info("SWMR mode information not available")
                
                # Check file size and recommend settings
                file_size = Path(filepath).stat().st_size
                if file_size > 1e9:  # 1GB
                    logger.info("\nRecommendations for large file:")
                    logger.info("  - Use chunked datasets for better performance")
                    logger.info("  - Consider compression (gzip level 1-4 for speed)")
                    logger.info("  - Enable SWMR mode if using concurrent readers")
                
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python investigate_h5.py <path_to_h5_file>")
        sys.exit(1)
    
    h5_file_path = sys.argv[1]
    
    # Run investigation
    investigate_h5_file(h5_file_path)
    check_h5_compatibility(h5_file_path)
    
    logger.info("\nInvestigation complete! Check the log file for full details.")