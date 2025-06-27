#!/usr/bin/env python3
"""
Check data quality in flat-structured HDF5 files.
Specifically designed for files with all data at root level.
"""

import h5py
import numpy as np
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any
import json
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

SPECIES_VARS = [
    "C2H2_evolution", "CH4_evolution", "CO2_evolution", "CO_evolution",
    "H2O_evolution", "H2_evolution", "HCN_evolution", "H_evolution",
    "N2_evolution", "NH3_evolution", "OH_evolution", "O_evolution"
]

LOG_EPSILON = 1e-40


def check_flat_h5_quality(
    filepath: str, 
    sample_size: int = 10000,
    check_gradients: bool = True
) -> Dict[str, Any]:
    """Check data quality issues in flat HDF5 structure."""
    
    results = {
        'file_info': {},
        'species_issues': {},
        'global_stats': {},
        'sample_profiles': []
    }
    
    with h5py.File(filepath, 'r') as f:
        # Get basic info
        n_profiles = f[SPECIES_VARS[0]].shape[0] if SPECIES_VARS[0] in f else 0
        n_timesteps = f[SPECIES_VARS[0]].shape[1] if SPECIES_VARS[0] in f else 0
        
        results['file_info'] = {
            'n_profiles': n_profiles,
            'n_timesteps': n_timesteps,
            'file_size_gb': Path(filepath).stat().st_size / 1e9
        }
        
        logger.info(f"Checking {n_profiles:,} profiles with {n_timesteps} timesteps each")
        
        # Sample random profiles for detailed checking
        sample_indices = np.random.choice(n_profiles, min(sample_size, n_profiles), replace=False)
        sample_indices.sort()
        
        # Check each species
        for species in SPECIES_VARS:
            if species not in f:
                logger.warning(f"Missing species: {species}")
                continue
            
            logger.info(f"\nChecking {species}...")
            
            species_issues = {
                'nan_count': 0,
                'inf_count': 0,
                'negative_count': 0,
                'zero_count': 0,
                'tiny_positive_count': 0,
                'extreme_jumps': [],
                'problematic_profiles': []
            }
            
            # Load sample data
            data = f[species][sample_indices, :]
            
            # Check each profile
            for i, profile_idx in enumerate(tqdm(sample_indices, desc=f"Checking {species}")):
                profile_data = data[i, :]
                profile_issues = []
                
                # Check for NaN/Inf
                if np.isnan(profile_data).any():
                    species_issues['nan_count'] += 1
                    profile_issues.append('NaN')
                
                if np.isinf(profile_data).any():
                    species_issues['inf_count'] += 1
                    profile_issues.append('Inf')
                
                # Check for negative values
                neg_mask = profile_data < 0
                if neg_mask.any():
                    species_issues['negative_count'] += 1
                    min_neg = profile_data[neg_mask].min()
                    profile_issues.append(f'Negative (min={min_neg:.2e})')
                
                # Check for zeros (except at t=0)
                zero_mask = profile_data == 0.0
                if zero_mask[1:].any():
                    species_issues['zero_count'] += 1
                    profile_issues.append(f'Zeros at t>0')
                
                # Check for very small positive values that could cause log issues
                positive_mask = profile_data > 0
                if positive_mask.any():
                    min_positive = profile_data[positive_mask].min()
                    if min_positive < LOG_EPSILON:
                        species_issues['tiny_positive_count'] += 1
                        profile_issues.append(f'Tiny positive (min={min_positive:.2e})')
                
                # Check for extreme gradients
                if check_gradients and len(profile_data) > 1:
                    # Calculate relative changes
                    diffs = np.diff(profile_data)
                    abs_profile = np.abs(profile_data[:-1])
                    # Avoid division by zero
                    abs_profile[abs_profile == 0] = 1e-50
                    relative_changes = np.abs(diffs) / abs_profile
                    
                    # Find extreme jumps (>10x change)
                    extreme_mask = relative_changes > 10
                    if extreme_mask.any():
                        max_jump_idx = np.argmax(relative_changes)
                        max_jump = relative_changes[max_jump_idx]
                        species_issues['extreme_jumps'].append({
                            'profile_idx': int(profile_idx),
                            'time_idx': int(max_jump_idx),
                            'relative_change': float(max_jump),
                            'from_value': float(profile_data[max_jump_idx]),
                            'to_value': float(profile_data[max_jump_idx + 1])
                        })
                        profile_issues.append(f'Jump {max_jump:.1f}x at t={max_jump_idx}')
                
                # Record problematic profiles
                if profile_issues:
                    species_issues['problematic_profiles'].append({
                        'index': int(profile_idx),
                        'issues': profile_issues
                    })
            
            # Calculate statistics for this species
            finite_data = data[np.isfinite(data)]
            positive_data = finite_data[finite_data > 0]
            
            species_issues['stats'] = {
                'min': float(finite_data.min()) if len(finite_data) > 0 else None,
                'max': float(finite_data.max()) if len(finite_data) > 0 else None,
                'mean': float(finite_data.mean()) if len(finite_data) > 0 else None,
                'min_positive': float(positive_data.min()) if len(positive_data) > 0 else None,
                'frac_problematic': len(species_issues['problematic_profiles']) / len(sample_indices)
            }
            
            results['species_issues'][species] = species_issues
        
        # Check T and P
        logger.info("\nChecking temperature and pressure...")
        
        for var in ['T', 'P']:
            if var in f:
                data = f[var][sample_indices]
                
                var_stats = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'nan_count': int(np.isnan(data).sum()),
                    'negative_count': int((data < 0).sum()) if var == 'T' else int((data <= 0).sum())
                }
                
                # Check for unreasonable values
                if var == 'T':
                    unreasonable = (data < 0) | (data > 10000)
                    var_stats['unreasonable_count'] = int(unreasonable.sum())
                elif var == 'P':
                    var_stats['zero_count'] = int((data == 0).sum())
                
                results['global_stats'][var] = var_stats
        
        # Sample some specific problematic profiles for inspection
        all_problematic = set()
        for species, issues in results['species_issues'].items():
            for prob in issues.get('problematic_profiles', [])[:5]:  # First 5 from each species
                all_problematic.add(prob['index'])
        
        if all_problematic:
            logger.info(f"\nExtracting {len(all_problematic)} problematic profiles for inspection...")
            for idx in list(all_problematic)[:10]:  # Limit to 10 total
                profile_data = {
                    'index': idx,
                    'T': float(f['T'][idx]),
                    'P': float(f['P'][idx]),
                    'species_t0': {},
                    'species_tend': {}
                }
                
                for species in SPECIES_VARS[:3]:  # Just first 3 species for brevity
                    if species in f:
                        profile_data['species_t0'][species] = float(f[species][idx, 0])
                        profile_data['species_tend'][species] = float(f[species][idx, -1])
                
                results['sample_profiles'].append(profile_data)
    
    return results


def generate_report(results: Dict[str, Any]) -> None:
    """Generate a human-readable report from the results."""
    
    logger.info("\n" + "="*80)
    logger.info("DATA QUALITY REPORT")
    logger.info("="*80)
    
    # File info
    info = results['file_info']
    logger.info(f"\nFile contains {info['n_profiles']:,} profiles with {info['n_timesteps']} timesteps each")
    logger.info(f"File size: {info['file_size_gb']:.2f} GB")
    
    # Summary of issues
    logger.info("\n" + "-"*60)
    logger.info("SPECIES DATA ISSUES SUMMARY")
    logger.info("-"*60)
    
    total_issues = {
        'nan': 0, 'inf': 0, 'negative': 0, 'zero': 0, 'tiny': 0, 'jumps': 0
    }
    
    for species, issues in results['species_issues'].items():
        frac_prob = issues['stats']['frac_problematic']
        if frac_prob > 0:
            logger.info(f"\n{species}: {frac_prob*100:.1f}% profiles have issues")
            
            if issues['nan_count'] > 0:
                logger.info(f"  - NaN values: {issues['nan_count']} profiles")
                total_issues['nan'] += issues['nan_count']
            
            if issues['negative_count'] > 0:
                logger.info(f"  - Negative values: {issues['negative_count']} profiles")
                total_issues['negative'] += issues['negative_count']
            
            if issues['zero_count'] > 0:
                logger.info(f"  - Zero values at t>0: {issues['zero_count']} profiles")
                total_issues['zero'] += issues['zero_count']
            
            if issues['tiny_positive_count'] > 0:
                logger.info(f"  - Values < {LOG_EPSILON}: {issues['tiny_positive_count']} profiles")
                total_issues['tiny'] += issues['tiny_positive_count']
            
            if issues['extreme_jumps']:
                logger.info(f"  - Extreme jumps: {len(issues['extreme_jumps'])} profiles")
                total_issues['jumps'] += len(issues['extreme_jumps'])
                
                # Show worst jump
                worst_jump = max(issues['extreme_jumps'], key=lambda x: x['relative_change'])
                logger.info(f"    Worst: {worst_jump['relative_change']:.1f}x change at profile {worst_jump['profile_idx']}")
    
    # Global variables
    logger.info("\n" + "-"*60)
    logger.info("GLOBAL VARIABLES (T, P)")
    logger.info("-"*60)
    
    for var, stats in results['global_stats'].items():
        logger.info(f"\n{var}: range [{stats['min']:.2e}, {stats['max']:.2e}], mean={stats['mean']:.2e}")
        if stats.get('nan_count', 0) > 0:
            logger.info(f"  - NaN values: {stats['nan_count']}")
        if var == 'T' and stats.get('unreasonable_count', 0) > 0:
            logger.info(f"  - Unreasonable values: {stats['unreasonable_count']}")
        if var == 'P' and stats.get('zero_count', 0) > 0:
            logger.info(f"  - Zero values: {stats['zero_count']}")
    
    # Recommendations
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    if sum(total_issues.values()) > 0:
        logger.info("\n1. Data Cleaning Required:")
        
        if total_issues['nan'] > 0 or total_issues['inf'] > 0:
            logger.info("   - Remove or interpolate NaN/Inf values")
        
        if total_issues['negative'] > 0:
            logger.info("   - Clip negative species concentrations to 0")
        
        if total_issues['zero'] > 0 or total_issues['tiny'] > 0:
            logger.info("   - Replace zeros and tiny values with 1e-50 to avoid log(0)")
        
        if total_issues['jumps'] > 0:
            logger.info("   - Investigate profiles with extreme jumps (possible integration errors)")
        
        logger.info("\n2. Use the restructuring script with --validate flag to fix these issues")
        logger.info("   python restructure_h5.py input.h5 output.h5 --compression gzip")
    else:
        logger.info("\nNo major data quality issues found in the sampled profiles!")
    
    logger.info("\n3. The flat structure needs to be converted to grouped structure")
    logger.info("   This is REQUIRED for the ML pipeline to work")


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_flat_h5_quality.py <h5_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Run quality check
    logger.info(f"Checking data quality in {filepath}...")
    results = check_flat_h5_quality(filepath, sample_size=10000)
    
    # Generate report
    generate_report(results)
    
    # Save detailed results
    output_file = 'flat_h5_quality_report.json'
    
    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()