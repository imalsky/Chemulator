#!/usr/bin/env python3
"""
prepare_data.py – Extract, clean, and consolidate *runXX-result.h5* files
into a single ML-ready HDF5 plus train/validation/test split indices.

FIXED ISSUES:
- Regex pattern now supports float T/P values
- More intelligent zero-value handling
- Validates consistency across all groups
- Better error messages
- Explicitly sorts split indices for h5py compatibility.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Sequence, Optional, Tuple, Any

import h5py
import numpy as np
from tqdm import tqdm

# ─────────────────────────────
# User-editable configuration
# ─────────────────────────────
CONFIG: Dict[str, object] = {
    "input_files": ["./run21-result.h5"],
    "output_h5": "data.h5",
    "val_frac": 0.15,
    "test_frac": 0.15,
    "batch_size": 2048,
    "allow_negative": [],
    "allow_zero": ["O_evolution", "H_evolution", "OH_evolution"],  # Species that can be zero
    "zero_threshold": 1e-50,  # Values below this are considered zero
    "log_file": None,
    "split_seed": 42,
}

# Updated pattern to handle float values and any seed format
_GRP_PAT = re.compile(r"run_T_([\d.]+)_P_([\d.]+)_SEED_(.+)")


def _jsonable(cfg: Dict[str, object]) -> Dict[str, object]:
    j: Dict[str, object] = {}
    for k, v in cfg.items():
        if isinstance(v, (set, tuple, Path)):
            j[k] = list(map(str, v))
        else:
            j[k] = v
    return j


def _init_logging() -> None:
    if CONFIG["log_file"] is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        CONFIG["log_file"] = f"prepare_data_{ts}.log"
    lp = Path(str(CONFIG["log_file"]))
    lp.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(lp, "w"),
        ],
    )
    logging.info("Configuration:\n%s", json.dumps(_jsonable(CONFIG), indent=2))


def _get_input_files() -> List[str]:
    files = [str(Path(p).expanduser().resolve()) for p in CONFIG["input_files"]]
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        logging.error("Missing input files: %s", missing)
        sys.exit(1)
    for f in files:
        logging.info("Using input: %s", f)
    return files


def _validate_schema_consistency(files: List[str]) -> tuple[Set[str], Dict[str, tuple], Dict[str, np.dtype]]:
    """Validate that all files have consistent group structure and keys."""
    all_schemas = []
    
    for file_path in files:
        with h5py.File(file_path, "r") as f:
            file_groups = []
            for gname in f.keys():
                if not _GRP_PAT.match(gname):
                    continue
                grp = f[gname]
                keys = set(grp.keys())
                shapes = {k: grp[k].shape for k in keys}
                dtypes = {k: grp[k].dtype for k in keys}
                file_groups.append((keys, shapes, dtypes))
            
            if file_groups:
                # Check consistency within file
                first_keys, first_shapes, first_dtypes = file_groups[0]
                for keys, shapes, dtypes in file_groups[1:]:
                    if keys != first_keys:
                        logging.error(f"Inconsistent keys in {file_path}")
                        sys.exit(1)
                    for k in keys:
                        if shapes[k] != first_shapes[k]:
                            logging.error(f"Inconsistent shape for {k} in {file_path}")
                            sys.exit(1)
                all_schemas.append((first_keys, first_shapes, first_dtypes))
    
    if not all_schemas:
        logging.error("No valid profile groups found in any file")
        sys.exit(1)
    
    # Check consistency across files
    first_keys, first_shapes, first_dtypes = all_schemas[0]
    for keys, shapes, dtypes in all_schemas[1:]:
        if keys != first_keys:
            logging.error("Inconsistent keys across files")
            sys.exit(1)
    
    return first_keys, first_shapes, first_dtypes


class _RunningStats:
    __slots__ = ("count", "sum", "min", "max", "nans", "neg", "zeros")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.nans = 0
        self.neg = 0
        self.zeros = 0

    def update(self, arr: np.ndarray, allow_neg: bool, zero_threshold: float) -> None:
        flat = arr.ravel()
        self.count += flat.size
        self.sum += np.nan_to_num(flat, copy=False).sum(dtype=np.float64)
        self.nans += np.isnan(flat).sum()
        if not allow_neg:
            self.neg += (flat < 0).sum()
        self.zeros += (np.abs(flat) < zero_threshold).sum()
        finite = flat[np.isfinite(flat)]
        if finite.size:
            self.min = float(min(self.min, finite.min()))
            self.max = float(max(self.max, finite.max()))

    def asdict(self) -> Dict[str, object]:
        if self.count == 0:
            return {"count": 0}
        return {
            "count": int(self.count),
            "mean": float(self.sum / self.count),
            "min": float(self.min),
            "max": float(self.max),
            "nans": int(self.nans),
            "negatives": int(self.neg),
            "zeros": int(self.zeros),
        }


def _analyze_hdf5_space(files: List[str]) -> Dict[str, Any]:
    """Analyze space usage in HDF5 files and estimate output size."""
    total_file_size = 0
    total_groups = 0
    valid_groups = 0
    profile_space = 0
    metadata_space = 0
    data_by_key: Dict[str, float] = {}
    
    for file_path in files:
        file_size = Path(file_path).stat().st_size
        total_file_size += file_size
        
        with h5py.File(file_path, "r") as f:
            # Analyze groups
            for gname in f.keys():
                total_groups += 1
                grp_size = 0
                
                if _GRP_PAT.match(gname):
                    valid_groups += 1
                    grp = f[gname]
                    
                    # Calculate size of each dataset in the group
                    for key in grp.keys():
                        if key in grp:
                            data = grp[key]
                            # Calculate bytes: shape * dtype size
                            bytes_size = np.prod(data.shape) * data.dtype.itemsize
                            grp_size += bytes_size
                            data_by_key[key] = data_by_key.get(key, 0) + bytes_size
                    
                    profile_space += grp_size
                else:
                    # Non-profile data (metadata, etc.)
                    if gname in f:
                        try:
                            data = f[gname]
                            if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                                bytes_size = np.prod(data.shape) * data.dtype.itemsize
                                metadata_space += bytes_size
                        except:
                            pass
    
    # Calculate percentages
    profile_percent = (profile_space / total_file_size * 100) if total_file_size > 0 else 0
    metadata_percent = (metadata_space / total_file_size * 100) if total_file_size > 0 else 0
    overhead_percent = 100 - profile_percent - metadata_percent
    
    # Sort data by key for consistent output
    sorted_keys = sorted(data_by_key.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total_file_size_mb": total_file_size / (1024**2),
        "total_groups": total_groups,
        "valid_profile_groups": valid_groups,
        "profile_space_mb": profile_space / (1024**2),
        "profile_percent": profile_percent,
        "metadata_space_mb": metadata_space / (1024**2),
        "metadata_percent": metadata_percent,
        "overhead_percent": overhead_percent,
        "data_by_key_mb": {k: v / (1024**2) for k, v in sorted_keys},
    }


def _log_space_analysis(space_info: Dict[str, Any]) -> None:
    """Log the space analysis results in a formatted way."""
    logging.info("=" * 60)
    logging.info("HDF5 SPACE USAGE ANALYSIS")
    logging.info("=" * 60)
    logging.info(f"Total input file size: {space_info['total_file_size_mb']:.2f} MB")
    logging.info(f"Total groups found: {space_info['total_groups']:,}")
    logging.info(f"Valid profile groups: {space_info['valid_profile_groups']:,}")
    logging.info("")
    logging.info("Space breakdown:")
    logging.info(f"  Profile data: {space_info['profile_space_mb']:.2f} MB ({space_info['profile_percent']:.1f}%)")
    logging.info(f"  Metadata: {space_info['metadata_space_mb']:.2f} MB ({space_info['metadata_percent']:.1f}%)")
    logging.info(f"  HDF5 overhead: ({space_info['overhead_percent']:.1f}%)")
    logging.info("")
    logging.info("Profile data by variable:")
    for key, size_mb in space_info['data_by_key_mb'].items():
        logging.info(f"  {key}: {size_mb:.2f} MB")
    logging.info("=" * 60)


def run() -> None:
    _init_logging()
    files = _get_input_files()
    
    # Analyze space usage
    logging.info("Analyzing HDF5 file structure...")
    space_info = _analyze_hdf5_space(files)
    _log_space_analysis(space_info)
    
    batch_size = int(CONFIG["batch_size"])
    allow_neg_keys = set(CONFIG["allow_negative"])
    allow_zero_keys = set(CONFIG["allow_zero"])
    zero_threshold = float(CONFIG["zero_threshold"])

    # Validate and discover schema
    first_keys, seq_shape, dtype_map = _validate_schema_consistency(files)
    
    scalar_keys = ["T", "P"]
    all_keys = scalar_keys + list(first_keys)
    logging.info("Profile sequence keys: %s", list(first_keys))

    out_path = Path(str(CONFIG["output_h5"])).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dsets: Dict[str, h5py.Dataset] = {}
    with h5py.File(out_path, "w") as fout:
        # Scalar datasets (no compression)
        for k in scalar_keys:
            dsets[k] = fout.create_dataset(
                k,
                shape=(0,),
                maxshape=(None,),
                dtype="float64",
                chunks=(batch_size,),
            )
        # Sequence datasets (no compression)
        for k in first_keys:
            dsets[k] = fout.create_dataset(
                k,
                shape=(0, *seq_shape[k]),
                maxshape=(None, *seq_shape[k]),
                dtype=dtype_map[k],
                chunks=(batch_size, *seq_shape[k]),
            )

        stats = {k: _RunningStats() for k in all_keys}
        dropped_nan = dropped_neg = dropped_zero = total_kept = 0
        dropped_reasons: Dict[str, int] = {}

        buf_scalar: Dict[str, List] = {k: [] for k in scalar_keys}
        buf_seq: Dict[str, List] = {k: [] for k in first_keys}

        def _flush() -> None:
            nonlocal total_kept
            if not buf_scalar["T"]:
                return
            n = len(buf_scalar["T"])
            start = total_kept
            stop = total_kept + n
            for k in scalar_keys:
                dsets[k].resize((stop,))
                dsets[k][start:stop] = np.asarray(buf_scalar[k])
                buf_scalar[k].clear()
            for k in first_keys:
                dsets[k].resize((stop, *seq_shape[k]))
                dsets[k][start:stop] = np.asarray(buf_seq[k])
                buf_seq[k].clear()
            total_kept = stop

        # Iterate over input files
        for fi, path in enumerate(files, 1):
            with h5py.File(path, "r") as fin:
                gnames = list(fin.keys())
                valid_gnames = [g for g in gnames if _GRP_PAT.match(g)]
                logging.info("[%d/%d] %s – %d groups (%d valid)",
                             fi, len(files), path, len(gnames), len(valid_gnames))
                progress = tqdm(total=len(valid_gnames), desc=Path(path).name,
                                unit="grp")

                for gname in valid_gnames:
                    progress.update()
                    m = _GRP_PAT.match(gname)
                    if not m:
                        continue
                    T_val, P_val = map(float, m.groups()[:2])
                    grp = fin[gname]

                    bad = False
                    drop_reason = ""
                    seq_data: Dict[str, np.ndarray] = {}
                    
                    for k in first_keys:
                        arr = grp[k][...]
                        
                        # Check for NaN
                        if np.isnan(arr).any():
                            dropped_nan += 1
                            bad = True
                            drop_reason = f"NaN in {k}"
                            break
                        
                        # Check for negative values (if not allowed)
                        if k not in allow_neg_keys and (arr < 0).any():
                            dropped_neg += 1
                            bad = True
                            drop_reason = f"Negative values in {k}"
                            break
                        
                        # Check for exact zeros or near-zeros (if not allowed)
                        if k not in allow_zero_keys:
                            if (np.abs(arr) < zero_threshold).any():
                                dropped_zero += 1
                                bad = True
                                drop_reason = f"Zero/near-zero values in {k}"
                                break
                        
                        seq_data[k] = arr
                    
                    if bad:
                        dropped_reasons[drop_reason] = dropped_reasons.get(drop_reason, 0) + 1
                        continue

                    # Buffer values
                    buf_scalar["T"].append(T_val)
                    buf_scalar["P"].append(P_val)
                    stats["T"].update(np.array([T_val]), allow_neg=True, zero_threshold=zero_threshold)
                    stats["P"].update(np.array([P_val]), allow_neg=True, zero_threshold=zero_threshold)
                    for k in first_keys:
                        stats[k].update(seq_data[k], 
                                      allow_neg=(k in allow_neg_keys),
                                      zero_threshold=zero_threshold)
                        buf_seq[k].append(seq_data[k])

                    if len(buf_scalar["T"]) >= batch_size:
                        _flush()

                progress.close()

        _flush()

        logging.info("Total kept profiles: %d", total_kept)
        logging.info(
            "Dropped (NaN): %d | Dropped (negative): %d | Dropped (zero): %d",
            dropped_nan, dropped_neg, dropped_zero,
        )
        logging.info("Drop reasons breakdown:")
        for reason, count in sorted(dropped_reasons.items(), key=lambda x: x[1], reverse=True):
            logging.info("  %s: %d", reason, count)
        
        for k in all_keys:
            logging.info("%s stats: %s", k, stats[k].asdict())

        # Generate splits
        n = total_kept
        if n == 0:
            logging.error("No profiles were kept! Check your data and filtering criteria.")
            sys.exit(1)
            
        n_val = int(round(n * float(CONFIG["val_frac"])))
        n_test = int(round(n * float(CONFIG["test_frac"])))
        n_train = n - n_val - n_test

        indices = list(range(n))
        random.Random(int(CONFIG["split_seed"])).shuffle(indices)

        # IMPORTANT: After creating random splits, the indices for each set MUST be
        # sorted. This is a requirement for efficient "fancy indexing" in the h5py
        # library, which expects a monotonically increasing list of indices.
        train_idx = sorted(indices[:n_train])
        val_idx = sorted(indices[n_train : n_train + n_val])
        test_idx = sorted(indices[n_train + n_val :])
        
        logging.info("Generated and sorted train/validation/test split indices.")

        split = {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        }

    # Write split file
    split_path = out_path.with_name(out_path.stem + "_splits.json")
    with open(split_path, "w") as fp:
        json.dump(split, fp, indent=2)
    logging.info("Split file written: %s", split_path)
    logging.info("Split sizes - Train: %d, Val: %d, Test: %d", 
                 len(train_idx), len(val_idx), len(test_idx))


if __name__ == "__main__":
    run()