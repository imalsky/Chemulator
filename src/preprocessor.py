#!/usr/bin/env python3
"""
HDF5 Preprocessing Pipeline with MPI Parallelization
=====================================================
Parallel preprocessing pipeline for converting raw HDF5 trajectory data into
training-ready sharded NPZ files. Uses MPI for efficient distributed scanning
and processing of large trajectory databases.

Key Features:
- MPI-based parallel file scanning and statistics computation
- Profile-level quality filtering (drops entire trajectories below thresholds)
- Normalization statistics with Welford's algorithm (numerically stable)
- Time grid validation and uniformity detection
- Deterministic train/val/test splitting with configurable ratios
- Sharded NPZ output for efficient batch loading during training
- Comprehensive data quality checks (NaN, negative values, monotonicity)

Quality Control:
- Minimum value thresholds: Drops profiles with any species below min_value
- This dropping of profiles with small values is intentional to sanitize the data
- Time grid validation: Ensures monotonic, consistent time grids per file
- NaN filtering: Removes trajectories with missing or invalid data
- Dimension consistency: Validates all species variables are present

Performance:
- MPI scales linearly with number of processes (tested up to 100+ nodes)
- Memory-efficient chunked HDF5 reading for large datasets
- Parallel statistics accumulation with minimal communication overhead
- Progress tracking with estimated time to completion

Output: Sharded NPZ files organized by split (train/val/test) with metadata
including normalization statistics, dt ranges, and data quality reports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Import all utilities from preprocessor_utils
from preprocessor_utils import (
    TIME_DECREASE_TOLERANCE, ALLOW_EQUAL_TIMEPOINTS,
    DEFAULT_HDF5_CHUNK_SIZE, SHARD_FILENAME_FORMAT,
    format_bytes, ensure_directory, load_config_value, get_storage_dtype,
    deterministic_hash, WelfordAccumulator, RunningStatistics,
    get_normalization_flags
)


def scan_hdf5_file_worker(
        path_str: str,
        species_vars: list,
        global_vars: list,
        time_key: str,
        min_value_threshold: float,
        chunk_size: int,
        # Optional scan controls
        scan_fraction_per_file: float | None = None,
        scan_max_groups: int | None = None,
        seed: int = 0,
        return_modal_time: bool = True,
        # Conflict logging controls
        log_conflicting_times: bool = True,
        max_conflict_examples: int = 3,
        logger_name: str = "main.pre",
):
    """
    Returns:
      file_report: dict
      valid_groups: list[str]
      time_candidate: np.ndarray | None (modal grid if mixed)
      progress_stats: dict
    """
    import math
    import hashlib
    from pathlib import Path
    import logging
    import numpy as np
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required but not available") from e

    logger = logging.getLogger(logger_name)

    def _hash_rank(s: str, seed_val: int) -> float:
        h = hashlib.sha256(f"{seed_val}:{s}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / float(1 << 64)

    def _summ(arr: np.ndarray, k: int = 5) -> str:
        n = arr.size
        if n <= 2 * k:
            with np.printoptions(precision=6, suppress=True, threshold=n):
                return np.array2string(arr, separator=", ")
        head, tail = arr[:k], arr[-k:]
        with np.printoptions(precision=6, suppress=True, threshold=2 * k):
            return f"[{', '.join(map(str, head))} … {', '.join(map(str, tail))}] (n={n})"

    def _dt_stats(arr: np.ndarray) -> tuple[float, float]:
        dt = np.diff(arr.astype(np.float64))
        return float(np.min(dt)), float(np.max(dt))

    def _first_mismatch(a: np.ndarray, b: np.ndarray, atol: float = 1e-12, rtol: float = 1e-12) -> int | None:
        if a.shape != b.shape:
            return min(a.size, b.size)
        diff = np.abs(a - b)
        tol = np.maximum(atol, rtol * np.maximum(np.abs(a), np.abs(b)))
        idx = np.nonzero(diff > tol)[0]
        return int(idx[0]) if idx.size else None

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    file_report: dict = {
        "file": str(path),
        "groups_total": 0,
        "groups_valid": 0,
        "groups_missing_time": 0,
        "groups_nonpositive_dt": 0,
        "groups_below_min_value": 0,
        "groups_nan_time": 0,            # NEW: counts any non-finite time vectors
        "min_forward_dt": None,
        # Mixed-grid diagnostics:
        "time_grids_detected": 0,
        "time_grid_modal_count": 0,
        "time_conflicts_logged": 0,
        "examples_nonpositive": [],      # [(group, j, t[j], t[j+1]), ...]
        "examples_nan_time": [],         # NEW: up to 5 group names with NaN time
    }
    progress_stats: dict = {"total_groups": 0, "groups_processed": 0}

    valid_groups: list[str] = []
    time_vectors: list[np.ndarray] = []

    with h5py.File(path, "r") as hdf:
        group_names = sorted(list(hdf.keys()))
        total_groups_in_file = len(group_names)
        file_report["groups_total"] = total_groups_in_file
        progress_stats["total_groups"] = total_groups_in_file

        # Optional fractional / capped scan
        target = total_groups_in_file
        if scan_fraction_per_file is not None and 0.0 < float(scan_fraction_per_file) < 1.0:
            target = max(1, int(math.ceil(float(scan_fraction_per_file) * total_groups_in_file)))
        if scan_max_groups is not None and int(scan_max_groups) > 0:
            target = min(target, int(scan_max_groups))
        if target < total_groups_in_file:
            group_names = sorted(group_names, key=lambda n: _hash_rank(f"{path.name}:{n}:scan", seed))[:target]

        global_min_forward = float("inf")

        for group_idx, group_name in enumerate(group_names, 1):
            progress_stats["groups_processed"] = group_idx
            grp = hdf[group_name]

            # Require time vector
            if time_key not in grp:
                file_report["groups_missing_time"] += 1
                continue

            # Load time vector as float64
            time_data = np.array(grp[time_key][...], dtype=np.float64).reshape(-1)

            # NEW: reject any non-finite times outright
            if not np.all(np.isfinite(time_data)):
                file_report["groups_nan_time"] += 1
                if len(file_report["examples_nan_time"]) < 5:
                    # Keep a short preview, not the whole vector
                    file_report["examples_nan_time"].append(f"{group_name}: t={_summ(time_data)}")
                continue

            if time_data.size < 2:
                file_report["groups_nonpositive_dt"] += 1
                file_report["examples_nonpositive"].append((group_name, -1, float("nan"), float("nan")))
                continue

            diffs = np.diff(time_data)
            local_min_forward = float(diffs.min())
            global_min_forward = min(global_min_forward, local_min_forward)

            # Strictly increasing time required
            if np.any(diffs <= 0.0):
                file_report["groups_nonpositive_dt"] += 1
                bad_js = np.nonzero(diffs <= 0.0)[0][:5]
                for j in bad_js:
                    file_report["examples_nonpositive"].append(
                        (group_name, int(j), float(time_data[j]), float(time_data[j + 1]))
                    )
                continue

            # Optional: below-min-value screening across species
            below_min = False
            if min_value_threshold > 0:
                for key in species_vars:
                    if key not in grp:
                        below_min = True
                        break
                    dset = grp[key]
                    if chunk_size and hasattr(dset, "shape") and dset.shape[0] > chunk_size:
                        hit = False
                        for s in range(0, dset.shape[0], chunk_size):
                            e = min(s + chunk_size, dset.shape[0])
                            if np.nanmin(dset[s:e]) < min_value_threshold:
                                hit = True
                                break
                        if hit:
                            below_min = True
                            break
                    else:
                        if np.nanmin(dset[...]) < min_value_threshold:
                            below_min = True
                            break

            if below_min:
                file_report["groups_below_min_value"] += 1
                continue

            valid_groups.append(group_name)
            time_vectors.append(time_data)

        file_report["min_forward_dt"] = (None if global_min_forward == float("inf") else float(global_min_forward))
        file_report["groups_valid"] = len(valid_groups)

        # ----- Canonical / conflicting time grids handling & logging -----
        time_candidate = None
        if len(time_vectors) > 0:
            # Bucket by exact equality (hash of raw bytes)
            buckets: dict[str, dict] = {}
            for t in time_vectors:
                h = hashlib.sha256(t.tobytes()).hexdigest()
                if h not in buckets:
                    dtmin, dtmax = _dt_stats(t)
                    buckets[h] = {"count": 1, "vec": t, "len": int(t.size), "dt_min": dtmin, "dt_max": dtmax}
                else:
                    buckets[h]["count"] += 1

            file_report["time_grids_detected"] = len(buckets)

            # Modal grid
            modal_key, modal_count = None, 0
            for k, meta in buckets.items():
                if meta["count"] > modal_count:
                    modal_key, modal_count = k, meta["count"]
            file_report["time_grid_modal_count"] = int(modal_count)

            if return_modal_time and modal_key is not None:
                time_candidate = buckets[modal_key]["vec"]

            # Log concise diffs if multiple distinct grids
            if log_conflicting_times and len(buckets) > 1 and modal_key is not None:
                modal_vec = buckets[modal_key]["vec"]
                logger.warning(
                    f"[time-grid mismatch] {path.name}: {len(buckets)} distinct grids among "
                    f"{len(time_vectors)} valid groups; using modal grid (count={modal_count})."
                )
                logger.info(
                    "  modal: len=%d, dt[min=%.6g,max=%.6g], t=%s",
                    buckets[modal_key]["len"],
                    buckets[modal_key]["dt_min"],
                    buckets[modal_key]["dt_max"],
                    _summ(modal_vec),
                )

                printed = 0
                for k, meta in buckets.items():
                    if k == modal_key:
                        continue
                    if printed >= max_conflict_examples:
                        break
                    vec = meta["vec"]
                    mm = _first_mismatch(vec, modal_vec)
                    logger.info(
                        "  nonmodal[%d/%d]: count=%d, len=%d, dt[min=%.6g,max=%.6g], first_mismatch=%s, t=%s",
                        printed + 1,
                        min(len(buckets) - 1, max_conflict_examples),
                        meta["count"],
                        meta["len"],
                        meta["dt_min"],
                        meta["dt_max"],
                        ("None" if mm is None else str(mm)),
                        _summ(vec),
                    )
                    printed += 1
                file_report["time_conflicts_logged"] = printed

        # If any NaN-time groups were dropped, emit a single concise warning
        if file_report["groups_nan_time"] > 0:
            samples = "; ".join(file_report["examples_nan_time"])
            logger.warning(
                f"[time-grid sanitize] Dropped {file_report['groups_nan_time']} group(s) with non-finite time in {path.name}. "
                f"Examples: {samples}"
            )

    return file_report, valid_groups, time_candidate, progress_stats




class DataPreprocessor:
    """
    Main preprocessing pipeline for converting HDF5 to NPZ shards.
    Uses MPI for parallel processing on HPC systems.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.cfg = config
        self.logger = logger or logging.getLogger("preprocessor")

        # Initialize MPI if available
        self.comm = MPI.COMM_WORLD if MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1

        # Extract configuration
        self._load_configuration()
        self._validate_configuration()

        # Initialize state
        self._drop_report = {
            "files": [],
            "overall": {
                "n_total": 0,
                "n_valid": 0,
                "n_dropped": 0,
                "n_nan": 0,
                "n_below_threshold": 0
            }
        }
        self._valid_group_names = {}
        self._canonical_time = None

    def _load_configuration(self) -> None:
        """Load configuration parameters."""
        # Paths - with auto-detection of raw files if not specified
        raw_files_config = load_config_value(
            self.cfg, ["paths", "raw_data_files"], required=False, default=[]
        )

        # Auto-detect raw files if not specified
        if not raw_files_config:
            raw_data_dir = Path("data/raw")
            if not raw_data_dir.exists():
                raise ValueError(f"No raw_data_files specified and {raw_data_dir} doesn't exist")

            # Find all HDF5 files in data/raw
            h5_files = sorted(raw_data_dir.glob("*.h5"))
            hdf5_files = sorted(raw_data_dir.glob("*.hdf5"))
            self.raw_files = [str(f) for f in h5_files + hdf5_files]

            if not self.raw_files:
                raise ValueError(f"No HDF5 files found in {raw_data_dir}")

            if self.rank == 0:
                self.logger.info(f"Auto-detected {len(self.raw_files)} HDF5 files from {raw_data_dir.absolute()}")
                for f in self.raw_files:
                    self.logger.info(f"  - {Path(f).name}")
        else:
            self.raw_files = list(raw_files_config)
            if self.rank == 0:
                self.logger.info(f"Using {len(self.raw_files)} files from config")

        self.processed_dir = Path(load_config_value(
            self.cfg, ["paths", "processed_data_dir"], required=True
        ))

        # Data schema
        data_cfg = self.cfg.setdefault("data", {})

        # 1) Load time key first (needed by auto-detect)
        self.time_key = str(load_config_value(data_cfg, ["time_variable"], required=True))

        # 2) Load globals (also needed by auto-detect to exclude them)
        self.global_vars = list(load_config_value(data_cfg, ["global_variables"], required=True))

        # 3) Species: use provided or auto-detect, then inject back into config
        if not data_cfg.get("species_variables"):
            self.species_vars = self._auto_detect_species_variables()
            # Ensure all ranks share the same list
            if self.comm:
                self.species_vars = self.comm.bcast(self.species_vars, root=0)
            # Inject into config for downstream code (dataset/model)
            data_cfg["species_variables"] = list(self.species_vars)
            if self.rank == 0:
                self.logger.info(f"Auto-detected species variables, e.g, {self.species_vars[0:3]}")
                self.logger.info("Injected species_variables into config.data")
        else:
            self.species_vars = list(data_cfg["species_variables"])

        # Normalization (strict validation)
        norm_cfg = self.cfg.get("normalization", {})
        self.default_method = str(load_config_value(norm_cfg, ["default_method"], required=True))
        self.methods = dict(load_config_value(norm_cfg, ["methods"], required=True))
        self.epsilon = float(load_config_value(norm_cfg, ["epsilon"], required=True))
        self.min_std = float(load_config_value(norm_cfg, ["min_std"], required=True))

        # Preprocessing - with better defaults for fewer shards
        preproc_cfg = self.cfg.get("preprocessing", {})
        self.npz_compressed = bool(load_config_value(
            preproc_cfg, ["npz_compressed"], required=True
        ))

        # CHANGED: Default to 4096 trajectories per shard for much fewer files
        self.traj_per_shard = int(load_config_value(
            preproc_cfg, ["trajectories_per_shard"], default=4096
        ))

        self.hdf5_chunk_size = int(load_config_value(
            preproc_cfg, ["hdf5_chunk_size"], default=DEFAULT_HDF5_CHUNK_SIZE
        ))
        self.min_value_threshold = float(load_config_value(
            preproc_cfg, ["min_value_threshold"], required=True
        ))

        # NEW: Skip first timestep option
        self.skip_first_timestep = bool(load_config_value(
            preproc_cfg, ["skip_first_timestep"], default=False
        ))

        # Worker configuration - use exactly what's in config (no ProcessPool; MPI or serial only)
        requested_workers = int(load_config_value(
            preproc_cfg, ["num_workers"], default=0
        ))

        if self.comm and self.size > 1:
            # In MPI mode, num_workers is ignored; use all MPI ranks
            self.num_workers = self.size
        else:
            # Non-MPI mode: keep the configured number just for logging (no pool)
            self.num_workers = requested_workers if requested_workers > 0 else (os.cpu_count() or 1)

        if self.rank == 0:
            if self.comm:
                self.logger.info(f"Using MPI with {self.size} ranks")
            else:
                self.logger.info("Serial mode (MPI disabled)")
            self.logger.info(f"Trajectories per shard: {self.traj_per_shard}")
            if self.skip_first_timestep:
                self.logger.info("Skipping first timestep in all trajectories")

        # Training configuration
        train_cfg = self.cfg.get("training", {})
        self.val_fraction = float(load_config_value(train_cfg, ["val_fraction"], required=True))
        self.test_fraction = float(load_config_value(train_cfg, ["test_fraction"], required=True))
        self.use_fraction = float(train_cfg.get("use_fraction", 1.0))
        self.min_steps = int(load_config_value(train_cfg, ["min_steps"], required=True))
        max_steps_raw = load_config_value(train_cfg, ["max_steps"], required=False, default=None)
        self.max_steps = int(max_steps_raw) if max_steps_raw else None

        # Storage dtype
        self.storage_dtype = get_storage_dtype(self.cfg)

        # Seed for reproducibility
        self.seed = int(self.cfg.get("system", {}).get("seed", 42))

    def _auto_detect_species_variables(self) -> List[str]:
        """
        Auto-detect species variables from the first HDF5 file.
        Species are identified as datasets (not attributes) that are arrays.
        Excludes the time variable and any global variables.
        """
        if not self.raw_files:
            raise ValueError("No raw data files to detect species from")

        # Only rank 0 does the detection
        if self.comm and self.rank != 0:
            # Other ranks wait for broadcast (handled in _load_configuration)
            return []

        # Rank 0 or serial mode: detect species
        first_file = self.raw_files[0]
        detected_species = []

        try:
            with h5py.File(first_file, "r") as hdf:
                # Get first group
                group_names = list(hdf.keys())
                if not group_names:
                    raise ValueError(f"No groups found in {first_file}")

                first_group = hdf[group_names[0]]

                # Identify datasets that are not time variable or global variables
                for key in first_group.keys():
                    # Skip time variable
                    if key == self.time_key:
                        continue

                    # Skip if it's a global variable (stored as attribute)
                    if key in self.global_vars:
                        continue

                    item = first_group[key]
                    if isinstance(item, h5py.Dataset):
                        # Accept 1D or 2D with last dim == 1
                        if item.ndim == 1 or (item.ndim == 2 and int(item.shape[1]) == 1):
                            detected_species.append(key)

                detected_species.sort()  # Ensure consistent ordering

                if not detected_species:
                    raise ValueError("No species variables detected")

                self.logger.info(
                    f"Auto-detected {len(detected_species)} species variables from {Path(first_file).name}")

        except Exception as e:
            raise RuntimeError(f"Failed to auto-detect species variables: {e}")

        return detected_species

    def _validate_configuration(self) -> None:
        """Validate configuration parameters (rank 0 only)."""
        if self.rank != 0:
            return

        # Raw files: allow auto-detect upstream; but if still empty, fail.
        if len(self.raw_files) == 0:
            raise ValueError("No raw data files specified")

        # Output dir policy: controlled by preprocessing.overwrite_data
        overwrite_data = bool(self.cfg.get("preprocessing", {}).get("overwrite_data", False))
        if self.processed_dir.exists():
            if any(self.processed_dir.iterdir()):
                if not overwrite_data:
                    raise FileExistsError(
                        f"Output directory already exists and is not empty: {self.processed_dir}"
                    )
            else:
                self.logger.warning(f"Output directory exists but is empty: {self.processed_dir}")
        else:
            ensure_directory(self.processed_dir)

        # Fractions / splits
        if not (0.0 <= self.use_fraction <= 1.0):
            raise ValueError("use_fraction must be in [0, 1]")
        if not (0.0 <= self.val_fraction <= 1.0):
            raise ValueError("val_fraction must be in [0, 1]")
        if not (0.0 <= self.test_fraction <= 1.0):
            raise ValueError("test_fraction must be in [0, 1]")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")

        # Offsets
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self.max_steps is not None and self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")

        # Normalization config presence
        norm = self.cfg.get("normalization", {})
        if "default_method" not in norm:
            raise KeyError("normalization.default_method is required")
        if "methods" not in norm:
            raise KeyError("normalization.methods is required")

    def run(self) -> None:
        """Execute the preprocessing pipeline."""
        if h5py is None:
            raise RuntimeError("h5py is required but not installed")

        start_time = time.time()

        if self.rank == 0:
            self.logger.info(f"Starting preprocessing of {len(self.raw_files)} HDF5 files")
            if self.comm:
                self.logger.info(f"Using MPI with {self.size} ranks")
            else:
                self.logger.info("Serial mode (MPI disabled)")

        # Phase 1: Scan and validate files
        scan_start = time.time()
        self._scan_files()
        scan_end = time.time()

        if self._canonical_time is None:
            raise RuntimeError("No valid trajectories found")

        # Apply skip_first_timestep to canonical time grid
        if self.skip_first_timestep:
            original_length = len(self._canonical_time)
            self._canonical_time = self._canonical_time[1:]
            if self.rank == 0:
                self.logger.info(
                    f"Canonical time grid adjusted: {len(self._canonical_time)} timesteps "
                    f"(skipped first timestep, was {original_length})"
                )

        # Log memory estimate
        if self.rank == 0:
            self._log_memory_estimate()

        # Phase 2: Compute dt specification (rank 0 only)
        dt_start = time.time()
        if self.rank == 0:
            dt_spec = self._compute_dt_specification(self._canonical_time)
        else:
            dt_spec = None
        dt_end = time.time()

        # Phase 3: Write shards and collect statistics (parallel)
        shard_start = time.time()
        train_stats = self._write_shards_and_collect_stats_parallel()
        shard_end = time.time()

        # Phase 4: Write normalization manifest (rank 0 only)
        manifest_start = time.time()
        if self.rank == 0:
            manifest = self._finalize_manifest(train_stats, dt_spec)
            manifest_path = self.processed_dir / "normalization.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            self.logger.info(f"Wrote normalization manifest: {manifest_path}")
        manifest_end = time.time()

        # Phase 5: Write reports and snapshot (rank 0 only)
        if self.rank == 0:
            self._write_reports(
                start_time, scan_start, scan_end,
                dt_start, dt_end, shard_start,
                shard_end, manifest_start, manifest_end
            )
            # Persist the (possibly mutated) config so downstream runs are reproducible
            snapshot_path = self.processed_dir / "config.snapshot.json"
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
            self.logger.info(f"Wrote config snapshot: {snapshot_path}")

        # Synchronize all ranks before exit
        if self.comm:
            self.comm.Barrier()

    def _scan_files(self) -> None:
        """Scan HDF5 files to identify valid trajectories using MPI if available."""
        time_candidates = []

        if self.comm and self.size > 1:
            self._scan_files_mpi(time_candidates)
        else:
            self._scan_files_serial(time_candidates)

        # NEW: gather time candidates to root so rank 0 always validates a global list
        if self.comm:
            gathered = self.comm.gather(time_candidates, root=0)
            if self.rank == 0:
                time_candidates = [tc for lst in gathered for tc in lst]

        # Validate only on rank 0, then broadcast
        if not self.comm or self.rank == 0:
            self._validate_canonical_time(time_candidates)
        if self.comm:
            self._canonical_time = self.comm.bcast(getattr(self, "_canonical_time", None), root=0)

    def _scan_files_mpi(self, time_candidates: List[np.ndarray]) -> None:
        """Scan files in parallel using MPI."""
        # Distribute files across ranks
        files_per_rank = len(self.raw_files) // self.size
        remainder = len(self.raw_files) % self.size

        start_idx = self.rank * files_per_rank + min(self.rank, remainder)
        end_idx = start_idx + files_per_rank + (1 if self.rank < remainder else 0)
        my_files = self.raw_files[start_idx:end_idx]

        # Each rank processes its files
        local_results = []
        for file_path in my_files:
            result = scan_hdf5_file_worker(
                file_path,
                self.species_vars,
                self.global_vars,
                self.time_key,
                self.min_value_threshold,
                self.hdf5_chunk_size,
            )
            local_results.append(result)

        # IMPORTANT: each rank contributes its own time candidates locally.
        for _, _, time_candidate, _ in local_results:
            if time_candidate is not None:
                time_candidates.append(time_candidate)

        # Gather results at rank 0 for reporting/metadata
        all_results = self.comm.gather(local_results, root=0)

        if self.rank == 0:
            # Process gathered results (do NOT append time_candidate here, since each rank
            # already added theirs locally and _validate_canonical_time will gather them)
            for rank_results in all_results:
                for file_report, valid_groups, _tc, _progress_stats in rank_results:
                    self._drop_report["files"].append(file_report)

                    # Map new per-file keys -> legacy overall counters
                    n_total = int(file_report.get("groups_total", 0))
                    n_valid = int(file_report.get("groups_valid", 0))
                    n_dropped = max(0, n_total - n_valid)
                    n_below = int(file_report.get("groups_below_min_value", 0))

                    overall = self._drop_report["overall"]
                    overall["n_total"] += n_total
                    overall["n_valid"] += n_valid
                    overall["n_dropped"] += n_dropped
                    overall["n_below_threshold"] += n_below
                    # overall["n_nan"] is accumulated later during NaN screening

                    # Use canonical key from worker report
                    self._valid_group_names[file_report["file"]] = valid_groups

                    # Clean logging against the new keys
                    self.logger.info(
                        f"File {Path(file_report['file']).name}: {n_valid} valid / {n_total} total"
                    )

        # Broadcast results to all ranks
        self._drop_report = self.comm.bcast(self._drop_report, root=0)
        self._valid_group_names = self.comm.bcast(self._valid_group_names, root=0)

    def _scan_files_serial(self, time_candidates: List[np.ndarray]) -> None:
        """Scan files serially."""
        for file_idx, path in enumerate(self.raw_files, 1):
            path_obj = Path(path)
            self.logger.info(f"Scanning file {file_idx}/{len(self.raw_files)}: {path_obj.name}")

            file_report, valid_groups, time_candidate, progress_stats = scan_hdf5_file_worker(
                path,
                self.species_vars,
                self.global_vars,
                self.time_key,
                self.min_value_threshold,
                self.hdf5_chunk_size,
            )

            # Update global report
            self._drop_report["files"].append(file_report)

            n_total = int(file_report.get("groups_total", 0))
            n_valid = int(file_report.get("groups_valid", 0))
            n_dropped = max(0, n_total - n_valid)
            n_below = int(file_report.get("groups_below_min_value", 0))

            overall = self._drop_report["overall"]
            overall["n_total"] += n_total
            overall["n_valid"] += n_valid
            overall["n_dropped"] += n_dropped
            overall["n_below_threshold"] += n_below
            # overall["n_nan"] is accumulated later during NaN screening

            # Keep a consistent key (absolute path string)
            self._valid_group_names[file_report["file"]] = valid_groups

            if time_candidate is not None:
                time_candidates.append(time_candidate)

            self.logger.info(f"  Found {n_valid} valid, {n_dropped} dropped trajectories")

    def _validate_canonical_time(self, time_candidates: List[np.ndarray]) -> None:
        """Validate that all files share the same time grid and set self._canonical_time."""
        import logging
        log = self.logger or logging.getLogger("preprocessor")

        # Rank-0 should pass in a flat list of candidates already (gathered upstream)
        if not time_candidates:
            raise RuntimeError("No time candidates collected from input files")

        # Require equality across all candidates
        canonical = time_candidates[0].astype(np.float64)
        for k, other in enumerate(time_candidates[1:], start=1):
            other64 = other.astype(np.float64)
            if not np.array_equal(canonical, other64):
                # Find first mismatch
                diff_idx = np.where(canonical != other64)[0]
                first = int(diff_idx[0]) if diff_idx.size else -1
                c_val = float(canonical[first]) if first >= 0 else float("nan")
                o_val = float(other64[first]) if first >= 0 else float("nan")
                log.error(f"[t-canon] mismatch between candidate 0 and {k} at index {first} "
                          f"(c={c_val}, o={o_val}); shapes: {canonical.shape} vs {other64.shape}")
                raise ValueError("Time grid differs across files")

        # Diagnostics on forward deltas
        diffs = np.diff(canonical)
        min_forward = float(np.min(diffs)) if diffs.size else float("inf")
        nonpos = int(np.sum(diffs <= 0.0)) if not ALLOW_EQUAL_TIMEPOINTS else int(
            np.sum(diffs < -float(TIME_DECREASE_TOLERANCE)))
        q01 = float(np.quantile(diffs, 0.01)) if diffs.size else float("nan")
        q50 = float(np.quantile(diffs, 0.50)) if diffs.size else float("nan")
        q99 = float(np.quantile(diffs, 0.99)) if diffs.size else float("nan")

        log.info(f"[t-canon] canonical T={canonical.shape[0]} "
                 f"min_forward_dt={min_forward:.6g} "
                 f"nonpositive_steps={nonpos} "
                 f"p01={q01:.6g} median={q50:.6g} p99={q99:.6g}")

        if nonpos > 0:
            j = int(np.where(diffs <= 0.0)[0][0]) if not ALLOW_EQUAL_TIMEPOINTS else int(
                np.where(diffs < -float(TIME_DECREASE_TOLERANCE))[0][0])
            log.warning(f"[t-canon] example nonpositive step at j={j} "
                        f"(t[j], t[j+1])=({canonical[j]}, {canonical[j + 1]})")

        self._canonical_time = canonical

    def _log_memory_estimate(self) -> None:
        """Log estimated memory usage for processing."""
        try:
            N = int(self._drop_report["overall"]["n_valid"])
            T = int(self._canonical_time.shape[0])
            S = int(len(self.species_vars))
            itemsize = int(np.dtype(self.storage_dtype).itemsize)

            estimated_memory = int(self._canonical_time.nbytes) + N * T * S * itemsize
            self.logger.info(f"Estimated memory for sharding: {format_bytes(estimated_memory)}")
        except Exception:
            pass

    def _compute_dt_specification(self, time_grid: np.ndarray) -> Dict[str, float]:
        """
        Compute centralized dt normalization specification from a canonical time grid.

        Returns:
            {
              "method": "log-min-max",
              "log_min": float,
              "log_max": float,
              # extra audit fields:
              "k1_min": float,
              "k1_max": float,
            }
        """
        import logging
        log = self.logger or logging.getLogger("preprocessor")

        t = np.asarray(time_grid, dtype=np.float64).reshape(-1)
        if t.size < 2:
            raise ValueError("Canonical time grid must have at least 2 points")

        diffs1 = np.diff(t)
        if not ALLOW_EQUAL_TIMEPOINTS:
            if np.any(diffs1 <= 0.0):
                j = int(np.where(diffs1 <= 0.0)[0][0])
                raise ValueError(f"Nonpositive forward step in canonical grid at j={j} "
                                 f"(t[j], t[j+1])=({t[j]}, {t[j + 1]})")
        else:
            if np.any(diffs1 < -float(TIME_DECREASE_TOLERANCE)):
                j = int(np.where(diffs1 < -float(TIME_DECREASE_TOLERANCE))[0][0])
                raise ValueError(f"Decreasing time step beyond tolerance at j={j} "
                                 f"(t[j], t[j+1])=({t[j]}, {t[j + 1]})")

        # k=1 diagnostics
        k1_min = float(diffs1.min())
        k1_max = float(diffs1.max())
        log.info(f"[dt-spec] k=1 forward Δt range: [{k1_min:.6g}, {k1_max:.6g}] s")

        # Sweep k in [min_steps, max_steps] (inclusive), vectorized
        T = t.shape[0]
        k_min = int(self.min_steps)
        k_max = int(self.max_steps) if self.max_steps is not None else (T - 1)
        k_min = max(1, min(k_min, T - 1))
        k_max = max(k_min, min(k_max, T - 1))

        dt_min_all = float("inf")
        dt_max_all = 0.0

        for k in range(k_min, k_max + 1):
            dtk = t[k:] - t[:-k]  # [T-k]
            # Strictly positive by earlier checks; still guard with epsilon
            dtk_min = float(np.min(dtk))
            dtk_max = float(np.max(dtk))
            dt_min_all = min(dt_min_all, dtk_min)
            dt_max_all = max(dt_max_all, dtk_max)

        # Clip by epsilon to avoid log of <= 0
        dt_min_clipped = max(dt_min_all, float(self.epsilon))
        dt_max_clipped = max(dt_max_all, float(self.epsilon))
        log_min = float(np.log10(dt_min_clipped))
        log_max = float(np.log10(dt_max_clipped))

        log.info(f"[dt-spec] trained Δt range over k in [{k_min},{k_max}]: "
                 f"[{10.0 ** log_min:.6g}, {10.0 ** log_max:.6g}]")

        return {
            "method": "log-min-max",
            "log_min": log_min,
            "log_max": log_max,
            "k1_min": k1_min,
            "k1_max": k1_max,
        }

    def _write_shards_and_collect_stats_parallel(self) -> Dict[str, RunningStatistics]:
        """
        Write NPZ shards with global buffers across files to create fewer shards.

        Returns:
            Dictionary of statistics for each variable (aggregated across all ranks)
        """
        seed = self.seed  # Use the seed from config for deterministic hashing

        # Initialize statistics accumulators for training data only
        all_keys = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        need_flags = {
            key: get_normalization_flags(self.methods.get(key, self.default_method))
            for key in all_keys
        }

        local_train_stats = {
            key: RunningStatistics(
                need_mean_std=flags[0],
                need_min_max=flags[1],
                need_log=flags[2],
                epsilon=self.epsilon
            )
            for key, flags in need_flags.items()
        }

        # Create split directories (rank 0 only)
        if self.rank == 0:
            split_dirs = {
                "train": self.processed_dir / "train",
                "validation": self.processed_dir / "validation",
                "test": self.processed_dir / "test",
            }
            for directory in split_dirs.values():
                directory.mkdir(parents=True, exist_ok=True)

        # Synchronize after directory creation
        if self.comm:
            self.comm.Barrier()

        # Global buffers that persist across all files
        global_buffers = {
            "train": {"x0": [], "g": [], "t": [], "y": []},
            "validation": {"x0": [], "g": [], "t": [], "y": []},
            "test": {"x0": [], "g": [], "t": [], "y": []},
        }

        # Use a consistent file tag for mixed trajectories
        filetag = f"mix_r{self.rank}"

        # Distribute work across ranks
        work_items = []
        for path_str in self.raw_files:
            valid_groups = self._valid_group_names.get(str(path_str), [])
            if valid_groups:
                # CHANGED: Don't chunk within files - process all groups at once
                work_items.append((path_str, valid_groups))

        # Distribute work items across ranks
        if self.comm:
            # Round-robin distribution
            my_work = [work_items[i] for i in range(len(work_items)) if i % self.size == self.rank]
        else:
            my_work = work_items

        # Process assigned work
        local_split_counts = {"train": 0, "validation": 0, "test": 0}
        local_shards_metadata = {"train": [], "validation": [], "test": []}

        # Use a rank-specific shard counter to avoid collisions
        shard_counter = self.rank * 100000  # Each rank gets a range of 100k shard numbers

        # Function to flush a buffer when it's full
        def flush_buffer_if_full(split_name):
            nonlocal shard_counter
            buffer = global_buffers[split_name]
            if len(buffer["x0"]) >= self.traj_per_shard:
                self._write_shard(
                    self.processed_dir / split_name,
                    buffer,
                    filetag,
                    split_name,
                    shard_counter
                )
                local_shards_metadata[split_name].append({
                    "filename": f"shard_{split_name}_{filetag}_{shard_counter:05d}.npz",
                    "n_trajectories": len(buffer["x0"])
                })
                shard_counter += 1
                # Clear the buffer
                for key in buffer:
                    buffer[key].clear()

        # Progress tracking
        if self.rank == 0:
            total_work = len(work_items)
            self.logger.info(f"Processing {total_work} files across all ranks")

        for work_idx, (path_str, groups) in enumerate(my_work):
            if self.rank == 0 and work_idx % 10 == 0:
                self.logger.info(f"Rank 0 processing file {work_idx + 1}/{len(my_work)}")

            # Process this file's groups into global buffers
            self._add_file_to_buffers(
                Path(path_str), groups, seed,
                global_buffers, local_split_counts,
                local_train_stats, flush_buffer_if_full
            )

        # Final flush of any remaining trajectories
        for split_name in ("train", "validation", "test"):
            buffer = global_buffers[split_name]
            if buffer["x0"]:  # If there's anything left
                self._write_shard(
                    self.processed_dir / split_name,
                    buffer,
                    filetag,
                    split_name,
                    shard_counter
                )
                local_shards_metadata[split_name].append({
                    "filename": f"shard_{split_name}_{filetag}_{shard_counter:05d}.npz",
                    "n_trajectories": len(buffer["x0"])
                })
                shard_counter += 1

        # Log shard statistics for this rank
        if self.rank == 0 or not self.comm:
            self.logger.info(
                f"Rank {self.rank} created shards: "
                f"train={len(local_shards_metadata['train'])}, "
                f"val={len(local_shards_metadata['validation'])}, "
                f"test={len(local_shards_metadata['test'])}"
            )

        # Aggregate statistics across all ranks
        if self.comm:
            # Gather all local statistics
            all_local_stats = self.comm.gather(local_train_stats, root=0)
            all_split_counts = self.comm.gather(local_split_counts, root=0)
            all_shards_metadata = self.comm.gather(local_shards_metadata, root=0)

            if self.rank == 0:
                # Merge statistics from all ranks
                merged_stats = self._merge_statistics(all_local_stats)

                # Merge split counts
                merged_split_counts = {"train": 0, "validation": 0, "test": 0}
                for counts in all_split_counts:
                    for split in merged_split_counts:
                        merged_split_counts[split] += counts[split]

                # Merge shard metadata
                merged_shards_metadata = {"train": [], "validation": [], "test": []}
                for metadata in all_shards_metadata:
                    for split in merged_shards_metadata:
                        merged_shards_metadata[split].extend(metadata[split])

                # Write metadata files
                self._write_shard_index(merged_shards_metadata)
                self._write_preprocessing_summary(merged_split_counts)

                self.logger.info(
                    f"Created shards - train: {len(merged_shards_metadata['train'])}, "
                    f"validation: {len(merged_shards_metadata['validation'])}, "
                    f"test: {len(merged_shards_metadata['test'])}"
                )
                self.logger.info(
                    f"Total trajectories - train: {merged_split_counts['train']}, "
                    f"validation: {merged_split_counts['validation']}, "
                    f"test: {merged_split_counts['test']}"
                )

                return merged_stats
            else:
                return {}
        else:
            # Serial mode
            self._write_shard_index(local_shards_metadata)
            self._write_preprocessing_summary(local_split_counts)
            self.logger.info(
                f"Created shards - train: {len(local_shards_metadata['train'])}, "
                f"validation: {len(local_shards_metadata['validation'])}, "
                f"test: {len(local_shards_metadata['test'])}"
            )
            self.logger.info(
                f"Total trajectories - train: {local_split_counts['train']}, "
                f"validation: {local_split_counts['validation']}, "
                f"test: {local_split_counts['test']}"
            )
            return local_train_stats

    def _add_file_to_buffers(
            self,
            path: Path,
            groups: List[str],
            seed: int,
            global_buffers: Dict[str, Dict[str, List]],
            split_counts: Dict[str, int],
            train_stats: Dict[str, RunningStatistics],
            flush_callback
    ) -> None:
        """
        Add a file's trajectories to global buffers.

        This replaces the old _process_chunk_to_shards method.
        """
        # Deterministic shuffling using deterministic_hash for reproducibility
        file_hash_seed = int(deterministic_hash(path.name, seed) * 2 ** 32)
        rng = np.random.default_rng(seed ^ file_hash_seed)
        order = rng.permutation(len(groups))
        groups_ordered = [groups[i] for i in order]

        with h5py.File(path, "r") as hdf:
            for group_name in groups_ordered:
                group = hdf[group_name]

                # Determine split using deterministic, file-aware hash
                split_key = f"{path.name}:{group_name}:split"
                hash_val = deterministic_hash(split_key, seed)
                if hash_val < self.test_fraction:
                    split = "test"
                elif hash_val < self.test_fraction + self.val_fraction:
                    split = "validation"
                else:
                    split = "train"

                # Apply use_fraction filter (file-aware)
                if self.use_fraction < 1.0:
                    use_key = f"{path.name}:{group_name}:use"
                    if deterministic_hash(use_key, seed) >= self.use_fraction:
                        continue

                # Load time data with full original length
                try:
                    time_data_full = np.array(group[self.time_key], dtype=np.float64, copy=False).reshape(-1)
                except (ValueError, TypeError):
                    # Fallback for newer h5py versions
                    time_data_full = np.array(group[self.time_key], dtype=np.float64).reshape(-1)

                T_full = int(time_data_full.shape[0])

                # Load species matrix with full original length
                species_matrix_full = self._read_species_matrix(group, T_full)

                # NOW skip first timestep if requested
                if self.skip_first_timestep:
                    time_data = time_data_full[1:]
                    species_matrix = species_matrix_full[1:, :]
                else:
                    time_data = time_data_full
                    species_matrix = species_matrix_full

                T = int(time_data.shape[0])

                # Validate after skipping
                if T < 2:
                    continue  # Need at least 2 timesteps after skipping

                if not np.isfinite(species_matrix).all():
                    continue

                if (species_matrix < self.min_value_threshold).any():
                    continue

                # Load globals
                global_values = np.array(
                    [float(group.attrs[key]) for key in self.global_vars],
                    dtype=np.float64
                )

                if not np.isfinite(global_values).all():
                    continue

                # Initial condition (now from the potentially shifted array)
                x0 = species_matrix[0, :].astype(self.storage_dtype, copy=False)

                # Add to global buffer
                buffer = global_buffers[split]
                buffer["x0"].append(x0)
                buffer["g"].append(global_values.astype(self.storage_dtype, copy=False))
                buffer["t"].append(time_data.astype(np.float64, copy=False))
                buffer["y"].append(species_matrix.astype(self.storage_dtype, copy=False))
                split_counts[split] += 1

                # Update training statistics
                if split == "train":
                    self._update_training_stats(
                        train_stats, time_data, global_values, species_matrix
                    )

                # Flush buffer if it's full
                flush_callback(split)

    def _merge_statistics(self, all_stats: List[Dict[str, RunningStatistics]]) -> Dict[str, RunningStatistics]:
        """
        Merge statistics from multiple ranks using Welford's algorithm.

        Args:
            all_stats: List of statistics dictionaries from each rank

        Returns:
            Merged statistics dictionary
        """
        if not all_stats:
            return {}

        # Initialize with first rank's stats structure
        merged = {}
        all_keys = all_stats[0].keys()

        for key in all_keys:
            # Get configuration from first rank
            first_stat = all_stats[0][key]
            merged_stat = RunningStatistics(
                need_mean_std=first_stat.need_mean_std,
                need_min_max=first_stat.need_min_max,
                need_log=first_stat.need_log,
                epsilon=first_stat.epsilon
            )

            # Merge raw statistics
            if first_stat.need_mean_std or first_stat.need_min_max:
                total_count = 0
                total_mean = 0.0
                total_M2 = 0.0
                global_min = math.inf
                global_max = -math.inf

                for rank_stats in all_stats:
                    stat = rank_stats[key]
                    if stat.raw and stat.raw.count > 0:
                        # Merge using parallel Welford's algorithm
                        rank_count = stat.raw.count
                        rank_mean = stat.raw.mean
                        rank_M2 = stat.raw.M2

                        delta = rank_mean - total_mean
                        new_count = total_count + rank_count
                        new_mean = total_mean + delta * (rank_count / new_count) if new_count > 0 else 0
                        new_M2 = total_M2 + rank_M2 + delta * delta * (
                                total_count * rank_count / new_count) if new_count > 0 else 0

                        total_count = new_count
                        total_mean = new_mean
                        total_M2 = new_M2
                        global_min = min(global_min, stat.raw.min_val)
                        global_max = max(global_max, stat.raw.max_val)

                if total_count > 0:
                    merged_stat.raw = WelfordAccumulator()
                    merged_stat.raw.count = total_count
                    merged_stat.raw.mean = total_mean
                    merged_stat.raw.M2 = total_M2
                    merged_stat.raw.min_val = global_min
                    merged_stat.raw.max_val = global_max

            # Merge log statistics
            if first_stat.need_log:
                total_count = 0
                total_mean = 0.0
                total_M2 = 0.0
                global_min = math.inf
                global_max = -math.inf

                for rank_stats in all_stats:
                    stat = rank_stats[key]
                    if stat.log and stat.log.count > 0:
                        rank_count = stat.log.count
                        rank_mean = stat.log.mean
                        rank_M2 = stat.log.M2

                        delta = rank_mean - total_mean
                        new_count = total_count + rank_count
                        new_mean = total_mean + delta * (rank_count / new_count) if new_count > 0 else 0
                        new_M2 = total_M2 + rank_M2 + delta * delta * (
                                total_count * rank_count / new_count) if new_count > 0 else 0

                        total_count = new_count
                        total_mean = new_mean
                        total_M2 = new_M2
                        global_min = min(global_min, stat.log.min_val)
                        global_max = max(global_max, stat.log.max_val)

                if total_count > 0:
                    merged_stat.log = WelfordAccumulator()
                    merged_stat.log.count = total_count
                    merged_stat.log.mean = total_mean
                    merged_stat.log.M2 = total_M2
                    merged_stat.log.min_val = global_min
                    merged_stat.log.max_val = global_max

            merged[key] = merged_stat

        return merged

    def _read_species_matrix(self, group, T: int) -> np.ndarray:
        """Read species data into matrix form."""
        S = len(self.species_vars)
        matrix = np.empty((T, S), dtype=np.float64)

        for species_idx, species_name in enumerate(self.species_vars):
            dataset = group[species_name]

            if int(dataset.shape[0]) != T:
                raise ValueError("Species length mismatch")
            if dataset.ndim not in (1, 2):
                raise ValueError(f"Species '{species_name}' must be 1D or 2D with last dim == 1")
            if dataset.ndim == 2 and int(dataset.shape[1]) != 1:
                raise ValueError(f"Species '{species_name}' has shape {dataset.shape}; expected second dim == 1")

            # Read in chunks
            chunk_len = self.hdf5_chunk_size if self.hdf5_chunk_size > 0 else T
            position = 0

            for start_idx in range(0, T, chunk_len):
                end_idx = min(T, start_idx + chunk_len)
                try:
                    chunk = np.array(dataset[start_idx:end_idx], dtype=np.float64, copy=False)
                except (ValueError, TypeError):
                    # Fallback for newer h5py versions
                    chunk = np.array(dataset[start_idx:end_idx], dtype=np.float64)

                if chunk.ndim == 2:
                    chunk = chunk[:, 0]
                chunk = chunk.reshape(-1)

                matrix[start_idx:end_idx, species_idx] = chunk
                position = end_idx

            if position != T:
                raise RuntimeError("Incomplete read of species data")

        return matrix

    def _update_training_stats(
            self,
            train_stats: Dict[str, RunningStatistics],
            time_data: np.ndarray,
            global_values: np.ndarray,
            species_matrix: np.ndarray
    ) -> None:
        """Update training statistics with new data."""
        # Time statistics
        train_stats[self.time_key].update(time_data[None, :])

        # Global statistics
        for global_idx, global_name in enumerate(self.global_vars):
            value = np.array([global_values[global_idx]], dtype=np.float64)
            train_stats[global_name].update(value)

        # Species statistics
        for species_idx, species_name in enumerate(self.species_vars):
            data = species_matrix[:, species_idx]
            train_stats[species_name].update(data)

    def _write_shard(
            self,
            output_dir: Path,
            buffer: Dict[str, List[np.ndarray]],
            file_tag: str,
            split_name: str,
            shard_idx: int
    ) -> None:
        """Write buffer contents to NPZ shard with strict time-grid checks and diagnostics."""
        import logging
        log = self.logger or logging.getLogger("preprocessor")

        if len(buffer.get("x0", [])) == 0:
            raise RuntimeError("Empty shard buffer: no trajectories")

        # Stack arrays
        x0 = np.stack(buffer["x0"], axis=0)
        globals_array = (
            np.stack(buffer["g"], axis=0) if self.global_vars
            else np.zeros((len(buffer["x0"]), 0), dtype=self.storage_dtype)
        )
        y = np.stack(buffer["y"], axis=0)

        # Validate time consistency across trajectories in this shard
        if len(buffer["t"]) == 0:
            raise RuntimeError("Empty shard buffer: no time vectors present")

        time_ref = np.asarray(buffer["t"][0], dtype=np.float64).reshape(-1)
        for idx, time_array in enumerate(buffer["t"][1:], start=1):
            arr = np.asarray(time_array, dtype=np.float64).reshape(-1)
            if not np.array_equal(time_ref, arr):
                diff_indices = np.where(time_ref != arr)[0]
                first_diff = int(diff_indices[0]) if diff_indices.size else -1
                raise RuntimeError(
                    f"Time vectors differ within shard at local trajectory {idx} "
                    f"(first mismatch index={first_diff})"
                )

        # Forward-dt checks and diagnostics
        diffs = np.diff(time_ref)
        min_forward = float(np.min(diffs)) if diffs.size else float("inf")
        nonpos = int(np.sum(diffs <= 0.0)) if not ALLOW_EQUAL_TIMEPOINTS else int(
            np.sum(diffs < -float(TIME_DECREASE_TOLERANCE)))
        log.info(f"[shard] split={split_name} idx={shard_idx} "
                 f"N={x0.shape[0]} T={time_ref.shape[0]} "
                 f"min_forward_dt={min_forward:.6g} nonpositive_steps={nonpos}")

        if nonpos > 0:
            j = int(np.where(diffs <= 0.0)[0][0]) if not ALLOW_EQUAL_TIMEPOINTS else int(
                np.where(diffs < -float(TIME_DECREASE_TOLERANCE))[0][0])
            raise RuntimeError(
                f"Nonpositive forward dt in shard at j={j} "
                f"(t[j], t[j+1])=({time_ref[j]}, {time_ref[j + 1]})"
            )

        # Write NPZ
        ensure_directory(output_dir)
        output_path = output_dir / SHARD_FILENAME_FORMAT.format(
            split=split_name,
            filetag=file_tag,
            idx=shard_idx
        )

        # Use float32 for t_vec to match downstream expectations and keep sizes reasonable
        time_1d = time_ref.astype(np.float32, copy=False)

        if self.npz_compressed:
            np.savez_compressed(
                output_path,
                x0=x0,
                globals=globals_array,
                t_vec=time_1d,
                y_mat=y
            )
        else:
            np.savez(
                output_path,
                x0=x0,
                globals=globals_array,
                t_vec=time_1d,
                y_mat=y
            )

    def _finalize_manifest(
            self,
            train_stats: Dict[str, RunningStatistics],
            dt_spec: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create final normalization manifest with metadata for auditability."""
        per_key_stats = {}
        for key, accumulator in train_stats.items():
            per_key_stats[key] = accumulator.to_manifest(self.min_std)

        methods = {}
        all_keys = list(dict.fromkeys(self.species_vars + self.global_vars + [self.time_key]))
        for key in all_keys:
            methods[key] = self.methods.get(key, self.default_method)

        return {
            "per_key_stats": per_key_stats,
            "normalization_methods": methods,
            "epsilon": self.epsilon,
            "min_std": self.min_std,
            "dt": dt_spec,
            "meta": {
                "species_variables": list(self.species_vars),
                "global_variables": list(self.global_vars),
                "time_variable": self.time_key,
                "raw_data_files": list(self.raw_files),
            }
        }

    def _write_shard_index(self, shards_metadata: Dict[str, List]) -> None:
        """Write shard index file."""
        T = int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0

        shard_index = {
            "sequence_mode": True,
            "variable_length": False,
            "M_per_sample": T,
            "n_input_species": len(self.species_vars),
            "n_target_species": len(self.species_vars),
            "n_globals": len(self.global_vars),
            "compression": "npz",
            "splits": {
                key: {
                    "shards": value,
                    "n_trajectories": sum(x["n_trajectories"] for x in value)
                }
                for key, value in shards_metadata.items()
            }
        }

        with open(self.processed_dir / "shard_index.json", "w", encoding="utf-8") as f:
            json.dump(shard_index, f, indent=2)

        self.logger.info("Wrote shard_index.json")

    def _write_preprocessing_summary(self, split_counts: Dict[str, int]) -> None:
        """Write preprocessing summary file."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "valid_trajectories": {
                "train": int(split_counts["train"]),
                "validation": int(split_counts["validation"]),
                "test": int(split_counts["test"]),
            },
            "overall_from_scan": self._drop_report["overall"],
            "time_grid_len": int(self._canonical_time.shape[0]) if self._canonical_time is not None else 0,
            "species_variables": self.species_vars,
            "global_variables": self.global_vars,
            "time_variable": self.time_key,
            "raw_data_files": list(self.raw_files)
        }

        with open(self.processed_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Wrote preprocessing_summary.json")

    def _write_reports(
            self,
            start_time: float,
            scan_start: float,
            scan_end: float,
            dt_start: float,
            dt_end: float,
            shard_start: float,
            shard_end: float,
            manifest_start: float,
            manifest_end: float
    ) -> None:
        """Write final preprocessing report."""
        report = {
            "min_value_threshold": self.min_value_threshold,
            "overall_from_scan": self._drop_report["overall"],
            "timings_sec": {
                "scan_phase": round(scan_end - scan_start, 3),
                "dt_computation": round(dt_end - dt_start, 3),
                "shard_writing": round(shard_end - shard_start, 3),
                "manifest_writing": round(manifest_end - manifest_start, 3),
                "total": round(manifest_end - start_time, 3),
            },
            "split_fractions": {
                "train": float(1.0 - self.val_fraction - self.test_fraction),
                "validation": float(self.val_fraction),
                "test": float(self.test_fraction),
            },
            "species_variables": self.species_vars,
            "global_variables": self.global_vars,
            "time_variable": self.time_key,
            "raw_data_files": list(self.raw_files),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(manifest_end))
        }

        with open(self.processed_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self.logger.info("Wrote preprocess_report.json")