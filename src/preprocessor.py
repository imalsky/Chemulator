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
- MPI scales linearly with number of processes
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

# Import all utilities from preprocessor_utils
from src.preprocessor_utils import (
    SHARD_FILENAME_FORMAT,
    format_bytes, ensure_directory, load_config_value, get_storage_dtype,
    parse_precision_np_dtype,
    deterministic_hash, WelfordAccumulator, RunningStatistics,
    get_normalization_flags
)

# MPI tag offset to avoid collisions with other message types.
_MPI_TAG_BASE = 100

_MPI_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "PMIX_RANK",
    "PMIX_SIZE",
    "MPI_LOCALRANKID",
    "MPI_LOCALNRANKS",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _mpi_launcher_detected() -> bool:
    """Best-effort detection for jobs launched under an MPI runtime."""
    for key in _MPI_LAUNCHER_ENV_KEYS:
        value = os.getenv(key)
        if value is None:
            continue
        value = value.strip()
        if value == "":
            continue
        # Any well-formed integer here indicates an MPI launcher context.
        try:
            int(value)
            return True
        except ValueError:
            continue
    return False


def _resolve_repo_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve config paths relative to the repository root when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def scan_hdf5_file_worker(
    path_str: str,
    species_vars: list,
    global_vars: list,
    time_key: str,
    min_value_threshold: float,
    chunk_size: int,
    *,
    skip_first_timestep: bool = False,
    logger_name: str = "main.pre",
):
    """Scan a single HDF5 file.

    Returns:
      file_report: dict
      valid_groups: list[str]
      time_candidate: np.ndarray | None

    Key invariant:
      - All *valid* groups must share an identical time grid.
        If a mismatch is observed, we raise immediately.
    """

    from pathlib import Path
    import logging

    import numpy as np

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required") from e

    logger = logging.getLogger(logger_name)
    # Threshold filtering is deferred to shard-writing so species arrays are read once.
    min_value_threshold = float(min_value_threshold)
    chunk_size = int(chunk_size)
    if not np.isfinite(min_value_threshold):
        raise ValueError("min_value_threshold must be finite")
    # hdf5_chunk_size=0 means "full-read chunks" (single slice), matching config docs.
    if chunk_size < 0:
        raise ValueError("hdf5_chunk_size must be >= 0")
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    file_report: dict = {
        "file": str(path),
        "groups_total": 0,
        "groups_valid": 0,
        "groups_missing_time": 0,
        "groups_missing_species": 0,
        "groups_missing_globals": 0,
        "groups_nonpositive_dt": 0,
        "groups_nan_time": 0,
        "min_forward_dt": None,
        "max_forward_dt": None,
    }

    valid_groups: list[str] = []
    time_candidate: Optional[np.ndarray] = None

    with h5py.File(path, "r") as hdf:
        group_names = sorted(list(hdf.keys()))
        file_report["groups_total"] = len(group_names)

        for group_name in group_names:
            grp = hdf[group_name]

            if time_key not in grp:
                file_report["groups_missing_time"] += 1
                continue

            t = np.asarray(grp[time_key][...], dtype=np.float64).reshape(-1)
            if skip_first_timestep and t.size > 0:
                t = t[1:]

            if t.size < 2 or not np.all(np.isfinite(t)):
                file_report["groups_nan_time"] += 1
                continue

            dt = np.diff(t)
            if np.any(dt <= 0.0):
                file_report["groups_nonpositive_dt"] += 1
                continue

            dt_min = float(dt.min())
            dt_max = float(dt.max())
            file_report["min_forward_dt"] = dt_min if file_report["min_forward_dt"] is None else min(
                float(file_report["min_forward_dt"]), dt_min
            )
            file_report["max_forward_dt"] = dt_max if file_report["max_forward_dt"] is None else max(
                float(file_report["max_forward_dt"]), dt_max
            )

            # Schema validation: required datasets/attrs must exist.
            # Do this here so we never include groups that would later crash
            # in the shard-writing phase.
            if any(key not in grp for key in species_vars):
                file_report["groups_missing_species"] += 1
                continue

            if any(key not in grp.attrs for key in global_vars):
                file_report["groups_missing_globals"] += 1
                continue

            # Time grids must match across valid groups only.
            if time_candidate is None:
                time_candidate = t
            elif not np.array_equal(time_candidate, t):
                logger.error("Time grids are not identical")
                raise ValueError("Time grids are not identical")

            valid_groups.append(group_name)

    file_report["groups_valid"] = len(valid_groups)
    return file_report, valid_groups, time_candidate




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

        # Lazy MPI setup: never initialize MPI at module import time.
        self.comm, self.rank, self.size = self._setup_mpi_context()

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

    def _setup_mpi_context(self) -> tuple[Any, int, int]:
        """Resolve MPI mode and initialize communicator only when required."""
        pre_cfg = self.cfg.get("preprocessing")
        if not isinstance(pre_cfg, dict):
            raise KeyError("Missing config: preprocessing")

        mode_raw = pre_cfg.get("use_mpi", "auto")
        if isinstance(mode_raw, bool):
            mode = "on" if mode_raw else "off"
        else:
            mode = str(mode_raw).strip().lower()
            if mode in {"true", "1", "yes", "on"}:
                mode = "on"
            elif mode in {"false", "0", "no", "off"}:
                mode = "off"
            elif mode == "auto":
                mode = "auto"
            else:
                raise ValueError(
                    "preprocessing.use_mpi must be one of: true/false or 'auto'/'on'/'off'"
                )

        if mode == "off":
            return None, 0, 1

        if mode == "auto" and not _mpi_launcher_detected():
            return None, 0, 1

        # MPI requested (explicitly, or auto-detected launcher context).
        try:
            from mpi4py import MPI as _MPI  # local import to avoid import-time initialization side effects
        except Exception as e:
            if mode == "on":
                raise RuntimeError(
                    "preprocessing.use_mpi=true but mpi4py/MPI is unavailable"
                ) from e
            raise RuntimeError(
                "MPI launcher detected but mpi4py/MPI runtime is unavailable. "
                "Configure MPI correctly or set preprocessing.use_mpi=false."
            ) from e

        comm = _MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return comm, rank, size

    def _load_configuration(self) -> None:
        """Load configuration parameters."""
        # Paths - raw files must be explicitly provided by the user.
        raw_files_config = load_config_value(
            self.cfg, ["paths", "raw_data_files"], required=True
        )
        if not isinstance(raw_files_config, list) or len(raw_files_config) == 0:
            raise ValueError("paths.raw_data_files must be explicitly set and non-empty")
        self.raw_files = [str(p) for p in raw_files_config]
        if self.rank == 0:
            self.logger.info(f"Using {len(self.raw_files)} files from config")

        # Normalize raw file paths to absolute, resolved paths.
        # Relative paths are resolved from repository root.
        self.raw_files = [str(_resolve_repo_path(p)) for p in self.raw_files]

        self.processed_dir = _resolve_repo_path(
            load_config_value(self.cfg, ["paths", "processed_data_dir"], required=True)
        )

        # Data schema
        data_cfg = self.cfg.setdefault("data", {})

        # 1) Load time key
        self.time_key = str(load_config_value(data_cfg, ["time_variable"], required=True))

        # 2) Load globals
        self.global_vars = list(load_config_value(data_cfg, ["global_variables"], required=True))

        # 3) Species: must be explicitly provided.
        species_vars = data_cfg.get("species_variables")
        if not isinstance(species_vars, list) or len(species_vars) == 0:
            raise ValueError("data.species_variables must be explicitly set and non-empty")
        self.species_vars = [str(v) for v in species_vars]

        # Normalization (strict validation)
        norm_cfg = self.cfg.get("normalization")
        if not isinstance(norm_cfg, dict):
            raise KeyError("Missing config: normalization")
        self.default_method = str(load_config_value(norm_cfg, ["default_method"], required=True))
        self.methods = dict(load_config_value(norm_cfg, ["methods"], required=True))
        self.epsilon = float(load_config_value(norm_cfg, ["epsilon"], required=True))
        self.min_std = float(load_config_value(norm_cfg, ["min_std"], required=True))

        # Preprocessing
        preproc_cfg = self.cfg.get("preprocessing")
        if not isinstance(preproc_cfg, dict):
            raise KeyError("Missing config: preprocessing")
        if "allow_empty_splits" in preproc_cfg:
            raise KeyError("Unsupported config key: preprocessing.allow_empty_splits")
        self.npz_compressed = bool(load_config_value(
            preproc_cfg, ["npz_compressed"], required=True
        ))

        self.traj_per_shard = int(
            load_config_value(preproc_cfg, ["trajectories_per_shard"], required=True)
        )

        self.hdf5_chunk_size = int(
            load_config_value(preproc_cfg, ["hdf5_chunk_size"], required=True)
        )
        self.min_value_threshold = float(load_config_value(
            preproc_cfg, ["min_value_threshold"], required=False, default=1e-30
        ))

        self.skip_first_timestep = bool(
            load_config_value(preproc_cfg, ["skip_first_timestep"], required=False, default=False)
        )

        if self.rank == 0:
            if self.comm:
                self.logger.info(f"Using MPI with {self.size} ranks")
            else:
                self.logger.info("Serial mode (MPI disabled)")
            self.logger.info(f"Trajectories per shard: {self.traj_per_shard}")
            if self.skip_first_timestep:
                self.logger.info("Skipping first timestep in all trajectories")

        # Training configuration
        train_cfg = self.cfg.get("training")
        if not isinstance(train_cfg, dict):
            raise KeyError("Missing config: training")
        self.val_fraction = float(load_config_value(train_cfg, ["val_fraction"], required=True))
        self.test_fraction = float(load_config_value(train_cfg, ["test_fraction"], required=True))
        self.use_fraction = float(load_config_value(train_cfg, ["use_fraction"], required=True))
        self.min_steps = int(load_config_value(train_cfg, ["min_steps"], required=True))
        max_steps_raw = load_config_value(train_cfg, ["max_steps"], required=False, default=None)
        self.max_steps = int(max_steps_raw) if max_steps_raw is not None else None

        # Storage dtype
        self.storage_dtype = get_storage_dtype(self.cfg)

        # Time dtype (stored once per shard as t_vec)
        precision_cfg = self.cfg.get("precision")
        if not isinstance(precision_cfg, dict):
            raise KeyError("Missing config: precision")
        self.time_storage_dtype = parse_precision_np_dtype(
            precision_cfg["time_io_dtype"],
            "precision.time_io_dtype",
        )

        # Seed for reproducibility
        if "system" not in self.cfg or "seed" not in self.cfg["system"]:
            raise KeyError("Missing config: system.seed")
        self.seed = int(self.cfg["system"]["seed"])

    def _validate_normalization_methods(self) -> None:
        """Validate normalization.methods coverage and unsupported keys."""
        allowed_methods = {"standard", "min-max", "log-standard", "log-min-max"}
        if self.default_method not in allowed_methods:
            raise ValueError(f"Unknown normalization.default_method: '{self.default_method}'")

        for key, method in self.methods.items():
            method_name = str(method)
            if method_name not in allowed_methods:
                raise ValueError(f"Unknown normalization method for key '{key}': '{method_name}'")

        data_keys = set(self.species_vars) | set(self.global_vars) | {self.time_key}
        extra_keys = sorted(set(self.methods.keys()) - data_keys - {"dt"})
        if extra_keys:
            raise KeyError(
                "Unknown normalization.methods key(s): "
                + ", ".join(extra_keys)
            )

        if "dt" in self.methods:
            dt_method = str(self.methods["dt"])
            if dt_method != "log-min-max":
                raise ValueError(
                    "normalization.methods.dt must be 'log-min-max'. "
                    "dt normalization is controlled by manifest['dt'] and must remain log-min-max."
                )
            if self.rank == 0:
                self.logger.warning(
                    "normalization.methods.dt is ignored at runtime; dt normalization uses manifest['dt']."
                )

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        self._validate_normalization_methods()
        if self.rank != 0:
            return

        # Raw files must be explicitly provided.
        if len(self.raw_files) == 0:
            raise ValueError("No raw data files specified")

        # Output dir policy: controlled by preprocessing.overwrite_data
        overwrite_data = bool(self.cfg["preprocessing"]["overwrite_data"])
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
        if self.val_fraction <= 0.0:
            raise ValueError("val_fraction must be > 0 (empty validation split is not supported)")
        if self.test_fraction <= 0.0:
            raise ValueError("test_fraction must be > 0 (empty test split is not supported)")

        # Offsets
        if self.traj_per_shard <= 0:
            raise ValueError("trajectories_per_shard must be > 0")
        if self.hdf5_chunk_size < 0:
            raise ValueError("hdf5_chunk_size must be >= 0 (0 means full-read chunks)")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self.max_steps is not None and self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")

        if not np.isfinite(self.min_value_threshold):
            raise ValueError("min_value_threshold must be finite")

        # Normalization config presence
        norm = self.cfg.get("normalization", {})
        if "default_method" not in norm:
            raise KeyError("normalization.default_method is required")
        if "methods" not in norm:
            raise KeyError("normalization.methods is required")

    def _require_non_empty_splits(self, split_counts: Dict[str, int]) -> None:
        """Fail fast if any train/validation/test split ended up empty."""
        missing = [name for name, count in split_counts.items() if int(count) <= 0]
        if not missing:
            return

        counts_str = ", ".join(f"{k}={int(v)}" for k, v in split_counts.items())
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Empty split(s) detected after preprocessing: {missing_str} ({counts_str}). "
            "Increase data volume or adjust split fractions/use_fraction."
        )

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

        # Synchronize all ranks before exit
        if self.comm:
            self.comm.Barrier()

    def _scan_files(self) -> None:
        """Scan HDF5 files to identify valid trajectories using MPI if available."""
        if self.comm and self.size > 1:
            # MPI path: rank 0 gets the full list of time candidates; others get []
            time_candidates = self._scan_files_mpi()
        else:
            # Serial path: scan directly and fill time_candidates locally
            time_candidates: List[np.ndarray] = []
            self._scan_files_serial(time_candidates)

        # Validate canonical time on rank 0, then broadcast
        if not self.comm or self.rank == 0:
            self._validate_canonical_time(time_candidates)

        if self.comm:
            # Broadcast canonical time grid to all ranks
            self._canonical_time = self.comm.bcast(
                getattr(self, "_canonical_time", None),
                root=0
            )

    def _scan_files_mpi(self) -> List[np.ndarray]:
        """
        Scan files in parallel using MPI, avoiding the ~2 GiB mpi4py gather limit
        by using point-to-point send/recv instead of Comm.gather on large objects.

        Returns (on rank 0):
            List[np.ndarray]: all time candidates collected from every rank.
        Returns (on non-root ranks):
            [] (empty list; canonical time is broadcast later).
        """
        # Distribute files across ranks
        n_files = len(self.raw_files)
        files_per_rank = n_files // self.size
        remainder = n_files % self.size

        start_idx = self.rank * files_per_rank + min(self.rank, remainder)
        end_idx = start_idx + files_per_rank + (self.rank < remainder)
        my_files = self.raw_files[start_idx:end_idx]

        local_results: List[Tuple[dict, List[str], Optional[np.ndarray]]] = []
        local_time_candidates: List[np.ndarray] = []

        # Heartbeat state (per rank)
        last_heartbeat = time.time()

        # Each rank processes its files
        for idx, file_path in enumerate(my_files, 1):
            result = scan_hdf5_file_worker(
                file_path,
                self.species_vars,
                self.global_vars,
                self.time_key,
                self.min_value_threshold,
                self.hdf5_chunk_size,
                skip_first_timestep=self.skip_first_timestep,
            )
            file_report, valid_groups, time_candidate = result
            local_results.append((file_report, valid_groups, time_candidate))

            if time_candidate is not None:
                local_time_candidates.append(time_candidate)

            # Simple heartbeat every ~30 seconds per rank
            now = time.time()
            if now - last_heartbeat >= 30.0:
                self.logger.info(
                    f"[heartbeat] rank={self.rank}/{self.size} "
                    f"scanned {idx}/{len(my_files)} local files"
                )
                last_heartbeat = now

        # Helper: accumulate results into drop_report / valid_group_names on rank 0
        def _accumulate_rank_results(
            rank_results: List[Tuple[dict, List[str], Optional[np.ndarray]]]
        ) -> None:

            for file_report, valid_groups, _ in rank_results:
                self._drop_report["files"].append(file_report)

                n_total = int(file_report.get("groups_total", 0))
                n_valid = int(file_report.get("groups_valid", 0))
                n_dropped = max(0, n_total - n_valid)

                overall = self._drop_report["overall"]
                overall["n_total"] += n_total
                overall["n_valid"] += n_valid
                overall["n_dropped"] += n_dropped
                overall["n_nan"] += int(file_report.get("groups_nan_time", 0))

                # Use canonical key from worker report
                key = file_report["file"]
                self._valid_group_names[key] = valid_groups
                self.logger.info(f"File {Path(file_report['file']).name}: {n_valid} valid / {n_total} total")

        # Root collects everything explicitly via send/recv; non-root sends its data
        if self.rank == 0:
            # Start with rank 0's own results
            time_candidates: List[np.ndarray] = list(local_time_candidates)
            _accumulate_rank_results(local_results)

            # Receive from all other ranks one-by-one
            for src in range(1, self.size):
                rank_results, rank_time_candidates = self.comm.recv(source=src, tag=_MPI_TAG_BASE + src)
                _accumulate_rank_results(rank_results)
                time_candidates.extend(rank_time_candidates)
        else:
            # Non-root ranks send their local results + time candidates to root
            self.comm.send((local_results, local_time_candidates), dest=0, tag=_MPI_TAG_BASE + self.rank)
            time_candidates = []  # Non-root does not need the full list

        # NOTE: no broadcast of _drop_report or _valid_group_names here.
        # Only rank 0 holds the global mapping; other ranks keep just their
        # local maps and will receive per-rank work assignments later.

        # Only rank 0 returns the full list; others return an empty list
        return time_candidates if self.rank == 0 else []


    def _scan_files_serial(self, time_candidates: List[np.ndarray]) -> None:
        """Scan files serially."""
        # Heartbeat state
        last_heartbeat = time.time()
        n_files = len(self.raw_files)

        for file_idx, path in enumerate(self.raw_files, 1):
            path_obj = Path(path)
            self.logger.info(f"Scanning file {file_idx}/{n_files}: {path_obj.name}")

            file_report, valid_groups, time_candidate = scan_hdf5_file_worker(
                path,
                self.species_vars,
                self.global_vars,
                self.time_key,
                self.min_value_threshold,
                self.hdf5_chunk_size,
                skip_first_timestep=self.skip_first_timestep,
            )

            # Update global report
            self._drop_report["files"].append(file_report)

            n_total = int(file_report.get("groups_total", 0))
            n_valid = int(file_report.get("groups_valid", 0))
            n_dropped = max(0, n_total - n_valid)

            overall = self._drop_report["overall"]
            overall["n_total"] += n_total
            overall["n_valid"] += n_valid
            overall["n_dropped"] += n_dropped
            overall["n_nan"] += int(file_report.get("groups_nan_time", 0))

            # Keep a consistent key (absolute path string)
            self._valid_group_names[file_report["file"]] = valid_groups

            if time_candidate is not None:
                time_candidates.append(time_candidate)

            self.logger.info(f"  Found {n_valid} valid, {n_dropped} dropped trajectories")

            # Heartbeat every ~60 seconds
            now = time.time()
            if now - last_heartbeat >= 60.0:
                self.logger.info(
                    f"[heartbeat] serial scanned {file_idx}/{n_files} files"
                )
                last_heartbeat = now

    def _validate_canonical_time(self, time_candidates: List[np.ndarray]) -> None:
        """Validate and set the canonical time grid.

        Requirement:
          - Time grids must be identical. If any candidate differs, we raise.
        """

        if not time_candidates:
            raise ValueError("No time candidates")

        canonical = time_candidates[0]
        if canonical.ndim != 1:
            raise ValueError("Time grid must be 1D")

        dt = np.diff(canonical.astype(np.float64))
        if np.any(dt <= 0.0):
            raise ValueError("Non-increasing time grid")

        for other in time_candidates[1:]:
            if other.shape != canonical.shape or not np.array_equal(other, canonical):
                raise ValueError("Time grids are not identical")

        # If time_variable normalization is log-domain, retained canonical times
        # must be strictly positive (no clamping/epsilon shifts).
        time_method = str(self.methods.get(self.time_key, self.default_method))
        if time_method in {"log-standard", "log-min-max"} and np.any(canonical <= 0.0):
            min_t = float(np.min(canonical))
            raise ValueError(
                f"time_variable '{self.time_key}' uses {time_method}, but canonical time contains non-positive values "
                f"(min={min_t:.6g}). Use a non-log time normalization method or drop non-positive timesteps."
            )

        self._canonical_time = canonical
        self._dt_min = float(dt.min())
        self._dt_max = float(dt.max())

        if self.rank == 0:
            self.logger.info(
                "[t-canon] Canonical time grid validated across %d candidate file(s); n=%d dt_min=%.6e dt_max=%.6e",
                len(time_candidates),
                int(canonical.size),
                self._dt_min,
                self._dt_max,
            )

    def _log_memory_estimate(self) -> None:
        """Log estimated memory usage for processing."""
        if self._canonical_time is None:
            raise RuntimeError("Canonical time grid not set")

        N = int(self._drop_report["overall"]["n_valid"])
        T = int(self._canonical_time.shape[0])
        S = int(len(self.species_vars))
        itemsize = int(np.dtype(self.storage_dtype).itemsize)

        estimated_memory = int(self._canonical_time.nbytes) + N * T * S * itemsize
        self.logger.info(f"Estimated memory for sharding: {format_bytes(estimated_memory)}")

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

        # Align dt normalization bounds with the exact persisted shard representation.
        # This avoids false out-of-range errors later when dataset code recomputes dt
        # from t_vec loaded from NPZ (which is saved at precision.time_io_dtype).
        t_quantized = np.asarray(t, dtype=self.time_storage_dtype).astype(np.float64, copy=False)
        if t_quantized.shape != t.shape:
            raise RuntimeError("Internal error while quantizing canonical time grid")
        if not np.array_equal(t_quantized, t):
            max_abs = float(np.max(np.abs(t_quantized - t)))
            log.warning(
                "[dt-spec] Canonical time quantized to precision.time_io_dtype=%s "
                "(max_abs_diff=%.6g).",
                str(self.time_storage_dtype),
                max_abs,
            )
        t = t_quantized

        diffs1 = np.diff(t)
        if np.any(diffs1 <= 0.0):
            min_dt = float(np.min(diffs1))
            raise ValueError(
                "Non-increasing time grid after applying precision.time_io_dtype="
                f"{self.time_storage_dtype} (min_dt={min_dt:.6g}). "
                "Increase time precision (e.g., float64)."
            )

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

        # For a strictly increasing time grid, the global min dt across all k
        # is at k=k_min (smallest step size), and global max is at k=k_max
        # (largest step size). We only need to compute those two.
        dtk_min_arr = t[k_min:] - t[:-k_min]
        dt_min_all = float(np.min(dtk_min_arr))

        dtk_max_arr = t[k_max:] - t[:-k_max]
        dt_max_all = float(np.max(dtk_max_arr))

        if dt_min_all <= 0.0 or dt_max_all <= 0.0:
            raise ValueError("Non-positive dt encountered while computing dt normalization spec")
        if dt_max_all <= dt_min_all:
            raise ValueError(
                "Degenerate dt normalization range: max dt must be greater than min dt"
            )

        log_min = float(np.log10(dt_min_all))
        log_max = float(np.log10(dt_max_all))
        if (not np.isfinite(log_min)) or (not np.isfinite(log_max)) or (log_max <= log_min):
            raise ValueError("Invalid log-domain dt normalization range")

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
        use_weighting = bool(
            self.cfg.get("training", {})
            .get("adaptive_stiff_loss", {})
            .get("use_weighting", False)
        )
        if use_weighting:
            # Weighting uses per-species log ranges regardless of active species normalization method.
            for key in self.species_vars:
                need_mean_std, need_min_max, _ = need_flags[key]
                need_flags[key] = (need_mean_std, need_min_max, True)
            if self.rank == 0:
                self.logger.info(
                    "training.adaptive_stiff_loss.use_weighting=true; forcing species log statistics collection."
                )

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
            "train": {"g": [], "t_vec": None, "y": []},
            "validation": {"g": [], "t_vec": None, "y": []},
            "test": {"g": [], "t_vec": None, "y": []},
        }

        # Use a consistent file tag for mixed trajectories
        filetag = f"mix_r{self.rank}"

        # ------------------------------------------------------------------
        # Build per-rank work assignments without broadcasting a huge
        # self._valid_group_names object. Root constructs the global list
        # and sends each rank only its share.
        # ------------------------------------------------------------------
        if self.comm:
            if self.rank == 0:
                work_items: List[Tuple[str, List[str]]] = []
                for path_str in self.raw_files:
                    valid_groups = self._valid_group_names.get(str(path_str), [])
                    if valid_groups:
                        # Process all valid groups for this file together
                        work_items.append((path_str, valid_groups))

                # Greedy load balancing by number of valid groups per file.
                # This better matches per-file work than plain round-robin.
                assignments: Dict[int, List[Tuple[str, List[str]]]] = {r: [] for r in range(self.size)}
                rank_loads: Dict[int, int] = {r: 0 for r in range(self.size)}
                weighted_items = sorted(
                    work_items,
                    key=lambda item: (-len(item[1]), item[0]),
                )
                for item in weighted_items:
                    r = min(rank_loads, key=lambda rid: (rank_loads[rid], rid))
                    assignments[r].append(item)
                    rank_loads[r] += max(1, len(item[1]))

                self.logger.info(
                    f"Processing {len(work_items)} files across {self.size} ranks in shard-writing phase"
                )

                my_work = assignments[0]
                for r in range(1, self.size):
                    self.comm.send(assignments[r], dest=r, tag=200 + r)
            else:
                my_work: List[Tuple[str, List[str]]] = self.comm.recv(source=0, tag=200 + self.rank)
        else:
            # Serial mode: build local work list directly
            my_work: List[Tuple[str, List[str]]] = []
            for path_str in self.raw_files:
                valid_groups = self._valid_group_names.get(str(path_str), [])
                if valid_groups:
                    my_work.append((path_str, valid_groups))
            self.logger.info(
                f"Processing {len(my_work)} files in serial shard-writing phase"
            )

        # Process assigned work
        local_split_counts = {"train": 0, "validation": 0, "test": 0}
        local_shards_metadata = {"train": [], "validation": [], "test": []}
        local_nan_count = 0
        local_below_threshold_count = 0
        local_use_fraction_count = 0
        # Canonical time grid is identical across all trajectories/ranks.
        # Collect time stats once on rank 0 from the canonical grid directly,
        # before entering the file loop, so we don't depend on rank 0 receiving
        # training trajectories.
        if self.rank == 0 and self._canonical_time is not None:
            local_train_stats[self.time_key].update(self._canonical_time[None, :])

        # Shard indices are local to this rank; filenames already include filetag=rank.
        shard_counter = 0

        # Function to flush a buffer when it's full
        def flush_buffer_if_full(split_name: str) -> None:
            nonlocal shard_counter
            buffer = global_buffers[split_name]
            if len(buffer["y"]) >= self.traj_per_shard:
                self._write_shard(
                    self.processed_dir / split_name,
                    buffer,
                    filetag,
                    split_name,
                    shard_counter
                )
                local_shards_metadata[split_name].append({
                    "filename": SHARD_FILENAME_FORMAT.format(
                        split=split_name,
                        filetag=filetag,
                        idx=shard_counter
                    ),
                    "n_trajectories": len(buffer["y"])
                })
                shard_counter += 1
                # Clear the buffer
                for key in buffer:
                    if isinstance(buffer[key], list):
                        buffer[key].clear()
                    else:
                        buffer[key] = None

        # Progress tracking (local)
        if self.rank == 0 and my_work:
            self.logger.info(f"Rank 0 will process {len(my_work)} local files")

        for work_idx, (path_str, groups) in enumerate(my_work):
            if self.rank == 0 and work_idx % 10 == 0:
                self.logger.info(f"Rank 0 processing file {work_idx + 1}/{len(my_work)}")

            # Process this file's groups into global buffers
            file_nan_count, file_below_threshold_count, file_use_fraction_count = self._add_file_to_buffers(
                Path(path_str), groups, seed,
                global_buffers, local_split_counts,
                local_train_stats, flush_buffer_if_full,
            )
            local_nan_count += file_nan_count
            local_below_threshold_count += file_below_threshold_count
            local_use_fraction_count += file_use_fraction_count

        # Final flush of any remaining trajectories
        for split_name in ("train", "validation", "test"):
            buffer = global_buffers[split_name]
            if buffer["y"]:  # If there's anything left
                self._write_shard(
                    self.processed_dir / split_name,
                    buffer,
                    filetag,
                    split_name,
                    shard_counter
                )
                local_shards_metadata[split_name].append({
                    "filename": SHARD_FILENAME_FORMAT.format(
                        split=split_name,
                        filetag=filetag,
                        idx=shard_counter
                    ),
                    "n_trajectories": len(buffer["y"])
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
            all_nan_counts = self.comm.gather(local_nan_count, root=0)
            all_below_threshold_counts = self.comm.gather(local_below_threshold_count, root=0)
            all_use_fraction_counts = self.comm.gather(local_use_fraction_count, root=0)

            if self.rank == 0:
                total_nan = int(sum(all_nan_counts))
                total_below = int(sum(all_below_threshold_counts))
                total_use_fraction = int(sum(all_use_fraction_counts))
                dropped_post_scan = total_nan + total_below + total_use_fraction
                overall = self._drop_report["overall"]
                overall["n_nan"] += total_nan
                overall["n_below_threshold"] += total_below
                overall["n_dropped"] += dropped_post_scan
                overall["n_valid"] = max(0, int(overall["n_valid"]) - dropped_post_scan)

                # Merge statistics from all ranks
                merged_stats = self._merge_statistics(all_local_stats)

                # Merge split counts
                merged_split_counts = {"train": 0, "validation": 0, "test": 0}
                for counts in all_split_counts:
                    for split in merged_split_counts:
                        merged_split_counts[split] += counts[split]

                self._require_non_empty_splits(merged_split_counts)

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
            dropped_post_scan = int(local_nan_count + local_below_threshold_count + local_use_fraction_count)
            overall = self._drop_report["overall"]
            overall["n_nan"] += int(local_nan_count)
            overall["n_below_threshold"] += int(local_below_threshold_count)
            overall["n_dropped"] += dropped_post_scan
            overall["n_valid"] = max(0, int(overall["n_valid"]) - dropped_post_scan)
            self._require_non_empty_splits(local_split_counts)
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
            flush_callback,
    ) -> Tuple[int, int, int]:
        """
        Add a file's trajectories to global buffers.

        Invariant:
          - Each trajectory's time grid must exactly match the canonical grid.
        """
        # Deterministic shuffling using deterministic_hash for reproducibility
        uint32_range = 2 ** 32
        file_id = str(path.expanduser().resolve())
        file_hash_seed = int(deterministic_hash(file_id, seed) * uint32_range)
        rng = np.random.default_rng(seed ^ file_hash_seed)
        order = rng.permutation(len(groups))
        groups_ordered = [groups[i] for i in order]

        local_nan_count = 0
        local_below_threshold_count = 0
        local_use_fraction_count = 0
        with h5py.File(path, "r") as hdf:
            for group_name in groups_ordered:
                group = hdf[group_name]

                # Determine split using deterministic, file-aware hash
                split_key = f"{file_id}:{group_name}:split"
                hash_val = deterministic_hash(split_key, seed)
                if hash_val < self.test_fraction:
                    split = "test"
                elif hash_val < self.test_fraction + self.val_fraction:
                    split = "validation"
                else:
                    split = "train"

                # Apply use_fraction filter (file-aware)
                if self.use_fraction < 1.0:
                    use_key = f"{file_id}:{group_name}:use"
                    if deterministic_hash(use_key, seed) >= self.use_fraction:
                        local_use_fraction_count += 1
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

                # Skip first timestep if requested
                if self.skip_first_timestep:
                    time_data_loaded = time_data_full[1:]
                    species_matrix = species_matrix_full[1:, :]
                else:
                    time_data_loaded = time_data_full
                    species_matrix = species_matrix_full

                T = int(time_data_loaded.shape[0])

                # Validate against canonical time grid (exact match).
                if self._canonical_time is None:
                    raise RuntimeError("Canonical time grid not set")

                if (time_data_loaded.shape != self._canonical_time.shape) or (
                    not np.array_equal(time_data_loaded, self._canonical_time)
                ):
                    raise ValueError("Time grids are not identical")

                # Use canonical time object (no copy needed).
                time_data = self._canonical_time

                # Validate remaining conditions after using canonical time
                if T < 2:
                    continue  # Need at least 2 timesteps

                if not np.isfinite(species_matrix).all():
                    local_nan_count += 1
                    continue

                if self.min_value_threshold > 0.0 and np.nanmin(species_matrix) < self.min_value_threshold:
                    local_below_threshold_count += 1
                    continue

                # Load globals
                global_values = np.array(
                    [float(group.attrs[key]) for key in self.global_vars],
                    dtype=np.float64
                )

                if not np.isfinite(global_values).all():
                    local_nan_count += 1
                    continue

                # Add to global buffer with CANONICAL time grid
                buffer = global_buffers[split]
                buffer["g"].append(global_values.astype(self.storage_dtype, copy=False))
                if buffer["t_vec"] is None:
                    buffer["t_vec"] = time_data.astype(np.float64, copy=False)
                buffer["y"].append(species_matrix.astype(self.storage_dtype, copy=False))
                split_counts[split] += 1

                # Update training statistics (time stats already collected above).
                if split == "train":
                    self._update_training_stats(
                        train_stats, global_values, species_matrix
                    )

                # Flush buffer if it's full
                flush_callback(split)
        return local_nan_count, local_below_threshold_count, local_use_fraction_count

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

        return matrix

    def _update_training_stats(
            self,
            train_stats: Dict[str, RunningStatistics],
            global_values: np.ndarray,
            species_matrix: np.ndarray
    ) -> None:
        """Update training statistics with new data."""
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
        """
        Write buffer contents to an NPZ shard.

        Invariant:
          - Time grids are identical across all trajectories. This is enforced during the scan
            stage; we do not attempt tolerance-based reconciliation here.
        """
        import logging
        log = self.logger or logging.getLogger("preprocessor")

        if len(buffer.get("y", [])) == 0:
            raise RuntimeError("Empty shard buffer: no trajectories")

        # Stack arrays
        globals_array = (
            np.stack(buffer["g"], axis=0) if self.global_vars
            else np.zeros((len(buffer["y"]), 0), dtype=self.storage_dtype)
        )
        y = np.stack(buffer["y"], axis=0)

        time_ref_obj = buffer.get("t_vec")
        if time_ref_obj is None:
            raise RuntimeError("Shard buffer missing time vector")

        time_ref = np.asarray(time_ref_obj, dtype=np.float64).reshape(-1)
        diffs = np.diff(time_ref)
        if np.any(diffs <= 0.0):
            raise RuntimeError("Non-increasing time grid")

        log.info(
            "[shard] split=%s idx=%d N=%d T=%d dt_min=%.6g dt_max=%.6g",
            split_name,
            int(shard_idx),
            int(y.shape[0]),
            int(time_ref.shape[0]),
            float(diffs.min()),
            float(diffs.max()),
        )

        # Write NPZ
        ensure_directory(output_dir)
        output_path = output_dir / SHARD_FILENAME_FORMAT.format(
            split=split_name,
            filetag=file_tag,
            idx=shard_idx
        )

        # Time dtype is controlled by cfg.precision.time_io_dtype.
        time_1d = time_ref.astype(self.time_storage_dtype, copy=False)

        if self.npz_compressed:
            np.savez_compressed(
                output_path,
                globals=globals_array,
                t_vec=time_1d,
                y_mat=y
            )
        else:
            np.savez(
                output_path,
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
