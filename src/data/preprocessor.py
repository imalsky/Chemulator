#!/usr/bin/env python3
"""
Chemical kinetics data preprocessor with sequence mode support for LiLaN.

Key changes:
- Use ONE global, fixed, log-spaced time grid shared by every trajectory.
- Grid always includes both endpoints (endpoint=True).
- Grid range comes from config.data.fixed_time_range {min,max} if present,
  otherwise falls back to the global raw time min/max found by collect_time_stats().
- Enforce coverage: any trajectory whose raw time window does not cover the
  fixed grid is dropped (no extrapolation).
- Remove per-trajectory time sampling and the uniform/log_spaced mixer.
- Keep normalization choice for time (time-norm or log-min-max) driven by config.
- Propagate the fixed grid metadata into normalization.json and shard_index.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from utils.utils import save_json


@dataclass
class DataStatistics:
    """Track preprocessing statistics."""
    total_groups: int = 0
    valid_trajectories: int = 0

    dropped_missing_keys: int = 0
    dropped_non_finite: int = 0
    dropped_below_threshold: int = 0
    dropped_insufficient_time: int = 0
    dropped_use_fraction: int = 0

    total_time_points: int = 0
    species_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    global_min_max: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range: Tuple[float, float] = (float('inf'), float('-inf'))
    split_distribution: Dict[str, int] = field(default_factory=lambda: {"train": 0,
                                                                        "validation": 0,
                                                                        "test": 0})
    processing_times: Dict[str, float] = field(default_factory=dict)
    file_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def merge_inplace(self, other: "DataStatistics") -> None:
        """Merge another stats object into this one."""
        self.total_groups += other.total_groups
        self.valid_trajectories += other.valid_trajectories
        self.dropped_missing_keys += other.dropped_missing_keys
        self.dropped_non_finite += other.dropped_non_finite
        self.dropped_below_threshold += other.dropped_below_threshold
        self.dropped_insufficient_time += other.dropped_insufficient_time
        self.dropped_use_fraction += other.dropped_use_fraction
        self.total_time_points += other.total_time_points

        for k, (mn, mx) in other.species_min_max.items():
            if k in self.species_min_max:
                omn, omx = self.species_min_max[k]
                self.species_min_max[k] = (min(omn, mn), max(omx, mx))
            else:
                self.species_min_max[k] = (mn, mx)

        for k, (mn, mx) in other.global_min_max.items():
            if k in self.global_min_max:
                omn, omx = self.global_min_max[k]
                self.global_min_max[k] = (min(omn, mn), max(omx, mx))
            else:
                self.global_min_max[k] = (mn, mx)

        self.time_range = (
            min(self.time_range[0], other.time_range[0]),
            max(self.time_range[1], other.time_range[1]),
        )

        for split, cnt in other.split_distribution.items():
            self.split_distribution[split] = self.split_distribution.get(split, 0) + cnt

        self.processing_times.update(other.processing_times)
        self.file_stats.update(other.file_stats)


class DataStatisticsLogger:
    """Log comprehensive data statistics during preprocessing."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stats = DataStatistics()
        self.logger = logging.getLogger(__name__)

    def update_species_range(self, var: str, data: np.ndarray):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        vmin, vmax = float(finite.min()), float(finite.max())
        if var in self.stats.species_min_max:
            omin, omax = self.stats.species_min_max[var]
            self.stats.species_min_max[var] = (min(omin, vmin), max(omax, vmax))
        else:
            self.stats.species_min_max[var] = (vmin, vmax)

    def update_global_range(self, var: str, value: float):
        if var in self.stats.global_min_max:
            omin, omax = self.stats.global_min_max[var]
            self.stats.global_min_max[var] = (min(omin, value), max(omax, value))
        else:
            self.stats.global_min_max[var] = (value, value)

    def update_time_range(self, times: np.ndarray):
        if times.size == 0:
            return
        tmin, tmax = float(times.min()), float(times.max())
        self.stats.time_range = (
            min(self.stats.time_range[0], tmin),
            max(self.stats.time_range[1], tmax),
        )
        self.stats.total_time_points += int(times.size)

    def save_summary(self):
        summary_path = self.output_dir / "preprocessing_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_groups_processed": self.stats.total_groups,
            "valid_trajectories": self.stats.valid_trajectories,
            "dropped_counts": {
                "missing_keys": self.stats.dropped_missing_keys,
                "non_finite": self.stats.dropped_non_finite,
                "below_threshold": self.stats.dropped_below_threshold,
                "insufficient_time": self.stats.dropped_insufficient_time,
                "use_fraction": self.stats.dropped_use_fraction,
            },
            "data_ranges": {
                "species": self.stats.species_min_max,
                "globals": self.stats.global_min_max,
                "time": {"min": self.stats.time_range[0], "max": self.stats.time_range[1]},
            },
            "split_distribution": self.stats.split_distribution,
            "total_time_points": self.stats.total_time_points,
            "processing_times": self.stats.processing_times,
            "file_statistics": self.stats.file_stats,
        }
        save_json(summary, summary_path)
        self.logger.info(f"Statistics summary saved to {summary_path}")


class ChunkedHDF5Reader:
    """Read HDF5 data in chunks to minimize memory usage."""
    def __init__(self, file_path: Optional[Path], chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    def read_dataset_chunked(self, group, var_name, indices=None):
        if var_name not in group:
            raise KeyError(f"Variable {var_name} not found in group")
        dset = group[var_name]
        if indices is not None:
            return dset[indices]  # basic indexed read
        total = dset.shape[0]
        if total <= self.chunk_size:
            return dset[:]
        chunks = [dset[i:min(i + self.chunk_size, total)] for i in range(0, total, self.chunk_size)]
        return np.concatenate(chunks)


class SequenceShardWriter:
    """Writes trajectory sequences to NPZ shards with improved buffering."""
    def __init__(
        self,
        output_dir: Path,
        trajectories_per_shard: int,
        shard_idx_base: str,
        M: int,
        n_species: int,
        n_globals: int,
        dtype: np.dtype,
        compressed: bool = True,
    ):
        self.output_dir = output_dir
        self.trajectories_per_shard = max(1, int(trajectories_per_shard))
        self.shard_idx_base = shard_idx_base
        self.M = M
        self.n_species = n_species
        self.n_globals = n_globals
        self.dtype = dtype
        self.compressed = compressed

        self.buffer: List[Dict[str, np.ndarray]] = []
        self.shard_id = 0
        self.shard_metadata: List[Dict[str, Any]] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_trajectory(
        self, x0_log: np.ndarray, globals_vec: np.ndarray, t_vec: np.ndarray, y_mat: np.ndarray
    ):
        """Add trajectory to buffer."""
        self.buffer.append(
            {
                "x0_log": x0_log.astype(self.dtype, copy=False),
                "globals": globals_vec.astype(self.dtype, copy=False),
                "t_vec": t_vec.astype(self.dtype, copy=False),
                "y_mat": y_mat.astype(self.dtype, copy=False),
            }
        )
        if len(self.buffer) >= self.trajectories_per_shard:
            self._write_shard()

    def _write_shard(self):
        """Write buffered trajectories to NPZ file."""
        if not self.buffer:
            return

        x0_log = np.stack([t["x0_log"] for t in self.buffer])
        globals_vec = np.stack([t["globals"] for t in self.buffer])
        t_vec = np.stack([t["t_vec"] for t in self.buffer])
        y_mat = np.stack([t["y_mat"] for t in self.buffer])

        filename = f"shard_{self.shard_idx_base}_{self.shard_id:04d}.npz"
        filepath = self.output_dir / filename

        save_fn = np.savez_compressed if self.compressed else np.savez
        save_fn(filepath, x0_log=x0_log, globals=globals_vec, t_vec=t_vec, y_mat=y_mat)

        self.shard_metadata.append({"filename": filename, "n_trajectories": len(self.buffer)})
        self.buffer = []
        self.shard_id += 1

    def flush(self):
        """Write any remaining trajectories."""
        if self.buffer:
            self._write_shard()


# ==================
# Core Preprocessing
# ==================

class CorePreprocessor:
    """Core preprocessing logic with chunked reading support and a fixed global time grid."""

    def __init__(
        self,
        config: Dict[str, Any],
        norm_stats: Optional[Dict[str, Any]] = None,
        stats_logger: Optional[DataStatisticsLogger] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.stats_logger = stats_logger

        self.data_cfg = config["data"]
        self.norm_cfg = config["normalization"]
        self.train_cfg = config["training"]
        self.proc_cfg = config["preprocessing"]
        self.system_cfg = config["system"]

        # Chunking parameters
        self.chunk_size = int(self.proc_cfg.get("hdf5_chunk_size", 10000))

        # Sequence mode settings
        self.M_per_sample = int(self.data_cfg.get("M_per_sample", 16))

        # Dtype
        dtype_str = self.system_cfg.get("dtype", "float32")
        self.np_dtype = np.float64 if dtype_str == "float64" else np.float32

        # Variables
        self.species_vars = list(self.data_cfg["species_variables"])
        self.target_species_vars = list(self.data_cfg.get("target_species_variables", self.species_vars))
        self.global_vars = list(self.data_cfg["global_variables"])
        self.time_var = str(self.data_cfg["time_variable"])

        self.n_target_species = len(self.target_species_vars)
        self.n_species = len(self.species_vars)
        self.n_globals = len(self.global_vars)

        self.min_value_threshold = float(self.proc_cfg.get("min_value_threshold", 1e-30))
        self.norm_stats = norm_stats or {}

        # Build the fixed global time grid (geomspace) once
        self.fixed_grid = self._init_fixed_time_grid()
        self.fixed_grid = self.fixed_grid.astype(np.float64)  # keep high precision for interpolation
        self.fixed_grid_cast = self.fixed_grid.astype(self.np_dtype)  # cast when storing
        self.log_tq = np.log10(np.maximum(self.fixed_grid, float(self.norm_cfg.get("epsilon", 1e-30))))

    def _init_fixed_time_grid(self) -> np.ndarray:
        """Initialize a single global log-spaced time grid with endpoints included."""
        # Highest priority: explicit config override
        cfg_range = self.data_cfg.get("fixed_time_range", None)
        if cfg_range is not None:
            tmin = float(cfg_range.get("min"))
            tmax = float(cfg_range.get("max"))
        else:
            # Next: from norm_stats (propagated by DataPreprocessor)
            tn = (self.norm_stats or {}).get("time_normalization", {})
            if "tmin_raw" in tn and "tmax_raw" in tn:
                tmin = float(tn["tmin_raw"])
                tmax = float(tn["tmax_raw"])
            else:
                raise ValueError("Time not specified")

        # strict positivity for geomspace
        eps = 1e-12 * max(1.0, abs(tmin))
        a = max(tmin, eps)
        b = max(a * (1.0 + 1e-12), tmax)

        grid = np.geomspace(a, b, int(self.M_per_sample), endpoint=True, dtype=np.float64)
        return grid

    # -------------------
    # Group data extraction
    # -------------------
    def _extract_trajectory_chunked(
        self, group: h5py.Group, gname: str, reader: ChunkedHDF5Reader
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Extract trajectory data using chunked reading. Drops entire trajectory on any bad value."""
        # Globals
        missing = [k for k in self.global_vars if k not in group.attrs]
        if missing:
            if self.stats_logger:
                self.stats_logger.stats.dropped_missing_keys += 1
            return None

        # time
        time_data = reader.read_dataset_chunked(group, self.time_var)
        if not np.all(np.isfinite(time_data)) or time_data.size == 0:
            if self.stats_logger:
                self.stats_logger.stats.dropped_non_finite += 1
            return None
        if self.stats_logger:
            self.stats_logger.update_time_range(time_data)

        # x0 (species at t0)
        x0 = np.empty(self.n_species, dtype=self.np_dtype)
        for i, var in enumerate(self.species_vars):
            if var not in group:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            v0 = float(group[var][0])
            if not np.isfinite(v0) or v0 <= self.min_value_threshold:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None
            x0[i] = v0

        # full target species time series
        species_mat = np.empty((time_data.shape[0], self.n_target_species), dtype=self.np_dtype)
        for j, var in enumerate(self.target_species_vars):
            if var not in group:
                if self.stats_logger:
                    self.stats_logger.stats.dropped_missing_keys += 1
                return None
            arr = reader.read_dataset_chunked(group, var)
            if not np.all(np.isfinite(arr)) or np.any(arr <= self.min_value_threshold):
                if self.stats_logger:
                    self.stats_logger.stats.dropped_below_threshold += 1
                return None

            species_mat[:, j] = arr
            if self.stats_logger:
                self.stats_logger.update_species_range(var, arr)

        globals_vec = np.array([float(group.attrs[k]) for k in self.global_vars], dtype=self.np_dtype)
        if self.stats_logger:
            for i, var in enumerate(self.global_vars):
                self.stats_logger.update_global_range(var, globals_vec[i])

        return time_data, x0, globals_vec, species_mat

    # --------------------------------------
    # Sequence shards + seeded split per file
    # --------------------------------------
    def process_file_for_sequence_shards(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process one file and write sequence-mode shards with seeded, reproducible splits. Returns metadata+stats."""
        start_time = time.time()

        # trajectories per shard (size-based)
        trajectories_per_shard = self.proc_cfg.get("trajectories_per_shard", None)
        if trajectories_per_shard is None:
            target_bytes = int(self.proc_cfg.get("target_shard_bytes", 200 * 1024 * 1024))
            bytes_per_val = 8 if self.np_dtype == np.float64 else 4
            vals_per_traj = (self.n_species + self.n_globals + self.M_per_sample + self.M_per_sample * self.n_target_species)
            traj_bytes = max(bytes_per_val * vals_per_traj, 1)
            trajectories_per_shard = max(1, target_bytes // traj_bytes)

        compressed_npz = bool(self.proc_cfg.get("npz_compressed", True))

        # shard writers for each split
        writers: Dict[str, SequenceShardWriter] = {}
        for split in ("train", "validation", "test"):
            writers[split] = SequenceShardWriter(
                output_dir / split,
                trajectories_per_shard,
                file_path.stem,
                self.M_per_sample,
                self.n_species,
                self.n_globals,
                self.np_dtype,
                compressed=compressed_npz,
            )

        split_counts = {"train": 0, "validation": 0, "test": 0}
        reader = ChunkedHDF5Reader(file_path, self.chunk_size)

        seed = int(self.config["system"].get("seed", 42))

        # Local stats accumulation (so it also works inside worker)
        local_stats = DataStatistics()
        local_time_points = 0

        with h5py.File(file_path, "r") as f:
            for gname in sorted(f.keys()):
                grp = f[gname]
                local_stats.total_groups += 1
                if self.stats_logger:
                    self.stats_logger.stats.total_groups += 1

                if self.time_var not in grp:
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_missing_keys += 1
                    local_stats.dropped_missing_keys += 1
                    continue

                n_t = grp[self.time_var].shape[0]
                local_time_points += int(n_t)
                # Need at least 2 raw points for interpolation
                if n_t < 2:
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_insufficient_time += 1
                    local_stats.dropped_insufficient_time += 1
                    continue

                # use_fraction (seeded)
                if float(self.train_cfg.get("use_fraction", 1.0)) < 1.0:
                    hv = int(hashlib.sha256(f"{seed}:{gname}:use".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if hv >= float(self.train_cfg["use_fraction"]):
                        if self.stats_logger:
                            self.stats_logger.stats.dropped_use_fraction += 1
                        local_stats.dropped_use_fraction += 1
                        continue

                extracted = self._extract_trajectory_chunked(grp, gname, reader)
                if extracted is None:
                    continue

                time_data, x0, globals_vec, species_mat = extracted

                # Enforce coverage: raw time must span the fixed grid range
                raw_min = float(np.min(time_data))
                raw_max = float(np.max(time_data))
                gmin = float(self.fixed_grid[0])
                gmax = float(self.fixed_grid[-1])

                # Relative tolerance on bounds
                rtol = float(self.proc_cfg.get("grid_coverage_rtol", 1e-6))
                if (raw_min > gmin * (1.0 + rtol)) or (raw_max < gmax * (1.0 - rtol)):
                    if self.stats_logger:
                        self.stats_logger.stats.dropped_insufficient_time += 1
                    local_stats.dropped_insufficient_time += 1
                    continue

                local_stats.valid_trajectories += 1
                if self.stats_logger:
                    self.stats_logger.stats.valid_trajectories += 1

                # --- log–log interpolation onto the fixed grid ---
                eps = float(self.norm_cfg.get("epsilon", 1e-30))
                log_t = np.log10(np.maximum(time_data, eps))      # original grid (monotonic)
                log_y = np.log10(np.maximum(species_mat, eps))    # [Nt, T] in log10

                y_mat_log = np.empty((len(self.fixed_grid), self.n_target_species), dtype=self.np_dtype)
                for j in range(self.n_target_species):
                    y_mat_log[:, j] = np.interp(self.log_tq, log_t, log_y[:, j]).astype(self.np_dtype, copy=False)

                x0_log = np.log10(np.maximum(x0, eps)).astype(self.np_dtype, copy=False)
                t_vec = self.fixed_grid_cast  # identical for all trajectories

                # seeded split
                split_hash = int(hashlib.sha256(f"{seed}:{gname}:split".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                test_frac = float(self.train_cfg.get("test_fraction", 0.0))
                val_frac = float(self.train_cfg.get("val_fraction", 0.0))
                if split_hash < test_frac:
                    split = "test"
                elif split_hash < test_frac + val_frac:
                    split = "validation"
                else:
                    split = "train"

                writers[split].add_trajectory(x0_log, globals_vec, t_vec, y_mat_log)
                split_counts[split] += 1
                if self.stats_logger:
                    self.stats_logger.stats.split_distribution[split] += 1

        # flush
        for w in writers.values():
            w.flush()

        elapsed = time.time() - start_time
        # record timing/file stats
        local_stats.total_time_points += local_time_points
        local_stats.processing_times[str(file_path)] = elapsed
        local_stats.file_stats[str(file_path)] = {
            "groups_processed": local_stats.total_groups,
            "valid_trajectories": local_stats.valid_trajectories,
            "processing_time": elapsed,
        }
        if self.stats_logger:
            self.stats_logger.stats.processing_times[str(file_path)] = elapsed
            self.stats_logger.stats.file_stats[str(file_path)] = dict(local_stats.file_stats[str(file_path)])

        # build metadata
        metadata: Dict[str, Any] = {"splits": {}}
        for split in ("train", "validation", "test"):
            metadata["splits"][split] = {
                "shards": writers[split].shard_metadata,
                "n_trajectories": split_counts[split],
                "total_samples": split_counts[split],
            }

        # attach per-file stats so parent can merge when running in parallel
        metadata["file_stats"] = asdict(local_stats)
        return metadata

    # -------------------------------------
    # Streaming time stats (two-pass method)
    # -------------------------------------
    def collect_time_stats(self, file_paths: List[Path]) -> Dict[str, float]:
        """
        Compute tau0 (5th percentile of raw times > 1e-10) and both:
        - tau min/max for 'time-norm' (paper)
        - raw t min/max for 'log-min-max'
        via streaming two-pass histogram.
        """
        if not file_paths:
            raise ValueError("No files provided")

        # Pass 1: global raw min/max and count
        global_min = float("inf")
        global_max = float("-inf")
        total_count = 0

        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var not in grp:
                        continue
                    ds = grp[self.time_var]
                    for start in range(0, ds.shape[0], self.chunk_size):
                        end = min(start + self.chunk_size, ds.shape[0])
                        chunk = ds[start:end]
                        # strictly positive times for log/percentile
                        chunk = chunk[chunk > 1e-10]
                        if chunk.size == 0:
                            continue
                        total_count += int(chunk.size)
                        cmin, cmax = float(chunk.min()), float(chunk.max())
                        if cmin < global_min:
                            global_min = cmin
                        if cmax > global_max:
                            global_max = cmax

        if not math.isfinite(global_min) or not math.isfinite(global_max) or total_count == 0:
            raise ValueError("No valid time data found")

        if not (global_max > global_min):
            global_max = global_min * (1.0 + 1e-12)

        # Pass 2: histogram for tau0 percentile
        num_bins = int(self.proc_cfg.get("time_hist_bins", 4096))
        edges = np.linspace(global_min, global_max, num_bins + 1, dtype=np.float64)
        hist = np.zeros(num_bins, dtype=np.int64)

        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]
                    if self.time_var not in grp:
                        continue
                    ds = grp[self.time_var]
                    for start in range(0, ds.shape[0], self.chunk_size):
                        end = min(start + self.chunk_size, ds.shape[0])
                        chunk = ds[start:end]
                        chunk = chunk[chunk > 1e-10]
                        if chunk.size == 0:
                            continue
                        h, _ = np.histogram(chunk, bins=edges)
                        hist += h

        target_rank = max(0, int(0.05 * (hist.sum())))
        cumsum = np.cumsum(hist)
        bin_idx = int(np.searchsorted(cumsum, target_rank, side="left"))
        bin_idx = max(0, min(bin_idx, num_bins - 1))
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        prev_cum = 0 if bin_idx == 0 else int(cumsum[bin_idx - 1])
        in_bin_rank = max(0, target_rank - prev_cum)
        bin_count = int(hist[bin_idx]) if hist[bin_idx] > 0 else 1
        frac = min(1.0, in_bin_rank / bin_count)
        tau0 = float(left + (right - left) * frac)

        # Paper time-norm bounds in tau
        tau_min = float(np.log(1.0 + global_min / tau0))
        tau_max = float(np.log(1.0 + global_max / tau0))

        # Which time method are we configured to use?
        time_method = (
            self.config.get("normalization", {})
                    .get("methods", {})
                    .get(self.time_var, "log-min-max")
        )

        return {
            "tau0": tau0,
            "tmin": tau_min,         # tau-space min (for time-norm)
            "tmax": tau_max,         # tau-space max (for time-norm)
            "tmin_raw": global_min,  # raw time min (>0) for log-min-max
            "tmax_raw": global_max,  # raw time max
            "time_transform": time_method,
        }


def _process_one_file_worker(
    file_path_str: str,
    output_dir_str: str,
    config_json: str,
    norm_stats_json: Optional[str],
) -> Dict[str, Any]:
    """
    Top-level worker for ProcessPoolExecutor. Avoids pickling bound methods / loggers.
    Returns per-file metadata with embedded per-file stats.
    """
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)
    config = json.loads(config_json)
    norm_stats = json.loads(norm_stats_json) if norm_stats_json is not None else None

    # Local stats logger solely for this process
    stats_logger = DataStatisticsLogger(output_dir)
    processor = CorePreprocessor(config, norm_stats, stats_logger)
    meta = processor.process_file_for_sequence_shards(file_path, output_dir)

    # Attach a compact stats dict for parent to merge
    meta["_worker_stats"] = asdict(stats_logger.stats)
    return meta


class DataPreprocessor:
    """Main preprocessor for sequence mode with statistics tracking."""
    def __init__(self, raw_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.raw_files = sorted(raw_files)
        self.output_dir = output_dir
        self.config = config
        self.sequence_mode = True
        self.processed_dir = output_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.stats_logger = DataStatisticsLogger(output_dir)

    # -------------------------------
    # Vectorized Welford (chunk-wise)
    # -------------------------------
    @staticmethod
    def _welford_merge(
        acc_count: int, acc_mean: float, acc_m2: float, chunk: np.ndarray
    ) -> Tuple[int, float, float, float, float]:
        """Merge a chunk into (count, mean, m2, min, max) using Chan-Welford."""
        d = chunk[np.isfinite(chunk)]
        if d.size == 0:
            return acc_count, acc_mean, acc_m2, float("inf"), float("-inf")

        n1 = int(d.size)
        c_min = float(d.min())
        c_max = float(d.max())
        c_mean = float(d.mean())
        c_m2 = float(((d - c_mean) ** 2).sum())

        if acc_count == 0:
            return n1, c_mean, c_m2, c_min, c_max

        n0 = acc_count
        delta = c_mean - acc_mean
        n = n0 + n1
        mean = acc_mean + delta * (n1 / n)
        m2 = acc_m2 + c_m2 + (delta * delta) * (n0 * n1 / n)
        return n, mean, m2, c_min, c_max

    def _collect_normalization_stats(self) -> Dict[str, Any]:
        """
        Collect normalization statistics for species/globals using a config-driven approach.
        Uses chunk-wise vectorized updates (no Python per-element loops).
        """
        self.logger.info("Collecting normalization statistics based on config...")

        stats: Dict[str, Any] = {"per_key_stats": {}, "normalization_methods": {}}
        accumulators: Dict[str, Dict[str, float]] = {}

        norm_cfg = self.config["normalization"]
        default_method = norm_cfg.get("default_method", "log-standard")
        methods_override = norm_cfg.get("methods", {})
        all_vars = self.config["data"]["species_variables"] + self.config["data"]["global_variables"]

        # init accumulators
        for var in all_vars:
            accumulators[var] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                # linear-domain min/max
                "lin_min": float("inf"),
                "lin_max": float("-inf"),
                # log-domain min/max
                "log_min": float("inf"),
                "log_max": float("-inf"),
            }

        chunk_size = int(self.config["preprocessing"].get("hdf5_chunk_size", 10000))

        # iterate files/groups; stream species datasets; globals from attrs
        for file_path in self.raw_files:
            with h5py.File(file_path, "r") as f:
                for gname in f.keys():
                    grp = f[gname]

                    for var in self.config["data"]["species_variables"]:
                        if var not in grp:
                            continue
                        ds = grp[var]
                        var_method = methods_override.get(var, default_method)

                        # stream over time axis
                        for start in range(0, ds.shape[0], chunk_size):
                            end = min(start + chunk_size, ds.shape[0])

                            arr_raw = ds[start:end]
                            # keep the summary in RAW space only
                            if self.stats_logger:
                                self.stats_logger.update_species_range(var, arr_raw)

                            # use a separate array for normalization stats
                            arr = arr_raw
                            if "log" in var_method:
                                eps = norm_cfg.get("epsilon", 1e-30)
                                arr = np.log10(np.maximum(arr_raw, eps))

                            a = accumulators[var]
                            n, m, m2, cmin, cmax = self._welford_merge(a["count"], a["mean"], a["m2"], arr)
                            a["count"], a["mean"], a["m2"] = n, m, m2

                            # Update linear-domain min/max from raw values
                            finite_lin = arr_raw[np.isfinite(arr_raw)]
                            if finite_lin.size:
                                lin_min = float(finite_lin.min())
                                lin_max = float(finite_lin.max())
                                a["lin_min"] = min(a["lin_min"], lin_min)
                                a["lin_max"] = max(a["lin_max"], lin_max)

                            # Update log-domain min/max from log10(raw)
                            eps = float(norm_cfg.get("epsilon", 1e-30))
                            raw_pos = np.maximum(arr_raw, eps)
                            finite_log = raw_pos[np.isfinite(raw_pos)]
                            if finite_log.size:
                                log_vals = np.log10(finite_log)
                                log_min = float(log_vals.min())
                                log_max = float(log_vals.max())
                                a["log_min"] = min(a["log_min"], log_min)
                                a["log_max"] = max(a["log_max"], log_max)

                        for var in self.config["data"]["global_variables"]:
                            if var in grp.attrs:
                                value = float(grp.attrs[var])
                                var_method = methods_override.get(var, default_method)

                                # 1) Update mean/std in the domain required by the method
                                a = accumulators[var]
                                if "standard" in var_method:  # 'standard' or 'log-standard'
                                    v = (math.log10(max(value, norm_cfg.get("epsilon", 1e-30)))
                                        if "log" in var_method else value)
                                    n, m, m2, cmin, cmax = self._welford_merge(
                                        a["count"], a["mean"], a["m2"],
                                        np.array([v], dtype=np.float64),
                                    )
                                    a["count"], a["mean"], a["m2"] = n, m, m2

                                # 2) Always track min/max in BOTH domains
                                if math.isfinite(value):
                                    a["lin_min"] = min(a["lin_min"], value)
                                    a["lin_max"] = max(a["lin_max"], value)
                                    eps = float(norm_cfg.get("epsilon", 1e-30))
                                    v_log = math.log10(max(value, eps))
                                    a["log_min"] = min(a["log_min"], v_log)
                                    a["log_max"] = max(a["log_max"], v_log)

                                if self.stats_logger:
                                    self.stats_logger.update_global_range(var, value)

        # finalize
        for var, a in accumulators.items():
            if a["count"] == 0:
                continue
            var_method = methods_override.get(var, default_method)
            stats["normalization_methods"][var] = var_method

            variance = a["m2"] / (a["count"] - 1) if a["count"] > 1 else 0.0
            std = float(np.sqrt(max(variance, 0.0)))
            min_std = float(norm_cfg.get("min_std", 1e-10))
            block = {"method": var_method, "min": float(a["min"]), "max": float(a["max"])}

            if "standard" in var_method:
                key_prefix = "log_" if "log" in var_method else ""
                block[f"{key_prefix}mean"] = float(a["mean"])
                block[f"{key_prefix}std"] = float(max(std, min_std))

            stats["per_key_stats"][var] = block

        time_var = self.config["data"]["time_variable"]
        stats["normalization_methods"][time_var] = (
            self.config.get("normalization", {})
                .get("methods", {})
                .get(time_var, "log-min-max")
        )

        self.logger.info("Normalization statistics collection complete.")
        return stats

    # ----------------------------------------
    # End-to-end preprocessing orchestration
    # ----------------------------------------
    def process_to_npy_shards(self) -> None:
        """Process data in sequence mode for LiLaN (with optional per-file parallelism)."""
        self.logger.info("Processing data in SEQUENCE MODE for LiLaN multi-time supervision")

        # streaming time stats
        processor = CorePreprocessor(self.config, stats_logger=self.stats_logger)
        time_stats = processor.collect_time_stats(self.raw_files)

        # Choose fixed grid bounds: config override or global raw min/max
        cfg_range = self.config.get("data", {}).get("fixed_time_range")
        if cfg_range is not None:
            grid_min = float(cfg_range.get("min"))
            grid_max = float(cfg_range.get("max"))
        else:
            grid_min = float(time_stats["tmin_raw"])
            grid_max = float(time_stats["tmax_raw"])

        # normalization stats
        norm_stats = self._collect_normalization_stats()
        norm_stats["time_normalization"] = time_stats
        # Save the fixed grid metadata so workers/serial path can reconstruct it identically
        norm_stats["fixed_time_grid"] = {
            "min": grid_min,
            "max": grid_max,
            "M": int(self.config["data"]["M_per_sample"]),
            "endpoint": True,
            "spacing": "log",
        }
        # NEW: explicitly record variables that are already log10-transformed in shards
        norm_stats["already_logged_vars"] = list(self.config["data"]["species_variables"])
        save_json(norm_stats, self.output_dir / "normalization.json")

        # per-file fan-out
        all_metadata = {
            "splits": {
                "train": {"shards": [], "n_trajectories": 0},
                "validation": {"shards": [], "n_trajectories": 0},
                "test": {"shards": [], "n_trajectories": 0},
            }
        }

        num_workers = int(self.config.get("preprocessing", {}).get("num_workers", 0))
        if num_workers > 0:
            self.logger.info(f"Parallel preprocessing with {num_workers} workers...")
            cfg_json = json.dumps(self.config)
            norm_json = json.dumps(norm_stats)
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as ex:
                futures = [
                    ex.submit(
                        _process_one_file_worker,
                        str(file_path),
                        str(self.processed_dir),
                        cfg_json,
                        norm_json,
                    )
                    for file_path in self.raw_files
                ]
                for fut in as_completed(futures):
                    metadata = fut.result()
                    self._merge_metadata(all_metadata, metadata)
                    # merge worker stats into master logger
                    if "_worker_stats" in metadata:
                        worker_stats = DataStatistics(**metadata["_worker_stats"])
                        self.stats_logger.stats.merge_inplace(worker_stats)
        else:
            # serial path
            for file_path in self.raw_files:
                self.logger.info(f"Processing file: {file_path}")
                cproc = CorePreprocessor(self.config, norm_stats, self.stats_logger)
                metadata = cproc.process_file_for_sequence_shards(file_path, self.processed_dir)
                self._merge_metadata(all_metadata, metadata)

        # index / summaries
        shard_index = {
            "sequence_mode": True,
            "M_per_sample": self.config["data"]["M_per_sample"],
            "n_input_species": len(self.config["data"]["species_variables"]),
            "n_target_species": len(self.config["data"].get("target_species_variables", self.config["data"]["species_variables"])),
            "n_globals": len(self.config["data"]["global_variables"]),
            "compression": "npz",
            "splits": all_metadata["splits"],
            "time_normalization": time_stats,
            "time_grid": {
                "type": "logspace",
                "min": grid_min,
                "max": grid_max,
                "M": int(self.config["data"]["M_per_sample"]),
                "endpoint": True,
            },
            "already_logged_vars": list(self.config["data"]["species_variables"]),
        }
        save_json(shard_index, self.output_dir / "shard_index.json")

        # save stats
        self.stats_logger.save_summary()

        self.logger.info(
            "Sequence mode preprocessing complete. "
            f"Train: {all_metadata['splits']['train']['n_trajectories']} trajectories, "
            f"Val: {all_metadata['splits']['validation']['n_trajectories']}, "
            f"Test: {all_metadata['splits']['test']['n_trajectories']}"
        )

    @staticmethod
    def _merge_metadata(all_meta: Dict[str, Any], meta: Dict[str, Any]) -> None:
        for split in ("train", "validation", "test"):
            all_meta["splits"][split]["shards"].extend(meta["splits"][split]["shards"])
            all_meta["splits"][split]["n_trajectories"] += meta["splits"][split]["n_trajectories"]