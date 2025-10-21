#!/usr/bin/env python3
"""
Flow-map DeepONet Dataset with Log-Δt Sampling (FAST VERSION)
==============================================================
Optimized to skip expensive scanning when dataset structure is known.
FIXED: Corrected initialization order and added safe defaults for all attributes.
"""

from __future__ import annotations

import json
import logging
import math
import time
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from normalizer import NormalizationHelper
from utils import load_json


# --------------------------------- Utilities ---------------------------------

def format_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def load_shard_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load arrays from NPZ shard file."""
    with np.load(path, allow_pickle=False, mmap_mode="r") as npz:
        return npz["x0"], npz["globals"], npz["t_vec"], npz["y_mat"]


# ------------------------------- Main Dataset --------------------------------

class FlowMapPairsDataset(Dataset):
    """
    Flow-map dataset with FAST loading - skips scanning when structure is known.
    """

    def __init__(
            self,
            processed_root: Path | str,
            split: str,
            config: dict,
            *,
            pairs_per_traj: int = 32,
            min_steps: int = 1,
            max_steps: int = 64,
            preload_to_gpu: bool = False,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            seed: int = 42,
            log_every_files: int = 1000,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.root = Path(processed_root)
        self.split = str(split)
        self.cfg = config
        self.base_seed = int(seed)
        self.log_every_files = int(log_every_files)

        # Dataset configuration
        dcfg = self.cfg.get("dataset", {})
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", True))
        self.K = int(dcfg.get("times_per_anchor", 4)) if self.multi_time else 1
        self.share_times_across_batch = bool(dcfg.get("share_times_across_batch", True))
        self.log_dt_sampling = bool(dcfg.get("log_dt_sampling", True))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))

        # Grid settings
        self.uniform_grid_rtol = float(dcfg.get("uniform_grid_rtol", 1e-6))
        self.uniform_grid_atol = float(dcfg.get("uniform_grid_atol", 1e-12))
        self.snap_shared_grid = bool(dcfg.get("snap_shared_grid", True))
        self.assume_shared_grid = bool(dcfg.get("assume_shared_grid", True))  # Default to fast

        # FAST MODE FLAGS
        self.skip_scan = bool(dcfg.get("skip_scan", False))
        self.skip_validate_grids = bool(dcfg.get("skip_validate_grids", False))

        # Load normalization manifest
        norm_path = self.root / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")
        manifest = load_json(norm_path)

        # Device & dtype
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype

        # Normalization helper
        self.norm = NormalizationHelper(manifest, device=self._stage_device)
        self.epsilon = float(getattr(self.norm, "epsilon", 1e-30))

        # FIXED: Get species variables BEFORE any scanning/metadata operations
        data_cfg = self.cfg.get("data", {})
        self.species_vars = list(data_cfg.get("species_variables", []))
        if not self.species_vars:
            raise KeyError("config.data.species_variables must be set")

        # FIXED: Initialize safe defaults for attributes checked later
        self.dt_table = None
        self._has_shared_uniform_grid = False
        self.shared_time_grid = None
        self._dt_shared = None
        self.dt_physical = 1.0

        # Discover shards
        self.files = self._discover_shards()

        # FAST PATH: Skip scanning entirely
        if self.skip_scan:
            # Try to load dimensions from shard_index.json or preprocessing_summary.json
            scan = self._fast_load_metadata()
            self.logger.info(f"[{self.split}] FAST MODE: Skipped scanning, loaded metadata directly")
        else:
            scan = self._scan_shards()

        self.N = int(scan["total_trajectories"])
        self.T = int(scan["time_length"])
        self.S = int(scan["state_dim"])
        self.G_dim = int(scan["global_dim"])

        # Validate species dimensions
        if len(self.species_vars) != self.S:
            raise RuntimeError(f"Species count mismatch: {self.S} vs {len(self.species_vars)}")

        # Grid setup
        self.has_shared_grid = bool(scan["has_shared_grid"])
        shared_grid_ref = scan["shared_grid"]

        # Step bounds
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps) if max_steps else self.K
        self._validate_step_bounds()

        # Length
        self.pairs_per_traj = int(pairs_per_traj)
        self._length = int(self.N * self.pairs_per_traj)

        # Allocate buffers
        self._allocate_buffers()

        # Time grids
        if self.has_shared_grid:
            self.shared_time_grid = torch.from_numpy(
                shared_grid_ref.astype(np.float64, copy=False)
            ).to(self._stage_device)
            self.times = self.shared_time_grid.unsqueeze(0).expand(self.N, -1)
            self.time_grid_per_row = None
        else:
            self.shared_time_grid = None
            self.time_grid_per_row = torch.empty((self.N, self.T), device=self._stage_device, dtype=torch.float64)
            self.times = self.time_grid_per_row

        # Load and normalize data
        self._load_and_stage_data()

        # FAST PATH: Skip grid validation if requested
        if self.skip_validate_grids:
            self.logger.info(f"[{self.split}] FAST MODE: Skipped time grid validation")
            self._has_shared_uniform_grid = self.has_shared_grid
            if self.has_shared_grid and shared_grid_ref is not None:
                dts = np.diff(shared_grid_ref)
                self.dt_physical = float(np.median(dts))
            else:
                self.dt_physical = 1.0
        else:
            self._check_time_grids()

        # Auto-create dt stats if missing
        self._ensure_dt_stats(manifest, norm_path)

        # Precompute dt lookup table
        if self.precompute_dt_table and self._has_shared_uniform_grid:
            self._precompute_dt_table()

        # Initialize epoch
        self.set_epoch(0)
        self._report_memory_usage()

    def _fast_load_metadata(self) -> dict:
        """Fast metadata loading without scanning shards. Fixed to not depend on self.species_vars."""
        # Try shard_index.json first
        shard_index_path = self.root / "shard_index.json"
        if shard_index_path.exists():
            with open(shard_index_path, 'r') as f:
                index = json.load(f)

            # Extract dimensions from shard_index
            if self.split in index.get("splits", {}):
                split_info = index["splits"][self.split]
                N = split_info.get("n_trajectories", 0)
            else:
                # Count trajectories from file count * trajectories_per_shard
                N = len(self.files) * self.cfg.get("preprocessing", {}).get("trajectories_per_shard", 100)

            T = index.get("M_per_sample", 100)
            # Use already-loaded species_vars since we moved that initialization earlier
            S = index.get("n_input_species", len(self.species_vars))
            G = index.get("n_globals", 2)
        else:
            # Fallback to preprocessing_summary.json
            summary_path = self.root / "preprocessing_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                valid_traj = summary.get("valid_trajectories", {})
                N = valid_traj.get(self.split, len(self.files) * 100)
                T = summary.get("time_grid_len", 100)
                S = len(self.species_vars)
                G = len(summary.get("global_variables", ["P", "T"]))
            else:
                # Ultimate fallback - reasonable defaults
                self.logger.warning(f"[{self.split}] No metadata files found, using defaults")
                N = len(self.files) * 100  # Assume 100 trajectories per file
                T = 100  # Common time grid length
                S = len(self.species_vars)
                G = 2  # Common for P, T

        # For shared grid, load from first shard (minimal overhead)
        shared_grid_ref = None
        if self.assume_shared_grid and len(self.files) > 0:
            _, _, t_np, _ = load_shard_arrays(Path(self.files[0]))
            if t_np.ndim == 1:
                shared_grid_ref = t_np.astype(np.float64)
            else:
                shared_grid_ref = t_np[0].astype(np.float64)

        return {
            "total_trajectories": N,
            "time_length": T,
            "state_dim": S,
            "global_dim": G,
            "has_shared_grid": self.assume_shared_grid,
            "shared_grid": shared_grid_ref,
            "bytes_on_disk": 0  # Skip computing
        }

    def _scan_shards(self) -> dict:
        """Original scanning method - only used if skip_scan=False."""
        N_total = 0
        T_global = None
        S_global = None
        G_global = None
        bytes_on_disk = 0
        shared_grid_possible = True
        shared_grid_ref = None

        start = time.perf_counter()
        for idx, f in enumerate(self.files, 1):
            path = Path(f)
            _, g_np, t_np, y_np = load_shard_arrays(path)

            N_shard, T_shard, S_shard = y_np.shape
            G_shard = g_np.shape[1]

            # Handle time vector
            if t_np.ndim == 1:
                current_grid = np.asarray(t_np, dtype=np.float64)
            elif t_np.ndim == 2:
                if not np.all(t_np == t_np[0]):
                    shared_grid_possible = False
                    current_grid = None
                else:
                    current_grid = np.asarray(t_np[0], dtype=np.float64)
            else:
                raise RuntimeError(f"Invalid t_vec ndim={t_np.ndim}")

            # Check grid consistency
            if shared_grid_possible and current_grid is not None:
                if shared_grid_ref is None:
                    shared_grid_ref = current_grid.copy()
                elif not self.assume_shared_grid:
                    if not np.allclose(shared_grid_ref, current_grid, rtol=1e-12, atol=1e-15):
                        shared_grid_possible = False

            N_total += N_shard

            if T_global is None:
                T_global, S_global, G_global = T_shard, S_shard, G_shard
            else:
                if T_global != T_shard or S_global != S_shard or G_global != G_shard:
                    raise RuntimeError("Heterogeneous shard dimensions")

            bytes_on_disk += path.stat().st_size

            if idx % self.log_every_files == 0 or idx == len(self.files):
                elapsed = time.perf_counter() - start
                mode = "fast" if self.assume_shared_grid else "full"
                print(f"[{self.split}] Scanned {idx}/{len(self.files)} shards ({mode} mode) "
                      f"({format_bytes(bytes_on_disk)}) in {elapsed:.1f}s")

        return {
            "total_trajectories": N_total,
            "time_length": T_global,
            "state_dim": S_global,
            "global_dim": G_global,
            "has_shared_grid": shared_grid_possible and (shared_grid_ref is not None),
            "shared_grid": shared_grid_ref,
            "bytes_on_disk": bytes_on_disk
        }

    def _check_time_grids(self) -> None:
        """Validate time grids - skip with skip_validate_grids flag."""
        rtol = self.uniform_grid_rtol
        atol = self.uniform_grid_atol
        snap = self.snap_shared_grid

        if not hasattr(self, "times"):
            raise ValueError("Dataset missing `self.times` tensor.")

        t = self.times
        if not torch.is_tensor(t) or t.ndim != 2:
            raise ValueError(f"`self.times` must be a [N,T] tensor")
        if t.shape[1] < 2:
            raise ValueError("Time grid must have at least 2 points per trajectory.")

        t64 = t.to(torch.float64)

        # Check strictly increasing (required)
        if not torch.all(t64[:, 1:] > t64[:, :-1]):
            bad = torch.where(~(t64[:, 1:] > t64[:, :-1]))
            i = int(bad[0][0].item())
            j = int(bad[1][0].item())
            raise ValueError(
                f"Time grid must be strictly increasing: traj={i}, step={j}, "
                f"t[{j}]={t64[i, j].item():.6g} >= t[{j + 1}]={t64[i, j + 1].item():.6g}"
            )

        # If not sharing across batch, we're done
        if not self.share_times_across_batch:
            self._dt_shared = None
            self._has_shared_uniform_grid = False
            self.shared_time_grid = None
            self.dt_physical = float((t64[:, 1:] - t64[:, :-1]).median().item())
            self.logger.info(f"Using per-trajectory time grids (median dt={self.dt_physical:.3e})")
            return

        # Check for shared grid across trajectories
        t_ref = t64[0]  # [T]
        same_as_ref = (t64 - t_ref).abs() <= (atol + rtol * t_ref.abs())

        if not bool(same_as_ref.all()):
            if snap:
                self.times = t_ref.to(self.times.dtype).expand_as(self.times).contiguous()
                t64 = self.times.to(torch.float64)
                self.logger.info("Snapped all trajectories to shared reference grid")
            else:
                max_abs = float((t64 - t_ref).abs().max().item())
                max_rel = float(((t64 - t_ref).abs() / t_ref.abs().clamp_min(1e-30)).max().item())
                raise ValueError(
                    f"Trajectories have different time grids beyond tolerance "
                    f"(rtol={rtol:.1e}, atol={atol:.1e}). "
                    f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
                )

        # We have a shared grid (possibly non-uniform)
        self.shared_time_grid = t_ref.clone().contiguous()  # [T], float64
        dts = t_ref[1:] - t_ref[:-1]
        self.dt_physical = float(dts.median().item())

        # Check if it's approximately uniform
        dt_min = dts.min().item()
        dt_max = dts.max().item()
        if (dt_max - dt_min) / dt_min < 0.01:  # Less than 1% variation
            self.logger.info(f"Shared uniform grid detected (dt={self.dt_physical:.3e})")
        else:
            self.logger.info(
                f"Shared NON-UNIFORM grid detected "
                f"(dt_min={dt_min:.3e}, dt_max={dt_max:.3e}, ratio={dt_max / dt_min:.1f})"
            )

        self._dt_shared = None
        self._has_shared_uniform_grid = True

    # ========== ALL OTHER METHODS REMAIN EXACTLY THE SAME ==========

    def _ensure_dt_stats(self, manifest: dict, norm_path: Path):
        """Create/update dt stats from actual time grid if missing."""
        if "dt" in manifest:
            return

        if not getattr(self, "_has_shared_uniform_grid", False) or self.shared_time_grid is None:
            self.logger.warning("Cannot auto-create dt stats without a shared time grid")
            return

        # Compute min/max physical dt across allowed step spans
        t_ref = self.shared_time_grid.cpu().numpy().astype(np.float64)
        T = t_ref.shape[0]

        dt_min_phys = np.inf
        dt_max_phys = 0.0

        for s in range(self.min_steps, min(self.max_steps + 1, T)):
            dts = t_ref[s:] - t_ref[:-s]
            if dts.size > 0:
                dt_min_phys = min(dt_min_phys, float(np.min(dts)))
                dt_max_phys = max(dt_max_phys, float(np.max(dts)))

        eps = float(manifest.get("epsilon", 1e-30))
        dt_min_phys = max(dt_min_phys, eps)
        dt_max_phys = max(dt_max_phys, dt_min_phys)

        manifest["dt"] = {
            "method": "log-min-max",
            "log_min": float(np.log10(dt_min_phys)),
            "log_max": float(np.log10(dt_max_phys)),
            "epsilon": eps
        }

        # Persist and refresh
        with open(norm_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        self.logger.info(
            f"Created dt stats from grid: "
            f"min_phys={dt_min_phys:.3e}, max_phys={dt_max_phys:.3e}, "
            f"log_range=[{manifest['dt']['log_min']:.2f}, {manifest['dt']['log_max']:.2f}]"
        )

    def steps_to_dt_phys(
            self,
            i_idx: torch.Tensor,
            j_idx: torch.Tensor,
            traj_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute physical Δt from actual timestamps."""
        i_idx = torch.as_tensor(i_idx, device=self.times.device)
        j_idx = torch.as_tensor(j_idx, device=self.times.device)

        if self._has_shared_uniform_grid and self.shared_time_grid is not None:
            t_ref = self.shared_time_grid.to(torch.float64)
            dt_phys = t_ref[j_idx.long()] - t_ref[i_idx.long()]
            return dt_phys.to(torch.float32)

        if traj_idx is None:
            raise ValueError("traj_idx required for per-trajectory grids")
        traj_idx = torch.as_tensor(traj_idx, device=self.times.device).long()

        t = self.times.to(torch.float64)
        t_i = t[traj_idx, i_idx.long()]
        t_j = t[traj_idx, j_idx.long()]
        return (t_j - t_i).to(torch.float32)

    def _precompute_dt_table(self):
        """Precompute normalized dt lookup table."""
        if self.shared_time_grid is None:
            raise RuntimeError("Cannot precompute dt table without shared grid")

        T = self.T
        t_ref = self.shared_time_grid.to(torch.float64)

        i_grid, j_grid = torch.meshgrid(
            torch.arange(T, device=self.device),
            torch.arange(T, device=self.device),
            indexing='ij'
        )

        dt_phys = t_ref[j_grid] - t_ref[i_grid]
        dt_phys = torch.clamp(dt_phys, min=self.epsilon).to(torch.float32)

        dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32)
        self.dt_table = dt_norm.contiguous()

    def _discover_shards(self) -> list[str]:
        pattern = str(self.root / self.split / "*.npz")
        files = sorted(glob(pattern))
        if not files:
            raise RuntimeError(f"No NPZ shards found at {pattern}")
        return files

    def _validate_step_bounds(self):
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1")
        if self.max_steps < self.min_steps:
            raise ValueError(f"max_steps must be >= min_steps")
        if self.max_steps > self.T - 1:
            raise ValueError(f"max_steps exceeds T-1")

    def _allocate_buffers(self):
        buf_dtype = self._runtime_dtype if self._stage_device.type == "cuda" else torch.float32
        # Keep globals in float32 for stability of K(g) even on CUDA
        self.G = torch.empty((self.N, self.G_dim), device=self._stage_device, dtype=torch.float32)
        # States/targets can follow the runtime dtype
        self.Y = torch.empty((self.N, self.T, self.S), device=self._stage_device, dtype=buf_dtype)

    def _load_and_stage_data(self):
        """Load and normalize data to device."""
        write_ptr = 0
        start = time.perf_counter()

        for idx, f in enumerate(self.files, 1):
            path = Path(f)
            _, g_np, t_np, y_np = load_shard_arrays(path)
            n = y_np.shape[0]
            sl = slice(write_ptr, write_ptr + n)

            # Normalize globals
            g = torch.from_numpy(g_np.astype(np.float32, copy=False)).to(self._stage_device)
            global_vars = self.cfg["data"].get("global_variables", [])
            if global_vars:
                g = self.norm.normalize(g, global_vars)
            # Write G as float32 (stable for Koopman K(g))
            self.G[sl] = g.to(self._stage_device, dtype=torch.float32, non_blocking=True)

            # Normalize species
            y = torch.from_numpy(y_np)
            y_norm = self.norm.normalize(y, self.species_vars).reshape(n, self.T, self.S)
            self.Y[sl] = y_norm.to(self._stage_device, dtype=self.Y.dtype, non_blocking=True)

            # Store time grids if not shared
            if not self.has_shared_grid:
                if t_np.ndim == 1:
                    t = torch.from_numpy(t_np.astype(np.float64, copy=False)).to(self._stage_device)
                    self.time_grid_per_row[sl] = t.view(1, self.T).expand(n, self.T)
                else:
                    self.time_grid_per_row[sl] = torch.from_numpy(
                        t_np.astype(np.float64, copy=False)
                    ).to(self._stage_device)

            write_ptr += n

            if idx % self.log_every_files == 0 or idx == len(self.files):
                elapsed = time.perf_counter() - start
                dev = "GPU" if self._stage_device.type == "cuda" else "CPU"
                print(f"[{self.split}] Loaded {idx}/{len(self.files)} shards to {dev} in {elapsed:.1f}s")

    def _report_memory_usage(self):
        if self._stage_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            print(f"[{self.split}] GPU: {used / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f} GiB")

        print(f"[{self.split}] Ready: N={self.N}, T={self.T}, S={self.S}, K={self.K}, "
              f"log_dt_sampling={self.log_dt_sampling}")

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic sampling."""
        self.epoch = int(epoch)
        N = self.N

        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.base_seed + epoch)

        traj = torch.randint(0, N, (self._length,), device=self.device, generator=gen)
        max_anchor = self.T - 1 - self.min_steps
        anchors = torch.randint(0, max_anchor + 1, (self._length,), device=self.device, generator=gen)

        self.index_map = torch.stack([traj, anchors], dim=1).to(torch.long)
        self._torch_gen = gen

    def _sample_log_dt_offsets(self, B: int, K: int, generator=None) -> torch.Tensor:
        """Sample offsets with log-uniform distribution."""
        log_min = math.log(float(self.min_steps))
        log_max = math.log(float(self.max_steps))

        log_offsets = torch.rand(B, K, device=self.device, generator=generator)
        log_offsets = log_min + log_offsets * (log_max - log_min)

        offsets = torch.exp(log_offsets).round().clamp(self.min_steps, self.max_steps).long()
        return offsets

    def _sample_uniform_offsets(self, B: int, K: int, generator=None) -> torch.Tensor:
        """Sample uniform offsets."""
        return torch.randint(
            self.min_steps, self.max_steps + 1,
            (B, K), device=self.device, generator=generator
        )

    @torch.no_grad()
    def _gather_batch(self, idx_list: torch.LongTensor):
        """Assemble batch with proper dt computation for non-uniform grids."""
        if not isinstance(idx_list, torch.Tensor):
            idx_list = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        elif idx_list.device != self.device:
            idx_list = idx_list.to(self.device, dtype=torch.long, non_blocking=True)

        pair = self.index_map[idx_list]
        traj = pair[:, 0]
        anchor_i = pair[:, 1]
        B = traj.shape[0]

        if self.share_times_across_batch:
            # Sample K offsets once and share them across the whole batch
            if self.log_dt_sampling:
                base_off = self._sample_log_dt_offsets(1, self.K, self._torch_gen)  # [1,K]
            else:
                base_off = self._sample_uniform_offsets(1, self.K, self._torch_gen)  # [1,K]
            offsets = base_off.expand(B, self.K)  # [B,K], identical per row
        else:
            # Per-row offsets (slower; many unique Δt)
            if self.log_dt_sampling:
                offsets = self._sample_log_dt_offsets(B, self.K, self._torch_gen)
            else:
                offsets = self._sample_uniform_offsets(B, self.K, self._torch_gen)

        target_j = anchor_i.unsqueeze(1) + offsets
        mask = target_j < self.T
        target_j = torch.clamp(target_j, max=self.T - 1)

        g = self.G[traj]
        y_i = self.Y[traj, anchor_i]

        traj_exp = traj.unsqueeze(1).expand(B, self.K)
        y_j = self.Y[traj_exp, target_j]

        if self.dt_table is not None and self._has_shared_uniform_grid:
            # Use precomputed normalized dt table → already [B,K]
            dt_norm = self.dt_table[anchor_i.unsqueeze(1), target_j]  # [B,K]
        else:
            # Compute physical Δt then normalize → [B,K]
            anchor_i_exp = anchor_i.unsqueeze(1).expand_as(target_j)
            dt_phys = self.steps_to_dt_phys(anchor_i_exp, target_j, traj_exp)
            dt_phys = torch.clamp(dt_phys, min=self.epsilon)
            dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32)  # [B,K]

        aux = {"i": anchor_i, "j": target_j}

        if self.share_times_across_batch:
            return y_i, dt_norm, y_j, g, aux, mask
        else:
            return y_i, dt_norm, y_j, g, aux

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def create_dataloader(
        dataset: FlowMapPairsDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
) -> torch.utils.data.DataLoader:
    """Create DataLoader for FlowMapPairsDataset."""

    on_cuda = dataset._stage_device.type == "cuda"
    if on_cuda:
        num_workers = 0
        pin_memory = False
        shuffle = False

    def _collate(idxs):
        if isinstance(idxs, (list, tuple)):
            idxs = torch.tensor(idxs, dtype=torch.long, device=dataset.device)
        elif isinstance(idxs, torch.Tensor):
            if idxs.device != dataset.device:
                idxs = idxs.to(dataset.device, dtype=torch.long, non_blocking=True)
        return dataset._gather_batch(idxs)

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": _collate
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)

    return torch.utils.data.DataLoader(**kwargs)