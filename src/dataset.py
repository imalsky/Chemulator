#!/usr/bin/env python3
"""
Flow-map DeepONet Dataset
==========================
Dataset for autonomous flow-map prediction with multi-time-per-anchor support.

This dataset implements deterministic sampling for reproducible training using
numpy.random.Generator with fixed seeds. For each anchor (trajectory n, step i),
it samples K strictly-later steps j_k > i, returning tensors suitable for
flow-map prediction.

Key Features:
- Deterministic sampling via numpy.random.Generator with epoch-based seeds
- GPU-resident data staging for high throughput
- Support for both shared and per-trajectory time grids
- Efficient batched tensor gathering with minimal data movement
"""

from __future__ import annotations

import logging
import time
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from normalizer import NormalizationHelper
from utils import load_json


def format_bytes(num_bytes: int | float) -> str:
    """Format byte count as human-readable string."""
    num_bytes = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


def load_shard_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load arrays from NPZ shard file.
    
    Args:
        path: Path to NPZ file
        
    Returns:
        Tuple of (x0, globals, t_vec, y_mat) arrays
        
    Expected shapes:
        x0: [N, S] - initial conditions
        globals: [N, G] - global parameters
        t_vec: [T] or [N, T] - time points
        y_mat: [N, T, S] - trajectory data
    """
    with np.load(path, allow_pickle=False, mmap_mode="r") as npz:
        return npz["x0"], npz["globals"], npz["t_vec"], npz["y_mat"]


class FlowMapPairsDataset(Dataset):
    """
    Dataset for flow-map DeepONet training.
    
    Samples (n, i) anchor points and K strictly-later times j_k for each anchor.
    Uses numpy.random.Generator for deterministic, reproducible sampling.
    
    Output tensors:
        y_i: [B, S] - state at anchor time
        dt_norm: [B, K] - normalized time differences
        y_j: [B, K, S] - states at target times
        g: [B, G] - global parameters
        ij_index: [B, K, 2] - indices for debugging
    """

    def __init__(
        self,
        processed_root: Path | str,
        split: str,
        config: dict,
        pairs_per_traj: int = 64,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        preload_to_gpu: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        log_every_files: int = 20,
    ):
        """
        Initialize dataset.
        
        Args:
            processed_root: Path to preprocessed data directory
            split: One of 'train', 'validation', 'test'
            config: Configuration dictionary
            pairs_per_traj: Number of anchor points per trajectory per epoch
            min_steps: Minimum time steps between anchor and target
            max_steps: Maximum time steps between anchor and target
            preload_to_gpu: Whether to stage all data to GPU memory
            device: Target device for staged data
            dtype: Data type for staged tensors
            seed: Random seed for deterministic sampling
            log_every_files: Logging frequency during loading
        """
        super().__init__()
        self.root = Path(processed_root)
        self.split = str(split)
        self.cfg = config
        self.base_seed = int(seed)
        self.logger = logging.getLogger("dataset")
        self.log_every_files = int(log_every_files)  # <-- ADD THIS LINE

        # Extract dataset configuration
        dataset_cfg = self.cfg.get("dataset", {})
        self.require_dt_stats = bool(dataset_cfg.get("require_dt_stats", True))
        self.precompute_dt_table = bool(dataset_cfg.get("precompute_dt_table", True))

        # Multi-time configuration
        self.multi_time = bool(dataset_cfg.get("multi_time_per_anchor", False))
        self.K = int(dataset_cfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dataset_cfg.get("share_times_across_batch", False))

        # Load normalization manifest
        norm_path = self.root / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")
        manifest = load_json(norm_path)

        # Setup staging device and normalization
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self._runtime_dtype = dtype
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Validate dt statistics if required
        if self.require_dt_stats:
            self._validate_dt_stats(manifest)

        # Discover and scan shards
        self.files = self._discover_shards()
        scan_results = self._scan_shards()
        
        # Unpack scan results
        self.N = scan_results["total_trajectories"]
        self.T = scan_results["time_length"]
        self.S = scan_results["state_dim"]
        self.G_dim = scan_results["global_dim"]
        self.has_shared_grid = scan_results["has_shared_grid"]
        shared_grid_ref = scan_results["shared_grid"]
        
        # Set step bounds
        self.min_steps = min_steps
        self.max_steps = max_steps if max_steps is not None else (self.T - 1)
        self._validate_step_bounds()
        
        self.pairs_per_traj = int(pairs_per_traj)
        self._length = self.N * self.pairs_per_traj

        # Allocate staging buffers
        self._allocate_buffers()
        
        # Setup time grid storage
        if self.has_shared_grid:
            self.shared_time_grid = torch.from_numpy(
                shared_grid_ref.astype(np.float64)
            ).to(self._stage_device)
            self.time_grid_per_row = None
        else:
            self.shared_time_grid = None
            self.time_grid_per_row = torch.empty(
                (self.N, self.T), device=self._stage_device, dtype=torch.float64
            )

        # Load and stage all data
        self._load_and_stage_data()
        
        # Create flattened view for efficient indexing
        self.Y_flat = self.Y.reshape(self.N * self.T, self.S)

        # Precompute dt normalization table if using shared grid
        self.dt_table = None
        if self.has_shared_grid and self.precompute_dt_table:
            self.dt_table = self.norm.make_dt_norm_table(
                time_grid_phys=self.shared_time_grid,
                min_steps=self.min_steps,
                max_steps=self.max_steps
            ).to(device=self._stage_device, dtype=torch.float32)
        elif self.precompute_dt_table and not self.has_shared_grid:
            self.logger.info(f"[{self.split}] Multiple time grids detected; disabling dt table precomputation")

        # Initialize epoch counter and random number generator
        self.epoch = 0
        self.rng = None  # Will be set in set_epoch
        self._initialize_rng()
        
        # Report memory usage
        self._report_memory_usage()

    def _validate_dt_stats(self, manifest: dict) -> None:
        """Validate that dt statistics are present in manifest."""
        dt_spec = manifest.get("dt", None)
        if not isinstance(dt_spec, dict) or "log_min" not in dt_spec or "log_max" not in dt_spec:
            raise RuntimeError(
                "normalization.json must include 'dt' with {'method', 'log_min', 'log_max'}. "
                "Re-run preprocessing to generate centralized dt statistics."
            )
        if str(dt_spec.get("method", "")).lower() != "log-min-max":
            raise RuntimeError("Only 'log-min-max' method is supported for dt normalization")

    def _discover_shards(self) -> list:
        """Discover all NPZ shard files for this split."""
        pattern = str(self.root / self.split / "*.npz")
        files = sorted(glob(pattern))
        if not files:
            raise RuntimeError(f"[{self.split}] No NPZ shards found at {pattern}")
        return files

    def _scan_shards(self) -> dict:
        """
        Scan all shards to determine dimensions and detect shared time grid.
        
        Returns:
            Dictionary with scan results
        """
        N_total = 0
        T_global = None
        S_global = None
        G_global = None
        
        bytes_on_disk = 0
        shared_grid_possible = True
        shared_grid_ref = None
        
        start_time = time.perf_counter()
        
        for idx, filepath in enumerate(self.files, 1):
            path = Path(filepath)
            _, g_np, t_np, y_np = load_shard_arrays(path)
            
            # Validate shapes
            N_shard, T_shard, S_shard = y_np.shape
            G_shard = g_np.shape[1]
            
            # Handle both [T] and [N,T] time formats
            if t_np.ndim == 1:
                if T_shard != t_np.shape[0]:
                    raise RuntimeError(f"[{self.split}] {path.name}: Shape mismatch between y and t")
                current_grid = np.asarray(t_np, dtype=np.float64)
            elif t_np.ndim == 2:
                if T_shard != t_np.shape[1] or N_shard != t_np.shape[0]:
                    raise RuntimeError(f"[{self.split}] {path.name}: Shape mismatch between y and t")
                # Check if all rows identical within shard
                if not np.all(t_np == t_np[0]):
                    shared_grid_possible = False
                    current_grid = None
                else:
                    current_grid = np.asarray(t_np[0], dtype=np.float64)
            else:
                raise RuntimeError(f"[{self.split}] Invalid t_vec dimensions in {path.name}")
            
            # Check for shared grid across shards
            if shared_grid_possible and current_grid is not None:
                if shared_grid_ref is None:
                    shared_grid_ref = current_grid.copy()
                elif not np.array_equal(shared_grid_ref, current_grid):
                    shared_grid_possible = False
            
            # Update totals
            N_total += N_shard
            if T_global is None:
                T_global, S_global, G_global = T_shard, S_shard, G_shard
            else:
                if T_global != T_shard or S_global != S_shard or G_global != G_shard:
                    raise RuntimeError(
                        f"[{self.split}] Heterogeneous shard dimensions detected. "
                        f"All shards must have same T/S/G dimensions."
                    )
            
            # Track disk usage
            try:
                bytes_on_disk += path.stat().st_size
            except Exception:
                pass
            
            # Log progress
            if idx % self.log_every_files == 0 or idx == len(self.files):
                elapsed = time.perf_counter() - start_time
                self.logger.info(
                    f"[{self.split}] Scanned {idx}/{len(self.files)} shards "
                    f"({format_bytes(bytes_on_disk)}) in {elapsed:.1f}s"
                )
        
        if N_total <= 0 or T_global is None:
            raise RuntimeError(f"[{self.split}] No valid data found in shards")
        
        return {
            "total_trajectories": N_total,
            "time_length": T_global,
            "state_dim": S_global,
            "global_dim": G_global,
            "has_shared_grid": shared_grid_possible and shared_grid_ref is not None,
            "shared_grid": shared_grid_ref,
            "bytes_on_disk": bytes_on_disk,
        }

    def _validate_step_bounds(self) -> None:
        """Validate min/max step configuration."""
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {self.min_steps}")
        if self.max_steps < self.min_steps:
            raise ValueError(f"max_steps must be >= min_steps, got {self.max_steps} < {self.min_steps}")
        if self.max_steps > self.T - 1:
            raise ValueError(f"max_steps exceeds trajectory length: {self.max_steps} > {self.T - 1}")

    def _allocate_buffers(self) -> None:
        """Allocate staging buffers for data."""
        self.G = torch.empty(
            (self.N, self.G_dim),
            device=self._stage_device,
            dtype=torch.float32
        )
        # Keep targets in float32 for label precision, even if runtime is bf16/fp16
        self.Y = torch.empty(
            (self.N, self.T, self.S),
            device=self._stage_device,
            dtype=torch.float32
        )

    def _load_and_stage_data(self) -> None:
        """Load all shards and stage to device memory."""
        write_ptr = 0
        start_time = time.perf_counter()

        for idx, filepath in enumerate(self.files, 1):
            path = Path(filepath)
            _, g_np, t_np, y_np = load_shard_arrays(path)
            N_shard = y_np.shape[0]

            slice_range = slice(write_ptr, write_ptr + N_shard)

            # Stage globals
            g_tensor = torch.from_numpy(g_np.astype(np.float32, copy=False))
            self.G[slice_range] = g_tensor.to(
                self._stage_device, dtype=torch.float32, non_blocking=True
            )

            # Normalize and stage trajectories (labels kept in float32)
            y_tensor = torch.from_numpy(y_np)
            target_vars = self.cfg["data"].get(
                "target_species_variables",
                self.cfg["data"]["species_variables"]
            )
            y_normalized = self.norm.normalize(y_tensor, target_vars).reshape(N_shard, self.T, self.S)
            self.Y[slice_range] = y_normalized.to(
                self._stage_device, dtype=torch.float32, non_blocking=True
            )

            # Handle time grids
            if not self.has_shared_grid:
                if t_np.ndim == 1:
                    # Broadcast to all trajectories in shard
                    time_tensor = torch.from_numpy(t_np.astype(np.float64, copy=False))
                    self.time_grid_per_row[slice_range] = time_tensor.to(
                        self._stage_device
                    ).view(1, self.T).expand(N_shard, self.T)
                else:
                    # Per-trajectory time grids
                    self.time_grid_per_row[slice_range] = torch.from_numpy(
                        t_np.astype(np.float64, copy=False)
                    ).to(self._stage_device)

            write_ptr += N_shard

            # Log progress
            if idx % self.log_every_files == 0 or idx == len(self.files):
                elapsed = time.perf_counter() - start_time
                device_type = 'GPU' if self._stage_device.type == 'cuda' else 'CPU'
                self.logger.info(
                    f"[{self.split}] Loaded {idx}/{len(self.files)} shards to {device_type} "
                    f"in {elapsed:.1f}s"
                )

    def _report_memory_usage(self) -> None:
        """Report memory usage statistics."""
        if self._stage_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            self.logger.info(
                f"[{self.split}] Memory usage: "
                f"G={format_bytes(self.G.numel() * self.G.element_size())}, "
                f"Y={format_bytes(self.Y.numel() * self.Y.element_size())} | "
                f"GPU: {used / (1024**3):.1f} / {total / (1024**3):.1f} GiB"
            )
        
        self.logger.info(
            f"[{self.split}] Dataset ready: "
            f"N={self.N}, T={self.T}, S={self.S}, G={self.G_dim} | "
            f"pairs_per_traj={self.pairs_per_traj}, "
            f"steps=[{self.min_steps}, {self.max_steps}] | "
            f"K={self.K}, multi_time={self.multi_time}, "
            f"share_offsets={self.share_offsets_across_batch}"
        )

    def _initialize_rng(self) -> None:
        """Initialize random number generator for current epoch."""
        # Create deterministic seed based on base_seed and epoch
        epoch_seed = self.base_seed + self.epoch * 1000
        self.rng = np.random.Generator(np.random.PCG64(epoch_seed))

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for deterministic sampling.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = int(epoch)
        self._initialize_rng()

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self._length

    def __getitem__(self, idx: int) -> int:
        """Return sample index; actual batch construction happens in _gather_batch."""
        return int(idx)

    def _gather_batch(self, idx_list):
        """
        Construct batch tensors on staging device.
        
        Args:
            idx_list: List or tensor of sample indices
            
        Returns:
            Tuple of (y_i, dt_norm, y_j, g, ij, k_mask) tensors
        """
        device = self.Y.device
        B = len(idx_list)
        K = self.K

        # Convert indices to numpy array for RNG operations
        if isinstance(idx_list, torch.Tensor):
            indices_np = idx_list.cpu().numpy()
        else:
            indices_np = np.array(idx_list)

        # Decode sample indices to trajectory IDs
        n = indices_np // self.pairs_per_traj

        # Sample anchor points (i) using numpy RNG
        i_max = max(0, self.T - 1 - self.min_steps)
        i_np = self.rng.integers(0, i_max + 1, size=B)

        # Convert to torch tensors on device
        n = torch.from_numpy(n).to(device=device, dtype=torch.int64)
        i = torch.from_numpy(i_np).to(device=device, dtype=torch.int64)

        # Compute valid target bounds
        j_lo = i + self.min_steps
        j_hi = torch.minimum(
            torch.full_like(i, self.T - 1),
            i + self.max_steps
        )
        span = torch.clamp(j_hi - j_lo + 1, min=0)

        # Build target indices
        j_out = self._sample_target_indices(B, K, i, j_lo, span)

        # Gather tensors
        y_i, y_j, g = self._gather_states(n, i, j_out, B, K)
        dt_norm = self._compute_dt_norm(n, i, j_out, B, K)

        # Build index tensor for debugging
        ij = torch.stack([
            i.view(B, 1).expand(B, K).to(torch.int32),
            j_out.to(torch.int32)
        ], dim=-1)

        # Valid-sample mask (True where j>i)
        k_mask = (j_out != i.view(B, 1))

        return y_i, dt_norm, y_j, g, ij, k_mask

    def _sample_target_indices(self, B: int, K: int, i: torch.Tensor, 
                               j_lo: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        """
        Sample K target indices for each anchor.
        
        Args:
            B: Batch size
            K: Number of targets per anchor
            i: Anchor indices
            j_lo: Lower bounds for targets
            span: Number of valid target positions
            
        Returns:
            Target indices tensor of shape [B, K]
        """
        device = self.Y.device
        j_out = torch.empty((B, K), dtype=torch.int64, device=device)
        j_out[:] = i.view(B, 1)  # Default to anchor (will be masked in trainer)

        # Convert tensors to numpy for RNG operations
        span_np = span.cpu().numpy()
        j_lo_np = j_lo.cpu().numpy()

        if K == 1:
            # Fast path for single target
            offsets_np = np.zeros(B, dtype=np.int64)
            valid_mask = span_np > 0
            if np.any(valid_mask):
                # Sample one offset per valid anchor
                for b in range(B):
                    if valid_mask[b]:
                        offsets_np[b] = self.rng.integers(0, span_np[b])
            
            j_selected = j_lo_np + offsets_np
            j_out[:, 0] = torch.from_numpy(j_selected).to(device=device, dtype=torch.int64)
            
        else:
            if self.share_offsets_across_batch:
                # Shared permutation across batch
                span_max = int(np.max(span_np)) if B > 0 else 0
                
                if span_max > 0:
                    # Generate shared permutation
                    shared_perm = self.rng.permutation(span_max)
                    
                    for b in range(B):
                        sb = int(span_np[b])
                        used_k = min(K, sb)
                        if used_k > 0:
                            # Use first used_k elements of shared permutation
                            j_selected = j_lo_np[b] + shared_perm[:used_k]
                            j_out[b, :used_k] = torch.from_numpy(j_selected).to(
                                device=device, dtype=torch.int64
                            )
            else:
                # Independent permutations per sample
                for b in range(B):
                    sb = int(span_np[b])
                    if sb <= 0:
                        continue
                    
                    used_k = min(K, sb)
                    # Generate permutation of available offsets
                    offsets = self.rng.permutation(sb)[:used_k]
                    j_selected = j_lo_np[b] + offsets
                    j_out[b, :used_k] = torch.from_numpy(j_selected).to(
                        device=device, dtype=torch.int64
                    )

        return j_out

    def _gather_states(self, n: torch.Tensor, i: torch.Tensor, 
                      j_out: torch.Tensor, B: int, K: int) -> Tuple[torch.Tensor, ...]:
        """Gather state tensors for batch."""
        # Anchor states
        lin_i = n * self.T + i
        y_i = self.Y_flat.index_select(0, lin_i)
        
        # Target states
        lin_j = n.view(B, 1) * self.T + j_out
        y_j = self.Y_flat.index_select(0, lin_j.reshape(-1)).reshape(B, K, self.S)
        
        # Global parameters
        g = self.G.index_select(0, n)
        
        return y_i, y_j, g

    def _compute_dt_norm(self, n: torch.Tensor, i: torch.Tensor,
                        j_out: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """Compute normalized time differences (kept as float32 for precision)."""
        # Mask for padded positions
        pad_mask = (j_out == i.view(B, 1))

        if self.dt_table is not None:
            # Use precomputed table (already float32)
            dt_norm = self.dt_table[i.view(B, 1), j_out]
        else:
            # Compute from physical times
            if self.time_grid_per_row is not None:
                t_i = self.time_grid_per_row[n, i]
                t_j = self.time_grid_per_row[
                    n.view(B, 1).expand(-1, K), j_out
                ]
            else:
                vec = self.shared_time_grid
                t_i = vec.index_select(0, i)
                t_j = vec.index_select(0, j_out.reshape(-1)).view(B, K)

            dt_phys = t_j - t_i.view(B, 1)
            eps = float(self.norm.epsilon)
            dt_phys = torch.where(
                dt_phys > 0, dt_phys,
                torch.as_tensor(eps, dtype=dt_phys.dtype, device=dt_phys.device)
            )
            # Keep dt_norm in float32 for numerical resolution; let AMP cast inside ops
            dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(dtype=torch.float32)

        # Zero out padded positions
        dt_norm = torch.where(pad_mask, torch.zeros_like(dt_norm), dt_norm)
        return dt_norm


def create_dataloader(
    dataset: FlowMapPairsDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for FlowMapPairsDataset.
    
    Note: When dataset tensors are GPU-resident, workers and pin_memory are disabled.
    """
    on_cuda = (dataset.Y.device.type == "cuda")
    
    if on_cuda:
        # Force single-threaded access for GPU-resident data
        num_workers = 0
        pin_memory = False
        shuffle = False

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
        "drop_last": False,
        "collate_fn": lambda idxs: dataset._gather_batch(idxs),
    }
    
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return torch.utils.data.DataLoader(**kwargs)