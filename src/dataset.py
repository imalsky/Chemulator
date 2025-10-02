#!/usr/bin/env python3
"""
Flow-map DeepONet Dataset (Δt trunk input, dt-spec normalization)
=================================================================

- Random anchors per trajectory each epoch (length = N * pairs_per_traj).
- For each anchor i, sample K strictly-later targets j (j > i), fully vectorized on device.
- Trunk input is **Δt = t_j - t_i** (in physical units), then **normalized using the
  manifest's dt spec** (log-min-max). Absolute time `t_time` is **not** a model input.
- A one-time INFO log prints a side-by-side comparison of:
    (A) time-spec normalization (historical behavior) and
    (B) dt-spec normalization (current behavior used for the model).
  The model always receives (B).

Options:
 - `uniform_offset_sampling`: for each row, sample K offsets k = j − i uniformly in
   [min_steps, max_steps], then choose a single anchor i_used ∈ [0, T−1−o_max] so all K
   targets fit (o_max is the row's largest sampled offset). This removes triangular (i,j)
   bias but induces an early-anchor skew when K or max_steps is large.
- `share_times_across_batch`: when the physical time grid is truly shared, reuse the same offset
  set across the batch (with masks for rows that can't take all offsets).
- `precompute_dt_table`: with a shared grid, precompute a [T, T] lookup of **dt-spec normalized**
  Δt for fast indexing.

Returned batch (matches Trainer expectations):
    y_i   : [B, S]         (normalized species at anchor)
    dt    : [B, K, 1]      (Δt normalized via **dt spec**)
    y_j   : [B, K, S]      (normalized species at target)
    g     : [B, G]         (normalized globals)
    aux   : {'i':[B], 'j':[B,K]}
    k_mask: [B, K] only when share_times_across_batch=True

Notes:
- Time grids must be strictly increasing. When shared, they must be bitwise-equal across shards.
- Δt is clamped to ε > 0 before any log transform, so log(0) cannot occur.
"""

from __future__ import annotations

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
    """
    Load arrays from NPZ shard file.

    Returns:
        x0: [N, S]     (initial states; not used here except for shape checks upstream)
        globals: [N,G]
        t_vec: [T] or [N,T] (PHYSICAL time)
        y_mat: [N,T,S]
    """
    with np.load(path, allow_pickle=False, mmap_mode="r") as npz:
        return npz["x0"], npz["globals"], npz["t_vec"], npz["y_mat"]


# ------------------------------- Main Dataset --------------------------------

class FlowMapPairsDataset(Dataset):
    """
    Flow-map dataset yielding (y_i, Δt_norm, y_j, g, aux[, k_mask]).

    Semantics
    ---------
    - y_i, y_j, g are normalized by `NormalizationHelper` according to `normalization.json`.
    - Δt = t_j - t_i is computed in **physical units** and then normalized using the
      manifest's **dt spec** (log-min-max). Absolute time is never fed to the model.
    - K targets per anchor (j > i). Offsets are sampled either per-row or uniformly in
      k ∈ [min_steps, max_steps], depending on `uniform_offset_sampling`.

    Shapes
    ------
    - y_i: [B, S]
    - dt : [B, K, 1]   (dt-spec normalized)
    - y_j: [B, K, S]
    - g  : [B, G]
    - aux: {'i':[B], 'j':[B,K]}
    - k_mask: [B, K] present only when `share_times_across_batch=True`.

    Grid assumptions
    ----------------
    - Strictly increasing time vectors. If all shards share the **same** vector, the dataset
      uses a shared grid path; otherwise it stores per-row grids.
    - With a shared grid and `precompute_dt_table=True`, the dataset builds a [T, T] table of
      **dt-spec normalized** Δt once and indexes into it during batching.

    Safety
    ------
    - Δt is clamped to ε > 0 prior to log transforms; j > i is enforced by the sampler.
    - No absolute-time conditioning is used, making the model invariant to time-axis shifts.
    """

    def __init__(
            self,
            processed_root: Path | str,
            split: str,
            config: dict,
            *,
            pairs_per_traj: int = 64,
            min_steps: int = 1,
            max_steps: Optional[int] = None,
            preload_to_gpu: bool = False,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            seed: int = 42,
            log_every_files: int = 1000,
    ):
        super().__init__()
        self.root = Path(processed_root)
        self.split = str(split)
        self.cfg = config
        self.base_seed = int(seed)
        self.log_every_files = int(log_every_files)

        # Dataset cfg
        dcfg = self.cfg.get("dataset", {})
        self.require_dt_stats = bool(dcfg.get("require_dt_stats", True))
        self.precompute_dt_table = bool(dcfg.get("precompute_dt_table", True))
        self.multi_time = bool(dcfg.get("multi_time_per_anchor", False))
        self.K = int(dcfg.get("times_per_anchor", 1)) if self.multi_time else 1
        self.share_offsets_across_batch = bool(dcfg.get("share_times_across_batch", False))
        self.uniform_offset_sampling = bool(dcfg.get("uniform_offset_sampling", False))

        # Manifest & normalization
        norm_path = self.root / "normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization.json at: {norm_path}")
        manifest = load_json(norm_path)

        # Device & helper
        self._stage_device = device if (preload_to_gpu and device is not None) else torch.device("cpu")
        self.device = self._stage_device
        self._runtime_dtype = dtype
        self.norm = NormalizationHelper(manifest, device=self._stage_device)

        # Time normalization setup
        self.time_var = self.cfg.get("data", {}).get("time_variable", "t_time")
        tn = getattr(self.norm, "time_norm", {}) or {}
        self.time_method = tn.get(
            "time_transform",
            self.cfg.get("normalization", {}).get("methods", {}).get(self.time_var, "log-min-max")
        )
        self.epsilon = float(getattr(self.norm, "epsilon", 1e-25))

        # Discover shards & scan shapes
        self.files = self._discover_shards()
        scan = self._scan_shards()
        self.N = int(scan["total_trajectories"])
        self.T = int(scan["time_length"])
        self.S_full = int(scan["state_dim"])
        self.G_dim = int(scan["global_dim"])

        # Species variables from config - simplified, no targets
        data_cfg = self.cfg.get("data", {})
        self.species_vars = list(data_cfg.get("species_variables", []))
        if not self.species_vars:
            raise KeyError("config.data.species_variables must be set and non-empty.")
        if len(self.species_vars) != self.S_full:
            raise RuntimeError(
                f"Shard species dim (S_full={self.S_full}) does not match "
                f"len(config.data.species_variables)={len(self.species_vars)}. "
                "Were the shards generated with a different species list?"
            )

        # Dataset species dimension is simply S_full
        self.S = self.S_full

        cfg_globals = data_cfg.get("global_variables", [])
        if self.G_dim > 0 and not cfg_globals:
            raise KeyError(f"Shards contain {self.G_dim} global feature(s), but config.data.global_variables is empty.")

        self.has_shared_grid = bool(scan["has_shared_grid"])
        shared_grid_ref = scan["shared_grid"]

        # Step bounds and length
        self.min_steps = int(min_steps)

        # Default max_steps behavior
        if max_steps is None:
            if self.uniform_offset_sampling and self.multi_time:
                # Default: max_steps = K ensures anchors ∈ [0, T−1−K] with K targets each,
                # i.e., "latest anchor = T−1−K". If you want longer horizons, raise max_steps.
                self.max_steps = self.K
                print(f"[{self.split}] uniform_offset_sampling=True: defaulting max_steps to K={self.K}")
            else:
                # Original behavior: allow full temporal range
                self.max_steps = self.T - 1
        else:
            # Explicit override from config or function argument
            self.max_steps = int(max_steps)

        # Validate step bounds after setting defaults
        self._validate_step_bounds()

        # Must set this BEFORE feasibility checks that reference it
        self.pairs_per_traj = int(pairs_per_traj)

        # Ensure each anchor can have K **unique** downstream samples (no masks, no duplicates)
        if self.multi_time:
            needed_span = self.min_steps + (self.K - 1)
            if self.max_steps < needed_span:
                raise ValueError(
                    f"max_steps ({self.max_steps}) must be ≥ min_steps+K-1 ({needed_span}) "
                    f"to guarantee K downstream samples per anchor."
                )
            self._max_anchor_for_k = self.T - 1 - needed_span
            if self._max_anchor_for_k < 0:
                raise ValueError(
                    f"T={self.T} too small for K={self.K} with min_steps={self.min_steps}."
                )
            if self.pairs_per_traj > (self._max_anchor_for_k + 1):
                raise ValueError(
                    f"pairs_per_traj exceeds number of available distinct anchors. "
                    f"pairs_per_traj={self.pairs_per_traj} > {self._max_anchor_for_k + 1}"
                )
        else:
            # Fallback when not multi-time; only need 1 valid j per anchor.
            self._max_anchor_for_k = self.T - 1 - self.min_steps

        self._length = int(self.N * self.pairs_per_traj)

        # Allocate/stage tensors
        self._allocate_buffers()

        # Time grids
        if self.has_shared_grid:
            self.shared_time_grid = torch.from_numpy(shared_grid_ref.astype(np.float64, copy=False)).to(
                self._stage_device)
            self.time_grid_per_row = None
        else:
            self.shared_time_grid = None
            self.time_grid_per_row = torch.empty((self.N, self.T), device=self._stage_device, dtype=torch.float64)

        # Load data & normalize Y on device
        self._load_and_stage_data()

        # Optional Δt lookup (normalized with dt spec)
        self.dt_table = None
        if self.has_shared_grid and self.precompute_dt_table:
            self._precompute_dt_table_from_time_stats()

        # Anchor sampling bounds & first epoch
        self.min_anchor = 0
        self.max_anchor = max(0, self.T - 1 - self.min_steps)
        self.set_epoch(0)

        # Report
        self._report_memory_usage()

    # ----------------------------- Initialization -----------------------------

    def _discover_shards(self) -> list[str]:
        pattern = str(self.root / self.split / "*.npz")
        files = sorted(glob(pattern))
        if not files:
            raise RuntimeError(f"[{self.split}] No NPZ shards found at {pattern}")
        return files

    def _scan_shards(self) -> dict:
        N_total = 0
        T_global = None
        S_global = None
        G_global = None
        bytes_on_disk = 0
        shared_grid_possible = True
        shared_grid_ref: Optional[np.ndarray] = None

        start = time.perf_counter()
        for idx, f in enumerate(self.files, 1):
            path = Path(f)
            _, g_np, t_np, y_np = load_shard_arrays(path)

            N_shard, T_shard, S_shard = y_np.shape
            G_shard = g_np.shape[1]

            if t_np.ndim == 1:
                if t_np.shape[0] != T_shard:
                    raise RuntimeError(f"[{self.split}] {path.name}: time vector length != T in y_mat")
                current_grid = np.asarray(t_np, dtype=np.float64)
            elif t_np.ndim == 2:
                if t_np.shape[0] != N_shard or t_np.shape[1] != T_shard:
                    raise RuntimeError(f"[{self.split}] {path.name}: t_vec shape mismatch with y_mat")
                if not np.all(t_np == t_np[0]):
                    shared_grid_possible = False
                    current_grid = None
                else:
                    current_grid = np.asarray(t_np[0], dtype=np.float64)
            else:
                raise RuntimeError(f"[{self.split}] {path.name}: invalid t_vec ndim={t_np.ndim}")

            if shared_grid_possible and current_grid is not None:
                if shared_grid_ref is None:
                    shared_grid_ref = current_grid.copy()
                elif not np.allclose(shared_grid_ref, current_grid, rtol=1e-12, atol=1e-15):
                    shared_grid_possible = False

            N_total += N_shard
            if T_global is None:
                T_global, S_global, G_global = T_shard, S_shard, G_shard
            else:
                if T_global != T_shard or S_global != S_shard or G_global != G_shard:
                    raise RuntimeError("Heterogeneous shard dims; all shards must share T/S/G.")

            try:
                bytes_on_disk += path.stat().st_size
            except Exception:
                pass

            if idx % self.log_every_files == 0 or idx == len(self.files):
                elapsed = time.perf_counter() - start
                print(f"[{self.split}] Scanned {idx}/{len(self.files)} shards "
                      f"({format_bytes(bytes_on_disk)}) in {elapsed:.1f}s")

        if N_total <= 0 or T_global is None:
            raise RuntimeError(f"[{self.split}] No valid data found in shards")

        return {
            "total_trajectories": N_total,
            "time_length": T_global,
            "state_dim": S_global,
            "global_dim": G_global,
            "has_shared_grid": shared_grid_possible and (shared_grid_ref is not None),
            "shared_grid": shared_grid_ref,
            "bytes_on_disk": bytes_on_disk,
        }

    def _validate_step_bounds(self) -> None:
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1 (got {self.min_steps})")
        if self.max_steps < self.min_steps:
            raise ValueError(f"max_steps must be >= min_steps (got {self.max_steps} < {self.min_steps})")
        if self.max_steps > self.T - 1:
            raise ValueError(f"max_steps {self.max_steps} exceeds T-1 = {self.T - 1}")

    def _allocate_buffers(self) -> None:
        buf_dtype = (self._stage_device.type == "cuda") and self._runtime_dtype or torch.float32
        self.G = torch.empty((self.N, self.G_dim), device=self._stage_device, dtype=buf_dtype)
        self.Y = torch.empty((self.N, self.T, self.S), device=self._stage_device, dtype=buf_dtype)

    def _load_and_stage_data(self) -> None:
        write_ptr = 0
        start = time.perf_counter()

        for idx, f in enumerate(self.files, 1):
            path = Path(f)
            _, g_np, t_np, y_np = load_shard_arrays(path)
            n = y_np.shape[0]
            sl = slice(write_ptr, write_ptr + n)

            # -------- Globals (normalize via Helper) --------
            g = torch.from_numpy(g_np.astype(np.float32, copy=False)).to(self._stage_device)
            global_vars = self.cfg["data"].get("global_variables", [])
            if global_vars:
                g = self.norm.normalize(g, global_vars)  # [n,G] normalized
            self.G[sl] = g.to(dtype=self.G.dtype, non_blocking=True)

            # -------- Species Y (normalize via Helper) --------
            y = torch.from_numpy(y_np)  # [n, T, S]
            species_vars = self.cfg["data"]["species_variables"]
            y_norm = self.norm.normalize(y, species_vars).reshape(n, self.T, self.S)
            self.Y[sl] = y_norm.to(self._stage_device, dtype=self.Y.dtype, non_blocking=True)

            # -------- Time grids --------
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

    def _report_memory_usage(self) -> None:
        if self._stage_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            print(
                f"[{self.split}] Mem: G={format_bytes(self.G.numel() * self.G.element_size())}, "
                f"Y={format_bytes(self.Y.numel() * self.Y.element_size())} | "
                f"GPU: {used / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f} GiB"
            )
        print(
            f"[{self.split}] Ready: N={self.N}, T={self.T}, S={self.S}, G={self.G_dim} | "
            f"pairs_per_traj={self.pairs_per_traj}, steps=[{self.min_steps},{self.max_steps}] | "
            f"K={self.K}, multi_time={self.multi_time}, share_offsets={self.share_offsets_across_batch}, "
            f"uniform_offset_sampling={self.uniform_offset_sampling}"
        )

    # ------------------------------ Precompute ---------------------------------

    def _precompute_dt_table_from_time_stats(self) -> None:
        """
        Build a dt_norm lookup table of shape [T, T] for a shared time grid.

        Despite the historical name, this path **uses the manifest's dt spec** (log-min-max)
        via `NormalizationHelper.normalize_dt_from_phys`, not the time variable spec.

        Steps:
        1) Form Δt in physical units as (t_j - t_i) for all i, j.
        2) Clamp non-positive entries to ε.
        3) Normalize with the **dt spec** and store as `self.dt_table` (float32, contiguous).

        Preconditions:
        - `self.has_shared_grid` is True and `self.shared_time_grid` is populated.
        - T is the shared time length across shards.

        Postconditions:
        - `self.dt_table[i, j]` equals the dt-spec normalized Δt for j > i (and ε-clamped otherwise).
        """
        t_phys = self.shared_time_grid.to(torch.float64)  # [T]
        t_i = t_phys.view(-1, 1)
        t_j = t_phys.view(1, -1)
        dt_phys = t_j - t_i  # [i,j]
        eps = self.epsilon
        dt_phys = torch.where(
            dt_phys > 0,
            dt_phys,
            torch.as_tensor(eps, dtype=dt_phys.dtype, device=dt_phys.device),
        )
        # normalize_dt_from_phys accepts a [T,T] Δt tensor; no extra feature dim required here.
        dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32)

        self.dt_table = dt_norm.to(torch.float32).contiguous()

    # --------------------------------- Epoch API --------------------------------

    def set_epoch(self, epoch: int) -> None:
        """
        Deterministically regenerate (traj_idx, anchor_i) for this epoch on device.
        Length remains N * pairs_per_traj.

        Note: Anchors are now distinct per trajectory to avoid duplicate (i,j) pairs.
        """
        self.epoch = int(epoch)
        N = self.N
        B = N * self.pairs_per_traj

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.base_seed) + int(epoch))

        traj = torch.arange(N, device=self.device).repeat_interleave(self.pairs_per_traj)  # [B]

        # Distinct anchors per trajectory: evenly spaced over [0, _max_anchor_for_k]
        max_anchor = int(max(0, self._max_anchor_for_k))
        if self.pairs_per_traj == 1:
            anchors_one = torch.zeros(1, device=self.device, dtype=torch.long)
        else:
            anchors_one = torch.linspace(0, max_anchor, steps=self.pairs_per_traj, device=self.device).floor().to(
                torch.long)
        anchors = anchors_one.unsqueeze(0).expand(N, -1).reshape(-1)  # [B]

        self.index_map = torch.stack([traj, anchors], dim=1).to(torch.long)  # [B,2] on device
        self._torch_gen = gen  # reuse in batch-time sampling for determinism

    # ------------------------------ Sampler / Gather -----------------------------

    def _sample_target_indices(
            self,
            i: torch.LongTensor,  # [B]
            K: int,
            min_steps: int,
            max_steps: int,
            shared_offsets: bool,
            *,
            generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
        """
        Vectorized sampler for target indices j > i.

        Modes
        -----
         1) uniform_offset_sampling=True (per-row uniform offsets with a shared anchor):
         - Sample o[b,k] ~ Uniform{min_steps,…,max_steps}.
         - Let o_max[b] = max_k o[b,k]; draw i_used[b] uniformly from [0, T−1−o_max[b]].
         - Set j[b,k] = i_used[b] + o[b,k]. Returns dense mask==True by construction.

        2) uniform_offset_sampling=False:
        - Per-row windows with j_lo = i + min_steps, j_hi = min(i + max_steps, T-1).
        - If `shared_offsets=True`, draw one offset set and mask rows that can't use all of them.
        - Otherwise, draw per-row offsets valid within each row's window without replacement.

        Returns
        -------
        j      : LongTensor [B, K]   (target indices)
        mask   : BoolTensor [B, K]   (True where the chosen offset is valid; all True unless shared_offsets)
        i_used : LongTensor [B]      (actual anchors used to form j)
        When uniform_offset_sampling=True, mask is all True by construction because i_used is chosen to satisfy o_max.
        """
        if i.device != self.device:
            i = i.to(self.device, non_blocking=True)
        assert isinstance(K, int) and K >= 1

        B = int(i.shape[0])
        T = self.T

        # --------- Option 1: per-row uniform offsets (shared anchor constrained by o_max) ----------
        # For each row, sample K offsets uniformly; compute o_max; then draw a single anchor i_used ∈ [0, T−1−o_max].
        # This guarantees all j = i_used + o are valid. Offsets are uniform in k; anchors skew earlier as o_max grows.
        if self.uniform_offset_sampling:
            # Sample offsets uniformly for each (b,k)
            o = torch.randint(min_steps, max_steps + 1, (B, K), device=self.device, generator=generator)  # [B,K]
            # For each row, we must pick an anchor that allows the largest chosen offset
            o_max = o.max(dim=1).values  # [B]
            i_max = (T - 1 - o_max).clamp_min(0)  # [B]
            # Note: The latest possible anchor depends on max_steps via o_max, not K directly.
            # With the default max_steps=K, anchors range from [0, T-1-K]. If max_steps > K is
            # explicitly set, anchors will be more constrained (skewed toward early indices).
            u = torch.rand((B,), device=self.device, generator=generator)
            i_used = (u * (i_max + 1).to(torch.float32)).floor().to(torch.long)  # [B]
            j = i_used.unsqueeze(1) + o  # [B,K]
            mask = torch.ones((B, K), dtype=torch.bool, device=self.device)
            return j, mask, i_used

        # --------- Option 2: per-row window sampling relative to provided anchor i ----------
        j_lo = (i + min_steps).clamp(max=T - 1)  # [B]
        j_hi = torch.minimum(i + max_steps, torch.full_like(i, T - 1))  # [B]
        span = (j_hi - j_lo + 1).clamp_min(1)  # [B]

        if K == 1:
            r = torch.randint(0, int(span.max().item()), (B,), device=self.device, generator=generator)
            r = torch.minimum(r, span - 1)  # [B]
            j = (j_lo + r).view(B, 1)  # [B,1]
            mask = torch.ones((B, 1), dtype=torch.bool, device=self.device)
            return j, mask, i  # i_used = original i

        if shared_offsets:
            max_span = int(span.max().item())
            offs = torch.randint(0, max_span, (K,), device=self.device, generator=generator)  # [K]
            offs = offs.unsqueeze(0).expand(B, K)  # [B,K]
            mask = offs < span.unsqueeze(1)  # [B,K]
            offs = torch.minimum(offs, (span - 1).unsqueeze(1))  # clamp
            j = j_lo.unsqueeze(1) + offs  # [B,K]
            return j, mask, i  # i_used = original i

        # Per-row random offsets **without replacement** (unique j per row)
        max_span = int(span.max().item())
        r = torch.rand((B, max_span), device=self.device, generator=generator)  # random scores
        col = torch.arange(max_span, device=self.device).unsqueeze(0).expand(B, -1)
        invalid = col >= span.unsqueeze(1)  # mask positions beyond span[b]
        r = r.masked_fill(invalid, float("inf"))  # exclude invalid by +inf
        # select K smallest random scores among valid columns → unique column indices per row
        _, offs = torch.topk(-r, k=K, dim=1)  # topk on -r == smallest of r
        offs = offs.to(torch.long)  # [B,K], unique per row
        j = j_lo.unsqueeze(1) + offs  # [B,K]
        mask = torch.ones((B, K), dtype=torch.bool, device=self.device)
        return j, mask, i  # i_used = original i

    @torch.no_grad()
    def _gather_batch(
            self,
            idx_list: torch.LongTensor,  # [B] indices into self.index_map
            *,
            K: int,
            min_steps: int,
            max_steps: int,
            shared_offsets: bool = False,
            gen: torch.Generator | None = None,
    ):
        """
        Assemble a batch on `dataset.device`.

        Pipeline
        --------
        1) Sample K strictly-later targets j for each anchor (possibly updating anchors if needed).
        2) Gather normalized states (y_i, y_j) and globals g.
        3) Compute Δt in **physical units** from the appropriate grid (shared or per-row).
        - Clamp Δt to ε > 0.
        - Normalize **twice** for debugging:
            (A) time-spec  : normalize(..., [self.time_var])   [historical]
            (B) dt-spec    : normalize_dt_from_phys(...)       [current, used]
        - Log a one-time INFO comparison (range stats and mean absolute difference).
        4) Return (y_i, dt_dt_norm, y_j, g, aux[, k_mask]), where `dt_dt_norm` is **dt-spec** normalized.

        Returns
        -------
        When `shared_offsets` is False:
            y_i, dt, y_j, g, aux
        When `shared_offsets` is True:
            y_i, dt, y_j, g, aux, k_mask

        Shapes
        ------
        - y_i: [B, S]
        - dt : [B, K, 1]  (normalized via **dt spec**)
        - y_j: [B, K, S]
        - g  : [B, G]
        - aux: dict with 'i':[B], 'j':[B,K]
        - k_mask: [B, K] only if `share_times_across_batch=True`.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Indices → device,long
        if not isinstance(idx_list, torch.Tensor):
            idx_list = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        elif idx_list.device != self.device:
            idx_list = idx_list.to(self.device, dtype=torch.long, non_blocking=True)
        else:
            idx_list = idx_list.to(torch.long)

        if self.index_map.device != self.device:
            self.index_map = self.index_map.to(self.device, non_blocking=True)

        pair = self.index_map[idx_list]  # [B,2]
        traj = pair[:, 0].to(torch.long)  # [B]
        i0 = pair[:, 1].to(torch.long)  # [B]
        B = int(traj.shape[0])

        # Targets & (possibly) updated anchors
        j, k_mask, i_used = self._sample_target_indices(
            i=i0,
            K=K,
            min_steps=min_steps,
            max_steps=max_steps,
            shared_offsets=shared_offsets,
            generator=(gen or getattr(self, "_torch_gen", None)),
        )  # j:[B,K], k_mask:[B,K], i_used:[B]

        # Gather states/globals (already normalized)
        g = self.G[traj, :]  # [B,G]
        y_i = self.Y[traj, i_used, :]  # [B,S]
        bK = traj.unsqueeze(1).expand(B, K)  # [B,K]
        y_j = self.Y[bK, j, :]  # [B,K,S]

        # ---- Build Δt in PHYSICAL units (always) for normalization & debugging ----
        if self.time_grid_per_row is not None:
            t_i = self.time_grid_per_row[traj, i_used]  # [B]
            t_j = self.time_grid_per_row[traj.unsqueeze(1), j]  # [B,K]
        else:
            vec = self.shared_time_grid  # [T]
            t_i = vec.index_select(0, i_used)  # [B]
            t_j = vec.index_select(0, j.reshape(-1)).view(B, K)  # [B,K]

        dt_phys = t_j - t_i.unsqueeze(1)  # [B,K]
        eps = self.epsilon
        dt_phys = torch.where(
            dt_phys > 0,
            dt_phys,
            torch.as_tensor(eps, dtype=dt_phys.dtype, device=dt_phys.device),
        )

        # ---- Normalize Δt via BOTH pathways for comparison ----
        # (A) time-spec normalization (previous behavior)
        dt_time_norm = self.norm.normalize(
            dt_phys.to(torch.float32).unsqueeze(-1),
            [self.time_var]
        ).to(torch.float32)  # [B,K,1]

        # (B) dt-spec normalization (recommended & USED)
        if self.dt_table is not None:
            # Use precomputed table, but also form a dt-spec tensor for consistent shape
            dt_dt_norm = self.dt_table[i_used.unsqueeze(1), j].unsqueeze(-1)  # [B,K,1]
        else:
            dt_dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32).unsqueeze(-1)  # [B,K,1]

        # ---- One-time log comparing both normalizations ----
        if not hasattr(self, "_dt_norm_debug_logged") or not self._dt_norm_debug_logged:
            with torch.no_grad():
                dtp_min = float(dt_phys.min().item())
                dtp_max = float(dt_phys.max().item())
                dtp_log = torch.log10(dt_phys)  # safe; dt_phys already clamped
                dtpl_min = float(dtp_log.min().item())
                dtpl_max = float(dtp_log.max().item())

                def _summ(x: torch.Tensor) -> tuple[float, float, float]:
                    return float(x.min().item()), float(x.mean().item()), float(x.max().item())

                t_min, t_mean, t_max = _summ(dt_time_norm)
                d_min, d_mean, d_max = _summ(dt_dt_norm)
                mad = float((dt_time_norm - dt_dt_norm).abs().mean().item())

            logger.info(
                "[dataset] Δt normalization debug:\n"
                "  raw Δt (phys): min=%.6g, max=%.6g, log10Δt: min=%.6g, max=%.6g\n"
                "  time-spec  norm Δt: min=%.6g, mean=%.6g, max=%.6g\n"
                "  dt-spec    norm Δt: min=%.6g, mean=%.6g, max=%.6g\n"
                "  mean |time-spec − dt-spec|: %.6g\n"
                "  Using: dt-spec normalization for model input.",
                dtp_min, dtp_max, dtpl_min, dtpl_max,
                t_min, t_mean, t_max,
                d_min, d_mean, d_max,
                mad,
            )
            self._dt_norm_debug_logged = True

        # ---- Use dt-spec normalization as the model input ----
        dt_norm = dt_dt_norm  # [B,K,1]

        aux = {"i": i_used, "j": j}
        if shared_offsets:
            return y_i, dt_norm, y_j, g, aux, k_mask
        else:
            return y_i, dt_norm, y_j, g, aux

    # ------------------------------- Dataloader API ------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> int:
        # Collate builds actual tensors via _gather_batch
        return int(idx)


def create_dataloader(
        dataset: FlowMapPairsDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Construct a DataLoader for FlowMapPairsDataset.

    When dataset tensors are GPU-resident:
      - workers = 0
      - pin_memory = False
      - shuffle = False (anchors randomized deterministically via set_epoch)
      - collate builds batches directly on device
    """
    on_cuda = (getattr(dataset, "_stage_device", torch.device("cpu")).type == "cuda")
    if on_cuda:
        num_workers = 0
        pin_memory = False
        shuffle = False

    def _collate(idxs):
        # Convert indices to tensor on the dataset's device
        if isinstance(idxs, (list, tuple)):
            idxs = torch.tensor(idxs, dtype=torch.long, device=dataset.device)
        elif isinstance(idxs, torch.Tensor):
            if idxs.device != dataset.device:
                idxs = idxs.to(dataset.device, dtype=torch.long, non_blocking=True)
            else:
                idxs = idxs.to(torch.long)
        else:
            idxs = torch.tensor(list(idxs), dtype=torch.long, device=dataset.device)

        return dataset._gather_batch(
            idxs,
            K=dataset.K,
            min_steps=dataset.min_steps,
            max_steps=dataset.max_steps,
            shared_offsets=dataset.share_offsets_across_batch,
            gen=getattr(dataset, "_torch_gen", None),
        )

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": _collate,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)

    return torch.utils.data.DataLoader(**kwargs)