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
- `uniform_offset_sampling`: unbiased offsets k = j - i ~ Uniform[min_steps, max_steps] to reduce
  the natural triangular bias in pair counts.
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
    k_mask: [B, K] only when share_offsets_across_batch=True

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

def pretty_dt_debug_string(
    dt_phys: torch.Tensor,         # [B,K]
    dt_time_norm: torch.Tensor,    # [B,K,1]
    dt_dt_norm: torch.Tensor,      # [B,K,1]
    min_dt_phys: float | None = None,
    max_dt_phys: float | None = None,
) -> str:
    """
    Return a formatted, human-readable summary of Δt ranges/stats.

    Includes percentiles, coverage relative to optional [min_dt_phys, max_dt_phys],
    and a compact table comparing time-spec vs dt-spec normalization stats.
    """
    with torch.no_grad():
        dtp = dt_phys.flatten()
        tnorm = dt_time_norm.flatten()
        dtnorm = dt_dt_norm.flatten()

        # Basic stats
        def stats(x: torch.Tensor):
            return (
                float(x.min().item()),
                float(x.mean().item()),
                float(x.median().item()),
                float(x.quantile(0.9).item()),
                float(x.max().item()),
            )

        dt_min, dt_mean, dt_med, dt_p90, dt_max = stats(dtp)
        t_min, t_mean, t_med, t_p90, t_max = stats(tnorm)
        d_min, d_mean, d_med, d_p90, d_max = stats(dtnorm)

        # Differences
        mad = float((tnorm - dtnorm).abs().mean().item())

        # Coverage in optional window
        cov_str = ""
        if (min_dt_phys is not None) or (max_dt_phys is not None):
            m = torch.ones_like(dtp, dtype=torch.bool)
            if min_dt_phys is not None:
                m &= (dtp >= float(min_dt_phys))
            if max_dt_phys is not None:
                m &= (dtp <= float(max_dt_phys))
            cov = float(m.float().mean().item()) * 100.0
            cov_str = f"\n  coverage within window: {cov:.2f}%"

        def fmt_row(label, vmin, vmean, vmed, vp90, vmax):
            return f"{label:<10}  min={vmin:>11.6g}  mean={vmean:>11.6g}  med={vmed:>11.6g}  p90={vp90:>11.6g}  max={vmax:>11.6g}"

        # Assemble message
        lines = [
            "[dataset] Δt normalization debug:",
            "  raw Δt (phys, seconds):",
            fmt_row("", dt_min, dt_mean, dt_med, dt_p90, dt_max),
            "  normalized Δt (time-spec vs dt-spec):",
            fmt_row("time-spec", t_min, t_mean, t_med, t_p90, t_max),
            fmt_row("dt-spec",   d_min, d_mean, d_med, d_p90, d_max),
            f"  mean |time-spec − dt-spec|: {mad:.6g}",
        ]
        if (min_dt_phys is not None) or (max_dt_phys is not None):
            wlow  = "-∞" if min_dt_phys is None else f"{float(min_dt_phys):.6g}"
            whigh = "+∞" if max_dt_phys is None else f"{float(max_dt_phys):.6g}"
            lines.append(f"  requested Δt window: [{wlow}, {whigh}]"+cov_str)

        lines.append("  Using: dt-spec normalization for model input.")
        return "\n".join(lines)

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
    - k_mask: [B, K] present only when `share_offsets_across_batch=True`.

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
        self.max_steps = int(max_steps if max_steps is not None else (self.T - 1))
        self._validate_step_bounds()
        self.pairs_per_traj = int(pairs_per_traj)
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
        self.G = torch.empty((self.N, self.G_dim), device=self._stage_device, dtype=torch.float32)
        self.Y = torch.empty((self.N, self.T, self.S), device=self._stage_device, dtype=torch.float32)

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
            self.G[sl] = g.to(dtype=torch.float32, non_blocking=True)

            # -------- Species Y (normalize via Helper) --------
            y = torch.from_numpy(y_np)  # [n, T, S]
            species_vars = self.cfg["data"]["species_variables"]
            y_norm = self.norm.normalize(y, species_vars).reshape(n, self.T, self.S)
            self.Y[sl] = y_norm.to(self._stage_device, dtype=torch.float32, non_blocking=True)

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
                f"[{self.split}] Mem: G={format_bytes(self.G.numel()*self.G.element_size())}, "
                f"Y={format_bytes(self.Y.numel()*self.Y.element_size())} | "
                f"GPU: {used/(1024**3):.1f}/{total/(1024**3):.1f} GiB"
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
        t_phys = self.shared_time_grid.to(torch.float64)                   # [T]
        t_i = t_phys.view(-1, 1)
        t_j = t_phys.view(1, -1)
        dt_phys = t_j - t_i  # [i,j]
        eps = self.epsilon
        dt_phys = torch.where(
            dt_phys > 0,
            dt_phys,
            torch.as_tensor(eps, dtype=dt_phys.dtype, device=dt_phys.device),
        )
        # Helper expects last dim == len(var_list); add a singleton feature dim
        #dt_in = dt_phys.to(torch.float32).unsqueeze(-1)                    # [T,T,1]
        dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32)

        self.dt_table = dt_norm.to(torch.float32).contiguous()

    # --------------------------------- Epoch API --------------------------------

    def set_epoch(self, epoch: int) -> None:
        """
        Deterministically regenerate (traj_idx, anchor_i) for this epoch on device.
        Length remains N * pairs_per_traj.
        """
        self.epoch = int(epoch)
        N = self.N
        B = N * self.pairs_per_traj

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.base_seed) + int(epoch))

        traj = torch.arange(N, device=self.device).repeat_interleave(self.pairs_per_traj)  # [B]

        min_anchor = 0
        max_anchor = max(0, self.T - 1 - self.min_steps)
        anchors = torch.randint(min_anchor, max_anchor + 1, (B,), device=self.device, generator=gen)  # [B]

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
        1) uniform_offset_sampling=True:
        - Sample offsets o ~ Uniform{min_steps, ..., max_steps} independently for each (b, k).
        - Choose anchors i_used per row so that all sampled offsets fit within [0, T-1].
        - Returns dense mask==True.

        2) uniform_offset_sampling=False:
        - Per-row windows with j_lo = i + min_steps, j_hi = min(i + max_steps, T-1).
        - If `shared_offsets=True`, draw one offset set and mask rows that can't use all of them.
        - Otherwise, draw per-row offsets valid within each row's window.

        Returns
        -------
        j      : LongTensor [B, K]   (target indices)
        mask   : BoolTensor [B, K]   (True where the chosen offset is valid; all True unless shared_offsets)
        i_used : LongTensor [B]      (actual anchors used to form j)
        """
        if i.device != self.device:
            i = i.to(self.device, non_blocking=True)
        assert isinstance(K, int) and K >= 1

        B = int(i.shape[0])
        T = self.T

        # --------- Option 1: unbiased uniform offsets across [min_steps, max_steps] ----------
        if self.uniform_offset_sampling:
            # Sample offsets uniformly for each (b,k)
            o = torch.randint(min_steps, max_steps + 1, (B, K), device=self.device, generator=generator)  # [B,K]
            # For each row, we must pick an anchor that allows the largest chosen offset
            o_max = o.max(dim=1).values                                                                     # [B]
            i_max = (T - 1 - o_max).clamp_min(0)                                                            # [B]
            # Sample anchors uniformly in the valid range [0, i_max]
            u = torch.rand((B,), device=self.device, generator=generator)
            i_used = (u * (i_max + 1).to(torch.float32)).floor().to(torch.long)                             # [B]
            j = i_used.unsqueeze(1) + o                                                                     # [B,K]
            mask = torch.ones((B, K), dtype=torch.bool, device=self.device)
            return j, mask, i_used

        # --------- Option 2: per-row window sampling relative to provided anchor i ----------
        j_lo = (i + min_steps).clamp(max=T - 1)                                          # [B]
        j_hi = torch.minimum(i + max_steps, torch.full_like(i, T - 1))                   # [B]
        span = (j_hi - j_lo + 1).clamp_min(1)                                            # [B]

        if K == 1:
            r = torch.randint(0, int(span.max().item()), (B,), device=self.device, generator=generator)
            r = torch.minimum(r, span - 1)                                               # [B]
            j = (j_lo + r).view(B, 1)                                                    # [B,1]
            mask = torch.ones((B, 1), dtype=torch.bool, device=self.device)
            return j, mask, i  # i_used = original i

        if shared_offsets:
            max_span = int(span.max().item())
            offs = torch.randint(0, max_span, (K,), device=self.device, generator=generator)  # [K]
            offs = offs.unsqueeze(0).expand(B, K)                                         # [B,K]
            mask = offs < span.unsqueeze(1)                                               # [B,K]
            offs = torch.minimum(offs, (span - 1).unsqueeze(1))                           # clamp
            j = j_lo.unsqueeze(1) + offs                                                  # [B,K]
            return j, mask, i  # i_used = original i

        # Per-row random offsets (valid for each row independently)
        u = torch.rand((B, K), device=self.device, generator=generator)                   # [B,K]
        offs = torch.clamp((u * span.unsqueeze(1)).floor().to(torch.long),
                        max=span.unsqueeze(1) - 1)
        j = j_lo.unsqueeze(1) + offs                                                      # [B,K]
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
        Assemble a batch on `dataset.device` and apply optional physical-Δt masking.

        Returns (shared_offsets=False):
            (y_i, dt_norm, y_j, g, aux)

        Returns (shared_offsets=True):
            (y_i, dt_norm, y_j, g, aux, k_mask)
        where k_mask is [B,K] boolean indicating which offsets are valid after all masking.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Indices -> proper device, dtype
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

        # Targets & possibly updated anchors
        j, k_mask, i_used = self._sample_target_indices(
            i=i0,
            K=K,
            min_steps=min_steps,
            max_steps=max_steps,
            shared_offsets=shared_offsets,
            gen=(gen or getattr(self, "_torch_gen", None)),
        )  # j:[B,K], k_mask:[B,K] (or None), i_used:[B]

        # Gather y_i, y_j, g
        g = self.G[traj, :]  # [B,G]
        y_i = self.Y[traj, i_used, :]  # [B,S]
        y_j = self.Y[traj, j, :].contiguous()  # [B,K,S]

        # Δt in physical seconds
        if self.shared_time_grid is None:
            row = self.time_grid_per_row[traj, :]  # [B,T]
            t_i = row.gather(1, i_used.view(B, 1)).squeeze(1)  # [B]
            t_j = row[torch.arange(B, device=self.device).unsqueeze(1), j]  # [B,K]
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

        # Normalize Δt (both pathways for one-time debug)
        # (A) time-spec normalization (previous behavior; not used for model)
        dt_time_norm = self.norm.normalize(
            dt_phys.to(torch.float32).unsqueeze(-1),
            [self.time_var]
        ).to(torch.float32)  # [B,K,1]

        # (B) dt-spec normalization (USED for model)
        if self.dt_table is not None:
            dt_dt_norm = self.dt_table[i_used.unsqueeze(1), j].unsqueeze(-1)  # [B,K,1]
        else:
            dt_dt_norm = self.norm.normalize_dt_from_phys(dt_phys).to(torch.float32).unsqueeze(-1)  # [B,K,1]

        # One-time debug print with nicer formatting
        if not getattr(self, "_dt_norm_debug_logged", False):
            try:
                ds_cfg = self.cfg.get("dataset", {}) or {}
                min_dt = ds_cfg.get("min_dt_phys", None)
                max_dt = ds_cfg.get("max_dt_phys", None)
                s = pretty_dt_debug_string(dt_phys, dt_time_norm, dt_dt_norm, min_dt, max_dt)
                logger.info(s)
            finally:
                self._dt_norm_debug_logged = True

        # Apply physical-Δt window mask if requested
        ds_cfg = self.cfg.get("dataset", {}) or {}
        min_dt_phys = ds_cfg.get("min_dt_phys", None)
        max_dt_phys = ds_cfg.get("max_dt_phys", None)

        final_mask = None
        if (min_dt_phys is not None) or (max_dt_phys is not None):
            m = torch.ones((B, K), dtype=torch.bool, device=self.device)
            if min_dt_phys is not None:
                m &= (dt_phys >= float(min_dt_phys))
            if max_dt_phys is not None:
                m &= (dt_phys <= float(max_dt_phys))
            final_mask = m

        # Merge with k_mask (shared-offset masking) if present
        if shared_offsets:
            if final_mask is not None:
                k_mask = k_mask & final_mask if k_mask is not None else final_mask
            # Ensure a mask tensor is returned
            if k_mask is None:
                k_mask = torch.ones((B, K), dtype=torch.bool, device=self.device)

        # Prepare outputs
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
    on_cuda = (dataset.Y.device.type == "cuda")
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
