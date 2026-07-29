#!/usr/bin/env python3
"""Rebuild the mini-chem comparison cache with correct pressure units.

The processed shards store pressure in dyn/cm^2 (VULCAN cgs, "barye"), while
mini-chem's namelist `P_in` is in Pa (mini_ch_i_dlsode.f90: P_cgs = P_in * 10).
The original notebook passed the stored value straight through, so every
mini-chem call ran at ten times the true pressure.

This script reproduces the notebook's selection and stepping logic with the
conversion applied, parallelised over per-worker mini-chem run directories, and
writes the cache the notebook's plotting cells consume.

    python -m testing.regen_minichem_cache            # from Chemulator/
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- paths
_here = Path(__file__).resolve()
REPO = _here.parent.parent
TESTING_DIR = REPO / "testing"
MODEL_DIR = REPO / "models" / "final_model"
SHARD = REPO / "data" / "processed" / "test" / "shard_test_mix_r2_200015.npz"
MINI_CHEM_DIR = REPO.parent / "mini_chem"
MINI_CHEM_BIN = MINI_CHEM_DIR / "mini_chem_dlsode"
WORK_ROOT = Path(os.environ.get("MC_WORK_ROOT", "/tmp/mc_workers"))

# Stored pressure is dyn/cm^2; mini-chem wants Pa.
BARYE_TO_PA = 0.1

RG_T = (1000.0, 2500.0)   # K,  Tsai+2022 validation range
RG_P = (1e2, 1e7)         # Pa, Tsai+2022 validation range
N_STATS = int(os.environ.get("MC_N_STATS", "1000"))
SEED = 42

CHEM_TO_MINI = [8, 7, 5, 4, 2, 1, 11, 3, 10, 9, 0, 6]


def chem_to_mini_vmr(vmr_chem: np.ndarray) -> np.ndarray:
    v = np.zeros(13)
    v[CHEM_TO_MINI] = vmr_chem
    return v


def mini_to_chem_vmr(vmr_mini: np.ndarray) -> np.ndarray:
    return np.array(vmr_mini[:12])[CHEM_TO_MINI]


# ----------------------------------------------------------------- worker
def _nml(T_K: float, P_Pa: float, vmr_mini: np.ndarray, t_step: float) -> str:
    vmr_str = ", ".join(f"{max(float(v), 0.0):.10e}" for v in vmr_mini)
    return (
        "&mini_chem\n"
        "network = 'NCHO'\n"
        f"T_in = {T_K:.6f}\n"
        f"P_in = {P_Pa:.10e}\n"
        f"t_step = {t_step:.10e}\n"
        "n_step = 1\n"
        "n_sp = 13\n"
        "data_file = 'chem_data/mini_chem_data_NCHO.txt'\n"
        "sp_file = 'chem_data/mini_chem_sp_NCHO.txt'\n"
        "net_dir = 'chem_data/1x/'\n"
        "met = '1x'\n"
        "/\n\n"
        "&mini_chem_VMR\n"
        "CE_IC = .False.\n"
        "IC_file = 'chem_data/IC/mini_chem_IC_FastChem_1x.txt'\n"
        f"VMR_IC = {vmr_str}\n"
        "/\n"
    )


def _make_workdir(wid: int) -> Path:
    d = WORK_ROOT / f"w{wid}"
    if not (d / "chem_data").exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / "outputs_dlsode").mkdir(exist_ok=True)
        link = d / "chem_data"
        if not link.exists():
            link.symlink_to(MINI_CHEM_DIR / "chem_data")
    return d


_WORK: dict[int, Path] = {}


def _workdir() -> Path:
    pid = os.getpid()
    if pid not in _WORK:
        _WORK[pid] = _make_workdir(pid % 100000)
    return _WORK[pid]


def run_trajectory(args) -> tuple[int, np.ndarray]:
    """Step mini-chem through the full time grid. Returns (traj_idx, (N_TIME, 12))."""
    traj_idx, T_K, P_Pa, y0, t_vec = args
    d = _workdir()
    nml_path = d / "mini_chem.nml"
    out_path = d / "outputs_dlsode" / "dlsode.txt"

    results = [np.asarray(y0, dtype=np.float64).copy()]
    vmr_now = np.asarray(y0, dtype=np.float64).copy()
    for i in range(1, len(t_vec)):
        dt = float(t_vec[i] - t_vec[i - 1])
        vmr_mini = chem_to_mini_vmr(np.clip(vmr_now, 0.0, None))
        nml_path.write_text(_nml(T_K, P_Pa, vmr_mini, dt))
        r = subprocess.run([str(MINI_CHEM_BIN)], cwd=str(d),
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(
                f"mini_chem failed (traj={traj_idx}, dt={dt:.3e} s, "
                f"T={T_K:.0f} K, P={P_Pa:.3e} Pa):\n{r.stderr[-400:]}"
            )
        data = np.loadtxt(out_path, skiprows=1, ndmin=2)
        vmr_now = mini_to_chem_vmr(data[-1, 2:]).astype(np.float64)
        results.append(vmr_now.copy())
    return traj_idx, np.array(results, dtype=np.float32)


# ----------------------------------------------------------------- sampling
def stratified_sample(rng, idxs, k, P_Pa_all, T_all, nbins=5):
    edges_p = np.linspace(np.log10(RG_P[0]), np.log10(RG_P[1]), nbins + 1)
    edges_t = np.linspace(RG_T[0], RG_T[1], nbins + 1)
    bins: dict = {}
    for i in idxs:
        bp = int(np.clip(np.digitize(np.log10(P_Pa_all[i]), edges_p) - 1, 0, nbins - 1))
        bt = int(np.clip(np.digitize(T_all[i], edges_t) - 1, 0, nbins - 1))
        bins.setdefault((bp, bt), []).append(int(i))
    keys = list(bins.keys())
    rng.shuffle(keys)
    out: list[int] = []
    while keys and len(out) < k:
        nxt = []
        for kk in keys:
            if not bins[kk]:
                continue
            j = int(rng.integers(0, len(bins[kk])))
            out.append(bins[kk].pop(j))
            if len(out) == k:
                break
            if bins[kk]:
                nxt.append(kk)
        keys = nxt
    return np.array(sorted(out), dtype=int)


def main() -> int:
    import torch

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)

    d = np.load(SHARD, allow_pickle=False)
    y_mat = d["y_mat"].astype(np.float32)
    g_mat = d["globals"].astype(np.float32)
    t_vec = d["t_vec"].astype(np.float32)
    N_TRAJ, N_TIME, N_SP = y_mat.shape

    P_Pa_all = BARYE_TO_PA * g_mat[:, 0].astype(np.float64)
    T_all = g_mat[:, 1].astype(np.float64)

    print(f"Loaded {N_TRAJ:,} trajectories x {N_TIME} times x {N_SP} species")
    print(f"P (stored, dyn/cm^2): {g_mat[:,0].min():.2e} -> {g_mat[:,0].max():.2e}")
    print(f"P (converted, Pa)   : {P_Pa_all.min():.2e} -> {P_Pa_all.max():.2e}")

    mask = ((P_Pa_all >= RG_P[0]) & (P_Pa_all <= RG_P[1])
            & (T_all >= RG_T[0]) & (T_all <= RG_T[1]))
    regime_idxs = np.where(mask)[0]
    print(f"In Tsai+2022 regime: {len(regime_idxs):,} / {N_TRAJ:,}")
    if len(regime_idxs) < N_STATS:
        print("ERROR: too few in-regime trajectories", file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    stat_idxs = stratified_sample(rng, regime_idxs, N_STATS, P_Pa_all, T_all)
    print(f"Sample of {len(stat_idxs)}:  P {P_Pa_all[stat_idxs].min():.2e} -> "
          f"{P_Pa_all[stat_idxs].max():.2e} Pa   "
          f"T {T_all[stat_idxs].min():.0f} -> {T_all[stat_idxs].max():.0f} K")

    # ---- emulator predictions (main process, batched) ----
    ep = torch.export.load(MODEL_DIR / "physical_model_k1_cpu.pt2")
    chem_model = ep.module()
    dt = (t_vec[1:] - t_vec[0]).astype(np.float32)
    y_chem_all = np.empty((len(stat_idxs), N_TIME - 1, N_SP), dtype=np.float32)
    with torch.no_grad():
        for k, ti in enumerate(stat_idxs):
            n = len(dt)
            y_b = torch.from_numpy(y_mat[ti, 0][None]).float().repeat(n, 1)
            g_b = torch.from_numpy(g_mat[ti][None]).float().repeat(n, 1)
            dt_b = torch.from_numpy(dt).float().view(-1, 1)
            y_chem_all[k] = chem_model(y_b, dt_b, g_b)[:, 0, :].numpy()
    print("Emulator predictions done.")

    # ---- mini-chem, parallel ----
    nw = int(os.environ.get("MC_WORKERS", str(max(2, (os.cpu_count() or 4) - 2))))
    jobs = [(int(ti), float(T_all[ti]), float(P_Pa_all[ti]),
             y_mat[ti, 0].astype(np.float64), t_vec) for ti in stat_idxs]
    print(f"Stepping mini-chem for {len(jobs)} trajectories on {nw} workers ...")

    out: dict[int, np.ndarray] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for n, (ti, arr) in enumerate(ex.map(run_trajectory, jobs, chunksize=1), 1):
            out[ti] = arr
            if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"[{n:4d}/{len(jobs)}] elapsed {el:6.0f}s  "
                      f"ETA {el/n*(len(jobs)-n):6.0f}s", flush=True)

    y_mc_all = np.stack([out[int(ti)] for ti in stat_idxs])
    y_truth_all = np.stack([y_mat[int(ti)] for ti in stat_idxs])

    cache_dir = TESTING_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"minichem_compare_n{N_STATS}_s{SEED}.npz"
    np.savez_compressed(cache, stat_idxs=stat_idxs, y_truth=y_truth_all,
                        y_chem=y_chem_all, y_mc=y_mc_all)
    print(f"Saved {cache} ({cache.stat().st_size/1e6:.1f} MB)")

    # ---- headline statistics ----
    EPS = 1e-30
    def traj_dex(a, b):
        return np.mean(np.abs(np.log10(np.clip(a, EPS, None))
                              - np.log10(np.clip(b, EPS, None))), axis=(1, 2))

    e_chem = traj_dex(y_chem_all, y_truth_all[:, 1:, :])
    e_mc = traj_dex(y_mc_all[:, 1:, :], y_truth_all[:, 1:, :])
    stats = {
        "n_trajectories": int(len(stat_idxs)),
        "P_Pa_min": float(P_Pa_all[stat_idxs].min()),
        "P_Pa_max": float(P_Pa_all[stat_idxs].max()),
        "T_min": float(T_all[stat_idxs].min()),
        "T_max": float(T_all[stat_idxs].max()),
        "emulator": {"median": float(np.median(e_chem)),
                     "p90": float(np.percentile(e_chem, 90))},
        "minichem": {"median": float(np.median(e_mc)),
                     "p90": float(np.percentile(e_mc, 90))},
    }
    (cache_dir / "minichem_stats_corrected.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
