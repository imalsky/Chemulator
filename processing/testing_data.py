#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


# ---- tweakables (no CLI args) ------------------------------------------------
N_GROUP_SAMPLE = 80          # number of top-level groups to sample per file
MAX_FILES_TO_CHECK = 1       # bump to 2-3 if you want a broader scan
SEED = 0
MAX_EXAMPLES = 5
MAX_TIME_SCAN_HITS = 50      # file-wide "time-like" dataset path hits to print


# ---- config loader (config is one dir up) -----------------------------------
@dataclass(frozen=True)
class Cfg:
    raw_dir: Path
    raw_file_patterns: List[str]
    time_key: str


def _resolve_path(root: Path, p: str) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (root / pp).resolve()


def load_cfg(cfg_path: Path) -> Cfg:
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    root = cfg_path.parent.resolve()

    raw_dir = _resolve_path(root, str(obj["paths"]["raw_dir"]))
    pats = [str(x) for x in obj["preprocessing"]["raw_file_patterns"]]
    time_key = str(obj["preprocessing"]["time_key"])
    return Cfg(raw_dir=raw_dir, raw_file_patterns=pats, time_key=time_key)


def list_raw_files(raw_dir: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(raw_dir.glob(pat)))
    return sorted({p.resolve() for p in out})


# ---- HDF5 helpers ------------------------------------------------------------
def reservoir_sample(keys: Iterable[str], k: int, rng: np.random.Generator) -> List[str]:
    pool: List[str] = []
    for i, key in enumerate(keys):
        if i < k:
            pool.append(key)
            continue
        j = int(rng.integers(0, i + 1))
        if j < k:
            pool[j] = key
    return pool


def leaf_dataset_index(grp: h5py.Group) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            idx.setdefault(leaf, []).append(name)

    grp.visititems(visitor)
    return idx


def resolve_time_dataset_path(grp: h5py.Group, time_key: str, leaf_idx: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    Returns (status, path_or_message)
      status in: ok_path | missing | ambiguous | not_a_dataset
    """
    if "/" in time_key:
        if time_key not in grp:
            return "missing", f"path '{time_key}' not found in group"
        obj = grp[time_key]
        if not isinstance(obj, h5py.Dataset):
            return "not_a_dataset", f"'{time_key}' exists but is not a dataset (type={type(obj)})"
        return "ok_path", time_key

    matches = leaf_idx.get(time_key, [])
    if not matches:
        if time_key in grp.attrs:
            return "not_a_dataset", f"found attribute '{time_key}' (not a dataset)"
        return "missing", f"leaf '{time_key}' not found in group"
    if len(matches) != 1:
        head = matches[:10]
        return "ambiguous", f"leaf '{time_key}' matches multiple datasets: {head}{' ...' if len(matches) > 10 else ''}"
    return "ok_path", matches[0]


def check_time_array(t: np.ndarray) -> Tuple[str, str]:
    """
    Returns (status, message)
      status in: ok | too_short | non_finite | non_increasing
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return "too_short", f"size={t.size} (<2)"
    if not np.all(np.isfinite(t)):
        bad = int(np.count_nonzero(~np.isfinite(t)))
        return "non_finite", f"non-finite count={bad}, size={t.size}"
    dt = np.diff(t)
    bad = np.where(dt <= 0)[0]
    if bad.size > 0:
        i = int(bad[0])
        left = max(0, i - 2)
        right = min(t.size, i + 4)
        snippet = np.array2string(t[left:right], precision=6, separator=", ")
        return "non_increasing", f"first dt<=0 at i={i}: t[{left}:{right}]={snippet}, dt[i]={dt[i]:.6g}"
    return "ok", "strictly increasing, finite, len>=2"


class _StopScan(Exception):
    pass


def scan_file_for_time_like_datasets(fp: h5py.File, *, max_hits: int) -> List[Tuple[str, Tuple[int, ...], str]]:
    hits: List[Tuple[str, Tuple[int, ...], str]] = []

    def visitor(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        leaf = name.split("/")[-1].lower()
        if "time" in name.lower() or leaf in {"t", "time", "times"}:
            hits.append((name, tuple(int(x) for x in obj.shape), str(obj.dtype)))
            if len(hits) >= max_hits:
                raise _StopScan()

    try:
        fp.visititems(visitor)
    except _StopScan:
        pass
    return hits


# ---- main --------------------------------------------------------------------
def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found at expected path: {cfg_path}")

    cfg = load_cfg(cfg_path)
    print(f"[config] path={cfg_path}")
    print(f"[config] raw_dir={cfg.raw_dir}")
    print(f"[config] patterns={cfg.raw_file_patterns}")
    print(f"[config] time_key={cfg.time_key!r}")

    files = list_raw_files(cfg.raw_dir, cfg.raw_file_patterns)
    if not files:
        raise SystemExit("No raw HDF5 files found for raw_dir/patterns in config.")

    rng = np.random.default_rng(SEED)

    for file_i, file_path in enumerate(files[:MAX_FILES_TO_CHECK]):
        print(f"\n[file] {file_path}")
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            print(f"  top_level_keys={len(keys)}")

            # Quick peek at first few keys + types
            for k in keys[:8]:
                obj = f[k]
                typ = "Group" if isinstance(obj, h5py.Group) else "Dataset" if isinstance(obj, h5py.Dataset) else str(type(obj))
                print(f"  root[{k!r}] -> {typ}")

            # Root-level time dataset check (helps detect global time grids)
            if cfg.time_key in f and isinstance(f[cfg.time_key], h5py.Dataset):
                ds = f[cfg.time_key]
                print(f"  NOTE: time_key exists at FILE ROOT as dataset: {cfg.time_key!r} shape={ds.shape} dtype={ds.dtype}")

            k = max(1, min(N_GROUP_SAMPLE, len(keys)))
            sample_names = reservoir_sample(keys, k, rng)
            rng.shuffle(sample_names)

            counts = Counter()
            examples: Dict[str, List[str]] = {}
            time_leaf_counter: Counter[str] = Counter()

            for name in sample_names:
                obj = f[name]
                if not isinstance(obj, h5py.Group):
                    counts["not_group"] += 1
                    if len(examples.setdefault("not_group", [])) < MAX_EXAMPLES:
                        examples["not_group"].append(name)
                    continue

                grp: h5py.Group = obj
                leaf_idx = leaf_dataset_index(grp)

                # Track time-ish leaves present under the group
                for leaf in leaf_idx.keys():
                    ll = leaf.lower()
                    if "time" in ll or ll in {"t", "time", "times"}:
                        time_leaf_counter[leaf] += 1

                p_status, p_msg = resolve_time_dataset_path(grp, cfg.time_key, leaf_idx)
                if p_status != "ok_path":
                    counts[f"time_path_{p_status}"] += 1
                    if len(examples.setdefault(f"time_path_{p_status}", [])) < MAX_EXAMPLES:
                        examples[f"time_path_{p_status}"].append(f"{name}: {p_msg}")
                    continue

                ds_path = p_msg
                try:
                    t = np.asarray(grp[ds_path][...], dtype=np.float64).reshape(-1)
                except Exception as e:  # noqa: BLE001
                    counts["time_read_error"] += 1
                    if len(examples.setdefault("time_read_error", [])) < MAX_EXAMPLES:
                        examples["time_read_error"].append(f"{name}: {ds_path}: {type(e).__name__}: {e}")
                    continue

                t_status, t_msg = check_time_array(t)
                counts[f"time_{t_status}"] += 1
                if t_status != "ok":
                    if len(examples.setdefault(f"time_{t_status}", [])) < MAX_EXAMPLES:
                        examples[f"time_{t_status}"].append(f"{name}: {ds_path}: {t_msg}")
                    continue

                # If OK, print one compact dt summary occasionally (helps spot weird scaling)
                if counts["time_ok"] <= 3:
                    dt = np.diff(t)
                    print(
                        f"  OK time example: group={name!r} ds={ds_path!r} "
                        f"len={t.size} t_min={t.min():.6g} t_max={t.max():.6g} "
                        f"dt_min={dt.min():.6g} dt_med={np.median(dt):.6g} dt_max={dt.max():.6g}"
                    )

            print(f"  sampled_groups={k}")
            for key in sorted(counts.keys()):
                print(f"  {key}={counts[key]}")

            if examples:
                print("  examples (up to 5 each):")
                for key in sorted(examples.keys()):
                    print(f"    {key}:")
                    for line in examples[key]:
                        print(f"      - {line}")

            if time_leaf_counter:
                print("  observed time-like dataset leaf names in sampled groups:")
                for leaf, c in time_leaf_counter.most_common(12):
                    print(f"    - {leaf!r}: seen in {c}/{k} sampled groups")
            else:
                print("  observed time-like dataset leaf names: (none found in sampled groups)")

            hits = scan_file_for_time_like_datasets(f, max_hits=MAX_TIME_SCAN_HITS)
            if hits:
                print("  file-scan hits for time-like dataset paths:")
                for name, shape, dtype in hits:
                    print(f"    - {name} shape={shape} dtype={dtype}")
            else:
                print("  file-scan: no dataset paths containing 'time' (or leaf in {t,time,times}) were found")

        # default: stop after first file unless MAX_FILES_TO_CHECK > 1
        _ = file_i

    print("\nDone.")


if __name__ == "__main__":
    main()