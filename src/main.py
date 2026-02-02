#!/usr/bin/env python3
"""
main.py - Strict entrypoint for training / evaluation.

Principles (by design):
- No alias keys. Config must use the canonical schema.
- No silent fallbacks (no “search for a checkpoint”, no “use CPU if CUDA missing”, etc.).
- Errors are concise and deterministic.

This script:
1) Loads config JSON.
2) Resolves relative paths relative to the config file directory.
3) Loads normalization manifest (processed_dir/normalization.json) and checks config consistency.
4) Builds datasets/dataloaders sized for the rollouts actually used.
5) Builds model strictly from model.type.
6) Runs Lightning training or test, using explicit checkpoint behavior.

Notes:
- Device selection here is ONLY for dataset preloading (dataset.preload_to_device).
  Lightning decides training device(s) based on cfg.runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

# Avoid MKL/OpenMP duplicate symbol aborts in some environments.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Prefer fast matmul where supported.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from lightning.pytorch import seed_everything

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config

log = logging.getLogger(__name__)


# ==============================================================================
# Small strict helpers
# ==============================================================================


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing: {key}")
    return mapping[key]


def _require_dict(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    val = _require(mapping, key)
    if not isinstance(val, dict):
        raise TypeError(f"bad type: {key}")
    return val


def _as_int(val: Any, key: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"bad type: {key}")
    return int(val)


def _as_bool(val: Any, key: str) -> bool:
    if not isinstance(val, bool):
        raise TypeError(f"bad type: {key}")
    return bool(val)


def _as_str(val: Any, key: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise TypeError(f"bad type: {key}")
    return val.strip()


def _as_opt_int(val: Any, key: str) -> Optional[int]:
    if val is None:
        return None
    return _as_int(val, key)


def _repo_root(cfg_path: Path) -> Path:
    return cfg_path.parent.resolve()


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((root / pth).resolve())


def resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Resolve cfg.paths[*] relative to the config file directory."""
    root = _repo_root(cfg_path)

    out = dict(cfg)
    paths = _require_dict(out, "paths")
    resolved: Dict[str, Any] = dict(paths)

    for k, v in paths.items():
        if isinstance(v, str) and v.strip():
            resolved[k] = _resolve_path(root, v)

    out["paths"] = resolved
    return out


def configure_logging(cfg: Mapping[str, Any]) -> None:
    sys_cfg = _require_dict(cfg, "system")
    level_str = str(_require(sys_cfg, "log_level")).upper().strip()
    level = getattr(logging, level_str, None)
    if level is None:
        raise ValueError("bad log_level")
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.captureWarnings(True)


def select_preload_device(cfg: Mapping[str, Any]) -> torch.device:
    """Select device ONLY for dataset preloading (dataset.preload_to_device=True)."""
    sys_cfg = _require_dict(cfg, "system")
    pref = str(_require(sys_cfg, "device")).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        return torch.device(pref)

    if pref == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("mps unavailable")
        return torch.device("mps")

    raise ValueError("bad system.device")


# ==============================================================================
# Manifest consistency
# ==============================================================================


def load_manifest_and_validate_config(
    cfg: Mapping[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load normalization.json and require config.data matches it (if present in manifest)."""
    mpath = processed_dir / "normalization.json"
    if not mpath.exists():
        raise FileNotFoundError("missing normalization.json")

    manifest = load_json_config(mpath)
    if not isinstance(manifest, dict):
        raise TypeError("bad normalization.json")

    data_cfg = _require_dict(cfg, "data")
    cfg_species = _require(data_cfg, "species_variables")
    cfg_globals = _require(data_cfg, "global_variables")

    if not isinstance(cfg_species, list) or not all(isinstance(x, str) for x in cfg_species) or not cfg_species:
        raise TypeError("bad data.species_variables")
    if not isinstance(cfg_globals, list) or not all(isinstance(x, str) for x in cfg_globals):
        raise TypeError("bad data.global_variables")

    man_species = manifest.get("species_variables", None)
    man_globals = manifest.get("global_variables", None)

    if man_species is not None:
        if not isinstance(man_species, list) or not all(isinstance(x, str) for x in man_species) or not man_species:
            raise TypeError("bad manifest.species_variables")
        if list(cfg_species) != list(man_species):
            raise ValueError("species_variables mismatch")

    if man_globals is not None:
        if not isinstance(man_globals, list) or not all(isinstance(x, str) for x in man_globals):
            raise TypeError("bad manifest.global_variables")
        if list(cfg_globals) != list(man_globals):
            raise ValueError("global_variables mismatch")

    return manifest, list(cfg_species), list(cfg_globals)


# ==============================================================================
# Rollout sizing
# ==============================================================================


def max_rollout_steps_for_training(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for training split."""
    base_k = _as_int(_require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")

    cur = tcfg.get("curriculum", None)
    if cur is None:
        return base_k
    if not isinstance(cur, dict):
        raise TypeError("bad training.curriculum")

    enabled = cur.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("bad curriculum.enabled")
    if not enabled:
        return base_k

    # Canonical keys only (no aliases).
    start_steps = _as_int(_require(cur, "start_steps"), "training.curriculum.start_steps")
    end_steps = _as_int(_require(cur, "end_steps"), "training.curriculum.end_steps")

    if start_steps < 1 or end_steps < 1:
        raise ValueError("bad curriculum steps")

    return int(max(base_k, end_steps))


def max_rollout_steps_for_eval(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for val/test split (fixed horizon)."""
    base_k = _as_int(_require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")
    return base_k


# ==============================================================================
# Dataloaders
# ==============================================================================


def create_dataloaders(
    cfg: Mapping[str, Any],
    *,
    tcfg: Mapping[str, Any],
    preload_device: torch.device,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Build train/val/test dataloaders (strict)."""
    paths = _require_dict(cfg, "paths")
    processed_dir = Path(_as_str(_require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    ds_cfg = _require_dict(cfg, "dataset")
    windows_per_traj = _as_int(_require(ds_cfg, "windows_per_trajectory"), "dataset.windows_per_trajectory")
    preload_to_device = _as_bool(_require(ds_cfg, "preload_to_device"), "dataset.preload_to_device")
    shard_cache_size = _as_int(_require(ds_cfg, "shard_cache_size"), "dataset.shard_cache_size")
    use_mmap = _as_bool(_require(ds_cfg, "use_mmap"), "dataset.use_mmap")

    batch_size = _as_int(_require(tcfg, "batch_size"), "training.batch_size")
    num_workers = _as_int(_require(tcfg, "num_workers"), "training.num_workers")
    pin_memory = _as_bool(_require(tcfg, "pin_memory"), "training.pin_memory")
    persistent_workers = _as_bool(_require(tcfg, "persistent_workers"), "training.persistent_workers")
    prefetch_factor = _as_opt_int(_require(tcfg, "prefetch_factor"), "training.prefetch_factor")

    if num_workers < 0:
        raise ValueError("bad num_workers")
    if batch_size <= 0:
        raise ValueError("bad batch_size")

    # Strict DataLoader semantics.
    if num_workers == 0:
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers>0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers>0")
    else:
        # prefetch_factor=None means "use torch default" (2) by not passing it through.
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("bad prefetch_factor")

    k_train = max_rollout_steps_for_training(tcfg)
    k_eval = max_rollout_steps_for_eval(tcfg)

    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload_to_device,
        device=preload_device,
        storage_dtype=torch.float32,
        shard_cache_size=shard_cache_size,
        use_mmap=use_mmap,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_eval, seed=seed + 1, **common_ds_kwargs)

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(processed_dir, "test", total_steps=k_eval, seed=seed + 2, **common_ds_kwargs)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_dl = create_dataloader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = create_dataloader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_dl = create_dataloader(test_ds, shuffle=False, drop_last=False, **dl_kwargs) if test_ds is not None else None

    log.info("Dataloaders: train(K=%d) val(K=%d) test=%s", k_train, k_eval, ("yes" if test_dl else "no"))
    return train_dl, val_dl, test_dl


# ==============================================================================
# Checkpoints (strict)
# ==============================================================================


def _load_weights_only(module: torch.nn.Module, ckpt_path: Path, *, strict: bool) -> None:
    """Load Lightning checkpoint weights only (no optimizer/scheduler state)."""
    if not ckpt_path.exists():
        raise FileNotFoundError("ckpt not found")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("bad ckpt")
    if "state_dict" not in obj:
        raise KeyError("missing state_dict")

    state_dict = obj["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("bad state_dict")

    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError("state mismatch")


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError("config not found")

    cfg = load_json_config(cfg_path)
    if not isinstance(cfg, dict):
        raise TypeError("bad config")

    cfg = resolve_paths(cfg, cfg_path)
    configure_logging(cfg)

    # Required top-level sections (strict).
    tcfg = _require_dict(cfg, "training")
    runtime = _require_dict(cfg, "runtime")
    sys_cfg = _require_dict(cfg, "system")
    paths = _require_dict(cfg, "paths")

    mode = _as_str(_require(runtime, "mode"), "runtime.mode").lower()
    if mode not in ("train", "test"):
        raise ValueError("bad runtime.mode")

    seed = _as_int(_require(sys_cfg, "seed"), "system.seed")
    seed_everything(seed, workers=True)

    work_dir = Path(_as_str(_require(paths, "work_dir"), "paths.work_dir")).expanduser().resolve()

    # Check checkpoint_mode early to decide if work_dir must be empty.
    ckpt_mode = _as_str(_require(tcfg, "checkpoint_mode"), "training.checkpoint_mode").lower()
    if ckpt_mode not in ("none", "resume", "weights_only"):
        raise ValueError("bad checkpoint_mode")

    # No automatic backups. Fresh training requires an empty work_dir.
    # Resume mode allows non-empty work_dir (continuing in same directory).
    if mode == "train" and ckpt_mode != "resume":
        if work_dir.exists() and any(work_dir.iterdir()):
            raise RuntimeError("work_dir not empty")
    ensure_dir(work_dir)

    processed_dir = Path(_as_str(_require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    # Manifest must exist; config.data must match it if manifest provides variable lists.
    manifest, species_vars, global_vars = load_manifest_and_validate_config(cfg, processed_dir)

    atomic_write_json(work_dir / "config.resolved.json", dict(cfg))

    preload_device = select_preload_device(cfg)
    log.info("preload device: %s", preload_device)

    train_dl, val_dl, test_dl = create_dataloaders(cfg, tcfg=tcfg, preload_device=preload_device, seed=seed)

    model = create_model(dict(cfg))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: type=%s S=%d G=%d params=%d trainable=%d",
             str(_require_dict(cfg, "model").get("type", "")),
             len(species_vars), len(global_vars), total_params, trainable_params)

    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
    )

    # On resume/weights-only runs, preserve any existing metrics.csv before Lightning touches it.
    if mode == "train" and ckpt_mode in ("resume", "weights_only") and os.environ.get("LOCAL_RANK", "0") == "0":
        m = work_dir / "metrics.csv"
        if m.exists():
            dst = work_dir / "metrics.pre_restart.csv"
            i = 1
            while dst.exists():
                dst = work_dir / f"metrics.pre_restart.{i}.csv"
                i += 1
            m.replace(dst)

    trainer = build_lightning_trainer(cfg, work_dir=work_dir)

    ckpt_mode = _as_str(_require(tcfg, "checkpoint_mode"), "training.checkpoint_mode").lower()
    ckpt_val = runtime.get("checkpoint", None)
    ckpt_path: Optional[Path] = None

    if ckpt_val is not None:
        ckpt_path = Path(_as_str(ckpt_val, "runtime.checkpoint")).expanduser().resolve()

    if mode == "train":
        strict_load = bool(_require(runtime, "load_weights_strict"))

        if ckpt_mode == "none":
            if ckpt_path is not None:
                raise ValueError("checkpoint_mode=none with checkpoint set")
            trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

        elif ckpt_mode == "resume":
            if ckpt_path is None:
                raise ValueError("resume requires checkpoint")
            trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=str(ckpt_path))

        elif ckpt_mode == "weights_only":
            if ckpt_path is None:
                raise ValueError("weights_only requires checkpoint")
            _load_weights_only(lit_module, ckpt_path, strict=strict_load)
            trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

        else:
            raise ValueError("bad checkpoint_mode")

        if test_dl is not None:
            trainer.test(lit_module, dataloaders=test_dl, ckpt_path="best")

    else:  # test
        if test_dl is None:
            raise RuntimeError("no test split")
        if ckpt_path is None:
            raise ValueError("test requires checkpoint")
        trainer.test(lit_module, dataloaders=test_dl, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Concise top-level error. No traceback by default.
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
