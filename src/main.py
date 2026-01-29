#!/usr/bin/env python3
"""
main.py - Entrypoint for training / evaluation (strict training + manifest).

Goals:
- Follow the strict training schema enforced by FlowMapRolloutModule (trainer.py).
- Enforce consistency between config.json and normalization.json (species/global vars).
- Remove redundant dataloader wiring (do NOT call lit_module.set_dataloaders; pass dataloaders to fit/test).
- Keep path handling deterministic: resolve relative paths against config.json directory.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import seed_everything

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config

log = logging.getLogger(__name__)


# ==============================================================================
# Basic config / path helpers
# ==============================================================================


def _repo_root(cfg_path: Path) -> Path:
    return cfg_path.parent.resolve()


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((root / pth).resolve())


def resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    root = _repo_root(cfg_path)
    cfg = dict(cfg)

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("config.json must contain a 'paths' dict.")
    paths = dict(paths)
    for k, v in list(paths.items()):
        if isinstance(v, str) and v.strip():
            paths[k] = _resolve_path(root, v)
    cfg["paths"] = paths

    runtime = cfg.get("runtime", {}) or {}
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a dict if provided.")
    runtime = dict(runtime)
    ckpt = runtime.get("checkpoint")
    if isinstance(ckpt, str) and ckpt.strip():
        runtime["checkpoint"] = _resolve_path(root, ckpt)
    cfg["runtime"] = runtime

    return cfg


def configure_logging(cfg: Dict[str, Any]) -> None:
    sys_cfg = cfg.get("system", {}) or {}
    if not isinstance(sys_cfg, dict):
        raise TypeError("system must be a dict if provided.")

    level_str = str(sys_cfg.get("log_level", "INFO")).upper().strip()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.captureWarnings(True)


def select_device(cfg: Dict[str, Any]) -> torch.device:
    """
    Used only for dataset preloading (preload_to_device=True) and any eager ops outside Lightning.
    Lightning handles model/device placement.
    """
    sys_cfg = cfg.get("system", {}) or {}
    pref = str(sys_cfg.get("device", "auto")).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref)
        log.warning("Requested device=%s but CUDA is unavailable; using CPU.", pref)
        return torch.device("cpu")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("Requested device=mps but MPS is unavailable; using CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# Strict validation for training schema + normalization manifest consistency
# ==============================================================================


def _require_dict(cfg: Dict[str, Any], key: str, *, context: str) -> Dict[str, Any]:
    if key not in cfg:
        raise KeyError(f"Missing required key '{key}' in {context}.")
    v = cfg[key]
    if not isinstance(v, dict):
        raise TypeError(f"Expected '{key}' in {context} to be a dict, got {type(v).__name__}.")
    return v


def _require_key(cfg: Dict[str, Any], key: str, *, context: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required key '{key}' in {context}.")
    return cfg[key]


def validate_training_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce the strict training schema expected by trainer.FlowMapRolloutModule.

    main.py should not 'setdefault' these; missing keys should fail fast.
    This function intentionally mirrors the required key access patterns in trainer.FlowMapRolloutModule.__init__.
    """
    tcfg = _require_dict(cfg, "training", context="config")

    # Required scalars (explicit keys; no aliases/defaults)
    required_scalars = ["max_epochs", "rollout_steps", "batch_size"]
    for k in required_scalars:
        if k not in tcfg:
            raise KeyError(f"Missing required training key: training.{k}")

    # Deprecated: this codebase uses a single rollout horizon for train/val/test.
    if "val_rollout_steps" in tcfg or "test_rollout_steps" in tcfg:
        raise ValueError("cfg.training must not include val_rollout_steps/test_rollout_steps; use rollout_steps only.")

    # Required nested dicts and keys
    burn = _require_dict(tcfg, "burn_in", context="config.training")
    for k in ("train", "val", "test"):
        if k not in burn:
            raise KeyError(f"Missing required training key: training.burn_in.{k}")

    tf = _require_dict(tcfg, "teacher_forcing", context="config.training")
    for k in ("mode", "p0", "p1", "ramp_epochs"):
        if k not in tf:
            raise KeyError(f"Missing required training key: training.teacher_forcing.{k}")

    loss = _require_dict(tcfg, "loss", context="config.training")
    for k in ("lambda_log10_mae", "lambda_z_mse"):
        if k not in loss:
            raise KeyError(f"Missing required training key: training.loss.{k}")

    opt = _require_dict(tcfg, "optimizer", context="config.training")
    for k in ("name", "lr"):
        if k not in opt:
            raise KeyError(f"Missing required training key: training.optimizer.{k}")

    # Optional dicts (but if present must be dict)
    for opt_key in ("scheduler", "curriculum", "long_rollout"):
        if opt_key in tcfg and not isinstance(tcfg[opt_key], dict):
            raise TypeError(f"training.{opt_key} must be a dict if provided.")

    # Optional runtime torch_compile config (if present must be dicts)
    runtime = cfg.get("runtime", {}) or {}
    if "torch_compile" in runtime and not isinstance(runtime["torch_compile"], dict):
        raise TypeError("runtime.torch_compile must be a dict if provided.")

    return tcfg

def _get_stats_dict(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    For strictness: accept known container keys, but require log_mean/log_std (no legacy-only).
    """
    for key in ("per_key_stats", "species_stats", "stats"):
        v = manifest.get(key)
        if isinstance(v, dict):
            return v
    raise KeyError(
        "normalization.json is missing a stats dict. Expected one of: per_key_stats, species_stats, stats."
    )


def load_manifest_and_sync_data_cfg(
    cfg: Dict[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """
    Strict policy:
    - species_variables/global_variables must be declared in normalization.json OR config.data.
    - If both declare them, they must match exactly.
    - For every species variable, stats must contain log_mean and log_std (legacy-only is rejected).
    """
    manifest_path = processed_dir / "normalization.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
    manifest = load_json_config(manifest_path)

    if not isinstance(manifest, dict):
        raise TypeError("normalization.json must parse as a JSON object/dict.")

    cfg = dict(cfg)
    data_cfg = cfg.get("data", {}) or {}
    if not isinstance(data_cfg, dict):
        raise TypeError("config.data must be a dict if provided.")
    data_cfg = dict(data_cfg)

    man_species = list(manifest.get("species_variables") or [])
    man_globals = list(manifest.get("global_variables") or [])

    cfg_species = list(data_cfg.get("species_variables") or [])
    cfg_globals = list(data_cfg.get("global_variables") or [])

    # Decide authoritative lists (manifest > config), but enforce equality if both present.
    species_vars = man_species or cfg_species
    global_vars = man_globals or cfg_globals

    if not species_vars:
        raise ValueError(
            "Species variables are missing. Provide data.species_variables in config.json or "
            "include 'species_variables' in normalization.json."
        )

    if man_species and cfg_species and cfg_species != man_species:
        raise ValueError(
            f"config.data.species_variables != normalization.json species_variables.\n"
            f"config: {cfg_species}\nmanifest: {man_species}"
        )
    if man_globals and cfg_globals and cfg_globals != man_globals:
        raise ValueError(
            f"config.data.global_variables != normalization.json global_variables.\n"
            f"config: {cfg_globals}\nmanifest: {man_globals}"
        )

    # Strict stats validation: require log_mean/log_std for all species.
    stats = _get_stats_dict(manifest)
    missing: List[str] = []
    legacy_only: List[str] = []
    for s in species_vars:
        entry = stats.get(s)
        if not isinstance(entry, dict):
            missing.append(s)
            continue
        has_log = ("log_mean" in entry) and ("log_std" in entry)
        has_legacy = ("log10_mean" in entry) or ("log10_std" in entry)
        if not has_log:
            if has_legacy:
                legacy_only.append(s)
            else:
                missing.append(s)

    if legacy_only:
        raise KeyError(
            "normalization.json contains legacy-only keys (log10_mean/log10_std) without required "
            f"log_mean/log_std for: {legacy_only}. Regenerate preprocessing outputs."
        )
    if missing:
        raise KeyError(
            f"normalization.json is missing required stats (log_mean/log_std) for: {missing}."
        )

    data_cfg["species_variables"] = list(species_vars)
    data_cfg["global_variables"] = list(global_vars)
    cfg["data"] = data_cfg
    return cfg, manifest, list(species_vars), list(global_vars)


# ==============================================================================
# Dataloaders (non-redundant; sized for max rollout lengths actually used)
# ==============================================================================


def _max_rollout_steps_for_stage(tcfg: Dict[str, Any], *, stage: str) -> int:
    """Compute maximum rollout length K used for a stage.

    Single base horizon: training.rollout_steps.
    Curriculum can increase K for training only; for sizing we take its max.
    long_rollout can increase K for last N epochs (train always; validation/test via flags); for sizing we take the larger K.
    """
    base_k = int(_require_key(tcfg, "rollout_steps", context="cfg.training"))

    stage = str(stage).lower().strip()
    if stage not in ("train", "validation", "test"):
        raise ValueError(f"Unknown stage: {stage} (expected 'train'|'validation'|'test').")

    k = base_k

    cur_cfg = tcfg.get("curriculum", None)
    if stage == "train" and isinstance(cur_cfg, dict) and bool(cur_cfg.get("enabled", False)):
        k = max(k, int(_require_key(cur_cfg, "max_k", context="cfg.training.curriculum")))

    long_cfg = tcfg.get("long_rollout", None)
    if isinstance(long_cfg, dict) and bool(long_cfg.get("enabled", False)):
        long_k = int(_require_key(long_cfg, "long_rollout_steps", context="cfg.training.long_rollout"))
        if stage == "train":
            k = max(k, long_k)
        elif stage == "validation":
            if bool(_require_key(long_cfg, "apply_to_validation", context="cfg.training.long_rollout")):
                k = max(k, long_k)
        elif stage == "test":
            if bool(_require_key(long_cfg, "apply_to_test", context="cfg.training.long_rollout")):
                k = max(k, long_k)

    return int(k)


def create_dataloaders(
    cfg: Dict[str, Any],
    *,
    tcfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[Any, Any, Optional[Any], Dict[str, Any]]:
    paths = _require_dict(cfg, "paths", context="config")
    processed_dir_raw = paths.get("processed_dir")
    if not isinstance(processed_dir_raw, str) or not processed_dir_raw.strip():
        raise KeyError("paths.processed_dir is required and must be a non-empty string.")
    processed_dir = Path(processed_dir_raw).expanduser()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {processed_dir}")

    # Per-stage required window sizes (K transitions => y length K+1).
    k_train = _max_rollout_steps_for_stage(tcfg, stage="train")
    k_val = _max_rollout_steps_for_stage(tcfg, stage="validation")
    k_test = _max_rollout_steps_for_stage(tcfg, stage="test")

    dcfg = cfg.get("dataset", {}) or {}
    if not isinstance(dcfg, dict):
        raise TypeError("dataset must be a dict if provided.")

    windows_per_traj = int(dcfg.get("windows_per_trajectory", 1))
    preload = bool(dcfg.get("preload_to_device", False))
    shard_cache_size = int(dcfg.get("shard_cache_size", 2))
    use_mmap = bool(dcfg.get("use_mmap", False))

    sys_cfg = cfg.get("system", {}) or {}
    seed = int(sys_cfg.get("seed", 1234))

    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", True))

    # Validate preload + pin_memory combination
    if preload and pin_memory:
        raise ValueError("cfg.training.pin_memory must be false when dataset.preload_to_device=true.")

    # Warn about data duplication risk with multi-GPU + preload_to_device.
    if preload:
        runtime = cfg.get("runtime", {}) or {}
        accel = str(runtime.get("accelerator", "auto")).lower().strip()
        if accel == "cpu" and device.type != "cpu":
            raise ValueError(
                "dataset.preload_to_device=True would place batches on CUDA/MPS, but runtime.accelerator=cpu. "
                "Set runtime.accelerator='auto'/'gpu' or disable dataset.preload_to_device."
            )
        if accel in ("gpu", "cuda") and device.type == "cpu":
            raise ValueError(
                "dataset.preload_to_device=True requires a GPU device, but runtime.accelerator requests GPU while system.device resolved to CPU. "
                "Set system.device to a CUDA device or set dataset.preload_to_device=false."
            )
        devices_cfg = runtime.get("devices", "auto")
        try:
            n_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) and str(devices_cfg).isdigit() else None
        except Exception:
            n_devices = None
        if n_devices is not None and n_devices > 1:
            raise RuntimeError(
                "dataset.preload_to_device=True with multiple devices would duplicate data on each device. "
                "Disable preload_to_device or use a single device."
            )

    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload,
        device=device,
        storage_dtype=torch.float32,
        shard_cache_size=shard_cache_size,
        use_mmap=use_mmap,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_val, seed=seed + 1, **common_ds_kwargs)

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(processed_dir, "test", total_steps=k_test, seed=seed + 2, **common_ds_kwargs)

    # DataLoader worker settings: be explicit and fail fast on invalid combinations.
    persistent_raw = tcfg.get("persistent_workers", None)
    prefetch_raw = tcfg.get("prefetch_factor", None)

    if num_workers == 0:
        if persistent_raw not in (None, False):
            raise ValueError("cfg.training.persistent_workers must be false when num_workers=0.")
        if prefetch_raw not in (None,):
            raise ValueError("cfg.training.prefetch_factor must be null/omitted when num_workers=0.")
        persistent = False
        prefetch = None
    else:
        if persistent_raw is None:
            raise KeyError("cfg.training.persistent_workers is required when num_workers>0.")
        if prefetch_raw is None:
            raise KeyError("cfg.training.prefetch_factor is required when num_workers>0.")
        persistent = bool(persistent_raw)
        prefetch = int(prefetch_raw)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,  # DataLoader accepts None; avoids error when num_workers==0
    )

    train_dl = create_dataloader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = create_dataloader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_dl = create_dataloader(test_ds, shuffle=False, drop_last=False, **dl_kwargs) if test_ds is not None else None

    log.info(
        "Dataloaders: train=%d batches (K=%d), val=%d batches (K=%d), test=%s",
        len(train_dl), k_train,
        len(val_dl), k_val,
        (f"{len(test_dl)} batches (K={k_test})" if test_dl is not None else "None"),
    )

    if len(train_dl) == 0:
        raise ValueError(
            f"Train DataLoader has 0 batches. dataset_len={len(train_ds)} batch_size={batch_size}."
        )

    # Return manifest too (loaded by dataset via normalization.json presence, but we want it in main anyway).
    manifest = load_json_config(processed_dir / "normalization.json")
    return train_dl, val_dl, test_dl, manifest


# ==============================================================================
# Checkpoints
# ==============================================================================


def find_resume_checkpoint(work_dir: Path) -> Optional[Path]:
    """
    Look for checkpoints produced by trainer.build_lightning_trainer():
      work_dir/checkpoints/last.ckpt
      otherwise newest .ckpt under work_dir/checkpoints
    """
    ckpt_dir = work_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


# ==============================================================================
# CLI + main
# ==============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_json_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path)
    configure_logging(cfg)

    # Strict training schema validation (fail fast).
    tcfg = validate_training_schema(cfg)

    # Seed (system-scoped; do not mutate training schema).
    sys_cfg = cfg.get("system", {}) or {}
    if not isinstance(sys_cfg, dict):
        raise TypeError("system must be a dict if provided.")
    seed = int(sys_cfg.get("seed", 1234))
    seed_everything(seed, workers=True)

    device = select_device(cfg)
    log.info("Data device (for preloading only): %s", device)

    paths = _require_dict(cfg, "paths", context="config")
    work_dir_raw = paths.get("work_dir")
    if not isinstance(work_dir_raw, str) or not work_dir_raw.strip():
        raise KeyError("paths.work_dir is required and must be a non-empty string.")
    work_dir = ensure_dir(work_dir_raw)

    processed_dir_raw = paths.get("processed_dir")
    if not isinstance(processed_dir_raw, str) or not processed_dir_raw.strip():
        raise KeyError("paths.processed_dir is required and must be a non-empty string.")
    processed_dir = Path(processed_dir_raw).expanduser()

    # Load manifest, enforce strict consistency, and sync cfg.data.{species,globals}.
    cfg, manifest, species_vars, global_vars = load_manifest_and_sync_data_cfg(cfg, processed_dir)

    # Persist the resolved config alongside the run artifacts.
    atomic_write_json(work_dir / "config.resolved.json", cfg)

    # Dataloaders (non-redundant; sized for max rollout lengths used by trainer).
    train_dl, val_dl, test_dl, _ = create_dataloaders(cfg, tcfg=tcfg, device=device)

    # Build model + module.
    model = create_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model: type=%s | S=%d | G=%d | params=%d (trainable=%d)",
        str(cfg.get("model", {}).get("type", "mlp")),
        len(species_vars),
        len(global_vars),
        total_params,
        trainable_params,
    )

    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
    )

    trainer = build_lightning_trainer(cfg, work_dir=work_dir)

    runtime = cfg.get("runtime", {}) or {}
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a dict if provided.")
    mode = str(runtime.get("mode", "train")).lower().strip()
    if mode not in ("train", "test"):
        raise ValueError("runtime.mode must be 'train' or 'test'.")

    # Resume logic (strict: explicit runtime.checkpoint wins; otherwise best-effort last/newest).
    ckpt_path = runtime.get("checkpoint")
    if isinstance(ckpt_path, str) and ckpt_path.strip():
        ckpt_path = str(Path(ckpt_path).expanduser())
    else:
        ckpt = find_resume_checkpoint(work_dir)
        ckpt_path = str(ckpt) if ckpt else None

    if mode == "train":
        resume = bool(tcfg.get("resume", True))
        log.info(
            "Training: max_epochs=%s lr=%s wd=%s batch_size=%s resume=%s ckpt=%s",
            tcfg["max_epochs"],
            tcfg["optimizer"]["lr"],
            tcfg["optimizer"].get("weight_decay", 0.0),
            tcfg["batch_size"],
            resume,
            (Path(ckpt_path).name if ckpt_path else "None"),
        )

        trainer.fit(
            lit_module,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=(ckpt_path if (resume and ckpt_path) else None),
        )

        # Optional post-fit test (only if test split exists).
        if test_dl is not None:
            trainer.test(lit_module, dataloaders=test_dl, ckpt_path="best")

    else:  # test
        if test_dl is None:
            raise ValueError("runtime.mode='test' but no test split found (processed_dir/test missing).")
        log.info("Testing: ckpt=%s", (Path(ckpt_path).name if ckpt_path else "None"))
        trainer.test(lit_module, dataloaders=test_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
