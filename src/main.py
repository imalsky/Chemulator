#!/usr/bin/env python3
"""
main.py - Entry point for training and evaluation.

Pipeline:
  1) Load config.json
  2) Resolve paths relative to repo root
  3) Load normalization manifest and validate/sync config keys
  4) Build datasets and dataloaders
  5) Create (optionally torch.compile) model
  6) Build Lightning trainer
  7) Run train / eval / test based on runtime.mode

Model compilation:
  - Controlled by config key: model.compile (bool)
  - Default: True
  - If torch.compile is unavailable or compilation fails, training proceeds uncompiled.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

try:
    from lightning.pytorch import seed_everything
except ImportError:
    try:
        from pytorch_lightning import seed_everything
    except ImportError as e:
        raise ImportError(
            "Neither 'lightning' nor 'pytorch_lightning' is installed. "
            "Install one of them to run this project."
        ) from e

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config


# ==============================================================================
# Configuration and Paths
# ==============================================================================


def _repo_root(config_path: Path) -> Path:
    """
    Infer repository root from config file location.

    Assumes config is either at repo_root/config.json or repo_root/config/config.json.
    """
    config_path = Path(config_path).resolve()
    if config_path.name == "config.json" and config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(root: Path, p: str) -> Path:
    """Resolve a path relative to root, or return as-is if absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def _config_path() -> Path:
    """
    Find configuration file location.

    Searches in order:
      1) repo_root/config/config.json
      2) repo_root/config.json
      3) Parent directories (for running from src/ subdirectory)
    """
    here = Path(__file__).resolve()
    root = here.parent

    candidates = [
        root / "config" / "config.json",
        root / "config.json",
        root.parent / "config" / "config.json",
        root.parent / "config.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def resolve_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """
    Resolve all path entries in config relative to repository root.

    Converts relative paths in config["paths"] to absolute paths, applying defaults
    for any missing entries.
    """
    cfg = dict(cfg)
    pcfg = dict(cfg.get("paths", {}))
    root = _repo_root(config_path)

    defaults = {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "model_dir": "models",
        "work_dir": "models/run",
    }

    for key, default in defaults.items():
        pcfg[key] = str(_resolve_path(root, str(pcfg.get(key, default))))

    cfg["paths"] = pcfg
    return cfg


# ==============================================================================
# Manifest Handling
# ==============================================================================


def load_manifest(processed_dir: Path) -> Dict[str, Any]:
    """Load normalization manifest from processed data directory."""
    path = processed_dir / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing first to generate "
            "normalization manifest and data shards."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_per_key_stats(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats = manifest.get("per_key_stats")
    if not isinstance(stats, dict) or not stats:
        raise KeyError(
            "Normalization manifest must contain non-empty mapping 'per_key_stats'. "
            "Regenerate normalization.json with the current preprocessing pipeline."
        )
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            out[str(k)] = v
    if not out:
        raise KeyError("Normalization manifest per_key_stats is empty or malformed.")
    return out


def _infer_species_from_methods(manifest: Dict[str, Any]) -> Optional[list]:
    methods = manifest.get("normalization_methods") or manifest.get("methods") or {}
    if not isinstance(methods, dict) or not methods:
        return None
    species = []
    for k, m in methods.items():
        mm = str(m).lower().strip()
        if mm in ("log-standard", "log10-standard"):
            species.append(str(k))
    return species or None


def sync_config_with_manifest(cfg: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize configuration with normalization manifest.

    Strictness note:
      - Trainer requires per_key_stats[species]["log_mean"/"log_std"].
      - This function enforces that requirement and will error if the manifest uses
        only legacy log10_mean/log10_std fields.
    """
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))

    stats = _require_per_key_stats(manifest)

    man_species = list(manifest.get("species_variables") or [])
    man_globals = list(manifest.get("global_variables") or [])

    cfg_species = list(data_cfg.get("species_variables") or [])
    cfg_globals = list(data_cfg.get("global_variables") or [])

    # Determine species_variables (manifest > config > infer from methods).
    if not man_species:
        if cfg_species:
            man_species = list(cfg_species)
        else:
            inferred = _infer_species_from_methods(manifest)
            if inferred:
                man_species = inferred

    if not man_species:
        raise ValueError(
            "Could not determine species_variables. Provide data.species_variables in config.json "
            "or include 'species_variables' in normalization.json."
        )

    # Enforce strict stats keys required by trainer/build_loss_buffers.
    missing = []
    legacy_only = []
    for s in man_species:
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
            "Normalization manifest uses legacy keys (log10_mean/log10_std) without "
            "the required (log_mean/log_std) for these species: "
            f"{legacy_only}. Regenerate normalization.json so species stats contain "
            "'log_mean' and 'log_std'."
        )
    if missing:
        raise KeyError(
            "Normalization manifest missing required stats (log_mean/log_std) for these species: "
            f"{missing}."
        )

    # Infer globals if missing (keep behavior, but deterministic).
    if not man_globals:
        if cfg_globals:
            man_globals = list(cfg_globals)
        else:
            methods = manifest.get("normalization_methods") or manifest.get("methods") or {}
            if isinstance(methods, dict) and methods:
                sp = set(man_species)
                man_globals = [str(k) for k in methods.keys() if str(k) not in sp]

    if cfg_species and cfg_species != man_species:
        raise ValueError(
            f"Config data.species_variables {cfg_species} doesn't match "
            f"manifest/inferred species_variables {man_species}. "
            "Either update config or re-run preprocessing."
        )
    if cfg_globals and man_globals and cfg_globals != man_globals:
        raise ValueError(
            f"Config data.global_variables {cfg_globals} doesn't match "
            f"manifest/inferred global_variables {man_globals}. "
            "Either update config or re-run preprocessing."
        )

    data_cfg["species_variables"] = list(man_species)
    if man_globals:
        data_cfg["global_variables"] = list(man_globals)

    cfg["data"] = data_cfg
    return cfg


# ==============================================================================
# Data Loading
# ==============================================================================


def create_dataloaders(cfg: Dict[str, Any], device: torch.device) -> Tuple:
    """
    Create train/validation/test dataloaders from configuration.

    Returns:
        (train_dl, val_dl, test_dl, manifest)
    """
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))

    manifest = load_manifest(processed_dir)

    tcfg = cfg.get("training", {})
    rollout_steps = int(tcfg.get("rollout_steps", 1))
    burn_in_steps = int(tcfg.get("burn_in_steps", 0))
    val_burn_in_steps = int(tcfg.get("val_burn_in_steps", burn_in_steps))

    long_cfg = dict(tcfg.get("long_rollout", {}) or {})
    long_enabled = bool(long_cfg.get("enabled", False))
    long_rollout_steps = int(long_cfg.get("long_rollout_steps", 0) or 0)
    apply_long_to_val = bool(long_cfg.get("apply_to_validation", True))
    apply_long_to_test = bool(long_cfg.get("apply_to_test", True))

    candidates = [burn_in_steps + rollout_steps, val_burn_in_steps + rollout_steps]
    if long_enabled and long_rollout_steps > 0:
        candidates.append(burn_in_steps + long_rollout_steps)
        if apply_long_to_val or apply_long_to_test:
            candidates.append(val_burn_in_steps + long_rollout_steps)

    total_steps = max(candidates)

    dcfg = cfg.get("dataset", {})
    windows_per_traj = int(dcfg.get("windows_per_trajectory", 1))
    seed = int(cfg.get("system", {}).get("seed", 1234))
    preload = bool(dcfg.get("preload_to_device", False))
    shard_cache_size = int(dcfg.get("shard_cache_size", 2))

    if preload:
        devices_cfg = tcfg.get("devices", 1)
        n_devices: Optional[int] = None
        if isinstance(devices_cfg, int):
            n_devices = devices_cfg
        elif isinstance(devices_cfg, str) and devices_cfg.strip().isdigit():
            n_devices = int(devices_cfg.strip())
        if n_devices is not None and n_devices > 1:
            warnings.warn(
                "preload_to_device=True may cause issues with multiple GPUs. "
                "Each GPU would have duplicate data in memory.",
                stacklevel=2,
            )

    common_kwargs = dict(
        total_steps=total_steps,
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload,
        device=device,
        storage_dtype=torch.float32,
        shard_cache_size=shard_cache_size,
        use_mmap=bool(dcfg.get("use_mmap", False)),
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", seed=seed, **common_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", seed=seed + 1, **common_kwargs)

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(processed_dir, "test", seed=seed + 2, **common_kwargs)

    batch_size = int(tcfg.get("batch_size", 256))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", device.type == "cuda"))

    if pin_memory and device.type == "mps":
        warnings.warn("pin_memory not supported on MPS device, disabling", stacklevel=2)
        pin_memory = False

    if pin_memory and preload:
        pin_memory = False

    persistent = bool(tcfg.get("persistent_workers", num_workers > 0))
    prefetch = int(tcfg.get("prefetch_factor", 2))

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    train_dl = create_dataloader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = create_dataloader(val_ds, shuffle=False, **dl_kwargs)
    test_dl = create_dataloader(test_ds, shuffle=False, **dl_kwargs) if test_ds else None

    print("\n" + "=" * 60)
    print("  DATASET CONFIGURATION")
    print("=" * 60)
    print(f"  Train samples:      {len(train_ds):,}")
    print(f"  Validation samples: {len(val_ds):,}")
    if test_ds:
        print(f"  Test samples:       {len(test_ds):,}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Train batches:      {len(train_dl):,}")
    print(f"  Val batches:        {len(val_dl):,}")
    print(f"  Burn-in steps (train): {burn_in_steps}")
    print(f"  Burn-in steps (val):   {val_burn_in_steps}")
    print(f"  Rollout steps (base):  {rollout_steps}")
    print(f"  Window steps (y_seq):  {total_steps}")
    if long_enabled and long_rollout_steps > 0:
        long_ft_epochs = int(long_cfg.get("long_ft_epochs", long_cfg.get("final_epochs", 0)) or 0)
        print(f"  Long rollout steps:    {long_rollout_steps} (final {long_ft_epochs} epochs)")
        print(f"  Apply long to val/test: val={apply_long_to_val}, test={apply_long_to_test}")
    print("=" * 60 + "\n")

    if len(train_dl) == 0:
        raise ValueError(
            "Train DataLoader has 0 batches. "
            "Check batch_size and dataset size. "
            f"Dataset has {len(train_ds)} samples, batch_size={batch_size}."
        )

    return train_dl, val_dl, test_dl, manifest


# ==============================================================================
# Checkpoint Handling
# ==============================================================================


def find_checkpoint(work_dir: Path) -> Optional[Path]:
    """Find best available checkpoint in work directory."""
    best = work_dir / "best.ckpt"
    if best.exists():
        return best

    last = work_dir / "last.ckpt"
    if last.exists():
        return last

    ckpts = sorted(work_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


# ==============================================================================
# Device Selection
# ==============================================================================


def select_device() -> torch.device:
    """Select device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# Model compilation
# ==============================================================================


def maybe_compile_model(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = dict(cfg.get("model", {}))
    compile_enabled = bool(mcfg.get("compile", True))  # default ON

    if not compile_enabled:
        return model

    if not hasattr(torch, "compile"):
        warnings.warn("model.compile=True but torch.compile is unavailable; continuing without compilation.", stacklevel=2)
        return model

    try:
        return torch.compile(model)  # default settings
    except Exception as e:
        warnings.warn(f"torch.compile failed; continuing without compilation: {e}", stacklevel=2)
        return model


# ==============================================================================
# Cleanup
# ==============================================================================


def cleanup() -> None:
    """Clean up PyTorch resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> int:
    trainer = None
    try:
        cfg_path = _config_path()
        if not cfg_path.exists():
            print(f"Error: Config not found at {cfg_path}", file=sys.stderr)
            return 1

        cfg = load_json_config(cfg_path)
        cfg = resolve_paths(cfg, cfg_path)

        # Ensure config has a place for model.compile (default True).
        cfg.setdefault("model", {})
        cfg["model"].setdefault("compile", True)

        sys_cfg = cfg.get("system", {})
        seed = int(sys_cfg.get("seed", 1234))
        deterministic = bool(sys_cfg.get("deterministic", False))
        seed_everything(seed, workers=True)

        cfg.setdefault("training", {})
        cfg["training"].setdefault("deterministic", deterministic)

        device = select_device()
        print(f"Using device: {device}")

        paths = cfg.get("paths", {})
        work_dir = ensure_dir(paths.get("work_dir", "models/run"))
        ensure_dir(paths.get("model_dir", "models"))

        train_dl, val_dl, test_dl, manifest = create_dataloaders(cfg, device)
        cfg = sync_config_with_manifest(cfg, manifest)

        atomic_write_json(work_dir / "config.json", cfg)

        species_vars = list(cfg.get("data", {}).get("species_variables", []))

        model = create_model(cfg)
        model = maybe_compile_model(model, cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("=" * 60)
        print("  MODEL CONFIGURATION")
        print("=" * 60)
        print(f"  Type:               {cfg.get('model', {}).get('type', 'mlp')}")
        print(f"  Species (S):        {len(species_vars)}")
        print(f"  Globals (G):        {len(cfg.get('data', {}).get('global_variables', []))}")
        print(f"  torch.compile:      {bool(cfg.get('model', {}).get('compile', True))}")
        print(f"  Total params:       {total_params:,}")
        print(f"  Trainable params:   {trainable_params:,}")
        print("=" * 60 + "\n")

        lit_module = FlowMapRolloutModule(
            cfg=cfg,
            model=model,
            normalization_manifest=manifest,
            species_variables=species_vars,
            work_dir=work_dir,
        )
        lit_module.set_dataloaders(train_dl, val_dl, test_dl)

        trainer = build_lightning_trainer(cfg, work_dir=work_dir)

        runtime = cfg.get("runtime", {})
        mode = str(runtime.get("mode", "train")).lower().strip()

        ckpt_path = runtime.get("checkpoint")
        if ckpt_path:
            ckpt_path = _resolve_path(_repo_root(cfg_path), str(ckpt_path))
        else:
            ckpt_path = find_checkpoint(work_dir)

        tcfg = cfg.get("training", {})
        print("=" * 60)
        print("  TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"  Max epochs:         {tcfg.get('max_epochs', 100)}")
        print(f"  Learning rate:      {tcfg.get('lr', 0.001)}")
        print(f"  Weight decay:       {tcfg.get('weight_decay', 0.0001)}")
        print(f"  Precision:          {tcfg.get('precision', 'bf16-mixed')}")
        if ckpt_path and mode == "train":
            print(f"  Resume from:        {Path(ckpt_path).name}")
        print("=" * 60)
        print("\nStarting...\n")

        if mode == "train":
            resume = bool(tcfg.get("resume", True))
            trainer.fit(lit_module, ckpt_path=str(ckpt_path) if (resume and ckpt_path) else None)

            if test_dl is not None:
                print("\nRunning test evaluation...")
                best_ckpt = work_dir / "best.ckpt"
                trainer.test(
                    lit_module,
                    dataloaders=test_dl,
                    ckpt_path=str(best_ckpt) if best_ckpt.exists() else None,
                )

        elif mode == "eval":
            if ckpt_path is None:
                print(f"Error: No checkpoint found in {work_dir}", file=sys.stderr)
                return 1
            trainer.validate(lit_module, ckpt_path=str(ckpt_path))

        elif mode == "test":
            if ckpt_path is None:
                print(f"Error: No checkpoint found in {work_dir}", file=sys.stderr)
                return 1
            if test_dl is None:
                print("Error: No test data available", file=sys.stderr)
                return 1
            trainer.test(lit_module, ckpt_path=str(ckpt_path), dataloaders=test_dl)

        else:
            print(f"Error: Unknown mode '{mode}'", file=sys.stderr)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        try:
            cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
