#!/usr/bin/env python3
"""
main.py - Entry point for flow-map training and evaluation.

Usage:
    python main.py

Configuration is loaded from <repo_root>/config/config.json.
Set runtime.mode to 'train' or 'eval' in the config.

Directory Structure:
    <repo_root>/
        config/config.json          # Configuration file
        data/processed/             # Preprocessed data shards
        models/                     # Saved checkpoints
"""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# Environment setup (must be before torch import)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config, seed_everything


# ==============================================================================
# Path Resolution
# ==============================================================================


def _config_path() -> Path:
    """Get the fixed config path: <repo_root>/config/config.json."""
    return (Path(__file__).resolve().parent.parent / "config" / "config.json").resolve()


def _repo_root(config_path: Path) -> Path:
    """Derive repo root from config path."""
    return config_path.resolve().parent.parent


def _resolve_path(base: Path, p: str) -> Path:
    """Resolve path relative to base if not absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def resolve_paths(cfg: Dict, config_path: Path) -> Dict:
    """Resolve all path entries in config relative to repo root."""
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
        if key in pcfg:
            pcfg[key] = str(_resolve_path(root, str(pcfg[key])))
        else:
            pcfg[key] = str(_resolve_path(root, default))

    cfg["paths"] = pcfg
    return cfg


# ==============================================================================
# Manifest Handling
# ==============================================================================


def load_manifest(processed_dir: Path) -> Dict:
    """Load normalization manifest from processed data directory."""
    path = processed_dir / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run preprocessing first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sync_config_with_manifest(cfg: Dict, manifest: Dict) -> Dict:
    """
    Synchronize config with manifest, populating missing fields.

    Ensures config has species_variables and global_variables if manifest has them.
    """
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))

    man_species = manifest.get("species_variables", [])
    man_globals = manifest.get("global_variables", [])

    cfg_species = data_cfg.get("species_variables", [])
    cfg_globals = data_cfg.get("global_variables", [])

    # Populate from manifest if missing
    if not cfg_species and man_species:
        data_cfg["species_variables"] = list(man_species)
    elif cfg_species and man_species and list(cfg_species) != list(man_species):
        raise ValueError(
            f"Config species_variables {cfg_species} doesn't match "
            f"manifest {man_species}"
        )

    if not cfg_globals and man_globals:
        data_cfg["global_variables"] = list(man_globals)
    elif cfg_globals and man_globals and list(cfg_globals) != list(man_globals):
        raise ValueError(
            f"Config global_variables {cfg_globals} doesn't match "
            f"manifest {man_globals}"
        )

    cfg["data"] = data_cfg
    return cfg


# ==============================================================================
# Data Loading
# ==============================================================================


def create_dataloaders(cfg: Dict, device: torch.device):
    """
    Create train/val/test dataloaders from config.

    Returns:
        (train_dl, val_dl, test_dl, manifest)
    """
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))

    # Load manifest
    manifest = load_manifest(processed_dir)

    # Dataset parameters
    tcfg = cfg.get("training", {})
    rollout_steps = int(tcfg.get("rollout_steps", 1))
    burn_in_steps = int(tcfg.get("burn_in_steps", 0))
    total_steps = rollout_steps + burn_in_steps

    dcfg = cfg.get("dataset", {})
    windows_per_traj = int(dcfg.get("windows_per_trajectory", 1))
    seed = int(cfg.get("system", {}).get("seed", 1234))
    preload = bool(dcfg.get("preload_to_device", False))
    shard_cache_size = int(dcfg.get("shard_cache_size", 2))

    # Warn about multi-GPU + preload
    if preload and int(tcfg.get("devices", 1)) > 1:
        warnings.warn(
            "preload_to_device=True may cause issues with multiple GPUs",
            stacklevel=2,
        )

    # Create datasets
    common_kwargs = dict(
        total_steps=total_steps,
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload,
        device=device,
        storage_dtype=torch.float32,
        shard_cache_size=shard_cache_size,
    )

    train_ds = FlowMapRolloutDataset(
        processed_dir, "train", seed=seed, **common_kwargs
    )
    val_ds = FlowMapRolloutDataset(
        processed_dir, "validation", seed=seed + 1, **common_kwargs
    )

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(
            processed_dir, "test", seed=seed + 2, **common_kwargs
        )

    # DataLoader parameters
    batch_size = int(tcfg.get("batch_size", 256))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", device.type == "cuda"))

    # Disable pin_memory for incompatible setups
    if pin_memory and device.type == "mps":
        warnings.warn("pin_memory not supported on MPS, disabling", stacklevel=2)
        pin_memory = False

    if pin_memory and preload:
        # create_dataloader will handle this, but be explicit
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
    test_dl = (
        create_dataloader(test_ds, shuffle=False, **dl_kwargs) if test_ds else None
    )

    # Print dataset info
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
    print(f"  Rollout steps:      {rollout_steps}")
    print("=" * 60 + "\n")

    if len(train_dl) == 0:
        raise ValueError("Train DataLoader has 0 batches. Check batch_size and dataset size.")

    return train_dl, val_dl, test_dl, manifest


# ==============================================================================
# Checkpoint Handling
# ==============================================================================


def find_checkpoint(work_dir: Path) -> Optional[Path]:
    """Find best checkpoint in work_dir, falling back to last or most recent."""
    best = work_dir / "best.ckpt"
    if best.exists():
        return best

    last = work_dir / "last.ckpt"
    if last.exists():
        return last

    ckpts = sorted(
        work_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return ckpts[0] if ckpts else None


# ==============================================================================
# Device Selection
# ==============================================================================


def select_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    """
    Main training/evaluation entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    trainer = None

    try:
        # Load config
        cfg_path = _config_path()
        if not cfg_path.exists():
            print(f"Error: Config not found at {cfg_path}", file=sys.stderr)
            return 1

        cfg = load_json_config(cfg_path)
        cfg = resolve_paths(cfg, cfg_path)

        # Set seed
        sys_cfg = cfg.get("system", {})
        seed = int(sys_cfg.get("seed", 1234))
        deterministic = bool(sys_cfg.get("deterministic", False))
        seed_everything(seed, deterministic=deterministic)

        # Select device
        device = select_device()
        print(f"Using device: {device}")

        # Setup directories
        paths = cfg.get("paths", {})
        work_dir = ensure_dir(paths.get("work_dir", "models/run"))
        ensure_dir(paths.get("model_dir", "models"))

        # Load data and sync config with manifest
        train_dl, val_dl, test_dl, manifest = create_dataloaders(cfg, device)
        cfg = sync_config_with_manifest(cfg, manifest)

        # Save config for this run
        atomic_write_json(work_dir / "config.json", cfg)

        # Create model
        species_vars = list(cfg.get("data", {}).get("species_variables", []))
        model = create_model(cfg)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print("=" * 60)
        print("  MODEL CONFIGURATION")
        print("=" * 60)
        print(f"  Type:               {cfg.get('model', {}).get('type', 'mlp')}")
        print(f"  Species (S):        {len(species_vars)}")
        print(
            f"  Globals (G):        {len(cfg.get('data', {}).get('global_variables', []))}"
        )
        print(f"  Total params:       {total_params:,}")
        print(f"  Trainable params:   {trainable_params:,}")
        print("=" * 60 + "\n")

        # Create Lightning module
        lit_module = FlowMapRolloutModule(
            cfg=cfg,
            model=model,
            normalization_manifest=manifest,
            species_variables=species_vars,
            work_dir=work_dir,
        )
        lit_module.set_dataloaders(train_dl, val_dl, test_dl)

        # Build trainer
        trainer = build_lightning_trainer(cfg, work_dir=work_dir)

        # Get mode and checkpoint
        runtime = cfg.get("runtime", {})
        mode = str(runtime.get("mode", "train")).lower().strip()

        ckpt_path = runtime.get("checkpoint")
        if ckpt_path:
            ckpt_path = _resolve_path(_repo_root(cfg_path), str(ckpt_path))
        else:
            ckpt_path = find_checkpoint(work_dir)

        # Training info
        tcfg = cfg.get("training", {})
        print("=" * 60)
        print("  TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"  Max epochs:         {tcfg.get('max_epochs', 100)}")
        print(f"  Learning rate:      {tcfg.get('lr', 0.001)}")
        print(f"  Weight decay:       {tcfg.get('weight_decay', 0.0001)}")
        print(f"  Precision:          {tcfg.get('precision', 'bf16-mixed')}")
        if ckpt_path and mode == "train":
            print(f"  Resume from:        {ckpt_path.name}")
        print("=" * 60)
        print("\nStarting training...\n")

        # Run
        if mode == "train":
            resume = bool(tcfg.get("resume", True))
            trainer.fit(
                lit_module, ckpt_path=str(ckpt_path) if (resume and ckpt_path) else None
            )

            # Run test if available
            if test_dl is not None:
                print("\nRunning test evaluation...")
                trainer.test(lit_module, dataloaders=test_dl)

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
            print(f"Error: Unknown runtime.mode: {mode}", file=sys.stderr)
            return 1

        print("\n" + "=" * 60)
        print("  Completed successfully!")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Clean up trainer resources
        if trainer is not None:
            try:
                if hasattr(trainer, "strategy") and trainer.strategy is not None:
                    if hasattr(trainer.strategy, "teardown"):
                        trainer.strategy.teardown()
            except Exception:
                pass

        cleanup()


if __name__ == "__main__":
    sys.exit(main())
