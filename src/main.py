#!/usr/bin/env python3
"""
main.py - Entry point for training and evaluation.

This is the main script for running flow-map model training, validation,
and testing. It orchestrates the full pipeline:

    1. Load configuration from config.json
    2. Resolve paths relative to repository root
    3. Load normalization manifest and validate against config
    4. Build datasets and dataloaders
    5. Create model architecture
    6. Set up PyTorch Lightning trainer
    7. Run training, evaluation, or testing based on mode

Usage:
    python main.py                    # Uses config.json in repo root or config/
    python main.py --config path.json # Specify custom config (if argparse added)

Configuration Modes (runtime.mode in config.json):
    - "train": Train model, optionally run test after
    - "eval": Validate using saved checkpoint
    - "test": Test using saved checkpoint

Checkpoint Resolution:
    If runtime.checkpoint is not specified, searches work_dir for:
    1. best.ckpt (best validation loss)
    2. last.ckpt (most recent)
    3. Any *.ckpt (most recently modified)

Directory Structure:
    repo_root/
    ├── config.json (or config/config.json)
    ├── data/
    │   ├── raw/             # Raw HDF5 files
    │   └── processed/       # Preprocessed shards
    │       ├── train/
    │       ├── validation/
    │       ├── test/
    │       └── normalization.json
    └── models/
        └── run/             # Training outputs
            ├── config.json  # Saved config
            ├── metrics.csv  # Training metrics
            ├── best.ckpt    # Best checkpoint
            └── last.ckpt    # Latest checkpoint
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    Args:
        config_path: Path to the configuration file

    Returns:
        Inferred repository root path
    """
    config_path = Path(config_path).resolve()
    if config_path.name == "config.json" and config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(root: Path, p: str) -> Path:
    """
    Resolve a path relative to root, or return as-is if absolute.

    Args:
        root: Base directory for relative paths
        p: Path string (relative or absolute)

    Returns:
        Resolved absolute path
    """
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def _config_path() -> Path:
    """
    Find configuration file location.

    Searches in order:
    1. repo_root/config/config.json
    2. repo_root/config.json
    3. Parent directories (for running from src/ subdirectory)

    Returns:
        Path to config file (may not exist)
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


def resolve_paths(cfg: Dict, config_path: Path) -> Dict:
    """
    Resolve all path entries in config relative to repository root.

    Converts relative paths in config["paths"] to absolute paths,
    applying defaults for any missing entries.

    Args:
        cfg: Configuration dictionary
        config_path: Path to the config file (used to find repo root)

    Returns:
        Updated configuration with resolved paths
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
    """
    Load normalization manifest from processed data directory.

    The manifest contains normalization statistics and methods computed
    during preprocessing, required for proper data loading and inference.

    Args:
        processed_dir: Path to processed data directory

    Returns:
        Manifest dictionary

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    path = processed_dir / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing first to generate "
            "normalization manifest and data shards."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sync_config_with_manifest(cfg: Dict, manifest: Dict) -> Dict:
    """
    Synchronize configuration with normalization manifest.

    Validates that species and global variables in config match those in
    the manifest, or populates them from manifest if missing.

    Args:
        cfg: Configuration dictionary
        manifest: Normalization manifest from preprocessing

    Returns:
        Updated configuration dictionary

    Raises:
        ValueError: If config variables don't match manifest
    """
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))

    man_species = manifest.get("species_variables", [])
    man_globals = manifest.get("global_variables", [])

    # normalizer.py writes/uses per_key_stats (or species_stats/stats), not species_variables.
    # Species are the keys with log-stat fields, because trainer.build_loss_buffers()
    # requires log10_* or log_* entries.
    if not man_species:
        stats = (
            manifest.get("per_key_stats")
            or manifest.get("species_stats")
            or manifest.get("stats")
            or {}
        )
        if isinstance(stats, dict) and stats:
            inferred_species = []
            for k, v in stats.items():
                if not isinstance(v, dict):
                    continue
                if (
                    ("log10_mean" in v or "log_mean" in v)
                    and ("log10_std" in v or "log_std" in v)
                ):
                    inferred_species.append(k)
            man_species = inferred_species

            # Only infer globals if we successfully inferred species; otherwise leave as-is.
            if not man_globals and man_species:
                sp = set(man_species)
                man_globals = [k for k in stats.keys() if k not in sp]

    cfg_species = data_cfg.get("species_variables", [])
    cfg_globals = data_cfg.get("global_variables", [])

    if not cfg_species and man_species:
        data_cfg["species_variables"] = list(man_species)
    elif cfg_species and man_species and list(cfg_species) != list(man_species):
        raise ValueError(
            f"Config species_variables {cfg_species} doesn't match "
            f"manifest species_variables {man_species}. "
            "Either update config or re-run preprocessing."
        )

    if not cfg_globals and man_globals:
        data_cfg["global_variables"] = list(man_globals)
    elif cfg_globals and man_globals and list(cfg_globals) != list(man_globals):
        raise ValueError(
            f"Config global_variables {cfg_globals} doesn't match "
            f"manifest global_variables {man_globals}. "
            "Either update config or re-run preprocessing."
        )

    cfg["data"] = data_cfg
    return cfg



# ==============================================================================
# Data Loading
# ==============================================================================


def create_dataloaders(cfg: Dict, device: torch.device) -> Tuple:
    """
    Create train/validation/test dataloaders from configuration.

    Loads the normalization manifest, creates datasets for each split,
    and wraps them in DataLoaders with appropriate settings.

    Args:
        cfg: Configuration dictionary with resolved paths
        device: Target device (used for optional preloading)

    Returns:
        Tuple of (train_dl, val_dl, test_dl, manifest) where test_dl may be None

    Raises:
        FileNotFoundError: If processed data doesn't exist
        ValueError: If configuration is invalid
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

    batch_size = int(tcfg.get("batch_size", 256))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", device.type == "cuda"))

    if pin_memory and device.type == "mps":
        warnings.warn(
            "pin_memory not supported on MPS device, disabling",
            stacklevel=2,
        )
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
    test_dl = (
        create_dataloader(test_ds, shuffle=False, **dl_kwargs) if test_ds else None
    )

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
        long_ft_epochs = int(
            long_cfg.get("long_ft_epochs", long_cfg.get("final_epochs", 0)) or 0
        )
        print(
            f"  Long rollout steps:    {long_rollout_steps} (final {long_ft_epochs} epochs)"
        )
        print(
            f"  Apply long to val/test: val={apply_long_to_val}, test={apply_long_to_test}"
        )
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
    """
    Find best available checkpoint in work directory.

    Search order:
    1. best.ckpt (best validation loss - primary checkpoint from ModelCheckpoint)
    2. last.ckpt (most recent training state - if save_last=True)
    3. Any *.ckpt file (most recently modified - fallback for legacy naming)

    Args:
        work_dir: Directory containing checkpoints

    Returns:
        Path to checkpoint file, or None if none found
    """
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
    """
    Select the best available compute device.

    Priority: CUDA GPU > Apple MPS > CPU

    Returns:
        Selected torch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# Cleanup
# ==============================================================================


def cleanup() -> None:
    """
    Clean up PyTorch resources.

    Runs garbage collection and clears CUDA cache to free memory.
    Called at the end of training to ensure clean shutdown.
    """
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

    Loads configuration, sets up data and model, and runs the appropriate
    mode (train/eval/test) based on runtime.mode setting.

    Returns:
        Exit code: 0 for success, non-zero for failure
    """
    trainer = None

    try:
        cfg_path = _config_path()
        if not cfg_path.exists():
            print(f"Error: Config not found at {cfg_path}", file=sys.stderr)
            return 1

        cfg = load_json_config(cfg_path)
        cfg = resolve_paths(cfg, cfg_path)

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
            print(f"  Resume from:        {ckpt_path.name}")
        print("=" * 60)
        print("\nStarting training...\n")

        if mode == "train":
            resume = bool(tcfg.get("resume", True))
            trainer.fit(
                lit_module,
                ckpt_path=str(ckpt_path) if (resume and ckpt_path) else None,
            )

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
