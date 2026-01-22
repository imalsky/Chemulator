#!/usr/bin/env python3
"""
main.py

Entrypoint for training and evaluation.

Typical usage:

1) Preprocess raw HDF5 into processed NPZ shards:
   python preprocessing.py --config config_job0.json

2) Train:
   python main.py --config config_job0.json --mode train

3) Evaluate (runs validation_step on test split if present):
   python main.py --config config_job0.json --mode eval

Config is strict JSON (no comments).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import load_json_config, seed_everything, ensure_dir


def _resolve_path(base: Path, p: str) -> Path:
    """Resolve a path relative to base if not absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def resolve_paths(cfg: Dict, *, config_path: Path) -> Dict:
    """Resolve relative paths in cfg['paths'] relative to the config file."""
    cfg = dict(cfg)
    pcfg = dict(cfg.get("paths", {}))
    base = config_path.resolve().parent

    # Default locations relative to config directory
    raw_dir = pcfg.get("raw_data_dir", "data/raw")
    processed_dir = pcfg.get("processed_data_dir", "data/processed")
    model_dir = pcfg.get("model_dir", "models")

    pcfg["raw_data_dir"] = str(_resolve_path(base, str(raw_dir)))
    pcfg["processed_data_dir"] = str(_resolve_path(base, str(processed_dir)))
    pcfg["model_dir"] = str(_resolve_path(base, str(model_dir)))

    # Optional explicit work_dir (where checkpoints/logs go)
    if "work_dir" in pcfg:
        pcfg["work_dir"] = str(_resolve_path(base, str(pcfg["work_dir"])))

    cfg["paths"] = pcfg
    return cfg


def pick_checkpoint(work_dir: Path) -> Optional[Path]:
    """Pick a checkpoint from a work directory (best > last > newest)."""
    best = work_dir / "best.ckpt"
    last = work_dir / "last.ckpt"
    if best.exists():
        return best
    if last.exists():
        return last
    ckpts = sorted(work_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def load_manifest(processed_dir: Path) -> Dict:
    """Load the normalization manifest from processed data directory."""
    man_path = processed_dir / "normalization.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}. Run preprocessing first.")
    with open(man_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reject_unsupported_config_keys(cfg: Dict) -> None:
    """
    Reject config keys that are no longer supported.
    
    This prevents silent misconfiguration when users expect removed features to work.
    """
    # target_species subsetting is no longer supported (S_in == S_out enforced)
    if cfg.get("data", {}).get("target_species"):
        raise ValueError(
            "data.target_species is no longer supported. "
            "This codebase now enforces S_in == S_out (all species predicted). "
            "Remove data.target_species from your config."
        )
    
    # allow_partial_simplex was only needed for target_species subsetting
    if cfg.get("model", {}).get("allow_partial_simplex"):
        raise ValueError(
            "model.allow_partial_simplex is no longer supported. "
            "Remove it from your config."
        )


def validate_species_ordering(cfg: Dict, manifest: Dict) -> List[str]:
    """
    Validate that species ordering in config matches manifest exactly.
    
    This is critical for correctness: the model's species dimension i must match
    the loss's species dimension i. A mismatch would cause silent training failure.
    
    Args:
        cfg: Configuration dictionary
        manifest: Normalization manifest from preprocessing
        
    Returns:
        The validated species_variables list
        
    Raises:
        ValueError: If species ordering doesn't match or is missing
    """
    # Get species from config
    cfg_species = list(cfg.get("data", {}).get("species_variables") or [])
    if not cfg_species:
        raise ValueError(
            "data.species_variables must be set in config. "
            "This ensures model dimensions match loss dimensions."
        )
    
    # Get species from manifest (authoritative source from preprocessing)
    manifest_species = list(manifest.get("species_variables") or [])
    if not manifest_species:
        raise ValueError(
            "species_variables not found in normalization manifest. "
            "Re-run preprocessing with data.species_variables set."
        )
    
    # Check exact match (same names in same order)
    if cfg_species != manifest_species:
        raise ValueError(
            f"Species ordering mismatch between config and manifest!\n"
            f"Config species ({len(cfg_species)}):   {cfg_species}\n"
            f"Manifest species ({len(manifest_species)}): {manifest_species}\n"
            f"These must match exactly (same names, same order) to ensure "
            f"model species dimension i matches loss species dimension i."
        )
    
    return cfg_species


def validate_global_ordering(cfg: Dict, manifest: Dict) -> List[str]:
    """
    Validate that global variable ordering in config matches manifest exactly.
    
    Preprocessing normalizes globals by iterating through cfg.global_variables in order
    and writes that column order into each NPZ shard. If you reorder global_variables
    between preprocessing and training, you'll feed wrong columns to the model.
    
    Args:
        cfg: Configuration dictionary
        manifest: Normalization manifest from preprocessing
        
    Returns:
        The validated global_variables list
        
    Raises:
        ValueError: If global ordering doesn't match or is missing
    """
    # Get globals from config
    cfg_globals = list(cfg.get("data", {}).get("global_variables") or [])
    
    # Get globals from manifest (authoritative source from preprocessing)
    manifest_globals = list(manifest.get("global_variables") or [])
    
    # Both empty is OK (no globals used)
    if not cfg_globals and not manifest_globals:
        return []
    
    # One empty but not the other is an error
    if not cfg_globals and manifest_globals:
        raise ValueError(
            f"data.global_variables is empty in config but manifest has: {manifest_globals}\n"
            "Set data.global_variables in config to match preprocessing."
        )
    if cfg_globals and not manifest_globals:
        raise ValueError(
            f"data.global_variables is set in config ({cfg_globals}) but manifest has none.\n"
            "Re-run preprocessing with the same global_variables."
        )
    
    # Check exact match (same names in same order)
    if cfg_globals != manifest_globals:
        raise ValueError(
            f"Global variable ordering mismatch between config and manifest!\n"
            f"Config globals ({len(cfg_globals)}):   {cfg_globals}\n"
            f"Manifest globals ({len(manifest_globals)}): {manifest_globals}\n"
            f"These must match exactly (same names, same order) to ensure "
            f"global column i in the shard matches what the model expects."
        )
    
    return cfg_globals


def validate_config_against_manifest(cfg: Dict, manifest: Dict) -> Tuple[List[str], List[str]]:
    """
    Validate all config orderings against manifest.
    
    Returns:
        (species_vars, global_vars) tuple of validated lists
    """
    species_vars = validate_species_ordering(cfg, manifest)
    global_vars = validate_global_ordering(cfg, manifest)
    return species_vars, global_vars


def make_dataloaders(
    cfg: Dict, *, device: torch.device
) -> Tuple[object, object, Optional[object], Dict]:
    """
    Create train, validation, and optionally test dataloaders.
    
    Returns:
        (train_dl, val_dl, test_dl, manifest) tuple
    """
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_data_dir"])

    manifest = load_manifest(processed_dir)
    
    # Validate species and global ordering before proceeding
    species_vars, global_vars = validate_config_against_manifest(cfg, manifest)

    # Training configuration
    tcfg = cfg.get("training", {})
    rollout_steps = int(tcfg.get("rollout_steps", 1))
    burn_in_steps = int(tcfg.get("burn_in_steps", 0))
    total_steps = rollout_steps + burn_in_steps

    # Dataset configuration
    dcfg = cfg.get("dataset", {})
    windows_per_trajectory = int(dcfg.get("windows_per_trajectory", 1))
    seed = int(cfg.get("system", {}).get("seed", 1234))

    preload_to_device = bool(dcfg.get("preload_to_device", False))
    storage_dtype = torch.float32

    # Warn about preload + multi-GPU incompatibility
    num_devices = int(tcfg.get("devices", 1))
    if preload_to_device and num_devices > 1:
        warnings.warn(
            "dataset.preload_to_device=true with multiple GPUs (training.devices > 1) "
            "may cause issues: each DDP process will preload to the same device. "
            "Set preload_to_device=false for multi-GPU training.",
            UserWarning,
            stacklevel=2,
        )

    # Create datasets
    train_ds = FlowMapRolloutDataset(
        processed_dir,
        "train",
        total_steps=total_steps,
        windows_per_trajectory=windows_per_trajectory,
        seed=seed,
        preload_to_device=preload_to_device,
        device=device,
        storage_dtype=storage_dtype,
    )
    val_ds = FlowMapRolloutDataset(
        processed_dir,
        "validation",
        total_steps=total_steps,
        windows_per_trajectory=windows_per_trajectory,
        seed=seed + 1,
        preload_to_device=preload_to_device,
        device=device,
        storage_dtype=storage_dtype,
    )

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(
            processed_dir,
            "test",
            total_steps=total_steps,
            windows_per_trajectory=windows_per_trajectory,
            seed=seed + 2,
            preload_to_device=preload_to_device,
            device=device,
            storage_dtype=storage_dtype,
        )

    # DataLoader configuration
    bs = int(tcfg.get("batch_size", 256))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(tcfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(tcfg.get("prefetch_factor", 2))

    train_dl = create_dataloader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dl = create_dataloader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_dl = None
    if test_ds is not None:
        test_dl = create_dataloader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    return train_dl, val_dl, test_dl, manifest


def main() -> None:
    """Main entry point for training and evaluation."""
    ap = argparse.ArgumentParser(description="Train or evaluate flow-map model")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    ap.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path (eval or resume)")
    args = ap.parse_args()

    # Load and resolve configuration
    cfg_path = Path(args.config).resolve()
    cfg = load_json_config(cfg_path)
    cfg = resolve_paths(cfg, config_path=cfg_path)

    # Reject unsupported config keys early (before any expensive operations)
    reject_unsupported_config_keys(cfg)

    # Set random seeds for reproducibility
    seed = int(cfg.get("system", {}).get("seed", 1234))
    seed_everything(seed)

    # Device choice: let Lightning handle it; still pass device to dataset preload
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Work directory for logs/checkpoints
    pcfg = cfg["paths"]
    model_dir = Path(pcfg["model_dir"])
    run_name = str(cfg.get("run_name") or cfg.get("experiment_name") or cfg.get("name") or "run")
    work_dir = Path(pcfg.get("work_dir", model_dir / run_name))
    ensure_dir(work_dir)

    # Create dataloaders (validates species and global ordering internally)
    train_dl, val_dl, test_dl, manifest = make_dataloaders(cfg, device=device)

    # Get validated species list (already checked in make_dataloaders)
    species_vars = list(manifest.get("species_variables") or [])

    # Create model and Lightning module
    model = create_model(cfg)
    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
        work_dir=work_dir,
    )

    # Build trainer
    trainer = build_lightning_trainer(cfg, work_dir=work_dir)

    # Determine checkpoint path
    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else None
    if ckpt_path is None:
        ckpt_path = pick_checkpoint(work_dir)

    if args.mode == "train":
        resume = bool(cfg.get("training", {}).get("resume", True))
        trainer.fit(
            lit_module,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=str(ckpt_path) if (resume and ckpt_path) else None,
        )
    else:
        # Evaluate on test split if present, otherwise on validation split
        dl = test_dl if test_dl is not None else val_dl
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {work_dir} and --ckpt not provided.")
        trainer.validate(lit_module, dataloaders=dl, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    main()
