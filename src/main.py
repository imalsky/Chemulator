#!/usr/bin/env python3
"""
main.py — Flow-map training entrypoint (Lightning-backed)

- JSON/JSONC config loading
- Hardware + reproducibility setup
- Preprocess if needed, then hydrate cfg.data.* from artifacts
- Build datasets/dataloaders (single set of knobs; no *_val variants)
- Build model
- Train via Lightning-backed Trainer (CSV only)
"""

from __future__ import annotations

import os
# Resolve duplicate OpenMP on macOS (harmless elsewhere)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from utils import (
    setup_logging,
    seed_everything,
    load_json_config,
    dump_json,
    resolve_precision_and_dtype,
)
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer  # Lightning-backed

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
GLOBAL_SEED = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "autoencoder-flowmap"
VALIDATION_SEED_OFFSET = 1337
CONFIG_SNAPSHOT_FILENAME = "config.snapshot.json"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _norm(p: Union[str, Path]) -> Path:
    return Path(p).expanduser().resolve()

# --------------------------------------------------------------------------------------
# Hydration from processed artifacts
# --------------------------------------------------------------------------------------

def hydrate_config_from_processed(
    cfg: Dict[str, Any],
    logger: logging.Logger,
    processed_dir: Optional[Path] = None,
) -> Path:
    if processed_dir is None:
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"[hydrate] Missing processed data dir: {processed_dir}")

    norm_path = processed_dir / "normalization.json"
    summary_path = processed_dir / "preprocessing_summary.json"
    index_path = processed_dir / "shard_index.json"
    report_path = processed_dir / "preprocess_report.json"

    def _load_json(p: Path) -> Optional[Dict[str, Any]]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[hydrate] Failed to read {p.name}: {e}")
        return None

    manifest = _load_json(norm_path)
    summary = _load_json(summary_path)
    index = _load_json(index_path)
    report = _load_json(report_path)

    meta = (manifest or {}).get("meta", {}) if manifest else {}
    species = meta.get("species_variables") or (summary or {}).get("species_variables")
    globals_ = meta.get("global_variables") or (summary or {}).get("global_variables")
    time_var = meta.get("time_variable") or (summary or {}).get("time_variable")
    if not species:
        raise RuntimeError("[hydrate] Unable to determine species_variables")

    # target_species: artifacts > valid cfg > all species
    targets = (meta.get("target_species") if meta else None) or (manifest or {}).get("target_species")
    if targets is None:
        targets = (summary or {}).get("target_species")
    if targets is None:
        targets = species

    data_cfg = cfg.setdefault("data", {})
    prev_species = list(data_cfg.get("species_variables", []))
    if species:
        if prev_species and prev_species != list(species):
            logger.warning("[hydrate] Overriding cfg.data.species_variables from processed artifacts")
        data_cfg["species_variables"] = list(species)

    prev_globals = list(data_cfg.get("global_variables", []))
    if globals_:
        if prev_globals and prev_globals != list(globals_):
            logger.warning("[hydrate] Overriding cfg.data.global_variables from processed artifacts")
        data_cfg["global_variables"] = list(globals_)

    prevt = data_cfg.get("time_variable")
    if time_var:
        if prevt and prevt != time_var:
            logger.warning("[hydrate] Overriding cfg.data.time_variable from processed artifacts")
        data_cfg["time_variable"] = str(time_var)

    prev_targets = list(data_cfg.get("target_species", []))
    if targets:
        if prev_targets and prev_targets != list(targets):
            logger.warning("[hydrate] Overriding cfg.data.target_species from processed artifacts")
        data_cfg["target_species"] = list(targets)
    else:
        if prev_targets:
            missing = set(prev_targets) - set(species)
            if missing:
                logger.warning("[hydrate] cfg.data.target_species not subset of species")

    return processed_dir

# --------------------------------------------------------------------------------------
# Preprocessing (ensure artifacts exist, then hydrate cfg)
# --------------------------------------------------------------------------------------

def ensure_preprocessed_data(cfg, logger):
    """
    Reuse processed data if complete; otherwise run the preprocessor.
    Completeness = normalization.json, preprocessing_summary.json, shard_index.json,
    and at least one shard in each of train/validation/test.

    NOTE: Processed-data overwrite is controlled by cfg.preprocessing.overwrite_data
          (NOT cfg.paths.overwrite, which only applies to the work_dir).
    """
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed")).resolve()
    # *** THE ONLY BEHAVIOR CHANGE: use preprocessing.overwrite_data ***
    overwrite_data = bool(cfg.get("preprocessing", {}).get("overwrite_data", False))

    req_files = [
        processed_dir / "normalization.json",
        processed_dir / "preprocessing_summary.json",
        processed_dir / "shard_index.json",
    ]
    have_required = all(p.exists() for p in req_files)
    have_splits = all(
        (processed_dir / split).exists() and any((processed_dir / split).glob("*.npz"))
        for split in ("train", "validation", "test")
    )

    # Reuse path when complete and not forcing overwrite
    if have_required and have_splits and not overwrite_data:
        # hydrate cfg.data from the saved snapshot if available (non-destructive)
        snap = processed_dir / "config.snapshot.json"
        if snap.exists():
            try:
                with snap.open("r") as f:
                    snap_cfg = json.load(f)
                data_snap = snap_cfg.get("data", {})
                data_cfg = cfg.setdefault("data", {})
                for k in ("species_variables", "global_variables", "time_variable", "target_species"):
                    if data_snap.get(k) and not data_cfg.get(k):
                        data_cfg[k] = data_snap[k]
                logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
                logger.info("[pre] Injected cfg.data keys from snapshot (non-destructive)")
            except Exception as e:
                logger.warning("[pre] Reusing existing data but failed to read config.snapshot.json: %s", e)
                logger.info("[pre] Proceeding without hydration; downstream may still work via manifests")
        else:
            logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
        return processed_dir

    # If overwrite_data is set, blow away the directory and rebuild
    if overwrite_data and processed_dir.exists():
        logger.warning("Deleting existing processed dir: %s", processed_dir)
        shutil.rmtree(processed_dir, ignore_errors=True)

    # Run preprocessor (will raise if dir exists & not empty and overwrite_data is False)
    dp = DataPreprocessor(cfg, logger=logger.getChild("pre"))
    dp.run()
    return processed_dir


# --------------------------------------------------------------------------------------
# Datasets + DataLoaders (single set of knobs)
# --------------------------------------------------------------------------------------

def build_datasets_and_loaders(
    cfg: Dict[str, Any],
    device: torch.device,
    runtime_dtype: torch.dtype,
    logger: logging.Logger,
) -> Tuple[FlowMapPairsDataset, FlowMapPairsDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    pairs_per_traj = int(training_cfg.get("pairs_per_traj", 64))
    min_steps = int(training_cfg.get("min_steps", 1))
    max_steps = training_cfg.get("max_steps")
    max_steps = int(max_steps) if (max_steps is not None) else None
    base_seed = int(cfg.get("system", {}).get("seed", 42))

    preload_to_gpu = bool(dataset_cfg.get("preload_to_gpu", True))

    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="train", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=base_seed,
    )
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="validation", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=base_seed + VALIDATION_SEED_OFFSET,
    )

    batch_size = int(dataset_cfg.get("batch_size_train", 64))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    persistent = bool(dataset_cfg.get("persistent_workers", False))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    prefetch_factor = int(dataset_cfg.get("prefetch_factor", 2))

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    logger.info(
        f"[dl] B={batch_size} workers={num_workers} prefetch={prefetch_factor} "
        f"pin={pin_memory} persistent_workers={persistent} "
        f"share_times_across_batch={bool(dataset_cfg.get('share_times_across_batch', False))}"
    )
    return train_dataset, val_dataset, train_loader, val_loader

def validate_dataloaders(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    logger: logging.Logger,
) -> None:
    try:
        n_train_items = len(train_loader.dataset)
        n_val_items = len(val_loader.dataset) if val_loader else 0
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
    except Exception:
        n_train_items = n_val_items = n_train_batches = n_val_batches = 0

    logger.info(f"Dataset stats: train={n_train_items} items/{n_train_batches} batches; val={n_val_items} items/{n_val_batches} batches")
    if n_train_batches == 0:
        raise RuntimeError("Empty training loader")
    if val_loader and n_val_batches == 0:
        raise RuntimeError("Empty validation loader")

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    logger.info("Creating model...")
    model = create_model(cfg, logger=logger.getChild("model"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")
    return model

# --------------------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main training entrypoint with fixed config handling."""
    parser = argparse.ArgumentParser(description="Flow-map training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.jsonc")
    args = parser.parse_args(argv)

    # Load config
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_json_config(cfg_path)

    # Validate early
    if "paths" not in cfg or "processed_data_dir" not in cfg["paths"]:
        raise KeyError("cfg.paths.processed_data_dir is required")

    cfg.setdefault("system", {"seed": GLOBAL_SEED})
    cfg.setdefault("dataset", {})
    cfg.setdefault("mixed_precision", {"mode": "bf16"})

    # Setup work dir
    work_dir = Path(cfg.get("paths", {}).get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()

    # Setup logging first
    setup_logging(log_file=work_dir / "train.log", level=logging.INFO)
    logger = logging.getLogger("main")

    # Handle overwrite AFTER logger is ready (applies to work_dir only)
    if work_dir.exists() and bool(cfg.get("paths", {}).get("overwrite", False)):
        logger.warning(f"Deleting existing work dir: {work_dir}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Work directory: {work_dir}")

    # Seed and hardware setup
    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device, logger)

    # Precision + runtime dtype resolution
    pl_precision, runtime_dtype = resolve_precision_and_dtype(cfg, device, logger)
    logger.info(f"Resolved Lightning precision={pl_precision}; dataset runtime dtype={runtime_dtype}")

    # Preprocess if needed, then hydrate cfg from artifacts
    processed_dir = ensure_preprocessed_data(cfg, logger)
    assert processed_dir.exists(), "Processed data dir must exist after preprocessing/hydration"

    # Single config save after all hydration is complete
    dump_json(work_dir / "config.json", cfg)
    logger.info("Saved final hydrated config")

    # Build datasets/loaders
    train_ds, val_ds, train_loader, val_loader = build_datasets_and_loaders(
        cfg=cfg, device=device, runtime_dtype=runtime_dtype, logger=logger
    )
    validate_dataloaders(train_loader=train_loader, val_loader=val_loader, logger=logger)

    # Build model
    model = build_model(cfg, logger)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=logger.getChild("trainer"),
    )
    best_val_loss = trainer.train()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.6e}")

if __name__ == "__main__":
    main()
