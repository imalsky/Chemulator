#!/usr/bin/env python3
"""
Main entry point for training Flow-map DeepONet with multi-time-per-anchor support.

Pipeline
--------
1) Load config (config/config.jsonc relative to repo root)
2) Setup device + hardware knobs
3) Ensure preprocessed NPZ shards + normalization.json exist (run preprocessor if not)
4) Build datasets/dataloaders (optionally GPU-resident)
5) Create model (supports K times per anchor)
6) Train with shape-agnostic Trainer

This script is compatible with:
- model.py  (FlowMapDeepONet with K-time forward)
- dataset.py (FlowMapPairsDataset that can emit [B,K] times and [B,K,S] targets)
- trainer.py (shape-agnostic across [B,S] and [B,K,S])
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch

# Local modules
from utils import setup_logging, seed_everything, load_json_config, dump_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer

# ------------------------------ Globals --------------------------------------

# Repo root assumed to be the parent directory of 'src/'
REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed config path (no CLI args)
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"

# Global overrides (no CLI args)
GLOBAL_SEED: int = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "flowmap-deeponet"


# ------------------------------ Preprocess gate ------------------------------

def ensure_preprocessed(cfg: Dict[str, Any], log: logging.Logger) -> Path:
    """
    Idempotent: if normalization.json is missing under processed_data_dir, run preprocessor.
    Returns the processed_data_dir path.
    """
    pdir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    norm = pdir / "normalization.json"
    if norm.exists():
        log.info(f"Found normalization manifest at {norm}")
        return pdir

    log.info(f"No normalization manifest found at {norm} — running preprocessing…")
    pre = DataPreprocessor(cfg, logger=log.getChild("preprocessor"))
    pre.run()  # expected to write normalization.json and NPZ shards

    if not norm.exists():
        raise RuntimeError("Preprocessing reported success but normalization.json was not created.")
    log.info("Preprocessing complete.")
    # Persist an expanded/normalized copy of the config alongside artifacts
    dump_json(pdir / "config.snapshot.json", cfg)
    return pdir


# --------------------------- Datasets & Dataloaders --------------------------

def build_datasets_and_loaders(
    cfg: Dict[str, Any],
    device: torch.device,
    runtime_dtype: torch.dtype,
    log: logging.Logger,
) -> Tuple[FlowMapPairsDataset, FlowMapPairsDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dcfg = cfg.get("dataset", {})
    tcfg = cfg.get("training", {})

    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    # Train dataset
    train_ds = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="train",
        config=cfg,
        pairs_per_traj=int(tcfg.get("pairs_per_traj", 64)),
        min_steps=int(tcfg.get("min_steps", 1)),
        max_steps=int(tcfg.get("max_steps", 0)) or None,
        preload_to_gpu=bool(dcfg.get("preload_train_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=int(cfg.get("system", {}).get("seed", 42)),
    )

    # Validation dataset
    val_ds = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="validation",
        config=cfg,
        pairs_per_traj=int(tcfg.get("pairs_per_traj_val", tcfg.get("pairs_per_traj", 64))),
        min_steps=int(tcfg.get("min_steps", 1)),
        max_steps=int(tcfg.get("max_steps", 0)) or None,
        preload_to_gpu=bool(dcfg.get("preload_val_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=int(cfg.get("system", {}).get("seed", 42)) + 1337,
    )

    # Batch sizes
    bs_train = int(tcfg.get("batch_size", 512))
    bs_val = int(tcfg.get("val_batch_size", bs_train))

    # DataLoaders (dataset controls sampling; shuffling is redundant and disabled on GPU)
    train_loader = create_dataloader(
        dataset=train_ds,
        batch_size=bs_train,
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 0)),
        pin_memory=bool(dcfg.get("pin_memory", False)),
        prefetch_factor=int(dcfg.get("prefetch_factor", 2)),
        persistent_workers=bool(dcfg.get("persistent_workers", False)),
    )

    val_loader = create_dataloader(
        dataset=val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=int(dcfg.get("num_workers_val", dcfg.get("num_workers", 0))),
        pin_memory=bool(dcfg.get("pin_memory_val", dcfg.get("pin_memory", False))),
        prefetch_factor=int(dcfg.get("prefetch_factor_val", dcfg.get("prefetch_factor", 2))),
        persistent_workers=bool(dcfg.get("persistent_workers_val", dcfg.get("persistent_workers", False))),
    )

    # NEW: fast sanity checks so we fail loudly if a loader is empty
    try:
        n_train_items = len(train_loader.dataset)
        n_val_items = len(val_loader.dataset) if val_loader is not None else -1
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader is not None else -1
    except Exception:
        n_train_items = n_val_items = n_train_batches = n_val_batches = -1

    log.info(
        f"sanity | train_ds={n_train_items} items, train_loader={n_train_batches} batches | "
        f"val_ds={n_val_items} items, val_loader={n_val_batches} batches"
    )

    if n_train_batches == 0:
        raise RuntimeError(
            f"Train DataLoader is empty (len(dataset)={n_train_items}, batch_size={bs_train}, "
            f"pairs_per_traj={getattr(train_ds, 'pairs_per_traj', None)}). "
            f"Check training.pairs_per_traj (>0), shards under {processed_dir/'train'}, and split settings."
        )
    if val_loader is not None and n_val_batches == 0:
        raise RuntimeError(
            f"Validation DataLoader is empty (len(dataset)={n_val_items}, batch_size={bs_val}, "
            f"pairs_per_traj={getattr(val_ds, 'pairs_per_traj', None)}). "
            f"Check shards under {processed_dir/'validation'} and config."
        )

    # Log K-time settings for clarity
    log.info(
        f"Dataset config: multi_time_per_anchor={bool(dcfg.get('multi_time_per_anchor', False))}, "
        f"times_per_anchor={int(dcfg.get('times_per_anchor', 1))}, "
        f"share_times_across_batch={bool(dcfg.get('share_times_across_batch', False))}"
    )

    return train_ds, val_ds, train_loader, val_loader


# ---------------------------------- Model ------------------------------------

def build_model(cfg: Dict[str, Any], device: torch.device, log: logging.Logger) -> torch.nn.Module:
    model = create_model(cfg)
    model.to(device)
    mcfg = cfg.get("model", {})
    log.info(
        f"Model: p={int(mcfg.get('p', 256))}, "
        f"branch_width={int(mcfg.get('branch_width', 1024))}, "
        f"branch_depth={int(mcfg.get('branch_depth', 3))}, "
        f"trunk_layers={list(mcfg.get('trunk_layers', [512,512]))}, "
        f"predict_delta={bool(mcfg.get('predict_delta', True))}, "
        f"trunk_dedup={bool(mcfg.get('trunk_dedup', False))}"
    )
    return model


# --------------------------------- Trainer -----------------------------------

def build_trainer(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    work_dir: Path,
    device: torch.device,
    log: logging.Logger,
) -> Trainer:
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=log.getChild("trainer"),
    )


# ----------------------------------- Main ------------------------------------

def main() -> None:
    # Logging
    setup_logging(None)
    log = logging.getLogger("main")

    # Config (fixed path, no CLI)
    cfg = load_json_config(str(DEFAULT_CONFIG_PATH))

    # Apply global overrides for work_dir and seed
    cfg.setdefault("paths", {})["work_dir"] = str(GLOBAL_WORK_DIR)
    cfg.setdefault("system", {})["seed"] = int(GLOBAL_SEED)

    # Work directory
    work_dir = Path(cfg["paths"]["work_dir"]).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Seed + device + hardware knobs
    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device)

    # Optionally choose storage/runtime dtype for staged tensors
    ds_storage = str(cfg.get("dataset", {}).get("storage_dtype", "float32")).lower()
    if ds_storage == "bf16":
        runtime_dtype = torch.bfloat16
    elif ds_storage in ("fp16", "float16", "half"):
        runtime_dtype = torch.float16
    else:
        runtime_dtype = torch.float32
    log.info(f"Dataset storage/runtime dtype set to {runtime_dtype}")

    # Preprocess if needed
    processed_dir = ensure_preprocessed(cfg, log)
    assert processed_dir.exists(), "Processed data directory must exist after preprocessing."

    # Datasets & loaders
    train_ds, val_ds, train_loader, val_loader = build_datasets_and_loaders(cfg, device, runtime_dtype, log)

    # Model
    model = build_model(cfg, device, log)

    # Trainer
    trainer = build_trainer(cfg, model, train_loader, val_loader, work_dir, device, log)

    # Train
    best_val = trainer.train()
    log.info(f"Training complete. Best val loss: {best_val:.6e}")


if __name__ == "__main__":
    main()
