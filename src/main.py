#!/usr/bin/env python3
"""
Flow-map DeepONet Training Pipeline
====================================
Main entry point for training Flow-map DeepONet with multi-time-per-anchor support.

Pipeline:
1. Load configuration from config/config.jsonc
2. Setup device and optimize hardware settings
3. Ensure preprocessed data exists (run preprocessor if needed)
4. Build datasets and dataloaders (optionally GPU-resident)
5. Create model with K-time forward support
6. Train with shape-agnostic trainer
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

# Resolve duplicate library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from utils import setup_logging, seed_everything, load_json_config, dump_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer


# Configuration constants
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
GLOBAL_SEED = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "flowmap-deeponet"


def ensure_preprocessed_data(
    cfg: Dict[str, Any], 
    logger: logging.Logger
) -> Path:
    """
    Ensure preprocessed data exists. Run preprocessor if needed.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Path to processed data directory
        
    Raises:
        RuntimeError: If preprocessing fails to create normalization.json
    """
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    normalization_path = processed_dir / "normalization.json"
    
    if normalization_path.exists():
        logger.info(f"Found normalization manifest at {normalization_path}")
        return processed_dir

    logger.info(f"Normalization manifest not found at {normalization_path}")
    logger.info("Running preprocessing...")
    
    preprocessor = DataPreprocessor(cfg, logger=logger.getChild("preprocessor"))
    preprocessor.run()

    if not normalization_path.exists():
        raise RuntimeError(
            "Preprocessing completed but normalization.json was not created"
        )
        
    logger.info("Preprocessing complete")
    
    # Save configuration snapshot with processed data
    dump_json(processed_dir / "config.snapshot.json", cfg)
    return processed_dir


def build_datasets_and_loaders(
    cfg: Dict[str, Any],
    device: torch.device,
    runtime_dtype: torch.dtype,
    logger: logging.Logger,
) -> Tuple[FlowMapPairsDataset, FlowMapPairsDataset, 
           torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build training and validation datasets with their dataloaders.
    
    Supports GPU-resident datasets for high throughput when memory permits.
    
    Args:
        cfg: Configuration dictionary
        device: Target compute device
        runtime_dtype: Data type for staged tensors
        logger: Logger instance
        
    Returns:
        Tuple of (train_dataset, val_dataset, train_loader, val_loader)
        
    Raises:
        RuntimeError: If dataloaders are empty
    """
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    # Dataset parameters
    pairs_per_traj = int(training_cfg.get("pairs_per_traj", 64))
    pairs_per_traj_val = int(training_cfg.get("pairs_per_traj_val", pairs_per_traj))
    min_steps = int(training_cfg.get("min_steps", 1))
    max_steps = int(training_cfg.get("max_steps", 0)) or None
    base_seed = int(cfg.get("system", {}).get("seed", 42))

    # Create training dataset
    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="train",
        config=cfg,
        pairs_per_traj=pairs_per_traj,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=bool(dataset_cfg.get("preload_train_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=base_seed,
    )

    # Create validation dataset with different seed
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="validation",
        config=cfg,
        pairs_per_traj=pairs_per_traj_val,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=bool(dataset_cfg.get("preload_val_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=base_seed + 1337,
    )

    # Batch sizes
    batch_size_train = int(training_cfg.get("batch_size", 512))
    batch_size_val = int(training_cfg.get("val_batch_size", batch_size_train))

    # Create dataloaders
    # Note: Dataset handles sampling internally; shuffling disabled
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=bool(dataset_cfg.get("pin_memory", False)),
        prefetch_factor=int(dataset_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(dataset_cfg.get("persistent_workers", False)),
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers_val", 
                                       dataset_cfg.get("num_workers", 0))),
        pin_memory=bool(dataset_cfg.get("pin_memory_val", 
                                       dataset_cfg.get("pin_memory", False))),
        prefetch_factor=int(dataset_cfg.get("prefetch_factor_val", 
                                           dataset_cfg.get("prefetch_factor", 2))),
        persistent_workers=bool(dataset_cfg.get("persistent_workers_val", 
                                               dataset_cfg.get("persistent_workers", False))),
    )

    # Validate dataloaders contain data
    validate_dataloaders(
        train_loader, val_loader, 
        batch_size_train, batch_size_val, 
        processed_dir, logger
    )

    # Log multi-time configuration
    logger.info(
        f"Dataset configuration: "
        f"multi_time_per_anchor={bool(dataset_cfg.get('multi_time_per_anchor', False))}, "
        f"times_per_anchor={int(dataset_cfg.get('times_per_anchor', 1))}, "
        f"share_times_across_batch={bool(dataset_cfg.get('share_times_across_batch', False))}"
    )

    return train_dataset, val_dataset, train_loader, val_loader


def validate_dataloaders(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    batch_size_train: int,
    batch_size_val: int,
    processed_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Validate that dataloaders contain data.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        processed_dir: Path to processed data
        logger: Logger instance
        
    Raises:
        RuntimeError: If either dataloader is empty
    """
    try:
        n_train_items = len(train_loader.dataset)
        n_val_items = len(val_loader.dataset) if val_loader else 0
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
    except Exception:
        n_train_items = n_val_items = n_train_batches = n_val_batches = 0

    logger.info(
        f"Dataset statistics: "
        f"train={n_train_items} items, {n_train_batches} batches | "
        f"val={n_val_items} items, {n_val_batches} batches"
    )

    if n_train_batches == 0:
        raise RuntimeError(
            f"Training dataloader is empty. "
            f"Items={n_train_items}, batch_size={batch_size_train}. "
            f"Check training data in {processed_dir/'train'}"
        )
        
    if val_loader and n_val_batches == 0:
        raise RuntimeError(
            f"Validation dataloader is empty. "
            f"Items={n_val_items}, batch_size={batch_size_val}. "
            f"Check validation data in {processed_dir/'validation'}"
        )


def build_model(
    cfg: Dict[str, Any], 
    device: torch.device, 
    logger: logging.Logger
) -> torch.nn.Module:
    """
    Build and configure the Flow-map DeepONet model.
    
    Args:
        cfg: Configuration dictionary
        device: Target compute device
        logger: Logger instance
        
    Returns:
        Configured model on device
    """
    model = create_model(cfg)
    model.to(device)
    
    model_cfg = cfg.get("model", {})
    logger.info(
        f"Model architecture: "
        f"p={int(model_cfg.get('p', 256))}, "
        f"branch_width={int(model_cfg.get('branch_width', 1024))}, "
        f"branch_depth={int(model_cfg.get('branch_depth', 3))}, "
        f"trunk_layers={list(model_cfg.get('trunk_layers', [512, 512]))}, "
        f"predict_delta={bool(model_cfg.get('predict_delta', True))}, "
        f"trunk_dedup={bool(model_cfg.get('trunk_dedup', False))}"
    )
    return model


def main() -> None:
    """Main training pipeline."""
    # Initialize logging
    setup_logging(None)
    logger = logging.getLogger("main")

    # Load configuration
    cfg = load_json_config(str(DEFAULT_CONFIG_PATH))
    
    # Apply global configuration overrides
    cfg.setdefault("paths", {})["work_dir"] = str(GLOBAL_WORK_DIR)
    cfg.setdefault("system", {})["seed"] = GLOBAL_SEED

    # Setup work directory
    work_dir = Path(cfg["paths"]["work_dir"]).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize hardware and reproducibility
    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device)

    # Determine runtime data type for staged tensors
    storage_dtype_str = str(cfg.get("dataset", {}).get("storage_dtype", "float32")).lower()
    if storage_dtype_str == "bf16":
        runtime_dtype = torch.bfloat16
    elif storage_dtype_str in ("fp16", "float16", "half"):
        runtime_dtype = torch.float16
    else:
        runtime_dtype = torch.float32
    logger.info(f"Dataset storage/runtime dtype: {runtime_dtype}")

    # Ensure preprocessed data exists
    processed_dir = ensure_preprocessed_data(cfg, logger)
    assert processed_dir.exists(), "Processed data directory must exist after preprocessing"

    # Build datasets and dataloaders
    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders(
        cfg, device, runtime_dtype, logger
    )

    # Build model
    model = build_model(cfg, device, logger)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=logger.getChild("trainer"),
    )

    # Execute training
    best_val_loss = trainer.train()
    logger.info(f"Training complete. Best validation loss: {best_val_loss:.6e}")


if __name__ == "__main__":
    main()