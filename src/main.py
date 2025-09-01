#!/usr/bin/env python3
"""
Main entry point for training Flow-map DeepONet using absolute normalized time.
"""
import argparse
import logging
from pathlib import Path

import torch

from utils import setup_logging, seed_everything, load_json_config
from hardware import setup_device, optimize_hardware
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Path to JSON/JSONC config")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_json_config(args.config)

    # Logging & reproducibility
    log_dir = Path(cfg.get("paths", {}).get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_dir / "train.log")
    seed_everything(int(cfg.get("system", {}).get("seed", 42)))
    log = logging.getLogger("main")

    # Device and hardware knobs
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device, logger=log)

    # Dims / types
    dtype = torch.bfloat16 if bool(cfg.get("system", {}).get("use_bfloat16", True)) else torch.float32

    # Data
    data_dir = Path(cfg["paths"]["processed_data_dir"])
    train_pairs = int(cfg.get("training", {}).get("pairs_per_traj", 64))
    batch_size = int(cfg.get("training", {}).get("batch_size", 256))
    num_workers = int(cfg.get("training", {}).get("num_workers", 8))
    min_steps = int(cfg.get("training", {}).get("min_steps", 2))
    max_steps = int(cfg.get("training", {}).get("max_steps", 100))

    train_ds = FlowMapPairsDataset(
        processed_root=data_dir,
        split="train",
        config=cfg,
        pairs_per_traj=train_pairs,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=True,  # GPU-resident for A100
        device=device,
        dtype=dtype,
        seed=int(cfg.get("system", {}).get("seed", 42)),
    )

    val_ds = FlowMapPairsDataset(
        processed_root=data_dir,
        split="validation",  # match preprocessor split name
        config=cfg,
        pairs_per_traj=max(1, train_pairs // 2),
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=True,
        device=device,
        dtype=dtype,
        seed=int(cfg.get("system", {}).get("seed", 42) + 1),
    )

    train_loader = create_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, num_workers=num_workers)

    # Model
    model = create_model(cfg, device=device)

    # Optimizer
    lr = float(cfg.get("training", {}).get("lr", 3e-4))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Trainer
    work_dir = Path(cfg.get("paths", {}).get("model_save_dir", "models")) / cfg.get("model", {}).get("name", "deeponet")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        work_dir=work_dir,
        device=device,
        epochs=int(cfg.get("training", {}).get("epochs", 100)),
        use_amp_bf16=bool(cfg.get("system", {}).get("use_bfloat16", True)),
        min_lr=float(cfg.get("training", {}).get("min_lr", 1e-6)),
    )

    best = trainer.train()
    log.info(f"Training complete. Best val loss: {best:.6e}")


if __name__ == "__main__":
    main()