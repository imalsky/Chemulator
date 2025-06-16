#!/usr/bin/env python3
"""
train.py - Main training script for the State-Evolution Predictor.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import optuna
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from dataset import ChemicalDataset
from hardware import configure_dataloader_settings
from model import create_prediction_model
from utils import save_json

logger = logging.getLogger(__name__)

def _split_profiles(
    data_dir: Path, val_frac: float, test_frac: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Discovers and splits profile files into train, validation, and test sets.
    This method is crucial for preventing data leakage by ensuring that all
    time-points from a single profile belong to only one set.
    """
    profiles = sorted([p for p in data_dir.glob("*.json") if p.name != "normalization_metadata.json"])
    if not profiles:
        raise FileNotFoundError(f"No profiles found in {data_dir} to split.")

    n = len(profiles)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    
    num_val = int(n * val_frac)
    num_test = int(n * test_frac)
    
    test_indices = indices[:num_test]
    val_indices = indices[num_test : num_test + num_val]
    train_indices = indices[num_test + num_val :]

    train_paths = [profiles[i] for i in train_indices]
    val_paths = [profiles[i] for i in val_indices]
    test_paths = [profiles[i] for i in test_indices]

    logger.info(f"Profiles split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test.")
    return train_paths, val_paths, test_paths

class ModelTrainer:
    def __init__(
        self, 
        config: Dict[str, Any], 
        device: torch.device, 
        save_dir: Path, 
        data_dir: Path,
        collate_fn: Callable, 
        *, 
        optuna_trial: Optional[optuna.Trial] = None
    ):
        self.cfg, self.device, self.save_dir, self.optuna_trial = config, device, save_dir, optuna_trial
        
        # Robustly split data by profile files to prevent leakage
        train_paths, val_paths, test_paths = _split_profiles(
            data_dir, self.cfg["val_frac"], self.cfg["test_frac"], seed=self.cfg.get("random_seed", 42)
        )
        self.test_filenames = [p.name for p in test_paths]
        
        # Create separate dataset instances for each split
        dataset_args = {
            "data_folder": data_dir,
            "species_variables": self.cfg["species_variables"],
            "global_variables": self.cfg["global_variables"],
        }
        self.train_ds = ChemicalDataset(**dataset_args, profile_paths=train_paths)
        self.val_ds = ChemicalDataset(**dataset_args, profile_paths=val_paths)
        self.test_ds = ChemicalDataset(**dataset_args, profile_paths=test_paths)
        
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimiser()
        self._build_scheduler()
        self.use_amp = bool(self.cfg.get("use_amp") and device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp: logger.info("Automatic Mixed Precision (AMP) enabled.")
        
        self.criterion = nn.MSELoss()
        self.max_grad_norm = self.cfg.get("gradient_clip_val", 1.0)
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s\n")
        self.best_val_loss = float("inf")
        
        self._save_test_set_info()

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        hw_settings = configure_dataloader_settings()
        num_workers = 4 if self.device.type == 'cuda' else 0
        dl_args = dict(batch_size=self.cfg["batch_size"], num_workers=num_workers, pin_memory=hw_settings.get("pin_memory", False), persistent_workers=hw_settings.get("persistent_workers", False) and num_workers > 0, collate_fn=collate_fn)
        # We shuffle the DataLoader, which now shuffles the flattened time-points.
        # This is the desired behavior for stochastic gradient descent.
        self.train_loader = DataLoader(self.train_ds, shuffle=True, **dl_args)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **dl_args)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **dl_args)
        logger.info(f"DataLoaders created with {num_workers} workers.")

    def _build_model(self) -> None:
        self.model = create_prediction_model(self.cfg, self.device)
        if self.cfg.get("use_torch_compile") and self.device.type == "cuda":
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

    def _build_optimiser(self) -> None:
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg.get("weight_decay", 0.0))

    def _build_scheduler(self) -> None:
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.cfg.get("lr_factor", 0.1), patience=self.cfg.get("lr_patience", 10))

    def _run_epoch(self, loader: DataLoader, train_phase: bool) -> float:
        self.model.train(train_phase)
        total_loss = 0.0
        
        desc = f"Epoch {getattr(self, 'current_epoch', 0):03d} [{'Train' if train_phase else 'Val'}]"
        pbar = tqdm(loader, desc=desc, leave=False)

        with torch.set_grad_enabled(train_phase):
            for inputs_dict, targets in pbar:
                input_tensor = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if train_phase: self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(self.device.type, enabled=self.use_amp):
                    predictions = self.model(input_tensor)
                    loss = self.criterion(predictions, targets)

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected ({loss.item()}). Skipping batch.")
                    continue

                if train_phase:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        logger.warning(f"Non-finite gradient norm detected ({grad_norm}). Skipping optimizer step.")
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                
        return total_loss / len(loader) if loader else 0.0

    def train(self) -> float:
        epochs_without_improvement = 0
        final_epoch = 0
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.current_epoch = epoch
            final_epoch = epoch
            start_time = time.time()
            
            train_loss = self._run_epoch(self.train_loader, train_phase=True)
            val_loss = self._run_epoch(self.val_loader, train_phase=False)
            
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            
            log_message = (
                f"Epoch {epoch:03d}/{self.cfg['epochs']} | "
                f"Train Loss: {train_loss:.4e} | "
                f"Val Loss: {val_loss:.4e} | "
                f"LR: {lr:.2e} | "
                f"Time: {time.time() - start_time:.1f}s"
            )

            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                logger.critical(f"Epoch {epoch:03d} failed due to non-finite loss. Stopping.")
                break

            if val_loss < self.best_val_loss - self.cfg.get("min_delta", 1e-6):
                log_message += " | New Best!"
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
            else:
                epochs_without_improvement += 1
            
            logger.info(log_message)
            
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e},{lr:.6e},{time.time() - start_time:.1f}\n")

            if epochs_without_improvement >= self.cfg.get("early_stopping_patience", 10):
                logger.info(f"Early stopping at epoch {epoch}."); break
        
        self._checkpoint("final_model.pt", final_epoch, val_loss, final=True)
        self.test()
        return self.best_val_loss

    def _checkpoint(self, filename: str, epoch: int, val_loss: float, *, final: bool = False):
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save({"state_dict": model_to_save.state_dict(), "epoch": epoch, "val_loss": val_loss}, self.save_dir / filename)
        jit_save_path = self.save_dir / (Path(filename).stem + "_jit.pt")
        try:
            torch.jit.script(model_to_save).save(str(jit_save_path))
        except Exception as e:
            logger.error(f"Failed to JIT-save model: {e}")

    def test(self) -> None:
        best_model_path = self.save_dir / "best_model.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', -1)} for testing.")
        else:
            logger.warning("No best model found. Testing with final model state.")
            
        test_loss = self._run_epoch(self.test_loader, train_phase=False)
        logger.info(f"✅ Final Test Loss: {test_loss:.4e}")
        save_json({"test_loss": test_loss}, self.save_dir / "test_metrics.json")
    
    def _save_test_set_info(self) -> None:
        """Saves the list of test set filenames to the model directory."""
        save_path = self.save_dir / "test_set_info.json"
        save_json({"test_filenames": sorted(self.test_filenames)}, save_path)
        logger.info(f"Test set filenames saved to {save_path}")

__all__ = ["ModelTrainer"]