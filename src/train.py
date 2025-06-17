#!/usr/b-in/env python3
"""
train.py - Main training script for the State-Evolution Predictor.
This version dynamically creates components based on the configuration.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
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
    """
    Manages the entire training, validation, and testing pipeline for a model.

    This class encapsulates all the components required for a training run,
    including data loading, model building, optimization, and logging. It is
    driven by a configuration dictionary.
    """
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
        """
        Initializes the ModelTrainer.

        Args:
            config (Dict[str, Any]): The configuration dictionary for the run.
            device (torch.device): The device (CPU or CUDA) to run on.
            save_dir (Path): The directory to save logs, models, and results.
            data_dir (Path): The directory containing the normalized dataset.
            collate_fn (Callable): The function to collate data into batches.
            optuna_trial (Optional[optuna.Trial]): An Optuna trial object for hyperparameter tuning.
        """
        self.cfg, self.device, self.save_dir, self.optuna_trial = config, device, save_dir, optuna_trial
        
        train_paths, val_paths, test_paths = _split_profiles(
            data_dir, self.cfg["val_frac"], self.cfg["test_frac"], seed=self.cfg.get("random_seed", 42)
        )
        self.test_filenames = [p.name for p in test_paths]
        
        dataset_args = {
            "data_folder": data_dir,
            "species_variables": self.cfg["species_variables"],
            "global_variables": self.cfg["global_variables"],
            "cache_size": self.cfg.get("dataset_cache_size", -1),
        }
        self.train_ds = ChemicalDataset(**dataset_args, profile_paths=train_paths)
        self.val_ds = ChemicalDataset(**dataset_args, profile_paths=val_paths)
        self.test_ds = ChemicalDataset(**dataset_args, profile_paths=test_paths)
        
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimiser()
        
        self.use_amp = bool(self.cfg.get("use_amp") and device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp: logger.info("Automatic Mixed Precision (AMP) enabled.")
        
        loss_function_name = self.cfg.get("loss_function", "huber").lower()
        if loss_function_name == "huber":
            delta = self.cfg.get("huber_delta", 0.1)
            self.criterion = nn.HuberLoss(delta=delta)
            logger.info(f"Using Huber loss with delta={delta}")
        elif loss_function_name == "mse":
            self.criterion = nn.MSELoss()
            logger.info("Using Mean Squared Error (MSE) loss")
        elif loss_function_name == "l1":
            self.criterion = nn.L1Loss()
            logger.info("Using L1 (Mean Absolute Error) loss")
        else:
            raise ValueError(f"Unsupported loss function: '{loss_function_name}'")

        self.max_grad_norm = self.cfg.get("gradient_clip_val", 1.0)
        self.non_finite_grad_threshold = self.cfg.get("non_finite_grad_threshold", 10)
        self.accumulation_steps = self.cfg.get("gradient_accumulation_steps", 1)
        self.warmup_epochs = self.cfg.get("warmup_epochs", 5)
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.min_delta = self.cfg.get("min_delta", 1e-10)

        # Build scheduler after its dependencies (optimizer, min_delta) are set
        self._build_scheduler()
        
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,grad_norm\n")
        self.best_val_loss = float("inf")

        self._save_test_set_info()

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Configures and creates DataLoaders for train, validation, and test sets."""
        hw_settings = configure_dataloader_settings()
        num_workers = self.cfg.get("num_workers", 4 if self.device.type == 'cuda' else 0)
        
        dl_args = dict(
            batch_size=self.cfg["batch_size"], 
            num_workers=num_workers, 
            pin_memory=hw_settings.get("pin_memory", False), 
            persistent_workers=hw_settings.get("persistent_workers", False) and num_workers > 0, 
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.train_loader = DataLoader(self.train_ds, shuffle=True, **dl_args)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **dl_args)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **dl_args)
        logger.info(f"DataLoaders created with batch_size={self.cfg['batch_size']}, num_workers={num_workers}")

    def _build_model(self) -> None:
        """Builds the prediction model based on the configuration."""
        self.model = create_prediction_model(self.cfg, self.device)
        if self.cfg.get("use_torch_compile") and self.device.type == "cuda":
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

    def _build_optimiser(self) -> None:
        """Builds the optimizer (e.g., AdamW) based on the configuration."""
        optimizer_name = self.cfg.get("optimizer", "adamw").lower()
        lr = self.cfg["learning_rate"]
        weight_decay = self.cfg.get("weight_decay", 0.01)

        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")
        
        logger.info(f"Using {optimizer_name.upper()} optimizer with lr={lr:.2e}, weight_decay={weight_decay:.2e}")

    def _build_scheduler(self) -> None:
        """Builds the learning rate scheduler (e.g., ReduceLROnPlateau)."""
        scheduler_name = self.cfg.get("scheduler_choice", "plateau").lower()
        
        if scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 'min', 
                factor=self.cfg.get("lr_factor", 0.5),
                patience=self.cfg.get("lr_patience", 10),
                min_lr=self.cfg.get("min_lr", 1e-7),
                threshold=self.min_delta, # Use min_delta from config
                threshold_mode='abs'      # Use absolute difference for comparison
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        elif scheduler_name == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.cfg.get("cosine_T_0", 10), 
                T_mult=self.cfg.get("cosine_T_mult", 2)
            )
            logger.info("Using CosineAnnealingWarmRestarts scheduler.")
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'")

    def _warmup_lr(self, epoch: int) -> None:
        """
        Linearly warms up the learning rate for the initial epochs.
        
        Args:
            epoch (int): The current epoch number (1-based).
        """
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor

    def _run_epoch(self, loader: DataLoader, train_phase: bool) -> Tuple[float, float]:
        """
        Runs a single epoch of training or validation.

        Args:
            loader (DataLoader): The DataLoader for the current phase.
            train_phase (bool): True if in training mode, False for validation.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and average gradient norm.
                                 Gradient norm is 0 for the validation phase.
        """
        self.model.train(train_phase)
        total_loss, total_grad_norm, batch_count, non_finite_batch_count = 0.0, 0.0, 0, 0
        
        show_progress = self.cfg.get("show_epoch_progress", False)
        desc = f"Epoch {getattr(self, 'current_epoch', 0):03d} [{'Train' if train_phase else 'Val'}]"
        iterable = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
        
        if train_phase: self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_phase):
            for i, (inputs_dict, targets) in enumerate(iterable):
                input_tensor = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.autocast(self.device.type, enabled=self.use_amp):
                    predictions = self.model(input_tensor)
                    loss = self.criterion(predictions, targets)
                    if train_phase: loss = loss / self.accumulation_steps

                if not torch.isfinite(loss):
                    logger.warning(f"Epoch {self.current_epoch}, Batch {i}: Non-finite loss detected. Skipping.")
                    non_finite_batch_count += 1
                    if train_phase:
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.use_amp: self.scaler.update()
                    if non_finite_batch_count > self.non_finite_grad_threshold:
                        return float('inf'), float('inf')
                    continue

                if train_phase:
                    self.scaler.scale(loss).backward()
                    if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if torch.isfinite(grad_norm):
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            total_grad_norm += grad_norm.item()
                            batch_count += 1
                        else:
                            non_finite_batch_count += 1
                            logger.warning(f"Epoch {self.current_epoch}, Batch {i}: Non-finite gradient. Skipping step.")
                            if self.use_amp: self.scaler.update()
                            if non_finite_batch_count > self.non_finite_grad_threshold:
                                return float('inf'), float('inf')
                        self.optimizer.zero_grad(set_to_none=True)

                actual_loss = loss.item() * (self.accumulation_steps if train_phase else 1)
                total_loss += actual_loss
                if show_progress:
                    grad_norm_val = grad_norm.item() if train_phase and 'grad_norm' in locals() else 0.0
                    iterable.set_postfix(loss=f"{actual_loss:.4e}", grad_norm=f"{grad_norm_val:.3f}")
                
        return total_loss / len(loader), total_grad_norm / batch_count if batch_count > 0 else 0.0

    def train(self) -> float:
        """
        Executes the main training loop over all epochs.

        This method handles:
        - Iterating through epochs.
        - Calling the training and validation epoch runs.
        - Stepping the learning rate scheduler.
        - Logging results to console and file.
        - Checkpointing the best model.
        - Handling Optuna pruning.
        - Triggering early stopping.
        - Running final testing after the loop.

        Returns:
            float: The best validation loss achieved during training.
        """
        final_epoch = 0
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.current_epoch, start_time = epoch, time.time()
            if epoch <= self.warmup_epochs: self._warmup_lr(epoch)
            
            train_loss, train_grad_norm = self._run_epoch(self.train_loader, train_phase=True)
            if not np.isfinite(train_loss):
                logger.critical(f"Epoch {epoch:03d} failed due to instability. Stopping training.")
                return float('inf')

            val_loss, _ = self._run_epoch(self.val_loader, train_phase=False)
            
            if isinstance(self.scheduler, ReduceLROnPlateau): self.scheduler.step(val_loss)
            else: self.scheduler.step()
            
            lr, elapsed_time = self.optimizer.param_groups[0]['lr'], time.time() - start_time
            log_message = f"Epoch {epoch:03d}/{self.cfg['epochs']} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | Grad Norm: {train_grad_norm:.3f} | LR: {lr:.2e} | Time: {elapsed_time:.1f}s"
            
            if val_loss < self.best_val_loss - self.min_delta:
                log_message += " | New Best!"
                self.best_val_loss = val_loss
                self._checkpoint("best_model.pt", epoch, val_loss)
            
            logger.info(log_message)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e},{lr:.6e},{elapsed_time:.1f},{train_grad_norm:.4f}\n")

            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune(): raise TrialPruned()
            
            epochs_without_improvement = self.scheduler.num_bad_epochs if isinstance(self.scheduler, ReduceLROnPlateau) else 0
            if epochs_without_improvement >= self.cfg.get("early_stopping_patience", 30):
                logger.info(f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} epochs with no improvement.")
                break
            
            if val_loss < self.cfg.get("target_loss", -1):
                logger.info("Target loss achieved! Stopping training.")
                break

            final_epoch = epoch
        
        self._checkpoint("final_model.pt", final_epoch, val_loss, final=True)
        self.test()
        return self.best_val_loss

    def _checkpoint(self, filename: str, epoch: int, val_loss: float, *, final: bool = False):
        """
        Saves a model checkpoint.

        Saves both the regular PyTorch model state and a JIT-scripted version
        for deployment.

        Args:
            filename (str): The name of the file to save the model to.
            epoch (int): The current epoch number.
            val_loss (float): The validation loss at this checkpoint.
            final (bool, optional): If True, indicates this is the final model save.
                                    Defaults to False.
        """
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        torch.save({
            "state_dict": model_to_save.state_dict(), "epoch": epoch, "val_loss": val_loss,
            "optimizer_state": self.optimizer.state_dict(), "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None, "config": self.cfg,
        }, self.save_dir / filename)
        
        jit_save_path = self.save_dir / (Path(filename).stem + "_jit.pt")
        try:
            original_device = self.device
            model_to_save.to('cpu').eval()
            # Create an example input based on model's expected input feature count
            num_input_features = len(self.cfg["species_variables"]) + len(self.cfg["global_variables"]) + 1 # +1 for time
            example_input = torch.randn(1, num_input_features, device='cpu')
            traced_model = torch.jit.trace(model_to_save, example_input)
            torch.jit.save(traced_model, str(jit_save_path))
            model_to_save.to(original_device).train()
        except Exception as e:
            logger.error(f"Failed to JIT-save model: {e}")

    def test(self) -> None:
        """
        Evaluates the best-performing model on the test set.

        Loads the checkpoint with the best validation loss and computes final
        test metrics, saving them to a JSON file.
        """
        best_model_path = self.save_dir / "best_model.pt"
        if not best_model_path.exists():
            logger.warning("No best model found. Testing with final model state.")
            return

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        model_to_load = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model_to_load.load_state_dict(checkpoint.get("state_dict", checkpoint))
        logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', -1)} for testing.")
            
        self.model.eval()
        total_loss, all_errors = 0.0, []
        show_progress = self.cfg.get("show_epoch_progress", False)
        test_iterable = tqdm(self.test_loader, desc="Testing", leave=False, disable=not show_progress)
        
        with torch.no_grad():
            for inputs_dict, targets in test_iterable:
                inputs, targets = inputs_dict['x'].to(self.device), targets.to(self.device)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                all_errors.append(torch.abs(predictions - targets))
        
        test_loss = total_loss / len(self.test_loader)
        all_errors = torch.cat(all_errors, dim=0)
        per_species_mae = all_errors.mean(dim=0).cpu().numpy()
        per_species_max = all_errors.max(dim=0).values.cpu().numpy()
        
        test_metrics = {
            "test_loss": float(test_loss), "test_mae": float(all_errors.mean().item()),
            "test_max_error": float(all_errors.max().item()), "test_samples": len(self.test_ds),
            "per_species_mae": {var: float(v) for var, v in zip(self.cfg["species_variables"], per_species_mae)},
            "per_species_max_error": {var: float(v) for var, v in zip(self.cfg["species_variables"], per_species_max)},
        }
        logger.info(f"Test Results - Loss: {test_metrics['test_loss']:.4e}, MAE: {test_metrics['test_mae']:.4e}, Max Error: {test_metrics['test_max_error']:.4e}")
        save_json(test_metrics, self.save_dir / "test_metrics.json")
    
    def _save_test_set_info(self) -> None:
        """Saves the list of filenames used in the test set to a JSON file."""
        save_json({"test_filenames": sorted(self.test_filenames)}, self.save_dir / "test_set_info.json")
        logger.info(f"Test set filenames saved to {self.save_dir / 'test_set_info.json'}")

__all__ = ["ModelTrainer"]