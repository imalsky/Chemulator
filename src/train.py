#!/usr/bin/env python3
"""
train.py - Orchestrates the training, validation, and testing of a model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChemicalDataset
from hardware import configure_dataloader_settings
from model import create_prediction_model
from normalizer import DataNormalizer
from utils import parse_species_atoms, save_json

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42
DEFAULT_CACHE_SIZE = -1
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
DEFAULT_LR = 5e-4
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-8

class ModelTrainer:
    """Manages the entire training, validation, and testing pipeline."""

    def __init__(
        self, config: Dict[str, Any], device: torch.device, save_dir: Path,
        data_dir: Path, collate_fn: Callable, *,
        optuna_trial: Optional[optuna.Trial] = None
    ):
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.optuna_trial = optuna_trial
        
        self.data_spec = self.cfg["data_specification"]
        self.model_params = self.cfg["model_hyperparameters"]
        self.train_params = self.cfg["training_hyperparameters"]
        self.misc_cfg = self.cfg["miscellaneous_settings"]

        self._setup_normalization_and_datasets(data_dir)
        self._setup_conservation_loss()
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._setup_loss_and_training_params()
        self._setup_logging()
        self._save_test_set_info()

    def _setup_normalization_and_datasets(self, data_dir: Path) -> None:
        """Correctly splits data, calculates normalization, and creates datasets."""
        train_paths, val_paths, test_paths = _split_profile_paths(
            data_dir, self.misc_cfg["val_frac"], self.misc_cfg["test_frac"],
            seed=self.misc_cfg.get("random_seed", DEFAULT_RANDOM_SEED)
        )
        self.test_filenames = [p.name for p in test_paths]
        
        logger.info("Calculating normalization stats from the training set ONLY.")
        try:
            normalizer = DataNormalizer(config_data=self.cfg)
            self.norm_metadata = normalizer.calculate_stats(train_paths)
        except ValueError as e:
            logger.error(f"FATAL: Could not calculate normalization stats. Error: {e}")
            raise RuntimeError("Halting due to data normalization error.") from e
            
        save_json(self.norm_metadata, self.save_dir / "normalization_metadata.json")

        ds_args = {
            "data_folder": data_dir,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "cache_size": self.misc_cfg.get("dataset_cache_size", DEFAULT_CACHE_SIZE),
        }
        self.train_ds = ChemicalDataset(**ds_args, profile_paths=train_paths)
        self.val_ds = ChemicalDataset(**ds_args, profile_paths=val_paths)
        self.test_ds = ChemicalDataset(**ds_args, profile_paths=test_paths)

    def _setup_conservation_loss(self) -> None:
        """Sets up the matrix for the physics-informed atom conservation loss."""
        self.use_conservation_loss = self.train_params.get("use_conservation_loss", False)
        if not self.use_conservation_loss: return
        
        logger.info("Setting up physics-informed atom conservation loss.")
        self.species_vars = sorted(self.data_spec["species_variables"])
        atom_matrix, self.atom_names = parse_species_atoms(self.species_vars)
        if not self.atom_names:
            logger.warning("No atoms parsed from species. Disabling conservation loss.")
            self.use_conservation_loss = False
            return
        self.atom_matrix = torch.tensor(atom_matrix, dtype=torch.float32, device=self.device)
        logger.info(f"Atom conservation will be enforced for: {self.atom_names}")

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        hw_settings = configure_dataloader_settings()
        dl_args = {
            "batch_size": self.train_params.get("batch_size", DEFAULT_BATCH_SIZE),
            "num_workers": self.misc_cfg.get("num_workers", 0),
            "collate_fn": collate_fn,
            "pin_memory": hw_settings.get("pin_memory", False),
            "persistent_workers": hw_settings.get("persistent_workers", False) and self.misc_cfg.get("num_workers", 0) > 0,
        }
        self.train_loader = DataLoader(self.train_ds, shuffle=True, drop_last=True, **dl_args)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **dl_args)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **dl_args)

    def _build_model(self) -> None:
        self.model = create_prediction_model(self.cfg, device=self.device)
        if self.misc_cfg.get("use_torch_compile", False):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile().")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Using uncompiled model.")

    def _build_optimizer(self) -> None:
        opt_name = self.train_params.get("optimizer", DEFAULT_OPTIMIZER).lower()
        opt_map = {"adamw": optim.AdamW, "adam": optim.Adam, "rmsprop": optim.RMSprop}
        opt_class = opt_map.get(opt_name, optim.AdamW)
        self.optimizer = opt_class(
            self.model.parameters(),
            lr=self.train_params.get("learning_rate", DEFAULT_LR),
            weight_decay=self.train_params.get("weight_decay", 1e-5),
        )

    def _build_scheduler(self) -> None:
        name = self.train_params.get("scheduler_choice", "plateau").lower()
        if name == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 
                                               factor=self.train_params.get("lr_factor", 0.5), 
                                               patience=self.train_params.get("lr_patience", 10),
                                               min_lr=self.train_params.get("min_lr", 1e-7))
        elif name == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 
                                                       T_0=self.train_params.get("cosine_T_0", 10), 
                                                       T_mult=self.train_params.get("cosine_T_mult", 2))
        else:
            raise ValueError(f"Unsupported scheduler: '{name}'.")

    def _setup_loss_and_training_params(self) -> None:
        loss_name = self.train_params.get("loss_function", "mse").lower()
        if loss_name == "mse": self.criterion = nn.MSELoss()
        elif loss_name == "huber": self.criterion = nn.HuberLoss(delta=self.train_params.get("huber_delta", 0.1))
        elif loss_name == "l1": self.criterion = nn.L1Loss()
        else: raise ValueError(f"Unsupported loss function: {loss_name}")
        
        self.use_amp = self.train_params.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = self.train_params.get("gradient_clip_val", DEFAULT_GRAD_CLIP)

    def _setup_logging(self) -> None:
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s\n")
        self.best_val_loss, self.best_epoch = float("inf"), -1

    def train(self) -> float:
        epochs = self.train_params.get("epochs", DEFAULT_EPOCHS)
        patience = self.train_params.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = self.train_params.get("min_delta", DEFAULT_MIN_DELTA)
        epochs_without_improvement = 0

        logger.info(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            train_loss = self._run_epoch(self.train_loader, is_train_phase=True)
            val_loss = self._run_epoch(self.val_loader, is_train_phase=False)
            
            if isinstance(self.scheduler, ReduceLROnPlateau): self.scheduler.step(val_loss)
            else: self.scheduler.step()

            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune(): raise TrialPruned()

            self._log_epoch_results(epoch, train_loss, val_loss, time.time() - start_time)

            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        
        self._checkpoint("final_model.pt", epoch, val_loss)
        self.test()
        return self.best_val_loss

    def _log_epoch_results(self, epoch: int, train_loss: float, val_loss: float, duration: float) -> None:
        lr = self.optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | LR: {lr:.2e} | Time: {duration:.1f}s"
        if self.best_epoch == epoch: log_msg += " | ★ New best!"
        logger.info(log_msg)
        with self.log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e},{lr:.6e},{duration:.1f}\n")

    def _run_epoch(self, loader: DataLoader, is_train_phase: bool) -> float:
        self.model.train(is_train_phase)
        total_loss, non_finite_count = 0.0, 0
        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        progress_bar = tqdm(loader, desc=desc, leave=False, disable=not self.misc_cfg.get("show_epoch_progress", False))
        context = torch.set_grad_enabled(is_train_phase) if is_train_phase else torch.inference_mode()
        with context:
            for inputs_dict, targets in progress_bar:
                if inputs_dict["x"].numel() == 0: continue
                inputs = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.bfloat16 if self.device.type != 'mps' else torch.float32, enabled=self.use_amp):
                    preds = self.model(inputs)
                    prediction_loss = self.criterion(preds, targets)
                    if not torch.isfinite(prediction_loss):
                        non_finite_count += 1
                        logger.warning(f"Non-finite loss detected. Skipping batch.")
                        continue
                    batch_loss = prediction_loss
                    if is_train_phase and self.use_conservation_loss:
                        cons_loss = self._calculate_conservation_loss(inputs[:, :len(self.species_vars)], preds)
                        batch_loss += self.train_params.get("conservation_loss_weight", 1.0) * cons_loss
                if is_train_phase:
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                total_loss += prediction_loss.item()
        if non_finite_count > len(loader) * 0.1:
             raise RuntimeError(f"Excessive non-finite losses ({non_finite_count} batches) encountered. Training is unstable.")
        return total_loss / len(loader) if len(loader) > 0 else float('inf')

    # CORRECTED: This version is much more efficient.
    def _calculate_conservation_loss(self, initial_norm: Tensor, predicted_norm: Tensor) -> Tensor:
        """Calculates atom conservation error using efficient batch denormalization."""
        def batch_denormalize(normalized_batch: Tensor, var_names: List[str]) -> Tensor:
            """Helper to denormalize a batch tensor one variable (column) at a time."""
            denorm_cols = []
            for i, var_name in enumerate(var_names):
                # Pass the entire column (all batch items for one variable)
                col_denorm = DataNormalizer.denormalize(
                    normalized_batch[:, i],
                    self.norm_metadata,
                    var_name
                )
                denorm_cols.append(col_denorm)
            return torch.stack(denorm_cols, dim=1)
        
        denorm_pred = batch_denormalize(predicted_norm, self.species_vars)
        denorm_initial = batch_denormalize(initial_norm, self.species_vars)
        
        initial_atoms = torch.matmul(denorm_initial, self.atom_matrix)
        predicted_atoms = torch.matmul(denorm_pred, self.atom_matrix)
        return torch.nn.functional.mse_loss(predicted_atoms, initial_atoms)

    def _checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        checkpoint = {"state_dict": model_to_save.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": self.cfg}
        torch.save(checkpoint, self.save_dir / filename)
        logger.info(f"Saved checkpoint: {filename}")

    def test(self) -> None:
        ckpt_path = self.save_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("No 'best_model.pt' found. Skipping test phase.")
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']} for testing.")
        test_loss = self._run_epoch(self.test_loader, is_train_phase=False)
        metrics = {"test_loss": test_loss}
        logger.info(f"Test Results: Loss = {test_loss:.4e}")
        save_json(metrics, self.save_dir / "test_metrics.json")
    
    def _save_test_set_info(self) -> None:
        save_json({"test_filenames": sorted(self.test_filenames)}, self.save_dir / "test_set_info.json")

def _split_profile_paths(
    data_dir: Path, val_frac: float, test_frac: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Deterministically splits raw profile file paths into train, val, and test sets."""
    all_profiles = sorted(data_dir.glob("*.json"))
    if not all_profiles:
        raise FileNotFoundError(f"No raw profiles ('*.json') found in {data_dir}")
    n = len(all_profiles)
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("Dataset split fractions are invalid (must sum to < 1).")
    num_val, num_test = int(n * val_frac), int(n * test_frac)
    num_train = n - num_val - num_test
    if num_train <= 0 or num_val <= 0 or num_test <= 0:
        raise ValueError(f"Invalid split for {n} samples: {num_train}/{num_val}/{num_test}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_paths = [all_profiles[i] for i in indices[:num_train]]
    val_paths = [all_profiles[i] for i in indices[num_train : num_train + num_val]]
    test_paths = [all_profiles[i] for i in indices[num_train + num_val :]]
    logger.info(f"Raw profiles split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test.")
    return train_paths, val_paths, test_paths

__all__ = ["ModelTrainer"]