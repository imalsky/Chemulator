#!/usr/bin/env python3
"""
train.py - Training script
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChemicalDataset
from hardware import configure_dataloader_settings
from model import create_prediction_model
# MODIFICATION: Import denormalizer and atom parser
from normalizer import DataNormalizer
from utils import save_json, parse_species_atoms

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_CACHE_SIZE = -1
DEFAULT_NUM_WORKERS = 4
DEFAULT_HUBER_DELTA = 0.1
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_NON_FINITE_THRESHOLD = 10
DEFAULT_ACCUMULATION_STEPS = 1
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_MIN_DELTA = 1e-10
DEFAULT_LR_FACTOR = 0.5
DEFAULT_LR_PATIENCE = 10
DEFAULT_MIN_LR = 1e-7
DEFAULT_COSINE_T0 = 10
DEFAULT_COSINE_T_MULT = 2
DEFAULT_EARLY_STOPPING_PATIENCE = 30
DEFAULT_TARGET_LOSS = -1
NORMALIZATION_METADATA_FILE = "normalization_metadata.json"

logger = logging.getLogger(__name__)


def _split_profiles(
    data_dir: Path, val_frac: float, test_frac: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Discovers and splits profile files into train, validation, and test sets.
    """
    profiles = sorted([
        p for p in data_dir.glob("*.json") 
        if p.name != NORMALIZATION_METADATA_FILE
    ])
    
    if not profiles:
        raise FileNotFoundError(f"No profiles found in {data_dir} to split.")

    n = len(profiles)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    
    num_val = int(n * val_frac)
    num_test = int(n * test_frac)
    
    test_indices = indices[:num_test]
    val_indices = indices[num_test : num_test + num_val]
    train_indices = indices[num_test + num_val :]

    train_paths = [profiles[i] for i in train_indices]
    val_paths = [profiles[i] for i in val_indices]
    test_paths = [profiles[i] for i in test_indices]

    logger.info(
        f"Profiles split: {len(train_paths)} train / "
        f"{len(val_paths)} val / {len(test_paths)} test."
    )
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
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            config: The configuration dictionary for the run.
            device: The device (CPU, CUDA, or MPS) to run on.
            save_dir: The directory to save logs, models, and results.
            data_dir: The directory containing the normalized dataset.
            collate_fn: The function to collate data into batches.
            optuna_trial: An Optuna trial object for hyperparameter tuning.
        """
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.optuna_trial = optuna_trial
        
        # Split data into train/val/test sets
        train_paths, val_paths, test_paths = _split_profiles(
            data_dir, 
            self.cfg["val_frac"], 
            self.cfg["test_frac"], 
            seed=self.cfg.get("random_seed", DEFAULT_RANDOM_SEED)
        )
        self.test_filenames = [p.name for p in test_paths]
        
        # Create datasets
        dataset_args = {
            "data_folder": data_dir,
            "species_variables": self.cfg["species_variables"],
            "global_variables": self.cfg["global_variables"],
            "cache_size": self.cfg.get("dataset_cache_size", DEFAULT_CACHE_SIZE),
        }
        self.train_ds = ChemicalDataset(**dataset_args, profile_paths=train_paths)
        self.val_ds = ChemicalDataset(**dataset_args, profile_paths=val_paths)
        self.test_ds = ChemicalDataset(**dataset_args, profile_paths=test_paths)

        # MODIFICATION: Setup for physics-informed loss
        self.use_conservation_loss = self.cfg.get("use_conservation_loss", False)
        if self.use_conservation_loss:
            self._setup_conservation_loss(data_dir)
        
        # Initialize all components
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimizer()
        self._setup_mixed_precision()
        self._setup_loss_function()
        self._setup_training_parameters()
        self._build_scheduler()
        self._setup_logging()
        self._save_test_set_info()

    # --- NEW METHOD for Physics-Informed Loss ---
    def _setup_conservation_loss(self, data_dir: Path) -> None:
        """Sets up the necessary components for the atom conservation loss."""
        logger.info("Setting up physics-informed atom conservation loss.")
        
        # Load normalization metadata required for denormalizing predictions
        metadata_path = data_dir / NORMALIZATION_METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Normalization metadata not found at {metadata_path}. "
                "Needed for conservation loss."
            )
        with metadata_path.open("r") as f:
            self.norm_metadata = json.load(f)

        # Parse atom counts from species names
        self.species_vars = sorted(self.cfg["species_variables"])
        atom_matrix, self.atom_names = parse_species_atoms(self.species_vars)
        
        # Move matrix to the correct device for calculations
        self.atom_matrix = torch.tensor(
            atom_matrix, dtype=torch.float32, device=self.device
        )
        logger.info(
            f"Atom conservation will be enforced for: {self.atom_names}"
        )

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Configures and creates DataLoaders for train, validation, and test sets."""
        hw_settings = configure_dataloader_settings()
        is_windows = (sys.platform == "win32")
        
        default_workers = (
            DEFAULT_NUM_WORKERS 
            if self.device.type != 'cpu' and not is_windows 
            else 0
        )
        num_workers = self.cfg.get("num_workers", default_workers)
        
        dataloader_kwargs = {
            "batch_size": self.cfg["batch_size"], 
            "num_workers": num_workers, 
            "pin_memory": hw_settings.get("pin_memory", False), 
            "persistent_workers": (
                hw_settings.get("persistent_workers", False) and num_workers > 0
            ), 
            "collate_fn": collate_fn,
            "drop_last": True,
        }
        
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, **dataloader_kwargs
        )
        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, **dataloader_kwargs
        )
        self.test_loader = DataLoader(
            self.test_ds, shuffle=False, **dataloader_kwargs
        )
        
        logger.info(
            f"DataLoaders created with batch_size={self.cfg['batch_size']}, "
            f"num_workers={num_workers}"
        )

    def _build_model(self) -> None:
        """Builds the prediction model, ensures float32, and compiles for performance."""
        # Create model on CPU first to have full control over type casting
        self.model = create_prediction_model(self.cfg, device=None)
        self.model.float()
        self.model.to(self.device)
        
        # Enable torch.compile on modern PyTorch for CUDA and MPS devices
        if (self.cfg.get("use_torch_compile") and 
            self.device.type in ("cuda", "mps")):
            self._compile_model()
        
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Model built with {trainable_params:,} trainable parameters.")

    def _compile_model(self) -> None:
        """Attempts to compile the model using torch.compile."""
        try:
            logger.info(
                f"Compiling model with torch.compile() for "
                f"{self.device.type.upper()} backend..."
            )
            self.model = torch.compile(self.model)
            logger.info("Model compilation successful.")
        except Exception as e:
            logger.warning(
                f"torch.compile() failed: {e}. Proceeding without compilation."
            )

    def _build_optimizer(self) -> None:
        """Builds the optimizer (e.g., AdamW) based on the configuration."""
        optimizer_name = self.cfg.get("optimizer", "adamw").lower()
        lr = self.cfg["learning_rate"]
        weight_decay = self.cfg.get("weight_decay", 0.01)

        optimizer_classes = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
        }
        
        if optimizer_name not in optimizer_classes:
            available_optimizers = list(optimizer_classes.keys())
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_name}'. "
                f"Available: {available_optimizers}"
            )
        
        optimizer_class = optimizer_classes[optimizer_name]
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        logger.info(
            f"Using {optimizer_name.upper()} optimizer with "
            f"lr={lr:.2e}, weight_decay={weight_decay:.2e}"
        )

    def _setup_mixed_precision(self) -> None:
        """Sets up Automatic Mixed Precision (AMP) if enabled and supported."""
        self.use_amp = (
            self.cfg.get("use_amp", False) and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled.")

    def _setup_loss_function(self) -> None:
        """Sets up the loss function based on configuration."""
        loss_function_name = self.cfg.get("loss_function", "huber").lower()
        
        if loss_function_name == "huber":
            delta = self.cfg.get("huber_delta", DEFAULT_HUBER_DELTA)
            self.criterion = nn.HuberLoss(delta=delta)
            logger.info(f"Using Huber loss with delta={delta}")
        elif loss_function_name == "mse":
            self.criterion = nn.MSELoss()
            logger.info("Using Mean Squared Error (MSE) loss")
        elif loss_function_name == "l1":
            self.criterion = nn.L1Loss()
            logger.info("Using L1 (Mean Absolute Error) loss")
        else:
            available_losses = ["huber", "mse", "l1"]
            raise ValueError(
                f"Unsupported loss function: '{loss_function_name}'. "
                f"Available: {available_losses}"
            )

    def _setup_training_parameters(self) -> None:
        """Sets up various training hyperparameters."""
        self.max_grad_norm = self.cfg.get("gradient_clip_val", DEFAULT_GRADIENT_CLIP)
        self.non_finite_grad_threshold = self.cfg.get(
            "non_finite_grad_threshold", DEFAULT_NON_FINITE_THRESHOLD
        )
        self.accumulation_steps = self.cfg.get(
            "gradient_accumulation_steps", DEFAULT_ACCUMULATION_STEPS
        )
        self.warmup_epochs = self.cfg.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.min_delta = self.cfg.get("min_delta", DEFAULT_MIN_DELTA)

    def _build_scheduler(self) -> None:
        """Builds the learning rate scheduler (e.g., ReduceLROnPlateau)."""
        scheduler_name = self.cfg.get("scheduler_choice", "plateau").lower()
        
        if scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.cfg.get("lr_factor", DEFAULT_LR_FACTOR),
                patience=self.cfg.get("lr_patience", DEFAULT_LR_PATIENCE),
                min_lr=self.cfg.get("min_lr", DEFAULT_MIN_LR),
                threshold=self.min_delta,
                threshold_mode='abs'
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        elif scheduler_name == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.cfg.get("cosine_T_0", DEFAULT_COSINE_T0), 
                T_mult=self.cfg.get("cosine_T_mult", DEFAULT_COSINE_T_MULT)
            )
            logger.info("Using CosineAnnealingWarmRestarts scheduler.")
        else:
            available_schedulers = ["plateau", "cosine"]
            raise ValueError(
                f"Unsupported scheduler: '{scheduler_name}'. "
                f"Available: {available_schedulers}"
            )

    def _setup_logging(self) -> None:
        """Sets up training logging infrastructure."""
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text(
            "epoch,train_loss,val_loss,lr,time_s,grad_norm\n"
        )
        self.best_val_loss = float("inf")

    def _warmup_lr(self, epoch: int) -> None:
        """
        Linearly warms up the learning rate for the initial epochs.
        """
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor

    def _calculate_conservation_loss(
        self,
        initial_species: torch.Tensor,
        predicted_species: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the loss based on atom conservation.
        This function assumes species are already DENORMALIZED.
        """
        # Calculate total atoms for each element at the initial state
        # (batch_size, num_species) @ (num_species, num_atoms) -> (batch_size, num_atoms)
        initial_atoms = torch.matmul(initial_species, self.atom_matrix)

        # Calculate total atoms for each element at the predicted state
        predicted_atoms = torch.matmul(predicted_species, self.atom_matrix)

        # The loss is the mean squared error between the atom counts
        conservation_loss = torch.nn.functional.mse_loss(
            predicted_atoms, initial_atoms
        )
        return conservation_loss

    def _handle_non_finite_loss(
        self, epoch: int, batch_idx: int, non_finite_count: int
    ) -> int:
        """Handles non-finite loss detection."""
        logger.warning(
            f"Epoch {epoch}, Batch {batch_idx}: "
            f"Non-finite loss detected. Skipping."
        )
        non_finite_count += 1
        
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.update()
        
        return non_finite_count

    def _handle_non_finite_gradient(
        self, epoch: int, batch_idx: int, non_finite_count: int
    ) -> int:
        """Handles non-finite gradient detection."""
        logger.warning(
            f"Epoch {epoch}, Batch {batch_idx}: "
            f"Non-finite gradient. Skipping step."
        )
        non_finite_count += 1
        
        if self.use_amp:
            self.scaler.update()
        
        return non_finite_count

    def _should_stop_due_to_instability(self, non_finite_count: int) -> bool:
        """Checks if training should stop due to too many non-finite values."""
        return non_finite_count > self.non_finite_grad_threshold

    def _run_epoch(self, loader: DataLoader, train_phase: bool) -> Tuple[float, float]:
        """
        Runs a single epoch of training or validation.
        """
        self.model.train(train_phase)
        
        total_loss = 0.0
        total_grad_norm = 0.0
        batch_count = 0
        non_finite_batch_count = 0
        
        show_progress = self.cfg.get("show_epoch_progress", False)
        current_epoch = getattr(self, 'current_epoch', 0)
        phase_name = 'Train' if train_phase else 'Val'
        desc = f"Epoch {current_epoch:03d} [{phase_name}]"
        
        iterable = tqdm(
            loader, desc=desc, leave=False, disable=not show_progress
        )
        
        if train_phase:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_phase):
            for batch_idx, (inputs_dict, targets) in enumerate(iterable):
                input_tensor = inputs_dict['x'].to(
                    self.device, non_blocking=True
                )
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    predictions = self.model(input_tensor)
                    
                    # --- MODIFICATION: Calculate combined loss ---
                    # 1. Standard prediction loss
                    prediction_loss = self.criterion(predictions, targets)
                    total_loss_val = prediction_loss

                    # 2. (Optional) Physics-informed conservation loss
                    if self.use_conservation_loss and train_phase:
                        # Denormalize predictions and initial state to physical values
                        num_species = len(self.species_vars)
                        initial_species_norm = input_tensor[:, :num_species]

                        denorm_predictions = torch.stack([
                            DataNormalizer.denormalize(predictions[:, i], self.norm_metadata, var)
                            for i, var in enumerate(self.species_vars)
                        ], dim=1)

                        denorm_initial_species = torch.stack([
                            DataNormalizer.denormalize(initial_species_norm[:, i], self.norm_metadata, var)
                            for i, var in enumerate(self.species_vars)
                        ], dim=1)
                        
                        # Calculate conservation loss on denormalized values
                        conservation_loss = self._calculate_conservation_loss(
                            denorm_initial_species, denorm_predictions
                        )
                        
                        weight = self.cfg.get("conservation_loss_weight", 1.0)
                        total_loss_val = prediction_loss + (weight * conservation_loss)
                    
                    if train_phase:
                        total_loss_val = total_loss_val / self.accumulation_steps

                # Handle non-finite loss
                if not torch.isfinite(total_loss_val):
                    non_finite_batch_count = self._handle_non_finite_loss(
                        self.current_epoch, batch_idx, non_finite_batch_count
                    )
                    if self._should_stop_due_to_instability(non_finite_batch_count):
                        return float('inf'), float('inf')
                    continue

                # Backward pass for training
                if train_phase:
                    self.scaler.scale(total_loss_val).backward()
                    
                    should_step = (
                        (batch_idx + 1) % self.accumulation_steps == 0 or 
                        (batch_idx + 1) == len(loader)
                    )
                    
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        
                        if torch.isfinite(grad_norm):
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            total_grad_norm += grad_norm.item()
                            batch_count += 1
                        else:
                            non_finite_batch_count = self._handle_non_finite_gradient(
                                self.current_epoch, batch_idx, non_finite_batch_count
                            )
                            if self._should_stop_due_to_instability(
                                non_finite_batch_count
                            ):
                                return float('inf'), float('inf')
                        
                        self.optimizer.zero_grad(set_to_none=True)

                # Update metrics and progress bar (using prediction loss for logging)
                actual_loss = prediction_loss.item()
                total_loss += actual_loss
                
                if show_progress:
                    grad_norm_val = (
                        grad_norm.item() 
                        if train_phase and 'grad_norm' in locals() 
                        else 0.0
                    )
                    iterable.set_postfix(
                        loss=f"{actual_loss:.4e}", 
                        grad_norm=f"{grad_norm_val:.3f}"
                    )
        
        average_loss = total_loss / len(loader)
        average_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0.0
        
        return average_loss, average_grad_norm

    def _get_epochs_without_improvement(self) -> int:
        """Gets the number of epochs without improvement for early stopping."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            return getattr(self.scheduler, 'num_bad_epochs', 0)
        return 0

    def _should_early_stop(self, epochs_without_improvement: int) -> bool:
        """Determines if training should stop early."""
        patience = self.cfg.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        return epochs_without_improvement >= patience

    def _should_stop_target_reached(self, val_loss: float) -> bool:
        """Determines if target loss has been reached."""
        target_loss = self.cfg.get("target_loss", DEFAULT_TARGET_LOSS)
        return target_loss > 0 and val_loss < target_loss

    def _log_epoch_results(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        train_grad_norm: float, 
        lr: float, 
        elapsed_time: float, 
        is_best: bool
    ) -> None:
        """Logs the results of an epoch."""
        log_message = (
            f"Epoch {epoch:03d}/{self.cfg['epochs']} | "
            f"Train Loss: {train_loss:.4e} | "
            f"Val Loss: {val_loss:.4e} | "
            f"Grad Norm: {train_grad_norm:.3f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )
        
        if is_best:
            log_message += " | New Best!"
        
        logger.info(log_message)
        
        # Write to CSV log
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6e},{val_loss:.6e},"
                f"{lr:.6e},{elapsed_time:.1f},{train_grad_norm:.4f}\n"
            )

    def train(self) -> float:
        """
        Executes the main training loop over all epochs.
        """
        final_epoch = 0
        
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Apply learning rate warmup
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch)
            
            # Training phase
            train_loss, train_grad_norm = self._run_epoch(
                self.train_loader, train_phase=True
            )
            
            if not np.isfinite(train_loss):
                logger.critical(
                    f"Epoch {epoch:03d} failed due to instability. "
                    f"Stopping training."
                )
                return float('inf')

            # Validation phase
            val_loss, _ = self._run_epoch(self.val_loader, train_phase=False)
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Calculate metrics
            lr = self.optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            
            # Check for new best model
            is_best_model = val_loss < self.best_val_loss - self.min_delta
            if is_best_model:
                self.best_val_loss = val_loss
                self._checkpoint("best_model.pt", epoch, val_loss)
            
            # Log results
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm, 
                lr, elapsed_time, is_best_model
            )

            # Optuna reporting
            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune():
                    raise TrialPruned()
            
            # Early stopping check
            epochs_without_improvement = self._get_epochs_without_improvement()
            if self._should_early_stop(epochs_without_improvement):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} after "
                    f"{epochs_without_improvement} epochs with no improvement."
                )
                break
            
            # Target loss check
            if self._should_stop_target_reached(val_loss):
                logger.info("Target loss achieved! Stopping training.")
                break

            final_epoch = epoch
        
        # Save final model and run test
        self._checkpoint("final_model.pt", final_epoch, val_loss, final=True)
        self.test()
        return self.best_val_loss

    def _get_model_to_save(self) -> nn.Module:
        """Gets the model to save, handling compiled models."""
        return (
            self.model._orig_mod 
            if hasattr(self.model, "_orig_mod") 
            else self.model
        )

    def _checkpoint(
        self, filename: str, epoch: int, val_loss: float, *, final: bool = False
    ) -> None:
        """
        Saves a model checkpoint, including a robust JIT-scripted version.
        """
        model_to_save = self._get_model_to_save()
        
        # Save the standard PyTorch checkpoint
        checkpoint_data = {
            "state_dict": model_to_save.state_dict(), 
            "epoch": epoch, 
            "val_loss": val_loss,
            "optimizer_state": self.optimizer.state_dict(), 
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None, 
            "config": self.cfg,
        }
        
        torch.save(checkpoint_data, self.save_dir / filename)
        
        # Save JIT-scripted version
        self._save_jit_model(model_to_save, filename)

    def _save_jit_model(self, model: nn.Module, filename: str) -> None:
        """Saves a JIT-scripted version of the model."""
        jit_save_path = self.save_dir / (Path(filename).stem + "_jit.pt")
        
        # Store original state
        original_device = next(model.parameters()).device
        original_training_mode = model.training
        
        try:
            # Move model to CPU and set to eval mode for consistent saving
            model_cpu = model.to('cpu').eval()
            
            # Use torch.jit.script for more robust JIT compilation
            scripted_model = torch.jit.script(model_cpu)
            torch.jit.save(scripted_model, str(jit_save_path))
        except Exception as e:
            logger.error(f"Failed to JIT-save model: {e}", exc_info=True)
        finally:
            # Always restore model to its original state
            model.to(original_device).train(original_training_mode)

    def test(self) -> None:
        """
        Evaluates the best-performing model on the test set.
        """
        best_model_path = self.save_dir / "best_model.pt"
        if not best_model_path.exists():
            logger.warning("No best model found. Testing with final model state.")
            return

        # Load best model
        checkpoint = torch.load(best_model_path, map_location=self.device)
        model_to_load = self._get_model_to_save()
        model_to_load.load_state_dict(checkpoint.get("state_dict"))
        
        epoch_loaded = checkpoint.get('epoch', -1)
        logger.info(f"Loaded best model from epoch {epoch_loaded} for testing.")
            
        # Run test evaluation
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        
        show_progress = self.cfg.get("show_epoch_progress", False)
        test_iterable = tqdm(
            self.test_loader, desc="Testing", leave=False, 
            disable=not show_progress
        )
        
        with torch.no_grad():
            for inputs_dict, targets in test_iterable:
                inputs = inputs_dict['x'].to(self.device)
                targets = targets.to(self.device)
                
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_errors.append(torch.abs(predictions - targets))
        
        # Calculate test metrics
        test_loss = total_loss / len(self.test_loader)
        all_errors = torch.cat(all_errors, dim=0)
        per_species_mae = all_errors.mean(dim=0).cpu().numpy()
        per_species_max = all_errors.max(dim=0).values.cpu().numpy()
        
        test_metrics = {
            "test_loss": float(test_loss), 
            "test_mae": float(all_errors.mean().item()),
            "test_max_error": float(all_errors.max().item()), 
            "test_samples": len(self.test_ds),
            "per_species_mae": {
                var: float(val) 
                for var, val in zip(self.cfg["species_variables"], per_species_mae)
            },
            "per_species_max_error": {
                var: float(val) 
                for var, val in zip(self.cfg["species_variables"], per_species_max)
            },
        }
        
        logger.info(
            f"Test Results - Loss: {test_metrics['test_loss']:.4e}, "
            f"MAE: {test_metrics['test_mae']:.4e}, "
            f"Max Error: {test_metrics['test_max_error']:.4e}"
        )
        
        save_json(test_metrics, self.save_dir / "test_metrics.json")
    
    def _save_test_set_info(self) -> None:
        """Saves the list of filenames used in the test set to a JSON file."""
        test_info = {"test_filenames": sorted(self.test_filenames)}
        save_json(test_info, self.save_dir / "test_set_info.json")
        
        logger.info(
            f"Test set filenames saved to "
            f"{self.save_dir / 'test_set_info.json'}"
        )


__all__ = ["ModelTrainer"]