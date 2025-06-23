#!/usr/bin/env python3
"""
train.py - Training pipeline for chemical kinetics prediction models.

Features:
- Learning rate warmup
- Gradient accumulation
- Model checkpointing with EMA
- Comprehensive logging
- JIT export
- Mixed precision training
- Option to train on a fraction of data
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim, Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChemicalDataset, collate_fn
from hardware import configure_dataloader_settings
from model import create_prediction_model, export_model_jit
from normalizer import DataNormalizer
from utils import parse_species_atoms, save_json

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
DEFAULT_LR = 5e-4
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-8
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_GRADIENT_ACCUMULATION = 1
DEFAULT_EMA_DECAY = 0.999


class ExponentialMovingAverage:
    """Maintains exponential moving average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = DEFAULT_EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.backup_params = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int = -1
) -> LambdaLR:
    """Create linear warmup scheduler."""
    def warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))
    
    return LambdaLR(optimizer, warmup_lambda)


class ModelTrainer:
    """Manages training, validation, and testing pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        h5_path: Path,
        splits: Dict[str, List[int]],
        collate_fn: Callable,
        *,
        optuna_trial: Optional[optuna.Trial] = None
    ):
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.optuna_trial = optuna_trial

        # Extract config sections
        self.data_spec = self.cfg["data_specification"]
        self.train_params = self.cfg["training_hyperparameters"]
        self.misc_cfg = self.cfg["miscellaneous_settings"]
        self.species_vars = sorted(self.data_spec["species_variables"])

        # Setup components
        self.use_conservation_loss = self.train_params.get("use_conservation_loss", False)
        self._setup_conservation_loss()
        self._setup_normalization_and_datasets(h5_path, splits)
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimizer()
        self._build_schedulers()
        self._setup_loss_and_training_params()
        self._setup_ema()
        self._setup_logging()
        self._save_metadata()
        
        # Initialize denormalization cache for efficiency
        self._denorm_cache = {}

    def _setup_conservation_loss(self) -> None:
        """Setup atom conservation matrix if enabled."""
        self.atom_matrix: Optional[Tensor] = None
        self.atom_names: List[str] = []
        
        if not self.use_conservation_loss:
            logger.info("Conservation loss is disabled.")
            return
        
        # Parse chemical formulas
        atom_matrix_np, self.atom_names = parse_species_atoms(self.species_vars)
        if not self.atom_names:
            logger.warning("No atoms parsed from species. Disabling conservation loss.")
            self.use_conservation_loss = False
            return
        
        self.atom_matrix = torch.tensor(
            atom_matrix_np, dtype=torch.float32, device=self.device
        )
        logger.info(f"Atom conservation enabled for: {self.atom_names}")

    def _setup_normalization_and_datasets(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Calculate normalization and create datasets."""
        train_indices = splits['train']
        val_indices = splits['validation']
        test_indices = splits['test']
        
        # Handle training on a fraction of data
        data_fraction = self.train_params.get("data_fraction", 1.0)
        if 0.0 < data_fraction < 1.0:
            rng = random.Random(self.misc_cfg.get("random_seed", 42))

            def sample_indices(indices: List[int], fraction: float) -> List[int]:
                if not indices: return []
                num_original = len(indices)
                num_new = int(num_original * fraction)
                return sorted(rng.sample(indices, num_new))

            original_sizes = (len(train_indices), len(val_indices), len(test_indices))
            train_indices = sample_indices(train_indices, data_fraction)
            val_indices = sample_indices(val_indices, data_fraction)
            test_indices = sample_indices(test_indices, data_fraction)
            new_sizes = (len(train_indices), len(val_indices), len(test_indices))

            logger.info(
                f"Using a fraction of data: {data_fraction:.2%}. "
                f"Train: {new_sizes[0]}/{original_sizes[0]}, "
                f"Val: {new_sizes[1]}/{original_sizes[1]}, "
                f"Test: {new_sizes[2]}/{original_sizes[2]}."
            )
            logger.warning("Normalization stats will be calculated on this training subset.")

        self.test_set_indices = test_indices

        logger.info("Calculating normalization stats from training set...")
        
        # Calculate normalization
        normalizer = DataNormalizer(config_data=self.cfg, device=self.device)
        self.norm_metadata = normalizer.calculate_stats(h5_path, train_indices)
        save_json(self.norm_metadata, self.save_dir / "normalization_metadata.json")

        # Create datasets
        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "atom_matrix": self.atom_matrix if self.use_conservation_loss else None,
            "cache_size": self.misc_cfg.get("dataset_cache_size", -1),
        }
        
        self.train_ds = ChemicalDataset(indices=train_indices, **ds_kwargs)
        self.val_ds = ChemicalDataset(indices=val_indices, **ds_kwargs)
        self.test_ds = ChemicalDataset(indices=test_indices, **ds_kwargs)
        
        logger.info(
            f"Datasets created - Train: {len(self.train_ds)}, "
            f"Val: {len(self.val_ds)}, Test: {len(self.test_ds)}"
        )

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Create data loaders with optimized settings."""
        # Force num_workers=0 for HDF5
        num_workers = 0
        if self.misc_cfg.get("num_workers", 0) > 0:
            logger.warning("HDF5 requires num_workers=0, forcing this value.")

        hw_settings = configure_dataloader_settings()
        batch_size = self.train_params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        dl_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": hw_settings.get("pin_memory", False) and self.device.type == "cuda",
            "persistent_workers": False,
        }
        
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=True, **dl_kwargs
        )
        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, **dl_kwargs
        )
        self.test_loader = DataLoader(
            self.test_ds, shuffle=False, **dl_kwargs
        )

    def _build_model(self) -> None:
        """Create and optionally compile the model."""
        self.model = create_prediction_model(self.cfg, device=self.device)
        
        # Torch compile (CUDA only)
        if self.misc_cfg.get("use_torch_compile", False) and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile().")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

    def _build_optimizer(self) -> None:
        """Create optimizer with weight decay settings."""
        opt_name = self.train_params.get("optimizer", DEFAULT_OPTIMIZER).lower()
        lr = self.train_params.get("learning_rate", DEFAULT_LR)
        weight_decay = self.train_params.get("weight_decay", 1e-5)
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and layer norms
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Create optimizer
        if opt_name == "adamw":
            self.optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        elif opt_name == "adam":
            self.optimizer = optim.Adam(param_groups, lr=lr)
        elif opt_name == "rmsprop":
            self.optimizer = optim.RMSprop(param_groups, lr=lr)
        else:
            logger.warning(f"Unknown optimizer '{opt_name}', using AdamW")
            self.optimizer = optim.AdamW(param_groups, lr=lr)

    def _build_schedulers(self) -> None:
        """Create learning rate schedulers including warmup."""
        # Calculate actual steps per epoch considering gradient accumulation
        batches_per_epoch = len(self.train_loader)
        self.gradient_accumulation = self.train_params.get(
            "gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION
        )
        self.steps_per_epoch = batches_per_epoch // self.gradient_accumulation
        
        # Main scheduler
        scheduler_name = self.train_params.get("scheduler_choice", "plateau").lower()
        
        if scheduler_name == "plateau":
            self.main_scheduler = ReduceLROnPlateau(
                self.optimizer, 'min',
                factor=self.train_params.get("factor", 0.5),
                patience=self.train_params.get("patience", 10),
                min_lr=self.train_params.get("min_lr", 1e-7)
            )
            self.scheduler_needs_loss = True
        elif scheduler_name == "cosine":
            # Adjust T_0 for gradient accumulation
            t0 = self.train_params.get("cosine_T_0", 10)
            self.main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0 * self.steps_per_epoch,
                T_mult=self.train_params.get("cosine_T_mult", 2)
            )
            self.scheduler_needs_loss = False
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'")
        
        # Warmup scheduler
        self.warmup_epochs = self.train_params.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
        if self.warmup_epochs > 0:
            self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
            self.warmup_scheduler = get_warmup_scheduler(
                self.optimizer, self.warmup_steps, -1
            )
        else:
            self.warmup_scheduler = None
            self.warmup_steps = 0

    def _setup_loss_and_training_params(self) -> None:
        """Setup loss function and training parameters."""
        # Loss function
        loss_name = self.train_params.get("loss_function", "mse").lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            delta = self.train_params.get("huber_delta", 0.1)
            self.criterion = nn.HuberLoss(delta=delta)
        elif loss_name == "l1":
            self.criterion = nn.L1Loss()
        else:
            logger.warning(f"Unknown loss '{loss_name}', using MSE")
            self.criterion = nn.MSELoss()
        
        # Training parameters
        self.use_amp = self.train_params.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = self.train_params.get("gradient_clip_val", DEFAULT_GRAD_CLIP)
        self.conservation_weight = self.train_params.get("conservation_loss_weight", 1.0)

    def _setup_ema(self) -> None:
        """Setup exponential moving average."""
        use_ema = self.train_params.get("use_ema", False)
        if use_ema:
            ema_decay = self.train_params.get("ema_decay", DEFAULT_EMA_DECAY)
            self.ema = ExponentialMovingAverage(self.model, ema_decay)
            logger.info(f"EMA enabled with decay={ema_decay}")
        else:
            self.ema = None

    def _setup_logging(self) -> None:
        """Setup training logs."""
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text(
            "epoch,train_loss,val_loss,lr,grad_norm,time_s,improvement\n"
        )
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.global_step = 0

    def _save_metadata(self) -> None:
        """Save training metadata."""
        metadata = {
            "test_set_indices": sorted(self.test_set_indices),
            "num_train_samples": len(self.train_ds),
            "num_val_samples": len(self.val_ds),
            "num_test_samples": len(self.test_ds),
            "atom_names": self.atom_names if self.use_conservation_loss else [],
            "gradient_accumulation_steps": self.gradient_accumulation,
            "effective_batch_size": self.train_params.get("batch_size", DEFAULT_BATCH_SIZE) * self.gradient_accumulation,
        }
        save_json(metadata, self.save_dir / "training_metadata.json")

    def train(self) -> float:
        """Main training loop."""
        epochs = self.train_params.get("epochs", DEFAULT_EPOCHS)
        patience = self.train_params.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = self.train_params.get("min_delta", DEFAULT_MIN_DELTA)
        
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation}")
        logger.info(f"Effective batch size: {self.train_params.get('batch_size', DEFAULT_BATCH_SIZE) * self.gradient_accumulation}")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_loss, train_grad_norm = self._run_epoch(
                self.train_loader, is_train_phase=True
            )
            
            # Validation phase
            val_loss, _ = self._run_epoch(self.val_loader, is_train_phase=False)
            
            # Update main scheduler (per epoch)
            if self.global_step > self.warmup_steps:
                if self.scheduler_needs_loss:
                    self.main_scheduler.step(val_loss)
                else:
                    # For cosine scheduler, it steps per optimizer step, handled in _run_epoch
                    pass
            
            # Optuna pruning
            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune():
                    raise TrialPruned()
            
            # Logging
            improvement = self.best_val_loss - val_loss
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm,
                time.time() - start_time, improvement
            )
            
            # Model checkpointing
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final checkpoint
        self._checkpoint("final_model.pt", epoch, val_loss)
        
        # Test and export
        self.test()
        self._export_jit_model()
        
        return self.best_val_loss

    def _run_epoch(
        self, loader: DataLoader, is_train_phase: bool
    ) -> Tuple[float, float]:
        """Run one epoch of training or validation."""
        self.model.train(is_train_phase)
        
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        num_optimizer_steps = 0
        
        # Progress bar
        show_progress = self.misc_cfg.get("show_epoch_progress", True)
        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        progress_bar = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
        
        # Gradient accumulation
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(is_train_phase):
            for batch_idx, (inputs_dict, targets, initial_atoms) in enumerate(progress_bar):
                # Move data to device
                inputs = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.use_conservation_loss:
                    initial_atoms = initial_atoms.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    preds = self.model(inputs)
                    prediction_loss = self.criterion(preds, targets)
                    
                    # Skip batch if loss is invalid
                    if not torch.isfinite(prediction_loss):
                        logger.warning(f"Non-finite loss detected, skipping batch")
                        continue
                    
                    # Add conservation loss if training
                    if is_train_phase and self.use_conservation_loss:
                        cons_loss = self._calculate_conservation_loss(preds, initial_atoms)
                        total_batch_loss = prediction_loss + self.conservation_weight * cons_loss
                    else:
                        total_batch_loss = prediction_loss
                    
                    # Scale for gradient accumulation
                    scaled_loss = total_batch_loss / self.gradient_accumulation
                
                # Backward pass
                if is_train_phase:
                    self.scaler.scale(scaled_loss).backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.gradient_accumulation == 0 or (batch_idx + 1) == len(loader):
                        # Unscale and clip gradients
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        total_grad_norm += grad_norm.item()
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Update global step counter
                        self.global_step += 1
                        num_optimizer_steps += 1
                        
                        # Update EMA
                        if self.ema:
                            self.ema.update()
                        
                        # Update schedulers
                        if self.global_step <= self.warmup_steps and self.warmup_scheduler:
                            self.warmup_scheduler.step()
                        elif not self.scheduler_needs_loss and self.global_step > self.warmup_steps:
                            # Step cosine scheduler per optimizer step
                            self.main_scheduler.step()
                
                # Accumulate loss (always use prediction loss for consistency)
                total_loss += prediction_loss.item()
                num_batches += 1
                
                # Update progress bar
                if show_progress:
                    progress_bar.set_postfix({
                        "loss": f"{prediction_loss.item():.4e}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
        
        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_grad_norm = total_grad_norm / num_optimizer_steps if is_train_phase and num_optimizer_steps > 0 else 0.0
        
        return avg_loss, avg_grad_norm

    def _calculate_conservation_loss(
        self, predicted_norm: Tensor, initial_atoms: Tensor
    ) -> Tensor:
        """Calculate atom conservation loss."""
        if self.atom_matrix is None or initial_atoms.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Denormalize predictions efficiently
        denorm_pred = self._batch_denormalize_species(predicted_norm)
        
        # Calculate atom counts
        predicted_atoms = torch.matmul(denorm_pred, self.atom_matrix)
        
        # Conservation loss
        return torch.nn.functional.mse_loss(predicted_atoms, initial_atoms)

    def _batch_denormalize_species(self, normalized_batch: Tensor) -> Tensor:
        """Efficiently denormalize a batch of species with caching."""
        batch_size = normalized_batch.shape[0]
        num_species = len(self.species_vars)
        
        # Pre-allocate
        denorm_batch = torch.zeros(
            batch_size, num_species,
            device=normalized_batch.device,
            dtype=torch.float32
        )
        
        # Denormalize each species (with caching)
        for i, var_name in enumerate(self.species_vars):
            if var_name not in self._denorm_cache:
                key_stats = self.norm_metadata["per_key_stats"].get(var_name)
                method = self.norm_metadata["normalization_methods"][var_name]
                if key_stats:
                    self._denorm_cache[var_name] = (method, key_stats)
                else:
                    self._denorm_cache[var_name] = None
            
            cache_entry = self._denorm_cache[var_name]
            if cache_entry:
                method, key_stats = cache_entry
                denorm_batch[:, i] = DataNormalizer.denormalize_tensor(
                    normalized_batch[:, i], method, key_stats
                )
        
        return denorm_batch

    def _log_epoch_results(
        self, epoch: int, train_loss: float, val_loss: float,
        grad_norm: float, duration: float, improvement: float
    ) -> None:
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']
        
        log_msg = (
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.4e} | "
            f"Val: {val_loss:.4e} | "
            f"LR: {lr:.2e} | "
            f"Grad: {grad_norm:.2f} | "
            f"Time: {duration:.1f}s"
        )
        
        logger.info(log_msg)
        
        # Write to CSV
        with self.log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6e},{val_loss:.6e},"
                f"{lr:.6e},{grad_norm:.4f},{duration:.1f},{improvement:.6e}\n"
            )

    def _checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        # Get base model (handle compiled models)
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod
        
        # Apply EMA if available
        if self.ema and "best" in filename:
            self.ema.apply_shadow()
        
        checkpoint = {
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg,
            "normalization_metadata": self.norm_metadata,
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "global_step": self.global_step,
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        logger.debug(f"Saved checkpoint: {filename}")
        
        # Restore non-EMA weights
        if self.ema and "best" in filename:
            self.ema.restore()

    def test(self) -> None:
        """Test the best model."""
        ckpt_path = self.save_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("No best model found, skipping test.")
            return
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Load state dict
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt["state_dict"])
        
        logger.info(f"Testing model from epoch {ckpt['epoch']}")
        
        # Run test
        test_loss, _ = self._run_epoch(self.test_loader, is_train_phase=False)
        
        # Save metrics
        metrics = {
            "test_loss": test_loss,
            "best_epoch": ckpt['epoch'],
            "best_val_loss": ckpt['val_loss'],
        }
        
        logger.info(f"Test Loss: {test_loss:.4e}")
        save_json(metrics, self.save_dir / "test_metrics.json")

    def _export_jit_model(self) -> None:
        """Export best model as JIT."""
        try:
            # Load best checkpoint
            ckpt_path = self.save_dir / "best_model.pt"
            if not ckpt_path.exists():
                logger.warning("No best model for JIT export")
                return
            
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            # Create fresh model
            fresh_model = create_prediction_model(self.cfg, device=self.device)
            fresh_model.load_state_dict(ckpt["state_dict"])
            fresh_model.eval()
            
            # Example input
            num_species = len(self.data_spec["species_variables"])
            num_global = len(self.data_spec["global_variables"])
            example_input = torch.randn(
                1, num_species + num_global + 1,
                device=self.device
            )
            
            # Export
            jit_path = self.save_dir / "best_model_jit.pt"
            export_model_jit(fresh_model, example_input, jit_path, optimize=True)
            
        except Exception as e:
            logger.error(f"JIT export failed: {e}")


__all__ = ["ModelTrainer"]