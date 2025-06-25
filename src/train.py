#!/usr/bin/env python3
"""
train.py - Optimized training pipeline for chemical kinetics prediction.

OPTIMIZATIONS:
- Removed conservation loss entirely
- Added configurable JIT export during training
- Fixed scheduler stepping for cosine scheduler
- Optimized batch operations
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
from utils import save_json

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
    """Optimized training pipeline manager."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        h5_path: Path,
        splits: Dict[str, List[int]],
        collate_fn: Callable,
        *,
        optuna_trial: Optional[optuna.Trial] = None,
        norm_metadata: Optional[Dict[str, Any]] = None,
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
        if norm_metadata:
            self.norm_metadata = norm_metadata
            logger.info("Using pre-calculated normalization metadata.")
            self._setup_datasets_with_precalculated_stats(h5_path, splits)
        else:
            self._setup_normalization_and_datasets(h5_path, splits)

        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimizer()
        self._build_schedulers()
        self._setup_loss_and_training_params()
        self._setup_ema()
        self._setup_logging()
        self._save_metadata()

    def _setup_datasets_with_precalculated_stats(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Creates datasets using pre-calculated normalization metadata."""
        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "cache_size": self.misc_cfg.get("dataset_cache_size", -1),
        }
        
        self.train_ds = ChemicalDataset(indices=splits['train'], **ds_kwargs)
        self.val_ds = ChemicalDataset(indices=splits['validation'], **ds_kwargs)
        self.test_ds = ChemicalDataset(indices=splits['test'], **ds_kwargs)
        self.test_set_indices = splits['test']
        
        logger.info(
            f"Datasets created - Train: {len(self.train_ds)}, "
            f"Val: {len(self.val_ds)}, Test: {len(self.test_ds)}"
        )

    def _setup_normalization_and_datasets(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Calculate normalization and create datasets."""
        train_indices = splits['train']
        val_indices = splits['validation']
        test_indices = splits['test']
        
        data_fraction = self.train_params.get("data_fraction", 1.0)
        if 0.0 < data_fraction < 1.0:
            rng = random.Random(self.misc_cfg.get("random_seed", 42))

            def sample_indices(indices: List[int], fraction: float) -> List[int]:
                if not indices:
                    return []
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

        self.test_set_indices = test_indices

        logger.info("Calculating normalization stats from training set...")
        
        normalizer = DataNormalizer(config_data=self.cfg, device=self.device)
        self.norm_metadata = normalizer.calculate_stats(h5_path, train_indices)
        save_json(self.norm_metadata, self.save_dir / "normalization_metadata.json")

        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
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
        """Create optimized data loaders."""
        num_workers = 0  # Must be 0 for HDF5
        
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
        
        # Check for torch compile option in config
        use_compile = self.misc_cfg.get("use_torch_compile", False)
        if use_compile and self.device.type == 'cuda':
            try:
                compile_mode = self.misc_cfg.get("torch_compile_mode", "reduce-overhead")
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info(f"Model compiled with torch.compile(mode='{compile_mode}').")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

    def _build_optimizer(self) -> None:
        """Create optimizer with weight decay settings."""
        opt_name = self.train_params.get("optimizer", DEFAULT_OPTIMIZER).lower()
        lr = self.train_params.get("learning_rate", DEFAULT_LR)
        weight_decay = self.train_params.get("weight_decay", 1e-5)
        
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
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
        """Create learning rate schedulers."""
        batches_per_epoch = len(self.train_loader)
        self.gradient_accumulation = self.train_params.get(
            "gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION
        )
        self.steps_per_epoch = batches_per_epoch // self.gradient_accumulation
        
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
            t0 = self.train_params.get("cosine_T_0", 10)
            self.main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0 * self.steps_per_epoch,
                T_mult=self.train_params.get("cosine_T_mult", 2)
            )
            self.scheduler_needs_loss = False
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'")
        
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
        
        self.use_amp = False  # As requested, no mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = self.train_params.get("gradient_clip_val", DEFAULT_GRAD_CLIP)

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
            "gradient_accumulation_steps": self.gradient_accumulation,
            "effective_batch_size": self.train_params.get("batch_size", DEFAULT_BATCH_SIZE) * self.gradient_accumulation,
        }
        save_json(metadata, self.save_dir / "training_metadata.json")

    def train(self) -> float:
        """Main training loop with optional JIT export."""
        epochs = self.train_params.get("epochs", DEFAULT_EPOCHS)
        patience = self.train_params.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = self.train_params.get("min_delta", DEFAULT_MIN_DELTA)
        
        # Check for JIT export config
        export_jit_during_training = self.misc_cfg.get("export_jit_during_training", False)
        
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation}")
        logger.info(f"Effective batch size: {self.train_params.get('batch_size', DEFAULT_BATCH_SIZE) * self.gradient_accumulation}")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss, train_grad_norm = self._run_epoch(
                self.train_loader, is_train_phase=True
            )
            
            val_loss, _ = self._run_epoch(self.val_loader, is_train_phase=False)
            
            # Update schedulers
            if self.global_step > self.warmup_steps:
                if self.scheduler_needs_loss:
                    self.main_scheduler.step(val_loss)
                # Fixed: step cosine scheduler during training
                else:
                    pass  # Cosine scheduler is stepped per batch in _run_epoch
            
            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune():
                    raise TrialPruned()
            
            improvement = self.best_val_loss - val_loss
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm,
                time.time() - start_time, improvement
            )
            
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
                
                # Export JIT if configured
                if export_jit_during_training:
                    self._export_jit_model(suffix="_epoch{}".format(epoch))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self._checkpoint("final_model.pt", epoch, val_loss)
        
        self.test()
        self._export_jit_model()  # Final JIT export
        
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
        
        show_progress = self.misc_cfg.get("show_epoch_progress", True)
        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        progress_bar = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(is_train_phase):
            for batch_idx, (inputs_dict, targets) in enumerate(progress_bar):
                inputs = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                preds = self.model(inputs)
                loss = self.criterion(preds, targets)
                
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected, skipping batch")
                    continue
                
                scaled_loss = loss / self.gradient_accumulation
                
                if is_train_phase:
                    scaled_loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation == 0 or (batch_idx + 1) == len(loader):
                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        total_grad_norm += grad_norm.item()
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        self.global_step += 1
                        num_optimizer_steps += 1
                        
                        if self.ema:
                            self.ema.update()
                        
                        # Step schedulers
                        if self.global_step <= self.warmup_steps and self.warmup_scheduler:
                            self.warmup_scheduler.step()
                        elif not self.scheduler_needs_loss and self.global_step > self.warmup_steps:
                            self.main_scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if show_progress:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4e}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_grad_norm = total_grad_norm / num_optimizer_steps if is_train_phase and num_optimizer_steps > 0 else 0.0
        
        return avg_loss, avg_grad_norm

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
        
        with self.log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6e},{val_loss:.6e},"
                f"{lr:.6e},{grad_norm:.4f},{duration:.1f},{improvement:.6e}\n"
            )

    def _checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod
        
        if self.ema and "best" in filename:
            self.ema.apply_shadow()
        
        checkpoint = {
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg,
            "normalization_metadata": self.norm_metadata,
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        logger.debug(f"Saved checkpoint: {filename}")
        
        if self.ema and "best" in filename:
            self.ema.restore()

    def test(self) -> None:
        """Test the best model."""
        ckpt_path = self.save_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("No best model found, skipping test.")
            return
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt["state_dict"])
        
        logger.info(f"Testing model from epoch {ckpt['epoch']}")
        
        test_loss, _ = self._run_epoch(self.test_loader, is_train_phase=False)
        
        metrics = {
            "test_loss": test_loss,
            "best_epoch": ckpt['epoch'],
            "best_val_loss": ckpt['val_loss'],
        }
        
        logger.info(f"Test Loss: {test_loss:.4e}")
        save_json(metrics, self.save_dir / "test_metrics.json")

    def _export_jit_model(self, suffix: str = "") -> None:
        """Export model as JIT with configurable suffix."""
        if not self.misc_cfg.get("export_jit_model", True):
            logger.info("JIT export disabled in config.")
            return
            
        try:
            ckpt_path = self.save_dir / "best_model.pt"
            if not ckpt_path.exists():
                logger.warning("No best model for JIT export")
                return
            
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            fresh_model = create_prediction_model(self.cfg, device=self.device)
            fresh_model.load_state_dict(ckpt["state_dict"])
            fresh_model.eval()
            
            num_species = len(self.data_spec["species_variables"])
            num_global = len(self.data_spec["global_variables"])
            example_input = torch.randn(
                1, num_species + num_global + 1,
                device=self.device
            )
            
            jit_path = self.save_dir / f"best_model_jit{suffix}.pt"
            export_model_jit(fresh_model, example_input, jit_path, optimize=True)
            
        except Exception as e:
            logger.error(f"JIT export failed: {e}")


__all__ = ["ModelTrainer"]