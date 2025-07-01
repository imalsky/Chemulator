#!/usr/bin/env python3
"""
train.py - Training pipeline.
"""
from __future__ import annotations

import gc
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import create_dataset, collate_fn
from hardware import configure_dataloader_settings
from model import create_prediction_model, export_model_jit
from normalizer import DataNormalizer
from utils import save_json

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-10
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_GRADIENT_ACCUMULATION = 1
DEFAULT_MAX_INVALID_BATCHES = 100
DEFAULT_INVALID_BATCH_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 0


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """
    Create a linear warmup learning rate scheduler.
    
    Gradually increases learning rate from 0 to the target value over warmup_steps,
    which helps stabilize training in the early stages.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for warmup
        
    Returns:
        LambdaLR scheduler with linear warmup
    """
    def warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))
    
    return LambdaLR(optimizer, warmup_lambda)


class ModelTrainer:
    """
    Training pipeline manager with dataset caching support.
    
    This class handles the entire training workflow including data loading,
    model training, validation, checkpointing, and export. It's optimized
    for both GPU and CPU training with features like:
    
    - Automatic mixed precision (AMP) for faster GPU training
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling with warmup
    - Robust error handling and recovery
    - Efficient dataset caching for faster epochs
    - HDF5-aware multi-worker data loading
    """

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
        preloaded_train_ds: Optional[Dataset] = None,
        preloaded_val_ds: Optional[Dataset] = None,
        preloaded_test_ds: Optional[Dataset] = None,
    ):
        """
        Initialize the trainer with all necessary components.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device for training
            save_dir: Directory to save outputs
            h5_path: Path to HDF5 dataset
            splits: Dictionary with train/val/test indices
            collate_fn: Function to collate batches
            optuna_trial: Optional Optuna trial for hyperparameter tuning
            norm_metadata: Pre-calculated normalization metadata
            preloaded_*_ds: Pre-loaded datasets for efficiency in hyperparameter search
        """
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.optuna_trial = optuna_trial
        self.h5_path = h5_path

        # Extract config sections
        self.data_spec = self.cfg["data_specification"]
        self.train_params = self.cfg["training_hyperparameters"]
        self.misc_cfg = self.cfg["miscellaneous_settings"]
        self.species_vars = sorted(self.data_spec["species_variables"])

        # Training stability parameters
        self.max_invalid_batches = self.train_params.get(
            "max_invalid_batches", DEFAULT_MAX_INVALID_BATCHES
        )
        self.invalid_batch_threshold = self.train_params.get(
            "invalid_batch_threshold", DEFAULT_INVALID_BATCH_THRESHOLD
        )
        self.log_gradient_norms = self.misc_cfg.get("log_gradient_norms", True)
        self.save_checkpoint_interval = self.misc_cfg.get("save_checkpoint_every_n_epochs", 10)
        
        # Dataset caching configuration
        self.cache_dataset = self.misc_cfg.get("cache_dataset", False)
        if self.cache_dataset:
            logger.info("Dataset caching enabled - will load all data into memory")
        
        # Enable anomaly detection for debugging if requested
        if self.misc_cfg.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - this will slow down training!")

        # Setup components in order
        if preloaded_train_ds and preloaded_val_ds and norm_metadata:
            logger.info("Using pre-loaded datasets for this trial.")
            self.train_ds = preloaded_train_ds
            self.val_ds = preloaded_val_ds
            self.test_ds = preloaded_test_ds
            self.norm_metadata = norm_metadata
            self.test_set_indices = splits['test']
            logger.info(f"Datasets assigned - Train: {len(self.train_ds):,}, Val: {len(self.val_ds):,}")
        elif norm_metadata:
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
        self._setup_logging()
        self._save_metadata()
        
        # Force garbage collection after setup to free memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_num_workers(self) -> int:
        """
        Determine optimal number of dataloader workers based on hardware and caching.
        
        Returns:
            Number of workers for DataLoader
        """
        num_workers_cfg = self.misc_cfg.get("num_dataloader_workers")
        if isinstance(num_workers_cfg, int):
            logger.info(f"Using configured num_dataloader_workers: {num_workers_cfg}")
            return num_workers_cfg

        # Auto-configuration based on hardware and dataset type
        if self.device.type == "cpu":
            # On CPU, limit workers to avoid context switching overhead
            num_workers = min(2, multiprocessing.cpu_count() // 2)
            logger.info(f"Auto-configured num_workers: {num_workers} (CPU-bound training)")
            return num_workers

        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # For GPU training, balance between data loading and GPU utilization
        if self.cache_dataset:
            # Cached datasets need fewer workers since data is already in memory
            num_workers = min(4, cpu_count // 4 if cpu_count > 4 else 1)
            logger.info(f"Auto-configured num_workers: {num_workers} (Cached dataset)")
        else:
            # Streaming from HDF5 benefits from more workers
            num_workers = min(8, cpu_count // 2 if cpu_count > 2 else 1)
            logger.info(f"Auto-configured num_workers: {num_workers} (Streaming HDF5)")
        
        return num_workers

    def _setup_datasets_with_precalculated_stats(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """
        Create datasets using pre-calculated normalization statistics.
        
        Args:
            h5_path: Path to HDF5 file
            splits: Dictionary with train/val/test indices
        """
        max_memory_gb = self.misc_cfg.get("max_memory_per_worker_gb", 2.0)
        profiles_per_chunk = self.misc_cfg.get("profiles_per_chunk", 2048)
        
        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "cache_dataset": self.cache_dataset,
            "profiles_per_chunk": profiles_per_chunk,
            "max_memory_gb": max_memory_gb,
        }
        
        self.train_ds = create_dataset(indices=splits['train'], **ds_kwargs)
        self.val_ds = create_dataset(indices=splits['validation'], **ds_kwargs)
        self.test_ds = create_dataset(indices=splits['test'], **ds_kwargs)
        self.test_set_indices = splits['test']
        
        # Get sample counts
        train_samples = len(self.train_ds) if hasattr(self.train_ds, '__len__') else self.train_ds.total_samples
        val_samples = len(self.val_ds) if hasattr(self.val_ds, '__len__') else self.val_ds.total_samples
        test_samples = len(self.test_ds) if hasattr(self.test_ds, '__len__') else self.test_ds.total_samples
        
        logger.info(
            f"Datasets created - Train: {train_samples:,}, Val: {val_samples:,}, Test: {test_samples:,}"
        )

    def _setup_normalization_and_datasets(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """
        Calculate normalization statistics and create datasets.
        
        Args:
            h5_path: Path to HDF5 file
            splits: Dictionary with train/val/test indices
        """
        train_indices, val_indices, test_indices = splits['train'], splits['validation'], splits['test']
        
        # Handle data fraction for quick experiments
        data_fraction = self.train_params.get("data_fraction", 1.0)
        if 0.0 < data_fraction < 1.0:
            def sample_indices(indices: List[int], fraction: float) -> List[int]:
                num_new = int(len(indices) * fraction)
                return sorted(random.sample(indices, num_new))

            original_sizes = (len(train_indices), len(val_indices))
            train_indices = sample_indices(train_indices, data_fraction)
            val_indices = sample_indices(val_indices, data_fraction)
            logger.info(
                f"Using {data_fraction:.1%} of data. Train: {len(train_indices):,}/{original_sizes[0]:,}, "
                f"Val: {len(val_indices):,}/{original_sizes[1]:,}"
            )

        self.test_set_indices = test_indices

        # Calculate normalization statistics
        logger.info("Calculating normalization statistics from training set...")
        normalizer = DataNormalizer(config_data=self.cfg)
        self.norm_metadata, raw_train_data = normalizer.calculate_stats(h5_path, train_indices)
        save_json(self.norm_metadata, self.save_dir / "normalization_metadata.json")

        # Create datasets
        max_memory_gb = self.misc_cfg.get("max_memory_per_worker_gb", 2.0)
        profiles_per_chunk = self.misc_cfg.get("profiles_per_chunk", 2048)
        
        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "cache_dataset": self.cache_dataset,
            "profiles_per_chunk": profiles_per_chunk,
            "max_memory_gb": max_memory_gb,
        }
        
        # Use raw data for training set if caching
        self.train_ds = create_dataset(
            indices=train_indices, 
            raw_data_for_caching=raw_train_data if self.cache_dataset else None,
            **ds_kwargs
        )
        self.val_ds = create_dataset(indices=val_indices, **ds_kwargs)
        self.test_ds = create_dataset(indices=test_indices, **ds_kwargs)
        
        # Clean up raw data
        del raw_train_data
        gc.collect()

        # Log dataset sizes
        train_samples = len(self.train_ds) if hasattr(self.train_ds, '__len__') else self.train_ds.total_samples
        val_samples = len(self.val_ds) if hasattr(self.val_ds, '__len__') else self.val_ds.total_samples
        test_samples = len(self.test_ds) if hasattr(self.test_ds, '__len__') else self.test_ds.total_samples
        
        logger.info(
            f"Datasets created - Train: {train_samples:,}, Val: {val_samples:,}, Test: {test_samples:,}"
        )

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """
        Create optimized DataLoaders with proper worker configuration.
        
        Args:
            collate_fn: Function to collate batches
        """
        num_workers = self._get_num_workers()
        hw_settings = configure_dataloader_settings()
        batch_size = self.train_params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        # DataLoader configuration
        dl_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": hw_settings.get("pin_memory", False) and self.device.type == "cuda",
            "persistent_workers": hw_settings.get("persistent_workers", False) and num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
        
        # Check if dataset supports shuffling (map-style vs iterable)
        is_map_style = hasattr(self.train_ds, '__len__')
        
        self.train_loader = DataLoader(self.train_ds, shuffle=is_map_style, **dl_kwargs)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **dl_kwargs)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **dl_kwargs)
        
        logger.info(
            f"DataLoaders created with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={dl_kwargs['pin_memory']}, "
            f"persistent_workers={dl_kwargs['persistent_workers']}"
        )

    def _build_model(self) -> None:
        """Create and optionally compile the model for faster execution."""
        self.model = create_prediction_model(self.cfg, device=self.device)
        
        # Try to compile model if requested and available
        use_compile = self.misc_cfg.get("use_torch_compile", False)
        if use_compile and hasattr(torch, 'compile'):
            try:
                compile_mode = self.misc_cfg.get("torch_compile_mode", "reduce-overhead")
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info(f"Model compiled with torch.compile(mode='{compile_mode}')")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    def _build_optimizer(self) -> None:
        """Create AdamW optimizer with proper weight decay handling."""
        lr = self.train_params.get("learning_rate", DEFAULT_LR)
        weight_decay = self.train_params.get("weight_decay", 1e-5)
        
        # Separate parameters that should and shouldn't have weight decay
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and normalization parameters
            if param.dim() == 1 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        self.optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        logger.info(f"Optimizer: AdamW with lr={lr:.2e}, weight_decay={weight_decay:.2e}")

    def _build_schedulers(self) -> None:
        """Create learning rate schedulers with warmup support."""
        # Calculate steps per epoch for scheduling
        total_samples = len(self.train_ds) if hasattr(self.train_ds, '__len__') else self.train_ds.total_samples
        batch_size = self.train_params.get("batch_size", DEFAULT_BATCH_SIZE)
        self.gradient_accumulation = self.train_params.get("gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION)
        
        self.steps_per_epoch = max(1, total_samples // (batch_size * self.gradient_accumulation))
        logger.info(f"Steps per epoch: ~{self.steps_per_epoch}")
        
        # Create main scheduler based on config
        scheduler_name = self.train_params.get("scheduler_choice", "plateau").lower()
        if scheduler_name == "plateau":
            self.main_scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', 
                factor=self.train_params.get("factor", 0.5),
                patience=self.train_params.get("patience", 10), 
                min_lr=self.train_params.get("min_lr", 1e-7),
            )
            self.scheduler_updates_on_epoch = True
        elif scheduler_name == "cosine":
            t0 = self.train_params.get("cosine_T_0", 10)
            self.main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=t0 * self.steps_per_epoch,
                T_mult=self.train_params.get("cosine_T_mult", 2), 
                eta_min=self.train_params.get("min_lr", 1e-7)
            )
            self.scheduler_updates_on_epoch = False
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. Use 'plateau' or 'cosine'.")
        
        # Setup warmup scheduler if requested
        self.warmup_epochs = self.train_params.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.warmup_scheduler = get_warmup_scheduler(self.optimizer, self.warmup_steps) if self.warmup_steps > 0 else None
        if self.warmup_scheduler:
            logger.info(f"Warmup enabled for {self.warmup_epochs} epochs ({self.warmup_steps} steps)")

    def _setup_loss_and_training_params(self) -> None:
        """Setup loss function and training parameters including AMP."""
        # Create loss function
        loss_name = self.train_params.get("loss_function", "mse").lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss(delta=self.train_params.get("huber_delta", 1.0))
        else:
            raise ValueError(f"Unsupported loss function '{loss_name}'. Use 'mse' or 'huber'.")
        
        # Setup automatic mixed precision
        self.use_amp = self.train_params.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled")
        
        # Gradient clipping value
        self.max_grad_norm = self.train_params.get("gradient_clip_val", DEFAULT_GRAD_CLIP)

    def _setup_logging(self) -> None:
        """Setup training logs and metrics tracking."""
        self.log_path = self.save_dir / "training_log.csv"
        headers = ["epoch", "train_loss", "val_loss", "lr", "grad_norm", "time_s", "improvement"]
        self.log_path.write_text(",".join(headers) + "\n")
        self.best_val_loss, self.best_epoch = float("inf"), -1
        self.global_step, self.total_invalid_batches = 0, 0

    def _save_metadata(self) -> None:
        """Save training metadata for reproducibility."""
        train_samples = len(self.train_ds) if hasattr(self.train_ds, '__len__') else self.train_ds.total_samples
        metadata = {
            "test_set_indices": sorted(self.test_set_indices),
            "num_train_samples": train_samples,
            "effective_batch_size": self.train_params.get("batch_size", DEFAULT_BATCH_SIZE) * self.gradient_accumulation,
            "device": str(self.device), 
            "cache_dataset": self.cache_dataset,
            "model_type": self.cfg["model_hyperparameters"].get("model_type", "siren"),
        }
        save_json(metadata, self.save_dir / "training_metadata.json")

    def train(self) -> float:
        """
        Execute the main training loop.
        
        Returns:
            Best validation loss achieved during training
        """
        epochs = self.train_params.get("epochs", DEFAULT_EPOCHS)
        patience = self.train_params.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = self.train_params.get("min_delta", DEFAULT_MIN_DELTA)
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for up to {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_loss, train_grad_norm = self._run_epoch(self.train_loader, is_train_phase=True)
            if train_loss is None: 
                logger.error("Catastrophic failure during training. Stopping.")
                break

            # Validation phase
            val_loss, _ = self._run_epoch(self.val_loader, is_train_phase=False)
            if val_loss is None: 
                val_loss = float('inf')

            # Update learning rate scheduler
            if self.scheduler_updates_on_epoch and self.global_step > self.warmup_steps:
                self.main_scheduler.step(val_loss)
            
            # Report to Optuna if this is a hyperparameter search
            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune():
                    raise TrialPruned()
            
            # Log results
            improvement = self.best_val_loss - val_loss
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm, 
                time.time() - start_time, improvement
            )
            
            # Check for improvement and save best model
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
            
            # Save periodic checkpoints
            if epoch % self.save_checkpoint_interval == 0:
                self._checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)
        
        # Final evaluation and export
        self.test()
        self._export_jit_model()
        
        logger.info(
            f"Training completed. Best validation loss: {self.best_val_loss:.4e} "
            f"at epoch {self.best_epoch}."
        )
        return self.best_val_loss

    def _run_epoch(self, loader: DataLoader, is_train_phase: bool) -> Tuple[Optional[float], Optional[float]]:
        """
        Run one epoch of training or validation.
        
        Args:
            loader: DataLoader for the epoch
            is_train_phase: Whether this is training (True) or validation (False)
            
        Returns:
            Tuple of (average loss, average gradient norm)
        """
        self.model.train(is_train_phase)
        total_loss, total_grad_norm, num_batches, num_opt_steps = 0.0, 0.0, 0, 0
        
        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        progress_bar = tqdm(
            loader, desc=desc, leave=False, 
            disable=not self.misc_cfg.get("show_epoch_progress", True)
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                inputs_dict, targets = batch
                inputs = inputs_dict['x'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.set_grad_enabled(is_train_phase):
                    # Forward pass with optional mixed precision
                    #with torch.cuda.amp.autocast(enabled=self.use_amp):
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        preds = self.model(inputs)
                        loss = self.criterion(preds, targets)
                    
                    # Check for invalid loss
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss in batch {batch_idx}. Skipping.")
                        self.total_invalid_batches += 1
                        if self.total_invalid_batches > self.max_invalid_batches:
                            raise RuntimeError("Exceeded maximum number of invalid batches.")
                        continue

                if is_train_phase:
                    # Backward pass with gradient accumulation
                    self.scaler.scale(loss / self.gradient_accumulation).backward()

                    # Optimizer step after accumulation
                    if (batch_idx + 1) % self.gradient_accumulation == 0:
                        # Unscale gradients for clipping
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping with safety check
                        try:
                            grad_norm = nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                        except RuntimeError:
                            # Handle case where all gradients are None
                            grad_norm = torch.tensor(0.0)
                        
                        # Check gradient validity
                        if not torch.isfinite(grad_norm):
                            logger.error("Non-finite gradient norm. Skipping optimizer step.")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue

                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                        # Update schedulers
                        if self.warmup_scheduler and self.global_step < self.warmup_steps:
                            self.warmup_scheduler.step()
                        elif not self.scheduler_updates_on_epoch:
                            self.main_scheduler.step()
                        
                        self.global_step += 1
                        num_opt_steps += 1
                        total_grad_norm += grad_norm.item() if grad_norm.numel() > 0 else 0.0
                        
                        # Update progress bar
                        progress_bar.set_postfix(
                            loss=f"{loss.item():.4e}", 
                            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                        )

                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}", exc_info=True)
                continue

        # Check if any valid batches were processed
        if num_batches == 0:
            logger.error(f"No valid batches processed in {'training' if is_train_phase else 'validation'} epoch.")
            return None, None
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_opt_steps if num_opt_steps > 0 else 0.0
        
        return avg_loss, avg_grad_norm

    def _log_epoch_results(
        self, epoch: int, train_loss: float, val_loss: float, 
        grad_norm: float, duration: float, improvement: float
    ) -> None:
        """
        Log epoch results to console and CSV file.
        
        Args:
            epoch: Current epoch number
            train_loss: Average training loss
            val_loss: Average validation loss
            grad_norm: Average gradient norm
            duration: Epoch duration in seconds
            improvement: Improvement from best validation loss
        """
        lr = self.optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
            f"Val Loss: {val_loss:.4e} | LR: {lr:.2e} | "
            f"Grad: {grad_norm:.2f} | Time: {duration:.1f}s"
        )
        if improvement > 0: 
            log_msg += f" | ↓ {improvement:.4e}"
        logger.info(log_msg)
        
        # Write to CSV
        with self.log_path.open("a") as f:
            f.write(f"{epoch},{train_loss},{val_loss},{lr},{grad_norm},{duration},{improvement}\n")

    def _checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint with all necessary information for resuming.
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        # Handle compiled models
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        
        checkpoint = {
            "state_dict": model_to_save.state_dict(), 
            "epoch": epoch, 
            "val_loss": val_loss,
            "config": self.cfg, 
            "normalization_metadata": self.norm_metadata,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.main_scheduler.state_dict() if hasattr(self.main_scheduler, 'state_dict') else None,
            "global_step": self.global_step,
        }
        torch.save(checkpoint, self.save_dir / filename)
        logger.debug(f"Saved checkpoint: {filename}")

    def test(self) -> None:
        """Evaluate the best model on the test set."""
        ckpt_path = self.save_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("No best_model.pt found, skipping test evaluation.")
            return
        
        logger.info("Loading best model for test evaluation...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_state = ckpt["state_dict"]
        
        # Handle compiled models
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        
        # Run test evaluation
        test_loss, _ = self._run_epoch(self.test_loader, is_train_phase=False)
        if test_loss is not None:
            metrics = {
                "test_loss": test_loss, 
                "best_epoch": ckpt['epoch'], 
                "best_val_loss": ckpt['val_loss']
            }
            logger.info(f"Test Loss: {test_loss:.4e}")
            save_json(metrics, self.save_dir / "test_metrics.json")

    def _export_jit_model(self) -> None:
        """Export the best model as TorchScript for optimized inference."""
        if not self.misc_cfg.get("export_jit_model", True):
            return
        
        try:
            ckpt_path = self.save_dir / "best_model.pt"
            if not ckpt_path.exists(): 
                logger.warning("No best model checkpoint found for JIT export.")
                return
            
            logger.info("Exporting JIT model...")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            # Create fresh model for export
            export_model = create_prediction_model(self.cfg, device=self.device)
            export_model.load_state_dict(ckpt["state_dict"])
            export_model.eval()
            
            # Create example input
            num_species = len(self.data_spec["species_variables"])
            num_global = len(self.data_spec["global_variables"])
            example_input = torch.randn(1, num_species + num_global + 1, device=self.device)
            
            # Export
            jit_path = self.save_dir / "best_model_jit.pt"
            export_model_jit(export_model, example_input, jit_path, optimize=True)
            
        except Exception as e:
            logger.error(f"JIT export failed: {e}", exc_info=True)


__all__ = ["ModelTrainer", "get_warmup_scheduler"]