#!/usr/bin/env python3
"""
train.py - Optimized training pipeline with dataset caching
"""
from __future__ import annotations

import gc
import logging
import random
import time
import warnings
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

from dataset import create_dataset, collate_fn
from hardware import configure_dataloader_settings
from model import create_prediction_model, export_model_jit
from normalizer import DataNormalizer
from utils import save_json

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 100
DEFAULT_LR = 5e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-8
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_GRADIENT_ACCUMULATION = 1
DEFAULT_MAX_INVALID_BATCHES = 100
DEFAULT_INVALID_BATCH_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 0  # Safe default for HDF5


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int = -1
) -> LambdaLR:
    """
    Create linear warmup scheduler.
    
    Gradually increases learning rate from 0 to target over warmup_steps.
    """
    def warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))
    
    return LambdaLR(optimizer, warmup_lambda)


class ModelTrainer:
    """
    Optimized training pipeline manager with dataset caching support.
    
    Features:
    - Automatic mixed precision (optional)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Invalid batch tracking
    - Checkpoint management
    - HDF5-aware data loading
    - Optional dataset caching for faster training
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
    ):
        """Initialize the trainer with all necessary components."""
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
        self.save_checkpoint_interval = self.misc_cfg.get(
            "save_checkpoint_every_n_epochs", 10
        )
        
        # Dataset caching
        self.cache_dataset = self.misc_cfg.get("cache_dataset", False)
        if self.cache_dataset:
            logger.info("Dataset caching enabled - will load all data into memory")
        
        # Anomaly detection for debugging
        if self.misc_cfg.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - this will slow down training!")

        # Setup components in order
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
        self._setup_logging()
        self._save_metadata()
        
        # Force garbage collection after setup
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_num_workers(self) -> int:
        """
        Get the number of dataloader workers, with caching and HDF5 awareness.
        
        The streaming ChemicalDataset is designed for multi-worker HDF5 access,
        so we enable it here.
        """
        # First, check for an explicit integer value in the config
        num_workers_cfg = self.misc_cfg.get("num_dataloader_workers")
        if isinstance(num_workers_cfg, int):
            # Respect explicit user configuration
            if num_workers_cfg > 0 and not self.cache_dataset:
                logger.info(
                    f"Using {num_workers_cfg} workers with HDF5 streaming as configured."
                )
            return num_workers_cfg

        # If config is "auto" or not set, determine the optimal number
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Reserve at least one CPU for the main process and other tasks
        available_cpus = max(1, cpu_count - 1)

        if self.cache_dataset:
            # Cached dataset is in RAM, it's always safe to use multiple workers.
            # An A100 can handle many workers. Let's be aggressive.
            default_workers = min(12, available_cpus)
            logger.info(f"Auto-configuring num_workers: {default_workers} (Cached Dataset in RAM)")
            return default_workers
        
        # If we are here, it's the streaming ChemicalDataset.
        # It is designed to be multi-worker safe. Let's enable it.
        # Start with a healthy but not excessive number to avoid lock contention.
        default_workers = min(8, available_cpus)
        logger.info(
            f"Auto-configuring num_workers: {default_workers} "
            f"(Streaming HDF5 with optimized parallel-safe reader)"
        )
        return default_workers

    def _setup_datasets_with_precalculated_stats(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Creates datasets using pre-calculated normalization metadata."""
        # Get memory settings
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
        if hasattr(self.train_ds, '__len__'):
            # Cached dataset
            train_samples = len(self.train_ds)
            val_samples = len(self.val_ds)
            test_samples = len(self.test_ds)
        else:
            # Streaming dataset
            train_samples = self.train_ds.total_samples
            val_samples = self.val_ds.total_samples
            test_samples = self.test_ds.total_samples
        
        logger.info(
            f"Datasets created - Train: {train_samples:,}, "
            f"Val: {val_samples:,}, Test: {test_samples:,}"
        )

    def _setup_normalization_and_datasets(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Calculate normalization and create datasets."""
        train_indices = splits['train']
        val_indices = splits['validation']
        test_indices = splits['test']
        
        # Handle data fraction if specified
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
                f"Using {data_fraction:.1%} of data. "
                f"Train: {new_sizes[0]:,}/{original_sizes[0]:,}, "
                f"Val: {new_sizes[1]:,}/{original_sizes[1]:,}, "
                f"Test: {new_sizes[2]:,}/{original_sizes[2]:,}"
            )

        self.test_set_indices = test_indices

        # Calculate normalization statistics
        logger.info("Calculating normalization statistics from training set...")
        normalizer = DataNormalizer(config_data=self.cfg)
        self.norm_metadata = normalizer.calculate_stats(h5_path, train_indices)
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
        
        self.train_ds = create_dataset(indices=train_indices, **ds_kwargs)
        self.val_ds = create_dataset(indices=val_indices, **ds_kwargs)
        self.test_ds = create_dataset(indices=test_indices, **ds_kwargs)
        
        # Get sample counts
        if hasattr(self.train_ds, '__len__'):
            # Cached dataset
            train_samples = len(self.train_ds)
            val_samples = len(self.val_ds)
            test_samples = len(self.test_ds)
        else:
            # Streaming dataset
            train_samples = self.train_ds.total_samples
            val_samples = self.val_ds.total_samples
            test_samples = self.test_ds.total_samples
        
        logger.info(
            f"Datasets created - Train: {train_samples:,}, "
            f"Val: {val_samples:,}, Test: {test_samples:,}"
        )

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Create optimized data loaders with proper worker configuration."""
        # Get number of workers with caching awareness
        num_workers = self._get_num_workers()
        
        # Get hardware-specific settings
        hw_settings = configure_dataloader_settings()
        batch_size = self.train_params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        # Warn if batch size seems too large
        if batch_size > 10000:
            logger.warning(
                f"Large batch size ({batch_size}) may cause memory issues. "
                "Consider reducing if you encounter OOM errors."
            )
        
        # Base dataloader arguments
        dl_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": hw_settings.get("pin_memory", False) and self.device.type == "cuda",
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
        
        # Add shuffle for cached datasets
        if hasattr(self.train_ds, '__len__'):
            # Cached dataset supports shuffle
            self.train_loader = DataLoader(self.train_ds, shuffle=True, **dl_kwargs)
            self.val_loader = DataLoader(self.val_ds, shuffle=False, **dl_kwargs)
            self.test_loader = DataLoader(self.test_ds, shuffle=False, **dl_kwargs)
        else:
            # IterableDataset doesn't support shuffle parameter
            self.train_loader = DataLoader(self.train_ds, **dl_kwargs)
            self.val_loader = DataLoader(self.val_ds, **dl_kwargs)
            self.test_loader = DataLoader(self.test_ds, **dl_kwargs)
        
        logger.info(
            f"DataLoaders created with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={dl_kwargs['pin_memory']}, "
            f"cached={self.cache_dataset}"
        )

    def _build_model(self) -> None:
        """Create and optionally compile the model."""
        self.model = create_prediction_model(self.cfg, device=self.device)
        
        # Check for torch compile option
        use_compile = self.misc_cfg.get("use_torch_compile", False)
        if use_compile and self.device.type == 'cuda':
            try:
                compile_mode = self.misc_cfg.get("torch_compile_mode", "reduce-overhead")
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info(f"Model compiled with torch.compile(mode='{compile_mode}')")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    def _build_optimizer(self) -> None:
        """Create AdamW optimizer with weight decay settings."""
        lr = self.train_params.get("learning_rate", DEFAULT_LR)
        weight_decay = self.train_params.get("weight_decay", 1e-5)
        
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to bias and normalization parameters
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Create AdamW optimizer
        self.optimizer = optim.AdamW(
            param_groups, 
            lr=lr, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        
        logger.info(
            f"Optimizer: AdamW with lr={lr:.2e}, weight_decay={weight_decay:.2e}"
        )

    def _build_schedulers(self) -> None:
        """Create learning rate schedulers (cosine or plateau)."""
        # Calculate steps per epoch
        if hasattr(self.train_ds, '__len__'):
            # Cached dataset - exact length known
            total_samples = len(self.train_ds)
        else:
            # Streaming dataset - estimate
            total_samples = self.train_ds.total_samples
        
        batch_size = self.train_params.get("batch_size", DEFAULT_BATCH_SIZE)
        self.gradient_accumulation = self.train_params.get(
            "gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION
        )
        
        # Calculate effective steps per epoch
        self.steps_per_epoch = max(
            1, total_samples // (batch_size * self.gradient_accumulation)
        )
        
        logger.info(
            f"Scheduler configured with ~{self.steps_per_epoch} steps per epoch "
            f"(gradient accumulation: {self.gradient_accumulation})"
        )
        
        # Create main scheduler
        scheduler_name = self.train_params.get("scheduler_choice", "plateau").lower()
        
        if scheduler_name == "plateau":
            self.main_scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=self.train_params.get("factor", 0.5),
                patience=self.train_params.get("patience", 10),
                min_lr=self.train_params.get("min_lr", 1e-7),
                verbose=True
            )
            self.scheduler_needs_loss = True
            
        elif scheduler_name == "cosine":
            t0 = self.train_params.get("cosine_T_0", 10)
            self.main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0 * self.steps_per_epoch,
                T_mult=self.train_params.get("cosine_T_mult", 2),
                eta_min=self.train_params.get("min_lr", 1e-7)
            )
            self.scheduler_needs_loss = False
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. Use 'plateau' or 'cosine'.")
        
        # Setup warmup scheduler
        self.warmup_epochs = self.train_params.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS)
        if self.warmup_epochs > 0:
            self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
            self.warmup_scheduler = get_warmup_scheduler(
                self.optimizer, self.warmup_steps, -1
            )
            logger.info(f"Warmup enabled for {self.warmup_epochs} epochs ({self.warmup_steps} steps)")
        else:
            self.warmup_scheduler = None
            self.warmup_steps = 0

    def _setup_loss_and_training_params(self) -> None:
        """Setup loss function (MSE or Huber) and training parameters."""
        loss_name = self.train_params.get("loss_function", "mse").lower()
        
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            delta = self.train_params.get("huber_delta", 0.1)
            self.criterion = nn.HuberLoss(delta=delta)
            logger.info(f"Using Huber loss with delta={delta}")
        else:
            logger.warning(f"Unknown loss '{loss_name}', using MSE")
            self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = self.train_params.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled")
        
        # Gradient clipping
        self.max_grad_norm = self.train_params.get("gradient_clip_val", DEFAULT_GRAD_CLIP)
        self.grad_clip_mode = self.train_params.get("gradient_clip_mode", "norm")

    def _setup_logging(self) -> None:
        """Setup training logs and metrics tracking."""
        # CSV log for training metrics
        self.log_path = self.save_dir / "training_log.csv"
        headers = [
            "epoch", "train_loss", "val_loss", "lr", "grad_norm", 
            "time_s", "improvement", "invalid_batches"
        ]
        if self.log_gradient_norms:
            headers.extend(["grad_min", "grad_max", "grad_mean"])
        
        self.log_path.write_text(",".join(headers) + "\n")
        
        # Initialize tracking variables
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.global_step = 0
        self.total_invalid_batches = 0

    def _save_metadata(self) -> None:
        """Save training metadata for reproducibility."""
        # Get dataset info
        if hasattr(self.train_ds, '__len__'):
            train_samples = len(self.train_ds)
            val_samples = len(self.val_ds)
            test_samples = len(self.test_ds)
        else:
            train_samples = self.train_ds.total_samples
            val_samples = self.val_ds.total_samples
            test_samples = self.test_ds.total_samples
        
        metadata = {
            "test_set_indices": sorted(self.test_set_indices),
            "num_train_samples": train_samples,
            "num_val_samples": val_samples,
            "num_test_samples": test_samples,
            "num_train_profiles": len(self.train_ds.indices),
            "num_val_profiles": len(self.val_ds.indices),
            "num_test_profiles": len(self.test_ds.indices),
            "gradient_accumulation_steps": self.gradient_accumulation,
            "effective_batch_size": (
                self.train_params.get("batch_size", DEFAULT_BATCH_SIZE) * 
                self.gradient_accumulation
            ),
            "max_invalid_batches": self.max_invalid_batches,
            "invalid_batch_threshold": self.invalid_batch_threshold,
            "num_workers": self._get_num_workers(),
            "device": str(self.device),
            "cache_dataset": self.cache_dataset,
        }
        save_json(metadata, self.save_dir / "training_metadata.json")

    def train(self) -> float:
        """
        Main training loop with improved stability checks.
        
        Returns:
            Best validation loss achieved during training.
        """
        epochs = self.train_params.get("epochs", DEFAULT_EPOCHS)
        patience = self.train_params.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = self.train_params.get("min_delta", DEFAULT_MIN_DELTA)
        
        export_jit_during_training = self.misc_cfg.get("export_jit_during_training", False)
        
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for up to {epochs} epochs...")
        logger.info(f"Early stopping patience: {patience} epochs")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation}")
        logger.info(
            f"Effective batch size: "
            f"{self.train_params.get('batch_size', DEFAULT_BATCH_SIZE) * self.gradient_accumulation}"
        )
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_results = self._run_epoch(self.train_loader, is_train_phase=True)
            train_loss = train_results["loss"]
            train_grad_norm = train_results["grad_norm"]
            epoch_invalid_batches = train_results["invalid_batches"]
            
            # Check for training instability
            if (self.steps_per_epoch > 0 and 
                epoch_invalid_batches > self.steps_per_epoch * self.invalid_batch_threshold):
                logger.error(
                    f"Too many invalid batches in epoch {epoch}: "
                    f"{epoch_invalid_batches}/{self.steps_per_epoch} "
                    f"(>{self.invalid_batch_threshold:.0%} threshold)"
                )
                if self.optuna_trial:
                    raise TrialPruned("Too many invalid batches")
                else:
                    raise RuntimeError("Training instability detected - too many invalid batches")
            
            # Validation phase
            val_results = self._run_epoch(self.val_loader, is_train_phase=False)
            val_loss = val_results["loss"]
            
            # Update learning rate scheduler
            if self.global_step > self.warmup_steps:
                if self.scheduler_needs_loss:
                    self.main_scheduler.step(val_loss)
                # Note: step-based schedulers are updated in _run_epoch
            
            # Report to Optuna if using hyperparameter search
            if self.optuna_trial:
                self.optuna_trial.report(val_loss, epoch)
                if self.optuna_trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch}")
                    raise TrialPruned()
            
            # Calculate improvement
            improvement = self.best_val_loss - val_loss
            
            # Log results
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm,
                time.time() - start_time, improvement, epoch_invalid_batches,
                train_results.get("grad_stats")
            )
            
            # Save checkpoint at intervals
            if epoch % self.save_checkpoint_interval == 0:
                self._checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch, val_loss)
                
                if export_jit_during_training:
                    self._export_jit_model(suffix=f"_epoch{epoch}")
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break
            
            # Check total invalid batches
            if self.total_invalid_batches > self.max_invalid_batches:
                logger.error(
                    f"Total invalid batches ({self.total_invalid_batches}) "
                    f"exceeded maximum ({self.max_invalid_batches})"
                )
                raise RuntimeError("Training instability - too many total invalid batches")
            
            # Periodic garbage collection
            if epoch % 5 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Save final checkpoint
        self._checkpoint("final_model.pt", epoch, val_loss)
        
        # Test the best model
        self.test()
        
        # Export JIT model
        self._export_jit_model()
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch}")
        
        return self.best_val_loss

    def _run_epoch(
        self, loader: DataLoader, is_train_phase: bool
    ) -> Dict[str, Any]:
        """
        Run one epoch of training or validation.
        
        Returns:
            Dictionary with loss, gradient norm, and other metrics.
        """
        self.model.train(is_train_phase)
        
        # Initialize metrics
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        num_optimizer_steps = 0
        invalid_batches = 0
        grad_norms = [] if self.log_gradient_norms and is_train_phase else None
        
        # Progress bar settings
        show_progress = self.misc_cfg.get("show_epoch_progress", True)
        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        
        # Get total batches
        if hasattr(loader.dataset, '__len__'):
            # Cached dataset - exact length
            total_batches = len(loader)
        else:
            # Streaming dataset - estimate
            total_batches = self.steps_per_epoch if is_train_phase else None
        
        progress_bar = tqdm(
            loader, 
            desc=desc, 
            leave=False, 
            disable=not show_progress,
            total=total_batches
        )

        # Context manager for gradient computation
        with torch.set_grad_enabled(is_train_phase):
            for batch_idx, (inputs_dict, targets) in enumerate(progress_bar):
                try:
                    # Move data to device
                    inputs = inputs_dict['x'].to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    # Forward pass with optional AMP
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        preds = self.model(inputs)
                        loss = self.criterion(preds, targets)
                    
                    # Check for valid loss
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss detected in batch {batch_idx}")
                        invalid_batches += 1
                        self.total_invalid_batches += 1
                        continue
                    
                    # Training step
                    if is_train_phase:
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.gradient_accumulation
                        
                        # Backward pass
                        if self.use_amp:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        
                        # Gradient accumulation and optimizer step
                        if ((batch_idx + 1) % self.gradient_accumulation == 0 or
                            (batch_idx + 1) == total_batches):
                            
                            # Gradient clipping
                            if self.use_amp:
                                self.scaler.unscale_(self.optimizer)
                            
                            if self.grad_clip_mode == "norm":
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.max_grad_norm
                                )
                            else:  # value clipping
                                torch.nn.utils.clip_grad_value_(
                                    self.model.parameters(), self.max_grad_norm
                                )
                                # Calculate norm for logging
                                grad_norm = torch.norm(
                                    torch.stack([
                                        torch.norm(p.grad.detach(), 2)
                                        for p in self.model.parameters()
                                        if p.grad is not None
                                    ]), 2
                                )
                            
                            # Check for valid gradients
                            if not torch.isfinite(grad_norm):
                                logger.error("Non-finite gradients detected!")
                                self.optimizer.zero_grad(set_to_none=True)
                                invalid_batches += 1
                                self.total_invalid_batches += 1
                                continue
                            
                            # Optimizer step
                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            
                            self.optimizer.zero_grad(set_to_none=True)
                            
                            # Update metrics
                            self.global_step += 1
                            num_optimizer_steps += 1
                            total_grad_norm += grad_norm.item()
                            
                            if grad_norms is not None:
                                grad_norms.append(grad_norm.item())
                            
                            # Update schedulers
                            if self.global_step <= self.warmup_steps and self.warmup_scheduler:
                                self.warmup_scheduler.step()
                            elif not self.scheduler_needs_loss and self.global_step > self.warmup_steps:
                                self.main_scheduler.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    if show_progress:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        postfix = {
                            "loss": f"{loss.item():.4e}",
                            "lr": f"{current_lr:.2e}",
                        }
                        if invalid_batches > 0:
                            postfix["invalid"] = invalid_batches
                        progress_bar.set_postfix(postfix)
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}", exc_info=True)
                    invalid_batches += 1
                    self.total_invalid_batches += 1
                    
                    if is_train_phase:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_grad_norm = total_grad_norm / num_optimizer_steps if num_optimizer_steps > 0 else 0.0
        
        results = {
            "loss": avg_loss,
            "grad_norm": avg_grad_norm,
            "invalid_batches": invalid_batches,
            "num_batches": num_batches,
        }
        
        # Add gradient statistics if available
        if grad_norms and len(grad_norms) > 0:
            results["grad_stats"] = {
                "min": min(grad_norms),
                "max": max(grad_norms),
                "mean": sum(grad_norms) / len(grad_norms),
                "std": torch.std(torch.tensor(grad_norms)).item(),
            }
        
        return results

    def _log_epoch_results(
        self, epoch: int, train_loss: float, val_loss: float, 
        grad_norm: float, duration: float, improvement: float, 
        invalid_batches: int, grad_stats: Optional[Dict[str, float]] = None
    ) -> None:
        """Log epoch results to console and CSV file."""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Console logging
        log_parts = [
            f"Epoch {epoch:03d}",
            f"Train: {train_loss:.4e}",
            f"Val: {val_loss:.4e}",
            f"LR: {lr:.2e}",
            f"Grad: {grad_norm:.2f}",
            f"Time: {duration:.1f}s",
        ]
        
        if invalid_batches > 0:
            log_parts.append(f"Invalid: {invalid_batches}")
        
        if improvement > 0:
            log_parts.append(f"↓ {improvement:.4e}")
        
        logger.info(" | ".join(log_parts))
        
        # CSV logging
        row_data = [
            epoch, train_loss, val_loss, lr, grad_norm,
            duration, improvement, invalid_batches
        ]
        
        if self.log_gradient_norms and grad_stats:
            row_data.extend([
                grad_stats['min'], grad_stats['max'], grad_stats['mean']
            ])
        
        row = ",".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in row_data)
        
        with self.log_path.open("a") as f:
            f.write(row + "\n")

    def _checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint with metadata."""
        # Handle compiled models
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod
        
        checkpoint = {
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg,
            "normalization_metadata": self.norm_metadata,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.main_scheduler.state_dict() 
                if hasattr(self.main_scheduler, "state_dict") else None
            ),
            "global_step": self.global_step,
            "total_invalid_batches": self.total_invalid_batches,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {filename}")

    def test(self) -> None:
        """Test the best model on test set."""
        ckpt_path = self.save_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("No best model found, skipping test.")
            return
        
        # Load best model
        logger.info("Loading best model for testing...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt["state_dict"])
        
        logger.info(f"Testing model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4e})")
        
        # Run test
        test_results = self._run_epoch(self.test_loader, is_train_phase=False)
        test_loss = test_results["loss"]
        
        # Save test metrics
        metrics = {
            "test_loss": test_loss,
            "best_epoch": ckpt['epoch'],
            "best_val_loss": ckpt['val_loss'],
            "test_invalid_batches": test_results["invalid_batches"],
            "test_num_batches": test_results["num_batches"],
        }
        
        logger.info(f"Test Loss: {test_loss:.4e}")
        save_json(metrics, self.save_dir / "test_metrics.json")

    def _export_jit_model(self, suffix: str = "") -> None:
        """Export model as TorchScript for inference."""
        if not self.misc_cfg.get("export_jit_model", True):
            logger.info("JIT export disabled in config.")
            return
        
        try:
            # Load best model
            ckpt_path = self.save_dir / "best_model.pt"
            if not ckpt_path.exists():
                logger.warning("No best model found for JIT export.")
                return
            
            logger.info("Exporting JIT model...")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            # Create fresh model for export
            fresh_model = create_prediction_model(self.cfg, device=self.device)
            fresh_model.load_state_dict(ckpt["state_dict"])
            fresh_model.eval()
            
            # Create example input
            num_species = len(self.data_spec["species_variables"])
            num_global = len(self.data_spec["global_variables"])
            example_input = torch.randn(
                1, num_species + num_global + 1,  # +1 for time
                device=self.device
            )
            
            # Export
            jit_path = self.save_dir / f"best_model_jit{suffix}.pt"
            export_model_jit(fresh_model, example_input, jit_path, optimize=True)
            
        except Exception as e:
            logger.error(f"JIT export failed: {e}", exc_info=True)


__all__ = ["ModelTrainer", "get_warmup_scheduler"]