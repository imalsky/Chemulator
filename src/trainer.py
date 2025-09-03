#!/usr/bin/env python3
"""
Flow-map DeepONet Trainer
==========================
Training loop implementation with support for multi-time-per-anchor targets.

Features:
- Shape-agnostic loss computation (handles both [B,S] and [B,K,S] targets)
- Mixed precision training with automatic mixed precision (AMP)
- AdamW optimizer with cosine annealing and linear warmup
- Deterministic per-epoch sampling via dataset.set_epoch()
- CSV logging for training metrics
- Model checkpointing for best validation loss
"""

from __future__ import annotations

import csv
import math
import time
import contextlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    Schedule:
        - Warmup phase (epoch < warmup_epochs):
            lr = base_lr * (epoch + 1) / warmup_epochs
        - Cosine phase (epoch >= warmup_epochs):
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
            where progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)
        
        # Store initial learning rates
        self.base_lrs = [group.get("lr", 1e-3) for group in self.optimizer.param_groups]
        for idx, group in enumerate(self.optimizer.param_groups):
            group["initial_lr"] = float(self.base_lrs[idx])
    
    @torch.no_grad()
    def step(self, epoch: int) -> float:
        """
        Update learning rate for given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Current learning rate
        """
        epoch = int(epoch)
        total = max(1, self.total_epochs)
        warmup = min(self.warmup_epochs, total - 1)
        
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = float(self.base_lrs[idx])
            
            if epoch < warmup and warmup > 0:
                # Linear warmup
                lr = base_lr * (epoch + 1) / warmup
            else:
                # Cosine annealing
                progress = (epoch - warmup) / max(1, total - warmup)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            
            param_group["lr"] = lr
        
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    """
    Trainer for Flow-map DeepONet models.
    
    Handles training loop, validation, checkpointing, and logging.
    Supports both single-time (K=1) and multi-time (K>1) predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        work_dir: Path,
        device: torch.device,
        logger: Optional[Any] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            cfg: Configuration dictionary
            work_dir: Working directory for outputs
            device: Compute device
            logger: Logger instance (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.logger = logger or self._create_null_logger()
        
        # Setup working directory
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract training configuration
        training_cfg = cfg.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 100))
        self.base_lr = float(training_cfg.get("lr", 1e-3))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(training_cfg.get("gradient_clip", 0.0))
        self.max_train_steps_per_epoch = training_cfg.get("max_train_steps_per_epoch", None)
        self.max_val_batches = training_cfg.get("max_val_batches", None)
        self.use_compile = bool(training_cfg.get("torch_compile", False))
        
        # Setup mixed precision training
        self._setup_mixed_precision()
        
        # Initialize loss function
        self.criterion = nn.MSELoss(reduction="mean")
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            total_epochs=self.epochs,
            warmup_epochs=int(training_cfg.get("warmup_epochs", 0)),
            min_lr=float(training_cfg.get("min_lr", 1e-6)),
        )
        
        # Optionally compile model for better performance
        if self.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False
            )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize best validation loss
        self.best_val_loss = float("inf")
        self.best_model_path = self.work_dir / "best_model.pt"
    
    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training settings."""
        mixed_precision_cfg = self.cfg.get("mixed_precision", {})
        self.amp_mode = str(mixed_precision_cfg.get("mode", "bf16")).lower()

        if self.amp_mode not in ("bf16", "fp16", "none"):
            self.amp_mode = "bf16"

        # autocast dtype
        if self.amp_mode == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif self.amp_mode == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = None

        # GradScaler only makes sense for fp16 on CUDA
        self.use_scaler = (self.amp_mode == "fp16" and self.device.type == "cuda")
        if self.use_scaler:
            # Use the CUDA GradScaler; no positional args
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

    
    def _setup_optimizer(self) -> None:
        """Initialize optimizer with fused operations if available."""
        # Check if fused AdamW is available (requires CUDA compute capability >= 8.0)
        use_fused = False
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                use_fused = capability[0] >= 8
            except Exception:
                use_fused = False
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            fused=use_fused,
        )
    
    def _setup_logging(self) -> None:
        """Initialize CSV logging file."""
        self.log_file = self.work_dir / "training_log.txt"
        
        # Create header if file doesn't exist
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
    
    def _create_null_logger(self):
        """Create a null logger that ignores all log messages."""
        class NullLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
            def debug(self, *args, **kwargs): pass
        return NullLogger()
    
    def train(self) -> float:
        """
        Execute training loop.
        
        Returns:
            Best validation loss achieved during training
        """
        start_time = time.perf_counter()
        
        for epoch in range(1, self.epochs + 1):
            # Set epoch for deterministic sampling in dataset
            self._set_dataset_epoch(epoch)
            
            epoch_start = time.perf_counter()
            
            # Training epoch
            train_loss = self._run_epoch(train=True)
            
            # Validation epoch
            if self.val_loader is not None:
                val_loss = self._run_epoch(train=False)
            else:
                val_loss = train_loss
            
            # Update learning rate
            current_lr = self.scheduler.step(epoch - 1)
            
            # Save best model
            saved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self._save_checkpoint()
                saved = True
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, val_loss, current_lr)
            
            # Console output
            epoch_time = time.perf_counter() - epoch_start
            loss_delta = val_loss - self.best_val_loss
            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4e} | "
                f"val_loss={val_loss:.4e} | "
                f"lr={current_lr:.2e} | "
                f"time={epoch_time:.1f}s | "
                f"delta_best={loss_delta:+.2e} | "
                f"saved={'yes' if saved else 'no'}"
            )
        
        total_time = time.perf_counter() - start_time
        self.logger.info(
            f"Training completed in {total_time/3600:.2f} hours. "
            f"Best validation loss: {self.best_val_loss:.4e}"
        )
        
        return self.best_val_loss
    
    def _set_dataset_epoch(self, epoch: int) -> None:
        """Set epoch for datasets that support deterministic sampling."""
        for loader in (self.train_loader, self.val_loader):
            if loader is not None:
                dataset = getattr(loader, "dataset", None)
                if dataset is not None and hasattr(dataset, "set_epoch"):
                    try:
                        dataset.set_epoch(epoch)
                    except Exception:
                        pass
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint with configuration."""
        checkpoint = {
            "model": self.model.state_dict(),
            "config": self.cfg,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.best_model_path)
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float
    ) -> None:
        """Append epoch metrics to CSV log file."""
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4e},{val_loss:.4e},{lr:.4e}\n")
    
    def _run_epoch(self, train: bool) -> float:
        """
        Run single training or validation epoch.
        
        Args:
            train: Whether to run training (True) or validation (False)
            
        Returns:
            Average loss for the epoch
        """
        # Select appropriate loader
        loader = self.train_loader if train else self.val_loader
        if loader is None:
            return float("nan")

        # Set model mode
        self.model.train(mode=train)

        # Setup autocast context
        if self.autocast_dtype is not None:
            autocast_context = torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype
            )
        else:
            autocast_context = contextlib.nullcontext()

        # Epoch statistics
        total_loss = 0.0
        num_batches = 0

        # Determine max steps for this epoch
        max_steps = None
        if train and self.max_train_steps_per_epoch:
            max_steps = self.max_train_steps_per_epoch
        elif not train and self.max_val_batches:
            max_steps = self.max_val_batches

        # No graph building during validation
        outer_ctx = torch.inference_mode() if not train else contextlib.nullcontext()
        with outer_ctx:
            # Process batches
            for step, batch in enumerate(loader, 1):
                # Check step limit
                if max_steps and step > max_steps:
                    break

                # Process batch
                loss = self._process_batch(
                    batch, train, autocast_context
                )

                total_loss += float(loss)
                num_batches += 1

        return total_loss / max(1, num_batches)
    
    def _process_batch(
        self,
        batch: tuple,
        train: bool,
        autocast_context: contextlib.AbstractContextManager
    ) -> float:
        """
        Process single batch.
        
        Args:
            batch: Batch data tuple
            train: Whether to compute gradients and update weights
            autocast_context: Context manager for mixed precision
            
        Returns:
            Batch loss value
        """
        # Unpack batch (supports both 5 and 6 element batches)
        if len(batch) == 6:
            y_i, dt_norm, y_j, g, ij, k_mask = batch
        else:
            y_i, dt_norm, y_j, g, ij = batch
            k_mask = None
        
        # Move to device if needed
        if y_i.device != self.device:
            y_i = y_i.to(self.device, non_blocking=True)
            dt_norm = dt_norm.to(self.device, non_blocking=True)
            y_j = y_j.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)
            if k_mask is not None:
                k_mask = k_mask.to(self.device, non_blocking=True)
        
        # Zero gradients
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with autocast
        with autocast_context:
            # ---- SQUEEZE dt_norm from [B,K,1] -> [B,K] when needed ----
            if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                dt_in = dt_norm.squeeze(-1)
            else:
                dt_in = dt_norm

            pred = self.model(y_i, dt_in, g)
            pred, y_j = self._harmonize_shapes(pred, y_j)
            loss = self._compute_loss(pred, y_j, k_mask)
        
        # Backward pass and optimization
        if train:
            self._backward_and_step(loss)
        
        return loss.detach().item()
    
    def _harmonize_shapes(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and ensure prediction and target have compatible shapes.
        
        Args:
            pred: Model predictions
            target: Target values
            
        Returns:
            Tuple of (pred, target) with compatible shapes
            
        Raises:
            RuntimeError: If shapes are incompatible
        """
        # Expected cases:
        # 1. Both have same shape: [B, S] or [B, K, S] - OK
        # 2. pred=[B, K, S] and target=[B, K, S] where K matches - OK
        # 3. Dataset returns [B, 1, S] and model returns [B, 1, S] - OK
        
        # Check for shape mismatches that indicate a bug
        if pred.ndim == 2 and target.ndim == 3:
            # Model returned [B, S] but target is [B, K, S]
            # This indicates the model is not handling multi-time correctly
            B_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}. "
                f"The model should return [B, K, S] when K={K_target} times are requested. "
                f"Check that the model's forward() method properly handles dt_norm shape."
            )
        
        elif pred.ndim == 3 and target.ndim == 2:
            # Model returned [B, K, S] but target is [B, S]
            # This shouldn't happen with current dataset implementation
            B_pred, K_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}. "
                f"This suggests a dataset configuration issue. "
                f"Check dataset.times_per_anchor and dataset.multi_time_per_anchor settings."
            )
        
        elif pred.ndim == 3 and target.ndim == 3:
            # Both are 3D - verify K dimension matches
            B_pred, K_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            
            if K_pred != K_target:
                raise RuntimeError(
                    f"Shape mismatch in time dimension: "
                    f"Model predicted K={K_pred} times but target has K={K_target} times. "
                    f"Predictions shape: {tuple(pred.shape)}, Target shape: {tuple(target.shape)}"
                )
            
            if B_pred != B_target:
                raise RuntimeError(
                    f"Batch size mismatch: "
                    f"Predictions batch={B_pred}, Target batch={B_target}"
                )
            
            if S_pred != S_target:
                raise RuntimeError(
                    f"State dimension mismatch: "
                    f"Predictions S={S_pred}, Target S={S_target}. "
                    f"Check model configuration and target_species_variables."
                )
        
        elif pred.ndim == 2 and target.ndim == 2:
            # Both are 2D - verify dimensions match
            B_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            
            if B_pred != B_target:
                raise RuntimeError(
                    f"Batch size mismatch: "
                    f"Predictions batch={B_pred}, Target batch={B_target}"
                )
            
            if S_pred != S_target:
                raise RuntimeError(
                    f"State dimension mismatch: "
                    f"Predictions S={S_pred}, Target S={S_target}. "
                    f"Check model configuration and target_species_variables."
                )
        
        else:
            # Unexpected dimensionality
            raise RuntimeError(
                f"Unexpected tensor dimensions: "
                f"Predictions ndim={pred.ndim} shape={tuple(pred.shape)}, "
                f"Target ndim={target.ndim} shape={tuple(target.shape)}"
            )
        
        return pred, target
    
    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            pred: Predictions [B, K, S]
            target: Targets [B, K, S]
            mask: Optional mask for valid samples [B, K]
            
        Returns:
            Scalar loss tensor
        """
        # Element-wise squared error
        loss_elements = (pred - target) ** 2
        
        # Apply mask if provided
        if mask is not None:
            if loss_elements.ndim == 3 and mask.ndim == 2:
                # Expand mask from [B, K] to [B, K, S]
                # Use expand instead of unsqueeze to properly broadcast
                B, K, S = loss_elements.shape
                mask_expanded = mask.unsqueeze(-1).expand(B, K, S)
                
                # Check if there are any valid samples
                if mask.any():
                    # Compute mean only over valid positions
                    loss = (loss_elements * mask_expanded).sum() / mask_expanded.sum()
                else:
                    # No valid samples, return zero loss to avoid NaN
                    loss = torch.tensor(0.0, device=loss_elements.device, dtype=loss_elements.dtype)
            else:
                # Direct masking for 2D case
                if mask.any():
                    loss = (loss_elements * mask).sum() / mask.sum()
                else:
                    loss = torch.tensor(0.0, device=loss_elements.device, dtype=loss_elements.dtype)
        else:
            loss = loss_elements.mean()
        
        return loss
    
    def _backward_and_step(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass and optimizer step.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        if self.use_scaler and self.scaler is not None:
            # Scaled backward pass for fp16
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip
                )
            
            # Optimizer step
            self.optimizer.step()