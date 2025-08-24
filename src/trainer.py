#!/usr/bin/env python3
"""
Training module for AE-DeepONet with learning rate scheduling and PoU regularization.

Features:
- Cosine annealing with warmup learning rate scheduler
- Automatic mixed precision (AMP) support
- Partition-of-Unity (PoU) regularization properly integrated
- Clear logging with normalized space metrics
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


class CosineAnnealingWarmupScheduler:
    """
    Cosine annealing learning rate scheduler with linear warmup.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_epochs: int,
            total_epochs: int,
            min_lr: float = 1e-6,
            base_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']

        # Create warmup scheduler
        if warmup_epochs > 0:
            self.warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            )
        else:
            self.warmup_scheduler = None

        # Create cosine annealing scheduler for post-warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )

        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Advance the scheduler by one epoch."""
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step()
        else:
            # Adjust cosine scheduler epoch
            cosine_epoch = self.current_epoch - self.warmup_epochs
            self.cosine_scheduler.step(cosine_epoch)

        self.current_epoch += 1

    def get_last_lr(self):
        """Get the last computed learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        state = {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr
        }
        if self.warmup_scheduler:
            state['warmup_scheduler'] = self.warmup_scheduler.state_dict()
        state['cosine_scheduler'] = self.cosine_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        if self.warmup_scheduler and 'warmup_scheduler' in state_dict:
            self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


class Trainer:
    """
    Trainer for AE-DeepONet models with proper PoU regularization integration.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            config: Dict[str, Any],
            save_dir: Path,
            device: torch.device,
            is_latent_stage: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.is_latent_stage = is_latent_stage

        # Extract training configuration
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.pou_weight = train_cfg.get("pou_weight", 0.01)

        # Learning rate scheduler configuration
        self.warmup_epochs = train_cfg.get("warmup_epochs", 10)
        self.min_lr = train_cfg.get("min_lr", 1e-6)

        # Setup optimizer (AdamW for weight decay)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=train_cfg.get("betas", (0.9, 0.999))
        )

        # Setup learning rate scheduler
        self.scheduler = None  # Will be initialized for each training stage

        # Setup AMP scaler
        self.scaler = GradScaler('cuda' if self.use_amp else 'cpu', enabled=self.use_amp)

        # Loss function (MSE as per paper)
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        # Metrics tracking
        self.train_history = []
        self.val_history = []

        # Ensure model has history lists
        for name in ("index_list", "train_loss_list", "val_loss_list"):
            if not hasattr(self.model, name):
                setattr(self.model, name, [])

        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled")

    def train_ae_pretrain(self, epochs: int) -> float:
        """
        Stage 1: Pretrain autoencoder with cosine annealing scheduler.
        """
        self.logger.info("Starting autoencoder pretraining...")

        # Initialize scheduler for autoencoder pretraining
        ae_warmup = self.config["training"].get("ae_warmup_epochs", 5)
        self.scheduler = CosineAnnealingWarmupScheduler(
            self.optimizer,
            warmup_epochs=ae_warmup,
            total_epochs=epochs,
            min_lr=self.min_lr,
            base_lr=self.lr
        )

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            current_lr = self.scheduler.get_last_lr()

            for inputs, targets in self.train_loader:
                # Move to device
                targets = targets.to(self.device)  # [B, M, n_species]

                # Flatten for autoencoder
                B, M, D = targets.shape
                targets_flat = targets.reshape(B * M, D)

                self.optimizer.zero_grad(set_to_none=True)

                # Forward with AMP
                with autocast('cuda', enabled=self.use_amp):
                    recon = self.model.autoencoder(targets_flat)
                    loss = self.criterion(recon, targets_flat)

                # Backward
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                n_batches += 1

            avg_loss = train_loss / n_batches
            epoch_time = time.time() - epoch_start_time

            # Step scheduler
            self.scheduler.step()

            # Improved logging
            self.logger.info(
                f"[AE Pretrain] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.3e} | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Update model tracking lists
            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(avg_loss)

            # Save periodically
            if epoch % 10 == 0 or epoch == epochs:
                self._save_checkpoint(epoch, avg_loss, prefix="ae_checkpoint")

        # Save final pretrained autoencoder
        final_checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epochs,
            "loss": avg_loss,
            "config": self.config
        }
        torch.save(final_checkpoint, self.save_dir / "ae_pretrained.pt")

        return avg_loss

    def train_deeponet(self) -> float:
        """
        Stage 3: Train DeepONet on latent space with PoU regularization.
        """
        self.logger.info("Starting DeepONet training on latent space...")

        # Initialize scheduler for DeepONet training
        self.scheduler = CosineAnnealingWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.epochs,
            min_lr=self.min_lr,
            base_lr=self.lr
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            current_lr = self.scheduler.get_last_lr()

            # Training
            train_loss, train_metrics = self._train_epoch()

            # Validation
            val_loss, val_metrics = self._validate()

            epoch_time = time.time() - epoch_start_time

            # Step scheduler
            self.scheduler.step()

            # Improved logging
            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {train_loss:.3e} (MSE: {train_metrics['mse']:.3e}, "
                f"PoU: {train_metrics.get('pou', 0):.3e}) | "
                f"Val: {val_loss:.3e} (MAE: {val_metrics.get('decoded_mae', 0):.3e}) | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Track history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            # Update model tracking lists
            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(train_loss)
            self.model.val_loss_list.append(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, is_best=True)

        self.logger.info(
            f"Training complete. Best val loss: {self.best_val_loss:.3e} at epoch {self.best_epoch}"
        )

        # Save training history
        self._save_history()

        return self.best_val_loss

    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train one epoch with PoU regularization properly integrated.
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_batches = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                if self.is_latent_stage:
                    # DeepONet training: predict latent trajectory with trunk outputs for PoU
                    z_pred, aux = self.model(inputs, decode=False, return_trunk_outputs=True)
                    mse_loss = self.criterion(z_pred, targets)

                    # PoU regularization (active only if use_pou=True and pou_weight>0)
                    pou_loss = self.model.pou_regularization(aux.get("trunk_outputs"))

                    # Combined loss
                    loss = mse_loss + self.pou_weight * pou_loss
                else:
                    # Regular training (autoencoder)
                    y_pred, _ = self.model(inputs, decode=True)
                    mse_loss = self.criterion(y_pred, targets)
                    pou_loss = torch.tensor(0.0)
                    loss = mse_loss

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_pou += pou_loss.item() if isinstance(pou_loss, torch.Tensor) else pou_loss
            n_batches += 1

        metrics = {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "pou": total_pou / n_batches
        }

        return metrics["loss"], metrics

    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate with metrics in NORMALIZED space.

        IMPORTANT: The 'decoded_mae' metric is computed in NORMALIZED space,
        not physical units. This represents the MAE between normalized species
        concentrations (after log10 and z-score transformations), NOT actual
        physical concentrations.
        """
        if self.val_loader is None:
            return float("inf"), {}

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_decoded_mae = 0.0
        n_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                if self.is_latent_stage:
                    # Compute latent MSE
                    z_pred, _ = self.model(inputs, decode=False)
                    loss = self.criterion(z_pred, targets)

                    # Also compute decoded MAE in NORMALIZED space for interpretability
                    y_pred, _ = self.model(inputs, decode=True)

                    # Decode targets for comparison
                    B, M, LD = targets.shape
                    targets_flat = targets.reshape(B * M, LD)
                    y_true_flat = self.model.decode(targets_flat)
                    y_true = y_true_flat.view(B, M, -1)

                    # MAE in normalized space (log10 + z-score transformed)
                    decoded_mae = (y_pred - y_true).abs().mean()
                else:
                    y_pred, _ = self.model(inputs, decode=True)
                    loss = self.criterion(y_pred, targets)
                    decoded_mae = torch.tensor(0.0)

            total_loss += loss.item()
            total_mse += loss.item()
            total_decoded_mae += decoded_mae.item()
            n_batches += 1

        metrics = {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "decoded_mae": total_decoded_mae / n_batches  # Normalized space MAE
        }

        return metrics["loss"], metrics

    def _save_checkpoint(self, epoch: int, loss: float, prefix: str = "checkpoint", is_best: bool = False):
        """Save model checkpoint with all training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.config,
            # Save model tracking lists
            "index_list": self.model.index_list,
            "train_loss_list": self.model.train_loss_list,
            "val_loss_list": self.model.val_loss_list
        }

        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"{prefix}_epoch_{epoch}.pt"

        torch.save(checkpoint, self.save_dir / filename)

        if is_best:
            self.logger.info(f"Saved best model at epoch {epoch} with loss {loss:.3e}")

    def _save_history(self):
        """Save complete training history to JSON file."""
        import json

        history = {
            "train": self.train_history,
            "val": self.val_history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            # Include model tracking lists
            "index_list": self.model.index_list,
            "train_loss_list": self.model.train_loss_list,
            "val_loss_list": self.model.val_loss_list,
            # Include configuration
            "config": self.config
        }

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)