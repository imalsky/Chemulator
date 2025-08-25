#!/usr/bin/env python3
"""
Training module for AE-DeepONet with learning rate scheduling and PoU regularization.

Features:
- Cosine annealing with warmup learning rate scheduler
- Automatic mixed precision (AMP) support
- Partition-of-Unity (PoU) regularization properly integrated
- Clear logging with normalized space metrics
- Support for flexible time point sampling during training
- UPDATED: Clarified loss accumulation (not gradient accumulation)
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

    Combines linear warmup phase with cosine annealing decay for smooth
    learning rate transitions and improved training stability.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_epochs: int,
            total_epochs: int,
            min_lr: float = 1e-6,
            base_lr: Optional[float] = None
    ):
        """
        Initialize the scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total number of training epochs
            min_lr: Minimum learning rate after cosine decay
            base_lr: Base learning rate (uses optimizer's lr if None)
        """
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

    Handles both autoencoder pretraining and DeepONet training stages with
    flexible time point sampling support.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            config: Dict[str, Any],
            save_dir: Path,
            device: torch.device,
            is_latent_stage: bool = False,
            epochs: int = None
    ):
        """
        Initialize the trainer.
        """
        self.logger = logging.getLogger(__name__)

        # Core state
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.is_latent = bool(is_latent_stage)

        # Training config
        train_cfg = self.config.get("training", {})
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        betas_cfg = train_cfg.get("betas", (0.9, 0.999))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
        self.use_amp = bool(train_cfg.get("use_amp", True))
        self.pou_weight = float(train_cfg.get("pou_weight", 0.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))

        # Set epochs for DeepONet training
        self.epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 200))

        # Choose parameter set by stage:
        # - AE pretrain: only autoencoder parameters
        # - Latent/DeepONet: only DeepONet parameters
        if self.is_latent:
            params = self.model.deeponet_parameters()
        else:
            params = self.model.ae_parameters()

        # Optimizer (AdamW)
        self.optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=betas_cfg
        )

        # LR scheduler placeholder; created in each training stage method
        self.scheduler = None

        # AMP scaler
        self.scaler = GradScaler('cuda' if self.use_amp else 'cpu', enabled=self.use_amp)

        # Criterion
        self.criterion = nn.MSELoss()

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.train_history = []
        self.val_history = []

        # Ensure model has tracking lists
        for name in ("index_list", "train_loss_list", "val_loss_list"):
            if not hasattr(self.model, name):
                setattr(self.model, name, [])

        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled")

    def train_ae_pretrain(self, epochs: int) -> float:
        """
        Stage 1: Pretrain autoencoder with cosine annealing scheduler.

        UPDATED: Handles both [B, S] and [B, M, S] input shapes correctly.

        Args:
            epochs: Number of epochs to train

        Returns:
            Final training loss
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
                targets = targets.to(self.device)  # [B, S] or [B, M, S]

                # Handle both [B, S] and [B, M, S] shapes
                # AE is vector→vector reconstruction
                if targets.dim() == 2:  # [B, S] - direct vector input
                    inputs_flat = targets
                elif targets.dim() == 3:  # [B, M, S] - trajectory input
                    B, M, D = targets.shape
                    inputs_flat = targets.reshape(B * M, D)
                else:
                    raise RuntimeError(f"Unexpected AE target shape: {tuple(targets.shape)}")

                self.optimizer.zero_grad(set_to_none=True)

                # Forward with AMP
                with autocast('cuda', enabled=self.use_amp):
                    recon = self.model.autoencoder(inputs_flat)
                    loss = self.criterion(recon, inputs_flat)

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

        Supports flexible time point sampling during training.

        Returns:
            Best validation loss achieved
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
        Train one epoch for DeepONet on latent space.

        CLARIFICATION: This implements loss accumulation across samples in a batch,
        not gradient accumulation across multiple batches. We compute individual
        losses for each sample (for variable time lengths), sum them, average,
        then do a single backward() and optimizer.step() per batch.

        This is mathematically equivalent to standard batched training and is NOT
        microbatching/gradient accumulation.
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_samples = 0

        for batch_data in self.train_loader:
            # Expect latent dataset: (inputs, targets_list, times_list)
            if len(batch_data) != 3:
                raise RuntimeError(
                    "This _train_epoch handles only the latent DeepONet stage. "
                    "Use train_ae_pretrain() for autoencoder pretraining."
                )

            inputs, targets_list, times_list = batch_data
            B = inputs.size(0)
            if not isinstance(targets_list, list) or not isinstance(times_list, list) or len(targets_list) != B:
                raise RuntimeError("Latent dataloader must return lists of length B for targets and times.")

            inputs = inputs.to(self.device, non_blocking=True)

            # ---- Loss accumulation within batch ----
            # We compute loss for each sample (which may have different time lengths),
            # sum them up, then average. This is standard batched training, NOT
            # gradient accumulation across multiple optimizer steps.
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss_sum = 0.0

            for i in range(B):
                input_i = inputs[i:i + 1]  # [1, L+G]
                targets_i = targets_list[i].unsqueeze(0).to(self.device, non_blocking=True)  # [1, M_i, L]
                times_i = times_list[i].to(self.device, non_blocking=True)  # [M_i] in [0,1]

                with autocast('cuda', enabled=self.use_amp):
                    loss_i, comps = self.model.compute_deeponet_loss(
                        input_i, targets_i, trunk_times=times_i, pou_weight=self.pou_weight
                    )

                # Accumulate scalar metrics
                total_mse += float(comps.get("mse", 0.0))
                total_pou += float(comps.get("pou", 0.0))
                n_samples += 1

                # Sum up losses for averaging
                batch_loss_sum = batch_loss_sum + loss_i

            # Average loss across batch (standard practice)
            with autocast('cuda', enabled=self.use_amp):
                loss = batch_loss_sum / float(B)

            # Single backward pass per batch (this is NOT gradient accumulation)
            self.scaler.scale(loss).backward()

            # Gradient clipping if configured
            if self.gradient_clip and self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Single optimizer step per batch
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach())

        # Averages over individual samples
        avg_loss = total_loss / max(1, len(self.train_loader))
        avg_mse = total_mse / max(1, n_samples)
        avg_pou = total_pou / max(1, n_samples)

        return float(avg_loss), {"mse": float(avg_mse), "pou": float(avg_pou)}

    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate model performance on validation set.

        Computes metrics in both normalized and physical space when possible.

        Returns:
            Average loss and metrics dictionary
        """
        if self.val_loader is None:
            return float("inf"), {}

        # Lazy-load normalization helper once (if available)
        if not hasattr(self, "_norm_helper"):
            try:
                from utils import load_json
                from normalizer import NormalizationHelper
                stats_path = Path(self.config["paths"]["processed_data_dir"]) / "normalization.json"
                stats = load_json(stats_path)
                self._norm_helper = NormalizationHelper(stats, self.device, self.config)
                self._species_vars = list(self.config["data"]["target_species_variables"])
            except Exception:
                self._norm_helper = None
                self._species_vars = None

        have_phys = self._norm_helper is not None and self._species_vars is not None

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae_norm = 0.0
        total_mae_phys = 0.0
        n_samples = 0

        for batch_data in self.val_loader:
            if len(batch_data) != 3:
                raise RuntimeError("Validation expects latent dataset batches: (inputs, targets_list, times_list).")

            inputs, targets_list, times_list = batch_data
            B = inputs.size(0)

            for i in range(B):
                input_i = inputs[i:i + 1]
                targets_i = targets_list[i].unsqueeze(0)
                times_i = times_list[i]

                with autocast('cuda', enabled=self.use_amp):
                    # Latent-space MSE
                    z_pred, _ = self.model(input_i, decode=False, trunk_times=times_i)
                    loss = self.criterion(z_pred, targets_i)

                    # Decode both pred and target to species (normalized) for MAE
                    y_pred, _ = self.model(input_i, decode=True, trunk_times=times_i)
                    B1, M1, LD = targets_i.shape
                    y_true = self.model.decode(targets_i.reshape(B1 * M1, LD)).view(B1, M1, -1)
                    mae_n = (y_pred - y_true).abs().mean()

                # Compute physical space MAE if possible
                if have_phys:
                    y_pred_phys = self._norm_helper.denormalize(
                        y_pred.reshape(-1, y_pred.shape[-1]), self._species_vars
                    ).view_as(y_pred)
                    y_true_phys = self._norm_helper.denormalize(
                        y_true.reshape(-1, y_true.shape[-1]), self._species_vars
                    ).view_as(y_true)
                    mae_p = (y_pred_phys - y_true_phys).abs().mean().item()
                else:
                    mae_p = 0.0

                total_loss += float(loss.item())
                total_mse += float(loss.item())
                total_mae_norm += float(mae_n.item())
                total_mae_phys += float(mae_p)
                n_samples += 1

        # Compute averages
        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)
        avg_mae_n = total_mae_norm / max(1, n_samples)
        avg_mae_p = total_mae_phys / max(1, n_samples)

        metrics = {
            "loss": avg_loss,
            "mse": avg_mse,
            "decoded_mae": avg_mae_n,
        }
        if have_phys:
            metrics["decoded_mae_phys"] = avg_mae_p

        return avg_loss, metrics

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