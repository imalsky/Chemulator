#!/usr/bin/env python3
"""
Training module for AE-DeepONet with AMP support and regularization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


class Trainer:
    """Trainer for AE-DeepONet models with AMP and regularization."""

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

        # Training config
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.pou_weight = train_cfg.get("pou_weight", 0.01)

        # Setup optimizer (AdamW for weight decay)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=train_cfg.get("betas", (0.9, 0.999))
        )

        # Setup AMP
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss function (MSE as per paper)
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        # Metrics tracking
        self.train_history = []
        self.val_history = []

        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled")

    def train_ae_pretrain(self, epochs: int) -> float:
        """Stage 1: Pretrain autoencoder."""
        self.logger.info("Starting autoencoder pretraining...")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for inputs, targets in self.train_loader:
                # Move to device
                targets = targets.to(self.device)  # [B, M, n_species]

                # Flatten for autoencoder
                B, M, D = targets.shape
                targets_flat = targets.reshape(B * M, D)

                self.optimizer.zero_grad(set_to_none=True)

                # Forward with AMP
                with autocast(enabled=self.use_amp):
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
            self.logger.info(f"[AE Pretrain] Epoch {epoch}/{epochs} - Loss: {avg_loss:.6e}")

            # Update model tracking lists (matching paper)
            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(avg_loss)

            # Save periodically
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, avg_loss, prefix="ae_pretrain")

        # Save final pretrained autoencoder
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epochs,
            "loss": avg_loss
        }, self.save_dir / "ae_pretrained.pt")

        return avg_loss

    def train_deeponet(self) -> float:
        """Stage 3: Train DeepONet on latent space."""
        self.logger.info("Starting DeepONet training on latent space...")

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Training
            train_loss, train_metrics = self._train_epoch()

            # Validation
            val_loss, val_metrics = self._validate()

            # Log progress
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_loss:.6e} (MSE: {train_metrics['mse']:.6e}, "
                f"PoU: {train_metrics.get('pou', 0):.6e}) | "
                f"Val Loss: {val_loss:.6e} (MSE: {val_metrics['mse']:.6e}, "
                f"Decoded MAE: {val_metrics.get('decoded_mae', 0):.6e})"
            )

            # Track history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            # Update model tracking lists (matching paper)
            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(train_loss)
            self.model.val_loss_list.append(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss)

        self.logger.info(f"Training complete. Best val loss: {self.best_val_loss:.6e} at epoch {self.best_epoch}")

        # Save training history
        self._save_history()

        return self.best_val_loss

    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train one epoch with metrics tracking."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_batches = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                if self.is_latent_stage:
                    # DeepONet training: predict latent trajectory
                    z_pred, aux = self.model(inputs, decode=False, return_trunk_outputs=True)
                    mse_loss = self.criterion(z_pred, targets)

                    # PoU regularization
                    pou_loss = self.model.pou_regularization(aux.get("trunk_outputs"))

                    loss = mse_loss + self.pou_weight * pou_loss
                else:
                    # Regular training
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
        """Validate with both latent and decoded metrics."""
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

            with autocast(enabled=self.use_amp):
                if self.is_latent_stage:
                    # Compute latent MSE
                    z_pred, _ = self.model(inputs, decode=False)
                    loss = self.criterion(z_pred, targets)

                    # Also compute decoded MAE for interpretability
                    y_pred, _ = self.model(inputs, decode=True)

                    # Decode targets for comparison
                    B, M, LD = targets.shape
                    targets_flat = targets.reshape(B * M, LD)
                    y_true_flat = self.model.decode(targets_flat)
                    y_true = y_true_flat.view(B, M, -1)

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
            "decoded_mae": total_decoded_mae / n_batches
        }

        return metrics["loss"], metrics

    def _save_checkpoint(self, epoch: int, loss: float, prefix: str = "best"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.config,
            # Save model tracking lists
            "index_list": self.model.index_list,
            "train_loss_list": self.model.train_loss_list,
            "val_loss_list": self.model.val_loss_list
        }
        torch.save(checkpoint, self.save_dir / f"{prefix}_model.pt")
        self.logger.info(f"Saved {prefix} checkpoint at epoch {epoch}")

    def _save_history(self):
        """Save training history."""
        import json
        history = {
            "train": self.train_history,
            "val": self.val_history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            # Include model tracking lists
            "index_list": self.model.index_list,
            "train_loss_list": self.model.train_loss_list,
            "val_loss_list": self.model.val_loss_list
        }
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)