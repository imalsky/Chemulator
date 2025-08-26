#!/usr/bin/env python3
"""
Training module for AE-DeepONet with learning rate scheduling and PoU regularization.
CORRECTED: Fixed masked MSE normalization, proper trunk feature extraction, consistent LR handling
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
    """Cosine annealing learning rate scheduler with linear warmup."""

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

        if warmup_epochs > 0:
            self.warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            )
        else:
            self.warmup_scheduler = None

        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step()
        else:
            cosine_epoch = self.current_epoch - self.warmup_epochs
            self.cosine_scheduler.step(cosine_epoch)

        self.current_epoch += 1

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
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
        self.current_epoch = state_dict['current_epoch']
        if self.warmup_scheduler and 'warmup_scheduler' in state_dict:
            self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


class Trainer:
    """
    Trainer for AE-DeepONet models with corrected batch processing.
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
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.is_latent = bool(is_latent_stage)

        train_cfg = self.config.get("training", {})
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        betas_cfg = train_cfg.get("betas", (0.9, 0.999))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
        self.use_amp = bool(train_cfg.get("use_amp", True))
        self.pou_weight = float(train_cfg.get("pou_weight", 0.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 200))

        # Choose parameter set
        if self.is_latent:
            params = self.model.deeponet_parameters()
        else:
            params = self.model.ae_parameters()

        self.optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=betas_cfg
        )

        self.scheduler = None
        amp_dtype = torch.float16

        if any(p.dtype == torch.bfloat16 for p in self.model.parameters()):
            amp_dtype = torch.bfloat16
        self.autocast_kwargs = dict(device_type='cuda', enabled=self.use_amp, dtype=amp_dtype)
        self.scaler = GradScaler(enabled=self.use_amp and amp_dtype == torch.float16)

        self.criterion = nn.MSELoss()

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.train_history = []
        self.val_history = []

        for name in ("index_list", "train_loss_list", "val_loss_list"):
            if not hasattr(self.model, name):
                setattr(self.model, name, [])

        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled")

    def _ensure_time_2d(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure time tensor is 2D [M, 1]."""
        return t if t.dim() == 2 else t.unsqueeze(-1)

    def train_ae_pretrain(self, epochs: int) -> float:
        """Stage 1: Pretrain autoencoder with corrected scheduler handling."""
        self.logger.info("Starting autoencoder pretraining...")

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
            self.current_epoch = epoch
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for inputs, targets in self.train_loader:
                targets = targets.to(self.device)

                # Handle both [B, S] and [B, M, S] shapes
                if targets.dim() == 2:
                    inputs_flat = targets
                elif targets.dim() == 3:
                    B, M, D = targets.shape
                    inputs_flat = targets.reshape(B * M, D)
                else:
                    raise RuntimeError(f"Unexpected AE target shape: {tuple(targets.shape)}")

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(**self.autocast_kwargs):
                    recon = self.model.autoencoder(inputs_flat)
                    loss = self.criterion(recon, inputs_flat)

                self.scaler.scale(loss).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    # Clip only optimizer params
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                n_batches += 1

            avg_loss = train_loss / n_batches
            epoch_time = time.time() - epoch_start_time

            # Step scheduler after computing metrics
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            self.logger.info(
                f"[AE Pretrain] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.3e} | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(avg_loss)

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
        """Stage 3: Train DeepONet with corrected batch processing."""
        self.logger.info("Starting DeepONet training on latent space...")

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

            train_loss, train_metrics = self._train_epoch_corrected()
            val_loss, val_metrics = self._validate_corrected()

            epoch_time = time.time() - epoch_start_time

            # Step scheduler after metrics
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {train_loss:.3e} (MSE: {train_metrics['mse']:.3e}, "
                f"PoU: {train_metrics.get('pou', 0):.3e}) | "
                f"Val: {val_loss:.3e} (MAE: {val_metrics.get('decoded_mae', 0):.3e}) | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(train_loss)
            self.model.val_loss_list.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, is_best=True)

        self.logger.info(
            f"Training complete. Best val loss: {self.best_val_loss:.3e} at epoch {self.best_epoch}"
        )

        self._save_history()
        return self.best_val_loss

    def _train_epoch_corrected(self) -> Tuple[float, Dict[str, float]]:
        """Training epoch with corrected masked loss and vectorized branch computation."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_samples = 0

        for batch_data in self.train_loader:
            if len(batch_data) != 3:
                raise RuntimeError("This _train_epoch handles only the latent DeepONet stage.")

            inputs, targets_data, times_data = batch_data
            B = inputs.size(0)
            inputs = inputs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Fast path: uniform time grid
            if torch.is_tensor(targets_data):
                targets = targets_data.to(self.device, non_blocking=True)
                times = self._ensure_time_2d(times_data.to(self.device, non_blocking=True))

                with autocast(**self.autocast_kwargs):
                    loss, comps = self.model.compute_deeponet_loss(
                        inputs, targets, trunk_times=times, pou_weight=self.pou_weight
                    )

                batch_loss = loss
                batch_mse = comps.get("mse", 0.0)
                batch_pou = comps.get("pou", 0.0)

            else:
                # Variable-time path with corrected masked loss
                targets_list = targets_data
                times_list = times_data

                # Find max length and create padded tensors
                max_len = max(t.size(0) for t in targets_list)
                L = targets_list[0].size(-1)  # latent dimension

                padded_targets = torch.zeros(B, max_len, L, device=self.device, dtype=inputs.dtype)
                mask = torch.zeros(B, max_len, device=self.device, dtype=torch.bool)

                # Fill padded tensors
                for i in range(B):
                    M_i = targets_list[i].size(0)
                    padded_targets[i, :M_i] = targets_list[i].to(self.device, non_blocking=True)
                    mask[i, :M_i] = True

                # Collect unique times
                all_times = torch.cat([t.to(self.device) for t in times_list])
                unique_times, inverse = torch.unique(all_times, sorted=True, return_inverse=True)

                with autocast(**self.autocast_kwargs):
                    # Compute branch for entire batch once
                    branch_all = self.model.deeponet.forward_branch(inputs)
                    branch_all = branch_all.view(B, self.model.latent_dim, self.model.p)

                    # Compute trunk for unique times once
                    unique_times_2d = self._ensure_time_2d(unique_times)
                    trunk_all = self.model.deeponet.forward_trunk(unique_times_2d)
                    trunk_all = trunk_all.view(-1, self.model.latent_dim, self.model.p)

                    # Apply basis transformation if needed
                    if self.model.deeponet.trunk_basis == "softmax":
                        trunk_all = torch.nn.functional.softmax(trunk_all, dim=-1)

                    # Compute predictions for each sample
                    z_pred_list = []
                    start_idx = 0
                    for i in range(B):
                        M_i = times_list[i].size(0)
                        sample_inverse = inverse[start_idx:start_idx + M_i]

                        # Get relevant trunk outputs
                        trunk_i = trunk_all[sample_inverse]  # [M_i, latent_dim, p]
                        branch_i = branch_all[i]  # [latent_dim, p]

                        # Compute prediction
                        z_i = torch.einsum('lp,mlp->ml', branch_i, trunk_i)  # [M_i, latent_dim]

                        # Pad if necessary
                        if M_i < max_len:
                            padding = torch.zeros(max_len - M_i, L, device=self.device, dtype=z_i.dtype)
                            z_i = torch.cat([z_i, padding], dim=0)

                        z_pred_list.append(z_i)
                        start_idx += M_i

                    z_pred = torch.stack(z_pred_list)  # [B, max_len, latent_dim]

                    # CORRECTED: Proper masked loss normalization
                    valid = mask.unsqueeze(-1)  # [B, max_len, 1]
                    squared_error = (z_pred - padded_targets).pow(2) * valid  # [B, max_len, L]
                    total_valid_elements = (valid.sum() * z_pred.size(-1)).clamp_min(1)  # Total valid elements
                    mse_loss = squared_error.sum() / total_valid_elements

                    # PoU regularization on trunk outputs
                    pou_loss = self.model.pou_regularization(trunk_all) if self.model.use_pou else torch.zeros_like(
                        mse_loss)

                    batch_loss = mse_loss + self.pou_weight * pou_loss
                    batch_mse = mse_loss
                    batch_pou = pou_loss

            self.scaler.scale(batch_loss).backward()

            if self.gradient_clip and self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                # Clip only optimizer params
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    self.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(batch_loss.item())
            total_mse += float(batch_mse.item())
            total_pou += float(batch_pou.item())
            n_samples += 1

        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)
        avg_pou = total_pou / max(1, n_samples)

        return float(avg_loss), {"mse": float(avg_mse), "pou": float(avg_pou)}

    @torch.no_grad()
    def _validate_corrected(self) -> Tuple[float, Dict[str, float]]:
        """Validation with corrected time handling."""
        if self.val_loader is None:
            return float("inf"), {}

        # Lazy-load normalization helper
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
                raise RuntimeError("Validation expects latent dataset batches.")

            inputs, targets_data, times_data = batch_data
            B = inputs.size(0)

            # Fast path: uniform time grid
            if torch.is_tensor(targets_data):
                targets = targets_data
                times = self._ensure_time_2d(times_data)

                with autocast(**self.autocast_kwargs):
                    z_pred, _ = self.model(inputs, decode=False, trunk_times=times)
                    loss = self.criterion(z_pred, targets)

                    # Decode for MAE
                    B1, M1, LD = targets.shape
                    y_pred = self.model.decode(z_pred.reshape(B1 * M1, LD)).view(B1, M1, -1)
                    y_true = self.model.decode(targets.reshape(B1 * M1, LD)).view(B1, M1, -1)
                    mae_n = (y_pred - y_true).abs().mean()

                # Physical space MAE if available
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

                total_loss += float(loss.item()) * B
                total_mse += float(loss.item()) * B
                total_mae_norm += float(mae_n.item()) * B
                total_mae_phys += float(mae_p) * B
                n_samples += B

            else:
                # Variable-time path
                targets_list = targets_data
                times_list = times_data

                for i in range(B):
                    input_i = inputs[i:i + 1]
                    targets_i = targets_list[i].unsqueeze(0)
                    times_i = self._ensure_time_2d(times_list[i])

                    with autocast(**self.autocast_kwargs):
                        z_pred, _ = self.model(input_i, decode=False, trunk_times=times_i)
                        loss = self.criterion(z_pred, targets_i)

                        B1, M1, LD = targets_i.shape
                        y_pred = self.model.decode(z_pred.reshape(B1 * M1, LD)).view(B1, M1, -1)
                        y_true = self.model.decode(targets_i.reshape(B1 * M1, LD)).view(B1, M1, -1)
                        mae_n = (y_pred - y_true).abs().mean()

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
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.config,
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
        """Save training history."""
        import json

        history = {
            "train": self.train_history,
            "val": self.val_history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "index_list": self.model.index_list,
            "train_loss_list": self.model.train_loss_list,
            "val_loss_list": self.model.val_loss_list,
            "config": self.config
        }

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)