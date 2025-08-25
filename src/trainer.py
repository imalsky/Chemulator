#!/usr/bin/env python3
"""
Training module for AE-DeepONet with learning rate scheduling and PoU regularization.
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
        if epoch is None:
            self.current_epoch += 1
            epoch = self.current_epoch
        else:
            self.current_epoch = epoch

        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step(epoch)
        else:
            cosine_epoch = max(0, epoch - self.warmup_epochs)
            self.cosine_scheduler.step(cosine_epoch)

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
    """Trainer for AE-DeepONet models with proper PoU regularization integration."""

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

        params = self.model.deeponet_parameters() if self.is_latent else self.model.ae_parameters()

        self.optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=betas_cfg
        )

        self.scheduler = None
        self.scaler = GradScaler('cuda' if self.use_amp else 'cpu', enabled=self.use_amp)
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

    def train_ae_pretrain(self, epochs: int) -> float:
        """Stage 1: Pretrain autoencoder with cosine annealing scheduler."""
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
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            current_lr = self.scheduler.get_last_lr()

            for inputs, targets in self.train_loader:
                targets = targets.to(self.device)

                if targets.dim() == 2:
                    inputs_flat = targets
                elif targets.dim() == 3:
                    B, M, D = targets.shape
                    inputs_flat = targets.reshape(B * M, D)
                else:
                    raise RuntimeError(f"Unexpected AE target shape: {tuple(targets.shape)}")

                self.optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=self.use_amp):
                    recon = self.model.autoencoder(inputs_flat)
                    loss = self.criterion(recon, inputs_flat)

                self.scaler.scale(loss).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                n_batches += 1

            avg_loss = train_loss / n_batches
            epoch_time = time.time() - epoch_start_time

            self.scheduler.step(epoch)

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
        """Stage 3: Train DeepONet on latent space with PoU regularization."""
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
            current_lr = self.scheduler.get_last_lr()

            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._validate()

            epoch_time = time.time() - epoch_start_time

            self.scheduler.step(epoch)

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

    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train one epoch for DeepONet on latent space with vectorized computation."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_samples = 0

        for batch_data in self.train_loader:
            if len(batch_data) != 3:
                raise RuntimeError("This _train_epoch handles only the latent DeepONet stage.")

            inputs, targets_list, times_list = batch_data
            B = inputs.size(0)

            inputs = inputs.to(self.device, non_blocking=True)

            # robust equality for float times
            t0 = times_list[0].to(dtype=torch.float32, device='cpu')
            times_are_identical = all(
                torch.allclose(t0, times_list[i].to(dtype=torch.float32, device='cpu'), rtol=0.0, atol=1e-8)
                for i in range(1, B)
            )

            self.optimizer.zero_grad(set_to_none=True)

            if times_are_identical:
                shared_times = times_list[0].to(self.device, non_blocking=True)
                targets = torch.stack([
                    t.squeeze(0) if t.dim() > 2 else t
                    for t in targets_list
                ]).to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp):
                    loss, comps = self.model.compute_deeponet_loss_vectorized(
                        inputs, targets, shared_times, pou_weight=self.pou_weight
                    )

                total_mse += float(comps.get("mse", 0.0)) * B
                total_pou += float(comps.get("pou", 0.0)) * B
                n_samples += B

            else:
                # Build union of times and align targets on union grid
                times_cpu = [t.flatten().to(dtype=torch.float32, device='cpu') for t in times_list]
                all_times = torch.cat(times_cpu)
                unique_times, inverse = torch.unique(all_times, sorted=True, return_inverse=True)
                U = unique_times.numel()

                lengths = [t.numel() for t in times_cpu]
                if len(lengths) > 1:
                    cum = torch.tensor(lengths[:-1]).cumsum(0)
                    offsets = torch.cat([torch.zeros(1, dtype=torch.long), cum])
                else:
                    offsets = torch.zeros(1, dtype=torch.long)
                idx_per_sample = [inverse[offsets[i]:offsets[i] + lengths[i]] for i in range(B)]

                unique_times = unique_times.to(self.device, non_blocking=True)

                # Prepare padded targets/mask on union grid [B, U, L]
                t0_ex = targets_list[0].squeeze(0) if targets_list[0].dim() > 2 else targets_list[0]
                Ldim = t0_ex.shape[-1]
                padded_targets = torch.zeros(B, U, Ldim, device=self.device, dtype=t0_ex.dtype)
                mask = torch.zeros(B, U, dtype=torch.bool, device=self.device)

                for i, idx in enumerate(idx_per_sample):
                    t_i = targets_list[i].squeeze(0) if targets_list[i].dim() > 2 else targets_list[i]
                    t_i = t_i.to(self.device, non_blocking=True)  # (M_i, L)
                    j = idx.to(self.device, non_blocking=True)  # (M_i,)
                    padded_targets[i, j] = t_i
                    mask[i, j] = True

                with autocast('cuda', enabled=self.use_amp):
                    # keep existing signature to avoid touching model
                    loss, comps = self.model.compute_deeponet_loss_padded(
                        inputs, padded_targets, unique_times, times_list, mask,
                        pou_weight=self.pou_weight
                    )

                total_mse += float(comps.get("mse", 0.0)) * B
                total_pou += float(comps.get("pou", 0.0)) * B
                n_samples += B

            self.scaler.scale(loss).backward()

            if self.gradient_clip and self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * B  # sample-weighted

        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)
        avg_pou = total_pou / max(1, n_samples)

        return float(avg_loss), {"mse": float(avg_mse), "pou": float(avg_pou)}

    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """FIXED: Vectorized validation for efficiency."""
        if self.val_loader is None:
            return float("inf"), {}

        # Lazy-load normalization helper once
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

            inputs, targets_list, times_list = batch_data
            B = inputs.size(0)

            inputs = inputs.to(self.device, non_blocking=True)

            # robust equality for float times
            t0 = times_list[0].to(dtype=torch.float32, device='cpu')
            times_are_identical = all(
                torch.allclose(t0, times_list[i].to(dtype=torch.float32, device='cpu'), rtol=0.0, atol=1e-8)
                for i in range(1, B)
            )

            if times_are_identical:
                shared_times = times_list[0].to(self.device, non_blocking=True)
                targets = torch.stack([
                    t.squeeze(0) if t.dim() > 2 else t
                    for t in targets_list
                ]).to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp):
                    z_pred, _ = self.model(inputs, decode=False, trunk_times=shared_times)
                    loss = self.criterion(z_pred, targets)

                    B1, M1, LD = targets.shape
                    y_pred = self.model.decode(z_pred.reshape(B1 * M1, LD)).view(B1, M1, -1)
                    y_true = self.model.decode(targets.reshape(B1 * M1, LD)).view(B1, M1, -1)
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

                total_loss += float(loss.item()) * B
                total_mse += float(loss.item()) * B
                total_mae_norm += float(mae_n.item()) * B
                total_mae_phys += float(mae_p) * B
                n_samples += B

            else:
                for i in range(B):
                    input_i = inputs[i:i + 1]
                    targets_i = targets_list[i].unsqueeze(0)
                    times_i = times_list[i]

                    with autocast('cuda', enabled=self.use_amp):
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