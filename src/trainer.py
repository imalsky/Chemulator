#!/usr/bin/env python3
"""
Simplified training module for AE-DeepONet.
Removed time warping complexity and MAE calculations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


class CosineAnnealingWarmupScheduler:
    """Cosine annealing with linear warmup."""

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
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step()
        else:
            cosine_epoch = self.current_epoch - self.warmup_epochs
            self.cosine_scheduler.step(cosine_epoch)

        self.current_epoch += 1

    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
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

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        if self.warmup_scheduler and 'warmup_scheduler' in state_dict:
            self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


class Trainer:
    """Trainer for AE-DeepONet models."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            config: Dict[str, Any],
            save_dir: Path,
            device: torch.device,
            is_latent_stage: bool = False,
            is_species_stage: bool = False,
            epochs: int = None
    ):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        self.is_latent = is_latent_stage
        self.is_species = is_species_stage

        # Training configuration
        train_cfg = self.config.get("training", {})
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.betas = train_cfg.get("betas", (0.9, 0.999))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
        self.pou_weight = float(train_cfg.get("pou_weight", 0.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 200))

        # Setup AMP
        self.use_amp = bool(train_cfg.get("use_amp", True)) and (self.device.type == "cuda")

        amp_dtype = torch.float16
        if self.device.type == "cuda" and any(p.dtype == torch.bfloat16 for p in self.model.parameters()):
            amp_dtype = torch.bfloat16

        if self.device.type == "cuda":
            self.autocast_kwargs = dict(
                device_type='cuda',
                enabled=self.use_amp,
                dtype=amp_dtype
            )
        else:
            self.autocast_kwargs = dict(
                device_type=self.device.type,
                enabled=False
            )

        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda" and amp_dtype == torch.float16)

        # Get parameters
        params = self._get_parameters()

        # Initialize optimizer
        self.optimizer = AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas
        )

        self.scheduler = None

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.train_history = []
        self.val_history = []

        self._initialize_model_tracking()

        if self.use_amp:
            self.logger.info(f"AMP enabled on {self.device.type}")

    def _get_parameters(self):
        """Get appropriate parameters for training stage."""
        if self.is_latent or self.is_species:
            freeze_ae = self.config.get("training", {}).get("freeze_ae_after_pretrain", True)
            bypass_ae = self.config.get("model", {}).get("bypass_autoencoder", False)

            if bypass_ae or freeze_ae:
                return self.model.deeponet_parameters()
            else:
                from itertools import chain
                return chain(self.model.deeponet_parameters(), self.model.ae_parameters())
        else:
            return self.model.ae_parameters()

    def _initialize_model_tracking(self):
        """Initialize tracking lists on model."""
        for name in ("index_list", "train_loss_list", "val_loss_list"):
            if not hasattr(self.model, name):
                setattr(self.model, name, [])

    def _ensure_time_2d(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure time tensor is 2D."""
        return t if t.dim() == 2 else t.unsqueeze(-1)

    def train_ae_pretrain(self, epochs: int) -> float:
        """Pretrain autoencoder."""
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
            epoch_start = time.time()
            self.current_epoch = epoch

            avg_loss = self._train_ae_epoch()

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"[AE] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.3e} | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs:
                self._save_checkpoint(epoch, avg_loss, prefix="ae_checkpoint")

        self._save_ae_final(epochs, avg_loss)
        return avg_loss

    def _train_ae_epoch(self) -> float:
        """Train autoencoder for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in self.train_loader:
            targets = targets.to(self.device)

            if targets.dim() == 2:
                inputs_flat = targets
            elif targets.dim() == 3:
                B, M, D = targets.shape
                inputs_flat = targets.reshape(B * M, D)
            else:
                raise RuntimeError(f"Unexpected target shape: {targets.shape}")

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(**self.autocast_kwargs):
                recon = self.model.autoencoder(inputs_flat)
                loss = self.criterion(recon, inputs_flat)

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _save_ae_final(self, epochs: int, final_loss: float):
        """Save final autoencoder."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epochs,
            "loss": final_loss,
            "config": self.config
        }
        torch.save(checkpoint, self.save_dir / "ae_pretrained.pt")

    def train_deeponet(self) -> float:
        """Train DeepONet."""
        stage_name = "species space" if self.is_species else "latent space"
        self.logger.info(f"Starting DeepONet training on {stage_name}...")

        self.scheduler = CosineAnnealingWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.epochs,
            min_lr=self.min_lr,
            base_lr=self.lr
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            self.current_epoch = epoch

            train_loss, train_metrics = self._train_deeponet_epoch()
            val_loss, val_metrics = self._validate_epoch()

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {train_loss:.3e} (MSE: {train_metrics['mse']:.3e}, "
                f"PoU: {train_metrics.get('pou', 0):.3e}) | "
                f"Val: {val_loss:.3e} | "
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

    def _train_deeponet_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train DeepONet for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pou = 0.0
        n_samples = 0

        for batch_data in self.train_loader:
            if len(batch_data) != 3:
                raise RuntimeError("Expected (inputs, targets, times) batches")

            inputs, targets_data, times_data = batch_data
            inputs = inputs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Process batch based on time grid type
            if torch.is_tensor(targets_data):
                loss, mse, pou = self._process_uniform_batch(inputs, targets_data, times_data)
            else:
                loss, mse, pou = self._process_variable_batch(inputs, targets_data, times_data)

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_pou += float(pou.item())
            n_samples += 1

        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)
        avg_pou = total_pou / max(1, n_samples)

        return avg_loss, {"mse": avg_mse, "pou": avg_pou}

    def _process_uniform_batch(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch with uniform time grid."""
        targets = targets.to(self.device, non_blocking=True)
        times = self._ensure_time_2d(times.to(self.device, non_blocking=True))

        with autocast(**self.autocast_kwargs):
            z_pred, aux = self.model(inputs, decode=False, trunk_times=times)
            trunk_outputs = aux.get("trunk_outputs") if self.model.use_pou else None

            loss, components = self.model.compute_deeponet_loss(
                z_pred=z_pred,
                z_true=targets,
                mask=None,
                trunk_outputs=trunk_outputs,
                pou_weight=self.pou_weight,
                pou_mask=None
            )

        mse = torch.as_tensor(components.get("mse", 0.0), device=self.device, dtype=inputs.dtype)
        pou = torch.as_tensor(components.get("pou", 0.0), device=self.device, dtype=inputs.dtype)

        return loss, mse, pou

    def _process_variable_batch(
            self,
            inputs: torch.Tensor,
            targets_list: List[torch.Tensor],
            times_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch with variable time grids."""
        B = inputs.size(0)
        max_len = max(t.size(0) for t in targets_list)
        working_dim = self.model.working_dim

        # Create padded tensors
        padded_targets = torch.zeros(B, max_len, working_dim, device=self.device, dtype=inputs.dtype)
        mask = torch.zeros(B, max_len, device=self.device, dtype=torch.bool)

        for i in range(B):
            M_i = targets_list[i].size(0)
            padded_targets[i, :M_i] = targets_list[i].to(self.device, non_blocking=True)
            mask[i, :M_i] = True

        # Compute predictions (simplified - no time warping)
        z_pred, trunk_outputs = self._compute_predictions(inputs, times_list, max_len)

        with autocast(**self.autocast_kwargs):
            loss, components = self.model.compute_deeponet_loss(
                z_pred=z_pred,
                z_true=padded_targets,
                mask=mask,
                trunk_outputs=trunk_outputs if self.model.use_pou else None,
                pou_weight=self.pou_weight,
                pou_mask=mask if self.model.use_pou and trunk_outputs is not None else None
            )

        mse = torch.as_tensor(components.get("mse", 0.0), device=self.device, dtype=inputs.dtype)
        pou = torch.as_tensor(components.get("pou", 0.0), device=self.device, dtype=inputs.dtype)

        return loss, mse, pou

    def _compute_predictions(
            self,
            inputs: torch.Tensor,
            times_list: List[torch.Tensor],
            max_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute predictions for variable time grids."""
        B = inputs.size(0)
        working_dim = self.model.working_dim

        # Collect unique times
        all_times = torch.cat([t.to(self.device).reshape(-1) for t in times_list])
        unique_times, inverse = torch.unique(all_times, sorted=True, return_inverse=True)

        with autocast(**self.autocast_kwargs):
            # Compute branch for entire batch
            branch_all = self.model.deeponet.forward_branch(inputs)
            branch_all = branch_all.view(B, working_dim, self.model.p)

            # Apply mixer if present
            if self.model.deeponet.mixer is not None:
                g = inputs[:, working_dim:]
                branch_all = self.model.deeponet.mixer(g, branch_all)

            # Compute trunk for unique times
            unique_times_2d = self._ensure_time_2d(unique_times).to(
                device=self.device,
                dtype=inputs.dtype
            )
            trunk_all = self.model.deeponet.forward_trunk(unique_times_2d)
            trunk_all = trunk_all.view(-1, working_dim, self.model.p)

            # Apply basis transformation
            if self.model.deeponet.trunk_type == "mlp" and self.model.deeponet.trunk_basis == "softmax":
                trunk_all = torch.nn.functional.softmax(trunk_all, dim=-1)

            # Compute predictions for each sample
            z_pred_list = []
            trunk_list = []
            start_idx = 0

            for i in range(B):
                M_i = times_list[i].size(0)
                sample_inverse = inverse[start_idx:start_idx + M_i]

                trunk_i = trunk_all[sample_inverse]
                branch_i = branch_all[i]

                z_i = torch.einsum('lp,mlp->ml', branch_i, trunk_i)

                # Pad predictions
                if M_i < max_len:
                    padding = torch.zeros(max_len - M_i, working_dim, device=self.device, dtype=z_i.dtype)
                    z_i = torch.cat([z_i, padding], dim=0)

                z_pred_list.append(z_i)

                # Pad trunk outputs for PoU
                if self.model.use_pou:
                    if M_i < max_len:
                        trunk_padding = torch.zeros(
                            max_len - M_i, trunk_i.size(1), trunk_i.size(2),
                            device=trunk_i.device, dtype=trunk_i.dtype
                        )
                        trunk_i = torch.cat([trunk_i, trunk_padding], dim=0)
                    trunk_list.append(trunk_i)

                start_idx += M_i

            z_pred = torch.stack(z_pred_list)
            trunk_outputs = torch.stack(trunk_list) if trunk_list else None

        return z_pred, trunk_outputs

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        if self.val_loader is None:
            return float("inf"), {}

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_samples = 0

        for batch_data in self.val_loader:
            if len(batch_data) != 3:
                raise RuntimeError("Expected DeepONet dataset batches")

            inputs, targets_data, times_data = batch_data
            B = inputs.size(0)

            if torch.is_tensor(targets_data):
                metrics = self._validate_uniform_batch(inputs, targets_data, times_data)
            else:
                metrics = self._validate_variable_batch(inputs, targets_data, times_data)

            total_loss += metrics['loss'] * metrics['batch_size']
            total_mse += metrics['mse'] * metrics['batch_size']
            n_samples += metrics['batch_size']

        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)

        return avg_loss, {"loss": avg_loss, "mse": avg_mse}

    def _validate_uniform_batch(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            times: torch.Tensor
    ) -> Dict[str, float]:
        """Validate uniform batch."""
        B = inputs.size(0)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        times = self._ensure_time_2d(times.to(self.device, non_blocking=True))

        with autocast(**self.autocast_kwargs):
            z_pred, _ = self.model(inputs, decode=False, trunk_times=times)
            loss = self.criterion(z_pred, targets.to(z_pred.dtype))

        return {
            'loss': float(loss.item()),
            'mse': float(loss.item()),
            'batch_size': B
        }

    def _validate_variable_batch(
            self,
            inputs: torch.Tensor,
            targets_list: List[torch.Tensor],
            times_list: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Validate variable batch."""
        B = inputs.size(0)
        inputs = inputs.to(self.device, non_blocking=True)

        total_loss = 0.0
        total_mse = 0.0

        for i in range(B):
            input_i = inputs[i:i + 1]
            targets_i = targets_list[i].unsqueeze(0).to(self.device, non_blocking=True)
            times_i = self._ensure_time_2d(times_list[i].to(self.device, non_blocking=True))

            with autocast(**self.autocast_kwargs):
                z_pred, _ = self.model(input_i, decode=False, trunk_times=times_i)
                loss = self.criterion(z_pred, targets_i.to(z_pred.dtype))

            total_loss += float(loss.item())
            total_mse += float(loss.item())

        return {
            'loss': total_loss / B,
            'mse': total_mse / B,
            'batch_size': B
        }

    def _save_checkpoint(
            self,
            epoch: int,
            loss: float,
            prefix: str = "checkpoint",
            is_best: bool = False
    ):
        """Save checkpoint."""
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