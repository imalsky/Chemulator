#!/usr/bin/env python3
"""
Training module for AE-DeepONet with learning rate scheduling and PoU regularization.
Supports both autoencoder pretraining and DeepONet training phases.
Handles batch-dependent trunk outputs when global-conditioned time-warping is enabled.
CORRECTED: Proper time-warp output regularization and PoU masking.
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
    """
    Trainer for AE-DeepONet models.
    Handles both autoencoder pretraining and DeepONet training phases.
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
        self.is_latent = bool(is_latent_stage)
        self.is_species = bool(is_species_stage)

        # Extract training configuration
        train_cfg = self.config.get("training", {})
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.betas = train_cfg.get("betas", (0.9, 0.999))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
        self.pou_weight = float(train_cfg.get("pou_weight", 0.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 10))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 200))

        # Setup AMP based on device capabilities
        self.use_amp = bool(train_cfg.get("use_amp", True)) and (self.device.type == "cuda")

        # Determine AMP dtype
        amp_dtype = torch.float16  # default for CUDA AMP
        if self.device.type == "cuda" and any(p.dtype == torch.bfloat16 for p in self.model.parameters()):
            amp_dtype = torch.bfloat16

        # Configure autocast based on device
        if self.device.type == "cuda":
            self.autocast_kwargs = dict(
                device_type='cuda',
                enabled=self.use_amp,
                dtype=amp_dtype
            )
        elif self.device.type == "cpu":
            # CPU autocast only supports bfloat16
            self.autocast_kwargs = dict(
                device_type='cpu',
                enabled=False,  # Disable by default, can enable for bf16
                dtype=torch.bfloat16
            )
        else:  # MPS or other
            self.autocast_kwargs = dict(
                device_type=self.device.type,
                enabled=False
            )

        # GradScaler only for CUDA FP16
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda" and amp_dtype == torch.float16)

        # Determine parameter set based on training stage
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

        # Initialize model tracking lists if needed
        self._initialize_model_tracking()

        if self.use_amp:
            self.logger.info(f"Automatic Mixed Precision (AMP) enabled on {self.device.type}")

    def _get_parameters(self):
        """Get the appropriate parameters based on training stage."""
        if self.is_latent or self.is_species:
            # DeepONet training phase
            freeze_ae = self.config.get("training", {}).get("freeze_ae_after_pretrain", True)
            bypass_ae = self.config.get("model", {}).get("bypass_autoencoder", False)

            if bypass_ae or freeze_ae:
                return self.model.deeponet_parameters()
            else:
                # Joint training of autoencoder and DeepONet
                from itertools import chain
                return chain(self.model.deeponet_parameters(), self.model.ae_parameters())
        else:
            # Autoencoder pretraining phase
            return self.model.ae_parameters()

    def _initialize_model_tracking(self):
        """Initialize tracking lists on the model if they don't exist."""
        for name in ("index_list", "train_loss_list", "val_loss_list"):
            if not hasattr(self.model, name):
                setattr(self.model, name, [])

    def _ensure_time_2d(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure time tensor is 2D [M, 1] for trunk network."""
        return t if t.dim() == 2 else t.unsqueeze(-1)

    def train_ae_pretrain(self, epochs: int) -> float:
        """
        Stage 1: Pretrain autoencoder.

        Args:
            epochs: Number of training epochs

        Returns:
            Final training loss
        """
        self.logger.info("Starting autoencoder pretraining...")

        # Setup learning rate scheduler
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

            # Training epoch
            avg_loss = self._train_ae_epoch()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            # Log progress
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"[AE Pretrain] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.3e} | "
                f"LR: {current_lr:.3e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Track progress
            self.model.index_list.append(epoch)
            self.model.train_loss_list.append(avg_loss)

            # Periodic checkpointing
            if epoch % 10 == 0 or epoch == epochs:
                self._save_checkpoint(epoch, avg_loss, prefix="ae_checkpoint")

        # Save final pretrained autoencoder
        self._save_ae_final(epochs, avg_loss)

        return avg_loss

    def _train_ae_epoch(self) -> float:
        """Train autoencoder for one epoch."""
        self.model.train()
        total_loss = 0.0
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
                # Clip all parameter groups
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _save_ae_final(self, epochs: int, final_loss: float):
        """Save final autoencoder checkpoint."""
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
        """
        Stage 3: Train DeepONet.

        Returns:
            Best validation loss achieved
        """
        stage_name = "species space" if self.is_species else "latent space"
        self.logger.info(f"Starting DeepONet training on {stage_name}...")

        # Setup learning rate scheduler
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

            # Training and validation
            train_loss, train_metrics = self._train_deeponet_epoch()
            val_loss, val_metrics = self._validate_epoch()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            # Log progress
            epoch_time = time.time() - epoch_start_time
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
                raise RuntimeError("DeepONet training expects (inputs, targets, times) batches")

            inputs, targets_data, times_data = batch_data
            B = inputs.size(0)
            inputs = inputs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Check if we have uniform or variable time grids
            if torch.is_tensor(targets_data):
                # Uniform time grid across batch
                loss, mse, pou = self._process_uniform_batch(
                    inputs, targets_data, times_data
                )
            else:
                # Variable time grids
                loss, mse, pou = self._process_variable_batch(
                    inputs, targets_data, times_data
                )

            # Backward pass
            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                # Clip all parameter groups
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_pou += float(pou.item())
            n_samples += 1

        # Average metrics
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
            # Forward pass
            z_pred, aux = self.model(inputs, decode=False, trunk_times=times)
            trunk_outputs = aux.get("trunk_outputs") if getattr(self.model, "use_pou", False) else None
            time_warp_b = aux.get("time_warp_b", None)

            # Compute loss with optional PoU regularization and time-warp regularization
            loss, components = self.model.compute_deeponet_loss(
                z_pred=z_pred,
                z_true=targets,
                mask=None,
                trunk_outputs=trunk_outputs,
                pou_weight=self.pou_weight,
                pou_mask=None,
                time_warp_b=time_warp_b
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

        # Find maximum sequence length
        max_len = max(t.size(0) for t in targets_list)
        working_dim = self.model.working_dim

        # Create padded tensors and mask
        padded_targets = torch.zeros(B, max_len, working_dim, device=self.device, dtype=inputs.dtype)
        mask = torch.zeros(B, max_len, device=self.device, dtype=torch.bool)

        for i in range(B):
            M_i = targets_list[i].size(0)
            padded_targets[i, :M_i] = targets_list[i].to(self.device, non_blocking=True)
            mask[i, :M_i] = True

        # Handle time-warping if enabled
        if getattr(self.model.deeponet, 'use_time_warp', False):
            z_pred, trunk_outputs, time_warp_b = self._compute_warped_predictions(
                inputs, times_list, max_len, mask
            )
        else:
            z_pred, trunk_outputs = self._compute_standard_predictions(
                inputs, times_list, max_len
            )
            time_warp_b = None

        with autocast(**self.autocast_kwargs):
            # Compute masked loss with proper PoU mask and time-warp regularization
            loss, components = self.model.compute_deeponet_loss(
                z_pred=z_pred,
                z_true=padded_targets,
                mask=mask,
                trunk_outputs=trunk_outputs if self.model.use_pou else None,
                pou_weight=self.pou_weight,
                pou_mask=mask if self.model.use_pou and trunk_outputs is not None else None,
                time_warp_b=time_warp_b
            )

        mse = torch.as_tensor(components.get("mse", 0.0), device=self.device, dtype=inputs.dtype)
        pou = torch.as_tensor(components.get("pou", 0.0), device=self.device, dtype=inputs.dtype)

        return loss, mse, pou

    def _compute_warped_predictions(
            self,
            inputs: torch.Tensor,
            times_list: List[torch.Tensor],
            max_len: int,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute predictions with time-warping enabled."""
        B = inputs.size(0)
        working_dim = self.model.working_dim

        # Collect time-warp bias values
        time_warp_b_list = []

        with autocast(**self.autocast_kwargs):
            # Time-warping makes trunk batch-dependent
            # Process each sample separately
            z_pred_list = []
            trunk_outputs_list = []

            for i in range(B):
                input_i = inputs[i:i + 1]
                times_i = self._ensure_time_2d(times_list[i].to(self.device))
                M_i = times_i.size(0)

                # Forward pass for single sample
                z_i, aux_i = self.model(input_i, decode=False, trunk_times=times_i)
                z_i = z_i.squeeze(0)  # Remove batch dimension [1, M, D] -> [M, D]

                # Collect time-warp bias
                if "time_warp_b" in aux_i:
                    time_warp_b_list.append(aux_i["time_warp_b"])

                # Pad predictions if necessary
                if M_i < max_len:
                    padding = torch.zeros(max_len - M_i, working_dim, device=self.device, dtype=z_i.dtype)
                    z_i = torch.cat([z_i, padding], dim=0)

                z_pred_list.append(z_i)

                # Handle trunk outputs for PoU (with padding)
                if self.model.use_pou and "trunk_outputs" in aux_i:
                    trunk_out_i = aux_i["trunk_outputs"]
                    # Remove batch dimension if present
                    if trunk_out_i.dim() == 4:  # [1, M, L, P]
                        trunk_out_i = trunk_out_i.squeeze(0)  # [M, L, P]

                    # Pad trunk outputs to match max_len
                    if M_i < max_len:
                        pad_shape = (max_len - M_i,) + trunk_out_i.shape[1:]
                        trunk_padding = torch.zeros(pad_shape, device=self.device, dtype=trunk_out_i.dtype)
                        trunk_out_i = torch.cat([trunk_out_i, trunk_padding], dim=0)

                    trunk_outputs_list.append(trunk_out_i)

            z_pred = torch.stack(z_pred_list)  # [B, max_len, working_dim]

            # Stack trunk outputs if collected
            trunk_outputs = torch.stack(trunk_outputs_list) if trunk_outputs_list else None

            # Stack time-warp bias if collected
            time_warp_b = torch.cat(time_warp_b_list) if time_warp_b_list else None

        return z_pred, trunk_outputs, time_warp_b

    def _compute_standard_predictions(
            self,
            inputs: torch.Tensor,
            times_list: List[torch.Tensor],
            max_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute predictions without time-warping (batch-independent trunk)."""
        B = inputs.size(0)
        working_dim = self.model.working_dim

        # Collect unique time points
        all_times = torch.cat([t.to(self.device) for t in times_list])
        unique_times, inverse = torch.unique(all_times, sorted=True, return_inverse=True)

        with autocast(**self.autocast_kwargs):
            # Compute branch for entire batch
            branch_all = self.model.deeponet.forward_branch(inputs)
            branch_all = branch_all.view(B, working_dim, self.model.p)

            # Compute trunk for unique times
            unique_times_2d = self._ensure_time_2d(unique_times)
            trunk_all = self.model.deeponet.forward_trunk(unique_times_2d)
            trunk_all = trunk_all.view(-1, working_dim, self.model.p)

            # Apply basis transformation
            if self.model.deeponet.trunk_basis == "softmax":
                trunk_all = torch.nn.functional.softmax(trunk_all, dim=-1)

            # Compute predictions and collect trunk outputs for each sample
            z_pred_list = []
            trunk_list = []
            start_idx = 0

            for i in range(B):
                M_i = times_list[i].size(0)
                sample_inverse = inverse[start_idx:start_idx + M_i]

                # Get relevant trunk outputs
                trunk_i = trunk_all[sample_inverse]  # [M_i, working_dim, p]
                branch_i = branch_all[i]  # [working_dim, p]

                # Compute prediction
                z_i = torch.einsum('lp,mlp->ml', branch_i, trunk_i)  # [M_i, working_dim]

                # Pad predictions if necessary
                if M_i < max_len:
                    padding = torch.zeros(max_len - M_i, working_dim, device=self.device, dtype=z_i.dtype)
                    z_i = torch.cat([z_i, padding], dim=0)

                z_pred_list.append(z_i)

                # Pad trunk outputs for PoU if enabled
                if self.model.use_pou:
                    if M_i < max_len:
                        trunk_padding = torch.zeros(
                            max_len - M_i, trunk_i.size(1), trunk_i.size(2),
                            device=trunk_i.device, dtype=trunk_i.dtype
                        )
                        trunk_i = torch.cat([trunk_i, trunk_padding], dim=0)
                    trunk_list.append(trunk_i)

                start_idx += M_i

            z_pred = torch.stack(z_pred_list)  # [B, max_len, working_dim]

            # Stack trunk outputs if PoU is enabled
            trunk_outputs = torch.stack(trunk_list) if trunk_list else None  # [B, max_len, working_dim, p]

        return z_pred, trunk_outputs

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate model performance."""
        if self.val_loader is None:
            return float("inf"), {}

        # Lazy-load normalization helper for physical space metrics
        if not hasattr(self, "_norm_helper"):
            self._load_normalization_helper()

        have_phys = self._norm_helper is not None and self._species_vars is not None

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae_norm = 0.0
        total_mae_phys = 0.0
        n_samples = 0

        for batch_data in self.val_loader:
            if len(batch_data) != 3:
                raise RuntimeError("Validation expects DeepONet dataset batches")

            inputs, targets_data, times_data = batch_data
            B = inputs.size(0)

            # Process batch based on time grid type
            if torch.is_tensor(targets_data):
                # Uniform time grid
                metrics = self._validate_uniform_batch(
                    inputs, targets_data, times_data, have_phys
                )
            else:
                # Variable time grids
                metrics = self._validate_variable_batch(
                    inputs, targets_data, times_data, have_phys
                )

            # Accumulate metrics
            total_loss += metrics['loss'] * metrics['batch_size']
            total_mse += metrics['mse'] * metrics['batch_size']
            total_mae_norm += metrics['mae_norm'] * metrics['batch_size']
            total_mae_phys += metrics.get('mae_phys', 0.0) * metrics['batch_size']
            n_samples += metrics['batch_size']

        # Compute averages
        avg_loss = total_loss / max(1, n_samples)
        avg_mse = total_mse / max(1, n_samples)
        avg_mae_norm = total_mae_norm / max(1, n_samples)
        avg_mae_phys = total_mae_phys / max(1, n_samples)

        metrics = {
            "loss": avg_loss,
            "mse": avg_mse,
            "decoded_mae": avg_mae_norm,
        }
        if have_phys:
            metrics["decoded_mae_phys"] = avg_mae_phys

        return avg_loss, metrics

    def _validate_uniform_batch(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            times: torch.Tensor,
            have_phys: bool
    ) -> Dict[str, float]:
        """Validate batch with uniform time grid."""
        B = inputs.size(0)

        # Move tensors to device
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        times = self._ensure_time_2d(times.to(self.device, non_blocking=True))

        with autocast(**self.autocast_kwargs):
            z_pred, _ = self.model(inputs, decode=False, trunk_times=times)
            # Ensure dtype consistency
            loss = self.criterion(z_pred, targets.to(z_pred.dtype))

            # Decode for MAE computation
            if self.model.bypass_autoencoder:
                y_pred = z_pred
                y_true = targets.to(z_pred.dtype)
            else:
                B1, M1, LD = targets.shape
                targets_matched = targets.to(z_pred.dtype)
                y_pred = self.model.decode(z_pred.reshape(B1 * M1, LD)).view(B1, M1, -1)
                y_true = self.model.decode(targets_matched.reshape(B1 * M1, LD)).view(B1, M1, -1)

            mae_norm = (y_pred - y_true).abs().mean()

        # Physical space MAE if available
        mae_phys = 0.0
        if have_phys:
            y_pred_phys = self._norm_helper.denormalize(
                y_pred.reshape(-1, y_pred.shape[-1]), self._species_vars
            ).view_as(y_pred)
            y_true_phys = self._norm_helper.denormalize(
                y_true.reshape(-1, y_true.shape[-1]), self._species_vars
            ).view_as(y_true)
            mae_phys = (y_pred_phys - y_true_phys).abs().mean().item()

        return {
            'loss': float(loss.item()),
            'mse': float(loss.item()),
            'mae_norm': float(mae_norm.item()),
            'mae_phys': float(mae_phys),
            'batch_size': B
        }

    def _validate_variable_batch(
            self,
            inputs: torch.Tensor,
            targets_list: List[torch.Tensor],
            times_list: List[torch.Tensor],
            have_phys: bool
    ) -> Dict[str, float]:
        """Validate batch with variable time grids."""
        B = inputs.size(0)
        inputs = inputs.to(self.device, non_blocking=True)

        total_loss = 0.0
        total_mse = 0.0
        total_mae_norm = 0.0
        total_mae_phys = 0.0

        for i in range(B):
            input_i = inputs[i:i + 1]
            targets_i = targets_list[i].unsqueeze(0).to(self.device, non_blocking=True)
            times_i = self._ensure_time_2d(times_list[i].to(self.device, non_blocking=True))

            with autocast(**self.autocast_kwargs):
                z_pred, _ = self.model(input_i, decode=False, trunk_times=times_i)
                # Ensure dtype consistency
                loss = self.criterion(z_pred, targets_i.to(z_pred.dtype))

                # Decode for MAE
                if self.model.bypass_autoencoder:
                    y_pred = z_pred
                    y_true = targets_i.to(z_pred.dtype)
                else:
                    B1, M1, LD = targets_i.shape
                    targets_matched = targets_i.to(z_pred.dtype)
                    y_pred = self.model.decode(z_pred.reshape(B1 * M1, LD)).view(B1, M1, -1)
                    y_true = self.model.decode(targets_matched.reshape(B1 * M1, LD)).view(B1, M1, -1)

                mae_norm = (y_pred - y_true).abs().mean()

            # Physical space MAE
            if have_phys:
                y_pred_phys = self._norm_helper.denormalize(
                    y_pred.reshape(-1, y_pred.shape[-1]), self._species_vars
                ).view_as(y_pred)
                y_true_phys = self._norm_helper.denormalize(
                    y_true.reshape(-1, y_true.shape[-1]), self._species_vars
                ).view_as(y_true)
                mae_phys = (y_pred_phys - y_true_phys).abs().mean().item()
            else:
                mae_phys = 0.0

            total_loss += float(loss.item())
            total_mse += float(loss.item())
            total_mae_norm += float(mae_norm.item())
            total_mae_phys += float(mae_phys)

        return {
            'loss': total_loss / B,
            'mse': total_mse / B,
            'mae_norm': total_mae_norm / B,
            'mae_phys': total_mae_phys / B,
            'batch_size': B
        }

    def _load_normalization_helper(self):
        """Load normalization helper for physical space metrics."""
        try:
            from utils import load_json
            from normalizer import NormalizationHelper

            stats_path = Path(self.config["paths"]["processed_data_dir"]) / "normalization.json"
            stats = load_json(stats_path)
            self._norm_helper = NormalizationHelper(stats, self.device, self.config)

            # Try multiple possible keys for species variables
            names = (self.config["data"].get("target_species_variables") or
                     self.config["data"].get("species_variables"))
            self._species_vars = list(names) if names else None
        except Exception:
            self._norm_helper = None
            self._species_vars = None

    def _save_checkpoint(
            self,
            epoch: int,
            loss: float,
            prefix: str = "checkpoint",
            is_best: bool = False
    ):
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
        """Save training history to JSON."""
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