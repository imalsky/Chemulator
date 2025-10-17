#!/usr/bin/env python3
"""
Production Trainer for Koopman Autoencoder (PyTorch Lightning Version)
======================================================================
Optimized for autoregressive stability with stiff chemical systems.
Uses adaptive stiff loss with configurable lambda_phys and lambda_z weights.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer as LightningTrainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader


class AdaptiveStiffLoss(nn.Module):
    """
    Adaptive loss for stiff systems.
    Combines MAE in log10 physical space with MSE in normalized space for stability.
    """

    def __init__(
            self,
            log_means: torch.Tensor,
            log_stds: torch.Tensor,
            lambda_phys: float = 1.0,
            lambda_z: float = 1.0,
            time_weight_mode: str = "none"
    ):
        super().__init__()
        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=1e-10))
        self.lambda_phys = lambda_phys
        self.lambda_z = lambda_z
        self.time_weight_mode = time_weight_mode

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            dt_norm: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [B,K,S] predictions in normalized target space
            target: [B,K,S] targets in normalized target space
            dt_norm: [B,K] normalized time steps (optional, for time weighting)
            mask: [B,K] validity mask (optional)

        Returns:
            Dictionary with 'total', 'phys', 'z' losses
        """
        # MSE in normalized space (stabilizer)
        loss_z = (pred - target) ** 2

        # MAE in log10 physical space
        pred_log = pred * self.log_stds + self.log_means
        true_log = target * self.log_stds + self.log_means
        loss_phys = torch.abs(pred_log - true_log)

        # Time weighting if requested
        if self.time_weight_mode != "none" and dt_norm is not None:
            if dt_norm.ndim == 3:
                dt_norm = dt_norm.squeeze(-1)
            weights = self._compute_time_weights(dt_norm)
            # Ensure weights broadcast correctly
            while weights.ndim < pred.ndim:
                weights = weights.unsqueeze(-1)
            loss_phys = loss_phys * weights
            loss_z = loss_z * weights

        # Masked averaging
        if mask is not None:
            m = mask
            if m.ndim < loss_phys.ndim:
                m = m.unsqueeze(-1)
            m = m.to(loss_phys.dtype)

            if m.shape[-1] == 1 and loss_phys.ndim >= 3:
                denom = (m.sum() * loss_phys.shape[-1]).clamp_min(1)
            else:
                denom = m.sum().clamp_min(1)

            phys_mean = (loss_phys * m).sum() / denom
            z_mean = (loss_z * m).sum() / denom
        else:
            phys_mean = loss_phys.mean()
            z_mean = loss_z.mean()

        total = self.lambda_phys * phys_mean + self.lambda_z * z_mean

        return {
            'total': total,
            'phys': phys_mean.detach(),
            'z': z_mean.detach()
        }

    def _compute_time_weights(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Compute time-dependent loss weights."""
        if self.time_weight_mode == "edges":
            # U-shaped: emphasize trajectory edges
            return 1.0 + 2.0 * (1.0 - 4.0 * dt_norm * (1.0 - dt_norm))
        elif self.time_weight_mode == "exponential":
            # Exponential decay from start
            return torch.exp(-2.0 * dt_norm)
        else:
            return torch.ones_like(dt_norm)


class EpochSetterCallback(Callback):
    """Callback to set epoch on dataset for deterministic sampling."""

    def on_train_epoch_start(self, trainer, pl_module):
        """Robustly find and set epoch on dataset."""
        epoch = trainer.current_epoch
        p = pl_module._get_teacher_forcing_prob(epoch)
        H = pl_module._get_rollout_horizon(epoch)
        pl_module.log('tf_p', p, on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('rollout_H', H, on_step=False, on_epoch=True, prog_bar=False)

        dl = getattr(trainer, "train_dataloader", None)

        # Handle callable dataloaders (PL 2.0+)
        if callable(dl):
            dl = dl()

        # Handle list of dataloaders
        if isinstance(dl, (list, tuple)):
            ds = getattr(dl[0], "dataset", None) if dl else None
        else:
            ds = getattr(dl, "dataset", None) if dl else None

        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

class KoopmanLightning(LightningModule):
    """
    Lightning wrapper for Koopman autoencoder training.
    """

    def __init__(
            self,
            model: nn.Module,
            cfg: Dict[str, Any],
            work_dir: Union[str, Path]
    ):
        super().__init__()

        # Save hyperparameters (excluding model to avoid duplication)
        self.save_hyperparameters(ignore=['model'])

        # Core model
        self.model = model
        self.cfg = cfg
        self.work_dir = Path(work_dir)

        # Extract training config
        tcfg = cfg.get('training', {})
        self.learning_rate = float(tcfg.get('lr', 3e-4))
        self.min_lr = float(tcfg.get('min_lr', 1e-6))
        self.weight_decay = float(tcfg.get('weight_decay', 1e-4))
        self.grad_clip = float(tcfg.get('gradient_clip', 1.0))
        self.warmup_epochs = int(tcfg.get('warmup_epochs', 10))
        self.epochs = int(tcfg.get('epochs', 150))

        # Auxiliary losses config
        aux_cfg = tcfg.get('auxiliary_losses', {})

        # Rollout loss
        self.rollout_enabled = aux_cfg.get('rollout_enabled', True)
        self.rollout_weight = float(aux_cfg.get('rollout_weight', 0.5))
        self.max_rollout_horizon = int(aux_cfg.get('rollout_horizon', 12))

        # Semigroup loss
        self.semigroup_enabled = aux_cfg.get('semigroup_enabled', True)
        self.semigroup_weight = float(aux_cfg.get('semigroup_weight', 0.2))

        # Teacher forcing schedule
        tf_cfg = aux_cfg.get('rollout_teacher_forcing', {})
        self.tf_mode = tf_cfg.get('mode', 'linear')
        self.tf_start = float(tf_cfg.get('start_p', 0.8))
        self.tf_end = float(tf_cfg.get('end_p', 0.0))
        self.tf_end_epoch = int(tf_cfg.get('end_epoch', 60))

        # Setup loss function
        self._setup_loss()
        self._cache_dt_stats()

        # Best validation loss tracking
        self.best_val_loss = float('inf')

        # Automatic optimization settings
        self.automatic_optimization = True

    def _setup_loss(self):
        """Setup adaptive stiff loss."""
        # Load normalization statistics
        manifest_path = Path(self.cfg['paths']['processed_data_dir']) / 'normalization.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Get target species statistics
        data_cfg = self.cfg.get('data', {})
        target_vars = data_cfg.get('species_variables', [])
        if not target_vars:
            raise ValueError("No species specified")

        stats = manifest['per_key_stats']
        log_means = []
        log_stds = []
        for name in target_vars:
            if name not in stats:
                raise KeyError(f"Species '{name}' not found in normalization stats")
            s = stats[name]
            log_means.append(float(s.get('log_mean', 0.0)))
            log_stds.append(float(s.get('log_std', 1.0)))

        # Get loss configuration
        loss_cfg = self.cfg['training'].get('adaptive_stiff_loss', {})

        # Ensure float32 for loss buffers to avoid dtype issues
        self.criterion = AdaptiveStiffLoss(
            log_means=torch.tensor(log_means, dtype=torch.float32),
            log_stds=torch.tensor(log_stds, dtype=torch.float32),
            lambda_phys=float(loss_cfg.get('lambda_phys', 1.0)),
            lambda_z=float(loss_cfg.get('lambda_z', 1.0)),
            time_weight_mode=loss_cfg.get('time_weight_mode', 'none')
        )

    def _cache_dt_stats(self):
        """Cache dt normalization stats."""
        manifest_path = Path(self.cfg['paths']['processed_data_dir']) / 'normalization.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if 'dt' not in manifest:
            raise ValueError("dt stats missing from normalization manifest")

        dt_stats = manifest['dt']
        self.register_buffer('dt_log_min', torch.tensor(dt_stats['log_min'], dtype=torch.float32))
        self.register_buffer('dt_log_max', torch.tensor(dt_stats['log_max'], dtype=torch.float32))

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Setup scheduler with warmup (portable across PyTorch versions)
        if self.warmup_epochs > 0:
            # Use only start_factor and total_iters for compatibility
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=self.warmup_epochs
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.epochs - self.warmup_epochs),
                eta_min=self.min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _get_teacher_forcing_prob(self, epoch: int) -> float:
        """Get teacher forcing probability for current epoch."""
        if self.tf_mode == 'none':
            return 0.0
        elif self.tf_mode == 'constant':
            return self.tf_start
        elif self.tf_mode == 'linear':
            if epoch >= self.tf_end_epoch:
                return self.tf_end
            t = epoch / max(1, self.tf_end_epoch)
            return self.tf_start + (self.tf_end - self.tf_start) * t
        elif self.tf_mode == 'cosine_ramp':
            if epoch >= self.tf_end_epoch:
                return self.tf_end
            t = epoch / max(1, self.tf_end_epoch)
            cos_t = 0.5 * (1.0 + np.cos(np.pi * t))
            return self.tf_end + (self.tf_start - self.tf_end) * cos_t
        else:
            return 0.0

    def _get_rollout_horizon(self, epoch: int) -> int:
        """Progressive horizon schedule for rollout."""
        if epoch < 5:
            return min(2, self.max_rollout_horizon)
        elif epoch < 10:
            return min(4, self.max_rollout_horizon)
        elif epoch < 15:
            return min(8, self.max_rollout_horizon)
        else:
            return self.max_rollout_horizon

    def _compute_rollout_loss(
            self,
            y_i: torch.Tensor,
            dt: torch.Tensor,
            y_j: torch.Tensor,
            g: torch.Tensor,
            mask: Optional[torch.Tensor],
            epoch: int
    ) -> Optional[torch.Tensor]:
        """Multi-step rollout loss with teacher forcing."""
        if dt.shape[1] < 2:
            return None

        if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        B, K = dt.shape
        horizon = min(self._get_rollout_horizon(epoch), K)
        tf_prob = self._get_teacher_forcing_prob(epoch)

        rollout_losses = []
        y_curr = y_i

        for t in range(horizon):
            # Single step prediction
            dt_step = dt[:, t:t + 1]
            if dt_step.ndim == 2:
                dt_step = dt_step.unsqueeze(-1)

            y_pred = self.model(y_curr, dt_step, g)
            if y_pred.ndim == 3 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)

            y_true = y_j[:, t]

            # Compute loss for this step
            mask_t = mask[:, t:t + 1] if mask is not None else None
            loss_dict = self.criterion(
                y_pred.unsqueeze(1),
                y_true.unsqueeze(1),
                dt_step[:, :, 0] if dt_step.ndim == 3 else dt_step,
                mask_t
            )
            rollout_losses.append(loss_dict['total'])

            # Teacher forcing for next step
            if t < horizon - 1:
                tf_mask = (torch.rand(B, 1, device=self.device) < tf_prob).float()
                if mask is not None:
                    tf_mask = tf_mask * mask[:, t:t + 1].float()
                y_curr = tf_mask * y_true + (1 - tf_mask) * y_pred.detach()

        return torch.stack(rollout_losses).mean() if rollout_losses else None

    def _compute_semigroup_loss(
            self,
            y_i: torch.Tensor,
            g: torch.Tensor
    ) -> torch.Tensor:
        """Semigroup property over full dt range [0,1]."""
        B = y_i.shape[0]

        # Sample across FULL normalized dt range
        t1 = torch.rand(B, 1, device=self.device)
        t2 = torch.rand(B, 1, device=self.device)
        t_sum = torch.clamp(t1 + t2, max=1.0)

        # Direct path
        y_direct = self.model(y_i, t_sum, g).squeeze(1)

        # Composed path (no detach - both steps learn composition)
        y_mid = self.model(y_i, t1, g).squeeze(1)
        y_composed = self.model(y_mid, t2, g).squeeze(1)

        return F.l1_loss(y_direct, y_composed)

    def _process_batch(self, batch):
        """Common batch processing for training and validation."""
        # Unpack batch
        if len(batch) == 6:
            y_i, dt, y_j, g, _, mask = batch
        else:
            y_i, dt, y_j, g, _ = batch
            mask = None

        # Handle dt dimensions
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        return y_i, dt, y_j, g, mask

    def training_step(self, batch, batch_idx):
        """Training step."""
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True)

        y_i, dt, y_j, g, mask = self._process_batch(batch)

        # Main prediction loss
        pred = self.model(y_i, dt, g)
        loss_dict = self.criterion(pred, y_j, dt, mask)
        main_loss = loss_dict['total']

        # Initialize total loss and metrics
        total_loss = main_loss

        # Log main loss components
        self.log('train_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_phys', loss_dict['phys'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_z', loss_dict['z'], on_step=False, on_epoch=True, prog_bar=False)

        # Auxiliary losses
        rollout_loss = torch.tensor(0.0, device=self.device)
        semigroup_loss = torch.tensor(0.0, device=self.device)

        # Rollout loss
        if self.rollout_enabled:
            rl = self._compute_rollout_loss(
                y_i, dt, y_j, g, mask, self.current_epoch
            )
            if rl is not None:
                total_loss = total_loss + self.rollout_weight * rl
                rollout_loss = rl

        # Semigroup loss
        if self.semigroup_enabled:
            sl = self._compute_semigroup_loss(y_i, g)
            total_loss = total_loss + self.semigroup_weight * sl
            semigroup_loss = sl

        # Log auxiliary losses
        self.log('rollout_loss', rollout_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('semigroup_loss', semigroup_loss, on_step=False, on_epoch=True, prog_bar=False)

        # Check for NaN - return zero loss that keeps graph valid
        if not torch.isfinite(total_loss):
            self.log('train_loss_nan', 1.0, on_step=True, prog_bar=True)
            # Return a zero loss with gradient enabled to avoid Lightning errors
            return torch.zeros((), device=self.device, requires_grad=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        y_i, dt, y_j, g, mask = self._process_batch(batch)

        # Forward pass
        pred = self.model(y_i, dt, g)
        loss_dict = self.criterion(pred, y_j, dt, mask)

        # Log metrics
        self.log('val_loss', loss_dict['total'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_phys', loss_dict['phys'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_z', loss_dict['z'], on_step=False, on_epoch=True, prog_bar=False)

        return loss_dict['total']

    def on_validation_epoch_end(self):
        """Track best validation loss - properly handle tensor to float conversion."""
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            # Convert tensor to float properly
            val_loss_float = float(val_loss.detach().cpu())
            if val_loss_float < self.best_val_loss:
                self.best_val_loss = val_loss_float

    @torch.no_grad()
    def evaluate_rollout(
            self,
            num_steps: int = 100,
            dt_value: float = 0.01,
            batch_size: int = 16,
            loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Evaluate autoregressive rollout stability.

        Args:
            num_steps: Number of rollout steps
            dt_value: Normalized dt in [0,1]
            batch_size: Batch size for evaluation
            loader: DataLoader to use (defaults to val or train)

        Returns:
            Dictionary of rollout metrics
        """
        self.eval()

        # Ensure dt_value is properly normalized
        dt_value = float(max(0.0, min(1.0, dt_value)))

        # Robustly get dataloader
        if loader is None:
            # Try validation dataloaders first
            dl = getattr(self.trainer, "val_dataloaders", None)
            if isinstance(dl, (list, tuple)) and dl:
                loader = dl[0]
            elif dl is not None:
                loader = dl
            else:
                # Fall back to train dataloader
                dl = getattr(self.trainer, "train_dataloader", None)
                loader = dl() if callable(dl) else dl

        if loader is None:
            raise ValueError("No dataloader available for rollout evaluation")

        batch = next(iter(loader))
        y_i, _, _, g = batch[:4]

        B = min(batch_size, y_i.shape[0])
        y_i = y_i[:B].to(self.device)
        g = g[:B].to(self.device)
        dt_step = torch.tensor([[dt_value]], device=self.device)

        # Rollout
        trajectory = self.model.rollout(y_i, g, dt_step, num_steps)

        # Compute metrics
        metrics = {}

        # Check for NaN/Inf
        metrics['has_nan'] = torch.isnan(trajectory).any().item()
        metrics['has_inf'] = torch.isinf(trajectory).any().item()

        # Variance evolution
        var_initial = trajectory[:, 0, :].var(dim=0, unbiased=False).mean().item()
        var_final = trajectory[:, -1, :].var(dim=0, unbiased=False).mean().item()
        metrics['variance_ratio'] = var_final / (var_initial + 1e-8)

        # Mean drift
        mean_initial = trajectory[:, 0].mean().item()
        mean_final = trajectory[:, -1].mean().item()
        metrics['mean_drift'] = abs(mean_final - mean_initial)

        return metrics


# Compatibility class that mimics the original Trainer interface
class Trainer:
    """
    Wrapper class that provides the same interface as the original Trainer
    but uses PyTorch Lightning internally.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            cfg: Dict[str, Any],
            work_dir: Union[str, Path],
            device: torch.device,
            logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    def train(self) -> float:
        """Train the model using PyTorch Lightning."""
        work_dir = self.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        # Wrap model in Lightning module
        lightning_model = KoopmanLightning(self.model, self.cfg, work_dir)

        # Training configuration
        tcfg = self.cfg.get('training', {})
        epochs = int(tcfg.get('epochs', 150))
        grad_clip = float(tcfg.get('gradient_clip', 1.0))

        # Mixed precision config
        mp_cfg = self.cfg.get('mixed_precision', {})
        mode = str(mp_cfg.get('mode', 'bf16')).lower()
        precision = '32-true'  # Default
        if torch.cuda.is_available() and mode != 'none':
            if 'bf16' in mode:
                precision = 'bf16-mixed'
            elif 'fp16' in mode or mode == '16':
                precision = '16-mixed'

        # Setup callbacks
        callbacks = []

        # Checkpoint callback with explicit filename
        checkpoint_callback = ModelCheckpoint(
            dirpath=work_dir,
            filename='best_model',
            auto_insert_metric_name=False,  # Ensure exact filename
            monitor='val_loss',
            mode='min',
            save_last=True,
            save_top_k=1,
            save_weights_only=False,
            verbose=True
        )
        callbacks.append(checkpoint_callback)

        # LR monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        # Epoch setter for dataset
        callbacks.append(EpochSetterCallback())

        # CSV logger with safe settings
        csv_logger = CSVLogger(save_dir=str(work_dir), name=".", version=None)

        # Check for resume
        resume_path = None
        resume_cfg = tcfg.get('resume', 'auto')

        # Check environment variable for resume override
        env_resume = os.environ.get('RESUME')
        if env_resume:
            resume_cfg = env_resume

        if resume_cfg == 'auto':
            best_path = work_dir / 'best_model.ckpt'
            last_path = work_dir / 'last.ckpt'
            best_pt = work_dir / 'best_model.pt'
            last_pt = work_dir / 'last_model.pt'

            # Check Lightning checkpoints first
            if best_path.exists():
                resume_path = best_path
            elif last_path.exists():
                resume_path = last_path
            # Also check old .pt files for compatibility
            elif best_pt.exists():
                self.logger.info(f"Found old .pt checkpoint, converting: {best_pt}")
                # Load old checkpoint and save model state for Lightning
                old_ckpt = torch.load(best_pt, map_location=self.device)
                self.model.load_state_dict(old_ckpt['model'])
            elif last_pt.exists():
                self.logger.info(f"Found old .pt checkpoint, converting: {last_pt}")
                old_ckpt = torch.load(last_pt, map_location=self.device)
                self.model.load_state_dict(old_ckpt['model'])
        elif resume_cfg and Path(resume_cfg).exists():
            resume_path = Path(resume_cfg)

        if resume_path and self.logger:
            self.logger.info(f"Resuming from {resume_path}")

        # Create Lightning trainer with safe gradient clipping
        trainer = LightningTrainer(
            max_epochs=epochs,
            accelerator='auto',
            devices=1,
            precision=precision,
            gradient_clip_val=float(max(0.0, grad_clip)),  # Ensure float
            gradient_clip_algorithm='norm',
            callbacks=callbacks,
            logger=csv_logger,
            default_root_dir=work_dir,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=False,  # Set to True for reproducibility (slower)
            benchmark=True,  # Enable cuDNN benchmark
            log_every_n_steps=50,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,  # Disable sanity check
        )

        # Train
        trainer.fit(
            lightning_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=resume_path
        )

        # Properly retrieve best validation loss
        best_val = None

        # Try to get from checkpoint callback first
        if checkpoint_callback.best_model_score is not None:
            best_val = float(checkpoint_callback.best_model_score.detach().cpu())
        elif 'val_loss' in trainer.callback_metrics:
            best_val = float(trainer.callback_metrics['val_loss'].detach().cpu())
        else:
            best_val = float(lightning_model.best_val_loss)

        # Optional: Save legacy .pt format for compatibility
        if checkpoint_callback.best_model_path:
            try:
                state = torch.load(checkpoint_callback.best_model_path, map_location=self.device)
                sd = state.get('state_dict', {})

                # Keep only model.* keys and strip the prefix
                model_state = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}

                # Save in old format
                torch.save({
                    'model': model_state,
                    'epoch': state.get('epoch', 0),
                    'best_val_loss': best_val,
                    'config': self.cfg
                }, self.work_dir / 'best_model.pt')
                self.logger.info("Saved legacy best_model.pt for compatibility")
            except Exception as e:
                self.logger.warning(f"Could not save legacy .pt file: {e}")

        return best_val if best_val is not None else float('inf')