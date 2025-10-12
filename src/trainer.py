#!/usr/bin/env python3
"""
Production Trainer for Koopman Autoencoder
===========================================
Optimized for autoregressive stability with stiff chemical systems.
Includes corrected rollout, semigroup, and KL losses for temporal consistency.
All critical stability fixes applied.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader


class AdaptiveStiffLoss(nn.Module):
    """
    Adaptive loss for stiff systems: MAE in log10 space + MSE in z-space.
    Simplified version without elemental conservation complexity.
    """

    def __init__(
            self,
            log_means: torch.Tensor,
            log_stds: torch.Tensor,
            lambda_phys: float = 1.0,
            lambda_z: float = 0.1,
            time_weight_mode: str = "none",  # "none", "edges", "exponential"
            device: Optional[torch.device] = None
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
            pred: [B,K,S] or [B,S] predictions in z-space
            target: [B,K,S] or [B,S] targets in z-space
            dt_norm: [B,K] normalized time steps (optional)
            mask: [B,K] validity mask (optional)

        Returns:
            Dictionary with 'total', 'phys', 'z' losses
        """
        # MSE in z-space (stabilizer)
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
            if pred.ndim == 3:
                weights = weights.unsqueeze(-1)  # [B,K,1] to broadcast over S
            loss_phys = loss_phys * weights
            loss_z = loss_z * weights

        # Masked averaging
        if mask is not None:
            m = mask
            if m.ndim < loss_phys.ndim:
                m = m.unsqueeze(-1)  # [B,K,1] to broadcast over S
            m = m.to(loss_phys.dtype)

            # If mask is [B,K,1], denom must count all species too
            if m.shape[-1] == 1 and loss_phys.ndim >= 3:
                denom = (m.sum() * loss_phys.shape[-1]).clamp_min(1)
            else:
                denom = m.sum().clamp_min(1)

            phys_mean = (loss_phys * m).sum() / denom
            z_mean = (loss_z * m).sum() / denom
        else:
            # True mean over all elements when unmasked
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


class Trainer:
    """
    Production trainer for Koopman autoencoder with autoregressive stability focus.
    Named 'Trainer' for compatibility with main.py.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            cfg: Dict[str, Any],  # Named 'cfg' to match main.py
            work_dir: Union[str, Path],
            device: torch.device,
            logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # Setup working directory
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = logger or logging.getLogger(__name__)

        # Extract training config
        tcfg = cfg.get('training', {})
        self.epochs = int(tcfg.get('epochs', 100))
        self.lr = float(tcfg.get('lr', 1e-4))
        self.min_lr = float(tcfg.get('min_lr', 1e-6))
        self.weight_decay = float(tcfg.get('weight_decay', 1e-5))
        self.grad_clip = float(tcfg.get('gradient_clip', 1.0))
        self.warmup_epochs = int(tcfg.get('warmup_epochs', 0))

        # Loss mode
        self.loss_mode = tcfg.get('loss_mode', 'adaptive_stiff')

        # Auxiliary losses config
        aux_cfg = tcfg.get('auxiliary_losses', {})
        self.rollout_enabled = aux_cfg.get('rollout_enabled', False)
        self.rollout_weight = float(aux_cfg.get('rollout_weight', 0.1))
        self.rollout_horizon = int(aux_cfg.get('rollout_horizon', 4))
        self.semigroup_enabled = aux_cfg.get('semigroup_enabled', False)
        self.semigroup_weight = float(aux_cfg.get('semigroup_weight', 0.1))
        self.kl_weight = float(tcfg.get('beta_kl', 0.0))

        # Teacher forcing schedule
        tf_cfg = aux_cfg.get('rollout_teacher_forcing', {})
        self.tf_mode = tf_cfg.get('mode', 'none')
        self.tf_start = float(tf_cfg.get('start_p', 1.0))
        self.tf_end = float(tf_cfg.get('end_p', 0.0))
        self.tf_end_epoch = int(tf_cfg.get('end_epoch', 50))

        # Setup components
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()

        # Cache dt stats to avoid I/O in hot path
        self._cache_dt_stats()

        # Mixed precision (read from top-level config)
        mp_cfg = self.cfg.get('mixed_precision', {})
        mode = str(mp_cfg.get('mode', 'bf16')).lower()
        self.use_amp = torch.cuda.is_available() and mode != 'none'
        self.amp_dtype = torch.bfloat16 if 'bf16' in mode else torch.float16

        if self.use_amp and self.amp_dtype == torch.float16:
            self.logger.warning("FP16 mode without GradScaler may be unstable. Consider using BF16.")

        # State
        self.start_epoch = 0
        self.current_epoch = 0  # Track current epoch for rollout horizon
        self.best_val_loss = float('inf')

        # Check for resume
        self._check_resume()

    def _cache_dt_stats(self):
        """Cache dt normalization stats to avoid I/O in rollout loss."""
        manifest_path = Path(self.cfg['paths']['processed_data_dir']) / 'normalization.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if 'dt' in manifest:
            dt_stats = manifest['dt']
            self.dt_log_min = torch.tensor(
                dt_stats['log_min'],
                device=self.device,
                dtype=torch.float32
            )
            self.dt_log_max = torch.tensor(
                dt_stats['log_max'],
                device=self.device,
                dtype=torch.float32
            )
        else:
            raise ValueError("dt stats missing from normalization manifest")

    def _setup_loss(self):
        """Setup loss function based on configuration."""
        if self.loss_mode == 'mse':
            self.criterion = nn.MSELoss()
            self.loss_returns_dict = False
        else:  # adaptive_stiff
            # Load normalization statistics
            manifest_path = Path(self.cfg['paths']['processed_data_dir']) / 'normalization.json'
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Get target species statistics
            data_cfg = self.cfg.get('data', {})
            target_vars = data_cfg.get('target_species', data_cfg.get('species_variables', []))
            if not target_vars:
                raise ValueError("No target species specified")

            stats = manifest['per_key_stats']
            log_means = []
            log_stds = []
            for name in target_vars:
                if name not in stats:
                    raise KeyError(f"Species '{name}' not found in normalization stats")
                s = stats[name]
                log_means.append(float(s.get('log_mean', 0.0)))
                log_stds.append(float(s.get('log_std', 1.0)))

            # Create adaptive loss
            loss_cfg = self.cfg['training'].get('adaptive_stiff_loss', {})
            self.criterion = AdaptiveStiffLoss(
                log_means=torch.tensor(log_means, device=self.device),
                log_stds=torch.tensor(log_stds, device=self.device),
                lambda_phys=float(loss_cfg.get('lambda_phys', 1.0)),
                lambda_z=float(loss_cfg.get('lambda_z', 0.1)),
                time_weight_mode=loss_cfg.get('time_weight_mode', 'none')
            )
            self.loss_returns_dict = True

    def _setup_optimizer(self):
        """Setup AdamW optimizer with optional fused kernels."""
        try:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                fused=torch.cuda.is_available()
            )
        except (TypeError, RuntimeError):
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with optional warmup."""
        if self.warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.epochs - self.warmup_epochs),
                eta_min=self.min_lr
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
            )

    def _setup_logging(self):
        """Setup CSV logging."""
        self.log_file = self.work_dir / 'training_log.csv'
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ['epoch', 'train_loss', 'val_loss', 'lr', 'time']
                if self.loss_returns_dict:
                    headers.extend(['train_phys', 'train_z', 'val_phys', 'val_z'])
                if self.rollout_enabled:
                    headers.append('rollout_loss')
                if self.semigroup_enabled:
                    headers.append('semigroup_loss')
                if self.kl_weight > 0:
                    headers.append('kl_loss')
                writer.writerow(headers)

    def _check_resume(self):
        """Check for checkpoint to resume from."""
        resume = self.cfg.get('training', {}).get('resume', None)

        if resume == 'auto':
            # Prefer best_model.pt if it exists, otherwise latest by mtime
            best_path = self.work_dir / 'best_model.pt'
            if best_path.exists():
                resume = best_path
            else:
                checkpoints = list(self.work_dir.glob('*.pt'))
                if checkpoints:
                    resume = max(checkpoints, key=lambda p: p.stat().st_mtime)
                else:
                    return
        elif resume and Path(resume).exists():
            resume = Path(resume)
        else:
            return

        self.logger.info(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint.get('epoch', 0)
        self.current_epoch = self.start_epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

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

    def _compute_rollout_loss(
            self,
            y_i: torch.Tensor,
            dt: torch.Tensor,
            y_j: torch.Tensor,
            g: torch.Tensor,
            mask: Optional[torch.Tensor],
            tf_prob: float
    ) -> Optional[torch.Tensor]:
        """
        Multi-step rollout loss with truncated BPTT (1-step gradients only).
        Uses the SAME loss criterion as main training.
        Computes Δt mapping in FP32 to avoid AMP precision issues.
        """
        if dt.shape[1] < 2:
            return None

        # Normalize mask shape if needed
        if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # Process dt shape
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)

        B, K = dt.shape

        # Sort by time to ensure proper ordering
        dt_sorted, sort_idx = torch.sort(dt, dim=1)

        # ---- Δt mapping done in FP32 (independent of outer autocast) ----
        dt_log32 = (self.dt_log_min + dt_sorted.float() * (self.dt_log_max - self.dt_log_min)).float()
        dt_phys32 = (10.0 ** dt_log32).clamp_min(1e-30)

        # Incremental physical Δt
        dt_incr_phys32 = torch.empty_like(dt_phys32)
        dt_incr_phys32[:, 0] = dt_phys32[:, 0]
        if K > 1:
            incr32 = dt_phys32[:, 1:] - dt_phys32[:, :-1]
            eps32 = (10.0 ** self.dt_log_min.item()) * 1e-6  # floor relative to min scale
            dt_incr_phys32[:, 1:] = incr32.clamp_min(eps32)

        # Re-normalize increments using SAME stats
        dt_incr_log32 = torch.log10(dt_incr_phys32)
        rng32 = (self.dt_log_max - self.dt_log_min).clamp_min(1e-12)
        dt_incr_norm = ((dt_incr_log32 - self.dt_log_min) / rng32).clamp(0.0, 1.0).to(y_i.dtype)

        # Sort targets/masks to match time ordering
        y_j_sorted = torch.gather(
            y_j, dim=1,
            index=sort_idx.unsqueeze(-1).expand(-1, -1, y_j.shape[-1])
        )
        mask_sorted = torch.gather(mask, dim=1, index=sort_idx) if mask is not None else None

        # Horizon schedule (grows with epoch)
        if self.current_epoch < 10:
            max_allowed_horizon = 2
        elif self.current_epoch < 20:
            max_allowed_horizon = 4
        elif self.current_epoch < 40:
            max_allowed_horizon = 6
        else:
            max_allowed_horizon = self.rollout_horizon

        max_h = min(max_allowed_horizon, K)
        horizon = int(torch.randint(1, max_h + 1, (), device=self.device).item())

        rollout_losses = []
        y_curr = y_i

        # Subset info
        S_in = getattr(self.model, 'S_in', None)
        S_out = getattr(self.model, 'S_out', None)
        target_idx = getattr(self.model, 'target_idx', None)

        for t in range(horizon):
            # Predict with incremental normalized dt
            y_pred = self.model(y_curr, dt_incr_norm[:, t:t + 1], g)
            if y_pred.ndim == 3 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
            y_true = y_j_sorted[:, t]

            # Non-finite guard
            if not torch.isfinite(y_pred).all():
                self.logger.warning(f"Non-finite y_pred at rollout step {t}; skipping rollout loss")
                return None

            # Same loss as main training
            if self.loss_returns_dict:
                dt_single = dt_incr_norm[:, t:t + 1]
                mask_single = mask_sorted[:, t:t + 1] if mask_sorted is not None else None
                loss_dict = self.criterion(y_pred.unsqueeze(1), y_true.unsqueeze(1), dt_single, mask_single)
                step_loss = loss_dict['total']
            else:
                if mask_sorted is not None:
                    m = mask_sorted[:, t].to(y_pred.dtype)  # [B]
                    l_elm = F.mse_loss(y_pred, y_true, reduction='none')  # [B,S]
                    l_sample = l_elm.mean(-1)  # [B]
                    step_loss = (l_sample * m).sum() / m.sum().clamp_min(1)
                else:
                    step_loss = F.mse_loss(y_pred, y_true)

            rollout_losses.append(step_loss)

            # Truncated BPTT (detach for next step)
            y_pred_det = y_pred.detach()
            y_true_det = y_true.detach()

            # Teacher forcing mask
            tf_vec = (torch.rand(B, 1, device=self.device) < tf_prob).to(y_pred.dtype)
            if mask_sorted is not None:
                tf_vec = tf_vec * mask_sorted[:, t:t + 1].to(tf_vec.dtype)

            mixed_next = tf_vec * y_true_det + (1.0 - tf_vec) * y_pred_det

            # Subset-aware state update
            if (S_in is not None) and (S_out is not None) and (S_in != S_out):
                if target_idx is None:
                    raise RuntimeError("Subset rollout requires model.target_idx")
                assert mixed_next.shape[-1] == len(target_idx)
                y_upd = y_curr.detach().clone()
                y_upd[:, target_idx] = mixed_next
                y_curr = y_upd
            else:
                y_curr = mixed_next

        return torch.stack(rollout_losses).mean() if rollout_losses else None

    def _compute_semigroup_loss(
            self,
            y_i: torch.Tensor,
            g: torch.Tensor
    ) -> torch.Tensor:
        """Semigroup property: f(t1+t2) = f(t2) ∘ f(t1)."""
        B = y_i.shape[0]

        # Sample random time steps in [0, 0.5] normalized space
        t1 = torch.rand(B, 1, device=self.device) * 0.5
        t2 = torch.rand(B, 1, device=self.device) * 0.5
        t_sum = t1 + t2

        # Direct path
        y_direct = self.model(y_i, t_sum, g).squeeze(1)

        # Composed path
        with torch.no_grad():
            y_mid = self.model(y_i, t1, g).squeeze(1)
        y_composed = self.model(y_mid, t2, g).squeeze(1)

        return F.mse_loss(y_direct, y_composed)

    def _compute_auxiliary_losses(
            self,
            y_i: torch.Tensor,
            dt: torch.Tensor,
            y_j: torch.Tensor,
            g: torch.Tensor,
            mask: Optional[torch.Tensor],
            epoch: int,
            tf_prob: float
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for temporal consistency."""
        aux_losses = {}

        # KL loss from VAE
        if self.kl_weight > 0 and hasattr(self.model, 'kl_loss') and self.model.kl_loss is not None:
            aux_losses['kl'] = self.kl_weight * self.model.kl_loss

        # Rollout loss
        if self.rollout_enabled and self.rollout_weight > 0:
            rollout_loss = self._compute_rollout_loss(
                y_i, dt, y_j, g, mask, tf_prob
            )
            if rollout_loss is not None:
                aux_losses['rollout'] = self.rollout_weight * rollout_loss

        # Semigroup loss
        if self.semigroup_enabled and self.semigroup_weight > 0:
            semigroup_loss = self._compute_semigroup_loss(y_i, g)
            aux_losses['semigroup'] = self.semigroup_weight * semigroup_loss

        return aux_losses

    def train(self) -> float:
        """Main training loop."""
        start_time = time.time()

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch  # Track current epoch for rollout horizon
            epoch_start = time.time()

            # Set dataset epoch for deterministic sampling
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)

            # Training
            train_metrics = self._train_epoch(epoch)

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
            else:
                val_metrics = {'loss': train_metrics['loss']}

            # Update scheduler
            self.scheduler.step()

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)

            # Always save last checkpoint
            self._save_checkpoint(epoch, is_best=False)

            # Log to CSV
            self._log_metrics(epoch, train_metrics, val_metrics, epoch_start)

            # Console output
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train={train_metrics['loss']:.4e} | "
                f"val={val_metrics['loss']:.4e} | "
                f"lr={lr:.2e} | "
                f"time={time.time() - epoch_start:.1f}s"
            )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        return self.best_val_loss

    # === Trainer._train_epoch ===
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch with non-finite guard before backward."""
        self.model.train()

        metrics = {
            'loss': 0.0, 'phys': 0.0, 'z': 0.0,
            'rollout': 0.0, 'semigroup': 0.0, 'kl': 0.0,
            'count': 0
        }

        tf_prob = self._get_teacher_forcing_prob(epoch)

        for batch in self.train_loader:
            try:
                # Unpack
                if len(batch) == 6:
                    y_i, dt, y_j, g, _, mask = batch
                else:
                    y_i, dt, y_j, g, _ = batch
                    mask = None

                # To device
                y_i = y_i.to(self.device, non_blocking=True)
                dt = dt.to(self.device, non_blocking=True)
                y_j = y_j.to(self.device, non_blocking=True)
                g = g.to(self.device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)

                if dt.ndim == 3 and dt.shape[-1] == 1:
                    dt = dt.squeeze(-1)

                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    # Forward
                    pred = self.model(y_i, dt, g)

                    # Main loss
                    if self.loss_returns_dict:
                        loss_dict = self.criterion(pred, y_j, dt, mask)
                        main_loss = loss_dict['total']
                        metrics['phys'] += loss_dict['phys'].item()
                        metrics['z'] += loss_dict['z'].item()
                    else:
                        if mask is not None and pred.ndim == 3:
                            l_elm = F.mse_loss(pred, y_j, reduction='none')  # [B,K,S]
                            m = mask.unsqueeze(-1).to(l_elm.dtype)  # [B,K,1]
                            denom = (m.sum() * l_elm.shape[-1]).clamp_min(1)
                            main_loss = (l_elm * m).sum() / denom
                        else:
                            main_loss = self.criterion(pred, y_j)

                    # Aux losses
                    aux_losses = self._compute_auxiliary_losses(y_i, dt, y_j, g, mask, epoch, tf_prob)
                    total_loss = main_loss
                    for key, value in aux_losses.items():
                        total_loss = total_loss + value
                        metrics[key] += value.item()

                # ---- Non-finite guard before backward ----
                if not torch.isfinite(total_loss):
                    self.logger.warning("Non-finite total_loss; skipping batch.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward & step
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                metrics['loss'] += total_loss.item()
                metrics['count'] += 1

            except Exception as e:
                self.logger.warning(f"Skipping bad batch: {e}")
                continue

        for k in metrics:
            if k != 'count':
                metrics[k] /= max(1, metrics['count'])
        return metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validation epoch."""
        self.model.eval()

        metrics = {
            'loss': 0.0,
            'phys': 0.0,
            'z': 0.0,
            'count': 0
        }

        for batch in self.val_loader:
            # Unpack
            if len(batch) == 6:
                y_i, dt, y_j, g, _, mask = batch
            else:
                y_i, dt, y_j, g, _ = batch
                mask = None

            # Move to device
            y_i = y_i.to(self.device, non_blocking=True)
            dt = dt.to(self.device, non_blocking=True)
            y_j = y_j.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)

            if dt.ndim == 3 and dt.shape[-1] == 1:
                dt = dt.squeeze(-1)

            # Forward
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                pred = self.model(y_i, dt, g)

                if self.loss_returns_dict:
                    loss_dict = self.criterion(pred, y_j, dt, mask)
                    metrics['loss'] += loss_dict['total'].item()
                    metrics['phys'] += loss_dict['phys'].item()
                    metrics['z'] += loss_dict['z'].item()
                else:
                    if mask is not None and pred.ndim == 3:
                        loss_elm = F.mse_loss(pred, y_j, reduction='none')
                        m = mask.unsqueeze(-1).to(loss_elm.dtype)
                        denom = (m.sum() * loss_elm.shape[-1]).clamp_min(1)
                        loss = (loss_elm * m).sum() / denom
                    else:
                        loss = self.criterion(pred, y_j)
                    metrics['loss'] += loss.item()

            metrics['count'] += 1

        # Average
        for key in metrics:
            if key != 'count':
                metrics[key] /= max(1, metrics['count'])

        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint with legacy naming for compatibility."""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg
        }

        # Use legacy names for compatibility with existing pipeline
        path = self.work_dir / ('best_model.pt' if is_best else 'last_model.pt')
        torch.save(checkpoint, path)

    def _log_metrics(
            self,
            epoch: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            epoch_start: float
    ):
        """Log metrics to CSV."""
        row = [
            epoch,
            f"{train_metrics['loss']:.6e}",
            f"{val_metrics['loss']:.6e}",
            f"{self.optimizer.param_groups[0]['lr']:.2e}",
            f"{time.time() - epoch_start:.1f}"
        ]

        if self.loss_returns_dict:
            row.extend([
                f"{train_metrics.get('phys', 0):.6e}",
                f"{train_metrics.get('z', 0):.6e}",
                f"{val_metrics.get('phys', 0):.6e}",
                f"{val_metrics.get('z', 0):.6e}"
            ])

        if self.rollout_enabled:
            row.append(f"{train_metrics.get('rollout', 0):.6e}")

        if self.semigroup_enabled:
            row.append(f"{train_metrics.get('semigroup', 0):.6e}")

        if self.kl_weight > 0:
            row.append(f"{train_metrics.get('kl', 0):.6e}")

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @torch.no_grad()
    def evaluate_rollout(
            self,
            num_steps: int = 100,
            dt_value: float = 0.01,  # Note: this is normalized dt
            batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate autoregressive rollout stability.

        Args:
            num_steps: Number of rollout steps
            dt_value: Normalized time step (0.01 is near minimum dt)
            batch_size: Batch size for meaningful variance computation

        Returns:
            Dictionary of rollout metrics
        """
        self.model.eval()

        # Get batch for evaluation
        batch = next(iter(self.val_loader or self.train_loader))
        y_i, _, _, g = batch[:4]

        # Use small batch for meaningful variance
        B = min(batch_size, y_i.shape[0])
        y_i = y_i[:B].to(self.device)
        g = g[:B].to(self.device)
        dt_step = torch.tensor([dt_value], device=self.device)

        # Rollout
        if hasattr(self.model, 'rollout'):
            trajectory = self.model.rollout(y_i, g, dt_step, num_steps)
        else:
            trajectory = []
            y_curr = y_i
            for _ in range(num_steps):
                y_next = self.model(y_curr, dt_step.unsqueeze(0).expand(B, -1), g)
                if y_next.ndim == 3:
                    y_next = y_next.squeeze(1)
                trajectory.append(y_next)
                y_curr = y_next
            trajectory = torch.stack(trajectory, dim=1)

        # Compute metrics
        metrics = {}

        # Check for NaN/Inf
        metrics['has_nan'] = torch.isnan(trajectory).any().item()
        metrics['has_inf'] = torch.isinf(trajectory).any().item()

        # Variance across species dimension (meaningful with batch > 1)
        var_initial = trajectory[:, 0, :].var(dim=0, unbiased=False).mean().item()
        var_final = trajectory[:, -1, :].var(dim=0, unbiased=False).mean().item()
        metrics['variance_ratio'] = var_final / (var_initial + 1e-8)

        # Mean drift
        mean_initial = trajectory[:, 0].mean().item()
        mean_final = trajectory[:, -1].mean().item()
        metrics['mean_drift'] = abs(mean_final - mean_initial)

        # Check simplex constraint if using softmax (only for full species)
        if getattr(self.model, 'softmax_head', False):
            full_simplex = (getattr(self.model, 'S_out', None) == getattr(self.model, 'S_in', None))
            has_stats = (getattr(self.model, 'log_mean', None) is not None) and (
                    getattr(self.model, 'log_std', None) is not None)
            if full_simplex and has_stats:
                # Convert to physical space and check sum using model stats
                mu = self.model.log_mean.to(trajectory.dtype).view(1, 1, -1)
                sig = self.model.log_std.to(trajectory.dtype).view(1, 1, -1)
                traj_log10 = trajectory * sig + mu
                traj_phys = torch.pow(10.0, traj_log10)
                sums = traj_phys.sum(dim=-1)
                metrics['max_simplex_error'] = (sums - 1.0).abs().max().item()
                metrics['mean_simplex_error'] = (sums - 1.0).abs().mean().item()

        return metrics

    def check_model_health(self) -> dict:
        """Check model parameters and gradients for issues."""
        health = {
            'has_nan_params': False,
            'has_inf_params': False,
            'max_param': 0.0,
            'max_grad': 0.0,
            'problem_layers': []
        }

        for name, param in self.model.named_parameters():
            # Check parameters
            if torch.isnan(param).any():
                health['has_nan_params'] = True
                health['problem_layers'].append(f"{name} (NaN param)")
            if torch.isinf(param).any():
                health['has_inf_params'] = True
                health['problem_layers'].append(f"{name} (Inf param)")

            param_max = param.abs().max().item()
            health['max_param'] = max(health['max_param'], param_max)

            # Check gradients
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    health['problem_layers'].append(f"{name} (NaN grad)")
                if torch.isinf(param.grad).any():
                    health['problem_layers'].append(f"{name} (Inf grad)")

                grad_max = param.grad.abs().max().item()
                health['max_grad'] = max(health['max_grad'], grad_max)

        return health