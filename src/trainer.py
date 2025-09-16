#!/usr/bin/env python3
"""
Flow-map DeepONet Trainer with Restart Support
================================================
Training loop implementation with checkpoint/restart capabilities.

Features:
- Shape-agnostic loss computation (handles both [B,S] and [B,K,S] targets)
- Mixed precision training with automatic mixed precision (AMP)
- AdamW optimizer with cosine annealing and linear warmup
- Deterministic per-epoch sampling via dataset.set_epoch()
- CSV logging for training metrics
- Full checkpoint saving/loading for restart
- SIGTERM handling for cluster preemption
"""

from __future__ import annotations

import csv
import math
import os
import random
import signal
import time
import contextlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
        """Update learning rate for given epoch."""
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
    Trainer for Flow-map DeepONet models with restart support.

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
        if self.max_train_steps_per_epoch == 0:
            self.max_train_steps_per_epoch = None
        self.max_val_batches = training_cfg.get("max_val_batches", None)
        if self.max_val_batches == 0:
            self.max_val_batches = None
        self.use_compile = bool(training_cfg.get("torch_compile", False))

        # Setup mixed precision training
        self._setup_mixed_precision()

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

        # Initialize training state
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self._current_epoch = 0  # Track current epoch for SIGTERM

        # Checkpoint paths
        self.best_model_path = self.work_dir / "best_model.pt"  # weights-only for backward compat
        self.best_ckpt_path = self.work_dir / "best.ckpt"  # full checkpoint
        self.last_ckpt_path = self.work_dir / "last.ckpt"  # full checkpoint

        # Setup SIGTERM handler for cluster preemption
        self._setup_sigterm_handler()

        # Check for resume
        self._check_resume()

    def _setup_sigterm_handler(self) -> None:
        """Setup handler for SIGTERM signal (cluster preemption)."""

        def sigterm_handler(signum, frame):
            self.logger.warning("SIGTERM received: saving checkpoint and exiting")
            # Save current epoch being processed (will be re-run on resume)
            self._save_full_checkpoint(self.last_ckpt_path, self._current_epoch)
            os._exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)

    def _check_resume(self) -> None:
        """Check for and load checkpoint if resuming."""
        # Check environment variable (set by main.py)
        resume = os.environ.get("RESUME", "").strip()
        if not resume:
            return

        # Resolve checkpoint path
        if resume.lower() == "auto":
            # Try to find best checkpoint
            if self.last_ckpt_path.exists():
                ckpt_path = self.last_ckpt_path
            elif self.best_ckpt_path.exists():
                ckpt_path = self.best_ckpt_path
            else:
                # Look for any checkpoint in work_dir
                ckpts = sorted(self.work_dir.glob("*.ckpt"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts:
                    ckpt_path = ckpts[0]
                else:
                    self.logger.info("No checkpoint found for auto-resume")
                    return
        else:
            ckpt_path = Path(resume).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load checkpoint
        self.logger.info(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Restore model
        self.model.load_state_dict(checkpoint["model"])

        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore scaler if using mixed precision
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Restore training state
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self._current_epoch = self.start_epoch  # Initialize current epoch tracker

        # Restore RNG states for reproducibility
        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])

        self.logger.info(f"Resumed from epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4e}")

    def _save_full_checkpoint(self, path: Path, epoch: int) -> None:
        """Save full checkpoint for restart."""
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
        }

        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

        if self.scaler:
            checkpoint["scaler"] = self.scaler.state_dict()

        # Atomic write with proper suffix handling
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        os.replace(str(tmp_path), str(path))

    def _save_weights_only(self) -> None:
        """Save model weights only (for backward compatibility and inference)."""
        checkpoint = {
            "model": self.model.state_dict(),
            "config": self.cfg,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.best_model_path)

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
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

    def _setup_optimizer(self) -> None:
        """Initialize optimizer with fused operations if available."""
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

    def _ensure_norm_helper(self):
        """
        Lazily load NormalizationHelper and resolve the correct species list
        (target subset if provided, else all species).
        """
        if not hasattr(self, "_norm_helper"):
            import json
            from pathlib import Path
            from normalizer import NormalizationHelper

            manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            self._norm_helper = NormalizationHelper(manifest, device=self.device)

            data_cfg = self.cfg.get("data", {})
            self._species_keys = list(
                data_cfg.get("target_species") or data_cfg.get("species_variables")
            )
        return self._norm_helper, self._species_keys

    def _logspace_abundance_weight(
            self,
            y_phys: torch.Tensor,
            *,
            low: float = 1e-25,
            high: float = 1e-10,
            eps: float = 1e-300,
    ) -> torch.Tensor:
        """
        Smooth weight in [0,1] as a function of PHYSICAL abundance using a quintic
        'smootherstep' in log10-space.

            x <= low  -> weight = 0
            x >= high -> weight = 1
            low < x < high -> smooth transition in log-space

        Args:
            y_phys:  Tensor of physical abundances, shape [..., S] or [..., K, S]
            low:     Lower abundance threshold (<= this gets weight 0)
            high:    Upper abundance threshold (>= this gets weight 1)
            eps:     Small clamp to avoid log10(0)

        Returns:
            weights with the same shape as y_phys
        """
        import math

        # Guard for nonsensical thresholds
        if not (low > 0.0 and high > 0.0 and high > low):
            raise ValueError(f"Invalid thresholds: low={low}, high={high} (require 0<low<high)")

        y = torch.clamp(y_phys, min=eps)
        logy = torch.log10(y)

        lo = math.log10(low)
        hi = math.log10(high)
        # Normalize to [0,1] in log-space, then clamp
        t = (logy - lo) / (hi - lo)
        t = torch.clamp(t, 0.0, 1.0)

        # Quintic smootherstep: 6t^5 - 15t^4 + 10t^3
        w = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        return w

    @torch.inference_mode()
    def evaluate_frac_l1_phys(
        self,
        loader: Optional[torch.utils.data.DataLoader] = None,
        max_batches: Optional[int] = None
    ) -> float:
        """
        Compute true mean fractional absolute error in PHYSICAL units over a loader:
          mean(|y_pred - y_true| / (|y_true| + eps))
        Respects target species subset if configured.
        """
        loader = loader or self.val_loader
        if loader is None:
            return float("nan")

        self.model.eval()
        eps = float(self.cfg.get("training", {}).get("loss_epsilon", 1e-27))
        norm, species_keys = self._ensure_norm_helper()

        total = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)

        # Ensure we have target index mapping like in _process_batch
        if not hasattr(self, "_target_idx"):
            data_cfg = self.cfg.get("data", {})
            species_vars = list(data_cfg.get("species_variables") or [])
            target_vars  = list(data_cfg.get("target_species") or species_vars)
            if target_vars != species_vars:
                name_to_idx = {n: i for i, n in enumerate(species_vars)}
                try:
                    idx_list = [name_to_idx[n] for n in target_vars]
                except KeyError as e:
                    raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")
                self._target_idx = torch.tensor(idx_list, dtype=torch.long, device=self.device)
            else:
                self._target_idx = None

        for step, batch in enumerate(loader, 1):
            if max_batches and step > max_batches:
                break

            # Unpack (supports 5- or 6-tuples)
            if len(batch) == 6:
                y_i, dt_norm, y_j, g, ij, k_mask = batch
            else:
                y_i, dt_norm, y_j, g, ij = batch
                k_mask = None

            # To device
            y_i = y_i.to(self.device, non_blocking=True)
            dt  = dt_norm.to(self.device, non_blocking=True)
            y_j = y_j.to(self.device, non_blocking=True)
            g   = g.to(self.device, non_blocking=True)
            if k_mask is not None:
                k_mask = k_mask.to(self.device, non_blocking=True)

            # Slice targets to match model S_out if needed
            if self._target_idx is not None:
                y_j = y_j.index_select(dim=-1, index=self._target_idx)

            # Forward (mirror training path)
            dt_in = dt.squeeze(-1) if (dt.ndim == 3 and dt.shape[-1] == 1) else dt
            pred = self.model(y_i, dt_in, g)
            pred, tgt = self._harmonize_shapes(pred, y_j)

            # Denormalize to physical units
            y_pred = norm.denormalize(pred, species_keys)
            y_true = norm.denormalize(tgt,  species_keys)

            rel = (y_pred - y_true).abs() / (y_true.abs() + eps)

            if k_mask is not None and rel.ndim == 3 and k_mask.ndim == 2:
                w = k_mask.unsqueeze(-1).expand_as(rel)
                total += (rel * w).sum()
                count += w.sum()
            else:
                total += rel.sum()
                count += rel.numel()

        return (total / count).item() if count.item() > 0 else float("nan")

    def train(self) -> float:
        """
        Execute training loop.

        Returns:
            Best validation loss achieved during training
        """
        start_time = time.perf_counter()

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            # Track current epoch for SIGTERM handler
            self._current_epoch = epoch

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

            # Save checkpoints
            saved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self._save_weights_only()  # Backward compatibility
                self._save_full_checkpoint(self.best_ckpt_path, epoch)
                saved = True

            # Always save last checkpoint for restart
            self._save_full_checkpoint(self.last_ckpt_path, epoch)

            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, val_loss, current_lr)

            # Console output
            epoch_time = time.perf_counter() - epoch_start
            self.logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_loss:.4e} | "
                f"val_loss={val_loss:.4e} | "
                f"lr={current_lr:.2e} | "
                f"time={epoch_time:.1f}s | "
                f"saved={'yes' if saved else 'no'}"
            )

        total_time = time.perf_counter() - start_time
        self.logger.info(
            f"Training completed in {total_time / 3600:.2f} hours. "
            f"Best validation loss: {self.best_val_loss:.4e}"
        )

        # --- Final metric on validation set (true physical fractional L1) ---
        which = "last-epoch"
        try:
            best_blob = torch.load(self.best_model_path, map_location=self.device)
            best_state = best_blob.get("model") if isinstance(best_blob, dict) else None
            if best_state:
                self.model.load_state_dict(best_state, strict=False)
                which = "best_model.pt"
        except Exception:
            pass

        final_frac = self.evaluate_frac_l1_phys(self.val_loader)
        self.logger.info(f"Final validation fractional L1 (physical) [{which}]: {final_frac:.6e}")

        # Optional: persist for downstream scripts
        try:
            import json
            with open(self.work_dir / "final_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"val_frac_l1_phys_mean": final_frac, "which": which}, f, indent=2)
        except Exception:
            pass

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

        # Lazily resolve and cache target indices from cfg (if any)
        if not hasattr(self, "_target_idx"):
            data_cfg = self.cfg.get("data", {})
            species_vars = list(data_cfg.get("species_variables") or [])
            target_vars = list(data_cfg.get("target_species") or species_vars)
            if target_vars != species_vars:
                name_to_idx = {n: i for i, n in enumerate(species_vars)}
                try:
                    idx_list = [name_to_idx[n] for n in target_vars]
                except KeyError as e:
                    raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")
                self._target_idx = torch.tensor(idx_list, dtype=torch.long, device=self.device)
            else:
                self._target_idx = None

        # Slice targets to match model's S_out before shape checks/loss
        if self._target_idx is not None:
            y_j = y_j.index_select(dim=-1, index=self._target_idx)

        # Zero gradients
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        # Forward pass with autocast
        with autocast_context:
            # Squeeze dt_norm from [B,K,1] -> [B,K] when needed
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Check for shape mismatches
        if pred.ndim == 2 and target.ndim == 3:
            B_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}. "
                f"The model should return [B, K, S] when K={K_target} times are requested."
            )

        elif pred.ndim == 3 and target.ndim == 2:
            B_pred, K_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}."
            )

        elif pred.ndim == 3 and target.ndim == 3:
            B_pred, K_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            if K_pred != K_target:
                raise RuntimeError(f"Shape mismatch in time dimension")
            if B_pred != B_target:
                raise RuntimeError("Batch size mismatch")
            if S_pred != S_target:
                raise RuntimeError("State dimension mismatch")

        elif pred.ndim == 2 and target.ndim == 2:
            B_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            if B_pred != B_target:
                raise RuntimeError("Batch size mismatch")
            if S_pred != S_target:
                raise RuntimeError("State dimension mismatch")

        else:
            raise RuntimeError("Unexpected tensor dimensions")

        return pred, target

    def _compute_loss(
            self,
            pred: torch.Tensor,  # [B,S] or [B,K,S] in normalized space
            target: torch.Tensor,  # same shape as pred
            mask: Optional[torch.Tensor]  # None or [B,K] for multi-time batches
    ) -> torch.Tensor:
        """
        Supports two modes (training.loss_mode):
          - "mse": unweighted MSE in normalized space. If a k_mask is provided, uses a masked mean.
          - "custom_mse": MSE in normalized space, but each element is weighted by a smooth
            log10(abundance) weight computed from the *physical* TARGET abundance:
                <= custom_mse_low_phys  -> weight = 0
                >= custom_mse_high_phys -> weight = 1
                in-between              -> quintic smootherstep in log-space.
            The mask (if present) multiplies the weights before reduction.

        Config keys:
          training.loss_mode             : "mse" | "custom_mse" (default "mse")
          training.custom_mse_low_phys   : float, default 1e-25
          training.custom_mse_high_phys  : float, default 1e-10
        """
        tr_cfg = self.cfg.get("training", {}) or {}
        mode = str(tr_cfg.get("loss_mode", "mse")).lower()

        # Squared error in the CURRENT training space (normalized)
        se = (pred - target) ** 2  # [B,S] or [B,K,S]

        if mode == "mse":
            # Plain (masked) mean squared error in normalized space
            if mask is not None:
                if se.ndim == 3 and mask.ndim == 2:
                    m = mask.unsqueeze(-1).to(dtype=se.dtype, device=se.device).expand_as(se)
                else:
                    m = mask.to(dtype=se.dtype, device=se.device)
                denom = m.sum()
                return (se * m).sum() / denom if denom.item() > 0 else se.new_tensor(0.0)
            return se.mean()

        if mode == "custom_mse":
            # Compute per-element weights from PHYSICAL target abundance (smooth in log10-space)
            norm, species_keys = self._ensure_norm_helper()
            y_true_phys = norm.denormalize(target, species_keys)  # same shape as target

            low = float(tr_cfg.get("custom_mse_low_phys", 1e-25))
            high = float(tr_cfg.get("custom_mse_high_phys", 1e-10))

            w = self._logspace_abundance_weight(
                y_true_phys, low=low, high=high
            ).to(dtype=se.dtype, device=se.device)  # [B,S] or [B,K,S]

            # Apply k_mask if present
            if mask is not None:
                if se.ndim == 3 and mask.ndim == 2:
                    w = w * mask.unsqueeze(-1).to(dtype=w.dtype, device=w.device)
                else:
                    w = w * mask.to(dtype=w.dtype, device=w.device)

            denom = w.sum()
            return (se * w).sum() / denom if denom.item() > 0 else se.new_tensor(0.0)

        raise ValueError("training.loss_mode must be 'mse' or 'custom_mse'.")

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