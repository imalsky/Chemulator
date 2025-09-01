#!/usr/bin/env python3
"""
Trainer for flow-map DeepONet using absolute normalized time.

- Expects DataLoader to yield (y_0_norm, t_norm, y_j_norm, g_norm, ij)
- Uses bfloat16 Automatic Mixed Precision on A100 for stability/speed
- Enables TF32 on matmul/convs
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader,
            val_loader,
            work_dir: Path,
            device: torch.device,
            epochs: int = 50,
            use_amp_bf16: bool = True,
            min_lr: float = 1e-6,
            log: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.work_dir = Path(work_dir)
        self.device = device
        self.epochs = int(epochs)
        self.use_amp_bf16 = bool(use_amp_bf16)
        self.min_lr = float(min_lr)
        self.log = log or logging.getLogger("trainer")

        # Set TF32/precision ONCE
        torch.set_float32_matmul_precision("high")

        # Cosine schedule over epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)

        self.criterion = nn.MSELoss(reduction="mean")
        self.best_val = float("inf")

    def train(self) -> float:
        """Train the model (autonomous Δt mode)."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Starting training with Δt (duration) as trunk input")

        for epoch in range(1, self.epochs + 1):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)  # refresh (i,j)

            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.log.info(f"Epoch {epoch:03d} | train={train_loss:.6e} | val={val_loss:.6e}")

            if val_loss < self.best_val:
                self.best_val = float(val_loss)
                self._save_checkpoint(epoch, val_loss)

        return self.best_val

    def _run_epoch(self, loader, train: bool) -> float:
        """Run one epoch of training or validation."""
        self.model.train(mode=train)

        total, n = 0.0, 0
        amp_dtype = torch.bfloat16 if self.use_amp_bf16 else torch.float32

        for y_0, t, y_target, g, _ij in loader:
            # Ensure tensors on device (dataset may already be GPU-resident)
            y_0 = y_0.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp_bf16):
                # Predict y(t) from initial state y_0
                pred = self.model(y_0, g, t)
                loss = self.criterion(pred, y_target)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total += float(loss.detach())
            n += 1

        return total / max(1, n)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        path = self.work_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )
        self.log.info(f"Saved checkpoint: {path}")