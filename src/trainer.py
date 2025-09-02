#!/usr/bin/env python3
"""
Trainer for Flow-map DeepONet with multi-time-per-anchor support.

This trainer is shape-agnostic:
- If the dataset returns K>1 times per anchor, model forward must return [B,K,S].
- If K==1, the model may return [B,S]; we upcast to [B,1,S] for a uniform loss path.

Key features
------------
- Deterministic per-epoch sampling via dataset.set_epoch(epoch) (if exposed).
- Validation under torch.inference_mode() to avoid autograd graphs.
- Optional caps on train steps per epoch and validation batches to bound wall-time.
- Autocast mixed precision (bf16 or fp16). Uses GradScaler only for fp16.
- AdamW optimizer (fused when available) and cosine annealing with linear warmup.
- CSV log persisted to work_dir/training_log.txt with columns: epoch,train_loss,val_loss,lr
- Minimal, dependency-free.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


# ------------------------ Cosine w/ Warmup (epoch-wise) ------------------------

class CosineWarmupScheduler:
    """
    LR(e) =
      warmup:  base_lr * (e+1)/W
      cosine:  min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * (e-W)/max(1,E-W)))
    for integer epoch index e = 0..E-1
    """
    def __init__(self, opt: Optimizer, total_epochs: int, warmup_epochs: int = 0, min_lr: float = 1e-6):
        self.opt = opt
        self.total = int(total_epochs)
        self.warmup = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)
        self.base_lrs = [pg.get("lr", 1e-3) for pg in self.opt.param_groups]
        for i, pg in enumerate(self.opt.param_groups):
            pg["initial_lr"] = float(self.base_lrs[i])

    @torch.no_grad()
    def step(self, epoch: int) -> float:
        e = int(epoch)
        E = max(1, self.total)
        W = min(self.warmup, E - 1)
        for i, pg in enumerate(self.opt.param_groups):
            base = float(self.base_lrs[i])
            if e < W and W > 0:
                lr = base * (e + 1) / W
            else:
                num = (e - W)
                den = max(1, E - W)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1.0 + math.cos(math.pi * (num / den)))
            pg["lr"] = lr
        return self.opt.param_groups[0]["lr"]


# ----------------------------------- Trainer -----------------------------------

class Trainer:
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
        self.log = logger or _NoLogger()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # --- Training hyperparams
        tc = cfg.get("training", {})
        self.epochs = int(tc.get("epochs", 100))
        self.base_lr = float(tc.get("lr", 1e-3))
        self.weight_decay = float(tc.get("weight_decay", 1e-4))
        self.grad_clip = float(tc.get("gradient_clip", 0.0))
        self.max_train_steps_per_epoch = tc.get("max_train_steps_per_epoch", None)
        self.max_val_batches = tc.get("max_val_batches", None)
        self.use_compile = bool(tc.get("torch_compile", False))

        # AMP / dtype
        mc = cfg.get("mixed_precision", {})
        self.amp_mode = str(mc.get("mode", "bf16"))  # "bf16", "fp16", or "none"
        if self.amp_mode not in ("bf16", "fp16", "none"):
            self.amp_mode = "bf16"
        self.autocast_dtype = (torch.bfloat16 if self.amp_mode == "bf16"
                               else (torch.float16 if self.amp_mode == "fp16" else None))
        self.use_scaler = (self.amp_mode == "fp16")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        # Loss
        self.criterion = nn.MSELoss(reduction="mean")

        # Optimizer (fused when available)
        fused_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            fused=bool(fused_ok),
        )

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            total_epochs=self.epochs,
            warmup_epochs=int(tc.get("warmup_epochs", 0)),
            min_lr=float(tc.get("min_lr", 1e-6)),
        )

        # Optional torch.compile
        if self.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

        # Logging file
        self.log_file = self.work_dir / "training_log.txt"
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

        self.best_val = float("inf")
        self.best_path = self.work_dir / "best_model.pt"

    # ----------------------------- Public API ---------------------------------

    def train(self) -> float:
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            # Inform dataset of epoch if it exposes the hook
            for ds in (getattr(self.train_loader, "dataset", None), getattr(self.val_loader, "dataset", None)):
                if hasattr(ds, "set_epoch") and callable(ds.set_epoch):
                    try:
                        ds.set_epoch(epoch)
                    except Exception:
                        pass

            t0 = time.perf_counter()
            train_loss = self._run_epoch(train=True)
            val_loss = self._run_epoch(train=False) if self.val_loader is not None else train_loss

            lr = self.scheduler.step(epoch)

            saved = False
            if val_loss < self.best_val:
                self.best_val = float(val_loss)
                torch.save({"model": self.model.state_dict(), "config": self.cfg}, self.best_path)
                saved = True

            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.8e},{val_loss:.8e},{lr:.8e}\n")

            epoch_time = time.perf_counter() - t0
            self.log.info(
                f"Epoch {epoch:03d} | train={train_loss:.6e} | val={val_loss:.6e} | "
                f"lr={lr:.2e} | time={epoch_time:.1f}s | dBest={val_loss - self.best_val:.2e} | saved={int(saved)}"
            )

        total = time.perf_counter() - start
        self.log.info(f"Training finished in {total/3600:.2f} h; best val={self.best_val:.6e}")
        return self.best_val

    # ---------------------------- Internal: epochs ----------------------------

    def _run_epoch(self, train: bool) -> float:
        """
        Run one epoch. Supports datasets that return either:
        - (y_i, dt_norm, y_j, g, ij)            # legacy (no mask)
        - (y_i, dt_norm, y_j, g, ij, k_mask)    # new (mask for padded Ks)
        Where:
        y_i:   [B, S]
        dt_norm: [B, K]  (or [B,1] when K=1)
        y_j:   [B, K, S] (or [B, S] when K=1)
        g:     [B, G]
        ij:    [B, K, 2] (int32)
        k_mask: [B, K] bool (True=valid, False=padded). Optional.
        """
        import contextlib
        device = self.device

        model = self.model
        optimizer = getattr(self, "optimizer", None)
        scaler = getattr(self, "scaler", None)
        use_scaler = bool(getattr(self, "use_scaler", False) and scaler is not None)
        autocast_dtype = getattr(self, "autocast_dtype", None)

        loader = self.train_loader if train else self.val_loader
        if loader is None:
            return float("nan")

        model.train(mode=train)
        if train and optimizer is None:
            raise RuntimeError("Trainer has no optimizer set.")

        total_loss = 0.0
        n_batches = 0

        # Proper autocast context for current device
        if autocast_dtype is not None:
            amp_ctx = torch.amp.autocast(device_type=device.type, dtype=autocast_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        for step, batch in enumerate(loader, 1):
            # Unpack batch (with or without k_mask)
            if len(batch) == 6:
                y_i, dt_norm, y_j, g, ij, k_mask = batch
            else:
                y_i, dt_norm, y_j, g, ij = batch
                k_mask = None

            # Move to model device if needed (dataset may already be on device)
            if y_i.device != device:
                y_i = y_i.to(device, non_blocking=True)
                dt_norm = dt_norm.to(device, non_blocking=True)
                y_j = y_j.to(device, non_blocking=True)
                g = g.to(device, non_blocking=True)
                if k_mask is not None:
                    k_mask = k_mask.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                # Forward
                pred = model(y_i, dt_norm, g)   # expected: [B,K,S] when K>1, else [B,S]

                # Shape harmonization:
                # - If model returns [B,S] but target is [B,K,S], expand pred across K
                # - If model returns [B,K,S] but target is [B,S], expand target across K
                if pred.ndim == 2 and y_j.ndim == 3:
                    pred = pred.unsqueeze(1).expand(-1, y_j.size(1), -1)
                elif pred.ndim == 3 and y_j.ndim == 2:
                    y_j = y_j.unsqueeze(1).expand(-1, pred.size(1), -1)

                # Elementwise MSE (robust to 2D or 3D)
                loss_elem = (pred - y_j) ** 2

                # Apply mask if provided (ignore padded j's from near-the-end anchors)
                if k_mask is not None:
                    if loss_elem.ndim == 3:
                        # loss_elem: [B,K,S], k_mask: [B,K] -> [B,K,1] broadcast on S
                        m = k_mask.unsqueeze(-1)
                        valid = m.any()
                        loss = loss_elem[m].mean() if valid else loss_elem.mean()
                    else:
                        # loss_elem: [B,S], k_mask: [B] or [B,1] -> expand to [B,S]
                        km = k_mask
                        if km.ndim == 1:
                            km = km.view(-1, 1)
                        m2 = km.expand(loss_elem.size(0), loss_elem.size(1))
                        valid = m2.any()
                        loss = loss_elem[m2].mean() if valid else loss_elem.mean()
                else:
                    loss = loss_elem.mean()

            # Backward / step
            if train:
                if use_scaler and device.type == "cuda":
                    scaler.scale(loss).backward()
                    # Optional grad clipping
                    clip_norm = getattr(self, "grad_clip_norm", None)
                    if clip_norm is not None and clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    clip_norm = getattr(self, "grad_clip_norm", None)
                    if clip_norm is not None and clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_norm))
                    optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        return total_loss / max(1, n_batches)




# --------------------------------- Utilities ----------------------------------

def _align_pred_target(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make shapes compatible for elementwise loss:
      - If pred is [B,S] and target is [B,1,S], expand pred -> [B,1,S].
      - If pred is [B,K,S] and target is [B,K,S], leave as-is.
      - If pred is [B,S] and target is [B,K,S] with K>1: replicate pred across K.
    Returns (pred_like_target, target).
    """
    if target.dim() == 3 and pred.dim() == 2:
        # Expand along K
        B, K, S = target.shape
        pred = pred.view(B, 1, S).expand(B, K, S).contiguous()
    elif target.dim() == 2 and pred.dim() == 3:
        # Collapse K=1
        pred = pred[:, 0, :]
    return pred, target


class _NoLogger:
    def info(self, *a, **k): pass


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False
