#!/usr/bin/env python3
"""
train.py - Simplified training pipeline for streaming data.

This module manages the training process for chemical kinetics prediction models
using streaming datasets to handle large-scale data efficiently.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataset
from hardware import configure_dataloader_settings
from model import create_prediction_model, export_model_jit
from normalizer import DataNormalizer
from utils import save_json

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Simplified training pipeline manager for streaming datasets.
    
    Features:
    - Streaming data loading with multi-worker support
    - Automatic mixed precision (AMP) for GPU training
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling
    - Early stopping
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        h5_path: Path,
        splits: Dict[str, List[int]],
        collate_fn: Callable,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device for training
            save_dir: Directory to save outputs
            h5_path: Path to HDF5 dataset
            splits: Dictionary with train/val/test indices
            collate_fn: Function to collate batches
        """
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.h5_path = h5_path

        # Extract config sections
        self.data_spec = self.cfg["data_specification"]
        self.train_params = self.cfg["training_hyperparameters"]
        self.misc_cfg = self.cfg["miscellaneous_settings"]
        self.num_constants = self.cfg.get("numerical_constants", {})
        self.species_vars = sorted(self.data_spec["species_variables"])

        # Get constants from config with defaults
        self.default_batch_size = 1024
        self.default_epochs = 100
        self.default_lr = 1e-4
        self.default_grad_clip = 1.0
        self.default_early_stopping_patience = 20
        self.default_min_delta = 1e-10
        self.default_gradient_accumulation = 1
        self.default_max_invalid_batches = 100
        self.default_invalid_batch_threshold = 0.5
        self.default_num_workers = 4

        # Training stability parameters
        self.max_invalid_batches = self.train_params.get("max_invalid_batches", self.default_max_invalid_batches)
        self.invalid_batch_threshold = self.train_params.get("invalid_batch_threshold", self.default_invalid_batch_threshold)
        self.max_steps_per_epoch = self.cfg["training_hyperparameters"].get("max_steps_per_epoch", None)
        self.log_gradient_norms = self.misc_cfg.get("log_gradient_norms", True)
        
        # Enable anomaly detection for debugging if requested
        if self.misc_cfg.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - this will slow down training!")

        # Setup components
        self._setup_normalization_and_datasets(h5_path, splits)
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimizer()
        self._build_schedulers()
        self._setup_loss_and_training_params()
        self._setup_logging()
        self._save_metadata()
        
        # Force garbage collection after setup to free memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _setup_normalization_and_datasets(
        self, h5_path: Path, splits: Dict[str, List[int]]
    ) -> None:
        """Calculate normalization statistics and create streaming datasets."""
        frac = float(self.train_params.get("frac_of_data", 1.0))
        seed = self.cfg["miscellaneous_settings"].get("random_seed", self.num_constants.get("default_seed", 42))
        rng = torch.Generator().manual_seed(seed)

        def _take_fraction(idx):
            """Return a random subset of indices respecting frac_of_data."""
            if frac >= 1.0 or frac <= 0.0:      # keep entire split
                return idx

            k = max(1, int(len(idx) * frac))
            selected = torch.randperm(len(idx), generator=rng)[:k]
            logger.info(f"Using {k}/{len(idx)} (~{100*frac:.1f} %) examples")
            return [idx[i] for i in selected.tolist()]

        train_indices = _take_fraction(splits["train"])
        val_indices   = _take_fraction(splits["validation"])
        test_indices  = _take_fraction(splits["test"])
        
        self.test_set_indices = test_indices

        # Calculate normalization statistics
        logger.info("Calculating normalization statistics from training set...")
        normalizer = DataNormalizer(config_data=self.cfg)
        # Note: We don't use the raw_train_data since we're streaming
        self.norm_metadata, _ = normalizer.calculate_stats(h5_path, train_indices)
        save_json(self.norm_metadata, self.save_dir / "normalization_metadata.json")

        # Create streaming datasets
        profiles_per_chunk = self.misc_cfg.get("profiles_per_chunk", self.num_constants.get("default_chunk_size", 2048))
        
        ds_kwargs = {
            "h5_path": h5_path,
            "species_variables": self.data_spec["species_variables"],
            "global_variables": self.data_spec["global_variables"],
            "normalization_metadata": self.norm_metadata,
            "profiles_per_chunk": profiles_per_chunk,
            "config": self.cfg,
        }
        
        self.train_ds = create_dataset(indices=train_indices, **ds_kwargs)
        self.val_ds = create_dataset(indices=val_indices, **ds_kwargs)
        self.test_ds = create_dataset(indices=test_indices, **ds_kwargs)

        logger.info(
            f"Datasets created - Train: ~{self.train_ds.total_samples:,}, "
            f"Val: ~{self.val_ds.total_samples:,}, Test: ~{self.test_ds.total_samples:,}"
        )

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Create DataLoaders for streaming data."""
        num_workers = self.misc_cfg.get("num_dataloader_workers", self.default_num_workers)
        hw_settings = configure_dataloader_settings()
        batch_size = self.train_params.get("batch_size", self.default_batch_size)
        
        # DataLoader configuration
        dl_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": hw_settings.get("pin_memory", False) and self.device.type == "cuda",
            "persistent_workers": hw_settings.get("persistent_workers", False) and num_workers > 0,
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 4        

        # Use the same settings for val/test, but drop prefetch_factor if workers==0
        train_kwargs = dl_kwargs.copy()
        val_kwargs   = dl_kwargs.copy()
        test_kwargs  = dl_kwargs.copy()

        if self.misc_cfg.get("use_torch_compile", False):
            train_kwargs["drop_last"] = True
            val_kwargs["drop_last"]   = True
            test_kwargs["drop_last"]  = True

        if num_workers == 0:
            val_kwargs.pop("prefetch_factor", None)
            test_kwargs.pop("prefetch_factor", None)

        self.train_loader = DataLoader(self.train_ds, **train_kwargs)
        self.val_loader   = DataLoader(self.val_ds,   **val_kwargs)
        self.test_loader  = DataLoader(self.test_ds,  **test_kwargs)
        
        logger.info(
            f"DataLoaders created with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={dl_kwargs['pin_memory']}"
        )

    def _build_model(self) -> None:
        """Create and optionally compile the model."""
        self.model = create_prediction_model(self.cfg, device=self.device)
        
        # Try to compile model if requested and available
        use_compile = self.misc_cfg.get("use_torch_compile", False)
        if use_compile and hasattr(torch, 'compile'):
            try:
                compile_mode = self.misc_cfg.get("torch_compile_mode", "reduce-overhead")
                compile_options = {
                    "mode": compile_mode,
                    "backend": "inductor",
                    "fullgraph": False,
                }
                self.model = torch.compile(self.model, **compile_options)
                logger.info(f"Model compiled with torch.compile(mode='{compile_mode}')")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    def _build_optimizer(self) -> None:
        """Create AdamW optimizer."""
        lr = self.train_params.get("learning_rate", self.default_lr)
        weight_decay = self.train_params.get("weight_decay", 1e-5)
        
        # Separate parameters that should and shouldn't have weight decay
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and normalization parameters
            if param.dim() == 1 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        self.optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        logger.info(f"Optimizer: AdamW with lr={lr:.2e}, weight_decay={weight_decay:.2e}")

    def _build_schedulers(self) -> None:
        """Create and configure the learning-rate scheduler."""
        # ── compute effective steps per epoch --------------------------------
        total_samples = self.train_ds.total_samples
        batch_size = self.train_params.get("batch_size", self.default_batch_size)
        self.gradient_accumulation = self.train_params.get(
            "gradient_accumulation_steps", self.default_gradient_accumulation
        )

        self.steps_per_epoch = max(1, total_samples // (batch_size * self.gradient_accumulation))
        if self.max_steps_per_epoch is not None:
            self.steps_per_epoch = min(self.steps_per_epoch, self.max_steps_per_epoch)
        logger.info(f"Steps per epoch: ~{self.steps_per_epoch}")

        # ── choose scheduler --------------------------------------------------
        scheduler_name = self.train_params.get("scheduler_choice", "plateau").lower()

        if scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_params.get("factor", 0.5),
                patience=self.train_params.get("patience", 10),
                min_lr=self.train_params.get("min_lr", 1e-7),
            )
            self.scheduler_updates_on_epoch = True

        elif scheduler_name == "cosine":
            #  -- first restart after `cosine_T_0` epochs  ---------------------
            t0_epochs = int(self.train_params.get("cosine_T_0", 10))
            t0_steps  = t0_epochs * int(self.steps_per_epoch)

            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0_steps,
                T_mult=self.train_params.get("cosine_T_mult", 2),
                eta_min=self.train_params.get("min_lr", 1e-7),
            )
            self.scheduler_updates_on_epoch = False
        else:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. Use 'plateau' or 'cosine'.")

    def _setup_loss_and_training_params(self) -> None:
        """Setup loss function and training parameters."""
        # Create loss function
        loss_name = self.train_params.get("loss_function", "mse").lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss(delta=self.train_params.get("huber_delta", 1.0))
        else:
            raise ValueError(f"Unsupported loss function '{loss_name}'. Use 'mse' or 'huber'.")
        
        # Setup automatic mixed precision
        self.use_amp = self.train_params.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled")
        
        # Gradient clipping value
        self.max_grad_norm = self.train_params.get("gradient_clip_val", self.default_grad_clip)

    def _setup_logging(self) -> None:
        """Setup training logs and metrics tracking."""
        self.log_path = self.save_dir / "training_log.csv"
        headers = ["epoch", "train_loss", "val_loss", "lr", "grad_norm", "time_s", "improvement"]
        self.log_path.write_text(",".join(headers) + "\n")
        self.best_val_loss, self.best_epoch = float("inf"), -1
        self.global_step, self.total_invalid_batches = 0, 0

    def _save_metadata(self) -> None:
        """Save training metadata."""
        metadata = {
            "test_set_indices": sorted(self.test_set_indices),
            "num_train_samples": self.train_ds.total_samples,
            "effective_batch_size": self.train_params.get("batch_size", self.default_batch_size) * self.gradient_accumulation,
            "device": str(self.device), 
            "model_type": self.cfg["model_hyperparameters"].get("model_type", "siren"),
        }
        save_json(metadata, self.save_dir / "training_metadata.json")

    def train(self) -> float:
        """
        Execute the main training loop.
        
        Returns:
            Best validation loss achieved during training
        """
        epochs = self.train_params.get("epochs", self.default_epochs)
        patience = self.train_params.get("early_stopping_patience", self.default_early_stopping_patience)
        min_delta = self.train_params.get("min_delta", self.default_min_delta)
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for up to {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self.total_invalid_batches = 0
            start_time = time.time()
            
            # Training phase
            train_loss, train_grad_norm = self._run_epoch(self.train_loader, is_train_phase=True)
            if train_loss is None: 
                logger.error("Catastrophic failure during training. Stopping.")
                break

            # Validation phase
            val_loss, _ = self._run_epoch(self.val_loader, is_train_phase=False)
            if val_loss is None: 
                val_loss = float('inf')

            # Update learning rate scheduler
            if self.scheduler_updates_on_epoch:
                self.scheduler.step(val_loss)
            
            # Log results
            improvement = self.best_val_loss - val_loss
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_grad_norm, 
                time.time() - start_time, improvement
            )
            
            # Check for improvement and save best model
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                self._save_best_model(epoch, val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        
        # Final evaluation
        self.test()
        
        logger.info(
            f"Training completed. Best validation loss: {self.best_val_loss:.4e} "
            f"at epoch {self.best_epoch}."
        )
        return self.best_val_loss

    def _run_epoch(
        self, loader: DataLoader, is_train_phase: bool
    ) -> Tuple[Optional[float], Optional[float]]:
        """Run one epoch of training or validation."""
        self.model.train(is_train_phase)
        total_loss, total_grad_norm, num_batches, num_opt_steps = 0.0, 0.0, 0, 0

        desc = f"Epoch {self.current_epoch:03d} {'Train' if is_train_phase else 'Val'}"
        progress_bar = tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not self.misc_cfg.get("show_epoch_progress", True),
        )

        # CUDAGraph marker (safe no-op on CPU)
        if is_train_phase and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        for batch_idx, batch in enumerate(progress_bar):

            # ── handle an all-invalid batch ────────────────────────────────────
            if batch is None:
                logger.warning(
                    f"All-invalid batch encountered at idx {batch_idx}; skipping."
                )
                continue

            if (
                self.max_steps_per_epoch is not None
                and batch_idx + 1 > self.max_steps_per_epoch
            ):
                break

            try:
                inputs_dict, targets = batch
                inputs = inputs_dict["x"].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.set_grad_enabled(is_train_phase):
                    with torch.amp.autocast(
                        device_type=self.device.type, enabled=self.use_amp
                    ):
                        preds = self.model(inputs)
                        loss = self.criterion(preds, targets)

                    # loss sanity-check
                    if not torch.isfinite(loss):
                        logger.warning(
                            f"Non-finite loss in batch {batch_idx}; skipping."
                        )
                        self.total_invalid_batches += 1
                        if self.total_invalid_batches > self.max_invalid_batches:
                            raise RuntimeError(
                                "Exceeded maximum number of invalid batches."
                            )
                        continue

                # ── backward / optimiser step ─────────────────────────────────
                if is_train_phase:
                    self.scaler.scale(loss / self.gradient_accumulation).backward()

                    if (batch_idx + 1) % self.gradient_accumulation == 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                        if not torch.isfinite(grad_norm):
                            logger.error("Non-finite gradient norm; skipping optimiser step.")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                        if not self.scheduler_updates_on_epoch:
                            self.scheduler.step()

                        self.global_step += 1
                        num_opt_steps += 1
                        total_grad_norm += (
                            grad_norm.item() if grad_norm.numel() > 0 else 0.0
                        )

                        progress_bar.set_postfix(
                            loss=f"{loss.item():.4e}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        )

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}", exc_info=True)
                continue

        # ── flush tail gradients if we broke the loop early ───────────────────
        if (
            is_train_phase
            and (num_batches % self.gradient_accumulation) != 0
            and num_batches > 0
        ):
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if torch.isfinite(grad_norm):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                num_opt_steps += 1
                total_grad_norm += grad_norm.item()
                if not self.scheduler_updates_on_epoch:
                    self.scheduler.step()        
                self.global_step += 1

        progress_bar.close()

        # ── abort if too many invalid batches ──────────────────────────────
        invalid_ratio = (
            self.total_invalid_batches / (num_batches + self.total_invalid_batches)
            if (num_batches + self.total_invalid_batches) > 0
            else 0.0
        )
        if invalid_ratio > self.invalid_batch_threshold:
            logger.error(
                f"Invalid-batch ratio {invalid_ratio:.2%} exceeded "
                f"threshold ({self.invalid_batch_threshold:.2%}); aborting epoch."
            )
            return None, None

        # ── epoch summary ─────────────────────────────────────────────────────
        if num_batches == 0:
            logger.error(
                f"No valid batches processed in "
                f"{'training' if is_train_phase else 'validation'} epoch."
            )
            return None, None

        avg_loss = total_loss / num_batches
        avg_grad_norm = (
            total_grad_norm / num_opt_steps if num_opt_steps > 0 else 0.0
        )

        return avg_loss, avg_grad_norm

    def _log_epoch_results(
        self, epoch: int, train_loss: float, val_loss: float, 
        grad_norm: float, duration: float, improvement: float
    ) -> None:
        """Log epoch results to console and CSV file."""
        lr = self.optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
            f"Val Loss: {val_loss:.4e} | LR: {lr:.2e} | "
            f"Grad: {grad_norm:.2f} | Time: {duration:.1f}s"
        )
        if improvement > 0: 
            log_msg += f" | ↓ {improvement:.4e}"
        logger.info(log_msg)
        
        # Write to CSV
        with self.log_path.open("a") as f:
            f.write(f"{epoch},{train_loss},{val_loss},{lr},{grad_norm},{duration},{improvement}\n")

    def _save_best_model(self, epoch: int, val_loss: float) -> None:
        """Save the best model state and a matching JIT version."""
        # ── 1. Serialize state_dict -------------------------------------------------
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        ckpt_path     = self.save_dir / "best_model.pt"
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "epoch":      epoch,
                "val_loss":   val_loss,
                "config":     self.cfg,
                "normalization_metadata": self.norm_metadata,
            },
            ckpt_path,
        )
        logger.debug(f"Saved best_model.pt from epoch {epoch}")
        
        if not self.misc_cfg.get("save_jit_model", False):
            return
        # ── 2. Trace & save TorchScript --------------------------------------------
        try:
            # recreate a fresh model on CPU to avoid mixed-precision oddities
            export_model = create_prediction_model(self.cfg, device=torch.device("cpu"))
            export_model.load_state_dict(model_to_save.state_dict())
            export_model.eval()

            # build a  single example input  from the current batch's shapes
            with torch.no_grad():
                dummy_input = torch.zeros_like(next(iter(self.val_loader))[0]["x"][:1]).to("cpu")

            jit_path = self.save_dir / "best_model_jit.pt"
            export_model_jit(export_model, dummy_input, jit_path, optimize=True)
            logger.debug(f"Saved best_model_jit.pt from epoch {epoch}")

        except Exception as e:
            logger.error(f"JIT export failed after epoch {epoch}: {e}", exc_info=True)

    def test(self) -> None:
        """Evaluate the best model on the test set."""
        model_path = self.save_dir / "best_model.pt"
        if not model_path.exists():
            logger.warning("No best_model.pt found, skipping test evaluation.")
            return
        
        logger.info("Loading best model for test evaluation...")
        state = torch.load(model_path, map_location=self.device)
        model_state = state["state_dict"]
        
        # Handle compiled models
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        
        # Run test evaluation
        test_loss, _ = self._run_epoch(self.test_loader, is_train_phase=False)
        if test_loss is not None:
            metrics = {
                "test_loss": test_loss, 
                "best_epoch": state['epoch'], 
                "best_val_loss": state['val_loss']
            }
            logger.info(f"Test Loss: {test_loss:.4e}")
            save_json(metrics, self.save_dir / "test_metrics.json")


__all__ = ["ModelTrainer"]