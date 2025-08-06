#!/usr/bin/env python3
"""
Unified training module for chemical kinetics models.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from data.normalizer import NormalizationHelper
from lion_pytorch import Lion
import math


class Trainer:
    """Unified trainer for chemical kinetics models."""
    def __init__(self, model: nn.Module, train_dataset, val_dataset, test_dataset,
                 config: Dict[str, Any], save_dir: Path, device: torch.device,
                 norm_helper: Optional[NormalizationHelper] = None):
        self.logger = logging.getLogger(__name__)
        
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.device = device
        self.norm_helper = norm_helper
        
        # Extract config sections
        self.train_config = config["training"]
        self.system_config = config["system"]
        self.data_config = config["data"]
        
        # Check sequence mode
        self.sequence_mode = self.data_config.get("sequence_mode", False)
        
        # Get dtype
        dtype_str = self.system_config.get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)
        
        # Dataset info
        self.has_validation = val_dataset is not None and len(val_dataset) > 0
        
        # Create data loaders
        self._setup_dataloaders(train_dataset, val_dataset, test_dataset)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.total_training_time = 0
        self.patience_counter = 0
        
        # Training parameters
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.min_delta = self.train_config["min_delta"]
        self.gradient_accumulation_steps = self.train_config["gradient_accumulation_steps"]
        
        # Setup components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        
        # Initialize CSV logger
        self._init_csv_logger()
        
        # History for backward compatibility
        self.training_history = {
            "config": config,
            "epochs": []
        }
    
    def _init_csv_logger(self):
        """Initialize CSV file for logging training metrics."""
        self.csv_path = self.save_dir / "training_metrics.csv"
        self.csv_headers = [
            "epoch", "timestamp", "train_loss", "val_loss", 
            "learning_rate", "epoch_time", "total_time", "best_val_loss"
        ]
        
        # Write headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def _log_to_csv(self, metrics: Dict[str, Any]):
        """Append metrics to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [metrics.get(h, "") for h in self.csv_headers]
            writer.writerow(row)
    
    def _setup_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Setup data loaders."""
        from data.dataset import create_dataloader
        
        self.train_loader = create_dataloader(
            train_dataset, self.config, shuffle=True, 
            device=self.device, drop_last=False
        ) if train_dataset else None
        
        self.val_loader = create_dataloader(
            val_dataset, self.config, shuffle=False,
            device=self.device, drop_last=False
        ) if val_dataset and len(val_dataset) > 0 else None
        
        self.test_loader = create_dataloader(
            test_dataset, self.config, shuffle=False,
            device=self.device, drop_last=False
        ) if test_dataset and len(test_dataset) > 0 else None
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay handling."""
        opt_name = self.train_config.get("optimizer", "adamw").lower()
        
        # Split parameters by weight decay
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or "bias" in name or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        
        param_groups = [
            {"params": decay, "weight_decay": self.train_config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        
        lr = self.train_config["learning_rate"]
        betas = tuple(self.train_config.get("betas", [0.9, 0.999]))
        eps = self.train_config.get("eps", 1e-8)
        
        if opt_name == "lion":
            self.optimizer = Lion(param_groups, lr=lr, betas=betas,
                                weight_decay=self.train_config["weight_decay"])
            self.logger.info(f"Using Lion optimizer (lr={lr}, betas={betas})")
        elif opt_name == "adamw":
            opt_kwargs = {"lr": lr, "betas": betas, "eps": eps}
            
            # Fixed: Try fused AdamW with actual param_groups
            if self.device.type == "cuda":
                try:
                    # Test with actual parameters to catch edge cases
                    test_optimizer = AdamW(param_groups, fused=True, **opt_kwargs)
                    del test_optimizer  # Clean up test
                    opt_kwargs["fused"] = True
                    self.logger.info("Using fused AdamW")
                except Exception as e:
                    self.logger.debug(f"Fused AdamW not available: {e}")
            
            self.optimizer = AdamW(param_groups, **opt_kwargs)
            self.logger.info(f"Using AdamW optimizer (lr={lr}, betas={betas}, eps={eps})")
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.train_config.get("scheduler", "none").lower()
        
        if scheduler_type == "none" or not self.train_loader:
            self.scheduler = None
            self.scheduler_step_on_batch = False
            return
        
        steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        )
        
        params = self.train_config.get("scheduler_params", {})
        
        if scheduler_type == "cosine":
            # Fixed: Document that T_0 is in epochs when specified, convert to steps
            T_0_epochs = params.get("T_0", 10)
            T_0_steps = T_0_epochs * steps_per_epoch
            
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0_steps,  # Now correctly in steps
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-8)
            )
            self.scheduler_step_on_batch = True
            self.logger.info(f"Cosine scheduler: T_0={T_0_epochs} epochs ({T_0_steps} steps)")
            
        elif scheduler_type == "plateau":
            if not self.has_validation:
                self.logger.warning("Plateau scheduler requires validation data")
                self.scheduler = None
                self.scheduler_step_on_batch = False
                return
            
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
                min_lr=params.get("min_lr", 1e-7)
            )
            self.scheduler_step_on_batch = False
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def _setup_loss(self):
        """Setup loss function."""
        self.criterion = nn.MSELoss()
    
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.dtype == torch.float64:
            self.use_amp = False
            self.scaler = GradScaler(enabled=False)
            self.amp_dtype = None
            return
        
        self.use_amp = self.train_config.get("use_amp", True)
        dtype_str = str(self.train_config.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        
        # GradScaler only for float16
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                     inputs: torch.Tensor) -> torch.Tensor:
        """Compute loss with regularization."""
        # MSE loss
        loss = self.criterion(outputs, targets)
        
        # Add regularization losses if available
        if hasattr(self.model, 'get_regularization_losses'):
            reg_losses = self.model.get_regularization_losses(inputs)
            
            # Gate entropy
            if 'entropy_loss' in reg_losses:
                lambda_entropy = self.train_config.get("regularization", {}).get("lambda_entropy", 0.01)
                if lambda_entropy > 0:
                    loss = loss + lambda_entropy * reg_losses['entropy_loss']
            
            # Generator diversity
            if 'generator_diversity' in reg_losses:
                lambda_diversity = self.train_config.get("regularization", {}).get("lambda_diversity", 0.01)
                if lambda_diversity > 0:
                    loss = loss + lambda_diversity * reg_losses['generator_diversity']
        
        return loss
    
    def train(self) -> float:
        """Execute training loop."""
        if not self.train_loader:
            self.logger.error("No training data available")
            return float("inf")
        
        self.logger.info(f"Starting training in {'sequence' if self.sequence_mode else 'row-wise'} mode")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.has_validation:
            self.logger.info(f"Val batches: {len(self.val_loader)}")
        
        try:
            self._run_training_loop()
            self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f} "
                           f"at epoch {self.best_epoch}")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            # Save training history
            save_path = self.save_dir / "training_log.json"
            with open(save_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            self.logger.info(f"Training metrics saved to {self.csv_path}")
        
        return self.best_val_loss
    
    def _run_training_loop(self):
        """Main training loop."""
        best_train_loss = float("inf")

        # Mixture temperature schedule (optional)
        mix_sched = self.train_config.get("mixture_temperature_schedule", {})
        t_start = float(mix_sched.get("start", 1.0))
        t_end = float(mix_sched.get("end", 0.3))
        t_anneal_frac = float(mix_sched.get("anneal_frac", 0.6))

        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Update gate temperature if model supports it
            if hasattr(self.model, "set_gate_temperature") and self.model.K > 1:
                progress = min(1.0, (epoch - 1) / max(1, int(self.train_config["epochs"] * t_anneal_frac)))
                temp = t_start + (t_end - t_start) * progress
                self.model.set_gate_temperature(temp)

            # Train
            train_loss, train_metrics = self._train_epoch()
            
            # Validate
            val_loss, val_metrics = self._validate()
            
            # Update scheduler
            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Save best model
            if self.has_validation:
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if train_loss < (best_train_loss - self.min_delta):
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                    self.best_epoch = epoch
                    self._save_best_model()
        
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with correct gradient-accumulation scaling."""
        self.model.train()
        total_loss    = 0.0
        total_samples = 0

        k         = self.gradient_accumulation_steps
        n_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move to device if necessary
            if inputs.device != self.device:
                inputs   = inputs.to(self.device, non_blocking=True)
                targets  = targets.to(self.device, non_blocking=True)

            # Forward pass
            with autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype = self.amp_dtype,
            ):
                outputs   = self.model(inputs)
                loss_raw  = self._compute_loss(outputs, targets, inputs)

            # Backward pass on the unscaled loss. We will average the gradients later.
            if self.scaler.is_enabled():
                self.scaler.scale(loss_raw).backward()
            else:
                loss_raw.backward()

            # Check if it's time to step the optimizer
            is_last_batch = (batch_idx + 1) == n_batches
            window_full   = ((batch_idx + 1) % k == 0)

            if window_full or is_last_batch:
                # --- CORRECTNESS FIX: Manual Gradient Averaging ---
                # Determine the number of micro-batches in this step's window
                if is_last_batch and n_batches % k != 0:
                    num_micro_batches = n_batches % k
                else:
                    num_micro_batches = k
                
                # Unscale gradients before we manipulate them
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                
                # Manually average the gradients across the window
                if num_micro_batches > 1:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.div_(num_micro_batches)
                # --------------------------------------------------

                # Optional grad-clip (applied to the now-averaged gradients)
                clip_val = self.train_config.get("gradient_clip", 0.0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

                # Optimizer step
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler and self.scheduler_step_on_batch:
                    self.scheduler.step()

                self.global_step += 1

            # Accumulate statistics using the raw, unscaled loss
            batch_size   = inputs.size(0)
            total_loss  += loss_raw.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, {}

    
    @torch.inference_mode()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Validation epoch."""
        if not self.has_validation or self.val_loader is None:
            return float("inf"), {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in self.val_loader:
            if inputs.device != self.device:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs)
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        return avg_loss, {}
    
    @torch.inference_mode()
    def evaluate_test(self) -> float:
        """Evaluate on test set."""
        if not self.test_loader:
            self.logger.warning("No test data available")
            return float("inf")
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in self.test_loader:
            if inputs.device != self.device:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs)
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        self.logger.info(f"Test loss: {avg_loss:.6f}")
        return avg_loss
    
    def _log_epoch(self, train_loss, val_loss, train_metrics, val_metrics, epoch_time):
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Log to memory (for backward compatibility)
        log_entry = {
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            "lr": lr
        }
        self.training_history["epochs"].append(log_entry)
        
        # Log to CSV
        csv_entry = {
            "epoch": self.current_epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": f"{train_loss:.6e}",
            "val_loss": f"{val_loss:.6e}" if self.has_validation else "",
            "learning_rate": f"{lr:.6e}",
            "epoch_time": f"{epoch_time:.2f}",
            "total_time": f"{self.total_training_time:.2f}",
            "best_val_loss": f"{self.best_val_loss:.6e}"
        }
        self._log_to_csv(csv_entry)
        
        # Console logging
        val_str = f"Val: {val_loss:.3e}" if self.has_validation else "Val: N/A"
        self.logger.info(
            f"Epoch {self.current_epoch}/{self.train_config['epochs']} | "
            f"Train: {train_loss:.3e} | {val_str} | "
            f"Time: {epoch_time:.1f}s | LR: {lr:.2e}"
        )
        
    def _save_best_model(self):
        """Save best model checkpoint and optionally export."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        # Export if requested
        if self.system_config.get("use_torch_export", False):
            exported_path = self.save_dir / "best_model_exported.pt"
            success = self._export_from_live_model(exported_path)
            if not success:
                self.logger.warning("Model export failed - continuing with training")

    @torch.inference_mode()
    def _export_from_live_model(self, output_path: Path) -> bool:
        """
        Export the current model with torch.export.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger.info("Exporting model...")

            # Unwrap compiled/DDP wrappers
            model_to_export = self.model
            if hasattr(model_to_export, "_orig_mod"):
                model_to_export = model_to_export._orig_mod
            if hasattr(model_to_export, "module"):
                model_to_export = model_to_export.module

            was_training = self.model.training
            model_to_export.eval()

            # Build example input
            n_species = len(self.data_config["species_variables"])
            n_globals = len(self.data_config["global_variables"])
            M = self.data_config["M_per_sample"]
            input_dim = n_species + n_globals + M

            # Use model's device/dtype
            p = next((p for p in model_to_export.parameters() if p.requires_grad), None)
            dtype = p.dtype if p is not None else torch.float32
            device = p.device if p is not None else torch.device("cpu")

            example_input = torch.randn(1, input_dim, dtype=dtype, device=device)

            # Fixed: Support both positional and keyword arguments
            # Try export with positional args first
            try:
                exported = torch.export.export(model_to_export, (example_input,))
            except Exception as e:
                # If positional fails, try with kwargs (future-proofing)
                self.logger.debug(f"Positional export failed: {e}, trying kwargs")
                exported = torch.export.export(
                    model_to_export, 
                    (), 
                    kwargs={"inputs": example_input}
                )
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(exported, output_path)
            self.logger.info(f"Model exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)
            return False
        finally:
            if 'was_training' in locals() and was_training:
                self.model.train()