#!/usr/bin/env python3
"""
Training pipeline for chemical kinetics models.
Fixed issues:
1. Correct scheduler stepping with gradient accumulation
2. Removed all MAE calculations
3. Fixed ratio mode loss calculation
4. Fixed no-validation logic
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import math

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from models.model import export_model
from data.normalizer import NormalizationHelper


class Trainer:
    """Trainer for chemical kinetics networks."""
    def __init__(self, model: nn.Module, train_dataset, val_dataset, test_dataset,
                config: Dict[str, Any], save_dir: Path, device: torch.device,
                norm_helper: NormalizationHelper):
        self.logger = logging.getLogger(__name__)
        
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.device = device
        self.norm_helper = norm_helper
        
        # Extract config sections
        self.train_config = config["training"]
        self.system_config = config["system"]
        self.prediction_config = config.get("prediction", {})
        
        # Prediction mode
        self.prediction_mode = self.prediction_config.get("mode", "absolute")
        self.output_clamp = self.prediction_config.get("output_clamp")
        
        # ADD THIS LINE - Call the validation method
        self.trainer_init_validation()
        
        # Dataset info
        self.n_species = len(config["data"]["species_variables"])
        self.n_globals = len(config["data"]["global_variables"])
        
        # Check for empty validation set
        self.has_validation = val_dataset is not None and len(val_dataset) > 0
        if not self.has_validation:
            self.logger.warning("No validation data – early‑stopping and LR‑plateau will be skipped")
        
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
        self.log_interval = self.train_config.get("log_interval", 1000)
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.min_delta = self.train_config["min_delta"]
        self.gradient_accumulation_steps = self.train_config["gradient_accumulation_steps"]
        self.empty_cache_interval = self.train_config.get("empty_cache_interval", 1000)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        
        # Training history
        self.training_history = {
            "config": config,
            "prediction_mode": self.prediction_mode,
            "epochs": []
        }
        
    def trainer_init_validation(self):
        """Add this to Trainer.__init__ after self.prediction_mode is set"""
        # Bug 2 Fix: Validate ratio mode requirements at initialization
        if self.prediction_mode == "ratio":
            if not hasattr(self.norm_helper, 'ratio_stats') or self.norm_helper.ratio_stats is None:
                raise ValueError(
                    "Training in 'ratio' mode requires ratio statistics from preprocessing. "
                    "Ensure data was preprocessed with prediction.mode='ratio' in config. "
                    "Current normalization data does not contain ratio_stats."
                )
            
            # Additional check for model compatibility
            model_type = self.config["model"]["type"]
            if model_type != "deeponet":
                raise ValueError(
                    f"Ratio prediction mode is only compatible with 'deeponet' model, "
                    f"but '{model_type}' was specified. Please use 'deeponet' or switch to 'absolute' mode."
                )
    
    def _setup_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Setup data loaders."""
        from data.dataset import create_dataloader
        
        self.train_loader = create_dataloader(
            train_dataset, self.config, shuffle=True, 
            device=self.device, drop_last=True
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
        """Setup AdamW optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.dim() == 1 or "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.train_config["learning_rate"],
            betas=tuple(self.train_config.get("betas", [0.9, 0.999])),
            eps=self.train_config.get("eps", 1e-8)
        )
    
    def _setup_scheduler(self):
        """Create the learning‑rate scheduler as requested in the config."""
        scheduler_type = self.train_config.get("scheduler", "none").lower()

        if scheduler_type == "none" or not self.train_loader:
            self.scheduler = None
            self.scheduler_step_on_batch = False
            return

        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.gradient_accumulation_steps
        )

        params: Dict[str, Any] = self.train_config.get("scheduler_params", {})

        if scheduler_type == "cosine":
            T_0_epochs: int = params.get("T_0", 1)
            if T_0_epochs <= 0:
                raise ValueError("`scheduler_params.T_0` must be > 0 for 'cosine'.")
            T_0_steps = T_0_epochs * steps_per_epoch

            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0_steps,
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-8),
            )
            self.scheduler_step_on_batch = True
            return

        if scheduler_type == "plateau":
            if not self.has_validation:
                self.logger.warning("Plateau scheduler requires validation data, falling back to no scheduler")
                self.scheduler = None
                self.scheduler_step_on_batch = False
                return
                
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
                min_lr=params.get("min_lr", 1e-7),
            )
            self.scheduler_step_on_batch = False
            return

        raise ValueError(f"Unknown scheduler '{scheduler_type}'")
    
    def _setup_loss(self):
        """Setup loss function."""
        loss_type = self.train_config["loss"]
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=self.train_config.get("huber_delta", 0.25))
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
    
    def _setup_amp(self):
        """Setup automatic mixed precision training."""
        self.use_amp = self.train_config.get("use_amp", False)
        if self.device.type not in ('cuda', 'mps', 'cpu'):
            self.use_amp = False
        self.scaler = None
        self.amp_dtype = None
        
        if not self.use_amp:
            return
        
        dtype_str = str(self.train_config.get("amp_dtype", "bfloat16")).lower()
        
        if dtype_str not in ["bfloat16", "float16"]:
            self.logger.warning(f"Invalid amp_dtype '{dtype_str}'. Falling back to 'bfloat16'.")
            dtype_str = "bfloat16"
        
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        if self.device.type == 'cpu' and self.amp_dtype != torch.bfloat16:
            self.logger.warning(f"AMP with {dtype_str} is not supported on CPU. Disabling AMP.")
            self.use_amp = False
            return

        if self.device.type == 'mps' and self.amp_dtype not in [torch.float16, torch.bfloat16]:
            self.logger.warning(f"AMP with {dtype_str} is not supported on MPS. Disabling AMP.")
            self.use_amp = False
            return

        # GradScaler only for float16 on CUDA
        if self.amp_dtype == torch.float16 and self.device.type == 'cuda':
            self.scaler = GradScaler(device_type='cuda')
    
    # --- helper ------------------------------------------------------------
    def _standardize_log_ratios(self, log_ratios: torch.Tensor) -> torch.Tensor:
        """
        Standardizes raw log-ratio model outputs for loss calculation.
        """
        # Bug 2 Fix: Raise error immediately if ratio stats missing
        if self.prediction_mode == "ratio" and self.norm_helper.ratio_stats is None:
            raise ValueError(
                "Ratio statistics are missing but prediction mode is 'ratio'. "
                "This likely means data was preprocessed in 'absolute' mode. "
                "Please reprocess data with prediction.mode='ratio' in config."
            )
        
        # If not in ratio mode, return as-is (shouldn't happen but defensive)
        if self.norm_helper.ratio_stats is None:
            return log_ratios

        stats = self.norm_helper.ratio_stats
        species = self.config["data"]["species_variables"]
        device = log_ratios.device
        dtype = log_ratios.dtype

        means = torch.tensor([stats[v]["mean"] for v in species],
                            device=device, dtype=dtype)
        stds = torch.tensor([stats[v]["std"] for v in species],
                            device=device, dtype=dtype)

        stds = torch.clamp(stds, min=self.config["normalization"]["min_std"])
        return (log_ratios - means) / stds

    def _compute_loss(self, outputs: torch.Tensor,
                    targets: torch.Tensor,
                    inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss with Bug 2 fix for ratio mode validation.
        """
        if self.prediction_mode == "ratio":
            # Bug 2 Fix: This will now raise an error if ratio_stats are missing
            outputs_std = self._standardize_log_ratios(outputs)
            return self.criterion(outputs_std, targets)

        # absolute mode
        if self.output_clamp is not None:
            outputs = torch.clamp(outputs, min=self.output_clamp)
        return self.criterion(outputs, targets)
    
    def train(self) -> float:
        """Execute the training loop."""
        if not self.train_loader:
            self.logger.error("Training loader not available. Cannot start training.")
            return float("inf")

        self.logger.info(f"Starting training...")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.has_validation:
            self.logger.info(f"Val batches: {len(self.val_loader)}")
        
        try:
            self._run_training_loop()
            
            self.logger.info(
                f"Training completed. Best validation loss: {self.best_val_loss:.6f} "
                f"at epoch {self.best_epoch}"
            )
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
            
        finally:
            # Save final training history
            save_path = self.save_dir / "training_log.json"
            with open(save_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Clear cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return self.best_val_loss
    
    def _run_training_loop(self):
        """Main training loop with fix for no-validation saving."""
        best_train_loss = float("inf")
        
        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self._train_epoch()
            
            # Validate if available
            val_loss, val_metrics = self._validate()

            # CORRECTED: Only step epoch-based schedulers here.
            # Batch-based schedulers are handled correctly in _train_epoch.
            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                # For any other epoch-based scheduler, add logic here.
                # The cosine scheduler is batch-based, so it's handled in _train_epoch

            # Log epoch
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)

            # Save best model and early stopping logic
            if self.has_validation:
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                # No validation set: save based on best training loss
                if train_loss < (best_train_loss - self.min_delta):
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                    self.best_epoch = epoch
                    self._save_best_model()
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch and return the average loss."""
        self.model.train()
        total_loss, total_samples = 0.0, 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs  = inputs.to(self.device,  non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_update_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )
            
            if is_update_step:
                if self.train_config["gradient_clip"] > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config["gradient_clip"]
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler and self.scheduler_step_on_batch:
                    self.scheduler.step()
                self.global_step += 1
                
                # Empty cache periodically
                if self.global_step % self.empty_cache_interval == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if self.global_step > 0 and self.global_step % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {total_loss / total_samples:.3e}"
                )
        
        return total_loss / total_samples, {}

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on the validation set and return the average loss."""
        if not self.has_validation or self.val_loader is None:
            return float("inf"), {}
            
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets, inputs)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        return avg_loss, {}

    def evaluate_test(self) -> float:
        """Compute the loss on the test set; returns `inf` if no test data."""
        if not self.test_loader:
            self.logger.warning("No test data available, skipping test evaluation.")
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets, inputs)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        self.logger.info(f"Test loss: {avg_loss:.6f}")
        return avg_loss

    def cleanup_dataloaders(self):
        """Properly cleanup DataLoader workers to prevent memory leaks."""
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            if loader is not None:
                try:
                    # Force workers to terminate
                    loader._shutdown_workers()
                    del loader
                except:
                    pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _log_epoch(self, train_loss, val_loss, train_metrics, val_metrics, epoch_time):
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
        log_entry = {
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            "lr": lr,
        }
        self.training_history["epochs"].append(log_entry)
        
        val_str = f"Val loss: {val_loss:.3e}" if self.has_validation else "Val loss: N/A"
        self.logger.info(
            f"Epoch {self.current_epoch}/{self.train_config['epochs']} "
            f"Train loss: {train_loss:.3e} {val_str} "
            f"Time: {epoch_time:.1f}s LR: {log_entry['lr']:.2e}"
        )
    
    def _save_best_model(self):
        """Save the best model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        # Export model if enabled
        if self.system_config.get("use_torch_export", False):
            # Get example input from the first available loader
            example_loader = self.val_loader or self.train_loader
            if example_loader:
                # Get a single sample batch for export
                example_inputs, _ = next(iter(example_loader))
                example_inputs = example_inputs.to(self.device)
                
                # Note: We pass the full batch, but export_model will handle
                # making the batch dimension dynamic
                export_path = self.save_dir / "exported_model.pt"
                
                # Log the shape being used for export
                self.logger.info(f"Exporting model with example input shape: {example_inputs.shape}")
                
                export_model(self.model, example_inputs, export_path)
            else:
                self.logger.warning("Cannot export model: no data loader available to create example input.")