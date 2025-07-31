#!/usr/bin/env python3
"""
Training pipeline for chemical kinetics models with adaptive loss
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from models.model import export_model
from data.normalizer import NormalizationHelper


class ScaleAwareAdaptiveLoss(nn.Module):
    """
    Adaptive loss that accounts for different scales and variations in species.
    Ensures uniform relative accuracy across all species.
    """
    def __init__(self, norm_stats: Dict[str, Any], species_names: list, device: torch.device):
        super().__init__()
        self.species_names = species_names
        
        # Extract standard deviations for each species
        stds = []
        for species in species_names:
            if species in norm_stats["per_key_stats"]:
                stats = norm_stats["per_key_stats"][species]
                if "log_std" in stats:
                    stds.append(stats["log_std"])
                else:
                    # For non-log normalized species
                    stds.append(stats.get("std", 1.0))
            else:
                stds.append(1.0)
        
        # Convert to tensor
        stds = torch.tensor(stds, device=device, dtype=torch.float64)
        
        # Create weights inversely proportional to std
        # Species with larger std (harder to predict) get higher weight
        self.weights = 1.0 / torch.sqrt(stds)
        self.weights = self.weights / self.weights.mean()  # Normalize to mean=1
        
        # Log the weights
        logger = logging.getLogger(__name__)
        logger.info("Species-specific loss weights (based on std):")
        for species, weight, std in zip(species_names, self.weights.cpu().numpy(), stds.cpu().numpy()):
            logger.info(f"  {species}: weight={weight:.3f} (std={std:.3f})")
        
        # Optional: Manual adjustment for extreme species
        # You can uncomment and modify this if certain species need extra attention
        # extreme_species = ["O_evolution", "OH_evolution"]
        # for i, species in enumerate(species_names):
        #     if species in extreme_species:
        #         self.weights[i] *= 2.0  # Double weight for extreme species
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predictions in normalized space (batch, n_species)
            target: Targets in normalized space (batch, n_species)
        """
        # Per-species squared error
        squared_errors = (pred - target) ** 2
        
        # Apply species-specific weights
        weighted_errors = squared_errors * self.weights.unsqueeze(0)
        
        # Return mean
        return weighted_errors.mean()


class UniformRelativeErrorLoss(nn.Module):
    """
    Alternative loss that directly optimizes for uniform relative error.
    Works in normalized space but accounts for what the errors mean in real space.
    """
    def __init__(self, norm_stats: Dict[str, Any], species_names: list, device: torch.device, 
                 target_relative_error: float = 0.01):
        super().__init__()
        self.species_names = species_names
        self.target_relative_error = target_relative_error
        
        # Extract normalization parameters
        self.log_stds = []
        self.is_log = []
        
        for species in species_names:
            if species in norm_stats["per_key_stats"]:
                stats = norm_stats["per_key_stats"][species]
                method = norm_stats["normalization_methods"].get(species, "standard")
                
                if "log" in method:
                    self.log_stds.append(stats.get("log_std", 1.0))
                    self.is_log.append(True)
                else:
                    self.log_stds.append(0.0)  # Not used for linear normalized
                    self.is_log.append(False)
            else:
                self.log_stds.append(1.0)
                self.is_log.append(True)
        
        self.log_stds = torch.tensor(self.log_stds, device=device, dtype=torch.float64)
        self.is_log = torch.tensor(self.is_log, device=device, dtype=torch.bool)
        
        # Target error in normalized space for each species
        # For log-normalized species: target_norm_error = log10(1 + target_relative_error) / log_std
        log_target = np.log10(1 + target_relative_error)
        self.target_normalized_errors = torch.where(
            self.is_log,
            log_target / self.log_stds,
            torch.tensor(target_relative_error, device=device, dtype=torch.float64)
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Target relative error: {target_relative_error:.1%}")
        logger.info("Target normalized RMSE per species:")
        for species, target_rmse in zip(species_names, self.target_normalized_errors.cpu().numpy()):
            logger.info(f"  {species}: {target_rmse:.4f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that penalizes deviations from target relative error.
        """
        # Compute normalized errors
        normalized_errors = torch.abs(pred - target)
        
        # Compute how far each species is from its target error
        error_ratios = normalized_errors / self.target_normalized_errors.unsqueeze(0)
        
        # Penalize both under and over-achievement
        # Use squared ratio to penalize large deviations more
        loss = (error_ratios ** 2).mean()
        
        return loss


class Trainer:
    """Trainer with adaptive loss functions for multi-scale chemical kinetics"""
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
        self.data_config = config["data"]
        
        # Prediction mode
        self.prediction_mode = self.prediction_config.get("mode", "absolute")
        self.output_clamp = self.prediction_config.get("output_clamp")
        
        self._validate_trainer_config()
        
        # Dataset info
        self.input_species_names = self.data_config["species_variables"]
        self.species_names = self.data_config.get("target_species_variables", self.input_species_names)
        self.n_species = len(self.species_names)
        self.n_globals = len(self.data_config["global_variables"])
        
        # Check for validation data
        self.has_validation = val_dataset is not None and len(val_dataset) > 0
        if not self.has_validation:
            self.logger.warning("No validation data. Bad! Using training loss for checkpointing")
        
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
        self.log_interval = self.train_config.get("log_interval", 100)
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.min_delta = self.train_config["min_delta"]
        self.gradient_accumulation_steps = self.train_config["gradient_accumulation_steps"]
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        
        # Training history
        self.training_history = {
            "config": config,
            "prediction_mode": self.prediction_mode,
            "epochs": [],
            "per_species_metrics": []
        }
        
        # Validation metrics tracking
        self.track_species_metrics = self.train_config.get("track_species_metrics", True)

    def _validate_trainer_config(self):
        """Validate trainer configuration for correctness."""
        # Validate ratio mode requirements
        if self.prediction_mode == "ratio":
            # Check model compatibility
            model_type = self.config["model"]["type"]
            if model_type != "deeponet":
                raise ValueError(f"Training in 'ratio' mode requires DeepONet")
            
            if not hasattr(self.norm_helper, 'ratio_stats') or self.norm_helper.ratio_stats is None:
                raise ValueError("Training in 'ratio' mode requires ratio statistics from preprocessing.")
            
            self.logger.info("Ratio mode validation passed: using DeepONet with ratio statistics")
        
        # Validate output clamping configuration
        if self.output_clamp is not None:
            if isinstance(self.output_clamp, (list, tuple)):
                if len(self.output_clamp) != 2:
                    raise ValueError("output_clamp must be None, a single value (min), or a tuple/list of (min, max)")
            elif not isinstance(self.output_clamp, (int, float)):
                raise ValueError("output_clamp must be None, a number, or a tuple/list of two numbers")

    def _setup_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Setup data loaders for GPU-cached data."""
        from data.dataset import create_dataloader
        
        self.train_loader = create_dataloader(
            train_dataset,
            self.config,
            shuffle=True,
            device=self.device,
            drop_last=True
        ) if train_dataset else None
        
        self.val_loader = create_dataloader(
            val_dataset,
            self.config,
            shuffle=False,
            device=self.device,
            drop_last=False
        ) if val_dataset and len(val_dataset) > 0 else None
        
        self.test_loader = create_dataloader(
            test_dataset,
            self.config,
            shuffle=False,
            device=self.device,
            drop_last=False
        ) if test_dataset and len(test_dataset) > 0 else None
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with safe feature detection."""
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
        
        # Safely check for fused optimizer support
        optimizer_kwargs = {
            "lr": self.train_config["learning_rate"],
            "betas": tuple(self.train_config.get("betas", [0.9, 0.999])),
            "eps": self.train_config.get("eps", 1e-8),
        }
        
        # Only use fused if available and on CUDA
        if self.device.type == "cuda" and hasattr(torch.optim.AdamW, "fused"):
            try:
                # Test if fused parameter actually works
                test_opt = torch.optim.AdamW([torch.zeros(1)], fused=True)
                optimizer_kwargs["fused"] = True
                self.logger.info("Using fused AdamW optimizer")
            except Exception:
                self.logger.info("Fused AdamW not available, using standard implementation")
        
        self.optimizer = AdamW(param_groups, **optimizer_kwargs)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.train_config.get("scheduler", "none").lower()

        if scheduler_type == "none" or not self.train_loader:
            self.scheduler = None
            self.scheduler_step_on_batch = False
            return

        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        
        # Guard against division-by-zero or T_0=0 errors on small datasets.
        if steps_per_epoch == 0:
            self.logger.warning(
                f"Number of batches ({len(self.train_loader)}) is smaller than "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps}). "
                f"Scheduler will step once per epoch."
            )
            steps_per_epoch = 1

        params = self.train_config.get("scheduler_params", {})

        if scheduler_type == "cosine":
            T_0_epochs = params.get("T_0", 10)
            T_0_steps = T_0_epochs * steps_per_epoch

            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0_steps,
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-8),
            )
            self.scheduler_step_on_batch = True

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
                min_lr=params.get("min_lr", 1e-7),
            )
            self.scheduler_step_on_batch = False
        else:
            raise ValueError(f"Unknown scheduler '{scheduler_type}'")
    
    def _setup_loss(self):
        """Setup loss function with adaptive options."""
        loss_type = self.train_config["loss"]
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=self.train_config.get("huber_delta", 0.5))
        elif loss_type == "adaptive" or loss_type == "scale_aware":
            # Use scale-aware adaptive loss
            self.criterion = ScaleAwareAdaptiveLoss(
                self.norm_helper.stats,
                self.species_names,
                self.device
            )
        elif loss_type == "uniform_relative":
            # Use uniform relative error loss
            target_error = self.train_config.get("target_relative_error", 0.01)
            self.criterion = UniformRelativeErrorLoss(
                self.norm_helper.stats,
                self.species_names,
                self.device,
                target_relative_error=target_error
            )
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
    
    def _setup_amp(self):
        """Setup automatic mixed precision - disabled for float64."""
        # Check if using float64
        model_dtype = next(self.model.parameters()).dtype
        
        if model_dtype == torch.float64:
            self.use_amp = False
            self.scaler = GradScaler(enabled=False)
            self.amp_dtype = None
            self.logger.info("AMP disabled for float64 training")
            return
        
        self.use_amp = self.train_config.get("use_amp", True)
        
        # Get dtype
        dtype_str = str(self.train_config.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        
        # GradScaler only for float16
        self.scaler = GradScaler(enabled=(self.amp_dtype == torch.float16))
    
    def _compute_loss(self, outputs: torch.Tensor, 
                    targets: torch.Tensor,
                    inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with proper output clamping.
        
        output_clamp can be:
        - None: no clamping
        - Single value: clamp minimum only (backward compatibility)
        - Tuple/list of (min, max): clamp both sides
        """
        if self.output_clamp is not None and self.prediction_mode == "absolute":
            if isinstance(self.output_clamp, (list, tuple)):
                # Two-sided clamp
                outputs = torch.clamp(outputs, min=self.output_clamp[0], max=self.output_clamp[1])
            else:
                # Single value - clamp minimum only (backward compatibility)
                outputs = torch.clamp(outputs, min=self.output_clamp)
                self.logger.warning(
                    "Using single-sided output clamping (min only). "
                    "Consider using (min, max) tuple for two-sided clamping."
                )
        
        return self.criterion(outputs, targets)

    def train(self) -> float:
        """Execute the training loop."""
        if not self.train_loader:
            self.logger.error("Training loader not available")
            return float("inf")

        self.logger.info(f"Starting training...")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.has_validation:
            self.logger.info(f"Val batches: {len(self.val_loader)}")

        # Log loss function info
        self.logger.info(f"Using loss function: {type(self.criterion).__name__}")

        if self.system_config.get("use_torch_compile", False):
            self.logger.info("Compiling model with torch.compile...")
            self.logger.warning("    This is a one-time process that can take several minutes.")

        try:
            self._run_training_loop()
            self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
            
            # Run final diagnostics
            if self.track_species_metrics:
                self.diagnose_accuracy()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
            
        finally:
            save_path = self.save_dir / "training_log.json"
            with open(save_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        return self.best_val_loss
    
    def _run_training_loop(self):
        """Main training loop - optimized for GPU."""
        best_train_loss = float("inf")
        
        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

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
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                # Use training loss if no validation
                if train_loss < (best_train_loss - self.min_delta):
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                    self.best_epoch = epoch
                    self._save_best_model()

    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Optimized training epoch for GPU."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        accumulation_steps = self.gradient_accumulation_steps
        
        is_gpu_cached = hasattr(self.train_loader.dataset, 'gpu_cache') and self.train_loader.dataset.gpu_cache is not None
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if not is_gpu_cached:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs)
                loss = loss / accumulation_steps
            
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.train_config["gradient_clip"] > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.train_config["gradient_clip"]
                    )
                
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler and self.scheduler_step_on_batch:
                    self.scheduler.step()
                
                self.global_step += 1
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, {}

    @torch.inference_mode()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Validation with optional per-species metrics."""
        if not self.has_validation or self.val_loader is None:
            return float("inf"), {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # For per-species tracking
        if self.track_species_metrics and self.current_epoch % 10 == 0:
            all_errors = []
        
        is_gpu_cached = hasattr(self.val_loader.dataset, 'gpu_cache') and self.val_loader.dataset.gpu_cache is not None
        
        for inputs, targets in self.val_loader:
            if not is_gpu_cached:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs)
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Track per-species errors
            if self.track_species_metrics and self.current_epoch % 10 == 0:
                errors = ((outputs - targets) ** 2).detach()
                all_errors.append(errors)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        
        # Log per-species metrics
        if self.track_species_metrics and self.current_epoch % 10 == 0 and all_errors:
            all_errors = torch.cat(all_errors, dim=0)
            mse_per_species = all_errors.mean(dim=0)
            
            self.logger.info(f"Per-species validation MSE at epoch {self.current_epoch}:")
            species_metrics = {}
            for i, species in enumerate(self.species_names):
                mse = mse_per_species[i].item()
                rmse = np.sqrt(mse)
                self.logger.info(f"  {species}: MSE={mse:.2e}, RMSE={rmse:.3f}")
                species_metrics[species] = {"mse": mse, "rmse": rmse}
            
            self.training_history["per_species_metrics"].append({
                "epoch": self.current_epoch,
                "metrics": species_metrics
            })
        
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
        
        is_gpu_cached = hasattr(self.test_loader.dataset, 'gpu_cache') and self.test_loader.dataset.gpu_cache is not None

        for inputs, targets in self.test_loader:
            if not is_gpu_cached:
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
        
        # Run detailed diagnostics on test set
        if self.track_species_metrics:
            self.diagnose_accuracy(use_test=True)
        
        return avg_loss

    def _log_epoch(self, train_loss, val_loss, train_metrics, val_metrics, epoch_time):
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']
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
            f"Time: {epoch_time:.1f}s LR: {lr:.2e}"
        )
    
    def _save_best_model(self):
        """Save the best model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        # Export model if enabled
        if self.system_config.get("use_torch_export", False):
            example_loader = self.val_loader or self.train_loader
            if example_loader:
                example_inputs, _ = next(iter(example_loader))
                self.logger.info(f"Exporting model with example input shape: {example_inputs.shape}")
                export_path = self.save_dir / "exported_model.pt"
                export_model(self.model, example_inputs, export_path)