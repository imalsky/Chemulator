#!/usr/bin/env python3
"""
Unified training module for chemical kinetics models using the LiLaN approach.

This module implements training logic for Linear Latent Network (LiLaN) models,
including support for gradient accumulation, mixed precision training, and 
various learning rate scheduling strategies.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import Dataset

from data.normalizer import NormalizationHelper
import math

# Optional import for Lion optimizer
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False


class Trainer:
    """
    Unified trainer for LiLaN chemical kinetics models.
    
    Handles training, validation, and testing of models with support for:
    - Mixed precision training (fp16/bf16)
    - Multiple optimizer types (AdamW, Lion)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing and export
    """
    
    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset],
            test_dataset: Optional[Dataset],
            config: Dict[str, Any],
            save_dir: Path,
            device: torch.device,
            norm_helper: Optional[NormalizationHelper] = None,
            processed_dir: Optional[Path] = None
        ):
            """
            Initialize the trainer
            
            Args:
                model: The neural network model to train
                train_dataset: Training dataset
                val_dataset: Validation dataset (optional)
                test_dataset: Test dataset (optional)
                config: Configuration dictionary containing all training parameters
                save_dir: Directory to save checkpoints and logs
                device: Device to run training on (cpu/cuda/mps)
                norm_helper: Helper for data normalization/denormalization
                processed_dir: Path to the preprocessed data directory
            """
            self.logger = logging.getLogger(__name__)
            
            # Core components
            self.model = model
            self.config = config
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.device = device
            self.norm_helper = norm_helper
            self.processed_dir = processed_dir
            self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"

            # Extract config sections for easier access
            self.train_config = config["training"]
            self.system_config = config["system"]
            self.data_config = config["data"]

            # Detect if we're in hyperparameter optimization mode
            self.is_hpo = self._detect_hpo_mode()
            
            # Set up data type
            self.dtype = self._setup_dtype()
            
            # Dataset validation
            self.has_validation = self._has_valid_dataset(val_dataset)
            
            # Create data loaders
            self._setup_dataloaders(train_dataset, val_dataset, test_dataset)
            
            # Initialize training state
            self._init_training_state()
            
            # Training configuration
            self._load_training_config()
            
            # Setup training components
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_loss()
            self._setup_amp()        
            
            # Initialize metrics logging
            self._init_csv_logger()
    
    def _detect_hpo_mode(self) -> bool:
        """Check if running in hyperparameter optimization mode."""
        return bool(
            self.config.get("optuna", {}).get("enabled", False) or 
            self.config.get("hpo", {}).get("enabled", False)
        )
    
    def _setup_dtype(self) -> torch.dtype:
        dtype_str = str(self.system_config.get("dtype", "float32")).lower()
        try:
            dt = getattr(torch, dtype_str)
        except AttributeError:
            self.logger.warning(f"Unknown dtype '{dtype_str}', defaulting to float32")
            return torch.float32
        if self.device.type == "cpu" and dt in (torch.float16, torch.bfloat16):
            self.logger.info("Forcing dtype to float32 on CPU (fp16/bf16 not fully supported in this pipeline).")
            return torch.float32
        return dt
    
    def _has_valid_dataset(self, dataset: Optional[Dataset]) -> bool:
        """Check if a valid dataset exists."""
        return dataset is not None and len(dataset) > 0
    
    def _init_training_state(self):
        """Initialize training state variables."""
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.total_training_time = 0.0
        self.patience_counter = 0
    
    def _load_training_config(self):
        """Load training configuration parameters."""
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.min_delta = self.train_config["min_delta"]
    
    def _init_csv_logger(self):
        """Initialize CSV file for logging training metrics."""
        self.csv_path = self.save_dir / "training_metrics.csv"
        self.csv_headers = [
            "epoch", "timestamp", "train_loss", "val_loss", 
            "learning_rate", "epoch_time", "total_time", "best_val_loss"
        ]
        
        # Write headers to new file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def _log_to_csv(self, metrics: Dict[str, Any]):
        """Append metrics row to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [metrics.get(h, "") for h in self.csv_headers]
            writer.writerow(row)
    
    def _setup_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset]
    ):
        """Create DataLoaders for train/val/test datasets."""
        from data.dataset import create_dataloader
        
        self.train_loader = create_dataloader(
            train_dataset, self.config, shuffle=True, 
            device=self.device, drop_last=True
        ) if train_dataset else None
        
        self.val_loader = create_dataloader(
            val_dataset, self.config, shuffle=False,
            device=self.device, drop_last=False
        ) if self.has_validation else None
        
        self.test_loader = create_dataloader(
            test_dataset, self.config, shuffle=False,
            device=self.device, drop_last=False
        ) if test_dataset and len(test_dataset) > 0 else None
    
    def _setup_optimizer(self):
        """
        Configure the optimizer with proper weight decay handling.
        
        Separates parameters into two groups:
        - Parameters with weight decay (matrices)
        - Parameters without weight decay (biases, normalization layers)
        """
        opt_name = self.train_config.get("optimizer", "adamw").lower()
        
        # Separate parameters by weight decay eligibility
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Biases and normalization parameters should not have weight decay
            if param.dim() == 1 or "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create parameter groups with appropriate weight decay
        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Extract optimizer hyperparameters
        lr = self.train_config["learning_rate"]
        betas = tuple(self.train_config.get("betas", [0.9, 0.999]))
        eps = self.train_config.get("eps", 1e-8)
        
        # Create optimizer
        if opt_name == "lion":
            if not LION_AVAILABLE:
                raise ImportError("Lion optimizer requested but lion-pytorch not installed")
            self.optimizer = Lion(param_groups, lr=lr, betas=betas)
            
        elif opt_name == "adamw":
            opt_kwargs = {"lr": lr, "betas": betas, "eps": eps}
            if self.device.type == "cuda":
                try:
                    self.optimizer = AdamW(param_groups, fused=True, **opt_kwargs)
                    self.logger.info("Using fused AdamW for improved performance")
                except (RuntimeError, TypeError) as e:
                    self.logger.debug(f"Fused AdamW not available: {e}")
                    self.optimizer = AdamW(param_groups, **opt_kwargs)
            else:
                self.optimizer = AdamW(param_groups, **opt_kwargs)
            self.logger.info(f"Using AdamW optimizer (lr={lr}, betas={betas}, eps={eps})")
            
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _setup_scheduler(self):
        """
        Configure learning rate scheduler.
        
        Supports:
        - Cosine annealing with warm restarts
        - ReduceLROnPlateau for validation-based scheduling
        - None (constant learning rate)
        """
        scheduler_type = self.train_config.get("scheduler", "none").lower()
        
        if scheduler_type == "none" or not self.train_loader:
            self.scheduler = None
            self.scheduler_step_on_batch = False
            return
        
        # Calculate steps per epoch for batch-level schedulers
        steps_per_epoch = len(self.train_loader)
        
        params = self.train_config.get("scheduler_params", {})
        
        if scheduler_type == "cosine":
            # Convert epoch-based T_0 to step-based
            T_0_epochs = params.get("T_0", 10)
            T_0_steps = T_0_epochs * steps_per_epoch
            
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0_steps,
                T_mult=params.get("T_mult", 2),
                eta_min=params.get("eta_min", 1e-8)
            )
            self.scheduler_step_on_batch = True
            self.logger.info(
                f"Cosine annealing scheduler: T_0={T_0_epochs} epochs ({T_0_steps} steps), "
                f"T_mult={params.get('T_mult', 2)}"
            )
            
        elif scheduler_type == "plateau":
            if not self.has_validation:
                self.logger.warning("ReduceLROnPlateau requires validation data, disabling scheduler")
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
            self.logger.info(f"ReduceLROnPlateau scheduler: factor={params.get('factor', 0.5)}, ")
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _setup_loss(self):
        """Configure loss function for training."""
        loss_name = self.train_config.get("loss", "mse").lower()
        
        if loss_name == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
            self.logger.info("Using MSE loss function")
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _setup_amp(self):
        """Configure automatic mixed precision training."""
        if self.dtype == torch.float64 or self.device.type != "cuda":
            self.use_amp = False
            self.amp_dtype = None
            self.scaler = GradScaler(enabled=False)
            
            if self.device.type != "cuda":
                self.logger.info(f"AMP disabled on device '{self.device.type}'")
            return

        self.use_amp = self.train_config.get("use_amp", True)
        dtype_str = str(self.train_config.get("amp_dtype", "bfloat16")).lower()

        # Parse AMP dtype
        dtype_map = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
            "float32": torch.float32, "fp32": torch.float32
        }
        
        self.amp_dtype = dtype_map.get(dtype_str, torch.float32)
        if self.amp_dtype == torch.float32:
            if dtype_str not in dtype_map:
                self.logger.warning(f"Unknown amp_dtype '{dtype_str}', defaulting to float32")

        # Only use AMP for actual mixed precision (fp16/bf16)
        if self.use_amp and self.amp_dtype not in {torch.float16, torch.bfloat16}:
            self.use_amp = False


        # GradScaler is only needed for fp16 (not bf16)
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        if self.use_amp:
            dtype_name = "fp16" if self.amp_dtype == torch.float16 else "bf16"
            self.logger.info(f"Automatic mixed precision enabled with {dtype_name}")
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the task loss."""
        # Compute main task loss
        main_loss = self.criterion(predictions, targets)
        
        if not isinstance(main_loss, torch.Tensor):
            main_loss = torch.tensor(main_loss, dtype=predictions.dtype, device=predictions.device)
        
        total_loss = main_loss
        return total_loss
    
    def train(self) -> float:
        """Execute the complete training loop."""
        if not self.train_loader:
            self.logger.error("No training data available")
            return float("inf")

        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.has_validation:
            self.logger.info(f"Validation batches: {len(self.val_loader)}")
        
        try:
            self._run_training_loop()
            
            self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f} ")
            
            # Export model once at end if in HPO mode
            if self.is_hpo and self.system_config.get("use_torch_export", False):
                exported_path = self.save_dir / "best_model_exported.pt"
                self.logger.info("Exporting model for HPO trial...")
                success = self._export_from_live_model(exported_path)
                if not success:
                    self.logger.warning("Model export failed after HPO trial")
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            self.logger.info(f"Training metrics saved to {self.csv_path}")
        
        return self.best_val_loss
    
    def _run_training_loop(self):
        """
        Main training loop implementation.
        
        Handles:
        - Epoch iteration
        - Learning rate scheduling
        - Early stopping
        - Checkpointing
        """
        best_train_loss = float("inf")
        
        
        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training epoch
            train_loss, train_metrics = self._train_epoch()
            
            # Validation
            val_loss, val_metrics = self._validate()

            # For tuning
            self._optuna_report_and_prune(val_loss)
            
            # Update learning rate scheduler
            self._update_scheduler(val_loss)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Check for improvement and save best model
            improved = self._check_improvement(train_loss, val_loss, best_train_loss)
            if improved:
                if self.has_validation:
                    self.best_val_loss = val_loss
                else:
                    best_train_loss = train_loss
                    self.best_val_loss = train_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_best_model()
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    def _optuna_report_and_prune(self, val_loss: float) -> None:
        """
        Report validation loss to Optuna and prune if requested.

        Safe no-op when no trial is attached. Attach a trial as `self.trial`
        (e.g., in __init__ or right after constructing the Trainer).

        Call this once per epoch, immediately after computing `val_loss`.
        """
        trial = getattr(self, "trial", None)
        if trial is None:
            return

        # Convert to plain float (handles tensors)
        try:
            loss_value = float(val_loss.detach().cpu().item())  # tensor path
        except Exception:
            loss_value = float(val_loss)  # already a number

        try:
            trial.report(loss_value, step=int(self.current_epoch))
            if trial.should_prune():
                import optuna
                self.logger.info(
                    "Optuna requested pruning at epoch %d (val_loss=%.6e)",
                    self.current_epoch, loss_value
                )
                raise optuna.TrialPruned()
        except Exception as e:
            # Don't fail training because reporting/pruning had an issue.
            # Keep at debug level to avoid noisy logs during non-Optuna runs.
            self.logger.debug("Optuna hook failed: %s", e, exc_info=True)

    
    def _update_scheduler(self, val_loss: float):
        """Update learning rate scheduler after epoch."""
        if self.scheduler and not self.scheduler_step_on_batch:
            if isinstance(self.scheduler, ReduceLROnPlateau) and self.has_validation:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
    def _check_improvement(
        self,
        train_loss: float,
        val_loss: float,
        best_train_loss: float
    ) -> bool:
        """Check if model performance improved."""
        if self.has_validation:
            return val_loss < (self.best_val_loss - self.min_delta)
        else:
            return train_loss < (best_train_loss - self.min_delta)
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = self._prepare_batch(inputs, targets)
            
            with autocast(device_type=self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                predictions, _ = self.model(inputs)
                loss = self._compute_loss(predictions, targets)
            
            # Backward pass
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step optimizer on every batch
            self._optimizer_step()
            self.global_step += 1
            
            # Step batch-level scheduler on every batch
            # This is now correct as the optimizer also steps on every batch
            if self.scheduler and self.scheduler_step_on_batch:
                self.scheduler.step()
            
            # Track metrics using unscaled loss
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, {}
    

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Unscale gradients if using fp16
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        
        # Apply gradient clipping if configured
        clip_value = self.train_config.get("gradient_clip", 0.0)
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        
        # Optimizer step
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
    
    @torch.inference_mode()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation epoch."""
        if not self.has_validation or self.val_loader is None:
            return float("inf"), {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = self._prepare_batch(inputs, targets)
            
            with autocast(device_type=self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                predictions, _ = self.model(inputs)
                loss = self._compute_loss(predictions, targets)
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        return avg_loss, {}
    
    @torch.inference_mode()
    def evaluate_test(self) -> float:
        """Evaluate model on test set."""
        if not self.test_loader:
            self.logger.warning("No test data available")
            return float("inf")
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in self.test_loader:
            inputs, targets = self._prepare_batch(inputs, targets)
            
            with autocast(device_type=self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                predictions, _ = self.model(inputs)
                loss = self._compute_loss(predictions, targets)
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        self.logger.info(f"Test loss: {avg_loss:.6f}")
        return avg_loss
    
    def _prepare_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training/validation."""
        if inputs.device != self.device or inputs.dtype != self.dtype:
            inputs = inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if targets.device != self.device or targets.dtype != self.dtype:
            targets = targets.to(device=self.device, dtype=self.dtype, non_blocking=True)
        
        return inputs, targets
    
    def _log_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log epoch results to console and CSV."""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Log to CSV file
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
        """
        Save best model checkpoint and optionally export for inference.
        
        Saves:
        - Full checkpoint with optimizer state for resuming training
        - Optionally exports model using torch.export for deployment
        """
        # Save full checkpoint
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

        # Export for inference if requested (skip during HPO)
        if self.system_config.get("use_torch_export", False) and not self.is_hpo:
            exported_path = self.save_dir / "best_model_exported.pt"
            success = self._export_from_live_model(exported_path)
            if not success:
                self.logger.warning("Model export failed - continuing with training")

    @torch.inference_mode()
    def _export_from_live_model(self, output_path: Path) -> bool:
        """
        Export model using torch.export for optimized inference.
        
        Args:
            output_path: Path to save exported model
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            self.logger.info("Exporting model for inference...")

            # Get the underlying model (unwrap any wrappers)
            model_to_export = self._unwrap_model(self.model)
            
            # Save training state and switch to eval mode
            was_training = self.model.training
            model_to_export.eval()

            # Create example input matching model expectations
            example_input = self._create_example_input(model_to_export)

            # Export model
            exported = self._perform_export(model_to_export, example_input)
            
            # Save exported model
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(exported, output_path)
            self.logger.info(f"Model exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)
            return False
        finally:
            # Restore training state
            if 'was_training' in locals() and was_training:
                self.model.train()
    
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap model from any wrappers (DataParallel, compile, etc)."""
        unwrapped = model
        
        # Unwrap torch.compile
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        
        # Unwrap DataParallel/DistributedDataParallel
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        
        return unwrapped
    
    def _create_example_input(self, model: nn.Module) -> torch.Tensor:
        """
        Create a structured example input tensor for model export.
        
        The LinearLatentNetwork model expects input with structure:
        - First n_species entries: species initial conditions
        - Next n_globals entries: global parameters  
        - Final M entries: monotonically increasing time points
        
        This function creates a valid input that respects these constraints,
        particularly ensuring the time vector is monotonically increasing
        to avoid errors during the model's internal time integration.
        """
        data_cfg = self.config["data"]
        n_species = len(data_cfg["species_variables"])
        n_globals = len(data_cfg["global_variables"])
        
        # Infer sequence length M from a data loader
        M = None
        try:
            loader = self.val_loader or self.train_loader
            if loader is not None:
                sample_inputs, _ = next(iter(loader))
                # M is the remaining dimensions after species and globals
                M = sample_inputs.shape[1] - n_species - n_globals
        except Exception as e:
            self.logger.warning(f"Could not infer M from data loader: {e}")
        
        # Fall back to shard metadata if needed
        if not M or M <= 1:
            try:
                from utils.utils import load_json
                shard_index = load_json(self.processed_dir / "shard_index.json")
                M = int(shard_index.get("M_per_sample", 0))
            except Exception:
                pass
        
        if not M or M <= 1:
            raise ValueError(
                f"Could not determine a valid sequence length M for model export. "
                f"Got M={M}, need M>1 for time series data."
            )
        
        # Get dtype and device from model parameters
        param = next((p for p in model.parameters() if p.requires_grad), None)
        dtype = param.dtype if param is not None else torch.float32
        device = param.device if param is not None else torch.device("cpu")
        
        # Create structured input components
        # 1. Random initial conditions for species (positive values typical for concentrations)
        x0 = torch.abs(torch.randn(1, n_species, dtype=dtype, device=device)) * 0.1 + 0.01
        
        # 2. Random global parameters (reasonable range)
        globals_vec = torch.randn(1, n_globals, dtype=dtype, device=device)
        
        # 3. Monotonically increasing time vector (CRITICAL for model correctness)
        # Use linspace to guarantee monotonic increase
        time_vec = torch.linspace(0.0, 10.0, steps=M, dtype=dtype, device=device).unsqueeze(0)
        
        # Concatenate to form the final structured input
        # Shape: [1, n_species + n_globals + M]
        example_input = torch.cat([x0, globals_vec, time_vec], dim=1)
        
        self.logger.debug(
            f"Created example input: shape={example_input.shape}, "
            f"n_species={n_species}, n_globals={n_globals}, M={M}"
        )
        
        return example_input
    
    def _perform_export(self, model: nn.Module, example_input: torch.Tensor):
        """Perform the actual model export."""
        # Export with the single positional argument the model expects
        return torch.export.export(model, (example_input,))