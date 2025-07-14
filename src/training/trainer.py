#!/usr/bin/env python3
"""
Optimized training pipeline for chemical kinetics models.

Features:
- Mixed precision training with BFloat16 on A100
- Gradient accumulation for large effective batch sizes
- Advanced learning rate scheduling
- Memory-efficient training with periodic cache clearing
- Comprehensive logging
- Compatible with all device types (CUDA, MPS, CPU)
- Full CUDA graph support with proper step marking
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.dataset import NPYDataset, create_dataloader
from models.model import export_jit_model
from data.device_prefetch import DevicePrefetchLoader


# Training constants
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
LOGGING_INTERVAL_SECONDS = 30.0
SCHEDULER_PLATEAU_FACTOR = 0.5
SCHEDULER_PLATEAU_PATIENCE = 10
SCHEDULER_PLATEAU_MIN_LR = 1e-7
DEFAULT_EMPTY_CACHE_INTERVAL = 200  # Increased to reduce overhead on A100


class Trainer:
    """
    Optimised trainer for chemical-kinetics networks.
    """
    def __init__(               
        self,
        model: nn.Module,
        train_dataset: NPYDataset,
        val_dataset: NPYDataset,
        test_dataset: NPYDataset,
        config: Dict[str, Any],
        save_dir: Path,
        device: torch.device,
    ):
        self.logger = logging.getLogger(__name__)

        self.model          = model
        self.config         = config
        self.save_dir       = save_dir
        self.device         = device
        self.train_config   = config["training"]
        self.system_config  = config["system"]

        # Data loaders
        base_train = create_dataloader(train_dataset, config, shuffle=True,  device=device)
        base_val   = create_dataloader(val_dataset,   config, shuffle=False, device=device)
        base_test  = create_dataloader(test_dataset,  config, shuffle=False, device=device)

        self.logger.info("Using pre-normalized data - direct GPU streaming enabled")
        
        try:
            from data.device_prefetch import DevicePrefetchLoader
            use_prefetch = device.type == "cuda"
        except ModuleNotFoundError:
            self.logger.warning("DevicePrefetchLoader not found - falling back to plain loaders")
            use_prefetch = False

        if use_prefetch:
            self.train_loader = DevicePrefetchLoader(base_train, device)
            self.val_loader = DevicePrefetchLoader(base_val, device)
            self.test_loader = DevicePrefetchLoader(base_test, device)
            self.logger.info("GPU pre-fetch enabled")
        else:
            self.train_loader = base_train
            self.val_loader = base_val
            self.test_loader = base_test

        # Misc setup
        self.log_interval            = self.train_config.get("log_interval", 10)
        self.current_epoch           = 0
        self.global_step             = 0
        self.best_val_loss           = float("inf")
        self.best_epoch              = -1
        self.total_training_time     = 0
        self.patience_counter        = 0
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.min_delta               = self.train_config["min_delta"]
        self.empty_cache_interval    = self.system_config.get("empty_cache_interval", DEFAULT_EMPTY_CACHE_INTERVAL)

        self.log_file         = self.save_dir / "training_log.json"
        self.training_history = {"config": config, "epochs": []}
        self.max_history_epochs     = 1_000
        self.history_save_interval  = 100

        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        self._cuda_graphs_enabled = False
        self._check_cuda_graphs_compatibility()
        
    def _check_cuda_graphs_compatibility(self):
        """Check if we should use CUDA graphs based on the configuration and device."""
        if (self.device.type == "cuda" and 
            self.system_config.get("use_torch_compile", False) and 
            self.train_config.get("use_amp", False) and
            self.system_config.get("compile_mode") == "max-autotune"):
            self._cuda_graphs_enabled = True
            self.logger.info("CUDA graphs compatibility mode enabled")
            
            # Import the cudagraph marking function with fallback
            try:
                from torch._inductor import cudagraph_mark_step_begin
                self.cudagraph_mark_step = cudagraph_mark_step_begin
            except ImportError:
                try:
                    from torch._dynamo import mark_step_begin
                    self.cudagraph_mark_step = mark_step_begin
                except ImportError:
                    self.logger.warning("CUDA graph step marking not available, disabling")
                    self._cuda_graphs_enabled = False
        else:
            self._cuda_graphs_enabled = False
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to biases, layer norm, and embeddings
            if param.dim() == 1 or "bias" in name or "norm" in name.lower() or "embed" in name.lower():
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
            betas=DEFAULT_BETAS,
            eps=DEFAULT_EPS
        )
        
        self.logger.info(
            f"Optimizer: AdamW with lr={self.train_config['learning_rate']:.2e}, "
            f"weight_decay={self.train_config['weight_decay']:.2e}"
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.train_config["scheduler"]
        
        if scheduler_type == "cosine":
            params = self.train_config["scheduler_params"]
            
            # Calculate T_0 in steps - ensure at least 1
            steps_per_epoch = max(1, len(self.train_loader) // self.train_config["gradient_accumulation_steps"])
            T_0 = max(1, params["T_0"] * steps_per_epoch)
            
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=params["T_mult"],
                eta_min=params["eta_min"]
            )
            self.scheduler_step_on_batch = True
            
        elif scheduler_type == "plateau":
            params = self.train_config.get("scheduler_params", {})
            
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", SCHEDULER_PLATEAU_FACTOR),
                patience=params.get("patience", SCHEDULER_PLATEAU_PATIENCE),
                min_lr=params.get("min_lr", SCHEDULER_PLATEAU_MIN_LR)
            )
            self.scheduler_step_on_batch = False
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def _setup_loss(self):
        """Setup loss function."""
        loss_type = self.train_config["loss"]
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=self.train_config["huber_delta"])
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
        
        self.logger.info(f"Loss function: {loss_type}")
    
    def _setup_amp(self):
        self.use_amp = self.train_config.get("use_amp", False) and self.device.type in ("cuda", "mps")
        self.scaler = None
        self.amp_dtype = None

        if not self.use_amp:
            self.logger.info("AMP disabled")
            return

        dtype_str = str(self.train_config.get("amp_dtype", "float16")).lower()
        if dtype_str not in ("float16", "bfloat16"):
            raise ValueError(f"Invalid amp_dtype '{dtype_str}' – choose 'float16' or 'bfloat16'.")

        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        # GradScaler only meaningful for fp16 on CUDA
        if self.amp_dtype is torch.float16 and self.device.type == "cuda":
            self.scaler = GradScaler()
        elif self.device.type == "mps" and self.amp_dtype is torch.float16:
            self.logger.warning("GradScaler not supported on MPS – continuing without it.")

        self.logger.info("AMP enabled (dtype=%s, device=%s)", dtype_str, self.device.type)
        
    def train(self) -> float:
        """
        Execute the training loop.
        
        Returns:
            Best validation loss achieved
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        
        if len(self.train_loader) == 0:
            self.logger.error("Training dataset is empty! Check data splits.")
            sys.exit(1)
        
        try:
            for epoch in range(1, self.train_config["epochs"] + 1):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                train_loss, train_metrics = self._train_epoch()
                
                # Validation phase
                val_loss, val_metrics = self._validate()
                
                # Update scheduler
                if not self.scheduler_step_on_batch:
                    self.scheduler.step(val_loss)
                
                # Track time
                epoch_time = time.time() - epoch_start
                self.total_training_time += epoch_time
                
                # Log results
                self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
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
            # Save training history
            with open(self.log_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        return self.best_val_loss
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        One training epoch with optional gradient accumulation.
        Keeps only an exponential-moving-average of gradient norms
        to avoid unbounded memory growth.
        """
        self.model.train()
        total_loss, total_samples = 0.0, 0
        grad_norm_ema            = 0.0
        ema_decay                = 0.98  # Exponential moving average decay for grad norm
        accumulation_steps       = self.train_config["gradient_accumulation_steps"]
        last_log_time            = time.time()

        if self._cuda_graphs_enabled and hasattr(self, "cudagraph_mark_step"):
            self.cudagraph_mark_step()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs  = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, targets) / accumulation_steps

            (self.scaler.scale(loss) if self.scaler else loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                grad = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.train_config["gradient_clip"])
                grad_norm_ema = ema_decay * grad_norm_ema + (1 - ema_decay) * float(grad)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler_step_on_batch:
                    self.scheduler.step()
                self.global_step += 1

                if self._cuda_graphs_enabled and hasattr(self, "cudagraph_mark_step"):
                    self.cudagraph_mark_step()

            # bookkeeping
            batch_loss     = loss.item() * accumulation_steps
            total_loss    += batch_loss * inputs.size(0)
            total_samples += inputs.size(0)

            if time.time() - last_log_time > LOGGING_INTERVAL_SECONDS:
                pct = 100.0 * (batch_idx + 1) / len(self.train_loader)
                self.logger.info(f"Epoch {self.current_epoch:03d} "
                                f"{pct:5.1f}%  Loss {batch_loss:.4e}")
                last_log_time = time.time()

            # Clear cache periodically to prevent OOM
            if self.device.type == 'cuda' and (batch_idx + 1) % self.empty_cache_interval == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples if total_samples else float("inf")
        metrics  = {
            "grad_norm_ema": grad_norm_ema,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        return avg_loss, metrics
    
    @torch.no_grad()
    def _run_eval_loop(self, loader: DataLoader, enable_logging: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Run evaluation loop on any dataset.
        
        Args:
            loader: DataLoader to evaluate
            enable_logging: Whether to enable progress logging
            
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        # Time tracking
        eval_start = time.time()
        last_log_time = eval_start
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            if enable_logging:
                current_time = time.time()
                if current_time - last_log_time > LOGGING_INTERVAL_SECONDS:
                    progress = (batch_idx + 1) / len(loader) * 100
                    elapsed = current_time - eval_start
                    self.logger.info(
                        f"Epoch {self.current_epoch:03d} Val: {progress:.1f}% "
                        f"({batch_idx+1}/{len(loader)} batches), "
                        f"Elapsed: {elapsed:.1f}s"
                    )
                    last_log_time = current_time
        
        elapsed_total = time.time() - eval_start
        if enable_logging:
            self.logger.info(f"Evaluation completed in {elapsed_total:.1f}s")
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = float('inf')
            
        return avg_loss, {}
    
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model during training."""
        return self._run_eval_loop(self.val_loader, enable_logging=True)
    
    @torch.no_grad()
    def evaluate_test(self) -> float:
        """Evaluate on test set."""
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        test_loss, _ = self._run_eval_loop(self.test_loader, enable_logging=False)
        
        self.logger.info(f"Test loss: {test_loss:.6f}")
        
        return test_loss
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_time": self.total_training_time
        }
        
        path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved best model (epoch {self.current_epoch})")
        
        # Also export JIT model if requested
        if self.system_config.get("save_jit_model", False):
            self._export_jit_model()
    
    def _export_jit_model(self):
        """Export model as TorchScript with improved error handling."""
        try:
            if len(self.val_loader) == 0:
                self.logger.warning("Validation loader is empty, trying train loader for JIT export")
                if len(self.train_loader) == 0:
                    self.logger.warning("Train loader is also empty, skipping JIT export")
                    return
                loader_to_use = self.train_loader
            else:
                loader_to_use = self.val_loader
            
            example_batch = next(iter(loader_to_use))
            example_input = example_batch[0][:1].to(self.device)  # Single sample
            
            jit_path = self.save_dir / "best_model_jit.pt"
            export_jit_model(self.model, example_input, jit_path)
            
            example_path = self.save_dir / "jit_example_input.pt"
            torch.save(example_input, example_path)
            self.logger.info(f"Saved JIT example input to {example_path}")
            
        except Exception as e:
            self.logger.warning(f"JIT export failed: {e}")
            self.logger.info("Model saved in standard format, JIT export can be done manually later")
    
    def _log_epoch(        
        self,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
    ):
        improvement = self.best_val_loss - val_loss
        grad_norm   = train_metrics.get("grad_norm_ema", 0.0)

        msg = (f"Epoch {self.current_epoch:03d} | "
               f"Train {train_loss:.4e} | Val {val_loss:.4e} | "
               f"LR {train_metrics['learning_rate']:.2e} | "
               f"Grad {grad_norm:.2f} | Time {epoch_time:.1f}s")
        if improvement > 0:
            msg += f" | ▲ {improvement:.4e}"
        self.logger.info(msg)

        self.training_history["epochs"].append(
            {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm": grad_norm,
                "lr": train_metrics["learning_rate"],
                "time_s": epoch_time,
                "improvement": max(improvement, 0.0),
            }
        )

        if len(self.training_history["epochs"]) > self.max_history_epochs:
            self.training_history["epochs"] = self.training_history["epochs"][-self.max_history_epochs :]

        if self.current_epoch % self.history_save_interval == 0:
            try:
                with open(self.log_file, "w") as f:
                    json.dump(self.training_history, f, indent=2)
            except Exception as exc:
                self.logger.warning(f"Failed to save history: {exc}")