#!/usr/bin/env python3
"""
Hyperparameter tuning for chemical kinetics models using Optuna.
Optimized for LiLaN (linear_latent / linear_latent_mixture) architectures.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
import json

from utils.hardware import setup_device
from data.dataset import SequenceDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper
from utils.utils import seed_everything, ensure_directories, load_json_config, save_json, load_json


class OptunaPruningCallback:
    """Report intermediate values to Optuna for pruning."""
    def __init__(self, trial: optuna.Trial, min_epochs: int = 10):
        self.trial = trial
        self.min_epochs = min_epochs
        
    def __call__(self, epoch: int, val_loss: float) -> bool:
        self.trial.report(val_loss, epoch)
        if epoch < self.min_epochs:
            return False
        return self.trial.should_prune()


class HyperparameterTuner:
    """Manages hyperparameter optimization for chemical kinetics models."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.base_config = load_json_config(config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        
        # Preprocess data once
        self.processed_dir = self._prepare_data()
        
        # Load normalization stats
        norm_stats_path = self.processed_dir / "normalization.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
        
        self.norm_stats = load_json(norm_stats_path)
        self.norm_helper = NormalizationHelper(
            stats=self.norm_stats,
            device=self.device,
            species_vars=self.base_config["data"]["species_variables"],
            global_vars=self.base_config["data"]["global_variables"],
            time_var=self.base_config["data"]["time_variable"],
            config=self.base_config
        )
        
        # Load datasets once
        self._load_datasets()
        
    def _prepare_data(self) -> Path:
        """Preprocess data if needed and return processed directory."""
        from main import ChemicalKineticsPipeline
        
        self.logger.info("Preparing data for hyperparameter optimization...")
        pipeline = ChemicalKineticsPipeline(self.base_config)
        pipeline.preprocess_data()
        return pipeline.processed_dir
    
    def _load_datasets(self):
        """Load datasets once for reuse across trials."""
        self.logger.info(f"Loading datasets from: {self.processed_dir}")
        
        # Load shard index to check if sequence mode
        index_path = self.processed_dir / "shard_index.json"
        with open(index_path) as f:
            shard_index = json.load(f)
        
        is_sequence = bool(shard_index.get("sequence_mode", False))
        if not is_sequence:
            raise ValueError("Hyperparameter tuning requires sequence mode data")
        
        self.train_dataset = SequenceDataset(
            self.processed_dir, "train", self.base_config, self.device, self.norm_stats
        )
        self.val_dataset = SequenceDataset(
            self.processed_dir, "validation", self.base_config, self.device, self.norm_stats
        )
        
        self.logger.info(f"Datasets loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
    
    def suggest_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate a trial configuration for LiLaN models."""
        cfg = copy.deepcopy(self.base_config)
        
        # Model architecture
        model_type = trial.suggest_categorical("model_type", ["linear_latent", "linear_latent_mixture"])
        cfg["model"]["type"] = model_type
        
        # Core dimensions
        cfg["model"]["latent_dim"] = trial.suggest_categorical("latent_dim", [32, 48, 64, 96])
        
        # Network architecture
        enc_depth = trial.suggest_int("encoder_depth", 2, 4)
        dec_depth = trial.suggest_int("decoder_depth", 2, 4)
        
        # Create layer widths with gradual sizing
        enc_widths = []
        dec_widths = []
        base_width = trial.suggest_categorical("base_width", [128, 256, 512])
        
        for i in range(enc_depth):
            width = int(base_width * (1 + i * 0.5))  # Gradually increase
            enc_widths.append(min(width, 1024))  # Cap at 1024
        
        for i in range(dec_depth):
            width = int(base_width * (1 + (dec_depth - i - 1) * 0.5))  # Gradually decrease
            dec_widths.append(min(width, 1024))
        
        cfg["model"]["encoder_layers"] = enc_widths
        cfg["model"]["decoder_layers"] = dec_widths
        
        # Generator configuration
        cfg["model"]["generator"] = {
            "rank": trial.suggest_categorical("generator_rank", [4, 8, 12, 16, 24])
        }
        
        # Mixture configuration
        if model_type == "linear_latent_mixture":
            cfg["model"]["mixture"] = {
                "K": trial.suggest_categorical("mixture_K", [2, 3, 4, 5]),
                "temperature": trial.suggest_float("mixture_temperature", 0.5, 2.0)
            }
        else:
            # remove any stale mixture block inherited from the template
            cfg["model"].pop("mixture", None)
        
        # Time warp configuration
        use_warp = trial.suggest_categorical("use_time_warp", [True, False])
        if use_warp:
            cfg["model"]["time_warp"] = {
                "enabled": True,
                "J_terms": trial.suggest_categorical("J_terms", [2, 3, 4, 5]),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256])
            }
        else:
            cfg["model"]["time_warp"] = {"enabled": False}
        
        # Activation and regularization
        cfg["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "tanh"])
        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.15, step=0.05)
        
        # Training hyperparameters
        cfg["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        
        # Optimizer choice
        cfg["training"]["optimizer"] = trial.suggest_categorical("optimizer", ["adamw"])
        
        # Batch size (power of 2 for efficiency)
        batch_power = trial.suggest_int("batch_power", 8, 12)  # 256 to 4096
        cfg["training"]["batch_size"] = 2 ** batch_power
        
        # Regularization for mixtures
        if model_type == "linear_latent_mixture":
            cfg["training"]["regularization"] = {
                "lambda_entropy": trial.suggest_float("lambda_entropy", 0.0, 0.0, step=0.01),
                "lambda_diversity": trial.suggest_float("lambda_diversity", 0.0, 0.0, step=0.01),
            }
        else:
            # Ensure regularization is set to zero for non-mixture models
            cfg["training"]["regularization"] = {
                "lambda_entropy": 0.0,
                "lambda_diversity": 0.0,
            }
        
        return cfg
    
    def run_trial(self, trial: optuna.Trial) -> float:
        """Execute a single trial and return validation loss."""
        config = self.suggest_config(trial)
        
        # Create save directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}"
        save_dir = Path(self.base_config["paths"]["model_save_dir"]) / "optuna" / f"{timestamp}_{trial_id}"
        ensure_directories(save_dir)
        
        try:
            # Set seed for reproducibility
            seed_everything(config.get("system", {}).get("seed", 42))
            
            # Create model
            model = create_model(config, self.device)
            
            # Set epochs based on Hyperband if available
            n_epochs = trial.user_attrs.get("n_epochs", config["training"].get("epochs", 100))
            config["training"]["epochs"] = n_epochs
            
            # Create pruning callback
            min_epochs = config["training"].get("hpo_min_epochs", 10)
            pruning_callback = OptunaPruningCallback(trial, min_epochs)
            
            # Log trial configuration
            self.logger.info(
                f"Trial {trial.number}: type={config['model']['type']}, "
                f"latent={config['model']['latent_dim']}, "
                f"lr={config['training']['learning_rate']:.2e}, "
                f"batch={config['training']['batch_size']}"
            )

            # build fresh datasets per worker to avoid CUDA-tensor pickling issues
            train_dataset = SequenceDataset(self.processed_dir, "train", config, self.device, self.norm_stats)
            val_dataset = SequenceDataset(self.processed_dir, "validation", config, self.device, self.norm_stats)

            trainer = PrunableTrainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=None,
                config=config,
                save_dir=save_dir,
                device=self.device,
                norm_helper=self.norm_helper,
                epoch_callback=pruning_callback
            )
            
            # Train and get best validation loss
            best_val_loss = trainer.train()
            
            # Save trial information
            trial.set_user_attr("config", config)
            trial.set_user_attr("best_epoch", trainer.best_epoch)
            save_json(config, save_dir / "config.json")
            
            self.logger.info(
                f"Trial {trial.number} completed: loss={best_val_loss:.6f}, "
                f"best_epoch={trainer.best_epoch}"
            )
            
            return best_val_loss
            
        except optuna.TrialPruned:
            self.logger.info(f"Trial {trial.number} pruned")
            raise
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return float("inf")
            
        finally:
            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class PrunableTrainer(Trainer):
    """Trainer that supports Optuna pruning callbacks."""
    
    def __init__(self, *args, epoch_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = epoch_callback
    
    def _run_training_loop(self):
        """Training loop with pruning support."""
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
            
            # Training epoch
            train_loss, train_metrics = self._train_epoch()
            
            # Validation
            val_loss, val_metrics = self._validate()
            
            # Update scheduler
            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch
            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Check for pruning
            if self.epoch_callback:
                loss_for_pruning = val_loss if self.has_validation else train_loss
                if self.epoch_callback(epoch, loss_for_pruning):
                    self.logger.info(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            
            # Early stopping logic
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


def optimize_hyperparameters(
    config_path: Path,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = None,
    use_hyperband: bool = True
) -> optuna.Study:
    """Run hyperparameter optimization."""
    logger = logging.getLogger(__name__)
    
    if study_name is None:
        study_name = f"lilan_hpo_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize tuner
    tuner = HyperparameterTuner(config_path)
    
    # Setup pruner
    if use_hyperband:
        base_config = load_json_config(config_path)
        min_epochs = base_config["training"].get("hpo_min_epochs", 10)
        max_epochs = base_config["training"].get("hpo_max_epochs", 100)
        pruner = HyperbandPruner(
            min_resource=min_epochs,
            max_resource=max_epochs,
            reduction_factor=3
        )
        logger.info(f"Using Hyperband pruner: min={min_epochs}, max={max_epochs}")
    else:
        pruner = None
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True
    )
    
    logger.info(f"Starting optimization: {n_trials} trials, {n_jobs} parallel jobs")
    
    # Run optimization
    study.optimize(tuner.run_trial, n_trials=n_trials, n_jobs=n_jobs)
    
    # Save results
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)
    
    best_config = study.best_trial.user_attrs.get("config", {})
    
    # Compute statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    results = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": best_config,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save results
    results_file = results_dir / f"{study_name}_results.json"
    save_json(results, results_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {results['best_value']:.6f}")
    print(f"Best trial: #{results['best_trial_number']}")
    print(f"Completed: {results['n_trials_completed']}, Pruned: {results['n_trials_pruned']}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {results_file}")
    
    return study