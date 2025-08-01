#!/usr/bin/env python3
"""
Hyperparameter tuning for chemical kinetics models using Optuna.
Simplified to target the LiLaN family (linear_latent / linear_latent_mixture)
and sequence-mode datasets. No DeepONet/SIREN here — this is the fast, stable path.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
import json

from main import ChemicalKineticsPipeline
from utils.hardware import setup_device, optimize_hardware
from data.dataset import SequenceDataset, NPYDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper
from utils.utils import (seed_everything, ensure_directories, 
                        load_json_config, save_json, load_json)


class OptunaPruningCallback:
    """Report intermediate values to Optuna and allow pruning after a warmup."""
    def __init__(self, trial: optuna.Trial, min_epochs: int = 10):
        self.trial = trial
        self.min_epochs = min_epochs
        
    def __call__(self, epoch: int, val_loss: float) -> bool:
        self.trial.report(val_loss, epoch)
        if epoch < self.min_epochs:
            return False
        return self.trial.should_prune()


class OptunaTrialRunner:
    """Prepares data once and executes trials with modified configs."""
    def __init__(self, base_config_path: Path):
        self.base_config_path = base_config_path
        self.base_config = load_json_config(base_config_path)
        self.device = setup_device()
        self.logger = logging.getLogger(__name__)
        self._pipelines = {}
        self._prepare_current_mode()

    def _prepare_current_mode(self):
        """Preprocess data for the current prediction mode in the base config."""
        mode = self.base_config["prediction"]["mode"]
        self.logger.info(f"Preparing data for '{mode}' mode...")
        pipeline = ChemicalKineticsPipeline(self.base_config)
        pipeline.normalize_only()
        self._pipelines[mode] = OptunaPipeline(self.base_config, pipeline.processed_dir)

    def run_trial(self, trial: optuna.Trial) -> float:
        """Configure and run a single trial."""
        config = suggest_lilan_config(trial, self.base_config)
        prediction_mode = config["prediction"]["mode"]
        pipeline = self._pipelines[prediction_mode]
        return pipeline.execute_trial(config, trial)


class OptunaPipeline:
    """Loads datasets and runs training for a specific processed_dir."""
    def __init__(self, config: Dict[str, Any], processed_dir: Path):
        self.config = config
        self.device = setup_device()
        self.logger = logging.getLogger(f"OptunaPipeline_{config['prediction']['mode']}")
        self.processed_dir = processed_dir
        self.model_save_root = Path(self.config["paths"]["model_save_dir"])

        # Normalization stats
        norm_stats_path = self.processed_dir / "normalization.json"
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found in {norm_stats_path}")
        norm_stats = load_json(norm_stats_path)
        self.norm_helper = NormalizationHelper(
            stats=norm_stats,
            device=self.device,
            species_vars=self.config["data"]["species_variables"],
            global_vars=self.config["data"]["global_variables"],
            time_var=self.config["data"]["time_variable"],
            config=self.config
        )

        self._load_datasets()

    def _load_datasets(self):
        """Choose SequenceDataset when sequence-mode shards are present."""
        index_path = self.processed_dir / "shard_index.json"
        with open(index_path) as f:
            shard_index = json.load(f)
        is_sequence = bool(shard_index.get("sequence_mode", False))

        self.logger.info(f"Loading datasets from: {self.processed_dir} (sequence_mode={is_sequence})")

        if is_sequence:
            self.train_dataset = SequenceDataset(self.processed_dir, "train", self.config, self.device)
            self.val_dataset   = SequenceDataset(self.processed_dir, "validation", self.config, self.device)
        else:
            # Fallback to legacy row-wise dataset if needed
            self.train_dataset = NPYDataset(self.processed_dir, "train", self.config, self.device)
            self.val_dataset   = NPYDataset(self.processed_dir, "validation", self.config, self.device)

        self.logger.info(f"Datasets loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

    def execute_trial(self, config: Dict[str, Any], trial: optuna.Trial) -> float:
        """Run training for a trial with pruning and return best val loss."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{trial.number:04d}_{config['prediction']['mode']}"
        save_dir = self.model_save_root / "optuna" / f"{timestamp}_{trial_id}"
        ensure_directories(save_dir)

        try:
            seed_everything(config["system"]["seed"])
            optimize_hardware(config["system"], self.device)
            model = create_model(config, self.device)

            # Hyperband sets epochs via user_attrs; default to hpo_max_epochs
            n_epochs = trial.user_attrs.get("n_epochs", config["training"]["hpo_max_epochs"])
            config["training"]["epochs"] = n_epochs

            min_epochs = config["training"]["hpo_min_epochs"]
            pruning_callback = OptunaPruningCallback(trial, min_epochs)

            # Log this trial
            self.logger.info(f"Trial {trial.number}: epochs={n_epochs} lr={config['training']['learning_rate']:.2e} "
                             f"type={config['model']['type']} latent={config['model']['latent_dim']} "
                             f"K={config['model'].get('mixture',{}).get('K','-')} "
                             f"rank={config['model'].get('generator',{}).get('rank','-')} "
                             f"warp={config['model'].get('time_warp',{}).get('enabled', False)}")

            trainer = PrunableTrainer(
                model=model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                test_dataset=None,
                config=config,
                save_dir=save_dir,
                device=self.device,
                norm_helper=self.norm_helper,
                epoch_callback=pruning_callback
            )

            best_val_loss = trainer.train()

            trial.set_user_attr("full_config", config)
            trial.set_user_attr("final_lr", trainer.optimizer.param_groups[0]['lr'])
            save_json(config, save_dir / "config.json")

            self.logger.info(f"Trial {trial.number} done. Best loss={best_val_loss:.6f} "
                             f"Final LR={trainer.optimizer.param_groups[0]['lr']:.2e}")
            return best_val_loss

        except optuna.TrialPruned:
            self.logger.info(f"Trial {trial.number} pruned.")
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return float("inf")
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class PrunableTrainer(Trainer):
    """Trainer with an epoch callback hook for Optuna pruning."""
    def __init__(self, *args, epoch_callback: Optional[Callable[[int, float], bool]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = epoch_callback
        
    def _run_training_loop(self):
        best_train_loss = float("inf")
        for epoch in range(1, self.train_config["epochs"] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._validate()

            if self.scheduler and not self.scheduler_step_on_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.has_validation:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.total_training_time += epoch_time
            self._log_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)

            if self.epoch_callback:
                loss_for_pruning = val_loss if self.has_validation and val_loss != float("inf") else train_loss
                if self.epoch_callback(epoch, loss_for_pruning):
                    self.logger.info(f"Trial pruned at epoch {epoch} with loss {loss_for_pruning:.6f}")
                    raise optuna.TrialPruned()

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


def suggest_lilan_config(trial: optuna.Trial, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a LiLaN (Linear Latent) config for this trial.
    We keep the search space tight and relevant:
      - type: linear_latent or linear_latent_mixture
      - latent_dim
      - generator.rank
      - mixture.K (if mixture)
      - time_warp.enabled + J_terms
      - encoder/decoder depth/width
      - learning rate and weight decay
      - regularization weights for mixture (entropy/diversity)
    """
    cfg = copy.deepcopy(base_config)

    # Always absolute mode for this benchmark
    cfg["prediction"]["mode"] = "absolute"

    # Ensure sequence mode stays on — HPO shouldn’t flip preprocessing mode
    cfg["data"]["sequence_mode"] = True

    # Model family choice: single generator vs mixture
    model_type = trial.suggest_categorical("model_type", ["linear_latent", "linear_latent_mixture"])
    cfg["model"]["type"] = model_type

    # Core sizes
    cfg["model"]["latent_dim"] = trial.suggest_categorical("latent_dim", [48, 64, 96])

    # Encoder/decoder depths (compact)
    enc_depth = trial.suggest_int("enc_depth", 2, 3)
    dec_depth = trial.suggest_int("dec_depth", 2, 3)

    def sample_width(tag: str) -> int:
        return trial.suggest_categorical(tag, [128, 192, 256, 320])

    cfg["model"]["encoder_layers"] = [sample_width(f"enc_w{i}") for i in range(enc_depth)]
    cfg["model"]["decoder_layers"] = [sample_width(f"dec_w{i}") for i in range(dec_depth)]

    # Generator rank
    cfg["model"]["generator"] = {"rank": trial.suggest_categorical("generator_rank", [4, 8, 12, 16])}

    # Mixture settings
    if model_type == "linear_latent_mixture":
        cfg["model"]["mixture"] = {"K": trial.suggest_categorical("mixture_K", [2, 3, 4])}
    else:
        cfg["model"].pop("mixture", None)

    # Time warp
    use_warp = trial.suggest_categorical("time_warp", [True, False])
    if use_warp:
        cfg["model"]["time_warp"] = {"enabled": True, "J_terms": trial.suggest_categorical("warp_J", [2, 3, 4])}
    else:
        cfg["model"]["time_warp"] = {"enabled": False}

    # Activations / dropout
    cfg["model"]["activation"] = trial.suggest_categorical("activation", ["gelu", "silu"])
    cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.10, step=0.05)

    # Training hyperparams
    cfg["training"]["learning_rate"] = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Mixture regularization weights (used by Trainer hooks)
    cfg["training"]["regularization"] = {
        "lambda_entropy": trial.suggest_float("lambda_entropy", 0.0, 0.05),
        "lambda_diversity": trial.suggest_float("lambda_diversity", 0.0, 0.05),
    }

    # Keep batch size and sequence sampling fixed to avoid data reprocessing mid-HPO
    # You can tune batch size if VRAM allows, but be consistent across trials.
    return cfg


def optimize(config_path: Path,
             n_trials: int = 25,
             n_jobs: int = 1,
             study_name: str = "chemulator_hpo",
             pruner: Optional[optuna.pruners.BasePruner] = None):
    """Run Optuna with Hyperband on the LiLaN search space."""
    logger = logging.getLogger(__name__)
    base_config = load_json_config(config_path)

    # Prepare data once
    trial_runner = OptunaTrialRunner(config_path)
    objective = trial_runner.run_trial

    # Hyperband for budget allocation across epochs
    if pruner is None:
        min_resource = base_config["training"]["hpo_min_epochs"]
        max_resource = base_config["training"]["hpo_max_epochs"]
        pruner = HyperbandPruner(min_resource=min_resource, max_resource=max_resource, reduction_factor=3)
        logger.info(f"Using Hyperband pruner: min={min_resource} max={max_resource} r=3")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=pruner,
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True
    )

    logger.info(f"Starting optimization: n_trials={n_trials}, n_jobs={n_jobs}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Save results
    results_dir = Path("optuna_results")
    ensure_directories(results_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    best_config = study.best_trial.user_attrs.get("full_config", {})
    if not best_config:
        logger.warning("Could not retrieve full config from user_attrs.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    epoch_distribution = {}
    for t in completed + pruned:
        n_epochs = t.user_attrs.get("n_epochs", "unknown")
        epoch_distribution[n_epochs] = epoch_distribution.get(n_epochs, 0) + 1

    best_results = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_config": best_config,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "epoch_distribution": epoch_distribution,
        "best_trial_final_lr": study.best_trial.user_attrs.get("final_lr", "unknown"),
        "study_db": f"{study_name}.db"
    }

    save_json(best_results, results_dir / f"best_config_{study_name}_{timestamp}.json")

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_results['best_value']:.6f}")
    print(f"Best trial final LR: {best_results['best_trial_final_lr']}")
    print(f"Trials: {best_results['n_trials_completed']} completed, {best_results['n_trials_pruned']} pruned")
    print("\nEpoch distribution:")
    for epochs, count in sorted(epoch_distribution.items(), key=lambda x: (str(x[0]), x[1])):
        print(f"  {epochs} epochs: {count} trials")
    print("\nBest parameters:")
    for key, value in best_results['best_params'].items():
        print(f"  {key}: {value}")

    return study
