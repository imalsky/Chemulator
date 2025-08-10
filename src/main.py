#!/usr/bin/env python3
"""
Main entry point for chemical kinetics neural network training and hyperparameter tuning.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
import sys
import time
from pathlib import Path
import torch
from typing import Dict, Union
import psutil


# Setup multiprocessing strategy early
import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from utils.hardware import setup_device, optimize_hardware
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json, load_json
from data.preprocessor import DataPreprocessor
from data.dataset import SequenceDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper


class ChemicalKineticsPipeline:
    """Training pipeline for chemical kinetics models."""
    
    def __init__(self, config_or_path: Union[Path, Dict]):
        """Initialize the pipeline."""
        if isinstance(config_or_path, (Path, str)):
            self.config = load_json_config(Path(config_or_path))
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError("config_or_path must be a Path, str, or dict")
        
        # Create directory tree
        self.setup_paths()
        
        # Setup logging with file handler
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Chemical Kinetics Pipeline initialized")
        
        # Set seed
        seed_everything(self.config.get("system", {}).get("seed", 42))
        
        # Setup hardware
        self.device = setup_device()
        optimize_hardware(self.config["system"], self.device)
    
    def setup_paths(self):
        """Create directory structure."""
        paths = self.config["paths"]
        
        # Create run directory
        model_type = self.config["model"]["type"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mix = self.config["model"].get("mixture", {}).get("K", 1)
        warp = self.config["model"].get("time_warp", {}).get("enabled", False)
        seq_tag = "seq" if self.config["data"].get("sequence_mode", False) else "row"
        extra = f"K{mix}_{'warp' if warp else 'nowarp'}_{seq_tag}"
        self.run_save_dir = Path(paths["model_save_dir"]) / f"{model_type}_{extra}_{timestamp}"
        
        # Convert paths
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        
        # Processed data directory
        base_processed_dir = Path(paths["processed_data_dir"])
        self.processed_dir = base_processed_dir / f"{seq_tag}_mode"
        
        self.log_dir = Path(paths["log_dir"])
        
        # Create directories
        ensure_directories(self.processed_dir, self.run_save_dir, self.log_dir)
    
    def _compute_data_hash(self) -> str:
        """
        Compute a hash of data-critical parameters that affect shard content or normalization.
        """
        import hashlib
        import json

        # Effective values with defaults matching the preprocessor
        norm_cfg = self.config.get("normalization", {})
        precfg = self.config.get("preprocessing", {})
        train_cfg = self.config.get("training", {})
        data_cfg = self.config.get("data", {})
        sys_cfg = self.config.get("system", {})

        # Variable-wise normalization methods (species/globals); time handled explicitly below
        methods = dict(norm_cfg.get("methods", {}))

        # Time normalization config (now included in the hash)
        time_norm_cfg = norm_cfg.get("time", {})
        time_norm_method = time_norm_cfg.get("method", None)
        time_norm_params = time_norm_cfg.get("params", {})

        time_var = data_cfg.get("time_variable")

        data_params = {
            # Inputs & schema
            "raw_files": sorted([str(f) for f in self.raw_data_files]),
            "species_variables": data_cfg.get("species_variables", []),
            "target_species_variables": data_cfg.get("target_species_variables", data_cfg.get("species_variables", [])),
            "global_variables": data_cfg.get("global_variables", []),
            "time_variable": time_var,

            # Sampling / supervision
            "sequence_mode": data_cfg.get("sequence_mode", False),
            "M_per_sample": data_cfg.get("M_per_sample", 16),

            # Fixed global time grid metadata
            "fixed_time_range": data_cfg.get("fixed_time_range", None),
            "grid_coverage_rtol": precfg.get("grid_coverage_rtol", 1e-6),

            # Preprocessing knobs that change outputs
            "min_value_threshold": precfg.get("min_value_threshold", 1e-30),
            "target_shard_bytes": precfg.get("target_shard_bytes", None),
            "trajectories_per_shard": precfg.get("trajectories_per_shard", None),
            "npz_compressed": precfg.get("npz_compressed", True),
            "time_hist_bins": precfg.get("time_hist_bins", 4096),  # affects tau0 estimate

            # Normalization behavior that changes normalization.json
            "default_norm_method": norm_cfg.get("default_method", "standard"),
            "time_norm_method": time_norm_method,
            "time_norm_params": time_norm_params,
            "epsilon": norm_cfg.get("epsilon", 1e-30),
            "min_std": norm_cfg.get("min_std", 1e-10),

            # Split policy (changes which sample lands in which split)
            "use_fraction": train_cfg.get("use_fraction", 1.0),
            "val_fraction": train_cfg.get("val_fraction", 0.0),
            "test_fraction": train_cfg.get("test_fraction", 0.0),

            # Seed controlling deterministic split hashing
            "seed": sys_cfg.get("seed", 42),

            # Dtype baked into saved arrays
            "system_dtype": sys_cfg.get("dtype", "float32"),
        }

        payload = json.dumps(data_params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    
    def preprocess_data(self):
        """Preprocess data if needed."""
        self.logger.info("Checking data preprocessing...")
        
        # Check if data already exists with matching hash
        current_hash = self._compute_data_hash()
        hash_file = self.processed_dir / "data_hash.json"
        
        regenerate = True
        if hash_file.exists():
            saved_hash_data = load_json(hash_file)
            if saved_hash_data.get("hash") == current_hash:
                self.logger.info("Data already preprocessed with matching hash. Skipping regeneration.")
                regenerate = False
            else:
                self.logger.info("Data hash mismatch. Regenerating data...")
        
        if regenerate:
            preprocessor = DataPreprocessor(
                raw_files=self.raw_data_files,
                output_dir=self.processed_dir,
                config=self.config
            )
            
            # Check for missing files
            missing = [p for p in self.raw_data_files if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing raw data files: {missing}")
            
            # Process to shards
            preprocessor.process_to_npy_shards()
            
            # Save the hash
            save_json({
                "hash": current_hash,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, hash_file)
            
            self.logger.info(f"Data preprocessing complete. Files saved to: {self.processed_dir}")
    
    def _log_memory_status(self):
        """Log current memory status."""
        mem = psutil.virtual_memory()
        self.logger.info(
            f"System memory: {mem.total/1024**3:.1f}GB total, "
            f"{mem.available/1024**3:.1f}GB available ({mem.percent:.1f}% used)"
        )
        
        if self.device.type == "cuda":
            idx = 0 if self.device.index is None else self.device.index
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
            used_mem = total_mem - free_mem
            self.logger.info(
                f"GPU memory: {total_mem/1024**3:.1f}GB total, "
                f"{free_mem/1024**3:.1f}GB free, "
                f"{used_mem/1024**3:.1f}GB used"
            )
    
    def train_model(self):
        """Train the neural network model."""
        mode_str = "sequence" if self.config["data"].get("sequence_mode", False) else "row-wise"
        self.logger.info(f"Starting model training in {mode_str} mode...")
        
        # Ensure data is preprocessed
        self.preprocess_data()
        
        # Save config for this run
        save_json(self.config, self.run_save_dir / "config.json")
        
        # Create model
        model = create_model(self.config, self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.config['model']['type']} - Parameters: {total_params:,}")
        
        # Load normalization stats
        norm_file = self.processed_dir / "normalization.json"
        if not norm_file.exists():
            raise FileNotFoundError(f"Normalization file missing: {norm_file}")
        norm_stats = load_json(norm_file)
        
        norm_helper = NormalizationHelper(
            stats=norm_stats,
            device=self.device,
            config=self.config
        )
        
        # Log memory status
        self._log_memory_status()
        
        # Create datasets
        self.logger.info("Loading datasets...")
        train_dataset = SequenceDataset(
            self.processed_dir, "train", self.config, self.device, norm_stats
        )
        
        val_dataset = None
        if self.config.get("training", {}).get("val_fraction", 0.0) > 0:
            val_dataset = SequenceDataset(
                self.processed_dir, "validation", self.config, self.device, norm_stats
            )
        
        test_dataset = None
        if self.config.get("training", {}).get("test_fraction", 0.0) > 0:
            test_dataset = SequenceDataset(
                self.processed_dir, "test", self.config, self.device, norm_stats
            )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            self.logger.info(f"Validation samples: {len(val_dataset)}")
        if test_dataset:
            self.logger.info(f"Test samples: {len(test_dataset)}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=self.config,
            save_dir=self.run_save_dir,
            device=self.device,
            norm_helper=norm_helper
        )
        
        # Train model
        best_val_loss = trainer.train()
        
        # Evaluate on test set
        test_loss = float("inf")
        if test_dataset:
            test_loss = trainer.evaluate_test()
        
        self.logger.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        if test_loss != float("inf"):
            self.logger.info(f"Test loss: {test_loss:.6f}")
        
        # Save results
        results = {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "model_path": str(self.run_save_dir / "best_model.pt"),
            "training_time": trainer.total_training_time,
            "best_epoch": trainer.best_epoch,
        }
        
        save_json(results, self.run_save_dir / "results.json")
        
        return results
    
    def run(self):
        """Execute the full training pipeline."""
        try:
            results = self.train_model()
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_save_dir}")
            return results
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chemical Kinetics Neural Network")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.jsonc"),
        help="Path to configuration file"
    )
    
    # Add mutually exclusive group for train/tune
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="Train a model using the configuration"
    )
    group.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    
    # Hyperparameter tuning specific arguments
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for hyperparameter optimization"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for optimization"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study"
    )
    parser.add_argument(
        "--no-hyperband",
        action="store_true",
        help="Disable Hyperband pruning"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Execute based on mode
    if args.train:
        # Regular training
        pipeline = ChemicalKineticsPipeline(args.config)
        pipeline.run()
        
    elif args.tune:
        # Hyperparameter optimization
        from hyperparameter_tuning import optimize_hyperparameters
        
        # Run optimization
        study = optimize_hyperparameters(
            config_path=args.config,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            study_name=args.study_name,
            use_hyperband=not args.no_hyperband
        )
        
        print(f"\nOptimization complete. Study saved as: {study.study_name}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()