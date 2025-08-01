#!/usr/bin/env python3
import logging
import sys
import time
from pathlib import Path
import torch
from typing import Dict, Any, Union
import hashlib
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y%m%d_%H%M%S",
    stream=sys.stdout
)

import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info("SUCCESS: Set multiprocessing sharing strategy to 'file_system'.")
except RuntimeError:
    logging.warning("Could not set multiprocessing sharing strategy.")

from utils.hardware import setup_device, optimize_hardware
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json, load_json
from data.preprocessor import DataPreprocessor
from data.dataset import NPYDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper


class ChemicalKineticsPipeline:
    """Optimized training pipeline for A100 GPU."""
    def __init__(self, config_or_path: Union[Path, Dict[str, Any]]):
        """Initialize the pipeline."""
        if isinstance(config_or_path, (Path, str)):
            self.config = load_json_config(Path(config_or_path))
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError(f"config_or_path must be a Path, str, or dict")
        
        # Get prediction mode
        self.prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
        
        # Setup paths
        self.setup_paths()
        
        # Setup logging
        log_file = self.log_dir / f"pipeline_{self.prediction_mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Chemical Kinetics Pipeline initialized - Mode: {self.prediction_mode}")
        
        seed_everything(self.config["system"]["seed"])
        
        # Setup hardware
        self.device = setup_device()
        optimize_hardware(self.config["system"], self.device)
        
    def setup_paths(self):
        """Create directory structure."""
        paths = self.config["paths"]
        
        # Create run directory
        model_type = self.config["model"]["type"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_save_dir = Path(paths["model_save_dir"]) / f"{model_type}_{self.prediction_mode}_{timestamp}"
        
        # Convert paths
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        
        # Mode-specific processed directory
        base_processed_dir = Path(paths["processed_data_dir"])
        self.processed_dir = base_processed_dir / f"mode_{self.prediction_mode}"
        
        self.log_dir = Path(paths["log_dir"])
        
        # Create directories
        ensure_directories(self.processed_dir, self.run_save_dir, self.log_dir)

    def _compute_data_hash(self) -> str:
        """
        Compute a hash of data-critical parameters.
        """
        data_params = {
            "raw_files": sorted([str(f) for f in self.raw_data_files]),
            "species_variables": self.config["data"]["species_variables"],
            "global_variables": self.config["data"]["global_variables"],
            "time_variable": self.config["data"]["time_variable"],
            "min_value_threshold": self.config["preprocessing"]["min_value_threshold"],
            "use_fraction": self.config["training"]["use_fraction"],
            "prediction_mode": self.prediction_mode,
            "normalization_methods": self.config["normalization"].get("methods", {}),
            "default_norm_method": self.config["normalization"]["default_method"],
        }
        
        hash_str = json.dumps(data_params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def normalize_only(self):
        """Run only the data preprocessing and normalization step."""
        self.logger.info("Running data normalization only...")
        
        # Check if data already exists
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
            
            missing = [p for p in self.raw_data_files if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing raw data files: {missing}")
            
            # Process to shards and compute normalization
            preprocessor.process_to_npy_shards()
            
            # Save the hash
            save_json({
                "hash": current_hash,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": self.prediction_mode
            }, hash_file)
            
            self.logger.info(f"Data normalization complete. Files saved to: {self.processed_dir}")

    def preprocess_data(self):
        """Preprocess data if needed."""
        self.logger.info(f"Preprocessing data for {self.prediction_mode} mode...")
        self.normalize_only()

    def _log_memory_status(self):
        """Log current memory status."""
        import psutil
        
        # CPU memory
        mem = psutil.virtual_memory()
        self.logger.info(f"System memory: {mem.total/1024**3:.1f}GB total, "
                         f"{mem.available/1024**3:.1f}GB available ({mem.percent:.1f}% used)")
        
        # GPU memory
        if self.device.type == "cuda":
            free_mem, total_mem = torch.cuda.mem_get_info(self.device.index)
            used_mem = total_mem - free_mem
            self.logger.info(f"GPU memory: {total_mem/1024**3:.1f}GB total, "
                             f"{free_mem/1024**3:.1f}GB free, "
                             f"{used_mem/1024**3:.1f}GB used")

    def train_model(self):
        """Train the neural network model."""
        self.logger.info("Starting model training...")

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
        norm_stats = load_json(self.processed_dir / "normalization.json")
        norm_helper = NormalizationHelper(
            norm_stats,
            self.device,
            self.config["data"]["species_variables"],
            self.config["data"]["global_variables"],
            self.config["data"]["time_variable"],
            self.config
        )
        
        # Log memory status before creating datasets
        self._log_memory_status()

        # Create datasets
        train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            split_name="train",
            config=self.config,
            device=self.device
        )
        
        # Log cache status to prevent crash
        cache_info = train_dataset.get_cache_info()
        if cache_info.get("type") == "gpu":
            self.logger.info(f"GPU caching active for '{train_dataset.split_name}': {cache_info.get('size_gb', 0):.1f}GB loaded.")
        elif cache_info.get("type") == "cpu":
            self.logger.warning(f"CPU fallback active for '{train_dataset.split_name}'. Reason: {cache_info.get('message', 'N/A')}")
        else:
            self.logger.error(f"Cache status error for '{train_dataset.split_name}': {cache_info.get('status', 'unknown')}")

        val_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            split_name="validation",
            config=self.config,
            device=self.device
        ) if self.config["training"]["val_fraction"] > 0 else None
        
        test_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            split_name="test",
            config=self.config,
            device=self.device
        ) if self.config["training"]["test_fraction"] > 0 else None
        
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
        test_loss = trainer.evaluate_test()
        
        self.logger.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Test loss: {test_loss:.6f}")
        
        # Save results
        results = {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "model_path": str(self.run_save_dir / "best_model.pt"),
            "training_time": trainer.total_training_time,
        }
        
        save_json(results, self.run_save_dir / "results.json")
    
    def run(self):
        """Execute the full training pipeline."""
        try:
            self.train_model()
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_save_dir}")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Chemical Kinetics Neural Network Training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.jsonc"),
        help="Path to configuration file"
    )
    
    # Operation mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--normalize",
        action="store_true",
        help="Only preprocess and normalize the data"
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a model using the configuration"
    )
    mode_group.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    
    # Additional arguments
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter optimization"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="chemical_kinetics_opt",
        help="Name for Optuna study"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Execute based on mode
    if args.normalize:
        # Just normalize data
        pipeline = ChemicalKineticsPipeline(args.config)
        pipeline.normalize_only()
        print("\nData normalization complete!")
        
    elif args.train:
        # Train model
        pipeline = ChemicalKineticsPipeline(args.config)
        pipeline.run()
        
        
    elif args.tune:
            try:
                import optuna
            except ImportError:
                print("Error: The 'optuna' package is not installed",file=sys.stderr)
                sys.exit(1)
            
            from hyperparameter_tuning import optimize
            
            print(f"Starting hyperparameter optimization with {args.trials} trials...")
            study = optimize(
                config_path=args.config,
                n_trials=args.trials,
                n_jobs=1,
                study_name=args.study_name
            )
            
            # Print results
            print("\n" + "="*60)
            print("Optimization Complete")
            print("="*60)
            print(f"Best validation loss: {study.best_value:.6f}")
            print(f"Best trial: {study.best_trial.number}")
            print("\nBest parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()