#!/usr/bin/env python3

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

"""
Main entry point for chemical kinetics neural network training.
Supports both standard training and hyperparameter optimization with Optuna.
Uses chunked HDF5 format for efficient data storage and loading.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import os
import hashlib
import warnings
import platform

# Check for OpenMP library conflicts on macOS
if platform.system() == "Darwin" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    # Only set this on macOS where the issue commonly occurs
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.warn(
        "Setting KMP_DUPLICATE_LIB_OK=TRUE for macOS compatibility. "
        "This may mask underlying library conflicts."
    )

import torch
import numpy as np

# Local imports
from utils.hardware import setup_device, optimize_hardware
from utils.utils import (
    setup_logging, 
    seed_everything, 
    ensure_directories,
    load_json_config,
    save_json
)
from data.preprocessor import DataPreprocessor
from data.dataset import HDF5Dataset
from models.model import create_model
from training.trainer import Trainer

# Only suppress specific known warnings
warnings.filterwarnings("ignore", message=".*torch.compile.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for cache validation.
    """
    # Extract all parts that affect data preprocessing
    relevant_config = {
        "raw_data_files": sorted(config["paths"]["raw_data_files"]),  # Sort for consistency
        "data": config["data"],
        "normalization": config["normalization"],
        "training": {
            "val_fraction": config["training"]["val_fraction"],
            "test_fraction": config["training"]["test_fraction"],
            "use_fraction": config["training"]["use_fraction"]
        }
    }
    
    # Convert to JSON string for hashing
    config_str = json.dumps(relevant_config, sort_keys=True)
    
    # Compute SHA256 hash
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ChemicalKineticsPipeline:
    """
    Complete training pipeline for chemical kinetics prediction.
    Uses chunked HDF5 format for optimal performance.
    """
    
    def __init__(self, config_path: Path):
        """Initialize the pipeline with configuration."""
        # Load configuration
        self.config = load_json_config(config_path)
        
        # Setup paths
        self.setup_paths()
        
        # Setup logging
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("Chemical Kinetics Neural Network Pipeline")
        self.logger.info(f"Configuration: {config_path}")
        
        # Set random seed for reproducibility
        seed = self.config["system"]["seed"]
        seed_everything(seed)
        
        # Setup hardware
        self.device = setup_device()
        optimize_hardware(self.config["system"], self.device)
        
        self.logger.info("Pipeline initialized with chunked HDF5 format for high-performance data loading")
        
    def setup_paths(self):
        """Create directory structure and setup paths."""
        paths = self.config["paths"]
        
        # Create run-specific directory
        model_type = self.config["model"]["type"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_save_dir = Path(paths["model_save_dir"]) / f"trained_model_{model_type}_{timestamp}"
        
        # Convert to Path objects
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        
        # Use environment variable if set, otherwise use config
        self.processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", paths["processed_data_dir"]))
        self.log_dir = Path(paths["log_dir"])
        
        # Create directories
        ensure_directories(self.processed_dir, self.run_save_dir, self.log_dir)
        
        # Define processed data paths
        self.processed_hdf5_file = self.processed_dir / paths.get("processed_hdf5_file", "preprocessed_data.h5")
        self.normalization_file = self.processed_dir / "normalization.json"
        self.config_hash_file = self.processed_dir / "config_hash.txt"
        
    def preprocess_data(self) -> bool:
        """
        Pre-process raw files to normalized chunked HDF5.
        """
        cfg_hash = compute_config_hash(self.config)
        
        # Check if preprocessed data already exists
        cache_ok = (
            self.processed_hdf5_file.exists()
            and self.normalization_file.exists()
            and self.config_hash_file.exists()
            and self.config_hash_file.read_text().strip() == cfg_hash
        )
        
        if cache_ok:
            self.logger.info("Using cached HDF5 data and normalization.")
            return False
        
        # Check raw files exist
        missing = [p for p in self.raw_data_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw data files: {missing}")
        
        # Process to HDF5 with integrated normalization
        self.logger.info("Starting HDF5 creation from raw files with normalization...")
        
        preprocessor = DataPreprocessor(
            raw_files=self.raw_data_files,
            output_dir=self.processed_dir,
            config=self.config
        )
        
        dataset_info = preprocessor.process_to_hdf5()
        
        # Check for errors
        if "error" in dataset_info:
            self.logger.error(f"Preprocessing failed: {dataset_info['error']}")
            sys.exit(1)
        
        # Check if any data was processed
        total_samples = dataset_info["total_samples"]
        if total_samples == 0:
            self.logger.error("No valid data processed from raw files. Exiting.")
            sys.exit(1)
        
        # Save config hash
        self.config_hash_file.write_text(cfg_hash)
        
        self.logger.info("Pre-processing complete.")
        return True
        
    def train_model(self):
        """Train the neural network model."""
        self.logger.info("Starting model training...")
        
        # Save the exact config used for this run
        save_json(self.config, self.run_save_dir / "run_config.json")

        # Load normalization stats (for potential denorm in inference)
        with open(self.normalization_file, 'r') as f:
            norm_stats = json.load(f)
        
        # Create model
        model = create_model(self.config, self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.config['model']['type'].upper()}")
        self.logger.info(f"Parameters: {total_params:,}")
        
        # Create datasets
        self.logger.info("Creating HDF5 datasets for high-performance loading...")
        
        train_dataset = HDF5Dataset(
            hdf5_path=self.processed_hdf5_file,
            split_name="train",
            config=self.config,
            device=self.device
        )
        
        val_dataset = HDF5Dataset(
            hdf5_path=self.processed_hdf5_file,
            split_name="validation",
            config=self.config,
            device=self.device
        )
        
        test_dataset = HDF5Dataset(
            hdf5_path=self.processed_hdf5_file,
            split_name="test",
            config=self.config,
            device=self.device
        )
        
        # Log dataset info
        train_info = train_dataset.get_batch_info()
        self.logger.info(f"HDF5 Dataset info: {train_info}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=self.config,
            save_dir=self.run_save_dir,
            device=self.device
        )
        
        # Train model
        best_val_loss = trainer.train()
        
        # Evaluate on test set
        test_loss = trainer.evaluate_test()
        
        self.logger.info(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        self.logger.info(f"Test loss: {test_loss:.6f}")
        
        # Save final results
        results = {
            "config_path": str(self.run_save_dir / "run_config.json"),
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "model_path": str(self.run_save_dir / "best_model.pt"),
            "training_time": trainer.total_training_time,
            "data_format": "hdf5_chunked"
        }
        
        save_json(results, self.run_save_dir / "results.json")
    
    def optimize_hyperparameters(self):
        """Run hyperparameter optimization using Optuna."""
        if not self.config["optuna"]["enabled"]:
            self.logger.info("Optuna optimization not enabled in config")
            return
        
        # Import here to avoid dependency if not using Optuna
        try:
            from optuna_optimization import OptunaOptimizer
        except ImportError:
            self.logger.error("Optuna not installed. Install with: pip install optuna")
            sys.exit(1)
        
        self.logger.info("Starting hyperparameter optimization with Optuna...")
        
        # Ensure data is preprocessed
        self.preprocess_data()
        
        # Create optimizer
        optimizer = OptunaOptimizer(
            base_config=self.config,
            preprocessed_hdf5_path=self.processed_hdf5_file,
            save_dir=self.run_save_dir,
            device=self.device
        )
        
        # Run optimization
        study = optimizer.optimize()
        
        # Train final model with best parameters
        self.logger.info("Training final model with best hyperparameters...")
        
        # Load best config
        best_config_path = self.run_save_dir / "optuna_study" / "best_config.json"
        self.config = load_json_config(best_config_path)
        
        # Train with best config
        self.train_model()
    
    def run(self, mode: str = "train"):
        """
        Execute the pipeline.
        
        Args:
            mode: "train" for standard training, "optimize" for hyperparameter optimization
        """
        try:
            # Step 1: Preprocess data (if needed)
            self.preprocess_data()
            
            # Step 2: Train or optimize
            if mode == "optimize":
                self.optimize_hyperparameters()
            else:
                self.train_model()
            
            self.logger.info("="*80)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_save_dir}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Chemical Kinetics Neural Network Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.jsonc"),
        help="Path to configuration file (default: config/config.jsonc)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "optimize"],
        default="train",
        help="Mode: train (standard training) or optimize (hyperparameter optimization)"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Check for json5 if using .jsonc file
    if args.config.suffix == ".jsonc":
        try:
            import json5
        except ImportError:
            print("Error: JSON with comments (.jsonc) requires json5 package", file=sys.stderr)
            print("Install it with: pip install json5", file=sys.stderr)
            sys.exit(1)
    
    # Run pipeline
    pipeline = ChemicalKineticsPipeline(args.config)
    pipeline.run(mode=args.mode)


if __name__ == "__main__":
    main()