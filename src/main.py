#!/usr/bin/env python3
"""
Main entry point for chemical kinetics neural network training.
Uses direct HDF5 to NPY shard conversion for optimal performance.
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

if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    warnings.warn("Setting KMP_DUPLICATE_LIB_OK=TRUE to allow multiple OpenMP libraries.")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from data.dataset import NPYDataset, create_dataloader
from data.device_prefetch import DevicePrefetchLoader
from models.model import create_model
from training.trainer import Trainer


# Suppress warnings
warnings.filterwarnings("ignore", ".*torch.compile.*")
warnings.filterwarnings("ignore", ".*flash attention.*")


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for cache validation.
    """
    # Extract all parts that affect data preprocessing
    relevant_config = {
        "raw_data_files": sorted(config["paths"]["raw_data_files"]),  # Sort for consistency
        "data": config["data"],  # Includes chunk_size
        "normalization": config["normalization"],  # Includes default_method
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
    Uses direct HDF5 to NPY shard conversion for optimal performance.
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
        
        self.logger.info("Pipeline initialized with NPY shard format for high-performance data loading")
        
    def setup_paths(self):
        """Create directory structure and setup paths."""
        paths = self.config["paths"]
        
        # Create run-specific directory
        model_type = self.config["model"]["type"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_save_dir = Path(paths["model_save_dir"]) / f"trained_model_{model_type}_{timestamp}"
        
        # Convert to Path objects
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]

        # If the job script sets $NPY_SHARD_DIR use that, otherwise fall back
        self.processed_dir = Path(os.getenv("NPY_SHARD_DIR", paths["processed_data_dir"]))
        self.log_dir = Path(paths["log_dir"])
        
        # Create directories
        ensure_directories(self.processed_dir, self.run_save_dir, self.log_dir)
        
        # Define processed data paths
        self.normalization_file = self.processed_dir / "normalization.json"
        self.config_hash_file = self.processed_dir / "config_hash.txt"
        
    def preprocess_data(self) -> bool:
        """
        Pre-process raw files directly to NPY shards.
        Always applies normalization during shard creation.
        """
        cfg_hash = compute_config_hash(self.config)
        # Check if NPY shards already exist
        npy_index_file = self.processed_dir / "shard_index.json"
        train_indices_file = self.processed_dir / "train_indices.npy"
        val_indices_file = self.processed_dir / "val_indices.npy"
        test_indices_file = self.processed_dir / "test_indices.npy"
        cache_ok = (
            npy_index_file.exists()
            and self.normalization_file.exists()
            and train_indices_file.exists()
            and val_indices_file.exists()
            and test_indices_file.exists()
            and self.config_hash_file.exists()
            and self.config_hash_file.read_text().strip() == cfg_hash
        )
        
        if cache_ok:
            self.logger.info("Using cached NPY shards and normalization data.")
            return False
        
        # Check raw files exist
        missing = [p for p in self.raw_data_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw data files: {missing}")
        
        # Process directly to NPY shards with integrated normalization
        self.logger.info("Starting direct NPY shard creation from raw files with normalization...")
        
        preprocessor = DataPreprocessor(
            raw_files=self.raw_data_files,
            output_dir=self.processed_dir,
            config=self.config
        )
        
        dataset_info = preprocessor.process_to_npy_shards()  # Now includes stats and norm
        
        # Check if any data was processed
        total_samples = len(dataset_info["train_indices"]) + len(dataset_info["val_indices"]) + len(dataset_info["test_indices"])
        if total_samples == 0:
            self.logger.error("No valid data processed from raw files. Exiting.")
            sys.exit(1)
        
        # Save splits as .npy files
        np.save(train_indices_file, np.array(dataset_info["train_indices"], dtype=np.int64))
        np.save(val_indices_file, np.array(dataset_info["val_indices"], dtype=np.int64))
        np.save(test_indices_file, np.array(dataset_info["test_indices"], dtype=np.int64))
        
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
        
        # Load splits
        splits = {}
        splits["train"] = np.load(self.processed_dir / "train_indices.npy")
        splits["validation"] = np.load(self.processed_dir / "val_indices.npy")
        splits["test"] = np.load(self.processed_dir / "test_indices.npy")
        
        # Create model
        model = create_model(self.config, self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.config['model']['type'].upper()}")
        self.logger.info(f"Parameters: {total_params:,}")
        
        # Create datasets
        self.logger.info("Creating NPY shard datasets for high-performance loading...")
        
        # Assume all data is normalized during preprocessing - zero runtime normalization overhead
        self.logger.info("Using pre-normalized shards")
        
        train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=splits["train"],
            config=self.config,
            device=self.device,
            split_name="train"
        )
        
        val_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=splits["validation"],
            config=self.config,
            device=self.device,
            split_name="validation"
        )
        
        test_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=splits["test"],
            config=self.config,
            device=self.device,
            split_name="test"
        )
        
        # Log dataset info
        train_info = train_dataset.get_batch_info()
        self.logger.info(f"NPY Dataset info: {train_info}")
        
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
            "data_format": "npy_shards"
        }
        
        save_json(results, self.run_save_dir / "results.json")
    
    def run(self):
        """Execute the complete pipeline."""
        try:
            # Step 1: Preprocess data (if needed)
            self.preprocess_data()
            
            # Step 2: Train model
            self.train_model()
            
            self.logger.info("="*80)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Trained model and results saved in: {self.run_save_dir}")
            
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
    pipeline.run()


if __name__ == "__main__":
    main()