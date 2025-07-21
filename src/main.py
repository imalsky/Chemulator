#!/usr/bin/env python3
import logging
import sys
import time
from pathlib import Path
import shutil

import numpy as np

from utils.hardware import setup_device, optimize_hardware
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json, load_json
from data.preprocessor import DataPreprocessor
from data.dataset import NPYDataset
from models.model import create_model
from training.trainer import Trainer
import hashlib
import json
from data.normalizer import NormalizationHelper


class ChemicalKineticsPipeline:
    """Simplified training pipeline for chemical kinetics prediction."""
    
    def __init__(self, config_path: Path):
        # Load configuration
        self.config = load_json_config(config_path)
        
        # Setup paths
        self.setup_paths()
        
        # Setup logging
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Chemical Kinetics Pipeline - Config: {config_path}")
        
        # Set random seed
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
        self.run_save_dir = Path(paths["model_save_dir"]) / f"{model_type}_{timestamp}"
        
        # Convert paths
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        self.processed_dir = Path(paths["processed_data_dir"])
        self.log_dir = Path(paths["log_dir"])
        
        # Create directories
        ensure_directories(self.processed_dir, self.run_save_dir, self.log_dir)

    def _clean_old_shards(self):
        """Remove old shard files from the processed directory."""
        self.logger.info("Cleaning old shard files...")
        
        # Remove all .npy and .npz shard files
        shard_patterns = ["shard_*.npy", "shard_*.npz"]
        removed_count = 0
        
        for pattern in shard_patterns:
            for shard_file in self.processed_dir.glob(pattern):
                try:
                    shard_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {shard_file}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} old shard files")
        
        # Also remove old index files to ensure clean state
        old_files = ["normalization.json", "shard_index.json", 
                     "train_indices.npy", "val_indices.npy", "test_indices.npy"]
        for filename in old_files:
            filepath = self.processed_dir / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove {filepath}: {e}")

    def preprocess_data(self):
        """Pre-process raw files to normalized NPY shards."""
        # Check if already processed
        required_files = [
            self.processed_dir / "normalization.json",
            self.processed_dir / "shard_index.json",
            self.processed_dir / "train_indices.npy",
            self.processed_dir / "val_indices.npy",
            self.processed_dir / "test_indices.npy",
            self.processed_dir / "config_hash.json"
        ]
        
        raw_files_info = {
            "files": [str(f) for f in self.raw_data_files],
            "sizes": [f.stat().st_size if f.exists() else 0 for f in self.raw_data_files],
            "mtimes": [f.stat().st_mtime if f.exists() else 0 for f in self.raw_data_files]
        }
        
        relevant_config = {
            "data": self.config["data"],
            "preprocessing": self.config["preprocessing"],
            "normalization": self.config["normalization"],
            "prediction": self.config["prediction"],
            "training": {
                "val_fraction": self.config["training"]["val_fraction"],
                "test_fraction": self.config["training"]["test_fraction"],
                "use_fraction": self.config["training"]["use_fraction"]
            },
            "raw_files": raw_files_info
        }
        config_str = json.dumps(relevant_config, sort_keys=True, default=lambda x: f"{x:.20e}" if isinstance(x, float) else x)
        current_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        
        if all(f.exists() for f in required_files):
            # Load saved hash and compare
            saved_hash = load_json(self.processed_dir / "config_hash.json").get("hash")
            if saved_hash == current_hash:
                self.logger.info("Using existing preprocessed data (config matches)")
                return
            else:
                self.logger.info("Config has changed; regenerating preprocessed data")
                # Clean old shards before regenerating
                self._clean_old_shards()
        else:
            self.logger.info("Preprocessed data missing; generating new data")
            # Ensure directory is clean for first-time processing
            if self.processed_dir.exists() and any(self.processed_dir.glob("shard_*.n*")):
                self._clean_old_shards()
        
        # Check raw files exist
        missing = [p for p in self.raw_data_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw data files: {missing}")
        
        # Process data
        self.logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor(
            raw_files=self.raw_data_files,
            output_dir=self.processed_dir,
            config=self.config
        )
        
        split_indices = preprocessor.process_to_npy_shards()
        
        # Save the new hash
        save_json({"hash": current_hash}, self.processed_dir / "config_hash.json")
        
        # Verify we have data
        total_samples = sum(len(indices) for indices in split_indices.values())
        if total_samples == 0:
            raise ValueError("No valid data processed from raw files")
            
    def train_model(self):
        """Train the neural network model with data loader cache warm-up."""
        self.logger.info("Starting model training...")

        # Enforce mode-model compatibility before proceeding
        prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
        model_type = self.config["model"]["type"]
        if prediction_mode == "ratio" and model_type != "deeponet":
            raise ValueError(f"Prediction mode 'ratio' is only compatible with model type 'deeponet', but '{model_type}' was specified.")

        # Save config for this run
        save_json(self.config, self.run_save_dir / "config.json")

        # Create model
        model = create_model(self.config, self.device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.config['model']['type']} - Parameters: {total_params:,}")

        # Load normalization stats and create helper
        norm_stats = load_json(self.processed_dir / "normalization.json")
        norm_helper = NormalizationHelper(
            norm_stats,
            self.device,
            self.config["data"]["species_variables"],
            self.config["data"]["global_variables"],
            self.config["data"]["time_variable"],
            self.config
        )

        # Create datasets
        train_indices = np.load(self.processed_dir / "train_indices.npy")
        train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=train_indices,
            config=self.config,
            device=self.device,
            split_name="train"
        )
        
        val_indices = np.load(self.processed_dir / "val_indices.npy")
        val_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=val_indices,
            config=self.config,
            device=self.device,
            split_name="validation"
        ) if len(val_indices) > 0 else None
        
        test_indices = np.load(self.processed_dir / "test_indices.npy")
        test_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            indices=test_indices,
            config=self.config,
            device=self.device,
            split_name="test"
        ) if len(test_indices) > 0 else None
        
        # Initialize trainer with norm_helper
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
        if trainer.train_loader and trainer.train_loader.num_workers > 0:
            self.logger.info(f"Warming up data loader cache with {trainer.train_loader.num_workers} workers...")
            start_warmup = time.time()
            for _ in trainer.train_loader:
                break # Iterating once is enough to trigger a parallel read.
            self.logger.info(f"Cache warmup complete in {time.time() - start_warmup:.2f}s.")

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
        """Execute the pipeline."""
        try:
            # Step 1: Preprocess data
            self.preprocess_data()
            
            # Step 2: Train model
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
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna hyperparameter optimization trials (default: normal training)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="chemical_kinetics_opt",
        help="Name for Optuna study (used with --trials)"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Run with or without hyperparameter optimization
    if args.trials:
        # Ensure optuna is installed
        try:
            import optuna
        except ImportError:
            print("Installing optuna...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        
        # Import and run optimization
        from hyperparameter_tuning import optimize
        
        print(f"Starting hyperparameter optimization with {args.trials} trials...")
        study = optimize(
            config_path=args.config,
            n_trials=args.trials,
            n_jobs=1,  # Always use 1 for GPU training
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
        
        # Show trial statistics
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"\nTrials: {completed} completed, {pruned} pruned")
        print(f"\nBest configuration saved to: optuna_results/")
        
    else:
        # Normal training
        pipeline = ChemicalKineticsPipeline(args.config)
        pipeline.run()


if __name__ == "__main__":
    main()