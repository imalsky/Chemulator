#!/usr/bin/env python3
import logging
import sys
import time
from pathlib import Path
import torch
from typing import Dict, Any, Union
import hashlib
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

# 2. Change the multiprocessing sharing strategy to prevent /dev/shm crashes.
import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info("SUCCESS: Set multiprocessing sharing strategy to 'file_system'.")
except RuntimeError:
    logging.warning("Could not set multiprocessing sharing strategy (already set or not supported).")

import numpy as np
from utils.hardware import setup_device, optimize_hardware
from utils.utils import setup_logging, seed_everything, ensure_directories, load_json_config, save_json, load_json
from data.preprocessor import DataPreprocessor
from data.dataset import NPYDataset
from models.model import create_model
from training.trainer import Trainer
from data.normalizer import NormalizationHelper


class ChemicalKineticsPipeline:
    """Training pipeline for chemical kinetics prediction."""
    def __init__(self, config_or_path: Union[Path, Dict[str, Any]]):
        """
        Initialize the pipeline with either a config file path or a config dictionary.
        
        Args:
            config_or_path: Either a Path to a config file or a config dictionary
        """
        if isinstance(config_or_path, (Path, str)):
            self.config = load_json_config(Path(config_or_path))
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError(f"config_or_path must be a Path, str, or dict, not {type(config_or_path)}")
        
        # Get prediction mode
        self.prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
        
        # Setup paths with mode-specific directories
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
        """Create directory structure with mode-specific paths."""
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
        Only includes parameters that affect the actual data content.
        """
        data_params = {
            "raw_files": sorted([str(f) for f in self.raw_data_files]),
            "species_variables": self.config["data"]["species_variables"],
            "global_variables": self.config["data"]["global_variables"],
            "time_variable": self.config["data"]["time_variable"],
            "min_value_threshold": self.config["preprocessing"]["min_value_threshold"],
            "use_fraction": self.config["training"]["use_fraction"],
            "prediction_mode": self.prediction_mode,
            "epsilon": self.config["normalization"]["epsilon"],
            "normalization_methods": self.config["normalization"].get("methods", {}),
            "default_norm_method": self.config["normalization"]["default_method"],
        }
        
        # Create stable JSON string
        hash_str = json.dumps(data_params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def normalize_only(self):
            """Run only the data preprocessing and normalization step."""
            self.logger.info("Running data normalization only...")
            
            # Check if data already exists with correct hash
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
                    self._clean_all_processed_data()
            
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

    def _generate_or_validate_splits(self):
        """Generate or validate train/val/test split indices."""
        split_config = {
            "val_fraction": self.config["training"]["val_fraction"],
            "test_fraction": self.config["training"]["test_fraction"],
            "use_fraction": self.config["training"]["use_fraction"],
            "seed": self.config["system"]["seed"]
        }
        current_split_hash = hashlib.sha256(
            json.dumps(split_config, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]
        
        split_hash_path = self.processed_dir / "split_hash.json"
        
        regenerate_splits = True
        if split_hash_path.exists():
            saved_split_hash = load_json(split_hash_path).get("hash")
            if saved_split_hash == current_split_hash:
                self.logger.info("Split configuration matches. Reusing existing splits.")
                regenerate_splits = False
        
        if regenerate_splits:
            self.logger.info("Generating new train/val/test splits...")
            preprocessor = DataPreprocessor(
                raw_files=self.raw_data_files,
                output_dir=self.processed_dir,
                config=self.config
            )
            preprocessor.generate_split_indices()
            save_json({"hash": current_split_hash}, split_hash_path)

    def _clean_all_processed_data(self):
        """Remove ALL processed files."""
        self.logger.info("Cleaning ALL old processed files...")
        if not self.processed_dir.exists():
            return
        
        patterns = ["shard_*.npy", "shard_*.npz", "*.json", "*_indices.npy"]
        removed_count = 0
        
        for pattern in patterns:
            for file in self.processed_dir.glob(pattern):
                try:
                    file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file}: {e}")
        
        self.logger.info(f"Removed {removed_count} old files from {self.processed_dir}")

    def preprocess_data(self):
        """Preprocess data with proper hash checking."""
        self.logger.info(f"Preprocessing data for {self.prediction_mode} mode...")
        self.normalize_only()

    def train_model(self):
        """Train the neural network model."""
        self.logger.info("Starting model training...")

        # Ensure data is preprocessed
        self.preprocess_data()

        # Enforce mode-model compatibility
        prediction_mode = self.config.get("prediction", {}).get("mode", "absolute")
        model_type = self.config["model"]["type"]
        if prediction_mode == "ratio" and model_type != "deeponet":
            raise ValueError(
                f"Prediction mode 'ratio' is only compatible with model type 'deeponet', "
                f"but '{model_type}' was specified."
            )

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
        train_dataset = NPYDataset(
            shard_dir=self.processed_dir,
            split_name="train",
            config=self.config,
            device=self.device
        )
        
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
        
        # Warm up cache
        _ = train_dataset[0]

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
    """Main entry point with multiple operation modes."""
    import argparse
    parser = argparse.ArgumentParser(description="Chemical Kinetics Neural Network Training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.jsonc"),
        help="Path to configuration file"
    )
    
    # Operation mode arguments (mutually exclusive)
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
    
    # Hyperparameter tuning specific arguments
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
        # Run hyperparameter optimization
        try:
            import optuna
        except ImportError:
            print("Installing optuna...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        
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
        
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"\nTrials: {completed} completed, {pruned} pruned")
        print(f"\nBest configuration saved to: optuna_results/")


if __name__ == "__main__":
    main()