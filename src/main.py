#!/usr/bin/env python3
"""
Main entry point for AE-DeepONet training following Goswami et al. (2023).
GPU-only implementation with strict validation.
"""

import logging
import sys
import time
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch


from utils import setup_logging, seed_everything, load_json_config, save_json, load_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import GPUSequenceDataset, GPULatentDataset, create_gpu_dataloader
from model import create_model
from trainer import Trainer
from generate_latent_dataset import generate_latent_dataset



class AEDeepONetPipeline:
    """Three-stage training pipeline following the paper."""

    def __init__(self, config_path: Path):
        self.config = load_json_config(config_path)
        self.config_path = config_path

        # Setup paths
        self.setup_paths()

        # Setup logging
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        self.logger = logging.getLogger(__name__)

        # Set seed
        seed_everything(self.config.get("system", {}).get("seed", 42))

        # Setup hardware - MUST be GPU
        self.device = setup_device()

        if self.device.type != "cuda":
            self.logger.error("AE-DeepONet requires CUDA GPU. CPU not supported.")
            raise RuntimeError("CUDA GPU required. Please run on a system with NVIDIA GPU.")

        # Check GPU memory
        _, total_mem = torch.cuda.mem_get_info(self.device.index or 0)
        total_gb = total_mem / 1e9

        if total_gb < 8:
            self.logger.warning(
                f"GPU has only {total_gb:.1f}GB memory. "
                "Recommended minimum is 8GB. May encounter out-of-memory errors."
            )

        optimize_hardware(self.config["system"], self.device)

        # Log GPU info
        props = torch.cuda.get_device_properties(self.device.index or 0)
        self.logger.info(
            f"Using GPU: {props.name} ({props.total_memory / 1e9:.1f}GB, "
            f"Compute Capability {props.major}.{props.minor})"
        )

        self.logger.info("AE-DeepONet pipeline initialized (GPU mode)")

    def setup_paths(self):
        """Create directory structure."""
        paths = self.config["paths"]

        # Run directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(paths["model_save_dir"]) / f"ae_deeponet_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Data paths
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        self.processed_dir = Path(paths["processed_data_dir"])
        self.latent_dir = self.processed_dir.parent / "latent_data"
        self.log_dir = Path(paths["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_data(self):
        """Preprocess raw data if needed with strict validation."""
        if not self.processed_dir.exists() or not (self.processed_dir / "normalization.json").exists():
            self.logger.info("Preprocessing data...")

            # STRICT validation of global variables
            global_vars = self.config["data"]["global_variables"]
            expected_globals = self.config["data"].get("expected_globals", ["P", "T"])

            if global_vars != expected_globals:
                raise ValueError(
                    f"Global variables mismatch: got {global_vars}, expected {expected_globals}. "
                    f"The AE-DeepONet branch network requires exactly {expected_globals}. "
                    f"Please update your config to match your data format."
                )

            if len(global_vars) != 2:
                raise ValueError(
                    f"Expected exactly 2 global variables, got {len(global_vars)}: {global_vars}"
                )

            self.logger.info(f"Global variables validated: {global_vars} (P=pressure, T=temperature)")

            # Validate time normalization method
            time_var = self.config["data"]["time_variable"]
            time_norm_method = self.config["normalization"]["methods"].get(time_var,
                                                                           self.config["normalization"][
                                                                               "default_method"])

            if time_norm_method != "log-min-max":
                self.logger.warning(
                    f"Time variable '{time_var}' using '{time_norm_method}' normalization. "
                    f"Recommended: 'log-min-max' for log-spaced time grids."
                )

            preprocessor = DataPreprocessor(
                raw_files=self.raw_data_files,
                output_dir=self.processed_dir,
                config=self.config
            )
            preprocessor.process_to_npy_shards()

            self.logger.info("Data preprocessing complete")
        else:
            self.logger.info("Using existing preprocessed data")

            # Still validate the existing data matches expectations
            norm_stats = load_json(self.processed_dir / "normalization.json")

            # Check time normalization
            time_var = self.config["data"]["time_variable"]
            time_method = norm_stats.get("normalization_methods", {}).get(time_var)

            if time_method != "log-min-max":
                self.logger.warning(
                    f"Existing data has time normalized with '{time_method}'. "
                    f"Config specifies 'log-min-max'. This may cause issues."
                )

    def stage1_train_autoencoder(self):
        """Stage 1: Train autoencoder for dimensionality reduction."""
        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Autoencoder Training")
        self.logger.info("=" * 60)

        # Check if already trained
        ae_checkpoint = self.run_dir / "ae_pretrained.pt"
        if ae_checkpoint.exists():
            self.logger.info("Autoencoder already trained, skipping...")
            return

        # Load data to GPU
        norm_stats = load_json(self.processed_dir / "normalization.json")

        try:
            train_dataset = GPUSequenceDataset(
                self.processed_dir, "train", self.config, self.device, norm_stats
            )

            val_dataset = None
            if self.config["training"].get("val_fraction", 0) > 0:
                val_dataset = GPUSequenceDataset(
                    self.processed_dir, "validation", self.config, self.device, norm_stats
                )
        except RuntimeError as e:
            self.logger.error(f"Failed to load datasets to GPU: {e}")
            self.logger.error("Consider reducing batch size or data size")
            raise

        # Create dataloaders (no workers needed for GPU datasets)
        train_loader = create_gpu_dataloader(train_dataset, self.config, shuffle=True)
        val_loader = create_gpu_dataloader(val_dataset, self.config, shuffle=False) if val_dataset else None

        # Create model
        model = create_model(self.config, self.device)

        # Create trainer and train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            save_dir=self.run_dir / "stage1_ae",
            device=self.device,
            is_latent_stage=False
        )

        # Train autoencoder
        ae_epochs = self.config["training"].get("ae_pretrain_epochs", 100)
        trainer.train_ae_pretrain(ae_epochs)

        self.logger.info("Autoencoder training complete")

        # Clean up GPU memory
        del train_dataset, val_dataset, trainer, model
        torch.cuda.empty_cache()

    def stage2_generate_latent_data(self):
        """Stage 2: Generate latent dataset."""
        self.logger.info("=" * 60)
        self.logger.info("Stage 2: Generating Latent Dataset")
        self.logger.info("=" * 60)

        # Check if already generated
        if self.latent_dir.exists() and (self.latent_dir / "latent_shard_index.json").exists():
            self.logger.info("Latent dataset already exists, skipping...")
            return

        # Load pretrained autoencoder
        model = create_model(self.config, self.device)
        checkpoint = torch.load(self.run_dir / "stage1_ae" / "ae_pretrained.pt", map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate latent dataset
        try:
            generate_latent_dataset(
                model=model,
                input_dir=self.processed_dir,
                output_dir=self.latent_dir,
                config=self.config,
                device=self.device
            )
        except RuntimeError as e:
            self.logger.error(f"Failed to generate latent dataset: {e}")
            raise

        self.logger.info("Latent dataset generation complete")

        # Verify the generated data
        latent_index = load_json(self.latent_dir / "latent_shard_index.json")
        self.logger.info(f"Generated latent data with dimension {latent_index['latent_dim']}")
        self.logger.info(f"Trunk times: {latent_index['trunk_times']}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    def stage3_train_deeponet(self):
        """Stage 3: Train DeepONet on latent space."""
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: DeepONet Training on Latent Space")
        self.logger.info("=" * 60)

        # Load latent datasets to GPU
        try:
            train_dataset = GPULatentDataset(self.latent_dir, "train", self.config, self.device)

            val_dataset = None
            if self.config["training"].get("val_fraction", 0) > 0:
                val_dataset = GPULatentDataset(self.latent_dir, "validation", self.config, self.device)
        except RuntimeError as e:
            self.logger.error(f"Failed to load latent datasets to GPU: {e}")
            self.logger.error("Consider reducing batch size")
            raise

        # Create dataloaders
        train_loader = create_gpu_dataloader(train_dataset, self.config, shuffle=True)
        val_loader = create_gpu_dataloader(val_dataset, self.config, shuffle=False) if val_dataset else None

        # Create model and load pretrained autoencoder
        model = create_model(self.config, self.device)
        ae_checkpoint = torch.load(self.run_dir / "stage1_ae" / "ae_pretrained.pt", map_location=self.device)

        # Load only autoencoder weights
        ae_state = {k: v for k, v in ae_checkpoint["model_state_dict"].items()
                    if 'autoencoder' in k}
        model.load_state_dict(ae_state, strict=False)

        # Freeze autoencoder if specified
        if self.config["training"].get("freeze_ae_after_pretrain", True):
            for param in model.autoencoder.parameters():
                param.requires_grad = False
            self.logger.info("Froze autoencoder parameters")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            save_dir=self.run_dir / "stage3_deeponet",
            device=self.device,
            is_latent_stage=True
        )

        # Train DeepONet
        best_loss = trainer.train_deeponet()

        self.logger.info(f"DeepONet training complete. Best loss: {best_loss:.6e}")

        # Save results
        results = {
            "best_val_loss": best_loss,
            "model_path": str(self.run_dir / "stage3_deeponet" / "best_model.pt"),
            "config": self.config
        }
        save_json(results, self.run_dir / "results.json")

        # Clean up
        del train_dataset, val_dataset, trainer, model
        torch.cuda.empty_cache()

    def run(self):
        """Execute the complete 3-stage pipeline."""
        try:
            # Log GPU memory at start
            free_mem, total_mem = torch.cuda.mem_get_info(self.device.index or 0)
            self.logger.info(
                f"Starting pipeline with {free_mem / 1e9:.1f}GB/{total_mem / 1e9:.1f}GB GPU memory free"
            )

            # Preprocess data if needed
            self.preprocess_data()

            # Stage 1: Train autoencoder
            self.stage1_train_autoencoder()

            # Stage 2: Generate latent dataset
            self.stage2_generate_latent_data()

            # Stage 3: Train DeepONet
            self.stage3_train_deeponet()

            self.logger.info("=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_dir}")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Final GPU cleanup
            torch.cuda.empty_cache()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AE-DeepONet for Chemical Kinetics (GPU-only)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file"
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA GPU required. No GPU detected.")
        print("This implementation requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)

    pipeline = AEDeepONetPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()