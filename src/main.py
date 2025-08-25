#!/usr/bin/env python3
"""
Main entry point for AE-DeepONet training pipeline.

Three-stage training following Goswami et al. (2023):
1. Autoencoder pretraining for dimensionality reduction
2. Latent dataset generation
3. DeepONet training on latent space

Features:
- GPU-optimized implementation
- Automatic latent data regeneration
- Unified model directory structure
- Configuration persistence
- Model export for deployment
- UPDATED: Flexible time point sampling for training
- UPDATED: Test set evaluation after training
"""

import logging
import shutil
import sys
import time
from pathlib import Path
import os

# Handle potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from utils import setup_logging, seed_everything, load_json_config, save_json, load_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import GPUSequenceDataset, GPULatentDataset, create_gpu_dataloader
from model import create_model
from trainer import Trainer
from generate_latent_dataset import generate_latent_dataset


class AEDeepONetPipeline:
    """
    Complete training pipeline for AE-DeepONet.

    Manages the three-stage training process:
    1. Autoencoder pretraining
    2. Latent dataset generation
    3. DeepONet training with flexible time sampling
    4. Test set evaluation

    Args:
        config_path: Path to JSON configuration file
    """

    def __init__(self, config_path: Path):
        """Initialize the training pipeline with configuration."""
        self.config = load_json_config(config_path)
        self.config_path = config_path

        # Setup paths
        self.setup_paths()

        # Setup logging
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        seed = self.config.get("system", {}).get("seed", 42)
        seed_everything(seed)

        # Setup hardware - MUST be GPU for performance
        self.device = setup_device()

        if self.device.type != "cuda":
            self.logger.error("AE-DeepONet requires CUDA GPU. CPU not supported.")
            raise RuntimeError("CUDA GPU required. Please run on a system with NVIDIA GPU.")

        # Check GPU memory availability
        free_mem, total_mem = torch.cuda.mem_get_info(self.device.index or 0)
        total_gb = total_mem / 1e9
        free_gb = free_mem / 1e9

        if total_gb < 8:
            self.logger.warning(
                f"GPU has only {total_gb:.1f}GB memory. "
                "Recommended minimum is 8GB. May encounter out-of-memory errors."
            )

        # Apply hardware optimizations
        optimize_hardware(self.config["system"], self.device)

        # Log GPU information
        props = torch.cuda.get_device_properties(self.device.index or 0)
        self.logger.info(
            f"Using GPU: {props.name} ({props.total_memory / 1e9:.1f}GB total, "
            f"{free_gb:.1f}GB free, Compute Capability {props.major}.{props.minor})"
        )

        # Save configuration to model directory for reproducibility
        self._save_config()

        self.logger.info("AE-DeepONet pipeline initialized (GPU mode)")

    def setup_paths(self):
        """Create and organize directory structure."""
        paths = self.config["paths"]

        # Create unique run directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(paths["model_save_dir"]) / f"ae_deeponet_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Data paths
        self.raw_data_files = [Path(f) for f in paths["raw_data_files"]]
        self.processed_dir = Path(paths["processed_data_dir"])
        self.latent_dir = self.processed_dir.parent / "latent_data"

        # Log directory
        self.log_dir = Path(paths["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Model directory: {self.run_dir}")

    def _save_config(self):
        """Save configuration to model directory for reproducibility."""
        config_copy = self.run_dir / "config.json"
        save_json(self.config, config_copy)
        self.logger.info(f"Configuration saved to {config_copy}")

    def _clean_latent_data(self):
        """Remove existing latent data to ensure fresh generation."""
        if self.latent_dir.exists():
            self.logger.info(f"Removing existing latent data at {self.latent_dir}")
            shutil.rmtree(self.latent_dir)
            self.logger.info("Latent data directory cleaned")

    def preprocess_data(self):
        """
        Preprocess raw data if needed with strict validation.

        Validates global variables and normalization methods according to
        the AE-DeepONet requirements.
        """
        if not self.processed_dir.exists() or not (self.processed_dir / "normalization.json").exists():
            self.logger.info("Preprocessing data...")

            # Strict validation of global variables
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
            time_norm_method = self.config["normalization"]["methods"].get(
                time_var,
                self.config["normalization"]["default_method"]
            )

            if time_norm_method != "log-min-max":
                self.logger.warning(
                    f"Time variable '{time_var}' using '{time_norm_method}' normalization. "
                    f"Recommended: 'log-min-max' for log-spaced time grids."
                )

            # Run preprocessing
            preprocessor = DataPreprocessor(
                raw_files=self.raw_data_files,
                output_dir=self.processed_dir,
                config=self.config
            )
            preprocessor.process_to_npy_shards()

            self.logger.info("Data preprocessing complete")
        else:
            self.logger.info("Using existing preprocessed data")

            # Validate existing data matches expectations
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
        """
        Stage 1: Train autoencoder for dimensionality reduction.

        Trains the autoencoder component to learn a compressed latent
        representation of the chemical species concentrations.
        """
        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Autoencoder Training")
        self.logger.info("=" * 60)

        # Check if already trained
        ae_checkpoint = self.run_dir / "ae_pretrained.pt"
        if ae_checkpoint.exists():
            self.logger.info("Autoencoder checkpoint found, skipping training...")
            return

        # Load normalization statistics
        norm_stats = load_json(self.processed_dir / "normalization.json")

        try:
            # Load training dataset to GPU
            train_dataset = GPUSequenceDataset(
                self.processed_dir, "train", self.config, self.device, norm_stats
            )

            # Load validation dataset if specified
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

        # Create trainer and train (unified directory)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            save_dir=self.run_dir,  # Single unified directory
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
        """
        Stage 2: Generate latent dataset.

        Uses the trained autoencoder to transform the full dataset into
        latent space representations for DeepONet training.
        Always regenerates to ensure consistency.
        """
        self.logger.info("=" * 60)
        self.logger.info("Stage 2: Generating Latent Dataset")
        self.logger.info("=" * 60)

        # Always clean and regenerate latent data for consistency
        self._clean_latent_data()

        # Load pretrained autoencoder
        model = create_model(self.config, self.device)
        checkpoint_path = self.run_dir / "ae_pretrained.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found at {checkpoint_path}. "
                "Please run Stage 1 first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    def stage3_train_deeponet(self):
        """
        Stage 3: Train DeepONet on latent space with flexible time sampling.

        Trains the DeepONet component to predict latent trajectories
        from initial conditions and global parameters.
        """
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: DeepONet Training on Latent Space")
        self.logger.info("=" * 60)

        # Get time sampling configuration
        train_cfg = self.config["training"]
        train_mode = train_cfg.get("train_time_sampling", "random")
        val_mode = train_cfg.get("val_time_sampling", "fixed")

        self.logger.info(f"Training time sampling mode: {train_mode}")
        if train_mode == "random":
            min_pts = train_cfg.get("train_min_time_points", 8)
            max_pts = train_cfg.get("train_max_time_points", 32)
            self.logger.info(f"  Random time points range: [{min_pts}, {max_pts}]")
        else:
            pts = train_cfg.get("train_time_points", 10)
            self.logger.info(f"  Fixed time points: {pts}")

        self.logger.info(f"Validation time sampling mode: {val_mode}")
        if val_mode == "random":
            min_pts = train_cfg.get("val_min_time_points", 8)
            max_pts = train_cfg.get("val_max_time_points", 32)
            self.logger.info(f"  Random time points range: [{min_pts}, {max_pts}]")
        else:
            pts = train_cfg.get("val_time_points", 50)
            self.logger.info(f"  Fixed time points: {pts}")

        # Load latent datasets to GPU
        try:
            train_dataset = GPULatentDataset(
                self.latent_dir,
                "train",
                self.config,
                self.device,
                time_sampling_mode=train_mode
            )

            val_dataset = None
            if self.config["training"].get("val_fraction", 0) > 0:
                val_dataset = GPULatentDataset(
                    self.latent_dir,
                    "validation",
                    self.config,
                    self.device,
                    time_sampling_mode=val_mode
                )
        except RuntimeError as e:
            self.logger.error(f"Failed to load latent datasets to GPU: {e}")
            self.logger.error("Consider reducing batch size")
            raise

        # Create dataloaders
        train_loader = create_gpu_dataloader(train_dataset, self.config, shuffle=True)
        val_loader = create_gpu_dataloader(val_dataset, self.config, shuffle=False) if val_dataset else None

        # Create model and load pretrained autoencoder
        model = create_model(self.config, self.device)
        ae_checkpoint = torch.load(
            self.run_dir / "ae_pretrained.pt",
            map_location=self.device,
            weights_only=False  # Need False for optimizer states and custom objects
        )

        # Load only autoencoder weights
        ae_state = {k: v for k, v in ae_checkpoint["model_state_dict"].items()
                    if 'autoencoder' in k}
        model.load_state_dict(ae_state, strict=False)

        # Freeze autoencoder if specified
        if self.config["training"].get("freeze_ae_after_pretrain", True):
            for param in model.autoencoder.parameters():
                param.requires_grad = False
            self.logger.info("Froze autoencoder parameters")

        # Create trainer (unified directory)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            save_dir=self.run_dir,  # Single unified directory
            device=self.device,
            is_latent_stage=True
        )

        # Train DeepONet
        best_loss = trainer.train_deeponet()

        self.logger.info(f"DeepONet training complete. Best val loss: {best_loss:.3e}")

        # Save final results
        results = {
            "best_val_loss": best_loss,
            "model_path": str(self.run_dir / "best_model.pt"),
            "config": self.config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_json(results, self.run_dir / "results.json")

        # Export model for deployment if configured
        self._export_model_for_deployment(model)

        # Clean up GPU memory
        del train_dataset, val_dataset, trainer, model
        torch.cuda.empty_cache()

    def evaluate_test_set(self):
        """
        Stage 4: Evaluate on test set after training completion.

        Performs comprehensive evaluation on held-out test data including:
        - Latent space MSE
        - Decoded species MAE (normalized and physical space)
        - Per-species error breakdown
        - Temporal error evolution
        """
        self.logger.info("=" * 60)
        self.logger.info("Stage 4: Test Set Evaluation")
        self.logger.info("=" * 60)

        # Check test fraction
        test_frac = self.config["training"].get("test_fraction", 0.0)
        if test_frac == 0:
            self.logger.warning("No test set configured (test_fraction=0). Skipping evaluation.")
            return

        # Load best model
        best_model_path = self.run_dir / "best_model.pt"
        if not best_model_path.exists():
            self.logger.error("Best model checkpoint not found. Please train model first.")
            return

        # Create model and load weights
        model = create_model(self.config, self.device)
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load normalization helper for physical space evaluation
        try:
            from normalizer import NormalizationHelper
            norm_stats = load_json(self.processed_dir / "normalization.json")
            norm_helper = NormalizationHelper(norm_stats, self.device, self.config)
            species_vars = self.config["data"]["target_species_variables"]
        except Exception as e:
            self.logger.warning(f"Could not load normalization helper: {e}")
            norm_helper = None
            species_vars = None

        # Load test dataset
        try:
            # Use all available time points for comprehensive evaluation
            test_dataset = GPULatentDataset(
                self.latent_dir,
                "test",
                self.config,
                self.device,
                time_sampling_mode="fixed",
                n_time_points=None  # Use all available points
            )
        except RuntimeError as e:
            self.logger.error(f"Failed to load test dataset: {e}")
            return

        test_loader = create_gpu_dataloader(test_dataset, self.config, shuffle=False)

        # Evaluation metrics
        total_mse = 0.0
        total_mae_norm = 0.0
        total_mae_phys = 0.0
        per_species_mae_norm = torch.zeros(len(species_vars), device=self.device)
        per_species_mae_phys = torch.zeros(len(species_vars), device=self.device)
        temporal_errors = []
        n_samples = 0
        n_time_points = 0

        self.logger.info(f"Evaluating on {len(test_dataset)} test trajectories...")

        with torch.no_grad():
            for batch_data in test_loader:
                inputs, targets_list, times_list = batch_data
                B = inputs.size(0)

                for i in range(B):
                    input_i = inputs[i:i + 1]
                    targets_i = targets_list[i].unsqueeze(0)
                    times_i = times_list[i]

                    # Get predictions
                    z_pred, _ = model(input_i, decode=False, trunk_times=times_i)

                    # Latent space MSE
                    mse = torch.nn.functional.mse_loss(z_pred, targets_i)
                    total_mse += mse.item()

                    # Decode to species space
                    y_pred, _ = model(input_i, decode=True, trunk_times=times_i)
                    B1, M, LD = targets_i.shape
                    y_true = model.decode(targets_i.reshape(B1 * M, LD)).view(B1, M, -1)

                    # Normalized space MAE
                    mae_norm = (y_pred - y_true).abs()
                    total_mae_norm += mae_norm.mean().item()

                    # Per-species normalized MAE
                    per_species_mae_norm += mae_norm.mean(dim=(0, 1))

                    # Physical space evaluation if available
                    if norm_helper is not None:
                        y_pred_phys = norm_helper.denormalize(
                            y_pred.reshape(-1, y_pred.shape[-1]), species_vars
                        ).view_as(y_pred)
                        y_true_phys = norm_helper.denormalize(
                            y_true.reshape(-1, y_true.shape[-1]), species_vars
                        ).view_as(y_true)

                        mae_phys = (y_pred_phys - y_true_phys).abs()
                        total_mae_phys += mae_phys.mean().item()
                        per_species_mae_phys += mae_phys.mean(dim=(0, 1))

                        # Temporal error evolution
                        temporal_mae = mae_phys.mean(dim=2).squeeze(0)  # [M]
                        temporal_errors.append((times_i.cpu().numpy(), temporal_mae.cpu().numpy()))

                    n_samples += 1
                    n_time_points += M

        # Compute averages
        avg_mse = total_mse / n_samples
        avg_mae_norm = total_mae_norm / n_samples
        avg_mae_phys = total_mae_phys / n_samples if norm_helper else 0.0
        per_species_mae_norm /= n_samples
        per_species_mae_phys /= n_samples

        # Log results
        self.logger.info("=" * 60)
        self.logger.info("TEST SET EVALUATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Number of test trajectories: {n_samples}")
        self.logger.info(f"Total time points evaluated: {n_time_points}")
        self.logger.info(f"Average MSE (latent space): {avg_mse:.3e}")
        self.logger.info(f"Average MAE (normalized): {avg_mae_norm:.3e}")

        if norm_helper:
            self.logger.info(f"Average MAE (physical): {avg_mae_phys:.3e}")
            self.logger.info("\nPer-species MAE (physical space):")
            for i, species in enumerate(species_vars):
                self.logger.info(f"  {species:20s}: {per_species_mae_phys[i].item():.3e}")

        # Save test results
        test_results = {
            "n_test_samples": n_samples,
            "n_time_points": n_time_points,
            "latent_mse": avg_mse,
            "normalized_mae": avg_mae_norm,
            "physical_mae": avg_mae_phys if norm_helper else None,
            "per_species_mae_norm": per_species_mae_norm.cpu().numpy().tolist(),
            "per_species_mae_phys": per_species_mae_phys.cpu().numpy().tolist() if norm_helper else None,
            "species_names": species_vars,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add temporal error information if available
        if temporal_errors:
            # Average temporal errors across all samples
            import numpy as np
            all_times = np.concatenate([t for t, _ in temporal_errors])
            unique_times = np.unique(all_times)
            avg_temporal_mae = []

            for t in unique_times:
                errors_at_t = []
                for times, errors in temporal_errors:
                    idx = np.where(np.abs(times - t) < 1e-6)[0]
                    if len(idx) > 0:
                        errors_at_t.append(errors[idx[0]])
                if errors_at_t:
                    avg_temporal_mae.append(np.mean(errors_at_t))

            test_results["temporal_evolution"] = {
                "times": unique_times.tolist(),
                "mae_physical": avg_temporal_mae
            }

        save_json(test_results, self.run_dir / "test_results.json")
        self.logger.info(f"\nTest results saved to {self.run_dir / 'test_results.json'}")

        # Clean up
        del model, test_dataset
        torch.cuda.empty_cache()

    def _export_model_for_deployment(self, model: torch.nn.Module):
        """
        Export the trained model for easy deployment.

        Args:
            model: Trained AE-DeepONet model
        """
        if self.config.get("system", {}).get("use_torch_export", False):
            try:
                self.logger.info("Exporting model for deployment...")

                # Create example input
                batch_size = 1
                example_input = torch.randn(
                    batch_size,
                    model.latent_dim + model.num_globals,
                    device=self.device,
                    dtype=torch.float32
                )

                # Export the model
                exported_program = torch.export.export(
                    model,
                    args=(example_input,),
                    kwargs={"decode": True, "return_trunk_outputs": False}
                )

                # Save in the model directory for easy access
                export_path = self.run_dir / "ae_deeponet_exported.pt2"
                torch.export.save(exported_program, str(export_path))

                self.logger.info(f"Model exported for deployment to {export_path}")

                # Also save a scripted version for broader compatibility
                scripted_model = torch.jit.script(model)
                script_path = self.run_dir / "ae_deeponet_scripted.pt"
                torch.jit.save(scripted_model, str(script_path))

                self.logger.info(f"Scripted model saved to {script_path}")

            except Exception as e:
                self.logger.warning(f"Model export failed: {e}")
                self.logger.info("Model saved in standard format at best_model.pt")

    def run(self):
        """
        Execute the complete 3-stage training pipeline plus test evaluation.

        Runs all stages sequentially:
        1. Data preprocessing (if needed)
        2. Autoencoder pretraining
        3. Latent data generation
        4. DeepONet training with flexible time sampling
        5. Test set evaluation
        """
        try:
            # Log initial GPU memory status
            free_mem, total_mem = torch.cuda.mem_get_info(self.device.index or 0)
            self.logger.info(
                f"Starting pipeline with {free_mem / 1e9:.1f}GB/{total_mem / 1e9:.1f}GB GPU memory free"
            )

            # Preprocess data if needed
            self.preprocess_data()

            # Stage 1: Train autoencoder
            self.stage1_train_autoencoder()

            # Stage 2: Generate latent dataset (always fresh)
            self.stage2_generate_latent_data()

            # Stage 3: Train DeepONet
            self.stage3_train_deeponet()

            # Stage 4: Evaluate on test set
            self.evaluate_test_set()

            self.logger.info("=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_dir}")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Final GPU cleanup
            torch.cuda.empty_cache()


def main():
    """Main entry point for the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AE-DeepONet for Chemical Kinetics (GPU-optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --config config.json

The pipeline will:
  1. Preprocess raw HDF5 data (if needed)
  2. Train autoencoder for dimensionality reduction
  3. Generate latent dataset (always fresh)
  4. Train DeepONet on latent space with flexible time sampling
  5. Evaluate on test set

All models and logs are saved in a timestamped directory.
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate configuration file
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA GPU required. No GPU detected.")
        print("This implementation requires an NVIDIA GPU with CUDA support.")
        print("\nAvailable devices:")
        print(f"  CPU: {torch.cuda.device_count()} cores")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  MPS: Available (Apple Silicon)")
        sys.exit(1)

    # Run the pipeline
    pipeline = AEDeepONetPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()