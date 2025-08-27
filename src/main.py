#!/usr/bin/env python3
"""
Main entry point for AE-DeepONet training pipeline.

Three-stage training following Goswami et al. (2023):
1. Autoencoder pretraining for dimensionality reduction (skippable if bypass_autoencoder=True)
2. Latent dataset generation (skippable if bypass_autoencoder=True)
3. DeepONet training on latent/species space

Features:
- GPU-optimized implementation
- Automatic latent data regeneration
- Unified model directory structure
- Configuration persistence
- Model export for deployment
- UPDATED: Flexible time point sampling for training
- UPDATED: Preprocessing-only mode for CPU systems
- NEW: Bypass autoencoder option for direct species space training
- NEW: Reuse existing compatible autoencoders
"""

import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
import os
from typing import Optional


# Handle potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from utils import setup_logging, seed_everything, load_json_config, save_json, load_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import GPUSequenceDataset, GPULatentDataset, GPUSpeciesDeepONetDataset, create_gpu_dataloader
from model import create_model
from trainer import Trainer
from generate_latent_dataset import generate_latent_dataset


class AEDeepONetPipeline:
    """
    Complete training pipeline for AE-DeepONet.

    Manages the three-stage training process (or direct DeepONet if bypassed):
    1. Autoencoder pretraining (skipped if bypassed or reused)
    2. Latent dataset generation (skipped if bypassed)
    3. DeepONet training with flexible time sampling

    Args:
        config_path: Path to JSON configuration file
        preprocess_only: If True, only run preprocessing then exit (no GPU required)
    """

    def __init__(self, config_path: Path, preprocess_only: bool = False):
        """Initialize the training pipeline with configuration."""
        self.config = load_json_config(config_path)
        self.config_path = config_path
        self.preprocess_only = preprocess_only

        # Check bypass mode
        self.bypass_autoencoder = self.config.get("model", {}).get("bypass_autoencoder", False)
        self.reuse_autoencoder = self.config.get("model", {}).get("reuse_autoencoder", False)

        # Setup paths
        self.setup_paths()

        # Setup logging
        log_file = self.log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=log_file)
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        seed = self.config.get("system", {}).get("seed", 42)
        seed_everything(seed)

        # Setup hardware - allow CPU for preprocessing
        self.device = setup_device()

        if self.preprocess_only:
            # For preprocessing only, CPU is fine
            self.logger.info("Running in preprocessing-only mode (CPU is sufficient)")
            if self.device.type == "cuda":
                self.logger.info(f"GPU detected but not required for preprocessing")
        else:
            # For training, require CUDA (unless bypassing autoencoder)
            if not self.bypass_autoencoder and self.device.type != "cuda":
                self.logger.error("AE-DeepONet training requires CUDA GPU. CPU not supported.")
                raise RuntimeError(
                    "CUDA GPU required for training with autoencoder. "
                    "Use --preprocess-only for preprocessing without GPU, "
                    "or set bypass_autoencoder=true in config for CPU training."
                )
            elif self.bypass_autoencoder and self.device.type != "cuda":
                self.logger.warning(
                    "Running DeepONet without autoencoder on CPU - training may be slow"
                )

            if self.device.type == "cuda":
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

        # Log configuration modes
        if self.bypass_autoencoder:
            self.logger.info("BYPASS MODE: Skipping autoencoder - DeepONet will work directly in species space")
        if self.reuse_autoencoder:
            self.logger.info("REUSE MODE: Will search for compatible existing autoencoders")

        self.logger.info(
            f"AE-DeepONet pipeline initialized ({'preprocessing-only' if self.preprocess_only else 'full training'} mode)"
        )

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

    def _compute_config_fingerprint(self, config_subset: dict) -> str:
        """Compute hash fingerprint of configuration subset for comparison."""
        # Create a canonical JSON representation
        canonical = json.dumps(config_subset, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _find_compatible_autoencoder(self) -> Optional[Path]:
        """
        Search for compatible pre-trained autoencoder in existing model directories.

        Returns:
            Path to compatible ae_pretrained.pt file, or None if not found
        """
        search_dirs = self.config.get("paths", {}).get("autoencoder_search_dirs", ["models/"])

        # Extract critical parameters that must match
        required_params = {
            "latent_dim": self.config["model"]["latent_dim"],
            "ae_encoder_layers": self.config["model"]["ae_encoder_layers"],
            "ae_decoder_layers": self.config["model"]["ae_decoder_layers"],
            "num_species": len(self.config["data"]["species_variables"]),
            "species_variables": self.config["data"]["species_variables"],
            "normalization": self.config["normalization"]
        }

        required_fingerprint = self._compute_config_fingerprint(required_params)

        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if not search_path.exists():
                continue

            # Look for ae_deeponet_* directories
            for model_dir in sorted(search_path.glob("ae_deeponet_*/")):
                ae_path = model_dir / "ae_pretrained.pt"
                config_path = model_dir / "config.json"

                if ae_path.exists() and config_path.exists():
                    # Load and check configuration
                    try:
                        saved_config = load_json(config_path)

                        # Extract same parameters from saved config
                        saved_params = {
                            "latent_dim": saved_config["model"]["latent_dim"],
                            "ae_encoder_layers": saved_config["model"]["ae_encoder_layers"],
                            "ae_decoder_layers": saved_config["model"]["ae_decoder_layers"],
                            "num_species": len(saved_config["data"]["species_variables"]),
                            "species_variables": saved_config["data"]["species_variables"],
                            "normalization": saved_config["normalization"]
                        }

                        saved_fingerprint = self._compute_config_fingerprint(saved_params)

                        if saved_fingerprint == required_fingerprint:
                            self.logger.info(f"Found compatible autoencoder at {ae_path}")

                            # Verify checkpoint is actually loadable
                            try:
                                checkpoint = torch.load(ae_path, map_location='cpu', weights_only=False)
                                if "model_state_dict" in checkpoint:
                                    return ae_path
                            except Exception as e:
                                self.logger.warning(f"Could not load checkpoint {ae_path}: {e}")

                    except Exception as e:
                        self.logger.debug(f"Could not check {model_dir}: {e}")

        return None

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

        Can be skipped if:
        - bypass_autoencoder=True
        - reuse_autoencoder=True and compatible autoencoder found

        Trains the autoencoder component to learn a compressed latent
        representation of the chemical species concentrations.
        """
        # Skip if bypassing autoencoder
        if self.bypass_autoencoder:
            self.logger.info("Bypassing autoencoder - skipping Stage 1")
            return

        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Autoencoder Training")
        self.logger.info("=" * 60)

        # If a checkpoint already exists in this run dir, don't retrain
        ae_checkpoint = self.run_dir / "ae_pretrained.pt"
        if ae_checkpoint.exists():
            self.logger.info("Autoencoder checkpoint found in run directory; skipping training")
            return

        # Try to reuse an existing AE if enabled
        if self.reuse_autoencoder:
            existing_ae_path = self._find_compatible_autoencoder()
            if existing_ae_path:
                self.logger.info(f"Reusing compatible autoencoder from {existing_ae_path}")

                # Copy the AE checkpoint into the current run directory
                ae_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_ae_path, ae_checkpoint)

                # Copy the exact normalization.json used by that AE
                # Prefer embedded config in the checkpoint; fall back to config.json in the same dir
                saved_cfg = None
                try:
                    ckpt = torch.load(existing_ae_path, map_location="cpu", weights_only=False)
                    saved_cfg = ckpt.get("config", None)
                except Exception as e:
                    self.logger.warning(
                        f"Could not load checkpoint to extract config ({e}); falling back to config.json")

                if saved_cfg is None:
                    cfg_json = existing_ae_path.parent / "config.json"
                    if cfg_json.exists():
                        try:
                            saved_cfg = load_json(cfg_json)
                        except Exception as e:
                            self.logger.warning(f"Failed to read {cfg_json}: {e}")

                if saved_cfg is not None:
                    src_norm = Path(saved_cfg["paths"]["processed_data_dir"]) / "normalization.json"
                    dst_norm = self.processed_dir / "normalization.json"
                    try:
                        if src_norm.exists():
                            dst_norm.parent.mkdir(parents=True, exist_ok=True)
                            # Overwrite to guarantee consistency with reused AE
                            shutil.copy2(src_norm, dst_norm)
                            self.logger.info(f"Copied normalization stats from {src_norm} → {dst_norm}")
                        else:
                            self.logger.warning(f"No normalization.json at {src_norm}; ensure normalization matches.")
                    except Exception as e:
                        self.logger.warning(f"Failed to copy normalization.json: {e}")
                else:
                    self.logger.warning(
                        "No saved config available from reused AE; cannot verify/copy normalization.json")

                # Done — we reused and synced normalization; skip training
                return
            else:
                self.logger.info("No compatible autoencoder found — training a new one")

        # Load normalization statistics (must exist after preprocessing)
        norm_stats = load_json(self.processed_dir / "normalization.json")

        # Build GPU datasets
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

        # Dataloaders
        train_loader = create_gpu_dataloader(train_dataset, self.config, shuffle=True)
        val_loader = create_gpu_dataloader(val_dataset, self.config, shuffle=False) if val_dataset else None

        # Model & trainer
        model = create_model(self.config, self.device)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            save_dir=self.run_dir,
            device=self.device,
            is_latent_stage=False
        )

        # Train AE
        ae_epochs = self.config["training"].get("ae_pretrain_epochs", 100)
        trainer.train_ae_pretrain(ae_epochs)
        self.logger.info("Autoencoder training complete")

        # Cleanup
        del train_dataset, val_dataset, trainer, model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def stage2_generate_latent_data(self):
        """
        Stage 2: Generate latent dataset.

        Skipped if bypass_autoencoder=True.

        Uses the trained autoencoder to transform the full dataset into
        latent space representations for DeepONet training.
        Always regenerates to ensure consistency.
        """
        # Skip if bypassing autoencoder
        if self.bypass_autoencoder:
            self.logger.info("Bypassing autoencoder - skipping Stage 2")
            return

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
        Stage 3: Train DeepONet on latent/species space with flexible time sampling.

        Works in:
        - Latent space if autoencoder is used
        - Species space if bypass_autoencoder=True

        Trains the DeepONet component to predict trajectories
        from initial conditions and global parameters.
        """
        self.logger.info("=" * 60)
        if self.bypass_autoencoder:
            self.logger.info("Stage 3: DeepONet Training on Species Space (Bypass Mode)")
        else:
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
            # Check what's actually configured for fixed sampling
            train_fixed_times = train_cfg.get("train_fixed_times", None)
            if train_fixed_times == "all":
                self.logger.info(f"  Using ALL available time points")
            elif isinstance(train_fixed_times, list):
                self.logger.info(f"  Fixed time points: {len(train_fixed_times)} specified points")
                self.logger.debug(f"  Points: {train_fixed_times}")
            else:
                # Fallback to count
                pts = train_cfg.get("train_time_points", 10)
                self.logger.info(f"  Fixed time points: {pts} evenly-spaced points")

        self.logger.info(f"Validation time sampling mode: {val_mode}")
        if val_mode == "random":
            min_pts = train_cfg.get("val_min_time_points", 8)
            max_pts = train_cfg.get("val_max_time_points", 32)
            self.logger.info(f"  Random time points range: [{min_pts}, {max_pts}]")
        else:
            # Check what's actually configured for fixed sampling
            val_fixed_times = train_cfg.get("val_fixed_times", None)
            if val_fixed_times == "all":
                self.logger.info(f"  Using ALL available time points")
            elif isinstance(val_fixed_times, list):
                self.logger.info(f"  Fixed time points: {len(val_fixed_times)} specified points")
                self.logger.debug(f"  Points: {val_fixed_times}")
            else:
                # Fallback to count
                pts = train_cfg.get("val_time_points", 50)
                self.logger.info(f"  Fixed time points: {pts} evenly-spaced points")

        # Load datasets - either latent or species space
        try:
            if self.bypass_autoencoder:
                # Load species space datasets
                norm_stats = load_json(self.processed_dir / "normalization.json")

                train_dataset = GPUSpeciesDeepONetDataset(
                    self.processed_dir,
                    "train",
                    self.config,
                    self.device,
                    norm_stats=norm_stats,
                    time_sampling_mode=train_mode
                )

                val_dataset = None
                if self.config["training"].get("val_fraction", 0) > 0:
                    val_dataset = GPUSpeciesDeepONetDataset(
                        self.processed_dir,
                        "validation",
                        self.config,
                        self.device,
                        norm_stats=norm_stats,
                        time_sampling_mode=val_mode
                    )
            else:
                # Load latent datasets
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

            # Log actual number of time points being used
            if hasattr(train_dataset, 'fixed_indices') and train_dataset.fixed_indices is not None:
                self.logger.info(f"Training dataset will use {len(train_dataset.fixed_indices)} time points")
            if val_dataset and hasattr(val_dataset, 'fixed_indices') and val_dataset.fixed_indices is not None:
                self.logger.info(f"Validation dataset will use {len(val_dataset.fixed_indices)} time points")

        except RuntimeError as e:
            self.logger.error(f"Failed to load datasets to GPU: {e}")
            self.logger.error("Consider reducing batch size")
            raise

        # Create dataloaders
        train_loader = create_gpu_dataloader(train_dataset, self.config, shuffle=True)
        val_loader = create_gpu_dataloader(val_dataset, self.config, shuffle=False) if val_dataset else None

        # Create model
        model = create_model(self.config, self.device)

        # Load pretrained autoencoder if not bypassing
        if not self.bypass_autoencoder:
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
            is_latent_stage=not self.bypass_autoencoder,
            is_species_stage=self.bypass_autoencoder
        )

        # Train DeepONet
        best_loss = trainer.train_deeponet()

        self.logger.info(f"DeepONet training complete. Best loss: {best_loss:.3e}")

        # Save final results
        results = {
            "best_val_loss": best_loss,
            "model_path": str(self.run_dir / "best_model.pt"),
            "config": self.config,
            "bypass_autoencoder": self.bypass_autoencoder,
            "reused_autoencoder": self.reuse_autoencoder and not self.bypass_autoencoder,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_json(results, self.run_dir / "results.json")

        # Export model for deployment if configured
        self._export_model_for_deployment(model)

        # Clean up GPU memory
        del train_dataset, val_dataset, trainer, model
        torch.cuda.empty_cache()

    def _export_model_for_deployment(self, model: torch.nn.Module):
        """
        Export the trained model for easy deployment.
        """
        if self.config.get("system", {}).get("use_torch_export", False):
            try:
                self.logger.info("Exporting model for deployment...")

                # Create example inputs consistent with model dtype/device
                batch_size = 1
                if self.bypass_autoencoder:
                    input_dim = model.num_species + model.num_globals
                else:
                    input_dim = model.latent_dim + model.num_globals

                model_dtype = next(model.parameters()).dtype

                example_input = torch.randn(
                    batch_size, input_dim,
                    device=self.device,
                    dtype=model_dtype
                )

                # Provide example trunk times [M,1]
                t_example = torch.linspace(0, 1, steps=100, device=self.device, dtype=model_dtype).unsqueeze(-1)

                # Export the model (forward requires trunk_times)
                exported_program = torch.export.export(
                    model,
                    args=(example_input,),
                    kwargs={"decode": True, "return_trunk_outputs": False, "trunk_times": t_example},
                )

                export_path = self.run_dir / "ae_deeponet_exported.pt2"
                torch.export.save(exported_program, str(export_path))
                self.logger.info(f"Model exported for deployment to {export_path}")

                # Optional scripted backup
                scripted_model = torch.jit.script(model)
                script_path = self.run_dir / "ae_deeponet_scripted.pt"
                torch.jit.save(scripted_model, str(script_path))
                self.logger.info(f"Scripted model saved to {script_path}")

            except Exception as e:
                self.logger.warning(f"Model export failed: {e}")
                self.logger.info("Model saved in standard format at best_model.pt")

    def run(self):
        """
        Execute the pipeline - preprocessing only or full training.

        Runs preprocessing and optionally the training stages:
        1. Data preprocessing (always)
        2. Autoencoder pretraining (if not preprocess_only and not bypass_autoencoder)
        3. Latent data generation (if not preprocess_only and not bypass_autoencoder)
        4. DeepONet training (if not preprocess_only)
        """
        try:
            if self.preprocess_only:
                # Only run preprocessing
                self.logger.info("=" * 60)
                self.logger.info("Running data preprocessing only")
                self.logger.info("=" * 60)

                self.preprocess_data()

                self.logger.info("=" * 60)
                self.logger.info("Preprocessing completed successfully!")
                self.logger.info(f"Processed data saved in: {self.processed_dir}")
                self.logger.info("To continue with training, run without --preprocess-only flag")
                self.logger.info("=" * 60)
                return

            # Full pipeline - check GPU is available (unless bypassing autoencoder)
            if not self.bypass_autoencoder and self.device.type != "cuda":
                raise RuntimeError("Training stages with autoencoder require CUDA GPU")

            # Log initial GPU memory status
            if self.device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info(self.device.index or 0)
                self.logger.info(
                    f"Starting pipeline with {free_mem / 1e9:.1f}GB/{total_mem / 1e9:.1f}GB GPU memory free"
                )

            # Preprocess data if needed
            self.preprocess_data()

            # Stage 1: Train autoencoder (skipped if bypass_autoencoder=True)
            self.stage1_train_autoencoder()

            # Stage 2: Generate latent dataset (skipped if bypass_autoencoder=True)
            self.stage2_generate_latent_data()

            # Stage 3: Train DeepONet
            self.stage3_train_deeponet()

            self.logger.info("=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved in: {self.run_dir}")
            if self.bypass_autoencoder:
                self.logger.info("Mode: Direct DeepONet (autoencoder bypassed)")
            elif self.reuse_autoencoder:
                self.logger.info("Mode: AE-DeepONet (autoencoder reused)")
            else:
                self.logger.info("Mode: AE-DeepONet (full training)")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Final GPU cleanup (only if using CUDA)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def main():
    """Main entry point for the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AE-DeepONet for Chemical Kinetics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Full training pipeline (requires GPU):
  python main.py --config config.json

  # Preprocessing only (CPU is sufficient):
  python main.py --config config.json --preprocess-only

  # Direct DeepONet without autoencoder:
  Set bypass_autoencoder=true in config.json

  # Reuse existing autoencoder:
  Set reuse_autoencoder=true in config.json

The pipeline will:
  1. Preprocess raw HDF5 data (CPU sufficient)
  2. Train autoencoder for dimensionality reduction (GPU required, unless bypassed)
  3. Generate latent dataset (GPU required, unless bypassed)
  4. Train DeepONet on latent/species space (GPU recommended)

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

    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run data preprocessing, then exit (no GPU required)"
    )

    args = parser.parse_args()

    # Validate configuration file
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Check for CUDA availability (only for full pipeline without bypass)
    config = load_json_config(args.config)
    bypass_ae = config.get("model", {}).get("bypass_autoencoder", False)

    if not args.preprocess_only and not bypass_ae and not torch.cuda.is_available():
        print("Error: CUDA GPU required for training with autoencoder. No GPU detected.")
        print("\nOptions:")
        print("  1. Run preprocessing only: python main.py --config config.json --preprocess-only")
        print("  2. Use bypass_autoencoder=true in config for CPU-compatible training")
        print("  3. Use a system with NVIDIA GPU for full training")
        print("\nAvailable devices:")
        print(f"  CPU: {os.cpu_count()} cores")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  MPS: Available (Apple Silicon)")
        sys.exit(1)

    # Run the pipeline
    pipeline = AEDeepONetPipeline(args.config, preprocess_only=args.preprocess_only)
    pipeline.run()


if __name__ == "__main__":
    main()