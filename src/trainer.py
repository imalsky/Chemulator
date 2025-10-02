#!/usr/bin/env python3
"""
Flow-map DeepONet Trainer with Restart Support
================================================
Training loop implementation with checkpoint/restart capabilities.

Features:
- Shape-agnostic loss computation (handles both [B,S] and [B,K,S] targets)
- Mixed precision training with automatic mixed precision (AMP)
- Built-in PyTorch schedulers with warmup and cosine annealing
- Deterministic per-epoch sampling via dataset.set_epoch()
- TensorBoard and CSV logging for training metrics
- Full checkpoint saving/loading for restart including scheduler state
- SIGTERM handling for cluster preemption
- Adaptive stiff loss for chemical systems with atomic conservation
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import signal
import time
import contextlib
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter


# ------------------------------ Loss Functions -------------------------------


class AdaptiveStiffLoss(nn.Module):
    """
    Composite loss for stiff chemical systems with log-normalized data.

    Combines:
    - Main term: MAE in log10 physical space (per-decade error)
    - Stabilizer: Small MSE in z-space
    - Species weights: Based on dynamic range
    - Time weights: Emphasize trajectory edges where stiff dynamics occur
    - Optional: Atomic conservation penalty
    """

    def __init__(
            self,
            log_means: torch.Tensor,
            log_stds: torch.Tensor,
            species_log_min: torch.Tensor,
            species_log_max: torch.Tensor,
            *,
            lambda_phys: float = 1.0,
            lambda_z: float = 0.1,
            epsilon_phys: float = 1e-20,
            use_fractional: bool = False,
            time_edge_gain: float = 2.0,
            # Atomic conservation parameters
            elemental_conservation: bool = False,
            elemental_penalty: float = 0.0,
            species_names: Optional[List[str]] = None,
            elements: Optional[List[str]] = None,
            elemental_mode: str = "relative",
            elemental_weights: Optional[Any] = "auto",
            eps_elem: float = 1e-20,
            debug_parser: bool = False,
            device=None
    ):
        super().__init__()

        # Store device for creating tensors
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # Register statistics as buffers (move with model to device)
        self.register_buffer("log_means", log_means.detach().clone())
        self.register_buffer("log_stds", torch.clamp(log_stds.detach().clone(), min=1e-10))
        self.register_buffer("log_min", species_log_min.detach().clone())
        self.register_buffer("log_max", species_log_max.detach().clone())

        # Species weights: sqrt of dynamic range, normalized and clipped
        log_range = torch.clamp(self.log_max - self.log_min, min=1e-6)
        w = torch.sqrt(log_range)
        w = w / (w.mean() + 1e-12)
        w = torch.clamp(w, 0.5, 2.0)
        self.register_buffer("w_species", w)

        # Store configuration
        self.lambda_phys = lambda_phys
        self.lambda_z = lambda_z
        self.eps_phys = epsilon_phys
        self.use_fractional = use_fractional
        self.time_edge_gain = time_edge_gain
        self.elemental_penalty = elemental_penalty

        # Setup atomic conservation if enabled
        self.elemental_conservation = elemental_conservation
        self.elemental_mode = elemental_mode
        self.eps_elem = eps_elem

        if self.elemental_conservation and self.elemental_penalty > 0:
            if species_names is None or len(species_names) == 0:
                raise ValueError("species_names required and must be non-empty for elemental conservation")

            # Default to common combustion elements
            if elements is None:
                elements = ["H", "C", "N", "O"]

            self.elements = elements
            self.species_names = species_names

            # Validate species count matches model dimension
            S_expected = len(species_names)
            S_actual = log_means.shape[0]
            if S_expected != S_actual:
                raise ValueError(
                    f"Species count mismatch: species_names has {S_expected} entries "
                    f"but model outputs {S_actual} species"
                )

            # Build stoichiometry matrix on the correct device
            stoich_matrix = self._build_stoichiometry_matrix(species_names, elements)
            self.register_buffer("stoich_matrix", stoich_matrix)

            # Run parser self-tests if requested
            if debug_parser:
                self._run_parser_tests()

            # Setup elemental weights on the correct device
            if elemental_weights == "auto":
                # Will be computed on first batch
                self.register_buffer("elem_weights", torch.ones(len(elements), dtype=torch.float32, device=self.device))
                self._need_auto_weights = True
            elif isinstance(elemental_weights, (list, tuple)):
                if len(elemental_weights) != len(elements):
                    raise ValueError(f"elemental_weights length {len(elemental_weights)} != {len(elements)} elements")
                weights_tensor = torch.tensor(elemental_weights, dtype=torch.float32, device=self.device)
                self.register_buffer("elem_weights", weights_tensor)
                self._need_auto_weights = False
            else:
                self.register_buffer("elem_weights", torch.ones(len(elements), dtype=torch.float32, device=self.device))
                self._need_auto_weights = False

            self._logged_conservation = False

    def _parse_species_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse a chemical formula to extract element counts.
        Handles formats like: C2H2, C2_H2, CH3OH, H2O, HCN, He, H+, OH-, M

        NOTE: Parentheses are stripped but not expanded (e.g., (OH)2 → OH2).
        This is a limitation; if parentheses are used, counts will be incorrect.

        Args:
            formula: Chemical formula string

        Returns:
            Dictionary mapping element symbols to counts
        """
        # Clean the formula
        original = formula
        formula = formula.strip()

        # Special cases
        if formula.upper() in ("M",):  # Third body
            return {}

        # Remove charge indicators and underscores
        formula = re.sub(r'[_\+\-]', '', formula)

        # Remove parentheses (NOTE: this doesn't handle multipliers correctly)
        # e.g., Ca(OH)2 becomes CaOH2 which is wrong
        # For proper handling, would need a more sophisticated parser
        if '(' in formula or ')' in formula:
            if not hasattr(self, '_warned_parentheses'):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Species formula '{original}' contains parentheses which are not properly expanded. "
                    f"E.g., Ca(OH)2 will be parsed as CaOH2, not CaO2H2. Consider expanding formulas manually."
                )
                self._warned_parentheses = True
            formula = re.sub(r'[()]', '', formula)

        element_counts = {}

        # Pattern to match element symbol followed by optional count
        # Matches two-letter elements first (He, Ne, etc.) then single letters
        # Case-sensitive: expects proper capitalization (He not he, HE, etc.)
        pattern = r'([A-Z][a-z]?)(\d*)'

        matches = re.findall(pattern, formula)

        if not matches and formula:
            # No matches but formula not empty - might be lowercase or other issue
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not parse species formula: '{original}' (cleaned: '{formula}')")

        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            if element in element_counts:
                element_counts[element] += count
            else:
                element_counts[element] = count

        return element_counts

    def _build_stoichiometry_matrix(self, species_names: List[str], elements: List[str]) -> torch.Tensor:
        """
        Build stoichiometry matrix E where E[s,e] = number of atoms of element e in species s.

        Args:
            species_names: List of species names (formulas)
            elements: List of element symbols to track

        Returns:
            Stoichiometry matrix [S, E] on the correct device
        """
        S = len(species_names)
        E = len(elements)

        # Create matrix on the correct device
        stoich = torch.zeros(S, E, dtype=torch.float32, device=self.device)

        for s, species in enumerate(species_names):
            counts = self._parse_species_formula(species)
            for e, element in enumerate(elements):
                stoich[s, e] = float(counts.get(element, 0))

        return stoich

    def _run_parser_tests(self):
        """Run self-tests on the parser to catch issues early."""
        test_cases = [
            ("C2H2", {"C": 2, "H": 2}),
            ("C2_H2", {"C": 2, "H": 2}),
            ("CH3OH", {"C": 1, "H": 4, "O": 1}),
            ("H2O", {"H": 2, "O": 1}),
            ("HCN", {"H": 1, "C": 1, "N": 1}),
            ("He", {"He": 1}),  # He should not contribute H
            ("CO2", {"C": 1, "O": 2}),
            ("N2", {"N": 2}),
            ("NH3", {"N": 1, "H": 3}),
        ]

        for formula, expected in test_cases:
            result = self._parse_species_formula(formula)
            # Check only tracked elements
            for elem in ["H", "C", "N", "O"]:
                exp_count = expected.get(elem, 0)
                res_count = result.get(elem, 0)
                if exp_count != res_count:
                    raise ValueError(
                        f"Parser test failed for '{formula}': "
                        f"expected {elem}={exp_count}, got {elem}={res_count}"
                    )

        import logging
        logger = logging.getLogger(__name__)
        logger.info("All parser self-tests passed")

    def _z_to_log10(self, z: torch.Tensor) -> torch.Tensor:
        """Convert from normalized z-space to log10 physical space."""
        return z * self.log_stds + self.log_means

    def _time_weights(self, t01: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent weights that emphasize trajectory edges.

        Args:
            t01: Normalized times in [0,1], shape [B,K]

        Returns:
            Weights with shape [B,K,1] for broadcasting
        """
        if self.time_edge_gain <= 1.0:
            w = torch.ones_like(t01)
        else:
            # U-shaped weight function: peaks at t=0 and t=1
            w = 1.0 + (self.time_edge_gain - 1.0) * (1.0 - 4.0 * t01 * (1.0 - t01))
        return w.unsqueeze(-1)

    def forward(
            self,
            pred_z: torch.Tensor,
            true_z: torch.Tensor,
            t_norm: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute adaptive loss.

        Args:
            pred_z: Predictions in z-space [B,K,S] or [B,S]
            true_z: Targets in z-space [B,K,S] or [B,S]
            t_norm: Normalized times [B,K] or [B,K,1] or [B]
            mask: Optional validity mask [B,K] or [B]
            return_components: If True, return dict with component losses

        Returns:
            Scalar loss value or dict of loss components
        """
        # MSE in z-space (stabilizer term)
        loss_z = (pred_z - true_z) ** 2

        # Convert to log10 physical space
        pred_log = self._z_to_log10(pred_z)
        true_log = self._z_to_log10(true_z)

        # Main loss term in physical space
        if self.use_fractional:
            pred_y = torch.pow(10.0, torch.clamp(pred_log, min=-45.0))
            true_y = torch.pow(10.0, torch.clamp(true_log, min=-45.0))
            loss_phys = torch.abs(pred_y - true_y) / (torch.abs(true_y) + self.eps_phys)
        else:
            loss_phys = torch.abs(pred_log - true_log)

        # Apply species weights
        loss_phys = loss_phys * self.w_species

        # Prepare time weights
        if t_norm.ndim == 3 and t_norm.shape[-1] == 1:
            t_norm = t_norm.squeeze(-1)
        elif t_norm.ndim == 1:
            t_norm = t_norm.unsqueeze(1)

        # Apply time weights
        if loss_phys.ndim == 3:
            wt = self._time_weights(torch.clamp(t_norm, 0.0, 1.0))
            loss_phys = loss_phys * wt
            loss_z = loss_z * wt

        # Apply mask and compute mean
        if mask is not None:
            m = mask.unsqueeze(-1).to(loss_phys.dtype)
            m_expanded = m.expand_as(loss_phys)
            loss_phys = loss_phys * m_expanded
            loss_z = loss_z * m_expanded
            denom = torch.count_nonzero(m_expanded).to(loss_phys.dtype)
            denom = torch.clamp(denom, min=1.0)
        else:
            denom = float(loss_phys.numel())

        # Atomic conservation penalty (if enabled)
        cons_elem = 0.0

        if self.elemental_conservation and self.elemental_penalty > 0.0:
            pred_y = torch.pow(10.0, torch.clamp(pred_log, min=-45.0))
            true_y = torch.pow(10.0, torch.clamp(true_log, min=-45.0))

            if pred_y.ndim == 2:
                assert mask is None or mask.ndim == 1, "2D predictions with 2D mask not supported"

            # Device check (only once)
            if not hasattr(self, "_device_checked"):
                assert self.stoich_matrix.device == pred_y.device, \
                    f"stoich_matrix on {self.stoich_matrix.device}, predictions on {pred_y.device}"
                self._device_checked = True

            # Compute elemental totals
            if pred_y.ndim == 2:  # [B,S]
                elem_pred = pred_y @ self.stoich_matrix  # [B,E]
                elem_true = true_y @ self.stoich_matrix  # [B,E]
            else:  # [B,K,S]
                B, K, S = pred_y.shape
                pred_y_flat = pred_y.reshape(B * K, S)
                true_y_flat = true_y.reshape(B * K, S)
                elem_pred_flat = pred_y_flat @ self.stoich_matrix  # [B*K,E]
                elem_true_flat = true_y_flat @ self.stoich_matrix  # [B*K,E]
                elem_pred = elem_pred_flat.reshape(B, K, -1)  # [B,K,E]
                elem_true = elem_true_flat.reshape(B, K, -1)  # [B,K,E]

            # Auto-compute weights on first batch
            if self._need_auto_weights:
                with torch.no_grad():
                    mean_elem = elem_true.mean(dim=tuple(range(elem_true.ndim - 1)))
                    mean_elem = torch.clamp(mean_elem, min=self.eps_elem)
                    new_weights = 1.0 / mean_elem
                    new_weights = new_weights / new_weights.mean()
                    new_weights = torch.clamp(new_weights, min=0.1, max=10.0)
                    self.elem_weights.copy_(new_weights)
                self._need_auto_weights = False

            # Compute conservation error
            if self.elemental_mode == "relative":
                den = torch.abs(elem_true) + torch.abs(elem_pred) + self.eps_elem
                elem_err = torch.abs(elem_pred - elem_true) / den
            else:
                elem_err = torch.abs(elem_pred - elem_true)

            # Apply element weights
            elem_err = elem_err * self.elem_weights

            # Apply mask if present
            if mask is not None:
                if elem_err.ndim == 3:
                    mask_elem = mask.unsqueeze(-1)
                    elem_err = elem_err * mask_elem
                    elem_denom = torch.count_nonzero(mask_elem.expand_as(elem_err)).to(elem_err.dtype)
                else:
                    if mask.ndim == 1:
                        mask_elem = mask.unsqueeze(-1)
                        elem_err = elem_err * mask_elem
                        elem_denom = torch.count_nonzero(mask_elem.expand_as(elem_err)).to(elem_err.dtype)
                    else:
                        elem_denom = float(elem_err.numel())
                elem_denom = torch.clamp(elem_denom, min=1.0)
            else:
                elem_denom = float(elem_err.numel())

            cons_elem = elem_err.sum() / elem_denom

            # Log conservation info on first call
            if not self._logged_conservation:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Elemental conservation enabled:")
                logger.info(f"  Elements: {self.elements}")
                logger.info(f"  Mode: {self.elemental_mode}")
                logger.info(f"  Element weights: {self.elem_weights.detach().cpu().numpy()}")
                logger.info(f"  Species count: {len(self.species_names)}")
                logger.info(f"  First 5 species stoichiometry:")
                for i in range(min(5, len(self.species_names))):
                    stoich_str = ", ".join([f"{e}:{self.stoich_matrix[i, j].item():.0f}"
                                            for j, e in enumerate(self.elements)])
                    logger.info(f"    {self.species_names[i]}: {stoich_str}")
                logger.info(f"  Mean elemental error (pre penalty-weight): {cons_elem.item():.6f}")
                self._logged_conservation = True

        # Calculate component losses
        phys_loss = self.lambda_phys * loss_phys.sum() / denom
        z_loss = self.lambda_z * loss_z.sum() / denom
        elem_loss = self.elemental_penalty * cons_elem

        # Combine all terms
        loss = phys_loss + z_loss + elem_loss

        if return_components:
            return {
                'total': loss,
                'phys': phys_loss,
                'z': z_loss,
                'elem': elem_loss if self.elemental_conservation and self.elemental_penalty > 0 else torch.tensor(0.0)
            }
        return loss


# ------------------------------- Trainer Class -------------------------------


class Trainer:
    """
    Trainer for Flow-map DeepONet models with restart support.

    Handles training loop, validation, checkpointing, and logging.
    Supports both single-time (K=1) and multi-time (K>1) predictions.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader],
            cfg: Dict[str, Any],
            work_dir: Path,
            device: torch.device,
            logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        # Setup logger without name prefix
        if logger is None:
            self.logger = logging.getLogger("trainer")
            self.logger.propagate = False  # Prevent name propagation
            self.logger.handlers.clear()  # Clear any existing handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            self.logger.propagate = False
            # Always override formatter to ensure no name prefix
            self.logger.handlers.clear()
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Setup working directory
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Extract training configuration
        training_cfg = cfg.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 100))
        self.base_lr = float(training_cfg.get("lr", 1e-3))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(training_cfg.get("gradient_clip", 0.0))

        # Optional step limits for debugging/quick testing
        self.max_train_steps_per_epoch = training_cfg.get("max_train_steps_per_epoch", None)
        if self.max_train_steps_per_epoch == 0:
            self.max_train_steps_per_epoch = None
        self.max_val_batches = training_cfg.get("max_val_batches", None)
        if self.max_val_batches == 0:
            self.max_val_batches = None

        # Torch compile option for potential speedup
        self.use_compile = bool(training_cfg.get("torch_compile", False))

        # Setup mixed precision training
        self._setup_mixed_precision()

        # Setup optimizer with automatic fused kernels when available
        self._setup_optimizer()

        # Setup learning rate scheduler using built-in PyTorch schedulers
        self._setup_scheduler()

        # Optionally compile model for better performance
        if self.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False
            )
            self.logger.info("Model compiled with torch.compile")

        # Initialize training state
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self._current_epoch = 0  # Track current epoch for SIGTERM handler

        # Initialize loss component tracking (absolute and relative)
        self.train_loss_components = {}
        self.val_loss_components = {}
        self.train_rel_components = {}
        self.val_rel_components = {}

        # Checkpoint paths
        self.best_model_path = self.work_dir / "best_model.pt"  # Weights only (backward compat)
        self.best_ckpt_path = self.work_dir / "best.ckpt"  # Full checkpoint
        self.last_ckpt_path = self.work_dir / "last.ckpt"  # Full checkpoint for resume

        # Setup SIGTERM handler for graceful cluster preemption
        self._setup_sigterm_handler()

        # Setup loss function BEFORE logging (FIX #1)
        self._setup_loss()

        # Setup logging AFTER loss (so we know elem_enabled)
        self._setup_logging()

        # Check for resume from checkpoint (must come after scheduler setup)
        self._check_resume()

    def _setup_loss(self):
        """Setup the loss function based on configuration."""
        training_cfg = self.cfg.get("training", {})
        loss_mode = training_cfg.get("loss_mode", "adaptive_stiff")

        if loss_mode == "adaptive_stiff":
            # Load normalization manifest for statistics
            manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Robustly resolve species names
            data_cfg = self.cfg.get("data", {})

            # Try multiple sources for species names
            target_vars = data_cfg.get("target_species")
            if target_vars:
                species_names = list(target_vars)
            else:
                species_vars = data_cfg.get("species_variables")
                if species_vars:
                    species_names = list(species_vars)
                else:
                    # Fall back to normalization.json meta or per_key_stats
                    meta = manifest.get("meta", {})
                    species_from_meta = meta.get("species_variables")
                    if species_from_meta:
                        species_names = list(species_from_meta)
                    else:
                        # Last resort: use keys from per_key_stats that are not time/globals
                        stats = manifest["per_key_stats"]
                        time_var = data_cfg.get("time_variable", "t_time")
                        global_vars = set(data_cfg.get("global_variables", []))
                        species_names = [
                            k for k in stats.keys()
                            if k != time_var and k not in global_vars
                        ]

            if not species_names:
                raise ValueError(
                    "Could not determine species names from config or normalization.json"
                )

            # If using target subset, use those for statistics
            if target_vars:
                stats_keys = target_vars
            else:
                stats_keys = species_names

            # Extract statistics for loss computation species
            stats = manifest["per_key_stats"]
            log_means = []
            log_stds = []
            log_mins = []
            log_maxs = []

            for name in stats_keys:
                if name not in stats:
                    raise KeyError(f"Species '{name}' not found in normalization statistics")
                s = stats[name]
                log_means.append(float(s.get("log_mean", 0.0)))
                log_stds.append(float(s.get("log_std", 1.0)))
                log_mins.append(float(s.get("log_min", -10.0)))
                log_maxs.append(float(s.get("log_max", 10.0)))

            # Get loss configuration
            loss_cfg = training_cfg.get("adaptive_stiff_loss", {})

            # Determine if we need elemental conservation
            elemental_conservation = bool(loss_cfg.get("elemental_conservation", False))
            elemental_penalty = float(loss_cfg.get("elemental_penalty", 0.0))

            # Track if elem is enabled
            self.elem_enabled = elemental_conservation and elemental_penalty > 0

            # Log resolved species for debugging
            self.logger.info(f"Resolved {len(species_names)} species for loss computation")
            self.logger.info(f"  First 5: {species_names[:5]}")

            # Create adaptive stiff loss with device parameter
            self.criterion = AdaptiveStiffLoss(
                log_means=torch.tensor(log_means, device=self.device),
                log_stds=torch.tensor(log_stds, device=self.device),
                species_log_min=torch.tensor(log_mins, device=self.device),
                species_log_max=torch.tensor(log_maxs, device=self.device),
                lambda_phys=float(loss_cfg.get("lambda_phys", 1.0)),
                lambda_z=float(loss_cfg.get("lambda_z", 0.1)),
                epsilon_phys=float(loss_cfg.get("epsilon_phys", 1e-20)),
                use_fractional=bool(loss_cfg.get("use_fractional", False)),
                time_edge_gain=float(loss_cfg.get("time_edge_gain", 2.0)),
                # Atomic conservation parameters
                elemental_conservation=elemental_conservation,
                elemental_penalty=elemental_penalty,
                species_names=species_names if self.elem_enabled else None,
                elements=loss_cfg.get("elements", ["H", "C", "N", "O"]),
                elemental_mode=loss_cfg.get("elemental_mode", "relative"),
                elemental_weights=loss_cfg.get("elemental_weights", "auto"),
                eps_elem=float(loss_cfg.get("eps_elem", 1e-20)),
                debug_parser=bool(loss_cfg.get("debug_parser", False)),
                device=self.device
            )

            # CRITICAL: Move the entire loss module to device
            self.criterion = self.criterion.to(self.device)

            # ============ Check model/loss stat consistency ============
            # If the model uses SoftMax head or corrected residual, verify stats match
            model_cfg = self.cfg.get("model", {})
            if model_cfg.get("softmax_head", False) or model_cfg.get("predict_delta_log_phys", False):
                if hasattr(self.model, 'check_stat_consistency'):
                    try:
                        self.model.check_stat_consistency(
                            self.criterion.log_means,
                            self.criterion.log_stds
                        )
                        self.logger.info("Model and loss normalization statistics verified to match")
                    except (ValueError, RuntimeError) as e:
                        self.logger.error(f"CRITICAL: Model/loss stat mismatch: {e}")
                        raise
                else:
                    self.logger.warning(
                        "Model should have check_stat_consistency method when using "
                        "softmax_head or predict_delta_log_phys. Skipping stat verification."
                    )
            # ============================================================

            self.use_adaptive_stiff = True
            self.logger.info("Using AdaptiveStiffLoss with dynamic range weighting")
            if self.elem_enabled:
                self.logger.info(f"  Atomic conservation penalty: {elemental_penalty}")

            self._need_species_validation = True
        else:
            # Use standard loss modes
            self.criterion = None
            self.use_adaptive_stiff = False
            self.elem_enabled = False
            # Store loss configuration for _compute_loss method
            self.loss_mode = loss_mode
            self.loss_epsilon = float(training_cfg.get("loss_epsilon", 1e-27))
            self.loss_rel_cap = training_cfg.get("loss_rel_cap", None)

    def _setup_sigterm_handler(self) -> None:
        """Setup handler for SIGTERM signal (cluster preemption)."""

        def sigterm_handler(signum, frame):
            self.logger.warning("SIGTERM received: saving checkpoint and exiting")
            # Save checkpoint at current epoch
            self._save_full_checkpoint(self.last_ckpt_path, self._current_epoch)
            os._exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)

    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training with CPU safety guards."""
        mixed_precision_cfg = self.cfg.get("mixed_precision", {})
        self.amp_mode = str(mixed_precision_cfg.get("mode", "bf16")).lower()

        if self.amp_mode not in ("bf16", "fp16", "none"):
            self.amp_mode = "bf16"  # Default to bf16 on modern hardware

        # Determine autocast dtype
        if self.amp_mode == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif self.amp_mode == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = None

        # GradScaler is only needed for fp16 mode on CUDA
        # bf16 doesn't need scaling as it has better numerical range
        self.use_fp16_scaler = (self.amp_mode == "fp16" and self.device.type == "cuda")

        if self.use_fp16_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        self.logger.info(f"Mixed precision mode: {self.amp_mode}")

    def _setup_optimizer(self) -> None:
        """Initialize optimizer with automatic fused kernel detection."""
        # Try to use fused AdamW for better performance on modern GPUs
        # PyTorch will automatically fall back if not supported
        try:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
                fused=True,  # Use fused kernels when available
            )
            self.logger.info("Using fused AdamW optimizer")
        except (TypeError, RuntimeError):
            # Fallback to standard AdamW if fused not supported
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )
            self.logger.info("Using standard AdamW optimizer")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler using PyTorch's built-in schedulers."""
        training_cfg = self.cfg.get("training", {})
        warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
        min_lr = float(training_cfg.get("min_lr", 1e-6))

        if warmup_epochs > 0:
            # Linear warmup from small factor to full learning rate
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.001,  # Start at 0.1% of base lr for smoother warmup
                end_factor=1.0,
                total_iters=warmup_epochs
            )

            # Cosine annealing for remaining epochs
            cosine_epochs = max(self.epochs - warmup_epochs, 1)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=min_lr
            )

            # Combine schedulers sequentially
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            self.logger.info(f"LR schedule: {warmup_epochs} warmup + cosine annealing to {min_lr:.2e}")
        else:
            # Just cosine annealing without warmup
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=min_lr
            )
            self.logger.info(f"LR schedule: cosine annealing to {min_lr:.2e}")

    def _setup_logging(self) -> None:
        """Initialize TensorBoard and CSV logging."""
        # TensorBoard writer for rich visualization
        tb_dir = self.work_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # CSV file with complete headers (FIX #3: include val relatives)
        self.log_file = self.work_dir / "training_log.txt"

        # Build headers based on loss type
        headers = ["epoch", "train", "val", "lr"]
        if self.use_adaptive_stiff:
            headers.extend(["train_rel_phys", "train_rel_z"])
            if self.elem_enabled:
                headers.append("train_rel_elem")
            headers.extend(["val_rel_phys", "val_rel_z"])
            if self.elem_enabled:
                headers.append("val_rel_elem")

        # Only write header if file doesn't exist
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _check_resume(self) -> None:
        """Load a checkpoint if resuming, honoring env var RESUME, then config:
           training.resume ('auto' | <path> | None) and training.auto_resume (bool)."""
        tr = self.cfg.get("training", {})
        env_resume = os.environ.get("RESUME", "").strip()

        # 1) Environment variable has highest precedence (backwards compatible)
        if env_resume:
            resume_spec = env_resume
            use_auto = (env_resume.lower() == "auto")
        else:
            # 2) Config-driven behavior
            auto_resume = bool(tr.get("auto_resume", False))
            resume_cfg = tr.get("resume", None)
            if resume_cfg is None:
                if not auto_resume:
                    return  # nothing to do
                # auto_resume=True with no explicit path -> auto discovery
                resume_spec = "auto"
                use_auto = True
            else:
                # explicit value in config
                resume_spec = str(resume_cfg)
                use_auto = (resume_spec.lower() == "auto")

        # Resolve checkpoint path
        if use_auto:
            # Prefer 'last.ckpt', then 'best.ckpt', then newest *.ckpt in work_dir
            if self.last_ckpt_path.exists():
                ckpt_path = self.last_ckpt_path
            elif self.best_ckpt_path.exists():
                ckpt_path = self.best_ckpt_path
            else:
                ckpts = sorted(
                    self.work_dir.glob("*.ckpt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if ckpts:
                    ckpt_path = ckpts[0]
                else:
                    self.logger.info("No checkpoint found for auto-resume")
                    return
        else:
            ckpt_path = Path(resume_spec).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load checkpoint
        self.logger.info(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Restore model/optimizer
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore scheduler (or fast-forward if missing)
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.logger.info("Restored scheduler state from checkpoint")
        else:
            start_epoch = checkpoint.get("epoch", 0)
            for _ in range(start_epoch):
                self.scheduler.step()
            self.logger.info(f"Fast-forwarded scheduler to epoch {start_epoch}")

        # Restore scaler if present/used
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Restore training state
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self._current_epoch = self.start_epoch

        # Restore RNG state (optional fields guarded)
        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            if "python" in rng_state: random.setstate(rng_state["python"])
            if "numpy" in rng_state:  np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:  torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])

        self.logger.info(f"Resumed from epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4e}")

    def _save_full_checkpoint(self, path: Path, epoch: int) -> None:
        """Save full checkpoint for restart with atomic write, including scheduler."""
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),  # Save scheduler state
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
        }

        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

        if self.scaler:
            checkpoint["scaler"] = self.scaler.state_dict()

        # Atomic write: save to temp file then rename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        os.replace(str(tmp_path), str(path))

    def _save_weights_only(self) -> None:
        """Save model weights only for inference and backward compatibility."""
        checkpoint = {
            "model": self.model.state_dict(),
            "config": self.cfg,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.best_model_path)

    def train(self) -> float:
        """
        Execute training loop.

        Returns:
            Best validation loss achieved during training
        """
        start_time = time.perf_counter()

        try:
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                # Track current epoch for SIGTERM handler
                self._current_epoch = epoch

                # Set epoch for datasets that support deterministic sampling
                # This is the primary mechanism for deterministic per-epoch sampling
                self._set_dataset_epoch(epoch)

                epoch_start = time.perf_counter()

                # Training epoch
                train_loss = self._run_epoch(train=True)

                # Validation epoch
                if self.val_loader is not None:
                    val_loss = self._run_epoch(train=False)
                else:
                    val_loss = train_loss
                    # Copy relative components when no validation
                    self.val_rel_components = self.train_rel_components.copy()

                # Step learning rate scheduler AFTER optimizer updates (epoch-level scheduling)
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Save checkpoints (without logging saved status)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)
                    self._save_weights_only()  # For backward compatibility
                    self._save_full_checkpoint(self.best_ckpt_path, epoch)

                # Always save last checkpoint for resume capability
                self._save_full_checkpoint(self.last_ckpt_path, epoch)

                # Log metrics to TensorBoard
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

                # Log absolute component losses to TensorBoard
                if self.train_loss_components:
                    for key, value in self.train_loss_components.items():
                        self.writer.add_scalar(f"loss/train_{key}", value, epoch)
                if self.val_loss_components:
                    for key, value in self.val_loss_components.items():
                        self.writer.add_scalar(f"loss/val_{key}", value, epoch)

                # Log relative component losses to TensorBoard
                if self.train_rel_components:
                    for key, value in self.train_rel_components.items():
                        self.writer.add_scalar(f"loss_rel/train_{key}", value, epoch)
                if self.val_rel_components:
                    for key, value in self.val_rel_components.items():
                        self.writer.add_scalar(f"loss_rel/val_{key}", value, epoch)

                # Log to CSV file
                self._log_epoch_metrics(epoch, train_loss, val_loss, current_lr)

                # Console output with BOTH train and val relative components (FIX #2)
                epoch_time = time.perf_counter() - epoch_start

                # Build relative loss strings for both train and val
                rel_train_str = ""
                rel_val_str = ""
                if self.use_adaptive_stiff:
                    if self.train_rel_components:
                        if self.elem_enabled:
                            rel_train_str = f" | rel_train[phys/z/elem]={self.train_rel_components['phys']:.2f}/{self.train_rel_components['z']:.2f}/{self.train_rel_components['elem']:.2f}"
                        else:
                            rel_train_str = f" | rel_train[phys/z]={self.train_rel_components['phys']:.2f}/{self.train_rel_components['z']:.2f}"

                    if self.val_rel_components:
                        if self.elem_enabled:
                            rel_val_str = f" | rel_val[phys/z/elem]={self.val_rel_components['phys']:.2f}/{self.val_rel_components['z']:.2f}/{self.val_rel_components['elem']:.2f}"
                        else:
                            rel_val_str = f" | rel_val[phys/z]={self.val_rel_components['phys']:.2f}/{self.val_rel_components['z']:.2f}"

                self.logger.info(
                    f"Epoch {epoch:03d}/{self.epochs} | "
                    f"train={train_loss:.4e} | "
                    f"val={val_loss:.4e} | "
                    f"lr={current_lr:.2e} | "
                    f"time={epoch_time:.1f}s"
                    f"{rel_train_str}"
                    f"{rel_val_str}"
                )

            # Training complete
            total_time = time.perf_counter() - start_time
            self.logger.info(
                f"Training completed in {total_time / 3600:.2f} hours. "
                f"Best validation loss: {self.best_val_loss:.4e}"
            )

            # Compute final validation metric in physical units
            which = "last-epoch"
            try:
                best_blob = torch.load(self.best_model_path, map_location=self.device)
                best_state = best_blob.get("model") if isinstance(best_blob, dict) else None
                if best_state:
                    self.model.load_state_dict(best_state, strict=False)
                    which = "best_model.pt"
            except Exception:
                pass

            final_frac = self.evaluate_frac_l1_phys(self.val_loader)
            self.logger.info(f"Final validation fractional L1 (physical) [{which}]: {final_frac:.6e}")

            # Save final metrics for downstream analysis
            try:
                with open(self.work_dir / "final_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"val_frac_l1_phys_mean": final_frac, "which": which}, f, indent=2)
            except Exception:
                pass

        finally:
            # Always close TensorBoard writer, even if training is interrupted
            self.writer.close()

        return self.best_val_loss

    def _set_dataset_epoch(self, epoch: int) -> None:
        """Set epoch for datasets that support deterministic sampling."""
        for loader in (self.train_loader, self.val_loader):
            if loader is not None:
                dataset = getattr(loader, "dataset", None)
                if dataset is not None and hasattr(dataset, "set_epoch"):
                    try:
                        dataset.set_epoch(epoch)
                    except Exception:
                        pass

    def _log_epoch_metrics(
            self,
            epoch: int,
            train_loss: float,
            val_loss: float,
            lr: float
    ) -> None:
        """Append epoch metrics to CSV log file with relative components for BOTH train and val (FIX #3)."""
        row = [epoch, f"{train_loss:.4e}", f"{val_loss:.4e}", f"{lr:.4e}"]

        if self.use_adaptive_stiff:
            # Train relatives
            if self.train_rel_components:
                row.append(f"{self.train_rel_components.get('phys', 0):.4f}")
                row.append(f"{self.train_rel_components.get('z', 0):.4f}")
                if self.elem_enabled:
                    row.append(f"{self.train_rel_components.get('elem', 0):.4f}")
            else:
                row.extend(["0.0000", "0.0000"])
                if self.elem_enabled:
                    row.append("0.0000")

            # Val relatives
            if self.val_rel_components:
                row.append(f"{self.val_rel_components.get('phys', 0):.4f}")
                row.append(f"{self.val_rel_components.get('z', 0):.4f}")
                if self.elem_enabled:
                    row.append(f"{self.val_rel_components.get('elem', 0):.4f}")
            else:
                row.extend(["0.0000", "0.0000"])
                if self.elem_enabled:
                    row.append("0.0000")

        with self.log_file.open("a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _run_epoch(self, train: bool) -> float:
        """
        Run single training or validation epoch.

        Args:
            train: Whether to run training (True) or validation (False)

        Returns:
            Average loss for the epoch
        """
        # Select appropriate loader
        loader = self.train_loader if train else self.val_loader
        if loader is None:
            return float("nan")

        # Set model mode
        self.model.train(mode=train)

        # Setup autocast context for mixed precision with CPU safety
        if self.autocast_dtype is not None:
            # Only enable autocast for CUDA or if explicitly using bf16 on CPU
            enable_amp = (self.device.type == "cuda" or
                          (self.device.type == "cpu" and self.amp_mode == "bf16"))
            autocast_context = torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=enable_amp
            )
        else:
            autocast_context = contextlib.nullcontext()

        # Epoch statistics
        total_loss = 0.0
        num_batches = 0

        # Component loss accumulators
        total_phys = 0.0
        total_z = 0.0
        total_elem = 0.0

        # Determine max steps for this epoch
        max_steps = None
        if train and self.max_train_steps_per_epoch:
            max_steps = self.max_train_steps_per_epoch
        elif not train and self.max_val_batches:
            max_steps = self.max_val_batches

        # Use inference mode for validation (disables autograd)
        outer_ctx = torch.inference_mode() if not train else contextlib.nullcontext()

        with outer_ctx:
            # Process batches
            for step, batch in enumerate(loader, 1):
                # Check step limit
                if max_steps and step > max_steps:
                    break

                # Process batch
                loss_info = self._process_batch(batch, train, autocast_context)

                if isinstance(loss_info, dict):
                    total_loss += float(loss_info['total'])
                    total_phys += float(loss_info.get('phys', 0))
                    total_z += float(loss_info.get('z', 0))
                    total_elem += float(loss_info.get('elem', 0))
                else:
                    total_loss += float(loss_info)

                num_batches += 1

        # Log actual batch counts when limits are active
        if max_steps:
            mode_str = "train" if train else "val"
            self.logger.debug(f"{mode_str} epoch used {num_batches} batches (limit: {max_steps})")

        # Calculate averages
        avg_loss = total_loss / max(1, num_batches)

        # Store absolute component averages and compute relatives
        if num_batches > 0 and self.use_adaptive_stiff:
            components = {
                'phys': total_phys / num_batches,
                'z': total_z / num_batches,
                'elem': total_elem / num_batches
            }

            # Calculate relative components
            total = avg_loss + 1e-12  # Guard against division by zero
            rel_components = {
                'phys': components['phys'] / total,
                'z': components['z'] / total,
                'elem': components['elem'] / total
            }

            if train:
                self.train_loss_components = components
                self.train_rel_components = rel_components
            else:
                self.val_loss_components = components
                self.val_rel_components = rel_components

        return avg_loss

    def _process_batch(
            self,
            batch: tuple,
            train: bool,
            autocast_context: contextlib.AbstractContextManager
    ) -> float | Dict[str, float]:
        """
        Process single batch with mixed precision support.

        Args:
            batch: Batch data tuple
            train: Whether to compute gradients and update weights
            autocast_context: Context manager for mixed precision

        Returns:
            Batch loss value or dict of loss components
        """
        # Unpack batch (supports both 5 and 6 element batches)
        if len(batch) == 6:
            y_i, dt_norm, y_j, g, ij, k_mask = batch
        else:
            y_i, dt_norm, y_j, g, ij = batch
            k_mask = None

        # Move to device efficiently
        y_i = y_i.to(self.device, non_blocking=True)
        dt_norm = dt_norm.to(self.device, non_blocking=True)
        y_j = y_j.to(self.device, non_blocking=True)
        g = g.to(self.device, non_blocking=True)
        if k_mask is not None:
            k_mask = k_mask.to(self.device, non_blocking=True)

        # Lazily resolve target indices for species subset
        if not hasattr(self, "_target_idx"):
            self._resolve_target_indices()

        # Slice targets to match model's output dimension
        if self._target_idx is not None:
            y_j = y_j.index_select(dim=-1, index=self._target_idx)

        # Zero gradients efficiently
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        # Forward pass with autocast
        with autocast_context:
            # Prepare time input (remove trailing singleton if present)
            if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
                dt_in = dt_norm.squeeze(-1)
            else:
                dt_in = dt_norm

            # Forward pass
            pred = self.model(y_i, dt_in, g)

            # Validate and harmonize shapes
            pred, y_j = self._harmonize_shapes(pred, y_j)

            # Validate species dimension on first forward
            if hasattr(self, '_need_species_validation') and self._need_species_validation and self.use_adaptive_stiff:
                if hasattr(self.criterion, 'species_names') and self.criterion.species_names:
                    expected_s = len(self.criterion.species_names)
                    actual_s = pred.shape[-1]
                    if expected_s != actual_s:
                        raise ValueError(
                            f"Model output dimension mismatch: expected {expected_s} species, got {actual_s}. "
                            f"Check target_species configuration."
                        )
                    self.logger.info(f"Model output validated: {actual_s} species")
                self._need_species_validation = False

            # Compute loss with components if using adaptive stiff
            if self.use_adaptive_stiff:
                loss_info = self.criterion(pred, y_j, dt_in, k_mask, return_components=True)
                loss = loss_info['total']
            else:
                loss = self._compute_loss(pred, y_j, k_mask)
                loss_info = loss

        # Backward pass and optimization (training only)
        if train:
            if self.use_fp16_scaler and self.scaler is not None:
                # Scaled backward for fp16
                self.scaler.scale(loss).backward()

                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping if configured
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip
                    )

                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard backward pass
                loss.backward()

                # Gradient clipping if configured
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip
                    )

                # Optimizer step
                self.optimizer.step()

        # Return loss components for adaptive stiff, scalar for others
        if isinstance(loss_info, dict):
            return {k: v.detach().item() for k, v in loss_info.items()}
        return loss.detach().item()

    def _resolve_target_indices(self):
        """Resolve target species indices from configuration."""
        data_cfg = self.cfg.get("data", {})
        species_vars = list(data_cfg.get("species_variables") or [])
        target_vars = list(data_cfg.get("target_species") or species_vars)

        if target_vars != species_vars:
            name_to_idx = {n: i for i, n in enumerate(species_vars)}
            try:
                idx_list = [name_to_idx[n] for n in target_vars]
            except KeyError as e:
                raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")
            self._target_idx = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        else:
            self._target_idx = None

    def _harmonize_shapes(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and ensure prediction and target have compatible shapes.

        Args:
            pred: Model predictions
            target: Target values

        Returns:
            Tuple of (pred, target) with compatible shapes

        Raises:
            RuntimeError: If shapes are incompatible
        """
        # Check for shape mismatches and provide helpful error messages
        if pred.ndim == 2 and target.ndim == 3:
            B_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}. "
                f"The model should return [B, K, S] when K={K_target} times are requested."
            )

        elif pred.ndim == 3 and target.ndim == 2:
            B_pred, K_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            raise RuntimeError(
                f"Shape mismatch: Model returned predictions with shape {tuple(pred.shape)} "
                f"but target has shape {tuple(target.shape)}."
            )

        elif pred.ndim == 3 and target.ndim == 3:
            B_pred, K_pred, S_pred = pred.shape
            B_target, K_target, S_target = target.shape
            if K_pred != K_target:
                raise RuntimeError(f"Shape mismatch in time dimension")
            if B_pred != B_target:
                raise RuntimeError("Batch size mismatch")
            if S_pred != S_target:
                raise RuntimeError("State dimension mismatch")

        elif pred.ndim == 2 and target.ndim == 2:
            B_pred, S_pred = pred.shape
            B_target, S_target = target.shape
            if B_pred != B_target:
                raise RuntimeError("Batch size mismatch")
            if S_pred != S_target:
                raise RuntimeError("State dimension mismatch")

        else:
            raise RuntimeError("Unexpected tensor dimensions")

        return pred, target

    def _compute_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss using standard loss modes.

        Supports:
        - 'mse': Mean squared error in normalized space
        - 'frac_l1_phys': Fractional L1 in physical space
        - 'mae_log_phys': MAE in log10 physical space
        """
        # Get loss configuration
        tr_cfg = self.cfg.get("training", {})
        mode = getattr(self, "loss_mode", tr_cfg.get("loss_mode", "mse"))
        eps = float(getattr(self, "loss_epsilon", tr_cfg.get("loss_epsilon", 1e-27)))
        rel_cap = getattr(self, "loss_rel_cap", tr_cfg.get("loss_rel_cap", None))

        if mode == "mse":
            loss_elems = (pred - target) ** 2

        elif mode in ("frac_l1_phys", "mae_log_phys"):
            # Lazy-load normalization helper on first use
            if not hasattr(self, "_norm_helper"):
                from normalizer import NormalizationHelper

                manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                self._norm_helper = NormalizationHelper(manifest, device=pred.device)

                data_cfg = self.cfg.get("data", {})
                self._species_keys = list(
                    data_cfg.get("target_species") or data_cfg.get("species_variables")
                )

            # Denormalize to physical units
            y_pred = self._norm_helper.denormalize(pred, self._species_keys)
            y_true = self._norm_helper.denormalize(target, self._species_keys)

            if mode == "frac_l1_phys":
                # Fractional absolute error: |pred - true| / (|true| + eps)
                denom = y_true.abs() + eps
                rel = (y_pred - y_true).abs() / denom
                if rel_cap is not None:
                    rel = torch.clamp(rel, max=float(rel_cap))
                loss_elems = rel

            else:  # 'mae_log_phys'
                # MAE in log10 space (smoother proxy for fractional error)
                y_pred = torch.clamp(y_pred, min=eps)
                y_true = torch.clamp(y_true, min=eps)
                loss_elems = (torch.log10(y_pred) - torch.log10(y_true)).abs()

            # Optional per-species weights (if configured)
            w = getattr(self, "loss_weights", None)
            if w is not None:
                w = w.to(loss_elems.device, dtype=loss_elems.dtype)
                while w.ndim < loss_elems.ndim:
                    w = w.unsqueeze(0)
                loss_elems = loss_elems * w

        else:
            raise ValueError(f"Unknown loss_mode='{mode}'")

        # Masked reduction for multi-time batches
        if mask is not None:
            if loss_elems.ndim == 3 and mask.ndim == 2:
                # Expand mask to match loss shape
                mask_exp = mask.unsqueeze(-1).expand_as(loss_elems)
                # Use count_nonzero for clearer counting
                denom = torch.count_nonzero(mask_exp).to(loss_elems.dtype)
                denom = torch.clamp(denom, min=1.0)
                return (loss_elems * mask_exp).sum() / denom
            else:
                denom = torch.count_nonzero(mask).to(loss_elems.dtype)
                denom = torch.clamp(denom, min=1.0)
                return (loss_elems * mask).sum() / denom

        return loss_elems.mean()

    def _ensure_norm_helper(self):
        """
        Lazily load NormalizationHelper and resolve the correct species list.
        Used for evaluation metrics.
        """
        if not hasattr(self, "_norm_helper"):
            from normalizer import NormalizationHelper

            manifest_path = Path(self.cfg["paths"]["processed_data_dir"]) / "normalization.json"
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            self._norm_helper = NormalizationHelper(manifest, device=self.device)

            data_cfg = self.cfg.get("data", {})
            self._species_keys = list(
                data_cfg.get("target_species") or data_cfg.get("species_variables")
            )
        return self._norm_helper, self._species_keys

    @torch.inference_mode()
    def evaluate_frac_l1_phys(
            self,
            loader: Optional[torch.utils.data.DataLoader] = None,
            max_batches: Optional[int] = None
    ) -> float:
        """
        Compute mean fractional absolute error in PHYSICAL units.

        This is the true evaluation metric: mean(|y_pred - y_true| / (|y_true| + eps))

        Args:
            loader: DataLoader to evaluate on (defaults to validation loader)
            max_batches: Maximum number of batches to evaluate

        Returns:
            Mean fractional L1 error in physical space
        """
        loader = loader or self.val_loader
        if loader is None:
            return float("nan")

        self.model.eval()
        eps = float(self.cfg.get("training", {}).get("loss_epsilon", 1e-27))
        norm, species_keys = self._ensure_norm_helper()

        # Accumulate statistics using tensors for consistency
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        count = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Ensure we have target index mapping
        if not hasattr(self, "_target_idx"):
            self._resolve_target_indices()

        for step, batch in enumerate(loader, 1):
            if max_batches and step > max_batches:
                break

            # Unpack batch
            if len(batch) == 6:
                y_i, dt_norm, y_j, g, ij, k_mask = batch
            else:
                y_i, dt_norm, y_j, g, ij = batch
                k_mask = None

            # Move to device
            y_i = y_i.to(self.device, non_blocking=True)
            dt = dt_norm.to(self.device, non_blocking=True)
            y_j = y_j.to(self.device, non_blocking=True)
            g = g.to(self.device, non_blocking=True)
            if k_mask is not None:
                k_mask = k_mask.to(self.device, non_blocking=True)

            # Slice targets if needed
            if self._target_idx is not None:
                y_j = y_j.index_select(dim=-1, index=self._target_idx)

            # Forward pass (match training path)
            dt_in = dt.squeeze(-1) if (dt.ndim == 3 and dt.shape[-1] == 1) else dt
            pred = self.model(y_i, dt_in, g)
            pred, tgt = self._harmonize_shapes(pred, y_j)

            # Denormalize to physical units
            y_pred = norm.denormalize(pred, species_keys)
            y_true = norm.denormalize(tgt, species_keys)

            # Compute fractional L1 error
            rel = (y_pred - y_true).abs() / (y_true.abs() + eps)

            # Apply mask if present
            if k_mask is not None and rel.ndim == 3 and k_mask.ndim == 2:
                w = k_mask.unsqueeze(-1).expand_as(rel)  # [B,K,S]
                total += (rel * w).sum()
                count += torch.count_nonzero(w).to(total.dtype)
            else:
                total += rel.sum()
                count += torch.tensor(rel.numel(), device=self.device, dtype=total.dtype)

        return (total / count).item() if count.item() > 0 else float("nan")