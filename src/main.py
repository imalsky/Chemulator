#!/usr/bin/env python3
"""
main.py
- JSON/JSONC config loading
- Hardware setup (TF32, cuDNN benchmark, thread control)
- Preprocess if needed, then hydrate cfg.data.* from artifacts
- Build datasets/dataloaders
- Build model
- Train via Lightning-backed Trainer (CSV only)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Resolve duplicate OpenMP on macOS
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from preprocessor import DataPreprocessor
from trainer import Trainer
from utils import (
    dump_json,
    load_json_config,
    resolve_precision_and_dtype,
    seed_everything,
    setup_logging,
)

# =============================================================================
# Constants
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
GLOBAL_SEED = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "autoencoder-flowmap"
VALIDATION_SEED_OFFSET = 1337

DDP_WAIT_TIMEOUT_SECS = 60.0
DDP_WAIT_INTERVAL_SECS = 0.1


# =============================================================================
# DDP / MPI Environment Normalization
# =============================================================================


def normalize_distributed_env() -> None:
    """
    Lightning/DDP expects torchrun-style env vars (RANK/WORLD_SIZE/LOCAL_RANK).
    Map common MPI/PMI/MVAPICH/SLURM env vars to torchrun equivalents when missing.
    """
    rank_vars = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK")
    world_vars = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
    local_vars = (
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "CUDA_LOCAL_RANK",
    )

    if os.getenv("RANK") is None:
        for k in rank_vars:
            v = os.getenv(k)
            if v is not None:
                os.environ["RANK"] = v
                break

    if os.getenv("WORLD_SIZE") is None:
        for k in world_vars:
            v = os.getenv(k)
            if v is not None:
                os.environ["WORLD_SIZE"] = v
                break

    if os.getenv("LOCAL_RANK") is None:
        for k in local_vars:
            v = os.getenv(k)
            if v is not None:
                os.environ["LOCAL_RANK"] = v
                break


# =============================================================================
# Hardware Setup
# =============================================================================


def setup_device(logger: logging.Logger) -> torch.device:
    """
    Device selection:
      - CUDA: uses LOCAL_RANK (or defaults to 0) and calls torch.cuda.set_device.
      - MPS: uses Apple MPS if available.
      - CPU: fallback.
    """
    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", "0") or "0")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        try:
            dev_name = torch.cuda.get_device_name(local_rank)
        except Exception:
            dev_name = "unknown"
        logger.info(f"Set CUDA device to local_rank={local_rank} ({dev_name})")
        return device

    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return torch.device("mps")

    logger.info("Using CPU")
    return torch.device("cpu")


def optimize_hardware(system_cfg: Dict[str, Any], device: torch.device, logger: logging.Logger) -> None:
    """
    Hardware optimization:
      - TF32 / matmul precision
      - cuDNN benchmark mode
      - CPU thread settings
    """
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tf32 = bool(system_cfg.get("tf32", system_cfg.get("allow_tf32", True)))

    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = tf32
        except Exception:
            pass

    omp = system_cfg.get("omp_num_threads")
    if omp is not None:
        try:
            omp_int = int(omp)
            if omp_int > 0:
                os.environ["OMP_NUM_THREADS"] = str(omp_int)
                torch.set_num_threads(omp_int)
                logger.info(f"Set OMP_NUM_THREADS={omp_int}, torch.num_threads={omp_int}")
        except Exception as e:
            logger.warning(f"Failed to set thread count: {e}")

    cudnn_benchmark = bool(system_cfg.get("cudnn_benchmark", True))
    try:
        torch.backends.cudnn.benchmark = cudnn_benchmark
    except Exception:
        pass

    logger.info(f"Hardware: TF32={tf32}, cudnn.benchmark={cudnn_benchmark}")


# =============================================================================
# Config Hydration from Processed Artifacts
# =============================================================================


def hydrate_config_from_processed(
    cfg: Dict[str, Any],
    logger: logging.Logger,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Hydrate cfg.data from processed artifacts (normalization.json / preprocessing_summary.json).

    Rules:
      - cfg.data.species_variables and cfg.data.target_species take precedence if non-empty.
      - Empty or missing species/targets are filled from artifacts.
      - species_variables and target_species must be identical after resolution.
    """
    if processed_dir is None:
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(f"[hydrate] Missing processed data dir: {processed_dir}")

    norm_path = processed_dir / "normalization.json"
    summary_path = processed_dir / "preprocessing_summary.json"

    def _load_json(p: Path) -> Optional[Dict[str, Any]]:
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[hydrate] Failed to read %s: %s", p.name, e)
            return None

    def _preview(items: Sequence[str], max_items: int = 10) -> str:
        items = list(items)
        if len(items) <= max_items:
            return "[" + ", ".join(repr(x) for x in items) + "]"
        head = ", ".join(repr(x) for x in items[:max_items])
        return f"[{head}, ...] (+{len(items) - max_items} more)"

    manifest = _load_json(norm_path)
    summary = _load_json(summary_path)

    meta = (manifest or {}).get("meta", {}) if manifest else {}
    processed_species = meta.get("species_variables") or (summary or {}).get("species_variables")
    processed_globals = meta.get("global_variables") or (summary or {}).get("global_variables")
    processed_time_var = meta.get("time_variable") or (summary or {}).get("time_variable")

    if not processed_species:
        raise RuntimeError(
            "[hydrate] Unable to determine processed species_variables from artifacts "
            f"in {processed_dir}"
        )
    processed_species = list(processed_species)
    processed_set = set(processed_species)

    # Validate processed invariants
    processed_targets = (
        meta.get("target_species")
        or (manifest or {}).get("target_species")
        or (summary or {}).get("target_species")
    )
    if processed_targets is not None:
        processed_targets = list(processed_targets)
        if processed_targets != processed_species:
            raise RuntimeError(
                "[hydrate] Processed artifacts have inconsistent species/targets. "
                f"species={_preview(processed_species)}; targets={_preview(processed_targets)}. "
                "This codebase requires species_variables == target_species."
            )

    data_cfg = cfg.setdefault("data", {})

    cfg_species_raw = data_cfg.get("species_variables", []) or []
    cfg_targets_raw = data_cfg.get("target_species", []) or []

    cfg_species: List[str] = list(cfg_species_raw) if cfg_species_raw else []
    cfg_targets: List[str] = list(cfg_targets_raw) if cfg_targets_raw else []

    def _check_duplicates(items: Sequence[str], label: str) -> None:
        items = list(items)
        if len(items) != len(set(items)):
            dupes = [x for x in dict.fromkeys(items) if items.count(x) > 1]
            raise ValueError(f"[hydrate] {label} contains duplicate entries: {dupes}")

    species_specified = len(cfg_species) > 0
    targets_specified = len(cfg_targets) > 0

    if species_specified:
        _check_duplicates(cfg_species, "cfg.data.species_variables")
        missing = [s for s in cfg_species if s not in processed_set]
        if missing:
            raise ValueError(
                "[hydrate] cfg.data.species_variables contains species not in processed artifacts: "
                f"{missing}. Processed={_preview(processed_species)}"
            )
        logger.info(
            "[hydrate] Using cfg.data.species_variables (%d of %d processed): %s",
            len(cfg_species),
            len(processed_species),
            _preview(cfg_species),
        )
        resolved_species = list(cfg_species)
    else:
        resolved_species = list(processed_species)
        logger.info(
            "[hydrate] cfg.data.species_variables empty; using %d from artifacts: %s",
            len(resolved_species),
            _preview(resolved_species),
        )

    if targets_specified:
        _check_duplicates(cfg_targets, "cfg.data.target_species")
        missing = [s for s in cfg_targets if s not in processed_set]
        if missing:
            raise ValueError(
                "[hydrate] cfg.data.target_species contains species not in processed artifacts: "
                f"{missing}. Processed={_preview(processed_species)}"
            )
        logger.info(
            "[hydrate] Using cfg.data.target_species (%d): %s",
            len(cfg_targets),
            _preview(cfg_targets),
        )
        resolved_targets = list(cfg_targets)
    else:
        resolved_targets = list(resolved_species)
        logger.info(
            "[hydrate] cfg.data.target_species empty; using species_variables (%d species).",
            len(resolved_targets),
        )

    # Enforce invariant: targets must match species exactly
    if resolved_targets != resolved_species:
        raise ValueError(
            "[hydrate] cfg.data.target_species must equal cfg.data.species_variables. "
            f"species={_preview(resolved_species)}; targets={_preview(resolved_targets)}"
        )

    data_cfg["species_variables"] = list(resolved_species)
    data_cfg["target_species"] = list(resolved_targets)

    # Keep global_variables/time_variable consistent with artifacts
    if processed_globals:
        prev_globals = list(data_cfg.get("global_variables", []) or [])
        if not prev_globals:
            logger.info("[hydrate] Using global_variables from artifacts: %s", _preview(processed_globals))
        elif prev_globals != list(processed_globals):
            logger.warning(
                "[hydrate] Overriding global_variables to match artifacts. Config=%s; artifacts=%s",
                _preview(prev_globals),
                _preview(processed_globals),
            )
        data_cfg["global_variables"] = list(processed_globals)

    if processed_time_var:
        prev_time = data_cfg.get("time_variable")
        if not prev_time:
            logger.info("[hydrate] Using time_variable from artifacts: %r", str(processed_time_var))
        elif str(prev_time) != str(processed_time_var):
            logger.warning(
                "[hydrate] Overriding time_variable. Config=%r; artifacts=%r",
                str(prev_time),
                str(processed_time_var),
            )
        data_cfg["time_variable"] = str(processed_time_var)

    return processed_dir


def ensure_preprocessed_data(cfg: Dict[str, Any], logger: logging.Logger) -> Path:
    """Reuse processed data if complete; otherwise run the preprocessor."""
    processed_dir = Path(cfg.get("paths", {}).get("processed_data_dir", "data/processed")).resolve()
    overwrite_data = bool(cfg.get("preprocessing", {}).get("overwrite_data", False))

    req_files = [
        processed_dir / "normalization.json",
        processed_dir / "preprocessing_summary.json",
        processed_dir / "shard_index.json",
    ]
    have_required = all(p.exists() for p in req_files)
    have_splits = all(
        (processed_dir / split).exists() and any((processed_dir / split).glob("*.npz"))
        for split in ("train", "validation", "test")
    )
    preprocessing_needed = overwrite_data or not (have_required and have_splits)

    world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
    if preprocessing_needed and world_size > 1:
        logger.warning("Preprocessing required but WORLD_SIZE>1; exiting")
        raise SystemExit(2)

    if have_required and have_splits and not overwrite_data:
        snap = processed_dir / "config.snapshot.json"
        if snap.exists():
            try:
                snap_cfg = json.loads(snap.read_text(encoding="utf-8"))
                data_snap = snap_cfg.get("data", {})
                data_cfg = cfg.setdefault("data", {})
                for k in ("species_variables", "global_variables", "time_variable", "target_species"):
                    if data_snap.get(k) and not data_cfg.get(k):
                        data_cfg[k] = data_snap[k]
                logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
            except Exception as e:
                logger.warning("[pre] Failed to read config.snapshot.json: %s", e)
        else:
            logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
        return processed_dir

    if overwrite_data and processed_dir.exists():
        logger.warning("Deleting existing processed dir: %s", processed_dir)
        shutil.rmtree(processed_dir, ignore_errors=True)

    dp = DataPreprocessor(cfg, logger=logger.getChild("pre"))
    dp.run()
    return processed_dir


# =============================================================================
# Datasets + DataLoaders
# =============================================================================


def build_datasets_and_loaders(
    cfg: Dict[str, Any],
    device: torch.device,
    runtime_dtype: torch.dtype,
    logger: logging.Logger,
) -> Tuple[FlowMapPairsDataset, FlowMapPairsDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    pairs_per_traj = int(training_cfg.get("pairs_per_traj", 64))
    min_steps = int(training_cfg.get("min_steps", 1))
    max_steps = training_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    base_seed = int(cfg.get("system", {}).get("seed", GLOBAL_SEED))

    preload_to_gpu = bool(dataset_cfg.get("preload_to_gpu", True))

    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="train",
        config=cfg,
        pairs_per_traj=pairs_per_traj,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=preload_to_gpu,
        device=device,
        dtype=runtime_dtype,
        seed=base_seed,
        logger=logger.getChild("dataset.train"),
    )
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="validation",
        config=cfg,
        pairs_per_traj=pairs_per_traj,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=preload_to_gpu,
        device=device,
        dtype=runtime_dtype,
        seed=base_seed + VALIDATION_SEED_OFFSET,
        logger=logger.getChild("dataset.val"),
    )

    batch_size = int(dataset_cfg.get("batch_size_train", 64))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    persistent = bool(dataset_cfg.get("persistent_workers", False))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    prefetch_factor = int(dataset_cfg.get("prefetch_factor", 2))

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    # Log rollout mode status
    rollout_enabled = cfg.get("training", {}).get("rollout", {}).get("enabled", False)
    logger.info(
        f"[dl] B={batch_size} workers={num_workers} prefetch={prefetch_factor} "
        f"pin={pin_memory} persistent={persistent} rollout_mode={rollout_enabled}"
    )
    return train_dataset, val_dataset, train_loader, val_loader


def validate_dataloaders(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    logger: logging.Logger,
) -> None:
    try:
        n_train_items = len(train_loader.dataset)
        n_val_items = len(val_loader.dataset) if val_loader else 0
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
    except Exception:
        n_train_items = n_val_items = n_train_batches = n_val_batches = 0

    logger.info(
        f"Dataset stats: train={n_train_items} items/{n_train_batches} batches; "
        f"val={n_val_items} items/{n_val_batches} batches"
    )
    if n_train_batches == 0:
        raise RuntimeError("Empty training loader")
    if val_loader and n_val_batches == 0:
        raise RuntimeError("Empty validation loader")


# =============================================================================
# Model
# =============================================================================


def build_model(cfg: Dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    logger.info("Creating model...")
    model = create_model(cfg, logger=logger.getChild("model"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")
    return model


# =============================================================================
# Distributed Helpers
# =============================================================================


def get_global_rank() -> int:
    """Get global rank from environment variables."""
    rank_env = (
        os.getenv("RANK")
        or os.getenv("SLURM_PROCID")
        or os.getenv("PMI_RANK")
        or os.getenv("OMPI_COMM_WORLD_RANK")
        or os.getenv("MV2_COMM_WORLD_RANK")
        or "0"
    )
    try:
        return int(rank_env)
    except Exception:
        return 0


def wait_for_marker(marker_path: Path, timeout_secs: float, interval_secs: float) -> bool:
    """Wait for a marker file to appear."""
    iterations = int(timeout_secs / interval_secs)
    for _ in range(iterations):
        if marker_path.exists():
            return True
        time.sleep(interval_secs)
    return False


# =============================================================================
# Training Entrypoint
# =============================================================================


def main() -> None:
    """Main training entrypoint."""
    cfg_path_str = os.getenv("FLOWMAP_CONFIG", str(DEFAULT_CONFIG_PATH))
    cfg_path = Path(cfg_path_str).expanduser().resolve()
    cfg = load_json_config(cfg_path)

    if "paths" not in cfg or "processed_data_dir" not in cfg["paths"]:
        raise KeyError("cfg.paths.processed_data_dir is required")

    cfg.setdefault("system", {})
    cfg["system"].setdefault("seed", GLOBAL_SEED)
    cfg.setdefault("dataset", {})
    cfg.setdefault("mixed_precision", {})
    cfg["mixed_precision"].setdefault("mode", "bf16")

    normalize_distributed_env()

    work_dir = Path(cfg.get("paths", {}).get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()
    overwrite = bool(cfg.get("paths", {}).get("overwrite", False))
    global_rank = get_global_rank()

    if global_rank == 0:
        if work_dir.exists() and overwrite:
            logging.warning(f"Deleting existing work dir: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            (work_dir / ".ready").write_text("ok", encoding="utf-8")
        except Exception:
            pass
    else:
        wait_for_marker(work_dir / ".ready", timeout_secs=DDP_WAIT_TIMEOUT_SECS, interval_secs=DDP_WAIT_INTERVAL_SECS)
        work_dir.mkdir(parents=True, exist_ok=True)

    if global_rank == 0:
        setup_logging(log_file=work_dir / "train.log", level=logging.INFO)
    else:
        setup_logging(level=logging.INFO)
    logger = logging.getLogger("main")
    logger.info(f"Work directory: {work_dir}")

    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device(logger)
    optimize_hardware(cfg.get("system", {}), device, logger)

    pl_precision, runtime_dtype = resolve_precision_and_dtype(cfg, device, logger)
    logger.info(f"Resolved precision={pl_precision}; runtime_dtype={runtime_dtype}")

    if global_rank == 0:
        processed_dir = ensure_preprocessed_data(cfg, logger)
        try:
            (work_dir / ".data.ready").write_text("ok", encoding="utf-8")
        except Exception:
            pass
    else:
        processed_dir = Path(cfg.get("paths", {}).get("processed_data_dir", "data/processed")).resolve()
        req_files = [
            processed_dir / "normalization.json",
            processed_dir / "preprocessing_summary.json",
            processed_dir / "shard_index.json",
        ]
        have_required = all(p.exists() for p in req_files)
        have_splits = all(
            (processed_dir / split).exists() and any((processed_dir / split).glob("*.npz"))
            for split in ("train", "validation", "test")
        )
        overwrite_data = bool(cfg.get("preprocessing", {}).get("overwrite_data", False))
        preprocessing_needed = overwrite_data or not (have_required and have_splits)

        world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
        if preprocessing_needed and world_size > 1:
            logger.warning("Preprocessing required but WORLD_SIZE>1")
            raise SystemExit(2)

        wait_for_marker(work_dir / ".data.ready", timeout_secs=360.0, interval_secs=DDP_WAIT_INTERVAL_SECS)
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    assert processed_dir.exists(), "Processed data dir must exist"

    hydrate_config_from_processed(cfg, logger, processed_dir)

    if global_rank == 0:
        dump_json(work_dir / "config.json", cfg)
        logger.info("Saved hydrated config")

    train_ds, val_ds, train_loader, val_loader = build_datasets_and_loaders(
        cfg=cfg, device=device, runtime_dtype=runtime_dtype, logger=logger
    )
    validate_dataloaders(train_loader=train_loader, val_loader=val_loader, logger=logger)

    model = build_model(cfg, logger)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=logger.getChild("trainer"),
        pl_precision_override=pl_precision,
    )
    best_val_loss = trainer.train()
    if global_rank == 0:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6e}")
    else:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
