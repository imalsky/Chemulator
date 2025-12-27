#!/usr/bin/env python3
"""
main.py
- JSON/JSONC config loading
- Hardware + reproducibility setup (inlined; hardware.py not required)
- Preprocess if needed, then hydrate cfg.data.* from artifacts
- Build datasets/dataloaders (single set of knobs; no *_val variants)
- Build model
- Train via Lightning-backed Trainer (CSV only)
"""

from __future__ import annotations

import os
import sys

# Resolve duplicate OpenMP on macOS (avoid setting this on Linux)
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from utils import (
    setup_logging,
    seed_everything,
    load_json_config,
    dump_json,
    resolve_precision_and_dtype,
)
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer
from preprocessor import DataPreprocessor


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
GLOBAL_SEED = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "autoencoder-flowmap"
VALIDATION_SEED_OFFSET = 1337


# --------------------------------------------------------------------------------------
# DDP / MPI env normalization (optional but helpful on clusters)
# --------------------------------------------------------------------------------------

def normalize_distributed_env() -> None:
    """
    Lightning/DDP expects torchrun-style env vars (RANK/WORLD_SIZE/LOCAL_RANK).
    Map common MPI/PMI/MVAPICH/SLURM env vars to torchrun equivalents *only when missing*.
    """
    if os.getenv("RANK") is None:
        for k in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
            v = os.getenv(k)
            if v is not None:
                os.environ["RANK"] = v
                break

    if os.getenv("WORLD_SIZE") is None:
        for k in ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
            v = os.getenv(k)
            if v is not None:
                os.environ["WORLD_SIZE"] = v
                break

    if os.getenv("LOCAL_RANK") is None:
        for k in (
            "SLURM_LOCALID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "CUDA_LOCAL_RANK",
        ):
            v = os.getenv(k)
            if v is not None:
                os.environ["LOCAL_RANK"] = v
                break


# --------------------------------------------------------------------------------------
# Inlined "hardware.py"
# --------------------------------------------------------------------------------------

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
    Mirrors the intent of hardware.optimize_hardware:
      - TF32 / matmul precision
      - cudnn benchmark vs deterministic
      - deterministic algorithms (optional)
      - optional CPU thread settings
    """
    # Matmul precision (safe no-op if unsupported)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # TF32 flags (CUDA only) — support both legacy key "tf32" and newer "allow_tf32"
    tf32 = system_cfg.get("tf32", system_cfg.get("allow_tf32", True))
    tf32 = bool(tf32)

    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = tf32
        except Exception:
            pass

    # CPU thread control (optional): set both env + torch threads
    omp = system_cfg.get("omp_num_threads")
    if omp is not None:
        try:
            omp_int = int(omp)
            if omp_int > 0:
                os.environ["OMP_NUM_THREADS"] = str(omp_int)
                torch.set_num_threads(omp_int)
                logger.info(f"Set OMP_NUM_THREADS={omp_int}")
                logger.info(f"Set torch num_threads={omp_int}")
        except Exception as e:
            logger.warning(f"Failed to set OMP_NUM_THREADS / torch num_threads: {e}")

    deterministic = bool(system_cfg.get("deterministic", False))
    cudnn_benchmark = bool(system_cfg.get("cudnn_benchmark", True)) and not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logger.warning(f"torch.use_deterministic_algorithms(True) failed: {e}")
        try:
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        logger.info("Deterministic mode enabled (cudnn.benchmark forced False).")
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = cudnn_benchmark
        except Exception:
            pass
        logger.info(f"cudnn.benchmark={cudnn_benchmark}")

    # Final effective settings summary (matches old behavior)
    try:
        eff_bench = bool(torch.backends.cudnn.benchmark)
    except Exception:
        eff_bench = cudnn_benchmark
    try:
        eff_det = bool(torch.backends.cudnn.deterministic)
    except Exception:
        eff_det = False

    logger.info(
        f"Hardware settings: TF32={tf32}, cudnn.benchmark={eff_bench}, "
        f"deterministic={deterministic}, cudnn.deterministic={eff_det}"
    )


# --------------------------------------------------------------------------------------
# Config hydration from processed artifacts
# --------------------------------------------------------------------------------------

def hydrate_config_from_processed(
    cfg: Dict[str, Any],
    logger: logging.Logger,
    processed_dir: Optional[Path] = None,
) -> Path:
    if processed_dir is None:
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"[hydrate] Missing processed data dir: {processed_dir}")

    norm_path = processed_dir / "normalization.json"
    summary_path = processed_dir / "preprocessing_summary.json"

    def _load_json(p: Path) -> Optional[Dict[str, Any]]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[hydrate] Failed to read {p.name}: {e}")
        return None

    manifest = _load_json(norm_path)
    summary = _load_json(summary_path)

    meta = (manifest or {}).get("meta", {}) if manifest else {}
    species = meta.get("species_variables") or (summary or {}).get("species_variables")
    globals_ = meta.get("global_variables") or (summary or {}).get("global_variables")
    time_var = meta.get("time_variable") or (summary or {}).get("time_variable")
    if not species:
        raise RuntimeError("[hydrate] Unable to determine species_variables")

    targets = (meta.get("target_species") if meta else None) or (manifest or {}).get("target_species")
    if targets is None:
        targets = (summary or {}).get("target_species")
    if targets is None:
        targets = species

    data_cfg = cfg.setdefault("data", {})
    prev_species = list(data_cfg.get("species_variables", []))
    if species:
        if prev_species and prev_species != list(species):
            logger.warning("[hydrate] Overriding cfg.data.species_variables from processed artifacts")
        data_cfg["species_variables"] = list(species)

    prev_globals = list(data_cfg.get("global_variables", []))
    if globals_:
        if prev_globals and prev_globals != list(globals_):
            logger.warning("[hydrate] Overriding cfg.data.global_variables from processed artifacts")
        data_cfg["global_variables"] = list(globals_)

    prevt = data_cfg.get("time_variable")
    if time_var:
        if prevt and prevt != time_var:
            logger.warning("[hydrate] Overriding cfg.data.time_variable from processed artifacts")
        data_cfg["time_variable"] = str(time_var)

    prev_targets = list(data_cfg.get("target_species", []))
    if targets:
        if prev_targets and prev_targets != list(targets):
            logger.warning("[hydrate] Overriding cfg.data.target_species from processed artifacts")
        data_cfg["target_species"] = list(targets)
    else:
        if prev_targets:
            missing = set(prev_targets) - set(species)
            if missing:
                logger.warning("[hydrate] cfg.data.target_species not subset of species")

    return processed_dir


def ensure_preprocessed_data(cfg: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    Reuse processed data if complete; otherwise run the preprocessor.

    Safety: if preprocessing is required and WORLD_SIZE>1, exit.
    """
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

    # Multi means multi-rank, not "node has multiple GPUs"
    world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
    multi = world_size > 1

    if preprocessing_needed and multi:
        logger.warning("Preprocessing is required but WORLD_SIZE>1 (multi-rank) is active")
        raise SystemExit(2)

    # Fast path: reuse existing artifacts (hydrate cfg.data non-destructively if snapshot exists)
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
                logger.warning("[pre] Reusing existing data but failed to read config.snapshot.json: %s", e)
        else:
            logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
        return processed_dir

    # Single-rank preprocessing
    if overwrite_data and processed_dir.exists():
        logger.warning("Deleting existing processed dir: %s", processed_dir)
        shutil.rmtree(processed_dir, ignore_errors=True)

    dp = DataPreprocessor(cfg, logger=logger.getChild("pre"))
    dp.run()
    return processed_dir


# --------------------------------------------------------------------------------------
# Datasets + DataLoaders (single set of knobs)
# --------------------------------------------------------------------------------------

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
    max_steps = int(max_steps) if (max_steps is not None) else None
    base_seed = int(cfg.get("system", {}).get("seed", 42))

    preload_to_gpu = bool(dataset_cfg.get("preload_to_gpu", True))

    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="train", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=base_seed,
    )
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="validation", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=base_seed + VALIDATION_SEED_OFFSET,
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

    logger.info(
        f"[dl] B={batch_size} workers={num_workers} prefetch={prefetch_factor} "
        f"pin={pin_memory} persistent_workers={persistent} "
        f"share_times_across_batch={bool(dataset_cfg.get('share_times_across_batch', False))}"
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


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    logger.info("Creating model...")
    model = create_model(cfg, logger=logger.getChild("model"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")
    return model


# --------------------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------------------

def main() -> None:
    """Main training entrypoint (rank-0 writes; N GPU safe)."""
    # ------------------------------- Load config -------------------------------
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

    # Normalize env vars so Lightning can DDP under MPI/torchrun/srun
    normalize_distributed_env()

    # ----------------------- Rank/env + work_dir prepare -----------------------
    work_dir = Path(cfg.get("paths", {}).get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()
    overwrite = bool(cfg.get("paths", {}).get("overwrite", False))

    # Robust rank detection across MPI stacks + SLURM
    rank_env = (
        os.getenv("RANK")
        or os.getenv("SLURM_PROCID")
        or os.getenv("PMI_RANK")
        or os.getenv("OMPI_COMM_WORLD_RANK")
        or os.getenv("MV2_COMM_WORLD_RANK")
        or "0"
    )
    try:
        global_rank = int(rank_env)
    except Exception:
        global_rank = 0

    if global_rank == 0:
        if work_dir.exists() and overwrite:
            print(f"[main] Deleting existing work dir: {work_dir}", flush=True)
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            (work_dir / ".ready").write_text("ok", encoding="utf-8")
        except Exception:
            pass
    else:
        # Wait up to ~60s for rank-0 to prep the dir
        for _ in range(600):
            if work_dir.exists() and (work_dir / ".ready").exists():
                break
            time.sleep(0.1)
        work_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Logging setup ------------------------------
    if global_rank == 0:
        setup_logging(log_file=work_dir / "train.log", level=logging.INFO)
    else:
        setup_logging(level=logging.INFO)
    logger = logging.getLogger("main")
    logger.info(f"Work directory: {work_dir}")

    # -------------------------- Repro + device setup ---------------------------
    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device(logger)
    optimize_hardware(cfg.get("system", {}), device, logger)

    # -------------------- Precision + runtime dtype resolve --------------------
    pl_precision, runtime_dtype = resolve_precision_and_dtype(cfg, device, logger)
    logger.info(f"Resolved Lightning precision={pl_precision}; dataset runtime dtype={runtime_dtype}")

    # ------------------- Preprocessing (rank-0 only writes) -------------------
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
        multi = world_size > 1

        if preprocessing_needed and multi:
            logger.warning("Preprocessing is required but WORLD_SIZE>1 (multi-rank) is active")
            raise SystemExit(2)

        # Otherwise safe to wait for rank-0 to finish
        for _ in range(3600):
            if (work_dir / ".data.ready").exists():
                break
            time.sleep(0.1)
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    assert processed_dir.exists(), "Processed data dir must exist after preprocessing/hydration"

    hydrate_config_from_processed(cfg, logger, processed_dir)

    if global_rank == 0:
        dump_json(work_dir / "config.json", cfg)
        logger.info("Saved final hydrated config")
    else:
        logger.info("Config hydrated (rank>0; no file write)")

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
        pl_precision_override=pl_precision,  # keep Trainer + main consistent
    )
    best_val_loss = trainer.train()
    if global_rank == 0:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6e}")
    else:
        logger.info("Training complete (rank>0).")


if __name__ == "__main__":
    main()
