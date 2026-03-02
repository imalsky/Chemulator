#!/usr/bin/env python3
"""
main.py
- JSON/JSONC config loading
- Hardware + reproducibility setup (inlined; hardware.py not required)
- Preprocess if needed, then hydrate cfg.data.* from artifacts
- Build datasets/dataloaders (single set of knobs; no *_val variants)
- Build model
- Train via a small, explicit PyTorch training loop (JSONL metrics)
"""

from __future__ import annotations

import os
import sys

# Resolve duplicate OpenMP on macOS (avoid setting this on Linux)
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
import logging
import inspect
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence


import torch

from utils import (
    setup_logging,
    seed_everything,
    load_json_config,
    dump_json,
    resolve_precision_policy,
)
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer
from preprocessor import DataPreprocessor


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config_job0.jsonc"

# Number of hash bytes used to derive a 32-bit seed.
_SEED_BYTES = 4

# Max items shown when previewing lists in error messages.
_PREVIEW_MAX_ITEMS = 10


def _resolve_repo_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve config paths relative to repository root when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def reject_deprecated_config_keys(cfg: Dict[str, Any]) -> None:
    """Fail fast on config keys that are not implemented in this codebase.

    The goal is to avoid silent no-ops where a user expects a feature to be active.
    """

    deprecated: list[tuple[str, str]] = []

    ds = cfg.get("dataset", {})
    for k in (
        "storage_dtype",
        "skip_scan",
        "skip_validate_grids",
        "precompute_dt_table",
        "share_times_across_batch",
        "mmap_mode",
        "uniform_offset_sampling_strict",
        "assume_shared_grid",
        "dt_epsilon",
    ):
        if k in ds:
            deprecated.append(("dataset", k))

    data_cfg = cfg.get("data", {})
    if "target_species" in data_cfg:
        deprecated.append(("data", "target_species"))

    pre_cfg = cfg.get("preprocessing", {})
    for k in ("num_workers", "allow_empty_splits"):
        if k in pre_cfg:
            deprecated.append(("preprocessing", k))

    tr = cfg.get("training", {})
    for k in (
        "use_swa",
        "swa_epoch_start",
        "swa_annealing_epochs",
        "swa_annealing_strategy",
        "accumulate_grad_batches",
        "auto_resume",
    ):
        if k in tr:
            deprecated.append(("training", k))

    loss_cfg = tr.get("adaptive_stiff_loss", {})
    if "epsilon_phys" in loss_cfg:
        deprecated.append(("training.adaptive_stiff_loss", "epsilon_phys"))

    mcfg = cfg.get("model", {})
    if "vae_mode" in mcfg:
        deprecated.append(("model", "vae_mode"))
    if "allow_partial_simplex" in mcfg:
        deprecated.append(("model", "allow_partial_simplex"))

    if deprecated:
        keys = ", ".join(f"{a}.{b}" for a, b in deprecated)
        raise KeyError(f"Unsupported config key(s): {keys}")


def _derive_seed(base_seed: int, tag: str) -> int:
    """Derive a deterministic, decorrelated seed from a base seed and a tag."""

    h = hashlib.sha256(f"{int(base_seed)}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:_SEED_BYTES], byteorder="little", signed=False)


# --------------------------------------------------------------------------------------
# Inlined "hardware.py"
# --------------------------------------------------------------------------------------

def setup_device(logger: logging.Logger) -> torch.device:
    """
    Single-device selection:
      - CUDA: always uses cuda:0.
      - MPS: uses Apple MPS if available.
      - CPU: fallback.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning("Could not query CUDA device name: %s", e)
            dev_name = "unknown"
        logger.info(f"Set CUDA device to cuda:0 ({dev_name})")
        return device

    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return torch.device("mps")

    logger.info("Using CPU")
    return torch.device("cpu")


def optimize_hardware(cfg: Dict[str, Any], device: torch.device, logger: logging.Logger) -> None:
    """
    Mirrors the intent of hardware.optimize_hardware:
      - TF32 / matmul precision
      - cudnn benchmark vs deterministic
      - deterministic algorithms (optional)
      - optional CPU thread settings
    """
    # Matmul precision is a best-effort global hint across backends.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as e:
        logger.warning("Failed to set float32 matmul precision hint: %s", e)

    system_cfg = cfg.get("system", {})
    precision_cfg = cfg.get("precision", {})

    # TF32 flags (CUDA only)
    tf32 = bool(precision_cfg.get("tf32", False))

    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        except Exception as e:
            raise RuntimeError("Failed to apply precision.tf32 to torch.backends.cuda.matmul.allow_tf32") from e
        try:
            torch.backends.cudnn.allow_tf32 = tf32
        except Exception as e:
            raise RuntimeError("Failed to apply precision.tf32 to torch.backends.cudnn.allow_tf32") from e

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
            raise RuntimeError(
                "system.deterministic=true but torch.use_deterministic_algorithms(True) failed"
            ) from e
        try:
            torch.backends.cudnn.deterministic = True
        except Exception as e:
            raise RuntimeError("system.deterministic=true but failed to set cudnn.deterministic=True") from e
        try:
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            raise RuntimeError("system.deterministic=true but failed to set cudnn.benchmark=False") from e
        logger.info("Deterministic mode enabled (cudnn.benchmark forced False).")
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception as e:
            logger.warning("Could not disable deterministic algorithms explicitly: %s", e)
        try:
            torch.backends.cudnn.deterministic = False
        except Exception as e:
            logger.warning("Could not set cudnn.deterministic=False: %s", e)
        try:
            torch.backends.cudnn.benchmark = cudnn_benchmark
        except Exception as e:
            logger.warning("Could not set cudnn.benchmark=%s: %s", cudnn_benchmark, e)
        logger.info(f"cudnn.benchmark={cudnn_benchmark}")

    # Final effective settings summary (matches old behavior)
    try:
        eff_bench = bool(torch.backends.cudnn.benchmark)
    except Exception as e:
        logger.warning("Could not read cudnn.benchmark effective state: %s", e)
        eff_bench = cudnn_benchmark
    try:
        eff_det = bool(torch.backends.cudnn.deterministic)
    except Exception as e:
        logger.warning("Could not read cudnn.deterministic effective state: %s", e)
        eff_det = False

    logger.info(
        f"Hardware settings: TF32={tf32}, cudnn.benchmark={eff_bench}, "
        f"deterministic={deterministic}, cudnn.deterministic={eff_det}"
    )


# --------------------------------------------------------------------------------------
# Config hydration from processed artifacts
# --------------------------------------------------------------------------------------

def _preview(items: Sequence[str]) -> str:
    """Format a list for error messages, truncating after _PREVIEW_MAX_ITEMS."""
    items = list(items)
    if len(items) <= _PREVIEW_MAX_ITEMS:
        return "[" + ", ".join(repr(x) for x in items) + "]"
    head = ", ".join(repr(x) for x in items[:_PREVIEW_MAX_ITEMS])
    return f"[{head}, ...] (+{len(items) - _PREVIEW_MAX_ITEMS} more)"


def hydrate_config_from_processed(
    cfg: Dict[str, Any],
    logger: logging.Logger,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Hydrate cfg.data from processed artifacts (normalization.json / preprocessing_summary.json).

    This codebase requires the configured schema to match the processed artifacts exactly.
    If the config omits schema fields, they are filled from artifacts. If the config specifies
    them, they must be identical (including order), otherwise an error is raised.
    """
    if processed_dir is None:
        processed_dir = _resolve_repo_path(cfg["paths"]["processed_data_dir"])
    else:
        processed_dir = _resolve_repo_path(processed_dir)

    if not processed_dir.exists():
        raise FileNotFoundError(f"[hydrate] Missing processed data dir: {processed_dir}")

    norm_path = processed_dir / "normalization.json"
    summary_path = processed_dir / "preprocessing_summary.json"

    if not norm_path.exists():
        raise FileNotFoundError(f"[hydrate] Missing normalization.json in {processed_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"[hydrate] Missing preprocessing_summary.json in {processed_dir}")

    with open(norm_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    meta = manifest.get("meta", {})
    processed_species = meta.get("species_variables") or summary.get("species_variables")
    processed_globals = meta.get("global_variables") or summary.get("global_variables")
    processed_time_var = meta.get("time_variable") or summary.get("time_variable")

    if not processed_species:
        raise RuntimeError(
            "[hydrate] Unable to determine processed species_variables from normalization.json or preprocessing_summary.json "
            f"in {processed_dir}"
        )
    processed_species = list(processed_species)

    processed_targets = meta.get("target_species")
    if processed_targets is not None and list(processed_targets) != processed_species:
        raise RuntimeError(
            "[hydrate] Processed artifacts contain inconsistent species/targets. "
            f"species_variables={_preview(processed_species)}; target_species={_preview(list(processed_targets))}. "
            "This codebase requires outputs to match inputs."
        )

    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise KeyError("[hydrate] cfg.data must be a mapping")
    if "target_species" in data_cfg:
        raise KeyError("[hydrate] Unsupported config key: data.target_species")

    cfg_species_raw = data_cfg.get("species_variables")
    if not isinstance(cfg_species_raw, list) or len(cfg_species_raw) == 0:
        raise ValueError("[hydrate] cfg.data.species_variables must be explicitly set and non-empty")
    cfg_species = list(cfg_species_raw)
    if cfg_species != processed_species:
        raise ValueError(
            "[hydrate] cfg.data.species_variables does not match processed artifacts. "
            f"config={_preview(cfg_species)}; processed={_preview(processed_species)}"
        )

    if processed_globals is None:
        raise RuntimeError("[hydrate] Missing global_variables in processed artifacts")
    processed_globals_list = list(processed_globals)
    cfg_globals = list(data_cfg.get("global_variables") or [])
    if cfg_globals:
        if cfg_globals != processed_globals_list:
            raise ValueError(
                "[hydrate] cfg.data.global_variables does not match processed artifacts. "
                f"config={_preview(cfg_globals)}; processed={_preview(processed_globals_list)}"
            )
    else:
        data_cfg["global_variables"] = processed_globals_list
        logger.info("[hydrate] cfg.data.global_variables is empty; using processed artifacts")

    if processed_time_var is None:
        raise RuntimeError("[hydrate] Missing time_variable in processed artifacts")
    cfg_time = data_cfg.get("time_variable")
    if cfg_time:
        if str(cfg_time) != str(processed_time_var):
            raise ValueError(
                "[hydrate] cfg.data.time_variable does not match processed artifacts. "
                f"config={str(cfg_time)!r}; processed={str(processed_time_var)!r}"
            )
    else:
        data_cfg["time_variable"] = str(processed_time_var)
        logger.info("[hydrate] cfg.data.time_variable is empty; using processed artifacts")

    return processed_dir


def ensure_preprocessed_data(cfg: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    Reuse processed data if complete; otherwise run the preprocessor.
    """
    processed_dir = _resolve_repo_path(cfg["paths"]["processed_data_dir"])
    overwrite_data = bool(cfg["preprocessing"]["overwrite_data"])
    reuse_existing_data = bool(cfg["preprocessing"]["reuse_existing_data"])

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
    if have_required and have_splits and not overwrite_data:
        if not reuse_existing_data:
            raise FileExistsError(
                "Preprocessed data already exists. "
                "Set preprocessing.reuse_existing_data=true to reuse it, "
                "or set preprocessing.overwrite_data=true to regenerate."
            )
        logger.info("[pre] Reusing existing preprocessed data at %s", processed_dir)
        return processed_dir

    if processed_dir.exists():
        nonempty = any(processed_dir.iterdir())
        if nonempty and not overwrite_data:
            raise FileExistsError(
                "Processed data directory exists but is incomplete (or overwrite is disabled). "
                "Set preprocessing.overwrite_data=true to regenerate, or point paths.processed_data_dir to a "
                "different (empty) directory."
            )
        if overwrite_data and nonempty:
            logger.warning("Deleting existing processed dir: %s", processed_dir)
            shutil.rmtree(processed_dir)

    dp = DataPreprocessor(cfg, logger=logger.getChild("pre"))
    dp.run()

    # Verify outputs exist after preprocessing.
    if not all(p.exists() for p in req_files):
        missing = [str(p) for p in req_files if not p.exists()]
        raise RuntimeError(f"Preprocessing completed but required artifact(s) are missing: {missing}")
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
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    processed_dir = _resolve_repo_path(cfg["paths"]["processed_data_dir"])

    for key in ("pairs_per_traj", "min_steps", "max_steps"):
        if key not in training_cfg:
            raise KeyError(f"Missing config: training.{key}")

    pairs_per_traj = int(training_cfg["pairs_per_traj"])
    min_steps = int(training_cfg["min_steps"])
    max_steps = training_cfg["max_steps"]
    max_steps = int(max_steps) if (max_steps is not None) else None
    base_seed = int(cfg["system"]["seed"])

    if "preload_to_gpu" not in dataset_cfg:
        raise KeyError("Missing config: dataset.preload_to_gpu")
    preload_to_gpu = bool(dataset_cfg["preload_to_gpu"])

    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="train", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=base_seed,
        logger=logger.getChild("dataset.train"),
    )
    val_seed = _derive_seed(base_seed, "validation")
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir, split="validation", config=cfg,
        pairs_per_traj=pairs_per_traj, min_steps=min_steps, max_steps=max_steps,
        preload_to_gpu=preload_to_gpu, device=device, dtype=runtime_dtype, seed=val_seed,
        logger=logger.getChild("dataset.val"),
    )

    for key in (
        "batch_size_train",
        "num_workers",
        "persistent_workers",
        "pin_memory",
        "prefetch_factor",
    ):
        if key not in dataset_cfg:
            raise KeyError(f"Missing config: dataset.{key}")

    batch_size = int(dataset_cfg["batch_size_train"])
    num_workers = int(dataset_cfg["num_workers"])
    persistent = bool(dataset_cfg["persistent_workers"])
    pin_memory = bool(dataset_cfg["pin_memory"])
    prefetch_factor = int(dataset_cfg["prefetch_factor"])

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
    )
    return train_dataset, val_dataset, train_loader, val_loader


def validate_dataloaders(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    logger: logging.Logger,
) -> None:
    n_train_items = len(train_loader.dataset)
    n_val_items = len(val_loader.dataset) if val_loader else 0
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader) if val_loader else 0

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

    tcfg = cfg["training"]
    if "torch_compile" not in tcfg:
        raise KeyError("Missing config: training.torch_compile")
    if bool(tcfg["torch_compile"]):
        if not hasattr(torch, "compile"):
            raise ValueError("torch.compile not available")

        for key in (
            "torch_compile_backend",
            "torch_compile_mode",
            "compile_dynamic",
            "compile_fullgraph",
        ):
            if key not in tcfg:
                raise KeyError(f"Missing config: training.{key}")

        backend = str(tcfg["torch_compile_backend"])
        mode = str(tcfg["torch_compile_mode"])
        dynamic = bool(tcfg["compile_dynamic"])
        fullgraph = bool(tcfg["compile_fullgraph"])

        compile_kwargs: dict[str, Any] = {"backend": backend, "mode": mode}
        sig = inspect.signature(torch.compile)
        if "dynamic" in sig.parameters:
            compile_kwargs["dynamic"] = dynamic
        if "fullgraph" in sig.parameters:
            compile_kwargs["fullgraph"] = fullgraph

        model = torch.compile(model, **compile_kwargs)
        logger.info("Enabled torch.compile")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")
    return model


# --------------------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------------------

def _require_export_script() -> Path:
    """Return the export script path or fail fast if it's missing."""
    export_script = REPO_ROOT / "testing" / "export.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"Missing export script: {export_script}")
    return export_script


def export_physical_artifacts(work_dir: Path, logger: logging.Logger) -> Tuple[Path, Path]:
    """Generate physical-I/O exported model and companion metadata in work_dir."""
    export_script = _require_export_script()

    env = os.environ.copy()
    env["CHEMULATOR_MODEL_DIR"] = str(work_dir)

    logger.info("Exporting physical-I/O artifact via %s", export_script)
    proc = subprocess.run(
        [sys.executable, str(export_script)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            logger.info("[export] %s", line)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(
            f"Physical artifact export failed (exit={proc.returncode}). "
            f"stdout={stdout!r}; stderr={stderr!r}"
        )

    model_path = work_dir / "physical_model_k1_cpu.pt2"
    metadata_path = work_dir / "physical_model_metadata.json"
    missing = [str(p) for p in (model_path, metadata_path) if not p.is_file()]
    if missing:
        raise RuntimeError(f"Physical artifact export completed but files are missing: {missing}")

    logger.info("Export complete: model=%s metadata=%s", model_path, metadata_path)
    return model_path, metadata_path


def main() -> None:
    """Main training entrypoint (single-process only)."""
    cfg_path_str = os.getenv("FLOWMAP_CONFIG", str(DEFAULT_CONFIG_PATH))
    cfg_path = Path(cfg_path_str).expanduser().resolve()
    cfg = load_json_config(cfg_path)
    reject_deprecated_config_keys(cfg)

    # Validate required top-level sections and their keys.
    _required_sections: dict[str, tuple[str, ...]] = {
        "paths": ("processed_data_dir", "raw_data_files", "work_dir", "overwrite"),
        "system": ("seed",),
        "precision": ("amp", "dataset_dtype", "io_dtype", "time_io_dtype", "normalize_dtype", "tf32"),
        "preprocessing": ("overwrite_data", "reuse_existing_data"),
        "dataset": (),
        "data": (),
        "normalization": (),
        "training": (),
    }
    for section, keys in _required_sections.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise KeyError(f"Missing config: {section}")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"Missing config: {section}.{key}")

    raw_files = cfg["paths"]["raw_data_files"]
    if not isinstance(raw_files, list) or len(raw_files) == 0:
        raise ValueError("paths.raw_data_files must be explicitly set and non-empty")

    # Fail early if the required physical-artifact exporter is unavailable.
    _require_export_script()

    work_dir = _resolve_repo_path(cfg["paths"]["work_dir"])
    overwrite = bool(cfg["paths"]["overwrite"])
    if work_dir.exists() and overwrite:
        print(f"[main] Deleting existing work dir: {work_dir}", flush=True)
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=work_dir / "train.log", level=logging.INFO)
    logger = logging.getLogger("main")
    logger.info(f"Work directory: {work_dir}")

    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device(logger)
    optimize_hardware(cfg, device, logger)

    policy = resolve_precision_policy(cfg, device)
    runtime_dtype = policy.dataset_dtype
    logger.info(
        "Precision: amp=%s amp_dtype=%s dataset_dtype=%s io_dtype=%s",
        str(policy.amp_mode),
        str(policy.amp_dtype).replace("torch.", ""),
        str(policy.dataset_dtype).replace("torch.", ""),
        str(cfg["precision"]["io_dtype"]),
    )

    processed_dir = ensure_preprocessed_data(cfg, logger)

    if not processed_dir.exists():
        raise FileNotFoundError("Processed data dir must exist after preprocessing/hydration")

    hydrate_config_from_processed(cfg, logger, processed_dir)

    dump_json(work_dir / "config.json", cfg)
    logger.info("Saved final hydrated config")

    _, _, train_loader, val_loader = build_datasets_and_loaders(
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
        precision_policy=policy,
    )
    best_val_loss = trainer.train()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.6e}")
    export_physical_artifacts(work_dir=work_dir, logger=logger)


if __name__ == "__main__":
    main()
