#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, Sequence

# Resolve duplicate library issue on macOS (harmless elsewhere)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from utils import setup_logging, seed_everything, load_json_config, dump_json
from hardware import setup_device, optimize_hardware
from preprocessor import DataPreprocessor
from dataset import FlowMapPairsDataset, create_dataloader
from model import create_model
from trainer import Trainer  # Lightning-backed wrapper


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.jsonc"
GLOBAL_SEED = 42
GLOBAL_WORK_DIR = REPO_ROOT / "models" / "autoencoder-flowmap"


def _parse_int_or_auto(val: Any, default_if_auto: int) -> int:
    """
    Return int(val) unless val is 'auto'/'none'/''/None -> default.
    """
    if val is None:
        return int(default_if_auto)
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    if s in ("auto", "none", "null", ""):
        return int(default_if_auto)
    return int(s)

def _auto_num_workers(default_if_auto: int = 0) -> int:
    """
    Reasonable default for DataLoader workers. If you are preloading to GPU, keep low.
    """
    try:
        cpu = os.cpu_count() or default_if_auto
    except Exception:
        cpu = default_if_auto
    # Favor low workers when tensors are GPU-resident already
    return max(0, min(8, cpu // 8))


def hydrate_config_from_processed(
        cfg: Dict[str, Any],
        logger: logging.Logger,
        processed_dir: Optional[Path] = None,
) -> Path:
    """
    Re-hydrate cfg.data.* from an existing processed data directory.

    Priority:
      species/globals/time:
        1) normalization.json -> meta.{species_variables, global_variables, time_variable}
        2) preprocessing_summary.json -> {species_variables, global_variables, time_variable}
      target_species:
        1) normalization.json -> meta.target_species (or top-level target_species)
        2) preprocessing_summary.json -> target_species
        3) fall back to cfg.data.target_species if valid subset
        4) else default to species_variables

    Split stats:
      - Prefer sample counts from preprocessing_summary.json (valid_trajectories).
      - Else shard_index.json (splits.*.n_trajectories).
      - Else hunt alternate keys; else shard counts.
    """
    # Resolve processed_dir
    if processed_dir is None:
        processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    else:
        processed_dir = Path(processed_dir).expanduser().resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(f"[hydrate] Processed data directory not found: {processed_dir}")

    norm_path = processed_dir / "normalization.json"
    summary_path = processed_dir / "preprocessing_summary.json"
    index_path = processed_dir / "shard_index.json"
    report_path = processed_dir / "preprocess_report.json"

    logger.info(f"[hydrate] Using processed data at: {processed_dir}")

    # Load artifacts (tolerant)
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
    index = _load_json(index_path)
    report = _load_json(report_path)

    # Extract core metadata
    meta = (manifest or {}).get("meta", {}) if manifest else {}
    species_from_processed = meta.get("species_variables")
    globals_from_processed = meta.get("global_variables")
    time_from_processed = meta.get("time_variable")

    if species_from_processed is None and summary is not None:
        species_from_processed = summary.get("species_variables")
    if globals_from_processed is None and summary is not None:
        globals_from_processed = summary.get("global_variables")
    if time_from_processed is None and summary is not None:
        time_from_processed = summary.get("time_variable")

    if not species_from_processed:
        raise RuntimeError("[hydrate] Unable to determine species_variables")

    # Try to discover target_species explicitly from artifacts
    targets_from_processed = None
    if manifest:
        targets_from_processed = meta.get("target_species") or manifest.get("target_species")
    if targets_from_processed is None and summary is not None:
        targets_from_processed = summary.get("target_species")

    # Validate target list is subset of species
    if targets_from_processed:
        missing = set(targets_from_processed) - set(species_from_processed)
        if missing:
            logger.warning(
                "[hydrate] target_species from artifacts not a subset of species_variables; "
                f"ignoring artifact targets. Missing={sorted(missing)}"
            )
            targets_from_processed = None

    # Inject into cfg.data with clear precedence
    data_cfg = cfg.setdefault("data", {})

    prev_species = list(data_cfg.get("species_variables", []))
    if prev_species and prev_species != species_from_processed:
        logger.warning("[hydrate] Overriding config.data.species_variables with values from processed data.")
    else:
        logger.info(f"[hydrate] Setting config.data.species_variables (example: {species_from_processed[0]})")
    data_cfg["species_variables"] = list(species_from_processed)

    if globals_from_processed:
        prev_globals = list(data_cfg.get("global_variables", []))
        if prev_globals and prev_globals != globals_from_processed:
            logger.warning("[hydrate] Overriding config.data.global_variables with values from processed data.")
        else:
            logger.info(f"[hydrate] Setting config.data.global_variables {globals_from_processed}")
        data_cfg["global_variables"] = list(globals_from_processed)

    if time_from_processed:
        prev_time = data_cfg.get("time_variable")
        if prev_time and prev_time != time_from_processed:
            logger.warning("[hydrate] Overriding config.data.time_variable with value from processed data.")
        else:
            logger.info(f"[hydrate] Setting config.data.time_variable = {time_from_processed}")
        data_cfg["time_variable"] = str(time_from_processed)

    # Decide target_species to use (artifacts > valid config > fallback to all species)
    prev_targets = list(data_cfg.get("target_species", []))
    chosen_targets: Optional[Sequence[str]] = None

    if targets_from_processed:
        chosen_targets = list(targets_from_processed)
        if prev_targets and prev_targets != chosen_targets:
            logger.warning("[hydrate] Overriding config.data.target_species with values from processed data.")
    else:
        # If config already specifies target_species and it's valid, keep it
        if prev_targets:
            missing = set(prev_targets) - set(species_from_processed)
            if missing:
                logger.warning(
                    "[hydrate] config.data.target_species contains names not present in processed species; "
                    f"overriding with full species_variables. Missing={sorted(missing)}"
                )
                chosen_targets = list(species_from_processed)
            else:
                chosen_targets = prev_targets
        else:
            # Default to all species
            chosen_targets = list(species_from_processed)

    data_cfg["target_species"] = list(chosen_targets)
    logger.info(f"[hydrate] target_species set (S_out={len(chosen_targets)}).")

    # Validate species dimensions against shard index when available
    if index is not None:
        # Input species dimension
        for key in ("n_species", "num_species", "n_input_species", "species_dim"):
            val = index.get(key)
            if isinstance(val, int) and val > 0:
                if val != len(data_cfg["species_variables"]):
                    raise RuntimeError(
                        "[hydrate] Species dimension mismatch:\n"
                        f"          shard_index: {val}  vs  "
                        f"len(config.data.species_variables): {len(data_cfg['species_variables'])}\n"
                        "          The processed folder was generated with a different species list."
                    )
                logger.info(f"[hydrate] Species dimension check passed (S_in={val}).")
                break

        # Target species dimension (if index provides it)
        for key in ("n_target_species", "target_species_dim", "species_dim_out"):
            val = index.get(key)
            if isinstance(val, int) and val > 0:
                if val != len(data_cfg["target_species"]):
                    raise RuntimeError(
                        "[hydrate] Target species dimension mismatch:\n"
                        f"          shard_index: {val}  vs  "
                        f"len(config.data.target_species): {len(data_cfg['target_species'])}\n"
                        "          The processed folder expects a different target subset."
                    )
                logger.info(f"[hydrate] Target species dimension check passed (S_out={val}).")
                break

    # dt spec presence (optional requirement)
    require_dt = bool(cfg.get("dataset", {}).get("require_dt_stats", False))
    dt_spec = (manifest or {}).get("dt") if manifest else None
    if require_dt and not dt_spec:
        raise RuntimeError(
            "[hydrate] cfg.dataset.require_dt_stats=True but 'dt' block is missing in normalization.json."
        )
    if dt_spec:
        logger.info(f"[hydrate] dt-spec present in normalization.json (keys={list(dt_spec.keys())}).")

    # Split stats logging (samples if possible; else shard counts)
    def _pick_split_counts() -> Optional[Dict[str, int]]:
        if summary:
            v = summary.get("valid_trajectories")
            if isinstance(v, dict) and v:
                try:
                    return {k: int(v[k]) for k in v}
                except Exception:
                    pass
        if index and isinstance(index.get("splits"), dict):
            try:
                return {k: int(index["splits"][k].get("n_trajectories", 0)) for k in index["splits"].keys()}
            except Exception:
                pass

        def _hunt(container: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
            if not container:
                return None
            for k in ("split_counts", "samples_per_split", "n_samples", "counts", "valid_trajectories"):
                v = container.get(k)
                if isinstance(v, dict) and v:
                    try:
                        return {str(x): int(v[x]) for x in v if isinstance(v[x], (int, float))}
                    except Exception:
                        continue
            return None

        sc = _hunt(summary) or _hunt(report) or _hunt((manifest or {}).get("meta") if manifest else None)
        if sc:
            return sc
        return None

    sample_counts = _pick_split_counts()
    if sample_counts:
        def _first(d: Dict[str, int], keys) -> Union[int, str]:
            for k in keys:
                if k in d:
                    return int(d[k])
            return "?"
        tr = _first(sample_counts, ("train", "training"))
        va = _first(sample_counts, ("validation", "val"))
        te = _first(sample_counts, ("test",))
        logger.info(f"[hydrate] Split counts (samples): train={tr}, validation={va}, test={te}")
    else:
        def _count_npz(split: str) -> int:
            p = processed_dir / split
            if not p.exists():
                return 0
            try:
                return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".npz")
            except Exception:
                return 0
        trf = _count_npz("train")
        vaf = _count_npz("validation")
        tef = _count_npz("test")
        logger.info(f"[hydrate] Split counts (shards): train={trf} files, validation={vaf} files, test={tef} files")

    logger.info(
        f"[hydrate] Completed re-injection from "
        f"{'normalization.json' if manifest else 'processed artifacts'} "
        f"(S_in={len(data_cfg['species_variables'])}, S_out={len(data_cfg['target_species'])})."
    )
    return processed_dir



# --------------------------------------------------------------------------------------
# Ensure preprocessed data exists; re-hydrate from it when present
# --------------------------------------------------------------------------------------

def ensure_preprocessed_data(
        cfg: Dict[str, Any],
        logger: logging.Logger,
) -> Path:
    """
    Ensure preprocessed data exists. If it already exists, re-hydrate cfg from it and
    skip heavy preprocessing. Otherwise, run the preprocessor and then hydrate.

    Returns:
        Path to processed data directory.

    Raises:
        RuntimeError if preprocessing fails to produce normalization.json when required.
    """
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()
    normalization_path = processed_dir / "normalization.json"

    if normalization_path.exists():
        logger.info(f"[ensure_preprocessed] Found existing normalization manifest: {normalization_path}")
        # Even when skipping preprocessing, always re-inject authoritative metadata
        hydrate_config_from_processed(cfg, logger, processed_dir)
        return processed_dir

    # No manifest -> run preprocessing now
    logger.info(f"[ensure_preprocessed] Normalization manifest not found at {normalization_path}")
    pre_logger = logger.getChild("preprocessor")
    dp = DataPreprocessor(cfg, logger=pre_logger)
    dp.run()

    # Verify artifacts were produced
    if not normalization_path.exists():
        raise RuntimeError(
            "[ensure_preprocessed] Preprocessing finished but normalization.json is missing.\n"
            f"Expected at: {normalization_path}"
        )

    logger.info("[ensure_preprocessed] Preprocessing completed successfully.")
    # Re-hydrate cfg from the newly written artifacts
    hydrate_config_from_processed(cfg, logger, processed_dir)

    # Persist a config snapshot next to the artifacts for full provenance
    dump_json(processed_dir / "config.snapshot.json", cfg)
    logger.info(f"[ensure_preprocessed] Wrote config snapshot: {processed_dir / 'config.snapshot.json'}")
    return processed_dir


# --------------------------------------------------------------------------------------
# Dataset and DataLoader construction
# --------------------------------------------------------------------------------------

def build_datasets_and_loaders(
        cfg: Dict[str, Any],
        device: torch.device,
        runtime_dtype: torch.dtype,
        logger: logging.Logger,
) -> Tuple[FlowMapPairsDataset, FlowMapPairsDataset,
torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build training and validation datasets with their dataloaders.

    - Supports GPU-resident datasets for maximum throughput when memory permits.
    - Sampling (pairs/anchor times) is implemented inside the dataset; no external shuffling.

    Returns:
        (train_dataset, val_dataset, train_loader, val_loader)
    """
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    processed_dir = Path(cfg["paths"]["processed_data_dir"]).expanduser().resolve()

    # Dataset sampling parameters
    pairs_per_traj = int(training_cfg.get("pairs_per_traj", 64))
    pairs_per_traj_val = int(training_cfg.get("pairs_per_traj_val", pairs_per_traj))
    min_steps = int(training_cfg.get("min_steps", 1))
    max_steps_raw = training_cfg.get("max_steps")
    max_steps_val = int(max_steps_raw) if max_steps_raw else None
    base_seed = int(cfg.get("system", {}).get("seed", 42))

    # Train dataset (optionally GPU-resident)
    train_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="train",
        config=cfg,
        pairs_per_traj=pairs_per_traj,
        min_steps=min_steps,
        max_steps=max_steps_val,
        preload_to_gpu=bool(dataset_cfg.get("preload_train_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=base_seed,
    )

    # Validation dataset (different seed for independent sampling)
    val_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="validation",
        config=cfg,
        pairs_per_traj=pairs_per_traj_val,
        min_steps=min_steps,
        max_steps=max_steps_val,
        preload_to_gpu=bool(dataset_cfg.get("preload_val_to_gpu", True)),
        device=device,
        dtype=runtime_dtype,
        seed=base_seed + 1337,
    )

    # Batch sizes (must be explicit integers in config)
    batch_size_train = int(training_cfg.get("batch_size", 512))
    batch_size_val = int(training_cfg.get("val_batch_size", batch_size_train))
    logger.info(f"[batch] train batch_size = {batch_size_train}")
    logger.info(f"[batch] validation batch_size = {batch_size_val}")

    # DataLoader worker/threading defaults (support "auto")
    nw_raw = dataset_cfg.get("num_workers", "auto")
    nw_val_raw = dataset_cfg.get("num_workers_val", nw_raw)
    num_workers = _parse_int_or_auto(nw_raw, _auto_num_workers(0))
    num_workers_val = _parse_int_or_auto(nw_val_raw, num_workers)
    logger.info(f"[loader] num_workers train/val = {num_workers}/{num_workers_val}")

    # Prefetch factors
    prefetch_train = int(dataset_cfg.get("prefetch_factor", 2))
    prefetch_val = int(dataset_cfg.get("prefetch_factor_val", prefetch_train))

    # DataLoaders (workers typically 0 when data are GPU-resident)
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(dataset_cfg.get("pin_memory", False)),
        prefetch_factor=prefetch_train,
        persistent_workers=bool(dataset_cfg.get("persistent_workers", False)),
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=bool(dataset_cfg.get("pin_memory_val", dataset_cfg.get("pin_memory", False))),
        prefetch_factor=prefetch_val,
        persistent_workers=bool(
            dataset_cfg.get("persistent_workers_val", dataset_cfg.get("persistent_workers", False))),
    )

    # Sanity-check that loaders are non-empty
    validate_dataloaders(
        train_loader, val_loader,
        batch_size_train, batch_size_val,
        processed_dir, logger
    )

    # Log multi-time configuration clearly
    logger.info(
        "Dataset configuration: "
        f"multi_time_per_anchor={bool(dataset_cfg.get('multi_time_per_anchor', False))}, "
        f"times_per_anchor={int(dataset_cfg.get('times_per_anchor', 1))}, "
        f"share_times_across_batch={bool(dataset_cfg.get('share_times_across_batch', False))}"
    )

    return train_dataset, val_dataset, train_loader, val_loader


def validate_dataloaders(
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        batch_size_train: int,
        batch_size_val: int,
        processed_dir: Path,
        logger: logging.Logger
) -> None:
    """
    Validate that dataloaders contain data and provide concise stats.
    """
    try:
        n_train_items = len(train_loader.dataset)
        n_val_items = len(val_loader.dataset) if val_loader else 0
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader) if val_loader else 0
    except Exception:
        n_train_items = n_val_items = n_train_batches = n_val_batches = 0

    logger.info(
        "Dataset statistics: "
        f"train={n_train_items} items, {n_train_batches} batches | "
        f"val={n_val_items} items, {n_val_batches} batches"
    )

    if n_train_batches == 0:
        raise RuntimeError(
            "Training dataloader is empty. "
            f"Items={n_train_items}, batch_size={batch_size_train}. "
            f"Check training data in {processed_dir / 'train'}"
        )

    if val_loader and n_val_batches == 0:
        raise RuntimeError(
            "Validation dataloader is empty. "
            f"Items={n_val_items}, batch_size={batch_size_val}. "
            f"Check validation data in {processed_dir / 'validation'}"
        )


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------

def build_model(
        cfg: Dict[str, Any],
        device: torch.device,
        logger: logging.Logger
) -> torch.nn.Module:
    """
    Build and configure the Flow-map model, then move it to the target device.
    """
    model = create_model(cfg)
    model.to(device)

    # --- I/O dimension diagnostics ---
    S_out = getattr(model, "S_out", getattr(model, "S", None))
    S_in = getattr(model, "S_in", S_out)  # assume symmetric I/O if not exposed
    G = getattr(model, "G", getattr(model, "G_in",
                                    len(cfg.get("data", {}).get("global_variables", [])) or "?"))
    p = getattr(model, "p",
                getattr(getattr(model, "dynamics", None), "r",
                        getattr(model, "rank_l", "?")))
    logger.info(f"Model dims: S_in={S_in}, S_out={S_out}, G={G}, p={p}")

    ti = getattr(model, "target_idx", None)
    if ti is not None:
        try:
            logger.info(f"Target indices: {ti.detach().cpu().tolist()}")
        except Exception:
            try:
                logger.info(f"Target indices: {list(map(int, ti))}")
            except Exception:
                logger.info("Target indices: <unavailable>")

    return model


# --------------------------------------------------------------------------------------
# Resume control
# --------------------------------------------------------------------------------------

def apply_resume_overrides(cfg: Dict[str, Any], cli_resume: Optional[str], logger: logging.Logger) -> None:
    """
    Set environment variable for resume based on precedence:
    CLI --resume > $RESUME/$RESUME_CHECKPOINT > config.training.resume (if auto_resume)
    """
    tr = cfg.setdefault("training", {})
    env_resume = os.environ.get("RESUME") or os.environ.get("RESUME_CHECKPOINT")

    # Normalize values
    def _norm(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in ("", "none", "null"):
            return None
        return s

    cli_resume = _norm(cli_resume)
    env_resume = _norm(env_resume)
    cfg_resume = _norm(tr.get("resume"))

    # Check auto_resume flag
    auto = bool(tr.get("auto_resume", True))

    chosen: Optional[str] = None
    src = None

    if cli_resume is not None:
        chosen, src = cli_resume, "cli"
    elif env_resume is not None:
        chosen, src = env_resume, "env"
    elif auto and cfg_resume:
        chosen, src = cfg_resume, "config"

    if chosen:
        os.environ["RESUME"] = chosen
        logger.info(f"Resume set from {src}: {chosen!r}")
    else:
        logger.info("No resume requested (fresh run).")


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    """
    Full training pipeline orchestration.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to config file (JSON/JSONC)")
    parser.add_argument("--resume", type=str, default=None,
                        help='Resume from checkpoint ("auto" or path)')
    args = parser.parse_args()

    # Initialize logging first so subsequent steps are visible
    setup_logging(None)
    logger = logging.getLogger("main")

    # Load configuration (JSONC tolerated by loader)
    cfg = load_json_config(str(args.config))

    # Set/ensure common run-time parameters
    cfg.setdefault("system", {})["seed"] = GLOBAL_SEED

    # Apply resume overrides
    apply_resume_overrides(cfg, args.resume, logger)

    # Ensure work_dir exists in config with fallback
    paths = cfg.setdefault("paths", {})
    work_dir = Path(paths.get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()
    paths["work_dir"] = str(work_dir)

    # Handle existing work directory based on config
    overwrite = bool(cfg.get("system", {}).get("overwrite_work_dir", False))
    if work_dir.exists() and overwrite:
        logger.warning(f"Work directory exists and will be DELETED: {work_dir}")
        import shutil
        shutil.rmtree(work_dir)
        logger.warning(f"Deleted existing work directory: {work_dir}")
    elif work_dir.exists():
        logger.info(f"Work directory exists and will be reused: {work_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Initial dump (pre-hydration snapshot)
    config_save_path = work_dir / "config.json"
    dump_json(config_save_path, cfg)
    logger.info(f"Saved training configuration to {config_save_path}")

    # Reproducibility and hardware optimization
    seed_everything(int(cfg["system"]["seed"]))
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device)

    # Determine runtime dtype for staged tensors (dataset storage)
    storage_dtype_str = str(cfg.get("dataset", {}).get("storage_dtype", "float32")).lower()
    if storage_dtype_str in ("bf16", "bfloat16"):
        runtime_dtype = torch.bfloat16
    elif storage_dtype_str in ("fp16", "float16", "half"):
        runtime_dtype = torch.float16
    else:
        runtime_dtype = torch.float32
    logger.info(f"Dataset storage/runtime dtype: {runtime_dtype}")

    # Ensure preprocessed data exist and re-hydrate cfg from that directory if present
    processed_dir = ensure_preprocessed_data(cfg, logger)
    assert processed_dir.exists(), "Processed data directory must exist after preprocessing or hydration"

    # >>> ADDED: persist the hydrated config (now includes species/globals/time)
    dump_json(config_save_path, cfg)
    logger.info("Saved hydrated configuration (with species) to work_dir/config.json")

    # Build datasets and dataloaders
    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders(
        cfg, device, runtime_dtype, logger
    )

    # Build model
    model = build_model(cfg, device, logger)

    # Resolve gradient accumulation (support "auto" with sensible default)
    tr_cfg = cfg.setdefault("training", {})
    accum_raw = tr_cfg.get("accumulate_grad_batches", 1)
    default_accum = 1 if len(
        train_loader) == 0 or train_loader.batch_size is None or train_loader.batch_size >= 1024 else 2
    accum_value = _parse_int_or_auto(accum_raw, default_accum)
    tr_cfg["accumulate_grad_batches"] = accum_value
    cfg.setdefault("lightning", {})["accumulate_grad_batches"] = accum_value
    logger.info(f"[batch] accumulate_grad_batches = {accum_value}")

    # Initialize trainer (Lightning-backed) and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=logger.getChild("trainer"),
    )

    best_val_loss = trainer.train()
    logger.info(f"Training complete. Best validation loss: {best_val_loss:.6e}")



if __name__ == "__main__":
    main()