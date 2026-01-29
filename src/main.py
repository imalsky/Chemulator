#!/usr/bin/env python3
"""
main.py - Entrypoint for training / evaluation.

Notes:
- This code assumes configuration and preprocessing outputs are internally consistent.
- Paths are resolved deterministically against the config.json directory.
- Dataloaders are sized for the maximum rollout length actually used (base horizon + optional curriculum for train).
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import seed_everything

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config

log = logging.getLogger(__name__)


# ==============================================================================
# Basic config / path helpers
# ==============================================================================


def _repo_root(cfg_path: Path) -> Path:
    return cfg_path.parent.resolve()


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((root / pth).resolve())


def resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    root = _repo_root(cfg_path)
    cfg = dict(cfg)

    paths = dict(cfg["paths"])
    for k, v in list(paths.items()):
        if isinstance(v, str) and v.strip():
            paths[k] = _resolve_path(root, v)
    cfg["paths"] = paths

    runtime = dict(cfg.get("runtime", {}) or {})
    ckpt = runtime.get("checkpoint")
    if isinstance(ckpt, str) and ckpt.strip():
        runtime["checkpoint"] = _resolve_path(root, ckpt)
    cfg["runtime"] = runtime

    return cfg


def configure_logging(cfg: Dict[str, Any]) -> None:
    sys_cfg = cfg.get("system", {}) or {}
    level_str = str(sys_cfg.get("log_level", "INFO")).upper().strip()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.captureWarnings(True)


def select_device(cfg: Dict[str, Any]) -> torch.device:
    """Used only for dataset preloading (dataset.preload_to_device=True) and any eager ops outside Lightning."""
    sys_cfg = cfg.get("system", {}) or {}
    pref = str(sys_cfg.get("device", "auto")).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref)
        log.warning("Requested device=%s but CUDA is unavailable; using CPU.", pref)
        return torch.device("cpu")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("Requested device=mps but MPS is unavailable; using CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# Minimal schema accessors (scientific code; rely on KeyError/TypeError from Python)
# ==============================================================================


def _require_dict(cfg: Dict[str, Any], key: str, *, context: str) -> Dict[str, Any]:
    _ = context
    return cfg[key]


def _require_key(cfg: Dict[str, Any], key: str, *, context: str) -> Any:
    _ = context
    return cfg[key]


def validate_training_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return cfg['training'] (no extra validation; trainer is the source of truth)."""
    return cfg["training"]


def load_manifest_and_sync_data_cfg(
    cfg: Dict[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """
    Load normalization.json and ensure cfg.data.{species_variables,global_variables} are populated.

    Authoritative precedence:
      manifest values (if present) > config.data values.

    This function intentionally avoids strict consistency checks; incorrect inputs should fail naturally
    when the dataset/model/trainer consume them.
    """
    manifest_path = processed_dir / "normalization.json"
    manifest = load_json_config(manifest_path)

    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}) or {})

    man_species = list((manifest or {}).get("species_variables") or [])
    man_globals = list((manifest or {}).get("global_variables") or [])

    cfg_species = list(data_cfg.get("species_variables") or [])
    cfg_globals = list(data_cfg.get("global_variables") or [])

    species_vars = man_species or cfg_species
    global_vars = man_globals or cfg_globals

    data_cfg["species_variables"] = list(species_vars)
    data_cfg["global_variables"] = list(global_vars)
    cfg["data"] = data_cfg

    return cfg, manifest, list(species_vars), list(global_vars)


# ==============================================================================
# Dataloaders (sized for rollout lengths actually used)
# ==============================================================================


def _max_rollout_steps_for_stage(tcfg: Dict[str, Any], *, stage: str) -> int:
    """
    Base horizon: training.rollout_steps.

    Curriculum (if enabled) can increase K for training only; for sizing we take its max_k.
    long_rollout is intentionally ignored (trainer no longer uses it).
    """
    base_k = int(tcfg["rollout_steps"])
    if str(stage).lower().strip() != "train":
        return base_k

    cur_cfg = tcfg.get("curriculum")
    if isinstance(cur_cfg, dict) and bool(cur_cfg.get("enabled", False)):
        return int(max(base_k, int(cur_cfg["max_k"])))
    return base_k


def create_dataloaders(
    cfg: Dict[str, Any],
    *,
    tcfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[Any, Any, Optional[Any], Dict[str, Any]]:
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"]).expanduser()

    k_train = _max_rollout_steps_for_stage(tcfg, stage="train")
    k_val = _max_rollout_steps_for_stage(tcfg, stage="validation")
    k_test = _max_rollout_steps_for_stage(tcfg, stage="test")

    dcfg = cfg.get("dataset", {}) or {}
    windows_per_traj = int(dcfg.get("windows_per_trajectory", 1))
    preload = bool(dcfg.get("preload_to_device", False))
    shard_cache_size = int(dcfg.get("shard_cache_size", 2))
    use_mmap = bool(dcfg.get("use_mmap", False))

    sys_cfg = cfg.get("system", {}) or {}
    seed = int(sys_cfg.get("seed", 1234))

    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", True))

    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload,
        device=device,
        storage_dtype=torch.float32,
        shard_cache_size=shard_cache_size,
        use_mmap=use_mmap,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_val, seed=seed + 1, **common_ds_kwargs)

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(processed_dir, "test", total_steps=k_test, seed=seed + 2, **common_ds_kwargs)

    if num_workers == 0:
        persistent = False
        prefetch = None
    else:
        persistent = bool(tcfg["persistent_workers"])
        prefetch = int(tcfg["prefetch_factor"])

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    train_dl = create_dataloader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = create_dataloader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_dl = create_dataloader(test_ds, shuffle=False, drop_last=False, **dl_kwargs) if test_ds is not None else None

    log.info(
        "Dataloaders: train=%d batches (K=%d), val=%d batches (K=%d), test=%s",
        len(train_dl), k_train,
        len(val_dl), k_val,
        (f"{len(test_dl)} batches (K={k_test})" if test_dl is not None else "None"),
    )

    manifest = load_json_config(processed_dir / "normalization.json")
    return train_dl, val_dl, test_dl, manifest


# ==============================================================================
# Checkpoints
# ==============================================================================


def find_resume_checkpoint(work_dir: Path) -> Optional[Path]:
    """Find a resume checkpoint under work_dir/checkpoints."""
    ckpt_dir = work_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


# ==============================================================================
# CLI + main
# ==============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_json_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path)
    configure_logging(cfg)

    tcfg = validate_training_schema(cfg)

    sys_cfg = cfg.get("system", {}) or {}
    seed = int(sys_cfg.get("seed", 1234))
    seed_everything(seed, workers=True)

    device = select_device(cfg)
    log.info("Data device (for preloading only): %s", device)

    work_dir = ensure_dir(cfg["paths"]["work_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"]).expanduser()

    cfg, manifest, species_vars, global_vars = load_manifest_and_sync_data_cfg(cfg, processed_dir)

    atomic_write_json(work_dir / "config.resolved.json", cfg)

    train_dl, val_dl, test_dl, _ = create_dataloaders(cfg, tcfg=tcfg, device=device)

    model = create_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model: type=%s | S=%d | G=%d | params=%d (trainable=%d)",
        str(cfg.get("model", {}).get("type", "mlp")),
        len(species_vars),
        len(global_vars),
        total_params,
        trainable_params,
    )

    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
    )

    trainer = build_lightning_trainer(cfg, work_dir=work_dir)

    runtime = cfg.get("runtime", {}) or {}
    mode = str(runtime.get("mode", "train")).lower().strip()
    if mode not in ("train", "test"):
        raise ValueError("runtime.mode must be 'train' or 'test'.")

    ckpt_path = runtime.get("checkpoint")
    if isinstance(ckpt_path, str) and ckpt_path.strip():
        ckpt_path = str(Path(ckpt_path).expanduser())
    else:
        ckpt = find_resume_checkpoint(work_dir)
        ckpt_path = str(ckpt) if ckpt else None

    if mode == "train":
        resume = bool(tcfg.get("resume", True))
        log.info(
            "Training: max_epochs=%s lr=%s wd=%s batch_size=%s resume=%s ckpt=%s",
            tcfg["max_epochs"],
            tcfg["optimizer"]["lr"],
            tcfg["optimizer"].get("weight_decay", 0.0),
            tcfg["batch_size"],
            resume,
            (Path(ckpt_path).name if ckpt_path else "None"),
        )

        trainer.fit(
            lit_module,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=(ckpt_path if (resume and ckpt_path) else None),
        )

        if test_dl is not None:
            trainer.test(lit_module, dataloaders=test_dl, ckpt_path="best")

    else:  # test
        if test_dl is None:
            raise ValueError("runtime.mode='test' but no test split found (processed_dir/test missing).")
        log.info("Testing: ckpt=%s", (Path(ckpt_path).name if ckpt_path else "None"))
        trainer.test(lit_module, dataloaders=test_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
