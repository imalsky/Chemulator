#!/usr/bin/env python3

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import warnings
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import atomic_write_json, ensure_dir, load_json_config, seed_everything


def _resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def _fixed_config_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "config" / "config.json").resolve()


def _repo_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def resolve_paths(cfg: Dict, *, config_path: Path) -> Dict:
    cfg = dict(cfg)
    pcfg = dict(cfg.get("paths", {}))

    repo_root = _repo_root_from_config(config_path)

    raw_dir = pcfg.get("raw_data_dir", "data/raw")
    processed_dir = pcfg.get("processed_data_dir", "data/processed")
    model_dir = pcfg.get("model_dir", "models")

    pcfg["raw_data_dir"] = str(_resolve_path(repo_root, str(raw_dir)))
    pcfg["processed_data_dir"] = str(_resolve_path(repo_root, str(processed_dir)))
    pcfg["model_dir"] = str(_resolve_path(repo_root, str(model_dir)))

    if "work_dir" in pcfg:
        pcfg["work_dir"] = str(_resolve_path(repo_root, str(pcfg["work_dir"])))

    cfg["paths"] = pcfg
    return cfg


def pick_checkpoint(work_dir: Path) -> Optional[Path]:
    best = work_dir / "best.ckpt"
    last = work_dir / "last.ckpt"
    if best.exists():
        return best
    if last.exists():
        return last
    ckpts = sorted(work_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def load_manifest(processed_dir: Path) -> Dict:
    man_path = processed_dir / "normalization.json"
    if not man_path.exists():
        raise FileNotFoundError(
            f"Missing {man_path}.\n"
            f"Resolved processed_dir={processed_dir}\n"
            "Expected convention is <repo_root>/data/processed/normalization.json."
        )
    with open(man_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config_against_manifest(cfg: Dict, manifest: Dict) -> None:
    spec = cfg.get("data", {})
    species_cfg = spec.get("species_variables", [])
    globals_cfg = spec.get("global_variables", [])

    species_man = manifest.get("species_variables", [])
    globals_man = manifest.get("global_variables", [])

    if species_cfg and species_man and list(species_cfg) != list(species_man):
        raise ValueError(
            "Config species_variables does not match normalization.json species_variables. "
            "These must match exactly (same names and ordering)."
        )
    if globals_cfg and globals_man and list(globals_cfg) != list(globals_man):
        raise ValueError(
            "Config global_variables does not match normalization.json global_variables. "
            "These must match exactly (same names and ordering)."
        )


def make_dataloaders(cfg: Dict, *, device: torch.device):
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))

    manifest = load_manifest(processed_dir)
    validate_config_against_manifest(cfg, manifest)

    tcfg = cfg.get("training", {})
    rollout_steps = int(tcfg.get("rollout_steps", 1))
    burn_in_steps = int(tcfg.get("burn_in_steps", 0))
    total_steps = rollout_steps + burn_in_steps

    dcfg = cfg.get("dataset", {})
    windows_per_trajectory = int(dcfg.get("windows_per_trajectory", 1))
    seed = int(cfg.get("system", {}).get("seed", 1234))

    preload_to_device = bool(dcfg.get("preload_to_device", False))

    num_devices = int(tcfg.get("devices", 1))
    if preload_to_device and num_devices > 1:
        warnings.warn(
            "dataset.preload_to_device=true with multiple GPUs (training.devices > 1) may cause issues; "
            "set preload_to_device=false for multi-GPU training.",
            UserWarning,
            stacklevel=2,
        )

    train_ds = FlowMapRolloutDataset(
        processed_dir,
        "train",
        total_steps=total_steps,
        windows_per_trajectory=windows_per_trajectory,
        seed=seed,
        preload_to_device=preload_to_device,
        device=device,
        storage_dtype=torch.float32,
    )
    val_ds = FlowMapRolloutDataset(
        processed_dir,
        "validation",
        total_steps=total_steps,
        windows_per_trajectory=windows_per_trajectory,
        seed=seed + 1,
        preload_to_device=preload_to_device,
        device=device,
        storage_dtype=torch.float32,
    )

    test_ds = None
    if (processed_dir / "test").exists():
        test_ds = FlowMapRolloutDataset(
            processed_dir,
            "test",
            total_steps=total_steps,
            windows_per_trajectory=windows_per_trajectory,
            seed=seed + 2,
            preload_to_device=preload_to_device,
            device=device,
            storage_dtype=torch.float32,
        )

    bs = int(tcfg.get("batch_size", 256))
    num_workers = int(tcfg.get("num_workers", 0))
    pin_memory = bool(tcfg.get("pin_memory", device.type == "cuda"))
    if pin_memory and device.type == "mps":
        warnings.warn(
            "training.pin_memory=true but torch MPS does not support pinned memory; disabling pin_memory.",
            UserWarning,
            stacklevel=2,
        )
        pin_memory = False
    persistent_workers = bool(tcfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(tcfg.get("prefetch_factor", 2))

    train_dl = create_dataloader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dl = create_dataloader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_dl = None
    if test_ds is not None:
        test_dl = create_dataloader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    # Diagnostics
    try:
        print(
            f"Dataset sizes: train_samples={len(train_ds)}, val_samples={len(val_ds)}, "
            f"test_samples={len(test_ds) if test_ds is not None else 0}; batch_size={bs}; "
            f"train_batches={len(train_dl)}, val_batches={len(val_dl)}, test_batches={len(test_dl) if test_dl is not None else 0}",
            flush=True,
        )
        if len(train_dl) == 0:
            raise ValueError("Train DataLoader has zero batches. Reduce training.batch_size or ensure drop_last=False.")
    except TypeError:
        pass

    return train_dl, val_dl, test_dl, manifest


def _runtime_mode(cfg: Dict) -> str:
    mode = str(cfg.get("runtime", {}).get("mode", "train")).strip().lower()
    if mode not in ("train", "eval"):
        raise ValueError(f"runtime.mode must be 'train' or 'eval', got: {mode!r}")
    return mode


def _runtime_checkpoint(cfg: Dict, *, config_path: Path) -> Optional[Path]:
    ckpt = cfg.get("runtime", {}).get("checkpoint", None)
    if ckpt in (None, ""):
        return None
    repo_root = _repo_root_from_config(config_path)
    return _resolve_path(repo_root, str(ckpt))


def main() -> None:
    cfg_path = _fixed_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Expected config at {cfg_path}, but it does not exist.\n"
            "Required convention: <repo_root>/config/config.json"
        )

    cfg = load_json_config(cfg_path)
    cfg = resolve_paths(cfg, config_path=cfg_path)

    seed = int(cfg.get("system", {}).get("seed", 1234))
    seed_everything(seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Work/model dirs
    paths = cfg.get("paths", {})
    work_dir = ensure_dir(paths.get("work_dir", "models/run"))
    ensure_dir(paths.get("model_dir", "models"))


    # Data
    train_dl, val_dl, test_dl, manifest = make_dataloaders(cfg, device=device)
    # If using cosine warmup scheduler and total_steps is not set, compute a deterministic default.
    sched_cfg = (cfg.get("training", {}) or {}).get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "none")).lower().strip()
    if sched_name not in ("", "none", "null") and int(sched_cfg.get("total_steps", 0)) <= 0:
        try:
            steps_per_epoch = int(math.ceil(len(train_dl) / float(max(1, int(cfg.get("training", {}).get("accumulate_grad_batches", 1))))))
            total_steps = steps_per_epoch * int(cfg.get("training", {}).get("max_epochs", 1))
            cfg["training"].setdefault("scheduler", {})["total_steps"] = int(total_steps)
        except Exception:
            # Fall back to Lightning inference in configure_optimizers; may require manual total_steps.
            pass

    # Save exact config used for this run next to checkpoints.
    atomic_write_json(Path(work_dir) / "config.json", cfg)



    # Model and Lightning module
    species_vars = list(manifest.get("species_variables") or [])
    model = create_model(cfg)
    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
        work_dir=Path(work_dir),
    )

    # Attach dataloaders to the module to satisfy Lightning versions that require them.
    lit_module.set_dataloaders(train_dl, val_dl, test_dl)

    trainer = build_lightning_trainer(cfg, work_dir=Path(work_dir))

    mode = _runtime_mode(cfg)
    ckpt_path = _runtime_checkpoint(cfg, config_path=cfg_path)
    if ckpt_path is None:
        ckpt_path = pick_checkpoint(Path(work_dir))

    if mode == "train":
        resume = bool(cfg.get("training", {}).get("resume", True))
        trainer.fit(lit_module, ckpt_path=str(ckpt_path) if (resume and ckpt_path) else None)
        return

    # eval
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {work_dir} and runtime.checkpoint is not set.")
    trainer.validate(lit_module, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    main()
