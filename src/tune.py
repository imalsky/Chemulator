#!/usr/bin/env python3
from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import optuna

from utils import setup_logging, seed_everything, load_json_config, dump_json
from hardware import setup_device, optimize_hardware
from main import (
    GLOBAL_WORK_DIR,
    DEFAULT_CONFIG_PATH,
    ensure_preprocessed_data,
    build_datasets_and_loaders,
    validate_dataloaders,
    _parse_int_or_auto,
    _auto_num_workers,
    build_model,
)
from dataset import create_dataloader
from trainer import Trainer


# -------------------- GLOBALS --------------------

CONFIG_PATH = DEFAULT_CONFIG_PATH
STUDY_NAME = "default"
N_TRIALS = 100
MONITOR_KEY = "val"

# Delete or comment any entry below to SKIP searching it.
# Lists => categorical
# Tuples of len 2 => uniform (int if both ints, else float)
# ("log", lo, hi) => log-uniform float
SWEEP: Dict[str, Any] = {
    # -------------------------
    # model.*  (ARCHITECTURE ONLY)
    # -------------------------
    "model.latent_dim": [48, 64, 96, 128, 160, 192, 256],

    # Depth/width menus – keep a few large options for GH200, plus lean options
    "model.encoder_hidden": [
        [256, 256],
        [384, 384, 384],
        [512, 512, 512],
        [512, 512, 512, 512],
        [768, 768, 768],
        [1024, 1024],
    ],
    "model.dynamics_hidden": [
        [256, 256],
        [256, 256, 256],
        [512, 256],
        [512, 512],
        [512, 512, 512],
    ],
    "model.decoder_hidden": [
        [256, 256],
        [384, 384, 384],
        [512, 512, 512],
        [512, 512, 512, 512],
        [768, 768, 768],
    ],

    # Safe, supported activations in your model.py
    "model.activation": ["silu", "gelu", "mish", "tanh", "relu"],

    # Regularization
    "model.dropout": (0.0, 0.30),  # uniform float in [0, 0.30]

    # Behavioral flags
    "model.dynamics_residual": [True, False],
    "model.decoder_condition_on_g": [True, False],
    "model.predict_logit_delta": [True, False],
    "model.vae_mode": [True, False],
    "model.allow_partial_simplex": [False],  # keep strict simplex unless you truly want a subset

    # -------------------------
    # training.*  (BASICS ONLY)
    # -------------------------
    "training.lr": ("log", 3e-5, 3e-3),
    "training.weight_decay": ("log", 1e-7, 1e-3),
    "training.batch_size": [512, 1024, 2048, 4096],

    # Only used when VAE is on; _apply_sweep already gates this
    "training.beta_kl": (0.0, 0.02),

    # -------------------------
    # auxiliary_losses.*  (KEPT OFF)
    # -------------------------
    "training.auxiliary_losses.rollout_enabled": [False],
    # rollout knobs intentionally commented out so they are NOT explored
    # "training.auxiliary_losses.rollout_weight": ("log", 5e-2, 1.0),
    # "training.auxiliary_losses.rollout_horizon": [1, 2, 4, 8],
    # "training.auxiliary_losses.rollout_warmup_epochs": (5, 40),
    # "training.auxiliary_losses.rollout_teacher_forcing.mode": ["none", "linear"],
    # "training.auxiliary_losses.rollout_teacher_forcing.start_p": (0.0, 1.0),
    # "training.auxiliary_losses.rollout_teacher_forcing.end_p": (0.0, 1.0),
    # "training.auxiliary_losses.rollout_teacher_forcing.end_epoch": (5, 60),

    "training.auxiliary_losses.semigroup.enabled": [False],
    "training.auxiliary_losses.semigroup.weight": ("log", 1e-4, 5e-2),
    "training.auxiliary_losses.semigroup.warmup_epochs": (0, 30),

    # lightning.* remains out of sweep (commented -> not explored)
    # "lightning.early_stop_patience": (3, 10),
}



# -------------------- UTILS --------------------

def _runtime_dtype_from_cfg(cfg: Dict[str, Any]) -> torch.dtype:
    """
    Map dataset.storage_dtype -> runtime dtype used for dataset buffers.
    Matches main() logic.
    """
    storage_dtype_str = str(cfg.get("dataset", {}).get("storage_dtype", "float32")).lower()

    if storage_dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if storage_dtype_str in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _get(cfg: Dict[str, Any], dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set(cfg: Dict[str, Any], dotted: str, value):
    parts = dotted.split(".")
    cur = cfg
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def _suggest(trial: optuna.Trial, name: str, space):
    if isinstance(space, list):
        return trial.suggest_categorical(name, space)

    if isinstance(space, tuple) and len(space) == 2:
        lo, hi = space
        if isinstance(lo, int) and isinstance(hi, int):
            return trial.suggest_int(name, lo, hi)
        return trial.suggest_float(name, float(lo), float(hi))

    if isinstance(space, tuple) and len(space) == 3 and space[0] == "log":
        _, lo, hi = space
        return trial.suggest_float(name, float(lo), float(hi), log=True)

    raise ValueError(f"Unsupported sweep space for {name}: {space}")


def _apply_sweep(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample trial config from SWEEP, then apply sanity cleanup:
      - Gate beta_kl on vae_mode
      - Disable rollout knobs if rollout_enabled=False
      - Clamp teacher forcing schedule
      - Clamp rollout warmup to <= total epochs
      - Force val_batch_size if missing
    """
    cfg = copy.deepcopy(base_cfg)

    # 1) Sample from SWEEP
    for key, space in SWEEP.items():
        if space is None:
            continue
        try:
            val = _suggest(trial, key, space)
            _set(cfg, key, val)
        except Exception as e:
            raise RuntimeError(f"Sampling {key} failed: {e}")

    # 2) Gate β search on VAE mode
    vae_on = bool(_get(cfg, "model.vae_mode", False))
    if not vae_on and "training.beta_kl" in SWEEP:
        _set(cfg, "training.beta_kl", _get(base_cfg, "training.beta_kl", 0.0))

    # 3) Rollout / teacher forcing cleanup
    rollout_enabled = bool(_get(cfg, "training.auxiliary_losses.rollout_enabled", False))
    total_epochs = int(_get(cfg, "training.epochs", 50))

    if not rollout_enabled:
        # rollout is off -> zero out related knobs so noise doesn't leak into the run
        _set(cfg, "training.auxiliary_losses.rollout_weight", 0.0)
        _set(cfg, "training.auxiliary_losses.rollout_horizon", 1)
        _set(cfg, "training.auxiliary_losses.rollout_warmup_epochs", 0)

        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.mode", "none")
        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", 0.0)
        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", 0.0)
        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_epoch", 0)
    else:
        # enforce sane TF schedule
        start_p = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", 1.0))
        end_p = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", 0.0))

        # enforce monotone decay: start_p >= end_p
        if start_p < end_p:
            start_p, end_p = end_p, start_p

        # clamp probs into [0,1]
        start_p = max(0.0, min(1.0, start_p))
        end_p = max(0.0, min(1.0, end_p))

        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", start_p)
        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", end_p)

        # end_epoch can't exceed total epochs
        end_epoch = int(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_epoch", total_epochs))
        if end_epoch > total_epochs:
            end_epoch = total_epochs
        _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_epoch", end_epoch)

        # rollout warmup shouldn't last longer than total training
        warmup_epochs = int(_get(cfg, "training.auxiliary_losses.rollout_warmup_epochs", 0))
        if warmup_epochs > total_epochs:
            warmup_epochs = total_epochs
        _set(cfg, "training.auxiliary_losses.rollout_warmup_epochs", warmup_epochs)

    # 4) semigroup cleanup
    semigroup_enabled = bool(_get(cfg, "training.auxiliary_losses.semigroup.enabled", False))
    if not semigroup_enabled:
        _set(cfg, "training.auxiliary_losses.semigroup.weight", 0.0)
        _set(cfg, "training.auxiliary_losses.semigroup.warmup_epochs", 0)

    # 5) Force val batch if missing
    bs = int(_get(cfg, "training.batch_size", 512))
    _set(cfg, "training.val_batch_size", int(_get(cfg, "training.val_batch_size", bs)))

    return cfg


def _make_loaders_for_trial(
    cfg_trial: Dict[str, Any],
    train_ds,
    val_ds,
    logger: logging.Logger,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Wrap the (persistent) dataset objects in DataLoaders configured for this trial.
    """
    dcfg = cfg_trial.get("dataset", {})
    tcfg = cfg_trial.get("training", {})

    bs_tr = int(tcfg.get("batch_size", 512))
    bs_va = int(tcfg.get("val_batch_size", bs_tr))

    # workers
    nw = _parse_int_or_auto(dcfg.get("num_workers", "auto"), _auto_num_workers(0))
    nw_val = _parse_int_or_auto(dcfg.get("num_workers_val", nw), nw)

    # prefetch
    pf_tr = int(dcfg.get("prefetch_factor", 2))
    pf_val = int(dcfg.get("prefetch_factor_val", pf_tr))

    # train loader
    train_loader = create_dataloader(
        dataset=train_ds,
        batch_size=bs_tr,
        shuffle=False,
        num_workers=nw,
        pin_memory=bool(dcfg.get("pin_memory", False)),
        prefetch_factor=pf_tr,
        persistent_workers=bool(dcfg.get("persistent_workers", False)),
    )

    # val loader
    val_loader = create_dataloader(
        dataset=val_ds,
        batch_size=bs_va,
        shuffle=False,
        num_workers=nw_val,
        pin_memory=bool(dcfg.get("pin_memory_val", dcfg.get("pin_memory", False))),
        prefetch_factor=pf_val,
        persistent_workers=bool(
            dcfg.get("persistent_workers_val", dcfg.get("persistent_workers", False))
        ),
    )

    processed_dir = Path(cfg_trial["paths"]["processed_data_dir"]).expanduser().resolve()
    validate_dataloaders(
        train_loader,
        val_loader,
        bs_tr,
        bs_va,
        processed_dir,
        logger,
    )

    return train_loader, val_loader


# -------------------- OBJECTIVE --------------------

def _make_objective(
    base_cfg: Dict[str, Any],
    study_root: Path,
    device: torch.device,
    train_ds,
    val_ds,
    logger: logging.Logger,
):
    """
    Optuna objective factory.

    Each trial:
      1. Re-seeds deterministically (base_seed + trial.number).
      2. Samples a cfg_trial via _apply_sweep.
      3. Resets dataset RNG/index maps so every trial sees the same sampling schedule.
      4. Builds DataLoaders with that cfg_trial.
      5. Resolves accumulate_grad_batches for Lightning.
      6. Builds model + Trainer, runs training.
      7. Returns the best validation loss.
    """

    def objective(trial: optuna.Trial) -> float:
        try:
            # deterministic per-trial seed
            base_seed = int(base_cfg.get("system", {}).get("seed", 42))
            trial_seed = base_seed + int(trial.number)
            seed_everything(trial_seed)

            trial_dir = study_root / f"trial_{trial.number:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # sample HPs
            cfg_trial = _apply_sweep(trial, base_cfg)

            # set trial work_dir
            paths_trial = cfg_trial.setdefault("paths", {})
            paths_trial["work_dir"] = str(trial_dir)

            # reset dataset sampling so each trial trains on the same initial schedule
            if hasattr(train_ds, "set_epoch"):
                try:
                    train_ds.set_epoch(0)
                except Exception as e:
                    logger.warning(f"train_ds.set_epoch(0) failed: {e}")
            if hasattr(val_ds, "set_epoch"):
                try:
                    val_ds.set_epoch(0)
                except Exception as e:
                    logger.warning(f"val_ds.set_epoch(0) failed: {e}")

            # loaders with this trial's batch size / workers
            data_logger = logger.getChild(f"trial{trial.number:03d}.data")
            train_loader, val_loader = _make_loaders_for_trial(
                cfg_trial, train_ds, val_ds, data_logger
            )

            # resolve gradient accumulation ("auto" allowed in config)
            default_accum = (
                1
                if (
                    train_loader.batch_size is None
                    or len(train_loader) == 0
                    or train_loader.batch_size >= 1024
                )
                else 2
            )
            tcfg = cfg_trial.setdefault("training", {})
            accum_raw = tcfg.get("accumulate_grad_batches", "auto")
            accum = _parse_int_or_auto(accum_raw, default_accum)
            tcfg["accumulate_grad_batches"] = accum
            cfg_trial.setdefault("lightning", {})["accumulate_grad_batches"] = accum

            # model
            model_logger = logger.getChild(f"trial{trial.number:03d}.model")
            model = build_model(cfg_trial, device, model_logger)

            # train
            run = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_trial,
                work_dir=trial_dir,
                device=device,
                logger=logger.getChild(f"trial{trial.number:03d}.trainer"),
                optuna_trial=trial,        # enable pruning callback in Trainer
                optuna_monitor=MONITOR_KEY,
            )

            best_val = float(run.train())

            # record + save cfg snapshot for this trial
            trial.set_user_attr("best_val_loss", best_val)
            dump_json(trial_dir / "trial_config.final.json", cfg_trial)

            return best_val

        except optuna.TrialPruned:
            # Let Optuna handle the pruning.
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return float("inf")

    return objective


# -------------------- MAIN --------------------

def main():
    setup_logging(None)
    logger = logging.getLogger("tune")

    # Load config and global seed
    cfg = load_json_config(str(CONFIG_PATH))
    cfg.setdefault("system", {})["seed"] = int(cfg["system"].get("seed", 42))
    seed_everything(int(cfg["system"]["seed"]))

    # work_dir base
    paths = cfg.setdefault("paths", {})
    base = Path(paths.get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()
    paths["work_dir"] = str(base)
    base.mkdir(parents=True, exist_ok=True)

    # study dir
    study_root = base / "tuning" / STUDY_NAME
    study_root.mkdir(parents=True, exist_ok=True)

    # hw/runtime setup
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device)
    runtime_dtype = _runtime_dtype_from_cfg(cfg)

    # make sure we have data + hydrated cfg from disk
    processed_dir = ensure_preprocessed_data(cfg, logger.getChild("pre"))
    assert processed_dir.exists(), "Processed data directory must exist."
    dump_json(study_root / "config.hydrated.json", cfg)

    # build datasets ONCE; reuse objects across trials
    train_ds, val_ds, _, _ = build_datasets_and_loaders(
        cfg, device, runtime_dtype, logger.getChild("data.init")
    )

    # Optuna study (sqlite so it's inspectable and resumable)
    storage = f"sqlite:///{study_root / 'optuna_study.db'}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["system"]["seed"])),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )

    objective = _make_objective(cfg, study_root, device, train_ds, val_ds, logger)
    study.optimize(objective, n_trials=N_TRIALS)

    # dump best config
    best = study.best_trial
    logging.getLogger().info(
        f"best #{best.number} loss={best.value:.6e} params={best.params}"
    )
    src = study_root / f"trial_{best.number:03d}" / "trial_config.final.json"
    if src.exists():
        shutil.copy(src, study_root / "best_config.json")


if __name__ == "__main__":
    main()
