#!/usr/bin/env python3
"""
tune.py

This version enforces apples-to-apples across trials (same rollout/semigroup curriculum,
same batch size/epochs, etc.) AND makes the validation metric flexible WITHOUT touching trainer.py.

How it works:
- Trainer already logs some metric called "val" and returns best_val from .train().
- Trainer may also (now or in the future) log other metrics like "val_long_rollout".
- Trainer accepts `optuna_monitor=...` so it knows which metric to treat as the key metric.

We DO NOT edit trainer.py. Instead:
- You can select which metric Trainer should treat as the key metric by either:
    1. setting environment variable TUNE_VAL_METRIC="val" or "val_long_rollout" or whatever Trainer knows about
    2. or adding "optuna_monitor": "<metric_name>" under config["training"] in the base config
    3. fallback is "val" if nothing is provided
- tune.py resolves that metric name once and passes it into Trainer, so Trainer can internally use that
  for checkpointing / pruning / best_val, etc.

We ALSO remove the per-trial curriculum branching. Every trial trains with the same rollout+semigroup
settings from the base config. Optuna only explores architecture, optimizer scalars, and loss weights,
not "is rollout loss even on". That keeps trials comparable.
"""

from __future__ import annotations

import copy
import logging
import shutil
import os
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

# Delete or comment any entry below to SKIP searching it.
# Lists => categorical
# Tuples of len 2 => uniform (int if both ints, else float)
# ("log", lo, hi) => log-uniform float
SWEEP: Dict[str, Any] = {
    # -------------------------
    # model.*  (ARCHITECTURE ONLY)
    # -------------------------
    "model.latent_dim": [48, 64, 96, 128, 160, 192, 256],

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

    # Runtime nonlinearities / heads
    "model.activation": ["silu", "gelu", "mish", "tanh", "relu"],
    "model.dropout": (0.0, 0.30),

    # Model behavior flags that do NOT change the existence of loss terms
    "model.dynamics_residual": [True, False],
    "model.decoder_condition_on_g": [True, False],
    "model.predict_logit_delta": [True, False],
    # NOTE:
    #   model.vae_mode is FIXED in the base config (do not sweep AE vs VAE).
    #   model.allow_partial_simplex is FIXED in the base config (don't sweep mass conservation policy).

    # -------------------------
    # training.* (optimizer only)
    # -------------------------
    "training.lr": ("log", 3e-5, 3e-3),
    "training.weight_decay": ("log", 1e-7, 1e-3),
    # NOTE:
    #   training.batch_size is FIXED in base config, not swept.
    #   training.beta_kl is FIXED in base config because vae_mode is fixed.

    # -------------------------
    # auxiliary_losses.* (strength only)
    # -------------------------
    # rollout_enabled / horizon / TF schedule / warmup_epochs are FIXED in base config
    "training.auxiliary_losses.rollout_weight": ("log", 1e-4, 5e-2),

    # semigroup.enabled / warmup_epochs are FIXED in base config
    "training.auxiliary_losses.semigroup.weight": ("log", 1e-5, 1e-2),

    # lightning.* stays out of sweep
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
      - Clamp teacher forcing schedule using fixed curriculum
      - Clamp rollout warmup to <= total epochs
      - Force val_batch_size if missing

    IMPORTANT:
    We do NOT:
      - zero out rollout_weight if rollout_enabled==False
      - zero out semigroup.weight if semigroup.enabled==False
      - overwrite rollout_horizon, warmup_epochs, TF mode, etc.
    Because those are fixed in the base config specifically to keep trials
    apples-to-apples. Every trial must train under the same curriculum.
    """
    cfg = copy.deepcopy(base_cfg)

    # 1) Sample all sweep parameters for this trial
    for key, space in SWEEP.items():
        if space is None:
            continue
        try:
            val = _suggest(trial, key, space)
            _set(cfg, key, val)
        except Exception as e:
            raise RuntimeError(f"Sampling {key} failed: {e}")

    # 2) beta_kl only matters if vae_mode is on
    vae_on = bool(_get(cfg, "model.vae_mode", False))
    if (not vae_on) and ("training.beta_kl" in SWEEP):
        _set(cfg, "training.beta_kl", _get(base_cfg, "training.beta_kl", 0.0))

    # 3) Clamp teacher forcing schedule / rollout warmup using the fixed curriculum
    total_epochs = int(_get(cfg, "training.epochs", 50))

    start_p = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", 1.0))
    end_p = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", 0.0))

    # enforce monotone decay: start_p >= end_p
    if start_p < end_p:
        start_p, end_p = end_p, start_p

    # clamp probs into [0,1]
    start_p = max(0.0, min(1.0, start_p))
    end_p   = max(0.0, min(1.0, end_p))

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

    # 4) Force val batch size if missing
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
    Deterministic order (shuffle=False) so each trial sees data in the same sequence.
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
        shuffle=False,  # keep deterministic for comparability
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


def _resolve_monitor_key(cfg: Dict[str, Any]) -> str:
    """
    Decide which metric Trainer should treat as the key Optuna metric,
    WITHOUT editing trainer.py.

    Priority:
      1. env var TUNE_VAL_METRIC
      2. cfg["training"]["optuna_monitor"]
      3. fallback "val"

    Whatever we return here is passed into Trainer(optuna_monitor=...).
    Trainer is responsible for:
      - logging that metric
      - tracking best value
      - returning that best value from .train()
    """
    env_choice = os.getenv("TUNE_VAL_METRIC", "").strip()
    if env_choice:
        return env_choice

    cfg_choice = (
        cfg.get("training", {})
        .get("optuna_monitor", "")
    )
    if isinstance(cfg_choice, str) and cfg_choice.strip():
        return cfg_choice.strip()

    return "val"


# -------------------- OBJECTIVE --------------------

def _make_objective(
    base_cfg: Dict[str, Any],
    monitor_key: str,
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
      2. Samples cfg_trial via _apply_sweep (architecture/optimizer/weights only).
      3. Resets dataset RNG/index maps so every trial sees the same sampling schedule.
      4. Builds loaders with that cfg_trial.
      5. Resolves accumulate_grad_batches for Lightning.
      6. Builds model + Trainer, runs training.
      7. Returns the best value of `monitor_key` as reported by Trainer.train().
    """

    def objective(trial: optuna.Trial) -> float:
        try:
            # deterministic per-trial seed
            base_seed = int(base_cfg.get("system", {}).get("seed", 42))
            trial_seed = base_seed + int(trial.number)
            seed_everything(trial_seed)

            trial_dir = study_root / f"trial_{trial.number:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # sample HPs for this trial
            cfg_trial = _apply_sweep(trial, base_cfg)

            # set per-trial work_dir in config
            paths_trial = cfg_trial.setdefault("paths", {})
            paths_trial["work_dir"] = str(trial_dir)

            # reset dataset sampling so each trial starts identically
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

            # loaders for this trial
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

            # build model for this trial
            model_logger = logger.getChild(f"trial{trial.number:03d}.model")
            model = build_model(cfg_trial, device, model_logger)

            # Trainer run for this trial
            run = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_trial,
                work_dir=trial_dir,
                device=device,
                logger=logger.getChild(f"trial{trial.number:03d}.trainer"),
                optuna_trial=trial,            # pruning callback inside Trainer
                optuna_monitor=monitor_key,    # <- chosen metric name
            )

            best_val = float(run.train())

            # record + save cfg snapshot
            trial.set_user_attr("best_val_loss", best_val)
            dump_json(trial_dir / "trial_config.final.json", cfg_trial)

            return best_val

        except optuna.TrialPruned:
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

    # make sure data is prepared + cfg is hydrated
    processed_dir = ensure_preprocessed_data(cfg, logger.getChild("pre"))
    assert processed_dir.exists(), "Processed data directory must exist."
    dump_json(study_root / "config.hydrated.json", cfg)

    # build datasets ONCE; reuse objects across trials
    train_ds, val_ds, _, _ = build_datasets_and_loaders(
        cfg, device, runtime_dtype, logger.getChild("data.init")
    )

    # resolve which metric to optimize (flexible, no trainer.py edit needed)
    monitor_key = _resolve_monitor_key(cfg)

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

    objective = _make_objective(cfg, monitor_key, study_root, device, train_ds, val_ds, logger)
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
