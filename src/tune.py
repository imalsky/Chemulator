#!/usr/bin/env python3
"""
tune.py

Aggressive 2-stage hyperparameter search with early culling,
WITHOUT touching trainer.py.

High-level flow per Optuna trial:
---------------------------------
Stage A ("probe"):
  - Train the model for PROBE_EPOCHS (e.g. 3 epochs max).
  - Get best validation metric ("val" by default, or whatever monitor_key is).
  - Save checkpoint + config in trial_dir.

Decision:
  - Compare this probe score to the current study's best score so far.
  - If it's clearly bad (worse than best_so_far * PRUNE_FACTOR, or non-finite),
    we prune this trial immediately (raise optuna.TrialPruned).
    -> That trial ends after ~3 epochs.

Stage B ("full"):
  - Reload from the Stage A checkpoint.
  - Resume training to the full configured number of epochs (e.g. 50).
  - Return final best metric to Optuna.

Why this works with NO trainer.py changes:
------------------------------------------
- We do not rely on per-epoch callbacks or trial.report() inside Trainer.
- We just call Trainer.train() twice ourselves with different cfgs:
    1) short probe, 2) (if promising) full.
- trainer.py already:
    * respects cfg["training"]["epochs"]
    * accepts cfg["lightning"]["resume_from"] as ckpt_path
    * exports best_model.pt / checkpoints/last.ckpt
    * returns best metric from .train()

Fairness / apples-to-apples:
----------------------------
- Every trial uses the SAME rollout/semigroup curriculum, teacher forcing
  schedule, batch size logic, etc. We do not toggle those features per trial.
- The only trials that get to finish are the ones that look promising
  by epoch 3, so we save GPU on obvious losers.

You can tune:
    PROBE_EPOCHS   (audition length)
    PRUNE_FACTOR   (strictness: smaller = harsher cut)

We also:
- Widen SWEEP (latent_dim, widths, beta_kl, etc.).
- Add an early exit if an optuna study already exists, so you don't
  accidentally resume a mismatched sweep.

NOTE: This file assumes Python 3.10+.
"""

from __future__ import annotations

import copy
import logging
import shutil
import os
import math
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


# -------------------- GLOBALS / KNOBS --------------------

CONFIG_PATH = DEFAULT_CONFIG_PATH           # base config path
STUDY_NAME = "v1"                           # tunes live in {work_dir}/tuning/{STUDY_NAME}
N_TRIALS = 100                              # target trials

PROBE_EPOCHS = 10                            # epochs to "audition" a trial
PRUNE_FACTOR = 1.5                          # prune if probe_loss > best_so_far * PRUNE_FACTOR

# Delete or comment any entry below to SKIP searching it.
# Interpretation rules:
#   list                      => categorical
#   (low, high)               => uniform (int if both ints, else float)
#   ("log", low, high)        => log-uniform float
SWEEP: Dict[str, Any] = {
    # ---- model.* (ARCH SHAPE / WIDTHS) ----
    "model.latent_dim": [16, 32, 64, 128, 256, 512],

    # IMPORTANT:
    # Use LISTS (not tuples). Optuna's TPE sampler serializes categorical
    # choices to JSON and will coerce tuples into lists in completed trials.
    # On later trials it then compares those stored lists against the current
    # choices. If we keep tuples here, that comparison blows up with:
    #   ValueError: '[384, 384, 384]' not in ((256, 256), (384, 384, 384), ...)
    # which kills every new trial. Lists avoid that by keeping the type
    # consistent between past trials and current choices.
    "model.encoder_hidden": [
        [256, 256],
        [512, 512, 512],
        [512, 512, 512, 512],
        [1024, 1024],
        [1024, 512, 256],
        [1536, 1024, 512],
        [1024, 1024, 1024],
    ],
    "model.dynamics_hidden": [
        [256, 256],
        [256, 256, 256],
        [512, 256],
        [512, 512],
        [512, 512, 512],
        [768, 768],
        [1024, 512],
        [1024, 1024],
    ],
    "model.decoder_hidden": [
        [256, 256],
        [512, 512, 512],
        [512, 512, 512, 512],
        [256, 512, 1024],
        [512, 1024, 1536],
        [1024, 1024, 1024],
    ],

    # ---- nonlinearities / heads ----
    "model.activation": [
        "silu", "gelu", "mish", "tanh",
        "elu", "leaky_relu", "prelu", "hardswish", "softplus",
    ],

    # Wider dropout band for regularization on larger models
    "model.dropout": (0.0, 0.10),

    # Behavior flags (do NOT remove losses entirely, so still comparable curriculum)
    "model.dynamics_residual": [True, False],
    "model.decoder_condition_on_g": [True, False],
    #"model.predict_logit_delta": [True, False],

    # ---- training.* (optimizer scalars) ----
    "training.lr": ("log", 1e-5, 1e-3),
    "training.weight_decay": ("log", 1e-7, 3e-3),

    # ---- auxiliary loss strengths ----
    # rollout_enabled/horizon/etc. remain fixed in base cfg for apples-to-apples.
    #"training.auxiliary_losses.rollout_weight": ("log", 5e-5, 1e-1),

    # semigroup.enabled / warmup_epochs also stay fixed in base cfg.
    #"training.auxiliary_losses.semigroup.weight": ("log", 1e-6, 3e-2),

    # ---- KL reg (only matters if model.vae_mode=True in base cfg) ----
    "training.beta_kl": ("log", 1e-5, 5e-2),
}


# -------------------- HELPER FUNCS --------------------

def _runtime_dtype_from_cfg(cfg: Dict[str, Any]) -> torch.dtype:
    """
    Map dataset.storage_dtype -> runtime dtype for dataset buffers.
    Mirrors main() logic.
    """
    storage_dtype_str = str(cfg.get("dataset", {}).get("storage_dtype", "float32")).lower()
    if storage_dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if storage_dtype_str in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _get(cfg: Dict[str, Any], dotted: str, default=None):
    """
    Safe dotted get, e.g. _get(cfg, "training.lr").
    """
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _set(cfg: Dict[str, Any], dotted: str, value):
    """
    Safe dotted set, creates nested dicts as needed.
    """
    parts = dotted.split(".")
    cur = cfg
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def _suggest(trial: optuna.Trial, name: str, space):
    """
    Interpret a SWEEP entry and call trial.suggest_* appropriately.
    """
    # Categorical
    if isinstance(space, list):
        return trial.suggest_categorical(name, space)

    # Uniform / Int or Float
    if isinstance(space, tuple) and len(space) == 2:
        lo, hi = space
        if isinstance(lo, int) and isinstance(hi, int):
            return trial.suggest_int(name, lo, hi)
        return trial.suggest_float(name, float(lo), float(hi))

    # Log-uniform float
    if isinstance(space, tuple) and len(space) == 3 and space[0] == "log":
        _, lo, hi = space
        return trial.suggest_float(name, float(lo), float(hi), log=True)

    raise ValueError(f"Unsupported sweep space for {name}: {space}")


def _apply_sweep(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample trial config from SWEEP, then apply cleanup rules:
      - Respect fixed curriculum (teacher forcing schedule, rollout warmups).
      - Clamp those schedules so they're valid for the trial's total epochs.
      - Gate beta_kl off if VAE mode is disabled.
      - Ensure val_batch_size is set.
    """
    cfg = copy.deepcopy(base_cfg)

    # 1) Sample hyperparams for this trial
    for key, space in SWEEP.items():
        if space is None:
            continue
        try:
            val = _suggest(trial, key, space)
            _set(cfg, key, val)
        except Exception as e:
            raise RuntimeError(f"Sampling {key} failed: {e}")

    # 2) beta_kl only matters if VAE mode is on
    vae_on = bool(_get(cfg, "model.vae_mode", False))
    if (not vae_on) and ("training.beta_kl" in SWEEP):
        # Reset to whatever base config had (often 0.0 for plain AE)
        _set(cfg, "training.beta_kl", _get(base_cfg, "training.beta_kl", 0.0))

    # 3) clamp teacher forcing schedule / rollout warmup
    total_epochs = int(_get(cfg, "training.epochs", 50))

    start_p = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", 1.0))
    end_p   = float(_get(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", 0.0))

    # enforce monotone decay (start >= end)
    if start_p < end_p:
        start_p, end_p = end_p, start_p

    # clamp probs into [0,1]
    start_p = max(0.0, min(1.0, start_p))
    end_p   = max(0.0, min(1.0, end_p))

    _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.start_p", start_p)
    _set(cfg, "training.auxiliary_losses.rollout_teacher_forcing.end_p", end_p)

    warmup_epochs = int(_get(cfg, "training.auxiliary_losses.rollout_warmup_epochs", 0))
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    _set(cfg, "training.auxiliary_losses.rollout_warmup_epochs", warmup_epochs)

    semi_warmup = int(_get(cfg, "training.auxiliary_losses.semigroup.warmup_epochs", 0))
    semi_warmup = max(0, min(semi_warmup, total_epochs))
    _set(cfg, "training.auxiliary_losses.semigroup.warmup_epochs", semi_warmup)

    # Ensure validation batch size falls back to train batch size if unset.
    bs_train = int(_get(cfg, "training.batch_size", 512))
    bs_val   = int(_get(cfg, "training.val_batch_size", bs_train))
    _set(cfg, "training.val_batch_size", bs_val)

    return cfg


def _build_trial_dataloaders(
    cfg_trial: Dict[str, Any],
    train_ds,
    val_ds,
    logger: logging.Logger,
):
    """
    Build train/val dataloaders for this trial using the trial's batch_size,
    num_workers, etc. Does NOT mutate the shared datasets.
    """
    dcfg = copy.deepcopy(cfg_trial.get("dataset", {}))
    tcfg = copy.deepcopy(cfg_trial.get("training", {}))

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
        shuffle=False,  # deterministic
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
    Trainer's key validation metric (optuna_monitor).
    Priority:
      1. env var TUNE_VAL_METRIC
      2. cfg["training"]["optuna_monitor"]
      3. fallback "val"
    """
    env_choice = os.getenv("TUNE_VAL_METRIC", "").strip()
    if env_choice:
        return env_choice

    cfg_choice = cfg.get("training", {}).get("optuna_monitor", "")
    if isinstance(cfg_choice, str) and cfg_choice.strip():
        return cfg_choice.strip()

    return "val"


def _prepare_accumulation(cfg_run: Dict[str, Any], train_loader) -> None:
    """
    Resolve accumulate_grad_batches="auto" to an int and stamp it into both:
      cfg_run["training"]["accumulate_grad_batches"]
      cfg_run["lightning"]["accumulate_grad_batches"]
    """
    default_accum = (
        1
        if (
            train_loader.batch_size is None
            or len(train_loader) == 0
            or train_loader.batch_size >= 1024
        )
        else 2
    )
    tcfg = cfg_run.setdefault("training", {})
    accum_raw = tcfg.get("accumulate_grad_batches", "auto")
    accum = _parse_int_or_auto(accum_raw, default_accum)
    tcfg["accumulate_grad_batches"] = accum
    cfg_run.setdefault("lightning", {})["accumulate_grad_batches"] = accum


def _best_so_far(study: optuna.study.Study) -> float | None:
    """
    Safely get the best value from completed trials, if any.
    Returns None if we don't have a finite best yet.
    """
    try:
        if not study.trials:
            return None
        val = float(study.best_trial.value)
        if math.isfinite(val):
            return val
        return None
    except Exception:
        # covers the case "no completed trials yet"
        return None


# -------------------- OBJECTIVE FACTORY --------------------

def _make_objective(
    base_cfg: Dict[str, Any],
    monitor_key: str,
    study_root: Path,
    device: torch.device,
    train_ds,
    val_ds,
    logger: logging.Logger,
    study: optuna.study.Study,
):
    """
    Two-stage objective:

      Stage A ("probe"):
        - Train for PROBE_EPOCHS (capped by full budget).
        - Measure best validation metric.
        - Log it.
        - Prune if it's clearly worse than the running best.

      Stage B ("full"):
        - Warm start only model weights from Stage A.
        - DO NOT restore optimizer/scheduler/epoch state.
        - Train to full epochs with a clean LR schedule.
        - Return best validation metric from the full run.

    Optuna will treat that return value as trial.value.
    """

    # We hold the best (lowest) probe score we've seen so far across trials.
    # Use a 1-element list so we can mutate it inside `objective`.
    best_probe_so_far = [None]  # type: list[float | None]

    def objective(trial: optuna.Trial) -> float:
        try:
            #
            # 1) Repro seeding + sample hyperparams
            #
            seed_everything(int(base_cfg["system"]["seed"]) + int(trial.number))
            cfg_trial_full = _apply_sweep(trial, base_cfg)

            #
            # 2) Per-trial work dir
            #
            trial_dir = study_root / f"trial_{trial.number:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            dump_json(trial_dir / "trial_config.sampled.json", cfg_trial_full)

            #
            # 3) Build loaders for this trial
            #
            trial_logger_data = logger.getChild(f"trial{trial.number:03d}.data")
            train_loader_full, val_loader_full = _build_trial_dataloaders(
                cfg_trial_full, train_ds, val_ds, trial_logger_data
            )

            # stamp accumulate_grad_batches so Trainer sees a concrete int
            _prepare_accumulation(cfg_trial_full, train_loader_full)

            #
            # 4) Build the full model now (will be warm-started later)
            #
            model_logger_full = logger.getChild(f"trial{trial.number:03d}.model.full")
            model_full = build_model(cfg_trial_full, device, model_logger_full)

            # remember the full epoch budget
            orig_epochs = int(_get(cfg_trial_full, "training.epochs", 50))

            #
            # ---------------- STAGE A: PROBE ----------------
            #
            cfg_probe = copy.deepcopy(cfg_trial_full)
            probe_epochs = min(PROBE_EPOCHS, orig_epochs)
            _set(cfg_probe, "training.epochs", probe_epochs)
            _set(cfg_probe, "lightning.resume_from", None)

            model_logger_probe = logger.getChild(f"trial{trial.number:03d}.model.probe")
            model_probe = build_model(cfg_probe, device, model_logger_probe)

            run_probe = Trainer(
                model=model_probe,
                train_loader=train_loader_full,
                val_loader=val_loader_full,
                cfg=cfg_probe,
                work_dir=trial_dir,
                device=device,
                logger=logger.getChild(f"trial{trial.number:03d}.trainer.probe"),
                optuna_trial=None,          # we do our own pruning after probe
                optuna_monitor=monitor_key,
            )

            # best validation metric from the short probe run
            best_val_probe = float(run_probe.train())

            dump_json(trial_dir / "trial_config.probe.json", cfg_probe)
            trial.set_user_attr("best_val_probe", best_val_probe)
            trial.report(best_val_probe, step=probe_epochs)

            #
            # Log heartbeat: probe score, current bar, prune threshold
            #
            bp_current = best_probe_so_far[0]
            if bp_current is None:
                prune_threshold_str = "N/A"
                best_probe_str = "inf"
            else:
                prune_threshold_str = f"{bp_current * PRUNE_FACTOR:.4e}"
                best_probe_str = f"{bp_current:.4e}"

            logger.info(
                f"[trial {trial.number:03d}] "
                f"probe_val={best_val_probe:.4e} | "
                f"best_probe_so_far={best_probe_str} | "
                f"prune_threshold={prune_threshold_str}"
            )

            #
            # ---------------- PRUNE DECISION ----------------
            #
            if bp_current is None:
                # first good probe establishes the bar
                best_probe_so_far[0] = best_val_probe
            else:
                threshold = bp_current * PRUNE_FACTOR
                # prune if bad or non-finite
                if (not math.isfinite(best_val_probe)) or (best_val_probe > threshold):
                    raise optuna.TrialPruned(
                        f"Probe {best_val_probe:.4g} is not competitive vs "
                        f"best_probe_so_far {bp_current:.4g} "
                        f"(threshold {threshold:.4g})"
                    )
                # tighten the bar if we beat it
                if math.isfinite(best_val_probe) and (best_val_probe < bp_current):
                    best_probe_so_far[0] = best_val_probe

            #
            # ---------------- STAGE B: FULL ----------------
            #
            # Warm-start model_full with just the weights from Stage A.
            # We do NOT resume optimizer/scheduler state, so LR schedule is clean.
            #
            best_model_pt = trial_dir / "best_model.pt"
            if best_model_pt.exists():
                try:
                    state = torch.load(best_model_pt, map_location="cpu")
                    sd = state.get("model", {})
                    missing, unexpected = model_full.load_state_dict(sd, strict=False)
                    if missing or unexpected:
                        logger.warning(
                            f"[trial {trial.number:03d}] Warm start: "
                            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
                        )
                except Exception as e:
                    logger.warning(
                        f"[trial {trial.number:03d}] Warm start failed: {e}"
                    )

            # Force clean Stage B run
            _set(cfg_trial_full, "lightning.resume_from", None)
            _set(cfg_trial_full, "training.epochs", orig_epochs)

            run_full = Trainer(
                model=model_full,
                train_loader=train_loader_full,
                val_loader=val_loader_full,
                cfg=cfg_trial_full,
                work_dir=trial_dir,
                device=device,
                logger=logger.getChild(f"trial{trial.number:03d}.trainer.full"),
                optuna_trial=trial,          # optional pruning callback (can be silenced if annoying)
                optuna_monitor=monitor_key,
            )

            best_val_full = float(run_full.train())

            dump_json(trial_dir / "trial_config.final.json", cfg_trial_full)
            trial.set_user_attr("best_val_full", best_val_full)

            # Optuna's objective value
            return best_val_full

        except optuna.TrialPruned:
            # Signal Optuna that this trial was intentionally pruned, not crashed.
            raise
        except Exception as e:
            # Hard failure: return +inf so Optuna treats it as terrible.
            logger.error(f"Trial {trial.number} crashed: {e}", exc_info=True)
            return float("inf")

    return objective


# -------------------- MAIN --------------------

def main():
    setup_logging(None)
    logger = logging.getLogger("tune")

    # Silence Lightning / Fabric "rank_zero_info" and "rank_zero_warn" noise
    for name in (
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "lightning.pytorch.utilities.rank_zero",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    # ----- load + seed -----
    cfg = load_json_config(str(CONFIG_PATH))
    cfg.setdefault("system", {})["seed"] = int(cfg["system"].get("seed", 42))
    seed_everything(int(cfg["system"]["seed"]))

    # ----- work_dir base -----
    paths = cfg.setdefault("paths", {})
    base = Path(paths.get("work_dir", GLOBAL_WORK_DIR)).expanduser().resolve()
    paths["work_dir"] = str(base)
    base.mkdir(parents=True, exist_ok=True)

    # study root dir
    study_root = base / "tuning" / STUDY_NAME
    study_root.mkdir(parents=True, exist_ok=True)

    # ----- bail if study already exists -----
    storage_path = study_root / "optuna_study.db"
    if storage_path.exists():
        logger.error(
            "Refusing to run because study already exists at %s. "
            "This prevents mixing old trials with a new sweep definition.",
            storage_path,
        )
        raise SystemExit(1)

    # ----- hardware/runtime setup -----
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device)
    runtime_dtype = _runtime_dtype_from_cfg(cfg)

    # ----- data prep -----
    processed_dir = ensure_preprocessed_data(cfg, logger.getChild("pre"))
    assert processed_dir.exists(), "Processed data directory must exist."

    # snapshot hydrated cfg for reproducibility
    dump_json(study_root / "config.hydrated.json", cfg)

    # Build datasets ONCE; reuse for all trials
    train_ds, val_ds, _, _ = build_datasets_and_loaders(
        cfg, device, runtime_dtype, logger.getChild("data.init")
    )

    # ----- metric selection -----
    monitor_key = _resolve_monitor_key(cfg)

    # ----- Optuna study setup -----
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=False,  # we just bailed if it exists
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["system"]["seed"])),
        # Mild built-in pruner. Real cutoff happens in our probe logic.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1,
            n_warmup_steps=1,
        ),
    )

    objective = _make_objective(
        base_cfg=cfg,
        monitor_key=monitor_key,
        study_root=study_root,
        device=device,
        train_ds=train_ds,
        val_ds=val_ds,
        logger=logger,
        study=study,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # ----- dump best config -----
    best = study.best_trial
    logging.getLogger().info(
        f"BEST TRIAL #{best.number} loss={best.value:.6e} params={best.params}"
    )
    src = study_root / f"trial_{best.number:03d}" / "trial_config.final.json"
    if src.exists():
        shutil.copy(src, study_root / "best_config.json")


if __name__ == "__main__":
    main()
