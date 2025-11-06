#!/usr/bin/env python3
"""
tune.py — Optuna hyperparameter search (globals-only)

- No argparse. All configuration lives in the GLOBALS section below.
- Comment out any line with a `trial.suggest_*` to freeze that param.
- Head modes are mutually exclusive: {delta, delta_log_phys, simplex(softmax)}.
- Work around macOS/mp_spawn pickling of lambda collate_fn by forcing num_workers=0.
"""

from __future__ import annotations

# =============================== GLOBALS =====================================
# Paths / run control
CONFIG_PATH: str      = "config/config.jsonc"   # Base config (JSON/JSONC)
WORK_BASE_DIR: str    = "models/tuning"         # Trials write under this directory
STUDY_NAME: str       = "optuna_tune"
STORAGE_URL: str      = ""                      # e.g., "sqlite:///tune.db" (empty => in-memory)

# Tuning budget
N_TRIALS: int         = 100
EPOCHS_PER_TRIAL: int = 25

# Seeds
BASE_SEED: int        = 42

# Search menus
LATENTS          = [128, 256, 512]
ENC_BASE_CHOICES = [256, 512, 1024]
DYN_BASE_CHOICES = [256, 512, 1024]
DEC_BASE_CHOICES = [256, 512, 1024]
ACTIVATIONS      = ["silu", "tanh", "elu", "leakyrelu"]

# =============================== IMPORTS =====================================
import copy
import logging
import json                       # ← added
from shutil import copy2          # ← added
from pathlib import Path
from typing import Any, Dict, List

import optuna

from utils import (
    setup_logging,
    seed_everything,
    load_json_config,
    dump_json,
    resolve_precision_and_dtype,
)
from hardware import setup_device, optimize_hardware
from main import (
    ensure_preprocessed_data,
    hydrate_config_from_processed,
    build_datasets_and_loaders,
    validate_dataloaders,
    build_model,
)
from trainer import Trainer

# ================================ HELPERS ====================================

def _make_shape(base: int, depth: int, shape: str, *, floor: int = 32) -> List[int]:
    """
    Generate sensible MLP widths for AE/dynamics with simple shape grammars.
    """
    depth = max(1, int(depth))
    base = int(base)
    if shape == "flat" or depth == 1:
        return [max(floor, base)] * depth
    if shape == "pyramid":
        return [max(floor, base // (2 ** i)) for i in range(depth)]
    if shape == "inverse_pyramid":
        return [max(floor, base // (2 ** (depth - 1 - i))) for i in range(depth)]
    if shape == "diamond":
        half = depth // 2
        up   = [max(floor, base // (2 ** (half - i))) for i in range(half)]
        mid  = [max(floor, base)]
        down = [max(floor, base // (2 ** (i + 1))) for i in range(depth - half - 1)]
        return up + mid + down
    return [max(floor, base)] * depth


def _apply_trial_cfg(cfg: Dict[str, Any], trial: optuna.Trial, *, trial_epochs: int) -> Dict[str, Any]:
    """
    Deep-copy cfg and inject trial-specific hyperparameters.

    Comment out any `trial.suggest_*` line below to stop searching that parameter.
    """
    c = copy.deepcopy(cfg)

    # ------------------------------- MODEL ------------------------------------
    m = c.setdefault("model", {})

    # Latent size
    m["latent_dim"] = trial.suggest_categorical("latent_dim", LATENTS)  # ← comment to freeze

    # Encoder hidden
    enc_depth = trial.suggest_int("enc_depth", 2, 5)                     # ← comment to freeze
    enc_base  = trial.suggest_categorical("enc_base", ENC_BASE_CHOICES)  # ← comment to freeze
    enc_shape = trial.suggest_categorical("enc_shape", ["flat", "pyramid", "diamond", "inverse_pyramid"])
    m["encoder_hidden"] = _make_shape(enc_base, enc_depth, enc_shape, floor=max(32, m["latent_dim"] // 2))

    # Dynamics hidden
    dyn_depth = trial.suggest_int("dyn_depth", 2, 5)                     # ← comment to freeze
    dyn_base  = trial.suggest_categorical("dyn_base", DYN_BASE_CHOICES)  # ← comment to freeze
    dyn_shape = trial.suggest_categorical("dyn_shape", ["flat", "pyramid", "diamond"])
    m["dynamics_hidden"] = _make_shape(dyn_base, dyn_depth, dyn_shape, floor=max(32, m["latent_dim"] // 2))

    # Decoder hidden
    dec_depth = trial.suggest_int("dec_depth", 2, 5)                     # ← comment to freeze
    dec_base  = trial.suggest_categorical("dec_base", DEC_BASE_CHOICES)  # ← comment to freeze
    dec_shape = trial.suggest_categorical("dec_shape", ["flat", "pyramid", "diamond", "inverse_pyramid"])
    m["decoder_hidden"] = _make_shape(dec_base, dec_depth, dec_shape, floor=max(32, m["latent_dim"] // 2))

    # Nonlinearity + dropout
    m["activation"] = trial.suggest_categorical("activation", ACTIVATIONS)   # ← comment to freeze
    m["dropout"]    = trial.suggest_float("dropout", 0.00, 0.30)             # ← comment to freeze

    # Dynamics residual
    m["dynamics_residual"] = trial.suggest_categorical("dynamics_residual", [False, True])  # ← comment to freeze

    # Head (mutually exclusive): {"delta","delta_log_phys","simplex"} → sets flags
    head_mode = trial.suggest_categorical("head_mode", ["delta", "delta_log_phys", "simplex"])  # ← comment to freeze
    if head_mode == "delta":
        m["predict_delta"] = True
        m["predict_delta_log_phys"] = False
        m["softmax_head"] = False
    elif head_mode == "delta_log_phys":
        m["predict_delta"] = False
        m["predict_delta_log_phys"] = True
        m["softmax_head"] = False
    else:
        m["predict_delta"] = False
        m["predict_delta_log_phys"] = False
        m["softmax_head"] = True
        # Safety with S_out != S_in
        m["allow_partial_simplex"] = True

    # ------------------------------ DATASET -----------------------------------
    d = c.setdefault("dataset", {})
    d["uniform_offset_sampling_strict"] = trial.suggest_categorical(
        "uniform_offset_sampling_strict", [False, True]
    )  # ← comment to freeze

    # Keep dataset fully on GPU for efficiency; disable pinning (pinning is for CPU→GPU copies)
    d["preload_to_gpu"] = True
    d["pin_memory"] = False

    # Avoid multiprocessing issues with lambda collate_fn and GPU-resident tensors
    d["num_workers"] = 0
    d["persistent_workers"] = False
    d.setdefault("prefetch_factor", 2)

    # ------------------------------ TRAINING ----------------------------------
    t = c.setdefault("training", {})
    t["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)  # ← comment to freeze
    t["beta_kl"]      = trial.suggest_float("beta_kl", 0.0, 1.0)                   # ← comment to freeze
    t["epochs"] = int(trial_epochs)

    # ------------------------------- SYSTEM -----------------------------------
    sys_cfg = c.setdefault("system", {})
    sys_cfg["seed"] = int(sys_cfg.get("seed", BASE_SEED)) + int(trial.number)

    return c


def _objective(trial: optuna.Trial, base_cfg: Dict[str, Any], work_root: Path, trial_epochs: int, study_name: str) -> float:
    # Per-trial working directory
    work_dir = work_root / study_name / f"trial_{trial.number:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    setup_logging(log_file=work_dir / "tune.log", level=logging.INFO)
    log = logging.getLogger("tune")
    log.info("Trial %d — work_dir=%s", trial.number, work_dir)

    # Build cfg for this trial
    cfg = _apply_trial_cfg(base_cfg, trial, trial_epochs=trial_epochs)
    cfg.setdefault("paths", {})
    cfg["paths"]["work_dir"] = str(work_dir)
    cfg["paths"]["overwrite"] = True  # isolate trials

    # Hardware and precision
    seed_everything(int(cfg.get("system", {}).get("seed", BASE_SEED)))
    device = setup_device()
    optimize_hardware(cfg.get("system", {}), device, log)
    _, runtime_dtype = resolve_precision_and_dtype(cfg, device, log)

    # Ensure processed data + hydrate cfg from artifacts
    processed_dir = ensure_preprocessed_data(cfg, log)
    hydrate_config_from_processed(cfg, log, processed_dir=processed_dir)

    # Save hydrated config for reproducibility
    dump_json(work_dir / "config.hydrated.json", cfg)

    # Data
    train_ds, val_ds, train_loader, val_loader = build_datasets_and_loaders(
        cfg=cfg, device=device, runtime_dtype=runtime_dtype, logger=log
    )
    validate_dataloaders(train_loader=train_loader, val_loader=val_loader, logger=log)

    # Model
    model = build_model(cfg, log)

    # Train and return best validation loss
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=log.getChild("trainer"),
    )
    best_val = float(trainer.train())
    log.info("Trial %d best val: %.6e", trial.number, best_val)

    trial.report(best_val, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_val


def _best_config_saver(work_root: Path, study_name: str):
    """
    Optuna callback: when a trial becomes the study best, copy its hydrated config
    and write a small manifest under <WORK_BASE_DIR>/<STUDY_NAME>/best/.
    """
    best_dir = (work_root / study_name / "best")
    best_dir.mkdir(parents=True, exist_ok=True)

    def _cb(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_trial
        if frozen_trial.number != best.number:
            return
        src = work_root / study_name / f"trial_{best.number:04d}" / "config.hydrated.json"
        if src.exists():
            copy2(src, best_dir / "config.hydrated.json")
        with open(best_dir / "best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2, sort_keys=True)
        with open(best_dir / "best.txt", "w") as f:
            f.write(f"trial={best.number}\nvalue={best.value}\npath={src.parent}\n")

    return _cb


def run() -> None:
    cfg_path = Path(CONFIG_PATH).expanduser().resolve()
    base_cfg = load_json_config(cfg_path)

    work_root = Path(WORK_BASE_DIR).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=(STORAGE_URL or None),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(base_cfg.get("system", {}).get("seed", BASE_SEED))),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0),
    )

    objective = lambda trial: _objective(  # noqa: E731
        trial=trial,
        base_cfg=base_cfg,
        work_root=work_root,
        trial_epochs=EPOCHS_PER_TRIAL,
        study_name=STUDY_NAME,
    )
    cb = _best_config_saver(work_root, STUDY_NAME)  # ← added
    study.optimize(objective, n_trials=int(N_TRIALS), gc_after_trial=True, callbacks=[cb])  # ← modified

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    run()
