#!/usr/bin/env python3
"""
tune.py — Optuna hyperparameter search
"""

from __future__ import annotations

# =============================== GLOBALS =====================================
# Paths / run control
CONFIG_PATH: str      = "config/config.jsonc"   # Base config (JSON/JSONC)
WORK_BASE_DIR: str    = "models/tuning"         # Trials write under this directory
STUDY_NAME: str       = "optuna_tune"
STORAGE_URL: str      = ""                      # e.g., "sqlite:///tune.db" (empty => in-memory)

# Tuning budget
N_TRIALS: int          = 500
EPOCHS_PER_TRIAL: int  = 25
EARLY_GATE_EPOCHS: int = 10
EARLY_GATE_FACTOR: float = 1.10   # prune if val > 1.10 × global_best

# Seeds
BASE_SEED: int        = 42

# Search menus (dimensions + activations)
LATENTS          = [128, 256, 512]
ENC_BASE_CHOICES = [256, 512, 1024]
DYN_BASE_CHOICES = [256, 512, 1024]
DEC_BASE_CHOICES = [256, 512, 1024]
ACTIVATIONS      = ["silu", "leakyrelu", "elu", "tanh"]

# =============================== IMPORTS =====================================
import os, warnings, logging, csv, shutil, json, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # silence TF/XLA if present
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"     # suppress C++/c10 INFO/WARN
os.environ["NCCL_DEBUG"] = "WARN"               # avoid NCCL info spam

os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_LOGS_REGEX", None)   # harmless if absent
import optuna

from utils import (
    setup_logging,
    seed_everything,
    load_json_config,
    dump_json,
    resolve_precision_and_dtype,
)
from main import (
    ensure_preprocessed_data,
    hydrate_config_from_processed,
    build_datasets_and_loaders,
    validate_dataloaders,
    build_model,
)
from trainer import Trainer

# Global container for best epoch val across trials (per run)
_GLOBAL_E_BEST: Dict[str, float] = {"value": float("inf")}


# ================================ LOGGING ====================================
def _quiet_external_logs() -> None:
    """Silence noisy stacks from third parties; keep our logs and epoch prints visible."""
    for name in (
        "torch", "torch._dynamo", "torch._inductor", "torch.distributed",
        "numba", "urllib3", "matplotlib",
        # Keep Lightning at WARNING so its epoch table can still print if it uses logging
        "pytorch_lightning", "lightning", "lightning.pytorch",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ================================ HELPERS ====================================
def _make_shape(base: int, depth: int, shape: str, *, floor: int = 32) -> List[int]:
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
    Active search limited: activation, beta_kl, model dimensions.
    Extra knobs are present but commented; uncomment to include in search.
    """
    import copy
    c = copy.deepcopy(cfg)

    # ------------------------------- MODEL ------------------------------------
    m = c.setdefault("model", {})

    # Latent size (TUNED)
    m["latent_dim"] = trial.suggest_categorical("latent_dim", LATENTS)

    # Encoder hidden (TUNED)
    enc_depth = trial.suggest_int("enc_depth", 2, 4)
    enc_base  = trial.suggest_categorical("enc_base", ENC_BASE_CHOICES)
    enc_shape = trial.suggest_categorical("enc_shape", ["flat", "pyramid", "diamond", "inverse_pyramid"])
    m["encoder_hidden"] = _make_shape(enc_base, enc_depth, enc_shape, floor=max(32, m["latent_dim"] // 2))

    # Dynamics hidden (TUNED)
    dyn_depth = trial.suggest_int("dyn_depth", 2, 5)
    dyn_base  = trial.suggest_categorical("dyn_base", DYN_BASE_CHOICES)
    dyn_shape = trial.suggest_categorical("dyn_shape", ["flat", "pyramid", "diamond"])
    m["dynamics_hidden"] = _make_shape(dyn_base, dyn_depth, dyn_shape, floor=max(32, m["latent_dim"] // 2))

    # Decoder hidden (TUNED)
    dec_depth = trial.suggest_int("dec_depth", 2, 4)
    dec_base  = trial.suggest_categorical("dec_base", DEC_BASE_CHOICES)
    dec_shape = trial.suggest_categorical("dec_shape", ["flat", "pyramid", "diamond", "inverse_pyramid"])
    m["decoder_hidden"] = _make_shape(dec_base, dec_depth, dec_shape, floor=max(32, m["latent_dim"] // 2))

    # Nonlinearity (TUNED)
    m["activation"] = trial.suggest_categorical("activation", ACTIVATIONS)

    # ---- OPTIONAL / COMMENTED knobs (UNCOMMENT to search) --------------------
    # m["dropout"] = trial.suggest_float("dropout", 0.00, 0.30)
    m["dynamics_residual"] = trial.suggest_categorical("dynamics_residual", [False, True])
    m["residual_decoder"]  = trial.suggest_categorical("residual_decoder",  [False, True])
    m["head_type"] = trial.suggest_categorical("head_type", ["delta", "delta_log_phys", "softmax"])
    if m.get("head_type") in ("simplex", "softmax"):
         pass

    # ------------------------------ DATASET -----------------------------------
    d = c.setdefault("dataset", {})
    d["preload_to_gpu"] = True
    d["pin_memory"] = False
    d["num_workers"] = 0
    d["persistent_workers"] = False
    d.setdefault("prefetch_factor", 2)
    d["uniform_offset_sampling_strict"] = trial.suggest_categorical(
         "uniform_offset_sampling_strict", [False, True]
    )

    # ------------------------------ TRAINING ----------------------------------
    t = c.setdefault("training", {})
    t["beta_kl"] = trial.suggest_float("beta_kl", 1e-4, 0.1, log=True)
    t["epochs"] = int(trial_epochs)
    t.setdefault("precision", "bf16-mixed")
    # t["kl_schedule"] = trial.suggest_categorical("kl_schedule", ["none", "linear", "cosine"])
    # t["kl_warmup_epochs"] = trial.suggest_int("kl_warmup_epochs", 0, 20)

    # ----------------------------- OPTIMIZER ----------------------------------
    opt = c.setdefault("optimizer", {})
    opt.setdefault("name", "AdamW")
    # opt["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # ------------------------------- SYSTEM -----------------------------------
    sys_cfg = c.setdefault("system", {})
    sys_cfg["seed"] = int(sys_cfg.get("seed", BASE_SEED)) + int(trial.number)

    return c


def _find_metrics_csv(root: Path) -> Optional[Path]:
    candidates = [
        root / "lightning_logs" / "version_0" / "metrics.csv",
        root / "metrics.csv",
        root / "logs" / "metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fall back: first found in subtree
    for p in root.rglob("metrics.csv"):
        return p
    return None


def _read_val_at_epoch(metrics_csv: Path, epoch_index_zero_based: int) -> Optional[float]:
    if metrics_csv is None or not metrics_csv.exists():
        return None
    target_epoch = epoch_index_zero_based
    last_match: Optional[float] = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "epoch" not in row:
                continue
            try:
                e = int(float(row["epoch"]))
            except Exception:
                continue
            if e == target_epoch:
                for key in ("val_loss", "val", "valid_loss", "val_metric"):
                    if key in row and row[key] not in ("", "nan", "None"):
                        try:
                            last_match = float(row[key])
                        except Exception:
                            pass
    return last_match

class _EpochTailer:
    def __init__(self, run_dir: Path, tag: str):
        self.run_dir = Path(run_dir)
        self.tag = tag  # e.g., "s1" or "s2"
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._printed_epochs = set()
        self._best_so_far = float("inf")

    def start(self) -> None:
        self._t.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._t.join(timeout=timeout)

    def _locate_metrics(self) -> Optional[Path]:
        return _find_metrics_csv(self.run_dir)

    def _run(self) -> None:
        metrics_path: Optional[Path] = None
        last_size = 0
        while not self._stop.is_set():
            if metrics_path is None or not metrics_path.exists():
                metrics_path = self._locate_metrics()
                time.sleep(0.2)
                continue
            try:
                size = metrics_path.stat().st_size
                if size == last_size:
                    time.sleep(0.2)
                    continue
                last_size = size
                # Parse full file (cheap) and emit lines only for unseen epochs.
                with metrics_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    latest_by_epoch: Dict[int, Dict[str, str]] = {}
                    for row in reader:
                        if "epoch" not in row:
                            continue
                        try:
                            e = int(float(row["epoch"]))
                        except Exception:
                            continue
                        latest_by_epoch[e] = row  # keep last row per epoch
                    for e in sorted(latest_by_epoch.keys()):
                        if e in self._printed_epochs:
                            continue
                        row = latest_by_epoch[e]
                        # helpers
                        def _getf(*names: str) -> Optional[float]:
                            for n in names:
                                if n in row and row[n] not in ("", "nan", "None"):
                                    try:
                                        return float(row[n])
                                    except Exception:
                                        pass
                            return None
                        v = _getf("val_loss", "val", "valid_loss", "val_metric")
                        tr = _getf("train_loss_epoch", "train_loss", "train")
                        lr = _getf("lr")
                        if v is not None and v < self._best_so_far:
                            self._best_so_far = v
                        parts = [f"E{e:04d}"]
                        parts.append(f"|   train={tr: .3e}" if tr is not None else "|   train=   n/a")
                        parts.append(f"|     val={v: .3e}" if v is not None else "|     val=   n/a")
                        parts.append(f"|     best={self._best_so_far: .3e}")
                        if lr is not None:
                            parts.append(f"|      lr={lr: .2e}")
                        print(f"[{self.tag}] " + " ".join(parts), flush=True)
                        self._printed_epochs.add(e)
            except Exception:
                time.sleep(0.1)


# ------------------------------ train one run --------------------------------
def _train_once(
    cfg: Dict[str, Any],
    work_dir: Path,
    device_log: logging.Logger,
    tag: str = "s1",
) -> Tuple[float, Optional[float]]:
    import torch

    setup_logging(log_file=work_dir / "tune.log", level=logging.INFO)
    log = logging.getLogger("tune.trainer")
    _quiet_external_logs()

    # Hardware + precision
    seed_everything(int(cfg.get("system", {}).get("seed", BASE_SEED)))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Resolve precision/dtype with a proper device
    _, runtime_dtype = resolve_precision_and_dtype(cfg, device, log)

    # Ensure processed data + hydrate cfg
    processed_dir = ensure_preprocessed_data(cfg, log)
    hydrate_config_from_processed(cfg, log, processed_dir=processed_dir)

    # Data
    train_ds, val_ds, train_loader, val_loader = build_datasets_and_loaders(
        cfg=cfg, device=device, runtime_dtype=runtime_dtype, logger=log
    )
    validate_dataloaders(train_loader=train_loader, val_loader=val_loader, logger=log)

    # Model
    model = build_model(cfg, log)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        work_dir=work_dir,
        device=device,
        logger=log.getChild(tag),
    )
    best_val = float(trainer.train())

    # Read val from metrics
    metrics_csv = _find_metrics_csv(work_dir)
    v = _read_val_at_epoch(metrics_csv, epoch_index_zero_based=EARLY_GATE_EPOCHS - 1)
    return best_val, v

def _objective(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    work_root: Path,
    trial_epochs: int,
    study_name: str,
) -> float:
    # Per-trial working directory
    trial_root = work_root / study_name / f"trial_{trial.number:04d}"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    # Logger for the trial
    setup_logging(log_file=trial_root / "tune.log", level=logging.INFO)
    log = logging.getLogger("tune")
    log.info("Trial %d — work_dir=%s", trial.number, trial_root)

    # ---------- Phase A: short gate run to EARLY_GATE_EPOCHS ----------
    cfg_short = _apply_trial_cfg(base_cfg, trial, trial_epochs=EARLY_GATE_EPOCHS)
    cfg_short.setdefault("paths", {})
    cfg_short["paths"]["work_dir"] = str(trial_root / "s1")
    cfg_short["paths"]["overwrite"] = True

    _, v = _train_once(cfg_short, trial_root / "s1", device_log=log, tag="s1")
    if v is None:
        log.info("Early gate — val missing; allowing trial to continue.")
    else:
        e_best = _GLOBAL_E_BEST["value"]
        if e_best < float("inf"):
            threshold = e_best * EARLY_GATE_FACTOR
            log.info("Early gate — gb=%.6e, cur=%.6e, thr=%.6e", e_best, v, threshold)
            if v > threshold:
                log.info("Pruning trial %d after %d epochs (val=%.6e > %.6e).",
                         trial.number, EARLY_GATE_EPOCHS, v, threshold)
                raise optuna.TrialPruned(f"Early prune: val={v:.4g} > {EARLY_GATE_FACTOR:.2f}×gb={threshold:.4g}")
        # Update global best
        if v < _GLOBAL_E_BEST["value"]:
            _GLOBAL_E_BEST["value"] = float(v)

    cfg_full = _apply_trial_cfg(base_cfg, trial, trial_epochs=trial_epochs)
    cfg_full.setdefault("paths", {})
    cfg_full["paths"]["work_dir"] = str(trial_root / "s2")
    cfg_full["paths"]["overwrite"] = True

    best_val, _ = _train_once(cfg_full, trial_root / "s2", device_log=log, tag="s2")

    # Persist hydrated config + params for this trial
    dump_json(trial_root / "config.hydrated.json", cfg_full)
    with open(trial_root / "params.json", "w") as f:
        json.dump(trial.params, f, indent=2, sort_keys=True)

    log.info("Trial %d best val: %.6e", trial.number, best_val)
    return float(best_val)

def _best_config_saver(work_root: Path, study_name: str):
    """
    On new study best, copy hydrated config + params to <WORK_BASE_DIR>/<STUDY_NAME>/best/.
    """
    best_dir = (work_root / study_name / "best")
    best_dir.mkdir(parents=True, exist_ok=True)

    def _cb(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
        try:
            best = study.best_trial
        except Exception:
            return
        if frozen_trial.number != best.number:
            return
        src = work_root / study_name / f"trial_{best.number:04d}" / "s2"
        src_cfg = src / "config.hydrated.json"
        if src_cfg.exists():
            shutil.copy2(src_cfg, best_dir / "config.hydrated.json")
        with open(best_dir / "best_params.json", "w") as f:
            json.dump(best.params, f, indent=2, sort_keys=True)
        with open(best_dir / "best.txt", "w") as f:
            f.write(f"trial={best.number}\nvalue={best.value}\npath={src}\n")

    return _cb


# ================================== RUN ======================================
def run() -> None:
    cfg_path = Path(CONFIG_PATH).expanduser().resolve()
    base_cfg = load_json_config(cfg_path)

    work_root = Path(WORK_BASE_DIR).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # Study with no-op pruner (we do our own early gate)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=(STORAGE_URL or None),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=int(base_cfg.get("system", {}).get("seed", BASE_SEED)),
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    objective = lambda trial: _objective(
        trial=trial,
        base_cfg=base_cfg,
        work_root=work_root,
        trial_epochs=EPOCHS_PER_TRIAL,
        study_name=STUDY_NAME,
    )
    cb = _best_config_saver(work_root, STUDY_NAME)
    study.optimize(objective, n_trials=int(N_TRIALS), gc_after_trial=True, callbacks=[cb])

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    run()
