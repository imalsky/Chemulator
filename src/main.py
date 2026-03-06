#!/usr/bin/env python3
"""
main.py - Strict entrypoint for training.

Principles (by design):
- No alias keys. Config must use the canonical schema.
- No silent fallbacks (no "search for a checkpoint", no "use CPU if CUDA missing", etc.).
- Errors are concise and deterministic.

This script:
1) Loads config JSON.
2) Resolves relative paths relative to the config file directory.
3) Loads normalization manifest (processed_dir/normalization.json) and checks config consistency.
4) Builds datasets/dataloaders sized for the rollouts actually used.
5) Builds model strictly from model.type.
6) Runs Lightning training using explicit checkpoint behavior.

Notes:
- Device selection here is ONLY for dataset preloading (dataset.preload_to_device).
  Lightning decides training device(s) based on cfg.runtime.
"""

from __future__ import annotations

import logging
import os
import warnings

# Avoid MKL/OpenMP duplicate symbol aborts in some environments.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from lightning.pytorch import seed_everything

from dataset import FlowMapRolloutDataset, create_dataloader
from model import create_model
from trainer import FlowMapRolloutModule, build_lightning_trainer
from utils import PrecisionConfig, atomic_write_json, ensure_dir, load_json_config, parse_precision_config

# Prefer fast matmul where supported.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

# Strict required config keys so saved config.resolved.json is always complete for debugging.
_REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
    "precision.compute_dtype",
    "precision.amp_mode",
    "precision.model_dtype",
    "precision.input_dtype",
    "precision.dataset_dtype",
    "precision.preload_dtype",
    "precision.loss_dtype",
    "paths.raw_dir",
    "paths.processed_dir",
    "paths.work_dir",
    "normalization.epsilon",
    "normalization.min_std",
    "normalization.globals_default_method",
    "normalization.methods",
    "preprocessing.raw_file_patterns",
    "preprocessing.dt_min",
    "preprocessing.dt_max",
    "preprocessing.dt_sampling",
    "preprocessing.n_steps",
    "preprocessing.t_min",
    "preprocessing.output_trajectories_per_file",
    "preprocessing.shard_size",
    "preprocessing.overwrite",
    "preprocessing.time_key",
    "preprocessing.val_fraction",
    "preprocessing.test_fraction",
    "preprocessing.seed",
    "preprocessing.pool_size",
    "preprocessing.samples_per_source_trajectory",
    "preprocessing.max_chunk_attempts_per_source",
    "preprocessing.drop_below",
    "system.device",
    "system.log_level",
    "system.seed",
    "runtime.checkpoint",
    "runtime.load_weights_strict",
    "runtime.accelerator",
    "runtime.devices",
    "runtime.strategy",
    "runtime.accumulate_grad_batches",
    "runtime.deterministic",
    "runtime.enable_progress_bar",
    "runtime.gradient_clip_val",
    "runtime.log_every_n_steps",
    "runtime.checkpointing.enabled",
    "runtime.checkpointing.every_n_epochs",
    "runtime.checkpointing.monitor",
    "runtime.checkpointing.save_top_k",
    "runtime.checkpointing.save_last",
    "runtime.torch_compile.enabled",
    "runtime.torch_compile.backend",
    "runtime.torch_compile.mode",
    "runtime.torch_compile.dynamic",
    "runtime.torch_compile.fullgraph",
    "runtime.torch_compile.compile_forward_step",
    "runtime.torch_compile.compile_open_loop_unroll",
    "data.global_variables",
    "data.species_variables",
    "dataset.windows_per_trajectory",
    "dataset.preload_to_device",
    "dataset.shard_cache_size",
    "model.type",
    "model.activation",
    "model.dropout",
    "model.layer_norm",
    "model.layer_norm_eps",
    "model.predict_delta",
    "model.mlp.hidden_dims",
    "model.mlp.residual",
    "model.autoencoder.latent_dim",
    "model.autoencoder.encoder_hidden",
    "model.autoencoder.decoder_hidden",
    "model.autoencoder.dynamics_hidden",
    "model.autoencoder.residual",
    "model.autoencoder.dynamics_residual",
    "training.batch_size",
    "training.max_epochs",
    "training.checkpoint_mode",
    "training.num_workers",
    "training.pin_memory",
    "training.persistent_workers",
    "training.prefetch_factor",
    "training.rollout_steps",
    "training.loss.lambda_log10_mae",
    "training.loss.lambda_z_mse",
    "training.optimizer.name",
    "training.optimizer.lr",
    "training.optimizer.weight_decay",
    "training.optimizer.exclude_norm_and_bias_from_weight_decay",
    "training.optimizer.betas",
    "training.optimizer.eps",
    "training.optimizer.fused",
    "training.optimizer.foreach",
    "training.scheduler.enabled",
    "training.scheduler.type",
    "training.scheduler.warmup_epochs",
    "training.autoregressive_training.enabled",
    "training.autoregressive_training.skip_steps",
    "training.autoregressive_training.detach_between_steps",
    "training.autoregressive_training.backward_per_step",
    "training.curriculum.enabled",
    "training.curriculum.start_steps",
    "training.curriculum.end_steps",
    "training.curriculum.mode",
    "training.curriculum.ramp_epochs",
)

# Optional keys accepted by schema validation.
_OPTIONAL_CONFIG_KEYS: Tuple[str, ...] = (
    # Kept as an explicitly-recognized unsupported key so we can emit a targeted error later.
    "runtime.mode",
    # Optional perf knob for manual-optimization training path (default in code if absent).
    "runtime.log_grad_norm",
    # Scheduler-specific keys are validated conditionally from training.scheduler.type.
    "training.scheduler.min_lr_ratio",
    "training.scheduler.factor",
    "training.scheduler.patience",
    "training.scheduler.threshold",
    "training.scheduler.min_lr",
    "training.scheduler.mode",
    "training.scheduler.monitor",
)

# Mapping keys under these dotted paths are dynamic (validated elsewhere).
_OPEN_MAP_CONFIG_KEYS: Tuple[str, ...] = (
    "normalization.methods",
)


# ==============================================================================
# Small strict helpers
# ==============================================================================


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing: {key}")
    return mapping[key]


def _require_dict(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    val = _require(mapping, key)
    if not isinstance(val, dict):
        raise TypeError(f"bad type: {key}")
    return val


def _as_int(val: Any, key: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"bad type: {key}")
    return int(val)


def _as_bool(val: Any, key: str) -> bool:
    if not isinstance(val, bool):
        raise TypeError(f"bad type: {key}")
    return bool(val)


def _as_str(val: Any, key: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise TypeError(f"bad type: {key}")
    return val.strip()


def _as_opt_int(val: Any, key: str) -> Optional[int]:
    if val is None:
        return None
    return _as_int(val, key)


def _require_dotted(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    cur: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"missing: {dotted_key}")
        cur = cur[part]
    return cur


def _build_allowed_config_prefixes() -> set[str]:
    out: set[str] = set()
    keys = list(_REQUIRED_CONFIG_KEYS) + list(_OPTIONAL_CONFIG_KEYS) + list(_OPEN_MAP_CONFIG_KEYS)
    for dotted in keys:
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return out


_ALLOWED_CONFIG_PREFIXES = _build_allowed_config_prefixes()
_OPEN_MAP_KEY_SET = set(_OPEN_MAP_CONFIG_KEYS)


def _validate_no_unknown_config_keys(mapping: Mapping[str, Any], *, prefix: str = "") -> None:
    for raw_key, val in mapping.items():
        if not isinstance(raw_key, str):
            raise TypeError("bad config key type")
        if raw_key.strip() != raw_key:
            raise KeyError("ambiguous config key whitespace")

        # Comments are allowed anywhere in the config tree.
        if raw_key.startswith("_"):
            continue

        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if path not in _ALLOWED_CONFIG_PREFIXES:
            raise KeyError(f"unknown config key: {path}")

        if path in _OPEN_MAP_KEY_SET:
            if not isinstance(val, Mapping):
                raise TypeError(f"bad type: {path}")
            continue

        if isinstance(val, Mapping):
            _validate_no_unknown_config_keys(val, prefix=path)


def validate_required_config_keys(cfg: Mapping[str, Any]) -> None:
    for key in _REQUIRED_CONFIG_KEYS:
        _require_dotted(cfg, key)
    _validate_no_unknown_config_keys(cfg)
    _validate_scheduler_config(cfg)


def _validate_scheduler_config(cfg: Mapping[str, Any]) -> None:
    tcfg = _require_dict(cfg, "training")
    sched_cfg = _require_dict(tcfg, "scheduler")
    sched_type = _as_str(_require(sched_cfg, "type"), "training.scheduler.type").lower()

    if sched_type == "cosine_with_warmup":
        _require(sched_cfg, "min_lr_ratio")
        return

    if sched_type in {"reduce_on_plateau", "reducelronplateau"}:
        for key in ("factor", "patience", "threshold", "min_lr", "mode", "monitor"):
            _require(sched_cfg, key)
        return

    raise ValueError("bad training.scheduler.type")


def _repo_root(cfg_path: Path) -> Path:
    return cfg_path.parent.resolve()


def _resolve_path(root: Path, p: str) -> str:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((root / pth).resolve())


def _to_relative_path_str(p: str, *, start: Path) -> str:
    """
    Persist path-like config fields as relative to `start` for portability.
    """
    pth = Path(str(p).strip()).expanduser()
    if not pth.is_absolute():
        return str(pth)
    try:
        return os.path.relpath(str(pth.resolve()), str(start.resolve()))
    except Exception:
        return str(pth)


def _portable_config_snapshot(cfg: Mapping[str, Any], *, save_dir: Path) -> Dict[str, Any]:
    """
    Build a config snapshot suitable for disk persistence.

    Runtime uses absolute resolved paths internally; for portability we save
    path-like fields relative to the run directory.
    """
    out = dict(cfg)

    paths_raw = out.get("paths")
    if isinstance(paths_raw, Mapping):
        paths_out: Dict[str, Any] = dict(paths_raw)
        for k, v in paths_out.items():
            if isinstance(v, str) and v.strip():
                paths_out[k] = _to_relative_path_str(v, start=save_dir)
        out["paths"] = paths_out

    runtime_raw = out.get("runtime")
    if isinstance(runtime_raw, Mapping):
        runtime_out: Dict[str, Any] = dict(runtime_raw)
        ckpt = runtime_out.get("checkpoint")
        if isinstance(ckpt, str) and ckpt.strip():
            runtime_out["checkpoint"] = _to_relative_path_str(ckpt, start=save_dir)
        out["runtime"] = runtime_out

    return out


def resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Resolve cfg.paths[*] relative to the config file directory."""
    root = _repo_root(cfg_path)

    out = dict(cfg)
    paths = _require_dict(out, "paths")
    resolved: Dict[str, Any] = dict(paths)

    for k, v in paths.items():
        if isinstance(v, str) and v.strip():
            resolved[k] = _resolve_path(root, v)

    out["paths"] = resolved
    return out


def configure_logging(cfg: Mapping[str, Any]) -> None:
    sys_cfg = _require_dict(cfg, "system")
    level_str = str(_require(sys_cfg, "log_level")).upper().strip()
    level = getattr(logging, level_str, None)
    if level is None:
        raise ValueError("bad log_level")
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.captureWarnings(True)


def configure_runtime_warning_filters(*, preload_to_device: bool, num_workers: int) -> None:
    """Suppress known false-positive Lightning warnings for preloaded-device streaming."""
    if not preload_to_device or num_workers != 0:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*'train_dataloader' does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'val_dataloader' does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Your `IterableDataset` has `__len__` defined\..*",
    )


def select_preload_device(cfg: Mapping[str, Any]) -> torch.device:
    """Select device ONLY for dataset preloading (dataset.preload_to_device=True)."""
    sys_cfg = _require_dict(cfg, "system")
    pref = str(_require(sys_cfg, "device")).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        return torch.device(pref)

    if pref == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("mps unavailable")
        return torch.device("mps")

    raise ValueError("bad system.device")


# ==============================================================================
# Manifest consistency
# ==============================================================================


_REQUIRED_GLOBALS: Tuple[str, str] = ("P", "T")


def load_manifest_and_validate_config(
    cfg: Mapping[str, Any],
    processed_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load normalization.json and require config.data matches it (if present in manifest)."""
    mpath = processed_dir / "normalization.json"
    if not mpath.exists():
        raise FileNotFoundError("missing normalization.json")

    manifest = load_json_config(mpath)
    if not isinstance(manifest, dict):
        raise TypeError("bad normalization.json")

    data_cfg = _require_dict(cfg, "data")
    cfg_species = _require(data_cfg, "species_variables")
    cfg_globals = _require(data_cfg, "global_variables")

    if not isinstance(cfg_species, list) or not all(isinstance(x, str) for x in cfg_species) or not cfg_species:
        raise TypeError("bad data.species_variables")
    if not isinstance(cfg_globals, list) or not all(isinstance(x, str) for x in cfg_globals):
        raise TypeError("bad data.global_variables")
    if list(cfg_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"data.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")

    if "species_variables" not in manifest:
        raise KeyError("normalization.json missing species_variables")
    if "global_variables" not in manifest:
        raise KeyError("normalization.json missing global_variables")

    man_species = manifest["species_variables"]
    man_globals = manifest["global_variables"]

    if not isinstance(man_species, list) or not all(isinstance(x, str) for x in man_species) or not man_species:
        raise TypeError("bad manifest.species_variables")
    if list(cfg_species) != list(man_species):
        raise ValueError("species_variables mismatch")

    if not isinstance(man_globals, list) or not all(isinstance(x, str) for x in man_globals):
        raise TypeError("bad manifest.global_variables")
    if list(cfg_globals) != list(man_globals):
        raise ValueError("global_variables mismatch")
    if list(man_globals) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"manifest.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")

    return manifest, list(cfg_species), list(cfg_globals)


# ==============================================================================
# Rollout sizing
# ==============================================================================


def max_rollout_steps_for_training(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for training split."""
    base_k = _as_int(_require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")

    cur = tcfg.get("curriculum", None)
    if cur is None:
        return base_k
    if not isinstance(cur, dict):
        raise TypeError("bad training.curriculum")

    enabled = cur.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("bad curriculum.enabled")
    if not enabled:
        return base_k

    # Canonical keys only (no aliases).
    start_steps = _as_int(_require(cur, "start_steps"), "training.curriculum.start_steps")
    end_steps = _as_int(_require(cur, "end_steps"), "training.curriculum.end_steps")

    if start_steps < 1 or end_steps < 1:
        raise ValueError("bad curriculum steps")

    return int(max(base_k, end_steps))


def max_rollout_steps_for_eval(tcfg: Mapping[str, Any]) -> int:
    """Dataset sizing K for validation split (fixed horizon)."""
    base_k = _as_int(_require(tcfg, "rollout_steps"), "training.rollout_steps")
    if base_k < 1:
        raise ValueError("bad rollout_steps")
    return base_k


# ==============================================================================
# Dataloaders
# ==============================================================================


def create_dataloaders(
    cfg: Mapping[str, Any],
    *,
    tcfg: Mapping[str, Any],
    precision: PrecisionConfig,
    preload_device: torch.device,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val dataloaders (strict)."""
    paths = _require_dict(cfg, "paths")
    processed_dir = Path(_as_str(_require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    ds_cfg = _require_dict(cfg, "dataset")
    windows_per_traj = _as_int(_require(ds_cfg, "windows_per_trajectory"), "dataset.windows_per_trajectory")
    preload_to_device = _as_bool(_require(ds_cfg, "preload_to_device"), "dataset.preload_to_device")
    shard_cache_size = _as_int(_require(ds_cfg, "shard_cache_size"), "dataset.shard_cache_size")

    batch_size = _as_int(_require(tcfg, "batch_size"), "training.batch_size")
    num_workers = _as_int(_require(tcfg, "num_workers"), "training.num_workers")
    pin_memory = _as_bool(_require(tcfg, "pin_memory"), "training.pin_memory")
    persistent_workers = _as_bool(_require(tcfg, "persistent_workers"), "training.persistent_workers")
    prefetch_factor = _as_opt_int(_require(tcfg, "prefetch_factor"), "training.prefetch_factor")

    if num_workers < 0:
        raise ValueError("bad num_workers")
    if batch_size <= 0:
        raise ValueError("bad batch_size")

    # Strict DataLoader semantics.
    if num_workers == 0:
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers>0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers>0")
    else:
        # prefetch_factor=None means "use torch default" (2) by not passing it through.
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("bad prefetch_factor")

    k_train = max_rollout_steps_for_training(tcfg)
    k_eval = max_rollout_steps_for_eval(tcfg)

    # Dataset dtype policy (centralized in cfg.precision):
    # - If preloading to an accelerator, store tensors using precision.preload_dtype (often bf16 to save HBM).
    # - Otherwise, emit precision.dataset_dtype.
    storage_dtype = precision.preload_dtype if preload_to_device else precision.dataset_dtype

    common_ds_kwargs = dict(
        windows_per_trajectory=windows_per_traj,
        preload_to_device=preload_to_device,
        device=preload_device,
        storage_dtype=storage_dtype,
        shard_cache_size=shard_cache_size,
    )

    train_ds = FlowMapRolloutDataset(processed_dir, "train", total_steps=k_train, seed=seed, **common_ds_kwargs)
    val_ds = FlowMapRolloutDataset(processed_dir, "validation", total_steps=k_eval, seed=seed + 1, **common_ds_kwargs)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_dl = create_dataloader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = create_dataloader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)

    log.info("Dataloaders: train(K=%d) val(K=%d)", k_train, k_eval)
    return train_dl, val_dl


# ==============================================================================
# Checkpoints (strict)
# ==============================================================================


def _load_weights_only(module: torch.nn.Module, ckpt_path: Path, *, strict: bool) -> None:
    """Load Lightning checkpoint weights only (no optimizer/scheduler state)."""
    if not ckpt_path.exists():
        raise FileNotFoundError("ckpt not found")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("bad ckpt")
    if "state_dict" not in obj:
        raise KeyError("missing state_dict")

    state_dict = obj["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("bad state_dict")

    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError("state mismatch")


def main() -> None:
    cfg_path = DEFAULT_CONFIG_PATH.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError("config not found")

    cfg = load_json_config(cfg_path)
    if not isinstance(cfg, dict):
        raise TypeError("bad config")
    validate_required_config_keys(cfg)

    cfg = resolve_paths(cfg, cfg_path)
    configure_logging(cfg)

    # Required top-level sections (strict).
    tcfg = _require_dict(cfg, "training")
    runtime = _require_dict(cfg, "runtime")
    sys_cfg = _require_dict(cfg, "system")
    paths = _require_dict(cfg, "paths")
    ds_cfg = _require_dict(cfg, "dataset")

    if "mode" in runtime:
        raise ValueError("runtime.mode is unsupported; training is the only runtime mode")

    seed = _as_int(_require(sys_cfg, "seed"), "system.seed")
    seed_everything(seed, workers=True)

    work_dir = Path(_as_str(_require(paths, "work_dir"), "paths.work_dir")).expanduser().resolve()

    # Check checkpoint_mode early to decide if work_dir must be empty.
    ckpt_mode = _as_str(_require(tcfg, "checkpoint_mode"), "training.checkpoint_mode").lower()
    if ckpt_mode not in ("none", "resume", "weights_only"):
        raise ValueError("bad checkpoint_mode")

    # No automatic backups. Fresh training requires an empty work_dir.
    # Resume mode allows non-empty work_dir (continuing in same directory).
    if ckpt_mode != "resume":
        if work_dir.exists() and any(work_dir.iterdir()):
            raise RuntimeError("work_dir not empty")
    ensure_dir(work_dir)

    processed_dir = Path(_as_str(_require(paths, "processed_dir"), "paths.processed_dir")).expanduser().resolve()

    # Manifest must exist; config.data must match it if manifest provides variable lists.
    manifest, species_vars, global_vars = load_manifest_and_validate_config(cfg, processed_dir)

    preload_to_device = _as_bool(_require(ds_cfg, "preload_to_device"), "dataset.preload_to_device")
    preload_device = select_preload_device(cfg) if preload_to_device else torch.device("cpu")
    log.info("preload device: %s", preload_device)

    num_workers = _as_int(_require(tcfg, "num_workers"), "training.num_workers")
    configure_runtime_warning_filters(preload_to_device=preload_to_device, num_workers=num_workers)

    prec = parse_precision_config(cfg)
    log.info(
        "precision: compute=%s amp=%s model=%s input=%s dataset=%s preload=%s loss=%s lightning=%s",
        str(prec.compute_dtype),
        prec.amp_mode,
        str(prec.model_dtype),
        str(prec.input_dtype),
        str(prec.dataset_dtype),
        str(prec.preload_dtype),
        str(prec.loss_dtype),
        prec.lightning_precision,
    )

    train_dl, val_dl = create_dataloaders(
        cfg,
        tcfg=tcfg,
        precision=prec,
        preload_device=preload_device,
        seed=seed,
    )
    try:
        train_batches_per_epoch: Optional[int] = len(train_dl)
    except TypeError:
        train_batches_per_epoch = None

    model = create_model(dict(cfg))
    model = model.to(dtype=prec.model_dtype)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: type=%s S=%d G=%d params=%d trainable=%d",
             str(_require_dict(cfg, "model").get("type", "")),
             len(species_vars), len(global_vars), total_params, trainable_params)

    lit_module = FlowMapRolloutModule(
        cfg=cfg,
        model=model,
        normalization_manifest=manifest,
        species_variables=species_vars,
        precision=prec,
    )

    # On resume/weights-only runs, preserve any existing metrics.csv before Lightning touches it.
    if ckpt_mode in ("resume", "weights_only") and os.environ.get("LOCAL_RANK", "0") == "0":
        m = work_dir / "metrics.csv"
        if m.exists():
            dst = work_dir / "metrics.pre_restart.csv"
            i = 1
            while dst.exists():
                dst = work_dir / f"metrics.pre_restart.{i}.csv"
                i += 1
            m.replace(dst)

    trainer = build_lightning_trainer(
        cfg,
        work_dir=work_dir,
        precision_config=prec,
        train_batches_per_epoch=train_batches_per_epoch,
    )
    atomic_write_json(work_dir / "config.resolved.json", _portable_config_snapshot(cfg, save_dir=work_dir))

    ckpt_val = runtime.get("checkpoint", None)
    ckpt_path: Optional[Path] = None

    if ckpt_val is not None:
        ckpt_raw = _as_str(ckpt_val, "runtime.checkpoint")
        ckpt_path = Path(_resolve_path(_repo_root(cfg_path), ckpt_raw))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        if not ckpt_path.is_file():
            raise ValueError(f"checkpoint must be a file: {ckpt_path}")

    strict_load = _as_bool(_require(runtime, "load_weights_strict"), "runtime.load_weights_strict")

    if ckpt_mode == "none":
        if ckpt_path is not None:
            raise ValueError("checkpoint_mode=none with checkpoint set")
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

    elif ckpt_mode == "resume":
        if ckpt_path is None:
            raise ValueError("resume requires checkpoint")
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=str(ckpt_path))

    elif ckpt_mode == "weights_only":
        if ckpt_path is None:
            raise ValueError("weights_only requires checkpoint")
        _load_weights_only(lit_module, ckpt_path, strict=strict_load)
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=None)

    else:
        raise ValueError("bad checkpoint_mode")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Concise top-level error. No traceback by default.
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
