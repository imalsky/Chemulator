#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export.py - Export a self-contained 1-step PHYSICAL-space model (dynamic batch).

Goal:
- Produce portable artifact(s) that can be loaded and run without carrying the processed dataset folder around.
- Bake normalization (species, globals, dt) into the exported module.
- Support dynamic batch size B (shape-polymorphic) for all inputs.
- Intended for inference: model is exported in eval() mode with gradients disabled.

Exported module signature (PHYSICAL units):
  y_next = model(y_phys, dt_seconds, g_phys)

Where:
  y_phys     : [B, S]  (positive; values are clamped to epsilon before log10)
  dt_seconds : [B]     (positive; clamped to epsilon before log10)
  g_phys     : [B, G]  (G = 2 mandatory: P, T)

Output:
  y_next_phys: [B, S]

Saved .pt2 includes an embedded JSON metadata blob via torch.export.save(extra_files=...).
Read it back with:
  extra = {"metadata.json": ""}
  ep = torch.export.load("...pt2", extra_files=extra)
  meta = json.loads(extra["metadata.json"])

Default behavior:
- Try to export CPU and CUDA via `EXPORT_DEVICES = "cpu,cuda"`.
- Any unavailable target is skipped.
- If all requested accelerator targets are unavailable, CPU export is used.

Typical usage:
  python -u testing/export.py

Notes:
- For correctness, export uses strict=True by default.
- The resulting ep.module() can be moved to CUDA/MPS via .to(device) at inference time.
"""

from __future__ import annotations

import json
import os


# Some MKL/OpenMP builds abort on duplicate symbols; this is a pragmatic default.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.export import Dim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import create_model  # noqa: E402


REQUIRED_GLOBALS: Tuple[str, str] = ("P", "T")

# Runtime settings (edit here; no argparse).
RUN_DIR = (REPO_ROOT / "models" / "v1_test").resolve()
CHECKPOINT = "checkpoints/last.ckpt"  # relative paths are resolved against run config directory
EXPORT_DEVICES = "cpu,cuda"  # requested targets; unavailable devices are skipped with CPU fallback
EXPORT_DTYPE = "float32"
EXPORT_STRICT = True
VERIFY_CUDA = False
VERIFY_MPS = False
EXAMPLE_BATCH = 4
B_MIN = 1
B_MAX = 16384


# =============================================================================
# Small device diagnostics
# =============================================================================


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def _print_device_diag() -> None:
    cuda_ok = _cuda_available()
    mps_ok = _mps_available()
    print(
        "Device diag:"
        f" torch={torch.__version__}"
        f" torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        f" cuda.is_available={cuda_ok}"
        f" cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        f" mps.is_available={mps_ok}"
    )
    if cuda_ok:
        try:
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass


# =============================================================================
# IO helpers
# =============================================================================


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return obj


def _load_resolved_config(run_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """
    Require the resolved training config from the run directory.
    """
    cfg_path = run_dir / "config.resolved.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {cfg_path}")
    return _load_json(cfg_path), cfg_path


def _resolve_checkpoint_path(raw: str, *, cfg_path: Path) -> Path:
    """
    Resolve a checkpoint path strictly:
      - relative paths are resolved against config directory
      - checkpoint must exist and be a file
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (cfg_path.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    if not p.is_file():
        raise ValueError(f"checkpoint must be a file: {p}")
    return p


# =============================================================================
# Checkpoint loading (robust to common Lightning prefixes)
# =============================================================================


_STRIP_PREFIXES = (
    "state_dict.",
    "model.",
    "module.",
    "_orig_mod.",
    "model._orig_mod.",
    "module.model.",
    "module._orig_mod.",
)


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _STRIP_PREFIXES:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


_IGNORED_STATE_PREFIXES = (
    "criterion.",
)


def _is_ignored_state_key(key: str) -> bool:
    return any(key.startswith(p) for p in _IGNORED_STATE_PREFIXES)


def _load_weights_strict(model: nn.Module, ckpt_path: Path) -> None:
    """
    Load checkpoint weights into `model` with strict key matching.

    - Accepts either a Lightning checkpoint dict (with "state_dict") or a raw state_dict.
    - Strips common module prefixes.
    - Errors out on missing keys (strong correctness signal).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint state_dict must be a mapping, got {type(state).__name__}")
    cleaned: Dict[str, Any] = {}
    ignored: List[str] = []
    for raw_k, v in state.items():
        k = _strip_prefixes(str(raw_k))
        if _is_ignored_state_key(k):
            ignored.append(k)
            continue
        cleaned[k] = v

    model_sd = model.state_dict()
    missing = [k for k in model_sd.keys() if k not in cleaned]
    unexpected = [k for k in cleaned.keys() if k not in model_sd]

    if missing or unexpected:
        lines: List[str] = []
        if missing:
            lines.append(f"missing={len(missing)}:\n  " + "\n  ".join(missing[:25]))
        if unexpected:
            lines.append(f"unexpected={len(unexpected)}:\n  " + "\n  ".join(unexpected[:25]))
        msg = "\n".join(lines)
        raise RuntimeError(f"Checkpoint/model key mismatch:\n{msg}\nckpt={ckpt_path}")

    if ignored:
        preview = ", ".join(ignored[:4])
        suffix = " ..." if len(ignored) > 4 else ""
        print(f"Ignored training-only checkpoint keys ({len(ignored)}): {preview}{suffix}")

    ordered_state = {k: cleaned[k] for k in model_sd.keys()}
    model.load_state_dict(ordered_state, strict=True)


def _freeze_for_inference(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# Normalization bake-in (must match preprocessing.py exactly)
# =============================================================================


def _canonical_method(method: str) -> str:
    m = str(method).lower().strip()
    return m


_SUPPORTED_SPECIES_METHODS = {"log-standard"}
_SUPPORTED_GLOBAL_METHODS = {"standard", "min-max", "log-min-max", "log-standard"}


class BakedNormalizer(nn.Module):
    """
    Torch module containing all normalization constants as buffers.

    This is intentionally simple and explicit (no dependencies on a separate normalizer.py),
    and should match processing/preprocessing.py behavior.
    """

    def __init__(
        self,
        *,
        species_vars: Sequence[str],
        global_vars: Sequence[str],
        species_methods: Sequence[str],
        global_methods: Sequence[str],
        s_log_mean: torch.Tensor,  # [S]
        s_log_std: torch.Tensor,   # [S]
        s_log_min: torch.Tensor,   # [S]
        s_log_max: torch.Tensor,   # [S]
        s_eps: torch.Tensor,       # [S]
        g_mean: torch.Tensor,      # [G]
        g_std: torch.Tensor,       # [G]
        g_min: torch.Tensor,       # [G]
        g_max: torch.Tensor,       # [G]
        g_log_mean: torch.Tensor,  # [G]
        g_log_std: torch.Tensor,   # [G]
        g_log_min: torch.Tensor,   # [G]
        g_log_max: torch.Tensor,   # [G]
        g_eps: torch.Tensor,       # [G]
        dt_log_min: float,
        dt_log_max: float,
        dt_eps: float,
    ) -> None:
        super().__init__()

        self.species_vars = tuple(species_vars)
        self.global_vars = tuple(global_vars)
        self.species_methods = tuple(species_methods)
        self.global_methods = tuple(global_methods)

        # Species (log-space stats)
        self.register_buffer("s_log_mean", s_log_mean.to(torch.float32))
        self.register_buffer("s_log_std", s_log_std.to(torch.float32))
        self.register_buffer("s_log_min", s_log_min.to(torch.float32))
        self.register_buffer("s_log_max", s_log_max.to(torch.float32))
        self.register_buffer("s_eps", s_eps.to(torch.float32))

        # Globals (raw + log-space min/max)
        self.register_buffer("g_mean", g_mean.to(torch.float32))
        self.register_buffer("g_std", g_std.to(torch.float32))
        self.register_buffer("g_min", g_min.to(torch.float32))
        self.register_buffer("g_max", g_max.to(torch.float32))
        self.register_buffer("g_log_mean", g_log_mean.to(torch.float32))
        self.register_buffer("g_log_std", g_log_std.to(torch.float32))
        self.register_buffer("g_log_min", g_log_min.to(torch.float32))
        self.register_buffer("g_log_max", g_log_max.to(torch.float32))
        self.register_buffer("g_eps", g_eps.to(torch.float32))

        # dt (log10 min/max from cfg.dt_min/cfg.dt_max)
        self.register_buffer("dt_log_min", torch.tensor(float(dt_log_min), dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(float(dt_log_max), dtype=torch.float32))
        self.register_buffer("dt_eps", torch.tensor(float(dt_eps), dtype=torch.float32))

    @property
    def S(self) -> int:
        return int(self.s_log_mean.numel())

    @property
    def G(self) -> int:
        return int(self.g_mean.numel())

    def normalize_dt_seconds(self, dt_seconds: torch.Tensor) -> torch.Tensor:
        # preprocessing.py:
        #   dt_norm = clip((log10(max(dt, eps)) - log10(dt_min)) / (log10(dt_max) - log10(dt_min)), 0, 1)
        dt_f = torch.clamp_min(dt_seconds.to(torch.float32), self.dt_eps)
        log_dt = torch.log10(dt_f)

        denom = (self.dt_log_max - self.dt_log_min).clamp_min(1e-12)
        dt_norm = (log_dt - self.dt_log_min) / denom
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        return dt_norm.to(dt_seconds.dtype)

    def normalize_species(self, y_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize species to z-space (model input space).

        All species use log-standard normalization:
            z = (log10(max(y, eps)) - log_mean) / log_std
        Vectorized for inference performance.
        """
        y_f = torch.maximum(y_phys.to(torch.float32), self.s_eps)
        z = (torch.log10(y_f) - self.s_log_mean) / self.s_log_std
        return z.to(y_phys.dtype)

    def denormalize_species(self, y_z: torch.Tensor) -> torch.Tensor:
        """
        Convert z-space species back to physical space.

        Inverse of normalize_species (all species use log-standard):
            y = 10^(z * log_std + log_mean)
        Vectorized for inference performance.
        """
        z_f = y_z.to(torch.float32)
        logx = z_f * self.s_log_std + self.s_log_mean
        y = torch.pow(10.0, logx)
        return y.to(y_z.dtype)

    def normalize_globals(self, g_phys: torch.Tensor) -> torch.Tensor:
        """
        Normalize globals to the model's expected space.

        preprocessing.py supports:
          standard | min-max | log-min-max | log-standard
        """
        g_f = g_phys.to(torch.float32)
        cols: List[torch.Tensor] = []
        for j, m in enumerate(self.global_methods):
            mj = _canonical_method(m)
            xj = g_f[:, j]

            if mj == "standard":
                denom = self.g_std[j].clamp_min(1e-12)
                zj = (xj - self.g_mean[j]) / denom
            elif mj == "min-max":
                denom = (self.g_max[j] - self.g_min[j]).clamp_min(1e-12)
                zj = (xj - self.g_min[j]) / denom
            elif mj == "log-min-max":
                denom = (self.g_log_max[j] - self.g_log_min[j]).clamp_min(1e-12)
                zj = (torch.log10(torch.clamp_min(xj, self.g_eps[j])) - self.g_log_min[j]) / denom
            elif mj == "log-standard":
                denom = self.g_log_std[j].clamp_min(1e-12)
                zj = (torch.log10(torch.clamp_min(xj, self.g_eps[j])) - self.g_log_mean[j]) / denom
            else:
                raise RuntimeError(f"Unsupported global normalization method: {m}")

            cols.append(zj)

        z = torch.stack(cols, dim=-1)
        return z.to(g_phys.dtype)


def build_baked_normalizer(
    manifest: Mapping[str, Any],
    *,
    species_vars: Sequence[str],
    global_vars: Sequence[str],
) -> BakedNormalizer:
    methods_map = manifest.get("normalization_methods")
    if not isinstance(methods_map, Mapping):
        raise TypeError("normalization.json: expected 'normalization_methods' mapping.")

    stats = manifest.get("per_key_stats")
    if not isinstance(stats, Mapping):
        raise TypeError("normalization.json: expected 'per_key_stats' mapping.")

    eps_global = float(manifest.get("epsilon", 1e-30))

    def _method_for(k: str, *, allowed: set[str]) -> str:
        raw = methods_map.get(k)
        m = _canonical_method("" if raw is None else str(raw))
        if m not in allowed:
            raise ValueError(f"Unsupported/missing normalization method for '{k}': {raw!r}")
        return m

    s_methods = [_method_for(k, allowed=_SUPPORTED_SPECIES_METHODS) for k in species_vars]
    for k, m in zip(species_vars, s_methods):
        if m != "log-standard":
            raise ValueError(f"Species normalization must be log-standard, got '{m}' for '{k}'")
    g_methods = [_method_for(k, allowed=_SUPPORTED_GLOBAL_METHODS) for k in global_vars]

    # Species log stats (required)
    s_log_mean = torch.tensor([float(stats[k]["log_mean"]) for k in species_vars], dtype=torch.float32)
    s_log_std = torch.tensor([float(stats[k]["log_std"]) for k in species_vars], dtype=torch.float32)
    s_log_min = torch.tensor([float(stats[k].get("log_min", 0.0)) for k in species_vars], dtype=torch.float32)
    s_log_max = torch.tensor([float(stats[k].get("log_max", 1.0)) for k in species_vars], dtype=torch.float32)
    s_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in species_vars], dtype=torch.float32)

    # Globals stats (required; produced by canonical preprocessing).
    g_mean = torch.tensor([float(stats[k]["mean"]) for k in global_vars], dtype=torch.float32)
    g_std = torch.tensor([float(stats[k]["std"]) for k in global_vars], dtype=torch.float32)
    g_min = torch.tensor([float(stats[k]["min"]) for k in global_vars], dtype=torch.float32)
    g_max = torch.tensor([float(stats[k]["max"]) for k in global_vars], dtype=torch.float32)
    g_log_mean = torch.tensor([float(stats[k]["log_mean"]) for k in global_vars], dtype=torch.float32)
    g_log_std = torch.tensor([float(stats[k]["log_std"]) for k in global_vars], dtype=torch.float32)
    g_log_min = torch.tensor([float(stats[k]["log_min"]) for k in global_vars], dtype=torch.float32)
    g_log_max = torch.tensor([float(stats[k]["log_max"]) for k in global_vars], dtype=torch.float32)
    g_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in global_vars], dtype=torch.float32)

    dt = manifest.get("dt") or {}
    if not isinstance(dt, Mapping) or "log_min" not in dt or "log_max" not in dt:
        raise KeyError("normalization.json: missing dt.log_min / dt.log_max")

    return BakedNormalizer(
        species_vars=species_vars,
        global_vars=global_vars,
        species_methods=s_methods,
        global_methods=g_methods,
        s_log_mean=s_log_mean,
        s_log_std=s_log_std,
        s_log_min=s_log_min,
        s_log_max=s_log_max,
        s_eps=s_eps,
        g_mean=g_mean,
        g_std=g_std,
        g_min=g_min,
        g_max=g_max,
        g_log_mean=g_log_mean,
        g_log_std=g_log_std,
        g_log_min=g_log_min,
        g_log_max=g_log_max,
        g_eps=g_eps,
        dt_log_min=float(dt["log_min"]),
        dt_log_max=float(dt["log_max"]),
        dt_eps=eps_global,
    )


# =============================================================================
# Exportable physical-space 1-step wrapper
# =============================================================================


class OneStepPhysical(nn.Module):
    """
    Wrapper: physical -> (normalize) -> base.forward_step(z) -> (denormalize) -> physical.
    """

    def __init__(self, base: nn.Module, norm: BakedNormalizer) -> None:
        super().__init__()
        self.base = base
        self.norm = norm

        self.S = int(getattr(base, "S"))
        self.G = int(getattr(base, "G"))
        if self.S != norm.S:
            raise RuntimeError(f"S mismatch: model.S={self.S} norm.S={norm.S}")
        if self.G != norm.G:
            raise RuntimeError(f"G mismatch: model.G={self.G} norm.G={norm.G}")

        if not hasattr(base, "forward_step"):
            raise TypeError("Base model must implement forward_step(y_z, dt_norm, g_z)")

    def forward(self, y_phys: torch.Tensor, dt_seconds: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        y_z = self.norm.normalize_species(y_phys)
        dt_norm = self.norm.normalize_dt_seconds(dt_seconds)  # [B]
        g_z = self.norm.normalize_globals(g_phys)
        y_next_z = self.base.forward_step(y_z, dt_norm, g_z)
        return self.norm.denormalize_species(y_next_z)


def _make_example_inputs(
    norm: BakedNormalizer,
    *,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a numerically sane example batch for export.

    We sample y and dt in *log space* within the min/max ranges captured during preprocessing
    to avoid extreme values that can produce inf/NaN during log10.
    """
    S = norm.S
    G = norm.G

    # y: sample log10(y) uniformly between observed log_min/log_max.
    u = torch.rand(B, S, device=device, dtype=torch.float32)
    logy = norm.s_log_min.to(device) + u * (norm.s_log_max.to(device) - norm.s_log_min.to(device))
    y = torch.pow(10.0, logy).to(dtype)

    # dt_seconds: sample log10(dt) uniformly between cfg log_min/log_max.
    u_dt = torch.rand(B, device=device, dtype=torch.float32)
    logdt = norm.dt_log_min.to(device) + u_dt * (norm.dt_log_max.to(device) - norm.dt_log_min.to(device))
    dt_seconds = torch.pow(10.0, logdt).to(dtype)

    # g: sample in a reasonable range (roughly within observed min/max).
    u_g = torch.rand(B, G, device=device, dtype=torch.float32)
    g = (norm.g_min.to(device) + u_g * (norm.g_max.to(device) - norm.g_min.to(device))).to(dtype)

    return y, dt_seconds, g


def _verify_dynamic_batch(
    ep: torch.export.ExportedProgram,
    *,
    device: torch.device,
    dtype: torch.dtype,
    norm: BakedNormalizer,
) -> None:
    """
    Fail hard unless the exported program runs with multiple batch sizes (proves dynamic B).
    """
    m = ep.module().to(device=device, dtype=dtype)

    for B in (1, 7):
        y, dt, g = _make_example_inputs(norm, B=B, device=device, dtype=dtype)
        with torch.inference_mode():
            out = m(y, dt, g)
        if out.shape[0] != B:
            raise RuntimeError(f"Dynamic batch verification failed: input B={B} produced output {tuple(out.shape)}")


# =============================================================================
# Manifest/config resolution
# =============================================================================


def _validate_manifest_vs_config(cfg: Mapping[str, Any], manifest: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Strong correctness check: channel order must match config <-> manifest.

    Config and manifest must both provide matching channel order.
    """
    cfg_data = cfg.get("data")
    if not isinstance(cfg_data, Mapping):
        raise TypeError("config missing data section")
    cfg_species = cfg_data.get("species_variables")
    cfg_globals = cfg_data.get("global_variables")

    man_species = list(manifest.get("species_variables", []) or [])
    man_globals = list(manifest.get("global_variables", []) or [])

    if not isinstance(cfg_species, list) or not all(isinstance(x, str) for x in cfg_species) or not cfg_species:
        raise TypeError("config: bad data.species_variables")
    if list(cfg_species) != man_species:
        raise ValueError("species_variables mismatch between config and normalization.json (order matters).")
    species = list(cfg_species)

    if not isinstance(cfg_globals, list) or not all(isinstance(x, str) for x in cfg_globals):
        raise TypeError("config: bad data.global_variables")
    if list(cfg_globals) != list(REQUIRED_GLOBALS):
        raise ValueError(f"config.data.global_variables must be exactly {list(REQUIRED_GLOBALS)}")
    if list(man_globals) != list(REQUIRED_GLOBALS):
        raise ValueError(f"normalization.json global_variables must be exactly {list(REQUIRED_GLOBALS)}")
    if list(cfg_globals) != man_globals:
        raise ValueError("global_variables mismatch between config and normalization.json (order matters).")
    globals_ = list(cfg_globals)

    return species, globals_


def _resolve_processed_dir(cfg: Mapping[str, Any], *, cfg_path: Path) -> Path:
    """
    Locate processed_dir containing normalization.json using cfg.paths.processed_dir.
    """
    paths = cfg.get("paths", {}) or {}
    if not isinstance(paths, Mapping):
        raise TypeError("config missing paths section")

    processed = paths.get("processed_dir")
    if not isinstance(processed, str) or not processed.strip():
        raise KeyError("config.paths.processed_dir is required")

    p = Path(processed).expanduser()
    if not p.is_absolute():
        p = (cfg_path.parent / p).resolve()
    if not (p / "normalization.json").exists():
        raise FileNotFoundError(f"normalization.json not found under processed_dir: {p}")
    return p


def _parse_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.strip().lower()
    if s == "float32":
        return torch.float32
    if s == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{dtype_str}'. Use float32|bfloat16.")


def _parse_devices(raw_devices: str) -> List[str]:
    raw = str(raw_devices).strip().lower()
    if not raw:
        raise ValueError("EXPORT_DEVICES produced empty list")
    devs = [d.strip().lower() for d in raw.split(",") if d.strip()]
    if not devs:
        raise ValueError("EXPORT_DEVICES produced empty list")
    seen: set[str] = set()
    out: List[str] = []
    for d in devs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    tags = ["cuda" if d.startswith("cuda") else d for d in out]
    if len(tags) != len(set(tags)):
        raise ValueError("devices list contains duplicate export targets (e.g., both 'cuda' and 'cuda:0')")
    return out


def _resolve_export_devices(devices: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for dev in devices:
        if dev == "cpu":
            resolved.append(dev)
            continue

        if dev.startswith("cuda"):
            if _cuda_available():
                resolved.append(dev)
            else:
                print(
                    "CUDA requested in EXPORT_DEVICES but torch.cuda.is_available() is false; "
                    f"skipping '{dev}'."
                )
            continue

        if dev == "mps":
            if _mps_available():
                resolved.append(dev)
            else:
                print("MPS requested in EXPORT_DEVICES but torch.backends.mps.is_available() is false; skipping.")
            continue

        raise ValueError(f"Unsupported device '{dev}'. Use cpu|cuda|mps (or cuda:0 style if needed).")

    if not resolved:
        print("No requested export devices are available. Falling back to CPU.")
        resolved = ["cpu"]

    return resolved


def _default_out_for(run_dir: Path, device_tag: str) -> Path:
    return (run_dir / f"export_{device_tag}_dynB_1step_phys.pt2").resolve()


# =============================================================================
# Export per-device
# =============================================================================


def _export_one(
    *,
    device_tag: str,
    device: torch.device,
    dtype: torch.dtype,
    cfg_path: Path,
    ckpt_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    species_vars: Sequence[str],
    global_vars: Sequence[str],
    base_cpu: nn.Module,
    norm_cpu: BakedNormalizer,
    out_path: Path,
    run_dir: Path,
    strict: bool,
    example_batch: int,
    b_min: int,
    b_max: int,
    verify_cuda: bool,
    verify_mps: bool,
) -> None:
    # Move model and normalizer to device/dtype for this export.
    base = base_cpu.to(device=device, dtype=dtype)
    base = _freeze_for_inference(base)

    norm = norm_cpu.to(device=device, dtype=dtype)
    norm.eval()

    step = OneStepPhysical(base, norm)
    step = _freeze_for_inference(step)

    B_ex = int(max(1, example_batch))
    example_inputs = _make_example_inputs(norm, B=B_ex, device=device, dtype=dtype)

    B = Dim("B", min=int(b_min), max=int(b_max))
    dynamic_shapes = (
        {0: B},  # y_phys: (B, S)
        {0: B},  # dt_seconds: (B,)
        {0: B},  # g_phys: (B, G)
    )

    ep = torch.export.export(step, example_inputs, dynamic_shapes=dynamic_shapes, strict=bool(strict))

    # Verify on export device.
    _verify_dynamic_batch(ep, device=device, dtype=dtype, norm=norm)

    meta = {
        "format": "1step_physical_dynB",
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "checkpoint_path": str(ckpt_path),
        "normalization_path": str(manifest_path),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "export_device": str(device),
        "export_device_tag": str(device_tag),
        "export_dtype": str(dtype).replace("torch.", ""),
        "species_variables": list(species_vars),
        "global_variables": list(global_vars),
        "normalization_methods": dict(manifest["normalization_methods"]),
        "epsilon": float(manifest.get("epsilon", 1e-30)),
        "dt_log10_min": float(manifest["dt"]["log_min"]),
        "dt_log10_max": float(manifest["dt"]["log_max"]),
        "dt_min_seconds": float(10.0 ** float(manifest["dt"]["log_min"])),
        "dt_max_seconds": float(10.0 ** float(manifest["dt"]["log_max"])),
        "dynamic_batch": {"min": int(b_min), "max": int(b_max)},
        "signature": {
            "inputs": {"y_phys": ["B", "S"], "dt_seconds": ["B"], "g_phys": ["B", "G"]},
            "output": {"y_next_phys": ["B", "S"]},
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path, extra_files={"metadata.json": json.dumps(meta, indent=2, sort_keys=True)})
    print(f"Saved export ({device_tag}): {out_path}")

    # Optional cross-device verification.
    if bool(verify_cuda) and _cuda_available():
        _verify_dynamic_batch(ep, device=torch.device("cuda"), dtype=dtype, norm=norm.to("cuda"))
        print(f"Verified on CUDA ({device_tag} export).")
    if bool(verify_mps) and _mps_available():
        _verify_dynamic_batch(ep, device=torch.device("mps"), dtype=dtype, norm=norm.to("mps"))
        print(f"Verified on MPS ({device_tag} export).")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    _print_device_diag()

    run_dir = RUN_DIR.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR not found: {run_dir}")

    dtype = _parse_dtype(EXPORT_DTYPE)

    cfg, cfg_path = _load_resolved_config(run_dir)
    ckpt_path = _resolve_checkpoint_path(CHECKPOINT, cfg_path=cfg_path)

    processed_dir = _resolve_processed_dir(cfg, cfg_path=cfg_path)
    manifest_path = processed_dir / "normalization.json"
    manifest = _load_json(manifest_path)

    species_vars, global_vars = _validate_manifest_vs_config(cfg, manifest)

    # Build base model on CPU once and load weights once.
    base_cpu = create_model(cfg)
    _load_weights_strict(base_cpu, ckpt_path)
    base_cpu = _freeze_for_inference(base_cpu.to(device=torch.device("cpu"), dtype=dtype))

    # Build baked normalizer once (buffers on CPU); we move it per-device for export.
    norm_cpu = build_baked_normalizer(manifest, species_vars=species_vars, global_vars=global_vars)
    norm_cpu = norm_cpu.to(device=torch.device("cpu"), dtype=dtype)
    norm_cpu.eval()

    device_tags = _resolve_export_devices(_parse_devices(EXPORT_DEVICES))

    for dev in device_tags:
        device = torch.device(dev)
        device_tag = "cuda" if dev.startswith("cuda") else dev
        out_path = _default_out_for(run_dir, device_tag)

        _export_one(
            device_tag=device_tag,
            device=device,
            dtype=dtype,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            manifest=manifest,
            manifest_path=manifest_path,
            species_vars=species_vars,
            global_vars=global_vars,
            base_cpu=base_cpu,
            norm_cpu=norm_cpu,
            out_path=out_path,
            run_dir=run_dir,
            strict=bool(EXPORT_STRICT),
            example_batch=int(EXAMPLE_BATCH),
            b_min=int(B_MIN),
            b_max=int(B_MAX),
            verify_cuda=bool(VERIFY_CUDA),
            verify_mps=bool(VERIFY_MPS),
        )


if __name__ == "__main__":
    main()
