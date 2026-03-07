#!/usr/bin/env python3
"""Export physical-I/O model artifact for testing/inference.

The exported model embeds normalization metadata so callers only pass
physical values:
  - y_phys: [B,S] species (physical)
  - dt_sec: [B,1] delta-time in seconds
  - g_phys: [B,G] globals (physical)
  - returns y_next_phys: [B,1,S] (physical)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import prepare_platform_environment

prepare_platform_environment()

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
from src import model as model_module  # noqa: E402


create_model = model_module.create_model


MODEL_DIR_ENV = "CHEMULATOR_MODEL_DIR"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "final_version"
CONFIG_NAME = "config.json"
PHYS_MODEL_FILENAME = "physical_model_k1_cpu.pt2"
PHYS_METADATA_FILENAME = "physical_model_metadata.json"

MIN_BATCH = 1
MAX_BATCH = 4096

METHOD_IDS = {
    "standard": 0,
    "min-max": 1,
    "log-standard": 2,
    "log-min-max": 3,
}


def _runtime_assert(cond: torch.Tensor, message: str) -> None:
    """Export-friendly runtime assert that survives torch.export tracing."""
    if hasattr(torch.ops.aten, "_assert_async") and hasattr(torch.ops.aten._assert_async, "msg"):
        torch.ops.aten._assert_async.msg(cond, message)
        return
    torch._assert(cond, message)


def _resolve_model_dir(default: Path = DEFAULT_MODEL_DIR) -> Path:
    raw = os.getenv(MODEL_DIR_ENV)
    model_dir = Path(raw).expanduser().resolve() if raw else Path(default).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Set {MODEL_DIR_ENV} or edit DEFAULT_MODEL_DIR."
        )
    return model_dir


def _resolve_cfg_path(path_like: str | os.PathLike[str], *, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _find_checkpoint(model_dir: Path) -> Path:
    for name in ("best.ckpt", "last.ckpt"):
        p = model_dir / name
        if p.is_file():
            return p
    ckpts = sorted(model_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")

    for key in ("model_state", "state_dict", "model_state_dict", "model"):
        state = payload.get(key)
        if isinstance(state, dict):
            break
    else:
        tensor_items = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        if not tensor_items:
            raise ValueError("No state dict found in checkpoint")
        state = tensor_items

    clean: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = str(k)
        for prefix in ("model.", "module.", "_orig_mod."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"State dict entry is not a tensor: {k}")
        clean[kk] = v
    return clean


def _load_weights(model: nn.Module, ckpt_path: Path) -> None:
    load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(ckpt_path, **load_kwargs)
    state = _extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing model parameters in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint parameters: {unexpected}")


def _freeze_for_inference(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    return model


def _resolve_config_and_manifest(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    cfg_path = model_dir / CONFIG_NAME
    cfg = _load_json(cfg_path)
    processed_dir = _resolve_cfg_path(cfg["paths"]["processed_data_dir"], base_dir=REPO_ROOT)
    manifest_path = processed_dir / "normalization.json"
    manifest = _load_json(manifest_path)
    return cfg, manifest, manifest_path


def _require_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    raw = cfg.get(key)
    if not isinstance(raw, list):
        raise KeyError(f"Config key must be a list: data.{key}")
    out = [str(x) for x in raw]
    if key == "species_variables" and not out:
        raise ValueError("data.species_variables must be non-empty")
    return out


def _required_float(stats: Mapping[str, Any], key: str, name: str) -> float:
    if key not in stats:
        raise KeyError(f"Missing normalization stat '{key}' for '{name}'")
    return float(stats[key])


def _build_group_tensors(
    *,
    keys: Sequence[str],
    methods: Mapping[str, Any],
    per_key_stats: Mapping[str, Any],
    min_std: float,
) -> Dict[str, torch.Tensor]:
    n = len(keys)
    method_code = torch.empty(n, dtype=torch.long)
    mean = torch.empty(n, dtype=torch.float32)
    std = torch.empty(n, dtype=torch.float32)
    vmin = torch.empty(n, dtype=torch.float32)
    vmax = torch.empty(n, dtype=torch.float32)
    log_mean = torch.empty(n, dtype=torch.float32)
    log_std = torch.empty(n, dtype=torch.float32)
    log_min = torch.empty(n, dtype=torch.float32)
    log_max = torch.empty(n, dtype=torch.float32)

    for i, key in enumerate(keys):
        method_name = str(methods.get(key, ""))
        if method_name not in METHOD_IDS:
            raise ValueError(f"Unsupported normalization method for '{key}': {method_name!r}")
        code = METHOD_IDS[method_name]
        method_code[i] = code

        row_raw = per_key_stats.get(key)
        if not isinstance(row_raw, Mapping):
            raise KeyError(f"Missing normalization stats for key '{key}'")
        row: Mapping[str, Any] = row_raw

        if code == METHOD_IDS["standard"]:
            mean_i = _required_float(row, "mean", key)
            std_i = _required_float(row, "std", key)
            if std_i <= 0.0 or std_i < min_std:
                raise ValueError(f"Invalid std for '{key}': {std_i}")
            mean[i] = mean_i
            std[i] = std_i
            vmin[i] = 0.0
            vmax[i] = 1.0
            log_mean[i] = 0.0
            log_std[i] = 1.0
            log_min[i] = 0.0
            log_max[i] = 1.0
            continue

        if code == METHOD_IDS["min-max"]:
            min_i = _required_float(row, "min", key)
            max_i = _required_float(row, "max", key)
            if max_i <= min_i:
                raise ValueError(f"Invalid min-max range for '{key}': [{min_i}, {max_i}]")
            mean[i] = 0.0
            std[i] = 1.0
            vmin[i] = min_i
            vmax[i] = max_i
            log_mean[i] = 0.0
            log_std[i] = 1.0
            log_min[i] = 0.0
            log_max[i] = 1.0
            continue

        if code == METHOD_IDS["log-standard"]:
            log_mean_i = _required_float(row, "log_mean", key)
            log_std_i = _required_float(row, "log_std", key)
            if log_std_i <= 0.0 or log_std_i < min_std:
                raise ValueError(f"Invalid log_std for '{key}': {log_std_i}")
            mean[i] = 0.0
            std[i] = 1.0
            vmin[i] = 0.0
            vmax[i] = 1.0
            log_mean[i] = log_mean_i
            log_std[i] = log_std_i
            log_min[i] = 0.0
            log_max[i] = 1.0
            continue

        if code == METHOD_IDS["log-min-max"]:
            log_min_i = _required_float(row, "log_min", key)
            log_max_i = _required_float(row, "log_max", key)
            if log_max_i <= log_min_i:
                raise ValueError(f"Invalid log-min-max range for '{key}': [{log_min_i}, {log_max_i}]")
            mean[i] = 0.0
            std[i] = 1.0
            vmin[i] = 0.0
            vmax[i] = 1.0
            log_mean[i] = 0.0
            log_std[i] = 1.0
            log_min[i] = log_min_i
            log_max[i] = log_max_i
            continue

        raise RuntimeError("Unexpected method code")

    return {
        "method_code": method_code,
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "log_mean": log_mean,
        "log_std": log_std,
        "log_min": log_min,
        "log_max": log_max,
    }


class PhysicalModelWrapper(nn.Module):
    """Wrapper that maps physical inputs -> base normalized model -> physical outputs."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        species_keys: Sequence[str],
        global_keys: Sequence[str],
        species_tensors: Dict[str, torch.Tensor],
        global_tensors: Dict[str, torch.Tensor],
        min_std: float,
        dt_log_min: float,
        dt_log_max: float,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.species_keys = list(species_keys)
        self.global_keys = list(global_keys)

        if hasattr(base_model, "S"):
            self.S = int(getattr(base_model, "S"))
        elif hasattr(base_model, "S_in"):
            self.S = int(getattr(base_model, "S_in"))
        else:
            raise AttributeError("Base model must define species dimension attribute 'S' or 'S_in'")
        self.G = int(getattr(base_model, "G", 0))
        if self.S != len(self.species_keys):
            raise ValueError(f"Model species dim mismatch: {self.S} != {len(self.species_keys)}")
        if self.G != len(self.global_keys):
            raise ValueError(f"Model globals dim mismatch: {self.G} != {len(self.global_keys)}")

        for name, tensor in species_tensors.items():
            self.register_buffer(f"species_{name}", tensor, persistent=True)
        if self.G > 0:
            for name, tensor in global_tensors.items():
                self.register_buffer(f"globals_{name}", tensor, persistent=True)

        min_std_value = float(min_std)
        if (not math.isfinite(min_std_value)) or min_std_value <= 0.0:
            raise ValueError("min_std must be finite and > 0")
        self.register_buffer("min_std", torch.tensor(min_std_value, dtype=torch.float32), persistent=True)

        if dt_log_max <= dt_log_min:
            raise ValueError("Invalid dt normalization range")
        self.register_buffer("dt_log_min", torch.tensor(float(dt_log_min), dtype=torch.float32), persistent=True)
        self.register_buffer(
            "dt_log_range",
            torch.tensor(float(dt_log_max - dt_log_min), dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _view(v: torch.Tensor, x_ndim: int) -> torch.Tensor:
        return v.view(*([1] * (x_ndim - 1)), -1)

    def _normalize_group(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        x = x.to(torch.float32)
        method_code = getattr(self, f"{prefix}_method_code")
        mean = getattr(self, f"{prefix}_mean")
        std = getattr(self, f"{prefix}_std")
        vmin = getattr(self, f"{prefix}_min")
        vmax = getattr(self, f"{prefix}_max")
        log_mean = getattr(self, f"{prefix}_log_mean")
        log_std = getattr(self, f"{prefix}_log_std")
        log_min = getattr(self, f"{prefix}_log_min")
        log_max = getattr(self, f"{prefix}_log_max")

        if x.shape[-1] != method_code.numel():
            raise ValueError(f"Expected {method_code.numel()} features, got {x.shape[-1]}")

        vid = self._view(method_code, x.ndim)
        vmean = self._view(mean, x.ndim)
        vstd = self._view(std, x.ndim)
        vvmin = self._view(vmin, x.ndim)
        vvmax = self._view(vmax, x.ndim)
        vlog_mean = self._view(log_mean, x.ndim)
        vlog_std = self._view(log_std, x.ndim)
        vlog_min = self._view(log_min, x.ndim)
        vlog_max = self._view(log_max, x.ndim)

        mask_std = (vid == METHOD_IDS["standard"])
        mask_mm = (vid == METHOD_IDS["min-max"])
        mask_log_std = (vid == METHOD_IDS["log-standard"])
        mask_log_mm = (vid == METHOD_IDS["log-min-max"])
        mask_any_log = mask_log_std | mask_log_mm

        std_ok = torch.logical_or(~mask_std, vstd >= self.min_std)
        _runtime_assert(torch.all(std_ok), "std below min_std in runtime normalization")

        log_std_ok = torch.logical_or(~mask_log_std, vlog_std >= self.min_std)
        _runtime_assert(torch.all(log_std_ok), "log_std below min_std in runtime normalization")

        mm_range = vvmax - vvmin
        mm_ok = torch.logical_or(~mask_mm, mm_range > 0.0)
        _runtime_assert(torch.all(mm_ok), "Invalid min-max range in runtime normalization")

        log_mm_range = vlog_max - vlog_min
        log_mm_ok = torch.logical_or(~mask_log_mm, log_mm_range > 0.0)
        _runtime_assert(torch.all(log_mm_ok), "Invalid log-min-max range in runtime normalization")

        positive_log_input = torch.logical_or(~mask_any_log, x > 0.0)
        _runtime_assert(torch.all(positive_log_input), "Non-positive value for log-normalized feature")

        out = torch.zeros_like(x)
        out = torch.where(mask_std, (x - vmean) / vstd, out)
        out = torch.where(mask_mm, (x - vvmin) / mm_range, out)

        x_log_input = torch.where(mask_any_log, x, torch.ones_like(x))
        x_log = torch.log10(x_log_input)
        out = torch.where(mask_log_std, (x_log - vlog_mean) / vlog_std, out)
        out = torch.where(mask_log_mm, (x_log - vlog_min) / log_mm_range, out)
        return out

    def _denormalize_group(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        x = x.to(torch.float32)
        method_code = getattr(self, f"{prefix}_method_code")
        mean = getattr(self, f"{prefix}_mean")
        std = getattr(self, f"{prefix}_std")
        vmin = getattr(self, f"{prefix}_min")
        vmax = getattr(self, f"{prefix}_max")
        log_mean = getattr(self, f"{prefix}_log_mean")
        log_std = getattr(self, f"{prefix}_log_std")
        log_min = getattr(self, f"{prefix}_log_min")
        log_max = getattr(self, f"{prefix}_log_max")

        if x.shape[-1] != method_code.numel():
            raise ValueError(f"Expected {method_code.numel()} features, got {x.shape[-1]}")

        vid = self._view(method_code, x.ndim)
        vmean = self._view(mean, x.ndim)
        vstd = self._view(std, x.ndim)
        vvmin = self._view(vmin, x.ndim)
        vvmax = self._view(vmax, x.ndim)
        vlog_mean = self._view(log_mean, x.ndim)
        vlog_std = self._view(log_std, x.ndim)
        vlog_min = self._view(log_min, x.ndim)
        vlog_max = self._view(log_max, x.ndim)

        mask_std = (vid == METHOD_IDS["standard"])
        mask_mm = (vid == METHOD_IDS["min-max"])
        mask_log_std = (vid == METHOD_IDS["log-standard"])
        mask_log_mm = (vid == METHOD_IDS["log-min-max"])

        out = torch.zeros_like(x)
        out = torch.where(mask_std, x * vstd + vmean, out)
        out = torch.where(mask_mm, x * (vvmax - vvmin) + vvmin, out)

        x_log_std_input = torch.where(mask_log_std, x, torch.zeros_like(x))
        y_log_std = torch.pow(10.0, x_log_std_input * vlog_std + vlog_mean)
        out = torch.where(mask_log_std, y_log_std, out)

        x_log_mm_input = torch.where(mask_log_mm, x, torch.zeros_like(x))
        y_log_mm = torch.pow(10.0, x_log_mm_input * (vlog_max - vlog_min) + vlog_min)
        out = torch.where(mask_log_mm, y_log_mm, out)
        return out

    def _normalize_dt(self, dt_sec: torch.Tensor) -> torch.Tensor:
        if dt_sec.ndim == 1:
            dt = dt_sec.unsqueeze(1)
        elif dt_sec.ndim == 3 and dt_sec.shape[-1] == 1:
            dt = dt_sec.squeeze(-1)
        else:
            dt = dt_sec
        dt = dt.to(torch.float32)
        _runtime_assert(torch.all(dt > 0.0), "Physical dt must be > 0")
        dt_log = torch.log10(dt)
        dt_norm = (dt_log - self.dt_log_min) / self.dt_log_range
        finite_ok = torch.all(torch.isfinite(dt_norm))
        in_range = torch.all((dt_norm >= 0.0) & (dt_norm <= 1.0))
        _runtime_assert(finite_ok, "Normalized dt contains non-finite values")
        _runtime_assert(
            in_range,
            "Normalized dt out of range [0, 1]. This indicates dt extrapolation beyond the trained range.",
        )
        return dt_norm

    def forward(self, y_phys: torch.Tensor, dt_sec: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        y_norm = self._normalize_group(y_phys, "species")
        g_norm = self._normalize_group(g_phys, "globals") if self.G > 0 else g_phys.to(torch.float32)
        dt_norm = self._normalize_dt(dt_sec)
        y_pred_norm = self.base(y_norm, dt_norm, g_norm)
        return self._denormalize_group(y_pred_norm, "species")


def _representative_value(method_name: str, stats: Mapping[str, Any]) -> float:
    if method_name == "standard":
        return float(stats["mean"])
    if method_name == "min-max":
        return 0.5 * (float(stats["min"]) + float(stats["max"]))
    if method_name == "log-standard":
        return float(math.pow(10.0, float(stats["log_mean"])))
    if method_name == "log-min-max":
        mid = 0.5 * (float(stats["log_min"]) + float(stats["log_max"]))
        return float(math.pow(10.0, mid))
    raise ValueError(f"Unknown method: {method_name}")


def _write_metadata(
    *,
    out_path: Path,
    species_keys: Sequence[str],
    global_keys: Sequence[str],
    methods: Mapping[str, Any],
    per_key_stats: Mapping[str, Any],
    dt_log_min: float,
    dt_log_max: float,
    config_path: Path,
    manifest_path: Path,
) -> None:
    species_means = {
        key: _representative_value(str(methods[key]), per_key_stats[key])
        for key in species_keys
    }
    global_means = {
        key: _representative_value(str(methods[key]), per_key_stats[key])
        for key in global_keys
    }
    meta = {
        "species_order": list(species_keys),
        "species_mean": species_means,
        "globals_order": list(global_keys),
        "globals_mean": global_means,
        "dt_bounds_sec": {
            "min": float(math.pow(10.0, dt_log_min)),
            "max": float(math.pow(10.0, dt_log_max)),
        },
        "sources": {
            "config_path": str(config_path),
            "normalization_path": str(manifest_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    model_dir = _resolve_model_dir(DEFAULT_MODEL_DIR)
    cfg_path = model_dir / CONFIG_NAME
    out_model_path = model_dir / PHYS_MODEL_FILENAME
    out_meta_path = model_dir / PHYS_METADATA_FILENAME

    cfg, manifest, manifest_path = _resolve_config_and_manifest(model_dir)
    data_cfg = cfg["data"]
    species_keys = [str(x) for x in _require_list(data_cfg, "species_variables")]
    global_keys = [str(x) for x in _require_list(data_cfg, "global_variables")]

    per_key_stats = manifest.get("per_key_stats")
    methods = manifest.get("normalization_methods")
    min_std = float(manifest.get("min_std"))
    dt_spec = manifest.get("dt")
    if not isinstance(per_key_stats, Mapping) or not isinstance(methods, Mapping):
        raise ValueError("Invalid normalization manifest: missing per_key_stats/normalization_methods")
    if not isinstance(dt_spec, Mapping):
        raise ValueError("Invalid normalization manifest: missing dt spec")
    if min_std <= 0.0:
        raise ValueError("Invalid normalization manifest: min_std must be > 0")

    dt_log_min = float(dt_spec["log_min"])
    dt_log_max = float(dt_spec["log_max"])
    if dt_log_max <= dt_log_min:
        raise ValueError("Invalid normalization manifest: dt log range must be positive")

    base_model = create_model(cfg).cpu().eval()
    ckpt_path = _find_checkpoint(model_dir)
    _load_weights(base_model, ckpt_path)
    _freeze_for_inference(base_model)

    species_tensors = _build_group_tensors(
        keys=species_keys,
        methods=methods,
        per_key_stats=per_key_stats,
        min_std=min_std,
    )
    global_tensors = _build_group_tensors(
        keys=global_keys,
        methods=methods,
        per_key_stats=per_key_stats,
        min_std=min_std,
    )

    wrapped_model = PhysicalModelWrapper(
        base_model=base_model,
        species_keys=species_keys,
        global_keys=global_keys,
        species_tensors=species_tensors,
        global_tensors=global_tensors,
        min_std=min_std,
        dt_log_min=dt_log_min,
        dt_log_max=dt_log_max,
    ).cpu().eval()
    _freeze_for_inference(wrapped_model)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)
    B = 2
    y_ex = torch.ones(B, len(species_keys), dtype=torch.float32)
    dt_ex = torch.ones(B, 1, dtype=torch.float32)
    g_ex = torch.ones(B, len(global_keys), dtype=torch.float32) if global_keys else torch.empty(B, 0, dtype=torch.float32)

    ep = torch.export.export(
        wrapped_model,
        (y_ex, dt_ex, g_ex),
        dynamic_shapes=(
            {0: Bdim},
            {0: Bdim},
            {0: Bdim},
        ),
        strict=False,
    )

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_model_path)

    _write_metadata(
        out_path=out_meta_path,
        species_keys=species_keys,
        global_keys=global_keys,
        methods=methods,
        per_key_stats=per_key_stats,
        dt_log_min=dt_log_min,
        dt_log_max=dt_log_max,
        config_path=cfg_path,
        manifest_path=manifest_path,
    )

    print(f"Model directory: {model_dir}")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Wrote physical model: {out_model_path}")
    print(f"Wrote metadata: {out_meta_path}")
    print(f"Tip: override model dir via {MODEL_DIR_ENV}=<path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
