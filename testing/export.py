#!/usr/bin/env python3
"""
export.py - SINGLE dynamic-batch torch.export for PHYSICAL-space 1-step model.

Hard requirements:
- Exports exactly ONE artifact (no static batch menu, no extra exports).
- Dynamic batch dimension B for all inputs: y_phys (B,S), dt_seconds (B,), g_phys (B,G).
- This REQUIRES src/model.py to be export-safe:
    * NO int(t.shape[0]), NO t.shape[0].item(), NO len(tensor) on dynamic dims
    * shape checks must use torch._assert, not Python if + int(...)
  Otherwise torch.export will correctly reject dynamic B.

Output:
  {RUN_DIR}/export_cpu_dynB_1step_phys.pt2
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.export import Dim



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import create_model  # noqa: E402


# -----------------------------
# GLOBALS
# -----------------------------
RUN_DIR = (ROOT / "models" / "fixed").resolve()
OUT_PATH = (RUN_DIR / "export_cpu_dynB_1step_phys.pt2").resolve()

# Example batch used for tracing (B will be dynamic in the exported program).
# Keep > 1 to avoid accidental “specialize to 1” patterns in user code.
B_EXAMPLE = 4

# Dynamic batch range guards
B_MIN = 1
B_MAX = 4096

# Force CPU-only single export
DEVICE = "cpu"
DTYPE = torch.float32


# -----------------------------
# Helpers
# -----------------------------
def load_config(run_dir: Path) -> Tuple[dict, Path]:
    candidates = [
        run_dir / "config.resolved.json",
        run_dir / "config.json",
        ROOT / "config.json",
    ]
    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError("Could not find config.json / config.resolved.json in run dir or repo root.")
    return json.loads(cfg_path.read_text()), cfg_path


def find_checkpoint(run_dir: Path) -> Path:
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    ckpts2 = sorted(run_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts2:
        return ckpts2[0]
    raise FileNotFoundError(f"No checkpoint found under {run_dir} or {run_dir / 'checkpoints'}.")


def _strip_prefixes(key: str) -> str:
    prefixes = (
        "state_dict.",
        "model.",
        "module.",
        "_orig_mod.",
        "model._orig_mod.",
        "module.model.",
        "module._orig_mod.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {_strip_prefixes(k): v for k, v in state.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in cleaned.items() if k in model_sd}
    missing = [k for k in model_sd.keys() if k not in filtered]
    if missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(f"Checkpoint missing {len(missing)} keys (showing 20):\n  {preview}\nckpt={ckpt_path}")
    model.load_state_dict(filtered, strict=True)


def prepare_model(model: nn.Module) -> nn.Module:
    #model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def find_normalization_manifest(cfg: dict) -> Path:
    paths = cfg.get("paths", {}) or {}
    if isinstance(paths, dict):
        processed = paths.get("processed_dir")
        if isinstance(processed, str) and processed.strip():
            p = Path(processed).expanduser()
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            cand = p / "normalization.json"
            if cand.exists():
                return cand
    cand = (ROOT / "data" / "processed" / "normalization.json").resolve()
    if cand.exists():
        return cand
    raise FileNotFoundError("normalization.json not found (checked cfg.paths.processed_dir and ROOT/data/processed).")


# -----------------------------
# Baked normalizer (expects manifest: per_key_stats, methods, dt)
# -----------------------------
_SUPPORTED = {"standard", "min-max", "log-standard", "log-min-max"}


class BakedNormalizer(nn.Module):
    def __init__(
        self,
        species_vars: List[str],
        global_vars: List[str],
        species_methods: List[str],
        global_methods: List[str],
        *,
        s_log_mean: torch.Tensor,
        s_log_std: torch.Tensor,
        s_log_min: torch.Tensor,
        s_log_max: torch.Tensor,
        s_eps: torch.Tensor,
        g_mean: torch.Tensor,
        g_std: torch.Tensor,
        g_min: torch.Tensor,
        g_max: torch.Tensor,
        g_log_mean: torch.Tensor,
        g_log_std: torch.Tensor,
        g_log_min: torch.Tensor,
        g_log_max: torch.Tensor,
        g_eps: torch.Tensor,
        dt_log_min: float,
        dt_log_max: float,
        dt_eps: float,
    ):
        super().__init__()
        self.species_vars = tuple(species_vars)
        self.global_vars = tuple(global_vars)
        self.species_methods = tuple(species_methods)
        self.global_methods = tuple(global_methods)

        self.register_buffer("s_log_mean", s_log_mean.float())
        self.register_buffer("s_log_std", s_log_std.float())
        self.register_buffer("s_log_min", s_log_min.float())
        self.register_buffer("s_log_max", s_log_max.float())
        self.register_buffer("s_eps", s_eps.float())

        self.register_buffer("g_mean", g_mean.float())
        self.register_buffer("g_std", g_std.float())
        self.register_buffer("g_min", g_min.float())
        self.register_buffer("g_max", g_max.float())
        self.register_buffer("g_log_mean", g_log_mean.float())
        self.register_buffer("g_log_std", g_log_std.float())
        self.register_buffer("g_log_min", g_log_min.float())
        self.register_buffer("g_log_max", g_log_max.float())
        self.register_buffer("g_eps", g_eps.float())

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
        dt_f = dt_seconds.to(torch.float32)
        dt_f = torch.clamp_min(dt_f, self.dt_eps)
        log_dt = torch.log10(dt_f)
        dt_norm = (log_dt - self.dt_log_min) / (self.dt_log_max - self.dt_log_min)
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)
        return dt_norm.to(dt_seconds.dtype)

    def normalize_species(self, y_phys: torch.Tensor) -> torch.Tensor:
        y_f = torch.clamp_min(y_phys.to(torch.float32), self.s_eps)
        out = []
        for j, m in enumerate(self.species_methods):
            xj = y_f[:, j]
            if m == "log-standard":
                zj = (torch.log10(torch.clamp_min(xj, self.s_eps[j])) - self.s_log_mean[j]) / self.s_log_std[j]
            elif m == "log-min-max":
                zj = (torch.log10(torch.clamp_min(xj, self.s_eps[j])) - self.s_log_min[j]) / (
                    self.s_log_max[j] - self.s_log_min[j]
                )
            else:
                raise RuntimeError(f"Unsupported species method: {m}")
            out.append(zj)
        z = torch.stack(out, dim=-1)
        return z.to(y_phys.dtype)

    def denormalize_species(self, y_z: torch.Tensor) -> torch.Tensor:
        z_f = y_z.to(torch.float32)
        out = []
        for j, m in enumerate(self.species_methods):
            zj = z_f[:, j]
            if m == "log-standard":
                logx = zj * self.s_log_std[j] + self.s_log_mean[j]
                xj = torch.pow(10.0, logx)
            elif m == "log-min-max":
                logx = zj * (self.s_log_max[j] - self.s_log_min[j]) + self.s_log_min[j]
                xj = torch.pow(10.0, logx)
            else:
                raise RuntimeError(f"Unsupported species method: {m}")
            out.append(xj)
        y = torch.stack(out, dim=-1)
        return y.to(y_z.dtype)

    def normalize_globals(self, g_phys: torch.Tensor) -> torch.Tensor:
        if self.G == 0:
            return g_phys
        g_f = g_phys.to(torch.float32)
        out = []
        for j, m in enumerate(self.global_methods):
            xj = g_f[:, j]
            eps = self.g_eps[j]
            if m == "min-max":
                zj = (xj - self.g_min[j]) / (self.g_max[j] - self.g_min[j])
            elif m == "standard":
                zj = (xj - self.g_mean[j]) / self.g_std[j]
            elif m == "log-standard":
                zj = (torch.log10(torch.clamp_min(xj, eps)) - self.g_log_mean[j]) / self.g_log_std[j]
            elif m == "log-min-max":
                zj = (torch.log10(torch.clamp_min(xj, eps)) - self.g_log_min[j]) / (
                    self.g_log_max[j] - self.g_log_min[j]
                )
            else:
                raise RuntimeError(f"Unsupported global method: {m}")
            out.append(zj)
        z = torch.stack(out, dim=-1)
        return z.to(g_phys.dtype)


def build_baked_normalizer(manifest: dict, *, species_vars: List[str], global_vars: List[str]) -> BakedNormalizer:
    methods: Dict[str, str] = (manifest.get("methods") or manifest.get("normalization_methods") or {})
    stats: Dict[str, dict] = manifest["per_key_stats"]
    eps_global = float(manifest.get("epsilon", 1e-30))

    def m(k: str) -> str:
        v = methods.get(k)
        if v not in _SUPPORTED:
            raise ValueError(f"Unsupported/missing normalization method for {k}: {v}")
        return v

    s_methods = [m(k) for k in species_vars]
    g_methods = [m(k) for k in global_vars]

    s_log_mean = torch.tensor([float(stats[k].get("log_mean", 0.0)) for k in species_vars])
    s_log_std = torch.tensor([float(stats[k].get("log_std", 1.0)) for k in species_vars])
    s_log_min = torch.tensor([float(stats[k].get("log_min", 0.0)) for k in species_vars])
    s_log_max = torch.tensor([float(stats[k].get("log_max", 1.0)) for k in species_vars])
    s_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in species_vars])

    g_mean = torch.tensor([float(stats[k].get("mean", 0.0)) for k in global_vars])
    g_std = torch.tensor([float(stats[k].get("std", 1.0)) for k in global_vars])
    g_min = torch.tensor([float(stats[k].get("min", 0.0)) for k in global_vars])
    g_max = torch.tensor([float(stats[k].get("max", 1.0)) for k in global_vars])
    g_log_mean = torch.tensor([float(stats[k].get("log_mean", 0.0)) for k in global_vars])
    g_log_std = torch.tensor([float(stats[k].get("log_std", 1.0)) for k in global_vars])
    g_log_min = torch.tensor([float(stats[k].get("log_min", 0.0)) for k in global_vars])
    g_log_max = torch.tensor([float(stats[k].get("log_max", 1.0)) for k in global_vars])
    g_eps = torch.tensor([float(stats[k].get("epsilon", eps_global)) for k in global_vars])

    dt = manifest.get("dt") or {}
    dt_log_min = float(dt["log_min"])
    dt_log_max = float(dt["log_max"])
    dt_eps = float(manifest.get("epsilon", 1e-30))

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
        dt_log_min=dt_log_min,
        dt_log_max=dt_log_max,
        dt_eps=dt_eps,
    )


# -----------------------------
# Exportable physical-space 1-step module
# -----------------------------
class OneStepARPhysical(nn.Module):
    def __init__(self, base: nn.Module, norm: BakedNormalizer, *, dt_as_column: bool):
        super().__init__()
        self.base = base
        self.norm = norm
        self.dt_as_column = bool(dt_as_column)

        self.S = int(getattr(base, "S"))
        self.G = int(getattr(base, "G"))
        if self.S != norm.S:
            raise RuntimeError(f"S mismatch: model={self.S} norm={norm.S}")
        if self.G != norm.G:
            raise RuntimeError(f"G mismatch: model={self.G} norm={norm.G}")

    def forward(self, y_phys: torch.Tensor, dt_seconds: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        y_z = self.norm.normalize_species(y_phys)
        dt_norm = self.norm.normalize_dt_seconds(dt_seconds)
        if self.dt_as_column:
            dt_norm = dt_norm.unsqueeze(-1)  # (B,) -> (B,1)
        g_z = self.norm.normalize_globals(g_phys)

        y_next_z = self.base.forward_step(y_z, dt_norm, g_z)
        y_next = self.norm.denormalize_species(y_next_z)
        return y_next


def make_example_inputs(step: OneStepARPhysical, B: int, device: str, dtype: torch.dtype):
    S = step.S
    G = step.G
    norm = step.norm

    u = torch.rand(B, S, device=device, dtype=torch.float32)
    logy = norm.s_log_min.to(device) + u * (norm.s_log_max.to(device) - norm.s_log_min.to(device))
    y = torch.pow(10.0, logy).to(dtype)

    u_dt = torch.rand(B, device=device, dtype=torch.float32)
    logdt = norm.dt_log_min.to(device) + u_dt * (norm.dt_log_max.to(device) - norm.dt_log_min.to(device))
    dt_seconds = torch.pow(10.0, logdt).to(dtype)

    if G == 0:
        g = torch.empty(B, 0, device=device, dtype=dtype)
    else:
        cols = []
        for j, m in enumerate(norm.global_methods):
            if m in ("min-max", "standard"):
                uj = torch.rand(B, device=device, dtype=torch.float32)
                xj = norm.g_min[j].to(device) + uj * (norm.g_max[j].to(device) - norm.g_min[j].to(device))
            elif m in ("log-min-max", "log-standard"):
                uj = torch.rand(B, device=device, dtype=torch.float32)
                logx = norm.g_log_min[j].to(device) + uj * (norm.g_log_max[j].to(device) - norm.g_log_min[j].to(device))
                xj = torch.pow(10.0, logx)
            else:
                raise RuntimeError(f"Unsupported global method: {m}")
            cols.append(xj)
        g = torch.stack(cols, dim=-1).to(dtype)

    return y, dt_seconds, g


def infer_dt_as_column(base: nn.Module, norm: BakedNormalizer) -> bool:
    """
    Eager probe (NOT captured) to decide whether base.forward_step expects dt_norm as (B,) or (B,1).
    This avoids putting try/except or data-dependent branches inside the exported graph.
    """
    Bp = 3
    # y_z and g_z should match what we pass at export time
    y_phys = torch.rand(Bp, norm.S, dtype=torch.float32)
    dt_seconds = torch.ones(Bp, dtype=torch.float32)
    if norm.G == 0:
        g_phys = torch.empty(Bp, 0, dtype=torch.float32)
    else:
        g_phys = torch.rand(Bp, norm.G, dtype=torch.float32)

    y_z = norm.normalize_species(y_phys)
    dt_norm = norm.normalize_dt_seconds(dt_seconds)
    g_z = norm.normalize_globals(g_phys)

    with torch.no_grad():
        try:
            _ = base.forward_step(y_z, dt_norm, g_z)  # (B,)
            return False
        except Exception as e1:
            try:
                _ = base.forward_step(y_z, dt_norm.unsqueeze(-1), g_z)  # (B,1)
                return True
            except Exception as e2:
                raise RuntimeError(
                    "Could not infer dt shape for base.forward_step. It rejected both dt_norm (B,) and (B,1).\n"
                    f"First error (B,): {type(e1).__name__}: {e1}\n"
                    f"Second error (B,1): {type(e2).__name__}: {e2}"
                )


def verify_dynB(ep: torch.export.ExportedProgram, step_ref: OneStepARPhysical) -> None:
    """
    Fail hard unless the exported program runs with multiple batch sizes (proof of dynamic B).
    """
    m = ep.module()
    #m.eval()

    for B in (1, 7):
        y, dt, g = make_example_inputs(step_ref, B, DEVICE, DTYPE)
        with torch.no_grad():
            out = m(y, dt, g)
        if out.shape[0] != B:
            raise RuntimeError(f"dynB verification failed: ran with B={B} but output batch was {out.shape[0]}.")


def main() -> None:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"RUN_DIR not found: {RUN_DIR}")

    cfg, _cfg_path = load_config(RUN_DIR)
    ckpt = find_checkpoint(RUN_DIR)

    base = create_model(cfg)
    load_weights(base, ckpt)
    base = prepare_model(base.to(device=DEVICE, dtype=DTYPE))

    manifest_path = find_normalization_manifest(cfg)
    manifest = json.loads(manifest_path.read_text())
    species_vars = list(manifest.get("species_variables", []))
    global_vars = list(manifest.get("global_variables", [])) or []
    if not species_vars:
        raise RuntimeError("normalization.json missing species_variables")

    S = int(getattr(base, "S"))
    G = int(getattr(base, "G"))
    if S != len(species_vars):
        raise RuntimeError(f"S mismatch: model.S={S} manifest.S={len(species_vars)}")
    if G != len(global_vars):
        raise RuntimeError(f"G mismatch: model.G={G} manifest.G={len(global_vars)}")

    norm = build_baked_normalizer(manifest, species_vars=species_vars, global_vars=global_vars)

    dt_as_column = infer_dt_as_column(base, norm)
    step = OneStepARPhysical(base, norm, dt_as_column=dt_as_column).to(device=DEVICE, dtype=DTYPE)
    step = prepare_model(step)

    args = make_example_inputs(step, B_EXAMPLE, DEVICE, DTYPE)

    B = Dim("B", min=B_MIN, max=B_MAX)
    dynamic_shapes = (
        {0: B},  # y_phys: (B,S)
        {0: B},  # dt_seconds: (B,)
        {0: B},  # g_phys: (B,G)
    )

    # Single export attempt. No fallbacks. No alternate artifacts.
    ep = torch.export.export(step, args, dynamic_shapes=dynamic_shapes, strict=False)

    # Refuse to save unless it truly runs with different B.
    verify_dynB(ep, step)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
