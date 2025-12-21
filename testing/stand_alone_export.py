#!/usr/bin/env python3
"""
Standalone Flow-map AE export (physical I/O, CPU only).

Exports a model that:
    Inputs:
        y_phys : [B, S_in]       physical species (same order as data.species_variables)
        dt_sec : [B, 1]          Δt in physical seconds
        g_phys : [B, G]          physical globals (same order as data.global_variables)
    Output:
        y_next_phys : [B, 1, S_out] physical species at t + Δt

No normalization files needed at inference time.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODEL_DIR = ROOT / "models" / "subset_big"

CPU_OUT = MODEL_DIR / "standalone_phys_k1_cpu.pt2"
MIN_BATCH, MAX_BATCH = 1, 4096

# Metadata file with species/global order + typical values
METADATA_OUT = MODEL_DIR / "standalone_phys_metadata.json"

os.chdir(ROOT)
sys.path.insert(0, str(SRC))

from model import create_model  # type: ignore  # noqa: E402

try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    candidates = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "trial_config.final.json",
        MODEL_DIR / "train_config.json",
    ]
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find any config JSON in {MODEL_DIR}")


def _load_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    data_root = Path(paths["processed_data_dir"]).expanduser().resolve()
    manifest_path = data_root / "normalization.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing normalization manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_variable_lists(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> tuple[List[str], List[str], List[str]]:
    data_cfg = cfg.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    target_vars = list(data_cfg.get("target_species") or species_vars)
    global_vars = list(data_cfg.get("global_variables") or [])

    if not species_vars:
        meta = manifest.get("meta", {})
        species_vars = list(meta.get("species_variables") or [])
        target_vars = list(meta.get("target_species") or species_vars)
        global_vars = list(meta.get("global_variables") or [])

    if not species_vars:
        raise RuntimeError("No species variables found in config or manifest.")

    return species_vars, target_vars, global_vars


def find_ckpt(directory: Path) -> Path:
    d = Path(directory)
    if (d / "best.ckpt").exists():
        return d / "best.ckpt"
    if (d / "last.ckpt").exists():
        return d / "last.ckpt"
    ckpts = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint found in {d}")


def load_weights(model: nn.Module, ckpt_path: Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        state = (
            payload.get("state_dict")
            or payload.get("model_state_dict")
            or payload.get("model")
            or payload.get("ema_model")
            or {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        )
    else:
        state = payload
    clean = {}
    for k, v in state.items():
        kk = k
        for prefix in ("model.", "module.", "_orig_mod."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        clean[kk] = v
    model.load_state_dict(clean, strict=False)


def optimize_inference(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


# ---------------------------------------------------------------------
# Standalone wrapper with baked-in normalization
# ---------------------------------------------------------------------

class StandaloneFlowMapAE(nn.Module):
    """
    Operates directly on physical inputs and returns physical outputs.

    Forward:
        y_phys : [B, S_in]
        dt_sec : [B, 1] (seconds)
        g_phys : [B, G]
    ->  y_next_phys : [B, 1, S_out]
    """

    METHOD_IDS = {
        "standard": 0,
        "min-max": 1,
        "log-standard": 2,
        "log-min-max": 3,
    }

    def __init__(
        self,
        base_model: nn.Module,
        manifest: Dict[str, Any],
        species_in: Sequence[str],
        species_out: Sequence[str],
        globals_: Sequence[str],
    ) -> None:
        super().__init__()
        self.base = base_model

        self.S_in = int(getattr(base_model, "S_in"))
        self.S_out = int(getattr(base_model, "S_out"))
        self.G = int(getattr(base_model, "G", 0))

        if self.S_in != len(species_in):
            raise ValueError(f"S_in={self.S_in} but len(species_in)={len(species_in)}")
        if self.S_out != len(species_out):
            raise ValueError(f"S_out={self.S_out} but len(species_out)={len(species_out)}")
        if self.G != len(globals_):
            raise ValueError(f"G={self.G} but len(globals_)={len(globals_)}")

        self.species_in_names = list(species_in)
        self.species_out_names = list(species_out)
        self.global_names = list(globals_)

        self._per_key_stats = dict(manifest.get("per_key_stats", {}))
        self._methods = dict(manifest.get("normalization_methods", {}))
        self.epsilon = float(manifest.get("epsilon", 1e-30))
        self.min_std = float(manifest.get("min_std", 1e-10))

        # dt spec
        dt_spec = manifest.get("dt", None)
        if dt_spec is None:
            raise ValueError("dt normalization spec missing in manifest")
        log_min = float(dt_spec["log_min"])
        log_max = float(dt_spec["log_max"])
        range_log = max(log_max - log_min, 1e-12)
        dt_min_phys = max(10.0 ** log_min, float(self.epsilon))
        dt_max_phys = 10.0 ** log_max

        self.register_buffer("dt_log_min", torch.tensor(log_min, dtype=torch.float32), persistent=True)
        self.register_buffer("dt_log_range", torch.tensor(range_log, dtype=torch.float32), persistent=True)
        self.register_buffer("dt_min_phys", torch.tensor(dt_min_phys, dtype=torch.float32), persistent=True)
        self.register_buffer("dt_max_phys", torch.tensor(dt_max_phys, dtype=torch.float32), persistent=True)

        # Stats for species_in, species_out, globals
        (m_in, stats_in) = self._build_stats_for_keys(self.species_in_names)
        (m_out, stats_out) = self._build_stats_for_keys(self.species_out_names)
        (m_g, stats_g) = self._build_stats_for_keys(self.global_names)

        # Species input stats
        self.register_buffer("spec_in_method_id", m_in, persistent=True)
        self.register_buffer("spec_in_mean", stats_in["mean"], persistent=True)
        self.register_buffer("spec_in_std", stats_in["std"], persistent=True)
        self.register_buffer("spec_in_min", stats_in["min"], persistent=True)
        self.register_buffer("spec_in_max", stats_in["max"], persistent=True)
        self.register_buffer("spec_in_log_mean", stats_in["log_mean"], persistent=True)
        self.register_buffer("spec_in_log_std", stats_in["log_std"], persistent=True)
        self.register_buffer("spec_in_log_min", stats_in["log_min"], persistent=True)
        self.register_buffer("spec_in_log_max", stats_in["log_max"], persistent=True)

        # Species output stats
        self.register_buffer("spec_out_method_id", m_out, persistent=True)
        self.register_buffer("spec_out_mean", stats_out["mean"], persistent=True)
        self.register_buffer("spec_out_std", stats_out["std"], persistent=True)
        self.register_buffer("spec_out_min", stats_out["min"], persistent=True)
        self.register_buffer("spec_out_max", stats_out["max"], persistent=True)
        self.register_buffer("spec_out_log_mean", stats_out["log_mean"], persistent=True)
        self.register_buffer("spec_out_log_std", stats_out["log_std"], persistent=True)
        self.register_buffer("spec_out_log_min", stats_out["log_min"], persistent=True)
        self.register_buffer("spec_out_log_max", stats_out["log_max"], persistent=True)

        # Globals stats
        if self.G > 0:
            self.register_buffer("glob_method_id", m_g, persistent=True)
            self.register_buffer("glob_mean", stats_g["mean"], persistent=True)
            self.register_buffer("glob_std", stats_g["std"], persistent=True)
            self.register_buffer("glob_min", stats_g["min"], persistent=True)
            self.register_buffer("glob_max", stats_g["max"], persistent=True)
            self.register_buffer("glob_log_mean", stats_g["log_mean"], persistent=True)
            self.register_buffer("glob_log_std", stats_g["log_std"], persistent=True)
            self.register_buffer("glob_log_min", stats_g["log_min"], persistent=True)
            self.register_buffer("glob_log_max", stats_g["log_max"], persistent=True)

    # ---- stats helpers ----

    def _build_stats_for_keys(
        self,
        keys: Sequence[str],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = len(keys)
        methods = torch.empty(n, dtype=torch.long)
        mean = torch.empty(n, dtype=torch.float32)
        std = torch.empty(n, dtype=torch.float32)
        min_val = torch.empty(n, dtype=torch.float32)
        max_val = torch.empty(n, dtype=torch.float32)
        log_mean = torch.empty(n, dtype=torch.float32)
        log_std = torch.empty(n, dtype=torch.float32)
        log_min = torch.empty(n, dtype=torch.float32)
        log_max = torch.empty(n, dtype=torch.float32)

        for i, key in enumerate(keys):
            method_str = str(self._methods.get(key, "log-standard"))
            method_id = self.METHOD_IDS.get(method_str)
            if method_id is None:
                raise ValueError(f"Unknown normalization method '{method_str}' for key '{key}'")
            methods[i] = method_id

            stats = self._per_key_stats.get(key, {})
            mean[i] = float(stats.get("mean", 0.0))
            std[i] = max(float(stats.get("std", 1.0)), self.min_std)
            min_val[i] = float(stats.get("min", 0.0))
            max_val[i] = float(stats.get("max", 1.0))
            log_mean[i] = float(stats.get("log_mean", 0.0))
            log_std[i] = max(float(stats.get("log_std", 1.0)), self.min_std)
            log_min[i] = float(stats.get("log_min", -3.0))
            log_max[i] = float(stats.get("log_max", 8.0))

        stats_tensors = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "log_mean": log_mean,
            "log_std": log_std,
            "log_min": log_min,
            "log_max": log_max,
        }
        return methods, stats_tensors

    @staticmethod
    def _view_for_broadcast(v: torch.Tensor, x_ndim: int) -> torch.Tensor:
        return v.view(*([1] * (x_ndim - 1)), -1)

    def _normalize_group(
        self,
        x: torch.Tensor,
        method_id: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        log_min: torch.Tensor,
        log_max: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor for normalization group")
        if x.shape[-1] != method_id.numel():
            raise ValueError(f"Shape mismatch: got {x.shape[-1]} cols vs {method_id.numel()} keys")

        x = x.to(torch.float32)

        vid = self._view_for_broadcast(method_id, x.ndim)
        vmean = self._view_for_broadcast(mean, x.ndim)
        vstd = self._view_for_broadcast(std, x.ndim)
        vmin = self._view_for_broadcast(min_val, x.ndim)
        vmax = self._view_for_broadcast(max_val, x.ndim)
        vlog_mean = self._view_for_broadcast(log_mean, x.ndim)
        vlog_std = self._view_for_broadcast(log_std, x.ndim)
        vlog_min = self._view_for_broadcast(log_min, x.ndim)
        vlog_max = self._view_for_broadcast(log_max, x.ndim)

        m_std = (vid == 0).to(x.dtype)
        m_minmax = (vid == 1).to(x.dtype)
        m_logstd = (vid == 2).to(x.dtype)
        m_logminmax = (vid == 3).to(x.dtype)

        # standard
        x_std = (x - vmean) / vstd

        # min-max
        range_mm = (vmax - vmin).clamp_min(1e-12)
        x_mm = (x - vmin) / range_mm

        # log-standard
        x_log = torch.log10(torch.clamp(x, min=self.epsilon))
        x_logstd = (x_log - vlog_mean) / vlog_std

        # log-min-max
        range_lmm = (vlog_max - vlog_min).clamp_min(1e-12)
        x_lmm = (x_log - vlog_min) / range_lmm

        return (
            x_std * m_std +
            x_mm * m_minmax +
            x_logstd * m_logstd +
            x_lmm * m_logminmax
        )

    def _denormalize_group(
        self,
        x: torch.Tensor,
        method_id: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        log_min: torch.Tensor,
        log_max: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Expected at least 2D tensor for denormalization group")
        if x.shape[-1] != method_id.numel():
            raise ValueError(f"Shape mismatch: got {x.shape[-1]} cols vs {method_id.numel()} keys")

        x = x.to(torch.float32)

        vid = self._view_for_broadcast(method_id, x.ndim)
        vmean = self._view_for_broadcast(mean, x.ndim)
        vstd = self._view_for_broadcast(std, x.ndim)
        vmin = self._view_for_broadcast(min_val, x.ndim)
        vmax = self._view_for_broadcast(max_val, x.ndim)
        vlog_mean = self._view_for_broadcast(log_mean, x.ndim)
        vlog_std = self._view_for_broadcast(log_std, x.ndim)
        vlog_min = self._view_for_broadcast(log_min, x.ndim)
        vlog_max = self._view_for_broadcast(log_max, x.ndim)

        m_std = (vid == 0).to(x.dtype)
        m_minmax = (vid == 1).to(x.dtype)
        m_logstd = (vid == 2).to(x.dtype)
        m_logminmax = (vid == 3).to(x.dtype)

        # standard
        y_std = x * vstd + vmean

        # min-max
        range_mm = (vmax - vmin).clamp_min(1e-12)
        y_mm = x * range_mm + vmin

        # log-standard
        log_vals_std = x * vlog_std + vlog_mean
        y_logstd = torch.pow(10.0, log_vals_std)

        # log-min-max
        range_lmm = (vlog_max - vlog_min).clamp_min(1e-12)
        log_vals_lmm = x * range_lmm + vlog_min
        y_lmm = torch.pow(10.0, log_vals_lmm)

        return (
            y_std * m_std +
            y_mm * m_minmax +
            y_logstd * m_logstd +
            y_lmm * m_logminmax
        )

    def _normalize_dt(self, dt_sec: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(dt_sec):
            dt_sec = dt_sec.float()

        dt = dt_sec
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)  # [B] -> [B,1]
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)   # [B,1,1] -> [B,1]

        dt_clamped = dt.clamp(min=self.dt_min_phys, max=self.dt_max_phys)
        dt_log = torch.log10(dt_clamped)
        dt_norm = (dt_log - self.dt_log_min) / self.dt_log_range
        return dt_norm.clamp_(0.0, 1.0)

    # ---- forward ----

    def forward(self, y_phys: torch.Tensor, dt_sec: torch.Tensor, g_phys: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        y_norm = self._normalize_group(
            y_phys,
            self.spec_in_method_id,
            self.spec_in_mean,
            self.spec_in_std,
            self.spec_in_min,
            self.spec_in_max,
            self.spec_in_log_mean,
            self.spec_in_log_std,
            self.spec_in_log_min,
            self.spec_in_log_max,
        )

        if self.G > 0:
            g_norm = self._normalize_group(
                g_phys,
                self.glob_method_id,
                self.glob_mean,
                self.glob_std,
                self.glob_min,
                self.glob_max,
                self.glob_log_mean,
                self.glob_log_std,
                self.glob_log_min,
                self.glob_log_max,
            )
        else:
            g_norm = g_phys.to(torch.float32)

        dt_norm = self._normalize_dt(dt_sec)

        # model works in normalized z-space; expects K dimension, here K=1
        y_pred_norm = self.base(y_norm, dt_norm, g_norm)  # [B,1,S_out]

        # denormalize outputs
        y_next_phys = self._denormalize_group(
            y_pred_norm,
            self.spec_out_method_id,
            self.spec_out_mean,
            self.spec_out_std,
            self.spec_out_min,
            self.spec_out_max,
            self.spec_out_log_mean,
            self.spec_out_log_std,
            self.spec_out_log_min,
            self.spec_out_log_max,
        )
        return y_next_phys


# ---------------------------------------------------------------------
# CPU export only
# ---------------------------------------------------------------------

def export_cpu(model: nn.Module) -> None:
    print("=" * 80)
    print("Exporting standalone CPU model (K=1, physical I/O)")
    print("=" * 80)

    device = "cpu"
    model = optimize_inference(model.to(device))

    S_in = int(getattr(model, "S_in"))
    G = int(getattr(model, "G", 0) or 0)

    Bdim = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)

    B = 2
    y_phys = torch.ones(B, S_in, dtype=torch.float32, device=device)   # [B,S]
    dt_sec = torch.ones(B, 1, dtype=torch.float32, device=device)      # [B,1] (K=1)
    g_phys = (
        torch.ones(B, G, dtype=torch.float32, device=device)
        if G > 0 else
        torch.empty(B, 0, dtype=torch.float32, device=device)
    )

    ep = torch.export.export(
        model,
        (y_phys, dt_sec, g_phys),
        dynamic_shapes=(
            {0: Bdim},   # y_phys: [B,S]
            {0: Bdim},   # dt_sec: [B,1]
            {0: Bdim},   # g_phys: [B,G]
        ),
    )

    CPU_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, CPU_OUT)
    print(f"Wrote {CPU_OUT}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Standalone FlowMap export (CPU, physical input/output)")
    print("=" * 80)
    print(f"Model dir: {MODEL_DIR}")

    cfg = _load_config()
    manifest = _load_manifest(cfg)
    species_in, species_out, globals_ = _get_variable_lists(cfg, manifest)

    base = create_model(cfg).eval().cpu()
    ckpt = find_ckpt(MODEL_DIR)
    print(f"Loading checkpoint: {ckpt}")
    load_weights(base, ckpt)

    standalone = StandaloneFlowMapAE(base, manifest, species_in, species_out, globals_)
    export_cpu(standalone)

    # ------------ write metadata (species/global order + means) --------------
    per_key = manifest["per_key_stats"]
    norm_methods = manifest.get("normalization_methods", {})

    # Species: use linear mean directly
    species_means = {
        name: per_key[name].get("mean", 0.0)
        for name in species_in
    }

    # Globals: prefer linear mean; else if log-normalized and log_mean exists, use 10**log_mean;
    # otherwise fall back to midpoint of [min, max].
    globals_means = {}
    for name in globals_:
        stats = per_key[name]
        method = str(norm_methods.get(name, ""))
        if "mean" in stats:
            globals_means[name] = stats["mean"]
        elif method.startswith("log-") and "log_mean" in stats:
            globals_means[name] = float(10.0 ** stats["log_mean"])
        else:
            globals_means[name] = 0.5 * (stats["min"] + stats["max"])

    meta = {
        "species_order": species_in,
        "species_mean": species_means,
        "globals_order": globals_,
        "globals_mean": globals_means,
    }

    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_OUT.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {METADATA_OUT}")
    # -------------------------------------------------------------------------

    print("Done.")


if __name__ == "__main__":
    main()
