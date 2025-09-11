#!/usr/bin/env python3
"""
Auto-export Flow-map DeepONet from a checkpoint.

What this does
--------------
- Infers architecture from the checkpoint (p, branch width/depth, trunk layers, S, G).
- Reads non-shape flags (predict_delta, activation, dropouts, trunk_dedup) from config
  files that live in MODEL_DIR (config.json|jsonc|config.snapshot.json).
- Hydrates data variable names from normalization.json (if present); otherwise synthesizes.
- Exports a dynamic (B,K) PT2 program and a static K=1 variant.
- Writes MODEL_DIR/config.snapshot.json capturing the exact config used to export.

Environment overrides
---------------------
MODEL_DIR         : directory with the checkpoint (default: <repo>/models/flowmap-deeponet_done)
MODEL_PATH        : explicit checkpoint path (default: $MODEL_DIR/best_model.pt)
PROCESSED_DIR     : processed data dir for normalization.json (default: <repo>/data/processed)
APPLY_DYNAMIC_QUANT: "1" to apply dynamic quantization of Linear layers for CPU

Outputs
-------
- <MODEL_DIR>/complete_model_exported.pt2
- <MODEL_DIR>/complete_model_exported_k1.pt2
- <MODEL_DIR>/config.snapshot.json
"""

from __future__ import annotations
import os
import sys
import json
import json as _json_std
import json5
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.export import export as texport, save as tsave, Dim

# --------------------------------------------------------------------------------------
# Paths & setup
# --------------------------------------------------------------------------------------
THIS       = Path(__file__).resolve()
PROJECT    = THIS.parent                 # .../testing
REPO_ROOT  = PROJECT.parent              # .../<repo>
SRC_DIR    = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from model import create_model  # uses cfg["data"], cfg["model"]

def _repo_rel(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()

# Default MODEL_DIR: prefer .../flowmap-deeponet_done if it exists; otherwise 14; else plain
_default_model_dir = REPO_ROOT / "models" / "flowmap-deeponet_done"
if not _default_model_dir.exists():
    alt = REPO_ROOT / "models" / "flowmap-deeponet_14"
    _default_model_dir = alt if alt.exists() else (REPO_ROOT / "models" / "flowmap-deeponet")

MODEL_DIR   = _repo_rel(os.environ.get("MODEL_DIR", _default_model_dir))
MODEL_PATH  = _repo_rel(os.environ.get("MODEL_PATH", MODEL_DIR / "best_model.pt"))
PROCESSED_DIR = _repo_rel(os.environ.get("PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
APPLY_DYNAMIC_QUANT = bool(int(os.environ.get("APPLY_DYNAMIC_QUANT", "0")))
DEVICE = torch.device("cpu")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# Example tracing sizes (keep >1 so shapes aren't constant-folded)
EX_B = 2
EX_K = 64

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
log = logging.getLogger("export")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

# --------------------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------------------
_PREFIXES = ("_orig_mod.", "module.", "model.", "_orig_mod.module.")

def _strip_prefix(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k

def _looks_like_state_dict(d) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    n, t = 0, 0
    for v in d.values():
        n += 1
        if torch.is_tensor(v) or hasattr(v, "shape"):
            t += 1
        if n >= 24:
            break
    return t >= max(1, n // 2)

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "ema_state_dict", "module", "net", "weights"):
            v = obj.get(k)
            if v is None:
                continue
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v
            if hasattr(v, "state_dict"):
                return v.state_dict()
        v = obj.get("model")
        if v is not None:
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v
            if hasattr(v, "state_dict"):
                return v.state_dict()
        if _looks_like_state_dict(obj):
            return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return None

def _remap_state_dict(model: nn.Module, raw):
    want = set(model.state_dict().keys())
    out = {}
    items = raw.items() if isinstance(raw, dict) else raw.state_dict().items()
    for k, v in items:
        k2 = _strip_prefix(k)
        if k2 in want:
            out[k2] = v
    return out

# --------------------------------------------------------------------------------------
# Inference from state_dict (shapes)
# --------------------------------------------------------------------------------------
def _sorted_layers(sd: dict, stem: str) -> List[Tuple[int, Tuple[int, int]]]:
    """Return sorted list of (idx, (out_features, in_features)) for 'stem.network.{i}.weight'."""
    layers = []
    for k, w in sd.items():
        k2 = _strip_prefix(k)
        if not (k2.startswith(f"{stem}.network.") and k2.endswith(".weight")):
            continue
        if not (hasattr(w, "shape") and w.ndim == 2):
            continue
        try:
            i = int(k2.split(".")[2])
        except Exception:
            continue
        layers.append((i, (int(w.shape[0]), int(w.shape[1]))))
    layers.sort(key=lambda x: x[0])
    return layers

def _infer_from_sd(sd: dict) -> Dict[str, Any]:
    """
    Infer as many model/data dims as possible:
    - S from out.weight rows (fallback: out.bias)
    - p from out.weight cols (fallbacks: last trunk out_features, last branch out_features)
    - trunk_layers from trunk.network.* (hidden; exclude final -> p)
    - branch_depth/branch_width from branch.network.* (hidden; exclude final -> p)
    - branch_input_dim (S+G) from branch.network.0 in_features
    """
    info: Dict[str, Any] = {}
    # Prefix-stripped view
    sd_np: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        k2 = _strip_prefix(k)
        if hasattr(v, "shape") and (v.ndim in (1, 2)):
            sd_np[k2] = v

    # S and p from out
    w_out = sd_np.get("out.weight", None)
    if w_out is not None and w_out.ndim == 2:
        info["S_from_out"] = int(w_out.shape[0])
        info["p_from_out"] = int(w_out.shape[1])
    else:
        b_out = sd_np.get("out.bias", None)
        if b_out is not None and b_out.ndim == 1:
            info["S_from_out"] = int(b_out.shape[0])

    # Trunk
    trunk_layers: List[Tuple[int, Tuple[int, int]]] = []
    for k, w in sd_np.items():
        if k.startswith("trunk.network.") and k.endswith(".weight") and w.ndim == 2:
            try:
                idx = int(k.split(".")[2])
            except Exception:
                continue
            trunk_layers.append((idx, (int(w.shape[0]), int(w.shape[1]))))
    trunk_layers.sort(key=lambda x: x[0])
    if trunk_layers:
        info["trunk_hidden"] = [o for _, (o, _) in trunk_layers[:-1]]
        info["p_from_trunk"] = int(trunk_layers[-1][1][0])

    # Branch
    branch_layers: List[Tuple[int, Tuple[int, int]]] = []
    for k, w in sd_np.items():
        if k.startswith("branch.network.") and k.endswith(".weight") and w.ndim == 2:
            try:
                idx = int(k.split(".")[2])
            except Exception:
                continue
            branch_layers.append((idx, (int(w.shape[0]), int(w.shape[1]))))
    branch_layers.sort(key=lambda x: x[0])
    if branch_layers:
        info["branch_input_dim"] = int(branch_layers[0][1][1])
        info["branch_hidden"] = [o for _, (o, _) in branch_layers[:-1]]
        info["p_from_branch"] = int(branch_layers[-1][1][0])

    # Choose p by majority vote (tie -> larger)
    p_candidates = [info.get("p_from_out"), info.get("p_from_trunk"), info.get("p_from_branch")]
    p_candidates = [int(x) for x in p_candidates if isinstance(x, int)]
    if p_candidates:
        c = Counter(p_candidates)
        info["p"] = max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # Branch width/depth
    bh = info.get("branch_hidden", [])
    if bh:
        width = Counter(bh).most_common(1)[0][0]
        info["branch_width"] = int(width)
        info["branch_depth"] = int(len(bh))

    # trunk_layers
    th = info.get("trunk_hidden", [])
    if th:
        info["trunk_layers"] = [int(x) for x in th]

    return info

# --------------------------------------------------------------------------------------
# Non-shape flags (predict_delta, activation, dropouts, trunk_dedup)
# --------------------------------------------------------------------------------------
def _load_nonshape_flags() -> Dict[str, Any]:
    """
    Recover non-shape hyperparams from files in MODEL_DIR, in priority order:
      1) config.json
      2) config.jsonc
      3) config.snapshot.json
    Defaults chosen to match common training (predict_delta=False if missing).
    """
    paths = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "config.jsonc",
        MODEL_DIR / "config.snapshot.json",
    ]
    model = {}
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json5.load(f)
            if isinstance(doc, dict):
                cand = doc.get("model") if "model" in doc else (
                    doc.get("config", {}).get("model") if isinstance(doc.get("config"), dict) else None
                )
                if isinstance(cand, dict):
                    model = cand
                    break

    return {
        "predict_delta": bool(model.get("predict_delta", False)),
        "trunk_dedup":   bool(model.get("trunk_dedup", False)),
        "activation":    str(model.get("activation", "leakyrelu")),
        "dropout":       float(model.get("dropout", 0.0)),
        "branch_dropout": float(model.get("branch_dropout", model.get("dropout", 0.0))),
        "trunk_dropout":  float(model.get("trunk_dropout",  model.get("dropout", 0.0))),
    }

# --------------------------------------------------------------------------------------
# Data configuration loading from MODEL_DIR
# --------------------------------------------------------------------------------------
def _load_data_overrides() -> Dict[str, Any]:
    """
    Recover data block (species_variables, target_species, global_variables, time_variable)
    from files in MODEL_DIR, in priority order: config.json, config.jsonc, config.snapshot.json.
    """
    paths = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "config.jsonc",
        MODEL_DIR / "config.snapshot.json",
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json5.load(f)
            data = (doc.get("data") or
                    (doc.get("config", {}) or {}).get("data"))
            if isinstance(data, dict):
                return {
                    "species_variables": list(data.get("species_variables") or []),
                    "target_species":   list(data.get("target_species")   or []),
                    "global_variables": list(data.get("global_variables") or []),
                    "time_variable":    data.get("time_variable"),
                }
    return {}

# --------------------------------------------------------------------------------------
# Data hydration (names)
# --------------------------------------------------------------------------------------
def _load_normalization(processed_dir: Path) -> Dict[str, Any] | None:
    path = processed_dir / "normalization.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json_std.load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json5.load(f)

def _synthesize_names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)]

def _build_data_cfg(inferred: Dict[str, Any], norm: Dict[str, Any] | None,
                    overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    overrides = overrides or {}
    # names from normalization.json (preferred for full species/globals)
    meta = (norm or {}).get("meta", {}) if norm else {}
    full_species_norm = list(meta.get("species_variables") or [])
    globals_norm      = list(meta.get("global_variables") or []) if meta.get("global_variables") is not None else None
    time_var_norm     = meta.get("time_variable") or "t_time"

    # overrides from MODEL_DIR configs
    full_species_cfg  = list(overrides.get("species_variables") or [])
    target_cfg        = list(overrides.get("target_species") or [])
    globals_cfg       = list(overrides.get("global_variables") or [])
    time_var_cfg      = overrides.get("time_variable") or None

    # infer dims from checkpoint
    S_out = int(inferred.get("S_from_out", 0))
    in_branch = inferred.get("branch_input_dim")  # = S_in + G
    # choose G from normalization/config if present; else try to infer minimal
    if globals_norm is not None:
        G = len(globals_norm)
    elif globals_cfg:
        G = len(globals_cfg)
    else:
        # last resort: assume no globals
        G = 0
    if isinstance(in_branch, int) and in_branch >= G:
        S_in = int(in_branch) - int(G)
    else:
        # fallbacks: if we can't infer, use normalization or overrides
        S_in = len(full_species_norm) or len(full_species_cfg) or S_out

    # choose full species list (inputs)
    full_species = full_species_cfg or full_species_norm
    if not full_species or len(full_species) != S_in:
        # synthesize or resize
        if not full_species:
            full_species = [f"species_{i}" for i in range(S_in)]
        elif len(full_species) != S_in:
            # truncate/extend to S_in to match checkpoint
            if len(full_species) > S_in:
                full_species = full_species[:S_in]
            else:
                full_species += [f"species_{i}" for i in range(len(full_species), S_in)]

    # choose target species (outputs)
    target_species = target_cfg[:] if target_cfg else []
    if not target_species:
        if S_out == S_in:
            target_species = full_species[:]  # all species
        else:
            # best-effort default when we don't have the training-time list
            # (keeps order stable; warn)
            log.warning(f"[data] target_species not found; defaulting to first {S_out} of full_species.")
            target_species = full_species[:S_out]
    else:
        if len(target_species) != S_out:
            log.warning(f"[data] target_species length {len(target_species)} != S_out {S_out}; "
                        f"truncating/expanding to match checkpoint.")
            if len(target_species) > S_out:
                target_species = target_species[:S_out]
            else:
                target_species += [f"{target_species[-1]}_pad{i}" for i in range(S_out - len(target_species))]

    # choose globals list
    globals_ = globals_cfg or globals_norm or []
    if len(globals_) != G:
        log.warning(f"[data] global_variables length {len(globals_)} != inferred G {G}; "
                    f"adjusting to {G}.")
        if len(globals_) > G:
            globals_ = globals_[:G]
        else:
            globals_ += [f"global_{i}" for i in range(len(globals_), G)]

    return {
        "species_variables": full_species,   # inputs (S_in)
        "target_species":    target_species, # outputs (S_out)
        "global_variables":  globals_,
        "time_variable":     (time_var_cfg or time_var_norm or "t_time"),
    }

# --------------------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------------------
def _maybe_dynamic_quantize(m: nn.Module) -> nn.Module:
    if not APPLY_DYNAMIC_QUANT:
        return m
    try:
        import platform
        is_arm = ("arm" in platform.machine().lower()) or ("aarch" in platform.machine().lower()) or ("apple" in platform.machine().lower())
        torch.backends.quantized.engine = "qnnpack" if is_arm else "fbgemm"
    except Exception:
        pass
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)

# --------------------------------------------------------------------------------------
# Wrapper to accept dt[K] and expand across batch WITHOUT zero-stride
# --------------------------------------------------------------------------------------
class InferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module, S: int, G: int):
        super().__init__()
        self.model = model.eval()
        self.S = int(S)
        self.G = int(G)

    @staticmethod
    def _tile_dt_for_batch(dt_norm: torch.Tensor, B: int) -> torch.Tensor:
        """
        Return dt as [B,K] with non-zero stride along batch.
        Uses repeat (contiguous), never expand (zero-stride).
        Accepts [K], [1,K], [K,1], [B,K], [B,1].
        """
        if dt_norm.ndim == 1:
            return dt_norm.unsqueeze(0).repeat(B, 1)  # [B,K]
        if dt_norm.ndim == 2:
            b, k = dt_norm.shape
            if b == 1:
                return dt_norm.repeat(B, 1)
            if b == B:
                return dt_norm
            raise RuntimeError(f"dt_norm 2D shape mismatch: got {tuple(dt_norm.shape)}, expected [1,K] or [B,K]")
        if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            return InferenceWrapper._tile_dt_for_batch(dt_norm.squeeze(-1), B)
        raise RuntimeError(f"Unsupported dt_norm shape {tuple(dt_norm.shape)}")

    def forward(self, y0_norm: torch.Tensor, globals_norm: torch.Tensor, dt_norm: torch.Tensor) -> torch.Tensor:
        if y0_norm.ndim != 2 or globals_norm.ndim != 2:
            raise RuntimeError("y0_norm and globals_norm must be [B,S] and [B,G].")
        if y0_norm.shape[0] != globals_norm.shape[0]:
            raise RuntimeError("Batch size mismatch between y0_norm and globals_norm.")
        if y0_norm.shape[1] != self.S or globals_norm.shape[1] != self.G:
            raise RuntimeError(f"Dim mismatch: expected S={self.S}, G={self.G} but got {y0_norm.shape[1]}, {globals_norm.shape[1]}")

        B = y0_norm.shape[0]
        dt_bk = self._tile_dt_for_batch(dt_norm, B)
        if dt_bk.stride(0) == 0 or not dt_bk.is_contiguous():
            dt_bk = dt_bk.contiguous()

        if dt_bk.shape[1] == 1:
            return self.model(y0_norm, dt_bk.squeeze(1).contiguous(), globals_norm)
        return self.model(y0_norm, dt_bk, globals_norm)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load checkpoint & state dict
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    if sd is None:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else "n/a"
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)} (keys={keys})")

    # 2) Infer shapes
    inferred = _infer_from_sd(sd)
    log.info(f"[infer] Derived from checkpoint: {{"
             + ", ".join(f"{k}: {inferred[k]}" for k in sorted(inferred.keys()))
             + "}}")

    # 3) Data names (normalization.json optional) with overrides
    norm = _load_normalization(PROCESSED_DIR)
    if norm is None:
        log.warning(f"[data] normalization.json not found at {PROCESSED_DIR}/normalization.json; synthesizing names.")
    data_overrides = _load_data_overrides()
    data_cfg = _build_data_cfg(inferred, norm, overrides=data_overrides)

    # 4) Non-shape flags from MODEL_DIR config(s)
    flags = _load_nonshape_flags()

    # 5) Compose config and build model
    model_cfg = {
        "p": int(inferred.get("p", 512)),
        "branch_width": int(inferred.get("branch_width", 1024)),
        "branch_depth": int(inferred.get("branch_depth", 4)),
        "trunk_layers": [int(x) for x in inferred.get("trunk_layers", [1024, 1024])],
        **flags,
    }
    cfg: Dict[str, Any] = {"data": data_cfg, "model": model_cfg}

    base = create_model(cfg).to(DEVICE).eval()
    clean_sd = _remap_state_dict(base, sd)
    missing, unexpected = base.load_state_dict(clean_sd, strict=False)
    if missing:
        log.warning(f"[load] Missing keys: {sorted(missing)[:12]}{' ...' if len(missing) > 12 else ''}")
    if unexpected:
        log.warning(f"[load] Unexpected keys: {sorted(unexpected)[:12]}{' ...' if len(unexpected) > 12 else ''}")

    # 6) Persist snapshot next to checkpoint
    snap = {
        "data": cfg["data"],
        "model": cfg["model"],
        "meta": {"source": "export.py (checkpoint-derived)", "checkpoint": str(MODEL_PATH)},
    }
    with open(MODEL_DIR / "config.snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    log.info(f"[save] Wrote snapshot -> {MODEL_DIR / 'config.snapshot.json'}")

    # 7) Optional quantization (CPU)
    model = _maybe_dynamic_quantize(base)

    # 8) Export dynamic (B,K)
    S = len(cfg["data"]["species_variables"])  # S_in
    G = len(cfg["data"]["global_variables"])
    wrapper = InferenceWrapper(model, S, G).to(DEVICE)

    ex_y0 = torch.randn(EX_B, S, dtype=torch.float32)
    ex_g  = torch.randn(EX_B, G, dtype=torch.float32) if G > 0 else torch.zeros(EX_B, 0, dtype=torch.float32)
    ex_dt = torch.linspace(0.0, 1.0, steps=EX_K, dtype=torch.float32)

    dyn = ({0: Dim("B")}, {0: Dim("B")}, {0: Dim("K")})
    prog = texport(wrapper, args=(ex_y0, ex_g, ex_dt), dynamic_shapes=dyn, strict=False)
    out_path = MODEL_DIR / ("complete_model_exported_int8.pt2" if APPLY_DYNAMIC_QUANT else "complete_model_exported.pt2")
    tsave(prog, str(out_path))
    log.info(f"[export] Dynamic B,K -> {out_path}")

    # Smoke test dynamic
    from torch.export import load as tload
    m = tload(str(out_path)).module()
    with torch.inference_mode():
        yA = m(torch.randn(1, S), torch.randn(1, G) if G>0 else torch.zeros(1,0), torch.linspace(0,1,1))
        yB = m(torch.randn(3, S), torch.randn(3, G) if G>0 else torch.zeros(3,0), torch.linspace(0,1,7))
    log.info(f"[test] Dynamic shapes OK: {tuple(yA.shape)} and {tuple(yB.shape)}")

    # 9) Export static K=1
    k1_dt = torch.tensor([0.5], dtype=torch.float32)  # any normalized dt; length-1
    dyn_k1 = ({0: Dim("B")}, {0: Dim("B")}, None)
    prog_k1 = texport(wrapper, args=(ex_y0, ex_g, k1_dt), dynamic_shapes=dyn_k1, strict=False)
    out_k1 = MODEL_DIR / ("complete_model_exported_k1_int8.pt2" if APPLY_DYNAMIC_QUANT else "complete_model_exported_k1.pt2")
    tsave(prog_k1, str(out_k1))
    log.info(f"[export] Static K=1 -> {out_k1}")

if __name__ == "__main__":
    main()