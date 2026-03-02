#!/usr/bin/env python3
"""Estimate chemistry-facing diagnostics on test split (physical-I/O model)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore[assignment]


REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.getenv("CHEMULATOR_MODEL_DIR", str(REPO / "models" / "final_version"))
).expanduser().resolve()

CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "physical_model_k1_cpu.pt2"
META_PATH = MODEL_DIR / "physical_model_metadata.json"

GROUND_TRUTH_ONLY = False
MAX_BATCHES = 0
BATCH_SIZE_OVERRIDE = 0
NUM_WORKERS_OVERRIDE = -1
FORCE_CPU = False
PRED_CHUNK = 32768
# Must match testing/export.py dynamic batch upper bound.
EXPORT_MAX_BATCH = 4096

OUTFILE = MODEL_DIR / "reports" / "chemistry_test_estimates.json"
PLOTFILE = MODEL_DIR / "plots" / "chemistry_test_estimates.png"
PLOT_MAX_POINTS = 20000

if plt is not None:
    try:
        plt.style.use("science.mplstyle")
    except OSError:
        warnings.warn("science.mplstyle not found; using matplotlib defaults.")


sys.path.insert(0, str(REPO / "src"))
from dataset import FlowMapPairsDataset  # noqa: E402
from utils import load_json_config as load_json  # noqa: E402
from utils import resolve_precision_policy, seed_everything, setup_logging  # noqa: E402


_SEED_BYTES = 4
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

_METALLICITY_ALIASES = {
    "metallicity",
    "logmetallicity",
    "log10metallicity",
    "metallicitydex",
    "mh",
    "logmh",
    "moverh",
    "zmetal",
}
_CO_ALIASES = {
    "co",
    "logco",
    "log10co",
    "coratio",
    "logcoratio",
    "cratio",
    "carbonoxygen",
    "carbonoxygenratio",
    "cto",
    "ctoratio",
    "logcto",
    "log10cto",
    "logctoratio",
    "log10coratio",
}

_METALLICITY_LOG10_NAMES = {
    "logmetallicity",
    "log10metallicity",
    "metallicitydex",
    "mh",
    "logmh",
    "moverh",
}
_CO_LOG10_NAMES = {
    "logco",
    "log10co",
    "logcoratio",
    "log10coratio",
    "logcto",
    "log10cto",
    "logctoratio",
}

# Solar photospheric abundances from Asplund et al. (2009),
# ARA&A 47, 481-522, on the standard log-epsilon scale:
#   log_epsilon(X) = log10(N_X / N_H) + 12
_SOLAR_LOG_EPSILON = {
    "H": 12.00,
    "He": 10.93,
    "C": 8.43,
    "N": 7.83,
    "O": 8.69,
    "Na": 6.24,
    "Mg": 7.60,
    "Al": 6.45,
    "Si": 7.51,
    "P": 5.41,
    "S": 7.12,
    "Cl": 5.50,
    "K": 5.03,
    "Ca": 6.34,
    "Ti": 4.95,
    "Cr": 5.64,
    "Mn": 5.43,
    "Fe": 7.50,
    "Ni": 6.22,
}


@dataclass
class RunningStats:
    total_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    sum_value: float = 0.0
    sum_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        arr = values.reshape(-1).to(torch.float64)
        self.total_count += int(arr.numel())
        finite = torch.isfinite(arr)
        n_valid = int(finite.sum().item())
        self.valid_count += n_valid
        self.invalid_count += int((~finite).sum().item())
        if n_valid == 0:
            return
        v = arr[finite]
        self.sum_value += float(v.sum().item())
        self.sum_sq += float((v * v).sum().item())
        self.min_value = min(self.min_value, float(v.min().item()))
        self.max_value = max(self.max_value, float(v.max().item()))

    def to_dict(self) -> Dict[str, Any]:
        if self.valid_count == 0:
            return {
                "total_count": self.total_count,
                "valid_count": 0,
                "invalid_count": self.invalid_count,
                "valid_fraction": 0.0 if self.total_count > 0 else None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum_value / float(self.valid_count)
        var = max(0.0, self.sum_sq / float(self.valid_count) - mean * mean)
        return {
            "total_count": self.total_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "valid_fraction": float(self.valid_count) / float(self.total_count) if self.total_count > 0 else None,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class ElementVectors:
    carbon: torch.Tensor
    oxygen: torch.Tensor
    hydrogen: torch.Tensor
    metals: torch.Tensor
    metal_elements: Tuple[str, ...]
    parsed_species: Tuple[str, ...]
    unparsed_species: Tuple[str, ...]
    species_with_carbon: Tuple[str, ...]
    species_with_oxygen: Tuple[str, ...]


def _derive_seed(base_seed: int, tag: str) -> int:
    h = hashlib.sha256(f"{int(base_seed)}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:_SEED_BYTES], byteorder="little", signed=False)


def _choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_cfg_path(path_like: str | os.PathLike[str], *, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_alias_index(names: Sequence[str], aliases: set[str]) -> Tuple[Optional[int], Optional[str]]:
    for idx, name in enumerate(names):
        if _canonical_name(name) in aliases:
            return idx, name
    return None, None


def _parse_formula(species_name: str) -> Optional[Dict[str, int]]:
    text = species_name.strip()
    if "_" in text:
        text = text.split("_", 1)[0]
    text = text.replace(" ", "").replace("+", "").replace("-", "")
    text = text.replace("(", "").replace(")", "")
    if not text:
        return None

    counts: Dict[str, int] = {}
    cursor = 0
    for m in _FORMULA_TOKEN_RE.finditer(text):
        if m.start() != cursor:
            return None
        elem = m.group(1)
        n = int(m.group(2)) if m.group(2) else 1
        if n <= 0:
            return None
        counts[elem] = counts.get(elem, 0) + n
        cursor = m.end()
    if cursor != len(text):
        return None
    return counts if counts else None


def _build_element_vectors(species_names: Sequence[str]) -> ElementVectors:
    c_vals: list[float] = []
    o_vals: list[float] = []
    h_vals: list[float] = []
    metals_vals: list[float] = []
    parsed: list[str] = []
    unparsed: list[str] = []
    with_c: list[str] = []
    with_o: list[str] = []
    metal_elements: set[str] = set()

    for s in species_names:
        counts = _parse_formula(s)
        if counts is None:
            c_vals.append(0.0)
            o_vals.append(0.0)
            h_vals.append(0.0)
            metals_vals.append(0.0)
            unparsed.append(s)
            continue
        parsed.append(s)
        c = float(counts.get("C", 0))
        o = float(counts.get("O", 0))
        h = float(counts.get("H", 0))
        metals = float(sum(v for k, v in counts.items() if k not in {"H", "He"}))
        metal_elements.update(k for k in counts if k not in {"H", "He"})
        c_vals.append(c)
        o_vals.append(o)
        h_vals.append(h)
        metals_vals.append(metals)
        if c > 0.0:
            with_c.append(s)
        if o > 0.0:
            with_o.append(s)

    return ElementVectors(
        carbon=torch.tensor(c_vals, dtype=torch.float64),
        oxygen=torch.tensor(o_vals, dtype=torch.float64),
        hydrogen=torch.tensor(h_vals, dtype=torch.float64),
        metals=torch.tensor(metals_vals, dtype=torch.float64),
        metal_elements=tuple(sorted(metal_elements)),
        parsed_species=tuple(parsed),
        unparsed_species=tuple(unparsed),
        species_with_carbon=tuple(with_c),
        species_with_oxygen=tuple(with_o),
    )


def _safe_ratio(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(numer, float("nan"), dtype=torch.float64)
    valid = torch.isfinite(numer) & torch.isfinite(denom) & (numer >= 0.0) & (denom > 0.0)
    if bool(valid.any()):
        out[valid] = numer[valid] / denom[valid]
    return out


def _species_ratios(y_phys_flat: torch.Tensor, elem: ElementVectors) -> Tuple[torch.Tensor, torch.Tensor]:
    y = y_phys_flat.to(torch.float64)
    total_c = y.matmul(elem.carbon)
    total_o = y.matmul(elem.oxygen)
    total_h = y.matmul(elem.hydrogen)
    total_metals = y.matmul(elem.metals)
    return _safe_ratio(total_c, total_o), _safe_ratio(total_metals, total_h)


def _solar_number_ratio(element: str) -> float:
    if element not in _SOLAR_LOG_EPSILON:
        raise KeyError(f"Missing solar abundance for element '{element}'")
    return float(10.0 ** (_SOLAR_LOG_EPSILON[element] - 12.0))


def _compute_solar_references(elem: ElementVectors) -> Tuple[float, float, Tuple[str, ...]]:
    solar_c_to_o = _solar_number_ratio("C") / _solar_number_ratio("O")
    if (not math.isfinite(solar_c_to_o)) or solar_c_to_o <= 0.0:
        raise RuntimeError("Invalid solar C/O reference")

    metal_elements = tuple(sorted(e for e in elem.metal_elements if e not in {"H", "He"}))
    if not metal_elements:
        raise RuntimeError("Could not derive solar metal/H proxy: no metal elements parsed from species")

    missing = [e for e in metal_elements if e not in _SOLAR_LOG_EPSILON]
    if missing:
        raise RuntimeError(
            "Missing solar abundances for parsed metal element(s): "
            + ", ".join(sorted(missing))
            + ". Extend _SOLAR_LOG_EPSILON."
        )

    solar_m_to_h_proxy = float(sum(_solar_number_ratio(e) for e in metal_elements))
    if (not math.isfinite(solar_m_to_h_proxy)) or solar_m_to_h_proxy <= 0.0:
        raise RuntimeError("Invalid solar metal/H proxy reference")

    return solar_c_to_o, solar_m_to_h_proxy, metal_elements


def _safe_divide_by_scalar(values: torch.Tensor, denom: float) -> torch.Tensor:
    out = torch.full_like(values, float("nan"), dtype=torch.float64)
    v = values.to(torch.float64)
    d = float(denom)
    if (not math.isfinite(d)) or d <= 0.0:
        return out
    valid = torch.isfinite(v)
    if bool(valid.any()):
        out[valid] = v[valid] / d
    return out


def _safe_log10(values: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(values, float("nan"), dtype=torch.float64)
    v = values.to(torch.float64)
    valid = torch.isfinite(v) & (v > 0.0)
    if bool(valid.any()):
        out[valid] = torch.log10(v[valid])
    return out


def _safe_pow10(values: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(values, float("nan"), dtype=torch.float64)
    v = values.to(torch.float64)
    valid = torch.isfinite(v)
    if bool(valid.any()):
        out[valid] = torch.pow(10.0, v[valid])
    return out


def _infer_scale(var_name: Optional[str], *, quantity: str) -> str:
    if var_name is None:
        return "unknown"
    cname = _canonical_name(var_name)
    if quantity == "metallicity":
        if cname in _METALLICITY_LOG10_NAMES or cname.startswith("log"):
            return "log10_dex_like"
        return "linear_ratio_like"
    if quantity == "c_to_o":
        if cname in _CO_LOG10_NAMES or cname.startswith("log"):
            return "log10_ratio"
        return "linear_ratio"
    return "unknown"


def _make_loader(
    *,
    dataset: FlowMapPairsDataset,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _load_physical_model(path: Path) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing physical model artifact: {path}. "
            "Run testing/export.py first."
        )
    ep = torch.export.load(path)
    # ExportedProgram-backed modules in some PyTorch versions do not support eval().
    return ep.module()


def _predict_physical(
    model: torch.nn.Module,
    y0_phys: torch.Tensor,   # [B,S]
    dt_phys: torch.Tensor,   # [B,K]
    g_phys: torch.Tensor,    # [B,G]
    chunk: int,
) -> torch.Tensor:
    B, K = dt_phys.shape
    S = y0_phys.shape[1]
    G = g_phys.shape[1]

    y_rep = y0_phys.unsqueeze(1).expand(B, K, S).reshape(-1, S).to(torch.float32)
    g_rep = g_phys.unsqueeze(1).expand(B, K, G).reshape(-1, G).to(torch.float32)
    dt_flat = dt_phys.reshape(-1, 1).to(torch.float32)

    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    effective_chunk = min(int(chunk), EXPORT_MAX_BATCH)

    outs: list[torch.Tensor] = []
    n = y_rep.shape[0]
    with torch.inference_mode():
        for start in range(0, n, effective_chunk):
            end = min(n, start + effective_chunk)
            y_chunk = y_rep[start:end]
            dt_chunk = dt_flat[start:end]
            g_chunk = g_rep[start:end]
            pred = model(y_chunk, dt_chunk, g_chunk)[:, 0, :]  # [n_chunk,S]
            outs.append(pred.detach().cpu())

    flat = torch.cat(outs, dim=0).reshape(B, K, S)
    return flat.to(torch.float64)


def _append_plot_points(
    x_store: List[float],
    y_store: List[float],
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int,
) -> None:
    if len(x_store) >= max_points:
        return
    x_flat = x.reshape(-1).to(torch.float64)
    y_flat = y.reshape(-1).to(torch.float64)
    valid = torch.isfinite(x_flat) & torch.isfinite(y_flat) & (x_flat > 0.0) & (y_flat > 0.0)
    if not bool(valid.any()):
        return

    xv = x_flat[valid].cpu()
    yv = y_flat[valid].cpu()
    remaining = max_points - len(x_store)
    if xv.numel() > remaining:
        stride = int(math.ceil(float(xv.numel()) / float(remaining)))
        xv = xv[::stride][:remaining]
        yv = yv[::stride][:remaining]

    x_store.extend(float(v) for v in xv)
    y_store.extend(float(v) for v in yv)


def _append_plot_points_finite(
    x_store: List[float],
    y_store: List[float],
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int,
) -> None:
    if len(x_store) >= max_points:
        return
    x_flat = x.reshape(-1).to(torch.float64)
    y_flat = y.reshape(-1).to(torch.float64)
    valid = torch.isfinite(x_flat) & torch.isfinite(y_flat)
    if not bool(valid.any()):
        return

    xv = x_flat[valid].cpu()
    yv = y_flat[valid].cpu()
    remaining = max_points - len(x_store)
    if xv.numel() > remaining:
        stride = int(math.ceil(float(xv.numel()) / float(remaining)))
        xv = xv[::stride][:remaining]
        yv = yv[::stride][:remaining]

    x_store.extend(float(v) for v in xv)
    y_store.extend(float(v) for v in yv)


def _plot_scatter_panel_loglog(ax: Any, x_vals: List[float], y_vals: List[float], title: str) -> None:
    if not x_vals or not y_vals:
        ax.text(0.5, 0.5, "No finite positive points", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return

    x = torch.tensor(x_vals, dtype=torch.float64).numpy()
    y = torch.tensor(y_vals, dtype=torch.float64).numpy()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    lo = max(lo * 0.8, 1e-30)
    hi = max(hi * 1.2, lo * 1.01)

    ax.loglog(x, y, ".", alpha=0.25, markersize=3)
    ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")


def _plot_scatter_panel_linear(ax: Any, x_vals: List[float], y_vals: List[float], title: str) -> None:
    if not x_vals or not y_vals:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return

    x = torch.tensor(x_vals, dtype=torch.float64).numpy()
    y = torch.tensor(y_vals, dtype=torch.float64).numpy()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if not math.isfinite(lo) or not math.isfinite(hi):
        ax.text(0.5, 0.5, "Non-finite points only", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return
    pad = max(1e-6, 0.05 * max(abs(lo), abs(hi), 1.0))
    lo -= pad
    hi += pad
    if hi <= lo:
        hi = lo + 1.0

    ax.plot(x, y, ".", alpha=0.25, markersize=3)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")


def _write_results_plot(
    *,
    out_path: Path,
    co_true: List[float],
    co_pred: List[float],
    mh_true: List[float],
    mh_pred: List[float],
) -> Optional[Path]:
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _plot_scatter_panel_loglog(axes[0], co_true, co_pred, "(C/O)/(C/O)_sun")
    _plot_scatter_panel_linear(axes[1], mh_true, mh_pred, "[M/H]_proxy (dex)")
    fig.suptitle("Chemistry Estimates on Test Split", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    setup_logging(level=logging.INFO)
    log = logging.getLogger("testing.chemistry_estimates")

    cfg = load_json(CONFIG_PATH)
    seed_everything(int(cfg["system"]["seed"]))
    device = _choose_device(FORCE_CPU)
    log.info("Using device: %s", device.type)

    policy = resolve_precision_policy(cfg, device)
    runtime_dtype = policy.dataset_dtype
    processed_dir = _resolve_cfg_path(cfg["paths"]["processed_data_dir"], base_dir=REPO)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Missing processed_data_dir: {processed_dir}")

    model_meta = load_json(META_PATH) if META_PATH.is_file() else None
    model: Optional[torch.nn.Module] = None
    if not GROUND_TRUTH_ONLY:
        model = _load_physical_model(MODEL_PATH)

    species_vars = [str(x) for x in cfg["data"]["species_variables"]]
    global_vars = [str(x) for x in cfg["data"]["global_variables"]]

    if model_meta is not None:
        meta_species = [str(x) for x in model_meta.get("species_order", [])]
        meta_globals = [str(x) for x in model_meta.get("globals_order", [])]
        if meta_species and meta_species != species_vars:
            raise ValueError("Model metadata species_order does not match config species_variables")
        if meta_globals != global_vars:
            raise ValueError("Model metadata globals_order does not match config global_variables")

    elem = _build_element_vectors(species_vars)
    if len(elem.species_with_carbon) == 0 or len(elem.species_with_oxygen) == 0:
        raise RuntimeError("Could not derive C/O: no parsed C- or O-bearing species found")
    if float(elem.hydrogen.sum().item()) <= 0.0:
        raise RuntimeError("Could not derive metal/H proxy: no parsed H-bearing species found")
    solar_c_to_o, solar_m_to_h_proxy, solar_metal_elements = _compute_solar_references(elem)
    log.info(
        "Solar references (Asplund+2009): C/O_sun=%.6g, (M/H)_sun_proxy=%.6g, elements=%s",
        solar_c_to_o,
        solar_m_to_h_proxy,
        ",".join(solar_metal_elements),
    )

    tcfg = cfg["training"]
    dcfg = cfg["dataset"]
    pairs_per_traj = int(tcfg["pairs_per_traj"])
    min_steps = int(tcfg["min_steps"])
    max_steps_raw = tcfg.get("max_steps")
    max_steps = int(max_steps_raw) if max_steps_raw is not None else None

    preload_to_gpu = bool(dcfg["preload_to_gpu"])
    base_seed = int(cfg["system"]["seed"])
    test_seed = _derive_seed(base_seed, "test")
    num_workers = int(dcfg["num_workers"]) if NUM_WORKERS_OVERRIDE < 0 else int(NUM_WORKERS_OVERRIDE)
    if preload_to_gpu and num_workers != 0:
        raise ValueError("dataset.preload_to_gpu=true requires num_workers=0")

    test_dataset = FlowMapPairsDataset(
        processed_root=processed_dir,
        split="test",
        config=cfg,
        pairs_per_traj=pairs_per_traj,
        min_steps=min_steps,
        max_steps=max_steps,
        preload_to_gpu=preload_to_gpu,
        device=device,
        dtype=runtime_dtype,
        seed=test_seed,
        logger=log.getChild("dataset.test"),
    )

    batch_size = int(dcfg["batch_size_train"]) if BATCH_SIZE_OVERRIDE <= 0 else int(BATCH_SIZE_OVERRIDE)
    loader = _make_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=bool(dcfg["persistent_workers"]),
        pin_memory=bool(dcfg["pin_memory"]),
        prefetch_factor=int(dcfg["prefetch_factor"]),
    )

    global_metal_idx, global_metal_name = _find_alias_index(global_vars, _METALLICITY_ALIASES)
    global_co_idx, global_co_name = _find_alias_index(global_vars, _CO_ALIASES)

    true_co_solar_stats = RunningStats()
    true_co_bracket_stats = RunningStats()
    true_mh_solar_stats = RunningStats()
    true_mh_bracket_stats = RunningStats()
    pred_co_solar_stats = RunningStats()
    pred_co_bracket_stats = RunningStats()
    pred_mh_solar_stats = RunningStats()
    pred_mh_bracket_stats = RunningStats()
    co_solar_abs_err_stats = RunningStats()
    co_bracket_abs_err_stats = RunningStats()
    co_solar_rel_err_stats = RunningStats()
    mh_solar_abs_err_stats = RunningStats()
    mh_bracket_abs_err_stats = RunningStats()
    mh_solar_rel_err_stats = RunningStats()
    species_log10_abs_err_stats = RunningStats()
    global_metal_stats = RunningStats() if global_metal_idx is not None else None
    global_metal_solar_stats = RunningStats() if global_metal_idx is not None else None
    global_metal_bracket_stats = RunningStats() if global_metal_idx is not None else None
    global_co_stats = RunningStats() if global_co_idx is not None else None
    global_co_solar_stats = RunningStats() if global_co_idx is not None else None
    global_co_bracket_stats = RunningStats() if global_co_idx is not None else None
    global_metal_scale = _infer_scale(global_metal_name, quantity="metallicity")
    global_co_scale = _infer_scale(global_co_name, quantity="c_to_o")

    n_batches = 0
    n_pairs = 0
    n_traj_samples = 0
    rel_eps = 1e-30
    co_true_points: List[float] = []
    co_pred_points: List[float] = []
    mh_true_points: List[float] = []
    mh_pred_points: List[float] = []

    for bidx, batch in enumerate(loader):
        if MAX_BATCHES > 0 and bidx >= MAX_BATCHES:
            break
        y_i, dt_norm, y_j, g = batch
        B, K, S = y_j.shape
        n_batches += 1
        n_pairs += int(B * K)
        n_traj_samples += int(B)

        y_true_phys = test_dataset.norm.denormalize(y_j.cpu(), species_vars).to(torch.float64)
        y_true_flat = y_true_phys.reshape(-1, S)
        true_co_raw, true_mh_raw = _species_ratios(y_true_flat, elem)
        true_co_solar = _safe_divide_by_scalar(true_co_raw, solar_c_to_o)
        true_mh_solar = _safe_divide_by_scalar(true_mh_raw, solar_m_to_h_proxy)
        true_co_bracket = _safe_log10(true_co_solar)
        true_mh_bracket = _safe_log10(true_mh_solar)
        true_co_solar_stats.update(true_co_solar)
        true_co_bracket_stats.update(true_co_bracket)
        true_mh_solar_stats.update(true_mh_solar)
        true_mh_bracket_stats.update(true_mh_bracket)

        g_phys = (
            test_dataset.norm.denormalize(g.cpu(), global_vars).to(torch.float64)
            if global_vars
            else torch.empty((B, 0), dtype=torch.float64)
        )
        if global_metal_stats is not None and global_metal_idx is not None:
            g_metal = g_phys[:, global_metal_idx]
            global_metal_stats.update(g_metal)
            if global_metal_scale == "log10_dex_like":
                global_metal_bracket_stats.update(g_metal)
                global_metal_solar_stats.update(_safe_pow10(g_metal))
            elif global_metal_scale == "linear_ratio_like":
                g_metal_solar = _safe_divide_by_scalar(g_metal, solar_m_to_h_proxy)
                global_metal_solar_stats.update(g_metal_solar)
                global_metal_bracket_stats.update(_safe_log10(g_metal_solar))
        if global_co_stats is not None and global_co_idx is not None:
            g_co = g_phys[:, global_co_idx]
            global_co_stats.update(g_co)
            if global_co_scale == "log10_ratio":
                g_co_solar = _safe_divide_by_scalar(_safe_pow10(g_co), solar_c_to_o)
                global_co_solar_stats.update(g_co_solar)
                global_co_bracket_stats.update(_safe_log10(g_co_solar))
            elif global_co_scale == "linear_ratio":
                g_co_solar = _safe_divide_by_scalar(g_co, solar_c_to_o)
                global_co_solar_stats.update(g_co_solar)
                global_co_bracket_stats.update(_safe_log10(g_co_solar))

        if model is None:
            continue

        y0_phys = test_dataset.norm.denormalize(y_i.cpu(), species_vars).to(torch.float64)  # [B,S]
        dt_phys = test_dataset.norm.denormalize_dt_to_phys(dt_norm.cpu()).to(torch.float64)  # [B,K]
        y_pred_phys = _predict_physical(model, y0_phys, dt_phys, g_phys, PRED_CHUNK)          # [B,K,S]

        y_pred_flat = y_pred_phys.reshape(-1, S)
        pred_co_raw, pred_mh_raw = _species_ratios(y_pred_flat, elem)
        pred_co_solar = _safe_divide_by_scalar(pred_co_raw, solar_c_to_o)
        pred_mh_solar = _safe_divide_by_scalar(pred_mh_raw, solar_m_to_h_proxy)
        pred_co_bracket = _safe_log10(pred_co_solar)
        pred_mh_bracket = _safe_log10(pred_mh_solar)

        pred_co_solar_stats.update(pred_co_solar)
        pred_co_bracket_stats.update(pred_co_bracket)
        pred_mh_solar_stats.update(pred_mh_solar)
        pred_mh_bracket_stats.update(pred_mh_bracket)
        _append_plot_points(co_true_points, co_pred_points, true_co_solar, pred_co_solar, PLOT_MAX_POINTS)
        _append_plot_points_finite(mh_true_points, mh_pred_points, true_mh_bracket, pred_mh_bracket, PLOT_MAX_POINTS)

        co_abs = (pred_co_solar - true_co_solar).abs()
        mh_abs = (pred_mh_solar - true_mh_solar).abs()
        co_rel = co_abs / (true_co_solar.abs() + rel_eps)
        mh_rel = mh_abs / (true_mh_solar.abs() + rel_eps)
        co_solar_abs_err_stats.update(co_abs)
        co_bracket_abs_err_stats.update((pred_co_bracket - true_co_bracket).abs())
        co_solar_rel_err_stats.update(co_rel)
        mh_solar_abs_err_stats.update(mh_abs)
        mh_bracket_abs_err_stats.update((pred_mh_bracket - true_mh_bracket).abs())
        mh_solar_rel_err_stats.update(mh_rel)

        log_abs = torch.full_like(y_true_flat, float("nan"))
        valid_log = (
            torch.isfinite(y_true_flat)
            & torch.isfinite(y_pred_flat)
            & (y_true_flat > 0.0)
            & (y_pred_flat > 0.0)
        )
        if bool(valid_log.any()):
            log_abs[valid_log] = (
                torch.log10(y_pred_flat[valid_log]) - torch.log10(y_true_flat[valid_log])
            ).abs()
        species_log10_abs_err_stats.update(log_abs)

    if n_batches == 0:
        raise RuntimeError("No test batches processed. Check test split and loader settings.")

    plot_path: Optional[Path] = None
    if model is not None:
        plot_path = _write_results_plot(
            out_path=PLOTFILE,
            co_true=co_true_points,
            co_pred=co_pred_points,
            mh_true=mh_true_points,
            mh_pred=mh_pred_points,
        )

    report: Dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "config_path": str(CONFIG_PATH),
        "physical_model_path": str(MODEL_PATH) if model is not None else None,
        "plot_path": str(plot_path) if plot_path is not None else None,
        "ground_truth_only": bool(model is None),
        "device": device.type,
        "split": "test",
        "sampling": {
            "pairs_per_traj": pairs_per_traj,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "seed_base": base_seed,
            "seed_test": test_seed,
            "batches_evaluated": n_batches,
            "trajectory_samples": n_traj_samples,
            "pair_samples": n_pairs,
            "dataset_len": len(test_dataset),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_batches_limit": MAX_BATCHES,
        },
        "units_and_scales": {
            "solar_abundance_reference": {
                "citation": "Asplund et al. 2009, ARA&A 47, 481-522",
                "log_epsilon_definition": "log_epsilon(X) = log10(N_X/N_H) + 12",
                "log_epsilon_used": {
                    "C": _SOLAR_LOG_EPSILON["C"],
                    "N": _SOLAR_LOG_EPSILON["N"],
                    "O": _SOLAR_LOG_EPSILON["O"],
                },
                "c_to_o_solar": solar_c_to_o,
                "m_to_h_proxy_solar": solar_m_to_h_proxy,
                "m_to_h_proxy_elements": list(solar_metal_elements),
            },
            "community_conventions": {
                "c_to_o": "Report (C/O)/(C/O)_sun and [C/O] = log10((C/O)/(C/O)_sun).",
                "metallicity": "Report [M/H] = log10((M/H)/(M/H)_sun); this script reports [M/H]_proxy from tracked elements.",
            },
            "reported_species_derived": {
                "c_to_o": ["over_solar_linear", "bracket_dex"],
                "m_to_h_proxy": ["over_solar_linear", "bracket_dex"],
            },
        },
        "species_formula_parsing": {
            "n_species_total": len(species_vars),
            "n_species_parsed": len(elem.parsed_species),
            "n_species_unparsed": len(elem.unparsed_species),
            "unparsed_species": list(elem.unparsed_species),
            "species_with_carbon": list(elem.species_with_carbon),
            "species_with_oxygen": list(elem.species_with_oxygen),
        },
        "ground_truth": {
            "species_derived": {
                "c_to_o_over_solar": true_co_solar_stats.to_dict(),
                "c_to_o_bracket_dex": true_co_bracket_stats.to_dict(),
                "m_to_h_proxy_over_solar": true_mh_solar_stats.to_dict(),
                "m_to_h_proxy_bracket_dex": true_mh_bracket_stats.to_dict(),
            },
            "global_inputs": {
                "detected_metallicity_name": global_metal_name,
                "detected_metallicity_scale_inferred": global_metal_scale,
                "detected_c_to_o_name": global_co_name,
                "detected_c_to_o_scale_inferred": global_co_scale,
                "metallicity_raw": None if global_metal_stats is None else global_metal_stats.to_dict(),
                "metallicity_over_solar": None if global_metal_solar_stats is None else global_metal_solar_stats.to_dict(),
                "metallicity_bracket_dex": None if global_metal_bracket_stats is None else global_metal_bracket_stats.to_dict(),
                "c_to_o_raw": None if global_co_stats is None else global_co_stats.to_dict(),
                "c_to_o_over_solar": None if global_co_solar_stats is None else global_co_solar_stats.to_dict(),
                "c_to_o_bracket_dex": None if global_co_bracket_stats is None else global_co_bracket_stats.to_dict(),
            },
        },
        "prediction": None,
    }

    if model is not None:
        report["prediction"] = {
            "species_derived": {
                "c_to_o_over_solar": pred_co_solar_stats.to_dict(),
                "c_to_o_bracket_dex": pred_co_bracket_stats.to_dict(),
                "m_to_h_proxy_over_solar": pred_mh_solar_stats.to_dict(),
                "m_to_h_proxy_bracket_dex": pred_mh_bracket_stats.to_dict(),
            },
            "errors": {
                "c_to_o_over_solar_abs": co_solar_abs_err_stats.to_dict(),
                "c_to_o_bracket_abs_dex": co_bracket_abs_err_stats.to_dict(),
                "c_to_o_over_solar_rel": co_solar_rel_err_stats.to_dict(),
                "m_to_h_proxy_over_solar_abs": mh_solar_abs_err_stats.to_dict(),
                "m_to_h_proxy_bracket_abs_dex": mh_bracket_abs_err_stats.to_dict(),
                "m_to_h_proxy_over_solar_rel": mh_solar_rel_err_stats.to_dict(),
                "species_log10_abs_error_dex": species_log10_abs_err_stats.to_dict(),
            },
        }

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote chemistry estimates report: %s", OUTFILE)
    if plot_path is not None:
        log.info("Wrote chemistry estimates plot: %s", plot_path)
    elif model is not None:
        log.warning("Skipping chemistry plot because matplotlib is unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
